"""Swap longer-trained members into the submitted base and carry the new base onto the rows the ADS-B stage leaves
alone (findings 10.22, project owner's request of 6 Oct 2026).

1. New blend: the submitted base blend's runs and weights (runs/<blend11_window_lobt>/metrics.json), with every run of
   a replaced configuration swapped for its longer-trained counterpart (the configuration's total weight kept), via
   `prc blend --weights --lobt-bound`.
2. New stack: `lab/final_stack.py <new blend> --inwindow 0.5` (the exact recipe of the submitted stack, reproduced to
   the second on 6 Oct).
3. Route A: on a finished candidate run (stage output on the covered rows), the rows that no stage run corrected
   (the union of the corrected_rows.parquet of every --resid-rows / --direct-rows run) take the new stack's prediction,
   on the holdout and on 2026 alike, clipped into the NM join window. For a candidate built only from stage runs on the
   old base this equals adding (new stack - old stack), because such a candidate equals the old stack on those rows
   (checked 7 Oct: maximum difference 0.0 s); setting rather than adding stays right when a member was trained on the
   new base, whose uncovered rows already carry part of the new stack. The script prints how far the candidate was
   from the old stack there. Covered rows are untouched (refitting their stage is a separate step).

Usage: python lab/base_swap.py --swap OLDCFG=NEWRUN[,NEWRUN2] [--swap ...] --candidate RUN --resid-rows RUN [--resid-rows RUN ...]
       --direct-rows RUN --name NAME [--new-stack RUN] [--old-blend RUN --old-stack RUN]
       --new-stack reuses a stack an earlier call built (steps 1-2 skipped; the swaps are then only recorded).
       OLDCFG is a run-name fragment of the configuration to replace (e.g. lean2_window__ for exp056's two seeds).
       --old-blend / --old-stack: the base blend and its stack to swap into (default the submitted blend11_window_lobt
       and blend11_stack); on another ranking set, the runbook's blend of predict runs and its stack. A predict run
       (`prc predict`) counts as its source run's configuration (predict.json `source_run`).
Operational availability: as the base (tiers of docs/feature_table.md); the swap itself adds nothing.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prc import data, paths  # noqa: E402
from prc.postprocess import clip_to_bounds, lobt_bounds  # noqa: E402

OLD_BLEND = "2026-10-01T04-48-58__blend11_window_lobt__f13b7e97"
OLD_STACK = "2026-10-01T04-48-58__blend11_stack__b942ea3d"
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")


def args_of(flag):
    return [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == flag]


def newest(fragment: str) -> str:
    return sorted(p.name for p in paths.RUNS.glob(f"*__{fragment}__*"))[-1]


def config_of(run_id: str) -> str:
    """The configuration name of a run: its own name, or for a predict run its source run's name."""
    pj = paths.run_dir(run_id) / "predict.json"
    if pj.exists():
        run_id = json.loads(pj.read_text(encoding="utf-8"))["source_run"]
    return run_id.split("__")[1]


def main():
    global OLD_BLEND, OLD_STACK
    OLD_BLEND = (args_of("--old-blend") or [OLD_BLEND])[0]
    OLD_STACK = (args_of("--old-stack") or [OLD_STACK])[0]
    swaps = dict(s.split("=", 1) for s in args_of("--swap"))
    cand, name = args_of("--candidate")[0], args_of("--name")[0]
    row_runs = args_of("--resid-rows") + args_of("--direct-rows")
    if "--new-stack" in sys.argv:   # reuse a stack built by an earlier call (the swaps are then only recorded)
        new_stack = args_of("--new-stack")[0]
    else:
        new_stack = build_stack(swaps, name)
    apply(new_stack, swaps, cand, row_runs, name)


def build_stack(swaps: dict, name: str) -> str:
    b = json.loads((paths.run_dir(OLD_BLEND) / "metrics.json").read_text())["blend"]
    runs, ws = list(b["runs"]), list(b["weights"])
    new_runs, new_ws = [], []
    for frag, new_run in swaps.items():
        hit = [i for i, r in enumerate(runs) if config_of(r) == frag.strip("_")]
        assert hit, f"no run of configuration {frag} in the blend"
        parts = new_run.split(",")   # several retrained runs (seeds) share the configuration's weight equally
        total = sum(ws[i] for i in hit)
        new_runs += parts; new_ws += [total / len(parts)] * len(parts)
        print(f"swap {frag}: {[runs[i] for i in hit]} (weight {total:.3f}) -> {parts}")
        for i in sorted(hit, reverse=True):
            runs.pop(i); ws.pop(i)
    runs += new_runs; ws += new_ws
    tag = hashlib.sha1(("|".join(f"{r}:{w}" for r, w in zip(runs, ws))).encode()).hexdigest()[:6]
    bname, sname = f"{name}_blend_{tag}", f"{name}_stack_{tag}"
    subprocess.run([PY, "-m", "prc", "blend", "--runs", *runs, "--weights", *map(str, ws), "--lobt-bound", "--name", bname],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    subprocess.run([PY, "lab/final_stack.py", newest(bname), "--inwindow", "0.5", "--write", sname], check=True, cwd=ROOT)
    return newest(sname)


def apply(new_stack: str, swaps: dict, cand: str, row_runs: list[str], name: str):
    O, N = paths.run_dir(OLD_STACK), paths.run_dir(new_stack)
    C = paths.run_dir(cand)
    rows = pd.concat([pd.read_parquet(paths.run_dir(r) / "corrected_rows.parquet") for r in row_runs])
    mvt = data.load_movements(columns=["MVT_TIME_UTC_mvt", "LOBT_flt"])
    out = {}
    for kind, fname, col in (("oof", "oof_predictions.parquet", "y_pred"), ("rank", "test_predictions.parquet", "y_pred")):
        c = pd.read_parquet(C / fname)
        old = pd.read_parquet(O / fname)[col].reindex(c.index)
        new = pd.read_parquet(N / fname)[col].reindex(c.index)
        covered = c.index.isin(rows.index[rows["kind"] == kind])
        lo, hi = lobt_bounds(mvt, c.index)
        target = np.where(covered, c[col].to_numpy(), new.fillna(c[col]).to_numpy())
        delta = target - c[col].to_numpy()
        c2 = c.copy(); c2[col] = clip_to_bounds(target, lo, hi)
        out[kind] = c2
        off = np.abs((c[col] - old).to_numpy()[~covered])
        print(f"{kind}: {(~covered).sum():,} uncovered rows, rms change there {np.sqrt(np.mean(delta[~covered] ** 2)):.1f} s; "
              f"candidate vs old stack there: max {np.nanmax(off):.2f} s, rms {np.sqrt(np.nanmean(off ** 2)):.2f} s")
    o = out["oof"]; y, wd = o["y_true"].to_numpy(float), o["dayoffset"].to_numpy(bool)
    co = pd.read_parquet(C / "oof_predictions.parquet")["y_pred"].reindex(o.index).to_numpy(float)
    month = mvt.loc[o.index, "MVT_TIME_UTC_mvt"].dt.month.to_numpy()
    for lab, m in (("ex_wd", ~wd), ("bulk", y < 3600), ("Jan", ~wd & (month == 1)), ("Jul", ~wd & (month == 7))):
        print(f"holdout {lab:<6} candidate {np.sqrt(((co - y)[m] ** 2).mean()):.3f} -> swapped {np.sqrt(((o['y_pred'].to_numpy() - y)[m] ** 2).mean()):.3f}")
    tag2 = hashlib.sha1(f"{cand}|{new_stack}".encode()).hexdigest()[:8]
    dst = paths.RUNS / f"{cand.split('__')[0]}__{name}__{tag2}"
    dst.mkdir(exist_ok=False)
    out["oof"].to_parquet(dst / "oof_predictions.parquet")
    out["rank"].to_parquet(dst / "test_predictions.parquet")
    (dst / "metrics.json").write_text(json.dumps({"stack": {"source": cand, "steps": ["base_swap_uncovered"], "new_stack": new_stack,
                                                             "swaps": swaps, "old_blend": OLD_BLEND, "old_stack": OLD_STACK}}, indent=1))
    print("new stack", new_stack); print("wrote", dst.name)


if __name__ == "__main__":
    main()
