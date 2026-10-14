"""Candidates from the depth-9 retrain of the best configuration (project owner's request, 8 Oct 2026; findings 10.30).

v69 is built in three steps on v60pre, the stage candidate:
1. lab/base_swap.py swaps the three longer-trained members (exp065/066/067) into the base, on the rows no ADS-B stage
   corrects.
2. lab/forced_long_shift.py applies the forced-long shift; the result is v62.
3. lab/ltfm_partner_candidate.py blends Istanbul's rows with the partner-runway specialist exp071 at 0.3.

This script rebuilds that recipe with the depth-9 runs:

  base variants  B0 = v62's base (the long members, v62's own stack)
                 R  = the deep members replace the long ones (exp073/074/075 for exp065/066/067)
                 A  = each configuration's weight split equally between its long and its deep run
  specialists    S7 = exp071 (v69's), S9 = exp076 (depth 9), S79 = their mean (prc blend)

B0 with S7 is the control: it must reproduce v69's predictions, otherwise the script stops. A deep run that is
missing (not finished, failed) drops out of the swap, and its configuration keeps its long run. The script prints the
holdout of every candidate against v69 (without wrong-date rows: all, January, July; bulk; LTFM), the 2026 change,
and the pre-registered choice (findings 10.30):
- the lowest whole-holdout RMSE without wrong-date rows among candidates whose January and July are no worse than
  v69's;
- that candidate passes if it is at least 0.02 s better than v69.
Everything is written to cache/lab/deep_candidates.json. Nothing is uploaded.

Usage: python lab/deep_candidates.py [--control-only]
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prc import data, paths  # noqa: E402

PY = sys.executable
V60PRE = "2026-10-06T10-47-37__v60pre_resid360x3_direct360x2__1c2367be"
RESID_ROWS = "2026-10-01T04-48-58__v55g_resid_360d_d10__f7e949fa"
DIRECT_ROWS = "2026-10-01T04-48-58__v55d_direct_360d_d10_8000__3ca2df95"
V62_STACK = "2026-10-07T00-19-23__v62pre_swap3_stack_4aa895__6f97365e"
V69 = "2026-10-06T10-47-37__v69_ltfm_partner__6e1bc8f1"
LONG = {
    "lean2_window": "2026-10-06T16-27-57__lean2_window_long__16c2b7dd__s42",
    "lean2_global_window": "2026-10-06T18-17-11__lean2_global_window_long__6d1b7355__s42",
    "lean2_window_lobt_anchor": "2026-10-06T20-26-54__lean2_window_lobt_anchor_long__8b4e2ed3__s42",
}
DEEP_NAME = {
    "lean2_window": "lean2_window_deep",
    "lean2_global_window": "lean2_global_window_deep",
    "lean2_window_lobt_anchor": "lean2_window_lobt_anchor_deep",
}
S7 = "2026-10-08T16-42-42__ltfm_spec_long_partner__6e4c6c09__s42"
S9_NAME = "ltfm_spec_deep_partner"
W_SPEC = 0.3
MIN_GAIN = 0.02
LOG = paths.CACHE / "lab" / "deep_candidates.log"


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def run(*args: str) -> str:
    log("$ python " + " ".join(args))
    p = subprocess.run([PY, *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    with LOG.open("a", encoding="utf-8") as f:
        f.write(p.stdout[-4000:] + p.stderr[-4000:] + "\n")
    if p.returncode != 0:
        raise RuntimeError(f"failed ({p.returncode}): {' '.join(args)}\n{p.stderr[-2000:]}")
    return p.stdout


def newest(fragment: str) -> str:
    hits = [p for p in paths.RUNS.glob(f"*__{fragment}__*") if p.is_dir()]
    if not hits:
        raise FileNotFoundError(fragment)
    return max(hits, key=lambda p: p.stat().st_mtime).name


def refit_run(name: str) -> str | None:
    """The newest run of config name `name` that has test predictions (a refitted run), else None."""
    hits = [p for p in paths.RUNS.glob(f"*__{name}__*__s*") if (p / "test_predictions.parquet").exists()
            and p.name.split("__")[1] == name]
    return max(hits, key=lambda p: p.stat().st_mtime).name if hits else None


def h6(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:6]


def existing(fragment: str) -> str | None:
    """A run an earlier call of this script already wrote under this (input-hashed) name, reused on a re-run."""
    try:
        name = newest(fragment)
    except FileNotFoundError:
        return None
    log(f"reuse {name}")
    return name


def swap_args(swaps: dict[str, str]) -> list[str]:
    return [a for frag, r in swaps.items() for a in ("--swap", f"{frag}={r}")]


def build(base: str, swaps: dict[str, str], tag: str, new_stack: str | None = None) -> str:
    """base_swap on v60pre (uncovered rows) -> forced-long shift; returns the shifted run (the base variant)."""
    tag = f"{tag}_{h6(sorted(swaps.items()), new_stack)}"
    extra = ["--new-stack", new_stack] if new_stack else []
    swapped = existing(f"{tag}_swap")
    if swapped is None:
        run("lab/base_swap.py", *swap_args(swaps), "--candidate", V60PRE, "--resid-rows", RESID_ROWS,
            "--direct-rows", DIRECT_ROWS, "--name", f"{tag}_swap", *extra)
        swapped = newest(f"{tag}_swap")
    shifted = existing(f"{tag}_shift")
    if shifted is None:
        run("lab/forced_long_shift.py", swapped, "--min-l", "1800", "--k", "30", "--write", f"{tag}_shift")
        shifted = newest(f"{tag}_shift")
    return shifted


def with_specialist(base_run: str, spec: str, tag: str) -> str:
    tag = f"{tag}_{h6(base_run, spec, W_SPEC)}"
    done = existing(tag)
    if done:
        return done
    run("lab/ltfm_partner_candidate.py", base_run, spec, "--write", tag, "--w", str(W_SPEC))
    return newest(tag)


def scores(run_id: str, mvt: pd.DataFrame) -> dict:
    o = pd.read_parquet(paths.run_dir(run_id) / "oof_predictions.parquet")
    y, p = o["y_true"].to_numpy(float), o["y_pred"].to_numpy(float)
    wd = o["dayoffset"].astype(bool).to_numpy()
    month = mvt.loc[o.index, "MVT_TIME_UTC_mvt"].dt.month.to_numpy()
    ltfm = (mvt.loc[o.index, "apt"] == "LTFM").to_numpy()
    e2 = (p - y) ** 2

    def r(m):
        return float(np.sqrt(e2[m].mean()))

    return {"ex_wd": r(~wd), "jan": r(~wd & (month == 1)), "jul": r(~wd & (month == 7)), "bulk": r(y < 3600),
            "ltfm": r(~wd & ltfm)}


def change_2026(run_id: str, ref: str) -> dict:
    a = pd.read_parquet(paths.run_dir(run_id) / "test_predictions.parquet")["y_pred"]
    b = pd.read_parquet(paths.run_dir(ref) / "test_predictions.parquet")["y_pred"].reindex(a.index)
    d = (a - b).to_numpy(float)
    return {"rows_changed": int((np.abs(d) > 1e-9).sum()), "rms_change_s": float(np.sqrt((d ** 2).mean())),
            "max_abs_change_s": float(np.abs(d).max()), "nan": int(a.isna().sum())}


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    mvt = data.load_movements(columns=["apt", "MVT_TIME_UTC_mvt"])
    out: dict = {"v69": V69, "v69_holdout": scores(V69, mvt), "candidates": {}}

    # control: v69 rebuilt from v62's stack with the exp071 specialist
    b0 = build(V60PRE, LONG, "dc_B0", new_stack=V62_STACK)
    ctrl = with_specialist(b0, S7, "dc_B0_S7")
    ch = change_2026(ctrl, V69)
    so, sv = scores(ctrl, mvt), out["v69_holdout"]
    log(f"control (B0 + S7) against v69: 2026 max |diff| {ch['max_abs_change_s']:.6f} s; holdout ex_wd {so['ex_wd']:.4f} vs {sv['ex_wd']:.4f}")
    if ch["max_abs_change_s"] > 1e-3 or abs(so["ex_wd"] - sv["ex_wd"]) > 1e-3:
        raise SystemExit("control does not reproduce v69: stop")
    out["control"] = {"run": ctrl, **ch}
    if "--control-only" in sys.argv:
        (paths.CACHE / "lab" / "deep_candidates.json").write_text(json.dumps(out, indent=1))
        return

    deep = {frag: refit_run(name) for frag, name in DEEP_NAME.items()}
    s9 = refit_run(S9_NAME)
    out["deep_runs"], out["s9"] = deep, s9
    log(f"deep members found: {deep}; Istanbul depth-9 specialist: {s9}")
    bases = {"B0": b0}
    have = {frag: r for frag, r in deep.items() if r}
    if have:
        bases["R"] = build(V60PRE, {**LONG, **have}, "dc_R")
        bases["A"] = build(V60PRE, {**LONG, **{f: f"{LONG[f]},{r}" for f, r in have.items()}}, "dc_A")
    specs = {"S7": S7}
    if s9:
        specs["S9"] = s9
        s79 = f"dc_S79_{h6(S7, s9)}"
        if existing(s79) is None:
            run("-m", "prc", "blend", "--runs", S7, s9, "--name", s79)
        specs["S79"] = newest(s79)
    for bk, brun in bases.items():
        for sk, srun in specs.items():
            key = f"{bk}+{sk}"
            cand = ctrl if key == "B0+S7" else with_specialist(brun, srun, f"dc_{bk}_{sk}")
            out["candidates"][key] = {"run": cand, **scores(cand, mvt), **change_2026(cand, V69)}

    v = out["v69_holdout"]
    log(f"\nv69 holdout: ex_wd {v['ex_wd']:.3f}  Jan {v['jan']:.3f}  Jul {v['jul']:.3f}  bulk {v['bulk']:.3f}  LTFM {v['ltfm']:.3f}")
    log(f"{'candidate':<10} {'ex_wd':>9} {'Jan':>9} {'Jul':>9} {'bulk':>9} {'LTFM':>9} {'2026 rows':>10} {'rms s':>7}")
    for k, c in out["candidates"].items():
        log(f"{k:<10} {c['ex_wd'] - v['ex_wd']:+9.3f} {c['jan'] - v['jan']:+9.3f} {c['jul'] - v['jul']:+9.3f} "
            f"{c['bulk'] - v['bulk']:+9.3f} {c['ltfm'] - v['ltfm']:+9.3f} {c['rows_changed']:>10,} {c['rms_change_s']:7.2f}")
    ok = {k: c for k, c in out["candidates"].items()
          if k != "B0+S7" and c["jan"] <= v["jan"] + 1e-9 and c["jul"] <= v["jul"] + 1e-9}
    if ok:
        best = min(ok, key=lambda k: ok[k]["ex_wd"])
        gain = v["ex_wd"] - ok[best]["ex_wd"]
        out["choice"] = {"candidate": best, "run": ok[best]["run"], "ex_wd_gain": gain, "passes": bool(gain >= MIN_GAIN)}
        log(f"\npre-registered choice: {best} ({ok[best]['run']}), ex_wd gain {gain:+.3f} s -> "
            f"{'PASSES' if gain >= MIN_GAIN else 'below the 0.02 s bar: no upload'}")
    else:
        out["choice"] = None
        log("\nno candidate keeps both months at or below v69: no upload")
    (paths.CACHE / "lab" / "deep_candidates.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
