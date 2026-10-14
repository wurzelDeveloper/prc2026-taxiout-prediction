"""The submitted recipe (v70) on another ranking set: the ADS-B stage, the base swap, the forced-long shift and the
Istanbul specialist, rebuilt from that set's predict runs (findings 10.32).

`lab/final_inputs.py runbook` predicts the base members on the new set and writes their blend and stack: the v32
level (260.72 on Jan+Jul 2026). This script adds what v70 adds on top (220.34), each step the lab script or prc
command of the submitted recipe with the submitted parameters:
  1. the swap members (exp065-067 and their depth-9 retrains exp073-075) and the Istanbul specialists (exp071, exp076)
     predicted under the tree's data.version (`prc predict`, as the runbook does; an existing predict run is reused);
  2. the stage's 2025 base: `lab/build_winter_base.py <stack> <v70's fifteen month-pair runs>` (their 2025 OOF; the
     new stack's ranking predictions), and the lab matrix of the tree (`lab/build_lab_matrix.py exp051_lean2.yaml
     -o data.version=V --rescale-from <the lean2 run> --out lean2__V`: the stage's NM clocks and the direct members'
     rich columns, whole-frame counts on the training-time scale);
  3. the five stage members: `lab/adsb_apply.py <base> <the member's flags> --load-models <saved run> --oof-from
     <v70's member> --lab-matrix lean2__V`: models trained and saved in advance (`--no-cv --save-models`), nothing
     fitted here (--expect-saved, on the delivered Jan+Jul set: each must reproduce its saved run's predictions);
  4. `prc blend` of the three residual members and of the two direct members, `lab/blend_direct.py --w 0.3`;
  5. `lab/base_swap.py --old-blend <runbook blend> --old-stack <runbook stack>`: each of exp065/066/067 averaged with
     its depth-9 retrain (variant A of lab/deep_candidates.py), on the rows no stage member corrects;
  6. `lab/forced_long_shift.py --min-l 1800 --k 30`;
  7. `prc blend` of the two specialists (S79) and `lab/ltfm_partner_candidate.py --w 0.3`.
Every step carries the submitted runs' holdout output (predict runs copy their sources' OOF; the stage members carry
v70's members' through --oof-from), so every parameter estimated on the holdout (the shift) is v70's.

--rehearse: on the delivered Jan+Jul set, steps 2-4 use v70's own stage candidate (v60pre: the saved models are
retrains, not v70's), and the result must equal v70 to 1e-6 s on every ranking and holdout row; otherwise it stops
and names the first step whose output differs from v70's chain.

Usage, from the checkout that holds runs/ (the script sets PRC_DATA_DIR to --tree itself):
  python lab/final_challenge.py --tree DIR --data-version V --blend RUN --stack RUN --name NAME
      (--stage-models v55g=RUN v55c=RUN v58g=RUN v55d=RUN v58d=RUN [--expect-saved] | --rehearse) [--threads 4]
      [--allow-unseen] [--allow-drift]
The ADS-B stage reads the per-day tables in cache/adsb/: for new months they must be built first (lab/adsb_new_days.sh,
the pools frozen).

Operational availability: as the submitted pipeline (tier M through the stage, C through the window and the shift).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def _arg(flag: str, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def _args_after(flag: str) -> list[str]:
    if flag not in sys.argv:
        return []
    out = []
    for a in sys.argv[sys.argv.index(flag) + 1:]:
        if a.startswith("--"):
            break
        out.append(a)
    return out


if __name__ == "__main__":   # the tree first: prc.paths reads PRC_DATA_DIR when it is imported
    if _arg("--tree") is None:
        raise SystemExit(__doc__)
    os.environ["PRC_DATA_DIR"] = str(Path(_arg("--tree")).resolve())

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from prc import paths  # noqa: E402

PY = sys.executable

# --- v70's recipe (docs/submission_manifest.md; lab/deep_candidates.py) -----------------------------------------
V70 = "2026-10-06T10-47-37__dc_A_S79_3289fc__3289fcc3"
V60PRE = "2026-10-06T10-47-37__v60pre_resid360x3_direct360x2__1c2367be"
V70_DIRECT_BLEND = "2026-10-06T16-08-35__direct_360x2__58632d82"
V70_CHAIN = {   # v70's own intermediate runs, for the rehearsal's step-by-step comparison
    "swap": "2026-10-06T10-47-37__dc_A_4af924_swap__8525af72",
    "shift": "2026-10-06T10-47-37__dc_A_4af924_shift__6fcfe7a5",
    "s79": "2026-10-09T04-43-58__dc_S79_da7281__9f8eae97",
}
STAGE = {   # key: (v70's cross-fitted member, the flags that shape its frame; PARAMS do not matter with --load-models)
    "v55g": ("2026-10-01T04-48-58__v55g_resid_360d_d10__f7e949fa", ["--regime", "--v5", "--drop-prefix", "tarr_,it_"]),
    "v55c": ("2026-10-01T04-48-58__v55c_resid_360d_d8_6000__59e75cb5", ["--regime", "--v5", "--drop-prefix", "tarr_,it_"]),
    "v58g": ("2026-10-01T04-48-58__v58g_resid_360d_d10_s142__e6924d0b", ["--regime", "--v5", "--drop-prefix", "tarr_,it_"]),
    "v55d": ("2026-10-01T04-48-58__v55d_direct_360d_d10_8000__3ca2df95", ["--direct", "--rich", "--v5", "--drop-prefix", "tarr_,it_"]),
    "v58d": ("2026-10-01T04-48-58__v58d_direct_360d_d10_8000_s142__15a9c9a9", ["--direct", "--rich", "--v5", "--drop-prefix", "tarr_,it_"]),
}
RESID, DIRECT = ("v55g", "v55c", "v58g"), ("v55d", "v58d")
RESID_ROWS, DIRECT_ROWS = "v55g", "v55d"   # base_swap's --resid-rows / --direct-rows (as lab/deep_candidates.py)
WINTER_GROUPS = [   # lab/build_winter_base.py's groups, in v70's order (runs/<base_plus_winter__80cbb34d>/winter_base.json)
    ["2026-10-01T18-30-08__lean2_window_winter__73d1a2fb__s42", "2026-10-03T13-27-39__lean2_window_lobt_anchor_winter__c72a7a85__s42",
     "2026-10-03T14-03-10__lean2_global_window_winter__eb8df5b9__s42"],
    ["2026-10-04T09-43-43__lean2_window_marnov__cd0c31a3__s42", "2026-10-04T10-19-37__lean2_window_lobt_anchor_marnov__c0bdacef__s42",
     "2026-10-04T11-01-53__lean2_global_window_marnov__a58e9139__s42"],
    ["2026-10-04T11-47-31__lean2_window_junaug__888b9252__s42", "2026-10-04T12-21-34__lean2_window_lobt_anchor_junaug__0501282a__s42",
     "2026-10-04T16-30-36__lean2_global_window_junaug__92c54422__s42"],
    ["2026-10-05T12-03-17__lean2_window_aproct__a731991d__s42", "2026-10-05T12-45-06__lean2_window_lobt_anchor_aproct__6306a2ef__s42",
     "2026-10-05T13-26-00__lean2_global_window_aproct__c28b202f__s42"],
    ["2026-10-05T14-08-01__lean2_window_maysep__0d72b2f3__s42", "2026-10-05T14-50-08__lean2_window_lobt_anchor_maysep__5aee7008__s42",
     "2026-10-05T15-32-15__lean2_global_window_maysep__35e95ab1__s42"],
]
SWAP = {   # configuration -> (longer-trained run, depth-9 retrain): variant A
    "lean2_window": ("2026-10-06T16-27-57__lean2_window_long__16c2b7dd__s42",
                     "2026-10-09T02-19-14__lean2_window_deep__bc91cf16__s42"),
    "lean2_global_window": ("2026-10-06T18-17-11__lean2_global_window_long__6d1b7355__s42",
                            "2026-10-08T21-25-31__lean2_global_window_deep__4d8bbfb6__s42"),
    "lean2_window_lobt_anchor": ("2026-10-06T20-26-54__lean2_window_lobt_anchor_long__8b4e2ed3__s42",
                                 "2026-10-09T00-00-48__lean2_window_lobt_anchor_deep__ff796544__s42"),
}
SPECIALISTS = ("2026-10-08T16-42-42__ltfm_spec_long_partner__6e4c6c09__s42",
               "2026-10-08T21-00-07__ltfm_spec_deep_partner__1ee04557__s42")
LEAN2_CONFIG, LEAN2_SOURCE = "exp051_lean2.yaml", "2026-09-29T21-06-05__lean2__c92fffbc__s42"
"""The lab matrix's config and a finished run of it (the count rescale's source frame)."""
W_DIRECT, W_SPEC = 0.3, 0.3
SHIFT = ["--min-l", "1800", "--k", "30"]


def log(msg: str) -> None:
    print(msg, flush=True)


def h6(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:6]


def newest(name: str) -> str | None:
    """The newest run whose name field is exactly ``name`` (lab scripts tag names with their inputs' hash)."""
    hits = [p for p in paths.RUNS.glob(f"*__{name}__*") if p.is_dir() and p.name.split("__")[1] == name]
    return max(hits, key=lambda p: p.stat().st_mtime).name if hits else None


def run(*args: str) -> str:
    log("$ python " + " ".join(args))
    p = subprocess.run([PY, *args], cwd=ROOT, env=os.environ.copy(), text=True, encoding="utf-8", errors="replace",
                       capture_output=True)
    tail = (p.stdout[-3000:] + p.stderr[-3000:]).strip()
    if tail:
        log("  | " + tail.replace("\n", "\n  | "))
    if p.returncode != 0:
        raise SystemExit(f"failed ({p.returncode}): python {' '.join(args)}")
    return p.stdout


def step(name: str, *args: str) -> str:
    """The run ``name`` (reused when an earlier start wrote it), made by ``args`` otherwise."""
    done = newest(name)
    if done:
        log(f"reuse {done}")
        return done
    run(*args)
    made = newest(name)
    if made is None:
        raise SystemExit(f"python {' '.join(args)} wrote no run named {name}")
    return made


def predict_run(source: str, version: str, tree: str, threads: str, extra: list[str]) -> str:
    """The predict run of ``source`` under ``version`` (made like the runbook's, reused if it exists)."""
    name = f"fc_{source.split('__')[1]}"
    out = run("lab/final_inputs.py", "latest", name, "--source", source, "--data-version", version, "--missing-ok")
    rid = out.strip().splitlines()[-1].strip() if out.strip() else ""
    if rid:
        log(f"reuse {rid}")
        return rid
    run("-m", "prc", "predict", "--run", source, "-o", f"data.version={version}", "--tree", tree, "--threads", threads,
        "--name", name, *extra)
    out = run("lab/final_inputs.py", "latest", name, "--source", source, "--data-version", version)
    return out.strip().splitlines()[-1].strip()


def blend_run(name: str, runs: list[str]) -> str:
    """`prc blend` of ``runs`` with equal weights and no window clip (as v70's resid360_x3, direct_360x2, dc_S79)."""
    return step(name, "-m", "prc", "blend", "--runs", *runs, "--name", name)


def compare(run_id: str, ref: str, files=("test_predictions.parquet", "oof_predictions.parquet")) -> float:
    """Max |difference| of two runs' predictions over ``files`` (another row set, or a NaN, counts as infinite)."""
    worst = 0.0
    for f in files:
        a = pd.read_parquet(paths.run_dir(run_id) / f)["y_pred"]
        b = pd.read_parquet(paths.run_dir(ref) / f)["y_pred"]
        if not a.index.sort_values().equals(b.index.sort_values()):
            return float("inf")
        d = (a - b.reindex(a.index)).abs()
        worst = max(worst, float(d.max()) if not d.isna().any() else float("inf"))
    return worst


def main() -> None:
    tree, version = str(Path(_arg("--tree")).resolve()), _arg("--data-version")
    blend, stack, name = _arg("--blend"), _arg("--stack"), _arg("--name")
    threads = _arg("--threads", "4")
    extra = [f for f in ("--allow-unseen", "--allow-drift") if f in sys.argv]
    rehearse = "--rehearse" in sys.argv
    saved = dict(a.split("=", 1) for a in _args_after("--stage-models"))
    if not (version and blend and stack and name):
        raise SystemExit(__doc__)
    if rehearse == bool(saved):
        raise SystemExit("give exactly one of --rehearse and --stage-models")
    if saved and set(saved) != set(STAGE):
        raise SystemExit(f"--stage-models needs {sorted(STAGE)}, got {sorted(saved)}")
    if Path(paths.DATA).resolve() != Path(tree):
        raise SystemExit(f"PRC_DATA_DIR did not reach prc.paths ({paths.DATA} != {tree})")
    log(f"tree {tree}, data.version {version}; base blend {blend}, stack {stack}; "
        f"{'rehearsal (v70 own stage)' if rehearse else 'stage from saved models'}")

    # 1. the swap members and the specialists, predicted on the new set
    pred = {src: predict_run(src, version, tree, threads, extra) for pair in SWAP.values() for src in pair}
    pred.update({src: predict_run(src, version, tree, threads, extra) for src in SPECIALISTS})

    # 2-4. the stage candidate (v60pre)
    if rehearse:
        cand, rows = V60PRE, {k: STAGE[k][0] for k in (RESID_ROWS, DIRECT_ROWS)}
        log(f"rehearsal: stage candidate = v70's {cand}")
    else:
        # build_winter_base.py names its run by the hash of its inputs (the name field is always base_plus_winter)
        tag = hashlib.sha1(("|".join([stack] + [w for g in WINTER_GROUPS for w in g])).encode()).hexdigest()[:8]
        base = f"{stack.split('__')[0]}__base_plus_winter__{tag}"
        if (paths.RUNS / base / "winter_base.json").exists():
            log(f"reuse {base}")
        else:
            run("lab/build_winter_base.py", stack, *[",".join(g) for g in WINTER_GROUPS])
            if not (paths.RUNS / base / "winter_base.json").exists():
                raise SystemExit(f"build_winter_base.py did not write {base}")
        lab = f"lean2__{version}"
        if (paths.CACHE / "lab" / lab / "X.parquet").exists():
            log(f"reuse the lab matrix {lab}")
        else:
            run("lab/build_lab_matrix.py", LEAN2_CONFIG, "-o", f"data.version={version}", "--rescale-from", LEAN2_SOURCE,
                "--out", lab)
        members = {}
        for key, (v70_member, flags) in STAGE.items():
            mname = f"{name}_{key}_{h6(base, saved[key], lab)}"
            members[key] = step(mname, "lab/adsb_apply.py", base, *flags, "--load-models", saved[key],
                                "--oof-from", v70_member, "--lab-matrix", lab, "--write", mname)
            if "--expect-saved" in sys.argv:   # the delivered set: the loaded models reproduce their saved run
                d = compare(members[key], saved[key], files=("test_predictions.parquet",))
                log(f"stage {key}: max |diff| to the saved run {saved[key]}: {d:.6f} s")
                if not d <= 1e-6:
                    raise SystemExit(f"STAGE REHEARSAL FAILED: {key} does not reproduce {saved[key]}")
        resid = blend_run(f"{name}_resid3_{h6(*[members[k] for k in RESID])}", [members[k] for k in RESID])
        direct = blend_run(f"{name}_direct2_{h6(*[members[k] for k in DIRECT])}", [members[k] for k in DIRECT])
        # the direct blend's rows for blend_direct.py: v70's holdout rows (carried), the new member's ranking rows
        cr = paths.run_dir(direct) / "corrected_rows.parquet"
        if not cr.exists():
            held = pd.read_parquet(paths.run_dir(V70_DIRECT_BLEND) / "corrected_rows.parquet")
            new = pd.read_parquet(paths.run_dir(members["v55d"]) / "corrected_rows.parquet")
            pd.concat([held[held["kind"].eq("oof")], new[new["kind"].eq("rank")]]).to_parquet(cr)
        cname = f"{name}_v60pre_{h6(resid, direct, W_DIRECT)}"
        cand = step(cname, "lab/blend_direct.py", resid, direct, "--write", cname, "--w", str(W_DIRECT))
        rows = {k: members[k] for k in (RESID_ROWS, DIRECT_ROWS)}
        d = compare(cand, V60PRE, files=("oof_predictions.parquet",))
        log(f"stage candidate {cand}: holdout max |diff| to v70's v60pre {d:.6f} s (carried, must be 0)")
        if not d <= 1e-6:
            raise SystemExit("the stage candidate's holdout output is not v70's: stop (check --oof-from and the blends)")

    # 5. the base swap, variant A: each configuration's weight split between its long run and its depth-9 retrain
    swaps = {cfg: f"{pred[long_run]},{pred[deep_run]}" for cfg, (long_run, deep_run) in SWAP.items()}
    sname = f"{name}_swap_{h6(cand, blend, stack, sorted(swaps.items()))}"
    swapped = step(sname, "lab/base_swap.py", *[a for c, r in swaps.items() for a in ("--swap", f"{c}={r}")],
                   "--candidate", cand, "--resid-rows", rows[RESID_ROWS], "--direct-rows", rows[DIRECT_ROWS],
                   "--name", sname, "--old-blend", blend, "--old-stack", stack)
    # 6. the forced-long shift
    fname = f"{name}_shift_{h6(swapped, *SHIFT)}"
    shifted = step(fname, "lab/forced_long_shift.py", swapped, *SHIFT, "--write", fname)
    # 7. the Istanbul specialist S79 at 0.3
    s79 = blend_run(f"{name}_s79_{h6(*[pred[s] for s in SPECIALISTS])}", [pred[s] for s in SPECIALISTS])
    final_name = f"{name}_{h6(shifted, s79, W_SPEC)}"
    final = step(final_name, "lab/ltfm_partner_candidate.py", shifted, s79, "--write", final_name, "--w", str(W_SPEC))

    out = {"tree": tree, "data_version": version, "blend": blend, "stack": stack, "predict_runs": pred,
           "stage": "v70 own (rehearsal)" if rehearse else saved, "candidate": cand, "swap": swapped,
           "shift": shifted, "s79": s79, "final": final}
    if rehearse:
        diffs = {"swap": compare(swapped, V70_CHAIN["swap"]), "shift": compare(shifted, V70_CHAIN["shift"]),
                 "s79": compare(s79, V70_CHAIN["s79"]), "final": compare(final, V70)}
        out["rehearsal_max_abs_diff_s"] = diffs
        for k, v in diffs.items():
            log(f"rehearsal {k:<6} max |diff| to v70's chain: {v:.6f} s")
        bad = [k for k, v in diffs.items() if not v <= 1e-6]
        (paths.run_dir(final) / "final_challenge.json").write_text(json.dumps(out, indent=1))
        if bad:
            raise SystemExit(f"REHEARSAL FAILED: {bad[0]} differs from v70's chain")
        log("REHEARSAL PASSED: the recipe reproduces v70 from the predict runs")
    else:
        (paths.run_dir(final) / "final_challenge.json").write_text(json.dumps(out, indent=1))
        t = pd.read_parquet(paths.run_dir(final) / "test_predictions.parquet")["y_pred"]
        log(f"final {final}: {len(t):,} ranking rows, NaN {int(t.isna().sum())}, mean {t.mean():.1f} s, "
            f"median {t.median():.1f} s; next: python -m prc submit --run {final} (with PRC_DATA_DIR={tree})")


if __name__ == "__main__":
    main()
