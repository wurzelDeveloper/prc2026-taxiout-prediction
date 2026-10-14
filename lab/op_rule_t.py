"""Rule T of findings 10.36: does traffic after the H1 anchor help, and is the gain the traffic of the day?

    python lab/op_rule_t.py stage1      # criteria 1-3 (Jan+Jul, seed 42): the runs, the placebo, the own-slot bound; the member
    python lab/op_rule_t.py passed13    # exit 0 if criteria 1-3 passed (gates the forward check and the month pairs)
    python lab/op_rule_t.py member13    # print stage 1's member config; exit 1 if none
    python lab/op_rule_t.py stage2      # criteria 4-5 (month pairs, forward) for that member; the final decision
    python lab/op_rule_t.py member      # print the member's config once the rule passed; exit 1 otherwise
    python lab/op_rule_t.py refits      # print the member's two newest refitted runs (for lab/op_deliver.py --runs)

Every Jan+Jul comparison is paired on the same OP rows (NM-matched, target below one hour) from 9 January on
(`FIRST_DAY`: the placebo reads a week back, the configuration age 15 h more; 1-8 January would reach before the data),
against the H1 deliverable's configuration at seed 42, with the ladder report's day-block bootstrap
(`op_ladder_report.paired_step`). Each run is the newest seed-42 run of its configuration at the configuration's
current hash (`seed_run`; a refit run of that hash counts), so a refit made with ``-o`` overrides never replaces it;
the member's override-named pair and forward runs have no configuration file and match any hash. The decisions go to
runs/op_decisions.json (``T13``, ``T``) and the read-out to runs/op_rule_t.md; a re-run overwrites both.

Rule T as amended before any run (findings 10.36): F4 must gain at least 2 s with the interval below zero (c1); the
shift-7 placebo, which bounds the airport's era and the flight's identity, may keep at most a quarter of that (c2); the
departure-side traces of the own slot are removed by construction in F4-nodepthru, which reads no realised
departure-side column after the anchor: if it keeps at least three quarters of F4's gain the member is F4 (or F5 if
F5 is 0.5 s better), else F4-nodepthru if it passes c1 on its own, else no member (c3). Stage 2 checks the member on the month pairs (8 of 10 months, c4) and forward (c5).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lab"))

from prc import paths  # noqa: E402
from prc.config import load_config  # noqa: E402

BASE = "op_h1_r4_nodoy"
RUNS_T = {"F1": "op_h1_f1_plan", "F2": "op_h1_f2_rec", "F3": "op_h1_f3_nm", "F4": "op_h1_f4_adsb",
          "F4-shift7": "op_h1_f4_shift7", "F4-norwy": "op_h1_f4_norwy", "F4-nodepthru": "op_h1_f4_nodepthru",
          "F5": "op_h1_f5_full"}
FIRST_DAY = "2025-01-09"
"""Jan+Jul rows scored from this day on (every member alike): the placebo's week-earlier reads, and the 15 h the runway
configuration age looks back from them, reach the data (tests/test_operational_future.py, PLACEBO_REACH)."""
PAIRS = ("febdec", "marnov", "aproct", "maysep", "junaug")
MIN_GAIN_S = 2.0
"""Criterion 1: F4's OP MAE at least this far below H1's, with the bootstrap interval below zero."""
PLACEBO_MAX = 0.25
NODEP_MIN = 0.75
MIN_MONTHS = 8
F5_MARGIN_S = 0.5
OUT_MD = paths.RUNS / "op_rule_t.md"


def seed_run(name: str, seed: int = 42) -> Path | None:
    """The newest finished run of config ``name`` at the given seed.

    When configs/<name>.yaml exists, only runs at that configuration's current hash count: a refit trained with
    ``-o model.refit_on_all=true`` (another hash, a newer stamp) or a run of an older edit is never read. A refit run
    at the current hash counts (rule T's H1 reference is the deliverable's own refit run). A name without a
    configuration file (the member's pair and forward runs, named by overrides) matches any hash.
    """
    cfg_file = paths.CONFIGS / f"{name}.yaml"
    want = load_config(cfg_file).hash() if cfg_file.exists() else None
    hits = sorted(p for p in paths.RUNS.glob(f"*__{name}__*__s{seed}")
                  if p.name.split("__")[1] == name and (want is None or p.name.split("__")[2] == want)
                  and (p / "oof_predictions.parquet").exists() and not (p / "predict.json").exists())
    return hits[-1] if hits else None


def _context(oofs: dict):
    import op_ladder_report as rep

    ids = pd.Index(sorted(set().union(*(o.index for o in oofs.values()))), name="mvt_id")
    disrupted = next((o["disrupted"] for o in oofs.values() if "disrupted" in o.columns), None)
    return rep.holdout_context(ids, disrupted)


def _metrics(oof: pd.DataFrame, ctx: pd.DataFrame) -> dict:
    import op_ladder_report as rep

    fr = rep.op_frame(oof, ctx)
    fr = fr[fr["op"]]
    return rep.metric_row(fr["y_true"], fr["y_pred"])


def stage1() -> dict:
    import op_ladder_report as rep
    from op_rules import record

    runs = {"H1": seed_run(BASE), **{k: seed_run(v) for k, v in RUNS_T.items()}}
    missing = [k for k, v in runs.items() if v is None]
    have = {k: v for k, v in runs.items() if v is not None}
    oofs = {k: rep.read_oof(v) for k, v in have.items()}
    ctx = _context(oofs)
    keep = ctx.index[ctx["day"].astype(str) >= FIRST_DAY]
    oofs = {k: o.loc[o.index.intersection(keep)] for k, o in oofs.items()}
    ctx = ctx.loc[keep]
    rows, steps = [], {}
    for k, v in have.items():
        m = _metrics(oofs[k], ctx)
        row = {"row": k, "run": v.name, "op_mae": round(m["mae"], 2), "op_rmse": round(m["rmse"], 2),
               "within_120": round(100 * m["within_120"], 2)}
        if k != "H1" and "H1" in oofs:
            st = rep.paired_step(oofs["H1"], oofs[k], ctx)
            if st is None:
                rows.append(row)
                continue
            steps[k] = st
            row.update(d_mae=round(st["d_mae"], 2), mae_ci=f"[{st['mae_lo']:.2f}, {st['mae_hi']:.2f}]",
                       d_w120_pp=round(100 * st["d_w120"], 2))
        rows.append(row)
    dec = {"missing": missing, "rows": rows, "first_day": FIRST_DAY}
    f4, plac, nodep, f5 = steps.get("F4"), steps.get("F4-shift7"), steps.get("F4-nodepthru"), steps.get("F5")
    member = None
    if f4 is not None and plac is not None and nodep is not None:
        gain = -f4["d_mae"]
        c1 = gain >= MIN_GAIN_S and f4["mae_hi"] < 0
        c2 = -plac["d_mae"] <= PLACEBO_MAX * gain if gain > 0 else False
        nodep_gain = -nodep["d_mae"]
        nodep_c1 = nodep_gain >= MIN_GAIN_S and nodep["mae_hi"] < 0
        if c1 and c2:
            if nodep_gain >= NODEP_MIN * gain:
                f5_better = (f4["d_mae"] - f5["d_mae"]) if f5 is not None else -1.0
                member = "op_h1_f5_full" if f5_better >= F5_MARGIN_S else "op_h1_f4_adsb"
            elif nodep_c1:
                member = "op_h1_f4_nodepthru"
        dec.update(gain_s=round(gain, 3), placebo_gain_s=round(-plac["d_mae"], 3), nodepthru_gain_s=round(nodep_gain, 3),
                   c1=bool(c1), c2=bool(c2), c3_nodep_share=round(nodep_gain / gain, 3) if gain > 0 else None,
                   nodepthru_c1=bool(nodep_c1), member=member, passed=member is not None)
    else:
        dec.update(passed=False, member=None, note="F4, its placebo or F4-nodepthru is missing")
    for k in ("F4-norwy", "F5"):
        if k in steps:
            dec[f"{k}_gain_s"] = round(-steps[k]["d_mae"], 3)
    f1 = steps.get("F1")
    if f1 is not None:
        dec["f1_candidate"] = bool(-f1["d_mae"] >= 1.0 and f1["mae_hi"] < 0)
    record("T13", dec)
    return dec


def stage2() -> dict:
    import op_ladder_report as rep
    from op_rules import record

    doc = json.loads((paths.RUNS / "op_decisions.json").read_text(encoding="utf-8"))
    t13 = doc.get("T13") or {}
    member = t13.get("member")
    stem = member or "op_h1_f4_adsb"
    months, rows = {}, []
    for p in PAIRS:
        a, b = seed_run(f"{BASE}_{p}"), seed_run(f"{stem}_{p}")
        if a is None or b is None:
            rows.append({"pair": p, "missing": [x for x, r in ((BASE, a), (stem, b)) if r is None]})
            continue
        oa, ob = rep.read_oof(a), rep.read_oof(b)
        ctx = _context({"a": oa, "b": ob})
        for m in sorted(ctx["month"].dropna().unique()):
            keep = ctx.index[ctx["month"] == m]
            ma = _metrics(oa.loc[oa.index.intersection(keep)], ctx)
            mb = _metrics(ob.loc[ob.index.intersection(keep)], ctx)
            months[m] = round(mb["mae"] - ma["mae"], 3)
        rows.append({"pair": p, "h1": a.name, "member": b.name})
    fa, fb = seed_run(f"{BASE}_fwd"), seed_run(f"{stem}_fwd")
    fwd = None
    if fa is not None and fb is not None:
        oa, ob = rep.read_oof(fa), rep.read_oof(fb)
        ctx = _context({"a": oa, "b": ob})
        fwd = {"h1": round(_metrics(oa, ctx)["mae"], 3), "member": round(_metrics(ob, ctx)["mae"], 3), "h1_run": fa.name,
               "member_run": fb.name}
    better = sum(1 for v in months.values() if v < 0)
    c4 = len(months) == 10 and better >= MIN_MONTHS
    c5 = fwd is not None and fwd["member"] < fwd["h1"]
    passed = bool(t13.get("passed") and c4 and c5)
    dec = {"months_delta_mae": months, "months_better": better, "c4": bool(c4), "forward": fwd, "c5": bool(c5),
           "t13_passed": bool(t13.get("passed")), "passed": passed, "member": member if passed else None, "pairs": rows}
    record("T", dec)
    return dec


def _md(dec1: dict, dec2: dict | None) -> str:
    lines = ["# Rule T (findings 10.36): traffic after the H1 anchor", "",
             "Paired on the Jan+Jul 2025 OP rows against the H1 deliverable's configuration (seed 42); day-block "
             "bootstrap 95 % intervals.", "",
             "| row | run | OP MAE | OP RMSE | +-2 min | MAE change | 95 % CI | +-2 min change (pp) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in dec1["rows"]:
        lines.append(f"| {r['row']} | `{r['run']}` | {r['op_mae']} | {r['op_rmse']} | {r['within_120']} % | "
                     f"{r.get('d_mae', '')} | {r.get('mae_ci', '')} | {r.get('d_w120_pp', '')} |")
    lines += ["", "Stage 1 (rows from " + str(dec1.get("first_day")) + "): " + ", ".join(
        f"{k} {dec1.get(k)}" for k in ("gain_s", "placebo_gain_s", "nodepthru_gain_s", "c1", "c2", "c3_nodep_share",
                                       "nodepthru_c1", "member", "passed")), ""]
    if dec2:
        lines += ["Month pairs (the member minus H1, OP MAE per validation month): " +
                  ", ".join(f"{k} {v:+.2f}" for k, v in dec2["months_delta_mae"].items()),
                  f"Better in {dec2['months_better']} of {len(dec2['months_delta_mae'])}; forward {dec2['forward']}; "
                  f"c4 {dec2['c4']}, c5 {dec2['c5']}; rule T passed: {dec2['passed']}; member: {dec2['member']}", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("stage1", "passed13", "member13", "stage2", "member", "refits"))
    a = ap.parse_args(argv)
    if a.cmd == "stage1":
        dec = stage1()
        OUT_MD.write_text(_md(dec, None), encoding="utf-8")
        print(OUT_MD.read_text(encoding="utf-8"))
        return 0
    doc = json.loads((paths.RUNS / "op_decisions.json").read_text(encoding="utf-8"))
    if a.cmd == "passed13":
        ok = bool((doc.get("T13") or {}).get("passed"))
        print(f"rule T criteria 1-3: {'passed' if ok else 'not passed'}")
        return 0 if ok else 1
    if a.cmd == "member13":
        m = (doc.get("T13") or {}).get("member")
        if not m:
            print("rule T stage 1: no member", file=sys.stderr)
            return 1
        print(f"{m}.yaml")
        return 0
    if a.cmd == "stage2":
        dec2 = stage2()
        OUT_MD.write_text(_md(doc.get("T13") or {"rows": []}, dec2), encoding="utf-8")
        print(OUT_MD.read_text(encoding="utf-8"))
        return 0
    member = (doc.get("T") or {}).get("member")
    if not member:
        print("rule T has not passed: no member", file=sys.stderr)
        return 1
    if a.cmd == "refits":
        hits = sorted(p.name for p in paths.RUNS.glob(f"*__{member}__*__s*")
                      if p.name.split("__")[1] == member and (p / "refit.json").exists()
                      and (p / "test_predictions.parquet").exists() and not (p / "predict.json").exists())
        if len(hits) < 2:
            print(f"only {len(hits)} refitted runs of {member}", file=sys.stderr)
            return 1
        print(" ".join(hits[-2:]))
        return 0
    print(f"{member}.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
