"""The pre-registered decision rules of the operational deliverable (findings 10.35), computed from run directories.

    python lab/op_rules.py metrics RUN_OR_NAME ...   OP metrics per run (newest run of a config name, or a run id)
    python lab/op_rules.py L                          rule L: learning rate 0.04 or 0.08 for the H1 deliverable
    python lab/op_rules.py F                          rule F: the forward tripwire (fit Jan-Oct, validate Nov+Dec)
    python lab/op_rules.py R --lead hl3|hl24          rule R: r3 (no NM history) or r4, per lead
    python lab/op_rules.py refits NAME [--n N]        the newest N refitted runs of a config (the seeds to deliver)

Every rule reads holdout predictions only (`oof_predictions.parquet`), on the OP population: NM-matched departures
with a target below 3,600 s (`lab/op_ladder_report.py`). The thresholds were fixed in findings 10.35 before the runs.
A rule writes its decision into runs/op_decisions.json (merged, one key per rule); rule L also writes the chosen
config file name to runs/op_decision_L.cfg and rule R to runs/op_decision_R_<lead>.cfg, which the queue's refit steps
read (lab/op_queue.sh). A rule whose run is missing falls back to its default and says so.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from op_ladder_report import metric_row, read_oof  # noqa: E402

from prc import data, metrics, paths  # noqa: E402

DECISIONS = paths.RUNS / "op_decisions.json"
L_CFG = paths.RUNS / "op_decision_L.cfg"

# --- the pre-registered numbers (findings 10.35; seed 42, Jan+Jul 2025, OP) -----------------------------------------
REF_NODOY = "2026-10-09T22-53-13__op_h1_r4_nodoy__c481ee66__s42"   # OP MAE 187.52 s, RMSE 279.61 s
SEED_FLOOR_MAE = 0.25          # op_h1_r4 seeds 42 and 43 (findings 10.34): a lower bound of the GPU seed noise
L_MAX_MAE = 186.77             # 187.52 - 3 x 0.25
L_MAX_RMSE = 279.61            # no worse than the reference
F_REF = {"2025-11": 173.46, "2025-12": 180.25}   # op_h1_r4_nodoy_marnov / _febdec, OP MAE of those months
F_MARGIN = 8.0
R_MARGIN = 0.5                 # r3 kept if within +0.5 s of r4's OP MAE, on Jan+Jul and on the forward check


def find_run(name_or_id: str, current: bool = False) -> Path | None:
    """A run id, or the newest finished run (metrics.json) whose directory carries the config name.

    With ``current`` the run must also carry the hash the config file has now, so that a run of an earlier state of
    the same config (the IOBT-clock long rungs of 10 Oct share their names with the schedule-clock ones) is never read.
    """
    direct = paths.RUNS / name_or_id
    if direct.is_dir():
        return direct if (direct / "metrics.json").exists() else None
    tag = f"__{name_or_id}__"
    if current:
        from prc.config import load_config

        tag += load_config(f"{name_or_id}.yaml").hash() + "__"
    hits = sorted(p for p in paths.RUNS.glob(f"*{tag}*s*")
                  if (p / "metrics.json").exists() and not (p / "predict.json").exists())
    return hits[-1] if hits else None


_CTX: pd.DataFrame | None = None


def op_metrics(run: Path) -> dict:
    """OP metrics overall and per holdout month, plus the airport with the largest MAE."""
    global _CTX
    if _CTX is None:
        _CTX = data.load_movements(columns=["apt", "MVT_TIME_UTC_mvt", metrics.NM_MATCH_COLUMN])
    oof = read_oof(run)
    ctx = _CTX.reindex(oof.index)
    y, p = oof["y_true"], oof["y_pred"]
    op = (y < metrics.BULK_MAX_SEC) & ctx[metrics.NM_MATCH_COLUMN].notna()
    month = ctx["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
    apt = ctx["apt"].astype(str)
    m = metric_row(y[op], p[op])
    out = {"run": run.name, "n": int(m["n"]), "mae": round(m["mae"], 3), "rmse": round(m["rmse"], 3),
           "within_120": round(m["within_120"], 4), "bias": round(m["bias"], 3)}
    out["months"] = {k: round(metric_row(y[op & (month == k)], p[op & (month == k)])["mae"], 3)
                     for k in sorted(month[op].unique())}
    out["airports"] = {k: round(metric_row(y[op & (apt == k)], p[op & (apt == k)])["mae"], 3)
                       for k in sorted(apt[op].unique())}
    return out


def record(key: str, value: dict) -> None:
    doc = json.loads(DECISIONS.read_text(encoding="utf-8")) if DECISIONS.exists() else {}
    doc[key] = value
    tmp = DECISIONS.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    tmp.replace(DECISIONS)
    print(json.dumps({key: value}, indent=2))


def rule_l() -> int:
    run = find_run("op_h1_r4_nodoy_lr04", current=True)
    if run is None:
        decision = {"choice": "op_h1_r4_nodoy.yaml", "reason": "no finished op_h1_r4_nodoy_lr04 run: default lr 0.08"}
    else:
        m = op_metrics(run)
        passed = m["mae"] <= L_MAX_MAE and m["rmse"] <= L_MAX_RMSE
        decision = {
            "choice": "op_h1_r4_nodoy_lr04.yaml" if passed else "op_h1_r4_nodoy.yaml",
            "candidate": m, "max_mae": L_MAX_MAE, "max_rmse": L_MAX_RMSE, "reference": REF_NODOY,
            "margin_mae": round(L_MAX_MAE - m["mae"], 3),
            "reason": ("lr 0.04 passes both thresholds" if passed else "lr 0.04 misses a threshold: keep lr 0.08")
            + (f"; within the seed floor ({SEED_FLOOR_MAE} s) of the MAE threshold, read as unresolved"
               if abs(L_MAX_MAE - m["mae"]) <= SEED_FLOOR_MAE else ""),
        }
    L_CFG.write_text(decision["choice"] + "\n", encoding="utf-8")
    record("L", decision)
    return 0


def rule_f() -> int:
    run = find_run("op_h1_r4_nodoy_fwd", current=True)
    if run is None:
        record("F", {"tripped": None, "reason": "no finished op_h1_r4_nodoy_fwd run"})
        return 0
    m = op_metrics(run)
    excess = {k: round(m["months"].get(k, np.nan) - ref, 3) for k, ref in F_REF.items()}
    tripped = any(v > F_MARGIN for v in excess.values())
    record("F", {
        "tripped": tripped, "forward": m, "reference": F_REF, "margin": F_MARGIN, "excess": excess,
        "action": ("the model card states the forward MAE as the expected out-of-time level beside the Jan+Jul figure "
                   "and lists the excess as a limit (the target-free tables are built over the whole frame)")
        if tripped else "none: the forward months stay within the month pairs' level plus the margin",
    })
    return 0


def rule_r(lead: str) -> int:
    names = {k: f"op_{lead}_{k}_rwyblind" for k in ("r3", "r4")}
    runs = {k: find_run(v, current=True) for k, v in names.items()}
    fwd = {k: find_run(v + "_fwd", current=True) for k, v in names.items()}
    cfg_file = paths.RUNS / f"op_decision_R_{lead}.cfg"
    if not all(runs.values()) or not all(fwd.values()):
        missing = [n for k, n in names.items() if runs[k] is None] + [n + "_fwd" for k, n in names.items() if fwd[k] is None]
        cfg_file.write_text(f"op_{lead}_r4_rwyblind.yaml\n", encoding="utf-8")
        record(f"R_{lead}", {"choice": f"op_{lead}_r4_rwyblind.yaml", "reason": f"runs missing ({missing}): default r4"})
        return 0
    m = {k: op_metrics(r) for k, r in runs.items()}
    f = {k: op_metrics(r) for k, r in fwd.items()}
    d_main, d_fwd = m["r3"]["mae"] - m["r4"]["mae"], f["r3"]["mae"] - f["r4"]["mae"]
    keep_r3 = d_main <= R_MARGIN and d_fwd <= R_MARGIN
    cfg_file.write_text(f"op_{lead}_{'r3' if keep_r3 else 'r4'}_rwyblind.yaml\n", encoding="utf-8")
    record(f"R_{lead}", {
        "choice": f"op_{lead}_{'r3' if keep_r3 else 'r4'}_rwyblind.yaml",
        "r3_minus_r4_mae": round(d_main, 3), "r3_minus_r4_mae_fwd": round(d_fwd, 3), "margin": R_MARGIN,
        "main": m, "forward": f,
    })
    return 0


def refits(name: str, n: int) -> list[str]:
    """The newest ``n`` runs of config ``name`` with a finished refit (refit.json and test_predictions.parquet), of the
    hash the config file has now (an earlier state of the same config, e.g. an IOBT-clock long rung, never counts)."""
    from prc.config import load_config

    tag = f"__{name}__{load_config(f'{name}.yaml').hash()}__s"
    hits = sorted(p.name for p in paths.RUNS.glob(f"*{tag}*")
                  if (p / "refit.json").exists() and (p / "test_predictions.parquet").exists()
                  and not (p / "predict.json").exists())
    if len(hits) < n:
        raise SystemExit(f"only {len(hits)} refitted runs of {name}, {n} wanted")
    return hits[-n:]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sm = sub.add_parser("metrics")
    sm.add_argument("runs", nargs="+")
    sub.add_parser("L")
    sub.add_parser("F")
    sr = sub.add_parser("R")
    sr.add_argument("--lead", choices=("hl3", "hl24"), required=True)
    sf = sub.add_parser("refits")
    sf.add_argument("name")
    sf.add_argument("--n", type=int, default=3)
    a = ap.parse_args(argv)
    if a.cmd == "refits":
        print(" ".join(refits(a.name, a.n)))
        return 0
    if a.cmd == "metrics":
        for name in a.runs:
            run = find_run(name)
            print(json.dumps(op_metrics(run) if run else {"run": name, "missing": True}))
        return 0
    if a.cmd == "L":
        return rule_l()
    if a.cmd == "F":
        return rule_f()
    return rule_r(a.lead)


if __name__ == "__main__":
    raise SystemExit(main())
