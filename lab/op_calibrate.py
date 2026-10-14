"""A median and an 80 % interval for the operational models, calibrated on held-out months (rule C, findings 10.35).

The operational models are fitted on squared error, so they predict a conditional mean. The taxi-out distribution is
skewed to the right (a queue adds minutes, nothing takes them away), so the mean lies above the typical flight's time.
For the absolute error and the share within a tolerance, the conditional median is the better point forecast, and an
interval says how far to trust it. Both are read off the residuals ``y - mean`` of held-out predictions:
- **Where the residuals come from.** The 2025 month pairs. Each month is predicted by a model fitted on the other ten
  months. January and July are in no pair, so the Jan+Jul holdout on which rule C is checked stays untouched.
- **How they are grouped.** By airport and by 20 quantile bins of the predicted mean. The spread grows with the
  predicted time.
- **Population.** The tables are fitted on the OP population (NM-matched, target below 3,600 s), the population the
  outputs are read on.
- **Outputs.** Per row:
  - ``y_mean``, the model's prediction;
  - ``y_med = y_mean + q50(bin)``;
  - ``y_p10`` and ``y_p90``, the same with the 10th and 90th percentiles of the residual;
  - each clipped at 30 s, the template's floor.

Commands:

    python lab/op_calibrate.py fit --pairs RUN ... --out TABLES.parquet
    python lab/op_calibrate.py check --tables TABLES.parquet --runs RUN [RUN ...] --member NAME
    python lab/op_calibrate.py apply --tables TABLES.parquet --run RUN --out FILE.parquet

- ``fit`` takes run ids or config names; a name means the newest finished run of that config.
- ``check`` applies the tables to the Jan+Jul holdout of the given runs (several runs: the mean of their predictions,
  e.g. three seeds) and evaluates rule C. It writes the decision under ``C_<member>`` in runs/op_decisions.json.
- ``apply`` writes the four outputs for a run's ``test_predictions.parquet``:
  - the airport comes from the current data tree (set ``PRC_DATA_DIR`` for the final set);
  - the output is indexed by ``mvt_id``.

Known mismatch: the tables are fitted on single-seed month-pair runs, while the deliverable averages three seeds
refitted on all twelve months. A mean of seeds has slightly less variance, so its interval is slightly too wide. The
check measures the coverage of the mean that ships.
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
from op_rules import find_run, record  # noqa: E402

from prc import data, metrics  # noqa: E402

N_BINS = 20
QUANTILES = (("p10", 0.10), ("med", 0.50), ("p90", 0.90))
FLOOR_S = 30.0
ALL = "*"
"""The pooled table, used for an airport the tables do not know."""

# rule C (findings 10.35), on Jan+Jul 2025 OP rows: all four must hold
C_MAE_GAIN = 1.0           # the median's OP MAE at least 1 s below the mean's
C_WITHIN_GAIN = 0.01       # its share within +-2 min at least 1 point above
C_APT_WORSE = 2.0          # no airport's OP MAE more than 2 s worse
C_COVERAGE = (0.75, 0.85)  # the P10-P90 interval covers 75-85 % of the OP rows


# --- tables -----------------------------------------------------------------------------------------

def fit_tables(y_true, y_pred, apt, n_bins: int = N_BINS) -> pd.DataFrame:
    """Residual quantiles per airport and per quantile bin of the prediction, plus a pooled table (``apt == "*"``).

    One row per (apt, bin): the bin's upper edge ``hi`` (inf for the last bin), its support ``n`` and the residual
    quantiles ``q_p10``, ``q_med`` and ``q_p90``. The edges are the predictions' quantiles over the airport's fit rows,
    so every bin holds about the same number of rows.
    """
    frame = pd.DataFrame({"y": np.asarray(y_true, dtype="float64"), "p": np.asarray(y_pred, dtype="float64"),
                          "apt": np.asarray(apt, dtype=object)})
    frame = frame[np.isfinite(frame["y"]) & np.isfinite(frame["p"])]
    rows = []
    groups = [(a, g) for a, g in frame.groupby("apt", sort=True)] + [(ALL, frame)]
    for a, g in groups:
        p = g["p"].to_numpy()
        r = (g["y"] - g["p"]).to_numpy()
        inner = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1))[1:-1])
        b = np.searchsorted(inner, p, side="right")
        for k in range(len(inner) + 1):
            rk = r[b == k]
            if not len(rk):
                continue
            row = {"apt": a, "bin": k, "hi": float(inner[k]) if k < len(inner) else np.inf, "n": int(len(rk))}
            row.update({f"q_{name}": float(np.quantile(rk, q)) for name, q in QUANTILES})
            rows.append(row)
    return pd.DataFrame(rows)


def apply_tables(tables: pd.DataFrame, y_pred, apt) -> pd.DataFrame:
    """``y_mean``, ``y_med``, ``y_p10`` and ``y_p90`` per row (floored at 30 s); an unknown airport takes the pooled table."""
    p = np.asarray(y_pred, dtype="float64")
    apt = np.asarray(apt, dtype=object)
    shift = {name: np.full(len(p), np.nan) for name, _ in QUANTILES}
    known = set(tables["apt"])
    for a in pd.unique(apt):
        rows = np.flatnonzero(apt == a)
        t = tables[tables["apt"] == (a if a in known else ALL)].sort_values("bin")
        b = np.searchsorted(t["hi"].to_numpy()[:-1], p[rows], side="right")
        for name, _ in QUANTILES:
            shift[name][rows] = t[f"q_{name}"].to_numpy()[b]
    out = {"y_mean": p}
    for name, _ in QUANTILES:
        out[f"y_{name}"] = p + shift[name]
    frame = pd.DataFrame(out)
    return frame.clip(lower=FLOOR_S)


# --- holdout frames ---------------------------------------------------------------------------------

def holdout(runs: list[str]) -> pd.DataFrame:
    """y_true, the mean y_pred over ``runs`` (same holdout rows), apt and the OP flag."""
    paths_ = [find_run(r) for r in runs]
    missing = [r for r, p in zip(runs, paths_) if p is None]
    if missing:
        raise SystemExit(f"no finished run for {missing}")
    oofs = [read_oof(p) for p in paths_]
    idx = oofs[0].index
    for o in oofs[1:]:
        if not o.index.equals(idx):
            raise SystemExit("the runs' holdout rows differ: average only runs of one validation scheme")
    pred = np.mean([o["y_pred"].to_numpy(dtype="float64") for o in oofs], axis=0)
    mvt = data.load_movements(columns=["apt", metrics.NM_MATCH_COLUMN]).reindex(idx)
    y = oofs[0]["y_true"].to_numpy(dtype="float64")
    return pd.DataFrame({"y_true": y, "y_pred": pred, "apt": mvt["apt"].astype(str).to_numpy(),
                         "op": (y < metrics.BULK_MAX_SEC) & mvt[metrics.NM_MATCH_COLUMN].notna().to_numpy()},
                        index=idx)


def rule_c(tables: pd.DataFrame, frame: pd.DataFrame) -> dict:
    """Rule C on the OP rows of ``frame``: the median against the mean, per airport, and the interval's coverage."""
    op = frame[frame["op"]]
    out = apply_tables(tables, op["y_pred"], op["apt"])
    y = op["y_true"].to_numpy()
    m_mean, m_med = metric_row(y, out["y_mean"]), metric_row(y, out["y_med"])
    inside = (y >= out["y_p10"].to_numpy()) & (y <= out["y_p90"].to_numpy())
    apts = {}
    for a in sorted(op["apt"].unique()):
        s = (op["apt"] == a).to_numpy()
        apts[a] = {"mae_mean": round(metric_row(y[s], out["y_mean"][s])["mae"], 3),
                   "mae_med": round(metric_row(y[s], out["y_med"][s])["mae"], 3),
                   "coverage": round(float(inside[s].mean()), 4)}
        apts[a]["delta"] = round(apts[a]["mae_med"] - apts[a]["mae_mean"], 3)
    worst = max(v["delta"] for v in apts.values())
    res = {
        "n": int(len(op)),
        "mean": {k: round(m_mean[k], 4) for k in ("mae", "rmse", "within_120", "within_300", "bias")},
        "median": {k: round(m_med[k], 4) for k in ("mae", "rmse", "within_120", "within_300", "bias")},
        "mae_gain": round(m_mean["mae"] - m_med["mae"], 3),
        "within_120_gain": round(m_med["within_120"] - m_mean["within_120"], 4),
        "worst_airport_delta": worst,
        "coverage_p10_p90": round(float(inside.mean()), 4),
        "below_p10": round(float((y < out["y_p10"].to_numpy()).mean()), 4),
        "above_p90": round(float((y > out["y_p90"].to_numpy()).mean()), 4),
        "median_width_s": round(float(np.median(out["y_p90"] - out["y_p10"])), 1),
        "airports": apts,
    }
    res["passed"] = bool(res["mae_gain"] >= C_MAE_GAIN and res["within_120_gain"] >= C_WITHIN_GAIN
                         and worst <= C_APT_WORSE and C_COVERAGE[0] <= res["coverage_p10_p90"] <= C_COVERAGE[1])
    return res


# --- commands ---------------------------------------------------------------------------------------

def cmd_fit(a) -> int:
    frames = [holdout([r]) for r in a.pairs]
    months = []
    for r in a.pairs:
        months += [m for m in json.loads((find_run(r) / "config.resolved.json").read_text(encoding="utf-8"))
                   ["validation"]["holdout_months"]]
    if {"2025-01", "2025-07"} & set(months):
        raise SystemExit(f"the tables must not see January or July 2025 (rule C is checked there): {months}")
    fit = pd.concat(frames)
    fit = fit[fit["op"]]
    tables = fit_tables(fit["y_true"], fit["y_pred"], fit["apt"])
    tables.attrs = {}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tables.to_parquet(out, index=False)
    meta = {"pairs": [find_run(r).name for r in a.pairs], "months": sorted(months), "n_fit": int(len(fit)),
            "n_bins": N_BINS, "quantiles": dict(QUANTILES)}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    pooled = tables[tables["apt"] == ALL]
    print(f"wrote {out} ({len(tables)} rows; {meta['n_fit']:,} fit rows from {len(a.pairs)} runs, months {meta['months']})")
    print(f"pooled median shift by bin (s): {np.round(pooled['q_med'].to_numpy(), 1).tolist()}")
    return 0


def cmd_check(a) -> int:
    tables = pd.read_parquet(a.tables)
    res = rule_c(tables, holdout(a.runs))
    res["runs"] = [find_run(r).name for r in a.runs]
    res["tables"] = str(a.tables)
    record(f"C_{a.member}", res)
    return 0


def cmd_apply(a) -> int:
    tables = pd.read_parquet(a.tables)
    run = find_run(a.run)
    if run is None:
        raise SystemExit(f"no run {a.run}")
    pred = pd.read_parquet(run / "test_predictions.parquet")["y_pred"]
    apt = data.load_movements(columns=["apt"])["apt"].reindex(pred.index)
    if apt.isna().any():
        raise SystemExit(f"{int(apt.isna().sum())} prediction ids are not in the current data tree's movement frame")
    out = apply_tables(tables, pred.to_numpy(), apt.astype(str).to_numpy())
    out.index = pred.index
    out.insert(0, "apt", apt.astype(str).to_numpy())
    dst = Path(a.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dst)  # float64: a float32 cast moved values within 6e-5 of x.5 onto x.5 before op_deliver.py rounds
    print(f"wrote {dst}: {len(out):,} rows; medians (s) mean {out['y_mean'].median():.0f}, "
          f"median {out['y_med'].median():.0f}, p10 {out['y_p10'].median():.0f}, p90 {out['y_p90'].median():.0f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--pairs", nargs="+", required=True)
    f.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--tables", required=True)
    c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--member", required=True)
    p = sub.add_parser("apply")
    p.add_argument("--tables", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"fit": cmd_fit, "check": cmd_check, "apply": cmd_apply}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
