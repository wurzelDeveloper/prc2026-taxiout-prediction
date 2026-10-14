"""Regression metrics and sliced breakdowns.

RMSE is the primary metric. `bias` (mean signed error) is reported alongside it
because the strongest feature in this problem -- the NM off-block proxy -- is
systematically offset, and a per-airport bias table is the fastest way to see a
global model quietly under-serving one airport.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METRIC_NAMES = ("rmse", "mae", "medae", "r2", "bias", "p90_abs_err", "n")
LOWER_IS_BETTER = {"rmse", "mae", "medae", "p90_abs_err"}
BULK_MAX_SEC = 3600.0
NM_MATCH_COLUMN = "FLIGHT_ID_mvt"
"""A departure without an NM flight id has no NM times (no AOBT3, LOBT, EOBT): its error runs three to
five times the matched rows' at every airport (findings 9.2), so every run reports that segment."""
"""The bulk metric scores holdout rows whose target is below this: the taxi times proper,
without the few hundred multi-hour stale and wrong-date rows that decide `rmse`. Its seed
noise is about +-0.15 s against +-2 s for `rmse_ex_dayoffset` (exp030, three GPU seeds), so a
change to the bulk model reads there first."""


def nm_unmatched_rows(mvt: pd.DataFrame, idx: pd.Index) -> np.ndarray:
    """True for the rows of ``idx`` without an NM flight match (the movement frame's own column, not a feature)."""
    return mvt.loc[idx, NM_MATCH_COLUMN].isna().to_numpy(dtype=bool)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if y_true.size == 0:
        return dict.fromkeys(METRIC_NAMES, float("nan")) | {"n": 0}

    err = y_pred - y_true
    var = y_true.var()
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "medae": float(np.median(np.abs(err))),
        "r2": float(1 - np.mean(err**2) / var) if var > 0 else float("nan"),
        "bias": float(np.mean(err)),
        "p90_abs_err": float(np.quantile(np.abs(err), 0.90)),
        "n": int(y_true.size),
    }


def breakdown(y_true: pd.Series, y_pred: pd.Series, by: pd.Series, name: str) -> pd.DataFrame:
    """Metrics per level of one slicing column, as tidy rows."""
    frame = pd.DataFrame({"y": y_true, "p": y_pred, "g": by}).dropna(subset=["g"])
    rows = []
    for level, part in frame.groupby("g", observed=True):
        stats = regression_metrics(part["y"], part["p"])
        for metric, value in stats.items():
            rows.append({"breakdown": name, "level": str(level), "metric": metric, "value": value})
    return pd.DataFrame(rows)


def breakdown_table(y_true: pd.Series, y_pred: pd.Series, by: pd.Series) -> pd.DataFrame:
    """Metrics per level, one row per level (wide) -- for `metrics_by_airport.csv`."""
    frame = pd.DataFrame({"y": y_true, "p": y_pred, "g": by}).dropna(subset=["g"])
    rows = {}
    for level, part in frame.groupby("g", observed=True):
        rows[str(level)] = regression_metrics(part["y"], part["p"])
    return pd.DataFrame(rows).T.sort_index()


def summarise(metrics: dict[str, float]) -> str:
    return "  ".join(
        f"{k}={metrics[k]:,.1f}" if k != "r2" else f"{k}={metrics[k]:.4f}"
        for k in ("rmse", "mae", "medae", "bias", "r2")
        if k in metrics
    )
