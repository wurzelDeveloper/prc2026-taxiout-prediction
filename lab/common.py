"""Shared loaders for the lab experiments (fast, outside `prc train`)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prc import data, paths  # noqa: E402

LAB = paths.CACHE / "lab"
DAY = 86_400.0


def load_lab(name: str = "lean2", refit: bool = False):
    """X (every departure), meta, movements, target. With ``refit`` the fold-dependent columns are the refit's."""
    d = LAB / name
    X = pd.read_parquet(d / "X.parquet")
    meta = json.loads((d / "meta.json").read_text())
    if refit:
        R = pd.read_parquet(d / "X_refit_dep.parquet")
        X[R.columns] = R.reindex(X.index)
    mvt = data.load_movements()
    y = data.load_target()
    return X, meta, mvt, y


def split_index(mvt: pd.DataFrame, y: pd.Series, holdout=("2025-01", "2025-07")):
    """fit / val (seasonal holdout) / rank departure indices."""
    tr = y.index
    month = mvt.loc[tr, "MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
    is_val = month.isin(set(holdout)).to_numpy()
    rank = mvt.index[(mvt["split"] == "rank").to_numpy() & mvt["is_dep"].to_numpy()]
    return tr[~is_val], tr[is_val], rank


def mixture_labels(X: pd.DataFrame, y: np.ndarray, min_base=1800.0, tol=60.0):
    """The lean line's regime labels: -1 outside the universe, 0 taxi, 1 stale (target = schedule delta)."""
    base = pd.to_numeric(X["sched_delta_full"], errors="coerce").to_numpy("float64")
    marked = pd.to_numeric(X["stale_suspect"], errors="coerce").fillna(0).to_numpy() > 0
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(base) & (base >= 0) & (marked | (base >= min_base))
        lab = np.where(np.abs(y - base) <= tol, 1, 0)
    return np.where(valid, lab, -1), base, valid


def dayoffset_mask(y: np.ndarray, tol=3600.0) -> np.ndarray:
    days = np.round(y / DAY)
    return (days >= 1) & (np.abs(y - days * DAY) <= tol)


def report(name: str, y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, "float64")
    p = np.asarray(p, "float64")
    e2 = (p - y) ** 2
    wd = dayoffset_mask(y)
    bulk = y < 3600
    out = {
        "name": name,
        "rmse": float(np.sqrt(e2.mean())),
        "ex_dayoffset": float(np.sqrt(e2[~wd].mean())),
        "bulk": float(np.sqrt(e2[bulk].mean())),
        "n": int(len(y)),
    }
    print(f"{name:<40} rmse {out['rmse']:8.2f}  ex_wd {out['ex_dayoffset']:8.2f}  bulk {out['bulk']:8.2f}", flush=True)
    return out
