"""Review item 4 (findings 10.20): a shrunk shift for rows whose NM join window forces a long taxi.

On an NM-matched departure the target lies in [L, L + 7,212] with L = takeoff - LOBT - 3,606 s (findings 10.3). Where
L exceeds 1,800 s the window itself forces a taxi above 30 minutes; the reviewers found the final model 211 s short of
the truth on those rows (truth 1,694 s above the edge, prediction 1,484 s). This script estimates, by day-level
cross-validation on the Jan+Jul 2025 holdout, a shrunk mean residual per band of (prediction - L) and reports what
adding it would do; with --write it writes the shifted run (holdout rows shifted out of fold, 2026 rows with the shift
fitted on all holdout days), clipped back into each row's window.

Operational availability: tier C (the join window is a challenge artefact).

Usage: python lab/forced_long_shift.py <run> [--min-l 1800] [--k 30] [--pmax 1.0] [--write NAME]
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prc import data, paths  # noqa: E402

BANDS = [0, 300, 900, 1800, np.inf]
HALF = 3606.0


def arg(flag, default, cast=float):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def lower_bound(mvt: pd.DataFrame, idx: pd.Index) -> pd.Series:
    m = mvt.loc[idx]
    return (m["MVT_TIME_UTC_mvt"] - m["LOBT_flt"]).dt.total_seconds() - HALF


def band_of(gap: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(BANDS, gap, side="right") - 1, 0, len(BANDS) - 2)


def fit_shift(res: np.ndarray, band: np.ndarray, k: float) -> np.ndarray:
    out = np.zeros(len(BANDS) - 1)
    for b in range(len(out)):
        r = res[band == b]
        out[b] = r.sum() / (len(r) + k) if len(r) else 0.0
    return out


def main():
    run = sys.argv[1]
    min_l, k, pmax = arg("--min-l", 1800.0), arg("--k", 30.0), arg("--pmax", 1.0)
    R = paths.run_dir(run)
    oof = pd.read_parquet(R / "oof_predictions.parquet")
    test = pd.read_parquet(R / "test_predictions.parquet")
    mvt = data.load_movements(columns=["MVT_TIME_UTC_mvt", "LOBT_flt"])
    y, p, wd = oof["y_true"].to_numpy(float), oof["y_pred"].to_numpy(float), oof["dayoffset"].to_numpy(bool)
    ps = oof["p_stale"].fillna(0).to_numpy(float) if "p_stale" in oof else np.zeros(len(oof))
    L = lower_bound(mvt, oof.index).to_numpy(float)
    day = mvt.loc[oof.index, "MVT_TIME_UTC_mvt"].dt.floor("D")
    month = mvt.loc[oof.index, "MVT_TIME_UTC_mvt"].dt.month.to_numpy()
    days = sorted(day.unique())
    fold = day.map({d: i % 5 for i, d in enumerate(days)}).to_numpy()
    sel = np.isfinite(L) & (L > min_l) & ~wd & (ps <= pmax)
    gap = p - L
    band = band_of(gap)
    res = y - p
    print(f"forced-long holdout rows (L > {min_l:.0f} s, matched, not wrong-date, p_stale <= {pmax}): {sel.sum():,}")
    print(f"  mean truth - L {np.mean((y - L)[sel]):.0f} s, mean prediction - L {np.mean(gap[sel]):.0f} s, "
          f"mean residual {res[sel].mean():+.0f} s, RMSE {np.sqrt((res[sel] ** 2).mean()):.0f}")
    for b in range(len(BANDS) - 1):
        m = sel & (band == b)
        if m.any():
            print(f"  band {BANDS[b]:>5.0f}-{BANDS[b + 1]:<5} n {m.sum():>4}  mean residual {res[m].mean():+7.0f}  RMSE {np.sqrt((res[m] ** 2).mean()):6.0f}")
    # day-CV
    cv = np.zeros(len(oof))
    for f in range(5):
        tr, te = sel & (fold != f), sel & (fold == f)
        sh = fit_shift(res[tr], band[tr], k)
        cv[te] = sh[band[te]]
    p2 = np.clip(p + cv, L, L + 2 * HALF, where=np.isfinite(L), out=(p + cv).copy()) if True else p + cv
    p2 = np.where(np.isfinite(L), np.clip(p + cv, L, L + 2 * HALF), p)
    for label, m in (("ex_wd", ~wd), ("Jan", ~wd & (month == 1)), ("Jul", ~wd & (month == 7)), ("forced rows", sel)):
        r0 = np.sqrt(((p - y)[m] ** 2).mean()); r1 = np.sqrt(((p2 - y)[m] ** 2).mean())
        print(f"day-CV {label:<12} {r0:8.3f} -> {r1:8.3f}  ({r1 - r0:+.3f})")
    full = fit_shift(res[sel], band[sel], k)
    print("shift per band, fitted on all holdout days:", dict(zip([f"{BANDS[b]:.0f}-{BANDS[b + 1]}" for b in range(len(full))], np.round(full, 0))))
    Lt = lower_bound(mvt, test.index).to_numpy(float)
    pt = test["y_pred"].to_numpy(float)
    selt = np.isfinite(Lt) & (Lt > min_l)
    print(f"2026 forced-long rows: {selt.sum():,}; mean shift {full[band_of(pt - Lt)][selt].mean():+.0f} s")
    if "--write" in sys.argv:
        name = sys.argv[sys.argv.index("--write") + 1]
        tag = hashlib.sha1(f"{run}|{name}|{min_l}|{k}|{pmax}".encode()).hexdigest()[:8]
        dst = paths.RUNS / f"{run.split('__')[0]}__{name}__{tag}"
        dst.mkdir(parents=True, exist_ok=False)
        o2 = oof.copy(); o2["y_pred"] = p2
        o2.to_parquet(dst / "oof_predictions.parquet")
        sh_t = np.where(selt, full[band_of(pt - Lt)], 0.0)
        pt2 = np.where(np.isfinite(Lt), np.clip(pt + sh_t, Lt, Lt + 2 * HALF), pt)
        pd.DataFrame({"y_pred": pt2}, index=test.index).to_parquet(dst / "test_predictions.parquet")
        for f in ("corrected_rows.parquet",):
            if (R / f).exists():
                shutil.copy(R / f, dst / f)
        (dst / "metrics.json").write_text(json.dumps({"stack": {"source": run, "steps": ["forced_long_shift"], "min_l": min_l,
                                                                 "k": k, "pmax": pmax, "shift": full.tolist()}}, indent=1))
        print("wrote", dst.name)


if __name__ == "__main__":
    main()
