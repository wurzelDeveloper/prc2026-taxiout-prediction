"""The NM join-window clip on a synthetic frame (no competition data)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc.postprocess import LOBT_WINDOW_SEC, apply_lobt_bound, clip_to_bounds, lobt_bounds


def _mvt():
    t = pd.Timestamp("2026-07-17 21:51:00", tz="UTC")
    return pd.DataFrame(
        {
            "MVT_TIME_UTC_mvt": [t, t, t, t],
            # LOBT 20 min, 3 h, a day before takeoff; one row without LOBT
            "LOBT_flt": [t - pd.Timedelta(minutes=20), t - pd.Timedelta(hours=3), t - pd.Timedelta(seconds=86_820), pd.NaT],
        },
        index=pd.Index([1, 2, 3, 4], name="mvt_id"),
    )


def test_bounds_are_the_join_window_around_takeoff_minus_lobt():
    lo, hi = lobt_bounds(_mvt(), pd.Index([1, 2, 3, 4]))
    assert lo[0] == 1_200 - LOBT_WINDOW_SEC and hi[0] == 1_200 + LOBT_WINDOW_SEC
    assert lo[1] == 10_800 - LOBT_WINDOW_SEC
    assert np.isnan(lo[3]) and np.isnan(hi[3])


def test_clip_moves_only_rows_outside_the_window_and_never_raises_an_error():
    mvt = _mvt()
    frame = pd.DataFrame({"y_pred": [900.0, 1_500.0, 1_683.0, 50_000.0]}, index=mvt.index)
    out, info = apply_lobt_bound(frame, mvt)
    assert out["y_pred"].tolist() == [900.0, 10_800 - LOBT_WINDOW_SEC, 86_820 - LOBT_WINDOW_SEC, 50_000.0]
    assert info == {"rows_with_bound": 3, "rows_raised": 2, "rows_lowered": 0,
                    "sum_sq_shift": (10_800 - LOBT_WINDOW_SEC - 1_500) ** 2 + (86_820 - LOBT_WINDOW_SEC - 1_683) ** 2}
    # any truth inside the window is at least as close to the clipped prediction
    rng = np.random.default_rng(0)
    lo, hi = np.zeros(1_000), np.full(1_000, 100.0)
    truth = rng.uniform(0, 100, 1_000)
    pred = rng.normal(50, 200, 1_000)
    assert np.all(np.abs(clip_to_bounds(pred, lo, hi) - truth) <= np.abs(pred - truth) + 1e-12)
