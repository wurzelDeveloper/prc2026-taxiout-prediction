"""Post-processing that follows from how the dataset was built.

**The NM join window.** A movement carries NM flight data only when the airport's
block time lies within about an hour of NM's last known off-block time: on every one
of the 2,062,577 NM-matched 2025 departures, ``LOBT_flt - BLOCK_TIME_UTC_mvt`` is in
[-3,606 s, +3,606 s], at every airport, and a departure with ``LOBT_flt`` always has
an NM match (findings 10.3). The ranking set was joined by the same process before its
block times were blanked, so every matched 2026 departure's target lies in

    [MVT - LOBT - 3,606, MVT - LOBT + 3,606].

Projecting a prediction onto an interval that contains the truth can only shorten its
distance to the truth, so clipping to the window never raises a row's error. It removes
the mixture's hedge toward a schedule an hour or more outside the window (a stale block
time that the join would not have matched) and lifts rows whose takeoff came hours
after NM's last off-block time (Amsterdam's snow days, one Rome record a day apart).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOBT_WINDOW_SEC = 3_606.0
"""Half-width of the join window, the largest |LOBT - block| in the 2025 training data."""


def lobt_bounds(mvt: pd.DataFrame, index: pd.Index) -> tuple[np.ndarray, np.ndarray]:
    """Per row of ``index``: the lower and upper bound of the target implied by the join window (NaN without LOBT)."""
    rows = mvt.loc[index]
    lob = (rows["MVT_TIME_UTC_mvt"] - rows["LOBT_flt"]).dt.total_seconds().to_numpy(dtype="float64")
    return lob - LOBT_WINDOW_SEC, lob + LOBT_WINDOW_SEC


def clip_to_bounds(pred, lo, hi) -> np.ndarray:
    """``pred`` clipped to [lo, hi] where the bounds exist, unchanged where they are NaN."""
    p = np.asarray(pred, dtype="float64")
    lo = np.asarray(lo, dtype="float64")
    hi = np.asarray(hi, dtype="float64")
    out = np.where(np.isfinite(lo), np.maximum(p, lo), p)
    return np.where(np.isfinite(hi), np.minimum(out, hi), out)


def apply_lobt_bound(frame: pd.DataFrame, mvt: pd.DataFrame, column: str = "y_pred") -> tuple[pd.DataFrame, dict]:
    """A copy of ``frame`` with ``column`` clipped to the join window, and a summary of what moved."""
    lo, hi = lobt_bounds(mvt, frame.index)
    before = frame[column].to_numpy(dtype="float64")
    after = clip_to_bounds(before, lo, hi)
    out = frame.copy()
    out[column] = after
    moved = after != before
    info = {
        "rows_with_bound": int(np.isfinite(lo).sum()),
        "rows_raised": int((after > before).sum()),
        "rows_lowered": int((after < before).sum()),
        "sum_sq_shift": float(np.sum((after - before)[moved] ** 2)),
    }
    return out, info
