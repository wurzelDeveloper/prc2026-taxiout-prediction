"""Traffic during the flight's own taxi, as NM saw it.

NM's ``AOBT_3_flt`` is an independent estimate of when this aircraft left its
stand. Counting what happened between that moment and the takeoff -- how many
aircraft used the same runway, how many departed or landed at the airport --
is the natural queue-position feature, and the order of NM off-block times
against the order of takeoffs says who overtook whom in the queue.

Findings 7.2 measured these against the residuals of exp028 and found no
signal (the fixed windows and the proxy already carry it); the block exists so
the ablation can settle it. NaN wherever the flight has no NM off-block time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc.features.congestion import window_count
from prc.features.core import epoch_seconds
from prc.features.registry import FeatureContext, feature

MAX_TAXI_WINDOW_S = 14_400.0
OVERTAKE_WINDOW_S = 1800.0
OVERTAKE_NEIGHBOURS = 60


def _runway_key(frame: pd.DataFrame) -> pd.Series:
    return frame["apt"].astype("string") + "_" + frame["RUNWAY_mvt"].astype("string").fillna("__NA__")


def overtakes(t: np.ndarray, aobt: np.ndarray, group: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per event: how many of the previous takeoffs in its group within `OVERTAKE_WINDOW_S` had a later NM
    off-block (they overtook me), and how many of the next takeoffs had an earlier one (I overtook them).

    Inputs are aligned arrays sorted by ``t`` within group (group codes int64,
    NaN off-block times excluded from every comparison); at most
    `OVERTAKE_NEIGHBOURS` neighbours each side are examined.
    """
    n = len(t)
    by = np.zeros(n, dtype="int32")
    me = np.zeros(n, dtype="int32")
    valid = np.isfinite(aobt)
    for k in range(1, OVERTAKE_NEIGHBOURS + 1):
        if k >= n:
            break
        same = group[k:] == group[:-k]
        close = (t[k:] - t[:-k]) <= OVERTAKE_WINDOW_S
        both = valid[k:] & valid[:-k] & same & close
        # rows i = k..n-1 look back at i-k: overtaken when the earlier takeoff pushed later
        later = both & (aobt[:-k] > aobt[k:])
        by[k:] += later
        # rows i = 0..n-1-k look ahead at i+k: overtook when the later takeoff pushed earlier
        earlier = both & (aobt[k:] < aobt[:-k])
        me[:-k] += earlier
    return by, me


@feature(
    group="taxi_windows",
    outputs=(
        "tw_rwy_dep_during_taxi",
        "tw_apt_dep_during_taxi",
        "tw_apt_arr_during_taxi",
        "tw_apt_dep_since_sched",
        "tw_overtaken_by_30m",
        "tw_i_overtook_30m",
    ),
    depends_on=("core_keys",),
    doc="Takeoffs, departures and landings between NM's off-block time and this takeoff; queue overtakes by NM off-block order.",
)
def taxi_window_counts(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    dep_all = mvt.loc[mvt["is_dep"]]
    arr_all = mvt.loc[~mvt["is_dep"]]
    q = mvt.loc[ctx.dep_index]

    t = epoch_seconds(q["MVT_TIME_UTC_mvt"])
    aobt = epoch_seconds(q["AOBT_3_flt"])
    sched = epoch_seconds(q["SCHED_TIME_UTC_mvt"])
    taxi_back = (t - aobt).where(lambda d: (d > 0) & (d <= MAX_TAXI_WINDOW_S)).to_numpy(dtype="float64")
    sched_back = (t - sched).where(lambda d: (d > 0) & (d <= MAX_TAXI_WINDOW_S)).to_numpy(dtype="float64")

    dep_t, arr_t = epoch_seconds(dep_all["MVT_TIME_UTC_mvt"]), epoch_seconds(arr_all["MVT_TIME_UTC_mvt"])
    q_apt, dep_apt, arr_apt = q["apt"].astype("string"), dep_all["apt"].astype("string"), arr_all["apt"].astype("string")
    q_rwy, dep_rwy = _runway_key(q), _runway_key(dep_all)

    def during(counts: np.ndarray, back: np.ndarray, own: int) -> pd.Series:
        # the window [t-back, t] includes the departure's own takeoff, hence `own`
        out = np.where(np.isfinite(back), np.maximum(counts - own, 0), np.nan)
        return pd.Series(out, index=ctx.dep_index).astype("float32")

    # Overtakes on the same runway, from the takeoff order of all departures.
    order = dep_t.sort_values(kind="stable").index
    codes = pd.Categorical(dep_rwy.loc[order]).codes.astype("int64")
    by, me = overtakes(dep_t.loc[order].to_numpy(dtype="float64"), epoch_seconds(dep_all.loc[order, "AOBT_3_flt"]).to_numpy(dtype="float64"), codes)
    has_aobt = dep_all.loc[order, "AOBT_3_flt"].notna().to_numpy()
    by_s = pd.Series(np.where(has_aobt, by, np.nan), index=order).reindex(ctx.dep_index)
    me_s = pd.Series(np.where(has_aobt, me, np.nan), index=order).reindex(ctx.dep_index)

    return pd.DataFrame(
        {
            "tw_rwy_dep_during_taxi": during(window_count(dep_t, dep_rwy, t, q_rwy, back=taxi_back, fwd=0), taxi_back, 1),
            "tw_apt_dep_during_taxi": during(window_count(dep_t, dep_apt, t, q_apt, back=taxi_back, fwd=0), taxi_back, 1),
            "tw_apt_arr_during_taxi": during(window_count(arr_t, arr_apt, t, q_apt, back=taxi_back, fwd=0), taxi_back, 0),
            "tw_apt_dep_since_sched": during(window_count(dep_t, dep_apt, t, q_apt, back=sched_back, fwd=0), sched_back, 1),
            "tw_overtaken_by_30m": by_s.astype("float32"),
            "tw_i_overtook_30m": me_s.astype("float32"),
        },
        index=ctx.dep_index,
    )
