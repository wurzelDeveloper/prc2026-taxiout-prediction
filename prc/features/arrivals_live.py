"""Live ground conditions, read off the arrivals that just landed.

Departure block times are blanked in 2026; the arrivals' taxi-in times are
not. An arrival that took five minutes longer than its stand/runway pairing
normally does is a measurement of the airport right now -- snow clearance,
a closed taxiway, an apron at capacity -- and on the 2025 data the hourly
mean of that anomaly moves with the departures' taxi-out anomaly at 0.25-0.37
(findings 7.5), most of all on disrupted days, which the Jan+Jul holdout has
none of and the 2026 ranking months have several.

The anomaly definition is shared with `prc.disruption` through `_arrlib`.
Windows are anchored on the arrivals' landing times before takeoff; their
taxi-in may end after the takeoff, which is hindsight of the same kind as
`wx_future` and legal here because the test months are historical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc.features import _arrlib
from prc.features.congestion import window_count, window_reduce
from prc.features.core import epoch_seconds
from prc.features.registry import FeatureContext, feature


@feature(
    group="arrivals_live",
    outputs=(
        "arr_anom_mean_30m",
        "arr_anom_max_30m",
        "arr_anom_mean_60m",
        "arr_anom_max_60m",
        "arr_anom_share_2x_60m",
        "arr_anom_mean_6h",
        "arr_anom_n_60m",
    ),
    depends_on=("core_keys",),
    doc="Taxi-in anomaly of the arrivals that landed at the airport in the last 30 min, 60 min and 6 h.",
)
def arrival_conditions(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    min_support = int(ctx.params.get("min_support", _arrlib.MIN_SUPPORT))
    arr = _arrlib.arrivals(mvt)
    anomaly = _arrlib.taxiin_anomaly(mvt, min_support).reindex(arr.index)
    baseline = _arrlib.taxiin_baseline(arr, min_support)
    twice = (arr["TAXITIME_SEC_mvt"].clip(upper=_arrlib.TAXIIN_CAP_S) > 2 * baseline).astype("float64").where(baseline.notna())

    ref_t = epoch_seconds(arr["MVT_TIME_UTC_mvt"])
    ref_g = arr["apt"].astype("string")
    q = mvt.loc[ctx.dep_index]
    q_t = epoch_seconds(q["MVT_TIME_UTC_mvt"])
    q_g = q["apt"].astype("string")

    def reduce(values: pd.Series, back: int, how: str) -> pd.Series:
        return pd.Series(window_reduce(ref_t, ref_g, values, q_t, q_g, back=back, fwd=0, how=how), index=ctx.dep_index).astype("float32")

    return pd.DataFrame(
        {
            "arr_anom_mean_30m": reduce(anomaly, 1800, "mean"),
            "arr_anom_max_30m": reduce(anomaly, 1800, "max"),
            "arr_anom_mean_60m": reduce(anomaly, 3600, "mean"),
            "arr_anom_max_60m": reduce(anomaly, 3600, "max"),
            "arr_anom_share_2x_60m": reduce(twice, 3600, "mean"),
            "arr_anom_mean_6h": reduce(anomaly, 6 * 3600, "mean"),
            "arr_anom_n_60m": window_count(ref_t, ref_g, q_t, q_g, back=3600, fwd=0),
        },
        index=ctx.dep_index,
    )
