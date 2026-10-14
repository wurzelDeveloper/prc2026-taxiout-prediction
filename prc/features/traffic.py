"""Traffic against the airport's normal day: cancellations show up as missing flights.

On 5 January 2026 Schiphol handled 208 departures instead of the usual ~600,
and nothing else in a departure's row says so. The day's (and the hour's)
movement counts against the median for that airport and weekday are computed
over the whole frame -- 2026 included, since they count rows, never targets --
so a snow day, a strike or a runway closure is visible to the 2026 rows the
same way it is to the 2025 ones. `load_normalised` already provides the
departure count of the last hour against its (weekday, hour) median; this
block adds the day and the arrivals.
"""

from __future__ import annotations

import pandas as pd

from prc.features.registry import FeatureContext, feature
from prc.features.temporal import local_naive


def _ratio(count_key: pd.Series, typical_key: pd.Series, counts: pd.Series) -> pd.Series:
    """count / median count over the typical key's groups (medians taken over distinct count keys)."""
    per_key = counts.groupby(count_key.to_numpy()).first()
    typical_of_key = pd.Series(typical_key.to_numpy(), index=count_key.to_numpy()).groupby(level=0).first()
    median = per_key.groupby(typical_of_key.reindex(per_key.index).to_numpy()).median()
    return counts / typical_key.map(median).replace(0, float("nan"))


@feature(
    group="traffic",
    outputs=("apt_day_dep_ratio", "apt_day_arr_ratio", "apt_day_mvt_ratio", "apt_day_dep_count", "apt_hour_arr_ratio"),
    depends_on=("core_keys",),
    doc="The airport's movements on this local day (and arrivals this hour) against its median for that weekday.",
)
def traffic_deviation(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    local = local_naive(mvt["MVT_TIME_UTC_mvt"], mvt["apt"])
    apt = mvt["apt"].astype("string")
    day = apt + "|" + local.dt.strftime("%Y-%m-%d")
    dow = apt + "|" + local.dt.dayofweek.astype("string")
    hour = day + "|" + local.dt.hour.astype("string")
    dow_hour = dow + "|" + local.dt.hour.astype("string")
    is_dep = mvt["is_dep"].to_numpy(dtype=bool)

    dep_per_day = day[is_dep].value_counts()
    arr_per_day = day[~is_dep].value_counts()
    arr_per_hour = hour[~is_dep].value_counts()

    dep_count = day.map(dep_per_day).fillna(0).astype("float64")
    arr_count = day.map(arr_per_day).fillna(0).astype("float64")
    arr_hour_count = hour.map(arr_per_hour).fillna(0).astype("float64")

    out = pd.DataFrame(
        {
            "apt_day_dep_ratio": _ratio(day, dow, dep_count),
            "apt_day_arr_ratio": _ratio(day, dow, arr_count),
            "apt_day_mvt_ratio": _ratio(day, dow, dep_count + arr_count),
            "apt_day_dep_count": dep_count,
            "apt_hour_arr_ratio": _ratio(hour, dow_hour, arr_hour_count),
        },
        index=mvt.index,
    )
    return out.reindex(ctx.dep_index).astype("float32")
