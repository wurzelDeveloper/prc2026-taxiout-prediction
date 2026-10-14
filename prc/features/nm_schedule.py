"""Network Manager timestamps versus the actual takeoff time.

This is the strongest group. The trick is that ranking blanks only the reporting
airport's off-block time -- NM's own actual off-block time (``AOBT_3_flt``)
survives, for ~98% of departures. So ``MVT_TIME - AOBT_3_flt`` is a direct,
independently-sourced estimate of the very quantity being predicted: where both
are present it scores RMSE 385 / MAE 238 against the true taxi time, versus
~750 RMSE for a global median.

It is an estimate, not the answer -- the two sources disagree with a std of
~390s -- which is what leaves room for a model.

Note the gap between that 385 and the 673 the `aobt3_proxy` *baseline* scores on
the holdout: the difference is almost entirely the ~1.5% of departures NM never
matched, which fall back to a global median and include some of the extreme
multi-hour records that dominate RMSE. Predicting those rows well is a separate
problem from using the proxy well, which is why `aobt3_missing` and
`nm_join_missing` are features in their own right.
"""

from __future__ import annotations

import pandas as pd

from prc.features.core import bucket_key, epoch_seconds
from prc.features.registry import FeatureContext, feature


def _delta(a: pd.Series, b: pd.Series, clip: float = 86_400.0) -> pd.Series:
    """Seconds from ``b`` to ``a``, with absurd values dropped to NaN.

    Real records contain multi-day offsets from mis-joins; keeping them would
    let a handful of rows dominate a split.
    """
    d = (a - b).dt.total_seconds()
    return d.where(d.abs() <= clip).astype("float32")


@feature(
    group="nm_schedule",
    outputs=(
        "aobt3_proxy",
        "aobt3_missing",
        "nm_join_missing",
        "eobt1_delta",
        "iobt_delta",
        "lobt_delta",
        "sched_delta",
        "aobt3_minus_eobt1",
        "aobt3_minus_sched",
        "eobt1_minus_sched",
        "m1_flight_dur",
        "m3_flight_dur",
        "m3_minus_m1_dur",
        "is_diverted",
        "ades_mvt_flt_mismatch",
    ),
    depends_on=("core_keys",),
    doc="Deltas between takeoff time and the NM/scheduled timestamps.",
)
def nm_deltas(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    dep = mvt.loc[ctx.dep_index]
    takeoff = dep["MVT_TIME_UTC_mvt"]

    aobt3 = _delta(takeoff, dep["AOBT_3_flt"])
    eobt1 = _delta(takeoff, dep["EOBT_1_flt"])

    ades_flt = dep["ADES_flt"].astype("string")
    ades_mvt = dep["ADES_mvt"].astype("string")
    ades_filed = dep["ADES_FILED_flt"].astype("string")

    return pd.DataFrame(
        {
            # The headline feature: NM's view of taxi-out duration.
            "aobt3_proxy": aobt3,
            "aobt3_missing": dep["AOBT_3_flt"].isna().astype("int8"),
            "nm_join_missing": dep["FLIGHT_ID_mvt"].isna().astype("int8"),
            "eobt1_delta": eobt1,
            "iobt_delta": _delta(takeoff, dep["IOBT_flt"]),
            "lobt_delta": _delta(takeoff, dep["LOBT_flt"]),
            "sched_delta": _delta(takeoff, dep["SCHED_TIME_UTC_mvt"]),
            # How late NM thinks the flight pushed back -- a delay signal that
            # says nothing about taxi duration directly.
            "aobt3_minus_eobt1": _delta(dep["AOBT_3_flt"], dep["EOBT_1_flt"]),
            "aobt3_minus_sched": _delta(dep["AOBT_3_flt"], dep["SCHED_TIME_UTC_mvt"]),
            "eobt1_minus_sched": _delta(dep["EOBT_1_flt"], dep["SCHED_TIME_UTC_mvt"]),
            "m1_flight_dur": _delta(dep["ARVT_1_flt"], dep["EOBT_1_flt"]),
            "m3_flight_dur": _delta(dep["ARVT_3_flt"], dep["AOBT_3_flt"]),
            "m3_minus_m1_dur": (
                _delta(dep["ARVT_3_flt"], dep["AOBT_3_flt"]) - _delta(dep["ARVT_1_flt"], dep["EOBT_1_flt"])
            ).astype("float32"),
            "is_diverted": (ades_flt.notna() & ades_filed.notna() & ades_flt.ne(ades_filed)).astype("int8"),
            "ades_mvt_flt_mismatch": (ades_flt.notna() & ades_mvt.notna() & ades_flt.ne(ades_mvt)).astype("int8"),
        },
        index=ctx.dep_index,
    )


@feature(
    group="nm_schedule",
    outputs=("aobt3_proxy_vs_apt_hour_med", "aobt3_proxy_rank_in_apt_hour", "aobt3_proxy_apt_hour_count"),
    depends_on=("core_keys", "nm_deltas"),
    doc="The AOBT3 proxy positioned against its airport-hour peers.",
)
def nm_proxy_context(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """Whether this flight's NM-implied taxi time is long *for that airport, that hour*.

    The raw proxy conflates two things: how congested the airport is right now,
    and how unusual this particular flight is. Centring it on the airport-hour
    median separates them.
    """
    dep = mvt.loc[ctx.dep_index]
    proxy = ctx.dep("nm_deltas")["aobt3_proxy"]
    hour_bucket = pd.Series(epoch_seconds(dep["MVT_TIME_UTC_mvt"]).to_numpy() // 3600, index=dep.index)
    key = bucket_key(dep["apt"].astype("string"), hour_bucket)

    grouped = proxy.groupby(key)
    med = grouped.transform("median")
    rank = grouped.rank(pct=True)
    count = grouped.transform("size")

    return pd.DataFrame(
        {
            "aobt3_proxy_vs_apt_hour_med": (proxy - med).astype("float32"),
            "aobt3_proxy_rank_in_apt_hour": rank.astype("float32"),
            "aobt3_proxy_apt_hour_count": count.astype("int32"),
        },
        index=ctx.dep_index,
    )
