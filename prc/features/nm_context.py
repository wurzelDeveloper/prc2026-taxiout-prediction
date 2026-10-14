"""NM's timestamps against each other, and the airport-hour they sit in.

`nm_deltas` measures every NM off-block time against the takeoff. Two things
it leaves out. First, the NM times against *each other*: the error of the
NM proxy swings from +1,100 s when the actual off-block precedes the last
planned one by half an hour to -900 s when it follows it by an hour (findings
7.2), and a tree needs the difference as a column to split on it; a handful of
data-source fingerprints (an actual off-block time carrying seconds, an actual
equal to the plan, a plan equal to the schedule) ride along. Second, the
airport-hour's NM picture: the mean push-back delay and the share of flights
whose actual off-block ran past the plan say how the hour went for everyone,
which is the shared hourly state findings 7.1 found in the residuals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc.features.core import bucket_key, epoch_seconds
from prc.features.registry import FeatureContext, feature

DAY_S = 86_400.0
EQUAL_TOL_S = 60.0


def _delta(a: pd.Series, b: pd.Series, cap: float | None) -> pd.Series:
    d = (epoch_seconds(a) - epoch_seconds(b)).astype("float64")
    return d if cap is None else d.clip(-cap, cap)


@feature(
    group="nm_context",
    outputs=(
        "aobt3_minus_lobt",
        "lobt_minus_eobt1",
        "aobt3_minus_iobt_full",
        "aobt3_minus_iobt_cap",
        "aobt_eq_lobt",
        "aobt_has_seconds",
        "lobt_eq_sched",
    ),
    depends_on=("core_keys",),
    doc="NM's actual, last-planned, initial and scheduled off-block times against each other, plus data-source fingerprints.",
)
def nm_cross_deltas(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    dep = mvt.loc[ctx.dep_index]
    aobt, lobt, iobt, eobt, sched = (dep[c] for c in ("AOBT_3_flt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "SCHED_TIME_UTC_mvt"))
    a_minus_l = _delta(aobt, lobt, None)
    l_minus_s = _delta(lobt, sched, None)
    return pd.DataFrame(
        {
            "aobt3_minus_lobt": a_minus_l.clip(-DAY_S, DAY_S).astype("float32"),
            "lobt_minus_eobt1": _delta(lobt, eobt, DAY_S).astype("float32"),
            "aobt3_minus_iobt_full": _delta(aobt, iobt, None).astype("float32"),
            "aobt3_minus_iobt_cap": _delta(aobt, iobt, 14_400.0).astype("float32"),
            "aobt_eq_lobt": a_minus_l.abs().le(EQUAL_TOL_S).fillna(False).astype("int8"),
            "aobt_has_seconds": aobt.dt.second.fillna(0).ne(0).astype("int8"),
            "lobt_eq_sched": l_minus_s.abs().le(EQUAL_TOL_S).fillna(False).astype("int8"),
        },
        index=ctx.dep_index,
    )


@feature(
    group="nm_context",
    outputs=(
        "nm_hour_mean_aobt3_minus_eobt1",
        "nm_hour_share_aobt3_gt_lobt",
        "nm_hour_mean_aobt3_minus_sched",
        "nm_hour_share_nm_missing",
        "nm_hour_n_dep",
    ),
    depends_on=("core_keys", "nm_deltas"),
    doc="The airport-hour's NM picture: mean push-back delay, share of actual off-blocks past the plan, share without an NM match.",
)
def nm_hour_context(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    dep = mvt.loc[ctx.dep_index]
    nm = ctx.dep("nm_deltas")
    hour = pd.Series(epoch_seconds(dep["MVT_TIME_UTC_mvt"]).to_numpy() // 3600, index=dep.index)
    key = bucket_key(dep["apt"].astype("string"), hour)

    past_plan = _delta(dep["AOBT_3_flt"], dep["LOBT_flt"], None).gt(EQUAL_TOL_S).astype("float64")
    past_plan = past_plan.where(dep["AOBT_3_flt"].notna() & dep["LOBT_flt"].notna())

    def hour_mean(values: pd.Series) -> pd.Series:
        return values.astype("float64").groupby(key).transform("mean").astype("float32")

    return pd.DataFrame(
        {
            "nm_hour_mean_aobt3_minus_eobt1": hour_mean(nm["aobt3_minus_eobt1"]),
            "nm_hour_share_aobt3_gt_lobt": hour_mean(past_plan),
            "nm_hour_mean_aobt3_minus_sched": hour_mean(nm["aobt3_minus_sched"]),
            "nm_hour_share_nm_missing": hour_mean(nm["nm_join_missing"]),
            "nm_hour_n_dep": pd.Series(key, index=dep.index).groupby(key).transform("size").astype("int32"),
        },
        index=ctx.dep_index,
    )
