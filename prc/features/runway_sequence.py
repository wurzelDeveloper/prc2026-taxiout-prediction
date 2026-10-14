"""The departure's place in the runway's sequence.

`runway_queue` counts takeoffs in fixed windows around this one. This block
reads the sequence itself: the ordinal position within the runway's clock
hour, what took off just before (a heavy ahead costs wake separation), how
the runway is shared with arrivals (mixed-mode gaps and arrivals nearby),
whether the runway was saturated (consecutive takeoffs at the minimum
separation), its throughput against its own best quarter-hours, and how long
the airport's dominant departure runway has been in force -- the data never
states the runway configuration, but a change of the dominant runway is
visible in the takeoffs themselves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc.features.congestion import neighbour_gap, window_count, window_reduce
from prc.features.core import bucket_key, epoch_seconds
from prc.features.registry import FeatureContext, feature

SATURATION_GAP_S = 90.0
"""Consecutive takeoffs closer than this are at the runway's minimum separation."""
CONFIG_BUCKET_S = 900
MAX_CONFIG_AGE_MIN = 720.0
HEAVY = ("H", "J")


def _runway_key(frame: pd.DataFrame) -> pd.Series:
    return frame["apt"].astype("string") + "_" + frame["RUNWAY_mvt"].astype("string").fillna("__NA__")


def config_age_minutes(apt: pd.Series, t: pd.Series, runway: pd.Series) -> pd.Series:
    """Minutes since the airport's dominant departure runway last changed, per departure.

    The dominant runway is the mode over each 15-minute bucket of departures;
    buckets without departures do not count as a change.
    """
    bucket = (t.to_numpy(dtype="float64") // CONFIG_BUCKET_S).astype("int64")
    frame = pd.DataFrame({"apt": apt.astype(str).to_numpy(), "bucket": bucket, "rwy": runway.astype(str).to_numpy()})
    counts = frame.groupby(["apt", "bucket", "rwy"], sort=True).size().rename("n").reset_index()
    mode = counts.sort_values(["apt", "bucket", "n", "rwy"], ascending=[True, True, False, True]).drop_duplicates(["apt", "bucket"])
    mode = mode.sort_values(["apt", "bucket"]).reset_index(drop=True)
    changed = mode["rwy"].ne(mode.groupby("apt")["rwy"].shift(1)) | mode["apt"].ne(mode["apt"].shift(1))
    change_bucket = mode["bucket"].where(changed).groupby(mode["apt"]).ffill()
    mode["age_min"] = ((mode["bucket"] - change_bucket) * CONFIG_BUCKET_S / 60).clip(upper=MAX_CONFIG_AGE_MIN)
    lookup = pd.Series(mode["age_min"].to_numpy(), index=(mode["apt"] + "|" + mode["bucket"].astype(str)).to_numpy())
    return pd.Series((frame["apt"] + "|" + frame["bucket"].astype(str)).map(lookup).to_numpy(), index=apt.index)


@feature(
    group="runway_sequence",
    outputs=(
        "rwy_seq_index_hour",
        "rwy_prev_takeoff_wake",
        "rwy_prev_takeoff_heavy",
        "rwy_gap_prev_arrival",
        "rwy_gap_next_arrival",
        "rwy_arr_pm10m",
        "rwy_saturation_15m",
        "rwy_throughput_ratio_15m",
        "apt_min_since_rwy_config_change",
        "apt_distinct_rwy_30m_bucket",
    ),
    depends_on=("core_keys", "runway_queue", "aircraft_identity"),
    categorical=("rwy_prev_takeoff_wake",),
    doc="Position in the runway's takeoff sequence, the preceding wake, mixed-mode arrivals, saturation and the runway configuration's age.",
)
def runway_sequence(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    dep_all = mvt.loc[mvt["is_dep"]]
    arr_all = mvt.loc[~mvt["is_dep"]]
    q = mvt.loc[ctx.dep_index]

    ref_t = epoch_seconds(dep_all["MVT_TIME_UTC_mvt"])
    ref_g = _runway_key(dep_all)
    q_t = epoch_seconds(q["MVT_TIME_UTC_mvt"])
    q_g = _runway_key(q)

    # Time order within each runway, for the ordinal and the preceding takeoff.
    order = ref_t.sort_values(kind="stable").index
    t_sorted, g_sorted = ref_t.loc[order], ref_g.loc[order]
    hour_key = bucket_key(g_sorted, pd.Series(t_sorted.to_numpy() // 3600, index=order))
    seq_index = pd.Series(hour_key, index=order).groupby(hour_key).cumcount()
    wake_sorted = dep_all.loc[order, "WK_TBL_CAT_flt"].astype("string")
    prev_wake = wake_sorted.groupby(g_sorted.to_numpy(), sort=False).shift(1)
    prev_gap = (t_sorted - t_sorted.groupby(g_sorted.to_numpy(), sort=False).shift(1))
    saturated = prev_gap.lt(SATURATION_GAP_S).astype("float64")  # NaN gap (first of its runway) counts as not saturated

    arr_t = epoch_seconds(arr_all["MVT_TIME_UTC_mvt"])
    arr_g = _runway_key(arr_all)

    back_15 = ctx.dep("runway_queue")["rwy_dep_back_15m"].astype("float64")
    p99 = back_15.groupby(q_g.to_numpy()).transform(lambda s: s.quantile(0.99))

    half_hour = bucket_key(q["apt"].astype("string"), pd.Series(q_t.to_numpy() // 1800, index=q.index))
    distinct = pd.Series(half_hour, index=q.index).map(
        pd.DataFrame({"k": half_hour, "rwy": q["RUNWAY_mvt"].astype("string").to_numpy()}).groupby("k")["rwy"].nunique()
    )

    return pd.DataFrame(
        {
            "rwy_seq_index_hour": seq_index.reindex(ctx.dep_index).astype("Int16"),
            "rwy_prev_takeoff_wake": prev_wake.reindex(ctx.dep_index).fillna("__NA__").astype("category"),
            "rwy_prev_takeoff_heavy": prev_wake.reindex(ctx.dep_index).isin(HEAVY).astype("int8"),
            "rwy_gap_prev_arrival": pd.Series(neighbour_gap(arr_t, arr_g, q_t, q_g, direction="backward"), index=ctx.dep_index).clip(upper=7200).astype("float32"),
            "rwy_gap_next_arrival": pd.Series(neighbour_gap(arr_t, arr_g, q_t, q_g, direction="forward"), index=ctx.dep_index).clip(upper=7200).astype("float32"),
            "rwy_arr_pm10m": window_count(arr_t, arr_g, q_t, q_g, back=600, fwd=600),
            "rwy_saturation_15m": pd.Series(
                window_reduce(t_sorted, g_sorted, saturated, q_t, q_g, back=900, fwd=0, how="mean"), index=ctx.dep_index
            ).astype("float32"),
            "rwy_throughput_ratio_15m": (back_15 / p99.replace(0, np.nan)).astype("float32"),
            "apt_min_since_rwy_config_change": config_age_minutes(dep_all["apt"], ref_t, dep_all["RUNWAY_mvt"]).reindex(ctx.dep_index).astype("float32"),
            "apt_distinct_rwy_30m_bucket": distinct.astype("Int16"),
        },
        index=ctx.dep_index,
    )
