"""Stale block times come in clusters: the airport-day around a suspect record.

The stale-schedule segment (departures with no aircraft type and no NM match)
is where the recorded block time is often the schedule rather than a
measurement. Five of the holdout's worst rows fall on 13-14 July 2025 at
Rome, which suggests the defect is a property of the day's data feed rather
than of the flight. The day's count and share of suspect departures at the
airport, the count with a long schedule delta, and the same stand's other
suspect departures that day are computable for 2026 (they count rows, never
targets) and give the regime classifier a day-level fingerprint to work with.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc.features.registry import FeatureContext, feature

LONG_DELTA_S = 7200.0


@feature(
    group="stale_context",
    outputs=("stale_day_suspect_n", "stale_day_suspect_share", "stale_day_long_delta_n", "stale_stand_day_suspect_n"),
    depends_on=("core_keys", "stale_schedule"),
    doc="Suspect (untyped, NM-less) departures at the airport that day and at the same stand, and how many carry a long schedule delta.",
)
def stale_day_context(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    dep = mvt.loc[ctx.dep_index]
    ss = ctx.dep("stale_schedule")
    suspect = ss["stale_suspect"].to_numpy(dtype="float64") > 0
    long_delta = suspect & (ss["sched_delta_full"].to_numpy(dtype="float64") >= LONG_DELTA_S)

    day = dep["apt"].astype("string") + "|" + dep["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d")
    stand_day = day + "|" + dep["STAND_mvt"].astype("string").fillna("__NA__")

    day_n = day.map(day.value_counts()).astype("float64")
    suspect_n = day.map(pd.Series(suspect).groupby(day.to_numpy()).sum()).astype("float64")
    long_n = day.map(pd.Series(long_delta).groupby(day.to_numpy()).sum()).astype("float64")
    stand_n = stand_day.map(pd.Series(suspect).groupby(stand_day.to_numpy()).sum()).astype("float64") - suspect

    return pd.DataFrame(
        {
            "stale_day_suspect_n": suspect_n.astype("float32"),
            "stale_day_suspect_share": (suspect_n / day_n).astype("float32"),
            "stale_day_long_delta_n": long_n.astype("float32"),
            "stale_stand_day_suspect_n": np.maximum(stand_n, 0).astype("float32"),
        },
        index=ctx.dep_index,
    )
