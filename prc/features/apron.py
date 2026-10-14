"""Apron congestion: the pier and the apron zone, forwards and by NM push-back time.

`pier_load` counts departures from the same pier in the minutes *before*
takeoff. Two things it leaves out: the departures that take off shortly
*after* this one (they were pushing back and taxiing at the same time, and the
takeoff time is known for every flight, so looking forward is legal), and
push-back conflicts read off NM's off-block times (two aircraft pushing from
the same pier within five minutes block each other's taxi lanes). Arrivals
blocking in at the same pier are the third source of apron traffic.

`zone_load` does the same on the apron zones of the OSM geometry
(`stand_geometry`), which cut the apron by position rather than by the stand
letter, and which place arrivals as well.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc import paths
from prc.features.congestion import window_count
from prc.features.core import epoch_seconds
from prc.features.geometry import zone_of
from prc.features.registry import FeatureContext, feature
from prc.features.stand_runway import stand_prefix_key

PUSHBACK_WINDOW_S = 300


def _apron_counts(mvt: pd.DataFrame, ctx: FeatureContext, dep_group: pd.Series, arr_group: pd.Series, prefix: str) -> pd.DataFrame:
    dep = mvt.loc[ctx.dep_index]
    arr = mvt.loc[~mvt["is_dep"]]
    t = epoch_seconds(dep["MVT_TIME_UTC_mvt"])
    aobt = epoch_seconds(dep["AOBT_3_flt"])
    arr_inblock = epoch_seconds(arr["BLOCK_TIME_UTC_mvt"])
    known = dep_group.notna().to_numpy()

    def own(counts: np.ndarray, extra_ok: np.ndarray | None = None) -> pd.Series:
        ok = known if extra_ok is None else known & extra_ok
        return pd.Series(np.where(ok, np.maximum(counts - 1, 0), np.nan), index=ctx.dep_index).astype("float32")

    has_aobt = aobt.notna().to_numpy()
    return pd.DataFrame(
        {
            f"{prefix}_dep_back_15m": own(window_count(t, dep_group, t, dep_group, back=900, fwd=0)),
            f"{prefix}_dep_fwd_15m": own(window_count(t, dep_group, t, dep_group, back=0, fwd=900)),
            f"{prefix}_arr_inblock_pm15m": pd.Series(
                np.where(known, window_count(arr_inblock, arr_group, t, dep_group, back=900, fwd=900), np.nan), index=ctx.dep_index
            ).astype("float32"),
            f"{prefix}_aobt3_pm5m": own(
                window_count(aobt, dep_group, aobt, dep_group, back=PUSHBACK_WINDOW_S, fwd=PUSHBACK_WINDOW_S), has_aobt
            ),
        },
        index=ctx.dep_index,
    )


@feature(
    group="pier",
    outputs=("pier_dep_fwd_15m", "pier_aobt3_pm5m", "pier_arr_inblock_pm15m"),
    depends_on=("core_keys", "stand_runway_keys"),
    doc="Same-pier departures in the 15 min after takeoff, push-back conflicts by NM off-block time, and arrivals blocking in at the pier.",
)
def pier_pushback(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    dep = mvt.loc[ctx.dep_index]
    arr = mvt.loc[~mvt["is_dep"]]
    dep_pier = ctx.dep("stand_runway_keys")["stand_prefix"].astype("string")
    arr_pier = stand_prefix_key(arr["apt"], arr["STAND_mvt"])
    counts = _apron_counts(mvt, ctx, dep_pier, arr_pier, "pier")
    # `pier_dep_back_15m` already exists in `pier_load`; only the new views are emitted.
    return counts[["pier_dep_fwd_15m", "pier_aobt3_pm5m", "pier_arr_inblock_pm15m"]]


@feature(
    group="zone",
    outputs=("zone_dep_back_15m", "zone_dep_fwd_15m", "zone_arr_inblock_pm15m", "zone_aobt3_pm5m"),
    depends_on=("core_keys", "stand_geometry"),
    external=(paths.GEO_STANDS_FILE, paths.GEO_STAND_MATCHES_FILE),
    doc="Departures, arrivals and NM push-backs in the same OSM apron zone around this takeoff.",
)
def zone_load(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    arr = mvt.loc[~mvt["is_dep"]]
    dep_zone = ctx.dep("stand_geometry")["geo_stand_zone"].astype("string")
    arr_zone = zone_of(arr["apt"], arr["STAND_mvt"]).astype("string")
    counts = _apron_counts(mvt, ctx, dep_zone, arr_zone, "zone")
    return counts[["zone_dep_back_15m", "zone_dep_fwd_15m", "zone_arr_inblock_pm15m", "zone_aobt3_pm5m"]]
