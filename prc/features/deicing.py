"""De-icing context: the cold-soaked aircraft and the queue at the de-icing pads.

EUROCONTROL's additional-taxi-out indicator leaves out flights de-iced after off-block (ATXOT
documentation, step 1), and its reference share in January 2026 fell to 0.67 at EDDM, 0.74 at LSZH, 0.79 at
LFPG (0.77-0.95 a year earlier): remote de-icing during the taxi-out was far more common in the scored
January than in any 2025 month (findings 9.1). Two things decide whether a departure needs it and how long
the pads keep it: the weather, which `wx_now`, `wx_past` and `era5_surface` carry, and whether the aircraft
spent the night (or hours) parked in it, which nothing carried so far.

``cold_soak``: the stand's previous arrival blocked in 6 h or more before this takeoff, or none within 48 h
(the aircraft has been parked since at least the previous evening). NaN when the airport had no arrival at
all in the previous 12 h, which means the data starts there (the frame has no December 2024 and no June
2026), not that the aircraft was parked. The conjunctions with frost, snow and METAR de-icing conditions are
separate columns on purpose: a rare three-way interaction the trees did not find on their own (on the Feb+Dec
winter holdout of exp028, snow or de-icing departures after a 6 h+ idle stand carried a +202 s residual
against +112 s for the rest, +285 to +489 s on frost-and-disrupted days). ``cold_soak_deice_dep_back_60m``
counts such departures at the airport in the previous hour, self excluded: the queue at the pads.
``heavy_deice``: a Heavy or Super under de-icing conditions or overnight frost (longer spraying).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc.features.congestion import window_count
from prc.features.core import epoch_seconds
from prc.features.registry import FeatureContext, feature

SOAK_GAP_S = 6 * 3600.0
DATA_EDGE_S = 12 * 3600.0
QUEUE_BACK_S = 3600.0
HEAVY = ("H", "J")

DEICING_CONTEXT = (
    "cold_soak",
    "cold_soak_frost_h",
    "cold_soak_snow_cm",
    "cold_soak_deice",
    "cold_soak_deice_dep_back_60m",
    "heavy_deice",
)


@feature(
    group="deicing",
    outputs=DEICING_CONTEXT,
    depends_on=("core_keys", "prev_stand_arrival", "era5_surface", "wx_now", "aircraft_identity"),
    doc="Cold-soaked aircraft (stand idle 6 h+ or overnight) under frost, snow or METAR de-icing conditions, the queue of such departures in the last hour, and heavies under de-icing conditions.",
)
def deicing_context(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    dep = mvt.loc[ctx.dep_index, ["apt", "MVT_TIME_UTC_mvt"]]
    arr = mvt.loc[~mvt["is_dep"], ["apt", "MVT_TIME_UTC_mvt"]]
    apt = dep["apt"].astype("string")
    t = epoch_seconds(dep["MVT_TIME_UTC_mvt"])

    arrivals_12h = window_count(
        epoch_seconds(arr["MVT_TIME_UTC_mvt"]), arr["apt"].astype("string"), t, apt, back=DATA_EDGE_S, fwd=0
    )
    gap = ctx.dep("prev_stand_arrival")["secs_since_prev_arr_inblock"].to_numpy(dtype="float64")
    soak = np.where(np.isnan(gap) | (gap >= SOAK_GAP_S), 1.0, 0.0)
    soak = np.where(arrivals_12h == 0, np.nan, soak)

    era5 = ctx.dep("era5_surface")
    frost = era5["era5_frost_hours_12h"].to_numpy(dtype="float64")
    snow = (era5["era5_snow_depth_cm"] + era5["era5_snowfall_24h_cm"]).to_numpy(dtype="float64")
    deice = np.nan_to_num(ctx.dep("wx_now")["wx_deice"].to_numpy(dtype="float64")) > 0
    heavy = ctx.dep("aircraft_identity")["wake_category"].astype("string").isin(HEAVY).to_numpy()

    soak_deice = soak * deice
    queued = np.nan_to_num(soak_deice) > 0
    queue = window_count(t[queued], apt[queued], t, apt, back=QUEUE_BACK_S, fwd=0) - queued.astype(np.int64)

    return pd.DataFrame(
        {
            "cold_soak": soak,
            "cold_soak_frost_h": soak * frost,
            "cold_soak_snow_cm": soak * snow,
            "cold_soak_deice": soak_deice,
            "cold_soak_deice_dep_back_60m": queue,
            "heavy_deice": (heavy & (deice | (np.nan_to_num(frost) > 0))).astype("float32"),
        },
        index=ctx.dep_index,
    ).astype("float32")
