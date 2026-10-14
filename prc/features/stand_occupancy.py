"""More about the stand: who occupied it before, who comes next, how busy it is.

`prev_stand_arrival` already reads the last arrival's taxi-in and how long ago
it parked. This block adds what that arrival was (its own delay against its
schedule, its wake category, whether it belongs to the same carrier as the
departure -- the strongest same-airframe proxy the data allows, since there is
no registration), the stand's traffic in the hours before takeoff, the
occupant before that one, and the **next** occupant: the arrival that blocks in
at this stand after the departure takes off. Arrivals keep their in-block
times in 2026, so looking forward is legal, and a stand re-occupied minutes
after takeoff says the aircraft left it well before.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc.features.congestion import window_count
from prc.features.core import epoch_seconds
from prc.features.registry import FeatureContext, feature
from prc.features.stale_schedule import carrier_prefix
from prc.features.turnaround import _gap, stand_asof, stand_key

SIX_HOURS = 6 * 3600


@feature(
    group="turnaround_ext",
    outputs=(
        "prev_arr_sched_delay_sec",
        "prev_arr_wake_cat",
        "prev_arr_carrier_matches",
        "prev_arr_origin_country",
        "stand_arr_inblock_back_6h",
        "stand_dep_back_6h",
        "secs_since_prev2_arr_inblock",
        "secs_to_next_arr_inblock",
        "sched_ground_time_sec",
    ),
    depends_on=("core_keys", "stand_runway_keys"),
    categorical=("prev_arr_wake_cat", "prev_arr_origin_country"),
    doc="The previous occupant's delay, wake and carrier, the stand's recent traffic, and the next occupant.",
)
def stand_occupancy(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    dep = mvt.loc[ctx.dep_index]
    arr = mvt.loc[~mvt["is_dep"]]
    extra = {
        "arr_sched": epoch_seconds(arr["SCHED_TIME_UTC_mvt"]).to_numpy(),
        "arr_wake": arr["WK_TBL_CAT_flt"].astype("string").to_numpy(),
        "arr_carrier": carrier_prefix(arr["FLIGHT_mvt"]).to_numpy(),
        "arr_origin": arr["ADEP_mvt"].astype("string").str[:2].to_numpy(),
    }
    back, table = stand_asof(dep, arr, direction="backward", extra=extra)
    fwd, _ = stand_asof(dep, arr, direction="forward", allow_exact_matches=False)

    max_age = float(ctx.params.get("max_age_sec", 86_400))
    since = _gap(back["t"], back["arr_inblock"])
    recent = since.le(max_age).fillna(False)

    dep_carrier = carrier_prefix(dep["FLIGHT_mvt"])
    arr_carrier = pd.Series(back["arr_carrier"].to_numpy(), index=ctx.dep_index, dtype="string")
    carrier_match = recent & arr_carrier.notna() & dep_carrier.ne("__NA__") & arr_carrier.eq(dep_carrier).fillna(False)

    # The occupant before the matched arrival: walk one row back within the stand's own arrivals.
    prev_inblock = table.groupby("stand_key", sort=False)["arr_inblock"].shift(1).to_numpy(dtype="float64")
    pos = back["arr_pos"].to_numpy(dtype="float64")
    has_pos = np.isfinite(pos)
    prev2 = np.full(len(dep), np.nan)
    prev2[has_pos] = prev_inblock[pos[has_pos].astype(int)]

    # Traffic at the stand in the six hours before takeoff: arrivals by in-block, departures by takeoff (self excluded).
    t = epoch_seconds(dep["MVT_TIME_UTC_mvt"])
    key = stand_key(dep)
    arr_key = stand_key(arr)
    arr_inblock = epoch_seconds(arr["BLOCK_TIME_UTC_mvt"])
    arr_6h = window_count(arr_inblock, arr_key, t, key, back=SIX_HOURS, fwd=0)
    dep_6h = np.maximum(window_count(t, key, t, key, back=SIX_HOURS, fwd=0) - 1, 0)

    sched = epoch_seconds(dep["SCHED_TIME_UTC_mvt"])
    ground = (sched - back["arr_inblock"]).astype("float64")
    ground = ground.where(carrier_match & (ground >= -3600) & (ground <= 172_800))

    next_gap = (fwd["arr_inblock"] - fwd["t"]).astype("float64")
    next_gap = next_gap.where((next_gap >= 0) & (next_gap <= 172_800))

    return pd.DataFrame(
        {
            "prev_arr_sched_delay_sec": (back["arr_inblock"] - back["arr_sched"]).where(recent).clip(-7200, 86_400).astype("float32"),
            "prev_arr_wake_cat": back["arr_wake"].where(recent).fillna("__NA__").astype("category"),
            "prev_arr_carrier_matches": carrier_match.astype("int8"),
            "prev_arr_origin_country": back["arr_origin"].where(recent).fillna("__NA__").astype("category"),
            "stand_arr_inblock_back_6h": arr_6h,
            "stand_dep_back_6h": dep_6h.astype("int32"),
            "secs_since_prev2_arr_inblock": _gap(back["t"], pd.Series(prev2, index=ctx.dep_index)),
            "secs_to_next_arr_inblock": next_gap.astype("float32"),
            "sched_ground_time_sec": ground.astype("float32"),
        },
        index=ctx.dep_index,
    )
