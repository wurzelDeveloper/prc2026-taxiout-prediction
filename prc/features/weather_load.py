"""Weather times traffic: the de-icing queue.

De-icing conditions alone say little about the taxi -- an empty airport in
freezing fog has no queue. What builds one is the number of aircraft that need
the pads at the same time, so this block counts departures in the last
30 minutes that took off under de-icing conditions, arrivals in the last hour
that landed under them (the aircraft that will need de-icing on the way out),
and multiplies the snow hours of the last day by the hour's departure load.
Depends on the METAR blocks and the METAR table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc import paths
from prc.features._wxlib import asof_index, load_metar, report_state, station_rows
from prc.features.congestion import window_count
from prc.features.core import epoch_seconds
from prc.features.registry import FeatureContext, feature
from prc.features.weather import _departures


@feature(
    group="weather_load",
    outputs=("wx_deice_dep_back_30m", "wx_deice_arr_back_60m", "wx_snow_hours_x_load"),
    depends_on=("core_keys", "wx_now", "wx_past", "airport_load"),
    external=(paths.METAR_FILE,),
    doc="Departures and arrivals under de-icing conditions around this takeoff, and snow hours times departure load.",
)
def wx_deice_load(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    dep, t, max_age = _departures(mvt, ctx)
    q_apt = dep["apt"].astype("string")
    q_t = pd.Series(t, index=dep.index)

    # Departures under de-icing conditions: the flag `wx_now` already computed for every departure.
    deice_dep = ctx.dep("wx_now")["wx_deice"].to_numpy(dtype="float64") > 0
    dep_ref = q_t[deice_dep]
    dep_ref_apt = q_apt[deice_dep]

    # Arrivals under de-icing conditions at their landing time.
    arr = mvt.loc[~mvt["is_dep"], ["apt", "MVT_TIME_UTC_mvt"]]
    arr_t = epoch_seconds(arr["MVT_TIME_UTC_mvt"]).to_numpy(dtype="float64")
    arr_deice = np.zeros(len(arr), dtype=bool)
    metar = load_metar()
    for station, rows in station_rows(arr["apt"]).items():
        if station not in metar or not len(rows):
            continue
        s = report_state(metar[station])
        i = asof_index(s["t"], arr_t[rows], max_age)
        hit = i >= 0
        arr_deice[rows[hit]] = s["deice"][i[hit]] > 0
    arr_ref = pd.Series(arr_t[arr_deice], index=arr.index[arr_deice])
    arr_ref_apt = arr["apt"].astype("string")[arr_deice]

    snow_hours = ctx.dep("wx_past")["wx_snowfrz_hours_24h"].astype("float64")
    load = ctx.dep("airport_load")["apt_dep_back_60m"].astype("float64")

    return pd.DataFrame(
        {
            "wx_deice_dep_back_30m": window_count(dep_ref, dep_ref_apt, q_t, q_apt, back=1800, fwd=0),
            "wx_deice_arr_back_60m": window_count(arr_ref, arr_ref_apt, q_t, q_apt, back=3600, fwd=0),
            "wx_snow_hours_x_load": (snow_hours * load).astype("float32"),
        },
        index=ctx.dep_index,
    )
