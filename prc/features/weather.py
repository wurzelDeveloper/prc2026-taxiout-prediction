"""Observed weather at the airport, from METARs: at, before and after takeoff.

The anchor is the takeoff time. Taxi-out happens in the ~15 minutes *before*
it, so the look-back windows are the causally central ones; the look-ahead
block is kept small and left for ablation to judge. Using observations after
takeoff is legal here -- the test months are historical, so every report
exists -- but it is hindsight, not something live operations would have.

NaN means "no METAR in force" and nothing else. An element a report does not
mention gets its good-weather value (RVR 2000 m, visibility 10 km, ceiling
5000 ft): CatBoost sorts NaN below every value, so an unreported RVR would
otherwise read as the worst RVR of all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc import paths
from prc.features._wxlib import (
    MAX_AGE_S,
    asof_index,
    flag_seconds,
    inforce_bounds,
    load_metar,
    range_reduce,
    report_state,
    runway_heading_deg,
    seconds_since,
    station_rows,
    wind_components,
)
from prc.features.core import epoch_seconds
from prc.features.registry import FeatureContext, feature

HOUR = 3600.0

WX_NOW = (
    "wx_wind_kt",
    "wx_gust_kt",
    "wx_headwind_kt",
    "wx_crosswind_kt",
    "wx_vis_m",
    "wx_rvr_min_m",
    "wx_ceiling_ft",
    "wx_temp_c",
    "wx_dewpt_spread_c",
    "wx_precip_int",
    "wx_snowfrz",
    "wx_ts",
    "wx_fog",
    "wx_lvp",
    "wx_deice",
)
WX_PAST = (
    "wx_b60_min_vis_m",
    "wx_b60_min_ceiling_ft",
    "wx_b60_max_gust_kt",
    "wx_b60_any_precip",
    "wx_b180_any_snowfrz",
    "wx_b180_lvp_min",
    "wx_deice_min_6h",
    "wx_snowfrz_hours_24h",
    "wx_hours_since_snowfrz",
)
WX_FUTURE = (
    "wx_f60_min_vis_m",
    "wx_f60_min_ceiling_ft",
    "wx_f60_max_gust_kt",
    "wx_f60_any_precip",
    "wx_f60_any_snowfrz",
)


def _departures(mvt: pd.DataFrame, ctx: FeatureContext, *extra: str) -> tuple[pd.DataFrame, np.ndarray, float]:
    dep = mvt.loc[ctx.dep_index, ["apt", "MVT_TIME_UTC_mvt", *extra]]
    t = epoch_seconds(dep["MVT_TIME_UTC_mvt"]).to_numpy(dtype="float64")
    max_age = float(ctx.params.get("max_age_min", MAX_AGE_S / 60)) * 60
    return dep, t, max_age


def _frame(cols: dict[str, np.ndarray], index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame({k: v.astype("float32") for k, v in cols.items()}, index=index)


def _covered_seconds(obs_t, flag, a, b, max_age) -> np.ndarray:
    """`flag_seconds`, but NaN where no report at all was in force in ``[a, b]``."""
    secs = flag_seconds(obs_t, flag, a, b, max_age)
    any_report = flag_seconds(obs_t, np.ones(len(obs_t)), a, b, max_age)
    return np.where(any_report > 0, secs, np.nan)


@feature(
    group="weather",
    outputs=WX_NOW,
    depends_on=("core_keys",),
    external=(paths.METAR_FILE,),
    doc="The METAR in force at takeoff: runway wind, visibility, ceiling, weather.",
)
def wx_now(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    dep, t, max_age = _departures(mvt, ctx, "RUNWAY_mvt")
    heading = runway_heading_deg(dep["RUNWAY_mvt"])
    metar = load_metar()
    cols = {c: np.full(len(dep), np.nan) for c in WX_NOW}

    for station, rows in station_rows(dep["apt"]).items():
        if station not in metar or not len(rows):
            continue
        s = report_state(metar[station])
        i = asof_index(s["t"], t[rows], max_age)
        r, j = rows[i >= 0], i[i >= 0]
        head, cross = wind_components(s["wind_dir_deg"][j], s["wind_kt"][j], heading[r])
        cols["wx_headwind_kt"][r] = head
        cols["wx_crosswind_kt"][r] = cross
        for out, key in (
            ("wx_wind_kt", "wind_kt"),
            ("wx_gust_kt", "gust_kt"),
            ("wx_vis_m", "vis_m"),
            ("wx_rvr_min_m", "rvr_min_m"),
            ("wx_ceiling_ft", "ceiling_ft"),
            ("wx_temp_c", "temp_c"),
            ("wx_dewpt_spread_c", "dewpt_spread_c"),
            ("wx_precip_int", "precip_int"),
            ("wx_snowfrz", "snowfrz"),
            ("wx_ts", "ts"),
            ("wx_fog", "fog"),
            ("wx_lvp", "lvp"),
            ("wx_deice", "deice"),
        ):
            cols[out][r] = s[key][j]
    return _frame(cols, ctx.dep_index)


@feature(
    group="weather",
    outputs=WX_PAST,
    depends_on=("core_keys",),
    external=(paths.METAR_FILE,),
    doc="Weather over the hours before takeoff: the taxi itself, and the de-icing backlog.",
)
def wx_past(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    dep, t, max_age = _departures(mvt, ctx)
    metar = load_metar()
    cols = {c: np.full(len(dep), np.nan) for c in WX_PAST}

    for station, rows in station_rows(dep["apt"]).items():
        if station not in metar or not len(rows):
            continue
        s = report_state(metar[station])
        o, q = s["t"], t[rows]

        lo, hi = inforce_bounds(o, q - HOUR, q, max_age)
        cols["wx_b60_min_vis_m"][rows] = range_reduce(s["vis_m"], lo, hi, "min")
        cols["wx_b60_min_ceiling_ft"][rows] = range_reduce(s["ceiling_ft"], lo, hi, "min")
        cols["wx_b60_max_gust_kt"][rows] = range_reduce(s["gust_kt"], lo, hi, "max")
        cols["wx_b60_any_precip"][rows] = range_reduce(s["precip"], lo, hi, "any")

        lo, hi = inforce_bounds(o, q - 3 * HOUR, q, max_age)
        cols["wx_b180_any_snowfrz"][rows] = range_reduce(s["snowfrz"], lo, hi, "any")
        cols["wx_b180_lvp_min"][rows] = _covered_seconds(o, s["lvp"], q - 3 * HOUR, q, max_age) / 60
        cols["wx_deice_min_6h"][rows] = _covered_seconds(o, s["deice"], q - 6 * HOUR, q, max_age) / 60
        cols["wx_snowfrz_hours_24h"][rows] = _covered_seconds(o, s["snowfrz"], q - 24 * HOUR, q, max_age) / HOUR

        # Snow and freezing weather leave work behind (contaminated runways,
        # aircraft needing de-icing) well after the precipitation stops.
        since = seconds_since(o, s["snowfrz"], q, 72 * HOUR) / HOUR
        cols["wx_hours_since_snowfrz"][rows] = np.where(asof_index(o, q, max_age) >= 0, since, np.nan)
    return _frame(cols, ctx.dep_index)


@feature(
    group="weather",
    outputs=WX_FUTURE,
    depends_on=("core_keys",),
    external=(paths.METAR_FILE,),
    doc="Observed weather in the hour after takeoff (hindsight: the test months are historical).",
)
def wx_future(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    dep, t, max_age = _departures(mvt, ctx)
    metar = load_metar()
    cols = {c: np.full(len(dep), np.nan) for c in WX_FUTURE}

    for station, rows in station_rows(dep["apt"]).items():
        if station not in metar or not len(rows):
            continue
        s = report_state(metar[station])
        lo, hi = inforce_bounds(s["t"], t[rows], t[rows] + HOUR, max_age)
        cols["wx_f60_min_vis_m"][rows] = range_reduce(s["vis_m"], lo, hi, "min")
        cols["wx_f60_min_ceiling_ft"][rows] = range_reduce(s["ceiling_ft"], lo, hi, "min")
        cols["wx_f60_max_gust_kt"][rows] = range_reduce(s["gust_kt"], lo, hi, "max")
        cols["wx_f60_any_precip"][rows] = range_reduce(s["precip"], lo, hi, "any")
        cols["wx_f60_any_snowfrz"][rows] = range_reduce(s["snowfrz"], lo, hi, "any")
    return _frame(cols, ctx.dep_index)
