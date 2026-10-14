"""Forecast weather: the TAF in force at takeoff, the TAF ops planned with, the METAR TREND.

The METAR blocks (`prc.features.weather`) say what the weather *was*; these say
what it was *expected* to be. The difference matters operationally: de-icing
crews, runway configurations and flow restrictions are planned hours ahead
from the TAF, and weather that arrives unforecast finds the airport unready.

* `taf_now`      -- the TAF in force at takeoff (the latest issued that covers it);
* `taf_planning` -- the TAF issued at least ``lead_hours`` (6) before takeoff, and
  the *surprise*: observed weather it did not forecast. Against the latest TAF
  the surprise would be near zero by construction, since amendments follow
  the observations;
* `wx_trend`     -- the 2-hour TREND appended to the METAR in force at takeoff.

A separate module from the METAR blocks, so that editing one never
invalidates the other's cache.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from prc import paths
from prc.features._wxlib import (
    MAX_AGE_S,
    VIS_CAP_M,
    asof_index,
    inforce_bounds,
    load_metar,
    range_reduce,
    report_state,
    station_rows,
)
from prc.features.core import epoch_seconds
from prc.features.registry import FeatureContext, feature

HOUR = 3600.0
BIG = float(2**21)  # seconds; longer than any TAF validity, so (position, offset) sorts as one key
TAF_FILES = (paths.TAF_ISSUES_FILE, paths.TAF_SEGMENTS_FILE)
SEGMENT_COLS = (
    "p_wind_kt", "p_gust_kt", "p_vis_m", "p_ceiling_ft", "p_snow", "p_freezing",
    "w_wind_kt", "w_vis_m", "w_ceiling_ft", "w_snow", "w_freezing", "w_ts", "tempo_prob",
)
TREND_KINDS = ("NONE", "NOSIG", "BECMG", "TEMPO", "BECMG+TEMPO", "__NA__")


# --- the TAF tables --------------------------------------------------------------

@functools.lru_cache(maxsize=2)
def _taf_by_station(issues_path: str, segments_path: str, *mtimes: int) -> dict[str, dict[str, np.ndarray]]:
    iss = pd.read_parquet(issues_path)
    seg = pd.read_parquet(segments_path)
    iss = iss.loc[iss["parse_ok"].eq(1)].copy()
    iss["issue"] = epoch_seconds(iss["issue_time"]).to_numpy()
    iss["late"] = iss["is_amd"] | iss["is_cor"]  # at equal issue times, the amendment is the later word
    seg["start"] = epoch_seconds(seg["seg_start"]).to_numpy()

    out = {}
    for station, g in iss.groupby("station", observed=True):
        g = g.sort_values(["issue", "late"], kind="stable").reset_index(drop=True)
        vfrom = epoch_seconds(g["valid_from"]).to_numpy(dtype="float64")
        position = pd.Series(np.arange(len(g)), index=g["taf_id"].to_numpy())
        s = seg.loc[seg["taf_id"].isin(position.index)].copy()
        s["pos"] = position.reindex(s["taf_id"].to_numpy()).to_numpy()
        s = s.sort_values(["pos", "start"], kind="stable")
        pos = s["pos"].to_numpy()
        tab = {
            "issue": g["issue"].to_numpy(dtype="float64"),
            "vfrom": vfrom,
            "vto": epoch_seconds(g["valid_to"]).to_numpy(dtype="float64"),
            "amd": g["is_amd"].to_numpy(dtype="float64"),
            "cnl": g["is_cnl"].to_numpy() > 0,
            "seg_key": pos * BIG + (s["start"].to_numpy(dtype="float64") - vfrom[pos]),
        }
        for c in SEGMENT_COLS:
            tab[c] = s[c].to_numpy(dtype="float64")
        tab["p_snowfrz"] = np.fmax(tab["p_snow"], tab["p_freezing"])
        tab["w_snowfrz"] = np.fmax(tab["w_snow"], tab["w_freezing"])
        out[str(station)] = tab
    return out


def load_tafs() -> dict[str, dict[str, np.ndarray]]:
    """Per station: TAF issues in issue order, and their segments keyed for lookup. Memoised."""
    a, b = TAF_FILES
    return _taf_by_station(str(a), str(b), a.stat().st_mtime_ns, b.stat().st_mtime_ns)


def select_taf(tab: dict, anchor: np.ndarray, tau: np.ndarray, max_back: int = 3) -> np.ndarray:
    """Position of the TAF in force for each query, or -1.

    The latest TAF issued at or before ``anchor`` whose validity covers
    ``tau`` -- stepping back over issues that do not cover it yet (at 05:30 the
    05Z TAF is valid only from 06Z, so the 23Z one still applies). A TAF in force
    that is a cancellation means there is no forecast: -1.
    """
    n = len(tab["issue"])
    pos = np.full(len(anchor), -1)
    if n == 0:
        return pos
    k = np.searchsorted(tab["issue"], anchor, side="right") - 1
    for step in range(max_back + 1):
        c = k - step
        safe = np.clip(c, 0, n - 1)
        with np.errstate(invalid="ignore"):
            ok = (pos < 0) & (c >= 0) & (tab["vfrom"][safe] <= tau) & (tau < tab["vto"][safe])
        pos = np.where(ok, c, pos)
    cancelled = (pos >= 0) & tab["cnl"][np.clip(pos, 0, n - 1)]
    return np.where(cancelled, -1, pos)


def segment_index(tab: dict, pos: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Index of the segment of TAF ``pos`` containing ``tau``, or -1."""
    if len(tab["vfrom"]) == 0:
        return np.full(len(pos), -1)
    safe = np.clip(pos, 0, len(tab["vfrom"]) - 1)
    key = safe * BIG + (tau - tab["vfrom"][safe])
    i = np.searchsorted(tab["seg_key"], key, side="right") - 1
    with np.errstate(invalid="ignore"):
        return np.where((pos >= 0) & (i >= 0) & np.isfinite(key), i, -1)


# --- blocks ------------------------------------------------------------------------

def _departures(mvt: pd.DataFrame, ctx: FeatureContext) -> tuple[pd.DataFrame, np.ndarray, float]:
    dep = mvt.loc[ctx.dep_index, ["apt", "MVT_TIME_UTC_mvt"]]
    t = epoch_seconds(dep["MVT_TIME_UTC_mvt"]).to_numpy(dtype="float64")
    max_age = float(ctx.params.get("max_age_min", MAX_AGE_S / 60)) * 60
    return dep, t, max_age


def _frame(cols: dict[str, np.ndarray], index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame({k: v.astype("float32") for k, v in cols.items()}, index=index)


TAF_NOW = (
    "taf_is_amd",
    "taf_amd_12h",
    "taf_vis_m",
    "taf_ceiling_ft",
    "taf_wind_kt",
    "taf_snowfrz",
    "taf_worst_vis_m",
    "taf_worst_ceiling_ft",
    "taf_worst_snowfrz",
    "taf_worst_ts",
    "taf_temp_prob",
    "taf_f60_worst_vis_m",
    "taf_f60_worst_snowfrz",
)


@feature(
    group="weather_fcst",
    outputs=TAF_NOW,
    depends_on=("core_keys",),
    external=TAF_FILES,
    doc="The TAF in force at takeoff: prevailing and worst-case forecast, and how often it was amended.",
)
def taf_now(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    dep, t, _ = _departures(mvt, ctx)
    tafs = load_tafs()
    cols = {c: np.full(len(dep), np.nan) for c in TAF_NOW}

    for station, rows in station_rows(dep["apt"]).items():
        tab = tafs.get(station)
        if tab is None or not len(rows):
            continue
        q = t[rows]
        # Amendments are issued when the forecast is going wrong: a busy
        # amendment record means weather that is hard to plan for.
        amd_t = tab["issue"][tab["amd"] > 0]
        cols["taf_amd_12h"][rows] = np.searchsorted(amd_t, q, "right") - np.searchsorted(amd_t, q - 12 * HOUR, "right")

        pos = select_taf(tab, q, q)
        i = segment_index(tab, pos, q)
        ok = i >= 0
        r, j, p = rows[ok], i[ok], pos[ok]
        cols["taf_is_amd"][r] = tab["amd"][p]
        for out, key in (
            ("taf_vis_m", "p_vis_m"),
            ("taf_ceiling_ft", "p_ceiling_ft"),
            ("taf_wind_kt", "p_wind_kt"),
            ("taf_snowfrz", "p_snowfrz"),
            ("taf_worst_vis_m", "w_vis_m"),
            ("taf_worst_ceiling_ft", "w_ceiling_ft"),
            ("taf_worst_snowfrz", "w_snowfrz"),
            ("taf_worst_ts", "w_ts"),
            ("taf_temp_prob", "tempo_prob"),
        ):
            cols[out][r] = tab[key][j]
        # The next hour of the same TAF, clipped to its validity.
        j_end = segment_index(tab, p, np.minimum(q[ok] + HOUR, tab["vto"][p] - 1))
        cols["taf_f60_worst_vis_m"][r] = range_reduce(tab["w_vis_m"], j, j_end + 1, "min")
        cols["taf_f60_worst_snowfrz"][r] = range_reduce(tab["w_snowfrz"], j, j_end + 1, "any")
    return _frame(cols, ctx.dep_index)


TAF_PLANNING = (
    "taf6_worst_snowfrz",
    "taf6_worst_vis_m",
    "taf6_lvp",
    "wxs_unforecast_snowfrz",
    "wxs_unforecast_lvp",
    "wxs_vis_log_ratio",
)


@feature(
    group="weather_fcst",
    outputs=TAF_PLANNING,
    depends_on=("core_keys",),
    external=(*TAF_FILES, paths.METAR_FILE),
    doc="The TAF ops planned with (issued >= 6 h before takeoff), and the weather it failed to forecast.",
)
def taf_planning(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    dep, t, max_age = _departures(mvt, ctx)
    lead = float(ctx.params.get("lead_hours", 6)) * HOUR
    tafs, metar = load_tafs(), load_metar()
    cols = {c: np.full(len(dep), np.nan) for c in TAF_PLANNING}

    for station, rows in station_rows(dep["apt"]).items():
        tab = tafs.get(station)
        if tab is None or not len(rows):
            continue
        q = t[rows]
        i = segment_index(tab, select_taf(tab, q - lead, q), q)
        ok = i >= 0
        fc = {k: np.full(len(q), np.nan) for k in ("w_snowfrz", "w_vis_m", "w_ceiling_ft", "p_vis_m")}
        for k in fc:
            fc[k][ok] = tab[k][i[ok]]
        with np.errstate(invalid="ignore"):
            fc_lvp = np.where(ok, ((fc["w_vis_m"] < 600) | (fc["w_ceiling_ft"] < 200)).astype("float64"), np.nan)
        cols["taf6_worst_snowfrz"][rows] = fc["w_snowfrz"]
        cols["taf6_worst_vis_m"][rows] = fc["w_vis_m"]
        cols["taf6_lvp"][rows] = fc_lvp

        if station not in metar:
            continue
        s = report_state(metar[station])
        lo, hi = inforce_bounds(s["t"], q - HOUR, q, max_age)
        k = asof_index(s["t"], q, max_age)
        seen_vis = np.where(k >= 0, s["vis_m"][np.clip(k, 0, None)], np.nan)
        cols["wxs_unforecast_snowfrz"][rows] = range_reduce(s["snowfrz"], lo, hi, "any") * (1 - fc["w_snowfrz"])
        cols["wxs_unforecast_lvp"][rows] = range_reduce(s["lvp"], lo, hi, "any") * (1 - fc_lvp)
        cols["wxs_vis_log_ratio"][rows] = np.log10((seen_vis + 50) / (fc["p_vis_m"] + 50))
    return _frame(cols, ctx.dep_index)


@feature(
    group="weather_fcst",
    outputs=("wx_trend_kind", "wx_trend_min_vis_m", "wx_trend_snowfrz", "wx_trend_lvp"),
    depends_on=("core_keys",),
    categorical=("wx_trend_kind",),
    external=(paths.METAR_FILE,),
    doc="The 2-hour TREND forecast appended to the METAR in force at takeoff.",
)
def wx_trend(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    dep, t, max_age = _departures(mvt, ctx)
    metar = load_metar()
    kind = np.full(len(dep), "__NA__", dtype=object)
    vis, snowfrz, lvp = (np.full(len(dep), np.nan) for _ in range(3))

    for station, rows in station_rows(dep["apt"]).items():
        if station not in metar or not len(rows):
            continue
        rep = metar[station]
        i = asof_index(rep["t"].to_numpy(), t[rows], max_age)
        r, j = rows[i >= 0], i[i >= 0]
        k = rep["trend_kind"].astype(object).to_numpy()[j]
        kind[r] = np.where(k == "", "NONE", k)
        trend_vis = rep["trend_vis_m"].to_numpy(dtype="float64")[j]
        trend_ceiling = rep["trend_ceiling_ft"].to_numpy(dtype="float64")[j]
        # Lowest visibility over the next two hours: now, or what the TREND announces.
        vis[r] = np.fmin(np.minimum(rep["vis_min_m"].to_numpy(dtype="float64")[j], VIS_CAP_M), trend_vis)
        snowfrz[r] = rep["trend_snowfrz"].to_numpy(dtype="float64")[j]
        with np.errstate(invalid="ignore"):
            lvp[r] = ((trend_vis < 600) | (trend_ceiling < 200)).astype("float64")

    kind = np.where(np.isin(kind, TREND_KINDS), kind, "__NA__")
    out = _frame({"wx_trend_min_vis_m": vis, "wx_trend_snowfrz": snowfrz, "wx_trend_lvp": lvp}, ctx.dep_index)
    out.insert(0, "wx_trend_kind", pd.Categorical(kind, categories=TREND_KINDS))
    return out
