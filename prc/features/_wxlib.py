"""Weather primitives for the feature blocks: METAR lookups, windows, thresholds.

The leading underscore keeps this module out of block auto-import, but its
source is still hashed into every block module that imports from it
(`build._related_modules`) -- so editing a threshold here invalidates exactly
the weather caches and nothing else.

Everything works per station on float64 epoch seconds with `np.searchsorted`:
exact (no time grid, no bin edges) and cheap, since the METAR table is a few
hundred thousand rows against a few million departures.

A report is *in force* from its observation time until the next report -- but
never longer than ``max_age_s``: a station that goes silent must not keep
reporting its last weather for hours.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from prc import paths
from prc.features.core import epoch_seconds

MAX_AGE_S = 90 * 60  # three missed half-hourly reports
VIS_CAP_M = 10_000.0
CEIL_CAP_FT = 5_000.0
RVR_CAP_M = 2_000.0  # RVR is only reported when it matters (below ~1500 m)


# --- time primitives ---------------------------------------------------------

def report_end(obs_t: np.ndarray, max_age_s: float) -> np.ndarray:
    """When each report stops being in force: the next report, or ``max_age_s`` later.

    Non-decreasing whenever ``obs_t`` is sorted, which `inforce_bounds` relies on.
    """
    nxt = np.append(obs_t[1:], np.inf)
    return np.minimum(nxt, obs_t + max_age_s)


def asof_index(obs_t: np.ndarray, q_t: np.ndarray, max_age_s: float) -> np.ndarray:
    """Index of the report in force at each query time, or -1 if none is."""
    if len(obs_t) == 0:
        return np.full(len(q_t), -1)
    idx = np.searchsorted(obs_t, q_t, side="right") - 1
    safe = np.clip(idx, 0, None)
    with np.errstate(invalid="ignore"):
        live = (idx >= 0) & (q_t - obs_t[safe] < max_age_s)
    return np.where(live, idx, -1)


def inforce_bounds(obs_t: np.ndarray, a: np.ndarray, b: np.ndarray, max_age_s: float) -> tuple[np.ndarray, np.ndarray]:
    """``[lo, hi)``: the reports whose in-force interval overlaps ``[a, b]``.

    Report *i* is in force over ``[o_i, e_i)``; it overlaps when ``o_i <= b`` and
    ``e_i > a``. Both conditions are monotone in *i*, so two searches suffice.
    """
    lo = np.searchsorted(report_end(obs_t, max_age_s), a, side="right")
    hi = np.searchsorted(obs_t, b, side="right")
    return lo, np.maximum(hi, lo)


def range_reduce(values: np.ndarray, lo: np.ndarray, hi: np.ndarray, how: str) -> np.ndarray:
    """``min`` / ``max`` / ``any`` of ``values[lo:hi]`` per query; NaN for an empty range.

    Missing values are skipped. ``any`` expects 0/1 flags (it is a max).
    """
    out = np.full(len(lo), np.nan)
    if len(values) == 0 or len(lo) == 0:
        return out
    vals = np.asarray(values, dtype="float64")
    reduce = np.fmin if how == "min" else np.fmax
    n = hi - lo
    for k in range(int(n.max())):
        v = vals[np.minimum(lo + k, len(vals) - 1)]
        out = np.where(n > k, reduce(out, v), out)
    return out


def flag_seconds(obs_t: np.ndarray, flag: np.ndarray, a: np.ndarray, b: np.ndarray, max_age_s: float) -> np.ndarray:
    """Exact seconds within ``[a, b]`` during which a report with ``flag`` set was in force.

    Integrates the piecewise-constant in-force signal: ``C(x)`` is the flagged
    time elapsed up to ``x`` (full durations of every earlier report, plus the
    part of the current one), and the answer is ``C(b) - C(a)``.
    """
    if len(obs_t) == 0:
        return np.where(np.isnan(a) | np.isnan(b), np.nan, 0.0)
    f = np.nan_to_num(np.asarray(flag, dtype="float64"))
    dur = report_end(obs_t, max_age_s) - obs_t
    before = np.concatenate([[0.0], np.cumsum(f * dur)])

    def elapsed(x: np.ndarray) -> np.ndarray:
        i = np.searchsorted(obs_t, x, side="right") - 1
        safe = np.clip(i, 0, None)
        partial = f[safe] * np.clip(x - obs_t[safe], 0.0, dur[safe])
        return np.where(i >= 0, before[safe] + partial, 0.0)

    return elapsed(b) - elapsed(a)


def seconds_since(obs_t: np.ndarray, flag: np.ndarray, q_t: np.ndarray, cap_s: float) -> np.ndarray:
    """Seconds since the latest flagged report at or before each query; ``cap_s`` if none."""
    ft = obs_t[np.nan_to_num(np.asarray(flag, dtype="float64")).astype(bool)]
    if len(ft) == 0:
        return np.where(np.isnan(q_t), np.nan, cap_s)
    i = np.searchsorted(ft, q_t, side="right") - 1
    since = np.where(i >= 0, q_t - ft[np.clip(i, 0, None)], cap_s)
    return np.minimum(since, cap_s)


# --- geometry ----------------------------------------------------------------

def runway_heading_deg(runway: pd.Series) -> np.ndarray:
    """Heading from a designator: ``25C`` -> 250. NaN for ``NA``, ``H``, ``HEL`` and the like.

    Designators are magnetic and rounded to 10 degrees while METAR wind is
    true; the combined error (~10 degrees at most) is immaterial for a
    head-/cross-wind split.
    """
    digits = runway.astype("string").str.extract(r"^(\d{2})[LCR]?$", expand=False)
    return (pd.to_numeric(digits, errors="coerce") * 10.0).to_numpy(dtype="float64")


def wind_components(dir_deg: np.ndarray, kt: np.ndarray, heading_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(headwind, |crosswind|) in knots; a negative headwind is a tailwind.

    Variable wind (no direction) has no headwind component and is counted as
    all crosswind -- the conservative reading.
    """
    ang = np.deg2rad(dir_deg - heading_deg)
    variable = np.isnan(dir_deg)
    head = np.where(variable, kt * 0.0, kt * np.cos(ang))
    cross = np.where(variable, kt, np.abs(kt * np.sin(ang)))
    no_rwy = np.isnan(heading_deg)
    return np.where(no_rwy, np.nan, head), np.where(no_rwy, np.nan, cross)


# --- the METAR table and the judgements made on it -----------------------------

@functools.lru_cache(maxsize=2)
def _metar_by_station(path: str, mtime_ns: int) -> dict[str, pd.DataFrame]:
    df = pd.read_parquet(path)
    df["t"] = epoch_seconds(df["obs_time"]).to_numpy()
    return {str(s): g.reset_index(drop=True) for s, g in df.groupby("station", observed=True)}


def load_metar() -> dict[str, pd.DataFrame]:
    """The METAR table split by station, time-sorted, with epoch seconds in ``t``. Memoised."""
    p = paths.METAR_FILE
    return _metar_by_station(str(p), p.stat().st_mtime_ns)


def station_rows(apt: pd.Series) -> dict[str, np.ndarray]:
    """Row positions of each airport in ``apt``."""
    cat = pd.Categorical(apt)
    return {str(s): np.flatnonzero(cat.codes == k) for k, s in enumerate(cat.categories)}


def report_state(rep: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per-report values the features use, with the judgement thresholds applied.

    Elements a report does not mention get their good-weather value (RVR above
    2000 m is simply not reported); NaN is left only where an element was
    reported as unobservable, which is rare.

    * ``lvp`` -- low-visibility procedures: RVR < 600 m (or, with no RVR, the
      minimum visibility < 600 m) or a ceiling below 200 ft.
    * ``deice`` -- conditions that make de-icing likely: at most 3 C with
      precipitation, fog, mist or visibility under 1500 m; any freezing or ice
      precipitation; or frost weather (at most 0 C, dew point within 2 C).
    """

    def col(name: str) -> np.ndarray:
        return rep[name].to_numpy(dtype="float64")

    vis = np.minimum(col("vis_min_m"), VIS_CAP_M)
    rvr = col("rvr_min_m")
    ceil = np.minimum(col("ceiling_ft"), CEIL_CAP_FT)
    temp, dew = col("temp_c"), col("dewpt_c")
    precip = col("precip_int") > 0
    fog, mist = col("wx_fog") > 0, col("wx_mist") > 0
    freezing, ice = col("wx_freezing") > 0, col("wx_ice") > 0
    snowfrz = (col("wx_snow") > 0) | freezing | ice
    with np.errstate(invalid="ignore"):
        lvp = (rvr < 600) | (np.isnan(rvr) & (vis < 600)) | (ceil < 200)
        deice = ((temp <= 3) & (precip | fog | mist | (vis < 1500))) | freezing | ice | ((temp <= 0) & (temp - dew <= 2))
    return {
        "t": rep["t"].to_numpy(dtype="float64"),
        "wind_dir_deg": col("wind_dir_deg"),
        "wind_kt": col("wind_kt"),
        "gust_kt": np.fmax(col("gust_kt"), col("wind_kt")),
        "vis_m": vis,
        "rvr_min_m": np.where(np.isnan(rvr), RVR_CAP_M, np.minimum(rvr, RVR_CAP_M)),
        "ceiling_ft": ceil,
        "temp_c": temp,
        "dewpt_spread_c": temp - dew,
        "precip_int": col("precip_int"),
        "precip": precip.astype("float64"),
        "snowfrz": snowfrz.astype("float64"),
        "ts": ((col("wx_ts") > 0) | (col("cb_tcu") > 0)).astype("float64"),
        "fog": fog.astype("float64"),
        "lvp": lvp.astype("float64"),
        "deice": deice.astype("float64"),
    }
