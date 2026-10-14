"""Primitives of the long-horizon operational models: a prediction made hours before the planned off-block.

The operational ladder of `_oplib` predicts 15 minutes before the planned off-block (H1) or at the push (H0). These
primitives serve two earlier moments, a fixed lead before the planned off-block:

* **planned off-block** ``P`` = the published schedule ``SCHED_TIME_UTC_mvt`` (`planned_offblock_long`), on every
  row: every departure carries one, training and ranking. Never the IOBT (findings 10.35): on the NM-matched 2025
  departures it equals NM's last known off-block time on 99.45 %, it carries most of a late flight's delay (IOBT minus
  schedule: median 0 min, P90 25 min on the 2025 departures and 31 min on the Jan+Jul 2026 ranking departures; above
  15 min on 14.8 % and 17.7 %), and the data hold no filing time, so its value hours ahead is unknown. Never NM's last
  known off-block time either: its value is the latest update. (The IOBT-clock runs of findings 10.31 stay in runs/
  as the sensitivity "filed plan known", O*; this code no longer builds them.)
* **anchor** ``a = P - lead`` (`anchor_long`), with the leads of `LEADS_S`: 3 h and 24 h.

What is known at ``a``, and therefore read:

* plan data of every flight: the schedule, published months ahead. Which flights operate at all is taken from the
  data (cancellations are not in it): a mild hindsight shared by every schedule-based count;
* OTHER flights' recorded events (takeoff, landing, NM's actual off-block ``AOBT_3_flt``) that happened at or before
  ``a``; every window is ``[a - w, a]``, inclusive at both ends, in whole seconds (`_oplib.secs`), and the flight
  itself is removed from every count, sum and mean (`_oplib.backward_stats`);
* weather: the METAR in force at ``a - _oplib.METAR_LAG_S`` and the windows ending there (persistence), and the TAF
  issued at or before ``a`` whose validity covers ``P`` (`taf_for`; the TAF a planner holds at ``a`` for the hour of
  the flight).

Never read: the IOBT of any flight, the flight's own takeoff, its own NM actuals (``AOBT_3_flt`` is back-computed from
the takeoff at LTFM and partly at EDDM, LEBL, LSZH and LEMD, findings 10.25, so it is neither an anchor nor an input),
its own off-block record, NM's last known off-block time, and EOBT_1.

Lead-independent pieces (the calendar at ``P``, whether a stand is allocated, the schedule demand around ``P``) are
computed once and shared by both leads: they read plan data only, so their values are the same at 3 h and 24 h.

Operational tiers, declared column by column in `prc.features._long_tiers` (the configs and tests check them):

* the flight's own recorded runway is the realised outcome of a future runway configuration hours ahead (O* at best;
  the long deliverables, the ``*_rwyblind`` refit rungs, drop every column that reads it), its own recorded stand is
  O* (decided on the day), its pier a weak O*;
* the schedule demand per runway and per pier groups the OTHER flights, which move after ``a``, by their recorded
  runway and stand (their planned allocation assumed known at ``a``; the data hold the actual ones):
  `O_STAR_FUTURE_ATTRIBUTES_LONG`, the only columns the future-attribute censoring test (T2b) may move;
* tier C: whether a record joined NM depends on the flight's own actual off-block, so the missing level of the
  ``_flt`` columns of `aircraft_identity` marks the challenge's join; constant on the NM-joined OP population, where
  the long ladder is read. The schedule clock is the same on joined and unjoined rows, and these primitives read no
  field of the flight's own NM record.

Archive table (target-free, transductive in time): the NM proxy baseline (`NmBaselineLong`), the medians of NM's
taxi-out reading ``takeoff - AOBT_3`` per (airport, stand, runway) over the *fit* rows, leaving out the fit rows of
the query's own planned month, so that a fit row's own reading never enters what it reads. It is `_oplib.NmBaseline`
keyed by `plan_month` (the schedule's month) instead of `_oplib.baseline_month` (IOBT, else LOBT, else schedule).

Underscore module: not auto-imported, but hashed into every feature module that imports it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc.features import _oplib, _proxylib, weather, weather_fcst
from prc.features._wxlib import range_reduce, station_rows
from prc.features.congestion import window_count
from prc.features.registry import FeatureContext

HOUR = 3600.0
LEADS_S = {"hl3": 3 * HOUR, "hl24": 24 * HOUR}
"""The two long horizons: the anchor sits this long before the planned off-block."""


# --- the plan clock and the anchors ----------------------------------------------------------------

def planned_offblock_long(frame: pd.DataFrame) -> np.ndarray:
    """The planned off-block known hours ahead: the published schedule ``SCHED_TIME_UTC_mvt`` (whole epoch seconds).

    On every row, whatever the IOBT or NM's last known off-block time say: neither is read (findings 10.35).
    """
    return _oplib.secs(frame["SCHED_TIME_UTC_mvt"])


def anchor_long(frame: pd.DataFrame, lead_s: float) -> np.ndarray:
    """The long-horizon anchor: ``lead_s`` seconds before the planned off-block."""
    return planned_offblock_long(frame) - float(lead_s)


def plan_month(frame: pd.DataFrame) -> np.ndarray:
    """Calendar month (``12 * year + month - 1``, UTC) of the planned off-block (the schedule); NaN without one.

    The month a month-out table leaves out. Read from the schedule, never from the takeoff or a filed plan, so the key
    is known months before the movement and moves with neither the own takeoff nor the IOBT or NM's last known
    off-block time.
    """
    dt = pd.Series(pd.to_datetime(planned_offblock_long(frame), unit="s", utc=True))
    return (dt.dt.year * 12 + dt.dt.month - 1).to_numpy(dtype="float64", na_value=np.nan)


def masked(frame: pd.DataFrame, t: np.ndarray) -> pd.DataFrame:
    """Every column NaN where ``t`` (the anchor or the planned off-block) is missing."""
    return _oplib._mask(frame, np.asarray(t, dtype="float64"))


# --- the flight's own record and plan ---------------------------------------------------------------

STATIC = tuple(c for c in _oplib.STATIC if c != "day_of_year")
"""Calendar at the planned off-block in airport local time, and the scheduled local hour (`_oplib.static_at`), without
the day of year (findings 10.35: it maps 2025's moving holidays onto 2026 dates). On the schedule clock ``hour_local``
and ``sched_hour_local`` coincide; both are kept, as `_oplib.STATIC` has them."""


def static_long(q: pd.DataFrame, planned: np.ndarray) -> pd.DataFrame:
    """The calendar at the planned off-block ``P`` (not at the anchor: 24 h before, the weekday is another)."""
    return masked(_oplib.static_at(q, planned)[list(STATIC)], planned)


PLAN = ("has_stand",)


def plan_long(q: pd.DataFrame, planned: np.ndarray) -> pd.DataFrame:
    """Whether a stand is allocated: the own stand allocation (O* hours ahead; `prc.features._long_tiers`).

    No filed plan is read (findings 10.35): ``iobt_minus_sched_min``, an output until 11 Oct, was most of a late
    flight's delay, unknown hours ahead, and its NaN was the challenge's NM-join flag (C). No explicit join flag is an
    output either (``has_iobt`` and ``has_actype``, both C, were dropped after the review of 9 Oct).
    """
    frame = pd.DataFrame(
        {"has_stand": q["STAND_mvt"].notna().to_numpy(dtype="float64")},
        index=q.index,
    ).astype("float32")
    return masked(frame, planned)


# --- weather: persistence at the anchor and the TAF for the planned hour ----------------------------

TAF_LONG = (
    "taf_issue_age_h",
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
WX_LONG = (*weather.WX_NOW, *weather.WX_PAST, *TAF_LONG)
TAF_FIELDS = (
    ("taf_vis_m", "p_vis_m"),
    ("taf_ceiling_ft", "p_ceiling_ft"),
    ("taf_wind_kt", "p_wind_kt"),
    ("taf_snowfrz", "p_snowfrz"),
    ("taf_worst_vis_m", "w_vis_m"),
    ("taf_worst_ceiling_ft", "w_ceiling_ft"),
    ("taf_worst_snowfrz", "w_snowfrz"),
    ("taf_worst_ts", "w_ts"),
    ("taf_temp_prob", "tempo_prob"),
)


def taf_for(q: pd.DataFrame, a: np.ndarray, tau: np.ndarray) -> pd.DataFrame:
    """The TAF a planner holds at ``a`` for the time ``tau``: the latest issued at or before ``a`` covering ``tau``.

    Prevailing and worst-case state of the segment containing ``tau`` (`weather_fcst.select_taf`,
    `weather_fcst.segment_index`: FM replaces, BECMG applies at the end of its window; worst case adds active TEMPO
    and PROB groups and BECMG in transition), the worst case over the following hour of the same TAF, the TAF's age
    at ``a`` (hours since its issue) and the amendments issued in the 12 h up to ``a``. NaN where no TAF issued by
    ``a`` covers ``tau`` (a 24 h TAF does not reach a day ahead), or a cancellation is in force.
    """
    a = np.asarray(a, dtype="float64")
    tau = np.asarray(tau, dtype="float64")
    tafs = weather_fcst.load_tafs()
    cols = {c: np.full(len(q), np.nan) for c in TAF_LONG}
    for station, rows in station_rows(q["apt"]).items():
        tab = tafs.get(station)
        if tab is None or not len(rows):
            continue
        qa, qt = a[rows], tau[rows]
        live = np.isfinite(qa) & np.isfinite(qt)
        amd_t = tab["issue"][tab["amd"] > 0]
        n_amd = np.searchsorted(amd_t, qa, "right") - np.searchsorted(amd_t, qa - 12 * HOUR, "right")
        cols["taf_amd_12h"][rows] = np.where(live, n_amd, np.nan)
        pos = np.where(live, weather_fcst.select_taf(tab, qa, qt), -1)
        i = weather_fcst.segment_index(tab, pos, qt)
        ok = (i >= 0) & live
        r, j, p = rows[ok], i[ok], pos[ok]
        cols["taf_issue_age_h"][r] = (qa[ok] - tab["issue"][p]) / HOUR
        cols["taf_is_amd"][r] = tab["amd"][p]
        for out, key in TAF_FIELDS:
            cols[out][r] = tab[key][j]
        j_end = weather_fcst.segment_index(tab, p, np.minimum(qt[ok] + HOUR, tab["vto"][p] - 1))
        cols["taf_f60_worst_vis_m"][r] = range_reduce(tab["w_vis_m"], j, j_end + 1, "min")
        cols["taf_f60_worst_snowfrz"][r] = range_reduce(tab["w_snowfrz"], j, j_end + 1, "any")
    return pd.DataFrame({k: v.astype("float32") for k, v in cols.items()}, index=q.index)


def wx_long(q: pd.DataFrame, a: np.ndarray, planned: np.ndarray, params: dict | None = None) -> pd.DataFrame:
    """Weather known at the anchor: the METAR side at ``a - METAR_LAG_S``, the TAF issued by ``a`` for ``P``.

    `weather.wx_now` and `weather.wx_past` on a shadow frame whose takeoff column is ``a - METAR_LAG_S`` (the
    report in force there and the windows ending there: persistence; head- and crosswind on the own runway are O*),
    and `taf_for` with issue time at most ``a`` and validity time ``P``. Nothing of the flight but its airport,
    runway and stand reaches the weather blocks (`_oplib.shadow_departures`).
    """
    a = np.asarray(a, dtype="float64")
    ctx = FeatureContext(dep_index=q.index, params=dict(params or {}))
    metar_side = _oplib.shadow_departures(q, a - _oplib.METAR_LAG_S)
    frame = pd.concat([weather.wx_now(metar_side, ctx), weather.wx_past(metar_side, ctx), taf_for(q, a, planned)], axis=1)
    return masked(frame[list(WX_LONG)].astype("float32"), a)


# --- schedule demand around the planned off-block -----------------------------------------------------

DEMAND_WINDOWS = {"w30": 1800.0, "w60": 3600.0}
"""Half-widths of the windows ``[P - w, P + w]`` around the own planned off-block."""
DEMAND_REFS = ("dep_sched", "arr_sched")
"""Other departures and arrivals by their schedule; no IOBT (findings 10.35: until 11 Oct ``dep_iobt`` counted the
other departures by their IOBT, assuming their plans filed by the anchor)."""
DEMAND_LEVELS = ("apt", "rwy", "pier")
DEMAND = tuple(f"dem_{ref}_{lvl}_{w}" for ref in DEMAND_REFS for lvl in DEMAND_LEVELS for w in DEMAND_WINDOWS)
O_STAR_FUTURE_ATTRIBUTES_LONG = tuple(c for c in DEMAND if "_rwy_" in c or "_pier_" in c)
"""Demand columns that group other flights by their recorded runway or stand (tier O*: planned allocation assumed)."""


def symmetric_count(ref_t, ref_g, q_t, q_g, half_width: float, own_t=None) -> np.ndarray:
    """Reference events in ``[t - half_width, t + half_width]`` of each query's group, the query's own event excluded.

    ``own_t`` is the query's own time in the reference table (NaN where it has none). Whole seconds, inclusive at
    both ends, so an own event exactly on an edge is excluded exactly when it was counted.
    """
    q = np.asarray(q_t, dtype="float64")
    n = window_count(
        _oplib._series(ref_t), _oplib._series(ref_g), _oplib._series(q), _oplib._series(q_g), back=half_width, fwd=half_width
    ).astype("int64")
    own = _oplib._own_in_window(q, _oplib._query_ok(q, q_g), own_t, half_width, half_width)
    return n - own.astype("int64")


def _groups(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "apt": frame["apt"].astype("string").to_numpy(dtype=object),
        "rwy": _oplib.runway_key(frame).to_numpy(dtype=object),
        "pier": _oplib.pier_key(frame).to_numpy(dtype=object),
    }


def _group_codes(*tables: dict[str, np.ndarray]) -> list[dict[str, np.ndarray]]:
    """The group keys of several tables as shared numeric codes (NaN where missing): `window_count` sorts and
    categorises numbers several times faster than millions of strings, and equal keys get equal codes across tables."""
    out: list[dict[str, np.ndarray]] = [{} for _ in tables]
    for lvl in tables[0]:
        codes = pd.factorize(np.concatenate([t[lvl] for t in tables]), use_na_sentinel=True)[0].astype("float64")
        codes[codes < 0] = np.nan
        start = 0
        for i, t in enumerate(tables):
            out[i][lvl] = codes[start:start + len(t[lvl])]
            start += len(t[lvl])
    return out


def departure_positions(dep: pd.DataFrame, q_index: pd.Index) -> np.ndarray:
    """Positions of the query departures in ``dep`` (every query must be one of its departures)."""
    pos = dep.index.get_indexer(q_index)
    if (pos < 0).any():
        raise ValueError("every query must be a departure of the frame")
    return pos


def demand_at(mvt: pd.DataFrame, q_index: pd.Index, planned: np.ndarray) -> pd.DataFrame:
    """Other flights planned around the own planned off-block ``P``: plan data only, the flight itself excluded.

    For each window ``[P - w, P + w]`` (w = 30, 60 min) and each grouping (the airport; the own runway; the own pier),
    the number of other departures whose schedule lies in it (``dep_sched``) and of arrivals whose schedule (in-block)
    lies in it (``arr_sched``); no IOBT is read. Runway and pier group the other flights by their recorded runway and
    stand (O*, `O_STAR_FUTURE_ATTRIBUTES_LONG`). The queries' groups and own times are read from the departure table
    by position (the queries are departures).
    """
    dep = _oplib.departures(mvt)
    arr = _oplib.arrivals(mvt)
    pos = departure_positions(dep, q_index)
    planned = np.asarray(planned, dtype="float64")
    dep_groups, arr_groups = _group_codes(_groups(dep), _groups(arr))
    q_groups = {k: v[pos] for k, v in dep_groups.items()}
    refs = {
        "dep_sched": (dep_groups, _oplib.secs(dep["SCHED_TIME_UTC_mvt"]), True),
        "arr_sched": (arr_groups, _oplib.secs(arr["SCHED_TIME_UTC_mvt"]), False),
    }
    out: dict[str, np.ndarray] = {}
    for ref, (groups, t, has_own) in refs.items():
        own_t = t[pos] if has_own else None
        for lvl in DEMAND_LEVELS:
            for w, half in DEMAND_WINDOWS.items():
                out[f"dem_{ref}_{lvl}_{w}"] = symmetric_count(t, groups[lvl], planned, q_groups[lvl], half, own_t)
    frame = pd.DataFrame(out, index=q_index)[list(DEMAND)].astype("float32")
    return masked(frame, planned)


# --- recent history: NM's readings of flights already airborne at the anchor ----------------------------

HIST_WINDOWS = {"b3h": 3 * HOUR, "b24h": 24 * HOUR}
HIST_LEVELS = ("apt", "rwy")
HIST_STATS = ("n", "anom_mean", "anom_share5", "push_delay_mean")
HIST = tuple(f"hist_{lvl}_{win}_{stat}" for lvl in HIST_LEVELS for win in HIST_WINDOWS for stat in HIST_STATS)

COVERAGE_LOOKBACK_S = 24 * HOUR
"""A history window counts as covered by the frame when another movement at the airport took place in the 24 h before
its start: longer than any stretch without movements inside the data (Heathrow's closure of 21 Mar 2025, 19.1 h; the
night curfews, 6-8 h), far shorter than the gaps between the frame's periods (2025, January 2026, July 2026)."""


def window_covered(mvt: pd.DataFrame, q_index: pd.Index, start: np.ndarray, lookback: float = COVERAGE_LOOKBACK_S) -> np.ndarray:
    """Whether the frame's records reach back to each query's window start; where not, the window is unknown (NaN).

    True where another movement at the query's airport (a takeoff or a landing; the query's own takeoff excluded)
    lies in ``[start - lookback, start]``. False where the window starts where the frame holds no records: before its
    first day (1-2 Jan 2025 reaching into December 2024) or between its periods (the first days of July 2026 reaching
    into June, which the frame does not hold), where an empty window would otherwise read as "no traffic" (n = 0).
    Reads only events at or before ``start``, so nothing after the anchor enters and the own takeoff never decides it
    (T1, T2). Conservative at the very start of a period (a window starting in the night before the first movement
    is unknown too); a gap inside a window is not detected (the frame has none within a period).
    """
    t = _oplib.secs(mvt["MVT_TIME_UTC_mvt"])
    pos = mvt.index.get_indexer(q_index)
    if (pos < 0).any():
        raise ValueError("window_covered: every query must be a movement of the frame")
    codes = pd.factorize(mvt["apt"], use_na_sentinel=True)[0].astype("float64")
    codes[codes < 0] = np.nan
    n = _oplib.backward_count(t, codes, np.asarray(start, dtype="float64"), codes[pos], lookback, t[pos])
    return np.asarray(n) > 0


class NmBaselineLong(_oplib.NmBaseline):
    """`_oplib.NmBaseline` keyed by `plan_month` (the schedule's month) instead of `_oplib.baseline_month`.

    The medians come from the fit rows only; `anomaly(m)` leaves out the fit readings whose planned month is ``m``,
    and a query reads the table of its own planned month (`key`), so its own reading never enters a median that
    defines the anomalies it averages. Queries of a month without fit readings (holdout, 2026) read the full table.
    The constructor is the parent's with the month replaced, written out so that neither the IOBT nor NM's last known
    off-block time (which the parent's month reads: IOBT, else LOBT, else schedule) is ever touched; `anomaly` and
    `key` are inherited.
    """

    def __init__(self, dep: pd.DataFrame, fit: np.ndarray, min_support: int = _proxylib.MIN_SUPPORT):
        self.index = dep.index
        self.proxy = _proxylib.proxy_seconds(dep).to_numpy(dtype="float64")
        apt = dep["apt"].astype("string")
        stand = dep["STAND_mvt"].astype("string").fillna("__NA__")
        rwy = dep["RUNWAY_mvt"].astype("string").fillna("__NA__")
        keys = (apt + "|" + stand + "|" + rwy, apt + "|" + stand, apt + "|" + rwy, apt)
        self.codes = [pd.factorize(k.to_numpy(dtype=object), use_na_sentinel=True)[0] for k in keys]
        self.month = plan_month(dep)
        self.use = np.isfinite(self.proxy) & np.asarray(fit, dtype=bool)
        self.min_support = int(min_support)
        self.fit_months = frozenset(int(m) for m in np.unique(self.month[self.use & np.isfinite(self.month)]))
        self._tables = {}


def nm_baseline_long(dep: pd.DataFrame, fit: np.ndarray, min_support: int = _proxylib.MIN_SUPPORT) -> NmBaselineLong:
    """The long-horizon NM proxy baseline over the fit rows (the one entry point the blocks use; T2 freezes it)."""
    return NmBaselineLong(dep, fit, min_support)


def month_keys(months: np.ndarray, fit_months) -> np.ndarray:
    """The table key of each query as an integer: its planned month when that month holds fit rows, else -1.

    The vectorised `_oplib.NmBaseline.key` (``-1`` for ``None``: the full-fit table).
    """
    months = np.asarray(months, dtype="float64")
    fit = np.array(sorted(fit_months), dtype="float64")
    ok = np.isfinite(months) & np.isin(months, fit)
    return np.where(ok, np.nan_to_num(months, nan=-1.0), -1.0).astype("int64")


def history_at(mvt: pd.DataFrame, q_index: pd.Index, a: np.ndarray, baseline: NmBaselineLong) -> pd.DataFrame:
    """NM readings of OTHER flights that took off in the 3 h / 24 h up to the anchor, per airport and own runway.

    For the others with a takeoff in ``[a - w, a]``: how many carry a plausible NM reading (``n``), the mean and the
    share above +5 min of their proxy anomaly (``takeoff - AOBT_3`` against the month-out fit median of
    ``baseline``), and their mean push delay ``AOBT_3`` minus their schedule (`planned_offblock_long`). A reading
    is plausible when ``0 <= takeoff - AOBT_3 <= 2 h`` (`_proxylib.proxy_seconds`), so ``AOBT_3 <= takeoff <= a``:
    nothing after the anchor enters. A window the frame's records do not reach back to (`window_covered`: it starts
    before the frame's first day or in a gap between its periods) is NaN in all its columns, not "no traffic". The
    own runway is O* (planned runway assumed known); the others have taken off, so their runway is a recorded fact.
    """
    dep = _oplib.departures(mvt)
    if not baseline.index.equals(dep.index):
        raise ValueError("history_at: the baseline must be built on this frame's departures")
    pos = departure_positions(dep, q_index)
    a = np.asarray(a, dtype="float64")
    t = _oplib.secs(dep["MVT_TIME_UTC_mvt"])
    push_raw = np.clip(_oplib.secs(dep["AOBT_3_flt"]) - planned_offblock_long(dep), *_oplib.PUSH_DELAY_CLIP)

    def own(values: np.ndarray) -> np.ndarray:
        return values[pos]

    (codes,) = _group_codes({"apt": dep["apt"].astype("string").to_numpy(dtype=object),
                             "rwy": _oplib.runway_key(dep).to_numpy(dtype=object)})
    groups = {lvl: (c, c[pos]) for lvl, c in codes.items()}
    out = {c: np.full(len(q_index), np.nan) for c in HIST}
    own_t = own(t)
    keys = month_keys(baseline.month[pos], baseline.fit_months)  # the queries keyed exactly as the baseline's records
    widest = max(HIST_WINDOWS.values())
    for key in np.unique(keys):
        rows = np.flatnonzero((keys == key) & np.isfinite(a))
        if not len(rows):
            continue
        an = baseline.anomaly(None if key < 0 else int(key))
        push = np.where(np.isfinite(an), push_raw, np.nan)
        qa = a[rows]
        with np.errstate(invalid="ignore"):  # only the events that can sit in one of these queries' windows
            sel = (t >= qa.min() - widest) & (t <= qa.max())
        for lvl in HIST_LEVELS:
            rg, qg = groups[lvl]
            for win, back in HIST_WINDOWS.items():
                n, mean, share = _oplib.backward_stats(
                    t[sel], rg[sel], an[sel], qa, qg[rows], back, own_t[rows], own(an)[rows], high=_oplib.HIGH_ANOMALY_S
                )
                _, push_mean, _ = _oplib.backward_stats(t[sel], rg[sel], push[sel], qa, qg[rows], back, own_t[rows], own(push)[rows])
                out[f"hist_{lvl}_{win}_n"][rows] = n
                out[f"hist_{lvl}_{win}_anom_mean"][rows] = mean
                out[f"hist_{lvl}_{win}_anom_share5"][rows] = share
                out[f"hist_{lvl}_{win}_push_delay_mean"][rows] = push_mean
    for win, back in HIST_WINDOWS.items():  # windows reaching before the frame's records: unknown, not empty
        unknown = ~window_covered(mvt, q_index, a - back)
        for lvl in HIST_LEVELS:
            for stat in HIST_STATS:
                out[f"hist_{lvl}_{win}_{stat}"][unknown] = np.nan
    frame = pd.DataFrame(out, index=q_index)[list(HIST)].astype("float32")
    return masked(frame, a)
