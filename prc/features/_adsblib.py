"""Primitives of the ADS-B surface blocks (`prc.features.operational_adsb`): the event table, the own-airframe exclusion,
windows, the coverage gates and the training support.

Underscore module: not auto-imported, but hashed into `prc.features.operational_adsb`, which imports it as a module.
Feature code never imports `prc.adsb` (the builder, `prc adsb-surface`); it reads only the consolidated event table of
a snapshot (`paths.ADSB_SURFACE_EVENTS_FILE`) and its support table (`paths.ADSB_SURFACE_SUPPORT_FILE`), which the
blocks declare as external inputs, so their content digests key the cache. The builder imports `norm_callsign` from
here (one normalisation for the events' callsigns and the flights' own): change it only with a builder version bump.

**Clock.** Whole seconds. The query time is ``q = floor(a) - ADSB_LAG_S`` (the feed's latency). Windows are
``[q - w, q]``, inclusive at both ends; an event counts in a window when its *knowable* time lies in it. An interval
is in progress at ``q`` when ``t_known_start <= q < stop`` with ``stop = min(t_known_end, t_start + cap)``: an end not
yet knowable at ``q`` counts as not ended (so deleting everything after ``q`` changes nothing), up to the cap.

**The own airframe** is removed at event level, by attributes fixed when each event became knowable (critique F4):
every event whose callsign equals the flight's ``CALLSIGN_flt``, and every event whose origin (push point, the stop it
left, or its first sighting) maps to a stand within ``NEIGHBOUR_M`` of the flight's own stand. Deleting or shifting
the own airframe's points therefore removes only events that were excluded anyway, and the stand's previous occupant
is always excluded, not sometimes. Counts and sums are exact by inclusion-exclusion over four key families
(group; group+callsign; group+origin stand; group+callsign+origin stand), values rounded to whole seconds first.
Arrivals' events (landings, taxi-ins) are other flights: never excluded (as ``op_h1_traffic``'s ``arr_taxiing``).

**Coverage, per column family** (critique F5/F6, review): every family is NaN, never zero, where its reception is not
established, so missing reception never reads as quiet traffic.

* base (``cov_ok``): ADS-B takeoffs + landings per reference movement of other flights over 120 min (the airport's
  records, or with ``coverage_ref: scheduled`` the public schedule; at least ``min_ref`` of them), and no outage: in
  the last 30 or 10 minutes ADS-B saw almost none of the reference movements, or the airport's surface activity (the
  other aircraft's event starts, each of which needs a fresh point) in the last ``ACT_RECENT_S`` fell below
  ``act_max_ratio`` of its mean over the ``ACT_BASE_S`` before (a feed loss shows within minutes at a busy airport; the
  heartbeat is own-excluded like every count, so the own airframe's points cannot move it);
* taxi (``TAXI``: taxiing out and in, remote stops, taxi-out timings): the share of ADS-B takeoffs whose taxi was
  received from well before the roll up to the runway (``taxi_seen``; the counts' completeness) at least ``min_taxi``;
* hold (``HOLD``: the runway queue): the taxi gate; ``rwy_own_hold_now`` also needs the own runway to be a runway end
  of the airport and the airport's holds to agree with the takeoffs that follow them (below);
* stand (``STAND``: pushes, push-origin taxi times): the share of ADS-B takeoffs whose taxi-out began at a detected
  push over the last ``PUSH_SHARE_S`` (6 h: a stable airport-level reading, not the 2-hour fleet mix) at least
  ``min_push``;
* training support (``support: true``): a family stays only at an airport where its ADS-B-only gate held on at least
  ``min_support`` of the 2025 training departures, and where the row's scale ratio (taxi-out starts and runway holds per
  ADS-B takeoff, the push share) lies in that airport's training range (1st-99th percentile, widened by
  ``support_margin``): a reception regime the model never saw at that airport (EGLL's push reception in 2026, CDG's
  ground coverage) is not extrapolated (`support_table`, frozen per snapshot by ``prc adsb-surface --support``). The
  same table carries the runway attribution of holds: a hold label (the nearest runway end) that the airframes'
  next takeoffs systematically contradict on the training days is relabelled to their modal runway end, and
  ``rwy_own_hold_now`` is NaN at an airport whose relabelled holds still agree with the next takeoff on fewer than
  ``min_hold_agree`` of them.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from prc import paths

ADSB_LAG_S = 30
"""ADS-B is read this long before the anchor (feed latency)."""
CAP_TAXI_OUT_S = 3600
CAP_TAXI_IN_S = 1800
CAP_HOLD_S = 1800
CAP_REMOTE_S = 3600
TAXI_DUR_CAP_S = 3600
NEIGHBOUR_M = 100.0
"""Events whose origin stand lies within this of the flight's own stand are the flight's own (or its stand's previous
occupant's): excluded."""
PUSH_SHARE_S = 6 * 3600
"""The stand gate's push share is read over this window (stable per airport; a 2-hour share follows the fleet mix)."""
EVENTS_MARGIN_S = 7 * 3600
"""Events known more than this before the earliest query are not loaded (the longest window is ``PUSH_SHARE_S``)."""
ACT_RECENT_S = 300
ACT_BASE_S = 7200
"""The activity heartbeat: event starts in ``[q - ACT_RECENT_S, q]`` against their mean per ``ACT_RECENT_S`` over the
rest of ``[q - ACT_BASE_S, q]``."""
ACT_DEP_KINDS = ("TAKEOFF", "TAXI_OUT", "PUSH", "RWY_HOLD", "REMOTE_STOP")
"""Event kinds of the heartbeat that may be the own airframe's (own-excluded); landings are other flights'."""
SUPPORT_QUANTILES = (0.01, 0.99)
HOLD_MAP_MIN_N = 30
HOLD_MAP_MIN_SHARE = 0.6
"""A hold label is relabelled only when at least this many of its holds were followed by a takeoff of the same airframe
and this share of those took off from one other runway end."""
DEFAULTS = {
    "shift_days": 0,            # placebo (critique F14): read the events of `shift_days` (>= 0) days earlier
    "coverage_ref": "recorded",  # reference movements: "recorded" (airport records, tier O) or "scheduled" (public)
    "min_ref": 6,               # reference movements in 120 min before the ratio is read (curfew mornings: NaN)
    "min_tko": 3,               # ADS-B takeoffs (120 min; 6 h for the push share) before their shares are read
    "min_ratio": 0.6,           # ADS-B takeoffs + landings per reference movement over 120 min (covered airports q10
                                # 0.68-0.91 on sampled days; F5 asked 0.8: a param for the sensitivity)
    "min_taxi": 0.75,           # share of ADS-B takeoffs with taxi_seen (taxi gate; covered July 2025 days: 0.84-0.99
                                # per airport, uncovered 0.22-0.46)
    "min_push": 0.15,           # share of ADS-B takeoffs whose taxi-out began at a detected push, 6 h (stand gate)
    "outage_min_ref": 4,        # an outage: at least this many reference movements in 30 min ...
    "outage_max_ratio": 0.3,    # ... and ADS-B saw fewer than this share of them
    "outage10_min_ref": 6,      # the same over 10 min (a sudden feed loss shows within ~7 min at a hub)
    "outage10_max_ratio": 0.25,
    "act_min": 8,               # heartbeat: the mean event starts per 5 min before, for the drop to be read ...
    "act_max_ratio": 0.2,       # ... and an outage when the last 5 min had fewer than this share of that mean
    "support": True,            # apply the snapshot's training support table
    "min_support": 0.2,         # share of an airport's 2025 departures with the family's gate, for the family to stay
    "support_margin": 1.25,     # the training range of a ratio, widened by this factor at both ends
    "min_hold_agree": 0.9,      # rwy_own_hold_now: holds agreeing with the airframe's next takeoff runway, training days
}

COVERAGE = ("cov_tko_n_120", "cov_ref_n_120", "cov_mov_ratio_120", "cov_mov_ratio_30", "cov_mov_ratio_10",
            "cov_act_ratio_5", "cov_taxi_share_120", "cov_push_share_360", "cov_taxi_ratio_120", "cov_hold_ratio_120",
            "cov_ok", "cov_taxi_ok", "cov_hold_ok", "cov_stand_ok")
"""Coverage: gates and strata only. Every rung drops them from the model matrix (they would identify reception eras)."""
TAXI = ("taxi_out_now", "taxi_out_now_norm", "taxi_in_now", "taxi_out_start_15", "remote_now", "remote_60",
        "taxi_out_elapsed_mean_now", "taxi_stop_s_mean_60", "taxi_stop_share_60", "remote_dur_mean_60")
HOLD = ("rwy_hold_now", "rwy_own_hold_now", "rwy_hold_dur_mean_30")
SURFACE = TAXI + HOLD
STAND = ("push_15", "push_15_norm", "taxi_dur_mean_30", "taxi_dur_n_30", "taxi_dur_mean_60", "taxi_dur_n_60")
FAMILIES = {"taxi": TAXI, "hold": HOLD, "stand": STAND}
COLUMNS = COVERAGE + SURFACE + STAND
TIME_COLUMNS = ("t_start", "t_known_start", "t_end", "t_known_end")
READ_COLUMNS = ("apt", "kind", "icao", "cs", *TIME_COLUMNS, "origin_kind", "origin_stand", "rwy_end", "dur", "stop_s",
                "taxi_seen")
SUPPORT_COLUMNS = ("apt", "family", "n", "share", "lo", "hi", "label", "target")
"""Families ``taxi`` / ``hold`` / ``stand`` (gated share and ratio range); ``hold_map`` (a hold ``label`` relabelled to
``target``, ``share`` of its holds followed by a takeoff from it); ``own_hold`` (the share of holds, relabelled, whose
runway end is the airframe's next takeoff's)."""
INTERNALS = ("apt", "taxi_rx", "stand_rx", "r_taxi", "r_hold", "r_stand")
"""What `surface_at(..., return_internals=True)` adds for the support table: the ADS-B-only family gates (no reference
movements: the support is public-data only) and the scale ratios."""


# --- inputs ----------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=4)
def _read_events(path: str, mtime_ns: int, lo: float | None, hi: float | None) -> pd.DataFrame:
    filters = None
    if lo is not None and hi is not None:
        filters = [("t_known_start", ">=", lo), ("t_known_start", "<=", hi)]
    frame = pd.read_parquet(path, columns=list(READ_COLUMNS), filters=filters)
    for c in ("apt", "kind", "icao", "cs", "origin_kind", "rwy_end"):
        frame[c] = frame[c].astype("category")
    return frame.reset_index(drop=True)


def load_events(t_min: float | None = None, t_max: float | None = None) -> pd.DataFrame:
    """The snapshot's consolidated event table, optionally only events known in ``[t_min, t_max]`` (memoised on mtime).

    Callers must not mutate the frame. Times are epoch seconds; ``icao``, ``cs`` and the labels are categoricals.
    """
    path = paths.ADSB_SURFACE_EVENTS_FILE
    lo = None if t_min is None or not np.isfinite(t_min) else float(np.floor(t_min / 86_400) * 86_400)
    hi = None if t_max is None or not np.isfinite(t_max) else float(np.ceil(t_max / 86_400) * 86_400)
    return _read_events(str(path), path.stat().st_mtime_ns, lo, hi)


@functools.lru_cache(maxsize=2)
def _read_support(path: str, mtime_ns: int) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = set(SUPPORT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: support table lacks {sorted(missing)} (rebuild it: prc adsb-surface --support)")
    return frame.assign(apt=frame["apt"].astype(str), family=frame["family"].astype(str))


def load_support() -> pd.DataFrame:
    """The snapshot's frozen airport x family support table (``prc adsb-surface --support``)."""
    path = paths.ADSB_SURFACE_SUPPORT_FILE
    return _read_support(str(path), path.stat().st_mtime_ns)


def load_stand_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(stands, stand_matches) of the geometry tables: the own stand and its neighbours."""
    from prc.features import _geolib

    return _geolib.load_stands(), _geolib.load_stand_matches()


def load_runway_keys() -> set[str]:
    """``apt|ident`` of every valid runway end of the geometry tables (``dropped_reason`` '' is valid, F17a)."""
    from prc.features import _geolib

    e = _geolib.load_runway_ends()
    e = e.loc[e["dropped_reason"].fillna("").eq("") & e["lat"].notna()]
    out = set()
    for apt, ident in zip(e["apt"].astype(str), e["ident"]):
        norm = _geolib.norm_runway_ident(ident)
        if norm:
            out.add(f"{apt}|{norm}")
    return out


def norm_callsign(values) -> np.ndarray:
    """Upper case, letters and digits only (as `prc.adsb.events._norm_callsign`); None for a missing or empty one."""
    s = pd.Series(values, dtype="object").astype("string").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    s = s.mask(s.eq(""))
    return s.astype(object).where(s.notna(), None).to_numpy(dtype=object)


def shifted(events: pd.DataFrame, shift_days: int) -> pd.DataFrame:
    """The events moved ``shift_days`` days later (the shifted placebo reads the events of that many days before)."""
    if int(shift_days) < 0:
        raise ValueError(f"shift_days must be >= 0 (a negative shift reads the future), got {shift_days}")
    if not shift_days:
        return events
    out = events.copy()
    for c in TIME_COLUMNS:
        out[c] = out[c] + 86_400.0 * shift_days
    return out


def check_params(params: dict | None) -> dict:
    prm = {**DEFAULTS, **(params or {})}
    unknown = set(prm) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown op_*_adsb params {sorted(unknown)}; known: {sorted(DEFAULTS)}")
    if int(prm["shift_days"]) != prm["shift_days"] or int(prm["shift_days"]) < 0:
        raise ValueError(f"op_*_adsb shift_days must be a whole number >= 0 (the placebo reads past days), got "
                         f"{prm['shift_days']!r}")
    if prm["coverage_ref"] not in ("recorded", "scheduled"):
        raise ValueError(f"coverage_ref must be 'recorded' or 'scheduled', got {prm['coverage_ref']!r}")
    return prm


# --- own-airframe exclusion --------------------------------------------------------------------------------

def _recode(cat: pd.Series, categories: pd.Index) -> np.ndarray:
    """A categorical column's codes re-expressed in ``categories`` (-1 for missing or unknown), without per-row strings."""
    lookup = categories.get_indexer(cat.cat.categories.astype(str))
    codes = cat.cat.codes.to_numpy(dtype="int64")
    return np.where(codes >= 0, lookup[np.maximum(codes, 0)] if len(lookup) else -1, -1).astype("int64")


def _codes(values, categories) -> np.ndarray:
    """Codes of ``values`` in ``categories`` (unique; -1 for missing or unknown values), as int64."""
    cats = categories if isinstance(categories, pd.Index) else pd.Index(categories)
    return cats.get_indexer(pd.Index(np.asarray(values, dtype=object) if not np.asarray(values).dtype.kind == "f" else np.asarray(values))).astype("int64")


class Queries:
    """The query rows of one statistic family: their time, group, own callsign and own stand's neighbours.

    ``group``/``cs``/``stands`` are integer codes in code spaces shared with the events (-1 = missing):
    ``n_group``, ``n_cs``, ``n_stand`` are the sizes. ``pairs`` expands every query over its neighbour stands.
    """

    def __init__(self, q_t, group, cs, nb_rows, nb_stand, n_group: int, n_cs: int, n_stand: int):
        self.t = np.asarray(q_t, dtype="float64")
        self.group = np.asarray(group, dtype="int64")
        self.cs = np.asarray(cs, dtype="int64")
        self.n = len(self.t)
        self.n_group, self.n_cs, self.n_stand = int(n_group), max(int(n_cs), 1), max(int(n_stand), 1)
        ok = np.isfinite(self.t) & (self.group >= 0)
        self.ok = ok
        nb_rows = np.asarray(nb_rows, dtype="int64")
        nb_stand = np.asarray(nb_stand, dtype="int64")
        keep = ok[nb_rows] & (nb_stand >= 0) if len(nb_rows) else np.zeros(0, dtype=bool)
        self.pair_rows, self.pair_stand = nb_rows[keep], nb_stand[keep]
        has_cs = ok & (self.cs >= 0)
        g, c = self.group, self.cs
        pr, ps = self.pair_rows, self.pair_stand
        pc = has_cs[pr]
        # the four key families: (query index, key)
        self.families = {
            "all": (np.flatnonzero(ok), g[ok]),
            "cs": (np.flatnonzero(has_cs), g[has_cs] * self.n_cs + c[has_cs]),
            "stand": (pr, g[pr] * self.n_stand + ps),
            "both": (pr[pc], (g[pr[pc]] * self.n_cs + c[pr[pc]]) * self.n_stand + ps[pc]),
        }
        self.sign = {"all": 1, "cs": -1, "stand": -1, "both": 1}
        self._uniq = {f: np.unique(k) for f, (_, k) in self.families.items()}

    def ref_keys(self, family: str, group, cs, stand) -> np.ndarray:
        g = np.asarray(group, dtype="int64")
        c = np.asarray(cs, dtype="int64")
        s = np.asarray(stand, dtype="int64")
        if family == "all":
            k, ok = g, g >= 0
        elif family == "cs":
            k, ok = g * self.n_cs + c, (g >= 0) & (c >= 0)
        elif family == "stand":
            k, ok = g * self.n_stand + s, (g >= 0) & (s >= 0)
        else:
            k, ok = (g * self.n_cs + c) * self.n_stand + s, (g >= 0) & (c >= 0) & (s >= 0)
        return np.where(ok, k, -1)


def _keyed_stats(ref_code: np.ndarray, ref_t: np.ndarray, values: dict, q_code: np.ndarray, q_t: np.ndarray,
                 backs: list) -> dict:
    """For refs keyed by dense codes: per query, the count (and value sums) of refs of its code with ``t`` in
    ``[q - back, q]`` for each ``back`` (``None``: every ref at or before ``q``). Whole seconds; exact integer sums."""
    out = {}
    finite = np.isfinite(ref_t) & (ref_code >= 0)
    rc, rt = ref_code[finite], ref_t[finite]
    vals = {k: np.asarray(v)[finite] for k, v in values.items()}
    q_ok = np.isfinite(q_t) & (q_code >= 0)
    if not len(rc) or not q_ok.any():
        for b in backs:
            out[(b, "n")] = np.zeros(len(q_t), dtype="int64")
            for k in values:
                out[(b, k)] = np.zeros(len(q_t), dtype="int64")
        return out
    t0 = min(rt.min(), np.nanmin(q_t[q_ok])) - max([b for b in backs if b is not None] or [0]) - 2
    span = int(max(rt.max(), np.nanmax(q_t[q_ok])) - t0) + 2
    key = rc.astype("int64") * span + (rt - t0).astype("int64")
    order = np.argsort(key, kind="stable")
    key = key[order]
    csum = {k: np.concatenate([[0], np.cumsum(np.round(v[order]).astype("int64"))]) for k, v in vals.items()}
    qc = np.where(q_ok, q_code, 0).astype("int64")
    qt = np.where(q_ok, q_t - t0, 0).astype("int64")
    hi = np.searchsorted(key, qc * span + qt, side="right")
    for b in backs:
        lo_t = np.zeros_like(qt) if b is None else np.maximum(qt - int(b), 0)
        lo = np.searchsorted(key, qc * span + lo_t, side="left")
        lo = np.minimum(lo, hi)
        out[(b, "n")] = np.where(q_ok, hi - lo, 0).astype("int64")
        for k, c in csum.items():
            out[(b, k)] = np.where(q_ok, c[hi] - c[lo], 0).astype("int64")
    return out


def others(q: Queries, ref_t, group, cs, stand, backs: list, values: dict | None = None, exclude: bool = True) -> dict:
    """Own-excluded counts and sums of reference events per query (see `_keyed_stats`), by inclusion-exclusion.

    ``group``/``cs``/``stand``: the events' codes in the query code spaces. Returns ``{(back, "n"|value): array}``.
    """
    values = values or {}
    ref_t = np.asarray(ref_t, dtype="float64")
    total = {(b, k): np.zeros(q.n, dtype="int64") for b in backs for k in ("n", *values)}
    for fam in (("all", "cs", "stand", "both") if exclude else ("all",)):
        rows, qkey = q.families[fam]
        if not len(rows):
            continue
        uniq = q._uniq[fam]
        rkey = q.ref_keys(fam, group, cs, stand)
        pos = np.searchsorted(uniq, rkey)
        pos_c = np.minimum(pos, len(uniq) - 1)
        rcode = np.where((rkey >= 0) & (uniq[pos_c] == rkey), pos_c, -1)
        if not (rcode >= 0).any():
            continue
        qcode = np.searchsorted(uniq, qkey)
        stats = _keyed_stats(rcode, ref_t, values, qcode, q.t[rows], backs)
        for key_, arr in stats.items():
            total[key_] += q.sign[fam] * np.bincount(rows, weights=arr, minlength=q.n).astype("int64")
    return total


# --- the surface picture at the query times -------------------------------------------------------------

def _in_progress_stop(ev: pd.DataFrame, cap: float) -> np.ndarray:
    ts = ev["t_start"].to_numpy(dtype="float64")
    tk = ev["t_known_start"].to_numpy(dtype="float64")
    end = np.fmin(ev["t_known_end"].to_numpy(dtype="float64"), ts + cap)
    return np.maximum(tk, end)


def neighbour_pairs(apt: np.ndarray, stand_mvt: np.ndarray, stands: pd.DataFrame, matches: pd.DataFrame,
                    stand_categories: pd.Index) -> tuple[np.ndarray, np.ndarray]:
    """(query row, neighbour stand code) for every query whose STAND_mvt is matched: the stands within NEIGHBOUR_M
    of its own stand (itself included), as codes in ``stand_categories``. Vectorised (CSR expansion)."""
    from scipy.spatial import cKDTree

    m = matches.loc[matches["match_source"].astype(str).ne("none")]
    lookup = pd.Series(m["stand_id"].to_numpy(dtype="float64"),
                       index=(m["apt"].astype(str) + "|" + m["stand_mvt"].astype(str)).to_numpy())
    lookup = lookup[~lookup.index.duplicated()]
    q_apt = pd.Series(apt, dtype="object").astype(str)
    key = q_apt + "|" + pd.Series(stand_mvt, dtype="object").astype(str)
    own = key.map(lookup).to_numpy(dtype="float64")
    q_apt = q_apt.to_numpy(dtype=object)
    rows_out, codes_out = [], []
    st = stands[["apt", "stand_id", "east_m", "north_m"]].copy()
    st["apt"] = st["apt"].astype(str)
    for a, g in st.groupby("apt", sort=False):
        q_rows = np.flatnonzero((q_apt == a) & np.isfinite(own))
        if not len(q_rows):
            continue
        ids = g["stand_id"].to_numpy(dtype="int64")
        nb = cKDTree(g[["east_m", "north_m"]].to_numpy(dtype="float64")).query_ball_point(
            g[["east_m", "north_m"]].to_numpy(dtype="float64"), NEIGHBOUR_M)
        deg = np.array([len(x) for x in nb], dtype="int64")
        indptr = np.concatenate([[0], np.cumsum(deg)])
        flat = np.concatenate([np.asarray(x, dtype="int64") for x in nb]) if len(nb) else np.zeros(0, dtype="int64")
        flat_codes = _codes(ids[flat].astype("float64"), stand_categories)
        idx = pd.Index(ids.astype("float64")).get_indexer(own[q_rows])
        ok = idx >= 0
        rows_q, s_idx = q_rows[ok], idx[ok]
        counts = deg[s_idx]
        offsets = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)
        rows_out.append(np.repeat(rows_q, counts))
        codes_out.append(flat_codes[np.repeat(indptr[s_idx], counts) + offsets])
    if not rows_out:
        return np.zeros(0, dtype="int64"), np.zeros(0, dtype="int64")
    return np.concatenate(rows_out), np.concatenate(codes_out)


def reference_counts(mvt: pd.DataFrame, q_index: pd.Index, q_t: np.ndarray, which: str, windows) -> dict:
    """Movements of OTHER flights in ``[q - w, q]`` per airport: recorded takeoffs + landings (``recorded``, the
    airport's records) or scheduled departures + arrivals (``scheduled``, public); the flight's own excluded."""
    from prc.features import _oplib

    if which not in ("recorded", "scheduled"):
        raise ValueError(f"coverage_ref must be 'recorded' or 'scheduled', got {which!r}")
    col = "MVT_TIME_UTC_mvt" if which == "recorded" else "SCHED_TIME_UTC_mvt"
    dep, arr = _oplib.departures(mvt), _oplib.arrivals(mvt)
    dep_t = pd.Series(_oplib.secs(dep[col]), index=dep.index)
    arr_t = _oplib.secs(arr[col])
    q_apt = mvt.loc[q_index, "apt"].astype("string").to_numpy(dtype=object)
    own_t = dep_t.reindex(q_index).to_numpy(dtype="float64")
    out = {}
    for w in windows:
        n_dep = _oplib.backward_count(dep_t.to_numpy(), dep["apt"].astype("string").to_numpy(dtype=object), q_t, q_apt, w, own_t)
        n_arr = _oplib.backward_count(arr_t, arr["apt"].astype("string").to_numpy(dtype=object), q_t, q_apt, w)
        out[w] = (n_dep + n_arr).astype("float64")
    return out


def supported(q_apt: np.ndarray, family: str, ratio: np.ndarray, support: pd.DataFrame | None,
              min_support: float, margin: float = 1.0) -> np.ndarray:
    """Rows where ``family`` has training support: the airport's gated share >= ``min_support`` and the row's scale
    ratio inside the airport's training range ``[lo / margin, hi * margin]``. ``support`` None: every row (the support
    table not applied)."""
    if support is None:
        return np.ones(len(ratio), dtype=bool)
    s = support.loc[support["family"].astype(str).eq(family)].drop_duplicates("apt").set_index("apt")
    apt = pd.Series(np.asarray(q_apt, dtype=object))
    share = apt.map(s["share"]).to_numpy(dtype="float64")
    lo = apt.map(s["lo"]).to_numpy(dtype="float64") / float(margin)
    hi = apt.map(s["hi"]).to_numpy(dtype="float64") * float(margin)
    r = np.asarray(ratio, dtype="float64")
    with np.errstate(invalid="ignore"):
        return (share >= float(min_support)) & (r >= lo) & (r <= hi)


def hold_label_map(support: pd.DataFrame | None) -> dict[str, str]:
    """``apt|label -> apt|target`` of the support table's ``hold_map`` rows (empty without a table)."""
    if support is None or not len(support):
        return {}
    m = support.loc[support["family"].astype(str).eq("hold_map")]
    return {f"{a}|{lab}": f"{a}|{tgt}" for a, lab, tgt in zip(m["apt"].astype(str), m["label"].astype(str), m["target"].astype(str))}


def hold_agreement(q_apt: np.ndarray, support: pd.DataFrame | None) -> np.ndarray:
    """Per query, the airport's training share of (relabelled) holds whose runway end the airframe's next takeoff used
    (NaN for an airport with fewer than ``HOLD_MAP_MIN_N`` such holds; 1 for every row without a table)."""
    if support is None:
        return np.ones(len(q_apt))
    s = support.loc[support["family"].astype(str).eq("own_hold")].drop_duplicates("apt").set_index("apt")
    share = s["share"].where(s["n"] >= HOLD_MAP_MIN_N)
    return pd.Series(np.asarray(q_apt, dtype=object)).map(share).to_numpy(dtype="float64")


def hold_runway_map(pairs: pd.DataFrame) -> pd.DataFrame:
    """``hold_map`` and ``own_hold`` support rows from hold/takeoff pairs (``apt``, ``label`` = the hold's runway end,
    ``target`` = the same airframe's next takeoff's within 30 min; `prc.adsb.surface.hold_takeoff_pairs`): a label whose
    holds took off from one other end on at least ``HOLD_MAP_MIN_SHARE`` of ``HOLD_MAP_MIN_N`` or more is relabelled
    to it, and the agreement after relabelling is recorded per airport."""
    recs = []
    if pairs is None or not len(pairs):
        return pd.DataFrame(recs, columns=list(SUPPORT_COLUMNS))
    p = pairs.assign(apt=pairs["apt"].astype(str), label=pairs["label"].astype(str), target=pairs["target"].astype(str))
    for apt, g in p.groupby("apt", sort=True):
        relabel = {}
        for label, h in g.groupby("label", sort=True):
            vc = h["target"].value_counts()
            top, share = str(vc.index[0]), float(vc.iloc[0] / len(h))
            if top != label and len(h) >= HOLD_MAP_MIN_N and share >= HOLD_MAP_MIN_SHARE:
                relabel[label] = top
                recs.append({"apt": apt, "family": "hold_map", "n": int(len(h)), "share": share, "label": label,
                             "target": top})
        mapped = g["label"].map(lambda x: relabel.get(x, x))
        recs.append({"apt": apt, "family": "own_hold", "n": int(len(g)), "share": float((mapped == g["target"]).mean())})
    return pd.DataFrame(recs, columns=list(SUPPORT_COLUMNS))


def support_table(internals: pd.DataFrame, hold_pairs: pd.DataFrame | None = None) -> pd.DataFrame:
    """The airport x family support from training rows' `INTERNALS`: per airport and family the rows, the share whose
    ADS-B-only gate holds (hold: the taxi gate), and the 1st-99th percentile of the family's scale ratio there; with
    ``hold_pairs`` the hold relabelling and agreement rows of `hold_runway_map`."""
    recs = []
    for apt, g in internals.groupby(internals["apt"].astype(str), sort=True):
        for family, gate, ratio in (("taxi", "taxi_rx", "r_taxi"), ("hold", "taxi_rx", "r_hold"), ("stand", "stand_rx", "r_stand")):
            ok = g[gate].to_numpy(dtype=bool)
            r = g[ratio].to_numpy(dtype="float64")[ok]
            r = r[np.isfinite(r)]
            lo, hi = (np.quantile(r, SUPPORT_QUANTILES) if len(r) else (np.nan, np.nan))
            recs.append({"apt": apt, "family": family, "n": int(len(g)), "share": float(ok.mean()) if len(g) else 0.0,
                         "lo": float(lo), "hi": float(hi)})
    out = pd.DataFrame(recs, columns=list(SUPPORT_COLUMNS))
    holds = hold_runway_map(hold_pairs)
    if len(holds):
        out = pd.concat([out, holds], ignore_index=True)
    for c in ("label", "target"):
        out[c] = out[c].astype(object).where(out[c].notna(), None)
    return out


def surface_at(mvt: pd.DataFrame, q_index: pd.Index, a: np.ndarray, params: dict | None = None,
               events: pd.DataFrame | None = None, return_internals: bool = False):
    """The ADS-B surface picture of OTHER aircraft at ``q = floor(a) - ADSB_LAG_S`` for the departures ``q_index``.

    Reads from the query rows only ``apt``, ``STAND_mvt``, ``RUNWAY_mvt``, ``CALLSIGN_flt`` and, for the reference
    movements and the own exclusion, the departures' and arrivals' movement times (``coverage_ref``). ``events``: an
    event table to read instead of the snapshot's (the audit, tests). With ``return_internals``: ``(frame, internals)``
    with the `INTERNALS` the support table is built from.
    """
    from prc.features import _geolib

    prm = check_params(params)
    a = np.asarray(a, dtype="float64")
    q_t = np.floor(a) - ADSB_LAG_S
    q = mvt.loc[q_index]
    n = len(q_index)
    out = {c: np.full(n, np.nan) for c in COLUMNS}
    finite = np.isfinite(q_t)
    shift_days = int(prm["shift_days"])
    if events is None:
        shift = shift_days * 86_400
        if finite.any():
            ev = load_events(np.nanmin(q_t[finite]) - EVENTS_MARGIN_S - shift, np.nanmax(q_t[finite]) - shift)
        else:
            ev = load_events(0.0, 0.0)
    else:
        ev = events
    ev = shifted(ev, shift_days)

    for c in ("apt", "kind", "cs", "rwy_end", "origin_kind"):  # codes, not per-event strings (millions of events)
        if not isinstance(ev[c].dtype, pd.CategoricalDtype):
            ev = ev.assign(**{c: ev[c].astype("category")})
    support = load_support() if prm["support"] else None

    # shared code spaces: airports, airport+runway end, callsigns (the events'), stands
    q_apt = q["apt"].astype(str).to_numpy(dtype=object)
    ev_apt_cats = ev["apt"].cat.categories.astype(str)
    apt_cats = pd.Index(sorted(set(q_apt) | set(ev_apt_cats)))
    ev_g = _recode(ev["apt"], apt_cats)
    raw_rwy = q["RUNWAY_mvt"].astype("string")
    ident = raw_rwy.map({v: _geolib.norm_runway_ident(v) for v in raw_rwy.dropna().unique()})
    q_rwy = (pd.Series(q_apt, index=q.index) + "|" + ident).astype(object).where(ident.notna(), None).to_numpy(dtype=object)
    # the events' (airport, runway end) pairs: labelled once per distinct pair; a hold's label relabelled by the
    # support table where the training days' next takeoffs contradict it
    r_codes = ev["rwy_end"].cat.codes.to_numpy(dtype="int64")
    n_r = len(ev["rwy_end"].cat.categories) + 1
    pair = np.where((ev_g >= 0) & (r_codes >= 0), ev_g * n_r + r_codes, -1)
    uniq = np.unique(pair[pair >= 0])
    r_labels = ev["rwy_end"].cat.categories.astype(str)
    labels = [f"{apt_cats[p // n_r]}|{r_labels[p % n_r]}" for p in uniq]
    relabel = hold_label_map(support)
    held = [relabel.get(x, x) for x in labels]
    rwy_cats = pd.Index(sorted({x for x in q_rwy if x} | set(labels) | set(held)))
    ev_gr = np.full(len(ev), -1, dtype="int64")
    ev_gr_hold = np.full(len(ev), -1, dtype="int64")
    ok = pair >= 0
    if labels:
        at = np.searchsorted(uniq, pair[ok])
        ev_gr[ok] = rwy_cats.get_indexer(labels)[at]
        ev_gr_hold[ok] = rwy_cats.get_indexer(held)[at]
    cs_cats = pd.Index(ev["cs"].cat.categories.astype(str))
    stands, matches = load_stand_tables()
    stand_cats = pd.Index(np.unique(np.concatenate([stands["stand_id"].to_numpy(dtype="float64"),
                                                    ev["origin_stand"].dropna().to_numpy(dtype="float64")])))
    q_cs = _codes(norm_callsign(q["CALLSIGN_flt"].to_numpy(dtype=object)) if "CALLSIGN_flt" in q.columns else np.full(n, None), cs_cats)
    nb_rows, nb_stand = neighbour_pairs(q_apt, q["STAND_mvt"].astype("string").to_numpy(dtype=object), stands, matches, stand_cats)
    qa = Queries(q_t, _codes(q_apt, apt_cats), q_cs, nb_rows, nb_stand, len(apt_cats), len(cs_cats), len(stand_cats))
    qr = Queries(q_t, _codes(q_rwy, rwy_cats), q_cs, nb_rows, nb_stand, len(rwy_cats), len(cs_cats), len(stand_cats))

    kind = np.asarray(ev["kind"].cat.categories.astype(str), dtype=object)[ev["kind"].cat.codes.to_numpy()]
    ev_cs = ev["cs"].cat.codes.to_numpy(dtype="int64")
    ev_os = _codes(ev["origin_stand"].to_numpy(dtype="float64"), stand_cats)
    tks = ev["t_known_start"].to_numpy(dtype="float64")
    tke = ev["t_known_end"].to_numpy(dtype="float64")
    dur = ev["dur"].to_numpy(dtype="float64")

    def sel(k):
        return np.flatnonzero(kind == k)

    def stat(idx, t, backs, values=None, exclude=True, by_rwy=False, rwy=None):
        qq = qr if by_rwy else qa
        g = (ev_gr if rwy is None else rwy) if by_rwy else ev_g
        return others(qq, t[idx] if t is not None else None, g[idx], ev_cs[idx], ev_os[idx], backs,
                      {k: v[idx] if len(v) == len(ev) else v for k, v in (values or {}).items()}, exclude)

    def in_progress(idx, cap, values=None, exclude=True, by_rwy=False, rwy=None):
        g = (ev_gr if rwy is None else rwy) if by_rwy else ev_g
        stop = _in_progress_stop(ev.iloc[idx], cap)
        started = stat(idx, tks, [None], values, exclude, by_rwy, rwy)
        ended = others(qr if by_rwy else qa, stop, g[idx], ev_cs[idx], ev_os[idx], [None],
                       {k: v[idx] for k, v in (values or {}).items()}, exclude)
        return {k: started[k] - ended[k] for k in started}

    # coverage: the base gate (ADS-B against the reference movements, the activity heartbeat), the family gates
    # (ADS-B only)
    tko, ldg, taxi, hold = sel("TAKEOFF"), sel("LANDING"), sel("TAXI_OUT"), sel("RWY_HOLD")
    seen = ev["taxi_seen"].to_numpy(dtype="int64")
    push_origin = (ev["origin_kind"] == "push").to_numpy(dtype=bool).astype("int64")
    windows = (7200, 1800, 600)
    s_tko = stat(tko, tks, [*windows, PUSH_SHARE_S], {"seen": seen, "push": push_origin})
    s_ldg = stat(ldg, tks, list(dict.fromkeys([*windows, ACT_RECENT_S, ACT_BASE_S])), exclude=False)
    ref = reference_counts(mvt, q_index, q_t, str(prm["coverage_ref"]), windows)
    n_tko = s_tko[(7200, "n")].astype("float64")
    n_tko6 = s_tko[(PUSH_SHARE_S, "n")].astype("float64")
    min_tko = float(prm["min_tko"])
    s_act = stat(np.flatnonzero(np.isin(kind, ACT_DEP_KINDS)), tks, [ACT_RECENT_S, ACT_BASE_S])
    act_recent = (s_act[(ACT_RECENT_S, "n")] + s_ldg[(ACT_RECENT_S, "n")]).astype("float64")
    act_before = (s_act[(ACT_BASE_S, "n")] + s_ldg[(ACT_BASE_S, "n")]).astype("float64") - act_recent
    act_base = act_before * ACT_RECENT_S / (ACT_BASE_S - ACT_RECENT_S)  # the mean per ACT_RECENT_S before
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = {w: np.where(ref[w] > 0, (s_tko[(w, "n")] + s_ldg[(w, "n")]) / np.maximum(ref[w], 1), np.nan) for w in windows}
        act_ratio = np.where(act_base > 0, act_recent / np.where(act_base > 0, act_base, 1.0), np.nan)
        taxi_share = np.where(n_tko >= min_tko, s_tko[(7200, "seen")] / np.maximum(n_tko, 1), np.nan)
        push_share = np.where(n_tko6 >= min_tko, s_tko[(PUSH_SHARE_S, "push")] / np.maximum(n_tko6, 1), np.nan)
        n_taxi = stat(taxi, tks, [7200])[(7200, "n")].astype("float64")
        n_hold = stat(hold, tks, [7200])[(7200, "n")].astype("float64")
        r_taxi = np.where(n_tko >= min_tko, n_taxi / np.maximum(n_tko, 1), np.nan)
        r_hold = np.where(n_tko >= min_tko, n_hold / np.maximum(n_tko, 1), np.nan)
    outage = (((ref[1800] >= float(prm["outage_min_ref"])) & ~(ratio[1800] >= float(prm["outage_max_ratio"])))
              | ((ref[600] >= float(prm["outage10_min_ref"])) & ~(ratio[600] >= float(prm["outage10_max_ratio"])))
              | ((act_base >= float(prm["act_min"])) & (act_recent < float(prm["act_max_ratio"]) * act_base)))
    cov_ok = (ref[7200] >= float(prm["min_ref"])) & (ratio[7200] >= float(prm["min_ratio"])) & ~outage
    taxi_rx = taxi_share >= float(prm["min_taxi"])
    stand_rx = push_share >= float(prm["min_push"])
    min_support, margin = float(prm["min_support"]), float(prm["support_margin"])
    taxi_ok = cov_ok & taxi_rx & supported(q_apt, "taxi", r_taxi, support, min_support, margin)
    hold_ok = taxi_ok & supported(q_apt, "hold", r_hold, support, min_support, margin)
    stand_ok = cov_ok & stand_rx & supported(q_apt, "stand", push_share, support, min_support, margin)
    out.update(cov_tko_n_120=n_tko, cov_ref_n_120=ref[7200], cov_mov_ratio_120=ratio[7200], cov_mov_ratio_30=ratio[1800],
               cov_mov_ratio_10=ratio[600], cov_act_ratio_5=act_ratio, cov_taxi_share_120=taxi_share,
               cov_push_share_360=push_share, cov_taxi_ratio_120=r_taxi, cov_hold_ratio_120=r_hold,
               cov_ok=cov_ok.astype("float64"), cov_taxi_ok=taxi_ok.astype("float64"),
               cov_hold_ok=hold_ok.astype("float64"), cov_stand_ok=stand_ok.astype("float64"))
    detect = np.clip(np.nan_to_num(ratio[7200]), 0.0, 1.0) * np.nan_to_num(taxi_share)

    # taxi family (taxiways, remote stops)
    t_start = ev["t_start"].to_numpy(dtype="float64")
    base = (np.nanmin(t_start) if len(t_start) and np.isfinite(t_start).any() else 0.0)
    t_rel = t_start - base
    p_out = in_progress(taxi, CAP_TAXI_OUT_S, {"start": t_rel})
    n_out = p_out[(None, "n")].astype("float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        elapsed = np.where(n_out > 0, (n_out * (q_t - base) - p_out[(None, "start")]) / np.maximum(n_out, 1), np.nan)
    out["taxi_out_now"] = n_out
    out["taxi_out_now_norm"] = n_out / np.maximum(detect, 0.25)
    out["taxi_out_elapsed_mean_now"] = elapsed
    # taxi-out starts: from a push, a stop, or a first sighting near a stand (a split taxi's restart is no new start)
    origin = ev["origin_kind"].astype(str).to_numpy(dtype=object)
    near = np.isfinite(ev["origin_stand"].to_numpy(dtype="float64"))
    starts = taxi[np.isin(origin[taxi], ("push", "stop")) | ((origin[taxi] == "first_seen") & near[taxi])]
    out["taxi_out_start_15"] = stat(starts, tks, [900])[(900, "n")].astype("float64")
    out["taxi_in_now"] = in_progress(sel("TAXI_IN"), CAP_TAXI_IN_S, exclude=False)[(None, "n")].astype("float64")
    remote = sel("REMOTE_STOP")
    out["remote_now"] = in_progress(remote, CAP_REMOTE_S)[(None, "n")].astype("float64")
    out["remote_60"] = stat(remote, tks, [3600])[(3600, "n")].astype("float64")
    has_parent = np.isfinite(dur).astype("int64")
    stop_s = np.nan_to_num(ev["stop_s"].to_numpy(dtype="float64"))
    s_par = stat(tko, tks, [3600], {"par": has_parent, "stop": np.where(has_parent > 0, stop_s, 0.0),
                                     "dur": np.where(has_parent > 0, np.clip(np.nan_to_num(dur), 0, TAXI_DUR_CAP_S), 0.0)})
    with np.errstate(invalid="ignore", divide="ignore"):
        npar = s_par[(3600, "par")].astype("float64")
        out["taxi_stop_s_mean_60"] = np.where(npar > 0, s_par[(3600, "stop")] / np.maximum(npar, 1), np.nan)
        out["taxi_stop_share_60"] = np.where(s_par[(3600, "dur")] > 0, s_par[(3600, "stop")] / np.maximum(s_par[(3600, "dur")], 1), np.nan)

    # hold family (the runway queue); the own runway (O*) only where it is a runway end of the airport and the
    # airport's holds, relabelled by the support table, agree with the airframes' next takeoffs
    out["rwy_hold_now"] = in_progress(hold, CAP_HOLD_S)[(None, "n")].astype("float64")
    own_rwy = in_progress(hold, CAP_HOLD_S, by_rwy=True, rwy=ev_gr_hold)[(None, "n")].astype("float64")
    keys = load_runway_keys()
    known_rwy = np.array([x is not None and x in keys for x in q_rwy], dtype=bool)
    agree = hold_agreement(q_apt, support) >= float(prm["min_hold_agree"])
    out["rwy_own_hold_now"] = np.where(known_rwy & (qr.group >= 0) & agree, own_rwy, np.nan)
    for name, idx, back in (("remote_dur_mean_60", remote, 3600), ("rwy_hold_dur_mean_30", hold, 1800)):
        done = idx[np.isfinite(tke[idx]) & np.isfinite(dur[idx])]
        s = stat(done, tke, [back], {"dur": np.clip(np.nan_to_num(dur), 0, CAP_REMOTE_S)})
        with np.errstate(invalid="ignore", divide="ignore"):
            out[name] = np.where(s[(back, "n")] > 0, s[(back, "dur")] / np.maximum(s[(back, "n")], 1), np.nan)

    # stand family (pushes, push-origin taxi durations)
    push = sel("PUSH")
    out["push_15"] = stat(push, tks, [900])[(900, "n")].astype("float64")
    out["push_15_norm"] = out["push_15"] / np.maximum(np.nan_to_num(push_share) * np.clip(np.nan_to_num(ratio[7200]), 0, 1), 0.1)
    pdur = tko[(push_origin[tko] > 0) & np.isfinite(dur[tko])]
    s_dur = stat(pdur, tks, [1800, 3600], {"dur": np.clip(np.nan_to_num(dur), 0, TAXI_DUR_CAP_S)})
    for w, tag in ((1800, "30"), (3600, "60")):
        nd = s_dur[(w, "n")].astype("float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"taxi_dur_mean_{tag}"] = np.where(nd > 0, s_dur[(w, "dur")] / np.maximum(nd, 1), np.nan)
        out[f"taxi_dur_n_{tag}"] = nd

    frame = pd.DataFrame(out, index=q_index)[list(COLUMNS)].astype("float64")
    frame.loc[~taxi_ok, list(TAXI)] = np.nan
    frame.loc[~hold_ok, list(HOLD)] = np.nan
    frame.loc[~stand_ok, list(STAND)] = np.nan
    frame.loc[~finite] = np.nan
    frame = frame.astype("float32")
    if not return_internals:
        return frame
    internals = pd.DataFrame({"apt": q_apt, "taxi_rx": taxi_rx & finite, "stand_rx": stand_rx & finite,
                              "r_taxi": r_taxi, "r_hold": r_hold, "r_stand": push_share}, index=q_index)
    return frame, internals
