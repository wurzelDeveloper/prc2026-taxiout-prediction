"""Primitives of the operational ladder: anchors before the movement, and windows that end at them.

The challenge blocks are anchored on the takeoff, which the ranking set gives for every flight. An
operational prediction is made *before* the aircraft moves, so every input here is anchored on a
prediction time ``a`` instead:

* **H1** -- 15 minutes before the planned off-block: ``a = planned - 900 s`` with ``planned`` = ``IOBT_flt``,
  else ``LOBT_flt``, else ``SCHED_TIME_UTC_mvt`` (`planned_offblock`, `anchor_h1`). Board-scorable: every input
  exists for 2026.
* **H0** -- at the recorded off-block (at pushback): ``a`` = the departure's ``BLOCK_TIME_UTC_mvt``, read only
  through `load_offblocks` (the target file, that one column). 2026 departure block times are blanked, so H0
  exists for 2025 only and is never board-scored (lab).

The flight's own takeoff, own NM actuals (``AOBT_3_flt``, ``ARVT_3_flt``) and ``EOBT_1_flt`` are never inputs.
Other flights' event *times* enter only when the event happened at or before ``a``; every window is
``[a - w, a]``, inclusive at both ends, in whole seconds (`secs`). The own flight is a reference event in several
of the tables (its own takeoff, its own NM reading, its own off-block): it is removed explicitly in every count,
sum and gap, and its own NM reading never enters the baseline its windows read (`NmBaseline`, month-out), so
moving any own event -- even to before ``a`` -- leaves the result bit-identical, also for a fit row
(tests/test_operational.py, T1). Values that are summed are rounded to whole seconds first, so subtracting the
own value is exact arithmetic.

One exception to "only what is known at ``a``", declared as tier O*: the runway and pier *groupings* of other
flights that have not taken off by ``a`` (``backlog_rwy``, ``backlog_pier``, ``feed_queue_rwy``,
``feed_queue_pier``). They use those flights' recorded ``RUNWAY_mvt`` / ``STAND_mvt``, i.e. the runway they
actually used, which is only certain at their takeoff; operationally the planned runway (and the stand they
are parked at) stands in for it. Test T2b censors those attributes after ``a`` and asserts that only these
four columns per horizon may move (`O_STAR_FUTURE_ATTRIBUTES`).

METAR convention: a report counts as known ``METAR_LAG_S`` after its observation time (dissemination delay),
so the METAR side of the weather is read at ``a - METAR_LAG_S``. TAFs count from their issue time.

Censoring convention (T2): an event that has not happened by ``a`` may as well be missing. An interval that has
started but whose end is missing or later than ``a`` is "in progress"; `in_progress` caps every interval at
``max_dur`` so that a record without an end (an arrival with no in-block time, a stale block time) does not
count for ever.

Archive tables (target-free, transductive in time): the arrivals' taxi-in baseline of `_arrlib` (median per
airport, runway and stand over every arrival in the frame, 2026 included) and the NM proxy baseline
(`NmBaseline`: median per airport, stand and runway over the *fit* rows only, leaving out the fit rows of the
query's own planned month, so a fit row's own reading never defines the anomalies it reads). Both are held fixed
in the future-deletion test.

Underscore module: not auto-imported, but hashed into `prc.features.operational`, which imports it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc import paths
from prc.features import _arrlib, _proxylib
from prc.features.congestion import _keyed, _prepare, count_upto, window_count, window_reduce
from prc.features.core import epoch_seconds
from prc.features.registry import FeatureContext
from prc.features.stand_runway import stand_prefix_key
from prc.features.temporal import local_naive
from prc.features.turnaround import stand_asof
from prc.features.weather import wx_now, wx_past
from prc.features.weather_fcst import taf_now

H1_LEAD_S = 900.0
"""H1 sits this long before the planned off-block."""
LAB_EXTERNAL = (paths.TARGET_FILE,)
"""The external file of every lab block (the airport's own off-block records); declared by `op_h0_*` and `*_feed_lab`."""
OFFBLOCK_COLUMN = "BLOCK_TIME_UTC_mvt"

GAP_CAP_S = 7200.0
"""Gaps to the previous event are capped here (as `runway_queue` does)."""
STAND_CAP_MIN = 1440.0
"""Minutes since the stand's last in-block are capped at a day."""
ARR_TAXIING_MAX_S = 3600.0
"""An arrival counts as taxiing in for at most an hour after landing (records without an in-block time)."""
FEED_QUEUE_MAX_S = 3600.0
"""A departure counts in the taxiing-out queue for at most an hour after its recorded off-block (stale block times)."""
BACKLOG_MAX_S = 3 * 3600.0
"""A departure counts in the plan backlog for at most three hours after its planned off-block (a schedule of another leg)."""
FEED_TAXI_CAP_S = 3600.0
"""Recorded taxi-outs of other flights are clipped here before they are averaged."""
HIGH_ANOMALY_S = 300.0
"""An NM proxy anomaly above this counts as a slow taxi in the ``share5`` columns."""
PUSH_DELAY_CLIP = (-3600.0, 4 * 3600.0)
CONFIG_BUCKET_S = 900
MAX_CONFIG_AGE_MIN = 720.0
CONFIG_LOOKBACK_BUCKETS = 60
"""Buckets the slow path of `config_age_minutes` re-reads: 15 h, more than the 12 h cap."""

METAR_LAG_S = 300.0
"""A METAR counts as known this long after its observation time (dissemination); the METAR side is read at a - lag."""

WINDOWS_TRAFFIC = {"b15": 900, "b30": 1800, "b60": 3600}

O_STAR_FUTURE_ATTRIBUTES = ("backlog_rwy", "backlog_pier", "feed_queue_rwy", "feed_queue_pier")
"""Columns that group other, not yet airborne flights by their recorded runway / stand (tier O*, see above)."""


# --- clock and anchors --------------------------------------------------------------------

def secs(ts: pd.Series) -> np.ndarray:
    """UTC timestamps as whole epoch seconds (float64, floored); NaT becomes NaN."""
    return np.floor(epoch_seconds(ts).to_numpy(dtype="float64", na_value=np.nan))


def to_timestamps(a: np.ndarray, index: pd.Index) -> pd.Series:
    """Epoch seconds back to UTC timestamps (NaN -> NaT)."""
    return pd.Series(pd.to_datetime(np.asarray(a, dtype="float64"), unit="s", utc=True), index=index)


def planned_offblock(dep: pd.DataFrame) -> np.ndarray:
    """The planned off-block known well before the flight: ``IOBT_flt``, else ``LOBT_flt``, else the schedule."""
    p = secs(dep["IOBT_flt"])
    for col in ("LOBT_flt", "SCHED_TIME_UTC_mvt"):
        p = np.where(np.isnan(p), secs(dep[col]), p)
    return p


def anchor_h1(dep: pd.DataFrame) -> np.ndarray:
    """H1: 15 minutes before the planned off-block (whole seconds)."""
    return planned_offblock(dep) - H1_LEAD_S


def load_offblocks() -> pd.Series:
    """The recorded off-block of every training departure (epoch seconds, indexed by ``mvt_id``).

    The one place in `prc.features` that reads the target file, and only its ``BLOCK_TIME_UTC_mvt`` column:
    the airport's own off-block records, which an airport has in real time (A-CDM AOBT). Lab only: the 2026
    departures have none, so every row outside the training split is NaN. Memoised on the file's mtime.
    """
    path = paths.TARGET_FILE
    key = (str(path), path.stat().st_mtime_ns)
    if _OFFBLOCKS.get("key") != key:
        frame = pd.read_parquet(path, columns=[OFFBLOCK_COLUMN])
        _OFFBLOCKS["key"] = key
        _OFFBLOCKS["value"] = pd.Series(secs(frame[OFFBLOCK_COLUMN]), index=frame.index, name="offblock")
    return _OFFBLOCKS["value"]


_OFFBLOCKS: dict = {}


def anchor_h0(dep: pd.DataFrame, offblocks: pd.Series) -> np.ndarray:
    """H0: the departure's recorded off-block; NaN where there is none (every 2026 row).

    For a stale record (block time = schedule) this is the schedule, not the push.
    """
    return offblocks.reindex(dep.index).to_numpy(dtype="float64")


# --- self-excluded windows on the integer clock ------------------------------------------------

def _series(x, index=None) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.reset_index(drop=True)
    return pd.Series(np.asarray(x), index=index)


def _own_in_window(q: np.ndarray, q_ok: np.ndarray, own_t, back: float, fwd: float = 0.0) -> np.ndarray:
    """Whether the query's own event sits in its window, with the bounds `window_count` uses."""
    if own_t is None:
        return np.zeros(len(q), dtype=bool)
    own = np.asarray(own_t, dtype="float64")
    with np.errstate(invalid="ignore"):
        return q_ok & np.isfinite(own) & (own >= q - back) & (own <= q + fwd)


def _obj(x) -> np.ndarray:
    """An object array with every missing value as ``None`` (``pd.NA`` breaks element-wise comparisons)."""
    arr = np.array(x, dtype=object)
    arr[pd.isna(arr)] = None
    return arr


def _query_ok(q_t, q_g) -> np.ndarray:
    q = np.asarray(q_t, dtype="float64")
    return np.isfinite(q) & pd.notna(np.asarray(q_g, dtype=object))


def backward_count(ref_t, ref_g, q_t, q_g, window: float, own_t=None) -> np.ndarray:
    """Reference events in ``[a - window, a]`` of each query's group, the query's own event excluded.

    ``own_t`` is the query's own time in the reference table (NaN where the query is not in it). Integer
    seconds throughout, so an own event exactly at ``a`` or at ``a - window`` is excluded exactly when it
    was counted.
    """
    q = np.asarray(q_t, dtype="float64")
    n = window_count(_series(ref_t), _series(ref_g), _series(q), _series(q_g), back=window, fwd=0).astype("int64")
    return n - _own_in_window(q, _query_ok(q, q_g), own_t, window).astype("int64")


def backward_stats(ref_t, ref_g, ref_v, q_t, q_g, window: float, own_t=None, own_v=None, high: float | None = None):
    """Count, mean and (with ``high``) share above ``high`` of the reference *values* in ``[a - window, a]``.

    Values are rounded to whole seconds, events with a missing value are left out, and the own event is
    removed from the count and the sum. The mean of an empty window is NaN. ``own_t`` and ``own_v`` come
    together or not at all (one without the other raises: the own event would silently stay in).
    """
    if (own_t is None) != (own_v is None):
        raise ValueError("backward_stats: pass own_t and own_v together (or neither), else the own event stays in")
    t = np.asarray(ref_t, dtype="float64")
    v = np.round(np.asarray(ref_v, dtype="float64"))
    g = np.asarray(ref_g, dtype=object)
    keep = np.isfinite(v) & np.isfinite(t)
    t, v, g = t[keep], v[keep], g[keep]
    q = np.asarray(q_t, dtype="float64")
    ok = _query_ok(q, q_g)
    ov = None if own_v is None else np.round(np.asarray(own_v, dtype="float64"))
    own = _own_in_window(q, ok, own_t, window) & (np.isfinite(ov) if ov is not None else False)

    n = window_count(_series(t), _series(g), _series(q), _series(q_g), back=window, fwd=0).astype("int64") - own.astype("int64")
    total = np.nan_to_num(window_reduce(_series(t), _series(g), _series(v), _series(q), _series(q_g), back=window, fwd=0, how="sum"))
    if ov is not None:
        total = total - np.where(own, ov, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, total / np.maximum(n, 1), np.nan)
    share = None
    if high is not None:
        hi = v > high
        n_hi = window_count(_series(t[hi]), _series(g[hi]), _series(q), _series(q_g), back=window, fwd=0).astype("int64")
        if ov is not None:
            with np.errstate(invalid="ignore"):
                n_hi = n_hi - (own & (ov > high)).astype("int64")
        share = np.where(n > 0, n_hi / np.maximum(n, 1), np.nan)
    return n, mean, share


def in_progress(start, end, grp, q_t, q_g, own_start=None, own_end=None, max_dur: float | None = None) -> np.ndarray:
    """Intervals of the query's group in progress at ``a``: ``start <= a < stop``, the own interval excluded.

    ``stop = max(start, end)``, with ``end`` capped at ``start + max_dur``. A missing ``end`` means "not ended
    yet" (so censoring every event after ``a`` changes nothing), up to the cap. Computed as
    ``count_upto(start) - count_upto(stop)`` on the integer clock.
    """
    s = np.asarray(start, dtype="float64")
    e = np.asarray(end, dtype="float64")
    stop = _stop(s, e, max_dur)
    q = np.asarray(q_t, dtype="float64")
    g = _series(grp)
    n = count_upto(_series(s), g, _series(q), _series(q_g)).astype("int64") - count_upto(
        _series(stop), g, _series(q), _series(q_g)
    ).astype("int64")
    if own_start is None:
        return n
    os_ = np.asarray(own_start, dtype="float64")
    ostop = _stop(os_, np.asarray(own_end, dtype="float64"), max_dur)
    ok = _query_ok(q, q_g)
    with np.errstate(invalid="ignore"):
        own = ok & np.isfinite(os_) & (os_ <= q) & ~(np.isfinite(ostop) & (ostop <= q))
    return n - own.astype("int64")


def _stop(s: np.ndarray, e: np.ndarray, max_dur: float | None) -> np.ndarray:
    if max_dur is not None:
        e = np.fmin(e, s + max_dur)
    return np.maximum(s, e)  # NaN start -> NaN (never started); NaN end without a cap -> NaN (never stops)


def last_gap(ref_t: pd.Series, ref_g: pd.Series, q_t, q_g, q_ids: pd.Index) -> np.ndarray:
    """Seconds from ``a`` back to the latest reference event at or before it, never the query's own event.

    ``ref_t``/``ref_g`` are indexed by event id; a query whose id is among them has that event skipped (the
    next earlier one is used). NaN where the group has no earlier event. `congestion.neighbour_gap` has no
    self-exclusion, hence this.
    """
    q = _series(np.asarray(q_t, dtype="float64"))
    keys, q_times, q_codes, span, order = _prepare(ref_t.reset_index(drop=True), ref_g.reset_index(drop=True), q, _series(q_g), 0)
    out = np.full(len(q), np.nan)
    if not len(keys):
        return out
    own_pos = ref_t.index.get_indexer(q_ids)
    pos = np.searchsorted(keys, _keyed(q_times, q_codes, span), side="right") - 1
    safe = np.clip(pos, 0, len(keys) - 1)
    is_own = (pos >= 0) & (own_pos >= 0) & (order[safe] == own_pos)
    pos = np.where(is_own, pos - 1, pos)
    safe = np.clip(pos, 0, len(keys) - 1)
    same = (q_codes >= 0) & (pos >= 0) & (keys[safe] // span == q_codes)
    gap = (q_times - keys[safe] % span).astype("float64")
    out[same] = gap[same]
    return out


# --- runway configuration before a -------------------------------------------------------------

def runway_mix(dep_t, dep_apt, dep_rwy, q_t, q_apt, q_rwy, window: float, own_t=None, own_rwy=None):
    """Departure runways at the airport in ``[a - window, a]``, self excluded.

    Returns (distinct runways used, the own runway's share of those takeoffs, whether the own runway had the
    most takeoffs -- ties count as dominant). Share and flag are NaN where the own runway is unknown or no
    other flight took off in the window.
    """
    t = np.asarray(dep_t, dtype="float64")
    apt, rwy = _obj(dep_apt), _obj(dep_rwy)
    q = np.asarray(q_t, dtype="float64")
    qa, qr = _obj(q_apt), _obj(q_rwy)
    ot = np.full(len(q), np.nan) if own_t is None else np.asarray(own_t, dtype="float64")
    orw = np.full(len(q), None, dtype=object) if own_rwy is None else _obj(own_rwy)

    distinct = np.zeros(len(q), dtype="float64")
    share = np.full(len(q), np.nan)
    dominant = np.full(len(q), np.nan)
    valid = pd.notna(rwy) & np.isfinite(t) & pd.notna(apt)
    for airport in pd.unique(qa[pd.notna(qa)]):
        rows = np.flatnonzero(qa == airport)
        here = valid & (apt == airport)
        runways = sorted(pd.unique(rwy[here]).tolist())
        if not runways:
            continue
        qs = q[rows]
        ok = np.isfinite(qs)
        counts = np.zeros((len(rows), len(runways)), dtype="int64")
        const = np.full(len(rows), "x", dtype=object)
        for k, r in enumerate(runways):
            sel = here & (rwy == r)
            n = window_count(_series(t[sel]), _series(np.full(int(sel.sum()), "x", dtype=object)), _series(qs), _series(const), back=window, fwd=0)
            own = _own_in_window(qs, ok, ot[rows], window) & (orw[rows] == r)
            counts[:, k] = n.astype("int64") - own.astype("int64")
        total = counts.sum(axis=1)
        distinct[rows] = (counts > 0).sum(axis=1)
        col = np.array([runways.index(r) if (pd.notna(r) and r in runways) else -1 for r in qr[rows]], dtype="int64")
        own_cnt = np.where(col >= 0, counts[np.arange(len(rows)), np.clip(col, 0, None)], 0)
        has = (col >= 0) & (total > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            share[rows] = np.where(has, own_cnt / np.maximum(total, 1), np.nan)
        dominant[rows] = np.where(has, ((own_cnt == counts.max(axis=1)) & (own_cnt > 0)).astype("float64"), np.nan)
    bad = ~np.isfinite(q)
    distinct[bad] = np.nan
    return distinct, share, dominant


def _bucket_modes(t: np.ndarray, apt: np.ndarray, rwy: np.ndarray):
    """Counts per (airport, 15-min bucket, runway) and the bucket's dominant runway (most takeoffs, then name)."""
    valid = np.isfinite(t) & pd.notna(rwy) & pd.notna(apt)
    frame = pd.DataFrame(
        {
            "apt": apt[valid].astype(str),
            "bucket": (t[valid] // CONFIG_BUCKET_S).astype("int64"),
            "rwy": rwy[valid].astype(str),
        }
    )
    counts = frame.groupby(["apt", "bucket", "rwy"], sort=True).size().rename("n").reset_index()
    mode = counts.sort_values(["apt", "bucket", "n", "rwy"], ascending=[True, True, False, True], kind="stable")
    mode = mode.drop_duplicates(["apt", "bucket"]).sort_values(["apt", "bucket"], kind="stable").reset_index(drop=True)
    return counts, mode


def _mode_of(counts: dict[str, int]) -> str | None:
    live = {r: n for r, n in counts.items() if n > 0}
    if not live:
        return None
    return sorted(live.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def config_age_minutes(dep_t, dep_apt, dep_rwy, q_t, q_apt, own_t=None, own_rwy=None, force_slow: bool = False) -> np.ndarray:
    """Minutes since the airport's dominant departure runway last changed, read from 15-min buckets complete by ``a``.

    The dominant runway of a bucket is its most used departure runway; buckets without departures are
    skipped. Only buckets that ended at or before ``a`` are read (``bucket <= floor(a / 900) - 1``), so nothing
    after ``a`` enters. Capped at 12 h. A query whose own takeoff falls in such a bucket (the takeoff before the
    anchor) is recomputed on a slow path with its own takeoff removed. ``force_slow`` sends every query there
    (tests check that both paths agree).
    """
    t = np.asarray(dep_t, dtype="float64")
    apt, rwy = _obj(dep_apt), _obj(dep_rwy)
    q = np.asarray(q_t, dtype="float64")
    qa = _obj(q_apt)
    ot = np.full(len(q), np.nan) if own_t is None else np.asarray(own_t, dtype="float64")
    orw = np.full(len(q), None, dtype=object) if own_rwy is None else _obj(own_rwy)

    counts, mode = _bucket_modes(t, apt, rwy)
    out = np.full(len(q), np.nan)
    with np.errstate(invalid="ignore"):
        B = np.floor(q / CONFIG_BUCKET_S) - 1
        own_b = np.floor(ot / CONFIG_BUCKET_S)
        slow = np.isfinite(q) & np.isfinite(ot) & pd.notna(orw) & (own_b <= B)
    if force_slow:
        slow = np.isfinite(q)
    tables: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for airport, m in mode.groupby("apt", sort=False):
        bk = m["bucket"].to_numpy(dtype="int64")
        md = m["rwy"].to_numpy(dtype=object)
        changed = np.ones(len(bk), dtype=bool)
        changed[1:] = md[1:] != md[:-1]
        run_start = pd.Series(np.where(changed, bk, np.nan)).ffill().to_numpy()
        tables[airport] = (bk, md, run_start)

    fast = np.isfinite(q) & ~slow & pd.notna(qa)
    for airport in pd.unique(qa[fast]):
        if airport not in tables:
            continue
        bk, _, run_start = tables[airport]
        rows = np.flatnonzero(fast & (qa == airport))
        idx = np.searchsorted(bk, B[rows], side="right") - 1
        rs = run_start[np.clip(idx, 0, None)]
        age = np.minimum((q[rows] - rs * CONFIG_BUCKET_S) / 60.0, MAX_CONFIG_AGE_MIN)
        out[rows] = np.where(idx >= 0, age, np.nan)

    slow_rows = np.flatnonzero(slow & pd.notna(qa))
    if len(slow_rows):
        # Only the own takeoff's bucket is ever modified, so only those buckets' runway counts are needed.
        wanted = {(str(qa[i]), int(own_b[i])) for i in slow_rows if np.isfinite(own_b[i])}
        pair = list(zip(counts["apt"], counts["bucket"].astype("int64")))
        sel = counts.loc[[p in wanted for p in pair]] if wanted else counts.iloc[:0]
        by_bucket = {
            (a_, int(b_)): dict(zip(g["rwy"], g["n"])) for (a_, b_), g in sel.groupby(["apt", "bucket"], sort=False)
        }
        for i in slow_rows:
            airport = str(qa[i])
            if airport not in tables:
                continue
            out[i] = _config_age_slow(tables[airport], by_bucket, airport, q[i], B[i], own_b[i], orw[i])
    return out


def _config_age_slow(table, by_bucket, airport: str, q: float, B: float, own_b: float, own_rwy) -> float:
    bk, md, _ = table
    K0 = B - CONFIG_LOOKBACK_BUCKETS
    i_hi = int(np.searchsorted(bk, B, side="right"))
    i_lo = int(np.searchsorted(bk, K0, side="left"))
    i_start = max(i_lo - 2, 0)
    seq = []
    for j in range(i_start, i_hi):
        b = int(bk[j])
        m = md[j]
        if np.isfinite(own_b) and b == int(own_b) and own_rwy is not None and pd.notna(own_rwy):
            c = dict(by_bucket.get((airport, b), {}))
            if str(own_rwy) in c:
                c[str(own_rwy)] -= 1
            m = _mode_of(c)
            if m is None:
                continue
        seq.append((b, m))
    if not seq:
        return np.nan
    run = None
    for j, (b, m) in enumerate(seq):
        first_ever = j == 0 and i_start == 0
        changed = first_ever or (j > 0 and m != seq[j - 1][1])
        if changed and K0 <= b <= B:
            run = b
    if run is None:
        return MAX_CONFIG_AGE_MIN
    return float(min((q - run * CONFIG_BUCKET_S) / 60.0, MAX_CONFIG_AGE_MIN))


# --- event tables ------------------------------------------------------------------------------

def runway_key(frame: pd.DataFrame) -> pd.Series:
    return frame["apt"].astype("string") + "_" + frame["RUNWAY_mvt"].astype("string").fillna("__NA__")


def pier_key(frame: pd.DataFrame) -> pd.Series:
    return stand_prefix_key(frame["apt"], frame["STAND_mvt"])


def departures(mvt: pd.DataFrame) -> pd.DataFrame:
    return mvt.loc[mvt["is_dep"].to_numpy(dtype=bool)]


def arrivals(mvt: pd.DataFrame) -> pd.DataFrame:
    return mvt.loc[~mvt["is_dep"].to_numpy(dtype=bool)]


def arrival_anomaly(mvt: pd.DataFrame) -> pd.Series:
    """Every arrival's taxi-in anomaly against its (airport, runway, stand) median (`_arrlib`; an archive table)."""
    return _arrlib.taxiin_anomaly(mvt)


def baseline_month(frame: pd.DataFrame) -> np.ndarray:
    """Calendar month (``12 * year + month - 1``, UTC) of each departure's planned off-block; NaN without a plan.

    The month whose fit readings `NmBaseline` leaves out of the baseline a query reads. Read from the plan, never
    from the takeoff, so the key itself is known before the movement and does not move with the own takeoff.
    """
    dt = pd.Series(pd.to_datetime(planned_offblock(frame), unit="s", utc=True))
    return (dt.dt.year * 12 + dt.dt.month - 1).to_numpy(dtype="float64", na_value=np.nan)


class NmBaseline:
    """NM's taxi-out reading ``takeoff - AOBT_3`` of every departure, and its anomaly against month-out medians.

    The median per (airport, stand, runway) is taken over the *fit* departures only (fall-backs (airport, stand),
    (airport, runway), airport, each needing ``min_support`` readings), clipped like `dep_live_state` and rounded
    to whole seconds. `anomaly(month)` leaves out the fit readings whose planned off-block falls in ``month``: a
    query reads the anomalies of the table keyed by its own planned month (`key`), so its own reading -- a fit
    row's near-copy of its own taxi-out -- never enters a median that defines the other flights' anomalies it
    averages. Queries whose month holds no fit rows (holdout, 2026) read the full-fit table (key ``None``).
    """

    def __init__(self, dep: pd.DataFrame, fit: np.ndarray, min_support: int = _proxylib.MIN_SUPPORT):
        self.index = dep.index
        self.proxy = _proxylib.proxy_seconds(dep).to_numpy(dtype="float64")
        apt = dep["apt"].astype("string")
        stand = dep["STAND_mvt"].astype("string").fillna("__NA__")
        rwy = dep["RUNWAY_mvt"].astype("string").fillna("__NA__")
        keys = (apt + "|" + stand + "|" + rwy, apt + "|" + stand, apt + "|" + rwy, apt)
        self.codes = [pd.factorize(k.to_numpy(dtype=object), use_na_sentinel=True)[0] for k in keys]
        self.month = baseline_month(dep)
        self.use = np.isfinite(self.proxy) & np.asarray(fit, dtype=bool)
        self.min_support = int(min_support)
        self.fit_months = frozenset(int(m) for m in np.unique(self.month[self.use & np.isfinite(self.month)]))
        self._tables: dict = {}

    def key(self, month: float):
        """The table a query of planned month ``month`` reads: that month left out, or the full fit (``None``)."""
        return int(month) if np.isfinite(month) and int(month) in self.fit_months else None

    def anomaly(self, exclude_month: int | None = None) -> np.ndarray:
        """Anomaly of every departure (aligned to ``index``) against the fit medians without ``exclude_month``."""
        if exclude_month not in self._tables:
            use = self.use if exclude_month is None else self.use & (self.month != exclude_month)
            base = np.full(len(self.proxy), np.nan)
            for codes in self.codes:
                if not len(codes):
                    continue
                ok = use & (codes >= 0)
                stats = pd.Series(self.proxy[ok]).groupby(codes[ok]).agg(["median", "size"])
                table = np.full(int(codes.max()) + 1, np.nan)
                good = stats["size"].to_numpy() >= self.min_support
                table[stats.index.to_numpy()[good]] = stats["median"].to_numpy()[good]
                need = np.isnan(base) & (codes >= 0)
                base[need] = table[codes[need]]
            with np.errstate(invalid="ignore"):
                self._tables[exclude_month] = np.round(np.clip(self.proxy - base, *_proxylib.ANOMALY_CLIP))
        return self._tables[exclude_month]


def nm_baseline(dep: pd.DataFrame, fit: np.ndarray, min_support: int = _proxylib.MIN_SUPPORT) -> NmBaseline:
    """The NM proxy baseline over the fit rows (the one entry point the blocks use; T2 freezes it)."""
    return NmBaseline(dep, fit, min_support)


def nm_anomaly(dep: pd.DataFrame, fit: np.ndarray, min_support: int = _proxylib.MIN_SUPPORT,
               exclude_month: int | None = None) -> pd.Series:
    """Every departure's NM proxy anomaly against the fit medians (``exclude_month`` left out), as a Series."""
    an = NmBaseline(dep, fit, min_support).anomaly(exclude_month)
    return pd.Series(an, index=dep.index, name="nm_anomaly")


# --- feature sets at an anchor -------------------------------------------------------------------

STATIC = ("hour_local", "minute_of_day", "dow", "is_weekend", "month", "day_of_year", "sched_hour_local")


def static_at(q: pd.DataFrame, a: np.ndarray) -> pd.DataFrame:
    """Calendar at the anchor in airport local time, and the scheduled local hour."""
    loc = local_naive(to_timestamps(a, q.index), q["apt"])
    sched = local_naive(q["SCHED_TIME_UTC_mvt"], q["apt"])
    return pd.DataFrame(
        {
            "hour_local": loc.dt.hour,
            "minute_of_day": loc.dt.hour * 60 + loc.dt.minute,
            "dow": loc.dt.dayofweek,
            "is_weekend": loc.dt.dayofweek.ge(5).astype("float64").where(loc.notna()),
            "month": loc.dt.month,
            "day_of_year": loc.dt.dayofyear,
            "sched_hour_local": sched.dt.hour,
        },
        index=q.index,
    ).astype("float32")


def plan_at(q: pd.DataFrame, a: np.ndarray | None = None) -> pd.DataFrame:
    """The filed plan against the schedule (minutes); at H0 also the realised off-block delay, floored to minutes."""
    planned = planned_offblock(q)
    sched = secs(q["SCHED_TIME_UTC_mvt"])
    cols = {"plan_minus_sched_min": (planned - sched) / 60.0}
    if a is not None:
        cols["delay_vs_plan_min"] = np.floor((a - planned) / 60.0)
        cols["delay_vs_sched_min"] = np.floor((a - sched) / 60.0)
    return pd.DataFrame(cols, index=q.index).astype("float32")


def shadow_departures(q: pd.DataFrame, a: np.ndarray) -> pd.DataFrame:
    """The query departures with ``MVT_TIME_UTC_mvt`` replaced by the anchor: all a takeoff-anchored block reads.

    Only the columns the weather blocks read are kept (airport, runway, stand), so nothing else of the row,
    and never the takeoff, can reach them.
    """
    return pd.DataFrame(
        {
            "apt": q["apt"],
            "RUNWAY_mvt": q["RUNWAY_mvt"],
            "STAND_mvt": q["STAND_mvt"],
            "MVT_TIME_UTC_mvt": to_timestamps(a, q.index),
        },
        index=q.index,
    )


def wx_at(q: pd.DataFrame, a: np.ndarray, params: dict | None = None) -> pd.DataFrame:
    """METAR known at ``a``, the windows ending there, and the TAF in force at ``a`` (issued at or before it).

    Calls the challenge blocks `wx_now`, `wx_past` and `taf_now` on shadow frames whose takeoff column is the
    query time. Those read the query time only through that column (weather.py `_departures`, weather_fcst.py
    `_departures`), and every report they use is observed or issued at or before it. The METAR side is read at
    ``a - METAR_LAG_S`` (a report observed at ``t`` is disseminated a few minutes later), so a METAR observed
    after ``a - METAR_LAG_S`` never enters; the TAF side at ``a`` itself (issue time <= a; validity time a).
    """
    a = np.asarray(a, dtype="float64")
    ctx = FeatureContext(dep_index=q.index, params=dict(params or {}))
    metar_side = shadow_departures(q, a - METAR_LAG_S)
    taf_side = shadow_departures(q, a)
    return pd.concat([wx_now(metar_side, ctx), wx_past(metar_side, ctx), taf_now(taf_side, ctx)], axis=1)


TRAFFIC = (
    "dep_apt_b15", "dep_apt_b60", "dep_rwy_b15", "dep_rwy_b30",
    "arr_apt_b15", "arr_apt_b60", "arr_rwy_b15",
    "arr_taxiing",
    "gap_last_dep_rwy_s", "gap_last_arr_rwy_s",
    "rwy_distinct_b60", "rwy_own_share_b60", "rwy_own_dominant_b60", "rwy_config_age_min",
    "arr_anom_b30_mean", "arr_anom_b30_n", "arr_anom_b60_mean", "arr_anom_b60_n",
    "stand_min_since_inblock",
)


def traffic_at(mvt: pd.DataFrame, q_index: pd.Index, a: np.ndarray) -> pd.DataFrame:
    """Other flights' movement records up to ``a``: counts, the state at ``a``, the runway mix, ground conditions, the stand."""
    dep = departures(mvt)
    arr = arrivals(mvt)
    q = mvt.loc[q_index]
    dep_t = pd.Series(secs(dep["MVT_TIME_UTC_mvt"]), index=dep.index)
    dep_apt = dep["apt"].astype("string")
    dep_rwy = runway_key(dep)
    arr_t = secs(arr["MVT_TIME_UTC_mvt"])
    arr_in = secs(arr["BLOCK_TIME_UTC_mvt"])
    arr_apt = arr["apt"].astype("string")
    arr_rwy = runway_key(arr)

    q_apt = q["apt"].astype("string").to_numpy(dtype=object)
    q_rwy = runway_key(q).to_numpy(dtype=object)
    own_t = dep_t.reindex(q_index).to_numpy(dtype="float64")

    out = {
        "dep_apt_b15": backward_count(dep_t, dep_apt, a, q_apt, 900, own_t),
        "dep_apt_b60": backward_count(dep_t, dep_apt, a, q_apt, 3600, own_t),
        "dep_rwy_b15": backward_count(dep_t, dep_rwy, a, q_rwy, 900, own_t),
        "dep_rwy_b30": backward_count(dep_t, dep_rwy, a, q_rwy, 1800, own_t),
        "arr_apt_b15": backward_count(arr_t, arr_apt, a, q_apt, 900),
        "arr_apt_b60": backward_count(arr_t, arr_apt, a, q_apt, 3600),
        "arr_rwy_b15": backward_count(arr_t, arr_rwy, a, q_rwy, 900),
        "arr_taxiing": in_progress(arr_t, arr_in, arr_apt, a, q_apt, max_dur=ARR_TAXIING_MAX_S),
        "gap_last_dep_rwy_s": np.minimum(last_gap(dep_t, dep_rwy, a, q_rwy, q_index), GAP_CAP_S),
        "gap_last_arr_rwy_s": np.minimum(
            last_gap(pd.Series(arr_t, index=arr.index), arr_rwy, a, q_rwy, q_index), GAP_CAP_S
        ),
    }
    rwy_raw = dep["RUNWAY_mvt"].astype("string").to_numpy(dtype=object)
    q_rwy_raw = q["RUNWAY_mvt"].astype("string").to_numpy(dtype=object)
    own_rwy_raw = pd.Series(rwy_raw, index=dep.index).reindex(q_index).to_numpy(dtype=object)
    distinct, share, dominant = runway_mix(
        dep_t.to_numpy(), dep_apt.to_numpy(dtype=object), rwy_raw, a, q_apt, q_rwy_raw, 3600, own_t, own_rwy_raw
    )
    out["rwy_distinct_b60"] = distinct
    out["rwy_own_share_b60"] = share
    out["rwy_own_dominant_b60"] = dominant
    out["rwy_config_age_min"] = config_age_minutes(
        dep_t.to_numpy(), dep_apt.to_numpy(dtype=object), rwy_raw, a, q_apt, own_t, own_rwy_raw
    )

    anomaly = arrival_anomaly(mvt).reindex(arr.index).to_numpy(dtype="float64")
    for win, back in (("b30", 1800), ("b60", 3600)):
        n, mean, _ = backward_stats(arr_in, arr_apt.to_numpy(dtype=object), anomaly, a, q_apt, back)
        out[f"arr_anom_{win}_mean"] = mean
        out[f"arr_anom_{win}_n"] = n

    out["stand_min_since_inblock"] = _stand_minutes(q, arr, a)
    frame = pd.DataFrame(out, index=q_index).astype("float32")
    return _mask(frame, a)


def _stand_minutes(q: pd.DataFrame, arr: pd.DataFrame, a: np.ndarray) -> np.ndarray:
    """Minutes since the last arrival blocked in at the flight's stand, in-block at or before ``a``; capped at a day."""
    out = np.full(len(q), np.nan)
    ok = np.isfinite(a) & q["STAND_mvt"].notna().to_numpy()
    if not ok.any():
        return out
    sub = q.loc[ok]
    merged, _ = stand_asof(shadow_departures(sub, a[ok]), arr, direction="backward", allow_exact_matches=True)
    gap = (merged["t"] - merged["arr_inblock"]).to_numpy(dtype="float64") / 60.0
    out[ok] = np.minimum(gap, STAND_CAP_MIN)
    return out


NM = tuple(
    f"nm_{lvl}_{win}_{stat}" for lvl in ("apt", "rwy") for win in ("b30", "b60") for stat in ("anom_mean", "anom_share5", "push_delay_mean")
) + ("backlog_apt", "backlog_rwy", "backlog_pier")


def nm_at(mvt: pd.DataFrame, q_index: pd.Index, a: np.ndarray, baseline: NmBaseline) -> pd.DataFrame:
    """NM data of other flights already airborne at ``a``, and the plan backlog.

    For others with a takeoff in ``[a - w, a]`` (w = 30, 60 min), per airport and per own runway: the mean
    and the share above +5 min of the NM proxy anomaly (``baseline``: the table that leaves out the query's own
    planned month, see `NmBaseline`), and their mean push delay ``AOBT_3 - planned``; both only where the proxy
    is a plausible reading, so ``AOBT_3 <= takeoff <= a``.
    Plan backlog: others whose planned off-block is at or before ``a`` and whose takeoff is not (airport, own
    runway, pier), each counted for at most three hours. ``backlog_rwy`` and ``backlog_pier`` group those
    not-yet-airborne flights by their *recorded* runway and stand: tier O* (their planned runway and the stand
    they are parked at are assumed known; the data holds the actual ones; `O_STAR_FUTURE_ATTRIBUTES`, test T2b).
    No NM-push queue (``AOBT_3 <= a`` and not airborne): AOBT_3 is back-computed from the takeoff at LTFM and
    partly elsewhere.
    """
    dep = departures(mvt)
    if not baseline.index.equals(dep.index):
        raise ValueError("nm_at: the baseline must be built on this frame's departures")
    q = mvt.loc[q_index]
    a = np.asarray(a, dtype="float64")
    t = secs(dep["MVT_TIME_UTC_mvt"])
    planned = planned_offblock(dep)
    push_raw = np.clip(secs(dep["AOBT_3_flt"]) - planned, *PUSH_DELAY_CLIP)
    pos = pd.Series(np.arange(len(dep)), index=dep.index).reindex(q_index).to_numpy()
    has = np.isfinite(pos)
    p = np.clip(np.nan_to_num(pos, nan=0).astype("int64"), 0, max(len(dep) - 1, 0))

    def own(values: np.ndarray) -> np.ndarray:
        return np.where(has, values[p], np.nan) if len(dep) else np.full(len(q_index), np.nan)

    groups = {
        "apt": (dep["apt"].astype("string").to_numpy(dtype=object), q["apt"].astype("string").to_numpy(dtype=object)),
        "rwy": (runway_key(dep).to_numpy(dtype=object), runway_key(q).to_numpy(dtype=object)),
        "pier": (pier_key(dep).to_numpy(dtype=object), pier_key(q).to_numpy(dtype=object)),
    }
    out = {c: np.full(len(q_index), np.nan) for c in NM}
    own_t = own(t)
    keys = [baseline.key(m) for m in baseline_month(q)]
    for key in dict.fromkeys(keys):
        rows = np.flatnonzero([k == key for k in keys])
        rows = rows[np.isfinite(a[rows])]
        if not len(rows):
            continue
        an = baseline.anomaly(key)
        push = np.where(np.isfinite(an), push_raw, np.nan)
        qa = a[rows]
        with np.errstate(invalid="ignore"):  # only the events that can sit in one of these queries' windows
            sel = (t >= qa.min() - 3600) & (t <= qa.max())
        for lvl in ("apt", "rwy"):
            rg, qg = groups[lvl]
            for win, back in (("b30", 1800), ("b60", 3600)):
                _, mean, share = backward_stats(
                    t[sel], rg[sel], an[sel], qa, qg[rows], back, own_t[rows], own(an)[rows], high=HIGH_ANOMALY_S
                )
                _, push_mean, _ = backward_stats(t[sel], rg[sel], push[sel], qa, qg[rows], back, own_t[rows], own(push)[rows])
                out[f"nm_{lvl}_{win}_anom_mean"][rows] = mean
                out[f"nm_{lvl}_{win}_anom_share5"][rows] = share
                out[f"nm_{lvl}_{win}_push_delay_mean"][rows] = push_mean
    for lvl in ("apt", "rwy", "pier"):
        rg, qg = groups[lvl]
        out[f"backlog_{lvl}"] = in_progress(planned, t, rg, a, qg, own(planned), own_t, max_dur=BACKLOG_MAX_S)
    frame = pd.DataFrame(out, index=q_index)[list(NM)].astype("float32")
    return _mask(frame, a)


FEED = (
    "feed_queue_apt", "feed_queue_rwy", "feed_queue_pier",
    "feed_apt_b30_taxi_mean", "feed_apt_b30_n", "feed_apt_b60_taxi_mean", "feed_apt_b60_n",
    "feed_rwy_b30_taxi_mean", "feed_rwy_b30_n", "feed_rwy_b60_taxi_mean", "feed_rwy_b60_n",
)


def feed_at(mvt: pd.DataFrame, q_index: pd.Index, a: np.ndarray, offblocks: pd.Series) -> pd.DataFrame:
    """The airport's own off-block records of other flights at ``a`` (lab, 2025 only).

    The taxiing-out queue: others with a recorded off-block at or before ``a`` and no takeoff yet (airport,
    own runway, pier; counted for at most an hour after the off-block). The recorded taxi-out of others that
    took off in the last 30/60 minutes (``takeoff - block``, clipped at an hour; negative ones left out), mean
    and count, per airport and own runway. Ranking rows (no records) are NaN. ``feed_queue_rwy`` and
    ``feed_queue_pier`` group the not-yet-airborne flights by their *recorded* runway and stand: tier O*
    (`O_STAR_FUTURE_ATTRIBUTES`, test T2b).
    """
    dep = departures(mvt)
    q = mvt.loc[q_index]
    t = secs(dep["MVT_TIME_UTC_mvt"])
    b = offblocks.reindex(dep.index).to_numpy(dtype="float64")
    with np.errstate(invalid="ignore"):
        taxi = np.where((t - b) >= 0, np.minimum(t - b, FEED_TAXI_CAP_S), np.nan)
    pos = pd.Series(np.arange(len(dep)), index=dep.index).reindex(q_index).to_numpy()
    has = np.isfinite(pos)
    p = np.clip(np.nan_to_num(pos, nan=0).astype("int64"), 0, max(len(dep) - 1, 0))

    def own(values: np.ndarray) -> np.ndarray:
        return np.where(has, values[p], np.nan) if len(dep) else np.full(len(q_index), np.nan)

    groups = {
        "apt": (dep["apt"].astype("string").to_numpy(dtype=object), q["apt"].astype("string").to_numpy(dtype=object)),
        "rwy": (runway_key(dep).to_numpy(dtype=object), runway_key(q).to_numpy(dtype=object)),
        "pier": (pier_key(dep).to_numpy(dtype=object), pier_key(q).to_numpy(dtype=object)),
    }
    out = {}
    for lvl in ("apt", "rwy", "pier"):
        rg, qg = groups[lvl]
        out[f"feed_queue_{lvl}"] = in_progress(b, t, rg, a, qg, own(b), own(t), max_dur=FEED_QUEUE_MAX_S)
    for lvl in ("apt", "rwy"):
        rg, qg = groups[lvl]
        for win, back in (("b30", 1800), ("b60", 3600)):
            n, mean, _ = backward_stats(t, rg, taxi, a, qg, back, own(t), own(taxi))
            out[f"feed_{lvl}_{win}_taxi_mean"] = mean
            out[f"feed_{lvl}_{win}_n"] = n
    frame = pd.DataFrame(out, index=q_index)[list(FEED)].astype("float32")
    if "split" in q.columns:
        frame.loc[q["split"].astype("string").eq("rank").fillna(False).to_numpy(dtype=bool)] = np.nan
    return _mask(frame, a)


def _mask(frame: pd.DataFrame, a: np.ndarray) -> pd.DataFrame:
    """Every column NaN where there is no anchor (no plan at H1; no off-block record at H0)."""
    frame.loc[~np.isfinite(np.asarray(a, dtype="float64"))] = np.nan
    return frame
