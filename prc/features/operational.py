"""The operational ladder: taxi-out inputs available *before* the aircraft moves.

Every challenge block is anchored on the takeoff. These are anchored on a prediction time instead (see
`prc.features._oplib`):

* ``op_h1_*`` -- H1, 15 minutes before the planned off-block (``IOBT_flt``, else ``LOBT_flt``, else the
  schedule). Board-scorable: every input exists for 2026.
* ``op_h0_*`` -- H0, at the recorded off-block (at pushback). Lab only: it needs the airport's own off-block
  records, which exist for 2025 only (2026 departure block times are blanked).
* ``op_h1_feed_lab`` -- the airport's own off-block records of OTHER flights at H1. Lab only, 2025 only.

Operational tiers (report/feature_table.md): **O** = available to an airport or NM before the movement;
**O*** = available under an assumption the data cannot check. Every runway-derived input is O*: the planned
departure runway is assumed known, while the data holds the actual one (it matches the hour's dominant
departure runway on 66 % of departures). The lab blocks and every ``op_h0_*`` block declare the target file
as an external input (`_oplib.LAB_EXTERNAL`) and read only its block-time column, through
`_oplib.load_offblocks`. No board/refit config may include them (tests/test_operational.py).

The flight's own takeoff, own ``AOBT_3_flt`` / ``ARVT_3_flt`` and ``EOBT_1_flt`` are never inputs; own events
in the reference tables are removed explicitly, and a fit row's own NM reading never enters the baseline its
windows read (month-out, `_oplib.NmBaseline`) (tests T1/T1b/T1c, perturbed rows kept in the fit). Every other
flight's event *time* enters only when the event happened at or before the anchor (test T2). Exception, tier O*:
the plan backlog and the taxiing-out queue per runway and pier group the other, not yet airborne flights by
their recorded runway and stand, which are certain only at their takeoff (`_oplib.O_STAR_FUTURE_ATTRIBUTES`;
test T2b censors those attributes and allows only these columns to move). METARs count from
``_oplib.METAR_LAG_S`` after their observation.

Auto-imported by the registry: no work at import time. The helper modules are imported here as modules so
that `build._related_modules` (one level deep) hashes every one of them into these blocks' cache keys.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc import paths
from prc.features import (  # noqa: F401  (module objects: hashed into these blocks' cache keys)
    _arrlib,
    _oplib,
    _proxylib,
    _wxlib,
    congestion,
    core,
    stand_runway,
    temporal,
    turnaround,
    weather,
    weather_fcst,
)
from prc.features.registry import FeatureContext, feature

WX_SOURCE = (*weather.WX_NOW, *weather.WX_PAST, *weather_fcst.TAF_NOW)
WX_EXTERNAL = (paths.METAR_FILE, *weather_fcst.TAF_FILES)


def _names(prefix: str, cols) -> tuple[str, ...]:
    return tuple(f"{prefix}{c}" for c in cols)


def _anchor(mvt: pd.DataFrame, ctx: FeatureContext, horizon: str) -> tuple[pd.DataFrame, np.ndarray]:
    q = mvt.loc[ctx.dep_index]
    if horizon == "h1":
        return q, _oplib.anchor_h1(q)
    return q, _oplib.anchor_h0(q, _oplib.load_offblocks())


def _prefixed(frame: pd.DataFrame, prefix: str, index: pd.Index) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [f"{prefix}{c}" for c in out.columns]
    return out.reindex(index).astype("float32")


def _fit_rows(mvt: pd.DataFrame, fit_mask: pd.Series | None) -> np.ndarray:
    dep = _oplib.departures(mvt)
    if fit_mask is None:
        return np.zeros(len(dep), dtype=bool)
    return fit_mask.reindex(dep.index).fillna(False).to_numpy(dtype=bool)


def _baseline(mvt: pd.DataFrame, fit_mask: pd.Series | None, params: dict) -> "_oplib.NmBaseline":
    """The NM proxy baseline over the fold's fit rows, month-out per query (`_oplib.NmBaseline`)."""
    dep = _oplib.departures(mvt)
    return _oplib.nm_baseline(dep, _fit_rows(mvt, fit_mask), int(params.get("min_support", _proxylib.MIN_SUPPORT)))


# --- H1: 15 minutes before the planned off-block -------------------------------------------------

@feature(
    group="op_h1",
    outputs=_names("op_h1_", _oplib.STATIC),
    doc="Calendar at H1 (15 min before the planned off-block) in local time, and the scheduled local hour.",
)
def op_h1_static(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """Local hour, minute, weekday, month and day of year at the H1 anchor; the scheduled local hour.

    Operational availability: tier O (the flight plan and the schedule exist hours before the flight).
    No takeoff-based calendar.
    """
    q, a = _anchor(mvt, ctx, "h1")
    return _prefixed(_oplib._mask(_oplib.static_at(q, a), a), "op_h1_", ctx.dep_index)


@feature(
    group="op_h1",
    outputs=_names("op_h1_", ("plan_minus_sched_min",)),
    doc="The filed plan against the schedule at H1: planned off-block minus schedule, minutes.",
)
def op_h1_plan(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """Planned off-block (IOBT, else LOBT, else schedule) minus the schedule, in minutes (positive = planned late).

    Operational availability: tier O (the filed flight plan). Nothing from EOBT_1, whose update time is not
    in the data.
    """
    q, _ = _anchor(mvt, ctx, "h1")
    return _prefixed(_oplib.plan_at(q), "op_h1_", ctx.dep_index)


@feature(
    group="op_h1",
    outputs=_names("op_h1_", WX_SOURCE),
    external=WX_EXTERNAL,
    doc="METAR known at H1 (observed at least 5 min before), the windows ending there, and the TAF in force at H1 (issued at or before it).",
)
def op_h1_wx(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """Weather at the H1 anchor: `wx_now`, `wx_past` and `taf_now` called with the anchor as their query time.

    Operational availability: tier O (METAR and TAF are public in real time); the head- and crosswind on the
    own runway are O* (planned runway assumed known). No ERA5 (tier P). The TAF is read at the anchor's own
    validity time; the METAR side ``_oplib.METAR_LAG_S`` before it (dissemination delay).
    """
    q, a = _anchor(mvt, ctx, "h1")
    return _prefixed(_oplib._mask(_oplib.wx_at(q, a, ctx.params), a), "op_h1_", ctx.dep_index)


@feature(
    group="op_h1",
    outputs=_names("op_h1_", _oplib.TRAFFIC),
    doc="Other flights' movement records up to H1: takeoffs, landings, arrivals taxiing, runway mix, ground conditions, the stand.",
)
def op_h1_traffic(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """Movement records of OTHER flights up to the H1 anchor, every count excluding the flight itself.

    Takeoffs at the airport (15/60 min) and on the own runway (15/30 min); landings at the airport (15/60) and
    on the own runway (15); arrivals landed but not in block; seconds since the last takeoff (never the own)
    and landing on the own runway; distinct departure runways, the own runway's share and dominance over the
    last hour; minutes since the dominant departure runway changed (15-min buckets complete by the anchor);
    the taxi-in anomaly of arrivals that blocked in during the last 30/60 min (`_arrlib`: a target-free
    archive median over every arrival in the frame, 2026 included, i.e. transductive in time); minutes since
    the last in-block at the flight's stand.

    Operational availability: tier O (the airport's movement records); every own-runway input is O*: the
    planned departure runway is assumed known; the data holds the actual one (it matches the hour's dominant
    departure runway on 66 % of departures).
    """
    q, a = _anchor(mvt, ctx, "h1")
    return _prefixed(_oplib.traffic_at(mvt, ctx.dep_index, a), "op_h1_", ctx.dep_index)


@feature(
    group="op_h1",
    outputs=_names("op_h1_", _oplib.NM),
    uses_target=True,
    doc="NM readings of flights already airborne at H1 (proxy anomaly, push delay) and the plan backlog; fold-safe baseline.",
)
def op_h1_nm(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """NM data of OTHER flights already airborne at H1, and the plan backlog.

    Per airport and own runway, over the last 30/60 min: the mean and the share above +5 min of the NM proxy
    anomaly (takeoff - AOBT_3 against its (airport, stand, runway) median), and the mean push delay
    AOBT_3 - planned. The median comes from the fold's fit rows only (``ctx.fit_mask``; no target is read),
    so the block is recomputed per fold, and it leaves out the fit rows of the query's own planned month, so a
    fit row's own reading never enters it (`_oplib.NmBaseline`). Plan backlog: others planned off at or before
    H1 and not yet airborne (airport, own runway, pier). No NM-push queue.

    Operational availability: tier O (NM's flight data of completed flights; the filed plans). Own-runway
    columns are O* (planned runway assumed known). ``backlog_rwy`` and ``backlog_pier`` are O* in a second way:
    they group the OTHER, not yet airborne flights by their recorded runway and stand (their planned runways and
    parking stands assumed known; the data holds the actual ones).
    """
    q, a = _anchor(mvt, ctx, "h1")
    return _prefixed(_oplib.nm_at(mvt, ctx.dep_index, a, _baseline(mvt, ctx.fit_mask, ctx.params)), "op_h1_", ctx.dep_index)


@feature(
    group="op_lab_h1",
    outputs=_names("op_h1_", _oplib.FEED),
    external=_oplib.LAB_EXTERNAL,
    doc="LAB, 2025 only: the airport's own off-block records of other flights at H1 (taxiing-out queue, recent taxi-outs).",
)
def op_h1_feed_lab(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """The airport's real-time off-block feed (A-CDM AOBT) of OTHER flights at H1.

    The taxiing-out queue (pushed back, not airborne) per airport, own runway and pier, and the mean recorded
    taxi-out of others that took off in the last 30/60 min with their counts.

    Operational availability: tier O for an airport (its own AOBT feed); ``feed_queue_rwy`` and
    ``feed_queue_pier`` are O* (the queued flights' recorded runway and stand stand in for the planned ones).
    Lab only: needs the airport's own off-block records of OTHER flights, 2025 only (2026 rows are NaN). Never
    in a board config.
    """
    q, a = _anchor(mvt, ctx, "h1")
    return _prefixed(_oplib.feed_at(mvt, ctx.dep_index, a, _oplib.load_offblocks()), "op_h1_", ctx.dep_index)


# --- H0: at the recorded off-block (lab) ------------------------------------------------------------



@feature(
    group="op_lab_h0",
    outputs=_names("op_h0_", _oplib.STATIC),
    external=_oplib.LAB_EXTERNAL,
    doc="LAB, 2025 only: calendar at the recorded off-block (H0) and the scheduled local hour.",
)
def op_h0_static(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """Calendar at H0 (the recorded off-block) and the scheduled local hour.

    Operational availability: tier O at pushback. Lab only: needs the airport's own off-block records of
    OTHER flights, 2025 only (the anchor itself is the recorded off-block; NaN for 2026).
    """
    q, a = _anchor(mvt, ctx, "h0")
    return _prefixed(_oplib._mask(_oplib.static_at(q, a), a), "op_h0_", ctx.dep_index)


@feature(
    group="op_lab_h0",
    outputs=_names("op_h0_", ("plan_minus_sched_min", "delay_vs_plan_min", "delay_vs_sched_min")),
    external=_oplib.LAB_EXTERNAL,
    doc="LAB, 2025 only: the plan against the schedule, and the realised off-block delay against both (whole minutes).",
)
def op_h0_plan(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """Plan minus schedule, and at the push the realised delay: floor((H0 - planned) / 60), floor((H0 - schedule) / 60).

    Floored to whole minutes so the stale fingerprint (block == schedule) is blurred; for a stale record the
    H0 anchor is the schedule, not the push, which is why H0 is reported on the operational population only
    (taxi < 3,600 s and NM-matched).

    Operational availability: tier O at pushback. Lab only: needs the airport's own off-block records of
    OTHER flights, 2025 only.
    """
    q, a = _anchor(mvt, ctx, "h0")
    return _prefixed(_oplib._mask(_oplib.plan_at(q, a), a), "op_h0_", ctx.dep_index)


@feature(
    group="op_lab_h0",
    outputs=_names("op_h0_", WX_SOURCE),
    external=(*WX_EXTERNAL, *_oplib.LAB_EXTERNAL),
    doc="LAB, 2025 only: METAR (observed at least 5 min before), windows and TAF at the recorded off-block (H0).",
)
def op_h0_wx(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """Weather at H0: `wx_now`, `wx_past` and `taf_now` with the recorded off-block as their query time.

    Operational availability: tier O at pushback (own-runway wind O*). Lab only: needs the airport's own
    off-block records of OTHER flights, 2025 only.
    """
    q, a = _anchor(mvt, ctx, "h0")
    return _prefixed(_oplib._mask(_oplib.wx_at(q, a, ctx.params), a), "op_h0_", ctx.dep_index)


@feature(
    group="op_lab_h0",
    outputs=_names("op_h0_", _oplib.TRAFFIC),
    external=_oplib.LAB_EXTERNAL,
    doc="LAB, 2025 only: other flights' movement records up to H0.",
)
def op_h0_traffic(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """`op_h1_traffic` at H0 (the recorded off-block), every count excluding the flight itself.

    Operational availability: tier O at pushback (own-runway inputs O*). Lab only: needs the airport's own
    off-block records of OTHER flights, 2025 only.
    """
    q, a = _anchor(mvt, ctx, "h0")
    return _prefixed(_oplib.traffic_at(mvt, ctx.dep_index, a), "op_h0_", ctx.dep_index)


@feature(
    group="op_lab_h0",
    outputs=_names("op_h0_", _oplib.NM),
    uses_target=True,
    external=_oplib.LAB_EXTERNAL,
    doc="LAB, 2025 only: NM readings of flights airborne at H0 and the plan backlog; fold-safe baseline.",
)
def op_h0_nm(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """`op_h1_nm` at H0; the proxy median comes from the fold's fit rows only (``ctx.fit_mask``), month-out.

    Operational availability: tier O at pushback (own-runway columns O*; ``backlog_rwy``/``backlog_pier`` O* for
    the other flights' recorded runway and stand). Lab only: needs the airport's own off-block records of
    OTHER flights, 2025 only.
    """
    q, a = _anchor(mvt, ctx, "h0")
    return _prefixed(_oplib.nm_at(mvt, ctx.dep_index, a, _baseline(mvt, ctx.fit_mask, ctx.params)), "op_h0_", ctx.dep_index)


@feature(
    group="op_lab_h0",
    outputs=_names("op_h0_", _oplib.FEED),
    external=_oplib.LAB_EXTERNAL,
    doc="LAB, 2025 only: the airport's own off-block records of other flights at H0.",
)
def op_h0_feed_lab(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """`op_h1_feed_lab` at H0: the taxiing-out queue and recent recorded taxi-outs of OTHER flights at the push.

    Operational availability: tier O for an airport (its own AOBT feed); ``feed_queue_rwy`` and
    ``feed_queue_pier`` O* (the queued flights' recorded runway and stand). Lab only: needs the airport's own
    off-block records of OTHER flights, 2025 only.
    """
    q, a = _anchor(mvt, ctx, "h0")
    return _prefixed(_oplib.feed_at(mvt, ctx.dep_index, a, _oplib.load_offblocks()), "op_h0_", ctx.dep_index)
