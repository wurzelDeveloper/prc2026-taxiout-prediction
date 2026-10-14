"""The long-horizon operational models: taxi-out inputs known hours before the planned off-block.

The operational ladder (`prc.features.operational`) predicts 15 minutes before the planned off-block (H1) or at the
push (H0). These blocks serve the owner's two earlier questions (9 Oct): what is the prediction from longer-term data
a few hours or a day before the planned movement? Anchors and conventions are in `prc.features._horizon`:

* ``P`` = the planned off-block: the published schedule, on every row (findings 10.35). Never the IOBT, which equals
  NM's last known off-block time on 99.45 % of the NM-matched 2025 departures, carries most of a late flight's delay
  and has no filing time in the data; never NM's last known off-block time, whose value is the latest update;
* ``a = P - lead``, lead = 3 h (``op_hl3_*``) or 24 h (``op_hl24_*``).

Lead-independent blocks (``op_hl_*``; group ``op_hl``) read plan data only, so their values are the same at both
leads and they are computed once: the calendar at ``P`` (no day of year: it maps 2025's moving holidays onto 2026
dates), whether a stand is allocated, and the schedule demand around ``P``. Lead-dependent blocks (groups ``op_hl3`` /
``op_hl24``): the weather known at ``a`` (persistence METAR and the TAF issued by ``a`` for the hour of ``P``) and the
recent history (NM's readings of OTHER flights that took off in the 3 h / 24 h before ``a``; fold-dependent like
``op_h1_nm``).

Operational tiers, column by column in `prc.features._long_tiers` (the configs and tests check them): **O** for the
calendar, the schedule, the weather and the NM history (NM's records of completed flights). Hours ahead the flight's
own recorded runway is the outcome of a future runway configuration, so every own-runway input is O* at best and the
long deliverables (the runway-blind refit rungs, r3 or r4 per lead by rule R of findings 10.35:
``op_hl3_r3_rwyblind`` / ``op_hl3_r4_rwyblind``, ``op_hl24_r3_rwyblind`` / ``op_hl24_r4_rwyblind``) drop them; the
runway-aware rungs are an O* upper bound. The own recorded stand is O* at both leads
(``op_hl24_r4_arch_rwyblind_pier`` keeps the pier only). The schedule demand per runway and pier is O* a second way
(the other flights' planned runway and stand assumed known, `_horizon.O_STAR_FUTURE_ATTRIBUTES_LONG`). Tier C: the
missing level of the ``_flt`` columns of `aircraft_identity` marks the challenge's NM join, so the long ladder is read
on the NM-joined OP population, where they are constant; these blocks read no field of the flight's own NM record.

The flight's own takeoff, its own NM actuals (NM's ``AOBT_3`` is back-computed from the takeoff at LTFM and partly at
EDDM, LEBL, LSZH and LEMD), its own off-block record, NM's last known off-block time, EOBT_1 and the IOBT of any flight
are never inputs (tests/test_operational_long.py: T1 moves each own field and asserts bit-identical outputs, the
perturbed rows kept in the fit; the IOBT of every flight is shifted, blanked and removed likewise; T2 censors every
event after the anchor). None of these blocks is lab-only: every input exists for 2026.

Auto-imported by the registry: no work at import time. The helper modules are imported as modules so that
`build._related_modules` (one level deep) hashes every one of them into these blocks' cache keys; `_oplib` and
`operational` are not edited, so the H1/H0 blocks keep their hashes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc import paths
from prc.features import (  # noqa: F401  (module objects: hashed into these blocks' cache keys)
    _horizon,
    _oplib,
    _proxylib,
    _wxlib,
    congestion,
    core,
    stand_runway,
    temporal,
    weather,
    weather_fcst,
)
from prc.features.registry import FeatureContext, feature

WX_EXTERNAL = (paths.METAR_FILE, *weather_fcst.TAF_FILES)


def _names(prefix: str, cols) -> tuple[str, ...]:
    return tuple(f"{prefix}{c}" for c in cols)


def _prefixed(frame: pd.DataFrame, prefix: str, index: pd.Index) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [f"{prefix}{c}" for c in out.columns]
    return out.reindex(index).astype("float32")


PLAN_COLUMNS = ("apt", "SCHED_TIME_UTC_mvt", "STAND_mvt", "RUNWAY_mvt")
"""The columns `_plan` takes of the query departures: neither the takeoff nor an NM field. (The demand and the history
read the other flights from the whole frame and remove the flight itself from every window.)"""


def _plan(mvt: pd.DataFrame, ctx: FeatureContext) -> tuple[pd.DataFrame, np.ndarray]:
    q = mvt.loc[ctx.dep_index, list(PLAN_COLUMNS)]
    return q, _horizon.planned_offblock_long(q)


def _wx(mvt: pd.DataFrame, ctx: FeatureContext, horizon: str) -> pd.DataFrame:
    q, planned = _plan(mvt, ctx)
    a = planned - _horizon.LEADS_S[horizon]
    return _prefixed(_horizon.wx_long(q, a, planned, ctx.params), f"op_{horizon}_", ctx.dep_index)


def _history(mvt: pd.DataFrame, ctx: FeatureContext, fit_mask: pd.Series | None, horizon: str) -> pd.DataFrame:
    q, planned = _plan(mvt, ctx)
    a = planned - _horizon.LEADS_S[horizon]
    dep = _oplib.departures(mvt)
    fit = np.zeros(len(dep), dtype=bool) if fit_mask is None else fit_mask.reindex(dep.index).fillna(False).to_numpy(dtype=bool)
    baseline = _horizon.nm_baseline_long(dep, fit, int(ctx.params.get("min_support", _proxylib.MIN_SUPPORT)))
    return _prefixed(_horizon.history_at(mvt, ctx.dep_index, a, baseline), f"op_{horizon}_", ctx.dep_index)


# --- lead-independent: plan data only ------------------------------------------------------------------

@feature(
    group="op_hl",
    outputs=_names("op_hl_", _horizon.STATIC),
    doc="Calendar at the planned off-block (the schedule) in local time, and the scheduled local hour; no day of year.",
)
def op_hl_static(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """Local hour, minute, weekday and month at the planned off-block (the schedule); the scheduled local hour.

    Lead-independent (the schedule is known at both anchors). No day of year (findings 10.35: it maps 2025's moving
    holidays onto 2026 dates) and no takeoff-based calendar. Operational availability: tier O (the published
    schedule).
    """
    q, planned = _plan(mvt, ctx)
    return _prefixed(_horizon.static_long(q, planned), "op_hl_", ctx.dep_index)


@feature(
    group="op_hl",
    outputs=_names("op_hl_", _horizon.PLAN),
    doc="Whether the record has a stand allocated (no filed plan is read).",
)
def op_hl_plan(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """Whether the record has a stand; no filed plan (the IOBT minus the schedule was an output until 11 Oct).

    Operational availability: ``has_stand`` reads the own stand allocation (O* hours ahead). The IOBT is not read
    (findings 10.35: it carries most of a late flight's delay, the data hold no filing time, and its NaN is the
    challenge's NM-join flag, C). No explicit join flag is an output.
    """
    q, planned = _plan(mvt, ctx)
    return _prefixed(_horizon.plan_long(q, planned), "op_hl_", ctx.dep_index)


@feature(
    group="op_hl",
    outputs=_names("op_hl_", _horizon.DEMAND),
    doc="Other flights' scheduled departures and scheduled arrivals within +-30 / +-60 min of the planned off-block (the schedule), per airport, own runway, own pier.",
)
def op_hl_demand(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """Schedule demand around the own planned off-block ``P``, the flight itself excluded.

    Counts of other departures and of arrivals by their schedule within ``[P - w, P + w]`` (w = 30, 60 min), at the
    airport, on the own runway and at the own pier. Plan data only: nothing that happens after the anchor (and
    nothing of the own movement) is read, and no IOBT.

    Operational availability: tier O for the schedule totals at the airport. The own runway is O* (planned runway
    assumed known; dropped by the runway-blind rungs), the own pier a weak O*; the per-runway and per-pier counts group
    the OTHER flights by their recorded runway and stand, O* in a second way (`_horizon.O_STAR_FUTURE_ATTRIBUTES_LONG`).
    """
    q, planned = _plan(mvt, ctx)
    return _prefixed(_horizon.demand_at(mvt, ctx.dep_index, planned), "op_hl_", ctx.dep_index)


# --- 3 h before the planned off-block ----------------------------------------------------------------------

@feature(
    group="op_hl3",
    outputs=_names("op_hl3_", _horizon.WX_LONG),
    external=WX_EXTERNAL,
    doc="3 h ahead: METAR known at the anchor (persistence, windows ending there) and the TAF issued by then for the planned off-block.",
)
def op_hl3_wx(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """Weather at the 3 h anchor ``a = P - 3 h``: `weather.wx_now` / `wx_past` at ``a - METAR_LAG_S``, the TAF for ``P``.

    Operational availability: tier O (METAR and TAF are public in real time); head- and crosswind on the own runway
    are O*. No ERA5 (tier P).
    """
    return _wx(mvt, ctx, "hl3")


@feature(
    group="op_hl3",
    outputs=_names("op_hl3_", _horizon.HIST),
    uses_target=True,
    doc="3 h ahead: NM readings (proxy anomaly, push delay) of other flights airborne in the 3 h / 24 h before the anchor; fold-safe baseline.",
)
def op_hl3_hist(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """Recent history at the 3 h anchor: other flights' NM taxi-out readings in ``[a - 3 h, a]`` and ``[a - 24 h, a]``.

    Per airport and own runway: the count of readings, the mean and the share above +5 min of the proxy anomaly
    (takeoff - AOBT_3 against its (airport, stand, runway) median), and the mean push delay (AOBT_3 minus the
    schedule). The median comes from the fold's fit rows only (``ctx.fit_mask``; no target is read), leaving out the
    query's own planned month (the schedule's; `_horizon.NmBaselineLong`), so the block is recomputed per fold.

    Operational availability: tier O (NM's records of completed flights); own-runway columns O*.
    """
    return _history(mvt, ctx, ctx.fit_mask, "hl3")


# --- 24 h before the planned off-block ---------------------------------------------------------------------

@feature(
    group="op_hl24",
    outputs=_names("op_hl24_", _horizon.WX_LONG),
    external=WX_EXTERNAL,
    doc="24 h ahead: METAR known at the anchor (persistence, windows ending there) and the TAF issued by then for the planned off-block.",
)
def op_hl24_wx(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """Weather at the 24 h anchor ``a = P - 24 h``: the METAR side at ``a - METAR_LAG_S``, the TAF issued by ``a`` for ``P``.

    A 30 h TAF issued in the six hours before the anchor reaches ``P``; a 24 h TAF (Barcelona) mostly does not (NaN).
    Operational availability: tier O; head- and crosswind on the own runway O*.
    """
    return _wx(mvt, ctx, "hl24")


@feature(
    group="op_hl24",
    outputs=_names("op_hl24_", _horizon.HIST),
    uses_target=True,
    doc="24 h ahead: NM readings (proxy anomaly, push delay) of other flights airborne in the 3 h / 24 h before the anchor; fold-safe baseline.",
)
def op_hl24_hist(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """Recent history at the 24 h anchor: other flights' NM readings in ``[a - 3 h, a]`` and ``[a - 24 h, a]``.

    As `op_hl3_hist`, a day earlier: the windows cover the day before the flight's day. Fold-dependent (the baseline
    comes from ``ctx.fit_mask`` rows, month-out). Operational availability: tier O; own-runway columns O*.
    """
    return _history(mvt, ctx, ctx.fit_mask, "hl24")
