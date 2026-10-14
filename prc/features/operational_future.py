"""Traffic after the H1 anchor: what an airport's planning systems would hold about the next hour (findings 10.36).

An A-CDM airport plans the traffic around a flight's planned off-block. The arrival manager (AMAN) plans the landing
sequence, the departure manager (DMAN) the start-up and takeoff sequence (TSAT, TTOT), and the surface manager (the
A-SMGCS planning function) the routes and taxi times. The challenge data hold none of these plans, only what happened.
These blocks describe the OTHER flights' traffic after the H1 anchor ``a`` in two ways:

* ``op_h1_fut_plan`` -- tier **O***: the plans themselves, where the data hold them. These are the other departures'
  planned off-blocks (IOBT, else LOBT, else the schedule) and the other arrivals' scheduled times in ``(a, a + w]``
  (w = 15, 30, 60 min: columns ``w15``, ``w30``, ``w60``), per airport, own runway and pier. The other flights' runway
  and stand are their recorded ones, standing in for the planned runway and stand allocation (as the plan backlog of
  ``op_h1_nm``). The flights that operated are taken from the record (no cancellations): a mild hindsight, as in the
  long members' scheduled demand.
* ``op_h1_fut_rec``, ``op_h1_fut_nm``, ``op_h1_fut_adsb`` -- tier **F**: the REALISED movements of the other flights,
  standing in for the planning systems' forecasts. Each reads the past-window machinery of H1 at the read times
  ``a + 15`` (the planned off-block), ``a + 30`` and ``a + 45`` min (columns ``f15``, ``f30``, ``f45``):
  - ``fut_rec``: the airport's records (`_oplib.traffic_at`): takeoffs and landings at the airport and on the own
    runway, arrivals taxiing in, the runway mix and configuration, the arrivals' taxi-in anomaly;
  - ``fut_nm``: NM's readings of the flights that took off (`_oplib.nm_at`): their proxy taxi anomaly and push delay,
    and the plan backlog; fold-dependent like ``op_h1_nm`` (month-out baseline from the fit rows);
  - ``fut_adsb``: the public ADS-B surface picture (`_adsblib.surface_at`): aircraft taxiing out and in, the runway
    queue, remote stops, pushes, the taxi times of recent push-origin takeoffs; coverage-gated (NaN, never zero, where
    the reception is not established), the coverage columns themselves left out.
  In operations these values would come from AMAN, DMAN and A-SMGCS plans, with their own errors. A gain here is
  therefore an upper bound for what perfect plans could give, and these inputs are not available as computed (tier F:
  planned in the concept, realised in the challenge data).

**Leakage.** The own flight's events are excluded as in the blocks reused: the own takeoff leaves every recorded
count, and the own callsign and the events originating within 100 m of the own stand leave the ADS-B events. After the
anchor those two ADS-B rules are not enough: the own taxi-out often still carries the inbound leg's callsign and starts
at a first sighting away from the stand (10-17 % of the exposed departures carried an own event at a + 30 min, against
the H1 audit's gate of 0.2 %). ``fut_adsb`` therefore also removes the own airframe's whole ground stay, found through
its own ADS-B takeoff, and is NaN where that takeoff is not found. Three past-window columns are left out because,
read after the own push or takeoff, they would locate it (`SLOT_COLUMNS`): the gaps since the last other takeoff and
landing on the own runway (the own takeoff fills a slot in the sequence) and the minutes since the last in-block at the
own stand (the next arrival there blocks in only after the own push).

What remains still carries traces of the own movement through the resources it shares with the other flights, and no
test can remove them, because they are in the other flights' realised movements themselves:
- the runway: the own takeoff takes one slot from the others' takeoff counts, at the own runway and at the airport
  (where one runway is in use the two are the same);
- the stand: the next arrival at the own stand waits for the own push (arrivals taxiing, their taxi-in anomaly, the
  ADS-B taxi-ins: arrival events are never own-excluded);
- the pier's pushback lanes, the de-icing pads and the runway holding queue: the own push, de-icing and hold delay or
  advance the others' (pier backlog, pushes, remote stops, holds, the others' taxi times).
In a plan these traces are the own TSAT and TTOT, not the realised ones. Rule T (findings 10.36) bounds them with a
variant without any realised departure-side column after the anchor; the shift-7 placebo (``shift_days``) bounds only
the airport's era and the flight's identity, not these traces (the own movement is never in the placebo's window).

Parameters (``features.params.<block>``): ``shift_days`` (a whole number >= 0, the placebo) for the three F blocks;
``op_h1_fut_nm`` also takes ``min_support`` (as ``op_h1_nm``), ``op_h1_fut_adsb`` every `_adsblib.DEFAULTS` key;
``op_h1_fut_plan`` takes none. With ``shift_days`` the records and NM blocks are NaN where a shifted read reaches
before the frame's records (`_horizon.window_covered`), never zero.

Auto-imported by the registry: no work at import time. The helper modules are imported as modules so that
`build._related_modules` hashes them into these blocks' cache keys (every module `traffic_at`, `nm_at` and
`surface_at` reach, as `prc.features.operational` imports them).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc import paths
from prc.features import (  # noqa: F401  (module objects: hashed into these blocks' cache keys)
    _adsblib,
    _arrlib,
    _geolib,
    _horizon,
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

SHIFTS = {"f15": 900, "f30": 1800, "f45": 2700}
"""Read times after the anchor ``a`` (15 min before the planned off-block): ``f15`` is the planned off-block, ``f30``
and ``f45`` 15 and 30 min after it. The median 2025 departure pushed 19 min after ``a`` and taxied 16 min."""
PLAN_WINDOWS = {"w15": 900, "w30": 1800, "w60": 3600}
"""The plan counts' windows ``(a, a + w]`` (``w``: a window after the anchor, not a read time)."""
SLOT_COLUMNS = ("gap_last_dep_rwy_s", "gap_last_arr_rwy_s", "stand_min_since_inblock")
"""Past-window columns left out after the anchor: read after the own push or takeoff, they would locate it."""
REC = tuple(c for c in _oplib.TRAFFIC if c not in SLOT_COLUMNS)
NM = _oplib.NM
ADSB = _adsblib.SURFACE + _adsblib.STAND
"""The ADS-B surface columns without the coverage gates (`_adsblib.COVERAGE`: gates and strata, never inputs)."""
PLAN = tuple(f"{kind}_{lvl}_{w}" for kind, lvls in (("dep_plan", ("apt", "rwy", "pier")), ("arr_sched", ("apt", "pier")))
             for lvl in lvls for w in PLAN_WINDOWS)
ADSB_EXTERNAL = (paths.ADSB_SURFACE_EVENTS_FILE, paths.ADSB_SURFACE_SUPPORT_FILE, paths.GEO_STANDS_FILE,
                 paths.GEO_STAND_MATCHES_FILE, paths.GEO_RUNWAY_ENDS_FILE)
LINK_TOL_S = 600.0
"""The own ADS-B takeoff: same airport and callsign, its roll within this of the recorded takeoff (as `prc.adsb.events`)."""
LINK_TIME_ONLY_S = 45.0
"""Else the only free ADS-B takeoff at the airport within this of the recorded takeoff, if no other departure without a
link took off within this of that takeoff (mutual uniqueness)."""
STAY_MAX_S = 8 * 3600.0
"""The own airframe's ground stay starts at its last landing at the airport, at most this before its takeoff."""
STAY_PAD_S = 300.0
"""The stay's margin after the own roll (the climb-out)."""
OWN_TOKEN = "ZZOWN"
"""The callsign shared by the stay and the query of a departure without a callsign of its own (``ZZOWN<mvt_id>``)."""
SURFACE_COLUMNS = ("apt", "is_dep", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "STAND_mvt", "RUNWAY_mvt", "CALLSIGN_flt")
"""What `_adsblib.surface_at` reads from the movement frame."""


def _names(prefix: str, cols, shifts=None) -> tuple[str, ...]:
    if shifts is None:
        return tuple(f"{prefix}{c}" for c in cols)
    return tuple(f"{prefix}{tag}_{c}" for tag in shifts for c in cols)


def _shift_days(params: dict | None, allowed: tuple[str, ...] = ()) -> int:
    prm = dict(params or {})
    unknown = set(prm) - {"shift_days", *allowed}
    if unknown:
        raise ValueError(f"unknown params {sorted(unknown)}; known: {sorted({'shift_days', *allowed})}")
    days = prm.get("shift_days", 0)
    if isinstance(days, bool) or not isinstance(days, (int, np.integer)) or int(days) < 0:
        raise ValueError(f"shift_days must be a whole number >= 0 (the placebo reads past days), got {days!r}")
    return int(days)


def _anchor(mvt: pd.DataFrame, ctx: FeatureContext) -> np.ndarray:
    return _oplib.anchor_h1(mvt.loc[ctx.dep_index])


def _forward(ref_t, ref_g, a, q_g, window: float, own_t=None) -> np.ndarray:
    """Reference events in ``(a, a + window]`` of each query's group (whole seconds), the query's own event excluded."""
    q = np.asarray(a, dtype="float64") + 1.0
    n = congestion.window_count(_oplib._series(ref_t), _oplib._series(ref_g), _oplib._series(q), _oplib._series(q_g),
                                back=0, fwd=window - 1).astype("int64")
    return n - _oplib._own_in_window(q, _oplib._query_ok(q, q_g), own_t, 0, window - 1).astype("int64")


def plan_after(mvt: pd.DataFrame, q_index: pd.Index, a: np.ndarray) -> pd.DataFrame:
    """Other departures' planned off-blocks and other arrivals' scheduled times in ``(a, a + w]`` (`PLAN`)."""
    dep, arr = _oplib.departures(mvt), _oplib.arrivals(mvt)
    q = mvt.loc[q_index]
    a = np.asarray(a, dtype="float64")
    planned = pd.Series(_oplib.planned_offblock(dep), index=dep.index)
    own_p = planned.reindex(q_index).to_numpy(dtype="float64")
    arr_s = _oplib.secs(arr["SCHED_TIME_UTC_mvt"])
    dep_groups = {
        "apt": (dep["apt"].astype("string").to_numpy(dtype=object), q["apt"].astype("string").to_numpy(dtype=object)),
        "rwy": (_oplib.runway_key(dep).to_numpy(dtype=object), _oplib.runway_key(q).to_numpy(dtype=object)),
        "pier": (_oplib.pier_key(dep).to_numpy(dtype=object), _oplib.pier_key(q).to_numpy(dtype=object)),
    }
    arr_groups = {
        "apt": (arr["apt"].astype("string").to_numpy(dtype=object), q["apt"].astype("string").to_numpy(dtype=object)),
        "pier": (_oplib.pier_key(arr).to_numpy(dtype=object), _oplib.pier_key(q).to_numpy(dtype=object)),
    }
    out = {}
    for tag, w in PLAN_WINDOWS.items():
        for lvl, (rg, qg) in dep_groups.items():
            out[f"dep_plan_{lvl}_{tag}"] = _forward(planned.to_numpy(dtype="float64"), rg, a, qg, w, own_p)
        for lvl, (rg, qg) in arr_groups.items():
            out[f"arr_sched_{lvl}_{tag}"] = _forward(arr_s, rg, a, qg, w)
    frame = pd.DataFrame(out, index=q_index)[list(PLAN)].astype("float32")
    return _oplib._mask(frame, a)


def _obj(values) -> np.ndarray:
    """An object array of strings with every missing value as None (one dtype for every merge key)."""
    s = pd.Series(values, dtype="object")
    return s.where(s.notna(), None).astype(object).to_numpy(dtype=object)


def own_airframe_stays(events: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    """The own airframe's ground stay of every departure in ``q`` whose own ADS-B takeoff is found.

    The own takeoff is the ADS-B ``TAKEOFF`` at the same airport with the flight's normalised callsign whose roll lies
    within `LINK_TOL_S` of the recorded takeoff (the nearest). Else it is the only free ADS-B takeoff at the airport
    within `LINK_TIME_ONLY_S` of it, provided no other departure without a link took off within `LINK_TIME_ONLY_S` of
    that takeoff (mutual uniqueness; `prc.adsb.events.link` keeps the first condition only). A takeoff claimed twice
    goes to the nearer departure; the other gets no stay. Events without an airport or an airframe never link.

    The stay runs from the airframe's last landing at the airport (inclusive) or just after its earlier takeoff there
    (an earlier rotation is another flight, its return landing unseen), looked up 10 min before the own roll and at
    most `STAY_MAX_S` back, to the roll plus `STAY_PAD_S`. Stays of one airframe at one airport overlap by at most
    `STAY_PAD_S` (a later stay that starts just after an earlier takeoff); there the later stay wins.

    Before the anchor the callsign and own-stand rules suffice; after it they do not, because the own taxi-out often
    still carries the inbound leg's callsign and starts at a first sighting away from the stand (findings 10.36). The
    recorded takeoff is read only to identify the airframe whose events are removed. Returns, indexed like ``q``,
    ``apt``, ``icao``, ``cs`` (the departure's normalised callsign or None), ``stay_start``, ``stay_end``.
    """
    cols = ["apt", "icao", "cs", "stay_start", "stay_end"]
    empty = pd.DataFrame({c: pd.Series(dtype="object" if c in ("apt", "icao", "cs") else "float64") for c in cols},
                         index=pd.Index([], name=q.index.name))
    kind = events["kind"].astype(str).to_numpy(dtype=object)
    tk = events.loc[kind == "TAKEOFF", ["apt", "cs", "icao", "t_start"]]
    tko = pd.DataFrame({"apt": _obj(tk["apt"].astype(object)), "cs": _obj(tk["cs"].astype(object)),
                        "icao": _obj(tk["icao"].astype(object)), "t_tko": tk["t_start"].to_numpy(dtype="float64")})
    tko = tko.loc[np.isfinite(tko["t_tko"]) & pd.notna(tko["apt"]) & pd.notna(tko["icao"])].reset_index(drop=True)
    tko["tko_id"] = np.arange(len(tko))
    dep = pd.DataFrame({"mvt_id": q.index.to_numpy(), "apt": _obj(q["apt"].astype(object)),
                        "cs": _obj(_adsblib.norm_callsign(q["CALLSIGN_flt"].to_numpy(dtype=object))),
                        "t_rec": _oplib.secs(q["MVT_TIME_UTC_mvt"])})
    dep = dep.loc[np.isfinite(dep["t_rec"]) & pd.notna(dep["apt"])].sort_values("t_rec", kind="stable")
    if not len(dep) or not len(tko):
        return empty
    by_cs = tko.loc[pd.notna(tko["cs"])].sort_values("t_tko", kind="stable")
    with_cs = dep.loc[pd.notna(dep["cs"])]
    keep = ["mvt_id", "apt", "t_rec", "tko_id", "icao", "t_tko"]
    if len(by_cs) and len(with_cs):
        link = pd.merge_asof(with_cs, by_cs, left_on="t_rec", right_on="t_tko", by=["apt", "cs"],
                             direction="nearest", tolerance=LINK_TOL_S)
        link = link.loc[link["tko_id"].notna(), keep]
    else:
        link = pd.DataFrame({c: pd.Series(dtype="object") for c in keep})
    # the time-only fallback: the only takeoff not yet claimed at the airport within LINK_TIME_ONLY_S, and no other
    # unlinked departure within LINK_TIME_ONLY_S of that takeoff
    rest = dep.loc[~dep["mvt_id"].isin(set(link["mvt_id"]))]
    free = tko.loc[~tko["tko_id"].isin(set(pd.to_numeric(link["tko_id"]).astype("int64")))].sort_values("t_tko", kind="stable")
    if len(rest) and len(free):
        n = congestion.window_count(pd.Series(free["t_tko"].to_numpy()), pd.Series(free["apt"].to_numpy(dtype=object)),
                                    pd.Series(rest["t_rec"].to_numpy()), pd.Series(rest["apt"].to_numpy(dtype=object)),
                                    back=LINK_TIME_ONLY_S, fwd=LINK_TIME_ONLY_S)
        one = rest.loc[n == 1].drop(columns="cs")
        fb = pd.merge_asof(one, free.drop(columns="cs"), left_on="t_rec", right_on="t_tko", by="apt",
                           direction="nearest", tolerance=LINK_TIME_ONLY_S)
        fb = fb.loc[fb["tko_id"].notna()]
        if len(fb):
            rivals = congestion.window_count(pd.Series(rest["t_rec"].to_numpy()), pd.Series(rest["apt"].to_numpy(dtype=object)),
                                             pd.Series(fb["t_tko"].to_numpy()), pd.Series(fb["apt"].to_numpy(dtype=object)),
                                             back=LINK_TIME_ONLY_S, fwd=LINK_TIME_ONLY_S)
            fb = fb.loc[rivals == 1, keep]
            link = pd.concat([link, fb], ignore_index=True) if len(link) else fb
    if not len(link):
        return empty
    link = link.assign(t_tko=pd.to_numeric(link["t_tko"]), t_rec=pd.to_numeric(link["t_rec"]),
                       tko_id=pd.to_numeric(link["tko_id"]).astype("int64"))
    link = link.assign(gap=(link["t_tko"] - link["t_rec"]).abs()).sort_values("gap", kind="stable")
    link = link.drop_duplicates("tko_id").drop_duplicates("mvt_id")
    link = link.assign(apt=_obj(link["apt"]), icao=_obj(link["icao"]))
    # the stay's start: the airframe's last landing or earlier takeoff at the airport, 10 min before the own roll
    is_prev = np.isin(kind, ("LANDING", "TAKEOFF"))
    pv = events.loc[is_prev, ["apt", "icao", "t_start"]]
    prev = pd.DataFrame({"apt": _obj(pv["apt"].astype(object)), "icao": _obj(pv["icao"].astype(object)),
                         "t_prev": pv["t_start"].to_numpy(dtype="float64"), "was_tko": kind[is_prev] == "TAKEOFF"})
    prev = prev.loc[np.isfinite(prev["t_prev"]) & pd.notna(prev["apt"]) & pd.notna(prev["icao"])]
    prev = prev.sort_values("t_prev", kind="stable")
    link = link.assign(t_look=link["t_tko"] - 600.0).sort_values("t_look", kind="stable")
    link = pd.merge_asof(link, prev, left_on="t_look", right_on="t_prev", by=["apt", "icao"], direction="backward")
    t_prev = link["t_prev"].to_numpy(dtype="float64") + np.where(link["was_tko"].fillna(False).to_numpy(dtype=bool), 1.0, 0.0)
    start = np.fmax(t_prev, link["t_tko"].to_numpy(dtype="float64") - STAY_MAX_S)
    own_cs = dep.set_index("mvt_id")["cs"]
    out = pd.DataFrame({"apt": link["apt"].to_numpy(dtype=object), "icao": link["icao"].to_numpy(dtype=object),
                        "cs": _obj(link["mvt_id"].map(own_cs)), "stay_start": start,
                        "stay_end": link["t_tko"].to_numpy(dtype="float64") + STAY_PAD_S},
                       index=pd.Index(link["mvt_id"].to_numpy(), name=q.index.name))
    return out[cols]


def without_own_airframes(mvt: pd.DataFrame, q_index: pd.Index, events: pd.DataFrame
                          ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
    """A slim movement frame, the event table and the departures whose own airframe was found (`own_airframe_stays`).

    In the event table every found stay carries its departure's own normalised callsign, so that `_adsblib.surface_at`'s
    callsign rule removes the whole stay besides every other event under that callsign, as before the anchor. A
    departure without a callsign gives its stay and its ``CALLSIGN_flt`` in the slim frame the token ``ZZOWN<mvt_id>``
    instead. The relabelling is done on the real day: in the shift-7 placebo the same flight a week earlier carries its
    own callsign, which is usually the query's, and is removed the same way. The stays' arrival events (the inbound's
    landing and taxi-in) carry the callsign too, but `surface_at` never own-excludes arrival kinds.
    """
    q = mvt.loc[q_index, list(SURFACE_COLUMNS)]
    stays = own_airframe_stays(events, q)
    label = pd.Series(stays["cs"].to_numpy(dtype=object), index=stays.index, dtype=object)
    no_cs = label.isna()
    tokens = pd.Series([f"{OWN_TOKEN}{int(i)}" for i in stays.index[no_cs.to_numpy()]], index=stays.index[no_cs.to_numpy()],
                       dtype=object)
    if len(tokens) and not (pd.Series(_adsblib.norm_callsign(tokens.to_numpy(dtype=object))).to_numpy() == tokens.to_numpy()).all():
        raise ValueError("an own-airframe token does not survive callsign normalisation")
    label.loc[tokens.index] = tokens
    slim = mvt[list(SURFACE_COLUMNS)]
    if len(tokens):
        cs = slim["CALLSIGN_flt"].astype(object).copy()
        cs.loc[tokens.index] = tokens
        slim = slim.assign(CALLSIGN_flt=cs)
    ev = events
    if len(stays):
        t = events["t_start"].to_numpy(dtype="float64")
        icao = _obj(events["icao"].astype(object))
        apt = _obj(events["apt"].astype(object))
        order = np.argsort(t, kind="stable")
        left = pd.DataFrame({"row": order, "apt": apt[order], "icao": icao[order], "t": t[order]})
        left = left.loc[np.isfinite(left["t"]) & pd.notna(left["apt"]) & pd.notna(left["icao"])]
        right = stays.assign(label=label).reset_index(drop=True)[["apt", "icao", "stay_start", "stay_end", "label"]]
        right = right.sort_values("stay_start", kind="stable")
        hit = pd.merge_asof(left, right, left_on="t", right_on="stay_start", by=["apt", "icao"], direction="backward")
        hit = hit.loc[hit["label"].notna() & (hit["t"] <= hit["stay_end"])]
        new_cs = np.array(events["cs"].astype(object).to_numpy(dtype=object), dtype=object)  # a writable copy
        new_cs[hit["row"].to_numpy(dtype="int64")] = hit["label"].to_numpy(dtype=object)
        ev = events.assign(cs=pd.Categorical(new_cs))
    return slim, ev, stays.index


def _at_shifts(compute, a: np.ndarray, cols, prefix: str, index: pd.Index, days: int = 0,
               mvt: pd.DataFrame | None = None) -> pd.DataFrame:
    """``compute(t)`` (a frame with ``cols``) at ``a + shift - days`` for every `SHIFTS` entry, side by side.

    With ``days`` > 0 and ``mvt``: rows whose shifted reads reach before the frame's records (the earliest read minus the
    plan backlog's three hours, `_horizon.window_covered`) are NaN, never zero."""
    a = np.asarray(a, dtype="float64")
    parts = []
    for tag, s in SHIFTS.items():
        frame = compute(a + s - 86_400.0 * days)[list(cols)]
        frame.columns = [f"{prefix}{tag}_{c}" for c in cols]
        parts.append(frame)
    out = pd.concat(parts, axis=1).reindex(index)
    if days and mvt is not None:
        start = a + min(SHIFTS.values()) - 86_400.0 * days - _oplib.BACKLOG_MAX_S
        ok = np.isfinite(a) & _horizon.window_covered(mvt, index, np.where(np.isfinite(a), start, 0.0))
        out.loc[~ok] = np.nan
    return _oplib._mask(out, a).astype("float32")


# --- the blocks ---------------------------------------------------------------------------------------------------

@feature(
    group="op_h1_fut_plan",
    outputs=_names("op_h1_fplan_", PLAN),
    doc="O*: other departures' planned off-blocks and other arrivals' scheduled times in the 15/30/60 min after H1 (airport, own runway, pier).",
)
def op_h1_fut_plan(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """Other flights' plans after the H1 anchor: departures planned off (IOBT, else LOBT, else the schedule) and arrivals
    scheduled in ``(a, a + w]``, w = 15, 30, 60 min; departures per airport, own runway and pier, arrivals per airport
    and pier. The own planned off-block (always ``a + 15 min``) is excluded. No parameters.

    Operational availability: tier O* (the filed plans and the schedule are known at H1; the other flights' runway and
    stand are their recorded ones, standing in for the planned runway and stand allocation; the last filed plans are
    assumed known 15 min ahead; flights that were cancelled are not in the data).
    """
    if ctx.params:
        raise ValueError(f"op_h1_fut_plan takes no params, got {sorted(ctx.params)}")
    a = _anchor(mvt, ctx)
    frame = plan_after(mvt, ctx.dep_index, a)
    frame.columns = [f"op_h1_fplan_{c}" for c in frame.columns]
    return frame.reindex(ctx.dep_index).astype("float32")


@feature(
    group="op_h1_fut_rec",
    outputs=_names("op_h1_frec_", REC, SHIFTS),
    doc="F: other flights' movement records read at H1 + 15/30/45 min (takeoffs, landings, arrivals taxiing, runway mix, taxi-in anomaly); planned in the concept, realised here.",
)
def op_h1_fut_rec(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """`op_h1_traffic`'s movement records of OTHER flights, read at ``a + 15``, ``a + 30`` and ``a + 45`` min
    (`SHIFTS`), without the columns that would locate the own slot (`SLOT_COLUMNS`). With ``shift_days`` d > 0: the
    same read d days earlier (the placebo), NaN where it reaches before the frame's records.

    Operational availability: tier F. In operations: the AMAN landing sequence, the DMAN takeoff sequence and runway
    configuration plan, the surface manager's taxi-in plan. Here: the realised movements of the other flights, an
    upper bound for perfect plans; not available as computed. Own-runway columns are also O* (planned runway).
    """
    a = _anchor(mvt, ctx)
    days = _shift_days(ctx.params)
    return _at_shifts(lambda t: _oplib.traffic_at(mvt, ctx.dep_index, t), a, REC, "op_h1_frec_", ctx.dep_index, days, mvt)


@feature(
    group="op_h1_fut_nm",
    outputs=_names("op_h1_fnm_", NM, SHIFTS),
    uses_target=True,
    doc="F: NM readings of the other flights airborne by H1 + 15/30/45 min (proxy taxi anomaly, push delay) and the plan backlog then; fold-safe baseline.",
)
def op_h1_fut_nm(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """`op_h1_nm` read at ``a + 15``, ``a + 30`` and ``a + 45`` min: per airport and own runway, over the 30/60 min
    before each read time, the NM proxy anomaly (mean, share above +5 min) and the push delay of the OTHER flights that
    took off; the plan backlog at each read time. The proxy baseline comes from the fold's fit rows only, month-out
    (`_oplib.NmBaseline`; ``ctx.fit_mask``, no target read). With ``shift_days`` d > 0: d days earlier (placebo), NaN
    where it reaches before the frame's records.

    Operational availability: tier F. In operations: the surface manager's and DMAN's view of the departures ahead.
    Here: the realised NM readings of the other flights, an upper bound; not available as computed.
    """
    a = _anchor(mvt, ctx)
    days = _shift_days(ctx.params, ("min_support",))
    min_support = dict(ctx.params or {}).get("min_support", _proxylib.MIN_SUPPORT)
    if isinstance(min_support, bool) or not isinstance(min_support, (int, np.integer)) or min_support < 1:
        raise ValueError(f"min_support must be a whole number >= 1, got {min_support!r}")
    dep = _oplib.departures(mvt)
    fit = (np.zeros(len(dep), dtype=bool) if ctx.fit_mask is None
           else ctx.fit_mask.reindex(dep.index).fillna(False).to_numpy(dtype=bool))
    base = _oplib.nm_baseline(dep, fit, int(min_support))
    return _at_shifts(lambda t: _oplib.nm_at(mvt, ctx.dep_index, t, base), a, NM, "op_h1_fnm_", ctx.dep_index, days, mvt)


@feature(
    group="op_h1_fut_adsb",
    outputs=_names("op_h1_fadsb_", ADSB, SHIFTS),
    external=ADSB_EXTERNAL,
    doc="F: public ADS-B surface picture of other aircraft read at H1 + 15/30/45 min (taxiing out and in, runway queue, pushes, recent taxi times), coverage-gated.",
)
def op_h1_fut_adsb(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """`op_h1_adsb`'s surface picture of OTHER aircraft read at ``a + 15``, ``a + 30`` and ``a + 45`` min (each minus
    the feed lag): taxiing out and in, taxi-outs begun, runway holds (all, own runway), remote stops, elapsed taxi and
    stationary shares, pushes and the taxi times of recent push-origin takeoffs. The own airframe is removed at event
    level: its whole ground stay, found through its own ADS-B takeoff (`own_airframe_stays`), besides the callsign and
    own-stand rules (`without_own_airframes`). A departure whose own takeoff is not found is NaN (its own taxi could not
    be told apart). Coverage gates as `_adsblib.surface_at` (NaN where reception is not established or outside the
    airport's 2025 regime); the gate columns are not outputs. ``shift_days`` (in the `_adsblib.DEFAULTS` params): the
    events of that many days earlier (placebo); the coverage gates make a read before the snapshot's days NaN.

    Operational availability: tier F. In operations: the A-SMGCS surface plan (routes, taxi times, the runway queue the
    DMAN builds). Here: the realised ADS-B tracks of the other aircraft, an upper bound; not available as computed.
    ``rwy_own_hold`` columns are also O* (planned runway).
    """
    a = _anchor(mvt, ctx)
    slim, ev, linked = without_own_airframes(mvt, ctx.dep_index, _adsblib.load_events())
    frame = _at_shifts(lambda t: _adsblib.surface_at(slim, ctx.dep_index, t, ctx.params, events=ev), a, ADSB,
                       "op_h1_fadsb_", ctx.dep_index)
    frame.loc[~ctx.dep_index.isin(linked)] = np.nan  # the own airframe not identified: no picture after the anchor
    return frame
