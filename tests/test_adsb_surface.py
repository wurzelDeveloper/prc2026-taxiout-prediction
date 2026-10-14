"""The ADS-B surface builder (`prc.adsb.surface`, `prc adsb-surface`): events against hand-written expectations and
brute force, and the properties the feature blocks rely on.

* scenario: a parked, pushed, held departure and its inbound give exactly the expected events and times (its stand an
  OSM way, so its id is negative); an arrival reported airborne through its rollout is a landing and a taxi-in, not a
  taxi-out;
* stops against a brute-force scan (with and without ground speeds); pushes against the lab detector
  (lab/ideas/pushback.py) on the same track;
* thinning invariance: the same trajectories sampled at 1 s and 2.5 s give the same events (times within 5 s);
* causality: a build censored at ``a`` (points at or before ``a``, cutoff ``a``) reproduces every event known by ``a``,
  ends known by ``a`` included, later ends as NaN -- on random cutoffs, and on every event boundary and timeout of a
  world with reception gaps, a stop followed by silence, a touch-and-go, sparse airborne points and lift-offs confirmed
  after the timeout of the last ground point;
* timeouts: a reception gap out on the taxiways keeps one taxi-out, the same gap after a point at the stand ends it;
* midnight: two days built separately and together give identical rows;
* the vehicle filter, the clip and deduplication of day files (critique F7), the satellite-airfield mask (F3), a missing
  day (open ends only), the manifest (resumable, rebuilds on changes, the support table, frozen snapshots) and the
  ``--check`` report.

Synthetic only (tests/adsb_synth.py): no competition data, no network.
"""

from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import pandas as pd
import pytest

import adsb_synth as S
from prc import data, paths
from prc.adsb import surface
from prc.features import _adsblib
from test_operational import synthetic_world

KEYS = ["apt", "kind", "icao", "t_start"]
END_FIELDS = ["t_end", "t_known_end", "dur", "stop_s", "path_m"]


@pytest.fixture(scope="module")
def geo():
    return S.geometry()


@pytest.fixture(scope="module")
def small_world():
    mvt, ob = synthetic_world(days=4, per_day=36)
    mvt = S.with_callsigns(mvt)
    pts, info = S.world(mvt, ob)
    return mvt, ob, pts, info


@pytest.fixture(scope="module")
def full(small_world, geo):
    _, _, pts, _ = small_world
    return surface.episodes(pts, geo, float(pts["t"].max()))


def view(ev: pd.DataFrame, a: float) -> pd.DataFrame:
    """The events known by ``a``, every end-dependent field masked where the end is not known by ``a``."""
    ev = ev.loc[ev["t_known_start"] <= a].copy()
    late = ~(ev["t_known_end"] <= a)
    ev.loc[late, ["t_end", "t_known_end"]] = np.nan
    ev.loc[late & ev["kind"].ne("TAKEOFF"), ["dur", "stop_s", "path_m"]] = np.nan
    return ev.sort_values(KEYS, kind="stable").reset_index(drop=True)


def same_rows(a: pd.DataFrame, b: pd.DataFrame) -> None:
    a = a.sort_values(KEYS, kind="stable").reset_index(drop=True)
    b = b.sort_values(KEYS, kind="stable").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_exact=True, check_dtype=False)


# --- scenario: one departure and its inbound, by hand -------------------------------------------------------

def test_scenario_departure_and_inbound(geo):
    tr = S.Tracks()
    t_land, b, t_lift = 1_742_500_000.0, 1_742_506_000.0, 1_742_507_200.0
    S.arrival(tr, "EDDF", "abc123", 3, "07C", t_land, t_land + 420.0, lambda t: "INB1", park_on=150.0)
    b_adsb = S.departure(tr, "EDDF", "abc123", 3, "25C", b, t_lift, S.switch("INB1", "OUT1", b + 300.0), dwell=400.0)
    assert b_adsb == b  # the taxi fits: the push starts at b
    ev = surface.episodes(tr.frame(), geo, cutoff=t_lift + 3600)
    kinds = ev["kind"].value_counts().to_dict()
    assert kinds == {"LANDING": 1, "TAXI_IN": 1, "PARK": 1, "PUSH": 1, "TAXI_OUT": 1, "RWY_HOLD": 1, "TAKEOFF": 1}, kinds
    one = {k: r for k, r in ev.set_index("kind").iterrows()}
    own_stand = float(S.stand_id("EDDF", 3))
    assert own_stand < 0  # an OSM way: its id is negated, and must never read as "no stand"
    # landing and taxi-in: touchdown known at once; the taxi-in ends at the stand stop, known 60 s into it
    assert one["LANDING"]["t_known_start"] == t_land and one["LANDING"]["rwy_end"] == "07C"
    assert one["TAXI_IN"]["t_start"] == t_land and one["TAXI_IN"]["t_known_end"] - one["TAXI_IN"]["t_end"] == pytest.approx(60, abs=20)
    # parked at the stand from b - 400 (the transponder came back on), parking known 300 s in
    assert one["PARK"]["t_start"] == b - 400 and one["PARK"]["t_known_start"] == pytest.approx(b - 100, abs=20)
    assert one["PARK"]["origin_stand"] == own_stand
    # the push starts at b, is recognised within half a minute, from the own stand
    assert abs(one["PUSH"]["t_start"] - b) <= 5 and 0 < one["PUSH"]["t_known_start"] - b <= 30
    assert one["PUSH"]["origin_stand"] == own_stand
    # the taxi-out: from the push, confirmed at 12 kt after the engine start, ended by the lift-off (known 20 s later)
    to = one["TAXI_OUT"]
    assert to["origin_kind"] == "push" and to["origin_stand"] == own_stand and abs(to["t_start"] - b) <= 5
    assert b + 57 + 60 <= to["t_known_start"] <= b + 57 + 60 + 10
    assert to["t_end"] == t_lift and t_lift + 20 <= to["t_known_end"] <= t_lift + 25
    assert to["cs"] == "OUT1" or to["cs"] == "INB1"
    # the runway hold beside 25C, known 30 s in, inside the taxi-out
    h = one["RWY_HOLD"]
    assert h["rwy_end"] == "25C" and h["t_known_start"] - h["t_start"] == pytest.approx(30, abs=20)
    assert h["origin_kind"] == "push" and h["origin_stand"] == own_stand and h["dur"] > 60
    # the takeoff: confirmed 20 s after the lift-off, on 25C, with its taxi-out's attributes
    k = one["TAKEOFF"]
    assert k["rwy_end"] == "25C" and t_lift + 20 <= k["t_known_start"] <= t_lift + 25 and k["cs"] == "OUT1"
    assert k["origin_kind"] == "push" and k["taxi_seen"] == 1 and k["dur"] == pytest.approx(t_lift - 13 - b, abs=6)
    assert k["parent"] == to["eid"] and h["parent"] == to["eid"]


def test_remote_stop_inside_a_taxi_out_and_a_first_seen_origin(geo):
    """A de-icing stop of 6 min off the stands and runways is a REMOTE_STOP; a transponder switched on mid-taxi gives
    a first_seen taxi-out (no push, no stand stop)."""
    tr = S.Tracks()
    b, t_lift = 1_742_506_000.0, 1_742_508_400.0
    s = S.stand_xy(5)
    cs = lambda t: "OUT2"  # noqa: E731
    pad = np.array([0.0, 100.0])
    t = tr.move("EDDF", "abc124", b, [s, np.array([s[0], S.APRON_Y]), S.HUB], 8.0, 2.5, cs)  # first seen moving
    t = tr.move("EDDF", "abc124", t, [S.HUB, pad], 8.0, 2.5, cs)
    tr.stationary("EDDF", "abc124", t + 20, t + 360, pad, 180.0, cs, 20.0)
    thr, u, h, _ref, hold, _exit = S.ends("EDDF")["25C"]
    t = tr.move("EDDF", "abc124", t + 380, [pad, hold, thr], 8.0, 2.5, cs)
    for k in np.arange(0.0, 40.0, 2.0):
        p = thr + 0.5 * 1.925 * k * k * u
        tr.add("EDDF", "abc124", t + k, p[0], p[1], 1.925 * k / S.KT, h, True, -1.0, "OUT2")
    lift = t + 40
    p0 = thr + 0.5 * 1.925 * 1600 * u
    for k in np.arange(0.0, 60.0, 2.0):
        p = p0 + 80 * k * u
        tr.add("EDDF", "abc124", lift + k, p[0], p[1], 160.0, h, False, 100 + 20 * k, "OUT2")
    pts = tr.frame()

    af = pd.Series({"abc124": b - 86_400.0})  # seen airborne before (the vehicle filter's history)
    ev = surface.episodes(pts, geo, cutoff=lift + 3600, airborne_first=af)
    r = ev.loc[ev["kind"].eq("REMOTE_STOP")]
    assert len(r) == 1 and r["dur"].iloc[0] == pytest.approx(360, abs=25)
    to = ev.loc[ev["kind"].eq("TAXI_OUT")].iloc[0]
    assert to["origin_kind"] == "first_seen" and r["parent"].iloc[0] == to["eid"]
    assert to["stop_s"] >= 340 and ev.loc[ev["kind"].eq("TAKEOFF"), "stop_s"].iloc[0] == to["stop_s"]


def landing_reported_airborne(tr: S.Tracks, icao: str, t_land: float, stand_i: int = 6, cs: str = "ARRX",
                              slowest_kt: float = 45.0) -> float:
    """An arrival at EDDF 07C whose transponder reports airborne through the rollout down to ``slowest_kt`` (measured:
    40-58 kt at EHAM, about 1 s before the first ground point at ~42 kt), then a taxi-in and a stand; returns the time
    of the first ground point."""
    thr, u, h, _ref, _hold, _exit = S.ends("EDDF")["07C"]
    td = thr + 300.0 * u
    for k in np.arange(-80.0, 0.0, 2.0):  # the final, 140 kt
        p = td + 72.0 * k * u
        tr.add("EDDF", icao, t_land + k, p[0], p[1], 140.0, h, False, max(-8.0 * k, 50.0), cs)
    v0, v1, dur = 72.0, slowest_kt * S.KT, 40.0  # the rollout, still airborne, 140 kt -> slowest_kt
    acc = (v0 - v1) / dur
    for k in np.arange(0.0, dur + 0.1, 2.0):
        p = td + (v0 * k - 0.5 * acc * k * k) * u
        tr.add("EDDF", icao, t_land + k, p[0], p[1], (v0 - acc * k) / S.KT, h, False, -1.0, cs)
    pr = td + (v0 * dur - 0.5 * acc * dur * dur) * u
    t_ground = t_land + dur + 1.0
    tr.add("EDDF", icao, t_ground, *(pr + 1.0 * v1 * u), 42.0, h, True, -1.0, cs)  # first ground point, 42 kt
    s = S.stand_xy(stand_i)
    t = tr.move("EDDF", icao, t_ground + 3.0, [pr + 3.0 * v1 * u, S.HUB, np.array([s[0], S.APRON_Y]), s], 8.0, 2.5, lambda _t: cs)
    tr.stationary("EDDF", icao, t + 20.0, t + 400.0, s, 0.0, lambda _t: cs, 20.0)
    return t_ground


def test_a_landing_reported_airborne_through_the_rollout_is_a_landing_and_a_taxi_in(geo, monkeypatch):
    """The landing is decided by the approach's speed in its last ``LDG_LOOK_S``, not by the last airborne point (45 kt
    here): no taxi-out out of an arrival's taxi-in (EHAM: 113 of 819 taxi-outs on 2025-07-15 with the old rule)."""
    tr = S.Tracks()
    t_ground = landing_reported_airborne(tr, "abe001", 1_742_500_000.0)
    pts = tr.frame()
    af = pd.Series({"abe001": 1_742_400_000.0})
    ev = surface.episodes(pts, geo, t_ground + 3600, af)
    kinds = ev["kind"].value_counts().to_dict()
    assert kinds.get("LANDING") == 1 and kinds.get("TAXI_IN") == 1 and "TAXI_OUT" not in kinds, kinds
    ldg = ev.loc[ev["kind"].eq("LANDING")].iloc[0]
    assert 0 <= ldg["t_known_start"] - t_ground < 2 * surface.THIN_S and ldg["rwy_end"] == "07C"  # (thinned to 5-s bins)
    # guard: the last airborne point alone (the old rule) makes the taxi-in a first_seen taxi-out
    monkeypatch.setattr(surface, "LDG_LOOK_S", 0)
    old = surface.episodes(pts, geo, t_ground + 3600, af)
    assert not old["kind"].eq("LANDING").any() and old["kind"].eq("TAXI_OUT").any()
    assert old.loc[old["kind"].eq("TAXI_OUT"), "origin_kind"].eq("first_seen").all()


def test_nearest_stand_returns_nan_beyond_the_radius_and_keeps_negative_ids(geo):
    g = geo["EDDF"]
    s3, s4 = S.stand_xy(3), S.stand_xy(4)
    d, ids = g.nearest_stand(np.array([s3[0], s4[0], 0.0]), np.array([s3[1], s4[1], -900.0]), 60.0)
    assert ids.dtype == np.float64 and ids[0] == S.stand_id("EDDF", 3) < 0 and ids[1] == S.stand_id("EDDF", 4) > 0
    assert np.isnan(ids[2]) and d[2] > 60.0


# --- brute force ----------------------------------------------------------------------------------------

def _thinned(pts: pd.DataFrame) -> pd.DataFrame:
    p = surface.clean_points(pts)
    return p.loc[~p.assign(_b=p["t"] // surface.THIN_S).duplicated(["icao", "_b"])]


@pytest.mark.parametrize("no_gs", [0.0, 0.08])
def test_stops_match_a_brute_force_scan(small_world, geo, no_gs):
    """Maximal runs of stationary ground points (gs < 1 kt, or without a ground speed: under 0.5 m/s and 15 m since the
    previous ground point of the visit; gaps <= 600 s; jumps <= 15 m) against the vectorised runs."""
    _, _, pts, info = small_world
    icaos = list(info["icao"].iloc[:25])
    sub = pts.loc[pts["icao"].isin(icaos)]
    if no_gs:
        sub = S.degrade(sub, np.random.default_rng(11), no_track=0.0, no_gs=no_gs)
        assert sub["gs_kt"].isna().sum() > 100
    p = _thinned(sub).loc[lambda d: d["apt"].eq("EDDF") | d["apt"].eq("LSZH")]
    want = set()
    for (apt, ic), g in p.groupby(["apt", "icao"], sort=True):
        x, y = geo[apt].xy(g["lat"].to_numpy(), g["lon"].to_numpy())
        t, gs = g["t"].to_numpy(), g["gs_kt"].to_numpy()
        ground = g["on_ground"].to_numpy() | (gs < surface.GROUND_KT)
        # the previous ground point of the same visit (every airborne run of this world is a real one)
        prev = np.full(len(t), -1)
        for i in range(1, len(t)):
            if ground[i] and ground[i - 1] and t[i] - t[i - 1] <= surface.SEG_GAP_S:
                prev[i] = i - 1
        stationary = np.zeros(len(t), dtype=bool)
        for i in range(len(t)):
            if not ground[i]:
                continue
            if np.isfinite(gs[i]):
                stationary[i] = gs[i] < surface.STATIONARY_KT
            elif prev[i] >= 0:
                step = np.hypot(x[i] - x[prev[i]], y[i] - y[prev[i]])
                stationary[i] = step / max(t[i] - t[prev[i]], 0.5) < surface.STATIONARY_MPS and step < surface.STOP_JUMP_M
        i, n = 0, len(t)
        while i < n:
            if not stationary[i]:
                i += 1
                continue
            j = i
            while (j + 1 < n and stationary[j + 1] and prev[j + 1] == j and t[j + 1] - t[j] <= surface.STOP_GAP_S
                   and np.hypot(x[j + 1] - x[j], y[j + 1] - y[j]) <= surface.STOP_JUMP_M):
                j += 1
            want.add((ic, float(t[i]), float(t[j])))
            i = j + 1
    q = p.copy()
    xs, ys = np.zeros(len(q)), np.zeros(len(q))
    for apt in ("EDDF", "LSZH"):
        m = q["apt"].eq(apt).to_numpy()
        xs[m], ys[m] = geo[apt].xy(q.loc[m, "lat"].to_numpy(), q.loc[m, "lon"].to_numpy())
    q = q.assign(x=xs, y=ys).sort_values(["apt", "icao", "t"], kind="stable").reset_index(drop=True)
    tr = surface._Track(q, geo, float(q["t"].max()))
    got = {(tr.icao[tr.gi[k0]], float(tr.gt[k0]), float(tr.gt[k1])) for k0, k1 in zip(tr.s_k0, tr.s_k1)}
    assert got == want
    assert len(want) > 3 * len(icaos)


def test_pushes_agree_with_the_lab_detector(small_world, full):
    """The reverse-motion rule of lab/ideas/pushback.py run on the same thinned ground track finds the same push."""
    spec = importlib.util.spec_from_file_location("lab_pushback", paths.ROOT / "lab" / "ideas" / "pushback.py")
    pb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pb)
    _, _, pts, info = small_world
    p = _thinned(pts)
    push = full.loc[full["kind"].eq("PUSH")].set_index("icao")
    checked = 0
    for mid, r in info.iloc[:40].iterrows():
        g = p.loc[p["icao"].eq(r["icao"]) & (p["t"] >= r["b"] - r["dwell"] - 1) & (p["t"] < r["t_lift"])]
        g = g.loc[g["on_ground"] | (g["gs_kt"] < 40)]
        res = pb.detect(g["t"].to_numpy(), g["lat"].to_numpy(), g["lon"].to_numpy(), g["gs_kt"].to_numpy(),
                        g["track"].to_numpy(), r["t_lift"])
        assert res["pb_found"] == 1, mid
        mine = push.loc[[r["icao"]]]
        mine = mine.loc[(mine["t_start"] >= r["b"] - 60) & (mine["t_start"] <= r["b"] + 60)]
        assert len(mine) == 1, mid
        assert np.floor(r["t_lift"] - res["pb_start"]) == mine["t_start"].iloc[0], mid
        checked += 1
    assert checked == 40


# --- invariances ------------------------------------------------------------------------------------------

def test_thinning_invariance(geo):
    """The same trajectories sampled at 1 s and at 2.5 s: the same events, every time within 5 s."""
    mvt, ob = synthetic_world(days=2, per_day=20)
    mvt = S.with_callsigns(mvt)
    a, _ = S.world(mvt, ob, seed=4, dt_move=1.0)
    b, _ = S.world(mvt, ob, seed=4, dt_move=2.5)
    cut = float(max(a["t"].max(), b["t"].max()))
    ea = surface.episodes(a, geo, cut).sort_values(KEYS).reset_index(drop=True)
    eb = surface.episodes(b, geo, cut).sort_values(KEYS).reset_index(drop=True)
    assert len(ea) > 200
    ca = ea.groupby(["icao", "kind"]).size()
    cb = eb.groupby(["icao", "kind"]).size()
    pd.testing.assert_series_equal(ca, cb)
    ea = ea.sort_values(["icao", "kind", "t_start"]).reset_index(drop=True)
    eb = eb.sort_values(["icao", "kind", "t_start"]).reset_index(drop=True)
    for c in ("t_start", "t_known_start", "t_end", "t_known_end"):
        d = (ea[c] - eb[c]).abs()
        assert (d.dropna() <= 5).all() and (ea[c].isna() == eb[c].isna()).all(), c
    assert (ea["origin_kind"].fillna("") == eb["origin_kind"].fillna("")).all()
    assert (ea["rwy_end"].fillna("") == eb["rwy_end"].fillna("")).all()


@pytest.mark.parametrize("hour", [8.4, 11.75, 15.1, 19.6])
def test_causality_a_censored_build_reproduces_every_event_known_by_the_cutoff(small_world, geo, full, hour):
    _, _, pts, _ = small_world
    af = surface.airborne_first_times(pts)
    d1 = np.floor(pts["t"].min() / 86_400) * 86_400 + 86_400
    # the cutoff: in the middle of the in-progress event nearest the nominal hour (so the censoring has teeth)
    spans = full.loc[full["t_known_end"] - full["t_known_start"] >= 120]
    mid = np.floor((spans["t_known_start"] + spans["t_known_end"]) / 2).to_numpy()
    a = float(mid[np.argmin(np.abs(mid - (d1 + hour * 3600)))])
    cut = surface.episodes(pts.loc[np.floor(pts["t"]) <= a], geo, a, af[af <= a])
    want = view(full, a)
    got = view(cut, a)
    assert len(want) > 100
    same_rows(want, got)
    # a later end appears as NaN, not as a guess
    assert (cut["t_known_end"].dropna() <= a).all() and (cut["t_known_start"] <= a).all()
    assert ((full["t_known_start"] <= a) & (full["t_known_end"] > a)).any(), "the cutoff ends nothing in progress"


def _lift_after_silence(tr: S.Tracks, icao: str, b: float, stand_i: int, silence: float, stationary_end: bool,
                        air_dt: float = 15.0, cs: str = "OUT9") -> float:
    """A departure from ``stand_i`` to the 25C hold whose ground reception stops at the hold (moving, or after a stop
    there); its first airborne point comes ``silence`` s after the last ground point, the airborne points ``air_dt``
    apart (sparse: the lift-off is confirmed one or two points later). Returns the last ground point's time."""
    thr, u, h, _ref, hold, _exit = S.ends("EDDF")["25C"]
    s = S.stand_xy(stand_i)
    p1 = s + np.array([0.0, -S.PUSH_M])
    c = lambda _t: cs  # noqa: E731
    tr.stationary("EDDF", icao, b - 60.0, b - 1.0, s, 0.0, c, 20.0)
    t = tr.move("EDDF", icao, b, [s, p1], 0.7, 2.5, c, reverse=True, gs_kt=1.4)
    tr.stationary("EDDF", icao, t + 20.0, t + 59.0, p1, 180.0, c, 20.0)
    t = tr.move("EDDF", icao, t + 60.0, [p1, np.array([s[0], S.APRON_Y]), S.HUB, hold], 8.0, 2.5, c)
    if stationary_end:
        tr.stationary("EDDF", icao, t + 20.0, t + 120.0, hold, h, c, 20.0)
    t_last = _last_ground(tr, icao)
    lift = t_last + silence
    p0 = thr + 1500.0 * u
    for k in np.arange(0.0, 90.0, air_dt):
        p = p0 + 80.0 * k * u
        tr.add("EDDF", icao, lift + k, p[0], p[1], 160.0, h, False, 100.0 + 25.0 * k, cs)
    return t_last


def _touch_and_go(tr: S.Tracks, icao: str, t_land: float, cs: str = "TNG1") -> float:
    """A landing on 07C, a rollout down to 49 kt and a new takeoff roll on the same runway (a taxi-in ended by a
    lift-off)."""
    thr, u, h, *_ = S.ends("EDDF")["07C"]
    td = thr + 300.0 * u
    for k in np.arange(-60.0, 0.0, 2.0):
        p = td + 72.0 * k * u
        tr.add("EDDF", icao, t_land + k, p[0], p[1], 140.0, h, False, max(-8.0 * k, 50.0), cs)
    dec, acc = (72.0 - 25.0) / 20.0, (77.0 - 25.0) / 20.0
    for k in np.arange(0.0, 20.1, 2.0):  # rollout 72 -> 25 m/s
        p = td + (72.0 * k - 0.5 * dec * k * k) * u
        tr.add("EDDF", icao, t_land + k, p[0], p[1], (72.0 - dec * k) / S.KT, h, True, -1.0, cs)
    p0 = td + (72.0 * 20 - 0.5 * dec * 400) * u
    for k in np.arange(2.0, 20.1, 2.0):  # power up again, 25 -> 77 m/s
        p = p0 + (25.0 * k + 0.5 * acc * k * k) * u
        tr.add("EDDF", icao, t_land + 20.0 + k, p[0], p[1], (25.0 + acc * k) / S.KT, h, True, -1.0, cs)
    pl = p0 + (25.0 * 20 + 0.5 * acc * 400) * u
    for k in np.arange(0.0, 60.0, 2.0):
        p = pl + 80.0 * (k + 1.0) * u
        tr.add("EDDF", icao, t_land + 41.0 + k, p[0], p[1], 160.0, h, False, 50.0 + 20.0 * k, cs)
    return t_land


def _stop_then_silence(tr: S.Tracks, icao: str, b: float, stand_i: int, cs: str = "OUT8") -> float:
    """A departure that pushes, taxis to the de-icing pad, stops there 6 min and is never heard of again; returns its
    last point's time."""
    s = S.stand_xy(stand_i)
    p1 = s + np.array([0.0, -S.PUSH_M])
    c = lambda _t: cs  # noqa: E731
    t = tr.move("EDDF", icao, b, [s, p1], 0.7, 2.5, c, reverse=True, gs_kt=1.4)
    t = tr.move("EDDF", icao, t + 60.0, [p1, np.array([s[0], S.APRON_Y]), S.HUB, S.PAD], 8.0, 2.5, c)
    tr.stationary("EDDF", icao, t + 20.0, t + 400.0, S.PAD, 180.0, c, 20.0)
    return _last_ground(tr, icao)


def _last_ground(tr: S.Tracks, icao: str) -> float:
    """The (whole-second) time of the airframe's last ground point so far."""
    return float(np.floor(max(r[2] for r in tr.rows if r[1] == icao and r[7])))


@pytest.fixture(scope="module")
def edge_world(small_world):
    """A day of the small world (EDDF) plus hand-made edge cases: lift-offs confirmed 10 s within and 20 s after the
    last ground point's timeout (moving at the hold; after a stop there), a ground-reception gap before the lift-off,
    a stop followed by silence, a touch-and-go, sparse airborne points, an arrival reported airborne through its
    rollout, ground points without a track or a ground speed."""
    _, _, pts, info = small_world
    t0 = float(np.floor(pts["t"].min() / 86_400) * 86_400) + 86_400 + 9 * 3600
    keep = pts["apt"].eq("EDDF") & pts["t"].between(t0 - 2 * 3600, t0 + 2.5 * 3600)
    tr = S.Tracks()
    lasts = []
    # lift-offs 30 s (dense airborne points: confirmed 20 s later, within the timeout) and 10 s (sparse: confirmed
    # 30 s later, after it) before the timeout of the last ground point, moving at the hold or after a stop there
    for k, (silence, stat_end) in enumerate(((surface.TIMEOUT_TAXIWAY_S - 30, False), (surface.TIMEOUT_TAXIWAY_S - 10, False),
                                             (surface.STOP_GAP_S - 30, True), (surface.STOP_GAP_S - 10, True),
                                             (40.0, False))):
        lasts.append(_lift_after_silence(tr, f"abf00{k}", t0 + 600.0 * k, 2 + 2 * k, silence, stat_end,
                                         air_dt=15.0 if k % 2 else 4.0))
    _touch_and_go(tr, "abf010", t0 + 900.0)
    lasts.append(_stop_then_silence(tr, "abf011", t0 + 1500.0, 15))
    landing_reported_airborne(tr, "abf012", t0 + 2100.0, stand_i=17)
    hand = tr.frame()
    gap = hand["icao"].eq("abf004") & hand["on_ground"] & (hand["t"] > lasts[4] - 400) & (hand["t"] < lasts[4] - 100)
    hand = hand.loc[~gap]  # a reception gap of 300 s before the lift-off
    world = pd.concat([pts.loc[keep], hand], ignore_index=True)
    world = S.degrade(world, np.random.default_rng(5), no_track=0.15, no_gs=0.03)
    af = surface.airborne_first_times(pts).copy()
    for ic in hand["icao"].unique():
        af[ic] = t0 - 86_400.0
    return world, af


def test_causality_on_every_event_boundary_and_timeout(edge_world, geo):
    """The censored build agrees with the full one at every event's knowable start and end (one second before, at, and
    one after), and around every timeout of the hand-made last ground points: no look-ahead of even one second."""
    world, af = edge_world
    end = float(world["t"].max())
    full = surface.episodes(world, geo, end, af)
    kinds = set(full["kind"])
    assert {"TAKEOFF", "LANDING", "TAXI_OUT", "TAXI_IN", "PUSH", "RWY_HOLD", "REMOTE_STOP"} <= kinds, kinds
    th = _thinned(world)
    th = th.loc[th["on_ground"].to_numpy() | (th["gs_kt"].to_numpy() < surface.GROUND_KT)]
    last_ground = th.groupby("icao")["t"].max()
    # the hand-made lift-offs after a silence: confirmed within the timeout, the takeoff ends the taxi-out; confirmed
    # after it, the taxi-out vanished at its last point, known a timeout later -- and the takeoff still carries it as
    # its parent (the lift-off itself came within the timeout)
    mine = full.loc[full["icao"].astype(str).str.startswith("abf00")]
    tko = mine.loc[mine["kind"].eq("TAKEOFF")].set_index("icao")
    out = mine.loc[mine["kind"].eq("TAXI_OUT")].set_index("icao")
    assert {"abf000", "abf001", "abf002", "abf003"} <= set(tko.index)
    vanished = 0
    for ic, timeout in (("abf000", surface.TIMEOUT_TAXIWAY_S), ("abf001", surface.TIMEOUT_TAXIWAY_S),
                        ("abf002", surface.STOP_GAP_S), ("abf003", surface.STOP_GAP_S)):
        last, conf = last_ground[ic], tko.loc[ic, "t_known_start"]
        o = out.loc[[ic]].sort_values("t_start").iloc[-1]
        if conf - last <= timeout:
            assert o["t_known_end"] == conf, ic
        else:
            assert o["t_end"] == last and o["t_known_end"] == last + timeout, ic
            vanished += 1
        assert tko.loc[ic, "parent"] == o["eid"] and np.isfinite(tko.loc[ic, "dur"]), ic
    assert vanished == 2
    hold = mine.loc[mine["kind"].eq("RWY_HOLD")].set_index("icao")
    assert hold.loc["abf003", "t_known_end"] == last_ground["abf003"] + surface.STOP_GAP_S  # vanished before the lift-off
    assert hold.loc["abf002", "t_known_end"] == tko.loc["abf002", "t_known_start"]
    tng = full.loc[full["icao"].astype(str).eq("abf010")].set_index("kind")
    assert tng.loc["TAXI_IN", "t_known_end"] == tng.loc["TAKEOFF", "t_known_start"]  # a taxi-in ended by a lift-off
    # the cutoffs
    cuts = set()
    for c in ("t_known_start", "t_known_end"):
        for t in full[c].dropna().to_numpy():
            cuts.update((t - 1, t, t + 1))
    for ic in [i for i in last_ground.index if str(i).startswith("abf")]:
        for timeout in (surface.TIMEOUT_TAXI_IN_S, surface.TIMEOUT_MOVING_S, surface.TIMEOUT_TAXIWAY_S, surface.STOP_GAP_S):
            cuts.update(last_ground[ic] + timeout + d for d in (-1, 0, 1))
    t_lo = float(world["t"].min()) + 3600
    cuts = sorted(c for c in cuts if t_lo <= c < end)
    assert len(cuts) > 300
    t_pts = np.floor(world["t"].to_numpy())
    for a in cuts:
        cut = surface.episodes(world.loc[t_pts <= a], geo, a, af[af <= a])
        same_rows(view(full, a), view(cut, a))


def _taxi_with_gap(geo, where: str) -> pd.DataFrame:
    """A departure from stand 4 with 400 s without points: during its push (slow, at the stand), as it starts to taxi
    (15 kt, 40 m from the stand: taxiing along the apron) or half way to the runway."""
    tr = S.Tracks()
    b, t_lift = 1_742_506_000.0, 1_742_508_400.0
    S.departure(tr, "EDDF", "abg001", 4, "25C", b, t_lift, lambda _t: "GAP1", dwell=40.0)
    pts = tr.frame()
    s = S.stand_xy(4)
    ex, ny = geo["EDDF"].xy(pts["lat"].to_numpy(), pts["lon"].to_numpy())
    d_stand = np.hypot(ex - s[0], ny - s[1])
    gs = pts["gs_kt"].to_numpy()
    if where == "push":  # the first push point after the stand dwell, 1.4 kt at the stand
        t_gap = float(pts.loc[(gs > 0) & (gs < surface.SWITCH_OFF_KT) & (d_stand <= surface.STAND_M), "t"].min())
    elif where == "apron":
        t_gap = float(pts.loc[(gs >= surface.CONFIRM_KT) & (d_stand <= surface.STAND_M), "t"].min())
    else:
        t_gap = float(pts.loc[(gs >= surface.CONFIRM_KT) & (d_stand > 400.0) & pts["on_ground"].to_numpy(), "t"].min())
    return pts.loc[~((pts["t"] > t_gap) & (pts["t"] <= t_gap + 400.0))]


@pytest.mark.parametrize("where", ["taxiway", "apron", "push"])
def test_a_reception_gap_keeps_the_taxi_out_unless_it_stopped_at_the_stand(geo, where):
    """The moving timeout depends on how the airframe was last seen: taxiing (on the taxiways, or past stands on the
    apron) a 400-s gap keeps the taxi-out (``TIMEOUT_TAXIWAY_S``); moving slowly at the stand (pushing, rolling in) it
    has switched off after ``TIMEOUT_MOVING_S`` -- the push ended unconfirmed and the taxi is a new episode."""
    pts = _taxi_with_gap(geo, where)
    ev = surface.episodes(pts, geo, float(pts["t"].max()) + 3600, pd.Series({"abg001": 0.0}))
    out = ev.loc[ev["kind"].eq("TAXI_OUT")]
    tko = ev.loc[ev["kind"].eq("TAKEOFF")].iloc[0]
    assert len(out) == 1, out[["origin_kind", "t_start", "t_end"]]
    if where == "push":
        assert out["origin_kind"].iloc[0] != "push" and tko["origin_kind"] != "push"
    else:
        assert out["origin_kind"].iloc[0] == "push" and tko["origin_kind"] == "push"


# --- day files, midnight, missing days ---------------------------------------------------------------------

@pytest.fixture
def day_tree(tmp_path, monkeypatch, small_world, geo):
    """The small world written as UTC day files (points/), the geometry and the snapshot root patched to tmp."""
    _, _, pts, _ = small_world
    pts_dir = tmp_path / "points"
    pts_dir.mkdir()
    monkeypatch.setattr(paths, "ADSB_POINTS", pts_dir)
    monkeypatch.setattr(paths, "ADSB_POINTS_PROD", tmp_path / "points_prod")
    monkeypatch.setattr(paths, "ADSB_SURFACE", tmp_path / "adsb_surface")
    monkeypatch.setattr(surface, "load_geometry", lambda: geo)
    monkeypatch.setattr(surface, "geometry_digest", lambda: "synthetic")
    monkeypatch.setattr(_adsblib, "load_stand_tables", S.stand_tables)  # the support table runs the block's gates
    monkeypatch.setattr(_adsblib, "load_runway_keys", S.runway_keys)
    surface._WORKER.clear()
    days = sorted(pd.to_datetime(pts["t"], unit="s", utc=True).dt.strftime("%Y-%m-%d").unique())
    for d in days:
        d0 = surface.day_start(d)
        pts.loc[(pts["t"] >= d0) & (pts["t"] < d0 + 86_400)].to_parquet(pts_dir / f"{d}.parquet", index=False)
    yield days, pts_dir
    surface._WORKER.clear()


def test_midnight_days_built_separately_and_together_agree(day_tree, geo, small_world):
    days, _ = day_tree
    _, _, pts, _ = small_world
    af = surface.airborne_first_times(pts)
    d1, d2 = days[1], days[2]
    sep = pd.concat([surface.build_events_for_day(d, geo, af) for d in (d1, d2)], ignore_index=True)
    # together: one window over both days, events known in [d1, d3)
    t1 = surface.day_start(d1)
    w = pts.loc[(pts["t"] >= t1 - surface.LOOKBACK_S) & (pts["t"] < t1 + 2 * 86_400 + surface.LOOKAHEAD_S)]
    tog = surface.episodes(w, geo, t1 + 2 * 86_400 + surface.LOOKAHEAD_S - 1, af)
    tog = tog.loc[(tog["t_known_start"] >= t1) & (tog["t_known_start"] < t1 + 2 * 86_400)]
    assert len(sep) > 300
    same_rows(sep, tog)


def test_day_files_are_clipped_to_their_day_and_deduplicated(day_tree, geo, small_world):
    """Stale points of an earlier day inside a later file, and duplicate (icao, t) rows, change nothing (F7)."""
    days, pts_dir = day_tree
    _, _, pts, _ = small_world
    af = surface.airborne_first_times(pts)
    d1, d2 = days[1], days[2]
    before = surface.build_events_for_day(d1, geo, af)
    t1 = surface.day_start(d1)
    stale = pts.loc[(pts["t"] >= t1 + 6 * 3600) & (pts["t"] < t1 + 8 * 3600)]
    f2 = pts_dir / f"{d2}.parquet"
    later = pd.read_parquet(f2)
    late_only = stale.iloc[::7].assign(t=lambda d: d["t"] + 0.5)  # points that only the later file has, dated in d1
    pd.concat([later, stale, late_only], ignore_index=True).to_parquet(f2, index=False)
    f1 = pts_dir / f"{d1}.parquet"
    own = pd.read_parquet(f1)
    pd.concat([own, own.iloc[::5]], ignore_index=True).to_parquet(f1, index=False)  # duplicate (icao, t) rows
    assert len(surface.read_day(d2)) == len(later.drop_duplicates(["icao", "t"]))
    assert len(surface.read_day(d1)) == len(own.drop_duplicates(["icao", "t"]))
    only_t = surface.read_day(d1, columns=["t"])  # deduplicated on (icao, t) whatever the caller reads
    assert list(only_t.columns) == ["t"] and len(only_t) == len(own.drop_duplicates(["icao", "t"]))
    same_rows(before, surface.build_events_for_day(d1, geo, af))


def test_a_missing_day_writes_nothing_and_leaves_its_neighbours_open(day_tree, geo, small_world):
    days, pts_dir = day_tree
    _, _, pts, _ = small_world
    af = surface.airborne_first_times(pts)
    d1, d2 = days[1], days[2]
    before = surface.build_events_for_day(d1, geo, af)
    (pts_dir / f"{d2}.parquet").unlink()
    assert surface.build_events_for_day(d2, geo, af).empty
    after = surface.build_events_for_day(d1, geo, af)
    t2 = surface.day_start(d2)
    # the same events; ends that needed the next day's points are open now, and nothing else moved
    same_rows(view(before, t2 - 1), after)
    assert not (after["t_known_end"] >= t2).any()


def test_event_ids_are_unique_and_parents_exist(full):
    """One row per (airframe, kind, start): the content id is unique, and every parent is a TAXI_OUT known by then."""
    assert full["eid"].is_unique and not full.duplicated(KEYS).any()
    child = full.loc[full["parent"] >= 0]
    taxi = full.loc[full["kind"].eq("TAXI_OUT")].set_index("eid")
    assert len(child) > 100 and child["parent"].isin(taxi.index).all()
    assert (taxi.loc[child["parent"], "t_known_start"].to_numpy() <= child["t_known_start"].to_numpy()).all()


def test_vehicle_filter_and_satellite_mask(full, small_world, geo):
    _, _, pts, info = small_world
    icaos = set(full["icao"])
    assert not {i for i in icaos if i.startswith("f")}, "a ground vehicle (never airborne) produced an event"
    assert not {i for i in icaos if i.startswith("e")}, "the satellite airfield 10 km out produced an event"
    # the satellite's ground points are off the airport's network; its own taxiways are cut from the network
    sat = pts.loc[pts["icao"].str.startswith("e") & pts["on_ground"]]
    x, y = geo["EDDF"].xy(sat["lat"].to_numpy(), sat["lon"].to_numpy())
    assert (geo["EDDF"].d_net(x, y) > surface.NET_MAX_M).all()
    far = surface.make_geometry("EDDF", *S.APTS["EDDF"], pd.DataFrame(
        [{"ident": "07C", "lat": S.to_latlon("EDDF", -1400, -1700)[0], "lon": S.to_latlon("EDDF", -1400, -1700)[1],
          "heading_deg": 70.0, "length_m": 3000.0, "pair_ref": ""}]),
        ways=[np.column_stack(S.to_latlon("EDDF", np.array([-8000.0, -8000.0]), np.array([6000.0, 5000.0])))])
    assert len(far.net_xy) == 0
    # an airframe seen airborne only after an event's knowable time does not count before it
    first = info.iloc[0]
    late_af = surface.airborne_first_times(pts).copy()
    late_af[first["icao"]] = first["t_lift"] + 1  # pretend its first airborne point is after its own lift-off
    ev = surface.episodes(pts.loc[pts["icao"].eq(first["icao"])], geo, float(pts["t"].max()), late_af)
    assert not len(ev.loc[ev["t_known_start"] < first["t_lift"] + 1])
    assert len(ev.loc[ev["kind"].eq("TAKEOFF")]) == 1  # known after it: kept


def test_takeoffs_need_a_runway_of_the_airport(geo):
    """A takeoff roll 8 km away (inside the box) and one off every centreline are no TAKEOFF (critique F3)."""
    for offset in (np.array([0.0, -8000.0]), np.array([0.0, 600.0])):
        tr = S.Tracks()
        thr, u, h, *_ = S.ends("EDDF")["25C"]
        start = thr + offset
        tr.stationary("EDDF", "abd001", 1_742_600_000.0, 1_742_600_100.0, start, h, lambda t: "X1", 20.0)
        for k in np.arange(0.0, 40.0, 2.0):
            p = start + 0.5 * 1.925 * k * k * u
            tr.add("EDDF", "abd001", 1_742_600_120.0 + k, p[0], p[1], 1.925 * k / S.KT, h, True, -1.0, "X1")
        for k in np.arange(0.0, 60.0, 2.0):
            p = start + 1540 * u + 80 * k * u
            tr.add("EDDF", "abd001", 1_742_600_160.0 + k, p[0], p[1], 160.0, h, False, 100.0 + 20 * k, "X1")
        ev = surface.episodes(tr.frame(), geo, 1_742_610_000.0, pd.Series({"abd001": 0.0}))
        assert not ev["kind"].eq("TAKEOFF").any(), offset


# --- the snapshot: manifest, rebuilds, frozen, check -----------------------------------------------------------

def test_snapshot_build_is_resumable_and_rebuilds_on_changes(day_tree, monkeypatch, small_world):
    days, pts_dir = day_tree
    out = surface.build(days=days, snapshot="t1", workers=1, log=lambda *_: None)
    man = surface.read_manifest("t1")
    assert out.exists() and set(man["days"]) == set(days)
    warm = [d for d in days if man["days"][d]["key"]["warmup"]]
    assert warm == days[: surface.WARMUP_DAYS]  # the vehicle filter's warm-up days write no events
    assert pd.read_parquet(paths.adsb_surface_dir("t1") / "days" / f"{days[0]}.parquet").empty
    cons = pd.read_parquet(out)
    assert set(cons["kind"].astype(str)) <= set(surface.BLOCK_KINDS) and "PARK" not in set(cons["kind"].astype(str))
    stamp = {d: (paths.adsb_surface_dir("t1") / "days" / f"{d}.parquet").stat().st_mtime_ns for d in days}
    built = []
    surface.build(snapshot="t1", log=built.append)
    assert any("0 days to build" in m for m in built)
    assert stamp == {d: (paths.adsb_surface_dir("t1") / "days" / f"{d}.parquet").stat().st_mtime_ns for d in days}
    # a neighbour's input changes: the days that read it are rebuilt, the others are not
    f = pts_dir / f"{days[-1]}.parquet"
    pd.read_parquet(f).iloc[:-50].to_parquet(f, index=False)
    built = []
    surface.build(snapshot="t1", log=built.append)
    assert any(f"{len([days[-2], days[-1]])} days to build" in m for m in built), built
    # a geometry change rebuilds everything
    monkeypatch.setattr(surface, "geometry_digest", lambda: "other")
    built = []
    surface.build(snapshot="t1", log=built.append)
    assert any(f"{len(days)} days to build" in m for m in built), built
    # the freeze needs the support table of the consolidated table as it is
    with pytest.raises(RuntimeError, match="support"):
        surface.build(snapshot="t1", log=lambda *_: None, freeze=True)
    mvt = small_world[0].assign(split=pd.Categorical(["train"] * len(small_world[0])))  # every departure a training row
    monkeypatch.setattr(data, "load_movements", lambda columns=None: mvt[list(columns)] if columns else mvt)
    surface.build_support("t1", log=lambda *_: None)
    sup = pd.read_parquet(paths.adsb_surface_dir("t1") / paths.ADSB_SURFACE_SUPPORT_NAME)
    assert {"taxi", "hold", "stand", "own_hold"} <= set(sup["family"]) and set(sup["apt"]) == {"EDDF", "LSZH"}
    assert surface.read_manifest("t1")["support"]["events_sha1"] == surface.read_manifest("t1")["consolidated"]["sha1"]
    surface.freeze_snapshot("t1", log=lambda *_: None)
    with pytest.raises(RuntimeError, match="frozen"):
        surface.build(snapshot="t1", log=lambda *_: None)
    with pytest.raises(RuntimeError, match="frozen"):
        surface.build_support("t1", log=lambda *_: None)
    report = surface.check("t1")
    assert "EDDF" in report and "LSZH" in report and "warm-up days" in report and "frozen" in report
    assert "push_with_stand" in report and "taxi_seen_share" in report and "split_share" in report


def test_a_rebuild_marks_the_support_table_stale(day_tree, monkeypatch, small_world):
    days, pts_dir = day_tree
    surface.build(days=days, snapshot="t3", log=lambda *_: None)
    mvt = small_world[0].assign(split=pd.Categorical(["train"] * len(small_world[0])))
    monkeypatch.setattr(data, "load_movements", lambda columns=None: mvt[list(columns)] if columns else mvt)
    surface.build_support("t3", log=lambda *_: None)
    f = pts_dir / f"{days[-2]}.parquet"
    full_day = pd.read_parquet(f)
    full_day.iloc[: len(full_day) // 2].to_parquet(f, index=False)  # half of a built day lost: its events change
    said = []
    surface.build(snapshot="t3", log=said.append)
    assert any("--support" in m for m in said), said
    with pytest.raises(RuntimeError, match="support"):
        surface.freeze_snapshot("t3", log=lambda *_: None)


def test_check_lists_missing_days(day_tree):
    days, pts_dir = day_tree
    (pts_dir / f"{days[1]}.parquet").unlink()
    surface.build(days=days, snapshot="t2", log=lambda *_: None)
    report = surface.check("t2")
    assert "missing days: " + days[1] in report
    man = json.loads((paths.adsb_surface_dir("t2") / "manifest.json").read_text(encoding="utf-8"))
    assert days[1] in man["missing"] and days[1] not in man["days"]


def test_parse_days_ranges():
    assert surface.parse_days(["2025-01-30..2025-02-02", "2025-07-01"]) == [
        "2025-01-30", "2025-01-31", "2025-02-01", "2025-02-02", "2025-07-01"]
    assert surface.parse_days(None) is None


# --- the real data (opt-in) ------------------------------------------------------------------------------------

REAL = os.environ.get("OPLAD_REAL") == "1"


@pytest.mark.skipif(not (REAL and paths.GEO_STANDS_FILE.exists()), reason="set OPLAD_REAL=1 (real ADS-B points of one day)")
def test_real_day_censored_builds_reproduce_every_event_known_by_the_cutoff():
    """The builder's causality on real points (``OPLAD_SURFACE_DAY``, default 2025-10-15; ``OPLAD_SURFACE_APT``,
    default LSZH): censored builds at cutoffs on event boundaries (a knowable start or end, one second before it) and
    at random seconds reproduce every event known by the cutoff. The day's window (6 h of the day before) is read as
    the builder reads it; the vehicle filter is the window's own."""
    day = os.environ.get("OPLAD_SURFACE_DAY", "2025-10-15")
    apt = os.environ.get("OPLAD_SURFACE_APT", "LSZH")
    pts, cutoff = surface.window(day)
    pts = pts.loc[pts["apt"].astype(str).eq(apt)]
    assert len(pts) > 20_000, f"no points for {apt} on {day}"
    geo = surface.load_geometry()
    af = surface.airborne_first_times(pts)
    full = surface.episodes(pts, geo, cutoff, af)
    d0 = surface.day_start(day)
    inside = full.loc[full["t_known_start"].between(d0 + 6 * 3600, d0 + 20 * 3600)]
    assert len(inside) > 300
    rng = np.random.default_rng(int(d0) % 1000)
    bounds = np.concatenate([inside["t_known_start"].to_numpy(), inside["t_known_end"].dropna().to_numpy()])
    cuts = [float(c) - d for c in rng.choice(bounds, 30, replace=False) for d in (0, 1)]
    cuts += list(d0 + rng.integers(7 * 3600, 19 * 3600, 6).astype(float))
    t_pts = np.floor(pd.to_numeric(pts["t"]).to_numpy(dtype="float64"))
    for a in cuts:
        cut = surface.episodes(pts.loc[t_pts <= a], geo, a, af[af <= a])
        same_rows(view(full, a), view(cut, a))

