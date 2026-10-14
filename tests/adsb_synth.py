"""Synthetic ADS-B worlds for tests/test_adsb_surface.py and tests/test_operational_adsb.py (no data, no network).

Two airports named like `test_operational.synthetic_world`'s (EDDF, LSZH) with the same runway idents, a row of 21
stands (A1..A12, B1..B8, H1) 80 m apart, an apron taxiway, holding points beside every runway end, a densified aeroway
network, and trajectories in local metres converted to WGS84:

* a departure: transponder on at the stand ``dwell`` s before the push, a 40-m pushback (reverse to the reported
  heading), an engine-start pause, a taxi to the runway's holding point, a hold, line-up, a 40-s roll and a climb; the
  callsign switches from the inbound one to the outbound one at a chosen time (measured: a median +295 s after the
  off-block); its airframe first lands as an inbound under another callsign, taxis in and parks (transponder off);
* an arrival of the movement records: approach, touchdown on its runway, rollout, taxi-in to its stand, parked;
* ground vehicles on the apron (never airborne) and a light aircraft at a satellite airfield 10 km out (off-network).

Odd stands carry negative ids (OSM ways, as most real stands). Opt-in realism (`world`): departures first seen moving
250 m from their stand (no stand stop, no push), ground points without track or ground speed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc.adsb import surface
from prc.features._geolib import EARTH_R_M

APTS = {"EDDF": (50.03, 8.55), "LSZH": (47.46, 8.55)}
RUNWAY_PAIRS = {
    "EDDF": [("07C", 70.0, "25C", 250.0, 0.0, -1200.0), ("18", 180.0, "36", 0.0, -2200.0, 0.0)],
    "LSZH": [("10", 100.0, "28", 280.0, 0.0, -1200.0), ("14", 140.0, "32", 320.0, 2200.0, 600.0),
             ("16", 160.0, "34", 340.0, -2200.0, 600.0)],
}
LENGTH = 3000.0
STANDS = [f"A{i}" for i in range(1, 13)] + [f"B{i}" for i in range(1, 9)] + ["H1"]
STAND_Y, APRON_Y, PUSH_M = 600.0, 300.0, 40.0
HUB = np.array([0.0, APRON_Y])
PAD = np.array([0.0, 100.0])
"""A de-icing pad between the apron and the runways: off the stands, off the runway areas."""
SATELLITE = np.array([-8000.0, 6000.0])
KT = 0.514444


def unit(h: float) -> np.ndarray:
    return np.array([np.sin(np.radians(h)), np.cos(np.radians(h))])


def to_latlon(apt: str, x, y):
    lat0, lon0 = APTS[apt]
    lat = lat0 + np.degrees(np.asarray(y, dtype="float64") / EARTH_R_M)
    lon = lon0 + np.degrees(np.asarray(x, dtype="float64") / (EARTH_R_M * np.cos(np.radians(lat0))))
    return lat, lon


def stand_xy(i: int) -> np.ndarray:
    return np.array([-800.0 + 80.0 * i, STAND_Y])


def stand_id(apt: str, i: int) -> int:
    """OSM stand ids: odd stands are mapped as ways, whose ids the geometry tables store NEGATED (as most real stands
    at EDDF, LEBL, LEMD and LSZH), so no negative number may mean "no stand" anywhere."""
    return (-1 if i % 2 else 1) * ((1000 if apt == "EDDF" else 2000) + i)


def ends(apt: str) -> dict:
    """ident -> (threshold xy, unit heading, heading, pair_ref, holding point xy, exit xy)."""
    out = {}
    for ia, ha, ib, hb, cx, cy in RUNWAY_PAIRS[apt]:
        c = np.array([cx, cy])
        ref = f"{ia}/{ib}"
        for ident, h in ((ia, ha), (ib, hb)):
            u = unit(h)
            thr = c - LENGTH / 2 * u
            n = np.array([u[1], -u[0]])
            if n @ (HUB - thr) < 0:
                n = -n
            out[ident] = (thr, u, h, ref, thr + 100.0 * u + 150.0 * n, thr + 1950.0 * u)
    return out


def geometry() -> dict:
    """The builder's geometry (`surface.make_geometry`) of both airports."""
    out = {}
    for apt, (lat0, lon0) in APTS.items():
        e = ends(apt)
        rows = []
        for ident, (thr, _u, h, ref, _hold, _exit) in e.items():
            la, lo = to_latlon(apt, thr[0], thr[1])
            rows.append({"ident": ident, "lat": float(la), "lon": float(lo), "heading_deg": h, "length_m": LENGTH, "pair_ref": ref})
        st = stand_table(apt)
        holds = np.array([to_latlon(apt, *v[4]) for v in e.values()])
        ways = []
        for i in range(len(STANDS)):
            s = stand_xy(i)
            ways.append([s, np.array([s[0], APRON_Y])])
        ways.append([np.array([-900.0, APRON_Y]), np.array([900.0, APRON_Y])])
        ways.append([HUB, PAD])
        for thr, _u, _h, _ref, hold, exit_ in e.values():
            ways += [[HUB, hold], [hold, thr], [exit_, HUB], [PAD, hold]]
        for ia, _ha, ib, _hb, _cx, _cy in RUNWAY_PAIRS[apt]:
            ways.append([e[ia][0], e[ib][0]])
        ways_ll = [np.column_stack(to_latlon(apt, np.array(w)[:, 0], np.array(w)[:, 1])) for w in ways]
        out[apt] = surface.make_geometry(apt, lat0, lon0, pd.DataFrame(rows), st, holds, ways_ll)
    return out


def stand_table(apt: str) -> pd.DataFrame:
    xy = np.array([stand_xy(i) for i in range(len(STANDS))])
    la, lo = to_latlon(apt, xy[:, 0], xy[:, 1])
    return pd.DataFrame({"apt": apt, "stand_id": [stand_id(apt, i) for i in range(len(STANDS))], "ref": STANDS,
                         "lat": la, "lon": lo, "east_m": xy[:, 0], "north_m": xy[:, 1]})


def stand_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(stands, stand_matches) as the geometry tables hold them."""
    stands = pd.concat([stand_table(a) for a in APTS], ignore_index=True)
    matches = pd.DataFrame({"apt": stands["apt"], "stand_mvt": stands["ref"], "stand_id": stands["stand_id"],
                            "match_source": "exact"})
    return stands, matches


def runway_keys() -> set[str]:
    """``apt|ident`` of every runway end (`_adsblib.load_runway_keys` on the synthetic geometry)."""
    return {f"{apt}|{ident}" for apt in APTS for ident in ends(apt)}


# --- trajectories ---------------------------------------------------------------------------------------

class Tracks:
    """Collects points (local metres) and turns them into the points frame of `surface.POINT_COLUMNS`."""

    def __init__(self):
        self.rows: list[tuple] = []

    def add(self, apt, icao, t, x, y, gs_kt, track, on_ground, alt, cs):
        self.rows.append((apt, icao, float(t), float(x), float(y), float(gs_kt), float(track), bool(on_ground), float(alt), cs))

    def stationary(self, apt, icao, t0, t1, p, track, cs, dt):
        t = t0
        while t <= t1 + 1e-9:
            self.add(apt, icao, t, p[0], p[1], 0.0, track, True, -1.0, cs(t))
            t += dt

    def move(self, apt, icao, t0, pts, speed, dt, cs, reverse=False, gs_kt=None):
        """Along a polyline at ``speed`` m/s, one point per ``dt`` from ``t0``; returns the arrival time."""
        pts = np.array([np.asarray(p, dtype="float64") for p in pts])
        keep = np.concatenate([[True], np.hypot(*np.diff(pts, axis=0).T) > 0])
        pts = pts[keep]
        cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
        dur = cum[-1] / speed if speed > 0 else 0.0
        t = t0
        while t <= t0 + dur + 1e-9:
            d = min((t - t0) * speed, cum[-1])
            j = int(np.clip(np.searchsorted(cum, d, side="right") - 1, 0, len(cum) - 2))
            a, b = pts[j], pts[j + 1]
            f = (d - cum[j]) / (cum[j + 1] - cum[j])
            p = a + f * (b - a)
            brg = float(np.degrees(np.arctan2(*(b - a))) % 360.0)
            trk = (brg + 180.0) % 360.0 if reverse else brg
            self.add(apt, icao, t, p[0], p[1], gs_kt if gs_kt is not None else speed / KT, trk, True, -1.0, cs(t))
            t += dt
        return t0 + dur

    def frame(self) -> pd.DataFrame:
        cols = ["apt", "icao", "t", "x", "y", "gs_kt", "track", "on_ground", "alt_ft", "callsign"]
        f = pd.DataFrame(self.rows, columns=cols)
        lat, lon = np.zeros(len(f)), np.zeros(len(f))
        for apt in APTS:
            m = f["apt"].eq(apt).to_numpy()
            lat[m], lon[m] = to_latlon(apt, f.loc[m, "x"], f.loc[m, "y"])
        f["lat"], f["lon"] = lat, lon
        f["t"] = np.floor(f["t"])
        return f[list(surface.POINT_COLUMNS)].sort_values(["icao", "t"], kind="stable").reset_index(drop=True)


def departure(tr: Tracks, apt, icao, stand_i, rwy, b, t_lift, cs, dwell=40.0, dt_move=2.5, dt_stat=20.0, deice_s=0.0):
    """A departure whose push starts at ``b`` (or earlier when the taxi does not fit) and lifts off at ``t_lift``;
    with ``deice_s`` it stops that long at the de-icing pad on the way."""
    e = ends(apt)[rwy]
    thr, u, h, _ref, hold, _exit = e
    s = stand_xy(stand_i)
    p1 = s + np.array([0.0, -PUSH_M])
    taxi_path = [p1, np.array([s[0], APRON_Y]), HUB, hold] if not deice_s else [p1, np.array([s[0], APRON_Y]), HUB, PAD]
    d_taxi = sum(np.hypot(*(b_ - a_)) for a_, b_ in zip(taxi_path[:-1], taxi_path[1:]))
    d_pad = float(np.hypot(*(hold - PAD))) if deice_s else 0.0
    push_s, start_s, line_s, roll_s = PUSH_M / 0.7, 60.0, np.hypot(*(thr - hold)) / 5.0, 40.0
    fixed = push_s + start_s + line_s + roll_s + (deice_s + d_pad / 8.0 if deice_s else 0.0)
    v = 8.0
    if t_lift - b < fixed + d_taxi / v + 10.0:
        v = min(15.0, max(d_taxi / max(t_lift - b - fixed - 10.0, 1.0), 8.0))
    hold_s = max(t_lift - b - fixed - d_taxi / v, 10.0)
    b = t_lift - (fixed + d_taxi / v + hold_s)  # == the given b unless the taxi does not fit
    tr.stationary(apt, icao, b - dwell, b - 1.0, s, 0.0, cs, dt_stat)
    t = tr.move(apt, icao, b, [s, p1], 0.7, dt_move, cs, reverse=True, gs_kt=1.4)
    tr.stationary(apt, icao, t + min(dt_stat, start_s / 2), t + start_s - 1.0, p1, 180.0, cs, dt_stat)
    t = tr.move(apt, icao, t + start_s, taxi_path, v, dt_move, cs)
    if deice_s:
        tr.stationary(apt, icao, t + min(dt_stat, deice_s / 2), t + deice_s - 1.0, PAD, 180.0, cs, dt_stat)
        t = tr.move(apt, icao, t + deice_s, [PAD, hold], 8.0, dt_move, cs)
    tr.stationary(apt, icao, t + min(dt_stat, hold_s / 2), t + hold_s - 1.0, hold, h, cs, dt_stat)
    t = tr.move(apt, icao, t + hold_s, [hold, thr], 5.0, dt_move, cs)
    # roll: accelerate along the runway to ~150 kt, lift-off at t_lift
    acc = 77.0 / roll_s
    k = 0.0
    while k < roll_s:
        p = thr + 0.5 * acc * k * k * u
        tr.add(apt, icao, t_lift - roll_s + k, p[0], p[1], acc * k / KT, h, True, -1.0, cs(t_lift - roll_s + k))
        k += min(dt_move, 2.0)
    p_lift = thr + 0.5 * acc * roll_s ** 2 * u
    for k in np.arange(0.0, 60.0, 2.0):
        p = p_lift + 80.0 * k * u
        tr.add(apt, icao, t_lift + k, p[0], p[1], 160.0, h, False, 100.0 + 25.0 * k, cs(t_lift + k))
    return b


def arrival(tr: Tracks, apt, icao, stand_i, rwy, t_land, t_inblock, cs, park_on=120.0, dt_move=2.5, dt_stat=20.0):
    """Approach, touchdown at ``t_land``, rollout, taxi-in to the stand (in-block at ``t_inblock`` when it fits)."""
    thr, u, h, _ref, _hold, exit_ = ends(apt)[rwy]
    td = thr + 300.0 * u
    for k in np.arange(-80.0, 0.0, 2.0):
        p = td + 72.0 * k * u
        tr.add(apt, icao, t_land + k, p[0], p[1], 140.0, h, False, max(-8.0 * k, 50.0), cs(t_land + k))
    # rollout: 72 -> 10 m/s over 40 s
    k = 0.0
    while k <= 40.0:
        d = 72.0 * k - 0.5 * (62.0 / 40.0) * k * k
        p = td + d * u
        tr.add(apt, icao, t_land + k, p[0], p[1], (72.0 - 62.0 / 40.0 * k) / KT, h, True, -1.0, cs(t_land + k))
        k += 2.0
    s = stand_xy(stand_i)
    path = [td + (72.0 * 40 - 0.5 * 62.0 * 40) * u, HUB, np.array([s[0], APRON_Y]), s]
    d = sum(np.hypot(*(b_ - a_)) for a_, b_ in zip(path[:-1], path[1:]))
    avail = t_inblock - (t_land + 42.0)
    v = min(15.0, max(d / max(avail, 1.0), 6.0))
    t = tr.move(apt, icao, t_land + 42.0, path, v, dt_move, cs)
    tr.stationary(apt, icao, t + dt_stat, t + park_on, s, 0.0, cs, dt_stat)
    return t


def vehicle(tr: Tracks, apt, icao, t0, hours=2.0, dt=5.0):
    t, k = t0, 0
    while t < t0 + hours * 3600:
        a, b_ = (np.array([-900.0, APRON_Y + 40]), np.array([900.0, APRON_Y + 40]))
        if k % 2:
            a, b_ = b_, a
        t = tr.move(apt, icao, t, [a, b_], 6.0, dt, lambda _t: "VEH1") + 30.0
        k += 1


def satellite(tr: Tracks, apt, icao, t0, cs="GA1"):
    """A light aircraft at a satellite airfield inside the box: stand, taxi, a takeoff roll and a climb, 10 km out."""
    p = SATELLITE
    tr.stationary(apt, icao, t0, t0 + 120, p, 270.0, lambda _t: cs, 20.0)
    t = tr.move(apt, icao, t0 + 120, [p, p + np.array([0.0, -500.0])], 6.0, 2.5, lambda _t: cs)
    u = unit(270.0)
    q = p + np.array([0.0, -500.0])
    for k in np.arange(0.0, 26.0, 2.0):
        r = q + 0.5 * 1.8 * k * k * u
        tr.add(apt, icao, t + k, r[0], r[1], 1.8 * k / KT, 270.0, True, -1.0, cs)
    r0 = q + 0.5 * 1.8 * 26 * 26 * u
    for k in np.arange(0.0, 60.0, 2.0):
        r = r0 + 50.0 * k * u
        tr.add(apt, icao, t + 26 + k, r[0], r[1], 100.0, 270.0, False, 50.0 + 20.0 * k, cs)


def switch(cs_in, cs_out, t_switch):
    return lambda t: cs_out if t >= t_switch else cs_in


def first_seen_cut(tr: Tracks, n0: int, stand_i: int, min_m: float = 250.0) -> float:
    """Delete the points a departure appended from row ``n0`` on until it first is ``min_m`` from its stand on the
    ground (a transponder switched on mid-taxi: no stand stop, no push); returns the first remaining time."""
    s = stand_xy(stand_i)
    mine = tr.rows[n0:]
    first = next((r[2] for r in mine if r[7] and np.hypot(r[3] - s[0], r[4] - s[1]) >= min_m), np.inf)
    tr.rows[n0:] = [r for r in mine if r[2] >= first]
    return first


def degrade(points: pd.DataFrame, rng, no_track: float, no_gs: float) -> pd.DataFrame:
    """Ground points without a reported track (``no_track`` of them) or ground speed (``no_gs``), as in the real feed
    (track missing on ~17 % of ground points, ground speed on ~1-2 %)."""
    out = points.copy()
    ground = out["on_ground"].to_numpy(dtype=bool)
    out.loc[ground & (rng.random(len(out)) < no_track), "track"] = np.nan
    out.loc[ground & (rng.random(len(out)) < no_gs), "gs_kt"] = np.nan
    return out


def world(mvt: pd.DataFrame, offblocks: pd.Series, seed: int = 0, dt_move: float = 2.5, dt_stat: float = 20.0,
          first_seen: float = 0.0, no_track: float = 0.0, no_gs: float = 0.0, switch_before: pd.Index | None = None,
          first_seen_rows: pd.Index | None = None):
    """Points for every movement of ``mvt`` (departures with their inbound legs, arrivals), vehicles and a satellite
    aircraft per airport-day; and per departure: icao, callsigns, the ADS-B push and lift-off times, stand, runway.

    Realism (critique F1/F4, review): ``first_seen`` of the departures (and every one in ``first_seen_rows``) are first
    seen moving 250 m from their stand (no stand stop, no push; ``info.first_seen``); ``no_track`` / ``no_gs`` of the
    ground points lack a track / a ground speed. ``switch_before``: departures whose outbound callsign is set before
    their first point."""
    rng = np.random.default_rng(seed)
    tr = Tracks()
    info = {}
    dep = mvt.loc[mvt["is_dep"].to_numpy(dtype=bool)]
    t_mvt = (dep["MVT_TIME_UTC_mvt"] - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()
    for k, (mid, r) in enumerate(dep.iterrows()):
        apt = str(r["apt"])
        rwys = [x for x in ends(apt)]
        rwy = str(r["RUNWAY_mvt"]) if pd.notna(r["RUNWAY_mvt"]) and str(r["RUNWAY_mvt"]) in rwys else rwys[0]
        si = STANDS.index(str(r["STAND_mvt"])) if pd.notna(r["STAND_mvt"]) else int(rng.integers(0, len(STANDS)))
        t_lift = float(np.floor(t_mvt[k]))
        b = offblocks.get(mid, np.nan)
        if not np.isfinite(b) or not (0 <= t_lift - b <= 3600):
            b = t_lift - float(rng.integers(500, 1300))  # ranking rows, stale and wrong-date records: a real taxi
        icao = f"a{k:05x}"
        cs_out = str(r["CALLSIGN_flt"]) if "CALLSIGN_flt" in r and pd.notna(r["CALLSIGN_flt"]) else f"UNM{k}"
        cs_in = f"INB{k}"
        t_switch = b + float(rng.integers(-60, 600))
        turnaround = float(rng.integers(45, 120)) * 60.0
        t_land = b - turnaround
        dwell = float(rng.choice([15.0, 40.0, 400.0]))
        deice = 360.0 if rng.random() < 0.25 else 0.0
        fs = bool(rng.random() < first_seen) or (first_seen_rows is not None and mid in first_seen_rows)
        if switch_before is not None and mid in switch_before:
            t_switch = b - dwell - 1200.0  # set before the first point of the departure (after the inbound parked)
        cs = switch(cs_in, cs_out, t_switch)
        arrival(tr, apt, icao, si, rwys[0], t_land, t_land + 420.0, lambda t, c=cs_in: c,
                park_on=float(rng.integers(30, 300)), dt_move=dt_move, dt_stat=dt_stat)
        n0 = len(tr.rows)
        b_adsb = departure(tr, apt, icao, si, rwy, b, t_lift, cs, dwell=dwell, dt_move=dt_move, dt_stat=dt_stat, deice_s=deice)
        t_first = first_seen_cut(tr, n0, si) if fs else b_adsb - dwell
        info[mid] = {"apt": apt, "icao": icao, "cs_in": cs_in, "cs_out": cs_out, "t_switch": t_switch, "b": b_adsb, "t_lift": t_lift,
                     "stand_i": si, "rwy": rwy, "dwell": dwell, "t_land": t_land, "deice": deice, "first_seen": fs,
                     "t_first": t_first, "switch_before": bool(switch_before is not None and mid in switch_before)}
    arr = mvt.loc[~mvt["is_dep"].to_numpy(dtype=bool)]
    t_arr = (arr["MVT_TIME_UTC_mvt"] - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()
    t_in = (arr["BLOCK_TIME_UTC_mvt"] - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()
    for k, (mid, r) in enumerate(arr.iterrows()):
        apt = str(r["apt"])
        rwys = list(ends(apt))
        rwy = str(r["RUNWAY_mvt"]) if pd.notna(r["RUNWAY_mvt"]) and str(r["RUNWAY_mvt"]) in rwys else rwys[0]
        si = STANDS.index(str(r["STAND_mvt"])) if pd.notna(r["STAND_mvt"]) else 0
        t_land = float(np.floor(t_arr[k]))
        tib = t_in[k] if np.isfinite(t_in[k]) else t_land + 600.0
        arrival(tr, apt, f"b{k:05x}", si, rwy, t_land, tib, lambda t, c=f"ARR{k}": c,
                park_on=float(rng.integers(30, 300)), dt_move=dt_move, dt_stat=dt_stat)
    t_all = (mvt["MVT_TIME_UTC_mvt"] - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    for d0 in np.unique(np.floor(t_all.dropna().to_numpy() / 86_400) * 86_400):
        for j, apt in enumerate(APTS):
            vehicle(tr, apt, f"f{int(d0 // 86_400) % 4096:03x}{j}0", d0 + 6 * 3600)
            satellite(tr, apt, f"e{int(d0 // 86_400) % 4096:03x}{j}0", d0 + 10 * 3600)
    points = tr.frame()
    if no_track or no_gs:
        points = degrade(points, rng, no_track, no_gs)
    return points, pd.DataFrame.from_dict(info, orient="index").rename_axis("mvt_id")


def support_of(mvt: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """The support table (`_adsblib.support_table`) of a synthetic world: every training departure at H1, and the
    holds of every day paired with their airframes' next takeoffs (as `surface.build_support`)."""
    from prc.features import _adsblib, _oplib

    dep = mvt.loc[mvt["is_dep"].to_numpy(dtype=bool) & mvt["split"].eq("train").to_numpy()]
    a = _oplib.anchor_h1(dep)
    _, internals = _adsblib.surface_at(mvt, dep.index, a, {"support": False}, events=events, return_internals=True)
    return _adsblib.support_table(internals, surface.hold_takeoff_pairs(events))


def with_callsigns(mvt: pd.DataFrame) -> pd.DataFrame:
    """The movement frame with ``CALLSIGN_flt`` on NM-matched departures (as in the data)."""
    out = mvt.copy()
    fid = out["FLIGHT_ID_mvt"].to_numpy(dtype="float64")
    cs = np.where(out["is_dep"].to_numpy(dtype=bool) & np.isfinite(fid), np.char.add("OUT", np.nan_to_num(fid).astype(int).astype(str)), None)
    out["CALLSIGN_flt"] = pd.Categorical(cs)
    return out


def table(events: pd.DataFrame) -> pd.DataFrame:
    """Events as the consolidated snapshot table holds them (`surface.consolidate`): block kinds, categoricals."""
    ev = events.loc[events["kind"].isin(surface.BLOCK_KINDS)].copy()
    ev = ev.sort_values(["apt", "kind", "t_known_start", "icao", "t_start"], kind="stable").reset_index(drop=True)
    for c in ("apt", "kind", "icao", "cs", "origin_kind", "rwy_end"):
        ev[c] = ev[c].astype("category")
    return ev
