"""`prc adsb-surface`: the airport surface picture of OTHER aircraft, as causal events, from the adsb.lol points.

One table of episodes per UTC day. Every event records the second it became knowable (``t_known_start``,
``t_known_end``), so a feature block can apply a strict "known by the anchor" rule (`prc.features.operational_adsb`,
tier O; findings 10.31). Events:

* ``TAKEOFF`` -- a ground-to-air transition confirmed by ``AIR_CONFIRM_S`` of airborne points reaching
  ``TKO_MIN_KT``, attributed to a runway of the airport (roll or lift-off point within ``ATT_PERP_*`` of a
  centreline, course within ``ATT_HEADING_DEG`` of the runway heading); known at the confirming airborne point.
  ``t_start`` is the roll (first point >= 100 kt after the last point < 60 kt). Carries its taxi-out's attributes
  (the visit's last episode, when the lift-off came within that episode's timeout of the last ground point) and
  ``taxi_seen``: the taxi was received (a confirmed taxi-out parent, known by the takeoff, begun at least
  ``TAXI_SEEN_MIN_S`` before the roll) -- the completeness the taxi counts need.
* ``LANDING`` -- the first ground point after a confirmed airborne run that reached ``LDG_MIN_KT`` in its last
  ``LDG_LOOK_S`` (many aircraft report airborne through the rollout and reach the ground at 40-58 kt), attributed
  like a takeoff; known then.
* ``TAXI_IN`` -- from touchdown until a stop of ``TAXI_IN_STAND_S`` at a stand, a stop of ``PARK_STAND_S`` outside
  the runway area, the visit's end (lift-off), or ``TIMEOUT_TAXI_IN_S`` without points (critique F10).
* ``TAXI_OUT`` -- ground motion of an airframe not landed in the last ``LANDED_RECENT_S`` (critique F1): it starts at
  a push, at the end of a stop, or at the first ground point of a visit (``origin_kind`` push / stop / first_seen),
  and ends at the takeoff, at a parking stop, or when the airframe vanishes: ``TIMEOUT_MOVING_S`` after a last point
  moving slowly at a stand (a switch-off), ``TIMEOUT_TAXIWAY_S`` after any other moving one (a reception gap must not
  split the taxi), ``STOP_GAP_S`` after a stationary one. Known only once it reached ``CONFIRM_KT`` on the ground
  (tows rarely do; critique F10).
* ``PUSH`` -- the reverse-motion rule of lab/ideas/pushback.py on the thinned track, known at the step that
  satisfies it (>= 2 reverse steps, >= 10 m net).
* ``RWY_HOLD`` -- a stop of ``STOP_MIN_S`` in the runway area (``RWY_END_M`` of an end, ``RWY_CL_M`` of a
  centreline, ``HOLD_M`` of an OSM runway/ILS holding position) by an airframe not landed recently; it needs no
  taxi-out parent (critique F1). ``rwy_end`` = the nearest runway end.
* ``REMOTE_STOP`` -- a stop of ``REMOTE_MIN_S`` outside stands and runway areas inside a confirmed taxi-out
  (de-icing pads, remote holds); it counts at most ``PARK_LONG_S``, after which it is parking.
* ``PARK`` (day files only, QA) -- a stop of ``PARK_STAND_S`` at a stand, or ``PARK_LONG_S`` elsewhere off the runways.

Corrections of the critique built in: each day file is clipped to its own UTC day and (icao, second) deduplicated
(F7); ground points off the airport's own OSM aeroway network (``NET_MAX_M``; ``NET_MOVING_MAX_M`` for moving ones,
which bridges taxiways OSM lacks), or more than ``NET_RWY_MAX_M`` from its runways, are dropped and takeoffs/landings
need a runway of that airport, so the satellite airfields inside the boxes (Le Bourget, Egelsbach, Northolt,
Duebendorf, Torrejon) never enter (F3); an icao counts only once it has been seen airborne at
any earlier time in the snapshot's history (cumulative vehicle filter, F8), and the first ``WARMUP_DAYS`` of a
snapshot's history emit nothing; the state machine is vectorised per day (runs over sorted (icao, t); a Python loop
runs over visits and stops only, F11); timeouts follow the measured cadence (F10); airports are strings (F15d);
outputs are versioned snapshots under cache/adsb_surface/<snapshot>/ (F9).

Causality: a decision taken at time ``tau`` uses only points at or before ``tau``; the absence of points is
observable only up to the build's ``cutoff`` (the last second of data loaded), so a timeout end is written only when
``last point + timeout <= cutoff``, else the end is NaN ("not ended yet"). A lift-off is knowable only at its
confirmation, so it ends what was going on only when that confirmation comes within the timeout. Every timeout
depends on the last point alone (where it was, whether it moved). `episodes(points[t <= a], cutoff=a)` reproduces
every event known by ``a``, on random cutoffs and on every event boundary (tests/test_adsb_surface.py).

Stand ids are OSM node ids or NEGATED way ids (`prc.geometry`), so "no stand" is NaN, never a negative sentinel.

Data: adsb.lol contributors, Open Database License (ODbL 1.0); the derived event tables are share-alike.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from prc import paths
from prc.features._adsblib import norm_callsign  # the one normalisation, shared with the blocks' own-callsign rule
from prc.features._geolib import local_east_north_m, norm_runway_ident

BUILDER_VERSION = 3
"""Part of every day key: bump it whenever the event logic changes (including `_adsblib.norm_callsign`, which the
builder shares with the blocks), so existing snapshots are rebuilt. 3: stand ids may be negative (no -1 sentinel),
lift-offs end episodes only once confirmed, landings from the approach's speed, location-dependent moving timeouts,
moving points bridge OSM gaps, ``taxi_seen`` replaces ``gnd600``."""

# --- points ------------------------------------------------------------------------------------------
THIN_S = 5
"""Tracks are thinned to the first point of every 5-s bin, so the June 2025 change from ~2.5 s to ~1 s sampling does
not change detections (moving ground points are 2-3 s apart, stationary ones ~19 s; critique F10)."""
SEG_GAP_S = 1800
"""Points of one airframe further apart than this belong to separate segments (first_seen after the gap)."""
GROUND_KT = 40.0
"""A point is a ground point when ``on_ground`` or slower than this."""
AIRBORNE_KT = 60.0
"""The vehicle filter: an icao counts as an aircraft once it had an airborne point at least this fast."""
NET_MAX_M = 300.0
"""Aerodrome mask: ground points further than this from the airport's OSM aeroway network are dropped (F3)."""
NET_MOVING_MAX_M = 600.0
MOVING_MASK_KT = 3.0
"""Moving ground points (at least ``MOVING_MASK_KT``) are kept up to ``NET_MOVING_MAX_M`` from the network: taxiways
OSM does not map (EHAM lost 1.2-1.7k points of taxiing aircraft a day at 300-480 m) would otherwise cut gaps into
taxi tracks. The satellite airfields lie kilometres from the airport's network."""
NET_RWY_MAX_M = 3000.0
"""Network points further than this from every runway centreline of the airport are not its own (satellites)."""

# --- takeoffs and landings ------------------------------------------------------------------------------
AIR_CONFIRM_S = 20
"""An airborne run is real (a lift-off, not a flag flicker) once it has lasted this long."""
TKO_MIN_KT = 100.0
ROLL_SLOW_KT = 60.0
LDG_MIN_KT = 60.0
LDG_LOOK_S = 90
"""A landing: the airborne run before the first ground point reached ``LDG_MIN_KT`` within this before it. Aircraft
that report airborne through the rollout reach the ground at 40-58 kt (EHAM: 14 % of its taxi-outs were such
arrivals' taxi-ins when the last airborne point alone decided)."""
ATT_HEADING_DEG = 20.0
ATT_PERP_GROUND_M = 100.0
ATT_PERP_AIR_M = 200.0
ATT_BEFORE_M = 300.0
ATT_BEYOND_GROUND_M = 300.0
ATT_BEYOND_AIR_M = 3000.0
"""An airborne takeoff point (just after lift-off) may lie this far beyond the far end (on the climb-out)."""
ATT_FINAL_AIR_M = 8000.0
"""An airborne landing point (the last one received before touchdown) may lie this far before the threshold (final)."""
TAXI_SEEN_MIN_S = 120
"""``taxi_seen`` of a takeoff: it has a confirmed taxi-out parent that began at least ``TAXI_SEEN_MIN_S`` before the
roll (more than a line-up). The share of takeoffs with it is the completeness of the taxi counts (a ground point near the
roll, the old ``gnd600``, held for nearly every takeoff at any reception). Received rolls are not required: an airframe
lost on the runway is still counted as taxiing until its lift-off is confirmed, and runway reception fails far more
often than taxiway reception (smoke days of July 2025: lift-off within 60 s of the last ground point on 62 % of the
parented takeoffs at EDDF and LSZH, 100 % at EHAM)."""

# --- stops, areas, episodes -------------------------------------------------------------------------------
STATIONARY_KT = 1.0
STATIONARY_MPS = 0.5
"""Points without a ground speed are stationary when they moved slower than this since the previous point."""
STOP_JUMP_M = 15.0
STOP_GAP_S = 600
"""Stationary points up to this far apart (and within ``STOP_JUMP_M``) are one stop; a stop whose airframe sends nothing
for this long has ended (stationary cadence: median ~19 s, p90 55-75 s, 1.3-2 % of intervals above 300 s)."""
STOP_MIN_S = 30
STAND_M = 60.0
STAND_CLOSE_M = 30.0
"""A stop this close to an OSM stand is at the stand even near a runway (74 of LSZH's 372 stands lie in a runway area)."""
STAND_EXCL_END_M = 600.0
STAND_EXCL_CL_M = 150.0
RWY_END_M = 400.0
RWY_CL_M = 90.0
HOLD_M = 60.0
PARK_STAND_S = 300
PARK_LONG_S = 1800
TAXI_IN_STAND_S = 60
LANDED_RECENT_S = 1200
CONFIRM_KT = 12.0
REMOTE_MIN_S = 300
TIMEOUT_TAXI_IN_S = 120
TIMEOUT_MOVING_S = 180
SWITCH_OFF_KT = 6.0
"""A slowly moving airframe (below ``SWITCH_OFF_KT``: pushing, or rolling into a stand) last seen within ``STAND_M`` of
a stand has gone (switched off) after ``TIMEOUT_MOVING_S`` without points."""
TIMEOUT_TAXIWAY_S = 600
"""Any other moving airframe (on the taxiways, the runway area, or taxiing along the apron past stands) is kept this
long: reception gaps there split one taxi into several episodes otherwise (LEMD 2026-07-14: 21 % of departures, against
1 % in 2025; with the stand radius alone, LEBL in July 2026 still lost 14 % of its taxis to 3-5 min gaps on apron
taxilanes within 60 m of stands)."""
ORIGIN_MAX_M = 150.0
"""An episode's origin point maps to the nearest OSM stand within this (else no origin stand)."""
EPISODE_MAX_S = 3 * 3600
"""Parent attributes (origin, duration) are only carried when the taxi-out began at most this long before."""
END_MAX_S = 4 * 3600
"""An end more than this after the event's start is written as NaN (not ended within the horizon)."""

# --- pushback (lab/ideas/pushback.py) ------------------------------------------------------------------
REV_MIN_STEP_M = 2.0
REV_MIN_DIFF = 110.0
FWD_MAX_DIFF = 70.0
FWD_BIG_STEP_M = 4.0
PUSH_GAP_S = 120.0
PUSH_MIN_REV = 2
PUSH_MIN_NET_M = 10.0
PUSH_MAX_KT = 15.0
PUSH_MAX_MPS = 8.0
PUSH_START_STEPS = 3
"""A push starts a taxi-out when its first reverse step is among the first steps of the episode's motion."""

# --- day windows ------------------------------------------------------------------------------------------
LOOKBACK_S = 6 * 3600
LOOKAHEAD_S = 5 * 3600
WARMUP_DAYS = 3
"""The first days of a snapshot's history emit no events: the cumulative vehicle filter is still empty (F8)."""

KINDS = ("TAKEOFF", "LANDING", "TAXI_OUT", "TAXI_IN", "PUSH", "RWY_HOLD", "REMOTE_STOP", "PARK")
BLOCK_KINDS = KINDS[:-1]
"""The kinds the consolidated table keeps (PARK stays in the day files, for QA)."""
ORIGIN_KINDS = ("push", "stop", "first_seen")
EVENT_COLUMNS = (
    "eid", "apt", "kind", "icao", "cs", "t_start", "t_known_start", "t_end", "t_known_end",
    "origin_kind", "origin_stand", "rwy_end", "dur", "stop_s", "path_m", "taxi_seen", "parent",
)
POINT_COLUMNS = ("icao", "t", "lat", "lon", "alt_ft", "on_ground", "gs_kt", "track", "callsign", "apt")
DAY_S = 86_400


# ============================================================================== geometry

@dataclass
class AirportGeometry:
    """One airport's reference objects in local metres (east, north of the reference point)."""

    apt: str
    lat0: float
    lon0: float
    end_ident: np.ndarray            # (m,) normalised idents of the valid runway ends
    end_xy: np.ndarray               # (m, 2) thresholds
    end_u: np.ndarray                # (m, 2) unit vector of each end's heading (the direction of a movement from it)
    end_heading: np.ndarray          # (m,) true heading, degrees
    end_length: np.ndarray           # (m,) runway length, metres
    segs: list = field(default_factory=list)   # centrelines [(a_xy, b_xy)]
    stand_id: np.ndarray = None      # (n,) OSM stand ids (int64)
    stand_xy: np.ndarray = None      # (n, 2)
    hold_xy: np.ndarray = None       # (k, 2) OSM runway / ILS holding positions
    net_xy: np.ndarray = None        # (j, 2) the airport's own aeroway network, densified

    def __post_init__(self):
        from scipy.spatial import cKDTree

        self.stand_id = np.asarray(self.stand_id if self.stand_id is not None else [], dtype="int64")
        self.stand_xy = _xy_array(self.stand_xy)
        self.hold_xy = _xy_array(self.hold_xy)
        self.net_xy = _xy_array(self.net_xy)
        self._stand_tree = cKDTree(self.stand_xy) if len(self.stand_xy) else None
        self._hold_tree = cKDTree(self.hold_xy) if len(self.hold_xy) else None
        self._net_tree = cKDTree(self.net_xy) if len(self.net_xy) else None

    def xy(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        return local_east_north_m(lat, lon, self.lat0, self.lon0)

    def d_end(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """Distance to the nearest runway end, and its index (-1 without ends)."""
        if not len(self.end_xy):
            return np.full(len(x), np.inf), np.full(len(x), -1)
        d = np.hypot(x[:, None] - self.end_xy[None, :, 0], y[:, None] - self.end_xy[None, :, 1])
        k = np.argmin(d, axis=1)
        return d[np.arange(len(x)), k], k

    def d_centreline(self, x, y) -> np.ndarray:
        """Perpendicular distance to a runway centreline, where the foot lies between the ends."""
        best = np.full(len(x), np.inf)
        p = np.column_stack([x, y])
        for a, b in self.segs:
            ab = b - a
            L2 = float(ab @ ab)
            if L2 <= 0:
                continue
            u = ((p - a) @ ab) / L2
            foot = a + np.outer(u, ab)
            d = np.hypot(*(p - foot).T)
            best = np.minimum(best, np.where((u >= 0) & (u <= 1), d, np.inf))
        return best

    def d_runway(self, x, y) -> np.ndarray:
        """Distance to the nearest runway centreline segment, clamped to its ends (the aerodrome extent)."""
        best = np.full(len(x), np.inf)
        p = np.column_stack([x, y])
        for a, b in self.segs:
            ab = b - a
            L2 = float(ab @ ab)
            u = np.clip(((p - a) @ ab) / L2, 0.0, 1.0) if L2 > 0 else np.zeros(len(p))
            foot = a + np.outer(u, ab)
            best = np.minimum(best, np.hypot(*(p - foot).T))
        if not self.segs and len(self.end_xy):
            best = self.d_end(x, y)[0]
        return best

    def nearest_stand(self, x, y, max_m: float) -> tuple[np.ndarray, np.ndarray]:
        """Distance to the nearest OSM stand, and its id where within ``max_m``, else NaN (float64).

        Stand ids are OSM node ids or negated way ids (most stands at EDDF, LEBL, LEMD and LSZH are ways), so no
        integer can mean "none"; float64 holds every id exactly (below 2**53).
        """
        if self._stand_tree is None or not len(x):
            return np.full(len(x), np.inf), np.full(len(x), np.nan)
        d, k = self._stand_tree.query(np.column_stack([x, y]))
        return d, np.where(d <= max_m, self.stand_id[k].astype("float64"), np.nan)

    def d_hold(self, x, y) -> np.ndarray:
        if self._hold_tree is None or not len(x):
            return np.full(len(x), np.inf)
        return self._hold_tree.query(np.column_stack([x, y]))[0]

    def d_net(self, x, y) -> np.ndarray:
        if self._net_tree is None or not len(x):
            return np.full(len(x), np.inf)
        return self._net_tree.query(np.column_stack([x, y]))[0]

    def attribute(self, x, y, course, airborne, landing: bool = False) -> np.ndarray:
        """The runway end a movement at (x, y) with ``course`` (deg) uses: aligned and on its (extended) centreline,
        else None. Airborne points may lie beyond the far end on a climb-out, or before the threshold on a final
        (``landing``)."""
        out = np.full(len(x), None, dtype=object)
        if not len(self.end_xy) or not len(x):
            return out
        rel_x = x[:, None] - self.end_xy[None, :, 0]
        rel_y = y[:, None] - self.end_xy[None, :, 1]
        along = rel_x * self.end_u[None, :, 0] + rel_y * self.end_u[None, :, 1]
        perp = np.abs(rel_x * self.end_u[None, :, 1] - rel_y * self.end_u[None, :, 0])
        diff = np.abs((np.asarray(course, dtype="float64")[:, None] - self.end_heading[None, :] + 180.0) % 360.0 - 180.0)
        air = np.asarray(airborne, dtype=bool)[:, None]
        before = np.where(air & landing, ATT_FINAL_AIR_M, ATT_BEFORE_M)
        beyond = np.where(air & (not landing), ATT_BEYOND_AIR_M, ATT_BEYOND_GROUND_M)
        ok = (
            np.isfinite(diff) & (diff <= ATT_HEADING_DEG)
            & (perp <= np.where(air, ATT_PERP_AIR_M, ATT_PERP_GROUND_M))
            & (along >= -before)
            & (along <= self.end_length[None, :] + beyond)
        )
        score = np.where(ok, perp, np.inf)
        k = np.argmin(score, axis=1)
        hit = np.isfinite(score[np.arange(len(x)), k])
        out[hit] = self.end_ident[k[hit]]
        return out


def _xy_array(v) -> np.ndarray:
    a = np.asarray(v if v is not None else np.zeros((0, 2)), dtype="float64")
    return a.reshape(-1, 2) if a.size else np.zeros((0, 2))


def make_geometry(apt: str, arp_lat: float, arp_lon: float, ends: pd.DataFrame, stands: pd.DataFrame | None = None,
                  holds=None, ways=()) -> AirportGeometry:
    """An `AirportGeometry` from plain tables (lat/lon degrees).

    ``ends``: ident, lat, lon, heading_deg, length_m, pair_ref (valid ends only); ``stands``: stand_id, lat, lon;
    ``holds``: (k, 2) lat/lon; ``ways``: polylines of the aeroway network, each (j, 2) lat/lon. The network is
    densified to points at most 25 m apart, completed by the stand positions, and cut to ``NET_RWY_MAX_M`` of the
    airport's own runways.
    """
    def xy(lat, lon):
        e, n = local_east_north_m(lat, lon, arp_lat, arp_lon)
        return np.column_stack([np.atleast_1d(e), np.atleast_1d(n)])

    ends = ends.reset_index(drop=True)
    exy = xy(ends["lat"].to_numpy(float), ends["lon"].to_numpy(float)) if len(ends) else np.zeros((0, 2))
    heading = ends["heading_deg"].to_numpy(float) if len(ends) else np.zeros(0)
    u = np.column_stack([np.sin(np.radians(heading)), np.cos(np.radians(heading))]) if len(ends) else np.zeros((0, 2))
    length = ends["length_m"].to_numpy(float) if len(ends) else np.zeros(0)
    segs = []
    for _, grp in ends.groupby("pair_ref", sort=True) if len(ends) else ():
        if len(grp) == 2 and grp["pair_ref"].iloc[0]:
            p = exy[grp.index.to_numpy()]
            segs.append((p[0], p[1]))
    if len(ends):  # a runway without a length in the table: assume a long one (the attribution window only)
        length = np.where(~np.isfinite(length) | (length <= 0), 3000.0, length)
    if stands is not None and len(stands):
        sxy = xy(stands["lat"].to_numpy(float), stands["lon"].to_numpy(float))
        sid = stands["stand_id"].to_numpy(dtype="int64")
    else:
        sxy, sid = np.zeros((0, 2)), np.zeros(0, dtype="int64")
    hxy = xy(np.asarray(holds)[:, 0], np.asarray(holds)[:, 1]) if holds is not None and len(holds) else np.zeros((0, 2))
    net = [sxy]
    for w in ways:
        w = np.asarray(w, dtype="float64")
        if len(w) < 1:
            continue
        pts = xy(w[:, 0], w[:, 1])
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(1, int(math.ceil(float(np.hypot(*(b - a))) / 25.0)))
            f = np.linspace(0.0, 1.0, n + 1)[:, None]
            net.append(a + f * (b - a))
        if len(pts) == 1:
            net.append(pts)
    net_xy = np.vstack([n for n in net if len(n)]) if any(len(n) for n in net) else np.zeros((0, 2))
    ident = np.array([norm_runway_ident(i) or str(i) for i in ends["ident"]], dtype=object) if len(ends) else np.zeros(0, dtype=object)
    geo = AirportGeometry(apt=apt, lat0=arp_lat, lon0=arp_lon, end_ident=ident, end_xy=exy, end_u=u, end_heading=heading,
                          end_length=length, segs=segs, stand_id=sid, stand_xy=sxy, hold_xy=hxy, net_xy=np.zeros((0, 2)))
    if len(net_xy) and (segs or len(exy)):  # the network points of this airport only (satellite airfields cut, F3)
        net_xy = net_xy[geo.d_runway(net_xy[:, 0], net_xy[:, 1]) <= NET_RWY_MAX_M]
    return AirportGeometry(apt=apt, lat0=arp_lat, lon0=arp_lon, end_ident=ident, end_xy=exy, end_u=u, end_heading=heading,
                           end_length=length, segs=segs, stand_id=sid, stand_xy=sxy, hold_xy=hxy, net_xy=net_xy)


def _osm_objects(apt: str):
    """Holding positions (runway / ILS / untyped) and the aeroway network polylines of one airport's raw OSM extract."""
    raw = paths.GEOMETRY_RAW / "osm" / f"{apt}.json"
    if not raw.exists():
        return np.zeros((0, 2)), []
    els = json.loads(raw.read_text(encoding="utf-8"))["elements"]
    nodes = {el["id"]: (el["lat"], el["lon"]) for el in els if el["type"] == "node" and "lat" in el}
    holds, ways = [], []
    for el in els:
        tg = el.get("tags", {})
        aw = tg.get("aeroway")
        if aw == "holding_position" and tg.get("holding_position:type") != "intermediate":
            if el["type"] == "node" and "lat" in el:
                holds.append((el["lat"], el["lon"]))
            elif el["type"] == "way":
                holds.extend(nodes[n] for n in el.get("nodes", []) if n in nodes)
        if el["type"] == "way" and aw in ("taxiway", "taxilane", "runway", "parking_position", "apron", "stopway"):
            ll = [nodes[n] for n in el.get("nodes", []) if n in nodes]
            if ll:
                ways.append(np.array(ll, dtype="float64"))
        elif el["type"] == "node" and aw in ("gate", "parking_position", "holding_position") and "lat" in el:
            ways.append(np.array([(el["lat"], el["lon"])], dtype="float64"))
    return (np.array(holds, dtype="float64") if holds else np.zeros((0, 2))), ways


@lru_cache(maxsize=2)
def _load_geometry(digest: str) -> dict[str, AirportGeometry]:
    airports = pd.read_parquet(paths.GEO_AIRPORTS_FILE)
    ends = pd.read_parquet(paths.GEO_RUNWAY_ENDS_FILE)
    ends = ends.loc[ends["dropped_reason"].fillna("").eq("") & ends["lat"].notna()]  # '' is a valid end (F17a)
    stands = pd.read_parquet(paths.GEO_STANDS_FILE, columns=["apt", "stand_id", "lat", "lon"])
    out = {}
    for r in airports.itertuples():
        apt = str(r.apt)
        e = ends.loc[ends["apt"].astype(str).eq(apt), ["ident", "lat", "lon", "heading_deg", "length_m", "pair_ref"]]
        s = stands.loc[stands["apt"].astype(str).eq(apt)]
        holds, ways = _osm_objects(apt)
        out[apt] = make_geometry(apt, float(r.arp_lat), float(r.arp_lon), e, s, holds, ways)
    return out


def geometry_digest() -> str:
    """sha1 over the geometry inputs the builder reads (the normalised tables and the raw OSM extracts)."""
    h = hashlib.sha1(f"surface-geometry-{BUILDER_VERSION}".encode())
    files = [paths.GEO_AIRPORTS_FILE, paths.GEO_RUNWAY_ENDS_FILE, paths.GEO_STANDS_FILE]
    files += sorted((paths.GEOMETRY_RAW / "osm").glob("*.json"))
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes() if f.exists() else b"missing")
    return h.hexdigest()[:12]


def load_geometry() -> dict[str, AirportGeometry]:
    return _load_geometry(geometry_digest())


# ============================================================================== points

def clean_points(points: pd.DataFrame, cutoff: float | None = None) -> pd.DataFrame:
    """Typed points on whole seconds, at or before ``cutoff``, one per (icao, second), sorted by (icao, t)."""
    p = points.loc[:, [c for c in POINT_COLUMNS if c in points.columns]].copy()
    for c in POINT_COLUMNS:
        if c not in p.columns:
            p[c] = np.nan
    p["t"] = np.floor(pd.to_numeric(p["t"], errors="coerce").to_numpy(dtype="float64"))
    p = p.loc[np.isfinite(p["t"].to_numpy()) & p["lat"].notna().to_numpy() & p["lon"].notna().to_numpy()]
    if cutoff is not None:
        p = p.loc[p["t"].to_numpy() <= cutoff]
    p["icao"] = p["icao"].astype(str).str.lower()
    p["apt"] = p["apt"].astype(str)  # a per-file dictionary<int8>: codes differ between days (F15d)
    p["on_ground"] = p["on_ground"].fillna(False).astype(bool)
    for c in ("gs_kt", "track", "alt_ft", "lat", "lon"):
        p[c] = pd.to_numeric(p[c], errors="coerce").astype("float64")
    p = p.sort_values(["icao", "t"], kind="stable")
    p = p.loc[~p.duplicated(["icao", "t"], keep="first")]
    return p.reset_index(drop=True)


def airborne_first_times(points: pd.DataFrame) -> pd.Series:
    """icao -> first second it was seen airborne (not on the ground, at least ``AIRBORNE_KT``), over ``points``."""
    gs = pd.to_numeric(points["gs_kt"], errors="coerce").to_numpy(dtype="float64")
    air = ~points["on_ground"].fillna(False).to_numpy(dtype=bool) & (gs >= AIRBORNE_KT)
    sub = points.loc[air]
    if sub.empty:
        return pd.Series(dtype="float64", name="first_air")
    t = np.floor(pd.to_numeric(sub["t"]).to_numpy(dtype="float64"))
    return pd.Series(t, index=sub["icao"].astype(str).str.lower().to_numpy()).groupby(level=0).min().rename("first_air")


# ============================================================================== the builder

def _course(x, y, track, same_prev) -> np.ndarray:
    """The reported track, else the bearing of the motion from the previous point (>= 5 m)."""
    dx = np.diff(x, prepend=np.nan)
    dy = np.diff(y, prepend=np.nan)
    step = np.hypot(dx, dy)
    brg = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
    brg = np.where(same_prev & (step >= 5.0), brg, np.nan)
    return np.where(np.isfinite(track), track, brg)


def _next_index(flags: np.ndarray) -> np.ndarray:
    """For every position, the first position at or after it where ``flags`` is true (``len + 1`` if none)."""
    big = len(flags) + 1
    idx = np.where(flags, np.arange(len(flags)), big)
    return np.minimum.accumulate(idx[::-1])[::-1] if len(flags) else idx


def episodes(points: pd.DataFrame, geometry: dict[str, AirportGeometry], cutoff: float,
             airborne_first: pd.Series | None = None) -> pd.DataFrame:
    """Every surface event of every airframe in ``points``, with the second each became knowable.

    ``points``: adsb.lol points (`POINT_COLUMNS`), already clipped to the days they belong to. ``cutoff``: the last
    second of data loaded (timeouts are only observable up to it; points after it are dropped). ``airborne_first``:
    icao -> first airborne second over the snapshot's history (the vehicle filter); computed from ``points`` when
    None. Pure: nothing is read.
    """
    p = clean_points(points, cutoff)
    if airborne_first is None:
        airborne_first = airborne_first_times(p)
    if p.empty:
        return _empty_events()
    # thinning: the first point of every THIN_S-second bin per airframe
    p = p.loc[~p.assign(_b=(p["t"].to_numpy() // THIN_S)).duplicated(["icao", "_b"])]
    p = p.loc[p["apt"].isin(list(geometry))]
    if p.empty:
        return _empty_events()
    gs_all = p["gs_kt"].to_numpy(dtype="float64")
    ground_all = p["on_ground"].to_numpy(dtype=bool) | (gs_all < GROUND_KT)
    moving_all = np.nan_to_num(gs_all, nan=0.0) >= MOVING_MASK_KT
    x_all = np.full(len(p), np.nan)
    y_all = np.full(len(p), np.nan)
    keep = np.zeros(len(p), dtype=bool)
    apt_all = p["apt"].to_numpy(dtype=object)
    for apt in pd.unique(apt_all):
        rows = np.flatnonzero(apt_all == apt)
        g = geometry[apt]
        x, y = g.xy(p["lat"].to_numpy()[rows], p["lon"].to_numpy()[rows])
        x_all[rows], y_all[rows] = x, y
        gr = ground_all[rows]
        ok = ~gr
        if gr.any():  # the aerodrome mask (critique F3): ground points on this airport's own aeroway network only
            gx, gy = x[gr], y[gr]
            d_net = g.d_net(gx, gy)
            on_net = (d_net <= NET_MAX_M) | (moving_all[rows][gr] & (d_net <= NET_MOVING_MAX_M))
            ok[gr] = on_net & (g.d_runway(gx, gy) <= NET_RWY_MAX_M)
        keep[rows] = ok
    p = p.loc[keep].assign(x=x_all[keep], y=y_all[keep])
    p = p.sort_values(["apt", "icao", "t"], kind="stable").reset_index(drop=True)
    return _events_from_points(p, geometry, float(cutoff), airborne_first)


def _empty_events() -> pd.DataFrame:
    frame = pd.DataFrame({c: pd.Series(dtype="float64") for c in EVENT_COLUMNS})
    for c in ("apt", "kind", "icao", "cs", "origin_kind", "rwy_end"):
        frame[c] = frame[c].astype(object)
    frame["eid"] = frame["eid"].astype("int64")
    frame["parent"] = frame["parent"].astype("int64")
    frame["taxi_seen"] = frame["taxi_seen"].astype("int8")
    return frame


def _eid(icao, kind, t_start) -> np.ndarray:
    """A content id per event: (icao, kind, start) -> int64, stable between builds of overlapping windows."""
    out = np.zeros(len(icao), dtype="int64")
    for i, (ic, k, ts) in enumerate(zip(icao, kind, t_start)):
        try:
            a = int(ic, 16) & 0xFFFFFF
        except (TypeError, ValueError):
            a = int(hashlib.sha1(str(ic).encode()).hexdigest()[:6], 16)
        out[i] = (a << 36) | ((KINDS.index(k) & 0xF) << 32) | ((int(ts) & 0xFFFFFFFF) if np.isfinite(ts) else 0)
    return out


class _Track:
    """The point arrays of one build, sorted by (airport, icao, t), with runs, visits, stops and pushes."""

    def __init__(self, p: pd.DataFrame, geometry: dict[str, AirportGeometry], cutoff: float):
        self.geometry = geometry
        self.cutoff = cutoff
        self.n = n = len(p)
        self.t = p["t"].to_numpy(dtype="float64")
        self.x = p["x"].to_numpy(dtype="float64")
        self.y = p["y"].to_numpy(dtype="float64")
        self.gs = p["gs_kt"].to_numpy(dtype="float64")
        self.trk = p["track"].to_numpy(dtype="float64")
        self.apt = p["apt"].to_numpy(dtype=object)
        self.icao = p["icao"].to_numpy(dtype=object)
        self.cs = norm_callsign(p["callsign"].to_numpy(dtype=object))
        self.ground = p["on_ground"].to_numpy(dtype=bool) | (self.gs < GROUND_KT)
        t = self.t
        same = np.zeros(n, dtype=bool)
        same[1:] = (self.icao[1:] == self.icao[:-1]) & (self.apt[1:] == self.apt[:-1]) & (t[1:] - t[:-1] <= SEG_GAP_S)
        self.same_prev = same
        self.course = _course(self.x, self.y, self.trk, same)
        self._runs()
        self._visits()
        self._ground_points()
        self._stops()
        self._pushes()

    # --- runs of ground / air points; an airborne run is real once it lasted AIR_CONFIRM_S ---------------
    def _runs(self):
        t, n = self.t, self.n
        new_run = ~self.same_prev.copy()
        new_run[1:] |= self.ground[1:] != self.ground[:-1]
        self.run = run = np.cumsum(new_run) - 1
        r_first = np.flatnonzero(new_run)
        r_last = np.append(r_first[1:] - 1, n - 1)
        r_ground = self.ground[r_first]
        t0 = t.min()
        span = int(t.max() - t0) + 10 * AIR_CONFIRM_S + 10
        key = run.astype("int64") * span + (t - t0).astype("int64")
        target = run[r_first].astype("int64") * span + (t[r_first] - t0).astype("int64") + AIR_CONFIRM_S
        conf = np.searchsorted(key, target, side="left")
        ok = ~r_ground & (conf <= r_last)
        self.r_first = r_first
        self.r_conf = np.where(ok, conf, -1)          # the confirming point of every real airborne run
        self.real_air = ok[run]                        # per point

    # --- visits: maximal stretches of a segment between real airborne runs ------------------------------
    def _visits(self):
        n, real = self.n, self.real_air
        prev_real = np.concatenate([[False], real[:-1]])
        new_visit = ~real & (~self.same_prev | prev_real)
        self.vid = vid = np.where(real, -1, np.cumsum(new_visit) - 1)
        self.nv = nv = int(new_visit.sum())
        self.v_first = np.flatnonzero(new_visit)
        pts = np.flatnonzero(vid >= 0)
        self.v_last = np.zeros(nv, dtype="int64")
        if nv:
            np.maximum.at(self.v_last, vid[pts], pts)
        prev_i = np.maximum(self.v_first - 1, 0)
        after_air = (self.v_first > 0) & self.same_prev[self.v_first] & real[prev_i]
        # a landing: the approach reached LDG_MIN_KT within LDG_LOOK_S before the first ground point (an aircraft that
        # reports airborne through its rollout has a slow last airborne point, already on the runway)
        self.v_landing = np.zeros(nv, dtype=bool)
        for v in np.flatnonzero(after_air):
            fi, li = int(self.v_first[v]), int(self.v_first[v]) - 1
            r0 = int(self.r_first[self.run[li]])
            lo = min(r0 + int(np.searchsorted(self.t[r0:li + 1], self.t[fi] - LDG_LOOK_S, side="left")), li)
            self.v_landing[v] = np.nanmax(np.append(self.gs[lo:li + 1], -1.0)) >= LDG_MIN_KT
        nxt = np.minimum(self.v_last + 1, n - 1)
        self.v_liftoff = (self.v_last + 1 < n) & self.same_prev[nxt] & real[nxt]
        # the time after which nothing of the visit's events can become known (vehicle pre-filter bound)
        conf_t = np.where(self.v_liftoff, self.t[np.maximum(self.r_conf[self.run[nxt]], 0)], -np.inf)
        self.v_tmax = np.maximum(self.t[self.v_last], conf_t)

    def last_ground_t(self, v: int) -> float:
        k_b = self.g_last[v]
        return float(self.t[self.gi[k_b]] if k_b >= 0 else self.t[self.v_last[v]])

    def visit_end(self, v: int, timeout: float):
        """(t_end, t_known_end, kind) of what is going on at the visit's end, for a state with ``timeout``.

        A lift-off ends it only when its CONFIRMATION (the moment it becomes knowable) comes within the timeout of the
        visit's last ground point; otherwise the timeout passed first: a vanish at the last point, known ``timeout``
        later -- the decision a build censored between the timeout and the confirmation takes too. A visit without a
        lift-off vanishes the same way, once the timeout is observable by the cutoff (else the end is unknown: NaN).
        Timeouts run from the last GROUND point: trailing airborne points that are not (yet) a confirmed lift-off may
        still become one.
        """
        tl = self.last_ground_t(v)
        if self.v_liftoff[v]:
            li = self.v_last[v] + 1
            conf = self.t[self.r_conf[self.run[li]]]
            if conf - tl <= timeout:
                return self.t[li], conf, "liftoff"
        known = tl + timeout if tl + timeout <= self.cutoff else np.nan
        return (tl if np.isfinite(known) else np.nan), known, "vanish"

    def lift_within(self, v: int, timeout: float) -> bool:
        """The visit's lift-off followed its last ground point within ``timeout`` (whenever it was confirmed): the
        visit's last taxi-out is the takeoff's, also when the confirmation came after the timeout had ended it."""
        return bool(self.v_liftoff[v] and self.t[self.v_last[v] + 1] - self.last_ground_t(v) <= timeout)

    # --- ground points of visits --------------------------------------------------------------------
    def _ground_points(self):
        self.gi = gi = np.flatnonzero((self.vid >= 0) & self.ground)
        self.ng = ng = len(gi)
        self.g_vid = g_vid = self.vid[gi]
        prev_ok = np.zeros(ng, dtype=bool)
        prev_ok[1:] = g_vid[1:] == g_vid[:-1]
        self.g_prev_ok = prev_ok
        self.gx, self.gy, self.gt, self.ggs = self.x[gi], self.y[gi], self.t[gi], self.gs[gi]
        self.dx = np.where(prev_ok, np.diff(self.gx, prepend=np.nan), np.nan)
        self.dy = np.where(prev_ok, np.diff(self.gy, prepend=np.nan), np.nan)
        self.dt = np.where(prev_ok, np.diff(self.gt, prepend=np.nan), np.nan)
        self.step = np.hypot(self.dx, self.dy)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.vel = self.step / np.maximum(self.dt, 0.5)
            self.stationary = np.where(np.isfinite(self.ggs), self.ggs < STATIONARY_KT,
                                       prev_ok & (self.vel < STATIONARY_MPS) & (self.step < STOP_JUMP_M))
        moving_step = prev_ok & np.isfinite(self.step) & ~self.stationary
        self.cum_path = np.cumsum(np.where(moving_step, np.round(np.nan_to_num(self.step)), 0.0).astype("int64"))  # whole metres: exact
        self.g_first = np.full(self.nv, -1, dtype="int64")
        self.g_last = np.full(self.nv, -1, dtype="int64")
        if ng:
            np.maximum.at(self.g_last, g_vid, np.arange(ng))
            u, first = np.unique(g_vid, return_index=True)
            self.g_first[u] = first
        self.next_fast = _next_index(np.isfinite(self.ggs) & (self.ggs >= CONFIRM_KT))
        self.next_moving = _next_index(~self.stationary)
        self.BIG = ng + 1
        # the timeout after each ground point, from that point alone: stationary STOP_GAP_S; moving slowly at a stand (a
        # switch-off ends motion there) TIMEOUT_MOVING_S; any other motion TIMEOUT_TAXIWAY_S (reception gaps)
        near = np.zeros(ng, dtype=bool)
        g_apt = self.apt[gi]
        for a in pd.unique(g_apt):
            rows = np.flatnonzero(g_apt == a)
            near[rows] = self.geometry[a].nearest_stand(self.gx[rows], self.gy[rows], STAND_M)[0] <= STAND_M
        slow = np.where(np.isfinite(self.ggs), self.ggs < SWITCH_OFF_KT, np.nan_to_num(self.vel, nan=0.0) < SWITCH_OFF_KT * 0.5144)
        self.timeout = np.where(self.stationary, float(STOP_GAP_S),
                                np.where(near & slow, float(TIMEOUT_MOVING_S), float(TIMEOUT_TAXIWAY_S)))
        # internal gaps of a visit: a timeout ends whatever was going on, whether or not the airframe comes back later
        # (a decision taken when the timeout passes, so a build censored there agrees with the full build)
        nxt_dt = np.full(ng, np.inf)
        if ng > 1:
            same_next = g_vid[1:] == g_vid[:-1]
            nxt_dt[:-1] = np.where(same_next, self.gt[1:] - self.gt[:-1], np.inf)
        self.nxt_dt = nxt_dt
        internal = np.isfinite(nxt_dt)
        self.gap_after = internal & (nxt_dt > self.timeout)
        self.next_gap = _next_index(self.gap_after)
        self.next_gap_in = _next_index(internal & (nxt_dt > TIMEOUT_TAXI_IN_S))

    # --- stops: runs of stationary ground points (gaps <= STOP_GAP_S, jumps <= STOP_JUMP_M) -----------------
    def _stops(self):
        ng, st, gt = self.ng, self.stationary, self.gt
        cont = np.zeros(ng, dtype=bool)
        if ng > 1:
            cont[1:] = st[1:] & st[:-1] & self.g_prev_ok[1:] & (self.dt[1:] <= STOP_GAP_S) & (self.step[1:] <= STOP_JUMP_M)
        start = st & ~cont
        self.stop_of = stop_of = np.where(st, np.cumsum(start) - 1, -1)
        self.s_k0 = k0 = np.flatnonzero(start)
        self.ns = ns = len(k0)
        k1 = np.zeros(ns, dtype="int64")
        st_pts = np.flatnonzero(st)
        if ns:
            np.maximum.at(k1, stop_of[st_pts], st_pts)
        self.s_k1 = k1
        self.s_vid = s_vid = self.g_vid[k0] if ns else np.zeros(0, dtype="int64")
        self.s_t0 = t0 = gt[k0]
        tl = gt[k1]
        # the end: the next ground point of the visit within STOP_GAP_S (motion, a jump, a new stop) ...
        nk = np.minimum(k1 + 1, max(ng - 1, 0))
        nk_ok = (k1 + 1 < ng) & (self.g_vid[nk] == s_vid) if ns else np.zeros(0, dtype=bool)
        gap = np.where(nk_ok, gt[nk] - tl, np.inf)
        observed = nk_ok & (gap <= STOP_GAP_S)
        tend = np.where(observed, gt[nk], np.nan)
        tkend = tend.copy()
        end_pt = np.where(observed, self.gi[nk], -1)
        # ... else the visit's lift-off (observed at its confirmation), else a vanish after STOP_GAP_S
        last = ~nk_ok
        li = np.minimum(self.v_last[s_vid] + 1, self.n - 1) if ns else np.zeros(0, dtype="int64")
        lift = last & self.v_liftoff[s_vid] if ns else np.zeros(0, dtype=bool)
        ci = np.where(lift, self.r_conf[self.run[li]], -1) if ns else np.zeros(0, dtype="int64")
        if ns:  # a lift-off ends the stop only when it is CONFIRMED (knowable) within STOP_GAP_S of the stop's last point
            lift &= self.t[np.maximum(ci, 0)] - tl <= STOP_GAP_S
            ci = np.where(lift, ci, -1)
        tend = np.where(lift, self.t[li], tend)
        tkend = np.where(lift, self.t[np.maximum(ci, 0)], tkend)
        end_pt = np.where(lift, ci, end_pt)
        vanish = ~observed & ~lift & (tl + STOP_GAP_S <= self.cutoff)
        tend = np.where(vanish, tl, tend)
        tkend = np.where(vanish, tl + STOP_GAP_S, tkend)
        self.s_tend, self.s_tkend, self.s_observed = tend, tkend, observed | lift
        # qualification for a duration X: the first point of the stop at least X after its start, else its observed end
        g0 = gt.min() if ng else 0.0
        span = int((gt.max() - g0) if ng else 0) + PARK_LONG_S + 10
        skey = stop_of[st_pts].astype("int64") * span + (gt[st_pts] - g0).astype("int64")

        def qualify(sec: int):
            target = np.arange(ns, dtype="int64") * span + (t0 - g0).astype("int64") + sec
            pos = np.searchsorted(skey, target, side="left")
            pos_c = np.minimum(pos, max(len(st_pts) - 1, 0))
            inside = (pos < len(st_pts)) & (stop_of[st_pts[pos_c]] == np.arange(ns)) if len(st_pts) else np.zeros(ns, bool)
            tq = np.where(inside, gt[st_pts[pos_c]], np.nan)
            qi = np.where(inside, self.gi[st_pts[pos_c]], -1)
            by_end = ~inside & self.s_observed & np.isfinite(tend) & (tend - t0 >= sec)
            return np.where(by_end, tkend, tq), np.where(by_end, end_pt, qi)

        self.tq30, self.qi30 = qualify(STOP_MIN_S)
        self.tq60, _ = qualify(TAXI_IN_STAND_S)
        self.tq300, self.qi300 = qualify(PARK_STAND_S)
        self.tq1800, _ = qualify(PARK_LONG_S)
        # where each stop is (its first point): stand, runway area, or elsewhere
        self.s_apt = s_apt = self.apt[self.gi[k0]] if ns else np.zeros(0, dtype=object)
        self.s_cls = np.zeros(ns, dtype="int8")
        self.s_stand = np.full(ns, np.nan)
        self.s_rwy_end = np.full(ns, None, dtype=object)
        for a in pd.unique(s_apt):
            rows = np.flatnonzero(s_apt == a)
            g = self.geometry[a]
            sx, sy = self.gx[k0[rows]], self.gy[k0[rows]]
            d_end, k_end = g.d_end(sx, sy)
            d_cl = g.d_centreline(sx, sy)
            d_st, st_id = g.nearest_stand(sx, sy, STAND_M)
            rwy = (d_end <= RWY_END_M) | (d_cl <= RWY_CL_M) | (g.d_hold(sx, sy) <= HOLD_M)
            stand = (d_st <= STAND_CLOSE_M) | ((d_st <= STAND_M) & (d_end > STAND_EXCL_END_M) & (d_cl > STAND_EXCL_CL_M))
            self.s_cls[rows] = np.where(stand, 1, np.where(rwy, 2, 0))
            self.s_stand[rows] = st_id
            if len(g.end_ident):
                self.s_rwy_end[rows] = np.where(rwy, g.end_ident[np.maximum(k_end, 0)], None)
        self.park_known = np.fmin(np.where(self.s_cls == 1, self.tq300, np.nan), np.where(self.s_cls != 2, self.tq1800, np.nan))
        self.stops_of_visit = pd.Series(np.arange(ns)).groupby(s_vid).agg(list).to_dict() if ns else {}

    # --- pushes: reverse steps against the reported track (lab/ideas/pushback.py) -------------------------
    def _pushes(self):
        ng = self.ng
        trk = self.trk[self.gi]
        prev_trk = np.concatenate([[np.nan], trk[:-1]])
        trk = np.where(np.isfinite(trk), trk, np.where(self.g_prev_ok, prev_trk, np.nan))
        brg = (np.degrees(np.arctan2(self.dx, self.dy)) + 360.0) % 360.0
        diff = np.abs((brg - trk + 180.0) % 360.0 - 180.0)
        known = np.isfinite(diff) & self.g_prev_ok
        with np.errstate(invalid="ignore"):
            rev = known & (self.step >= REV_MIN_STEP_M) & (diff > REV_MIN_DIFF) & ~(self.ggs >= PUSH_MAX_KT) & (self.vel < PUSH_MAX_MPS)
            big_fwd = known & (self.step >= FWD_BIG_STEP_M) & (diff < FWD_MAX_DIFF)
        cum_bf = np.cumsum(big_fwd)
        R = np.flatnonzero(rev)
        rows = []
        if len(R):
            gt = self.gt
            new_ep = np.ones(len(R), dtype=bool)
            new_ep[1:] = (self.g_vid[R[1:]] != self.g_vid[R[:-1]]) | (gt[R[1:] - 1] - gt[R[:-1]] > PUSH_GAP_S) \
                | (cum_bf[R[1:] - 1] - cum_bf[R[:-1]] > 0)
            ep = np.cumsum(new_ep) - 1
            starts = np.flatnonzero(new_ep)
            first = R[starts][ep]
            s0 = first - 1
            count = np.arange(len(R)) - starts[ep] + 1
            net = np.hypot(self.gx[R] - self.gx[s0], self.gy[R] - self.gy[s0])
            ok = (count >= PUSH_MIN_REV) & (net >= PUSH_MIN_NET_M)
            if ok.any():
                qual = pd.Series(np.flatnonzero(ok)).groupby(ep[ok]).first()
                for j in qual.to_numpy():
                    f = int(first[j])
                    v = max(float(self.ggs[f]) if np.isfinite(self.ggs[f]) else 0.0, 0.0) * 0.5144
                    est = gt[f] - (self.step[f] / v if v > 0.3 else self.dt[f] / 2.0)
                    rows.append((int(self.g_vid[f]), f - 1, f, int(R[j]), float(np.floor(max(est, gt[f - 1]))), float(gt[R[j]])))
        self.push = np.array(rows, dtype="float64").reshape(-1, 6)  # vid, s0 (pre-push), first rev point, qualifying point, t_start, t_known
        self.pushes_of_visit = (pd.Series(np.arange(len(self.push))).groupby(self.push[:, 0].astype("int64")).agg(list).to_dict()
                                if len(self.push) else {})


def _episode_rows(tr: _Track, airborne_first: dict) -> tuple[list, list]:
    """TAXI_IN and TAXI_OUT rows, and every taxi-out episode (confirmed or not) for the parent attributes.

    Per visit: a taxi-in from the touchdown; then episodes, each starting at the first moving ground point after a
    boundary (the visit's start, the taxi-in's end, a parking stop, an internal gap) and ending at the earliest of
    the next parking stop, the next internal gap (the timeout of the point before it: `_Track.timeout`), the
    lift-off or the visit's last point (+ timeout, if observable by the cutoff). The visit's last episode is the
    takeoff's parent when the lift-off followed the last ground point within that timeout (``takeoff_parent``),
    even when its confirmation came later and the episode had ended by the timeout.
    """
    rows, eps = [], []
    gt, gx, gy, gi = tr.gt, tr.gx, tr.gy, tr.gi
    for v in range(tr.nv):
        k_a, k_b = int(tr.g_first[v]), int(tr.g_last[v])
        if k_a < 0:
            continue
        ic = tr.icao[gi[k_a]]
        fa = airborne_first.get(ic, np.nan)
        if not (np.isfinite(fa) and fa <= tr.v_tmax[v]):
            continue  # not an aircraft by anything this visit can know (the event filter would drop it all)
        a_ = tr.apt[gi[k_a]]
        stops = tr.stops_of_visit.get(v, [])
        pushes = tr.pushes_of_visit.get(v, [])
        landing = bool(tr.v_landing[v])
        t_td = tr.t[tr.v_first[v]] if landing else np.nan
        k_from = k_a
        if landing:
            # the earliest known of: a qualifying stop, an internal gap > TIMEOUT_TAXI_IN_S, the visit's end
            cand = []
            for s in stops:
                if tr.s_cls[s] == 1 and np.isfinite(tr.tq60[s]):
                    cand.append((tr.tq60[s], tr.s_t0[s], int(tr.s_k0[s])))
                    break
                if tr.s_cls[s] != 2 and np.isfinite(tr.tq300[s]):
                    cand.append((tr.tq300[s], tr.s_t0[s], int(tr.s_k0[s])))
                    break
            k_g = int(tr.next_gap_in[k_a])
            if k_g < k_b:
                cand.append((gt[k_g] + TIMEOUT_TAXI_IN_S, gt[k_g], k_g + 1))
            if cand:
                ti_known, ti_end, k_from = min(cand, key=lambda c: (c[0], c[2]))
            else:
                ti_end, ti_known, _ = tr.visit_end(v, TIMEOUT_TAXI_IN_S)
                k_from = k_b + 1
            rows.append(dict(apt=a_, kind="TAXI_IN", icao=ic, cs=tr.cs[tr.v_first[v]], t_start=t_td, t_known_start=t_td,
                             t_end=ti_end, t_known_end=ti_known, dur=ti_end - t_td))
        j = 0
        while k_from <= k_b:
            k_s = int(tr.next_moving[k_from])
            if k_s > k_b:
                break
            while j < len(stops) and tr.s_k0[stops[j]] < k_s:
                j += 1
            nxt_park = next((jj for jj in range(j, len(stops)) if np.isfinite(tr.park_known[stops[jj]])), None)
            k_g = int(tr.next_gap[k_s])
            takeoff_parent = False
            if nxt_park is not None and tr.s_k0[stops[nxt_park]] <= min(k_g, k_b):
                sp = stops[nxt_park]
                t_e, t_ke, k_end, k_next, ends_visit = tr.s_t0[sp], tr.park_known[sp], int(tr.s_k0[sp]), int(tr.s_k0[sp]), False
            elif k_g < k_b:
                t_e, t_ke, k_end, k_next, ends_visit = gt[k_g], gt[k_g] + tr.timeout[k_g], k_g, k_g + 1, False
            else:
                t_e, t_ke, end_kind = tr.visit_end(v, tr.timeout[k_b])
                k_end, k_next, ends_visit = k_b, k_b + 1, end_kind == "liftoff"
                takeoff_parent = tr.lift_within(v, tr.timeout[k_b])
            t_s0 = gt[k_s]
            if not (landing and t_s0 - t_td < LANDED_RECENT_S):
                prev_stop = tr.stop_of[k_s - 1] if k_s > k_from else -1
                if prev_stop >= 0 and np.isfinite(tr.tq30[prev_stop]):
                    origin0, o_k0 = "stop", k_s - 1
                else:
                    origin0, o_k0 = "first_seen", k_from
                stand0 = float(tr.geometry[a_].nearest_stand(gx[o_k0:o_k0 + 1], gy[o_k0:o_k0 + 1], ORIGIN_MAX_M)[1][0])
                kc = int(tr.next_fast[k_s])
                t_conf = gt[kc] if kc <= k_end and (not np.isfinite(t_e) or gt[kc] <= t_e) else np.nan
                # a push whose first reverse step starts the motion: the origin from the moment it is recognised
                push_known, t_sp, stand_p = np.nan, t_s0, np.nan
                for pi in pushes:
                    if max(k_from, k_s - 2) <= int(tr.push[pi, 2]) <= min(k_s + PUSH_START_STEPS, k_end):  # inside this episode
                        push_known, t_sp = tr.push[pi, 5], float(min(t_s0, tr.push[pi, 4]))
                        o_kp = int(tr.push[pi, 1])
                        stand_p = float(tr.geometry[a_].nearest_stand(gx[o_kp:o_kp + 1], gy[o_kp:o_kp + 1], ORIGIN_MAX_M)[1][0])
                        break
                ep = dict(origin0=origin0, stand0=stand0, t_s0=t_s0, push_known=push_known, t_sp=t_sp, stand_p=stand_p)
                origin, o_stand, t_s = _origin_at(ep, t_conf if np.isfinite(t_conf) else np.inf)
                stop_s = 0.0
                for jj in range(j, len(stops)):
                    s = stops[jj]
                    if tr.s_k0[s] >= k_end and not (ends_visit and tr.s_k0[s] == k_end):
                        break
                    if np.isfinite(tr.tq30[s]) and np.isfinite(tr.s_tend[s]) and tr.s_tend[s] <= (t_e if np.isfinite(t_e) else np.inf):
                        stop_s += tr.s_tend[s] - tr.s_t0[s]
                confirmed = bool(np.isfinite(t_conf) and t_conf - t_s <= EPISODE_MAX_S)
                e_i = len(eps)
                eps.append(dict(ep, v=v, k0=k_s, k1=k_end, t_conf=t_conf if confirmed else np.nan, ends_visit=ends_visit,
                                takeoff_parent=takeoff_parent, stop_s=stop_s,
                                path=float(tr.cum_path[k_end] - tr.cum_path[k_s]) if k_end >= k_s else 0.0))
                if confirmed:
                    ended = bool(np.isfinite(t_ke))
                    rows.append(dict(apt=a_, kind="TAXI_OUT", icao=ic, cs=tr.cs[gi[kc]], t_start=t_s, t_known_start=t_conf,
                                     t_end=t_e, t_known_end=t_ke, origin_kind=origin, origin_stand=float(o_stand),
                                     dur=t_e - t_s if ended else np.nan, stop_s=stop_s if ended else np.nan,
                                     path_m=eps[-1]["path"] if ended else np.nan, _ep=e_i))
            k_from = k_next
    return rows, eps


def _origin_at(ep: dict, known: float) -> tuple[str, float, float]:
    """An episode's (origin kind, origin stand (NaN: none), start) as known at ``known``: the push only once it was
    recognised."""
    if np.isfinite(ep["push_known"]) and ep["push_known"] <= known:
        return "push", ep["stand_p"], ep["t_sp"]
    return ep["origin0"], ep["stand0"], ep["t_s0"]


def _origin_cols(ep: dict, known: float) -> dict:
    origin, stand, _ = _origin_at(ep, known)
    return {"origin_kind": origin, "origin_stand": float(stand)}


def _takeoff_landing_rows(tr: _Track, eps: list) -> list:
    rows = []
    last_ep = {e["v"]: i for i, e in enumerate(eps) if e["takeoff_parent"]}
    gt, ggs, gi = tr.gt, tr.ggs, tr.gi
    for v in np.flatnonzero(tr.v_liftoff):
        k_a, k_b = tr.g_first[v], tr.g_last[v]
        if k_a < 0:
            continue
        li = tr.v_last[v] + 1
        ci = tr.r_conf[tr.run[li]]
        seg_gs = ggs[k_a:k_b + 1]
        slow = np.flatnonzero(seg_gs < ROLL_SLOW_KT)
        if not len(slow) or not np.nanmax(np.append(tr.gs[li:ci + 1], -1.0)) >= TKO_MIN_KT:
            continue  # never slow before the lift-off, or never fast after it: not a departure from this ground visit
        k_slow = k_a + slow[-1]
        fast = np.flatnonzero(ggs[k_slow + 1:k_b + 1] >= TKO_MIN_KT)
        t_roll = gt[k_slow + 1 + fast[0]] if len(fast) else tr.t[li]
        lg = gi[k_b]
        use_ground = bool(np.isfinite(tr.gs[lg]) and tr.gs[lg] >= GROUND_KT)
        ai = lg if use_ground else li
        g = tr.geometry[tr.apt[li]]
        rwy = g.attribute(tr.x[ai:ai + 1], tr.y[ai:ai + 1], tr.course[ai:ai + 1], np.array([not use_ground]))[0]
        if rwy is None:
            continue  # not a runway of this airport
        row = dict(apt=tr.apt[li], kind="TAKEOFF", icao=tr.icao[li], cs=tr.cs[ci], t_start=t_roll, t_known_start=tr.t[ci],
                   t_end=tr.t[li], t_known_end=tr.t[ci], rwy_end=rwy, _seen=0)
        e_i = last_ep.get(v)
        if e_i is not None:
            e = eps[e_i]
            t_s = _origin_at(e, tr.t[ci])[2]
            if t_roll - t_s <= EPISODE_MAX_S:
                row.update(_origin_cols(e, tr.t[ci]), dur=t_roll - t_s, stop_s=e["stop_s"], path_m=e["path"], _ep=e_i)
                # the taxi was received for more than a line-up (the parent's row is checked once the events are known)
                row["_seen"] = int(t_roll - t_s >= TAXI_SEEN_MIN_S)
        rows.append(row)
    for v in np.flatnonzero(tr.v_landing):
        fi = tr.v_first[v]
        li = fi - 1
        g = tr.geometry[tr.apt[fi]]
        rwy = None
        if np.isfinite(tr.gs[fi]) and tr.gs[fi] >= GROUND_KT:  # touchdown / rollout received on the runway
            rwy = g.attribute(tr.x[fi:fi + 1], tr.y[fi:fi + 1], tr.course[fi:fi + 1], np.array([False]))[0]
        if rwy is None:  # else the last airborne point, on the final
            rwy = g.attribute(tr.x[li:li + 1], tr.y[li:li + 1], tr.course[li:li + 1], np.array([True]), landing=True)[0]
        if rwy is None:
            continue
        rows.append(dict(apt=tr.apt[fi], kind="LANDING", icao=tr.icao[fi], cs=tr.cs[fi], t_start=tr.t[fi],
                         t_known_start=tr.t[fi], t_end=tr.t[fi], t_known_end=tr.t[fi], rwy_end=rwy))
    return rows


def _stop_rows(tr: _Track, eps: list) -> list:
    """RWY_HOLD (any airframe not landed recently), REMOTE_STOP (inside a confirmed taxi-out), PARK (QA), PUSH."""
    rows = []
    eps_of_visit: dict[int, list[int]] = {}
    for i, e in enumerate(eps):
        eps_of_visit.setdefault(e["v"], []).append(i)
    gi = tr.gi
    for s in range(tr.ns):
        v = int(tr.s_vid[s])
        k0 = tr.s_k0[s]
        ic, a_, t0 = tr.icao[gi[k0]], tr.s_apt[s], tr.s_t0[s]
        recent = bool(tr.v_landing[v]) and (t0 - tr.t[tr.v_first[v]] < LANDED_RECENT_S)
        e_i = next((i for i in eps_of_visit.get(v, ()) if eps[i]["k0"] <= k0 <= eps[i]["k1"]), -1)
        e = eps[e_i] if e_i >= 0 else None
        tend, tkend = tr.s_tend[s], tr.s_tkend[s]
        moved = tr.next_moving[tr.g_first[v]] < k0  # a queue forms behind motion: a parked airframe is no runway hold
        if tr.s_cls[s] == 2 and np.isfinite(tr.tq30[s]) and not recent and moved:
            row = dict(apt=a_, kind="RWY_HOLD", icao=ic, cs=tr.cs[max(int(tr.qi30[s]), 0)], t_start=t0,
                       t_known_start=tr.tq30[s], t_end=tend, t_known_end=tkend, rwy_end=tr.s_rwy_end[s], dur=tend - t0)
            if e is not None and tr.tq30[s] - _origin_at(e, tr.tq30[s])[2] <= EPISODE_MAX_S:
                row.update(_origin_cols(e, tr.tq30[s]), _ep=e_i)
            rows.append(row)
        tq = tr.tq300[s]
        if (tr.s_cls[s] == 0 and np.isfinite(tq) and e is not None and np.isfinite(e["t_conf"]) and e["t_conf"] <= tq
                and tq - _origin_at(e, tq)[2] <= EPISODE_MAX_S):
            if np.isfinite(tr.tq1800[s]):  # a remote stop of PARK_LONG_S is parking: it counts until then
                r_end, r_kend = t0 + PARK_LONG_S, tr.tq1800[s]
            else:
                r_end, r_kend = tend, tkend
            rows.append(dict(apt=a_, kind="REMOTE_STOP", icao=ic, cs=tr.cs[max(int(tr.qi300[s]), 0)], t_start=t0,
                             t_known_start=tq, t_end=r_end, t_known_end=r_kend, dur=r_end - t0, _ep=e_i,
                             **_origin_cols(e, tq)))
        if np.isfinite(tr.park_known[s]):
            rows.append(dict(apt=a_, kind="PARK", icao=ic, cs=tr.cs[gi[k0]], t_start=t0, t_known_start=tr.park_known[s],
                             t_end=tend, t_known_end=tkend, dur=tend - t0, origin_stand=float(tr.s_stand[s])))
    for v_, s0, _rf, rq, t_start, t_known in tr.push:
        v_, s0 = int(v_), int(s0)
        if tr.v_landing[v_] and t_known - tr.t[tr.v_first[v_]] < LANDED_RECENT_S:
            continue
        a_ = tr.apt[gi[s0]]
        o_stand = float(tr.geometry[a_].nearest_stand(tr.gx[s0:s0 + 1], tr.gy[s0:s0 + 1], ORIGIN_MAX_M)[1][0])
        rows.append(dict(apt=a_, kind="PUSH", icao=tr.icao[gi[s0]], cs=tr.cs[gi[int(rq)]], t_start=t_start,
                         t_known_start=t_known, origin_kind="push", origin_stand=o_stand))
    return rows


def _events_from_points(p: pd.DataFrame, geometry: dict[str, AirportGeometry], cutoff: float,
                        airborne_first: pd.Series) -> pd.DataFrame:
    if not len(p):
        return _empty_events()
    tr = _Track(p, geometry, cutoff)
    if not tr.nv or not tr.ng:
        return _empty_events()
    af = airborne_first.reindex(pd.unique(tr.icao)).to_dict()
    rows, eps = _episode_rows(tr, af)
    rows += _takeoff_landing_rows(tr, eps)
    rows += _stop_rows(tr, eps)
    if not rows:
        return _empty_events()
    ev = pd.DataFrame(rows)
    for c in (*EVENT_COLUMNS, "_ep", "_seen"):
        if c not in ev.columns and c not in ("eid", "parent"):
            ev[c] = np.nan
    # the vehicle filter (critique F8): an icao counts from its first airborne second on, over the snapshot's history
    fa = ev["icao"].map(af).to_numpy(dtype="float64")
    ev = ev.loc[np.isfinite(fa) & (fa <= ev["t_known_start"].to_numpy(dtype="float64"))].copy()
    if ev.empty:
        return _empty_events()
    # ends: only those knowable by the cutoff, and only within END_MAX_S of the start
    tk = ev["t_known_end"].to_numpy(dtype="float64")
    te = ev["t_end"].to_numpy(dtype="float64")
    ts = ev["t_start"].to_numpy(dtype="float64")
    kind = ev["kind"].to_numpy(dtype=object)
    with np.errstate(invalid="ignore"):
        bad = ~np.isfinite(tk) | (tk > cutoff) | ~np.isfinite(te) | (te - ts > END_MAX_S)
    ev.loc[bad, ["t_end", "t_known_end"]] = np.nan
    ev.loc[bad & (kind != "TAKEOFF"), ["dur", "stop_s", "path_m"]] = np.nan  # a takeoff's are known at the takeoff
    ev["eid"] = _eid(ev["icao"].to_numpy(), kind, ts)
    # a child's parent: its taxi-out, when that was known (confirmed) by the child's own knowable time
    taxi = ev.loc[ev["kind"].eq("TAXI_OUT") & ev["_ep"].notna(), ["_ep", "eid", "t_known_start"]]
    ep_eid = {int(e): (i, k) for e, i, k in zip(taxi["_ep"], taxi["eid"], taxi["t_known_start"])}
    parent = []
    for e, k, known in zip(ev["_ep"].to_numpy(dtype="float64"), kind, ev["t_known_start"].to_numpy(dtype="float64")):
        p_ = ep_eid.get(int(e)) if np.isfinite(e) and k != "TAXI_OUT" else None
        parent.append(p_[0] if p_ is not None and p_[1] <= known else -1)
    ev["parent"] = parent
    # taxi_seen: a takeoff whose taxi-out (a TAXI_OUT row, known by the takeoff) was received for more than a line-up
    ev["taxi_seen"] = (ev["_seen"].fillna(0).to_numpy() > 0) & (np.asarray(parent) >= 0)
    ev["taxi_seen"] = ev["taxi_seen"].astype("int8")
    for c in ("t_start", "t_known_start", "t_end", "t_known_end", "dur", "stop_s", "path_m", "origin_stand"):
        ev[c] = ev[c].astype("float64")
    for c in ("apt", "kind", "icao", "cs", "origin_kind", "rwy_end"):
        ev[c] = ev[c].astype(object).where(ev[c].notna(), None)
    ev["parent"] = ev["parent"].astype("int64")
    ev = ev[list(EVENT_COLUMNS)]
    return ev.sort_values(["apt", "kind", "t_known_start", "icao", "t_start"], kind="stable").reset_index(drop=True)


# ============================================================================== day files

def point_file(day: str) -> Path | None:
    """The prod-only points of one UTC day: points_prod/<day> where it exists (the merged Jan/Jul days), else points/."""
    prod = paths.ADSB_POINTS_PROD / f"{day}.parquet"
    if prod.exists():
        return prod
    p = paths.ADSB_POINTS / f"{day}.parquet"
    return p if p.exists() else None


def available_days() -> list[str]:
    days = {f.stem for d in (paths.ADSB_POINTS, paths.ADSB_POINTS_PROD) if d.exists() for f in d.glob("????-??-??.parquet")}
    return sorted(days)


def day_start(day: str) -> float:
    return pd.Timestamp(day, tz="UTC").timestamp()


def read_day(day: str, t_lo: float | None = None, t_hi: float | None = None, columns=None) -> pd.DataFrame:
    """One day file clipped to its own UTC day (and to [t_lo, t_hi)), deduplicated on (icao, t) (critique F7)."""
    f = point_file(day)
    if f is None:
        return pd.DataFrame(columns=list(columns or POINT_COLUMNS))
    lo = day_start(day)
    hi = lo + DAY_S
    if t_lo is not None:
        lo = max(lo, t_lo)
    if t_hi is not None:
        hi = min(hi, t_hi)
    cols = list(columns) if columns else None
    if cols is not None:  # deduplication needs the airframe and the time, whatever the caller reads
        cols += [c for c in ("icao", "t") if c not in cols]
    frame = pd.read_parquet(f, columns=cols, filters=[("t", ">=", lo), ("t", "<", hi)])
    if "apt" in frame.columns:
        frame["apt"] = frame["apt"].astype(str)
    frame = frame.drop_duplicates(["icao", "t"]).reset_index(drop=True)
    return frame[list(columns)] if columns else frame


def file_identity(day: str) -> list | None:
    f = point_file(day)
    if f is None:
        return None
    st = f.stat()
    return [f.parent.name, f.name, st.st_size, st.st_mtime_ns]


def window(day: str) -> tuple[pd.DataFrame, float]:
    """The points a build of ``day`` reads, and the cutoff (the last second covered)."""
    d0 = day_start(day)
    prev = (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    nxt = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    parts = []
    if point_file(prev) is not None:
        parts.append(read_day(prev, t_lo=d0 - LOOKBACK_S))
    parts.append(read_day(day))
    if point_file(nxt) is not None:
        parts.append(read_day(nxt, t_hi=d0 + DAY_S + LOOKAHEAD_S))
        cutoff = d0 + DAY_S + LOOKAHEAD_S - 1
    else:
        cutoff = d0 + DAY_S - 1
    parts = [p for p in parts if len(p)]
    pts = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(POINT_COLUMNS))
    return pts, float(cutoff)


def build_events_for_day(day: str, geometry: dict[str, AirportGeometry], airborne_first: pd.Series) -> pd.DataFrame:
    """The events of ``day``: built over its window, kept where known within the day."""
    pts, cutoff = window(day)
    ev = episodes(pts, geometry, cutoff, airborne_first)
    d0 = day_start(day)
    k = ev["t_known_start"].to_numpy(dtype="float64")
    return ev.loc[(k >= d0) & (k < d0 + DAY_S)].reset_index(drop=True)


def hourly_qa(day: str, airborne_first: pd.Series) -> pd.DataFrame:
    """Per (airport, hour) of the day: points, distinct ground aircraft, ground vehicles (not yet airborne), airborne."""
    p = read_day(day, columns=["icao", "t", "on_ground", "gs_kt", "apt"])
    if p.empty:
        return pd.DataFrame(columns=["apt", "hour", "n_points", "ground_ac", "ground_veh", "air_ac"])
    p["icao"] = p["icao"].astype(str).str.lower()
    p["hour"] = (np.floor(p["t"].to_numpy(dtype="float64") / 3600) * 3600).astype("int64")
    gs = p["gs_kt"].to_numpy(dtype="float64")
    ground = p["on_ground"].to_numpy(dtype=bool) | (gs < GROUND_KT)
    fa = p["icao"].map(airborne_first).to_numpy(dtype="float64")
    aircraft = np.isfinite(fa) & (fa <= p["t"].to_numpy(dtype="float64"))
    p = p.assign(g_ac=np.where(ground & aircraft, p["icao"], None), g_veh=np.where(ground & ~aircraft, p["icao"], None),
                 a_ac=np.where(~ground, p["icao"], None))
    out = p.groupby(["apt", "hour"]).agg(n_points=("t", "size"), ground_ac=("g_ac", "nunique"),
                                          ground_veh=("g_veh", "nunique"), air_ac=("a_ac", "nunique")).reset_index()
    return out


# ============================================================================== snapshot build

def parse_days(tokens: list[str] | None) -> list[str] | None:
    """``YYYY-MM-DD`` tokens and ``A..B`` ranges (inclusive) -> sorted unique days; None for no tokens."""
    if not tokens:
        return None
    out = set()
    for tok in tokens:
        if ".." in tok:
            a, b = tok.split("..", 1)
            out.update(d.strftime("%Y-%m-%d") for d in pd.date_range(a, b, freq="D"))
        else:
            out.add(pd.Timestamp(tok).strftime("%Y-%m-%d"))
    return sorted(out)


def _manifest_path(snapshot: str) -> Path:
    return paths.adsb_surface_dir(snapshot) / "manifest.json"


def read_manifest(snapshot: str) -> dict:
    f = _manifest_path(snapshot)
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"snapshot": snapshot, "days": {}, "day_set": []}


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def airborne_history(snapshot: str, days: list[str], log=print) -> pd.Series:
    """icao -> first airborne second over the snapshot's days; per-day partials cached under airborne/."""
    d = paths.adsb_surface_dir(snapshot) / "airborne"
    d.mkdir(parents=True, exist_ok=True)
    parts = []
    for day in days:
        ident = file_identity(day)
        if ident is None:
            continue
        f = d / f"{day}.parquet"
        meta = d / f"{day}.json"
        if f.exists() and meta.exists() and json.loads(meta.read_text(encoding="utf-8")) == ident:
            parts.append(pd.read_parquet(f)["first_air"])
            continue
        s = airborne_first_times(read_day(day, columns=["icao", "t", "on_ground", "gs_kt"]))
        s.to_frame().to_parquet(f)
        meta.write_text(json.dumps(ident), encoding="utf-8")
        parts.append(s)
    if not parts:
        return pd.Series(dtype="float64", name="first_air")
    return pd.concat(parts).groupby(level=0).min().rename("first_air")


def _history_digest(airborne_first: pd.Series, horizon: float) -> str:
    s = airborne_first[airborne_first <= horizon].sort_index()
    h = hashlib.sha1()
    h.update(np.asarray(s.index, dtype=str).astype("U").tobytes())
    h.update(s.to_numpy(dtype="float64").tobytes())
    return h.hexdigest()[:12]


def _day_key(day: str, geo_digest: str, airborne_first: pd.Series, history_days: int) -> dict:
    prev = (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    nxt = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return {
        "builder": BUILDER_VERSION,
        "geometry": geo_digest,
        "inputs": [file_identity(prev), file_identity(day), file_identity(nxt)],
        "history": _history_digest(airborne_first, day_start(day) + DAY_S + LOOKAHEAD_S),
        "warmup": history_days < WARMUP_DAYS,
    }


_WORKER: dict = {}


def _build_one(args) -> tuple[str, dict, str]:
    day, snapshot, key = args
    t0 = time.time()
    out = paths.adsb_surface_dir(snapshot)
    if "geometry" not in _WORKER:
        _WORKER["geometry"] = load_geometry()
    if _WORKER.get("af_snapshot") != snapshot:
        _WORKER["af"] = pd.read_parquet(out / "airborne_first.parquet")["first_air"]
        _WORKER["af_snapshot"] = snapshot
    af = _WORKER["af"]
    if key["warmup"]:
        ev = _empty_events()
    else:
        ev = build_events_for_day(day, _WORKER["geometry"], af)
    tmp = out / "days" / f"{day}.tmp"
    ev.to_parquet(tmp, index=False)
    tmp.replace(out / "days" / f"{day}.parquet")
    qa = hourly_qa(day, af)
    qa.to_parquet(out / "qa" / f"{day}.parquet", index=False)
    info = {"key": key, "n_events": int(len(ev)), "seconds": round(time.time() - t0, 1),
            "counts": ev["kind"].value_counts().to_dict() if len(ev) else {}}
    return day, info, f"{day}: {len(ev):,} events{' (warm-up: none written)' if key['warmup'] else ''}, {time.time() - t0:.1f}s"


def build(days: list[str] | None = None, snapshot: str | None = None, workers: int = 1, force: bool = False,
          freeze: bool = False, log=print) -> Path:
    """Build (resumably) the day files of a snapshot and consolidate them; returns the consolidated table's path.

    The snapshot's day set is sticky: the first build records it (``days``, default every day with points), a later
    build adds the ``days`` it is given. A day is rebuilt when its key (builder version, geometry digest, the input
    files of D-1, D, D+1, the vehicle filter's history up to the day's horizon, the warm-up flag) changes. A frozen
    snapshot is never modified (critique F9). ``freeze``: `freeze_snapshot` after the build (it needs the support
    table of the table just built: build, ``--support``, then ``--freeze``).
    """
    snapshot = snapshot or paths.ADSB_SURFACE_SNAPSHOT
    out = paths.adsb_surface_dir(snapshot)
    man = read_manifest(snapshot)
    if man.get("frozen"):
        raise RuntimeError(f"snapshot {snapshot!r} is frozen ({out}); build a new snapshot instead")
    day_set = sorted(set(man.get("day_set", [])) | set(days or ([] if man.get("day_set") else available_days())))
    present = [d for d in day_set if point_file(d) is not None]
    missing = [d for d in day_set if point_file(d) is None]
    for sub in ("days", "qa"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    log(f"adsb-surface {snapshot}: {len(day_set)} days in the set, {len(present)} with points, {len(missing)} missing")
    af = airborne_history(snapshot, present, log)
    af.to_frame().to_parquet(out / "airborne_first.parquet")
    geo = geometry_digest()
    todo = []
    for i, day in enumerate(present):
        key = _day_key(day, geo, af, history_days=i)
        entry = man["days"].get(day)
        if not force and entry and entry.get("key") == key and (out / "days" / f"{day}.parquet").exists():
            continue
        todo.append((day, snapshot, key))
    log(f"  {len(todo)} days to build with {workers} worker(s)")
    if todo:
        if workers > 1 and len(todo) > 1:
            with mp.Pool(workers) as pool:
                for day, info, msg in pool.imap_unordered(_build_one, todo):
                    man["days"][day] = info
                    log(f"  {msg}")
        else:
            for arg in todo:
                day, info, msg = _build_one(arg)
                man["days"][day] = info
                log(f"  {msg}")
    for day in missing:
        man["days"].pop(day, None)
    man.update(snapshot=snapshot, builder=BUILDER_VERSION, geometry=geo, day_set=day_set, missing=missing)
    path = consolidate(snapshot, man, log)
    _write_json(_manifest_path(snapshot), man)
    if (man.get("support") or {}).get("events_sha1") not in (None, man["consolidated"]["sha1"]):
        log(f"  the support table was computed from another event table: rerun `prc adsb-surface --snapshot {snapshot} --support`")
    if freeze:
        freeze_snapshot(snapshot, log)
    return path


def freeze_snapshot(snapshot: str | None = None, log=print) -> None:
    """Mark a snapshot frozen (never modified again). Refused unless its support table was computed from the
    consolidated table as it is now: the blocks read both, and a stale support would gate with another table's ranges."""
    snapshot = snapshot or paths.ADSB_SURFACE_SNAPSHOT
    man = read_manifest(snapshot)
    cons, sup = man.get("consolidated") or {}, man.get("support") or {}
    if not cons:
        raise RuntimeError(f"snapshot {snapshot!r} has no consolidated table: build it first")
    if sup.get("events_sha1") != cons.get("sha1") or not (paths.adsb_surface_dir(snapshot) / paths.ADSB_SURFACE_SUPPORT_NAME).exists():
        raise RuntimeError(f"snapshot {snapshot!r}: the support table is missing or was computed from another event table; "
                           f"run `prc adsb-surface --snapshot {snapshot} --support` first, then --freeze")
    man["frozen"] = True
    _write_json(_manifest_path(snapshot), man)
    log(f"adsb-surface {snapshot}: frozen (events sha1 {cons['sha1']}, support sha1 {sup.get('sha1')})")


SUPPORT_MVT_COLUMNS = ["apt", "is_dep", "split", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "IOBT_flt", "LOBT_flt",
                       "STAND_mvt", "RUNWAY_mvt", "CALLSIGN_flt"]


def build_support(snapshot: str | None = None, log=print) -> Path:
    """The snapshot's airport x family support table (review of 9 Oct, critique F6/F7), from every 2025 training
    departure whose H1 anchor falls on a built (non-warm-up) day: per airport and column family (taxi, hold, stand),
    the share of departures whose ADS-B-only family gate holds and the 1st-99th percentile of the family's scale ratio
    where it does (`prc.features._adsblib.support_table`). The blocks read it as an external input and NaN a family at
    an airport without support, or on a row whose ratio leaves the training range: a reception regime of 2026 that
    2025 never showed at that airport is not extrapolated. Public data only (no reference movement enters the gates).
    Recomputed by hand after any rebuild (`freeze_snapshot` checks it matches the consolidated table)."""
    from prc import data
    from prc.features import _adsblib, _oplib

    snapshot = snapshot or paths.ADSB_SURFACE_SNAPSHOT
    out = paths.adsb_surface_dir(snapshot)
    man = read_manifest(snapshot)
    if man.get("frozen"):
        raise RuntimeError(f"snapshot {snapshot!r} is frozen ({out}); its support is fixed")
    events_file = out / paths.ADSB_SURFACE_EVENTS_NAME
    if not events_file.exists() or not man.get("consolidated"):
        raise FileNotFoundError(f"{events_file}: build the snapshot first (prc adsb-surface --snapshot {snapshot})")
    built = {d for d, e in man.get("days", {}).items() if not e.get("key", {}).get("warmup")}
    mvt = data.load_movements(columns=SUPPORT_MVT_COLUMNS)
    dep = _oplib.departures(mvt)
    dep = dep.loc[dep["split"].astype(str).eq("train").to_numpy()]
    a = pd.Series(_oplib.anchor_h1(dep), index=dep.index)
    day = pd.Series(pd.to_datetime(a.to_numpy(), unit="s", utc=True).strftime("%Y-%m-%d"), index=dep.index)
    use = a.notna().to_numpy() & day.isin(built).to_numpy()
    sup_days = set(day[use])
    parts, pairs = [], []
    for month in sorted({d[:7] for d in sup_days}):
        idx = dep.index[use & day.str[:7].eq(month).to_numpy()]
        t_lo = float(a[idx].min()) - _adsblib.EVENTS_MARGIN_S - _adsblib.ADSB_LAG_S
        m0 = day_start(f"{month}-01")
        m1 = (pd.Timestamp(f"{month}-01", tz="UTC") + pd.offsets.MonthBegin(1)).timestamp()
        ev = pd.read_parquet(events_file, columns=list(_adsblib.READ_COLUMNS),
                             filters=[("t_known_start", ">=", t_lo),
                                      ("t_known_start", "<=", max(float(a[idx].max()), m1 + HOLD_TAKEOFF_S + 600))])
        q_ev = ev.loc[ev["t_known_start"].to_numpy(dtype="float64") <= float(a[idx].max())]
        _, internals = _adsblib.surface_at(mvt, idx, a[idx].to_numpy(), {"support": False}, events=q_ev, return_internals=True)
        parts.append(internals)
        # holds that began in this month on a support day, paired with the airframe's next takeoff
        hp = hold_takeoff_pairs(ev)
        hs = ev.loc[hp.index, "t_start"].to_numpy(dtype="float64")
        hd = pd.to_datetime(hs, unit="s", utc=True).strftime("%Y-%m-%d")
        pairs.append(hp.loc[(hs >= m0) & (hs < m1) & np.isin(hd, list(sup_days))])
        log(f"  support {month}: {len(idx):,} departures, taxi gate {internals['taxi_rx'].mean():.2f}, "
            f"stand gate {internals['stand_rx'].mean():.2f}, {len(pairs[-1]):,} holds paired with a takeoff")
    if not parts:
        raise RuntimeError(f"snapshot {snapshot!r}: no training departure on a built day")
    table = _adsblib.support_table(pd.concat(parts), pd.concat(pairs, ignore_index=True))
    dest = out / paths.ADSB_SURFACE_SUPPORT_NAME
    tmp = dest.with_suffix(".tmp")
    table.to_parquet(tmp, index=False)
    tmp.replace(dest)
    man["support"] = {"file": dest.name, "rows": int(len(table)), "sha1": hashlib.sha1(dest.read_bytes()).hexdigest()[:12],
                      "events_sha1": man["consolidated"]["sha1"], "departures": int(use.sum())}
    _write_json(_manifest_path(snapshot), man)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        log(table.round(3).to_string(index=False))
    return dest


def consolidate(snapshot: str, man: dict, log=print) -> Path:
    """One table of the block kinds over the snapshot's days, sorted by (apt, kind, t_known_start)."""
    out = paths.adsb_surface_dir(snapshot)
    frames = []
    for day in man.get("day_set", []):
        f = out / "days" / f"{day}.parquet"
        if day in man["days"] and f.exists():
            frames.append(pd.read_parquet(f))
    ev = pd.concat(frames, ignore_index=True) if frames else _empty_events()
    ev = ev.loc[ev["kind"].isin(BLOCK_KINDS)]
    ev = ev.sort_values(["apt", "kind", "t_known_start", "icao", "t_start"], kind="stable").reset_index(drop=True)
    for c in ("apt", "kind", "origin_kind", "rwy_end"):
        ev[c] = ev[c].astype("category")
    dest = out / paths.ADSB_SURFACE_EVENTS_NAME
    tmp = dest.with_suffix(".tmp")
    ev.to_parquet(tmp, index=False)
    tmp.replace(dest)
    sha = hashlib.sha1(dest.read_bytes()).hexdigest()[:12]
    man["consolidated"] = {"file": dest.name, "rows": int(len(ev)), "sha1": sha, "days": len(frames)}
    log(f"  consolidated {len(ev):,} events of {len(frames)} days -> {dest} (sha1 {sha})")
    return dest


# ============================================================================== coverage report

def _per_takeoff_split(ev: pd.DataFrame) -> pd.Series:
    """Per TAKEOFF row (index of ``ev``): the airframe's TAXI_OUT episodes starting in the hour before the roll (2+ means
    one taxi split by reception gaps)."""
    tko = ev.loc[ev["kind"].eq("TAKEOFF")]
    taxi = ev.loc[ev["kind"].eq("TAXI_OUT")]
    cats = pd.Index(pd.unique(pd.concat([tko["icao"], taxi["icao"]]).astype(str)))
    big = 4_000_000_000
    tk = cats.get_indexer(taxi["icao"].astype(str)).astype("int64") * big + taxi["t_start"].to_numpy(dtype="float64").astype("int64")
    tk = np.sort(tk)
    q = cats.get_indexer(tko["icao"].astype(str)).astype("int64") * big + tko["t_start"].to_numpy(dtype="float64").astype("int64")
    return pd.Series(np.searchsorted(tk, q, "right") - np.searchsorted(tk, q - 3600, "left"), index=tko.index)


HOLD_TAKEOFF_S = 1800
"""A hold is paired with the same airframe's next takeoff when that rolls within this after the hold began."""


def hold_takeoff_pairs(ev: pd.DataFrame) -> pd.DataFrame:
    """Every RWY_HOLD row (index of ``ev``) followed by a TAKEOFF of the same airframe at the same airport within
    ``HOLD_TAKEOFF_S``: ``apt``, ``label`` (the hold's runway end) and ``target`` (the takeoff's). Hindsight by
    construction (the takeoff follows the hold): for the support table's static relabelling and the check only."""
    cols = ["apt", "icao", "t_start", "rwy_end"]
    hold = ev.loc[ev["kind"].astype(str).eq("RWY_HOLD"), cols]
    tko = ev.loc[ev["kind"].astype(str).eq("TAKEOFF"), cols]
    hold = hold.loc[hold["rwy_end"].notna()]
    if hold.empty or tko.empty:
        return pd.DataFrame(columns=["apt", "label", "target"])
    key = lambda d: d["apt"].astype(str) + "|" + d["icao"].astype(str)  # noqa: E731
    hold = hold.assign(k=key(hold), t_start=hold["t_start"].astype("float64")).rename_axis("_row").reset_index()
    tko = tko.assign(k=key(tko), t_tko=tko["t_start"].astype("float64"), target=tko["rwy_end"])[["k", "t_tko", "target"]]
    m = pd.merge_asof(hold.sort_values("t_start"), tko.sort_values("t_tko"), left_on="t_start", right_on="t_tko", by="k",
                      direction="forward", tolerance=float(HOLD_TAKEOFF_S))
    m = m.loc[m["target"].notna()]
    return pd.DataFrame({"apt": m["apt"].astype(str).to_numpy(), "label": m["rwy_end"].astype(str).to_numpy(),
                         "target": m["target"].astype(str).to_numpy()}, index=pd.Index(m["_row"].to_numpy()))


def _hold_runway_agreement(ev: pd.DataFrame) -> pd.Series:
    """Per RWY_HOLD row with a takeoff of the same airframe within 30 min after it: whether the hold's ``rwy_end`` is
    that takeoff's runway end (the own-runway hold column relies on it)."""
    pairs = hold_takeoff_pairs(ev)
    return pd.Series(pairs["label"].eq(pairs["target"]).to_numpy(dtype=bool), index=pairs.index)


def check(snapshot: str | None = None) -> str:
    """Per airport-month, over the built days: ADS-B takeoffs and landings per recorded movement, the taxi-seen,
    push-origin and stand-origin shares of the takeoffs, pushes per recorded departure, the share of pushes and
    taxi-outs with an origin stand, the share of takeoffs whose taxi was split into 2+ episodes, the agreement of runway
    holds with the airframe's takeoff runway (flagged below 95 %), mean distinct ground aircraft per daytime hour, the
    vehicle share of ground transmitters; then the missing and warm-up days."""
    snapshot = snapshot or paths.ADSB_SURFACE_SNAPSHOT
    out = paths.adsb_surface_dir(snapshot)
    man = read_manifest(snapshot)
    lines = [f"adsb-surface --check: snapshot {snapshot} ({out})"]
    if not man.get("days"):
        lines.append("  nothing built yet")
        return "\n".join(lines)
    lines.append(f"  days in the set {len(man.get('day_set', []))}, built {len(man['days'])}, missing {len(man.get('missing', []))}"
                 f"{' (frozen)' if man.get('frozen') else ''}; consolidated: {man.get('consolidated')}")
    if man.get("missing"):
        lines.append("  missing days: " + ", ".join(man["missing"][:60]) + (" ..." if len(man["missing"]) > 60 else ""))
    warm = sorted(d for d, e in man["days"].items() if e.get("key", {}).get("warmup"))
    if warm:
        lines.append("  warm-up days (no events written): " + ", ".join(warm))
    built = sorted(d for d in man["days"] if d not in warm and (out / "days" / f"{d}.parquet").exists())
    if not built:
        return "\n".join(lines)
    ev = pd.concat([pd.read_parquet(out / "days" / f"{d}.parquet") for d in built], ignore_index=True)
    qa_files = [out / "qa" / f"{d}.parquet" for d in built if (out / "qa" / f"{d}.parquet").exists()]
    qa = pd.concat([pd.read_parquet(f) for f in qa_files], ignore_index=True) if qa_files else pd.DataFrame()
    day_of = lambda s: pd.to_datetime(s, unit="s", utc=True).dt.strftime("%Y-%m-%d")  # noqa: E731
    ev = ev.assign(day=day_of(ev["t_known_start"]))
    ev["month"] = ev["day"].str[:7]
    tab = ev.pivot_table(index=["apt", "month"], columns="kind", values="t_known_start", aggfunc="size", fill_value=0)
    tab["days"] = ev.groupby(["apt", "month"])["day"].nunique()
    tko = ev.loc[ev["kind"].eq("TAKEOFF")]
    tab["taxi_seen_share"] = tko.groupby(["apt", "month"])["taxi_seen"].mean()
    tab["push_share"] = tko.assign(p=tko["origin_kind"].eq("push")).groupby(["apt", "month"])["p"].mean()
    tab["stand_share"] = tko.assign(p=tko["origin_kind"].isin(["push", "stop"])).groupby(["apt", "month"])["p"].mean()
    for kind, col in (("PUSH", "push_with_stand"), ("TAXI_OUT", "taxi_with_stand")):
        k = ev.loc[ev["kind"].eq(kind)]
        tab[col] = k.assign(s=k["origin_stand"].notna()).groupby(["apt", "month"])["s"].mean()
    split = _per_takeoff_split(ev)
    tab["split_share"] = tko.assign(s=split.reindex(tko.index).to_numpy() >= 2).groupby(["apt", "month"])["s"].mean()
    agree = _hold_runway_agreement(ev)
    if len(agree):
        hold = ev.loc[agree.index]
        tab["hold_rwy_agree"] = hold.assign(a=agree.to_numpy()).groupby(["apt", "month"])["a"].mean()
    if len(qa):
        qa = qa.assign(month=day_of(qa["hour"]).str[:7], hod=(qa["hour"] % DAY_S) // 3600)
        tab["gnd_ac_per_hour"] = qa.loc[qa["hod"].between(6, 20)].groupby(["apt", "month"])["ground_ac"].mean()
        sums = qa.groupby(["apt", "month"])[["ground_veh", "ground_ac"]].sum()
        tab["veh_share"] = sums["ground_veh"] / (sums["ground_veh"] + sums["ground_ac"]).clip(lower=1)
    if paths.MOVEMENTS_FILE.exists():
        mv = pd.read_parquet(paths.MOVEMENTS_FILE, columns=["apt", "is_dep", "MVT_TIME_UTC_mvt"])
        mv = mv.assign(apt=mv["apt"].astype(str), day=mv["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d"))
        mv = mv.loc[mv["day"].isin(built)]
        mv["month"] = mv["day"].str[:7]
        idx = tab.index
        dep = mv.loc[mv["is_dep"]].groupby(["apt", "month"]).size().reindex(idx).to_numpy(dtype="float64")
        arr = mv.loc[~mv["is_dep"]].groupby(["apt", "month"]).size().reindex(idx).to_numpy(dtype="float64")
        for col, num, den in (("tko_per_dep", "TAKEOFF", dep), ("ldg_per_arr", "LANDING", arr), ("push_per_dep", "PUSH", dep)):
            tab[col] = (tab[num] if num in tab.columns else 0) / den
    cols = [c for c in ("days", "TAKEOFF", "LANDING", "TAXI_OUT", "PUSH", "RWY_HOLD", "REMOTE_STOP", "tko_per_dep",
                        "ldg_per_arr", "taxi_seen_share", "push_share", "stand_share", "push_per_dep", "push_with_stand",
                        "taxi_with_stand", "split_share", "hold_rwy_agree", "gnd_ac_per_hour", "veh_share") if c in tab.columns]
    with pd.option_context("display.width", 300, "display.max_rows", 1000, "display.max_columns", 50):
        lines.append(tab[cols].round(3).to_string())
    if "hold_rwy_agree" in tab.columns:
        weak = tab.loc[tab["hold_rwy_agree"] < 0.95, "hold_rwy_agree"].dropna()
        if len(weak):
            lines.append("  runway holds attributed to another runway end than the airframe's takeoff in > 5 % of cases "
                         "(op_*_adsb_rwy_own_hold_now is unreliable there): "
                         + ", ".join(f"{a} {m} {v:.2f}" for (a, m), v in weak.items()))
    return "\n".join(lines)
