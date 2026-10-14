"""Idea family: the departing aircraft's own inbound arrival and turnaround, as seen in ADS-B (24 h look-back).

The existing ADS-B features look back 4 h from the ADS-B takeoff and never use the arrival records. This module looks
back up to 24 h over the same aircraft (the linked icao, the previous day's file included) and reads its inbound:

* the inbound touchdown: the first ground point after the last airborne point (not on_ground, >= 60 kt) of the
  aircraft at this airport before its takeoff (airborne points in the last 180 s before the ADS-B takeoff ignored);
  no airborne point in the 24 h look-back = no inbound found (parked > 24 h, or the landing was not received); when
  the first ground point comes > 600 s after the last airborne point the touchdown is extrapolated from that point's
  altitude (``it_td_seen`` = 0, end kind ``unseen``);
* the end of the inbound track: the first "park" run after touchdown (points within 20 m of the run's first point for
  >= 180 s, a gap inside the run allowed) -> kind ``park`` (ADS-B saw the aircraft stop) or, if a gap > 600 s comes
  first, the last point before that gap -> kind ``lost`` (the track ended while still taxiing in: where reception of
  this aircraft ends on its way to the stand); kind ``cont`` = neither;
* the arrival record: the inbound touchdown linked to the airport's arrival movement (same airport, normalised
  callsign of the inbound = CALLSIGN_flt / FLIGHT_mvt, |touchdown - MVT_TIME| <= 300 s; else the single arrival
  within 45 s by time). Arrivals keep their actual in-block time in 2025 and 2026 (no departure target is read):
  the recorded in-block, the recorded arrival stand, and the time from the inbound's last ADS-B moving point to the
  recorded in-block = the inbound's UNSEEN taxi-in;
* parking and tows: the "park locations" after touchdown (runs >= 600 s, or >= 180 s within 60 m of an OSM stand),
  merged when < 80 m apart; >= 2 locations = a relocation / tow; the arrival record's stand differing from the
  departure's STAND_mvt is a recorded tow (or a stand change) independent of ADS-B reception;
* transponder continuity: largest gap between touchdown and takeoff, and whether the departure's track (the
  stretch after the last > 1800 s gap, the same start as ``adsb_first_seen``) is the inbound's track continued;
* the callsign switch: the first point carrying the outbound callsign (crew set up the next flight) after the last
  point carrying another one (24 h look-back, inbound or not), how precisely it is timed, where it happened (ground
  speed, distance to the stand and to the final park location).

Taxi estimates (``*_est``, seconds before the RECORDED takeoff = t_mvt):

* ``it_mirror_est``: the departure's first sighting as a taxi estimate plus the inbound's unseen taxi-in, for
  inbound tracks that were lost while taxiing (the outbound is assumed invisible over the same stretch);
* ``it_mirror2_est``: the same with the unseen time scaled by (first-sighting distance to the departure stand) /
  (inbound loss point's distance to the arrival stand), clipped to [0, 3];
* ``it_path_est``: the departure's first sighting as a taxi estimate plus the time the INBOUND took from its point
  nearest to that first sighting (within 200 m, ``it_path_d``) to its recorded in-block (``it_path_in_s``): the
  aircraft's own measured time over the stretch stand <-> first sighting, in the other direction;
* ``it_cs_switch_est``: the outbound-callsign switch time (an upper bound on the taxi when it happened at the stand);
* ``it_final_park_last_est``: the last point within 25 m of the final park location (upper bound on the taxi);
* ``it_tow_end_est``: the arrival at the final park location after a relocation (upper bound on the taxi);
* ``it_out_move_est``: the first ground movement (>= 3 kt) under the OUTBOUND callsign after the inbound track's end:
  the aircraft was moving as the departing flight then, so the off-block came earlier (lower bound on the taxi);
* ``it_old_cs_last_est`` / ``it_old_cs_move_last_est``: the last ground point / movement under another callsign after
  the inbound track's end (a tow or repositioning flown under the inbound callsign: upper bound on the taxi).

Other times are seconds before t_mvt as well (suffix ``_ago``). Only ADS-B points before the departure's ADS-B
takeoff, the links, the geometry tables, and movement columns available in 2026 (arrivals' actual times included)
are read. Nothing reads a departure target.

Operational tier: M for the family as screened. The callsign columns mostly observe the departure moving under its
outbound callsign (after its off-block), and the mirror / path estimates add to the departure's own first sighting.
The inbound, arrival-record, parking and tow quantities alone are O in substance (landing, in-block, park position,
arrival stand all precede the off-block), but finding the aircraft goes through the takeoff link (an operational
system would use the flight plan's 24-bit address) and the times are anchored on the recorded takeoff (T).

Screen (gain vs control on the v38a harness): v1 all columns -0.632, v2 callsign
columns alone -0.420, v3 inbound / arrival / parking columns alone -0.244; both months improve in each.

    python lab/ideas/inbound_turn.py days [--workers 2] [--force]   # every ADS-B day -> cache/lab/ideas/inbound_turn_days/
    python lab/ideas/inbound_turn.py assemble                       # -> cache/lab/ideas/inbound_turn.parquet (both years)
    python lab/ideas/inbound_turn.py --days 2025-01-15 ...          # selected days only
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from prc import data, paths  # noqa: E402

KEY = "inbound_turn"
OUT = paths.CACHE / "lab" / "ideas"
DAY_DIR = OUT / f"{KEY}_days"
LOOKBACK_S = 24 * 3600.0
K = 111_320.0

AIR_GS = 60.0
TKO_GUARD_S = 180.0
TD_MAX_GAP_S = 600.0
RUN_R_M = 20.0
PARK_MIN_S = 180.0
PARK_LONG_S = 600.0
PARK_STAND_M = 60.0
LOC_MERGE_M = 80.0
LOST_GAP_S = 600.0
SEG_GAP_S = 1800.0
ARR_CS_TOL_S = 300.0
ARR_TIME_TOL_S = 45.0
UNSEEN_MAX_S = 1800.0
PATH_MAX_D_M = 200.0


def _norm_cs(s) -> pd.Series:
    return pd.Series(s).astype("string").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)


def _epoch(s: pd.Series) -> np.ndarray:
    return (s - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy(dtype="float64")


_MV = None


def _movements():
    """Departures (index mvt_id: apt, stand, t_mvt) and arrivals (apt, cs, ALDT, IBT, stand), target-free."""
    global _MV
    if _MV is None:
        m = data.load_movements(columns=["is_dep", "apt", "STAND_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
                                         "CALLSIGN_flt", "FLIGHT_mvt"])
        dep = m[m["is_dep"]]
        arr = m[~m["is_dep"]]
        d = pd.DataFrame({"apt": dep["apt"].astype(str), "stand": dep["STAND_mvt"].astype("string").fillna("").astype(str)},
                         index=dep.index)
        a = pd.DataFrame({"apt": arr["apt"].astype(str),
                          "cs": _norm_cs(arr["CALLSIGN_flt"].astype("string").fillna(arr["FLIGHT_mvt"].astype("string"))).to_numpy(),
                          "aldt": _epoch(arr["MVT_TIME_UTC_mvt"]), "ibt": _epoch(arr["BLOCK_TIME_UTC_mvt"]),
                          "stand": arr["STAND_mvt"].astype("string").fillna("").astype(str).to_numpy()}, index=arr.index)
        a.index.name = "arr_id"
        _MV = (d, a)
    return _MV


_GEO = None


def _geometry():
    """(apt, stand_mvt) -> lat/lon of the matched OSM stand; per airport a KD-tree over every OSM stand (local metres)."""
    global _GEO
    if _GEO is None:
        st = pd.read_parquet(paths.GEO_STANDS_FILE, columns=["apt", "stand_id", "lat", "lon"])
        st["apt"] = st["apt"].astype(str)
        sm = pd.read_parquet(paths.GEO_STAND_MATCHES_FILE, columns=["apt", "stand_mvt", "stand_id"])
        sm = sm.assign(apt=sm["apt"].astype(str), stand_mvt=sm["stand_mvt"].astype(str))
        sm = sm.merge(st, on=["apt", "stand_id"], how="inner").drop_duplicates(["apt", "stand_mvt"])
        pos = {(a, s): (la, lo) for a, s, la, lo in zip(sm["apt"], sm["stand_mvt"], sm["lat"], sm["lon"])}
        trees = {}
        for a, g in st.groupby("apt"):
            lat0, lon0 = float(g["lat"].median()), float(g["lon"].median())
            xy = np.c_[(g["lon"].to_numpy() - lon0) * K * np.cos(np.radians(lat0)), (g["lat"].to_numpy() - lat0) * K]
            trees[a] = (lat0, lon0, cKDTree(xy))
        _GEO = (pos, trees)
    return _GEO


def _dist(lat1, lon1, lat2, lon2) -> float:
    if not (np.isfinite(lat1) and np.isfinite(lat2)):
        return np.nan
    return float(np.hypot((lat2 - lat1) * K, (lon2 - lon1) * K * np.cos(np.radians(lat1))))


def _day_points(day: str, icaos: set) -> pd.DataFrame:
    cols = ["icao", "t", "lat", "lon", "alt_ft", "on_ground", "gs_kt", "callsign", "apt"]
    parts = []
    prev = (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    for d in (prev, day):
        f = paths.ADSB_POINTS / f"{d}.parquet"
        if f.exists():
            q = pd.read_parquet(f, columns=cols)
            parts.append(q[q["icao"].isin(icaos)])
    p = pd.concat(parts, ignore_index=True)
    p = p.assign(apt=p["apt"].astype(str)).sort_values(["icao", "apt", "t"], kind="stable")
    p = p[~p.duplicated(["icao", "apt", "t"])].reset_index(drop=True)
    return p


def _runs(t, x, y, min_dwell):
    """Maximal runs of points within RUN_R_M of the run's first point with dwell >= min_dwell: (i0, i1)."""
    out = []
    i, n = 0, len(t)
    while i < n:
        j = i
        while j + 1 < n and (x[j + 1] - x[i]) ** 2 + (y[j + 1] - y[i]) ** 2 <= RUN_R_M ** 2:
            j += 1
        if t[j] - t[i] >= min_dwell:
            out.append((i, j))
        i = j + 1
    return out


def _cs_switch(t, lat, lon, gs, ong, cs, slat, slon) -> dict:
    """The switch to the outbound callsign (the one carried just before takeoff) over the 24 h look-back: the first
    point carrying it after the last point carrying another (non-empty) callsign. ``it_cs_prev_gap`` = seconds
    between those two points (small: the transponder stayed on and the switch is timed; large: it came back on with
    the new callsign set), ``it_cs_prev_seen`` = 0 when the look-back holds no other callsign at all."""
    out: dict = {}
    valid = np.array([isinstance(c, str) and bool(c) for c in cs], dtype=bool)
    vi = np.nonzero(valid)[0]
    if not len(vi):
        return out
    cs_out = cs[vi[-1]]
    is_out = valid & np.array([c == cs_out for c in cs], dtype=bool)
    io = np.nonzero(valid & ~is_out)[0]
    out["it_cs_prev_seen"] = int(len(io) > 0)
    if not len(io):
        return out
    jp = int(io[-1])
    hit = np.nonzero(is_out[jp + 1:])[0]
    if not len(hit):
        return out
    h = jp + 1 + int(hit[0])
    out["_cs_h"] = h
    out["it_cs_switch_t"] = float(t[h])
    out["it_cs_prev_gap"] = float(t[h] - t[jp])
    out["it_cs_prev_d"] = _dist(float(lat[jp]), float(lon[jp]), float(lat[h]), float(lon[h]))
    out["it_cs_switch_gs"] = float(gs[h])
    out["it_cs_switch_ground"] = int(bool(ong[h]) or gs[h] < 40)
    out["it_cs_switch_d_stand"] = _dist(float(lat[h]), float(lon[h]), slat, slon)
    return out


def _cs_moves(t, lat, lon, gs, ong, cs, start, slat, slon) -> dict:
    """Ground movement (>= 3 kt) from index `start` on, split by the callsign it was flown under: the first movement
    as the outbound flight (the departure itself: off-block before it), and the last movement under another
    callsign (a tow or repositioning under the inbound callsign: off-block after it)."""
    out: dict = {}
    valid = np.array([isinstance(c, str) and bool(c) for c in cs], dtype=bool)
    vi = np.nonzero(valid)[0]
    if not len(vi) or start >= len(t):
        return out
    cs_out = cs[vi[-1]]
    is_out = valid & np.array([c == cs_out for c in cs], dtype=bool)
    after = np.arange(len(t)) >= start
    ground = ong | (gs < 40)
    moving = ground & (gs >= 3.0) & after
    om = np.nonzero(moving & is_out)[0]
    if len(om):
        k = int(om[0])
        out["it_out_move_t"] = float(t[k])
        out["it_out_move_gs"] = float(gs[k])
        out["it_out_move_d_stand"] = _dist(float(lat[k]), float(lon[k]), slat, slon)
    old = np.nonzero(after & ground & valid & ~is_out)[0]
    out["it_old_cs_n"] = int(len(old))
    if len(old):
        out["it_old_cs_last_t"] = float(t[old[-1]])
        la0, lo0 = float(lat[old[0]]), float(lon[old[0]])
        dd = np.hypot((lat[old] - la0) * K, (lon[old] - lo0) * K * np.cos(np.radians(la0)))
        out["it_old_cs_span_m"] = float(dd.max())
        mv = old[moving[old]]
        if len(mv):
            out["it_old_cs_move_last_t"] = float(t[mv[-1]])
    return out


def analyse(t, lat, lon, gs, ong, cs, tk, apt, slat, slon, trees, alt=None) -> dict:
    """Inbound / turnaround quantities of one aircraft (arrays sorted by t, all < tk), times as epoch seconds."""
    r: dict = {"it_n_pts24": int(len(t))}
    n = len(t)
    if n == 0:
        r["it_inb_found"] = 0
        return r
    r.update(_cs_switch(t, lat, lon, gs, ong, cs, slat, slon))
    air = (~ong) & (gs >= AIR_GS) & (t < tk - TKO_GUARD_S)
    ia = np.nonzero(air)[0]
    ground_after = ia[-1] + 1 if len(ia) else 0
    # the departure segment: points after the last gap > SEG_GAP_S (the start adsb_first_seen uses), after touchdown
    gaps_all = np.nonzero(np.diff(t) > SEG_GAP_S)[0]
    if not len(ia) or ground_after >= n:
        r["it_inb_found"] = 0
        r["it_first24_t"] = float(t[0])
        r.update(_cs_moves(t, lat, lon, gs, ong, cs, 0, slat, slon))
        return r
    i_td = ground_after
    r["it_inb_found"] = 1
    # touchdown: the first ground point when it follows the last airborne point within TD_MAX_GAP_S; otherwise the
    # landing itself was not received on the ground (e.g. Heathrow's sparse ground coverage) and is extrapolated from
    # the last airborne point at ~750 ft/min
    r["it_td_gap"] = float(t[i_td] - t[ia[-1]])
    td_seen = r["it_td_gap"] <= TD_MAX_GAP_S
    r["it_td_seen"] = int(td_seen)
    if td_seen:
        r["it_td_t"] = float(t[i_td])
    else:
        a_ft = float(alt[ia[-1]]) if alt is not None and np.isfinite(alt[ia[-1]]) else 1500.0
        r["it_td_t"] = float(t[ia[-1]] + np.clip(a_ft, 0.0, 5000.0) / 750.0 * 60.0)
    # the inbound callsign (last non-null near the last airborne point) and the outbound one (near takeoff)
    near = cs[max(0, ia[-1] - 5):i_td + 3]
    near = [c for c in near if isinstance(c, str) and c]
    r["_cs_in"] = near[-1] if near else None
    # local metres around the touchdown-and-after part
    g = slice(i_td, n)
    tg, lag, log_, gsg = t[g], lat[g], lon[g], gs[g]
    lat0, lon0 = float(np.median(lag)), float(np.median(log_))
    x = (log_ - lon0) * K * np.cos(np.radians(lat0))
    y = (lag - lat0) * K
    m = len(tg)
    gaps = np.nonzero(np.diff(tg) > LOST_GAP_S)[0]
    runs = _runs(tg, x, y, PARK_MIN_S)
    first_gap = int(gaps[0]) if len(gaps) else None
    first_run = runs[0] if runs else None
    if not td_seen:
        r["it_end_kind"] = "unseen"
        iend_g = None
        elat = elon = np.nan
    elif first_run is not None and (first_gap is None or first_gap >= first_run[0]):
        a, b = first_run
        r["it_end_kind"] = "park"
        iend_g = a
        r["it_end_t"] = float(tg[a])
        elat, elon = float(np.median(lag[a:b + 1])), float(np.median(log_[a:b + 1]))
    elif first_gap is not None:
        r["it_end_kind"] = "lost"
        iend_g = first_gap
        r["it_end_t"] = float(tg[first_gap])
        elat, elon = float(lag[first_gap]), float(log_[first_gap])
        r["it_end_gs"] = float(gsg[first_gap])
    else:
        r["it_end_kind"] = "cont"
        iend_g = None
        elat = elon = np.nan
    r["it_end_lat"], r["it_end_lon"] = elat, elon
    r["it_end_d_stand"] = _dist(elat, elon, slat, slon)
    # park locations: runs >= PARK_LONG_S, or >= PARK_MIN_S within PARK_STAND_M of an OSM stand
    locs = []  # [x, y, t_start, t_end_last_point, lat, lon]
    tr = trees.get(apt)
    for a, b in runs:
        cx, cy = float(np.median(x[a:b + 1])), float(np.median(y[a:b + 1]))
        clat, clon = float(np.median(lag[a:b + 1])), float(np.median(log_[a:b + 1]))
        dwell = tg[b] - tg[a]
        ok = dwell >= PARK_LONG_S
        if not ok and tr is not None:
            la0, lo0, tree = tr
            px = (clon - lo0) * K * np.cos(np.radians(la0))
            py = (clat - la0) * K
            ok = tree.query([px, py])[0] <= PARK_STAND_M
        if not ok:
            continue
        if locs and np.hypot(cx - locs[-1][0], cy - locs[-1][1]) < LOC_MERGE_M:
            locs[-1][3] = float(tg[b])
        else:
            locs.append([cx, cy, float(tg[a]), float(tg[b]), clat, clon])
    r["it_n_park_locs"] = len(locs)
    if locs:
        fx, fy, fts, fte, flat, flon = locs[-1]
        r["it_final_park_d_stand"] = _dist(flat, flon, slat, slon)
        r["it_first_park_d_stand"] = _dist(locs[0][4], locs[0][5], slat, slon)
        # last point within 25 m of the final park location (sustained presence), before takeoff
        dd = np.hypot(x - fx, y - fy)
        inside = np.nonzero(dd <= 25.0)[0]
        r["it_final_park_last_t"] = float(tg[inside[-1]]) if len(inside) else np.nan
        r["it_final_park_start_t"] = fts
        if len(locs) >= 2:
            px_, py_ = locs[-2][0], locs[-2][1]
            r["it_tow_dist"] = float(np.hypot(fx - px_, fy - py_))
            r["it_tow_end_t"] = fts
            r["it_tow_start_t"] = locs[-2][3]
    # transponder continuity through the turnaround
    r["it_turn_max_gap"] = float(np.max(np.diff(np.append(tg, tk))))
    seg_start = int(gaps_all[-1] + 1) if len(gaps_all) else 0
    seg_start = max(seg_start, i_td)
    r["it_contig"] = int(seg_start == i_td) if td_seen else np.nan
    r["it_dep_first_t"] = float(t[seg_start])
    r["it_dep_first_d_end"] = _dist(elat, elon, float(lat[seg_start]), float(lon[seg_start]))
    r["it_dep_first_d_stand"] = _dist(float(lat[seg_start]), float(lon[seg_start]), slat, slon)
    # the inbound's own path past the departure's first sighting: the inbound point nearest to it, and how long the
    # inbound took from there to the end of its track (and, in compute_day, to its recorded in-block)
    j_rel = seg_start - i_td
    if iend_g is not None and j_rel > iend_g:
        dd = np.hypot(x[:iend_g + 1] - x[j_rel], y[:iend_g + 1] - y[j_rel])
        j = int(np.argmin(dd))
        r["it_path_d"] = float(dd[j])
        r["_path_t"] = float(tg[j])
        r["it_path_in_ads"] = float(tg[iend_g] - tg[j])
    # points per hour while parked (between the inbound end and the departure segment start)
    if np.isfinite(r.get("it_end_t", np.nan)) and r["it_dep_first_t"] > r["it_end_t"] + 60:
        sel = (tg > r["it_end_t"]) & (tg < r["it_dep_first_t"])
        r["it_turn_pts_h"] = float(sel.sum() / ((r["it_dep_first_t"] - r["it_end_t"]) / 3600.0))
    # the callsign switch's position against the final park location
    if "_cs_h" in r and locs:
        h = r["_cs_h"]
        r["it_cs_switch_d_park"] = _dist(float(lat[h]), float(lon[h]), locs[-1][4], locs[-1][5])
    # movements under the outbound vs another callsign, after the inbound track's end (after touchdown if unseen)
    r.update(_cs_moves(t, lat, lon, gs, ong, cs, i_td + (iend_g + 1 if iend_g is not None else 0), slat, slon))
    return r


def _link_arrivals(td: pd.DataFrame, arr: pd.DataFrame) -> pd.DataFrame:
    """Link inbound touchdowns (index mvt_id; apt, td_t, cs_in) to arrival records: callsign + time, else time alone."""
    if td.empty:
        return pd.DataFrame(columns=["arr_id", "arr_dt", "arr_by_time"])
    lo, hi = td["td_t"].min() - ARR_CS_TOL_S, td["td_t"].max() + ARR_CS_TOL_S
    a = arr[(arr["aldt"] >= lo) & (arr["aldt"] <= hi)].reset_index()
    d = td.reset_index()
    d["cs"] = _norm_cs(d["cs_in"]).to_numpy()
    m = d.dropna(subset=["cs"]).merge(a, on=["apt", "cs"], how="inner")
    m["arr_dt"] = m["td_t"] - m["aldt"]
    m = m[m["arr_dt"].abs() <= ARR_CS_TOL_S]
    m = m.reindex(m["arr_dt"].abs().sort_values().index).drop_duplicates("mvt_id").drop_duplicates("arr_id")
    out = m.set_index("mvt_id")[["arr_id", "arr_dt"]].assign(arr_by_time=0)
    rest = d[~d["mvt_id"].isin(out.index)]
    used = set(out["arr_id"])
    extra = []
    for apt, g in rest.groupby("apt"):
        aa = a[(a["apt"] == apt) & ~a["arr_id"].isin(used)].sort_values("aldt")
        ta = aa["aldt"].to_numpy()
        ids = aa["arr_id"].to_numpy()
        for mid, tt in zip(g["mvt_id"], g["td_t"]):
            c = np.nonzero(np.abs(ta - tt) <= ARR_TIME_TOL_S)[0]
            if len(c) == 1:
                extra.append((mid, ids[c[0]], float(tt - ta[c[0]]), 1))
    if extra:
        ex = pd.DataFrame(extra, columns=["mvt_id", "arr_id", "arr_dt", "arr_by_time"])
        ex = ex[~ex.duplicated("arr_id", keep=False)].set_index("mvt_id")
        out = pd.concat([out, ex])
    return out


def compute_day(day: str) -> pd.DataFrame:
    """Inbound / turnaround features of every linked departure of `day` (index mvt_id)."""
    lk = pd.read_parquet(paths.CACHE / "adsb" / f"links_{day}.parquet")
    dep, arr = _movements()
    pos, trees = _geometry()
    p = _day_points(day, set(lk["icao"]))
    groups = p.groupby(["icao", "apt"], sort=False).indices
    T = p["t"].to_numpy(dtype="float64")
    LA, LO = p["lat"].to_numpy(dtype="float64"), p["lon"].to_numpy(dtype="float64")
    GS = p["gs_kt"].to_numpy(dtype="float64")
    ALT = p["alt_ft"].to_numpy(dtype="float64")
    ONG = p["on_ground"].to_numpy(bool)
    CS = p["callsign"].astype("string").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True).astype(object)
    CS = CS.where(CS.notna(), None).to_numpy()
    dv = dep.reindex(lk.index)
    rows = {}
    for mvt_id, icao, apt, tk, tm in zip(lk.index, lk["icao"], lk["apt"].astype(str), lk["t_takeoff"], lk["t_mvt"]):
        idx = groups.get((icao, apt))
        if idx is None:
            continue
        tt = T[idx]
        sel = idx[np.searchsorted(tt, tk - LOOKBACK_S):np.searchsorted(tt, tk)]
        sl = pos.get((apt, dv.at[mvt_id, "stand"]), (np.nan, np.nan)) if mvt_id in dv.index else (np.nan, np.nan)
        r = analyse(T[sel], LA[sel], LO[sel], GS[sel], ONG[sel], CS[sel], tk, apt, sl[0], sl[1], trees, ALT[sel])
        r["_tm"] = tm
        r["_apt"] = apt
        rows[mvt_id] = r
    f = pd.DataFrame.from_dict(rows, orient="index")
    f.index.name = "mvt_id"
    if f.empty:
        return f
    # link the inbound touchdown to the arrival record
    has = f["it_inb_found"] == 1
    td = pd.DataFrame({"apt": f.loc[has, "_apt"], "td_t": f.loc[has, "it_td_t"], "cs_in": f.loc[has, "_cs_in"]})
    al = _link_arrivals(td, arr)
    f["it_arr_linked"] = f.index.isin(al.index).astype("int8")
    f["it_arr_dt"] = al["arr_dt"].reindex(f.index)
    f["it_arr_by_time"] = al["arr_by_time"].reindex(f.index)
    ar = arr.reindex(al["arr_id"].reindex(f.index).to_numpy())
    ar.index = f.index
    tm = f["_tm"].astype("float64")
    ibt = ar["ibt"]
    dstand = dep["stand"].reindex(f.index)
    f["it_arr_same_stand"] = np.where(f["it_arr_linked"] == 1, (ar["stand"] == dstand).astype(float), np.nan)
    f["it_arr_stand_known"] = np.where(f["it_arr_linked"] == 1, (ar["stand"].fillna("") != "").astype(float), np.nan)
    f["it_arr_taxiin"] = ibt - ar["aldt"]
    f["it_arr_inblock_ago"] = tm - ibt
    # the arrival stand's position, and the inbound end's distance to it
    alat = np.array([pos.get((a, s), (np.nan, np.nan))[0] for a, s in zip(f["_apt"], ar["stand"].fillna(""))])
    alon = np.array([pos.get((a, s), (np.nan, np.nan))[1] for a, s in zip(f["_apt"], ar["stand"].fillna(""))])
    elat = f["it_end_lat"].to_numpy(dtype="float64") if "it_end_lat" in f else np.full(len(f), np.nan)
    elon = f["it_end_lon"].to_numpy(dtype="float64") if "it_end_lon" in f else np.full(len(f), np.nan)
    f["it_end_d_arrstand"] = np.hypot((alat - elat) * K, (alon - elon) * K * np.cos(np.radians(elat)))
    slat = np.array([pos.get((a, s), (np.nan, np.nan))[0] for a, s in zip(f["_apt"], dstand.fillna(""))])
    slon = np.array([pos.get((a, s), (np.nan, np.nan))[1] for a, s in zip(f["_apt"], dstand.fillna(""))])
    f["it_arrstand_d_depstand"] = np.hypot((alat - slat) * K, (alon - slon) * K * np.cos(np.radians(slat)))
    end_t = f["it_end_t"] if "it_end_t" in f else pd.Series(np.nan, index=f.index)
    f["it_unseen_in"] = ibt - end_t
    f["it_ads_ib_err"] = np.where(f.get("it_end_kind", pd.Series(index=f.index, dtype=object)) == "park", end_t - ibt, np.nan)
    # times as seconds before the recorded takeoff
    for c in ("it_td_t", "it_end_t", "it_dep_first_t", "it_final_park_start_t", "it_tow_start_t", "it_first24_t"):
        if c in f:
            f[c[:-2] + "_ago"] = tm - f[c]
    f = f.rename(columns={"it_td_ago": "it_land_ago"})
    # taxi estimates
    first_est = tm - f["it_dep_first_t"] if "it_dep_first_t" in f else pd.Series(np.nan, index=f.index)
    kind = f.get("it_end_kind", pd.Series(index=f.index, dtype=object))
    sep = f.get("it_contig", pd.Series(np.nan, index=f.index)) == 0
    unseen = f["it_unseen_in"].clip(0, UNSEEN_MAX_S)
    mir = (kind == "lost") & sep & f["it_unseen_in"].notna()
    f["it_mirror_est"] = np.where(mir, first_est + unseen, np.nan)
    ratio = (f.get("it_dep_first_d_stand") / f["it_end_d_arrstand"].clip(lower=30.0)).clip(0, 3)
    f["it_mirror2_est"] = np.where(mir & ratio.notna(), first_est + unseen * ratio, np.nan)
    if "_path_t" in f:
        f["it_path_in_s"] = ibt - f["_path_t"]
        ok = (f["it_path_d"] <= PATH_MAX_D_M) & f["it_path_in_s"].between(0, 3600)
        f["it_path_est"] = np.where(ok, first_est + f["it_path_in_s"], np.nan)
    if "it_cs_switch_t" in f:
        f["it_cs_switch_est"] = tm - f["it_cs_switch_t"]
    if "it_final_park_last_t" in f:
        f["it_final_park_last_est"] = tm - f["it_final_park_last_t"]
    if "it_tow_end_t" in f:
        f["it_tow_end_est"] = tm - f["it_tow_end_t"]
    if "it_out_move_t" in f:
        f["it_out_move_est"] = tm - f["it_out_move_t"]
    for c in ("it_old_cs_last_t", "it_old_cs_move_last_t"):
        if c in f:
            f[c[:-2] + "_est"] = tm - f[c]
    drop = [c for c in f.columns if c.startswith("_") or (c.endswith("_t") and c.startswith("it_"))
            or c in ("it_end_lat", "it_end_lon")]
    f = f.drop(columns=drop)
    if "it_end_kind" in f:
        f["it_end_kind"] = f["it_end_kind"].fillna("none").astype(str)
    return f


def _one(day: str) -> str:
    DAY_DIR.mkdir(parents=True, exist_ok=True)
    out = compute_day(day)
    out.to_parquet(DAY_DIR / f"{day}.parquet")
    return (f"{day}: {len(out):,} rows, inbound {out['it_inb_found'].mean():.3f}, "
            f"arr linked {out['it_arr_linked'].mean():.3f}")


ALL_DAYS = [d.strftime("%Y-%m-%d") for d in list(pd.date_range("2025-01-01", "2025-01-31"))
            + list(pd.date_range("2025-07-01", "2025-07-31")) + list(pd.date_range("2026-01-01", "2026-01-31"))
            + list(pd.date_range("2026-07-01", "2026-07-31"))]


def run(days: list[str], workers: int = 1):
    todo = [d for d in days if "--force" in sys.argv or not (DAY_DIR / f"{d}.parquet").exists()]
    if workers > 1 and len(todo) > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            for msg in pool.imap_unordered(_one, todo):
                print(msg, flush=True)
    else:
        for d in todo:
            print(_one(d), flush=True)


def assemble() -> pd.DataFrame:
    fr = pd.concat([pd.read_parquet(DAY_DIR / f"{d}.parquet") for d in ALL_DAYS if (DAY_DIR / f"{d}.parquet").exists()])
    fr = fr[~fr.index.duplicated(keep="last")]
    fr.to_parquet(OUT / f"{KEY}.parquet")
    print(f"wrote {OUT / f'{KEY}.parquet'}: {len(fr):,} rows")
    return fr


if __name__ == "__main__":
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 1
    if "days" in sys.argv[1:2]:
        run(ALL_DAYS, workers)
    if "--days" in sys.argv:
        run([a for a in sys.argv[sys.argv.index("--days") + 1:] if not a.startswith("--")], workers)
    if "assemble" in sys.argv[1:]:
        assemble()
