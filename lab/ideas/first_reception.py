"""Idea family: the state of a departing aircraft at its first ADS-B reception.

For a departure first received away from its stand, the part ADS-B never saw is off-block -> first sighting. This
module describes the aircraft at that first sighting and estimates how much of the stand -> runway route it had already
covered, so a second stage can estimate the unobserved time:

* kinematics at the first point (the first ground point of the segment after the last > 30 min gap, the same point as
  `adsb_first_seen` / `adsb_first_lat/lon` of `prc.adsb.events.own_movement`): ground speed there and over the first
  60 s, displacement after 60 s, moving vs stationary, the direction of motion relative to the bearing stand -> first
  point and first point -> departure threshold;
* route coverage: the great-circle distance first point -> departure threshold (RUNWAY_mvt normalised like '07L') and
  the OSM-network distance from the first point (snapped to the taxi graph) to that threshold, against the stand's
  route length (`stand_runway_routes.route_m`, same graph): covered metres and covered fraction at first reception;
* unobserved-time estimates: covered metres / a fixed 7 m/s, and covered metres / the pace the aircraft showed once
  seen (observed ground path / observed time), each turned into a taxi estimate
  first_seen - adsb_takeoff_offset + unobserved estimate;
* (second pass, `stand_stats`) a target-free reception boundary per (airport, STAND_mvt) over all 124 ADS-B days: the
  median first-sighting distance and the median covered metres of that stand's departures, and the row's deviation.

Only ADS-B points before the ADS-B takeoff, the links, the geometry tables and movement columns are read, so 2026 rows
are computed identically. Nothing here reads a target.

    python lab/ideas/first_reception.py graph                 # per-airport network distances (once)
    python lab/ideas/first_reception.py days [--workers 2]    # every ADS-B day -> cache/lab/ideas/first_reception/<day>.parquet
    python lab/ideas/first_reception.py assemble              # holdout rows + stand stats -> cache/lab/ideas/first_reception.parquet
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from prc import data, paths  # noqa: E402
from prc.features._geolib import haversine_m, local_east_north_m, norm_runway_ident  # noqa: E402

KEY = "first_reception"
OUT = paths.CACHE / "lab" / "ideas"
DAY_DIR = OUT / KEY
GRAPH_DIR = OUT / f"{KEY}_graph"
LOOKBACK_S = 4 * 3600.0
CUT_GAP_S = 1800.0
SNAP_M = 150.0
"""A first point farther than this from any usable taxi-graph node gets no network distance."""
V_TYP = 7.0
"""Fixed taxi speed (m/s, ~13.6 kt) for the first unobserved-time estimate."""
KT = 0.514444


# --- the taxi graph: network metres from every usable node to every runway threshold ------------------------------

def build_graphs() -> None:
    """Rebuild each airport's taxi graph exactly as `prc.geometry.normalise` does (same anchors, same bridges) and
    store, per runway end, the true metres along the penalised-shortest route from every usable node."""
    from scipy.sparse.csgraph import dijkstra

    from prc.geometry import routing, sources
    from prc.geometry.normalise import BRIDGE_GAP_M, RUNWAY_FALLBACK_M, RUNWAY_PENALTY

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    apts = pd.read_parquet(paths.GEO_AIRPORTS_FILE)
    icaos = sorted(apts["apt"].astype(str))
    kept, _ = sources.filter_runways(sources.load_ourairports_runways(icaos))
    ends_all = sources.runway_ends(kept)
    for r in apts.itertuples(index=False):
        apt, lat0, lon0 = str(r.apt), float(r.arp_lat), float(r.arp_lon)
        payload = json.loads(paths.geometry_raw_file("osm", f"{apt}.json").read_text(encoding="utf-8"))
        nodes, ways = sources.parse_overpass(payload)
        g = routing.build_graph(nodes, ways, lat0=lat0, lon0=lon0, runway_penalty=RUNWAY_PENALTY)
        ends = routing.anchor_runway_ends(g, ends_all.loc[ends_all["apt"].eq(apt)], ways, lat0=lat0, lon0=lon0,
                                          fallback_m=RUNWAY_FALLBACK_M)
        routing.bridge_gaps(g, max_gap_m=BRIDGE_GAP_M)
        usable = np.flatnonzero(g.keep & (g.degree > 0))
        ends = ends[ends["graph_node"] >= 0]
        sources_ = np.unique(ends["graph_node"].to_numpy())
        dist, pred = dijkstra(g.weights, directed=False, indices=sources_, return_predecessors=True)
        out = {"east": g.east[usable], "north": g.north[usable], "lat0": lat0, "lon0": lon0}
        idents = []
        for e in ends.itertuples(index=False):
            k = int(np.searchsorted(sources_, e.graph_node))
            true, _ = routing._tree_accumulate(g, dist[k], pred[k], int(e.graph_node))
            out[f"d_{e.ident}"] = true[usable].astype("float32")
            idents.append(e.ident)
        out["idents"] = np.array(idents)
        np.savez_compressed(GRAPH_DIR / f"{apt}.npz", **out)
        print(f"{apt}: {len(usable):,} usable nodes, ends {idents}", flush=True)


_GRAPHS: dict = {}


def _graph(apt: str):
    if apt not in _GRAPHS:
        f = GRAPH_DIR / f"{apt}.npz"
        if not f.exists():
            _GRAPHS[apt] = None
        else:
            from scipy.spatial import cKDTree
            z = np.load(f)
            _GRAPHS[apt] = {"tree": cKDTree(np.column_stack([z["east"], z["north"]])), "lat0": float(z["lat0"]),
                            "lon0": float(z["lon0"]),
                            "d": {str(i): z[f"d_{i}"] for i in z["idents"]}}
    return _GRAPHS[apt]


def network_to_threshold(apt: str, ident: str | None, lat: float, lon: float) -> float:
    """Network metres from the taxi-graph node nearest (lat, lon) (within SNAP_M) to runway end `ident`."""
    g = _graph(apt)
    if g is None or ident is None or ident not in g["d"]:
        return np.nan
    e, n = local_east_north_m(lat, lon, g["lat0"], g["lon0"])
    dd, i = g["tree"].query((float(e), float(n)), k=1, distance_upper_bound=SNAP_M)
    if not np.isfinite(dd):
        return np.nan
    return float(g["d"][ident][i])


# --- static lookups -----------------------------------------------------------------------------------------------

_STATIC = None


def _static():
    global _STATIC
    if _STATIC is None:
        st = pd.read_parquet(paths.GEO_STANDS_FILE, columns=["apt", "stand_id", "lat", "lon"])
        sm = pd.read_parquet(paths.GEO_STAND_MATCHES_FILE, columns=["apt", "stand_mvt", "stand_id"])
        sm = sm.merge(st, on=["apt", "stand_id"], how="left")
        sm = sm.assign(apt=sm["apt"].astype(str), stand_mvt=sm["stand_mvt"].astype(str))
        stand = sm.set_index(["apt", "stand_mvt"])[["stand_id", "lat", "lon"]]
        ends = pd.read_parquet(paths.GEO_RUNWAY_ENDS_FILE)
        ends = ends[ends["dropped_reason"].fillna("").eq("")]
        thr = ends.assign(apt=ends["apt"].astype(str), ident=ends["ident"].astype(str)).set_index(["apt", "ident"])[["lat", "lon"]]
        rt = pd.read_parquet(paths.GEO_ROUTES_FILE, columns=["apt", "stand_id", "runway_ident", "gc_m", "route_m", "route_found"])
        rt = rt.assign(apt=rt["apt"].astype(str), runway_ident=rt["runway_ident"].astype(str))
        rt.loc[rt["route_found"] == 0, "route_m"] = np.nan
        routes = rt.set_index(["apt", "stand_id", "runway_ident"])[["gc_m", "route_m"]]
        mvt = data.load_movements(columns=["apt", "is_dep", "STAND_mvt", "RUNWAY_mvt"])
        mvt = mvt[mvt["is_dep"]]
        _STATIC = (stand, thr, routes, mvt)
    return _STATIC


def _bearing(lat1, lon1, lat2, lon2):
    k = 111_320.0
    de = (lon2 - lon1) * k * np.cos(np.radians(lat1))
    dn = (lat2 - lat1) * k
    return np.degrees(np.arctan2(de, dn)) % 360.0


def _cosdiff(a, b):
    return float(np.cos(np.radians(a - b))) if np.isfinite(a) and np.isfinite(b) else np.nan


def _metres(lat0, lon0, lat, lon):
    k = 111_320.0
    return np.hypot((lat - lat0) * k, (lon - lon0) * k * np.cos(np.radians(lat0)))


# --- per day --------------------------------------------------------------------------------------------------------

def compute_day(day: str) -> pd.DataFrame:
    """One row per linked departure of `day` (index mvt_id) with the first-reception columns (prefix `fr_`)."""
    stand, thr, routes, mvt = _static()
    lk = pd.read_parquet(paths.CACHE / "adsb" / f"links_{day}.parquet")
    pts = pd.read_parquet(paths.ADSB_POINTS / f"{day}.parquet",
                          columns=["icao", "t", "lat", "lon", "on_ground", "gs_kt", "track", "apt"])
    pts = pts[pts["icao"].isin(set(lk["icao"]))]
    pts["apt"] = pts["apt"].astype(str)
    by = {key: g.sort_values("t") for key, g in pts.groupby(["icao", "apt"], sort=False)}
    m = mvt.reindex(lk.index)
    rows = {}
    for mvt_id, r in lk.iterrows():
        apt = str(r["apt"])
        g = by.get((r["icao"], apt))
        if g is None:
            continue
        tk = float(r["t_takeoff"])
        seg = g[(g["t"].to_numpy() < tk) & (g["t"].to_numpy() >= tk - LOOKBACK_S)]
        seg = seg[seg["on_ground"].to_numpy(bool) | (seg["gs_kt"].to_numpy(dtype="float64") < 40)]
        if seg.empty:
            continue
        t = seg["t"].to_numpy()
        cut = np.nonzero(np.diff(t) > CUT_GAP_S)[0]
        if len(cut):
            seg = seg.iloc[cut[-1] + 1:]
            t = seg["t"].to_numpy()
        lat, lon = seg["lat"].to_numpy(), seg["lon"].to_numpy()
        gs = seg["gs_kt"].to_numpy(dtype="float64")
        trk = seg["track"].to_numpy(dtype="float64")
        t0, la0, lo0 = t[0], lat[0], lon[0]
        disp = _metres(la0, lo0, lat, lon)
        w60 = t <= t0 + 60
        out = {"fr_first_seen": tk - t0, "fr_offset": tk - float(r["t_mvt"]), "fr_gs0": gs[0],
               "fr_gs60": float(np.nanmean(gs[w60])) if np.isfinite(gs[w60]).any() else np.nan,
               "fr_disp60": float(disp[w60].max()), "fr_n60": int(w60.sum())}
        out["fr_moving0"] = float((np.nan_to_num(gs[0]) >= 3.0) or (out["fr_disp60"] > 30.0))
        # direction of motion: from the first point to the first point at least 30 m away within 180 s
        mv = np.nonzero((disp >= 30.0) & (t <= t0 + 180))[0]
        motion = _bearing(la0, lo0, lat[mv[0]], lon[mv[0]]) if len(mv) else np.nan
        out["fr_dt_to_30m"] = float(t[mv[0]] - t0) if len(mv) else np.nan
        # pace once seen: observed ground path / observed time; moving speed over the first 5 min
        step = _metres(lat[:-1], lon[:-1], lat[1:], lon[1:]) if len(lat) > 1 else np.zeros(0)
        path = float(np.sum(step[step < 500]))
        out["fr_pace"] = path / max(tk - t0, 1.0)
        w300 = (t <= t0 + 300) & (gs >= 5)
        out["fr_move_speed"] = float(np.median(gs[w300]) * KT) if w300.any() else np.nan
        # the stand and the departure threshold
        smv = str(m.at[mvt_id, "STAND_mvt"]) if pd.notna(m.at[mvt_id, "STAND_mvt"]) else None
        ident = norm_runway_ident(m.at[mvt_id, "RUNWAY_mvt"]) if pd.notna(m.at[mvt_id, "RUNWAY_mvt"]) else None
        s_lat = s_lon = sid = np.nan
        if smv is not None and (apt, smv) in stand.index:
            sid, s_lat, s_lon = stand.loc[(apt, smv)]
        if np.isfinite(s_lat):
            d_stand = float(_metres(s_lat, s_lon, la0, lo0))
            out["fr_dist_stand"] = d_stand
            b_sf = _bearing(s_lat, s_lon, la0, lo0) if d_stand > 20 else np.nan
            out["fr_motion_vs_away"] = _cosdiff(motion, b_sf)
            out["fr_track_vs_away"] = _cosdiff(trk[0], b_sf)
        if ident is not None and (apt, ident) in thr.index:
            t_lat, t_lon = thr.loc[(apt, ident)]
            out["fr_gc_first_thr"] = float(haversine_m(la0, lo0, t_lat, t_lon))
            out["fr_motion_vs_thr"] = _cosdiff(motion, _bearing(la0, lo0, t_lat, t_lon))
            out["fr_net_first_thr"] = network_to_threshold(apt, ident, la0, lo0)
            if np.isfinite(sid) and (apt, int(sid), ident) in routes.index:
                gc, rm = routes.loc[(apt, int(sid), ident)]
                out["fr_gc_stand_thr"], out["fr_route_m"] = float(gc), float(rm)
        rows[mvt_id] = out
    f = pd.DataFrame.from_dict(rows, orient="index")
    f.index.name = "mvt_id"
    if f.empty:
        return f
    for c in ["fr_gc_stand_thr", "fr_route_m", "fr_net_first_thr", "fr_gc_first_thr", "fr_dist_stand"]:
        if c not in f:
            f[c] = np.nan
    f["fr_frac_gc"] = 1.0 - f["fr_gc_first_thr"] / f["fr_gc_stand_thr"]
    f["fr_covered_m"] = f["fr_route_m"] - f["fr_net_first_thr"]
    f["fr_frac_route"] = f["fr_covered_m"] / f["fr_route_m"]
    cov = f["fr_covered_m"].clip(lower=0.0)
    f["fr_unobs_v"] = cov / V_TYP
    f["fr_unobs_pace"] = cov / f["fr_pace"].clip(2.0, 12.0)
    f["fr_taxi_est_v"] = f["fr_first_seen"] - f["fr_offset"] + f["fr_unobs_v"]
    f["fr_taxi_est_pace"] = f["fr_first_seen"] - f["fr_offset"] + f["fr_unobs_pace"]
    return f


def _one(day: str) -> str:
    DAY_DIR.mkdir(parents=True, exist_ok=True)
    f = compute_day(day)
    f.to_parquet(DAY_DIR / f"{day}.parquet")
    return f"{day}: {len(f):,} rows, net known {f['fr_net_first_thr'].notna().mean():.2f}"


def all_days(workers: int = 2, force: bool = False) -> None:
    days = sorted(p.stem for p in paths.ADSB_POINTS.glob("*.parquet"))
    todo = [d for d in days if force or not (DAY_DIR / f"{d}.parquet").exists()]
    if workers > 1 and len(todo) > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            for msg in pool.imap_unordered(_one, todo):
                print(msg, flush=True)
    else:
        for d in todo:
            print(_one(d), flush=True)


# --- target-free reception boundary per stand, over all 124 days ----------------------------------------------------

MIN_STAND_N = 10


def stand_stats(f: pd.DataFrame) -> pd.DataFrame:
    """Per (apt, STAND_mvt), over every ADS-B day: median first-sighting distance and covered metres, the share first
    received beyond 100 m, and the row's deviation from the stand medians. No target is read."""
    _, _, _, mvt = _static()
    k = mvt.reindex(f.index)[["apt", "STAND_mvt"]].astype(str)
    g = f[["fr_dist_stand", "fr_covered_m"]].join(k).groupby(["apt", "STAND_mvt"])
    agg = pd.DataFrame({"fr_stand_med_dist": g["fr_dist_stand"].median(), "fr_stand_med_cov": g["fr_covered_m"].median(),
                        "fr_stand_far_share": g["fr_dist_stand"].apply(lambda s: (s > 100).mean()),
                        "n": g["fr_dist_stand"].size()})
    agg = agg[agg["n"] >= MIN_STAND_N].drop(columns="n")
    out = k.reset_index().merge(agg.reset_index(), on=["apt", "STAND_mvt"], how="left").set_index("mvt_id")
    out = out.drop(columns=["apt", "STAND_mvt"])
    out["fr_dev_dist"] = f["fr_dist_stand"] - out["fr_stand_med_dist"]
    out["fr_dev_cov"] = f["fr_covered_m"] - out["fr_stand_med_cov"]
    return out


def assemble() -> pd.DataFrame:
    files = sorted(DAY_DIR.glob("*.parquet"))
    f = pd.concat([pd.read_parquet(p) for p in files])
    f = f[~f.index.duplicated(keep="last")]
    f = f.join(stand_stats(f))
    f.to_parquet(OUT / f"{KEY}_all.parquet")
    sys.path.insert(0, str(ROOT / "lab"))
    from idea_harness import load_frame
    hold = f.reindex(load_frame().index)
    hold.to_parquet(OUT / f"{KEY}.parquet")
    print(f"assembled {len(f):,} rows; holdout {hold.notna().any(axis=1).sum():,} non-null")
    return f


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "graph":
        build_graphs()
    elif cmd == "days":
        w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 2
        all_days(min(w, 2), force="--force" in sys.argv)
    elif cmd == "assemble":
        assemble()
    elif cmd == "day":
        print(compute_day(sys.argv[2]).describe().T.to_string())
    else:
        print(__doc__)
