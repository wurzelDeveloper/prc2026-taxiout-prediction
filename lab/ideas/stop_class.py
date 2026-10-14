"""Idea family: the stops between leaving the stand and takeoff, classified by where they happen.

For every linked departure of a day: take the aircraft's ground points in the 4 h before its ADS-B takeoff (the segment
after the last gap > 30 min, as `prc.adsb.events.own_movement` does), find the stationary periods (runs of points
within 15 m of the run's first point lasting >= 30 s), and classify each by location against the OSM / OurAirports
geometry (all target-free, identical for 2026):

* ``stand``  : within 60 m of the flight's matched stand, or within 45 m of any OSM parking position / gate while
               away from the runways;
* ``rwy``    : within 400 m of a runway end, within 90 m of a runway centreline (between its ends), or within 60 m of
               an OSM runway / ILS holding position;
* ``remote`` : >= 300 s, neither stand nor runway area (remote de-icing pad, remote hold, a remote stand OSM lacks);
* ``hold``   : < 300 s, neither stand nor runway area (a short mid-taxi hold).

The parked position is the last ``stand`` stop; the taxi stops are those that start after it (after the first
sighting when there is none). Times are seconds before the ADS-B takeoff like the existing ADS-B features; taxi
estimates subtract ``adsb_takeoff_offset`` (t_takeoff - t_mvt).

    python lab/ideas/stop_class.py [--days D ...] [--workers 2] [--force]     # writes cache/lab/ideas/stop_class/<day>.parquet
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from prc import data, paths  # noqa: E402

OUT = paths.CACHE / "lab" / "ideas" / "stop_class"
LOOKBACK_S = 4 * 3600.0
SEG_GAP_S = 1800.0
STOP_R_M = 15.0
STOP_MIN_S = 30.0
LONG_S = 300.0
OWN_STAND_M = 60.0
ANY_STAND_M = 45.0
RWY_END_M = 400.0
RWY_CL_M = 90.0
HOLD_M = 60.0
STAND_RWY_EXCL_M = 600.0
"""A stop near an OSM stand counts as a stand stop only if it is at least this far from every runway end."""
CLASSES = ("stand", "rwy", "remote", "hold")


# ---------------------------------------------------------------------------------------------- geometry
@lru_cache(maxsize=1)
def geometry() -> dict:
    """Per airport: local projection origin and KD-trees / segments of the reference objects (east/north metres)."""
    apts = pd.read_parquet(paths.GEO_AIRPORTS_FILE, columns=["apt", "arp_lat", "arp_lon"])
    ends = pd.read_parquet(paths.GEO_RUNWAY_ENDS_FILE)
    ends = ends[(ends["dropped_reason"].fillna("") == "") & ends["lat"].notna()]
    stands = pd.read_parquet(paths.GEO_STANDS_FILE, columns=["apt", "stand_id", "lat", "lon"])
    out = {}
    for _, a in apts.iterrows():
        apt, lat0, lon0 = str(a["apt"]), float(a["arp_lat"]), float(a["arp_lon"])

        def xy(lat, lon, lat0=lat0, lon0=lon0):
            lat, lon = np.asarray(lat, float), np.asarray(lon, float)
            return np.column_stack([(lon - lon0) * 111_320.0 * np.cos(np.radians(lat0)), (lat - lat0) * 111_320.0])

        e = ends[ends["apt"].astype(str) == apt]
        exy = xy(e["lat"], e["lon"]) if len(e) else np.zeros((0, 2))
        segs = []
        for _, grp in e.groupby("pair_ref"):
            if len(grp) == 2 and grp["pair_ref"].iloc[0]:
                p = xy(grp["lat"], grp["lon"])
                segs.append((p[0], p[1]))
        s = stands[stands["apt"].astype(str) == apt]
        sxy = xy(s["lat"], s["lon"])
        # runway / ILS / untyped holding positions from the raw OSM extract (nodes, and the nodes of holding ways)
        hxy = np.zeros((0, 2))
        raw = paths.GEOMETRY_RAW / "osm" / f"{apt}.json"
        if raw.exists():
            els = json.loads(raw.read_text(encoding="utf-8"))["elements"]
            nodes = {el["id"]: (el["lat"], el["lon"]) for el in els if el["type"] == "node" and "lat" in el}
            hp = []
            for el in els:
                tg = el.get("tags", {})
                if tg.get("aeroway") != "holding_position" or tg.get("holding_position:type") == "intermediate":
                    continue
                if el["type"] == "node" and "lat" in el:
                    hp.append((el["lat"], el["lon"]))
                elif el["type"] == "way":
                    hp.extend(nodes[n] for n in el.get("nodes", []) if n in nodes)
            if hp:
                hp = np.array(hp)
                hxy = xy(hp[:, 0], hp[:, 1])
        out[apt] = {"lat0": lat0, "lon0": lon0, "xy": xy, "ends": exy, "segs": segs,
                    "stands": cKDTree(sxy) if len(sxy) else None,
                    "holds": cKDTree(hxy) if len(hxy) else None}
    return out


def _dist_to_segments(p: np.ndarray, segs) -> np.ndarray:
    """Perpendicular distance of points p (n,2) to runway centrelines, only where the foot lies between the ends."""
    best = np.full(len(p), np.inf)
    for a, b in segs:
        ab = b - a
        L2 = float(ab @ ab)
        u = ((p - a) @ ab) / L2
        foot = a + np.outer(u, ab)
        d = np.hypot(*(p - foot).T)
        d = np.where((u >= 0) & (u <= 1), d, np.inf)
        best = np.minimum(best, d)
    return best


def _distances(g: dict, p: np.ndarray) -> dict:
    d_end = np.min(np.hypot(p[:, None, 0] - g["ends"][None, :, 0], p[:, None, 1] - g["ends"][None, :, 1]), axis=1) \
        if len(g["ends"]) else np.full(len(p), np.inf)
    d_cl = _dist_to_segments(p, g["segs"])
    d_st = g["stands"].query(p)[0] if g["stands"] is not None else np.full(len(p), np.inf)
    d_hp = g["holds"].query(p)[0] if g["holds"] is not None else np.full(len(p), np.inf)
    return {"d_end": d_end, "d_cl": d_cl, "d_stand": d_st, "d_hold": d_hp}


# ---------------------------------------------------------------------------------------------- inputs
@lru_cache(maxsize=1)
def _stand_xy() -> pd.DataFrame:
    """Matched stand coordinates of every departure (both splits): index mvt_id, apt, lat, lon."""
    mv = data.load_movements(columns=["apt", "is_dep", "STAND_mvt"])
    mv = mv[mv["is_dep"]]
    st = pd.read_parquet(paths.GEO_STANDS_FILE, columns=["apt", "stand_id", "lat", "lon"])
    sm = pd.read_parquet(paths.GEO_STAND_MATCHES_FILE, columns=["apt", "stand_mvt", "stand_id"])
    sm = sm.merge(st, on=["apt", "stand_id"], how="left")
    sm = sm.assign(apt=sm["apt"].astype(str), stand_mvt=sm["stand_mvt"].astype(str))
    key = pd.DataFrame({"apt": mv["apt"].astype(str), "stand_mvt": mv["STAND_mvt"].astype(str)}, index=mv.index)
    key.index.name = "mvt_id"
    return key.reset_index().merge(sm, on=["apt", "stand_mvt"], how="left").set_index("mvt_id")[["apt", "lat", "lon"]]


def _points(day: str, icaos: set) -> pd.DataFrame:
    """Ground points of the linked aircraft: the day's file plus the last 4 h of the previous day's file."""
    f = paths.ADSB_POINTS / f"{day}.parquet"
    frames = [pd.read_parquet(f)]
    prev = paths.ADSB_POINTS / f"{(pd.Timestamp(day) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')}.parquet"
    if prev.exists():
        p = pd.read_parquet(prev)
        t0 = pd.Timestamp(day, tz="UTC").timestamp() - LOOKBACK_S
        frames.insert(0, p[p["t"] >= t0])
    pts = pd.concat(frames, ignore_index=True)
    pts = pts[pts["icao"].isin(icaos)]
    pts = pts[pts["on_ground"].to_numpy(bool) | (pts["gs_kt"].to_numpy(dtype="float64") < 40)]
    pts = pts.assign(apt=pts["apt"].astype(str)).sort_values(["icao", "apt", "t"])
    return pts.drop_duplicates(["icao", "apt", "t"])


# ---------------------------------------------------------------------------------------------- stops
def _stops(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of points within STOP_R_M of the run's first point lasting >= STOP_MIN_S: (i_first, i_last)."""
    out = []
    n = len(t)
    i = 0
    xl, yl, tl = x.tolist(), y.tolist(), t.tolist()
    r2 = STOP_R_M * STOP_R_M
    while i < n:
        j = i
        xi, yi = xl[i], yl[i]
        while j + 1 < n and (xl[j + 1] - xi) ** 2 + (yl[j + 1] - yi) ** 2 <= r2:
            j += 1
        if tl[j] - tl[i] >= STOP_MIN_S:
            out.append((i, j))
        i = j + 1
    return out


def stop_table(seg_t, seg_x, seg_y, g, own_xy) -> pd.DataFrame:
    """One row per stop of a ground segment: start, end, duration, centroid, distances and class."""
    st = _stops(seg_t, seg_x, seg_y)
    if not st:
        return pd.DataFrame(columns=["b", "e", "dur", "x", "y", "d_end", "d_cl", "d_stand", "d_hold", "d_own", "cls"])
    b = np.array([seg_t[i] for i, _ in st])
    e = np.array([seg_t[j] for _, j in st])
    cx = np.array([seg_x[i:j + 1].mean() for i, j in st])
    cy = np.array([seg_y[i:j + 1].mean() for i, j in st])
    p = np.column_stack([cx, cy])
    d = _distances(g, p)
    d_own = np.hypot(cx - own_xy[0], cy - own_xy[1]) if own_xy is not None else np.full(len(p), np.inf)
    dur = e - b
    rwy = (d["d_end"] <= RWY_END_M) | (d["d_cl"] <= RWY_CL_M) | (d["d_hold"] <= HOLD_M)
    stand = (d_own <= OWN_STAND_M) | ((d["d_stand"] <= ANY_STAND_M) & (d["d_end"] > STAND_RWY_EXCL_M) & (d["d_cl"] > 150))
    cls = np.where(stand, "stand", np.where(rwy, "rwy", np.where(dur >= LONG_S, "remote", "hold")))
    return pd.DataFrame({"b": b, "e": e, "dur": dur, "x": cx, "y": cy, **d, "d_own": d_own, "cls": cls})


def features_one(tk: float, s: pd.DataFrame, seg_t0: float) -> dict:
    """Features of one departure from its stop table (times in seconds before the ADS-B takeoff tk)."""
    o = {"sc_n_stops": len(s)}
    stand = s[s["cls"] == "stand"]
    since = float(stand["e"].max()) if len(stand) else seg_t0
    o["sc_park_end"] = tk - float(stand["e"].max()) if len(stand) else np.nan
    o["sc_park_dur"] = float(stand.loc[stand["e"].idxmax(), "dur"]) if len(stand) else np.nan
    o["sc_park_d_stand"] = float(stand.loc[stand["e"].idxmax(), "d_stand"]) if len(stand) else np.nan
    after = s[(s["b"] >= since) & (s["cls"] != "stand")]
    for c in ("rwy", "remote", "hold"):
        a = after[after["cls"] == c]
        o[f"sc_n_{c}"] = int(len(a))
        o[f"sc_s_{c}"] = float(a["dur"].sum())
    r = after[after["cls"] == "rwy"]
    o["sc_rwy_last_end"] = tk - float(r["e"].max()) if len(r) else np.nan
    o["sc_rwy_first_start"] = tk - float(r["b"].min()) if len(r) else np.nan
    o["sc_rwy_max"] = float(r["dur"].max()) if len(r) else 0.0
    # the de-icing-like stop: the longest remote stop (>= LONG_S) after the parked position, or a runway-area stop
    # >= LONG_S (pads at the runway heads) when there is no remote one
    rem = after[after["cls"] == "remote"]
    if not len(rem):
        rem = r[r["dur"] >= LONG_S]
    if len(rem):
        k = rem["dur"].idxmax()
        o["sc_deice_dur"] = float(rem.loc[k, "dur"])
        o["sc_deice_start"] = tk - float(rem.loc[k, "b"])
        o["sc_deice_end"] = tk - float(rem.loc[k, "e"])
        o["sc_deice_d_end"] = float(rem.loc[k, "d_end"])
        o["sc_deice_is_rwy"] = int(after.loc[k, "cls"] == "rwy")
    else:
        o["sc_deice_dur"] = 0.0
    # the class of the last long stop anywhere (what the unclassified adsb_stop_end_300 was), and of the first stop
    lng = s[s["dur"] >= LONG_S]
    if len(lng):
        k = lng["e"].idxmax()
        o["sc_last_long_cls"] = s.loc[k, "cls"]
        o["sc_last_long_d_stand"] = float(s.loc[k, "d_stand"])
        o["sc_last_long_d_end"] = float(s.loc[k, "d_end"])
    o["sc_first_cls"] = s["cls"].iloc[0] if len(s) and s["b"].iloc[0] - seg_t0 < 60 else "moving"
    return o


def compute_day(day: str, with_stops: bool = False):
    """Features of every linked departure of `day`: DataFrame indexed by mvt_id (and the stop tables if asked)."""
    G = geometry()
    lk = pd.read_parquet(paths.CACHE / "adsb" / f"links_{day}.parquet")
    lk = lk.assign(apt=lk["apt"].astype(str))
    pts = _points(day, set(lk["icao"].astype(str)))
    sxy = _stand_xy()
    by = {k: (g["t"].to_numpy(), g["lat"].to_numpy(), g["lon"].to_numpy())
          for k, g in pts.groupby(["icao", "apt"], sort=False)}
    rows, tabs = {}, {}
    for mvt_id, r in lk.iterrows():
        apt = r["apt"]
        arr = by.get((r["icao"], apt))
        if arr is None or apt not in G:
            continue
        g = G[apt]
        tk = float(r["t_takeoff"])
        t, la, lo = arr
        m = (t < tk) & (t >= tk - LOOKBACK_S)
        if not m.any():
            continue
        t, la, lo = t[m], la[m], lo[m]
        cut = np.nonzero(np.diff(t) > SEG_GAP_S)[0]
        if len(cut):
            t, la, lo = t[cut[-1] + 1:], la[cut[-1] + 1:], lo[cut[-1] + 1:]
        p = g["xy"](la, lo)
        own = None
        if mvt_id in sxy.index:
            sl, so = sxy.at[mvt_id, "lat"], sxy.at[mvt_id, "lon"]
            if np.isfinite(sl) and np.isfinite(so):
                own = g["xy"]([sl], [so])[0]
        s = stop_table(t, p[:, 0], p[:, 1], g, own)
        o = features_one(tk, s, float(t[0]))
        off = tk - float(r["t_mvt"])
        o["sc_offset"] = off
        rows[mvt_id] = o
        if with_stops:
            tabs[mvt_id] = s.assign(tk=tk, seg_t0=float(t[0]))
    f = pd.DataFrame.from_dict(rows, orient="index")
    f.index.name = "mvt_id"
    return (f, tabs) if with_stops else f


def derive_from_stops(stops: pd.DataFrame) -> pd.DataFrame:
    """The last long stop (>= LONG_S) of each departure read both ways, from the saved stop tables.

    ll_excl : seconds before the ADS-B takeoff at which the stop ended (off-block at its end: a stand / tow / remote
              parking position);
    ll_incl : seconds before the takeoff of the end of the previous stop >= 120 s, or of the first sighting (the stop
              is part of the taxi: queue, hold, de-icing);
    ll_cls, ll_dur and the stop's distances (runway end, centreline, any OSM stand, own stand, runway holding point).
    """
    s = stops.sort_values(["mvt_id", "b"]).reset_index(drop=True)
    lng = s[s["dur"] >= LONG_S]
    last = lng.groupby("mvt_id").tail(1).set_index("mvt_id")
    o = pd.DataFrame(index=last.index)
    o["ll_excl"] = last["tk"] - last["e"]
    o["ll_dur"] = last["dur"]
    o["ll_cls"] = last["cls"]
    for c in ("d_end", "d_cl", "d_stand", "d_own", "d_hold"):
        o[f"ll_{c}"] = last[c].where(np.isfinite(last[c]))
    # the previous stop >= 120 s that ended before the long stop began
    lb = last["b"].reindex(s["mvt_id"]).to_numpy()
    prev = s[(s["e"].to_numpy() <= lb) & (s["dur"].to_numpy() >= 120)]
    pe = prev.groupby("mvt_id")["e"].max()
    start = pe.reindex(o.index).fillna(last["seg_t0"])
    o["ll_incl"] = last["tk"] - start
    o["ll_prev_is_stop"] = pe.reindex(o.index).notna().astype(int)
    return o


def compute_day_full(day: str) -> pd.DataFrame:
    """Everything the screen uses for one day: `compute_day` plus `derive_from_stops` (index mvt_id)."""
    f, tabs = compute_day(day, with_stops=True)
    tabs = {k: v for k, v in tabs.items() if len(v)}
    if tabs:
        st = pd.concat(tabs, names=["mvt_id", "k"]).reset_index()
        f = f.join(derive_from_stops(st))
    return f


def _one(day: str) -> str:
    out = OUT / f"{day}.parquet"
    f, tabs = compute_day(day, with_stops=True)
    f.to_parquet(out)
    if tabs:
        st = pd.concat({k: v for k, v in tabs.items() if len(v)}, names=["mvt_id", "k"]).reset_index()
        st["apt"] = st["mvt_id"].map(_stand_xy()["apt"])
        st.to_parquet(OUT / f"stops_{day}.parquet")
    return f"{day}: {len(f):,} rows"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    days = sorted(p.stem for p in paths.ADSB_POINTS.glob("*.parquet"))
    if "--days" in sys.argv:
        want = {a for a in sys.argv[sys.argv.index("--days") + 1:] if not a.startswith("--")}
        days = [d for d in days if d in want]
    if "--force" not in sys.argv:
        days = [d for d in days if not (OUT / f"{d}.parquet").exists()]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 1
    workers = min(workers, 2)
    if workers > 1 and len(days) > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            for msg in pool.imap_unordered(_one, days):
                print(msg, flush=True)
    else:
        for d in days:
            print(_one(d), flush=True)


if __name__ == "__main__":
    main()
