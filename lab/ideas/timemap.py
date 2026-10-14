"""Idea family: a target-free time-to-location map for the unseen part of the taxi.

For most linked departures no pushback is observed: the aircraft is first received somewhere on its way (often already
on the taxiway), so the off-block lies an unknown time before the first sighting. This module learns, from ADS-B alone,
how long it typically takes from push start to leaving a given surface location, and turns each departure's track
into off-block estimates.

1. ``extract`` (per day, every linked departure): the aircraft's ground points before its ADS-B takeoff (ground =
   on_ground | gs < 40 kt, 4 h lookback, the previous day's file included), cut after the last > 30 min gap and at
   the start of the last parking run (>= 300 s at gs <= 1 kt within 50 m, centred within 100 m of an OSM stand), so a
   visible taxi-in is not read as departure taxi. The points are binned into ``CELL_M`` grid cells (local metres from
   the airport reference point); per (departure, cell) the EXIT time = the last point in that cell (``lookup``
   visits). For departures with a clean observed push (``features3``: pb_found, not censored, pb_start_gap <= 60 s,
   push cluster <= 100 m from the matched stand when known) the points from the push start on give ``cal`` visits.
2. ``build_map`` (pooled over all 124 ADS-B days, target-free): elapsed = exit time - push start (push start =
   t_mvt - pb_start_est) per calibration visit; per cell the median / quartiles / count of the elapsed time at three
   levels: the departure's stand (apt, STAND_mvt), its OSM apron zone (apt, zone), the airport. A calibration
   departure looking up its own cell gets the leave-self-out median.
3. ``compute_day``: each lookup visit takes the finest level with enough support (stand >= 8 tracks, zone >= 10,
   airport >= 15), estimates the push start as exit time - median elapsed, and so a taxi estimate (seconds before the
   RECORDED takeoff) t_mvt - (exit - median). Over the track's first ``K_FIRST`` usable cells (in exit order):
   ``tmap_est_first`` (first cell), ``tmap_est_best`` (the cell with the narrowest quartile range), ``tmap_est_med``
   (median of the estimates), ``tmap_est_wmean`` (1/IQR^2-weighted mean), the elapsed time and support at the first
   cell (``tmap_elapsed_first``, ``tmap_n_first``, ``tmap_iqr_first``, ``tmap_level_first``), the narrowest IQR
   (``tmap_iqr_best``) and the number of usable cells (``tmap_n_cells``).

   Distance-corrected twins ``tmap_est_first_adj`` / ``tmap_est_wmean_adj``: when the map came from the zone or the
   airport, the estimate is shifted by (own stand -> cell distance - the calibration stands' median distance) times a
   per-airport slope (s/m) fitted within (zone, cell) groups of the calibration visits; ``tmap_dcal_first`` and
   ``tmap_down_first`` are the two distances at the first cell.
4. Arrival map (``arr`` / ``arrmap``; ``tarr_*``): arrivals (linked to ADS-B by callsign within 15 min of landing)
   give, per (arrival stand, cell) and (zone, cell), the median time from entering the cell to the recorded in-block
   (an allowed 2026 movement column). It covers the stands where departures are never seen at the stand, because the
   arrival's in-block time is recorded even where reception ends. A departure's estimate per lookup cell is
   t_mvt - (exit - remaining_arr - c_apt), where c_apt (push + engine start + direction asymmetry) is the median, over
   clean observed pushes, of departure elapsed minus arrival remaining time at the same stand and cell.

Inputs: ADS-B points (only the departing aircraft's own points before its ADS-B takeoff), links, the first-round
pushback detector output (itself ADS-B only), OSM geometry and movement columns available in 2026. No target.

    python lab/ideas/timemap.py extract [--workers 2] [--force]   # cache/lab/ideas/timemap_visits/<day>.parquet
    python lab/ideas/timemap.py map                               # cache/lab/ideas/timemap_map.parquet (+ self table)
    python lab/ideas/timemap.py arr [--workers 2] [--force]       # cache/lab/ideas/timemap_arrvisits/<day>.parquet
    python lab/ideas/timemap.py arrmap                            # cache/lab/ideas/timemap_arrmap.parquet (+ constants)
    python lab/ideas/timemap.py days [--workers 2] [--force]      # cache/lab/ideas/timemap_days/<day>.parquet
    python lab/ideas/timemap.py assemble                          # cache/lab/ideas/timemap.parquet (both years)
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lab" / "ideas"))
from prc import data, paths  # noqa: E402

KEY = "timemap"
OUT = paths.CACHE / "lab" / "ideas"
VISIT_DIR = OUT / f"{KEY}_visits"
DAY_DIR = OUT / f"{KEY}_days"
MAP_FILE = OUT / f"{KEY}_map.parquet"
SELF_FILE = OUT / f"{KEY}_self.parquet"
SLOPE_FILE = OUT / f"{KEY}_slope.parquet"
ARR_DIR = OUT / f"{KEY}_arrvisits"
ARR_MAP_FILE = OUT / f"{KEY}_arrmap.parquet"
ARR_C_FILE = OUT / f"{KEY}_arrconst.parquet"
ARR_MIN_N = {"stand": 5, "zone": 10}
ARR_LINK_S = 900.0
LOOKBACK_S = 4 * 3600.0
CUT_GAP_S = 1800.0
CELL_M = 75.0
PARK_MIN_S = 300.0
PARK_EXT_M = 50.0
PARK_STAND_M = 100.0
CAL_MAX_GAP_S = 60.0
CAL_MAX_STAND_M = 100.0
MAX_ELAPSED_S = 3600.0
MIN_N = {"stand": 8, "zone": 10, "apt": 15}
LEVELS = ("stand", "zone", "apt")
K_FIRST = 8
K = 111_320.0


# ---------------------------------------------------------------------------------------------- static lookups
@lru_cache(maxsize=1)
def _airports() -> dict:
    a = pd.read_parquet(paths.GEO_AIRPORTS_FILE, columns=["apt", "arp_lat", "arp_lon"])
    return {str(r.apt): (float(r.arp_lat), float(r.arp_lon)) for r in a.itertuples(index=False)}


@lru_cache(maxsize=1)
def _stand_trees() -> dict:
    st = pd.read_parquet(paths.GEO_STANDS_FILE, columns=["apt", "lat", "lon"])
    out = {}
    for apt, g in st.groupby(st["apt"].astype(str)):
        if apt not in _airports():
            continue
        lat0, lon0 = _airports()[apt]
        e, n = _xy(lat0, lon0, g["lat"].to_numpy(), g["lon"].to_numpy())
        out[apt] = cKDTree(np.column_stack([e, n]))
    return out


@lru_cache(maxsize=2)
def _meta(dep: bool = True) -> pd.DataFrame:
    """Per departure (or arrival; both years): apt, stand key, zone key, matched stand position (local metres)."""
    m = data.load_movements(columns=["is_dep", "apt", "STAND_mvt"])
    m = m[m["is_dep"] == dep]
    st = pd.read_parquet(paths.GEO_STANDS_FILE, columns=["apt", "stand_id", "lat", "lon", "zone"])
    sm = pd.read_parquet(paths.GEO_STAND_MATCHES_FILE, columns=["apt", "stand_mvt", "stand_id"])
    sm = sm.merge(st, on=["apt", "stand_id"], how="inner")
    sm = sm.assign(apt=sm["apt"].astype(str), stand_mvt=sm["stand_mvt"].astype(str))
    sm = sm.drop_duplicates(["apt", "stand_mvt"]).set_index(["apt", "stand_mvt"])
    apt = m["apt"].astype(str)
    stand = m["STAND_mvt"].astype("string").fillna("").astype(str)
    pos = sm.reindex(pd.MultiIndex.from_arrays([apt, stand]))
    out = pd.DataFrame({"apt": apt.to_numpy(), "stand": stand.to_numpy(), "zone": pos["zone"].to_numpy(),
                        "slat": pos["lat"].to_numpy(), "slon": pos["lon"].to_numpy()}, index=m.index)
    ref = out["apt"].map(_airports())
    lat0 = ref.map(lambda r: r[0] if isinstance(r, tuple) else np.nan).astype(float)
    lon0 = ref.map(lambda r: r[1] if isinstance(r, tuple) else np.nan).astype(float)
    out["se"], out["sn"] = _xy(lat0.to_numpy(), lon0.to_numpy(), out["slat"].to_numpy(), out["slon"].to_numpy())
    out["k_stand"] = np.where(out["stand"] != "", out["apt"] + "|" + out["stand"], None)
    out["k_zone"] = np.where(out["zone"].notna(), out["apt"] + "|z" + out["zone"].astype("Int64").astype(str), None)
    out["k_apt"] = out["apt"]
    return out


@lru_cache(maxsize=1)
def _push() -> pd.DataFrame:
    """Clean observed pushes from the first-round detector (ADS-B only): push start in epoch seconds needs t_mvt."""
    f = pd.read_parquet(paths.CACHE / "adsb" / "features3.parquet",
                        columns=["pb_found", "pb_censored", "pb_start_gap", "pb_cl_dist_stand", "pb_start_est"])
    ok = (f["pb_found"] == 1) & (f["pb_censored"] == 0) & (f["pb_start_gap"] <= CAL_MAX_GAP_S) \
        & ~(f["pb_cl_dist_stand"] > CAL_MAX_STAND_M) & f["pb_start_est"].notna()
    return f.loc[ok, ["pb_start_est"]]


def _xy(lat0, lon0, lat, lon):
    return (lon - lon0) * K * np.cos(np.radians(lat0)), (lat - lat0) * K


def _cell_centre(cell):
    cell = np.asarray(cell, dtype=np.int64)
    return (cell // 100_000 - 50_000 + 0.5) * CELL_M, (cell % 100_000 - 50_000 + 0.5) * CELL_M


def _cell_key(e, n):
    cx = np.floor(e / CELL_M).astype(np.int64) + 50_000
    cy = np.floor(n / CELL_M).astype(np.int64) + 50_000
    return cx * 100_000 + cy


# ---------------------------------------------------------------------------------------------- 1. extract
def _parking_cut(t, e, n, gs, tree) -> int:
    """Index of the first point of the last parking run (0 when there is none)."""
    still = ~(gs > 1.0)
    if not still.any():
        return 0
    d = np.diff(np.concatenate([[0], still.astype(np.int8), [0]]))
    starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1
    for a, b in zip(starts[::-1], ends[::-1]):
        if t[b] - t[a] < PARK_MIN_S:
            continue
        ext = np.hypot(e[a:b + 1] - e[a], n[a:b + 1] - n[a]).max()
        if ext > PARK_EXT_M:
            continue
        if tree is None:
            continue
        dd, _ = tree.query((float(np.median(e[a:b + 1])), float(np.median(n[a:b + 1]))), k=1)
        if dd <= PARK_STAND_M:
            return int(a)
    return 0


def _visits(t, ck):
    """Exit time (last point) and entry time (first point) per cell."""
    order = np.lexsort((t, ck))
    c, tt = ck[order], t[order]
    first = np.concatenate([[True], c[1:] != c[:-1]])
    last = np.concatenate([c[1:] != c[:-1], [True]])
    return c[first], tt[first], tt[last]


def extract_day(day: str) -> pd.DataFrame:
    from pushback import _day_points
    lk = pd.read_parquet(paths.CACHE / "adsb" / f"links_{day}.parquet")
    p = _day_points(day)
    groups = p.groupby(["icao", "apt"], sort=False).indices
    T = p["t"].to_numpy()
    LA, LO = p["lat"].to_numpy(), p["lon"].to_numpy()
    GS = p["gs_kt"].to_numpy(dtype="float64")
    push = _push()
    trees = _stand_trees()
    recs = []
    for mvt_id, icao, apt, tk, tm in zip(lk.index, lk["icao"], lk["apt"].astype(str), lk["t_takeoff"], lk["t_mvt"]):
        idx = groups.get((icao, apt))
        if idx is None or apt not in _airports():
            continue
        tt = T[idx]
        sel = idx[np.searchsorted(tt, tk - LOOKBACK_S):np.searchsorted(tt, tk)]
        if len(sel) < 2:
            continue
        keep = np.concatenate([[True], np.diff(T[sel]) > 0])
        sel = sel[keep]
        t = T[sel]
        lat0, lon0 = _airports()[apt]
        e, n = _xy(lat0, lon0, LA[sel], LO[sel])
        ck = _cell_key(e, n)
        # calibration visits: points from the observed push start on (full lookback, no cut)
        if mvt_id in push.index:
            t_push = float(tm) - float(push.at[mvt_id, "pb_start_est"])
            s = t >= t_push - 1.0
            if s.sum() >= 2:
                c, tin, tout = _visits(t[s], ck[s])
                recs.append(pd.DataFrame({"mvt_id": mvt_id, "cell": c, "t_in": tin, "t_out": tout, "cal": 1,
                                          "t_ref": t_push}))
        # lookup visits: segment after the last long gap and the last parking run
        cut = np.nonzero(np.diff(t) > CUT_GAP_S)[0]
        a0 = cut[-1] + 1 if len(cut) else 0
        a1 = a0 + _parking_cut(t[a0:], e[a0:], n[a0:], GS[sel][a0:], trees.get(apt))
        if len(t) - a1 < 1:
            continue
        c, tin, tout = _visits(t[a1:], ck[a1:])
        recs.append(pd.DataFrame({"mvt_id": mvt_id, "cell": c, "t_in": tin, "t_out": tout, "cal": 0,
                                  "t_ref": t[a1]}))
    if not recs:
        return pd.DataFrame(columns=["mvt_id", "cell", "t_in", "t_out", "cal", "t_ref"])
    out = pd.concat(recs, ignore_index=True)
    out["cal"] = out["cal"].astype(np.int8)
    return out


def _one_extract(day: str) -> str:
    VISIT_DIR.mkdir(parents=True, exist_ok=True)
    v = extract_day(day)
    v.to_parquet(VISIT_DIR / f"{day}.parquet")
    nc = v.loc[v["cal"] == 1, "mvt_id"].nunique()
    nl = v.loc[v["cal"] == 0, "mvt_id"].nunique()
    return f"{day}: {len(v):,} visits, {nl:,} lookup tracks, {nc:,} calibration tracks"


def _pool(fn, days, workers):
    if workers > 1 and len(days) > 1:
        import multiprocessing as mp
        with mp.Pool(min(workers, 2)) as pool:
            for msg in pool.imap_unordered(fn, days):
                print(msg, flush=True)
    else:
        for d in days:
            print(fn(d), flush=True)


def all_days() -> list[str]:
    return sorted(p.stem for p in paths.ADSB_POINTS.glob("*.parquet"))


# ---------------------------------------------------------------------------------------------- 2. map
def _stats(cal: pd.DataFrame, key_col: str):
    """Per (key, cell): count, median / quartiles of the elapsed time, median distance calibration stand -> cell; per
    calibration record the leave-self-out median."""
    d = cal[[key_col, "cell", "mvt_id", "elapsed", "dst"]].dropna(subset=[key_col])
    d = d.sort_values([key_col, "cell", "elapsed"], kind="stable").reset_index(drop=True)
    grp = d.groupby([key_col, "cell"], sort=False)
    gi = grp.ngroup().to_numpy()
    sizes = np.bincount(gi)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    rank = np.arange(len(d)) - starts[gi]
    vals = d["elapsed"].to_numpy()

    def q(frac):  # linear-interpolated quantile of each sorted group (pandas' default)
        pos = frac * (sizes - 1)
        lo = np.floor(pos).astype(np.int64)
        hi = np.minimum(lo + 1, sizes - 1)
        return vals[starts + lo] + (pos - lo) * (vals[starts + hi] - vals[starts + lo])

    keys = d.loc[starts, [key_col, "cell"]].reset_index(drop=True)
    st = keys.assign(n=sizes, med=q(0.5), q25=q(0.25), q75=q(0.75),
                     dmed=grp["dst"].median().to_numpy())
    m = sizes[gi] - 1
    loo = np.full(len(d), np.nan)

    def at(g, r, pos):
        pos = np.where(pos >= r, pos + 1, pos)
        return vals[starts[g] + pos]

    odd = m % 2 == 1
    if odd.any():
        loo[odd] = at(gi[odd], rank[odd], (m[odd] - 1) // 2)
    ev = (m % 2 == 0) & (m >= 2)
    if ev.any():
        loo[ev] = 0.5 * (at(gi[ev], rank[ev], m[ev] // 2 - 1) + at(gi[ev], rank[ev], m[ev] // 2))
    self_ = pd.DataFrame({"mvt_id": d["mvt_id"].to_numpy(), "cell": d["cell"].to_numpy(), "loo": loo})
    return st, self_


def _slopes(cal: pd.DataFrame) -> pd.Series:
    """Per airport, seconds of elapsed time per metre of stand -> cell distance, within (zone, cell) groups (pooled
    demeaned least squares over calibration visits; target-free)."""
    d = cal.dropna(subset=["k_zone", "dst"])
    d = d[d["elapsed"] <= 1200]
    g = d.groupby(["k_zone", "cell"])
    de = d["elapsed"] - g["elapsed"].transform("mean")
    dd = d["dst"] - g["dst"].transform("mean")
    num = (de * dd).groupby(d["k_apt"]).sum()
    den = (dd * dd).groupby(d["k_apt"]).sum()
    return (num / den).clip(0.0, 0.5).rename("slope")


def build_map() -> None:
    files = sorted(VISIT_DIR.glob("*.parquet"))
    cal = pd.concat([pd.read_parquet(f, filters=[("cal", "==", 1)]) for f in files], ignore_index=True)
    cal = cal.drop_duplicates(["mvt_id", "cell"], keep="last")
    cal["elapsed"] = cal["t_out"] - cal["t_ref"]
    cal = cal[(cal["elapsed"] >= 0) & (cal["elapsed"] <= MAX_ELAPSED_S)]
    meta = _meta()
    cal = cal.join(meta[["k_stand", "k_zone", "k_apt", "se", "sn"]], on="mvt_id")
    ce, cn = _cell_centre(cal["cell"].to_numpy())
    cal["dst"] = np.hypot(ce - cal["se"].to_numpy(), cn - cal["sn"].to_numpy())
    sl = _slopes(cal)
    sl.to_frame().to_parquet(SLOPE_FILE)
    print("slopes s/m:", sl.round(3).to_dict())
    maps, selfs = [], []
    for lev in LEVELS:
        st, sf = _stats(cal, f"k_{lev}")
        maps.append(st.rename(columns={f"k_{lev}": "key"}).assign(level=lev))
        selfs.append(sf.assign(level=lev))
    mp_ = pd.concat(maps, ignore_index=True)
    mp_.to_parquet(MAP_FILE)
    pd.concat(selfs, ignore_index=True).to_parquet(SELF_FILE)
    print(f"map: {cal['mvt_id'].nunique():,} calibration tracks, {len(cal):,} visits;",
          mp_.groupby("level")["n"].agg(["size", "median"]).to_dict())


# ---------------------------------------------------------------------------------------------- 2b. arrival map
@lru_cache(maxsize=1)
def _arrivals() -> pd.DataFrame:
    m = data.load_movements(columns=["is_dep", "apt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "CALLSIGN_flt"])
    m = m[~m["is_dep"]]
    ep = pd.Timestamp("1970-01-01", tz="UTC")
    return pd.DataFrame({"apt": m["apt"].astype(str),
                         "t_land": (m["MVT_TIME_UTC_mvt"] - ep).dt.total_seconds(),
                         "t_block": (m["BLOCK_TIME_UTC_mvt"] - ep).dt.total_seconds(),
                         "cs": m["CALLSIGN_flt"].astype("string").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True),
                         "day": m["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d")}, index=m.index)


def _arr_points(day: str) -> pd.DataFrame:
    """Ground points (on_ground | gs < 40 kt) of `day`, plus the first hour of the next day, with callsigns."""
    cols = ["icao", "t", "lat", "lon", "on_ground", "gs_kt", "callsign", "apt"]
    p = pd.read_parquet(paths.ADSB_POINTS / f"{day}.parquet", columns=cols)
    nxt = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    fn = paths.ADSB_POINTS / f"{nxt}.parquet"
    if fn.exists():  # taxi-ins that end after midnight
        q = pd.read_parquet(fn, columns=cols)
        p = pd.concat([p, q[q["t"] < pd.Timestamp(nxt, tz="UTC").timestamp() + 3600]], ignore_index=True)
    p = p.assign(apt=p["apt"].astype(str))
    return p.sort_values(["icao", "apt", "t"], kind="stable").reset_index(drop=True)


def extract_arr_day(day: str) -> pd.DataFrame:
    """Arrivals landing on `day`, linked to an ADS-B aircraft by callsign and time: per (arrival, cell) the time from
    entering the cell to the recorded in-block (an allowed movement column), from touchdown to in-block only."""
    arr = _arrivals()
    arr = arr[(arr["day"] == day) & arr["cs"].notna() & arr["t_block"].notna()]
    p = _arr_points(day)
    cs = p[p["callsign"].notna()]
    cs = cs.assign(cs=cs["callsign"].astype("string").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True),
                   b=(cs["t"] // 600).astype(np.int64))
    red = cs.groupby(["cs", "apt", "icao", "b"])["t"].agg(["min", "max"]).reset_index()
    cand = arr.reset_index()[["mvt_id", "apt", "cs", "t_land"]].merge(red, on=["cs", "apt"], how="inner")
    cand["dist"] = np.maximum(0.0, np.maximum(cand["min"] - cand["t_land"], cand["t_land"] - cand["max"]))
    cand = cand[cand["dist"] <= ARR_LINK_S].sort_values(["mvt_id", "dist"])
    best = cand.drop_duplicates("mvt_id").set_index("mvt_id")["icao"]
    g_ = p[p["on_ground"].to_numpy(bool) | (p["gs_kt"].to_numpy(dtype="float64") < 40)].reset_index(drop=True)
    groups = g_.groupby(["icao", "apt"], sort=False).indices
    T = g_["t"].to_numpy()
    LA, LO = g_["lat"].to_numpy(), g_["lon"].to_numpy()
    A = arr.reindex(best.index)
    recs = []
    for mvt_id, icao, apt, tl, tb in zip(best.index, best.to_numpy(), A["apt"], A["t_land"], A["t_block"]):
        idx = groups.get((icao, apt))
        if idx is None or apt not in _airports():
            continue
        tt = T[idx]
        sel = idx[np.searchsorted(tt, tl - 60.0):np.searchsorted(tt, tb + 30.0)]
        if len(sel) < 2:
            continue
        keep = np.concatenate([[True], np.diff(T[sel]) > 0])
        sel = sel[keep]
        lat0, lon0 = _airports()[apt]
        e, n = _xy(lat0, lon0, LA[sel], LO[sel])
        c, tin, _ = _visits(T[sel], _cell_key(e, n))
        rem = tb - tin
        ok = (rem >= 0) & (rem <= MAX_ELAPSED_S)
        if ok.any():
            recs.append(pd.DataFrame({"mvt_id": mvt_id, "cell": c[ok], "rem": rem[ok]}))
    if not recs:
        return pd.DataFrame(columns=["mvt_id", "cell", "rem"])
    return pd.concat(recs, ignore_index=True)


def _one_arr(day: str) -> str:
    ARR_DIR.mkdir(parents=True, exist_ok=True)
    v = extract_arr_day(day)
    v.to_parquet(ARR_DIR / f"{day}.parquet")
    n_arr = int((_arrivals()["day"] == day).sum())
    return f"{day}: {v['mvt_id'].nunique():,} of {n_arr:,} arrivals linked, {len(v):,} visits"


def build_arr_map() -> None:
    """Per (arrival stand, cell) and (zone, cell): median time from entering the cell to in-block, over every linked
    arrival of the 124 ADS-B days; plus a per-airport constant (push + engine start + direction asymmetry) = median
    over clean observed pushes of (departure elapsed at a cell - arrival remaining time there, same stand).
    Target-free."""
    v = pd.concat([pd.read_parquet(f) for f in sorted(ARR_DIR.glob("*.parquet"))], ignore_index=True)
    v = v.drop_duplicates(["mvt_id", "cell"])
    v = v.join(_meta(False)[["k_stand", "k_zone", "k_apt", "se", "sn"]], on="mvt_id")
    ce, cn = _cell_centre(v["cell"].to_numpy())
    v["dst"] = np.hypot(ce - v["se"].to_numpy(), cn - v["sn"].to_numpy())
    v = v.rename(columns={"rem": "elapsed"})
    maps = []
    for lev in ("stand", "zone"):
        st, _ = _stats(v, f"k_{lev}")
        maps.append(st.rename(columns={f"k_{lev}": "key"}).assign(level=lev))
    am = pd.concat(maps, ignore_index=True)
    am.to_parquet(ARR_MAP_FILE)
    cal = pd.concat([pd.read_parquet(f, filters=[("cal", "==", 1)]) for f in sorted(VISIT_DIR.glob("*.parquet"))],
                    ignore_index=True).drop_duplicates(["mvt_id", "cell"], keep="last")
    cal["elapsed"] = cal["t_out"] - cal["t_ref"]
    cal = cal[(cal["elapsed"] >= 0) & (cal["elapsed"] <= MAX_ELAPSED_S)]
    cal = cal.join(_meta(True)[["k_stand", "k_apt"]], on="mvt_id")
    s_ = am[(am["level"] == "stand") & (am["n"] >= ARR_MIN_N["stand"])][["key", "cell", "med"]]
    j = cal.merge(s_.rename(columns={"key": "k_stand"}), on=["k_stand", "cell"], how="inner")
    c = (j["elapsed"] - j["med"]).groupby(j["k_apt"]).median().rename("c")
    c.to_frame().to_parquet(ARR_C_FILE)
    print(f"arrival map: {v['mvt_id'].nunique():,} arrivals, {len(v):,} visits;",
          am.groupby("level")["n"].agg(["size", "median"]).to_dict(), "constants:", c.round(0).to_dict())


@lru_cache(maxsize=1)
def _arr_map():
    if not ARR_MAP_FILE.exists():
        return None
    return pd.read_parquet(ARR_MAP_FILE), pd.read_parquet(ARR_C_FILE)["c"]


def _arr_features(v: pd.DataFrame) -> pd.DataFrame:
    """Arrival-map estimates for lookup visits v (mvt_id, cell, t_out, t_mvt, k_stand, k_zone, k_apt, down, slope)."""
    am = _arr_map()
    if am is None:
        return pd.DataFrame(index=pd.Index([], name="mvt_id"))
    am, cst = am
    chosen = None
    for li, lev in enumerate(("stand", "zone")):
        m = am[am["level"] == lev].drop(columns="level").rename(columns={"key": f"k_{lev}"})
        x = v[["cell", f"k_{lev}"]].merge(m, on=[f"k_{lev}", "cell"], how="left")
        ok = ((x["n"] >= ARR_MIN_N[lev]) & x["med"].notna()).to_numpy()
        rem = x["med"].to_numpy()
        if lev == "zone":  # own stand farther from the cell than the zone's arrival stands -> longer
            rem = rem + np.clip((v["down"].to_numpy() - x["dmed"].to_numpy()) * v["slope"].to_numpy(), -600, 600)
        cur = pd.DataFrame({"rem": np.where(ok, rem, np.nan), "an": np.where(ok, x["n"].to_numpy(), np.nan),
                            "aiqr": np.where(ok, (x["q75"] - x["q25"]).to_numpy(), np.nan),
                            "alevel": np.where(ok, li, np.nan)}, index=v.index)
        chosen = cur if chosen is None else chosen.where(chosen["rem"].notna(), cur)
    w = v[["mvt_id", "t_out", "t_mvt", "k_apt"]].join(chosen)
    w = w[w["rem"].notna()].sort_values(["mvt_id", "t_out"], kind="stable")
    w["est"] = w["t_mvt"] - (w["t_out"] - w["rem"] - w["k_apt"].map(cst).astype(float).fillna(0.0))
    w["k"] = w.groupby("mvt_id").cumcount()
    w = w[w["k"] < K_FIRST]
    if w.empty:
        return pd.DataFrame(index=pd.Index([], name="mvt_id"))
    w["w"] = 1.0 / w["aiqr"].clip(lower=10.0) ** 2
    g = w.groupby("mvt_id", sort=False)
    first = g.head(1).set_index("mvt_id")
    return pd.DataFrame({
        "tarr_est_first": first["est"],
        "tarr_rem_first": first["rem"],
        "tarr_n_first": first["an"],
        "tarr_iqr_first": first["aiqr"],
        "tarr_level_first": first["alevel"],
        "tarr_est_med": g["est"].median(),
        "tarr_est_wmean": (w["est"] * w["w"]).groupby(w["mvt_id"]).sum() / g["w"].sum(),
        "tarr_n_cells": g.size(),
    })


# ---------------------------------------------------------------------------------------------- 3. per day
@lru_cache(maxsize=1)
def _map():
    mp_ = pd.read_parquet(MAP_FILE)
    sf = pd.read_parquet(SELF_FILE)
    sl = pd.read_parquet(SLOPE_FILE)["slope"] if SLOPE_FILE.exists() else pd.Series(dtype=float)
    return mp_, sf, sl


def compute_day(day: str) -> pd.DataFrame:
    """Time-map features of every linked departure of `day` (index mvt_id); estimates in seconds before the recorded
    takeoff."""
    v = pd.read_parquet(VISIT_DIR / f"{day}.parquet")
    v = v[v["cal"] == 0].drop(columns=["cal"])
    lk = pd.read_parquet(paths.CACHE / "adsb" / f"links_{day}.parquet")
    v = v[v["mvt_id"].isin(lk.index)]
    if v.empty:
        return pd.DataFrame(index=pd.Index([], name="mvt_id"))
    mp_, sf, sl = _map()
    meta = _meta()
    v = v.join(meta[["k_stand", "k_zone", "k_apt", "se", "sn"]], on="mvt_id")
    ce, cn = _cell_centre(v["cell"].to_numpy())
    v["down"] = np.hypot(ce - v["se"].to_numpy(), cn - v["sn"].to_numpy())
    v["slope"] = v["k_apt"].map(sl).astype(float)
    v["t_mvt"] = lk["t_mvt"].reindex(v["mvt_id"]).to_numpy()
    sf_day = sf[sf["mvt_id"].isin(set(v["mvt_id"]))]
    chosen = None
    for li, lev in enumerate(LEVELS):
        m = mp_[mp_["level"] == lev].drop(columns="level").rename(columns={"key": f"k_{lev}"})
        x = v[["mvt_id", "cell", f"k_{lev}"]].merge(m, on=[f"k_{lev}", "cell"], how="left")
        s = sf_day[sf_day["level"] == lev][["mvt_id", "cell", "loo"]]
        x = x.merge(s, on=["mvt_id", "cell"], how="left")
        # a calibration track found in this (key, cell): leave itself out
        is_self = x.set_index(["mvt_id", "cell"]).index.isin(s.set_index(["mvt_id", "cell"]).index)
        x.loc[is_self, "med"] = x.loc[is_self, "loo"]
        x.loc[is_self, "n"] = x.loc[is_self, "n"] - 1
        ok = (x["n"] >= MIN_N[lev]) & x["med"].notna()
        cur = pd.DataFrame({"med": x["med"].where(ok).to_numpy(), "n": x["n"].where(ok).to_numpy(),
                            "iqr": (x["q75"] - x["q25"]).where(ok).to_numpy(),
                            "dmed": x["dmed"].where(ok).to_numpy(),
                            "level": np.where(ok, li, np.nan)}, index=v.index)
        chosen = cur if chosen is None else chosen.where(chosen["med"].notna(), cur)
    v = v.join(chosen)
    arr_f = _arr_features(v)
    v = v[v["med"].notna()].sort_values(["mvt_id", "t_out"], kind="stable")
    v["est"] = v["t_mvt"] - (v["t_out"] - v["med"])
    # own stand farther from the cell than the calibration stands -> more elapsed time (per-airport slope)
    v["adj"] = ((v["down"] - v["dmed"]) * v["slope"]).clip(-600, 600)
    v["est_adj"] = v["est"] + v["adj"].fillna(0.0)
    v["k"] = v.groupby("mvt_id").cumcount()
    v = v[v["k"] < K_FIRST]
    if v.empty:
        return arr_f.rename_axis("mvt_id")
    v["iqr_c"] = v["iqr"].clip(lower=10.0)
    v["w"] = 1.0 / v["iqr_c"] ** 2
    g = v.groupby("mvt_id", sort=False)
    first = g.head(1).set_index("mvt_id")
    best = v.loc[v.groupby("mvt_id", sort=False)["iqr_c"].idxmin()].set_index("mvt_id")
    out = pd.DataFrame({
        "tmap_est_first": first["est"],
        "tmap_elapsed_first": first["med"],
        "tmap_n_first": first["n"],
        "tmap_iqr_first": first["iqr"],
        "tmap_level_first": first["level"],
        "tmap_est_best": best["est"],
        "tmap_iqr_best": best["iqr"],
        "tmap_est_med": g["est"].median(),
        "tmap_est_wmean": (v["est"] * v["w"]).groupby(v["mvt_id"]).sum() / g["w"].sum(),
        "tmap_n_cells": g.size(),
        "tmap_est_first_adj": first["est_adj"],
        "tmap_est_wmean_adj": (v["est_adj"] * v["w"]).groupby(v["mvt_id"]).sum() / g["w"].sum(),
        "tmap_dcal_first": first["dmed"],
        "tmap_down_first": first["down"],
    })
    out = out.join(arr_f, how="outer")
    out.index.name = "mvt_id"
    return out


def _one_day(day: str) -> str:
    DAY_DIR.mkdir(parents=True, exist_ok=True)
    f = compute_day(day)
    f.to_parquet(DAY_DIR / f"{day}.parquet")
    return f"{day}: {len(f):,} rows"


def assemble() -> pd.DataFrame:
    f = pd.concat([pd.read_parquet(p) for p in sorted(DAY_DIR.glob("*.parquet"))])
    f = f[~f.index.duplicated(keep="last")]
    f.to_parquet(OUT / f"{KEY}.parquet")
    print(f"assembled {len(f):,} rows -> {OUT / (KEY + '.parquet')}")
    return f


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 1
    force = "--force" in sys.argv
    if cmd == "extract":
        todo = [d for d in all_days() if force or not (VISIT_DIR / f"{d}.parquet").exists()]
        _pool(_one_extract, todo, w)
    elif cmd == "map":
        build_map()
    elif cmd == "arr":
        todo = [d for d in all_days() if force or not (ARR_DIR / f"{d}.parquet").exists()]
        _pool(_one_arr, todo, w)
    elif cmd == "arrmap":
        build_arr_map()
    elif cmd == "arr1":
        x = extract_arr_day(sys.argv[2])
        print(x.describe().T.to_string(), x["mvt_id"].nunique())
    elif cmd == "days":
        todo = [d for d in all_days() if force or not (DAY_DIR / f"{d}.parquet").exists()]
        _pool(_one_day, todo, w)
    elif cmd == "assemble":
        assemble()
    elif cmd == "extract1":
        print(extract_day(sys.argv[2]).describe().T.to_string())
    else:
        print(__doc__)
