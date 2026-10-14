"""Robust pushback / engine-start detector from adsb.lol surface tracks (idea family: pushback and engine start).

The observation this rests on: on the ground most transponders report the aircraft's *heading* in the track field,
so while it is pushed back the bearing of its displacement is ~180 deg off the reported track (median 158 deg in a
Jan 2025 sample), and while it taxis under its own power the two agree within ~1 deg. A pushback is therefore a run
of "reverse" steps (displacement >= 2 m against the reported track), which position jitter at the stand does not
produce consistently and a tow or an earlier taxi-in (forward) does not produce at all.

Per linked departure (ground points of its aircraft in the 4 h before the ADS-B takeoff, the previous day's file
included across midnight), all times in seconds before the RECORDED takeoff (= taxi estimates; the ADS-B takeoff
offset is already subtracted):

* push episode: maximal run of reverse steps (gaps <= 120 s, not broken by a forward step >= 4 m), kept when it
  has >= 2 reverse steps and >= 10 m net displacement; the last such episode before takeoff is the pushback.
  ``pb_start``: interpolated start of the push (first reverse point minus step/speed, bounded by the previous point),
  ``pb_start_gap``: seconds between the last pre-push point and the first reverse point (timing uncertainty),
  ``pb_censored``: the push was already under way at the first ground point of the track (off-block >= pb_start),
  ``pb_end``: last reverse point, ``pb_push_dur``, ``pb_push_m`` (net metres), ``pb_n_rev``;
  ``pb_go``: first forward step (>= 3 m) after the push (taxi under own power), ``pb_es_dur`` = pb_end - pb_go
  (the engine-start / tug-disconnect pause), ``pb_taxi8``: first point >= 8 kt after the push.
* stand cluster (anchor): with a push, the contiguous run of points within 25 m of the pre-push point; without one,
  the last stationary run (>= 60 s within 20 m) whose centre lies within 80 m of the matched OSM stand, else the
  earliest stationary run of >= 300 s. ``pb_anchor_src`` (push / stand / long / none), ``pb_cl_n``, ``pb_cl_dwell``,
  ``pb_cl_dist_stand`` (cluster centre to the matched stand, the confidence check), ``pb_leave``: the last point
  within 20 m of the cluster centre after which the track never comes back within 20 m (sustained departure).

Only ADS-B points before the ADS-B takeoff, the links, the geometry tables and movement columns are read; no target.

    python lab/ideas/pushback.py --holdout [--workers 2]     # -> cache/lab/ideas/pushback.parquet
    python lab/ideas/pushback.py --days 2026-01-05 ...       # -> cache/lab/ideas/pushback_2026.parquet (appended)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from prc import paths  # noqa: E402

OUT_DIR = paths.CACHE / "lab" / "ideas"
DAY_DIR = OUT_DIR / "pushback_days"
LOOKBACK_S = 4 * 3600.0
K = 111_320.0

REV_MIN_STEP_M = 2.0
REV_MIN_DIFF = 110.0
FWD_MAX_DIFF = 70.0
EPISODE_GAP_S = 120.0
PUSH_MIN_REV = 2
PUSH_MIN_NET_M = 10.0
CLUSTER_R_M = 25.0
STOP_R_M = 20.0
LEAVE_R_M = 20.0


def _stand_positions() -> pd.DataFrame:
    st = pd.read_parquet(paths.GEO_STANDS_FILE, columns=["apt", "stand_id", "lat", "lon"])
    sm = pd.read_parquet(paths.GEO_STAND_MATCHES_FILE, columns=["apt", "stand_mvt", "stand_id"])
    sm = sm.merge(st, on=["apt", "stand_id"], how="inner")
    sm = sm.assign(apt=sm["apt"].astype(str), stand_mvt=sm["stand_mvt"].astype(str))
    return sm.drop_duplicates(["apt", "stand_mvt"]).set_index(["apt", "stand_mvt"])[["lat", "lon"]]


_MVT = None


def _movements() -> pd.DataFrame:
    global _MVT
    if _MVT is None:
        from prc import data
        m = data.load_movements(columns=["is_dep", "apt", "STAND_mvt"])
        m = m[m["is_dep"]]
        _MVT = pd.DataFrame({"apt": m["apt"].astype(str), "stand_mvt": m["STAND_mvt"].astype(str)}, index=m.index)
    return _MVT


def _day_points(day: str) -> pd.DataFrame:
    cols = ["icao", "t", "lat", "lon", "on_ground", "gs_kt", "track", "apt"]
    p = pd.read_parquet(paths.ADSB_POINTS / f"{day}.parquet", columns=cols)
    prev = (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    fp = paths.ADSB_POINTS / f"{prev}.parquet"
    if fp.exists():  # departures shortly after midnight: their ground history sits in the previous file
        q = pd.read_parquet(fp, columns=cols)
        start = pd.Timestamp(day, tz="UTC").timestamp()
        p = pd.concat([q[q["t"] >= start - LOOKBACK_S], p], ignore_index=True)
    p = p[p["on_ground"].to_numpy(bool) | (p["gs_kt"].to_numpy(dtype="float64") < 40)]
    p = p.assign(apt=p["apt"].astype(str)).sort_values(["icao", "apt", "t"], kind="stable").reset_index(drop=True)
    return p


def _metres_xy(lat0, lon0, lat, lon):
    return (lon - lon0) * K * np.cos(np.radians(lat0)), (lat - lat0) * K


def _stationary_runs(t, x, y, r=STOP_R_M, min_dwell=60.0):
    """Maximal runs of points within r of the run's first point; (i_first, i_last) with dwell >= min_dwell."""
    out = []
    i, n = 0, len(t)
    while i < n:
        j = i
        while j + 1 < n and np.hypot(x[j + 1] - x[i], y[j + 1] - y[i]) <= r:
            j += 1
        if t[j] - t[i] >= min_dwell:
            out.append((i, j))
        i = j + 1
    return out


def detect(t, lat, lon, gs, trk, tk, slat=np.nan, slon=np.nan) -> dict:
    """The pushback features of one aircraft's ground track (arrays sorted by t, all before tk)."""
    res: dict = {}
    n = len(t)
    if n < 2:
        return {"pb_found": 0, "pb_anchor_src": "none", "pb_n_pts": n}
    lat0, lon0 = lat[-1], lon[-1]
    x, y = _metres_xy(lat0, lon0, lat, lon)
    dx, dy = np.diff(x), np.diff(y)
    dt = np.diff(t)
    step = np.hypot(dx, dy)
    brg = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
    tr = trk[1:].astype("float64").copy()
    miss = ~np.isfinite(tr)
    tr[miss] = trk[:-1][miss]
    diff = np.abs((brg - tr + 180.0) % 360.0 - 180.0)
    known = np.isfinite(diff)
    speed = step / np.maximum(dt, 0.5)
    rev = known & (step >= REV_MIN_STEP_M) & (diff > REV_MIN_DIFF) & ~(gs[1:] >= 15) & (speed < 8.0)
    fwd = known & (step >= REV_MIN_STEP_M) & (diff < FWD_MAX_DIFF)
    big_fwd = fwd & (step >= 4.0)
    res["pb_n_pts"] = n

    # push episodes over steps s (step s goes from point s to point s+1)
    ridx = np.nonzero(rev)[0]
    episodes = []
    if len(ridx):
        cum_fwd = np.concatenate([[0], np.cumsum(big_fwd)])
        cur = [ridx[0]]
        for a, b in zip(ridx[:-1], ridx[1:]):
            broken = (t[b] - t[a + 1] > EPISODE_GAP_S) or (cum_fwd[b] - cum_fwd[a + 1] > 0)
            if broken:
                episodes.append(cur)
                cur = [b]
            else:
                cur.append(b)
        episodes.append(cur)
    push = None
    for ep in episodes[::-1]:
        i0, i1 = ep[0], ep[-1] + 1
        net = float(np.hypot(x[i1] - x[i0], y[i1] - y[i0]))
        if len(ep) >= PUSH_MIN_REV and net >= PUSH_MIN_NET_M:
            push = (i0, i1, len(ep), net)
            break

    anchor = None  # (i_first, i_last, src)
    if push is not None:
        i0, i1, nrev, net = push
        res["pb_found"] = 1
        res["pb_n_rev"] = nrev
        res["pb_push_m"] = net
        first = i0 + 1
        v = max(float(gs[first]), 0.0) * 0.5144
        est = t[first] - (step[i0] / v if v > 0.3 else dt[i0] / 2.0)
        res["pb_start"] = tk - max(est, t[i0])
        res["pb_start_gap"] = float(dt[i0])
        res["pb_censored"] = int(i0 == 0)
        res["pb_end"] = tk - t[i1]
        res["pb_push_dur"] = t[i1] - max(est, t[i0])
        after = np.nonzero(fwd[i1:] & (step[i1:] >= 3.0))[0]
        if len(after):
            jg = i1 + after[0] + 1
            res["pb_go"] = tk - t[jg]
            res["pb_es_dur"] = t[jg] - t[i1]
        t8 = np.nonzero(gs[i1:] >= 8.0)[0]
        if len(t8):
            res["pb_taxi8"] = tk - t[i1 + t8[0]]
        # the stand cluster: contiguous points within CLUSTER_R_M of the pre-push point, going back
        j = i0
        while j - 1 >= 0 and np.hypot(x[j - 1] - x[i0], y[j - 1] - y[i0]) <= CLUSTER_R_M:
            j -= 1
        anchor = (j, i0, "push")
    else:
        res["pb_found"] = 0
        runs = _stationary_runs(t, x, y)
        if np.isfinite(slat) and np.isfinite(slon):
            sx, sy = _metres_xy(lat0, lon0, slat, slon)
            near = [(a, b) for a, b in runs if np.hypot(np.median(x[a:b + 1]) - sx, np.median(y[a:b + 1]) - sy) <= 80.0]
            if near:
                anchor = (near[-1][0], near[-1][1], "stand")
        if anchor is None:
            long_ = [(a, b) for a, b in runs if t[b] - t[a] >= 300.0]
            if long_:
                anchor = (long_[0][0], long_[0][1], "long")
    if anchor is None:
        res["pb_anchor_src"] = "none"
        return res
    a, b, src = anchor
    res["pb_anchor_src"] = src
    cx, cy = float(np.median(x[a:b + 1])), float(np.median(y[a:b + 1]))
    res["pb_cl_n"] = b - a + 1
    res["pb_cl_dwell"] = t[b] - t[a]
    if np.isfinite(slat) and np.isfinite(slon):
        sx, sy = _metres_xy(lat0, lon0, slat, slon)
        res["pb_cl_dist_stand"] = float(np.hypot(cx - sx, cy - sy))
    d = np.hypot(x - cx, y - cy)
    inside = np.nonzero(d[a:] <= LEAVE_R_M)[0]
    if len(inside):
        last_in = a + inside[-1]
        if last_in < n - 1:
            res["pb_leave"] = tk - t[last_in]
            res["pb_leave_next"] = tk - t[last_in + 1]
    return res


def compute_day(day: str) -> pd.DataFrame:
    """Pushback features of every linked departure of `day` (index mvt_id), times as taxi estimates."""
    lk = pd.read_parquet(paths.CACHE / "adsb" / f"links_{day}.parquet")
    p = _day_points(day)
    groups = p.groupby(["icao", "apt"], sort=False).indices
    T = p["t"].to_numpy()
    LA, LO = p["lat"].to_numpy(), p["lon"].to_numpy()
    GS = p["gs_kt"].to_numpy(dtype="float64")
    TR = p["track"].to_numpy(dtype="float64")
    stands = _stand_positions()
    mv = _movements().reindex(lk.index)
    pos = stands.reindex(pd.MultiIndex.from_arrays([mv["apt"].fillna(""), mv["stand_mvt"].fillna("")]))
    slat = dict(zip(lk.index, pos["lat"].to_numpy()))
    slon = dict(zip(lk.index, pos["lon"].to_numpy()))
    rows = {}
    for mvt_id, icao, apt, tk, tm in zip(lk.index, lk["icao"], lk["apt"].astype(str), lk["t_takeoff"], lk["t_mvt"]):
        idx = groups.get((icao, apt))
        if idx is None:
            continue
        tt = T[idx]
        lo_, hi_ = np.searchsorted(tt, tk - LOOKBACK_S), np.searchsorted(tt, tk)
        sel = idx[lo_:hi_]
        if len(sel):  # drop repeated timestamps
            keep = np.concatenate([[True], np.diff(T[sel]) > 0])
            sel = sel[keep]
        r = detect(T[sel], LA[sel], LO[sel], GS[sel], TR[sel], tk, slat.get(mvt_id, np.nan), slon.get(mvt_id, np.nan))
        off = tk - tm  # ADS-B takeoff minus recorded takeoff
        for c in ("pb_start", "pb_end", "pb_go", "pb_taxi8", "pb_leave", "pb_leave_next"):
            if c in r:
                r[c] = r[c] - off
        rows[mvt_id] = r
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "mvt_id"
    return out


def _one(day: str) -> str:
    DAY_DIR.mkdir(parents=True, exist_ok=True)
    f = DAY_DIR / f"{day}.parquet"
    out = compute_day(day)
    out.to_parquet(f)
    return f"{day}: {len(out):,} rows, push found {out['pb_found'].mean():.3f}"


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


def collect(days: list[str]) -> pd.DataFrame:
    fr = pd.concat([pd.read_parquet(DAY_DIR / f"{d}.parquet") for d in days])
    return fr[~fr.index.duplicated(keep="last")]


HOLDOUT_DAYS = [d.strftime("%Y-%m-%d") for d in list(pd.date_range("2025-01-01", "2025-01-31"))
                + list(pd.date_range("2025-07-01", "2025-07-31"))]
CHECK_2026 = ["2026-01-05", "2026-01-15", "2026-01-25", "2026-07-05", "2026-07-15", "2026-07-25"]


if __name__ == "__main__":
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 1
    if "--holdout" in sys.argv:
        run(HOLDOUT_DAYS, workers)
        collect(HOLDOUT_DAYS).to_parquet(OUT_DIR / "pushback.parquet")
        print("wrote", OUT_DIR / "pushback.parquet")
    if "--check2026" in sys.argv:
        run(CHECK_2026, workers)
        collect(CHECK_2026).to_parquet(OUT_DIR / "pushback_2026.parquet")
        print("wrote", OUT_DIR / "pushback_2026.parquet")
    if "--days" in sys.argv:
        days = [a for a in sys.argv[sys.argv.index("--days") + 1:] if not a.startswith("--")]
        run(days, workers)
