"""Idea family "track quality": telling genuine ADS-B ground tracks from false readings.

Per linked departure (cache/adsb/links_<day>.parquet), over the aircraft's points at the airport in the 4 h before
its ADS-B takeoff ``tk`` (only points before ``tk``; ground = on_ground | gs < 40 kt). Every time is in seconds
BEFORE ``tk`` (like the existing adsb_* features). Nothing reads the target. Columns (prefix adsbQ_):

reception / density
  dens30, dens60          ground points per minute in the last 30 / 60 min before tk
  seg_dens                ground points per minute over the segment (after the last > 30 min gap, as adsb_first_seen)
  gap60_n, gap60_s        number / total length of gaps > 60 s in the segment (the final gap to tk included)
  gap300_n, gap300_s      same for gaps > 300 s
  onground_share          share of the segment's ground points that carry the on_ground flag
  first_gs, first_ong     ground speed / on_ground flag of the segment's first point (moving at first sight = the
                          aircraft entered coverage mid-taxi, not a transponder switch-on at the stand)
  first_next              seconds from the segment's first point to its second (an isolated early reading)
jitter at the parked position
  jit_std, jit_max, jit_n RMS / max distance (m) from the median position of the initial stationary cluster (gs < 1,
                          within 100 m of the first point, before the first point >= 8 kt), and its point count
  fm_gs                   ground speed at the first point > 25 m from the first point (adsb_first_move's point)
  fm_slow_n               points with gs < 1 that lie > 25 m from the first point before taxi start (jitter-driven
                          displacement: what makes adsb_first_move fire early)
implausible jumps
  jumps                   consecutive segment ground points, both reporting < 40 kt, with implied speed > 60 kt
                          over > 100 m (position jumps, not the takeoff roll)
  jump_last               time of the last such jump before tk
inbound landing
  landed                  1 when the lookback holds an airborne point (not on_ground, gs >= 60) followed by ground
  since_landing           tk - first ground point after the last airborne point (the inbound touchdown)
  seg_after_landing       segment start - touchdown (about 0: the segment includes the taxi-in)
relocations (stationary clusters: gs < 1, sequentially within 100 m of the cluster's first point)
  n_locs, n_locs300       distinct stationary locations (> 100 m apart) with dwell >= 120 s / >= 300 s
  reloc                   start of the last >= 300 s cluster when there were >= 2 distinct such locations (the end of
                          a tow / repositioning), NaN otherwise
  park_end                last ground point within 50 m of the last >= 300 s cluster's median (jitter-robust)
  park_dist_stand         metres from that cluster's median to the matched OSM stand
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from prc import data, paths  # noqa: E402

LOOKBACK = 4 * 3600.0
OUT = paths.CACHE / "lab" / "ideas"
K = 111_320.0


def _m(lat0, lon0, lat, lon):
    return np.hypot((lat - lat0) * K, (lon - lon0) * K * np.cos(np.radians(lat0)))


_STAND = None


def _stand_pos() -> pd.DataFrame:
    """Matched OSM stand coordinates of every departure (index mvt_id; lat, lon), target-free."""
    global _STAND
    if _STAND is None:
        mvt = data.load_movements(columns=["apt", "is_dep", "STAND_mvt"])
        mvt = mvt[mvt["is_dep"]]
        st = pd.read_parquet(paths.GEO_STANDS_FILE, columns=["apt", "stand_id", "lat", "lon"])
        sm = pd.read_parquet(paths.GEO_STAND_MATCHES_FILE, columns=["apt", "stand_mvt", "stand_id"])
        sm = sm.merge(st, on=["apt", "stand_id"], how="left")
        sm["apt"] = sm["apt"].astype(str)
        sm["stand_mvt"] = sm["stand_mvt"].astype(str)
        key = pd.DataFrame({"apt": mvt["apt"].astype(str), "stand_mvt": mvt["STAND_mvt"].astype(str)}, index=mvt.index)
        key.index.name = "mvt_id"
        _STAND = key.reset_index().merge(sm[["apt", "stand_mvt", "lat", "lon"]], on=["apt", "stand_mvt"],
                                         how="left").drop_duplicates("mvt_id").set_index("mvt_id")[["lat", "lon"]]
    return _STAND


def _load_points(day: str) -> pd.DataFrame:
    f = paths.ADSB_POINTS / f"{day}.parquet"
    p = pd.read_parquet(f)
    prev = paths.ADSB_POINTS / f"{(pd.Timestamp(day) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')}.parquet"
    if prev.exists():
        start = (pd.Timestamp(day, tz="UTC") - pd.Timestamp("1970-01-01", tz="UTC")).total_seconds() - LOOKBACK - 600
        q = pd.read_parquet(prev)
        p = pd.concat([q[q["t"] >= start], p], ignore_index=True)
    p["apt"] = p["apt"].astype(str)
    return p.sort_values(["icao", "apt", "t"], kind="stable").reset_index(drop=True)


def _clusters(t, la, lo, stat):
    """Sequential stationary clusters over the points flagged `stat`: (start, end, median lat, median lon, n)."""
    idx = np.nonzero(stat)[0]
    out = []
    if not len(idx):
        return out
    cur = [idx[0]]
    for i in idx[1:]:
        a = cur[0]
        if _m(la[a], lo[a], la[i], lo[i]) <= 100.0:
            cur.append(i)
        else:
            out.append(cur)
            cur = [i]
    out.append(cur)
    res = []
    for c in out:
        c = np.asarray(c)
        res.append((t[c[0]], t[c[-1]], float(np.median(la[c])), float(np.median(lo[c])), len(c)))
    return res


def _distinct(cl):
    locs = []
    for _, _, a, o, _ in cl:
        if all(_m(a0, o0, a, o) > 100.0 for a0, o0 in locs):
            locs.append((a, o))
    return len(locs)


def _one(t, la, lo, gs, ong, tk, slat, slon) -> dict:
    out = {}
    air = (~ong) & (gs >= 60)
    gr = ong | (gs < 40)
    # inbound landing: last airborne point well before the takeoff roll, then the first ground point after it
    ab = np.nonzero(air & (t < tk - 120))[0]
    t_land = np.nan
    if len(ab):
        after = np.nonzero(gr & (np.arange(len(t)) > ab[-1]))[0]
        if len(after):
            t_land = t[after[0]]
    out["adsbQ_landed"] = float(np.isfinite(t_land))
    out["adsbQ_since_landing"] = tk - t_land if np.isfinite(t_land) else np.nan
    # ground points only from here on
    t, la, lo, gs, ong = t[gr], la[gr], lo[gr], gs[gr], ong[gr]
    if not len(t):
        return out
    out["adsbQ_dens30"] = np.sum(t >= tk - 1800) / 30.0
    out["adsbQ_dens60"] = np.sum(t >= tk - 3600) / 60.0
    # relocations over the whole ground history of the lookback
    stat = gs < 1.0
    cl = [c for c in _clusters(t, la, lo, stat) if c[1] - c[0] >= 120]
    cl300 = [c for c in cl if c[1] - c[0] >= 300]
    out["adsbQ_n_locs"] = _distinct(cl)
    out["adsbQ_n_locs300"] = _distinct(cl300)
    if cl300:
        last = cl300[-1]
        near = np.nonzero(_m(last[2], last[3], la, lo) <= 50.0)[0]
        out["adsbQ_park_end"] = tk - t[near[-1]] if len(near) else tk - last[1]
        if np.isfinite(slat) and np.isfinite(slon):
            out["adsbQ_park_dist_stand"] = float(_m(slat, slon, last[2], last[3]))
        if out["adsbQ_n_locs300"] >= 2:
            out["adsbQ_reloc"] = tk - last[0]
    # the segment after the last > 30 min gap (the one adsb_first_seen starts)
    cut = np.nonzero(np.diff(t) > 1800)[0]
    if len(cut):
        s = cut[-1] + 1
        t, la, lo, gs, ong = t[s:], la[s:], lo[s:], gs[s:], ong[s:]
    n = len(t)
    span = tk - t[0]
    out["adsbQ_seg_dens"] = n / max(span / 60.0, 0.5)
    g = np.diff(np.append(t, tk))
    for th in (60, 300):
        big = g > th
        out[f"adsbQ_gap{th}_n"] = int(big.sum())
        out[f"adsbQ_gap{th}_s"] = float(g[big].sum())
    out["adsbQ_onground_share"] = float(ong.mean())
    out["adsbQ_first_gs"] = float(gs[0]) if np.isfinite(gs[0]) else np.nan
    out["adsbQ_first_ong"] = float(ong[0])
    out["adsbQ_first_next"] = float(g[0])
    if np.isfinite(t_land):
        out["adsbQ_seg_after_landing"] = t[0] - t_land
    # jitter of the initial stationary cluster
    taxi = np.nonzero(gs >= 8.0)[0]
    end = taxi[0] if len(taxi) else n
    d0 = _m(la[0], lo[0], la, lo)
    c = np.nonzero((np.arange(n) < end) & (gs < 1.0) & (d0 <= 100.0))[0]
    out["adsbQ_jit_n"] = int(len(c))
    if len(c) >= 3:
        dm = _m(np.median(la[c]), np.median(lo[c]), la[c], lo[c])
        out["adsbQ_jit_std"] = float(np.sqrt(np.mean(dm ** 2)))
        out["adsbQ_jit_max"] = float(dm.max())
    mv = np.nonzero(d0 > 25.0)[0]
    if len(mv):
        out["adsbQ_fm_gs"] = float(gs[mv[0]]) if np.isfinite(gs[mv[0]]) else np.nan
    out["adsbQ_fm_slow_n"] = int(np.sum((np.arange(n) < end) & (gs < 1.0) & (d0 > 25.0)))
    # implausible jumps
    if n > 1:
        step = _m(la[:-1], lo[:-1], la[1:], lo[1:])
        dt = np.maximum(np.diff(t), 0.5)
        slow2 = np.fmax(gs[:-1], gs[1:]) < 40.0  # both ends report taxi speed: the takeoff roll is not a jump
        jump = (step > 100.0) & (step / dt * 1.943844 > 60.0) & slow2
        out["adsbQ_jumps"] = int(jump.sum())
        jj = np.nonzero(jump)[0]
        out["adsbQ_jump_last"] = tk - t[jj[-1] + 1] if len(jj) else np.nan
    else:
        out["adsbQ_jumps"] = 0
    return out


def compute_day(day: str) -> pd.DataFrame:
    """Track-quality features of every linked departure of `day` (index mvt_id)."""
    lk = pd.read_parquet(paths.CACHE / "adsb" / f"links_{day}.parquet")
    p = _load_points(day)
    grp = p.groupby(["icao", "apt"], sort=False).indices
    T = p["t"].to_numpy()
    LA, LO = p["lat"].to_numpy(), p["lon"].to_numpy()
    GS = p["gs_kt"].to_numpy(dtype="float64")
    ONG = p["on_ground"].to_numpy(bool)
    sp = _stand_pos()
    slat = sp["lat"].reindex(lk.index).to_numpy()
    slon = sp["lon"].reindex(lk.index).to_numpy()
    rows = {}
    for k, (mvt_id, ic, ap, tk) in enumerate(zip(lk.index, lk["icao"], lk["apt"].astype(str), lk["t_takeoff"])):
        ix = grp.get((ic, ap))
        if ix is None:
            continue
        tt = T[ix]
        lo_, hi_ = np.searchsorted(tt, tk - LOOKBACK), np.searchsorted(tt, tk)
        if hi_ <= lo_:
            continue
        sel = ix[lo_:hi_]
        rows[mvt_id] = _one(T[sel], LA[sel], LO[sel], GS[sel], ONG[sel], tk, slat[k], slon[k])
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "mvt_id"
    return df


def _run(day: str) -> str:
    out = OUT / "track_quality_days" / f"{day}.parquet"
    if not out.exists():
        compute_day(day).to_parquet(out)
    return day


def main():
    (OUT / "track_quality_days").mkdir(parents=True, exist_ok=True)
    days = [f.stem for f in sorted(paths.ADSB_POINTS.glob("*.parquet"))]
    if "--days" in sys.argv:
        days = [a for a in sys.argv[sys.argv.index("--days") + 1:] if not a.startswith("--")]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 1
    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            for d in pool.imap_unordered(_run, days):
                print(d, flush=True)
    else:
        for d in days:
            print(_run(d), flush=True)
    # assemble the holdout file
    hold = [d for d in days if d.startswith("2025-")]
    parts = [pd.read_parquet(OUT / "track_quality_days" / f"{d}.parquet") for d in hold
             if (OUT / "track_quality_days" / f"{d}.parquet").exists()]
    if parts:
        h = pd.concat(parts)
        h = h[~h.index.duplicated(keep="last")]
        h.to_parquet(OUT / "track_quality.parquet")
        print(f"wrote {OUT / 'track_quality.parquet'}: {len(h):,} rows")


if __name__ == "__main__":
    main()
