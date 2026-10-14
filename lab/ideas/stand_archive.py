"""Idea family: a target-free stand-type archive from ADS-B over all 124 days.

The per-row pushback detector (`pushback.py`, features3 `pb_*`) sees the off-block only when the aircraft's own
track shows it (about a third of the linked departures). This module transfers what the fully observed departures of a
stand show to the departures of the same stand that were not fully observed:

Pass 1, `obs_day(day)` (one row per linked departure): the first ground sighting of the segment after the last
> 30 min gap (the same point as `adsb_first_seen`), and, where the off-block is fully observed, its kind and time:

* ``push``: a detected reverse-motion push (pb_found, not censored, pre-push point <= 60 s before the first reverse
  point, push cluster within 80 m of the matched stand when the stand is matched); off-block = the push start;
* ``power``: no push, the aircraft sat >= 120 s within 80 m of its matched stand and left it with a gap <= 30 s
  (power-out or self-manoeuvring, or a push the detector missed); off-block = the last point at the stand.

For those, the origin (position at off-block), the push duration, the engine-start pause, the time from off-block to
the first forward move and to the first point >= 8 kt, the switch-on lead (off-block minus the first sighting, when
the first sighting was at the stand), and the radial time profile: seconds from off-block until the track first
reaches 10, 25, ..., 3000 m from the origin (linearly interpolated; NaN when the crossing step spans > 120 s).

Pass 2, `build_archive()`: per (airport, STAND_mvt), pooled over all 124 days (both years, no target): the share of
fully observed off-blocks that are pushes, the medians above, the empirical stand position (median origin), the
median first-sighting distance from it, the share of linked departures fully observed, the median radial profile,
and the support. Each statistic is shrunk toward the pier (stand prefix) and the pier toward the airport
(n / (n + K) weights). A second profile is kept per (airport, STAND_mvt, RUNWAY_mvt) for the runway-specific variant.

Pass 3, `compute_day(day)`: per linked departure, the archive values and off-block estimates, all in seconds before
the RECORDED takeoff (``*_est``): ``sa_go_est`` (first forward move after the push + the stand's typical
push-start-to-go), ``sa_8kt_est`` (first >= 8 kt point, seen slower first, + the stand's typical off-block-to-8 kt),
``sa_seen_est`` (first sighting at the stand - the stand's typical switch-on lead), ``sa_rad_est`` / ``sa_radrw_est``
(first sighting away from the stand + the stand's (stand x runway) typical time to reach that radial distance), and
``sa_best_est`` with its source (the best-timed of them, the row's own well-timed push start first).

Only ADS-B points before each aircraft's ADS-B takeoff, the links, the pushback day files (themselves target-free),
the geometry tables and movement columns available in 2026 are read. No target, no residual.

Tier M: every estimate reads the departing aircraft's own movement after its off-block (or its first sighting, which
for the rows this transfers to lies after the off-block); the archive values alone are tier O (a stand property from
other flights' archived tracks) but are only useful together with the row's own track.

    python lab/ideas/stand_archive.py obs [--workers 2]       # pass 1 for all 124 days
    python lab/ideas/stand_archive.py archive                 # pass 2
    python lab/ideas/stand_archive.py days [--workers 2]      # pass 3 for all 124 days + combined parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lab" / "ideas"))
from prc import paths  # noqa: E402
import pushback as pbmod  # noqa: E402

KEY = "stand_archive"
OUT = paths.CACHE / "lab" / "ideas"
OBS_DIR = OUT / f"{KEY}_obs"
DAY_DIR = OUT / f"{KEY}_days"
ARCHIVE_FILE = OUT / f"{KEY}_table.parquet"
ARCHIVE_RW_FILE = OUT / f"{KEY}_table_rw.parquet"
LOOKBACK_S = 4 * 3600.0
CUT_GAP_S = 1800.0
K = 111_320.0
RADII = [10, 25, 50, 100, 150, 200, 300, 400, 500, 700, 1000, 1500, 2000, 3000]
RCOLS = [f"tau_{r}" for r in RADII]
MAX_CROSS_DT = 120.0
SHRINK_K = 5.0
STATS = ["push_dur", "es_dur", "start_go", "start_8kt", "seen_lead"]

ALL_DAYS = sorted(q.stem for q in paths.ADSB_POINTS.glob("*.parquet"))
"""Every day with streamed ADS-B points (Jan+Jul 2025 and 2026, plus the Feb+Dec 2025 training days of findings 10.13)."""


def _xy(lat0, lon0, lat, lon):
    return (lon - lon0) * K * np.cos(np.radians(lat0)), (lat - lat0) * K


_MV = None


def _movements() -> pd.DataFrame:
    global _MV
    if _MV is None:
        from prc import data
        m = data.load_movements(columns=["is_dep", "apt", "STAND_mvt", "RUNWAY_mvt"])
        m = m[m["is_dep"]]
        _MV = pd.DataFrame({"apt": m["apt"].astype(str), "stand_mvt": m["STAND_mvt"].astype("string").fillna("__NA__"),
                            "runway": m["RUNWAY_mvt"].astype("string").fillna("__NA__")}, index=m.index)
    return _MV


# --- pass 1: per-departure observations -------------------------------------------------------------------------

def _radial_profile(t, x, y, i0, t_off):
    """Seconds from t_off until the track (points i0..) first reaches each radius from point i0's position."""
    ox, oy = x[i0], y[i0]
    d = np.hypot(x[i0:] - ox, y[i0:] - oy)
    tt = t[i0:]
    out = np.full(len(RADII), np.nan)
    for k, r in enumerate(RADII):
        j = np.nonzero(d >= r)[0]
        if not len(j):
            break
        j = j[0]
        if j == 0:
            out[k] = max(tt[0] - t_off, 0.0)
            continue
        if tt[j] - tt[j - 1] > MAX_CROSS_DT:
            continue
        fr = (r - d[j - 1]) / max(d[j] - d[j - 1], 1e-6)
        out[k] = tt[j - 1] + fr * (tt[j] - tt[j - 1]) - t_off
    return out


def obs_day(day: str) -> pd.DataFrame:
    lk = pd.read_parquet(paths.CACHE / "adsb" / f"links_{day}.parquet")
    pb = pd.read_parquet(paths.CACHE / "lab" / "ideas" / "pushback_days" / f"{day}.parquet")
    p = pbmod._day_points(day)
    groups = p.groupby(["icao", "apt"], sort=False).indices
    T = p["t"].to_numpy()
    LA, LO = p["lat"].to_numpy(), p["lon"].to_numpy()
    GS = p["gs_kt"].to_numpy(dtype="float64")
    mv = _movements().reindex(lk.index)
    pbr = pb.reindex(lk.index)
    rows = {}
    for k_, (mvt_id, icao, apt, tk, tm) in enumerate(zip(lk.index, lk["icao"], lk["apt"].astype(str), lk["t_takeoff"],
                                                          lk["t_mvt"])):
        idx = groups.get((icao, apt))
        if idx is None:
            continue
        tt = T[idx]
        sel = idx[np.searchsorted(tt, tk - LOOKBACK_S):np.searchsorted(tt, tk)]
        if not len(sel):
            continue
        sel = sel[np.concatenate([[True], np.diff(T[sel]) > 0])]
        t, la, lo, gs = T[sel], LA[sel], LO[sel], GS[sel]
        cut = np.nonzero(np.diff(t) > CUT_GAP_S)[0]
        s0 = cut[-1] + 1 if len(cut) else 0
        x, y = _xy(la[-1], lo[-1], la, lo)
        r = {"apt": apt, "stand_mvt": mv.at[mvt_id, "stand_mvt"], "runway": mv.at[mvt_id, "runway"],
             "seen_est": tm - t[s0], "first_lat": la[s0], "first_lon": lo[s0], "gs0": gs[s0], "n_seg": len(t) - s0}
        # 60 s after the first sighting: moving at all?
        j60 = np.searchsorted(t, t[s0] + 60.0)
        j60 = min(max(j60, s0), len(t) - 1)
        r["disp60"] = float(np.hypot(x[j60] - x[s0], y[j60] - y[s0]))
        # first point >= 8 kt in the segment, and whether a slower point preceded it
        f8 = np.nonzero(gs[s0:] >= 8.0)[0]
        if len(f8):
            r["t8_est"] = tm - t[s0 + f8[0]]
            r["t8_slow_before"] = int(f8[0] > 0)
        q = pbr.loc[mvt_id]
        kind, t_off = None, np.nan
        cl = q.get("pb_cl_dist_stand", np.nan)
        if q.get("pb_found") == 1 and q.get("pb_censored") == 0 and q.get("pb_start_gap", 1e9) <= 60 and \
                (not np.isfinite(cl) or cl <= 80):
            kind, t_off = "push", tm - q["pb_start"]
        elif q.get("pb_found") == 0 and q.get("pb_anchor_src") == "stand" and np.isfinite(cl) and cl <= 80 and \
                q.get("pb_cl_dwell", 0) >= 120 and np.isfinite(q.get("pb_leave", np.nan)) and \
                (q["pb_leave"] - q["pb_leave_next"]) <= 30:
            kind, t_off = "power", tm - q["pb_leave"]
        if kind is not None:
            i0 = int(np.searchsorted(t, t_off, side="right") - 1)
            i0 = min(max(i0, 0), len(t) - 1)
            r["kind"] = kind
            r["off_est"] = tm - t_off
            r["origin_lat"], r["origin_lon"] = la[i0], lo[i0]
            prof = _radial_profile(t, x, y, i0, t_off)
            for c, v in zip(RCOLS, prof):
                r[c] = v
            a8 = np.nonzero((gs[i0:] >= 8.0) & (t[i0:] > t_off))[0]
            if len(a8):
                r["start_8kt"] = t[i0 + a8[0]] - t_off
            if kind == "push":
                r["push_dur"] = q["pb_push_dur"]
                if np.isfinite(q.get("pb_go", np.nan)):
                    r["start_go"] = (tm - q["pb_go"]) - t_off
                    r["es_dur"] = q["pb_es_dur"]
            else:
                r["start_go"] = 0.0
            # switch-on lead: only when the first sighting of the segment was at the origin
            if s0 <= i0 and np.hypot(x[s0] - x[i0], y[s0] - y[i0]) <= 60.0:
                r["seen_lead"] = t_off - t[s0]
        rows[mvt_id] = r
    out = pd.DataFrame.from_dict(rows, orient="index")
    if "kind" not in out:   # no departure shows a push or a power-out: a near-empty release (2026-06-11, 3.8 MB of points)
        out["kind"] = pd.Series(index=out.index, dtype=object)
    out.index.name = "mvt_id"
    out["day"] = day
    return out


def _obs_one(day: str) -> str:
    OBS_DIR.mkdir(parents=True, exist_ok=True)
    o = obs_day(day)
    o.to_parquet(OBS_DIR / f"{day}.parquet")
    return f"obs {day}: {len(o):,} rows, full {o['kind'].notna().mean():.3f}, push {(o['kind'] == 'push').mean():.3f}"


# --- pass 2: the archive ----------------------------------------------------------------------------------------

def _pier(apt: pd.Series, stand: pd.Series) -> pd.Series:
    from prc.features.stand_runway import stand_prefix_key
    return stand_prefix_key(apt, stand).astype(str)


def _agg(o: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    g = o.groupby(keys, sort=False, observed=True)
    full = o[o["kind"].notna()]
    gf = full.groupby(keys, sort=False, observed=True)
    a = pd.DataFrame({"n_lnk": g.size()})
    a["n_full"] = gf.size()
    a["n_push"] = full[full["kind"] == "push"].groupby(keys, sort=False, observed=True).size()
    a = a.fillna({"n_full": 0, "n_push": 0})
    a["obs_share"] = a["n_full"] / a["n_lnk"]
    a["push_share"] = a["n_push"] / a["n_full"].where(a["n_full"] > 0)
    for c in STATS + RCOLS:
        a[c] = gf[c].median()
        a["n_" + c] = gf[c].count()
    a["first_dist"] = g["first_dist"].median()
    a["n_first_dist"] = g["first_dist"].count()
    return a


def _shrink(child: pd.DataFrame, parent: pd.DataFrame, pkey: pd.Series) -> pd.DataFrame:
    """Each statistic of `child` shrunk toward the parent row `pkey` names (n/(n+K))."""
    out = child.copy()
    P = parent.reindex(pkey.to_numpy())
    P.index = child.index
    for c, n in [("push_share", "n_full"), ("obs_share", "n_lnk"), ("first_dist", "n_first_dist")] + \
            [(c, "n_" + c) for c in STATS + RCOLS]:
        n_ = child[n].fillna(0).to_numpy()
        v = child[c].to_numpy(dtype="float64")
        pv = P[c].to_numpy(dtype="float64")
        w = n_ / (n_ + SHRINK_K)
        res = np.where(np.isfinite(v), w * v + (1 - w) * np.nan_to_num(pv, nan=0.0), pv)
        res = np.where(np.isfinite(v) & ~np.isfinite(pv), v, res)
        out[c] = res
    return out


def build_archive() -> None:
    o = pd.concat([pd.read_parquet(OBS_DIR / f"{d}.parquet") for d in ALL_DAYS])
    o = o[~o.index.duplicated(keep="last")]
    o["stand_mvt"] = o["stand_mvt"].astype(str)
    o["runway"] = o["runway"].astype(str)
    o["pier"] = _pier(o["apt"], o["stand_mvt"])
    # empirical stand position: median origin of the fully observed off-blocks (fallback: the OSM stand)
    emp = o[o["kind"].notna()].groupby(["apt", "stand_mvt"])[["origin_lat", "origin_lon"]].median()
    osm = pbmod._stand_positions()
    osm.index = osm.index.set_names(["apt", "stand_mvt"])
    keys = pd.MultiIndex.from_arrays([o["apt"], o["stand_mvt"]])
    e = emp.reindex(keys)
    s = osm.reindex(keys)
    slat = np.where(np.isfinite(e["origin_lat"].to_numpy()), e["origin_lat"].to_numpy(), s["lat"].to_numpy())
    slon = np.where(np.isfinite(e["origin_lon"].to_numpy()), e["origin_lon"].to_numpy(), s["lon"].to_numpy())
    dx, dy = _xy(slat, slon, o["first_lat"].to_numpy(), o["first_lon"].to_numpy())
    o["first_dist"] = np.hypot(dx, dy)
    # monotone profiles per flight (first reach times can only grow with the radius)
    R = o[RCOLS].to_numpy(dtype="float64")
    o[RCOLS] = np.fmax.accumulate(np.where(np.isfinite(R), R, -np.inf), axis=1).clip(min=0) * \
        np.where(np.isfinite(R), 1.0, np.nan)

    A = _agg(o, ["apt"])
    Pi = _agg(o, ["apt", "pier"])
    St = _agg(o, ["apt", "pier", "stand_mvt"])
    Pi = _shrink(Pi, A, Pi.index.get_level_values("apt").to_series(index=Pi.index))
    St = _shrink(St, Pi, pd.Series(list(zip(St.index.get_level_values("apt"), St.index.get_level_values("pier"))),
                                   index=St.index))
    St = St.reset_index()
    St = St.merge(emp.rename(columns={"origin_lat": "emp_lat", "origin_lon": "emp_lon"}).reset_index(),
                  on=["apt", "stand_mvt"], how="left")
    St.to_parquet(ARCHIVE_FILE)
    # runway-specific profiles: (apt, stand, runway) shrunk toward the stand profile
    Rw = _agg(o, ["apt", "stand_mvt", "runway"])
    Sx = St.set_index(["apt", "stand_mvt"])
    Rw = _shrink(Rw, Sx, pd.Series(list(zip(Rw.index.get_level_values("apt"), Rw.index.get_level_values("stand_mvt"))),
                                   index=Rw.index))
    Rw[["n_full"] + RCOLS].reset_index().to_parquet(ARCHIVE_RW_FILE)
    print(f"archive: {len(St):,} stands ({(St['n_full'] >= 5).sum():,} with >= 5 full observations), "
          f"{len(Rw):,} stand x runway rows; full share {o['kind'].notna().mean():.3f}")


# --- pass 3: per-departure estimates ---------------------------------------------------------------------------

_AR = None


def _archive():
    global _AR
    if _AR is None:
        St = pd.read_parquet(ARCHIVE_FILE)
        Rw = pd.read_parquet(ARCHIVE_RW_FILE)
        osm = pbmod._stand_positions()
        osm.index = osm.index.set_names(["apt", "stand_mvt"])
        _AR = (St.set_index(["apt", "stand_mvt"]), Rw.set_index(["apt", "stand_mvt", "runway"]), osm)
    return _AR


def _interp_profile(d: np.ndarray, prof: np.ndarray) -> np.ndarray:
    """Per row, the profile (rows x RADII, monotone) at distance d; linear in r, 0 at r = 0, clamped at the end."""
    out = np.full(len(d), np.nan)
    rr = np.array([0.0] + RADII, dtype="float64")
    for i in range(len(d)):
        if not np.isfinite(d[i]):
            continue
        pv = np.concatenate([[0.0], prof[i]])
        ok = np.isfinite(pv)
        if ok.sum() < 2:
            continue
        pv = np.fmax.accumulate(np.where(ok, pv, -np.inf))[ok]
        out[i] = np.interp(d[i], rr[ok], pv) if d[i] <= rr[ok][-1] else np.nan
    return out


def compute_day(day: str) -> pd.DataFrame:
    """Archive values and off-block estimates (seconds before the recorded takeoff) of every linked departure."""
    o = pd.read_parquet(OBS_DIR / f"{day}.parquet")
    pb = pd.read_parquet(paths.CACHE / "lab" / "ideas" / "pushback_days" / f"{day}.parquet").reindex(o.index)
    St, Rw, osm = _archive()
    keys = pd.MultiIndex.from_arrays([o["apt"].astype(str), o["stand_mvt"].astype(str)])
    a = St.reindex(keys)
    a.index = o.index
    pier = _pier(o["apt"], o["stand_mvt"].astype(str))
    # stands without their own archive row: the pier / airport values (rebuilt from the stand table)
    miss = a["n_lnk"].isna().to_numpy()
    if miss.any():
        Pt = St.reset_index().groupby(["apt", "pier"]).apply(
            lambda g: g[["push_share", "obs_share", "first_dist"] + STATS + RCOLS].median(), include_groups=False)
        Pa = St.reset_index().groupby(["apt"])[["push_share", "obs_share", "first_dist"] + STATS + RCOLS].median()
        fill = Pt.reindex(pd.MultiIndex.from_arrays([o["apt"].astype(str), pier]))
        fill.index = o.index
        fa = Pa.reindex(o["apt"].astype(str))
        fa.index = o.index
        fill = fill.fillna(fa)
        for c in fill.columns:
            a.loc[miss, c] = fill.loc[miss, c]
        a.loc[miss, ["n_lnk", "n_full"]] = 0
    rk = pd.MultiIndex.from_arrays([o["apt"].astype(str), o["stand_mvt"].astype(str), o["runway"].astype(str)])
    rw = Rw.reindex(rk)
    rw.index = o.index
    out = pd.DataFrame(index=o.index)
    out["ar_n_full"] = a["n_full"].to_numpy()
    out["ar_n_lnk"] = a["n_lnk"].to_numpy()
    out["ar_push_share"] = a["push_share"].to_numpy()
    out["ar_obs_share"] = a["obs_share"].to_numpy()
    out["ar_push_dur"] = a["push_dur"].to_numpy()
    out["ar_es_dur"] = a["es_dur"].to_numpy()
    out["ar_start_go"] = a["start_go"].to_numpy()
    out["ar_start_8kt"] = a["start_8kt"].to_numpy()
    out["ar_seen_lead"] = a["seen_lead"].to_numpy()
    out["ar_first_dist"] = a["first_dist"].to_numpy()
    out["ar_t300"] = a["tau_300"].to_numpy()
    # first-sighting distance from the empirical stand position (fallback OSM stand)
    s = osm.reindex(keys)
    elat = np.where(np.isfinite(a["emp_lat"].to_numpy(dtype="float64")), a["emp_lat"].to_numpy(dtype="float64"),
                    s["lat"].to_numpy())
    elon = np.where(np.isfinite(a["emp_lon"].to_numpy(dtype="float64")), a["emp_lon"].to_numpy(dtype="float64"),
                    s["lon"].to_numpy())
    dx, dy = _xy(elat, elon, o["first_lat"].to_numpy(), o["first_lon"].to_numpy())
    d = np.hypot(dx, dy)
    out["sa_first_d"] = d
    out["sa_first_d_rel"] = d - out["ar_first_dist"].to_numpy()
    seen = o["seen_est"].to_numpy(dtype="float64")
    gs0 = o["gs0"].to_numpy(dtype="float64")
    at_stand = (d <= 60.0) & (gs0 < 2.0)
    prof = a[RCOLS].to_numpy(dtype="float64")
    tprof = _interp_profile(d, prof)
    prof_rw = rw[RCOLS].to_numpy(dtype="float64")
    prof_rw = np.where(np.isfinite(prof_rw), prof_rw, prof)
    tprof_rw = _interp_profile(d, prof_rw)
    away = (d > 25.0) & ~at_stand
    out["sa_rad_est"] = np.where(away, seen + tprof, np.nan)
    out["sa_radrw_est"] = np.where(away, seen + tprof_rw, np.nan)
    out["sa_seen_est"] = np.where(at_stand, seen - out["ar_seen_lead"].to_numpy(), np.nan)
    go = pb["pb_go"].to_numpy(dtype="float64")
    out["sa_go_est"] = go + out["ar_start_go"].to_numpy()
    t8 = o["t8_est"].to_numpy(dtype="float64") if "t8_est" in o else np.full(len(o), np.nan)
    slow = (o["t8_slow_before"].to_numpy(dtype="float64") == 1) if "t8_slow_before" in o else np.zeros(len(o), bool)
    out["sa_8kt_est"] = np.where(slow, t8 + out["ar_start_8kt"].to_numpy(), np.nan)
    # best: own well-timed off-block, else the archive transfers in order of timing quality
    own = o["off_est"].to_numpy(dtype="float64") if "off_est" in o else np.full(len(o), np.nan)
    kind = o["kind"].astype("string").fillna("") if "kind" in o else pd.Series("", index=o.index)
    cand = [("own_" + kind.to_numpy().astype(str), own), ("go", out["sa_go_est"].to_numpy()),
            ("8kt", out["sa_8kt_est"].to_numpy()), ("seen", out["sa_seen_est"].to_numpy()),
            ("rad", out["sa_radrw_est"].to_numpy())]
    best = np.full(len(o), np.nan)
    src = np.full(len(o), "none", dtype=object)
    for name, v in cand:
        take = ~np.isfinite(best) & np.isfinite(v)
        best[take] = v[take]
        src[take] = name[take] if isinstance(name, np.ndarray) else name
    out["sa_best_est"] = best
    out["sa_best_src"] = src
    # the transfer's own uncertainty: spread of the stand's profile at that distance is not stored; the support is
    out.index.name = "mvt_id"
    return out


def _day_one(day: str) -> str:
    DAY_DIR.mkdir(parents=True, exist_ok=True)
    out = compute_day(day)
    out.to_parquet(DAY_DIR / f"{day}.parquet")
    return f"day {day}: {len(out):,} rows, best src {out['sa_best_src'].value_counts(normalize=True).round(3).to_dict()}"


def _run(fn, days, workers, outdir):
    todo = [d for d in days if "--force" in sys.argv or not (outdir / f"{d}.parquet").exists()]
    if workers > 1 and len(todo) > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            for msg in pool.imap_unordered(fn, todo):
                print(msg, flush=True)
    else:
        for d in todo:
            print(fn(d), flush=True)


def assemble() -> pd.DataFrame:
    f = pd.concat([pd.read_parquet(DAY_DIR / f"{d}.parquet") for d in ALL_DAYS])
    f = f[~f.index.duplicated(keep="last")]
    f.to_parquet(OUT / f"{KEY}.parquet")
    print("wrote", OUT / f"{KEY}.parquet", f.shape)
    return f


if __name__ == "__main__":
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 1
    days = [a for a in sys.argv[sys.argv.index("--days") + 1:] if not a.startswith("--")] if "--days" in sys.argv \
        else ALL_DAYS
    if "obs" in sys.argv:
        _run(_obs_one, days, workers, OBS_DIR)
    if "archive" in sys.argv:
        build_archive()
    if "days" in sys.argv:
        _run(_day_one, days, workers, DAY_DIR)
        if days == ALL_DAYS:
            assemble()
