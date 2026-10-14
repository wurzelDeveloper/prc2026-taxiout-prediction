"""Build the ADS-B event features for every departure of the days streamed so far.

For each day file in data/additional_data/adsb/points/: detect takeoffs, link them to that day's departures
(callsign + takeoff time), and compute family B (own movement) and family A (surface context at the planned
off-block, LOBT). Writes cache/adsb/features_<day>.parquet (one row per linked or context-covered departure) and
prints link coverage per airport.

Operational availability: family B is tier M (challenge-only: the flight's own movement after off-block); family A
is tier O when computed at its planned off-block anchor (LOBT).

Usage: python lab/adsb_features.py [--days YYYY-MM-DD ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prc import data, paths  # noqa: E402
from prc.adsb.events import link, own_movement, surface_context, takeoffs  # noqa: E402

OUT = paths.CACHE / "adsb"


def epoch(s: pd.Series) -> pd.Series:
    return (s - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()


_DEP = None


def _deps():
    global _DEP
    if _DEP is None:
        mvt = data.load_movements()
        dep = mvt[mvt["is_dep"]]
        _DEP = (dep, dep["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d"))
    return _DEP


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    days = [a for a in sys.argv[sys.argv.index("--days") + 1:] if not a.startswith("--")] if "--days" in sys.argv else None
    files = sorted(paths.ADSB_POINTS.glob("*.parquet"))
    if days:
        files = [f for f in files if f.stem in set(days)]
    force = "--force" in sys.argv
    todo = [str(f) for f in files if force or not (OUT / f"features_{f.stem}.parquet").exists()]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 1
    if workers > 1 and len(todo) > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            for msg in pool.imap_unordered(one_day, todo):
                print(msg, flush=True)
    else:
        for f in todo:
            print(one_day(f), flush=True)


def one_day(path: str) -> str:
    f = Path(path)
    dep, dep_day = _deps()
    out = OUT / f"features_{f.stem}.parquet"
    if True:
        pts = pd.read_parquet(f)
        # the day's departures, plus the first hour of the next day's (points after midnight sit in the next file)
        d = dep[dep_day == f.stem]
        callsign = d["CALLSIGN_flt"].astype("string").fillna(d["FLIGHT_mvt"].astype("string"))
        deps = pd.DataFrame({"apt": d["apt"].astype(str), "callsign": callsign, "t_mvt": epoch(d["MVT_TIME_UTC_mvt"])}, index=d.index)
        tko = takeoffs(pts)
        lk = link(deps, tko)
        own = own_movement(pts, lk, deps["t_mvt"], deps["apt"])
        when = epoch(d["LOBT_flt"])
        ctx = surface_context(pts, when, deps["apt"], lk)
        feat = pd.DataFrame(index=d.index)
        feat["adsb_linked"] = feat.index.isin(lk.index).astype("int8")
        feat["adsb_takeoff_dt"] = lk["dt"].reindex(feat.index)
        feat["adsb_link_by_time"] = lk["by_time"].reindex(feat.index)
        feat = feat.join(own).join(ctx)
        # where the first ADS-B sighting was, relative to the stand: how much of the taxi happened before reception
        st = pd.read_parquet(paths.GEO_STANDS_FILE, columns=["apt", "stand_id", "lat", "lon"])
        sm = pd.read_parquet(paths.GEO_STAND_MATCHES_FILE, columns=["apt", "stand_mvt", "stand_id"])
        sm = sm.merge(st, on=["apt", "stand_id"], how="left")
        key = pd.DataFrame({"apt": d["apt"].astype(str), "stand_mvt": d["STAND_mvt"].astype(str)}, index=d.index)
        pos = key.reset_index().merge(sm.assign(apt=sm["apt"].astype(str), stand_mvt=sm["stand_mvt"].astype(str)),
                                      on=["apt", "stand_mvt"], how="left").set_index("mvt_id")
        if "adsb_first_lat" in feat:
            k = 111_320.0
            dlat = (feat["adsb_first_lat"] - pos["lat"].reindex(feat.index)) * k
            dlon = (feat["adsb_first_lon"] - pos["lon"].reindex(feat.index)) * k * np.cos(np.radians(pos["lat"].reindex(feat.index)))
            feat["adsb_first_dist_stand"] = np.hypot(dlat, dlon)
        feat.to_parquet(out)
        cov = feat.groupby(deps["apt"])[["adsb_linked"]].mean().round(3)
        cov["first_move"] = feat.groupby(deps["apt"])["adsb_first_move"].apply(lambda s: s.notna().mean()).round(3) if "adsb_first_move" in feat else 0
        print(f"{f.stem}: {len(d):,} departures, linked {feat['adsb_linked'].mean():.3f}; per airport:\n{cov.T.to_string()}", flush=True)


if __name__ == "__main__":
    main()
