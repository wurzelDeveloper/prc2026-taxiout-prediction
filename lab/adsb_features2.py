"""Second ADS-B feature set: the ground history read against the flight's own stand and its stops.

For each day file in data/additional_data/adsb/points/: detect takeoffs and link them as `adsb_features.py` does,
then compute `prc.adsb.events.stand_movement` with the stand coordinates of the geometry tables. Writes
cache/adsb/features2_<day>.parquet; `lab/adsb_apply.py --v2` joins it to the first set.

Operational availability: tier M (challenge-only: the flight's own movement after off-block).

Usage: python lab/adsb_features2.py [--days YYYY-MM-DD ...] [--workers N] [--force]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lab"))
from prc import paths  # noqa: E402
from prc.adsb.events import link, stand_movement, takeoffs  # noqa: E402
from adsb_features import _deps, epoch  # noqa: E402

OUT = paths.CACHE / "adsb"


def stand_positions(d: pd.DataFrame) -> pd.DataFrame:
    st = pd.read_parquet(paths.GEO_STANDS_FILE, columns=["apt", "stand_id", "lat", "lon"])
    sm = pd.read_parquet(paths.GEO_STAND_MATCHES_FILE, columns=["apt", "stand_mvt", "stand_id"])
    sm = sm.merge(st, on=["apt", "stand_id"], how="left")
    sm = sm.assign(apt=sm["apt"].astype(str), stand_mvt=sm["stand_mvt"].astype(str))
    key = pd.DataFrame({"apt": d["apt"].astype(str), "stand_mvt": d["STAND_mvt"].astype(str)}, index=d.index)
    return key.reset_index().merge(sm, on=["apt", "stand_mvt"], how="left").set_index("mvt_id")[["lat", "lon"]]


def one_day(path: str) -> str:
    f = Path(path)
    dep, dep_day = _deps()
    d = dep[dep_day == f.stem]
    pts = pd.read_parquet(f)
    callsign = d["CALLSIGN_flt"].astype("string").fillna(d["FLIGHT_mvt"].astype("string"))
    deps = pd.DataFrame({"apt": d["apt"].astype(str), "callsign": callsign, "t_mvt": epoch(d["MVT_TIME_UTC_mvt"])}, index=d.index)
    lk = link(deps, takeoffs(pts))
    pos = stand_positions(d)
    feat = stand_movement(pts, lk, deps["apt"], pos["lat"], pos["lon"])
    feat.index.name = "mvt_id"
    feat.to_parquet(OUT / f"features2_{f.stem}.parquet")
    return f"{f.stem}: {len(lk):,} linked, {len(feat):,} rows, left_stand_100 known {feat.get('adsb_left_stand_100', pd.Series(dtype=float)).notna().mean():.2f}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(paths.ADSB_POINTS.glob("*.parquet"))
    if "--days" in sys.argv:
        days = {a for a in sys.argv[sys.argv.index("--days") + 1:] if not a.startswith("--")}
        files = [f for f in files if f.stem in days]
    force = "--force" in sys.argv
    todo = [str(f) for f in files if force or not (OUT / f"features2_{f.stem}.parquet").exists()]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 1
    if workers > 1 and len(todo) > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            for msg in pool.imap_unordered(one_day, todo):
                print(msg, flush=True)
    else:
        for f in todo:
            print(one_day(f), flush=True)


if __name__ == "__main__":
    main()
