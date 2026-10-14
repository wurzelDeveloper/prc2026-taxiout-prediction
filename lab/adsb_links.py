"""Write the departure -> ADS-B aircraft links of every streamed day: cache/adsb/links_<day>.parquet.

Columns: index mvt_id; icao, t_takeoff (epoch s, ADS-B takeoff), dt (|ADS-B - recorded takeoff|), by_time (1 = time-only
fallback link), apt, t_mvt (recorded takeoff, epoch s). The same `takeoffs` + `link` as `adsb_features.py`, saved so
that feature experiments can read an aircraft's points without redoing the detection.

Usage: python lab/adsb_links.py [--workers N] [--force]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lab"))
from prc import paths  # noqa: E402
from prc.adsb.events import link, takeoffs  # noqa: E402
from adsb_features import _deps, epoch  # noqa: E402

OUT = paths.CACHE / "adsb"


def one_day(path: str) -> str:
    f = Path(path)
    dep, dep_day = _deps()
    d = dep[dep_day == f.stem]
    callsign = d["CALLSIGN_flt"].astype("string").fillna(d["FLIGHT_mvt"].astype("string"))
    deps = pd.DataFrame({"apt": d["apt"].astype(str), "callsign": callsign, "t_mvt": epoch(d["MVT_TIME_UTC_mvt"])}, index=d.index)
    lk = link(deps, takeoffs(pd.read_parquet(f)))
    lk = lk.join(deps[["apt", "t_mvt"]])
    lk.index.name = "mvt_id"
    lk.to_parquet(OUT / f"links_{f.stem}.parquet")
    return f"{f.stem}: {len(lk):,} links"


def main():
    files = sorted(paths.ADSB_POINTS.glob("*.parquet"))
    todo = [str(f) for f in files if "--force" in sys.argv or not (OUT / f"links_{f.stem}.parquet").exists()]
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
