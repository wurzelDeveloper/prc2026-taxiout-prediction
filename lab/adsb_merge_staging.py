"""Merge adsb.lol's `staging` release into the `prod` points the features read (findings 10.11).

adsb.lol publishes, for most days, a second aggregator instance's history (`staging`) next to `prod`; same source,
licence (ODbL) and format, but not the same feeders, so each holds aircraft and positions the other lacks (2026-07-11:
departures covered 67.1 % with prod, 70.4 % with both). For every day with a staging file in
data/additional_data/adsb/points_staging/, this writes the union of both point sets to
data/additional_data/adsb/points/<day>.parquet, after copying the original prod file to points_prod/<day>.parquet
(once; a rerun merges from the backup, so it is idempotent). Points both instances received are deduplicated on
(icao, time rounded to 0.1 s, apt).

Usage: python lab/adsb_merge_staging.py [--restore]    # --restore puts the prod files back
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prc import paths  # noqa: E402

STAGING = paths.ADSB / "points_staging"
BACKUP = paths.ADSB / "points_prod"


def merge_day(day: str) -> str:
    cur = paths.ADSB_POINTS / f"{day}.parquet"
    bak = BACKUP / f"{day}.parquet"
    if not bak.exists():
        shutil.copy2(cur, bak)
    prod = pd.read_parquet(bak)
    stg = pd.read_parquet(STAGING / f"{day}.parquet")
    both = pd.concat([prod, stg], ignore_index=True)
    both["apt"] = both["apt"].astype(str)
    key = both["t"].round(1)
    both = both.loc[~pd.DataFrame({"i": both["icao"], "t": key, "a": both["apt"]}).duplicated()]
    both = both.sort_values(["icao", "t"]).reset_index(drop=True)
    both["apt"] = both["apt"].astype("category")
    tmp = cur.with_suffix(".tmp")
    both.to_parquet(tmp, index=False)
    tmp.replace(cur)
    return f"{day}: prod {len(prod):,} + staging {len(stg):,} -> {len(both):,} points"


def main():
    BACKUP.mkdir(parents=True, exist_ok=True)
    if "--restore" in sys.argv:
        for b in sorted(BACKUP.glob("*.parquet")):
            shutil.copy2(b, paths.ADSB_POINTS / b.name)
        print(f"restored {len(list(BACKUP.glob('*.parquet')))} prod files")
        return
    days = sorted(p.stem for p in STAGING.glob("*.parquet"))
    for d in days:
        print(merge_day(d), flush=True)
    print(f"merged {len(days)} days")


if __name__ == "__main__":
    main()
