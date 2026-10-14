"""`prc fetch-adsb`: stream adsb.lol days and keep the points at the ten airports, resumably.

One parquet per day under ``data/additional_data/adsb/points/`` (``YYYY-MM-DD.parquet``); a day that
exists is skipped, so a killed crawl resumes where it stopped. Days run in parallel worker processes
(the link, ~13 MB/s here, is the limit at about three workers). See `prc.adsb.stream` for the format.

Data: adsb.lol contributors, ODbL 1.0.
"""

from __future__ import annotations

import multiprocessing as mp
import time
import traceback

import pandas as pd

from prc import paths
from prc.adsb.stream import Box, stream_day


def boxes() -> list[Box]:
    a = pd.read_parquet(paths.GEO_AIRPORTS_FILE)
    return [Box(str(r.apt), float(r.arp_lat), float(r.arp_lon)) for r in a.itertuples()]


KIND = "prod"
"""Which adsb.lol release to stream: "prod" (the main aggregator, into points/) or "staging" (a second aggregator
instance published for many days, same licence and format, into points_staging/; merged with prod by
`lab/adsb_merge_staging.py`)."""


def out_dir(kind: str | None = None):
    kind = kind or KIND
    return paths.ADSB_POINTS if kind == "prod" else paths.ADSB / f"points_{kind}"


def day_file(day: pd.Timestamp, kind: str | None = None):
    return out_dir(kind) / f"{day:%Y-%m-%d}.parquet"


def _one(arg) -> str:
    day_str, kind = arg if isinstance(arg, tuple) else (arg, KIND)
    day = pd.Timestamp(day_str)
    out = day_file(day, kind)
    if out.exists():
        return f"{day_str}: exists"
    log_file = paths.ADSB / "fetch.log"
    def log(msg):
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    try:
        frame = stream_day(day, boxes(), log=log, kind=kind)
        tmp = out.with_suffix(".tmp")
        frame.to_parquet(tmp, index=False)
        tmp.replace(out)
        return f"{day_str}: {len(frame):,} points"
    except Exception:
        log(f"  {day_str}: FAILED\n{traceback.format_exc()}")
        return f"{day_str}: FAILED"


def default_days() -> list[str]:
    """Jan+Jul 2025 (the holdout) and Jan+Jul 2026 (the ranking months), interleaved month by month."""
    months = [pd.date_range(f"{m}-01", periods=31, freq="D") for m in ("2026-07", "2025-07", "2026-01", "2025-01")]
    order = []
    for i in range(31):
        for m in months:
            order.append(f"{m[i]:%Y-%m-%d}")
    return order


def fetch(days: list[str] | None = None, workers: int = 3, kind: str = "prod") -> None:
    out_dir(kind).mkdir(parents=True, exist_ok=True)
    todo = [d for d in (days or default_days()) if not day_file(pd.Timestamp(d), kind).exists()]
    print(f"fetch-adsb ({kind}): {len(todo)} days to stream with {workers} workers; progress in {paths.ADSB / 'fetch.log'}", flush=True)
    with mp.Pool(workers) as pool:
        for msg in pool.imap_unordered(_one, [(d, kind) for d in todo]):
            print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)
