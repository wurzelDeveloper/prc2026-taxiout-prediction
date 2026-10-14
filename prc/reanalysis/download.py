"""Download the Open-Meteo ERA5 archive: one request per airport for the whole period.

Idempotent and resumable like the PRU download: a file is written atomically,
only for a complete answer, and skipped on later runs unless ``force``. Ten
requests of a few MB each; a two-second pause between them is courtesy (the
free tier allows 10,000 calls a day).
"""

from __future__ import annotations

import json
import time
from urllib.parse import urlencode

import pandas as pd

from prc import paths
from prc.reanalysis import END, START, VARIABLES
from prc.weather.download import _write_atomic_bytes, http_get_bytes

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
USER_AGENT = "prc-challenge-2026-era5/1.0 (taxi-out time research; python-urllib)"
DELAY_S = 2.0


def airports() -> pd.DataFrame:
    """The ten reference points from the geometry tables (`fetch-geometry` first)."""
    if not paths.GEO_AIRPORTS_FILE.exists():
        raise FileNotFoundError(f"{paths.GEO_AIRPORTS_FILE} missing: run `python -m prc fetch-geometry` first")
    return pd.read_parquet(paths.GEO_AIRPORTS_FILE, columns=["apt", "arp_lat", "arp_lon"])


def url(lat: float, lon: float) -> str:
    query = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": START,
        "end_date": END,
        "hourly": ",".join(VARIABLES),
        "timezone": "GMT",
    }
    return f"{BASE_URL}?{urlencode(query)}"


def complete(payload: bytes) -> bool:
    """An answer counts only with every variable and an hourly axis spanning the period."""
    try:
        body = json.loads(payload)
    except ValueError:
        return False
    hourly = body.get("hourly") or {}
    times = hourly.get("time") or []
    return bool(times) and all(v in hourly for v in VARIABLES) and times[0].startswith(START) and times[-1].startswith(END)


def fetch_all(*, force: bool = False) -> list[paths.Path]:
    paths.ensure_dirs()
    apts = airports()
    todo = [r for r in apts.itertuples(index=False) if force or not paths.era5_raw_file(r.apt).exists()]
    print(f"era5: {len(apts) - len(todo)} of {len(apts)} airports on disk, {len(todo)} to download")
    written = []
    for n, r in enumerate(todo, 1):
        if n > 1:
            time.sleep(DELAY_S)
        payload = http_get_bytes(url(r.arp_lat, r.arp_lon), headers={"User-Agent": USER_AGENT})
        if not complete(payload):
            raise RuntimeError(f"incomplete ERA5 answer for {r.apt}:\n{payload[:300]!r}")
        path = paths.era5_raw_file(r.apt)
        _write_atomic_bytes(path, payload)
        print(f"  [{n}/{len(todo)}] {path.name}: {len(payload):,} bytes")
        written.append(path)
    return written
