"""Download the PRU open CSVs: one file per dataset and year.

Idempotent and resumable like the weather download: a file is written
atomically and skipped on later runs unless ``force``. The files are small
(tens of kB to a few MB) and served from a plain web server, so there is no
quota logic; a two-second pause between requests is courtesy.
"""

from __future__ import annotations

import time
from typing import Literal

from prc import paths
from prc.weather.download import _write_atomic_bytes, http_get_bytes

BASE_URL = "https://www.eurocontrol.int/performance/data/download/csv/"
USER_AGENT = "prc-challenge-2026-pru/1.0 (taxi-out time research; python-urllib)"
DELAY_S = 2.0
BZIP2_MAGIC = b"BZh"


def url(kind: str, year: int) -> str:
    return f"{BASE_URL}{paths.pru_raw_file(kind, year).name}"


def classify_pru(payload: bytes, kind: str) -> Literal["ok", "empty", "error"]:
    """The server answers a missing dataset with an HTML page, not a 404: check the body.

    ``apt_dly`` is bzip2-compressed (magic ``BZh``); the others are CSVs whose
    header names the year and the airport code.
    """
    if kind == "apt_dly":
        return "ok" if payload.startswith(BZIP2_MAGIC) else "error"
    lines = [ln for ln in payload.decode("latin-1").splitlines() if ln.strip()]
    if not lines:
        return "error"
    header = lines[0].upper()
    if "YEAR" not in header or "APT_ICAO" not in header:
        return "error"
    return "ok" if len(lines) > 1 else "empty"


def fetch_all(kinds: tuple[str, ...], years: tuple[int, ...], *, force: bool = False) -> list[paths.Path]:
    paths.ensure_dirs()
    todo = [(k, y) for k in kinds for y in years if force or not paths.pru_raw_file(k, y).exists()]
    print(f"pru: {len(kinds) * len(years) - len(todo)} of {len(kinds) * len(years)} files on disk, {len(todo)} to download")
    written = []
    for n, (kind, year) in enumerate(todo, 1):
        if n > 1:
            time.sleep(DELAY_S)
        payload = http_get_bytes(url(kind, year), headers={"User-Agent": USER_AGENT})
        status = classify_pru(payload, kind)
        if status == "error":
            raise RuntimeError(f"unexpected response for {url(kind, year)}:\n{payload[:300]!r}")
        path = paths.pru_raw_file(kind, year)
        _write_atomic_bytes(path, payload)
        print(f"  [{n}/{len(todo)}] {path.name}: {len(payload):,} bytes ({status})")
        written.append(path)
    return written
