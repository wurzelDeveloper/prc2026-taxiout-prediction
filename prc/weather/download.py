"""Download METAR (IEM) and TAF (Ogimet) archives: one file per station-month.

Idempotent and resumable. A chunk is written atomically (``.part`` then
``os.replace``) and skipped on later runs unless ``force`` -- so an interrupted
download (Ogimet's quota, a dropped connection) is finished by running the
command again, without re-requesting anything already on disk.
"""

from __future__ import annotations

import calendar
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Literal

import pandas as pd

from prc import data, paths

IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
OGIMET_TAF_URL = "https://www.ogimet.com/cgi-bin/gettafor"
USER_AGENT = "prc-challenge-2026-weather/1.0 (python-urllib)"

# IEM throttles at one request per second per IP; Ogimet asks not to be abused
# and answers over-frequent queries with a quota notice instead of data.
DELAY_S = {"metar": 2.0, "taf": 20.0}
QUOTA_SLEEP_S = 15 * 60
QUOTA_RETRIES = 4

_OGIMET_ROW = re.compile(r"^[A-Z]{4},\d{4},\d{2},\d{2},\d{2},\d{2},")


class QuotaExceeded(RuntimeError):
    """Ogimet kept refusing; running the command again later resumes the download."""


def needed_months(mvt: pd.DataFrame) -> list[str]:
    """Calendar months (``YYYY-MM``) touched by any movement day, padded by -2/+1 days.

    The padding covers the 24 h look-back windows and TAFs issued up to 30 h
    before the time they forecast (backwards), and the look-ahead windows
    (forwards).
    """
    days = mvt["MVT_TIME_UTC_mvt"].dt.floor("D").drop_duplicates()
    padded = pd.concat([days + pd.Timedelta(days=k) for k in (-2, -1, 0, 1)])
    return sorted(set(padded.dt.strftime("%Y-%m")))


def _year_month(month: str) -> tuple[int, int]:
    y, m = month.split("-")
    return int(y), int(m)


def iem_metar_url(station: str, month: str) -> str:
    y, m = _year_month(month)
    y2, m2 = (y + 1, 1) if m == 12 else (y, m + 1)
    query = [
        ("station", station), ("data", "all"),
        ("year1", y), ("month1", m), ("day1", 1),
        ("year2", y2), ("month2", m2), ("day2", 1),  # the end is exclusive
        ("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "no"), ("elev", "no"),
        ("missing", "empty"), ("trace", "T"), ("direct", "no"),
        ("report_type", 3), ("report_type", 4),  # routine reports and SPECIs
    ]
    return f"{IEM_URL}?{urllib.parse.urlencode(query)}"


def ogimet_taf_url(station: str, month: str) -> str:
    y, m = _year_month(month)
    last = calendar.monthrange(y, m)[1]
    query = [
        ("begin", f"{y}{m:02d}010000"), ("end", f"{y}{m:02d}{last:02d}2359"),
        ("icao", station), ("lang", "eng"), ("header", "yes"),
    ]
    return f"{OGIMET_TAF_URL}?{urllib.parse.urlencode(query)}"


URLS = {"metar": iem_metar_url, "taf": ogimet_taf_url}


def http_get_bytes(
    url: str,
    *,
    timeout: float = 120.0,
    backoff: tuple[float, ...] = (5, 15, 45, 120),
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
) -> bytes:
    """GET as bytes, retrying server errors, throttling and dropped connections.

    ``data`` turns the request into a POST (Overpass takes its query as a form
    body); ``headers`` are merged over the default User-Agent. `http_get` is
    the text form; this one serves compressed downloads (the PRU's bzip2 CSVs).
    """
    request = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT, **(headers or {})})
    for wait in (*backoff, None):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:  # before URLError: it is a subclass
            if (exc.code < 500 and exc.code != 429) or wait is None:
                raise
            reason = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if wait is None:
                raise
            reason = repr(exc)
        print(f"    {reason}; retrying in {wait:.0f}s")
        time.sleep(wait)
    raise AssertionError("unreachable")


def http_get(
    url: str,
    *,
    timeout: float = 120.0,
    backoff: tuple[float, ...] = (5, 15, 45, 120),
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
) -> str:
    """GET as UTF-8 text (undecodable bytes replaced); see `http_get_bytes`."""
    return http_get_bytes(url, timeout=timeout, backoff=backoff, headers=headers, data=data).decode("utf-8", errors="replace")


def classify_iem(text: str) -> Literal["ok", "empty", "error"]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines or not lines[0].startswith("station,valid"):
        return "error"
    return "ok" if len(lines) > 1 else "empty"


def classify_ogimet(text: str) -> Literal["ok", "empty", "quota", "error"]:
    """Ogimet answers every query with HTTP 200; only the body says what happened.

    The header row is Spanish (``ESTACION,ANO,...``) even with ``lang=eng``.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if any("quota" in ln.lower() for ln in lines[:20]):
        return "quota"
    if any(_OGIMET_ROW.match(ln) for ln in lines):
        return "ok"
    if not lines or lines[0].upper().startswith(("ESTACION", "ICAOIND")) or all(ln.startswith("#") for ln in lines):
        return "empty"
    return "error"


def _get_checked(kind: str, url: str) -> str:
    classify = classify_iem if kind == "metar" else classify_ogimet
    for attempt in range(QUOTA_RETRIES + 1):
        text = http_get(url)
        status = classify(text)
        if status in ("ok", "empty"):
            return text
        if status == "quota" and attempt < QUOTA_RETRIES:
            print(f"    quota reached; sleeping {QUOTA_SLEEP_S // 60} min")
            time.sleep(QUOTA_SLEEP_S)
            continue
        if status == "quota":
            raise QuotaExceeded(f"still over quota at {url}\nrun `python -m prc fetch-weather --only {kind}` again later")
        raise RuntimeError(f"unexpected {kind} response from {url}:\n{text[:400]}")
    raise AssertionError("unreachable")


def _write_atomic(path: paths.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(text, encoding="utf-8", newline="")
    os.replace(part, path)


def _write_atomic_bytes(path: paths.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_bytes(payload)
    os.replace(part, path)


def fetch_chunks(kind: str, stations: list[str], months: list[str], *, force: bool = False) -> list[paths.Path]:
    total = len(stations) * len(months)
    todo = [(s, m) for s in stations for m in months if force or not paths.weather_raw_file(kind, s, m).exists()]
    print(f"{kind}: {total - len(todo)} of {total} chunks on disk, {len(todo)} to download")
    written = []
    for n, (station, month) in enumerate(todo, 1):
        if n > 1:
            time.sleep(DELAY_S[kind])
        url = URLS[kind](station, month)
        text = _get_checked(kind, url)
        path = paths.weather_raw_file(kind, station, month)
        _write_atomic(path, text)
        rows = max(sum(1 for ln in text.splitlines() if ln.strip()) - 1, 0)
        print(f"  [{n:>3}/{len(todo)}] {kind} {station} {month}: {rows:>5} rows")
        written.append(path)
    return written


def fetch_all(kinds: tuple[str, ...], *, force: bool = False) -> None:
    paths.ensure_dirs()
    mvt = data.load_movements(columns=["apt", "MVT_TIME_UTC_mvt"])
    stations, months = data.airports(mvt), needed_months(mvt)
    print(f"stations: {stations}\nmonths:   {months[0]} .. {months[-1]} ({len(months)})")
    for kind in kinds:
        fetch_chunks(kind, stations, months, force=force)
