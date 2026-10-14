"""Stream one day of the adsb.lol globe history and keep what happens at the ten airports.

adsb.lol publishes one GitHub release per day (``adsblol/globe_history_<year>``, tag
``v<YYYY.MM.DD>-planes-readsb-prod-0``) as a tar split into ~2 GB parts. The tar holds one
gzip-compressed readsb ``trace_full_<icao>.json`` per aircraft: a base timestamp and a list of
points ``[dt, lat, lon, alt, gs, track, flags, vrate, details, ...]`` where ``alt`` is feet or the
string ``"ground"`` and ``details`` (occasionally a dict) carries the callsign as ``flight``.

The parts are streamed over HTTPS and concatenated into one tar stream (``tarfile`` stream mode),
so nothing but the filtered points touches the disk: ~3-4 GB per day in, a few MB out. A point is
kept when it lies inside an airport's box (+-0.10 deg latitude, +-0.15 deg longitude around its
reference point) at or below 5,000 ft (or on the ground). The callsign is carried forward along the
trace, so every kept point knows the flight it belonged to at that moment.

Data: adsb.lol contributors, Open Database License (ODbL 1.0).
"""

from __future__ import annotations

import gzip
import io
import json
import zlib

import orjson
import tarfile
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

BOX_LAT = 0.10
BOX_LON = 0.15
MAX_ALT_FT = 5_000
USER_AGENT = "prc-taxiout-research (github.com/wurzelDeveloper/prc2026-taxiout-prediction)"


@dataclass(frozen=True)
class Box:
    apt: str
    lat: float
    lon: float


def release_parts(day: pd.Timestamp, session: requests.Session, kind: str = "prod") -> list[str]:
    """Download URLs of the day's tar parts, in order. `kind`: "prod" (default) or "staging" (a second aggregator
    instance adsb.lol publishes for some days, same licence)."""
    tag = f"v{day:%Y.%m.%d}-planes-readsb-{kind}-0"
    repo = f"adsblol/globe_history_{day.year}"
    r = session.get(f"https://api.github.com/repos/{repo}/releases/tags/{tag}", timeout=60)
    if r.ok:
        assets = sorted(r.json()["assets"], key=lambda a: a["name"])
        return [a["browser_download_url"] for a in assets if ".tar" in a["name"]]
    if r.status_code == 404:
        raise FileNotFoundError(f"no release {tag}")
    # the API refused (anonymous rate limit): the parts follow the pattern <tag>.tar.aa, .ab, ... on the download host
    urls = []
    for i in range(26):
        url = f"https://github.com/{repo}/releases/download/{tag}/{tag}.tar.a{chr(ord('a') + i)}"
        if session.head(url, allow_redirects=True, timeout=60).status_code != 200:
            break
        urls.append(url)
    if not urls:
        raise FileNotFoundError(f"no release parts for {tag}")
    return urls


class _ConcatStream(io.RawIOBase):
    """A readable stream over several HTTP responses, one after the other.

    A read error mid-asset (a timeout or a reset connection on GitHub's asset host) reopens the same asset with an
    HTTP Range request at the byte offset already read, so the tar stream continues where it stopped."""

    def __init__(self, urls: list[str], session: requests.Session):
        self.urls = list(urls)
        self.session = session
        self.resp = None
        self.url = None
        self.offset = 0
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def _open(self, url: str, offset: int) -> None:
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        for attempt in range(8):
            try:
                self.resp = self.session.get(url, stream=True, timeout=(30, 300), headers=headers)
                self.resp.raise_for_status()
                self.resp.raw.decode_content = False
                return
            except requests.RequestException:
                time.sleep(15 * (attempt + 1))
        raise RuntimeError(f"could not open {url} at {offset}")

    def _next(self) -> bool:
        if self.resp is not None:
            self.resp.close()
        if not self.urls:
            self.resp = None
            return False
        self.url = self.urls.pop(0)
        self.offset = 0
        self._open(self.url, 0)
        return True

    def readinto(self, b) -> int:
        while True:
            if self.resp is None and not self._next():
                return 0
            try:
                n = self.resp.raw.readinto(b)
            except Exception:  # noqa: BLE001 - urllib3 timeouts/resets: resume the asset with a Range request
                self.resp.close()
                time.sleep(10)
                self._open(self.url, self.offset)
                continue
            if n:
                self.bytes_read += n
                self.offset += n
                return n
            if not self._next():
                return 0


def _points(trace: dict, boxes: list[Box]) -> list[tuple]:
    """The kept points of one aircraft trace, as tuples (vectorised filter per trace)."""
    pts = trace.get("trace") or []
    if not pts:
        return []
    lat = np.fromiter((p[1] if p[1] is not None else np.nan for p in pts), dtype="float64", count=len(pts))
    lon = np.fromiter((p[2] if p[2] is not None else np.nan for p in pts), dtype="float64", count=len(pts))
    near = np.zeros(len(pts), dtype="int64") - 1
    for k, b in enumerate(boxes):
        hit = (np.abs(lat - b.lat) <= BOX_LAT) & (np.abs(lon - b.lon) <= BOX_LON)
        near[hit & (near < 0)] = k
    idx = np.nonzero(near >= 0)[0]
    if not len(idx):
        return []
    base = float(trace.get("timestamp", 0.0))
    icao = trace.get("icao", "")
    # the callsign in force at each point: carried forward from the last details dict that named one
    callsign = None
    calls = [None] * len(pts)
    for i, p in enumerate(pts):
        if len(p) > 8 and isinstance(p[8], dict):
            flight = p[8].get("flight")
            if flight:
                callsign = flight.strip()
        calls[i] = callsign
    out = []
    for i in idx:
        p = pts[i]
        alt = p[3]
        on_ground = alt == "ground"
        if not on_ground and (alt is None or alt > MAX_ALT_FT):
            continue
        out.append((
            icao, base + float(p[0]), float(lat[i]), float(lon[i]),
            -1.0 if on_ground else float(alt), on_ground,
            float(p[4]) if p[4] is not None else np.nan,
            float(p[5]) if p[5] is not None else np.nan,
            calls[i], boxes[near[i]].apt,
        ))
    return out


COLUMNS = ("icao", "t", "lat", "lon", "alt_ft", "on_ground", "gs_kt", "track", "callsign", "apt")


def stream_day(day: pd.Timestamp, boxes: list[Box], max_members: int | None = None, log=print,
               kind: str = "prod") -> pd.DataFrame:
    """All kept points of one day as a frame (see `COLUMNS`)."""
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    urls = release_parts(day, session, kind)
    raw = _ConcatStream(urls, session)
    buf = io.BufferedReader(raw, buffer_size=4 * 1024 * 1024)
    rows: list[tuple] = []
    t0 = time.time()
    try:
        _read_tar(tarfile.open(fileobj=buf, mode="r|"), day, boxes, rows, raw, log, max_members, t0)
    except tarfile.ReadError as exc:  # the published archive itself ends early: keep what was read before
        log(f"  {day:%Y-%m-%d}: archive truncated ({exc}); keeping {len(rows):,} points read before")
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    frame["t"] = frame["t"].astype("float64")
    for c in ("lat", "lon"):
        frame[c] = frame[c].astype("float64")
    for c in ("alt_ft", "gs_kt", "track"):
        frame[c] = frame[c].astype("float32")
    frame["apt"] = frame["apt"].astype("category")
    log(f"  {day:%Y-%m-%d}: done, {raw.bytes_read / 1e9:.2f} GB read, {len(frame):,} points kept, {time.time() - t0:.0f}s")
    return frame


def _read_tar(tar, day, boxes, rows, raw, log, max_members, t0):
    """Append the kept points of every aircraft trace in the tar stream to `rows`."""
    n = 0
    with tar:
        for member in tar:
            if not member.isfile() or "trace_full" not in member.name:
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            data = f.read()
            try:
                trace = orjson.loads(gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data)
            except (OSError, ValueError, EOFError, zlib.error):  # a truncated or corrupt member: skip that aircraft
                continue
            rows.extend(_points(trace, boxes))
            n += 1
            if n % 20_000 == 0:
                log(f"  {day:%Y-%m-%d}: {n:,} traces, {raw.bytes_read / 1e9:.2f} GB, {len(rows):,} points, {time.time() - t0:.0f}s")
            if max_members and n >= max_members:
                break
