"""Download the raw geometry: one Overpass extract per airport, two OurAirports CSVs.

Idempotent and resumable, like the weather download: a file is written
atomically and skipped on later runs unless ``force``. Overpass is a shared,
volunteer-run service -- the main instance answers HTTP 406 to requests
without a descriptive User-Agent and 429 to bursts, and the mirrors time out
under load -- so each airport is one request, endpoints are rotated on
failure, and a raw file is only written for a complete answer.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.parse
from typing import Literal

import pandas as pd

from prc import data, paths
from prc.weather.download import _write_atomic, http_get

OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)
OURAIRPORTS_URLS = {
    "airports.csv": "https://davidmegginson.github.io/ourairports-data/airports.csv",
    "runways.csv": "https://davidmegginson.github.io/ourairports-data/runways.csv",
}
USER_AGENT = "prc-challenge-2026-geometry/1.0 (taxi-out time research; python-urllib)"

DELAY_S = 5.0
OVERPASS_TIMEOUT_S = 180
DEFAULT_RADIUS_KM = 7.0
"""Half-size of the bounding box around the airport reference point. 7 km
covers every runway end of the ten airports (EHAM's Polderbaan is ~6 km from
the reference point) and keeps Le Bourget out of the LFPG box."""
BBOX_RADIUS_KM: dict[str, float] = {}
"""Per-airport overrides of the radius, by ICAO code."""

AEROWAY_NODES = ("parking_position", "gate", "holding_position")
AEROWAY_WAYS = ("parking_position", "taxiway", "taxilane", "runway", "holding_position")

KM_PER_DEG = math.pi / 180 * 6_371.0


class GeometryFetchError(RuntimeError):
    """Every Overpass endpoint refused or timed out; running the command again later resumes."""


def bbox(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    """``(south, west, north, east)`` of the square ``radius_km`` around a point."""
    dlat = radius_km / KM_PER_DEG
    dlon = radius_km / (KM_PER_DEG * math.cos(math.radians(lat)))
    return (lat - dlat, lon - dlon, lat + dlat, lon + dlon)


def overpass_query(south: float, west: float, north: float, east: float) -> str:
    """Stands, gates, holding positions and every taxiway/taxilane/runway way, with all their nodes.

    ``>;`` recurses from the ways to their nodes so way geometry arrives in the
    same answer; ``out skel qt`` prints those nodes without tags.
    """
    nodes = "|".join(AEROWAY_NODES)
    ways = "|".join(AEROWAY_WAYS)
    return (
        f"[out:json][timeout:{OVERPASS_TIMEOUT_S}][bbox:{south:.5f},{west:.5f},{north:.5f},{east:.5f}];"
        f'(node["aeroway"~"^({nodes})$"];way["aeroway"~"^({ways})$"];);'
        "out body;>;out skel qt;"
    )


def classify_overpass(text: str) -> Literal["ok", "empty", "timeout", "throttled", "error"]:
    """Overpass answers many failures with HTTP 200; the body says what happened."""
    stripped = text.lstrip()
    if not stripped.startswith("{"):
        low = stripped[:4000].lower()
        if "too many requests" in low or "rate_limited" in low or "rate limited" in low or "429" in low:
            return "throttled"
        return "error"
    try:
        payload = json.loads(text)
    except ValueError:
        return "error"
    remark = str(payload.get("remark", "")).lower()
    if "timed out" in remark or "out of memory" in remark or "runtime error" in remark:
        return "timeout"
    elements = payload.get("elements")
    if not isinstance(elements, list):
        return "error"
    return "ok" if elements else "empty"


def classify_ourairports(text: str, name: str) -> Literal["ok", "error"]:
    head = text.lstrip()[:200]
    want = '"id","airport_ref","airport_ident"' if name == "runways.csv" else '"id","ident","type"'
    return "ok" if head.startswith(want) else "error"


def fetch_ourairports(*, force: bool = False) -> list[paths.Path]:
    written = []
    for name, url in OURAIRPORTS_URLS.items():
        path = paths.geometry_raw_file("ourairports", name)
        if path.exists() and not force:
            print(f"ourairports: {name} on disk")
            continue
        text = http_get(url, headers={"User-Agent": USER_AGENT})
        if classify_ourairports(text, name) != "ok":
            raise RuntimeError(f"unexpected OurAirports response from {url}:\n{text[:400]}")
        _write_atomic(path, text)
        rows = sum(1 for ln in text.splitlines() if ln.strip()) - 1
        print(f"ourairports: {name}: {rows:,} rows")
        written.append(path)
    return written


def _overpass_once(endpoint: str, query: str) -> tuple[str, str]:
    """``(status, text)`` for one attempt at one endpoint; never raises for a refused request."""
    body = urllib.parse.urlencode({"data": query}).encode("utf-8")
    try:
        text = http_get(
            endpoint,
            timeout=OVERPASS_TIMEOUT_S + 30,
            backoff=(10, 30),
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            data=body,
        )
    except urllib.error.HTTPError as exc:
        return f"http {exc.code}", ""
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        return f"connection: {exc!r}", ""
    return classify_overpass(text), text


def fetch_osm(airports: pd.DataFrame, *, radius_km: float | None = None, force: bool = False) -> list[paths.Path]:
    """One JSON extract per airport (``apt``, ``arp_lat``, ``arp_lon`` columns)."""
    todo = [r for r in airports.itertuples(index=False)
            if force or not paths.geometry_raw_file("osm", f"{r.apt}.json").exists()]
    print(f"osm: {len(airports) - len(todo)} of {len(airports)} extracts on disk, {len(todo)} to download")
    written = []
    for n, r in enumerate(todo, 1):
        if n > 1:
            time.sleep(DELAY_S)
        radius = radius_km if radius_km is not None else BBOX_RADIUS_KM.get(r.apt, DEFAULT_RADIUS_KM)
        query = overpass_query(*bbox(float(r.arp_lat), float(r.arp_lon), radius))
        text = None
        for endpoint in OVERPASS_ENDPOINTS:
            status, body = _overpass_once(endpoint, query)
            if status in ("ok", "empty"):
                text = body
                break
            print(f"    {r.apt}: {endpoint.split('/')[2]} -> {status}; trying the next endpoint")
            time.sleep(DELAY_S)
        if text is None:
            raise GeometryFetchError(
                f"no Overpass endpoint answered for {r.apt}; run `python -m prc fetch-geometry --only osm` again later"
            )
        path = paths.geometry_raw_file("osm", f"{r.apt}.json")
        _write_atomic(path, text)
        n_el = len(json.loads(text).get("elements", []))
        print(f"  [{n:>2}/{len(todo)}] osm {r.apt} (radius {radius:.1f} km): {n_el:,} elements")
        written.append(path)
    return written


def fetch_all(kinds: tuple[str, ...], *, force: bool = False, radius_km: float | None = None) -> None:
    from prc.geometry.sources import load_ourairports_airports

    paths.ensure_dirs()
    icaos = data.airports(data.load_movements(columns=["apt"]))
    print(f"airports: {icaos}")
    if "ourairports" in kinds:
        fetch_ourairports(force=force)
    if "osm" in kinds:
        fetch_osm(load_ourairports_airports(icaos), radius_km=radius_km, force=force)
