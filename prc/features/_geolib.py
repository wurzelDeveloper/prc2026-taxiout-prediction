"""Geometry primitives for the feature blocks: distances, stand-ref matching keys, table loaders.

The leading underscore keeps this module out of block auto-import, but its
source is hashed into every block module that imports from it
(`build._related_modules`) -- so changing a normalisation rule here
invalidates exactly the geometry caches and nothing else. `prc.geometry`
imports from here too (the same rules must build the tables and read them),
which is the direction `prc.weather.coverage` already takes; feature code
never imports `prc.geometry`.

Coordinates are WGS84 degrees; every distance is in metres on a spherical
Earth, which is accurate to well under 1 % at airport scale.
"""

from __future__ import annotations

import functools
import re

import numpy as np
import pandas as pd

from prc import paths

EARTH_R_M = 6_371_000.0

# The same three-part shape `stand_runway_keys` parses ("A21L": pier, number,
# position). Applied to a cleaned, upper-case string.
_STAND = re.compile(r"^(?P<prefix>[A-Z]*)(?P<number>\d*)(?P<suffix>[A-Z]*)$")
_NON_ALNUM = re.compile(r"[^A-Z0-9]")
_RUNWAY = re.compile(r"^(\d{1,2})([LCR]?)$")
_MISSING = frozenset({"", "NAN", "<NA>", "NA", "NONE", "NULL", "__NA__"})


# --- distances -----------------------------------------------------------------

def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in metres between two points (or arrays of points)."""
    lat1, lon1, lat2, lon2 = (np.deg2rad(np.asarray(x, dtype="float64")) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing_deg(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Initial bearing from point 1 to point 2, degrees clockwise from north in [0, 360)."""
    lat1, lon1, lat2, lon2 = (np.deg2rad(np.asarray(x, dtype="float64")) for x in (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.mod(np.rad2deg(np.arctan2(x, y)), 360.0)


def local_east_north_m(lat, lon, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    """Metres east and north of ``(lat0, lon0)``: an equirectangular projection about that point."""
    lat = np.asarray(lat, dtype="float64")
    lon = np.asarray(lon, dtype="float64")
    east = np.deg2rad(lon - lon0) * EARTH_R_M * np.cos(np.deg2rad(lat0))
    north = np.deg2rad(lat - lat0) * EARTH_R_M
    return east, north


# --- identifiers ---------------------------------------------------------------

def clean_ref(raw) -> str:
    """Upper-case with everything but letters and digits removed; ``""`` for any missing value."""
    if raw is None:
        return ""
    if isinstance(raw, float) and np.isnan(raw):
        return ""
    s = str(raw).upper().strip()
    if s in _MISSING:
        return ""
    return _NON_ALNUM.sub("", s)


def norm_stand_ref(raw) -> tuple[str, str, str] | None:
    """``(prefix, number, suffix)`` of a stand reference, or None if it is missing.

    The number loses its leading zeros ("A021" and "A21" are the same stand);
    a string the pattern cannot parse is kept whole as the prefix so it still
    matches itself exactly.
    """
    s = clean_ref(raw)
    if not s:
        return None
    m = _STAND.match(s)
    if not m:
        return (s, "", "")
    number = m.group("number")
    if number:
        number = number.lstrip("0") or "0"
    return (m.group("prefix"), number, m.group("suffix"))


def stand_keys(raw) -> tuple[str | None, str | None, str | None]:
    """The lookup keys tried in order when matching a stand: ``(exact, norm, nosuffix)``.

    ``exact`` is the cleaned string, ``norm`` the normalised prefix+number+suffix,
    ``nosuffix`` prefix+number (None when there is no number to anchor it).
    """
    parsed = norm_stand_ref(raw)
    if parsed is None:
        return (None, None, None)
    prefix, number, suffix = parsed
    return (clean_ref(raw), prefix + number + suffix, (prefix + number) if number else None)


def norm_runway_ident(raw) -> str | None:
    """A runway designator as ``NN[LCR]``: "7L" -> "07L", "25c" -> "25C"; None for "NA", "H", "HEL", missing."""
    s = clean_ref(raw)
    m = _RUNWAY.match(s)
    if not m:
        return None
    n = int(m.group(1))
    if not 1 <= n <= 36:
        return None
    return f"{n:02d}{m.group(2)}"


def runway_idents(ref) -> tuple[str, ...]:
    """The normalised designators named by a runway ``ref`` such as "07C/25C" (sorted, deduplicated)."""
    if ref is None or (isinstance(ref, float) and np.isnan(ref)):
        return ()
    parts = re.split(r"[^A-Za-z0-9]+", str(ref))
    found = {i for i in (norm_runway_ident(p) for p in parts if p) if i is not None}
    return tuple(sorted(found))


def runway_key(idents: tuple[str, ...]) -> str:
    """One identity per physical runway: its designators joined, e.g. "07C/25C"."""
    return "/".join(idents)


# --- the normalised tables ------------------------------------------------------

@functools.lru_cache(maxsize=8)
def _table(path: str, mtime_ns: int) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_table(path: paths.Path) -> pd.DataFrame:
    """A geometry table, memoised on (path, mtime). Callers must not mutate it."""
    return _table(str(path), path.stat().st_mtime_ns)


def load_airports() -> pd.DataFrame:
    return load_table(paths.GEO_AIRPORTS_FILE)


def load_runway_ends() -> pd.DataFrame:
    return load_table(paths.GEO_RUNWAY_ENDS_FILE)


def load_stands() -> pd.DataFrame:
    return load_table(paths.GEO_STANDS_FILE)


def load_stand_matches() -> pd.DataFrame:
    return load_table(paths.GEO_STAND_MATCHES_FILE)


def load_routes() -> pd.DataFrame:
    return load_table(paths.GEO_ROUTES_FILE)
