"""External weather data: METAR observations (IEM) and TAF forecasts (Ogimet).

`python -m prc fetch-weather` downloads the archives once into
``data/additional_data/weather/raw/`` (one file per station-month) and
normalises them into the parquet tables the weather feature blocks read.

Feature code never imports this package. Blocks read only the parquets, and
those files' content digest is part of each block's cache key
(`prc.features.build.block_hash`) -- so a parser change reaches the feature
cache by re-normalising (`fetch-weather --offline`), with no ``version=`` bump.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from prc import paths

PARSER_VERSION = "wx-1"
KINDS = ("metar", "taf")


def _kinds(only: str | None) -> tuple[str, ...]:
    return (only,) if only else KINDS


def fetch(only: str | None = None, force: bool = False) -> None:
    """Download every station-month chunk not yet on disk (all of them with ``force``)."""
    from prc.weather.download import fetch_all

    fetch_all(_kinds(only), force=force)


def normalise(only: str | None = None) -> None:
    """Rebuild the parquet tables from the raw chunks. No network."""
    written: dict[str, paths.Path] = {}
    if "metar" in _kinds(only):
        from prc.weather.metar import normalise_metar

        written["metar"] = normalise_metar()
    if "taf" in _kinds(only):
        from prc.weather.taf import normalise_taf

        written.update(normalise_taf())
    _write_manifest(written)


def check(only: str | None = None) -> str:
    """Coverage and weather-regime report over what is on disk. No network."""
    from prc.weather.coverage import report

    return report(_kinds(only))


def _sha1(path: paths.Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(written: dict[str, paths.Path]) -> None:
    """Provenance only: which raw chunks and parser built which table.

    The feature cache does not read this -- it hashes the parquets themselves.
    """
    manifest = json.loads(paths.WEATHER_MANIFEST.read_text()) if paths.WEATHER_MANIFEST.exists() else {}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["parser_version"] = PARSER_VERSION
    raw = manifest.setdefault("raw", {})
    for f in sorted(paths.WEATHER_RAW.glob("*/*.csv")):
        st = f.stat()
        raw[f.relative_to(paths.WEATHER_RAW).as_posix()] = {
            "bytes": st.st_size,
            "fetched_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds"),
            "sha1": _sha1(f),
        }
    tables = manifest.setdefault("tables", {})
    for name, path in written.items():
        tables[name] = {"file": path.name, "sha1": _sha1(path), "built_utc": now, "parser_version": PARSER_VERSION}
    paths.WEATHER_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))
