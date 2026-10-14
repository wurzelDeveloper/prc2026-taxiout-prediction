"""External airport geometry: stands and taxiways from OpenStreetMap, runways from OurAirports.

`python -m prc fetch-geometry` downloads one OSM extract per airport and the
OurAirports tables into ``data/additional_data/airport_geometry/raw/``, then
normalises them into the parquet tables the geometry feature blocks read:
stand positions, runway thresholds, the stand-to-movement-ID match, and the
shortest taxiway route from every stand to every runway end.

Feature code never imports this package. Blocks read only the parquets, and
those files' content digest is part of each block's cache key
(`prc.features.build.block_hash`) -- so a change to the matching or routing
rules reaches the feature cache by re-normalising (`fetch-geometry
--offline`), with no ``version=`` bump.

Licences: OpenStreetMap data is (c) OpenStreetMap contributors, ODbL 1.0;
OurAirports data is public domain.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from prc import paths

PARSER_VERSION = "geo-1"
KINDS = ("ourairports", "osm")  # OurAirports first: the OSM bounding boxes come from its reference points


def _kinds(only: str | None) -> tuple[str, ...]:
    return (only,) if only else KINDS


def fetch(only: str | None = None, force: bool = False, radius_km: float | None = None) -> None:
    """Download every raw file not yet on disk (all of them with ``force``)."""
    from prc.geometry.download import fetch_all

    fetch_all(_kinds(only), force=force, radius_km=radius_km)


def normalise(stands_file: str | paths.Path | None = None) -> None:
    """Rebuild the parquet tables from the raw files (and the hand-placed stands, if any). No network."""
    from prc.geometry.normalise import normalise_geometry

    resolved = None if stands_file is None else (paths.ROOT / stands_file).resolve()
    _write_manifest(normalise_geometry(stands_file=resolved), stands_file=resolved)


def check() -> str:
    """Match, coverage and routing report over what is on disk. No network."""
    from prc.geometry.coverage import report

    return report()


def _sha1(path: paths.Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(written: dict[str, paths.Path], stands_file: paths.Path | None = None) -> None:
    """Provenance only: which raw files, parser and hand-placed stands built which table.

    The feature cache does not read this -- it hashes the parquets themselves.
    """
    manifest = json.loads(paths.GEOMETRY_MANIFEST.read_text()) if paths.GEOMETRY_MANIFEST.exists() else {}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["parser_version"] = PARSER_VERSION
    manifest["stands_file"] = None if stands_file is None else {"path": str(stands_file), "sha1": _sha1(stands_file)}
    raw = manifest.setdefault("raw", {})
    for f in sorted(p for p in paths.GEOMETRY_RAW.glob("*/*") if p.is_file()):
        st = f.stat()
        raw[f.relative_to(paths.GEOMETRY_RAW).as_posix()] = {
            "bytes": st.st_size,
            "fetched_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds"),
            "sha1": _sha1(f),
        }
    tables = manifest.setdefault("tables", {})
    for name, path in written.items():
        tables[name] = {"file": path.name, "sha1": _sha1(path), "built_utc": now, "parser_version": PARSER_VERSION}
    paths.GEOMETRY_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))
