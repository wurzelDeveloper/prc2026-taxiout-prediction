"""ERA5 reanalysis per airport and hour, from the Open-Meteo historical weather API.

`python -m prc fetch-era5` downloads one JSON answer per airport from
https://archive-api.open-meteo.com/v1/archive (no key; ERA5 and ERA5-Land,
hourly, UTC) into ``data/additional_data/reanalysis/raw/`` and normalises them
into ``era5_hourly.parquet``. It fills what the METARs lack: the IEM archive
strips the runway-state groups, so no METAR says how much snow lies on the
ground, and none says how cold the pavement is. January 2026 had several times
the snow reports of any 2025 month at EHAM, EDDF, EDDM, LSZH and LTFM, so
accumulated snow, recent snowfall and overnight frost are conditions the 2026
ranking months hold and the Jan+Jul 2025 holdout does not.

The airport coordinates are the OurAirports reference points of the geometry
tables, so `fetch-geometry` runs first. The grid cell is ~9 km (ERA5-Land) to
~25 km (ERA5): a regional state, not the apron's. Feature code never imports this
package; the `era5_surface` block reads the parquet only.

Attribution: Open-Meteo.com (CC BY 4.0), with data from ERA5 / ERA5-Land,
Copernicus Climate Change Service.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from prc import paths

PARSER_VERSION = "era5-1"
START = "2024-12-24"  # a week before the training year, for the 24 h and overnight look-backs
END = "2026-07-31"
VARIABLES = (
    "temperature_2m",
    "dew_point_2m",
    "precipitation",
    "rain",
    "snowfall",
    "snow_depth",
    "weather_code",
    "cloud_cover_low",
    "wind_speed_10m",
    "wind_gusts_10m",
    "soil_temperature_0_to_7cm",
)


def fetch(force: bool = False) -> None:
    """Download every airport not yet on disk (all of them with ``force``)."""
    from prc.reanalysis.download import fetch_all

    fetch_all(force=force)


def normalise() -> None:
    """Rebuild ``era5_hourly.parquet`` from the raw JSON answers. No network."""
    from prc.reanalysis.normalise import normalise_era5

    _write_manifest(normalise_era5())


def check() -> str:
    """Coverage report over what is on disk. No network."""
    from prc.reanalysis.normalise import coverage_report

    return coverage_report()


def _sha1(path: paths.Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(written: paths.Path) -> None:
    """Provenance only: which raw files and parser built the table (the feature cache hashes the parquet)."""
    manifest = {
        "parser_version": PARSER_VERSION,
        "source": "Open-Meteo historical weather API (ERA5, ERA5-Land), https://archive-api.open-meteo.com/v1/archive, CC BY 4.0",
        "period": [START, END],
        "variables": list(VARIABLES),
        "raw": {f.name: {"bytes": f.stat().st_size, "sha1": _sha1(f)} for f in sorted(paths.REANALYSIS_RAW.glob("*.json"))},
        "table": {"file": written.name, "sha1": _sha1(written), "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
    }
    paths.REANALYSIS_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))
