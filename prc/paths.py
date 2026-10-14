"""All filesystem locations in one place.

Nothing else in the codebase should build a path from string literals.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- raw, immutable inputs -------------------------------------------------
# PRC_DATA_DIR points the whole data tree (competition files, interim, additional
# data) at another directory, for experiments on an edited copy of the inputs.
# Give such a config its own `data.version`, because the feature cache and the
# runs stay shared and are partitioned by that key only.
DATA = Path(os.environ["PRC_DATA_DIR"]).resolve() if os.environ.get("PRC_DATA_DIR") else ROOT / "data"
COMPETITION = DATA / "competition_files"
ADDITIONAL = DATA / "additional_data"
WEATHER = ADDITIONAL / "weather"
WEATHER_RAW = WEATHER / "raw"  # downloaded METAR/TAF chunks, see `weather_raw_file`
GEOMETRY = ADDITIONAL / "airport_geometry"
GEOMETRY_RAW = GEOMETRY / "raw"  # downloaded OSM extracts and OurAirports tables, see `geometry_raw_file`
PRU = ADDITIONAL / "pru"
PRU_RAW = PRU / "raw"  # downloaded Eurocontrol PRU open CSVs, one per dataset-year, see `pru_raw_file`
REANALYSIS = ADDITIONAL / "reanalysis"
REANALYSIS_RAW = REANALYSIS / "raw"  # Open-Meteo ERA5 hourly JSON, one file per airport, see `era5_raw_file`
ADSB = ADDITIONAL / "adsb"
ADSB_POINTS = ADSB / "points"  # one parquet per day: adsb.lol points inside the airport boxes (`fetch-adsb`)
ADSB_POINTS_PROD = ADSB / "points_prod"  # prod-only originals of the days merged with staging (Jan+Jul 2025/2026)

RANKING_FILE = COMPETITION / "ranking.parquet"
SUBMITTING_FILE = COMPETITION / "submitting.parquet"
TRAINING_GLOB = "training_*.parquet"

# --- derived -----------------------------------------------------------------
INTERIM = DATA / "interim"
MOVEMENTS_FILE = INTERIM / "movements.parquet"
TARGET_FILE = INTERIM / "target.parquet"
PREPARE_RECORD = INTERIM / "prepare.json"  # `prepare`: the sha1 of every input file that built the two above

# Normalised weather tables, rebuilt from WEATHER_RAW by `fetch-weather --offline`.
METAR_FILE = WEATHER / "metar.parquet"
TAF_ISSUES_FILE = WEATHER / "taf_issues.parquet"
TAF_SEGMENTS_FILE = WEATHER / "taf_segments.parquet"
WEATHER_MANIFEST = WEATHER / "manifest.json"

# Normalised airport geometry tables, rebuilt from GEOMETRY_RAW by `fetch-geometry --offline`.
GEO_AIRPORTS_FILE = GEOMETRY / "airports.parquet"
GEO_RUNWAY_ENDS_FILE = GEOMETRY / "runway_ends.parquet"
GEO_STANDS_FILE = GEOMETRY / "stands.parquet"
GEO_STAND_MATCHES_FILE = GEOMETRY / "stand_matches.parquet"
GEO_ROUTES_FILE = GEOMETRY / "stand_runway_routes.parquet"
GEO_STAND_REPORT_FILE = GEOMETRY / "stand_match_report.csv"  # human-readable: every movement stand, matched or not
GEOMETRY_MANIFEST = GEOMETRY / "manifest.json"

# Normalised PRU tables, rebuilt from PRU_RAW by `fetch-pru --offline`.
PRU_DAILY_FILE = PRU / "pru_daily.parquet"      # per (airport, day): pre-departure and airport ATFM delays
PRU_MONTHLY_FILE = PRU / "pru_monthly.parquet"  # per (airport, month): additional taxi-out time (`prc levelcheck` only)
PRU_MANIFEST = PRU / "manifest.json"

# Normalised ERA5 reanalysis, rebuilt from REANALYSIS_RAW by `fetch-era5 --offline`.
ERA5_FILE = REANALYSIS / "era5_hourly.parquet"  # per (airport, hour): snow depth, snowfall, precipitation, temperatures
REANALYSIS_MANIFEST = REANALYSIS / "manifest.json"

CACHE = ROOT / "cache"
FEATURE_CACHE = CACHE / "features"

# The lab's per-departure ADS-B tables (links_<day>, features_<day>, features2_<day>, features3.parquet; lab/adsb_*.py),
# the per-flight off-block observations of the stand archive (lab/ideas/stand_archive.py pass 1, one parquet per day),
# and the archive input consolidated from them by `python -m prc.adsb.archive` (block op_adsb_archive, external).
ADSB_TABLES = CACHE / "adsb"
ADSB_STAND_OBS = CACHE / "lab" / "ideas" / "stand_archive_obs"
ADSB_ARCHIVE_FILE = CACHE / "adsb_archive" / "adsb_archive_flights.parquet"
ADSB_ARCHIVE_MANIFEST = CACHE / "adsb_archive" / "manifest.json"

# The ADS-B surface picture (`prc adsb-surface`, prc/adsb/surface.py): per snapshot, one event table per UTC day and
# one consolidated table that `op_h1_adsb` / `op_h0_adsb` read as their external input. A snapshot is a frozen,
# versioned build (findings 10.31, critique F9): experiments read the snapshot named by PRC_ADSB_SNAPSHOT (default
# "s1" = every downloaded day, 2025 and Jan-Jul 2026, so the final set's Jan, Feb, Jun and Jul 2026 are in it). It sits
# under cache/ (follows the checkout, not PRC_DATA_DIR); the block's cache key holds the table's content digest.
ADSB_SURFACE = CACHE / "adsb_surface"
ADSB_SURFACE_SNAPSHOT = os.environ.get("PRC_ADSB_SNAPSHOT") or "s1"
ADSB_SURFACE_EVENTS_NAME = "surface_events.parquet"
ADSB_SURFACE_EVENTS_FILE = ADSB_SURFACE / ADSB_SURFACE_SNAPSHOT / ADSB_SURFACE_EVENTS_NAME
# The snapshot's frozen airport x family support (`prc adsb-surface --support`): where, and in which reception regime,
# each column family held on the 2025 training departures; the blocks NaN a family outside it (an external, too).
ADSB_SURFACE_SUPPORT_NAME = "support.parquet"
ADSB_SURFACE_SUPPORT_FILE = ADSB_SURFACE / ADSB_SURFACE_SNAPSHOT / ADSB_SURFACE_SUPPORT_NAME

CONFIGS = ROOT / "configs"
GENERATED_CONFIGS = CONFIGS / "_generated"

RUNS = ROOT / "runs"
RUN_INDEX = RUNS / "index.csv"
FEATURE_EFFECTS = RUNS / "feature_effects.csv"

SUBMISSIONS = ROOT / "submissions"


def training_files() -> list[Path]:
    """The monthly training parquets, in chronological (filename) order."""
    return sorted(COMPETITION.glob(TRAINING_GLOB))


def same_dir(a: Path | str, b: Path | str) -> bool:
    """Whether two spellings name the same directory (resolved, case-folded the way the OS compares paths)."""
    return os.path.normcase(str(Path(a).resolve())) == os.path.normcase(str(Path(b).resolve()))


def feature_cache_dir(data_version: str, fold: str | None = None) -> Path:
    """Cache directory for one data version, optionally scoped to a CV fold.

    Fold-scoped subdirectories hold blocks with ``uses_target=True``, which must
    be recomputed per fold from training rows only.
    """
    d = FEATURE_CACHE / data_version
    if fold is not None:
        d = d / f"fold_{fold}"
    return d


def run_dir(run_id: str) -> Path:
    return RUNS / run_id


def weather_raw_file(kind: str, station: str, month: str) -> Path:
    """One downloaded chunk: ``kind`` is ``metar`` or ``taf``, ``month`` is ``YYYY-MM``."""
    return WEATHER_RAW / kind / f"{station}_{month}.csv"


def geometry_raw_file(kind: str, name: str) -> Path:
    """One downloaded file: ``kind`` is ``osm`` (``<ICAO>.json``) or ``ourairports`` (``runways.csv``)."""
    return GEOMETRY_RAW / kind / name


def pru_raw_file(kind: str, year: int) -> Path:
    """One downloaded PRU dataset-year: ``kind`` is the dataset group (``apt_dly`` is served bzip2-compressed)."""
    suffix = ".csv.bz2" if kind == "apt_dly" else ".csv"
    return PRU_RAW / f"{kind}_{year}{suffix}"


def era5_raw_file(apt: str) -> Path:
    """One downloaded Open-Meteo archive answer: every hourly variable for one airport over the whole period."""
    return REANALYSIS_RAW / f"{apt}.json"


def adsb_surface_dir(snapshot: str | None = None) -> Path:
    """One ADS-B surface snapshot: ``days/<day>.parquet`` (events), ``qa/<day>.parquet``, the consolidated table."""
    return ADSB_SURFACE / (snapshot or ADSB_SURFACE_SNAPSHOT)


def ensure_dirs() -> None:
    """Create every directory the pipeline writes to."""
    for d in (INTERIM, FEATURE_CACHE, RUNS, SUBMISSIONS, GENERATED_CONFIGS, ADDITIONAL, WEATHER_RAW, GEOMETRY_RAW, PRU_RAW, REANALYSIS_RAW):
        d.mkdir(parents=True, exist_ok=True)
