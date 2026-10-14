"""External open data from Eurocontrol's Performance Review Unit (PRU).

`python -m prc fetch-pru` downloads four open CSV datasets from
https://ansperformance.eu/csv/ (one file per dataset and year) into
``data/additional_data/pru/raw/`` and normalises them into two parquet tables:

* ``pru_daily.parquet`` -- per airport and day: ATC pre-departure delays (the
  PRU's two computations), all pre-departure delays, and airport arrival ATFM
  delays. Read by the `pru_daily` feature block.
* ``pru_monthly.parquet`` -- per airport and month: the *additional taxi-out
  time* indicator, whose reference plus additional minutes over the referenced
  flights is the month's mean taxi-out. It reproduces the training data's
  monthly means within a few seconds, and it exists for the 2026 ranking
  months too. It is **not** a feature: an operational model would not have the
  answer's monthly mean, so it serves `prc levelcheck` only, as a sanity check
  of a run's predicted level per airport-month.

The 2026 files are republished monthly; ``fetch-pru --force`` refreshes them,
and the parquets' content digest then invalidates exactly the `pru_*` feature
caches. Feature code never imports this package (it is not hashed).

Attribution: EUROCONTROL Performance Review Unit open data, ansperformance.eu.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from prc import paths

PARSER_VERSION = "pru-1"
KINDS = ("taxi_out_additional_time", "atc_pre_departure_delays", "all_pre_departure_delays", "apt_dly")
YEARS = (2025, 2026)


def _kinds(only: str | None) -> tuple[str, ...]:
    if only is None:
        return KINDS
    if only not in KINDS:
        raise ValueError(f"unknown PRU dataset '{only}' (one of {KINDS})")
    return (only,)


def fetch(only: str | None = None, force: bool = False) -> None:
    """Download every dataset-year not yet on disk (all of them with ``force``)."""
    from prc.pru.download import fetch_all

    fetch_all(_kinds(only), YEARS, force=force)


def normalise() -> None:
    """Rebuild the two parquet tables from the raw CSVs. No network."""
    from prc.pru.normalise import normalise_pru

    _write_manifest(normalise_pru())


def check() -> str:
    """Coverage report over what is on disk. No network."""
    from prc.pru.coverage import report

    return report()


def _sha1(path: paths.Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(written: dict[str, paths.Path]) -> None:
    """Provenance only: which raw files and parser built which table (the feature cache hashes the parquets)."""
    manifest = json.loads(paths.PRU_MANIFEST.read_text()) if paths.PRU_MANIFEST.exists() else {}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["parser_version"] = PARSER_VERSION
    manifest["source"] = "EUROCONTROL Performance Review Unit open data, https://ansperformance.eu/csv/"
    raw = manifest.setdefault("raw", {})
    for f in sorted(paths.PRU_RAW.glob("*.csv*")):
        st = f.stat()
        raw[f.name] = {
            "bytes": st.st_size,
            "fetched_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds"),
            "sha1": _sha1(f),
        }
    tables = manifest.setdefault("tables", {})
    for name, path in written.items():
        tables[name] = {"file": path.name, "sha1": _sha1(path), "built_utc": now, "parser_version": PARSER_VERSION}
    paths.PRU_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))
