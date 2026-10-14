"""Lab: the data tree for the final ranking set, a check of the released files, and the runbook for predicting it.

The final submission is scored on a ranking set the organisers released after training ended: `final_ranking.parquet`
of 8 Oct 2026 (January, February, June and July 2026, ``ranking.parquet``'s columns, the recorded times jittered by up
to 12 s; findings 10.33) with a new template, `final_submitting.parquet`. The trained
runs stay where they are; the new data goes into a data tree of its own, selected with ``PRC_DATA_DIR`` and given its
own ``data.version``, so the original tree, its feature cache and the runs are never touched, and `prc predict` applies
the runs' refit models to it without retraining.

Subcommands:

  init --tree DIR --source DIR [--dry-run] [--force]
      Creates DIR/competition_files, DIR/additional_data and DIR/interim and copies from the source tree (the data tree
      the runs were trained on) what predicting needs besides the released files: the twelve training parquets, the raw
      weather tree (METAR and TAF, every station-month on disk -- here 2024-12 to 2026-08, so Mar-May 2026 included;
      ``fetch-weather --offline`` rebuilds the tables), the raw and normalised airport geometry, the ERA5 raw and
      normalised files, and the PRU tables (``prc levelcheck``). Never ADS-B (not read by the harness) and never the
      old ranking file or template. Every file goes into DIR/final_inputs.json with its size and sha1; a file is copied
      to ``<name>.part`` and moved into place, identical files are skipped, so a re-run completes an interrupted copy,
      and a differing file is only replaced with --force.
  validate --file RANKING [--template T] [--months 2026-02 ...] [--training-dir DIR] [--reference PARQUET]
           [--baseline PARQUET] [--tree DIR] [--install [--force]] [--content-as-warnings]
      Checks the released ranking parquet before anything is built on it. Structure: columns and types equal
      ranking.parquet's, ids integral, unique and disjoint from the training ids, takeoff/landing times inside the
      expected months, departures and arrivals present, every reporting airport one of the training airports. Content,
      against the training files and the delivered Jan+Jul ranking file (--baseline): the share of empty values per
      column the members read (an NM time column, stand, runway or aircraft type emptied, arrivals without taxi-in
      times), labels of the low-cardinality columns and the runway designators per airport the training data never
      had (a label spelled differently turns a whole model column unseen), the share of the stale-suspect segment, the
      first hours whose look-back finds no earlier rows and days without rows. With --tree, the inputs the new months
      need: raw METAR/TAF station-months and ERA5 hours. With --template, its columns and types and that its ids are
      the ranking departures. Exit code 1 on any error (--content-as-warnings demotes the content findings, never the
      structural ones). --install then copies both into DIR/competition_files (ranking.parquet, submitting.parquet),
      only into a tree `init` made, never into the repository's data/ or the source tree.
  runbook --tree DIR (--from-blend RUN | --runs RUN ... [--weights W ...]) [--out FILE] [--shell ps1|sh] [--rehearse]
          [--version-prefix P] [--threads N] [--no-stack] [--allow-warnings] [--months ...]
      Writes a self-contained script (PowerShell, or bash for a detached chain) that predicts the final set from
      finished runs: it sets and restores PRC_DATA_DIR itself, stops at the first command that fails, and re-uses
      what an interrupted earlier start already made. In order: init, validate --install, the data.version derived from
      the installed ranking file (a re-release gets its own version and cache), prepare --force, the offline
      re-normalisations and coverage checks, prc predict per trained run (features first, every one with --tree), the
      blend(s) with the source's weights (always explicit) and clip flags, lab/final_stack.py --inwindow 0.5, submit and
      levelcheck. --from-blend expands a finished `prc blend` run (nested blends included) into its trained runs; a run
      directory that is missing is an error, and so is any warning (no refit model, a stack over non-mixture runs)
      unless --allow-warnings. --rehearse writes the rehearsal instead: the delivered Jan+Jul files into a tree of
      their own, and at the end `check` against the source blend and stack, which must come out identical.
  derive version|rank-rows --tree DIR [--prefix P]
      Prints, for the ranking file installed in DIR, the data.version the runbook uses (``<prefix>_<sha1[:8]>``) or its
      row count (PRC_EXPECTED_RANK_ROWS).
  latest NAME [--source RUN] [--data-version V] [--blend-of RUN ... [--blend-weights W ...]] [--stack-of RUN
         [--inwindow F]] [--missing-ok]
      Prints the id of the newest run named NAME that is what the checks say (a `prc predict` of that source under that
      version, a blend of exactly these runs, a stack of that blend); nothing (exit 0) with --missing-ok when there is
      none, else exit 1.
  check --members RUN ... [--blend RUN --source-blend RUN] [--stack RUN --source-stack RUN]
      The rehearsal's verdict: every predict run reproduced its source exactly, nothing was rescaled, and the blend and
      the stack equal the source's. Exit code 1 otherwise.

Usage: python lab/final_inputs.py <subcommand> ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prc import data, paths  # noqa: E402

FINAL_MONTHS = ("2026-01", "2026-02", "2026-06", "2026-07")
"""The final release (8 Oct 2026, findings 10.33): January, February, June and July 2026, the times jittered."""
"""The months the final ranking set is expected to cover (``validate --months`` overrides)."""
REHEARSAL_MONTHS = ("2026-01", "2026-07")
"""The delivered ranking set's months, for the rehearsal."""
DEFAULT_PREFIX = "final26"
"""The runbook's data.version is ``<prefix>_<sha1 of the installed ranking file>``: anything but the runs' own (``v1``),
and a new one whenever the file changes."""
REHEARSAL_PREFIX = "rehearse"
MANIFEST = "final_inputs.json"
STACK_FACTOR = 0.5
"""``lab/final_stack.py --inwindow``: the submitted stack's recipe (findings 10.7)."""
DEFAULT_THREADS = 4
"""CatBoost prediction threads the runbook passes (the machine is shared with a GPU chain and its owner)."""

RANKING_SCHEMA: tuple[tuple[str, str], ...] = (
    ("MVT_ID_mvt", "double"), ("FLIGHT_ID_mvt", "double"), ("FLIGHT_mvt", "string"), ("FLIGHT_RULE_mvt", "string"),
    ("ADEP_mvt", "string"), ("ADES_mvt", "string"), ("PHASE_mvt", "string"),
    ("MVT_TIME_UTC_mvt", "timestamp[us, tz=UTC]"), ("BLOCK_TIME_UTC_mvt", "timestamp[us, tz=UTC]"),
    ("SCHED_TIME_UTC_mvt", "timestamp[us, tz=UTC]"), ("AIRCRAFT_TYPE_mvt", "string"), ("RUNWAY_mvt", "string"),
    ("STAND_mvt", "string"), ("TAXITIME_SEC_mvt", "int32"), ("LOBT_flt", "timestamp[us, tz=UTC]"),
    ("CALLSIGN_flt", "string"), ("ADEP_flt", "string"), ("ADES_flt", "string"), ("ADES_FILED_flt", "string"),
    ("MARKET_SEGMENT_flt", "string"), ("IOBT_flt", "timestamp[us, tz=UTC]"), ("FLIGHT_RULE_flt", "string"),
    ("FLIGHT_TYPE_flt", "string"), ("AIRCRAFT_TYPE_flt", "string"), ("WK_TBL_CAT_flt", "string"),
    ("AIRCRAFT_OPERATOR_flt", "string"), ("EOBT_1_flt", "timestamp[us, tz=UTC]"), ("ARVT_1_flt", "timestamp[us, tz=UTC]"),
    ("AOBT_3_flt", "timestamp[us, tz=UTC]"), ("ARVT_3_flt", "timestamp[us, tz=UTC]"),
)
"""``ranking.parquet``'s columns and Arrow types as delivered (Jan+Jul 2026; every training parquet has the same)."""
TEMPLATE_SCHEMA: tuple[tuple[str, str], ...] = (("MVT_ID_mvt", "double"), ("TAXITIME_SEC_mvt", "int32"))
SAME_LOGICAL = {"large_string": "string", "string_view": "string", "large_binary": "binary"}
"""Arrow types pandas reads exactly like the delivered ones: reported, not rejected."""

COPY_SPEC: tuple[tuple[str, str], ...] = (
    ("competition_files", "training_*.parquet"),
    ("additional_data/weather/raw", "**/*.csv"),
    ("additional_data/weather", "manifest.json"),
    ("additional_data/airport_geometry", "**/*"),
    ("additional_data/reanalysis", "**/*"),
    ("additional_data/pru", "**/*"),
)
"""What `init` copies, relative to the source tree: (directory, glob)."""


def regenerated_in_tree(rel: Path) -> bool:
    """A copied file the runbook's own steps rewrite in the tree (`fetch-weather --offline` updates the weather manifest,
    `fetch-geometry --offline` rebuilds every normalised geometry table from the raw files): on a restart `init` keeps
    the tree's version instead of calling it a conflict (the raw inputs it was rebuilt from are still compared)."""
    p = rel.as_posix()
    return p == "additional_data/weather/manifest.json" or (
        p.startswith("additional_data/airport_geometry/") and not p.startswith("additional_data/airport_geometry/raw/"))

# --- what `validate` reads in a release beyond its structure ---------------------------------------------------------
NULL_CHECK = {
    # phase: columns the members read (NM times, schedule, stand, runway, type, the NM join itself)
    "DEP": ("LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt", "SCHED_TIME_UTC_mvt", "STAND_mvt", "RUNWAY_mvt",
            "AIRCRAFT_TYPE_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt"),
    "ARR": ("TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt"),
}
ARRIVAL_TIMES = ("TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt")
NULL_REJECT = 0.95
"""A column the members read that is (nearly) empty in the release, though not in training, is an error ..."""
NULL_RISE = 0.20
"""... so is one emptier than in training by this much (the arrivals' taxi-in times: `ARRIVALS_REJECT`) ..."""
ARRIVALS_REJECT = 0.50
NULL_DRIFT = 0.03
"""... and one that moved this far from the delivered Jan+Jul file (or training, without it) is a warning."""
VOCABULARY = ("FLIGHT_RULE_mvt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt")
"""Low-cardinality labels: a level the training data never had, on more than `NEW_LEVELS_REJECT` of the rows, is an
error (the column turns unseen for every such row)."""
NEW_LEVELS_REJECT = 0.01
NEW_RUNWAY_MIN_ROWS = 100
NEW_STANDS_WARN = 0.05
NEW_TYPES_WARN = 0.02
STALE_RATIO = 2.0
STALE_ABS = 0.005
LOOKBACK_HOURS = 6
"""The longest look-back of the members' blocks (prev_stand_arrival, stand_occupancy, arrival_conditions)."""
CONTEXT_COLUMNS = sorted({"MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt", "ADES_mvt", "TAXITIME_SEC_mvt",
                          "BLOCK_TIME_UTC_mvt", *NULL_CHECK["DEP"], *NULL_CHECK["ARR"], *VOCABULARY})


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _inside(a: Path, b: Path) -> bool:
    a, b = a.resolve(), b.resolve()
    return a == b or a.is_relative_to(b)


def _posix(path: Path | str) -> str:
    return Path(path).as_posix()


def needed_months(months) -> list[str]:
    """Raw weather months the given ranking months read: the months plus the look-back and look-ahead padding
    (`prc.weather.download.needed_months`: two days back, one forward)."""
    out = set()
    for m in months:
        start = pd.Timestamp(f"{m}-01")
        end = start + pd.offsets.MonthEnd(0)
        out |= {(start - pd.Timedelta(days=2)).strftime("%Y-%m"), m, (end + pd.Timedelta(days=1)).strftime("%Y-%m")}
    return sorted(out)


# --- init ------------------------------------------------------------------------------------------------------------

def plan_copy(source: Path) -> list[Path]:
    """Every file `init` copies, relative to ``source``."""
    files: set[Path] = set()
    for sub, pattern in COPY_SPEC:
        base = source / sub
        if base.is_dir():
            files.update(f.relative_to(source) for f in base.glob(pattern) if f.is_file() and not f.name.endswith(".part"))
    return sorted(files)


def weather_coverage(tree: Path, months=FINAL_MONTHS, stations=None) -> dict[str, dict[str, list[str]]]:
    """Per kind (metar, taf): the stations on disk and the station-months the ranking ``months`` need that are missing,
    for ``stations`` (the release's airports: the weather download fetches one station per airport) or, without
    them, for every station on disk."""
    out = {}
    need = needed_months(months)
    raw = Path(tree) / paths.WEATHER_RAW.relative_to(paths.DATA)
    for kind in ("metar", "taf"):
        have: dict[str, set[str]] = {}
        for f in (raw / kind).glob("*.csv"):
            station, _, month = f.stem.partition("_")
            have.setdefault(station, set()).add(month)
        wanted = sorted(stations) if stations is not None else sorted(have)
        out[kind] = {
            "stations": sorted(have),
            "months": need,
            "missing": sorted(f"{s}_{m}" for s in wanted for m in need if m not in have.get(s, set())),
        }
    return out


def init(tree: Path, source: Path, *, dry_run: bool = False, force: bool = False) -> dict:
    """Create the final data tree and copy the inputs into it (see the module docstring)."""
    tree, source = Path(tree).resolve(), Path(source).resolve()
    if _inside(tree, source) or _inside(source, tree):
        raise SystemExit(f"the new tree {tree} and the source tree {source} must not contain each other")
    if _inside(tree, paths.ROOT / "data"):
        raise SystemExit(f"{tree} is (inside) the repository's data/: the final tree must be a directory of its own")
    files = plan_copy(source)
    training = [f for f in files if f.parent.name == "competition_files"]
    if len(training) != 12:
        raise SystemExit(f"expected the 12 training parquets under {source / 'competition_files'}, found {len(training)}")
    total = sum((source / f).stat().st_size for f in files)
    print(f"source {source}\ntree   {tree}\n{len(files):,} files, {total / 1e6:,.1f} MB")
    if dry_run:
        groups = {sub: 0 for sub, _ in COPY_SPEC}
        for f in files:
            owner = max((sub for sub in groups if f.as_posix().startswith(sub + "/")), key=len)
            groups[owner] += 1
        for sub, count in groups.items():
            print(f"  {sub:<36} {count:>5} files")
        return {"files": [f.as_posix() for f in files], "dry_run": True}

    for sub in ("competition_files", "additional_data", "interim"):
        (tree / sub).mkdir(parents=True, exist_ok=True)
    record: dict[str, dict] = {}
    copied = skipped = kept = 0
    conflicts = []
    for rel in files:
        src, dst = source / rel, tree / rel
        digest = _sha1(src)
        if dst.exists():
            if dst.stat().st_size == src.stat().st_size and _sha1(dst) == digest:
                skipped += 1
                record[rel.as_posix()] = {"bytes": src.stat().st_size, "sha1": digest}
                continue
            if regenerated_in_tree(rel) and not force:   # rebuilt by a later step of an earlier start: the tree's own
                kept += 1
                record[rel.as_posix()] = {"bytes": src.stat().st_size, "sha1": digest, "regenerated_in_tree": True}
                continue
            if not force:
                conflicts.append(rel.as_posix())
                continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        part = dst.with_name(dst.name + ".part")
        shutil.copy2(src, part)        # an interrupted copy leaves only the .part file behind
        os.replace(part, dst)
        copied += 1
        record[rel.as_posix()] = {"bytes": src.stat().st_size, "sha1": digest}
    if conflicts:
        raise SystemExit(f"{len(conflicts)} file(s) in {tree} differ from the source (pass --force to replace): {conflicts[:10]}")
    coverage = weather_coverage(tree)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(source),
        "tree": str(tree),
        "git_sha": git,
        "copied": copied,
        "skipped_identical": skipped,
        "kept_regenerated": kept,
        "files": record,
        "weather_coverage": coverage,
        "not_copied": ["competition_files/ranking.parquet", "competition_files/submitting.parquet", "additional_data/adsb"],
    }
    (tree / MANIFEST).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"copied {copied:,}, identical and skipped {skipped:,}, regenerated in the tree and kept {kept:,}; "
          f"manifest {tree / MANIFEST}")
    for kind, cov in coverage.items():
        gap = f"missing {cov['missing']}" if cov["missing"] else "complete"
        print(f"  raw {kind}: {len(cov['stations'])} stations, {cov['months'][0]}..{cov['months'][-1]} {gap}")
    return manifest


# --- validate --------------------------------------------------------------------------------------------------------

@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def _schema_of(file: Path) -> list[tuple[str, str]]:
    schema = pq.read_schema(file)
    return [(name, str(t)) for name, t in zip(schema.names, schema.types)]


def check_schema(actual: list[tuple[str, str]], expected: tuple[tuple[str, str], ...] | list, what: str, report: Report) -> None:
    want = dict(expected)
    have = dict(actual)
    missing = [c for c in want if c not in have]
    extra = [c for c in have if c not in want]
    if missing:
        report.errors.append(f"{what}: missing columns {missing}")
    if extra:
        report.errors.append(f"{what}: unexpected columns {extra}")
    for col in want:
        if col in have and have[col] != want[col]:
            if SAME_LOGICAL.get(have[col], have[col]) == SAME_LOGICAL.get(want[col], want[col]):
                report.warnings.append(f"{what}: {col} is {have[col]}, delivered as {want[col]} (read alike by pandas)")
            else:
                report.errors.append(f"{what}: {col} is {have[col]}, expected {want[col]}")
    if not missing and not extra and [c for c, _ in actual] != [c for c, _ in expected]:
        report.warnings.append(f"{what}: the columns are in another order than delivered (harmless: prepare joins by name)")


@dataclass
class Profile:
    """Counts `validate` compares a release with: rows per phase, empty values per column and phase, the stale-suspect
    segment, and the levels of the labels that must not change."""

    n: dict[str, int] = field(default_factory=lambda: {"DEP": 0, "ARR": 0})
    nulls: dict[str, dict[str, int]] = field(default_factory=lambda: {"DEP": {}, "ARR": {}})
    stale: int = 0
    levels: dict[str, pd.Series] = field(default_factory=dict)

    def share(self, phase: str, col: str) -> float | None:
        n = self.n.get(phase, 0)
        return self.nulls[phase].get(col, 0) / n if n and col in self.nulls[phase] else None

    def add(self, other: "Profile") -> "Profile":
        for phase in ("DEP", "ARR"):
            self.n[phase] += other.n[phase]
            for col, v in other.nulls[phase].items():
                self.nulls[phase][col] = self.nulls[phase].get(col, 0) + v
        self.stale += other.stale
        for name, s in other.levels.items():
            self.levels[name] = s if name not in self.levels else self.levels[name].add(s, fill_value=0)
        return self


def _reporting_airport(frame: pd.DataFrame, is_dep: np.ndarray) -> pd.Series:
    return frame["ADEP_mvt"].where(is_dep, frame["ADES_mvt"]).astype("string")


def profile(frame: pd.DataFrame) -> Profile:
    phase = frame["PHASE_mvt"].astype("string")
    is_dep = phase.eq("DEP").fillna(False).to_numpy(dtype=bool)
    is_arr = phase.eq("ARR").fillna(False).to_numpy(dtype=bool)
    out = Profile()
    for name, mask in (("DEP", is_dep), ("ARR", is_arr)):
        out.n[name] = int(mask.sum())
        for col in NULL_CHECK[name]:
            if col in frame.columns:
                out.nulls[name][col] = int(frame.loc[mask, col].isna().sum())
    if {"AIRCRAFT_TYPE_mvt", "FLIGHT_ID_mvt"} <= set(frame.columns):
        typ = frame["AIRCRAFT_TYPE_mvt"].astype("string")
        untyped = (typ.isna() | typ.str.strip().eq("")).fillna(True).to_numpy(dtype=bool)
        out.stale = int((is_dep & untyped & frame["FLIGHT_ID_mvt"].isna().to_numpy()).sum())
    for col in VOCABULARY:
        if col in frame.columns:
            out.levels[col] = frame[col].dropna().astype("string").value_counts()
    if {"ADEP_mvt", "ADES_mvt"} <= set(frame.columns):
        apt = _reporting_airport(frame, is_dep)
        out.levels["airport"] = apt.dropna().value_counts()
        for col, name in (("RUNWAY_mvt", "runway"), ("STAND_mvt", "stand")):
            if col in frame.columns:
                key = (apt + "|" + frame[col].astype("string")).dropna()
                out.levels[name] = key.value_counts()
    if "AIRCRAFT_TYPE_mvt" in frame.columns:
        out.levels["aircraft_type"] = frame["AIRCRAFT_TYPE_mvt"].dropna().astype("string").value_counts()
    return out


def training_profile(training_dir: Path) -> tuple[np.ndarray, Profile]:
    """The training ids and the training files' `Profile`, one file at a time."""
    files = sorted(Path(training_dir).glob(paths.TRAINING_GLOB))
    if not files:
        raise FileNotFoundError(f"no training parquets ({paths.TRAINING_GLOB}) under {training_dir}")
    ids, total = [], Profile()
    for f in files:
        present = set(pq.read_schema(f).names)
        t = pd.read_parquet(f, columns=[c for c in CONTEXT_COLUMNS if c in present])
        ids.append(t["MVT_ID_mvt"].to_numpy(dtype="float64"))
        total.add(profile(t))
    return np.concatenate(ids), total


def _new_level_share(release: pd.Series, known: pd.Index) -> tuple[float, dict[str, int]]:
    """The share of the release's (non-empty) rows whose level is not ``known``, and the levels with their rows."""
    total = float(release.sum())
    new = release[~release.index.isin(known)]
    return (float(new.sum()) / total if total else 0.0), {str(k): int(v) for k, v in new.sort_values(ascending=False).head(8).items()}


def compare_content(rel: Profile, train: Profile, base: Profile | None, report: Report, *, demote: bool) -> None:
    """Empty values, labels and the stale segment of a release against training (and the delivered Jan+Jul file)."""
    problems = report.warnings if demote else report.errors
    table = {}
    for phase, cols in NULL_CHECK.items():
        if not rel.n[phase]:
            continue
        for col in cols:
            s = rel.share(phase, col)
            if s is None:
                continue
            t = train.share(phase, col)
            b = base.share(phase, col) if base is not None else None
            table[f"{phase} {col}"] = {"release": round(s, 4), "training": None if t is None else round(t, 4),
                                       "baseline": None if b is None else round(b, 4)}
            what = f"{col} empty on {100 * s:.1f} % of the {phase.lower()} rows"
            ref = (f"training {100 * t:.1f} %" if t is not None else "training n/a") + (
                f", delivered Jan+Jul {100 * b:.1f} %" if b is not None else "")
            if phase == "ARR" and col in ARRIVAL_TIMES:
                if s > ARRIVALS_REJECT:
                    problems.append(f"{what} ({ref}): the arrival, occupancy and turnaround features need them")
                elif s > 0.05:
                    report.warnings.append(f"{what} ({ref}): the arrival features read them")
                continue
            if t is not None and s >= NULL_REJECT and t < NULL_REJECT:
                problems.append(f"{what} ({ref}): a column the members read is (nearly) empty")
            elif t is not None and s - t > NULL_RISE:
                problems.append(f"{what} ({ref}): {100 * (s - t):.0f} points emptier than in training")
            elif (r := b if b is not None else t) is not None and abs(s - r) > max(NULL_DRIFT, 4 * np.sqrt(r * (1 - r) / rel.n[phase])):
                report.warnings.append(f"{what} ({ref})")   # beyond 3 points and beyond sampling noise
    report.info["null_shares"] = table
    if rel.n["DEP"]:
        lobt = rel.share("DEP", "LOBT_flt")
        if lobt is not None:
            report.info["departures_with_lobt"] = round(1 - lobt, 4)

    known = {name: s.index for name, s in train.levels.items()}
    if base is not None:
        for name, s in base.levels.items():
            known[name] = known[name].union(s.index) if name in known else s.index
    for col in VOCABULARY:
        if col not in rel.levels or col not in known:
            continue
        share, new = _new_level_share(rel.levels[col], known[col])
        if share > NEW_LEVELS_REJECT:
            problems.append(f"{col}: labels the training data never had on {100 * share:.1f} % of the rows {new} "
                            "(a renamed label turns the model column unseen)")
        elif new:
            report.warnings.append(f"{col}: new labels on {100 * share:.2f} % of the rows {new}")
    if "runway" in rel.levels and "runway" in known:
        runways = rel.levels["runway"]
        apt_of = runways.index.str.split("|").str[0]
        for apt in sorted(set(apt_of)):
            part = runways[apt_of == apt]
            if part.sum() < NEW_RUNWAY_MIN_ROWS:
                continue
            share, new = _new_level_share(part, known["runway"])
            if share > NEW_LEVELS_REJECT:
                problems.append(f"RUNWAY_mvt at {apt}: designators the training data never had on {100 * share:.1f} % "
                                f"of its rows {new} (the runway columns turn unseen)")
            elif new:
                report.warnings.append(f"RUNWAY_mvt at {apt}: new designators on {100 * share:.2f} % of its rows {new}")
    for name, limit, what in (("stand", NEW_STANDS_WARN, "stands"), ("aircraft_type", NEW_TYPES_WARN, "aircraft types")):
        if name in rel.levels and name in known:
            share, new = _new_level_share(rel.levels[name], known[name])
            report.info[f"new_{name}_share"] = round(share, 4)
            if share > limit:
                report.warnings.append(f"{what} the training data never had on {100 * share:.1f} % of the rows, e.g. {new}")

    if rel.n["DEP"] and train.n["DEP"]:
        s, t = rel.stale / rel.n["DEP"], train.stale / train.n["DEP"]
        b = base.stale / base.n["DEP"] if base is not None and base.n["DEP"] else None
        report.info["stale_suspect_share"] = {"release": round(s, 5), "training": round(t, 5),
                                              "baseline": None if b is None else round(b, 5)}
        ref = b if b is not None else t
        if abs(s - ref) > STALE_ABS and (s > STALE_RATIO * ref or s < ref / STALE_RATIO):
            report.warnings.append(
                f"the stale-suspect segment (untyped, no NM join) is {100 * s:.2f} % of the departures "
                f"(training {100 * t:.2f} %" + (f", delivered Jan+Jul {100 * b:.2f} %" if b is not None else "")
                + "): the mixture's universe and the join-window clip move with it"
            )


def context_report(frame: pd.DataFrame, months, report: Report) -> None:
    """The first hours have no earlier rows to look back on (the training months end on 31 Dec 2025), nor does any day
    after a gap: the share of departures that close to the start, and the days without rows."""
    when = frame["MVT_TIME_UTC_mvt"].dropna()
    if when.empty:
        return
    is_dep = frame["PHASE_mvt"].astype("string").eq("DEP").fillna(False).to_numpy(dtype=bool)
    start = when.min()
    early = (frame["MVT_TIME_UTC_mvt"] < start + pd.Timedelta(hours=LOOKBACK_HOURS)).to_numpy() & is_dep
    report.info["context"] = {"first_movement": str(start), "departures_within_lookback_of_start": int(early.sum()),
                              "lookback_hours": LOOKBACK_HOURS}
    days = set(when.dt.strftime("%Y-%m-%d"))
    expected = {d.strftime("%Y-%m-%d") for m in months
                for d in pd.date_range(pd.Timestamp(f"{m}-01"), pd.Timestamp(f"{m}-01") + pd.offsets.MonthEnd(0))}
    report.info["context"]["missing_days"] = sorted(expected - days)   # reported (print_report), not judged


def coverage_report(tree: Path, frame: pd.DataFrame, report: Report) -> None:
    """With the tree: the raw weather station-months and the ERA5 hours the release's takeoffs need."""
    when = frame["MVT_TIME_UTC_mvt"].dropna()
    if when.empty:
        return
    months = sorted(set(when.dt.strftime("%Y-%m")))
    is_dep = frame["PHASE_mvt"].astype("string").eq("DEP").fillna(False).to_numpy(dtype=bool)
    apt = _reporting_airport(frame, is_dep)
    cov = weather_coverage(Path(tree), months, stations=sorted(set(apt.dropna())))
    report.info["weather_coverage"] = cov
    if not cov["metar"]["stations"]:
        report.errors.append(f"no raw METAR files in {tree}: run `init` first")
    elif cov["metar"]["missing"]:
        report.errors.append(f"raw METAR missing for {cov['metar']['missing']}: drop --offline from fetch-weather to "
                             "download them (the weather features would be empty there)")
    if cov["taf"]["missing"]:
        report.warnings.append(f"raw TAF missing for {cov['taf']['missing']}")
    era5 = Path(tree) / paths.ERA5_FILE.relative_to(paths.DATA)
    if not era5.exists():
        report.errors.append(f"{era5} is missing: run `init` first")
        return
    hours = pd.read_parquet(era5, columns=["apt", "time"])
    need_lo = (when.min() - pd.Timedelta(hours=24)).floor("h")
    need_hi = when.max().floor("h")
    short = {}
    for a in sorted(set(apt.dropna())):
        t = hours.loc[hours["apt"].astype("string").eq(a), "time"]
        if t.empty or t.min() > need_lo or t.max() < need_hi:
            short[a] = None if t.empty else f"{t.min()} .. {t.max()}"
    report.info["era5_needed"] = f"{need_lo} .. {need_hi}"
    if short:
        report.errors.append(f"ERA5 does not cover {need_lo} .. {need_hi} at {short}: `fetch-era5` (online) extends it")


def validate(
    file: Path,
    *,
    months: tuple[str, ...] | list[str] = FINAL_MONTHS,
    training_dir: Path | None = None,
    template: Path | None = None,
    reference: Path | None = None,
    baseline: Path | None = None,
    tree: Path | None = None,
    content_as_warnings: bool = False,
) -> Report:
    """Every check of the module docstring; nothing is written."""
    report = Report()
    file = Path(file)
    try:
        actual = _schema_of(file)
    except Exception as exc:  # noqa: BLE001 - any unreadable file is a rejection, with the reader's reason
        report.errors.append(f"{file.name}: not a readable parquet file ({exc})")
        return report
    expected = _schema_of(Path(reference)) if reference else RANKING_SCHEMA
    check_schema(actual, expected, file.name, report)
    present = {c for c, _ in actual}
    frame = pd.read_parquet(file, columns=[c for c in CONTEXT_COLUMNS if c in present])
    report.info["rows"] = int(len(frame))
    if not len(frame):
        report.errors.append(f"{file.name}: no rows")
        return report

    train_ids, train_prof = None, None
    try:
        train_ids, train_prof = training_profile(Path(training_dir) if training_dir else paths.COMPETITION)
    except FileNotFoundError as exc:
        report.errors.append(f"cannot check against the training files: {exc}")
    train_airports = set(map(str, train_prof.levels["airport"].index)) if train_prof is not None else None

    ids = None
    if "MVT_ID_mvt" in frame:
        ids = pd.to_numeric(frame["MVT_ID_mvt"], errors="coerce").to_numpy(dtype="float64")
        if not np.isfinite(ids).all():
            report.errors.append(f"MVT_ID_mvt: {int((~np.isfinite(ids)).sum()):,} missing ids")
        elif not (ids % 1 == 0).all():
            report.errors.append("MVT_ID_mvt: ids are not integral")
        dupes = int(pd.Series(ids).duplicated().sum())
        if dupes:
            report.errors.append(f"MVT_ID_mvt: {dupes:,} duplicated ids")
        if train_ids is not None:
            overlap = int(np.isin(ids, train_ids).sum())
            if overlap:
                report.errors.append(f"MVT_ID_mvt: {overlap:,} ids are training ids (the sets must be disjoint)")

    phase = frame["PHASE_mvt"].astype("string") if "PHASE_mvt" in frame else pd.Series(pd.NA, index=frame.index, dtype="string")
    is_dep = phase.eq("DEP").fillna(False).to_numpy(dtype=bool)
    is_arr = phase.eq("ARR").fillna(False).to_numpy(dtype=bool)
    if "PHASE_mvt" in frame and (~(is_dep | is_arr)).any():
        report.errors.append(f"PHASE_mvt: {int((~(is_dep | is_arr)).sum()):,} rows neither DEP nor ARR "
                             f"({sorted(map(str, phase[~(is_dep | is_arr)].dropna().unique()))[:5]})")
    report.info |= {"departures": int(is_dep.sum()), "arrivals": int(is_arr.sum())}
    if not is_dep.any():
        report.errors.append("no departures: nothing to predict")
    if not is_arr.any():
        report.errors.append("no arrivals: the arrival, occupancy and turnaround features need them")

    if "MVT_TIME_UTC_mvt" in frame:
        when = frame["MVT_TIME_UTC_mvt"]
        if when.isna().any():
            report.errors.append(f"MVT_TIME_UTC_mvt: {int(when.isna().sum()):,} missing times")
        month = when.dt.strftime("%Y-%m")
        outside = ~month.isin(list(months)) & when.notna()
        if outside.any():
            counts = month[outside].value_counts().sort_index().to_dict()
            report.errors.append(f"MVT_TIME_UTC_mvt: {int(outside.sum()):,} rows outside the expected months {list(months)}: {counts}")
        by = pd.DataFrame({"month": month, "dep": is_dep, "arr": is_arr}).groupby("month")[["dep", "arr"]].sum()
        report.info["by_month"] = {m: {"dep": int(r.dep), "arr": int(r.arr)} for m, r in by.iterrows()}
        absent = [m for m in months if m not in by.index]
        if absent:
            report.warnings.append(f"no rows in expected month(s) {absent}")
        thin = [m for m in by.index if m in months and by.loc[m, "arr"] == 0]
        if thin:
            report.warnings.append(f"no arrivals in {thin}")
        context_report(frame, months, report)

    if {"ADEP_mvt", "ADES_mvt"} <= set(frame.columns):
        apt = _reporting_airport(frame, is_dep)
        if apt.isna().any():
            report.errors.append(f"{int(apt.isna().sum()):,} rows without a reporting airport (ADEP on departures, ADES on arrivals)")
        airports = sorted(apt.dropna().astype(str).unique())
        report.info["airports"] = airports
        if train_airports is not None and (new := sorted(set(airports) - train_airports)):
            report.errors.append(
                f"airports not in the training data: {new}: no model was fitted there, and its geometry, weather and "
                "ERA5 inputs were never fetched (fetch-geometry / fetch-weather / fetch-era5 online for it, and decide "
                "how to predict it, before installing)"
            )

    if "TAXITIME_SEC_mvt" in frame:
        taxi = pd.to_numeric(frame["TAXITIME_SEC_mvt"], errors="coerce")
        if taxi[is_dep].notna().any() or ("BLOCK_TIME_UTC_mvt" in frame and frame.loc[is_dep, "BLOCK_TIME_UTC_mvt"].notna().any()):
            report.warnings.append("departures carry a taxi time or block time (prepare masks them on every departure anyway)")

    if train_prof is not None and "PHASE_mvt" in frame:
        base = None
        if baseline is not None and Path(baseline).exists():
            have = set(pq.read_schema(baseline).names)
            base = profile(pd.read_parquet(baseline, columns=[c for c in CONTEXT_COLUMNS if c in have]))
            report.info["baseline"] = str(baseline)
        compare_content(profile(frame), train_prof, base, report, demote=content_as_warnings)

    if tree is not None and "MVT_TIME_UTC_mvt" in frame and "PHASE_mvt" in frame:
        coverage_report(Path(tree), frame, report)

    if template is not None:
        template = Path(template)
        try:
            check_schema(_schema_of(template), TEMPLATE_SCHEMA, template.name, report)
            tmpl = pd.read_parquet(template, columns=["MVT_ID_mvt"])
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"{template.name}: not readable ({exc})")
            tmpl = None
        if tmpl is not None and ids is not None:
            t_ids = pd.to_numeric(tmpl["MVT_ID_mvt"], errors="coerce").to_numpy(dtype="float64")
            dep_ids = ids[is_dep]
            if pd.Series(t_ids).duplicated().any():
                report.errors.append(f"{template.name}: duplicated ids")
            stray = int((~np.isin(t_ids, dep_ids)).sum())
            if stray:
                report.errors.append(f"{template.name}: {stray:,} ids are not departures of {file.name}")
            unsubmitted = int((~np.isin(dep_ids, t_ids)).sum())
            if unsubmitted:
                report.warnings.append(f"{unsubmitted:,} departures of {file.name} are not in {template.name} (predicted, not submitted)")
            report.info["template_rows"] = int(len(t_ids))
    report.info["expected_rank_rows_env"] = f"{data.RANK_ROWS_ENV}={report.info['rows']}"
    return report


def install(file: Path, template: Path | None, tree: Path, *, force: bool = False) -> list[Path]:
    """Copy the checked files into ``tree/competition_files`` as ranking.parquet (and submitting.parquet): only into a
    tree `init` made (``final_inputs.json``), never into the repository's data/ or the tree it was copied from."""
    tree = Path(tree).resolve()
    manifest_file = tree / MANIFEST
    if not manifest_file.exists():
        raise SystemExit(f"{tree} has no {MANIFEST}: install only into a tree `init --tree {tree}` made")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    protected = [(paths.ROOT / "data", "the repository's data/")]
    if manifest.get("source"):
        protected.append((Path(manifest["source"]), "the source tree"))
    for path, what in protected:
        if path.exists() and paths.same_dir(tree, path):
            raise SystemExit(f"{tree} is {what}: its delivered ranking file would be overwritten")
    dest = tree / "competition_files"
    if not dest.is_dir():
        raise SystemExit(f"{dest} does not exist: run `init --tree {tree}` first")
    pairs = [(Path(file), dest / "ranking.parquet")] + ([(Path(template), dest / "submitting.parquet")] if template else [])
    for src, dst in pairs:
        if dst.exists() and _sha1(dst) != _sha1(src) and not force:
            raise SystemExit(f"{dst} exists and differs from {src}; pass --force to replace it")
    for src, dst in pairs:
        part = dst.with_name(dst.name + ".part")
        shutil.copy2(src, part)
        os.replace(part, dst)
        print(f"installed {src} -> {dst}")
    return [dst for _, dst in pairs]


def print_report(report: Report, file: Path) -> None:
    info = report.info
    print(f"{file}: {info.get('rows', 0):,} rows ({info.get('departures', 0):,} departures, {info.get('arrivals', 0):,} arrivals)")
    for month, row in (info.get("by_month") or {}).items():
        print(f"  {month}: {row['dep']:>8,} dep {row['arr']:>8,} arr")
    if info.get("airports"):
        print(f"  airports: {info['airports']}")
    if info.get("departures_with_lobt") is not None:
        print(f"  departures with LOBT (the join-window clip): {100 * info['departures_with_lobt']:.2f} %")
    if info.get("stale_suspect_share"):
        print(f"  stale-suspect share of the departures: {info['stale_suspect_share']}")
    ctx = info.get("context")
    if ctx:
        print(f"  first movement {ctx['first_movement']}: {ctx['departures_within_lookback_of_start']:,} departures within "
              f"{ctx['lookback_hours']} h of it (their look-back finds no earlier rows)")
        missing = ctx.get("missing_days") or []
        if missing:
            print(f"  {len(missing)} day(s) of the expected months without any row ({missing[:6]}"
                  f"{' ...' if len(missing) > 6 else ''}): the first {ctx['lookback_hours']} h after each gap lose "
                  "their look-back context too")
    for w in report.warnings:
        print(f"WARNING {w}")
    for e in report.errors:
        print(f"ERROR   {e}")
    print("OK" if report.ok else f"REJECTED ({len(report.errors)} error(s))")
    if report.ok:
        print(f'for prepare: $env:{data.RANK_ROWS_ENV} = "{info["rows"]}"')


# --- runbook ---------------------------------------------------------------------------------------------------------

class RunbookError(SystemExit):
    """The runbook cannot be written as asked; the message says why."""


@dataclass
class Node:
    run_id: str
    weight: float = 1.0
    children: list["Node"] = field(default_factory=list)
    weights: list[float] | None = None
    lobt_bound: bool = False

    @property
    def is_blend(self) -> bool:
        return bool(self.children)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def blend_tree(run_id: str, *, _seen: tuple[str, ...] = ()) -> Node:
    """A finished run as a tree: a `prc blend` run's sources (recursively) with its weights and clip flag; any other
    run is a leaf. A run directory that does not exist is an error (a run without metrics.json is a leaf: a run
    interrupted after its predictions were written has none)."""
    if run_id in _seen:
        raise ValueError(f"blend cycle at {run_id}")
    d = paths.run_dir(run_id)
    if not d.is_dir():
        raise FileNotFoundError(f"no run directory {d}: run the runbook from the checkout that holds the runs")
    blend = _read_json(d / "metrics.json").get("blend")
    if not blend:
        return Node(run_id)
    runs = list(blend["runs"])
    weights = blend.get("weights")
    node = Node(run_id, weights=None if weights is None else [float(w) for w in weights], lobt_bound=bool(blend.get("lobt_bound")))
    node.children = [blend_tree(r, _seen=(*_seen, run_id)) for r in runs]
    return node


def leaves(node: Node, weight: float = 1.0) -> list[tuple[str, float]]:
    """The trained runs under ``node`` with their effective weights (nested weights multiplied, duplicates summed)."""
    if not node.is_blend:
        return [(node.run_id, weight)]
    w = node.weights or [1.0] * len(node.children)
    total = sum(w)
    out: dict[str, float] = {}
    for child, wi in zip(node.children, w):
        for run, ww in leaves(child, weight * wi / total):
            out[run] = out.get(run, 0.0) + ww
    return list(out.items())


def _refit_status(run_id: str) -> str | None:
    """None when `prc predict` can load the run's refit model, else why not."""
    from prc.predict import PredictRefused, load_source, refit_model_files

    try:
        refit_model_files(load_source(run_id))
    except (PredictRefused, FileNotFoundError) as exc:
        return str(exc)
    return None


def _source_tree_of(run_id: str) -> Path | None:
    from prc.predict import load_source, source_data_tree

    try:
        return source_data_tree(load_source(run_id))[0]
    except Exception:  # noqa: BLE001 - reported by _refit_status
        return None


def _is_mixture(run_id: str) -> bool:
    cfg = _read_json(paths.run_dir(run_id) / "config.resolved.json")
    return bool(((cfg.get("model") or {}).get("mixture") or {}).get("enabled"))


def stacks_of(blend_id: str) -> list[str]:
    """Finished `lab/final_stack.py` runs over ``blend_id``: named ``<the blend's stamp>__<name>__<sha1(blend|name)[:8]>``
    by that script, with the blend as the source in their metrics.json (runs that copied such a record, like the
    winter bases, carry other tags)."""
    out = []
    stamp = blend_id.split("__")[0]
    for m in sorted(paths.RUNS.glob(f"{stamp}__*/metrics.json")):
        parts = m.parent.name.split("__")
        if len(parts) != 3 or parts[2] != hashlib.sha1(f"{blend_id}|{parts[1]}".encode()).hexdigest()[:8]:
            continue
        stack = (_read_json(m).get("stack") or {})
        if stack.get("source") == blend_id and stack.get("steps") == ["lobt_window", "linked_exclusion"]:
            if stack.get("inwindow") in (None, STACK_FACTOR):
                out.append(m.parent.name)
    return out


KNOWN_LEVELS = {
    "2026-10-01T04-48-58__blend11_window_lobt__f13b7e97": [
        "What this produces, measured on Jan+Jul 2026: the eleven-run base blend clipped to the NM join window is the",
        "v31 level (blend17_wide_lobt 263.17 on the board; blend11_window_lobt itself was never uploaded), and the",
        "stack on it (lab/final_stack.py --inwindow 0.5) the v32 level (v32b_stack 260.72; blend11_stack holdout",
        "311.81 / 257.11 without wrong-date rows against v32b's 310.26 / 257.97). The best submission, v70 (220.34), is",
        "about 40 s better than that: it adds the ADS-B second stage (~32 s of it), the swap of the longer-trained",
        "members, the forced-long shift and the LTFM partner -- none of which this script does (see below).",
    ],
}
NOT_COVERED = [
    "Not done here, each the owner's decision:",
    "- the ADS-B second stage (lab/rebuild_adsb_extra.sh, lab/adsb_apply.py; v33-v62): needs `prc fetch-adsb --days`",
    "  for every day of the new months (~150 days at 3-4 GB streamed each) and the stage refitted and re-applied;",
    "- the base swap (lab/base_swap.py: exp065-067 and their depth-9 retrains exp073-075 at their configurations'",
    "  weights): base_swap.py itself needs stage runs; without the stage it is a blend with other members and",
    "  weights, which `runbook --runs ... --weights ...` writes;",
    "- lab/forced_long_shift.py on the stack (estimated on the holdout OOF the predict runs carry; tier C) and",
    "  lab/ltfm_partner_candidate.py (needs a predict run of the exp071 / exp076 refits).",
]


@dataclass(frozen=True)
class _Splat:
    """Stands for the extra predict options the script was started with (-AllowUnseen / --allow-unseen ...)."""


SPLAT = _Splat()
PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
SAFE = re.compile(r"^[A-Za-z0-9_\-./=+]+$")
NUMERIC = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


@dataclass
class Step:
    kind: str                 # comment, blank, py, get, env, use
    text: str = ""
    args: tuple = ()
    var: str = ""
    name: str = ""
    check: tuple = ()


def _ps_token(tok) -> str:
    if tok is SPLAT:
        return "@PredictArgs"
    s = str(tok)
    if not PLACEHOLDER.search(s) and SAFE.match(s) and not NUMERIC.match(s):
        return s   # numbers stay quoted: PowerShell would turn them into doubles and print them back rounded
    out, pos = [], 0
    for m in PLACEHOLDER.finditer(s):
        out.append(s[pos:m.start()].replace("`", "``").replace('"', '`"').replace("$", "`$"))
        out.append("${" + m.group(1) + "}")
        pos = m.end()
    out.append(s[pos:].replace("`", "``").replace('"', '`"').replace("$", "`$"))
    return '"' + "".join(out) + '"'


def _ps_string(tok) -> str:
    """A token as a PowerShell string expression (an array element must be one, numbers included)."""
    t = _ps_token(tok)
    return t if t.startswith('"') else f'"{t}"'


def _sh_token(tok) -> str:
    if tok is SPLAT:
        return '${PredictArgs[@]+"${PredictArgs[@]}"}'
    s = str(tok)
    if not PLACEHOLDER.search(s) and SAFE.match(s):
        return s
    out, pos = [], 0

    def esc(t: str) -> str:
        return t.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")

    for m in PLACEHOLDER.finditer(s):
        out.append(esc(s[pos:m.start()]))
        out.append("${" + m.group(1) + "}")
        pos = m.end()
    out.append(esc(s[pos:]))
    return '"' + "".join(out) + '"'


def _render_ps1(steps: list[Step], indent: str = "    ") -> list[str]:
    lines = []
    for st in steps:
        if st.kind == "blank":
            lines.append("")
        elif st.kind == "comment":
            lines.append(f"{indent}# {st.text}" if st.text else f"{indent}#")
        elif st.kind == "py":
            lines.append(indent + "Invoke-Py " + " ".join(_ps_token(a) for a in st.args))
        elif st.kind == "get":
            lines.append(f"{indent}${st.var} = Get-Py " + " ".join(_ps_token(a) for a in st.args))
        elif st.kind == "env":
            lines.append(f"{indent}$env:{st.var} = Get-Py " + " ".join(_ps_token(a) for a in st.args))
        elif st.kind == "use":
            check = ", ".join(_ps_string(c) for c in st.check)
            lines.append(f"{indent}${st.var} = Use-Run -Name {st.name} -Check @({check}) -Build {{ Invoke-Py "
                         + " ".join(_ps_token(a) for a in st.args) + " }")
        else:
            raise ValueError(st.kind)
    return lines


def _render_sh(steps: list[Step]) -> list[str]:
    lines = []
    for st in steps:
        if st.kind == "blank":
            lines.append("")
        elif st.kind == "comment":
            lines.append(f"# {st.text}" if st.text else "#")
        elif st.kind == "py":
            lines.append("py " + " ".join(_sh_token(a) for a in st.args))
        elif st.kind == "get":
            lines.append(f"{st.var}=\"$(get " + " ".join(_sh_token(a) for a in st.args) + ')"')
        elif st.kind == "env":
            lines.append(f"{st.var}=\"$(get " + " ".join(_sh_token(a) for a in st.args) + f')"; export {st.var}')
        elif st.kind == "use":
            lines.append(f"{st.var}=\"$(use_run {st.name} " + " ".join(_sh_token(c) for c in st.check) + " -- "
                         + " ".join(_sh_token(a) for a in st.args) + ')"')
        else:
            raise ValueError(st.kind)
    return lines


@dataclass
class Plan:
    tree: Path
    source_tree: Path
    source: str
    members: list[str]
    names: dict[str, str]
    blends: list[tuple[str, list[str], list[float], bool, str]]
    final: str
    stack: bool
    months: list[str]
    prefix: str
    threads: int
    rehearse: bool
    ranking: str | None
    template: str | None
    baseline: str | None
    stands_file: str | None
    source_blend: str | None
    source_stack: str | None
    header: list[str]
    warnings: list[str]


def plan_runbook(
    tree: Path,
    *,
    from_blend: str | None = None,
    runs: list[str] | None = None,
    weights: list[float] | None = None,
    months=None,
    rehearse: bool = False,
    prefix: str | None = None,
    threads: int = DEFAULT_THREADS,
    stack: bool = True,
    allow_warnings: bool = False,
) -> Plan:
    """Everything the runbook needs, checked: the trained runs, the blends, the source tree, the warnings."""
    if bool(from_blend) == bool(runs):
        raise RunbookError("give either --from-blend RUN or --runs RUN ...")
    if runs and weights and len(weights) != len(runs):
        raise RunbookError(f"{len(weights)} weights for {len(runs)} runs")
    tree = Path(tree).resolve()
    if _inside(tree, paths.ROOT / "data"):
        raise RunbookError(f"{tree} is (inside) the repository's data/: the final tree must be a directory of its own")
    names: dict[str, str] = {}
    blends: list[tuple[str, list[str], list[float], bool, str]] = []
    if from_blend:
        try:
            root = blend_tree(from_blend)
        except FileNotFoundError as exc:
            raise RunbookError(str(exc)) from None
        order: list[Node] = []

        def walk(node: Node) -> None:
            for child in node.children:
                walk(child)
            order.append(node)

        walk(root)
        members = list(dict.fromkeys(n.run_id for n in order if not n.is_blend))
        for k, run_id in enumerate(members, 1):
            names[run_id] = f"final_m{k:02d}"
        k = 0
        for node in order:
            if node.is_blend and node.run_id not in names:
                k += 1
                names[node.run_id] = f"final_b{k:02d}"
                w = node.weights or [1.0] * len(node.children)
                blends.append((names[node.run_id], [names[c.run_id] for c in node.children], list(w), node.lobt_bound, node.run_id))
        final = names[root.run_id]
        if not root.is_blend:
            raise RunbookError(f"{from_blend} is not a `prc blend` run (its metrics.json has no blend record); use --runs")
        source = [f"Source: {from_blend}: {len(members)} trained run(s); effective weights after the nesting:"]
        source += [f"  {w:.5f}  {r}" for r, w in leaves(root)]
    else:
        members = list(dict.fromkeys(runs))
        if len(members) != len(runs):
            raise RunbookError("a run is given twice")
        for run_id in members:
            if not paths.run_dir(run_id).is_dir():
                raise RunbookError(f"no run directory {paths.run_dir(run_id)}: run the runbook from the checkout that holds the runs")
        for k, run_id in enumerate(members, 1):
            names[run_id] = f"final_m{k:02d}"
        blends.append(("final_blend", [names[r] for r in members], list(weights or [1.0] * len(members)), True, "the runs given"))
        final = "final_blend"
        source = [f"Source: {len(members)} run(s) given on the command line."]

    warnings = []
    for run_id in members:
        status = _refit_status(run_id)
        if status:
            warnings.append(f"{run_id}: {status}")
    trees = {str(t) for t in (_source_tree_of(r) for r in members) if t is not None}
    if len(trees) > 1:
        warnings.append(f"the runs were trained on different data trees {sorted(trees)}; the final tree copies one")
    source_tree = Path(sorted(trees)[0]) if trees else paths.ROOT / "data"
    mixtures = [r for r in members if not _is_mixture(r)]
    if stack and mixtures:
        warnings.append(f"lab/final_stack.py needs mixture runs (refit_detail.parquet); not mixtures: {mixtures}")
        stack = False
    if warnings and not allow_warnings:
        raise RunbookError("the runbook would not run as written:\n  " + "\n  ".join(warnings)
                           + "\n(--allow-warnings writes it anyway, the warnings as comments)")

    months = list(months or (REHEARSAL_MONTHS if rehearse else FINAL_MONTHS))
    competition = source_tree / paths.COMPETITION.relative_to(paths.DATA)
    baseline = competition / paths.RANKING_FILE.name
    geometry_manifest = source_tree / paths.GEOMETRY_MANIFEST.relative_to(paths.DATA)
    stands_file = (_read_json(geometry_manifest).get("stands_file") or {}).get("path")
    source_stack = None
    if rehearse and from_blend:
        found = stacks_of(from_blend)
        source_stack = found[0] if len(found) == 1 else None
        if len(found) > 1:
            warnings.append(f"several stacks of {from_blend}: {found}; the rehearsal compares none (check --stack ... --source-stack ...)")
    header = list(KNOWN_LEVELS.get(from_blend or "", [
        "What this produces: the blend of the runs given, recomputed on the new rows and clipped to the NM join window,",
        "then the stack (lab/final_stack.py --inwindow 0.5) when every run is a regime mixture.",
    ])) + [""] + NOT_COVERED
    return Plan(
        tree=tree, source_tree=source_tree, source=from_blend or "the runs given", members=members, names=names,
        blends=blends, final=final, stack=stack, months=months,
        prefix=prefix or (REHEARSAL_PREFIX if rehearse else DEFAULT_PREFIX), threads=int(threads), rehearse=rehearse,
        ranking=str(baseline) if rehearse else None,
        template=str(competition / paths.SUBMITTING_FILE.name) if rehearse else None,
        baseline=str(baseline) if baseline.exists() else None, stands_file=stands_file,
        source_blend=from_blend, source_stack=source_stack, header=source + [""] + header, warnings=warnings,
    )


def runbook_steps(plan: Plan) -> list[Step]:
    """The commands, in order (`Step` objects the PowerShell and bash renderers share)."""
    s: list[Step] = []

    def comment(*text: str) -> None:
        s.extend(Step("comment", t) for t in text)

    def py(*args) -> None:
        s.append(Step("py", args=args))

    comment("1. The data tree (identical files are skipped, so a re-run costs a checksum pass): training files, raw",
            "   weather (Mar-May 2026 included), geometry, ERA5, PRU.")
    py("lab/final_inputs.py", "init", "--tree", "{Tree}", "--source", "{SourceTree}")
    s.append(Step("blank"))
    comment("2. The released files: checked, then installed (a rejected file stays out; a re-release replaces the old).")
    v = ["lab/final_inputs.py", "validate", "--file", "{Ranking}", "--template", "{Template}", "--months", *plan.months]
    if plan.baseline:
        v += ["--baseline", _posix(plan.baseline)]
    py(*v, "--tree", "{Tree}", "--install", "--force")
    s.append(Step("blank"))
    comment("3. This file's data.version (prefix + sha1 of the installed ranking file: a re-release gets a cache of its own)",
            "   and the row count prepare asserts.")
    s.append(Step("get", var="Version", args=("lab/final_inputs.py", "derive", "version", "--tree", "{Tree}", "--prefix", plan.prefix)))
    s.append(Step("env", var="PRC_EXPECTED_RANK_ROWS", args=("lab/final_inputs.py", "derive", "rank-rows", "--tree", "{Tree}")))
    s.append(Step("blank"))
    comment("4. The movement frame, always rebuilt (--force: never the frame of an earlier file), then the inputs",
            "   re-normalised over the new rows (no network: everything was copied) and their coverage.")
    py("-m", "prc", "prepare", "--force")
    py("-m", "prc", "fetch-weather", "--offline")
    py("-m", "prc", "fetch-weather", "--check")
    geo = ["-m", "prc", "fetch-geometry", "--offline"]
    if plan.stands_file:
        geo += ["--stands-file", _posix(plan.stands_file)]
    py(*geo)
    py("-m", "prc", "fetch-geometry", "--check")
    py("-m", "prc", "fetch-era5", "--check")
    s.append(Step("blank"))
    comment(f"5a. Every trained run's features under the new data.version, all before any prediction, so a missing input",
            "    fails early; blocks shared by several runs are built once, and the training-row parity against each",
            "    run's own cache is measured here (remembered for 5b).")
    for run_id in plan.members:
        py("-m", "prc", "predict", "--run", run_id, "-o", "data.version={Version}", "--tree", "{Tree}",
           "--threads", str(plan.threads), "--features-only", SPLAT)
    s.append(Step("blank"))
    comment("5b. Each refit model's predictions (a run already predicted under this version is re-used).")
    for run_id in plan.members:
        m = plan.names[run_id]
        s.append(Step("use", var=m, name=m, check=("--source", run_id, "--data-version", "{Version}"),
                      args=("-m", "prc", "predict", "--run", run_id, "-o", "data.version={Version}", "--tree", "{Tree}",
                            "--threads", str(plan.threads), "--name", m, SPLAT)))
    s.append(Step("blank"))
    comment("6. The blend(s), innermost first, with the source's weights (always explicit) and its clip flags.")
    for name, member_names, w, clip, origin in plan.blends:
        comment(f"   as {origin}")
        weights = [repr(float(x)) for x in w]
        s.append(Step("use", var=name, name=name,
                      check=("--blend-of", *[f"{{{m}}}" for m in member_names], "--blend-weights", *weights),
                      args=("-m", "prc", "blend", "--runs", *[f"{{{m}}}" for m in member_names], "--weights", *weights,
                            *(("--lobt-bound",) if clip else ()), "--name", name)))
    last = plan.final
    if plan.stack:
        s.append(Step("blank"))
        comment(f"7. The stack: join-window clip, in-window estimate (x{STACK_FACTOR}) and the linked-untyped regime exclusion,",
                "   its rates and offsets from the training months (lab/final_stack.py).")
        s.append(Step("use", var="Stack", name="final_stack", check=("--stack-of", f"{{{plan.final}}}", "--inwindow", str(STACK_FACTOR)),
                      args=("lab/final_stack.py", f"{{{plan.final}}}", "--inwindow", str(STACK_FACTOR), "--write", "final_stack")))
        last = "Stack"
    s.append(Step("blank"))
    if plan.rehearse:
        comment("8. The rehearsal's verdict: every predict run reproduced its source, nothing was rescaled, and the blend",
                "   (and the stack) equal the source's to 1e-6 s.")
        c = ["lab/final_inputs.py", "check", "--members", *[f"{{{plan.names[r]}}}" for r in plan.members]]
        if plan.source_blend:
            c += ["--blend", f"{{{plan.final}}}", "--source-blend", plan.source_blend]
        if plan.stack and plan.source_stack:
            c += ["--stack", "{Stack}", "--source-stack", plan.source_stack]
        py(*c)
    else:
        comment("8. The submission file (mapped onto the new template, clipped to [30, 200000] s, int32) and a level check.")
        py("-m", "prc", "submit", "--run", f"{{{last}}}")
        py("-m", "prc", "levelcheck", "--run", f"{{{last}}}")
    return s


PS1_HELPERS = r'''function Invoke-Py {
    # python with these arguments, output to the console; stops the script on a non-zero exit code
    & $Py @args | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "failed (exit $LASTEXITCODE): python $($args -join ' ')" }
}
function Get-Py {
    # python with these arguments; returns the last line it printed
    $out = @(& $Py @args)
    if ($LASTEXITCODE -ne 0) { throw "failed (exit $LASTEXITCODE): python $($args -join ' ')" }
    if ($out.Count -eq 0) { return "" }
    return ([string]$out[-1]).Trim()
}
function Use-Run {
    # the newest run NAME that passes the checks, made by the Build block first if there is none
    param([string]$Name, [string[]]$Check, [scriptblock]$Build)
    $id = Get-Py lab/final_inputs.py latest $Name @Check --missing-ok
    if (-not $id) {
        & $Build
        $id = Get-Py lab/final_inputs.py latest $Name @Check
    }
    Write-Host "  $Name = $id"
    return $id
}'''

SH_HELPERS = r'''shopt -s inherit_errexit 2>/dev/null || true
py() { "$Py" "$@"; }
get() { "$Py" "$@" | tail -n 1; }
use_run() {
  # the newest run NAME that passes the checks (the arguments before --), made by the command after -- if there is none
  local name="$1"; shift
  local check=()
  while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do check+=("$1"); shift; done
  shift
  local id
  id="$("$Py" lab/final_inputs.py latest "$name" "${check[@]}" --missing-ok | tail -n 1)" || return 1
  if [ -z "$id" ]; then
    "$Py" "$@" >&2 || return 1
    id="$("$Py" lab/final_inputs.py latest "$name" "${check[@]}" | tail -n 1)" || return 1
  fi
  [ -n "$id" ] || return 1
  echo "  $name = $id" >&2
  printf '%s\n' "$id"
}'''


def render(plan: Plan, shell: str = "ps1") -> str:
    """The plan as a PowerShell (``ps1``) or bash (``sh``) script."""
    steps = runbook_steps(plan)
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    title = "Rehearsal" if plan.rehearse else "Runbook"
    head = [
        f"{title}: predict the {'delivered Jan+Jul' if plan.rehearse else 'final'} ranking set ({', '.join(plan.months)}) "
        "from finished runs, without retraining.",
        f"Written {datetime.now(timezone.utc).isoformat(timespec='seconds')} by lab/final_inputs.py runbook (git {git or '?'}).",
        "It sets PRC_DATA_DIR itself (and restores it), runs from the checkout that holds the runs, stops at the first",
        "failing command, and re-uses what an interrupted earlier start already made: start it again after a fix.",
        "Re-used means: a predict run of the same source under the same data.version, a blend of the same runs and",
        "weights, a stack of the same blend. To redo one (after a code fix that changes predictions), delete its run",
        "directory first; everything after it is then redone too, because it names the new run.",
    ]
    if plan.rehearse:
        head += ["A rehearsal: the delivered Jan+Jul files in a tree of their own; every prediction must equal the",
                 "source's (step 8). Run it while no GPU chain needs the CPU."]
    head += [""] + plan.header
    if plan.warnings:
        head += [""] + [f"WARNING {w}" for w in plan.warnings]
    runs_present = list(dict.fromkeys([*plan.members, *([plan.source_blend] if plan.source_blend else []),
                                       *([plan.source_stack] if plan.source_stack else [])]))
    if shell == "ps1":
        usage = (f"powershell -ExecutionPolicy Bypass -File <this file>{'' if plan.rehearse else ' -Ranking <released ranking parquet> -Template <released template>'}"
                 " [-AllowUnseen] [-AllowDrift]")
        lines = [f"# {t}" if t else "#" for t in head] + [f"# Usage: {usage}", ""]
        lines += [
            "param(",
            f'    [string]$Ranking = "{_posix(plan.ranking) if plan.ranking else ""}",',
            f'    [string]$Template = "{_posix(plan.template) if plan.template else ""}",',
            "    [switch]$AllowUnseen,",
            "    [switch]$AllowDrift",
            ")",
            '$ErrorActionPreference = "Stop"',
            "Set-StrictMode -Version 2.0",
            'if (-not $Ranking -or -not $Template) { throw "pass -Ranking <released ranking parquet> -Template <released template parquet>" }',
            f'$Checkout = "{_posix(ROOT)}"',
            f'$Py = "{_posix(sys.executable)}"',
            f'$Tree = "{_posix(plan.tree)}"',
            f'$SourceTree = "{_posix(plan.source_tree)}"',
            "$PredictArgs = @()",
            'if ($AllowUnseen) { $PredictArgs += "--allow-unseen" }',
            'if ($AllowDrift) { $PredictArgs += "--allow-drift" }',
            "foreach ($f in @($Ranking, $Template, $SourceTree, (Join-Path $Checkout \"prc/predict.py\"))) {",
            '    if (-not (Test-Path -LiteralPath $f)) { throw "not found: $f" }',
            "}",
            "$SourceRuns = @(",
            *[f'    "{r}",' for r in runs_present[:-1]], f'    "{runs_present[-1]}"',
            ")",
            "foreach ($r in $SourceRuns) {",
            '    if (-not (Test-Path -LiteralPath (Join-Path $Checkout "runs/$r"))) { throw "run not found in this checkout: $r" }',
            "}",
            "",
            PS1_HELPERS,
            "",
            "$Saved = @{ PRC_DATA_DIR = $env:PRC_DATA_DIR; PRC_EXPECTED_RANK_ROWS = $env:PRC_EXPECTED_RANK_ROWS; PYTHONIOENCODING = $env:PYTHONIOENCODING }",
            "Push-Location $Checkout",
            "try {",
            "    $env:PRC_DATA_DIR = $Tree",
            '    $env:PYTHONIOENCODING = "utf-8"',
            "    Remove-Item Env:PRC_EXPECTED_RANK_ROWS -ErrorAction SilentlyContinue",
            "",
            *_render_ps1(steps),
            "} finally {",
            "    Pop-Location",
            "    foreach ($k in $Saved.Keys) {",
            "        if ($null -eq $Saved[$k]) { Remove-Item \"Env:$k\" -ErrorAction SilentlyContinue }",
            "        else { Set-Item \"Env:$k\" $Saved[$k] }",
            "    }",
            "}",
        ]
        return "\n".join(lines) + "\n"
    if shell == "sh":
        usage = f"bash <this file>{'' if plan.rehearse else ' <released ranking parquet> <released template>'} [--allow-unseen] [--allow-drift]"
        lines = ["#!/usr/bin/env bash"] + [f"# {t}" if t else "#" for t in head] + [f"# Usage: {usage}", ""]
        lines += [
            "set -euo pipefail",
            f'Ranking="{_posix(plan.ranking) if plan.ranking else ""}"',
            f'Template="{_posix(plan.template) if plan.template else ""}"',
            "PredictArgs=()",
            'for a in "$@"; do',
            '  case "$a" in',
            '    --allow-unseen|--allow-drift) PredictArgs+=("$a") ;;',
            '    *) if [ -z "${GotRanking:-}" ]; then Ranking="$a"; GotRanking=1; else Template="$a"; fi ;;',
            "  esac",
            "done",
            'if [ -z "$Ranking" ] || [ -z "$Template" ]; then echo "usage: bash $0 <released ranking parquet> <released template> [--allow-unseen] [--allow-drift]" >&2; exit 2; fi',
            f'Checkout="{_posix(ROOT)}"',
            f'Py="{_posix(sys.executable)}"',
            f'Tree="{_posix(plan.tree)}"',
            f'SourceTree="{_posix(plan.source_tree)}"',
            'for f in "$Ranking" "$Template" "$SourceTree" "$Checkout/prc/predict.py"; do',
            '  [ -e "$f" ] || { echo "not found: $f" >&2; exit 1; }',
            "done",
            "for r in " + " ".join(runs_present) + "; do",
            '  [ -d "$Checkout/runs/$r" ] || { echo "run not found in this checkout: $r" >&2; exit 1; }',
            "done",
            'cd "$Checkout"',
            'export PRC_DATA_DIR="$Tree"',
            "export PYTHONIOENCODING=utf-8",
            "unset PRC_EXPECTED_RANK_ROWS",
            "",
            SH_HELPERS,
            "",
            *_render_sh(steps),
        ]
        return "\n".join(lines) + "\n"
    raise ValueError(f"unknown shell '{shell}' (ps1 or sh)")


def runbook(tree: Path, *, shell: str = "ps1", **kw) -> str:
    """The script that predicts the final set (see `plan_runbook` for the options)."""
    return render(plan_runbook(tree, **kw), shell)


# --- derive, latest, check -------------------------------------------------------------------------------------------

def derive(what: str, tree: Path, prefix: str = DEFAULT_PREFIX) -> str:
    """``version``: ``<prefix>_<sha1[:8]>`` of the tree's installed ranking file; ``rank-rows``: its row count."""
    ranking = Path(tree) / paths.RANKING_FILE.relative_to(paths.DATA)
    if not ranking.exists():
        raise SystemExit(f"no ranking file installed in {tree} ({ranking}): run validate --install first")
    if what == "version":
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", prefix):
            raise SystemExit(f"the version prefix must be a plain name, got {prefix!r}")
        return f"{prefix}_{_sha1(ranking)[:8]}"
    if what == "rank-rows":
        return str(pq.read_metadata(ranking).num_rows)
    raise SystemExit(f"derive what? {what!r} (version or rank-rows)")


def latest_run(
    name: str,
    *,
    source: str | None = None,
    data_version: str | None = None,
    blend_of: list[str] | None = None,
    blend_weights: list[float] | None = None,
    stack_of: str | None = None,
    inwindow: float | None = None,
) -> str | None:
    """The newest run directory named ``<stamp>__<name>__<hash>[__s<seed>]`` that passes the checks, or None."""
    pattern = re.compile(rf"^\d{{4}}-\d{{2}}-\d{{2}}T\d{{2}}-\d{{2}}-\d{{2}}__{re.escape(name)}__[0-9a-f]{{8}}(__s\d+)?$")
    hits = sorted((d.name for d in paths.RUNS.glob(f"*__{name}__*") if d.is_dir() and pattern.match(d.name)), reverse=True)
    for run_id in hits:
        d = paths.run_dir(run_id)
        if source is not None or data_version is not None:
            rec = _read_json(d / "predict.json")
            if not rec or (source is not None and rec.get("source_run") != source):
                continue
            if data_version is not None and rec.get("data_version") != data_version:
                continue
        metrics = _read_json(d / "metrics.json") if (blend_of is not None or stack_of is not None) else {}
        if blend_of is not None:
            blend = metrics.get("blend") or {}
            if list(blend.get("runs") or []) != list(blend_of):
                continue
            if blend_weights is not None:
                have = blend.get("weights") or [1.0] * len(blend_of)
                a = np.asarray(have, dtype="float64")
                b = np.asarray(blend_weights, dtype="float64")
                if len(a) != len(b) or not np.allclose(a / a.sum(), b / b.sum(), rtol=0, atol=1e-12):
                    continue
        if stack_of is not None:
            stack = metrics.get("stack") or {}
            if stack.get("source") != stack_of:
                continue
            if inwindow is not None and stack.get("inwindow") != inwindow:
                continue
        return run_id
    return None


def _max_diff(a: Path, b: Path) -> tuple[float, str | None]:
    """The largest |difference| of ``y_pred`` between two prediction files of the same rows (inf when they differ)."""
    if not a.exists() or not b.exists():
        return float("inf"), f"missing {a if not a.exists() else b}"
    x = pd.read_parquet(a)["y_pred"]
    y = pd.read_parquet(b)["y_pred"]
    if len(x) != len(y) or not x.index.isin(y.index).all():
        return float("inf"), "other rows"
    d = np.abs(x.to_numpy(dtype="float64") - y.reindex(x.index).to_numpy(dtype="float64"))
    return float(d.max()) if len(d) else 0.0, None


def check_rehearsal(
    members: list[str],
    *,
    blend: str | None = None,
    source_blend: str | None = None,
    stack: str | None = None,
    source_stack: str | None = None,
    tol: float = 1e-6,
) -> tuple[list[str], list[str]]:
    """``(problems, notes)``: see the module docstring. The parity report's moved blocks are notes, not problems
    (the reproduction is the hard criterion; a moved block that does not move the predictions is harmless)."""
    problems, notes = [], []
    for run_id in members:
        rec = _read_json(paths.run_dir(run_id) / "predict.json")
        if not rec:
            problems.append(f"{run_id}: no predict.json")
            continue
        rep = rec.get("reproduction") or {}
        if not rep.get("comparable"):
            problems.append(f"{run_id}: no reproduction check ({rep.get('reason')})")
        elif not rep.get("within_tol"):
            problems.append(f"{run_id}: differs from {rec.get('source_run')} by up to {rep.get('max_abs_diff')} s")
        rescale = rec.get("count_rescale") or {}
        moved = {c: v.get("rows_changed") for c, v in (rescale.get("columns") or {}).items() if v.get("rows_changed")}
        if moved:
            problems.append(f"{run_id}: the count rescale changed ranking rows {moved} (on the source's own rows it must not)")
        for block, entry in (rec.get("parity") or {}).items():
            if entry.get("status") == "changed":
                notes.append(f"{run_id}: parity {block} changed on the training rows (max share {entry['max_changed_share']:.4f})")
            elif entry.get("status") == "unavailable":
                notes.append(f"{run_id}: parity {block} not measured ({entry.get('reason')})")
    for what, new, old in (("blend", blend, source_blend), ("stack", stack, source_stack)):
        if new and old:
            for kind in ("test_predictions.parquet", "oof_predictions.parquet"):
                diff, why = _max_diff(paths.run_dir(new) / kind, paths.run_dir(old) / kind)
                if why or diff > tol:
                    problems.append(f"{what} {new} vs {old}, {kind}: " + (why or f"max |diff| {diff:.3g} s"))
                else:
                    notes.append(f"{what} {kind}: max |diff| {diff:.3g} s against {old}")
    return problems, notes


# --- command line ----------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="final_inputs.py", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="create the final data tree and copy the inputs into it")
    p.add_argument("--tree", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path, help="the data tree the runs were trained on (the repository's data/)")
    p.add_argument("--dry-run", action="store_true", help="list what would be copied, write nothing")
    p.add_argument("--force", action="store_true", help="replace files that differ from the source")

    p = sub.add_parser("validate", help="check a released ranking parquet (and template)")
    p.add_argument("--file", required=True, type=Path)
    p.add_argument("--template", type=Path, default=None)
    p.add_argument("--months", nargs="+", default=list(FINAL_MONTHS))
    p.add_argument("--training-dir", type=Path, default=None, help="where the training parquets are (default: the tree's, else the current tree's)")
    p.add_argument("--reference", type=Path, default=None, help="a parquet whose schema to require instead of ranking.parquet's")
    p.add_argument("--baseline", type=Path, default=None, help="the delivered Jan+Jul ranking.parquet, to compare the content with")
    p.add_argument("--tree", type=Path, default=None, help="the final tree: check its weather and ERA5 coverage (and --install into it)")
    p.add_argument("--install", action="store_true", help="after a clean check, copy the files into --tree")
    p.add_argument("--force", action="store_true", help="with --install: replace a differing installed file")
    p.add_argument("--content-as-warnings", action="store_true", help="report the content findings as warnings (never the structural ones)")

    p = sub.add_parser("runbook", help="write the script that predicts the final set")
    p.add_argument("--tree", required=True, type=Path)
    p.add_argument("--from-blend", default=None, help="a finished prc blend run to expand into its trained runs")
    p.add_argument("--runs", nargs="+", default=None)
    p.add_argument("--weights", nargs="+", type=float, default=None)
    p.add_argument("--months", nargs="+", default=None)
    p.add_argument("--out", type=Path, default=None, help="write the script here (default: print it)")
    p.add_argument("--shell", choices=["ps1", "sh"], default=None, help="default: from --out's suffix, else ps1")
    p.add_argument("--rehearse", action="store_true", help="the rehearsal on the delivered Jan+Jul files")
    p.add_argument("--version-prefix", default=None)
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    p.add_argument("--no-stack", action="store_true", help="end at the blend (no lab/final_stack.py)")
    p.add_argument("--allow-warnings", action="store_true", help="write the script although a check warned")

    p = sub.add_parser("derive", help="the data.version or the row count for the ranking file installed in a tree")
    p.add_argument("what", choices=["version", "rank-rows"])
    p.add_argument("--tree", required=True, type=Path)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)

    p = sub.add_parser("latest", help="print the newest run id with this name (that passes the checks)")
    p.add_argument("name")
    p.add_argument("--source", default=None, help="a predict run of this source run")
    p.add_argument("--data-version", default=None, help="a predict run under this data.version")
    p.add_argument("--blend-of", nargs="+", default=None, help="a blend of exactly these runs, in this order")
    p.add_argument("--blend-weights", nargs="+", type=float, default=None)
    p.add_argument("--stack-of", default=None, help="a lab/final_stack.py run over this run")
    p.add_argument("--inwindow", type=float, default=None)
    p.add_argument("--missing-ok", action="store_true", help="print nothing and exit 0 when there is none")

    p = sub.add_parser("check", help="the rehearsal's verdict")
    p.add_argument("--members", nargs="+", required=True)
    p.add_argument("--blend", default=None)
    p.add_argument("--source-blend", default=None)
    p.add_argument("--stack", default=None)
    p.add_argument("--source-stack", default=None)

    args = parser.parse_args(argv)
    if args.command == "init":
        init(args.tree, args.source, dry_run=args.dry_run, force=args.force)
        return 0
    if args.command == "validate":
        if args.install and args.tree is None:
            parser.error("--install needs --tree")
        training_dir = args.training_dir or (args.tree / "competition_files" if args.tree else None)
        report = validate(args.file, months=args.months, training_dir=training_dir, template=args.template,
                          reference=args.reference, baseline=args.baseline, tree=args.tree,
                          content_as_warnings=args.content_as_warnings)
        print_report(report, args.file)
        if not report.ok:
            return 1
        if args.install:
            install(args.file, args.template, args.tree, force=args.force)
        return 0
    if args.command == "runbook":
        shell = args.shell or ("sh" if args.out is not None and args.out.suffix == ".sh" else "ps1")
        try:
            text = runbook(args.tree, shell=shell, from_blend=args.from_blend, runs=args.runs, weights=args.weights,
                           months=args.months, rehearse=args.rehearse, prefix=args.version_prefix, threads=args.threads,
                           stack=not args.no_stack, allow_warnings=args.allow_warnings)
        except RunbookError as exc:
            print(f"runbook: {exc}", file=sys.stderr)
            return 1
        if args.out is None:
            print(text, end="")
        else:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text, encoding="utf-8", newline="\n")
            run = (f"powershell -ExecutionPolicy Bypass -File {args.out}" if shell == "ps1" else f"bash {args.out}")
            print(f"wrote {args.out}; run it with: {run}" + ("" if args.rehearse else " <ranking> <template> (see its header)"))
        return 0
    if args.command == "derive":
        print(derive(args.what, args.tree, args.prefix))
        return 0
    if args.command == "latest":
        found = latest_run(args.name, source=args.source, data_version=args.data_version, blend_of=args.blend_of,
                           blend_weights=args.blend_weights, stack_of=args.stack_of, inwindow=args.inwindow)
        if found is None:
            if args.missing_ok:
                return 0
            print(f"no run named '{args.name}' under {paths.RUNS} that passes the checks", file=sys.stderr)
            return 1
        print(found)
        return 0
    if args.command == "check":
        problems, notes = check_rehearsal(args.members, blend=args.blend, source_blend=args.source_blend,
                                          stack=args.stack, source_stack=args.source_stack)
        for n in notes:
            print(f"  {n}")
        for p_ in problems:
            print(f"PROBLEM {p_}")
        print("REHEARSAL PASSED" if not problems else f"REHEARSAL FAILED ({len(problems)} problem(s))")
        return 0 if not problems else 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
