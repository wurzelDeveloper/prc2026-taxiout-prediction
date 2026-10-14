"""A tiny synthetic competition data tree, and `prc.paths` pointed at it, for tests that run the real pipeline
(`prepare`, `train`, `predict`) without the competition data. Imported by tests; pytest does not collect it.

The movements follow the delivered files' schema (`SCHEMA`, the 30 columns of ``ranking.parquet``) at three airports,
a few days per month. Departures carry a taxi time driven by stand and runway; the untyped, NM-less segment holds
stale block times (target = takeoff minus schedule, hours) for one carrier and genuine taxis for the others, and a
few training rows carry a block time on the wrong date (a day plus a taxi), which ``data.clean.day_offset_tol_sec``
drops from the fit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from prc import paths as _paths

AIRPORTS = ("EDDF", "LIRF", "LSZH")
RUNWAYS = {"EDDF": ("07C", "25C"), "LIRF": ("16L", "25"), "LSZH": ("28", "32")}
STANDS = tuple(f"{p}{i}" for p in "AB" for i in range(1, 9))
DESTINATIONS = ("EGLL", "LFPG", "LEMD", "EHAM", "LTFM", "KJFK")
CARRIERS = ("AZ", "LH", "XX", "U2")
STALE_CARRIER = "XX"
TYPES = ("A320", "B738", "A321", "E190")

SCHEMA = pa.schema([
    ("MVT_ID_mvt", pa.float64()), ("FLIGHT_ID_mvt", pa.float64()), ("FLIGHT_mvt", pa.string()),
    ("FLIGHT_RULE_mvt", pa.string()), ("ADEP_mvt", pa.string()), ("ADES_mvt", pa.string()), ("PHASE_mvt", pa.string()),
    ("MVT_TIME_UTC_mvt", pa.timestamp("us", tz="UTC")), ("BLOCK_TIME_UTC_mvt", pa.timestamp("us", tz="UTC")),
    ("SCHED_TIME_UTC_mvt", pa.timestamp("us", tz="UTC")), ("AIRCRAFT_TYPE_mvt", pa.string()), ("RUNWAY_mvt", pa.string()),
    ("STAND_mvt", pa.string()), ("TAXITIME_SEC_mvt", pa.int32()), ("LOBT_flt", pa.timestamp("us", tz="UTC")),
    ("CALLSIGN_flt", pa.string()), ("ADEP_flt", pa.string()), ("ADES_flt", pa.string()), ("ADES_FILED_flt", pa.string()),
    ("MARKET_SEGMENT_flt", pa.string()), ("IOBT_flt", pa.timestamp("us", tz="UTC")), ("FLIGHT_RULE_flt", pa.string()),
    ("FLIGHT_TYPE_flt", pa.string()), ("AIRCRAFT_TYPE_flt", pa.string()), ("WK_TBL_CAT_flt", pa.string()),
    ("AIRCRAFT_OPERATOR_flt", pa.string()), ("EOBT_1_flt", pa.timestamp("us", tz="UTC")),
    ("ARVT_1_flt", pa.timestamp("us", tz="UTC")), ("AOBT_3_flt", pa.timestamp("us", tz="UTC")),
    ("ARVT_3_flt", pa.timestamp("us", tz="UTC")),
])
"""The delivered files' columns and types (ranking.parquet and every training parquet share them)."""

_ORIGINAL = {k: v for k, v in vars(_paths).items() if isinstance(v, Path)}
_ORIG_DATA, _ORIG_ROOT, _ORIG_CONFIGS = _paths.DATA, _paths.ROOT, _paths.CONFIGS


def retarget_paths(mp, data_dir: Path, work_dir: Path) -> None:
    """Point every `prc.paths` location at ``data_dir`` (the data tree) and ``work_dir`` (cache, runs, submissions,
    generated configs), as ``PRC_DATA_DIR`` plus a private checkout would. ``configs/`` stays the repository's."""
    for name, value in _ORIGINAL.items():
        if name in ("ROOT", "CONFIGS"):
            continue
        if name == "DATA":
            new = data_dir
        elif value.is_relative_to(_ORIG_DATA):
            new = data_dir / value.relative_to(_ORIG_DATA)
        elif value.is_relative_to(_ORIG_CONFIGS):
            new = work_dir / "configs" / value.relative_to(_ORIG_CONFIGS)
        elif value.is_relative_to(_ORIG_ROOT):
            new = work_dir / value.relative_to(_ORIG_ROOT)
        else:
            continue
        mp.setattr(_paths, name, Path(new))


def _days(months: list[str], per_month: int) -> list[pd.Timestamp]:
    return [pd.Timestamp(f"{m}-{d:02d}", tz="UTC") for m in months for d in range(1, per_month + 1)]


def movements(
    days: list[pd.Timestamp],
    rng: np.random.Generator,
    *,
    id_start: int,
    n_dep: int = 10,
    n_arr: int = 10,
    stands: tuple[str, ...] = STANDS,
    destinations: tuple[str, ...] = DESTINATIONS,
    wrong_date: int = 0,
    blank_departures: bool = False,
) -> pd.DataFrame:
    """Movements on ``days`` at every airport, ids from ``id_start``; ``blank_departures`` blanks the departures'
    target and block time the way the ranking file does."""
    rows = []
    next_id = id_start
    for day in days:
        for apt in AIRPORTS:
            base = {"EDDF": 900.0, "LIRF": 700.0, "LSZH": 600.0}[apt]
            for k in range(n_dep + n_arr):
                is_dep = k < n_dep
                takeoff = day + pd.Timedelta(seconds=int(rng.integers(5 * 3600, 22 * 3600)))
                stand_i = int(rng.integers(0, len(stands)))
                rwy_i = int(rng.integers(0, 2))
                carrier = CARRIERS[int(rng.integers(0, len(CARRIERS)))]
                untyped = rng.random() < 0.25
                matched = not untyped and rng.random() < 0.9
                taxi_s = int(round(max(base + 40.0 * (stand_i % 8) + 150.0 * rwy_i + rng.normal(0, 30), 60.0)))
                if is_dep:
                    # The untyped, NM-less segment departs hours after its schedule; for one carrier the recorded
                    # block time is that schedule, so the target is takeoff minus schedule.
                    suspect = untyped and not matched
                    delay_s = int(round(rng.uniform(2.5, 8.0) * 3600)) if suspect else int(rng.integers(0, 1500))
                    off_block = takeoff - pd.Timedelta(seconds=taxi_s)
                    sched = off_block - pd.Timedelta(seconds=delay_s)
                    target_s = taxi_s + delay_s if suspect and carrier == STALE_CARRIER else taxi_s
                    block = takeoff - pd.Timedelta(seconds=target_s)
                    lobt = off_block + pd.Timedelta(seconds=int(rng.integers(-300, 300))) if matched else pd.NaT
                else:
                    target_s = int(round(taxi_s * 0.6))  # taxi-in
                    block = takeoff + pd.Timedelta(seconds=target_s)
                    sched = takeoff - pd.Timedelta(seconds=600)
                    lobt = pd.NaT
                other = destinations[int(rng.integers(0, len(destinations)))]
                rows.append({
                    "MVT_ID_mvt": float(next_id),
                    "FLIGHT_ID_mvt": float(10_000_000 + next_id) if matched else np.nan,
                    "FLIGHT_mvt": f"{carrier}{int(rng.integers(100, 999))}",
                    "FLIGHT_RULE_mvt": "I",
                    "ADEP_mvt": apt if is_dep else other,
                    "ADES_mvt": other if is_dep else apt,
                    "PHASE_mvt": "DEP" if is_dep else "ARR",
                    "MVT_TIME_UTC_mvt": takeoff,
                    "BLOCK_TIME_UTC_mvt": block,
                    "SCHED_TIME_UTC_mvt": sched,
                    "AIRCRAFT_TYPE_mvt": None if untyped else TYPES[int(rng.integers(0, len(TYPES)))],
                    "RUNWAY_mvt": RUNWAYS[apt][rwy_i],
                    "STAND_mvt": stands[stand_i],
                    "TAXITIME_SEC_mvt": target_s,
                    "LOBT_flt": lobt,
                    "CALLSIGN_flt": f"{carrier}C{next_id % 997}" if matched else None,
                    "ADEP_flt": (apt if is_dep else other) if matched else None,
                    "ADES_flt": (other if is_dep else apt) if matched else None,
                    "ADES_FILED_flt": (other if is_dep else apt) if matched else None,
                    "MARKET_SEGMENT_flt": "Mainline" if matched else None,
                    "IOBT_flt": lobt,
                    "FLIGHT_RULE_flt": "I" if matched else None,
                    "FLIGHT_TYPE_flt": "S" if matched else None,
                    "AIRCRAFT_TYPE_flt": "A320" if matched else None,
                    "WK_TBL_CAT_flt": "M" if matched else None,
                    "AIRCRAFT_OPERATOR_flt": carrier if matched else None,
                    "EOBT_1_flt": lobt,
                    "ARVT_1_flt": pd.NaT,
                    "AOBT_3_flt": lobt,
                    "ARVT_3_flt": pd.NaT,
                })
                next_id += 1
    frame = pd.DataFrame(rows)
    frame["TAXITIME_SEC_mvt"] = frame["TAXITIME_SEC_mvt"].astype("Int32")
    if wrong_date:
        # Block times on the previous day's date: the target is a day plus the taxi. Taken from the suspect segment,
        # so that dropping them from the fit moves the segment's fold-dependent stale rates.
        dep = frame["PHASE_mvt"].eq("DEP")
        suspect = dep & frame["AIRCRAFT_TYPE_mvt"].isna() & frame["FLIGHT_ID_mvt"].isna()
        hit = rng.choice(frame.index[suspect], size=wrong_date, replace=False)
        frame.loc[hit, "TAXITIME_SEC_mvt"] = frame.loc[hit, "TAXITIME_SEC_mvt"] + 86_400
        frame.loc[hit, "BLOCK_TIME_UTC_mvt"] = frame.loc[hit, "BLOCK_TIME_UTC_mvt"] - pd.Timedelta(days=1)
    if blank_departures:
        dep = frame["PHASE_mvt"].eq("DEP")
        frame.loc[dep, "TAXITIME_SEC_mvt"] = pd.NA
        frame.loc[dep, "BLOCK_TIME_UTC_mvt"] = pd.NaT
    return frame


def write_parquet(frame: pd.DataFrame, path: Path) -> Path:
    """``frame`` written with the delivered schema (nullable int32 target, microsecond UTC timestamps)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame[SCHEMA.names], schema=SCHEMA, preserve_index=False)
    pq.write_table(table, path)
    return path


TRAIN_MONTHS = [f"2025-{m:02d}" for m in range(1, 13)]


def write_training(root: Path, *, seed: int = 0, per_month: int = 3, wrong_date: int = 4) -> int:
    """Twelve monthly training parquets under ``root/competition_files``; returns the row count."""
    rng = np.random.default_rng(seed)
    frame = movements(_days(TRAIN_MONTHS, per_month), rng, id_start=1, wrong_date=wrong_date)
    for month in TRAIN_MONTHS:
        part = frame.loc[frame["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m").eq(month)]
        start = pd.Timestamp(f"{month}-01")
        end = (start + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d")
        write_parquet(part, root / "competition_files" / f"training_{start:%Y-%m-%d}_{end}.parquet")
    return len(frame)


def ranking_frame(months: list[str], *, seed: int, id_start: int, per_month: int = 2, **kw) -> pd.DataFrame:
    """A ranking file's rows: departures blanked as delivered."""
    return movements(_days(months, per_month), np.random.default_rng(seed), id_start=id_start, blank_departures=True, **kw)


def write_ranking(root: Path, frame: pd.DataFrame) -> int:
    """``ranking.parquet`` and its submission template (one row per departure, empty target)."""
    write_parquet(frame, root / "competition_files" / "ranking.parquet")
    dep = frame.loc[frame["PHASE_mvt"].eq("DEP"), ["MVT_ID_mvt"]].copy()
    template = pa.Table.from_pandas(
        dep.assign(TAXITIME_SEC_mvt=pd.array([None] * len(dep), dtype="Int32")),
        schema=pa.schema([("MVT_ID_mvt", pa.float64()), ("TAXITIME_SEC_mvt", pa.int32())]),
        preserve_index=False,
    )
    pq.write_table(template, root / "competition_files" / "submitting.parquet")
    return len(frame)
