"""Canonical data layer.

``prepare()`` reads the immutable competition parquets once and writes two
derived artifacts:

* ``data/interim/movements.parquet`` -- every movement (DEP *and* ARR) from
  training *and* ranking, time-sorted, indexed by ``mvt_id``, with the target
  and off-block time **masked out on all departures**.
* ``data/interim/target.parquet``    -- the masked values, for training
  departures only.

Masking the target inside the canonical frame is the load-bearing decision of
this pipeline: feature code receives ``movements`` and therefore *cannot* see
the answer for any departure, in training or in ranking.  Leakage becomes
structurally impossible rather than a matter of discipline, and the feature
distributions a model is fitted on are identical to the ones it will predict on.

Arrival rows keep their ``BLOCK_TIME_UTC_mvt`` and ``TAXITIME_SEC_mvt``: those
are fully populated in the ranking set too, which is what makes ground-occupancy
and stand-turnaround features legal for the 2026 months.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from prc import paths

# Columns blanked in the ranking set for departures -- and therefore blanked
# here for *all* departures, training included.
TARGET_COL = "TAXITIME_SEC_mvt"
BLOCK_COL = "BLOCK_TIME_UTC_mvt"
MASKED_DEP_COLS = (TARGET_COL, BLOCK_COL)

ID_COL = "MVT_ID_mvt"
INDEX_NAME = "mvt_id"

# Cast to `category`: cuts the in-memory frame roughly 10x and lets CatBoost
# take the string levels directly.
CATEGORICAL_COLS = (
    "FLIGHT_mvt",
    "FLIGHT_RULE_mvt",
    "ADEP_mvt",
    "ADES_mvt",
    "PHASE_mvt",
    "AIRCRAFT_TYPE_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "CALLSIGN_flt",
    "ADEP_flt",
    "ADES_flt",
    "ADES_FILED_flt",
    "MARKET_SEGMENT_flt",
    "FLIGHT_RULE_flt",
    "FLIGHT_TYPE_flt",
    "AIRCRAFT_TYPE_flt",
    "WK_TBL_CAT_flt",
    "AIRCRAFT_OPERATOR_flt",
    "apt",
    "split",
)

TIME_COLS = (
    "MVT_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "LOBT_flt",
    "IOBT_flt",
    "EOBT_1_flt",
    "ARVT_1_flt",
    "AOBT_3_flt",
    "ARVT_3_flt",
)

# Verified against the delivered files; `prepare` asserts these.
EXPECTED_TRAIN_ROWS = 4_167_797
EXPECTED_RANK_ROWS = 689_534
EXPECTED_SUBMIT_ROWS = 344_841
RANK_ROWS_ENV = "PRC_EXPECTED_RANK_ROWS"
"""Overrides `EXPECTED_RANK_ROWS` for a data tree holding another ranking file (the final set: January, February, June
and July 2026, findings 10.33): `lab/final_inputs.py validate` prints the released file's row count to put here.
Unset, the Jan+Jul 2026 count is asserted as before; the training count is never overridden (the twelve 2025 files do
not change)."""


def expected_rank_rows() -> int:
    """The ranking row count `prepare` asserts: `EXPECTED_RANK_ROWS`, or the `PRC_EXPECTED_RANK_ROWS` override."""
    raw = os.environ.get(RANK_ROWS_ENV, "").strip()
    if not raw:
        return EXPECTED_RANK_ROWS
    try:
        value = int(raw.replace("_", "").replace(",", ""))
    except ValueError:
        raise ValueError(f"{RANK_ROWS_ENV} must be a whole number of rows, got {raw!r}") from None
    if value <= 0:
        raise ValueError(f"{RANK_ROWS_ENV} must be positive, got {value}")
    return value


def rank_rows_by_month(mvt: pd.DataFrame) -> pd.DataFrame:
    """Ranking movements per UTC month of takeoff/landing: departures, arrivals and their sum."""
    rank = mvt.loc[mvt["split"].eq("rank").to_numpy(dtype=bool), ["MVT_TIME_UTC_mvt", "is_dep"]]
    month = rank["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m").to_numpy()
    is_dep = rank["is_dep"].to_numpy(dtype=bool)
    table = pd.DataFrame({"month": month, "dep": is_dep, "arr": ~is_dep}).groupby("month")[["dep", "arr"]].sum()
    table["rows"] = table["dep"] + table["arr"]
    return table.astype("int64")


def _read_one(path, split: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["split"] = split
    return df


def build_movements() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the canonical frame and the extracted target. Pure; writes nothing."""
    train_files = paths.training_files()
    if not train_files:
        raise FileNotFoundError(f"no training parquets under {paths.COMPETITION}")

    frames = [_read_one(p, "train") for p in train_files]
    n_train = sum(len(f) for f in frames)
    frames.append(_read_one(paths.RANKING_FILE, "rank"))
    n_rank = len(frames[-1])

    if n_train != EXPECTED_TRAIN_ROWS:
        raise ValueError(f"expected {EXPECTED_TRAIN_ROWS} training rows, got {n_train}")
    expected_rank = expected_rank_rows()
    if n_rank != expected_rank:
        source = f" (from {RANK_ROWS_ENV})" if os.environ.get(RANK_ROWS_ENV, "").strip() else ""
        raise ValueError(f"expected {expected_rank} ranking rows{source}, got {n_rank}")

    mvt = pd.concat(frames, ignore_index=True)
    del frames

    # --- identity ----------------------------------------------------------
    ids = mvt[ID_COL]
    if not (ids % 1 == 0).all():
        raise ValueError(f"{ID_COL} is not integral")
    mvt[INDEX_NAME] = ids.astype("int64")
    if mvt[INDEX_NAME].duplicated().any():
        n = int(mvt[INDEX_NAME].duplicated().sum())
        raise ValueError(f"{ID_COL} is not unique across files ({n} duplicates)")

    # --- reporting airport --------------------------------------------------
    is_dep = mvt["PHASE_mvt"].eq("DEP")
    mvt["is_dep"] = is_dep
    mvt["apt"] = mvt["ADES_mvt"].where(~is_dep, mvt["ADEP_mvt"])
    if mvt["apt"].isna().any():
        raise ValueError("could not derive reporting airport `apt` for some rows")

    # --- extract, then mask, the target ------------------------------------
    is_train_dep = is_dep & mvt["split"].eq("train")
    target = mvt.loc[is_train_dep, [INDEX_NAME, TARGET_COL, BLOCK_COL]].copy()
    target = target.set_index(INDEX_NAME).sort_index()
    if target[TARGET_COL].isna().any():
        raise ValueError("training departures have missing taxi times")

    mvt[TARGET_COL] = mvt[TARGET_COL].astype("float64")
    mvt.loc[is_dep, TARGET_COL] = np.nan
    mvt.loc[is_dep, BLOCK_COL] = pd.NaT

    # --- dtypes, ordering ---------------------------------------------------
    for col in CATEGORICAL_COLS:
        if col in mvt.columns:
            mvt[col] = mvt[col].astype("category")

    mvt = mvt.sort_values("MVT_TIME_UTC_mvt", kind="stable").reset_index(drop=True)
    mvt = mvt.set_index(INDEX_NAME)
    mvt = mvt.drop(columns=[ID_COL])

    _assert_masked(mvt)
    return mvt, target


def _assert_masked(mvt: pd.DataFrame) -> None:
    dep = mvt.loc[mvt["is_dep"]]
    for col in MASKED_DEP_COLS:
        if dep[col].notna().any():
            raise AssertionError(f"{col} leaked into {int(dep[col].notna().sum())} departure rows")
    arr = mvt.loc[~mvt["is_dep"]]
    if arr[TARGET_COL].isna().all():
        raise AssertionError("arrival taxi-in times were masked; they must be kept")


def _sha1(path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def input_digests() -> dict[str, str]:
    """sha1 of every file `build_movements` reads: the twelve training parquets and ``ranking.parquet``."""
    files = [*paths.training_files(), paths.RANKING_FILE]
    return {f.name: (_sha1(f) if f.exists() else "missing") for f in files}


def frame_id(digests: dict[str, str]) -> str:
    """One name for the movement frame those inputs build (with this `prc/data.py`; bump `data.version` when it
    changes): the key `prc.features.build` binds a feature-cache directory to."""
    return hashlib.sha1(json.dumps(sorted(digests.items())).encode("utf-8")).hexdigest()[:16]


def prepare_record() -> dict | None:
    """What `prepare` recorded about the frame on disk (``interim/prepare.json``), or None for a frame built before
    the record existed."""
    if not paths.PREPARE_RECORD.exists():
        return None
    return json.loads(paths.PREPARE_RECORD.read_text(encoding="utf-8"))


def changed_inputs(record: dict, current: dict[str, str] | None = None) -> list[str]:
    """The input files whose sha1 differs from what built the frame (added and removed files included)."""
    current = input_digests() if current is None else current
    recorded = record.get("inputs") or {}
    return sorted(k for k in set(current) | set(recorded) if current.get(k) != recorded.get(k))


def unfinished(record: dict) -> str | None:
    """Why the frame on disk is not the recorded one, when a `prepare --force` started and did not finish."""
    if record.get("building"):
        return (f"a `prepare --force` started {record.get('started_utc', '?')} did not finish, so "
                f"{paths.MOVEMENTS_FILE.name} may be an older frame")
    return None


def prepare(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and persist the canonical artifacts, and record the inputs that built them (``interim/prepare.json``).

    Without ``force`` an existing frame is kept, but only while its inputs are the ones recorded: a ranking file
    installed (or re-released) since the frame was built is refused rather than silently ignored."""
    paths.ensure_dirs()
    if paths.MOVEMENTS_FILE.exists() and paths.TARGET_FILE.exists() and not force:
        record = prepare_record()
        if record is None:
            print(f"{paths.MOVEMENTS_FILE.name} already exists (built before {paths.PREPARE_RECORD.name} recorded the "
                  "inputs, so whether they changed is unknown); use --force to rebuild")
        else:
            problem = unfinished(record) or (
                f"the competition files changed since {paths.MOVEMENTS_FILE} was built: {changed_inputs(record)}"
                if changed_inputs(record) else None
            )
            if problem:
                raise ValueError(f"{problem}. Rebuild with `prc prepare --force` (and give the new frame its own data.version).")
            print(f"{paths.MOVEMENTS_FILE.name} already exists and its inputs are unchanged; use --force to rebuild")
        mvt = load_movements()
        print("ranking rows per month (UTC):")
        print(rank_rows_by_month(mvt).to_string())
        return mvt, load_target()

    # Until the new frame is written, the record says so: a build that fails (a wrong row count) must not leave the
    # old record vouching for whatever frame is on disk.
    digests = input_digests()
    paths.PREPARE_RECORD.write_text(
        json.dumps({"building": True, "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "inputs": digests}, indent=1),
        encoding="utf-8",
    )
    mvt, target = build_movements()
    mvt.to_parquet(paths.MOVEMENTS_FILE, index=True)
    target.to_parquet(paths.TARGET_FILE, index=True)
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_dir": str(paths.DATA),
        "inputs": digests,
        "frame_id": frame_id(digests),
        "rows": {"train": int(mvt["split"].eq("train").sum()), "rank": int(mvt["split"].eq("rank").sum())},
        "departures": {"train": int(len(target)), "rank": int(len(dep_index(mvt, "rank")))},
    }
    paths.PREPARE_RECORD.write_text(json.dumps(record, indent=1), encoding="utf-8")

    print(f"movements : {len(mvt):>9,} rows x {mvt.shape[1]} cols -> {paths.MOVEMENTS_FILE}")
    print(f"  departures {int(mvt['is_dep'].sum()):>9,}   arrivals {int((~mvt['is_dep']).sum()):>9,}")
    print(f"  airports   {sorted(mvt['apt'].cat.categories.tolist())}")
    print(f"target    : {len(target):>9,} rows -> {paths.TARGET_FILE}")
    print(f"frame id  : {record['frame_id']} ({paths.PREPARE_RECORD.name})")
    print("ranking rows per month (UTC):")
    print(rank_rows_by_month(mvt).to_string())
    return mvt, target


def load_movements(columns: list[str] | None = None) -> pd.DataFrame:
    if not paths.MOVEMENTS_FILE.exists():
        raise FileNotFoundError("run `python -m prc prepare` first")
    return pd.read_parquet(paths.MOVEMENTS_FILE, columns=columns)


def load_target() -> pd.Series:
    """Taxi-out seconds for training departures, indexed by ``mvt_id``."""
    if not paths.TARGET_FILE.exists():
        raise FileNotFoundError("run `python -m prc prepare` first")
    return pd.read_parquet(paths.TARGET_FILE)[TARGET_COL].astype("float64")


def load_submission_template() -> pd.DataFrame:
    return pd.read_parquet(paths.SUBMITTING_FILE)


def dep_index(mvt: pd.DataFrame, split: str | None = None) -> pd.Index:
    """Canonical departure index: every row we ever predict on, in frame order."""
    mask = mvt["is_dep"]
    if split is not None:
        mask = mask & mvt["split"].eq(split)
    return mvt.index[mask]


def airports(mvt: pd.DataFrame) -> list[str]:
    """The airport list, derived from the data (10) -- not from the challenge's data page (11)."""
    return sorted(mvt["apt"].cat.categories.tolist())
