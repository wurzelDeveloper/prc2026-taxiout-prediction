"""Turning a run's ranking predictions into a submission file.

The assertions here are not defensive padding. A submission that is the right
length but has drifted row order, or carries an int64 where int32 was expected,
fails on upload after the run has already cost an hour -- so every property the
template guarantees is checked before anything is written.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from prc import data, paths

CLIP = (30.0, 200_000.0)
"""The upper bound is not a taxi time: training targets reach 131,167 s where a
record's block time is a stale schedule, and the model predicts those on
purpose (see prc/features/stale_schedule.py). It only guards against a runaway
prediction. The lower bound matches the organisers' template floor."""


def build_submission(run_id: str, clip: tuple[float, float] = CLIP) -> Path:
    run_path = paths.run_dir(run_id)
    preds_file = run_path / "test_predictions.parquet"
    if not preds_file.exists():
        raise FileNotFoundError(f"{preds_file} not found -- was the run trained with test predictions enabled?")

    template = data.load_submission_template()
    preds = pd.read_parquet(preds_file)["y_pred"]

    out = template.copy()
    mapped = out["MVT_ID_mvt"].astype("int64").map(preds)
    missing = int(mapped.isna().sum())
    if missing:
        raise ValueError(f"{missing:,} of {len(out):,} submission rows have no prediction")

    values = np.clip(mapped.to_numpy(dtype="float64"), *clip)
    out["TAXITIME_SEC_mvt"] = np.rint(values).astype("int32")

    _validate(out, template)

    target = paths.SUBMISSIONS / f"{run_id}__submitting.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(target, index=False)
    shutil.copyfile(target, paths.SUBMISSIONS / "submitting.parquet")

    # Round-trip: what we assert about must be what actually landed on disk.
    _validate(pd.read_parquet(target), template)

    print(f"wrote {target}")
    print(f"  rows {len(out):,}  median {out['TAXITIME_SEC_mvt'].median():,.0f}s  "
          f"p01 {out['TAXITIME_SEC_mvt'].quantile(0.01):,.0f}s  p99 {out['TAXITIME_SEC_mvt'].quantile(0.99):,.0f}s")
    print(f"  copied to {paths.SUBMISSIONS / 'submitting.parquet'}")
    return target


def _validate(out: pd.DataFrame, template: pd.DataFrame) -> None:
    if list(out.columns) != list(template.columns):
        raise AssertionError(f"columns {list(out.columns)} != template {list(template.columns)}")
    if len(out) != len(template):
        raise AssertionError(f"{len(out):,} rows, template has {len(template):,}")
    if set(out["MVT_ID_mvt"]) != set(template["MVT_ID_mvt"]):
        raise AssertionError("MVT_ID_mvt set differs from the template")
    if out["MVT_ID_mvt"].dtype != template["MVT_ID_mvt"].dtype:
        raise AssertionError(f"MVT_ID_mvt dtype {out['MVT_ID_mvt'].dtype}, template has {template['MVT_ID_mvt'].dtype}")
    if str(out["TAXITIME_SEC_mvt"].dtype) != "int32":
        raise AssertionError(f"TAXITIME_SEC_mvt dtype is {out['TAXITIME_SEC_mvt'].dtype}, expected int32")
    if out.isna().any().any():
        raise AssertionError("submission contains NaN")
