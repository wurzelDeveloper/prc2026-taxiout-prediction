"""`prc levelcheck`: does a run's predicted level per airport-month agree with what the PRU published?

The PRU's *additional taxi-out time* indicator gives, per airport and month, the
referenced flights' total reference and additional minutes; their sum over the
flight count is the month's mean taxi-out time, computed from the same airport
data flow as this challenge. For 2025 it matches the training data's monthly
mean (rows below 7,200 s) within a few seconds (EGLL within 2 s every month),
and it exists for January and July 2026. So a submission run's mean prediction
per airport-month can be checked against the answer's mean before uploading.

Two caveats the table carries along. The PRU excludes flights without a
stand-runway reference, and their share (``1 - pru_ref_share``) grows in
disrupted months, where the excluded flights are the long ones -- so where the
reference share is well below the same month's 2025 value the PRU mean is a
floor, not the mean. And a level agreement says nothing about the tail. The
check is a benchmark, never a feature (`prc.pru`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc import data, paths

BULK_CAP_S = 7200.0
"""Predictions and targets above this are left out of the means, mirroring the PRU's exclusion of long flights."""
FLAG_BIAS_S = 50.0
MIN_REF_SHARE = 0.9
COLUMNS = [
    "apt", "year", "month", "n_rows", "pred_mean", "pru_mean_s", "pru_ref_share",
    "data_mean_2025", "pru_mean_2025", "pru_ref_share_2025", "yoy_pru", "yoy_pred", "implied_bias", "flag",
]


def level_table(
    pred: pd.Series,
    pred_rows: pd.DataFrame,
    y_train: pd.Series,
    train_rows: pd.DataFrame,
    pru_monthly: pd.DataFrame,
) -> pd.DataFrame:
    """One row per (airport, month) of the predicted rows.

    ``pred_rows`` / ``train_rows`` carry ``apt`` and ``MVT_TIME_UTC_mvt`` for the
    predicted and the training departures; ``pru_monthly`` is the normalised PRU
    table. ``yoy_pred`` compares the predicted mean with the training data's
    mean in the same month a year earlier, ``yoy_pru`` the PRU's two years, and
    ``implied_bias`` the predicted mean with the PRU mean of the predicted month.
    """
    p = pd.DataFrame(
        {
            "apt": pred_rows["apt"].astype(str).to_numpy(),
            "year": pred_rows["MVT_TIME_UTC_mvt"].dt.year.to_numpy(),
            "month": pred_rows["MVT_TIME_UTC_mvt"].dt.month.to_numpy(),
            "pred": pred.reindex(pred_rows.index).to_numpy(dtype="float64"),
        }
    )
    bulk = p.loc[p["pred"] < BULK_CAP_S]
    out = p.groupby(["apt", "year", "month"], sort=True).size().rename("n_rows").reset_index()
    out = out.merge(
        bulk.groupby(["apt", "year", "month"])["pred"].mean().rename("pred_mean").reset_index(),
        on=["apt", "year", "month"], how="left",
    )

    pru = pru_monthly[["apt", "year", "month", "mean_taxi_out_s", "ref_share"]].rename(
        columns={"mean_taxi_out_s": "pru_mean_s", "ref_share": "pru_ref_share"}
    )
    out = out.merge(pru, on=["apt", "year", "month"], how="left")

    t = pd.DataFrame(
        {
            "apt": train_rows["apt"].astype(str).to_numpy(),
            "year": train_rows["MVT_TIME_UTC_mvt"].dt.year.to_numpy(),
            "month": train_rows["MVT_TIME_UTC_mvt"].dt.month.to_numpy(),
            "y": y_train.reindex(train_rows.index).to_numpy(dtype="float64"),
        }
    )
    t = t.loc[t["y"] < BULK_CAP_S]
    prev = t.groupby(["apt", "year", "month"])["y"].mean().rename("data_mean_2025").reset_index()
    prev["year"] = prev["year"] + 1  # align last year's month with this year's
    out = out.merge(prev, on=["apt", "year", "month"], how="left")
    prev_pru = pru.rename(columns={"pru_mean_s": "pru_mean_2025", "pru_ref_share": "pru_ref_share_2025"}).copy()
    prev_pru["year"] = prev_pru["year"] + 1
    out = out.merge(prev_pru, on=["apt", "year", "month"], how="left")

    out["yoy_pru"] = out["pru_mean_s"] - out["pru_mean_2025"]
    out["yoy_pred"] = out["pred_mean"] - out["data_mean_2025"]
    out["implied_bias"] = out["pred_mean"] - out["pru_mean_s"]
    out["flag"] = (out["implied_bias"].abs() > FLAG_BIAS_S) & (out["pru_ref_share"] >= MIN_REF_SHARE)
    return out[COLUMNS]


def report(run_id: str) -> str:
    run_dir = paths.run_dir(run_id)
    pred_file = run_dir / "test_predictions.parquet"
    if not pred_file.exists():
        raise FileNotFoundError(
            f"{run_id}: no test_predictions.parquet (a --no-test run has no ranking predictions to check)"
        )
    if not paths.PRU_MONTHLY_FILE.exists():
        raise FileNotFoundError("no PRU monthly table: run `python -m prc fetch-pru` first")
    pred = pd.read_parquet(pred_file)["y_pred"]
    mvt = data.load_movements(columns=["apt", "MVT_TIME_UTC_mvt", "is_dep", "split"])
    y = data.load_target()
    dep = mvt.loc[mvt["is_dep"].to_numpy(dtype=bool)]
    table = level_table(
        pred,
        dep.loc[pred.index, ["apt", "MVT_TIME_UTC_mvt"]],
        y,
        dep.loc[y.index, ["apt", "MVT_TIME_UTC_mvt"]],
        pd.read_parquet(paths.PRU_MONTHLY_FILE),
    )
    table.to_csv(run_dir / "level_check.csv", index=False)

    shown = table.copy()
    for col in ("pred_mean", "pru_mean_s", "data_mean_2025", "pru_mean_2025", "yoy_pru", "yoy_pred", "implied_bias"):
        shown[col] = shown[col].round(0)
    for col in ("pru_ref_share", "pru_ref_share_2025"):
        shown[col] = shown[col].round(2)
    both = table.dropna(subset=["yoy_pru", "yoy_pred"])
    corr = float(np.corrcoef(both["yoy_pru"], both["yoy_pred"])[0, 1]) if len(both) > 2 else float("nan")
    flagged = table.loc[table["flag"]]
    lines = [
        f"=== level check of {run_id}: predicted mean per airport-month (predictions below {BULK_CAP_S:.0f} s) "
        "against the PRU's published mean taxi-out ===",
        shown.to_string(index=False),
        f"\ncorrelation of the year-over-year change of the predictions with the PRU's: {corr:.2f}",
        (
            f"flagged (|implied_bias| > {FLAG_BIAS_S:.0f} s with a reference share of at least {MIN_REF_SHARE:.2f}): "
            + (", ".join(f"{r.apt} {r.year}-{r.month:02d} ({r.implied_bias:+.0f} s)" for r in flagged.itertuples()) or "none")
        ),
        "read the rest with the reference share: where it fell against 2025, the PRU mean is a floor (the excluded flights are the long ones)",
        f"written to {run_dir / 'level_check.csv'}",
    ]
    return "\n".join(lines)
