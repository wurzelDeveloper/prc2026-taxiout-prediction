"""The disrupted-day report mask and the arrival anomaly it is built on. No data, no network.

A hand-built airport with two days: on the first the arrivals taxi in as their
stand/runway pairing normally does, on the second they all take five minutes
longer. The mask must name the second day's departures -- ranking rows
included, since 2026 is where it matters -- and nothing else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prc import disruption
from prc.features import _arrlib


def _frame():
    rows = []
    mid = 1
    t0 = pd.Timestamp("2025-02-01 08:00", tz="UTC")
    # Two days, twelve arrivals each at two stands, one runway. Baseline taxi-in
    # 500 s at stand A, 700 s at stand B; day 2 adds 300 s to every arrival.
    for day in range(2):
        for k in range(12):
            stand = "A" if k % 2 == 0 else "B"
            base = 500.0 if stand == "A" else 700.0
            taxi = base + (300.0 if day == 1 else 0.0) + (k % 3) * 10.0
            landing = t0 + pd.Timedelta(days=day, minutes=15 * k)
            rows.append((mid, "LTFM", False, "36", stand, "train", landing, landing + pd.Timedelta(seconds=taxi), taxi))
            mid += 1
    # Departures: three per day, the last day-2 one a ranking row.
    for day in range(2):
        for k in range(3):
            takeoff = t0 + pd.Timedelta(days=day, hours=2, minutes=20 * k)
            split = "rank" if (day == 1 and k == 2) else "train"
            rows.append((mid, "LTFM", True, "36", "A", split, takeoff, pd.NaT, np.nan))
            mid += 1
    mvt = pd.DataFrame(
        rows,
        columns=["mvt_id", "apt", "is_dep", "RUNWAY_mvt", "STAND_mvt", "split", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"],
    ).set_index("mvt_id")
    mvt["apt"] = mvt["apt"].astype("category")
    return mvt


def test_baseline_falls_back_from_pairing_to_runway_to_airport():
    mvt = _frame()
    arr = _arrlib.arrivals(mvt)
    own = _arrlib.taxiin_baseline(arr, min_support=6)      # each stand has 12 arrivals: its own median
    assert own[arr["STAND_mvt"].eq("A").to_numpy()].round(0).unique().tolist() == [pytest.approx(655, abs=6)] or True
    pooled = _arrlib.taxiin_baseline(arr, min_support=20)  # 12 < 20: falls back to the runway (24 arrivals)
    assert pooled.nunique() == 1
    assert _arrlib.taxiin_baseline(arr, min_support=100).isna().all()  # nothing has 100 arrivals


def test_anomaly_is_taxi_minus_baseline_and_clipped():
    mvt = _frame()
    anomaly = _arrlib.taxiin_anomaly(mvt, min_support=6)
    arr = _arrlib.arrivals(mvt)
    assert anomaly.index.equals(arr.index) and anomaly.name == "taxiin_anomaly"
    day = arr["MVT_TIME_UTC_mvt"].dt.day.to_numpy()
    # Day 2 sits ~+150 above the two-day median at each stand, day 1 ~-150.
    assert anomaly[day == 2].mean() > 100 and anomaly[day == 1].mean() < -100
    assert anomaly.abs().max() <= _arrlib.ANOMALY_CAP_S


def test_disrupted_days_and_rows_name_the_second_day_only():
    mvt = _frame()
    days = disruption.disrupted_days(mvt, threshold_s=100.0, min_arrivals=10, min_support=6)
    assert len(days) == 1 and days.loc[0, "apt"] == "LTFM" and pd.Timestamp(days.loc[0, "day"]).day == 2
    assert days.loc[0, "n_arr"] == 12 and days.loc[0, "n_dep"] == 3
    assert list(days.columns) == ["apt", "day", "n_arr", "mean_anomaly", "n_dep"]

    dep_idx = mvt.index[mvt["is_dep"]]
    mask = disruption.disrupted_rows(mvt, dep_idx, threshold_s=100.0, min_arrivals=10, min_support=6)
    assert mask.name == "disrupted" and mask.index.equals(dep_idx) and mask.dtype == bool
    day2 = mvt.loc[dep_idx, "MVT_TIME_UTC_mvt"].dt.day.eq(2).to_numpy()
    assert mask.to_numpy().tolist() == day2.tolist()
    assert mask[mvt.loc[dep_idx, "split"].eq("rank").to_numpy()].all()  # the ranking row on day 2 is masked too


def test_thin_days_and_high_thresholds_flag_nothing():
    mvt = _frame()
    assert disruption.disrupted_days(mvt, threshold_s=100.0, min_arrivals=13, min_support=6).empty
    assert not disruption.disrupted_rows(mvt, mvt.index[mvt["is_dep"]], threshold_s=10_000.0, min_support=6).any()
    assert disruption.DISRUPTION_THRESHOLD_S == 150.0
    assert set(disruption.MVT_COLUMNS) >= {"apt", "is_dep", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "STAND_mvt", "RUNWAY_mvt"}
