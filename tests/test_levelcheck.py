"""`levelcheck.level_table`: predicted monthly means against the PRU's, with last year's calibration. No data, no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prc.levelcheck import BULK_CAP_S, COLUMNS, FLAG_BIAS_S, level_table


def _rows(apt, times):
    return pd.DataFrame({"apt": pd.Categorical([apt] * len(times)), "MVT_TIME_UTC_mvt": pd.to_datetime(times, utc=True)})


def test_level_table_means_flags_and_year_over_year():
    pred_rows = pd.concat(
        [
            _rows("EHAM", ["2026-01-05 10:00", "2026-01-06 10:00", "2026-01-07 10:00"]),
            _rows("EGLL", ["2026-07-01 10:00", "2026-07-02 10:00"]),
        ],
        ignore_index=True,
    )
    pred_rows.index = pd.Index([11, 12, 13, 21, 22], name="mvt_id")
    pred = pd.Series([900.0, 1000.0, 20_000.0, 1300.0, 1340.0], index=pred_rows.index)  # the 20,000 s row is left out

    train_rows = pd.concat([_rows("EHAM", ["2025-01-09 10:00", "2025-01-10 10:00"]), _rows("EGLL", ["2025-07-03 10:00"])], ignore_index=True)
    train_rows.index = pd.Index([1, 2, 3], name="mvt_id")
    y = pd.Series([800.0, 820.0, 1420.0], index=train_rows.index)

    pru = pd.DataFrame(
        {
            "apt": ["EHAM", "EHAM", "EGLL", "EGLL"],
            "year": [2025, 2026, 2025, 2026],
            "month": [1, 1, 7, 7],
            "mean_taxi_out_s": [803.0, 982.0, 1427.0, 1344.0],
            "ref_share": [0.95, 0.86, 0.99, 0.99],
        }
    )
    table = level_table(pred, pred_rows, y, train_rows, pru)
    assert list(table.columns) == COLUMNS and len(table) == 2

    egll = table.loc[table["apt"].eq("EGLL")].iloc[0]
    assert egll["n_rows"] == 2 and egll["pred_mean"] == pytest.approx(1320.0)
    assert egll["pru_mean_s"] == 1344.0 and egll["pru_mean_2025"] == 1427.0 and egll["data_mean_2025"] == 1420.0
    assert egll["yoy_pru"] == pytest.approx(-83.0) and egll["yoy_pred"] == pytest.approx(-100.0)
    assert egll["implied_bias"] == pytest.approx(-24.0) and not egll["flag"]  # within 50 s

    eham = table.loc[table["apt"].eq("EHAM")].iloc[0]
    assert eham["n_rows"] == 3 and eham["pred_mean"] == pytest.approx(950.0)  # 20,000 s prediction excluded
    assert eham["implied_bias"] == pytest.approx(950.0 - 982.0)
    assert not eham["flag"]  # 32 s under, and the reference share is below 0.9 anyway
    assert BULK_CAP_S == 7200.0 and FLAG_BIAS_S == 50.0


def test_level_table_flags_a_large_bias_only_at_a_high_reference_share():
    pred_rows = _rows("LTFM", ["2026-01-05 10:00", "2026-01-06 10:00"])
    pred_rows.index = pd.Index([1, 2], name="mvt_id")
    pred = pd.Series([1250.0, 1250.0], index=pred_rows.index)
    train_rows = _rows("LTFM", ["2025-01-05 10:00"])
    train_rows.index = pd.Index([9], name="mvt_id")
    y = pd.Series([1040.0], index=train_rows.index)
    pru = pd.DataFrame({"apt": ["LTFM"], "year": [2026], "month": [1], "mean_taxi_out_s": [1089.0], "ref_share": [0.83]})
    low_share = level_table(pred, pred_rows, y, train_rows, pru).iloc[0]
    assert low_share["implied_bias"] == pytest.approx(161.0) and not low_share["flag"]
    assert np.isnan(low_share["pru_mean_2025"]) and np.isnan(low_share["yoy_pru"])
    high_share = level_table(pred, pred_rows, y, train_rows, pru.assign(ref_share=[0.95])).iloc[0]
    assert high_share["flag"]
