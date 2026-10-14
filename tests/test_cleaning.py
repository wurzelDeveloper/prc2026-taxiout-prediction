"""Cleaning rules on a synthetic movement frame. No data, no model.

The frame mixes training and ranking rows so the tests can show that a
ranking row never acquires a flag, and it includes a helicopter pad so the
exemption is exercised rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prc import metrics
from prc.cleaning import (
    RULES,
    apply_to_splits,
    clean_metrics,
    describe_flags,
    flag_rows,
    flag_table,
    is_heli_runway,
    is_untyped,
)
from prc.config import CleanConfig, load_config
from prc.validation import Split


def _frame():
    rows = [
        # id, apt, runway, type, split, taxi
        (1, "EDDF", "25C", "A320", "train", 900.0),
        (2, "EDDF", "25L", "B738", "train", 2.0),        # impossible for a jet
        (3, "LSZH", "H", "AS50", "train", 2.0),          # helicopter: exempt
        (4, "LSZH", " hel ", "EC20", "train", -8.0),     # helicopter, sloppy designator: exempt
        (5, "LIRF", "25", None, "train", 9000.0),        # untyped Rome, long
        (6, "LIRF", "25", "", "train", 800.0),           # untyped Rome ("" counts as missing), normal
        (7, "LIRF", "16R", "A321", "train", 8000.0),     # typed Rome, long
        (8, "LFPG", "08L", "A319", "train", 86400.0 + 600.0),   # one day off
        (9, "LFPG", "26R", "A320", "train", 2 * 86400.0 - 900.0),  # two days off
        (10, "LFPG", "26R", "A320", "train", 43200.0),   # half a day: not an offset
        (11, "EDDF", "25C", "A320", "train", 91000.0),   # 25.3 h: 4600 s past a day, outside tol; only max_taxi
        (12, "EDDF", "25C", "A320", "rank", np.nan),     # ranking row: no target
        (13, "LIRF", "25", None, "rank", np.nan),        # ranking row, untyped Rome
    ]
    mvt = pd.DataFrame(rows, columns=["mvt_id", "apt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "split", "y"]).set_index("mvt_id")
    mvt["apt"] = mvt["apt"].astype("category")
    mvt["is_dep"] = True
    y = mvt.loc[mvt["split"].eq("train"), "y"].astype("float64")
    return mvt.drop(columns="y"), y


def _ids(table: pd.DataFrame, rule: str) -> list[int]:
    return sorted(table.index[table[rule]].tolist())


def test_helpers():
    assert is_heli_runway(pd.Series(["H", " hel ", "25C", None])).tolist() == [True, True, False, False]
    assert is_untyped(pd.Series(["A320", "", " ", None])).tolist() == [False, True, True, True]


def test_no_rules_flag_nothing():
    mvt, y = _frame()
    flagged = flag_rows(mvt, y, CleanConfig())
    assert not flagged.any() and flagged.index.equals(y.index)
    assert not CleanConfig().is_active and CleanConfig().summary() == ""


def test_each_rule_alone():
    mvt, y = _frame()
    assert _ids(flag_table(mvt, y, CleanConfig(max_taxi_sec=7200)), "max_taxi") == [5, 7, 8, 9, 10, 11]
    t = flag_table(mvt, y, CleanConfig(min_taxi_sec=30))
    assert _ids(t, "min_taxi") == [2]                       # the jet, not the helicopters
    assert _ids(t, "heli_exempt") == [3, 4]
    assert _ids(flag_table(mvt, y, CleanConfig(day_offset_tol_sec=3600)), "day_offset") == [8, 9]
    assert _ids(flag_table(mvt, y, CleanConfig(untyped_airports=["LIRF"])), "untyped") == [5, 6]
    assert _ids(flag_table(mvt, y, CleanConfig(untyped_airports=["EDDF"])), "untyped") == []


def test_unknown_airport_in_untyped_rule_raises():
    mvt, y = _frame()
    with pytest.raises(ValueError, match="not in the data"):
        flag_table(mvt, y, CleanConfig(untyped_airports=["LTAI"]))


def test_ranking_rows_never_appear():
    mvt, y = _frame()
    flagged = flag_rows(mvt, y, CleanConfig(max_taxi_sec=1, untyped_airports=["LIRF"]))
    assert 12 not in flagged.index and 13 not in flagged.index
    # every training row above 1 s (row 4 is negative), plus the untyped Rome rows
    assert sorted(flagged.index[flagged]) == [1, 2, 3, 5, 6, 7, 8, 9, 10, 11]


def test_apply_to_splits_touches_the_fit_only():
    mvt, y = _frame()
    flagged = flag_rows(mvt, y, CleanConfig(max_taxi_sec=7200, min_taxi_sec=30))
    train_idx = pd.Index([1, 2, 3, 4, 5, 6, 7])
    val_idx = pd.Index([8, 9, 10, 11])
    (s,) = apply_to_splits([Split("seasonal", train_idx, val_idx)], flagged, drop_from_fit=True)
    assert s.train_idx.tolist() == [1, 3, 4, 6]  # 2 (too short), 5 and 7 (too long) gone; helicopters kept
    assert s.val_idx.equals(val_idx)             # flagged validation rows stay
    (same,) = apply_to_splits([Split("seasonal", train_idx, val_idx)], flagged, drop_from_fit=False)
    assert same.train_idx.equals(train_idx)
    with pytest.raises(ValueError, match="every training row"):
        apply_to_splits([Split("x", pd.Index([5, 7]), val_idx)], flagged, drop_from_fit=True)


def test_clean_metrics():
    y_true = np.array([100.0, 200.0, 5000.0])
    y_pred = np.array([110.0, 190.0, 1000.0])
    assert clean_metrics(y_true, y_pred, [False, False, False]) == metrics.regression_metrics(y_true, y_pred)
    assert clean_metrics(y_true, y_pred, [False, False, True])["rmse"] == pytest.approx(10.0)
    assert clean_metrics(y_true, y_pred, [True, True, True])["n"] == 0


def test_day_offset_rows_is_a_report_mask():
    from prc.cleaning import DAY_OFFSET_REPORT_TOL_SEC, day_offset_rows

    mvt, y = _frame()
    mask = day_offset_rows(mvt, y)
    assert mask.name == "dayoffset" and mask.index.equals(y.index) and mask.dtype == bool
    assert sorted(mask.index[mask]) == [8, 9] and DAY_OFFSET_REPORT_TOL_SEC == 3600
    wide = day_offset_rows(mvt, y, tol=5000)
    assert sorted(wide.index[wide]) == [8, 9, 11]   # 91,000 s is 4,600 s past a day


def test_describe_flags_totals():
    mvt, y = _frame()
    table = flag_table(mvt, y, CleanConfig(min_taxi_sec=30, max_taxi_sec=7200, untyped_airports=["LIRF"]))
    desc = describe_flags(table, mvt["apt"])
    assert desc.loc["total", "any"] == desc.drop(index="total")["any"].sum()
    assert desc.loc["LSZH", "heli_exempt"] == 2 and desc.loc["LSZH", "any"] == 0
    assert desc.loc["LIRF", "untyped"] == 2 and desc.loc["LIRF", "max_taxi"] == 2 and desc.loc["LIRF", "any"] == 3
    assert list(desc.columns[: len(RULES)]) == list(RULES)


def test_clean_config_validation_and_summary():
    assert CleanConfig(min_taxi_sec=30, max_taxi_sec=7200, day_offset_tol_sec=3600, untyped_airports=["LIRF"]).summary() == "min30|max7200|day3600|untyped:LIRF"
    assert CleanConfig(max_taxi_sec=7200, drop_from_fit=False).summary() == "max7200|flag-only"
    with pytest.raises(ValueError, match="below max_taxi_sec"):
        CleanConfig(min_taxi_sec=7200, max_taxi_sec=30)
    with pytest.raises(ValueError, match="negative"):
        CleanConfig(day_offset_tol_sec=-1)
    with pytest.raises(ValueError, match="list of ICAO"):
        CleanConfig(untyped_airports="LIRF")


def test_clean_config_through_yaml_and_overrides():
    base = load_config("base.yaml")
    assert not base.data.clean.is_active and base.data.clean.drop_from_fit is True
    cfg = load_config("base.yaml", ["data.clean.max_taxi_sec=7200", "data.clean.untyped_airports=[LIRF]"])
    assert cfg.data.clean.max_taxi_sec == 7200 and cfg.data.clean.untyped_airports == ["LIRF"]
    assert cfg.split_hash() != base.split_hash()  # rules change the fold-dependent caches
    with pytest.raises(ValueError, match="unknown key"):
        load_config("base.yaml", ["data.clean.max_taxi=7200"])
    fit = load_config("exp011_clean_fit.yaml")
    assert fit.data.clean.summary() == "min30|max7200|day3600"
    rome = load_config("exp012_clean_rome.yaml")
    assert rome.data.clean.summary() == "min30|max7200|day3600|untyped:LIRF"  # scalars inherited, list replaced
