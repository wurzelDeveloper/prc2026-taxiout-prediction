"""The stale-schedule blocks: residual base only on the untyped, NM-less segment;
stale rates only from the fold's fit rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc.features.registry import FeatureContext
from prc.features.stale_schedule import PRIOR_WEIGHT, stale_schedule, stale_target_stats


def _frame() -> pd.DataFrame:
    t0 = pd.Timestamp("2025-03-01 10:00:00", tz="UTC")
    rows = [
        # id, takeoff offset (s), sched offset (s), type, flight id, flight, ades, LOBT offset (s)
        (1, 0, -900, "A320", 11, "RYR100", "EGKK", -1_200),   # typed, joined: not suspect
        (2, 0, -900, None, 12, "RYR101", "EGKK", None),       # untyped but joined: not suspect
        (3, 0, -900, "A320", None, "RYR102", "EGKK", -20_000),  # typed, unjoined: not suspect
        (4, 0, -900, None, None, "RYR103", "EGKK", None),     # suspect, 15 min after schedule
        (5, 0, -90_000, "", None, "ISR200", "LLBG", None),    # suspect, blank type, a day and 3,600 s after schedule
        (6, 0, None, None, None, "ISR201", "LLBG", None),     # suspect but no schedule: base stays NaN
        (7, 0, 3_600, None, None, "RYR104", "EGKK", None),    # suspect but the schedule is after takeoff: no base
    ]
    return pd.DataFrame(
        {
            "MVT_TIME_UTC_mvt": [t0 + pd.Timedelta(seconds=r[1]) for r in rows],
            "SCHED_TIME_UTC_mvt": [pd.NaT if r[2] is None else t0 + pd.Timedelta(seconds=r[2]) for r in rows],
            "AIRCRAFT_TYPE_mvt": pd.array([r[3] for r in rows], dtype="string"),
            "FLIGHT_ID_mvt": pd.array([r[4] for r in rows], dtype="Int64"),
            "FLIGHT_mvt": pd.array([r[5] for r in rows], dtype="string"),
            "ADES_mvt": pd.array([r[6] for r in rows], dtype="string"),
            "LOBT_flt": [pd.NaT if r[7] is None else t0 + pd.Timedelta(seconds=r[7]) for r in rows],
            # IOBT: the schedule itself for row 1, half an hour before it for row 3
            "IOBT_flt": [t0 - pd.Timedelta(seconds=900), pd.NaT, t0 - pd.Timedelta(seconds=2_700)] + [pd.NaT] * 4,
        },
        index=pd.Index([r[0] for r in rows], name="mvt_id"),
    )


def test_base_is_the_delta_on_suspect_rows_with_a_non_negative_delta():
    mvt = _frame()
    out = stale_schedule(mvt, FeatureContext(dep_index=mvt.index))
    assert list(out.columns) == [
        "stale_base", "stale_suspect", "sched_delta_full", "sched_delta_days", "sched_delta_day_frac",
        "lobt_delta_full", "typed_unjoined", "lobt_minus_sched", "iobt_minus_sched",
    ]
    assert out["stale_suspect"].tolist() == [0, 0, 0, 1, 1, 1, 1]
    assert out["typed_unjoined"].tolist() == [0, 0, 1, 0, 0, 0, 0]
    assert np.isnan(out.loc[[1, 2, 3, 6, 7], "stale_base"]).all()
    assert out.loc[4, "stale_base"] == 900
    assert out.loc[5, "stale_base"] == 90_000
    assert out.loc[7, "sched_delta_full"] == -3_600  # the delta itself is still reported


def test_full_delta_keeps_multi_day_values_and_splits_days():
    mvt = _frame()
    out = stale_schedule(mvt, FeatureContext(dep_index=mvt.index))
    assert out.loc[5, "sched_delta_full"] == 90_000        # nm_deltas would have NaN here
    assert out.loc[5, "sched_delta_days"] == 1
    assert abs(out.loc[5, "sched_delta_day_frac"] - 3_600 / 86_400) < 1e-6
    assert out.loc[1, "sched_delta_days"] == 0 and abs(out.loc[1, "sched_delta_day_frac"] - 900 / 86_400) < 1e-6
    assert np.isnan(out.loc[6, "sched_delta_full"])


def test_lobt_delta_is_unclipped_and_nan_without_an_nm_join():
    mvt = _frame()
    out = stale_schedule(mvt, FeatureContext(dep_index=mvt.index))
    assert out.loc[1, "lobt_delta_full"] == 1_200
    assert out.loc[3, "lobt_delta_full"] == 20_000
    assert np.isnan(out.loc[[2, 4, 5, 6, 7], "lobt_delta_full"]).all()
    # NM's off-block times against the schedule: row 1's LOBT is 300 s before it, its IOBT equals it
    assert out.loc[1, "lobt_minus_sched"] == -300 and out.loc[1, "iobt_minus_sched"] == 0
    assert out.loc[3, "lobt_minus_sched"] == -19_100 and out.loc[3, "iobt_minus_sched"] == -1_800
    assert np.isnan(out.loc[[2, 4, 5, 6, 7], "lobt_minus_sched"]).all()
    assert np.isnan(out.loc[[2, 4, 5, 6, 7], "iobt_minus_sched"]).all()


def _stats(mvt: pd.DataFrame, fit_ids: list[int], target: dict[int, float]) -> pd.DataFrame:
    base = stale_schedule(mvt, FeatureContext(dep_index=mvt.index))
    ctx = FeatureContext(
        dep_index=mvt.index,
        deps={"stale_schedule": base},
        fit_mask=pd.Series(mvt.index.isin(fit_ids), index=mvt.index),
        target=pd.Series(target).reindex(mvt.index),
    )
    return stale_target_stats(mvt, ctx)


def test_stale_rates_come_from_fit_rows_of_the_segment_only():
    mvt = _frame()
    # Row 4 (RYR) is genuine: target 1,000 against a 900 s delta -> within tolerance, stale.
    # Row 5 (ISR) is stale: target equals the delta. Row 7 has no base, so it never counts.
    out = _stats(mvt, fit_ids=[1, 2, 3, 4, 5, 6, 7], target={1: 800, 2: 800, 3: 800, 4: 1_000, 5: 90_000, 6: 700, 7: 700})
    assert out["stale_n_carrier"].loc[4] == 1 and out["stale_n_carrier"].loc[5] == 1
    assert out["stale_n_carrier"].loc[7] == 1          # RYR's count is shared by every RYR row
    assert out["stale_n_carrier_ades"].loc[1] == 1     # RYR|EGKK, looked up for the typed row too
    prior = 1.0                                        # both labelled rows are stale
    assert abs(out["stale_rate_carrier"].loc[4] - (1 + PRIOR_WEIGHT * prior) / (1 + PRIOR_WEIGHT)) < 1e-6
    # A genuine row: target far from the delta.
    out2 = _stats(mvt, fit_ids=[4, 5], target={4: 30_000, 5: 90_000})
    assert abs(out2["stale_rate_carrier"].loc[4] - (0 + PRIOR_WEIGHT * 0.5) / (1 + PRIOR_WEIGHT)) < 1e-6
    assert abs(out2["stale_rate_ades"].loc[5] - (1 + PRIOR_WEIGHT * 0.5) / (1 + PRIOR_WEIGHT)) < 1e-6


def test_stale_rates_change_with_the_fit_mask():
    mvt = _frame()
    target = {4: 30_000, 5: 90_000}   # RYR genuine, ISR stale
    with_both = _stats(mvt, fit_ids=[4, 5], target=target)
    only_isr = _stats(mvt, fit_ids=[5], target=target)
    assert with_both["stale_n_carrier"].loc[4] == 1 and only_isr["stale_n_carrier"].loc[4] == 0
    assert with_both["stale_rate_carrier"].loc[4] < 0.5 < only_isr["stale_rate_carrier"].loc[4]  # RYR: measured vs prior only
    none = _stats(mvt, fit_ids=[], target=target)
    assert (none["stale_rate_carrier"] == 0).all() and (none["stale_n_carrier"] == 0).all()
