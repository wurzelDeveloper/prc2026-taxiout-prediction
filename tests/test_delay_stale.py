"""Fold-safe stale rates among delayed departures: only fit rows in the delay band teach."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prc.features.delay_stale import PRIOR_WEIGHT, delay_stale_stats
from prc.features.registry import FeatureContext


def _frame():
    t0 = pd.Timestamp("2025-07-01T10:00", tz="UTC")
    rows = []
    # carrier AZ at LIRF: four delayed fit rows, three stale (target == schedule delta)
    for i, stale in enumerate([True, True, True, False]):
        rows.append((i + 1, "LIRF", "AZ100", "801", t0 + pd.Timedelta(hours=i), 2400.0, 2400.0 if stale else 900.0))
    # carrier KL at LIRF: two delayed fit rows, none stale
    rows.append((5, "LIRF", "KL200", "401", t0, 3000.0, 1000.0))
    rows.append((6, "LIRF", "KL200", "401", t0, 3000.0, 1100.0))
    # an AZ row that is NOT in the fit (validation): stale, must not count
    rows.append((7, "LIRF", "AZ100", "801", t0, 2400.0, 2400.0))
    # an on-time AZ row in the fit: outside the delay band, must not count
    rows.append((8, "LIRF", "AZ100", "801", t0, 600.0, 600.0))
    idx = pd.Index([r[0] for r in rows], name="mvt_id")
    mvt = pd.DataFrame(
        {"apt": [r[1] for r in rows], "FLIGHT_mvt": [r[2] for r in rows], "STAND_mvt": [r[3] for r in rows], "is_dep": True},
        index=idx,
    )
    stale_schedule = pd.DataFrame({"sched_delta_full": [r[5] for r in rows], "lobt_delta_full": np.nan}, index=idx)
    target = pd.Series([r[6] for r in rows], index=idx)
    fit = pd.Series(True, index=idx)
    fit.loc[7] = False
    return mvt, stale_schedule, target, fit


def test_rates_read_only_fit_rows_in_the_delay_band():
    mvt, st, target, fit = _frame()
    out = delay_stale_stats(mvt, FeatureContext(dep_index=mvt.index, deps={"stale_schedule": st}, fit_mask=fit, target=target))
    apt_rate = 3 / 6
    az = (3 + PRIOR_WEIGHT * apt_rate) / (4 + PRIOR_WEIGHT)
    kl = (0 + PRIOR_WEIGHT * apt_rate) / (2 + PRIOR_WEIGHT)
    assert out.loc[1, "dstale_n_carrier"] == 4
    assert out.loc[1, "dstale_rate_carrier"] == pytest.approx(az)
    assert out.loc[5, "dstale_rate_carrier"] == pytest.approx(kl)
    # the validation row gets its carrier's rate, computed without it
    assert out.loc[7, "dstale_rate_carrier"] == pytest.approx(az)
    assert out.loc[1, "dstale_rate_apt"] == pytest.approx(apt_rate)


def test_rates_move_with_the_fit_mask():
    mvt, st, target, fit = _frame()
    a = delay_stale_stats(mvt, FeatureContext(dep_index=mvt.index, deps={"stale_schedule": st}, fit_mask=fit, target=target))
    fit2 = fit.copy()
    fit2.loc[[1, 2]] = False
    b = delay_stale_stats(mvt, FeatureContext(dep_index=mvt.index, deps={"stale_schedule": st}, fit_mask=fit2, target=target))
    assert a.loc[1, "dstale_rate_carrier"] != b.loc[1, "dstale_rate_carrier"]
