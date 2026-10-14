"""`_proxylib`, `dep_live_state` and `deice_location`: NM's proxy anomaly, the windows around a takeoff, the de-icing uplift."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prc.features import _proxylib
from prc.features.deice_location import DEICE_LOCATION, deice_location
from prc.features.live_state import LIVE_STATE, dep_live_state
from prc.features.registry import FeatureContext


def at(hhmm: str, day: str = "2025-01-15") -> pd.Timestamp:
    return pd.Timestamp(f"{day}T{hhmm}", tz="UTC")


def _frame(rows) -> pd.DataFrame:
    """rows: (id, apt, stand, runway, takeoff, proxy seconds or None)."""
    takeoff = [r[4] for r in rows]
    aobt = [t - pd.Timedelta(seconds=r[5]) if r[5] is not None else pd.NaT for t, r in zip(takeoff, rows)]
    return pd.DataFrame(
        {
            "apt": [r[1] for r in rows],
            "STAND_mvt": [r[2] for r in rows],
            "RUNWAY_mvt": [r[3] for r in rows],
            "MVT_TIME_UTC_mvt": pd.to_datetime(takeoff, utc=True),
            "AOBT_3_flt": pd.to_datetime(aobt, utc=True),
            "is_dep": True,
        },
        index=pd.Index([r[0] for r in rows], name="mvt_id"),
    )


def test_proxy_anomaly_falls_back_through_the_levels():
    mvt = _frame([
        (1, "EDDF", "A1", "18", at("10:00"), 600),
        (2, "EDDF", "A1", "18", at("10:10"), 700),
        (3, "EDDF", "A1", "18", at("10:20"), 800),
        (4, "EDDF", "B7", "18", at("10:30"), 1000),  # alone at B7: falls back to the (EDDF, 18) median
        (5, "EDDF", "A1", "18", at("10:40"), None),  # no NM off-block time
    ])
    mvt.loc[6] = ["EDDF", "A1", "18", at("10:50"), at("11:00"), True]  # NM off-block after takeoff: not a reading
    a = _proxylib.proxy_anomaly(mvt, min_support=2)
    # the A1/18 pairing's median counts its three readings only
    assert a.loc[[1, 2, 3]].tolist() == [-100.0, 0.0, 100.0]
    assert a.loc[4] == pytest.approx(1000.0 - 750.0)
    assert np.isnan(a.loc[5]) and np.isnan(a.loc[6])


def _live_case():
    # min_support 5: only the airport level has five readings, so every baseline is the airport median (600 s)
    return _frame([
        (1, "EDDF", "A1", "18", at("10:00"), 600),   # anomaly    0
        (2, "EDDF", "A2", "18", at("10:20"), 900),   # anomaly +300
        (3, "EDDF", "B1", "25", at("10:40"), 1200),  # anomaly +600, the other runway and the other pier
        (4, "EDDF", "A1", "18", at("10:50"), 300),   # anomaly -300
        (5, "EDDF", "B2", "18", at("11:40"), 600),   # anomaly    0
        (6, "EDDF", "A2", "18", at("10:45"), None),  # no proxy: gets features, never counts for others
    ])


def test_live_state_windows_exclude_the_flight_itself():
    mvt = _live_case()
    ctx = FeatureContext(dep_index=mvt.index, params={"min_support": 5})
    out = dep_live_state(mvt, ctx)
    assert list(out.columns) == list(LIVE_STATE)
    r4 = out.loc[4]
    assert r4["live_apt_b30_mean"] == pytest.approx(450.0)    # flights 2 and 3
    assert r4["live_apt_b30_share5"] == pytest.approx(0.5)    # +600 only; +300 is not above +300
    assert r4["live_rwy_b30_mean"] == pytest.approx(300.0)    # flight 3 took off from 25
    assert r4["live_pier_b30_mean"] == pytest.approx(300.0)   # flight 2 at pier A
    assert r4["live_apt_b60_mean"] == pytest.approx(300.0)    # flights 1, 2, 3
    assert r4["live_pier_b60_mean"] == pytest.approx(150.0)   # flights 1, 2
    assert r4["live_apt_pm15_mean"] == pytest.approx(600.0)   # flight 3, 10 min earlier
    assert r4["live_apt_pm15_share5"] == pytest.approx(1.0)
    assert np.isnan(out.loc[1, "live_apt_b30_mean"])           # nobody before it: itself is left out
    assert out.loc[5, "live_apt_b60_mean"] == pytest.approx(150.0)  # flights 3 and 4
    assert np.isnan(out.loc[5, "live_apt_b30_mean"])
    assert out.loc[6, "live_apt_b30_mean"] == pytest.approx(450.0)  # a flight without a proxy still reads the others


def _deice_case():
    rows, target, fit, deice = [], [], [], []
    n = 0
    # pier A: 600 s taxis outside de-icing hours, 900 s in them; pier B: 600 s either way (all in January 2025)
    for stand, on_values, off_values in (("A1", (900, 900), (600, 600)), ("B1", (600, 600), (600, 600))):
        for values, flag in ((off_values, 0), (on_values, 1)):
            for v in values:
                n += 1
                rows.append((n, "EDDF", stand, "18", at(f"{8 + n:02d}:00"), v))
                target.append(float(v))
                fit.append(True)
                deice.append(flag)
    # queries, not fit rows and without NM times: a July departure at pier A under de-icing, one at unknown pier C
    rows.append((101, "EDDF", "A1", "18", at("10:00", "2025-07-10"), None))
    rows.append((102, "EDDF", "C4", "18", at("11:00", "2025-01-20"), None))
    target += [np.nan, np.nan]
    fit += [False, False]
    deice += [1, 1]
    mvt = _frame(rows)
    idx = mvt.index
    return mvt, pd.Series(target, index=idx), pd.Series(fit, index=idx), pd.DataFrame({"wx_deice": deice}, index=idx)


def _deice_ctx(mvt, target, fit, wx, shrink=0.0):
    return FeatureContext(dep_index=mvt.index, params={"min_support": 1, "shrink": shrink}, deps={"wx_now": wx}, fit_mask=fit, target=target)


def test_deice_uplift_per_pier_and_the_airport_fallback():
    mvt, target, fit, wx = _deice_case()
    out = deice_location(mvt, _deice_ctx(mvt, target, fit, wx))
    assert list(out.columns) == list(DEICE_LOCATION)
    # pier A: +150 in de-icing hours against -150 outside (both against A1's 750 s median); pier B: 0
    assert out.loc[101, "deice_uplift_pier"] == pytest.approx(300.0)
    assert out.loc[101, "deice_uplift_pier_now"] == pytest.approx(300.0)
    assert out.loc[1, "deice_uplift_pier_now"] == 0.0  # not in de-icing hours
    assert out.loc[5, "deice_uplift_stand"] == pytest.approx(0.0)
    # an unseen pier gets the airport's uplift: +75 against -75
    assert out.loc[102, "deice_uplift_pier"] == pytest.approx(150.0)
    # NM's proxy equals the target here, so the target-free twin agrees
    assert out.loc[101, "deice_proxy_uplift_pier"] == pytest.approx(300.0)


def test_deice_uplift_is_fold_safe_and_shrinks():
    mvt, target, fit, wx = _deice_case()
    base = deice_location(mvt, _deice_ctx(mvt, target, fit, wx))
    # take pier A's de-icing rows out of the fit: its target uplift falls back to the airport's (now 0)
    fit2 = fit.copy()
    fit2.loc[[3, 4]] = False
    moved = deice_location(mvt, _deice_ctx(mvt, target, fit2, wx))
    assert moved.loc[101, "deice_uplift_pier"] == pytest.approx(0.0)
    pd.testing.assert_series_equal(moved["deice_proxy_uplift_pier"], base["deice_proxy_uplift_pier"])  # no target in it
    # two de-icing rows against a shrink of 2: halfway between pier A's 300 and the airport's 150
    shrunk = deice_location(mvt, _deice_ctx(mvt, target, fit, wx, shrink=2.0))
    assert shrunk.loc[101, "deice_uplift_pier"] == pytest.approx(225.0)
