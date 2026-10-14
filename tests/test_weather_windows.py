"""The METAR time primitives in `prc.features._wxlib`, checked against brute force.

Every weather feature is one of these lookups. An off-by-one at a window edge,
or a stale report treated as current, would quietly shift all of them -- so
they are verified against obvious implementations rather than trusted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prc.features._wxlib import (
    asof_index,
    flag_seconds,
    inforce_bounds,
    range_reduce,
    runway_heading_deg,
    seconds_since,
    wind_components,
)

A = 90 * 60


def brute_end(t: np.ndarray, i: int) -> float:
    nxt = t[i + 1] if i + 1 < len(t) else np.inf
    return min(nxt, t[i] + A)


def brute_in_force(t: np.ndarray, x: float) -> int:
    earlier = [i for i in range(len(t)) if t[i] <= x]
    if not earlier:
        return -1
    i = earlier[-1]
    return i if x < brute_end(t, i) else -1


@pytest.fixture
def reports():
    """Half-hourly reports with jitter and some long silences (stale beyond A)."""
    rng = np.random.default_rng(1)
    t = np.cumsum(rng.choice([600, 1800, 1800, 1800, 7200], size=120)).astype("float64")
    flag = rng.integers(0, 2, size=len(t)).astype("float64")
    value = rng.normal(size=len(t))
    return t, flag, value


def test_asof_index_matches_brute_force(reports):
    t, _, _ = reports
    q = np.arange(t[0] - 3000, t[-1] + 8000, 97.0)
    want = np.array([brute_in_force(t, x) for x in q])
    assert np.array_equal(asof_index(t, q, A), want)


def test_a_silent_station_does_not_keep_its_last_report_in_force():
    t = np.array([0.0, 1800.0])
    assert asof_index(t, np.array([1800.0 + A - 1, 1800.0 + A]), A).tolist() == [1, -1]


@pytest.mark.parametrize("back,fwd", [(3600, 0), (10800, 0), (0, 3600), (1800, 1800)])
def test_window_reductions_match_brute_force(reports, back, fwd):
    t, flag, value = reports
    q = np.arange(t[0] - 5000, t[-1] + 9000, 131.0)
    lo, hi = inforce_bounds(t, q - back, q + fwd, A)
    for how, vals in (("min", value), ("max", value), ("any", flag)):
        got = range_reduce(vals, lo, hi, how)
        for k, x in enumerate(q):
            members = [i for i in range(len(t)) if t[i] <= x + fwd and brute_end(t, i) > x - back]
            if not members:
                assert np.isnan(got[k])
            else:
                agg = min if how == "min" else max
                assert got[k] == agg(vals[i] for i in members)


def test_range_reduce_skips_missing_values():
    lo, hi = np.array([0]), np.array([3])
    assert range_reduce(np.array([np.nan, 5.0, 2.0]), lo, hi, "min")[0] == 2.0
    assert np.isnan(range_reduce(np.array([np.nan, np.nan]), np.array([0]), np.array([2]), "max")[0])


def painted(t: np.ndarray, flag: np.ndarray, start: int, stop: int) -> np.ndarray:
    """1-second samples of the flag in force, by painting each report's interval."""
    out = np.zeros(stop - start)
    for i in range(len(t)):
        s, e = int(t[i]) - start, int(min(brute_end(t, i), stop)) - start
        out[s:e] = flag[i]
    return out


def test_flag_seconds_matches_one_second_sampling(reports):
    t, flag, _ = reports
    start, stop = int(t[0]) - 5000, int(t[-1]) + 9000
    samples = painted(t, flag, start, stop)
    rng = np.random.default_rng(2)
    a = rng.integers(start, stop - 30_000, 60).astype("float64")
    b = a + rng.choice([3600, 10800, 21600], 60)
    want = [samples[int(ai) - start : int(bi) - start].sum() for ai, bi in zip(a, b)]
    assert np.allclose(flag_seconds(t, flag, a, b, A), want)


def test_seconds_since_matches_brute_force(reports):
    t, flag, _ = reports
    cap = 72 * 3600.0
    q = np.arange(t[0] - 3000, t[-1] + 5000, 233.0)
    got = seconds_since(t, flag, q, cap)
    for k, x in enumerate(q):
        past = [t[i] for i in range(len(t)) if flag[i] and t[i] <= x]
        assert got[k] == (min(x - past[-1], cap) if past else cap)


def test_missing_query_times_stay_missing(reports):
    """A NaT takeoff must produce NaN, not a lookup at some arbitrary time."""
    t, flag, value = reports
    q = np.array([np.nan, t[5]])
    assert asof_index(t, q, A)[0] == -1
    lo, hi = inforce_bounds(t, q - 3600, q, A)
    assert np.isnan(range_reduce(value, lo, hi, "min")[0])
    assert np.isnan(flag_seconds(t, flag, q - 3600, q, A)[0])
    assert np.isnan(seconds_since(t, flag, q, 3600.0)[0])


def test_runway_heading_from_designator():
    got = runway_heading_deg(pd.Series(["25C", "07R", "18", "09L", "NA", "H", "HEL", None]))
    assert got[:4].tolist() == [250.0, 70.0, 180.0, 90.0]
    assert np.isnan(got[4:]).all()


def test_wind_components():
    heading = np.full(5, 250.0)
    head, cross = wind_components(
        np.array([250.0, 70.0, 340.0, np.nan, 0.0]), np.array([10.0, 10.0, 10.0, 4.0, 0.0]), heading
    )
    assert head[0] == pytest.approx(10.0) and cross[0] == pytest.approx(0.0, abs=1e-9)
    assert head[1] == pytest.approx(-10.0)  # straight tailwind
    assert head[2] == pytest.approx(0.0, abs=1e-9) and cross[2] == pytest.approx(10.0)
    assert head[3] == 0.0 and cross[3] == 4.0  # variable wind: all crosswind
    assert head[4] == 0.0 and cross[4] == 0.0  # calm
    h, c = wind_components(np.array([250.0]), np.array([10.0]), np.array([np.nan]))
    assert np.isnan(h[0]) and np.isnan(c[0])
