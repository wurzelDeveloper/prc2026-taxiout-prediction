"""The window-counting primitives, checked against brute force.

`window_count` folds (group, time) into a single sorted int64 array so that one
`np.searchsorted` serves every group at once. It is by far the most error-prone
code in the repo -- an off-by-one in the group stride silently counts events
from a neighbouring airport -- so it is verified against an obvious O(n*m)
implementation rather than trusted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prc.features.congestion import count_upto, neighbour_gap, window_count, window_reduce
from prc.features.core import bucket_key, epoch_seconds


def brute_window(ref_t, ref_g, q_t, q_g, back, fwd):
    out = []
    for t, g in zip(q_t, q_g):
        if pd.isna(t) or pd.isna(g):
            out.append(0)
            continue
        n = sum(
            1
            for rt, rg in zip(ref_t, ref_g)
            if not pd.isna(rt) and rg == g and (t - back) <= rt <= (t + fwd)
        )
        out.append(n)
    return np.array(out, dtype="int32")


@pytest.fixture
def events():
    rng = np.random.default_rng(0)
    n = 400
    times = pd.Series(np.sort(rng.integers(0, 20_000, n)).astype("float64"))
    groups = pd.Series(rng.choice(["A", "B", "C"], n))
    return times, groups


@pytest.mark.parametrize("back,fwd", [(300, 0), (900, 0), (600, 600), (0, 900), (3600, 3600)])
def test_window_count_matches_brute_force(events, back, fwd):
    t, g = events
    got = window_count(t, g, t, g, back=back, fwd=fwd)
    want = brute_window(t.tolist(), g.tolist(), t.tolist(), g.tolist(), back, fwd)
    assert np.array_equal(got, want)


def test_groups_never_bleed_into_one_another():
    """The failure this guards: a stride too small, so group B's events land
    inside group A's key range and get counted as A's."""
    t = pd.Series([100.0, 110.0, 120.0, 130.0])
    g = pd.Series(["A", "B", "A", "B"])
    got = window_count(t, g, t, g, back=1000, fwd=1000)
    assert got.tolist() == [2, 2, 2, 2]


def test_counts_are_inclusive_at_both_edges():
    ref_t = pd.Series([0.0, 300.0, 600.0])
    ref_g = pd.Series(["A", "A", "A"])
    q_t = pd.Series([300.0])
    q_g = pd.Series(["A"])
    assert window_count(ref_t, ref_g, q_t, q_g, back=300, fwd=300)[0] == 3
    assert window_count(ref_t, ref_g, q_t, q_g, back=299, fwd=299)[0] == 1


def test_missing_times_are_excluded_not_treated_as_epoch_zero():
    """A NaT cast through int64 becomes INT64_MIN, which would read as a real
    event at an absurd time and corrupt every count in its group."""
    ref_t = pd.Series([100.0, np.nan, 200.0])
    ref_g = pd.Series(["A", "A", "A"])
    q_t = pd.Series([150.0])
    q_g = pd.Series(["A"])
    assert window_count(ref_t, ref_g, q_t, q_g, back=1000, fwd=1000)[0] == 2


def test_query_with_missing_group_scores_zero():
    ref_t = pd.Series([100.0, 200.0])
    ref_g = pd.Series(["A", "A"])
    q_t = pd.Series([150.0, 150.0])
    q_g = pd.Series(["A", None])
    assert window_count(ref_t, ref_g, q_t, q_g, back=1000, fwd=1000).tolist() == [2, 0]


def test_unseen_query_group_scores_zero():
    ref_t = pd.Series([100.0])
    ref_g = pd.Series(["A"])
    q_t = pd.Series([100.0])
    q_g = pd.Series(["Z"])
    assert window_count(ref_t, ref_g, q_t, q_g, back=10, fwd=10)[0] == 0


def test_reference_and_query_may_be_different_sets(events):
    """Arrivals counted around departures: the two sides have different lengths
    and only partly overlapping groups."""
    t, g = events
    ref_t, ref_g = t[:200].reset_index(drop=True), g[:200].reset_index(drop=True)
    q_t, q_g = t[200:].reset_index(drop=True), g[200:].reset_index(drop=True)
    got = window_count(ref_t, ref_g, q_t, q_g, back=900, fwd=0)
    want = brute_window(ref_t.tolist(), ref_g.tolist(), q_t.tolist(), q_g.tolist(), 900, 0)
    assert np.array_equal(got, want)


def test_count_upto_is_a_running_total(events):
    t, g = events
    got = count_upto(t, g, t, g)
    want = np.array(
        [sum(1 for rt, rg in zip(t, g) if rg == gg and rt <= tt) for tt, gg in zip(t, g)],
        dtype="int32",
    )
    assert np.array_equal(got, want)


def test_ground_occupancy_identity():
    """`apt_arr_taxiing_now` is landed-minus-parked; check the two counts compose."""
    landing = pd.Series([0.0, 100.0, 200.0])
    inblock = pd.Series([500.0, 600.0, 700.0])
    apt = pd.Series(["A", "A", "A"])
    q_t = pd.Series([250.0, 650.0])
    q_g = pd.Series(["A", "A"])

    landed = count_upto(landing, apt, q_t, q_g)
    parked = count_upto(inblock, apt, q_t, q_g)
    assert (landed - parked).tolist() == [3, 1]


# --- per-row windows, value reductions, neighbour gaps -----------------------

def _brute_reduce(ref_t, ref_g, ref_v, q_t, q_g, back, fwd, how):
    out = []
    for t, g, b, f in zip(q_t, q_g, back, fwd):
        if pd.isna(t) or pd.isna(g) or pd.isna(b) or pd.isna(f):
            out.append(np.nan)
            continue
        vals = [
            rv for rt, rg, rv in zip(ref_t, ref_g, ref_v)
            if not pd.isna(rt) and rg == g and (t - b) <= rt <= (t + f) and not pd.isna(rv)
        ]
        if not vals:
            out.append(np.nan)
        elif how == "sum":
            out.append(sum(vals))
        elif how == "mean":
            out.append(sum(vals) / len(vals))
        else:
            out.append(max(vals))
    return np.array(out, dtype="float64")


@pytest.fixture
def per_row(events):
    t, g = events
    rng = np.random.default_rng(1)
    back = rng.integers(0, 2000, len(t)).astype("float64")
    fwd = rng.integers(0, 1000, len(t)).astype("float64")
    back[::37] = np.nan  # a query with no window (this flight has no NM off-block time)
    values = rng.normal(600, 200, len(t))
    values[::11] = np.nan
    return t, g, back, fwd, pd.Series(values)


def test_per_row_windows_match_brute_force(per_row):
    t, g, back, fwd, _ = per_row
    got = window_count(t, g, t, g, back=back, fwd=fwd)
    want = np.array(
        [
            0 if np.isnan(b) else brute_window(t.tolist(), g.tolist(), [tt], [gg], b, f)[0]
            for tt, gg, b, f in zip(t, g, back, fwd)
        ],
        dtype="int32",
    )
    assert np.array_equal(got, want)


def test_scalar_windows_are_unchanged_by_the_generalisation(events):
    t, g = events
    scalar = window_count(t, g, t, g, back=900, fwd=300)
    arrays = window_count(t, g, t, g, back=np.full(len(t), 900.0), fwd=np.full(len(t), 300.0))
    assert np.array_equal(scalar, arrays)


@pytest.mark.parametrize("how", ["sum", "mean", "max"])
def test_window_reduce_matches_brute_force(per_row, how):
    t, g, back, fwd, values = per_row
    got = window_reduce(t, g, values, t, g, back=back, fwd=fwd, how=how)
    want = _brute_reduce(t.tolist(), g.tolist(), values.tolist(), t.tolist(), g.tolist(), back, fwd, how)
    assert np.allclose(got, want, equal_nan=True)


def test_window_reduce_gives_nan_not_zero_for_an_empty_window():
    ref_t = pd.Series([0.0, 5000.0])
    ref_g = pd.Series(["A", "A"])
    v = pd.Series([100.0, 200.0])
    q_t = pd.Series([2500.0, 5000.0])
    q_g = pd.Series(["A", "A"])
    got = window_reduce(ref_t, ref_g, v, q_t, q_g, back=60, fwd=0, how="mean")
    assert np.isnan(got[0]) and got[1] == 200.0
    with pytest.raises(ValueError, match="how"):
        window_reduce(ref_t, ref_g, v, q_t, q_g, back=60, fwd=0, how="median")


def test_window_reduce_skips_missing_values_but_counts_the_window_empty_when_all_are_missing():
    ref_t = pd.Series([100.0, 200.0, 300.0])
    ref_g = pd.Series(["A", "A", "A"])
    v = pd.Series([np.nan, 40.0, np.nan])
    q_t = pd.Series([300.0, 100.0])
    q_g = pd.Series(["A", "A"])
    got = window_reduce(ref_t, ref_g, v, q_t, q_g, back=300, fwd=0, how="mean")
    assert got[0] == 40.0 and np.isnan(got[1])


@pytest.mark.parametrize("direction,strict", [("backward", True), ("backward", False), ("forward", True), ("forward", False)])
def test_neighbour_gap_matches_brute_force(events, direction, strict):
    t, g = events
    ref_t, ref_g = t[:250].reset_index(drop=True), g[:250].reset_index(drop=True)
    got = neighbour_gap(ref_t, ref_g, t, g, direction=direction, strict=strict)
    want = []
    for tt, gg in zip(t, g):
        if direction == "backward":
            cands = [rt for rt, rg in zip(ref_t, ref_g) if rg == gg and (rt < tt if strict else rt <= tt)]
            want.append(tt - max(cands) if cands else np.nan)
        else:
            cands = [rt for rt, rg in zip(ref_t, ref_g) if rg == gg and (rt > tt if strict else rt >= tt)]
            want.append(min(cands) - tt if cands else np.nan)
    assert np.allclose(got, np.array(want), equal_nan=True)


def test_neighbour_gap_never_crosses_groups_or_finds_itself():
    t = pd.Series([100.0, 100.0, 400.0])
    g = pd.Series(["A", "B", "A"])
    back = neighbour_gap(t, g, t, g, direction="backward")
    fwd = neighbour_gap(t, g, t, g, direction="forward")
    assert np.isnan(back[0]) and np.isnan(back[1]) and back[2] == 300.0
    assert fwd[0] == 300.0 and np.isnan(fwd[1]) and np.isnan(fwd[2])
    assert np.isnan(neighbour_gap(pd.Series([], dtype="float64"), pd.Series([], dtype="string"), t, g)).all()


# --- key helpers ------------------------------------------------------------

def test_bucket_key_separates_distinct_combinations():
    a = pd.Series(["X", "X", "Y", "Y"])
    b = pd.Series([1, 2, 1, 2])
    key = bucket_key(a, b)
    assert len(set(key.tolist())) == 4


def test_bucket_key_is_stable_for_equal_rows():
    a = pd.Series(["X", "Y", "X"])
    b = pd.Series([1, 2, 1])
    key = bucket_key(a, b)
    assert key[0] == key[2] and key[0] != key[1]


def test_epoch_seconds_keeps_missing_timestamps_missing():
    ts = pd.Series(pd.to_datetime(["2025-01-01 00:00:00", None], utc=True))
    out = epoch_seconds(ts)
    assert out.iloc[0] == 1_735_689_600.0
    assert np.isnan(out.iloc[1])
