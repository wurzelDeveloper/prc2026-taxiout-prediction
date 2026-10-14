"""The partner-runway block against brute force on a hand-made frame (no competition data needed).

Every output looks strictly before the own takeoff: the windows are [t - w, t), the gaps take the last event before
t. The frame puts events exactly on the window edges (t - 600, t - 1200) and at t itself, a second takeoff on the own
runway at the very same second, traffic on partner and non-partner runways of both pair airports, and an airport
without pairs.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from prc import paths
from prc.features.partner_runway import OUTPUTS, PARTNER, partner_runway
from prc.features.registry import FeatureContext, all_specs

T0 = pd.Timestamp("2025-07-01T08:00", tz="UTC")


def _frame(rows: list[tuple[int, str, str, bool, int, str]]) -> pd.DataFrame:
    """rows: (mvt_id, apt, runway, is_dep, seconds after T0, split)."""
    frame = pd.DataFrame(
        {
            "apt": pd.Categorical([r[1] for r in rows]),
            "RUNWAY_mvt": pd.Categorical([r[2] for r in rows]),
            "is_dep": [r[3] for r in rows],
            "MVT_TIME_UTC_mvt": [T0 + pd.Timedelta(seconds=r[4]) for r in rows],
            "split": pd.Categorical([r[5] for r in rows]),
        },
        index=pd.Index([r[0] for r in rows], name="mvt_id"),
    )
    return frame.sort_values("MVT_TIME_UTC_mvt", kind="stable")


def brute(mvt: pd.DataFrame) -> pd.DataFrame:
    t = (mvt["MVT_TIME_UTC_mvt"] - T0).dt.total_seconds()
    out = {}
    for i, row in mvt.loc[mvt["is_dep"]].iterrows():
        apt, rwy, tq = str(row["apt"]), str(row["RUNWAY_mvt"]), t[i]
        partner = PARTNER.get(apt, {}).get(rwy)
        if partner is None:
            out[i] = {c: np.nan for c in OUTPUTS} | {"prw_has_partner": 0}
            continue
        other = mvt.index != i  # the flight itself, explicitly
        same_apt = (mvt["apt"].astype(str) == apt).to_numpy()
        on_p = same_apt & (mvt["RUNWAY_mvt"].astype(str) == partner).to_numpy() & other
        on_own = same_apt & (mvt["RUNWAY_mvt"].astype(str) == rwy).to_numpy() & other
        arr, dep = (~mvt["is_dep"]).to_numpy(), mvt["is_dep"].to_numpy()
        tt = t.to_numpy()

        def n(mask, w):
            return float(((tt >= tq - w) & (tt < tq) & mask).sum())

        def gap(mask):
            before = tt[mask & (tt < tq)]
            return min(tq - before.max(), 7200.0) if before.size else np.nan

        out[i] = {
            "prw_has_partner": 1,
            "prw_arr_10": n(on_p & arr, 600),
            "prw_arr_20": n(on_p & arr, 1200),
            "prw_dep_10": n(on_p & dep, 600),
            "prw_gap_prev_arr": gap(on_p & arr),
            "prw_gap_prev_mvt": gap(on_p),
            "prw_pair_mvt_10": n(on_p | on_own, 600),
        }
    return pd.DataFrame.from_dict(out, orient="index")[list(OUTPUTS)]


ROWS = [
    # id, apt, runway, is_dep, seconds, split
    (1, "LTFM", "35L", True, 3000, "train"),   # the query
    (2, "LTFM", "35L", True, 3000, "train"),   # another takeoff on the own runway at the same second
    (3, "LTFM", "35R", False, 3000, "train"),  # partner landing exactly at t: excluded
    (4, "LTFM", "35R", False, 2400, "train"),  # partner landing exactly at t - 600: in both windows
    (5, "LTFM", "35R", False, 1800, "train"),  # partner landing exactly at t - 1200: in the 20-min window only
    (6, "LTFM", "35R", False, 1799, "train"),  # just outside
    (7, "LTFM", "35R", True, 2900, "train"),   # partner takeoff
    (8, "LTFM", "35L", False, 2950, "train"),  # landing on the own runway: pair count only
    (9, "LTFM", "36", False, 2990, "train"),   # independent runway: never counted
    (10, "LTFM", "36", True, 2995, "train"),   # departure without partner
    (11, "LTFM", "17R", True, 2999, "rank"),   # another pair: never counted for 35L
    (12, "LTFM", "17L", False, 2500, "rank"),
    (13, "LFPG", "35R", False, 2980, "train"),  # same designator at another airport: never counted
    (14, "LFPG", "08L", True, 3000, "rank"),
    (15, "LFPG", "08R", False, 2700, "rank"),
    (16, "LFPG", "08R", False, 3001, "rank"),  # after t
    (17, "LFPG", "08L", True, 2600, "train"),
    (18, "EDDF", "25C", True, 3000, "train"),  # airport without pairs
    (19, "EDDF", "07C", False, 2950, "train"),
    (20, "LTFM", "35L", True, 12000, "train"),  # partner's last landing long ago: gap capped at 7,200
    (21, "LTFM", "16R", True, 500, "train"),   # first movement on its pair: gaps NaN, counts 0
]


def test_hand_cases():
    mvt = _frame(ROWS)
    dep_index = mvt.index[mvt["is_dep"].to_numpy()]
    out = partner_runway(mvt, FeatureContext(dep_index=dep_index))
    assert list(out.columns) == list(OUTPUTS)
    assert out.index.equals(dep_index)
    q = out.loc[1]
    assert q["prw_has_partner"] == 1
    assert q["prw_arr_10"] == 1          # id 4 only (3 is at t)
    assert q["prw_arr_20"] == 2          # ids 4, 5
    assert q["prw_dep_10"] == 1          # id 7
    assert q["prw_gap_prev_arr"] == 600  # id 4
    assert q["prw_gap_prev_mvt"] == 100  # id 7
    assert q["prw_pair_mvt_10"] == 3     # ids 4, 7, 8 (not 1, 2, 3 at t)
    assert out.loc[2].equals(out.loc[1])  # the same-second twin sees the same context
    for i in (10, 18):
        assert out.loc[i, "prw_has_partner"] == 0
        assert out.loc[i, list(OUTPUTS[1:])].isna().all()
    assert out.loc[14, "prw_arr_10"] == 1 and out.loc[14, "prw_gap_prev_arr"] == 300
    assert out.loc[14, "prw_pair_mvt_10"] == 2  # ids 15, 17
    assert out.loc[20, "prw_gap_prev_arr"] == 7200
    assert np.isnan(out.loc[21, "prw_gap_prev_arr"]) and out.loc[21, "prw_arr_10"] == 0


def test_hand_cases_match_brute_force():
    mvt = _frame(ROWS)
    dep_index = mvt.index[mvt["is_dep"].to_numpy()]
    got = partner_runway(mvt, FeatureContext(dep_index=dep_index)).astype("float64")
    want = brute(mvt).loc[dep_index].astype("float64")
    pd.testing.assert_frame_equal(got, want, check_names=False)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_frame_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = 500
    apts = rng.choice(["LTFM", "LFPG", "EDDF"], n)
    runways = {
        "LTFM": ["16L", "16R", "34L", "34R", "17L", "17R", "35L", "35R", "36", "18"],
        "LFPG": ["08L", "08R", "26L", "26R", "09L", "09R", "27L", "27R", "NA"],
        "EDDF": ["25C", "07C", "18"],
    }
    rows = [
        (i + 1, a, str(rng.choice(runways[a])), bool(rng.random() < 0.5), int(rng.integers(0, 6000)), "train" if rng.random() < 0.7 else "rank")
        for i, a in enumerate(apts)
    ]
    # force exact-edge and same-second coincidences
    rows += [(n + 1, "LTFM", "35R", False, rows[0][4] - 600, "train"), (n + 2, "LTFM", "35R", True, rows[0][4], "rank")]
    mvt = _frame(rows)
    dep_index = mvt.index[mvt["is_dep"].to_numpy()]
    got = partner_runway(mvt, FeatureContext(dep_index=dep_index)).astype("float64")
    want = brute(mvt).loc[dep_index].astype("float64")
    pd.testing.assert_frame_equal(got, want, check_names=False)


def test_outputs_unique_across_registered_blocks():
    seen: dict[str, str] = {}
    for spec in all_specs().values():
        for col in spec.outputs:
            assert col not in seen, f"'{col}' declared by both {seen[col]} and {spec.name}"
            seen[col] = spec.name
    assert all(seen[c] == "partner_runway" for c in OUTPUTS)
    assert all_specs()["partner_runway"].group == "partner_runway"


def test_block_is_in_no_config_but_the_ab_variant():
    """The submission line's configs must not pick the block up: only exp071 (the A/B arm) names it."""
    pattern = re.compile(r"\bpartner_runway\b")

    def code(path) -> str:  # comments may name the block (exp070 explains the A/B); only YAML content counts
        return "\n".join(line.split("#", 1)[0] for line in path.read_text(encoding="utf-8").splitlines())

    hits = sorted(p.name for p in paths.CONFIGS.glob("*.yaml") if pattern.search(code(p)))
    assert set(hits) <= {"exp071_ltfm_spec_long_partner.yaml"}, hits


def test_sham_keeps_structure_and_breaks_association():
    """`partner_runway_sham` (the A/B's sham control): the indicator and the NaN pattern of the real block, its paired
    rows a per-airport permutation of the real ones, deterministic in the seed, and most paired rows changed."""
    from prc.features.partner_runway_sham import OUTPUTS as SHAM
    from prc.features.partner_runway_sham import partner_runway_sham

    rng = np.random.default_rng(7)
    runways = {"LTFM": ["16L", "16R", "35L", "35R", "36"], "LFPG": ["08L", "08R", "27R"], "EDDF": ["25C"]}
    rows = []
    for i in range(600):
        a = str(rng.choice(list(runways)))
        rows.append((i + 1, a, str(rng.choice(runways[a])), bool(rng.random() < 0.6), int(rng.integers(0, 20000)), "train"))
    mvt = _frame(rows)
    dep_index = mvt.index[mvt["is_dep"].to_numpy()]
    real = partner_runway(mvt, FeatureContext(dep_index=dep_index))
    sham = partner_runway_sham(mvt, FeatureContext(dep_index=dep_index))
    assert list(sham.columns) == list(SHAM) and sham.index.equals(dep_index)
    named = sham.set_axis(list(OUTPUTS), axis=1)
    again = partner_runway_sham(mvt, FeatureContext(dep_index=dep_index)).set_axis(list(OUTPUTS), axis=1)
    pd.testing.assert_frame_equal(named, again)
    other = partner_runway_sham(mvt, FeatureContext(dep_index=dep_index, params={"seed": 1}))
    assert not other.set_axis(list(OUTPUTS), axis=1).equals(named)

    has = (real["prw_has_partner"] == 1).to_numpy()
    assert (named["prw_has_partner"] == real["prw_has_partner"]).all()
    assert named.loc[~has, list(OUTPUTS[1:])].isna().all().all()
    apt = mvt.loc[dep_index, "apt"].astype(str).to_numpy()
    for a in ("LTFM", "LFPG"):
        m = has & (apt == a)
        assert m.sum() > 20

        def rows_of(f):
            return sorted(map(tuple, f.loc[m, list(OUTPUTS[1:])].fillna(-1.0).to_numpy().tolist()))

        assert rows_of(named) == rows_of(real)
    moved = (named.loc[has, list(OUTPUTS[1:])].fillna(-1.0) != real.loc[has, list(OUTPUTS[1:])].fillna(-1.0)).any(axis=1)
    assert moved.mean() > 0.5
