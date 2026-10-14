"""`prc.frame_shift`: the whole-frame counts put back on the training-time scale, a block's parity on the training rows,
and the unseen-level verdict. Synthetic frames, no competition data."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import synthetic_tree as st

from prc import data, frame_shift, paths
from prc.features import resolve
from prc.features.build import build_matrix


def _keys(rows: list[tuple]) -> pd.DataFrame:
    """Key columns as the key blocks emit them: (id, apt, stand, runway, carrier, type) per departure."""
    ids, apt, stand, rwy, carrier, typ = map(list, zip(*rows))
    a = pd.Series(apt, dtype="string")
    s = pd.Series(stand, dtype="string")
    r = pd.Series(rwy, dtype="string")
    out = pd.DataFrame({
        "apt": pd.Categorical(apt),
        "apt_stand": pd.Categorical(a + "_" + s),
        "apt_runway": pd.Categorical(a + "_" + r),
        "apt_stand_runway": pd.Categorical(a + "_" + s + "_" + r),
        "stand_prefix": pd.Categorical(a + "_" + s.str[0]),
        "carrier_flightno": pd.Categorical(carrier),
        "aircraft_type": pd.Categorical(typ),
    })
    out.index = pd.Index(ids, name="mvt_id")
    return out


TRAIN = [(i, "AAAA", "S1" if i < 6 else "S2", "R1", "LH", "A320") for i in range(8)] + [
    (100 + i, "BBBB", "S1", "R2", "AZ", "B738") for i in range(4)]
SOURCE_RANK = [(50, "AAAA", "S1", "R1", "LH", "A320"), (51, "AAAA", "S2", "R1", "LH", "A320")]
NEW_RANK = [(70, "AAAA", "S1", "R1", "LH", "A320"), (71, "AAAA", "S1", "R1", "LH", "A320"),
            (72, "AAAA", "S3", "R1", "LH", "A320"), (73, "AAAA", "S3", "R1", "LH", "A320"),
            (90, "BBBB", "S9", "R2", "AZ", "B738")]
BLOCKS = list(frame_shift.COUNT_BLOCKS)


def test_seen_keys_take_the_source_count_and_new_keys_the_scaled_one():
    src, new = _keys(TRAIN + SOURCE_RANK), _keys(TRAIN + NEW_RANK)
    rank = pd.Index([r[0] for r in NEW_RANK])
    corrected, record, seen = frame_shift.rescale_counts(new, src, rank, pd.Index([r[0] for r in TRAIN]), BLOCKS)
    # AAAA: 2 source ranking departures against 4 new ones; BBBB had none in the source set (w unknown -> 1)
    assert record["w"] == {"AAAA": pytest.approx(0.5), "BBBB": None}
    stand = dict(zip(rank, corrected["stand_freq"]))
    assert stand[70] == stand[71] == 7.0           # S1 at AAAA: 6 training + 1 source ranking, transplanted
    assert stand[72] == stand[73] == 1.0           # S3 is new: 0 training + 0.5 x 2 new ranking departures
    assert stand[90] == 1.0                        # S9 at BBBB, new; w falls back to 1
    assert list(seen["stand_freq"]) == [True, True, False, False, False]
    assert list(corrected["runway_freq"][:4]) == [10.0] * 4          # R1 at AAAA: 8 + 2
    assert corrected["stand_runway_freq"][2] == 1.0 and corrected["runway_share_at_stand"][2] == 1.0
    assert corrected["runway_share_at_stand"][0] == np.float32(7.0) / np.float32(7.0)
    assert list(corrected["carrier_apt_freq"][:4]) == [10.0] * 4 and corrected["carrier_apt_share"][0] == 1.0
    assert record["keys"]["apt_stand"] == {"seen": 2, "new": 3, "new_levels": 2}
    assert all(v.dtype == np.float32 for v in corrected.values())


def test_on_the_sources_own_rows_nothing_changes():
    keys = _keys(TRAIN + SOURCE_RANK)
    rank = pd.Index([r[0] for r in SOURCE_RANK])
    corrected, record, _ = frame_shift.rescale_counts(keys, keys, rank, pd.Index([r[0] for r in TRAIN]), BLOCKS)
    values = frame_shift.frame_values(keys, BLOCKS)
    pos = keys.index.get_indexer(rank)
    for col, got in corrected.items():
        assert np.array_equal(got, values[col][pos]), col
    assert record["w"]["AAAA"] == 1.0


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """The two count blocks and their keys, built by the blocks themselves on a prepared synthetic tree."""
    root = tmp_path_factory.mktemp("frame_shift")
    mp = pytest.MonkeyPatch()
    st.retarget_paths(mp, root / "data", root)
    n_train = st.write_training(root / "data", per_month=1, wrong_date=0)
    n_rank = st.write_ranking(root / "data", st.ranking_frame(["2026-01"], seed=2, id_start=900_000, per_month=1))
    mp.setattr(data, "EXPECTED_TRAIN_ROWS", n_train)
    mp.setenv(data.RANK_ROWS_ENV, str(n_rank))
    mvt, _ = data.prepare(force=True)
    specs = resolve(BLOCKS)
    fm = build_matrix(specs, mvt, data.dep_index(mvt), data_version="probe", use_cache=False, verbose=False)
    yield fm
    mp.undo()


def test_the_count_logic_is_the_blocks(built):
    """`COUNT_COLUMNS` / `RATIO_COLUMNS` / `count_keys` reproduce both blocks exactly; a changed value is caught."""
    X = built.X
    keys = X[[c for cols in frame_shift.key_columns(BLOCKS).values() for c in cols]]
    result = frame_shift.check_counts(keys, X, BLOCKS)
    assert set(result["checked"]) == set(frame_shift.COUNT_COLUMNS) | set(frame_shift.RATIO_COLUMNS)
    assert result["mismatch"] == []
    tampered = X.copy()
    tampered.loc[tampered.index[0], "type_apt_freq"] += 1
    assert frame_shift.check_counts(keys, tampered, BLOCKS)["mismatch"] == ["type_apt_freq"]


def test_parity_report_names_the_columns_that_moved(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FEATURE_CACHE", tmp_path / "features")
    spec = next(s for s in resolve(["stand_runway_frequency"]) if s.name == "stand_runway_frequency")
    idx = pd.Index(range(10), name="mvt_id")
    old = pd.DataFrame({"stand_freq": np.arange(10, dtype="float32"), "runway_freq": np.ones(10, dtype="float32"),
                        "stand_runway_freq": np.ones(10, dtype="float32"), "runway_share_at_stand": np.ones(10, dtype="float32")},
                       index=idx)
    src_dir = paths.feature_cache_dir("v_src")
    src_dir.mkdir(parents=True)
    old.to_parquet(src_dir / f"{spec.name}__aaaaaaaaaaaa.parquet")
    source = frame_shift.SourceFrame("v_src", {spec.name: "aaaaaaaaaaaa"}, "refit__x", tmp_path)
    new = old.copy()
    new["stand_freq"] = new["stand_freq"] * 1.2
    new.loc[[8, 9], "stand_freq"] = np.nan
    train = idx[:8]
    lines = []
    out = frame_shift.parity_report([spec], {spec.name: list(old.columns)}, {spec.name: "bbbbbbbbbbbb"}, new, train,
                                    source, "v_new", "refit__y", log=lines.append)
    entry = out[spec.name]
    assert entry["status"] == "changed" and set(entry["changed_columns"]) == {"stand_freq"} and entry["rows"] == 8
    moved = entry["changed_columns"]["stand_freq"]
    assert moved["changed"] == pytest.approx(7 / 8) and moved["mean_ratio"] == pytest.approx(1.2)   # row 0 is 0 either way
    assert any("stand_freq" in line for line in lines)
    memo = list((paths.feature_cache_dir("v_new") / "_parity").glob("*.json"))
    assert len(memo) == 1 and json.loads(memo[0].read_text(encoding="utf-8")) == entry
    again = frame_shift.parity_report([spec], {spec.name: list(old.columns)}, {spec.name: "bbbbbbbbbbbb"},
                                      new.iloc[:0], train, source, "v_new", "refit__y", log=lambda _: None)
    assert again[spec.name] == entry                      # read back, not recomputed
    gone = frame_shift.SourceFrame("v_src", {spec.name: "cccccccccccc"}, "refit__x", tmp_path)
    missing = frame_shift.parity_report([spec], {spec.name: list(old.columns)}, {spec.name: "dddddddddddd"}, new,
                                        train, gone, "v_new", "refit__y", log=lambda _: None)
    assert missing[spec.name]["status"] == "unavailable"


def test_unseen_levels_are_per_airport_for_a_per_airport_model():
    idx = pd.Index(range(8), name="mvt_id")
    apt = pd.Series(["A", "A", "B", "B", "A", "B", "A", "B"], index=idx)
    stand = pd.Series(["s1", "s2", "s1", "s3", "s3", "s1", "s9", "s2"], index=idx)
    fit, rank = idx[:4], idx[4:]
    glob = frame_shift.unseen_rows(stand, apt, fit, rank, per_airport=False)
    local = frame_shift.unseen_rows(stand, apt, fit, rank, per_airport=True)
    assert list(glob) == [False, False, True, False]        # s3 and s2 occur somewhere in the fit
    assert list(local) == [True, False, True, True]         # s3 never at A, s2 never at B
    summary = frame_shift.unseen_summary(local, stand, apt)
    assert summary["rank_rows"] == 3 and summary["by_airport"]["B"] == {"share": 0.5, "rows": 2}
    assert summary["examples"] == {"s3": 1, "s9": 1, "s2": 1}


def test_the_unseen_verdict_reads_against_the_sources_own_share():
    def entry(share, base, by=None, base_by=None):
        return {"rank_share": share, "by_airport": by or {},
                "baseline": None if base is None else {"rank_share": base, "by_airport": base_by or {}}}

    assert frame_shift.unseen_verdict({"stand": entry(0.10, 0.05)}) == []                 # limit 0.15
    assert frame_shift.unseen_verdict({"stand": entry(0.20, 0.05)})                       # beyond 3x
    assert frame_shift.unseen_verdict({"ades": entry(0.015, 0.0)}) == []                  # within +2 points
    assert frame_shift.unseen_verdict({"ades": entry(0.03, 0.0)})
    assert frame_shift.unseen_verdict({"rule": entry(0.09, None)}) == []                  # no baseline: 10 %
    assert frame_shift.unseen_verdict({"rule": entry(1.0, None)})
    one_airport = entry(0.02, 0.01, by={"LIRF": {"share": 0.6, "rows": 5000}, "EDDF": {"share": 0.0, "rows": 9000}},
                        base_by={"LIRF": {"share": 0.01, "rows": 4000}})
    found = frame_shift.unseen_verdict({"runway": one_airport})
    assert len(found) == 1 and "at LIRF" in found[0]                                     # a label changed at one airport
    small = entry(0.0, 0.0, by={"LTFM": {"share": 1.0, "rows": 50}})
    assert frame_shift.unseen_verdict({"runway": small}) == []                            # too few rows to judge alone
