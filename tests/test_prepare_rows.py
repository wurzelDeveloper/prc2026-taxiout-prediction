"""`prepare`'s ranking-row assertion: ``PRC_EXPECTED_RANK_ROWS`` overrides the Jan+Jul 2026 count for a data tree that
holds another ranking file (the final Feb-Jun 2026 set), nothing changes while it is unset, and `prepare` prints the
ranking rows per month. ``prepare.json`` records the inputs that built the frame, and a frame older than its inputs is
refused without ``--force``. On a synthetic tree (`synthetic_tree`), no competition data."""

from __future__ import annotations

import pytest
import synthetic_tree as st

from prc import data, paths

FINAL_MONTHS = ["2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]


def test_expected_rank_rows_reads_the_override(monkeypatch):
    monkeypatch.delenv(data.RANK_ROWS_ENV, raising=False)
    assert data.expected_rank_rows() == data.EXPECTED_RANK_ROWS == 689_534
    for raw, want in (("1234", 1234), ("1_234", 1234), (" 1,234 ", 1234), ("", 689_534)):
        monkeypatch.setenv(data.RANK_ROWS_ENV, raw)
        assert data.expected_rank_rows() == want
    for bad in ("abc", "0", "-5"):
        monkeypatch.setenv(data.RANK_ROWS_ENV, bad)
        with pytest.raises(ValueError, match=data.RANK_ROWS_ENV):
            data.expected_rank_rows()


def test_prepare_asserts_the_rank_rows_and_prints_them_per_month(tmp_path, monkeypatch, capsys):
    st.retarget_paths(monkeypatch, tmp_path / "data", tmp_path)
    n_train = st.write_training(tmp_path / "data", per_month=1, wrong_date=0)
    n_rank = st.write_ranking(tmp_path / "data", st.ranking_frame(FINAL_MONTHS, seed=3, id_start=500_000, per_month=1))
    monkeypatch.setattr(data, "EXPECTED_TRAIN_ROWS", n_train)

    monkeypatch.delenv(data.RANK_ROWS_ENV, raising=False)
    with pytest.raises(ValueError, match=rf"^expected 689534 ranking rows, got {n_rank}$"):  # unchanged when unset
        data.build_movements()
    monkeypatch.setenv(data.RANK_ROWS_ENV, str(n_rank + 1))
    with pytest.raises(ValueError, match=rf"expected {n_rank + 1} ranking rows \(from {data.RANK_ROWS_ENV}\), got {n_rank}"):
        data.build_movements()

    monkeypatch.setenv(data.RANK_ROWS_ENV, str(n_rank))
    mvt, _ = data.prepare(force=True)
    out = capsys.readouterr().out
    assert "ranking rows per month (UTC):" in out
    table = data.rank_rows_by_month(mvt)
    assert list(table.index) == FINAL_MONTHS and int(table["rows"].sum()) == n_rank
    assert (table["dep"] > 0).all() and (table["arr"] > 0).all()
    for month in FINAL_MONTHS:
        assert month in out


def test_prepare_records_its_inputs_and_refuses_a_changed_file_without_force(tmp_path, monkeypatch, capsys):
    """``interim/prepare.json``: the sha1 of every input and the frame id the feature cache binds a version to. A
    ranking file installed after the frame was built is refused instead of being ignored; ``--force`` rebuilds."""
    st.retarget_paths(monkeypatch, tmp_path / "data", tmp_path)
    n_train = st.write_training(tmp_path / "data", per_month=1, wrong_date=0)
    frame = st.ranking_frame(FINAL_MONTHS, seed=3, id_start=500_000, per_month=1)
    n_rank = st.write_ranking(tmp_path / "data", frame)
    monkeypatch.setattr(data, "EXPECTED_TRAIN_ROWS", n_train)
    monkeypatch.setenv(data.RANK_ROWS_ENV, str(n_rank))
    data.prepare(force=True)
    record = data.prepare_record()
    assert set(record["inputs"]) == {f.name for f in paths.training_files()} | {"ranking.parquet"}
    assert record["frame_id"] == data.frame_id(record["inputs"]) and record["rows"]["rank"] == n_rank
    assert data.changed_inputs(record) == []

    data.prepare()                                                   # unchanged: kept
    assert "inputs are unchanged" in capsys.readouterr().out
    frame.loc[frame.index[0], "STAND_mvt"] = "Z9"                    # a corrected re-release, same ids
    st.write_parquet(frame, paths.RANKING_FILE)
    assert data.changed_inputs(record) == ["ranking.parquet"]
    with pytest.raises(ValueError, match="changed since"):
        data.prepare()
    from prc.cli import main

    assert main(["prepare"]) == 1 and "prepare --force" in capsys.readouterr().err
    data.prepare(force=True)
    assert data.prepare_record()["frame_id"] != record["frame_id"]
    assert data.load_movements().loc[int(frame["MVT_ID_mvt"].iloc[0]), "STAND_mvt"] == "Z9"


def test_a_frame_without_a_record_is_kept_with_a_note(tmp_path, monkeypatch, capsys):
    st.retarget_paths(monkeypatch, tmp_path / "data", tmp_path)
    n_train = st.write_training(tmp_path / "data", per_month=1, wrong_date=0)
    n_rank = st.write_ranking(tmp_path / "data", st.ranking_frame(FINAL_MONTHS, seed=4, id_start=600_000, per_month=1))
    monkeypatch.setattr(data, "EXPECTED_TRAIN_ROWS", n_train)
    monkeypatch.setenv(data.RANK_ROWS_ENV, str(n_rank))
    data.prepare(force=True)
    paths.PREPARE_RECORD.unlink()                                    # a frame built before the record existed
    data.prepare()
    assert "whether they changed is unknown" in capsys.readouterr().out


def test_a_rebuild_that_failed_leaves_the_frame_marked_unfinished(tmp_path, monkeypatch):
    """`prepare --force` writes the record last: a build that fails (a wrong row count) leaves the old frame on disk
    marked as not matching, and both `prepare` and `prc predict` refuse it until a rebuild finishes."""
    from prc.predict import PredictRefused, check_prepared

    st.retarget_paths(monkeypatch, tmp_path / "data", tmp_path)
    n_train = st.write_training(tmp_path / "data", per_month=1, wrong_date=0)
    n_rank = st.write_ranking(tmp_path / "data", st.ranking_frame(FINAL_MONTHS, seed=5, id_start=700_000, per_month=1))
    monkeypatch.setattr(data, "EXPECTED_TRAIN_ROWS", n_train)
    monkeypatch.setenv(data.RANK_ROWS_ENV, str(n_rank))
    data.prepare(force=True)
    assert check_prepared()["frame_id"] == data.prepare_record()["frame_id"]
    monkeypatch.setenv(data.RANK_ROWS_ENV, str(n_rank + 7))          # the wrong count for this file
    with pytest.raises(ValueError, match="ranking rows"):
        data.prepare(force=True)
    assert data.prepare_record()["building"] is True
    with pytest.raises(ValueError, match="did not finish"):
        data.prepare()
    with pytest.raises(PredictRefused, match="did not finish"):
        check_prepared()
    monkeypatch.setenv(data.RANK_ROWS_ENV, str(n_rank))
    data.prepare(force=True)
    assert "building" not in data.prepare_record() and check_prepared()["frame_id"]
