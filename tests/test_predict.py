"""`prc predict` end to end on a synthetic data tree.

Three tiny CPU runs (one global CatBoost on a log1p target, one per airport on the stale residual base, one per-airport
regime mixture with two classifiers, an anchor and the join window) train with a refit on all twelve months. `prc
predict` must then rebuild their features and models from the run directories and reproduce the refit's ranking
predictions -- from the cache, from scratch under a new ``data.version``, and for a run saved before the state files
existed -- and refuse what it cannot do safely. A second data tree with a new ranking set (Feb-Jun 2026, new stands and
a new destination) is the final submission's situation: its whole-frame counts are put back on the training-time
scale, the parity of every block on the training rows is measured, and its unseen levels are judged against the
source's own ranking rows. No competition data, no GPU, everything under ``tmp_path``.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import synthetic_tree as st

from prc import data, paths
from prc.config import load_config
from prc.models.mixture import STATE_SUFFIX as MIXTURE_STATE
from prc.models.per_airport import STATE_SUFFIX as PER_AIRPORT_STATE

BLOCKS = [
    "core_keys", "stand_runway_keys", "stand_runway_frequency", "stand_runway_target_stats",
    "stale_schedule", "stale_target_stats", "aircraft_identity", "operator_presence",
]
CB = {"task_type": "CPU", "iterations": 40, "depth": 3, "learning_rate": 0.2, "thread_count": 2, "verbose": 0}
MIXTURE = {
    "enabled": True, "bases": ["sched_delta_full"], "universe_column": "stale_suspect", "min_base_sec": 1800,
    "tol_sec": 60, "regressor_max_target_sec": 7200,
    "classifier_params": {"iterations": 30, "depth": 3, "verbose": 0, "thread_count": 2},
    "classifier_weight_power": 1, "classifier_seeds": 2,
    "regressor_anchor": "lobt_delta_full", "anchor_clip": [-200000.0, 200000.0], "window_column": "lobt_delta_full",
}
FINAL_MONTHS = ["2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
NEW_STANDS = ("C1", "C2", "C3")
COUNT_COLUMNS = ("stand_freq", "runway_freq", "stand_runway_freq", "runway_share_at_stand",
                 "carrier_apt_freq", "carrier_apt_share", "type_apt_freq", "carrier_stand_concentration")


def _configs() -> dict:
    base = load_config("base.yaml").replace(
        data={"version": "syn1", "clean": {"day_offset_tol_sec": 3600.0}},
        features={"include": BLOCKS, "params": {"stand_runway_target_stats": {"min_support": 3}}},
        model={"refit_on_all": True, "params": CB},
        evaluation={"permutation": {"enabled": False}},
    )
    return {
        "global": base.replace(name="syn_global", target={"transform": "log1p", "clip": [30.0, 7200.0]}),
        "per_airport": base.replace(name="syn_apt", target={"residual_of": "stale_base"}, model={"per_airport": True}),
        "mixture": base.replace(name="syn_mix", model={"per_airport": True, "mixture": MIXTURE}),
    }


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """The repository layout under a temporary root: ``data/`` (the trained-on tree), ``cache/``, ``runs/``."""
    root = tmp_path_factory.mktemp("predict_world")
    mp = pytest.MonkeyPatch()
    st.retarget_paths(mp, root / "data", root)
    n_train = st.write_training(root / "data")
    n_rank = st.write_ranking(root / "data", st.ranking_frame(["2026-01", "2026-07"], seed=1, id_start=1_000_000))
    mp.setattr(data, "EXPECTED_TRAIN_ROWS", n_train)
    mp.setenv(data.RANK_ROWS_ENV, str(n_rank))
    data.prepare(force=True)
    from prc.train import run_experiment

    runs = {kind: run_experiment(cfg).run_id for kind, cfg in _configs().items()}
    yield SimpleNamespace(root=root, tree=root / "data", runs=runs, n_train=n_train, n_rank=n_rank)
    mp.undo()


def _second_tree(world, monkeypatch, name: str, frame: pd.DataFrame, edit_training=None) -> Path:
    """A data tree beside the first: the same twelve training files (``edit_training`` may change one), the given
    ranking rows, prepared; `prc.paths` points at it (as ``PRC_DATA_DIR`` would), the cache and the runs stay shared."""
    tree = world.root / name
    if not (tree / "interim" / "movements.parquet").exists():
        (tree / "competition_files").mkdir(parents=True, exist_ok=True)
        for f in sorted((world.tree / "competition_files").glob("training_*.parquet")):
            shutil.copyfile(f, tree / "competition_files" / f.name)
        if edit_training is not None:
            edit_training(tree / "competition_files")
        n_rank = st.write_ranking(tree, frame)
        st.retarget_paths(monkeypatch, tree, world.root)
        monkeypatch.setenv(data.RANK_ROWS_ENV, str(n_rank))
        data.prepare(force=True)
    else:
        st.retarget_paths(monkeypatch, tree, world.root)
    monkeypatch.setenv("PRC_DATA_DIR", str(tree))
    return tree


def _final_frame() -> pd.DataFrame:
    return st.ranking_frame(
        FINAL_MONTHS, seed=7, id_start=2_000_000, stands=st.STANDS + NEW_STANDS, destinations=st.DESTINATIONS + ("ZZZZ",)
    )


@pytest.fixture
def final_tree(world, monkeypatch):
    """The final submission's situation: a new ranking set (Feb-Jun 2026) with new stands and a new destination."""
    return _second_tree(world, monkeypatch, "data_final", _final_frame())


def _copy_run(world, kind: str, suffix: str):
    """A copy of a trained run under a new id, to edit without touching the original."""
    src = paths.run_dir(world.runs[kind])
    dst = paths.run_dir(f"{src.name}__{suffix}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def _preds(run_dir) -> pd.Series:
    return pd.read_parquet(run_dir / "test_predictions.parquet")["y_pred"]


def _record(run) -> dict:
    return json.loads((run.dir / "predict.json").read_text(encoding="utf-8"))


# --- reproduction -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["global", "per_airport", "mixture"])
def test_predict_reproduces_the_refit_predictions(world, kind):
    from prc.predict import predict_run

    src = paths.run_dir(world.runs[kind])
    run = predict_run(world.runs[kind], threads=2)
    want, got = _preds(src), _preds(run.dir)
    assert got.index.equals(want.index)
    assert np.allclose(got.to_numpy(), want.to_numpy(), rtol=0, atol=1e-6)
    record = _record(run)
    assert record["source_run"] == src.name and record["data_version"] == "syn1" and record["same_tree"]
    assert record["reproduction"]["comparable"] and record["reproduction"]["within_tol"]
    assert {b["status"] for b in record["blocks"].values()} == {"same"} and record["external"] == {}
    assert all(len(b["code_hash"]) == 12 for b in record["blocks"].values())
    assert record["rows"]["refit_fit"] < record["rows"]["training"]  # the wrong-date rows left the fit
    assert set(record["inputs"]) == {"movements.parquet", "target.parquet", "ranking.parquet", "submitting.parquet"}
    assert record["prepare"]["frame_id"] and record["training_target"]["checked"]
    # on the source's own rows the count rescale transplants every value it already had, and parity is trivial
    rescale = record["count_rescale"]
    assert rescale["applied"] and set(rescale["columns"]) == set(COUNT_COLUMNS)
    assert all(c["rows_changed"] == 0 for c in rescale["columns"].values())
    assert all(v == pytest.approx(1.0) for v in rescale["w"].values())
    assert not rescale["checks"]["source"]["mismatch"] and set(rescale["checks"]["new"]["checked"]) == set(COUNT_COLUMNS)
    assert {e["status"] for e in record["parity"].values()} == {"identical"}
    assert not record["unseen_exceeded"]
    assert (run.dir / "oof_predictions.parquet").exists() and (run.dir / "refit.json").exists()
    assert (run.dir / "refit_detail.parquet").exists() == (kind == "mixture")
    metrics = json.loads((run.dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["predict"]["source_run"] == src.name and metrics["overall"] == json.loads(
        (src / "metrics.json").read_text(encoding="utf-8"))["overall"]
    env = json.loads((run.dir / "env.json").read_text(encoding="utf-8"))
    assert env["data_dir"] == str(paths.DATA)


def test_a_new_data_version_recomputes_every_block_from_the_refit_rows(world):
    """No cache to lean on: the fold-dependent blocks are rebuilt from the cleaned refit rows and still reproduce. Fitted
    on the uncleaned rows instead, the stale rates would differ -- the check has teeth. Every block's training-row
    parity against the source's cache is exact."""
    from prc import cleaning
    from prc.features import resolve
    from prc.features.build import build_matrix
    from prc.predict import predict_run

    kind = "mixture"
    run = predict_run(world.runs[kind], ["data.version=syn1b"], threads=2, tree=paths.DATA)
    assert np.allclose(_preds(run.dir).to_numpy(), _preds(paths.run_dir(world.runs[kind])).to_numpy(), rtol=0, atol=1e-6)
    assert (paths.FEATURE_CACHE / "syn1b").is_dir()
    parity = _record(run)["parity"]
    assert set(BLOCKS) <= set(parity) and {e["status"] for e in parity.values()} == {"identical"}
    assert all(e.get("rows", 0) > 0 for e in parity.values())   # compared, not the same file
    assert any((paths.FEATURE_CACHE / "syn1b" / "_parity").glob("*.json"))

    mvt, y = data.load_movements(), data.load_target()
    dep_idx = data.dep_index(mvt)
    specs = resolve(["stale_target_stats"])
    flagged = cleaning.flag_rows(mvt, y, _configs()[kind].data.clean)
    assert flagged.sum() > 0

    def stats(fit_index):
        mask = pd.Series(False, index=mvt.index)
        mask.loc[fit_index] = True
        return build_matrix(specs, mvt, dep_idx, data_version="probe", fit_mask=mask, target=y.reindex(dep_idx),
                            use_cache=False, verbose=False).X
    assert not stats(y.index[~flagged.to_numpy()]).equals(stats(y.index))


def test_a_run_saved_before_the_state_files_is_restored_from_its_training_rows(world):
    from prc.predict import predict_run

    legacy = _copy_run(world, "mixture", "legacy")
    for suffix in (MIXTURE_STATE, PER_AIRPORT_STATE):
        (legacy / "model" / f"refit{suffix}").unlink()
    env = json.loads((legacy / "env.json").read_text(encoding="utf-8"))
    for key in ("data_dir", "prc_data_dir"):
        env.pop(key)  # nor did such runs record their data tree: data/ beside runs/ is assumed
    (legacy / "env.json").write_text(json.dumps(env), encoding="utf-8")

    run = predict_run(legacy.name, threads=2)
    assert np.allclose(_preds(run.dir).to_numpy(), _preds(legacy).to_numpy(), rtol=0, atol=1e-6)
    record = _record(run)
    state = record["model"]["state"]
    assert state["mixture"]["recomputed"] == ["anchor_fallback"]
    assert state["per_airport"]["recomputed"] == ["fallback", "sizes"]
    assert record["source_data_tree_recorded"] is False and record["same_tree"] is True


# --- the final tree -------------------------------------------------------------------------------------------------

def test_another_data_tree_under_the_same_data_version_is_refused(world, final_tree):
    """The stale-cache trap: refused before anything is written, by predict and by the cache itself."""
    from prc.features import resolve
    from prc.features.build import build_matrix
    from prc.predict import PredictRefused, predict_run

    runs_before = sorted(p.name for p in paths.RUNS.iterdir())
    with pytest.raises(PredictRefused, match="data.version"):
        predict_run(world.runs["global"], threads=2)
    assert sorted(p.name for p in paths.RUNS.iterdir()) == runs_before

    mvt = data.load_movements()
    with pytest.raises(ValueError, match="another data tree"):
        build_matrix(resolve(["core_keys"]), mvt, data.dep_index(mvt), data_version="syn1", verbose=False)


def test_a_new_data_version_must_name_its_tree(world, final_tree, tmp_path):
    """``--tree``: a session whose PRC_DATA_DIR is not the tree a new version is for cannot fill that version's cache."""
    from prc.predict import PredictRefused, predict_run

    with pytest.raises(PredictRefused, match="--tree"):
        predict_run(world.runs["global"], ["data.version=final_guard"], threads=2)
    with pytest.raises(PredictRefused, match="not the data tree this process reads"):
        predict_run(world.runs["global"], ["data.version=final_guard"], threads=2, tree=world.tree)
    assert not (paths.FEATURE_CACHE / "final_guard").exists()


def test_a_version_belongs_to_the_tree_that_built_it(world, final_tree, monkeypatch):
    """The cache stamp: ``final1`` built from the final tree is refused to a session reading another tree, and to a
    frame re-prepared from another ranking file under the same version."""
    from prc.features import resolve
    from prc.features.build import CacheTreeMismatch, build_matrix, check_tree_stamp
    from prc.predict import PredictRefused, predict_run

    mvt = data.load_movements()
    build_matrix(resolve(["core_keys"]), mvt, data.dep_index(mvt), data_version="final1", verbose=False)
    stamp = json.loads((paths.FEATURE_CACHE / "final1" / "TREE.json").read_text(encoding="utf-8"))
    assert paths.same_dir(stamp["data_tree"], final_tree) and stamp["frame_id"] == data.prepare_record()["frame_id"]

    st.retarget_paths(monkeypatch, world.tree, world.root)        # a session without PRC_DATA_DIR
    with pytest.raises(CacheTreeMismatch, match="belongs to the data tree"):
        check_tree_stamp("final1")
    with pytest.raises(PredictRefused, match="belongs to the data tree"):
        predict_run(world.runs["global"], ["data.version=final1"], threads=2, tree=world.tree)

    st.retarget_paths(monkeypatch, final_tree, world.root)
    record = data.prepare_record()
    other = json.loads(json.dumps(stamp))
    other["frame_id"] = "0" * 16
    (paths.FEATURE_CACHE / "final1" / "TREE.json").write_text(json.dumps(other), encoding="utf-8")
    try:
        with pytest.raises(CacheTreeMismatch, match="another movement frame"):
            check_tree_stamp("final1")
    finally:
        (paths.FEATURE_CACHE / "final1" / "TREE.json").write_text(json.dumps(stamp), encoding="utf-8")
    assert record["frame_id"] == stamp["frame_id"]


@pytest.mark.parametrize("kind", ["global", "per_airport", "mixture"])
def test_the_final_tree_is_predicted_under_its_own_data_version(world, final_tree, kind):
    from prc.predict import predict_run

    run = predict_run(world.runs[kind], ["data.version=final1"], threads=2, tree=final_tree, allow_unseen=True)
    mvt = data.load_movements()
    rank = data.dep_index(mvt, "rank")
    got = _preds(run.dir)
    assert got.index.equals(rank) and np.isfinite(got.to_numpy()).all()
    months = sorted(set(mvt.loc[rank, "MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")))
    assert months == FINAL_MONTHS
    record = _record(run)
    assert record["data_tree"] == str(final_tree) and record["prc_data_dir"] == str(final_tree)
    assert record["data_version"] == "final1" and record["source_data_version"] == "syn1" and not record["same_tree"]
    assert record["reproduction"]["comparable"] is False
    assert record["training_target"]["checked"] is True and record["training_target"]["rows"] == world.n_train // 2
    stand = record["unseen_levels"]["stand"]
    assert stand["rank_rows"] > 0 and set(stand["examples"]) <= set(NEW_STANDS)
    assert stand["baseline"] is not None and stand["baseline"]["rank_share"] == 0.0
    assert set(record["unseen_levels"]["ades"]["examples"]) == {"ZZZZ"}
    assert record["unseen_exceeded"] and record["unseen_allowed"]
    # the training rows are the same twelve months: the fold-dependent blocks see the same fit rows
    assert record["rows"]["training"] == world.n_train // 2
    submit_ids = pd.read_parquet(paths.SUBMITTING_FILE)["MVT_ID_mvt"].astype("int64")
    assert set(submit_ids) == set(got.index)


def test_the_final_trees_counts_are_put_back_on_the_training_time_scale(world, final_tree):
    """Five ranking months instead of two: every count over the frame grows. The ranking rows get the source frame's
    own count for a key it had, and ``w * n_rank_new`` (w = source ranking departures over new, per airport) for a
    new key; the training rows' counts show the inflation in the parity report."""
    from prc.predict import predict_run

    run = predict_run(world.runs["mixture"], ["data.version=final1"], threads=2, tree=final_tree, allow_unseen=True)
    rescale = _record(run)["count_rescale"]
    assert rescale["applied"] and rescale["source_keys"] == "source cache"
    assert rescale["w"] == pytest.approx({a: 0.4 for a in st.AIRPORTS})   # 4 ranking days against 10, per airport
    assert not rescale["checks"]["source"]["mismatch"] and not rescale["checks"]["new"]["mismatch"]

    # independently: the source frame's counts per (airport, stand) and the final frame's new stands
    def stand_keys(tree: Path) -> tuple[pd.Series, np.ndarray]:
        frame = pd.read_parquet(tree / "interim" / "movements.parquet", columns=["apt", "STAND_mvt", "is_dep", "split"])
        dep = frame.loc[frame["is_dep"]]
        keys = (dep["apt"].astype(str) + "_" + dep["STAND_mvt"].astype(str)).astype(object)
        return keys, dep["split"].astype(str).eq("rank").to_numpy()

    def lookup(keys: pd.Series, counts: pd.Series) -> np.ndarray:
        return np.array(keys.map(counts.to_dict()).to_numpy(dtype="float64", na_value=np.nan))

    src_keys, _ = stand_keys(world.tree)
    new_keys, is_rank = stand_keys(final_tree)
    rank_keys = new_keys[is_rank]
    expect = lookup(rank_keys, src_keys.value_counts())
    new = np.isnan(expect)
    expect[new] = 0.4 * lookup(rank_keys[new], rank_keys.value_counts())
    before = lookup(rank_keys, new_keys.value_counts())
    col = rescale["columns"]["stand_freq"]
    assert col["mean_after"] == pytest.approx(float(expect.mean()), rel=1e-6)
    assert col["mean_before"] == pytest.approx(float(before.mean()), rel=1e-6)
    assert col["mean_after"] < col["mean_before"] and col["inflation_seen"]["mean"] > 1.0
    assert rescale["keys"]["apt_stand"]["new"] == int(new.sum()) > 0

    parity = _record(run)["parity"]
    moved = parity["stand_runway_frequency"]
    assert moved["status"] == "changed" and moved["changed_columns"]["stand_freq"]["mean_ratio"] > 1.0
    assert parity["stale_schedule"]["status"] == "identical"   # a row-wise block does not move

    unscaled = predict_run(world.runs["mixture"], ["data.version=final1"], threads=2, tree=final_tree,
                           allow_unseen=True, count_rescale=False)
    assert _record(unscaled)["count_rescale"] == {"applied": False, "reason": "--no-count-rescale"}
    other = _record(predict_run(world.runs["per_airport"], ["data.version=final1"], threads=2, tree=final_tree,
                                allow_unseen=True))["count_rescale"]
    assert other["memo"] and other["columns"]["stand_freq"] == col   # same key blocks: the first member's work, re-used


def test_unseen_levels_far_beyond_the_sources_own_are_refused(world, final_tree):
    """Three new stands and a new destination on a sixth of the rows, against none on the source's own ranking rows."""
    from prc.predict import PredictRefused, predict_run

    runs_before = sorted(p.name for p in paths.RUNS.iterdir())
    with pytest.raises(PredictRefused, match="--allow-unseen") as caught:
        predict_run(world.runs["per_airport"], ["data.version=final1"], threads=2, tree=final_tree)
    assert "stand" in str(caught.value) and "ades" in str(caught.value)
    assert sorted(p.name for p in paths.RUNS.iterdir()) == runs_before


def test_other_training_targets_than_the_sources_are_refused(world, monkeypatch):
    """A tree whose training files were edited (one target changed) would refit the fold-dependent blocks on other
    rows: refused."""
    from prc.predict import PredictRefused, predict_run

    def edit(folder: Path) -> None:
        f = sorted(folder.glob("training_*.parquet"))[3]
        frame = pd.read_parquet(f)
        dep = frame.index[frame["PHASE_mvt"].eq("DEP")][:5]
        frame.loc[dep, "TAXITIME_SEC_mvt"] = frame.loc[dep, "TAXITIME_SEC_mvt"] + 1
        st.write_parquet(frame, f)

    tree = _second_tree(world, monkeypatch, "data_edited", st.ranking_frame(["2026-03"], seed=9, id_start=4_000_000), edit)
    with pytest.raises(PredictRefused, match="training targets"):
        predict_run(world.runs["global"], ["data.version=edited1"], threads=2, tree=tree)


def test_a_ranking_file_installed_after_prepare_is_refused(world, final_tree):
    """``prepare.json`` records the files that built the frame: a re-release installed without ``prepare --force``."""
    from prc.predict import PredictRefused, predict_run

    original = paths.RANKING_FILE.read_bytes()
    frame = pd.read_parquet(paths.RANKING_FILE)
    frame.loc[frame.index[0], "STAND_mvt"] = "Z9"
    st.write_parquet(frame, paths.RANKING_FILE)
    try:
        with pytest.raises(PredictRefused, match="prepare --force"):
            predict_run(world.runs["global"], ["data.version=final1"], threads=2, tree=final_tree, allow_unseen=True)
        with pytest.raises(ValueError, match="changed since"):
            data.prepare()
    finally:
        paths.RANKING_FILE.write_bytes(original)
    data.prepare()   # unchanged again: the existing frame is kept


def test_blend_and_submit_take_the_predict_runs(world, final_tree):
    """The runbook's last steps: a weighted blend clipped to the NM join window, the stack's inputs, then the
    submission file."""
    from prc.blend import blend_runs
    from prc.postprocess import clip_to_bounds, lobt_bounds
    from prc.predict import predict_run
    from prc.submit import build_submission

    a = predict_run(world.runs["per_airport"], ["data.version=final1"], threads=2, tree=final_tree, allow_unseen=True)
    b = predict_run(world.runs["mixture"], ["data.version=final1"], threads=2, tree=final_tree, allow_unseen=True)
    blend = blend_runs([a.run_id, b.run_id], name="final_blend", lobt_bound=True, weights=[0.25, 0.75])
    mean = 0.25 * _preds(a.dir) + 0.75 * _preds(b.dir)
    lo, hi = lobt_bounds(data.load_movements(), mean.index)
    assert np.allclose(_preds(blend.dir).reindex(mean.index).to_numpy(), clip_to_bounds(mean, lo, hi), rtol=0, atol=1e-9)
    detail = pd.read_parquet(b.dir / "refit_detail.parquet")   # what lab/final_stack.py averages over the leaves
    assert detail.index.equals(_preds(b.dir).index) and {"p_stale", "y_reg"} <= set(detail.columns)
    sub = pd.read_parquet(build_submission(blend.run_id))
    template = pd.read_parquet(paths.SUBMITTING_FILE)
    assert len(sub) == len(template) and str(sub["TAXITIME_SEC_mvt"].dtype) == "int32"
    assert set(sub["MVT_ID_mvt"]) == set(template["MVT_ID_mvt"])


# --- refusals -------------------------------------------------------------------------------------------------------

def test_runs_without_a_refit_model_are_refused(world):
    from prc.predict import PredictRefused, predict_run

    no_refit = _copy_run(world, "global", "norefit")
    cfg = json.loads((no_refit / "config.resolved.json").read_text(encoding="utf-8"))
    cfg["model"]["refit_on_all"] = False
    (no_refit / "config.resolved.json").write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(PredictRefused, match="no refit model"):
        predict_run(no_refit.name)

    gone = _copy_run(world, "per_airport", "gone")
    for f in (gone / "model").glob("refit__*.cbm"):
        f.unlink()
    with pytest.raises(PredictRefused, match="no refit model"):
        predict_run(gone.name)

    other = _copy_run(world, "global", "hgb")
    cfg = json.loads((other / "config.resolved.json").read_text(encoding="utf-8"))
    cfg["model"]["type"] = "hgb"
    (other / "config.resolved.json").write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(PredictRefused, match="catboost"):
        predict_run(other.name)


def test_only_the_data_version_and_notes_may_be_overridden(world):
    from prc.predict import PredictRefused, load_source, predict_config, predict_run

    with pytest.raises(PredictRefused, match="data.version"):
        predict_run(world.runs["global"], ["model.params.depth=5"])
    src = load_source(world.runs["global"])
    with pytest.raises(PredictRefused, match="key=value"):
        predict_config(src, ["data.version"])
    with pytest.raises(PredictRefused, match="non-empty"):
        predict_config(src, ["data.version="])
    cfg = predict_config(src, ["data.version=2026", "notes=final set"], name="final_m01")
    assert cfg.data.version == "2026" and cfg.notes == "final set" and cfg.name == "final_m01"
    assert cfg.model == src.cfg.model and cfg.features == src.cfg.features and cfg.seed == src.seed


def test_feature_code_drift_is_refused_unless_allowed(world):
    from prc.predict import PredictRefused, predict_run

    drifted = _copy_run(world, "global", "drift")
    flist = json.loads((drifted / "feature_list.json").read_text(encoding="utf-8"))
    flist["hashes"]["stand_runway_keys"] = "0123456789ab"   # as if the block's code had changed since the run
    (drifted / "feature_list.json").write_text(json.dumps(flist), encoding="utf-8")
    with pytest.raises(PredictRefused, match="stand_runway_keys"):
        predict_run(drifted.name)
    run = predict_run(drifted.name, allow_drift=True, threads=2)
    record = _record(run)
    assert record["drift_allowed"] == ["stand_runway_keys"]
    assert record["blocks"]["stand_runway_keys"]["status"] == "drift"


def _external_probe(table: Path, version: int = 1):
    """A block that reads one input file (under the data tree), as the weather and geometry blocks do."""
    from prc.features.registry import FeatureSpec

    def probe_external(mvt, ctx):  # pragma: no cover - never computed, only hashed
        return pd.DataFrame(index=ctx.dep_index)

    return FeatureSpec(name="probe_external", fn=probe_external, group="probe", outputs=("probe_x",),
                       external=(table,), version=version)


def test_blocks_with_input_files_are_checked_against_the_source_trees_files(world, final_tree):
    """In a new tree a block that reads an input file hashes differently anyway (the tree rebuilt the file); with the
    file of the run's own tree it must hash as recorded, else its code changed: drift. Gone there: unverifiable."""
    from prc.features.build import block_hash
    from prc.predict import compare_blocks, load_source, predict_config

    rel = Path("additional_data") / "probe" / "table.parquet"
    for tree, content in ((world.tree, b"the source tree's table"), (final_tree, b"rebuilt here, with more rows")):
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_bytes(content)
    spec = _external_probe(final_tree / rel)
    src = load_source(world.runs["global"])
    cfg = predict_config(src, ["data.version=final1"])
    recorded = block_hash(spec, "syn1", {}, {}, external_root=world.tree)

    def status(spec_now, recorded_hash):
        fake = dataclasses.replace(src, feature_list={"hashes": {"probe_external": recorded_hash}})
        return compare_blocks([spec_now], fake, cfg, world.tree)["probe_external"]

    same_code = status(spec, recorded)
    assert same_code["status"] == "external" and same_code["hash_with_source_inputs"] == recorded
    assert same_code["hash_under_source_version"] != recorded   # this tree's file differs: not comparable directly
    assert status(_external_probe(final_tree / rel, version=2), recorded)["status"] == "drift"
    (world.tree / rel).unlink()
    gone = status(spec, recorded)
    assert gone["status"] == "external_unverified" and gone["source_inputs_missing"]


def test_features_only_warms_the_cache_and_writes_no_run(world):
    from prc.predict import predict_run

    before = sorted(p.name for p in paths.RUNS.iterdir())
    assert predict_run(world.runs["mixture"], ["data.version=syn_warm"], features_only=True, tree=paths.DATA) is None
    assert sorted(p.name for p in paths.RUNS.iterdir()) == before
    assert any((paths.FEATURE_CACHE / "syn_warm").rglob("stale_target_stats__*.parquet"))
    assert any((paths.FEATURE_CACHE / "syn_warm" / "_parity").glob("*.json"))   # measured for the real call


def test_cli_predict(world, capsys):
    from prc.cli import main

    assert main(["predict", "--run", world.runs["per_airport"], "--threads", "2", "--name", "cli_predict"]) == 0
    assert "artifacts:" in capsys.readouterr().out
    assert main(["predict", "--run", world.runs["per_airport"], "-o", "target.transform=log1p"]) == 2
    assert "refused" in capsys.readouterr().err
    assert main(["predict", "--run", world.runs["per_airport"], "-o", "data.version=cli_v2"]) == 2
    assert "--tree" in capsys.readouterr().err


def test_predict_runs_stay_off_the_default_leaderboard(world):
    from prc import leaderboard, tracking

    index = tracking.load_index()
    predicted = index.loc[index["predict_from"].notna()]
    assert len(predicted) and set(predicted["predict_from"]) <= set(world.runs.values()) | {
        f"{world.runs[k]}__{s}" for k in world.runs for s in ("legacy", "drift")}
    shown = leaderboard.runs(limit=1000)
    assert len(shown) and not set(shown["run_id"]) & set(predicted["run_id"])
    named = leaderboard.runs(limit=1000, name=str(predicted["name"].iloc[0]))   # asked for by name, they are listed
    assert set(named["run_id"]) & set(predicted["run_id"])
