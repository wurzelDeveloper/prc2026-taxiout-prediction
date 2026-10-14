"""Saved models back from disk: CatBoost global and per-airport regressors and the regime mixture, `save` then `load`
predicting exactly what the fitted object predicted; runs saved before the state files existed restored from their
fit rows; categorical levels the fit never saw. Tiny CPU fits on synthetic frames, no competition data."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostError

from prc.config import MixtureConfig, load_config
from prc.models import feature_contract, load_model, restore_fit_state, set_predict_threads, state_columns
from prc.models.base import NA_LEVEL, CategoryUniverse, prepare_matrix
from prc.models.gbdt import CatBoostModel
from prc.models.mixture import STATE_SUFFIX as MIXTURE_STATE
from prc.models.mixture import RegimeMixtureModel
from prc.models.per_airport import STATE_SUFFIX as PER_AIRPORT_STATE
from prc.models.per_airport import PerAirportModel

CB = {"iterations": 25, "depth": 3, "learning_rate": 0.3, "thread_count": 2, "verbose": 0}


def _frame(n: int = 600, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    apt = rng.choice(["EDDF", "LIRF", "LSZH"], n)
    stand = rng.choice([f"S{i}" for i in range(30)], n)
    small = rng.choice(["a", "b", "c"], n)
    x = rng.normal(size=n)
    suspect = rng.random(n) < 0.3
    carrier = rng.choice(["XX", "AZ", "LH"], n)
    delta = np.where(suspect, rng.uniform(9_000, 30_000, n), rng.uniform(-600, 1_500, n))
    lobt = np.where(rng.random(n) < 0.8, 700 + 80 * x, np.nan)
    taxi = 600 + 80 * x + np.where(small == "a", 120, 0) + rng.normal(0, 10, n)
    y = np.where(suspect & (carrier == "XX"), delta, taxi)
    X = pd.DataFrame(
        {
            "apt": pd.array(apt, dtype="string"),
            "stand": pd.Categorical(stand),
            "small": pd.array(small, dtype="string"),
            "carrier": pd.array(carrier, dtype="string"),
            "x": x.astype("float32"),
            "stale_suspect": suspect.astype("int8"),
            "sched_delta_full": delta.astype("float32"),
            "lobt_delta_full": lobt.astype("float32"),
        },
        index=pd.RangeIndex(n, name="mvt_id"),
    )
    return X, y


CATS = ["apt", "stand", "small", "carrier"]


def _mixture_cfg(**over) -> MixtureConfig:
    base = dict(
        enabled=True, bases=["sched_delta_full"], universe_column="stale_suspect", min_base_sec=None, tol_sec=60,
        regressor_max_target_sec=7_200, classifier_params=CB, regressor_anchor="lobt_delta_full",
        anchor_clip=[-200_000.0, 200_000.0], window_column="lobt_delta_full",
    )
    return MixtureConfig(**(base | over))


def _per_airport(universe):
    return PerAirportModel(lambda tag=None: CatBoostModel(universe, seed=1, **CB))


def test_catboost_round_trip_predicts_exactly_and_keeps_its_contract(tmp_path):
    X, y = _frame()
    universe = CategoryUniverse(X, CATS)
    model = CatBoostModel(universe, seed=1, **CB).fit(X, y, cat_features=CATS)
    model.save(tmp_path / "refit")
    loaded = CatBoostModel.load(tmp_path / "refit", universe)
    assert np.array_equal(loaded.predict(X), model.predict(X))
    names, cats = feature_contract(loaded)
    assert names == list(X.columns) and cats == CATS
    set_predict_threads(loaded, 1)
    assert loaded.predict_threads == 1 and np.array_equal(loaded.predict(X), model.predict(X))
    with pytest.raises(FileNotFoundError):
        CatBoostModel.load(tmp_path / "absent", universe)


def test_catboost_binds_features_by_position_so_the_order_is_the_contract(tmp_path):
    """Handed the right columns in another order, CatBoost either fails or mispredicts: the caller aligns to
    `feature_contract` (what `prc predict` does) and then gets the fitted model's answer back."""
    X, y = _frame()
    universe = CategoryUniverse(X, CATS)
    model = CatBoostModel(universe, seed=1, **CB).fit(X, y, cat_features=CATS)
    model.save(tmp_path / "m")
    loaded = CatBoostModel.load(tmp_path / "m", universe)
    shuffled = X[list(reversed(X.columns))]
    with pytest.raises(CatBoostError):
        loaded.predict(shuffled)  # a float where the model binds a categorical
    names, _ = feature_contract(loaded)
    assert np.array_equal(loaded.predict(shuffled[names]), model.predict(X))


def test_unseen_levels_reach_catboost_as_strings_and_are_all_treated_alike(tmp_path):
    """String encoding: a level the fit never saw is not an error and not ``__NA__``; every unseen level of a column
    gives the same prediction (prior-only CTR, no one-hot indicator), so a new 2026 stand is an "unknown stand"."""
    X, y = _frame()
    universe = CategoryUniverse(X, CATS)
    model = CatBoostModel(universe, seed=1, **CB).fit(X, y, cat_features=CATS)
    rows = X.iloc[:20].copy()
    one, two = rows.copy(), rows.copy()
    one["stand"] = pd.Categorical(["NEW_STAND_1"] * len(rows))
    two["stand"] = pd.Categorical(["NEW_STAND_2"] * len(rows))
    one["small"] = pd.array(["zz"] * len(rows), dtype="string")   # one-hot column (3 levels in the fit)
    two["small"] = pd.array(["yy"] * len(rows), dtype="string")
    p1, p2 = model.predict(one), model.predict(two)
    assert np.isfinite(p1).all() and np.array_equal(p1, p2)
    frame = prepare_matrix(one, universe, encoding="string")
    assert frame["stand"].tolist() == ["NEW_STAND_1"] * len(rows)  # passed through, not mapped to __NA__


def test_codes_encoding_sends_unseen_levels_to_the_explicit_na_level():
    X = pd.DataFrame({"stand": pd.Categorical(["A1", "B2", None]), "n": [1.0, 2.0, 3.0]})
    universe = CategoryUniverse(X, ["stand"])
    na_code = universe.levels["stand"].get_loc(NA_LEVEL)
    later = pd.DataFrame({"stand": pd.Categorical(["A1", "Z9", None]), "n": [1.0, 2.0, 3.0]})
    codes = prepare_matrix(later, universe, encoding="codes")["stand"].tolist()
    assert codes[0] == prepare_matrix(X, universe, encoding="codes")["stand"].iloc[0]
    assert codes[1] == na_code and codes[2] == na_code and min(codes) >= 0


def test_per_airport_round_trip_and_the_state_file(tmp_path):
    X, y = _frame()
    universe = CategoryUniverse(X, CATS)
    model = _per_airport(universe).fit(X, y, cat_features=CATS)
    model.save(tmp_path / "refit")
    assert (tmp_path / f"refit{PER_AIRPORT_STATE}").exists()
    loaded = PerAirportModel.load(tmp_path / "refit", lambda stem: CatBoostModel.load(stem, universe))
    assert loaded.state_source_ == "saved" and loaded.fallback == model.fallback and loaded.sizes == model.sizes
    assert np.array_equal(loaded.predict(X), model.predict(X))
    unknown = X.iloc[:3].copy()
    unknown["apt"] = pd.array(["ZZZZ"] * 3, dtype="string")
    assert np.array_equal(loaded.predict(unknown), np.full(3, model.fallback))  # an airport the fit never saw


def test_per_airport_saved_without_state_is_restored_from_its_fit_rows(tmp_path):
    X, y = _frame()
    universe = CategoryUniverse(X, CATS)
    model = _per_airport(universe).fit(X, y, cat_features=CATS)
    model.save(tmp_path / "refit")
    (tmp_path / f"refit{PER_AIRPORT_STATE}").unlink()
    loaded = PerAirportModel.load(tmp_path / "refit", lambda stem: CatBoostModel.load(stem, universe))
    assert sorted(loaded.models) == ["EDDF", "LIRF", "LSZH"] and loaded.fallback is None
    with pytest.raises(RuntimeError, match="restore_fit_state"):
        loaded.predict(X)
    report = loaded.restore_fit_state(X[["apt"]], y)
    assert report["recomputed"] == ["fallback", "sizes"] and loaded.fallback == model.fallback
    assert loaded.sizes == model.sizes and np.array_equal(loaded.predict(X), model.predict(X))


@pytest.mark.parametrize("per_airport", [True, False])
def test_mixture_round_trip_with_two_classifiers(tmp_path, per_airport):
    X, y = _frame()
    universe = CategoryUniverse(X, CATS)
    regressor = _per_airport(universe) if per_airport else CatBoostModel(universe, seed=1, **CB)
    model = RegimeMixtureModel(regressor, _mixture_cfg(classifier_seeds=2), universe, seed=1)
    model.fit(X, y, cat_features=CATS)
    model.save(tmp_path / "refit")
    cfg = load_config("base.yaml").replace(model={"per_airport": per_airport, "mixture": asdict(_mixture_cfg(classifier_seeds=2))})
    loaded = load_model(cfg, universe, tmp_path / "refit")
    assert isinstance(loaded, RegimeMixtureModel) and len(loaded.classifiers) == 2 and loaded.state_source_ == "saved"
    assert loaded.anchor_fallback_ == model.anchor_fallback_ and loaded.classes_ == model.classes_
    want, got = model.predict_detail(X), loaded.predict_detail(X)
    pd.testing.assert_frame_equal(got, want)
    names, cats = feature_contract(loaded)
    assert names == list(X.columns) and cats == CATS


def test_mixture_saved_without_state_replays_its_fit_rows(tmp_path):
    """A run from before ``__mixture.json``: the anchor's per-airport fallback medians come back from the fit rows."""
    X, y = _frame()
    universe = CategoryUniverse(X, CATS)
    model = RegimeMixtureModel(_per_airport(universe), _mixture_cfg(), universe, seed=1)
    model.fit(X, y, cat_features=CATS)
    model.save(tmp_path / "refit")
    for suffix in (MIXTURE_STATE, PER_AIRPORT_STATE):
        (tmp_path / f"refit{suffix}").unlink()
    cfg = load_config("base.yaml").replace(model={"per_airport": True, "mixture": asdict(_mixture_cfg())})
    loaded = load_model(cfg, universe, tmp_path / "refit")
    assert loaded.state_source_ == "missing" and loaded.anchor_fallback_ is None
    with pytest.raises(RuntimeError, match="restore_fit_state"):
        loaded.predict_detail(X)
    report = restore_fit_state(loaded, X[state_columns(loaded)], y)
    assert report["mixture"]["recomputed"] == ["anchor_fallback"]
    assert report["per_airport"]["recomputed"] == ["fallback", "sizes"]
    assert loaded.anchor_fallback_ == model.anchor_fallback_
    assert loaded.anchor_fallback_global_ == model.anchor_fallback_global_
    assert loaded.regressor.fallback == model.regressor.fallback
    pd.testing.assert_frame_equal(loaded.predict_detail(X), model.predict_detail(X))
    assert set(state_columns(loaded)) == {"sched_delta_full", "stale_suspect", "lobt_delta_full", "apt"}


def test_single_regime_mixture_has_no_classifier_and_restores_its_class(tmp_path):
    X, y = _frame()
    y = np.where(X["stale_suspect"].to_numpy() == 1, X["sched_delta_full"].to_numpy(dtype="float64"), y)  # all stale
    universe = CategoryUniverse(X, CATS)
    cfg_m = _mixture_cfg(regressor_anchor=None, window_column=None)
    model = RegimeMixtureModel(CatBoostModel(universe, seed=1, **CB), cfg_m, universe, seed=1)
    model.fit(X, y, cat_features=CATS)
    assert model.classifier is None and model.constant_class_ == 1
    model.save(tmp_path / "refit")
    cfg = load_config("base.yaml").replace(model={"mixture": asdict(cfg_m)})
    saved = load_model(cfg, universe, tmp_path / "refit")
    assert saved.classifier is None and saved.constant_class_ == 1
    pd.testing.assert_frame_equal(saved.predict_detail(X), model.predict_detail(X))
    (tmp_path / f"refit{MIXTURE_STATE}").unlink()
    legacy = load_model(cfg, universe, tmp_path / "refit")
    assert legacy.constant_class_ is None
    report = restore_fit_state(legacy, X[state_columns(legacy)], y)
    assert report["mixture"]["recomputed"] == ["anchor_fallback", "constant_class"] and legacy.constant_class_ == 1
    pd.testing.assert_frame_equal(legacy.predict_detail(X), model.predict_detail(X))
    with pytest.raises(ValueError, match="regimes"):
        # replayed on rows where the universe has two regimes, the missing classifier is an error, not a guess
        restore_fit_state(load_model(cfg, universe, tmp_path / "refit"), X[state_columns(legacy)], _frame()[1])


def test_feature_contract_rejects_members_fitted_on_other_columns(tmp_path):
    X, y = _frame()
    universe = CategoryUniverse(X, CATS)
    a = CatBoostModel(universe, seed=1, **CB).fit(X, y, cat_features=CATS)
    b = CatBoostModel(universe, seed=1, **CB).fit(X.drop(columns=["x"]), y, cat_features=CATS)
    wrapped = PerAirportModel(None)
    wrapped.models = {"EDDF": a, "LIRF": b}
    with pytest.raises(ValueError, match="other feature columns"):
        feature_contract(wrapped)


def test_only_catboost_runs_load():
    cfg = load_config("base.yaml", ["model.type=hgb"])
    with pytest.raises(NotImplementedError, match="catboost"):
        load_model(cfg, CategoryUniverse(pd.DataFrame({"a": [1.0]}), []), Path("nowhere"))


def test_mixture_with_the_day_offset_regime_and_isotonic_calibration_round_trips(tmp_path):
    """No current submission member uses either, but `prc predict` must load both: the calibration curves come back
    from ``__calibration.joblib``, the day-offset class from the classifier, and a run saved without the state file
    is restored from its fit rows."""
    X, y = _frame(n=900, seed=3)
    suspect = np.flatnonzero(X["stale_suspect"].to_numpy() == 1)
    wrong_date = suspect[(X["carrier"].to_numpy()[suspect] != "XX")][:40]
    y = y.copy()
    y[wrong_date] = 86_400 + 600 + 5 * np.arange(len(wrong_date))       # block times on the previous day's date
    universe = CategoryUniverse(X, CATS)
    cfg_m = _mixture_cfg(day_offset=True, calibration="isotonic", calibration_min_rows=20,
                         regressor_anchor=None, window_column=None)
    model = RegimeMixtureModel(_per_airport(universe), cfg_m, universe, seed=1)
    model.fit(X, y, cat_features=CATS)
    assert model.calibrators_ and 2 in model.classes_                   # class 2: a day plus a taxi
    model.save(tmp_path / "refit")
    assert (tmp_path / "refit__calibration.joblib").exists()
    cfg = load_config("base.yaml").replace(model={"per_airport": True, "mixture": asdict(cfg_m)})
    loaded = load_model(cfg, universe, tmp_path / "refit")
    assert len(loaded.calibrators_) == len(model.calibrators_) and loaded.classes_ == model.classes_
    want = model.predict_detail(X)
    pd.testing.assert_frame_equal(loaded.predict_detail(X), want)
    assert (want["p_stale"] != want["p_stale_raw"]).any()                # the calibration acts

    for suffix in (MIXTURE_STATE, PER_AIRPORT_STATE):
        (tmp_path / f"refit{suffix}").unlink()
    legacy = load_model(cfg, universe, tmp_path / "refit")
    restore_fit_state(legacy, X[state_columns(legacy)], y)
    pd.testing.assert_frame_equal(legacy.predict_detail(X), want)
