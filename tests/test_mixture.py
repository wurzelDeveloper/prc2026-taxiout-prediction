"""The regime mixture on synthetic frames: candidates, labels, arithmetic and the
model contracts. Stub models stand in for CatBoost except in one small CPU
smoke test. No competition data, no GPU."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prc.config import MixtureConfig, load_config
from prc.models import make_model
from prc.models.base import CategoryUniverse
from prc.models.mixture import SEED_STRIDE, TAXI, RegimeMixtureModel, hedging_report
from prc.models.per_airport import PerAirportModel


def _frame() -> tuple[pd.DataFrame, np.ndarray]:
    rows = [
        # id, apt, suspect, sched delta, LOBT delta, regressor's answer, target
        (1, "LIRF", 1, 900.0, np.nan, 800.0, 1_000.0),        # suspect, small delta: within 600 s of it
        (2, "LIRF", 1, 20_000.0, np.nan, 900.0, 20_005.0),    # suspect, stale
        (3, "LIRF", 1, 20_000.0, np.nan, 900.0, 900.0),       # suspect, genuine
        (4, "LIRF", 1, -3_600.0, np.nan, 800.0, 800.0),       # suspect, negative delta: no candidate
        (5, "EDDF", 0, 30_000.0, np.nan, 1_000.0, 30_100.0),  # typed, large delta: in only with min_base_sec
        (6, "EGLL", 0, 500.0, 15_000.0, 1_200.0, 15_100.0),   # typed, the LOBT delta is the only candidate
        (7, "EDDF", 0, 600.0, 700.0, 700.0, 700.0),           # an ordinary row
        (8, "LIRF", 1, np.nan, np.nan, 800.0, 800.0),         # suspect without a schedule
        (9, "EGLL", 0, 20_000.0, 20_500.0, 1_100.0, 20_450.0),  # both candidates; LOBT is nearer
    ]
    X = pd.DataFrame(
        rows, columns=["mvt_id", "apt", "stale_suspect", "sched_delta_full", "lobt_delta_full", "reg", "y"]
    ).set_index("mvt_id")
    y = X.pop("y").to_numpy(dtype="float64")
    X["apt"] = X["apt"].astype("string")
    # The stub classifier reads its probabilities per class back from these columns.
    X["p0"], X["p1"], X["p2"] = 0.2, 0.5, 0.3
    return X, y


def _cfg(**over) -> MixtureConfig:
    base = dict(enabled=True, bases=["sched_delta_full"], universe_column="stale_suspect", min_base_sec=None)
    return MixtureConfig(**(base | over))


TWO = dict(bases=["sched_delta_full", "lobt_delta_full"], min_base_sec=7_200.0)


class _StubRegressor:
    def __init__(self, iterations=None):
        self.iterations = iterations
        self.fitted_index = None
        self.eval_index = None
        self.saved = None

    def fit(self, X, y, *, cat_features, eval_set=None):
        self.fitted_index = X.index
        self.eval_index = None if eval_set is None else eval_set[0].index
        return self

    def predict(self, X):
        return X["reg"].to_numpy(dtype="float64")

    def refit_iterations(self):
        return self.iterations

    def native_importance(self):
        return pd.DataFrame({"feature": ["reg"], "gain": [1.0]})

    def save(self, path):
        self.saved = path


class _StubClassifier:
    """Probabilities come from the ``p<k>`` columns of the prepared frame."""

    def __init__(self, params, classes=(0, 1, 2)):
        self.params = params
        self.classes_ = list(classes)
        self.best_iteration_ = 9
        self.n_fit = self.n_eval = self.n_cols = None

    def fit(self, pool, eval_set=None, use_best_model=False):
        self.n_fit = pool.num_row()
        self.n_cols = pool.num_col()
        self.n_eval = None if eval_set is None else eval_set.num_row()
        self.weights = np.asarray(pool.get_weight(), dtype="float64")
        self.eval_weights = None if eval_set is None else np.asarray(eval_set.get_weight(), dtype="float64")
        return self

    def predict_proba(self, frame):
        return frame[[f"p{c}" for c in self.classes_]].to_numpy(dtype="float64")

    def get_feature_importance(self):
        return np.ones(self.n_cols)


def _model(cfg, regressor=None, classes=(0, 1, 2), X=None):
    X = _frame()[0] if X is None else X
    holder = {}

    def factory(params):
        holder["clf"] = _StubClassifier(params, classes)
        return holder["clf"]

    model = RegimeMixtureModel(
        regressor or _StubRegressor(), cfg, CategoryUniverse(X, ["apt"]), seed=1, classifier_factory=factory
    )
    return model, holder


def test_candidates_and_universe():
    X, _ = _frame()
    B, valid, universe = _model(_cfg())[0].candidates(X)
    assert B.shape == (9, 1)
    assert universe.tolist() == [True, True, True, False, False, False, False, False, False]
    # a floor under the base drops the small delta
    assert _model(_cfg(base_min_sec=1_200.0))[0].candidates(X)[2].tolist()[:3] == [False, True, True]
    B, valid, universe = _model(_cfg(**TWO))[0].candidates(X)
    assert universe.tolist() == [True, True, True, False, True, True, False, False, True]
    assert valid[0].tolist() == [True, False]   # marked row: the 900 s delta counts, there is no LOBT
    assert valid[5].tolist() == [False, True]   # unmarked: 500 s is below the floor, LOBT 15,000 is not
    assert valid[8].tolist() == [True, True]
    only_large = _model(_cfg(universe_column=None, min_base_sec=7_200.0))[0].candidates(X)[2]
    assert only_large.tolist() == [False, True, True, False, True, False, False, False, True]
    with pytest.raises(KeyError, match="not features"):
        _model(_cfg(bases=["nope"]))[0].candidates(X)


def test_labels_take_the_nearest_candidate_within_tolerance():
    X, y = _frame()
    assert _model(_cfg())[0].labels(X, y).tolist() == [1, 1, 0, -1, -1, -1, -1, -1, -1]
    assert _model(_cfg(**TWO))[0].labels(X, y).tolist() == [1, 1, 0, -1, 1, 2, -1, -1, 2]
    # a relative tolerance: 900 s off a 20,000 s delta is stale at 5 %, not at a flat 600 s
    y2 = y.copy()
    y2[1] = 20_900.0
    assert _model(_cfg())[0].labels(X, y2)[1] == TAXI
    assert _model(_cfg(tol_frac=0.05))[0].labels(X, y2)[1] == 1
    assert _model(_cfg())[0].labels(X.iloc[:0], y[:0]).tolist() == []


def test_regressor_sees_no_stale_rows_and_the_eval_set_is_filtered_alike():
    X, y = _frame()
    reg = _StubRegressor()
    model, holder = _model(_cfg(), regressor=reg)
    model.fit(X, y, cat_features=["apt"], eval_set=(X, y))
    assert reg.fitted_index.tolist() == [3, 4, 5, 6, 7, 8, 9]
    assert reg.eval_index.tolist() == [3, 4, 5, 6, 7, 8, 9]
    assert holder["clf"].n_fit == 3 and holder["clf"].n_eval == 3

    reg2 = _StubRegressor()
    model2, _ = _model(_cfg(exclude_stale_from_regressor=False), regressor=reg2)
    model2.fit(X, y, cat_features=["apt"], eval_set=(X, y))
    assert reg2.fitted_index.tolist() == list(range(1, 10)) and reg2.eval_index.tolist() == list(range(1, 10))

    model3, holder3 = _model(_cfg(classifier_eval=False))
    model3.fit(X, y, cat_features=["apt"], eval_set=(X, y))
    assert holder3["clf"].n_eval is None and "od_type" not in holder3["clf"].params

    # an upper target bound removes unexplained multi-hour rows wherever they sit (row 5 is typed, outside)
    reg4 = _StubRegressor()
    model4, holder4 = _model(_cfg(regressor_max_target_sec=25_000.0), regressor=reg4)
    model4.fit(X, y, cat_features=["apt"], eval_set=(X, y))
    assert reg4.fitted_index.tolist() == [3, 4, 6, 7, 8, 9] and reg4.eval_index.tolist() == [3, 4, 6, 7, 8, 9]
    assert holder4["clf"].n_fit == 3   # the classifier's universe is unaffected
    with pytest.raises(ValueError, match="positive"):
        _cfg(regressor_max_target_sec=0)


def test_classifier_params():
    X, y = _frame()
    model, holder = _model(_cfg())
    model.fit(X, y, cat_features=["apt"], eval_set=(X, y))
    p = holder["clf"].params
    assert p["loss_function"] == "Logloss" and p["eval_metric"] == "Logloss"
    assert p["random_seed"] == 1 and p["od_type"] == "Iter" and p["allow_writing_files"] is False

    model, holder = _model(_cfg(**TWO, classifier_params={"depth": 6}))
    model.fit(X, y, cat_features=["apt"])
    p = holder["clf"].params
    assert p["loss_function"] == "MultiClass" and p["depth"] == 6 and "od_type" not in p

    with pytest.raises(ValueError, match="calibrated"):
        _model(_cfg(classifier_params={"auto_class_weights": "Balanced"}))[0].fit(X, y, cat_features=["apt"])

    # a fixed budget (the refit's) overrides the config's count
    fixed = RegimeMixtureModel(
        _StubRegressor(), _cfg(), CategoryUniverse(X, ["apt"]), seed=1,
        classifier_iterations=77, classifier_factory=lambda p: _StubClassifier(p),
    )
    fixed.fit(X, y, cat_features=["apt"])
    assert fixed.classifier.params["iterations"] == 77


def test_predict_masks_invalid_classes_and_renormalises():
    X, y = _frame()
    model, _ = _model(_cfg(**TWO))
    model.fit(X, y, cat_features=["apt"])
    detail = model.predict_detail(X)
    # row 5: schedule valid, no LOBT -> [0.2, 0.5, 0] / 0.7
    assert detail.loc[5, "y_pred"] == pytest.approx((0.5 * 30_000 + 0.2 * 1_000) / 0.7, rel=1e-6)
    assert detail.loc[5, "p_stale"] == pytest.approx(0.5 / 0.7, rel=1e-6)
    # row 6: only the LOBT candidate -> [0.2, 0, 0.3] / 0.5
    assert detail.loc[6, "y_pred"] == pytest.approx(0.6 * 15_000 + 0.4 * 1_200, rel=1e-6)
    # row 9: both valid -> the probabilities as delivered
    assert detail.loc[9, "y_pred"] == pytest.approx(0.5 * 20_000 + 0.3 * 20_500 + 0.2 * 1_100, rel=1e-6)
    assert detail.loc[9, "p_stale"] == pytest.approx(0.8, rel=1e-6)
    # outside the universe: the regressor's answer, no stale mass
    assert detail.loc[7, "y_pred"] == 700 and detail.loc[7, "p_stale"] == 0 and not detail.loc[7, "in_universe"]
    assert detail.loc[4, "y_pred"] == 800 and not detail.loc[4, "in_universe"]
    assert detail.loc[5, "y_reg"] == 1_000
    assert np.array_equal(model.predict(X), detail["y_pred"].to_numpy())


def test_no_valid_probability_mass_falls_back_to_the_regressor():
    X, y = _frame()
    X = X.assign(p0=0.0, p1=0.0, p2=1.0)
    model, _ = _model(_cfg(**TWO), X=X)
    model.fit(X, y, cat_features=["apt"])
    detail = model.predict_detail(X)
    assert detail.loc[5, "y_pred"] == 1_000 and detail.loc[5, "p_stale"] == 0   # all mass on the invalid LOBT
    assert detail.loc[6, "y_pred"] == 15_000 and detail.loc[6, "p_stale"] == 1


def test_absent_classes_and_binary_probabilities_map_onto_the_class_axis():
    X, y = _frame()
    model, _ = _model(_cfg(**TWO), classes=(0, 1))   # the fit never saw a LOBT row
    model.fit(X, y, cat_features=["apt"])
    detail = model.predict_detail(X)
    assert detail.loc[6, "y_pred"] == 1_200 and detail.loc[6, "p_stale"] == 0
    assert detail.loc[9, "y_pred"] == pytest.approx((0.5 * 20_000 + 0.2 * 1_100) / 0.7, rel=1e-6)

    class _Binary(_StubClassifier):
        def predict_proba(self, frame):
            return frame["p1"].to_numpy(dtype="float64")   # one column, the positive class

    X2 = X.assign(p1=0.25)
    binary = RegimeMixtureModel(
        _StubRegressor(), _cfg(), CategoryUniverse(X2, ["apt"]), seed=1,
        classifier_factory=lambda p: _Binary(p, classes=(0, 1)),
    )
    binary.fit(X2, y, cat_features=["apt"])
    d = binary.predict_detail(X2)
    assert d.loc[2, "y_pred"] == pytest.approx(0.25 * 20_000 + 0.75 * 900, rel=1e-6)
    assert d.loc[2, "p_stale"] == pytest.approx(0.25, rel=1e-6)


def test_day_offset_regime_labels_and_mixes_a_day_plus_the_taxi():
    from prc.models.mixture import DAY

    X, y = _frame()
    y = y.copy()
    y[2] = DAY + 800.0      # row 3: a wrong-date block, an hour-ish taxi on the previous day's date
    y[0] = DAY + 900.0      # row 1: also a day out, but base 900 is not within 600 of it either -> day
    cfg = _cfg(day_offset=True)
    model, holder = _model(cfg)
    assert model.day_class == 2 and model.n_classes == 3
    assert model.labels(X, y).tolist() == [2, 1, 2, -1, -1, -1, -1, -1, -1]
    # an exact candidate wins over the day rule: a stale block whose schedule is a day out
    X2 = X.copy()
    X2.loc[3, "sched_delta_full"] = DAY + 800.0
    assert model.labels(X2, y)[2] == 1
    # no wrong-date rows outside the universe, and a two-day offset counts too
    y3 = y.copy()
    y3[6] = DAY + 700.0
    y3[1] = 2 * DAY + 300.0
    assert model.labels(X, y3).tolist()[1] == 2 and model.labels(X, y3).tolist()[6] == -1

    model.fit(X, y, cat_features=["apt"], eval_set=(X, y))
    assert holder["clf"].params["loss_function"] == "MultiClass"
    detail = model.predict_detail(X)
    # row 2: [0.2, 0.5, 0.3] over taxi / schedule / day
    assert detail.loc[2, "y_pred"] == pytest.approx(0.5 * 20_000 + 0.3 * (DAY + 900) + 0.2 * 900, rel=1e-6)
    assert detail.loc[2, "p_stale"] == pytest.approx(0.8, rel=1e-6)
    assert detail.loc[7, "y_pred"] == 700   # outside the universe: no day mass
    info = model.diagnostics_
    assert info["labels_val"] == {"1": 1, "2": 2}
    assert info["rmse_universe_oracle"] == pytest.approx(np.sqrt((100.0**2 + 5.0**2 + 100.0**2) / 3))


def test_classifier_row_weights_follow_the_largest_candidate():
    from prc.models.mixture import DAY

    X, y = _frame()
    model, holder = _model(_cfg())
    model.fit(X, y, cat_features=["apt"], eval_set=(X, y))
    assert np.allclose(holder["clf"].weights, 1.0)            # unweighted by default
    assert model.row_weights(X) is None

    model, holder = _model(_cfg(**TWO, classifier_weight_power=1.0))
    model.fit(X, y, cat_features=["apt"], eval_set=(X, y))
    # universe rows 1, 2, 3, 5, 6, 9 with largest candidates 900, 20000, 20000, 30000, 15000, 20500
    ref = np.array([900.0, 20_000.0, 20_000.0, 30_000.0, 15_000.0, 20_500.0])
    assert np.allclose(holder["clf"].weights, ref / ref.mean())
    assert np.allclose(holder["clf"].eval_weights, ref / ref.mean())
    assert np.allclose(holder["clf"].weights.mean(), 1.0)

    model, holder = _model(_cfg(day_offset=True, classifier_weight_power=2.0))
    model.fit(X, y, cat_features=["apt"])
    ref = np.array([900.0, 20_000.0, 20_000.0]) ** 2          # the day regime does not enter the weight
    assert np.allclose(holder["clf"].weights, ref / ref.mean())
    assert model.row_weights(X)[0] == 900.0**2 and model.row_weights(X)[1] == 20_000.0**2 != DAY**2
    with pytest.raises(ValueError, match="negative"):
        _cfg(classifier_weight_power=-1)


def test_single_regime_universe_needs_no_classifier():
    X, y = _frame()
    y1 = y.copy()
    y1[2] = 20_000.0   # row 3 turns stale: every universe row is now stale
    model, holder = _model(_cfg())
    model.fit(X, y1, cat_features=["apt"])
    assert "clf" not in holder and model.classifier is None and model.classes_ == [1]
    d = model.predict_detail(X)
    assert d.loc[2, "y_pred"] == 20_000 and d.loc[2, "p_stale"] == 1 and d.loc[7, "y_pred"] == 700
    assert model.classifier_refit_iterations() is None and model.classifier_importance().empty


def test_contracts_diagnostics_and_save(tmp_path):
    X, y = _frame()
    model, _ = _model(_cfg(), regressor=_StubRegressor(iterations={"LIRF": 300}))
    model.fit(X, y, cat_features=["apt"], eval_set=(X, y))
    assert model.refit_iterations() == {"LIRF": 300}
    assert model.classifier_refit_iterations() == 10
    info = model.diagnostics_
    assert info["n_universe_train"] == 3 and info["n_universe_val"] == 3
    assert info["labels_val"] == {"0": 1, "1": 2} and info["classifier_best_iteration"] == 10
    assert info["rmse_universe_oracle"] < info["rmse_universe_mixture"] < info["rmse_universe_regressor"]
    assert set(info["p_stale_mean_by_label"]) == {"0", "1"} and "auc" in info and "logloss" in info
    imp = model.classifier_importance()
    assert list(imp.columns) == ["feature", "gain"] and len(imp) == X.shape[1]
    assert model.native_importance()["feature"].tolist() == ["reg"]
    model.save(tmp_path / "model" / "seasonal")
    assert model.regressor.saved == tmp_path / "model" / "seasonal"


MIX = ["model.mixture.enabled=true", "model.mixture.bases=[sched_delta_full]", "model.mixture.universe_column=stale_suspect"]


def test_config_rejects_conflicts_and_keeps_the_split_hash():
    base = load_config("base.yaml")
    cfg = load_config("base.yaml", MIX)
    assert cfg.model.mixture.enabled and cfg.model.mixture.bases == ["sched_delta_full"]
    assert cfg.split_hash() == base.split_hash() and cfg.hash() != base.hash()
    with pytest.raises(ValueError, match="exclusive"):
        load_config("base.yaml", MIX + ["target.residual_of=stale_base"])
    with pytest.raises(ValueError, match="transform"):
        load_config("base.yaml", MIX + ["target.transform=log1p"])
    with pytest.raises(ValueError, match="unknown key"):
        load_config("base.yaml", ["model.mixture.tolerance=5"])
    with pytest.raises(ValueError, match="universe"):
        MixtureConfig(enabled=True, bases=["a"], universe_column=None, min_base_sec=None)
    with pytest.raises(ValueError, match="distinct"):
        MixtureConfig(enabled=True, bases=["a", "a"], universe_column="u")
    assert not MixtureConfig().enabled   # the default is inert, whatever the other fields say

    exp = load_config("exp023_raw_per_airport_mixture.yaml")
    assert exp.model.mixture.enabled and exp.target.residual_of is None and exp.model.per_airport
    assert exp.model.mixture.min_base_sec is None and exp.model.mixture.universe_column == "stale_suspect"
    final = load_config("exp024_final_mixture.yaml")
    assert final.model.refit_on_all and final.model.mixture.enabled


def test_make_model_wraps_the_per_airport_regressor():
    cfg = load_config("base.yaml", MIX + ["model.per_airport=true"])
    universe = CategoryUniverse(pd.DataFrame({"apt": pd.array(["LIRF"], dtype="string")}), ["apt"])
    model = make_model(cfg, universe, 42, classifier_iterations=5)
    assert isinstance(model, RegimeMixtureModel) and isinstance(model.regressor, PerAirportModel)
    assert model.classifier_iterations == 5
    assert not isinstance(make_model(load_config("base.yaml"), universe, 42), RegimeMixtureModel)


def _calibration_frame() -> tuple[pd.DataFrame, np.ndarray]:
    """Twelve universe rows in two delta bands: half of the 8,000 s rows are stale, five of six 20,000 s rows."""
    rows = []
    for i in range(6):
        rows.append((100 + i, "LIRF", 1, 8_000.0, 900.0, 8_000.0 if i < 3 else 900.0))
    for i in range(6):
        rows.append((200 + i, "LIRF", 1, 20_000.0, 900.0, 20_000.0 if i < 5 else 900.0))
    rows.append((300, "EDDF", 0, 600.0, 700.0, 700.0))  # outside the universe
    X = pd.DataFrame(rows, columns=["mvt_id", "apt", "stale_suspect", "sched_delta_full", "reg", "y"]).set_index("mvt_id")
    y = X.pop("y").to_numpy(dtype="float64")
    X["apt"] = X["apt"].astype("string")
    X["p0"], X["p1"] = 0.2, 0.8  # the stub classifier is 80 % sure of "stale" on every row
    return X, y


def test_isotonic_calibration_maps_the_probability_to_the_band_rate(tmp_path):
    X, y = _calibration_frame()
    cfg = _cfg(calibration="isotonic", calibration_delta_bands=[10_000.0], calibration_folds=2, calibration_min_rows=1)
    model, _ = _model(cfg, classes=(0, 1), X=X)
    model.fit(X, y, cat_features=["apt"], eval_set=(X, y))
    detail = model.predict_detail(X)
    low, high = detail.loc[100:105], detail.loc[200:205]
    assert np.allclose(low["p_stale_raw"], 0.8) and np.allclose(high["p_stale_raw"], 0.8)
    assert np.allclose(low["p_stale"], 0.5, atol=1e-6) and np.allclose(high["p_stale"], 5 / 6, atol=1e-6)
    assert detail.loc[100, "y_pred"] == pytest.approx(0.5 * 8_000 + 0.5 * 900)
    assert detail.loc[300, "p_stale"] == 0 and detail.loc[300, "p_stale_raw"] == 0 and detail.loc[300, "y_pred"] == 700
    info = model.diagnostics_
    assert info["rmse_universe_mixture"] < info["rmse_universe_mixture_uncalibrated"]
    assert [c["bands"] for c in info["calibration"]] == [[0], [1]]
    assert info["calibration"][0]["n"] == 6 and info["calibration"][0]["stale_share"] == 0.5
    assert info["calibration"][1]["logloss_calibrated"] < info["calibration"][1]["logloss_raw"]
    model.save(tmp_path / "model" / "fold")
    assert (tmp_path / "model" / "fold__calibration.joblib").exists()


def test_thin_bands_pool_and_calibration_none_keeps_the_raw_probability():
    X, y = _calibration_frame()
    pooled, _ = _model(
        _cfg(calibration="isotonic", calibration_delta_bands=[10_000.0], calibration_folds=2, calibration_min_rows=100),
        classes=(0, 1), X=X,
    )
    pooled.fit(X, y, cat_features=["apt"])  # the refit path: no eval set, still calibrated
    detail = pooled.predict_detail(X)
    assert np.allclose(detail.loc[100:205, "p_stale"], 8 / 12, atol=1e-6)
    assert [c["bands"] for c in pooled.calibration_info_] == [[0, 1]]

    raw, _ = _model(_cfg(), classes=(0, 1), X=X)
    raw.fit(X, y, cat_features=["apt"])
    detail = raw.predict_detail(X)
    assert (detail["p_stale"] == detail["p_stale_raw"]).all() and not raw.calibrators_

    with pytest.raises(ValueError, match="calibration"):
        MixtureConfig(enabled=True, bases=["a"], universe_column="u", calibration="platt")
    with pytest.raises(ValueError, match="calibration_folds"):
        MixtureConfig(enabled=True, bases=["a"], universe_column="u", calibration="isotonic", calibration_folds=1)
    with pytest.raises(ValueError, match="calibration_delta_bands"):
        MixtureConfig(enabled=True, bases=["a"], universe_column="u", calibration_delta_bands=[7200.0, 3600.0])
    cal = load_config("exp029_calibrated.yaml")
    base = load_config("exp028_final_stale_only.yaml")
    assert cal.model.mixture.calibration == "isotonic" and not cal.model.refit_on_all
    assert cal.split_hash() == base.split_hash() and cal.hash() != base.hash()  # same folds and feature caches


def test_catboost_end_to_end_on_cpu(tmp_path):
    from prc.models.gbdt import CatBoostModel

    rng = np.random.default_rng(0)
    n = 400
    carrier = rng.choice(["ISR", "RYR", "EJU"], size=n)
    suspect = rng.random(n) < 0.5
    delta = np.where(suspect, rng.uniform(3_000, 30_000, n), rng.uniform(-600, 1_800, n))
    x1 = rng.normal(size=n)
    taxi = 600 + 100 * x1 + rng.normal(scale=30, size=n)
    stale = suspect & (carrier == "ISR")
    y = np.where(stale, delta + rng.normal(scale=20, size=n), taxi)
    X = pd.DataFrame(
        {
            "apt": pd.array(["LIRF"] * n, dtype="string"),
            "carrier": pd.array(carrier, dtype="string"),
            "x1": x1.astype("float32"),
            "stale_suspect": suspect.astype("int8"),
            "sched_delta_full": delta.astype("float32"),
        },
        index=pd.RangeIndex(n, name="mvt_id"),
    )
    universe = CategoryUniverse(X, ["apt", "carrier"])
    tr, va = np.arange(n) < 300, np.arange(n) >= 300
    reg = CatBoostModel(universe, seed=0, iterations=30, verbose=0)
    model = RegimeMixtureModel(
        reg, _cfg(classifier_params={"iterations": 30, "verbose": 0}), universe, seed=0, train_dir=tmp_path / "clf"
    )
    model.fit(X.loc[tr], y[tr], cat_features=["apt", "carrier"], eval_set=(X.loc[va], y[va]))

    detail = model.predict_detail(X.loc[va])
    assert np.isfinite(detail["y_pred"]).all() and detail["p_stale"].between(0, 1).all()
    ps = detail["p_stale"].to_numpy()
    assert ps[(carrier[va] == "ISR") & suspect[va]].mean() > ps[(carrier[va] != "ISR") & suspect[va]].mean()
    assert isinstance(model.refit_iterations(), int) and isinstance(model.classifier_refit_iterations(), int)
    assert model.diagnostics_["auc"] > 0.9
    assert model.diagnostics_["rmse_universe_mixture"] < model.diagnostics_["rmse_universe_regressor"]
    model.save(tmp_path / "model" / "seasonal")
    assert (tmp_path / "model" / "seasonal.cbm").exists()
    assert (tmp_path / "model" / "seasonal__classifier.cbm").exists()


def test_hedge_variance_is_the_regime_variance_around_the_prediction():
    X, y = _frame()
    model, _ = _model(_cfg(**TWO))
    model.fit(X, y, cat_features=["apt"])
    detail = model.predict_detail(X)
    # row 9: all three regimes valid, probabilities as delivered
    P, V = np.array([0.2, 0.5, 0.3]), np.array([1_100.0, 20_000.0, 20_500.0])
    mean = (P * V).sum()
    assert detail.loc[9, "y_pred"] == pytest.approx(mean)
    assert detail.loc[9, "hedge_var"] == pytest.approx((P * (V - mean) ** 2).sum())
    # row 5: one valid base, so the variance is p(1 - p)(base - taxi)^2
    p = 0.5 / 0.7
    assert detail.loc[5, "hedge_var"] == pytest.approx(p * (1 - p) * (30_000.0 - 1_000.0) ** 2)
    # outside the universe the regressor answers alone: nothing hedged
    assert detail.loc[7, "hedge_var"] == 0 and detail.loc[4, "hedge_var"] == 0


class _SeedStub(_StubClassifier):
    """The first seed reads the ``p<k>`` columns; every other seed says "taxi" with certainty."""

    def __init__(self, params):
        super().__init__(params)
        self.best_iteration_ = 9 + (params["random_seed"] - 1) // 100

    def predict_proba(self, frame):
        if self.params["random_seed"] == 1:
            return super().predict_proba(frame)
        out = np.zeros((len(frame), len(self.classes_)))
        out[:, 0] = 1.0
        return out


def test_classifier_seeds_average_the_probabilities(tmp_path):
    X, y = _frame()
    made = []

    def factory(params):
        made.append(_SeedStub(params))
        return made[-1]

    model = RegimeMixtureModel(
        _StubRegressor(), _cfg(**TWO, classifier_seeds=2), CategoryUniverse(X, ["apt"]), seed=1, classifier_factory=factory
    )
    model.fit(X, y, cat_features=["apt"], eval_set=(X, y))
    assert [c.params["random_seed"] for c in made] == [1, 1 + SEED_STRIDE]
    assert model.classifier is made[0] and model.classifiers == made
    # row 9: (0.8 stale + 0 stale) / 2
    assert model.predict_detail(X).loc[9, "p_stale"] == pytest.approx(0.4)
    # best iterations 10 and 20: the refit gets their mean
    assert model.classifier_refit_iterations() == 15
    model.save(tmp_path / "m" / "seasonal")
    one = _model(_cfg(**TWO))[0]
    one.fit(X, y, cat_features=["apt"])
    assert len(one.classifiers) == 1 and one.predict_detail(X).loc[9, "p_stale"] == pytest.approx(0.8)
    with pytest.raises(ValueError, match="positive integer"):
        _cfg(classifier_seeds=0)


def test_hedging_report_reads_the_cost_and_the_calibration():
    detail = pd.DataFrame({
        "p_stale": [0.01, 0.01, 0.3, 0.3, 0.9, 0.0],
        "in_universe": [True, True, True, True, True, False],
        "hedge_var": [10.0, 10.0, 100.0, 100.0, 40.0, 0.0],
    })
    regime = np.array([0, 0, 1, 0, 1, -1])
    rep = hedging_report(detail, regime)
    assert rep["hedge"] == pytest.approx(260.0 / 6) and rep["n_universe"] == 5
    bands = {b["band"]: b for b in rep["calibration"]}
    assert bands["0.00-0.05"]["n"] == 2 and bands["0.00-0.05"]["stale_rate"] == 0.0
    assert bands["0.20-0.50"]["stale_rate"] == pytest.approx(0.5) and bands["0.80-0.95"]["mean_p"] == pytest.approx(0.9)
    # |0.01 - 0| twice, |0.3 - 0.5| twice, |0.9 - 1| once, over five universe rows
    assert rep["ece"] == pytest.approx((2 * 0.01 + 2 * 0.2 + 0.1) / 5)
    assert rep["brier"] == pytest.approx((2 * 0.01**2 + 0.7**2 + 0.3**2 + 0.1**2) / 5)
    assert set(hedging_report(detail)) == {"hedge", "n_universe"}   # without labels: the cost only


def test_classifier_seeds_leave_the_config_hash_alone_while_one():
    base = load_config("exp043_final_screen.yaml")
    assert base.model.mixture.classifier_seeds == 1 and base.hash() == "f76b35c9"
    assert base.replace(model={"mixture": {"classifier_seeds": 3}}).hash() != base.hash()



class _WeightedStubRegressor(_StubRegressor):
    def fit(self, X, y, *, cat_features, eval_set=None, sample_weight=None):
        self.sample_weight = None if sample_weight is None else np.asarray(sample_weight, dtype="float64")
        return super().fit(X, y, cat_features=cat_features, eval_set=eval_set)


def test_sample_weights_reach_the_regressor_rows_and_multiply_the_classifier_weights():
    X, y = _frame()
    reg = _WeightedStubRegressor()
    model, holder = _model(_cfg(classifier_weight_power=1.0), regressor=reg)
    w = np.arange(1.0, 10.0)                       # one weight per row, ids 1..9
    model.fit(X, y, cat_features=["apt"], sample_weight=w)
    # the regressor fits rows 3..9 (rows 1 and 2 are stale-labelled), with their weights
    assert reg.fitted_index.tolist() == [3, 4, 5, 6, 7, 8, 9]
    assert reg.sample_weight.tolist() == [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    # the classifier's universe (rows 1-3): delta ** 1 times the row weight, normalised to a mean of one
    raw = np.array([900.0 * 1, 20_000.0 * 2, 20_000.0 * 3])
    assert holder["clf"].weights == pytest.approx(raw / raw.mean())


def test_per_airport_model_slices_the_weights():
    from prc.models.per_airport import PerAirportModel

    X, y = _frame()
    seen = {}

    class _Rec:
        def fit(self, Xa, ya, *, cat_features, eval_set=None, sample_weight=None):
            seen[Xa["apt"].iloc[0]] = None if sample_weight is None else list(sample_weight)
            return self

    model = PerAirportModel(lambda tag: _Rec(), group_column="apt")
    model.fit(X, y, cat_features=["apt"], sample_weight=np.arange(1.0, 10.0))
    assert seen == {"EDDF": [5.0, 7.0], "EGLL": [6.0, 9.0], "LIRF": [1.0, 2.0, 3.0, 4.0, 8.0]}
    model.fit(X, y, cat_features=["apt"])
    assert set(v for v in seen.values()) == {None}


class _RecordingRegressor(_StubRegressor):
    """Records the fit targets; predicts a constant correction of 10 s."""

    def fit(self, X, y, *, cat_features, eval_set=None):
        self.fit_y = np.asarray(y, dtype="float64")
        self.eval_y = None if eval_set is None else np.asarray(eval_set[1], dtype="float64")
        return super().fit(X, y, cat_features=cat_features, eval_set=eval_set)

    def predict(self, X):
        return np.full(len(X), 10.0)


def test_anchored_regressor_fits_the_residual_and_adds_the_anchor_back():
    X, y = _frame()
    # the anchor column: a proxy per row, missing on rows 7 and 8, absurd on row 9 (clipped to 7,200)
    X["aobt3_proxy"] = [700.0, 800.0, 850.0, 750.0, 950.0, 1_100.0, np.nan, np.nan, 90_000.0]
    reg = _RecordingRegressor()
    model, _ = _model(_cfg(regressor_anchor="aobt3_proxy"), regressor=reg, X=X)
    model.fit(X, y, cat_features=["apt"], eval_set=(X, y))
    # regressor rows are 3..9 (rows 1, 2 stale-labelled); the fallback for a missing anchor is the
    # per-airport median of those rows' targets: EDDF rows 5 and 7 -> median(30,100, 700) = 15,400
    anchor = np.array([700.0, 800.0, 850.0, 750.0, 950.0, 1_100.0, 15_400.0, 800.0, 7_200.0])
    # LIRF rows 3, 4, 8 -> median(900, 800, 800) = 800 for row 8
    keep = np.array([False, False, True, True, True, True, True, True, True])
    assert reg.fit_y == pytest.approx((y - anchor)[keep])
    assert reg.eval_y == pytest.approx((y - anchor)[keep])
    detail = model.predict_detail(X)
    assert detail["y_reg"].to_numpy() == pytest.approx(anchor + 10.0)
    # outside the universe the prediction is the anchored regressor's
    assert detail["y_pred"].to_numpy()[6] == pytest.approx(15_410.0)
    # an unanchored model is unchanged
    reg2 = _RecordingRegressor()
    model2, _ = _model(_cfg(), regressor=reg2, X=X)
    model2.fit(X, y, cat_features=["apt"])
    assert reg2.fit_y == pytest.approx(y[keep])
    with pytest.raises(ValueError, match="anchor_clip"):
        _cfg(anchor_clip=[10.0, 5.0])


def test_regressor_anchor_leaves_the_config_hash_alone_while_unset():
    base = load_config("exp043_final_screen.yaml")
    assert base.hash() == "f76b35c9"
    assert base.replace(model={"mixture": {"regressor_anchor": "aobt3_proxy"}}).hash() != base.hash()


def test_join_window_invalidates_candidates_keeps_long_targets_and_clips():
    X, y = _frame()
    # the window centre per row (takeoff - LOBT); rows 1-3 have a window around 1,000 s, so the 20,000 s schedule
    # of rows 2-3 cannot be their block time; row 5 has a window around 30,000 s; the rest have none
    X["lobt_delta_full"] = [1_000.0, 1_000.0, 1_000.0, np.nan, 30_000.0, np.nan, np.nan, np.nan, np.nan]
    cfg = _cfg(window_column="lobt_delta_full", regressor_max_target_sec=7_200.0)
    model, _ = _model(cfg, X=X)
    B, valid, universe = model.candidates(X)
    assert universe.tolist() == [True, False, False, False, False, False, False, False, False]
    # the long target of row 5 (30,100 s) sits inside its window, so the regressor keeps it despite the cap;
    # row 9 (20,450 s, no window) is still dropped
    reg = _StubRegressor()
    model, _ = _model(cfg, regressor=reg, X=X)
    model.fit(X, y, cat_features=["apt"])
    assert 5 in reg.fitted_index and 9 not in reg.fitted_index
    # predictions are clipped to the window: row 5's regressor answer (1,000 s) is lifted to 30,000 - 3,606
    detail = model.predict_detail(X)
    assert detail.loc[5, "y_pred"] == pytest.approx(30_000.0 - 3_606.0)
    assert detail.loc[7, "y_pred"] == pytest.approx(700.0)
    base = load_config("exp043_final_screen.yaml")
    assert base.replace(model={"mixture": {"window_column": "lobt_delta_full"}}).hash() != base.hash() == "f76b35c9"


def test_anchor_airports_limit_the_anchor():
    X, y = _frame()
    X["aobt3_proxy"] = [700.0, 800.0, 850.0, 750.0, 950.0, 1_100.0, 600.0, 900.0, 1_000.0]
    reg = _RecordingRegressor()
    model, _ = _model(_cfg(regressor_anchor="aobt3_proxy", anchor_airports=["EGLL"]), regressor=reg, X=X)
    model.fit(X, y, cat_features=["apt"])
    # only the EGLL rows (6 and 9) are anchored; the rest learn the raw target
    keep = np.array([False, False, True, True, True, True, True, True, True])
    anchor = np.array([0, 0, 0, 0, 0, 1_100.0, 0, 0, 1_000.0])
    assert reg.fit_y == pytest.approx((y - anchor)[keep])
    assert model.predict_detail(X)["y_reg"].to_numpy() == pytest.approx(anchor + 10.0)
