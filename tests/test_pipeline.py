"""Config, validation, metrics, models and submission plumbing.

These run without the competition data where possible, so a broken harness
surfaces in seconds rather than after a feature build.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prc import metrics, paths, validation
from prc.config import ExperimentConfig, load_config
from prc.models.base import CategoryUniverse, inverse_transform_target, prepare_matrix, transform_target
from prc.models.baselines import AOBT3Proxy, GlobalMedian, GroupMedian


# --- config -----------------------------------------------------------------

def test_base_config_loads():
    cfg = load_config("base.yaml")
    assert cfg.validation.scheme == "seasonal_holdout"
    assert cfg.evaluation.primary_metric == "rmse"


def test_extends_is_resolved_and_child_wins():
    cfg = load_config("exp002_catboost_core.yaml")
    assert cfg.model.type == "catboost"
    assert cfg.target.transform == "log1p"
    assert cfg.validation.holdout_months == ["2025-01", "2025-07"]  # inherited


def test_overrides_are_typed_not_stringly():
    cfg = load_config("base.yaml", ["model.params.depth=10", "model.per_airport=true", "data.sample_frac=0.25"])
    assert cfg.model.params["depth"] == 10
    assert cfg.model.per_airport is True
    assert cfg.data.sample_frac == 0.25


def test_unknown_config_keys_fail_loudly():
    with pytest.raises(ValueError, match="unknown key"):
        load_config("base.yaml", ["validaton.scheme=group_kfold_day"])


def test_hash_ignores_notes_but_tracks_substance():
    a = load_config("base.yaml")
    assert a.replace(notes="anything at all").hash() == a.hash()
    assert a.replace(model={"params": {"depth": 9}}).hash() != a.hash()


def test_queue_builds_one_train_command_per_config():
    import sys

    from prc.queue import commands

    cmds = commands(["a.yaml", "b.yaml"], seeds=3, no_test=True, overrides=["data.sample_frac=0.2"])
    assert len(cmds) == 2
    assert cmds[0][:5] == [sys.executable, "-m", "prc", "train", "-c"] and cmds[0][5] == "a.yaml"
    assert cmds[1][6:] == ["--seeds", "3", "--no-test", "-o", "data.sample_frac=0.2"]
    plain = commands(["a.yaml"])[0]
    assert "--seeds" not in plain and "--no-test" not in plain and "-o" not in plain
    kept = commands(["a.yaml"], seeds=3, keep_test=True)[0]
    assert kept[6:] == ["--seeds", "3", "--keep-test"]


def test_data_dir_override_moves_the_whole_data_tree(tmp_path):
    import os
    import subprocess
    import sys

    env = os.environ | {"PRC_DATA_DIR": str(tmp_path)}
    code = "from prc import paths; print(paths.DATA); print(paths.MOVEMENTS_FILE); print(paths.RANKING_FILE); print(paths.CACHE)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=paths.ROOT, check=True).stdout.split("\n")
    assert out[0] == str(tmp_path.resolve())
    assert out[1] == str(tmp_path.resolve() / "interim" / "movements.parquet")
    assert out[2] == str(tmp_path.resolve() / "competition_files" / "ranking.parquet")
    assert out[3] == str(paths.ROOT / "cache")   # the cache and the runs stay shared
    assert paths.DATA == paths.ROOT / "data" or os.environ.get("PRC_DATA_DIR")


def test_geometry_configs_load():
    geo = load_config("exp008_geometry.yaml")
    assert "group:geometry" in geo.features.include and "group:core" in geo.features.include
    assert geo.target.transform == "log1p"  # inherited through exp002
    per_apt = load_config("exp009_geometry_per_airport.yaml")
    assert per_apt.model.per_airport is True
    assert per_apt.model.params["depth"] == 7 and per_apt.model.params["iterations"] == 3000
    everything = load_config("exp010_all.yaml")
    assert {"group:weather", "group:weather_fcst", "group:geometry"} <= set(everything.features.include)


def test_bad_enum_values_are_rejected():
    with pytest.raises(ValueError, match="transform"):
        load_config("base.yaml", ["target.transform=cuberoot"])
    with pytest.raises(ValueError, match="scheme"):
        load_config("base.yaml", ["validation.scheme=random_kfold"])


# --- refit on all rows ------------------------------------------------------

def test_refit_on_all_is_a_typed_model_key_that_leaves_the_folds_alone():
    cfg = load_config("base.yaml")
    assert cfg.model.refit_on_all is False
    on = load_config("base.yaml", ["model.refit_on_all=true"])
    assert on.model.refit_on_all is True
    assert on.hash() != cfg.hash()
    assert on.split_hash() == cfg.split_hash()  # the refit does not move the folds
    final = load_config("exp017_final_raw_per_airport.yaml")
    assert final.model.refit_on_all and final.model.per_airport
    assert final.target.transform == "none" and final.target.clip is None  # exp005's raw target, inherited
    assert final.model.params["iterations"] == 4000
    assert final.data.clean.summary() == "day3600"  # wrong-date rows only; the 2 h tail stays in the fit


def test_stale_configs_load():
    stale = load_config("exp020_raw_per_airport_stale.yaml")
    assert "group:stale" in stale.features.include and stale.target.residual_of == "stale_base"
    assert stale.target.transform == "none" and stale.model.per_airport
    feat_only = load_config("exp021_raw_per_airport_stale_feat.yaml")
    assert feat_only.target.residual_of is None and "group:stale" in feat_only.features.include
    final = load_config("exp022_final_stale.yaml")
    assert final.model.refit_on_all and final.target.residual_of == "stale_base" and not final.data.clean.is_active


def test_winter_check_config_loads():
    base = load_config("exp028_final_stale_only.yaml")
    winter = load_config("exp029_winter_check.yaml")
    assert winter.validation.holdout_months == ["2025-02", "2025-12"]
    assert winter.model.refit_on_all is False and winter.model.mixture.enabled and winter.model.per_airport
    assert winter.features.include == base.features.include and winter.data.clean.summary() == "day3600"
    assert winter.split_hash() != base.split_hash()  # other holdout months, other fold caches


def test_add_to_base_plans_one_variant_per_block(tmp_path):
    from prc.ablate import emit_configs, plan_variants

    cfg = load_config("exp002_catboost_core.yaml")
    base, variants, moved, sign = plan_variants(cfg, "add-to-base", ["stale_schedule"])
    assert base is cfg and sign == -1.0
    (name, variant), = variants
    assert name == "stale_schedule" and variant.name == "catboost_core__plus_stale_schedule"
    assert variant.features.include == [*cfg.features.include, "stale_schedule"]
    assert moved == {"stale_schedule": ["stale_schedule"]}  # its dependency core_keys was already there
    with pytest.raises(ValueError, match="already in the config"):
        plan_variants(cfg, "add-to-base", ["nm_deltas"])
    with pytest.raises(KeyError):
        plan_variants(cfg, "add-to-base", ["no_such_block"])
    with pytest.raises(ValueError, match="needs --blocks"):
        plan_variants(cfg, "add-to-base", None)

    # leave-one-out is untouched by the new mode.
    loo_base, loo_variants, loo_moved, loo_sign = plan_variants(cfg, "leave-one-out", ["nm_deltas"])
    assert loo_base is cfg and loo_sign == 1.0 and loo_variants[0][1].name == "catboost_core__wo_nm_deltas"
    assert "nm_proxy_context" in loo_moved["nm_deltas"]

    (path,) = emit_configs([variant], tmp_path)
    assert path.name == "catboost_core__plus_stale_schedule.yaml"
    back = load_config(path)
    assert back.name == variant.name and back.features.include == variant.features.include
    assert back.hash() == variant.hash()


def test_per_airport_cleaning_configs_load():
    dayfix = load_config("exp018_raw_per_airport_dayfix.yaml")
    assert dayfix.model.per_airport and dayfix.target.transform == "none"
    assert dayfix.data.clean.summary() == "day3600"
    full = load_config("exp019_raw_per_airport_clean.yaml")
    assert full.data.clean.summary() == "min30|max7200|day3600"
    assert full.split_hash() != dayfix.split_hash()  # different rules, different fold caches


def test_refit_iterations_are_planned_from_the_folds():
    from prc.models import plan_refit_iterations

    assert plan_refit_iterations([None, None], 1.2) is None
    assert plan_refit_iterations([100, 200], 1.2) == 180
    assert plan_refit_iterations([1], 0.1) == 1  # never zero rounds
    per_airport = plan_refit_iterations([{"EDDF": 100, "LSZH": 10}, {"EDDF": 300}], 2.0)
    assert per_airport == {"EDDF": 400, "LSZH": 20}  # LSZH from the one fold that reported it
    with pytest.raises(TypeError):
        plan_refit_iterations([100, {"EDDF": 1}], 1.0)


def test_make_model_fixes_the_round_count_for_a_refit():
    from prc.models import make_model

    X = pd.DataFrame({"apt": ["EDDF", "LSZH"], "x": [1.0, 2.0]})
    universe = CategoryUniverse(X, ["apt"])
    cfg = load_config("base.yaml", ["model.params.iterations=3000"])
    assert make_model(cfg, universe, 42).params["iterations"] == 3000
    assert make_model(cfg, universe, 42, iterations=1234).params["iterations"] == 1234
    wrapped = make_model(cfg.replace(model={"per_airport": True}), universe, 42, iterations={"EDDF": 50})
    assert wrapped.factory("EDDF").params["iterations"] == 50
    assert wrapped.factory("LSZH").params["iterations"] == 3000  # not in the map: the config's count


def test_per_airport_refit_iterations_skip_models_without_an_early_stopped_count():
    from prc.models import refit_iterations
    from prc.models.per_airport import PerAirportModel

    class Stub:
        def __init__(self, n):
            self.n = n

        def refit_iterations(self):
            return self.n

    wrapped = PerAirportModel(lambda tag=None: None)
    wrapped.models = {"EDDF": Stub(120), "LSZH": Stub(None)}
    assert refit_iterations(wrapped) == {"EDDF": 120}
    assert refit_iterations(GlobalMedian()) is None  # baselines have nothing to say


# --- validation -------------------------------------------------------------

def _fake_movements(days: int = 120, per_day: int = 40) -> tuple[pd.DataFrame, pd.Index]:
    times = pd.date_range("2025-01-01", periods=days, freq="D", tz="UTC")
    rows = []
    for i, day in enumerate(times):
        for j in range(per_day):
            rows.append({"mvt_id": i * per_day + j, "MVT_TIME_UTC_mvt": day + pd.Timedelta(minutes=j)})
    mvt = pd.DataFrame(rows).set_index("mvt_id")
    return mvt, mvt.index


def test_seasonal_holdout_selects_the_named_months():
    mvt, idx = _fake_movements(days=200)
    cfg = ExperimentConfig().replace(validation={"scheme": "seasonal_holdout", "holdout_months": ["2025-01"]})
    (split,) = validation.make_splits(cfg, mvt, idx)
    val_months = mvt.loc[split.val_idx, "MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m").unique()
    assert list(val_months) == ["2025-01"]
    assert not set(split.train_idx) & set(split.val_idx)


def test_group_kfold_never_splits_a_day():
    """A day on both sides of a split leaks: congestion features make flights in
    the same window near-duplicates of one another."""
    mvt, idx = _fake_movements()
    cfg = ExperimentConfig().replace(validation={"scheme": "group_kfold_day", "n_folds": 5})
    splits = validation.make_splits(cfg, mvt, idx)
    assert len(splits) == 5
    day = mvt["MVT_TIME_UTC_mvt"].dt.floor("D")
    for split in splits:
        assert not set(day.loc[split.train_idx]) & set(day.loc[split.val_idx])
    covered = set().union(*(set(s.val_idx) for s in splits))
    assert covered == set(idx)


def test_splits_are_deterministic():
    """Ablations compare runs pairwise, so folds must not move between runs."""
    mvt, idx = _fake_movements()
    cfg = ExperimentConfig().replace(validation={"scheme": "group_kfold_day", "n_folds": 4})
    first = validation.make_splits(cfg, mvt, idx)
    second = validation.make_splits(cfg, mvt, idx)
    for a, b in zip(first, second):
        assert a.val_idx.equals(b.val_idx)


def test_strict_forward_trains_only_on_later_months():
    mvt, idx = _fake_movements(days=300)
    cfg = ExperimentConfig().replace(validation={"scheme": "strict_forward", "holdout_months": ["2025-01"]})
    (split,) = validation.make_splits(cfg, mvt, idx)
    train_months = mvt.loc[split.train_idx, "MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
    assert train_months.min() > "2025-01"


def test_forward_holdout_trains_only_on_earlier_months():
    mvt, idx = _fake_movements(days=300)
    cfg = ExperimentConfig().replace(validation={"scheme": "forward_holdout", "holdout_months": ["2025-08", "2025-09"]})
    (split,) = validation.make_splits(cfg, mvt, idx)
    train_months = mvt.loc[split.train_idx, "MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
    val_months = mvt.loc[split.val_idx, "MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
    assert train_months.max() < "2025-08" and set(val_months) == {"2025-08", "2025-09"}
    # months after the holdout (October onward) are in neither side
    assert len(split.train_idx) + len(split.val_idx) < len(idx)
    with pytest.raises(ValueError, match="before"):
        validation.make_splits(ExperimentConfig().replace(validation={"scheme": "forward_holdout", "holdout_months": ["2025-01"]}), mvt, idx)


# --- metrics ----------------------------------------------------------------

def test_regression_metrics_are_correct():
    y = np.array([100.0, 200.0, 300.0])
    p = np.array([110.0, 190.0, 300.0])
    m = metrics.regression_metrics(y, p)
    assert m["rmse"] == pytest.approx(np.sqrt((100 + 100 + 0) / 3))
    assert m["mae"] == pytest.approx(20 / 3)
    assert m["bias"] == pytest.approx(0.0)
    assert m["n"] == 3


def test_metrics_ignore_non_finite_pairs():
    m = metrics.regression_metrics([1.0, 2.0, np.nan], [1.0, 2.0, 99.0])
    assert m["n"] == 2
    assert m["rmse"] == 0.0


# --- target transforms ------------------------------------------------------

@pytest.mark.parametrize("transform", ["none", "log1p", "sqrt"])
def test_transforms_round_trip(transform):
    y = np.array([60.0, 900.0, 7200.0])
    back = inverse_transform_target(transform_target(y, transform), transform)
    assert np.allclose(back, y)


# --- categorical universe ---------------------------------------------------

def test_category_codes_are_stable_across_subsets():
    """The bug this prevents: a level missing from one fold shifting every code."""
    X = pd.DataFrame({"stand": pd.Categorical(["A1", "A2", "B7", "A1", "C3"]), "n": [1.0, 2, 3, 4, 5]})
    universe = CategoryUniverse(X, ["stand"])

    full = prepare_matrix(X, universe, encoding="codes")
    subset = prepare_matrix(X.iloc[[0, 2]], universe, encoding="codes")
    assert list(subset["stand"]) == [full["stand"].iloc[0], full["stand"].iloc[2]]


def test_missing_categoricals_become_a_named_level():
    X = pd.DataFrame({"stand": pd.Series(["A1", None], dtype="string").astype("category")})
    universe = CategoryUniverse(X, ["stand"])
    out = prepare_matrix(X, universe, encoding="string")
    assert out["stand"].tolist() == ["A1", "__NA__"]
    assert not out["stand"].isna().any()


def test_nullable_integer_features_survive_preparation():
    X = pd.DataFrame({"k": pd.Series([1, None, 3], dtype="Int16"), "c": pd.Categorical(["x", "y", "x"])})
    out = prepare_matrix(X, CategoryUniverse(X, ["c"]), encoding="string")
    assert out["k"].dtype == "float32"
    assert np.isnan(out["k"].iloc[1])


# --- baselines --------------------------------------------------------------

def test_global_median_predicts_the_median():
    X = pd.DataFrame({"a": range(5)})
    model = GlobalMedian().fit(X, np.array([1.0, 2, 3, 4, 100]))
    assert model.predict(X)[0] == 3.0


def test_group_median_falls_back_when_support_is_thin():
    X = pd.DataFrame(
        {
            "apt": ["E"] * 30 + ["E"],
            "stand": ["A1"] * 30 + ["ZZ"],
            "runway": ["07"] * 30 + ["25"],
        }
    )
    y = np.concatenate([np.full(30, 500.0), [9999.0]])
    model = GroupMedian(min_support=5).fit(X, y)
    pred = model.predict(X)
    assert pred[0] == pytest.approx(500.0)
    # The one-off pairing has no support at any specific level, so it must land
    # on the airport median rather than memorising its own value.
    assert pred[-1] != pytest.approx(9999.0)


def test_aobt3_proxy_learns_an_offset_and_survives_gaps():
    X = pd.DataFrame({"aobt3_proxy": [900.0, 1000.0, np.nan]})
    model = AOBT3Proxy().fit(X, np.array([960.0, 1060.0, 1200.0]))
    pred = model.predict(X)
    assert pred[0] == pytest.approx(960.0)
    assert np.isfinite(pred[2])


def test_aobt3_proxy_says_which_block_is_missing():
    with pytest.raises(KeyError, match="nm_schedule"):
        AOBT3Proxy().fit(pd.DataFrame({"other": [1.0]}), np.array([1.0]))


# --- submission -------------------------------------------------------------

@pytest.mark.skipif(not paths.SUBMITTING_FILE.exists(), reason="competition data not present")
def test_submission_validator_rejects_a_broken_frame():
    from prc.submit import _validate

    template = pd.read_parquet(paths.SUBMITTING_FILE)
    good = template.copy()
    good["TAXITIME_SEC_mvt"] = np.int32(900)
    _validate(good, template)

    with pytest.raises(AssertionError, match="int32"):
        bad = good.copy()
        bad["TAXITIME_SEC_mvt"] = bad["TAXITIME_SEC_mvt"].astype("float64")
        _validate(bad, template)

    with pytest.raises(AssertionError, match="rows"):
        _validate(good.iloc[:-1], template)

    with pytest.raises(AssertionError, match="MVT_ID_mvt set"):
        shifted = good.copy()
        shifted.loc[shifted.index[0], "MVT_ID_mvt"] += 1
        _validate(shifted, template)


def test_residual_learning_rejects_a_nonlinear_transform():
    """Residuals are signed; log1p of a negative is NaN, which would train the
    model on an all-NaN target and only surface much later."""
    with pytest.raises(ValueError, match="must be 'none' when residual_of"):
        load_config("base.yaml", ["target.residual_of=aobt3_proxy", "target.transform=log1p"])


# --- permutation importance -------------------------------------------------

class _ColumnEcho:
    """A model whose prediction is exactly column 'a' -- so the true importance
    of each block is known in advance."""

    def predict(self, X):
        return pd.to_numeric(X["a"], errors="coerce").to_numpy(dtype="float64")


class _Blocks:
    block_columns = {"useful": ["a"], "useless": ["b"], "categorical": ["c"]}


def _permutation_fixture():
    rng = np.random.default_rng(0)
    n = 4000
    X = pd.DataFrame(
        {
            "a": rng.normal(1000, 100, n).astype("float32"),
            "b": rng.normal(0, 1, n).astype("float32"),
            "c": pd.Categorical(rng.choice(list("xyz"), n)),
        }
    )
    return X, X["a"].to_numpy(dtype="float64")


def test_permutation_credits_only_the_block_the_model_uses():
    from prc.importance import permutation_importance

    X, y = _permutation_fixture()
    cfg = load_config("base.yaml", ["evaluation.permutation.repeats=2", "evaluation.permutation.sample=0"])
    out = permutation_importance(cfg, _ColumnEcho(), X, y, _Blocks(), seed=1).set_index("block")
    assert out.loc["useful", "rmse_delta"] > 100
    assert out.loc["useless", "rmse_delta"] == pytest.approx(0.0, abs=1e-9)
    assert out.loc["categorical", "rmse_delta"] == pytest.approx(0.0, abs=1e-9)


def test_permutation_leaves_the_validation_frame_untouched():
    """It shuffles columns in place for speed, so it must put them back exactly
    -- otherwise every block measured after the first sees corrupted data."""
    from prc.importance import permutation_importance

    X, y = _permutation_fixture()
    before = X.copy()
    cfg = load_config("base.yaml", ["evaluation.permutation.repeats=2", "evaluation.permutation.sample=0"])
    permutation_importance(cfg, _ColumnEcho(), X, y, _Blocks(), seed=1)
    pd.testing.assert_frame_equal(X, before)


class _PreparingEcho(_ColumnEcho):
    """A backend that exposes the prepare/predict_prepared fast path, as the
    GBDT adapters do."""

    def prepare(self, X):
        out = X.copy()
        out["c"] = out["c"].astype("string").fillna("__NA__")
        return out

    def predict_prepared(self, frame):
        return self.predict(frame)


def test_permutation_fast_path_agrees_with_the_slow_one():
    """Preparation is a pure per-column transform, so shuffling before or after
    it must give the same answer -- that equivalence is what licenses preparing
    the matrix once instead of once per shuffle."""
    from prc.importance import permutation_importance

    X, y = _permutation_fixture()
    cfg = load_config("base.yaml", ["evaluation.permutation.repeats=2", "evaluation.permutation.sample=0"])
    slow = permutation_importance(cfg, _ColumnEcho(), X, y, _Blocks(), seed=7).set_index("block")
    fast = permutation_importance(cfg, _PreparingEcho(), X, y, _Blocks(), seed=7).set_index("block")
    for block in ("useful", "useless", "categorical"):
        assert fast.loc[block, "rmse_delta"] == pytest.approx(slow.loc[block, "rmse_delta"])


def test_permutation_fast_path_does_not_mutate_the_input():
    from prc.importance import permutation_importance

    X, y = _permutation_fixture()
    before = X.copy()
    cfg = load_config("base.yaml", ["evaluation.permutation.repeats=1", "evaluation.permutation.sample=0"])
    permutation_importance(cfg, _PreparingEcho(), X, y, _Blocks(), seed=3)
    pd.testing.assert_frame_equal(X, before)


def test_drop_columns_is_hash_neutral_while_empty_and_guards_the_mixture():
    """An empty list leaves every existing run name alone; a non-empty one is substance."""
    a = load_config("base.yaml")
    payload = a.to_dict()
    payload["features"].pop("drop_columns")
    payload["model"]["mixture"].pop("classifier_seeds")  # added later, hash-neutral at 1 the same way
    payload["model"].pop("month_weights")  # added later, hash-neutral while empty
    assert payload["model"].pop("eval_max_target_sec") is None  # added 7 Oct, hash-neutral while None
    assert payload["model"].pop("block_dropout") == {}  # added 9 Oct, hash-neutral while empty
    mixture = payload["model"]["mixture"]
    # added 30 Sep, hash-neutral while unset (Config.hash drops the clip and the half-width with their switches)
    assert mixture["regressor_anchor"] is None and mixture["anchor_airports"] is None
    assert mixture["window_column"] is None
    for key in ("regressor_anchor", "anchor_clip", "anchor_airports", "window_column", "window_half_width"):
        mixture.pop(key)
    import hashlib
    import json

    payload.pop("notes", None)
    legacy = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:8]
    assert a.hash() == legacy
    assert a.replace(features={"drop_columns": ["day_of_year"]}).hash() != a.hash()
    mix = load_config("exp028_final_stale_only.yaml")
    base = mix.model.mixture.bases[0]
    with pytest.raises(ValueError, match="drop_columns"):
        mix.replace(features={"drop_columns": [base]})


def test_eval_max_target_sec_filters_only_the_early_stopping_rows_and_guards_the_mixture():
    """The early-stopping set drops rows above the cap (the metrics never see this filter); unset, it is the whole
    holdout; it is part of the hash once set; the mixture rejects it."""
    from prc.train import _eval_set

    class _Log:
        lines: list[str] = []

        def log(self, msg):
            self.lines.append(msg)

    base = load_config("base.yaml")
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}, index=pd.Index([10, 11, 12, 13], name="mvt_id"))
    y_raw = np.array([600.0, 7200.0, 7201.0, 87000.0])
    y_fit = y_raw.copy()
    Xe, ye = _eval_set(base, X, y_fit, y_raw, _Log())
    assert Xe is X and ye is y_fit
    capped = base.replace(model={"eval_max_target_sec": 7200})
    Xe, ye = _eval_set(capped, X, y_fit, y_raw, _Log())
    assert list(Xe.index) == [10, 11] and list(ye) == [600.0, 7200.0]
    assert capped.hash() != base.hash()
    with pytest.raises(ValueError, match="positive"):
        base.replace(model={"eval_max_target_sec": 0})
    with pytest.raises(ValueError, match="regressor_max_target_sec"):
        load_config("exp028_final_stale_only.yaml").replace(model={"eval_max_target_sec": 7200})


def test_tier1_configs_of_2026_09_25_load():
    """exp038 drops the four time-position columns; exp039 adds the LOBT delta as a second stale base."""
    base = load_config("exp037_final_era5_mixture.yaml")
    nodates = load_config("exp038_final_nodates_mixture.yaml")
    assert nodates.features.drop_columns == ["takeoff_epoch", "days_since_epoch", "day_of_year", "week_of_year"]
    assert nodates.features.include == base.features.include
    lobt = load_config("exp039_mixture_lobt.yaml")
    assert lobt.model.mixture.bases == ["sched_delta_full", "lobt_delta_full"]
    assert lobt.model.mixture.min_base_sec == 1800 and lobt.model.mixture.tol_sec == 60
    assert base.hash() == "f992f6e6"


def test_winter_base_and_screen_configs_of_2026_09_24_load():
    """exp040 is exp037 on the Feb+Dec holdout; exp041 adds every unused block and the two Phase A blocks."""
    base = load_config("exp037_final_era5_mixture.yaml")
    winter = load_config("exp040_winter_era5_mixture.yaml")
    assert winter.validation.holdout_months == ["2025-02", "2025-12"] and not winter.model.refit_on_all
    assert winter.features.include == base.features.include
    assert winter.split_hash(42) != base.split_hash(42)
    screen = load_config("exp041_screen_unused.yaml")
    assert screen.features.include[: len(base.features.include)] == base.features.include
    assert {"dep_live_state", "deice_location", "deicing_context", "group:geometry"} <= set(screen.features.include)
    assert screen.evaluation.permutation.enabled and not screen.model.refit_on_all
    assert base.hash() == "f992f6e6"


def test_nm_unmatched_rows_reads_the_flight_id():
    mvt = pd.DataFrame({"FLIGHT_ID_mvt": [1.0, np.nan, 3.0, np.nan]}, index=pd.Index([10, 11, 12, 13], name="mvt_id"))
    assert metrics.nm_unmatched_rows(mvt, pd.Index([11, 12, 13])).tolist() == [True, False, True]


def test_block_dropout_blanks_fit_rows_by_airport_day_deterministically_and_is_hash_neutral_while_empty():
    """``model.block_dropout`` (critique F6): the named blocks' float columns blanked on a fixed share of the fit rows'
    (airport, day) groups -- the same groups on every call, whole groups, other columns untouched; validated; empty
    leaves every run name alone."""
    from prc.train import block_dropout, dropout_groups

    base = load_config("exp043_final_screen.yaml")
    assert base.model.block_dropout == {} and base.hash() == "f76b35c9"
    cfg = base.replace(model={"block_dropout": {"surface": 0.25}})
    assert cfg.hash() != base.hash()
    for bad in ({"surface": 0.0}, {"surface": 1.0}, {"surface": -0.2}):
        with pytest.raises(ValueError, match="block_dropout"):
            base.replace(model={"block_dropout": bad})
    rng = np.random.default_rng(4)
    n = 6000
    days = pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(rng.integers(0, 120 * 86_400, n), unit="s")
    mvt = pd.DataFrame({"apt": rng.choice(["EDDF", "LSZH", "EHAM"], n), "MVT_TIME_UTC_mvt": days},
                       index=pd.Index(np.arange(n), name="mvt_id"))
    X = pd.DataFrame({"a1": rng.random(n), "a2": rng.random(n), "b1": rng.random(n)}, index=mvt.index)
    blocks = {"surface": ["a1", "a2"], "other": ["b1"]}
    out = block_dropout(cfg, mvt, X, blocks, cat_cols=[])
    again = block_dropout(cfg, mvt, X.iloc[::-1], blocks, cat_cols=[]).loc[X.index]
    pd.testing.assert_frame_equal(out, again)                      # deterministic, independent of row order
    blank = out["a1"].isna()
    assert blank.equals(out["a2"].isna()) and not out["b1"].isna().any() and not X.isna().any().any()
    assert 0.15 < blank.mean() < 0.35                              # about the share
    group = mvt["apt"] + mvt["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d")
    assert blank.groupby(group).nunique().max() == 1               # whole (airport, day) groups
    assert np.array_equal(blank.to_numpy(), dropout_groups(mvt, X.index, "surface", 0.25))
    assert block_dropout(base, mvt, X, blocks, cat_cols=[]) is X   # nothing set: the matrix itself
    with pytest.raises(ValueError, match="not selected"):
        block_dropout(base.replace(model={"block_dropout": {"nope": 0.2}}), mvt, X, blocks, cat_cols=[])
    with pytest.raises(ValueError, match="float columns"):
        block_dropout(cfg, mvt, X, blocks, cat_cols=["a2"])


def test_month_weights_are_validated_normalised_and_hash_neutral_while_empty():
    from prc.train import fit_weights

    base = load_config("exp043_final_screen.yaml")
    assert base.model.month_weights == {} and base.hash() == "f76b35c9"
    cfg = base.replace(model={"month_weights": {1: 2.0, 7: 2.0, 12: 1.5}})
    assert cfg.hash() != base.hash()
    with pytest.raises(ValueError, match="months 1-12"):
        base.replace(model={"month_weights": {13: 2.0}})
    with pytest.raises(ValueError, match="months 1-12"):
        base.replace(model={"month_weights": {1: 0.0}})
    mvt = pd.DataFrame({"MVT_TIME_UTC_mvt": pd.to_datetime(["2025-01-05", "2025-07-05", "2025-10-05", "2025-12-05"], utc=True)},
                       index=pd.Index([1, 2, 3, 4], name="mvt_id"))
    w = fit_weights(cfg, mvt, mvt.index)["sample_weight"]
    assert w.mean() == pytest.approx(1.0)
    assert w.tolist() == pytest.approx(np.array([2.0, 2.0, 1.0, 1.5]) / 1.625)
    assert fit_weights(base, mvt, mvt.index) == {}
