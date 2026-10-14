"""Gradient-boosting backends behind one interface.

CatBoost is the day-one model: its ordered target statistics handle the
high-cardinality categoricals here (~1600 stands, ~960 destinations) better than
the alternatives, without any manual encoding.

The other adapters are thin on purpose. They exist so that swapping backend is a
one-line config change rather than a refactor -- which matters mostly for speed:
CatBoost on 2.4M rows is slow enough that a leave-one-out ablation sweep is
better run on a subsample, on LightGBM, or both.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from prc.models.base import CategoryUniverse, prepare_matrix

CATBOOST_DEFAULTS = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": 5000,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 3.0,
    "one_hot_max_size": 16,
    "od_type": "Iter",
    "od_wait": 200,
    # NB: `metric_period` does not help here -- CatBoost ignores it for the
    # evaluation metric whenever the overfitting detector is on, and says so.
    # The eval set is scored every iteration; `od_wait` is the only lever that
    # shortens a fit once it has stopped improving.
    "thread_count": -1,
    "verbose": 200,
}


class CatBoostModel:
    encoding = "string"
    ITERATIONS_KEY = "iterations"
    """The params key that fixes the number of boosting rounds (a refit has no eval set to stop on)."""
    predict_threads = -1
    """CatBoost threads for `predict` (`prc predict --threads`); -1 is CatBoost's own default, all cores."""

    def __init__(self, universe: CategoryUniverse, seed: int = 42, train_dir: Path | None = None, **params):
        self.universe = universe
        self.params = CATBOOST_DEFAULTS | params | {"random_seed": seed}
        self.model = None
        self.feature_names: list[str] = []

        # Point CatBoost at the run directory so it streams `learn_error.tsv`
        # and `test_error.tsv` as it goes. Without this a long fit is completely
        # opaque -- Python buffers the verbose output, so there is no way to
        # tell a slow run from a hung one. The files double as the run's saved
        # learning curve.
        if train_dir is not None:
            Path(train_dir).mkdir(parents=True, exist_ok=True)
            self.params["train_dir"] = str(train_dir)
            self.params.setdefault("allow_writing_files", True)
        else:
            self.params.setdefault("allow_writing_files", False)

    def fit(self, X, y, *, cat_features, eval_set=None, sample_weight=None):
        from catboost import CatBoostRegressor, Pool

        self.feature_names = list(X.columns)
        cats = [c for c in cat_features if c in X.columns]
        train_pool = Pool(prepare_matrix(X, self.universe, encoding=self.encoding), y, cat_features=cats, weight=sample_weight)
        valid_pool = None
        if eval_set is not None:
            Xv, yv = eval_set
            valid_pool = Pool(prepare_matrix(Xv, self.universe, encoding=self.encoding), yv, cat_features=cats)

        params = dict(self.params)
        if valid_pool is None:
            # Early stopping needs a validation pool; without one the od_* keys
            # are silently inert, so drop them rather than imply they applied.
            params.pop("od_type", None)
            params.pop("od_wait", None)
        self.model = CatBoostRegressor(**params)
        self.model.fit(train_pool, eval_set=valid_pool, use_best_model=valid_pool is not None)
        return self

    def prepare(self, X) -> pd.DataFrame:
        return prepare_matrix(X, self.universe, encoding=self.encoding)

    def predict_prepared(self, frame) -> np.ndarray:
        return self.model.predict(frame, thread_count=self.predict_threads)

    def predict(self, X) -> np.ndarray:
        return self.predict_prepared(self.prepare(X))

    @classmethod
    def load(cls, path: Path, universe: CategoryUniverse) -> "CatBoostModel":
        """The fitted model `save` wrote at ``path`` (``<stem>.cbm``), ready to predict.

        CatBoost binds features by position, not by name, so the caller must hand `predict` the columns in
        `feature_names` order (`prc.models.feature_contract`); categoricals go in as strings via ``universe``
        (only its ``cat_features`` list matters for the string encoding, never its levels).
        """
        from catboost import CatBoostRegressor

        file = Path(path).with_suffix(".cbm")
        if not file.exists():
            raise FileNotFoundError(f"no saved CatBoost model at {file}")
        obj = cls(universe)
        obj.model = CatBoostRegressor()
        obj.model.load_model(str(file))
        obj.feature_names = list(obj.model.feature_names_)
        return obj

    def categorical_names(self) -> list[str]:
        """The feature names CatBoost treats as categorical, in feature order."""
        return [self.feature_names[i] for i in self.model.get_cat_feature_indices()]

    def native_importance(self) -> pd.DataFrame:
        values = self.model.get_feature_importance()
        return pd.DataFrame({"feature": self.feature_names, "gain": values}).sort_values("gain", ascending=False)

    @property
    def best_iteration(self) -> int | None:
        return getattr(self.model, "best_iteration_", None)

    def refit_iterations(self) -> int | None:
        """Rounds a refit without an eval set should run: the early-stopped best
        iteration as a tree count (CatBoost's index is zero-based). None when the
        fit had no eval set, so the config's own count applies."""
        best = self.best_iteration
        return None if best is None else int(best) + 1

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path.with_suffix(".cbm")))


class LightGBMModel:
    encoding = "codes"
    ITERATIONS_KEY = "n_estimators"

    def __init__(self, universe: CategoryUniverse, seed: int = 42, train_dir: Path | None = None, **params):
        self.universe = universe
        self.params = {
            "objective": "l2",
            "learning_rate": 0.05,
            "num_leaves": 255,
            "min_data_in_leaf": 100,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "n_estimators": 5000,
            "verbosity": -1,
        } | params | {"random_state": seed}
        self.early_stopping_rounds = self.params.pop("early_stopping_rounds", 200)
        self.model = None
        self.feature_names: list[str] = []

    def fit(self, X, y, *, cat_features, eval_set=None, sample_weight=None):
        import lightgbm as lgb

        self.feature_names = list(X.columns)
        cats = [c for c in cat_features if c in X.columns]
        Xt = prepare_matrix(X, self.universe, encoding=self.encoding)
        self.model = lgb.LGBMRegressor(**self.params)
        callbacks, eval_arg = [], None
        if eval_set is not None:
            Xv, yv = eval_set
            eval_arg = [(prepare_matrix(Xv, self.universe, encoding=self.encoding), yv)]
            callbacks = [lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
        self.model.fit(Xt, y, sample_weight=sample_weight, eval_set=eval_arg, categorical_feature=cats, callbacks=callbacks)
        return self

    def prepare(self, X) -> pd.DataFrame:
        return prepare_matrix(X, self.universe, encoding=self.encoding)

    def predict_prepared(self, frame) -> np.ndarray:
        return self.model.predict(frame)

    def predict(self, X) -> np.ndarray:
        return self.predict_prepared(self.prepare(X))

    def native_importance(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"feature": self.feature_names, "gain": self.model.booster_.feature_importance("gain")}
        ).sort_values("gain", ascending=False)

    def refit_iterations(self) -> int | None:
        best = getattr(self.model, "best_iteration_", None)
        return int(best) if best else None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.booster_.save_model(str(path.with_suffix(".lgb")))


class HistGBModel:
    """sklearn's HistGradientBoosting -- available without extra installs.

    It caps categorical cardinality at 255, which `stand` (~1600) and `ades`
    (~960) blow straight past, so those columns fall back to numeric codes and
    are split on ordinally. That is a real handicap: read a weak HistGB score as
    a property of the backend, not of the features.
    """

    encoding = "codes"
    ITERATIONS_KEY = "max_iter"

    def __init__(self, universe: CategoryUniverse, seed: int = 42, train_dir: Path | None = None, **params):
        self.universe = universe
        self.params = {"max_iter": 500, "learning_rate": 0.05, "early_stopping": False} | params
        self.params["random_state"] = seed
        self.model = None
        self.feature_names: list[str] = []
        self.demoted: list[str] = []

    def fit(self, X, y, *, cat_features, eval_set=None, sample_weight=None):
        from sklearn.ensemble import HistGradientBoostingRegressor

        self.feature_names = list(X.columns)
        Xt = prepare_matrix(X, self.universe, encoding=self.encoding)
        cats, self.demoted = [], []
        for col in cat_features:
            if col not in Xt.columns:
                continue
            (cats if len(self.universe.levels[col]) <= 255 else self.demoted).append(col)
        if self.demoted:
            print(f"  HistGB: {len(self.demoted)} categorical(s) exceed 255 levels, demoted to ordinal: {self.demoted}")
        mask = [c in cats for c in Xt.columns]
        self.model = HistGradientBoostingRegressor(categorical_features=mask, **self.params)
        self.model.fit(Xt, y, sample_weight=sample_weight)
        return self

    def prepare(self, X) -> pd.DataFrame:
        return prepare_matrix(X, self.universe, encoding=self.encoding)

    def predict_prepared(self, frame) -> np.ndarray:
        return self.model.predict(frame)

    def predict(self, X) -> np.ndarray:
        return self.predict_prepared(self.prepare(X))

    def native_importance(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["feature", "gain"])

    def refit_iterations(self) -> int | None:
        return None  # no early stopping here, so the config's `max_iter` is already the count

    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path.with_suffix(".joblib"))
