"""The model interface, and the categorical handling every backend shares.

`prepare_matrix` exists because of one specific silent bug. The stand column has
~1600 levels and not every level appears in every fold; if each backend derived
its own category codes per fold, the integer meaning of a code would shift
between `fit` and `predict` and the model would quietly learn nonsense. So the
level universe is frozen once, over training *and* ranking rows together, and
every fold and every backend uses it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

NA_LEVEL = "__NA__"


@runtime_checkable
class Model(Protocol):
    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        *,
        cat_features: list[str],
        eval_set: tuple[pd.DataFrame, np.ndarray] | None = None,
    ) -> "Model": ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def native_importance(self) -> pd.DataFrame: ...

    def save(self, path: Path) -> None: ...


class CategoryUniverse:
    """Frozen category levels, shared by every fold and every backend."""

    def __init__(self, X: pd.DataFrame, cat_features: list[str]):
        self.cat_features = [c for c in cat_features if c in X.columns]
        self.levels: dict[str, pd.Index] = {}
        for col in self.cat_features:
            values = X[col]
            if isinstance(values.dtype, pd.CategoricalDtype):
                levels = values.cat.categories
            else:
                levels = pd.Index(values.dropna().unique())
            if NA_LEVEL not in levels:
                levels = levels.append(pd.Index([NA_LEVEL]))
            self.levels[col] = pd.Index(levels).astype("string").sort_values()

    def as_strings(self, X: pd.DataFrame) -> pd.DataFrame:
        """Categoricals as plain strings with missing values named, for CatBoost."""
        out = X.copy()
        for col in self.cat_features:
            out[col] = out[col].astype("string").fillna(NA_LEVEL)
        return out

    def as_codes(self, X: pd.DataFrame) -> pd.DataFrame:
        """Categoricals as stable int32 codes, for backends that want integers.

        A level outside the frozen universe takes the code of the explicit ``__NA__`` level, never -1. In a run that
        cannot happen (the universe covers every row it is built from); it can on a frame the universe was not built
        from -- a ranking set released after the fit."""
        out = X.copy()
        for col in self.cat_features:
            levels = self.levels[col]
            values = out[col].astype("string").fillna(NA_LEVEL)
            # The position of each value among the sorted levels (the Categorical code), -1 for an unseen level.
            codes = np.asarray(levels.get_indexer(values), dtype="int32")
            unseen = codes < 0
            if unseen.any():
                codes[unseen] = levels.get_loc(NA_LEVEL)
            out[col] = codes
        return out


def to_float32(values: pd.Series) -> pd.Series:
    """Any numeric-ish column as plain float32.

    `np.issubdtype` raises on pandas nullable dtypes (`Int16Dtype` and friends),
    which feature blocks produce freely, so the check is done through the pandas
    API and the conversion goes via `to_numeric` to flatten nullable integers
    and their NA sentinel into float NaN.
    """
    if values.dtype == "float32":
        return values
    return pd.to_numeric(values, errors="coerce").astype("float32")


def prepare_matrix(X: pd.DataFrame, universe: CategoryUniverse, *, encoding: str) -> pd.DataFrame:
    """Assemble a backend-ready matrix, preserving column order."""
    if encoding == "string":
        frame = universe.as_strings(X)
    elif encoding == "codes":
        frame = universe.as_codes(X)
    else:
        raise ValueError(f"unknown encoding '{encoding}'")

    for col in frame.columns:
        if col not in universe.cat_features:
            frame[col] = to_float32(frame[col])
    return frame


def transform_target(y: np.ndarray, transform: str) -> np.ndarray:
    if transform == "none":
        return y
    if transform == "log1p":
        return np.log1p(np.clip(y, 0, None))
    if transform == "sqrt":
        return np.sqrt(np.clip(y, 0, None))
    raise ValueError(f"unknown target transform '{transform}'")


def inverse_transform_target(y: np.ndarray, transform: str) -> np.ndarray:
    if transform == "none":
        return y
    if transform == "log1p":
        return np.expm1(y)
    if transform == "sqrt":
        return np.square(y)
    raise ValueError(f"unknown target transform '{transform}'")
