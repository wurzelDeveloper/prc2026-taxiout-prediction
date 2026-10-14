"""Reference models that need no ML library.

These matter for two reasons beyond being a score floor.

First, they let the whole pipeline -- config, splits, feature build, metrics,
tracking, submission -- be built and verified before CatBoost is involved at
all, so a bug in the harness cannot hide behind a bad model.

Second, they have known scores on the default holdout (Jan + Jul 2025):

    GlobalMedian   RMSE 693.6   MAE 315.9
    GroupMedian    RMSE 636.1   MAE 222.4
    AOBT3Proxy     RMSE 672.8   MAE 258.7

If a run ever comes in dramatically *below* those, that is not a breakthrough,
it is the off-block time leaking back into the features.

That `GroupMedian` beats `AOBT3Proxy` on RMSE is not a mistake: RMSE here is
dominated by a very small tail (targets reach 131,167s), and the proxy is
unavailable on exactly the ragged rows where the damage is done.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class _BaselineMixin:
    def native_importance(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["feature", "gain"])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".json").write_text(json.dumps(self.state(), default=str), encoding="utf-8")

    def state(self) -> dict:  # pragma: no cover - overridden
        return {}


class GlobalMedian(_BaselineMixin):
    """Predict the same number for every flight."""

    def __init__(self, **_: object):
        self.value = float("nan")

    def fit(self, X, y, *, cat_features=None, eval_set=None):
        self.value = float(np.nanmedian(y))
        return self

    def predict(self, X) -> np.ndarray:
        return np.full(len(X), self.value, dtype="float64")

    def state(self) -> dict:
        return {"value": self.value}


class GroupMedian(_BaselineMixin):
    """Median taxi time for the flight's (stand, runway) pairing.

    Falls back through progressively coarser keys when a pairing is unseen or
    too rare, which is the same hierarchy `stand_runway_target_stats` uses -- so
    this baseline shows how much of that block's value is just the lookup.
    """

    def __init__(self, keys: list[list[str]] | None = None, min_support: int = 20, **_: object):
        self.key_levels = keys or [
            ["apt", "stand", "runway"],
            ["apt", "stand"],
            ["apt", "runway"],
            ["apt"],
        ]
        self.min_support = min_support
        self.tables: list[pd.Series] = []
        self.fallback = float("nan")

    @staticmethod
    def _key(X: pd.DataFrame, cols: list[str]) -> pd.Series:
        parts = [X[c].astype("string").fillna("__NA__") for c in cols]
        joined = parts[0]
        for part in parts[1:]:
            joined = joined + "_" + part
        return joined

    def fit(self, X, y, *, cat_features=None, eval_set=None):
        y = pd.Series(np.asarray(y, dtype="float64"), index=X.index)
        self.tables = []
        for cols in self.key_levels:
            missing = [c for c in cols if c not in X.columns]
            if missing:
                continue
            grouped = y.groupby(self._key(X, cols), observed=True)
            table = grouped.median()
            table = table[grouped.size() >= self.min_support]
            self.tables.append(table)
        self.fallback = float(np.nanmedian(y))
        return self

    def predict(self, X) -> np.ndarray:
        out = pd.Series(np.nan, index=X.index, dtype="float64")
        for cols, table in zip(self.key_levels, self.tables):
            if out.notna().all():
                break
            need = out.isna()
            out[need] = self._key(X.loc[need], cols).map(table).to_numpy()
        return out.fillna(self.fallback).to_numpy()

    def state(self) -> dict:
        return {"key_levels": self.key_levels, "min_support": self.min_support, "fallback": self.fallback}


class AOBT3Proxy(_BaselineMixin):
    """Use the Network Manager off-block time directly.

    ``aobt3_proxy`` is already `takeoff - AOBT_3_flt`, so the "model" is a single
    learned offset correcting the systematic bias between the two sources, with
    the global median standing in for the ~1.5% of flights NM never matched.
    """

    def __init__(self, column: str = "aobt3_proxy", **_: object):
        self.column = column
        self.offset = 0.0
        self.fallback = float("nan")

    def fit(self, X, y, *, cat_features=None, eval_set=None):
        if self.column not in X.columns:
            raise KeyError(f"AOBT3Proxy needs the '{self.column}' feature; include group:nm_schedule")
        proxy = pd.to_numeric(X[self.column], errors="coerce").to_numpy(dtype="float64")
        y = np.asarray(y, dtype="float64")
        mask = np.isfinite(proxy) & np.isfinite(y)
        self.offset = float(np.median(y[mask] - proxy[mask])) if mask.any() else 0.0
        self.fallback = float(np.nanmedian(y))
        return self

    def predict(self, X) -> np.ndarray:
        proxy = pd.to_numeric(X[self.column], errors="coerce").to_numpy(dtype="float64")
        out = proxy + self.offset
        return np.where(np.isfinite(out), out, self.fallback)

    def state(self) -> dict:
        return {"column": self.column, "offset": self.offset, "fallback": self.fallback}
