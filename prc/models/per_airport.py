"""One model per airport, behind the single-model interface.

Taxi geometry is airport-specific -- EGLL's median taxi-out is 1373s against
LSZH's 720s, and the stand and runway vocabularies do not even overlap -- so
ten focused models on ~208k rows each is a genuine contender against one global
model with an airport categorical, not merely a fallback.

Importance is aggregated as an n-weighted mean so a per-airport run stays
directly comparable to a global one on the same leaderboard.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

STATE_SUFFIX = "__per_airport.json"
"""`PerAirportModel.save` records the airports, their fit sizes and the fallback in ``<stem>__per_airport.json``."""
NOT_AIRPORTS = ("classifier",)
"""Member-file suffixes that belong to a wrapping model, not to an airport (``<stem>__classifier[_s<i>].cbm``)."""


class PerAirportModel:
    def __init__(self, factory, group_column: str = "apt"):
        self.factory = factory
        self.group_column = group_column
        self.models: dict[str, object] = {}
        self.sizes: dict[str, int] = {}
        self.fallback = None
        self.state_source_ = "fit"
        """``fit``, ``saved`` (`load` with the state file), ``missing`` (`load` without it) or ``restored``."""

    def _groups(self, X: pd.DataFrame) -> pd.Series:
        if self.group_column not in X.columns:
            raise KeyError(
                f"per-airport training needs the '{self.group_column}' column; include group:core"
            )
        return X[self.group_column].astype("string")

    def fit(self, X, y, *, cat_features, eval_set=None, sample_weight=None):
        y = pd.Series(np.asarray(y, dtype="float64"), index=X.index)
        w = None if sample_weight is None else np.asarray(sample_weight, dtype="float64")
        groups = self._groups(X)
        eval_groups = None
        if eval_set is not None:
            Xv, yv = eval_set
            yv = pd.Series(np.asarray(yv, dtype="float64"), index=Xv.index)
            eval_groups = self._groups(Xv)

        for airport in sorted(groups.dropna().unique()):
            rows = groups.eq(airport).to_numpy()
            sub_eval = None
            if eval_set is not None:
                vrows = eval_groups.eq(airport).to_numpy()
                if vrows.any():
                    sub_eval = (eval_set[0].loc[vrows], yv.loc[vrows].to_numpy())
            print(f"  [{airport}] fitting on {int(rows.sum()):,} rows", flush=True)
            model = self.factory(airport)
            extra = {} if w is None else {"sample_weight": w[rows]}
            model.fit(X.loc[rows], y.loc[rows].to_numpy(), cat_features=cat_features, eval_set=sub_eval, **extra)
            self.models[airport] = model
            self.sizes[airport] = int(rows.sum())

        self.fallback = float(y.median())
        return self

    def predict(self, X) -> np.ndarray:
        if self.state_source_ == "missing":
            raise RuntimeError("this per-airport model was loaded without its fallback; call restore_fit_state on its fit rows first")
        out = pd.Series(np.nan, index=X.index, dtype="float64")
        groups = self._groups(X)
        for airport, model in self.models.items():
            rows = groups.eq(airport).to_numpy()
            if rows.any():
                out[rows] = model.predict(X.loc[rows])
        # An airport present at predict time but absent from training would
        # otherwise return NaN and poison the metrics silently.
        return out.fillna(self.fallback).to_numpy()

    def refit_iterations(self) -> dict[str, int]:
        """Per airport, the rounds a refit should run. An airport whose model has
        no early-stopped count (no validation rows, or a backend without early
        stopping) is left out and keeps the config's count."""
        out = {}
        for airport, model in self.models.items():
            fn = getattr(model, "refit_iterations", None)
            n = fn() if callable(fn) else None
            if n is not None:
                out[airport] = int(n)
        return out

    def native_importance(self) -> pd.DataFrame:
        frames = []
        total = sum(self.sizes.values()) or 1
        for airport, model in self.models.items():
            imp = model.native_importance()
            if imp.empty:
                continue
            imp = imp.copy()
            imp["gain"] = imp["gain"] * (self.sizes[airport] / total)
            frames.append(imp)
        if not frames:
            return pd.DataFrame(columns=["feature", "gain"])
        return (
            pd.concat(frames)
            .groupby("feature", as_index=False)["gain"]
            .sum()
            .sort_values("gain", ascending=False)
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        for airport, model in self.models.items():
            model.save(path.with_name(f"{path.stem}__{airport}"))
        state = {
            "group_column": self.group_column,
            "airports": sorted(self.models),
            "sizes": {str(k): int(v) for k, v in self.sizes.items()},
            "fallback": None if self.fallback is None else float(self.fallback),
        }
        path.with_name(f"{path.stem}{STATE_SUFFIX}").write_text(json.dumps(state, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, load_member, *, suffix: str = ".cbm") -> "PerAirportModel":
        """The per-airport model `save` wrote at ``path``: ``load_member(<stem>__<airport>)`` for each airport.

        ``<stem>__per_airport.json`` names the airports and holds the fallback. A run saved before that file existed
        has neither: the airports are read off the member files (``<stem>__<airport><suffix>``, the classifier files
        of a mixture excepted) and the fallback stays None until `restore_fit_state` recomputes it.
        """
        path = Path(path)
        state_file = path.with_name(f"{path.stem}{STATE_SUFFIX}")
        state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else None
        if state is not None:
            airports = [str(a) for a in state["airports"]]
        else:
            prefix = f"{path.stem}__"
            airports = sorted(
                f.name[len(prefix): -len(suffix)]
                for f in path.parent.glob(f"{prefix}*{suffix}")
                if not f.name[len(prefix):].startswith(NOT_AIRPORTS)
            )
        if not airports:
            raise FileNotFoundError(f"no per-airport members '{path.stem}__<airport>{suffix}' in {path.parent}")
        model = cls(factory=None, group_column=(state or {}).get("group_column", "apt"))
        for airport in airports:
            model.models[airport] = load_member(path.with_name(f"{path.stem}__{airport}"))
        if state is not None:
            model.sizes = {str(k): int(v) for k, v in state["sizes"].items()}
            model.fallback = None if state["fallback"] is None else float(state["fallback"])
        model.state_source_ = "saved" if state is not None else "missing"
        return model

    def restore_fit_state(self, X: pd.DataFrame, y) -> dict:
        """Replay `fit`'s bookkeeping on its fit rows: the fallback (median of the fit target) and the airport sizes.

        A loaded model whose files lacked them (``state_source_ == "missing"``) takes the replayed values; one that
        had them keeps its own and the replay is only checked against them."""
        y = pd.Series(np.asarray(y, dtype="float64"), index=X.index)
        groups = self._groups(X)
        fallback = float(y.median())
        sizes = {str(k): int(v) for k, v in groups.value_counts().items()}
        fitted = sorted(groups.dropna().unique())
        report: dict = {"recomputed": [], "checked": {}, "airports_fit": [str(a) for a in fitted]}
        if sorted(self.models) != [str(a) for a in fitted]:
            report["checked"]["airports"] = False
            report["airports_loaded"] = sorted(self.models)
        if self.fallback is None:
            self.fallback = fallback
            report["recomputed"].append("fallback")
        else:
            report["checked"]["fallback"] = bool(np.isclose(self.fallback, fallback, rtol=0, atol=1e-9))
        if not self.sizes:
            self.sizes = {a: sizes.get(a, 0) for a in self.models}
            report["recomputed"].append("sizes")
        if self.state_source_ == "missing":
            self.state_source_ = "restored"
        return report
