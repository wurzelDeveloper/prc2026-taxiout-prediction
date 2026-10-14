"""Model backends, selected by `model.type` in an experiment config."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from prc.config import ExperimentConfig
from prc.models.base import CategoryUniverse, Model, prepare_matrix
from prc.models.baselines import AOBT3Proxy, GlobalMedian, GroupMedian
from prc.models.gbdt import CatBoostModel, HistGBModel, LightGBMModel
from prc.models.mixture import RegimeMixtureModel
from prc.models.per_airport import PerAirportModel

# Baselines take no category universe: they either ignore features entirely or
# read one named column, so they are constructed differently from the GBDTs.
BASELINES = {
    "median": GlobalMedian,
    "group_median": GroupMedian,
    "aobt3_proxy": AOBT3Proxy,
}

GBDTS = {
    "catboost": CatBoostModel,
    "lightgbm": LightGBMModel,
    "hgb": HistGBModel,
}

__all__ = [
    "AOBT3Proxy",
    "BASELINES",
    "CatBoostModel",
    "CategoryUniverse",
    "GBDTS",
    "GlobalMedian",
    "GroupMedian",
    "HistGBModel",
    "LightGBMModel",
    "Model",
    "PerAirportModel",
    "RegimeMixtureModel",
    "feature_contract",
    "load_model",
    "make_model",
    "is_baseline",
    "plan_refit_iterations",
    "prepare_matrix",
    "refit_iterations",
    "restore_fit_state",
    "set_predict_threads",
    "state_columns",
]

LOADABLE = ("catboost",)
"""Backends whose saved models `load_model` can rebuild. Every run of the submission is CatBoost; LightGBM and HistGB
would also need the run's frozen `CategoryUniverse` levels (they take integer codes), which runs do not save."""


def is_baseline(model_type: str) -> bool:
    return model_type in BASELINES


def make_model(
    cfg: ExperimentConfig,
    universe: CategoryUniverse,
    seed: int,
    train_dir=None,
    *,
    iterations: int | dict[str, int] | None = None,
    classifier_iterations: int | None = None,
):
    """Build the model named by the config, wrapped per-airport if requested.

    ``train_dir`` is where a backend may stream its training log. The factory
    takes a ``tag`` so per-airport runs get one subdirectory each rather than
    ten models overwriting a single learning curve.

    ``iterations`` fixes a GBDT's number of rounds -- for a refit, which has no
    eval set to early-stop on. One count, or for a per-airport model a
    ``{airport: count}`` map; an airport missing from the map keeps the count
    the config gives. Baselines ignore it. ``classifier_iterations`` is the
    same budget for the regime classifier of a ``model.mixture`` run.
    """
    kind = cfg.model.type

    def subdir(tag: str | None):
        if train_dir is None:
            return None
        return train_dir / tag if tag else train_dir

    if kind in BASELINES:
        def factory(tag: str | None = None):
            return BASELINES[kind](**cfg.model.params)
    elif kind in GBDTS:
        backend = GBDTS[kind]

        def factory(tag: str | None = None):
            params = dict(cfg.model.params)
            count = iterations.get(tag) if isinstance(iterations, dict) else iterations
            if count is not None:
                params[backend.ITERATIONS_KEY] = int(count)
            return backend(universe, seed=seed, train_dir=subdir(tag), **params)
    else:
        raise KeyError(f"unknown model.type '{kind}'. Known: {sorted(BASELINES) + sorted(GBDTS)}")

    model = PerAirportModel(factory) if cfg.model.per_airport else factory()
    if cfg.model.mixture.enabled:
        # The classifier's curves stream to `model/<fold>/classifier/`, which
        # `prc progress` labels like an airport's.
        model = RegimeMixtureModel(
            model, cfg.model.mixture, universe, seed=seed,
            train_dir=subdir("classifier"), classifier_iterations=classifier_iterations,
        )
    return model


def refit_iterations(model) -> int | dict[str, int] | None:
    """What a fitted model says a refit of it should run: an int, a per-airport
    map, or None when it has no early-stopped count (baselines, hgb)."""
    fn = getattr(model, "refit_iterations", None)
    return fn() if callable(fn) else None


def plan_refit_iterations(per_fold: list, scale: float) -> int | dict[str, int] | None:
    """Turn the folds' early-stopped counts into the refit's: the mean over the
    folds that reported one, times ``scale`` (the refit's rows over a fold's),
    never below 1. Per-airport maps combine airport by airport; an airport no
    fold reported is absent from the result and keeps the config's count.
    """
    counts = [c for c in per_fold if c is not None]
    if not counts:
        return None

    def grow(n: float) -> int:
        return max(1, int(round(n * scale)))

    if all(isinstance(c, dict) for c in counts):
        airports = sorted(set().union(*(c.keys() for c in counts)))
        return {a: grow(float(np.mean([c[a] for c in counts if a in c]))) for a in airports}
    if all(isinstance(c, (int, np.integer)) for c in counts):
        return grow(float(np.mean(counts)))
    raise TypeError(f"refit iteration counts differ in kind across folds: {counts!r}")


# --- loading a saved model ------------------------------------------------------

def load_model(cfg: ExperimentConfig, universe: CategoryUniverse, path: Path):
    """The model a run saved at ``path`` (``runs/<id>/model/refit`` for the refit), rebuilt for prediction.

    The mirror of `make_model`: the config decides the shape (one CatBoost regressor, one per airport, wrapped in
    the regime mixture or not) and each piece reads its own files (`CatBoostModel.load`, `PerAirportModel.load`,
    `RegimeMixtureModel.load`). ``universe`` supplies the categorical column list for the string encoding. A run
    saved before the state files existed comes back with ``state_source_ == "missing"`` on the per-airport wrapper
    and the mixture; `restore_fit_state` fills that state in from the fit rows.
    """
    kind = cfg.model.type
    if kind not in LOADABLE:
        raise NotImplementedError(f"loading a saved '{kind}' model is not supported (loadable: {list(LOADABLE)})")
    path = Path(path)

    def member(stem: Path) -> CatBoostModel:
        return CatBoostModel.load(stem, universe)

    model = PerAirportModel.load(path, member, suffix=".cbm") if cfg.model.per_airport else member(path)
    if cfg.model.mixture.enabled:
        model = RegimeMixtureModel.load(path, cfg.model.mixture, universe, model, seed=cfg.seed)
    return model


def restore_fit_state(model, X_fit: pd.DataFrame, y_fit) -> dict:
    """Replay the fit-time bookkeeping of a loaded model on its fit rows (features ``X_fit``, prepared target ``y_fit``).

    Fills in what the run's files did not record -- the per-airport fallback, the mixture's anchor fallback and
    single regime -- and checks what they did. The mixture's regressor sees the rows and target the mixture fitted
    it on (stale and long rows out, the anchor subtracted). ``X_fit`` needs only the columns the replay reads
    (`state_columns`). Returns a report per component.
    """
    report: dict = {}
    regressor, X_reg, y_reg = model, X_fit, np.asarray(y_fit, dtype="float64")
    if isinstance(model, RegimeMixtureModel):
        keep, y_reg, report["mixture"] = model.restore_fit_state(X_fit, y_reg)
        regressor, X_reg = model.regressor, (X_fit if keep.all() else X_fit.loc[keep])
    if isinstance(regressor, PerAirportModel):
        report["per_airport"] = regressor.restore_fit_state(X_reg, y_reg)
    return report


def state_columns(model) -> list[str]:
    """The feature columns `restore_fit_state` reads for ``model``."""
    cols: list[str] = []
    if isinstance(model, RegimeMixtureModel):
        cols += model.state_columns()
        model = model.regressor
    if isinstance(model, PerAirportModel):
        cols.append(model.group_column)
    return list(dict.fromkeys(cols))


def _catboost_members(model) -> list[tuple[str, object, list[str], list[str]]]:
    """Every fitted CatBoost object in ``model``: (label, object, feature names, categorical names)."""
    out = []
    if isinstance(model, RegimeMixtureModel):
        for i, clf in enumerate(model.classifiers):
            names = list(clf.feature_names_)
            out.append((f"classifier_{i}", clf, names, [names[j] for j in clf.get_cat_feature_indices()]))
        model = model.regressor
    if isinstance(model, PerAirportModel):
        for airport, member in sorted(model.models.items()):
            out.append((f"regressor_{airport}", member, list(member.feature_names), member.categorical_names()))
    elif isinstance(model, CatBoostModel):
        out.append(("regressor", model, list(model.feature_names), model.categorical_names()))
    return out


def feature_contract(model) -> tuple[list[str], list[str]]:
    """The feature columns (in the order the model binds them) and the categorical ones, agreed by every member.

    CatBoost binds features by position, so prediction hands every member the matrix in this order; members that
    disagree (a regressor and a classifier fitted on different matrices) cannot share one, and raise."""
    members = _catboost_members(model)
    if not members:
        raise ValueError("the model holds no fitted CatBoost member")
    _, _, names, cats = members[0]
    for label, _, other, other_cats in members[1:]:
        if other != names or other_cats != cats:
            raise ValueError(f"{label} was fitted on other feature columns than {members[0][0]}")
    return names, cats


def set_predict_threads(model, threads: int) -> None:
    """CatBoost threads for every prediction ``model`` makes (-1: CatBoost's default, all cores)."""
    if isinstance(model, RegimeMixtureModel):
        model.predict_threads = int(threads)
        model = model.regressor
    members = model.models.values() if isinstance(model, PerAirportModel) else [model]
    for member in members:
        if isinstance(member, CatBoostModel):
            member.predict_threads = int(threads)
