"""Train/validation splits.

One rule dominates everything here: **no row-level random KFold.** Congestion
features make flights in the same fifteen-minute window near-duplicates of one
another, so a random split puts near-copies on both sides and reports a score
the leaderboard will not reproduce. Every scheme below splits on time or on
whole days.

The real test set is January and July *2026*, so the default scheme validates on
January and July 2025: it matches the seasonality, which is the larger effect.
It does not capture year-over-year drift -- `strict_forward` is there to be run
alongside it as an honest extrapolation check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from prc.config import ExperimentConfig


@dataclass(frozen=True)
class Split:
    name: str
    train_idx: pd.Index
    val_idx: pd.Index

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"Split({self.name}, train={len(self.train_idx):,}, val={len(self.val_idx):,})"


def _months(mvt: pd.DataFrame, idx: pd.Index) -> pd.Series:
    t = mvt.loc[idx, "MVT_TIME_UTC_mvt"]
    return t.dt.strftime("%Y-%m")


def make_splits(cfg: ExperimentConfig, mvt: pd.DataFrame, train_idx: pd.Index) -> list[Split]:
    """Build the fold list for one experiment. Deterministic -- no RNG anywhere.

    Ablations compare runs pairwise, so fold membership must depend only on the
    data, never on a seed; otherwise a feature's measured effect is confounded
    with a reshuffle.
    """
    scheme = cfg.validation.scheme
    month = _months(mvt, train_idx)

    if scheme == "seasonal_holdout":
        holdout = set(cfg.validation.holdout_months)
        is_val = month.isin(holdout).to_numpy()
        if not is_val.any():
            raise ValueError(f"no training rows in holdout months {sorted(holdout)}")
        return [Split("seasonal", train_idx[~is_val], train_idx[is_val])]

    if scheme == "forward_holdout":
        # Train only on the months before the validation months: the situation of
        # the ranking set, which lies after every training month. The seasonal
        # holdout interpolates (its fit has the months on both sides) and ranked
        # exp044 above exp043 where the board did the opposite (findings 9.16).
        holdout = sorted(cfg.validation.holdout_months)
        is_val = month.isin(set(holdout)).to_numpy()
        is_train = (month < min(holdout)).to_numpy()
        if not is_val.any() or not is_train.any():
            raise ValueError(f"forward_holdout needs training rows before and in {holdout}")
        return [Split("forward", train_idx[is_train], train_idx[is_val])]

    if scheme == "strict_forward":
        # Train on everything after the validation months. Answers "can this
        # model predict a period it has never seen the neighbourhood of".
        holdout = sorted(cfg.validation.holdout_months)
        is_val = month.isin(set(holdout)).to_numpy()
        is_train = (month > max(holdout)).to_numpy()
        return [Split("forward", train_idx[is_train], train_idx[is_val])]

    if scheme == "group_kfold_day":
        day = mvt.loc[train_idx, "MVT_TIME_UTC_mvt"].dt.floor("D")
        codes = pd.factorize(day, sort=True)[0]
        fold_of_day = codes % cfg.validation.n_folds
        return [
            Split(f"fold{k}", train_idx[fold_of_day != k], train_idx[fold_of_day == k])
            for k in range(cfg.validation.n_folds)
        ]

    if scheme == "month_forward":
        # Expanding window: train on months 1..k, validate on month k+1.
        ordered = sorted(month.unique())
        splits = []
        for k in range(1, len(ordered)):
            is_val = month.eq(ordered[k]).to_numpy()
            is_train = month.isin(ordered[:k]).to_numpy()
            splits.append(Split(f"upto_{ordered[k - 1]}", train_idx[is_train], train_idx[is_val]))
        return splits

    raise ValueError(f"unknown validation scheme '{scheme}'")


def subsample(idx: pd.Index, frac: float, seed: int) -> pd.Index:
    """Thin a training index for fast ablation sweeps."""
    if frac >= 1.0:
        return idx
    rng = np.random.default_rng(seed)
    keep = rng.random(len(idx)) < frac
    return idx[keep]
