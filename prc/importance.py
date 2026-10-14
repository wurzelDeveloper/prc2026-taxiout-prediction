"""Measuring what a feature is worth.

Three methods, deliberately different in what they can be trusted for:

* **gain** -- the backend's own split-gain accounting. Nearly free, but
  correlated features steal gain from one another, so read it as a ranking hint
  and nothing more.
* **permutation** -- shuffle a feature on the holdout and watch RMSE move.
  Shuffling is done **per block**, all of the block's columns together, because
  columns inside a block are correlated by construction: permute
  `rwy_dep_back_15m` alone and `rwy_dep_back_30m` covers for it, and the block
  measures as worthless when it is not.
* **ablation** -- retrain without the block (`prc ablate`). The honest number,
  and the expensive one.

None of these mean anything without a noise floor. `prc train --seeds 3` records
the run-to-run RMSE spread at fixed config; a +3s ablation delta against a +-8s
seed spread is not a finding. `prc leaderboard --features` prints the floor at
the top of the table for exactly this reason.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from prc import metrics
from prc.config import ExperimentConfig
from prc.models.base import inverse_transform_target


def _effect_row(run, cfg, block, columns, method, metric, value, std=None) -> dict:
    return {
        "run_id": run.run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_run_id": None,
        "block": block,
        "columns": ",".join(columns) if isinstance(columns, (list, tuple)) else columns,
        "method": method,
        "metric": metric,
        "value": value,
        "std": std,
        "n_seeds": 1,
        "config_hash": cfg.hash(),
        "val_scheme": cfg.validation.scheme,
    }


def measure(run, cfg: ExperimentConfig, model, matrix, *, seed: int) -> list[dict]:
    """Write importance artifacts for a finished run and return effect rows."""
    if model is None or matrix is None:
        return []
    X_val, y_val, fm = matrix
    rows: list[dict] = []

    gain = model.native_importance()
    if not gain.empty:
        run.write_csv("importance_gain.csv", gain)
        col_to_block = fm.columns_to_block
        by_block = (
            gain.assign(block=gain["feature"].map(col_to_block))
            .groupby("block", as_index=False)["gain"]
            .sum()
            .sort_values("gain", ascending=False)
        )
        total = by_block["gain"].sum() or 1.0
        run.log("\ntop blocks by gain share:")
        for _, r in by_block.head(10).iterrows():
            run.log(f"  {r['block']:<26} {100 * r['gain'] / total:5.1f}%")
        for _, r in by_block.iterrows():
            rows.append(
                _effect_row(
                    run, cfg, r["block"], fm.block_columns.get(r["block"], []),
                    "gain", "gain_share", float(r["gain"] / total),
                )
            )

    if cfg.evaluation.permutation.enabled:
        perm = permutation_importance(cfg, model, X_val, y_val, fm, seed=seed)
        if not perm.empty:
            run.write_csv("importance_perm.csv", perm)
            run.log("\npermutation importance (holdout RMSE increase when block is shuffled):")
            for _, r in perm.head(12).iterrows():
                run.log(f"  {r['block']:<26} {r['rmse_delta']:+8.1f} +-{r['rmse_delta_std']:.1f}")
            for _, r in perm.iterrows():
                rows.append(
                    _effect_row(
                        run, cfg, r["block"], fm.block_columns.get(r["block"], []),
                        "permutation", "rmse_delta", float(r["rmse_delta"]), float(r["rmse_delta_std"]),
                    )
                )
    return rows


def permutation_importance(
    cfg: ExperimentConfig,
    model,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    fm,
    *,
    seed: int,
) -> pd.DataFrame:
    opts = cfg.evaluation.permutation
    rng = np.random.default_rng(seed)

    if opts.sample and len(X_val) > opts.sample:
        keep = rng.choice(len(X_val), size=opts.sample, replace=False)
        X_val = X_val.iloc[keep]
        y_val = np.asarray(y_val)[keep]

    # Preparing the matrix -- casting ~20 categoricals to strings for CatBoost --
    # costs more than the prediction itself, and re-running it for every shuffle
    # dominated the measurement. Preparation is a pure per-column transform, so
    # permuting rows before or after it gives identical results: prepare once,
    # then shuffle the prepared frame.
    if hasattr(model, "prepare") and hasattr(model, "predict_prepared"):
        working = model.prepare(X_val)
        predict = model.predict_prepared
    else:
        working = X_val.copy()
        predict = model.predict

    def score(frame: pd.DataFrame) -> float:
        pred = inverse_transform_target(np.asarray(predict(frame), dtype="float64"), cfg.target.transform)
        if cfg.target.residual_of and cfg.target.residual_of in frame.columns:
            base = pd.to_numeric(frame[cfg.target.residual_of], errors="coerce").to_numpy(dtype="float64")
            pred = pred + np.where(np.isfinite(base), base, 0.0)
        return metrics.regression_metrics(y_val, pred)["rmse"]

    baseline = score(working)

    if opts.granularity == "column":
        units = {c: [c] for c in working.columns}
    else:
        units = {b: [c for c in cols if c in working.columns] for b, cols in fm.block_columns.items()}

    rows = []
    frame = working
    for unit, cols in units.items():
        if not cols:
            continue
        deltas = []
        original = {col: frame[col] for col in cols}
        for _ in range(max(opts.repeats, 1)):
            order = rng.permutation(len(frame))
            # One shared permutation across the block's columns keeps their
            # joint structure intact while destroying the link to the target --
            # otherwise we would also be measuring the loss of within-block
            # coherence, which is not what the block is being credited for.
            # `.iloc[order].set_axis(...)` permutes values while preserving the
            # column's dtype, which a numpy round-trip would flatten to object.
            for col in cols:
                frame[col] = original[col].iloc[order].set_axis(frame.index)
            deltas.append(score(frame) - baseline)
        # Restore in place rather than re-copying the frame each round: with
        # ~40 blocks and 3 repeats, copying a 200k x 100 matrix per shuffle is
        # most of the runtime of an otherwise cheap measurement.
        for col, values in original.items():
            frame[col] = values
        rows.append(
            {
                "block": unit,
                "n_columns": len(cols),
                "rmse_delta": float(np.mean(deltas)),
                "rmse_delta_std": float(np.std(deltas)),
                "baseline_rmse": baseline,
            }
        )
    return pd.DataFrame(rows).sort_values("rmse_delta", ascending=False).reset_index(drop=True)
