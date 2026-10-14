"""Running one experiment end to end.

The shape of a run: resolve the feature blocks, build the matrix once, walk the
folds, and write everything measurable to `runs/<run_id>/`.

Two details are load-bearing:

* Fold-dependent blocks (anything with ``uses_target``) are rebuilt inside each
  fold from that fold's training rows only. The rest of the matrix is computed
  once for the whole corpus and reused, because it never touches the target.
* Ranking predictions are the mean over the fold models rather than a separate
  refit. It costs nothing extra, and it means the predictions come from models
  whose target encodings were fitted the same way the validated ones were.
  ``model.refit_on_all`` is the exception for a submission run: the folds still
  do the measuring, then one more model is fitted on every training row --
  holdout months included, which under `seasonal_holdout` are the seasonal
  twins of the test months -- and the ranking predictions come from that one.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from prc import cleaning, data, disruption, importance, metrics, paths, tracking, validation
from prc.config import ExperimentConfig
from prc.features import resolve
from prc.features.build import build_matrix
from prc.models import CategoryUniverse, is_baseline, make_model, plan_refit_iterations, refit_iterations
from prc.models.base import inverse_transform_target, transform_target
from prc.models.mixture import DETAIL_COLUMNS, hedging_report


def _training_index(cfg: ExperimentConfig, mvt: pd.DataFrame, y: pd.Series, seed: int) -> pd.Index:
    idx = y.index
    if cfg.data.airports != "all":
        wanted = set(cfg.data.airports)
        idx = idx[mvt.loc[idx, "apt"].astype("string").isin(wanted).to_numpy()]
    return validation.subsample(idx, cfg.data.sample_frac, seed)


def _prepare_targets(cfg: ExperimentConfig, y: pd.Series, X: pd.DataFrame, idx: pd.Index):
    """Apply clipping, the residual base and the target transform, in that order."""
    raw = y.reindex(idx).to_numpy(dtype="float64")
    base = np.zeros(len(idx), dtype="float64")
    if cfg.target.residual_of:
        col = cfg.target.residual_of
        if col not in X.columns:
            raise KeyError(f"target.residual_of='{col}' is not a feature; include the block that makes it")
        base = pd.to_numeric(X.loc[idx, col], errors="coerce").to_numpy(dtype="float64")
        # Where the base is missing, fall back to plain target learning; the
        # `<col>_missing` indicator tells the model which regime it is in.
        base = np.where(np.isfinite(base), base, 0.0)

    fit = raw - base
    if cfg.target.clip and not cfg.target.residual_of:
        lo, hi = cfg.target.clip
        fit = np.clip(fit, lo, hi)
    return transform_target(fit, cfg.target.transform), base, raw


def _eval_set(cfg: ExperimentConfig, X_val: pd.DataFrame, y_val_fit: np.ndarray, y_val_raw: np.ndarray, run):
    """The early-stopping set: every holdout row, or with ``model.eval_max_target_sec`` only the rows whose raw target
    is at or below it. The metrics score every holdout row either way."""
    cap = cfg.model.eval_max_target_sec
    if cap is None:
        return X_val, y_val_fit
    keep = np.asarray(y_val_raw, dtype="float64") <= float(cap)
    run.log(f"    early stopping on {int(keep.sum()):,} of {len(keep):,} holdout rows (target <= {float(cap):,.0f} s)")
    return X_val.loc[keep], np.asarray(y_val_fit)[keep]


def run_experiment(
    cfg: ExperimentConfig,
    *,
    seed: int | None = None,
    compute_test: bool = True,
    run_id: str | None = None,
) -> tracking.Run:
    seed = cfg.seed if seed is None else seed
    run = tracking.Run(cfg, run_id=run_id, seed=seed)
    run.log(f"=== {run.run_id}")
    run.log(f"model={cfg.model.type} per_airport={cfg.model.per_airport} scheme={cfg.validation.scheme} seed={seed}")

    mvt = data.load_movements()
    y = data.load_target()
    specs = resolve(cfg.features.include, cfg.features.exclude)
    dep_idx = data.dep_index(mvt)
    rank_idx = data.dep_index(mvt, "rank")
    train_idx = _training_index(cfg, mvt, y, seed)
    run.log(f"blocks={len(specs)} train_rows={len(train_idx):,} rank_rows={len(rank_idx):,}")

    splits = validation.make_splits(cfg, mvt, train_idx)
    run.log(f"splits: {[ (s.name, len(s.train_idx), len(s.val_idx)) for s in splits ]}")

    # Cleaning happens *after* the split, on the training side only: flagged
    # rows may leave the fit, but every validation row is still scored, so the
    # headline RMSE stays comparable with a leaderboard that scores uncleaned
    # 2026 data. A second metric set on the unflagged rows is reported alongside.
    clean = cfg.data.clean
    flagged = cleaning.flag_rows(mvt, y, clean)
    n_flagged_fit = 0
    if clean.is_active:
        table = cleaning.flag_table(mvt, y, clean).reindex(train_idx)
        n_flagged_fit = int(np.mean([flagged.reindex(s.train_idx, fill_value=False).sum() for s in splits]))
        run.log(f"cleaning rules: {clean.summary()}")
        run.log("\n" + cleaning.describe_flags(table, mvt.loc[train_idx, "apt"]).to_string())
        splits = cleaning.apply_to_splits(splits, flagged, clean.drop_from_fit)
        mode = "dropped from the fit" if clean.drop_from_fit else "flagged only"
        run.log(f"splits after cleaning ({mode}): {[ (s.name, len(s.train_idx), len(s.val_idx)) for s in splits ]}")

    oof, test_preds, fold_metrics = [], [], []
    fold_iterations: list = []
    fold_clf_iterations: list = []
    mixture_info: list[dict] = []
    fold_rank_hedge: list[float] = []
    block_columns: dict[str, list[str]] = {}
    block_hashes: dict[str, str] = {}
    last_model = last_matrix = None

    for split in splits:
        run.log(f"--- {split.name}: build features")
        fit_mask = pd.Series(False, index=mvt.index)
        fit_mask.loc[split.train_idx] = True
        fm = build_matrix(
            specs,
            mvt,
            dep_idx,
            data_version=cfg.data.version,
            params=cfg.features.params,
            fold=f"{split.name}__{cfg.split_hash(seed)}",
            fit_mask=fit_mask,
            target=y.reindex(dep_idx),
            cfg=cfg,
        )
        block_columns, block_hashes = fm.block_columns, fm.block_hashes
        X = fm.X
        universe = CategoryUniverse(X, fm.cat_cols)

        y_fit, _, _ = _prepare_targets(cfg, y, X, split.train_idx)
        y_val_fit, val_base, y_val_raw = _prepare_targets(cfg, y, X, split.val_idx)

        X_tr, X_val = X.loc[split.train_idx], X.loc[split.val_idx]
        X_tr = block_dropout(cfg, mvt, X_tr, fm.block_columns, fm.cat_cols, run)
        model = make_model(cfg, universe, seed, train_dir=run.path("model") / split.name)

        t0 = time.perf_counter()
        eval_set = None if is_baseline(cfg.model.type) else _eval_set(cfg, X_val, y_val_fit, y_val_raw, run)
        model.fit(X_tr, y_fit, cat_features=fm.cat_cols, eval_set=eval_set, **fit_weights(cfg, mvt, X_tr.index, run))
        fit_seconds = time.perf_counter() - t0

        # A mixture model reports its regime probabilities next to the prediction.
        detail = model.predict_detail(X_val) if hasattr(model, "predict_detail") else None
        raw_pred = detail["y_pred"].to_numpy() if detail is not None else np.asarray(model.predict(X_val), dtype="float64")
        pred = inverse_transform_target(raw_pred, cfg.target.transform)
        pred = pred + val_base
        stats = metrics.regression_metrics(y_val_raw, pred)
        stats["fold"] = split.name
        stats["fit_seconds"] = round(fit_seconds, 1)
        fold_metrics.append(stats)
        run.log(f"    {split.name}: {metrics.summarise(stats)}  ({fit_seconds:.0f}s)")
        fold_iterations.append(refit_iterations(model))
        if fold_iterations[-1] is not None:
            run.log(f"    {split.name}: best iteration {fold_iterations[-1]}")
        fold_clf_iterations.append(model.classifier_refit_iterations() if detail is not None else None)

        fold_oof = pd.DataFrame({"fold": split.name, "y_true": y_val_raw, "y_pred": pred}, index=split.val_idx)
        if detail is not None:
            for col in DETAIL_COLUMNS:
                fold_oof[col] = detail[col].to_numpy()
            # The regime each holdout row turned out to be in (-1 outside the universe): the calibration check.
            fold_oof["regime"] = model.labels(X_val, y_val_raw)
            mixture_info.append(_log_mixture(run, split.name, model, fold_oof))
        oof.append(fold_oof)
        model.save(run.path(f"model/{split.name}"))

        if compute_test and len(rank_idx):
            _, test_base, _ = _prepare_targets(cfg, y, X, rank_idx)
            if detail is not None:
                rank_detail = model.predict_detail(X.loc[rank_idx])
                raw_rank = rank_detail["y_pred"].to_numpy()
                fold_rank_hedge.append(hedging_report(rank_detail)["hedge"])
            else:
                raw_rank = np.asarray(model.predict(X.loc[rank_idx]), dtype="float64")
            tp = inverse_transform_target(raw_rank, cfg.target.transform)
            test_preds.append(pd.Series(tp + test_base, index=rank_idx))

        last_model, last_matrix = model, (X_val, y_val_raw, fm)

    oof_frame = pd.concat(oof)
    oof_frame["flagged"] = flagged.reindex(oof_frame.index, fill_value=False).to_numpy(dtype=bool)
    overall = metrics.regression_metrics(oof_frame["y_true"], oof_frame["y_pred"])
    overall_clean = cleaning.clean_metrics(oof_frame["y_true"], oof_frame["y_pred"], oof_frame["flagged"])
    run.log(f"=== OOF {metrics.summarise(overall)}")
    if clean.is_active:
        n_flagged_val = int(oof_frame["flagged"].sum())
        run.log(f"=== OOF clean ({n_flagged_val:,} flagged rows excluded) {metrics.summarise(overall_clean)}")
        if overall_clean["n"] == 0:
            run.log("    every validation row is flagged; the clean metrics are empty")

    # The secondary number for tail work: the same rows minus the wrong-date
    # block times, which nothing in a row announces and which two holdout rows
    # alone can swing the headline by 60 s.
    oof_frame["dayoffset"] = (
        cleaning.day_offset_rows(mvt, y).reindex(oof_frame.index, fill_value=False).to_numpy(dtype=bool)
    )
    overall_ex_dayoffset = cleaning.clean_metrics(oof_frame["y_true"], oof_frame["y_pred"], oof_frame["dayoffset"])
    n_dayoffset_val = int(oof_frame["dayoffset"].sum())
    run.log(f"=== OOF without wrong-date rows ({n_dayoffset_val:,} excluded) {metrics.summarise(overall_ex_dayoffset)}")

    # The winter number: the same rows restricted to the airport-days whose
    # arrivals taxied in far longer than usual (snow, closures). Jan+Jul 2025
    # holds almost none of them; Feb and Dec 2025 and the 2026 ranking months do.
    oof_frame["disrupted"] = disruption.disrupted_rows(mvt, oof_frame.index).to_numpy(dtype=bool)
    overall_disrupted = cleaning.clean_metrics(oof_frame["y_true"], oof_frame["y_pred"], ~oof_frame["disrupted"])
    n_disrupted_val = int(oof_frame["disrupted"].sum())
    run.log(f"=== OOF on disrupted airport-days ({n_disrupted_val:,} rows) {metrics.summarise(overall_disrupted)}")
    overall_bulk = cleaning.clean_metrics(
        oof_frame["y_true"], oof_frame["y_pred"], oof_frame["y_true"].to_numpy(dtype="float64") >= metrics.BULK_MAX_SEC
    )
    run.log(f"=== OOF bulk (target below {metrics.BULK_MAX_SEC:,.0f} s) {metrics.summarise(overall_bulk)}")
    oof_frame["nm_unmatched"] = metrics.nm_unmatched_rows(mvt, oof_frame.index)
    unmatched = oof_frame["nm_unmatched"].to_numpy(dtype=bool)
    below = oof_frame["y_true"].to_numpy(dtype="float64") < metrics.BULK_MAX_SEC
    overall_nm_unmatched = cleaning.clean_metrics(oof_frame["y_true"], oof_frame["y_pred"], ~unmatched)
    overall_bulk_nm_unmatched = cleaning.clean_metrics(oof_frame["y_true"], oof_frame["y_pred"], ~(unmatched & below))
    run.log(f"=== OOF without an NM match ({int(unmatched.sum()):,} rows) {metrics.summarise(overall_nm_unmatched)}; "
            f"bulk rmse {overall_bulk_nm_unmatched['rmse']:.1f}")

    hedging = None
    if "hedge_var" in oof_frame.columns:
        # The hedge's expected cost on the holdout and on the ranking rows (target-free), and the
        # calibration that makes it mean something (findings 9.11).
        holdout = hedging_report(oof_frame, oof_frame["regime"] if "regime" in oof_frame.columns else None)
        hedging = {
            "holdout": holdout["hedge"],
            "rank_folds": float(np.mean(fold_rank_hedge)) if fold_rank_hedge else None,
            "rank_refit": None,
            "ece": holdout.get("ece"),
            "brier": holdout.get("brier"),
            "calibration": holdout.get("calibration"),
        }
        rank_note = f", ranking rows (fold models) {hedging['rank_folds']:,.0f}" if hedging["rank_folds"] is not None else ""
        ece_note = f"; calibration ece {hedging['ece']:.4f}, brier {hedging['brier']:.4f}" if hedging["ece"] is not None else ""
        run.log(f"=== hedging cost (mean regime variance per row, s^2): holdout {hedging['holdout']:,.0f}{rank_note}{ece_note}")

    run.write_parquet("oof_predictions.parquet", oof_frame)
    run.log_feature_list(block_columns, block_hashes)

    by_airport = metrics.breakdown_table(
        oof_frame["y_true"], oof_frame["y_pred"], mvt.loc[oof_frame.index, "apt"]
    )
    unflagged = oof_frame.loc[~oof_frame["flagged"]]
    clean_by_airport = metrics.breakdown_table(
        unflagged["y_true"], unflagged["y_pred"], mvt.loc[unflagged.index, "apt"]
    )
    by_airport["rmse_clean"] = clean_by_airport["rmse"].reindex(by_airport.index)
    by_airport["n_clean"] = clean_by_airport["n"].reindex(by_airport.index).fillna(0).astype(int)
    run.write_csv("metrics_by_airport.csv", by_airport, index=True)
    run.log("\n" + by_airport[["rmse", "mae", "bias", "n", "rmse_clean", "n_clean"]].round(1).to_string())

    _write_breakdowns(run, cfg, mvt, oof_frame)

    refit_info = None
    test = pd.concat(test_preds, axis=1).mean(axis=1).rename("y_pred") if test_preds else None
    if cfg.model.refit_on_all and test is not None:
        run.write_parquet("test_predictions_folds.parquet", test.to_frame())
        test, refit_info = _refit_on_all(
            run, cfg, mvt, y, specs, dep_idx, rank_idx, train_idx, flagged, splits,
            seed=seed, fold_iterations=fold_iterations, fold_clf_iterations=fold_clf_iterations, fold_test=test,
        )
    elif cfg.model.refit_on_all:
        run.log(
            "refit_on_all is set, but test predictions are off "
            "(--no-test, --seeds without --keep-test, or ablation): no refit"
        )
    if test is not None:
        run.write_parquet("test_predictions.parquet", test.to_frame())
        origin = "refit model" if refit_info else "mean over fold models"
        run.log(f"ranking predictions: {len(test):,} rows, median {test.median():,.0f}s ({origin})")
    if hedging is not None and refit_info and refit_info.get("hedge_rank") is not None:
        hedging["rank_refit"] = refit_info["hedge_rank"]

    effects = importance.measure(run, cfg, last_model, last_matrix, seed=seed)

    run.log_metrics(
        {
            "overall": overall,
            "overall_clean": overall_clean,
            "n_flagged_fit": n_flagged_fit,
            "n_flagged_val": int(oof_frame["flagged"].sum()),
            "clean": clean.summary(),
            "overall_ex_dayoffset": overall_ex_dayoffset,
            "n_dayoffset_val": n_dayoffset_val,
            "overall_disrupted": overall_disrupted,
            "n_disrupted_val": n_disrupted_val,
            "overall_bulk": overall_bulk,
            "overall_nm_unmatched": overall_nm_unmatched,
            "overall_bulk_nm_unmatched": overall_bulk_nm_unmatched,
            "hedging": hedging,
            "mixture": mixture_info or None,
            "refit": refit_info,
            "folds": fold_metrics,
            "by_airport": by_airport.to_dict(orient="index"),
            "n_features": sum(len(v) for v in block_columns.values()),
            "n_blocks": len(block_columns),
            "n_train": int(sum(len(s.train_idx) for s in splits) / max(len(splits), 1)),
            "n_val": int(len(oof_frame)),
            "primary_metric": cfg.evaluation.primary_metric,
            "primary_value": overall.get(cfg.evaluation.primary_metric),
        }
    )
    run.log_feature_effects(effects)
    run.finish()
    return run


def dropout_groups(mvt: pd.DataFrame, idx: pd.Index, block: str, share: float) -> np.ndarray:
    """Rows of ``idx`` whose (airport, takeoff day) group is blanked for ``block``: a crc32 of
    ``block|airport|day`` below ``share`` of 2**32. Deterministic (no RNG, no seed), stable across folds and runs."""
    import zlib

    apt = mvt.loc[idx, "apt"].astype(str).to_numpy(dtype=object)
    day = mvt.loc[idx, "MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d").to_numpy(dtype=object)
    keys = pd.Series([f"{block}|{a}|{d}" for a, d in zip(apt, day)])
    uniq = keys.unique()
    u = pd.Series([zlib.crc32(k.encode()) / 2**32 for k in uniq], index=uniq)
    return (keys.map(u).to_numpy(dtype="float64") < float(share))


def block_dropout(cfg, mvt: pd.DataFrame, X_fit: pd.DataFrame, block_columns: dict, cat_cols, run=None) -> pd.DataFrame:
    """``model.block_dropout``: the fit rows with the named blocks' columns blanked on their (airport, day) groups
    (`dropout_groups`); a copy, or ``X_fit`` itself when nothing is set. Only ever applied to fit rows."""
    drop = dict(cfg.model.block_dropout)
    if not drop:
        return X_fit
    if unknown := sorted(set(drop) - set(block_columns)):
        raise ValueError(f"model.block_dropout names blocks that are not selected: {unknown}")
    out = X_fit.copy()
    for block, share in drop.items():
        cols = [c for c in block_columns[block] if c in out.columns]
        if bad := sorted(set(cols) & set(cat_cols)):
            raise ValueError(f"model.block_dropout blanks float columns only; {block} has categoricals {bad}")
        mask = dropout_groups(mvt, out.index, block, share)
        out.loc[mask, cols] = np.nan
        if run is not None:
            run.log(f"    block dropout {block}: {len(cols)} columns blanked on {mask.mean():.1%} of {len(mask):,} fit rows")
    return out


def fit_weights(cfg, mvt: pd.DataFrame, idx: pd.Index, run=None) -> dict:
    """``{"sample_weight": w}`` from `model.month_weights` (mean one over ``idx``), or ``{}`` when none are set."""
    weights = dict(cfg.model.month_weights)
    if not weights:
        return {}
    if is_baseline(cfg.model.type):
        raise ValueError("model.month_weights needs a GBDT; the baselines take no row weights")
    month = mvt.loc[idx, "MVT_TIME_UTC_mvt"].dt.month.to_numpy()
    w = np.array([float(weights.get(int(m), 1.0)) for m in range(13)])[month]
    w = w / w.mean()
    if run is not None:
        by = pd.Series(w).groupby(month).first().round(3).to_dict()
        run.log(f"    month weights on {len(w):,} fit rows (normalised): {by}")
    return {"sample_weight": w}


def _log_mixture(run, name: str, model, fold_oof: pd.DataFrame) -> dict:
    """Log what the regime classifier did on this fold; the dict goes into metrics.json."""
    info = {"fold": name} | dict(model.diagnostics_ or {})
    u = fold_oof["in_universe"].to_numpy(dtype=bool)
    err = (fold_oof["y_pred"] - fold_oof["y_true"]).to_numpy(dtype="float64")
    info["rmse_outside"] = float(np.sqrt(np.mean(err[~u] ** 2))) if (~u).any() else None
    info["p_stale_sum_val"] = float(fold_oof["p_stale"].sum())

    def fmt(key: str, digits: int = 0) -> str:
        value = info.get(key)
        return "n/a" if value is None else f"{value:,.{digits}f}"

    run.log(
        f"    {name}: mixture universe {info.get('n_universe_val', 0):,} val rows, labels {info.get('labels_val')}, "
        f"expected stale {info['p_stale_sum_val']:,.1f}; auc {fmt('auc', 3)}, logloss {fmt('logloss', 3)}; "
        f"universe rmse {fmt('rmse_universe_mixture')} (regressor alone {fmt('rmse_universe_regressor')}, "
        f"oracle {fmt('rmse_universe_oracle')}), outside {fmt('rmse_outside')}"
    )
    importance_table = model.classifier_importance()
    if not importance_table.empty:
        run.write_csv("importance_gain_classifier.csv", importance_table)
    return info


def refit_index(cfg: ExperimentConfig, train_idx: pd.Index, flagged: pd.Series) -> pd.Index:
    """The rows the refit is fitted on: every training row (holdout months included) minus what the cleaning rules
    drop from a fit. `prc predict` rebuilds the refit's fold-dependent blocks from exactly these rows."""
    clean = cfg.data.clean
    if not clean.is_active:
        return train_idx
    whole = validation.Split("refit", train_idx, train_idx[:0])
    return cleaning.apply_to_splits([whole], flagged, clean.drop_from_fit)[0].train_idx


def refit_features(cfg: ExperimentConfig, specs, mvt: pd.DataFrame, y: pd.Series, dep_idx: pd.Index, refit_idx: pd.Index, seed: int):
    """The refit's feature matrix: every block over every departure, the fold-dependent ones fitted on ``refit_idx``
    and cached under ``fold_refit__<split_hash>``."""
    fit_mask = pd.Series(False, index=mvt.index)
    fit_mask.loc[refit_idx] = True
    return build_matrix(
        specs, mvt, dep_idx,
        data_version=cfg.data.version, params=cfg.features.params,
        fold=f"refit__{cfg.split_hash(seed)}", fit_mask=fit_mask, target=y.reindex(dep_idx), cfg=cfg,
    )


def rank_predictions(cfg: ExperimentConfig, model, X: pd.DataFrame, y: pd.Series, rank_idx: pd.Index):
    """A fitted model's ranking predictions with the run's post-processing: the mixture's regime blend (its detail
    frame is returned too, else None), the inverse target transform and the residual base added back.
    ``target.clip`` shapes the fitting target only, so nothing is clipped here."""
    _, test_base, _ = _prepare_targets(cfg, y, X, rank_idx)
    detail = None
    if hasattr(model, "predict_detail"):
        detail = model.predict_detail(X.loc[rank_idx])
        raw_pred = detail["y_pred"].to_numpy()
    else:
        raw_pred = np.asarray(model.predict(X.loc[rank_idx]), dtype="float64")
    pred = inverse_transform_target(raw_pred, cfg.target.transform)
    return pd.Series(pred + test_base, index=rank_idx, name="y_pred"), detail


def _refit_on_all(
    run, cfg, mvt, y, specs, dep_idx, rank_idx, train_idx, flagged, splits,
    *, seed: int, fold_iterations: list, fold_clf_iterations: list = (), fold_test: pd.Series,
) -> tuple[pd.Series, dict]:
    """Fit once more on every training row and predict the ranking set with that model.

    The folds' job is measurement, and the model they leave behind has never
    seen the holdout months. The refit gets them back at the price of early
    stopping: it runs the folds' best iteration scaled by the row ratio
    (`plan_refit_iterations`). Cleaning applies exactly as it did to the folds,
    and the fold-dependent blocks are recomputed over all training rows, cached
    under ``fold_refit__<split_hash>``. The fold mean is kept next to the refit
    predictions (`test_predictions_folds.parquet`) and the two are compared in
    the log, which is the only sanity check a refit without a holdout has.
    """
    refit_idx = refit_index(cfg, train_idx, flagged)
    fold_rows = float(np.mean([len(s.train_idx) for s in splits])) if splits else 0.0
    scale = len(refit_idx) / fold_rows if fold_rows else 1.0
    iterations = plan_refit_iterations(fold_iterations, scale)
    clf_iterations = plan_refit_iterations(list(fold_clf_iterations), scale)

    run.log(f"--- refit: build features on {len(refit_idx):,} rows (x{scale:.3f} of a fold's fit)")
    if iterations is None:
        run.log("    no early-stopped count from the folds; the config's iteration count applies")
    else:
        run.log(f"    iterations = folds' best x{scale:.3f}: {iterations}")
    if clf_iterations is not None:
        run.log(f"    classifier iterations = folds' best x{scale:.3f}: {clf_iterations}")
    info = {
        "n_rows": int(len(refit_idx)),
        "scale": round(scale, 4),
        "iterations": iterations,
        "classifier_iterations": clf_iterations,
    }
    # `prc progress` reads this to show the refit's own iteration budget.
    run.write_json("refit.json", info)

    fm = refit_features(cfg, specs, mvt, y, dep_idx, refit_idx, seed)
    X = fm.X
    universe = CategoryUniverse(X, fm.cat_cols)
    y_fit, _, _ = _prepare_targets(cfg, y, X, refit_idx)
    model = make_model(
        cfg, universe, seed, train_dir=run.path("model") / "refit",
        iterations=iterations, classifier_iterations=clf_iterations,
    )

    t0 = time.perf_counter()
    X_fit = block_dropout(cfg, mvt, X.loc[refit_idx], fm.block_columns, fm.cat_cols, run)
    model.fit(X_fit, y_fit, cat_features=fm.cat_cols, eval_set=None, **fit_weights(cfg, mvt, refit_idx, run))
    fit_seconds = time.perf_counter() - t0
    model.save(run.path("model/refit"))

    test, detail = rank_predictions(cfg, model, X, y, rank_idx)
    if detail is not None:
        # What the submission's stale share is, row by row: auditable after the fact.
        run.write_parquet("refit_detail.parquet", detail[list(DETAIL_COLUMNS)])
        n_universe = int(detail["in_universe"].sum())
        info |= {
            "n_universe_rank": n_universe,
            "p_stale_sum_rank": float(detail["p_stale"].sum()),
            "hedge_rank": hedging_report(detail)["hedge"],
        }
        run.log(
            f"    refit mixture: {n_universe:,} ranking rows in the universe, "
            f"expected stale rows {info['p_stale_sum_rank']:,.1f}, hedging cost {info['hedge_rank']:,.0f} s^2"
        )

    folds = fold_test.reindex(test.index).to_numpy(dtype="float64")
    diff = test.to_numpy() - folds
    info |= {
        "fit_seconds": round(fit_seconds, 1),
        "rms_diff_vs_folds": float(np.sqrt(np.mean(diff**2))),
        "corr_vs_folds": float(np.corrcoef(test.to_numpy(), folds)[0, 1]),
        "median": float(test.median()),
        "median_folds": float(np.median(folds)),
    }
    run.log(
        f"    refit done ({fit_seconds:.0f}s): vs the fold mean, rms difference {info['rms_diff_vs_folds']:,.1f}s, "
        f"correlation {info['corr_vs_folds']:.4f}, medians {info['median']:,.0f}s / {info['median_folds']:,.0f}s"
    )
    return test, info


def _write_breakdowns(run, cfg, mvt, oof_frame) -> None:
    frames = []
    for name in cfg.evaluation.breakdowns:
        if name in mvt.columns:
            by = mvt.loc[oof_frame.index, name]
        elif name == "month":
            by = mvt.loc[oof_frame.index, "MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
        elif name == "hour_local":
            by = mvt.loc[oof_frame.index, "MVT_TIME_UTC_mvt"].dt.hour
        else:
            continue
        frames.append(metrics.breakdown(oof_frame["y_true"], oof_frame["y_pred"], by, name))
    if frames:
        run.write_csv("metrics_by_breakdown.csv", pd.concat(frames, ignore_index=True))
