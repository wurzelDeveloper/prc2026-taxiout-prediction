"""`prc blend`: average the predictions of several finished runs into one new run.

Seed averaging is the cheapest variance reduction a GBDT has: a blend of a
few seeds of one config typically gains a few seconds of RMSE. The blend is
written as a run of its own -- OOF and ranking predictions, per-airport
metrics, an index row -- so `prc submit --run` and `prc leaderboard` treat it
like any other run. The sources must share the validation rows (same folds),
and for ranking predictions all of them must have predicted the ranking set
(`train --seeds N --keep-test`, or single runs without `--no-test`).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from prc import cleaning, data, disruption, metrics, paths, postprocess, tracking
from prc.config import from_dict

MEAN_COLUMNS = ("y_pred", "y_reg", "p_stale", "p_stale_raw")
"""Averaged when present in every source (``y_pred`` is required)."""
CARRIED_COLUMNS = ("fold", "y_true", "flagged", "dayoffset", "disrupted", "in_universe")
"""Taken from the first source; ``y_true`` is checked to agree across sources."""


def average_predictions(frames: list[pd.DataFrame], column: str = "y_pred", weights=None) -> pd.DataFrame:
    """The row-wise (weighted) mean of ``column`` (and of the other `MEAN_COLUMNS` every frame has).

    ``weights`` (one per frame, normalised to sum to one) default to equal weights."""
    if not frames:
        raise ValueError("nothing to blend")
    first = frames[0]
    if column not in first.columns:
        raise KeyError(f"'{column}' missing from the first frame")
    for k, frame in enumerate(frames[1:], 1):
        if not frame.index.equals(first.index):
            raise ValueError(f"source {k} predicts a different row set ({len(frame):,} rows vs {len(first):,})")
        if column not in frame.columns:
            raise KeyError(f"'{column}' missing from source {k}")
        if "y_true" in first.columns and "y_true" in frame.columns:
            same = np.allclose(
                first["y_true"].to_numpy(dtype="float64"), frame["y_true"].to_numpy(dtype="float64"), equal_nan=True
            )
            if not same:
                raise ValueError(f"source {k} has different y_true values: not the same validation rows")
    out = pd.DataFrame(index=first.index)
    for col in CARRIED_COLUMNS:
        if col in first.columns:
            out[col] = first[col]
    w = np.full(len(frames), 1.0 / len(frames)) if weights is None else np.asarray(weights, dtype="float64")
    if len(w) != len(frames) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError(f"weights must be {len(frames)} non-negative numbers, got {weights!r}")
    w = w / w.sum()
    out[column] = np.tensordot(w, np.stack([f[column].to_numpy(dtype="float64") for f in frames]), axes=1)
    for col in MEAN_COLUMNS:
        if col != column and all(col in f.columns for f in frames):
            out[col] = np.tensordot(w, np.stack([f[col].to_numpy(dtype="float64") for f in frames]), axes=1)
    return out


def blend_runs(run_ids: list[str], name: str | None = None, lobt_bound: bool = False, weights=None) -> tracking.Run:
    """Average the named runs into a new run directory and index row.

    With ``lobt_bound`` the averaged OOF and ranking predictions are clipped to the NM join window
    (`prc.postprocess`); a single run is then allowed, which writes a bounded copy of it.
    """
    if len(set(run_ids)) < (1 if lobt_bound else 2):
        raise ValueError("a blend needs at least two distinct runs")
    dirs = [paths.run_dir(r) for r in run_ids]
    for d in dirs:
        for needed in ("oof_predictions.parquet", "config.resolved.json"):
            if not (d / needed).exists():
                raise FileNotFoundError(f"{d.name}: no {needed} (not a finished run)")

    payload = json.loads((dirs[0] / "config.resolved.json").read_text(encoding="utf-8"))
    # A run interrupted after its predictions were written (a crash during the
    # importance measurement) has no metrics.json; its predictions are still valid.
    source_metrics = [
        json.loads((d / "metrics.json").read_text(encoding="utf-8")) if (d / "metrics.json").exists() else {}
        for d in dirs
    ]
    cfg = from_dict(payload).replace(
        name=name or f"blend_{payload.get('name', 'run')}", notes="blend of: " + ", ".join(run_ids)
    )
    run = tracking.Run(cfg)
    run.log(f"=== {run.run_id}")
    for r, m, d in zip(run_ids, source_metrics, dirs):
        rmse = (m.get("overall") or {}).get("rmse")
        if rmse is not None:
            run.log(f"    source {r}: rmse {rmse:,.1f}")
        else:
            run.log(f"    source {r}: no metrics.json (interrupted after its predictions were written)")

    if weights is not None:
        run.log(f"    weights: {dict(zip(run_ids, weights))}")
    oof = average_predictions([pd.read_parquet(d / "oof_predictions.parquet") for d in dirs], weights=weights)
    mvt = data.load_movements(
        columns=sorted(
            set(cleaning.MVT_COLUMNS) | set(disruption.MVT_COLUMNS) | {metrics.NM_MATCH_COLUMN, "MVT_TIME_UTC_mvt", "LOBT_flt"}
        )
    )
    bound_info = {}
    if lobt_bound:
        raw = metrics.regression_metrics(oof["y_true"], oof["y_pred"])
        oof, bound_info["oof"] = postprocess.apply_lobt_bound(oof, mvt)
        run.log(f"    LOBT join window on the OOF rows: {bound_info['oof']}; rmse before {raw['rmse']:,.2f}")
    if "dayoffset" not in oof.columns:
        oof["dayoffset"] = (
            cleaning.day_offset_rows(mvt, data.load_target()).reindex(oof.index, fill_value=False).to_numpy(dtype=bool)
        )
    if "disrupted" not in oof.columns:
        oof["disrupted"] = disruption.disrupted_rows(mvt, oof.index).to_numpy(dtype=bool)
    if "flagged" not in oof.columns:
        oof["flagged"] = False
    oof["flagged"] = oof["flagged"].to_numpy(dtype=bool)
    oof["dayoffset"] = oof["dayoffset"].to_numpy(dtype=bool)
    oof["disrupted"] = oof["disrupted"].to_numpy(dtype=bool)
    run.write_parquet("oof_predictions.parquet", oof)

    overall = metrics.regression_metrics(oof["y_true"], oof["y_pred"])
    overall_clean = cleaning.clean_metrics(oof["y_true"], oof["y_pred"], oof["flagged"])
    overall_ex_dayoffset = cleaning.clean_metrics(oof["y_true"], oof["y_pred"], oof["dayoffset"])
    overall_disrupted = cleaning.clean_metrics(oof["y_true"], oof["y_pred"], ~oof["disrupted"])
    run.log(f"=== OOF {metrics.summarise(overall)}")
    run.log(
        f"=== OOF without wrong-date rows ({int(oof['dayoffset'].sum()):,} excluded) "
        f"{metrics.summarise(overall_ex_dayoffset)}"
    )
    run.log(f"=== OOF on disrupted airport-days ({int(oof['disrupted'].sum()):,} rows) {metrics.summarise(overall_disrupted)}")
    overall_bulk = cleaning.clean_metrics(oof["y_true"], oof["y_pred"], oof["y_true"].to_numpy(dtype="float64") >= metrics.BULK_MAX_SEC)
    run.log(f"=== OOF bulk (target below {metrics.BULK_MAX_SEC:,.0f} s) {metrics.summarise(overall_bulk)}")
    oof["nm_unmatched"] = metrics.nm_unmatched_rows(mvt, oof.index)
    unmatched = oof["nm_unmatched"].to_numpy(dtype=bool)
    below = oof["y_true"].to_numpy(dtype="float64") < metrics.BULK_MAX_SEC
    overall_nm_unmatched = cleaning.clean_metrics(oof["y_true"], oof["y_pred"], ~unmatched)
    overall_bulk_nm_unmatched = cleaning.clean_metrics(oof["y_true"], oof["y_pred"], ~(unmatched & below))
    run.log(f"=== OOF without an NM match ({int(unmatched.sum()):,} rows) {metrics.summarise(overall_nm_unmatched)}")

    apt = mvt.loc[oof.index, "apt"]
    by_airport = metrics.breakdown_table(oof["y_true"], oof["y_pred"], apt)
    unflagged = oof.loc[~oof["flagged"]]
    clean_by_airport = metrics.breakdown_table(unflagged["y_true"], unflagged["y_pred"], apt.reindex(unflagged.index))
    by_airport["rmse_clean"] = clean_by_airport["rmse"].reindex(by_airport.index)
    by_airport["n_clean"] = clean_by_airport["n"].reindex(by_airport.index).fillna(0).astype(int)
    run.write_csv("metrics_by_airport.csv", by_airport, index=True)
    run.log("\n" + by_airport[["rmse", "mae", "bias", "n"]].round(1).to_string())

    test = None
    test_files = [d / "test_predictions.parquet" for d in dirs]
    if all(t.exists() for t in test_files):
        test = average_predictions([pd.read_parquet(t) for t in test_files], weights=weights)
        if lobt_bound:
            test, bound_info["rank"] = postprocess.apply_lobt_bound(test, mvt)
            run.log(f"    LOBT join window on the ranking rows: {bound_info['rank']}")
        run.write_parquet("test_predictions.parquet", test)
        run.log(f"ranking predictions: {len(test):,} rows, median {test['y_pred'].median():,.0f}s (mean over {len(dirs)} runs)")
    else:
        run.log(
            "no ranking predictions: not every source has test_predictions.parquet "
            "(train without --no-test, or --seeds with --keep-test)"
        )

    # A source's ranking predictions are the refit's when it refitted: refit.json is
    # written when the refit starts and test_predictions.parquet only after it ends.
    refitted = [(d / "refit.json").exists() and (d / "test_predictions.parquet").exists() for d in dirs]
    refit_info = {"blend": True, "sources": run_ids} if test is not None and all(refitted) else None
    first = next((m for m in source_metrics if m), {})
    run.log_metrics(
        {
            "overall": overall,
            "overall_clean": overall_clean,
            "overall_ex_dayoffset": overall_ex_dayoffset,
            "n_dayoffset_val": int(oof["dayoffset"].sum()),
            "overall_disrupted": overall_disrupted,
            "n_disrupted_val": int(oof["disrupted"].sum()),
            "overall_bulk": overall_bulk,
            "overall_nm_unmatched": overall_nm_unmatched,
            "overall_bulk_nm_unmatched": overall_bulk_nm_unmatched,
            "n_flagged_fit": first.get("n_flagged_fit", 0),
            "n_flagged_val": int(oof["flagged"].sum()),
            "clean": cfg.data.clean.summary(),
            "refit": refit_info,
            "blend": {
                "runs": run_ids,
                "weights": None if weights is None else [float(x) for x in weights],
                "lobt_bound": bound_info or None,
                "rmse_sources": [(m.get("overall") or {}).get("rmse") for m in source_metrics],
            },
            "folds": [],
            "by_airport": by_airport.to_dict(orient="index"),
            "n_features": first.get("n_features"),
            "n_blocks": first.get("n_blocks"),
            "n_train": first.get("n_train"),
            "n_val": int(len(oof)),
            "primary_metric": cfg.evaluation.primary_metric,
            "primary_value": overall.get(cfg.evaluation.primary_metric),
        }
    )
    run.finish()
    return run
