"""Reading back what all those runs measured."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from prc import paths, tracking

RUN_COLUMNS = [
    "run_id",
    "name",
    "model",
    "per_airport",
    "refit",
    "val_scheme",
    "target_transform",
    "n_features",
    "n_blocks",
    "rmse",
    "board",
    "mae",
    "medae",
    "bias",
    "rmse_clean",
    "rmse_ex_dayoffset",
    "rmse_disrupted",
    "n_disrupted_val",
    "rmse_bulk",
    "hedge_rank",
    "mae_clean",
    "n_clean",
    "clean",
    "wall_seconds",
    "sample_frac",
    "seed",
]

CLEAN_COLUMNS = {"rmse_clean", "mae_clean", "n_clean", "rmse_ex_dayoffset", "rmse_disrupted", "rmse_bulk", "rmse_nm_unmatched"}
"""Metric columns that share the ``rmse_`` prefix with the per-airport ones but are not airports."""


def board_scores(path: Path | None = None) -> pd.Series:
    """Leaderboard score per run id, from the upload log `submissions/uploads.csv` (written by a separate upload tool).

    A run uploaded more than once keeps its latest fetched score. Rows without a
    run id (uploads from another machine) cannot be joined and are skipped.
    """
    path = paths.SUBMISSIONS / "uploads.csv" if path is None else path
    empty = pd.Series(dtype="float64", name="board")
    if not path.exists():
        return empty
    log = pd.read_csv(path)
    if not {"run_id", "score"} <= set(log.columns):
        return empty
    scored = log.dropna(subset=["run_id", "score"])
    scored = scored[scored["run_id"].astype(str).str.len() > 0]
    if scored.empty:
        return empty
    return scored.groupby("run_id", sort=False)["score"].last().astype("float64").rename("board")


def runs(sort_by: str = "rmse", limit: int = 30, name: str | None = None) -> pd.DataFrame:
    frame = tracking.load_index()
    if frame.empty:
        return frame
    board = board_scores()
    if not board.empty:
        frame = frame.merge(board, left_on="run_id", right_index=True, how="left")
    if name:
        frame = frame[frame["name"].astype("string").str.contains(name, na=False)]
    elif "predict_from" in frame.columns:
        # `prc predict` runs repeat their source's holdout numbers: listed only when asked for by name.
        frame = frame[frame["predict_from"].isna()]
    cols = [c for c in RUN_COLUMNS if c in frame.columns]
    if sort_by in frame.columns:
        frame = frame.sort_values(sort_by, ascending=sort_by not in {"r2"})
    return frame[cols].head(limit).reset_index(drop=True)


def by_airport(limit: int = 15) -> pd.DataFrame:
    frame = tracking.load_index()
    cols = ["run_id", "name"] + sorted(c for c in frame.columns if c.startswith("rmse_") and c not in CLEAN_COLUMNS)
    if len(cols) <= 2:
        return pd.DataFrame()
    return frame.sort_values("rmse").head(limit)[cols].reset_index(drop=True)


def features(val_scheme: str | None = None) -> tuple[pd.DataFrame, float | None]:
    """One row per block, aggregating every measurement ever recorded for it.

    Returned alongside the seed-noise floor, because a block whose measured
    effect is smaller than the run-to-run spread has not been shown to do
    anything -- the two numbers are only meaningful together.
    """
    effects = tracking.load_feature_effects()
    if effects.empty:
        return pd.DataFrame(), None
    if val_scheme:
        effects = effects[effects["val_scheme"] == val_scheme]

    noise_rows = effects[effects["method"] == "seed_noise"]
    noise = float(noise_rows["value"].median()) if not noise_rows.empty else None
    effects = effects[effects["block"] != "__seed_noise__"]
    if effects.empty:
        return pd.DataFrame(), noise

    pivots = []
    for method, group in effects.groupby("method"):
        agg = group.groupby("block")["value"].agg(["median", "std", "count"])
        agg.columns = [f"{method}_median", f"{method}_std", f"{method}_n"]
        pivots.append(agg)

    table = pd.concat(pivots, axis=1)
    table["last_seen"] = effects.groupby("block")["timestamp"].max()

    order = next(
        (c for c in ("ablation_leave_one_out_median", "permutation_median", "gain_median") if c in table.columns),
        table.columns[0],
    )
    return table.sort_values(order, ascending=False).round(4).reset_index(), noise


def format_features(table: pd.DataFrame, noise: float | None) -> str:
    if table.empty:
        return "no feature effects recorded yet -- run `prc train` or `prc ablate` first"
    header = (
        f"seed-noise floor: +-{noise:.1f} RMSE  "
        "(effects smaller than this are not evidence)\n"
        if noise is not None
        else "seed-noise floor: not measured -- run `prc train --seeds 3` to establish it\n"
    )
    return header + "\n" + table.to_string(index=False)
