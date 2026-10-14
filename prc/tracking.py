"""Run artifacts: the only module that writes to `runs/`.

Everything a run produces goes through one `Run` object. That is what keeps the
on-disk layout consistent across dozens of experiments, and it is the single
seam where an MLflow or W&B backend would attach -- subclass `Run`, have the
`log_*` methods also call the tracker, and no other module changes.

Two append-only stores live at `runs/` level:

* `index.csv`          -- one flattened row per run. Rebuildable by rescanning
                          `runs/*/metrics.json`, so deleting it is always safe.
* `feature_effects.csv` -- tidy per-block usefulness measurements, accumulated
                          across every run ever made. This is the file that
                          answers "was the feature I wrote last week worth it".
"""

from __future__ import annotations

import getpass
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from prc import paths
from prc.config import ExperimentConfig

FEATURE_EFFECT_COLUMNS = [
    "run_id",
    "timestamp",
    "base_run_id",
    "block",
    "columns",
    "method",
    "metric",
    "value",
    "std",
    "n_seeds",
    "config_hash",
    "val_scheme",
]


def make_run_id(cfg: ExperimentConfig, seed: int | None = None) -> str:
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    suffix = f"__s{seed}" if seed is not None else ""
    return f"{stamp}__{cfg.name}__{cfg.hash()}{suffix}"


def _env_info() -> dict:
    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=paths.ROOT, capture_output=True, text=True, timeout=5
            ).stdout.strip()
        except Exception:  # pragma: no cover - git absent or not a repo
            return ""

    versions = {}
    for mod in ("pandas", "numpy", "pyarrow", "catboost", "sklearn", "lightgbm", "xgboost"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            pass

    return {
        "git_sha": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "host": socket.gethostname(),
        "user": getpass.getuser(),
        "packages": versions,
        # The data tree the run read (PRC_DATA_DIR or the repo's data/): `prc predict` refuses a different tree
        # under the same data.version, because the feature cache is partitioned by data.version only.
        "data_dir": str(paths.DATA),
        "prc_data_dir": os.environ.get("PRC_DATA_DIR"),
    }


class Run:
    """One experiment run and its artifact directory."""

    def __init__(self, cfg: ExperimentConfig, run_id: str | None = None, seed: int | None = None):
        paths.ensure_dirs()
        self.cfg = cfg
        self.seed = cfg.seed if seed is None else seed
        self.run_id = run_id or make_run_id(cfg, seed if seed is not None else None)
        # Ids carry the second they were made in: a second run of the same config within that second (two quick
        # `prc predict` calls) would write into the first one's directory. A generated id waits for a free second.
        for _ in range(5):
            if run_id is not None or not paths.run_dir(self.run_id).exists():
                break
            time.sleep(1.0 - (time.time() % 1.0) + 0.01)
            self.run_id = make_run_id(cfg, seed if seed is not None else None)
        self.dir = paths.run_dir(self.run_id)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "model").mkdir(exist_ok=True)
        self._t0 = time.perf_counter()
        self._log_lines: list[str] = []

        self.write_json("config.resolved.json", cfg.to_dict() | {"seed": self.seed})
        self.write_json("env.json", _env_info())

    # --- primitives ---------------------------------------------------------
    def path(self, name: str) -> Path:
        return self.dir / name

    def write_json(self, name: str, payload: dict) -> None:
        self.path(name).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def write_csv(self, name: str, frame: pd.DataFrame, index: bool = False) -> None:
        frame.to_csv(self.path(name), index=index)

    def write_parquet(self, name: str, frame: pd.DataFrame, index: bool = True) -> None:
        frame.to_parquet(self.path(name), index=index)

    def copy_in(self, source: Path, name: str | None = None) -> None:
        """Copy another run's artifact into this run (`prc predict` carries its source's holdout records)."""
        shutil.copyfile(source, self.path(name or Path(source).name))

    def log(self, message: str) -> None:
        print(message)
        self._log_lines.append(message)

    # --- run-level artifacts ------------------------------------------------
    def log_metrics(self, metrics: dict) -> None:
        payload = dict(metrics)
        payload["wall_seconds"] = round(time.perf_counter() - self._t0, 2)
        payload["run_id"] = self.run_id
        payload["name"] = self.cfg.name
        payload["config_hash"] = self.cfg.hash()
        payload["seed"] = self.seed
        self.write_json("metrics.json", payload)
        self._append_index(payload)

    def log_feature_list(self, block_columns: dict[str, list[str]], block_hashes: dict[str, str]) -> None:
        self.write_json(
            "feature_list.json",
            {"blocks": block_columns, "hashes": block_hashes, "n_features": sum(len(v) for v in block_columns.values())},
        )

    def log_feature_effects(self, rows: list[dict]) -> None:
        """Append per-block usefulness measurements to the global store."""
        if not rows:
            return
        frame = pd.DataFrame(rows)
        for col in FEATURE_EFFECT_COLUMNS:
            if col not in frame.columns:
                frame[col] = None
        frame = frame[FEATURE_EFFECT_COLUMNS]
        self.write_csv("feature_effects.csv", frame)
        _append_csv(paths.FEATURE_EFFECTS, frame)

    def finish(self) -> None:
        self.path("log.txt").write_text("\n".join(self._log_lines), encoding="utf-8")

    # --- global index -------------------------------------------------------
    def _append_index(self, metrics: dict) -> None:
        row = {
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "name": self.cfg.name,
            "config_hash": self.cfg.hash(),
            "seed": self.seed,
            "model": self.cfg.model.type,
            "per_airport": self.cfg.model.per_airport,
            # True only when the ranking predictions actually came from a refit
            # (the key is set and the run computed test predictions).
            "refit": bool(metrics.get("refit")),
            "val_scheme": self.cfg.validation.scheme,
            "target_transform": self.cfg.target.transform,
            "residual_of": self.cfg.target.residual_of,
            "sample_frac": self.cfg.data.sample_frac,
            "n_features": metrics.get("n_features"),
            "n_blocks": metrics.get("n_blocks"),
            "n_train": metrics.get("n_train"),
            "n_val": metrics.get("n_val"),
            "wall_seconds": metrics.get("wall_seconds"),
            "notes": self.cfg.notes,
            # A `prc predict` run repeats its source's holdout metrics; the leaderboard leaves these rows out by default.
            "predict_from": (metrics.get("predict") or {}).get("source_run"),
        }
        for key, value in (metrics.get("overall") or {}).items():
            row[key] = value
        # Suffixed, so the `overall` splat above and `rmse` itself stay untouched.
        clean = metrics.get("overall_clean") or {}
        for key in ("rmse", "mae", "n"):
            row[f"{key}_clean"] = clean.get(key)
        row["n_flagged_fit"] = metrics.get("n_flagged_fit", 0)
        row["n_flagged_val"] = metrics.get("n_flagged_val", 0)
        row["clean"] = metrics.get("clean", "")
        row["rmse_ex_dayoffset"] = (metrics.get("overall_ex_dayoffset") or {}).get("rmse")
        row["n_dayoffset_val"] = metrics.get("n_dayoffset_val")
        row["rmse_disrupted"] = (metrics.get("overall_disrupted") or {}).get("rmse")
        row["n_disrupted_val"] = metrics.get("n_disrupted_val")
        row["rmse_bulk"] = (metrics.get("overall_bulk") or {}).get("rmse")
        row["rmse_nm_unmatched"] = (metrics.get("overall_nm_unmatched") or {}).get("rmse")
        hedging = metrics.get("hedging") or {}
        row["hedge_rank"] = hedging.get("rank_refit") if hedging.get("rank_refit") is not None else hedging.get("rank_folds")
        row["hedge_holdout"] = hedging.get("holdout")
        for airport, stats in (metrics.get("by_airport") or {}).items():
            row[f"rmse_{airport}"] = stats.get("rmse")
        _append_csv(paths.RUN_INDEX, pd.DataFrame([row]))


LOCK_TIMEOUT_S = 60.0
LOCK_STALE_S = 300.0


class _FileLock:
    """``<file>.lock`` created exclusively: two runs finishing together (a GPU chain and `prc predict`) take turns at
    the read-modify-write of an index instead of one dropping the other's row. A lock older than `LOCK_STALE_S` is a
    crashed writer's and is taken over."""

    def __init__(self, path: Path):
        self.lock = path.with_name(path.name + ".lock")

    def __enter__(self):
        deadline = time.monotonic() + LOCK_TIMEOUT_S
        while True:
            try:
                fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode("ascii"))
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    if time.time() - self.lock.stat().st_mtime > LOCK_STALE_S:
                        self.lock.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{self.lock} is held by another writer (delete it if no run is finishing)")
                time.sleep(0.1)

    def __exit__(self, *exc):
        self.lock.unlink(missing_ok=True)
        return False


def _append_csv(path: Path, frame: pd.DataFrame) -> None:
    """Append rows, widening the header if new columns appeared since last time. Under a lock, written to a temporary
    file and moved into place, so a reader never sees half a file and concurrent writers do not lose rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _FileLock(path):
        combined = pd.concat([pd.read_csv(path), frame], ignore_index=True) if path.exists() else frame
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        combined.to_csv(tmp, index=False)
        for attempt in range(20):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:  # the file is open elsewhere (Windows): try again shortly
                if attempt == 19:
                    tmp.unlink(missing_ok=True)
                    raise
                time.sleep(0.25)


def rebuild_index() -> pd.DataFrame:
    """Regenerate `runs/index.csv` by rescanning every run directory."""
    rows = []
    for metrics_file in sorted(paths.RUNS.glob("*/metrics.json")):
        payload = json.loads(metrics_file.read_text(encoding="utf-8"))
        config_file = metrics_file.parent / "config.resolved.json"
        cfg = json.loads(config_file.read_text(encoding="utf-8")) if config_file.exists() else {}
        row = {
            "run_id": payload.get("run_id", metrics_file.parent.name),
            "name": payload.get("name"),
            "config_hash": payload.get("config_hash"),
            "seed": payload.get("seed"),
            "model": cfg.get("model", {}).get("type"),
            "per_airport": cfg.get("model", {}).get("per_airport"),
            "refit": bool(payload.get("refit")),
            "val_scheme": cfg.get("validation", {}).get("scheme"),
            "n_features": payload.get("n_features"),
            "n_blocks": payload.get("n_blocks"),
            "n_train": payload.get("n_train"),
            "n_val": payload.get("n_val"),
            "wall_seconds": payload.get("wall_seconds"),
            "predict_from": (payload.get("predict") or {}).get("source_run"),
        }
        row |= payload.get("overall") or {}
        # Runs from before cleaning existed had no rules, so clean == overall by definition.
        clean = payload.get("overall_clean") or payload.get("overall") or {}
        for key in ("rmse", "mae", "n"):
            row[f"{key}_clean"] = clean.get(key)
        row["n_flagged_fit"] = payload.get("n_flagged_fit", 0)
        row["n_flagged_val"] = payload.get("n_flagged_val", 0)
        row["clean"] = payload.get("clean", "")
        row["rmse_ex_dayoffset"] = (payload.get("overall_ex_dayoffset") or {}).get("rmse")
        row["n_dayoffset_val"] = payload.get("n_dayoffset_val")
        row["rmse_disrupted"] = (payload.get("overall_disrupted") or {}).get("rmse")
        row["n_disrupted_val"] = payload.get("n_disrupted_val")
        row["rmse_bulk"] = (payload.get("overall_bulk") or {}).get("rmse")
        row["rmse_nm_unmatched"] = (payload.get("overall_nm_unmatched") or {}).get("rmse")
        hedging = payload.get("hedging") or {}
        row["hedge_rank"] = hedging.get("rank_refit") if hedging.get("rank_refit") is not None else hedging.get("rank_folds")
        row["hedge_holdout"] = hedging.get("holdout")
        for airport, stats in (payload.get("by_airport") or {}).items():
            row[f"rmse_{airport}"] = stats.get("rmse")
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.to_csv(paths.RUN_INDEX, index=False)
    return frame


def load_index() -> pd.DataFrame:
    if not paths.RUN_INDEX.exists():
        return rebuild_index()
    return pd.read_csv(paths.RUN_INDEX)


def load_feature_effects() -> pd.DataFrame:
    if not paths.FEATURE_EFFECTS.exists():
        return pd.DataFrame(columns=FEATURE_EFFECT_COLUMNS)
    return pd.read_csv(paths.FEATURE_EFFECTS)
