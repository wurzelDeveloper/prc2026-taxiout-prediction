"""`prc predict`: a finished run's refit model applied to the ranking rows of the current data tree, without retraining.

The final submission is scored on January, February, June and July 2026 (findings 10.33), released after training
ended: the board's January and July rows with their times jittered, and two months no model has seen, February and
June. The trained runs stay as they are; the released ranking file goes into a data tree of its own
(``PRC_DATA_DIR``, prepared by `lab/final_inputs.py`) with its own ``data.version``, and this command, for one run:

1. reads the run's ``config.resolved.json`` and ``feature_list.json`` (``-o`` may change ``data.version`` and
   ``notes`` only: anything else would not be the model that was trained);
2. resolves the same feature blocks in the current data tree and builds them over every departure, the
   fold-dependent ones (``uses_target``) fitted on the refit's own rows -- every training row minus what the cleaning
   rules drop from a fit -- exactly as `prc.train._refit_on_all` did (`prc.train.refit_index`, `refit_features`);
3. puts the whole-frame counts of the ranking rows back on the scale the model was fitted on (`prc.frame_shift`: the
   frame now holds other ranking months, and counts over it grow) and measures what else the new frame moved: every
   block's values on the training rows against the source run's own cached blocks (the parity report);
4. loads the refit model the run saved (``model/refit.cbm`` or ``model/refit__<APT>.cbm``, and for the regime mixture
   ``model/refit__classifier*.cbm``, the calibration curves and ``model/refit__mixture.json``) with
   `prc.models.load_model`, hands it the matrix in the column order it was fitted on (`feature_contract`), and replays
   the fit-time bookkeeping on the training rows (`restore_fit_state`): runs saved before the state files existed get
   the per-airport fallback and the mixture's anchor fallback recomputed, newer runs have them checked;
5. predicts the ranking departures with the run's own post-processing (`prc.train.rank_predictions`: the mixture,
   the inverse target transform, the residual base; ``target.clip`` shapes only the fitting target, so nothing is
   clipped, as in the run itself);
6. writes a new run directory (`prc.tracking.Run`): ``test_predictions.parquet`` (what `prc blend` and `prc submit`
   read), the mixture's ``refit_detail.parquet``, and ``predict.json`` -- the source run, the data tree, both
   ``data.version`` values, the digests of the movement frame, the target, the ranking file, the template and every
   external input, each block's hash, code hash and status, the model files' digests, the state report, the count
   rescale, the parity report, the unseen categorical levels and, where the ranking rows are the source's, the
   reproduction check against its ``test_predictions``. ``oof_predictions.parquet``, ``refit.json`` and the holdout
   metrics are the source run's, copied: they describe the model, which has not changed, and they let `prc blend` and
   the leaderboard treat the run like its source (the index marks it with ``predict_from``).

Refusals, all before anything is written to ``runs/``:

* no refit model (``model.refit_on_all`` was off, the run was ``--no-test``, or the files are gone);
* a model type other than CatBoost (`prc.models.LOADABLE`);
* the wrong data tree: a ``data.version`` other than the source's must name its tree with ``--tree`` (the directory
  ``PRC_DATA_DIR`` points at), so a session without ``PRC_DATA_DIR`` cannot fill that version's cache from the
  repository's ``data/``; another data tree under the source's own ``data.version`` (the stale-cache trap: the cache is
  shared and keyed by ``data.version`` only); a version whose cache was built from another tree or another frame
  (`prc.features.build.check_tree_stamp`);
* a frame older than its inputs: ``interim/prepare.json`` records the files that built the movement frame, and a
  ranking file installed since then (without ``prepare --force``) is refused;
* other training targets than the source tree's (the fold-dependent blocks would be fitted on other rows);
* feature-code drift: a block whose hash under the source's ``data.version``, with the input files of the source's
  own data tree, differs from the one the run recorded -- its code (or that tree's input file) changed since the run,
  so the model would see features it was not trained on. ``--allow-drift`` predicts anyway (``predict.json`` says so);
* categorical levels the model never saw on far more ranking rows than on the source's own ranking rows (a label
  spelled differently in the release turns a whole column unseen); ``--allow-unseen`` predicts anyway.

Categorical columns reach CatBoost as strings, exactly as in training (`prc.models.base.prepare_matrix`, encoding
``string``); the `CategoryUniverse` levels play no part. A level the model never saw -- a stand or a destination new
in 2026 -- is valid input: CatBoost treats every unseen value alike, with the prior-only statistics of its CTR
features and none of its one-hot indicators set. It is not mapped to ``__NA__``, which is a level of its own wherever
training had missing values. ``predict.json`` counts the unseen levels per column and airport against the refit's fit
rows (of the row's own airport for a per-airport model), next to the same count on the source's own ranking rows.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from prc import cleaning, data, frame_shift, paths, tracking
from prc.config import ExperimentConfig, apply_overrides, from_dict
from prc.features import resolve
from prc.features.build import (
    CacheTreeMismatch,
    _file_digest,
    block_hash,
    build_matrix,
    check_tree_stamp,
    code_hash,
    external_path,
)
from prc.models import (
    LOADABLE,
    CategoryUniverse,
    feature_contract,
    load_model,
    restore_fit_state,
    set_predict_threads,
    state_columns,
)
from prc.models.mixture import DETAIL_COLUMNS, classifier_file, hedging_report
from prc.models.per_airport import STATE_SUFFIX as PER_AIRPORT_STATE
from prc.train import _prepare_targets, _training_index, rank_predictions, refit_features, refit_index

ALLOWED_OVERRIDES = ("data.version", "notes")
"""What ``-o`` may change: everything else defines the model that was trained."""
REFIT_STEM = "refit"
REPRODUCTION_TOL = 1e-6
"""Seconds. On the source's own ranking rows the predictions must match its ``test_predictions`` to this."""


class PredictRefused(RuntimeError):
    """`prc predict` declined before writing anything; the message says why and what to do."""


@dataclass
class SourceRun:
    dir: Path
    run_id: str
    payload: dict
    cfg: ExperimentConfig
    seed: int
    feature_list: dict
    env: dict
    metrics: dict


def resolve_run_dir(run: str | Path) -> Path:
    """A run id under ``runs/`` or a run directory path."""
    candidate = paths.run_dir(str(run))
    if candidate.is_dir():
        return candidate
    direct = Path(run)
    if direct.is_dir() and (direct / "config.resolved.json").exists():
        return direct.resolve()
    raise FileNotFoundError(f"no run directory '{run}' under {paths.RUNS} (nor as a path)")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_source(run: str | Path) -> SourceRun:
    run_dir = resolve_run_dir(run)
    for needed in ("config.resolved.json", "feature_list.json"):
        if not (run_dir / needed).exists():
            raise PredictRefused(f"{run_dir.name}: no {needed} (not a finished training run)")
    payload = _read_json(run_dir / "config.resolved.json")
    cfg = from_dict(payload)
    return SourceRun(
        dir=run_dir,
        run_id=run_dir.name,
        payload=payload,
        cfg=cfg,
        seed=int(payload.get("seed", cfg.seed)),
        feature_list=_read_json(run_dir / "feature_list.json"),
        env=_read_json(run_dir / "env.json"),
        metrics=_read_json(run_dir / "metrics.json"),
    )


def predict_config(src: SourceRun, overrides: list[str] | None, *, name: str | None = None) -> ExperimentConfig:
    """The source config with the allowed overrides, renamed ``predict_<source name>`` (or ``name``)."""
    bad = [o for o in overrides or [] if "=" not in o or o.split("=", 1)[0].strip() not in ALLOWED_OVERRIDES]
    if bad:
        raise PredictRefused(
            f"prc predict applies the run's own configuration; only {list(ALLOWED_OVERRIDES)} may be overridden "
            f"(as key=value), got {bad}"
        )
    payload = apply_overrides(json.loads(json.dumps(src.payload)), overrides)
    version = payload["data"].get("version")
    if version is None or not str(version).strip():
        raise PredictRefused(f"data.version must be a non-empty name, got {version!r}")
    payload["data"]["version"] = str(version).strip()  # `-o data.version=2026` parses as an int; the cache wants a name
    cfg = from_dict(payload)
    notes = cfg.notes if any(o.startswith("notes=") for o in overrides or []) else (
        f"prc predict: the refit model of {src.run_id} on the ranking rows of {paths.DATA} (data.version {cfg.data.version})"
    )
    return cfg.replace(name=name or f"predict_{src.cfg.name}", notes=notes)


def source_data_tree(src: SourceRun) -> tuple[Path, bool]:
    """The data tree the source run read, and whether the run recorded it (`prc.tracking` writes ``data_dir`` into
    ``env.json``). Older runs did not: they are taken to have read the ``data/`` beside their ``runs/`` directory,
    the repository default."""
    recorded = src.env.get("data_dir")
    if recorded:
        return Path(recorded), True
    return src.dir.parent.parent / "data", False


def check_data_tree(src: SourceRun, cfg: ExperimentConfig, tree: str | Path | None = None) -> dict:
    """Refuse the wrong data tree (see the module docstring): ``--tree`` against the tree this process reads, a new
    ``data.version`` without ``--tree``, another tree under the source's own version, and a version whose cache was
    built from another tree or frame."""
    source_tree, recorded = source_data_tree(src)
    same = paths.same_dir(source_tree, paths.DATA)
    env_dir = os.environ.get("PRC_DATA_DIR")
    info = {
        "data_tree": str(paths.DATA),
        "prc_data_dir": env_dir,
        "source_data_tree": str(source_tree),
        "source_data_tree_recorded": recorded,
        "same_tree": same,
        "data_version": cfg.data.version,
        "source_data_version": src.cfg.data.version,
    }
    if tree is not None and not paths.same_dir(tree, paths.DATA):
        raise PredictRefused(
            f"--tree {tree} is not the data tree this process reads: {paths.DATA} "
            f"({'PRC_DATA_DIR=' + env_dir if env_dir else 'PRC_DATA_DIR is not set, so the repository data/'}). "
            f"Set PRC_DATA_DIR to {tree} in this session first (the runbook script does)."
        )
    if cfg.data.version != src.cfg.data.version and tree is None:
        raise PredictRefused(
            f"data.version '{cfg.data.version}' is not the run's ('{src.cfg.data.version}'): name the data tree it is "
            f"for with --tree <dir> (the directory PRC_DATA_DIR points at), so that a session without PRC_DATA_DIR "
            f"cannot fill this version's cache from {paths.DATA}"
        )
    if not same and cfg.data.version == src.cfg.data.version:
        raise PredictRefused(
            f"the data tree {paths.DATA} is not the one {src.run_id} was trained on ({source_tree}"
            f"{'' if recorded else ', assumed: the run predates the record'}), but data.version is the same "
            f"('{cfg.data.version}'). The feature cache is shared by every data tree and keyed by data.version only, "
            f"so this tree would be served the other tree's cached blocks. Give it its own version: "
            f"-o data.version=<new> --tree {paths.DATA}"
        )
    try:
        info["tree_stamp"] = check_tree_stamp(cfg.data.version, write=False)
    except CacheTreeMismatch as exc:
        raise PredictRefused(str(exc)) from None
    return info


def check_prepared() -> dict:
    """The movement frame must be the one its current input files build: ``prepare.json`` against the files' sha1."""
    record = data.prepare_record()
    if record is None:
        return {"record": None, "note": "the frame predates prepare.json: whether its inputs changed is unknown"}
    if why := data.unfinished(record):
        raise PredictRefused(f"{why}. Run `prc prepare --force` until it finishes.")
    changed = data.changed_inputs(record)
    if changed:
        raise PredictRefused(
            f"the competition files changed since {paths.MOVEMENTS_FILE} was built ({changed}): the frame still holds "
            "the earlier rows. Run `prc prepare --force` and give the new frame its own data.version."
        )
    return {"record": str(paths.PREPARE_RECORD), "frame_id": record.get("frame_id"), "created_utc": record.get("created_utc")}


def check_training_target(src_tree: Path, same_tree: bool, y: pd.Series) -> dict:
    """The training targets here must be the source tree's: the fold-dependent blocks are refitted on them and the
    model's fit-time state is replayed on them."""
    if same_tree:
        return {"checked": True, "how": "same data tree"}
    file = src_tree / paths.TARGET_FILE.relative_to(paths.DATA)
    if not file.exists():
        return {"checked": False, "reason": f"{file} not found"}
    old = pd.read_parquet(file)[data.TARGET_COL].astype("float64")
    same = old.index.equals(y.index) and np.array_equal(old.to_numpy(), y.to_numpy(dtype="float64"), equal_nan=True)
    if not same:
        n_ids = int((~y.index.isin(old.index)).sum()) + int((~old.index.isin(y.index)).sum())
        common = old.index.intersection(y.index)
        n_values = int((old.reindex(common).to_numpy() != y.reindex(common).to_numpy()).sum())
        raise PredictRefused(
            f"the training targets of {paths.DATA} are not those of {src_tree}, the tree the run was trained on "
            f"({n_ids:,} ids differ, {n_values:,} values differ): the fold-dependent blocks would be fitted on other "
            "rows. Create the tree from the run's own data tree (final_inputs.py init --source <that tree>)."
        )
    return {"checked": True, "rows": int(len(y)), "source_file": str(file)}


def refit_model_files(src: SourceRun) -> list[Path]:
    """The model files `prc predict` will load; refuses when the run has no refit model."""
    cfg = src.cfg
    if cfg.model.type not in LOADABLE:
        raise PredictRefused(
            f"{src.run_id} is a '{cfg.model.type}' run; prc predict loads {list(LOADABLE)} models only"
        )
    model_dir = src.dir / "model"
    stem = model_dir / REFIT_STEM
    if not cfg.model.refit_on_all:
        raise PredictRefused(
            f"{src.run_id} has no refit model: model.refit_on_all was off, so its ranking predictions are the fold "
            "models' mean. Train with model.refit_on_all: true (and without --no-test) to get one."
        )
    if cfg.model.per_airport:
        files = sorted(
            f for f in model_dir.glob(f"{REFIT_STEM}__*.cbm") if not f.name.startswith(f"{REFIT_STEM}__classifier")
        )
        state = model_dir / f"{REFIT_STEM}{PER_AIRPORT_STATE}"
        if state.exists():
            files = [stem.with_name(f"{REFIT_STEM}__{a}.cbm") for a in _read_json(state)["airports"]]
    else:
        files = [stem.with_suffix(".cbm")]
    if cfg.model.mixture.enabled:
        files += [f for f in (classifier_file(stem, i) for i in range(max(cfg.model.mixture.classifier_seeds, 1))) if f.exists()]
    missing = [f.name for f in files if not f.exists()]
    if not files or missing:
        started = (src.dir / "refit.json").exists()
        why = "the refit started (refit.json) but its model files are missing" if started else (
            "the run never refitted (trained with --no-test, or as a seed/ablation run)"
        )
        raise PredictRefused(f"{src.run_id} has no refit model: {why}; missing {missing or ['refit*.cbm']}")
    return files


def _has_external_ancestry(specs) -> dict[str, bool]:
    by_name = {s.name: s for s in specs}
    memo: dict[str, bool] = {}

    def visit(name: str) -> bool:
        if name not in memo:
            spec = by_name[name]
            memo[name] = bool(spec.external) or any(visit(d) for d in spec.depends_on if d in by_name)
        return memo[name]

    return {name: visit(name) for name in by_name}


def compare_blocks(specs, src: SourceRun, cfg: ExperimentConfig, source_tree: Path | None = None) -> dict[str, dict]:
    """Per block: the hash the source recorded, the current code's hash under the source's ``data.version`` (what the
    source's would be if nothing changed), the same with the external inputs read from the source's own data tree
    (``source_tree``), the hash under the predict ``data.version``, the code-only hash, and a status:

    ``same`` (the recorded hash, here), ``external`` (reads an input file, directly or through a parent: differs here,
    because this tree rebuilt its inputs, and equals the recorded hash with the source tree's files -- the code is
    unchanged), ``external_unverified`` (the source tree's input files are gone: not checkable), ``drift`` (the code
    changed, or the source tree's input file changed since the run), ``new`` (not in the source run) or ``absent``
    (in the source run, not resolved now)."""
    recorded = (src.feature_list or {}).get("hashes") or {}
    ext = _has_external_ancestry(specs)
    under_source: dict[str, str] = {}
    under_cfg: dict[str, str] = {}
    under_tree: dict[str, str] = {}
    unverifiable: dict[str, bool] = {}
    out: dict[str, dict] = {}
    for spec in specs:
        params = cfg.features.params.get(spec.name, {})
        under_source[spec.name] = block_hash(spec, src.cfg.data.version, params, under_source)
        under_cfg[spec.name] = block_hash(spec, cfg.data.version, params, under_cfg)
        if ext[spec.name] and source_tree is not None:
            under_tree[spec.name] = block_hash(spec, src.cfg.data.version, params, under_tree, external_root=source_tree)
        else:
            under_tree[spec.name] = under_source[spec.name]
        gone = [str(external_path(p, source_tree)) for p in spec.external
                if source_tree is not None and not external_path(p, source_tree).exists()]
        unverifiable[spec.name] = bool(gone) or any(unverifiable.get(d, False) for d in spec.depends_on)
        old = recorded.get(spec.name)
        if old is None:
            status = "new"
        elif old == under_source[spec.name]:
            status = "same"
        elif not ext[spec.name]:
            status = "drift"
        elif old == under_tree[spec.name]:
            status = "external"
        elif unverifiable[spec.name]:
            status = "external_unverified"
        else:
            status = "drift"
        out[spec.name] = {
            "status": status,
            "source_hash": old,
            "hash_under_source_version": under_source[spec.name],
            "hash": under_cfg[spec.name],
            "code_hash": code_hash(spec, params),
            "external_inputs": {p.name: _file_digest(p) for p in spec.external},
        }
        if ext[spec.name]:
            out[spec.name]["hash_with_source_inputs"] = under_tree[spec.name]
            if gone:
                out[spec.name]["source_inputs_missing"] = gone
    for name in sorted(set(recorded) - set(out)):
        out[name] = {"status": "absent", "source_hash": recorded[name]}
    return out


def _digests(files: dict[str, Path]) -> dict[str, str]:
    return {name: _file_digest(path) for name, path in files.items()}


def _reproduction(src: SourceRun, test: pd.Series) -> dict:
    """On the source's own ranking rows: how far the predictions are from its ``test_predictions``."""
    previous_file = src.dir / "test_predictions.parquet"
    if not previous_file.exists():
        return {"comparable": False, "reason": "the source run has no test_predictions.parquet"}
    previous = pd.read_parquet(previous_file)["y_pred"]
    if len(previous) != len(test) or not previous.index.isin(test.index).all():
        return {"comparable": False, "reason": "other ranking rows than the source run's"}
    diff = test.reindex(previous.index).to_numpy(dtype="float64") - previous.to_numpy(dtype="float64")
    return {
        "comparable": True,
        "n": int(len(diff)),
        "max_abs_diff": float(np.max(np.abs(diff))) if len(diff) else 0.0,
        "rms_diff": float(np.sqrt(np.mean(diff**2))) if len(diff) else 0.0,
        "within_tol": bool(np.all(np.abs(diff) <= REPRODUCTION_TOL)),
        "tol": REPRODUCTION_TOL,
    }


# --- the whole-frame counts ------------------------------------------------------------------------------------------

def _key_frame(by_name: dict, cols_by_block: dict[str, list[str]], read) -> pd.DataFrame | None:
    parts = []
    for block, cols in cols_by_block.items():
        spec = by_name.get(block)
        frame = read(spec, cols) if spec is not None else None
        if frame is None:
            return None
        parts.append(frame[cols])
    return pd.concat(parts, axis=1)


def _source_keys_from_frame(source: frame_shift.SourceFrame, cols_by_block: dict[str, list[str]]) -> pd.DataFrame | None:
    """The key columns over the source tree's movement frame, computed (the source's cached key blocks are gone)."""
    file = source.tree / paths.MOVEMENTS_FILE.relative_to(paths.DATA)
    if not file.exists():
        return None
    mvt_src = pd.read_parquet(file)
    specs = resolve(list(cols_by_block))
    fm = build_matrix(specs, mvt_src, data.dep_index(mvt_src), data_version="source_keys", use_cache=False, verbose=False)
    return fm.X[[c for cols in cols_by_block.values() for c in cols]]


def _rescaled_counts(by_name: dict, blocks: list[str], fm, cfg: ExperimentConfig, source: frame_shift.SourceFrame,
                     X: pd.DataFrame, rank_idx: pd.Index, train_ids: pd.Index):
    """`prc.frame_shift.rescale_counts` with both self-checks, remembered in the new version's cache (``_rescale/``)
    under the key and count blocks' hashes on both sides: the members of one blend share those blocks, so only the
    first one pays the string work over the two frames (half a minute on the real ones)."""
    cols_by_block = frame_shift.key_columns(blocks)
    involved = sorted(set(cols_by_block) | set(blocks))
    key = {"blocks": blocks, "source_version": source.data_version, "new_version": cfg.data.version,
           "source": {b: source.hashes.get(b) for b in involved}, "new": {b: fm.block_hashes.get(b) for b in involved}}
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    memo = paths.feature_cache_dir(cfg.data.version) / "_rescale"
    values_file, record_file = memo / f"{digest}.parquet", memo / f"{digest}.json"
    if values_file.exists() and record_file.exists():
        frame = pd.read_parquet(values_file)
        if frame.index.equals(rank_idx):
            record = json.loads(record_file.read_text(encoding="utf-8"))
            corrected = {c: frame[c].to_numpy(dtype=np.float32) for c in record["corrected"]}
            seen = {c: frame[f"seen__{c}"].to_numpy(dtype=bool) for c in record["corrected"]}
            return corrected, record | {"memo": values_file.name}, seen

    new_keys = _key_frame(by_name, cols_by_block,
                          lambda spec, cols: frame_shift.read_cached(spec, cfg.data.version, fm.block_hashes[spec.name], cols))
    if new_keys is None:
        raise RuntimeError("the key blocks just built are not in the cache")
    src_keys = _key_frame(by_name, cols_by_block, source.read)
    how = "source cache"
    if src_keys is None:
        src_keys = _source_keys_from_frame(source, cols_by_block)
        how = "source movement frame"
    if src_keys is None:
        raise PredictRefused(
            "cannot put the whole-frame counts back on the training-time scale: the source run's cached key blocks "
            f"are gone and its movement frame ({source.tree}) is not there. Pass --no-count-rescale to predict with "
            "this tree's counts (about 13 % higher on every row for the final set)."
        )
    # The key logic must still be the blocks': on the source frame (its cached values) and on this one (uncorrected).
    checks = {}
    src_values = [source.read(by_name[b]) for b in blocks]
    if all(v is not None for v in src_values):
        checks["source"] = frame_shift.check_counts(src_keys, pd.concat(src_values, axis=1), blocks)
    else:
        checks["source"] = {"checked": [], "mismatch": [], "note": "the source's cached count blocks are gone"}
    new_values = [frame_shift.read_cached(by_name[b], cfg.data.version, fm.block_hashes[b]) for b in blocks]
    checks["new"] = frame_shift.check_counts(new_keys, pd.concat(new_values, axis=1) if all(
        v is not None for v in new_values) else X, blocks)
    bad = {k: v["mismatch"] for k, v in checks.items() if v["mismatch"]}
    if bad:
        raise RuntimeError(
            f"prc.frame_shift no longer reproduces the count blocks {blocks} ({bad}): their code changed; update "
            "COUNT_COLUMNS / RATIO_COLUMNS / count_keys to match, or pass --no-count-rescale"
        )
    corrected, record, seen = frame_shift.rescale_counts(new_keys, src_keys, rank_idx, train_ids, blocks)
    record |= {"source_keys": how, "checks": checks, "corrected": list(corrected)}
    memo.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({**corrected, **{f"seen__{c}": s for c, s in seen.items()}}, index=rank_idx)
    frame.to_parquet(values_file)
    record_file.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return corrected, record, seen


def rescale_frame_counts(X: pd.DataFrame, specs, fm, cfg: ExperimentConfig, source: frame_shift.SourceFrame,
                         rank_idx: pd.Index, train_ids: pd.Index, run_log) -> dict:
    """`prc.frame_shift.rescale_counts` applied to the ranking rows of ``X`` (in place), with both self-checks."""
    by_name = {s.name: s for s in specs}
    blocks = [b for b in frame_shift.COUNT_BLOCKS if b in by_name]
    if not blocks:
        return {"applied": False, "reason": "the run has no count block"}
    corrected, record, seen = _rescaled_counts(by_name, blocks, fm, cfg, source, X, rank_idx, train_ids)
    how = record["source_keys"]
    pos = X.index.get_indexer(rank_idx)
    columns = {}
    for col, values in corrected.items():
        if col not in X.columns:
            continue
        before = X[col].to_numpy(dtype=np.float32, copy=True)
        after = before.copy()
        after[pos] = values
        columns[col] = frame_shift.column_summary(before[pos], after[pos], seen[col])
        X[col] = after
    record = {k: v for k, v in record.items() if k != "corrected"} | {"applied": True, "columns": columns}
    w = {a: (round(v, 3) if v is not None else None) for a, v in record["w"].items()}
    run_log(f"    count rescale ({how}): w per airport {w}")
    for col, s in columns.items():
        infl = s.get("inflation_seen", {}).get("mean")
        run_log(f"      {col}: mean {s['mean_before']:,.4g} -> {s['mean_after']:,.4g} on the ranking rows"
                + (f" (uncorrected x{infl:.3f} on keys the source had)" if infl else ""))
    return record


# --- the command -----------------------------------------------------------------------------------------------------

def predict_run(
    run: str | Path,
    overrides: list[str] | None = None,
    *,
    name: str | None = None,
    threads: int = -1,
    allow_drift: bool = False,
    features_only: bool = False,
    tree: str | Path | None = None,
    allow_unseen: bool = False,
    count_rescale: bool = True,
    parity: bool = True,
) -> tracking.Run | None:
    """`prc predict`; returns the new run (None with ``features_only``, which builds the matrix, measures the parity
    and stops)."""
    src = load_source(run)
    cfg = predict_config(src, overrides, name=name)
    tree_info = check_data_tree(src, cfg, tree)
    prepared = check_prepared()
    model_files = refit_model_files(src)
    source_tree = Path(tree_info["source_data_tree"])
    print(f"=== prc predict: {src.run_id}")
    print(f"    data tree {tree_info['data_tree']} (data.version {cfg.data.version}); source tree {source_tree} "
          f"(data.version {src.cfg.data.version})")

    specs = resolve(cfg.features.include, cfg.features.exclude)
    blocks = compare_blocks(specs, src, cfg, source_tree)
    drift = sorted(b for b, info in blocks.items() if info["status"] == "drift")
    if drift and not allow_drift:
        raise PredictRefused(
            f"feature code changed since {src.run_id} was trained: {drift} hash differently under its data.version "
            "(blocks with input files: with the input files of the run's own data tree, so the code or that tree's "
            "file changed), and the model would see features it was not trained on. Check out the run's commit "
            f"(env.json git_sha {src.env.get('git_sha', '?')}) or pass --allow-drift to predict anyway."
        )
    counts = pd.Series([b["status"] for b in blocks.values()]).value_counts().to_dict()
    print(f"    blocks: {counts}" + (f"; DRIFT allowed: {drift}" if drift else ""))

    mvt = data.load_movements()
    y = data.load_target()
    training = check_training_target(source_tree, tree_info["same_tree"], y)
    dep_idx = data.dep_index(mvt)
    rank_idx = data.dep_index(mvt, "rank")
    if not len(rank_idx):
        raise PredictRefused(f"no ranking departures in {paths.MOVEMENTS_FILE}")
    train_idx = _training_index(cfg, mvt, y, src.seed)
    flagged = cleaning.flag_rows(mvt, y, cfg.data.clean)
    fit_idx = refit_index(cfg, train_idx, flagged)
    print(f"    refit fit rows {len(fit_idx):,} (training rows {len(train_idx):,}); ranking departures {len(rank_idx):,}")

    fm = refit_features(cfg, specs, mvt, y, dep_idx, fit_idx, src.seed)
    source = frame_shift.SourceFrame(
        data_version=src.cfg.data.version,
        hashes=(src.feature_list or {}).get("hashes") or {},
        refit_fold=f"refit__{src.cfg.split_hash(src.seed)}",
        tree=source_tree,
    )
    parity_info = None
    if parity:
        parity_info = frame_shift.parity_report(
            specs, fm.block_columns, fm.block_hashes, fm.X, y.index, source,
            cfg.data.version, f"refit__{cfg.split_hash(src.seed)}",
        )
    if features_only:
        print(f"    features only: matrix {fm.X.shape[0]:,} rows x {fm.X.shape[1]} columns under data.version {cfg.data.version}")
        return None

    universe = CategoryUniverse(fm.X, fm.cat_cols)
    model = load_model(cfg, universe, src.dir / "model" / REFIT_STEM)
    names, model_cats = feature_contract(model)
    missing = [c for c in names if c not in fm.X.columns]
    if missing:
        raise PredictRefused(f"the model needs columns the blocks no longer produce: {missing}")
    matrix_cats = [c for c in fm.cat_cols if c in names]
    if sorted(matrix_cats) != sorted(model_cats):
        raise PredictRefused(
            f"categorical columns differ from the model's: the blocks declare {sorted(matrix_cats)}, "
            f"the model was fitted with {sorted(model_cats)}"
        )
    extra = [c for c in fm.X.columns if c not in names]
    # CatBoost binds features by position: the matrix goes in exactly in the order the model was fitted on (already
    # the case unless the blocks changed; then a reordered copy). Every member holds the one universe object, so
    # narrowing its categorical list here reaches them all.
    X = fm.X if list(fm.X.columns) == names else fm.X[names]
    universe.cat_features = [c for c in universe.cat_features if c in names]

    lines: list[str] = []

    def note(message: str) -> None:
        print(message)
        lines.append(message)

    if count_rescale:
        rescale = rescale_frame_counts(X, specs, fm, cfg, source, rank_idx, y.index, note)
    else:
        rescale = {"applied": False, "reason": "--no-count-rescale"}
        note("    count rescale: off (--no-count-rescale): the whole-frame counts are this tree's")

    apt = mvt.loc[dep_idx, "apt"]
    unseen = frame_shift.unseen_report(
        X, matrix_cats, fit_idx, rank_idx, apt, bool(cfg.model.per_airport),
        specs=specs, block_columns=fm.block_columns, source=source, train_ids=y.index,
    )
    exceeded = frame_shift.unseen_verdict(unseen)
    if exceeded and not allow_unseen:
        raise PredictRefused(
            "categorical levels the model never saw cover far more ranking rows than on the source's own ranking rows "
            f"(a label spelled differently in the release, or a renumbered stand list?): {exceeded}. Check the file "
            "(final_inputs.py validate), or pass --allow-unseen to predict anyway."
        )

    y_fit, _, _ = _prepare_targets(cfg, y, X, fit_idx)
    state_cols = [c for c in state_columns(model) if c in X.columns]
    state = restore_fit_state(model, X.loc[fit_idx, state_cols], y_fit)
    set_predict_threads(model, threads)
    test, detail = rank_predictions(cfg, model, X, y, rank_idx)
    if not np.isfinite(test.to_numpy()).all():
        raise RuntimeError(f"{int((~np.isfinite(test.to_numpy())).sum())} ranking predictions are not finite")
    reproduction = _reproduction(src, test)

    run = tracking.Run(cfg, seed=src.seed)
    run.log(f"=== {run.run_id}: prc predict from {src.run_id}")
    run.log(f"data tree {tree_info['data_tree']} (data.version {cfg.data.version}), source tree {source_tree} "
            f"(data.version {src.cfg.data.version}{'' if tree_info['source_data_tree_recorded'] else ', assumed'})")
    run.log(f"blocks {counts}" + (f"; drift allowed on {drift}" if drift else ""))
    run.log(f"model: {len(model_files)} file(s), {len(names)} features ({len(model_cats)} categorical)"
            + (f"; {len(extra)} matrix column(s) the model does not use: {extra}" if extra else ""))
    run.log(f"fit-time state: {json.dumps(state, default=str)}")
    for line in lines:
        run.log(line.strip())
    for col, info in unseen.items():
        if info["rank_rows"]:
            base = (info.get("baseline") or {}).get("rank_share")
            run.log(f"unseen levels in {col}: {info['rank_rows']:,} ranking rows ({100 * info['rank_share']:.2f} %"
                    + (f"; source {100 * base:.2f} %" if base is not None else "") + f"), {info['levels']:,} levels, "
                    f"e.g. {info['examples']}")
    if exceeded:
        run.log(f"UNSEEN levels out of line, allowed (--allow-unseen): {exceeded}")
    if reproduction.get("comparable"):
        verdict = "reproduced" if reproduction["within_tol"] else "DIFFERS from"
        run.log(f"{verdict} the source's test_predictions: max |diff| {reproduction['max_abs_diff']:.3g} s, "
                f"rms {reproduction['rms_diff']:.3g} s over {reproduction['n']:,} rows")
    else:
        run.log(f"no reproduction check: {reproduction['reason']}")

    run.write_parquet("test_predictions.parquet", test.to_frame())
    if detail is not None:
        run.write_parquet("refit_detail.parquet", detail[list(DETAIL_COLUMNS)])
    for copied in ("oof_predictions.parquet", "metrics_by_airport.csv"):
        if (src.dir / copied).exists():
            run.copy_in(src.dir / copied)
    if (src.dir / "refit.json").exists():
        run.write_json("refit.json", _read_json(src.dir / "refit.json") | {"predict_from": src.run_id})
    run.log_feature_list(fm.block_columns, {b: info["hash"] for b, info in blocks.items() if "hash" in info})

    inputs = {
        "movements.parquet": paths.MOVEMENTS_FILE,
        "target.parquet": paths.TARGET_FILE,
        "ranking.parquet": paths.RANKING_FILE,
        "submitting.parquet": paths.SUBMITTING_FILE,
    }
    external = {p.name: _file_digest(p) for s in specs for p in s.external}
    record = {
        "source_run": src.run_id,
        "source_dir": str(src.dir),
        "source_git_sha": src.env.get("git_sha"),
        **tree_info,
        "prepare": prepared,
        "training_target": training,
        "inputs": _digests(inputs),
        "external": external,
        "blocks": blocks,
        "drift_allowed": drift,
        "model": {
            "files": {f.name: _file_digest(f) for f in model_files},
            "n_features": len(names),
            "categorical": model_cats,
            "unused_matrix_columns": extra,
            "state": state,
            "threads": int(threads),
        },
        "rows": {"rank": int(len(rank_idx)), "refit_fit": int(len(fit_idx)), "training": int(len(train_idx))},
        "count_rescale": rescale,
        "parity": parity_info,
        "unseen_levels": unseen,
        "unseen_exceeded": exceeded,
        "unseen_allowed": bool(exceeded) and allow_unseen,
        "reproduction": reproduction,
        "prediction": {"median": float(test.median()), "mean": float(test.mean()), "min": float(test.min()), "max": float(test.max())},
    }
    if detail is not None:
        record["mixture"] = {
            "n_universe_rank": int(detail["in_universe"].sum()),
            "p_stale_sum_rank": float(detail["p_stale"].sum()),
            "hedge_rank": hedging_report(detail)["hedge"],
        }
    run.write_json("predict.json", record)

    metrics = {k: v for k, v in src.metrics.items() if k not in ("wall_seconds", "run_id", "name", "config_hash", "seed")}
    metrics["refit"] = (src.metrics.get("refit") or {}) | {"predict_from": src.run_id}
    metrics["predict"] = {
        "source_run": src.run_id,
        "data_tree": tree_info["data_tree"],
        "data_version": cfg.data.version,
        "n_rank": int(len(rank_idx)),
        "reproduction_max_abs_diff": reproduction.get("max_abs_diff"),
        "count_rescale": bool(rescale.get("applied")),
        "note": "holdout metrics, oof_predictions and refit.json are the source run's (same model)",
    }
    run.log(f"ranking predictions: {len(test):,} rows, median {test.median():,.0f}s (refit model of {src.run_id})")
    run.log_metrics(metrics)
    run.finish()
    return run
