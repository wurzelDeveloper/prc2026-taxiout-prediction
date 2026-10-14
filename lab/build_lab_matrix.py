"""Materialise one config's feature matrix for fast experiments outside `prc train`.

Writes to cache/lab/<config name>/:
  X.parquet          every departure (train and rank), fold-independent blocks plus the
                     fold-dependent blocks computed for the seasonal split (fit = Feb-Jun, Aug-Dec 2025)
  X_refit_dep.parquet the fold-dependent columns recomputed on every training row (the refit's view)
  meta.json          categorical columns, block -> columns, split definitions

Usage: python lab/build_lab_matrix.py exp051_lean2.yaml [-o key=value ...] [--rescale-from RUN] [--out NAME]

On another data tree (PRC_DATA_DIR set, its own data.version through -o): --rescale-from <a finished run of this
config> puts the whole-frame counts of the ranking rows back on that run's training-time scale
(`prc.predict.rescale_frame_counts`, as `prc predict` does for the base members: a frame holding the final set's
January, February, June and July 2026 instead of Jan+Jul counts about 13 % more on every row; findings 10.33), and
--out writes the matrix under another name (e.g. lean2__<version>), so the delivered set's matrix stays;
`lab/adsb_apply.py --lab-matrix NAME` reads it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prc import data, paths, validation  # noqa: E402
from prc.config import load_config  # noqa: E402
from prc.features import resolve  # noqa: E402
from prc.features.build import build_matrix  # noqa: E402


def main(cfg_name: str, overrides: list[str] | None = None, rescale_from: str | None = None,
         out_name: str | None = None) -> None:
    cfg_path = Path(cfg_name)
    if not cfg_path.exists():
        cfg_path = paths.CONFIGS / cfg_name
    cfg = load_config(cfg_path, overrides)
    out = paths.CACHE / "lab" / (out_name or cfg.name)
    out.mkdir(parents=True, exist_ok=True)

    mvt = data.load_movements()
    y = data.load_target()
    specs = resolve(cfg.features.include, cfg.features.exclude)
    dep_idx = data.dep_index(mvt)
    train_idx = y.index
    splits = validation.make_splits(cfg, mvt, train_idx)
    assert len(splits) == 1, "lab matrices assume a single holdout split"
    split = splits[0]
    seed = cfg.seed

    fit_mask = pd.Series(False, index=mvt.index)
    fit_mask.loc[split.train_idx] = True
    fm = build_matrix(
        specs, mvt, dep_idx, data_version=cfg.data.version, params=cfg.features.params,
        fold=f"{split.name}__{cfg.split_hash(seed)}", fit_mask=fit_mask, target=y.reindex(dep_idx), cfg=cfg,
    )
    rescale = None
    if rescale_from:   # the ranking rows' whole-frame counts on the source run's training-time scale
        from prc import frame_shift, predict

        src = predict.load_source(rescale_from)
        source_tree, _ = predict.source_data_tree(src)
        source = frame_shift.SourceFrame(data_version=src.cfg.data.version,
                                         hashes=(src.feature_list or {}).get("hashes") or {},
                                         refit_fold=f"refit__{src.cfg.split_hash(src.seed)}", tree=source_tree)
        rescale = predict.rescale_frame_counts(fm.X, specs, fm, cfg, source, data.dep_index(mvt, "rank"), y.index, print)
    fm.X.to_parquet(out / "X.parquet")
    dep_blocks = [s.name for s in specs if s.uses_target]
    dep_cols = [c for b in dep_blocks for c in fm.block_columns[b]]

    fit_all = pd.Series(False, index=mvt.index)
    fit_all.loc[train_idx] = True
    fr = build_matrix(
        specs, mvt, dep_idx, data_version=cfg.data.version, params=cfg.features.params,
        fold=f"refit__{cfg.split_hash(seed)}", fit_mask=fit_all, target=y.reindex(dep_idx), cfg=cfg,
    )
    fr.X[dep_cols].to_parquet(out / "X_refit_dep.parquet")

    meta = {
        "config": str(cfg_path),
        "cat_cols": fm.cat_cols,
        "block_columns": fm.block_columns,
        "fold_dependent_columns": dep_cols,
        "split": split.name,
        "n_fit": int(len(split.train_idx)),
        "n_val": int(len(split.val_idx)),
        "holdout_months": list(cfg.validation.holdout_months),
        "data_version": cfg.data.version,
        "data_tree": str(paths.DATA),
        "count_rescale": {"from": rescale_from, **rescale} if rescale else None,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"wrote {out} : X {fm.X.shape}, fold-dependent {dep_cols}")


def _cli(argv: list[str]) -> None:
    def value(flag):
        return argv[argv.index(flag) + 1] if flag in argv else None

    overrides = []
    if "-o" in argv:
        for a in argv[argv.index("-o") + 1:]:
            if a.startswith("--"):
                break
            overrides.append(a)
    cfg_name = argv[1] if len(argv) > 1 and not argv[1].startswith("-") else "exp051_lean2.yaml"
    main(cfg_name, overrides, value("--rescale-from"), value("--out"))


if __name__ == "__main__":
    _cli(sys.argv)
