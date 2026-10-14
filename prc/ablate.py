"""Ablation sweeps: the honest measurement of what a block is worth.

Three modes:

* ``leave-one-out`` -- run the full config, then rerun it once per block with
  that block removed. ``RMSE(without) - RMSE(with)``, positive means useful.
* ``add-one-in``    -- start from a lean base (`PROTECTED` only) and add one
  block at a time. This tells you what a block carries on its own.
* ``add-to-base``   -- run the config as it is, then rerun it once per named
  block with that block added. This is the test for a feature you have just
  written: "does exp028 plus this block beat exp028" is the question a
  submission decision needs, and leave-one-out cannot ask it because the block
  is not in the config yet, while add-one-in answers a different one.

Every mode holds the seed and the fold assignment fixed across the pair, so the
delta is attributable to the block and not to a reshuffle. Pair that with a
seed-noise run to know whether the delta means anything. Besides the headline
``rmse_delta``, the sweep records the deltas of ``rmse_ex_dayoffset`` and
``rmse_disrupted`` when the runs report them, under their own method names.

With ``--emit-configs`` nothing is trained: the variants are written to
``configs/_generated/`` and the matching `prc queue` line is printed, so a long
sweep runs as one child process per config (a crash or an out-of-memory GPU
stops one variant, not the night).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from prc import paths, tracking
from prc.config import ExperimentConfig, dump_config
from prc.features import resolve
from prc.features.registry import dependents
from prc.train import run_experiment

PROTECTED = {"core_keys"}
"""Blocks every other block depends on; removing them is not an experiment."""
MODES = ("leave-one-out", "add-one-in", "add-to-base")
METRICS = (
    ("rmse_delta", "overall", ""),
    ("rmse_ex_dayoffset_delta", "overall_ex_dayoffset", "__ex_dayoffset"),
    ("rmse_disrupted_delta", "overall_disrupted", "__disrupted"),
    ("rmse_bulk_delta", "overall_bulk", "__bulk"),
)
"""(effect metric, metrics.json key, method suffix): the headline delta, the two report-mask deltas and the bulk delta."""


def _rmses(run: tracking.Run) -> dict[str, float]:
    """The RMSE of every metric block in `METRICS` the run reported (skipping empty ones)."""
    payload = json.loads((run.dir / "metrics.json").read_text(encoding="utf-8"))
    out = {}
    for _, key, _ in METRICS:
        block = payload.get(key) or {}
        value = block.get("rmse")
        if value is not None and value == value and block.get("n", 1):
            out[key] = float(value)
    return out


def _rmse(run: tracking.Run) -> float:
    return _rmses(run)["overall"]


def plan_variants(
    cfg: ExperimentConfig, mode: str = "leave-one-out", blocks: list[str] | None = None
) -> tuple[ExperimentConfig, list[tuple[str, ExperimentConfig]], dict[str, list[str]], float]:
    """What an ablation would run, without running it.

    Returns the base config, the ``(block, variant config)`` pairs, the blocks
    that actually move with each variant (dependents removed, dependencies
    added) and the sign that turns ``variant - base`` into "positive = useful".
    """
    if mode not in MODES:
        raise ValueError(f"unknown ablation mode '{mode}' (one of {MODES})")
    specs = resolve(cfg.features.include, cfg.features.exclude)
    in_config = [s.name for s in specs]
    removed: dict[str, list[str]] = {}

    if mode == "add-to-base":
        if not blocks:
            raise ValueError("add-to-base needs --blocks: the registered blocks to add to the config, one variant each")
        variants = []
        for name in blocks:
            if name in in_config:
                raise ValueError(f"'{name}' is already in the config; measure it with leave-one-out")
            # `resolve` raises KeyError for an unknown block; what really joins the
            # matrix is the block plus whatever dependency the config lacked.
            removed[name] = [s.name for s in resolve([name]) if s.name not in in_config]
            variants.append(
                (name, cfg.replace(
                    name=f"{cfg.name}__plus_{name}",
                    features={"include": [*cfg.features.include, name]},
                ))
            )
        return cfg, variants, removed, -1.0  # variant minus base: negative = block helped

    candidates = [s.name for s in specs if s.name not in PROTECTED]
    if blocks:
        unknown = set(blocks) - set(candidates)
        if unknown:
            raise KeyError(f"blocks not in this config (or protected): {sorted(unknown)}")
        candidates = list(blocks)

    if mode == "leave-one-out":
        variants = []
        for name in candidates:
            # A block cannot be dropped while something downstream still needs
            # it, so drop its dependents too and report what actually went. The
            # number then answers "what are this block and everything built on
            # it worth", which is the only question that has an answer.
            group = [b for b in dependents(name, in_config) if b not in PROTECTED]
            removed[name] = group
            variants.append(
                (name, cfg.replace(
                    name=f"{cfg.name}__wo_{name}",
                    features={"exclude": sorted(set(cfg.features.exclude) | set(group))},
                ))
            )
        return cfg, variants, removed, 1.0  # variant RMSE minus base RMSE: positive = block was useful

    base_cfg = cfg.replace(
        name=f"{cfg.name}__lean",
        features={"include": sorted(PROTECTED), "exclude": []},
    )
    variants = []
    for name in candidates:
        # Symmetrically, a block cannot be added without its dependencies.
        group = sorted(set(PROTECTED) | {s.name for s in resolve([name])})
        removed[name] = group
        variants.append(
            (name, cfg.replace(
                name=f"{cfg.name}__only_{name}",
                features={"include": group, "exclude": []},
            ))
        )
    return base_cfg, variants, removed, -1.0  # variant RMSE minus lean RMSE: negative = block helped


def emit_configs(variants: list[ExperimentConfig], directory: Path | None = None) -> list[Path]:
    """Write each variant as a standalone YAML (the `extends` chain resolved) and return the paths."""
    directory = paths.GENERATED_CONFIGS if directory is None else Path(directory)
    written = []
    for variant in variants:
        path = directory / f"{variant.name}.yaml"
        dump_config(variant, path)
        written.append(path)
    return written


def run_ablation(
    cfg: ExperimentConfig,
    mode: str = "leave-one-out",
    blocks: list[str] | None = None,
    *,
    seed: int | None = None,
    emit_only: bool = False,
) -> pd.DataFrame:
    seed = cfg.seed if seed is None else seed
    base_cfg, variants, removed, sign = plan_variants(cfg, mode, blocks)
    print(f"ablation mode={mode} over {len(variants)} block(s), seed={seed}")

    verb = {"leave-one-out": "also removed", "add-one-in": "also required", "add-to-base": "also added"}[mode]
    for name, group in removed.items():
        extra = [b for b in group if b != name]
        if extra:
            print(f"  {name}: {verb} {extra}")

    if emit_only:
        files = emit_configs([v for _, v in variants])
        for f in files:
            print(f"  wrote {f}")
        print("run them with:\n  python -m prc queue --no-test -c " + " ".join(str(f) for f in files))
        return pd.DataFrame({"block": [n for n, _ in variants], "config": [str(f) for f in files]})

    print(f"--- base: {base_cfg.name}")
    base_run = run_experiment(base_cfg, seed=seed, compute_test=False)
    base = _rmses(base_run)

    rows = []
    for name, variant in variants:
        print(f"--- variant: {name}")
        run = run_experiment(variant, seed=seed, compute_test=False)
        got = _rmses(run)
        for metric, key, suffix in METRICS:
            if key not in got or key not in base:
                continue
            delta = (got[key] - base[key]) * sign
            rows.append(
                {
                    "run_id": run.run_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "base_run_id": base_run.run_id,
                    "block": name,
                    # Record the blocks that actually moved, so a delta is never
                    # read as belonging to `name` alone when dependents came too.
                    "columns": ",".join(removed.get(name, [name])),
                    "method": f"ablation_{mode.replace('-', '_')}{suffix}",
                    "metric": metric,
                    "value": delta,
                    "std": None,
                    "n_seeds": 1,
                    "config_hash": variant.hash(),
                    "val_scheme": variant.validation.scheme,
                }
            )
        headline = (got["overall"] - base["overall"]) * sign
        print(f"    {name}: variant rmse={got['overall']:.1f} base={base['overall']:.1f} -> delta {headline:+.1f}")

    frame = pd.DataFrame(rows).sort_values("value", ascending=False)
    tracking._append_csv(paths.FEATURE_EFFECTS, frame)
    return frame


def seed_noise(cfg: ExperimentConfig, seeds: list[int], *, compute_test: bool = False) -> pd.DataFrame:
    """Run the same config under several seeds to establish the noise floor.

    The spread is recorded into feature_effects.csv under the pseudo-block
    ``__seed_noise__`` so `leaderboard --features` can print it next to the
    block effects; a delta smaller than this is not a finding. Each seed's run
    also keeps its own index row, so `prc blend --runs` can average them; with
    ``compute_test`` every seed predicts the ranking set as well (and refits,
    if the config says so), which is what a blended submission needs.
    """
    rmses = []
    run_ids = []
    for s in seeds:
        run = run_experiment(cfg, seed=s, compute_test=compute_test)
        rmses.append(_rmse(run))
        run_ids.append(run.run_id)
        print(f"  seed {s}: rmse={rmses[-1]:.1f}")

    frame = pd.DataFrame(
        [
            {
                "run_id": run_ids[-1],
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "base_run_id": run_ids[0],
                "block": "__seed_noise__",
                "columns": "",
                "method": "seed_noise",
                "metric": "rmse_std",
                "value": float(pd.Series(rmses).std(ddof=1)) if len(rmses) > 1 else 0.0,
                "std": None,
                "n_seeds": len(seeds),
                "config_hash": cfg.hash(),
                "val_scheme": cfg.validation.scheme,
            }
        ]
    )
    tracking._append_csv(paths.FEATURE_EFFECTS, frame)
    print(f"seed noise over {len(seeds)} seeds: std={frame['value'].iloc[0]:.2f} rmse ({', '.join(f'{r:.1f}' for r in rmses)})")
    return frame
