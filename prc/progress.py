"""How far along an in-flight training run is.

A long CatBoost fit is otherwise opaque: the process looks identical whether it
is boosting normally, crawling, or wedged. `CatBoostModel` points CatBoost's
`train_dir` at the run directory so it streams `learn_error.tsv` and
`test_error.tsv` as it goes; this reads them back and turns them into an
iteration count, a rate and an ETA.

The same files are the run's saved learning curve once it finishes.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from prc import paths


def latest_run() -> Path | None:
    runs = [d for d in paths.RUNS.iterdir() if d.is_dir()] if paths.RUNS.exists() else []
    return max(runs, key=lambda d: d.stat().st_mtime) if runs else None


def _curves(run_path: Path) -> list[tuple[str, Path]]:
    """Every CatBoost learning curve under the run, labelled by fold/airport.

    A fold fit writes `learn_error.tsv` and `test_error.tsv`; the refit has no
    eval set, so only the former. Per directory, the test curve is preferred.
    """
    found = []
    for learn in sorted(run_path.glob("model/**/learn_error.tsv")):
        test = learn.with_name("test_error.tsv")
        found.append((str(learn.parent.relative_to(run_path / "model")), test if test.exists() else learn))
    return found


def _refit_budget(run_path: Path, label: str) -> int | None:
    """The refit's own iteration count for a curve labelled ``refit`` or ``refit/<airport>``."""
    parts = label.replace("\\", "/").split("/")
    if parts[0] != "refit" or not (run_path / "refit.json").exists():
        return None
    payload = json.loads((run_path / "refit.json").read_text(encoding="utf-8"))
    if parts[-1] == "classifier":
        # The regime classifier of a mixture model has its own budget.
        return payload.get("classifier_iterations")
    its = payload.get("iterations")
    if isinstance(its, dict):
        return its.get(parts[-1])
    return its


def report(run_id: str | None = None, tail: int = 5) -> str:
    run_path = paths.run_dir(run_id) if run_id else latest_run()
    if run_path is None or not run_path.exists():
        return f"no runs found under {paths.RUNS}"

    lines = [f"run: {run_path.name}"]
    config_file = run_path / "config.resolved.json"
    total = None
    if config_file.exists():
        cfg = json.loads(config_file.read_text(encoding="utf-8"))
        total = cfg.get("model", {}).get("params", {}).get("iterations")
        lines.append(
            f"model: {cfg.get('model', {}).get('type')}  "
            f"iterations: {total}  scheme: {cfg.get('validation', {}).get('scheme')}"
        )

    started = datetime.fromtimestamp(config_file.stat().st_mtime) if config_file.exists() else None
    if started:
        elapsed = time.time() - config_file.stat().st_mtime
        lines.append(f"started: {started:%H:%M:%S}   elapsed: {elapsed / 60:.1f} min")

    if (run_path / "metrics.json").exists():
        payload = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
        overall = payload.get("overall", {})
        lines.append(f"\nFINISHED in {payload.get('wall_seconds', 0) / 60:.1f} min")
        lines.append(f"  rmse={overall.get('rmse', float('nan')):.1f}  mae={overall.get('mae', float('nan')):.1f}")
        return "\n".join(lines)

    curves = _curves(run_path)
    if not curves:
        lines.append(
            "\nno learning curve yet -- either the fit has not started, or this run predates "
            "`train_dir` logging (older runs are opaque; nothing to do but wait)."
        )
        return "\n".join(lines)

    for label, path in curves:
        try:
            curve = pd.read_csv(path, sep="\t")
        except Exception:  # pragma: no cover - file being written mid-read
            continue
        if curve.empty:
            continue
        done = int(curve["iter"].iloc[-1]) + 1
        limit = _refit_budget(run_path, label) or total
        lines.append(f"\n[{label or 'fit'}] iteration {done:,}" + (f" of {limit:,}" if limit else ""))
        if elapsed and done:
            rate = elapsed / done
            lines.append(f"  {rate:.2f}s/iteration")
            if limit and done < limit:
                lines.append(f"  ~{(limit - done) * rate / 60:.0f} min remaining (if it runs to the limit)")
        lines.append(curve.tail(tail).to_string(index=False))
    return "\n".join(lines)
