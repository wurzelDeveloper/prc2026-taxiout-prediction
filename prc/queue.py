"""`prc queue`: run several configs one after another, live in the terminal.

Each config is one ``python -m prc train`` child process with the terminal's
stdout, so CatBoost's iteration log and the fold summaries stream as they
happen and `prc progress` works from another window. A failing config is
reported and the queue moves on (``--stop-on-error`` to stop instead); at the
end the runs the queue produced are read back from ``runs/index.csv`` into
one summary table.

CatBoost uses every core (and the GPU) of one fit, so a queue is the way to
run many experiments: one at a time, unattended, with nothing competing.
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime

import pandas as pd

from prc import paths, tracking

SUMMARY_COLUMNS = [
    "name", "seed", "rmse", "rmse_clean", "rmse_ex_dayoffset", "rmse_disrupted", "n_disrupted_val", "rmse_bulk",
    "hedge_rank", "mae", "bias", "wall_seconds", "run_id",
]


def commands(
    configs: list[str],
    *,
    seeds: int = 1,
    no_test: bool = False,
    keep_test: bool = False,
    overrides: list[str] | None = None,
) -> list[list[str]]:
    """The ``prc train`` command line for every config, in order."""
    out = []
    for cfg in configs:
        cmd = [sys.executable, "-m", "prc", "train", "-c", cfg]
        if seeds > 1:
            cmd += ["--seeds", str(seeds)]
        if no_test:
            cmd.append("--no-test")
        if keep_test:
            cmd.append("--keep-test")
        if overrides:
            cmd += ["-o", *overrides]
        out.append(cmd)
    return out


def _index_rows() -> int:
    return len(pd.read_csv(paths.RUN_INDEX)) if paths.RUN_INDEX.exists() else 0


def summary(since_row: int) -> pd.DataFrame:
    """The index rows appended since the queue started."""
    frame = tracking.load_index()
    if frame.empty or len(frame) <= since_row:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    new = frame.iloc[since_row:]
    cols = [c for c in SUMMARY_COLUMNS if c in new.columns]
    return new[cols].reset_index(drop=True)


def run(configs: list[str], *, seeds: int = 1, no_test: bool = False, keep_test: bool = False,
        overrides: list[str] | None = None, stop_on_error: bool = False, dry_run: bool = False) -> int:
    cmds = commands(configs, seeds=seeds, no_test=no_test, keep_test=keep_test, overrides=overrides)
    print(f"queue: {len(cmds)} config(s)" + (f", {seeds} seeds each" if seeds > 1 else ""))
    for cfg, cmd in zip(configs, cmds):
        print(f"  {cfg:<40} {' '.join(cmd[3:])}")
    if dry_run:
        return 0

    before = _index_rows()
    failures: list[str] = []
    t_queue = time.perf_counter()
    for k, (cfg, cmd) in enumerate(zip(configs, cmds), 1):
        print(f"\n{'=' * 100}\n[{k}/{len(cmds)}] {cfg}   started {datetime.now():%H:%M:%S}\n{'=' * 100}", flush=True)
        t0 = time.perf_counter()
        code = subprocess.run(cmd, cwd=paths.ROOT).returncode
        minutes = (time.perf_counter() - t0) / 60
        if code != 0:
            failures.append(cfg)
            print(f"\n[{k}/{len(cmds)}] {cfg} FAILED (exit {code}) after {minutes:.1f} min", flush=True)
            if stop_on_error:
                break
        else:
            print(f"\n[{k}/{len(cmds)}] {cfg} done in {minutes:.1f} min", flush=True)

    table = summary(before)
    print(f"\n{'=' * 100}\nqueue finished in {(time.perf_counter() - t_queue) / 60:.1f} min; "
          f"{len(cmds) - len(failures)} of {len(cmds)} config(s) succeeded")
    if failures:
        print("failed: " + ", ".join(failures))
    if not table.empty:
        print("\n" + table.round(1).to_string(index=False))
    return 1 if failures else 0
