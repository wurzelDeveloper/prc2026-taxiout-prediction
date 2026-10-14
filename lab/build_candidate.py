"""One submission candidate from finished ADS-B stage runs, in the order of the submitted pipeline:

1. residual member = equal-weight average of the residual stage runs (`prc blend`), skipped for a single run;
2. covered rows = (1 - w) x residual member + w x direct member (`lab/blend_direct.py`, w = 0.3 by default);
3. base swap: the rows no stage run corrects take a retrained base stack (`lab/base_swap.py --new-stack`), if given;
4. the forced-long shift (`lab/forced_long_shift.py`).

Prints the holdout of the result (without wrong-date rows, bulk, January, July) and the final run id, which
`prc submit` takes. Every step writes its own run directory with its record, so `lab/manifest.py` traces the result.

Usage: python lab/build_candidate.py --resid RUN [RUN ...] --direct RUN --name NAME [--w 0.3] [--new-stack RUN]
Operational availability: tier M (the stage reads the flight's own ADS-B movement), as its inputs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prc import data, paths  # noqa: E402

PY = str(ROOT / ".venv" / "Scripts" / "python.exe")


def values(flag: str) -> list[str]:
    if flag not in sys.argv:
        return []
    out = []
    for a in sys.argv[sys.argv.index(flag) + 1:]:
        if a.startswith("--"):
            break
        out.append(a)
    return out


def newest(fragment: str) -> str:
    return sorted(p.name for p in paths.RUNS.glob(f"*__{fragment}__*"))[-1]


def run(*args: str):
    subprocess.run([PY, *args], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)


def holdout(run_id: str) -> str:
    o = pd.read_parquet(paths.run_dir(run_id) / "oof_predictions.parquet")
    mvt = data.load_movements(columns=["MVT_TIME_UTC_mvt"])
    month = mvt.loc[o.index, "MVT_TIME_UTC_mvt"].dt.month.to_numpy()
    e, wd, y = (o.y_pred - o.y_true).to_numpy(float), o.dayoffset.to_numpy(bool), o.y_true.to_numpy(float)
    s = lambda m: np.sqrt((e[m] ** 2).mean())
    return f"ex_wd {s(~wd):.3f}  bulk {s(y < 3600):.3f}  Jan {s(~wd & (month == 1)):.3f}  Jul {s(~wd & (month == 7)):.3f}"


def main():
    resid, direct, name = values("--resid"), values("--direct")[0], values("--name")[0]
    w = values("--w")[0] if "--w" in sys.argv else "0.3"
    new_stack = values("--new-stack")[0] if "--new-stack" in sys.argv else None
    if len(resid) > 1:
        run("-m", "prc", "blend", "--runs", *resid, "--name", f"{name}_resid")
        r = newest(f"{name}_resid")
    else:
        r = resid[0]
    print(f"residual member  {r}: {holdout(r)}")
    run("lab/blend_direct.py", r, direct, "--write", f"{name}_pre", "--w", w)
    c = newest(f"{name}_pre")
    print(f"+ direct at {w}  {c}: {holdout(c)}")
    if new_stack:
        rows = [a for x in resid for a in ("--resid-rows", x)]
        run("lab/base_swap.py", "--candidate", c, *rows, "--direct-rows", direct, "--new-stack", new_stack,
            "--name", f"{name}_swap")
        c = newest(f"{name}_swap")
        print(f"+ base swap      {c}: {holdout(c)}")
    run("lab/forced_long_shift.py", c, "--write", name)
    f = newest(name)
    print(f"+ forced-long    {f}: {holdout(f)}")
    print("final", f)


if __name__ == "__main__":
    main()
