"""Combined base for the ADS-B stage: the seasonal stack's holdout OOF (Jan+Jul 2025) plus a winter base for the
Feb+Dec 2025 training days (findings 10.13).

The submitted stack has no out-of-fold prediction for February or December (those months are in its training folds),
so the extra ADS-B training days take their base from the winter-split runs (the same configurations trained on the
other ten months, validated on Feb+Dec): the mean of the given runs' y_pred, p_stale and y_reg, clipped to the NM join
window. The output run directory holds oof_predictions.parquet with a `src` column (0 = stack rows, 1 = winter rows),
the stack's test_predictions.parquet and metrics.json, so `lab/adsb_apply.py <combined>` trains on both and predicts
2026 exactly as before.

Usage: python lab/build_winter_base.py <stack_run> <group> [<group> ...]

A group is one or more runs that share a holdout (comma-separated, e.g. the three Feb+Dec members
`exp056w,exp059w,exp060w`); several groups cover different month pairs (Feb+Dec, Mar+Nov, Jun+Aug) and are concatenated,
each group averaged over its own members on the intersection of their rows. Rows already in the stack's OOF are left out.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prc import data, paths  # noqa: E402
from prc.postprocess import clip_to_bounds, lobt_bounds  # noqa: E402


def main():
    stack, groups = sys.argv[1], [g.split(",") for g in sys.argv[2:]]
    winters = [w for g in groups for w in g]
    S = paths.run_dir(stack)
    so = pd.read_parquet(S / "oof_predictions.parquet")
    mvt = data.load_movements(columns=["MVT_TIME_UTC_mvt", "LOBT_flt"])
    cols = ["y_pred", "p_stale", "y_reg"]
    parts, taken = [], so.index
    for gi, g in enumerate(groups, start=1):
        frames = [pd.read_parquet(paths.run_dir(w) / "oof_predictions.parquet") for w in g]
        idx = frames[0].index
        for f in frames[1:]:
            idx = idx.intersection(f.index)
        idx = idx.difference(taken)
        mean = sum(f.loc[idx, cols].astype("float64") for f in frames) / len(frames)
        lo, hi = lobt_bounds(mvt, idx)
        w = frames[0].loc[idx].copy()
        w["y_pred"] = clip_to_bounds(mean["y_pred"].to_numpy(), lo, hi)
        w["p_stale"] = mean["p_stale"].to_numpy()
        w["y_reg"] = mean["y_reg"].to_numpy()
        w = w[[c for c in so.columns if c in w.columns]]
        w["src"] = gi
        parts.append(w)
        taken = taken.union(idx)
        print(f"group {gi}: {len(g)} members, {len(w):,} rows")
    w = pd.concat(parts)
    s2 = so.copy(); s2["src"] = 0
    out = pd.concat([s2, w])
    months = mvt.loc[out.index, "MVT_TIME_UTC_mvt"].dt.month.value_counts().sort_index().to_dict()
    tag = hashlib.sha1(("|".join([stack] + winters)).encode()).hexdigest()[:8]
    dst = paths.RUNS / f"{stack.split('__')[0]}__base_plus_winter__{tag}"
    dst.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dst / "oof_predictions.parquet")
    for f in ("test_predictions.parquet", "config.resolved.json", "env.json", "metrics.json"):
        if (S / f).exists():
            shutil.copy(S / f, dst / f)
    (dst / "winter_base.json").write_text(json.dumps({"stack": stack, "winter_runs": winters, "n_stack": int(len(s2)),
                                                       "n_winter": int(len(w)), "rows_by_month": {int(k): int(v) for k, v in months.items()}}, indent=1))
    print(f"wrote {dst.name}: {len(s2):,} stack rows + {len(w):,} winter rows; rows by month {months}")
    print(f"winter base: mean y_pred {w.y_pred.mean():.0f} s, mean p_stale {w.p_stale.mean():.3f}; members {len(winters)}")


if __name__ == "__main__":
    main()
