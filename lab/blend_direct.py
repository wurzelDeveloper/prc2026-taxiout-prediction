"""Blend a direct-formulation ADS-B run (`adsb_apply.py --direct`) with a residual-correction run on the rows the direct
model predicts (its corrected_rows.parquet); every other row keeps the reference run.

Prints the holdout (without wrong-date rows, bulk, January, July) for blend weights 0..1 on the direct model and, with
--write NAME --w W, writes the blend as a submittable run.

Usage: python lab/blend_direct.py <reference_run> <direct_run> [--write NAME --w W]

Operational availability: tier M (both inputs read the flight's own ADS-B movement after off-block).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prc import data, paths  # noqa: E402


def blend(ref: pd.Series, direct: pd.Series, rows: pd.Index, w: float) -> pd.Series:
    out = ref.copy()
    r = rows.intersection(out.index)
    out.loc[r] = w * direct.reindex(r).to_numpy() + (1 - w) * ref.loc[r].to_numpy()
    return out


def main():
    ref_run, dir_run = sys.argv[1], sys.argv[2]
    R, D = paths.run_dir(ref_run), paths.run_dir(dir_run)
    rows = pd.read_parquet(D / "corrected_rows.parquet")
    oof_rows, rank_rows = rows.index[rows["kind"] == "oof"], rows.index[rows["kind"] == "rank"]
    ro = pd.read_parquet(R / "oof_predictions.parquet")
    do = pd.read_parquet(D / "oof_predictions.parquet").reindex(ro.index)
    y, wd = ro["y_true"], ro["dayoffset"].astype(bool)
    mo = data.load_movements(columns=["MVT_TIME_UTC_mvt"]).loc[ro.index, "MVT_TIME_UTC_mvt"].dt.month
    print(f"direct rows: holdout {len(oof_rows):,}, ranking {len(rank_rows):,}")
    for w in np.round(np.arange(0, 1.01, 0.1), 1):
        p = blend(ro["y_pred"], do["y_pred"], oof_rows, w)
        e = (p - y) ** 2
        print(f"w_direct {w:.1f}: ex_wd {np.sqrt(e[~wd].mean()):.2f}  bulk {np.sqrt(e[y < 3600].mean()):.2f}  "
              f"Jan {np.sqrt(e[~wd & (mo == 1)].mean()):.2f}  Jul {np.sqrt(e[~wd & (mo == 7)].mean()):.2f}")
    if "--write" in sys.argv:
        name = sys.argv[sys.argv.index("--write") + 1]
        w = float(sys.argv[sys.argv.index("--w") + 1])
        import hashlib
        tag = hashlib.sha1(f"{ref_run}|{dir_run}|{w}".encode()).hexdigest()[:8]
        dst = paths.RUNS / f"{ref_run.split('__')[0]}__{name}__{tag}"
        dst.mkdir(parents=True, exist_ok=False)
        o2 = ro.copy()
        o2["y_pred"] = blend(ro["y_pred"], do["y_pred"], oof_rows, w)
        o2.to_parquet(dst / "oof_predictions.parquet")
        rt = pd.read_parquet(R / "test_predictions.parquet")["y_pred"]
        dt = pd.read_parquet(D / "test_predictions.parquet")["y_pred"]
        pd.DataFrame({"y_pred": blend(rt, dt, rank_rows, w)}).to_parquet(dst / "test_predictions.parquet")
        (dst / "metrics.json").write_text(json.dumps({"stack": {"source": ref_run, "direct": dir_run, "w_direct": w,
                                                                 "steps": ["blend_direct"]}}, indent=1))
        print("wrote", dst.name)


if __name__ == "__main__":
    main()
