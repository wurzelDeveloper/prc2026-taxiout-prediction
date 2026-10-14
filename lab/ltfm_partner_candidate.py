"""Blend an Istanbul specialist run (exp070 / exp071, findings 10.28) into a reference submission on the LTFM rows,
re-clipped to the NM join window; every other row keeps the reference.

Prints the holdout (LTFM without wrong-date rows, LTFM bulk, the whole holdout without wrong-date rows, bulk, and
the whole holdout without wrong-date rows in January and in July) for specialist weights 0..0.6, for the specialist
and, with --control RUN, for a control specialist at the same weights: the partner-runway block's effect inside the
blend is specialist minus control at equal weight. With --write NAME --w W it writes the blend as a submittable run
(the specialist needs test_predictions.parquet, i.e. a refitted run).

Usage: python lab/ltfm_partner_candidate.py <reference_run> <specialist_run> [--control RUN] [--write NAME --w W]

Operational availability: the reference's tiers (v62: up to M; LTFM itself has no ADS-B coverage, so its rows carry
the base's tiers); the specialist adds `partner_runway` (tier T).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prc import data, paths  # noqa: E402
from prc.postprocess import clip_to_bounds, lobt_bounds  # noqa: E402

APT = "LTFM"
WEIGHTS = np.round(np.arange(0, 0.61, 0.1), 1)


def blend(ref: pd.Series, spec: pd.Series, rows: pd.Index, w: float, mvt: pd.DataFrame) -> pd.Series:
    """The reference with `rows` replaced by w * specialist + (1 - w) * reference, clipped to the join window."""
    out = ref.copy()
    r = rows.intersection(out.index).intersection(spec.index)
    lo, hi = lobt_bounds(mvt, r)
    out.loc[r] = clip_to_bounds(w * spec.loc[r].to_numpy() + (1 - w) * ref.loc[r].to_numpy(), lo, hi)
    return out


def scores(pred: pd.Series, y: pd.Series, wd: np.ndarray, ltfm: np.ndarray, month: np.ndarray) -> str:
    e = ((pred - y) ** 2).to_numpy()
    bulk = (y < 3600).to_numpy()

    def r(m):
        return np.sqrt(e[m].mean())

    return (f"LTFM ex_wd {r(ltfm & ~wd):7.3f}  LTFM bulk {r(ltfm & bulk):7.3f}  |  all ex_wd {r(~wd):7.3f}  "
            f"all bulk {r(bulk):7.3f}  Jan {r(~wd & (month == 1)):7.3f}  Jul {r(~wd & (month == 7)):7.3f}")


def main():
    args = sys.argv[1:]
    ref_run, spec_run = args[0], args[1]
    control = args[args.index("--control") + 1] if "--control" in args else None
    mvt = data.load_movements(columns=["apt", "MVT_TIME_UTC_mvt", "LOBT_flt"])
    R = paths.run_dir(ref_run)
    ro = pd.read_parquet(R / "oof_predictions.parquet")
    y, wd = ro["y_true"], ro["dayoffset"].astype(bool).to_numpy()
    ltfm = (mvt.loc[ro.index, "apt"] == APT).to_numpy()
    month = mvt.loc[ro.index, "MVT_TIME_UTC_mvt"].dt.month.to_numpy()
    ltfm_rows = ro.index[ltfm]
    for label, run in [("specialist", spec_run)] + ([("control", control)] if control else []):
        so = pd.read_parquet(paths.run_dir(run) / "oof_predictions.parquet")["y_pred"]
        missing = ltfm_rows.difference(so.index)
        print(f"{label} {run}: {len(ltfm_rows) - len(missing):,} of {len(ltfm_rows):,} LTFM holdout rows"
              + (f" ({len(missing):,} missing keep the reference)" if len(missing) else ""))
        for w in WEIGHTS:
            print(f"  w {w:.1f}: {scores(blend(ro['y_pred'], so, ltfm_rows, w, mvt), y, wd, ltfm, month)}")
    if "--write" in args:
        name, w = args[args.index("--write") + 1], float(args[args.index("--w") + 1])
        S = paths.run_dir(spec_run)
        tag = hashlib.sha1(f"{ref_run}|{spec_run}|{w}".encode()).hexdigest()[:8]
        dst = paths.RUNS / f"{ref_run.split('__')[0]}__{name}__{tag}"
        dst.mkdir(parents=True, exist_ok=False)
        so = pd.read_parquet(S / "oof_predictions.parquet")["y_pred"]
        o2 = ro.copy()
        o2["y_pred"] = blend(ro["y_pred"], so, ltfm_rows, w, mvt)
        o2.to_parquet(dst / "oof_predictions.parquet")
        rt = pd.read_parquet(R / "test_predictions.parquet")["y_pred"]
        st = pd.read_parquet(S / "test_predictions.parquet")["y_pred"]
        rank_ltfm = rt.index[(mvt.loc[rt.index, "apt"] == APT).to_numpy()]
        out = blend(rt, st, rank_ltfm, w, mvt)
        pd.DataFrame({"y_pred": out}).to_parquet(dst / "test_predictions.parquet")
        changed = int((out != rt).sum())
        rms = float(np.sqrt(((out - rt) ** 2).mean()))
        (dst / "metrics.json").write_text(json.dumps({"stack": {
            "source": ref_run, "ltfm_specialist": spec_run, "w_specialist": w, "airport": APT,
            "rank_rows_changed": changed, "rank_rms_change_s": rms, "steps": ["ltfm_partner_candidate"]}}, indent=1))
        print(f"wrote {dst.name} ({changed:,} ranking rows changed, rms change {rms:.2f} s over all rows)")


if __name__ == "__main__":
    main()
