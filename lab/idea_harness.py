"""Quick test of a candidate ADS-B feature against what the v37 correction already explains.

The full test of a feature is refitting the correction with it (`lab/adsb_apply.py`, ~35 min). This harness is the
cheap screen in front of it: a small second-stage CatBoost, cross-fitted by day (the same five day-folds as the
correction), fitted on the residual of v37c's cross-fitted holdout prediction over the covered holdout rows without a
stale candidate. It is run twice, with a fixed context alone (control) and with the context plus the candidate; the
difference is the candidate's incremental information. The result is also expressed as the holdout RMSE without
wrong-date rows that v37c would reach if the second stage were applied (v38a: 237.50).

    python lab/idea_harness.py --build                      # writes cache/lab/adsb_resid.parquet (once)
    python lab/idea_harness.py cand.parquet [--cats c1,c2]  # cand.parquet: index mvt_id, one or more columns

From Python: `from idea_harness import load_frame, test; test(cand_df, cats=[...])`. A candidate may cover any subset of
the rows (NaN elsewhere). Threads are capped (`THREADS`) so several screens can run next to a GPU fit.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prc import data, paths  # noqa: E402
from prc.postprocess import clip_to_bounds, lobt_bounds  # noqa: E402

BASE_RUN = "2026-10-01T04-48-58__blend11_stack__b942ea3d"
V37_RUN = "2026-10-01T04-48-58__v38a_adsb_v4__6233174f"
"""The reference correction (named v37 in the frame for compatibility; v38a since the second exploration round)."""
FRAME = paths.CACHE / "lab" / "adsb_resid.parquet"
THREADS = 3
PARAMS = dict(iterations=800, learning_rate=0.05, depth=6, loss_function="RMSE", verbose=0, thread_count=THREADS,
              random_seed=42)
CONTEXT = ["apt", "v37", "base", "adsb_first_dist_stand"]
"""What the control second stage sees; the candidate adds its columns to this."""


def build():
    o = pd.read_parquet(paths.run_dir(V37_RUN) / "oof_predictions.parquet")
    b = pd.read_parquet(paths.run_dir(BASE_RUN) / "oof_predictions.parquet").reindex(o.index)
    import json
    sys.path.insert(0, str(ROOT / "lab"))
    from final_stack import averaged_detail, leaf_runs
    src = json.loads((paths.run_dir(BASE_RUN) / "metrics.json").read_text(encoding="utf-8"))["stack"]["source"]
    det = averaged_detail(leaf_runs(src), "oof").reindex(o.index)
    mvt = data.load_movements(columns=["apt", "MVT_TIME_UTC_mvt", "LOBT_flt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt",
                                       "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "CALLSIGN_flt", "FLIGHT_mvt"])
    lo, hi = lobt_bounds(mvt, o.index)
    f1 = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(paths.CACHE / "adsb" / "features_*.parquet")))])
    f2 = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(paths.CACHE / "adsb" / "features2_*.parquet")))])
    f1 = f1[~f1.index.duplicated(keep="last")]
    f2 = f2[~f2.index.duplicated(keep="last")]
    fr = pd.DataFrame({"y": o["y_true"], "v37": o["y_pred"], "base": clip_to_bounds(b["y_pred"].to_numpy(), lo, hi),
                       "p_stale": det["p_stale"].fillna(0.0), "y_reg": det["y_reg"], "dayoffset": o["dayoffset"].astype(bool)},
                      index=o.index)
    fr = fr.join(mvt).join(f1).join(f2.drop(columns=[c for c in f2.columns if c in f1.columns]))
    f3 = paths.CACHE / "adsb" / "features3.parquet"
    if f3.exists():  # the third set (pushback detector, first reception, track quality, stop classes)
        fr = fr.join(pd.read_parquet(f3))
    fr["apt"] = fr["apt"].astype(str)
    fr["day"] = fr["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d")
    fr["month"] = fr["MVT_TIME_UTC_mvt"].dt.month
    fr["t_mvt"] = (fr["MVT_TIME_UTC_mvt"] - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    fr["covered"] = (fr["adsb_linked"].fillna(0) > 0) & fr["adsb_first_seen"].notna()
    fr["test_row"] = fr["covered"] & (fr["p_stale"] <= 0.05) & ~fr["dayoffset"]
    FRAME.parent.mkdir(parents=True, exist_ok=True)
    fr.to_parquet(FRAME)
    print(f"wrote {FRAME}: {len(fr):,} holdout rows, {fr['covered'].sum():,} covered, {fr['test_row'].sum():,} test rows")


_FR = None


def load_frame() -> pd.DataFrame:
    """The whole Jan+Jul 2025 holdout (344,419 rows), v37c's prediction, the base, the ADS-B features (both sets),
    the movement columns, and the flags `covered` / `test_row` (covered, p_stale <= 0.05, not wrong-date)."""
    global _FR
    if _FR is None:
        _FR = pd.read_parquet(FRAME)
    return _FR


def _cv(X: pd.DataFrame, r: np.ndarray, fold: np.ndarray, cats: list[str]) -> np.ndarray:
    out = np.zeros(len(X))
    for k in range(5):
        tr, te = fold != k, fold == k
        m = CatBoostRegressor(**PARAMS)
        m.fit(Pool(X[tr], r[tr], cat_features=cats))
        out[te] = m.predict(Pool(X[te], cat_features=cats))
    return out


def test(cand: pd.DataFrame, cats: list[str] | None = None, name: str = "candidate", verbose: bool = True) -> dict:
    """Incremental value of `cand` (index mvt_id) over v37c on the covered test rows; see the module docstring."""
    fr = load_frame()
    cats = list(cats or [])
    rows = fr.index[fr["test_row"].to_numpy()]
    t = fr.loc[rows]
    days = sorted(fr["day"].unique())
    fold = t["day"].map({d: i % 5 for i, d in enumerate(days)}).to_numpy()
    r = (t["y"] - t["v37"]).clip(-3600, 3600).to_numpy()
    ctx = t[CONTEXT].copy()
    cand = cand.reindex(rows)
    for c in cats:
        cand[c] = cand[c].astype("string").fillna("__NA__").astype(str)
    Xc = ctx.join(cand.add_prefix("cand_"))
    cc = ["apt"] + ["cand_" + c for c in cats]
    import hashlib
    tag = hashlib.sha1(repr((sorted(PARAMS.items()), CONTEXT, len(rows), V37_RUN)).encode()).hexdigest()[:10]
    ckey = paths.CACHE / "lab" / f"harness_control_{tag}.npy"
    if ckey.exists():
        pc = np.load(ckey)
    else:
        pc = _cv(ctx, r, fold, ["apt"])
        np.save(ckey, pc)
    pn = _cv(Xc, r, fold, cc)
    y, v = t["y"].to_numpy(), t["v37"].to_numpy()
    ex = fr[~fr["dayoffset"]]
    sse_all = float(((ex["v37"] - ex["y"]) ** 2).sum())
    sse_rows = float(((v - y) ** 2).sum())
    n = len(ex)

    def holdout(pred):
        return np.sqrt((sse_all - sse_rows + float(((pred - y) ** 2).sum())) / n)

    res = {"name": name, "n_rows": int(len(rows)), "coverage": float(cand.notna().any(axis=1).mean()),
           "holdout_v37": holdout(v), "holdout_control": holdout(v + pc), "holdout_candidate": holdout(v + pn)}
    res["gain_vs_control"] = res["holdout_candidate"] - res["holdout_control"]
    month = t["month"].to_numpy()
    for mo in (1, 7):
        s = month == mo
        res[f"rows_rmse_m{mo}_control"] = float(np.sqrt(((v + pc - y)[s] ** 2).mean()))
        res[f"rows_rmse_m{mo}_candidate"] = float(np.sqrt(((v + pn - y)[s] ** 2).mean()))
    apt = t["apt"].to_numpy()
    res["per_airport_delta"] = {a: round(float(np.sqrt(((v + pn - y)[apt == a] ** 2).mean()) -
                                               np.sqrt(((v + pc - y)[apt == a] ** 2).mean())), 2)
                                for a in sorted(set(apt))}
    if verbose:
        print(f"[{name}] rows {res['n_rows']:,}, candidate non-null {res['coverage']:.2f}")
        print(f"  holdout ex_wd: v37c {res['holdout_v37']:.3f}  control {res['holdout_control']:.3f}  "
              f"candidate {res['holdout_candidate']:.3f}  (gain vs control {res['gain_vs_control']:+.3f})")
        print(f"  covered rows Jan {res['rows_rmse_m1_control']:.2f} -> {res['rows_rmse_m1_candidate']:.2f}, "
              f"Jul {res['rows_rmse_m7_control']:.2f} -> {res['rows_rmse_m7_candidate']:.2f}")
        print(f"  per airport (candidate - control, rows RMSE): {res['per_airport_delta']}")
    return res


if __name__ == "__main__":
    if "--build" in sys.argv:
        build()
    else:
        cats = sys.argv[sys.argv.index("--cats") + 1].split(",") if "--cats" in sys.argv else []
        test(pd.read_parquet(sys.argv[1]), cats=cats, name=Path(sys.argv[1]).stem)
