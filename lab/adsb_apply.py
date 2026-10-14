"""Apply the ADS-B own-movement correction (family B) to a finished run and write a submittable run.

Train: the Jan+Jul 2025 holdout departures on ADS-B days that ADS-B covers (linked to a takeoff, with a first ground
sighting), residual of the base run's out-of-fold prediction (clipped to the NM join window), CatBoost on family B plus
the base, the regime probability, the window, NM clocks and the airport/stand/runway keys. Rows with a stale
probability above `MAX_P_STALE` are left alone: ADS-B shows where the aircraft was, not whether the airport recorded
the schedule. Apply: the 2026 departures ADS-B covers, residual added to the base run's ranking prediction, clipped to
the window again. Every other row keeps the base.

The holdout numbers it prints are cross-fitted by day (5 folds); the 2026 model is fitted on every covered holdout row.

Operational availability: tier M (challenge-only). The correction reads the flight's own pushback and taxi, which
exist only after its off-block; it reconstructs the recorded off-block for the challenge and is not a pre-movement
prediction (see docs/feature_table.md for the tiers).

Usage: python lab/adsb_apply.py <base_run_id> --write NAME [--weight W]
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prc import data, paths  # noqa: E402
from prc.postprocess import clip_to_bounds, lobt_bounds  # noqa: E402

FAM_B = ["adsb_takeoff_dt", "adsb_first_seen", "adsb_first_move", "adsb_taxi_start", "adsb_hold_s",
         "adsb_n_ground", "adsb_max_gap", "adsb_path_m", "adsb_first_dist_stand",
         "adsb_parked_until", "adsb_parked_dur", "adsb_link_by_time"]
NM = ["aobt3_proxy", "lobt_delta", "sched_delta", "eobt1_delta"]
CATS = ["apt", "stand", "runway"]
MAX_P_STALE = 0.05
PARAMS = dict(iterations=2000, learning_rate=0.05, depth=6, loss_function="RMSE", verbose=0, thread_count=12)
import os  # noqa: E402
PARAMS["train_dir"] = str(paths.CACHE / "catboost_info" / f"adsb_{os.getpid()}")
Path(PARAMS["train_dir"]).mkdir(parents=True, exist_ok=True)  # CatBoost creates only the last level itself
"""One CatBoost log directory per process: two stage fits sharing the default `catboost_info` wrote the same files and
stalled each other for 12 h on 3-4 October (findings 10.18; review of 4 October)."""
for _flag, _key, _cast in (("--depth", "depth", int), ("--iters", "iterations", int), ("--lr", "learning_rate", float)):
    if _flag in sys.argv:
        PARAMS[_key] = _cast(sys.argv[sys.argv.index(_flag) + 1])
if "--gpu" in sys.argv:  # screening only: GPU fits are not deterministic
    PARAMS.update(task_type="GPU", devices="0")
    PARAMS.pop("thread_count")
if "--l2" in sys.argv:
    PARAMS["l2_leaf_reg"] = float(sys.argv[sys.argv.index("--l2") + 1])
SEEDS = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 1
SEED0 = 42 + (int(sys.argv[sys.argv.index("--seed-offset") + 1]) if "--seed-offset" in sys.argv else 0)
"""Base random seed of every CatBoost fit (42 by default); --seed-offset N gives a seed-varied copy of a member for averaging."""
RESID_CLIP = float(sys.argv[sys.argv.index("--resid-clip") + 1]) if "--resid-clip" in sys.argv else 3600.0
DIRECT_CLIP = float(sys.argv[sys.argv.index("--direct-clip") + 1]) if "--direct-clip" in sys.argv else 7200.0
"""Training-target clips: the residual (y - base) is clipped to +-RESID_CLIP, the direct target to [0, DIRECT_CLIP].
Defaults are the 62-day optimum's settings; the review of 4 October asked for +-7,200 / 10,800 at the larger training
size (21 covered holdout rows with a true residual above 3,600 s hold 3.3 % of the ex_wd squared error)."""
FOLD_SEED = int(sys.argv[sys.argv.index("--fold-seed") + 1]) if "--fold-seed" in sys.argv else 0
CURVE = int(sys.argv[sys.argv.index("--curve") + 1]) if "--curve" in sys.argv else 0
"""Learning-curve check: fit cross-fit fold 0 for CURVE rounds with the fold's held-out days as an eval set, print
the held-out RMSE at milestones and the best iteration, then stop (nothing is written). Shows whether the stage's
fixed round count stops too early at the current training size."""
XGB = "--xgb" in sys.argv
"""Review item 7 (findings 10.20): the same stage with an XGBoost learner instead of CatBoost, for
learner-family diversity in the blend. Categoricals use XGBoost's native categorical splits with the training frame's
levels frozen (unseen levels become missing). Depth, rounds and learning rate come from --depth/--iters/--lr; --gpu
selects CUDA."""


class XGBModel:
    def __init__(self, seed):
        self.seed, self.booster, self.levels = seed, None, {}

    def _prep(self, X, cats, fit):
        X = X.copy()
        for c in X.columns:
            if c in cats:
                v = X[c].astype("string").astype(object)
                if fit:
                    self.levels[c] = pd.Index(sorted(set(v.dropna())))
                X[c] = pd.Categorical(v, categories=self.levels[c])
            elif X[c].dtype.kind not in "fiub" or str(X[c].dtype) in ("Int8", "Int16", "Int32", "Int64", "boolean"):
                X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
        return X

    def fit(self, X, y, w, cats):
        import xgboost as xgb
        dtr = xgb.QuantileDMatrix(self._prep(X, cats, True), y, weight=w, enable_categorical=True, max_bin=256)
        params = dict(tree_method="hist", device="cuda" if "--gpu" in sys.argv else "cpu", max_depth=PARAMS["depth"],
                      learning_rate=PARAMS["learning_rate"], subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                      reg_lambda=3.0, max_cat_to_onehot=16, objective="reg:squarederror", seed=self.seed, nthread=12)
        self.booster = xgb.train(params, dtr, num_boost_round=PARAMS["iterations"])
        return self

    def predict(self, X, cats):
        import xgboost as xgb
        return self.booster.predict(xgb.DMatrix(self._prep(X, cats, False), enable_categorical=True))
"""Day-to-fold assignment of the cross-fit. Default 0: the sorted days interleaved (day i -> fold i % 5), identical in
every run so that two runs differ only in what was changed. A non-zero seed permutes the days before the interleave,
to measure how much the holdout numbers move with the fold assignment alone (the 2026 model is unaffected: it is fitted
on every covered day)."""


WX = "--wx" in sys.argv
"""Weather inputs for the correction (tier P/T, challenge-only as anchored at takeoff): the base matrix's ERA5 columns
(snow depth, snowfall over 3 h and 24 h, freezing precipitation, ground temperature, frost hours) and the hours since the
airport-day's first ERA5 snowfall hour (findings 10.14: the de-icing uplift peaks in the 6 h after the first snowfall and
halves by 12 h). The residual correction had no weather input before; the direct model's rich matrix has the ERA5 columns."""
_WX = None
SRC = None
"""Per-row source of the base OOF when the base run mixes the seasonal stack (0) with winter-split members (1): the
combined base of `lab/build_winter_base.py`. Becomes the model input `base_src`; metrics are reported for the Jan+Jul
days (comparable with earlier runs) and the extra days separately."""
DIRECT = "--direct" in sys.argv
"""Second formulation: fit the taxi time itself (clipped to [0, 7200] s) instead of the residual of the base; the base
stays an input. The prediction is turned into an implied correction (prediction - base) so the rest is unchanged.
Not combined with --regime (hedged rows keep the base); blend with a residual-correction run (`lab/blend_direct.py`)."""
DROP_PREFIX = tuple(sys.argv[sys.argv.index("--drop-prefix") + 1].split(",")) if "--drop-prefix" in sys.argv else ()
V5 = "--v5" in sys.argv
"""V4 plus the fourth feature set (`lab/adsb_features4.py`: inbound/turnaround, stand-type archive, time-to-location
map; tier M)."""
V5_CATS = ["it_end_kind", "sa_best_src"]
V4 = "--v4" in sys.argv or V5
"""V2 plus the third feature set (`lab/adsb_features3.py`: pushback detector, first-reception state, track quality,
stop classes; tier M). `--v4-only FAMILY[,FAMILY]` keeps only those families (pushback, first_reception,
track_quality, stop_classes) for ablations."""
V4_FAMILIES = {"pushback": "pb_", "first_reception": "fr_", "track_quality": "adsbQ_", "stop_classes": ("ll_", "deice_", "dz_", "rwy_queue", "park2")}
V4_CATS = ["pb_anchor_src", "ll_cls"]
V3 = "--v3" in sys.argv
"""V2 plus the routed stand-to-runway distance (`taxi_route` block) and the part of it ADS-B did not see."""
V2 = "--v2" in sys.argv or V3 or V4
"""Also the second feature set (`lab/adsb_features2.py`: the stand and the stops) and the signed takeoff offset."""
FAM_B2 = ["adsb_stop_end_300", "adsb_stop_end_600", "adsb_stop_end_1200", "adsb_left_stand_50", "adsb_left_stand_100",
          "adsb_n_near_stand", "adsb_stand_stop_end", "adsb_n_stops_after", "adsb_stop_s_after", "adsb_max_stop_after",
          "adsb_tracked_from", "adsb_takeoff_offset"]
TIMING = ["adsb_first_seen", "adsb_first_move", "adsb_taxi_start", "adsb_parked_until", "adsb_stop_end_300",
          "adsb_stop_end_600", "adsb_stop_end_1200", "adsb_left_stand_50", "adsb_left_stand_100", "adsb_stand_stop_end",
          "adsb_tracked_from"]
FIT_FIX = "--fit-fix" in sys.argv
"""Fit the final (2026) model on the rows the cross-fitted models were fitted on (covered, stale probability <= 0.05).
Without it the final fit also takes the hedged rows that `--regime` adds to `cov` (their residual is against the
mixture prediction, stale or taxi), which the validated cross-fitted models never saw (v36, v37c)."""
CVH = "--cv-with-hedged" in sys.argv
"""Diagnostic: fit the cross-fitted models also on the hedged rows, as the final fit does without --fit-fix, to measure
on the holdout what the shipped 2026 model's training set costs."""
MIXW = "--mixweight" in sys.argv
"""Weight the correction's training rows per airport by (2026 covered departures / 2025 covered departures)."""
RICH = "--rich" in sys.argv
REGIME = "--regime" in sys.argv
"""Also correct hedged rows (stale probability above MAX_P_STALE) through their taxi regime only:
pred += (1 - p) * correction(taxi), the correction computed with the regressor's taxi answer as the base."""
RICH_X = None
LAB_MATRIX = paths.CACHE / "lab" / (sys.argv[sys.argv.index("--lab-matrix") + 1] if "--lab-matrix" in sys.argv else "lean2")
"""The lab matrix (lab/build_lab_matrix.py) the stage reads the NM clocks, the ERA5 columns and the rich columns from:
lean2 for the delivered set, or the one built for another tree (--lab-matrix lean2__<version>, counts rescaled)."""
NO_CV = "--no-cv" in sys.argv
"""Skip the day cross-fit (and the regime cross-fit) and its holdout read-out: fit only the final models on every
covered day. For the final dataset, whose models need no new holdout estimate; --write then needs --oof-from."""
SAVE_MODELS = "--save-models" in sys.argv
"""Save the final models into the written run (`stage_models/`: one CatBoost file per seed and `stage_models.json`
with the column order, the categoricals and the flags), so `--load-models` can apply them to another ranking set."""
LOAD_MODELS = sys.argv[sys.argv.index("--load-models") + 1] if "--load-models" in sys.argv else None
"""A run written with --save-models: its models predict this base run's ranking rows; nothing is fitted. The flags
that shape the prediction frame must equal the saved ones; --write needs --oof-from."""
OOF_FROM = sys.argv[sys.argv.index("--oof-from") + 1] if "--oof-from" in sys.argv else None
"""A finished stage run whose holdout output (oof_predictions.parquet and the holdout rows of corrected_rows.parquet)
the written run carries instead of its own: with --no-cv or --load-models there is no cross-fit, and the steps after
the stage (base_swap.py, forced_long_shift.py) estimate on the holdout rows, so carrying the cross-fitted member's OOF
keeps every holdout-estimated parameter of the submitted recipe."""
FRAME_FLAGS = ("--direct", "--rich", "--regime", "--v2", "--v3", "--v4", "--v5", "--wx")
"""Flags that change the prediction frame or how a prediction is turned into a correction (checked on --load-models);
--drop-prefix and --v4-only are compared by value. Target clips, weights and --fit-fix change only the fit."""


def frame_flags() -> dict:
    out = {f: f in sys.argv for f in FRAME_FLAGS}
    for f in ("--drop-prefix", "--v4-only"):
        out[f] = sys.argv[sys.argv.index(f) + 1] if f in sys.argv else None
    return out


def save_stage_models(dst: Path, models, cols, cats_all, base_run: str, n_fit: int) -> dict:
    """Write the final models and their contract into dst/stage_models; returns the contract."""
    out = dst / "stage_models"
    out.mkdir()
    files = []
    for k, m in enumerate(models):
        f = out / f"model_{k}.cbm"
        m.save_model(str(f))
        files.append({"file": f.name, "sha1": hashlib.sha1(f.read_bytes()).hexdigest()})
    meta = {"columns": list(cols), "cat_features": list(cats_all), "frame_flags": frame_flags(),
            "params": {k: v for k, v in PARAMS.items() if k != "train_dir"}, "seeds": SEEDS, "seed0": SEED0,
            "base_run": base_run, "n_fit_rows": int(n_fit), "models": files}
    (out / "stage_models.json").write_text(json.dumps(meta, indent=1))
    return meta


def load_stage_models(run_id: str):
    """The models, column order and categoricals a --save-models run wrote; refuses another frame or a changed file."""
    src = paths.run_dir(run_id) / "stage_models"
    meta = json.loads((src / "stage_models.json").read_text())
    if meta["frame_flags"] != frame_flags():
        raise SystemExit(f"--load-models {run_id}: the frame flags differ from the saved ones\n  saved {meta['frame_flags']}\n"
                         f"  given {frame_flags()}")
    models = []
    for e in meta["models"]:
        f = src / e["file"]
        if hashlib.sha1(f.read_bytes()).hexdigest() != e["sha1"]:
            raise SystemExit(f"--load-models {run_id}: {f.name} changed since it was saved")
        m = CatBoostRegressor()
        m.load_model(str(f))
        models.append(m)
    print(f"loaded {len(models)} model(s) from {run_id} (fitted on {meta['n_fit_rows']:,} rows of {meta['base_run']})")
    return models, meta["columns"], meta["cat_features"], meta


def rich_columns(index):
    """The main model's whole feature matrix (lean2), categoricals as strings, for the rich variant."""
    global RICH_X
    if RICH_X is None:
        meta = json.loads((LAB_MATRIX / "meta.json").read_text())
        X = pd.read_parquet(LAB_MATRIX / "X.parquet")
        for c in meta["cat_cols"]:
            X[c] = X[c].astype("string").fillna("__NA__").astype(str)
        for c in X.columns:
            if c not in meta["cat_cols"] and str(X[c].dtype) in ("Int8", "Int16", "Int32", "Int64"):
                X[c] = X[c].astype("float32")
        RICH_X = (X, [c for c in meta["cat_cols"]])
    X, cats = RICH_X
    drop = set(NM) | {"apt", "stand", "runway"}
    keep = [c for c in X.columns if c not in drop]
    return X.reindex(index)[keep].add_prefix("x_"), ["x_" + c for c in cats if c not in drop]


_ROUTE = None
_F3 = None


def wx_table(index):
    """ERA5 columns of the base matrix plus hours since the day's first snowfall (NaN on days without snowfall)."""
    global _WX
    if _WX is None:
        X = pd.read_parquet(LAB_MATRIX / "X.parquet")
        era_cols = [c for c in X.columns if c.startswith("era5_")]
        _WX = X[era_cols].copy()
        era = pd.read_parquet(paths.ERA5_FILE); era["apt"] = era["apt"].astype(str)
        era["t"] = pd.to_datetime(era["time"]); era["day"] = era["t"].dt.strftime("%Y-%m-%d")
        first = era[era["snowfall_cm"] > 0.05].groupby(["apt", "day"])["t"].min()
        mv = data.load_movements(columns=["apt", "MVT_TIME_UTC_mvt"]).loc[_WX.index]
        key = pd.MultiIndex.from_arrays([mv["apt"].astype(str), mv["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d")])
        t_first = first.reindex(key).to_numpy()
        tk = mv["MVT_TIME_UTC_mvt"].dt.tz_localize(None) if mv["MVT_TIME_UTC_mvt"].dt.tz is not None else mv["MVT_TIME_UTC_mvt"]
        tf = pd.to_datetime(pd.Series(t_first, index=_WX.index)); tf = tf.dt.tz_localize(None) if tf.dt.tz is not None else tf
        _WX["wx_since_first_snow_h"] = ((tk.to_numpy() - tf.to_numpy()) / np.timedelta64(1, "h")).astype("float64")
    return _WX.reindex(index)


def features3():
    """The third ADS-B feature set, restricted to the families named by --v4-only."""
    global _F3
    if _F3 is None:
        f = pd.read_parquet(paths.CACHE / "adsb" / "features3.parquet")
        if "--v4-only" in sys.argv:
            fams = sys.argv[sys.argv.index("--v4-only") + 1].split(",")
            pre = tuple(p for fam in fams for p in (V4_FAMILIES[fam] if isinstance(V4_FAMILIES[fam], tuple) else (V4_FAMILIES[fam],)))
            f = f[[c for c in f.columns if c.startswith(pre)]]
        _F3 = f
    return _F3


def route_table():
    """The routed stand-to-runway distance and runway crossings of every departure (latest `taxi_route` cache)."""
    global _ROUTE
    if _ROUTE is None:
        f = max((paths.CACHE / "features").glob("*/taxi_route__*.parquet"), key=lambda q: q.stat().st_mtime)
        _ROUTE = pd.read_parquet(f, columns=["geo_route_dist_m", "geo_route_rwy_crossings"])
    return _ROUTE


def frame(index, base, p_stale, adsb, X, mvt):
    d = pd.DataFrame(index=index)
    d["base"] = base
    d["p_stale"] = p_stale
    lo, hi = lobt_bounds(mvt, index)
    d["win_lo"], d["win_hi"] = lo, hi
    feats = FAM_B + (FAM_B2 if V2 else [])
    d = d.join(X.reindex(index)[NM]).join(adsb.reindex(index)[[c for c in feats if c in adsb.columns]])
    d["apt"] = mvt.loc[index, "apt"].astype(str)
    d["stand"] = mvt.loc[index, "STAND_mvt"].astype("string").fillna("__NA__").astype(str)
    d["runway"] = mvt.loc[index, "RUNWAY_mvt"].astype("string").fillna("__NA__").astype(str)
    for c in ("adsb_first_seen", "adsb_first_move", "adsb_taxi_start"):
        if c in d:
            d[c + "_m_proxy"] = d[c] - d["aobt3_proxy"]
            d[c + "_m_base"] = d[c] - d["base"]
    derive(d)
    if V3:
        d = d.join(route_table().reindex(index))
        derive(d)
    if V5:
        d = d.join(pd.read_parquet(paths.CACHE / "adsb" / "features4.parquet").reindex(index))
        for c in V5_CATS:
            if c in d:
                d[c] = d[c].astype("string").fillna("__NA__").astype(str)
    if WX:
        d = d.join(wx_table(index))
    if V4:
        d = d.join(features3().reindex(index))
        for c in V4_CATS:
            if c in d:
                d[c] = d[c].astype("string").fillna("__NA__").astype(str)
        derive(d)
    if RICH:
        extra, _ = rich_columns(index)
        d = d.join(extra)
    d["base_src"] = SRC.reindex(index).fillna(0).astype("int8").to_numpy() if SRC is not None else 0
    if DROP_PREFIX:  # ablation: leave out columns (and their derived gaps) by name prefix
        d = d[[c for c in d.columns if not c.startswith(DROP_PREFIX)]]
    covered = (adsb.reindex(index)["adsb_linked"].fillna(0).to_numpy() > 0) & d["adsb_first_seen"].notna().to_numpy()
    covered &= np.nan_to_num(d["p_stale"].to_numpy()) <= MAX_P_STALE
    return d, lo, hi, covered


def derive(d):
    """V2: each timing as an estimate of the taxi (seconds before the recorded takeoff), and its gap to the base."""
    if not V2:
        return
    if V4:
        for c in [c for c in d.columns if c.endswith("_est") or c.startswith(("tmap_est", "tarr_est"))]:
            d[c + "_m_base"] = d[c] - d["base"]
        for c in [c for c in d.columns if c.endswith("_tk")]:
            d[c + "_m_base"] = d[c] - d["adsb_takeoff_offset"] - d["base"]
    if V3 and "geo_route_dist_m" in d:
        d["route_unseen_m"] = d["geo_route_dist_m"] - d["adsb_path_m"].fillna(0)
    for c in TIMING:
        if c in d:
            d[c + "_obs_m_base"] = d[c] - d["adsb_takeoff_offset"] - d["base"]


def taxi_regime_rows(frame_fn, index, y_reg, p, adsb, X, mvt):
    """Frame for hedged covered rows with the regressor's taxi answer as the base and the stale probability set to 0."""
    d, lo, hi, _ = frame_fn(index, y_reg, np.zeros(len(index)), adsb, X, mvt)
    d["base"] = clip_to_bounds(d["base"].to_numpy(), lo, hi)
    for c in ("adsb_first_seen", "adsb_first_move", "adsb_taxi_start"):
        d[c + "_m_base"] = d[c] - d["base"]
    derive(d)
    linked = (adsb.reindex(index)["adsb_linked"].fillna(0).to_numpy() > 0) & d["adsb_first_seen"].notna().to_numpy()
    hedged = linked & (np.nan_to_num(p) > MAX_P_STALE) & np.isfinite(np.asarray(y_reg, dtype="float64"))
    return d, hedged


def main():
    base_run = sys.argv[1]
    if "--write" in sys.argv and (NO_CV or LOAD_MODELS) and not OOF_FROM:
        raise SystemExit("--no-cv / --load-models carry no holdout estimate of their own: --write needs --oof-from RUN")
    if SAVE_MODELS and ("--write" not in sys.argv or XGB or LOAD_MODELS):
        raise SystemExit("--save-models needs --write and a CatBoost fit here (not --xgb, not --load-models)")
    if LOAD_MODELS and (XGB or CURVE):
        raise SystemExit("--load-models applies saved CatBoost models; no --xgb, no --curve")
    weight = float(sys.argv[sys.argv.index("--weight") + 1]) if "--weight" in sys.argv else 1.0
    adsb = pd.concat([pd.read_parquet(f) for f in sorted((paths.CACHE / "adsb").glob("features_*.parquet"))])
    adsb = adsb[~adsb.index.duplicated(keep="last")]
    if V2:
        a2 = pd.concat([pd.read_parquet(f) for f in sorted((paths.CACHE / "adsb").glob("features2_*.parquet"))])
        adsb = adsb.join(a2[~a2.index.duplicated(keep="last")].drop(columns=[c for c in a2.columns if c in adsb.columns]))
    mvt = data.load_movements(columns=["apt", "MVT_TIME_UTC_mvt", "LOBT_flt", "STAND_mvt", "RUNWAY_mvt"])
    X = pd.read_parquet(LAB_MATRIX / "X.parquet", columns=NM)
    src = paths.run_dir(base_run)
    oof = pd.read_parquet(src / "oof_predictions.parquet")
    test = pd.read_parquet(src / "test_predictions.parquet")
    global SRC
    SRC = oof["src"].astype("int8") if "src" in oof else None
    if SRC is not None:
        print(f"combined base: {int((SRC == 0).sum()):,} stack rows + {int((SRC == 1).sum()):,} winter rows")

    # p_stale for the ranking rows: averaged refit details of the leaf runs when the base is a stack/blend
    from final_stack import averaged_detail, leaf_runs  # noqa: E402
    leaves = leaf_runs(base_run.replace("__stack", "")) if False else None
    rank_p = pd.Series(0.0, index=test.index)
    try:
        meta = json.loads((src / "metrics.json").read_text(encoding="utf-8"))
        source = meta.get("stack", {}).get("source", base_run)
        rank_det = averaged_detail(leaf_runs(source), "rank")
        oof_det = averaged_detail(leaf_runs(source), "oof")
        rank_p = rank_det["p_stale"].reindex(test.index).fillna(0.0)
        oof_p = oof_det["p_stale"].reindex(oof.index)
        oof_p = (oof_p.fillna(oof["p_stale"]) if "p_stale" in oof else oof_p).fillna(0.0)
        rank_reg = rank_det["y_reg"].reindex(test.index)
        oof_reg = oof_det["y_reg"].reindex(oof.index)
        if "y_reg" in oof:
            oof_reg = oof_reg.fillna(oof["y_reg"])
    except Exception as exc:  # noqa: BLE001
        print("no averaged p_stale, using the OOF column:", exc)
        oof_p = oof["p_stale"] if "p_stale" in oof else pd.Series(0.0, index=oof.index)

    loaded = None
    if LOAD_MODELS is None:
        rows = oof.index.intersection(adsb.index)
        d, lo, hi, cov = frame(rows, oof.loc[rows, "y_pred"].to_numpy(), oof_p.reindex(rows).to_numpy(), adsb, X, mvt)
        d["base"] = clip_to_bounds(d["base"].to_numpy(), lo, hi)
        for c in ("adsb_first_seen", "adsb_first_move", "adsb_taxi_start"):
            d[c + "_m_base"] = d[c] - d["base"]
        derive(d)
        y = oof.loc[rows, "y_true"].to_numpy()
        resid = np.clip(y - d["base"].to_numpy(), -RESID_CLIP, RESID_CLIP)
        if DIRECT:
            assert not REGIME, "--direct is not combined with --regime"
            resid = np.clip(y, 0, DIRECT_CLIP)  # the target itself; predictions become prediction - base below
        cols = [c for c in d.columns]
        cats_all = CATS + (rich_columns(d.index[:1])[1] if RICH else []) + ([c for c in V4_CATS if c in d.columns] if V4 else []) + ([c for c in V5_CATS if c in d.columns] if V5 else [])
    else:  # the saved models of an earlier --save-models run: nothing is fitted
        models, cols, cats_all, loaded = load_stage_models(LOAD_MODELS)

    def new_model(seed):
        return XGBModel(seed) if XGB else CatBoostRegressor(**PARAMS, random_seed=seed)

    def fit_model(m, X, yy, ww):
        return m.fit(X, yy, ww, cats_all) if XGB else m.fit(Pool(X, yy, cat_features=cats_all, weight=ww))

    def predict_model(m, X):
        return m.predict(X, cats_all) if XGB else m.predict(Pool(X, cat_features=cats_all))

    if LOAD_MODELS is None:
        day = mvt.loc[rows, "MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d")
        days_sorted = sorted(day.unique())
        if FOLD_SEED:
            days_sorted = [days_sorted[i] for i in np.random.default_rng(FOLD_SEED).permutation(len(days_sorted))]
        fold = day.map({dd: i % 5 for i, dd in enumerate(days_sorted)}).to_numpy()
        cov_fit = cov.copy()
        W = np.ones(len(d))
        if MIXW:
            lk_all = (adsb["adsb_linked"].fillna(0) > 0) & adsb["adsb_first_seen"].notna()
            rk = test.index.intersection(adsb.index)
            n26 = pd.Series(lk_all.reindex(rk).to_numpy(), index=rk).groupby(mvt.loc[rk, "apt"].astype(str)).sum()
            apt_d = d["apt"].to_numpy()
            n25 = pd.Series(cov).groupby(apt_d).sum()
            ratio = (n26 / n25).replace([np.inf], np.nan).fillna(1.0)
            W = pd.Series(apt_d).map(ratio).fillna(1.0).to_numpy(dtype="float64")
            W = W / W[cov].mean()
            print("2026-mix weights per airport (mean 1 over covered rows):",
                  {a: round(float(W[cov & (apt_d == a)].mean()), 2) for a in sorted(set(apt_d[cov]))})
        fit_rows = cov
        if CVH and REGIME:
            _, hedged0 = taxi_regime_rows(frame, rows, oof_reg.reindex(rows).to_numpy(), oof_p.reindex(rows).to_numpy(), adsb, X, mvt)
            fit_rows = cov | hedged0
            print(f"cross-fitted models also fitted on {int((hedged0 & ~cov).sum()):,} hedged rows")
        if CURVE:
            tr, te = (fold != 0) & fit_rows, (fold == 0) & cov
            cp = dict(PARAMS); cp["iterations"] = CURVE; cp["metric_period"] = 100
            m = CatBoostRegressor(**cp, random_seed=SEED0)
            m.fit(Pool(d.loc[tr, cols], resid[tr], cat_features=cats_all, weight=W[tr]),
                  eval_set=Pool(d.loc[te, cols], resid[te], cat_features=cats_all), use_best_model=False)
            ev = m.get_evals_result()
            key = [k for k in ev if k != "learn"][0]
            curve_rmse = np.asarray(ev[key]["RMSE"], dtype=float)
            print(f"learning curve, fold 0: {tr.sum():,} fit rows, {te.sum():,} held-out covered rows, params {cp}")
            for it in (500, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 15000, 20000):
                if it <= len(curve_rmse):
                    print(f"  round {it:>6}: held-out RMSE {curve_rmse[it - 1]:.3f}")
            print(f"  best round {int(np.argmin(curve_rmse)) + 1} at {curve_rmse.min():.3f}; final {curve_rmse[-1]:.3f}")
            return
        cv = np.zeros(len(d))
        for k in range(5 if not NO_CV else 0):  # --no-cv: no cross-fit, only the final models below
            tr, te = (fold != k) & fit_rows, (fold == k) & cov
            if te.any():
                for sd in range(SEEDS):
                    m = new_model(SEED0 + sd)
                    fit_model(m, d.loc[tr, cols], resid[tr], W[tr])
                    pte = predict_model(m, d.loc[te, cols])
                    cv[te] += (pte - d["base"].to_numpy()[te] if DIRECT else pte) / SEEDS
        hedge_add = np.zeros(len(d))
        if REGIME:
            dt, hedged = taxi_regime_rows(frame, rows, oof_reg.reindex(rows).to_numpy(), oof_p.reindex(rows).to_numpy(), adsb, X, mvt)
            pr = np.nan_to_num(oof_p.reindex(rows).to_numpy())
            for k in range(5 if not NO_CV else 0):
                tr, te = (fold != k) & fit_rows, (fold == k) & hedged
                if te.any():
                    m = new_model(SEED0)
                    fit_model(m, d.loc[tr, cols], resid[tr], W[tr])
                    hedge_add[te] = (1 - pr[te]) * predict_model(m, dt.loc[te, cols])
            print(f"taxi-regime {'correction' if not NO_CV else 'rows (final fit)'} on {hedged.sum():,} hedged covered holdout rows")
            cv = cv + hedge_add
            cov = cov | hedged
        if not NO_CV:  # the holdout read-out of the cross-fit
            wd = oof.loc[rows, "dayoffset"].to_numpy(bool)
            jj = np.isin(mvt.loc[rows, "MVT_TIME_UTC_mvt"].dt.month.to_numpy(), (1, 7))
            curve = [("base", d["base"].to_numpy())] + [(f"corrected x{w}", clip_to_bounds(d["base"].to_numpy() + w * cv, lo, hi)) for w in (0.8, 1.0, 1.2)]
            for label, mk in (("holdout ADS-B days", jj), ("extra (winter) days", ~jj)):
                if not mk.any():
                    continue
                for name, p in curve:
                    e = (p - y) ** 2
                    print(f"{label} {name:<16} rmse {np.sqrt(e[mk].mean()):.2f}  ex_wd {np.sqrt(e[mk & ~wd].mean()):.2f}  "
                          f"bulk {np.sqrt(e[mk & (y < 3600)].mean()):.2f}  covered {np.sqrt(e[mk & cov].mean()):.2f} (n {(mk & cov).sum():,} of {mk.sum():,})")
            # projected 2026: holdout MSE (without wrong-date rows) per airport x covered cell, weighted to the 2026 cell counts
            lk_all = (adsb["adsb_linked"].fillna(0) > 0) & adsb["adsb_first_seen"].notna()
            rk = test.index
            cell26 = pd.DataFrame({"apt": mvt.loc[rk, "apt"].astype(str).to_numpy(),
                                   "cov": lk_all.reindex(rk).fillna(False).to_numpy(bool)}).value_counts()
            cell25 = pd.DataFrame({"apt": d["apt"].to_numpy(), "cov": lk_all.reindex(rows).fillna(False).to_numpy(bool)})
            for name, p in curve[:2] + curve[2:3]:
                e = pd.Series((p - y) ** 2)
                keep25 = (~wd) & jj
                mse = e[keep25].groupby([cell25["apt"][keep25], cell25["cov"][keep25]]).mean()
                proj = np.sqrt(float((mse.reindex(cell26.index).fillna(mse.mean()) * cell26).sum() / cell26.sum()))
                print(f"projected 2026 {name:<16} {proj:.2f}")

        models = []
        for sd in range(SEEDS):
            mm = new_model(SEED0 + sd)
            fm = cov_fit if FIT_FIX else cov
            fit_model(mm, d.loc[fm, cols], resid[fm], W[fm])
            models.append(mm)
        n_fit = int((cov_fit if FIT_FIX else cov).sum())
    rrows = test.index.intersection(adsb.index)
    r, rlo, rhi, rcov = frame(rrows, test.loc[rrows, "y_pred"].to_numpy(), rank_p.reindex(rrows).to_numpy(), adsb, X, mvt)
    r["base"] = clip_to_bounds(r["base"].to_numpy(), rlo, rhi)
    for c in ("adsb_first_seen", "adsb_first_move", "adsb_taxi_start"):
        r[c + "_m_base"] = r[c] - r["base"]
    derive(r)
    missing = [c for c in cols if c not in r.columns]
    if missing:
        raise SystemExit(f"the ranking frame lacks {len(missing)} model column(s): {missing[:8]}")
    corr = np.zeros(len(r))
    corr[rcov] = np.mean([predict_model(mm, r.loc[rcov, cols]) for mm in models], axis=0)
    if DIRECT:
        corr[rcov] -= r["base"].to_numpy()[rcov]
    if REGIME:
        rt, rhedged = taxi_regime_rows(frame, rrows, rank_reg.reindex(rrows).to_numpy(), rank_p.reindex(rrows).to_numpy(), adsb, X, mvt)
        rp = np.nan_to_num(rank_p.reindex(rrows).to_numpy())
        if rhedged.any():
            corr[rhedged] = (1 - rp[rhedged]) * np.mean([predict_model(mm, rt.loc[rhedged, cols]) for mm in models], axis=0)
        print(f"taxi-regime correction on {rhedged.sum():,} hedged ranking rows")
        rcov = rcov | rhedged
    new_rank = test["y_pred"].copy()
    new_rank.loc[rrows] = clip_to_bounds(r["base"].to_numpy() + weight * corr, rlo, rhi)
    print(f"params {loaded['params'] if loaded else PARAMS}, seeds {loaded['seeds'] if loaded else SEEDS}")
    print(f"ranking rows on ADS-B days: {len(rrows):,}, corrected {rcov.sum():,}; mean correction {corr[rcov].mean():.1f} s, "
          f"rms {np.sqrt((corr[rcov] ** 2).mean()):.1f} s")

    if "--write" in sys.argv:
        name = sys.argv[sys.argv.index("--write") + 1]
        tag = hashlib.sha1(f"{src.name}|{name}".encode()).hexdigest()[:8]
        dst = paths.RUNS / f"{src.name.split('__')[0]}__{name}__{tag}"
        dst.mkdir(parents=True, exist_ok=False)
        for f in ("config.resolved.json", "env.json"):
            if (src / f).exists():
                shutil.copy(src / f, dst / f)
        rank_rows = pd.DataFrame({"kind": "rank"}, index=pd.Index(rrows[rcov], name="mvt_id"))
        if OOF_FROM:  # the cross-fitted member's holdout output, carried (no cross-fit here)
            osrc = paths.run_dir(OOF_FROM)
            shutil.copy(osrc / "oof_predictions.parquet", dst / "oof_predictions.parquet")
            held = pd.read_parquet(osrc / "corrected_rows.parquet")
            held = held[held["kind"].eq("oof")]
        else:
            o2 = oof[SRC == 0].copy().drop(columns=["src"]) if SRC is not None else oof.copy()
            newp = pd.Series(clip_to_bounds(d["base"].to_numpy() + weight * cv, lo, hi), index=rows)
            keep = o2.index.intersection(rows)
            o2.loc[keep, "y_pred"] = newp.loc[keep].to_numpy()
            o2.to_parquet(dst / "oof_predictions.parquet")
            held = pd.DataFrame({"kind": "oof"}, index=pd.Index(rows[cov], name="mvt_id"))
        pd.DataFrame({"y_pred": new_rank}).to_parquet(dst / "test_predictions.parquet")
        pd.concat([held, rank_rows]).to_parquet(dst / "corrected_rows.parquet")
        saved = save_stage_models(dst, models, cols, cats_all, base_run, n_fit) if SAVE_MODELS else None
        # the resolved flag set and CatBoost parameters travel with the run (review 4 Oct: a typo in a hand-parsed flag
        # would otherwise change a submission's feature set without a trace)
        (dst / "metrics.json").write_text(json.dumps({"stack": {"source": base_run, "steps": ["adsb_own_movement"],
                                                                 "weight": weight, "n_rank_corrected": int(rcov.sum()),
                                                                 "argv": sys.argv[1:],
                                                                 "params": loaded["params"] if loaded else {k: v for k, v in PARAMS.items() if k != "train_dir"},
                                                                 "seeds": loaded["seeds"] if loaded else SEEDS,
                                                                 "seed0": loaded["seed0"] if loaded else SEED0, "fold_seed": FOLD_SEED,
                                                                 "learner": "xgboost" if XGB else "catboost", "no_cv": NO_CV,
                                                                 "oof_from": OOF_FROM, "models_from": LOAD_MODELS,
                                                                 "models_saved": bool(saved)}}, indent=1))
        print("wrote", dst.name)


if __name__ == "__main__":
    main()
