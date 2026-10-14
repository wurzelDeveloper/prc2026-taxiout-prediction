"""Lab: the final post-processing stack on a finished (blend) run, with a holdout check.

Steps, each optional:
  1. the NM join-window clip (always; `prc.postprocess`),
  2. regime exclusion for linked untyped rows (`lab/linked_untyped.py`): an unmatched departure's block lies
     outside NM's LOBT +- 3,606 s, and for untyped rows whose flight is NM-joined at its destination that LOBT
     is known, so the stale or the taxi regime can be ruled out. Applied as a change of the stale weight:
     pred += (p_new - p) * (delta - y_reg), with p and y_reg averaged over the leaf runs (OOF frames on the
     holdout, refit_detail.parquet on the ranking rows).

Writes a new run directory via `prc.blend.blend_runs` semantics is not needed: the output is a parquet of the
ranking predictions next to a holdout report, and `--write <name>` materialises it as a run (copying the source's
OOF with the same steps applied) so `prc submit` takes it.

Usage: python lab/final_stack.py <run_id> [--write NAME]
"""

from __future__ import annotations

import json
import shutil
import sys

import numpy as np
import pandas as pd

from common import ROOT
from linked_untyped import W, linked_frame

sys.path.insert(0, str(ROOT))
from prc import data, paths  # noqa: E402
from prc.postprocess import clip_to_bounds, lobt_bounds  # noqa: E402


def leaf_runs(run_id: str, weight: float = 1.0) -> list[tuple[str, float]]:
    """A blend's source runs down to the trained ones, with their effective weights."""
    m_path = paths.run_dir(run_id) / "metrics.json"
    blend = json.loads(m_path.read_text(encoding="utf-8")).get("blend") if m_path.exists() else None
    if not blend:
        return [(run_id, weight)]
    runs = blend["runs"]
    ws = blend.get("weights") or [1.0] * len(runs)
    ws = np.asarray(ws, dtype="float64") / np.sum(ws)
    out = []
    for r, w in zip(runs, ws):
        out.extend(leaf_runs(r, weight * float(w)))
    return out


def averaged_detail(leaves, kind: str) -> pd.DataFrame:
    """Weighted mean of p_stale and y_reg over the leaves: 'oof' (holdout) or 'rank' (refit_detail)."""
    acc_p = acc_r = None
    total = 0.0
    for run, w in leaves:
        f = paths.run_dir(run) / ("oof_predictions.parquet" if kind == "oof" else "refit_detail.parquet")
        d = pd.read_parquet(f, columns=["p_stale", "y_reg"])
        acc_p = d["p_stale"] * w if acc_p is None else acc_p + d["p_stale"].reindex(acc_p.index) * w
        acc_r = d["y_reg"] * w if acc_r is None else acc_r + d["y_reg"].reindex(acc_r.index) * w
        total += w
    return pd.DataFrame({"p_stale": acc_p / total, "y_reg": acc_r / total})


def exclusion(pred: pd.Series, detail: pd.DataFrame, links: pd.DataFrame, p_no_stale: float, p_no_gen: float) -> pd.Series:
    rows = links.index.intersection(pred.index)
    lk = links.loc[rows]
    p = detail.loc[rows, "p_stale"].to_numpy()
    reg = detail.loc[rows, "y_reg"].to_numpy()
    sd = lk["sd"].to_numpy()
    lob = lk["lob"].to_numpy()
    stale_ok = (sd < lob - W) | (sd > lob + W)
    gen_ok = (reg < lob - W) | (reg > lob + W)
    p_new = np.where(~stale_ok & gen_ok, p_no_stale, np.where(stale_ok & ~gen_ok, p_no_gen, p))
    out = pred.copy()
    out.loc[rows] = pred.loc[rows].to_numpy() + (p_new - p) * (np.maximum(sd, 0) - reg)
    return out


EDGES = [-1e9, 0, 300, 900, 1800, 3600, 1e9]


def inwindow_offsets(mvt: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Mean of (target - lower window edge) per band of (lower edge - NM proxy), from the fit months (not Jan/Jul)."""
    dep = mvt[mvt["is_dep"] & (mvt["split"] == "train")]
    lob = (dep["MVT_TIME_UTC_mvt"] - dep["LOBT_flt"]).dt.total_seconds()
    px = (dep["MVT_TIME_UTC_mvt"] - dep["AOBT_3_flt"]).dt.total_seconds()
    ok = lob.notna() & ~dep["MVT_TIME_UTC_mvt"].dt.month.isin([1, 7])
    lo = lob - W
    band = pd.cut((lo - px).fillna(-1e9 + 1), EDGES, labels=False)
    return (y.reindex(dep.index) - lo)[ok].groupby(band[ok]).mean()


def inwindow(pred: pd.Series, raw: pd.Series, mvt: pd.DataFrame, offsets: pd.Series, factor: float) -> pd.Series:
    """Rows whose unclipped prediction lies below the window's lower edge move to edge + factor * band offset."""
    rows = mvt.loc[pred.index]
    lob = (rows["MVT_TIME_UTC_mvt"] - rows["LOBT_flt"]).dt.total_seconds().to_numpy()
    px = (rows["MVT_TIME_UTC_mvt"] - rows["AOBT_3_flt"]).dt.total_seconds().to_numpy()
    lo, hi = lob - W, lob + W
    # raised rows: below the edge before clipping, or sitting exactly on it (a source already clipped)
    below = np.isfinite(lo) & ((raw.to_numpy() < lo) | (np.abs(raw.to_numpy() - lo) < 1e-3))
    band = pd.cut(pd.Series(np.where(np.isfinite(px), lo - px, -1e9 + 1)), EDGES, labels=False).to_numpy()
    add = np.nan_to_num(pd.Series(band).map(offsets).to_numpy()) * factor
    out = pred.copy()
    out[below] = np.minimum(lo[below] + add[below], hi[below])
    return out


def main():
    run = sys.argv[1]
    factor = float(sys.argv[sys.argv.index("--inwindow") + 1]) if "--inwindow" in sys.argv else 0.0
    mvt = data.load_movements()
    y = data.load_target()
    links = linked_frame(mvt, y)
    fit = links[(links["split"] == "train") & ~links["month"].isin([1, 7])].copy()
    fit["stale"] = (fit["y"] - fit["sd"]).abs() <= 60
    f_stale_ok = (fit["sd"] < fit["lob"] - W) | (fit["sd"] > fit["lob"] + W)
    f_gen_ok = (1_000 < fit["lob"] - W) | (1_000 > fit["lob"] + W)
    p_no_stale = (fit["stale"][~f_stale_ok & f_gen_ok].sum() + 1) / ((~f_stale_ok & f_gen_ok).sum() + 2)
    p_no_gen = (fit["stale"][f_stale_ok & ~f_gen_ok].sum() + 1) / ((f_stale_ok & ~f_gen_ok).sum() + 2)
    leaves = leaf_runs(run)
    print(f"{len(leaves)} leaf runs; exclusion rates {p_no_stale:.3f} / {p_no_gen:.3f}")

    src = paths.run_dir(run)
    oof = pd.read_parquet(src / "oof_predictions.parquet")
    lo, hi = lobt_bounds(mvt, oof.index)
    base = pd.Series(clip_to_bounds(oof["y_pred"], lo, hi), index=oof.index)
    offsets = inwindow_offsets(mvt, y)
    if factor:
        base = inwindow(base, oof["y_pred"], mvt, offsets, factor)
    new = exclusion(base, averaged_detail(leaves, "oof"), links, p_no_stale, p_no_gen)
    new = pd.Series(clip_to_bounds(new, lo, hi), index=oof.index)
    yt = oof["y_true"].to_numpy()
    wd = oof["dayoffset"].to_numpy(bool)
    for name, pr in ((f"window clip{' + in-window x' + str(factor) if factor else ''}", base), ("+ linked exclusion", new)):
        e = (pr.to_numpy() - yt) ** 2
        print(f"  holdout {name:<20} rmse {np.sqrt(e.mean()):.2f}  ex_wd {np.sqrt(e[~wd].mean()):.2f}")

    test = pd.read_parquet(src / "test_predictions.parquet")
    lo_t, hi_t = lobt_bounds(mvt, test.index)
    t_base = pd.Series(clip_to_bounds(test["y_pred"], lo_t, hi_t), index=test.index)
    if factor:
        t_base = inwindow(t_base, test["y_pred"], mvt, offsets, factor)
    t_new = exclusion(t_base, averaged_detail(leaves, "rank"), links, p_no_stale, p_no_gen)
    t_new = pd.Series(clip_to_bounds(t_new, lo_t, hi_t), index=test.index)
    moved = (t_new != t_base)
    print(f"  ranking rows moved by the exclusion: {int(moved.sum())}, sum of squared shifts {((t_new - t_base)[moved] ** 2).sum() / 1e6:.1f}e6")

    if "--write" in sys.argv:
        name = sys.argv[sys.argv.index("--write") + 1]
        import hashlib
        tag = hashlib.sha1(f"{src.name}|{name}".encode()).hexdigest()[:8]
        dst = paths.RUNS / f"{src.name.split('__')[0]}__{name}__{tag}"
        dst.mkdir(parents=True, exist_ok=False)
        for f in ("config.resolved.json", "env.json"):
            if (src / f).exists():
                shutil.copy(src / f, dst / f)
        o2 = oof.copy()
        o2["y_pred"] = new
        o2.to_parquet(dst / "oof_predictions.parquet")
        pd.DataFrame({"y_pred": t_new}).to_parquet(dst / "test_predictions.parquet")
        (dst / "metrics.json").write_text(json.dumps({"stack": {"source": run, "steps": ["lobt_window", "linked_exclusion"],
                                                                 "rates": [p_no_stale, p_no_gen], "inwindow": factor}}, indent=1))
        print(f"  wrote {dst.name}")


if __name__ == "__main__":
    main()
