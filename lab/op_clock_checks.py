"""The operational ladder's data evidence: the clocks around the off-block, and what is knowable when.

    python lab/op_clock_checks.py [--section N ...] [--out docs/op_clock_checks.txt]

Prints these tables (ported on 7 Oct 2026 from one-off analysis scripts that are not part of this repository):

1. availability of the NM columns, schedule, stand, runway and type per airport and split;
2. clock offsets of AOBT_3, LOBT, IOBT, EOBT_1 and the schedule against the recorded off-block per airport
   (median, MAD, within 60 / 300 s), and how often the recorded off-block (or the takeoff) precedes the H1
   anchor (IOBT, else LOBT, else schedule, minus 15 min);
3. EOBT_1 against LOBT on the NM-matched 2025 bulk: how often it differs, and how much closer to the recorded
   off-block it is where it does (a late-updated estimate of unknown update time, hence never an input);
4. runway knowability at H1: the own runway against the dominant departure runway of the four complete 15-min
   buckets before ``floor(a, 15 min)`` (the own takeoff removed from them; the up to 15 min between
   ``floor(a)`` and ``a`` are not read, which is causal but conservative), and against the set of runways in
   use; reported unconditionally, as the share of rows with any departure in that hour (coverage), and
   conditional on coverage, so "no recent traffic" and "another runway" are not mixed;
5. the taxiing-out queue at H0 and H1 (true queue from the airport's records, the NM-based queue, the plan
   backlog), correlated with the bulk taxi-out, overall and per airport;
6. AOBT_3's modal 60-s share per airport and runway against the target's (NM back-computing the off-block from
   the takeoff at LTFM and partly elsewhere);
7. runway knowability by decision moment (findings 10.35): section 4's measure at the 24 h and 3 h anchors on the
   schedule clock (``SOBT - lead``), at H1, and at the own takeoff (a hindsight reference).

Reads columns of the whole movement frame and the target file (about 2 GB of RAM at peak): run it when no GPU
fit is running. Nothing is written unless ``--out`` is given.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this checkout's `prc`, also when run as a script

from prc import data, metrics, paths  # noqa: E402
from prc.features import _horizon, _oplib  # noqa: E402

COLUMNS = ["apt", "split", "is_dep", "FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt",
           "EOBT_1_flt", "AOBT_3_flt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt"]
BULK = metrics.BULK_MAX_SEC


def load() -> tuple[pd.DataFrame, pd.Series]:
    """Departures (both splits) with the recorded off-block and the target joined on (training rows)."""
    mvt = data.load_movements(columns=COLUMNS)
    dep = mvt.loc[mvt["is_dep"].to_numpy(dtype=bool)].copy()
    del mvt
    tgt = pd.read_parquet(paths.TARGET_FILE)
    dep = dep.join(tgt, how="left")
    return dep, tgt["TAXITIME_SEC_mvt"]


def _sec(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a - b).dt.total_seconds()


def _robust(s: pd.Series) -> dict:
    s = s.dropna()
    if not len(s):
        return {"n": 0}
    med = s.median()
    return {"n": len(s), "median": med, "mad": (s - med).abs().median(), "w60": (s.abs() <= 60).mean(), "w300": (s.abs() <= 300).mean()}


# --- 1 ------------------------------------------------------------------------------------------------

def availability(dep: pd.DataFrame) -> pd.DataFrame:
    cols = {"nm_match": "FLIGHT_ID_mvt", "aobt3": "AOBT_3_flt", "lobt": "LOBT_flt", "iobt": "IOBT_flt", "eobt1": "EOBT_1_flt",
            "sched": "SCHED_TIME_UTC_mvt", "stand": "STAND_mvt", "runway": "RUNWAY_mvt", "actype": "AIRCRAFT_TYPE_mvt"}
    out = dep.groupby(["apt", "split"], observed=True).agg(n=("is_dep", "size"), **{k: (v, lambda s: s.notna().mean()) for k, v in cols.items()})
    return out.round(4)


# --- 2 ------------------------------------------------------------------------------------------------

def clock_offsets(dep: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = dep.loc[dep["split"].eq("train") & dep["BLOCK_TIME_UTC_mvt"].notna()]
    block = tr["BLOCK_TIME_UTC_mvt"]
    rows = []
    for apt, g in tr.groupby("apt", observed=True):
        r = {"apt": apt}
        for name, col in (("aobt3", "AOBT_3_flt"), ("lobt", "LOBT_flt"), ("iobt", "IOBT_flt"), ("eobt1", "EOBT_1_flt"), ("sched", "SCHED_TIME_UTC_mvt")):
            for k, v in _robust(_sec(g[col], block.loc[g.index])).items():
                if k != "n":
                    r[f"{name}_{k}"] = v
        rows.append(r)
    offsets = pd.DataFrame(rows).set_index("apt")

    matched = tr.loc[tr["FLIGHT_ID_mvt"].notna()]
    a = pd.Series(_oplib.anchor_h1(matched), index=matched.index)
    b = pd.Series(_oplib.secs(matched["BLOCK_TIME_UTC_mvt"]), index=matched.index)
    t = pd.Series(_oplib.secs(matched["MVT_TIME_UTC_mvt"]), index=matched.index)
    early = pd.DataFrame({"apt": matched["apt"].astype(str), "block_before_h1": b < a, "takeoff_before_h1": t <= a,
                          "block_before_lobt": b < _oplib.secs(matched["LOBT_flt"])})
    per = early.groupby("apt")[["block_before_h1", "takeoff_before_h1", "block_before_lobt"]].mean()
    per.loc["ALL"] = early[["block_before_h1", "takeoff_before_h1", "block_before_lobt"]].mean()
    return offsets.round(3), per.round(4)


# --- 3 ------------------------------------------------------------------------------------------------

def eobt1_vs_lobt(dep: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    m = dep.loc[dep["split"].eq("train") & dep["LOBT_flt"].notna() & (dep["TAXITIME_SEC_mvt"] < BULK)]
    blk = m["BLOCK_TIME_UTC_mvt"]
    d_lobt, d_e1 = _sec(blk, m["LOBT_flt"]), _sec(blk, m["EOBT_1_flt"])
    e1l, il = _sec(m["EOBT_1_flt"], m["LOBT_flt"]), _sec(m["IOBT_flt"], m["LOBT_flt"])
    sub = e1l.abs() > 0
    summary = {
        "n": len(m),
        "eobt1_differs": (e1l.abs() > 0).mean(),
        "eobt1_earlier": (e1l < 0).mean(),
        "eobt1_later_15min": (e1l > 900).mean(),
        "iobt_differs_from_lobt": (il.abs() > 0).mean(),
        "differing: med |block - lobt|": d_lobt[sub].abs().median(),
        "differing: med |block - eobt1|": d_e1[sub].abs().median(),
        "differing: |block - eobt1| <= 120": (d_e1[sub].abs() <= 120).mean(),
        "differing: |block - lobt| <= 120": (d_lobt[sub].abs() <= 120).mean(),
        "corr(eobt1 - lobt, taxi)": np.corrcoef(e1l.clip(-3600, 7200).fillna(0), m["TAXITIME_SEC_mvt"])[0, 1],
    }
    rows = []
    for apt, g in m.groupby("apt", observed=True):
        rows.append({"apt": apt, "eobt1_differs": (e1l.loc[g.index].abs() > 0).mean(),
                     "med_abs_block_eobt1": d_e1.loc[g.index].abs().median(), "med_abs_block_lobt": d_lobt.loc[g.index].abs().median()})
    return summary, pd.DataFrame(rows).set_index("apt").round(3)


# --- 4 ------------------------------------------------------------------------------------------------

def anchor_seconds(d: pd.DataFrame, moment: str) -> np.ndarray:
    """The anchor of a decision moment: H1, the long leads on the schedule clock, or the own takeoff (hindsight)."""
    if moment == "h1":
        return _oplib.anchor_h1(d)
    if moment in _horizon.LEADS_S:
        return _horizon.anchor_long(d, _horizon.LEADS_S[moment])
    if moment == "takeoff":
        return _oplib.secs(d["MVT_TIME_UTC_mvt"])
    raise ValueError(moment)


def runway_knowability(dep: pd.DataFrame, moment: str = "h1") -> pd.DataFrame:
    """Own runway against the departure runways of the four complete 15-min buckets before ``floor(a, 15 min)``.

    The own takeoff is removed from those buckets (it lands there only when the flight took off before its H1
    anchor). ``coverage`` is the share of rows with at least one other departure in the window; the ``*_cov``
    columns are conditional on it, the others unconditional (a row without coverage counts as "not dominant").
    """
    d = dep.loc[dep["split"].eq("train")].copy()
    d["a"] = pd.to_datetime(anchor_seconds(d, moment), unit="s", utc=True)
    d["b"] = d["MVT_TIME_UTC_mvt"].dt.floor("15min")
    cnt = d.groupby(["apt", "b", "RUNWAY_mvt"], observed=True).size().rename("n").reset_index()
    d["ab"] = d["a"].dt.floor("15min")
    parts = []
    for k in range(1, 5):
        t = d[["apt", "ab"]].copy()
        t["b"] = t["ab"] - pd.Timedelta(minutes=15 * k)
        t["row"] = d.index
        parts.append(t)
    q = pd.concat(parts).merge(cnt, on=["apt", "b"], how="inner")
    # the own takeoff out of its bucket
    own_b = d["b"].reindex(q["row"]).to_numpy()
    own_r = d["RUNWAY_mvt"].astype(str).reindex(q["row"]).to_numpy()
    is_own = (q["b"].to_numpy() == own_b) & (q["RUNWAY_mvt"].astype(str).to_numpy() == own_r)
    q["n"] = q["n"] - is_own.astype("int64")
    q = q.loc[q["n"] > 0]
    agg = q.groupby(["row", "RUNWAY_mvt"], observed=True)["n"].sum().reset_index().sort_values(["row", "n"], ascending=[True, False])
    top = agg.drop_duplicates("row").set_index("row")["RUNWAY_mvt"]
    own = d["RUNWAY_mvt"].astype(str)
    dom = top.reindex(d.index)
    d["covered"] = dom.notna().to_numpy()
    d["eq_dominant"] = d["covered"] & own.eq(dom.astype(str)).to_numpy()
    # the runways in use in the window, vectorised: the own runway is in use where (row, own runway) is a row of agg
    used = pd.MultiIndex.from_arrays([agg["row"].to_numpy(), agg["RUNWAY_mvt"].astype(str).to_numpy()])
    d["in_used_set"] = pd.MultiIndex.from_arrays([d.index.to_numpy(), own.to_numpy()]).isin(used)
    d["n_distinct"] = agg.groupby("row").size().reindex(d.index).fillna(0).to_numpy(dtype="int64")
    cov = d.loc[d["covered"]]
    out = d.groupby("apt", observed=True).agg(coverage=("covered", "mean"), own_eq_dominant_60min=("eq_dominant", "mean"),
                                              own_in_used_set=("in_used_set", "mean"), mean_distinct_dep_rwys=("n_distinct", "mean"),
                                              n_rwys_year=("RUNWAY_mvt", "nunique"))
    conditional = cov.groupby("apt", observed=True).agg(own_eq_dominant_cov=("eq_dominant", "mean"), own_in_used_set_cov=("in_used_set", "mean"))
    out = out.join(conditional)
    out.loc["ALL"] = [d["covered"].mean(), d["eq_dominant"].mean(), d["in_used_set"].mean(), d["n_distinct"].mean(), np.nan,
                      cov["eq_dominant"].mean(), cov["in_used_set"].mean()]
    return out.round(3)


# --- 5 ------------------------------------------------------------------------------------------------

def queue_correlations(dep: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = dep.loc[dep["split"].eq("train")]
    t_off = _oplib.secs(d["BLOCK_TIME_UTC_mvt"])
    t_to = _oplib.secs(d["MVT_TIME_UTC_mvt"])
    t_a3 = _oplib.secs(d["AOBT_3_flt"])
    plan = _oplib.planned_offblock(d)
    apt = d["apt"].astype("string").to_numpy(dtype=object)
    rwy = _oplib.runway_key(d).to_numpy(dtype=object)
    y = d["TAXITIME_SEC_mvt"].to_numpy(dtype="float64")
    rows, by_apt = [], None
    for hname, a in (("H0 (recorded off-block)", t_off), ("H1 (planned - 15 min)", plan - _oplib.H1_LEAD_S)):
        for lvl, g in (("airport", apt), ("runway", rwy)):
            q_true = _oplib.in_progress(t_off, t_to, g, a, g, t_off, t_to, max_dur=_oplib.FEED_QUEUE_MAX_S)
            q_nm = _oplib.in_progress(t_a3, t_to, g, a, g, t_a3, t_to, max_dur=_oplib.FEED_QUEUE_MAX_S)
            backlog = _oplib.in_progress(plan, t_to, g, a, g, plan, t_to, max_dur=_oplib.BACKLOG_MAX_S)
            f = pd.DataFrame({"q_true": q_true, "q_nm": q_nm, "backlog": backlog, "y": y, "apt": apt})
            f = f.loc[(f["y"] < BULK) & np.isfinite(a)]
            c = f[["q_true", "q_nm", "backlog", "y"]].corr()
            rows.append({"horizon": hname, "level": lvl, "n": len(f), "r(true queue, taxi)": c.loc["q_true", "y"],
                         "r(NM queue, taxi)": c.loc["q_nm", "y"], "r(plan backlog, taxi)": c.loc["backlog", "y"],
                         "r(true, NM queue)": c.loc["q_true", "q_nm"]})
            if hname.startswith("H0") and lvl == "runway":
                by_apt = f.groupby("apt").apply(lambda g_: pd.Series({
                    "n": len(g_), "r_true": g_["q_true"].corr(g_["y"]), "r_nm": g_["q_nm"].corr(g_["y"]),
                    "r_backlog": g_["backlog"].corr(g_["y"]), "r_true_nm": g_["q_true"].corr(g_["q_nm"]),
                    "mean_true": g_["q_true"].mean(), "mean_nm": g_["q_nm"].mean()}))
    return pd.DataFrame(rows).round(3), by_apt.round(3)


# --- 6 ------------------------------------------------------------------------------------------------

def best_window_share(x: np.ndarray, width: float = 60.0) -> float:
    """Largest share of values inside one window of ``width`` seconds."""
    x = np.sort(x[np.isfinite(x)])
    if len(x) < 100:
        return np.nan
    j = np.searchsorted(x, x + width, side="left")
    return float((j - np.arange(len(x))).max() / len(x))


def aobt3_modal_share(dep: pd.DataFrame, min_rows: int = 1000) -> pd.DataFrame:
    """Per airport and split: the share of NM proxies (and targets) inside their runway's modal 60-s window.

    Runways are weighted by their FINITE values (the proxy's and the target's separately), and a runway needs
    ``min_rows`` finite proxies to count. Empty frame when no runway qualifies.
    """
    d = dep.assign(prox=_sec(dep["MVT_TIME_UTC_mvt"], dep["AOBT_3_flt"]))
    rows = []
    for (apt, split), g in d.groupby(["apt", "split"], observed=True):
        num_p = num_t = w_p = w_t = 0.0
        for _, h in g.groupby("RUNWAY_mvt", observed=True):
            prox = h["prox"].to_numpy(dtype="float64")
            n_p = int(np.isfinite(prox).sum())
            if n_p < min_rows:
                continue
            share = best_window_share(prox)
            if np.isfinite(share):
                num_p += share * n_p
                w_p += n_p
            if split == "train" and "TAXITIME_SEC_mvt" in h.columns:
                y = h["TAXITIME_SEC_mvt"].to_numpy(dtype="float64")
                n_t = int(np.isfinite(y).sum())
                share_t = best_window_share(y)
                if n_t and np.isfinite(share_t):
                    num_t += share_t * n_t
                    w_t += n_t
        if w_p:
            rows.append({"apt": apt, "split": split, "n": w_p, "proxy_in_modal_60s": num_p / w_p,
                         "target_in_modal_60s": num_t / w_t if w_t else np.nan})
    if not rows:
        print(f"(no runway with at least {min_rows} finite proxies)")
        return pd.DataFrame()
    return pd.DataFrame(rows).pivot(index="apt", columns="split", values=["proxy_in_modal_60s", "target_in_modal_60s"]).round(3)


SECTIONS = {
    1: ("Availability per airport and split", lambda d: availability(d)),
    2: ("Clock offsets against the recorded off-block; off-block / takeoff before the H1 anchor", lambda d: clock_offsets(d)),
    3: ("EOBT_1 against LOBT (NM-matched 2025 bulk)", lambda d: eobt1_vs_lobt(d)),
    4: ("Runway knowability at H1", lambda d: runway_knowability(d)),
    5: ("Taxiing-out queue and the bulk taxi-out (H0, H1)", lambda d: queue_correlations(d)),
    6: ("AOBT_3 modal 60-s share against the target's", lambda d: aobt3_modal_share(d)),
    7: ("Runway knowability by decision moment (24 h, 3 h on the schedule clock; H1; the own takeoff)",
        lambda d: pd.DataFrame({m: runway_knowability(d, m).loc["ALL"] for m in ("hl24", "hl3", "h1", "takeoff")}).T),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--section", type=int, action="append", choices=sorted(SECTIONS))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    dep, _ = load()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for k in args.section or sorted(SECTIONS):
            title, fn = SECTIONS[k]
            print(f"\n=== {k}. {title}")
            res = fn(dep)
            for part in res if isinstance(res, tuple) else (res,):
                print(pd.Series(part).to_string() if isinstance(part, dict) else part.to_string())
                print()
    text = buf.getvalue()
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
