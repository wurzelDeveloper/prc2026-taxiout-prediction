"""The numbers of docs/report.md section 1.4's data findings: the records' defects in 2025 and each airport's
strongest disrupted day of 2025 (Table 6).

    python lab/report_data_findings.py

Reads data/interim/ (`prc prepare`) and v70's run record. A day is the airport's local calendar day; its mean
taxi-out and its share above 30 minutes are taken over the departures with a target below 2 h (the stale and
wrong-date records left out); a day counts only with at least half the airport's median number of departures (the
first local hours of 1 January 2026 at Paris are not a day); the airport's typical day is its median day of 2025.
Stale fingerprint: the target equals the takeoff minus the schedule within 120 s. Wrong date: the target lies within
3,600 s of a whole number (at least one) of days. Prints only; writes nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prc import paths  # noqa: E402
from prc.features.temporal import AIRPORT_TZ  # noqa: E402

V70 = "2026-10-06T10-47-37__dc_A_S79_3289fc__3289fcc3"


def departures_2025() -> pd.DataFrame:
    t = pd.read_parquet(paths.INTERIM / "target.parquet")["TAXITIME_SEC_mvt"]
    m = pd.read_parquet(paths.INTERIM / "movements.parquet",
                        columns=["split", "is_dep", "apt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
                                 "AIRCRAFT_TYPE_mvt", "LOBT_flt"])
    d = m[(m["split"] == "train") & m["is_dep"]].join(t.rename("y"), how="inner")
    ts = d["MVT_TIME_UTC_mvt"]
    ts = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts
    d["day"] = None
    for apt, tz in AIRPORT_TZ.items():
        s = d["apt"] == apt
        if s.any():
            d.loc[s, "day"] = ts[s].dt.tz_convert(tz).dt.strftime("%Y-%m-%d")
    return d


def records(d: pd.DataFrame) -> None:
    delta = (d["MVT_TIME_UTC_mvt"] - d["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    long2h = d["y"] > 7200
    stale = (d["y"] - delta).abs() <= 120
    k = np.round(d["y"] / 86400)
    wrong = (k >= 1) & ((d["y"] - k * 86400).abs() <= 3600)
    seg = d["AIRCRAFT_TYPE_mvt"].isna() & d["LOBT_flt"].isna()
    print(f"departures of 2025 with a target: {len(d):,}")
    print(f"targets above 2 h: {long2h.sum()}, of which the stale fingerprint: {(long2h & stale).sum()}; "
          f"at Rome: {(long2h & (d['apt'] == 'LIRF')).sum()}")
    print(f"wrong-date records: {wrong.sum()}, at Rome: {(wrong & (d['apt'] == 'LIRF')).sum()}")
    print(f"no aircraft type and no NM match: {seg.sum()}, at Rome: {(seg & (d['apt'] == 'LIRF')).sum()}; "
          f"with the stale fingerprint and a target above 10 min: {(seg & stale & (d['y'] > 600)).sum()}")
    o = pd.read_parquet(paths.RUNS / V70 / "oof_predictions.parquet")
    se = (o["y_pred"] - o["y_true"]) ** 2
    tot = se.sum()
    worst = se.sort_values(ascending=False)
    n01 = int(np.ceil(0.001 * len(o)))
    print(f"v70 holdout, {len(o):,} rows: targets above 1 h {int((o['y_true'] > 3600).sum())} "
          f"({(o['y_true'] > 3600).mean():.2%}) carry {se[o['y_true'] > 3600].sum() / tot:.1%} of the squared error; "
          f"the worst 0.1 % ({n01} rows) {worst.iloc[:n01].sum() / tot:.1%}; the wrong-date rows "
          f"({int(o['dayoffset'].sum())}) {se[o['dayoffset']].sum() / tot:.1%}; targets above 20,000 s "
          f"({int((o['y_true'] > 20000).sum())}) {se[o['y_true'] > 20000].sum() / tot:.1%}")


def disrupted_days(d: pd.DataFrame) -> None:
    d = d[d["day"].str.startswith("2025")]
    b = d[d["y"] < 7200]
    g = b.groupby(["apt", "day"])["y"].agg(n="size", mean="mean", p30=lambda s: (s > 1800).mean())
    typ = g.groupby("apt").agg(typ_n=("n", "median"), typ_mean=("mean", "median"), typ_p30=("p30", "median"))
    g = g.join(typ, on="apt")
    g = g[g["n"] >= 0.5 * g["typ_n"]]
    g["incr"] = g["mean"] - g["typ_mean"]
    best = g.reset_index().sort_values("incr", ascending=False).groupby("apt").head(1)
    print("each airport's strongest day of 2025 (departures, mean taxi-out, increase, share above 30 min; "
          "the airport's median day in brackets):")
    for r in best.itertuples():
        day = pd.Timestamp(r.day).strftime("%a %d %b")
        print(f"  {r.apt} {day}: {r.n} ({r.typ_n:.0f}); {r.mean:,.0f} s ({r.typ_mean:,.0f} s); +{r.incr:,.0f} s "
              f"(x{r.mean / r.typ_mean:.1f}); {r.p30:.1%} ({r.typ_p30:.1%})")
    top = g.reset_index().sort_values("incr", ascending=False).head(10)
    print("the ten strongest airport-days:", ", ".join(f"{a} {x}" for a, x in zip(top["apt"], top["day"])))
    half = g[g["p30"] > 0.5].reset_index()
    print("more than half of the departures above 30 min:",
          ", ".join(f"{a} {x}" for a, x in zip(half["apt"], half["day"])))
    print(f"airport-days {len(g)}; mean taxi-out more than 10 min above the median day: {(g['incr'] > 600).sum()}; "
          f"more than 5 min: {(g['incr'] > 300).sum()}")


if __name__ == "__main__":
    dep = departures_2025()
    records(dep)
    disrupted_days(dep)
