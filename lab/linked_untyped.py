"""Lab: regime exclusion for unmatched, untyped departures whose flight is NM-joined at its destination.

An unmatched departure's block time lies outside NM's LOBT +- 3,606 s (the join window, findings 10.3). When the
flight's arrival at another of the ten airports is NM-joined, that record carries the departure's LOBT, so a regime
whose value would fall inside the window is impossible: stale (target = schedule delta) or a genuine taxi (target
= the regressor's taxi). The rule's regime rates come from training rows outside the evaluated months.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import ROOT

sys.path.insert(0, str(ROOT))
from prc import data  # noqa: E402

W = 3_606.0


def linked_frame(mvt: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    apts = set(mvt["apt"].astype(str).unique())
    dep = mvt[mvt["is_dep"]].copy()
    dep["y"] = y.reindex(dep.index)
    un = dep[dep["FLIGHT_ID_mvt"].isna() & dep["AIRCRAFT_TYPE_mvt"].isna()]
    arr = mvt[~mvt["is_dep"] & mvt["FLIGHT_ID_mvt"].notna()][
        ["apt", "ADEP_mvt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "LOBT_flt"]
    ].reset_index().rename(columns={"mvt_id": "arr_id", "MVT_TIME_UTC_mvt": "land", "apt": "arr_apt"})
    u = un[un["ADES_mvt"].astype(str).isin(apts)][
        ["apt", "ADES_mvt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "y", "split"]
    ].reset_index()
    m = u.merge(arr, left_on=["FLIGHT_mvt", "ADES_mvt", "apt"], right_on=["FLIGHT_mvt", "arr_apt", "ADEP_mvt"])
    gap = (m["land"] - m["MVT_TIME_UTC_mvt"]).dt.total_seconds()
    m = m[(gap > 600) & (gap < 72_000)].sort_values("land").drop_duplicates("mvt_id").set_index("mvt_id")
    m["lob"] = (m["MVT_TIME_UTC_mvt"] - m["LOBT_flt"]).dt.total_seconds()
    m["sd"] = (m["MVT_TIME_UTC_mvt"] - m["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    m["month"] = m["MVT_TIME_UTC_mvt"].dt.month
    return m


def main():
    mvt = data.load_movements()
    y = data.load_target()
    m = linked_frame(mvt, y)
    run = sys.argv[1] if len(sys.argv) > 1 else "2026-09-30T20-18-18__blend17_wide_lobt__a41a47d5"
    oof = pd.read_parquet(ROOT / "runs" / run / "oof_predictions.parquet")
    h = m.join(oof[["y_pred", "y_reg", "p_stale", "dayoffset"]], how="inner")
    reg = h["y_reg"].to_numpy()
    stale_ok = (h["sd"] < h["lob"] - W) | (h["sd"] > h["lob"] + W)
    genuine_ok = (reg < h["lob"] - W) | (reg > h["lob"] + W)
    fit = m[(m["split"] == "train") & ~m["month"].isin([1, 7])].copy()
    fit["stale"] = (fit["y"] - fit["sd"]).abs() <= 60
    # the fit rows have no OOF regressor answer; a 1,000 s taxi stands in for the genuine regime there
    f_stale_ok = (fit["sd"] < fit["lob"] - W) | (fit["sd"] > fit["lob"] + W)
    f_gen_ok = (1_000 < fit["lob"] - W) | (1_000 > fit["lob"] + W)
    p_no_stale = (fit["stale"][~f_stale_ok & f_gen_ok].sum() + 1) / ((~f_stale_ok & f_gen_ok).sum() + 2)
    p_no_gen = (fit["stale"][f_stale_ok & ~f_gen_ok].sum() + 1) / ((f_stale_ok & ~f_gen_ok).sum() + 2)
    print(f"fit-month rates: p_stale when stale is impossible {p_no_stale:.3f}, when a genuine taxi is impossible {p_no_gen:.3f}")
    p = h["p_stale"].to_numpy()
    p_new = np.where(~stale_ok & genuine_ok, p_no_stale, np.where(stale_ok & ~genuine_ok, p_no_gen, p))
    new = p_new * h["sd"].to_numpy() + (1 - p_new) * reg
    yt = h["y"].to_numpy()
    wd = h["dayoffset"].to_numpy(bool)
    for name, pred in (("current", h["y_pred"].to_numpy()), ("regime exclusion", new)):
        e = (pred - yt) ** 2
        print(f"holdout linked untyped rows {len(h)} (ex wd {int((~wd).sum())}): {name:<18} SSE {e[~wd].sum() / 1e6:9.1f}e6 "
              f"-> holdout MSE contribution {e[~wd].sum() / 344_412:8.1f}")
    r = m[m["split"] == "rank"]
    print("2026 linked untyped rows:", len(r))


if __name__ == "__main__":
    main()
