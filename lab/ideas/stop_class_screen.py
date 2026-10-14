"""Assemble the stop-class candidates from cache/lab/ideas/stop_class/<day>.parquet and screen them with the harness.

    python lab/ideas/stop_class_screen.py v1|v2|v3      # one harness call per variant
    python lab/ideas/stop_class_screen.py avail         # 2026 availability vs 2025 (no harness call)

Variants (all target-free; the harness is the only place the target enters):
* v1 - geometric classes: counts/durations per class after the parked position, the runway-area queue entry and
       line-up end, the de-icing-like stop (duration, start, end), the parked-stop end, the class of the last long
       stop and of the first stop; times as taxi estimates (minus the ADS-B takeoff offset) and gaps to v37.
* v2 - v1 plus a location map learnt from the ADS-B stops of all 124 days (`location_map`): per airport and 60 m
       cell, how often a stop there is the segment's first (parking-like) and how much more often long stops happen
       there in January than in July (de-icing signature); the last long stop and the de-icing-like stop carry their
       cell's values, and a "parking-like" last stop gives a second off-block estimate.
* v3 - a compact subset of v2 (15 columns): the last long stop read both ways with its location and map values,
       the de-icing-like stop, the runway-area queue entry and the map-based off-block estimate.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lab"))
sys.path.insert(0, str(ROOT / "lab" / "ideas"))
from prc import paths  # noqa: E402

SRC = paths.CACHE / "lab" / "ideas" / "stop_class"
OUTDIR = paths.CACHE / "lab" / "ideas"
CELL_M = 60.0
CATS_V1 = ["ll_cls", "sc_first_cls"]


def load_days(pattern: str = "20*.parquet") -> pd.DataFrame:
    fs = sorted(glob.glob(str(SRC / pattern)))
    f = pd.concat([pd.read_parquet(x) for x in fs])
    return f[~f.index.duplicated(keep="last")]


def load_stops() -> pd.DataFrame:
    fs = sorted(glob.glob(str(SRC / "stops_*.parquet")))
    return pd.concat([pd.read_parquet(x).assign(day=Path(x).stem[6:]) for x in fs], ignore_index=True)


def v1_frame(f: pd.DataFrame, ref: pd.Series) -> pd.DataFrame:
    """The v1 columns; `ref` is the prediction the gaps are taken against (v37 on the holdout)."""
    off = f["sc_offset"]
    c = pd.DataFrame(index=f.index)
    # the last long stop read both ways, with where it happened
    c["ll_cls"] = f["ll_cls"]
    c["ll_dur"] = f["ll_dur"]
    c["ll_excl_gap"] = f["ll_excl"] - off - ref
    c["ll_incl_gap"] = f["ll_incl"] - off - ref
    for d in ("d_end", "d_cl", "d_stand", "d_own"):
        c[f"ll_{d}"] = f[f"ll_{d}"]
    # stops after the parked position, per class
    for k in ("rwy", "remote", "hold"):
        c[f"n_{k}"] = f[f"sc_n_{k}"]
        c[f"s_{k}"] = f[f"sc_s_{k}"]
    c["rwy_queue_gap"] = f["sc_rwy_first_start"] - off - ref   # entry into the runway-area queue, as a taxi gap
    c["rwy_lineup_rem"] = f["sc_rwy_last_end"] - off           # recorded takeoff - end of the last runway-area stop
    c["deice_dur"] = f["sc_deice_dur"]
    c["deice_is_rwy"] = f["sc_deice_is_rwy"]
    c["deice_start_gap"] = f["sc_deice_start"] - off - ref
    c["deice_rem"] = f["sc_deice_end"] - off                    # taxi time remaining after the de-icing-like stop
    c["park_gap"] = f["sc_park_end"] - off - ref
    c["sc_first_cls"] = f["sc_first_cls"]
    return c


def location_map(stops: pd.DataFrame) -> pd.DataFrame:
    """Per (apt, cell): target-free statistics of the stops of all linked departures over all 124 days.

    first_share : share of the cell's stops >= 120 s that open the ground segment (the aircraft was first seen standing
                  there: a parking position);
    jan_lift    : (long stops in Jan / all stops in Jan) / (long stops in Jul / all stops in Jul) for the cell, with a
                  prior of 5 stops at the airport's rate (de-icing pads light up in January only);
    n_long      : number of stops >= 300 s in the cell.
    """
    s = stops.copy()
    s["cx"] = np.floor(s["x"] / CELL_M).astype(int)
    s["cy"] = np.floor(s["y"] / CELL_M).astype(int)
    s["first"] = (s["b"] - s["seg_t0"] < 60).astype(float)
    s["long"] = (s["dur"] >= 300).astype(float)
    s["jan"] = s["day"].str[5:7].eq("01")
    s["mid"] = (s["dur"] >= 120).astype(float)
    s["n"] = 1.0
    s["n_mid"] = s["mid"]
    s["n_first_mid"] = s["first"] * s["mid"]
    s["n_jan"] = s["jan"].astype(float)
    s["n_jan_long"] = s["long"] * s["jan"]
    s["n_jul"] = (~s["jan"]).astype(float)
    s["n_jul_long"] = s["long"] * (~s["jan"])
    s["n_long"] = s["long"]
    cols = ["n", "n_long", "n_mid", "n_first_mid", "n_jan", "n_jan_long", "n_jul", "n_jul_long"]
    m = s.groupby(["apt", "cx", "cy"], observed=True)[cols].sum().reset_index()
    # smooth over the 3x3 neighbourhood (stop centroids scatter across cell borders)
    out = []
    for apt, g in m.groupby("apt"):
        idx = {(a, b): i for i, (a, b) in enumerate(zip(g["cx"], g["cy"]))}
        cols = ["n", "n_long", "n_mid", "n_first_mid", "n_jan", "n_jan_long", "n_jul", "n_jul_long"]
        V = g[cols].to_numpy()
        S = np.zeros_like(V)
        for i, (a, b) in enumerate(zip(g["cx"], g["cy"])):
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    j = idx.get((a + da, b + db))
                    if j is not None:
                        S[i] += V[j]
        h = pd.DataFrame(S, columns=cols)
        sub = s[s["apt"] == apt]
        jr = sub.loc[sub["jan"], "long"].mean() if sub["jan"].any() else 0.0
        lr = sub.loc[~sub["jan"], "long"].mean() if (~sub["jan"]).any() else 0.0
        fr_ = sub.loc[sub["mid"] > 0, "first"].mean()
        k = 5.0
        h["first_share"] = (h["n_first_mid"] + k * fr_) / (h["n_mid"] + k)
        h["jan_lift"] = np.log(((h["n_jan_long"] + k * jr) / (h["n_jan"] + k) + 1e-3) /
                               ((h["n_jul_long"] + k * lr) / (h["n_jul"] + k) + 1e-3))
        h["n_long9"] = h["n_long"]
        h["apt"], h["cx"], h["cy"] = apt, g["cx"].to_numpy(), g["cy"].to_numpy()
        out.append(h[["apt", "cx", "cy", "first_share", "jan_lift", "n_long9"]])
    return pd.concat(out, ignore_index=True)


def v2_extra(f: pd.DataFrame, stops: pd.DataFrame, lmap: pd.DataFrame, ref: pd.Series) -> pd.DataFrame:
    """Per departure: the location-map values of its last long stop and of its de-icing-like stop, the longest
    January-lifted stop after the parked position, and an off-block estimate from the last parking-like stop."""
    s = stops.copy()
    s["cx"] = np.floor(s["x"] / CELL_M).astype(int)
    s["cy"] = np.floor(s["y"] / CELL_M).astype(int)
    s = s.merge(lmap, on=["apt", "cx", "cy"], how="left")
    s = s.sort_values(["mvt_id", "b"])
    off = f["sc_offset"]
    out = pd.DataFrame(index=f.index)
    lng = s[s["dur"] >= 300]
    last = lng.groupby("mvt_id").tail(1).set_index("mvt_id")
    out["ll_first_share"] = last["first_share"]
    out["ll_jan_lift"] = last["jan_lift"]
    out["ll_n_long9"] = last["n_long9"]
    # parking-like stops: the cell is mostly where aircraft are first seen standing (or the geometric stand class)
    park = s[(s["dur"] >= 120) & ((s["first_share"] >= 0.5) | (s["cls"] == "stand"))]
    pl = park.groupby("mvt_id").tail(1).set_index("mvt_id")
    out["park2_gap"] = (pl["tk"] - pl["e"]) - off - ref
    # after that parked position: the stops in January-lifted cells (de-icing signature) and their total
    since = pl["e"].reindex(s["mvt_id"]).to_numpy()
    since = np.where(np.isfinite(since), since, s["seg_t0"].to_numpy())
    aft = s[(s["b"].to_numpy() >= since) & (s["dur"] >= 120)]
    dz = aft[(aft["jan_lift"] >= 0.7) & (aft["n_long9"] >= 10)]
    out["dz_s"] = dz.groupby("mvt_id")["dur"].sum()
    dl = dz.loc[dz.groupby("mvt_id")["dur"].idxmax()].set_index("mvt_id") if len(dz) else dz.set_index("mvt_id")
    out["dz_start_gap"] = (dl["tk"] - dl["b"]) - off - ref
    out["dz_rem"] = (dl["tk"] - dl["e"]) - off
    out["dz_s"] = out["dz_s"].fillna(0.0)
    return out


def main():
    import idea_harness as H
    fr = H.load_frame()
    f = load_days("2025-*.parquet")
    f = f[f.index.isin(fr.index)]
    ref = fr["v37"].reindex(f.index)
    arg = sys.argv[1]
    if arg == "avail":
        a25 = load_days("2025-*.parquet")
        a26 = load_days("2026-*.parquet")
        cols = ["sc_park_end", "sc_rwy_last_end", "sc_deice_start", "sc_n_rwy", "sc_s_rwy", "sc_deice_dur"]
        for nm, a in (("2025", a25), ("2026", a26)):
            print(nm, len(a), {c: (round(float(a[c].notna().mean() if c not in ("sc_n_rwy", "sc_s_rwy", "sc_deice_dur") else (a[c] > 0).mean()), 3),
                                   round(float(a[c].median()), 1)) for c in cols})
        return
    from stop_class import derive_from_stops
    stops = load_stops()
    f = f.join(derive_from_stops(stops[stops["mvt_id"].isin(f.index)]))
    c1 = v1_frame(f, ref)
    if arg == "v1":
        c1.to_parquet(OUTDIR / "stop_class_v1.parquet")
        res = H.test(c1, cats=CATS_V1, name="stop_class_v1")
    else:
        lmap = location_map(stops)
        lmap.to_parquet(OUTDIR / "stop_class_locmap.parquet")
        st = stops[stops["mvt_id"].isin(f.index)]
        ex = v2_extra(f, st, lmap, ref)
        if arg == "v2":
            c2 = c1.join(ex)
            c2.to_parquet(OUTDIR / "stop_class_v2.parquet")
            res = H.test(c2, cats=CATS_V1, name="stop_class_v2")
        else:
            keep = ["ll_cls", "ll_dur", "ll_excl_gap", "ll_incl_gap", "ll_d_end", "ll_d_stand", "ll_d_own",
                    "ll_first_share", "ll_jan_lift", "deice_dur", "deice_start_gap", "deice_rem", "dz_s",
                    "rwy_queue_gap", "park2_gap"]
            c3 = c1.join(ex)[keep]
            c3.to_parquet(OUTDIR / "stop_class_v3.parquet")
            res = H.test(c3, cats=["ll_cls"], name="stop_class_v3")
    import json
    (OUTDIR / f"stop_class_{arg}_result.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
