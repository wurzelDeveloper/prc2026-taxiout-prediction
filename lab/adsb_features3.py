"""Third ADS-B feature set: the four families the exploration workflow of 1 Oct 2026 adopted (findings 10.10).

Assembles cache/adsb/features3.parquet (index mvt_id, holdout and 2026 departures) from the per-family outputs under
cache/lab/ideas/ (code in lab/ideas/), with the columns the reviewers kept:

* pushback (`lab/ideas/pushback.py`): reverse-motion pushback detector. On the ground most transponders report the
  heading in the track field, so a push is motion ~180 deg off the reported track. pb_start (push start, median error
  against the recorded off-block +6 s, MAE 29 s on 2025 push rows), push end, first forward move, taxi start, the
  sustained departure from the stand cluster, durations (push, engine-start pause), confidence (cluster distance to
  the stand, censoring, timing gap);
* first reception (`lab/ideas/first_reception.py`): the state at the first ground point for departures first seen away
  from the stand: speed, motion direction, route already covered, deviation from the stand's usual reception
  boundary, an estimate of the taxi including the unseen part (the two pace columns are left out: they change
  meaning in 2026);
* track quality (`lab/ideas/track_quality.py`): reception density, gaps, stand jitter, implausible jumps, the inbound
  landing, relocations (gap300_s and first_gs left out: duplicates);
* stop classes (`lab/ideas/stop_class.py`, `stop_class_screen.py`): the last long stop read as part of the taxi, its
  class (stand / runway area / remote / hold) and location statistics learnt target-free from all 124 days, the
  de-icing-like stop, the runway-area queue entry, a parking-like off-block estimate (ll_excl_gap left out: duplicate
  of adsb_stop_end_300).

Naming: `*_est` = a taxi estimate (seconds before the recorded takeoff); `*_tk` = seconds before the ADS-B takeoff
(`lab/adsb_apply.py --v4` turns both into gaps to the base).

Operational availability: tier M (challenge-only) for every column: all four families observe the flight's own
movement after its off-block. The location statistics of the stop classes pool all 124 days of ADS-B stops (no target).

Usage: python lab/adsb_features3.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lab"))
sys.path.insert(0, str(ROOT / "lab" / "ideas"))
from prc import paths  # noqa: E402

IDEAS = paths.CACHE / "lab" / "ideas"
OUT = paths.CACHE / "adsb" / "features3.parquet"


def _dedup(f: pd.DataFrame) -> pd.DataFrame:
    return f[~f.index.duplicated(keep="last")]


def pushback() -> pd.DataFrame:
    f = _dedup(pd.concat([pd.read_parquet(x) for x in sorted(glob.glob(str(IDEAS / "pushback_days" / "*.parquet")))]))
    est = {"pb_start": "pb_start_est", "pb_end": "pb_end_est", "pb_go": "pb_go_est", "pb_taxi8": "pb_taxi8_est",
           "pb_leave": "pb_leave_est", "pb_leave_next": "pb_leave_next_est"}
    keep = ["pb_found", "pb_n_rev", "pb_push_m", "pb_start_gap", "pb_censored", "pb_push_dur", "pb_es_dur",
            "pb_anchor_src", "pb_cl_n", "pb_cl_dwell", "pb_cl_dist_stand"]
    return f[keep + list(est)].rename(columns=est)


def first_reception() -> pd.DataFrame:
    f = _dedup(pd.read_parquet(IDEAS / "first_reception_all.parquet"))
    keep = ["fr_gs0", "fr_moving0", "fr_motion_vs_away", "fr_covered_m", "fr_frac_route", "fr_dev_dist", "fr_dev_cov"]
    return f[keep + ["fr_taxi_est_v"]].rename(columns={"fr_taxi_est_v": "fr_taxi_est"})


def track_quality() -> pd.DataFrame:
    f = _dedup(pd.concat([pd.read_parquet(x) for x in sorted(glob.glob(str(IDEAS / "track_quality_days" / "*.parquet")))]))
    keep = ["adsbQ_first_next", "adsbQ_jit_std", "adsbQ_jit_max", "adsbQ_fm_gs", "adsbQ_fm_slow_n",
            "adsbQ_seg_after_landing", "adsbQ_n_locs", "adsbQ_n_locs300", "adsbQ_jumps", "adsbQ_onground_share",
            "adsbQ_dens30", "adsbQ_gap300_n"]
    tk = {"adsbQ_park_end": "adsbQ_park_end_tk", "adsbQ_reloc": "adsbQ_reloc_tk", "adsbQ_since_landing": "adsbQ_since_landing_tk"}
    return f[keep + list(tk)].rename(columns=tk)


def stop_classes() -> pd.DataFrame:
    import stop_class_screen as S
    from stop_class import derive_from_stops
    f = S.load_days("20*.parquet")
    stops = S.load_stops()
    f = f.join(derive_from_stops(stops[stops["mvt_id"].isin(f.index)]))
    zero = pd.Series(0.0, index=f.index)
    c = S.v1_frame(f, zero).join(S.v2_extra(f, stops[stops["mvt_id"].isin(f.index)], S.location_map(stops), zero))
    est = {"ll_incl_gap": "ll_incl_est", "deice_start_gap": "deice_start_est", "rwy_queue_gap": "rwy_queue_est",
           "park2_gap": "park2_est"}
    keep = ["ll_cls", "ll_dur", "ll_d_end", "ll_d_stand", "ll_d_own", "ll_first_share", "ll_jan_lift", "deice_dur",
            "deice_rem", "dz_s"]
    return c[keep + list(est)].rename(columns=est)


def main():
    parts = {"pushback": pushback(), "first_reception": first_reception(), "track_quality": track_quality(),
             "stop_classes": stop_classes()}
    out = None
    for name, p in parts.items():
        print(f"{name:<16} {len(p):>8,} rows, {p.shape[1]} columns")
        out = p if out is None else out.join(p, how="outer")
    out.index.name = "mvt_id"
    for c in ("pb_anchor_src", "ll_cls"):
        out[c] = out[c].astype("string")
    out.to_parquet(OUT)
    print(f"wrote {OUT}: {out.shape}")


if __name__ == "__main__":
    main()
