"""Fourth ADS-B feature set: the three families the second exploration round adopted (findings 10.10).

Assembles cache/adsb/features4.parquet (index mvt_id, holdout and 2026 departures) from cache/lab/ideas/ (code in
lab/ideas/), with the columns the reviewers kept:

* inbound and turnaround (`lab/ideas/inbound_turn.py`): the departing aircraft followed back up to 24 h: inbound
  touchdown (extrapolated where ground reception is missing), the end of its inbound track, the link to the airport's
  arrival record (recorded in-block, arrival stand, unseen taxi-in), park locations and tows over the turnaround,
  callsign switches, and off-block estimates from them (`it_*_est`). Dropped as duplicates: it_arr_inblock_ago,
  it_dep_first_d_stand, it_end_ago, it_land_ago.
* stand-type archive (`lab/ideas/stand_archive.py`): per stand, target-free from all 124 days of ADS-B: share of
  push-back off-blocks, typical push and engine-start durations, switch-on lead, first-sighting distance (`ar_*`), and
  the best archive-based off-block estimate for what this flight shows (`sa_best_est`, `sa_8kt_est`, `sa_best_src`).
  Dropped as duplicates: sa_first_d, sa_first_d_rel, sa_rad_est, sa_radrw_est, sa_seen_est, sa_go_est.
* time-to-location map (`lab/ideas/timemap.py`): the typical time from push start to leaving each 75 m surface cell
  (per stand, zone, airport; learnt from clean observed pushes) and from entering a cell to the recorded in-block
  (arrivals), applied to this flight's own cell exits (`tmap_*`, `tarr_*`).

Operational availability: tier M (challenge-only) for every column: the estimates read the flight's own track after
its off-block; the archives and maps pool all 124 days of ADS-B and arrivals' recorded in-block times (no target).

Usage: python lab/adsb_features4.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prc import paths  # noqa: E402

IDEAS = paths.CACHE / "lab" / "ideas"
OUT = paths.CACHE / "adsb" / "features4.parquet"
DROP = {
    "inbound_turn": ["it_arr_inblock_ago", "it_dep_first_d_stand", "it_end_ago", "it_land_ago"],
    "stand_archive": ["sa_first_d", "sa_first_d_rel", "sa_rad_est", "sa_radrw_est", "sa_seen_est", "sa_go_est"],
    "timemap": [],
}


def main():
    out = None
    for name, drop in DROP.items():
        f = pd.read_parquet(IDEAS / f"{name}.parquet")
        f = f[~f.index.duplicated(keep="last")].drop(columns=drop)
        print(f"{name:<14} {len(f):>8,} rows, {f.shape[1]} columns")
        out = f if out is None else out.join(f, how="outer")
    out.index.name = "mvt_id"
    for c in ("it_end_kind", "sa_best_src"):
        out[c] = out[c].astype("string")
    out.to_parquet(OUT)
    print(f"wrote {OUT}: {out.shape}")


if __name__ == "__main__":
    main()
