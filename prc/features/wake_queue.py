"""Wake-turbulence separation and the runway queue each takeoff sits in.

`runway_sequence` names the preceding takeoff's wake category and a crude
saturation share (gaps under 90 s). This block applies the separation rules
themselves. ICAO Doc 4444 time-based minima for departures from the same
runway: a Light or Medium behind a Heavy waits 2 min, anything behind a Super
(A380, wake ``J``) 2 min (Heavy) or 3 min (Medium, Light), a Light behind a
Medium 2 min; any other pair is bound only by runway occupancy and radar
spacing, taken here as 60 s. From the takeoff times, which every departure has
in training and ranking alike:

* the gaps to the previous and the next takeoff on the runway, the required
  minimum for the pair and the slack between them;
* the continuous queue: consecutive takeoffs whose gap stays within the
  pair's minimum plus ``QUEUE_TOL_S`` form one run of a busy runway. A flight
  cannot have waited at the holding point longer than the run it belongs to
  has lasted (before it began, the runway was free), so the run's age bounds
  the queue wait from above, and the flight's position and the followers
  behind it say how deep the queue was.

The followers are later takeoffs: hindsight within the hour, like the
forward windows other congestion blocks use; every one exists in the ranking
data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc.features.core import epoch_seconds
from prc.features.registry import FeatureContext, feature

BASE_SEPARATION_S = 60.0
QUEUE_TOL_S = 45.0
QUEUE_FLOOR_S = 105.0
"""Two takeoffs are in the same queue when the gap is within the pair's minimum plus
``QUEUE_TOL_S``, and never when it exceeds ``max(minimum + tol, QUEUE_FLOOR_S)``."""
MAX_GAP_S = 3600.0

WAKE_QUEUE = (
    "wq_gap_prev_s",
    "wq_gap_next_s",
    "wq_required_s",
    "wq_slack_s",
    "wq_next_required_s",
    "wq_queue_pos",
    "wq_queue_age_s",
    "wq_queue_followers",
    "wq_queue_len",
)


def required_separation(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """ICAO time-based wake minimum (seconds) for ``cur`` taking off behind ``prev`` on the same runway."""
    prev = np.asarray(prev, dtype=object)
    cur = np.asarray(cur, dtype=object)
    r = np.full(len(prev), BASE_SEPARATION_S)
    light_med = np.isin(cur, ["M", "L"])
    r[(prev == "H") & light_med] = 120.0
    r[(prev == "J") & (cur == "H")] = 120.0
    r[(prev == "J") & light_med] = 180.0
    r[(prev == "M") & (cur == "L")] = 120.0
    return r


@feature(
    group="wake_queue",
    outputs=WAKE_QUEUE,
    depends_on=("core_keys",),
    doc="ICAO wake minimum to the previous takeoff and its slack, and the continuous runway queue (position, age, followers, length) the takeoff belongs to.",
)
def wake_queue(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    dep = mvt.loc[mvt["is_dep"], ["apt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt", "WK_TBL_CAT_flt"]]
    frame = pd.DataFrame(
        {
            "t": np.asarray(epoch_seconds(dep["MVT_TIME_UTC_mvt"]), dtype="float64"),
            "rw": (dep["apt"].astype("string") + "_" + dep["RUNWAY_mvt"].astype("string").fillna("__NA__")).to_numpy(),
            "wk": dep["WK_TBL_CAT_flt"].astype("string").fillna("?").to_numpy(),
        },
        index=dep.index,
    )
    frame = frame[np.isfinite(frame["t"])].sort_values(["rw", "t"], kind="stable")
    g = frame.groupby("rw", sort=False)
    prev_wk = g["wk"].shift(1).to_numpy(dtype=object)
    next_wk = g["wk"].shift(-1).to_numpy(dtype=object)
    gap_prev = (frame["t"] - g["t"].shift(1)).to_numpy(dtype="float64")
    gap_next = (g["t"].shift(-1) - frame["t"]).to_numpy(dtype="float64")
    req = required_separation(prev_wk, frame["wk"].to_numpy(dtype=object))
    req[pd.isna(prev_wk)] = np.nan
    next_req = required_separation(frame["wk"].to_numpy(dtype=object), next_wk)
    next_req[pd.isna(next_wk)] = np.nan

    linked = np.nan_to_num(gap_prev, nan=np.inf) <= np.maximum(np.nan_to_num(req, nan=BASE_SEPARATION_S) + QUEUE_TOL_S, QUEUE_FLOOR_S)
    run = pd.Series((~linked).astype("int64"), index=frame.index).groupby(frame["rw"].to_numpy(), sort=False).cumsum()
    run_key = pd.Series(frame["rw"].to_numpy() + "#" + run.astype(str).to_numpy(), index=frame.index)
    rg = frame["t"].groupby(run_key.to_numpy(), sort=False)
    pos = rg.cumcount().to_numpy()
    age = (frame["t"] - rg.transform("first")).to_numpy()
    length = rg.transform("size").to_numpy()

    out = pd.DataFrame(
        {
            "wq_gap_prev_s": np.clip(gap_prev, 0, MAX_GAP_S),
            "wq_gap_next_s": np.clip(gap_next, 0, MAX_GAP_S),
            "wq_required_s": req,
            "wq_slack_s": np.clip(gap_prev - req, -MAX_GAP_S, MAX_GAP_S),
            "wq_next_required_s": next_req,
            "wq_queue_pos": pos,
            "wq_queue_age_s": age,
            "wq_queue_followers": length - pos - 1,
            "wq_queue_len": length,
        },
        index=frame.index,
    ).astype("float32")
    return out.reindex(ctx.dep_index)[list(WAKE_QUEUE)]
