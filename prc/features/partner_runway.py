"""Traffic on the partner runway of a close parallel pair (dependent runways).

At Istanbul (LTFM) the pairs 16L/16R (34R/34L) and 17L/17R (35R/35L) lie about 205-212 m apart and run segregated:
one runway departs while its partner lands, and takeoffs on 35L fall to 0.46x their normal rate 30-90 s before a
landing on 35R. At Paris-CDG the doublets 08L/08R (26R/26L) and 09R/09L (27L/27R) are run the same way, departures
inner and landings outer, and every landing taxies across the departure runway. A departure therefore waits on its
partner runway's traffic, which the other runway blocks never see: `runway_queue`, `runway_sequence`, `wake_queue`
and `taxi_window_counts` key on the own runway, `airport_load` on the airport. Pre-tests: findings 10.28,
``lab/pretests/dependent_runways/`` (``ltfm_pairs_pretest.py``, ``cdg_02_features.py``).

The pairs are a static map per airport (``PARTNER``); every other runway and airport has no partner, and its rows get
``prw_has_partner = 0`` and NaN in every other output. The own runway key is ``apt + "_" + RUNWAY_mvt`` as in
`runway_queue` and `runway_sequence`; a landing is an arrival's ``MVT_TIME_UTC_mvt``, a takeoff a departure's.
References are the arrivals and departures of both splits, so the 2026 rows see their own context.

Every count and gap looks strictly before the anchor ``t`` (windows ``[t - w, t)``), so the flight itself is never
counted, and neither is another movement at the very same second. Times are whole seconds in the data, which is
what the `congestion` primitives' integer windows assume.

Operational availability: tier **T** (takeoff-anchored): the anchor is the flight's own actual takeoff
(``MVT_TIME_UTC_mvt``), exactly like `runway_queue`; the same counts are operational once re-anchored on a planned or
predicted time. The partner runway follows from the own *recorded* runway (the planned runway, O*).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc.features.congestion import neighbour_gap, window_count
from prc.features.core import epoch_seconds
from prc.features.registry import FeatureContext, feature

PARTNER: dict[str, dict[str, str]] = {
    "LTFM": {
        "16L": "16R", "16R": "16L", "34L": "34R", "34R": "34L",
        "17L": "17R", "17R": "17L", "35L": "35R", "35R": "35L",
    },
    "LFPG": {
        "08L": "08R", "08R": "08L", "26R": "26L", "26L": "26R",
        "09R": "09L", "09L": "09R", "27L": "27R", "27R": "27L",
    },
}
"""Close parallel pairs per airport: runway -> the other runway of its pair, same direction."""

GAP_CAP_S = 7200.0

OUTPUTS = (
    "prw_has_partner",
    "prw_arr_10",
    "prw_arr_20",
    "prw_dep_10",
    "prw_gap_prev_arr",
    "prw_gap_prev_mvt",
    "prw_pair_mvt_10",
)


def _runway_key(frame: pd.DataFrame) -> pd.Series:
    return frame["apt"].astype("string") + "_" + frame["RUNWAY_mvt"].astype("string").fillna("__NA__")


def partner_key(frame: pd.DataFrame) -> pd.Series:
    """``apt_<partner runway>`` for every row whose runway belongs to a pair, else <NA>."""
    lookup = {f"{apt}_{rwy}": f"{apt}_{other}" for apt, pairs in PARTNER.items() for rwy, other in pairs.items()}
    return _runway_key(frame).map(lookup).astype("string")


def _before(ref_t: pd.Series, ref_g: pd.Series, q_t: pd.Series, q_g: pd.Series, window: int) -> np.ndarray:
    """Reference events in ``[t - window, t)`` within the query's group: the closed window minus the events at ``t``."""
    closed = window_count(ref_t, ref_g, q_t, q_g, back=window, fwd=0)
    at_t = window_count(ref_t, ref_g, q_t, q_g, back=0, fwd=0)
    return (closed - at_t).astype("float64")


@feature(
    group="partner_runway",
    outputs=OUTPUTS,
    version=1,
    doc="Landings, takeoffs and the last movement on the partner runway of a close parallel pair (LTFM, LFPG) before takeoff.",
)
def partner_runway(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    dep_all = mvt.loc[mvt["is_dep"]]
    arr_all = mvt.loc[~mvt["is_dep"]]
    q = mvt.loc[ctx.dep_index]

    dep_t, dep_g = epoch_seconds(dep_all["MVT_TIME_UTC_mvt"]), _runway_key(dep_all)
    arr_t, arr_g = epoch_seconds(arr_all["MVT_TIME_UTC_mvt"]), _runway_key(arr_all)
    all_t = pd.concat([dep_t, arr_t], ignore_index=True)
    all_g = pd.concat([dep_g, arr_g], ignore_index=True)

    q_t = epoch_seconds(q["MVT_TIME_UTC_mvt"])
    q_own = _runway_key(q)
    q_par = partner_key(q)
    has = (q_par.notna() & np.isfinite(q_t.to_numpy(dtype="float64", na_value=np.nan))).to_numpy()

    def masked(values: np.ndarray, cap: float | None = None) -> np.ndarray:
        out = np.asarray(values, dtype="float64").copy()
        if cap is not None:
            out = np.minimum(out, cap)
        out[~has] = np.nan
        return out.astype("float32")

    gap_arr = neighbour_gap(arr_t, arr_g, q_t, q_par, direction="backward", strict=True)
    gap_mvt = neighbour_gap(all_t, all_g, q_t, q_par, direction="backward", strict=True)
    pair_10 = _before(all_t, all_g, q_t, q_par, 600) + _before(all_t, all_g, q_t, q_own, 600)

    return pd.DataFrame(
        {
            "prw_has_partner": has.astype("int8"),
            "prw_arr_10": masked(_before(arr_t, arr_g, q_t, q_par, 600)),
            "prw_arr_20": masked(_before(arr_t, arr_g, q_t, q_par, 1200)),
            "prw_dep_10": masked(_before(dep_t, dep_g, q_t, q_par, 600)),
            "prw_gap_prev_arr": masked(gap_arr, GAP_CAP_S),
            "prw_gap_prev_mvt": masked(gap_mvt, GAP_CAP_S),
            "prw_pair_mvt_10": masked(pair_10),
        },
        index=ctx.dep_index,
    )
