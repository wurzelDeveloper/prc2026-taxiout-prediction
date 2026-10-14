"""Target encodings the trees do not build well on their own.

`stand_runway_target_stats` gives the taxi distance as the (stand, runway)
median. This block adds the other keys a taxi-out time depends on -- the
carrier at the airport (its push-back discipline and stands), the aircraft
type at the airport, the local hour at the airport -- and, for the
(stand, runway) pairing, the tail statistics a median hides: the share of
long taxis (an hour to four hours, not a stale schedule) and the 90th
percentile. Fold-dependent: everything is computed on the fold's fit rows and
looked up for every departure, with `min_support` and prior smoothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc.features.flight_number import LONG_HI_SEC, LONG_LO_SEC, PRIOR_WEIGHT, STALE_TOL_SEC, _smoothed_rate
from prc.features.registry import FeatureContext, feature


@feature(
    group="key_stats",
    outputs=(
        "y_med_carrier_apt",
        "y_med_type_apt",
        "y_med_apt_hour_local",
        "y_long_rate_stand_runway",
        "y_p90_stand_runway",
        "y_n_stand_runway",
    ),
    depends_on=("core_keys", "stand_runway_keys", "aircraft_identity", "calendar", "stale_schedule"),
    uses_target=True,
    doc="Fold-safe median taxi-out per carrier, type and local hour at the airport; long-taxi rate and p90 per stand/runway pairing.",
)
def key_target_stats(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    dep = mvt.loc[ctx.dep_index]
    min_support = int(ctx.params.get("min_support", 20))
    y = ctx.target.reindex(ctx.dep_index).to_numpy(dtype="float64")
    fit = ctx.fit_mask.reindex(ctx.dep_index, fill_value=False).to_numpy(dtype=bool) & np.isfinite(y)

    apt = dep["apt"].astype("string")
    ident = ctx.dep("aircraft_identity")
    keys = {
        "y_med_carrier_apt": apt + "|" + ident["carrier_flightno"].astype("string").fillna("__NA__"),
        "y_med_type_apt": apt + "|" + ident["aircraft_type"].astype("string").fillna("__NA__"),
        "y_med_apt_hour_local": apt + "|" + ctx.dep("calendar")["hour_local"].astype("string").fillna("__NA__"),
    }
    out = {}
    for name, key in keys.items():
        yf = pd.Series(y[fit], index=key[fit].to_numpy())
        grouped = yf.groupby(level=0)
        n = key.map(grouped.size()).fillna(0)
        out[name] = key.map(grouped.median()).where(n >= min_support).astype("float32")

    pair = ctx.dep("stand_runway_keys")["apt_stand_runway"].astype("string")
    delta = ctx.dep("stale_schedule")["sched_delta_full"].to_numpy(dtype="float64")
    with np.errstate(invalid="ignore"):
        stale = fit & (np.abs(y - delta) <= STALE_TOL_SEC) & (delta >= LONG_LO_SEC)
        long = fit & (y > LONG_LO_SEC) & (y <= LONG_HI_SEC) & ~stale
    long_rate, n_pair = _smoothed_rate(long, fit, pair, apt)
    yf = pd.Series(y[fit], index=pair[fit].to_numpy())
    p90 = pair.map(yf.groupby(level=0).quantile(0.9)).where(n_pair >= min_support)

    out["y_long_rate_stand_runway"] = long_rate
    out["y_p90_stand_runway"] = p90.astype("float32")
    out["y_n_stand_runway"] = n_pair.astype("float32")
    _ = PRIOR_WEIGHT  # the rate above uses `_smoothed_rate`'s prior weight; named here so the docstring's claim is traceable
    return pd.DataFrame(out, index=ctx.dep_index)
