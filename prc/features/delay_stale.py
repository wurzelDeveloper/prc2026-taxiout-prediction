"""Fold-safe stale rates among delayed departures, by carrier and pier.

A departure 30-120 min behind schedule whose block time is the schedule has a
target equal to the schedule delta (a "stale" record). Among such delayed
departures that happens on 9.1 % at LIRF, 3.4 % at EGLL, 2.3 % at LTFM and
under 1.1 % elsewhere, and it is a carrier and pier trait: one Rome carrier
reports the schedule on 99.7 % of its delayed flights, several on 24-32 %,
KLM, Swiss and TAP on under 1 %; Rome's pier 9 on 22 %, pier 4 on 2 %
(findings 8.2). `stale_target_stats` covers only the untyped NM-less segment;
this block gives the mixture's classifier the same kind of rates for every
delayed departure, plus the rate of records equal to NM's last off-block time.

Fold-dependent: the label is read only on ``ctx.fit_mask`` rows with a delay
in [`DELAY_LO_SEC`, `DELAY_HI_SEC`]; the rates are then looked up for every
departure, ranking rows included. Each rate is smoothed toward its airport's
rate with `PRIOR_WEIGHT` pseudo-rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc.features.registry import FeatureContext, feature
from prc.features.stale_schedule import carrier_prefix
from prc.features.stand_runway import stand_prefix_key

DELAY_LO_SEC = 1800.0
DELAY_HI_SEC = 7200.0
MATCH_TOL_SEC = 60.0
PRIOR_WEIGHT = 20.0

DELAY_STALE = (
    "dstale_rate_carrier",
    "dstale_n_carrier",
    "dstale_rate_pier",
    "dstale_n_pier",
    "dstale_rate_carrier_pier",
    "dstale_n_carrier_pier",
    "dstale_lobt_rate_carrier",
    "dstale_rate_apt",
)


def _rates(label: pd.Series, fit: np.ndarray, key: pd.Series, apt: pd.Series, apt_rate: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    counts = label[fit].groupby(key[fit].to_numpy()).agg(["sum", "size"])
    n = key.map(counts["size"]).fillna(0).to_numpy(dtype="float64")
    s = key.map(counts["sum"]).fillna(0).to_numpy(dtype="float64")
    prior = apt.map(apt_rate).fillna(0).to_numpy(dtype="float64")
    return (s + PRIOR_WEIGHT * prior) / (n + PRIOR_WEIGHT), n


@feature(
    group="delay_stale",
    outputs=DELAY_STALE,
    depends_on=("core_keys", "stale_schedule"),
    uses_target=True,
    doc="Smoothed share of block times equal to the schedule (and to NM's last off-block time) among the fold's departures 30-120 min late, by carrier, pier and both, per airport.",
)
def delay_stale_stats(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    dep = mvt.loc[ctx.dep_index]
    st = ctx.dep("stale_schedule")
    sched = st["sched_delta_full"].to_numpy(dtype="float64")
    lobt = st["lobt_delta_full"].to_numpy(dtype="float64")
    target = ctx.target.reindex(ctx.dep_index).to_numpy(dtype="float64")
    in_band = (sched >= DELAY_LO_SEC) & (sched <= DELAY_HI_SEC)
    fit = ctx.fit_mask.reindex(ctx.dep_index, fill_value=False).to_numpy(dtype=bool) & in_band & np.isfinite(target)

    apt = dep["apt"].astype("string")
    stale = pd.Series(np.abs(target - sched) <= MATCH_TOL_SEC, index=ctx.dep_index)
    lobt_match = pd.Series((np.abs(target - lobt) <= MATCH_TOL_SEC) & ~stale.to_numpy(), index=ctx.dep_index)
    apt_rate = stale[fit].groupby(apt[fit].to_numpy()).mean()
    apt_lobt_rate = lobt_match[fit].groupby(apt[fit].to_numpy()).mean()

    carrier = apt + "|" + carrier_prefix(dep["FLIGHT_mvt"])
    pier = stand_prefix_key(dep["apt"], dep["STAND_mvt"])
    keys = {"carrier": carrier, "pier": pier, "carrier_pier": carrier + "|" + pier}

    out = {}
    for name, key in keys.items():
        rate, n = _rates(stale, fit, key, apt, apt_rate)
        out[f"dstale_rate_{name}"] = rate.astype("float32")
        out[f"dstale_n_{name}"] = n.astype("float32")
    lobt_rate, _ = _rates(lobt_match, fit, carrier, apt, apt_lobt_rate)
    out["dstale_lobt_rate_carrier"] = lobt_rate.astype("float32")
    out["dstale_rate_apt"] = apt.map(apt_rate).fillna(0).to_numpy(dtype="float32")
    return pd.DataFrame(out, index=ctx.dep_index)[list(DELAY_STALE)]
