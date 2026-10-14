"""The stale-block regime: departures whose block time is a schedule, not a measurement.

Measured on the 2025 training data: 584 departures have a taxi-out time above
two hours, and for 438 of them the target equals takeoff minus
``SCHED_TIME_UTC_mvt`` to within two minutes (507 within thirty). Their recorded
block time is the scheduled off-block time, so the "taxi time" is a departure
delay, up to 131,167 s. The regime is concentrated in one segment: departures
with no ``AIRCRAFT_TYPE_mvt`` and no NM join (1,487 training rows, all but 20 of
them at LIRF; the 2026 ranking set holds 383 twins, 181 with a delta above two
hours). About half of the segment is stale (target == delta) and half genuine
(a normal taxi while the delta is hours); which half a row falls in follows the
carrier and the destination more than anything else. The segment carries about
42 % of the holdout squared error of the best model without this block.

Two things stop a GBDT from learning this from ``nm_deltas.sched_delta`` alone:
that column is NaN beyond one day (the very rows that matter), and CatBoost
quantises a feature into ``border_count`` bins, so every delta above two hours
shares one bin. This block therefore hands the model the exact delta as a
*residual base*: ``stale_base`` is the delta on suspect rows and NaN elsewhere.
With ``target.residual_of: stale_base`` the model fits ``y - delta`` on the
segment (zero for a stale row, minus the delay for a genuine one) and ``y``
everywhere else, and the exact delta is added back outside the trees. A
negative delta (schedule after takeoff: the schedule belongs to another leg)
cannot be a stale block, so those rows keep a NaN base; with the base set, the
model dragged two such holdout rows to -20,000 s.

Which half of the segment a row belongs to follows the carrier and the
destination: some carriers are stale in every row, others in none. The
fold-dependent block below turns that into smoothed stale rates computed from
the fold's fit rows, the same way `stand_runway_target_stats` works.

``model.mixture`` (`prc.models.mixture`) is the other consumer of this block:
it reads ``sched_delta_full`` (and ``lobt_delta_full``, takeoff minus NM's last
off-block time, unclipped) as candidate block times and ``stale_suspect`` as
the segment marker, and predicts the classifier-weighted mixture of the
candidates and the regressor instead of a residual. Mind that NM's LOBT is
close to the real off-block time on ordinary rows, so ``lobt_delta_full`` is a
taxi time there; as a candidate it only makes sense above a floor of hours.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc.features.registry import FeatureContext, feature

DAY = 86_400.0
STALE_TOL_SEC = 600.0
"""A fit row counts as stale when its target is within this of the schedule delta."""
PRIOR_WEIGHT = 5.0
"""Pseudo-count of the segment-wide stale rate mixed into every per-key rate."""


def is_untyped(aircraft_type: pd.Series) -> pd.Series:
    s = aircraft_type.astype("string")
    return (s.isna() | (s.str.strip() == "")).fillna(True)


def carrier_prefix(flight: pd.Series) -> pd.Series:
    """The carrier code at the front of ``FLIGHT_mvt`` (IATA or ICAO, as delivered)."""
    return flight.astype("string").str.extract(r"^([A-Z0-9]{2,3}?)(?=\d)", expand=False).fillna("__NA__")


@feature(
    group="stale",
    outputs=(
        "stale_base",
        "stale_suspect",
        "sched_delta_full",
        "sched_delta_days",
        "sched_delta_day_frac",
        "lobt_delta_full",
        "typed_unjoined",
        "lobt_minus_sched",
        "iobt_minus_sched",
    ),
    depends_on=("core_keys",),
    version=4,
    doc="Takeoff minus schedule as a residual base on the untyped, NM-less segment; the unclipped schedule and LOBT deltas for everyone.",
)
def stale_schedule(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    dep = mvt.loc[ctx.dep_index]
    delta = (dep["MVT_TIME_UTC_mvt"] - dep["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    lobt_delta = (dep["MVT_TIME_UTC_mvt"] - dep["LOBT_flt"]).dt.total_seconds()
    # NM's off-block times against the schedule. On a flight delayed by hours NM
    # normally carries the delayed off-block; when its last or initial off-block
    # time still equals the schedule, the movement record's block time is a
    # schedule too in 40 % of cases (0.07 % otherwise) -- the fingerprint of a
    # stale block on a typed, NM-joined row. Unclipped, NaN without a join.
    lobt_minus_sched = (dep["LOBT_flt"] - dep["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    iobt_minus_sched = (dep["IOBT_flt"] - dep["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    untyped = is_untyped(dep["AIRCRAFT_TYPE_mvt"])
    unjoined = dep["FLIGHT_ID_mvt"].isna()
    suspect = untyped & unjoined

    days = np.floor(delta / DAY)
    return pd.DataFrame(
        {
            # The residual base: exact seconds on suspect rows with a non-negative
            # delta, NaN elsewhere (NaN means "learn the raw target", see
            # train._prepare_targets).
            "stale_base": delta.where(suspect & (delta >= 0)).astype("float32"),
            "stale_suspect": suspect.astype("int8"),
            # Unlike nm_deltas.sched_delta this keeps multi-day values.
            "sched_delta_full": delta.astype("float32"),
            "sched_delta_days": days.astype("float32"),
            "sched_delta_day_frac": ((delta - days * DAY) / DAY).astype("float32"),
            # Unlike nm_deltas.lobt_delta this keeps multi-day values; NaN without an NM join.
            "lobt_delta_full": lobt_delta.astype("float32"),
            # Typed but without an NM join: the neighbouring segment, rarely stale.
            "typed_unjoined": (~untyped & unjoined).astype("int8"),
            "lobt_minus_sched": lobt_minus_sched.astype("float32"),
            "iobt_minus_sched": iobt_minus_sched.astype("float32"),
        },
        index=ctx.dep_index,
    )


@feature(
    group="stale",
    outputs=(
        "stale_rate_carrier",
        "stale_n_carrier",
        "stale_rate_ades",
        "stale_n_ades",
        "stale_rate_carrier_ades",
        "stale_n_carrier_ades",
    ),
    depends_on=("core_keys", "stale_schedule"),
    uses_target=True,
    doc="Smoothed share of stale blocks among the fold's fit rows of the suspect segment, by carrier, destination and both.",
)
def stale_target_stats(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """Fold-dependent: the label (target within `STALE_TOL_SEC` of the delta) is
    only ever read on ``ctx.fit_mask`` rows that carry a residual base, and the
    rates are then looked up for every departure, ranking rows included."""
    dep = mvt.loc[ctx.dep_index]
    base = ctx.dep("stale_schedule")["stale_base"]
    fit = ctx.fit_mask.reindex(ctx.dep_index, fill_value=False).to_numpy(dtype=bool) & base.notna().to_numpy()
    target = ctx.target.reindex(ctx.dep_index).to_numpy(dtype="float64")
    stale = pd.Series(np.abs(target - base.to_numpy(dtype="float64")) <= STALE_TOL_SEC, index=ctx.dep_index)
    prior = float(stale[fit].mean()) if fit.any() else 0.0

    keys = {
        "carrier": carrier_prefix(dep["FLIGHT_mvt"]),
        "ades": dep["ADES_mvt"].astype("string").fillna("__NA__"),
    }
    keys["carrier_ades"] = keys["carrier"] + "|" + keys["ades"]

    out = {}
    for name, key in keys.items():
        counts = stale[fit].groupby(key[fit].to_numpy()).agg(["sum", "size"])
        n = key.map(counts["size"]).fillna(0).to_numpy(dtype="float64")
        s = key.map(counts["sum"]).fillna(0).to_numpy(dtype="float64")
        out[f"stale_rate_{name}"] = ((s + PRIOR_WEIGHT * prior) / (n + PRIOR_WEIGHT)).astype("float32")
        out[f"stale_n_{name}"] = n.astype("float32")
    return pd.DataFrame(out, index=ctx.dep_index)
