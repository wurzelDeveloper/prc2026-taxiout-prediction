"""Where the aircraft starts and where it takes off from.

Taxi-out time is, first of all, a distance. The (stand, runway) pair encodes
that distance, and `stand_runway_target_stats` recovers it from history: the
median taxi time of the pairing. That block reads the target and is therefore
fold-dependent: it is recomputed inside every CV fold from that fold's
training rows only. `prc.features.geometry` measures the same distance from
airport layout data instead (`group:geometry`); the two are meant to be
compared, not chosen between blindly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc.features.registry import FeatureContext, feature

STAND_PATTERN = r"^(?P<prefix>[A-Za-z]*)(?P<number>\d*)(?P<suffix>[A-Za-z]*)$"
"""Stand IDs are structured: a pier/terminal letter, a number, sometimes a position suffix ("A21L")."""


def parse_stand(stand: pd.Series) -> pd.DataFrame:
    """``prefix`` / ``number`` / ``suffix`` of each stand ID; all NaN where the ID has another shape."""
    return stand.astype("string").fillna("__NA__").str.extract(STAND_PATTERN)


def stand_prefix_key(apt: pd.Series, stand: pd.Series) -> pd.Series:
    """``<ICAO>_<prefix>``, the pier key `stand_runway_keys` emits as ``stand_prefix``, for any rows (arrivals too)."""
    return apt.astype("string") + "_" + parse_stand(stand)["prefix"].fillna("")


@feature(
    group="stand_runway",
    outputs=(
        "runway",
        "stand",
        "apt_runway",
        "apt_stand",
        "apt_stand_runway",
        "stand_prefix",
        "stand_number",
        "stand_suffix",
        "ades",
        "ades_country",
        "is_domestic",
    ),
    depends_on=("core_keys",),
    categorical=(
        "runway",
        "stand",
        "apt_runway",
        "apt_stand",
        "apt_stand_runway",
        "stand_prefix",
        "stand_suffix",
        "ades",
        "ades_country",
    ),
    doc="Stand/runway identity, composite keys and parsed stand structure.",
)
def stand_runway_keys(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    dep = mvt.loc[ctx.dep_index]
    apt = dep["apt"].astype("string")
    rwy = dep["RUNWAY_mvt"].astype("string").fillna("__NA__")
    stand = dep["STAND_mvt"].astype("string").fillna("__NA__")
    ades = dep["ADES_mvt"].astype("string").fillna("__NA__")

    # The prefix generalises to stands never seen in training; the raw ID does not.
    parsed = parse_stand(stand)

    return pd.DataFrame(
        {
            "runway": rwy.astype("category"),
            "stand": stand.astype("category"),
            "apt_runway": (apt + "_" + rwy).astype("category"),
            "apt_stand": (apt + "_" + stand).astype("category"),
            "apt_stand_runway": (apt + "_" + stand + "_" + rwy).astype("category"),
            "stand_prefix": stand_prefix_key(apt, stand).astype("category"),
            "stand_number": pd.to_numeric(parsed["number"], errors="coerce").astype("Int32"),
            "stand_suffix": parsed["suffix"].fillna("").astype("category"),
            "ades": ades.astype("category"),
            "ades_country": ades.str[:2].astype("category"),
            "is_domestic": (ades.str[:2] == apt.str[:2]).astype("int8"),
        },
        index=ctx.dep_index,
    )


@feature(
    group="stand_runway",
    outputs=("stand_freq", "runway_freq", "stand_runway_freq", "runway_share_at_stand"),
    depends_on=("core_keys", "stand_runway_keys"),
    doc="How common this stand / runway / pairing is at the airport.",
)
def stand_runway_frequency(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """Frequency encodings.

    Computed over *all* departures including ranking: these count rows, never
    targets, so there is nothing to leak and the 2026 months get the same
    treatment as 2025.
    """
    keys = ctx.dep("stand_runway_keys")
    stand, rwy, pair = keys["apt_stand"], keys["apt_runway"], keys["apt_stand_runway"]

    stand_n = stand.map(stand.value_counts()).astype("float32")
    rwy_n = rwy.map(rwy.value_counts()).astype("float32")
    pair_n = pair.map(pair.value_counts()).astype("float32")

    return pd.DataFrame(
        {
            "stand_freq": stand_n,
            "runway_freq": rwy_n,
            "stand_runway_freq": pair_n,
            # Is this a routine runway for this stand, or an unusual routing?
            "runway_share_at_stand": (pair_n / stand_n).astype("float32"),
        },
        index=ctx.dep_index,
    )


# Fallback chain, most specific first. Each level needs `min_support` observed
# taxi times before it is trusted; otherwise we fall through to a coarser one.
_TARGET_STAT_LEVELS = (
    ("apt_stand_runway", "sr"),
    ("apt_stand", "s"),
    ("apt_runway", "r"),
    ("apt", "a"),
)


@feature(
    group="stand_runway",
    outputs=(
        "taxi_med_stand_runway",
        "taxi_med_stand",
        "taxi_med_runway",
        "taxi_med_apt",
        "taxi_med_hier",
        "taxi_iqr_hier",
        "taxi_med_hier_level",
        "taxi_med_hier_support",
    ),
    depends_on=("core_keys", "stand_runway_keys"),
    categorical=("taxi_med_hier_level",),
    uses_target=True,
    doc="Historical taxi time for this stand/runway pairing (fold-safe target stats).",
)
def stand_runway_target_stats(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """A learned taxi-distance matrix.

    With no airport layout data, the median historical taxi time for a
    (stand, runway) pair *is* the distance estimate. Rare pairings fall back
    through stand, then runway, then airport, and the level actually used is
    exposed as a feature so the model can discount thin estimates.

    Fitted strictly on ``ctx.fit_mask`` rows -- the training half of the current
    fold -- so a row never contributes to its own encoding.
    """
    keys = ctx.dep("stand_runway_keys").copy()
    keys["apt"] = mvt.loc[ctx.dep_index, "apt"].astype("string")

    min_support = int(ctx.params.get("min_support", 20))
    fit_rows = ctx.fit_mask.reindex(ctx.dep_index).fillna(False)
    y = ctx.target.reindex(ctx.dep_index)
    fit = keys.loc[fit_rows]
    y_fit = y.loc[fit_rows]

    out: dict[str, pd.Series] = {}
    stats: dict[str, pd.DataFrame] = {}
    for col, short in _TARGET_STAT_LEVELS:
        g = y_fit.groupby(fit[col].astype("string"), observed=True)
        agg = pd.DataFrame(
            {
                "med": g.median(),
                "iqr": g.quantile(0.75) - g.quantile(0.25),
                "n": g.size(),
            }
        )
        agg.loc[agg["n"] < min_support, ["med", "iqr"]] = np.nan
        stats[short] = agg
        mapped = keys[col].astype("string").map(agg["med"])
        out[short] = mapped.astype("float32")

    hier = pd.Series(np.nan, index=ctx.dep_index, dtype="float32")
    hier_iqr = pd.Series(np.nan, index=ctx.dep_index, dtype="float32")
    level = pd.Series("none", index=ctx.dep_index, dtype="object")
    support = pd.Series(0.0, index=ctx.dep_index, dtype="float32")

    for col, short in _TARGET_STAT_LEVELS:
        need = hier.isna()
        if not need.any():
            break
        k = keys.loc[need, col].astype("string")
        hier[need] = k.map(stats[short]["med"]).astype("float32")
        hier_iqr[need] = k.map(stats[short]["iqr"]).astype("float32")
        filled = need & hier.notna()
        level[filled] = short
        support[filled] = keys.loc[filled, col].astype("string").map(stats[short]["n"]).astype("float32")

    return pd.DataFrame(
        {
            "taxi_med_stand_runway": out["sr"],
            "taxi_med_stand": out["s"],
            "taxi_med_runway": out["r"],
            "taxi_med_apt": out["a"],
            "taxi_med_hier": hier,
            "taxi_iqr_hier": hier_iqr,
            "taxi_med_hier_level": level.astype("category"),
            "taxi_med_hier_support": support,
        },
        index=ctx.dep_index,
    )
