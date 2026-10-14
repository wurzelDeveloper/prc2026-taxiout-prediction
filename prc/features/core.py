"""Keys every other block builds on."""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc.features.registry import FeatureContext, feature


_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def epoch_seconds(ts: pd.Series) -> pd.Series:
    """UTC timestamps as float64 epoch seconds; NaT becomes NaN.

    Float rather than int so missing timestamps stay missing -- casting NaT
    through int64 silently yields INT64_MIN, which then reads as a real event at
    an absurd time and quietly corrupts every window count built on it.
    """
    return (ts - _EPOCH).dt.total_seconds()


def bucket_key(*parts: pd.Series) -> np.ndarray:
    """Fold several small-cardinality columns into one int64 grouping key.

    Grouping millions of rows by a tuple-valued MultiIndex is slow and depends
    on hashing behaviour; a single integer column is neither.

    Codes are derived per call, so keys from two separate calls are NOT
    comparable. Build one key over one frame and reuse it.
    """
    key = np.zeros(len(parts[0]), dtype="int64")
    for part in parts:
        codes = pd.Categorical(part).codes.astype("int64")
        key = key * (int(codes.max()) + 2) + (codes + 1)
    return key


@feature(
    group="core",
    outputs=("apt", "takeoff_epoch"),
    categorical=("apt",),
    doc="Reporting airport and takeoff time as epoch seconds.",
)
def core_keys(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    dep = mvt.loc[ctx.dep_index]
    return pd.DataFrame(
        {
            "apt": dep["apt"],
            "takeoff_epoch": epoch_seconds(dep["MVT_TIME_UTC_mvt"]).astype("int64"),
        },
        index=ctx.dep_index,
    )
