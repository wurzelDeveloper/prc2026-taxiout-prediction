"""Sham control for `partner_runway` (findings 10.28): the same columns, with their rows permuted among the
paired-runway departures of each airport.

The full-model A/B of the partner-runway block at Istanbul (exp070 against exp071) lowered the bulk RMSE by about
0.5 s, and the independent runway 36/18 -- no partner, NaN in every partner column -- by about as much. The gain may
therefore come from the block's structure (an indicator of the paired runways and a NaN pattern the trees can split
on) rather than from what the partner runway's traffic says about the flight's own moment. This block separates the
two. It computes `partner_runway`'s columns and permutes whole rows among the departures with a partner, per airport,
with one permutation for all columns (``features.params.partner_runway_sham.seed``, default 0). The indicator, the
NaN pattern and the joint distribution of the columns are kept; their association with the flight's own moment is
destroyed. exp072 = exp070 + this block: if exp072 matches exp071, the partner traffic adds nothing beyond the
structure.

Lab only: it carries no information about its row by design and is never in a submission config. Auto-imported by
the registry; the `partner_runway` module is imported as a module object so that `build._related_modules` hashes it
into this block's cache key.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc.features import partner_runway as _pr  # noqa: F401  (module object: hashed into this block's cache key)
from prc.features.registry import FeatureContext, feature

OUTPUTS = tuple("prws_" + c[len("prw_"):] for c in _pr.OUTPUTS)


def permute_partner_rows(frame: pd.DataFrame, apt: np.ndarray, seed: int) -> pd.DataFrame:
    """``frame`` (the `partner_runway` columns) with the rows of every column but the indicator permuted among the
    rows whose ``prw_has_partner`` is 1, separately per airport, one permutation for all columns."""
    rng = np.random.default_rng(seed)
    out = frame.copy()
    has = out["prw_has_partner"].to_numpy() == 1
    cols = [out.columns.get_loc(c) for c in _pr.OUTPUTS if c != "prw_has_partner"]
    for a in sorted(set(apt[has])):
        rows = np.flatnonzero(has & (apt == a))
        out.iloc[rows, cols] = frame.iloc[rng.permutation(rows), cols].to_numpy()
    return out


@feature(
    group="partner_runway_sham",
    outputs=OUTPUTS,
    version=1,
    doc="Sham control: the partner-runway columns with their rows permuted among the paired-runway departures (lab).",
)
def partner_runway_sham(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    real = _pr.partner_runway(mvt, ctx)
    apt = mvt.loc[ctx.dep_index, "apt"].astype("string").to_numpy(dtype=object, na_value="")
    out = permute_partner_rows(real, apt, int(ctx.params.get("seed", 0)))
    out.columns = list(OUTPUTS)
    return out
