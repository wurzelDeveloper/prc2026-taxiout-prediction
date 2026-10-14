"""What happened at this stand just before the aircraft left it.

The arrival that occupied the stand before this departure is informative twice
over: its taxi-in time is a fresh measurement of ground conditions at that exact
spot, and the time since it parked tells us whether this was a rushed
turnaround or an aircraft that had been sitting for hours.

This works in the ranking set because arrivals there keep their block times --
only departures are blanked. Matching is by ``merge_asof`` on takeoff time,
which is known for every flight, and pairs ~98% of departures.

`stand_asof` is the shared join; `prc.features.stand_occupancy` builds on it
for the arrival's own delay, the next occupant of the stand and the stand's
traffic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc.features.core import epoch_seconds
from prc.features.registry import FeatureContext, feature

ARRIVAL_COLUMNS = ("arr_inblock", "arr_landing", "arr_taxiin", "arr_runway", "arr_type", "arr_pos")
"""What `stand_asof` reports about the matched arrival, besides any ``extra`` columns."""


def stand_key(frame: pd.DataFrame) -> pd.Series:
    """``<ICAO>_<STAND_mvt>`` with a placeholder for missing stands, for departures or arrivals."""
    return frame["apt"].astype("string") + "_" + frame["STAND_mvt"].astype("string").fillna("__NA__")


def stand_asof(
    dep: pd.DataFrame,
    arr: pd.DataFrame,
    *,
    direction: str = "backward",
    allow_exact_matches: bool = True,
    extra: dict[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match every departure to the arrival at the same stand whose in-block time is nearest its takeoff.

    ``backward`` finds the arrival that last occupied the stand before takeoff,
    ``forward`` the next one to occupy it after takeoff (legal: arrivals keep
    their in-block times in 2026). Returns ``(merged, arrivals)``: ``merged`` is
    indexed like ``dep`` with ``t`` (takeoff epoch), ``stand_key``, the
    `ARRIVAL_COLUMNS` and any ``extra`` arrival columns (name -> array aligned
    with ``arr``); ``arrivals`` is the in-block-sorted arrival table that
    ``arr_pos`` indexes into, so a caller can walk to the previous occupant.
    """
    left = pd.DataFrame(
        {
            "t": epoch_seconds(dep["MVT_TIME_UTC_mvt"]).to_numpy(),
            "stand_key": stand_key(dep).to_numpy(),
        },
        index=dep.index,
    ).reset_index()

    # Anchor the arrival side on in-block time: that is the moment the stand was
    # actually occupied, and it is populated for arrivals even in ranking.
    inblock = epoch_seconds(arr["BLOCK_TIME_UTC_mvt"]).to_numpy()
    right = pd.DataFrame(
        {
            "t": inblock,
            # `t` is consumed as the merge key, so keep a copy to measure against.
            "arr_inblock": inblock,
            "stand_key": stand_key(arr).to_numpy(),
            "arr_landing": epoch_seconds(arr["MVT_TIME_UTC_mvt"]).to_numpy(),
            "arr_taxiin": arr["TAXITIME_SEC_mvt"].to_numpy(dtype="float64"),
            "arr_runway": arr["RUNWAY_mvt"].astype("string").to_numpy(),
            "arr_type": arr["AIRCRAFT_TYPE_mvt"].astype("string").to_numpy(),
        }
    )
    for name, values in (extra or {}).items():
        right[name] = np.asarray(values)
    right = right.dropna(subset=["t"]).sort_values("t", kind="stable").reset_index(drop=True)
    right["arr_pos"] = np.arange(len(right), dtype="int64")

    left = left.sort_values("t", kind="stable")
    merged = pd.merge_asof(
        left,
        right,
        on="t",
        by="stand_key",
        direction=direction,
        allow_exact_matches=allow_exact_matches,
        suffixes=("", "_arr"),
    ).set_index("mvt_id")
    return merged.reindex(dep.index), right


@feature(
    group="turnaround",
    outputs=(
        "turnaround_matched",
        "secs_since_prev_arr_inblock",
        "secs_since_prev_arr_landing",
        "prev_arr_taxiin_sec",
        "prev_arr_runway",
        "prev_arr_type_matches",
        "prev_arr_taxiin_vs_stand_med",
    ),
    depends_on=("core_keys", "stand_runway_keys"),
    categorical=("prev_arr_runway",),
    doc="The arrival that last occupied this stand.",
)
def prev_stand_arrival(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    dep = mvt.loc[ctx.dep_index]
    arr = mvt.loc[~mvt["is_dep"]]
    merged, _ = stand_asof(dep, arr, direction="backward")

    since_inblock = _gap(merged["t"], merged["arr_inblock"])

    # `merge_asof` finds the most recent arrival at that stand however long ago
    # that was -- at a quiet stand it can be days. A three-day-old arrival says
    # nothing about current ground conditions, so the arrival-derived columns
    # are gated on a recency window; the idle time itself stays, since a long
    # gap is meaningful in its own right.
    max_age = float(ctx.params.get("max_age_sec", 86_400))
    recent = since_inblock.le(max_age).fillna(False)

    taxiin = merged["arr_taxiin"].astype("float32").where(recent)
    # Centre the previous arrival's taxi-in on what is normal for that stand, so
    # the feature carries "conditions are slow right now" rather than "this
    # stand is far from the runway" (which `stand_runway` already covers).
    stand_med = taxiin.groupby(merged["stand_key"], observed=True).transform("median")
    arr_type = merged["arr_type"].where(recent)
    dep_type = dep["AIRCRAFT_TYPE_mvt"].astype("string")

    return pd.DataFrame(
        {
            "turnaround_matched": recent.astype("int8"),
            "secs_since_prev_arr_inblock": since_inblock,
            "secs_since_prev_arr_landing": _gap(merged["t"], merged["arr_landing"]),
            "prev_arr_taxiin_sec": taxiin.clip(upper=14_400),
            "prev_arr_runway": merged["arr_runway"].where(recent).fillna("__NA__").astype("category"),
            "prev_arr_type_matches": (dep_type.notna() & arr_type.notna() & dep_type.eq(arr_type)).astype("int8"),
            "prev_arr_taxiin_vs_stand_med": (taxiin - stand_med).astype("float32"),
        },
        index=ctx.dep_index,
    )


def _gap(now: pd.Series, then: pd.Series, cap: float = 172_800.0) -> pd.Series:
    """Seconds between two epoch-second series, with implausible gaps dropped."""
    d = (now - then).astype("float64")
    return d.where((d >= 0) & (d <= cap)).astype("float32")
