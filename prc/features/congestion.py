"""How busy the airport and the runway are around this takeoff.

Every window here is anchored on **takeoff time**, which the ranking set gives
us for every flight. That is the unusual affordance of this competition: we may
look *forward* as well as backward, so "how many aircraft used this runway in
the ten minutes either side of me" is a legal feature, not a leak. Only the
departure's own off-block time and taxi time are withheld, and `prc.data` has
already masked those out of the frame these functions receive.

Counting is done with `np.searchsorted` over a per-group sorted array rather
than `groupby(...).rolling('15min')` -- the latter does not accept a value
series without a DatetimeIndex, and is orders of magnitude slower.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc.features.core import bucket_key, epoch_seconds
from prc.features.registry import FeatureContext, feature


def _keyed(times: np.ndarray, codes: np.ndarray, span: int) -> np.ndarray:
    """Fold (group, time) into a single sortable int64 so one searchsorted serves all groups."""
    return codes.astype("int64") * span + times


def _offsets(back, fwd, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Per-query window offsets in whole seconds.

    ``back``/``fwd`` are scalars (one window for every query) or arrays of
    length ``n`` (a window per query, e.g. "since this flight's own NM
    off-block time"). A non-finite offset means the query has no window, and
    the caller scores it as empty.
    """
    b = np.broadcast_to(np.asarray(back, dtype="float64"), (n,))
    f = np.broadcast_to(np.asarray(fwd, dtype="float64"), (n,))
    ok = np.isfinite(b) & np.isfinite(f)
    b = np.where(ok, b, 0.0).astype("int64")
    f = np.where(ok, f, 0.0).astype("int64")
    max_offset = int(max(np.abs(b).max(initial=0), np.abs(f).max(initial=0)))
    return b, f, ok, max_offset


def _prepare(ref_t: pd.Series, ref_g: pd.Series, q_t: pd.Series, q_g: pd.Series, max_offset: int):
    """Shared setup: common group coding, time normalisation, sorted reference keys.

    Both sides are coded against one shared category list, so reference and
    query keys are directly comparable. Events with a missing time or an
    unknown group are dropped from the reference and score zero as queries.
    The last value returned is the position (in the original reference
    arrays) of every kept event in key order, so reference *values* can be
    reduced over the same windows.
    """
    cats = pd.Index(sorted(set(ref_g.dropna().unique()) | set(q_g.dropna().unique())))
    ref_codes = pd.Categorical(ref_g, categories=cats).codes.astype("int64")
    q_codes = pd.Categorical(q_g, categories=cats).codes.astype("int64")

    ref_raw = ref_t.to_numpy(dtype="float64", na_value=np.nan)
    q_raw = q_t.to_numpy(dtype="float64", na_value=np.nan)
    valid_ref = (ref_codes >= 0) & np.isfinite(ref_raw)
    valid_q = (q_codes >= 0) & np.isfinite(q_raw)

    finite = np.concatenate([ref_raw[valid_ref], q_raw[valid_q]])
    t0 = finite.min() if finite.size else 0.0

    ref_times = np.where(valid_ref, ref_raw - t0, 0).astype("int64")
    q_times = np.where(valid_q, q_raw - t0, 0).astype("int64")

    span = int(max(ref_times.max(initial=0), q_times.max(initial=0))) + 2 * max_offset + 2
    keyed = _keyed(ref_times[valid_ref], ref_codes[valid_ref], span)
    argsort = np.argsort(keyed, kind="stable")
    order = np.flatnonzero(valid_ref)[argsort]
    return keyed[argsort], q_times, np.where(valid_q, q_codes, -1), span, order


def _bounds(keys, q_times, q_codes, span, back_i, fwd_i) -> tuple[np.ndarray, np.ndarray]:
    """[lo, hi) positions in ``keys`` of the events within each query's window."""
    lo_t = np.clip(q_times - back_i, 0, span - 1)
    hi_t = np.clip(q_times + fwd_i, 0, span - 1)
    lo = np.searchsorted(keys, _keyed(lo_t, q_codes, span), side="left")
    hi = np.searchsorted(keys, _keyed(hi_t, q_codes, span), side="right")
    return lo, np.maximum(hi, lo)


def window_count(
    ref_t: pd.Series,
    ref_g: pd.Series,
    q_t: pd.Series,
    q_g: pd.Series,
    *,
    back,
    fwd,
) -> np.ndarray:
    """Reference events within [t-back, t+fwd] of each query, within its own group.

    ``back`` and ``fwd`` are seconds, either one scalar each or one value per
    query (see `_offsets`); a query with a non-finite window scores 0.
    """
    back_i, fwd_i, ok, max_offset = _offsets(back, fwd, len(q_t))
    keys, q_times, q_codes, span, _ = _prepare(ref_t, ref_g, q_t, q_g, max_offset)
    lo, hi = _bounds(keys, q_times, q_codes, span, back_i, fwd_i)
    counts = (hi - lo).astype("int32")
    counts[(q_codes < 0) | ~ok] = 0
    return counts


def count_upto(ref_t: pd.Series, ref_g: pd.Series, q_t: pd.Series, q_g: pd.Series) -> np.ndarray:
    """Reference events at or before each query time, within its own group."""
    keys, q_times, q_codes, span, _ = _prepare(ref_t, ref_g, q_t, q_g, 0)
    lo = np.searchsorted(keys, _keyed(np.zeros_like(q_times), q_codes, span), side="left")
    hi = np.searchsorted(keys, _keyed(q_times, q_codes, span), side="right")
    counts = (hi - lo).astype("int32")
    counts[q_codes < 0] = 0
    return counts


REDUCTIONS = ("sum", "mean", "max")
MAX_REDUCE_WIDTH = 5000
"""Most events one window may hold for the ``max`` reduction, which walks the window in passes."""


def window_reduce(
    ref_t: pd.Series,
    ref_g: pd.Series,
    ref_v: pd.Series,
    q_t: pd.Series,
    q_g: pd.Series,
    *,
    back,
    fwd,
    how: str = "mean",
) -> np.ndarray:
    """Reduce the reference *values* whose events fall within [t-back, t+fwd] of each query, within its group.

    ``how`` is ``sum``, ``mean`` or ``max``. Missing values are skipped, and a
    window with nothing in it gives NaN (never 0: "no arrival landed in the
    last hour" is not the same as "the arrivals taxied normally"). Windows
    follow the `window_count` conventions, per-query offsets included.
    """
    if how not in REDUCTIONS:
        raise ValueError(f"how must be one of {REDUCTIONS}, got {how!r}")
    n = len(q_t)
    back_i, fwd_i, ok, max_offset = _offsets(back, fwd, n)
    keys, q_times, q_codes, span, order = _prepare(ref_t, ref_g, q_t, q_g, max_offset)
    lo, hi = _bounds(keys, q_times, q_codes, span, back_i, fwd_i)
    valid_q = (q_codes >= 0) & ok

    v = ref_v.to_numpy(dtype="float64", na_value=np.nan)[order]
    finite = np.isfinite(v)
    out = np.full(n, np.nan)
    if how in ("sum", "mean"):
        csum = np.concatenate([[0.0], np.cumsum(np.where(finite, v, 0.0))])
        ccnt = np.concatenate([[0], np.cumsum(finite)])
        cnt = ccnt[hi] - ccnt[lo]
        total = csum[hi] - csum[lo]
        has = valid_q & (cnt > 0)
        out[has] = total[has] if how == "sum" else total[has] / cnt[has]
        return out

    width = int((hi - lo).max(initial=0))
    if width > MAX_REDUCE_WIDTH:
        raise ValueError(f"a window holds {width} events; `max` is meant for windows of at most {MAX_REDUCE_WIDTH}")
    vf = np.where(finite, v, -np.inf)
    best = np.full(n, -np.inf)
    for k in range(width):
        idx = lo + k
        take = valid_q & (idx < hi)
        best[take] = np.maximum(best[take], vf[idx[take]])
    found = np.isfinite(best)
    out[found] = best[found]
    return out


def neighbour_gap(
    ref_t: pd.Series,
    ref_g: pd.Series,
    q_t: pd.Series,
    q_g: pd.Series,
    *,
    direction: str = "backward",
    strict: bool = True,
) -> np.ndarray:
    """Seconds from each query to the nearest reference event in its own group.

    ``backward`` looks at events at or before the query, ``forward`` at or
    after it; ``strict`` leaves out events at exactly the query time (so a set
    queried against itself does not find itself). NaN when there is none.
    """
    if direction not in ("backward", "forward"):
        raise ValueError(f"direction must be 'backward' or 'forward', got {direction!r}")
    keys, q_times, q_codes, span, _ = _prepare(ref_t, ref_g, q_t, q_g, 0)
    out = np.full(len(q_t), np.nan)
    if not len(keys):
        return out
    q_keys = _keyed(q_times, q_codes, span)
    if direction == "backward":
        pos = np.searchsorted(keys, q_keys, side="left" if strict else "right") - 1
        in_range = pos >= 0
    else:
        pos = np.searchsorted(keys, q_keys, side="right" if strict else "left")
        in_range = pos < len(keys)
    pos = np.clip(pos, 0, len(keys) - 1)
    same_group = (q_codes >= 0) & in_range & (keys[pos] // span == q_codes)
    ref_times = keys[pos] % span
    gap = q_times - ref_times if direction == "backward" else ref_times - q_times
    out[same_group] = gap[same_group].astype("float64")
    return out


@feature(
    group="congestion",
    outputs=(
        "rwy_dep_back_5m",
        "rwy_dep_back_15m",
        "rwy_dep_back_30m",
        "rwy_dep_back_60m",
        "rwy_dep_pm10m",
        "rwy_gap_prev_takeoff",
        "rwy_gap_next_takeoff",
        "rwy_dep_fwd_15m",
    ),
    depends_on=("core_keys",),
    doc="Departure pressure on the same runway around takeoff.",
)
def runway_queue(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    dep_all = mvt.loc[mvt["is_dep"]]
    ref_t = epoch_seconds(dep_all["MVT_TIME_UTC_mvt"])
    ref_g = dep_all["apt"].astype("string") + "_" + dep_all["RUNWAY_mvt"].astype("string").fillna("__NA__")

    q = mvt.loc[ctx.dep_index]
    q_t = epoch_seconds(q["MVT_TIME_UTC_mvt"])
    q_g = q["apt"].astype("string") + "_" + q["RUNWAY_mvt"].astype("string").fillna("__NA__")

    def cnt(back: int, fwd: int) -> np.ndarray:
        return window_count(ref_t, ref_g, q_t, q_g, back=back, fwd=fwd)

    # Gaps to the neighbouring takeoff on the same runway: a direct read on
    # whether this aircraft was one of a rapid-fire sequence or alone.
    order = ref_t.groupby(ref_g, observed=True)
    prev_gap = (ref_t - order.shift(1)).reindex(ctx.dep_index)
    next_gap = (order.shift(-1) - ref_t).reindex(ctx.dep_index)

    return pd.DataFrame(
        {
            "rwy_dep_back_5m": cnt(300, 0),
            "rwy_dep_back_15m": cnt(900, 0),
            "rwy_dep_back_30m": cnt(1800, 0),
            "rwy_dep_back_60m": cnt(3600, 0),
            "rwy_dep_pm10m": cnt(600, 600),
            "rwy_gap_prev_takeoff": prev_gap.clip(upper=7200).astype("float32"),
            "rwy_gap_next_takeoff": next_gap.clip(upper=7200).astype("float32"),
            "rwy_dep_fwd_15m": cnt(0, 900),
        },
        index=ctx.dep_index,
    )


@feature(
    group="congestion",
    outputs=(
        "apt_dep_back_15m",
        "apt_dep_back_60m",
        "apt_arr_back_15m",
        "apt_arr_back_60m",
        "apt_arr_taxiing_now",
        "apt_movements_pm30m",
        "apt_runways_in_use_hour",
        "apt_dep_share_this_rwy_60m",
    ),
    depends_on=("core_keys", "runway_queue"),
    doc="Airport-wide traffic load and ground occupancy at takeoff time.",
)
def airport_load(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """Airport-level load, including how many arrivals are still on the taxiways.

    ``apt_arr_taxiing_now`` counts arrivals that have landed but not yet reached
    a stand -- genuine competition for taxiway capacity. It is computable for
    the 2026 months only because ranking keeps arrival block times.
    """
    dep_all = mvt.loc[mvt["is_dep"]]
    arr_all = mvt.loc[~mvt["is_dep"]]

    q = mvt.loc[ctx.dep_index]
    q_t = epoch_seconds(q["MVT_TIME_UTC_mvt"])
    q_apt = q["apt"].astype("string")

    dep_t, dep_apt = epoch_seconds(dep_all["MVT_TIME_UTC_mvt"]), dep_all["apt"].astype("string")
    arr_t, arr_apt = epoch_seconds(arr_all["MVT_TIME_UTC_mvt"]), arr_all["apt"].astype("string")
    arr_block = epoch_seconds(arr_all["BLOCK_TIME_UTC_mvt"])

    landed = count_upto(arr_t, arr_apt, q_t, q_apt)
    parked = count_upto(arr_block, arr_apt, q_t, q_apt)

    dep_60 = window_count(dep_t, dep_apt, q_t, q_apt, back=3600, fwd=0)
    rwy_60 = ctx.dep("runway_queue")["rwy_dep_back_60m"].to_numpy()

    # Distinct runways handling departures in the same clock hour: a proxy for
    # the runway configuration in force, which the data never states directly.
    q_hour = pd.Series(q_t.to_numpy() // 3600, index=q.index)
    hour_key = bucket_key(q_apt, q_hour)
    n_rwy = (
        pd.DataFrame({"k": hour_key, "rwy": q["RUNWAY_mvt"].astype("string").to_numpy()})
        .groupby("k")["rwy"]
        .nunique()
    )
    runways_in_use = pd.Series(hour_key, index=ctx.dep_index).map(n_rwy)

    return pd.DataFrame(
        {
            "apt_dep_back_15m": window_count(dep_t, dep_apt, q_t, q_apt, back=900, fwd=0),
            "apt_dep_back_60m": dep_60,
            "apt_arr_back_15m": window_count(arr_t, arr_apt, q_t, q_apt, back=900, fwd=0),
            "apt_arr_back_60m": window_count(arr_t, arr_apt, q_t, q_apt, back=3600, fwd=0),
            "apt_arr_taxiing_now": np.maximum(landed - parked, 0).astype("int32"),
            "apt_movements_pm30m": (
                window_count(dep_t, dep_apt, q_t, q_apt, back=1800, fwd=1800)
                + window_count(arr_t, arr_apt, q_t, q_apt, back=1800, fwd=1800)
            ).astype("int32"),
            "apt_runways_in_use_hour": runways_in_use.astype("Int16"),
            "apt_dep_share_this_rwy_60m": np.divide(
                rwy_60, dep_60, out=np.zeros(len(q), dtype="float64"), where=dep_60 > 0
            ).astype("float32"),
        },
        index=ctx.dep_index,
    )


@feature(
    group="congestion",
    outputs=("apt_dep_load_ratio", "apt_dep_load_typical", "rwy_dep_load_ratio"),
    depends_on=("core_keys", "calendar", "airport_load", "runway_queue"),
    doc="Traffic load relative to what is normal for this airport at this hour.",
)
def load_normalised(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """Raw counts are not comparable across airports; ratios are.

    Twenty departures an hour is a quiet morning at LTFM and a rush at LSZH. The
    ratio against that airport's own (day-of-week, hour) median makes the
    congestion signal mean the same thing everywhere.
    """
    cal = ctx.dep("calendar")
    load = ctx.dep("airport_load")
    rwy = ctx.dep("runway_queue")
    apt = mvt.loc[ctx.dep_index, "apt"].astype("string")

    key = bucket_key(apt, cal["dow"].astype("float"), cal["hour_local"].astype("float"))
    typical = load["apt_dep_back_60m"].groupby(key).transform("median")
    rwy_typical = rwy["rwy_dep_back_60m"].groupby(key).transform("median")

    return pd.DataFrame(
        {
            "apt_dep_load_ratio": (load["apt_dep_back_60m"] / typical.replace(0, np.nan)).astype("float32"),
            "apt_dep_load_typical": typical.astype("float32"),
            "rwy_dep_load_ratio": (rwy["rwy_dep_back_60m"] / rwy_typical.replace(0, np.nan)).astype("float32"),
        },
        index=ctx.dep_index,
    )


@feature(
    group="congestion",
    outputs=("pier_dep_back_15m", "pier_dep_back_60m"),
    depends_on=("core_keys", "stand_runway_keys"),
    doc="Departures pushing back from the same pier/terminal around this takeoff.",
)
def pier_load(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """Local apron congestion, which airport-wide counts wash out."""
    keys = ctx.dep("stand_runway_keys")
    q_t = epoch_seconds(mvt.loc[ctx.dep_index, "MVT_TIME_UTC_mvt"])
    q_g = keys["stand_prefix"].astype("string")
    return pd.DataFrame(
        {
            "pier_dep_back_15m": window_count(q_t, q_g, q_t, q_g, back=900, fwd=0),
            "pier_dep_back_60m": window_count(q_t, q_g, q_t, q_g, back=3600, fwd=0),
        },
        index=ctx.dep_index,
    )
