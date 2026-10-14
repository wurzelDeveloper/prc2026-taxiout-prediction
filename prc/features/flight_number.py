"""The flight's own chain: what this flight number did on other days.

There is no registration in the data, so an airframe cannot be followed; a
flight number can. Long taxi-out times recur by flight number (findings 7.2:
69 flight numbers have three or more rows above an hour that are not stale
schedules, at rates 30-70 times the base; 29 % of such rows have a flight
number that is long in two or more months), and so do stale block times. The
target-based part is fold-dependent: rates and quantiles are computed on the
fold's fit rows only and looked up for every departure, ranking rows included.
The count-based part reads the whole frame, 2026 included, because it counts
rows, never targets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc.features.registry import FeatureContext, feature
from prc.features.temporal import local_naive

PRIOR_WEIGHT = 5.0
"""Pseudo-count of the airport-wide rate mixed into every flight number's rate."""
LONG_LO_SEC = 3600.0
LONG_HI_SEC = 14_400.0
"""A "long" taxi: above an hour and at most four hours (beyond that the record is a defect or a stale block)."""
STALE_TOL_SEC = 600.0
MIN_SUPPORT = 5
MAX_GAP_H = 720.0


def _key(dep: pd.DataFrame) -> pd.Series:
    return dep["apt"].astype("string") + "|" + dep["FLIGHT_mvt"].astype("string").fillna("__NA__")


def _smoothed_rate(flag: np.ndarray, fit: np.ndarray, key: pd.Series, apt: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Per key: ``(hits + PRIOR * airport rate) / (n + PRIOR)`` over fit rows, looked up for every row; and ``n``."""
    fit_key = key[fit].to_numpy()
    hits = pd.Series(flag[fit]).groupby(fit_key).agg(["sum", "size"])
    prior = pd.Series(flag[fit]).groupby(apt[fit].to_numpy()).mean()
    n = key.map(hits["size"]).fillna(0).astype("float64")
    s = key.map(hits["sum"]).fillna(0).astype("float64")
    p = apt.map(prior).fillna(0).astype("float64")
    return ((s + PRIOR_WEIGHT * p) / (n + PRIOR_WEIGHT)).astype("float32"), n


@feature(
    group="flight_number",
    outputs=("fn_long_rate", "fn_stale_rate", "fn_y_median", "fn_y_p90", "fn_n_fit"),
    depends_on=("core_keys", "stale_schedule"),
    uses_target=True,
    doc="The flight number's history in the fold's fit rows: long-taxi and stale-block rates, median and p90 taxi-out.",
)
def flight_number_history(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    dep = mvt.loc[ctx.dep_index]
    y = ctx.target.reindex(ctx.dep_index).to_numpy(dtype="float64")
    fit = ctx.fit_mask.reindex(ctx.dep_index, fill_value=False).to_numpy(dtype=bool) & np.isfinite(y)
    delta = ctx.dep("stale_schedule")["sched_delta_full"].to_numpy(dtype="float64")
    with np.errstate(invalid="ignore"):
        stale = fit & (np.abs(y - delta) <= STALE_TOL_SEC) & (delta >= LONG_LO_SEC)
        long = fit & (y > LONG_LO_SEC) & (y <= LONG_HI_SEC) & ~stale

    key = _key(dep)
    apt = dep["apt"].astype("string")
    known = dep["FLIGHT_mvt"].notna().to_numpy()
    long_rate, n = _smoothed_rate(long, fit, key, apt)
    stale_rate, _ = _smoothed_rate(stale, fit, key, apt)

    yf = pd.Series(y[fit], index=key[fit].to_numpy())
    grouped = yf.groupby(level=0)
    support = n.where(pd.Series(known, index=ctx.dep_index) & n.ge(MIN_SUPPORT))
    median = key.map(grouped.median()).where(support.notna())
    p90 = key.map(grouped.quantile(0.9)).where(support.notna())

    return pd.DataFrame(
        {
            "fn_long_rate": long_rate,
            "fn_stale_rate": stale_rate,
            "fn_y_median": median.astype("float32"),
            "fn_y_p90": p90.astype("float32"),
            "fn_n_fit": n.astype("float32"),
        },
        index=ctx.dep_index,
    )


@feature(
    group="flight_number",
    outputs=(
        "fn_n_rows_apt",
        "fn_n_months_active",
        "fn_rows_per_active_month",
        "fn_prev_day_exists",
        "fn_hours_since_prev_dep",
        "fn_same_day_dupes",
        "fn_sched_mod_std",
    ),
    depends_on=("core_keys",),
    doc="How regular this flight number is at the airport: rows, months active, the day before, duplicates, schedule spread.",
)
def flight_number_context(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    dep_all = mvt.loc[mvt["is_dep"]]
    key_all = _key(dep_all)
    t_all = dep_all["MVT_TIME_UTC_mvt"]
    day_all = t_all.dt.floor("D")
    ym_all = t_all.dt.strftime("%Y-%m")

    n_rows = key_all.map(key_all.value_counts())
    n_months = key_all.map(pd.Series(ym_all.to_numpy()).groupby(key_all.to_numpy()).nunique())

    day_key = key_all + "|" + day_all.dt.strftime("%Y-%m-%d")
    seen_days = set(day_key.to_numpy())
    prev_day_key = key_all + "|" + (day_all - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    prev_day_exists = prev_day_key.isin(seen_days)
    same_day = day_key.map(day_key.value_counts()) - 1

    order = t_all.sort_values(kind="stable").index
    t_sorted = t_all.loc[order]
    prev_t = t_sorted.groupby(key_all.loc[order].to_numpy(), sort=False).shift(1)
    hours_since = ((t_sorted - prev_t).dt.total_seconds() / 3600).clip(upper=MAX_GAP_H).reindex(dep_all.index)

    sched_local = local_naive(dep_all["SCHED_TIME_UTC_mvt"], dep_all["apt"])
    sched_minute = (sched_local.dt.hour * 60 + sched_local.dt.minute).astype("float64")
    sched_std = key_all.map(sched_minute.groupby(key_all.to_numpy()).std())

    known = dep_all["FLIGHT_mvt"].notna()
    out = pd.DataFrame(
        {
            "fn_n_rows_apt": n_rows.astype("float32"),
            "fn_n_months_active": n_months.astype("float32"),
            "fn_rows_per_active_month": (n_rows / n_months).astype("float32"),
            "fn_prev_day_exists": prev_day_exists.astype("int8"),
            "fn_hours_since_prev_dep": hours_since.astype("float32"),
            "fn_same_day_dupes": same_day.astype("float32"),
            "fn_sched_mod_std": sched_std.astype("float32"),
        },
        index=dep_all.index,
    )
    # An unknown flight number is not "a flight number seen 30,000 times".
    out.loc[~known.to_numpy(), out.columns.difference(["fn_prev_day_exists"])] = np.nan
    out.loc[~known.to_numpy(), "fn_prev_day_exists"] = 0
    out["fn_prev_day_exists"] = out["fn_prev_day_exists"].astype("int8")
    return out.reindex(ctx.dep_index)
