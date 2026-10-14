"""Calendar and time-of-day features.

Local time matters more than UTC here -- taxi congestion follows the airport's
own bank structure -- so `hour_local` is computed against each airport's real
timezone (DST included) rather than a fixed offset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc.features.core import epoch_seconds
from prc.features.registry import FeatureContext, feature

AIRPORT_TZ = {
    "EDDF": "Europe/Berlin",
    "EDDM": "Europe/Berlin",
    "EGLL": "Europe/London",
    "EHAM": "Europe/Amsterdam",
    "LEBL": "Europe/Madrid",
    "LEMD": "Europe/Madrid",
    "LFPG": "Europe/Paris",
    "LIRF": "Europe/Rome",
    "LSZH": "Europe/Zurich",
    "LTFM": "Europe/Istanbul",
    "LTAI": "Europe/Istanbul",  # documented but absent from the delivered data
}


def local_naive(ts: pd.Series, apt: pd.Series) -> pd.Series:
    """UTC timestamps converted to each airport's wall-clock time (tz-naive)."""
    out = pd.Series(pd.NaT, index=ts.index, dtype="datetime64[ns]")
    for airport, tz in AIRPORT_TZ.items():
        mask = apt.eq(airport).to_numpy()
        if not mask.any():
            continue
        out[mask] = ts[mask].dt.tz_convert(tz).dt.tz_localize(None).astype("datetime64[ns]")
    return out


@feature(
    group="temporal",
    outputs=(
        "hour_utc",
        "hour_local",
        "minute_of_day_local",
        "hour_sin",
        "hour_cos",
        "dow",
        "is_weekend",
        "month",
        "day_of_year",
        "week_of_year",
        "sched_hour_local",
        "sched_dow",
        "days_since_epoch",
    ),
    depends_on=("core_keys",),
    doc="Calendar / time-of-day features in airport local time.",
)
def calendar(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    dep = mvt.loc[ctx.dep_index]
    t_utc = dep["MVT_TIME_UTC_mvt"]
    t_loc = local_naive(t_utc, dep["apt"])
    s_loc = local_naive(dep["SCHED_TIME_UTC_mvt"], dep["apt"])

    minute_of_day = t_loc.dt.hour * 60 + t_loc.dt.minute
    angle = 2 * np.pi * minute_of_day / 1440.0

    return pd.DataFrame(
        {
            "hour_utc": t_utc.dt.hour.astype("int16"),
            "hour_local": t_loc.dt.hour.astype("Int16"),
            "minute_of_day_local": minute_of_day.astype("Int32"),
            "hour_sin": np.sin(angle).astype("float32"),
            "hour_cos": np.cos(angle).astype("float32"),
            "dow": t_loc.dt.dayofweek.astype("Int16"),
            "is_weekend": t_loc.dt.dayofweek.ge(5).astype("Int8"),
            "month": t_loc.dt.month.astype("Int16"),
            "day_of_year": t_loc.dt.dayofyear.astype("Int16"),
            "week_of_year": t_loc.dt.isocalendar().week.astype("Int16"),
            "sched_hour_local": s_loc.dt.hour.astype("Int16"),
            "sched_dow": s_loc.dt.dayofweek.astype("Int16"),
            # Absolute time: lets a model pick up a year-long trend, but it is
            # also the one feature that cannot extrapolate to 2026. Ablate it.
            "days_since_epoch": (epoch_seconds(t_utc) // 86_400).astype("int32"),
        },
        index=ctx.dep_index,
    )
