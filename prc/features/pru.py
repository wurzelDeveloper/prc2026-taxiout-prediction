"""Eurocontrol PRU open data as day-level context.

The PRU publishes, per airport and day, the ATC pre-departure delay minutes
(two computations, column suffixes ``_2`` and ``_3`` in its CSV), all
pre-departure delay minutes, and the airport arrival ATFM delay minutes. They
are day-level aggregates of what happened at the airport, available for the
2026 ranking months as well (`python -m prc fetch-pru`).

Two caveats belong next to the numbers. They are published after the fact,
like the METARs and the forward takeoff windows the harness already uses, so
an operational model would not have them; and the ATC pre-departure delay is
itself computed from the day's actual off-block times, so it is a published
aggregate of the target's sibling -- legal open data, but not independent.
Whether the block enters a submission config is a decision for the user once
its measured value is known.

NaN means the airport does not report that series (EGLL, EHAM and LTFM have no
``_2`` ATC pre-departure delay) or the day is missing; the ``reported`` flag
says so. Only ``pru_daily.parquet`` is read; the monthly taxi-out table is for
`prc levelcheck`, never a feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc import paths
from prc.features.registry import FeatureContext, feature

SERIES = {
    "pru_atc_predep2_per_dep": "atc_predep2_dly_min_per_dep",
    "pru_atc_predep3_per_dep": "atc_predep3_dly_min_per_dep",
    "pru_all_predep_per_dep": "all_predep_dly_min_per_dep",
    "pru_apt_arr_atfm_per_arr": "apt_arr_atfm_dly_min_per_arr",
    "pru_apt_arr_atfm_wx_per_arr": "apt_arr_atfm_weather_min_per_arr",
}
"""Feature column -> column of ``pru_daily.parquet`` (delay minutes per movement that day)."""

PRU_DAILY = (*SERIES, *(f"{c}_prev_day" for c in SERIES), "pru_daily_reported")


def _day_key(apt: pd.Series, day: pd.Series) -> pd.Series:
    return apt.astype(str) + "|" + pd.to_datetime(day).dt.strftime("%Y-%m-%d")


@feature(
    group="pru",
    outputs=PRU_DAILY,
    depends_on=("core_keys",),
    external=(paths.PRU_DAILY_FILE,),
    doc="PRU open data for the departure's airport-day: pre-departure and airport ATFM delay minutes per movement, and the day before.",
)
def pru_daily(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    dep = mvt.loc[ctx.dep_index, ["apt", "MVT_TIME_UTC_mvt"]]
    table = pd.read_parquet(paths.PRU_DAILY_FILE)
    lookup = table.set_index(_day_key(table["apt"], table["day"]).to_numpy())

    day = dep["MVT_TIME_UTC_mvt"].dt.floor("D")
    today = _day_key(dep["apt"], day).to_numpy()
    yesterday = _day_key(dep["apt"], day - pd.Timedelta(days=1)).to_numpy()

    out = {}
    for col, source in SERIES.items():
        series = lookup[source]
        out[col] = pd.Series(today, index=ctx.dep_index).map(series).astype("float32")
        out[f"{col}_prev_day"] = pd.Series(yesterday, index=ctx.dep_index).map(series).astype("float32")
    reported = np.zeros(len(dep), dtype=bool)
    for col in SERIES:
        reported |= out[col].notna().to_numpy()
    out["pru_daily_reported"] = pd.Series(reported.astype("int8"), index=ctx.dep_index)
    return pd.DataFrame(out, index=ctx.dep_index)[list(PRU_DAILY)]
