"""Arrival taxi-in anomalies: the one live measurement of ground conditions the ranking months keep.

Departures' block times are blanked in 2026, but every arrival's in-block and
taxi-in time is there. An arrival that taxied five minutes longer than its
stand/runway pairing normally takes says something about the airport right
now -- snow clearance, a closed taxiway, a saturated apron -- that nothing in
the departure's own row does.

Shared by the `arrival_conditions` feature block (`prc.features.arrivals_live`)
and by `prc.disruption`, which turns airport-days with a large mean anomaly
into a report mask. Not auto-imported (underscore module), but hashed into
every feature module that imports it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ARR_COLUMNS = ("apt", "is_dep", "RUNWAY_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt")
"""What these functions read from the movement frame."""
TAXIIN_CAP_S = 14_400.0
"""A taxi-in above four hours is a record defect, not a slow taxi; it is capped before anything is measured."""
ANOMALY_CAP_S = 7_200.0
"""The anomaly is clipped to +-2 h so one broken record cannot make a day look disrupted."""
MIN_SUPPORT = 20
"""Arrivals a (stand, runway) pairing needs before its own median is trusted as the baseline."""
MIN_ARRIVALS_PER_DAY = 20
"""Arrivals an airport-day needs before its mean anomaly is read at all."""


def arrivals(mvt: pd.DataFrame) -> pd.DataFrame:
    """Arrival rows with a usable taxi-in time (positive; negative ones are record defects)."""
    arr = mvt.loc[~mvt["is_dep"].to_numpy(dtype=bool), ["apt", "RUNWAY_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]]
    taxi = arr["TAXITIME_SEC_mvt"]
    return arr.loc[(taxi.notna() & (taxi > 0)).to_numpy(dtype=bool)]


def taxiin_baseline(arr: pd.DataFrame, min_support: int = MIN_SUPPORT) -> pd.Series:
    """What a normal taxi-in is for each arrival: the median of its (airport, runway, stand) pairing.

    Pairings with fewer than ``min_support`` arrivals fall back to the (airport,
    runway) median, then to the airport's. Computed over every arrival in the
    frame, 2026 included: these are arrivals' own times, never a departure's
    target, so the 2026 months get the same treatment as 2025.
    """
    taxi = arr["TAXITIME_SEC_mvt"].clip(upper=TAXIIN_CAP_S).astype("float64")
    apt = arr["apt"].astype("string")
    rwy = arr["RUNWAY_mvt"].astype("string").fillna("__NA__")
    stand = arr["STAND_mvt"].astype("string").fillna("__NA__")
    out = pd.Series(np.nan, index=arr.index, dtype="float64")
    for key in (apt + "|" + rwy + "|" + stand, apt + "|" + rwy, apt):
        grouped = taxi.groupby(key.to_numpy())
        median = grouped.transform("median")
        n = grouped.transform("size")
        fill = (out.isna() & (n >= min_support)).to_numpy(dtype=bool)
        out[fill] = median[fill]
    return out


def taxiin_anomaly(mvt: pd.DataFrame, min_support: int = MIN_SUPPORT) -> pd.Series:
    """Each arrival's taxi-in minus its baseline, clipped to +-`ANOMALY_CAP_S`; indexed by arrival id."""
    arr = arrivals(mvt)
    anomaly = arr["TAXITIME_SEC_mvt"].clip(upper=TAXIIN_CAP_S).astype("float64") - taxiin_baseline(arr, min_support)
    return anomaly.clip(-ANOMALY_CAP_S, ANOMALY_CAP_S).rename("taxiin_anomaly")


def airport_day_anomaly(mvt: pd.DataFrame, min_support: int = MIN_SUPPORT) -> pd.DataFrame:
    """Per (airport, UTC day of landing): arrivals counted and their mean taxi-in anomaly."""
    arr = arrivals(mvt)
    anomaly = taxiin_anomaly(mvt, min_support)
    frame = pd.DataFrame(
        {
            "apt": arr["apt"].astype(str).to_numpy(),
            "day": arr["MVT_TIME_UTC_mvt"].dt.floor("D").to_numpy(),
            "anomaly": anomaly.reindex(arr.index).to_numpy(dtype="float64"),
        }
    )
    out = frame.groupby(["apt", "day"], sort=True)["anomaly"].agg(n_arr="size", mean_anomaly="mean").reset_index()
    return out
