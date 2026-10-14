"""ERA5 reanalysis at takeoff: snow on the ground, recent snowfall, frost.

The METARs say whether it snows at a given half hour; they do not say how much
snow lies on the ground, how much fell overnight, or whether the night was
cold and clear enough to frost a parked aircraft. Those decide de-icing and
taxiway conditions, and January 2026 held far more of them than any 2025
month (ERA5 snow depth at EHAM 13 cm against at most 3 cm in 2025; Madrid 12 cm
of snowfall against almost none). `python -m prc fetch-era5` provides the
hourly table (`prc/reanalysis/`).

Causality: an hourly sum at hour ``h`` covers ``(h - 1 h, h]`` and a state
variable is the value at ``h``, so every row used here sits at or before
``floor(takeoff, 1 h)``. NaN means only that the takeoff falls outside the
table's period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc import paths
from prc.features.core import epoch_seconds
from prc.features.registry import FeatureContext, feature

HOUR = 3600.0
FROST_SPREAD_C = 2.5  # temperature minus dew point at or below which a sub-zero night deposits frost
FREEZING_TEMP_C = 1.0  # precipitation at or below this 2 m temperature is treated as freezing / snow

ERA5_SURFACE = (
    "era5_snow_depth_cm",
    "era5_snowfall_3h_cm",
    "era5_snowfall_24h_cm",
    "era5_precip_3h_mm",
    "era5_freezing_precip_hours_6h",
    "era5_temp_c",
    "era5_soil_temp_c",
    "era5_temp_min_12h_c",
    "era5_frost_hours_12h",
)


def _derived(table: pd.DataFrame) -> pd.DataFrame:
    """Per airport, the trailing-window columns, each ending at the row's hour."""
    out = []
    for _, g in table.groupby("apt", sort=False):
        g = g.sort_values("time")
        wet = (g["precip_mm"] > 0.1) & (g["temp_c"] <= FREEZING_TEMP_C)
        frost = (g["temp_c"] <= 0.0) & ((g["temp_c"] - g["dewpt_c"]) <= FROST_SPREAD_C)
        out.append(
            pd.DataFrame(
                {
                    "apt": g["apt"],
                    "t": epoch_seconds(g["time"]),
                    "era5_snow_depth_cm": g["snow_depth_cm"],
                    "era5_snowfall_3h_cm": g["snowfall_cm"].rolling(3, min_periods=1).sum(),
                    "era5_snowfall_24h_cm": g["snowfall_cm"].rolling(24, min_periods=1).sum(),
                    "era5_precip_3h_mm": g["precip_mm"].rolling(3, min_periods=1).sum(),
                    "era5_freezing_precip_hours_6h": wet.astype("float32").rolling(6, min_periods=1).sum(),
                    "era5_temp_c": g["temp_c"],
                    "era5_soil_temp_c": g["soil_temp_c"],
                    "era5_temp_min_12h_c": g["temp_c"].rolling(12, min_periods=1).min(),
                    "era5_frost_hours_12h": frost.astype("float32").rolling(12, min_periods=1).sum(),
                }
            )
        )
    return pd.concat(out, ignore_index=True)


@feature(
    group="era5",
    outputs=ERA5_SURFACE,
    depends_on=("core_keys",),
    external=(paths.ERA5_FILE,),
    doc="ERA5 at the last full hour before takeoff: snow depth, snowfall over 3 h and 24 h, freezing precipitation, ground temperature, overnight minimum and frost hours.",
)
def era5_surface(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    dep = mvt.loc[ctx.dep_index, ["apt", "MVT_TIME_UTC_mvt"]]
    derived = _derived(pd.read_parquet(paths.ERA5_FILE))
    t_dep = np.asarray(epoch_seconds(dep["MVT_TIME_UTC_mvt"]), dtype="float64")
    hour = np.floor(t_dep / HOUR) * HOUR
    result = pd.DataFrame(np.nan, index=ctx.dep_index, columns=list(ERA5_SURFACE), dtype="float32")
    apt = dep["apt"].astype(str).to_numpy()
    for code, g in derived.groupby("apt", sort=False):
        rows = np.flatnonzero(apt == code)
        if not len(rows):
            continue
        t = g["t"].to_numpy(dtype="float64")
        pos = np.searchsorted(t, hour[rows], side="right") - 1
        # Only an exact hourly match counts: before the table starts, or past its end, stays NaN.
        ok = (pos >= 0) & (np.take(t, np.clip(pos, 0, len(t) - 1)) == hour[rows])
        values = g[list(ERA5_SURFACE)].to_numpy(dtype="float32")
        block = np.full((len(rows), len(ERA5_SURFACE)), np.nan, dtype="float32")
        block[ok] = values[pos[ok]]
        result.iloc[rows] = block
    return result
