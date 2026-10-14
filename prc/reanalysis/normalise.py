"""Raw Open-Meteo JSON -> ``era5_hourly.parquet``: one row per airport and UTC hour.

Units are converted once, here: snow depth metres -> cm; snowfall stays cm of
snow (not water equivalent); precipitation mm. Hourly sums (``precip_mm``,
``rain_mm``, ``snowfall_cm``) cover the hour *ending* at the row's time; state
variables are the value at that time. So a feature may use every row at or
before takeoff without looking ahead. Readers validate like the PRU
normaliser: a missing variable, a unit change or a gap in the hourly axis
raises instead of producing silent NaN.
"""

from __future__ import annotations

import json

import pandas as pd

from prc import paths
from prc.reanalysis import VARIABLES

RENAME = {
    "temperature_2m": "temp_c",
    "dew_point_2m": "dewpt_c",
    "precipitation": "precip_mm",
    "rain": "rain_mm",
    "snowfall": "snowfall_cm",
    "snow_depth": "snow_depth_cm",
    "weather_code": "wmo_code",
    "cloud_cover_low": "cloud_low_pct",
    "wind_speed_10m": "wind_kmh",
    "wind_gusts_10m": "gust_kmh",
    "soil_temperature_0_to_7cm": "soil_temp_c",
}
COLUMNS = ("apt", "time", *RENAME.values())


def parse_answer(apt: str, body: dict) -> pd.DataFrame:
    hourly = body.get("hourly") or {}
    missing = [v for v in VARIABLES if v not in hourly]
    if missing:
        raise ValueError(f"{apt}: ERA5 answer lacks {missing}")
    units = body.get("hourly_units") or {}
    if units.get("snow_depth", "m") != "m" or units.get("snowfall", "cm") != "cm":
        raise ValueError(f"{apt}: unexpected units {units}")
    t = pd.to_datetime(pd.Series(hourly["time"]), format="%Y-%m-%dT%H:%M", utc=True)
    step = t.diff().dropna()
    if len(step) and not (step == pd.Timedelta(hours=1)).all():
        raise ValueError(f"{apt}: the hourly axis has gaps or repeats")
    out = pd.DataFrame({"apt": apt, "time": t.astype("datetime64[us, UTC]")})
    for src, dst in RENAME.items():
        out[dst] = pd.to_numeric(pd.Series(hourly[src]), errors="coerce").astype("float32")
    out["snow_depth_cm"] = (out["snow_depth_cm"] * 100).astype("float32")
    return out[list(COLUMNS)]


def normalise_era5() -> paths.Path:
    files = sorted(paths.REANALYSIS_RAW.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no raw ERA5 answers in {paths.REANALYSIS_RAW}: run `python -m prc fetch-era5`")
    frames = [parse_answer(f.stem, json.loads(f.read_text(encoding="utf-8"))) for f in files]
    table = pd.concat(frames, ignore_index=True).sort_values(["apt", "time"], kind="stable").reset_index(drop=True)
    paths.ERA5_FILE.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(paths.ERA5_FILE, index=False)
    print(f"era5: {len(table):,} airport-hours for {table['apt'].nunique()} airports -> {paths.ERA5_FILE}")
    return paths.ERA5_FILE


def coverage_report() -> str:
    if not paths.ERA5_FILE.exists():
        return f"{paths.ERA5_FILE} missing: run `python -m prc fetch-era5`"
    t = pd.read_parquet(paths.ERA5_FILE)
    t["month"] = t["time"].dt.strftime("%Y-%m")
    lines = [f"era5: {len(t):,} airport-hours, {t['time'].min()} .. {t['time'].max()}"]
    nan = t.drop(columns=["apt", "time", "month"]).isna().mean()
    nan = nan[nan > 0]
    lines.append("NaN share per variable: " + (", ".join(f"{k} {v:.1%}" for k, v in nan.items()) or "none"))
    snow = t.groupby(["month", "apt"])["snowfall_cm"].sum().unstack("apt")
    snow = snow[snow.sum(axis=1) > 0]
    lines.append("snowfall cm per airport-month (months with any):\n" + snow.round(1).to_string())
    depth = t.groupby(["month", "apt"])["snow_depth_cm"].max().unstack("apt")
    depth = depth[depth.max(axis=1) > 0]
    lines.append("max snow depth cm per airport-month:\n" + depth.round(1).to_string())
    return "\n".join(lines)
