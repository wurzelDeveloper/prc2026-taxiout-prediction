"""`fetch-weather --check`: is the weather data complete and sane enough to use?

Answered from what is on disk, with no network:

* per station -- reports, off-cycle reports, the longest silence, how many
  parsed cleanly, and whether the parse agrees with IEM's own decoding
  (plus every station-month that looks wrong);
* per split and airport -- the share of departures with a METAR in force at
  takeoff (the phase gate is >= 99 %, for train *and* rank);
* per airport and month -- hours of snow/freezing, LVP, de-icing conditions
  and thunder, so it is visible whether a rank month holds weather that the
  training months never had.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc import data, paths
from prc.features._wxlib import MAX_AGE_S, asof_index, report_end, report_state, station_rows
from prc.features.core import epoch_seconds

REGIMES = ("snowfrz", "lvp", "deice", "ts")


def report(kinds: tuple[str, ...]) -> str:
    parts = []
    if "metar" in kinds:
        parts.append(_metar_report())
    if "taf" in kinds:
        parts.append(_taf_report())
    return "\n\n".join(parts)


def _agree(parsed: pd.Series, qa: pd.Series, tol: pd.Series | float) -> pd.Series:
    """1.0/0.0 where both sides have a value, NaN where they cannot be compared."""
    both = parsed.notna() & qa.notna()
    return ((parsed - qa).abs() <= tol).astype("float64").where(both)


def _metar_report() -> str:
    if not paths.METAR_FILE.exists():
        return "metar: no normalised table yet -- run `python -m prc fetch-weather --only metar`"
    met = pd.read_parquet(paths.METAR_FILE)
    met["station"] = met["station"].astype(str)
    met["month"] = met["obs_time"].dt.strftime("%Y-%m")
    met["t"] = epoch_seconds(met["obs_time"]).to_numpy()
    gap = met.groupby("station")["t"].diff() / 60
    # Months that were never downloaded (between the Jan and Jul test months)
    # are a hole by design, not a station outage.
    met["gap_min"] = gap.where(gap < 20 * 24 * 60)
    met["parse_ok"] = met["unparsed"].eq("").astype("float64")
    met["temp_ok"] = _agree(met["temp_c"], met["qa_temp_c"], 1.0)
    met["wind_ok"] = _agree(met["wind_kt"], met["qa_wind_kt"], 1.0)
    vis, qa_vis = met["vis_m"].clip(upper=9000), met["qa_vis_m"].clip(upper=9000)
    met["vis_ok"] = _agree(vis, qa_vis, np.maximum(200.0, 0.1 * vis))

    def summarise(g) -> pd.DataFrame:
        return pd.DataFrame({
            "reports": g.size(),
            "offcycle": g["is_speci"].sum(),
            "max_gap_min": g["gap_min"].max().round(0),
            "parse_ok_%": (100 * g["parse_ok"].mean()).round(2),
            "temp_ok_%": (100 * g["temp_ok"].mean()).round(2),
            "wind_ok_%": (100 * g["wind_ok"].mean()).round(2),
            "vis_ok_%": (100 * g["vis_ok"].mean()).round(2),
            "time_mismatch": g["time_mismatch"].sum(),
        })

    per_station = summarise(met.groupby("station"))
    per_month = summarise(met.groupby(["station", "month"]))
    suspicious = per_month.loc[
        (per_month["reports"] < 1300) | (per_month["max_gap_min"] > 90) | (per_month["parse_ok_%"] < 99.5)
    ]
    unparsed = met.loc[met["unparsed"].ne(""), "unparsed"].str.split().explode().value_counts().head(15)

    mvt = data.load_movements(columns=["apt", "is_dep", "split", "MVT_TIME_UTC_mvt"])
    dep = mvt.loc[mvt["is_dep"]]
    rank_months = sorted(set(dep.loc[dep["split"].eq("rank"), "MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")))
    t = epoch_seconds(dep["MVT_TIME_UTC_mvt"]).to_numpy()
    by_station = {s: g.reset_index(drop=True) for s, g in met.groupby("station")}
    in_force = np.zeros(len(dep), dtype=bool)
    for station, rows in station_rows(dep["apt"]).items():
        if station in by_station:
            in_force[rows] = asof_index(by_station[station]["t"].to_numpy(), t[rows], MAX_AGE_S) >= 0
    cov = pd.DataFrame({"apt": dep["apt"].astype(str), "split": dep["split"].astype(str), "ok": in_force})
    coverage = (100 * cov.pivot_table(index="apt", columns="split", values="ok", aggfunc="mean")).round(2)
    coverage.loc["ALL"] = (100 * cov.groupby("split")["ok"].mean()).round(2)

    regimes = []
    for station, rep in by_station.items():
        state = report_state(rep)
        hours = (report_end(state["t"], MAX_AGE_S) - state["t"]) / 3600
        frame = pd.DataFrame({"month": rep["month"], **{k: state[k] * hours for k in REGIMES}})
        regimes.append(frame.groupby("month").sum().assign(apt=station))
    regime = pd.concat(regimes).reset_index()
    regime["month"] = regime["month"].where(~regime["month"].isin(rank_months), regime["month"] + " *")

    out = [
        "=== METAR: per station ===",
        per_station.to_string(),
        f"\n=== METAR: suspicious station-months ({len(suspicious)}) ===",
        suspicious.to_string() if len(suspicious) else "none",
        "\n=== METAR: most frequent unparsed groups ===",
        unparsed.to_string() if len(unparsed) else "none",
        f"\n=== departures with a METAR in force at takeoff (%, max age {MAX_AGE_S // 60} min) ===",
        coverage.to_string(),
    ]
    for k in REGIMES:
        table = regime.pivot(index="month", columns="apt", values=k).round(1)
        out += [f"\n=== hours of {k} per month (* = rank month) ===", table.to_string()]
    return "\n".join(out)


def _taf_report() -> str:
    raw = sorted((paths.WEATHER_RAW / "taf").glob("*.csv"))
    if not paths.TAF_ISSUES_FILE.exists():
        return f"taf: {len(raw)} raw chunks on disk, no normalised tables yet"
    from prc.weather.taf import coverage_report

    return coverage_report()
