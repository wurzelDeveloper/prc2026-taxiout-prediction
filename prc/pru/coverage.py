"""`fetch-pru --check`: which PRU series exist for which airport, and do they cover the ranking months?"""

from __future__ import annotations

import pandas as pd

from prc import paths

RANK_MONTHS = ("2026-01", "2026-07")
SERIES = {
    "atc_predep2": "atc_predep2_dly_min_per_dep",
    "atc_predep3": "atc_predep3_dly_min_per_dep",
    "all_predep": "all_predep_dly_min_per_dep",
    "apt_arr_atfm": "apt_arr_atfm_dly_min_per_arr",
}


def report() -> str:
    parts = []
    if not paths.PRU_DAILY_FILE.exists() or not paths.PRU_MONTHLY_FILE.exists():
        return "pru: no normalised tables yet -- run `python -m prc fetch-pru`"

    daily = pd.read_parquet(paths.PRU_DAILY_FILE)
    daily["ym"] = daily["day"].dt.strftime("%Y-%m")
    rows = []
    for apt, g in daily.groupby("apt", sort=True):
        row = {"apt": apt}
        for label, col in SERIES.items():
            has = g[col].notna()
            row[f"{label}_2025"] = int((has & g["ym"].str.startswith("2025")).sum())
            for m in RANK_MONTHS:
                row[f"{label}_{m}"] = int((has & g["ym"].eq(m)).sum())
        rows.append(row)
    table = pd.DataFrame(rows).set_index("apt")
    verdict = []
    for apt, row in table.iterrows():
        rank_days = sum(row[f"{label}_{m}"] for label in SERIES for m in RANK_MONTHS)
        possible = len(SERIES) * (31 + 31)
        state = "full" if rank_days >= 0.95 * possible else ("partial" if rank_days else "none")
        verdict.append(f"  {apt}: daily series in the ranking months: {state} ({rank_days} of {possible} airport-day-series)")
    parts.append(
        "=== PRU daily series: days with a value per airport (2025, then each ranking month; 365 / 31 possible) ===\n"
        + table.to_string()
        + "\n"
        + "\n".join(verdict)
    )

    monthly = pd.read_parquet(paths.PRU_MONTHLY_FILE)
    monthly["ym"] = monthly["year"].astype(int).astype(str) + "-" + monthly["month"].astype(int).map("{:02d}".format)
    monthly["ym"] = monthly["ym"].where(~monthly["ym"].isin(RANK_MONTHS), monthly["ym"] + "*")
    means = monthly.pivot(index="apt", columns="ym", values="mean_taxi_out_s").round(0)
    shares = monthly.pivot(index="apt", columns="ym", values="ref_share").round(2)
    parts.append(
        "=== PRU additional taxi-out time: implied mean taxi-out (s) per airport-month (* = ranking month; `prc levelcheck` uses this) ===\n"
        + means.to_string()
        + "\n\n=== share of the month's flights with a reference (the excluded ones are the long ones) ===\n"
        + shares.to_string()
    )
    return "\n\n".join(parts)
