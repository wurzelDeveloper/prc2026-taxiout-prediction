"""Raw PRU CSVs -> two parquet tables (`prc.pru`).

Every reader validates the way `prc.geometry.manual` does: everything is read
as text (Latin-1: the files carry ``Zürich``), the required columns are
checked by name, numbers are coerced, and every problem found is collected and
raised at once, so a changed upstream layout fails loudly rather than
producing a table of NaN.

Daily table (``pru_daily.parquet``), one row per (airport, day) over a full
calendar grid, NaN where a series is not reported:

* ``atc_predep2_*`` / ``atc_predep3_*`` -- the PRU's two computations of ATC
  pre-departure delay (its column suffixes ``_2`` and ``_3``): departures,
  delay minutes, minutes per departure;
* ``all_predep_*`` -- all-cause pre-departure delay, the same three;
* ``apt_arr_atfm_*`` -- airport arrival ATFM delay: arrivals, delay minutes,
  minutes per arrival, the weather-attributed and de-icing-attributed minutes
  per arrival (ATFM regulation reasons ``W`` and ``D``), and delayed arrivals.

Monthly table (``pru_monthly.parquet``), one row per (airport, year, month):
the additional taxi-out indicator's counts and minutes, the reference share
``ref_nb_fl / tf`` and ``mean_taxi_out_s = 60 * (ref + add) / ref_nb_fl``.
"""

from __future__ import annotations

import bz2
from pathlib import Path

import numpy as np
import pandas as pd

from prc import paths

ENCODING = "latin-1"
REQUIRED: dict[str, tuple[str, ...]] = {
    "taxi_out_additional_time": (
        "YEAR", "MONTH_NUM", "APT_ICAO", "TF", "VALID_FL", "TOTAL_REF_NB_FL", "TOTAL_REF_TIME_MIN", "TOTAL_ADD_TIME_MIN",
    ),
    "atc_pre_departure_delays": ("FLT_DATE", "APT_ICAO", "FLT_DEP_1", "FLT_DEP_IFR_2", "DLY_ATC_PRE_2", "FLT_DEP_3", "DLY_ATC_PRE_3"),
    "all_pre_departure_delays": ("FLT_DATE", "APT_ICAO", "FLT_DEP_1", "FLT_DEP_IFR_2", "DLY_ALL_PRE_2"),
    "apt_dly": ("FLT_DATE", "APT_ICAO", "FLT_ARR_1", "DLY_APT_ARR_1", "DLY_APT_ARR_W_1", "DLY_APT_ARR_D_1", "FLT_ARR_1_DLY"),
}
DAILY_KINDS = ("atc_pre_departure_delays", "all_pre_departure_delays", "apt_dly")
DAILY_COLUMNS = [
    "apt", "day",
    "atc_predep2_n_dep", "atc_predep2_dly_min", "atc_predep2_dly_min_per_dep",
    "atc_predep3_n_dep", "atc_predep3_dly_min", "atc_predep3_dly_min_per_dep",
    "all_predep_n_dep", "all_predep_dly_min", "all_predep_dly_min_per_dep",
    "apt_arr_atfm_n_arr", "apt_arr_atfm_dly_min", "apt_arr_atfm_dly_min_per_arr",
    "apt_arr_atfm_weather_min_per_arr", "apt_arr_atfm_deice_min_per_arr", "apt_arr_atfm_n_delayed",
]
MONTHLY_COLUMNS = [
    "apt", "year", "month", "tf", "valid_fl", "ref_nb_fl", "ref_share", "ref_time_min", "add_time_min",
    "mean_taxi_out_s", "add_time_s_per_fl",
]


def read_raw(kind: str, path: Path) -> pd.DataFrame:
    """One raw file as text columns, with the required columns present."""
    if kind not in REQUIRED:
        raise ValueError(f"unknown PRU dataset '{kind}' (one of {tuple(REQUIRED)})")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PRU file not found: {path}")
    opener = bz2.open if path.name.endswith(".bz2") else open
    with opener(path, "rt", encoding=ENCODING) as f:
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED[kind] if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing column(s) {missing}; need {list(REQUIRED[kind])}")
    df["APT_ICAO"] = df["APT_ICAO"].str.strip().str.upper()
    return df


def _numeric(df: pd.DataFrame, col: str, problems: list[str], name: str, *, nonneg: bool = True) -> pd.Series:
    raw = df[col].str.strip()
    out = pd.to_numeric(raw.replace("", np.nan), errors="coerce")
    bad = out.isna() & raw.ne("")
    if bad.any():
        problems.append(f"{name}: {int(bad.sum())} unparsable value(s) in {col} (e.g. {raw[bad].iloc[0]!r})")
    if nonneg and (out < 0).any():
        problems.append(f"{name}: {int((out < 0).sum())} negative value(s) in {col}")
    return out.astype("float64")


def _days(df: pd.DataFrame, problems: list[str], name: str) -> pd.Series:
    raw = df["FLT_DATE"].str.strip().str.slice(0, 10)
    out = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce", utc=True)
    if out.isna().any():
        problems.append(f"{name}: {int(out.isna().sum())} unparsable FLT_DATE value(s) (e.g. {df['FLT_DATE'][out.isna()].iloc[0]!r})")
    return out


def _per(minutes: pd.Series, count: pd.Series) -> pd.Series:
    return (minutes / count.where(count > 0)).astype("float64")


def daily_table(frames: dict[str, list[pd.DataFrame]], airports: list[str]) -> pd.DataFrame:
    """Merge the daily datasets onto one (airport, day) grid spanning every day any of them covers."""
    problems: list[str] = []
    parts: dict[str, pd.DataFrame] = {}
    for kind in DAILY_KINDS:
        pieces = [f for f in frames.get(kind, []) if len(f)]
        if not pieces:
            continue
        df = pd.concat(pieces, ignore_index=True)
        df = df.loc[df["APT_ICAO"].isin(airports)].reset_index(drop=True)
        name = kind
        day = _days(df, problems, name)
        cols = {"apt": df["APT_ICAO"].to_numpy(), "day": day}
        if kind == "atc_pre_departure_delays":
            n2 = _numeric(df, "FLT_DEP_IFR_2", problems, name)
            d2 = _numeric(df, "DLY_ATC_PRE_2", problems, name)
            n3 = _numeric(df, "FLT_DEP_3", problems, name)
            d3 = _numeric(df, "DLY_ATC_PRE_3", problems, name)
            cols |= {
                "atc_predep2_n_dep": n2, "atc_predep2_dly_min": d2, "atc_predep2_dly_min_per_dep": _per(d2, n2),
                "atc_predep3_n_dep": n3, "atc_predep3_dly_min": d3, "atc_predep3_dly_min_per_dep": _per(d3, n3),
            }
        elif kind == "all_pre_departure_delays":
            n = _numeric(df, "FLT_DEP_IFR_2", problems, name)
            d = _numeric(df, "DLY_ALL_PRE_2", problems, name)
            cols |= {"all_predep_n_dep": n, "all_predep_dly_min": d, "all_predep_dly_min_per_dep": _per(d, n)}
        else:
            n = _numeric(df, "FLT_ARR_1", problems, name)
            d = _numeric(df, "DLY_APT_ARR_1", problems, name)
            w = _numeric(df, "DLY_APT_ARR_W_1", problems, name)
            de = _numeric(df, "DLY_APT_ARR_D_1", problems, name)
            cols |= {
                "apt_arr_atfm_n_arr": n, "apt_arr_atfm_dly_min": d, "apt_arr_atfm_dly_min_per_arr": _per(d, n),
                "apt_arr_atfm_weather_min_per_arr": _per(w, n), "apt_arr_atfm_deice_min_per_arr": _per(de, n),
                "apt_arr_atfm_n_delayed": _numeric(df, "FLT_ARR_1_DLY", problems, name),
            }
        part = pd.DataFrame(cols).dropna(subset=["day"])
        dupes = part.duplicated(subset=["apt", "day"], keep=False)
        if dupes.any():
            problems.append(f"{name}: {int(dupes.sum())} duplicate (airport, day) row(s)")
        parts[kind] = part.drop_duplicates(subset=["apt", "day"]).set_index(["apt", "day"])
    if problems:
        raise ValueError("PRU daily data: " + "; ".join(problems))
    if not parts:
        raise ValueError("PRU daily data: no daily dataset on disk (run `python -m prc fetch-pru`)")

    lo = min(p.index.get_level_values("day").min() for p in parts.values())
    hi = max(p.index.get_level_values("day").max() for p in parts.values())
    grid = pd.MultiIndex.from_product([sorted(airports), pd.date_range(lo, hi, freq="D", tz="UTC")], names=["apt", "day"])
    out = pd.DataFrame(index=grid)
    for part in parts.values():
        out = out.join(part, how="left")
    out = out.reset_index()
    for col in DAILY_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[DAILY_COLUMNS]


def monthly_table(frames: list[pd.DataFrame], airports: list[str]) -> pd.DataFrame:
    """The additional taxi-out indicator per (airport, year, month), with the derived mean taxi-out."""
    pieces = [f for f in frames if len(f)]
    if not pieces:
        raise ValueError("PRU monthly data: no taxi_out_additional_time file on disk (run `python -m prc fetch-pru`)")
    problems: list[str] = []
    df = pd.concat(pieces, ignore_index=True)
    df = df.loc[df["APT_ICAO"].isin(airports)].reset_index(drop=True)
    name = "taxi_out_additional_time"
    year = _numeric(df, "YEAR", problems, name)
    month = _numeric(df, "MONTH_NUM", problems, name)
    if ((month < 1) | (month > 12)).any():
        problems.append(f"{name}: MONTH_NUM outside 1..12")
    tf = _numeric(df, "TF", problems, name)
    valid = _numeric(df, "VALID_FL", problems, name)
    ref_n = _numeric(df, "TOTAL_REF_NB_FL", problems, name)
    ref_min = _numeric(df, "TOTAL_REF_TIME_MIN", problems, name)
    add_min = _numeric(df, "TOTAL_ADD_TIME_MIN", problems, name)
    out = pd.DataFrame(
        {
            "apt": df["APT_ICAO"].to_numpy(),
            "year": year.astype("Int64"),
            "month": month.astype("Int64"),
            "tf": tf, "valid_fl": valid, "ref_nb_fl": ref_n,
            "ref_share": ref_n / tf.where(tf > 0),
            "ref_time_min": ref_min, "add_time_min": add_min,
            "mean_taxi_out_s": 60.0 * (ref_min + add_min) / ref_n.where(ref_n > 0),
            "add_time_s_per_fl": 60.0 * add_min / ref_n.where(ref_n > 0),
        }
    )
    dupes = out.duplicated(subset=["apt", "year", "month"], keep=False)
    if dupes.any():
        problems.append(f"{name}: {int(dupes.sum())} duplicate (airport, year, month) row(s)")
    if problems:
        raise ValueError("PRU monthly data: " + "; ".join(problems))
    return out.sort_values(["apt", "year", "month"]).reset_index(drop=True)[MONTHLY_COLUMNS]


def normalise_pru(airports: list[str] | None = None) -> dict[str, Path]:
    """Read every raw file on disk and write the two tables; returns ``{table: path}``."""
    if airports is None:
        from prc import data

        airports = data.airports(data.load_movements(columns=["apt"]))
    frames: dict[str, list[pd.DataFrame]] = {}
    for f in sorted(paths.PRU_RAW.glob("*.csv*")):
        kind = next((k for k in REQUIRED if f.name.startswith(k + "_")), None)
        if kind is None:
            continue
        frames.setdefault(kind, []).append(read_raw(kind, f))
    written = {}
    daily = daily_table(frames, airports)
    daily.to_parquet(paths.PRU_DAILY_FILE, index=False)
    written["pru_daily"] = paths.PRU_DAILY_FILE
    monthly = monthly_table(frames.get("taxi_out_additional_time", []), airports)
    monthly.to_parquet(paths.PRU_MONTHLY_FILE, index=False)
    written["pru_monthly"] = paths.PRU_MONTHLY_FILE
    print(f"pru: {len(daily):,} airport-days ({daily['day'].min().date()} .. {daily['day'].max().date()}), "
          f"{len(monthly):,} airport-months -> {paths.PRU}")
    return written
