"""METAR/SPECI reports -> one row of observations per report.

The raw text is the source of truth, not IEM's decoded columns: those are in
US units, cap visibility at 6.21 miles, and drop RVR, recent weather, the TREND
and runway state. They are kept only as QA cross-checks (``qa_*``).

The table holds *observations*. Judgements such as "low-visibility
procedures" or "de-icing conditions" are thresholds, and live in the feature
code (`prc.features._wxlib`), whose source is part of the cache key.
"""

from __future__ import annotations

import re

import pandas as pd

from prc import paths
from prc.weather.tokens import NAN, fmin, parse_groups, wx_flags

TREND_START = frozenset({"NOSIG", "BECMG", "TEMPO"})
_STATION = re.compile(r"^[A-Z]{4}$")
_TIME = re.compile(r"^\d{6}Z$")
_TREND_TIME = re.compile(r"^(?:FM|TL|AT)\d{4}$")

INT_COLS = (
    "is_auto", "is_speci", "cb_tcu", "precip_int",
    "wx_rain", "wx_drizzle", "wx_snow", "wx_freezing", "wx_ice", "wx_hail",
    "wx_ts", "wx_fog", "wx_mist", "wx_shower",
    "re_precip", "re_snow", "re_freezing", "re_ts", "rwy_contam", "snoclo",
    "trend_snowfrz", "trend_fog", "trend_precip",
)
FLOAT_COLS = (
    "wind_dir_deg", "wind_kt", "gust_kt", "vis_m", "vis_min_m", "rvr_min_m", "ceiling_ft",
    "temp_c", "dewpt_c", "qnh_hpa", "trend_vis_m", "trend_ceiling_ft",
)


def parse_metar(raw: str) -> dict:
    """One report -> flat dict of observations; NaN where an element was not reported."""
    tokens = raw.replace("=", " ").split()
    if "RMK" in tokens:  # free-form remarks: LTFM puts per-runway winds there
        tokens = tokens[: tokens.index("RMK")]

    rec = {"is_auto": 0, "is_cor": 0, "is_nil": 0, "report_ddhhmm": ""}
    i = 0
    while i < len(tokens) and tokens[i] in ("METAR", "SPECI", "COR"):
        rec["is_cor"] |= tokens[i] == "COR"
        i += 1
    if i < len(tokens) and _STATION.match(tokens[i]):
        i += 1
    if i < len(tokens) and _TIME.match(tokens[i]):
        rec["report_ddhhmm"] = tokens[i][:6]
        i += 1

    body: list[str] = []
    trend: list[str] = []
    for tok in tokens[i:]:
        if trend or tok in TREND_START:
            trend.append(tok)
        elif tok == "AUTO":
            rec["is_auto"] = 1
        elif tok == "COR":
            rec["is_cor"] = 1
        elif tok == "NIL":
            rec["is_nil"] = 1
        else:
            body.append(tok)

    el, unparsed = parse_groups(body)
    rec.update(_observations(el))
    rec.update(_trend(trend))
    rec["unparsed"] = " ".join(unparsed)
    return rec


def _observations(el: dict) -> dict:
    vis = el.get("vis_m", NAN)
    now = wx_flags(el.get("wx", []))
    recent = wx_flags(el.get("recent", []))
    return {
        "wind_dir_deg": el.get("wind_dir_deg", NAN),
        "wind_kt": el.get("wind_kt", NAN),
        "gust_kt": el.get("gust_kt", NAN),
        "vis_m": vis,
        "vis_min_m": fmin(vis, el.get("vis_min_m", NAN)),
        "rvr_min_m": el.get("rvr_min_m", NAN),
        "ceiling_ft": el.get("ceiling_ft", NAN),
        "cb_tcu": int(el.get("cb_tcu", False)),
        "temp_c": el.get("temp_c", NAN),
        "dewpt_c": el.get("dewpt_c", NAN),
        "qnh_hpa": el.get("qnh_hpa", NAN),
        "precip_int": now["precip_int"],
        **{f"wx_{k}": v for k, v in now.items() if k != "precip_int"},
        "re_precip": int(recent["precip_int"] > 0),
        "re_snow": recent["snow"],
        "re_freezing": recent["freezing"],
        "re_ts": recent["ts"],
        "rwy_contam": int(el.get("rwy_contam", False)),
        "snoclo": int(el.get("snoclo", False)),
    }


def _trend(tokens: list[str]) -> dict:
    """The report's 2-hour TREND: its kind, and the worst it announces."""
    out = {
        "trend_kind": "", "trend_vis_m": NAN, "trend_ceiling_ft": NAN,
        "trend_snowfrz": 0, "trend_fog": 0, "trend_precip": 0,
    }
    if not tokens:
        return out
    kinds = [k for k in ("BECMG", "TEMPO") if k in tokens]
    out["trend_kind"] = "+".join(kinds) if kinds else "NOSIG"
    groups: list[list[str]] = []
    for tok in tokens:
        if tok in TREND_START:
            groups.append([])
        elif groups and not _TREND_TIME.match(tok):
            groups[-1].append(tok)
    for g in groups:
        el, _ = parse_groups(g)
        out["trend_vis_m"] = fmin(out["trend_vis_m"], el.get("vis_m", NAN))
        out["trend_ceiling_ft"] = fmin(out["trend_ceiling_ft"], el.get("ceiling_ft", NAN))
        f = wx_flags(el.get("wx", []))
        out["trend_snowfrz"] |= f["snow"] | f["freezing"] | f["ice"]
        out["trend_fog"] |= f["fog"]
        out["trend_precip"] |= int(f["precip_int"] > 0)
    return out


def load_raw() -> pd.DataFrame:
    files = sorted((paths.WEATHER_RAW / "metar").glob("*.csv"))
    if not files:
        raise FileNotFoundError("no raw METAR chunks; run `python -m prc fetch-weather --only metar`")
    # dtype=str + keep_default_na=False: pandas would otherwise read "NA" as missing.
    frames = [pd.read_csv(f, dtype=str, keep_default_na=False) for f in files]
    return pd.concat([f for f in frames if len(f)], ignore_index=True)


def normalise_metar() -> paths.Path:
    raw = load_raw()
    raw = raw.loc[raw["metar"].str.strip().ne("")].reset_index(drop=True)
    parsed = pd.DataFrame.from_records([parse_metar(s) for s in raw["metar"]])
    obs_time = pd.to_datetime(raw["valid"], utc=True, format="%Y-%m-%d %H:%M").astype("datetime64[us, UTC]")
    df = pd.concat([raw[["station"]], obs_time.rename("obs_time"), parsed], axis=1)
    df["raw"] = raw["metar"]
    df["time_mismatch"] = (
        df["report_ddhhmm"].ne("") & df["report_ddhhmm"].ne(df["obs_time"].dt.strftime("%d%H%M"))
    ).astype("int8")

    df = df.loc[df["is_nil"].eq(0)]
    # A correction is listed under the same observation time: keep the correction.
    df = (
        df.assign(_len=df["raw"].str.len())
        .sort_values(["station", "obs_time", "is_cor", "_len"], ascending=[True, True, False, False], kind="stable")
        .drop_duplicates(["station", "obs_time"], keep="first")
        .drop(columns="_len")
    )
    # Routine reports come at two fixed minutes per station (:20/:50, :25/:55,
    # :00/:30); anything else is an off-cycle special.
    minute = df["obs_time"].dt.minute.rename("minute")
    counts = df.groupby([df["station"], minute]).size()
    regular = counts.sort_values(ascending=False).groupby(level=0).head(2).index
    df["is_speci"] = (~pd.MultiIndex.from_arrays([df["station"], minute]).isin(regular)).astype("int8")

    qa = raw.loc[df.index]
    out = pd.DataFrame({"station": df["station"].astype("category"), "obs_time": df["obs_time"]})
    for c in INT_COLS:
        out[c] = df[c].astype("int8")
    for c in FLOAT_COLS:
        out[c] = df[c].astype("float32")
    out["trend_kind"] = df["trend_kind"].astype("string")
    out["qa_temp_c"] = ((pd.to_numeric(qa["tmpf"], errors="coerce") - 32) * 5 / 9).astype("float32")
    out["qa_wind_kt"] = pd.to_numeric(qa["sknt"], errors="coerce").astype("float32")
    out["qa_vis_m"] = (pd.to_numeric(qa["vsby"], errors="coerce") * 1609.344).astype("float32")
    out["time_mismatch"] = df["time_mismatch"]
    out["raw"] = df["raw"].astype("string")
    out["unparsed"] = df["unparsed"].astype("string")
    out["source"] = pd.Categorical(["iem"] * len(df))
    out = out.sort_values(["station", "obs_time"], kind="stable").reset_index(drop=True)

    paths.WEATHER.mkdir(parents=True, exist_ok=True)
    out.to_parquet(paths.METAR_FILE, index=False)
    unparsed_share = out["unparsed"].ne("").mean()
    print(f"metar: {len(out):,} reports, {out['station'].nunique()} stations, "
          f"{unparsed_share:.2%} with unparsed groups -> {paths.METAR_FILE}")
    return paths.METAR_FILE
