"""TAF forecasts -> issues, and constant-state segments over each TAF's validity.

A TAF is a base forecast plus change groups:

* ``FMddhhmm``                      -- from then on, a complete new forecast;
* ``BECMG ddhh/ddhh``               -- a lasting change, complete by the period's end;
* ``TEMPO ddhh/ddhh``               -- temporary fluctuations within the period;
* ``PROB30|PROB40 [TEMPO] ddhh/ddhh`` -- the same, with a probability.

Two states follow at any time tau:

* *prevailing* P(tau) -- the base, replaced by every FM that has started, with
  every BECMG whose transition has *ended* applied element-wise;
* *worst case* W(tau) -- the element-wise worst of P and every TEMPO/PROB group
  active at tau, plus any BECMG still in transition: lowest visibility and
  ceiling, strongest wind, any weather.

Both are constant between consecutive group boundaries, so each TAF is stored
as a handful of segments and the feature blocks only ever look one up.
"""

from __future__ import annotations

import calendar
import math
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from prc import paths
from prc.weather.tokens import NAN, fmin, parse_groups, wx_flags

_ROW = re.compile(r"^(?P<st>[A-Z]{4}),(?P<y>\d{4}),(?P<mo>\d{2}),(?P<d>\d{2}),(?P<h>\d{2}),(?P<mi>\d{2}),(?P<text>.*)$")
_STATION = re.compile(r"^[A-Z]{4}$")
_ISSUE = re.compile(r"^(?P<d>\d{2})(?P<h>\d{2})(?P<m>\d{2})Z$")
_PERIOD = re.compile(r"^(?P<d1>\d{2})(?P<h1>\d{2})/(?P<d2>\d{2})(?P<h2>\d{2})$")
_FM = re.compile(r"^FM(?P<d>\d{2})(?P<h>\d{2})(?P<m>\d{2})$")
_PROB = re.compile(r"^PROB(?P<p>30|40)$")
_TEMPERATURE = re.compile(r"^T[XN]M?\d{2}/\d{4}Z$")

WX_KEYS = ("precip", "snow", "freezing", "ts", "fog")
PREVAILING = ("wind_kt", "gust_kt", "vis_m", "ceiling_ft", *WX_KEYS)
WORST = ("wind_kt", "vis_m", "ceiling_ft", *WX_KEYS)
TEMPO_PROB = 50  # a TEMPO group without an explicit probability


def resolve_ddhh(day: int, hour: int, minute: int, ref: datetime) -> datetime | None:
    """A TAF day-of-month and time, placed in the month that puts it nearest ``ref``.

    TAF times carry no month -- a TAF issued on 31 January is valid ``0106/0212``
    -- and hour ``24`` is the midnight that ends the day. ``None`` for a time
    that cannot be meant (more than 3 days from the reference).
    """
    if not (0 <= hour <= 24 and 0 <= minute < 60):
        return None
    best = None
    for shift in (-1, 0, 1):
        y, m0 = divmod(ref.year * 12 + ref.month - 1 + shift, 12)
        if not 1 <= day <= calendar.monthrange(y, m0 + 1)[1]:
            continue
        cand = datetime(y, m0 + 1, day, tzinfo=timezone.utc) + timedelta(hours=hour, minutes=minute)
        if best is None or abs(cand - ref) < abs(best - ref):
            best = cand
    return best if best is not None and abs(best - ref) <= timedelta(days=3) else None


def _period(m: re.Match, issued: datetime) -> tuple[datetime | None, datetime | None]:
    return (
        resolve_ddhh(int(m["d1"]), int(m["h1"]), 0, issued),
        resolve_ddhh(int(m["d2"]), int(m["h2"]), 0, issued),
    )


def parse_taf(text: str, issued: datetime) -> dict:
    """One TAF -> header flags, validity, and its groups.

    Each group is ``{kind, prob, start, end, elements}`` with ``kind`` one of
    BASE, FM, BECMG, TEMPO, PROB; ``elements`` is what `parse_groups` found.
    """
    tokens = [t for t in text.replace("=", " ").split() if not _TEMPERATURE.match(t)]
    out: dict = {
        "is_amd": 0, "is_cor": 0, "is_nil": 0, "is_cnl": 0,
        "header_issue": None, "valid_from": None, "valid_to": None, "groups": [], "unparsed": [],
    }
    i = 0
    while i < len(tokens) and tokens[i] in ("TAF", "AMD", "COR", "RTD"):
        out["is_amd"] |= tokens[i] == "AMD"
        out["is_cor"] |= tokens[i] == "COR"
        i += 1
    if i < len(tokens) and _STATION.match(tokens[i]):
        i += 1
    if i < len(tokens) and (m := _ISSUE.match(tokens[i])):
        out["header_issue"] = resolve_ddhh(int(m["d"]), int(m["h"]), int(m["m"]), issued)
        i += 1
    if i < len(tokens) and (m := _PERIOD.match(tokens[i])):
        out["valid_from"], out["valid_to"] = _period(m, issued)
        i += 1
    rest = tokens[i:]
    if "NIL" in rest:
        out["is_nil"] = 1
        return out
    if "CNL" in rest:
        out["is_cnl"] = 1
        return out

    groups = [{"kind": "BASE", "prob": 0, "start": out["valid_from"], "end": out["valid_to"], "tokens": []}]
    k = 0
    while k < len(rest):
        tok = rest[k]
        if m := _FM.match(tok):
            start = resolve_ddhh(int(m["d"]), int(m["h"]), int(m["m"]), issued)
            groups.append({"kind": "FM", "prob": 0, "start": start, "end": out["valid_to"], "tokens": []})
        elif tok in ("BECMG", "TEMPO") or _PROB.match(tok):
            pm = _PROB.match(tok)
            prob = int(pm["p"]) if pm else 0
            if prob and k + 1 < len(rest) and rest[k + 1] == "TEMPO":
                k += 1  # PROB30 TEMPO: a temporary change, with a probability
            start = end = None
            if k + 1 < len(rest) and (per := _PERIOD.match(rest[k + 1])):
                start, end = _period(per, issued)
                k += 1
            groups.append({"kind": "PROB" if prob else tok, "prob": prob, "start": start, "end": end, "tokens": []})
        else:
            groups[-1]["tokens"].append(tok)
        k += 1

    for g in groups:
        g["elements"], unparsed = parse_groups(g.pop("tokens"))
        out["unparsed"] += unparsed
    out["groups"] = groups
    return out


def _fmax(a: float, b: float) -> float:
    return b if math.isnan(a) else a if math.isnan(b) else max(a, b)


def _apply(state: dict, el: dict, *, replace: bool) -> dict:
    """``state`` changed by a group's elements.

    ``replace`` is for FM, a complete new forecast: weather it does not repeat is
    over. Elements it leaves out are inherited, the defensive reading.
    """
    new = dict(state)
    if replace:
        new.update(dict.fromkeys(WX_KEYS, 0))
    if "wind_kt" in el:
        new["wind_kt"], new["gust_kt"] = el["wind_kt"], el["gust_kt"]
    if "vis_m" in el:
        new["vis_m"] = el["vis_m"]
    if "ceiling_ft" in el:
        new["ceiling_ft"] = el["ceiling_ft"]
    if "wx" in el:  # weather named in a group replaces the weather; NSW (an empty list) ends it
        f = wx_flags(el["wx"])
        new.update(
            precip=int(f["precip_int"] > 0), snow=f["snow"], freezing=f["freezing"] | f["ice"], ts=f["ts"], fog=f["fog"]
        )
    return new


def _worst(states: list[dict]) -> dict:
    wind = vis = ceiling = NAN
    for s in states:
        wind = _fmax(wind, _fmax(s["wind_kt"], s["gust_kt"]))
        vis = fmin(vis, s["vis_m"])
        ceiling = fmin(ceiling, s["ceiling_ft"])
    return {"wind_kt": wind, "vis_m": vis, "ceiling_ft": ceiling, **{k: max(s[k] for s in states) for k in WX_KEYS}}


def taf_segments(parsed: dict) -> list[dict]:
    """Constant-state segments tiling the TAF's validity, prevailing and worst case."""
    v0, v1 = parsed["valid_from"], parsed["valid_to"]
    groups = parsed["groups"]
    changes = [g for g in groups[1:] if g["start"] is not None and (g["kind"] == "FM" or g["end"] is not None)]
    empty = {"wind_kt": NAN, "gust_kt": NAN, "vis_m": NAN, "ceiling_ft": NAN, **dict.fromkeys(WX_KEYS, 0)}
    base = _apply(empty, groups[0]["elements"], replace=True)

    cuts = {v0, v1}
    for g in changes:
        for x in (g["start"], None if g["kind"] == "FM" else g["end"]):
            if x is not None and v0 < x < v1:
                cuts.add(x)
    edges = sorted(cuts)

    out = []
    for s, e in zip(edges[:-1], edges[1:]):
        prevailing = base
        for g in changes:  # in text order, which is chronological
            if g["kind"] == "FM" and g["start"] <= s:
                prevailing = _apply(prevailing, g["elements"], replace=True)
            elif g["kind"] == "BECMG" and g["end"] <= s:
                prevailing = _apply(prevailing, g["elements"], replace=False)
        active = [g for g in changes if g["kind"] in ("TEMPO", "PROB", "BECMG") and g["start"] <= s < g["end"]]
        worst = _worst([prevailing] + [_apply(prevailing, g["elements"], replace=False) for g in active])
        out.append({
            "seg_start": s,
            "seg_end": e,
            **{f"p_{k}": prevailing[k] for k in PREVAILING},
            **{f"w_{k}": worst[k] for k in WORST},
            "tempo_prob": max((g["prob"] or TEMPO_PROB for g in active if g["kind"] != "BECMG"), default=0),
            "becmg_transition": int(any(g["kind"] == "BECMG" for g in active)),
        })
    return out


# --- normalisation -------------------------------------------------------------

def load_raw() -> pd.DataFrame:
    """Every downloaded TAF as (station, issued, text).

    Ogimet's CSV header is Spanish even in English mode, so rows are matched by
    shape, not by column name; its timestamp is the TAF's issue time.
    """
    files = sorted((paths.WEATHER_RAW / "taf").glob("*.csv"))
    if not files:
        raise FileNotFoundError("no raw TAF chunks; run `python -m prc fetch-weather --only taf`")
    rows: list[dict] = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            if m := _ROW.match(line):
                issued = datetime(int(m["y"]), int(m["mo"]), int(m["d"]), int(m["h"]), int(m["mi"]), tzinfo=timezone.utc)
                rows.append({"station": m["st"], "issued": issued, "text": m["text"].strip()})
            elif rows and not line.upper().startswith(("ESTACION", "ICAOIND", "#")):
                rows[-1]["text"] += " " + line  # a report continued on the next line
    return pd.DataFrame(rows)


SEGMENT_FLOATS = ("p_wind_kt", "p_gust_kt", "p_vis_m", "p_ceiling_ft", "w_wind_kt", "w_vis_m", "w_ceiling_ft")
SEGMENT_INTS = (
    "p_precip", "p_snow", "p_freezing", "p_ts", "p_fog",
    "w_precip", "w_snow", "w_freezing", "w_ts", "w_fog", "tempo_prob", "becmg_transition",
)


def _utc(values) -> pd.Series:
    return pd.to_datetime(pd.Series(values), utc=True).astype("datetime64[us, UTC]")


def normalise_taf() -> dict[str, paths.Path]:
    raw = load_raw().drop_duplicates(["station", "issued", "text"])
    raw = raw.sort_values(["station", "issued"], kind="stable").reset_index(drop=True)
    issues: list[dict] = []
    segments: list[dict] = []
    header_mismatch = 0
    for taf_id, (station, issued, text) in enumerate(zip(raw["station"], raw["issued"], raw["text"])):
        p = parse_taf(text, issued)
        v0, v1 = p["valid_from"], p["valid_to"]
        ok = not p["is_nil"] and v0 is not None and v1 is not None and v0 < v1 and bool(p["is_cnl"] or p["groups"])
        header_mismatch += p["header_issue"] is not None and p["header_issue"] != issued
        issues.append({
            "taf_id": taf_id, "station": station, "issue_time": issued, "valid_from": v0, "valid_to": v1,
            "is_amd": p["is_amd"], "is_cor": p["is_cor"], "is_cnl": p["is_cnl"], "is_nil": p["is_nil"],
            "n_groups": len(p["groups"]), "parse_ok": int(ok), "unparsed": " ".join(p["unparsed"]), "raw": text,
        })
        if ok and not p["is_cnl"]:
            segments.extend({"taf_id": taf_id, "station": station, **s} for s in taf_segments(p))

    iss = pd.DataFrame(issues)
    for c in ("issue_time", "valid_from", "valid_to"):
        iss[c] = _utc(iss[c])
    iss = iss.astype({
        "taf_id": "int32", "is_amd": "int8", "is_cor": "int8", "is_cnl": "int8", "is_nil": "int8",
        "n_groups": "int16", "parse_ok": "int8", "station": "category", "unparsed": "string", "raw": "string",
    })
    seg = pd.DataFrame(segments)
    seg["seg_start"], seg["seg_end"] = _utc(seg["seg_start"]), _utc(seg["seg_end"])
    seg = seg.astype({
        "taf_id": "int32", "station": "category",
        **dict.fromkeys(SEGMENT_FLOATS, "float32"), **dict.fromkeys(SEGMENT_INTS, "int8"),
    })

    paths.WEATHER.mkdir(parents=True, exist_ok=True)
    iss.to_parquet(paths.TAF_ISSUES_FILE, index=False)
    seg.to_parquet(paths.TAF_SEGMENTS_FILE, index=False)
    print(f"taf: {len(iss):,} issues ({iss['parse_ok'].mean():.2%} parsed, {int(iss['is_amd'].sum()):,} AMD, "
          f"{header_mismatch} header/issue-time mismatches), {len(seg):,} segments -> {paths.WEATHER}")
    return {"taf_issues": paths.TAF_ISSUES_FILE, "taf_segments": paths.TAF_SEGMENTS_FILE}


def coverage_report() -> str:
    """Per-station TAF counts, and the share of departures with a TAF in force."""
    from prc import data
    from prc.features._wxlib import station_rows
    from prc.features.core import epoch_seconds
    from prc.features.weather_fcst import load_tafs, segment_index, select_taf

    iss = pd.read_parquet(paths.TAF_ISSUES_FILE)
    iss["station"] = iss["station"].astype(str)
    g = iss.groupby("station")
    per_station = pd.DataFrame({
        "issues": g.size(),
        "amd_%": (100 * g["is_amd"].mean()).round(1),
        "cnl": g["is_cnl"].sum(),
        "nil": g["is_nil"].sum(),
        "parse_ok_%": (100 * g["parse_ok"].mean()).round(2),
        "with_unparsed_%": (100 * iss["unparsed"].ne("").groupby(iss["station"]).mean()).round(2),
    })
    unparsed = iss.loc[iss["unparsed"].ne(""), "unparsed"].str.split().explode().value_counts().head(15)

    mvt = data.load_movements(columns=["apt", "is_dep", "split", "MVT_TIME_UTC_mvt"])
    dep = mvt.loc[mvt["is_dep"]]
    t = epoch_seconds(dep["MVT_TIME_UTC_mvt"]).to_numpy()
    tafs = load_tafs()
    now = np.zeros(len(dep), dtype=bool)
    planning = np.zeros(len(dep), dtype=bool)
    for station, rows in station_rows(dep["apt"]).items():
        tab = tafs.get(station)
        if tab is None or not len(rows):
            continue
        q = t[rows]
        now[rows] = segment_index(tab, select_taf(tab, q, q), q) >= 0
        planning[rows] = segment_index(tab, select_taf(tab, q - 6 * 3600, q), q) >= 0
    cov = pd.DataFrame({"apt": dep["apt"].astype(str), "split": dep["split"].astype(str), "now": now, "planning": planning})
    coverage = (100 * cov.groupby(["apt", "split"])[["now", "planning"]].mean()).round(2).unstack("split")

    return "\n".join([
        "=== TAF: per station ===",
        per_station.to_string(),
        "\n=== TAF: most frequent unparsed groups ===",
        unparsed.to_string() if len(unparsed) else "none",
        "\n=== departures with a TAF in force at takeoff (%): latest, and issued >= 6 h before ===",
        coverage.to_string(),
    ])
