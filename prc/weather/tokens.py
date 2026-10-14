"""The group grammar shared by METAR and TAF parsing (ICAO / European format).

A METAR body and a TAF change group are built from the same vocabulary --
wind, visibility, RVR, present weather, cloud -- so one tokenizer serves both.

`parse_groups` reports only the elements that are *present*: a TAF ``BECMG``
group changes just the elements it names, so "not mentioned" has to stay
distinguishable from "none". Callers apply their own defaults.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

NAN = math.nan

VIS_CAP_M = 10_000.0  # 9999 and CAVOK both mean "10 km or more"
CEIL_NONE_FT = 5_000.0  # no BKN/OVC/VV layer below this: operationally, no ceiling
KT_PER_MPS = 1.943844

_WIND = re.compile(r"^(?P<dir>\d{3}|VRB|///)(?P<spd>P?\d{2,3}|//)(?:G(?P<gust>P?\d{2,3}))?(?P<unit>KT|MPS)$")
_WIND_VARIATION = re.compile(r"^\d{3}V\d{3}$")
_VIS = re.compile(r"^(?P<m>\d{4})(?P<dir>NDV|N|NE|E|SE|S|SW|W|NW)?$")
_RVR = re.compile(r"^R\d{2}[LCR]?/(?P<lo>[PM]?\d{4})(?:V[PM]?\d{4})?(?:FT)?[UDN]?$")
_RWY_STATE = re.compile(r"^R\d{2}[LCR]?/(?P<s>[0-9/]{6}|CLRD[0-9/]{2}|SNOCLO)$")
_RWY_STATE_OLD = re.compile(r"^\d{2}(?P<s>[0-9/]{6})$")
_CLOUD = re.compile(r"^(?P<cov>FEW|SCT|BKN|OVC|///)(?P<h>\d{3}|///)?(?P<typ>CB|TCU|///)?$")
_UNAVAILABLE = re.compile(r"^(?:R\d{2}[LCR]?/+|RE/+)$")  # a runway or recent-weather group with no data
_WIND_SHEAR = re.compile(r"^(?:ALL|RWY|R(?:WY)?\d{2}[LCR]?)$")  # the runway part of "WS R32R" / "WS ALL RWY"
_VV = re.compile(r"^VV(?P<h>\d{3}|///)$")
_TEMP = re.compile(r"^(?P<t>M?\d{2})/(?P<d>M?\d{2}|//)?$")
_QNH = re.compile(r"^Q(?P<q>\d{4})$")
_ALTIMETER = re.compile(r"^A(?P<a>\d{4})$")
_WX = re.compile(
    r"^(?P<int>[+-]|VC)?(?P<desc>MI|PR|BC|DR|BL|SH|TS|FZ)?"
    r"(?P<ph>(?:DZ|RA|SN|SG|IC|PL|GR|GS|UP|BR|FG|FU|VA|DU|SA|HZ|PO|SQ|FC|SS|DS)*)$"
)
PRECIP = frozenset({"DZ", "RA", "SN", "SG", "IC", "PL", "GR", "GS", "UP"})
NO_CLOUD = frozenset({"NSC", "NCD", "SKC", "CLR"})

WX_FLAGS = ("precip_int", "rain", "drizzle", "snow", "freezing", "ice", "hail", "ts", "fog", "mist", "shower")


@dataclass(frozen=True)
class Wx:
    """One present-weather group, e.g. ``-SHSN`` or ``VCTS``."""

    intensity: int  # -1 light, 0 moderate, +1 heavy
    vicinity: bool
    desc: str
    phenomena: tuple[str, ...]


def parse_wx(token: str) -> Wx | None:
    m = _WX.match(token)
    if not m or not (m["desc"] or m["ph"]):
        return None
    ph = m["ph"]
    return Wx(
        intensity={"-": -1, "+": 1}.get(m["int"] or "", 0),
        vicinity=m["int"] == "VC",
        desc=m["desc"] or "",
        phenomena=tuple(ph[i : i + 2] for i in range(0, len(ph), 2)),
    )


def wx_flags(groups: list[Wx]) -> dict[str, int]:
    """Collapse weather groups into the flags the features use.

    Weather "in the vicinity" (VC) is not at the aerodrome and is ignored --
    except thunder: lightning within a few miles stops ramp work all the same.
    ``precip_int`` is 0 (none), 1 light, 2 moderate, 3 heavy.
    """
    out = dict.fromkeys(WX_FLAGS, 0)
    for w in groups:
        ph = set(w.phenomena)
        if w.desc == "TS":
            out["ts"] = 1
        if w.vicinity:
            continue
        if ph & PRECIP and w.desc not in ("BL", "DR"):  # blowing/drifting snow is not falling
            out["precip_int"] = max(out["precip_int"], w.intensity + 2)
        out["rain"] |= "RA" in ph
        out["drizzle"] |= "DZ" in ph
        out["snow"] |= bool(ph & {"SN", "SG"})
        out["freezing"] |= w.desc == "FZ"
        out["ice"] |= bool(ph & {"PL", "IC"})
        out["hail"] |= bool(ph & {"GR", "GS"})
        out["fog"] |= "FG" in ph
        out["mist"] |= "BR" in ph
        out["shower"] |= w.desc == "SH"
    return {k: int(v) for k, v in out.items()}


def parse_groups(tokens: list[str]) -> tuple[dict, list[str]]:
    """Parse a run of groups into (elements present, tokens not understood).

    Element keys, each present only if reported: ``wind_dir_deg``, ``wind_kt``,
    ``gust_kt``, ``vis_m``, ``vis_min_m``, ``rvr_min_m``, ``ceiling_ft``,
    ``cb_tcu``, ``wx`` (a list of `Wx`; an empty list means explicitly none),
    ``recent``, ``temp_c``, ``dewpt_c``, ``qnh_hpa``, ``rwy_contam``, ``snoclo``,
    ``windshear``.
    """
    el: dict = {}
    wx: list[Wx] = []
    recent: list[Wx] = []
    layers: list[tuple[str, float, str]] = []
    rvr: list[float] = []
    no_cloud = cavok = nsw = False
    unparsed: list[str] = []

    for tok in tokens:
        if set(tok) == {"/"} or _UNAVAILABLE.match(tok):  # an element the station could not observe
            continue
        if tok == "WS":  # wind shear on a runway: "WS R32R", "WS ALL RWY"
            el["windshear"] = True
            continue
        if el.get("windshear") and _WIND_SHEAR.match(tok):
            continue
        if m := _WIND.match(tok):
            unit = KT_PER_MPS if m["unit"] == "MPS" else 1.0
            el["wind_dir_deg"] = float(m["dir"]) if m["dir"].isdigit() else NAN
            el["wind_kt"] = _number(m["spd"]) * unit
            el["gust_kt"] = _number(m["gust"]) * unit
        elif _WIND_VARIATION.match(tok):
            continue
        elif tok == "CAVOK":
            cavok = True
        elif m := _VIS.match(tok):
            v = VIS_CAP_M if m["m"] == "9999" else float(m["m"])
            if "vis_m" not in el:
                el["vis_m"] = v
            else:  # a second visibility group is the directional minimum
                el["vis_min_m"] = min(el.get("vis_min_m", math.inf), v)
        elif m := _RVR.match(tok):
            rvr.append(float(m["lo"].lstrip("PM")))
        elif (m := _RWY_STATE.match(tok)) or (m := _RWY_STATE_OLD.match(tok)):
            state = m["s"]
            if state == "SNOCLO":
                el["snoclo"] = True
            else:
                # First digit is the deposit: 0-2 dry/damp/wet, 3-9 frost, snow, slush, ice.
                el["rwy_contam"] = el.get("rwy_contam", False) or state[0] in "3456789"
        elif tok == "SNOCLO":
            el["snoclo"] = True
        elif tok in NO_CLOUD:
            no_cloud = True
        elif m := _VV.match(tok):
            # VV/// is a sky obscured with no measured vertical visibility.
            layers.append(("VV", 0.0 if m["h"] == "///" else float(m["h"]) * 100, ""))
        elif m := _CLOUD.match(tok):
            layers.append((m["cov"], NAN if not m["h"] or m["h"] == "///" else float(m["h"]) * 100, m["typ"] or ""))
        elif tok == "NSW":
            nsw = True
        elif tok.startswith("RE") and (w := parse_wx(tok[2:])):
            recent.append(w)
        elif w := parse_wx(tok):
            wx.append(w)
        elif m := _TEMP.match(tok):
            el["temp_c"] = _signed(m["t"])
            el["dewpt_c"] = _signed(m["d"]) if m["d"] and m["d"] != "//" else NAN
        elif m := _QNH.match(tok):
            el["qnh_hpa"] = float(m["q"])
        elif m := _ALTIMETER.match(tok):
            el["qnh_hpa"] = float(m["a"]) / 100 * 33.8639
        else:
            unparsed.append(tok)

    if cavok:
        el["vis_m"] = VIS_CAP_M
        el.pop("vis_min_m", None)
        el["ceiling_ft"] = CEIL_NONE_FT
        el["cb_tcu"] = False
        el["wx"] = []
    else:
        if layers or no_cloud:
            el["ceiling_ft"] = _ceiling(layers)
            el["cb_tcu"] = any(typ in ("CB", "TCU") for _, _, typ in layers)
        if wx or nsw:
            el["wx"] = wx
    if recent:
        el["recent"] = recent
    if rvr:
        el["rvr_min_m"] = min(rvr)
    return el, unparsed


def _ceiling(layers: list[tuple[str, float, str]]) -> float:
    """Lowest BKN/OVC base or vertical visibility, capped; CEIL_NONE_FT if there is none."""
    bases = [h for cov, h, _ in layers if cov in ("BKN", "OVC", "VV")]
    known = [h for h in bases if not math.isnan(h)]
    if known:
        return min(min(known), CEIL_NONE_FT)
    if bases or any(cov == "///" for cov, _, _ in layers):
        return NAN  # a ceiling (or an unknown cover) was reported without a usable height
    return CEIL_NONE_FT


def _number(s: str | None) -> float:
    """``12`` -> 12.0; missing (``None``, ``//``) -> NaN; ``P49`` ("above 49") -> 49.0."""
    if not s or "/" in s:
        return NAN
    return float(s.lstrip("P"))


def _signed(s: str) -> float:
    return -float(s[1:]) if s.startswith("M") else float(s)


def fmin(a: float, b: float) -> float:
    """min() that ignores NaN (NaN only if both are)."""
    if math.isnan(a):
        return b
    if math.isnan(b):
        return a
    return min(a, b)
