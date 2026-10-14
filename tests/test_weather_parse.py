"""METAR text parsing and download-response classification. No network, no data.

The fixtures are real report shapes from the ten airports, chosen for the
groups that break naive parsers: remarks carrying extra wind groups (LTFM),
RVR and directional minimum visibility (EHAM, LFPG), obscured skies (VV///),
corrected reports (EGLL), freezing precipitation and TREND forecasts.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from prc.weather.download import classify_iem, classify_ogimet, needed_months
from prc.weather.metar import parse_metar
from prc.weather.tokens import CEIL_NONE_FT, VIS_CAP_M, parse_wx, wx_flags


def test_routine_automatic_report():
    r = parse_metar("EDDF 150020Z AUTO 21007KT 9999 OVC018 01/M04 Q1032 NOSIG")
    assert r["is_auto"] == 1 and r["report_ddhhmm"] == "150020"
    assert (r["wind_dir_deg"], r["wind_kt"]) == (210.0, 7.0) and math.isnan(r["gust_kt"])
    assert r["vis_m"] == VIS_CAP_M and r["ceiling_ft"] == 1800.0
    assert (r["temp_c"], r["dewpt_c"], r["qnh_hpa"]) == (1.0, -4.0, 1032.0)
    assert r["trend_kind"] == "NOSIG" and r["unparsed"] == ""


def test_remarks_are_not_parsed():
    """LTFM appends per-runway winds after RMK; only the main wind group counts."""
    r = parse_metar("LTFM 150020Z 07011KT 9999 SCT008 24/22 Q1011 NOSIG RMK RWY17L 08011KT RWY34L 06009KT")
    assert (r["wind_dir_deg"], r["wind_kt"]) == (70.0, 11.0)
    assert r["ceiling_ft"] == CEIL_NONE_FT  # SCT is not a ceiling


def test_rvr_minimum_visibility_and_fog():
    r = parse_metar("EHAM 150025Z 20004KT 1200 0500NW R18C/0750D BR BCFG FEW001 04/04 Q1030 TEMPO 0400 FG")
    assert (r["vis_m"], r["vis_min_m"], r["rvr_min_m"]) == (1200.0, 500.0, 750.0)
    assert r["wx_fog"] == 1 and r["wx_mist"] == 1 and r["precip_int"] == 0
    assert r["trend_kind"] == "TEMPO" and r["trend_vis_m"] == 400.0 and r["trend_fog"] == 1


def test_obscured_sky_and_rvr_above_range():
    r = parse_metar("LFPG 150030Z 00000KT 2200 0350 R26R/P2000 BR VV/// M01/M01 Q1035 NOSIG")
    assert r["ceiling_ft"] == 0.0  # VV///: sky obscured, height not measured
    assert r["rvr_min_m"] == 2000.0 and r["vis_min_m"] == 350.0
    assert (r["wind_dir_deg"], r["wind_kt"]) == (0.0, 0.0)


def test_no_cloud_detected_means_no_ceiling():
    r = parse_metar("LSZH 150020Z AUTO 04005KT 7000 3400E NCD M02/M03 Q1036 NOSIG")
    assert r["ceiling_ft"] == CEIL_NONE_FT and r["vis_min_m"] == 3400.0


def test_corrected_report_with_snow_showers_and_cb():
    r = parse_metar("EGLL 150020Z COR AUTO 24012G25KT 9999 -SHSN FEW012CB SCT025 02/M01 Q1003 TEMPO SHSNRA")
    assert r["is_cor"] == 1 and r["gust_kt"] == 25.0
    assert r["precip_int"] == 1 and r["wx_snow"] == 1 and r["wx_shower"] == 1 and r["cb_tcu"] == 1
    assert r["trend_snowfrz"] == 1 and r["trend_precip"] == 1


def test_freezing_rain():
    r = parse_metar("EDDM 150020Z 08006KT 3000 -FZRA BR OVC004 M01/M02 Q1025 TEMPO -FZDZ")
    assert r["wx_freezing"] == 1 and r["wx_rain"] == 1 and r["precip_int"] == 1
    assert r["ceiling_ft"] == 400.0 and r["trend_snowfrz"] == 1


def test_weather_in_the_vicinity():
    r = parse_metar("EDDF 150050Z 25015G28KT 9999 VCTS VCSH FEW///CB 18/12 Q1010 NOSIG")
    assert r["wx_ts"] == 1  # lightning nearby stops ramp work regardless
    assert r["wx_shower"] == 0 and r["precip_int"] == 0
    assert r["cb_tcu"] == 1 and r["ceiling_ft"] == CEIL_NONE_FT


def test_cavok():
    r = parse_metar("LIRF 150020Z 30006KT CAVOK 22/18 Q1017 NOSIG")
    assert r["vis_m"] == VIS_CAP_M and r["ceiling_ft"] == CEIL_NONE_FT and r["precip_int"] == 0


def test_wind_in_metres_per_second():
    r = parse_metar("LTFM 150020Z 05005MPS 9999 FEW030 20/10 Q1015")
    assert r["wind_kt"] == pytest.approx(9.72, abs=0.01)


def test_runway_state_and_recent_weather():
    wet = parse_metar("EDDF 150050Z 25005KT 9999 FEW030 05/02 Q1010 RERA R25C/290050 NOSIG")
    assert wet["rwy_contam"] == 0 and wet["re_precip"] == 1
    snow = parse_metar("EDDM 150050Z 05005KT 4000 -SN OVC010 M02/M03 Q1010 RESN R26L/590235 NOSIG")
    assert snow["rwy_contam"] == 1 and snow["re_snow"] == 1
    closed = parse_metar("LSZH 150050Z 05005KT 1500 SN OVC005 M01/M01 Q1010 R88/SNOCLO")
    assert closed["snoclo"] == 1


def test_unobservable_elements_are_missing_not_unparsed():
    r = parse_metar("LSZH 150020Z AUTO /////KT //// // ////// M01/M02 Q1020")
    assert math.isnan(r["wind_kt"]) and math.isnan(r["vis_m"])
    assert r["unparsed"] == ""


def test_wind_shear_and_unavailable_groups_are_recognised():
    """Seen at LEMD/LEBL: wind-shear groups, runway groups with no data, TCU of unknown cover."""
    r = parse_metar("LEMD 150030Z 36010KT 9999 R32R///// FEW040 ///TCU 20/05 Q1015 RE// WS R32R NOSIG")
    assert r["unparsed"] == "" and r["cb_tcu"] == 1 and math.isnan(r["rvr_min_m"])
    assert parse_metar("LEBL 150030Z 24015KT 9999 SCT020 18/10 Q1010 WS ALL RWY")["unparsed"] == ""


def test_nil_report():
    assert parse_metar("EDDF 150020Z NIL")["is_nil"] == 1


def test_trend_with_time_groups():
    r = parse_metar("EHAM 150025Z 20010KT 9999 BKN030 05/02 Q1012 BECMG FM1200 3000 -SN BKN008")
    assert r["trend_kind"] == "BECMG" and r["trend_vis_m"] == 3000.0
    assert r["trend_ceiling_ft"] == 800.0 and r["trend_snowfrz"] == 1


def test_blowing_snow_is_snow_but_not_precipitation():
    flags = wx_flags([parse_wx("BLSN")])
    assert flags["snow"] == 1 and flags["precip_int"] == 0


def test_heavy_precipitation_intensity():
    assert wx_flags([parse_wx("+TSRA")])["precip_int"] == 3
    assert wx_flags([parse_wx("RA"), parse_wx("-DZ")])["precip_int"] == 2


# --- download responses --------------------------------------------------------

def test_classify_iem():
    assert classify_iem("station,valid,metar\nEDDF,2025-01-15 00:20,EDDF 150020Z ...\n") == "ok"
    assert classify_iem("station,valid,metar\n") == "empty"
    assert classify_iem("<html>Service unavailable</html>") == "error"


def test_classify_ogimet():
    header = "ESTACION,ANO,MES,DIA,HORA,MINUTO,PARTE\n"
    row = "EDDF,2025,01,15,05,00,TAF EDDF 150500Z 1506/1612 22004KT 9999 BKN020=\n"
    assert classify_ogimet(header + row) == "ok"
    assert classify_ogimet(header) == "empty"
    assert classify_ogimet("#Sorry, Your quota limit for slow queries rate has been reached") == "quota"
    assert classify_ogimet("<html><body>Internal error</body></html>") == "error"


def test_needed_months_pad_both_ends():
    t = pd.Series(pd.to_datetime(["2025-01-01 00:10", "2025-03-31 23:00"], utc=True))
    assert needed_months(pd.DataFrame({"MVT_TIME_UTC_mvt": t})) == ["2024-12", "2025-01", "2025-03", "2025-04"]
