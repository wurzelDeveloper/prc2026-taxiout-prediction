"""The PRU CSV readers, the two tables, the download-response check and the `pru_daily` block. No data, no network.

The fixtures are the real files' layouts cut down to a few rows, with a
Latin-1 airport name, an airport the challenge does not have, a blank cell and
a bzip2-compressed ATFM file, because those are the shapes that have to be
handled without a NaN table appearing silently.
"""

from __future__ import annotations

import bz2

import numpy as np
import pandas as pd
import pytest

from prc.pru.download import classify_pru
from prc.pru.normalise import DAILY_COLUMNS, MONTHLY_COLUMNS, daily_table, monthly_table, read_raw

AIRPORTS = ["EDDF", "LSZH"]

TAXI_OUT = """YEAR,MONTH_NUM,MONTH_MON,APT_ICAO,APT_NAME,STATE_NAME,TF,VALID_FL,NO_REF,TOTAL_REF_NB_FL,TOTAL_REF_TIME_MIN,TOTAL_ADD_TIME_MIN,COMMENT
2026,01,JAN,EDDF,Frankfurt,Germany,15667,15000,400,13292,144998.0,44105.0,
2026,01,JAN,LSZH,Zürich,Switzerland,10012,9000,300,7457,64131.0,24236.0,
2026,01,JAN,LOWW,Vienna,Austria,8025,5852,433,5419,39512.4,13566.6,
"""

ATC_PRE = """YEAR,MONTH_NUM,MONTH_MON,FLT_DATE,APT_ICAO,APT_NAME,STATE_NAME,FLT_DEP_1,FLT_DEP_IFR_2,DLY_ATC_PRE_2,FLT_DEP_3,DLY_ATC_PRE_3
2026,1,JAN,2026-01-01,EDDF,Frankfurt,Germany,429,429.0,27.0,296.0,125.0
2026,1,JAN,2026-01-02,EDDF,Frankfurt,Germany,513,513.0,313.0,354.0,733.0
2026,1,JAN,2026-01-01,LSZH,Zürich,Switzerland,300,,,200.0,50.0
"""

ALL_PRE = """YEAR,MONTH_NUM,MONTH_MON,FLT_DATE,APT_ICAO,APT_NAME,STATE_NAME,FLT_DEP_1,FLT_DEP_IFR_2,DLY_ALL_PRE_2
2026,1,JAN,2026-01-01,EDDF,Frankfurt,Germany,429,429.0,6513.82
2026,1,JAN,2026-01-03,EDDF,Frankfurt,Germany,516,515.0,24310.05
"""

APT_DLY_HEADER = (
    "YEAR,MONTH_NUM,MONTH_MON,FLT_DATE,APT_ICAO,APT_NAME,STATE_NAME,FLT_ARR_1,DLY_APT_ARR_1,DLY_APT_ARR_A_1,"
    "DLY_APT_ARR_C_1,DLY_APT_ARR_D_1,DLY_APT_ARR_W_1,FLT_ARR_1_DLY,FLT_ARR_1_DLY_15,ATFM_VERSION"
)
APT_DLY = APT_DLY_HEADER + """
2026,1,JAN,2026-01-03T00:00:00Z,EDDF,Frankfurt,Germany,430,19823,0,0,0,19823,249,212,v2
2026,1,JAN,2026-01-04T00:00:00Z,EDDF,Frankfurt,Germany,400,1000,0,0,200,800,50,20,v2
"""


def _write(tmp_path, name, text, compress=False):
    path = tmp_path / name
    if compress:
        path.write_bytes(bz2.compress(text.encode("latin-1")))
    else:
        path.write_bytes(text.encode("latin-1"))
    return path


def test_read_raw_handles_latin1_and_bzip2_and_checks_columns(tmp_path):
    taxi = read_raw("taxi_out_additional_time", _write(tmp_path, "taxi_out_additional_time_2026.csv", TAXI_OUT))
    assert taxi.loc[1, "APT_NAME"] == "Zürich" and len(taxi) == 3
    atfm = read_raw("apt_dly", _write(tmp_path, "apt_dly_2026.csv.bz2", APT_DLY, compress=True))
    assert atfm["FLT_ARR_1"].tolist() == ["430", "400"]
    with pytest.raises(FileNotFoundError):
        read_raw("apt_dly", tmp_path / "missing.csv.bz2")
    with pytest.raises(ValueError, match="missing column"):
        read_raw("apt_dly", _write(tmp_path, "apt_dly_2025.csv", ALL_PRE))
    with pytest.raises(ValueError, match="unknown PRU dataset"):
        read_raw("something_else", tmp_path / "x.csv")


def test_daily_table_is_a_full_grid_with_per_movement_rates(tmp_path):
    frames = {
        "atc_pre_departure_delays": [read_raw("atc_pre_departure_delays", _write(tmp_path, "a.csv", ATC_PRE))],
        "all_pre_departure_delays": [read_raw("all_pre_departure_delays", _write(tmp_path, "b.csv", ALL_PRE))],
        "apt_dly": [read_raw("apt_dly", _write(tmp_path, "c.csv.bz2", APT_DLY, compress=True))],
    }
    daily = daily_table(frames, AIRPORTS)
    assert list(daily.columns) == DAILY_COLUMNS
    # Two airports x four days (01-01 .. 01-04), whether or not a series has a row.
    assert len(daily) == 8 and daily["day"].dt.tz is not None
    eddf = daily.loc[daily["apt"].eq("EDDF")].set_index(daily.loc[daily["apt"].eq("EDDF"), "day"].dt.day)
    assert eddf.loc[1, "atc_predep2_dly_min_per_dep"] == pytest.approx(27 / 429)
    assert eddf.loc[2, "atc_predep3_dly_min_per_dep"] == pytest.approx(733 / 354)
    assert eddf.loc[3, "all_predep_dly_min_per_dep"] == pytest.approx(24310.05 / 515)
    assert eddf.loc[3, "apt_arr_atfm_weather_min_per_arr"] == pytest.approx(19823 / 430)
    assert eddf.loc[4, "apt_arr_atfm_deice_min_per_arr"] == pytest.approx(200 / 400)
    assert eddf.loc[4, "apt_arr_atfm_n_delayed"] == 50
    assert np.isnan(eddf.loc[4, "atc_predep2_dly_min_per_dep"])  # no ATC row for 01-04
    lszh = daily.loc[daily["apt"].eq("LSZH")].set_index(daily.loc[daily["apt"].eq("LSZH"), "day"].dt.day)
    assert np.isnan(lszh.loc[1, "atc_predep2_dly_min_per_dep"]) and lszh.loc[1, "atc_predep3_dly_min_per_dep"] == pytest.approx(0.25)
    assert lszh["apt_arr_atfm_n_arr"].isna().all()  # no ATFM rows for Zürich in the fixture


def test_daily_table_rejects_what_would_be_silently_wrong(tmp_path):
    bad_date = ATC_PRE.replace("2026-01-02", "yesterday")
    with pytest.raises(ValueError, match="unparsable FLT_DATE"):
        daily_table({"atc_pre_departure_delays": [read_raw("atc_pre_departure_delays", _write(tmp_path, "d.csv", bad_date))]}, AIRPORTS)
    negative = ALL_PRE.replace("6513.82", "-5")
    with pytest.raises(ValueError, match="negative"):
        daily_table({"all_pre_departure_delays": [read_raw("all_pre_departure_delays", _write(tmp_path, "e.csv", negative))]}, AIRPORTS)
    dupe = ALL_PRE + "2026,1,JAN,2026-01-01,EDDF,Frankfurt,Germany,429,429.0,1.0\n"
    with pytest.raises(ValueError, match="duplicate"):
        daily_table({"all_pre_departure_delays": [read_raw("all_pre_departure_delays", _write(tmp_path, "f.csv", dupe))]}, AIRPORTS)
    with pytest.raises(ValueError, match="no daily dataset"):
        daily_table({}, AIRPORTS)


def test_monthly_table_derives_the_mean_taxi_out(tmp_path):
    monthly = monthly_table([read_raw("taxi_out_additional_time", _write(tmp_path, "t.csv", TAXI_OUT))], AIRPORTS)
    assert list(monthly.columns) == MONTHLY_COLUMNS and monthly["apt"].tolist() == ["EDDF", "LSZH"]  # LOWW dropped
    eddf = monthly.iloc[0]
    assert eddf["year"] == 2026 and eddf["month"] == 1
    assert eddf["mean_taxi_out_s"] == pytest.approx(60 * (144998.0 + 44105.0) / 13292)
    assert eddf["ref_share"] == pytest.approx(13292 / 15667)
    assert eddf["add_time_s_per_fl"] == pytest.approx(60 * 44105.0 / 13292)
    with pytest.raises(ValueError, match="no taxi_out"):
        monthly_table([], AIRPORTS)


def test_classify_pru():
    assert classify_pru(TAXI_OUT.encode("latin-1"), "taxi_out_additional_time") == "ok"
    assert classify_pru(TAXI_OUT.splitlines()[0].encode(), "taxi_out_additional_time") == "empty"
    assert classify_pru(b"<html><body>Not found</body></html>", "atc_pre_departure_delays") == "error"
    assert classify_pru(b"", "all_pre_departure_delays") == "error"
    assert classify_pru(bz2.compress(b"x"), "apt_dly") == "ok"
    assert classify_pru(b"<html>", "apt_dly") == "error"


def test_pru_daily_block_reads_the_day_and_the_day_before(tmp_path, monkeypatch):
    from prc import paths
    from prc.features.pru import PRU_DAILY, pru_daily
    from prc.features.registry import FeatureContext

    days = pd.date_range("2026-01-01", "2026-01-03", freq="D", tz="UTC")
    table = pd.DataFrame(
        {
            "apt": ["EDDF"] * 3,
            "day": days,
            "atc_predep2_dly_min_per_dep": [0.1, 0.6, np.nan],
            "atc_predep3_dly_min_per_dep": [0.4, 2.0, 1.0],
            "all_predep_dly_min_per_dep": [15.0, 50.0, 47.0],
            "apt_arr_atfm_dly_min_per_arr": [0.0, 46.0, 3.4],
            "apt_arr_atfm_weather_min_per_arr": [0.0, 46.0, 2.5],
        }
    )
    table.to_parquet(tmp_path / "pru_daily.parquet", index=False)
    monkeypatch.setattr(paths, "PRU_DAILY_FILE", tmp_path / "pru_daily.parquet")

    mvt = pd.DataFrame(
        {
            "apt": pd.Categorical(["EDDF", "EDDF", "LSZH"]),
            "MVT_TIME_UTC_mvt": pd.to_datetime(["2026-01-02 10:00", "2026-01-03 23:30", "2026-01-02 08:00"], utc=True),
            "is_dep": [True, True, True],
        },
        index=pd.Index([1, 2, 3], name="mvt_id"),
    )
    out = pru_daily(mvt, FeatureContext(dep_index=mvt.index))
    assert list(out.columns) == list(PRU_DAILY)
    assert out.loc[1, "pru_atc_predep2_per_dep"] == pytest.approx(0.6)
    assert out.loc[1, "pru_atc_predep2_per_dep_prev_day"] == pytest.approx(0.1)
    assert out.loc[1, "pru_apt_arr_atfm_wx_per_arr"] == pytest.approx(46.0)
    assert np.isnan(out.loc[2, "pru_atc_predep2_per_dep"]) and out.loc[2, "pru_atc_predep3_per_dep"] == pytest.approx(1.0)
    assert out.loc[2, "pru_all_predep_per_dep_prev_day"] == pytest.approx(50.0)
    assert out["pru_daily_reported"].tolist() == [1, 1, 0]  # Zürich is not in the table
    assert out.loc[3].drop("pru_daily_reported").isna().all()
