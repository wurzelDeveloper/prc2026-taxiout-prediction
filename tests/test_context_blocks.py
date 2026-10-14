"""The context blocks added for the 2026 sweep, on one hand-built airport morning. No data, no network.

Frankfurt, 10 March 2025: four arrivals, five departures (one the evening
before, one untyped and NM-less), one more departure the next morning and one
ranking row in January 2026. Every expected value below is worked out by hand
from that timeline; the point is that each block reads the right neighbours.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prc.features.registry import FeatureContext

T = pd.Timestamp("2025-03-10 00:00", tz="UTC")


def at(day_hhmm: str, day: int = 0) -> pd.Timestamp:
    return T + pd.Timedelta(days=day) + pd.Timedelta(hours=int(day_hhmm[:2]), minutes=int(day_hhmm[3:5]), seconds=int(day_hhmm[6:8]) if len(day_hhmm) > 5 else 0)


def _frame():
    cols = [
        "mvt_id", "apt", "is_dep", "split", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt",
        "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_mvt", "FLIGHT_ID_mvt", "WK_TBL_CAT_flt", "ADEP_mvt",
        "AOBT_3_flt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt",
    ]
    NaT = pd.NaT
    rows = [
        # arrivals: landing, in-block, schedule, taxi-in
        (101, "EDDF", False, "train", at("05:30"), at("05:40"), at("05:35"), 600.0, "07C", "A1", "A320", "LH100", 1.0, "M", "EGLL", NaT, NaT, NaT, NaT),
        (102, "EDDF", False, "train", at("06:20"), at("06:30"), at("06:25"), 600.0, "18", "A1", "A320", "LH102", 2.0, "M", "LFPG", NaT, NaT, NaT, NaT),
        (103, "EDDF", False, "train", at("05:50"), at("06:00"), at("05:55"), 600.0, "18", "B1", "B77W", "UA900", 3.0, "H", "KJFK", NaT, NaT, NaT, NaT),
        (104, "EDDF", False, "train", at("06:05"), at("06:26"), at("06:00"), 1260.0, "18", "A2", "A320", "LH200", 4.0, "M", "LEMD", NaT, NaT, NaT, NaT),
        # departures: takeoff, (block masked), schedule; NM times
        (1, "EDDF", True, "train", at("05:00"), NaT, at("04:50"), np.nan, "18", "B1", "A320", "LH050", 11.0, "M", "EDDF", at("04:48"), at("04:48"), at("04:45"), at("04:45")),
        (2, "EDDF", True, "train", at("06:10"), NaT, at("05:50"), np.nan, "18", "A1", "A320", "LH101", 12.0, "M", "EDDF", at("05:55"), at("05:55"), at("05:45"), at("05:50")),
        (3, "EDDF", True, "train", at("06:12"), NaT, at("06:00"), np.nan, "18", "B1", "B77W", "UA901", 13.0, "H", "EDDF", at("05:58:30"), at("05:50"), at("05:50"), at("05:50")),
        (4, "EDDF", True, "train", at("06:13"), NaT, at("06:00"), np.nan, "18", "A2", None, "LH201", np.nan, None, "EDDF", NaT, NaT, NaT, NaT),
        (5, "EDDF", True, "train", at("07:30"), NaT, at("07:15"), np.nan, "25C", "A1", "A320", "LH300", 15.0, "M", "EDDF", at("07:10"), at("07:10"), at("07:00"), at("07:00")),
        (6, "EDDF", True, "train", at("06:10", day=1), NaT, at("05:50", day=1), np.nan, "18", "A1", "A320", "LH101", 16.0, "M", "EDDF", at("05:55", day=1), at("05:55", day=1), at("05:45", day=1), at("05:50", day=1)),
        (7, "EDDF", True, "rank", pd.Timestamp("2026-01-05 06:10", tz="UTC"), NaT, pd.Timestamp("2026-01-05 05:50", tz="UTC"), np.nan, "18", "A1", "A320", "LH101", 17.0, "M", "EDDF",
         pd.Timestamp("2026-01-05 05:55", tz="UTC"), pd.Timestamp("2026-01-05 05:55", tz="UTC"), NaT, NaT),
    ]
    mvt = pd.DataFrame(rows, columns=cols).set_index("mvt_id")
    mvt["apt"] = mvt["apt"].astype("category")
    for col, value in {
        "CALLSIGN_flt": None, "ADES_mvt": "EGLL", "ADES_flt": "EGLL", "ADES_FILED_flt": "EGLL", "ARVT_1_flt": pd.NaT, "ARVT_3_flt": pd.NaT,
        "AIRCRAFT_TYPE_flt": None, "MARKET_SEGMENT_flt": None, "FLIGHT_TYPE_flt": None, "FLIGHT_RULE_mvt": "I", "FLIGHT_RULE_flt": None,
        "AIRCRAFT_OPERATOR_flt": None,
    }.items():
        mvt[col] = value
    for col in ("ARVT_1_flt", "ARVT_3_flt"):
        mvt[col] = pd.to_datetime(mvt[col], utc=True)
    y = pd.Series({1: 700.0, 2: 900.0, 3: 840.0, 4: 1380.0, 5: 1200.0, 6: 5000.0})
    return mvt, y


def _deps(mvt, names, y=None, fit_ids=None):
    from prc.features import aircraft, congestion, nm_schedule, stale_schedule, stand_runway, temporal

    dep_idx = mvt.index[mvt["is_dep"]]
    fns = {
        "stand_runway_keys": stand_runway.stand_runway_keys,
        "aircraft_identity": aircraft.aircraft_identity,
        "calendar": temporal.calendar,
        "stale_schedule": stale_schedule.stale_schedule,
        "nm_deltas": nm_schedule.nm_deltas,
        "runway_queue": congestion.runway_queue,
    }
    deps = {}
    for name in names:
        if name == "airport_load":
            deps[name] = congestion.airport_load(mvt, FeatureContext(dep_index=dep_idx, deps={"runway_queue": fns["runway_queue"](mvt, FeatureContext(dep_index=dep_idx))}))
        else:
            deps[name] = fns[name](mvt, FeatureContext(dep_index=dep_idx))
    ctx = FeatureContext(dep_index=dep_idx, deps=deps)
    if y is not None:
        fit = pd.Series(False, index=mvt.index)
        fit.loc[list(fit_ids or y.index)] = True
        ctx.fit_mask = fit
        ctx.target = y.reindex(dep_idx)
    return ctx


def test_nm_cross_deltas_and_fingerprints():
    from prc.features.nm_context import nm_cross_deltas

    mvt, _ = _frame()
    out = nm_cross_deltas(mvt, _deps(mvt, []))
    assert out.loc[2, "aobt_eq_lobt"] == 1 and out.loc[2, "aobt3_minus_lobt"] == 0 and out.loc[2, "lobt_minus_eobt1"] == 300
    assert out.loc[2, "aobt3_minus_iobt_full"] == 600 and out.loc[2, "lobt_eq_sched"] == 0 and out.loc[2, "aobt_has_seconds"] == 0
    assert out.loc[3, "aobt_has_seconds"] == 1 and out.loc[3, "aobt3_minus_lobt"] == 510 and out.loc[3, "lobt_eq_sched"] == 0
    assert out.loc[1, "lobt_eq_sched"] == 0 and out.loc[5, "lobt_minus_eobt1"] == 600
    assert np.isnan(out.loc[4, "aobt3_minus_lobt"]) and out.loc[4, "aobt_eq_lobt"] == 0 and out.loc[4, "aobt_has_seconds"] == 0


def test_nm_hour_context_averages_the_airport_hour():
    from prc.features.nm_context import nm_hour_context

    mvt, _ = _frame()
    out = nm_hour_context(mvt, _deps(mvt, ["nm_deltas"]))
    # 06:00 UTC hour at EDDF: departures 2, 3, 4 (4 has no NM match).
    assert out.loc[2, "nm_hour_n_dep"] == 3 and out.loc[5, "nm_hour_n_dep"] == 1
    assert out.loc[2, "nm_hour_share_nm_missing"] == pytest.approx(1 / 3)
    # actual off-block past the plan: row 3 (05:58:30 vs 05:50) yes, row 2 no, row 4 unknown -> 1/2
    assert out.loc[3, "nm_hour_share_aobt3_gt_lobt"] == pytest.approx(0.5)
    # aobt3 - eobt1: row 2 = 300 s, row 3 = 510 s -> mean 405
    assert out.loc[4, "nm_hour_mean_aobt3_minus_eobt1"] == pytest.approx(405.0)


def test_stale_day_context_counts_the_suspect_day():
    from prc.features.stale_context import stale_day_context

    mvt, _ = _frame()
    out = stale_day_context(mvt, _deps(mvt, ["stale_schedule"]))
    # Row 4 is the only untyped, NM-less departure; five departures on 10 March.
    assert out.loc[2, "stale_day_suspect_n"] == 1 and out.loc[2, "stale_day_suspect_share"] == pytest.approx(0.2)
    assert out.loc[4, "stale_day_long_delta_n"] == 0 and out.loc[6, "stale_day_suspect_n"] == 0
    assert out.loc[4, "stale_stand_day_suspect_n"] == 0  # itself excluded


def test_traffic_deviation_against_the_weekday_median():
    from prc.features.traffic import traffic_deviation

    mvt, _ = _frame()
    out = traffic_deviation(mvt, _deps(mvt, []))
    # Mondays at EDDF: 10 March 2025 (5 departures) and 5 January 2026 (1) -> median 3.
    assert out.loc[2, "apt_day_dep_count"] == 5 and out.loc[2, "apt_day_dep_ratio"] == pytest.approx(5 / 3)
    assert out.loc[7, "apt_day_dep_ratio"] == pytest.approx(1 / 3)
    assert out.loc[6, "apt_day_dep_ratio"] == pytest.approx(1.0)  # the only Tuesday
    assert out.loc[2, "apt_day_arr_ratio"] == pytest.approx(4 / 2) and out.loc[7, "apt_day_arr_ratio"] == 0
    # 07:00 local hour on Mondays: two arrivals on 10 March, none in the same hour on 5 January 2026 -> median 1.
    assert out.loc[2, "apt_hour_arr_ratio"] == pytest.approx(2.0)


def test_stand_occupancy_reads_previous_and_next_occupant():
    from prc.features.stand_occupancy import stand_occupancy

    mvt, _ = _frame()
    out = stand_occupancy(mvt, _deps(mvt, ["stand_runway_keys"]))
    d1 = out.loc[2]
    assert d1["prev_arr_sched_delay_sec"] == 300 and d1["prev_arr_wake_cat"] == "M" and d1["prev_arr_carrier_matches"] == 1
    assert d1["prev_arr_origin_country"] == "EG" and d1["stand_arr_inblock_back_6h"] == 1 and d1["stand_dep_back_6h"] == 0
    assert np.isnan(d1["secs_since_prev2_arr_inblock"]) and d1["secs_to_next_arr_inblock"] == 1200 and d1["sched_ground_time_sec"] == 600
    # Row 4 at A2: the arrival there blocks in after the takeoff, so nothing before, nothing recent.
    assert out.loc[4, "prev_arr_carrier_matches"] == 0 and out.loc[4, "prev_arr_wake_cat"] == "__NA__" and np.isnan(out.loc[4, "sched_ground_time_sec"])
    assert out.loc[4, "secs_to_next_arr_inblock"] == 13 * 60
    # Row 3 at B1: the departure at 05:00 from B1 counts as stand traffic.
    assert out.loc[3, "stand_dep_back_6h"] == 1 and out.loc[3, "prev_arr_carrier_matches"] == 1
    # Next morning at A1: the previous occupant is the 06:30 arrival, the one before it the 05:40 one.
    assert out.loc[6, "secs_since_prev2_arr_inblock"] == pytest.approx(24.5 * 3600)


def test_pier_pushback_counts_forward_and_by_nm_pushback():
    from prc.features.apron import pier_pushback

    mvt, _ = _frame()
    out = pier_pushback(mvt, _deps(mvt, ["stand_runway_keys"]))
    d1 = out.loc[2]  # pier A at 06:10, NM push-back 05:55
    assert d1["pier_dep_fwd_15m"] == 1  # row 4 at 06:13
    assert d1["pier_aobt3_pm5m"] == 0  # row 4 has no NM time; nobody else pushed within 5 min
    assert d1["pier_arr_inblock_pm15m"] == 0  # the A2 in-block at 06:26 is a minute outside [05:55, 06:25]
    assert out.loc[4, "pier_arr_inblock_pm15m"] == 1  # but inside row 4's [05:58, 06:28]
    assert np.isnan(out.loc[4, "pier_aobt3_pm5m"])  # no own NM time


def test_runway_sequence_reads_the_sequence():
    from prc.features.runway_sequence import config_age_minutes, runway_sequence

    mvt, _ = _frame()
    out = runway_sequence(mvt, _deps(mvt, ["runway_queue", "aircraft_identity"]))
    assert out.loc[[2, 3, 4], "rwy_seq_index_hour"].tolist() == [0, 1, 2]
    assert out.loc[2, "rwy_prev_takeoff_wake"] == "M" and out.loc[3, "rwy_prev_takeoff_wake"] == "M" and out.loc[4, "rwy_prev_takeoff_wake"] == "H"
    assert out.loc[4, "rwy_prev_takeoff_heavy"] == 1 and out.loc[3, "rwy_prev_takeoff_heavy"] == 0
    assert out.loc[2, "rwy_gap_prev_arrival"] == 300 and out.loc[2, "rwy_gap_next_arrival"] == 600  # landings 06:05 and 06:20 on 18
    assert out.loc[2, "rwy_arr_pm10m"] == 2
    assert out.loc[4, "rwy_saturation_15m"] == pytest.approx(1 / 3)  # gaps 70 min, 120 s, 60 s -> one below 90 s
    assert out.loc[4, "apt_min_since_rwy_config_change"] == 60  # runway 18 since the 05:00 bucket
    assert out.loc[5, "apt_min_since_rwy_config_change"] == 0  # 25C is new
    assert out.loc[2, "apt_distinct_rwy_30m_bucket"] == 1

    idx = pd.Index([1, 2, 3, 4])
    age = config_age_minutes(pd.Series(["EDDF"] * 4, index=idx), pd.Series([0, 900, 1800, 2700], index=idx, dtype="float64"),
                             pd.Series(["18", "18", "25C", "25C"], index=idx))
    assert age.tolist() == [0, 15, 0, 15]


def test_taxi_window_counts_and_overtakes():
    from prc.features.taxi_windows import overtakes, taxi_window_counts

    mvt, _ = _frame()
    out = taxi_window_counts(mvt, _deps(mvt, []))
    d1, d3 = out.loc[2], out.loc[4]
    assert d1["tw_rwy_dep_during_taxi"] == 0 and d1["tw_apt_arr_during_taxi"] == 1  # the 06:05 landing
    assert d1["tw_apt_dep_since_sched"] == 0
    assert np.isnan(d3["tw_rwy_dep_during_taxi"]) and np.isnan(d3["tw_overtaken_by_30m"])  # no NM time
    assert out.loc[3, "tw_rwy_dep_during_taxi"] == 1 and out.loc[3, "tw_apt_arr_during_taxi"] == 1  # window 05:58:30-06:12: row 2, landing 06:05
    # Row 3 pushed at 05:58:30, after row 2 (05:55), and took off after it: no overtake either way between them.
    assert out.loc[2, "tw_i_overtook_30m"] == 0 and out.loc[3, "tw_overtaken_by_30m"] == 0

    t = np.array([0.0, 100.0, 200.0, 5000.0])
    aobt = np.array([-900.0, -1000.0, -800.0, 3000.0])
    g = np.zeros(4, dtype="int64")
    by, me = overtakes(t, aobt, g)
    # Event 1 pushed before event 0 but took off after it: 0 overtook 1; event 2 pushed after both and took off last.
    assert by.tolist() == [0, 1, 0, 0] and me.tolist() == [1, 0, 0, 0]


def test_arrival_conditions_average_recent_taxi_in_anomalies():
    from prc.features.arrivals_live import arrival_conditions

    mvt, _ = _frame()
    ctx = _deps(mvt, [])
    ctx.params = {"min_support": 4}  # pairings and runways are too thin: the airport median (600) is the baseline
    out = arrival_conditions(mvt, ctx)
    d1 = out.loc[2]  # landings in [05:10, 06:10]: 05:30 (0), 05:50 (0), 06:05 (+660)
    assert d1["arr_anom_n_60m"] == 3 and d1["arr_anom_mean_60m"] == pytest.approx(220.0) and d1["arr_anom_max_60m"] == 660
    assert d1["arr_anom_share_2x_60m"] == pytest.approx(1 / 3)
    assert d1["arr_anom_mean_30m"] == pytest.approx(330.0)  # 05:50 and 06:05
    assert np.isnan(out.loc[1, "arr_anom_mean_60m"]) and out.loc[1, "arr_anom_n_60m"] == 0  # nothing landed before 05:00


def test_flight_number_history_smooths_and_follows_the_fit_mask():
    from prc.features.flight_number import PRIOR_WEIGHT, flight_number_history

    mvt, y = _frame()
    out = flight_number_history(mvt, _deps(mvt, ["stale_schedule"], y=y))
    # LH101 has two fit rows, one long (5,000 s); the airport rate is 1 of 6.
    want = (1 + PRIOR_WEIGHT * (1 / 6)) / (2 + PRIOR_WEIGHT)
    assert out.loc[2, "fn_long_rate"] == pytest.approx(want, rel=1e-5) and out.loc[7, "fn_long_rate"] == pytest.approx(want, rel=1e-5)
    assert out.loc[2, "fn_n_fit"] == 2 and out.loc[7, "fn_n_fit"] == 2 and np.isnan(out.loc[2, "fn_y_median"])  # below min_support
    assert out.loc[3, "fn_stale_rate"] == 0
    half = flight_number_history(mvt, _deps(mvt, ["stale_schedule"], y=y, fit_ids=[1, 2, 3, 4, 5]))
    assert half.loc[2, "fn_long_rate"] == 0 and half.loc[2, "fn_n_fit"] == 1


def test_flight_number_context_counts_rows_and_days():
    from prc.features.flight_number import flight_number_context

    mvt, _ = _frame()
    out = flight_number_context(mvt, _deps(mvt, []))
    assert out.loc[2, "fn_n_rows_apt"] == 3 and out.loc[2, "fn_n_months_active"] == 2 and out.loc[2, "fn_rows_per_active_month"] == 1.5
    assert out.loc[2, "fn_prev_day_exists"] == 0 and out.loc[6, "fn_prev_day_exists"] == 1 and out.loc[7, "fn_prev_day_exists"] == 0
    assert out.loc[6, "fn_hours_since_prev_dep"] == 24 and out.loc[7, "fn_hours_since_prev_dep"] == 720 and np.isnan(out.loc[2, "fn_hours_since_prev_dep"])
    assert out["fn_same_day_dupes"].fillna(0).eq(0).all() and out.loc[2, "fn_sched_mod_std"] == 0


def test_key_target_stats_medians_and_pair_tail():
    from prc.features.key_stats import key_target_stats

    mvt, y = _frame()
    ctx = _deps(mvt, ["stand_runway_keys", "aircraft_identity", "calendar", "stale_schedule"], y=y)
    ctx.params = {"min_support": 1}
    out = key_target_stats(mvt, ctx)
    assert out.loc[3, "y_med_carrier_apt"] == 840 and out.loc[2, "y_med_carrier_apt"] == 1200  # LH: 700, 900, 1380, 1200, 5000
    assert out.loc[7, "y_med_carrier_apt"] == 1200  # the ranking row looks up the same table
    assert out.loc[2, "y_n_stand_runway"] == 2 and out.loc[2, "y_p90_stand_runway"] == pytest.approx(900 + 0.9 * 4100)
    assert out.loc[2, "y_long_rate_stand_runway"] > out.loc[3, "y_long_rate_stand_runway"]
    strict = key_target_stats(mvt, _deps(mvt, ["stand_runway_keys", "aircraft_identity", "calendar", "stale_schedule"], y=y))
    assert strict["y_med_carrier_apt"].isna().all()  # min_support 20 on a six-row fit


def test_wx_deice_load_counts_departures_and_arrivals_under_deicing(monkeypatch):
    from prc.features import weather, weather_load
    from prc.features.weather_load import wx_deice_load

    def fake_metar():
        rep = pd.DataFrame(
            {
                "obs_time": [at("05:00"), at("06:12")],
                "wind_dir_deg": [180.0, 180.0], "wind_kt": [5.0, 5.0], "gust_kt": [np.nan, np.nan],
                "vis_min_m": [3000.0, 9999.0], "rvr_min_m": [np.nan, np.nan], "ceiling_ft": [800.0, 5000.0],
                "temp_c": [-2.0, 5.0], "dewpt_c": [-4.0, 0.0], "precip_int": [1.0, 0.0],
                "wx_fog": [0.0, 0.0], "wx_mist": [0.0, 0.0], "wx_freezing": [0.0, 0.0], "wx_ice": [0.0, 0.0],
                "wx_snow": [1.0, 0.0], "wx_ts": [0.0, 0.0], "cb_tcu": [0.0, 0.0],
            }
        )
        from prc.features.core import epoch_seconds

        rep["t"] = epoch_seconds(rep["obs_time"]).to_numpy()
        return {"EDDF": rep}

    monkeypatch.setattr(weather, "load_metar", fake_metar)
    monkeypatch.setattr(weather_load, "load_metar", fake_metar)
    mvt, _ = _frame()
    dep_idx = mvt.index[mvt["is_dep"]]
    base = _deps(mvt, ["airport_load"])
    deps = dict(base.deps)
    deps["wx_now"] = weather.wx_now(mvt, FeatureContext(dep_index=dep_idx))
    deps["wx_past"] = weather.wx_past(mvt, FeatureContext(dep_index=dep_idx))
    out = wx_deice_load(mvt, FeatureContext(dep_index=dep_idx, deps=deps))
    assert deps["wx_now"].loc[2, "wx_deice"] == 1 and deps["wx_now"].loc[3, "wx_deice"] == 0
    assert out.loc[4, "wx_deice_dep_back_30m"] == 1  # row 2 at 06:10 under the 05:00 report; row 3 under the 06:12 one
    assert out.loc[4, "wx_deice_arr_back_60m"] == 3  # landings 05:30, 05:50, 06:05 under the 05:00 report
    assert out.loc[1, "wx_deice_arr_back_60m"] == 0
    assert out.loc[2, "wx_snow_hours_x_load"] == pytest.approx(deps["wx_past"].loc[2, "wx_snowfrz_hours_24h"] * deps["airport_load"].loc[2, "apt_dep_back_60m"])
