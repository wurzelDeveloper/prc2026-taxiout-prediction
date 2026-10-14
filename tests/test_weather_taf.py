"""TAF parsing, time resolution, forecast state and TAF selection. No network, no data.

The state semantics are the part most easily got subtly wrong: a BECMG that
applies at the start of its window instead of the end, an FM that inherits
weather it should end, a TEMPO leaking into the prevailing forecast, or a
query answered from a TAF not yet valid.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from prc.features.weather_fcst import BIG, segment_index, select_taf
from prc.weather.taf import parse_taf, resolve_ddhh, taf_segments

UTC = timezone.utc
H = 3600.0


def dt(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)


def at(segments: list[dict], when: datetime) -> dict:
    """The segment in force at ``when``."""
    return next(s for s in segments if s["seg_start"] <= when < s["seg_end"])


# --- times without a month ---------------------------------------------------------

def test_validity_rolls_into_the_next_month_and_year():
    assert resolve_ddhh(1, 6, 0, dt(2025, 1, 31, 23)) == dt(2025, 2, 1, 6)
    assert resolve_ddhh(1, 0, 0, dt(2025, 12, 31, 23)) == dt(2026, 1, 1, 0)
    assert resolve_ddhh(31, 18, 0, dt(2026, 1, 1, 5)) == dt(2025, 12, 31, 18)


def test_hour_24_is_the_midnight_ending_the_day():
    assert resolve_ddhh(16, 24, 0, dt(2025, 1, 15, 23)) == dt(2025, 1, 17, 0)
    assert resolve_ddhh(28, 24, 0, dt(2025, 2, 28, 5)) == dt(2025, 3, 1, 0)  # 2025 is not a leap year


def test_impossible_times_are_rejected():
    assert resolve_ddhh(30, 6, 0, dt(2025, 3, 1, 5)) is None
    assert resolve_ddhh(15, 25, 0, dt(2025, 1, 15, 5)) is None


# --- one TAF, every kind of group ----------------------------------------------------

EDDF = (
    "TAF EDDF 150500Z 1506/1612 22004KT 9999 BKN020 "
    "TEMPO 1506/1510 4000 BR BKN008 "
    "BECMG 1512/1514 30012KT "
    "PROB30 TEMPO 1516/1520 -SN BKN006 "
    "FM152100 VRB03KT CAVOK TX05/1513Z TNM02/1606Z="
)


def eddf() -> tuple[dict, list[dict]]:
    p = parse_taf(EDDF, dt(2025, 1, 15, 5))
    return p, taf_segments(p)


def test_header_and_groups():
    p, _ = eddf()
    assert (p["valid_from"], p["valid_to"]) == (dt(2025, 1, 15, 6), dt(2025, 1, 16, 12))
    assert [g["kind"] for g in p["groups"]] == ["BASE", "TEMPO", "BECMG", "PROB", "FM"]
    assert p["groups"][3]["prob"] == 30
    assert p["unparsed"] == [] and p["is_amd"] == 0


def test_segments_tile_the_validity():
    p, segs = eddf()
    assert segs[0]["seg_start"] == p["valid_from"] and segs[-1]["seg_end"] == p["valid_to"]
    assert all(a["seg_end"] == b["seg_start"] for a, b in zip(segs, segs[1:]))


def test_tempo_changes_the_worst_case_not_the_prevailing():
    _, segs = eddf()
    s = at(segs, dt(2025, 1, 15, 7))
    assert (s["p_vis_m"], s["p_ceiling_ft"]) == (10_000.0, 2000.0)
    assert (s["w_vis_m"], s["w_ceiling_ft"]) == (4000.0, 800.0)
    assert s["tempo_prob"] == 50
    later = at(segs, dt(2025, 1, 15, 11))
    assert later["w_vis_m"] == 10_000.0 and later["tempo_prob"] == 0


def test_becmg_applies_at_the_end_of_its_transition():
    _, segs = eddf()
    during = at(segs, dt(2025, 1, 15, 13))
    assert during["p_wind_kt"] == 4.0 and during["w_wind_kt"] == 12.0 and during["becmg_transition"] == 1
    assert at(segs, dt(2025, 1, 15, 15))["p_wind_kt"] == 12.0


def test_prob_group_carries_its_probability():
    _, segs = eddf()
    s = at(segs, dt(2025, 1, 15, 17))
    assert s["tempo_prob"] == 30 and s["w_snow"] == 1 and s["p_snow"] == 0
    assert s["w_ceiling_ft"] == 600.0 and s["p_wind_kt"] == 12.0


def test_fm_replaces_the_forecast():
    _, segs = eddf()
    s = at(segs, dt(2025, 1, 16, 3))
    assert s["p_wind_kt"] == 3.0 and s["p_vis_m"] == 10_000.0 and s["p_ceiling_ft"] == 5000.0


def test_fm_ends_weather_it_does_not_repeat():
    p = parse_taf("TAF EGLL 150500Z 1506/1612 24010KT 6000 -RA BKN010 FM151200 27015G25KT 9999 SCT030=", dt(2025, 1, 15, 5))
    segs = taf_segments(p)
    before, after = at(segs, dt(2025, 1, 15, 8)), at(segs, dt(2025, 1, 15, 13))
    assert before["p_precip"] == 1 and before["p_ceiling_ft"] == 1000.0
    assert after["p_precip"] == 0 and after["p_gust_kt"] == 25.0 and after["p_ceiling_ft"] == 5000.0


def test_becmg_nsw_ends_the_weather_and_keeps_the_rest():
    p = parse_taf("TAF LSZH 150500Z 1506/1612 05005KT 9999 -RA BKN015 BECMG 1508/1510 NSW=", dt(2025, 1, 15, 5))
    segs = taf_segments(p)
    assert at(segs, dt(2025, 1, 15, 7))["p_precip"] == 1
    during = at(segs, dt(2025, 1, 15, 9))
    assert during["p_precip"] == 1 and during["w_precip"] == 1
    after = at(segs, dt(2025, 1, 15, 11))
    assert after["p_precip"] == 0 and after["p_ceiling_ft"] == 1500.0 and after["p_vis_m"] == 10_000.0


def test_amendment_cancellation_and_nil():
    assert parse_taf("TAF AMD EDDF 150620Z 1506/1612 22004KT 9999 BKN020=", dt(2025, 1, 15, 6, 20))["is_amd"] == 1
    assert parse_taf("TAF AMD EDDF 151000Z 1509/1612 CNL=", dt(2025, 1, 15, 10))["is_cnl"] == 1
    assert parse_taf("TAF EDDF 150500Z NIL=", dt(2025, 1, 15, 5))["is_nil"] == 1


def test_validity_ending_at_hour_24():
    p = parse_taf("TAF LEBL 161500Z 1618/1624 24010KT CAVOK=", dt(2025, 1, 16, 15))
    assert p["valid_to"] == dt(2025, 1, 17, 0)


# --- which TAF answers a query ----------------------------------------------------------

def _tab(issues: list[tuple]) -> dict:
    """A one-station table, one segment per TAF: (issue, valid_from, valid_to, amd, cnl)."""
    issue, vfrom, vto, amd, cnl = (np.array(x, dtype="float64") for x in zip(*issues))
    return {
        "issue": issue, "vfrom": vfrom, "vto": vto, "amd": amd, "cnl": cnl > 0,
        "seg_key": np.arange(len(issue)) * BIG,
    }


def test_the_latest_taf_that_covers_the_time_answers():
    tab = _tab([
        (-1 * H, 0 * H, 30 * H, 0, 0),  # 23Z the day before
        (5 * H, 6 * H, 36 * H, 0, 0),  # 05Z, valid only from 06Z
        (6.33 * H, 6 * H, 36 * H, 1, 0),  # 06:20 amendment of it
        (11 * H, 12 * H, 42 * H, 0, 0),  # 11Z, valid from 12Z
        (13 * H, 12 * H, 42 * H, 1, 1),  # 13Z cancellation
    ])
    q = np.array([5.5, 7.0, 11.5, 12.5, 13.5, np.nan]) * H
    assert select_taf(tab, q, q).tolist() == [0, 2, 2, 3, -1, -1]
    # The planning TAF: issued at least 6 h before the time it answers for.
    assert select_taf(tab, q - 6 * H, q)[:2].tolist() == [0, 0]


def test_segment_lookup_stays_inside_the_selected_taf():
    tab = _tab([(0.0, 0.0, 10 * H, 0, 0), (5 * H, 6 * H, 16 * H, 0, 0)])
    tab["seg_key"] = np.array([0.0, 4 * H, BIG])  # TAF 0: [0, 4h) and [4h, 10h); TAF 1: from 6h
    pos = np.array([0, 0, 1, -1])
    tau = np.array([1 * H, 5 * H, 7 * H, 7 * H])
    assert segment_index(tab, pos, tau).tolist() == [0, 1, 2, -1]
