"""Stand-ref normalisation, stand matching, source parsing and download-response checks. No network, no data.

The fixtures follow the shapes the ten airports actually use: "A14" and
"V108" (EDDF), plain numbers (EDDM, LIRF), zero-padded "C08" (EHAM), suffixed
"G11L" (LTFM), and the "NA" / "H" / "HEL" runway placeholders.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from prc.features._geolib import (
    bearing_deg,
    haversine_m,
    local_east_north_m,
    norm_runway_ident,
    norm_stand_ref,
    runway_idents,
    stand_keys,
)
from prc.geometry.download import bbox, classify_overpass, classify_ourairports, overpass_query
from prc.geometry.match import duplicate_counts, match_stands
from prc.geometry.sources import explode_refs, filter_runways, osm_stands, parse_overpass, runway_ends, split_ref


# --- identifiers ---------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, want",
    [
        ("a21l", ("A", "21", "L")),
        ("A021", ("A", "21", "")),
        (" b 7 ", ("B", "7", "")),
        ("V-101A", ("V", "101", "A")),
        ("12", ("", "12", "")),
        ("000", ("", "0", "")),
        ("C08", ("C", "8", "")),
        ("T4S501", ("T4S501", "", "")),  # unparsable: kept whole so it still matches itself
        (None, None),
        ("", None),
        ("nan", None),
        (float("nan"), None),
    ],
)
def test_norm_stand_ref(raw, want):
    assert norm_stand_ref(raw) == want


def test_stand_keys_are_tried_from_literal_to_loose():
    assert stand_keys("A021L") == ("A021L", "A21L", "A21")
    assert stand_keys("A21") == ("A21", "A21", "A21")
    assert stand_keys("ABC") == ("ABC", "ABC", None)  # nothing numeric to anchor a suffix-less key
    assert stand_keys(None) == (None, None, None)


@pytest.mark.parametrize(
    "raw, want",
    [("7L", "07L"), ("25c", "25C"), ("18", "18"), ("36", "36"), ("07C", "07C"),
     ("NA", None), ("H", None), ("HEL", None), ("", None), (None, None), ("37", None), ("00", None), ("7LL", None)],
)
def test_norm_runway_ident(raw, want):
    assert norm_runway_ident(raw) == want


def test_runway_idents_from_an_osm_ref():
    assert runway_idents("07C/25C") == ("07C", "25C")
    assert runway_idents("7/25") == ("07", "25")
    assert runway_idents("18") == ("18",)
    assert runway_idents("") == ()
    assert runway_idents(None) == ()


# --- distances -------------------------------------------------------------------

def test_haversine_and_bearing():
    assert haversine_m(0, 0, 0, 1) == pytest.approx(111_195, rel=1e-3)
    assert haversine_m(50, 8, 50, 8) == 0.0
    assert bearing_deg(0, 0, 1, 0) == pytest.approx(0.0)
    assert bearing_deg(0, 0, 0, 1) == pytest.approx(90.0)
    assert bearing_deg(0, 0, -1, 0) == pytest.approx(180.0)


def test_local_east_north():
    east, north = local_east_north_m([51.0], [0.0], 50.0, 0.0)
    assert north[0] == pytest.approx(111_195, rel=1e-3) and east[0] == 0.0
    east, north = local_east_north_m([50.0], [1.0], 50.0, 0.0)
    assert east[0] == pytest.approx(111_195 * math.cos(math.radians(50)), rel=1e-3)


# --- download ---------------------------------------------------------------------

def test_bbox_at_lat_50():
    south, west, north, east = bbox(50.0, 8.0, 7.0)
    assert north - 50.0 == pytest.approx(0.0630, abs=1e-3)
    assert 50.0 - south == pytest.approx(0.0630, abs=1e-3)
    assert east - 8.0 == pytest.approx(0.0979, abs=1e-3)
    assert 8.0 - west == pytest.approx(0.0979, abs=1e-3)


def test_overpass_query_asks_for_the_network_and_its_nodes():
    q = overpass_query(49.9, 8.4, 50.1, 8.7)
    assert "[bbox:49.90000,8.40000,50.10000,8.70000]" in q
    for tag in ("parking_position", "taxiway", "taxilane", "runway", "holding_position", "gate"):
        assert tag in q
    assert ">;" in q and "out skel qt;" in q


@pytest.mark.parametrize(
    "text, want",
    [
        ('{"version": 0.6, "elements": [{"type": "node", "id": 1, "lat": 50.0, "lon": 8.5}]}', "ok"),
        ('{"version": 0.6, "elements": []}', "empty"),
        ('{"version": 0.6, "elements": [], "remark": "runtime error: Query timed out in \\"query\\" at line 1"}', "timeout"),
        ("<html><body>Too Many Requests</body></html>", "throttled"),
        ("<html><head><title>429 rate_limited</title></head></html>", "throttled"),
        ("<html><body><h1>Not Acceptable</h1></body></html>", "error"),
        ("{not json", "error"),
        ('{"elements": "what"}', "error"),
    ],
)
def test_classify_overpass(text, want):
    assert classify_overpass(text) == want


def test_classify_ourairports():
    assert classify_ourairports('"id","airport_ref","airport_ident","length_ft"\n1,2,"EDDF",100\n', "runways.csv") == "ok"
    assert classify_ourairports('"id","ident","type","name"\n', "airports.csv") == "ok"
    assert classify_ourairports("<html>maintenance</html>", "runways.csv") == "error"


# --- OurAirports ------------------------------------------------------------------

def _runway_row(apt, le, he, length, closed="0", le_lat="50.0", he_lat="50.1"):
    return {
        "id": "1", "airport_ref": "1", "airport_ident": apt, "length_ft": length, "width_ft": "148",
        "surface": "ASP", "lighted": "1", "closed": closed,
        "le_ident": le, "le_latitude_deg": le_lat, "le_longitude_deg": "8.5", "le_elevation_ft": "300",
        "le_heading_degT": "69.6", "le_displaced_threshold_ft": "",
        "he_ident": he, "he_latitude_deg": he_lat, "he_longitude_deg": "8.6", "he_elevation_ft": "300",
        "he_heading_degT": "249.6", "he_displaced_threshold_ft": "",
    }


def test_filter_runways_drops_what_never_takes_a_departure():
    df = pd.DataFrame([
        _runway_row("EDDF", "07C", "25C", "13123"),
        _runway_row("LIRF", "16C", "34C", "11818", closed="1", le_lat="", he_lat=""),
        _runway_row("LFPG", "08H", "26H", "1444"),
        _runway_row("LSZH", "01H", "19H", "1010"),
        _runway_row("EHAM", "06", "24", "11283", he_lat=""),
        _runway_row("XXXX", "AB", "CD", "9000"),
    ])
    kept, dropped = filter_runways(df)
    assert kept["airport_ident"].tolist() == ["EDDF"]
    assert dict(zip(dropped["airport_ident"], dropped["reason"])) == {
        "LIRF": "closed", "LFPG": "heli", "LSZH": "heli", "EHAM": "no_coords", "XXXX": "ident_unparsable",
    }
    ends = runway_ends(kept)
    assert ends["ident"].tolist() == ["07C", "25C"]
    assert ends["pair_ref"].unique().tolist() == ["07C/25C"]
    assert ends["length_m"].iloc[0] == pytest.approx(13123 * 0.3048)
    assert ends.loc[ends["ident"].eq("25C"), "heading_deg"].iloc[0] == pytest.approx(249.6)


# --- OSM --------------------------------------------------------------------------

def _payload():
    return {"elements": [
        {"type": "node", "id": 1, "lat": 50.0, "lon": 8.50, "tags": {"aeroway": "parking_position", "ref": "A21"}},
        {"type": "node", "id": 5, "lat": 50.0, "lon": 8.51, "tags": {"aeroway": "gate", "ref": "D5"}},
        {"type": "way", "id": 100, "tags": {"aeroway": "taxiway", "ref": "N"}, "nodes": [1, 2, 3]},
        {"type": "way", "id": 106, "tags": {"aeroway": "parking_position", "ref": "C3;C3A"}, "nodes": [3, 4]},
        {"type": "way", "id": 107, "tags": {"aeroway": "runway", "ref": "07C/25C"}, "nodes": [3, 2]},
        {"type": "node", "id": 1, "lat": 50.0, "lon": 8.50},  # the skel copy of a tagged node
        {"type": "node", "id": 2, "lat": 50.0, "lon": 8.52},
        {"type": "node", "id": 3, "lat": 50.0, "lon": 8.53},
        {"type": "node", "id": 4, "lat": 50.0, "lon": 8.54},
    ]}


def test_parse_overpass_keeps_tags_and_dedups_nodes():
    nodes, ways = parse_overpass(_payload())
    assert nodes["id"].tolist() == [1, 5, 2, 3, 4]
    assert nodes.set_index("id").loc[1, "ref"] == "A21"
    assert ways["id"].tolist() == [100, 106, 107]
    assert ways.set_index("id").loc[106, "nodes"] == [3, 4]


@pytest.mark.parametrize(
    "raw, want",
    [
        ("A21", ["A21"]),
        ("A21;A21L", ["A21", "A21L"]),
        ("A26 A28", ["A26", "A28"]),   # two stands (EDDF)
        ("1 1A", ["1", "1A"]),
        ("V 117", ["V117"]),           # one stand with a space in it (EDDF)
        ("GA 6", ["GA6"]),
        ("1/1A, 2", ["1", "1A", "2"]),
        ("", []),
    ],
)
def test_split_ref(raw, want):
    assert split_ref(raw) == want


def test_osm_stands_from_nodes_ways_and_gates():
    stands = osm_stands(*parse_overpass(_payload()))
    by_id = stands.set_index("stand_id")
    assert set(by_id.index) == {1, 5, -106}
    assert by_id.loc[1, "kind"] == "pp_node" and by_id.loc[5, "kind"] == "gate" and by_id.loc[-106, "kind"] == "pp_way"
    # a parking-position way ends at the nose wheel and starts on the taxiway
    assert by_id.loc[-106, "lon"] == pytest.approx(8.54) and by_id.loc[-106, "entry_node"] == 3
    refs = explode_refs(stands)
    assert sorted(refs.loc[refs["stand_id"].eq(-106), "ref"]) == ["C3", "C3A"]


# --- matching -----------------------------------------------------------------------

def _refs():
    rows = [
        (1, "pp_node", "A21", 0.0, 0.0),
        (2, "pp_node", "A21L", 100.0, 0.0),
        (3, "pp_node", "B07", -100.0, 0.0),
        (-4, "pp_way", "C3", 0.0, 100.0),    # a parking position mapped as a way: negative id
        (-4, "pp_way", "C3A", 0.0, 100.0),
        (6, "pp_node", "A021", 0.0, -60.0),
        (5, "gate", "D5", 50.0, 50.0),
        (7, "gate", "C3A", 5.0, 105.0),      # the gate serving C3A must not outrank the stand itself
    ]
    return pd.DataFrame(rows, columns=["stand_id", "kind", "ref", "east_m", "north_m"])


def _mvt_stands(names):
    return pd.DataFrame({"apt": "TEST", "stand_mvt": names, "n_dep_train": 10, "n_dep_rank": 1})


def test_match_stands_rules_in_order():
    out = match_stands(_mvt_stands(["A21", "a21l", "B7", "C3A", "D5", "Z9", "A21X"]), _refs()).set_index("stand_mvt")
    assert out.loc["A21", ["stand_id", "match_source", "ambiguous"]].tolist() == [1, "exact", 0]
    assert out.loc["a21l", ["stand_id", "match_source"]].tolist() == [2, "exact"]  # case is cleaned before "exact"
    assert out.loc["B7", ["stand_id", "match_source"]].tolist() == [3, "norm"]
    assert out.loc["C3A", ["stand_id", "match_source"]].tolist() == [-4, "exact"]
    assert out.loc["D5", ["stand_id", "match_source"]].tolist() == [5, "gate"]
    assert out.loc["Z9", ["stand_id", "match_source"]].tolist() == [-1, "none"]
    # "A21X" has no literal twin; its suffix-less key hits A21, A21L and A021,
    # and the one nearest the apron centroid (0, 8) wins, flagged ambiguous.
    assert out.loc["A21X", ["stand_id", "match_source", "ambiguous"]].tolist() == [1, "nosuffix", 1]
    assert out["n_dep_train"].eq(10).all()


def test_duplicate_counts_share_a_normalised_key():
    dup = duplicate_counts(_refs())
    assert dup.loc[1] == 2 and dup.loc[6] == 2  # "A21" and "A021"
    assert dup.loc[2] == 1 and dup.loc[-4] == 1
    assert 5 not in dup.index and 7 not in dup.index  # gates are not counted
