"""Taxiway-graph routing on a synthetic airport. No network, no data.

Metric coordinates near (0, 0) are converted to degrees, where one metre is
about 9e-6 degrees, so the haversine lengths below are planar to well
within a millimetre.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from prc.geometry.routing import anchor_runway_ends, anchor_stands, bridge_gaps, build_graph, route_table
from prc.geometry.sources import osm_stands

DEG_PER_M = 180.0 / (math.pi * 6_371_000.0)

# node id -> (x metres east, y metres north)
NODES = {
    1: (0, 0),        # stand A1: a parking_position node on the taxiway
    2: (100, 0),
    3: (200, 0),      # taxiway meets runway 09/27 here
    4: (200, 100),    # runway 18/36 threshold "18"
    5: (200, 300),    # runway 18/36 threshold "36"
    6: (700, 700),    # stand ISO: a parking_position node on nothing
    7: (200, -100),
    8: (100, 320),    # detour taxiway node
    13: (100, -30),   # stand W: a parking_position way, entry node ...
    14: (100, -60),   # ... and nose-wheel node
}


def _nodes(tagged: dict[int, tuple[str, str]]):
    rows = []
    for nid, (x, y) in NODES.items():
        aeroway, ref = tagged.get(nid, ("", ""))
        rows.append({"id": nid, "lat": y * DEG_PER_M, "lon": x * DEG_PER_M, "aeroway": aeroway, "ref": ref})
    return pd.DataFrame(rows)


def _ways(detour: bool):
    rows = [
        {"id": 100, "aeroway": "taxiway", "ref": "A", "nodes": [1, 2, 3]},
        {"id": 101, "aeroway": "taxiway", "ref": "B", "nodes": [3, 4]},
        {"id": 102, "aeroway": "runway", "ref": "18/36", "nodes": [4, 5]},
        {"id": 103, "aeroway": "runway", "ref": "09/27", "nodes": [7, 3]},
        {"id": 104, "aeroway": "taxiway", "ref": "one-node", "nodes": [6]},
        {"id": 106, "aeroway": "parking_position", "ref": "W", "nodes": [13, 14]},
    ]
    if detour:
        rows.append({"id": 105, "aeroway": "taxiway", "ref": "C", "nodes": [2, 8, 5]})
    return pd.DataFrame(rows)


def _ends():
    return pd.DataFrame([
        {"apt": "TEST", "ident": "36", "pair_ref": "18/36", "lat": 300 * DEG_PER_M, "lon": 200 * DEG_PER_M,
         "heading_deg": 360.0, "length_m": 200.0, "width_m": 45.0, "surface": "ASP"},
        {"apt": "TEST", "ident": "18", "pair_ref": "18/36", "lat": 100 * DEG_PER_M, "lon": 200 * DEG_PER_M,
         "heading_deg": 180.0, "length_m": 200.0, "width_m": 45.0, "surface": "ASP"},
        {"apt": "TEST", "ident": "27", "pair_ref": "09/27", "lat": 0.0, "lon": 5000 * DEG_PER_M,  # nowhere near the network
         "heading_deg": 270.0, "length_m": 100.0, "width_m": 45.0, "surface": "ASP"},
    ])


def _airport(detour: bool):
    nodes = _nodes({1: ("parking_position", "A1"), 6: ("parking_position", "ISO")})
    ways = _ways(detour)
    g = build_graph(nodes, ways, lat0=0.0, lon0=0.0, runway_penalty=3.0)
    ends = anchor_runway_ends(g, _ends(), ways, lat0=0.0, lon0=0.0, fallback_m=300.0)
    stands = anchor_stands(g, osm_stands(nodes, ways), lat0=0.0, lon0=0.0, snap_m=150.0, gate_snap_m=250.0)
    routes = route_table(g, stands, ends, "TEST").set_index(["stand_id", "runway_ident"])
    return g, ends.set_index("ident"), stands.set_index("stand_id"), routes


def test_graph_edges_components_and_keep():
    g, ends, stands, _ = _airport(detour=True)
    i = {nid: k for k, nid in enumerate(g.node_id)}
    assert g.degree[i[6]] == 0  # a one-node way makes no edge
    assert g.metres[i[1], i[2]] == pytest.approx(100.0, abs=1e-3)
    assert g.weights[i[4], i[5]] == pytest.approx(600.0, abs=1e-3)  # runway edge, penalised x3
    assert g.metres[i[4], i[5]] == pytest.approx(200.0, abs=1e-3)   # ... but true metres are kept
    assert not g.keep[i[6]] and g.keep[i[1]]                          # only components with a runway end
    assert g.node_runways[i[3]] == {"09/27"} and g.node_runways[i[4]] == {"18/36"}


def test_runway_ends_snap_onto_their_own_way_or_fall_back():
    g, ends, _, _ = _airport(detour=True)
    assert ends.loc["36", "osm_node"] == 5 and ends.loc["36", "snap_m"] == pytest.approx(0.0, abs=1e-3)
    assert ends.loc["18", "osm_node"] == 4
    assert ends.loc["27", "osm_node"] == -1  # 5 km from anything, no way carries "27" near it


def test_stand_anchoring():
    _, _, stands, _ = _airport(detour=True)
    assert stands.loc[1, "snap_m"] == 0.0                     # its own node is on the taxiway
    assert stands.loc[6, "graph_node"] == -1                   # isolated, nothing within 150 m
    assert stands.loc[-106, "snap_m"] == pytest.approx(30.0, abs=1e-3)  # detached way: entry node snaps to node 2
    assert stands.loc[-106, "lat"] == pytest.approx(-60 * DEG_PER_M)     # position is the nose-wheel end


def test_penalty_steers_but_true_metres_are_reported():
    _, _, _, routes = _airport(detour=True)
    r = routes.loc[(1, "36")]
    # 1-2-3-4-5 is 500 m but 900 penalised; the detour 1-2-8-5 is 521.98 m either way.
    assert r["route_found"] == 1
    assert r["route_m"] == pytest.approx(100 + 320 + math.hypot(100, 20), abs=0.05)
    assert r["rwy_crossings"] == 0
    assert r["gc_m"] == pytest.approx(math.hypot(200, 300), abs=0.05)
    assert r["ratio"] == pytest.approx(r["route_m"] / r["gc_m"], rel=1e-5)


def test_forced_runway_crossing_is_counted():
    _, _, _, routes = _airport(detour=False)
    r = routes.loc[(1, "36")]
    assert r["route_found"] == 1 and r["route_m"] == pytest.approx(500.0, abs=0.05)
    assert r["rwy_crossings"] == 1  # node 3 belongs to runway 09/27; 18/36 is the departure runway itself


def test_unroutable_pairs_are_flagged_not_guessed():
    _, _, _, routes = _airport(detour=True)
    iso = routes.loc[(6, "36")]
    assert iso["route_found"] == 0 and np.isnan(iso["route_m"]) and np.isnan(iso["ratio"])
    assert iso["gc_m"] == pytest.approx(math.hypot(500, 400), abs=0.05)  # still measured
    far = routes.loc[(1, "27")]
    assert far["route_found"] == 0 and np.isnan(far["route_m"])


def test_small_gaps_are_bridged_but_islands_are_not():
    """An apron drawn 10 m short of the taxiway joins the network; the stand 700 m out does not."""
    nodes = _nodes({1: ("parking_position", "A1"), 6: ("parking_position", "ISO")})
    extra = pd.DataFrame([
        {"id": 22, "lat": 0.0, "lon": 210 * DEG_PER_M, "aeroway": "", "ref": ""},   # (210, 0): 10 m east of node 3
        {"id": 20, "lat": 0.0, "lon": 310 * DEG_PER_M, "aeroway": "", "ref": ""},   # (310, 0)
        {"id": 21, "lat": 0.0, "lon": 500 * DEG_PER_M, "aeroway": "", "ref": ""},   # (500, 0)
    ])
    nodes = pd.concat([nodes, extra], ignore_index=True)
    ways = pd.concat([_ways(detour=True), pd.DataFrame([
        {"id": 200, "aeroway": "taxilane", "ref": "island", "nodes": [22, 20, 21]},
    ])], ignore_index=True)
    g = build_graph(nodes, ways, lat0=0.0, lon0=0.0)
    anchor_runway_ends(g, _ends(), ways, lat0=0.0, lon0=0.0)
    i = {nid: k for k, nid in enumerate(g.node_id)}
    assert not g.keep[i[22]] and not g.keep[i[13]]
    # Two islands within reach: the apron (10 m from node 3) and the detached
    # parking-position way (its entry node 13 is 30 m from node 2).
    assert bridge_gaps(g, max_gap_m=40.0) == 2
    assert g.keep[i[22]] and g.keep[i[21]] and g.keep[i[13]] and not g.keep[i[6]]
    assert g.metres[i[13], i[2]] == pytest.approx(30.0, abs=0.01)
    assert g.metres[i[22], i[3]] == pytest.approx(10.0, abs=0.01)
    assert g.degree[i[3]] == 4  # 2, 4 and 7 before, plus the bridge


def test_detached_parking_way_routes_from_its_snapped_entry():
    _, _, _, routes = _airport(detour=True)
    r = routes.loc[(-106, "36")]
    assert r["route_found"] == 1
    assert r["route_m"] == pytest.approx(320 + math.hypot(100, 20), abs=0.05)  # from node 2 via the detour
    assert r["gc_m"] == pytest.approx(math.hypot(100, 360), abs=0.05)         # from the nose wheel to the threshold
