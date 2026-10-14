"""Measured airport geometry: where the stand is, and how far it is to the runway.

`stand_runway_target_stats` learns the taxi distance from history, one
(stand, runway) pair at a time; these blocks measure it. Coordinates and
apron zones generalise to stands the training year never used, and a route
length is a number even for a pairing seen twice.

Three blocks, so each idea is measured on its own: the stand's position
(`stand_geometry`), the straight-line distance to the departure threshold
(`taxi_distance`), and the shortest taxiway route to it (`taxi_route`).

NaN means "not known" -- the stand was not found in OSM, the runway
designator names no threshold, or no route exists -- and the flag columns
say which. The tables are built by `python -m prc fetch-geometry`; every
one a block reads is declared as ``external`` so its digest keys the cache.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc import paths
from prc.features._geolib import (
    bearing_deg,
    haversine_m,
    load_routes,
    load_runway_ends,
    load_stand_matches,
    load_stands,
    norm_runway_ident,
)
from prc.features.registry import FeatureContext, feature

MATCH_SOURCES = ("exact", "norm", "nosuffix", "gate", "manual", "none")

STAND_GEOMETRY = (
    "geo_stand_east_m",
    "geo_stand_north_m",
    "geo_stand_arp_dist_m",
    "geo_stand_zone",
    "geo_stand_matched",
    "geo_stand_match_source",
)
TAXI_DISTANCE = (
    "geo_gc_dist_m",
    "geo_rwy_matched",
    "geo_rwy_length_m",
    "geo_rwy_threshold_bearing_deg",
)
TAXI_ROUTE = (
    "geo_route_dist_m",
    "geo_route_ratio",
    "geo_route_rwy_crossings",
    "geo_route_found",
)


def _departures(mvt: pd.DataFrame, ctx: FeatureContext) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """The departure rows, the airport as strings, and each row's matched OSM stand id (NaN if none)."""
    dep = mvt.loc[ctx.dep_index, ["apt", "STAND_mvt", "RUNWAY_mvt"]]
    apt = dep["apt"].astype("string")
    key = apt + "|" + dep["STAND_mvt"].astype("string").fillna("")
    m = load_stand_matches()
    m = m.loc[m["match_source"].ne("none")]  # ids of way-mapped stands are negative; the source says "found"
    lookup = pd.Series(m["stand_id"].to_numpy(dtype="float64"), index=(m["apt"] + "|" + m["stand_mvt"]).to_numpy())
    return dep, apt, key.map(lookup)


def zone_of(apt: pd.Series, stand: pd.Series) -> pd.Series:
    """The apron-zone label (``<ICAO>_zNN``) of any rows' (airport, ``STAND_mvt``); NaN when the stand is unmatched.

    The same lookup `stand_geometry` emits as ``geo_stand_zone``, exposed so
    arrival rows can be placed in a zone too (`prc.features.apron.zone_load`).
    """
    apt_s = apt.astype("string")
    key = apt_s + "|" + stand.astype("string").fillna("")
    m = load_stand_matches()
    m = m.loc[m["match_source"].ne("none")]
    lookup = pd.Series(m["stand_id"].to_numpy(dtype="float64"), index=(m["apt"] + "|" + m["stand_mvt"]).to_numpy())
    zone = key.map(lookup).map(load_stands().set_index("stand_id")["zone"])
    label = np.where(zone.notna(), apt_s.astype(object) + "_z" + zone.fillna(-1).astype(int).astype(str).str.zfill(2), None)
    return pd.Series(label, index=apt.index)


def _runway_idents(dep: pd.DataFrame) -> pd.Series:
    raw = dep["RUNWAY_mvt"].astype("string")
    table = {v: norm_runway_ident(v) for v in raw.dropna().unique()}
    return raw.map(table)


@feature(
    group="geometry",
    outputs=STAND_GEOMETRY,
    depends_on=("core_keys",),
    categorical=("geo_stand_zone", "geo_stand_match_source"),
    external=(paths.GEO_STANDS_FILE, paths.GEO_STAND_MATCHES_FILE),
    doc="Where the stand is: metres from the airport reference point, apron zone, and how it was found in OSM.",
)
def stand_geometry(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    dep, apt, stand_id = _departures(mvt, ctx)
    stands = load_stands().set_index("stand_id")
    matches = load_stand_matches()
    source_lookup = pd.Series(
        matches["match_source"].to_numpy(), index=(matches["apt"] + "|" + matches["stand_mvt"]).to_numpy()
    )
    key = apt + "|" + dep["STAND_mvt"].astype("string").fillna("")
    source = key.map(source_lookup).fillna("none")

    return pd.DataFrame(
        {
            "geo_stand_east_m": stand_id.map(stands["east_m"]).astype("float32"),
            "geo_stand_north_m": stand_id.map(stands["north_m"]).astype("float32"),
            "geo_stand_arp_dist_m": stand_id.map(stands["arp_dist_m"]).astype("float32"),
            "geo_stand_zone": zone_of(apt, dep["STAND_mvt"]).astype("category"),
            "geo_stand_matched": stand_id.notna().astype("int8"),
            "geo_stand_match_source": pd.Categorical(source.astype(object), categories=list(MATCH_SOURCES)),
        },
        index=ctx.dep_index,
    )


@feature(
    group="geometry",
    outputs=TAXI_DISTANCE,
    depends_on=("core_keys",),
    external=(paths.GEO_STANDS_FILE, paths.GEO_STAND_MATCHES_FILE, paths.GEO_RUNWAY_ENDS_FILE),
    doc="Straight-line distance from the stand to the departure runway threshold.",
)
def taxi_distance(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """The heuristic: great-circle metres from the stand to the threshold named by the designator.

    ``geo_rwy_threshold_bearing_deg`` is the bearing from the stand to the
    threshold relative to the runway heading, in (-180, 180]: near zero the
    stand sits behind the threshold (a short backtrack-free taxi), near 180
    it sits beyond it (the aircraft taxis the runway's length to get back).
    """
    dep, apt, stand_id = _departures(mvt, ctx)
    stands = load_stands().set_index("stand_id")
    ends = load_runway_ends()
    ends = ends.loc[ends["dropped_reason"].eq("")]
    ends = ends.set_index((ends["apt"] + "|" + ends["ident"]).to_numpy())
    rkey = apt + "|" + _runway_idents(dep).fillna("")

    s_lat, s_lon = stand_id.map(stands["lat"]).to_numpy(dtype="float64"), stand_id.map(stands["lon"]).to_numpy(dtype="float64")
    e_lat, e_lon = rkey.map(ends["lat"]).to_numpy(dtype="float64"), rkey.map(ends["lon"]).to_numpy(dtype="float64")
    heading = rkey.map(ends["heading_deg"]).to_numpy(dtype="float64")
    gc = haversine_m(s_lat, s_lon, e_lat, e_lon)
    rel = (bearing_deg(s_lat, s_lon, e_lat, e_lon) - heading + 180.0) % 360.0 - 180.0
    return pd.DataFrame(
        {
            "geo_gc_dist_m": gc.astype("float32"),
            "geo_rwy_matched": rkey.isin(ends.index).to_numpy().astype("int8"),
            "geo_rwy_length_m": rkey.map(ends["length_m"]).astype("float32"),
            "geo_rwy_threshold_bearing_deg": rel.astype("float32"),
        },
        index=ctx.dep_index,
    )


@feature(
    group="geometry",
    outputs=TAXI_ROUTE,
    depends_on=("core_keys",),
    external=(paths.GEO_STANDS_FILE, paths.GEO_STAND_MATCHES_FILE, paths.GEO_ROUTES_FILE),
    doc="Shortest taxiway route from the stand to the departure runway threshold, and the runways it crosses.",
)
def taxi_route(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    dep, apt, stand_id = _departures(mvt, ctx)
    routes = load_routes()
    routes = routes.set_index((routes["apt"] + "|" + routes["stand_id"].astype(str) + "|" + routes["runway_ident"]).to_numpy())
    key = apt + "|" + stand_id.fillna(-1).astype(int).astype(str) + "|" + _runway_idents(dep).fillna("")
    found = key.map(routes["route_found"]).fillna(0).astype("int8")
    return pd.DataFrame(
        {
            "geo_route_dist_m": key.map(routes["route_m"]).astype("float32"),
            "geo_route_ratio": key.map(routes["ratio"]).astype("float32"),
            "geo_route_rwy_crossings": key.map(routes["rwy_crossings"]).astype("float32"),
            "geo_route_found": found,
        },
        index=ctx.dep_index,
    )
