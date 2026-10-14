"""Reading the raw sources: OurAirports CSVs and Overpass JSON. No network.

OurAirports gives every runway end an identifier, coordinates and a true
heading; OSM gives the stands (``aeroway=parking_position``, with the stand
name in ``ref``), the gates, and the taxiway network as ways sharing nodes at
their junctions. Both are kept as close to the source as possible here; the
judgement calls (which stand a movement ID means, how far the taxi is) live
in `match` and `routing`.
"""

from __future__ import annotations

import re

import pandas as pd

from prc import paths
from prc.features._geolib import clean_ref, norm_runway_ident, runway_idents, runway_key

FT_TO_M = 0.3048
STAND_KINDS = ("pp_node", "pp_way", "gate")


# --- OurAirports ---------------------------------------------------------------

def _read_csv(name: str) -> pd.DataFrame:
    path = paths.geometry_raw_file("ourairports", name)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python -m prc fetch-geometry --only ourairports`")
    # dtype=str + keep_default_na=False: "NA" is a valid string here, not a missing value.
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_ourairports_airports(icaos: list[str]) -> pd.DataFrame:
    """``apt, arp_lat, arp_lon, name`` for the airports in the data, from ``airports.csv``."""
    df = _read_csv("airports.csv")
    df = df.loc[df["ident"].isin(icaos)]
    out = pd.DataFrame(
        {
            "apt": df["ident"].to_numpy(),
            "arp_lat": pd.to_numeric(df["latitude_deg"], errors="coerce").to_numpy(),
            "arp_lon": pd.to_numeric(df["longitude_deg"], errors="coerce").to_numpy(),
            "name": df["name"].to_numpy(),
        }
    )
    missing = sorted(set(icaos) - set(out["apt"]))
    if missing or out[["arp_lat", "arp_lon"]].isna().any().any():
        raise ValueError(f"OurAirports has no reference point for {missing or 'some airports'}")
    return out.sort_values("apt").reset_index(drop=True)


def load_ourairports_runways(icaos: list[str]) -> pd.DataFrame:
    df = _read_csv("runways.csv")
    return df.loc[df["airport_ident"].isin(icaos)].reset_index(drop=True)


def filter_runways(df: pd.DataFrame, *, min_length_ft: int = 3000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(kept, dropped)``; ``dropped`` carries a ``reason`` column.

    Closed runways (LIRF 16C/34C, no coordinates), ends without coordinates,
    helicopter and grass strips (LFPG 08H/26H, LSZH 01H/19H: ``H`` idents,
    a few hundred metres long) and unparsable designators never take a
    departure in this data and would only pollute the ident match.
    """
    df = df.copy()
    coords = [pd.to_numeric(df[c], errors="coerce") for c in
              ("le_latitude_deg", "le_longitude_deg", "he_latitude_deg", "he_longitude_deg")]
    no_coords = pd.concat(coords, axis=1).isna().any(axis=1)
    length_ft = pd.to_numeric(df["length_ft"], errors="coerce")
    le, he = df["le_ident"].str.upper().str.strip(), df["he_ident"].str.upper().str.strip()
    heli = le.str.endswith("H") | he.str.endswith("H") | (length_ft < min_length_ft)
    unparsable = le.map(norm_runway_ident).isna() | he.map(norm_runway_ident).isna()

    closed = df["closed"].astype(str).str.strip().eq("1")
    # Most informative reason first: a closed runway is "closed" even when it
    # also lacks coordinates.
    reason = pd.Series("", index=df.index, dtype="object")
    for mask, label in ((closed, "closed"), (no_coords, "no_coords"), (heli, "heli"), (unparsable, "ident_unparsable")):
        reason[mask & reason.eq("")] = label

    df["reason"] = reason
    kept = df.loc[reason.eq("")].drop(columns="reason").reset_index(drop=True)
    dropped = df.loc[reason.ne("")].reset_index(drop=True)
    return kept, dropped


def runway_ends(kept: pd.DataFrame) -> pd.DataFrame:
    """Two rows per runway: the threshold each designator names.

    ``pair_ref`` ("07C/25C") identifies the physical runway on both sides --
    it is what an OSM runway way's ``ref`` normalises to.
    """
    rows = []
    for r in kept.itertuples(index=False):
        le, he = norm_runway_ident(r.le_ident), norm_runway_ident(r.he_ident)
        pair = runway_key(tuple(sorted((le, he))))
        common = {
            "apt": r.airport_ident,
            "pair_ref": pair,
            "length_m": float(pd.to_numeric(r.length_ft, errors="coerce")) * FT_TO_M,
            "width_m": float(pd.to_numeric(r.width_ft, errors="coerce")) * FT_TO_M,
            "surface": str(r.surface),
        }
        for ident, lat, lon, heading in (
            (le, r.le_latitude_deg, r.le_longitude_deg, r.le_heading_degT),
            (he, r.he_latitude_deg, r.he_longitude_deg, r.he_heading_degT),
        ):
            rows.append({
                "ident": ident, "lat": float(lat), "lon": float(lon),
                "heading_deg": float(pd.to_numeric(heading, errors="coerce")), **common,
            })
    cols = ["apt", "ident", "pair_ref", "lat", "lon", "heading_deg", "length_m", "width_m", "surface"]
    return pd.DataFrame(rows, columns=cols).sort_values(["apt", "ident"]).reset_index(drop=True)


# --- OpenStreetMap -------------------------------------------------------------

def parse_overpass(payload: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(nodes, ways)`` from an Overpass JSON answer.

    ``nodes``: ``id, lat, lon, aeroway, ref`` -- every node in the answer, the
    tagged ones (stands, gates, holding positions) with their tags, the rest
    with empty strings. ``ways``: ``id, aeroway, ref, nodes`` (a list of node
    ids, in way order).
    """
    node_rows, way_rows = [], []
    for el in payload.get("elements", []):
        tags = el.get("tags") or {}
        if el.get("type") == "node":
            node_rows.append({
                "id": int(el["id"]), "lat": float(el["lat"]), "lon": float(el["lon"]),
                "aeroway": str(tags.get("aeroway", "")), "ref": str(tags.get("ref", "")),
            })
        elif el.get("type") == "way":
            way_rows.append({
                "id": int(el["id"]), "aeroway": str(tags.get("aeroway", "")), "ref": str(tags.get("ref", "")),
                "nodes": [int(n) for n in el.get("nodes", [])],
            })
    nodes = pd.DataFrame(node_rows, columns=["id", "lat", "lon", "aeroway", "ref"])
    # A tagged node comes out twice (once from `out body`, once from `out skel`);
    # keep the tagged copy, which `out body` prints first.
    nodes = nodes.drop_duplicates("id", keep="first").reset_index(drop=True)
    ways = pd.DataFrame(way_rows, columns=["id", "aeroway", "ref", "nodes"])
    ways = ways.drop_duplicates("id", keep="first").reset_index(drop=True)
    return nodes, ways


def osm_stands(nodes: pd.DataFrame, ways: pd.DataFrame) -> pd.DataFrame:
    """Every stand-like object: parking positions (as nodes or ways) and gates.

    ``stand_id`` is the OSM id for nodes and the *negated* OSM id for ways, so
    the two id spaces cannot collide. A parking-position way runs from the
    taxiway to the nose-wheel position, so its last node is where the aircraft
    stands (``lat, lon``) and its first node is where the taxi starts
    (``entry_node``); for a node both are the node itself.
    """
    rows = []
    by_id = nodes.set_index("id")
    pp_nodes = nodes.loc[nodes["aeroway"].eq("parking_position")]
    for r in pp_nodes.itertuples(index=False):
        rows.append({"stand_id": int(r.id), "kind": "pp_node", "ref_raw": r.ref, "lat": r.lat, "lon": r.lon,
                     "entry_node": int(r.id)})
    for r in nodes.loc[nodes["aeroway"].eq("gate")].itertuples(index=False):
        rows.append({"stand_id": int(r.id), "kind": "gate", "ref_raw": r.ref, "lat": r.lat, "lon": r.lon,
                     "entry_node": int(r.id)})
    for w in ways.loc[ways["aeroway"].eq("parking_position")].itertuples(index=False):
        ids = [n for n in w.nodes if n in by_id.index]
        if not ids:
            continue
        last = by_id.loc[ids[-1]]
        rows.append({"stand_id": -int(w.id), "kind": "pp_way", "ref_raw": w.ref,
                     "lat": float(last["lat"]), "lon": float(last["lon"]), "entry_node": int(ids[0])})
    out = pd.DataFrame(rows, columns=["stand_id", "kind", "ref_raw", "lat", "lon", "entry_node"])
    out["ref_norm"] = out["ref_raw"].map(clean_ref)
    return out.sort_values(["kind", "stand_id"]).reset_index(drop=True)


def split_ref(raw: str) -> list[str]:
    """The stand designations named by one OSM ``ref``.

    A position serving several stands is mapped once with all of them in the
    ``ref`` ("A21;A21L", "A26 A28", "1/1A"), and each must match on its own.
    But a space can also sit inside one designation ("V 117"), so whitespace
    splits only when every part carries a digit; otherwise it is removed.
    """
    out: list[str] = []
    for part in re.split(r"[;,/]+", str(raw)):
        tokens = part.split()
        if not tokens:
            continue
        if len(tokens) > 1 and all(any(ch.isdigit() for ch in t) for t in tokens):
            out.extend(tokens)
        else:
            out.append("".join(tokens))
    return out


def explode_refs(stands: pd.DataFrame) -> pd.DataFrame:
    """One row per stand designation (see `split_ref`)."""
    out = stands.assign(ref=stands["ref_raw"].map(split_ref)).explode("ref")
    out["ref"] = out["ref"].fillna("").astype(str).str.strip()
    return out.loc[out["ref"].ne("")].reset_index(drop=True)


def runway_ways(ways: pd.DataFrame) -> pd.DataFrame:
    """Runway ways with their normalised designators and physical-runway key."""
    rw = ways.loc[ways["aeroway"].eq("runway")].copy()
    rw["idents"] = rw["ref"].map(runway_idents)
    rw["runway_key"] = rw["idents"].map(runway_key)
    return rw.reset_index(drop=True)
