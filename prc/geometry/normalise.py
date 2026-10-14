"""Raw sources -> the five geometry tables.

Per airport: parse the OSM extract, build the taxiway graph, anchor the
OurAirports thresholds and the stands onto it, cluster the stands into apron
zones, route every stand to every runway end, and match the movement
frame's stand IDs to the OSM stands. Everything the feature blocks need is a
plain merge on the resulting tables.

The movement frame is read for its stand IDs and departure counts only --
never the target -- so the match table may be built over training and
ranking rows alike.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from prc import data, paths
from prc.features._geolib import clean_ref, haversine_m, local_east_north_m, norm_runway_ident
from prc.geometry import manual, match, routing, sources

RUNWAY_PENALTY = 3.0
STAND_SNAP_M = 150.0
GATE_SNAP_M = 250.0
RUNWAY_FALLBACK_M = 300.0
BRIDGE_GAP_M = 40.0
ZONES = 12

TABLES = {
    "airports": paths.GEO_AIRPORTS_FILE,
    "runway_ends": paths.GEO_RUNWAY_ENDS_FILE,
    "stands": paths.GEO_STANDS_FILE,
    "stand_matches": paths.GEO_STAND_MATCHES_FILE,
    "routes": paths.GEO_ROUTES_FILE,
}
END_COLS = ["apt", "ident", "pair_ref", "lat", "lon", "heading_deg", "length_m", "width_m", "surface",
            "osm_node", "snap_m", "dropped_reason"]
STAND_COLS = ["apt", "stand_id", "kind", "ref_raw", "ref_norm", "lat", "lon", "east_m", "north_m", "arp_dist_m",
              "zone", "entry_node", "anchor_node", "snap_m", "n_dup"]


def _movement_stands(mvt: pd.DataFrame) -> pd.DataFrame:
    """Distinct ``(apt, STAND_mvt)`` of departures with train/rank departure counts."""
    dep = mvt.loc[mvt["is_dep"] & mvt["STAND_mvt"].notna(), ["apt", "STAND_mvt", "split"]].astype("string")
    counts = dep.groupby(["apt", "STAND_mvt", "split"]).size().unstack("split", fill_value=0)
    counts = counts.rename(columns={"train": "n_dep_train", "rank": "n_dep_rank"}).reset_index()
    for col in ("n_dep_train", "n_dep_rank"):
        if col not in counts.columns:
            counts[col] = 0
    counts = counts.rename(columns={"STAND_mvt": "stand_mvt"})
    counts["apt"] = counts["apt"].astype(str)
    counts["stand_mvt"] = counts["stand_mvt"].astype(str)
    return counts[["apt", "stand_mvt", "n_dep_train", "n_dep_rank"]]


def _zones(stands: pd.DataFrame, k: int) -> np.ndarray:
    """Apron zones: k-means over stand positions, labelled by distance from the reference point.

    Fitted on the OSM parking positions only (gates and hand-placed stands
    then take the zone of their neighbourhood, so adding a stand by hand never
    moves the zones of the others -- a categorical the trained models read),
    relabelled so the numbering is a deterministic function of the data rather
    than of k-means' initialisation order.
    """
    from sklearn.cluster import KMeans

    xy = stands[["east_m", "north_m"]].to_numpy(dtype="float64")
    pp = stands["kind"].isin(["pp_node", "pp_way"]).to_numpy()
    fit_xy = xy[pp] if pp.any() else xy
    n_clusters = min(k, len(np.unique(fit_xy, axis=0))) if len(fit_xy) else 0
    if n_clusters < 1:
        return np.full(len(stands), -1, dtype="int16")
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(fit_xy)
    order = np.argsort(np.hypot(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1]), kind="stable")
    relabel = np.empty(n_clusters, dtype="int64")
    relabel[order] = np.arange(n_clusters)
    return relabel[km.predict(xy)].astype("int16")


def _dropped_ends(dropped: pd.DataFrame) -> pd.DataFrame:
    """The runways `filter_runways` rejected, as end rows carrying the reason (report only)."""
    rows = []
    for r in dropped.itertuples(index=False):
        for ident, lat, lon, heading in (
            (r.le_ident, r.le_latitude_deg, r.le_longitude_deg, r.le_heading_degT),
            (r.he_ident, r.he_latitude_deg, r.he_longitude_deg, r.he_heading_degT),
        ):
            rows.append({
                "apt": r.airport_ident, "ident": norm_runway_ident(ident) or clean_ref(ident), "pair_ref": "",
                "lat": pd.to_numeric(lat, errors="coerce"), "lon": pd.to_numeric(lon, errors="coerce"),
                "heading_deg": pd.to_numeric(heading, errors="coerce"),
                "length_m": pd.to_numeric(r.length_ft, errors="coerce") * sources.FT_TO_M,
                "width_m": pd.to_numeric(r.width_ft, errors="coerce") * sources.FT_TO_M,
                "surface": str(r.surface), "osm_node": -1, "snap_m": np.nan, "dropped_reason": r.reason,
            })
    return pd.DataFrame(rows, columns=END_COLS)


def normalise_geometry(stands_file: paths.Path | None = None) -> dict[str, paths.Path]:
    mvt = data.load_movements(columns=["apt", "is_dep", "split", "STAND_mvt"])
    icaos = data.airports(mvt)
    arps = sources.load_ourairports_airports(icaos).set_index("apt")
    kept, dropped = sources.filter_runways(sources.load_ourairports_runways(icaos))
    ends_all = sources.runway_ends(kept)
    mvt_stands = _movement_stands(mvt)
    if stands_file is not None:
        placed_rows = manual.load_manual_stands(stands_file, icaos)
        known = set(zip(mvt_stands["apt"], mvt_stands["stand_mvt"]))
        unused = [f"{a}:{s}" for a, s in zip(placed_rows["apt"], placed_rows["stand_mvt"]) if (a, s) not in known]
        print(f"hand-placed stands: {int(placed_rows['has_coords'].sum())} placed, "
              f"{int((~placed_rows['has_coords']).sum())} blanked, from {stands_file}")
        if unused:
            print(f"  not in the movement data (ignored): {unused}")
    else:
        placed_rows = pd.DataFrame(columns=manual.COLUMNS)

    airport_rows, end_frames, stand_frames, match_frames, route_frames = [], [_dropped_ends(dropped)], [], [], []
    for apt in icaos:
        raw = paths.geometry_raw_file("osm", f"{apt}.json")
        if not raw.exists():
            raise FileNotFoundError(f"{raw} missing; run `python -m prc fetch-geometry --only osm`")
        payload = json.loads(raw.read_text(encoding="utf-8"))
        lat0, lon0 = float(arps.loc[apt, "arp_lat"]), float(arps.loc[apt, "arp_lon"])

        nodes, ways = sources.parse_overpass(payload)
        g = routing.build_graph(nodes, ways, lat0=lat0, lon0=lon0, runway_penalty=RUNWAY_PENALTY)
        ends = routing.anchor_runway_ends(
            g, ends_all.loc[ends_all["apt"].eq(apt)], ways, lat0=lat0, lon0=lon0, fallback_m=RUNWAY_FALLBACK_M
        )
        n_bridges = routing.bridge_gaps(g, max_gap_m=BRIDGE_GAP_M)
        placed = manual.manual_stands(placed_rows, apt)
        stands = pd.concat([sources.osm_stands(nodes, ways), placed.drop(columns="stand_mvt")], ignore_index=True)
        stands = routing.anchor_stands(g, stands, lat0=lat0, lon0=lon0, snap_m=STAND_SNAP_M, gate_snap_m=GATE_SNAP_M)
        east, north = local_east_north_m(stands["lat"].to_numpy(), stands["lon"].to_numpy(), lat0, lon0)
        stands["east_m"], stands["north_m"] = east.astype("float32"), north.astype("float32")
        stands["arp_dist_m"] = haversine_m(stands["lat"], stands["lon"], lat0, lon0).astype("float32")
        stands["zone"] = _zones(stands, ZONES)
        stands["apt"] = apt

        # Hand-placed stands apply to their own STAND_mvt only, so they stay out
        # of the reference matching of everything else.
        refs = sources.explode_refs(stands.loc[stands["kind"].ne("manual")])
        stands["n_dup"] = stands["stand_id"].map(match.duplicate_counts(refs)).fillna(0).astype("int16")
        matches = match.match_stands(mvt_stands.loc[mvt_stands["apt"].eq(apt)], refs)
        blank = set(placed_rows.loc[placed_rows["apt"].eq(apt) & ~placed_rows["has_coords"], "stand_mvt"])
        matches = manual.apply_manual(matches, placed, blank)
        routes = routing.route_table(g, stands, ends, apt)

        stands["anchor_node"] = np.where(stands["graph_node"] >= 0, g.node_id[np.maximum(stands["graph_node"], 0)], -1)
        ends = ends.assign(dropped_reason="")
        n_pp_way = int(stands["kind"].eq("pp_way").sum())
        n_pp = int(stands["kind"].isin(["pp_node", "pp_way"]).sum())
        deps = matches["n_dep_train"] + matches["n_dep_rank"]
        matched_share = float(deps[matches["match_source"].ne("none")].sum() / max(deps.sum(), 1))
        airport_rows.append({
            "apt": apt, "arp_lat": lat0, "arp_lon": lon0, "name": str(arps.loc[apt, "name"]),
            "n_pp_nodes": n_pp - n_pp_way, "n_pp_ways": n_pp_way, "n_gates": int(stands["kind"].eq("gate").sum()),
            "n_manual": int(stands["kind"].eq("manual").sum()),
            "n_taxiway_ways": int(ways["aeroway"].isin(["taxiway", "taxilane"]).sum()),
            "n_runway_ways": int(ways["aeroway"].eq("runway").sum()),
            "n_graph_nodes": len(g.node_id), "n_graph_edges": int(g.weights.nnz // 2),
            "n_components": int(g.component.max() + 1) if len(g.component) else 0,
            "n_bridges": n_bridges,
            "main_share": float(g.keep.mean()) if len(g.keep) else 0.0,
            "n_dup_refs": int((stands["n_dup"] > 1).sum()),
            "n_stands_unsnapped": int((stands["graph_node"] < 0).sum()),
            "n_ends": len(ends), "n_ends_anchored": int((ends["graph_node"] >= 0).sum()),
            "matched_dep_share": matched_share,
        })
        manual_txt = f", {len(placed)} hand-placed" if len(placed) else ""
        print(
            f"  {apt}: {n_pp} stands ({n_pp_way} as ways), {airport_rows[-1]['n_gates']} gates{manual_txt}, "
            f"{airport_rows[-1]['n_taxiway_ways']} taxiway ways, {airport_rows[-1]['n_runway_ways']} runway ways | "
            f"graph {len(g.node_id):,} nodes {airport_rows[-1]['n_graph_edges']:,} edges, {n_bridges} bridges, "
            f"kept {g.keep.mean():.0%} | "
            f"ends anchored {airport_rows[-1]['n_ends_anchored']}/{len(ends)} | stands unsnapped "
            f"{airport_rows[-1]['n_stands_unsnapped']} | matched {matched_share:.1%} of departures"
        )
        end_frames.append(ends[END_COLS])
        stand_frames.append(stands[STAND_COLS])
        match_frames.append(matches)
        route_frames.append(routes)

    airports = pd.DataFrame(airport_rows)
    ends_out = pd.concat(end_frames, ignore_index=True).sort_values(["apt", "ident"]).reset_index(drop=True)
    ends_out["osm_node"] = ends_out["osm_node"].astype("int64")
    stands_out = pd.concat(stand_frames, ignore_index=True)
    matches_out = pd.concat(match_frames, ignore_index=True)
    routes_out = pd.concat(route_frames, ignore_index=True)

    paths.GEOMETRY.mkdir(parents=True, exist_ok=True)
    airports.to_parquet(TABLES["airports"], index=False)
    ends_out.to_parquet(TABLES["runway_ends"], index=False)
    stands_out.to_parquet(TABLES["stands"], index=False)
    matches_out.to_parquet(TABLES["stand_matches"], index=False)
    routes_out.to_parquet(TABLES["routes"], index=False)
    print(
        f"geometry: {len(stands_out):,} stands, {int(ends_out['dropped_reason'].eq('').sum())} runway ends, "
        f"{len(matches_out):,} movement stands matched, {len(routes_out):,} routes -> {paths.GEOMETRY}"
    )
    write_stand_report()
    return dict(TABLES)


def write_stand_report() -> paths.Path:
    """A CSV to look at: every stand ID in the movement data, how it matched, and where it is.

    One row per (airport, ``STAND_mvt``), most departures first. Unmatched
    stands have ``match_source == "none"`` and empty coordinates; the OSM
    reference shown is the one the match resolved to, so a ``norm`` or
    ``nosuffix`` row shows what was actually taken.
    """
    matches = pd.read_parquet(TABLES["stand_matches"])
    stands = pd.read_parquet(TABLES["stands"]).set_index("stand_id")
    out = matches.copy()
    out["n_dep"] = out["n_dep_train"] + out["n_dep_rank"]
    for col, src in (("osm_ref", "ref_raw"), ("osm_kind", "kind"), ("lat", "lat"), ("lon", "lon"),
                     ("zone", "zone"), ("snap_m", "snap_m")):
        out[col] = out["stand_id"].map(stands[src])
    out.loc[out["match_source"].eq("none"), ["stand_id", "osm_ref", "osm_kind", "lat", "lon", "zone", "snap_m"]] = None
    out["osm_id"] = [
        "" if pd.isna(s) else ("manual" if k == "manual" else (f"node/{int(s)}" if s > 0 else f"way/{-int(s)}"))
        for s, k in zip(out["stand_id"], out["osm_kind"])
    ]
    cols = ["apt", "stand_mvt", "n_dep", "n_dep_train", "n_dep_rank", "match_source", "ambiguous",
            "osm_ref", "osm_kind", "osm_id", "lat", "lon", "zone", "snap_m"]
    out = out[cols].sort_values(["apt", "n_dep"], ascending=[True, False])
    out.to_csv(paths.GEO_STAND_REPORT_FILE, index=False, float_format="%.6f")
    print(f"stand report: {len(out):,} movement stands -> {paths.GEO_STAND_REPORT_FILE}")
    return paths.GEO_STAND_REPORT_FILE
