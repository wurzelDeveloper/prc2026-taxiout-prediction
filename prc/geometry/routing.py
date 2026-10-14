"""Shortest taxi routes over the OSM taxiway network.

The graph is the one OSM draws: ways share a node where they meet, so
consecutive nodes of every taxiway, taxilane, runway, parking-position and
holding-position way become edges weighted by their length in metres.
Runway edges are penalised (not removed) so a route uses a runway only where
no taxiway exists -- backtracks happen, and some aprons connect through a
runway crossing -- while the reported length is always the true metres.

Anchors: a runway end is the OurAirports threshold snapped onto the OSM way
that carries that designator; a stand is its own node when that node is
part of the network, otherwise the nearest network node within a short
radius. Nodes in a component that holds no runway end (a heliport in the
box, a fenced-off apron) are never used as anchors.

One Dijkstra per runway end (`scipy.sparse.csgraph`) answers every stand at
once, and the predecessor tree gives the true length and the runways touched
along each route with one accumulating pass in distance order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from prc.features._geolib import haversine_m, local_east_north_m

GRAPH_WAYS = ("taxiway", "taxilane", "runway", "parking_position", "holding_position")
MIN_EDGE_M = 0.01  # two nodes at one coordinate must still be an edge, and csgraph reads 0 as "none"


@dataclass
class TaxiGraph:
    node_id: np.ndarray
    """OSM node ids; the position in this array is the graph index."""
    lat: np.ndarray
    lon: np.ndarray
    east: np.ndarray
    north: np.ndarray
    weights: csr_matrix
    """Symmetric edge weights in penalised metres -- what Dijkstra minimises."""
    metres: csr_matrix
    """The same sparsity with true metres -- what is reported."""
    node_runways: dict[int, frozenset[str]]
    """Graph index -> keys of the physical runways whose ways contain the node."""
    degree: np.ndarray
    component: np.ndarray
    keep: np.ndarray
    """Nodes in a component that contains at least one runway anchor (set by `anchor_runway_ends`)."""

    def index_of(self, osm_ids) -> np.ndarray:
        """Graph indices of OSM node ids, -1 where unknown."""
        lookup = pd.Index(self.node_id)
        return lookup.get_indexer(np.asarray(osm_ids, dtype="int64"))


def build_graph(nodes: pd.DataFrame, ways: pd.DataFrame, *, lat0: float, lon0: float,
                runway_penalty: float = 3.0) -> TaxiGraph:
    from prc.geometry.sources import runway_ways

    node_id = nodes["id"].to_numpy(dtype="int64")
    lat, lon = nodes["lat"].to_numpy(dtype="float64"), nodes["lon"].to_numpy(dtype="float64")
    east, north = local_east_north_m(lat, lon, lat0, lon0)
    index = pd.Index(node_id)

    us, vs, kinds = [], [], []
    for w in ways.loc[ways["aeroway"].isin(GRAPH_WAYS)].itertuples(index=False):
        idx = index.get_indexer(np.asarray(w.nodes, dtype="int64"))
        idx = idx[idx >= 0]
        if len(idx) < 2:  # a one-node way (or nodes outside the box) makes no edge
            continue
        us.append(idx[:-1])
        vs.append(idx[1:])
        kinds.append(np.full(len(idx) - 1, w.aeroway == "runway"))

    n = len(node_id)
    if us:
        u, v, on_rwy = np.concatenate(us), np.concatenate(vs), np.concatenate(kinds)
        length = np.maximum(haversine_m(lat[u], lon[u], lat[v], lon[v]), MIN_EDGE_M)
        edges = pd.DataFrame({"u": np.minimum(u, v), "v": np.maximum(u, v), "m": length,
                              "w": length * np.where(on_rwy, runway_penalty, 1.0)})
        edges = edges.loc[edges["u"] != edges["v"]]
        # Parallel edges (a taxiway and a taxilane over the same nodes): keep the cheapest.
        edges = edges.groupby(["u", "v"], as_index=False).min()
        u, v = edges["u"].to_numpy(), edges["v"].to_numpy()
        rows, cols = np.concatenate([u, v]), np.concatenate([v, u])
        metres = csr_matrix((np.tile(edges["m"].to_numpy(), 2), (rows, cols)), shape=(n, n))
        weights = csr_matrix((np.tile(edges["w"].to_numpy(), 2), (rows, cols)), shape=(n, n))
    else:
        metres = csr_matrix((n, n), dtype="float64")
        weights = csr_matrix((n, n), dtype="float64")

    node_runways: dict[int, set[str]] = {}
    for w in runway_ways(ways).itertuples(index=False):
        if not w.runway_key:
            continue
        for i in index.get_indexer(np.asarray(w.nodes, dtype="int64")):
            if i >= 0:
                node_runways.setdefault(int(i), set()).add(w.runway_key)

    _, component = connected_components(weights, directed=False)
    return TaxiGraph(
        node_id=node_id, lat=lat, lon=lon, east=east, north=north,
        weights=weights, metres=metres,
        node_runways={k: frozenset(v) for k, v in node_runways.items()},
        degree=np.diff(weights.indptr), component=component,
        keep=np.ones(n, dtype=bool),
    )


def bridge_gaps(g: TaxiGraph, *, max_gap_m: float = 40.0, rounds: int = 3) -> int:
    """Join components that come within ``max_gap_m`` of the kept network; returns the bridges added.

    OSM apron networks are sometimes drawn without sharing a node at one
    junction, which leaves a whole apron's taxilanes as an island a few metres
    from the taxiway. A synthetic edge across the smallest such gap restores the
    connection; the gap length counts as taxi distance. Repeated, because an
    island may only touch the network through another island.
    """
    added = 0
    n = len(g.node_id)
    for _ in range(rounds):
        kept = np.flatnonzero(g.keep & (g.degree > 0))
        if len(kept) == 0:
            break
        tree = cKDTree(np.column_stack([g.east[kept], g.north[kept]]))
        us, vs, lengths = [], [], []
        for comp in np.unique(g.component[~g.keep & (g.degree > 0)]):
            members = np.flatnonzero(g.component == comp)
            d, i = tree.query(np.column_stack([g.east[members], g.north[members]]), k=1, distance_upper_bound=max_gap_m)
            if not np.isfinite(d).any():
                continue
            k = int(np.argmin(d))
            us.append(members[k])
            vs.append(kept[i[k]])
            lengths.append(max(float(d[k]), MIN_EDGE_M))
        if not us:
            break
        rows = np.concatenate([us, vs])
        cols = np.concatenate([vs, us])
        vals = np.tile(np.asarray(lengths), 2)
        g.metres = (g.metres + csr_matrix((vals, (rows, cols)), shape=(n, n))).tocsr()
        g.weights = (g.weights + csr_matrix((vals, (rows, cols)), shape=(n, n))).tocsr()
        g.degree = np.diff(g.weights.indptr)
        _, g.component = connected_components(g.weights, directed=False)
        g.keep = np.isin(g.component, np.unique(g.component[kept]))
        added += len(us)
    return added


def nearest_node(g: TaxiGraph, east: float, north: float, *, max_m: float,
                 candidates: np.ndarray | None = None) -> tuple[int, float]:
    """``(graph index, metres)`` of the nearest candidate node within ``max_m``; ``(-1, nan)`` if none."""
    cand = np.arange(len(g.node_id)) if candidates is None else np.asarray(candidates)
    if len(cand) == 0:
        return -1, float("nan")
    tree = cKDTree(np.column_stack([g.east[cand], g.north[cand]]))
    d, i = tree.query((east, north), k=1, distance_upper_bound=max_m)
    if not np.isfinite(d):
        return -1, float("nan")
    return int(cand[i]), float(d)


def anchor_runway_ends(g: TaxiGraph, ends: pd.DataFrame, ways: pd.DataFrame, *, lat0: float, lon0: float,
                       fallback_m: float = 300.0, own_way_max_m: float = 1000.0) -> pd.DataFrame:
    """Snap each threshold onto the OSM runway way that carries its designator; set ``g.keep``.

    Adds ``graph_node`` (index, -1 if unanchored), ``osm_node`` and ``snap_m``.
    A threshold whose designator no OSM runway way names -- or whose way lies
    more than ``own_way_max_m`` away, which means the two sources disagree --
    falls back to the nearest network node within ``fallback_m``.
    """
    from prc.geometry.sources import runway_ways

    rw = runway_ways(ways)
    by_ident: dict[str, set[int]] = {}
    for w in rw.itertuples(index=False):
        idx = g.index_of(w.nodes)
        for ident in w.idents:
            by_ident.setdefault(ident, set()).update(int(i) for i in idx if i >= 0)

    ends = ends.copy()
    east, north = local_east_north_m(ends["lat"].to_numpy(), ends["lon"].to_numpy(), lat0, lon0)
    graph_node, snap = np.full(len(ends), -1), np.full(len(ends), np.nan)
    connected = np.flatnonzero(g.degree > 0)
    for k, r in enumerate(ends.itertuples(index=False)):
        cand = np.array(sorted(by_ident.get(r.ident, ())), dtype="int64")
        cand = cand[g.degree[cand] > 0] if len(cand) else cand
        if len(cand):
            graph_node[k], snap[k] = nearest_node(g, east[k], north[k], max_m=own_way_max_m, candidates=cand)
        if graph_node[k] < 0:
            graph_node[k], snap[k] = nearest_node(g, east[k], north[k], max_m=fallback_m, candidates=connected)
    ends["graph_node"] = graph_node
    ends["osm_node"] = np.where(graph_node >= 0, g.node_id[np.maximum(graph_node, 0)], -1)
    ends["snap_m"] = snap.astype("float32")

    anchored = graph_node[graph_node >= 0]
    if len(anchored):
        g.keep = np.isin(g.component, np.unique(g.component[anchored]))
    return ends


def anchor_stands(g: TaxiGraph, stands: pd.DataFrame, *, lat0: float, lon0: float,
                  snap_m: float = 150.0, gate_snap_m: float = 250.0) -> pd.DataFrame:
    """Where each stand joins the network: its entry node if that is connected, else a nearby node.

    Adds ``graph_node`` and ``snap_m``. A parking-position way's nodes are in
    the graph but form a component of their own unless the way touches a
    taxiway; ``g.keep`` decides, so a detached way snaps like a lone node.
    """
    stands = stands.copy()
    n = len(stands)
    entry = g.index_of(stands["entry_node"].to_numpy())
    usable = np.flatnonzero(g.keep & (g.degree > 0))
    east, north = local_east_north_m(stands["lat"].to_numpy(), stands["lon"].to_numpy(), lat0, lon0)
    graph_node, snap = np.full(n, -1), np.full(n, np.nan)
    if n == 0:
        stands["graph_node"], stands["snap_m"] = graph_node, snap.astype("float32")
        return stands

    known = entry >= 0
    direct = known & g.keep[np.maximum(entry, 0)] & (g.degree[np.maximum(entry, 0)] > 0)
    graph_node[direct], snap[direct] = entry[direct], 0.0

    # Everything else snaps from its entry point (the node's own position, or
    # the taxiway end of a parking-position way) to the nearest usable node.
    todo = np.flatnonzero(~direct)
    if len(todo) and len(usable):
        qe = np.where(known[todo], g.east[np.maximum(entry[todo], 0)], east[todo])
        qn = np.where(known[todo], g.north[np.maximum(entry[todo], 0)], north[todo])
        # Gates sit at the terminal wall and hand-placed stands may be a rough
        # position, so both get the wider radius.
        radius = np.where(np.isin(stands["kind"].to_numpy()[todo], ("gate", "manual")), gate_snap_m, snap_m)
        tree = cKDTree(np.column_stack([g.east[usable], g.north[usable]]))
        d, i = tree.query(np.column_stack([qe, qn]), k=1, distance_upper_bound=float(radius.max()))
        ok = np.isfinite(d) & (d <= radius)
        graph_node[todo[ok]] = usable[i[ok]]
        snap[todo[ok]] = d[ok]
    stands["graph_node"] = graph_node
    stands["snap_m"] = snap.astype("float32")
    return stands


def _tree_accumulate(g: TaxiGraph, dist: np.ndarray, pred: np.ndarray, source: int):
    """True metres and runways touched from ``source`` to every reached node, along the Dijkstra tree."""
    n = len(dist)
    reached = np.flatnonzero(np.isfinite(dist))
    order = reached[np.argsort(dist[reached], kind="stable")]
    true = np.full(n, np.nan)
    touched: list[frozenset[str] | None] = [None] * n
    empty = frozenset()
    true[source] = 0.0
    touched[source] = g.node_runways.get(source, empty)
    children = order[order != source]
    if len(children):
        parents = pred[children]
        step = np.asarray(g.metres[parents, children]).ravel()
        for v, p, s in zip(children.tolist(), parents.tolist(), step.tolist()):
            true[v] = true[p] + s
            own = g.node_runways.get(v)
            touched[v] = touched[p] if own is None else (touched[p] | own)
    return true, touched


def route_table(g: TaxiGraph, stands: pd.DataFrame, ends: pd.DataFrame, apt: str) -> pd.DataFrame:
    """Every (stand, runway end) pair: great-circle metres, route metres, ratio, runway crossings.

    ``gc_m`` runs from the stand's own position to the threshold itself,
    unsnapped, so it is meaningful even where routing failed.
    """
    cols = ["apt", "stand_id", "runway_ident", "gc_m", "route_m", "ratio", "rwy_crossings", "route_found"]
    if stands.empty or ends.empty:
        return pd.DataFrame(columns=cols)

    s_lat, s_lon = stands["lat"].to_numpy(), stands["lon"].to_numpy()
    s_node = stands["graph_node"].to_numpy()
    sources = np.unique(ends["graph_node"].to_numpy()[ends["graph_node"].to_numpy() >= 0])
    if len(sources):
        dist, pred = dijkstra(g.weights, directed=False, indices=sources, return_predecessors=True)
    frames = []
    for e in ends.itertuples(index=False):
        gc = haversine_m(s_lat, s_lon, e.lat, e.lon)
        route = np.full(len(stands), np.nan)
        crossings = np.zeros(len(stands), dtype="int16")
        found = np.zeros(len(stands), dtype="int8")
        if e.graph_node >= 0:
            k = int(np.searchsorted(sources, e.graph_node))
            true, touched = _tree_accumulate(g, dist[k], pred[k], int(e.graph_node))
            ok = (s_node >= 0) & np.isfinite(true[np.maximum(s_node, 0)])
            route[ok] = true[s_node[ok]]
            found[ok] = 1
            own = e.pair_ref
            for i in np.flatnonzero(ok):
                t = touched[s_node[i]]
                crossings[i] = len(t) - (own in t) if t else 0
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(gc >= 50.0, route / gc, np.nan)
        frames.append(pd.DataFrame({
            "apt": apt, "stand_id": stands["stand_id"].to_numpy(), "runway_ident": e.ident,
            "gc_m": gc.astype("float32"), "route_m": route.astype("float32"), "ratio": ratio.astype("float32"),
            "rwy_crossings": crossings, "route_found": found,
        }))
    return pd.concat(frames, ignore_index=True)[cols]
