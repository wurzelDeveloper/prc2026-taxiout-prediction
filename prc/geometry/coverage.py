"""`fetch-geometry --check`: how much of the data the geometry tables actually cover.

Answered from what is on disk, with no network:

* per airport -- what OSM holds and how well the taxiway graph hangs together;
* per airport and split -- the share of departures whose stand was matched,
  by which rule, and the unmatched stands that carry the most departures;
* per airport -- the runway designators the movements use against the ends
  OurAirports knows, and the ends that were dropped and why;
* per airport and split -- the share of departures with a routed taxi
  distance, and what those routes look like.

The verdict line is the gate for the feature work: a "weak" airport still
gets features, the flags carry the gaps, but its numbers will not move much.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from prc import data, paths
from prc.features._geolib import norm_runway_ident
from prc.geometry.match import MATCH_SOURCES
from prc.geometry.normalise import TABLES, write_stand_report

MATCH_GATE, ROUTE_GATE = 0.90, 0.85


def _pct(frame: pd.DataFrame, by: list[str], flag: str) -> pd.DataFrame:
    return (100 * frame.groupby(by, observed=True)[flag].mean()).unstack(by[-1]).round(1)


def report() -> str:
    missing = [name for name, path in TABLES.items() if not path.exists()]
    if missing:
        return f"geometry: tables missing {missing} -- run `python -m prc fetch-geometry`"
    airports = pd.read_parquet(TABLES["airports"])
    ends = pd.read_parquet(TABLES["runway_ends"])
    stands = pd.read_parquet(TABLES["stands"])
    matches = pd.read_parquet(TABLES["stand_matches"])
    routes = pd.read_parquet(TABLES["routes"])

    mvt = data.load_movements(columns=["apt", "is_dep", "split", "STAND_mvt", "RUNWAY_mvt"])
    dep = mvt.loc[mvt["is_dep"]].copy()
    dep["apt"] = dep["apt"].astype(str)
    dep["split"] = dep["split"].astype(str)
    dep["stand_mvt"] = dep["STAND_mvt"].astype("string").fillna("").astype(str)
    key = dep["apt"] + "|" + dep["stand_mvt"]
    mkey = matches["apt"] + "|" + matches["stand_mvt"]
    dep["source"] = key.map(pd.Series(matches["match_source"].to_numpy(), index=mkey)).fillna("none")
    dep["stand_id"] = key.map(pd.Series(matches["stand_id"].to_numpy(), index=mkey)).fillna(-1).astype("int64")
    dep["matched"] = dep["source"].ne("none")

    # --- OSM content -----------------------------------------------------------
    apt_table = airports.set_index("apt")[[
        "n_pp_nodes", "n_pp_ways", "n_gates", "n_manual", "n_taxiway_ways", "n_runway_ways", "n_graph_nodes",
        "n_graph_edges", "n_components", "n_bridges", "main_share", "n_dup_refs", "n_stands_unsnapped",
        "n_ends_anchored", "n_ends",
    ]].copy()
    manifest = json.loads(paths.GEOMETRY_MANIFEST.read_text()) if paths.GEOMETRY_MANIFEST.exists() else {}
    stands_file = (manifest.get("stands_file") or {}).get("path", "none")
    apt_table["main_share"] = (100 * apt_table["main_share"]).round(1).rename("main_share_%")

    # --- stand matching ----------------------------------------------------------
    by_source = (
        dep.groupby(["apt", "split", "source"], observed=True).size()
        .groupby(level=[0, 1]).transform(lambda s: 100 * s / s.sum())
        .unstack("source").reindex(columns=list(MATCH_SOURCES), fill_value=0.0).round(1)
    )
    by_source["matched"] = (100 - by_source["none"]).round(1)
    unmatched = dep.loc[~dep["matched"] & dep["stand_mvt"].ne("")]
    top_unmatched = []
    for apt, g in unmatched.groupby("apt", observed=True):
        counts = g["stand_mvt"].value_counts().head(10)
        top_unmatched.append(f"  {apt}: " + ", ".join(f"{s} ({n:,})" for s, n in counts.items()))
    distinct = matches.assign(ok=matches["match_source"].ne("none")).groupby("apt")["ok"].agg(["sum", "size"])
    distinct = distinct.rename(columns={"sum": "stands_matched", "size": "stands_total"})

    # --- runways -------------------------------------------------------------------
    raw_rwy = dep["RUNWAY_mvt"].astype("string")
    ident_map = {v: norm_runway_ident(v) for v in raw_rwy.dropna().unique()}
    dep["ident"] = raw_rwy.map(ident_map)
    kept_ends = ends.loc[ends["dropped_reason"].eq("")]
    known = set(zip(kept_ends["apt"], kept_ends["ident"]))
    dep["rwy_ok"] = [(a, i) in known for a, i in zip(dep["apt"], dep["ident"].astype(object))]
    rwy_lines = []
    for apt, g in dep.groupby("apt", observed=True):
        counts = g.groupby(raw_rwy.loc[g.index].fillna("<NA>").astype(str), observed=True).size().sort_values(ascending=False)
        ok = {r for r in counts.index if (apt, ident_map.get(r)) in known}
        matched = ", ".join(f"{r}:{n:,}" for r, n in counts.items() if r in ok)
        bad = ", ".join(f"{r}:{n:,}" for r, n in counts.items() if r not in ok) or "none"
        dropped = ends.loc[ends["apt"].eq(apt) & ends["dropped_reason"].ne("")]
        dropped_txt = ", ".join(f"{r.ident} ({r.dropped_reason})" for r in dropped.itertuples(index=False)) or "none"
        rwy_lines.append(f"  {apt}: matched [{matched}]  unmatched [{bad}]  dropped by OurAirports filter [{dropped_txt}]")
    rwy_share = _pct(dep, ["apt", "split"], "rwy_ok")

    # --- routes --------------------------------------------------------------------
    rkey = dep["apt"] + "|" + dep["stand_id"].astype(str) + "|" + dep["ident"].fillna("").astype(str)
    rt = routes.assign(k=routes["apt"] + "|" + routes["stand_id"].astype(str) + "|" + routes["runway_ident"]).set_index("k")
    for col in ("route_found", "route_m", "ratio", "rwy_crossings"):
        dep[col] = rkey.map(rt[col])
    dep["routed"] = dep["route_found"].fillna(0).astype(bool)
    routed = _pct(dep, ["apt", "split"], "routed")
    with_route = dep.loc[dep["routed"]]
    route_stats = with_route.groupby("apt", observed=True).agg(
        median_route_m=("route_m", "median"), median_ratio=("ratio", "median"),
        p95_ratio=("ratio", lambda s: s.quantile(0.95)), crossings_pct=("rwy_crossings", lambda s: 100 * (s > 0).mean()),
    ).round(2)
    anchored = stands.set_index("stand_id")["anchor_node"]
    matched_dep = dep.loc[dep["matched"]]
    unsnapped = matched_dep.groupby("apt", observed=True)["stand_id"].apply(
        lambda s: 100 * (s.map(anchored).fillna(-1) < 0).mean()
    ).round(1).rename("matched_deps_at_unsnapped_stand_%")

    # --- verdict -------------------------------------------------------------------
    overall = dep.groupby("apt", observed=True)[["matched", "routed"]].mean()
    verdicts = [
        f"  {apt}: matched {r.matched:.1%}, routed {r.routed:.1%} -> "
        + ("ok" if r.matched >= MATCH_GATE and r.routed >= ROUTE_GATE else "weak")
        for apt, r in overall.iterrows()
    ]

    out = [
        f"=== geometry: OSM content per airport (hand-placed stands file: {stands_file}) ===",
        apt_table.to_string(),
        "\n=== stands: departures by match rule (%, rows = airport x split) ===",
        by_source.to_string(),
        "\n=== stands: distinct movement stands matched ===",
        distinct.to_string(),
        "\n=== stands: top unmatched by departures ===",
        "\n".join(top_unmatched) if top_unmatched else "  none",
        "\n=== runways: movement designators against OurAirports ends ===",
        "\n".join(rwy_lines),
        "\n=== runways: departures whose designator has a known threshold (%) ===",
        rwy_share.to_string(),
        "\n=== routes: departures with a routed taxi distance (%) ===",
        routed.to_string(),
        "\n=== routes: shape of the routed distances ===",
        route_stats.join(unsnapped).to_string(),
        f"\n=== verdict (gate: matched >= {MATCH_GATE:.0%}, routed >= {ROUTE_GATE:.0%} of departures) ===",
        "\n".join(verdicts),
        f"\nper-stand detail (every movement stand, matched or not): {write_stand_report()}",
    ]
    return "\n".join(out)
