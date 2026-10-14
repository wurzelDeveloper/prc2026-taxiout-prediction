"""Hand-maintained stand positions, merged over the OSM stands by the normaliser.

OSM does not know every stand (Munich's 350s and 510s, Zürich's GA6, a
cargo apron here and there), and now and then it puts one in the wrong
place. A CSV with ``apt, stand_mvt, lat, lon`` fixes both: a row with
coordinates becomes a stand of its own -- matched to that ``STAND_mvt``
directly, snapped onto the taxiway graph, zoned and routed like any other --
and a row with both coordinates left empty says "treat this stand as
unknown", which drops a wrong OSM match. Optional ``ref`` and ``note``
columns are kept; every other column is ignored, so an edited copy of
``stand_match_report.csv`` works as input once it is cut down to the rows
that should override.

Manual rows apply to the listed ``STAND_mvt`` only: they never take part in
the reference matching of other stands.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from prc.features._geolib import clean_ref

MANUAL_ID_BASE = 10**12
"""Stand ids for manual rows start here: far above any OSM node id, and positive
(way-mapped OSM stands are negative), so the three id spaces never meet."""

REQUIRED = ("apt", "stand_mvt", "lat", "lon")
COLUMNS = ["apt", "stand_mvt", "lat", "lon", "ref", "note", "has_coords"]


def load_manual_stands(path: str | Path, icaos: list[str] | None = None) -> pd.DataFrame:
    """Read and validate the override file. Raises on anything that would be silently wrong."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"stands file not found: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing column(s) {missing}; need {list(REQUIRED)}")

    def numeric(col: str) -> pd.Series:
        return pd.to_numeric(df[col].str.strip().replace("", np.nan), errors="coerce")

    out = pd.DataFrame({
        "apt": df["apt"].str.strip().str.upper(),
        "stand_mvt": df["stand_mvt"].str.strip(),
        "lat": numeric("lat"),
        "lon": numeric("lon"),
        "ref": df["ref"].str.strip() if "ref" in df.columns else "",
        "note": df["note"].str.strip() if "note" in df.columns else "",
    })
    raw_blank = (df["lat"].str.strip().eq("")) & (df["lon"].str.strip().eq(""))
    out["has_coords"] = ~raw_blank

    problems = []
    if out["stand_mvt"].eq("").any():
        problems.append(f"{int(out['stand_mvt'].eq('').sum())} row(s) with an empty stand_mvt")
    half = out["has_coords"] & (out["lat"].isna() | out["lon"].isna())
    if half.any():
        problems.append(f"unparsable or half-filled coordinates at stand(s) {out.loc[half, 'stand_mvt'].tolist()}")
    out_of_range = out["has_coords"] & ~half & ((out["lat"].abs() > 90) | (out["lon"].abs() > 180))
    if out_of_range.any():
        problems.append(f"coordinates out of range at stand(s) {out.loc[out_of_range, 'stand_mvt'].tolist()}")
    dup = out.duplicated(["apt", "stand_mvt"], keep=False)
    if dup.any():
        problems.append(f"duplicate (apt, stand_mvt) rows: {sorted(set(map(tuple, out.loc[dup, ['apt', 'stand_mvt']].values)))}")
    if icaos is not None:
        unknown = sorted(set(out["apt"]) - set(icaos))
        if unknown:
            problems.append(f"airport(s) not in the data: {unknown}")
    if problems:
        raise ValueError(f"{path.name}: " + "; ".join(problems))
    return out[COLUMNS].reset_index(drop=True)


def manual_stands(rows: pd.DataFrame, apt: str, start_id: int = MANUAL_ID_BASE) -> pd.DataFrame:
    """Stand rows (the `sources.osm_stands` schema plus ``stand_mvt``) for one airport's placed stands."""
    r = rows.loc[rows["apt"].eq(apt) & rows["has_coords"]].reset_index(drop=True)
    ref_raw = r["ref"].where(r["ref"].ne(""), r["stand_mvt"])
    return pd.DataFrame({
        "stand_id": start_id + np.arange(len(r), dtype="int64"),
        "kind": "manual",
        "ref_raw": ref_raw,
        "lat": r["lat"].astype("float64"),
        "lon": r["lon"].astype("float64"),
        "entry_node": -1,
        "ref_norm": ref_raw.map(clean_ref),
        "stand_mvt": r["stand_mvt"],
    })


def apply_manual(matches: pd.DataFrame, placed: pd.DataFrame, blank: set[str]) -> pd.DataFrame:
    """Override one airport's match table: placed stands win, blanked stands become unknown."""
    out = matches.copy()
    by_mvt = dict(zip(placed["stand_mvt"], placed["stand_id"]))
    is_placed = out["stand_mvt"].isin(by_mvt)
    out.loc[is_placed, "stand_id"] = out.loc[is_placed, "stand_mvt"].map(by_mvt).astype("int64")
    out.loc[is_placed, "match_source"] = "manual"
    out.loc[is_placed, "ambiguous"] = 0
    is_blank = out["stand_mvt"].isin(blank) & ~is_placed
    out.loc[is_blank, "stand_id"] = -1
    out.loc[is_blank, "match_source"] = "none"
    out.loc[is_blank, "ambiguous"] = 0
    return out
