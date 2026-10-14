"""Which OSM stand a movement's ``STAND_mvt`` means.

The two vocabularies were written by different people: the airport's
operational system on one side, volunteer mappers on the other. Most stands
agree verbatim; the rest differ in case, padding ("A021"), punctuation
("A-21") or a position suffix ("A21" against "A21L"/"A21R"). The keys are
tried from the most literal to the loosest and the first unique hit wins;
the source of every match is recorded so the report -- and the model -- can
see how much of the data rests on each rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc.features._geolib import clean_ref, stand_keys

MATCH_SOURCES = ("exact", "norm", "nosuffix", "gate", "manual", "none")
KEY_FORMS = ("exact", "norm", "nosuffix")


def _key_index(refs: pd.DataFrame) -> dict[str, dict[str, list[int]]]:
    """``form -> key -> [stand_id, ...]`` over one airport's exploded OSM references."""
    index: dict[str, dict[str, list[int]]] = {form: {} for form in KEY_FORMS}
    for r in refs.itertuples(index=False):
        for form, key in zip(KEY_FORMS, stand_keys(r.ref)):
            if key:
                index[form].setdefault(key, []).append(int(r.stand_id))
    return index


def _resolve(candidates: list[int], centroid_dist: dict[int, float]) -> tuple[int, int]:
    """``(stand_id, ambiguous)``: one candidate, or the one nearest the apron centroid."""
    unique = sorted(set(candidates))
    if len(unique) == 1:
        return unique[0], 0
    return min(unique, key=lambda s: (centroid_dist.get(s, np.inf), s)), 1


def match_stands(mvt_stands: pd.DataFrame, refs: pd.DataFrame) -> pd.DataFrame:
    """Match every movement stand of one airport.

    ``mvt_stands``: ``apt, stand_mvt, n_dep_train, n_dep_rank`` (distinct
    values of ``STAND_mvt``). ``refs``: the airport's exploded OSM references
    with ``stand_id, kind, ref, east_m, north_m``.
    Returns ``apt, stand_mvt, stand_id, match_source, ambiguous, n_dep_train, n_dep_rank``.
    """
    pp = refs.loc[refs["kind"].ne("gate")]
    gates = refs.loc[refs["kind"].eq("gate")]
    centroid = (pp["east_m"].mean(), pp["north_m"].mean()) if len(pp) else (0.0, 0.0)
    centroid_dist = {
        int(r.stand_id): float(np.hypot(r.east_m - centroid[0], r.north_m - centroid[1]))
        for r in refs.drop_duplicates("stand_id").itertuples(index=False)
    }
    pp_index, gate_index = _key_index(pp), _key_index(gates)

    rows = []
    for r in mvt_stands.itertuples(index=False):
        keys = dict(zip(KEY_FORMS, stand_keys(r.stand_mvt)))
        stand_id, source, ambiguous = -1, "none", 0
        for form in KEY_FORMS:
            key = keys[form]
            if key and key in pp_index[form]:
                stand_id, ambiguous = _resolve(pp_index[form][key], centroid_dist)
                source = form
                break
        # Parking positions mapped as ways carry negative ids, so "not found"
        # is the source, never the sign of the id.
        if source == "none":
            for form in KEY_FORMS:
                key = keys[form]
                if key and key in gate_index[form]:
                    stand_id, ambiguous = _resolve(gate_index[form][key], centroid_dist)
                    source = "gate"
                    break
        rows.append({
            "apt": r.apt, "stand_mvt": r.stand_mvt, "stand_id": stand_id, "match_source": source,
            "ambiguous": ambiguous, "n_dep_train": int(r.n_dep_train), "n_dep_rank": int(r.n_dep_rank),
        })
    cols = ["apt", "stand_mvt", "stand_id", "match_source", "ambiguous", "n_dep_train", "n_dep_rank"]
    return pd.DataFrame(rows, columns=cols)


def duplicate_counts(refs: pd.DataFrame) -> pd.Series:
    """``stand_id -> number of parking positions at the airport sharing its normalised key``."""
    pp = refs.loc[refs["kind"].ne("gate")].copy()
    pp["norm"] = [stand_keys(x)[1] or clean_ref(x) for x in pp["ref"]]
    per_key = pp.drop_duplicates(["stand_id", "norm"]).groupby("norm")["stand_id"].nunique()
    return pp.drop_duplicates("stand_id").set_index("stand_id")["norm"].map(per_key).astype("int16")
