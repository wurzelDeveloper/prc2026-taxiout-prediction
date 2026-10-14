"""Training-data cleaning: rules that flag departures by their target value.

The challenge data is uncleaned operational data, and RMSE is a tail metric:
on the Jan+Jul 2025 holdout the worst 0.1 % of rows carry about 80 % of the
squared error, most of it at Rome, where departures with no aircraft type
record taxi-out times of hours. The 2026 ranking set contains the same rows
and they are scored, so cleaning can only ever remove rows from the *fit*:
every ranking row is predicted, every validation row is scored, and a second
metric set on the unflagged rows shows what the bulk of the data looks like.

Not everything odd is wrong. Helicopters at LSZH and EHAM lift off from a pad
(``RUNWAY_mvt`` ``H``/``HEL``) seconds after block-off, sometimes before it by
the clock, and they recur in 2026 -- so the minimum-taxi rule exempts them.

Flags read the target, so they are never features and never computed for
ranking rows (which have none). This module imports nothing from
`prc.features`: cleaning is not feature code and must not enter its hashes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc import metrics
from prc.config import CleanConfig
from prc.validation import Split

HELI_RUNWAYS = frozenset({"H", "HEL"})
"""Runway placeholders that mean "helicopter pad". `prc.features._geolib.norm_runway_ident`
maps the same designators to None; here they are kept, not dropped."""
RULES = ("max_taxi", "min_taxi", "day_offset", "untyped")
DAY_SEC = 86_400.0
DAY_OFFSET_REPORT_TOL_SEC = 3600.0
"""Tolerance of the wrong-date report mask (`day_offset_rows`): a target within
this of a whole number of days is a block time on the wrong date. Every run
reports its holdout RMSE without those rows next to the headline number, so a
change to the tail is not judged on a handful of unlearnable rows."""
MVT_COLUMNS = ("apt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt")
"""What `flag_table` reads from the movement frame."""


def is_heli_runway(runway: pd.Series) -> pd.Series:
    return runway.astype("string").str.strip().str.upper().isin(HELI_RUNWAYS).fillna(False).astype(bool)


def is_untyped(aircraft_type: pd.Series) -> pd.Series:
    s = aircraft_type.astype("string").str.strip()
    return (s.isna() | s.eq("")).fillna(True).astype(bool)


def _known_airports(mvt: pd.DataFrame) -> set[str]:
    col = mvt["apt"]
    if isinstance(col.dtype, pd.CategoricalDtype):
        return {str(a) for a in col.cat.categories}
    return {str(a) for a in col.dropna().unique()}


def flag_table(mvt: pd.DataFrame, y: pd.Series, clean: CleanConfig) -> pd.DataFrame:
    """One bool column per rule, indexed like ``y`` (training departures only).

    A fifth column, ``heli_exempt``, marks rows the minimum-taxi rule would
    have caught but for the helicopter exemption; it is reported, never
    counted as a flag.
    """
    rows = mvt.loc[y.index, list(MVT_COLUMNS)]
    yv = y.to_numpy(dtype="float64")
    out = pd.DataFrame(False, index=y.index, columns=[*RULES, "heli_exempt"])
    with np.errstate(invalid="ignore"):
        if clean.max_taxi_sec is not None:
            out["max_taxi"] = yv > clean.max_taxi_sec
        if clean.min_taxi_sec is not None:
            below = yv < clean.min_taxi_sec
            heli = is_heli_runway(rows["RUNWAY_mvt"]).to_numpy()
            out["min_taxi"] = below & ~heli
            out["heli_exempt"] = below & heli
        if clean.day_offset_tol_sec is not None:
            k = np.round(yv / DAY_SEC)
            out["day_offset"] = (k >= 1) & (np.abs(yv - k * DAY_SEC) <= clean.day_offset_tol_sec)
        if clean.untyped_airports:
            unknown = sorted(set(clean.untyped_airports) - _known_airports(mvt))
            if unknown:
                raise ValueError(f"data.clean.untyped_airports: not in the data: {unknown}")
            apt = rows["apt"].astype("string").isin(clean.untyped_airports).fillna(False).to_numpy(dtype=bool)
            out["untyped"] = is_untyped(rows["AIRCRAFT_TYPE_mvt"]).to_numpy() & apt
    return out


def flag_rows(mvt: pd.DataFrame, y: pd.Series, clean: CleanConfig) -> pd.Series:
    """True where any rule fires. Indexed like ``y``; ranking rows cannot appear."""
    return flag_table(mvt, y, clean)[list(RULES)].any(axis=1).rename("flagged")


def day_offset_rows(mvt: pd.DataFrame, y: pd.Series, tol: float = DAY_OFFSET_REPORT_TOL_SEC) -> pd.Series:
    """True where the target sits within ``tol`` of a whole number of days.

    A report mask, not a cleaning rule: it never leaves the fit or the score,
    it only names the rows the secondary ``rmse_ex_dayoffset`` metric leaves out.
    """
    return flag_table(mvt, y, CleanConfig(day_offset_tol_sec=tol))["day_offset"].rename("dayoffset")


def describe_flags(table: pd.DataFrame, apt: pd.Series) -> pd.DataFrame:
    """Per airport (plus a total row): rows caught by each rule, by any rule, and exempted helicopters."""
    frame = table[[*RULES, "heli_exempt"]].copy()
    frame["any"] = table[list(RULES)].any(axis=1)
    frame["apt"] = apt.reindex(table.index).astype("string").fillna("?").astype(str).to_numpy()
    agg = frame.groupby("apt")[[*RULES, "any", "heli_exempt"]].sum().astype(int)
    agg["n"] = frame.groupby("apt").size()
    agg.loc["total"] = agg.sum()
    agg["share_pct"] = (100 * agg["any"] / agg["n"].clip(lower=1)).round(3)
    return agg[[*RULES, "any", "n", "share_pct", "heli_exempt"]]


def apply_to_splits(splits: list[Split], flagged: pd.Series, drop_from_fit: bool) -> list[Split]:
    """Remove flagged rows from every fold's training index. Validation indices are never touched."""
    if not drop_from_fit or not bool(flagged.any()):
        return list(splits)
    out = []
    for s in splits:
        keep = ~flagged.reindex(s.train_idx, fill_value=False).to_numpy(dtype=bool)
        if not keep.any():
            raise ValueError(f"cleaning rules removed every training row of fold '{s.name}'")
        out.append(Split(s.name, s.train_idx[keep], s.val_idx))
    return out


def clean_metrics(y_true, y_pred, flagged) -> dict[str, float]:
    """`regression_metrics` over the unflagged rows; an ``n == 0`` record if none are left."""
    mask = ~np.asarray(flagged, dtype=bool)
    if not mask.any():
        return dict.fromkeys(metrics.METRIC_NAMES, float("nan")) | {"n": 0}
    return metrics.regression_metrics(np.asarray(y_true, dtype="float64")[mask], np.asarray(y_pred, dtype="float64")[mask])
