"""Leakage checks on the cached ADS-B stage tables (prerequisite (b) of the data-science review, findings 10.23).

The ADS-B stage (`lab/adsb_apply.py`) reads per-departure tables in `cache/adsb/` that the lab scripts build from the
adsb.lol points. Two properties keep them honest:

1. Every own-movement timing is measured before the aircraft's ADS-B takeoff: the tables store "seconds before the
   ADS-B takeoff", so a negative value would mean a point after takeoff (the takeoff itself or the climb) leaked into a
   ground event. The link to the recorded takeoff stays within the matching tolerance.
2. No stage input is a near-copy of the target: the absolute Pearson correlation of every numeric column with the
   2025 taxi time stays below 0.99 (the threshold `test_features.py` applies to the harness's blocks). A block time
   reintroduced anywhere would show up as |r| close to 1, because TAXITIME = MVT_TIME - BLOCK_TIME exactly.

Skipped when the cache has not been built (no competition data or no ADS-B crawl on this machine).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prc import paths
from prc.adsb.events import MATCH_TOL_S

ADSB = paths.CACHE / "adsb"
pytestmark = pytest.mark.skipif(not (ADSB / "features3.parquet").exists() or not (paths.INTERIM / "target.parquet").exists(),
                                reason="ADS-B stage tables or the target are not built on this machine")

OWN_TIMINGS = ("adsb_parked_until", "adsb_first_seen", "adsb_first_move", "adsb_taxi_start")


def _daily(prefix: str) -> pd.DataFrame:
    f = pd.concat([pd.read_parquet(p) for p in sorted(ADSB.glob(f"{prefix}_*.parquet"))])
    return f[~f.index.duplicated(keep="last")]


@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
    return {"features": _daily("features"), "features2": _daily("features2"),
            "features3": pd.read_parquet(ADSB / "features3.parquet"), "features4": pd.read_parquet(ADSB / "features4.parquet")}


def test_own_movement_timings_precede_the_adsb_takeoff(tables):
    f1 = tables["features"]
    for c in OWN_TIMINGS:
        v = f1[c].dropna()
        assert len(v) > 0, c
        assert (v >= 0).all(), f"{c}: {(v < 0).sum()} values after the ADS-B takeoff (min {v.min():.1f} s)"
    for name in ("features2", "features3", "features4"):
        for c in [c for c in tables[name].columns if c.endswith("_tk")]:
            v = pd.to_numeric(tables[name][c], errors="coerce").dropna()
            assert (v >= 0).all(), f"{name}.{c}: {(v < 0).sum()} values after the ADS-B takeoff"
    off = f1["adsb_takeoff_offset"].dropna()
    assert off.abs().max() <= MATCH_TOL_S, "an ADS-B takeoff linked outside the matching tolerance"


def test_no_stage_input_is_a_copy_of_the_target(tables):
    y = pd.read_parquet(paths.INTERIM / "target.parquet")["TAXITIME_SEC_mvt"].astype("float64")
    y = y[y.notna() & (y < 7200)]
    worst = (0.0, "")
    for name, f in tables.items():
        num = f.select_dtypes(include="number").astype("float64")
        idx = num.index.intersection(y.index)
        num, yy = num.loc[idx], y.loc[idx]
        num = num.loc[:, (num.notna().sum() >= 1000) & (num.std() > 0)]
        r = num.corrwith(yy).abs().dropna()
        if len(r) and r.max() > worst[0]:
            worst = (float(r.max()), f"{name}.{r.idxmax()}")
        assert (r < 0.99).all(), f"{name}: {r[r >= 0.99].to_dict()} correlate with the target at |r| >= 0.99"
    assert worst[0] > 0.1, "no column correlates with the target at all: the join to the target failed"
