"""`deicing_context`: the cold-soak rule, its data-edge NaN, the pad queue and heavies under de-icing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prc.features.deicing import DEICING_CONTEXT, deicing_context
from prc.features.registry import FeatureContext


def at(hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(f"2026-01-05T{hhmm}", tz="UTC")


def _case():
    rows = [
        # id, apt, is_dep, time
        (100, "EHAM", False, at("05:00")),  # an arrival: EHAM has data in the 12 h before its departures
        (1, "EHAM", True, at("06:00")),     # no stand arrival within 48 h: parked overnight
        (2, "EHAM", True, at("06:10")),     # turned around in 12 min
        (3, "EHAM", True, at("06:30")),     # stand idle 8 h
        (4, "LSZH", True, at("06:00")),     # no LSZH arrival in the previous 12 h: the data edge
    ]
    mvt = pd.DataFrame(
        {"apt": [r[1] for r in rows], "is_dep": [r[2] for r in rows], "MVT_TIME_UTC_mvt": [r[3] for r in rows]},
        index=pd.Index([r[0] for r in rows], name="mvt_id"),
    )
    dep_idx = pd.Index([1, 2, 3, 4], name="mvt_id")
    deps = {
        "prev_stand_arrival": pd.DataFrame({"secs_since_prev_arr_inblock": [np.nan, 720.0, 8 * 3600.0, np.nan]}, index=dep_idx),
        "era5_surface": pd.DataFrame(
            {"era5_frost_hours_12h": [5.0, 5.0, 0.0, 3.0], "era5_snow_depth_cm": [2.0, 2.0, 2.0, 0.0], "era5_snowfall_24h_cm": [1.0, 1.0, 1.0, 0.0]},
            index=dep_idx,
        ),
        "wx_now": pd.DataFrame({"wx_deice": [1, 1, 1, 0]}, index=dep_idx),
        "aircraft_identity": pd.DataFrame({"wake_category": pd.Categorical(["M", "H", "M", "H"])}, index=dep_idx),
    }
    return mvt, FeatureContext(dep_index=dep_idx, deps=deps)


def test_cold_soak_rules_queue_and_heavies():
    mvt, ctx = _case()
    out = deicing_context(mvt, ctx)
    assert list(out.columns) == list(DEICING_CONTEXT)
    assert out["cold_soak"].tolist()[:3] == [1.0, 0.0, 1.0]
    assert np.isnan(out.loc[4, "cold_soak"])  # the data edge, not a parked aircraft
    assert out.loc[1, "cold_soak_frost_h"] == pytest.approx(5.0)
    assert out.loc[3, "cold_soak_snow_cm"] == pytest.approx(3.0)
    assert out["cold_soak_deice"].tolist()[:3] == [1.0, 0.0, 1.0]
    # departure 3 queues behind departure 1 (30 min earlier, cold-soaked, de-icing); nobody queues behind itself
    assert out.loc[1, "cold_soak_deice_dep_back_60m"] == 0
    assert out.loc[3, "cold_soak_deice_dep_back_60m"] == 1
    # a heavy under METAR de-icing conditions, and a heavy under overnight frost without them
    assert out.loc[2, "heavy_deice"] == 1 and out.loc[4, "heavy_deice"] == 1 and out.loc[1, "heavy_deice"] == 0
