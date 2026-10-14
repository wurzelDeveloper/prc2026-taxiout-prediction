"""ERA5 reanalysis: the normaliser's validation and the block's causality.

No network and no competition data: a hand-made Open-Meteo answer and a
three-departure frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prc import paths
from prc.features.reanalysis import ERA5_SURFACE, era5_surface
from prc.features.registry import FeatureContext
from prc.reanalysis import VARIABLES
from prc.reanalysis.normalise import parse_answer


def _answer(hours: list[str], **overrides) -> dict:
    n = len(hours)
    hourly = {"time": hours}
    for v in VARIABLES:
        hourly[v] = [0.0] * n
    hourly.update(overrides)
    return {"hourly": hourly, "hourly_units": {"snow_depth": "m", "snowfall": "cm"}}


HOURS = [f"2026-01-05T{h:02d}:00" for h in range(0, 8)]


def test_parse_converts_snow_depth_to_cm_and_keeps_utc():
    body = _answer(HOURS, snow_depth=[0.13] * len(HOURS))
    out = parse_answer("EHAM", body)
    assert out["snow_depth_cm"].iloc[0] == pytest.approx(13.0)
    assert str(out["time"].dtype) == "datetime64[us, UTC]"
    assert (out["apt"] == "EHAM").all()


def test_parse_rejects_a_missing_variable_and_a_gap():
    body = _answer(HOURS)
    del body["hourly"]["snowfall"]
    with pytest.raises(ValueError, match="lacks"):
        parse_answer("EHAM", body)
    with pytest.raises(ValueError, match="gaps"):
        parse_answer("EHAM", _answer(HOURS[:3] + HOURS[4:]))


def test_block_uses_only_hours_at_or_before_takeoff(tmp_path, monkeypatch):
    # Snow falls only in the hour ending 05:00 (2 cm) and the hour ending 07:00 (5 cm).
    snow = [0, 0, 0, 0, 0, 2.0, 0, 5.0]
    temp = [-3.0] * len(HOURS)
    table = parse_answer("EHAM", _answer(HOURS, snowfall=snow, temperature_2m=temp, dew_point_2m=[-4.0] * len(HOURS)))
    f = tmp_path / "era5_hourly.parquet"
    table.to_parquet(f, index=False)
    monkeypatch.setattr(paths, "ERA5_FILE", f)

    t = pd.to_datetime(["2026-01-05T05:30", "2026-01-05T06:59", "2026-01-06T10:00"], utc=True)
    mvt = pd.DataFrame({"apt": ["EHAM", "EHAM", "EHAM"], "MVT_TIME_UTC_mvt": t}, index=pd.Index([1, 2, 3], name="mvt_id"))
    out = era5_surface(mvt, FeatureContext(dep_index=mvt.index))

    assert list(out.columns) == list(ERA5_SURFACE)
    # 05:30 and 06:59 both see the 05:00 hour's 2 cm and never the 07:00 hour's 5 cm.
    assert out.loc[1, "era5_snowfall_3h_cm"] == pytest.approx(2.0)
    assert out.loc[2, "era5_snowfall_3h_cm"] == pytest.approx(2.0)
    assert out.loc[2, "era5_snowfall_24h_cm"] == pytest.approx(2.0)
    # Sub-zero and a 1 degree spread every hour: frost hours count up to the window.
    assert out.loc[2, "era5_frost_hours_12h"] == pytest.approx(7.0)
    # Past the table's end: NaN, not the last row carried forward.
    assert np.isnan(out.loc[3]).all()
