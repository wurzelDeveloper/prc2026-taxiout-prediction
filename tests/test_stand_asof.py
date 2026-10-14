"""`turnaround.stand_asof`: the stand join both turnaround blocks build on. No data, no network."""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc.features.turnaround import ARRIVAL_COLUMNS, stand_asof

T0 = pd.Timestamp("2025-03-01 06:00", tz="UTC")


def _frame():
    def at(minutes):
        return T0 + pd.Timedelta(minutes=minutes)

    arr = pd.DataFrame(
        {
            "apt": ["EDDF"] * 3,
            "STAND_mvt": ["A1", "A1", "B2"],
            "MVT_TIME_UTC_mvt": [at(0), at(120), at(30)],            # landing
            "BLOCK_TIME_UTC_mvt": [at(10), at(130), at(40)],         # in-block
            "TAXITIME_SEC_mvt": [600.0, 600.0, 600.0],
            "RUNWAY_mvt": ["25C", "07C", "25C"],
            "AIRCRAFT_TYPE_mvt": ["A320", "B738", "A320"],
        },
        index=pd.Index([101, 102, 103], name="mvt_id"),
    )
    dep = pd.DataFrame(
        {
            "apt": ["EDDF"] * 3,
            "STAND_mvt": ["A1", "A1", None],
            "MVT_TIME_UTC_mvt": [at(60), at(200), at(60)],
            "AIRCRAFT_TYPE_mvt": ["A320", "A320", "C56X"],
        },
        index=pd.Index([1, 2, 3], name="mvt_id"),
    )
    return dep, arr


def test_backward_finds_the_last_occupant_and_forward_the_next():
    dep, arr = _frame()
    back, table = stand_asof(dep, arr, direction="backward")
    assert back.index.tolist() == [1, 2, 3]
    assert set(ARRIVAL_COLUMNS) <= set(back.columns)
    # Departure 1 at 06:60 follows the arrival that blocked in at 06:10; departure 2 the one at 08:10.
    assert back.loc[1, "arr_inblock"] == (T0 + pd.Timedelta(minutes=10)).timestamp()
    assert back.loc[2, "arr_inblock"] == (T0 + pd.Timedelta(minutes=130)).timestamp()
    assert back.loc[2, "arr_type"] == "B738" and back.loc[2, "arr_runway"] == "07C"
    assert np.isnan(back.loc[3, "arr_inblock"])  # no stand, no match

    fwd, _ = stand_asof(dep, arr, direction="forward", allow_exact_matches=False)
    assert fwd.loc[1, "arr_inblock"] == (T0 + pd.Timedelta(minutes=130)).timestamp()  # the next occupant
    assert np.isnan(fwd.loc[2, "arr_inblock"])  # nobody after 09:20

    # arr_pos indexes the sorted arrival table the helper returns.
    pos = int(back.loc[2, "arr_pos"])
    assert table.loc[pos, "arr_inblock"] == back.loc[2, "arr_inblock"]
    assert table["arr_inblock"].is_monotonic_increasing


def test_extra_arrival_columns_ride_along():
    dep, arr = _frame()
    merged, _ = stand_asof(dep, arr, direction="backward", extra={"arr_flight": np.array(["LH1", "LH2", "LH3"])})
    assert merged.loc[1, "arr_flight"] == "LH1" and merged.loc[2, "arr_flight"] == "LH2"
