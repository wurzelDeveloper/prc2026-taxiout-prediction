"""Wake minima and runway queues on a hand-made takeoff sequence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prc.features.registry import FeatureContext
from prc.features.wake_queue import WAKE_QUEUE, required_separation, wake_queue


def test_icao_minima():
    prev = np.array(["H", "H", "J", "J", "M", "M", "L", "H"], dtype=object)
    cur = np.array(["M", "L", "H", "M", "L", "M", "H", "H"], dtype=object)
    assert required_separation(prev, cur).tolist() == [120, 120, 120, 180, 120, 60, 60, 60]


def test_queue_runs_follow_the_pair_minimum():
    t0 = pd.Timestamp("2026-01-05T08:00", tz="UTC")
    s = pd.to_timedelta
    rows = [
        # id, runway, takeoff offset, wake
        (1, "18", s("0s"), "H"),
        (2, "18", s("125s"), "M"),   # 2 min behind a heavy: at the minimum, queue position 1
        (3, "18", s("190s"), "M"),   # 65 s behind a medium: still queued
        (4, "18", s("900s"), "M"),   # a free runway: new run
        (5, "25C", s("130s"), "M"),  # another runway: its own sequence
    ]
    mvt = pd.DataFrame(
        {
            "apt": "EDDF",
            "is_dep": True,
            "RUNWAY_mvt": [r[1] for r in rows],
            "MVT_TIME_UTC_mvt": [t0 + r[2] for r in rows],
            "WK_TBL_CAT_flt": [r[3] for r in rows],
        },
        index=pd.Index([r[0] for r in rows], name="mvt_id"),
    )
    out = wake_queue(mvt, FeatureContext(dep_index=mvt.index))
    assert list(out.columns) == list(WAKE_QUEUE)
    assert out.loc[2, "wq_required_s"] == 120 and out.loc[2, "wq_slack_s"] == pytest.approx(5)
    assert out.loc[1, "wq_next_required_s"] == 120
    assert out.loc[[1, 2, 3, 4], "wq_queue_pos"].tolist() == [0, 1, 2, 0]
    assert out.loc[3, "wq_queue_age_s"] == pytest.approx(190)
    assert out.loc[1, "wq_queue_followers"] == 2 and out.loc[1, "wq_queue_len"] == 3
    assert out.loc[4, "wq_queue_len"] == 1
    assert np.isnan(out.loc[5, "wq_gap_prev_s"]) and out.loc[5, "wq_queue_pos"] == 0
