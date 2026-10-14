"""`prc blend`'s arithmetic on synthetic prediction frames. No runs, no data."""

from __future__ import annotations

import pandas as pd
import pytest

from prc.blend import average_predictions


def _oof(pred, extra=None) -> pd.DataFrame:
    idx = pd.Index([10, 11, 12], name="mvt_id")
    frame = pd.DataFrame({"fold": "seasonal", "y_true": [100.0, 200.0, 300.0], "y_pred": pred}, index=idx)
    for key, value in (extra or {}).items():
        frame[key] = value
    return frame


def test_mean_of_predictions_and_carried_columns():
    a = _oof([90.0, 210.0, 330.0], {"flagged": [False, True, False], "p_stale": [0.0, 0.2, 1.0]})
    b = _oof([110.0, 190.0, 270.0], {"flagged": [False, True, False], "p_stale": [0.2, 0.4, 1.0]})
    out = average_predictions([a, b])
    assert out["y_pred"].tolist() == [100.0, 200.0, 300.0]
    assert out["p_stale"].tolist() == pytest.approx([0.1, 0.3, 1.0])
    assert out["flagged"].tolist() == [False, True, False] and out["y_true"].tolist() == [100.0, 200.0, 300.0]
    assert out["fold"].tolist() == ["seasonal"] * 3 and out.index.name == "mvt_id"


def test_optional_columns_need_every_source():
    out = average_predictions([_oof([1.0, 2.0, 3.0], {"p_stale": [0.0, 0.0, 0.0]}), _oof([1.0, 2.0, 3.0])])
    assert "p_stale" not in out.columns and out["y_pred"].tolist() == [1.0, 2.0, 3.0]


def test_mismatched_rows_or_truth_raise():
    a = _oof([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="different row set"):
        average_predictions([a, a.iloc[::-1]])
    c = _oof([1.0, 2.0, 3.0])
    c["y_true"] = [100.0, 200.0, 999.0]
    with pytest.raises(ValueError, match="y_true"):
        average_predictions([a, c])
    with pytest.raises(ValueError, match="nothing"):
        average_predictions([])
    with pytest.raises(KeyError, match="y_pred"):
        average_predictions([a.drop(columns="y_pred")])


def test_ranking_frames_carry_only_the_prediction():
    test = pd.DataFrame({"y_pred": [5.0, 7.0]}, index=pd.Index([1, 2], name="mvt_id"))
    out = average_predictions([test, test.assign(y_pred=[7.0, 9.0])])
    assert out["y_pred"].tolist() == [6.0, 8.0] and list(out.columns) == ["y_pred"]


def test_weighted_average_normalises_the_weights():
    a = _oof([100.0, 200.0, 300.0])
    b = _oof([200.0, 400.0, 600.0])
    out = average_predictions([a, b], weights=[3.0, 1.0])
    assert out["y_pred"].tolist() == pytest.approx([125.0, 250.0, 375.0])
    with pytest.raises(ValueError, match="weights"):
        average_predictions([a, b], weights=[1.0])
