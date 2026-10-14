"""lab/op_calibrate.py on synthetic residuals: the tables recover known quantiles, the outputs are ordered and floored,
an unknown airport takes the pooled table, and rule C passes and fails where it should. No competition data."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lab"))
op_calibrate = pytest.importorskip("op_calibrate")


def _world(n: int = 60_000, seed: int = 0) -> pd.DataFrame:
    """Two airports; the truth is the mean plus a right-skewed error whose spread grows with the mean."""
    rng = np.random.default_rng(seed)
    apt = np.where(rng.random(n) < 0.5, "AAAA", "BBBB")
    mean = rng.uniform(400, 1500, n) + np.where(apt == "BBBB", 200, 0)
    err = rng.gamma(2.0, mean / 10.0) - 2.0 * mean / 10.0  # mean zero, skewed right, spread ~ mean
    return pd.DataFrame({"y_true": mean + err, "y_pred": mean, "apt": apt, "op": True})


def test_tables_recover_the_residual_quantiles():
    w = _world()
    t = op_calibrate.fit_tables(w["y_true"], w["y_pred"], w["apt"], n_bins=10)
    assert set(t["apt"]) == {"AAAA", "BBBB", op_calibrate.ALL}
    for a in ("AAAA", "BBBB"):
        ta = t[t["apt"] == a]
        assert ta["n"].sum() == (w["apt"] == a).sum()
        assert (ta["q_p10"] < ta["q_med"]).all() and (ta["q_med"] < ta["q_p90"]).all()
        assert (ta["q_med"] < 0).all()  # a right-skewed error: the median lies below the mean
        assert ta["q_p90"].iloc[-1] > ta["q_p90"].iloc[0]  # the spread grows with the prediction
        assert np.isinf(ta["hi"].iloc[-1])


def test_apply_is_ordered_floored_and_pools_unknown_airports():
    w = _world()
    t = op_calibrate.fit_tables(w["y_true"], w["y_pred"], w["apt"])
    out = op_calibrate.apply_tables(t, np.array([10.0, 800.0, 800.0, 5000.0]), np.array(["AAAA", "AAAA", "ZZZZ", "BBBB"]))
    assert (out["y_p10"] <= out["y_med"]).all() and (out["y_med"] <= out["y_p90"]).all()
    assert (out >= op_calibrate.FLOOR_S).all().all()
    assert out["y_mean"].iloc[1] == 800.0
    pooled = op_calibrate.apply_tables(t[t["apt"] == op_calibrate.ALL].assign(apt="ZZZZ"), np.array([800.0]), np.array(["ZZZZ"]))
    assert out["y_med"].iloc[2] == pytest.approx(pooled["y_med"].iloc[0])


def test_rule_c_passes_on_a_skewed_world_and_fails_on_a_symmetric_one():
    fit, check = _world(seed=1), _world(seed=2)
    t = op_calibrate.fit_tables(fit["y_true"], fit["y_pred"], fit["apt"])
    res = op_calibrate.rule_c(t, check)
    assert res["mae_gain"] > 0 and abs(res["coverage_p10_p90"] - 0.8) < 0.02
    assert abs(res["below_p10"] - 0.1) < 0.02 and abs(res["above_p90"] - 0.1) < 0.02

    rng = np.random.default_rng(3)
    sym = check.assign(y_true=check["y_pred"] + rng.normal(0, 200, len(check)))
    t_sym = op_calibrate.fit_tables(sym["y_true"], sym["y_pred"], sym["apt"])
    res_sym = op_calibrate.rule_c(t_sym, sym.assign(y_true=sym["y_pred"] + rng.normal(0, 200, len(sym))))
    assert not res_sym["passed"] and abs(res_sym["mae_gain"]) < op_calibrate.C_MAE_GAIN


def test_rule_c_checks_only_op_rows():
    w = _world()
    t = op_calibrate.fit_tables(w["y_true"], w["y_pred"], w["apt"])
    frame = w.assign(op=np.arange(len(w)) % 2 == 0)
    assert op_calibrate.rule_c(t, frame)["n"] == int(frame["op"].sum())
