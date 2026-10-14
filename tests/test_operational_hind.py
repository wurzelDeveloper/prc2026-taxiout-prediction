"""The hindsight rung (`prc.features.operational_hind`): brute force, quarantine, and the hindsight itself.

``op_h0_hind_lab`` counts traffic over the flight's OWN taxi ``[recorded off-block, own takeoff]``. It is lab only
and HINDSIGHT by design: it measures how much of the literature's accuracy comes from own-taxi counts, and must
never reach an operational (``op_h1_*``) or a board/refit config. Asserted here:

* the four counts against brute force on a synthetic frame with dense ties at the window edges, stale and
  negative records, missing runways and ranking rows;
* the quarantine: no ``op_h1_*`` config and no config with ``refit_on_all`` resolves the block, its group is its
  own (``group:op_lab_h0`` does not pull it in), and only the hindsight rung includes it;
* the hindsight is real: moving the own takeoff moves the outputs (the opposite of T1 in test_operational.py).

Synthetic only: no competition data, no network.
"""

from __future__ import annotations

import inspect
import re

import numpy as np
import pandas as pd
import pytest

from prc import paths
from prc.config import load_config
from prc.features import _oplib, all_specs, operational_hind, resolve
from prc.features.build import compute_block
from prc.features.registry import FeatureContext

BLOCK = "op_h0_hind_lab"
GROUP = "op_lab_h0_hind"
CAP = operational_hind.HIND_CAP_S
T0 = pd.Timestamp("2025-05-01", tz="UTC").timestamp()


def _ts(seconds) -> np.ndarray:
    return pd.to_datetime(np.asarray(seconds, dtype="float64"), unit="s", utc=True).to_numpy()


def hind_world(seed: int = 3, n_dep: int = 260, n_arr: int = 200):
    """Two airports over six hours on a 30-s grid (many ties), departures and arrivals; the last 40 departures rank.

    Off-blocks: mostly 1-30 min before the takeoff, some exactly an hour, some stale (2-4 h), some after the
    takeoff (negative taxi), some missing; a few departures without a runway.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for kind, n in (("dep", n_dep), ("arr", n_arr)):
        t = T0 + rng.integers(0, 6 * 3600 // 30, n) * 30.0
        apt = rng.choice(["EDDF", "LSZH"], n)
        rwy = rng.choice(["25C", "18", "07C"], n).astype(object)
        if kind == "dep":
            rwy[rng.random(n) < 0.05] = None
        for i in range(n):
            rows.append({"is_dep": kind == "dep", "t": t[i], "apt": apt[i], "rwy": rwy[i]})
    raw = pd.DataFrame(rows).sort_values("t", kind="stable").reset_index(drop=True)
    raw.index = pd.Index(np.arange(1, len(raw) + 1, dtype="int64") * 11, name="mvt_id")
    is_dep = raw["is_dep"].to_numpy(dtype=bool)
    dep_ids = raw.index[is_dep]
    split = np.where(is_dep, "train", "train").astype(object)
    split[np.flatnonzero(is_dep)[-40:]] = "rank"
    mvt = pd.DataFrame(
        {
            "MVT_TIME_UTC_mvt": _ts(raw["t"]),
            "RUNWAY_mvt": pd.Categorical(raw["rwy"]),
            "STAND_mvt": pd.Categorical(np.full(len(raw), "A1")),
            "split": pd.Categorical(split),
            "is_dep": is_dep,
            "apt": pd.Categorical(raw["apt"]),
        },
        index=raw.index,
    )
    t_dep = raw.loc[dep_ids, "t"].to_numpy()
    u = rng.random(len(dep_ids))
    taxi = rng.integers(2, 61, len(dep_ids)) * 30.0
    taxi = np.where(u < 0.06, CAP, taxi)  # exactly the cap
    taxi = np.where((u >= 0.06) & (u < 0.12), rng.integers(2, 5, len(dep_ids)) * 3600.0, taxi)  # stale
    taxi = np.where((u >= 0.12) & (u < 0.16), -60.0, taxi)  # off-block after the takeoff
    b = t_dep - taxi
    b[(u >= 0.16) & (u < 0.19)] = np.nan
    train = mvt.loc[dep_ids, "split"].eq("train").to_numpy()
    offblocks = pd.Series(b[train], index=dep_ids[train], name="offblock")
    return mvt, offblocks


def build(mvt: pd.DataFrame, offblocks: pd.Series, monkeypatch, dep_index: pd.Index | None = None) -> pd.DataFrame:
    monkeypatch.setattr(_oplib, "load_offblocks", lambda: offblocks)
    if dep_index is None:
        dep_index = mvt.index[mvt["is_dep"].to_numpy()]
    return compute_block(all_specs()[BLOCK], mvt, FeatureContext(dep_index=dep_index))


def brute(mvt: pd.DataFrame, offblocks: pd.Series) -> pd.DataFrame:
    """The four counts by a loop over every pair, straight from the definitions in the module docstring."""
    t_all = _oplib.secs(mvt["MVT_TIME_UTC_mvt"])
    dep = mvt["is_dep"].to_numpy(dtype=bool)
    apt = mvt["apt"].astype(str).to_numpy()
    rwy = (mvt["apt"].astype(str) + "_" + mvt["RUNWAY_mvt"].astype("string").fillna("__NA__")).to_numpy()
    b_all = offblocks.reindex(mvt.index).to_numpy(dtype="float64")
    ids = mvt.index.to_numpy()
    out = {}
    for i in np.flatnonzero(dep):
        b, t = b_all[i], t_all[i]
        if not (np.isfinite(b) and t >= b):
            out[ids[i]] = [np.nan] * 4
            continue
        s = max(b, t - CAP)
        others = dep & (ids != ids[i])
        in_win = (t_all >= s) & (t_all <= t)
        taxi = t_all - b_all
        is_iv = others & np.isfinite(taxi) & (taxi >= 0) & (taxi <= CAP)
        out[ids[i]] = [
            int((others & in_win & (apt == apt[i])).sum()),
            int((others & in_win & (rwy == rwy[i])).sum()),
            int((~dep & in_win & (apt == apt[i])).sum()),
            int((is_iv & (apt == apt[i]) & (b_all <= t) & (t_all >= s)).sum()),
        ]
    cols = [f"op_h0_hind_{c}" for c in operational_hind.HIND]
    return pd.DataFrame.from_dict(out, orient="index", columns=cols).rename_axis("mvt_id").astype("float32")


def test_counts_match_brute_force(monkeypatch):
    mvt, offblocks = hind_world()
    got = build(mvt, offblocks, monkeypatch)
    want = brute(mvt, offblocks).reindex(got.index)
    pd.testing.assert_frame_equal(got, want, check_exact=True)
    # the frame exercises what it should: windows with traffic, capped windows, rows without a window
    assert (got["op_h0_hind_dep_apt"] > 0).mean() > 0.5 and (got["op_h0_hind_dep_taxiing"] > 0).mean() > 0.5
    assert got.isna().all(axis=1).sum() >= 40
    rank = mvt.loc[got.index, "split"].eq("rank").to_numpy()
    assert got.loc[rank].isna().all().all(), "no recorded off-block for ranking rows: NaN"


def test_window_is_capped_and_never_negative():
    start, back = operational_hind.own_taxi_window(np.array([0.0, 0.0, 100.0, np.nan]), np.array([600.0, 9000.0, 50.0, 10.0]))
    np.testing.assert_array_equal(start[:2], [0.0, 9000.0 - CAP])
    np.testing.assert_array_equal(back[:2], [600.0, CAP])
    assert np.isnan(start[2:]).all() and np.isnan(back[2:]).all()


def test_hindsight_is_real_moving_the_own_takeoff_moves_the_outputs(monkeypatch):
    """The opposite of T1: this block reads the own takeoff, so moving it later lengthens the counted window."""
    mvt, offblocks = hind_world()
    base = build(mvt, offblocks, monkeypatch)
    ok = base.notna().all(axis=1) & (_oplib.secs(mvt.loc[base.index, "MVT_TIME_UTC_mvt"])
                                     - offblocks.reindex(base.index).to_numpy() <= CAP - 1200)
    rows = base.index[ok.to_numpy()][::7]
    assert len(rows) >= 10
    moved = mvt.copy()
    moved.loc[rows, "MVT_TIME_UTC_mvt"] = moved.loc[rows, "MVT_TIME_UTC_mvt"] + pd.Timedelta(seconds=1200)
    after = build(moved, offblocks, monkeypatch)
    pd.testing.assert_frame_equal(after, brute(moved, offblocks).reindex(after.index), check_exact=True)
    changed = (after.loc[rows] != base.loc[rows]).any(axis=1)
    assert changed.mean() > 0.9, changed.mean()
    assert (after.loc[rows, "op_h0_hind_dep_apt"] >= base.loc[rows, "op_h0_hind_dep_apt"]).all()


def test_the_block_is_quarantined_to_its_lab_rung():
    spec = all_specs()[BLOCK]
    assert spec.group == GROUP and not spec.uses_target and set(_oplib.LAB_EXTERNAL) <= set(spec.external)
    assert BLOCK not in {s.name for s in resolve(["group:op_lab_h0"])}, "the hindsight block needs its own group"
    resolving = []
    for path in sorted(paths.CONFIGS.glob("*.yaml")):
        cfg = load_config(path.name)
        blocks = {s.name for s in resolve(cfg.features.include, cfg.features.exclude)}
        if BLOCK not in blocks:
            continue
        resolving.append(path.name)
        assert not path.name.startswith("op_h1_"), f"{path.name}: an operational rung resolves the hindsight block"
        assert not cfg.model.refit_on_all, f"{path.name}: a refit/board config resolves the hindsight block"
        assert not any(b.startswith("op_h1_") for b in blocks), path.name
    assert resolving == ["op_h0_r5_hind.yaml"], resolving


def test_the_hindsight_rung_is_op_h0_r5_plus_the_block():
    h0 = {s.name for s in resolve(load_config("op_h0_r5.yaml").features.include)}
    cfg = load_config("op_h0_r5_hind.yaml")
    hind = {s.name for s in resolve(cfg.features.include, cfg.features.exclude)}
    assert hind - h0 == {BLOCK} and h0 <= hind
    assert not cfg.model.refit_on_all and cfg.name == "op_h0_r5_hind"


def test_no_import_time_work_and_helpers_are_hashed():
    from prc.features.build import _related_modules

    names = {m.__name__ for m in _related_modules(operational_hind)}
    assert {"prc.features._oplib", "prc.features.congestion"} <= names
    src = inspect.getsource(operational_hind)
    body = src.split('"""', 2)[2]
    assert not [ln for ln in body.splitlines() if re.match(r"^[a-z_]+\(", ln)]
    # the block reads the own off-block only through the loader, and never outputs the window length (the target)
    assert "read_parquet" not in src and "TARGET_FILE" not in src
    assert all(not c.endswith(("_s", "_sec", "_min", "_len")) for c in all_specs()[BLOCK].outputs)


@pytest.mark.parametrize("name", ["op_h1_r4_mae.yaml"])
def test_the_mae_rung_is_op_h1_r4_with_an_mae_loss(name):
    cfg, base = load_config(name), load_config("op_h1_r4.yaml")
    assert cfg.model.params["loss_function"] == "MAE" and cfg.model.params["eval_metric"] == "MAE"
    assert not cfg.model.refit_on_all and not cfg.evaluation.permutation.enabled
    assert {k: v for k, v in dict(cfg.model.params).items() if k not in ("loss_function", "eval_metric")} == dict(base.model.params)
    assert [s.name for s in resolve(cfg.features.include)] == [s.name for s in resolve(base.features.include)]
