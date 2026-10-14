"""The historical ADS-B archive (`prc.features.adsb_archive`, builder `prc.adsb.archive`): rules, leakage, builder.

``op_adsb_archive`` turns the lab's per-flight off-block observations of PAST departures (lab/ideas/stand_archive.py
pass 1) into a lookup table per stand, pier, stand x runway and runway. Asserted here, on synthetic data:

* the per-flight rules (`flight_values`): push / power-out / unknown, the power-out share only at matched stands,
  push duration, engine-start pause, time to 8 kt and to the radii, every range on both sides;
* the table against brute force: medians, the power-out share and the shrinkage chains, with missing stands and
  runways, month-out per query; no support count is an output;
* leakage: only fit rows' records enter (records of holdout days or ranking rows can change freely), a fit row's own
  record never enters what it reads (month-out by the schedule's month; moving or blanking every IOBT changes
  nothing), fit records do enter, and the fit mask matters;
* the builder end to end: per-day observation tables and the pushback table written to disk and consolidated; a
  holdout day's table changed on disk leaves the block bit-identical, a fit day's table changes it; the manifest's
  digests and the guard against records of the scored period.

Synthetic only: no competition data, no ADS-B crawl, no network.
"""

from __future__ import annotations

import hashlib
import inspect
import json

import numpy as np
import pandas as pd
import pytest

from prc import paths
from prc.adsb import archive as builder
from prc.features import _horizon, _oplib, adsb_archive, all_specs
from prc.features.build import compute_block
from prc.features.registry import FeatureContext
from test_operational import assert_identical, synthetic_world

BLOCK = "op_adsb_archive"
INPUT = adsb_archive.INPUT_COLUMNS
TAUS = tuple(f"tau_{r}" for r in adsb_archive.RADII)


# --- synthetic archive input --------------------------------------------------------------------------

def _stand_effect(stand: pd.Series) -> np.ndarray:
    """A deterministic per-stand offset (str hashes are salted per process)."""
    return stand.astype("string").fillna("").map(lambda s: 20.0 * (sum(map(ord, s)) % 9)).to_numpy(dtype="float64")


def synthetic_archive(mvt: pd.DataFrame, seed: int = 4, share: float = 0.7) -> pd.DataFrame:
    """Archive input rows (the `RAW` columns, ``day``) for ``share`` of the departures of both splits.

    Every case of the rules occurs: pushes at matched and unmatched stands, power-outs, unknown off-blocks, timings out
    of range on both sides, stray values on rows whose kind does not use them, missing profile radii.
    """
    rng = np.random.default_rng(seed)
    dep = mvt.loc[mvt["is_dep"].to_numpy(dtype=bool)]
    ids = dep.index[rng.random(len(dep)) < share]
    d = dep.loc[ids]
    n = len(ids)
    eff = _stand_effect(d["STAND_mvt"])
    kind = rng.choice(["push", "power", "none"], n, p=[0.5, 0.2, 0.3])
    push, power = kind == "push", kind == "power"
    seen = push | power
    push_dur = np.where(push, 30.0 + eff + rng.integers(0, 200, n), np.nan)
    push_dur[push & (rng.random(n) < 0.03)] = 1200.0
    push_dur[push & (rng.random(n) < 0.02)] = 0.0
    push_dur[power & (rng.random(n) < 0.3)] = 77.0  # stray: a power-out has no push
    es_dur = np.where(push, 100.0 + eff + rng.integers(0, 200, n), np.nan)
    es_dur[push & (rng.random(n) < 0.03)] = 1000.0
    start_8kt = np.where(push, 200.0 + eff + rng.integers(0, 300, n), np.where(power, 10.0 + rng.integers(0, 90, n), np.nan))
    start_8kt[seen & (rng.random(n) < 0.03)] = 2500.0
    tau_100 = np.where(push, 80.0 + eff, 20.0) + rng.integers(0, 120, n)
    tau_300 = tau_100 + rng.integers(60, 300, n)
    tau_1000 = tau_300 + rng.integers(100, 400, n)
    tau_1000[rng.random(n) < 0.03] = 5000.0
    taus = {"tau_100": tau_100, "tau_300": tau_300, "tau_1000": tau_1000}
    for name, t in taus.items():
        t[~seen | (rng.random(n) < 0.1)] = np.nan
    cluster = np.where(power, rng.choice([10.0, 60.0], n), rng.choice([10.0, 60.0, np.nan], n, p=[0.6, 0.3, 0.1]))
    frame = pd.DataFrame(
        {
            "apt": d["apt"].astype(str).to_numpy(),
            "kind": np.where(seen, kind, None),
            "push_dur": push_dur,
            "es_dur": es_dur,
            "start_8kt": start_8kt,
            **taus,
            "pb_cl_dist_stand": np.where(rng.random(n) < 0.05, np.nan, cluster),
            "day": d["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d").to_numpy(),
        },
        index=pd.Index(ids, name="mvt_id"),
    )
    frame.loc[frame["kind"].eq("power").to_numpy(dtype=bool), "pb_cl_dist_stand"] = 10.0  # the lab rule: a power-out needs a matched stand
    frame["kind"] = frame["kind"].astype("string")
    frame["apt"] = frame["apt"].astype("string")
    return frame


def archive_world(days: int = 60, per_day: int = 40):
    """A frame over three months (20 Mar - 18 May 2025) and its archive; April is the holdout month.

    fit = training departures outside April (the seasonal holdout's shape: queries of a fit month read the table
    without their month, April and ranking queries read the full one).
    """
    mvt, _ = synthetic_world(days=days, per_day=per_day)
    raw = synthetic_archive(mvt)
    dep = _oplib.departures(mvt)
    month = pd.Series(dep["MVT_TIME_UTC_mvt"].dt.month.to_numpy(), index=dep.index)
    train = dep["split"].eq("train").to_numpy()
    fit = pd.Series(False, index=mvt.index)
    fit.loc[dep.index[train & (month != 4).to_numpy()]] = True
    return mvt, raw, fit


def build_archive(mvt: pd.DataFrame, raw: pd.DataFrame, fit: pd.Series, monkeypatch, dep_index=None,
                  params: dict | None = None) -> pd.DataFrame:
    monkeypatch.setattr(adsb_archive, "load_flights", lambda: raw)
    dep_index = mvt.index[mvt["is_dep"].to_numpy()] if dep_index is None else dep_index
    ctx = FeatureContext(dep_index=dep_index, params=dict(params or {}), fit_mask=fit,
                         target=pd.Series(0.0, index=dep_index))
    return compute_block(all_specs()[BLOCK], mvt, ctx)


# --- per-flight rules ------------------------------------------------------------------------------------

def _rule_rows() -> pd.DataFrame:
    push = {"apt": "EDDF", "kind": "push", "push_dur": 120.0, "es_dur": 200.0, "start_8kt": 400.0, "tau_100": 150.0,
            "tau_300": 350.0, "tau_1000": 500.0, "pb_cl_dist_stand": 30.0}
    power = {**push, "kind": "power", "push_dur": np.nan, "es_dur": np.nan, "start_8kt": 40.0, "tau_100": 30.0,
             "tau_300": 80.0, "tau_1000": 190.0}
    cases = {
        "push": push,
        "push_unmatched": {**push, "pb_cl_dist_stand": np.nan},
        "push_dur_zero": {**push, "push_dur": 0.0},
        "push_dur_at_cap": {**push, "push_dur": 900.0},
        "push_dur_long": {**push, "push_dur": 901.0},
        "es_zero": {**push, "es_dur": 0.0},
        "es_long": {**push, "es_dur": 900.5},
        "es_negative": {**push, "es_dur": -1.0},
        "start_at_cap": {**push, "start_8kt": 1800.0},
        "start_long": {**push, "start_8kt": 1801.0},
        "tau_at_cap": {**push, "tau_1000": 3600.0},
        "tau_long": {**push, "tau_1000": 3601.0},
        "tau_negative": {**push, "tau_100": -0.5},
        "tau_missing": {**push, "tau_300": np.nan},
        "power": power,
        "power_unmatched": {**power, "pb_cl_dist_stand": np.nan},
        "power_stray_push": {**power, "push_dur": 50.0, "es_dur": 60.0},
        "unknown": {**push, "kind": None},
        "unknown_kind_text": {**push, "kind": "towed"},
    }
    frame = pd.DataFrame.from_dict(cases, orient="index")
    frame.index.name = "mvt_id"
    frame["kind"] = frame["kind"].astype("string")
    return frame


def test_flight_values_follow_the_rules():
    v = adsb_archive.flight_values(_rule_rows())
    assert list(v.columns) == list(adsb_archive.FLIGHT)
    nan = np.nan
    want = {  # power, push_dur_s, es_dur_s, start_8kt_s, tau_100_s, tau_300_s, tau_1000_s
        "push": (0, 120, 200, 400, 150, 350, 500),
        "push_unmatched": (nan, 120, 200, 400, 150, 350, 500),  # timings kept, no classification
        "push_dur_zero": (0, nan, 200, 400, 150, 350, 500),
        "push_dur_at_cap": (0, 900, 200, 400, 150, 350, 500),
        "push_dur_long": (0, nan, 200, 400, 150, 350, 500),
        "es_zero": (0, 120, 0, 400, 150, 350, 500),
        "es_long": (0, 120, nan, 400, 150, 350, 500),
        "es_negative": (0, 120, nan, 400, 150, 350, 500),
        "start_at_cap": (0, 120, 200, 1800, 150, 350, 500),
        "start_long": (0, 120, 200, nan, 150, 350, 500),
        "tau_at_cap": (0, 120, 200, 400, 150, 350, 3600),
        "tau_long": (0, 120, 200, 400, 150, 350, nan),
        "tau_negative": (0, 120, 200, 400, nan, 350, 500),
        "tau_missing": (0, 120, 200, 400, 150, nan, 500),
        "power": (1, nan, nan, 40, 30, 80, 190),
        "power_unmatched": (nan, nan, nan, 40, 30, 80, 190),
        "power_stray_push": (1, nan, nan, 40, 30, 80, 190),  # a power-out has no push and no engine-start pause
        "unknown": (nan,) * 7,
        "unknown_kind_text": (nan,) * 7,
    }
    for case, values in want.items():
        got = v.loc[case].to_numpy(dtype="float64")
        np.testing.assert_array_equal(got, np.asarray(values, dtype="float64"), err_msg=case)


def test_no_support_count_or_whole_taxi_is_an_output():
    """Counts would name a fit row's month (month-out tables) and sit beyond every fit value at serving time; the whole
    taxi-out is a lower-support copy of the target statistics. Neither is an output."""
    assert not [c for c in adsb_archive.ARCH if c.endswith(("_n", "_n_push", "_count")) or "taxi" in c]
    assert set(adsb_archive.CHAINS) == {"stand", "pier", "sr", "rwy"}
    assert len(adsb_archive.ARCH) == len(adsb_archive.CHAINS) * len(adsb_archive.STATS)


# --- the table against brute force ----------------------------------------------------------------------

def schedule_month(dep: pd.DataFrame) -> np.ndarray:
    """The month-out key by hand: the calendar month (``12 * year + month - 1``, UTC) of the schedule, which is the
    long horizons' planned off-block (findings 10.35: no IOBT)."""
    sched = dep["SCHED_TIME_UTC_mvt"]
    return (sched.dt.year * 12 + sched.dt.month - 1).to_numpy("float64")


def brute_archive(dep: pd.DataFrame, values: pd.DataFrame, fit: np.ndarray, q_ids, k: float = adsb_archive.SHRINK_K):
    """The archive of each query by plain masks: records of fit rows, month-out, medians and shrinkage by hand."""
    v = values.reindex(dep.index)
    month = schedule_month(dep)
    has = v[list(adsb_archive.FLIGHT)].notna().any(axis=1).to_numpy()
    rec = np.asarray(fit, dtype=bool) & has & np.isfinite(month)
    fit_months = set(month[rec].tolist())
    apt = dep["apt"].astype(str).to_numpy()
    stand_s, rwy_s = dep["STAND_mvt"].astype("string"), dep["RUNWAY_mvt"].astype("string")
    s_ok, r_ok = stand_s.notna().to_numpy(), rwy_s.notna().to_numpy()
    stand, rwy = stand_s.fillna("").to_numpy(dtype=object), rwy_s.fillna("").to_numpy(dtype=object)
    pier = _oplib.pier_key(dep).astype(str).to_numpy()
    cols = {c: v[c].to_numpy(dtype="float64") for c in adsb_archive.FLIGHT}
    rows = []
    for qid in q_ids:
        i = dep.index.get_loc(qid)
        use = rec & (month != month[i]) if month[i] in fit_months else rec
        same_apt = apt == apt[i]
        masks = {
            "apt": same_apt,
            "pier": (pier == pier[i]) & s_ok if s_ok[i] else None,
            "stand": same_apt & (stand == stand[i]) & s_ok if s_ok[i] else None,
            "rwy": same_apt & (rwy == rwy[i]) & r_ok if r_ok[i] else None,
            "pier_rwy": (pier == pier[i]) & (rwy == rwy[i]) & s_ok & r_ok if s_ok[i] and r_ok[i] else None,
            "stand_rwy": same_apt & (stand == stand[i]) & (rwy == rwy[i]) & s_ok & r_ok if s_ok[i] and r_ok[i] else None,
        }

        def stat(lvl, col, how):
            m = masks[lvl]
            if m is None:
                return np.nan, 0
            x = cols[col][use & m]
            x = x[np.isfinite(x)]
            if not len(x):
                return np.nan, 0
            return (float(np.median(x)) if how == "median" else float(np.mean(x))), len(x)

        out = {}
        for level, chain in adsb_archive.CHAINS.items():
            for name, col, how in adsb_archive.STATS:
                value = None
                for lvl in chain:
                    raw, n = stat(lvl, col, how)
                    if value is None:
                        value = raw
                    elif np.isfinite(raw):
                        value = raw if not np.isfinite(value) else (n * raw + k * value) / (n + k)
                out[f"{level}_{name}"] = value
        rows.append(out)
    return pd.DataFrame(rows, index=pd.Index(q_ids))[list(adsb_archive.ARCH)].astype("float64")


def test_archive_matches_brute_force_month_out():
    mvt, raw, fit = archive_world()
    dep = _oplib.departures(mvt)
    values = adsb_archive.flight_values(raw)
    fit_arr = fit.reindex(dep.index).to_numpy()
    rng = np.random.default_rng(1)
    month = dep["MVT_TIME_UTC_mvt"].dt.month.to_numpy()
    rank = dep["split"].eq("rank").to_numpy()
    # queries of every kind: fit months (March, May), the holdout month, ranking rows, no stand, no runway
    picks = [rng.choice(dep.index[(month == m) & ~rank], 25, replace=False) for m in (3, 4, 5)]
    picks.append(rng.choice(dep.index[rank], 10, replace=False))
    picks.append(dep.index[dep["STAND_mvt"].isna().to_numpy()][:5])
    picks.append(dep.index[dep["RUNWAY_mvt"].isna().to_numpy()][:5])
    q = pd.Index(np.unique(np.concatenate(picks)), name="mvt_id")
    assert dep.loc[q, "STAND_mvt"].isna().any() and dep.loc[q, "RUNWAY_mvt"].isna().any()
    np.testing.assert_array_equal(_horizon.plan_month(dep), schedule_month(dep))  # the block's key is the schedule's month
    got = adsb_archive.archive_at(dep, q, values, fit_arr).astype("float64")
    want = brute_archive(dep, values, fit_arr, q)
    np.testing.assert_allclose(got.to_numpy(), want.to_numpy(), rtol=2e-6, atol=1e-3, equal_nan=True)
    # something is measured, and the levels differ
    for col in ("stand_push_dur_s", "stand_tau_300_s", "pier_es_dur_s", "sr_start_8kt_s", "rwy_tau_1000_s"):
        assert got[col].notna().mean() > 0.9 and got[col].std() > 0, col
    assert got["stand_power_share"].between(0, 1).all() and got["stand_power_share"].std() > 0
    assert not np.allclose(got["stand_push_dur_s"], got["pier_push_dur_s"], equal_nan=True)
    assert not np.allclose(got["stand_tau_300_s"], got["rwy_tau_300_s"], equal_nan=True)
    # the shrinkage constant is a parameter of the table
    loose = adsb_archive.archive_at(dep, q, values, fit_arr, shrink_k=1.0).astype("float64")
    np.testing.assert_allclose(loose.to_numpy(), brute_archive(dep, values, fit_arr, q, k=1.0).to_numpy(),
                               rtol=2e-6, atol=1e-3, equal_nan=True)


def test_unmatched_stand_reads_its_parent_power_share_not_zero():
    """At a stand whose records are all unmatched pushes, the power share comes from pier and airport (no fake 0)."""
    mvt, raw, fit = archive_world()
    dep = _oplib.departures(mvt)
    stand = dep["STAND_mvt"].astype("string")
    target = (dep["apt"].astype("string") == "EDDF") & (stand == "A3")
    moved = raw.copy()
    at = moved.index.isin(dep.index[target.fillna(False).to_numpy(dtype=bool)])
    moved.loc[at, "kind"] = "push"
    moved.loc[at, "pb_cl_dist_stand"] = np.nan
    values = adsb_archive.flight_values(moved)
    q = dep.index[target.fillna(False).to_numpy(dtype=bool)]  # every month key: fit months, holdout, ranking
    assert len(q) >= 20
    got = adsb_archive.archive_at(dep, q, values, fit.reindex(dep.index).to_numpy())
    # no classified record at the stand: its share is the pier's, not 0 (before the fix every one read a push)
    np.testing.assert_array_equal(got["stand_power_share"].to_numpy(), got["pier_power_share"].to_numpy())
    assert (got["stand_power_share"] > 0).all()
    assert got["stand_push_dur_s"].notna().all()  # the timings of unmatched pushes still count


def test_month_keys_match_the_nm_baseline_key():
    fit_months = {24_302, 24_304}
    months = np.array([24_302.0, 24_303.0, np.nan, 24_304.0, 0.0])
    got = _horizon.month_keys(months, fit_months)
    np.testing.assert_array_equal(got, [24_302, -1, -1, 24_304, -1])


# --- leakage --------------------------------------------------------------------------------------------

def test_block_contract_and_board_rows(monkeypatch):
    mvt, raw, fit = archive_world()
    X = build_archive(mvt, raw, fit, monkeypatch)
    dep_idx = mvt.index[mvt["is_dep"].to_numpy()]
    assert X.index.equals(dep_idx) and list(X.columns) == list(all_specs()[BLOCK].outputs)
    rank = mvt.loc[dep_idx, "split"].eq("rank").to_numpy()
    for col in ("op_arch_stand_power_share", "op_arch_stand_push_dur_s", "op_arch_pier_tau_300_s"):
        assert X.loc[rank, col].notna().mean() > 0.9, f"{col}: the table must exist for ranking rows"
    spec = all_specs()[BLOCK]
    assert spec.uses_target and "ctx.fit_mask" in inspect.getsource(spec.fn) and "ctx.target" not in inspect.getsource(spec.fn)
    assert spec.external == (paths.ADSB_ARCHIVE_FILE,)


def test_records_outside_the_fit_never_enter(monkeypatch):
    """Holdout-day and ranking records can change freely: the archive reads the fit rows' records only."""
    mvt, raw, fit = archive_world()
    base = build_archive(mvt, raw, fit, monkeypatch)
    outside = ~raw.index.isin(fit.index[fit.to_numpy()])
    assert outside.sum() > 500
    moved = raw.copy()
    moved.loc[outside, "kind"] = "power"
    moved.loc[outside, "pb_cl_dist_stand"] = 5.0
    for col in ("push_dur", "es_dur", "start_8kt", *TAUS):
        moved.loc[outside, col] = 11.0
    assert_identical(base, build_archive(mvt, moved, fit, monkeypatch))
    # removing them altogether changes nothing either
    assert_identical(base, build_archive(mvt, raw.loc[~outside], fit, monkeypatch))


def test_fit_records_enter_but_never_the_own_month(monkeypatch):
    """A fit row's own record (and its month) never enters what it reads; other months' fit records do."""
    mvt, raw, fit = archive_world()
    base = build_archive(mvt, raw, fit, monkeypatch)
    dep = _oplib.departures(mvt)
    month = pd.Series(schedule_month(dep), index=dep.index)
    may = raw.index[raw.index.isin(fit.index[fit.to_numpy()]) & raw.index.isin(month.index[month.eq(month.max())])]
    assert len(may) > 500
    moved = raw.copy()
    for col in ("push_dur", "start_8kt", *TAUS):
        moved.loc[may, col] = moved.loc[may, col] + 60.0
    moved.loc[may, "kind"] = "push"
    after = build_archive(mvt, moved, fit, monkeypatch)
    may_q = month.index[month.eq(month.max())]
    other_q = month.index[month.lt(month.max())]
    assert_identical(base.loc[may_q], after.loc[may_q])  # May queries read the table without May
    for col in ("op_arch_stand_push_dur_s", "op_arch_stand_tau_300_s", "op_arch_pier_power_share"):
        changed = ~np.isclose(base.loc[other_q, col], after.loc[other_q, col], equal_nan=True)
        assert changed.mean() > 0.5, col  # March, April and the ranking rows read May's records


def _iobt_month(frame: pd.DataFrame) -> np.ndarray:
    """The former month key (findings 10.31, until 11 Oct): the month of the IOBT, else of the schedule."""
    iobt = _oplib.secs(frame["IOBT_flt"])
    planned = np.where(np.isnan(iobt), _oplib.secs(frame["SCHED_TIME_UTC_mvt"]), iobt)
    when = pd.Series(pd.to_datetime(planned, unit="s", utc=True))
    return (when.dt.year * 12 + when.dt.month - 1).to_numpy("float64")


@pytest.mark.parametrize("mode", ["shifted", "blanked"])
def test_the_month_key_never_reads_the_iobt(monkeypatch, mode):
    """The month a query leaves out is the schedule's (findings 10.35): every departure's IOBT moved by 31 days, or
    blanked, leaves the block bit-identical (the moved rows are fit rows); under the former key (the IOBT's month)
    the same move does change it."""
    mvt, raw, fit = archive_world()
    dep_idx = mvt.index[mvt["is_dep"].to_numpy()]
    moved = mvt.copy()
    moved.loc[dep_idx, "IOBT_flt"] = moved.loc[dep_idx, "IOBT_flt"] + pd.Timedelta(days=31) if mode == "shifted" else pd.NaT
    assert fit.loc[dep_idx].sum() > 1000
    base = build_archive(mvt, raw, fit, monkeypatch)
    assert_identical(base, build_archive(moved, raw, fit, monkeypatch))
    if mode == "shifted":  # guard the guard
        monkeypatch.setattr(_horizon, "plan_month", _iobt_month)
        assert not build_archive(mvt, raw, fit, monkeypatch).equals(build_archive(moved, raw, fit, monkeypatch))


def test_the_fit_mask_matters(monkeypatch):
    mvt, raw, fit = archive_world()
    half = fit.copy()
    half.loc[half.index[half.to_numpy()][::2]] = False
    assert not build_archive(mvt, raw, fit, monkeypatch).equals(build_archive(mvt, raw, half, monkeypatch))
    nothing = pd.Series(False, index=mvt.index)
    assert build_archive(mvt, raw, nothing, monkeypatch).isna().all().all()


def test_queries_are_independent_of_the_query_set(monkeypatch):
    """A row's archive values do not depend on which other rows are queried (no transductive query statistics)."""
    mvt, raw, fit = archive_world()
    full = build_archive(mvt, raw, fit, monkeypatch)
    sub = full.index[::7]
    assert_identical(full.loc[sub], build_archive(mvt, raw, fit, monkeypatch, dep_index=sub))


# --- the builder end to end -------------------------------------------------------------------------------

def write_day_tables(obs, src, raw: pd.DataFrame, unobserved: pd.Index | None = None) -> None:
    """The lab's layout: stand_archive_obs/<day>.parquet per day (with columns the builder must not read), and
    features3.parquet over all days (with departures that have no observation)."""
    obs.mkdir(parents=True, exist_ok=True)
    src.mkdir(parents=True, exist_ok=True)
    for day, g in raw.groupby("day"):
        g[list(INPUT["obs"])].assign(stand_mvt="X1", off_est=600.0, tau_10=5.0, seen_lead=30.0).to_parquet(obs / f"{day}.parquet")
    f3 = raw[list(INPUT["features3"])].assign(pb_go_est=1.0, pb_es_dur=2.0)
    if unobserved is not None:
        extra = pd.DataFrame({"pb_cl_dist_stand": 3.0, "pb_go_est": 1.0, "pb_es_dur": 2.0},
                             index=pd.Index(unobserved + 10**9, name="mvt_id"))
        f3 = pd.concat([f3, extra])
    f3.to_parquet(src / "features3.parquet")


def test_consolidate_reads_exactly_the_input_columns(tmp_path):
    mvt, raw, _ = archive_world(days=20, per_day=20)
    obs, src = tmp_path / "obs", tmp_path / "adsb"
    write_day_tables(obs, src, raw, unobserved=pd.Index([1, 2, 3]))
    days = builder.archive_days(obs)
    assert days == sorted(raw["day"].unique().tolist())
    got = builder.consolidate(obs, src)
    assert got.index.name == "mvt_id" and not got.index.has_duplicates
    assert set(adsb_archive.RAW) | {"day"} == set(got.columns)
    cols = list(adsb_archive.RAW) + ["day"]
    pd.testing.assert_frame_equal(got[cols], raw.sort_index()[cols], check_dtype=False)
    keep = raw.index[::3]
    assert builder.consolidate(obs, src, keep=keep).index.equals(keep.sort_values())
    # the same departure observed on two days is an error
    raw.iloc[:1][list(INPUT["obs"])].to_parquet(obs / "2099-01-01.parquet")
    with pytest.raises(ValueError, match="two days"):
        builder.consolidate(obs, src)
    (obs / "2099-01-01.parquet").unlink()
    # the pushback table is required
    (src / "features3.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="features3"):
        builder.consolidate(obs, src)


def _sha1(path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def test_builder_main_writes_the_file_and_a_manifest(tmp_path):
    _, raw, _ = archive_world(days=10, per_day=10)
    obs, src, out = tmp_path / "obs", tmp_path / "adsb", tmp_path / "out" / "arch.parquet"
    write_day_tables(obs, src, raw)
    assert builder.main(["--obs", str(obs), "--src", str(src), "--out", str(out), "--all-splits"]) == 0
    written = pd.read_parquet(out)
    assert len(written) == len(raw) and written.index.name == "mvt_id"
    manifest = json.loads(out.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["rows"] == len(raw) and manifest["splits"] == "all" and manifest["guard"].startswith("skipped")
    assert manifest["files_days"] == manifest["rows_days"] == raw["day"].nunique()
    assert manifest["rows_first_day"] == raw["day"].min() and manifest["rows_last_day"] == raw["day"].max()
    assert manifest["rows_fully_observed"] == int(raw["kind"].isin(["push", "power"]).sum())
    # provenance: the digests of every source and of the output
    days = builder.archive_days(obs)
    lines = "\n".join(f"{d}.parquet {_sha1(obs / f'{d}.parquet')}" for d in days)
    assert manifest["obs_files_sha1"] == hashlib.sha1(lines.encode("utf-8")).hexdigest()
    assert manifest["features3_sha1"] == _sha1(src / "features3.parquet") and manifest["out_sha1"] == _sha1(out)
    # a lab table rewritten in place changes the digest
    first = obs / f"{days[0]}.parquet"
    frame = pd.read_parquet(first)
    frame["push_dur"] = frame["push_dur"] + 1.0
    frame.to_parquet(first)
    assert builder.source_digests(obs, src, days)["obs_files_sha1"] != manifest["obs_files_sha1"]


def _splits(mvt: pd.DataFrame) -> pd.DataFrame:
    dep = _oplib.departures(mvt)
    return pd.DataFrame({"split": dep["split"].astype("string"), "MVT_TIME_UTC_mvt": dep["MVT_TIME_UTC_mvt"]}, index=dep.index)


def test_builder_keeps_training_rows_and_guards_the_scored_period(tmp_path, monkeypatch):
    mvt, raw, _ = archive_world(days=12, per_day=12)
    dep = _oplib.departures(mvt)
    splits = _splits(mvt)
    rank = splits["split"].eq("rank").to_numpy()
    splits.loc[rank, "MVT_TIME_UTC_mvt"] = splits.loc[rank, "MVT_TIME_UTC_mvt"] + pd.Timedelta(days=30)  # strictly later
    movements = pd.DataFrame({"is_dep": True, **{c: splits[c] for c in splits}}, index=splits.index)
    mfile = tmp_path / "movements.parquet"
    movements.to_parquet(mfile)
    monkeypatch.setattr(paths, "MOVEMENTS_FILE", mfile)
    obs, src, out = tmp_path / "obs", tmp_path / "adsb", tmp_path / "out" / "arch.parquet"
    write_day_tables(obs, src, raw)
    assert builder.main(["--obs", str(obs), "--src", str(src), "--out", str(out)]) == 0
    written = pd.read_parquet(out)
    train = dep.index[dep["split"].eq("train").to_numpy()]
    assert written.index.equals(raw.index[raw.index.isin(train)].sort_values())
    manifest = json.loads(out.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["splits"] == "train" and manifest["guard"]["ok"]
    # a kept departure at or after the first scored takeoff fails the build
    bad = splits.copy()
    bad.loc[written.index[:1], "MVT_TIME_UTC_mvt"] = bad.loc[rank, "MVT_TIME_UTC_mvt"].min()
    with pytest.raises(ValueError, match="guard"):
        builder.guard(written.index, bad)
    assert builder.guard(written.index, splits)["ok"]


def test_holdout_day_tables_change_nothing_fit_day_tables_do(tmp_path, monkeypatch):
    """On disk: rewrite a holdout day's table -> the block is bit-identical; rewrite a fit day's -> it moves."""
    mvt, raw, fit = archive_world()
    obs, src = tmp_path / "obs", tmp_path / "adsb"
    write_day_tables(obs, src, raw)

    def block():
        return build_archive(mvt, builder.consolidate(obs, src), fit, monkeypatch)

    base = block()
    april = sorted(d for d in raw["day"].unique() if d.startswith("2025-04"))
    for day in april:  # the holdout month: every one of its days rewritten
        f = pd.read_parquet(obs / f"{day}.parquet")
        f["kind"] = "power"
        f["start_8kt"] = 1.0
        f["tau_300"] = 2.0
        f.to_parquet(obs / f"{day}.parquet")
    assert_identical(base, block())
    fit_day = "2025-03-25"
    f = pd.read_parquet(obs / f"{fit_day}.parquet")
    f["kind"] = "push"
    f["push_dur"] = 800.0
    f.to_parquet(obs / f"{fit_day}.parquet")
    assert not base.equals(block())


def test_load_flights_validates_the_file(tmp_path, monkeypatch):
    _, raw, _ = archive_world(days=6, per_day=10)
    good = tmp_path / "good.parquet"
    raw.to_parquet(good)
    monkeypatch.setattr(paths, "ADSB_ARCHIVE_FILE", good)
    pd.testing.assert_frame_equal(adsb_archive.load_flights(), raw)
    bad = tmp_path / "bad.parquet"
    raw.drop(columns=["es_dur"]).to_parquet(bad)
    monkeypatch.setattr(paths, "ADSB_ARCHIVE_FILE", bad)
    with pytest.raises(ValueError, match="es_dur"):
        adsb_archive.load_flights()
    dup = tmp_path / "dup.parquet"
    pd.concat([raw, raw.iloc[:1]]).to_parquet(dup)
    monkeypatch.setattr(paths, "ADSB_ARCHIVE_FILE", dup)
    with pytest.raises(ValueError, match="twice"):
        adsb_archive.load_flights()


def test_the_builder_lives_outside_the_feature_package():
    """Feature code never imports prc.adsb (unhashed); the builder imports the column list from the feature module."""
    src = inspect.getsource(adsb_archive)
    assert "prc.adsb" not in src.replace("prc.adsb.archive", "")
    assert builder.INPUT_COLUMNS is adsb_archive.INPUT_COLUMNS
