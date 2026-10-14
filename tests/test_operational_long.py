"""The long-horizon operational models (`prc.features.operational_long`, `prc.features._horizon`): leakage and contracts.

A long-horizon prediction is made 3 h or 24 h before the planned off-block ``P``, the published schedule (findings
10.35), so these blocks must never see the flight's own takeoff, its own NM actuals, its own off-block record or NM's
last known off-block time (LOBT), nor any other flight's event after the anchor, nor the IOBT of any flight (it
equals LOBT on 99.45 % of the NM-matched 2025 departures and the data hold no filing time). Asserted, not assumed:

* T1 -- moving the flight's own takeoff (+600 s, -300 s, and to 60 s before each long anchor), its own AOBT_3 /
  ARVT_3, its own recorded off-block or its own LOBT leaves every long-horizon value (and the ADS-B archive)
  bit-identical, with the perturbed rows kept in the fit (the NM baseline and the archive are month-out);
* T1e -- the IOBT of every flight shifted (by 3 h, or by a different whole number of half-hours per flight), blanked
  or removed with its column leaves every long-horizon value and the archive bit-identical, the perturbed rows kept in
  the fit; under the former IOBT clock the same perturbations do move them (the guard has teeth). This replaces T2c
  (other flights' plans not yet filed by the anchor), whose object, the IOBT demand counts, no longer exists;
* T2 -- censoring every event after a query's anchor leaves its values bit-identical, at both leads;
* T2b -- censoring, in addition, the runway and stand of every other flight that moves after the anchor changes only
  the declared O* demand columns (`_horizon.O_STAR_FUTURE_ATTRIBUTES_LONG`), and does change them;
* the anchor is the schedule on every row, never the IOBT or LOBT; the lead-free blocks read plan data only;
* the primitives against brute force: symmetric demand windows with the own event on every edge, the demand block,
  the history block (windows, month-out baseline, self-exclusion), the TAF selection for (anchor, planned time);
* weather causality at both anchors: a METAR observed after ``a - METAR_LAG_S`` or a TAF issued after ``a`` changes
  nothing; a TAF issued at ``a`` that covers ``P`` counts, one that does not cover ``P`` is not used;
* the operational tiers (`prc.features._long_tiers`): swapping the flight's own runway moves only the columns tagged
  ``own_runway`` (and does move them), swapping its stand within the pier only the ``own_stand`` ones, across piers
  only ``own_stand`` / ``own_pier``; no explicit NM-join flag is an output;
* configs: the 3 h and 24 h ladders are cumulative and mirror each other (runway-aware: an O* upper bound); the
  archive rungs add exactly the archive; only the runway-blind deliverable candidates (r3 and r4 per lead, rule R)
  refit; every runway-blind rung drops exactly the ``own_runway`` columns, the pier-only rung exactly the
  ``own_runway`` and ``own_stand`` ones; each forward twin keeps exactly its rung's columns; no long config resolves
  an IOBT or day-of-year column.

Synthetic (the world of tests/test_operational.py). ``HORIZON_REAL=1`` adds T1 and T2 on a 50k-row slice of the real
movements (and the real archive input when it has been built).
"""

from __future__ import annotations

import inspect
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prc import paths
from prc.config import load_config
from prc.features import _horizon, _long_tiers, _oplib, adsb_archive, all_specs, operational_long, resolve, weather, weather_fcst
from prc.features._wxlib import _metar_by_station
from prc.features.build import compute_block
from prc.features.registry import FeatureContext
from test_adsb_archive import synthetic_archive
from test_operational import (
    RUNWAYS,
    UTC,
    _late_reports,
    _real_slice,
    _set,
    _shift,
    assert_identical,
    censor,
    censor_attributes,
    spaced_departures,
    synthetic_weather,
    synthetic_world,
)

LEADS = _horizon.LEADS_S
FREE_BLOCKS = ("op_hl_static", "op_hl_plan", "op_hl_demand")
LEAD_BLOCKS = {h: (f"op_{h}_wx", f"op_{h}_hist") for h in LEADS}
LONG_BLOCKS = FREE_BLOCKS + LEAD_BLOCKS["hl3"] + LEAD_BLOCKS["hl24"]
ALL_BLOCKS = LONG_BLOCKS + ("op_adsb_archive",)


@pytest.fixture
def long_world(tmp_path, monkeypatch):
    """The synthetic world with its weather, its off-block records and an ADS-B archive input (all monkeypatched)."""
    mvt, offblocks = synthetic_world()
    metar, tafs = synthetic_weather(tmp_path)
    monkeypatch.setattr(weather, "load_metar", lambda: metar)
    monkeypatch.setattr(weather_fcst, "load_tafs", lambda: tafs)
    state = {"offblocks": offblocks, "archive": synthetic_archive(mvt)}
    monkeypatch.setattr(_oplib, "load_offblocks", lambda: state["offblocks"])
    monkeypatch.setattr(adsb_archive, "load_flights", lambda: state["archive"])
    return mvt, offblocks, state


def fit_rows(mvt: pd.DataFrame) -> pd.Series:
    """The fit mask of training: every training departure (perturbed rows included)."""
    return pd.Series(mvt["split"].eq("train").to_numpy() & mvt["is_dep"].to_numpy(), index=mvt.index)


def build(mvt: pd.DataFrame, names, dep_index: pd.Index, fit_mask: pd.Series | None = None) -> pd.DataFrame:
    """Run the named blocks through the contract validator, side by side."""
    specs = all_specs()
    fit_mask = fit_rows(mvt) if fit_mask is None else fit_mask
    parts = []
    for name in names:
        spec = specs[name]
        ctx = FeatureContext(dep_index=dep_index, fit_mask=fit_mask if spec.uses_target else None,
                             target=pd.Series(0.0, index=dep_index) if spec.uses_target else None)
        parts.append(compute_block(spec, mvt, ctx))
    return pd.concat(parts, axis=1)


def t1_rows(mvt: pd.DataFrame, hours: float = 52.0) -> pd.Index:
    """Training departures at least ``hours`` apart per airport (more than the widest reach, 24 h lead + 24 h window),
    of ONE planned month (the NM baseline and the archive are month-out, so perturbed flights must share a month)."""
    rows = spaced_departures(mvt, hours)
    month = _horizon.plan_month(mvt.loc[rows])
    top = pd.Series(month).mode().iloc[0]
    return rows[month == top]


# --- the anchor ---------------------------------------------------------------------------------------------

def _schedule_seconds(frame: pd.DataFrame) -> np.ndarray:
    """The schedule in whole epoch seconds, computed here without the module's helpers."""
    sched = frame["SCHED_TIME_UTC_mvt"]
    return np.floor((sched - pd.Timestamp("1970-01-01", tz=UTC)).dt.total_seconds().to_numpy(dtype="float64"))


def test_planned_offblock_is_the_schedule_never_iobt_or_lobt():
    """P is the schedule on every row: where the IOBT equals it, where it differs, where it is NaN (findings 10.35)."""
    mvt, _ = synthetic_world(days=4)
    dep = _oplib.departures(mvt)
    p = _horizon.planned_offblock_long(dep)
    iobt, sched = _oplib.secs(dep["IOBT_flt"]), _schedule_seconds(dep)
    np.testing.assert_array_equal(p, sched)
    assert np.isfinite(p).all()
    differs, missing = np.isfinite(iobt) & (iobt != sched), np.isnan(iobt)
    assert differs.sum() > 50 and missing.sum() > 5 and (np.isfinite(iobt) & (iobt == sched)).sum() > 50
    np.testing.assert_array_equal(p[differs], sched[differs])
    np.testing.assert_array_equal(p[missing], sched[missing])
    sched_ts = dep["SCHED_TIME_UTC_mvt"]
    np.testing.assert_array_equal(_horizon.plan_month(dep), (sched_ts.dt.year * 12 + sched_ts.dt.month - 1).to_numpy("float64"))
    # neither the IOBT nor NM's last known off-block time matters: moved, removed, present where the other is missing
    moved = dep.copy()
    moved["LOBT_flt"] = moved["LOBT_flt"] + pd.Timedelta(hours=2)
    moved["IOBT_flt"] = moved["IOBT_flt"] + pd.Timedelta(hours=5)
    moved.loc[dep.index[missing], "LOBT_flt"] = moved.loc[dep.index[missing], "SCHED_TIME_UTC_mvt"] + pd.Timedelta(minutes=40)
    moved.loc[dep.index[differs][:20], "IOBT_flt"] = pd.NaT
    for frame in (moved, dep.drop(columns=["IOBT_flt", "LOBT_flt"])):
        np.testing.assert_array_equal(_horizon.planned_offblock_long(frame), p)
        np.testing.assert_array_equal(_horizon.plan_month(frame), _horizon.plan_month(dep))
    for h, lead in LEADS.items():
        np.testing.assert_array_equal(_horizon.anchor_long(dep, lead), p - lead)
    assert LEADS == {"hl3": 10_800.0, "hl24": 86_400.0}


# --- the blocks' contract -------------------------------------------------------------------------------------

def test_long_blocks_build_on_the_synthetic_world(long_world):
    mvt, _, _ = long_world
    dep_idx = mvt.index[mvt["is_dep"].to_numpy()]
    X = build(mvt, ALL_BLOCKS, dep_idx)
    assert X.index.equals(dep_idx)
    specs = all_specs()
    assert list(X.columns) == [c for b in ALL_BLOCKS for c in specs[b].outputs]
    for col in ("op_hl_hour_local", "op_hl_has_stand", "op_hl_dem_dep_sched_apt_w60", "op_hl_dem_arr_sched_rwy_w30",
                "op_hl_dem_dep_sched_pier_w60", "op_hl3_wx_vis_m", "op_hl3_taf_vis_m", "op_hl24_wx_deice_min_6h",
                "op_hl3_hist_apt_b3h_anom_mean", "op_hl3_hist_rwy_b24h_n", "op_hl24_hist_apt_b24h_push_delay_mean",
                "op_arch_stand_push_dur_s", "op_arch_sr_power_share", "op_arch_pier_tau_300_s"):
        v = X[col].to_numpy(dtype="float64")
        assert np.isfinite(v).mean() > 0.3 and np.nanstd(v) > 0, col
    assert np.isfinite(X["op_hl24_taf_vis_m"]).mean() > 0.05  # a 25 h TAF reaches a day ahead only when just issued
    # the calendar is the schedule's (its hour is the scheduled hour) and has no day of year (findings 10.35)
    np.testing.assert_array_equal(X["op_hl_hour_local"].to_numpy(), X["op_hl_sched_hour_local"].to_numpy())
    assert not [c for c in X.columns if "day_of_year" in c or "iobt" in c]
    rank = mvt.loc[dep_idx, "split"].eq("rank").to_numpy()
    for col in ("op_hl_dem_dep_sched_apt_w60", "op_hl3_wx_vis_m", "op_hl24_hist_apt_b24h_n", "op_arch_stand_push_dur_s"):
        assert X.loc[rank, col].notna().all(), f"{col} must exist for ranking rows (board-legal)"
    # the lead-dependent blocks really differ between the leads
    assert not np.allclose(X["op_hl3_hist_apt_b3h_n"], X["op_hl24_hist_apt_b3h_n"])


def test_no_import_time_work_and_helpers_are_hashed():
    """`build._related_modules` is one level deep: every helper module must sit in the blocks' namespaces."""
    from prc.features.build import _related_modules

    names = {m.__name__ for m in _related_modules(operational_long)}
    for mod in ("_horizon", "_oplib", "_proxylib", "_wxlib", "congestion", "core", "stand_runway", "temporal",
                "weather", "weather_fcst"):
        assert f"prc.features.{mod}" in names, mod
    names = {m.__name__ for m in _related_modules(adsb_archive)}
    for mod in ("_horizon", "_oplib", "core", "stand_runway"):
        assert f"prc.features.{mod}" in names, mod
    for module in (operational_long, adsb_archive):
        body = inspect.getsource(module).split('"""', 2)[2]
        assert not [ln for ln in body.splitlines() if re.match(r"^[a-z_]+\(", ln)], module.__name__


def test_target_using_long_blocks_are_declared():
    for name in ALL_BLOCKS:
        spec = all_specs()[name]
        src = inspect.getsource(spec.fn)
        assert ("ctx.target" in src or "ctx.fit_mask" in src) == spec.uses_target, name
        assert "ctx.target" not in src, f"{name} must never read the target"
    assert {n for n in ALL_BLOCKS if all_specs()[n].uses_target} == {"op_hl3_hist", "op_hl24_hist", "op_adsb_archive"}


H1_HELPERS = ("planned_offblock", "baseline_month", "anchor_h1", "anchor_h0", "nm_baseline", "nm_anomaly", "plan_at",
              "wx_at", "traffic_at", "nm_at", "feed_at")
"""`_oplib` helpers built on the H1 clock (IOBT, else LOBT, else schedule) or the H0 records: never called here."""


def test_no_own_only_or_last_plan_columns_are_read():
    """Code (docstrings and comments stripped) of the new modules never names the IOBT, LOBT, EOBT_1, ARVT_3 or a block
    time, and never calls an `_oplib` helper of the H1 clock (IOBT, else LOBT, else schedule) or of H0."""
    root = Path(operational_long.__file__).parent
    for name in ("_horizon.py", "operational_long.py", "adsb_archive.py"):
        src = (root / name).read_text(encoding="utf-8")
        code = re.sub(r'"""[\s\S]*?"""', "", src)
        code = "\n".join(ln.split("#", 1)[0] for ln in code.splitlines())
        for col in ("IOBT_flt", "LOBT_flt", "EOBT_1_flt", "ARVT_3_flt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt",
                    "load_offblocks"):
            assert col not in code, (name, col)
        calls = [h for h in H1_HELPERS if re.search(rf"\b{h}\(", code)]
        assert not calls, (name, calls)
    assert "MVT_TIME_UTC_mvt" not in re.sub(r'"""[\s\S]*?"""', "", (root / "operational_long.py").read_text(encoding="utf-8"))


def test_no_long_block_is_lab_only():
    for name in ALL_BLOCKS:
        spec = all_specs()[name]
        assert paths.TARGET_FILE not in spec.external and not name.endswith("_lab"), name


# --- T1: own events --------------------------------------------------------------------------------------------

def _anchor(mvt: pd.DataFrame, rows: pd.Index, horizon: str) -> np.ndarray:
    return _horizon.anchor_long(mvt.loc[rows], LEADS[horizon])


@pytest.mark.parametrize("mode", ["plus600", "minus300", "cross_hl3", "cross_hl24", "far_before"])
def test_t1_own_takeoff_invariance(long_world, mode):
    mvt, _, _ = long_world
    rows = t1_rows(mvt)
    assert len(rows) >= 8
    fit = fit_rows(mvt)
    assert fit.loc[rows].all()
    base = build(mvt, ALL_BLOCKS, rows, fit)
    if mode == "plus600":
        moved = _shift(mvt, "MVT_TIME_UTC_mvt", rows, 600)
    elif mode == "minus300":
        moved = _shift(mvt, "MVT_TIME_UTC_mvt", rows, -300)
    elif mode.startswith("cross_"):  # the own takeoff crosses the long anchor: it now lies 60 s before it
        moved = _set(mvt, "MVT_TIME_UTC_mvt", rows, _anchor(mvt, rows, mode.removeprefix("cross_")) - 60)
    else:  # deep inside the 24 h history window of the 24 h anchor
        moved = _set(mvt, "MVT_TIME_UTC_mvt", rows, _anchor(mvt, rows, "hl24") - 6 * 3600)
    assert_identical(base, build(moved, ALL_BLOCKS, rows, fit))


def test_t1_crossing_case_does_put_the_own_takeoff_into_the_windows(long_world):
    """Guard the guard: without self-exclusion the crossing case would change the history counts."""
    mvt, _, _ = long_world
    rows = t1_rows(mvt)
    a = _anchor(mvt, rows, "hl3")
    moved = _set(mvt, "MVT_TIME_UTC_mvt", rows, a - 60)
    dep = _oplib.departures(moved)
    t = pd.Series(_oplib.secs(dep["MVT_TIME_UTC_mvt"]), index=dep.index)
    g = dep["apt"].astype("string")
    q_g = moved.loc[rows, "apt"].astype("string")
    raw = _oplib.backward_count(t, g, a, q_g, LEADS["hl3"])
    excl = _oplib.backward_count(t, g, a, q_g, LEADS["hl3"], t.reindex(rows).to_numpy())
    assert (raw - excl == 1).all()


@pytest.mark.parametrize("shift", [600, -600, "before_hl24"])
def test_t1b_own_nm_actuals_invariance(long_world, shift):
    mvt, _, _ = long_world
    rows = t1_rows(mvt)
    fit = fit_rows(mvt)
    base = build(mvt, ALL_BLOCKS, rows, fit)
    if shift == "before_hl24":
        a = _anchor(mvt, rows, "hl24")
        moved = _set(_set(mvt, "AOBT_3_flt", rows, a - 300), "ARVT_3_flt", rows, a - 200)
    else:
        moved = _shift(_shift(mvt, "AOBT_3_flt", rows, shift), "ARVT_3_flt", rows, shift)
    assert_identical(base, build(moved, ALL_BLOCKS, rows, fit))


@pytest.mark.parametrize("shift", [600, -600, "before_hl24"])
def test_t1c_own_offblock_invariance(long_world, shift):
    mvt, offblocks, state = long_world
    rows = t1_rows(mvt)
    fit = fit_rows(mvt)
    base = build(mvt, ALL_BLOCKS, rows, fit)
    moved = offblocks.copy()
    moved.loc[rows] = _anchor(mvt, rows, "hl24") - 120 if shift == "before_hl24" else moved.loc[rows] + shift
    state["offblocks"] = moved
    assert_identical(base, build(mvt, ALL_BLOCKS, rows, fit))


@pytest.mark.parametrize("mode", ["plus3600", "minus7200", "removed"])
def test_t1d_own_lobt_invariance(long_world, mode):
    """NM's last known off-block time is the last update of the plan, unknown hours ahead: the queries' own never
    matters (the reference side, every other row's, is `test_lobt_is_never_read_on_any_row`)."""
    mvt, _, _ = long_world
    rows = t1_rows(mvt)
    fit = fit_rows(mvt)
    base = build(mvt, ALL_BLOCKS, rows, fit)
    if mode == "removed":
        moved = _set(mvt, "LOBT_flt", rows, np.full(len(rows), np.nan))
    else:
        moved = _shift(mvt, "LOBT_flt", rows, 3600 if mode == "plus3600" else -7200)
    assert_identical(base, build(moved, ALL_BLOCKS, rows, fit))
    # the H1 clock does read it where the IOBT is missing (one reason the long clock is the schedule)
    gone = _set(mvt, "IOBT_flt", rows, np.full(len(rows), np.nan))
    assert not np.array_equal(_oplib.planned_offblock(_shift(gone, "LOBT_flt", rows, 3600).loc[rows]),
                              _oplib.planned_offblock(gone.loc[rows]))


def test_t1_own_archive_record_invariance(long_world):
    """A fit row's own ADS-B record is a near-copy of its taxi-out: changing it leaves its own archive values alone."""
    mvt, _, state = long_world
    rows = t1_rows(mvt)
    fit = fit_rows(mvt)
    raw = state["archive"]
    own = rows.intersection(raw.index)
    assert len(own) >= 4
    base = build(mvt, ("op_adsb_archive",), rows, fit)
    moved = raw.copy()
    moved.loc[own, "kind"] = "power"
    moved.loc[own, "pb_cl_dist_stand"] = 1.0
    moved.loc[own, "start_8kt"] = 1500.0
    moved.loc[own, "tau_300"] = 3000.0
    state["archive"] = moved
    assert_identical(base, build(mvt, ("op_adsb_archive",), rows, fit))


# --- T1e: the IOBT of every flight (findings 10.35) -------------------------------------------------------------------

IOBT_MODES = ("plus3h", "per_flight", "blanked", "removed")


def perturb_iobt(mvt: pd.DataFrame, mode: str) -> pd.DataFrame:
    """The IOBT of every departure shifted by 3 h, or by -6 h .. +6 h in whole half-hours per flight, blanked, or the
    column removed. A 3 h shift moves late plans across midnight and the March-April boundary of the synthetic world,
    so the planned day and month of an IOBT clock would move too."""
    dep_idx = mvt.index[mvt["is_dep"].to_numpy()]
    if mode == "plus3h":
        return _shift(mvt, "IOBT_flt", dep_idx, 3 * 3600)
    if mode == "per_flight":
        return _shift(mvt, "IOBT_flt", dep_idx, np.random.default_rng(22).integers(-12, 13, len(dep_idx)) * 1800)
    if mode == "blanked":
        return _set(mvt, "IOBT_flt", dep_idx, np.full(len(dep_idx), np.nan))
    assert mode == "removed", mode
    return mvt.drop(columns=["IOBT_flt"])


def _iobt_clock(frame: pd.DataFrame) -> np.ndarray:
    """The former long clock (findings 10.31, until 11 Oct): the IOBT, else the schedule. Only for the guards."""
    iobt = _oplib.secs(frame["IOBT_flt"])
    return np.where(np.isnan(iobt), _oplib.secs(frame["SCHED_TIME_UTC_mvt"]), iobt)


@pytest.mark.parametrize("mode", IOBT_MODES)
def test_t1e_iobt_of_every_flight_invariance(long_world, mode):
    """No long block and not the archive reads the IOBT, of the query or of any other flight: shifted, blanked or
    removed for EVERY departure, every departure queried and the perturbed training rows kept in the fit, every value
    is bit-identical. This replaces T2c, which censored the other flights' plans not yet filed by the anchor and had
    the IOBT demand counts (gone with the IOBT clock) as its object."""
    mvt, _, _ = long_world
    q = mvt.index[mvt["is_dep"].to_numpy()]
    fit = fit_rows(mvt)
    assert fit.loc[q].mean() > 0.75  # the training departures, perturbed like every other
    moved = perturb_iobt(mvt, mode)
    if mode != "removed":  # the perturbation is real: it moves an IOBT clock (blanked: wherever the IOBT was not the schedule)
        before, after = _iobt_clock(_oplib.departures(mvt)), _iobt_clock(_oplib.departures(moved))
        assert (before != after).mean() > (0.2 if mode == "blanked" else 0.6)
    assert_identical(build(mvt, ALL_BLOCKS, q, fit), build(moved, ALL_BLOCKS, q, fit))


@pytest.mark.parametrize("mode", ["plus3h", "per_flight", "blanked"])
def test_t1e_has_teeth_under_the_former_iobt_clock(long_world, monkeypatch, mode):
    """Guard the guard: with the former clock patched back in (the IOBT, else the schedule, and the IOBT among the
    columns the blocks take), the same perturbation moves every block that reads the clock: the calendar, the demand,
    the weather and the history of both leads, and the archive whenever a plan's month moves (a shifted IOBT; a
    blanked one keeps it, since no synthetic plan crosses midnight). Only ``op_hl_plan`` (the stand) reads no clock."""
    mvt, _, _ = long_world
    monkeypatch.setattr(_horizon, "planned_offblock_long", _iobt_clock)
    monkeypatch.setattr(operational_long, "PLAN_COLUMNS", (*operational_long.PLAN_COLUMNS, "IOBT_flt"))
    q = mvt.index[mvt["is_dep"].to_numpy()]
    fit = fit_rows(mvt)
    moved = _moved_columns(build(mvt, ALL_BLOCKS, q, fit), build(perturb_iobt(mvt, mode), ALL_BLOCKS, q, fit))
    clocked = [b for b in ALL_BLOCKS if b != "op_hl_plan" and not (b == "op_adsb_archive" and mode == "blanked")]
    for b in clocked:
        assert moved & set(all_specs()[b].outputs), f"{b} did not move under the IOBT clock: T1e has no teeth there"
    for col in ("op_hl_minute_of_day", "op_hl_dem_dep_sched_apt_w60", "op_hl_dem_arr_sched_pier_w30", "op_hl3_wx_vis_m",
                "op_hl3_taf_issue_age_h", "op_hl3_hist_apt_b3h_n", "op_hl24_wx_vis_m", "op_hl24_hist_apt_b24h_anom_mean"):
        assert col in moved, f"{col} did not move under the IOBT clock: T1e has no teeth"


# --- T2: future deletion -------------------------------------------------------------------------------------------

def _freeze_baseline(monkeypatch, mvt: pd.DataFrame, fit: pd.Series) -> None:
    """The archive table of the history blocks (NM proxy baseline, month-out), computed once on the full frame."""
    dep = _oplib.departures(mvt)
    frozen = _horizon.nm_baseline_long(dep, fit.reindex(dep.index).to_numpy())
    monkeypatch.setattr(_horizon, "nm_baseline_long", lambda d, f, s=20: frozen)


def _queries(mvt: pd.DataFrame, seed: int, n: int = 24) -> pd.Index:
    rng = np.random.default_rng(seed)
    cand = mvt.index[(mvt["is_dep"] & mvt["split"].eq("train")).to_numpy()]
    return pd.Index(rng.choice(cand, n, replace=False), name="mvt_id")


@pytest.mark.parametrize("horizon", ["hl3", "hl24"])
def test_t2_future_deletion_invariance(long_world, monkeypatch, horizon):
    """Every event after the anchor censored. The ADS-B archive is not in the list: it is a table of the fit rows'
    archived tracks, transductive in time by design like the NM baseline (tests/test_adsb_archive.py checks that only
    fit records enter and a query's own month never does); censoring the movement frame cannot exercise it."""
    mvt, offblocks, state = long_world
    fit = fit_rows(mvt)
    _freeze_baseline(monkeypatch, mvt, fit)
    blocks = FREE_BLOCKS + LEAD_BLOCKS[horizon]
    checked = 0
    for r in _queries(mvt, 11 if horizon == "hl3" else 12):
        idx = pd.Index([r], name="mvt_id")
        a = _anchor(mvt, idx, horizon)[0]
        full = build(mvt, blocks, idx, fit)
        cm, cob = censor(mvt, offblocks, a)
        assert cm["MVT_TIME_UTC_mvt"].isna().sum() > 100  # the censoring removes events
        state["offblocks"] = cob
        cut = build(cm, blocks, idx, fit)
        state["offblocks"] = offblocks
        assert_identical(full, cut)
        checked += 1
    assert checked == 24


@pytest.mark.parametrize("horizon", ["hl3", "hl24"])
def test_t2b_future_runway_and_stand_move_only_the_declared_o_star_columns(long_world, monkeypatch, horizon):
    mvt, offblocks, state = long_world
    fit = fit_rows(mvt)
    _freeze_baseline(monkeypatch, mvt, fit)
    blocks = FREE_BLOCKS + LEAD_BLOCKS[horizon]
    allowed = {"op_hl_" + c for c in _horizon.O_STAR_FUTURE_ATTRIBUTES_LONG}
    assert allowed == set(_long_tiers.with_tag([c for b in FREE_BLOCKS for c in all_specs()[b].outputs], "others_future"))
    moved_cols: set[str] = set()
    for r in _queries(mvt, 13):
        idx = pd.Index([r], name="mvt_id")
        a = _anchor(mvt, idx, horizon)[0]
        full = build(mvt, blocks, idx, fit)
        cm, cob = censor(censor_attributes(mvt, a, keep=idx), offblocks, a)
        state["offblocks"] = cob
        cut = build(cm, blocks, idx, fit)
        state["offblocks"] = offblocks
        same = [c for c in full.columns if c not in allowed]
        assert_identical(full[same], cut[same])
        moved_cols |= {c for c in allowed if not full[c].equals(cut[c])}
    assert any("_rwy_" in c for c in moved_cols) and any("_pier_" in c for c in moved_cols), "the O* declaration has no teeth"
    assert not [c for c in allowed if "_apt_" in c]


def test_lobt_is_never_read_on_any_row(long_world):
    """NM's last known off-block time moved and removed for EVERY departure (not only the queries), in a frame where
    some rows have it without an IOBT (the only place an H1-style clock would read it): bit-identical outputs. Catches
    a read on the reference side (other flights' push delay, the plan months of the baseline and the archive)."""
    mvt, _, _ = long_world
    dep_idx = mvt.index[mvt["is_dep"].to_numpy()]
    rng = np.random.default_rng(20)
    no_iobt = pd.Index(rng.choice(dep_idx[mvt.loc[dep_idx, "IOBT_flt"].notna().to_numpy()], 300, replace=False))
    base_frame = _set(mvt, "IOBT_flt", no_iobt, np.full(len(no_iobt), np.nan))  # LOBT present, IOBT missing
    assert (base_frame.loc[no_iobt, "LOBT_flt"].notna()).all()
    shifted = _shift(base_frame, "LOBT_flt", dep_idx, 5400)
    gone = rng.choice(dep_idx, len(dep_idx) // 3, replace=False)
    shifted = _set(shifted, "LOBT_flt", pd.Index(gone), np.full(len(gone), np.nan))
    q = pd.Index(rng.choice(dep_idx, 150, replace=False)).union(no_iobt[:50]).rename("mvt_id")
    fit = fit_rows(mvt)
    assert_identical(build(base_frame, ALL_BLOCKS, q, fit), build(shifted, ALL_BLOCKS, q, fit))
    no_lobt = base_frame.drop(columns=["LOBT_flt"])  # and the blocks run without the column at all
    assert_identical(build(base_frame, ALL_BLOCKS, q, fit), build(no_lobt, ALL_BLOCKS, q, fit))


# --- primitives against brute force ---------------------------------------------------------------------------------

@pytest.fixture
def sym_events():
    rng = np.random.default_rng(15)
    n, m, w = 300, 140, 1800.0
    ref_t = np.sort(rng.integers(0, 40_000, n)).astype("float64")
    ref_g = rng.choice(["A", "B", "C"], n).astype(object)
    q_t = rng.integers(0, 40_000, m).astype("float64")
    q_g = rng.choice(["A", "B", "C", None], m, p=[0.3, 0.3, 0.3, 0.1]).astype(object)
    own = rng.integers(0, n, m)
    own[rng.random(m) < 0.2] = -1
    # the OWN event on both window edges, at the centre, and one second outside each edge; others on the edges too
    for lo, hi, offset in ((40, 52, -w), (52, 64, w), (64, 76, 0.0), (76, 84, -w - 1), (84, 92, w + 1)):
        own[lo:hi] = rng.integers(0, n, hi - lo)
        q_t[lo:hi] = ref_t[own[lo:hi]] - offset
    q_t[100:120] = ref_t[rng.integers(0, n, 20)] + w
    q_t[120:140] = ref_t[rng.integers(0, n, 20)] - w
    q_g = np.where(own >= 0, ref_g[np.clip(own, 0, None)], q_g).astype(object)
    return ref_t, ref_g, q_t, q_g, own, w


def test_symmetric_count_with_self_exclusion_matches_brute_force(sym_events):
    ref_t, ref_g, q_t, q_g, own, w = sym_events
    own_t = np.where(own >= 0, ref_t[np.clip(own, 0, None)], np.nan)
    d = q_t[own >= 0] - own_t[own >= 0]
    for edge in (-w, w, 0.0, -w - 1, w + 1):
        assert (d == -edge).sum() >= 5, edge
    got = _horizon.symmetric_count(ref_t, ref_g, q_t, q_g, w, own_t)
    for i in range(len(q_t)):
        want = 0
        if q_g[i] is not None:
            want = sum(1 for j in range(len(ref_t)) if j != own[i] and ref_g[j] == q_g[i] and q_t[i] - w <= ref_t[j] <= q_t[i] + w)
        assert got[i] == want, i


def _brute_demand(mvt: pd.DataFrame, q_ids) -> pd.DataFrame:
    dep, arr = _oplib.departures(mvt), _oplib.arrivals(mvt)

    def groups(frame):
        return {"apt": frame["apt"].astype(str).to_numpy(), "rwy": _oplib.runway_key(frame).astype(str).to_numpy(),
                "pier": _oplib.pier_key(frame).astype(str).to_numpy()}

    refs = {"dep_sched": (dep, _oplib.secs(dep["SCHED_TIME_UTC_mvt"])), "arr_sched": (arr, _oplib.secs(arr["SCHED_TIME_UTC_mvt"]))}
    out = []
    for qid in q_ids:
        q = mvt.loc[[qid]]
        P = _schedule_seconds(q)[0]  # the clock is the schedule (asserted on its own above)
        qg = {k: v[0] for k, v in groups(q).items()}
        row = {}
        for ref, (frame, t) in refs.items():
            g = groups(frame)
            not_self = frame.index != qid
            for lvl in _horizon.DEMAND_LEVELS:
                for wname, half in _horizon.DEMAND_WINDOWS.items():
                    with np.errstate(invalid="ignore"):
                        hit = not_self & (g[lvl] == qg[lvl]) & (t >= P - half) & (t <= P + half)
                    row[f"op_hl_dem_{ref}_{lvl}_{wname}"] = float(hit.sum())
        out.append(row)
    return pd.DataFrame(out, index=pd.Index(q_ids, name="mvt_id"))


def test_demand_block_matches_brute_force(long_world):
    mvt, _, _ = long_world
    dep = _oplib.departures(mvt)
    rng = np.random.default_rng(16)
    q = pd.Index(rng.choice(dep.index, 60, replace=False), name="mvt_id")
    got = build(mvt, ("op_hl_demand",), q)
    want = _brute_demand(mvt, q)
    pd.testing.assert_frame_equal(got.astype("float64"), want[got.columns], check_exact=True)
    assert (got.to_numpy() > 0).mean() > 0.5


def _brute_history(mvt: pd.DataFrame, q_ids, horizon: str, baseline) -> pd.DataFrame:
    dep = _oplib.departures(mvt)
    t = _oplib.secs(dep["MVT_TIME_UTC_mvt"])
    t_all, apt_all = _oplib.secs(mvt["MVT_TIME_UTC_mvt"]), mvt["apt"].astype(str).to_numpy()
    push_raw = np.clip(_oplib.secs(dep["AOBT_3_flt"]) - _schedule_seconds(dep), *_oplib.PUSH_DELAY_CLIP)  # vs the schedule
    groups = {"apt": dep["apt"].astype(str).to_numpy(), "rwy": _oplib.runway_key(dep).astype(str).to_numpy()}
    out = []
    for qid in q_ids:
        q = mvt.loc[[qid]]
        a = _schedule_seconds(q)[0] - LEADS[horizon]  # the anchor: the schedule minus the lead
        sched = q["SCHED_TIME_UTC_mvt"].iloc[0]
        key = baseline.key(float(sched.year * 12 + sched.month - 1))  # month-out by the schedule's month
        an = baseline.anomaly(key)
        qg = {"apt": str(q["apt"].iloc[0]), "rwy": str(_oplib.runway_key(q).iloc[0])}
        row = {}
        for lvl in _horizon.HIST_LEVELS:
            for win, back in _horizon.HIST_WINDOWS.items():
                p = f"op_{horizon}_hist_{lvl}_{win}_"
                start = a - back
                with np.errstate(invalid="ignore"):  # coverage: another movement at the airport in the day before the start
                    covered = ((mvt.index != qid) & (apt_all == qg["apt"]) & (t_all >= start - _horizon.COVERAGE_LOOKBACK_S)
                               & (t_all <= start)).any()
                if not covered:
                    row.update({p + s: np.nan for s in _horizon.HIST_STATS})
                    continue
                with np.errstate(invalid="ignore"):
                    sel = (dep.index != qid) & (groups[lvl] == qg[lvl]) & (t >= a - back) & (t <= a) & np.isfinite(an)
                vals, push = an[sel], push_raw[sel]
                row[p + "n"] = float(sel.sum())
                row[p + "anom_mean"] = float(vals.mean()) if len(vals) else np.nan
                row[p + "anom_share5"] = float((vals > _oplib.HIGH_ANOMALY_S).mean()) if len(vals) else np.nan
                pv = push[np.isfinite(push)]
                row[p + "push_delay_mean"] = float(pv.mean()) if len(pv) else np.nan
        out.append(row)
    return pd.DataFrame(out, index=pd.Index(q_ids, name="mvt_id"))


@pytest.mark.parametrize("horizon", ["hl3", "hl24"])
def test_history_block_matches_brute_force(long_world, horizon):
    mvt, _, _ = long_world
    fit = fit_rows(mvt)
    dep = _oplib.departures(mvt)
    rng = np.random.default_rng(17)
    q = pd.Index(rng.choice(dep.index[dep["split"].eq("train").to_numpy()], 40, replace=False).tolist()
                 + rng.choice(dep.index[dep["split"].eq("rank").to_numpy()], 10, replace=False).tolist(), name="mvt_id")
    got = build(mvt, (f"op_{horizon}_hist",), q, fit).astype("float64")
    baseline = _horizon.nm_baseline_long(dep, fit.reindex(dep.index).to_numpy())
    want = _brute_history(mvt, q, horizon, baseline)[got.columns]
    np.testing.assert_allclose(got.to_numpy(), want.to_numpy(), rtol=2e-6, atol=1e-3, equal_nan=True)
    assert np.nanmax(got.filter(like="_n").to_numpy()) > 0


def _gapped_world(mvt: pd.DataFrame, first: int = 6, last: int = 8) -> pd.DataFrame:
    """The frame without any movement of days ``first`` to ``last`` (a gap between two periods, as June 2026 is)."""
    day = np.floor((_oplib.secs(mvt["MVT_TIME_UTC_mvt"]) - pd.Timestamp("2025-03-20", tz=UTC).timestamp()) / 86_400.0)
    return mvt.loc[~((day >= first) & (day <= last))]


@pytest.mark.parametrize("horizon", ["hl3", "hl24"])
def test_history_windows_reaching_into_a_gap_are_unknown_not_empty(long_world, horizon):
    """A window that starts where the frame holds no records reads NaN in every column, never n = 0; elsewhere the
    values are those of the brute force; the anchors one gap-length later read ordinary values again."""
    mvt, _, _ = long_world
    gapped = _gapped_world(mvt)
    fit = fit_rows(gapped)
    dep = _oplib.departures(gapped)
    P = _horizon.planned_offblock_long(dep)
    t0 = pd.Timestamp("2025-03-20", tz=UTC).timestamp()
    after = dep.index[(P >= t0 + 9 * 86_400) & (P < t0 + 12 * 86_400) & dep["split"].eq("train").to_numpy()]
    rng = np.random.default_rng(19)
    q = pd.Index(rng.choice(after, 60, replace=False), name="mvt_id")
    got = build(gapped, (f"op_{horizon}_hist",), q, fit).astype("float64")
    baseline = _horizon.nm_baseline_long(dep, fit.reindex(dep.index).to_numpy())
    want = _brute_history(gapped, q, horizon, baseline)[got.columns]
    np.testing.assert_allclose(got.to_numpy(), want.to_numpy(), rtol=2e-6, atol=1e-3, equal_nan=True)
    s = _horizon.anchor_long(gapped.loc[q], LEADS[horizon]) - 86_400  # the start of the 24 h window
    gap_end = t0 + 9 * 86_400
    n24 = got[f"op_{horizon}_hist_apt_b24h_n"].to_numpy()
    in_gap = (s >= t0 + 7 * 86_400) & (s < gap_end)  # the whole day before the start lies in the gap
    assert in_gap.any() and np.isnan(n24[in_gap]).all(), "a window starting in the gap must be unknown"
    assert not (n24 == 0).any(), "an empty window caused by the gap must never read as zero traffic"
    late = s >= gap_end + 6 * 3600  # windows starting well after the gap: ordinary values
    assert late.any() and np.isfinite(n24[late]).all() and (n24[late] > 0).all()


def test_window_covered_ignores_the_own_takeoff(long_world):
    """Coverage reads the day before the window start only, and never the query's own takeoff (T1)."""
    mvt, _, _ = long_world
    gapped = _gapped_world(mvt)
    dep = _oplib.departures(gapped)
    t0 = pd.Timestamp("2025-03-20", tz=UTC).timestamp()
    q = dep.index[(_oplib.secs(dep["MVT_TIME_UTC_mvt"]) >= t0 + 9 * 86_400)][:40]
    start = np.full(len(q), t0 + 9 * 86_400 - 60.0)  # one minute before the end of the gap: nothing in the day before
    assert not _horizon.window_covered(gapped, q, start).any()
    apt = dep["apt"].astype(str)
    for qid in q[:5]:  # one flight at a time: its own takeoff moved into its look-back
        idx = pd.Index([qid], name="mvt_id")
        own = _set(gapped, "MVT_TIME_UTC_mvt", idx, start[:1] - 3600.0)
        assert not _horizon.window_covered(own, idx, start[:1]).any()
        peer = dep.index[(apt == apt.loc[qid]).to_numpy() & (dep.index != qid)][-1]  # another departure of the airport
        other = _set(gapped, "MVT_TIME_UTC_mvt", pd.Index([peer], name="mvt_id"), start[:1] - 3600.0)
        assert _horizon.window_covered(other, idx, start[:1]).all()  # another flight there does cover it
    assert _horizon.window_covered(gapped, q, start + 12 * 3600.0).all()  # the movements after the gap cover later starts


def test_nm_baseline_long_is_month_out_by_the_plan_month():
    mvt, _ = synthetic_world(days=16)
    dep = _oplib.departures(mvt)
    fit = dep["split"].eq("train").to_numpy()
    bl = _horizon.nm_baseline_long(dep, fit, min_support=5)
    assert isinstance(bl, _oplib.NmBaseline)
    np.testing.assert_array_equal(bl.month, _horizon.plan_month(dep))
    assert bl.fit_months == frozenset(int(m) for m in np.unique(_horizon.plan_month(dep)[fit & np.isfinite(bl.proxy)]))
    assert len(bl.fit_months) == 2
    for m in bl.fit_months:  # leaving out a month changes the medians the other months read
        assert not np.array_equal(bl.anomaly(m), bl.anomaly(None), equal_nan=True)


def test_nm_baseline_long_never_touches_iobt_or_lobt_and_matches_the_parent():
    """Built without the IOBT and LOBT columns at all, keyed by the schedule's month; where the two month clocks agree
    (in the synthetic world no plan crosses midnight, so the IOBT's month is the schedule's) its anomalies equal
    `_oplib.NmBaseline`'s (IOBT, else LOBT, else schedule), for every month left out."""
    mvt, _ = synthetic_world(days=16)
    dep = _oplib.departures(mvt)
    fit = dep["split"].eq("train").to_numpy()
    long_ = _horizon.nm_baseline_long(dep.drop(columns=["IOBT_flt", "LOBT_flt"]), fit, min_support=5)
    parent = _oplib.NmBaseline(dep, fit, min_support=5)
    sched = dep["SCHED_TIME_UTC_mvt"]
    np.testing.assert_array_equal(long_.month, (sched.dt.year * 12 + sched.dt.month - 1).to_numpy("float64"))
    np.testing.assert_array_equal(long_.month, parent.month)
    assert long_.fit_months == parent.fit_months
    for m in (None, *sorted(parent.fit_months)):
        np.testing.assert_array_equal(long_.anomaly(m), parent.anomaly(m))
    assert {k: v for k, v in vars(long_).items() if k != "_tables"}.keys() == {k for k in vars(parent) if k != "_tables"}


def _taf_tables(tmp_path: Path, extra: list[dict] | None = None):
    _, tafs = synthetic_weather(tmp_path, None, extra)
    return tafs


def test_taf_selection_matches_brute_force(tmp_path, monkeypatch):
    """The TAF read for (a, tau): the latest issued at or before ``a`` (stepping back at most three) that covers ``tau``."""
    tafs = _taf_tables(tmp_path)
    monkeypatch.setattr(weather_fcst, "load_tafs", lambda: tafs)
    tab = tafs["EDDF"]
    rng = np.random.default_rng(18)
    t0 = pd.Timestamp("2025-03-20", tz=UTC).timestamp()
    a = t0 + rng.integers(0, 14 * 24, 300) * 1800.0 + rng.integers(-3, 4, 300)
    tau = a + rng.choice([3 * 3600.0, 24 * 3600.0, 23 * 3600.0, 0.0], 300)
    q = pd.DataFrame({"apt": pd.Categorical(["EDDF"] * len(a))}, index=pd.RangeIndex(len(a)))
    got = _horizon.taf_for(q, a, tau)
    for i in range(len(a)):
        latest = np.flatnonzero(tab["issue"] <= a[i])[-4:][::-1]
        pick = next((k for k in latest if tab["vfrom"][k] <= tau[i] < tab["vto"][k]), None)
        if pick is not None and tab["cnl"][pick]:
            pick = None
        if pick is None:
            assert np.isnan(got["taf_vis_m"].iloc[i]) and np.isnan(got["taf_issue_age_h"].iloc[i]), i
            continue
        seg_starts = tab["seg_key"] // weather_fcst.BIG == pick
        offsets = tab["seg_key"][seg_starts] - pick * weather_fcst.BIG
        j = np.flatnonzero(seg_starts)[np.searchsorted(offsets, tau[i] - tab["vfrom"][pick], side="right") - 1]
        assert got["taf_vis_m"].iloc[i] == np.float32(tab["p_vis_m"][j]), i
        assert got["taf_worst_snowfrz"].iloc[i] == np.float32(tab["w_snowfrz"][j]), i
        assert got["taf_issue_age_h"].iloc[i] == np.float32((a[i] - tab["issue"][pick]) / 3600.0), i
    assert got["taf_vis_m"].notna().mean() > 0.3


# --- weather causality at the long anchors -----------------------------------------------------------------------

def _taf(apt: str, issue: pd.Timestamp, valid_from: pd.Timestamp, valid_to: pd.Timestamp, tag: str) -> list[dict]:
    return [
        {"station": apt, "taf_id": tag, "parse_ok": 1, "issue_time": issue, "is_amd": 0, "is_cor": 0, "is_cnl": 0,
         "valid_from": valid_from, "valid_to": valid_to},
        {"taf_id": tag, "seg_start": valid_from, **{c: 77.0 for c in weather_fcst.SEGMENT_COLS}},
    ]


@pytest.mark.parametrize("horizon", ["hl3", "hl24"])
def test_weather_after_the_anchor_changes_nothing(tmp_path, monkeypatch, horizon):
    """One query at a time: reports after its anchor change nothing; a TAF issued by then counts only if it covers P."""
    mvt, _ = synthetic_world(days=8)
    rows = spaced_departures(mvt, hours=30)
    rows = rows[_horizon.anchor_long(mvt.loc[rows], LEADS[horizon]) > pd.Timestamp("2025-03-20", tz=UTC).timestamp()][:6]
    assert len(rows) >= 4

    def wx(q, extra_metar=None, extra_taf=None):
        metar, tafs = synthetic_weather(tmp_path, extra_metar, extra_taf)
        monkeypatch.setattr(weather, "load_metar", lambda: metar)
        monkeypatch.setattr(weather_fcst, "load_tafs", lambda: tafs)
        planned = _horizon.planned_offblock_long(q)
        return _horizon.wx_long(q, planned - LEADS[horizon], planned)

    lag = _oplib.METAR_LAG_S
    at = lambda sec: pd.Timestamp(sec, unit="s", tz=UTC)  # noqa: E731
    for k, r in enumerate(rows):
        q = mvt.loc[[r]]
        P = _horizon.planned_offblock_long(q)[0]
        a = P - LEADS[horizon]
        apt = str(q["apt"].iloc[0])
        base = wx(q)
        metar_cols = [c for c in base.columns if not c.startswith("taf_")]
        taf_cols = [c for c in base.columns if c.startswith("taf_")]
        late_m, _ = _late_reports(apt, at(a - lag + 1), f"late{k}")
        assert_identical(base, wx(q, [late_m]))  # observed after a - lag: not yet known
        edge_m, _ = _late_reports(apt, at(a - lag), f"edge{k}")
        moved = wx(q, [edge_m])
        assert not base[metar_cols].equals(moved[metar_cols]) and base[taf_cols].equals(moved[taf_cols])
        # a TAF covering P: issued 1 s after the anchor -> nothing; issued at the anchor -> the TAF columns move
        cover = (at(a), at(a + 600), at(P + 7200))
        assert_identical(base, wx(q, None, _taf(apt, at(a + 1), *cover[1:], f"after{k}")))
        moved = wx(q, None, _taf(apt, cover[0], *cover[1:], f"at{k}"))
        assert not base[taf_cols].equals(moved[taf_cols]) and base[metar_cols].equals(moved[metar_cols])
        assert moved["taf_vis_m"].iloc[0] == 77.0 and moved["taf_issue_age_h"].iloc[0] == 0.0
        # issued by the anchor but ending before P: not the TAF of the planned hour
        short = wx(q, None, _taf(apt, cover[0], at(a + 600), at(P - 1), f"short{k}"))
        assert short["taf_vis_m"].iloc[0] != 77.0


# --- configs ---------------------------------------------------------------------------------------------------

LADDER = {h: ["op_hl_r0.yaml", f"op_{h}_r2.yaml", f"op_{h}_r3.yaml", f"op_{h}_r4.yaml", f"op_{h}_r4_arch.yaml"] for h in LEADS}


def _blocks(name: str) -> set[str]:
    cfg = load_config(name)
    return {s.name for s in resolve(cfg.features.include, cfg.features.exclude)}


def test_long_ladders_are_cumulative_and_mirror_each_other():
    for h, order in LADDER.items():
        sets = [_blocks(n) for n in order]
        for small, big in zip(sets, sets[1:]):
            assert small < big, (h, small, big)
        assert sets[-1] - sets[-2] == {"op_adsb_archive"}
        assert sets[1] - sets[0] == {"taxi_distance", "taxi_route", "stand_geometry", f"op_{h}_wx"}
        assert sets[2] - sets[1] == {"op_hl_demand"}
        assert sets[3] - sets[2] == {f"op_{h}_hist"}
        other = "hl24" if h == "hl3" else "hl3"
        assert not any(b.startswith(f"op_{other}_") or b.startswith(("op_h1_", "op_h0_")) for s in sets for b in s)
    for a3, a24 in zip(LADDER["hl3"], LADDER["hl24"]):
        assert {b.replace("op_hl3_", "op_hl24_") for b in _blocks(a3)} == _blocks(a24)


def test_archive_rungs_add_exactly_the_archive():
    assert _blocks("op_h1_r4_arch.yaml") - _blocks("op_h1_r4.yaml") == {"op_adsb_archive"}
    assert _blocks("op_h1_r4.yaml") < _blocks("op_h1_r4_arch.yaml")
    assert _blocks("op_h0_r5_arch.yaml") - _blocks("op_h0_r5.yaml") == {"op_adsb_archive"}
    assert _blocks("op_h0_r5.yaml") < _blocks("op_h0_r5_arch.yaml")
    for name in ("op_h1_r4_arch.yaml", "op_h0_r5_arch.yaml"):
        cfg = load_config(name)
        assert not cfg.model.refit_on_all and not cfg.evaluation.permutation.enabled, name


DELIVERABLES = {"hl3": ("op_hl3_r3_rwyblind.yaml", "op_hl3_r4_rwyblind.yaml"),
                "hl24": ("op_hl24_r3_rwyblind.yaml", "op_hl24_r4_rwyblind.yaml")}
"""The refit candidates of each lead, (r3, r4): runway-blind, on the schedule clock, without the archive. Rule R of
findings 10.35 picks r3 (no NM history) if its OP MAE is within +0.5 s of r4's on Jan+Jul and on the forward check,
else r4. No archive since the forward check of 10 Oct (findings 10.31: fitted on Jan-Oct and validated on Nov+Dec,
the 24 h archive rung lost 3.1 s of MAE to its archive-free twin)."""
ARCHIVE_RUNGS = {"hl3": "op_hl3_r4_arch_rwyblind.yaml", "hl24": "op_hl24_r4_arch_rwyblind.yaml"}
"""The runway-blind archive rungs (the deliverables until the forward check), holdout only."""
FORWARD = {
    "op_hl3_r3_rwyblind_fwd.yaml": "op_hl3_r3_rwyblind.yaml",
    "op_hl3_r4_rwyblind_fwd.yaml": "op_hl3_r4_rwyblind.yaml",
    "op_hl24_r3_rwyblind_fwd.yaml": "op_hl24_r3_rwyblind.yaml",
    "op_hl24_r4_rwyblind_fwd.yaml": "op_hl24_r4_rwyblind.yaml",
    "op_hl24_r4_arch_rwyblind_fwd.yaml": "op_hl24_r4_arch_rwyblind.yaml",
}
"""Each forward twin (fit Jan-Oct 2025, validate Nov+Dec) and the rung it repeats: rule R's pairs and the archive's."""
RWYBLIND = (*(c for pair in DELIVERABLES.values() for c in pair), *ARCHIVE_RUNGS.values(), *FORWARD)
PIER_ONLY = ("op_hl24_r4_arch_rwyblind_pier.yaml",)
OWN_RUNWAY_RX = r"^runway$|apt_runway|stand_runway|geo_rwy|geo_route|geo_gc|headwind|crosswind|_rwy_w\d|hist_rwy_|^op_arch_(sr|rwy)_"
"""An independent pattern for the own-runway columns (the H1 runway-blind test's, extended to the long blocks)."""
OWN_STAND_RX = r"^stand$|^apt_stand|^stand_(number|suffix)$|^geo_stand_(east|north|arp|matched|match)|has_stand|^op_arch_stand_"


def _outputs(name: str) -> tuple:
    cfg = load_config(name)
    specs = resolve(cfg.features.include, cfg.features.exclude)
    return cfg, {s.name for s in specs}, [c for s in specs for c in s.outputs]


def _kept(name: str) -> list[str]:
    """The columns a config puts into the model matrix, in order (resolved outputs minus its drop list)."""
    cfg, _, cols = _outputs(name)
    return [c for c in cols if c not in cfg.features.drop_columns]


def test_refit_only_on_the_runway_blind_deliverables():
    long_cfgs = sorted(p.name for p in paths.CONFIGS.glob("op_hl*.yaml"))
    assert set(LADDER["hl3"] + LADDER["hl24"] + list(RWYBLIND + PIER_ONLY)) == set(long_cfgs)
    refit = [n for n in long_cfgs if load_config(n).model.refit_on_all]
    assert refit == sorted(c for pair in DELIVERABLES.values() for c in pair), refit
    for name in long_cfgs:
        cfg = load_config(name)
        assert not cfg.evaluation.permutation.enabled, name
        if name in FORWARD:
            assert cfg.validation.scheme == "forward_holdout" and cfg.validation.holdout_months == ["2025-11", "2025-12"]
        else:
            assert cfg.validation.scheme == "seasonal_holdout" and cfg.validation.holdout_months == ["2025-01", "2025-07"]
    for name in refit:  # every refit long rung is runway-blind and carries no archive
        cfg, blocks, cols = _outputs(name)
        assert "op_adsb_archive" not in blocks
        assert not [c for c in cols if c not in cfg.features.drop_columns and "own_runway" in _long_tiers.tags_of(c)]


@pytest.mark.parametrize("name", RWYBLIND + PIER_ONLY)
def test_blind_rungs_drop_exactly_the_tagged_columns(name):
    """A runway-blind long rung keeps no ``own_runway`` column and drops nothing else but ``takeoff_epoch`` (the
    pier-only rung: and the ``own_stand`` columns), so a difference to its runway-aware rung is that set's."""
    cfg, blocks, cols = _outputs(name)
    tags = ("own_runway", "own_stand") if name in PIER_ONLY else ("own_runway",)
    assert set(cfg.features.drop_columns) == {"takeoff_epoch", *_long_tiers.with_tag(cols, *tags)}
    kept = _kept(name)
    assert not _long_tiers.with_tag(kept, *tags), name
    leaks = [c for c in kept if re.search(OWN_RUNWAY_RX, c)]
    assert not leaks, leaks  # the independent pattern agrees with the tags
    assert "op_hl_dem_dep_sched_apt_w60" in kept and "op_hl_hour_local" in kept
    if any(b.endswith("_hist") for b in blocks):
        assert any(c.endswith("_hist_apt_b24h_anom_mean") for c in kept)
    assert "stand_prefix" in kept and "geo_stand_zone" in kept and "op_hl_dem_dep_sched_pier_w30" in kept  # the pier stays
    if "op_adsb_archive" in blocks:
        assert "op_arch_pier_power_share" in kept and "op_arch_rwy_tau_300_s" not in kept
    if name in PIER_ONLY:
        leaks = [c for c in kept if re.search(OWN_STAND_RX, c)]
        assert not leaks, leaks
    else:
        assert "apt_stand" in kept and "op_hl_has_stand" in kept


def test_runway_blind_track_mirrors_the_ladder():
    for h in LEADS:
        r3, r4 = DELIVERABLES[h]
        arch = ARCHIVE_RUNGS[h]
        assert _blocks(r3) == _blocks(f"op_{h}_r3.yaml") and _blocks(r4) == _blocks(f"op_{h}_r4.yaml")
        assert _blocks(arch) == _blocks(f"op_{h}_r4_arch.yaml")
        drop = {n: set(load_config(n).features.drop_columns) for n in (r3, r4, arch)}
        # r4 adds the NM history: its runway half is the only extra drop; the archive adds its runway-keyed tables
        hist_rwy = {c for c in all_specs()[f"op_{h}_hist"].outputs if "_hist_rwy_" in c}
        arch_rwy = {c for c in all_specs()["op_adsb_archive"].outputs if c.startswith(("op_arch_sr_", "op_arch_rwy_"))}
        assert drop[r3] < drop[r4] and drop[r4] - drop[r3] == hist_rwy and drop[arch] - drop[r4] == arch_rwy
    for hl3, hl24 in (*zip(DELIVERABLES["hl3"], DELIVERABLES["hl24"]), (ARCHIVE_RUNGS["hl3"], ARCHIVE_RUNGS["hl24"])):
        assert {d.replace("op_hl3_", "op_hl24_") for d in load_config(hl3).features.drop_columns} == set(
            load_config(hl24).features.drop_columns)
    assert _blocks(PIER_ONLY[0]) == _blocks(ARCHIVE_RUNGS["hl24"])


@pytest.mark.parametrize("fwd", list(FORWARD))
def test_forward_twins_keep_exactly_their_rungs_columns(fwd):
    """A forward twin changes the validation only (fit Jan-Oct 2025, validate Nov+Dec, no refit): the same blocks,
    drop list, data rules and model, so exactly its rung's kept columns, in order."""
    rung = FORWARD[fwd]
    a, b = load_config(fwd), load_config(rung)
    assert _blocks(fwd) == _blocks(rung) and _kept(fwd) == _kept(rung)
    assert a.features == b.features and a.data == b.data and a.target == b.target and a.model.params == b.model.params
    assert a.validation.scheme == "forward_holdout" and list(a.validation.holdout_months) == ["2025-11", "2025-12"]
    assert not a.model.refit_on_all and a.name == rung.removesuffix(".yaml") + "_fwd"


def test_no_long_input_is_an_iobt_or_day_of_year_column():
    """findings 10.35: no long block and not the archive outputs, and no long config resolves, a column of the IOBT
    (the clock is the schedule) or the day of year (it maps 2025's moving holidays onto 2026 dates)."""
    rx = re.compile(r"iobt|day_of_year", re.I)
    outs = [c for s in all_specs().values() if s.name.startswith(("op_hl", "op_adsb")) for c in s.outputs]
    assert len(outs) > 100 and not [c for c in outs if rx.search(c)]
    assert set(_horizon.STATIC) == set(_oplib.STATIC) - {"day_of_year"} and _horizon.PLAN == ("has_stand",)
    assert _horizon.DEMAND_REFS == ("dep_sched", "arr_sched")
    long_cfgs = sorted(p.name for p in paths.CONFIGS.glob("op_hl*.yaml"))
    assert len(long_cfgs) == len(set(LADDER["hl3"] + LADDER["hl24"])) + len(RWYBLIND + PIER_ONLY)
    for name in long_cfgs:
        _, _, cols = _outputs(name)
        assert not [c for c in cols if rx.search(c)], name


def test_no_explicit_join_flag_is_an_output():
    """Whether the NM record joined depends on the own actual off-block (tier C): no long block flags it, and on the
    schedule clock no long output carries it (its carriers are `aircraft_identity`'s columns)."""
    outs = [c for s in all_specs().values() if s.name.startswith(("op_hl", "op_adsb")) for c in s.outputs]
    assert not [c for c in outs if re.search(r"has_iobt|has_actype|has_flt|joined|iobt", c)]
    assert not [c for c in outs if "join_c" in _long_tiers.tags_of(c)]
    for col in ("aircraft_type_flt", "operator_hash", "wake_category", "carrier_icao", "aircraft_type"):
        assert "join_c" in _long_tiers.tags_of(col), col
    assert not _long_tiers.tags_of("carrier_flightno") and not _long_tiers.tags_of("flight_rule_mvt")  # movement record
    assert "others_iobt" not in _long_tiers.TAGS
    assert _long_tiers.tier_of("op_hl_dem_dep_sched_rwy_w30") == "O* (own runway, others' allocation)"
    assert _long_tiers.tier_of("op_hl_dem_dep_sched_apt_w60") == "O" and _long_tiers.tier_of("apt") == "O"


# --- the tier tags against the blocks: swap the own runway / stand ---------------------------------------------

def _swap_rows(mvt: pd.DataFrame) -> pd.Index:
    """T1 rows (one planned month, spaced beyond every reach) at an A or B stand (piers with several stands)."""
    rows = t1_rows(mvt)
    stand = mvt.loc[rows, "STAND_mvt"].astype("string")
    return rows[stand.str.match(r"^[AB]\d+$").fillna(False).to_numpy(dtype=bool)]


def _moved_columns(base: pd.DataFrame, moved: pd.DataFrame) -> set[str]:
    return {c for c in base.columns if not base[c].equals(moved[c])}


def test_own_runway_moves_only_the_own_runway_columns(long_world):
    mvt, _, _ = long_world
    rows = _swap_rows(mvt)
    assert len(rows) >= 6
    fit = fit_rows(mvt)
    base = build(mvt, ALL_BLOCKS, rows, fit)
    swapped = mvt.copy()
    for r in rows:
        own = str(mvt.at[r, "RUNWAY_mvt"])
        swapped.loc[r, "RUNWAY_mvt"] = next(x for x in RUNWAYS[str(mvt.at[r, "apt"])] if x != own)
    moved = _moved_columns(base, build(swapped, ALL_BLOCKS, rows, fit))
    tagged = set(_long_tiers.with_tag(base.columns, "own_runway"))
    assert moved <= tagged, sorted(moved - tagged)
    for col in ("op_hl_dem_dep_sched_rwy_w60", "op_hl3_wx_headwind_kt", "op_hl24_hist_rwy_b24h_n", "op_arch_sr_start_8kt_s",
                "op_arch_rwy_tau_300_s"):
        assert col in moved, f"{col} is tagged own_runway but did not move: the tag has no teeth"


@pytest.mark.parametrize("across_piers", [False, True])
def test_own_stand_moves_only_the_own_stand_columns(long_world, across_piers):
    mvt, _, _ = long_world
    rows = _swap_rows(mvt)
    assert len(rows) >= 6
    fit = fit_rows(mvt)
    base = build(mvt, ALL_BLOCKS, rows, fit)
    swapped = mvt.copy()
    for r in rows:  # A1..A12 and B1..B8: number % 8 + 1 is another stand of either pier
        stand = str(mvt.at[r, "STAND_mvt"])
        pier, number = stand[0], int(stand[1:])
        other = ("B" if pier == "A" else "A") if across_piers else pier
        swapped.loc[r, "STAND_mvt"] = f"{other}{number % 8 + 1}"
    assert (swapped.loc[rows, "STAND_mvt"].astype(str) != mvt.loc[rows, "STAND_mvt"].astype(str)).all()
    moved = _moved_columns(base, build(swapped, ALL_BLOCKS, rows, fit))
    tags = ("own_stand", "own_pier") if across_piers else ("own_stand",)
    tagged = set(_long_tiers.with_tag(base.columns, *tags))
    assert moved <= tagged, sorted(moved - tagged)
    assert {"op_arch_stand_tau_300_s", "op_arch_sr_push_dur_s"} <= moved, "the own_stand tag has no teeth"
    if across_piers:
        assert {"op_hl_dem_dep_sched_pier_w60", "op_arch_pier_push_dur_s"} <= moved


# --- the real data (opt-in) ------------------------------------------------------------------------------------------

REAL = os.environ.get("HORIZON_REAL") == "1"


def _day_keys(frame: pd.DataFrame) -> np.ndarray:
    """Planned UTC day instead of month: the slice holds a single month, so month-out would leave no fit rows."""
    return np.floor(_horizon.planned_offblock_long(frame) / 86_400.0)


@pytest.mark.skipif(not (REAL and paths.MOVEMENTS_FILE.exists()), reason="set HORIZON_REAL=1 (reads a 50k-row slice)")
def test_real_slice_t1_and_t2(monkeypatch):
    """T1 (crossing both anchors, own NM actuals, own LOBT) and T2 on the first 50k real movement rows.

    The month-out keys are planned days here (`_day_keys`), so the leave-out logic is exercised on one month of data;
    the T1 rows share one planned day for the same reason. The archive block joins when its input file exists.
    """
    mvt, ob = _real_slice()
    monkeypatch.setattr(_oplib, "load_offblocks", lambda: ob)
    monkeypatch.setattr(_horizon, "plan_month", _day_keys)
    blocks = LONG_BLOCKS + (("op_adsb_archive",) if paths.ADSB_ARCHIVE_FILE.exists() else ())
    rows = spaced_departures(mvt, hours=52)
    day = _day_keys(mvt.loc[rows])
    rows = rows[day == pd.Series(day).mode().iloc[0]]
    if len(rows) < 4:
        rows = spaced_departures(mvt, hours=26)[:6]
        rows = rows[_day_keys(mvt.loc[rows]) == _day_keys(mvt.loc[rows[:1]])[0]]
    assert len(rows) >= 2
    fit = fit_rows(mvt)
    base = build(mvt, blocks, rows, fit)
    assert base.filter(like="hist_apt_b3h_n").notna().all().all()
    for h in LEADS:
        moved = _set(mvt, "MVT_TIME_UTC_mvt", rows, _anchor(mvt, rows, h) - 60)
        assert_identical(base, build(moved, blocks, rows, fit))
    moved = _shift(_shift(mvt, "AOBT_3_flt", rows, 600), "ARVT_3_flt", rows, 600)
    assert_identical(base, build(moved, blocks, rows, fit))
    assert_identical(base, build(_shift(mvt, "LOBT_flt", rows, 3600), blocks, rows, fit))

    dep = _oplib.departures(mvt)
    frozen = _horizon.nm_baseline_long(dep, fit.reindex(dep.index).to_numpy())
    monkeypatch.setattr(_horizon, "nm_baseline_long", lambda d, f, s=20: frozen)
    rng = np.random.default_rng(14)
    train = dep.index[dep["split"].astype("string").eq("train").to_numpy(dtype=bool)]
    n_rows = int(os.environ.get("HORIZON_REAL_T2_ROWS", "12"))
    for r in rng.choice(train, n_rows, replace=False):
        idx = pd.Index([r], name=mvt.index.name)
        for h in LEADS:
            names = FREE_BLOCKS + LEAD_BLOCKS[h]
            a = _anchor(mvt, idx, h)[0]
            cm, _ = censor(mvt, ob, a)
            assert_identical(build(mvt, names, idx, fit), build(cm, names, idx, fit))
