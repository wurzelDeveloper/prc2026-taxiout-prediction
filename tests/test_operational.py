"""The operational ladder (`prc.features.operational`, `prc.features._oplib`): leakage and contract tests.

An operational prediction is made before the aircraft moves, so these blocks must never see the flight's own
takeoff, its own NM actuals or (at H1) its own off-block, nor any other flight's event after the anchor. The
properties are asserted, not assumed:

* T1 / T1b / T1c -- moving the flight's own takeoff (by +600 s, -300 s, and to 60 s before its own anchor),
  its own AOBT_3 / ARVT_3, or its own recorded off-block leaves every op_h1_* value bit-identical (and the
  op_h0_* values for the takeoff and the NM actuals), with the perturbed rows kept in the fit, as in training
  (the NM baseline is month-out, so a fit row's own reading never enters what it reads);
* T2 -- censoring every event after a query's anchor leaves its values bit-identical;
* T2b -- censoring, in addition, the runway and stand of every other flight that moves after the anchor
  changes nothing but the declared O* columns (`_oplib.O_STAR_FUTURE_ATTRIBUTES`);
* the primitives against brute force, events exactly at the window edges included;
* weather causality: a METAR observed or a TAF issued after the anchor changes nothing;
* config allowlist and the lab quarantine (source scan).

Everything runs on synthetic frames. ``OPLAD_REAL=1`` adds T1, T1b and T2 on a 50k-row slice of the real
movements (the slice's off-blocks streamed from the target file in 50k-row batches). ``OPLAD_FULL=1`` runs the anchor
sanity check on the full 2025 departures (reads columns of the whole frame: not while a GPU fit runs); it is a
REQUIRED gate before the op_h1_r4 board upload, as is ``OPLAD_REAL=1``.
"""

from __future__ import annotations

import dataclasses
import inspect
import itertools
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prc import paths
from prc.config import load_config
from prc.features import _oplib, all_specs, operational, resolve, weather, weather_fcst
from prc.features._wxlib import _metar_by_station
from prc.features.build import compute_block
from prc.features.registry import FeatureContext

UTC = "UTC"
T0 = pd.Timestamp("2025-03-20", tz=UTC)  # the 16 days span a month boundary (the NM baseline is month-out)
RUNWAYS = {"EDDF": ["25C", "18", "07C"], "LSZH": ["28", "32", "16"]}

H1_BLOCKS = ("op_h1_static", "op_h1_plan", "op_h1_wx", "op_h1_traffic", "op_h1_nm", "op_h1_feed_lab")
H0_BLOCKS = ("op_h0_static", "op_h0_plan", "op_h0_wx", "op_h0_traffic", "op_h0_nm", "op_h0_feed_lab")


# --- the synthetic world -------------------------------------------------------------------------

def _ts(seconds) -> pd.Series:
    return pd.Series(pd.to_datetime(np.asarray(seconds, dtype="float64"), unit="s", utc=True))


def synthetic_world(seed: int = 0, days: int = 16, per_day: int = 110):
    """Movements (departures and arrivals, two airports) and the departures' recorded off-blocks.

    Whole seconds everywhere, as in the real data. Departures carry a schedule, IOBT/LOBT (5 % unmatched),
    an off-block a little after the plan (a few before the H1 anchor), a taxi of 4-25 min (a few stale and
    wrong-date records), NM's AOBT_3 near the off-block, a runway that changes with the hour, a stand.
    Arrivals carry landing, in-block and taxi-in. The last three days are the ranking split.
    """
    rng = np.random.default_rng(seed)
    t0 = T0.timestamp()
    deps, arrs = [], []
    for apt, runways in RUNWAYS.items():
        stands = [f"A{i}" for i in range(1, 13)] + [f"B{i}" for i in range(1, 9)] + ["H1"]
        for d in range(days):
            day0 = t0 + d * 86_400
            n = per_day + int(rng.integers(-10, 10))
            sched = day0 + np.sort(rng.integers(5 * 3600 // 300, 23 * 3600 // 300, n)) * 300.0
            delay = np.where(rng.random(n) < 0.3, rng.integers(1, 60, n) * 60.0, 0.0)
            iobt = sched + delay
            lobt = iobt + np.where(rng.random(n) < 0.05, 300.0, 0.0)
            block = iobt + rng.integers(-1200, 900, n).astype("float64")
            taxi = rng.integers(240, 1500, n).astype("float64")
            stale = rng.random(n) < 0.01
            wrong_date = rng.random(n) < 0.004
            takeoff = block + taxi
            takeoff = np.where(stale, sched + rng.integers(3 * 3600, 6 * 3600, n), takeoff)
            block = np.where(stale, sched, block)
            block = np.where(wrong_date, block - 86_400, block)
            aobt3 = np.floor((block + rng.integers(-120, 180, n)) / 60.0) * 60.0
            matched = rng.random(n) > 0.05
            hour = ((sched - day0) // 3600).astype(int)
            main = np.where((hour // 4) % 2 == 0, runways[0], runways[1])
            rwy = np.where(rng.random(n) < 0.12, runways[2], main).astype(object)
            rwy[rng.random(n) < 0.01] = None
            stand = rng.choice(stands, n).astype(object)
            stand[rng.random(n) < 0.01] = None
            for i in range(n):
                deps.append(
                    {
                        "apt": apt, "is_dep": True, "sched": sched[i], "iobt": iobt[i] if matched[i] else np.nan,
                        "lobt": lobt[i] if matched[i] else np.nan, "aobt3": aobt3[i] if matched[i] else np.nan,
                        "arvt3": takeoff[i] + 3600.0 if matched[i] else np.nan, "eobt1": lobt[i] if matched[i] else np.nan,
                        "mvt": takeoff[i], "block": block[i], "rwy": rwy[i], "stand": stand[i], "taxi": np.nan,
                        "fid": float(len(deps)) if matched[i] else np.nan, "day": d,
                    }
                )
            m = per_day + int(rng.integers(-10, 10))
            land = day0 + np.sort(rng.integers(5 * 3600, 23 * 3600, m)).astype("float64")
            taxi_in = rng.integers(150, 900, m).astype("float64")
            inblock = land + taxi_in
            a_rwy = np.where(rng.random(m) < 0.5, runways[0], runways[1]).astype(object)
            a_stand = rng.choice(stands, m).astype(object)
            missing_in = rng.random(m) < 0.01
            for i in range(m):
                arrs.append(
                    {
                        "apt": apt, "is_dep": False, "sched": land[i] - 600, "iobt": np.nan, "lobt": np.nan,
                        "aobt3": np.nan, "arvt3": land[i] + 30.0, "eobt1": np.nan, "mvt": land[i],
                        "block": np.nan if missing_in[i] else inblock[i], "rwy": a_rwy[i], "stand": a_stand[i],
                        "taxi": np.nan if missing_in[i] else taxi_in[i], "fid": np.nan, "day": d,
                    }
                )
    raw = pd.DataFrame(deps + arrs).sort_values("mvt", kind="stable").reset_index(drop=True)
    raw.index = pd.Index(np.arange(1, len(raw) + 1, dtype="int64") * 7, name="mvt_id")
    is_dep = raw["is_dep"].to_numpy(dtype=bool)
    mvt = pd.DataFrame(
        {
            "FLIGHT_ID_mvt": raw["fid"],
            "MVT_TIME_UTC_mvt": _ts(raw["mvt"]).to_numpy(),
            "BLOCK_TIME_UTC_mvt": _ts(np.where(is_dep, np.nan, raw["block"])).to_numpy(),
            "SCHED_TIME_UTC_mvt": _ts(raw["sched"]).to_numpy(),
            "AIRCRAFT_TYPE_mvt": pd.Categorical(np.where(is_dep, "A320", "B738")),
            "RUNWAY_mvt": pd.Categorical(raw["rwy"]),
            "STAND_mvt": pd.Categorical(raw["stand"]),
            "TAXITIME_SEC_mvt": raw["taxi"].astype("float64"),
            "LOBT_flt": _ts(raw["lobt"]).to_numpy(),
            "IOBT_flt": _ts(raw["iobt"]).to_numpy(),
            "EOBT_1_flt": _ts(raw["eobt1"]).to_numpy(),
            "AOBT_3_flt": _ts(raw["aobt3"]).to_numpy(),
            "ARVT_3_flt": _ts(raw["arvt3"]).to_numpy(),
            "split": pd.Categorical(np.where(raw["day"] >= days - 3, "rank", "train")),
            "is_dep": is_dep,
            "apt": pd.Categorical(raw["apt"]),
        },
        index=raw.index,
    )
    train_dep = is_dep & (raw["day"] < days - 3).to_numpy()
    offblocks = pd.Series(np.floor(raw["block"].to_numpy()[train_dep]), index=raw.index[train_dep], name="offblock")
    return mvt, offblocks


_TABLE_IDS = itertools.count()


def synthetic_weather(tmp_path: Path, extra_metar: list[dict] | None = None, extra_taf: list[dict] | None = None):
    """METAR (half-hourly) and TAF (6-hourly, two segments) tables for both airports, through the real loaders."""
    rng = np.random.default_rng(3)
    rows = []
    for apt in RUNWAYS:
        for k in range(20 * 48):
            when = T0 + pd.Timedelta(minutes=30 * k - 50)
            rows.append(
                {
                    "station": apt, "obs_time": when, "vis_min_m": float(rng.choice([800, 4000, 9999])),
                    "rvr_min_m": np.nan, "ceiling_ft": float(rng.choice([300, 1500, 5000])), "temp_c": float(rng.integers(-5, 10)),
                    "dewpt_c": float(rng.integers(-8, 5)), "precip_int": float(rng.integers(0, 3)), "wx_fog": 0.0,
                    "wx_mist": float(rng.integers(0, 2)), "wx_freezing": 0.0, "wx_ice": 0.0, "wx_snow": float(rng.random() < 0.1),
                    "wx_ts": 0.0, "cb_tcu": 0.0, "wind_dir_deg": float(rng.integers(0, 36) * 10), "wind_kt": float(rng.integers(0, 25)),
                    "gust_kt": np.nan,
                }
            )
    rows += extra_metar or []
    metar = pd.DataFrame(rows).sort_values(["station", "obs_time"], kind="stable")
    tag = next(_TABLE_IDS)  # a fresh file per call: the loaders are memoised on (path, mtime)
    mp = tmp_path / f"metar_{tag}.parquet"
    metar.to_parquet(mp)

    issues, segs = [], []
    seg_cols = weather_fcst.SEGMENT_COLS
    for apt in RUNWAYS:
        for k in range(20 * 4):
            issue = T0 + pd.Timedelta(hours=6 * k - 7)
            issues.append(
                {"station": apt, "taf_id": f"{apt}{k}", "parse_ok": 1, "issue_time": issue, "is_amd": int(k % 5 == 0),
                 "is_cor": 0, "is_cnl": 0, "valid_from": issue + pd.Timedelta(hours=1), "valid_to": issue + pd.Timedelta(hours=25)}
            )
            for j, start in enumerate((issue + pd.Timedelta(hours=1), issue + pd.Timedelta(hours=9))):
                segs.append({"taf_id": f"{apt}{k}", "seg_start": start, **{c: float(rng.integers(0, 3) * (j + 1)) for c in seg_cols}})
    issues += [r for r in (extra_taf or []) if "issue_time" in r]
    segs += [r for r in (extra_taf or []) if "seg_start" in r]
    ip, sp = tmp_path / f"iss_{tag}.parquet", tmp_path / f"seg_{tag}.parquet"
    pd.DataFrame(issues).to_parquet(ip)
    pd.DataFrame(segs).to_parquet(sp)
    return _metar_by_station(str(mp), 0), weather_fcst._taf_by_station(str(ip), str(sp), 0, 0)


@pytest.fixture
def world(tmp_path, monkeypatch):
    mvt, offblocks = synthetic_world()
    metar, tafs = synthetic_weather(tmp_path)
    monkeypatch.setattr(weather, "load_metar", lambda: metar)
    monkeypatch.setattr(weather_fcst, "load_tafs", lambda: tafs)
    state = {"offblocks": offblocks}
    monkeypatch.setattr(_oplib, "load_offblocks", lambda: state["offblocks"])
    return mvt, offblocks, state


def build(mvt: pd.DataFrame, names, dep_index: pd.Index, fit_mask: pd.Series | None = None) -> pd.DataFrame:
    """Run the named blocks through the contract validator, side by side."""
    specs = all_specs()
    if fit_mask is None:
        fit_mask = pd.Series(mvt["split"].eq("train").to_numpy() & mvt["is_dep"].to_numpy(), index=mvt.index)
    parts = []
    for name in names:
        spec = specs[name]
        ctx = FeatureContext(dep_index=dep_index, fit_mask=fit_mask if spec.uses_target else None,
                             target=pd.Series(0.0, index=dep_index) if spec.uses_target else None)
        parts.append(compute_block(spec, mvt, ctx))
    return pd.concat(parts, axis=1)


def spaced_departures(mvt: pd.DataFrame, hours: float = 30.0, split: str = "train") -> pd.Index:
    """Departures at least ``hours`` apart per airport, with a plan (so a perturbation of one cannot reach another)."""
    dep = mvt.loc[mvt["is_dep"] & mvt["split"].eq(split)]
    picked = []
    for _, g in dep.groupby("apt", observed=True):
        last = -np.inf
        t = _oplib.secs(g["MVT_TIME_UTC_mvt"])
        planned = _oplib.planned_offblock(g)
        for i, (idx, ti) in enumerate(zip(g.index, t)):
            if ti - last >= hours * 3600 and np.isfinite(planned[i]) and pd.notna(g["STAND_mvt"].iloc[i]) and pd.notna(g["RUNWAY_mvt"].iloc[i]):
                picked.append(idx)
                last = ti
    return pd.Index(picked, name="mvt_id")


def _shift(mvt: pd.DataFrame, col: str, rows: pd.Index, seconds) -> pd.DataFrame:
    out = mvt.copy()
    out.loc[rows, col] = out.loc[rows, col] + pd.to_timedelta(np.asarray(seconds, dtype="float64"), unit="s")
    return out


def _set(mvt: pd.DataFrame, col: str, rows: pd.Index, seconds) -> pd.DataFrame:
    out = mvt.copy()
    out.loc[rows, col] = pd.to_datetime(np.asarray(seconds, dtype="float64"), unit="s", utc=True)
    return out


def assert_identical(a: pd.DataFrame, b: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(a, b, check_exact=True)


# --- the blocks' contract ----------------------------------------------------------------------------

def test_blocks_build_on_the_synthetic_world(world):
    mvt, _, _ = world
    dep_idx = mvt.index[mvt["is_dep"].to_numpy()]
    X = build(mvt, H1_BLOCKS + H0_BLOCKS, dep_idx)
    assert X.index.equals(dep_idx)
    assert all(c.startswith(("op_h1_", "op_h0_")) for c in X.columns)
    # something is measured: not a frame of NaN and zeros
    for col in ("op_h1_dep_apt_b60", "op_h1_arr_taxiing", "op_h1_rwy_config_age_min", "op_h1_nm_apt_b60_anom_mean",
                "op_h1_backlog_apt", "op_h1_feed_queue_apt", "op_h1_wx_vis_m", "op_h1_taf_vis_m", "op_h1_stand_min_since_inblock",
                "op_h0_feed_queue_apt", "op_h0_delay_vs_plan_min"):
        v = X[col].to_numpy(dtype="float64")
        assert np.isfinite(v).mean() > 0.3 and np.nanstd(v) > 0, col
    rank = mvt.loc[dep_idx, "split"].eq("rank").to_numpy()
    assert X.loc[rank, [c for c in X.columns if c.startswith("op_h0_")]].isna().all().all(), "H0 must be NaN for ranking rows"
    assert X.loc[rank, [c for c in X.columns if "feed" in c]].isna().all().all(), "the feed must be NaN for ranking rows"
    assert X.loc[rank, "op_h1_dep_apt_b60"].notna().all(), "H1 must exist for ranking rows"


def test_no_import_time_work_and_helpers_are_hashed():
    """`build._related_modules` is one level deep: every helper module must sit in operational's namespace."""
    from prc.features.build import _related_modules

    names = {m.__name__ for m in _related_modules(operational)}
    for mod in ("_oplib", "_arrlib", "_proxylib", "_wxlib", "congestion", "core", "stand_runway", "temporal",
                "turnaround", "weather", "weather_fcst"):
        assert f"prc.features.{mod}" in names, mod
    src = inspect.getsource(operational)
    body = src.split('"""', 2)[2]
    top_level_calls = [ln for ln in body.splitlines() if re.match(r"^[a-z_]+\(", ln)]
    assert not top_level_calls, top_level_calls


# --- T1: own events ----------------------------------------------------------------------------------

def _t1_frames(mvt: pd.DataFrame, rows: pd.Index):
    """The fit mask of training: every training departure, the perturbed ones included."""
    return pd.Series(mvt["split"].eq("train").to_numpy() & mvt["is_dep"].to_numpy(), index=mvt.index)


def t1_rows(mvt: pd.DataFrame, hours: float = 30.0) -> pd.Index:
    """Spaced training departures of ONE planned month.

    Each query reads the NM baseline that leaves out its own planned month; a perturbed flight of another month
    is legitimately inside that baseline, so perturbing several flights at once is only a clean T1 when they
    share the month.
    """
    rows = spaced_departures(mvt, hours)
    month = _oplib.baseline_month(mvt.loc[rows])
    top = pd.Series(month).mode().iloc[0]
    return rows[month == top]


@pytest.mark.parametrize("mode", ["plus600", "minus300", "cross_h1", "cross_bucket"])
def test_t1_own_takeoff_invariance(world, mode):
    mvt, offblocks, _ = world
    rows = t1_rows(mvt)
    assert len(rows) >= 12
    fit = _t1_frames(mvt, rows)
    assert fit.loc[rows].all()  # the perturbed flights are fit rows, as in training
    base = build(mvt, H1_BLOCKS + H0_BLOCKS, rows, fit)
    if mode == "plus600":
        moved = _shift(mvt, "MVT_TIME_UTC_mvt", rows, 600)
    elif mode == "minus300":
        moved = _shift(mvt, "MVT_TIME_UTC_mvt", rows, -300)
    elif mode == "cross_h1":  # the own takeoff crosses the H1 anchor: it now lies 60 s before it
        moved = _set(mvt, "MVT_TIME_UTC_mvt", rows, _oplib.anchor_h1(mvt.loc[rows]) - 60)
    else:  # ... into a 15-min bucket complete before the anchor (the runway configuration's slow path)
        a = _oplib.anchor_h1(mvt.loc[rows])
        moved = _set(mvt, "MVT_TIME_UTC_mvt", rows, (np.floor(a / 900) - 2) * 900 + 17)
    assert_identical(base, build(moved, H1_BLOCKS + H0_BLOCKS, rows, fit))


def test_t1_crossing_case_does_put_the_own_takeoff_into_the_windows(world):
    """Guard the guard: without self-exclusion the crossing case would change the counts."""
    mvt, _, _ = world
    rows = spaced_departures(mvt)
    a = _oplib.anchor_h1(mvt.loc[rows])
    moved = _set(mvt, "MVT_TIME_UTC_mvt", rows, a - 60)
    dep = _oplib.departures(moved)
    t = pd.Series(_oplib.secs(dep["MVT_TIME_UTC_mvt"]), index=dep.index)
    g = dep["apt"].astype("string")
    raw = _oplib.backward_count(t, g, a, moved.loc[rows, "apt"].astype("string"), 900)  # no own_t: own counted
    excl = _oplib.backward_count(t, g, a, moved.loc[rows, "apt"].astype("string"), 900, t.reindex(rows).to_numpy())
    assert (raw - excl == 1).all()


@pytest.mark.parametrize("shift", [600, -600, "before_a"])
def test_t1b_own_nm_actuals_invariance(world, shift):
    mvt, _, _ = world
    rows = t1_rows(mvt)
    fit = _t1_frames(mvt, rows)
    base = build(mvt, H1_BLOCKS + H0_BLOCKS, rows, fit)
    if shift == "before_a":
        a = _oplib.anchor_h1(mvt.loc[rows])
        moved = _set(mvt, "AOBT_3_flt", rows, a - 300)
        moved = _set(moved, "ARVT_3_flt", rows, a - 200)
    else:
        moved = _shift(_shift(mvt, "AOBT_3_flt", rows, shift), "ARVT_3_flt", rows, shift)
    assert_identical(base, build(moved, H1_BLOCKS + H0_BLOCKS, rows, fit))


@pytest.mark.parametrize("shift", [600, -600, "before_a"])
def test_t1c_own_offblock_invariance_at_h1(world, shift):
    mvt, offblocks, state = world
    rows = t1_rows(mvt)
    fit = _t1_frames(mvt, rows)
    base = build(mvt, H1_BLOCKS, rows, fit)
    moved = offblocks.copy()
    if shift == "before_a":
        moved.loc[rows] = _oplib.anchor_h1(mvt.loc[rows]) - 120
    else:
        moved.loc[rows] = moved.loc[rows] + shift
    state["offblocks"] = moved
    assert_identical(base, build(mvt, H1_BLOCKS, rows, fit))
    # at H0 the anchor itself moves: those columns may change, and do
    after = build(mvt, ("op_h0_static", "op_h0_plan"), rows, fit)
    state["offblocks"] = offblocks
    before = build(mvt, ("op_h0_static", "op_h0_plan"), rows, fit)
    assert not after.equals(before)


# --- T2: future deletion ---------------------------------------------------------------------------

def censor(mvt: pd.DataFrame, offblocks: pd.Series, a: float) -> tuple[pd.DataFrame, pd.Series]:
    """Every event after ``a`` removed: takeoffs, landings, in-blocks (and taxi-in), AOBT_3, recorded off-blocks."""
    out = mvt.copy()
    t = _oplib.secs(out["MVT_TIME_UTC_mvt"])
    out.loc[t > a, "MVT_TIME_UTC_mvt"] = pd.NaT
    inb = _oplib.secs(out["BLOCK_TIME_UTC_mvt"])
    late_in = inb > a
    out.loc[late_in, "BLOCK_TIME_UTC_mvt"] = pd.NaT
    out.loc[late_in, "TAXITIME_SEC_mvt"] = np.nan
    out.loc[_oplib.secs(out["AOBT_3_flt"]) > a, "AOBT_3_flt"] = pd.NaT
    out.loc[_oplib.secs(out["ARVT_3_flt"]) > a, "ARVT_3_flt"] = pd.NaT
    ob = offblocks.where(~(offblocks > a))
    return out, ob


def _freeze_archives(monkeypatch, mvt: pd.DataFrame, fit: pd.Series) -> None:
    """The archive tables (arrival taxi-in baseline, NM proxy baseline), computed once on the full frame."""
    frozen_arr = _oplib.arrival_anomaly(mvt)
    dep = _oplib.departures(mvt)
    frozen_nm = _oplib.nm_baseline(dep, fit.reindex(dep.index).to_numpy())
    monkeypatch.setattr(_oplib, "arrival_anomaly", lambda m: frozen_arr)
    monkeypatch.setattr(_oplib, "nm_baseline", lambda d, f, s=20: frozen_nm)


def censor_attributes(mvt: pd.DataFrame, a: float, keep) -> pd.DataFrame:
    """Runway and stand removed from every other flight that moves after ``a`` (takeoff or landing > a)."""
    out = mvt.copy()
    later = (_oplib.secs(mvt["MVT_TIME_UTC_mvt"]) > a) & ~mvt.index.isin(list(keep))
    out.loc[later, "RUNWAY_mvt"] = np.nan
    out.loc[later, "STAND_mvt"] = np.nan
    return out


@pytest.mark.parametrize("horizon", ["h1", "h0"])
def test_t2_future_deletion_invariance(world, monkeypatch, horizon):
    mvt, offblocks, state = world
    fit = pd.Series(mvt["split"].eq("train").to_numpy() & mvt["is_dep"].to_numpy(), index=mvt.index)
    # archive tables are held fixed: computed once on the full frame
    _freeze_archives(monkeypatch, mvt, fit)

    blocks = H1_BLOCKS if horizon == "h1" else H0_BLOCKS
    rng = np.random.default_rng(11)
    cand = mvt.index[(mvt["is_dep"] & mvt["split"].eq("train")).to_numpy()]
    rows = pd.Index(rng.choice(cand, 24, replace=False), name="mvt_id")
    for r in rows:
        q = mvt.loc[[r]]
        a = (_oplib.anchor_h1(q) if horizon == "h1" else _oplib.anchor_h0(q, offblocks))[0]
        if not np.isfinite(a):
            continue
        idx = pd.Index([r], name="mvt_id")
        full = build(mvt, blocks, idx, fit)
        cm, cob = censor(mvt, offblocks, a)
        assert cm["MVT_TIME_UTC_mvt"].isna().sum() > 100  # the censoring removes events
        state["offblocks"] = cob
        cut = build(cm, blocks, idx, fit)
        state["offblocks"] = offblocks
        assert_identical(full, cut)


@pytest.mark.parametrize("horizon", ["h1", "h0"])
def test_t2b_future_runway_and_stand_move_only_the_declared_o_star_columns(world, monkeypatch, horizon):
    """The runway / pier groupings of not-yet-airborne flights use their recorded runway and stand (tier O*).

    Censoring those attributes too (every other flight moving after ``a``) may change exactly the declared
    columns and nothing else; and it does change them (else the declaration would be stale).
    """
    mvt, offblocks, state = world
    fit = pd.Series(mvt["split"].eq("train").to_numpy() & mvt["is_dep"].to_numpy(), index=mvt.index)
    _freeze_archives(monkeypatch, mvt, fit)
    blocks = H1_BLOCKS if horizon == "h1" else H0_BLOCKS
    prefix = f"op_{horizon}_"
    allowed = {prefix + c for c in _oplib.O_STAR_FUTURE_ATTRIBUTES}
    rng = np.random.default_rng(12)
    cand = mvt.index[(mvt["is_dep"] & mvt["split"].eq("train")).to_numpy()]
    moved_cols: set[str] = set()
    for r in rng.choice(cand, 24, replace=False):
        q = mvt.loc[[r]]
        a = (_oplib.anchor_h1(q) if horizon == "h1" else _oplib.anchor_h0(q, offblocks))[0]
        if not np.isfinite(a):
            continue
        idx = pd.Index([r], name="mvt_id")
        full = build(mvt, blocks, idx, fit)
        cm, cob = censor(censor_attributes(mvt, a, keep=idx), offblocks, a)
        state["offblocks"] = cob
        cut = build(cm, blocks, idx, fit)
        state["offblocks"] = offblocks
        same = [c for c in full.columns if c not in allowed]
        assert_identical(full[same], cut[same])
        moved_cols |= {c for c in allowed if not full[c].equals(cut[c])}
    assert moved_cols & {prefix + "backlog_rwy", prefix + "backlog_pier"}, "the O* declaration has no teeth"


# --- primitives against brute force --------------------------------------------------------------

@pytest.fixture
def events():
    rng = np.random.default_rng(5)
    n, m = 300, 120
    ref_t = np.sort(rng.integers(0, 20_000, n)).astype("float64")
    ref_g = rng.choice(["A", "B", "C"], n).astype(object)
    q_t = rng.integers(0, 20_000, m).astype("float64")
    q_t[:20] = ref_t[rng.integers(0, n, 20)]  # queries exactly at an event
    q_t[20:40] = ref_t[rng.integers(0, n, 20)] + 900  # an event exactly at a - 900
    q_t[40:50] = ref_t[rng.integers(0, n, 10)] + 1800  # an event exactly at a - 1800
    q_g = rng.choice(["A", "B", "C", None], m, p=[0.3, 0.3, 0.3, 0.1]).astype(object)
    own = rng.integers(0, n, m)
    own[rng.random(m) < 0.2] = -1
    # the OWN event on the window edges: exactly at a, at a - 900, at a - 1800, and one second outside each
    for lo, hi, offset in ((50, 62, 0.0), (62, 74, 900.0), (74, 86, 1800.0), (86, 92, 901.0), (92, 98, 1801.0)):
        own[lo:hi] = rng.integers(0, n, hi - lo)
        q_t[lo:hi] = ref_t[own[lo:hi]] + offset
    return ref_t, ref_g, q_t, q_g, own


def test_the_events_fixture_puts_the_own_event_on_every_edge(events):
    ref_t, _, q_t, _, own = events
    has = own >= 0
    d = q_t[has] - ref_t[own[has]]
    for edge in (0.0, 900.0, 1800.0, 901.0, 1801.0):
        assert (d == edge).sum() >= 5, edge


def test_backward_count_with_self_exclusion_matches_brute_force(events):
    ref_t, ref_g, q_t, q_g, own = events
    own_t = np.where(own >= 0, ref_t[np.clip(own, 0, None)], np.nan)
    # the own event must be in the query's own group (it is the flight itself)
    q_g = np.where(own >= 0, ref_g[np.clip(own, 0, None)], q_g).astype(object)
    got = _oplib.backward_count(ref_t, ref_g, q_t, q_g, 900, own_t)
    for i in range(len(q_t)):
        want = 0
        if q_g[i] is not None:
            want = sum(1 for j in range(len(ref_t)) if j != own[i] and ref_g[j] == q_g[i] and q_t[i] - 900 <= ref_t[j] <= q_t[i])
        assert got[i] == want, i


def test_backward_stats_with_self_exclusion_matches_brute_force(events):
    ref_t, ref_g, q_t, q_g, own = events
    rng = np.random.default_rng(6)
    v = rng.normal(100, 300, len(ref_t)).round(1)
    v[rng.random(len(v)) < 0.1] = np.nan
    q_g = np.where(own >= 0, ref_g[np.clip(own, 0, None)], q_g).astype(object)
    own_t = np.where(own >= 0, ref_t[np.clip(own, 0, None)], np.nan)
    own_v = np.where(own >= 0, v[np.clip(own, 0, None)], np.nan)
    n, mean, share = _oplib.backward_stats(ref_t, ref_g, v, q_t, q_g, 1800, own_t, own_v, high=300)
    vr = np.round(v)
    for i in range(len(q_t)):
        sel = [j for j in range(len(ref_t)) if j != own[i] and q_g[i] is not None and ref_g[j] == q_g[i]
               and q_t[i] - 1800 <= ref_t[j] <= q_t[i] and np.isfinite(vr[j])]
        assert n[i] == len(sel), i
        if sel:
            assert mean[i] == pytest.approx(np.mean(vr[sel]), abs=1e-9)
            assert share[i] == pytest.approx(np.mean(vr[sel] > 300), abs=1e-12)
        else:
            assert np.isnan(mean[i]) and np.isnan(share[i])


def test_in_progress_matches_brute_force(events):
    ref_t, ref_g, q_t, q_g, own = events
    rng = np.random.default_rng(7)
    start = ref_t
    end = start + rng.integers(-100, 4000, len(start))
    end[rng.random(len(end)) < 0.1] = np.nan  # not ended (or censored)
    q_g = np.where(own >= 0, ref_g[np.clip(own, 0, None)], q_g).astype(object)
    os_ = np.where(own >= 0, start[np.clip(own, 0, None)], np.nan)
    oe = np.where(own >= 0, end[np.clip(own, 0, None)], np.nan)
    # intervals ending exactly at a query time, the own interval included (start <= a < stop: not in progress)
    own_rows = np.flatnonzero((own >= 0) & (start[np.clip(own, 0, None)] <= q_t))[:15]
    end[own[own_rows]] = q_t[own_rows]
    others = rng.integers(0, len(start), 20)
    ok = start[others] <= q_t[100:120]
    end[others[ok]] = q_t[100:120][ok]
    oe = np.where(own >= 0, end[np.clip(own, 0, None)], np.nan)
    assert (oe == q_t).sum() >= 10 and np.isin(q_t, end).sum() >= 20
    for cap in (None, 3600.0):
        got = _oplib.in_progress(start, end, ref_g, q_t, q_g, os_, oe, max_dur=cap)
        for i in range(len(q_t)):
            want = 0
            for j in range(len(start)):
                if j == own[i] or q_g[i] is None or ref_g[j] != q_g[i]:
                    continue
                e = end[j] if cap is None else np.fmin(end[j], start[j] + cap)
                stop = max(start[j], e) if np.isfinite(e) else np.inf
                if start[j] <= q_t[i] < stop:
                    want += 1
            assert got[i] == want, (cap, i)


def test_backward_stats_refuses_an_own_time_without_its_value():
    t = np.array([0.0, 10.0])
    g = np.array(["A", "A"], dtype=object)
    with pytest.raises(ValueError):
        _oplib.backward_stats(t, g, [1.0, 2.0], [10.0], ["A"], 100, own_t=[10.0])
    with pytest.raises(ValueError):
        _oplib.backward_stats(t, g, [1.0, 2.0], [10.0], ["A"], 100, own_v=[2.0])
    n, mean, _ = _oplib.backward_stats(t, g, [1.0, 2.0], [10.0], ["A"], 100, own_t=[10.0], own_v=[2.0])
    assert n[0] == 1 and mean[0] == 1.0


def test_last_gap_is_self_aware_and_matches_brute_force(events):
    ref_t, ref_g, q_t, q_g, own = events
    ids = pd.Index(np.arange(len(ref_t)) * 3 + 1)
    q_ids = pd.Index(np.where(own >= 0, ids.to_numpy()[np.clip(own, 0, None)], -5))
    q_g = np.where(own >= 0, ref_g[np.clip(own, 0, None)], q_g).astype(object)
    # half the own events sit exactly at the query time, the case `neighbour_gap(strict=False)` would return as 0
    q_t = q_t.copy()
    at = (own >= 0) & (np.arange(len(q_t)) % 2 == 0)
    q_t[at] = ref_t[own[at]]
    got = _oplib.last_gap(pd.Series(ref_t, index=ids), pd.Series(ref_g, index=ids), q_t, q_g, q_ids)
    for i in range(len(q_t)):
        cands = [q_t[i] - ref_t[j] for j in range(len(ref_t))
                 if j != own[i] and q_g[i] is not None and ref_g[j] == q_g[i] and ref_t[j] <= q_t[i]]
        want = min(cands) if cands else np.nan
        assert (np.isnan(got[i]) and np.isnan(want)) or got[i] == want, i


def test_runway_mix_matches_brute_force():
    rng = np.random.default_rng(8)
    n, m = 400, 150
    t = np.sort(rng.integers(0, 30_000, n)).astype("float64")
    apt = rng.choice(["EDDF", "LSZH"], n).astype(object)
    rwy = rng.choice(["25C", "18", "07C", None], n, p=[0.45, 0.35, 0.15, 0.05]).astype(object)
    q_ix = rng.integers(0, n, m)
    q = t[q_ix] + rng.integers(-600, 1200, m)
    qa, qr = apt[q_ix], rwy[q_ix]
    own_t, own_r = t[q_ix], rwy[q_ix]
    distinct, share, dominant = _oplib.runway_mix(t, apt, rwy, q, qa, qr, 3600, own_t, own_r)
    for i in range(m):
        c = {}
        for j in range(n):
            if j == q_ix[i] or apt[j] != qa[i] or rwy[j] is None:
                continue
            if q[i] - 3600 <= t[j] <= q[i]:
                c[rwy[j]] = c.get(rwy[j], 0) + 1
        assert distinct[i] == len(c), i
        total = sum(c.values())
        if qr[i] is None or total == 0:
            assert np.isnan(share[i]) and np.isnan(dominant[i])
        else:
            own_c = c.get(qr[i], 0)
            assert share[i] == pytest.approx(own_c / total)
            assert dominant[i] == float(own_c > 0 and own_c == max(c.values()))


def _brute_config_age(t, apt, rwy, q, qa):
    """Modes of complete 15-min buckets before q, per airport; minutes since the latest change, capped at 12 h."""
    out = np.full(len(q), np.nan)
    for i in range(len(q)):
        B = np.floor(q[i] / 900) - 1
        buckets: dict[int, dict] = {}
        for j in range(len(t)):
            if apt[j] != qa[i] or rwy[j] is None or not np.isfinite(t[j]):
                continue
            b = int(t[j] // 900)
            buckets.setdefault(b, {})
            buckets[b][rwy[j]] = buckets[b].get(rwy[j], 0) + 1
        bks = sorted(buckets)
        modes = [sorted(buckets[b].items(), key=lambda kv: (-kv[1], kv[0]))[0][0] for b in bks]
        idx = [k for k, b in enumerate(bks) if b <= B]
        if not idx:
            continue
        k = idx[-1]
        while k > 0 and modes[k - 1] == modes[k]:
            k -= 1
        out[i] = min((q[i] - bks[k] * 900) / 60.0, 720.0)
    return out


def test_config_age_matches_brute_force_and_the_slow_path():
    rng = np.random.default_rng(9)
    n = 500
    t = np.sort(rng.integers(0, 40 * 3600, n)).astype("float64")
    apt = rng.choice(["EDDF", "LSZH"], n).astype(object)
    rwy = np.where((t // 7200) % 2 == 0, "25C", "18").astype(object)
    rwy[rng.random(n) < 0.2] = "07C"
    m = 120
    q = rng.integers(0, 42 * 3600, m).astype("float64")
    qa = rng.choice(["EDDF", "LSZH"], m).astype(object)
    fast = _oplib.config_age_minutes(t, apt, rwy, q, qa)
    slow = _oplib.config_age_minutes(t, apt, rwy, q, qa, force_slow=True)
    want = _brute_config_age(t, apt, rwy, q, qa)
    np.testing.assert_array_equal(fast, want)
    np.testing.assert_array_equal(slow, want)


def test_config_age_excludes_an_own_takeoff_before_the_anchor():
    rng = np.random.default_rng(10)
    n = 400
    t = np.sort(rng.integers(0, 30 * 3600, n)).astype("float64")
    apt = np.full(n, "EDDF", dtype=object)
    rwy = np.where((t // 5400) % 2 == 0, "25C", "18").astype(object)
    own = rng.choice(n, 60, replace=False)
    q = t[own] + rng.integers(0, 3 * 3600, 60)
    # own takeoffs exactly at a bucket start, anchors at that bucket's edges +0 / 1 / 899 / 900 / 901 s (+ 0-2 buckets)
    t[own[:20]] = np.floor(t[own[:20]] / 900) * 900
    q[:20] = t[own[:20]] + np.array([0, 1, 899, 900, 901] * 4) + 900 * rng.integers(0, 3, 20)
    assert (t[own] % 900 == 0).sum() >= 20
    got = _oplib.config_age_minutes(t, apt, rwy, q, apt[own], t[own], rwy[own])
    for i, j in enumerate(own):
        keep = np.arange(n) != j
        want = _oplib.config_age_minutes(t[keep], apt[keep], rwy[keep], q[i : i + 1], apt[own][i : i + 1])
        assert got[i] == want[0] or (np.isnan(got[i]) and np.isnan(want[0])), i


# --- weather causality -------------------------------------------------------------------------------

def _late_reports(apt: str, when: pd.Timestamp, tag: str):
    metar = {"station": apt, "obs_time": when, "vis_min_m": 50.0, "rvr_min_m": 100.0, "ceiling_ft": 0.0,
             "temp_c": -20.0, "dewpt_c": -20.0, "precip_int": 3.0, "wx_fog": 1.0, "wx_mist": 1.0,
             "wx_freezing": 1.0, "wx_ice": 1.0, "wx_snow": 1.0, "wx_ts": 1.0, "cb_tcu": 1.0,
             "wind_dir_deg": 90.0, "wind_kt": 60.0, "gust_kt": 80.0}
    taf = [
        {"station": apt, "taf_id": tag, "parse_ok": 1, "issue_time": when, "is_amd": 1, "is_cor": 0, "is_cnl": 0,
         "valid_from": when - pd.Timedelta(hours=1), "valid_to": when + pd.Timedelta(hours=24)},
        {"taf_id": tag, "seg_start": when - pd.Timedelta(hours=1), **{c: 99.0 for c in weather_fcst.SEGMENT_COLS}},
    ]
    return metar, taf


def test_weather_after_the_anchor_changes_nothing(tmp_path, monkeypatch):
    """One query at a time (a report after one query's anchor is legitimately before a later query's)."""
    mvt, _ = synthetic_world(days=6)
    rows = spaced_departures(mvt, hours=20)[:6]
    a = _oplib.anchor_h1(mvt.loc[rows])

    def wx(q, aq, extra_metar=None, extra_taf=None):
        metar, tafs = synthetic_weather(tmp_path, extra_metar, extra_taf)
        monkeypatch.setattr(weather, "load_metar", lambda: metar)
        monkeypatch.setattr(weather_fcst, "load_tafs", lambda: tafs)
        return _oplib.wx_at(q, aq)

    lag = _oplib.METAR_LAG_S
    at = lambda sec: pd.Timestamp(sec, unit="s", tz=UTC)  # noqa: E731
    changed_before = 0
    for k, r in enumerate(rows):
        q, aq = mvt.loc[[r]], a[k : k + 1]
        base = wx(q, aq)
        metar_cols = [c for c in base.columns if not c.startswith("taf_")]
        taf_cols = [c for c in base.columns if c.startswith("taf_")]
        apt = str(mvt.loc[r, "apt"])
        late_m, late_t = _late_reports(apt, at(aq[0] + 1), f"late{k}")  # 1 s after the anchor
        assert_identical(base, wx(q, aq, [late_m], late_t))
        # a METAR observed less than METAR_LAG_S before the anchor is not yet disseminated: nothing changes
        recent_m, _ = _late_reports(apt, at(aq[0] - lag + 1), f"recent{k}")
        assert_identical(base, wx(q, aq, [recent_m], None))
        # the conventions pinned: a METAR observed exactly METAR_LAG_S before the anchor counts, ...
        edge_m, _ = _late_reports(apt, at(aq[0] - lag), f"edge{k}")
        moved = wx(q, aq, [edge_m], None)
        assert not base[metar_cols].equals(moved[metar_cols]) and base[taf_cols].equals(moved[taf_cols])
        # ... and a TAF issued exactly at the anchor counts
        _, edge_t = _late_reports(apt, at(aq[0]), f"tafedge{k}")
        moved = wx(q, aq, None, edge_t)
        assert not base[taf_cols].equals(moved[taf_cols]) and base[metar_cols].equals(moved[metar_cols])
        # ... while the same reports well before the anchor do change it
        early_m, early_t = _late_reports(apt, at(aq[0] - lag - 61), f"early{k}")
        changed_before += not base.equals(wx(q, aq, [early_m], early_t))
    assert changed_before == len(rows)


def test_weather_reads_only_the_shadow_columns():
    """The shadow frame handed to the weather blocks carries the anchor and nothing of the flight's takeoff."""
    mvt, _ = synthetic_world(days=2)
    q = mvt.loc[mvt["is_dep"]].head(5)
    a = _oplib.anchor_h1(q)
    shadow = _oplib.shadow_departures(q, a)
    assert list(shadow.columns) == ["apt", "RUNWAY_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt"]
    np.testing.assert_array_equal(_oplib.secs(shadow["MVT_TIME_UTC_mvt"]), a)
    for fn in (weather.wx_now, weather.wx_past, weather_fcst.taf_now):
        src = inspect.getsource(fn)
        assert "_departures(mvt, ctx" in src
    for fn in (weather._departures, weather_fcst._departures):
        assert "MVT_TIME_UTC_mvt" in inspect.getsource(fn)


# --- configs ---------------------------------------------------------------------------------------

ALLOWED_BASE = {"core_keys", "stand_runway_keys", "aircraft_identity", "taxi_distance", "taxi_route", "stand_geometry"}
OP_CONFIGS = sorted(p.name for p in paths.CONFIGS.glob("op_*.yaml"))
HIND_GROUP = "op_lab_h0_hind"
HIND_CONFIGS = ("op_h0_r5_hind.yaml",)  # lab, HINDSIGHT (own takeoff): holdout only, never refit, never board
MAE_CONFIGS = ("op_h1_r4_mae.yaml", "op_h1_r4_mae_fast.yaml",  # metric sensitivity: MAE loss and early stopping, holdout only
               "op_h1_r4_nodoy_mae_exact.yaml")
# the H1 median members (rule M), each with MAE loss, refitted, on the CPU: op_h1_r4_nodoy_mae_exact_cpu is the
# superseded deliverable's, for which rule M passed (findings 10.35; kept on record); op_h1_f1_plan_mae_exact_cpu is the
# adopted deliverable F1's, for rule M's re-run (findings 10.37)
H1_MEDIAN = ("op_h1_r4_nodoy_mae_exact_cpu.yaml", "op_h1_f1_plan_mae_exact_cpu.yaml")
# the H1 deliverable until 13 Oct (findings 10.35: rule L picks one, refitted), superseded by op_h1_f1_plan
# (findings 10.37) and kept on record
H1_DELIVERABLE = ("op_h1_r4_nodoy.yaml", "op_h1_r4_nodoy_lr04.yaml")
H1_SENSITIVITIES = ("op_h1_r4_nodoy_fwd.yaml", "op_h1_r4_nodoy_noplan.yaml", "op_h1_r4_nodoy_notaf.yaml",
                    "op_h1_r4_nodoy_rwyblind.yaml", "op_h1_r4_nodoy_mae_exact.yaml")  # findings 10.35, holdout only


LONG_CONFIGS = (  # the long horizons (tests/test_operational_long.py) and the ADS-B archive rungs
    "op_hl_r0.yaml",
    "op_hl3_r2.yaml", "op_hl3_r3.yaml", "op_hl3_r4.yaml", "op_hl3_r4_arch.yaml",
    "op_hl3_r3_rwyblind.yaml", "op_hl3_r4_rwyblind.yaml", "op_hl3_r4_arch_rwyblind.yaml",
    "op_hl3_r3_rwyblind_fwd.yaml", "op_hl3_r4_rwyblind_fwd.yaml",
    "op_hl24_r2.yaml", "op_hl24_r3.yaml", "op_hl24_r4.yaml", "op_hl24_r4_arch.yaml",
    "op_hl24_r3_rwyblind.yaml", "op_hl24_r4_rwyblind.yaml", "op_hl24_r4_arch_rwyblind.yaml",
    "op_hl24_r4_arch_rwyblind_pier.yaml",
    "op_hl24_r3_rwyblind_fwd.yaml", "op_hl24_r4_rwyblind_fwd.yaml", "op_hl24_r4_arch_rwyblind_fwd.yaml",
    "op_h1_r4_arch.yaml", "op_h0_r5_arch.yaml",
)
PAIRS = ("febdec", "marnov", "aproct", "maysep", "junaug")
ADSB_CONFIGS = ("op_h1_r6.yaml", "op_h1_r6_s43.yaml", "op_h1_r6_shift7.yaml", "op_h1_r2_adsb.yaml", "op_h0_r4.yaml",
                "op_h0_r4_adsb.yaml", *(f"op_h1_r6_{p}.yaml" for p in PAIRS))


def test_the_op_configs_exist():
    for name in ("op_base.yaml", "op_h1_r0.yaml", "op_h1_r1.yaml", "op_h1_r2.yaml", "op_h1_r3.yaml", "op_h1_r4.yaml",
                 "op_h1_r5.yaml", "op_h0_r5.yaml", "op_h1_r4_rwyblind.yaml", "op_h1_r1_winter.yaml", "op_h1_r2_winter.yaml",
                 "op_h1_r4_mae.yaml", "op_h1_r4_mae_fast.yaml", "op_h0_r5_hind.yaml", *LONG_CONFIGS,
                 *ADSB_CONFIGS):
        assert name in OP_CONFIGS, name


@pytest.mark.parametrize("name", OP_CONFIGS)
def test_op_config_allowlist(name):
    cfg = load_config(name)
    specs = resolve(cfg.features.include, cfg.features.exclude)
    blocks = {s.name for s in specs}
    assert blocks <= ALLOWED_BASE | {s for s in all_specs() if s.startswith("op_")}, blocks - ALLOWED_BASE
    assert "stand_runway_frequency" not in blocks and "operator_presence" not in blocks
    assert "takeoff_epoch" in cfg.features.drop_columns
    assert not cfg.model.mixture.enabled
    assert cfg.target.residual_of is None and cfg.target.transform == "none" and cfg.target.clip is None
    assert cfg.model.mixture.regressor_anchor is None and cfg.model.mixture.window_column is None
    assert not cfg.model.per_airport and not cfg.model.month_weights
    assert cfg.data.clean.max_taxi_sec == 7200 and cfg.data.clean.day_offset_tol_sec == 3600 and cfg.data.clean.drop_from_fit
    outputs = [c for s in specs for c in s.outputs]
    assert not [c for c in outputs if "eobt" in c.lower()], "EOBT_1 is not an operational input"
    assert not [c for c in outputs if re.search(r"push_queue|nm_queue|aobt", c.lower())], "no NM-push queue"
    # a config is H1 or H0, never both
    assert not (any(b.startswith("op_h1_") for b in blocks) and any(b.startswith("op_h0_") for b in blocks))
    # ... and one anchor family at most: H1, H0, 3 h or 24 h ahead; the lead-free long blocks (op_hl_*: plan data)
    # only with a long lead or alone, never with H1 / H0 (the archive op_adsb_archive is horizon-free)
    families = {f for f in ("h1", "h0", "hl3", "hl24") if any(b.startswith(f"op_{f}_") for b in blocks)}
    assert len(families) <= 1, (name, families)
    assert not (any(b.startswith("op_hl_") for b in blocks) and families & {"h1", "h0"}), (name, families)
    if cfg.model.refit_on_all:
        lab = [s.name for s in specs if paths.TARGET_FILE in s.external or s.name.startswith("op_h0_") or s.name.endswith("_lab")]
        assert not lab, f"{name} is a board config but resolves lab blocks {lab}"
    # the hindsight block (own takeoff; lab, never operational) only in its own lab rung (tests/test_operational_hind.py)
    hind = [s.name for s in specs if s.group == HIND_GROUP]
    assert not hind or (name in HIND_CONFIGS and not cfg.model.refit_on_all), f"{name} resolves the hindsight block {hind}"
    assert name not in HIND_CONFIGS or hind, f"{name} is a hindsight rung without the hindsight block"
    # the loss: RMSE everywhere but the MAE metric-sensitivity rung, whose early stopping is on MAE too
    loss = cfg.model.params.get("loss_function", "RMSE")
    assert loss == ("MAE" if name in MAE_CONFIGS + H1_MEDIAN else "RMSE"), (name, loss)
    assert cfg.model.params.get("eval_metric", "RMSE") == loss, name
    if name in MAE_CONFIGS:
        assert not cfg.model.refit_on_all, f"{name} is a metric-sensitivity rung, never a board run"
    # every rung on the GPU, but the H1 median member: CatBoost's GPU fails with exact leaves (CUDA error 700)
    assert cfg.model.params.get("task_type") == ("CPU" if name in H1_MEDIAN else "GPU"), name
    assert str(cfg.model.params.get("devices")) == "0"
    if name == "op_h1_r4_mae_fast.yaml":   # MAE converges only at a larger step (op_h1_r4_mae hit its cap)
        assert cfg.model.params.get("iterations") == 30000 and cfg.model.params.get("learning_rate") == 0.25
    elif name == "op_h1_r4_nodoy_lr04.yaml":   # rule L (findings 10.35): half the step, as exp065-exp067
        assert cfg.model.params.get("iterations") == 20000 and cfg.model.params.get("learning_rate") == 0.04
        assert cfg.model.params.get("od_wait") == 400
    elif name in ("op_h1_r4_nodoy_mae_exact.yaml", *H1_MEDIAN):   # rule M: exact leaves need plain boosting on the GPU
        assert cfg.model.params.get("iterations") == 16000 and cfg.model.params.get("learning_rate") == 0.08
        assert cfg.model.params.get("leaf_estimation_method") == "Exact" and cfg.model.params.get("boosting_type") == "Plain"
    else:
        assert cfg.model.params.get("iterations") == 8000 and cfg.model.params.get("learning_rate") == 0.08
    assert cfg.model.params.get("depth") == 7 and cfg.model.params.get("border_count") == 128


def test_permutation_and_refit_per_rung():
    perm = {n: load_config(n).evaluation.permutation.enabled for n in OP_CONFIGS}
    # op_h1_f4_adsb: rule T's concept member (findings 10.36), each F block's value
    permuted = {"op_h1_r4.yaml", "op_h1_r6.yaml", "op_h0_r5.yaml", "op_h0_r4_adsb.yaml", "op_h1_f4_adsb.yaml"}
    assert {k for k, v in perm.items() if v} == permuted
    refit = [n for n in OP_CONFIGS if load_config(n).model.refit_on_all]
    # the adopted H1 deliverable F1 and its median member (findings 10.37); the H1 board rungs (R4, and R6 with the
    # ADS-B surface picture); the superseded H1 deliverable op_h1_r4_nodoy with its median member and rule L's
    # alternative, which keep their refit flag so that their runs keep their hashes; and the runway-blind deliverable
    # candidates of each long lead, r3 and r4 (rule R; findings 10.35); all refit for the final dataset
    assert refit == ["op_h1_f1_plan.yaml", "op_h1_f1_plan_mae_exact_cpu.yaml",
                     "op_h1_r4.yaml", "op_h1_r4_nodoy.yaml", "op_h1_r4_nodoy_lr04.yaml", "op_h1_r4_nodoy_mae_exact_cpu.yaml",
                     "op_h1_r6.yaml",
                     "op_hl24_r3_rwyblind.yaml", "op_hl24_r4_rwyblind.yaml", "op_hl3_r3_rwyblind.yaml",
                     "op_hl3_r4_rwyblind.yaml"], refit


def _kept(name: str) -> list[str]:
    cfg = load_config(name)
    return [c for s in resolve(cfg.features.include, cfg.features.exclude) for c in s.outputs
            if c not in cfg.features.drop_columns]


def test_h1_deliverable_and_its_sensitivities():
    """findings 10.35: the deliverable, superseded by F1 (findings 10.37) and kept on record, is R4 without the
    calendar's day of year; every sensitivity keeps its blocks and drops only what it names (planned delay and plan
    backlog; the TAF; the own runway), so a difference is that set's."""
    r4, base = _blocks("op_h1_r4.yaml"), set(_kept("op_h1_r4_nodoy.yaml"))
    assert set(load_config("op_h1_r4_nodoy.yaml").features.drop_columns) == {"takeoff_epoch", "op_h1_day_of_year"}
    for name in (*H1_DELIVERABLE, *H1_SENSITIVITIES):
        assert _blocks(name) == r4, name
        assert not [c for c in _kept(name) if "day_of_year" in c], name
    assert set(_kept("op_h1_r4_nodoy_lr04.yaml")) == base and set(_kept("op_h1_r4_nodoy_fwd.yaml")) == base
    assert set(_kept("op_h1_r4_nodoy_mae_exact.yaml")) == base
    assert base - set(_kept("op_h1_r4_nodoy_noplan.yaml")) == {
        "op_h1_plan_minus_sched_min", "op_h1_backlog_apt", "op_h1_backlog_rwy", "op_h1_backlog_pier"}
    taf = {c for c in base if c.startswith("op_h1_taf_")}
    assert len(taf) == 13 and base - set(_kept("op_h1_r4_nodoy_notaf.yaml")) == taf
    own_runway = r"^runway$|apt_runway|stand_runway|geo_rwy|geo_route|geo_gc|headwind|crosswind|_rwy_b|_rwy_s$|rwy_own|nm_rwy|backlog_rwy"
    blind = set(_kept("op_h1_r4_nodoy_rwyblind.yaml"))
    assert not [c for c in blind if re.search(own_runway, c)]
    assert blind == set(_kept("op_h1_r4_rwyblind.yaml")) - {"op_h1_day_of_year"}
    fwd = load_config("op_h1_r4_nodoy_fwd.yaml").validation
    assert fwd.scheme == "forward_holdout" and list(fwd.holdout_months) == ["2025-11", "2025-12"]
    for name in H1_SENSITIVITIES:
        assert not load_config(name).model.refit_on_all, name


def _flat(name: str) -> dict:
    """A loaded config's `dataclasses.asdict` as dotted keys, every nested dict flattened down to its leaves (an empty
    dict stays a leaf, so a key that only one of two configs sets still shows as a difference)."""
    out: dict = {}

    def walk(prefix: str, value) -> None:
        if isinstance(value, dict) and value:
            for key, sub in value.items():
                walk(f"{prefix}.{key}" if prefix else str(key), sub)
        else:
            out[prefix] = value

    walk("", dataclasses.asdict(load_config(name)))
    return out


def test_adopted_h1_deliverable_f1():
    """findings 10.37: F1 (op_h1_f1_plan), the H1 deliverable since 13 Oct, is op_h1_r4_nodoy plus the other flights'
    plans after the anchor (op_h1_fut_plan, tier O*); it resolves no tier F block. F1, its month pairs, its forward run
    and its rule M median member each equal their op_h1_r4_nodoy twin in everything but the block list, the name and the
    notes, so a difference between the two families is the plans'. F1 and its median member are refitted; the pairs and
    the forward run are holdout only. F1, its median member and the superseded deliverable's two refitted
    configurations keep the hashes their runs carry (`lab/op_rules.refits` finds the refits by them)."""
    h1, plan = "op_h1_r4_nodoy.yaml", "op_h1_fut_plan"
    tier_f = {"op_h1_fut_rec", "op_h1_fut_nm", "op_h1_fut_adsb"}
    twins = {"op_h1_f1_plan.yaml": h1,
             **{f"op_h1_f1_plan_{p}.yaml": f"op_h1_r4_nodoy_{p}.yaml" for p in PAIRS},
             "op_h1_f1_plan_fwd.yaml": "op_h1_r4_nodoy_fwd.yaml",
             "op_h1_f1_plan_mae_exact_cpu.yaml": "op_h1_r4_nodoy_mae_exact_cpu.yaml"}
    plan_cols = set(all_specs()[plan].outputs)
    assert plan not in _blocks(h1) and not plan_cols & set(_kept(h1))
    blocks, kept = _blocks(h1) | {plan}, set(_kept(h1)) | plan_cols
    named = ("features.include", "name", "notes")
    for name, twin in twins.items():
        assert name in OP_CONFIGS, name  # so test_op_config_allowlist covers it
        cfg = load_config(name)
        assert _blocks(name) == blocks and not _blocks(name) & tier_f, name
        assert set(_kept(name)) == kept, name
        assert cfg.seed == 42 and not cfg.evaluation.permutation.enabled, name
        assert cfg.name == name.removesuffix(".yaml"), name
        mine = {k: v for k, v in _flat(name).items() if k not in named}
        theirs = {k: v for k, v in _flat(twin).items() if k not in named}
        assert mine == theirs, (name, twin)
    refit = {name for name in twins if load_config(name).model.refit_on_all}
    assert refit == {"op_h1_f1_plan.yaml", "op_h1_f1_plan_mae_exact_cpu.yaml"}, refit
    for p, months in zip(PAIRS, (["2025-02", "2025-12"], ["2025-03", "2025-11"], ["2025-04", "2025-10"],
                                 ["2025-05", "2025-09"], ["2025-06", "2025-08"])):
        val = load_config(f"op_h1_f1_plan_{p}.yaml").validation
        assert val.scheme == "seasonal_holdout" and list(val.holdout_months) == months, p
    fwd = load_config("op_h1_f1_plan_fwd.yaml").validation
    assert fwd.scheme == "forward_holdout" and list(fwd.holdout_months) == ["2025-11", "2025-12"]
    rule_m = {"loss_function": "MAE", "eval_metric": "MAE", "leaf_estimation_method": "Exact", "boosting_type": "Plain",
              "iterations": 16000, "task_type": "CPU"}
    params = load_config("op_h1_f1_plan_mae_exact_cpu.yaml").model.params
    assert {k: params.get(k) for k in rule_m} == rule_m and "op_h1_f1_plan_mae_exact_cpu.yaml" in H1_MEDIAN
    # lab/op_deliver.py finds the refits by their configuration's current hash (`lab/op_rules.refits`): F1's three
    # (made with -o model.refit_on_all=true before the file set the flag) and its median member's, and the superseded
    # deliverable's, kept on record; an edit that moves a hash cuts the configuration off from its runs
    hashes = {"op_h1_f1_plan.yaml": "40ac5f36", "op_h1_f1_plan_mae_exact_cpu.yaml": "d7b90aac",
              "op_h1_r4_nodoy.yaml": "4e884d3b", "op_h1_r4_nodoy_mae_exact_cpu.yaml": "00a32933"}
    assert {name: load_config(name).hash() for name in hashes} == hashes


def _blocks(name: str) -> set[str]:
    cfg = load_config(name)
    return {s.name for s in resolve(cfg.features.include, cfg.features.exclude)}


def test_ladder_is_cumulative():
    order = ["op_h1_r0.yaml", "op_h1_r1.yaml", "op_h1_r2.yaml", "op_h1_r3.yaml", "op_h1_r4.yaml", "op_h1_r5.yaml"]
    sets = [{s.name for s in resolve(load_config(n).features.include)} for n in order]
    for small, big in zip(sets, sets[1:]):
        assert small < big
    assert sets[-1] - sets[-2] == {"op_h1_feed_lab"}
    h0 = {s.name for s in resolve(load_config("op_h0_r5.yaml").features.include)}
    assert {b.replace("op_h1_", "op_h0_") for b in sets[-1]} == h0
    # the ADS-B rungs (findings 10.31): R6 = R4 + op_h1_adsb; the public-data variant = R2 + op_h1_adsb; H0 r4 = H0 r5
    # without the feed, H0 r4 + ADS-B = R6 at the push; the placebo, the seed repeat and the pairs keep R6's blocks
    r4, r6 = _blocks("op_h1_r4.yaml"), _blocks("op_h1_r6.yaml")
    assert r6 - r4 == {"op_h1_adsb"} and r4 < r6
    assert _blocks("op_h1_r2_adsb.yaml") - _blocks("op_h1_r2.yaml") == {"op_h1_adsb"}
    h0_r4, h0_r4_adsb = _blocks("op_h0_r4.yaml"), _blocks("op_h0_r4_adsb.yaml")
    assert h0_r4 == h0 - {"op_h0_feed_lab"}
    assert h0_r4_adsb - h0_r4 == {"op_h0_adsb"}
    assert {b.replace("op_h1_", "op_h0_") for b in r6} == h0_r4_adsb
    for name in ("op_h1_r6_s43.yaml", "op_h1_r6_shift7.yaml", *(f"op_h1_r6_{p}.yaml" for p in PAIRS)):
        assert _blocks(name) == r6, name


def test_adsb_rungs_keep_the_coverage_columns_out_of_the_matrix():
    """The coverage columns gate the surface columns and serve as strata; as model inputs they would identify
    reception eras (critique F6), so every config that resolves an ADS-B block -- whatever its name, not only the
    listed rungs -- drops all of them and blanks the block on part of its fit (`model.block_dropout`, so "no feed" is
    learned everywhere); the listed rungs drop nothing else beyond takeoff_epoch."""
    from prc.features import _adsblib

    seen = set()
    for path in sorted(paths.CONFIGS.glob("*.yaml")):
        horizons = [h for h in ("h1", "h0") if f"op_{h}_adsb" in _blocks(path.name)]
        if not horizons:
            if path.name in ADSB_CONFIGS:  # op_h0_r4: the rung without ADS-B
                assert set(load_config(path.name).features.drop_columns) == {"takeoff_epoch"}, path.name
            continue
        seen.add(path.name)
        cfg = load_config(path.name)
        cov = {f"op_{h}_adsb_{c}" for h in horizons for c in _adsblib.COVERAGE}
        assert cov <= set(cfg.features.drop_columns), (path.name, sorted(cov - set(cfg.features.drop_columns)))
        assert {f"op_{h}_adsb" for h in horizons} <= set(cfg.model.block_dropout), path.name
        if path.name in ADSB_CONFIGS:
            assert set(cfg.features.drop_columns) == {"takeoff_epoch"} | cov, path.name
    assert set(ADSB_CONFIGS) - {"op_h0_r4.yaml"} <= seen
    shift = load_config("op_h1_r6_shift7.yaml").features.params
    assert shift == {"op_h1_adsb": {"shift_days": 7}}
    assert load_config("op_h1_r2_adsb.yaml").features.params == {"op_h1_adsb": {"coverage_ref": "scheduled"}}
    for p, months in zip(PAIRS, (["2025-02", "2025-12"], ["2025-03", "2025-11"], ["2025-04", "2025-10"],
                                 ["2025-05", "2025-09"], ["2025-06", "2025-08"])):
        cfg = load_config(f"op_h1_r6_{p}.yaml")
        assert list(cfg.validation.holdout_months) == months and not cfg.model.refit_on_all
        assert list(load_config(f"op_h1_r4_{p}.yaml").validation.holdout_months) == months
    assert load_config("op_h1_r6_s43.yaml").seed == 43


def test_runway_blind_rung_drops_every_own_runway_column():
    cfg = load_config("op_h1_r4_rwyblind.yaml")
    specs = resolve(cfg.features.include, cfg.features.exclude)
    kept = [c for s in specs for c in s.outputs if c not in cfg.features.drop_columns]
    own_runway = r"^runway$|apt_runway|stand_runway|geo_rwy|geo_route|geo_gc|headwind|crosswind|_rwy_b|_rwy_s$|rwy_own|nm_rwy|backlog_rwy"
    leaks = [c for c in kept if re.search(own_runway, c)]
    assert not leaks, leaks
    assert "op_h1_rwy_distinct_b60" in kept and "op_h1_rwy_config_age_min" in kept  # airport-level runway use stays


# --- the lab quarantine --------------------------------------------------------------------------

FEATURES_DIR = Path(operational.__file__).parent


def test_target_file_is_read_only_by_the_offblock_loader():
    for path in FEATURES_DIR.glob("*.py"):
        src = path.read_text(encoding="utf-8")
        if path.name == "_oplib.py":
            continue
        assert "TARGET_FILE" not in src and "load_target" not in src and "target.parquet" not in src, path.name
    src = (FEATURES_DIR / "_oplib.py").read_text(encoding="utf-8")
    assert "load_target" not in src and "target.parquet" not in src
    # the module attribute may be monkeypatched by another test's fixture: read the function from the source
    loader = re.search(r"\ndef load_offblocks\(\)[\s\S]*?\n(?=\S)", src).group(0)
    lines = [ln for ln in src.splitlines() if "TARGET_FILE" in ln]
    in_loader = [ln for ln in loader.splitlines() if "TARGET_FILE" in ln]
    assert len(lines) == len(in_loader) + 1, lines  # the loader, plus the LAB_EXTERNAL declaration
    assert any(ln.startswith("LAB_EXTERNAL = (paths.TARGET_FILE,)") for ln in lines)
    assert "columns=[OFFBLOCK_COLUMN]" in loader and _oplib.OFFBLOCK_COLUMN == "BLOCK_TIME_UTC_mvt"
    assert src.count("read_parquet") == 1


def test_only_lab_blocks_declare_the_target_file():
    for spec in all_specs().values():
        if paths.TARGET_FILE in spec.external:
            assert spec.name.startswith("op_h0_") or spec.name.endswith("_lab"), spec.name
        if spec.name.startswith("op_h0_") or spec.name.endswith("_lab"):
            assert paths.TARGET_FILE in spec.external, spec.name


def test_no_own_only_columns_are_read():
    for path in (FEATURES_DIR / "_oplib.py", FEATURES_DIR / "operational.py", FEATURES_DIR / "_adsblib.py",
                 FEATURES_DIR / "operational_adsb.py"):
        src = path.read_text(encoding="utf-8")
        code = re.sub(r'"""[\s\S]*?"""', "", src)
        assert "EOBT_1_flt" not in code and "ARVT_3_flt" not in code, path.name
        if "adsb" in path.name:  # the ADS-B modules never read NM's own-flight actuals at all, docstrings included
            assert "AOBT_3_flt" not in src and "EOBT_1_flt" not in src and "ARVT_3_flt" not in src, path.name


def test_nm_baseline_is_the_fit_rows_month_out_median():
    """`NmBaseline.anomaly(M)` against brute force: medians over fit rows outside month M, with the fall-backs."""
    mvt, _ = synthetic_world(days=16)
    dep = _oplib.departures(mvt)
    rng = np.random.default_rng(21)
    fit = dep["split"].eq("train").to_numpy() & (rng.random(len(dep)) < 0.8)
    bl = _oplib.NmBaseline(dep, fit, min_support=5)
    proxy = pd.Series(bl.proxy, index=dep.index)
    month = bl.month
    assert len(bl.fit_months) == 2  # March and April
    s = lambda col: dep[col].astype("string").fillna("__NA__")  # noqa: E731
    keys = (s("apt") + "|" + s("STAND_mvt") + "|" + s("RUNWAY_mvt"), s("apt") + "|" + s("STAND_mvt"),
            s("apt") + "|" + s("RUNWAY_mvt"), s("apt"))
    for excl in (None, *sorted(bl.fit_months)):
        use = fit & proxy.notna().to_numpy() & (np.ones(len(dep), bool) if excl is None else month != excl)
        got = bl.anomaly(excl)
        for i in rng.choice(len(dep), 60, replace=False):
            if not np.isfinite(proxy.iloc[i]):
                assert np.isnan(got[i])
                continue
            base = np.nan
            for k in keys:
                vals = proxy[use & (k == k.iloc[i]).to_numpy(dtype=bool)]
                if len(vals) >= 5:
                    base = vals.median()
                    break
            want = np.round(np.clip(proxy.iloc[i] - base, -900, 1800))
            assert (np.isnan(got[i]) and np.isnan(want)) or got[i] == want, (excl, i)
    # a query reads the table without its own planned month; a month without fit rows reads the full fit
    first = min(bl.fit_months)
    assert bl.key(float(first)) == first and bl.key(12.0 * 2026) is None and bl.key(np.nan) is None


def test_nm_blocks_depend_on_the_fit_mask_and_only_on_fit_rows(world):
    """op_h1_nm / op_h0_nm are fold-dependent for a reason: the proxy baseline is built from fit rows only."""
    mvt, _, _ = world
    dep_idx = mvt.index[(mvt["is_dep"] & mvt["split"].eq("train")).to_numpy()]
    fit_a = pd.Series(mvt["split"].eq("train").to_numpy() & mvt["is_dep"].to_numpy(), index=mvt.index)
    fit_b = fit_a.copy()
    fit_b.loc[fit_b.index[fit_b.to_numpy()][::2]] = False
    for name in ("op_h1_nm", "op_h0_nm"):
        assert not build(mvt, (name,), dep_idx, fit_a).equals(build(mvt, (name,), dep_idx, fit_b)), name
    # moving AOBT_3 of rows outside the fit leaves every fit row's anomaly (any month left out) bit-identical
    dep = _oplib.departures(mvt)
    fit = fit_b.reindex(dep.index).to_numpy()
    outside = dep.index[~fit]
    moved = _oplib.departures(_shift(mvt, "AOBT_3_flt", outside, -400))
    before, after = _oplib.NmBaseline(dep, fit), _oplib.NmBaseline(moved, fit)
    for excl in (None, *before.fit_months):
        np.testing.assert_array_equal(before.anomaly(excl)[fit], after.anomaly(excl)[fit])
    assert not np.array_equal(before.proxy[~fit], after.proxy[~fit], equal_nan=True)  # the shift did something


def test_target_using_op_blocks_are_declared():
    for name in H1_BLOCKS + H0_BLOCKS:
        spec = all_specs()[name]
        src = inspect.getsource(spec.fn)
        assert ("ctx.target" in src or "ctx.fit_mask" in src) == spec.uses_target, name


# --- the real data (opt-in) -------------------------------------------------------------------------

REAL = os.environ.get("OPLAD_REAL") == "1"
FULL = os.environ.get("OPLAD_FULL") == "1"


def _real_slice():
    """The first 50k movement rows, and their departures' recorded off-blocks.

    The target file is sorted by ``mvt_id``: it is streamed in 50k-row batches of two columns and only the
    slice's ids are kept, so at most one batch is in memory and every slice departure gets its off-block.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(paths.MOVEMENTS_FILE)
    mvt = next(pf.iter_batches(batch_size=50_000)).to_pandas()
    if "mvt_id" in mvt.columns:
        mvt = mvt.set_index("mvt_id")
    dep_ids = mvt.index[mvt["is_dep"].to_numpy(dtype=bool)]
    wanted, hi = set(dep_ids.tolist()), int(dep_ids.max())
    parts = []
    for batch in pq.ParquetFile(paths.TARGET_FILE).iter_batches(batch_size=50_000, columns=["BLOCK_TIME_UTC_mvt", "mvt_id"]):
        tgt = batch.to_pandas()
        tgt = tgt.set_index("mvt_id") if "mvt_id" in tgt.columns else tgt
        parts.append(tgt.loc[tgt.index.isin(wanted)])
        if int(tgt.index.min()) > hi:
            break
    tgt = pd.concat(parts)
    ob = pd.Series(_oplib.secs(tgt["BLOCK_TIME_UTC_mvt"]), index=tgt.index).reindex(dep_ids).dropna()
    return mvt, ob


def _day_keys(frame: pd.DataFrame) -> np.ndarray:
    """Planned UTC day instead of month: the slice holds a single month, so month-out would leave no fit rows."""
    return np.floor(_oplib.planned_offblock(frame) / 86_400.0)


@pytest.mark.skipif(not (REAL and paths.MOVEMENTS_FILE.exists()), reason="set OPLAD_REAL=1 (reads a 50k-row slice)")
def test_real_slice_t1_t1b_t2(monkeypatch):
    """T1 (crossing case), T1b and T2 on the first 50k real movement rows, every op block.

    The NM baseline is keyed by planned day here (`_day_keys`), so its leave-out logic is exercised on one
    month of data; the T1 rows share one day for the same reason as `t1_rows`. ``OPLAD_REAL_T2_ROWS`` (default
    20) sets how many rows T2 censors one at a time.
    """
    mvt, ob = _real_slice()
    dep = mvt.loc[mvt["is_dep"].to_numpy(dtype=bool)]
    train = dep.index[dep["split"].astype("string").eq("train").to_numpy(dtype=bool)]
    assert ob.index.isin(train).all() and len(ob) >= 0.9 * len(train), (len(ob), len(train))
    monkeypatch.setattr(_oplib, "load_offblocks", lambda: ob)
    monkeypatch.setattr(_oplib, "baseline_month", _day_keys)
    blocks = H1_BLOCKS + H0_BLOCKS
    rows = t1_rows(mvt, hours=26)  # one per airport: config age looks back 15 h, so no other query may sit in it
    assert len(rows) >= 8
    fit = _t1_frames(mvt, rows)
    base = build(mvt, blocks, rows, fit)
    assert base.filter(like="nm_apt_b60_anom_mean").notna().any().all()  # the NM columns are exercised
    moved = _set(mvt, "MVT_TIME_UTC_mvt", rows, _oplib.anchor_h1(mvt.loc[rows]) - 60)
    assert_identical(base, build(moved, blocks, rows, fit))
    moved = _shift(_shift(mvt, "AOBT_3_flt", rows, 600), "ARVT_3_flt", rows, 600)
    assert_identical(base, build(moved, blocks, rows, fit))

    # T2, one query at a time, the archives frozen; the blocks that read other flights' events
    _freeze_archives(monkeypatch, mvt, fit)
    event_blocks = ("op_h1_traffic", "op_h1_nm", "op_h1_feed_lab", "op_h0_traffic", "op_h0_nm", "op_h0_feed_lab")
    rng = np.random.default_rng(13)
    n_rows = int(os.environ.get("OPLAD_REAL_T2_ROWS", "20"))
    checked = 0
    for r in rng.choice(ob.index.intersection(train), n_rows, replace=False):
        q = mvt.loc[[r]]
        idx = pd.Index([r], name=mvt.index.name)
        for horizon in ("h1", "h0"):
            a = (_oplib.anchor_h1(q) if horizon == "h1" else _oplib.anchor_h0(q, ob))[0]
            if not np.isfinite(a):
                continue
            names = [b for b in event_blocks if b.startswith(f"op_{horizon}_")]
            full = build(mvt, names, idx, fit)
            cm, cob = censor(mvt, ob, a)
            monkeypatch.setattr(_oplib, "load_offblocks", lambda cob=cob: cob)
            cut = build(cm, names, idx, fit)
            monkeypatch.setattr(_oplib, "load_offblocks", lambda: ob)
            assert_identical(full, cut)
            checked += 1
    assert checked >= n_rows


@pytest.mark.skipif(not (FULL and paths.MOVEMENTS_FILE.exists()), reason="set OPLAD_FULL=1 (reads columns of the full frame)")
def test_anchor_sanity_on_2025():
    """Recorded off-block before the H1 anchor: below 1 % of matched 2025 departures overall, below 3 % per airport."""
    cols = ["apt", "split", "is_dep", "FLIGHT_ID_mvt", "IOBT_flt", "LOBT_flt", "SCHED_TIME_UTC_mvt"]
    mvt = pd.read_parquet(paths.MOVEMENTS_FILE, columns=cols)
    dep = mvt.loc[mvt["is_dep"] & mvt["split"].eq("train") & mvt["FLIGHT_ID_mvt"].notna()]
    ob = pd.read_parquet(paths.TARGET_FILE, columns=["BLOCK_TIME_UTC_mvt"])
    a = pd.Series(_oplib.anchor_h1(dep), index=dep.index)
    b = pd.Series(_oplib.secs(ob["BLOCK_TIME_UTC_mvt"]), index=ob.index).reindex(dep.index)
    early = (b < a)[b.notna()]
    assert early.mean() < 0.01
    per = early.groupby(dep.loc[early.index, "apt"].astype(str)).mean()
    assert (per < 0.03).all(), per.to_dict()
    np.testing.assert_array_equal(_oplib.anchor_h0(dep, _oplib.load_offblocks()), b.to_numpy())


# --- the read-out script's scoring (synthetic runs) ----------------------------------------------------

def _report_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("op_ladder_report", paths.ROOT / "lab" / "op_ladder_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_report_metrics_and_populations(tmp_path):
    rep = _report_module()
    m = rep.metric_row([100, 200, 300, 86_500], [130, 80, 300, 900])
    assert m["n"] == 4 and m["mae"] == pytest.approx((30 + 120 + 0 + 85_600) / 4)
    assert m["within_60"] == 0.5 and m["within_120"] == 0.75 and m["bias"] == pytest.approx((30 - 120 + 0 - 85_600) / 4)

    rng = np.random.default_rng(1)
    n = 4000
    ids = pd.Index(np.arange(n), name="mvt_id")
    y = pd.Series(rng.gamma(4, 250, n), index=ids)
    y.iloc[:5] = 86_400 + 600  # wrong-date rows
    ctx = pd.DataFrame({"y_true": y, "apt": rng.choice(["EDDF", "LSZH"], n), "month": rng.choice(["2025-01", "2025-07"], n),
                        "nm_matched": rng.random(n) > 0.02, "dayoffset": y > 80_000, "disrupted": rng.random(n) < 0.05,
                        "h1_early": rng.random(n) < 0.01}, index=ids)
    masks = rep.population_masks(ctx)
    assert set(masks) == set(rep.POPULATIONS)
    assert not masks["BULK"].iloc[:5].any() and not masks["EX_WD"].iloc[:5].any()
    assert (masks["H1_SENS"] <= masks["OP"]).all() and (masks["OP"] <= masks["BULK"]).all()
    assert 0.997 < masks["TRIM"].mean() < 0.999

    # three fake runs: a good op rung, a "leaky" one better than C1, and C1
    roots = tmp_path / "runs"
    preds = {"op_h1_r3": y + rng.normal(0, 200, n), "op_h1_r4": y + rng.normal(0, 5, n), "C1": y + rng.normal(0, 100, n)}
    names = {"op_h1_r3": "2026-10-08T01-00-00__op_h1_r3__aaaa__s42", "op_h1_r4": "2026-10-08T02-00-00__op_h1_r4__bbbb__s42",
             "C1": "2026-10-06T18-17-11__lean2_global_window_long__6d1b7355__s42"}
    for key, p in preds.items():
        d = roots / names[key]
        d.mkdir(parents=True)
        pd.DataFrame({"fold": "holdout", "y_true": y, "y_pred": p}, index=ids).to_parquet(d / "oof_predictions.parquet")
    older = roots / "2026-10-07T01-00-00__op_h1_r3__cccc__s42"
    older.mkdir()
    pd.DataFrame({"y_true": y, "y_pred": y * 0}, index=ids).to_parquet(older / "oof_predictions.parquet")
    assert rep.find_run([roots], "op_h1_r3", by_name=True).name == names["op_h1_r3"]  # the newest
    assert rep.find_run([roots], "op_h1_r", by_name=True) is None  # exact config names only
    rows = []
    for key in preds:
        rows += rep.score(key, rep.read_oof(roots / names[key]), ctx)
    table = pd.DataFrame(rows)
    trip = "\n".join(rep.tripwire(table))
    assert "op_h1_r3: " in trip and "ok (below C1)" in trip.split("op_h1_r3")[1].split("\n")[0]
    assert "LEAK" in trip.split("op_h1_r4")[1].split("\n")[0]
    meta = {k: {"title": k, "run": names[k], "tiers": "O"} for k in preds}
    md = rep.markdown(table, meta, rep.tripwire(table))
    assert "## Ladder" in md and "OP MAE (s) per airport" in md and "Swatowska" in md

    # the config hash filter: a smoke run (other hash) is skipped even when it is the newest
    smoke = roots / "2026-10-09T01-00-00__op_h1_r3__dddd__s42"
    smoke.mkdir()
    pd.DataFrame({"y_true": y, "y_pred": y * 0}, index=ids).to_parquet(smoke / "oof_predictions.parquet")
    assert rep.find_run([roots], "op_h1_r3", by_name=True).name == smoke.name
    assert rep.find_run([roots], "op_h1_r3", by_name=True, expected_hash="aaaa").name == names["op_h1_r3"]
    assert rep.find_run([roots], "op_h1_r3", by_name=True, expected_hash="ffff") is None
    assert rep.config_hash("op_h1_r4") == load_config("op_h1_r4.yaml").hash()


def test_report_h0_winter_ordering_steps_and_row_sets(tmp_path):
    rep = _report_module()
    rng = np.random.default_rng(2)
    n = 6000
    ids = pd.Index(np.arange(n), name="mvt_id")
    y = pd.Series(rng.gamma(4, 250, n), index=ids)
    days = pd.Series(pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 40, n), unit="D"), index=ids)
    ctx = pd.DataFrame({"y_true": y, "apt": rng.choice(["EDDF", "LSZH"], n), "month": days.dt.strftime("%Y-%m"),
                        "day": days.dt.strftime("%Y-%m-%d"), "nm_matched": True, "dayoffset": False,
                        "disrupted": rng.random(n) < 0.05, "h1_early": rng.random(n) < 0.01}, index=ids)
    oof = lambda noise: pd.DataFrame({"y_true": y, "y_pred": y + rng.normal(0, noise, n)}, index=ids)  # noqa: E731
    oofs = {"op_h1_r3": oof(220), "op_h1_r4": oof(150), "op_h0_r5": oof(180), "C1": oof(100),
            "op_h1_r2_winter": oof(250).iloc[:3000]}
    oofs["op_h1_r5"] = oofs["op_h1_r4"].iloc[:-10]  # a rung that scored other rows
    rows = []
    for key, o in oofs.items():
        rows += rep.score(key, o, ctx)
    table = pd.DataFrame(rows)
    # H0 is scored on OP and H1_SENS only
    assert set(table.loc[table["row"] == "op_h0_r5", "population"]) == {"OP", "H1_SENS"}
    trip = "\n".join(rep.tripwire(table))
    line = [ln for ln in trip.splitlines() if ln.startswith("- op_h1_r4 vs op_h0_r5")][0]
    assert "LEAK" in line  # an H1 rung better than everything at pushback
    warnings = rep.row_set_warnings(oofs)
    assert len(warnings) == 1 and warnings[0].startswith("op_h1_r5")  # the winter rung is not compared with C1's rows
    st = rep.paired_step(oofs["op_h1_r3"], oofs["op_h1_r4"], ctx, n_boot=300)
    assert st["n"] == int((y < 3600).sum()) and st["d_mae"] < 0 and st["mae_lo"] <= st["d_mae"] <= st["mae_hi"] < 0
    assert st["w120_hi"] >= st["d_w120"] >= st["w120_lo"] > 0
    meta = {k: {"title": k, "run": k, "tiers": "O", "best_iteration": "123"} for k in oofs}
    md = rep.markdown(table, meta, trip.splitlines(), [{"from": "op_h1_r3", "to": "op_h1_r4", **st}], warnings)
    ladder = md.split("## Ladder")[1].split("\n## ")[0]
    winter = md.split("## Other holdout")[1].split("\n## ")[0]
    assert "op_h1_r2_winter" in winter and "op_h1_r2_winter" not in ladder
    h0_line = [ln for ln in ladder.splitlines() if ln.startswith("| op_h0_r5 |")][0].split("|")
    assert h0_line[-5].strip() == "-" and h0_line[-3].strip() == "-"  # no BULK / ALL RMSE for H0
    assert "| 123 |" in ladder and "Rung steps" in md and "Warnings" in md
    every = md.split("## Every population")[1].split("\n## ")[0]
    assert not [ln for ln in every.splitlines() if ln.startswith("| op_h0_r5 | ALL")]
    # best iterations read from a run's log
    d = tmp_path / "run"
    d.mkdir()
    (d / "log.txt").write_text("x\n    seasonal: best iteration 16503\n", encoding="utf-8")
    assert rep.best_iterations(d) == "16503" and rep.best_iterations(tmp_path) is None


def _clock_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("op_clock_checks", paths.ROOT / "lab" / "op_clock_checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_clock_checks_runway_and_modal_sections_on_the_synthetic_world():
    clk = _clock_module()
    mvt, offblocks = synthetic_world()
    dep = mvt.loc[mvt["is_dep"].to_numpy()].copy()
    block = offblocks.reindex(dep.index)
    dep["BLOCK_TIME_UTC_mvt"] = _ts(block.to_numpy()).to_numpy()
    dep["TAXITIME_SEC_mvt"] = (_oplib.secs(dep["MVT_TIME_UTC_mvt"]) - block.to_numpy())
    rk = clk.runway_knowability(dep)
    all_row = rk.loc["ALL"]
    assert 0 < all_row["coverage"] <= 1
    assert all_row["own_eq_dominant_cov"] >= all_row["own_eq_dominant_60min"] - 1e-12
    assert all_row["own_eq_dominant_60min"] <= all_row["coverage"] + 1e-12
    assert clk.aobt3_modal_share(dep, min_rows=10**6).empty  # no runway qualifies: empty, no KeyError
    share = clk.aobt3_modal_share(dep, min_rows=100)
    assert not share.empty and share.notna().any().any()


def test_output_names_are_unique_across_every_block():
    """test_features.py asserts this too, but only where the competition data exists."""
    seen: dict[str, str] = {}
    for spec in all_specs().values():
        for col in spec.outputs:
            assert col not in seen, f"'{col}' declared by both {seen[col]} and {spec.name}"
            seen[col] = spec.name
    op_prefixes = ("op_h1_", "op_h0_", "op_hl_", "op_hl3_", "op_hl24_", "op_arch_")
    assert all(c.startswith(op_prefixes) for s in all_specs().values() if s.name.startswith("op_") for c in s.outputs)
    # each family's columns carry its own prefix (an op_hl3_ block cannot emit an op_h1_ column, and so on)
    for spec in all_specs().values():
        for family in ("h1", "h0", "hl", "hl3", "hl24"):
            if spec.name.startswith(f"op_{family}_"):
                assert all(c.startswith(f"op_{family}_") for c in spec.outputs), spec.name
        if spec.name == "op_adsb_archive":
            assert all(c.startswith("op_arch_") for c in spec.outputs)
