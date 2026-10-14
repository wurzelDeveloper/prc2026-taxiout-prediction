"""The ADS-B surface blocks of the operational ladder (`prc.features.operational_adsb`, `_adsblib`): leakage and contract.

``op_h1_adsb`` (H1, board/final) and ``op_h0_adsb`` (H0, lab) must never see the flight's own movement, nor any event
after the query time ``q = floor(a) - 30 s``. Asserted on a synthetic world (tests/adsb_synth.py over
`test_operational.synthetic_world`; odd stands carry negative ids, as most real stands):

* contract, non-trivial values on a covered airport, every column family NaN exactly where its gate fails, the
  primitives against brute force (own exclusion by callsign and origin stand on every window edge);
* T1 / T1b / T1c -- the own recorded takeoff, own NM actuals, own off-block (H1) moved: bit-identical;
* T1d -- the own airframe's departure points deleted, shifted by +-600 s, or its push moved to just before the anchor:
  bit-identical at H1 and H0 (the event table rebuilt from the changed points) on spaced rows whose taxi family is
  gated in, by the stand rule; and for departures first seen mid-taxi with no stand origin, by the callsign rule
  alone; guards: without the rules (or without the callsign rule) the crossing case changes ``taxi_out_now`` and
  ``push_15``;
* T2 -- for one query at a time, the movement records censored after ``q`` AND the event table rebuilt from the points
  at or before ``q``: bit-identical at H1 and H0, on random anchors, on anchors placed on event boundaries (an event
  known at ``q`` or ``q + 1``; an end known then), and with the shifted placebo; and, without a rebuild, the event table
  censored at ``q`` on many boundary anchors; the edge itself: an event known at ``q`` counts, one at ``q + 1`` not;
* no data is never zero; the taxi, hold and stand gates are separate (taxiway reception without pushes keeps the taxi
  columns, NaNs the stand ones); the 30- and 10-minute outage gates; the activity heartbeat (a feed lost at T: the gate
  holds at T + 1 min, is shut from T + 5 min); the training support (no support or a ratio outside the airport's range,
  widened by the margin: NaN); the support table's hold relabelling and the own-runway agreement; an unknown own
  runway: NaN, not 0;
* the shifted placebo reads the past (an independent expectation; a negative shift raises); the public-data
  coverage reads no movement record; the production loader agrees with the table it loads;
* the quarantine of the lab block, hashing, no import-time work.

``OPLAD_REAL=1`` adds T1 (crossing case), T1d (the linked own airframe's points deleted; the tier-T link is read only
inside the test, from ``OPLAD_LINKS_DIR``, default cache/adsb) and T2 with a rebuild on real ADS-B points of a few days
(``OPLAD_ADSB_DAYS``, default 2025-10-14..2025-10-16, when every airport but LTFM is received).
"""

from __future__ import annotations

import inspect
import os
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import adsb_synth as S
from prc import paths
from prc.adsb import surface
from prc.config import load_config
from prc.features import _adsblib, _oplib, all_specs, operational_adsb, resolve
from prc.features.build import compute_block
from prc.features.registry import FeatureContext
from test_operational import _set, _shift, assert_identical, censor, synthetic_world, t1_rows

BLOCKS = ("op_h1_adsb", "op_h0_adsb")
LAG = _adsblib.ADSB_LAG_S


SPACING_H = 30.0
"""T1d rows of one airport lie this far apart: no row's own airframe can reach another row's windows (6 h at most)."""


def _make_world(seed: int = 0, **realism) -> SimpleNamespace:
    """A synthetic ADS-B world: movements, points, events, the support table and, as ``t1d``, the covered spaced
    training departures the identity tests compare (`_covered_spaced`); ``X``: both blocks on every training
    departure."""
    mvt, ob = synthetic_world(days=8, per_day=44, seed=seed)
    mvt = S.with_callsigns(mvt)
    pts, info = S.world(mvt, ob, **realism)
    geo = S.geometry()
    af = surface.airborne_first_times(pts)
    ev = surface.episodes(pts, geo, float(pts["t"].max()), af)
    w = SimpleNamespace(mvt=mvt, ob=ob, pts=pts, info=info, geo=geo, af=af, ev=ev, table=S.table(ev))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_adsblib, "load_stand_tables", S.stand_tables)
        mp.setattr(_adsblib, "load_runway_keys", S.runway_keys)
        w.support = S.support_of(mvt, w.table)
        _patch(mp, w)
        dep = mvt.loc[mvt["is_dep"].to_numpy(dtype=bool) & mvt["split"].eq("train").to_numpy()]
        w.X = build(mvt, BLOCKS, dep.index)
    w.t1d = _covered_spaced(mvt, info, w.X)
    return w


def _covered_spaced(mvt: pd.DataFrame, info: pd.DataFrame, X: pd.DataFrame, hours: float = SPACING_H) -> pd.Index:
    """Training departures at least ``hours`` apart per airport, with a stand, a runway and a callsign (the identity
    rules' inputs) and an airframe of the world, whose taxi family is gated in at H1 and at H0: a T1d comparison of NaN
    would prove nothing."""
    dep = mvt.loc[mvt["is_dep"].to_numpy(dtype=bool) & mvt["split"].eq("train").to_numpy()]
    ok = (dep["STAND_mvt"].notna() & dep["RUNWAY_mvt"].notna() & dep["CALLSIGN_flt"].notna()).to_numpy()
    ok = ok & dep.index.isin(info.index)
    for h in ("h1", "h0"):
        ok = ok & X[f"op_{h}_adsb_cov_taxi_ok"].reindex(dep.index).eq(1).to_numpy()
    picked = []
    for _, g in dep.loc[ok].groupby("apt", observed=True):
        last = -np.inf
        for r, t in zip(g.index, _oplib.secs(g["MVT_TIME_UTC_mvt"])):
            if t - last >= hours * 3600:
                picked.append(r)
                last = t
    return pd.Index(picked, name="mvt_id")


@pytest.fixture(scope="module")
def adsb_world():
    return _make_world()


def _patch(monkeypatch, w: SimpleNamespace) -> dict:
    state = {"events": w.table, "offblocks": w.ob, "support": w.support}
    monkeypatch.setattr(_adsblib, "load_events", lambda t_min=None, t_max=None: state["events"])
    monkeypatch.setattr(_adsblib, "load_stand_tables", S.stand_tables)
    monkeypatch.setattr(_adsblib, "load_runway_keys", S.runway_keys)
    monkeypatch.setattr(_adsblib, "load_support", lambda: state["support"])
    monkeypatch.setattr(_oplib, "load_offblocks", lambda: state["offblocks"])
    return state


@pytest.fixture
def world(adsb_world, monkeypatch):
    return adsb_world, _patch(monkeypatch, adsb_world)


def build(mvt: pd.DataFrame, names, idx: pd.Index, params: dict | None = None) -> pd.DataFrame:
    specs = all_specs()
    parts = [compute_block(specs[n], mvt, FeatureContext(dep_index=idx, params=dict(params or {}))) for n in names]
    return pd.concat(parts, axis=1)


def rebuild(pts: pd.DataFrame, geo, cutoff: float | None = None) -> pd.DataFrame:
    cut = float(pts["t"].max()) if cutoff is None else cutoff
    return S.table(surface.episodes(pts, geo, cut, surface.airborne_first_times(pts)))


def dep_index(mvt: pd.DataFrame) -> pd.Index:
    return mvt.index[mvt["is_dep"].to_numpy(dtype=bool)]


def view_at(table: pd.DataFrame, q: float) -> pd.DataFrame:
    """The event table as it stood at ``q``: events known by then, ends known later blanked (with what needs them)."""
    v = table.loc[table["t_known_start"].to_numpy() <= q].copy()
    late = ~(v["t_known_end"].to_numpy() <= q)
    v.loc[late, ["t_end", "t_known_end"]] = np.nan
    v.loc[late & v["kind"].astype(str).ne("TAKEOFF").to_numpy(), ["dur", "stop_s", "path_m"]] = np.nan
    return v.reset_index(drop=True)


def anchored(mvt: pd.DataFrame, ob: pd.Series, row, a: float, horizon: str) -> tuple[pd.DataFrame, pd.Series]:
    """The frames with ``row``'s anchor at ``a``: H1 through its IOBT (H1 = IOBT - 15 min), H0 through its off-block."""
    if horizon == "h1":
        return _set(mvt, "IOBT_flt", pd.Index([row]), np.array([a + _oplib.H1_LEAD_S])), ob
    ob2 = ob.copy()
    ob2.loc[row] = a
    return mvt, ob2


def anchor_of(mvt: pd.DataFrame, ob: pd.Series, row, horizon: str) -> float:
    q = mvt.loc[[row]]
    return float((_oplib.anchor_h1(q) if horizon == "h1" else _oplib.anchor_h0(q, ob))[0])


def boundary_anchors(w: SimpleNamespace, n_rows: int, seed: int, shift: int = 0) -> list[tuple]:
    """(row, a) pairs whose query time sits on an event boundary at the row's airport: ``q`` = an event's knowable start
    or end, or one second before it (so an off-by-one read is visible), the event another airframe's. With ``shift``
    (the placebo's days), the boundary of an event that many days before ``q``."""
    rng = np.random.default_rng(seed)
    dep = w.mvt.loc[w.mvt["is_dep"].to_numpy(dtype=bool) & w.mvt["split"].eq("train").to_numpy()]
    rows = rng.choice(dep.index.intersection(w.info.index), n_rows, replace=False)
    t0 = float(w.pts["t"].min()) + 9 * 3600
    t_end = float(w.mvt.loc[w.mvt["split"].eq("train").to_numpy(), "MVT_TIME_UTC_mvt"].max().timestamp()) - 86_400 * shift
    kinds = ["TAXI_OUT", "PUSH", "RWY_HOLD", "REMOTE_STOP", "TAKEOFF", "LANDING", "TAXI_IN"]
    out = []
    for k, r in enumerate(rows):
        apt, own = str(w.mvt.loc[r, "apt"]), w.info.loc[r, "icao"]
        e = w.ev.loc[w.ev["apt"].eq(apt) & w.ev["icao"].ne(own) & w.ev["kind"].isin(kinds)]
        col = "t_known_end" if k % 2 else "t_known_start"
        e = e.loc[e[col].between(t0, t_end - 3600)]
        if not len(e):
            continue
        t = float(e[col].iloc[int(rng.integers(0, len(e)))]) + 86_400 * shift
        out.append((r, t + LAG - (k // 2) % 2))   # q = t (counted) or q = t - 1 (not yet)
    return out


# --- the contract -------------------------------------------------------------------------------------------

def test_blocks_build_and_keep_their_contract(world):
    w, _ = world
    idx = dep_index(w.mvt)
    X = build(w.mvt, BLOCKS, idx)
    assert X.index.equals(idx)
    for name in BLOCKS:
        spec = all_specs()[name]
        assert list(X.columns[X.columns.str.startswith(name + "_")]) == list(spec.outputs)
        assert all(c.startswith(name + "_") for c in spec.outputs) and len(spec.outputs) == len(_adsblib.COLUMNS)
    rank = w.mvt.loc[idx, "split"].eq("rank").to_numpy()
    assert X.loc[rank, [c for c in X.columns if c.startswith("op_h0_")]].isna().all().all(), "H0 is NaN for ranking rows"
    assert X.loc[rank, "op_h1_adsb_cov_ok"].notna().all() and (X.loc[rank, "op_h1_adsb_cov_ok"] == 1).mean() > 0.5
    assert X.loc[rank, "op_h1_adsb_taxi_out_now"].notna().mean() > 0.5, "H1 exists for ranking rows"


def test_a_covered_airport_is_not_just_nan_and_zeros_and_each_family_follows_its_gate(world):
    w, _ = world
    X = build(w.mvt, ("op_h1_adsb",), dep_index(w.mvt))
    gate = {c: f"op_h1_adsb_cov_{fam}_ok" for fam, cols in _adsblib.FAMILIES.items() for c in cols}
    for col in ("taxi_out_now", "rwy_hold_now", "taxi_dur_mean_60", "push_15", "taxi_in_now", "remote_60",
                "taxi_stop_share_60", "rwy_own_hold_now", "taxi_out_elapsed_mean_now", "taxi_out_start_15"):
        on = X[gate[col]].to_numpy() == 1
        v = X.loc[on, f"op_h1_adsb_{col}"].to_numpy(dtype="float64")
        assert on.mean() > 0.5 and np.isfinite(v).mean() > 0.3 and np.nanstd(v) > 0, col
    # every family is NaN exactly where its own gate fails, and its counts are finite where it holds
    for fam, gate, probe in (("taxi", "cov_taxi_ok", "taxi_out_now"), ("hold", "cov_hold_ok", "rwy_hold_now"),
                             ("stand", "cov_stand_ok", "push_15")):
        ok = X[f"op_h1_adsb_{gate}"].to_numpy()
        cols = [f"op_h1_adsb_{c}" for c in _adsblib.FAMILIES[fam]]
        assert X.loc[ok == 0, cols].isna().all().all(), fam
        assert np.isfinite(X.loc[ok == 1, f"op_h1_adsb_{probe}"]).all() and (ok == 1).mean() > 0.5, fam
    assert (X["op_h1_adsb_cov_taxi_ok"] <= X["op_h1_adsb_cov_ok"]).all()
    assert (X["op_h1_adsb_cov_hold_ok"] <= X["op_h1_adsb_cov_taxi_ok"]).all()
    # a world received from stand to runway: nearly every takeoff's taxi was seen
    assert X["op_h1_adsb_cov_taxi_share_120"].median() > 0.8


# --- primitives against brute force ---------------------------------------------------------------------------

def test_own_excluded_window_stats_match_brute_force():
    rng = np.random.default_rng(31)
    n, m, n_st = 500, 160, 9
    ev_t = np.sort(rng.integers(0, 20_000, n)).astype("float64")
    ev_g = rng.integers(0, 3, n)
    ev_cs = rng.integers(-1, 8, n)
    ev_os = rng.integers(-1, n_st, n)
    ev_v = rng.integers(0, 900, n).astype("float64")
    q_t = rng.integers(0, 20_000, m).astype("float64")
    j = rng.integers(0, n, m)
    q_t[:30] = ev_t[j[:30]]                     # an event exactly at q
    q_t[30:60] = ev_t[j[30:60]] + 900            # ... exactly at q - w
    q_t[60:80] = ev_t[j[60:80]] + 901            # ... one second outside
    q_t[80:100] = ev_t[j[80:100]] - 1            # ... one second after q
    q_g = rng.integers(-1, 3, m)
    q_g[:100] = ev_g[j[:100]]
    q_cs = np.where(rng.random(m) < 0.8, rng.integers(0, 8, m), -1)
    q_cs[:50] = ev_cs[j[:50]]                    # the event at the edge is the query's own (callsign)...
    nb_rows, nb_stand = [], []
    nb = {}
    for i in range(m):
        k = set(rng.choice(n_st, rng.integers(0, 4), replace=False).tolist())
        if i < 100 and rng.random() < 0.6 and ev_os[j[i]] >= 0:
            k.add(int(ev_os[j[i]]))              # ... or near its own stand
        nb[i] = k
        for s in sorted(k):
            nb_rows.append(i)
            nb_stand.append(s)
    q = _adsblib.Queries(q_t, q_g, q_cs, nb_rows, nb_stand, 3, 8, n_st)
    got = _adsblib.others(q, ev_t, ev_g, ev_cs, ev_os, [900, None], {"v": ev_v})
    got_all = _adsblib.others(q, ev_t, ev_g, ev_cs, ev_os, [900], exclude=False)
    for i in range(m):
        for back in (900, None):
            sel = [e for e in range(n) if q_g[i] >= 0 and ev_g[e] == q_g[i]
                   and ev_t[e] <= q_t[i] and (back is None or ev_t[e] >= q_t[i] - back)
                   and not ((q_cs[i] >= 0 and ev_cs[e] == q_cs[i]) or (ev_os[e] >= 0 and ev_os[e] in nb[i]))]
            assert got[(back, "n")][i] == len(sel), (i, back)
            assert got[(back, "v")][i] == int(ev_v[sel].sum()), (i, back)
        plain = sum(1 for e in range(n) if q_g[i] >= 0 and ev_g[e] == q_g[i] and q_t[i] - 900 <= ev_t[e] <= q_t[i])
        assert got_all[(900, "n")][i] == plain
    assert (got[(900, "n")] < got_all[(900, "n")]).sum() >= 20, "the exclusion has no teeth"


def test_in_progress_semantics_match_brute_force():
    """t_known_start <= q < min(t_known_end, t_start + cap): an end not yet known at q counts as not ended."""
    rng = np.random.default_rng(32)
    n, m = 300, 120
    ts = np.sort(rng.integers(0, 15_000, n)).astype("float64")
    tk = ts + rng.integers(0, 60, n)
    te = tk + rng.integers(1, 5000, n)
    te[rng.random(n) < 0.2] = np.nan
    ev = pd.DataFrame({"t_start": ts, "t_known_start": tk, "t_known_end": te})
    stop = _adsblib._in_progress_stop(ev, 1800.0)
    g = np.zeros(n, dtype="int64")
    q_t = rng.integers(0, 17_000, m).astype("float64")
    q_t[:20] = tk[rng.integers(0, n, 20)]
    q_t[20:40] = np.nan_to_num(te[rng.integers(0, n, 20)], nan=5.0)
    q = _adsblib.Queries(q_t, np.zeros(m, dtype="int64"), np.full(m, -1), [], [], 1, 1, 1)
    started = _adsblib.others(q, tk, g, np.full(n, -1), np.full(n, -1), [None])[(None, "n")]
    ended = _adsblib.others(q, stop, g, np.full(n, -1), np.full(n, -1), [None])[(None, "n")]
    for i in range(m):
        want = sum(1 for e in range(n) if tk[e] <= q_t[i] < min(np.inf if np.isnan(te[e]) else te[e], ts[e] + 1800))
        assert started[i] - ended[i] == want, i


def test_the_callsign_normalisation_is_the_builders():
    """One implementation: the builder's event callsigns and the blocks' own-callsign rule cannot drift apart."""
    assert surface.norm_callsign is _adsblib.norm_callsign
    got = _adsblib.norm_callsign(["  klm 1821 ", "KLM-1821", "klm1821", "", None, np.nan, "dlh7aj"])
    assert got.tolist() == ["KLM1821", "KLM1821", "KLM1821", None, None, None, "DLH7AJ"]


# --- T1: own events in the movement records ----------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["plus600", "minus300", "cross_h1", "cross_bucket"])
def test_t1_own_takeoff_invariance(world, mode):
    w, _ = world
    rows = t1_rows(w.mvt)
    assert len(rows) >= 6
    base = build(w.mvt, BLOCKS, rows)
    if mode == "plus600":
        moved = _shift(w.mvt, "MVT_TIME_UTC_mvt", rows, 600)
    elif mode == "minus300":
        moved = _shift(w.mvt, "MVT_TIME_UTC_mvt", rows, -300)
    elif mode == "cross_h1":
        moved = _set(w.mvt, "MVT_TIME_UTC_mvt", rows, _oplib.anchor_h1(w.mvt.loc[rows]) - 60)
    else:
        a = _oplib.anchor_h1(w.mvt.loc[rows])
        moved = _set(w.mvt, "MVT_TIME_UTC_mvt", rows, (np.floor(a / 900) - 2) * 900 + 17)
    assert_identical(base, build(moved, BLOCKS, rows))


@pytest.mark.parametrize("shift", [600, -600, "before_a"])
def test_t1b_own_nm_actuals_invariance(world, shift):
    w, _ = world
    rows = t1_rows(w.mvt)
    base = build(w.mvt, BLOCKS, rows)
    if shift == "before_a":
        a = _oplib.anchor_h1(w.mvt.loc[rows])
        moved = _set(_set(w.mvt, "AOBT_3_flt", rows, a - 300), "ARVT_3_flt", rows, a - 200)
    else:
        moved = _shift(_shift(w.mvt, "AOBT_3_flt", rows, shift), "ARVT_3_flt", rows, shift)
    assert_identical(base, build(moved, BLOCKS, rows))


def test_t1c_own_offblock_invariance_at_h1(world):
    w, state = world
    rows = t1_rows(w.mvt)
    base = build(w.mvt, ("op_h1_adsb",), rows)
    h0_before = build(w.mvt, ("op_h0_adsb",), rows)
    moved = w.ob.copy()
    moved.loc[rows] = _oplib.anchor_h1(w.mvt.loc[rows]) - 120
    state["offblocks"] = moved
    assert_identical(base, build(w.mvt, ("op_h1_adsb",), rows))
    assert not build(w.mvt, ("op_h0_adsb",), rows).equals(h0_before), "at H0 the anchor itself moves"


# --- T1d: the own airframe's points -------------------------------------------------------------------------------

def _own_departures(pts: pd.DataFrame, info: pd.DataFrame, rows: pd.Index) -> np.ndarray:
    """The points of the rows' own departures (not their inbound legs)."""
    mask = np.zeros(len(pts), dtype=bool)
    for r in rows:
        i = info.loc[r]
        mask |= (pts["icao"].eq(i["icao"]) & (pts["t"] >= i["b"] - i["dwell"] - 1)).to_numpy()
    return mask


def _regenerate(pts, info, rows, pushes: dict) -> pd.DataFrame:
    """The rows' departures re-flown with their push at ``pushes[row]`` (same stand, runway, lift-off, callsigns, and
    first sighting: a departure first seen mid-taxi stays so)."""
    keep = pts.loc[~_own_departures(pts, info, rows)]
    tr = S.Tracks()
    for r in rows:
        i = info.loc[r]
        n0 = len(tr.rows)
        # a callsign set before the first point stays so (else the moved departure would start under its inbound one)
        t_switch = float(pushes[r]) - float(i["dwell"]) - 1200.0 if bool(i.get("switch_before", False)) else i["t_switch"]
        S.departure(tr, i["apt"], i["icao"], int(i["stand_i"]), i["rwy"], float(pushes[r]), float(i["t_lift"]),
                    S.switch(i["cs_in"], i["cs_out"], t_switch), dwell=float(i["dwell"]), deice_s=float(i["deice"]))
        if bool(i.get("first_seen", False)):
            S.first_seen_cut(tr, n0, int(i["stand_i"]))
    return pd.concat([keep, tr.frame()], ignore_index=True).sort_values(["icao", "t"], kind="stable").reset_index(drop=True)


@pytest.mark.parametrize("mode", ["delete", "plus600", "minus600", "push_60_before_h1", "push_300_before_h1",
                                  "push_60_before_h0"])
def test_t1d_own_airframe_points_invariance(world, mode):
    w, state = world
    rows = w.t1d
    assert len(rows) >= 6
    base = build(w.mvt, BLOCKS, rows)
    assert (base["op_h1_adsb_cov_taxi_ok"] == 1).all() and (base["op_h0_adsb_cov_taxi_ok"] == 1).all()
    assert (base["op_h1_adsb_cov_stand_ok"] == 1).mean() >= 0.5 and (base["op_h1_adsb_cov_hold_ok"] == 1).mean() >= 0.5
    own = _own_departures(w.pts, w.info, rows)
    if mode == "delete":
        changed = w.pts.loc[~own]
    elif mode in ("plus600", "minus600"):
        changed = w.pts.copy()
        changed.loc[own, "t"] = changed.loc[own, "t"] + (600 if mode == "plus600" else -600)
    else:
        lead = 60 if "60" in mode else 300
        a = _oplib.anchor_h1(w.mvt.loc[rows]) if mode.endswith("h1") else _oplib.anchor_h0(w.mvt.loc[rows], w.ob)
        changed = _regenerate(w.pts, w.info, rows, dict(zip(rows, a - lead)))
    state["events"] = rebuild(changed, w.geo)
    assert not state["events"].equals(w.table)
    assert_identical(base, build(w.mvt, BLOCKS, rows))


def test_t1d_guard_without_the_identity_rules_the_crossing_case_moves_the_counts(world, monkeypatch):
    w, state = world
    rows = w.t1d
    monkeypatch.setattr(_adsblib, "norm_callsign", lambda values: np.full(len(values), None, dtype=object))
    monkeypatch.setattr(_adsblib, "neighbour_pairs", lambda *a, **k: (np.zeros(0, dtype="int64"), np.zeros(0, dtype="int64")))
    base = build(w.mvt, ("op_h1_adsb",), rows)
    a = _oplib.anchor_h1(w.mvt.loc[rows])
    state["events"] = rebuild(_regenerate(w.pts, w.info, rows, dict(zip(rows, a - 300))), w.geo)
    moved = build(w.mvt, ("op_h1_adsb",), rows)
    for col in ("op_h1_adsb_push_15", "op_h1_adsb_taxi_out_now"):
        ok = base[col].notna() & moved[col].notna()
        # (a row whose own push already preceded the anchor keeps its count: it was in the window before)
        assert (moved.loc[ok, col] >= base.loc[ok, col]).all() and (moved.loc[ok, col] > base.loc[ok, col]).sum() >= 3, col


@pytest.fixture(scope="module")
def cs_world(adsb_world):
    """The T1d departures first seen moving 250 m from their stand (no stand stop, no push: no origin stand within
    150 m) with the outbound callsign set before their first point: only the callsign rule can identify them (critique
    F1/F4: 29-99 % of real departures are never seen at their stand); the rest of the world realistic: 30 % first seen
    mid-taxi, ground points without a track or a ground speed."""
    rows = adsb_world.t1d
    w = _make_world(seed=0, switch_before=rows, first_seen_rows=rows, first_seen=0.3, no_track=0.15, no_gs=0.01)
    first = w.info.index[w.info["first_seen"].to_numpy(dtype=bool)]
    covered = [r for r in rows if all(w.X.loc[r, f"op_{h}_adsb_cov_taxi_ok"] == 1 for h in ("h1", "h0"))]
    w.rows = pd.Index(covered, name="mvt_id").intersection(first)
    # the own stand unknown too (unmatched STAND_mvt): no stand rule at all for these rows
    mv = w.mvt.copy()
    mv.loc[w.rows, "STAND_mvt"] = np.nan
    w.mvt_nostand = mv
    return w


@pytest.mark.parametrize("mode", ["delete", "push_600_before_h1", "push_600_before_h0"])
def test_t1d_the_callsign_rule_alone_identifies_a_departure_first_seen_mid_taxi(cs_world, monkeypatch, mode):
    w = cs_world
    state = _patch(monkeypatch, w)
    rows = w.rows
    assert len(rows) >= 6, "too few first-seen test departures"
    own_ev = w.ev.loc[w.ev["icao"].isin(set(w.info.loc[rows, "icao"])) & w.ev["kind"].isin(["TAXI_OUT", "TAKEOFF", "PUSH"])]
    assert not own_ev["kind"].eq("PUSH").any() and own_ev["origin_stand"].isna().all()
    assert own_ev["origin_kind"].dropna().eq("first_seen").all() and own_ev["origin_kind"].eq("first_seen").sum() >= len(rows)
    base = build(w.mvt_nostand, BLOCKS, rows)
    if mode == "delete":
        changed = w.pts.loc[~_own_departures(w.pts, w.info, rows)]
    else:  # the push 10 min before the anchor: first seen (250 m out) before q, so the own taxi is in progress at q
        a = _oplib.anchor_h1(w.mvt.loc[rows]) if mode.endswith("h1") else _oplib.anchor_h0(w.mvt.loc[rows], w.ob)
        changed = _regenerate(w.pts, w.info, rows, dict(zip(rows, a - 600)))
    state["events"] = rebuild(changed, w.geo)
    assert_identical(base, build(w.mvt_nostand, BLOCKS, rows))
    if mode == "push_600_before_h1":  # guard: the callsign rule alone switched off, the own taxi enters the counts
        monkeypatch.setattr(_adsblib, "norm_callsign", lambda values: np.full(len(values), None, dtype=object))
        state["events"] = w.table
        before = build(w.mvt_nostand, ("op_h1_adsb",), rows)
        state["events"] = rebuild(changed, w.geo)
        after = build(w.mvt_nostand, ("op_h1_adsb",), rows)
        col = "op_h1_adsb_taxi_out_now"
        ok = before[col].notna() & after[col].notna()
        assert (after.loc[ok, col] > before.loc[ok, col]).sum() >= 1, "the callsign rule has no teeth here"


# --- T2: future deletion, the table rebuilt from points up to q --------------------------------------------------

def _t2_one(w, state, r, a, horizon: str, params: dict | None = None, mvt=None, ob=None):
    """One query: the block on the full frames vs on the records censored after q and the table rebuilt from the
    points at or before q (the anchor's airport). Returns the full row."""
    mvt = w.mvt if mvt is None else mvt
    ob = w.ob if ob is None else ob
    name = f"op_{horizon}_adsb"
    q = np.floor(a) - LAG
    shift = int((params or {}).get("shift_days", 0)) * 86_400
    idx = pd.Index([r], name="mvt_id")
    state["events"], state["offblocks"] = w.table, ob
    full = build(mvt, (name,), idx, params)
    cm, _ = censor(mvt, ob, q)
    # other flights' off-blocks after q censored too; the own one is the H0 anchor itself
    state["offblocks"] = ob.where(~(ob > q) | (ob.index == r))
    apt = str(mvt.loc[r, "apt"])
    t_pts = w.pts["t"].to_numpy()
    window = w.pts.loc[(np.floor(t_pts) <= q) & (t_pts >= q - shift - 14 * 3600) & w.pts["apt"].eq(apt).to_numpy()]
    cut_ev = surface.episodes(window, w.geo, q, w.af[w.af <= q])
    assert (cut_ev["t_known_start"] <= q).all() and (cut_ev["t_known_end"].dropna() <= q).all()
    state["events"] = S.table(cut_ev)
    cut = build(cm, (name,), idx, params)
    state["events"], state["offblocks"] = w.table, w.ob
    assert_identical(full, cut)
    return full


@pytest.mark.parametrize("horizon,shift", [("h1", 0), ("h0", 0), ("h1", 1)])
def test_t2_censored_records_and_a_table_rebuilt_at_q_change_nothing(world, horizon, shift):
    w, state = world
    rng = np.random.default_rng(17 + shift)
    cand = w.mvt.index[(w.mvt["is_dep"] & w.mvt["split"].eq("train")).to_numpy()]
    if shift:  # the placebo needs a day of events before
        cand = cand[(w.mvt.loc[cand, "MVT_TIME_UTC_mvt"] >= w.mvt["MVT_TIME_UTC_mvt"].min() + pd.Timedelta(days=2)).to_numpy()]
    rows = pd.Index(rng.choice(cand, 12, replace=False), name="mvt_id")
    params = {"shift_days": shift} if shift else None
    covered = 0
    for r in rows:
        a = anchor_of(w.mvt, w.ob, r, horizon)
        if not np.isfinite(a):
            continue
        full = _t2_one(w, state, r, a, horizon, params)
        covered += int(full[f"op_{horizon}_adsb_cov_taxi_ok"].iloc[0] == 1)
    assert covered >= 6, "most T2 rows must be covered (else the comparison is of NaN)"


@pytest.mark.parametrize("horizon", ["h1", "h0"])
def test_t2_on_event_boundaries_with_a_rebuild(world, horizon):
    """Anchors placed so that ``q`` is an event's knowable start or end, or one second before it."""
    w, state = world
    picked = boundary_anchors(w, 8, seed=41 if horizon == "h1" else 43)
    assert len(picked) >= 6
    for r, a in picked:
        mvt, ob = anchored(w.mvt, w.ob, r, a, horizon)
        assert anchor_of(mvt, ob, r, horizon) == a
        _t2_one(w, state, r, a, horizon, mvt=mvt, ob=ob)


@pytest.mark.parametrize("horizon,shift", [("h1", 0), ("h0", 0), ("h1", 1)])
def test_the_event_table_censored_at_q_changes_nothing_on_boundary_anchors(world, horizon, shift):
    """No rebuild: the full table against the table as it stood at q (events known later dropped, ends known later
    blanked), on anchors that put q exactly on, or one second before, the boundaries of many events."""
    w, state = world
    name = f"op_{horizon}_adsb"
    params = {"shift_days": shift} if shift else None
    picked = boundary_anchors(w, 60, seed=7 + shift + (0 if horizon == "h1" else 100), shift=shift)
    assert len(picked) >= 40
    finite = 0
    for r, a in picked:
        mvt, ob = anchored(w.mvt, w.ob, r, a, horizon)
        idx = pd.Index([r], name="mvt_id")
        state["offblocks"] = ob
        state["events"] = w.table
        full = build(mvt, (name,), idx, params)
        state["events"] = view_at(w.table, np.floor(a) - LAG - 86_400 * shift)
        assert_identical(full, build(mvt, (name,), idx, params))
        finite += int(np.isfinite(full[f"{name}_taxi_out_now"].iloc[0]))
    state["events"], state["offblocks"] = w.table, w.ob
    assert finite >= 0.5 * len(picked)


def _inject(table: pd.DataFrame, row: dict) -> pd.DataFrame:
    extra = pd.DataFrame([{**{c: np.nan for c in table.columns}, **row}])
    out = pd.concat([table.astype({c: object for c in ("apt", "kind", "icao", "cs", "origin_kind", "rwy_end")}), extra],
                    ignore_index=True)
    return S.table(out)


def _covered_row(w, state, horizon="h1", params=None):
    """A training departure whose three families are gated in, and its query time."""
    dep = w.mvt.loc[w.mvt["is_dep"].to_numpy(dtype=bool) & w.mvt["split"].eq("train").to_numpy()]
    X = build(w.mvt, (f"op_{horizon}_adsb",), dep.index, params)
    ok = (X[f"op_{horizon}_adsb_cov_taxi_ok"] == 1) & (X[f"op_{horizon}_adsb_cov_stand_ok"] == 1) & (X[f"op_{horizon}_adsb_cov_hold_ok"] == 1)
    cand = ok.index[ok.to_numpy()]
    r = cand[len(cand) // 3]
    return r, np.floor(anchor_of(w.mvt, w.ob, r, horizon)) - LAG


def _push(apt: str, t_known: float, cs: str = "FOREIGN9", stand: float = 987_654.0) -> dict:
    return {"eid": 1, "apt": apt, "kind": "PUSH", "icao": "fff999", "cs": cs, "t_start": t_known - 15.0,
            "t_known_start": t_known, "origin_kind": "push", "origin_stand": stand, "taxi_seen": 0, "parent": -1}


def test_an_event_known_at_q_counts_and_one_at_q_plus_one_does_not(world):
    """The feed latency is pinned: q = floor(a) - 30 s exactly."""
    w, state = world
    r, q = _covered_row(w, state)
    apt = str(w.mvt.loc[r, "apt"])
    idx = pd.Index([r], name="mvt_id")
    base = build(w.mvt, ("op_h1_adsb",), idx)["op_h1_adsb_push_15"].iloc[0]
    for t, plus in ((q, 1), (q + 1, 0), (q - 900, 1), (q - 901, 0)):
        state["events"] = _inject(w.table, _push(apt, t))
        assert build(w.mvt, ("op_h1_adsb",), idx)["op_h1_adsb_push_15"].iloc[0] == base + plus, (t - q, plus)
    state["events"] = w.table


# --- coverage: no data, families, outages, support, runway -----------------------------------------------------

def test_no_reception_is_nan_never_zero(world):
    w, state = world
    state["events"] = rebuild(w.pts.loc[w.pts["apt"].ne("LSZH")], w.geo)  # LSZH: no ADS-B at all (as LTFM)
    idx = dep_index(w.mvt)
    X = build(w.mvt, ("op_h1_adsb",), idx)
    lszh = w.mvt.loc[idx, "apt"].eq("LSZH").to_numpy()
    assert (X.loc[lszh, "op_h1_adsb_cov_tko_n_120"] == 0).all() and (X.loc[lszh, "op_h1_adsb_cov_ok"] == 0).all()
    surface_cols = [f"op_h1_adsb_{c}" for c in _adsblib.SURFACE + _adsblib.STAND]
    assert X.loc[lszh, surface_cols].isna().all().all()
    assert X.loc[~lszh, "op_h1_adsb_taxi_out_now"].notna().mean() > 0.5
    # an outage of one day at EDDF: once the 120-min window lies inside it, nothing reads as quiet traffic
    d0 = np.floor(w.pts["t"].min() / 86_400) * 86_400 + 2 * 86_400
    gap = w.pts.loc[~(w.pts["apt"].eq("EDDF") & (w.pts["t"] >= d0) & (w.pts["t"] < d0 + 86_400))]
    state["events"] = rebuild(gap, w.geo)
    X = build(w.mvt, ("op_h1_adsb",), idx)
    q = _oplib.anchor_h1(w.mvt.loc[idx]) - LAG
    inside = w.mvt.loc[idx, "apt"].eq("EDDF").to_numpy() & (q >= d0 + 3 * 3600) & (q < d0 + 86_400)
    assert inside.sum() >= 10
    assert (X.loc[inside, "op_h1_adsb_cov_ok"] == 0).all() and X.loc[inside, surface_cols].isna().all().all()


def test_taxiway_reception_without_pushes_keeps_the_taxi_family_and_nans_the_stand_family(world):
    """The stand gate is its own: no point within 60 m of a stand at LSZH (no push, no stand stop) leaves the taxi and
    hold columns, and NaNs the push and push-origin columns (critique F5)."""
    w, state = world
    st, _ = S.stand_tables()
    lszh = w.pts["apt"].eq("LSZH").to_numpy()
    x, y = w.geo["LSZH"].xy(w.pts["lat"].to_numpy(), w.pts["lon"].to_numpy())
    d = w.geo["LSZH"].nearest_stand(x, y, 60.0)[0]
    state["events"] = rebuild(w.pts.loc[~(lszh & (d <= 60.0))], w.geo)
    assert not state["events"].loc[state["events"]["apt"].eq("LSZH"), "kind"].eq("PUSH").any()
    idx = dep_index(w.mvt)
    X = build(w.mvt, ("op_h1_adsb",), idx)
    at = w.mvt.loc[idx, "apt"].eq("LSZH").to_numpy()
    taxi_ok = X.loc[at, "op_h1_adsb_cov_taxi_ok"] == 1
    assert taxi_ok.mean() > 0.5 and (X.loc[at, "op_h1_adsb_cov_stand_ok"] == 0).all()
    assert X.loc[at, [f"op_h1_adsb_{c}" for c in _adsblib.STAND]].isna().all().all()
    assert X.loc[at & (X["op_h1_adsb_cov_taxi_ok"] == 1).to_numpy(), "op_h1_adsb_taxi_out_now"].notna().all()
    assert (X.loc[~at, "op_h1_adsb_cov_stand_ok"] == 1).mean() > 0.5  # EDDF keeps its pushes


@pytest.mark.parametrize("window", [1800, 600])
def test_an_outage_in_the_last_30_or_10_minutes_closes_the_gate(world, window):
    """Reception lost at EDDF in the ``window`` before q of a few rows: the 120-min ratio still passes, the outage gate
    of that window shuts every family (and only it: with the gate off the rows read as covered)."""
    w, state = world
    dep = w.mvt.loc[w.mvt["is_dep"].to_numpy(dtype=bool) & w.mvt["split"].eq("train").to_numpy() & w.mvt["apt"].eq("EDDF").to_numpy()]
    q_all = pd.Series(_oplib.anchor_h1(dep) - LAG, index=dep.index)
    refs = _adsblib.reference_counts(w.mvt, dep.index, q_all.to_numpy(), "recorded", (window,))[window]
    t_lo = float(w.pts["t"].min()) + 10 * 3600
    cand = q_all.index[(refs >= 2) & (q_all.to_numpy() >= t_lo)]
    rows, last = [], -np.inf
    for r in cand.sort_values():
        if q_all[r] - last > 4 * 3600:
            rows.append(r)
            last = q_all[r]
    rows = pd.Index(rows[:6], name="mvt_id")
    assert len(rows) >= 3
    t = w.pts["t"].to_numpy()
    gap = np.zeros(len(w.pts), dtype=bool)
    for r in rows:
        gap |= w.pts["apt"].eq("EDDF").to_numpy() & (t > q_all[r] - window) & (t <= q_all[r])
    state["events"] = rebuild(w.pts.loc[~gap], w.geo)
    quiet = {"act_min": 10**9}  # the heartbeat off: this test is the reference ratios' alone
    on = {"outage_min_ref": 2, "outage10_min_ref": 10**6} if window == 1800 else {"outage_min_ref": 10**6, "outage10_min_ref": 1}
    on, off = {**on, **quiet}, {"outage_min_ref": 10**6, "outage10_min_ref": 10**6, **quiet}
    X = build(w.mvt, ("op_h1_adsb",), rows, on)
    Y = build(w.mvt, ("op_h1_adsb",), rows, off)
    ratio = X[f"op_h1_adsb_cov_mov_ratio_{window // 60}"]
    assert (ratio < 0.25).all(), ratio
    assert (Y["op_h1_adsb_cov_ok"] == 1).sum() >= 2, "the 120-min ratio alone lets the outage rows through"
    assert (X["op_h1_adsb_cov_ok"] == 0).all()
    assert X[[f"op_h1_adsb_{c}" for c in _adsblib.SURFACE + _adsblib.STAND]].isna().all().all()


def test_the_training_support_nans_a_family_without_support_or_outside_the_range(world):
    w, state = world
    idx = dep_index(w.mvt)
    X = build(w.mvt, ("op_h1_adsb",), idx)
    at = w.mvt.loc[idx, "apt"].eq("LSZH").to_numpy()
    sup = w.support.copy()
    # LSZH's stand family without support (as EGLL's in 2025): NaN there, everything else unchanged
    state["support"] = sup.assign(share=np.where(sup["apt"].eq("LSZH") & sup["family"].eq("stand"), 0.05, sup["share"]))
    Y = build(w.mvt, ("op_h1_adsb",), idx)
    stand = [f"op_h1_adsb_{c}" for c in _adsblib.STAND]
    assert Y.loc[at, stand].isna().all().all() and (X.loc[at, "op_h1_adsb_cov_stand_ok"] == 1).mean() > 0.5
    keep = [c for c in X.columns if c not in stand and c != "op_h1_adsb_cov_stand_ok"]
    assert_identical(X[keep], Y[keep])
    assert_identical(X.loc[~at], Y.loc[~at])
    # a taxi range the rows' ratios leave: taxi and hold NaN at LSZH
    state["support"] = sup.assign(hi=np.where(sup["apt"].eq("LSZH") & sup["family"].eq("taxi"), 0.01, sup["hi"]))
    Z = build(w.mvt, ("op_h1_adsb",), idx)
    assert Z.loc[at, [f"op_h1_adsb_{c}" for c in _adsblib.TAXI + _adsblib.HOLD]].isna().all().all()
    # an airport missing from the table: nothing; support off: the gates alone
    state["support"] = sup.loc[sup["apt"].ne("LSZH")]
    assert build(w.mvt, ("op_h1_adsb",), idx).loc[at, [f"op_h1_adsb_{c}" for c in _adsblib.SURFACE + _adsblib.STAND]].isna().all().all()
    off = build(w.mvt, ("op_h1_adsb",), idx, {"support": False})
    assert (off["op_h1_adsb_cov_taxi_ok"] >= X["op_h1_adsb_cov_taxi_ok"]).all()
    state["support"] = w.support
    # the table itself: one row per airport and family, shares in [0, 1], ranges ordered; the holds' agreement
    fam = sup.loc[sup["family"].isin(["taxi", "hold", "stand"])]
    assert set(zip(fam["apt"], fam["family"])) == {(a, f) for a in S.APTS for f in ("taxi", "hold", "stand")} and len(fam) == 6
    assert fam["share"].between(0, 1).all() and (fam["lo"] <= fam["hi"]).all() and (fam["share"] > 0.5).all()
    own = sup.loc[sup["family"].eq("own_hold")].set_index("apt")
    assert set(own.index) == set(S.APTS) and (own["share"] > 0.95).all() and (own["n"] >= _adsblib.HOLD_MAP_MIN_N).all()
    assert not sup["family"].eq("hold_map").any()  # holds beside their own runway end: nothing to relabel


def test_the_support_range_is_widened_by_the_margin(world):
    """A row whose taxi ratio lies 10 % outside the airport's training range keeps the family under the default margin
    (1.25) and loses it without one: one noisy 2-hour ratio is no new reception regime."""
    w, state = world
    X = w.X
    r = X.index[(X["op_h1_adsb_cov_taxi_ok"] == 1).to_numpy() & (X["op_h1_adsb_cov_taxi_ratio_120"] > 0).to_numpy()][5]
    apt, r0 = str(w.mvt.loc[r, "apt"]), float(X.loc[r, "op_h1_adsb_cov_taxi_ratio_120"])
    idx = pd.Index([r], name="mvt_id")
    sup = w.support.copy()
    taxi = sup["apt"].eq(apt) & sup["family"].eq("taxi")
    for lo, hi in ((0.0, r0 / 1.1), (r0 * 1.1, 99.0)):
        state["support"] = sup.assign(lo=np.where(taxi, lo, sup["lo"]), hi=np.where(taxi, hi, sup["hi"]))
        assert build(w.mvt, ("op_h1_adsb",), idx)["op_h1_adsb_cov_taxi_ok"].iloc[0] == 1, (lo, hi)
        assert build(w.mvt, ("op_h1_adsb",), idx, {"support_margin": 1.0})["op_h1_adsb_cov_taxi_ok"].iloc[0] == 0, (lo, hi)
    state["support"] = w.support


def test_holds_are_relabelled_by_the_support_table_and_the_own_runway_needs_agreeing_holds(world):
    """`rwy_own_hold_now` counts the holds of the own runway end after the support table's relabelling (EDDF's holds
    beside 18 relabelled to 25C: a 25C row then counts both, an 18 row none), and is NaN at an airport whose holds
    agreed with the next takeoff too rarely, or too few times, on the training days; nothing else moves."""
    w, state = world
    dep = w.mvt.loc[w.mvt["is_dep"].to_numpy(dtype=bool) & w.mvt["split"].eq("train").to_numpy()]
    idx = dep.index
    eddf = dep["apt"].eq("EDDF").to_numpy()
    own = "op_h1_adsb_rwy_own_hold_now"
    base = build(w.mvt, ("op_h1_adsb",), idx)
    as_18 = w.mvt.copy()
    as_18.loc[idx[eddf], "RUNWAY_mvt"] = "18"
    base_18 = build(as_18, ("op_h1_adsb",), idx)[own]
    extra = pd.DataFrame([{"apt": "EDDF", "family": "hold_map", "n": 50, "share": 0.9, "lo": np.nan, "hi": np.nan,
                           "label": "18", "target": "25C"}])
    state["support"] = pd.concat([w.support, extra], ignore_index=True)
    new = build(w.mvt, ("op_h1_adsb",), idx)
    rwy = dep["RUNWAY_mvt"].astype(str).to_numpy()
    on = base[own].notna().to_numpy()
    r25, r18 = on & eddf & (rwy == "25C"), on & eddf & (rwy == "18")
    assert r25.sum() >= 10 and r18.sum() >= 10
    assert (new.loc[r25, own].to_numpy() == base.loc[r25, own].to_numpy() + base_18[r25].to_numpy()).all()
    assert (new.loc[r18, own] == 0).all()
    # teeth (holds in progress are rare in this world): a 25C row gained an 18 hold, an 18 row lost its own
    assert (new.loc[r25, own] > base.loc[r25, own]).sum() >= 1 and (base.loc[r18, own] > 0).sum() >= 1
    others = [c for c in base.columns if c != own]
    assert_identical(base[others], new[others])
    assert_identical(base.loc[~eddf], new.loc[~eddf])
    # the agreement: too low, or too few holds, at EDDF -> NaN there, the rest unchanged
    sup = w.support.copy()
    at = sup["apt"].eq("EDDF") & sup["family"].eq("own_hold")
    for change in ({"share": 0.5}, {"n": _adsblib.HOLD_MAP_MIN_N - 1}):
        state["support"] = sup.assign(**{k: np.where(at, v, sup[k]) for k, v in change.items()})
        low = build(w.mvt, ("op_h1_adsb",), idx)
        assert low.loc[eddf, own].isna().all() and low.loc[eddf, "op_h1_adsb_rwy_hold_now"].notna().mean() > 0.5, change
        assert_identical(base[others], low[others])
        assert_identical(base.loc[~eddf], low.loc[~eddf])
    state["support"] = w.support


def test_a_feed_loss_closes_the_gate_within_five_minutes(world):
    """The activity heartbeat (critique F5; review): the other aircraft's event starts in the last 5 minutes against
    their mean 5 minutes before. A dense stream of surface events (a busy hub: 15 per 5 min) that stops at T, a feed
    lost at every airport: the gate still holds one minute later and is shut from T + 5 min on, long before the 10- and
    30-minute reference ratios could notice; with the heartbeat off the same queries read as covered."""
    w, state = world
    X = w.X
    cand = X.index[(X["op_h1_adsb_cov_ok"] == 1).to_numpy() & (X["op_h1_adsb_cov_taxi_ok"] == 1).to_numpy()]
    off = {"act_min": 10**9}
    for r in cand[3:]:
        apt = str(w.mvt.loc[r, "apt"])
        T = float(np.floor(anchor_of(w.mvt, w.ob, r, "h1"))) - LAG
        stream = [{**_push(apt, T - k * 20.0, cs=f"HB{k:04d}", stand=np.nan), "icao": f"ee{k:04d}"} for k in range(360)]
        extra = pd.DataFrame([{**{c: np.nan for c in w.table.columns}, **row} for row in stream])
        table = S.table(pd.concat([w.table.astype({c: object for c in ("apt", "kind", "icao", "cs", "origin_kind", "rwy_end")}),
                                   extra], ignore_index=True))
        state["events"] = view_at(table, T)  # nothing known after T, anywhere
        got = {}
        for d in (60, 300, 600):
            mvt, ob = anchored(w.mvt, w.ob, r, T + d + LAG, "h1")
            idx = pd.Index([r], name="mvt_id")
            got[d] = (build(mvt, ("op_h1_adsb",), idx).iloc[0], build(mvt, ("op_h1_adsb",), idx, off).iloc[0])
        if not all(g[1]["op_h1_adsb_cov_ok"] == 1 for g in got.values()):
            continue  # this row's base gates do not hold at all three times: the next one
        break
    else:
        pytest.fail("no covered row for the heartbeat test")
    state["events"] = w.table
    assert got[60][0]["op_h1_adsb_cov_ok"] == 1 and got[60][0]["op_h1_adsb_cov_act_ratio_5"] > 0.5
    for d in (300, 600):
        on = got[d][0]
        assert on["op_h1_adsb_cov_ok"] == 0 and on["op_h1_adsb_cov_act_ratio_5"] < _adsblib.DEFAULTS["act_max_ratio"], d
        assert on[[f"op_h1_adsb_{c}" for c in _adsblib.SURFACE + _adsblib.STAND]].isna().all(), d
        assert got[d][1]["op_h1_adsb_cov_ok"] == 1, d  # the heartbeat alone shut it


def test_an_unknown_own_runway_is_nan_not_zero(world):
    w, _ = world
    rows = t1_rows(w.mvt)
    mv = w.mvt.copy()
    mv["RUNWAY_mvt"] = mv["RUNWAY_mvt"].cat.add_categories(["09"])
    mv.loc[rows[:3], "RUNWAY_mvt"] = "09"        # a well-formed ident that is no runway end of EDDF / LSZH
    X = build(mv, ("op_h1_adsb",), rows)
    hold_ok = X["op_h1_adsb_cov_hold_ok"] == 1
    assert X.loc[rows[:3], "op_h1_adsb_rwy_own_hold_now"].isna().all()
    assert X.loc[rows[3:][hold_ok.loc[rows[3:]].to_numpy()], "op_h1_adsb_rwy_own_hold_now"].notna().all()
    assert X.loc[rows[:3][hold_ok.loc[rows[:3]].to_numpy()], "op_h1_adsb_rwy_hold_now"].notna().all()


# --- placebo, public coverage, the production loader --------------------------------------------------------------

def test_the_shifted_placebo_reads_the_past_and_a_negative_shift_raises(world):
    w, state = world
    params = {"shift_days": 1}
    r, q = _covered_row(w, state, params=params)
    apt = str(w.mvt.loc[r, "apt"])
    idx = pd.Index([r], name="mvt_id")
    base = build(w.mvt, ("op_h1_adsb",), idx, params)["op_h1_adsb_push_15"].iloc[0]
    assert np.isfinite(base)
    # a push known a day before q - 60 s is read (shifted onto q - 60); one known a day AFTER q - 60 is not
    state["events"] = _inject(w.table, _push(apt, q - 86_400 - 60))
    assert build(w.mvt, ("op_h1_adsb",), idx, params)["op_h1_adsb_push_15"].iloc[0] == base + 1
    state["events"] = _inject(w.table, _push(apt, q + 86_400 - 60))
    assert build(w.mvt, ("op_h1_adsb",), idx, params)["op_h1_adsb_push_15"].iloc[0] == base
    state["events"] = w.table
    assert not build(w.mvt, ("op_h1_adsb",), idx).equals(build(w.mvt, ("op_h1_adsb",), idx, params))
    for bad in (-1, -7, 0.5):
        with pytest.raises(ValueError, match="shift_days"):
            build(w.mvt, ("op_h1_adsb",), idx, {"shift_days": bad})
    with pytest.raises(ValueError, match="shift_days"):
        _adsblib.shifted(w.table, -1)


def test_public_coverage_reads_no_movement_record_of_other_flights(world):
    """With ``coverage_ref: scheduled`` the block reads the schedule of other flights, never their recorded times."""
    w, _ = world
    rows = t1_rows(w.mvt)
    params = {"coverage_ref": "scheduled"}
    base = build(w.mvt, ("op_h1_adsb",), rows, params)
    others = w.mvt.index.difference(rows)
    moved = _shift(w.mvt, "MVT_TIME_UTC_mvt", others, 1234)
    assert_identical(base, build(moved, ("op_h1_adsb",), rows, params))
    assert not build(w.mvt, ("op_h1_adsb",), rows).equals(build(moved, ("op_h1_adsb",), rows)), "the recorded one does"
    with pytest.raises(ValueError):
        build(w.mvt, ("op_h1_adsb",), rows, {"coverage_ref": "nope"})
    with pytest.raises(ValueError):
        build(w.mvt, ("op_h1_adsb",), rows, {"no_such": 1})


@pytest.mark.parametrize("shift", [0, 1])
def test_the_production_loader_agrees_with_the_table_it_loads(adsb_world, tmp_path, monkeypatch, shift):
    """`_adsblib.load_events` on a snapshot consolidated by `surface.consolidate` (its pyarrow filter, the day-rounded
    window, the categoricals) gives the values of the full table, for queries around midnight too."""
    w = adsb_world
    monkeypatch.setattr(paths, "ADSB_SURFACE", tmp_path / "adsb_surface")
    out = paths.adsb_surface_dir("loader")
    (out / "days").mkdir(parents=True)
    day = pd.to_datetime(w.ev["t_known_start"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    for d, g in w.ev.groupby(day):
        g.to_parquet(out / "days" / f"{d}.parquet", index=False)
    days = sorted(day.unique())
    path = surface.consolidate("loader", {"day_set": days, "days": {d: {} for d in days}}, log=lambda *_: None)
    monkeypatch.setattr(paths, "ADSB_SURFACE_EVENTS_FILE", path)
    monkeypatch.setattr(_adsblib, "load_stand_tables", S.stand_tables)
    monkeypatch.setattr(_adsblib, "load_runway_keys", S.runway_keys)
    monkeypatch.setattr(_adsblib, "load_support", lambda: w.support)
    dep = w.mvt.loc[w.mvt["is_dep"].to_numpy(dtype=bool)]
    a = pd.Series(_oplib.anchor_h1(dep), index=dep.index).dropna()
    tod = np.mod(a - LAG, 86_400)
    near_midnight = a.index[(tod < 2 * 3600) | (tod > 22 * 3600)]
    rest = a.index[::9]
    params = {"shift_days": shift}
    for idx in (near_midnight, rest):
        assert len(idx) >= 5
        got = _adsblib.surface_at(w.mvt, idx, a[idx].to_numpy(), params)
        want = _adsblib.surface_at(w.mvt, idx, a[idx].to_numpy(), params, events=w.table)
        assert_identical(got, want)
        assert got["cov_ok"].notna().all() and np.isfinite(got["taxi_out_now"]).mean() > 0.3


# --- quarantine and hashing ---------------------------------------------------------------------------------------

def test_the_lab_block_is_quarantined():
    spec = all_specs()["op_h0_adsb"]
    assert spec.group == "op_lab_h0_adsb" and set(_oplib.LAB_EXTERNAL) <= set(spec.external)
    assert paths.ADSB_SURFACE_EVENTS_FILE in spec.external
    assert "op_h0_adsb" not in {s.name for s in resolve(["group:op_lab_h0"])}
    h1 = all_specs()["op_h1_adsb"]
    assert paths.TARGET_FILE not in h1.external
    for f in (paths.ADSB_SURFACE_EVENTS_FILE, paths.ADSB_SURFACE_SUPPORT_FILE, paths.GEO_STANDS_FILE,
              paths.GEO_STAND_MATCHES_FILE, paths.GEO_RUNWAY_ENDS_FILE):
        assert f in h1.external and f in spec.external, f
    for path in sorted(paths.CONFIGS.glob("*.yaml")):
        cfg = load_config(path.name)
        blocks = {s.name for s in resolve(cfg.features.include, cfg.features.exclude)}
        if "op_h0_adsb" in blocks:
            assert not cfg.model.refit_on_all and path.name.startswith("op_h0_"), path.name
        if "op_h1_adsb" in blocks:
            assert path.name.startswith("op_h1_"), path.name


def test_the_modules_read_nothing_own_and_hash_their_helpers():
    from prc.features.build import _related_modules

    for mod in (_adsblib, operational_adsb):
        src = inspect.getsource(mod)
        for word in ("TARGET_FILE", "EOBT_1_flt", "AOBT_3_flt", "ARVT_3_flt", "target.parquet", "load_target"):
            assert word not in src, (mod.__name__, word)
        body = src.split('"""', 2)[2]
        assert not [ln for ln in body.splitlines() if re.match(r"^[a-z_]+\(", ln)], "import-time work"
    names = {m.__name__ for m in _related_modules(operational_adsb)}
    assert {"prc.features._oplib", "prc.features._adsblib", "prc.features._geolib", "prc.features.congestion"} <= names
    for mod in (_adsblib, operational_adsb):
        assert not re.search(r"(from|import)\s+prc\.adsb", inspect.getsource(mod)), "feature code never imports prc.adsb"
    assert all_specs()["op_h1_adsb"].uses_target is False and all_specs()["op_h0_adsb"].uses_target is False


# --- the real data (opt-in) ------------------------------------------------------------------------------------

REAL = os.environ.get("OPLAD_REAL") == "1"


def _real_days() -> list[str]:
    return surface.parse_days(os.environ.get("OPLAD_ADSB_DAYS", "2025-10-14..2025-10-16").split())


def _real_links(days: list[str]) -> pd.DataFrame:
    """The challenge link (tier T: the ADS-B takeoff matched to the recorded one): read ONLY here, to find the own
    airframe whose deletion must change nothing."""
    d = Path(os.environ.get("OPLAD_LINKS_DIR", str(paths.CACHE / "adsb")))
    parts = [pd.read_parquet(d / f"links_{x}.parquet", columns=["icao", "t_takeoff"]) for x in days if (d / f"links_{x}.parquet").exists()]
    return pd.concat(parts) if parts else pd.DataFrame(columns=["icao", "t_takeoff"])


@pytest.mark.skipif(not (REAL and paths.MOVEMENTS_FILE.exists() and paths.GEO_STANDS_FILE.exists()),
                    reason="set OPLAD_REAL=1 (real ADS-B points of a few days)")
def test_real_slice_t1_t1d_and_t2_with_a_rebuild(monkeypatch):
    days = _real_days()
    d0, d1 = surface.day_start(days[0]), surface.day_start(days[-1]) + 86_400
    pts = pd.concat([surface.read_day(d) for d in days], ignore_index=True)
    assert len(pts) > 100_000, "no points for the real slice"
    geo = surface.load_geometry()
    af = surface.airborne_first_times(pts)
    ev = surface.episodes(pts, geo, d1 - 1, af)
    mvt = pd.read_parquet(paths.MOVEMENTS_FILE, filters=[("MVT_TIME_UTC_mvt", ">=", pd.Timestamp(d0 - 6 * 3600, unit="s", tz="UTC")),
                                                           ("MVT_TIME_UTC_mvt", "<", pd.Timestamp(d1, unit="s", tz="UTC"))])
    if "mvt_id" in mvt.columns:
        mvt = mvt.set_index("mvt_id")
    table = S.table(ev)
    support = _adsblib.support_table(_adsblib.surface_at(
        mvt, mvt.index[mvt["is_dep"].to_numpy(dtype=bool)], _oplib.anchor_h1(_oplib.departures(mvt)), {"support": False},
        events=table, return_internals=True)[1])
    state = {"events": table}
    monkeypatch.setattr(_adsblib, "load_events", lambda t_min=None, t_max=None: state["events"])
    monkeypatch.setattr(_adsblib, "load_support", lambda: support)
    dep = mvt.loc[mvt["is_dep"].to_numpy(dtype=bool) & mvt["split"].astype(str).eq("train").to_numpy()]
    a_all = _oplib.anchor_h1(dep)
    covered = build(mvt, ("op_h1_adsb",), dep.index)["op_h1_adsb_cov_taxi_ok"].eq(1).to_numpy()
    inside = (a_all >= d0 + 8 * 3600) & (a_all < d1 - 3600) & covered
    rng = np.random.default_rng(23)
    picked = []
    for apt, g in dep.loc[inside].groupby(dep.loc[inside, "apt"].astype(str)):
        # rows of one airport at least 8 h apart: one row's own airframe must not reach another row's windows (6 h)
        for r in rng.permutation(g.index):
            a_r = _oplib.anchor_h1(dep.loc[[r]])[0]
            if all(abs(a_r - _oplib.anchor_h1(dep.loc[[x]])[0]) >= 8 * 3600 for x in picked if dep.loc[x, "apt"] == apt):
                picked.append(r)
            if sum(dep.loc[x, "apt"] == apt for x in picked) >= 2:
                break
    rows = pd.Index(picked, name=mvt.index.name)
    assert len(rows) >= 10, len(rows)
    # T1, crossing case: the own takeoff 60 s before its own H1 anchor
    base = build(mvt, ("op_h1_adsb",), rows)
    moved = _set(mvt, "MVT_TIME_UTC_mvt", rows, _oplib.anchor_h1(mvt.loc[rows]) - 60)
    assert_identical(base, build(moved, ("op_h1_adsb",), rows))
    # T1d: the linked own airframe's departure points deleted (after its last landing before the takeoff), one row at a
    # time, the table rebuilt at the row's airport: the identity rules must already have excluded all of it
    links = _real_links(days)
    t_pts = pts["t"].to_numpy()
    n_t1d = 0
    for r in rows.intersection(links.index):
        icao, t_tko = str(links.loc[r, "icao"]), float(links.loc[r, "t_takeoff"])
        land = ev.loc[ev["icao"].eq(icao) & ev["kind"].eq("LANDING") & (ev["t_start"] < t_tko), "t_start"]
        t_from = max(float(land.max()) + 1200 if len(land) else -np.inf, t_tko - 3 * 3600)
        apt = str(mvt.loc[r, "apt"])
        at = pts["apt"].astype(str).eq(apt).to_numpy() & (t_pts >= d0 - 6 * 3600)
        own = pts["icao"].astype(str).str.lower().eq(icao.lower()).to_numpy() & (t_pts >= t_from) & (t_pts <= t_tko + 300)
        idx = pd.Index([r], name=mvt.index.name)
        state["events"] = S.table(surface.episodes(pts.loc[at], geo, d1 - 1, af))
        full = build(mvt, ("op_h1_adsb",), idx)
        state["events"] = S.table(surface.episodes(pts.loc[at & ~own], geo, d1 - 1, af))
        assert_identical(full, build(mvt, ("op_h1_adsb",), idx))
        n_t1d += 1
    assert n_t1d >= 5 or len(links) == 0, n_t1d
    # T2 with a rebuild at q = floor(a) - 30 s, one anchor at a time, the anchor's airport only
    for r in rows:
        q_r = mvt.loc[[r]]
        a = _oplib.anchor_h1(q_r)[0]
        q = np.floor(a) - LAG
        idx = pd.Index([r], name=mvt.index.name)
        state["events"] = table
        full = build(mvt, ("op_h1_adsb",), idx)
        cm, _ = censor(mvt, pd.Series(dtype="float64"), q)
        apt = str(q_r["apt"].iloc[0])
        window = pts.loc[(np.floor(t_pts) <= q) & (t_pts >= q - 14 * 3600) & pts["apt"].astype(str).eq(apt).to_numpy()]
        state["events"] = S.table(surface.episodes(window, geo, q, af[af <= q]))
        assert_identical(full, build(cm, ("op_h1_adsb",), idx))
