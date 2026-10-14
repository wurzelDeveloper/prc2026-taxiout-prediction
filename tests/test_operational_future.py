"""Traffic after the H1 anchor (`prc.features.operational_future`, findings 10.36): leakage gates and the contract.

The four blocks describe OTHER flights' traffic after the H1 anchor ``a`` (15 min before the planned off-block):
``op_h1_fut_plan`` (tier O*: the other departures' planned off-blocks and the other arrivals' scheduled times in
``(a, a + w]``, columns ``w15`` / ``w30`` / ``w60``) and three tier F blocks, the H1 past-window reads run again at
the read times ``a + 15``, ``a + 30`` and ``a + 45`` min (columns ``f15`` / ``f30`` / ``f45``): ``op_h1_fut_rec``
(movement records), ``op_h1_fut_nm`` (NM readings; fold-dependent) and ``op_h1_fut_adsb`` (the public ADS-B surface
picture, the own airframe's whole ground stay removed). Tier F reads the REALISED movements of the other flights where
an A-CDM airport would hold plans (AMAN, DMAN, A-SMGCS): an upper bound for perfect plans, never an operational input.
The own flight's outcome (its takeoff, off-block, NM actuals, taxi) must never enter, except to identify -- and remove
-- the own airframe. The gates, each test named after its gate:

1. contract: exactly the declared outputs on the query index (float32); a NaN anchor gives an all-NaN row; no slot
   column (`operational_future.SLOT_COLUMNS`) and no ADS-B coverage column is an output; the F columns are the H1
   reads at the read times; the NM block's fold-dependence; flags and the source scan; every prc.features module the
   reused reads reach (`traffic_at`, `nm_at`, `surface_at`, `window_covered`; function-local imports included) is
   hashed into the blocks' cache keys;
2. T1: the own takeoff moved anywhere (records; +-30 s for ADS-B, inside its link tolerance, and +-300 s where the
   callsign links it), its AOBT_3 / ARVT_3 / EOBT_1 moved: bit-identical, the perturbed rows kept in the frame and fit;
   the own airframe's ADS-B points moved or re-flown with the push before a read, the event table rebuilt: the same;
3. bounded future: every OTHER flight's events after the last read time (``a + 45`` min) censored: bit-identical
   (records and NM, the archive tables frozen as in tests/test_operational.py; ADS-B: the other airframes' events as
   they stood at the last query time, on random and on boundary anchors); plans outside ``(a, a + 60 min]``: no effect;
4. inverted T2: the blocks DO read the other flights' future -- moving it changes them; an event exactly at a read
   time counts, one a second later does not;
5. ``op_h1_fut_plan`` against brute force, with plans exactly at ``a`` (out), ``a + 1`` and ``a + w`` (in), ``a + w + 1``
   (out), and the IOBT / LOBT / schedule fall-back;
6. the own-airframe stay: the link rules (the flight's callsign within 600 s, nearest; else the only free takeoff
   within 45 s, provided no other unlinked departure took off within 45 s of it -- mutual uniqueness; a takeoff claimed
   twice goes to the nearer departure; an event without an airport or an airframe never links); exactly the own stay
   relabelled (not the airframe's previous rotation, not its events at another airport, not another airframe) with
   the departure's own normalised callsign, or -- for a departure without one -- with the token ``ZZOWN<mvt_id>``,
   which its query then carries too; both rules act: the own taxi under the inbound callsign from a first sighting
   away from the stand is not counted (the H1 rules alone would count it), and neither is another airframe's event
   under the flight's own callsign; no own ADS-B takeoff, or two unlinked departures near one free takeoff: all-NaN
   rows;
7. parameters: ``shift_days`` a whole number >= 0 (bools, floats, None, strings and negatives raise), ``min_support`` a
   whole number >= 1, unknown keys raise, the plan block takes none; the placebo reads exactly the past, is NaN --
   never zero -- where it reaches before the frame's records, and keeps every other row as read; the ADS-B placebo
   leaves out the same flight a week earlier through its callsign;
8. quarantine and tiers: no config with ``model.refit_on_all`` resolves a tier F block (`TIER_F_REFIT`: empty until
   decided); every config with the ADS-B F block blanks it on part of the fit; op_h1_f4_norwy drops exactly the F
   columns `_long_tiers` tags ``own_runway``; op_h1_f4_nodepthru keeps, after the anchor, only the plans and the
   realised arrivals; the rule T variants differ from op_h1_f4_adsb only in what they name; every F output is tier F
   at H1, every plan output O or O*.

Known residual traces of the own movement, which no test here can remove because they are in the other flights'
realised movements themselves (the own movement shares resources with them; findings 10.36):
- the runway slot: the own takeoff takes one slot from the others' takeoff counts, at the own runway and at the
  airport (where one runway is in use the two are the same);
- the next arrival at the own stand: it blocks in only after the own push (arrivals taxiing, their taxi-in anomaly,
  the ADS-B taxi-ins; arrival events are never own-excluded);
- the pier's pushback lanes, the de-icing pads and the runway holding queue: the own push, de-icing and hold delay or
  advance the others' (pier backlog, pushes, remote stops, holds, the others' taxi times).
Rule T removes the departure-side traces by construction: its member, op_h1_f4_nodepthru (delivered as the tier F
file h1f), reads no realised departure-side column after the anchor, and nothing from that side is credited. F5, the
full concept delivered beside it as h1f5 (findings 10.37), does read the realised departure side, so it carries these
traces. What stays
in the member is bounded by no run: the next arrival at the own stand (unmeasured); where a runway is in mixed mode,
the landings on the own runway, whose slots the own takeoff shares (unmeasured); and the missingness of the ADS-B
taxi-in columns (below; about 0.15 s of MAE for F4). The shift-7 placebo (``shift_days``) bounds only the airport's
era and the flight's identity, not these traces: the own movement is never in the placebo's window. Which rows of
``op_h1_fut_adsb`` are NaN depends on identifying the own takeoff
(`test_own_stay_no_own_adsb_takeoff_is_an_all_nan_row`), a missingness path the placebo shares (its stays come from
the real day), so a gain carried by that missingness survives the placebo and counts against rule T's criterion 2.

Strict xfails pin what these gates found, until it is fixed: the placebo guard reaches 3 h before the earliest
shifted read, but the runway configuration age looks back up to 12 h (`PLACEBO_REACH`); a time-only link to an aircraft
without a movement record (the own takeoff not received) leaves the own taxi under its inbound callsign counted
(`WRONG_LINK`); ``op_h1_fut_adsb`` accepts ``shift_days`` values the record blocks reject (`ADSB_SHIFT_PARAMS`).

Synthetic only (the record world of tests/test_operational.py; the ADS-B world of tests/adsb_synth.py through
tests/test_operational_adsb.py): no competition data, no network.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re

import numpy as np
import pandas as pd
import pytest

import adsb_synth as S
from prc import paths
from prc.config import load_config
from prc.features import _adsblib, _long_tiers, _oplib, _proxylib, all_specs, resolve
from prc.features import operational_future as future
from prc.features.build import compute_block
from prc.features.registry import FeatureContext
from test_operational import (_freeze_archives, _set, _shift, assert_identical, censor, spaced_departures,
                              synthetic_world, t1_rows)
from test_operational_adsb import (LAG, _make_world, _own_departures, _patch, _regenerate, anchored, boundary_anchors,
                                   rebuild, view_at)

PLAN, REC, NM, ADSB = "op_h1_fut_plan", "op_h1_fut_rec", "op_h1_fut_nm", "op_h1_fut_adsb"
BLOCKS = (PLAN, REC, NM, ADSB)
RECORD_BLOCKS = (PLAN, REC, NM)
"""The blocks that read the movement frame alone."""
TIER_F = (REC, NM, ADSB)
TIER_F_REFIT: tuple[str, ...] = ()
"""Refit (board / final) configs allowed to resolve a tier F block. Empty: tier F is holdout-only until a decision
names a config here, with its reason (findings 10.36)."""
LAST_READ_S = max(future.SHIFTS.values())
PLAN_HORIZON_S = max(future.PLAN_WINDOWS.values())
NM_ACTUALS = ("AOBT_3_flt", "ARVT_3_flt", "EOBT_1_flt")
CENSORED = ("MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt", "ARVT_3_flt")
"""The columns `test_operational.censor` censors."""
PLAN_COLUMNS = ("IOBT_flt", "LOBT_flt", "SCHED_TIME_UTC_mvt")
EVENT_CATS = ("apt", "kind", "icao", "cs", "origin_kind", "rwy_end")
FUTURE_PREFIXES = ("op_h1_fplan_", "op_h1_frec_", "op_h1_fnm_", "op_h1_fadsb_")
"""Every column of the four blocks (after the anchor) starts with one of these."""


# --- helpers ---------------------------------------------------------------------------------------------------------

def _refuse(*_args, **_kwargs):
    raise AssertionError("an H1 block read the airport's off-block records (the target file)")


def build(mvt: pd.DataFrame, names, idx: pd.Index, params: dict | None = None,
          fit_mask: pd.Series | None = None) -> pd.DataFrame:
    """The named blocks through the contract validator, side by side; ``params`` per block name; the fit mask of
    training (every training departure, perturbed rows included) for the fold-dependent block."""
    specs = all_specs()
    if fit_mask is None:
        fit_mask = pd.Series(mvt["split"].eq("train").to_numpy() & mvt["is_dep"].to_numpy(), index=mvt.index)
    parts = []
    for name in names:
        spec = specs[name]
        ctx = FeatureContext(dep_index=idx, params=dict((params or {}).get(name, {})),
                             fit_mask=fit_mask if spec.uses_target else None,
                             target=pd.Series(0.0, index=idx) if spec.uses_target else None)
        parts.append(compute_block(spec, mvt, ctx))
    return pd.concat(parts, axis=1)


def outputs(name: str, tag: str | None = None) -> list[str]:
    """A block's outputs, optionally those of one read time (``f15`` / ``f30`` / ``f45``)."""
    return [c for c in all_specs()[name].outputs if tag is None or f"_{tag}_" in c]


def anchor(mvt: pd.DataFrame, r) -> float:
    return float(_oplib.anchor_h1(mvt.loc[[r]])[0])


def one(r) -> pd.Index:
    return pd.Index([r], name="mvt_id")


def training_departures(mvt: pd.DataFrame) -> pd.Index:
    return mvt.index[(mvt["is_dep"] & mvt["split"].eq("train")).to_numpy()]


def censor_others(mvt: pd.DataFrame, offblocks: pd.Series, cutoff: float, own: pd.Index) -> pd.DataFrame:
    """Every OTHER flight's events after ``cutoff`` removed (`test_operational.censor`); the own rows as they were."""
    out, _ = censor(mvt, offblocks, cutoff)
    out.loc[own, list(CENSORED)] = mvt.loc[own, list(CENSORED)]
    return out


def others_moved_later(mvt: pd.DataFrame, lo: float, hi: float, by: float, own: pd.Index) -> pd.DataFrame:
    """Other flights' takeoffs and landings in ``(lo, hi]`` moved ``by`` s later, the arrivals' in-blocks with them."""
    t = _oplib.secs(mvt["MVT_TIME_UTC_mvt"])
    rows = mvt.index[(t > lo) & (t <= hi) & ~mvt.index.isin(own)]
    out = _shift(mvt, "MVT_TIME_UTC_mvt", rows, by)
    return _shift(out, "BLOCK_TIME_UTC_mvt", rows[~mvt.loc[rows, "is_dep"].to_numpy(dtype=bool)], by)


def plans(mvt: pd.DataFrame) -> np.ndarray:
    """Every row's plan: a departure's planned off-block (IOBT, else LOBT, else schedule), an arrival's schedule."""
    return _oplib.planned_offblock(mvt)


def plans_moved_out(mvt: pd.DataFrame, a: float, own: pd.Index) -> pd.DataFrame:
    """Other departures planned off, and other arrivals scheduled, in ``(a, a + 60 min]`` moved to ``a + 70 min``."""
    p = plans(mvt)
    inside = (p > a) & (p <= a + PLAN_HORIZON_S) & ~mvt.index.isin(own)
    dep = mvt["is_dep"].to_numpy(dtype=bool)
    later = a + PLAN_HORIZON_S + 600.0
    out = _set(mvt, "IOBT_flt", mvt.index[inside & dep], np.full(int((inside & dep).sum()), later))
    return _set(out, "SCHED_TIME_UTC_mvt", mvt.index[inside & ~dep], np.full(int((inside & ~dep).sum()), later))


def plans_outside_removed(mvt: pd.DataFrame, a: float, own: pd.Index) -> pd.DataFrame:
    """Every other flight's plan outside ``(a, a + 60 min]`` removed (IOBT, LOBT and schedule blanked)."""
    p = plans(mvt)
    outside = ~((p > a) & (p <= a + PLAN_HORIZON_S)) & ~mvt.index.isin(own)
    out = mvt.copy()
    out.loc[outside, list(PLAN_COLUMNS)] = pd.NaT
    return out


def events_table(rows: list[dict], like: pd.DataFrame | None = None) -> pd.DataFrame:
    """Events as the snapshot table holds them (`adsb_synth.table`): ``rows`` (unset columns NaN, no taxi seen, no
    parent) after the events of ``like``."""
    cols = list(like.columns) if like is not None else ["eid", "apt", "kind", "icao", "cs", "t_start", "t_known_start",
                                                        "t_end", "t_known_end", "origin_kind", "origin_stand", "rwy_end",
                                                        "dur", "stop_s", "path_m", "taxi_seen", "parent"]
    extra = pd.DataFrame([{**{c: np.nan for c in cols}, "eid": -1, "taxi_seen": 0, "parent": -1, **r} for r in rows])
    parts = [extra] if like is None else [like.astype({c: object for c in EVENT_CATS}), extra]
    return S.table(pd.concat(parts, ignore_index=True))


def event(apt, kind: str, icao, cs, t: float, known: float = 0.0, end: float = np.nan,
          known_end: float = np.nan, **kw) -> dict:
    return {"apt": apt, "kind": kind, "icao": icao, "cs": cs, "t_start": float(t), "t_known_start": float(t) + known,
            "t_end": float(end), "t_known_end": float(known_end), **kw}


def _feature_imports(module) -> set[str]:
    """The prc.features modules ``module`` imports anywhere in its source, function-local imports included (the
    registry and the builder excepted: they are never hashed)."""
    out: set[str] = set()
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == "prc.features":
                out |= {a.name for a in node.names}
            elif node.module.startswith("prc.features."):
                out.add(node.module.split(".")[2])
        elif isinstance(node, ast.Import):
            out |= {a.name.split(".")[2] for a in node.names if a.name.startswith("prc.features.")}
    return out - {"registry", "build"}


def _plain(values) -> list:
    """The values with every missing one as None: `own_airframe_stays`' ``cs`` is a string column, which holds NaN
    for a departure without a callsign once another row holds a string (pandas 3), None otherwise."""
    return [v if isinstance(v, str) or not pd.isna(v) else None for v in values]


def _kept(name: str) -> list[str]:
    """The model matrix's columns of a config: every output of its resolved blocks but its ``drop_columns``."""
    cfg = load_config(name)
    return [c for s in resolve(cfg.features.include, cfg.features.exclude) for c in s.outputs
            if c not in cfg.features.drop_columns]


def _blocks(name: str) -> set[str]:
    cfg = load_config(name)
    return {s.name for s in resolve(cfg.features.include, cfg.features.exclude)}


# --- worlds ----------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def records():
    """The record world of tests/test_operational.py: 16 days across a month boundary, two airports."""
    return synthetic_world()


@pytest.fixture
def world(records, monkeypatch):
    """The record world, with the off-block loader refusing: no H1 block may read an off-block record."""
    monkeypatch.setattr(_oplib, "load_offblocks", _refuse)
    return records


def _callsign_linked(w) -> pd.Index:
    """Departures whose own ADS-B takeoff carries the flight's callsign (the link's first rule finds them)."""
    tko = w.table.loc[w.table["kind"].astype(str).eq("TAKEOFF")]
    assert not tko["icao"].astype(str).duplicated().any()  # one takeoff per airframe in this world
    cs_of = dict(zip(tko["icao"].astype(str), tko["cs"].astype(str)))
    dep = w.mvt.loc[w.mvt["is_dep"].to_numpy(dtype=bool)]
    flt = pd.Series(_adsblib.norm_callsign(dep["CALLSIGN_flt"].to_numpy(dtype=object)), index=dep.index)
    own = w.info["icao"].reindex(dep.index).map(cs_of)
    return dep.index[(flt.notna() & own.eq(flt)).to_numpy(dtype=bool)]


def _covered_spaced(w, hours: float = 24.0) -> pd.Index:
    """Training departures with a stand and a runway whose ADS-B taxi family is gated in at all three read times, at
    least ``hours`` apart per airport (one row's own flight cannot reach another row's windows)."""
    dep = w.mvt.loc[w.mvt["is_dep"].to_numpy(dtype=bool) & w.mvt["split"].eq("train").to_numpy()]
    ok = (dep["STAND_mvt"].notna() & dep["RUNWAY_mvt"].notna()).to_numpy()
    for tag in future.SHIFTS:
        ok = ok & w.F[f"op_h1_fadsb_{tag}_taxi_out_now"].reindex(dep.index).notna().to_numpy()
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
    """The ADS-B world of tests/test_operational_adsb.py; ``F``: op_h1_fut_adsb on every departure; ``linked_cs``:
    the callsign-linked departures; ``t1f``: covered spaced training departures."""
    w = _make_world()
    with pytest.MonkeyPatch.context() as mp:
        _patch(mp, w)
        w.F = build(w.mvt, (ADSB,), w.mvt.index[w.mvt["is_dep"].to_numpy(dtype=bool)])
    w.linked_cs = _callsign_linked(w)
    w.t1f = _covered_spaced(w)
    return w


@pytest.fixture
def aw(adsb_world, monkeypatch):
    state = _patch(monkeypatch, adsb_world)
    monkeypatch.setattr(_oplib, "load_offblocks", _refuse)
    return adsb_world, state


# --- 1. contract -----------------------------------------------------------------------------------------------------

def test_contract_column_families_and_names():
    specs = all_specs()
    assert future.SHIFTS == {"f15": 900, "f30": 1800, "f45": 2700}
    assert future.PLAN_WINDOWS == {"w15": 900, "w30": 1800, "w60": 3600}
    assert not set(future.PLAN_WINDOWS) & set(future.SHIFTS), "a plan window is named apart from a read time"
    assert set(future.SLOT_COLUMNS) <= set(_oplib.TRAFFIC), "a slot column that is no traffic column drops nothing"
    assert future.REC == tuple(c for c in _oplib.TRAFFIC if c not in future.SLOT_COLUMNS)
    assert set(future.ADSB) == set(_adsblib.SURFACE + _adsblib.STAND) and not set(future.ADSB) & set(_adsblib.COVERAGE)
    expected = {
        PLAN: tuple(f"op_h1_fplan_{c}" for c in future.PLAN),
        REC: tuple(f"op_h1_frec_{tag}_{c}" for tag in future.SHIFTS for c in future.REC),
        NM: tuple(f"op_h1_fnm_{tag}_{c}" for tag in future.SHIFTS for c in _oplib.NM),
        ADSB: tuple(f"op_h1_fadsb_{tag}_{c}" for tag in future.SHIFTS for c in future.ADSB),
    }
    for name in BLOCKS:
        outs = specs[name].outputs
        assert outs == expected[name], name
        assert len(set(outs)) == len(outs), name
        assert not [c for c in outs for slot in future.SLOT_COLUMNS if slot in c], name
        assert specs[name].group == name, "one block per group: group:<block> selects exactly it"
        assert all(c.startswith(FUTURE_PREFIXES) for c in outs), name
    assert not [c for c in specs[ADSB].outputs if "cov_" in c], "coverage columns gate, they are no inputs"
    assert len(future.PLAN) == 15 and all(c.startswith(("dep_plan_", "arr_sched_")) for c in future.PLAN)
    assert all(c.rsplit("_", 1)[1] in future.PLAN_WINDOWS for c in future.PLAN), "every plan column ends in its window"


def test_contract_flags_externals_and_the_source():
    specs = all_specs()
    for name in BLOCKS:
        src = inspect.getsource(specs[name].fn)
        assert ("ctx.target" in src or "ctx.fit_mask" in src) == specs[name].uses_target, name
    assert [n for n in BLOCKS if specs[n].uses_target] == [NM]
    assert set(specs[ADSB].external) == set(future.ADSB_EXTERNAL) and paths.ADSB_SURFACE_EVENTS_FILE in specs[ADSB].external
    assert not [n for n in BLOCKS if paths.TARGET_FILE in specs[n].external]
    assert not [n for n in (PLAN, REC, NM) if specs[n].external]
    src = inspect.getsource(future)
    code = re.sub(r'"""[\s\S]*?"""', "", src)
    for word in ("EOBT_1_flt", "ARVT_3_flt", "AOBT_3_flt", "TARGET_FILE", "load_offblocks", "load_target",
                 "target.parquet"):
        assert word not in code, word  # the NM actuals reach these blocks only through `_oplib.nm_at` (others')
    assert not re.search(r"(from|import)\s+prc\.adsb", src), "feature code never imports prc.adsb"
    body = src.split('"""', 2)[2]
    assert not [ln for ln in body.splitlines() if re.match(r"^[a-z_]+\(", ln)], "import-time work"


def test_contract_every_module_the_reads_reach_is_hashed():
    """`build._related_modules` hashes a block's module and the modules in its namespace, one level deep. The F blocks
    reuse `_oplib.traffic_at` / `nm_at`, `_adsblib.surface_at` and `_horizon.window_covered`, whose helpers sit further
    down (and `_adsblib` imports `_geolib` and `_oplib` inside its functions): every prc.features module reachable from
    operational_future, by any import, must sit in its namespace, or editing it would leave these blocks' caches warm.
    Derived from the sources, not listed (as tests/test_operational.py lists them for `prc.features.operational`)."""
    from prc.features import operational, operational_adsb
    from prc.features.build import _related_modules

    hashed = {m.__name__.rsplit(".", 1)[-1] for m in _related_modules(future)}
    seen: set[str] = set()
    todo = ["operational_future"]
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        todo += sorted(_feature_imports(importlib.import_module(f"prc.features.{name}")))
    assert {"_oplib", "_adsblib", "_horizon", "_geolib"} <= seen, "the closure follows the reused reads"
    assert sorted(seen - hashed) == [], "reachable from operational_future but not hashed into its blocks"
    for mod in (operational, operational_adsb):  # every helper the reused blocks hash, these blocks hash too
        theirs = {m.__name__.rsplit(".", 1)[-1] for m in _related_modules(mod)} - {mod.__name__.rsplit(".", 1)[-1]}
        assert sorted(theirs - hashed) == [], mod.__name__
    for mod in ("_oplib", "_adsblib", "_arrlib", "_geolib", "_horizon", "_proxylib", "_wxlib", "congestion", "core",
                "stand_runway", "temporal", "turnaround", "weather", "weather_fcst"):
        assert mod in hashed, mod


def test_contract_every_block_returns_its_outputs_on_the_query_index(world):
    mvt, _ = world
    dep = mvt.index[mvt["is_dep"].to_numpy()]
    idx = pd.Index(np.random.default_rng(1).permutation(dep)[:500], name="mvt_id")  # shuffled, both splits
    assert mvt.loc[idx, "split"].eq("rank").any() and not idx.is_monotonic_increasing
    fit = pd.Series(mvt["split"].eq("train").to_numpy() & mvt["is_dep"].to_numpy(), index=mvt.index)
    for name in RECORD_BLOCKS:
        spec = all_specs()[name]
        ctx = FeatureContext(dep_index=idx, fit_mask=fit if spec.uses_target else None,
                             target=pd.Series(0.0, index=idx) if spec.uses_target else None)
        out = spec.fn(mvt, ctx)
        assert list(out.columns) == list(spec.outputs), name
        assert out.index.equals(idx), name
        assert (out.dtypes == "float32").all(), name
    X = build(mvt, RECORD_BLOCKS, idx)
    # something is measured, at every read time: not a frame of NaN and zeros
    probes = ["op_h1_fplan_dep_plan_apt_w60", "op_h1_fplan_dep_plan_rwy_w30", "op_h1_fplan_arr_sched_pier_w60"]
    for tag in future.SHIFTS:
        probes += [f"op_h1_frec_{tag}_dep_apt_b15", f"op_h1_frec_{tag}_arr_taxiing", f"op_h1_frec_{tag}_rwy_config_age_min",
                   f"op_h1_fnm_{tag}_nm_apt_b60_anom_mean", f"op_h1_fnm_{tag}_backlog_apt"]
    for col in probes:
        v = X[col].to_numpy(dtype="float64")
        assert np.isfinite(v).mean() > 0.3 and np.nanstd(v) > 0, col


def test_contract_a_nan_anchor_gives_an_all_nan_row(world):
    mvt, _ = world
    dep = mvt.index[mvt["is_dep"].to_numpy()]
    rng = np.random.default_rng(2)
    blank = pd.Index(rng.choice(dep, 12, replace=False), name="mvt_id")
    unplanned = mvt.copy()
    unplanned.loc[blank, list(PLAN_COLUMNS)] = pd.NaT
    assert np.isnan(_oplib.anchor_h1(unplanned.loc[blank])).all()
    idx = pd.Index(rng.choice(dep, 200, replace=False), name="mvt_id").union(blank)
    X = build(unplanned, RECORD_BLOCKS, idx)
    assert X.loc[blank].isna().all().all()
    assert X.loc[idx.difference(blank)].notna().any(axis=1).all()


def test_contract_adsb_block_on_its_world(aw):
    w, _ = aw
    dep = w.mvt.index[w.mvt["is_dep"].to_numpy(dtype=bool)]
    idx = pd.Index(np.random.default_rng(3).permutation(dep)[:300], name="mvt_id")
    spec = all_specs()[ADSB]
    out = spec.fn(w.mvt, FeatureContext(dep_index=idx))
    assert list(out.columns) == list(spec.outputs) and out.index.equals(idx) and (out.dtypes == "float32").all()
    # a covered world: every family measured at every read time, and every own airframe found -- and only it
    for tag in future.SHIFTS:
        for col in ("taxi_out_now", "taxi_in_now", "rwy_hold_now", "push_15", "taxi_dur_mean_60"):
            v = w.F[f"op_h1_fadsb_{tag}_{col}"].to_numpy(dtype="float64")
            assert np.isfinite(v).mean() > 0.5 and np.nanstd(v) > 0, (tag, col)
    stays = future.own_airframe_stays(w.table, w.mvt.loc[dep])
    assert len(stays) == len(dep)
    assert (stays["icao"].astype(str) == w.info.loc[stays.index, "icao"].astype(str)).all(), "a wrong airframe linked"
    # a NaN anchor: an all-NaN row
    blank = pd.Index(idx[:8], name="mvt_id")
    unplanned = w.mvt.copy()
    unplanned.loc[blank, list(PLAN_COLUMNS)] = pd.NaT
    X = build(unplanned, (ADSB,), idx)
    assert X.loc[blank].isna().all().all() and X.loc[idx[8:]].notna().any(axis=1).mean() > 0.5


def test_contract_the_nm_block_is_fold_dependent_through_the_fit_mask(world):
    """``uses_target`` has teeth: the NM proxy baseline comes from the fold's fit rows (month-out), so another fit mask
    moves the anomaly columns while the plan backlog (no baseline) stays; the target values handed over are never
    read."""
    mvt, _ = world
    rows = t1_rows(mvt)
    fit = pd.Series(mvt["split"].eq("train").to_numpy() & mvt["is_dep"].to_numpy(), index=mvt.index)
    half = fit.copy()
    half.loc[half.index[half.to_numpy()][::2]] = False
    a, b = build(mvt, (NM,), rows, fit_mask=fit), build(mvt, (NM,), rows, fit_mask=half)
    anomaly = [c for c in outputs(NM) if "anom" in c]
    assert not a[anomaly].equals(b[anomaly])
    backlog = [c for c in outputs(NM) if "backlog" in c]
    assert_identical(a[backlog], b[backlog])
    spec = all_specs()[NM]
    ctx = FeatureContext(dep_index=rows, fit_mask=fit, target=pd.Series(1e9, index=rows))
    assert_identical(a, compute_block(spec, mvt, ctx))


def test_contract_the_f_columns_are_the_h1_reads_at_the_shifted_times(world):
    """``op_h1_frec_<tag>_<c>`` / ``op_h1_fnm_<tag>_<c>`` equal ``op_h1_traffic`` / ``op_h1_nm`` of the same flight with its
    anchor moved to ``a + shift`` (its IOBT moved; only rows whose planned month stays, so the NM baseline is the same)."""
    mvt, _ = world
    rows = spaced_departures(mvt, hours=20)
    rows = rows[mvt.loc[rows, "IOBT_flt"].notna().to_numpy()]
    month = _oplib.baseline_month(mvt.loc[rows])
    stays = month == _oplib.baseline_month(_shift(mvt, "IOBT_flt", rows, LAST_READ_S).loc[rows])
    rows = rows[stays]
    assert len(rows) >= 10
    base = build(mvt, (REC, NM), rows)
    for tag, s in future.SHIFTS.items():
        h1 = build(_shift(mvt, "IOBT_flt", rows, s), ("op_h1_traffic", "op_h1_nm"), rows)
        for prefix, block, cols in (("op_h1_frec_", REC, future.REC), ("op_h1_fnm_", NM, _oplib.NM)):
            got = base[[f"{prefix}{tag}_{c}" for c in cols]].set_axis(list(cols), axis=1)
            want = h1[[f"op_h1_{c}" for c in cols]].set_axis(list(cols), axis=1)
            assert_identical(got, want)


# --- 2. T1: the own flight -------------------------------------------------------------------------------------------

OWN_TAKEOFF_MODES = ("plus600", "minus300", "cross_h1", "cross_bucket", "at_f15", "at_f30", "at_f45",
                     "edge_f45_b60", "after_f45", "bucket_before_f45")


def own_takeoff_moved(mvt: pd.DataFrame, rows: pd.Index, mode: str) -> pd.DataFrame:
    """The rows' own recorded takeoffs moved: by +600 / -300 s, before the own anchor, into a 15-min bucket complete
    before the anchor or before the last read, exactly onto the read times (the windows' inclusive edge), onto the lower
    edge of the last read's 60-min window, one second after the last read."""
    if mode == "plus600":
        return _shift(mvt, "MVT_TIME_UTC_mvt", rows, 600)
    if mode == "minus300":
        return _shift(mvt, "MVT_TIME_UTC_mvt", rows, -300)
    a = _oplib.anchor_h1(mvt.loc[rows])
    target = {
        "cross_h1": a - 60,
        "cross_bucket": (np.floor(a / 900) - 2) * 900 + 17,
        "at_f15": a + future.SHIFTS["f15"],
        "at_f30": a + future.SHIFTS["f30"],
        "at_f45": a + LAST_READ_S,
        "edge_f45_b60": a + LAST_READ_S - 3600,
        "after_f45": a + LAST_READ_S + 1,
        "bucket_before_f45": (np.floor((a + LAST_READ_S) / 900) - 2) * 900 + 17,
    }[mode]
    return _set(mvt, "MVT_TIME_UTC_mvt", rows, target)


@pytest.mark.parametrize("mode", OWN_TAKEOFF_MODES)
def test_t1_own_takeoff_moved_anywhere_changes_no_record_block(world, mode):
    mvt, _ = world
    rows = t1_rows(mvt)  # spaced, one planned month (the NM baseline is month-out)
    assert len(rows) >= 12
    base = build(mvt, RECORD_BLOCKS, rows)
    assert base[outputs(REC, "f45")].notna().any(axis=1).all() and base[outputs(NM, "f45")].notna().any(axis=1).all()
    assert_identical(base, build(own_takeoff_moved(mvt, rows, mode), RECORD_BLOCKS, rows))


@pytest.mark.parametrize("mode", ["plus600", "minus600", "before_a", "at_f30"])
def test_t1_own_nm_actuals_and_eobt_moved_change_no_record_block(world, mode):
    mvt, _ = world
    rows = t1_rows(mvt)
    base = build(mvt, RECORD_BLOCKS, rows)
    a = _oplib.anchor_h1(mvt.loc[rows])
    moved = mvt
    for k, col in enumerate(NM_ACTUALS):
        if mode in ("plus600", "minus600"):
            moved = _shift(moved, col, rows, 600 if mode == "plus600" else -600)
        else:
            moved = _set(moved, col, rows, (a - 300 + 100 * k) if mode == "before_a" else (a + future.SHIFTS["f30"] - k))
    assert_identical(base, build(moved, RECORD_BLOCKS, rows))


@pytest.mark.parametrize("mode", ["cross_bucket", "at_f45", "bucket_before_f45"])
def test_t1_own_takeoff_moved_with_the_own_runway_unknown(world, mode):
    """The rows' own runway blank (the runway mix and the configuration age then never count the own takeoff, and the
    own-runway groups are the airport's unknown runway): the own takeoff still changes nothing."""
    mvt, _ = world
    rows = t1_rows(mvt)
    blind = mvt.copy()
    blind.loc[rows, "RUNWAY_mvt"] = np.nan
    base = build(blind, RECORD_BLOCKS, rows)
    assert base["op_h1_frec_f45_rwy_own_share_b60"].isna().all() and base["op_h1_frec_f45_rwy_config_age_min"].notna().all()
    assert_identical(base, build(own_takeoff_moved(blind, rows, mode), RECORD_BLOCKS, rows))


@pytest.mark.parametrize("mode", ["at_f30", "plus600"])
def test_t1_own_flight_moved_changes_nothing_for_rows_outside_the_fit(world, mode):
    """Ranking rows (never in the fit, so never in the NM baseline): their own takeoff, AOBT_3, ARVT_3 and EOBT_1 moved."""
    mvt, _ = world
    rows = spaced_departures(mvt, hours=20, split="rank")
    assert len(rows) >= 4
    base = build(mvt, RECORD_BLOCKS, rows)
    moved = own_takeoff_moved(mvt, rows, mode)
    for col in NM_ACTUALS:
        moved = _shift(moved, col, rows, 900)
    assert_identical(base, build(moved, RECORD_BLOCKS, rows))


def test_t1_guard_the_own_takeoff_does_reach_the_unexcluded_counts(world):
    """Guard the guard: at the f45 read the own takeoff sits in the airport's 15-min window; counted without the
    self-exclusion, it would add one."""
    mvt, _ = world
    rows = t1_rows(mvt)
    moved = own_takeoff_moved(mvt, rows, "at_f45")
    dep = _oplib.departures(moved)
    t = _oplib.secs(dep["MVT_TIME_UTC_mvt"])
    g = dep["apt"].astype("string").to_numpy(dtype=object)
    q = _oplib.anchor_h1(moved.loc[rows]) + LAST_READ_S
    qa = moved.loc[rows, "apt"].astype("string").to_numpy(dtype=object)
    own = pd.Series(t, index=dep.index).reindex(rows).to_numpy()
    raw, excl = _oplib.backward_count(t, g, q, qa, 900), _oplib.backward_count(t, g, q, qa, 900, own)
    assert (raw - excl == 1).all()
    np.testing.assert_array_equal(build(moved, (REC,), rows)["op_h1_frec_f45_dep_apt_b15"].to_numpy(), excl)


@pytest.mark.parametrize("mode", ["plus30", "minus30", "callsign_plus300", "callsign_minus300", "nm_actuals"])
def test_t1_own_flight_moved_changes_nothing_in_the_adsb_block(aw, mode):
    """+-30 s for every row (inside the time-only link's 45 s); +-300 s where the callsign links the own takeoff (600 s);
    the own AOBT_3 / ARVT_3 / EOBT_1 by +600 s."""
    w, _ = aw
    rows = w.t1f.intersection(w.linked_cs) if mode.startswith("callsign") else w.t1f
    assert len(rows) >= 6
    assert len(w.t1f.difference(w.linked_cs)) >= 1 or mode.startswith("callsign"), "no time-only linked row"
    base = build(w.mvt, (ADSB,), rows)
    assert base[[f"op_h1_fadsb_{tag}_taxi_out_now" for tag in future.SHIFTS]].notna().all().all()
    if mode == "nm_actuals":
        moved = w.mvt
        for col in NM_ACTUALS:
            moved = _shift(moved, col, rows, 600)
    else:
        moved = _shift(w.mvt, "MVT_TIME_UTC_mvt", rows, {"plus30": 30, "minus30": -30, "callsign_plus300": 300,
                                                          "callsign_minus300": -300}[mode])
    assert_identical(base, build(moved, (ADSB,), rows))


@pytest.mark.parametrize("mode", ["plus300", "minus300", "push_600_before_f30", "delete"])
def test_t1_own_airframe_points_moved_change_nothing_in_the_adsb_block(aw, mode):
    """T1d after the anchor: the rows' own departure points (not their inbound legs) moved and the event table rebuilt
    from the points (`surface.episodes`): by +-300 s, lift-off included (the callsign still links it), or re-flown with
    the push 10 min before the f30 query, so the own push, taxi and hold are in progress at the reads: bit-identical,
    the whole stay removed. Deleted: no own ADS-B takeoff, so the rows whose recorded takeoff has no other ADS-B takeoff
    within 45 s are all NaN (the own airframe cannot be told apart)."""
    w, state = aw
    rows = w.t1f.intersection(w.linked_cs)
    assert len(rows) >= 6
    base = build(w.mvt, (ADSB,), rows)
    own = _own_departures(w.pts, w.info, rows)
    if mode == "delete":
        changed = w.pts.loc[~own]
    elif mode in ("plus300", "minus300"):
        changed = w.pts.copy()
        changed.loc[own, "t"] = changed.loc[own, "t"] + (300 if mode == "plus300" else -300)
    else:
        q30 = np.floor(_oplib.anchor_h1(w.mvt.loc[rows]) + future.SHIFTS["f30"]) - LAG
        changed = _regenerate(w.pts, w.info, rows, dict(zip(rows, q30 - 600)))
    state["events"] = rebuild(changed, w.geo)
    assert not state["events"].equals(w.table)
    moved = build(w.mvt, (ADSB,), rows)
    state["events"] = w.table
    if mode != "delete":
        assert_identical(base, moved)
        return
    tko = w.table.loc[w.table["kind"].astype(str).eq("TAKEOFF")]
    t_rec = pd.Series(_oplib.secs(w.mvt.loc[rows, "MVT_TIME_UTC_mvt"]), index=rows)
    alone = [r for r in rows if not ((tko["apt"].astype(str) == str(w.mvt.loc[r, "apt"]))
                                     & tko["icao"].astype(str).ne(str(w.info.loc[r, "icao"]))
                                     & (tko["t_start"] - t_rec[r]).abs().le(future.LINK_TIME_ONLY_S)).any()]
    assert len(alone) >= 4
    assert moved.loc[alone].isna().all(axis=None)


# --- 3. bounded future -----------------------------------------------------------------------------------------------

def test_bounded_future_other_flights_after_the_last_read_change_no_record_block(world, monkeypatch):
    """One query at a time: every other flight's takeoff, landing, in-block, AOBT_3 and ARVT_3 after ``a + 45 min``
    removed. The archives (arrival taxi-in medians, the NM proxy baseline) are held fixed, as in T2 of
    tests/test_operational.py: they are target-free tables over the whole frame, not events read at a time."""
    mvt, offblocks = world
    fit = pd.Series(mvt["split"].eq("train").to_numpy() & mvt["is_dep"].to_numpy(), index=mvt.index)
    _freeze_archives(monkeypatch, mvt, fit)
    rng = np.random.default_rng(11)
    rows = pd.Index(rng.choice(training_departures(mvt), 12, replace=False), name="mvt_id")
    checked = 0
    for r in rows:
        a = anchor(mvt, r)
        if not np.isfinite(a):
            continue
        full = build(mvt, (REC, NM), one(r))
        cut = censor_others(mvt, offblocks, a + LAST_READ_S, one(r))
        assert cut["MVT_TIME_UTC_mvt"].isna().sum() > 100  # the censoring removes events ...
        between = _oplib.secs(cut["MVT_TIME_UTC_mvt"])
        assert ((between > a) & (between <= a + LAST_READ_S)).sum() >= 5  # ... and keeps the ones the reads use
        assert_identical(full, build(cut, (REC, NM), one(r)))
        checked += 1
    assert checked >= 10


def test_bounded_window_plans_outside_the_next_hour_change_nothing(world):
    """``op_h1_fut_plan``: every other flight's plan outside ``(a, a + 60 min]`` removed, and every other flight's
    takeoff and landing moved: bit-identical (it reads the plans of that window, nothing else)."""
    mvt, _ = world
    rng = np.random.default_rng(12)
    rows = pd.Index(rng.choice(mvt.index[mvt["is_dep"].to_numpy()], 12, replace=False), name="mvt_id")
    for r in rows:
        a = anchor(mvt, r)
        full = build(mvt, (PLAN,), one(r))
        cut = plans_outside_removed(mvt, a, one(r))
        assert np.isfinite(plans(cut)).sum() < 0.1 * len(cut)
        assert_identical(full, build(cut, (PLAN,), one(r)))
        others = mvt.index.difference(one(r))
        moved = _shift(_shift(mvt, "MVT_TIME_UTC_mvt", others, 1234), "BLOCK_TIME_UTC_mvt", others, -777)
        assert_identical(full, build(moved, (PLAN,), one(r)))


@pytest.mark.parametrize("anchors", ["random", "boundary"])
def test_bounded_future_other_airframes_after_the_last_query_change_nothing_in_the_adsb_block(aw, anchors):
    """One query at a time, ``q`` = the last query time ``floor(a + 45 min) - 30 s``: every OTHER airframe's events as
    the table stood at ``q`` (`view_at`: later events dropped, later ends blanked) and the other flights' records after
    ``q`` removed: bit-identical. The own airframe's events stay: its ADS-B takeoff (after ``q``, often) identifies the
    stay that is removed. Rows the callsign links (the time-only rule reads the other takeoffs near the own one).
    ``boundary``: anchors placed so that ``q`` is another airframe's event start or end known exactly then, or one
    second before (`test_operational_adsb.boundary_anchors`)."""
    w, state = aw
    if anchors == "random":
        cand = training_departures(w.mvt).intersection(w.linked_cs)
        cand = cand[w.F.loc[cand, "op_h1_fadsb_f45_taxi_out_now"].notna().to_numpy()]
        rows = np.random.default_rng(19).choice(cand, 10, replace=False)
        picks = [(r, anchor(w.mvt, r)) for r in rows]
    else:
        picks = [(r, a - LAST_READ_S) for r, a in boundary_anchors(w, 40, seed=29) if r in w.linked_cs][:12]
    assert len(picks) >= 8
    finite = 0
    for r, a in picks:
        mvt, ob = anchored(w.mvt, w.ob, r, a, "h1")
        assert anchor(mvt, r) == a
        q = np.floor(a + LAST_READ_S) - LAG
        own = w.table["icao"].astype(str).eq(str(w.info.loc[r, "icao"])).to_numpy()
        state["events"] = w.table
        full = build(mvt, (ADSB,), one(r))
        state["events"] = events_table([], like=pd.concat([w.table.loc[own], view_at(w.table.loc[~own], q)],
                                                          ignore_index=True).astype({c: object for c in EVENT_CATS}))
        cut = build(censor_others(mvt, ob, q, one(r)), (ADSB,), one(r))
        state["events"] = w.table
        assert_identical(full, cut)
        finite += int(np.isfinite(full["op_h1_fadsb_f45_taxi_out_now"].iloc[0]))
    assert finite >= len(picks) // 2


# --- 4. inverted T2: the future is read --------------------------------------------------------------------------------

def test_inverted_t2_the_blocks_do_read_the_other_flights_future(world):
    """Moving the other flights' takeoffs and landings out of ``(a, a + 45 min]`` changes ``op_h1_fut_rec`` (each read
    time on its own window), and moving their plans out of ``(a, a + 60 min]`` changes ``op_h1_fut_plan``: else these
    gates would hold for blocks that read nothing after the anchor."""
    mvt, _ = world
    rng = np.random.default_rng(13)
    rows = pd.Index(rng.choice(training_departures(mvt), 12, replace=False), name="mvt_id")
    changed = {"plan": 0, **{tag: 0 for tag in future.SHIFTS}}
    for r in rows:
        a = anchor(mvt, r)
        base = build(mvt, (PLAN, REC), one(r))
        moved = build(others_moved_later(mvt, a, a + LAST_READ_S, 3000.0, one(r)), (REC,), one(r))
        for tag in future.SHIFTS:
            changed[tag] += int(not base[outputs(REC, tag)].equals(moved[outputs(REC, tag)]))
        changed["plan"] += int(not base[outputs(PLAN)].equals(build(plans_moved_out(mvt, a, one(r)), (PLAN,), one(r))))
    assert min(changed.values()) >= 0.8 * len(rows), changed


def test_inverted_t2_an_event_exactly_at_a_read_time_counts_and_one_a_second_later_does_not(world):
    """Another departure's takeoff put exactly at ``a + s`` adds one to the airport's takeoffs at that read (the windows
    end at the read time, inclusive), at ``a + s + 1`` it does not; the earlier reads never see it."""
    mvt, _ = world
    rows = t1_rows(mvt)[:3]
    dep = _oplib.departures(mvt)
    for r in rows:
        a, apt = anchor(mvt, r), mvt.loc[r, "apt"]
        t = _oplib.secs(dep["MVT_TIME_UTC_mvt"])
        far = dep.index[(dep["apt"] == apt).to_numpy() & (np.abs(t - a) > 2 * 86_400)]
        j = one(far[0])  # another departure, days away: moved here, it is the only change
        base = build(mvt, (REC,), one(r))
        for tag, s in future.SHIFTS.items():
            at = build(_set(mvt, "MVT_TIME_UTC_mvt", j, [a + s]), (REC,), one(r))
            after = build(_set(mvt, "MVT_TIME_UTC_mvt", j, [a + s + 1]), (REC,), one(r))
            col = f"op_h1_frec_{tag}_dep_apt_b15"
            assert at[col].iloc[0] == base[col].iloc[0] + 1 and after[col].iloc[0] == base[col].iloc[0], (r, tag)
            for earlier in [x for x, sx in future.SHIFTS.items() if sx < s]:
                assert_identical(base[outputs(REC, earlier)], at[outputs(REC, earlier)])


def test_inverted_t2_the_adsb_block_reads_the_other_airframes_between_the_reads(aw):
    """The other airframes' events as they stood 15 min before the last query: the f45 columns change for most rows,
    the f15 and f30 ones never (their queries lie before the cut)."""
    w, state = aw
    cand = training_departures(w.mvt).intersection(w.linked_cs)
    cand = cand[w.F.loc[cand, "op_h1_fadsb_f45_taxi_out_now"].notna().to_numpy()]
    rows = np.random.default_rng(23).choice(cand, 10, replace=False)
    changed = 0
    for r in rows:
        a = anchor(w.mvt, r)
        q30 = np.floor(a + future.SHIFTS["f30"]) - LAG
        own = w.table["icao"].astype(str).eq(str(w.info.loc[r, "icao"])).to_numpy()
        state["events"] = w.table
        full = build(w.mvt, (ADSB,), one(r))
        state["events"] = events_table([], like=pd.concat([w.table.loc[own], view_at(w.table.loc[~own], q30)],
                                                          ignore_index=True).astype({c: object for c in EVENT_CATS}))
        cut = build(w.mvt, (ADSB,), one(r))
        state["events"] = w.table
        early = outputs(ADSB, "f15") + outputs(ADSB, "f30")
        assert_identical(full[early], cut[early])
        changed += int(not full[outputs(ADSB, "f45")].equals(cut[outputs(ADSB, "f45")]))
    assert changed >= 0.6 * len(rows), changed


# --- 5. op_h1_fut_plan against brute force -----------------------------------------------------------------------------

EDGES = (0, 1, 900, 901, 1800, 1801, 3600, 3601, -1)
"""Offsets from a query's anchor at which other flights' plans are put: ``a`` and ``a - 1`` are outside every window,
``a + 1`` and each ``a + w`` inside, each ``a + w + 1`` outside that window."""


def plan_world(seed: int = 4):
    """A small record world whose plans sit on the windows' edges, and its query departures.

    For six queries per airport (plans left alone), other departures and arrivals get a plan at ``a + edge`` for every
    `EDGES` offset, sharing the query's runway and stand (so every level sees them); the planned off-block comes from
    the IOBT, the LOBT (IOBT blank) or the schedule (both blank) in turn. Two queries have no stand and two no runway
    (the airport's unknown-stand pier and unknown-runway group, as `_oplib.pier_key` and `_oplib.runway_key` make
    them); three departures have no plan at all (no anchor, never counted).
    """
    mvt, _ = synthetic_world(days=3, per_day=40, seed=seed)
    rng = np.random.default_rng(seed)
    out = mvt.copy()
    dep = out.index[out["is_dep"].to_numpy()]
    arr = out.index[~out["is_dep"].to_numpy()]
    queries = []
    for apt in sorted(out["apt"].astype(str).unique()):
        at = dep[out.loc[dep, "apt"].astype(str).eq(apt).to_numpy()]
        queries += list(rng.choice(at, 6, replace=False))
    queries = pd.Index(queries, name="mvt_id")
    out.loc[queries[[0, 6]], "STAND_mvt"] = np.nan
    out.loc[queries[[1, 7]], "RUNWAY_mvt"] = np.nan
    free_dep = [j for j in rng.permutation(dep) if j not in set(queries)]
    free_arr = list(rng.permutation(arr))
    k = 0
    for q in queries:
        aq = anchor(out, q)
        apt = out.loc[q, "apt"]
        for e in EDGES:
            j = next(x for x in free_dep if out.loc[x, "apt"] == apt)
            free_dep.remove(j)
            source = PLAN_COLUMNS[k % 3]
            out.loc[j, list(PLAN_COLUMNS[: PLAN_COLUMNS.index(source)])] = pd.NaT
            out.loc[[j], source] = pd.to_datetime([aq + e], unit="s", utc=True)
            out.loc[j, "RUNWAY_mvt"] = out.loc[q, "RUNWAY_mvt"]
            out.loc[j, "STAND_mvt"] = out.loc[q, "STAND_mvt"]
            i = next(x for x in free_arr if out.loc[x, "apt"] == apt)
            free_arr.remove(i)
            out.loc[[i], "SCHED_TIME_UTC_mvt"] = pd.to_datetime([aq + e], unit="s", utc=True)
            out.loc[i, "STAND_mvt"] = out.loc[q, "STAND_mvt"]
            k += 1
    unplanned = pd.Index(free_dep[:3], name="mvt_id")
    out.loc[unplanned, list(PLAN_COLUMNS)] = pd.NaT
    return out, queries, unplanned


def _seconds(col: pd.Series) -> np.ndarray:
    since = (col - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    return np.floor(since.to_numpy(dtype="float64", na_value=np.nan))


def brute_plan(mvt: pd.DataFrame) -> pd.DataFrame:
    """`future.PLAN` by a direct loop: for every departure with a plan, other departures planned off (IOBT, else LOBT,
    else schedule) and other arrivals scheduled in ``(a, a + w]``, ``a`` = planned - 15 min, per airport, runway
    (unknown: the airport's ``__NA__`` runway) and pier (the stand's letter prefix; unknown or unparsable: ``""``)."""
    dep, arr = mvt.loc[mvt["is_dep"].to_numpy()], mvt.loc[~mvt["is_dep"].to_numpy()]
    planned = _seconds(dep["IOBT_flt"])
    for col in ("LOBT_flt", "SCHED_TIME_UTC_mvt"):
        planned = np.where(np.isnan(planned), _seconds(dep[col]), planned)
    sched = _seconds(arr["SCHED_TIME_UTC_mvt"])

    def prefix(stand) -> str:
        m = re.match(r"^([A-Za-z]*)\d*[A-Za-z]*$", str(stand)) if pd.notna(stand) else None
        return m.group(1) if m else ""

    def keys(frame: pd.DataFrame) -> dict[str, np.ndarray]:
        apt = frame["apt"].astype(str).to_numpy()
        rwy = np.array(["__NA__" if pd.isna(x) else str(x) for x in frame["RUNWAY_mvt"].astype(object)])
        pier = np.array([prefix(x) for x in frame["STAND_mvt"].astype(object)])
        at = np.char.add(apt, "_")
        return {"apt": apt, "rwy": np.char.add(at, rwy), "pier": np.char.add(at, pier)}

    kd, ka = keys(dep), keys(arr)
    out = np.full((len(dep), len(future.PLAN)), np.nan)
    for i in range(len(dep)):
        a = planned[i] - 900
        if np.isnan(a):
            continue
        others = np.arange(len(dep)) != i
        row = {}
        for tag, w in future.PLAN_WINDOWS.items():
            d_in = others & (planned > a) & (planned <= a + w)
            a_in = (sched > a) & (sched <= a + w)
            for lvl in ("apt", "rwy", "pier"):
                row[f"dep_plan_{lvl}_{tag}"] = int((d_in & (kd[lvl] == kd[lvl][i])).sum())
            for lvl in ("apt", "pier"):
                row[f"arr_sched_{lvl}_{tag}"] = int((a_in & (ka[lvl] == kd[lvl][i])).sum())
        out[i] = [row[c] for c in future.PLAN]
    return pd.DataFrame(out, index=dep.index, columns=[f"op_h1_fplan_{c}" for c in future.PLAN])


def test_the_plan_world_puts_plans_on_every_edge():
    mvt, queries, unplanned = plan_world()
    p = plans(mvt)
    dep = mvt["is_dep"].to_numpy()
    for q in queries:
        a, apt = anchor(mvt, q), mvt.loc[q, "apt"]
        here = (mvt["apt"] == apt).to_numpy()
        for e in EDGES:
            assert ((p == a + e) & here & dep).sum() >= 1 and ((p == a + e) & here & ~dep).sum() >= 1, (q, e)
    assert np.isnan(_oplib.anchor_h1(mvt.loc[unplanned])).all()
    for col in PLAN_COLUMNS:  # each plan source is exercised
        others = mvt.loc[dep & ~mvt.index.isin(queries)]
        assert (others[col].notna() & others[list(PLAN_COLUMNS[: PLAN_COLUMNS.index(col)])].isna().all(axis=1)).sum() >= 10


def test_brute_force_plan_counts_match_on_the_edges():
    mvt, queries, unplanned = plan_world()
    dep = mvt.index[mvt["is_dep"].to_numpy()]
    got = build(mvt, (PLAN,), dep)
    want = brute_plan(mvt)
    np.testing.assert_array_equal(got.to_numpy(dtype="float64"), want.loc[dep].to_numpy())
    assert got.loc[unplanned].isna().all().all()
    # the edges reach the queries: with the own plan (always a + 15 min) excluded, the edge plans alone are counted
    q = got.loc[queries]
    assert (q["op_h1_fplan_dep_plan_rwy_w15"] >= 2).all() and (q["op_h1_fplan_arr_sched_pier_w60"] >= 4).all()


def test_brute_force_plan_counts_match_on_the_record_world(world):
    mvt, _ = world
    day = mvt.loc[(mvt["MVT_TIME_UTC_mvt"] < mvt["MVT_TIME_UTC_mvt"].min() + pd.Timedelta(days=3)).to_numpy()]
    dep = day.index[day["is_dep"].to_numpy()]
    np.testing.assert_array_equal(build(day, (PLAN,), dep).to_numpy(dtype="float64"), brute_plan(day).loc[dep].to_numpy())


# --- 6. the own airframe's ground stay -------------------------------------------------------------------------------

T = 1_742_500_000.0  # a whole second in March 2025


def flights(rows: list[tuple]) -> pd.DataFrame:
    """A movement frame of departures, ``(mvt_id, apt, recorded takeoff, callsign)``: what the stay rule reads."""
    idx = pd.Index([r[0] for r in rows], name="mvt_id")
    t = pd.to_datetime([r[2] for r in rows], unit="s", utc=True)
    return pd.DataFrame({"apt": pd.Categorical([r[1] for r in rows]), "is_dep": True, "MVT_TIME_UTC_mvt": t,
                         "SCHED_TIME_UTC_mvt": t - pd.Timedelta(minutes=30),
                         "STAND_mvt": pd.Categorical(["A1"] * len(rows)), "RUNWAY_mvt": pd.Categorical(["25C"] * len(rows)),
                         "CALLSIGN_flt": pd.Categorical([r[3] for r in rows])}, index=idx)


def stay_events(inbound: bool = True, previous: bool = True) -> pd.DataFrame:
    """Departure 11 (EDDF, recorded takeoff ``T``): its airframe ``aaa001`` landed at ``T - 6000`` as DLH123
    (``inbound``), pushed, taxied from a first sighting and held as DLH123, and took off as DLH400 at ``T - 10``; a
    spurious landing during the line-up (``T - 300``); a point exactly at the stay's end and one a second later. Its
    previous rotation at EDDF (``previous``): landed ``T - 20000``, took off as DLH399 at ``T - 14000``. The same
    airframe at LSZH during the stay, and another airframe at EDDF under the same inbound callsign. ``tag`` names each."""
    tko, pad, own = T - 10.0, future.STAY_PAD_S, "aaa001"
    rows = [event("EDDF", "TAXI_IN", own, "DLH123", T - 5990, tag="own_taxi_in"),
            event("EDDF", "PUSH", own, "DLH123", T - 1500, tag="own_push"),
            event("EDDF", "TAXI_OUT", own, "DLH123", T - 1200, tag="own_taxi_out"),
            event("EDDF", "LANDING", own, "DLH123", T - 300, tag="own_spurious_landing"),
            event("EDDF", "RWY_HOLD", own, "DLH123", T - 200, tag="own_hold"),
            event("EDDF", "TAKEOFF", own, "DLH400", tko, tag="own_takeoff"),
            event("EDDF", "REMOTE_STOP", own, "DLH400", tko + pad, tag="stay_end"),
            event("EDDF", "REMOTE_STOP", own, "DLH400", tko + pad + 1, tag="after_stay_end"),
            event("EDDF", "REMOTE_STOP", own, "DLH399", tko - future.STAY_MAX_S, tag="eight_hours"),
            event("EDDF", "REMOTE_STOP", own, "DLH399", tko - future.STAY_MAX_S - 1, tag="eight_hours_and_1_s"),
            event("LSZH", "TAXI_OUT", own, "DLH123", T - 1000, tag="other_airport"),
            event("EDDF", "TAXI_OUT", "bbb002", "DLH123", T - 1100, tag="other_airframe"),
            event("EDDF", "TAKEOFF", "bbb002", "DLH123", T + 900, tag="other_airframe_takeoff")]
    if inbound:
        rows.append(event("EDDF", "LANDING", own, "DLH123", T - 6000, tag="own_inbound_landing"))
    if previous:
        rows += [event("EDDF", "LANDING", own, "DLH398", T - 20000, tag="prev_landing"),
                 event("EDDF", "TAXI_OUT", own, "DLH399", T - 14600, tag="prev_taxi"),
                 event("EDDF", "TAKEOFF", own, "DLH399", T - 14000, tag="prev_takeoff"),
                 event("EDDF", "REMOTE_STOP", own, "DLH399", T - 13999, tag="prev_takeoff_and_1_s")]
    return events_table(rows)


@pytest.mark.parametrize("callsign", ["DLH 400", None])
@pytest.mark.parametrize("history", ["inbound", "no_inbound", "neither"])
def test_own_stay_relabels_exactly_the_own_stay(history, callsign):
    """The stay starts at the last landing (inclusive) at the airport read 10 min before the roll, else just after the
    previous takeoff there, else 8 h before the roll; it ends 300 s after the roll. Exactly its events carry the
    departure's own normalised callsign (``DLH 400`` -> ``DLH400``, linked by the callsign rule), or, for a departure
    without a callsign (linked by time), the token ``ZZOWN11``, which the departure's query then carries too -- not the
    previous rotation, not the airframe's events at another airport, not another airframe under the same inbound
    callsign; nothing else changes. A filed callsign stays as filed in the slim frame."""
    mvt = flights([(11, "EDDF", T, callsign), (12, "LSZH", T + 50_000, "SWR9")])
    ev = stay_events(inbound=history == "inbound", previous=history != "neither")
    tko = T - 10.0
    start = {"inbound": T - 6000, "no_inbound": T - 14000 + 1, "neither": tko - future.STAY_MAX_S}[history]
    label = "DLH400" if callsign else f"{future.OWN_TOKEN}11"
    stays = future.own_airframe_stays(ev, mvt)
    assert list(stays.index) == [11] and list(stays.columns) == ["apt", "icao", "cs", "stay_start", "stay_end"]
    assert _plain(stays.loc[11].tolist()) == ["EDDF", "aaa001", "DLH400" if callsign else None, start,
                                              tko + future.STAY_PAD_S]
    slim, out, linked = future.without_own_airframes(mvt, mvt.index, ev)
    assert list(linked) == [11]
    assert slim.loc[11, "CALLSIGN_flt"] == (callsign or label) and slim.loc[12, "CALLSIGN_flt"] == "SWR9"
    assert list(slim.columns) == list(future.SURFACE_COLUMNS)
    t = ev["t_start"].to_numpy()
    in_stay = (ev["apt"].astype(str).eq("EDDF") & ev["icao"].astype(str).eq("aaa001")).to_numpy() & (t >= start) & (
        t <= tko + future.STAY_PAD_S)
    got = out["cs"].astype(object).to_numpy()
    assert (got[in_stay] == label).all()
    np.testing.assert_array_equal(got[~in_stay], ev["cs"].astype(object).to_numpy()[~in_stay])
    assert_identical(out.drop(columns="cs"), ev.drop(columns="cs"))  # only the callsigns are relabelled
    tags = set(ev.loc[in_stay, "tag"])
    assert {"own_taxi_in", "own_push", "own_taxi_out", "own_spurious_landing", "own_hold", "own_takeoff",
            "stay_end"} <= tags
    assert not tags & {"after_stay_end", "eight_hours_and_1_s", "other_airport", "other_airframe",
                       "other_airframe_takeoff", "prev_landing", "prev_taxi", "prev_takeoff"}
    assert ("own_inbound_landing" in tags) == (history == "inbound")
    assert ("prev_takeoff_and_1_s" in tags) == (history == "no_inbound")
    assert ("eight_hours" in tags) == (history == "neither")
    assert _adsblib.norm_callsign([label]).tolist() == [label], "the label survives the callsign normalisation"


def test_own_stay_link_rules():
    """The own takeoff: the flight's callsign at the airport within 600 s (nearest), even with another takeoff closer in
    time; else the ONLY free takeoff at the airport within 45 s (never one the callsign rule claimed), provided no other
    unlinked departure took off within 45 s of it (mutual uniqueness); a takeoff two departures claim by callsign goes
    to the nearer one, the other gets no stay; a takeoff without an airframe or an airport never links. Only a linked
    departure without a callsign carries a token; the stay of a linked departure with one carries its callsign."""
    B = [T + 40_000.0 * k for k in range(12)]
    mvt = flights([(1, "EDDF", B[0], "DLH1"), (2, "EDDF", B[1], "DLH2"), (3, "EDDF", B[2], "DLH3"),
                   (4, "EDDF", B[3], None), (5, "EDDF", B[4], "DLH5"), (6, "EDDF", B[4] + 20, None),
                   (7, "EDDF", B[5], None), (8, "EDDF", B[5] + 30, None), (9, "LSZH", B[6], "DLH9"),
                   (10, "EDDF", B[7], "dlh 10"), (11, "EDDF", B[8], None), (12, "EDDF", B[9], "DLH12"),
                   (13, "EDDF", B[9] + 300, "DLH12"), (14, "EDDF", B[10], "DLH14"), (15, "EDDF", B[11], "DLH15")])
    ev = events_table([
        event("EDDF", "TAKEOFF", "c1", "DLH1", B[0] + 590), event("EDDF", "TAKEOFF", "x1", "ZZZ1", B[0] + 5),
        event("EDDF", "TAKEOFF", "c2", "DLH2", B[1] + 610),
        event("EDDF", "TAKEOFF", "c3", "INB3", B[2] - 12),
        event("EDDF", "TAKEOFF", "c4", "INB4", B[3] + 10), event("EDDF", "TAKEOFF", "x4", "ZZZ4", B[3] - 30),
        event("EDDF", "TAKEOFF", "c5", "DLH5", B[4] + 5),
        event("EDDF", "TAKEOFF", "c7", "QQQ7", B[5] + 20),
        event("EDDF", "TAKEOFF", "c9", "DLH9", B[6] + 5),
        event("EDDF", "TAKEOFF", "c10", "DLH10", B[7] + 100),
        event("EDDF", "TAKEOFF", "c11", "QQQ11", B[8] + 30),
        event("EDDF", "TAKEOFF", "c12", "DLH12", B[9] + 100),
        event("EDDF", "TAKEOFF", np.nan, "DLH14", B[10] + 5),
        event(np.nan, "TAKEOFF", "c15", "DLH15", B[11] + 5),
    ])
    stays = future.own_airframe_stays(ev, mvt)
    # 2: callsign 610 s off, nothing free within 45 s; 4: two free takeoffs within 45 s; 6: the only takeoff near it
    # was claimed by 5's callsign; 7 and 8: one free takeoff within 45 s of both; 9: the callsign's takeoff is at
    # another airport; 13: 12 is nearer to their callsign's takeoff; 14 / 15: that takeoff has no airframe / airport
    assert stays["icao"].to_dict() == {1: "c1", 3: "c3", 5: "c5", 10: "c10", 11: "c11", 12: "c12"}
    assert dict(zip(stays.index, _plain(stays["cs"]))) == {1: "DLH1", 3: "DLH3", 5: "DLH5", 10: "DLH10", 11: None,
                                                            12: "DLH12"}
    slim, out, linked = future.without_own_airframes(mvt, mvt.index, ev)
    assert set(linked) == {1, 3, 5, 10, 11, 12}
    token = slim["CALLSIGN_flt"].astype("string").str.startswith(future.OWN_TOKEN).fillna(False)
    assert list(slim.index[token.to_numpy(dtype=bool)]) == [11]
    assert slim.loc[11, "CALLSIGN_flt"] == f"{future.OWN_TOKEN}11"
    keep = mvt.index.difference([11])
    np.testing.assert_array_equal(slim.loc[keep, "CALLSIGN_flt"].astype(object).fillna("-").to_numpy(),
                                  mvt.loc[keep, "CALLSIGN_flt"].astype(object).fillna("-").to_numpy())
    cs_of = dict(zip(out["icao"].astype(object), out["cs"].astype(object)))
    assert cs_of["c3"] == "DLH3" and cs_of["c11"] == f"{future.OWN_TOKEN}11", "a stay carries its departure's label"
    assert (cs_of["c4"], cs_of["c7"], cs_of["x1"]) == ("INB4", "QQQ7", "ZZZ1"), "an unlinked airframe keeps its own"


@pytest.mark.parametrize("rival", [0.0, 30.0, -35.0, 45.0, 46.0, 120.0])
def test_own_stay_mutual_uniqueness_of_the_time_rule(rival):
    """Two departures without a callsign; one free ADS-B takeoff 10 s after the first's recorded takeoff. The second
    departure took off ``rival`` s after that ADS-B takeoff: within 45 s (inclusive) it is a rival and neither links;
    farther, the first links alone. A rival that has a callsign the callsign rule cannot use counts alike."""
    t_tko = T + 10.0
    for second_cs in (None, "NOMATCH7"):
        mvt = flights([(1, "EDDF", T, None), (2, "EDDF", t_tko + rival, second_cs)])
        ev = events_table([event("EDDF", "TAKEOFF", "c1", "INB1", t_tko)])
        got = future.own_airframe_stays(ev, mvt)["icao"].to_dict()
        assert got == ({} if abs(rival) <= future.LINK_TIME_ONLY_S else {1: "c1"}), (rival, second_cs)
        assert future.own_airframe_stays(ev, mvt.loc[[1]])["icao"].to_dict() == {1: "c1"}, "alone, it links"


@pytest.mark.parametrize("missing", ["query_callsigns", "takeoff_callsigns"])
def test_own_stay_links_by_time_without_any_callsign(missing):
    """No callsign on one side at all (pandas 3 merge keys of one dtype): the time rule alone links, and a departure
    with a callsign keeps it as its label. A query set of one callsign-less departure is a legal input (a partial
    build); the blocks are built on every departure (`prc.data.dep_index`)."""
    mvt = flights([(1, "EDDF", T, None), (2, "EDDF", T + 50_000, "DLH2")])
    cs = (None, None) if missing == "takeoff_callsigns" else ("INB1", "DLH2")
    ev = events_table([event("EDDF", "TAKEOFF", "c1", cs[0], T - 10), event("EDDF", "TAKEOFF", "c2", cs[1], T + 49_990)])
    q = mvt.loc[[1]] if missing == "query_callsigns" else mvt
    want = {1: "c1"} if missing == "query_callsigns" else {1: "c1", 2: "c2"}
    stays = future.own_airframe_stays(ev, q)
    assert stays["icao"].to_dict() == want
    assert dict(zip(stays.index, _plain(stays["cs"]))) == {k: (None if k == 1 else "DLH2") for k in want}
    slim, out, linked = future.without_own_airframes(mvt, q.index, ev)
    assert set(linked) == set(want) and slim.loc[1, "CALLSIGN_flt"] == f"{future.OWN_TOKEN}1"
    cs_of = dict(zip(out["icao"].astype(object), out["cs"].astype(object)))
    assert cs_of["c1"] == f"{future.OWN_TOKEN}1" and cs_of["c2"] == "DLH2"


def test_own_stay_no_link_at_all_returns_empty_frames():
    """No departure links (no takeoff near any, or no takeoff at all): empty stays of the documented columns, the slim
    frame's callsigns and the event table unchanged, nothing linked."""
    mvt = flights([(1, "EDDF", T, None), (2, "EDDF", T + 50_000, "DLH2")])
    for ev in (events_table([event("EDDF", "TAKEOFF", "c9", "XYZ9", T + 9_000)]),
               events_table([event("EDDF", "LANDING", "c9", "XYZ9", T - 9_000)])):
        stays = future.own_airframe_stays(ev, mvt)
        assert stays.empty and list(stays.columns) == ["apt", "icao", "cs", "stay_start", "stay_end"]
        slim, out, linked = future.without_own_airframes(mvt, mvt.index, ev)
        assert len(linked) == 0
        assert_identical(out, ev)
        np.testing.assert_array_equal(slim["CALLSIGN_flt"].astype(object).fillna("-").to_numpy(),
                                      mvt["CALLSIGN_flt"].astype(object).fillna("-").to_numpy())


def test_own_stay_the_adsb_block_on_departures_without_callsigns(aw):
    """op_h1_fut_adsb on the ADS-B world's departures that carry no callsign (NM-unmatched): the time rule links most
    of them, each to its own airframe, their queries carry their tokens, and the rows are measured."""
    w, _ = aw
    dep = training_departures(w.mvt)
    rows = dep[w.mvt.loc[dep, "CALLSIGN_flt"].isna().to_numpy()]
    assert len(rows) >= 10
    stays = future.own_airframe_stays(w.table, w.mvt.loc[rows])
    assert len(stays) >= 0.8 * len(rows)
    assert (stays["icao"].astype(str) == w.info.loc[stays.index, "icao"].astype(str)).all() and stays["cs"].isna().all()
    slim, _, linked = future.without_own_airframes(w.mvt, rows, w.table)
    assert set(linked) == set(stays.index)
    want = [f"{future.OWN_TOKEN}{i}" for i in stays.index]
    assert slim.loc[stays.index, "CALLSIGN_flt"].astype(object).tolist() == want
    X = build(w.mvt, (ADSB,), rows)
    assert X.notna().any(axis=1).mean() > 0.5
    assert X.loc[rows.difference(stays.index)].isna().all(axis=None), "not identified: no picture"


@pytest.mark.parametrize("callsigns", [("DLH21", "DLH22"), (None, None)])
def test_own_stays_of_one_airframe_on_two_rotations_carry_their_own_labels(callsigns):
    """Two departures of one airframe a few hours apart: each stay carries its own departure's label (its callsign, or
    its token without one); the second stay starts at the airframe's landing between them."""
    t1, t2 = T, T + 12_000.0
    mvt = flights([(21, "EDDF", t1, callsigns[0]), (22, "EDDF", t2, callsigns[1])])
    ev = events_table([event("EDDF", kind, "ddd004", cs, t) for kind, cs, t in (
        ("LANDING", "DLH20", t1 - 5000), ("TAXI_OUT", "DLH20", t1 - 900), ("TAKEOFF", "DLH21", t1 - 10),
        ("LANDING", "DLH21", t2 - 6000), ("TAXI_OUT", "DLH21", t2 - 800), ("TAKEOFF", "DLH22", t2 - 10))])
    _, out, linked = future.without_own_airframes(mvt, mvt.index, ev)
    assert set(linked) == {21, 22}
    first, second = (callsigns[0] or f"{future.OWN_TOKEN}21"), (callsigns[1] or f"{future.OWN_TOKEN}22")
    want = np.where(ev["t_start"].to_numpy() < t2 - 6000, first, second)
    np.testing.assert_array_equal(out["cs"].astype(object).to_numpy(), want)


def _crafted_stay(w, r) -> dict:
    """Row ``r``'s airframe re-flown by hand in the ADS-B world: landed 90 min before its recorded takeoff (inbound
    callsign), pushed at its stand 10 min before the f30 query, then -- after a reception gap -- a TAXI_OUT from a
    first sighting away from the stand under the INBOUND callsign, in progress at the f30 query, a runway hold (inbound
    callsign) in progress then too, and its ADS-B takeoff under the flight's callsign 10 s before the recorded one. Its
    previous rotation at the same airport hours before (another callsign), and the airframe at the other airport during
    the stay. The world's own events of the airframe are replaced."""
    q = w.mvt.loc[[r]]
    i = w.info.loc[r]
    apt, icao, rwy = str(q["apt"].iloc[0]), str(i["icao"]), str(i["rwy"])
    a = anchor(w.mvt, r)
    t_rec = float(_oplib.secs(q["MVT_TIME_UTC_mvt"])[0])
    q30 = float(np.floor(a + future.SHIFTS["f30"]) - LAG)
    tko, land = t_rec - 10.0, t_rec - 5400.0
    stand = float(S.stand_id(apt, int(i["stand_i"])))
    other = next(x for x in S.APTS if x != apt)
    own = {
        "LANDING": event(apt, "LANDING", icao, i["cs_in"], land, 0.0, land, land, rwy_end=rwy),
        "TAXI_IN": event(apt, "TAXI_IN", icao, i["cs_in"], land, 20.0, land + 420, land + 440),
        "PUSH": event(apt, "PUSH", icao, i["cs_in"], q30 - 600, 15.0, origin_kind="push", origin_stand=stand),
        "TAXI_OUT": event(apt, "TAXI_OUT", icao, i["cs_in"], q30 - 300, 20.0, tko, tko + 20, origin_kind="first_seen"),
        "RWY_HOLD": event(apt, "RWY_HOLD", icao, i["cs_in"], q30 - 120, 30.0, tko - 60, tko - 50, rwy_end=rwy,
                          dur=tko - 60 - (q30 - 120)),
        "TAKEOFF": event(apt, "TAKEOFF", icao, _adsblib.norm_callsign([i["cs_out"]])[0], tko, 25.0, tko + 15, tko + 25,
                         rwy_end=rwy, origin_kind="first_seen", taxi_seen=1, dur=tko - (q30 - 300), stop_s=0.0,
                         path_m=3000.0),
    }
    rest = [event(apt, "LANDING", icao, "PRV0", tko - 20_000, rwy_end=rwy),
            event(apt, "PUSH", icao, "PRV1", tko - 15_000, 15.0, origin_kind="push", origin_stand=stand),
            event(apt, "TAXI_OUT", icao, "PRV1", tko - 14_900, 20.0, tko - 14_000, tko - 13_980, origin_kind="push",
                  origin_stand=stand),
            event(apt, "TAKEOFF", icao, "PRV1", tko - 14_000, 25.0, tko - 13_985, tko - 13_975, rwy_end=rwy),
            event(other, "TAXI_OUT", icao, i["cs_in"], q30 - 200, 20.0, q30 + 100, q30 + 120, origin_kind="first_seen")]
    world = w.table.loc[~w.table["icao"].astype(str).eq(icao)]
    return {"r": r, "apt": apt, "icao": icao, "a": a, "t_rec": t_rec, "q30": q30, "tko": tko, "land": land,
            "cs": _adsblib.norm_callsign(q["CALLSIGN_flt"].to_numpy(dtype=object))[0],
            "own": own, "rest": rest, "world": world, "mvt": w.mvt,
            "table": lambda drop=(), **swap: events_table(
                [swap.get(k, v) for k, v in own.items() if k not in drop] + rest, like=world)}


@pytest.fixture(scope="module")
def stay_case(adsb_world):
    """The first covered, callsign-linked training departure of the ADS-B world that fits `_crafted_stay`: its recorded
    takeoff 400-1500 s after the f30 query, no other ADS-B takeoff at its airport within 90 s of it, and its taxi and
    hold families gated in at the f30 read with and without the stay rule."""
    w = adsb_world
    tko = w.table.loc[w.table["kind"].astype(str).eq("TAKEOFF")]
    day_two = float(w.pts["t"].min()) + 86_400
    cols = ["taxi_out_now", "rwy_hold_now"]
    with pytest.MonkeyPatch.context() as mp:
        state = _patch(mp, w)
        for r in training_departures(w.mvt).intersection(w.linked_cs):
            q = w.mvt.loc[[r]]
            if q[["STAND_mvt", "RUNWAY_mvt"]].isna().any(axis=None) or anchor(w.mvt, r) < day_two:
                continue
            t_rec = float(_oplib.secs(q["MVT_TIME_UTC_mvt"])[0])
            q30 = np.floor(anchor(w.mvt, r) + future.SHIFTS["f30"]) - LAG
            others = tko.loc[tko["apt"].astype(str).eq(str(q["apt"].iloc[0]))
                             & tko["icao"].astype(str).ne(str(w.info.loc[r, "icao"]))]
            if not 400 <= t_rec - q30 <= 1500 or (others["t_start"] - t_rec).abs().le(90).any():
                continue
            case = _crafted_stay(w, r)
            state["events"] = case["table"]()
            X = build(w.mvt, (ADSB,), one(r))
            plain = _adsblib.surface_at(w.mvt, one(r), np.array([case["a"] + future.SHIFTS["f30"]]),
                                        events=state["events"])
            if X[[f"op_h1_fadsb_f30_{c}" for c in cols]].notna().all(axis=None) and plain[cols].notna().all(axis=None):
                return case
    pytest.fail("no departure of the ADS-B world fits the own-stay scenario")


def _in_stay(c, table: pd.DataFrame) -> np.ndarray:
    t = table["t_start"].to_numpy()
    return (table["apt"].astype(str).eq(c["apt"]) & table["icao"].astype(str).eq(c["icao"])).to_numpy() & (
        t >= c["land"]) & (t <= c["tko"] + future.STAY_PAD_S)


def test_own_stay_in_the_world_relabels_exactly_the_own_stay(stay_case):
    """The hand-made stay in the ADS-B world: its six events carry the flight's own normalised callsign; the previous
    rotation, the airframe at the other airport and every other airframe keep theirs; the slim frame keeps every
    filed callsign (the departure has one, so its query's callsign rule matches the relabelled stay)."""
    c = stay_case
    mvt, table = c["mvt"], c["table"]()
    stays = future.own_airframe_stays(table, mvt.loc[one(c["r"])])
    assert stays.loc[c["r"]].tolist() == [c["apt"], c["icao"], c["cs"], c["land"], c["tko"] + future.STAY_PAD_S]
    slim, out, linked = future.without_own_airframes(mvt, one(c["r"]), table)
    assert list(linked) == [c["r"]]
    in_stay = _in_stay(c, table)
    assert in_stay.sum() == len(c["own"]), "the stay is the six hand-made events"
    assert (table.loc[~in_stay, "icao"].astype(str) == c["icao"]).sum() == len(c["rest"])
    got = out["cs"].astype(object).to_numpy()
    assert (got[in_stay] == c["cs"]).all()
    np.testing.assert_array_equal(got[~in_stay], table["cs"].astype(object).to_numpy()[~in_stay])
    np.testing.assert_array_equal(slim["CALLSIGN_flt"].astype(object).fillna("-").to_numpy(),
                                  mvt["CALLSIGN_flt"].astype(object).fillna("-").to_numpy())


def test_own_stay_the_own_taxi_under_its_inbound_callsign_is_not_counted(stay_case, aw):
    """At the f30 read the own airframe taxis (from a first sighting away from its stand, under its inbound callsign)
    and holds: the block equals the block without those events, at every read time; the H1 rules alone (callsign and
    own stand, no stay) count both."""
    w, state = aw
    c = stay_case
    r = one(c["r"])
    state["events"] = c["table"]()
    X = build(w.mvt, (ADSB,), r)
    state["events"] = c["table"](drop=("PUSH", "TAXI_OUT", "RWY_HOLD"))
    assert_identical(X, build(w.mvt, (ADSB,), r))
    plain = _adsblib.surface_at(w.mvt, r, np.array([c["a"] + future.SHIFTS["f30"]]), events=c["table"]())
    for col in ("taxi_out_now", "rwy_hold_now"):
        assert plain[col].iloc[0] == X[f"op_h1_fadsb_f30_{col}"].iloc[0] + 1, col
    assert np.isfinite(X.to_numpy(dtype="float64")).mean() > 0.5


def test_own_stay_both_rules_act_the_stay_and_the_flights_callsign(stay_case, aw):
    """After the fix both rules act at once: the own taxi under the INBOUND callsign is not counted (the stay rule:
    the stay carries the flight's callsign), and neither is a taxi of ANOTHER airframe under the flight's own
    callsign, in progress at the f30 read (the callsign rule, as at H1). Under a foreign callsign that taxi counts."""
    w, state = aw
    c = stay_case
    r = one(c["r"])
    state["events"] = c["table"]()
    X = build(w.mvt, (ADSB,), r)
    state["events"] = c["table"](drop=("PUSH", "TAXI_OUT", "RWY_HOLD"))
    assert_identical(X, build(w.mvt, (ADSB,), r))  # the stay rule
    intruder = event(c["apt"], "TAXI_OUT", "fff003", c["cs"], c["q30"] - 250, 20.0, c["tko"] + 40, c["tko"] + 60,
                     origin_kind="first_seen")
    assert intruder["t_known_start"] <= c["q30"] < c["tko"] + 60, "in progress at the f30 query"
    col = "op_h1_fadsb_f30_taxi_out_now"
    for cs, plus in ((c["cs"], 0.0), ("XYZ987", 1.0)):
        state["events"] = events_table([dict(intruder, cs=cs)], like=c["table"]())
        got = build(w.mvt, (ADSB,), r)
        if plus:
            assert got[col].iloc[0] == X[col].iloc[0] + 1, "a foreign taxi in progress counts"
        else:
            assert_identical(X, got)  # the callsign rule
    state["events"] = w.table


def test_own_stay_a_departure_without_a_callsign_carries_the_token_and_its_taxi_is_not_counted(stay_case, aw):
    """The stay case with the flight's callsign removed (NM-unmatched): the time rule links its own airframe, the stay
    and the query carry ``ZZOWN<mvt_id>``, and the own taxi under the inbound callsign and the own hold are not counted;
    the H1 rules alone (no callsign, the own stand) count both."""
    w, state = aw
    c = stay_case
    r = one(c["r"])
    mv = w.mvt.copy()
    mv.loc[r, "CALLSIGN_flt"] = np.nan
    table = c["table"]()
    stays = future.own_airframe_stays(table, mv.loc[r])
    assert stays.loc[c["r"], "icao"] == c["icao"] and pd.isna(stays.loc[c["r"], "cs"])
    slim, out, linked = future.without_own_airframes(mv, r, table)
    token = f"{future.OWN_TOKEN}{c['r']}"
    assert list(linked) == [c["r"]] and slim.loc[c["r"], "CALLSIGN_flt"] == token
    in_stay = _in_stay(c, table)
    got = out["cs"].astype(object).to_numpy()
    assert in_stay.sum() == len(c["own"]) and (got[in_stay] == token).all()
    np.testing.assert_array_equal(got[~in_stay], table["cs"].astype(object).to_numpy()[~in_stay])
    state["events"] = table
    X = build(mv, (ADSB,), r)
    assert np.isfinite(X.to_numpy(dtype="float64")).mean() > 0.5
    state["events"] = c["table"](drop=("PUSH", "TAXI_OUT", "RWY_HOLD"))
    assert_identical(X, build(mv, (ADSB,), r))
    plain = _adsblib.surface_at(mv, r, np.array([c["a"] + future.SHIFTS["f30"]]), events=table)
    for col in ("taxi_out_now", "rwy_hold_now"):
        assert plain[col].iloc[0] == X[f"op_h1_fadsb_f30_{col}"].iloc[0] + 1, col
    state["events"] = w.table


def test_own_stay_two_unlinked_departures_near_one_free_takeoff_link_neither(stay_case, aw):
    """The stay case's own ADS-B takeoff under the inbound callsign (only the time rule can link it), and another
    departure of the airport -- days away, unlinked there -- moved to 20 s after, or 35 s before, the recorded takeoff:
    two unlinked departures within 45 s of one free takeoff, so neither links and both rows are NaN. Moved 120 s after,
    it is no rival and the flight links its own airframe again."""
    w, state = aw
    c = stay_case
    rid = c["r"]
    table = c["table"](TAKEOFF=dict(c["own"]["TAKEOFF"], cs=w.info.loc[rid, "cs_in"]))
    state["events"] = table
    dep = training_departures(w.mvt)
    t = _oplib.secs(w.mvt.loc[dep, "MVT_TIME_UTC_mvt"])
    same = w.mvt.loc[dep, "apt"].astype(str).eq(c["apt"]).to_numpy()
    j = dep[same & (np.abs(t - c["t_rec"]) > 2 * 86_400)][0]
    both = pd.Index([rid, j], name="mvt_id")
    for gap, links in ((20.0, False), (-35.0, False), (120.0, True)):
        mv = _set(w.mvt, "MVT_TIME_UTC_mvt", one(j), [c["t_rec"] + gap])
        stays = future.own_airframe_stays(table, mv.loc[both])
        X = build(mv, (ADSB,), both)
        if links:
            assert stays.loc[rid, "icao"] == c["icao"] and X.loc[rid].notna().any(), gap
        else:
            assert rid not in stays.index and j not in stays.index, gap
            assert X.isna().all(axis=None), gap
    state["events"] = w.table


def test_own_stay_no_own_adsb_takeoff_is_an_all_nan_row(stay_case, aw):
    """No own ADS-B takeoff (and no other within 45 s): the own airframe cannot be told apart, the row is all NaN. The
    time-only rule finds a takeoff whose callsign is still the inbound one, unless a second takeoff is within 45 s.
    Which rows are NaN therefore depends on the identification (the own takeoff's callsign; for the time rule, other
    takeoffs near the own one): a missingness path, not an input, which `model.block_dropout` dilutes."""
    w, state = aw
    c = stay_case
    r = one(c["r"])
    state["events"] = c["table"]()
    X = build(w.mvt, (ADSB,), r)
    state["events"] = c["table"](drop=("TAKEOFF",))
    assert build(w.mvt, (ADSB,), r).isna().all(axis=None)
    assert build(w.mvt, (ADSB,), r, {ADSB: {"shift_days": 1}}).isna().all(axis=None), "the placebo: the same NaN row"
    inbound = dict(c["own"]["TAKEOFF"], cs=w.info.loc[c["r"], "cs_in"])
    state["events"] = c["table"](TAKEOFF=inbound)
    assert_identical(X, build(w.mvt, (ADSB,), r))
    second = event(c["apt"], "TAKEOFF", "fff001", "XXX1", c["t_rec"] + 20, 25.0, rwy_end=c["own"]["TAKEOFF"]["rwy_end"])
    state["events"] = events_table([second], like=c["table"](TAKEOFF=inbound))
    assert build(w.mvt, (ADSB,), r).isna().all(axis=None)
    state["events"] = w.table


def test_own_stay_the_flights_callsign_still_excludes_after_a_time_only_link(stay_case, aw):
    """The own ADS-B takeoff not received, another airframe's takeoff (no movement record, so no rival) alone within
    45 s of the recorded one: the time rule links that airframe. The own taxi, under the flight's OWN callsign, is in
    progress at the f30 read: the query keeps its filed callsign, so the callsign rule leaves the taxi out, as at H1."""
    w, state = aw
    c = stay_case
    r = one(c["r"])
    other = event(c["apt"], "TAKEOFF", "fff001", "XXX1", c["t_rec"] + 20, 25.0, rwy_end=c["own"]["TAKEOFF"]["rwy_end"])
    taxi = dict(c["own"]["TAXI_OUT"], cs=c["cs"])
    tables = [events_table([other], like=c["table"](drop=drop, TAXI_OUT=taxi))
              for drop in (("TAKEOFF", "RWY_HOLD"), ("TAKEOFF", "RWY_HOLD", "TAXI_OUT"))]
    assert future.own_airframe_stays(tables[0], w.mvt.loc[r])["icao"].tolist() == ["fff001"]
    state["events"] = tables[0]
    X = build(w.mvt, (ADSB,), r)
    assert X.notna().any(axis=None), "linked (to the wrong airframe): the row is measured"
    state["events"] = tables[1]
    assert_identical(X, build(w.mvt, (ADSB,), r))
    state["events"] = w.table


WRONG_LINK = ("operational_future.own_airframe_stays' time-only rule cannot tell the own airframe from an aircraft "
              "without a movement record (no rival departure): with the own ADS-B takeoff not received and such a "
              "takeoff alone within 45 s of the recorded one, it links that aircraft, the row is not NaN, and the own "
              "taxi under its INBOUND callsign from a first sighting away from the stand (and the own hold) is counted "
              "at the reads after the anchor")


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=WRONG_LINK)
def test_own_stay_a_wrong_time_only_link_leaves_no_own_taxi_counted(stay_case, aw):
    """The own ADS-B takeoff not received; an aircraft without a movement record takes off alone within 45 s of the
    recorded takeoff. The own taxi and hold, under the INBOUND callsign, are in progress at the f30 read: they must not
    be counted -- the row NaN, or the block as without them."""
    w, state = aw
    c = stay_case
    r = one(c["r"])
    other = event(c["apt"], "TAKEOFF", "fff001", "XXX1", c["t_rec"] + 20, 25.0, rwy_end=c["own"]["TAKEOFF"]["rwy_end"])
    table, without = (events_table([other], like=c["table"](drop=drop))
                      for drop in (("TAKEOFF",), ("TAKEOFF", "PUSH", "TAXI_OUT", "RWY_HOLD")))
    try:
        assert future.own_airframe_stays(table, w.mvt.loc[r])["icao"].tolist() == ["fff001"]
        state["events"] = table
        X = build(w.mvt, (ADSB,), r)
        state["events"] = without
        Y = build(w.mvt, (ADSB,), r)
    finally:
        state["events"] = w.table
    cols = [f"op_h1_fadsb_{tag}_{c}" for tag in ("f30", "f45") for c in ("taxi_out_now", "rwy_hold_now")]
    assert X.isna().all(axis=None) or X.equals(Y), {"wrong link": X[cols].iloc[0].to_dict(),
                                                    "without the own taxi and hold": Y[cols].iloc[0].to_dict()}


# --- 7. parameters ---------------------------------------------------------------------------------------------------

BAD_SHIFT_DAYS = (-1, -7, 0.5, 1.5, 7.0, True, False, None, "7", np.nan, np.float64(7.0))
GOOD_SHIFT_DAYS = (0, 7, np.int64(7), np.int32(1))


@pytest.mark.parametrize("name", [REC, NM])
def test_params_shift_days_must_be_a_whole_number_and_unknown_keys_raise(world, name):
    """``shift_days``: an integer >= 0 (numpy integers too); negatives, fractions, floats written whole, bools, None and
    strings raise. Unknown keys raise; ``min_support`` is the NM block's only."""
    mvt, _ = world
    rows = t1_rows(mvt)[:4]
    for bad in BAD_SHIFT_DAYS:
        with pytest.raises(ValueError, match="shift_days"):
            build(mvt, (name,), rows, {name: {"shift_days": bad}})
    for good in GOOD_SHIFT_DAYS:
        build(mvt, (name,), rows, {name: {"shift_days": good}})
    assert_identical(build(mvt, (name,), rows, {name: {"shift_days": 0}}), build(mvt, (name,), rows))
    with pytest.raises(ValueError, match="unknown"):
        build(mvt, (name,), rows, {name: {"no_such": 1}})
    if name == NM:
        build(mvt, (name,), rows, {name: {"min_support": 5}})
    else:
        with pytest.raises(ValueError, match="unknown"):
            build(mvt, (name,), rows, {name: {"min_support": 5}})


def test_params_min_support_must_be_a_whole_number_of_at_least_one(world):
    mvt, _ = world
    rows = t1_rows(mvt)[:4]
    for bad in (0, -1, 2.5, 5.0, True, None, "20"):
        with pytest.raises(ValueError, match="min_support"):
            build(mvt, (NM,), rows, {NM: {"min_support": bad}})
    for good in (1, 5, np.int64(20)):
        build(mvt, (NM,), rows, {NM: {"min_support": good}})
    default = build(mvt, (NM,), rows)
    assert_identical(default, build(mvt, (NM,), rows, {NM: {"min_support": _proxylib.MIN_SUPPORT}}))
    anomaly = [c for c in outputs(NM) if "anom" in c]
    assert not default[anomaly].equals(build(mvt, (NM,), rows, {NM: {"min_support": 1}})[anomaly]), "the param is read"


def test_params_the_adsb_block_checks_its_params(aw):
    w, _ = aw
    rows = w.t1f[:3]
    for bad in (-1, 0.5):
        with pytest.raises(ValueError, match="shift_days"):
            build(w.mvt, (ADSB,), rows, {ADSB: {"shift_days": bad}})
    with pytest.raises(ValueError, match="unknown"):
        build(w.mvt, (ADSB,), rows, {ADSB: {"no_such": 1}})
    one_day = build(w.mvt, (ADSB,), rows, {ADSB: {"shift_days": 1}})
    assert not one_day.equals(build(w.mvt, (ADSB,), rows)), "the placebo reads other days"


ADSB_SHIFT_PARAMS = ("op_h1_fut_adsb hands shift_days to _adsblib.check_params, which accepts True (read as 1 day) "
                     "and 7.0, while op_h1_fut_rec and op_h1_fut_nm reject both (operational_future._shift_days): the "
                     "three F blocks of one placebo config do not share one contract")


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=ADSB_SHIFT_PARAMS)
def test_params_the_adsb_block_rejects_the_shift_days_the_record_blocks_reject(aw):
    w, _ = aw
    rows = w.t1f[:2]
    accepted = []
    for bad in (True, 7.0):
        try:
            build(w.mvt, (ADSB,), rows, {ADSB: {"shift_days": bad}})
        except ValueError:
            continue
        accepted.append(bad)
    assert not accepted, f"op_h1_fut_adsb accepted shift_days {accepted}"


def test_params_the_placebo_reads_exactly_the_past(world):
    """``shift_days: 7`` changes ``op_h1_fut_rec`` on a world whose traffic varies by day, and equals the unshifted
    read of the same flight with its anchor a week earlier (its IOBT moved): the same reads, seven days back."""
    mvt, _ = world
    rows = spaced_departures(mvt, hours=20)
    week_in = mvt["MVT_TIME_UTC_mvt"].min() + pd.Timedelta(days=8)
    rows = rows[(mvt.loc[rows, "SCHED_TIME_UTC_mvt"] >= week_in).to_numpy() & mvt.loc[rows, "IOBT_flt"].notna().to_numpy()]
    assert len(rows) >= 8
    base = build(mvt, (REC,), rows)
    week = build(mvt, (REC,), rows, {REC: {"shift_days": 7}})
    changed = (base != week).any(axis=1) & base.notna().any(axis=1)
    assert changed.mean() >= 0.8, changed.mean()
    earlier = build(_shift(mvt, "IOBT_flt", rows, -7 * 86_400), (REC,), rows)
    assert_identical(week, earlier)


def _unguarded(mvt: pd.DataFrame, name: str, dep: pd.Index, a: np.ndarray, days: int) -> pd.DataFrame:
    """The block's shifted reads without the placebo guard (`_at_shifts` without the frame), as the block reads them."""
    if name == REC:
        return future._at_shifts(lambda t: _oplib.traffic_at(mvt, dep, t), a, future.REC, "op_h1_frec_", dep, days)
    deps = _oplib.departures(mvt)
    fit = (deps["split"].eq("train") & deps["is_dep"]).to_numpy(dtype=bool)
    base = _oplib.nm_baseline(deps, fit, _proxylib.MIN_SUPPORT)
    return future._at_shifts(lambda t: _oplib.nm_at(mvt, dep, t, base), a, _oplib.NM, "op_h1_fnm_", dep, days)


@pytest.mark.parametrize("name", [REC, NM])
def test_params_the_placebo_is_nan_where_it_reaches_before_the_frame(world, name):
    """``shift_days: 7`` on the 16-day record world, whose first week has traffic: every row whose shifted reads reach
    before the frame's records -- the guard's start ``a + 15 min - 7 d - 3 h`` before the airport's first record: the
    whole first week and day 8's early hours -- is all NaN, never zero (unguarded, its empty windows read as no
    traffic); every other row is exactly the unguarded read, finite on the count columns wherever the unshifted read
    is."""
    mvt, _ = world
    dep = mvt.index[mvt["is_dep"].to_numpy()]
    a = _oplib.anchor_h1(mvt.loc[dep])
    t = pd.Series(_oplib.secs(mvt["MVT_TIME_UTC_mvt"]), index=mvt.index)
    first = mvt.loc[dep, "apt"].astype(str).map(t.groupby(mvt["apt"].astype(str).to_numpy()).min()).to_numpy()
    start = a + future.SHIFTS["f15"] - 7 * 86_400 - _oplib.BACKLOG_MAX_S
    before = start < first
    week_one = a < np.nanmin(t) + 7 * 86_400
    empty = a + LAST_READ_S - 7 * 86_400 < first - 6 * 3600  # every read hours before the airport's first record
    assert np.isfinite(a).all() and week_one.sum() >= 1000 and (~before).sum() >= 1000
    assert before[week_one].all() and (before & ~week_one).any() and empty.sum() >= 1000
    X = build(mvt, (name,), dep, {name: {"shift_days": 7}})
    assert X.loc[before].isna().all(axis=None)
    raw = _unguarded(mvt, name, dep, a, 7)
    counts = [c for c in X.columns if c.endswith(("_dep_apt_b15", "_arr_apt_b60", "_backlog_apt"))]
    assert len(counts) == 3 * (2 if name == REC else 1)
    assert (raw.loc[empty, counts] == 0).all(axis=None), "unguarded, a read before the frame is zero"
    assert_identical(X.loc[~before], raw.loc[~before])
    assert X.loc[~before].notna().any(axis=1).all()
    plain = build(mvt, (name,), dep)
    for c in counts:
        np.testing.assert_array_equal(X.loc[~before, c].notna().to_numpy(), plain.loc[~before, c].notna().to_numpy())


PLACEBO_REACH = ("operational_future._at_shifts: the placebo guard (window_covered at a + 15 min - shift - "
                 "BACKLOG_MAX_S, 3 h) is shorter than the longest record read: rwy_config_age_min looks back up to 12 h "
                 "(_oplib.MAX_CONFIG_AGE_MIN; CONFIG_LOOKBACK_BUCKETS in the slow path), so a placebo row whose shifted "
                 "read lies 3-12 h after the frame's first record keeps a configuration age truncated at the frame's "
                 "start (210 min against 720 min on a frame one day longer) instead of NaN")


@pytest.mark.parametrize("runways", [
    "alternating",
    pytest.param("one_configuration", marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason=PLACEBO_REACH)),
])
def test_params_the_placebo_reads_nothing_before_the_frame_on_the_rows_it_keeps(world, runways):
    """What the guard promises, tested from outside: the frame's first day cut off, every row the placebo keeps (not
    all NaN) reads what it reads on the uncut frame -- nothing before the cut frame's first record. Compared on every
    column whose value does not rest on a frame-wide table (the arrivals' taxi-in medians and the NM proxy baseline
    change with the frame): the record counts, the runway mix and configuration, the plan backlog. ``alternating``:
    the record world, whose runway configuration changes every four hours; ``one_configuration``: one departure runway
    all day at EDDF, so the configuration age reaches its 12-hour cap."""
    full, _ = world
    full = full.copy()
    if runways == "one_configuration":
        rwy = full["RUNWAY_mvt"].astype(object)
        rwy.loc[full["is_dep"].to_numpy() & full["apt"].astype(str).eq("EDDF").to_numpy()] = "25C"
        full["RUNWAY_mvt"] = pd.Categorical(rwy)
    t = _oplib.secs(full["MVT_TIME_UTC_mvt"])
    cut = full.loc[t >= np.floor(np.nanmin(t) / 86_400) * 86_400 + 86_400]
    dep = cut.index[cut["is_dep"].to_numpy()]
    params = {REC: {"shift_days": 7}, NM: {"shift_days": 7}}
    X_cut, X_full = build(cut, (REC, NM), dep, params), build(full, (REC, NM), dep, params)
    cols = [c for c in X_cut.columns if (c.startswith("op_h1_frec_") and "_arr_anom_" not in c) or "_backlog_" in c]
    kept = X_cut.notna().any(axis=1).to_numpy()
    assert kept.sum() >= 1000 and (~kept).sum() >= 1000
    a, b = X_cut.loc[kept, cols], X_full.loc[kept, cols]
    differ = ~((a == b) | (a.isna() & b.isna()))
    assert not differ.any(axis=None), differ.sum()[differ.sum() > 0].to_dict()


def test_params_the_adsb_placebo_removes_the_same_flight_a_week_earlier(aw):
    """``shift_days: 7`` reads the events of a week before (the world's last day reads its first). A push of another
    airframe under the flight's own callsign, known 7 days before the f15 query -- the same flight a week earlier -- is
    left out by the callsign rule, as the flight's own push is at the real read; under a foreign callsign it counts."""
    w, state = aw
    params = {ADSB: {"shift_days": 7}}
    dep = w.mvt.index[w.mvt["is_dep"].to_numpy(dtype=bool)]
    a = pd.Series(_oplib.anchor_h1(w.mvt.loc[dep]), index=dep)
    day0 = np.floor(float(w.pts["t"].min()) / 86_400) * 86_400
    late = dep[((a - day0) >= 7 * 86_400).to_numpy() & w.mvt.loc[dep, "CALLSIGN_flt"].notna().to_numpy()]
    X = build(w.mvt, (ADSB,), late, params)
    rows = late[X["op_h1_fadsb_f15_push_15"].notna().to_numpy()][:4]
    assert len(rows) >= 3
    col = "op_h1_fadsb_f15_push_15"
    try:
        for r in rows:
            q15 = np.floor(a[r] + future.SHIFTS["f15"]) - LAG
            own_cs = _adsblib.norm_callsign(w.mvt.loc[[r], "CALLSIGN_flt"].to_numpy(dtype=object))[0]
            state["events"] = w.table
            base = build(w.mvt, (ADSB,), one(r), params)[col].iloc[0]
            assert np.isfinite(base)
            for cs, plus in ((own_cs, 0.0), ("FOREIGN9", 1.0)):
                push = event(str(w.mvt.loc[r, "apt"]), "PUSH", "fff777", cs, q15 - 7 * 86_400 - 75, 15.0,
                             origin_kind="push", origin_stand=987_654.0)
                state["events"] = events_table([push], like=w.table)
                assert build(w.mvt, (ADSB,), one(r), params)[col].iloc[0] == base + plus, (r, cs)
    finally:
        state["events"] = w.table


def test_params_the_plan_block_takes_none_and_no_config_gives_it_any(world):
    """``op_h1_fut_plan`` takes no params (a placebo given to it would otherwise be silently ignored): any key raises,
    an empty dict is fine; no config gives it any, and the placebo config shifts exactly the three F blocks."""
    mvt, _ = world
    rows = t1_rows(mvt)[:3]
    for bad in ({"shift_days": 7}, {"shift_days": 0}, {"no_such": 1}):
        with pytest.raises(ValueError, match="takes no params"):
            build(mvt, (PLAN,), rows, {PLAN: bad})
    build(mvt, (PLAN,), rows, {PLAN: {}})
    for path in sorted(paths.CONFIGS.glob("*.yaml")):
        assert PLAN not in load_config(path.name).features.params, path.name
    shift = load_config("op_h1_f4_shift7.yaml").features.params
    assert {k: dict(v) for k, v in shift.items()} == {n: {"shift_days": 7} for n in TIER_F}


# --- 8. quarantine and tiers -----------------------------------------------------------------------------------------

def test_quarantine_no_refit_config_resolves_a_tier_f_block():
    """Tier F is the realised traffic standing in for plans: holdout-only. No config with ``model.refit_on_all`` (the
    board and final members) may resolve a tier F block unless `TIER_F_REFIT` names it -- empty until decided."""
    assert TIER_F_REFIT == ()
    seen = {}
    for path in sorted(paths.CONFIGS.glob("*.yaml")):
        cfg = load_config(path.name)
        blocks = {s.name for s in resolve(cfg.features.include, cfg.features.exclude)}
        tier_f = sorted(blocks & set(TIER_F))
        if not tier_f:
            continue
        seen[path.name] = tier_f
        assert path.name.startswith("op_h1_"), f"{path.name}: tier F blocks are H1 blocks"
        assert not (cfg.model.refit_on_all and path.name not in TIER_F_REFIT), (
            f"{path.name} is a refit (board / final) config but resolves the tier F blocks {tier_f}")
    assert seen, "no config resolves a tier F block: the scan would prove nothing"
    assert {s.name for s in resolve(["group:op_h1"])}.isdisjoint(BLOCKS), "the H1 board group pulls in no future block"


def test_quarantine_every_config_with_the_adsb_f_block_blanks_it_on_part_of_the_fit():
    """As R6 (tests/test_operational.py): a fit that never sees the ADS-B F block blank would read NaN as more than
    "no feed" -- here also "own airframe not identified"; `model.block_dropout` makes NaN common in every fit."""
    seen = []
    for path in sorted(paths.CONFIGS.glob("*.yaml")):
        cfg = load_config(path.name)
        if ADSB in {s.name for s in resolve(cfg.features.include, cfg.features.exclude)}:
            seen.append(path.name)
            assert cfg.model.block_dropout.get(ADSB, 0) > 0, path.name
    assert {"op_h1_f4_adsb.yaml", "op_h1_f4_nodepthru.yaml", "op_h1_f4_norwy.yaml"} <= set(seen)


OWN_RUNWAY = r"_rwy_b|_rwy_s$|rwy_own|nm_rwy|backlog_rwy"
"""Columns grouped by the own runway (`test_operational`'s runway-blind pattern, the parts that reach these blocks): a
cross-check of `_long_tiers`' ``own_runway`` tag on the F columns."""


def _f_outputs() -> list[str]:
    return [c for name in TIER_F for c in all_specs()[name].outputs]


def test_quarantine_the_own_runway_free_variant_drops_exactly_the_f_columns_of_the_own_runway():
    """Rule T's own-runway variant (findings 10.36): op_h1_f4_norwy is op_h1_f4_adsb without every tier F column
    `_long_tiers` tags ``own_runway`` -- the own runway's realised sequence, whose slot the own takeoff takes, the NM
    readings and plan backlog of that runway, its ADS-B holds -- and nothing else; the plans (O*) stay, the own
    runway's included."""
    norwy, full = load_config("op_h1_f4_norwy.yaml"), load_config("op_h1_f4_adsb.yaml")
    assert _blocks("op_h1_f4_norwy.yaml") == _blocks("op_h1_f4_adsb.yaml")
    f_cols = _f_outputs()
    own_runway = {c for c in f_cols if "own_runway" in _long_tiers.tags_of(c)}
    assert own_runway == {c for c in f_cols if re.search(OWN_RUNWAY, c)}, "the tags and the runway-blind pattern agree"
    assert len(own_runway) == 3 * (5 + 7 + 1)
    dropped = set(norwy.features.drop_columns) - set(full.features.drop_columns)
    assert dropped == own_runway, (sorted(dropped - own_runway), sorted(own_runway - dropped))
    kept = set(_kept("op_h1_f4_norwy.yaml"))
    assert not [c for c in kept if c.startswith(FUTURE_PREFIXES[1:]) and "own_runway" in _long_tiers.tags_of(c)]
    assert set(all_specs()[PLAN].outputs) <= kept
    assert set(_kept("op_h1_f4_adsb.yaml")) - kept == own_runway


def _arrival_side() -> set[str]:
    """The realised ARRIVAL columns after the anchor: the records' landings, arrivals taxiing and taxi-in anomaly, and
    the ADS-B taxi-ins."""
    return ({f"op_h1_frec_{tag}_{c}" for tag in future.SHIFTS for c in future.REC if c.startswith("arr_")}
            | {f"op_h1_fadsb_{tag}_taxi_in_now" for tag in future.SHIFTS})


def test_quarantine_the_departure_free_variant_keeps_only_plans_and_realised_arrivals():
    """Rule T's own-slot bound (findings 10.36, amended): op_h1_f4_nodepthru keeps, after the anchor, exactly the
    other flights' plans (O*) and the realised ARRIVAL columns (tier F); no NM block (every column departure-side), no
    realised departure-side record or ADS-B column, at airport or runway level. Before the anchor it is the H1
    deliverable, as op_h1_f4_adsb; it stays holdout-only with the ADS-B F block blanked on part of the fit."""
    name, full = "op_h1_f4_nodepthru.yaml", "op_h1_f4_adsb.yaml"
    assert _blocks(name) == _blocks(full) - {NM}
    kept = set(_kept(name))
    plans_ = set(all_specs()[PLAN].outputs)
    arrivals = _arrival_side()
    assert len(arrivals) == 3 * (8 + 1)
    assert {c for c in kept if c.startswith(FUTURE_PREFIXES)} == plans_ | arrivals
    assert not [c for c in kept if c.startswith("op_h1_fnm_")]
    departure_side = set(_f_outputs()) - arrivals
    assert not kept & departure_side
    assert {c for c in kept if not c.startswith(FUTURE_PREFIXES)} == {
        c for c in _kept(full) if not c.startswith(FUTURE_PREFIXES)}
    cfg = load_config(name)
    assert set(cfg.features.drop_columns) - {"takeoff_epoch", "op_h1_day_of_year"} <= departure_side
    assert not cfg.model.refit_on_all and cfg.model.block_dropout.get(ADSB, 0) > 0


def test_quarantine_the_rule_t_variants_differ_from_f4_only_in_what_they_name():
    """Rule T's ladder adds one block per step to op_h1_r4_nodoy, the H1 deliverable it was run against (F1 plans, F2
    records, F3 NM, F4 ADS-B); the placebo keeps F4's columns and changes only its params; F5 adds R6's ADS-B picture
    before the anchor (its coverage columns dropped, as R6). Every rule T config runs at seed 42, as `lab/op_rule_t.py`
    reads them, and all are holdout-only but F1, refitted since its adoption as the H1 deliverable (findings 10.37: its
    inputs are O and O*, no tier F)."""
    h1 = _blocks("op_h1_r4_nodoy.yaml")
    ladder = [("op_h1_f1_plan.yaml", PLAN), ("op_h1_f2_rec.yaml", REC), ("op_h1_f3_nm.yaml", NM), ("op_h1_f4_adsb.yaml", ADSB)]
    previous = h1
    for name, block in ladder:
        assert _blocks(name) - previous == {block} and previous < _blocks(name), name
        previous = _blocks(name)
    assert set(_kept("op_h1_f1_plan.yaml")) == set(_kept("op_h1_r4_nodoy.yaml")) | set(all_specs()[PLAN].outputs)
    f4 = set(_kept("op_h1_f4_adsb.yaml"))
    assert _blocks("op_h1_f4_shift7.yaml") == _blocks("op_h1_f4_adsb.yaml") and set(_kept("op_h1_f4_shift7.yaml")) == f4
    assert _blocks("op_h1_f5_full.yaml") - _blocks("op_h1_f4_adsb.yaml") == {"op_h1_adsb"}
    assert set(_kept("op_h1_f5_full.yaml")) == f4 | {f"op_h1_adsb_{c}" for c in _adsblib.SURFACE + _adsblib.STAND}
    for name in ("op_h1_f1_plan.yaml", "op_h1_f2_rec.yaml", "op_h1_f3_nm.yaml", "op_h1_f4_adsb.yaml",
                 "op_h1_f4_shift7.yaml", "op_h1_f4_norwy.yaml", "op_h1_f4_nodepthru.yaml", "op_h1_f5_full.yaml"):
        cfg = load_config(name)
        assert cfg.seed == 42, name
        assert cfg.model.refit_on_all == (name == "op_h1_f1_plan.yaml"), name  # F1 alone: the H1 deliverable
        assert cfg.name == name.removesuffix(".yaml"), "the run directory carries the config's name (op_rule_t.seed_run)"


def test_tiers_every_f_output_is_tier_f_at_h1_and_every_plan_output_o_or_o_star():
    """`_long_tiers` (the feature catalogue's tiers): every records, NM and ADS-B output after the anchor carries
    ``others_realised`` and reads as tier F at H1; every plan output is O (the airport's scheduled arrivals) or O* (the
    other flights' plans final, their runway and stand allocation), never F; and every pattern of these blocks still
    matches a column (none went stale with a rename)."""
    for c in _f_outputs():
        assert "others_realised" in _long_tiers.tags_of(c), c
        assert _long_tiers.tier_at(c, "h1").startswith("F ("), c
    for c in all_specs()[PLAN].outputs:
        assert "others_realised" not in _long_tiers.tags_of(c), c
        tier = _long_tiers.tier_at(c, "h1")
        assert tier == "O" or tier.startswith("O* ("), (c, tier)
    assert _long_tiers.tier_at("op_h1_fplan_arr_sched_apt_w60", "h1") == "O"
    assert _long_tiers.tags_of("op_h1_fplan_dep_plan_rwy_w15") == {"own_runway", "others_future", "others_plan"}
    outs = [c for name in BLOCKS for c in all_specs()[name].outputs]
    for rx, _ in _long_tiers.PATTERNS:
        if rx.startswith("^op_h1_f"):
            assert any(re.search(rx, c) for c in outs), rx
