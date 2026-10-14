"""lab/op_adsb_audit.py on the synthetic ADS-B world: the identity rules leave no own contamination, an injected own
event under a foreign callsign and stand is flagged (and the block counts it), one under the own callsign is neither
flagged nor counted, the audit's `relevant` windows are exactly the block's (one injected event of each kind changes
the block's output if and only if `relevant` says it enters a statistic), the gates are on the linked exposed rows and
fail above their thresholds or without enough links (the exit code follows the failures). Synthetic only."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

import adsb_synth as S
from prc import paths
from prc.adsb import surface
from prc.features import _adsblib, _oplib, all_specs
from prc.features.build import compute_block
from prc.features.registry import FeatureContext
from test_operational import synthetic_world


def _audit():
    spec = importlib.util.spec_from_file_location("op_adsb_audit", paths.ROOT / "lab" / "op_adsb_audit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def setup():
    mvt, ob = synthetic_world(days=7, per_day=30)
    mvt = S.with_callsigns(mvt)
    pts, info = S.world(mvt, ob)
    ev = surface.episodes(pts, S.geometry(), float(pts["t"].max()))
    dep = mvt.loc[mvt["is_dep"].to_numpy(dtype=bool) & mvt["split"].eq("train").to_numpy()]
    rows = dep.index[dep["STAND_mvt"].notna().to_numpy() & dep["CALLSIGN_flt"].notna().to_numpy()].intersection(info.index)
    links = info.loc[rows, ["icao", "t_lift"]].rename(columns={"t_lift": "t_takeoff"})
    anchors = {"h1": pd.Series(_oplib.anchor_h1(dep.loc[rows]), index=rows),
               "h0": pd.Series(_oplib.anchor_h0(dep.loc[rows], ob), index=rows)}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_adsblib, "load_stand_tables", S.stand_tables)
        mp.setattr(_adsblib, "load_runway_keys", S.runway_keys)
        support = S.support_of(mvt, S.table(ev))
    return mvt, ob, info, ev, dep.loc[rows], links, anchors, support


@pytest.fixture
def block(setup, monkeypatch):
    """``run(table, row)``: op_h1_adsb of one row on the given event table (synthetic geometry and support)."""
    mvt, *_, support = setup
    monkeypatch.setattr(_adsblib, "load_stand_tables", S.stand_tables)
    monkeypatch.setattr(_adsblib, "load_runway_keys", S.runway_keys)
    monkeypatch.setattr(_adsblib, "load_support", lambda: support)

    def run(table: pd.DataFrame, r) -> pd.Series:
        t = S.table(table)
        monkeypatch.setattr(_adsblib, "load_events", lambda t_min=None, t_max=None, t=t: t)
        ctx = FeatureContext(dep_index=pd.Index([r], name="mvt_id"))
        return compute_block(all_specs()["op_h1_adsb"], mvt, ctx).iloc[0]

    return run


def _event(apt: str, kind: str, ts: float, tks: float, tke: float = np.nan, icao: str = "fff999",
           cs: str = "FOREIGN1", stand: float = np.nan, origin: str | None = None) -> dict:
    return {"eid": 1, "apt": apt, "kind": kind, "icao": icao, "cs": cs, "t_start": ts, "t_known_start": tks,
            "t_end": tke, "t_known_end": tke, "origin_kind": origin, "origin_stand": stand,
            "rwy_end": "25C" if kind in ("TAKEOFF", "LANDING", "RWY_HOLD") else None,
            "dur": (tke - ts) if np.isfinite(tke) else np.nan, "stop_s": np.nan, "path_m": np.nan, "taxi_seen": 0,
            "parent": -1}


def _with(ev: pd.DataFrame, *rows: dict) -> pd.DataFrame:
    return pd.concat([ev, pd.DataFrame(list(rows))], ignore_index=True)


def test_rules_leave_no_contamination_and_an_injected_own_event_is_flagged(setup, block):
    mvt, ob, info, ev, dep, links, anchors, _ = setup
    aud = _audit()
    stands, matches = S.stand_tables()
    flags = aud.own_contamination(ev, dep, links, anchors, stands, matches)
    assert len(flags) == len(dep) and len(dep) > 150
    assert not flags["h1_own"].any() and not flags["h0_own"].any(), "the identity rules let an own event through"
    # the own departure did enter windows before the rules (pushes before H1 exist in this world): the rules have teeth
    no_rules = aud.own_contamination(ev.assign(cs=None, origin_stand=np.nan), dep, links, anchors, stands, matches)
    assert no_rules["h1_own"].sum() >= 5 and no_rules["h0_own"].sum() >= 5
    # a row covered at H1 (the stand family too) whose flight takes off within the hour after its anchor
    lead = links["t_takeoff"] - anchors["h1"]
    cand = lead.index[(lead > 900) & (lead < 3600)]
    base = {r: block(ev, r) for r in cand[:15]}
    r = next(r for r in base if base[r]["op_h1_adsb_cov_stand_ok"] == 1)
    q = float(np.floor(anchors["h1"][r]) - _adsblib.ADSB_LAG_S)
    apt, own_icao = str(dep.loc[r, "apt"]), info.loc[r, "icao"]
    far = float(S.stand_id(apt, 20 if int(info.loc[r, "stand_i"]) < 10 else 0))  # a stand 800 m and more away
    # one own push under a foreign callsign, from a far stand, known 100 s before the H1 query: flagged, and counted
    fake = _event(apt, "PUSH", q - 115.0, q - 100.0, icao=own_icao, cs="FOREIGN1", stand=far, origin="push")
    flags2 = aud.own_contamination(_with(ev, fake), dep, links, anchors, stands, matches)
    assert flags2.loc[r, "h1_own"] and flags2["h1_own"].sum() == 1
    assert block(_with(ev, fake), r)["op_h1_adsb_push_15"] == base[r]["op_h1_adsb_push_15"] + 1
    # the same push under the flight's own callsign: the block excludes it, the audit does not flag it
    same_cs = _event(apt, "PUSH", q - 115.0, q - 100.0, icao=own_icao, cs=str(dep.loc[r, "CALLSIGN_flt"]), stand=far,
                     origin="push")
    assert not aud.own_contamination(_with(ev, same_cs), dep, links, anchors, stands, matches).loc[r, "h1_own"]
    pd.testing.assert_series_equal(block(_with(ev, same_cs), r), base[r], check_exact=True)
    # the gates and the exit code (every row exposed and linked here)
    op = dep.assign(apt=dep["apt"].astype(str))
    assert not aud.gate_failures(aud.contamination_table(flags, op))
    fails = aud.gate_failures(aud.contamination_table(flags2, op))
    assert fails and any(f.startswith("h1 ") for f in fails)


def test_relevant_mirrors_the_block_windows():
    aud = _audit()
    q = np.array([30_000.0] * 12)
    kind = np.array(["PUSH", "PUSH", "TAXI_OUT", "TAXI_OUT", "RWY_HOLD", "RWY_HOLD", "REMOTE_STOP", "TAKEOFF", "TAKEOFF",
                     "LANDING", "TAXI_IN", "TAXI_IN"])
    ts = np.array([29_000, 29_000, 26_500, 20_000, 29_800, 20_000, 29_000, 8_390, 8_399, 22_700, 29_000, 20_000], dtype=float)
    tks = np.array([29_100, 30_001, 26_600, 22_799, 29_830, 22_799, 29_300, 8_399, 8_400, 22_799, 29_000, 20_000], dtype=float)
    tke = np.array([np.nan, np.nan, np.nan, 23_000, np.nan, 23_000, np.nan, 8_399, 8_400, 22_799, np.nan, 21_000],
                   dtype=float)
    got = aud.relevant(kind, ts, tks, tke, q)
    # push in the 2-h heartbeat window / after q; taxi-out in progress / known 7,201 s before q and ended; hold in
    # progress / known 7,201 s before and ended long ago; remote stop in progress; takeoff known 21,601 s / 21,600 s
    # before q (the 6-h push share); landing 7,201 s before; taxi-in in progress / ended
    assert got.tolist() == [True, False, True, False, True, False, True, False, True, False, True, False]
    # the columns the model sees: no coverage window (2-h counts, the 6-h push share, the heartbeat, landings)
    model = aud.relevant(kind, ts, tks, tke, q, model=True)
    assert model.tolist() == [True, False, True, False, True, False, True, False, False, False, True, False]
    q2 = np.array([30_000.0] * 3)
    k2 = np.array(["TAKEOFF"] * 3)
    assert aud.relevant(k2, [27_000.0] * 3, [27_000.0, 27_000.0, 26_000.0], [27_000.0] * 3, q2, dur=[600.0, np.nan, 600.0],
                        model=True).tolist() == [True, False, False]  # within the hour, with a taxi-out duration


SLOT_SPACING_S = 8 * 3600
"""Query slots of one airport lie this far apart: an event placed around one slot (at most 6 h before it, open events
capped at 1 h, ends at most 1 s after it) never enters another slot's windows."""


@pytest.fixture(scope="module")
def slots(setup):
    """Departures whose three column families are gated in at H1 and whose activity ratio is positive (an event in the
    heartbeat's base window then shows), at least ``SLOT_SPACING_S`` apart per airport: (row, q, airport) and the
    block's output on the unchanged table."""
    mvt, ob, info, ev, dep, links, anchors, support = setup
    alld = mvt.loc[mvt["is_dep"].to_numpy(dtype=bool)]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_adsblib, "load_stand_tables", S.stand_tables)
        mp.setattr(_adsblib, "load_runway_keys", S.runway_keys)
        mp.setattr(_adsblib, "load_support", lambda: support)
        t = S.table(ev)
        mp.setattr(_adsblib, "load_events", lambda t_min=None, t_max=None: t)
        X = compute_block(all_specs()["op_h1_adsb"], mvt, FeatureContext(dep_index=alld.index))
    ok = (X[[f"op_h1_adsb_{g}" for g in ("cov_taxi_ok", "cov_hold_ok", "cov_stand_ok")]] == 1).all(axis=1).to_numpy()
    ok = ok & (X["op_h1_adsb_cov_act_ratio_5"] > 0).to_numpy()
    q = np.floor(_oplib.anchor_h1(alld)) - _adsblib.ADSB_LAG_S
    picked = []
    for apt in sorted(alld["apt"].astype(str).unique()):
        at = np.flatnonzero(ok & alld["apt"].astype(str).eq(apt).to_numpy() & np.isfinite(q))
        last = -np.inf
        for i in at[np.argsort(q[at], kind="stable")]:
            if q[i] - last >= SLOT_SPACING_S:
                picked.append((alld.index[i], float(q[i]), apt))
                last = q[i]
    return picked, X


def _cases(kind: str) -> list[tuple[float, float, float]]:
    """(t_start, t_known_start, t_known_end) offsets from q: around every window edge of the block."""
    caps = {"TAXI_OUT": _adsblib.CAP_TAXI_OUT_S, "RWY_HOLD": _adsblib.CAP_HOLD_S, "REMOTE_STOP": _adsblib.CAP_REMOTE_S,
            "TAXI_IN": _adsblib.CAP_TAXI_IN_S}
    out = []
    for back in (-1, 0, 1, 299, 301, 899, 901, 1799, 1801, 3599, 3601, 7199, 7200, 7201, 21_599, 21_601):
        for ends in (None, -1, 1, 1799, 1801, 3599, 3601):  # open, or ended this long before q (-1: just after it)
            tke = np.nan if ends is None else -float(ends)
            if np.isfinite(tke) and tke <= -back:
                continue  # an end before the start
            out.append((-back - 20.0, -float(back), tke))
        if kind in caps and back >= 0:  # open, started just within / just beyond the cap
            out += [(-caps[kind] + 1.0, -float(back), np.nan), (-caps[kind] - 1.0, -float(back), np.nan)]
    return [c for c in out if c[0] <= c[1]]


@pytest.mark.parametrize("kind", ["PUSH", "TAXI_OUT", "RWY_HOLD", "REMOTE_STOP", "TAKEOFF", "LANDING", "TAXI_IN"])
def test_one_injected_event_changes_the_block_exactly_when_relevant_says_so(setup, slots, monkeypatch, kind):
    """Events of ``kind`` placed in and around every window of the block, one per covered query slot: the block's
    output changes if and only if the audit's `relevant` says the event enters a statistic -- so the audit cannot miss
    an own event the block would count, nor flag one it would not."""
    mvt, ob, info, ev, dep, links, anchors, support = setup
    picked, X = slots
    assert len(picked) >= 6
    aud = _audit()
    monkeypatch.setattr(_adsblib, "load_stand_tables", S.stand_tables)
    monkeypatch.setattr(_adsblib, "load_runway_keys", S.runway_keys)
    monkeypatch.setattr(_adsblib, "load_support", lambda: support)
    cases = _cases(kind)
    changed, rel, where, changed_model, rel_model = [], [], [], [], []
    cov = [f"op_h1_adsb_{c}" for c in _adsblib.COVERAGE]
    for b0 in range(0, len(cases), len(picked)):
        batch = cases[b0:b0 + len(picked)]
        rows, events = [], []
        for k, ((r, q, apt), (o_ts, o_tks, o_tke)) in enumerate(zip(picked, batch)):
            ts, tks, tke = q + o_ts, q + o_tks, q + o_tke
            origin = "push" if kind in ("PUSH", "TAXI_OUT") else None  # a taxi-out from a push: a new start
            events.append(_event(apt, kind, ts, tks, tke, icao=f"dd{k:04x}", cs=f"NOTOWN{k}", origin=origin))
            rows.append(r)
            rel.append(bool(aud.relevant(np.array([kind], dtype=object), [ts], [tks], [tke], [q])[0]))
            rel_model.append(bool(aud.relevant(np.array([kind], dtype=object), [ts], [tks], [tke], [q], dur=[events[-1]["dur"]],
                                               start=aud.taxi_out_starts(pd.DataFrame([events[-1]])), model=True)[0]))
            where.append((o_ts, o_tks, o_tke))
        t = S.table(_with(ev, *events))
        monkeypatch.setattr(_adsblib, "load_events", lambda t_min=None, t_max=None, t=t: t)
        out = compute_block(all_specs()["op_h1_adsb"], mvt, FeatureContext(dep_index=pd.Index(rows, name="mvt_id")))
        changed += [not out.loc[r].equals(X.loc[r]) for r in rows]
        changed_model += [not out.loc[r].drop(cov).equals(X.loc[r].drop(cov)) for r in rows]
    changed, rel, changed_model, rel_model = map(np.array, (changed, rel, changed_model, rel_model))
    assert rel.sum() >= 5 and (~rel).sum() >= 5, kind
    bad = [w for w, ch, re in zip(where, changed, rel) if ch != re]
    assert not bad, (kind, "changed != relevant at (t_start, t_known_start, t_known_end) - q:", bad[:6])
    # an event in a window of the model's columns does change them (a takeoff's own effect there depends on the
    # stops of the other taxi-outs, a landing has none)
    if kind not in ("TAKEOFF", "LANDING"):
        assert rel_model.sum() >= 3, kind
        bad = [w for w, ch, re in zip(where, changed_model, rel_model) if re and not ch]
        assert not bad, (kind, "in a model window, but the model's columns did not change:", bad[:6])
    assert not (rel_model & ~rel).any()


def test_gates_need_linked_exposed_rows():
    """Contamination counts on the linked exposed rows; an airport whose exposed rows are mostly unlinked fails, one
    without exposed rows passes whatever its links."""
    aud = _audit()
    idx = pd.Index(np.arange(1, 401), name="mvt_id")
    op = pd.DataFrame({"apt": np.repeat(["AAAA", "BBBB"], 200)}, index=idx)
    flags = pd.DataFrame({"h1_own": False, "h1_any": False}, index=idx[:250])  # AAAA all linked, BBBB 50 of 200
    flags.loc[idx[:1], "h1_own"] = True
    t = aud.contamination_table(flags, op, {"h1": pd.Series(True, index=idx)}).set_index("apt")
    assert t.loc["AAAA", "share"] == 1 / 200 and t.loc["BBBB", "link_share"] == 0.25 and t.loc["ALL", "linked"] == 250
    assert t.loc["ALL", "share_op"] == 1 / 400
    assert aud.gate_failures(t.reset_index()) == ["h1 ALL: 0.400 % of the linked exposed OP rows contaminated (gate < 0.2 %)"]
    unlinked_b = flags.iloc[:180].assign(h1_own=False)  # BBBB: no row linked
    only_a = {"h1": pd.Series(np.r_[np.ones(200), np.zeros(200)].astype(bool), index=idx)}
    assert not aud.gate_failures(aud.contamination_table(unlinked_b, op, only_a))  # ... and never exposed
    fails = aud.gate_failures(aud.contamination_table(unlinked_b, op, {"h1": pd.Series(True, index=idx)}))
    assert any("BBBB" in f and "not measured" in f for f in fails), fails
