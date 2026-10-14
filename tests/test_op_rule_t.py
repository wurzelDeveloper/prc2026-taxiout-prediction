"""Rule T's decisions (lab/op_rule_t.py, findings 10.36) on synthetic step results: no runs, no data, no GPU.

`stage1` reads the H1 deliverable's and rule T's eight holdout runs at seed 42 (`seed_run`: at the configuration's
current hash where configs/ has the name; `op_ladder_report.read_oof`), keeps the rows from `FIRST_DAY` on (the
placebo reads a week back and the configuration age 15 h more, so 1-8 January would reach before the data) and steps
each run against H1 (`op_ladder_report.paired_step`, a day-block bootstrap). Here the runs are synthetic holdout
frames whose per-row errors fix each run's OP MAE change from `FIRST_DAY` on, and make the earlier rows much worse, so
a comparison that looked at them would decide otherwise. The ladder report is a stand-in module in ``sys.modules``
whose `paired_step` is the exact paired change with a chosen interval half-width; `op_rules.record` is a stand-in
that keeps the decision (the real runs/op_decisions.json is never touched). The member choice follows rule T as
amended before any run (findings 10.36):

* c1: F4 gains at least 2 s of OP MAE with the interval below zero;
* c2: the shift-7 placebo keeps at most a quarter of that gain;
* c3: F4-nodepthru (no realised departure-side column after the anchor) keeps at least three quarters of it: the
  member is F4, or F5 if F5 is at least 0.5 s better than F4; else F4-nodepthru if it passes c1 on its own (F4 having
  passed c1 and c2); else none.

The bounds are inclusive as worded ("at least", "at most"), the intervals strictly below zero. `stage2` (the month
pairs, 8 of 10 months better, and the forward check) and the commands ``passed13``, ``member13`` and ``member`` are read
on a temporary runs directory (`paths.RUNS`).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prc import paths
from prc.config import load_config


def _load_rule_t():
    spec = importlib.util.spec_from_file_location("op_rule_t", paths.ROOT / "lab" / "op_rule_t.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rt = _load_rule_t()

H1_ABS = 100.0
"""H1's absolute error on every row (s)."""
EARLY_EXTRA = 60.0
"""Every rule T run's error on the rows before `FIRST_DAY`, above H1's: included, they would sink each gain."""
ROWS_PER_DAY = 8
STAMP = "2026-10-12T00-00-00"
PAIR_MONTHS = {"febdec": ("2025-02", "2025-12"), "marnov": ("2025-03", "2025-11"), "aproct": ("2025-04", "2025-10"),
               "maysep": ("2025-05", "2025-09"), "junaug": ("2025-06", "2025-08")}


# --- the synthetic holdout --------------------------------------------------------------------------------------------

def _context(months: tuple[str, ...] = ("2025-01", "2025-07")) -> pd.DataFrame:
    """The holdout's context as `op_ladder_report.holdout_context` gives it (day, month, apt), every row OP."""
    days = pd.DatetimeIndex([])
    for m in months:
        start = pd.Timestamp(f"{m}-01")
        days = days.append(pd.date_range(start, start + pd.offsets.MonthEnd(0), freq="D"))
    day = np.repeat(days.strftime("%Y-%m-%d").to_numpy(), ROWS_PER_DAY)
    ids = pd.Index(np.arange(1, len(day) + 1, dtype="int64") + 1000 * len(months), name="mvt_id")
    return pd.DataFrame({"day": day, "month": [d[:7] for d in day],
                         "apt": np.where(np.arange(len(day)) % 2, "EDDF", "LSZH"), "op": True}, index=ids)


def _oof(ctx: pd.DataFrame, gain: float, hw: float = 1.0, early: float = EARLY_EXTRA) -> pd.DataFrame:
    """A run's holdout frame: target 600 s, absolute error ``H1_ABS - gain`` from `FIRST_DAY` on and
    ``H1_ABS + early`` before, its sign alternating by row; ``hw``: the half-width of its interval against H1, which
    the stand-in `paired_step` reads. Every value is a binary fraction, so the boundary cases compare exactly."""
    late = (ctx["day"] >= rt.FIRST_DAY).to_numpy()
    err = np.where(late, H1_ABS - gain, H1_ABS + early)
    sign = np.where(np.arange(len(ctx)) % 2, 1.0, -1.0)
    y = np.full(len(ctx), 600.0)
    return pd.DataFrame({"y_true": y, "y_pred": y + sign * err, "hw": hw}, index=ctx.index)


def _run_name(config: str) -> str:
    return f"{STAMP}__{config}__00000000__s42"


def _ladder(ctx: pd.DataFrame, oofs: dict[str, pd.DataFrame]):
    """The stand-in `op_ladder_report` (what lab/op_rule_t.py calls) and the days each comparison was handed."""
    seen: dict[str, list[str]] = {"op_frame": [], "paired_step": []}
    day = ctx["day"]

    def earliest(*indexes) -> str:
        return min(day.reindex(ix).min() for ix in indexes)

    def read_oof(run: Path) -> pd.DataFrame:
        return oofs[Path(run).name].copy()

    def holdout_context(ids: pd.Index, disrupted) -> pd.DataFrame:
        assert disrupted is None
        return ctx.reindex(ids)

    def op_frame(oof: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
        seen["op_frame"].append(earliest(oof.index, c.index))
        frame = c.reindex(oof.index).copy()
        frame["y_true"], frame["y_pred"] = oof["y_true"], oof["y_pred"]
        frame["op"] = frame["op"].fillna(False).astype(bool)
        return frame[["y_true", "y_pred", "apt", "op"]]

    def metric_row(y_true, y_pred) -> dict:
        e = np.asarray(y_pred, dtype="float64") - np.asarray(y_true, dtype="float64")
        return {"n": float(len(e)), "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt(np.mean(e ** 2))),
                "within_120": float((np.abs(e) <= 120).mean())}

    def paired_step(a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame, n_boot: int = 1000, seed: int = 0,
                    apt: str | None = None) -> dict:
        seen["paired_step"].append(earliest(a.index, b.index, c.index))
        ids = a.index.intersection(b.index).intersection(c.index)
        ea = (a.loc[ids, "y_pred"] - a.loc[ids, "y_true"]).abs()
        eb = (b.loc[ids, "y_pred"] - b.loc[ids, "y_true"]).abs()
        d_mae, hw = float((eb - ea).mean()), float(b["hw"].iloc[0])
        d_w = float(((eb <= 120).astype(float) - (ea <= 120).astype(float)).mean())
        return {"n": int(len(ids)), "days": int(c.loc[ids, "day"].nunique()), "d_mae": d_mae, "mae_lo": d_mae - hw,
                "mae_hi": d_mae + hw, "d_w120": d_w, "w120_lo": d_w, "w120_hi": d_w, "d_bias": 0.0, "bias_lo": 0.0,
                "bias_hi": 0.0}

    mod = types.ModuleType("op_ladder_report")
    mod.__dict__.update(read_oof=read_oof, holdout_context=holdout_context, op_frame=op_frame, metric_row=metric_row,
                        paired_step=paired_step)
    return mod, seen


def _rules(store: dict):
    """The stand-in `op_rules`: `record` keeps the decision -- as JSON, so a value the real file cannot hold fails."""
    mod = types.ModuleType("op_rules")
    mod.record = lambda key, value: store.__setitem__(key, json.loads(json.dumps(value)))
    return mod


def run_stage1(monkeypatch, gains: dict[str, tuple[float, float] | None], h1: bool = True):
    """`stage1` on synthetic runs. ``gains``: rule T row (`RUNS_T` key) -> (OP MAE gain over H1 from `FIRST_DAY` on,
    interval half-width), or None for a run that does not exist; ``h1``: whether the H1 run exists."""
    ctx = _context()
    rows = {"H1": (0.0, 0.0)} if h1 else {}
    rows.update({k: v for k, v in gains.items() if v is not None})
    configs = {"H1": rt.BASE, **rt.RUNS_T}
    oofs = {_run_name(configs[k]): _oof(ctx, g, hw, early=0.0 if k == "H1" else EARLY_EXTRA)
            for k, (g, hw) in rows.items()}
    available = {configs[k] for k in rows}
    monkeypatch.setattr(rt, "seed_run", lambda name, seed=42: Path(_run_name(name)) if seed == 42 and name in available
                        else None)
    ladder, seen = _ladder(ctx, oofs)
    store: dict = {}
    monkeypatch.setitem(sys.modules, "op_ladder_report", ladder)
    monkeypatch.setitem(sys.modules, "op_rules", _rules(store))
    dec = rt.stage1()
    assert store == {"T13": json.loads(json.dumps(dec))}, "the decision is recorded under T13, as returned"
    return dec, seen, ctx


BASE_GAINS = {"F1": (0.5, 1.0), "F2": (2.0, 1.0), "F3": (3.0, 1.0), "F4": (5.0, 1.0), "F4-shift7": (1.0, 1.0),
              "F4-norwy": (4.5, 1.0), "F4-nodepthru": (4.0, 1.0), "F5": (5.25, 1.0)}
"""F4 gains 5 s (interval [-6, -4]), the placebo 1 s, F4-nodepthru 4 s (80 %), F5 0.25 s more than F4."""


# --- stage 1: the member ----------------------------------------------------------------------------------------------

CASES = {
    "f4": ({}, "op_h1_f4_adsb"),
    "f5_better_by_0.6": ({"F5": (5.6, 1.0)}, "op_h1_f5_full"),
    "f5_better_by_exactly_0.5": ({"F5": (5.5, 1.0)}, "op_h1_f5_full"),
    "f5_missing": ({"F5": None}, "op_h1_f4_adsb"),
    "nodepthru_keeps_exactly_three_quarters": ({"F4-nodepthru": (3.75, 1.0)}, "op_h1_f4_adsb"),
    "nodepthru_better_than_f4": ({"F4-nodepthru": (6.0, 1.0)}, "op_h1_f4_adsb"),
    "nodepthru_2s_share_0.4": ({"F4-nodepthru": (2.0, 1.0)}, "op_h1_f4_nodepthru"),
    "nodepthru_2s_interval_reaches_zero": ({"F4-nodepthru": (2.0, 2.0)}, None),
    "nodepthru_1s": ({"F4-nodepthru": (1.0, 0.5)}, None),
    "placebo_2s_of_5s": ({"F4-shift7": (2.0, 1.0)}, None),
    "placebo_exactly_a_quarter": ({"F4-shift7": (1.25, 1.0)}, "op_h1_f4_adsb"),
    "placebo_worse_than_h1": ({"F4-shift7": (-3.0, 1.0)}, "op_h1_f4_adsb"),
    "f4_1.5s": ({"F4": (1.5, 0.5), "F4-shift7": (0.25, 0.5), "F4-nodepthru": (1.5, 0.5), "F5": (1.5, 0.5)}, None),
    "f4_exactly_2s": ({"F4": (2.0, 1.0), "F4-shift7": (0.5, 1.0), "F4-nodepthru": (1.5, 1.0), "F5": (2.0, 1.0)},
                      "op_h1_f4_adsb"),
    "f4_interval_reaches_zero": ({"F4": (5.0, 5.0)}, None),
    "f4_fails_while_nodepthru_alone_would_pass": ({"F4": (1.5, 0.5), "F4-nodepthru": (3.0, 0.5)}, None),
}


@pytest.mark.parametrize("case", list(CASES))
def test_stage1_chooses_the_member_by_rule_t(monkeypatch, case):
    change, member = CASES[case]
    dec, _, _ = run_stage1(monkeypatch, {**BASE_GAINS, **change})
    assert dec["member"] == member, (case, dec)
    assert dec["passed"] is (member is not None)
    assert dec["first_day"] == rt.FIRST_DAY == "2025-01-09"
    gains = {**BASE_GAINS, **change}
    f4, plac, nodep = gains["F4"][0], gains["F4-shift7"][0], gains["F4-nodepthru"][0]
    assert (dec["gain_s"], dec["placebo_gain_s"], dec["nodepthru_gain_s"]) == (f4, plac, nodep)
    assert dec["c1"] is (f4 >= rt.MIN_GAIN_S and -f4 + gains["F4"][1] < 0)
    assert dec["c2"] is (plac <= rt.PLACEBO_MAX * f4)
    assert dec["c3_nodep_share"] == round(nodep / f4, 3)
    assert dec["missing"] == [k for k, v in gains.items() if v is None]


def test_stage1_the_task_cases_and_the_decision_record(monkeypatch):
    """The pre-registered examples: F4 5 s with its interval below zero, placebo 1 s, nodepthru 4 s: F4; F5 0.6 s
    better: F5; nodepthru 2 s (share 0.4) with its interval below zero: nodepthru; nodepthru 1 s: none; a placebo of
    2 s of a 5 s gain: none; F4 1.5 s: none. And what the read-out records for every run."""
    task = {"F4": (5.0, 1.0), "F4-shift7": (1.0, 1.0), "F4-nodepthru": (4.0, 1.0)}
    expect = [({**task, "F5": (5.0, 1.0)}, "op_h1_f4_adsb"),
              ({**task, "F5": (5.6, 1.0)}, "op_h1_f5_full"),
              ({"F4-nodepthru": (2.0, 1.0)}, "op_h1_f4_nodepthru"),
              ({"F4-nodepthru": (1.0, 0.25)}, None),
              ({"F4-shift7": (2.0, 1.0)}, None),
              ({"F4": (1.5, 0.25)}, None)]
    for change, member in expect:
        dec, _, _ = run_stage1(monkeypatch, {**BASE_GAINS, **change})
        assert dec["member"] == member and dec["passed"] is (member is not None), (change, dec["member"])
    dec, _, _ = run_stage1(monkeypatch, BASE_GAINS)
    rows = {r["row"]: r for r in dec["rows"]}
    assert list(rows) == ["H1", *rt.RUNS_T], "one row per run, H1 first"
    assert rows["H1"]["run"] == _run_name(rt.BASE) and "d_mae" not in rows["H1"]
    assert rows["H1"]["op_mae"] == H1_ABS and rows["F4"]["op_mae"] == H1_ABS - 5.0  # the scored rows: from FIRST_DAY on
    for k, (gain, hw) in BASE_GAINS.items():
        assert rows[k]["run"] == _run_name(rt.RUNS_T[k])
        assert rows[k]["d_mae"] == round(-gain, 2) and rows[k]["mae_ci"] == f"[{-gain - hw:.2f}, {-gain + hw:.2f}]"
    assert dec["F4-norwy_gain_s"] == 4.5 and dec["F5_gain_s"] == 5.25
    assert dec["nodepthru_c1"] is True and dec["f1_candidate"] is False
    dec, _, _ = run_stage1(monkeypatch, {**BASE_GAINS, "F1": (1.0, 0.5)})
    assert dec["f1_candidate"] is True, "F1 (the plans alone) is a candidate at 1 s with its interval below zero"


def test_stage1_compares_only_rows_from_the_first_day_on(monkeypatch):
    """Every run handed over holds 1-8 January, where each rule T run is far worse than H1: with those rows F4 would
    fail c1. Every comparison -- each run's metrics, each paired step -- is handed only rows from `FIRST_DAY` on, and
    F4 passes."""
    dec, seen, ctx = run_stage1(monkeypatch, BASE_GAINS)
    early = ctx["day"] < rt.FIRST_DAY
    assert early.sum() == 8 * ROWS_PER_DAY  # 1-8 January
    h1, f4 = _oof(ctx, 0.0, early=0.0), _oof(ctx, *BASE_GAINS["F4"])
    unfiltered = float(((f4["y_pred"] - f4["y_true"]).abs() - (h1["y_pred"] - h1["y_true"]).abs()).mean())
    assert -unfiltered < rt.MIN_GAIN_S, "the early rows would decide"
    assert len(seen["op_frame"]) == 1 + len(rt.RUNS_T) and len(seen["paired_step"]) == len(rt.RUNS_T)
    assert min(seen["op_frame"] + seen["paired_step"]) >= rt.FIRST_DAY
    assert dec["member"] == "op_h1_f4_adsb" and dec["gain_s"] == 5.0


@pytest.mark.parametrize("missing", ["H1", "F4", "F4-shift7", "F4-nodepthru"])
def test_stage1_without_a_run_it_needs_names_no_member(monkeypatch, missing):
    gains = {**BASE_GAINS, **({} if missing == "H1" else {missing: None})}
    dec, _, _ = run_stage1(monkeypatch, gains, h1=missing != "H1")
    assert dec["passed"] is False and dec["member"] is None and "missing" in dec["note"]
    assert dec["missing"] == [missing]


def _current_hash(name: str) -> str:
    return load_config(f"{name}.yaml").hash()


def _other_hash(h: str) -> str:
    return "deadbeef" if h != "deadbeef" else "feedface"


def test_seed_run_picks_the_newest_finished_run_of_that_config_and_seed(tmp_path, monkeypatch):
    """The newest run directory of exactly that config name at that seed with holdout predictions; never an
    unfinished run, a `prc predict` run, another seed or a config whose name only starts alike."""
    monkeypatch.setattr(paths, "RUNS", tmp_path)

    def run(ts: str, name: str, seed: int = 42, oof: bool = True, predict: bool = False) -> Path:
        h = _current_hash(name) if (paths.CONFIGS / f"{name}.yaml").exists() else "0123abcd"
        d = tmp_path / f"{ts}__{name}__{h}__s{seed}"
        d.mkdir()
        if oof:
            (d / "oof_predictions.parquet").write_bytes(b"PAR1")
        if predict:
            (d / "predict.json").write_text("{}", encoding="utf-8")
        return d

    run("2026-10-12T01-00-00", "op_h1_f4_adsb")
    newest = run("2026-10-12T02-00-00", "op_h1_f4_adsb")
    run("2026-10-12T03-00-00", "op_h1_f4_adsb", oof=False)
    run("2026-10-12T04-00-00", "op_h1_f4_adsb", predict=True)
    other_seed = run("2026-10-12T05-00-00", "op_h1_f4_adsb", seed=43)
    pair = run("2026-10-12T06-00-00", "op_h1_f4_adsb_febdec")
    assert rt.seed_run("op_h1_f4_adsb") == newest
    assert rt.seed_run("op_h1_f4_adsb", 43) == other_seed
    assert rt.seed_run("op_h1_f4_adsb_febdec") == pair
    assert rt.seed_run("op_h1_f4") is None and rt.seed_run("op_h1_f4_nodepthru") is None


def test_seed_run_reads_a_configured_name_at_its_current_hash_only(tmp_path, monkeypatch):
    """A name with a file in configs/ is read at that configuration's current hash. The member's refit, trained with
    ``-o model.refit_on_all=true`` (another hash, a newer stamp), never replaces the holdout run rule T decided on, and
    neither does a run of an older edit of the H1 configuration. A refit run AT the current hash counts: rule T's H1
    reference is the deliverable's own refit run. A name without a configuration file (the member's pair and forward
    runs, named by overrides) matches any hash, the newest run winning."""
    monkeypatch.setattr(paths, "RUNS", tmp_path)

    def run(ts: str, name: str, h: str, refit: bool = False) -> Path:
        d = tmp_path / f"{ts}__{name}__{h}__s42"
        d.mkdir()
        (d / "oof_predictions.parquet").write_bytes(b"PAR1")
        if refit:
            (d / "refit.json").write_text("{}", encoding="utf-8")
            (d / "test_predictions.parquet").write_bytes(b"PAR1")
        return d

    member = "op_h1_f4_nodepthru"
    decided = run("2026-10-13T00-35-59", member, _current_hash(member))
    run("2026-10-13T03-48-19", member, _other_hash(_current_hash(member)), refit=True)  # the refit: newer, other hash
    assert rt.seed_run(member) == decided
    h1 = run("2026-10-11T12-02-25", rt.BASE, _current_hash(rt.BASE), refit=True)
    run("2026-10-12T00-00-00", rt.BASE, _other_hash(_current_hash(rt.BASE)))  # an older edit's hash, a newer stamp
    assert rt.seed_run(rt.BASE) == h1
    only_other = "op_h1_f1_plan"
    run("2026-10-13T06-12-27", only_other, _other_hash(_current_hash(only_other)), refit=True)
    assert rt.seed_run(only_other) is None  # no run at the current hash: missing, never another hash
    for name in (f"{member}_fwd", f"{member}_febdec"):
        assert not (paths.CONFIGS / f"{name}.yaml").exists(), name
        run("2026-10-13T01-29-44", name, "391ccc45")
        newest = run("2026-10-13T09-00-00", name, "aaaaaaaa")
        assert rt.seed_run(name) == newest


# --- stage 2: the month pairs and the forward check -------------------------------------------------------------------

def run_stage2(monkeypatch, tmp_path, t13: dict, month_delta: dict[str, float], fwd_delta: float | None,
               pairs=tuple(PAIR_MONTHS)):
    """`stage2` on synthetic month-pair and forward runs of H1 and of the stage-1 member (F4 when stage 1 named none):
    the member's OP MAE minus H1's is ``month_delta[month]`` in each validation month, ``fwd_delta`` on Nov+Dec."""
    monkeypatch.setattr(paths, "RUNS", tmp_path)
    (tmp_path / "op_decisions.json").write_text(json.dumps({"T13": t13}), encoding="utf-8")
    stem = t13.get("member") or "op_h1_f4_adsb"
    oofs, frames = {}, []
    for k, p in enumerate(pairs):
        ctx = _context(PAIR_MONTHS[p])
        ctx.index = ctx.index + 100_000 * (k + 1)
        frames.append(ctx)
        delta = ctx["month"].map(month_delta).to_numpy()
        oofs[_run_name(f"{rt.BASE}_{p}")] = _oof(ctx, 0.0, early=0.0)
        oofs[_run_name(f"{stem}_{p}")] = _oof(ctx, 0.0, early=0.0).assign(
            y_pred=lambda f, d=delta: f["y_true"] + np.sign(f["y_pred"] - f["y_true"]) * (H1_ABS + d))
    if fwd_delta is not None:
        ctx = _context(("2025-11", "2025-12"))
        ctx.index = ctx.index + 900_000
        frames.append(ctx)
        oofs[_run_name(f"{rt.BASE}_fwd")] = _oof(ctx, 0.0, early=0.0)
        oofs[_run_name(f"{stem}_fwd")] = _oof(ctx, -fwd_delta, early=0.0)
    ladder, _ = _ladder(pd.concat(frames), oofs)
    names = {n.split("__")[1] for n in oofs}
    monkeypatch.setattr(rt, "seed_run", lambda name, seed=42: Path(_run_name(name)) if name in names else None)
    store: dict = {}
    monkeypatch.setitem(sys.modules, "op_ladder_report", ladder)
    monkeypatch.setitem(sys.modules, "op_rules", _rules(store))
    dec = rt.stage2()
    assert store == {"T": json.loads(json.dumps(dec))}
    return dec


MONTHS = [m for p in PAIR_MONTHS.values() for m in p]


@pytest.mark.parametrize("worse, fwd_delta, t13_passed, passed", [
    (2, -1.0, True, True),      # 8 of 10 months better, forward better: the member
    (3, -1.0, True, False),     # 7 of 10
    (2, 0.5, True, False),      # forward worse
    (2, 0.0, True, False),      # forward level: not better
    (0, -1.0, False, False),    # stage 1 failed: never a member
])
def test_stage2_needs_eight_months_the_forward_check_and_stage_1(monkeypatch, tmp_path, worse, fwd_delta, t13_passed,
                                                               passed):
    member = "op_h1_f4_nodepthru"
    t13 = {"passed": t13_passed, "member": member if t13_passed else None, "rows": []}
    delta = {m: (1.5 if i < worse else -2.0) for i, m in enumerate(sorted(MONTHS))}
    dec = run_stage2(monkeypatch, tmp_path, t13, delta, fwd_delta)
    assert dec["months_delta_mae"] == {m: delta[m] for m in sorted(MONTHS)}
    assert dec["months_better"] == 10 - worse and dec["c4"] is (10 - worse >= rt.MIN_MONTHS)
    assert dec["c5"] is (fwd_delta < 0) and dec["forward"]["member"] == H1_ABS + fwd_delta
    assert dec["passed"] is passed and dec["member"] == (member if passed else None)


def test_stage2_a_missing_pair_fails_criterion_4(monkeypatch, tmp_path):
    t13 = {"passed": True, "member": "op_h1_f4_adsb", "rows": []}
    pairs = tuple(PAIR_MONTHS)[:-1]
    dec = run_stage2(monkeypatch, tmp_path, t13, {m: -2.0 for m in MONTHS}, -1.0, pairs=pairs)
    assert len(dec["months_delta_mae"]) == 8 and dec["months_better"] == 8
    assert dec["c4"] is False and dec["passed"] is False and dec["member"] is None
    assert dec["pairs"][-1] == {"pair": "junaug", "missing": [rt.BASE, "op_h1_f4_adsb"]}


# --- the commands -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("doc, passed, member", [
    ({"T13": {"passed": True, "member": "op_h1_f4_adsb"}}, True, "op_h1_f4_adsb"),
    ({"T13": {"passed": True, "member": "op_h1_f4_nodepthru"}}, True, "op_h1_f4_nodepthru"),
    ({"T13": {"passed": False, "member": None}}, False, None),
    ({"L": {"choice": "op_h1_r4_nodoy.yaml"}}, False, None),  # stage 1 not run yet
])
def test_commands_passed13_and_member13_read_the_decisions(tmp_path, monkeypatch, capsys, doc, passed, member):
    """``passed13`` gates the forward check and the month pairs (exit 0 only if criteria 1-3 passed); ``member13``
    prints stage 1's member config (the queue's next steps read it) and exits 1 without one."""
    monkeypatch.setattr(paths, "RUNS", tmp_path)
    (tmp_path / "op_decisions.json").write_text(json.dumps(doc), encoding="utf-8")
    assert rt.main(["passed13"]) == (0 if passed else 1)
    out = capsys.readouterr().out
    assert out.strip() == f"rule T criteria 1-3: {'passed' if passed else 'not passed'}"
    assert rt.main(["member13"]) == (0 if member else 1)
    out = capsys.readouterr()
    assert out.out.strip() == (f"{member}.yaml" if member else "")
    assert ("no member" in out.err) is (member is None)


@pytest.mark.parametrize("doc, member", [
    ({"T": {"passed": True, "member": "op_h1_f5_full"}, "T13": {"passed": True, "member": "op_h1_f5_full"}},
     "op_h1_f5_full"),
    ({"T": {"passed": False, "member": None}, "T13": {"passed": True, "member": "op_h1_f4_adsb"}}, None),
    ({"T13": {"passed": True, "member": "op_h1_f4_adsb"}}, None),  # stage 1 alone is not the rule
])
def test_command_member_reads_the_final_decision(tmp_path, monkeypatch, capsys, doc, member):
    monkeypatch.setattr(paths, "RUNS", tmp_path)
    (tmp_path / "op_decisions.json").write_text(json.dumps(doc), encoding="utf-8")
    assert rt.main(["member"]) == (0 if member else 1)
    cap = capsys.readouterr()
    assert cap.out.strip() == (f"{member}.yaml" if member else "")  # stdout feeds the queue's $(...): nothing on no
    assert ("no member" in cap.err) is (member is None)


def test_command_refits_prints_the_members_two_newest_refitted_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(paths, "RUNS", tmp_path)
    (tmp_path / "op_decisions.json").write_text(json.dumps({"T": {"passed": True, "member": "op_h1_f4_adsb"}}),
                                                encoding="utf-8")
    names = ["2026-10-13T01-00-00__op_h1_f4_adsb__aaaaaaaa__s42", "2026-10-13T02-00-00__op_h1_f4_adsb__aaaaaaaa__s43",
             "2026-10-13T03-00-00__op_h1_f4_adsb__bbbbbbbb__s42", "2026-10-13T04-00-00__op_h1_f4_adsb_fwd__cccccccc__s42"]
    for i, n in enumerate(names):
        d = tmp_path / n
        d.mkdir()
        if i < 2 or i == 3:
            (d / "refit.json").write_text("{}", encoding="utf-8")
            (d / "test_predictions.parquet").write_bytes(b"x")
    assert rt.main(["refits"]) == 0
    assert capsys.readouterr().out.split() == names[:2]  # the run without a refit and the other config never count


def test_stage1_writes_the_decision_the_commands_read(tmp_path, monkeypatch, capsys):
    """End to end on a temporary runs directory: `stage1` through ``main`` records T13 (here with the real
    `op_rules.record` semantics: merged into op_decisions.json) and writes the read-out; ``passed13`` and ``member13``
    then read what it recorded."""
    monkeypatch.setattr(paths, "RUNS", tmp_path)
    monkeypatch.setattr(rt, "OUT_MD", tmp_path / "op_rule_t.md")
    decisions = tmp_path / "op_decisions.json"
    decisions.write_text(json.dumps({"L": {"choice": "op_h1_r4_nodoy.yaml"}}), encoding="utf-8")

    def record(key: str, value: dict) -> None:
        doc = json.loads(decisions.read_text(encoding="utf-8"))
        doc[key] = value
        decisions.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    store: dict = {}
    run_stage1(monkeypatch, {**BASE_GAINS, "F4-nodepthru": (2.0, 1.0)})  # installs the stand-ins
    rules = _rules(store)
    rules.record = record
    monkeypatch.setitem(sys.modules, "op_rules", rules)
    capsys.readouterr()
    assert rt.main(["stage1"]) == 0
    md = (tmp_path / "op_rule_t.md").read_text(encoding="utf-8")
    assert "| F4-nodepthru |" in md and "member op_h1_f4_nodepthru" in md and "passed True" in md
    doc = json.loads(decisions.read_text(encoding="utf-8"))
    assert set(doc) == {"L", "T13"} and doc["T13"]["member"] == "op_h1_f4_nodepthru"
    capsys.readouterr()
    assert rt.main(["passed13"]) == 0 and rt.main(["member13"]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == "op_h1_f4_nodepthru.yaml"
