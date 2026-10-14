"""The ladder read-out's comparability additions (lab/op_ladder_report.py), on synthetic runs only.

Covers the verified literature table, the spread-normalised metrics and the extra tolerances, the bias interval and
the per-airport rung steps, the table baselines computed by the script (B1-fit, B-HAT, B3) with their brute-force
checks, the unimpeded reference, the two lab rungs outside the tripwire and rule T's rungs (F1 among them) inside it
and the row-set check, the runs the rule R rows read, the delivered configurations as defaults (F1 as the H1
deliverable from 13 Oct, findings 10.37, beside the superseded one), and an end-to-end ``main`` with the data loaders
replaced by synthetic frames. No competition data, no network, no GPU.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prc import paths
from prc.models.baselines import GroupMedian


@pytest.fixture(scope="module")
def rep():
    spec = importlib.util.spec_from_file_location("op_ladder_report_t", paths.ROOT / "lab" / "op_ladder_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ctx(n: int, seed: int = 0, apts=("EDDF", "LSZH", "LTFM")) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    ids = pd.Index(np.arange(n), name="mvt_id")
    y = pd.Series(rng.gamma(4, 250, n), index=ids)
    days = pd.Series(pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 40, n), unit="D"), index=ids)
    ctx = pd.DataFrame({"y_true": y, "apt": rng.choice(list(apts), n), "month": days.dt.strftime("%Y-%m"),
                        "day": days.dt.strftime("%Y-%m-%d"), "nm_matched": True, "dayoffset": False,
                        "disrupted": rng.random(n) < 0.05, "h1_early": rng.random(n) < 0.01}, index=ids)
    return ctx, y


# --- metrics ---------------------------------------------------------------------------------------

def test_metric_row_spread_normalised_and_tolerances(rep):
    assert rep.WITHIN == (60, 120, 180, 240, 300, 420, 900)
    y = np.array([100.0, 200.0, 300.0, 1400.0])
    p = np.array([130.0, 80.0, 300.0, 400.0])
    m = rep.metric_row(y, p)
    sd = float(np.std(y))
    assert m["sd_y"] == pytest.approx(sd)
    assert m["mae_sd"] == pytest.approx(m["mae"] / sd) and m["rmse_sd"] == pytest.approx(m["rmse"] / sd)
    assert m["within_240"] == 0.75 and m["within_900"] == 0.75 and m["within_60"] == 0.5
    assert not any("mape" in k for k in m)  # no MAPE, by decision
    empty = rep.metric_row([], [])
    assert empty["n"] == 0 and np.isnan(empty["mae_sd"]) and np.isnan(empty["within_900"])
    flat = rep.metric_row([500, 500], [400, 600])  # no spread: the ratios are undefined, not infinite
    assert np.isnan(flat["mae_sd"]) and np.isnan(flat["rmse_sd"]) and flat["mae"] == 100


# --- paired steps -----------------------------------------------------------------------------------

def test_paired_step_bias_interval_and_airport_filter(rep):
    ctx, y = _ctx(6000, seed=3)
    rng = np.random.default_rng(4)
    a = pd.DataFrame({"y_true": y, "y_pred": y + rng.normal(0, 150, len(y))}, index=y.index)
    b = a.assign(y_pred=a["y_pred"] - 40.0)  # the same model, 40 s lower everywhere
    st = rep.paired_step(a, b, ctx, n_boot=300)
    assert st["d_bias"] == pytest.approx(-40.0)
    assert st["bias_lo"] == pytest.approx(-40.0) and st["bias_hi"] == pytest.approx(-40.0)  # constant: no spread
    # bias interval against a brute-force signed mean on a noisier pair
    c = a.assign(y_pred=y + rng.normal(25, 200, len(y)))
    st2 = rep.paired_step(a, c, ctx, n_boot=400)
    op = y < 3600
    brute = float(((c["y_pred"] - y) - (a["y_pred"] - y))[op].mean())
    assert st2["d_bias"] == pytest.approx(brute)
    assert st2["bias_lo"] <= st2["d_bias"] <= st2["bias_hi"]
    # per airport: only that airport's OP rows and days
    st3 = rep.paired_step(a, c, ctx, n_boot=200, apt="LSZH")
    assert st3["n"] == int((op & (ctx["apt"] == "LSZH")).sum())
    brute3 = float(((c["y_pred"] - a["y_pred"])[op & (ctx["apt"] == "LSZH")]).mean())
    assert st3["d_bias"] == pytest.approx(brute3)
    assert rep.paired_step(a, c, ctx, apt="EGLL") is None


def test_rung_steps_overall_and_per_airport_include_the_hindsight_step(rep):
    ctx, y = _ctx(3000, seed=5)
    rng = np.random.default_rng(6)
    oof = lambda noise: pd.DataFrame({"y_true": y, "y_pred": y + rng.normal(0, noise, len(y))}, index=y.index)  # noqa: E731
    oofs = {"op_h1_r2": oof(260), "op_h1_r3": oof(230), "op_h0_r5": oof(180), "op_h0_r5_hind": oof(90),
            "op_h1_r4_mae": oof(150)}
    steps, per_apt = rep.rung_steps(oofs, ctx, n_boot=100)
    pairs = {(s["from"], s["to"]) for s in steps}
    assert pairs == {("op_h1_r2", "op_h1_r3"), ("op_h0_r5", "op_h0_r5_hind")}  # the MAE rung is in no step
    assert {(s["from"], s["to"], s["apt"]) for s in per_apt} == {
        (a, b, apt) for a, b in pairs for apt in ("EDDF", "LSZH", "LTFM")}
    hind = [s for s in steps if s["to"] == "op_h0_r5_hind"][0]
    assert hind["d_mae"] < 0 and hind["mae_hi"] < 0
    for a_, b_ in pairs:  # the airport steps add up to the overall row count
        assert sum(s["n"] for s in per_apt if (s["from"], s["to"]) == (a_, b_)) == [
            s for s in steps if (s["from"], s["to"]) == (a_, b_)][0]["n"]


# --- table baselines --------------------------------------------------------------------------------

def _baseline_world(seed: int = 7, n: int = 30_000) -> pd.DataFrame:
    """2025 departures: taxi = stand base + an hour effect + noise; flight numbers repeat daily at the same stand."""
    rng = np.random.default_rng(seed)
    ids = pd.Index(np.arange(10_000, 10_000 + n), name="mvt_id")
    apt = rng.choice(["EDDF", "LSZH"], n)
    flight_no = rng.integers(0, 300, n)
    stand = np.array([f"S{f % 40}" for f in flight_no])
    runway = rng.choice(["07", "25"], n)
    hour = (flight_no * 7) % 24  # a flight keeps its hour
    t0 = pd.Timestamp("2025-01-01", tz="UTC").timestamp()
    day = rng.integers(0, 365, n)
    t = t0 + day * 86400.0 + hour * 3600.0 + rng.integers(0, 3000, n)
    anchor = t - 1500.0 - rng.integers(0, 600, n)
    stand_base = 400 + 20 * (flight_no % 40)
    y = stand_base + np.where(hour >= 17, 300.0, 0.0) + rng.normal(0, 60, n)
    y[rng.random(n) < 0.002] = 50_000.0  # stale records: never in a fit
    month = pd.to_datetime(t, unit="s", utc=True).month
    flight = pd.Series([f"XY{f}" for f in flight_no], index=ids, dtype="string")
    flight.iloc[:50] = pd.NA
    return pd.DataFrame({"apt": apt, "stand": pd.Series(stand, index=ids, dtype="string"),
                         "runway": pd.Series(runway, index=ids, dtype="string"), "flight": flight,
                         "hour": pd.Series(hour, index=ids, dtype="Int64"), "t": t, "anchor": anchor,
                         "month": np.asarray(month), "y": y}, index=ids)


def test_chain_quantile_reproduces_group_median(rep):
    w = _baseline_world(n=8000)
    w.loc[w.index[:30], "stand"] = "RARE"  # below the support: falls back a level
    fit, q = w.iloc[:6000], w.iloc[6000:]
    levels = [list(lv) for lv in rep.B1_CHAIN]
    gm = GroupMedian(keys=levels, min_support=rep.MIN_SUPPORT).fit(fit, fit["y"])
    np.testing.assert_allclose(rep.chain_quantile(fit, q, rep.B1_CHAIN, 0.5).to_numpy(), gm.predict(q), rtol=0, atol=1e-9)


def _history_brute(w: pd.DataFrame, ids, max_target: float, days: int) -> pd.Series:
    out = {}
    for i in ids:
        r = w.loc[i]
        if pd.isna(r["flight"]):
            out[i] = np.nan
            continue
        h = w[(w["apt"] == r["apt"]) & (w["flight"] == r["flight"]) & (w["y"] < max_target) & (w.index != i)
              & (w["t"] >= np.floor(r["anchor"]) - days * 86400) & (w["t"] < np.floor(r["anchor"]))]
        out[i] = h["y"].mean() if len(h) else np.nan
    return pd.Series(out, dtype="float64")


def test_flight_history_mean_matches_brute_force(rep):
    w = _baseline_world(n=6000)
    # a flight whose own takeoff precedes its own anchor: it must not count itself
    j = w.index[100]
    w.loc[j, "anchor"] = w.loc[j, "t"] + 60.0
    rng = np.random.default_rng(0)
    ids = pd.Index(np.r_[w.index[:60], [j], rng.choice(w.index, 300, replace=False)]).unique()
    got = rep.flight_history_mean(w, ids, 7200.0, 14)
    want = _history_brute(w, ids, 7200.0, 14)
    np.testing.assert_allclose(got.to_numpy(), want.reindex(ids).to_numpy(), rtol=1e-12, equal_nan=True)
    assert got.loc[w.index[:50]].isna().all()  # no flight number
    assert got.notna().sum() > 50  # the comparison is not vacuous


def test_table_baselines_fit_rows_hour_effect_and_fallback(rep):
    w = _baseline_world()
    hold = w.index[np.isin(w["month"], rep.HOLDOUT_MONTHS) & (w["y"] < 7200)]
    preds, unimp = rep.table_baselines(w, hold)
    assert set(preds) == {"B1-fit", "B-HAT", "B3"}
    # holdout rows never enter the fit tables: poison them and nothing moves
    w2 = w.copy()
    w2.loc[hold, "y"] = w2.loc[hold, "y"] + 5000.0
    preds2, unimp2 = rep.table_baselines(w2, hold)
    for k in ("B1-fit", "B-HAT"):
        np.testing.assert_allclose(preds2[k]["y_pred"].to_numpy(), preds[k]["y_pred"].to_numpy())
    np.testing.assert_allclose(unimp2.to_numpy(), unimp.to_numpy())
    # the hour term carries the evening effect the stand x runway table cannot see
    mae = {k: float((v["y_pred"] - v["y_true"]).abs().mean()) for k, v in preds.items()}
    assert mae["B-HAT"] < mae["B1-fit"] - 50
    # B3 is the flight's history where it has one, B1-fit elsewhere
    hist = rep.flight_history_mean(w, hold)
    b3 = preds["B3"]["y_pred"]
    np.testing.assert_allclose(b3[hist.notna()], hist[hist.notna()])
    np.testing.assert_allclose(b3[hist.isna()], preds["B1-fit"]["y_pred"][hist.isna()])
    # the unimpeded reference is a low quantile: below the median table
    assert (unimp <= preds["B1-fit"]["y_pred"] + 1e-9).all()


def test_unimpeded_table_reports_additional_time_per_airport(rep):
    ctx, y = _ctx(2000, seed=8)
    ref = pd.Series(400.0, index=y.index)
    tab = rep.unimpeded_table(ref, ctx)
    assert list(tab.index) == ["EDDF", "LSZH", "LTFM", "all"]
    op = y < 3600
    assert tab.loc["all", "n"] == int(op.sum())
    assert tab.loc["all", "mean_additional"] == pytest.approx(float(y[op].mean() - 400.0))
    assert tab.loc["LSZH", "median_target"] == pytest.approx(float(y[op & (ctx["apt"] == "LSZH")].median()))


# --- tripwire, the lab rungs and the markdown -------------------------------------------------------

def test_lab_rungs_are_reported_but_outside_the_tripwire(rep):
    names = [n for n, *_ in rep.OP_CONFIGS]
    assert "op_h0_r5_hind" in names and "op_h1_r4_mae" in names
    assert "op_h0_r5_hind" in rep.H0_RUNGS and ("op_h0_r5", "op_h0_r5_hind") in rep.STEPS
    assert not [s for s in rep.STEPS if "op_h1_r4_mae" in s]
    hind_title = dict((n, t) for n, t, _ in rep.OP_CONFIGS)["op_h0_r5_hind"]
    assert "NOT operational" in hind_title and "HINDSIGHT" in hind_title
    ctx, y = _ctx(4000, seed=9)
    rng = np.random.default_rng(10)
    oof = lambda noise: pd.DataFrame({"y_true": y, "y_pred": y + rng.normal(0, noise, len(y))}, index=y.index)  # noqa: E731
    oofs = {"op_h1_r3": oof(220), "op_h1_r4_mae": oof(20), "op_h0_r5": oof(180), "op_h0_r5_hind": oof(30),
            "C1": oof(100), "C2": oof(90)}
    rows = []
    for k, o in oofs.items():
        rows += rep.score(k, o, ctx)
    table = pd.DataFrame(rows)
    assert set(table.loc[table["row"] == "op_h0_r5_hind", "population"]) == {"OP", "H1_SENS"}
    trip = "\n".join(rep.tripwire(table))
    assert "op_h1_r4_mae" not in trip and "op_h0_r5_hind" not in trip  # far better than C1, but not a leak signal
    assert "- op_h1_r3:" in trip


def test_rule_t_rungs_and_f1_are_in_the_tripwire_and_the_row_set_check(rep):
    """findings 10.37: F1 (op_h1_f1_plan), the H1 deliverable since 13 Oct, sits in rule T's table, but rule T's rungs
    score the Jan+Jul holdout, so the leak tripwire and the check that a rung scored exactly C1's rows read them like
    the ladder's rungs. The forward rungs score another holdout and stay outside both."""
    assert set(rep.OWN_TABLE_JANJUL) == set(rep.RULE_T) and set(rep.RULE_T) <= set(rep.OTHER_HOLDOUT)
    assert not set(rep.OWN_TABLE_JANJUL) & set(rep.FORWARD)
    ctx, y = _ctx(4000, seed=14)
    rng = np.random.default_rng(15)
    oof = lambda noise: pd.DataFrame({"y_true": y, "y_pred": y + rng.normal(0, noise, len(y))}, index=y.index)  # noqa: E731
    c1, f1, f2, fwd = oof(100), oof(5), oof(220), oof(5)
    rows = []
    for k, o in {"C1": c1, "op_h1_f1_plan": f1, "op_h1_f2_rec": f2, "op_h1_f1_plan_fwd": fwd}.items():
        rows += rep.score(k, o, ctx)
    trip = rep.tripwire(pd.DataFrame(rows))
    f1_lines = [ln for ln in trip if ln.startswith("- op_h1_f1_plan:")]
    assert len(f1_lines) == 1 and "LEAK" in f1_lines[0]  # far better than C1: a leak in the deliverable is caught
    assert "ok (below C1)" in [ln for ln in trip if ln.startswith("- op_h1_f2_rec:")][0]
    assert not [ln for ln in trip if "op_h1_f1_plan_fwd" in ln]  # another holdout: never against C1
    warnings = rep.row_set_warnings({"C1": c1, "op_h1_f1_plan": f1.iloc[:-10], "op_h1_f2_rec": f2,
                                     "op_h1_f1_plan_fwd": fwd.iloc[:-10]})
    assert len(warnings) == 1 and warnings[0].startswith("op_h1_f1_plan scored 3,990 rows, C1 4,000")


def test_find_run_does_not_confuse_a_rung_with_its_lab_variant(rep, tmp_path):
    ids = pd.Index(np.arange(10), name="mvt_id")
    for name in ("2026-10-08T01-00-00__op_h0_r5__aaaa__s42", "2026-10-08T05-00-00__op_h0_r5_hind__bbbb__s42"):
        (tmp_path / name).mkdir()
        pd.DataFrame({"y_true": 1.0, "y_pred": 1.0}, index=ids).to_parquet(tmp_path / name / "oof_predictions.parquet")
    assert rep.find_run([tmp_path], "op_h0_r5", by_name=True).name.split("__")[1] == "op_h0_r5"
    assert rep.find_run([tmp_path], "op_h0_r5_hind", by_name=True).name.split("__")[1] == "op_h0_r5_hind"


def test_rule_r_rows_read_the_runs_rule_r_decided_on(rep, tmp_path, monkeypatch):
    """findings 10.35, rule R: the rows labelled rule R are the seed-42 runs it decided on. The deliverables' seed-42
    refits share the config name, hash and seed and are newer, so the row reads the run runs/op_decisions.json records;
    without the record, or when the recorded run is not under the roots, the newest seed-42 run at the current hash. A
    ``--run`` pin wins over both."""
    monkeypatch.setattr(paths, "RUNS", tmp_path)
    assert {"op_hl3_r3_rwyblind", "op_hl3_r4_rwyblind", "op_hl24_r3_rwyblind", "op_hl24_r4_rwyblind"} <= set(rep.SEED42)
    names = [n for n, *_ in rep.OP_CONFIGS]
    assert set(rep.RULE_R_ROWS) <= set(names)
    assert {n for n, t, _ in rep.OP_CONFIGS if "rule R" in t} == set(rep.RULE_R_ROWS)
    name = "op_hl3_r4_rwyblind"
    h = rep.config_hash(name)

    def run(stamp: str, seed: int) -> Path:
        d = tmp_path / f"{stamp}__{name}__{h}__s{seed}"
        d.mkdir()
        (d / "oof_predictions.parquet").write_bytes(b"PAR1")
        return d

    decided = run("2026-10-11T14-18-57", 42)
    refit42 = run("2026-10-11T15-20-52", 42)
    run("2026-10-11T15-38-22", 43)
    assert rep.rule_r_runs() == {}
    assert rep.rung_dir([tmp_path], name, {}, {}) == refit42  # no record: the newest seed-42 run, never seed 43
    (tmp_path / "op_decisions.json").write_text(json.dumps(
        {"R_hl3": {"choice": f"{name}.yaml", "main": {"r4": {"run": decided.name}}}}), encoding="utf-8")
    recorded = rep.rule_r_runs()
    assert recorded == {name: decided.name}
    assert rep.rung_dir([tmp_path], name, {}, recorded) == decided
    assert rep.rung_dir([tmp_path], name, {name: refit42.name}, recorded) == refit42
    assert rep.rung_dir([tmp_path], name, {}, {name: "2026-10-11T00-00-00__gone__00000000__s42"}) == refit42


def test_the_24h_default_is_the_delivered_configuration_and_rule_t_rows_say_their_rows(rep, tmp_path, monkeypatch):
    """Without rule R's decision file the 24 h deliverable rows read the delivered configuration (r3); the file, when
    present, decides. The rule T table says its rows and steps cover every OP row from 1 January, unlike rule T's own
    figures (from 9 January)."""
    monkeypatch.setattr(paths, "RUNS", tmp_path)
    assert rep.DEFAULT_CONFIG == {"h1": "op_h1_f1_plan", "h1_r4_nodoy": "op_h1_r4_nodoy", "hl3": "op_hl3_r4_rwyblind",
                                  "hl24": "op_hl24_r3_rwyblind"}
    assert rep.deliverable_config("hl24") == "op_hl24_r3_rwyblind"
    (tmp_path / "op_decision_R_hl24.cfg").write_text("op_hl24_r4_rwyblind.yaml\n", encoding="utf-8")
    assert rep.deliverable_config("hl24") == "op_hl24_r4_rwyblind"
    title = next(t for t, names in rep.OTHER_TABLES if names == rep.RULE_T)
    assert "every OP row from 1 January" in title and "scored from 9 January" in title
    assert "runs/op_rule_t.md" in title and "planned in the concept, realised in the challenge data" in title


def test_the_h1_deliverable_is_f1_whatever_rule_l_recorded(rep, tmp_path, monkeypatch):
    """findings 10.37: F1 (op_h1_f1_plan) is the H1 deliverable from 13 Oct. It extends rule L's choice, so rule L's
    file (runs/op_decision_L.cfg, which names op_h1_r4_nodoy.yaml) no longer decides the H1 rows; the superseded
    deliverable keeps its own rows (D-H1-R4, with its calibration tables and median member) and the adoption's paired
    step, and neither reads the file. F1's rule T row reads seed 42 of its refits and says it was adopted."""
    monkeypatch.setattr(paths, "RUNS", tmp_path)
    assert rep.deliverable_config("h1") == "op_h1_f1_plan"
    assert rep.deliverable_config("h1_r4_nodoy") == "op_h1_r4_nodoy"
    (tmp_path / "op_decision_L.cfg").write_text("op_h1_r4_nodoy.yaml\n", encoding="utf-8")
    assert rep.deliverable_config("h1") == "op_h1_f1_plan"
    (tmp_path / "op_decision_L.cfg").write_text("op_h1_f1_plan.yaml\n", encoding="utf-8")  # a rewritten file
    assert rep.deliverable_config("h1_r4_nodoy") == "op_h1_r4_nodoy"
    assert rep.DEFAULT_TABLES["h1"] == "runs/op_calib/h1_f1_pairs_lr08.parquet"
    assert rep.DEFAULT_TABLES["h1_r4_nodoy"] == "runs/op_calib/h1_nodoy_pairs_lr08.parquet"
    assert rep.MEDIAN_MEMBER == {"h1": "op_h1_f1_plan_mae_exact_cpu", "h1_r4_nodoy": "op_h1_r4_nodoy_mae_exact_cpu"}
    members = {label: (member, n, title) for label, member, n, title in rep.DELIVERABLES}
    assert members["D-H1"][:2] == ("h1", 3) and members["D-H1-R4"][:2] == ("h1_r4_nodoy", 3)
    assert "H1 deliverable until 13 Oct, superseded by F1 (findings 10.37)" in members["D-H1-R4"][2]
    assert set(rep.DEFAULT_CONFIG) == set(rep.DEFAULT_TABLES) == set(rep.DELIVERABLE_TIERS) == {
        m for m, *_ in members.values()}
    for step in (("D-H1-R4", "D-H1"), ("D-H1-R4", "D-H1-R4-MED"), ("D-H1-R4-MED", "D-H1-R4-MAE"),
                 ("D-H1-R4", "op_h0_r4_nodoy"), ("op_h1_f1_plan", "op_h1_f1_plan_mae_exact_cpu"),
                 ("op_h1_r4_nodoy_fwd", "op_h1_f1_plan_fwd")):
        assert step in rep.STEPS, step
    # H0 (op_h0_r4_nodoy) differs from the superseded deliverable by the moment alone (10.35), from F1 also by F1's
    # plans after the anchor: only the like-for-like step is drawn
    assert ("D-H1", "op_h0_r4_nodoy") not in rep.STEPS
    titles = {n: t for n, t, _ in rep.OP_CONFIGS}
    assert "adopted as the H1 deliverable" in titles["op_h1_f1_plan"] and "findings 10.37" in titles["op_h1_f1_plan"]
    assert titles["op_h1_f1_plan"].startswith("F1: op_h1_r4_nodoy (the H1 deliverable until 13 Oct) + ")
    assert "op_h1_f1_plan" in rep.SEED42 and "op_h1_f1_plan" in rep.RULE_T and "op_h1_f1_plan_fwd" in rep.FORWARD
    for name in (*rep.DEFAULT_CONFIG.values(), *rep.MEDIAN_MEMBER.values(), "op_h1_f1_plan_fwd"):
        assert (paths.CONFIGS / f"{name}.yaml").exists(), name


def test_the_adsb_rungs_are_reported_with_their_steps(rep):
    """The ADS-B rungs (findings 10.31) are in the report: R6 and its placebo next to R4, the public-data variant next
    to R2, H0 r4 with and without ADS-B as H0 rungs, the paired steps between them; every step's ends are rungs or
    comparison rows, and every rung config exists."""
    names = [n for n, *_ in rep.OP_CONFIGS]
    for n in ("op_h1_r6", "op_h1_r6_shift7", "op_h1_r2_adsb", "op_h0_r4", "op_h0_r4_adsb"):
        assert n in names and (paths.CONFIGS / f"{n}.yaml").exists(), n
    assert {"op_h0_r4", "op_h0_r4_adsb"} <= set(rep.H0_RUNGS)
    for step in (("op_h1_r4", "op_h1_r6"), ("op_h1_r4", "op_h1_r6_shift7"), ("op_h1_r2", "op_h1_r2_adsb"),
                 ("op_h0_r4", "op_h0_r4_adsb"), ("op_h0_r4_adsb", "op_h0_r5")):
        assert step in rep.STEPS, step
    labels = set(names) | {c[0] for c in rep.COMPARISON} | {x for d in rep.DELIVERABLES for x in (d[0], d[0] + "-MED")}
    labels |= {d[0] + "-MAE" for d in rep.DELIVERABLES if d[1] in rep.MEDIAN_MEMBER}
    assert all(m in names for m in rep.MEDIAN_MEMBER.values())
    assert all(a in labels and b in labels for a, b in rep.STEPS)


def _cells(line: str) -> int:
    return len(line.strip().strip("|").split("|"))


def test_literature_table_is_the_verified_one(rep):
    lit = rep.LITERATURE
    rows = [ln for ln in lit.splitlines() if ln.startswith("|")]
    assert len({_cells(ln) for ln in rows}) == 1  # no stray pipe inside a cell
    body = "\n".join(rows)
    for needle in ("Idris", "4.56", "5.69", "0.41", "0.52", "65.6", "53.7",  # the H0 comparator
                   "MAN / ZRH / HKG", "3.50 / 3.27 / 4.06", "3.43 / 3.16 / 3.28", "3.37 / 3.10 / 3.10",
                   "2.96 / 2.15 / 2.07", "34.1 / 74.6 / 89.8", "line-up", "pooled", "random 70/30",
                   "4th at MAN and HKG, 8th at ZRH",
                   "Table V", "-35 %", "-1 %", "-6 %", "APDF", "year-ahead",
                   "Herrema", "ATOT", "AOBT", "random 70/15/15",
                   "Wu et al. 2026", "140.5", "Tang et al. 2025", "1.561", "random 8:2", "random 9:1",
                   "EUROCONTROL 2017", "+-7 / +-5 / +-2 min"):
        assert needle in body, needle
    for study in ("Guenther 2011", "Lim et al. 2024"):
        line = [ln for ln in rows if study in ln][0]
        assert "not verified: full text not read" in line
    assert "heading to stand or runway" not in lit  # the old misdescription of Wang's Group B
    # every result row carries a non-empty caveat
    for ln in rows[2:]:
        assert ln.strip().strip("|").split("|")[-1].strip() not in ("", "-")


def test_markdown_carries_the_new_tables_with_consistent_columns(rep):
    ctx, y = _ctx(5000, seed=11)
    rng = np.random.default_rng(12)
    oof = lambda noise: pd.DataFrame({"y_true": y, "y_pred": y + rng.normal(0, noise, len(y))}, index=y.index)  # noqa: E731
    oofs = {"B1": oof(300), "B-HAT": oof(280), "op_h1_r2": oof(250), "op_h1_r3": oof(220), "op_h0_r5": oof(180),
            "op_h0_r5_hind": oof(90), "op_h1_r4_mae": oof(200), "C1": oof(100)}
    rows = []
    for k, o in oofs.items():
        rows += rep.score(k, o, ctx)
    table = pd.DataFrame(rows)
    steps, per_apt = rep.rung_steps(oofs, ctx, n_boot=100)
    unimp = rep.unimpeded_table(pd.Series(350.0, index=y.index), ctx)
    meta = {k: {"title": k, "run": k, "tiers": "O"} for k in oofs}
    md = rep.markdown(table, meta, rep.tripwire(table), steps, None, per_apt, unimp)
    for needle in ("## Rung steps per airport", "mean signed error change (s)", "OP MAE/SD", "+-240 s", "+-900 s",
                   "## OP MAE/SD per airport", "## OP target SD (s) per airport", "## OP MAE minus B1 per airport",
                   "## Unimpeded reference (not a scored row)", "vs Idris 2002", "LAB HINDSIGHT", "Literature anchors"):
        assert needle in md, needle
    assert "MAPE" not in md
    # every markdown table has as many cells per row as its header
    blocks, cur = [], []
    for ln in md.splitlines():
        if ln.startswith("|"):
            cur.append(ln)
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    for b in blocks:
        assert len({_cells(ln) for ln in b}) == 1, b[:3]
    # MAE minus B1 is the difference of the per-airport MAEs
    sub = table[(table["population"] == "OP") & (table["slice"] == "apt")].pivot_table(
        index="level", columns="row", values="mae", aggfunc="first")
    sect = md.split("## OP MAE minus B1 per airport")[1].split("\n## ")[0]
    line = [ln for ln in sect.splitlines() if ln.startswith("| LSZH |")][0].split("|")
    header = [h.strip() for h in [ln for ln in sect.splitlines() if ln.startswith("| airport |")][0].split("|")]
    col = header.index("C1")
    assert float(line[col]) == pytest.approx(sub.loc["LSZH", "C1"] - sub.loc["LSZH", "B1"], abs=0.051)
    # the per-airport step lines carry the bias interval
    lsz = [ln for ln in md.splitlines() if ln.startswith("| op_h0_r5 | op_h0_r5_hind | LSZH |")]
    assert len(lsz) == 1 and lsz[0].count("[") == 3


# --- end to end with the data loaders replaced ------------------------------------------------------

def test_main_end_to_end_on_synthetic_runs(rep, tmp_path, monkeypatch):
    w = _baseline_world(n=12_000)
    hold = w.index[np.isin(w["month"], rep.HOLDOUT_MONTHS) & (w["y"] < 3600)]
    y = w.loc[hold, "y"]
    rng = np.random.default_rng(13)
    days = pd.to_datetime(w.loc[hold, "t"], unit="s", utc=True)
    ctx = pd.DataFrame({"y_true": y, "apt": w.loc[hold, "apt"], "month": days.dt.strftime("%Y-%m"),
                        "day": days.dt.strftime("%Y-%m-%d"), "nm_matched": True, "dayoffset": False,
                        "disrupted": False, "h1_early": False}, index=hold)
    monkeypatch.setattr(rep, "holdout_context", lambda ids, disrupted: ctx.reindex(ids))
    monkeypatch.setattr(rep, "baseline_frame", lambda: w)
    runs = tmp_path / "runs"
    c1_id = [r for r in rep.COMPARISON if r[0] == "C1"][0][1]
    b1_id = [r for r in rep.COMPARISON if r[0] == "B1"][0][1]
    dirs = {"C1": c1_id, "B1": b1_id}
    for name in ("op_h1_r3", "op_h1_r4", "op_h0_r5", "op_h0_r5_hind", "op_h1_r4_mae"):
        dirs[name] = f"2026-10-08T03-00-00__{name}__{rep.config_hash(name) or 'nohash'}__s42"
    for k, d in dirs.items():
        (runs / d).mkdir(parents=True)
        pd.DataFrame({"y_true": y, "y_pred": y + rng.normal(0, 150, len(y))}, index=hold).to_parquet(
            runs / d / "oof_predictions.parquet")
    out = tmp_path / "docs" / "operational_ladder"
    assert rep.main(["--runs-root", str(runs), "--out", str(out)]) == 0
    md = out.with_suffix(".md").read_text(encoding="utf-8")
    for label in ("B1-fit", "B-HAT", "B3", "op_h0_r5_hind", "op_h1_r4_mae"):
        assert f"| {label} |" in md, label
    table = pd.read_csv(out.with_suffix(".csv"))
    assert {"B1-fit", "B-HAT", "B3"} <= set(table["row"])
    assert (table.loc[table["row"] == "B3", "run_id"] == "computed in this report").all()
    assert {"sd_y", "mae_sd", "rmse_sd", "within_240", "within_900"} <= set(table.columns)
    steps = pd.read_csv(out.parent / "operational_ladder_steps.csv")
    assert {"d_bias", "bias_lo", "bias_hi", "apt"} <= set(steps.columns)
    assert ((steps["from"] == "op_h0_r5") & (steps["to"] == "op_h0_r5_hind") & (steps["apt"] == "all")).sum() == 1
    unimp = pd.read_csv(out.parent / "operational_ladder_unimpeded.csv", index_col=0)
    assert "all" in unimp.index
    # --no-table-baselines leaves them out
    out2 = tmp_path / "report2" / "operational_ladder"
    assert rep.main(["--runs-root", str(runs), "--out", str(out2), "--no-table-baselines"]) == 0
    assert "| B-HAT |" not in out2.with_suffix(".md").read_text(encoding="utf-8")


def test_deliverable_rows_seed_mean_median_and_month_pairs(rep, tmp_path, monkeypatch, capsys):
    """findings 10.35 and 10.37: a deliverable row is the mean of its seeds at the current hash (an older hash and a run
    before the schedule clock are ignored), its -MED row applies the calibration tables, its -MAE row clamps rule M's
    candidate into their band, and the sections for the seeds, rule C and the twelve months are written. The H1 row is
    F1's seeds; the H1 deliverable until 13 Oct keeps its own rows (D-H1-R4, -MED, -MAE) and the adoption's step to
    D-H1. Without F1's tables (fitted after its month pairs) the report still runs, says so, and draws F1's mean
    only."""
    import sys

    sys.path.insert(0, str(paths.ROOT / "lab"))
    import op_calibrate

    w = _baseline_world(n=12_000)
    hold = w.index[np.isin(w["month"], rep.HOLDOUT_MONTHS) & (w["y"] < 3600)]
    y = w.loc[hold, "y"]
    rng = np.random.default_rng(21)
    days = pd.to_datetime(w.loc[hold, "t"], unit="s", utc=True)
    ctx = pd.DataFrame({"y_true": y, "apt": w.loc[hold, "apt"], "month": days.dt.strftime("%Y-%m"),
                        "day": days.dt.strftime("%Y-%m-%d"), "nm_matched": True, "dayoffset": False,
                        "disrupted": False, "h1_early": False}, index=hold)
    monkeypatch.setattr(rep, "holdout_context", lambda ids, disrupted: ctx.reindex(ids))
    monkeypatch.setattr(rep, "month_pair_mae", lambda roots, name: {"2025-02": {"mae": 150.0, "within_120": 0.5}})
    monkeypatch.setattr(rep, "deliverable_config", lambda member: rep.DEFAULT_CONFIG[member])
    runs = tmp_path / "runs"
    monkeypatch.setattr(paths, "RUNS", runs)  # no decision file of the checkout's runs/ is read

    def write(name: str, stamp: str, seed: int, pred, hh: str | None = None) -> Path:
        d = runs / f"{stamp}__{name}__{hh or rep.config_hash(name)}__s{seed}"
        d.mkdir(parents=True)
        pd.DataFrame({"y_true": y, "y_pred": pred}, index=hold).to_parquet(d / "oof_predictions.parquet")
        return d

    def tables_for(pred, file: str) -> Path:
        tpath = tmp_path / file
        op_calibrate.fit_tables(y, pred, w.loc[hold, "apt"]).to_parquet(tpath, index=False)
        return tpath

    def far(pred) -> np.ndarray:
        """Rule M's candidate: the seed mean, but 3,000 s too long on every tenth row (clamping pulls it back)."""
        return np.asarray(pred, dtype="float64") + np.where(np.arange(len(pred)) % 10 == 0, 3000.0, 0.0)

    name = rep.DEFAULT_CONFIG["h1"]
    assert name == "op_h1_f1_plan"
    h = rep.config_hash(name)
    preds, dirs = {}, {}
    for stamp, seed, hh in (("2026-10-11T12-02-25", 42, h), ("2026-10-11T12-40-00", 43, h),
                            ("2026-10-11T13-20-00", 44, h), ("2026-10-09T22-53-13", 45, "deadbeef")):
        preds[seed] = y + 60 + rng.gamma(2.0, 60, len(y)) - 120
        dirs[seed] = write(name, stamp, seed, preds[seed], hh)
    found = rep.seed_runs([runs], name, 3)
    assert [d.name.split("__")[3] for d in found] == ["s42", "s43", "s44"]
    assert rep.seed_runs([runs], name, 4) == []
    mean, frames = rep.seed_mean(found)
    assert np.allclose(mean["y_pred"].to_numpy(), np.mean([preds[s] for s in (42, 43, 44)], axis=0))
    mm = rep.MEDIAN_MEMBER["h1"]
    write(mm, "2026-10-13T14-00-00", 42, far(mean["y_pred"]))
    tpath = tables_for(mean["y_pred"], "h1.parquet")
    # the H1 deliverable until 13 Oct: three refitted seeds of op_h1_r4_nodoy, each worse than F1's
    old = rep.DEFAULT_CONFIG["h1_r4_nodoy"]
    assert old == "op_h1_r4_nodoy"
    old_preds = [y + 60 + rng.gamma(2.0, 90, len(y)) - 120 for _ in range(3)]
    for (stamp, seed), p in zip((("2026-10-11T12-02-25", 42), ("2026-10-11T12-35-59", 43),
                                 ("2026-10-11T13-09-32", 44)), old_preds):
        write(old, stamp, seed, p)
    old_mean = np.mean(old_preds, axis=0)
    mm_old = rep.MEDIAN_MEMBER["h1_r4_nodoy"]
    write(mm_old, "2026-10-12T14-00-49", 42, far(old_mean))
    tpath_old = tables_for(old_mean, "h1_r4_nodoy.parquet")

    out = tmp_path / "docs" / "operational_ladder"
    assert rep.main(["--runs-root", str(runs), "--out", str(out), "--no-table-baselines",
                     "--tables", f"h1={tpath}", f"h1_r4_nodoy={tpath_old}"]) == 0
    md = out.with_suffix(".md").read_text(encoding="utf-8")
    assert "H1 deliverable (headline): `op_h1_f1_plan`, 3 seeds averaged" in md and "(calibrated median)" in md
    assert "H1 deliverable until 13 Oct, superseded by F1 (findings 10.37): `op_h1_r4_nodoy`, 3 seeds averaged" in md
    assert "## The deliverables' seeds" in md and "spread of the seeds' MAE" in md
    assert "## Rule C: the calibrated median" in md and "## Every month of 2025" in md
    assert "| month | D-H1 | D-H1-R4 |" in md  # F1's months beside the superseded deliverable's
    assert "deadbeef" not in md
    table = pd.read_csv(out.with_suffix(".csv"))
    assert {"D-H1", "D-H1-MED", "D-H1-MAE", mm, "D-H1-R4", "D-H1-R4-MED", "D-H1-R4-MAE", mm_old} <= set(table["row"])
    assert set(table.loc[table["row"] == name, "run_id"]) == {dirs[42].name}  # F1's rung row: seed 42, not the newest
    op_all = table[(table["population"] == "OP") & (table["slice"] == "all")].set_index("row")
    assert op_all.loc["D-H1-MAE", "mae"] < op_all.loc[mm, "mae"]  # the far rows were clamped into the band
    assert op_all.loc["D-H1-R4-MAE", "mae"] < op_all.loc[mm_old, "mae"]
    assert op_all.loc["D-H1-R4", "mae"] == pytest.approx(float(np.abs(old_mean - y.to_numpy()).mean()))
    assert "## Rule M: the median member" in md
    # the -MAE row is drawn whatever rules C and M decided; the text says what the file then holds
    assert "the file's median if rules C and M passed" in md
    assert "Where rule C did not pass, the file holds the mean only" in md
    steps = pd.read_csv(out.parent / "operational_ladder_steps.csv")
    overall = steps[steps["apt"] == "all"].set_index(["from", "to"])
    for step in (("D-H1", "D-H1-MED"), ("D-H1-MED", "D-H1-MAE"), ("D-H1-R4", "D-H1"), ("D-H1-R4", "D-H1-R4-MED"),
                 ("D-H1-R4-MED", "D-H1-R4-MAE")):
        assert step in overall.index, step
    adoption = overall.loc[("D-H1-R4", "D-H1")]
    assert adoption["d_mae"] < 0 and adoption["mae_lo"] <= adoption["d_mae"] <= adoption["mae_hi"] < 0
    assert ((steps["from"] == "D-H1-R4") & (steps["to"] == "D-H1") & (steps["apt"] != "all")).sum() == 2  # per airport
    cal = pd.read_csv(out.parent / "operational_ladder_calibration.csv")
    assert set(cal["deliverable"]) == {"D-H1", "D-H1-R4"} and "all" in set(cal["apt"])
    assert cal["coverage"].between(0, 1).all()

    # F1's tables do not exist yet: no failure, a printed note, F1's mean without -MED and -MAE, the rest unchanged
    capsys.readouterr()
    out2 = tmp_path / "report_no_f1_tables" / "operational_ladder"
    assert rep.main(["--runs-root", str(runs), "--out", str(out2), "--no-table-baselines",
                     "--tables", f"h1={tmp_path / 'not_fitted_yet.parquet'}", f"h1_r4_nodoy={tpath_old}"]) == 0
    printed = capsys.readouterr().out
    assert "deliverable D-H1: no calibration tables at" in printed and "no D-H1-MED or D-H1-MAE row" in printed
    rows2 = set(pd.read_csv(out2.with_suffix(".csv"))["row"])
    assert {"D-H1", mm, "D-H1-R4", "D-H1-R4-MED", "D-H1-R4-MAE"} <= rows2 and not {"D-H1-MED", "D-H1-MAE"} & rows2
    cal2 = pd.read_csv(out2.parent / "operational_ladder_calibration.csv")
    assert set(cal2["deliverable"]) == {"D-H1-R4"}
    steps2 = pd.read_csv(out2.parent / "operational_ladder_steps.csv")
    assert ((steps2["from"] == "D-H1-R4") & (steps2["to"] == "D-H1") & (steps2["apt"] == "all")).sum() == 1
