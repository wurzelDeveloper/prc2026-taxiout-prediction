"""The operational feature catalogue (prc/features/_catalogue.py, lab/feature_catalogue.py; findings 10.35, 10.37):
every column an operational model reads has exactly one complete entry, every source it names exists, every column has
a tier at every moment that reads it, and the generator writes both files. Since 13 Oct the H1 deliverable is F1 (the
H1 model before it plus the other flights' plans after the anchor), and Part D counts the concept members beyond it,
two of which are delivered beside it as tier F files (rule T's member as h1f, the full concept F5 as h1f5; findings
10.37). No competition data."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from prc.config import load_config
from prc.features import _catalogue as cat
from prc.features import _long_tiers as tiers
from prc.features import resolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lab"))

H1 = "op_h1_f1_plan.yaml"
"""The H1 deliverable since 13 Oct (findings 10.37): `op_h1_r4_nodoy` plus the other flights' plans after the anchor."""
F1_FAMILY = ("op_h1_f1_plan_febdec.yaml", "op_h1_f1_plan_marnov.yaml", "op_h1_f1_plan_aproct.yaml",
             "op_h1_f1_plan_maysep.yaml", "op_h1_f1_plan_junaug.yaml", "op_h1_f1_plan_fwd.yaml",
             "op_h1_f1_plan_mae_exact_cpu.yaml")
"""F1's month pairs, forward check and rule M median member (findings 10.37): F1's columns, other folds or loss."""
MEMBERS = {  # every configuration whose columns the catalogue must cover: the deliverables, H0 and the sensitivities
    "h1": (H1, *F1_FAMILY,
           # the H1 deliverable until 13 Oct and its sensitivities, kept on record for comparison
           "op_h1_r4_nodoy.yaml", "op_h1_r4_nodoy_lr04.yaml", "op_h1_r4_nodoy_noplan.yaml",
           "op_h1_r4_nodoy_notaf.yaml", "op_h1_r4_nodoy_rwyblind.yaml", "op_h1_r4_nodoy_fwd.yaml"),
    "hl3": ("op_hl3_r4_rwyblind.yaml", "op_hl3_r3_rwyblind.yaml"),
    "hl24": ("op_hl24_r4_rwyblind.yaml", "op_hl24_r3_rwyblind.yaml"),
    "h0": ("op_h0_r4_nodoy.yaml",),
}


def _kept(name: str) -> list[str]:
    cfg = load_config(name)
    drop = set(cfg.features.drop_columns)
    return [c for s in resolve(cfg.features.include, cfg.features.exclude) for c in s.outputs if c not in drop]


def test_no_key_in_two_families_and_entries_complete():
    seen: dict[str, str] = {}
    ids = {d.id for d in cat.DATA_POINTS}
    for fam in cat.FAMILIES:
        assert fam.title and fam.rationale and fam.literature and fam.columns, fam.id
        assert set(fam.sources) <= ids, (fam.id, set(fam.sources) - ids)
        assert set(fam.anchor_note) <= set(cat.HORIZONS), fam.id
        for key, definition in fam.columns.items():
            assert definition and definition.strip().endswith("."), (fam.id, key)
            assert key not in seen, f"{key} in {seen.get(key)} and {fam.id}"
            seen[key] = fam.id
    assert len({f.id for f in cat.FAMILIES}) == len(cat.FAMILIES)
    for d in cat.DATA_POINTS:
        assert set(d.known) == set(cat.HORIZONS), d.id
        assert d.what and d.system and d.use, d.id


@pytest.mark.parametrize("horizon,name", [(h, n) for h, names in MEMBERS.items() for n in names])
def test_every_column_has_one_entry_and_a_tier(horizon, name):
    cols = _kept(name)
    assert cols, name
    missing = [c for c in cols if cat.family_of(c) is None]
    assert not missing, (name, missing)
    for c in cols:
        t = tiers.tier_at(c, horizon)
        assert t == "O" or t.startswith("O*") or t.startswith("C"), (c, horizon, t)


def test_tiers_follow_the_moment():
    assert tiers.tier_at("runway", "h1").startswith("O*")
    assert tiers.tier_at("stand", "h1") == "O" and tiers.tier_at("stand", "hl3").startswith("O*")
    assert tiers.tier_at("op_h1_plan_minus_sched_min", "h1").startswith("O*")
    assert tiers.tier_at("op_h0_plan_minus_sched_min", "h0") == "O"
    assert tiers.tier_at("op_h1_backlog_apt", "h1").startswith("O*")
    assert tiers.tier_at("op_h1_wx_vis_m", "h1") == "O" and tiers.tier_at("op_h1_wx_headwind_kt", "h1").startswith("O*")
    assert tiers.tier_at("aircraft_type_flt", "h1").startswith("C")
    assert tiers.tier_at("op_hl_dem_dep_sched_apt_w30", "hl24") == "O"


def test_f1_is_the_h1_deliverable():
    """F1 (findings 10.37) is the H1 deliverable before it, rule L's choice, plus exactly the other flights' plans after
    the anchor: 15 columns of the fut_plan family, each O or O* at H1 (no tier F), with the O* reasons the catalogue
    states. Its month pairs, forward check and median member read the same columns."""
    import yaml

    import feature_catalogue
    from prc import paths

    assert feature_catalogue.DEFAULTS["h1"] == H1
    assert yaml.safe_load((paths.CONFIGS / H1).read_text(encoding="utf-8"))["extends"] == \
        feature_catalogue.H1_BEFORE_F1 == "op_h1_r4_nodoy.yaml"
    before, f1 = _kept(feature_catalogue.H1_BEFORE_F1), _kept(H1)
    added = [c for c in f1 if c not in set(before)]
    assert set(before) <= set(f1) and len(added) == 15
    assert all(c.startswith("op_h1_fplan_") and cat.family_of(c).id == "fut_plan" for c in added), added
    t = {c: tiers.tier_at(c, "h1") for c in added}
    assert all(x == "O" or x.startswith("O*") for x in t.values()), t
    assert t["op_h1_fplan_arr_sched_apt_w60"] == "O"
    assert "others' plans final" in t["op_h1_fplan_dep_plan_apt_w60"]
    assert "own runway" in t["op_h1_fplan_dep_plan_rwy_w15"]
    assert "others' allocation" in t["op_h1_fplan_arr_sched_pier_w30"]
    for name in F1_FAMILY:
        assert _kept(name) == f1, name
    fam = next(f for f in cat.FAMILIES if f.id == "fut_plan")
    assert "In the H1 deliverable since 13 Oct (findings 10.37)" in fam.caveats
    assert "Not in the strict deliverable" not in fam.caveats and "arr_events" not in fam.sources
    assert "the H1 deliverable reads them" in fam.rationale
    # the families of the deliverables come before the concept members' (Part B's order): the plans follow the backlog
    ids = [f.id for f in cat.FAMILIES]
    assert ids.index("fut_plan") == ids.index("plan_backlog") + 1 and ids.index("fut_plan") < ids.index("fut_records")
    use = {d.id: d.use for d in cat.DATA_POINTS}
    assert "the departures planned in the hour after the H1 anchor" in use["iobt"]
    assert "the arrivals scheduled in the hour after the H1 anchor" in use["sobt"]
    assert "the plans after the anchor (H1)" in use["stand"]
    assert "the departures planned after the anchor (H1)" in use["runway"]
    # H0 reads the plan too (its planned delay, its delay against plan and schedule, the backlog): the use texts say so
    h0 = set(_kept("op_h0_r4_nodoy.yaml"))
    assert {"op_h0_plan_minus_sched_min", "op_h0_delay_vs_plan_min", "op_h0_delay_vs_sched_min", "op_h0_backlog_apt",
            "op_h0_backlog_rwy", "op_h0_backlog_pier"} <= h0
    assert "the planned delay (H1, H0)" in use["iobt"] and "the planned delay (H1, H0)" in use["sobt"]
    assert "the off-block's delay against the plan" in use["iobt"]
    assert "the off-block's delay against the schedule" in use["sobt"]
    assert "the plan backlog (H1, H0)" in use["iobt"] and "the H1 plan backlog" not in use["iobt"]
    assert "the planned delay at H1" not in use["sobt"]
    # the 3 h deliverable reads the push delay against the schedule (the long history), the 24 h one does not
    assert "the push delay of departed flights (3 h)" in use["sobt"]
    assert any(c.endswith("_push_delay_mean") for c in _kept("op_hl3_r4_rwyblind.yaml"))
    assert not any(c.endswith("_push_delay_mean") for c in _kept("op_hl24_r3_rwyblind.yaml"))


def test_generator_writes_both_files(tmp_path):
    import pandas as pd

    import feature_catalogue

    out = tmp_path / "ops"
    assert feature_catalogue.main(["--out", str(out)]) == 0
    md = out.with_suffix(".md").read_text(encoding="utf-8")
    csv = out.with_suffix(".csv").read_text(encoding="utf-8")
    assert "Part A: the data points" in md and "Part B: the model inputs, by family" in md
    assert "op_h1_wx_lvp" in csv and "op_hl24_" in csv and "op_h0_" in csv
    assert "day_of_year" not in csv and "iobt_minus" not in csv
    front = md.split("## Part A")[0]  # the contract's exception and tier F are stated before any table
    assert "Part D's other concept members break the second rule on purpose" in front
    assert "`tests/test_operational_future.py`" in front and "never instead of it" in front
    # two concept members are delivered as tier F files (findings 10.37): rule T's member (h1f) and F5 (h1f5)
    assert ("Two of them are delivered as separate tier F files beside the H1 deliverable, never instead of it: rule "
            "T's member, `op_h1_f4_nodepthru` (`h1f`), an upper bound for perfect plans, and, at the owner's request, "
            "the full concept F5, `op_h1_f5_full` (`h1f5`, findings 10.37), whose realised departure side carries "
            "traces of the own slot, so its gain is an upper bound even for perfect plans.") in front
    assert "- **F** (Part D only): *planned in the concept, realised in the challenge data*" in front
    assert "findings 10.35, 10.36 and 10.37" in front
    assert "**Never read by the deliverables:**" in md and "Never read by any operational model" not in md
    assert "(+0.05 s of MAE, findings 10.35)" in md and "gained at most 0.6 s" not in md
    assert ("Of the two delivered as tier F files beside the deliverable, rule T's member (`op_h1_f4_nodepthru`, "
            "`h1f`) reads only the arrivals' taxi-ins; the full concept F5 (`op_h1_f5_full`, `h1f5`) reads all of "
            "them, up to the anchor as well.") in md  # the ADS-B entry of the never-read list
    # H1 is F1, and the front matter says what that adds (findings 10.37)
    assert f"| H1, 15 min before the last filed off-block | `{H1}` | {len(_kept(H1))} | deliverable (headline): F1" \
        in front
    assert (f"H1 has been F1 since 13 Oct (findings 10.37): the H1 deliverable until then, `op_h1_r4_nodoy.yaml` "
            f"({len(_kept('op_h1_r4_nodoy.yaml'))} columns, rule L's choice), plus the other flights' plans after the "
            "anchor (15 columns: 0 tier F, 12 O*, 3 O;") in front
    assert "- Other flights' plans count as known at the anchor, also where they lie after it" in front
    assert "Rule T's first step, F1, adds only the other flights' plans after the anchor" in front
    assert "the other flights' up to an hour ahead (the plans after the anchor)" in front
    for stale in ("all but F1 read", "None of these members is the strict H1 deliverable",
                  "Not in the strict deliverable", "the last filed off-block time is assumed known 15 min ahead.",
                  "the decisions findings 10.35 and 10.36.",
                  # F5 is delivered too (findings 10.37): nothing says otherwise, nor names one delivered member
                  "not delivered", "the delivered concept member", "The delivered concept member",
                  "One of them, `op_h1_f4_nodepthru`, is delivered", "the delivered one (`op_h1_f4_nodepthru`)"):
        assert stale not in md, stale
    frame = pd.read_csv(out.with_suffix(".csv"))
    h1 = frame[frame["member"] == "h1"]
    assert set(h1["config"]) == {H1} and len(h1) == len(_kept(H1))
    plans = h1[h1["family"] == "fut_plan"]
    assert len(plans) == 15 and not plans["tier"].str.startswith("F").any()


def _fake_run(runs: Path, run_id: str, gain: dict[str, float] | None = None, perm: str | None = None) -> str:
    """A finished run directory with the evidence files the generator reads (no competition data)."""
    d = runs / run_id
    d.mkdir(parents=True)
    (d / "metrics.json").write_text("{}", encoding="utf-8")
    if gain is not None:
        (d / "importance_gain.csv").write_text(
            "feature,gain\n" + "".join(f"{k},{v}\n" for k, v in gain.items()), encoding="utf-8")
    if perm is not None:
        (d / "feature_effects.csv").write_text(
            f'method,metric,columns,value,std\npermutation,rmse_delta,"{perm}",19.5,0.1\n', encoding="utf-8")
    return run_id


def test_a_gain_run_must_be_of_the_member_configuration(tmp_path, monkeypatch):
    """The gain shares are the member's own: a --gain run of another configuration (the H1 model before F1, which the
    documented commands named until 13 Oct, or one of F1's month pairs) stops the generator; F1's own run, given by id
    or by configuration name, passes. --perm may name another configuration's run (H1's comes from op_h1_r4)."""
    import feature_catalogue
    from prc import paths

    runs = tmp_path / "runs"
    monkeypatch.setattr(paths, "RUNS", runs)
    before = _fake_run(runs, "2026-10-11T13-09-32__op_h1_r4_nodoy__4e884d3b__s44", {"apt": 4.4})
    pair = _fake_run(runs, "2026-10-13T10-25-06__op_h1_f1_plan_febdec__90fc7858__s42", {"apt": 4.4})
    f1 = _fake_run(runs, "2026-10-13T07-54-55__op_h1_f1_plan__40ac5f36__s44",
                   {"apt": 4.39, "op_h1_fplan_dep_plan_rwy_w15": 2.05})
    perm = _fake_run(runs, "2026-10-08T02-37-29__op_h1_r4__3fcdc779__s42", perm="op_h1_backlog_apt,op_h1_backlog_rwy")
    out = tmp_path / "ops"
    common = ["--out", str(out), "--concept", ""]
    for wrong in (before, pair):
        with pytest.raises(SystemExit, match=f"is not a run of the h1 member {H1}"):
            feature_catalogue.main([*common, "--gain", f"h1={wrong}"])
    assert not out.with_suffix(".md").exists()  # refused before anything is written
    for given in (f1, "op_h1_f1_plan"):  # a run id, or the configuration's name (its newest finished run)
        assert feature_catalogue.main([*common, "--gain", f"h1={given}", "--perm", f"h1={perm}"]) == 0
        md = out.with_suffix(".md").read_text(encoding="utf-8")
        assert f"  - gain, {cat.HORIZON_NAMES['h1']}: `{f1}`" in md
        assert f"  - block permutation, {cat.HORIZON_NAMES['h1']}: `{perm}`" in md
        rows = {line.split(" | ")[0]: line for line in md.splitlines() if line.startswith("| `")}
        assert "| 2.05 |" in rows["| `fplan_dep_plan_rwy_w15`"]  # F1's own share on F1's plan column
        assert "19.5 ± 0.1" in rows["| `backlog_apt`"]  # the block permutation of op_h1_r4
    # an explicit H1 member takes its own runs: the H1 model before F1 with its run
    assert feature_catalogue.main([*common, "--h1", "op_h1_r4_nodoy.yaml", "--gain", f"h1={before}"]) == 0
    with pytest.raises(SystemExit, match="is not a run of the h1 member op_h1_r4_nodoy.yaml"):
        feature_catalogue.main([*common, "--h1", "op_h1_r4_nodoy.yaml", "--gain", f"h1={f1}"])


def test_the_defaults_are_the_delivered_configurations(tmp_path, monkeypatch):
    """Without decision records the catalogue documents the delivered members (H1: F1, findings 10.37; 24 h: r3, rule
    R's choice); a recorded rule R decision takes precedence, and an explicit option over both. F1 extends rule L's
    choice, so a rule L record naming another configuration stops the generator unless --h1 names the H1 member."""
    import argparse
    import json

    import feature_catalogue
    from prc import paths

    monkeypatch.setattr(paths, "RUNS", tmp_path)
    args = argparse.Namespace(**{h: None for h in cat.HORIZONS})
    assert feature_catalogue.member_configs(args) == {
        "hl24": "op_hl24_r3_rwyblind.yaml", "hl3": "op_hl3_r4_rwyblind.yaml", "h1": "op_h1_f1_plan.yaml",
        "h0": "op_h0_r4_nodoy.yaml"}
    (tmp_path / "op_decision_L.cfg").write_text("op_h1_r4_nodoy.yaml\n", encoding="utf-8")  # rule L's record (10.35)
    assert feature_catalogue.member_configs(args)["h1"] == "op_h1_f1_plan.yaml"
    (tmp_path / "op_decision_L.cfg").write_text("op_h1_r4_nodoy_lr04.yaml\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="rule L chose op_h1_r4_nodoy_lr04.yaml"):
        feature_catalogue.member_configs(args)
    args.h1 = "op_h1_r4_nodoy.yaml"  # an explicit H1 member skips the check: here the H1 model before F1
    assert feature_catalogue.member_configs(args)["h1"] == "op_h1_r4_nodoy.yaml"
    (tmp_path / "op_decisions.json").write_text(json.dumps({"R_hl24": {"choice": "op_hl24_r4_rwyblind.yaml"}}),
                                                encoding="utf-8")
    assert feature_catalogue.member_configs(args)["hl24"] == "op_hl24_r4_rwyblind.yaml"
    args.hl24 = "op_hl24_r3_rwyblind.yaml"
    assert feature_catalogue.member_configs(args)["hl24"] == "op_hl24_r3_rwyblind.yaml"


CONCEPT = ("op_h1_f1_plan.yaml", "op_h1_f2_rec.yaml", "op_h1_f3_nm.yaml", "op_h1_f4_adsb.yaml", "op_h1_f4_shift7.yaml",
           "op_h1_f4_norwy.yaml", "op_h1_f4_nodepthru.yaml", "op_h1_f5_full.yaml")
"""Rule T's concept members (findings 10.36): the H1 model before F1 plus traffic after the anchor. The first, F1 (the
plans), is the H1 deliverable since 13 Oct (findings 10.37); every other member reads F1's columns and more."""
F_FAMILIES = {"fut_records", "fut_nm", "fut_adsb"}
PART_D_FAMILIES = F_FAMILIES | {"fut_plan", "adsb_now"}
"""What Part D lists: the traffic after the anchor (the plans and the tier F families) and the ADS-B picture before
it."""


@pytest.mark.parametrize("name", CONCEPT)
def test_concept_members_every_column_has_one_entry_and_tier_f_only_after_the_anchor(name):
    """Every column of a concept member has one entry; tier F belongs to exactly the realised traffic after the anchor
    (and only there), the plans after it are O or O*, and the rest is the deliverable's (O, O*, C)."""
    cols = _kept(name)
    missing = [c for c in cols if cat.family_of(c) is None]
    assert not missing, (name, missing)
    for c in cols:
        t, fam = tiers.tier_at(c, "h1"), cat.family_of(c).id
        if fam in F_FAMILIES:
            assert t.startswith("F"), (c, t)
        else:
            assert t == "O" or t.startswith("O*") or t.startswith("C"), (c, t)
    deliverable = set(_kept(H1))
    assert deliverable <= set(cols), name


def test_the_concept_data_points_say_what_stands_in():
    for d in ("aman", "dman", "asmgcs"):
        dp = next(x for x in cat.DATA_POINTS if x.id == d)
        assert dp.known["h1"].startswith("F:") and "stand in" in dp.known["h1"], d
        assert all(dp.known[h] == "not used" for h in ("hl24", "hl3", "h0")), d
    # who reads each: never "the concept member" alone, nor "the delivered concept member" as if there were one. Two
    # concept members are delivered as tier F files (findings 10.37), and each text names both and what each reads:
    # rule T's member (`h1f`) only the arrivals' part, F5 (`h1f5`) all of it, the realised departure side included
    use = {d.id: d.use for d in cat.DATA_POINTS}
    for d in ("aman", "dman", "asmgcs", "adsb_surface"):
        assert "(the concept member)" not in use[d] and "the delivered concept member" not in use[d], d
        assert "`h1f`" in use[d] and "`h1f5`" in use[d] and "not delivered" not in use[d], d
    assert "so both delivered tier F files: the member's `h1f` and F5's `h1f5`" in use["aman"]
    assert "so the delivered F5 file `h1f5`, whose gain from them carries traces of the own slot" in use["dman"]
    assert "not rule T's member, delivered as `h1f`, which reads no realised departure-side column" in use["dman"]
    assert "rule T's member, delivered as `h1f`, reads only the arrivals' taxi-ins" in use["asmgcs"]
    assert "rule T's member, delivered as `h1f`, reads only the arrivals' taxi-ins after the anchor" \
        in use["adsb_surface"]
    assert "so the delivered F5 file `h1f5`, before the anchor too" in use["adsb_surface"]
    # up to the anchor no deliverable reads other aircraft's positions; of the tier F files only F5's does
    known = next(x for x in cat.DATA_POINTS if x.id == "adsb_surface").known["h1"]
    assert "no deliverable reads them, since R6 gained nothing, and of the tier F files only F5's `h1f5`" in known
    assert "in no deliverable: R6 gained nothing" not in known


def test_the_tier_f_caveat_credits_nothing_from_the_departure_side():
    """Rule T removes the departure-side traces by construction (its member reads no realised departure-side column)
    and credits nothing from that side; the traces that stay in the member are named (findings 10.36). F5, delivered
    as the second tier F file (findings 10.37), reads that side: its gain is an upper bound even for perfect plans."""
    for fam in ("fut_records", "fut_nm", "fut_adsb"):
        caveat = next(f for f in cat.FAMILIES if f.id == fam).caveats
        assert "removes the departure-side traces by construction" in caveat, fam
        assert "nothing from that side is credited" in caveat and "bounds them" not in caveat, fam
        for trace in ("the next arrival at the own stand (unmeasured)", "ADS-B taxi-in columns",
                      "the landings on the own runway (unmeasured"):
            assert trace in caveat, (fam, trace)
        assert "its member, `op_h1_f4_nodepthru` (delivered as `h1f`), reads no realised departure-side" in caveat, fam
        assert "the delivered `op_h1_f4_nodepthru`" not in caveat, fam  # one of two delivered members, not the one
        assert ("The full concept F5, delivered at the owner's request as the second tier F file (`h1f5`, findings "
                "10.37), reads the realised departure side as well: its gain carries traces of the own slot, which "
                "rule T could not credit, so it is an upper bound even for perfect plans.") in caveat, fam
    # the ADS-B picture before the anchor: no deliverable reads it, F5 does and is delivered, rule T's member does not
    now = next(f for f in cat.FAMILIES if f.id == "adsb_now").caveats
    assert "so no deliverable reads it. The full concept F5 reads it beside the traffic after the anchor" in now
    assert "delivered at the owner's request as the second tier F file (`h1f5`, findings 10.37)" in now
    assert "rule T's member (`h1f`) does not read it" in now and "not delivered" not in now


def _concept_counts(name: str, h1: str = H1, part_d: bool = False) -> tuple[int, int, int, int]:
    """(columns, tier F, O*, O) of a concept configuration beyond the H1 member (default: the deliverable, F1), from
    the configurations themselves; with ``part_d`` only the columns of `PART_D_FAMILIES`."""
    base = set(_kept(h1))
    cols = [c for c in _kept(name) if c not in base and (not part_d or cat.family_of(c).id in PART_D_FAMILIES)]
    t = [tiers.tier_at(c, "h1") for c in cols]
    return len(t), sum(x.startswith("F") for x in t), sum(x.startswith("O*") for x in t), sum(x == "O" for x in t)


def test_generator_writes_part_d_for_the_concept_member(tmp_path):
    """Part D lists the full concept F5 beyond the H1 deliverable F1 and counts rule T's member op_h1_f4_nodepthru
    beside it. Both are delivered as tier F files (findings 10.37: rule T's member as h1f, F5 at the owner's request as
    h1f5), and Part D names both with the files the delivery manifest records; the counts of both come from their
    configurations, against F1: F1 itself adds nothing, and rule T's member adds only tier F columns."""
    import json

    import pandas as pd

    import feature_catalogue
    from prc import paths

    out = tmp_path / "ops"
    assert feature_catalogue.main(["--out", str(out)]) == 0
    md = out.with_suffix(".md").read_text(encoding="utf-8")
    assert "## Part D: traffic after the anchor" in md and "planned in the concept, realised in the challenge data" in md
    part_b = md.split("## Part D")[0]
    part_d = md.split("## Part D")[1]
    for title in ("Realised traffic after the anchor: the airport's records",
                  "Realised surface picture after the anchor: public ADS-B"):
        assert f"### {title}" in part_d, title
    # F1's plans are the deliverable's: Part B lists them, Part D does not
    assert "### Other flights' plans after the anchor" in part_b
    assert "### Other flights' plans after the anchor" not in part_d and "fplan_" not in part_d
    assert "fut_records" not in part_b and "frec_" not in part_b  # the deliverables' parts hold no tier F column
    assert feature_catalogue.CONCEPT_DEFAULT == "op_h1_f5_full.yaml"
    assert feature_catalogue.DELIVERED_CONCEPT == "op_h1_f4_nodepthru.yaml"
    # the two tier F files Part D names are the delivered ones (the manifest's tracked copy): h1f and h1f5
    manifest = json.loads((paths.ROOT / "docs" / "operational_manifest.json").read_text(encoding="utf-8"))
    for key, name in (("h1f", feature_catalogue.DELIVERED_FILE), ("h1f5", feature_catalogue.F5_FILE)):
        assert manifest[key]["file"].replace("\\", "/").rsplit("/", 1)[-1] == name, key
    assert "| F1 | `op_h1_f1_plan.yaml` | the plans (O*): the H1 deliverable since 13 Oct (findings 10.37) |" in part_d
    assert "F1 reads only these and has been the H1 deliverable since 13 Oct (findings 10.37), so Part B lists them;" \
        in part_d
    assert (f"| F4-nodepthru | `op_h1_f4_nodepthru.yaml` | the plans and the realised arrivals only (no realised "
            f"departure side): rule T's member, delivered as a tier F file (`{feature_catalogue.DELIVERED_FILE}`) |") \
        in part_d
    assert (f"| F5 | `op_h1_f5_full.yaml` | F4 + the ADS-B picture before the anchor (actual and planned): the full "
            f"concept, delivered at the owner's request as a second tier F file (`{feature_catalogue.F5_FILE}`, "
            "findings 10.37) |") in part_d
    assert ("Two are delivered beside it as separate tier F files, never instead of it: rule T's member, "
            "`op_h1_f4_nodepthru` (`h1f`), and, at the owner's request, the full concept F5, `op_h1_f5_full` (`h1f5`, "
            "findings 10.37). F5 also reads the realised departure side, whose traces of the own slot rule T could not "
            "credit (findings 10.36): its gain is an upper bound even for perfect plans.") in part_d
    assert ("The tables below list the columns of the full concept F5 (`op_h1_f5_full.yaml`, delivered as `h1f5`) "
            "beyond the H1 deliverable (`op_h1_f1_plan.yaml`).") in part_d
    assert "### Surface picture up to the anchor: public ADS-B" in part_d  # F5's picture before the anchor
    assert "The concept member `op_h1_f5_full.yaml`" not in md
    assert "not delivered" not in part_d and "delivered concept member" not in md
    assert set(feature_catalogue.CONCEPT_FAMILIES) == PART_D_FAMILIES
    assert _concept_counts(H1) == (0, 0, 0, 0)  # F1 adds nothing to itself
    # beyond F1 (and the H1 model before it) every concept column is one of Part D's: nothing is counted apart
    for h1 in (H1, "op_h1_r4_nodoy.yaml"):
        for name in ("op_h1_f5_full.yaml", "op_h1_f4_nodepthru.yaml"):
            assert _concept_counts(name, h1) == _concept_counts(name, h1, part_d=True), (name, h1)
    assert "in this part's families" not in md and "in other families" not in md
    n, f, ostar, o = _concept_counts("op_h1_f5_full.yaml")
    assert (f"**Counts.** The full concept F5 (`op_h1_f5_full.yaml`, delivered as `h1f5`) reads {n} columns beyond the "
            f"H1 deliverable: {f} tier F, {ostar} O*, {o} O.") in part_d
    nd, fd, ostar_d, od = _concept_counts("op_h1_f4_nodepthru.yaml")
    assert (f" Rule T's member `op_h1_f4_nodepthru.yaml` (delivered as `h1f`) reads {nd}: {fd} tier F, {ostar_d} O*, "
            f"{od} O.") in part_d
    assert nd < n and fd < f
    assert nd == fd > 0 and ostar_d == od == 0  # beyond F1 rule T's member reads tier F only
    # F5's columns beyond F1 that are not tier F are exactly what F5 adds to F4: the ADS-B picture before the anchor
    assert n - f == len(_kept("op_h1_f5_full.yaml")) - len(_kept("op_h1_f4_adsb.yaml"))
    frame = pd.read_csv(out.with_suffix(".csv"))
    concept_rows = frame[frame["member"].str.startswith("concept")]
    assert not concept_rows["family"].eq("fut_plan").any()
    delivered = frame[frame["member"] == "concept_delivered"]
    assert set(delivered["config"]) == {"op_h1_f4_nodepthru.yaml"} and len(delivered) == nd
    assert set(delivered["column"]) <= set(frame.loc[frame["member"] == "concept", "column"])  # a subset of F5's
    assert not delivered["family"].isin(["fut_nm", "adsb_now"]).any()  # no NM after the anchor, no ADS-B before it
    out2 = tmp_path / "ops2"
    assert feature_catalogue.main(["--out", str(out2), "--concept", ""]) == 0
    assert "## Part D" not in out2.with_suffix(".md").read_text(encoding="utf-8")
    assert not pd.read_csv(out2.with_suffix(".csv"))["member"].str.startswith("concept").any()
    out3 = tmp_path / "ops3"  # Part D on rule T's member itself: one count, no second member
    assert feature_catalogue.main(["--out", str(out3), "--concept", "op_h1_f4_nodepthru.yaml"]) == 0
    md3 = out3.with_suffix(".md").read_text(encoding="utf-8")
    assert (f"**Counts.** Rule T's member `op_h1_f4_nodepthru.yaml` (delivered as `h1f`) reads {nd} columns beyond the "
            f"H1 deliverable: {fd} tier F, {ostar_d} O*, {od} O.\n") in md3
    assert "concept_delivered" not in set(pd.read_csv(out3.with_suffix(".csv"))["member"])
    out4 = tmp_path / "ops4"  # Part D on F1: nothing beyond the deliverable, no table, rule T's member counted
    assert feature_catalogue.main(["--out", str(out4), "--concept", H1]) == 0
    part_d4 = out4.with_suffix(".md").read_text(encoding="utf-8").split("## Part D")[1]
    assert (f"**Counts.** F1 (`{H1}`, the H1 deliverable) reads 0 columns beyond the H1 deliverable: 0 tier F, 0 O*, "
            f"0 O. Rule T's member `op_h1_f4_nodepthru.yaml` (delivered as `h1f`) reads {nd}: {fd} tier F, 0 O*, "
            "0 O.") in part_d4
    assert "### " not in part_d4 and "The tables below" not in part_d4


def test_an_explicit_h1_member_moves_the_plans_back_to_part_d(tmp_path):
    """--h1 names the H1 member (here the deliverable before F1): Part D then counts and lists the concept beyond that
    member, the plans after the anchor included, and the text says 'the H1 member', not 'the H1 deliverable'. A
    runway-blind H1 member leaves Part D to the traffic after the anchor: the own runway's columns the concept reads
    beyond it are counted apart, never listed as concept families."""
    import pandas as pd

    import feature_catalogue

    out = tmp_path / "ops"
    assert feature_catalogue.main(["--out", str(out), "--h1", "op_h1_r4_nodoy.yaml"]) == 0
    md = out.with_suffix(".md").read_text(encoding="utf-8")
    part_b, part_d = md.split("## Part D")
    assert "| `op_h1_r4_nodoy.yaml` |" in part_b and "given with --h1; the H1 deliverable is F1" in part_b
    assert "H1 has been F1 since 13 Oct" not in part_b
    plans_title = "### Other flights' plans after the anchor"
    assert plans_title in part_d and plans_title not in part_b
    assert "so Part B lists them" not in part_d
    n, f, ostar, o = _concept_counts("op_h1_f5_full.yaml", "op_h1_r4_nodoy.yaml")
    assert n == _concept_counts("op_h1_f5_full.yaml")[0] + 15
    assert (f"**Counts.** The full concept F5 (`op_h1_f5_full.yaml`, delivered as `h1f5`) reads {n} columns beyond the "
            f"H1 member: {f} tier F, {ostar} O*, {o} O.") in part_d
    assert "beyond the H1 member (`op_h1_r4_nodoy.yaml`)" in part_d
    frame = pd.read_csv(out.with_suffix(".csv"))
    assert frame.loc[frame["member"] == "concept", "family"].eq("fut_plan").sum() == 15
    blind = "op_h1_r4_nodoy_rwyblind.yaml"
    out2 = tmp_path / "ops_blind"
    assert feature_catalogue.main(["--out", str(out2), "--h1", blind]) == 0
    part_d2 = out2.with_suffix(".md").read_text(encoding="utf-8").split("## Part D")[1]
    for title in ("Departure runway", "Distance to the runway (map)", "Plan backlog", "Weather now (METAR)"):
        assert f"### {title}\n" not in part_d2, title
    assert f"{plans_title}\n" in part_d2
    # the same concept columns as beyond the H1 model before F1; the own runway's are counted apart
    n2, f2, ostar2, o2 = _concept_counts("op_h1_f5_full.yaml", blind, part_d=True)
    assert (n2, f2, ostar2, o2) == (n, f, ostar, o)
    k = _concept_counts("op_h1_f5_full.yaml", blind)[0] - n2
    nd2, fd2, ostar_d2, od2 = _concept_counts("op_h1_f4_nodepthru.yaml", blind, part_d=True)
    kd = _concept_counts("op_h1_f4_nodepthru.yaml", blind)[0] - nd2
    assert k > 0 and kd > 0
    assert (f"beyond the H1 member (`{blind}`) in this part's families. Columns are listed once") in part_d2
    assert (f"**Counts.** The full concept F5 (`op_h1_f5_full.yaml`, delivered as `h1f5`) reads {n2} columns beyond "
            f"the H1 member in this part's families (and {k} in other families, which this part does not list): {f2} "
            f"tier F, {ostar2} O*, {o2} O.") in part_d2
    assert (f"Rule T's member `op_h1_f4_nodepthru.yaml` (delivered as `h1f`) reads {nd2} in this part's families (and "
            f"{kd} in other families, which this part does not list): {fd2} tier F, {ostar_d2} O*, {od2} O.") in part_d2
    frame2 = pd.read_csv(out2.with_suffix(".csv"))
    concept2 = frame2[frame2["member"].str.startswith("concept")]
    assert set(concept2["family"]) <= PART_D_FAMILIES
    assert (concept2["member"] == "concept").sum() == n2 and (concept2["member"] == "concept_delivered").sum() == nd2
