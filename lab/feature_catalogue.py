"""Writes docs/operational_features.md and .csv: every input of the operational models, with its definition, source,
availability, tier, rationale and measured evidence (findings 10.35).

    python lab/feature_catalogue.py [--h1 CFG] [--hl3 CFG] [--hl24 CFG] [--h0 CFG]
                                    [--gain h1=RUN hl3=RUN ...] [--perm h1=RUN ...] [--out docs/operational_features]

**Inputs.**
- *Hand-written facts:* the data points, the families of columns, each column's definition, the rationale with its
  literature anchors and the caveats. They live in `prc/features/_catalogue.py`.
- *Tiers:* `prc.features._long_tiers.tier_at`, per column and moment.
- *Column lists:* each member's configuration, its blocks' outputs minus `features.drop_columns`. The defaults are the
  deliverables (`DEFAULTS`). H1 is F1, `op_h1_f1_plan.yaml` (findings 10.37): rule L's choice (runs/op_decision_L.cfg:
  `op_h1_r4_nodoy.yaml` at learning rate 0.08, the H1 deliverable until 13 Oct, `H1_BEFORE_F1`) plus the other
  flights' plans after the anchor. A rule L record naming another configuration stops the generator, since F1 would
  then not be rule L's choice plus the plans; ``--h1`` names the H1 member explicitly and skips that check (e.g.
  ``--h1 op_h1_r4_nodoy.yaml`` for the H1 model before F1). The long members are rule R's choices when
  runs/op_decisions.json holds them.
- *Part D (findings 10.36, 10.37):* the columns of the full concept F5 (``--concept``) beyond the H1 member, and the
  counts of rule T's member `op_h1_f4_nodepthru`. Both are delivered as separate tier F files beside the H1
  deliverable, never instead of it: rule T's member as `h1f` (`DELIVERED_FILE`), F5 at the owner's request as `h1f5`
  (`F5_FILE`; its realised departure side carries traces of the own slot, so its gain is an upper bound even for
  perfect plans). Part D keeps to the families of the traffic after the anchor and the ADS-B picture before it
  (`CONCEPT_FAMILIES`). With F1 as H1 the plans after the anchor are the deliverable's (Parts B and C), so Part D holds
  the tier F columns and F5's ADS-B picture before the anchor. An H1 member the concept extends by other families (a
  runway-blind ``--h1``) gets those columns counted apart, never listed.
- *Evidence.* Both kinds are optional and per member, given as run ids or config names; a name means the newest
  finished run of that config.
  - ``--gain`` reads each run's `importance_gain.csv`: CatBoost's loss-change share per column, in % of the model.
    It must be a run of the moment's member configuration (its id carries the configuration's name), or the generator
    stops: a run of another configuration, e.g. of the H1 model before F1, would put its shares on F1's columns.
  - ``--perm`` reads the block permutation of a run that measured it (`feature_effects.csv`, method `permutation`).
    It gives the holdout RMSE rise when the whole block is shuffled, so every column of a block shows its block's value.
    Its run may be of another configuration (the deliverables measure none; H1's comes from `op_h1_r4`), which the
    catalogue names.

**Outputs.** The markdown groups the columns by family. The csv has one row per (member, column); Part D's rows are
the members ``concept`` (``--concept``, by default the full concept F5, the `h1f5` file) and ``concept_delivered``
(rule T's member, the `h1f` file; named before F5 was delivered as well, and kept for the csv's readers).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from op_rules import find_run  # noqa: E402

from prc import paths  # noqa: E402
from prc.config import load_config  # noqa: E402
from prc.features import _catalogue as cat  # noqa: E402
from prc.features import _long_tiers as tiers  # noqa: E402
from prc.features import resolve  # noqa: E402

DEFAULTS = {"hl24": "op_hl24_r3_rwyblind.yaml", "hl3": "op_hl3_r4_rwyblind.yaml",
            "h1": "op_h1_f1_plan.yaml", "h0": "op_h0_r4_nodoy.yaml"}
"""The delivered configurations: at H1 F1, rule L's choice plus the other flights' plans after the anchor (findings
10.37); hours ahead rule R's choices (findings 10.35; r4 at 3 h, r3 at 24 h). Rule R's records under runs/ take
precedence when they exist, the options over both (`member_configs`)."""
H1_BEFORE_F1 = "op_h1_r4_nodoy.yaml"
"""The H1 deliverable until 13 Oct and rule L's choice (findings 10.35; learning rate 0.08), which F1 extends by the
other flights' plans after the anchor (findings 10.37). The H1 sensitivities Part B names (`_noplan`, `_notaf`,
`_rwyblind`) were measured on it."""
SHORT = {"hl24": "24 h", "hl3": "3 h", "h1": "H1", "h0": "H0"}
MEMBER_NOTE = {
    "hl24": "deliverable, refitted on 2025, predicts the final set",
    "hl3": "deliverable, refitted on 2025, predicts the final set",
    "h1": f"deliverable (headline): F1 since 13 Oct, `{H1_BEFORE_F1}` plus the other flights' plans after the anchor "
          "(findings 10.37); three seeds refitted on 2025, predicts the final set",
    "h0": "holdout only (2026 blanks the off-blocks): the member at the push, for comparison",
}


CONCEPT_DEFAULT = "op_h1_f5_full.yaml"
"""The full concept F5 documented in Part D (findings 10.36): every block after the anchor and the ADS-B picture
before. Delivered at the owner's request as the second tier F file beside the H1 deliverable, never instead of it
(`F5_FILE`, findings 10.37). Its realised departure side carries traces of the own slot, which rule T could not
credit, so its gain is an upper bound even for perfect plans. Its columns are a superset of rule T's member's."""
DELIVERED_CONCEPT = "op_h1_f4_nodepthru.yaml"
"""Rule T's member (findings 10.36), delivered beside the H1 deliverable, never instead of it, as a separate tier F
file (`DELIVERED_FILE`). It is one of the two delivered concept files; F5 (`CONCEPT_DEFAULT`, `F5_FILE`) is the
other. Part D gives its column counts beside the full concept's."""
DELIVERED_FILE = "h1f__2026-10-13T05-41-39__final_op_h1f_blend__08a90f35.parquet"
"""Rule T's member's tier F file (`lab/op_deliver.py --member h1f`, the mean only; findings 10.36)."""
F5_FILE = "h1f5__2026-10-13T17-23-14__final_op_h1f5_blend__f7393913.parquet"
"""The full concept F5's tier F file (`lab/op_deliver.py --member h1f5`, the mean only; findings 10.37)."""
CONCEPT_FAMILIES = ("fut_plan", "fut_records", "fut_nm", "fut_adsb", "adsb_now")
"""The families Part D lists: the traffic after the anchor (the plans, O*, and the tier F families) and F5's ADS-B
picture before the anchor. A concept column of another family beyond the H1 member (the own runway's, with a
runway-blind ``--h1``) is counted apart (`_left_out`), never listed."""
CONCEPT_RUNS = (
    ("F1", "op_h1_f1_plan.yaml", "the plans (O*): the H1 deliverable since 13 Oct (findings 10.37)"),
    ("F2", "op_h1_f2_rec.yaml", "+ the airport's records (F)"),
    ("F3", "op_h1_f3_nm.yaml", "+ NM's readings (F)"),
    ("F4", "op_h1_f4_adsb.yaml", "+ public ADS-B (F): the full concept after the anchor"),
    ("F4-nodepthru", DELIVERED_CONCEPT, "the plans and the realised arrivals only (no realised departure side): rule "
     f"T's member, delivered as a tier F file (`{DELIVERED_FILE}`)"),
    ("F5", CONCEPT_DEFAULT, "F4 + the ADS-B picture before the anchor (actual and planned): the full concept, "
     f"delivered at the owner's request as a second tier F file (`{F5_FILE}`, findings 10.37)"),
)
"""The rule T members (findings 10.36) and what each reads after the anchor. The first, F1, became the H1 deliverable
(findings 10.37); F4-nodepthru (rule T's member) and F5 are delivered beside it as tier F files."""


def member_configs(args) -> dict[str, str]:
    """The configuration of each moment: `DEFAULTS`, then rule R's records, then the options. H1 is F1 unless ``--h1``
    names another configuration. F1 extends rule L's choice, so without ``--h1`` a rule L record
    (runs/op_decision_L.cfg) must name `H1_BEFORE_F1` (or F1 itself); any other choice stops here."""
    cfgs = dict(DEFAULTS)
    rule_l = paths.RUNS / "op_decision_L.cfg"
    if not getattr(args, "h1", None) and rule_l.exists():
        chosen = Path(rule_l.read_text(encoding="utf-8").strip()).name
        if chosen not in (H1_BEFORE_F1, DEFAULTS["h1"]):
            raise SystemExit(f"rule L chose {chosen} ({rule_l}), but the H1 deliverable {DEFAULTS['h1']} extends "
                             f"{H1_BEFORE_F1} (findings 10.37): pass --h1 to name the H1 member")
    dec = paths.RUNS / "op_decisions.json"
    if dec.exists():
        doc = json.loads(dec.read_text(encoding="utf-8"))
        for lead in ("hl3", "hl24"):
            choice = (doc.get(f"R_{lead}") or {}).get("choice")
            if choice:
                cfgs[lead] = choice
    for h in cat.HORIZONS:
        if getattr(args, h):
            cfgs[h] = getattr(args, h)
    return cfgs


def kept_columns(cfg_name: str) -> list[tuple[str, str]]:
    """(block, column) of every column the configuration's model reads, in block order."""
    cfg = load_config(cfg_name)
    drop = set(cfg.features.drop_columns)
    return [(s.name, c) for s in resolve(cfg.features.include, cfg.features.exclude) for c in s.outputs if c not in drop]


def parse_pairs(items) -> dict[str, str]:
    out = {}
    for item in items or ():
        h, _, run = item.partition("=")
        if h not in cat.HORIZONS or not run:
            raise SystemExit(f"expected MOMENT=RUN with MOMENT in {cat.HORIZONS}, got {item!r}")
        out[h] = run
    return out


def gain_of(run_name: str) -> tuple[str, dict[str, float]]:
    run = find_run(run_name)
    if run is None or not (run / "importance_gain.csv").exists():
        raise SystemExit(f"no importance_gain.csv for {run_name}")
    g = pd.read_csv(run / "importance_gain.csv")
    return run.name, dict(zip(g["feature"], g["gain"]))


def perm_of(run_name: str) -> tuple[str, dict[str, tuple[float, float]]]:
    """Block permutation per column: (RMSE rise in s, its std over repeats) of the column's block."""
    run = find_run(run_name)
    if run is None or not (run / "feature_effects.csv").exists():
        raise SystemExit(f"no feature_effects.csv for {run_name}")
    fe = pd.read_csv(run / "feature_effects.csv")
    fe = fe[(fe["method"] == "permutation") & (fe["metric"] == "rmse_delta")]
    out = {}
    for _, r in fe.iterrows():
        for c in str(r["columns"]).split(","):
            out[c] = (float(r["value"]), float(r["std"]))
    return run.name, out


def concept_rows(cfg_name: str, member: str, base: set[str]) -> list[dict]:
    """The columns of a concept configuration beyond ``base`` (the H1 member's columns) in `CONCEPT_FAMILIES`, tiered
    at H1. With F1 as the H1 member these are the tier F columns after the anchor (and F5's ADS-B picture before it):
    F1's plans are the deliverable's. A column without a catalogue entry keeps family None, which `main` refuses."""
    rows = []
    for block, col in kept_columns(cfg_name):
        fam = cat.family_of(col)
        if col in base or (fam is not None and fam.id not in CONCEPT_FAMILIES):
            continue
        rows.append({"member": member, "config": cfg_name, "block": block, "column": col,
                     "key": cat.strip_prefix(col), "family": fam.id if fam else None,
                     "definition": cat.definition_of(col), "tier": tiers.tier_at(col, "h1"), "gain_pct": None,
                     "block_perm_rmse_s": None, "block_perm_std_s": None})
    return rows


def build(cfgs: dict[str, str], gains: dict[str, dict], perms: dict[str, dict], concept: str | None = None,
          delivered: str | None = None) -> pd.DataFrame:
    """One row per (member, column). With ``concept``, its columns beyond the H1 member's in `CONCEPT_FAMILIES` as
    member ``concept`` (Part D's tables) and, with ``delivered`` (another configuration; `main` passes rule T's member),
    those of ``delivered`` as member ``concept_delivered`` (Part D's counts)."""
    rows = []
    if concept:
        base = {col for _, col in kept_columns(cfgs["h1"])}
        rows += concept_rows(concept, "concept", base)
        if delivered and delivered != concept:
            rows += concept_rows(delivered, "concept_delivered", base)
    for h in cat.HORIZONS:
        for block, col in kept_columns(cfgs[h]):
            fam = cat.family_of(col)
            g = gains.get(h, {}).get(col)
            p = perms.get(h, {}).get(col)
            rows.append({
                "member": h, "config": cfgs[h], "block": block, "column": col, "key": cat.strip_prefix(col),
                "family": fam.id if fam else None, "definition": cat.definition_of(col),
                "tier": tiers.tier_at(col, h),
                "gain_pct": None if g is None else round(g, 3),
                "block_perm_rmse_s": None if p is None else round(p[0], 2),
                "block_perm_std_s": None if p is None else round(p[1], 2),
            })
    return pd.DataFrame(rows)


def _cell(v, fmt="{:.2f}") -> str:
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else fmt.format(v)


def markdown(frame: pd.DataFrame, cfgs: dict[str, str], gain_runs: dict[str, str], perm_runs: dict[str, str],
             concept: str | None = None, delivered: str | None = None) -> str:
    """The catalogue. ``concept`` and ``delivered`` name the configurations of Part D's rows (see `build`); without
    ``concept`` there is no Part D."""
    concept_frame = frame[frame["member"].isin(("concept", "concept_delivered"))]
    frame = frame[frame["member"].isin(cat.HORIZONS)]
    h1 = frame[frame["member"] == "h1"]
    L = []
    w = L.append
    w("# The operational models' inputs: what, from where, when known, and why")
    w("")
    w("Generated by `python lab/feature_catalogue.py` from `prc/features/_catalogue.py` (the hand-written facts), "
      "`prc/features/_long_tiers.py` (the tiers) and the configurations; do not edit by hand. The model card is "
      "`docs/operational_model.md`, the ladder `docs/operational_ladder.md`, the decisions findings 10.35, 10.36 "
      "and 10.37.")
    w("")
    w("## The models and their moments")
    w("")
    w("Every model is one global CatBoost over the ten airports, fitted on 2025 (targets above 2 h and wrong-date rows "
      "left out of the fit). Each predicts at a fixed moment and reads only what exists at that moment:")
    w("")
    w("| moment | configuration | columns | role |")
    w("|---|---|---|---|")
    for h in cat.HORIZONS:
        n = int((frame["member"] == h).sum())
        note = MEMBER_NOTE[h]
        if h == "h1" and cfgs[h] != DEFAULTS[h]:
            note = f"given with --h1; the H1 deliverable is F1 (`{DEFAULTS[h]}`, findings 10.37)"
        w(f"| {cat.HORIZON_NAMES[h]} | `{cfgs[h]}` | {n} | {note} |")
    w("")
    if cfgs["h1"] == DEFAULTS["h1"]:
        before = {col for _, col in kept_columns(H1_BEFORE_F1)}
        added = h1.loc[~h1["column"].isin(before), "tier"]
        w(f"H1 has been F1 since 13 Oct (findings 10.37): the H1 deliverable until then, `{H1_BEFORE_F1}` "
          f"({len(before)} columns, rule L's choice), plus the other flights' plans after the anchor "
          f"({len(added)} columns: {_counts(added)}; Part B, *Other flights' plans after the anchor*). The H1 "
          f"sensitivities Part B names (`op_h1_r4_nodoy_noplan`, `_notaf`, `_rwyblind`) were measured on "
          f"`{H1_BEFORE_F1}`.")
        w("")
    w("**The information contract**, enforced by the leakage gates in `tests/test_operational.py`, "
      "`tests/test_operational_long.py` and, for the plans after the anchor, `tests/test_operational_future.py`:")
    w("- The flight's own takeoff and its own off-block record are never inputs. Neither are its own NM actuals "
      "(AOBT_3, ARVT_3) or EOBT_1.")
    w("- Other flights' events count only when they happened at or before the anchor.")
    w("- Other flights' plans count as known at the anchor, also where they lie after it: the schedule (O) and the "
      "last filed off-blocks (O*: a filed plan can still change after the anchor). The long models read the scheduled "
      "demand around the planned off-block, H1 the plans of the next hour (findings 10.37).")
    w("- A METAR counts from 5 min after its observation. A TAF counts only when issued at or before the anchor.")
    w("- Statistics that read targets or NM readings come from the fit rows only, month-out. Target-free tables (the "
      "arrivals' taxi-in medians) are built over the whole frame: transductive in time, with no departure target.")
    w("")
    w("Rule T's first step, F1, adds only the other flights' plans after the anchor (the third rule) and has been the "
      "H1 deliverable since 13 Oct (findings 10.37). Part D's other concept members break the second rule on purpose: "
      "beyond F1 they read the other flights' realised movements after the anchor (tier F). Two of them are delivered "
      "as separate tier F files beside the H1 deliverable, never instead of it: rule T's member, "
      f"`{Path(DELIVERED_CONCEPT).stem}` (`h1f`), an upper bound for perfect plans, and, at the owner's request, the "
      f"full concept F5, `{Path(CONCEPT_DEFAULT).stem}` (`h1f5`, findings 10.37), whose realised departure side "
      "carries traces of the own slot, so its gain is an upper bound even for perfect plans. Their gates are in "
      "`tests/test_operational_future.py`.")
    w("")
    w("**Tiers.**")
    w("- **O**, operational: an airport or NM holds the value at the anchor.")
    w("- **O***: available under an assumption the data cannot check; the reason is named in the cell. The main cases:")
    w("  - the planned departure runway is assumed equal to the one recorded;")
    w("  - hours ahead, the stand plan is assumed final;")
    w("  - other flights' recorded runway or stand stands in for the one planned;")
    w("  - at H1, the last filed off-block times are assumed known at the anchor: the flight's own 15 min ahead, the "
      "other flights' up to an hour ahead (the plans after the anchor).")
    w("- **C**: an artefact of how the challenge data were built (the NM-join level of the flight-plan columns). It is "
      "constant on the NM-joined population, where the models are read.")
    w("- **F** (Part D only): *planned in the concept, realised in the challenge data*. The other flights' realised "
      "movements after the anchor stand in for the plans of the arrival, departure and surface managers (AMAN, DMAN, "
      "A-SMGCS). An F input is not available as computed; a gain from it is an upper bound for perfect plans. An F "
      "model is read beside the deliverable, never instead of it.")
    w("")
    w("**The applicability domain.** NM-joined departures, i.e. movements NM matched to a flight plan: 98.9 % of the "
      "2025 departures and 98.5 % of the January and July 2026 ranking departures. The unjoined rest holds the stale "
      "and wrong-date block times (in 2025, 42 % of the target's squared deviation). The models still predict those "
      "rows, but nothing is claimed there.")
    w("")
    if gain_runs or perm_runs:
        w("**Evidence columns.**")
        if gain_runs:
            w("- *gain %*: CatBoost's loss-change share of the column in the run named below, in % of the model.")
        if perm_runs:
            w("- *block perm.*: the holdout RMSE rise in seconds when the column's whole block is shuffled, ± its std "
              "over repeats, measured on the run named below. Every column of a block shows its block's value.")
        for h, r in gain_runs.items():
            w(f"  - gain, {cat.HORIZON_NAMES[h]}: `{r}`")
        for h, r in perm_runs.items():
            w(f"  - block permutation, {cat.HORIZON_NAMES[h]}: `{r}`")
        w("")
    # data points
    w("## Part A: the data points")
    w("")
    w("One row per raw field or external source, with when an operator holds it at each moment.")
    w("")
    w("| data point | field | what it is | system holding it | 24 h | 3 h | H1 | H0 | used for |")
    w("|---|---|---|---|---|---|---|---|---|")
    for d in cat.DATA_POINTS:
        k = d.known
        w(f"| {d.id} | `{d.field}` | {d.what} | {d.system} | {k['hl24']} | {k['hl3']} | {k['h1']} | {k['h0']} | {d.use} |")
    w("")
    w("**Never read by the deliverables:**")
    w("")
    for what, why in cat.NEVER_USED:
        w(f"- {what}: {why}.")
    w("")
    # families
    w("## Part B: the model inputs, by family")
    w("")
    w("Per family: the definition of each column, the moments that read it with the column's tier there (blank: that "
      "model does not read it), the evidence, the data points it is built from, and why it should move the taxi-out "
      "time, with the literature that supports it.")
    w("")
    dp = {d.id: d.field for d in cat.DATA_POINTS}
    for fam in cat.FAMILIES:
        sub = frame[frame["family"] == fam.id]
        if sub.empty:
            continue
        w(f"### {fam.title}")
        w("")
        w(f"**Why it matters.** {fam.rationale}")
        w("")
        w(f"**Literature.** {fam.literature}")
        w("")
        w(f"**Built from:** {', '.join(f'`{dp[s]}`' for s in fam.sources)}.")
        if fam.anchor_note:
            notes = "; ".join(f"{cat.HORIZON_NAMES[h].split(',')[0].split(' (')[0]}: {t}" for h, t in fam.anchor_note.items())
            w(f"**Read at:** {notes}.")
        if fam.caveats:
            w("")
            w(f"**Caveats.** {fam.caveats}")
        w("")
        head = "| column | definition | 24 h | 3 h | H1 | H0 |"
        sep = "|---|---|---|---|---|---|"
        if gain_runs:
            head += " gain % (" + " / ".join(SHORT[h] for h in gain_runs) + ") |"
            sep += "---|"
        if perm_runs:
            head += " block perm. (s) |"
            sep += "---|"
        w(head)
        w(sep)
        for key, definition in fam.columns.items():
            rows = sub[sub["key"] == key]
            if rows.empty:
                continue
            cells = []
            for h in cat.HORIZONS:
                r = rows[rows["member"] == h]
                cells.append(r["tier"].iloc[0] if len(r) else "")
            line = f"| `{key}` | {definition} | " + " | ".join(cells) + " |"
            if gain_runs:
                vals = []
                for h in gain_runs:
                    r = rows[rows["member"] == h]
                    vals.append(_cell(r["gain_pct"].iloc[0]) if len(r) else "")
                line += " " + " / ".join(v or "-" for v in vals) + " |"
            if perm_runs:
                vals = []
                for h in perm_runs:
                    r = rows[rows["member"] == h]
                    if len(r) and pd.notna(r["block_perm_rmse_s"].iloc[0]):
                        vals.append(f"{r['block_perm_rmse_s'].iloc[0]:.1f} ± {r['block_perm_std_s'].iloc[0]:.1f}")
                line += " " + (" / ".join(vals) or "") + " |"
            w(line)
        w("")
    # summary
    w("## Part C: counts by tier")
    w("")
    w("| moment | columns | O | O* | C | C + O* |")
    w("|---|---|---|---|---|---|")
    for h in cat.HORIZONS:
        t = frame.loc[frame["member"] == h, "tier"]
        o = int((t == "O").sum())
        ostar = int(t.str.startswith("O*").sum())
        c_only = int((t.str.startswith("C") & ~t.str.contains(r"O\*", regex=True)).sum())
        c_star = int((t.str.startswith("C") & t.str.contains(r"O\*", regex=True)).sum())
        w(f"| {cat.HORIZON_NAMES[h]} | {len(t)} | {o} | {ostar} | {c_only} | {c_star} |")
    w("")
    if concept:
        L += concept_part(concept_frame, concept, delivered if delivered != concept else None, cfgs["h1"],
                          bool((h1["family"] == "fut_plan").any()))
    return "\n".join(L) + "\n"


def _concept_name(cfg_name: str) -> str:
    """How Part D names a concept configuration: rule T's member and the full concept F5 (the two delivered as tier F
    files, `h1f` and `h1f5`), F1, or another one."""
    if cfg_name == DELIVERED_CONCEPT:
        return f"rule T's member `{cfg_name}` (delivered as `h1f`)"
    if cfg_name == CONCEPT_DEFAULT:
        return f"the full concept F5 (`{cfg_name}`, delivered as `h1f5`)"
    if cfg_name == DEFAULTS["h1"]:
        return f"F1 (`{cfg_name}`, the H1 deliverable)"
    return f"`{cfg_name}` (not delivered)"


def _capital(text: str) -> str:
    return text[:1].upper() + text[1:]


def _counts(t: pd.Series) -> str:
    return (f"{int(t.str.startswith('F').sum())} tier F, {int(t.str.startswith('O*').sum())} O*, "
            f"{int((t == 'O').sum())} O")


def _left_out(cfg_name: str, h1_cfg: str) -> int:
    """How many columns ``cfg_name`` reads beyond the H1 member ``h1_cfg`` outside `CONCEPT_FAMILIES`, which Part D
    counts apart and does not list: none beyond F1 or the H1 model before it, the own runway's beyond a runway-blind
    H1 member."""
    base = {col for _, col in kept_columns(h1_cfg)}
    fams = (cat.family_of(col) for _, col in kept_columns(cfg_name) if col not in base)
    return sum(1 for f in fams if f is not None and f.id not in CONCEPT_FAMILIES)


def _scope(left_out: int) -> str:
    """What a Part D count covers when the configuration also reads ``left_out`` columns of other families."""
    if not left_out:
        return ""
    return f" in this part's families (and {left_out} in other families, which this part does not list)"


def concept_part(frame: pd.DataFrame, name: str, delivered: str | None, h1_cfg: str,
                 h1_reads_plans: bool) -> list[str]:
    """Part D: the traffic after the anchor, the concept members of rule T (findings 10.36). The tables list the
    columns of ``name`` beyond the H1 member ``h1_cfg`` in `CONCEPT_FAMILIES` (member ``concept`` of ``frame``); the
    counts add rule T's member's (member ``concept_delivered``) when ``delivered`` names another
    configuration, and say how many columns of other families a configuration reads beyond the H1 member (none beyond
    F1). With F1 as the H1 member (``h1_reads_plans``) the plans after the anchor are in Part B, and Part D holds what
    lies beyond them."""
    delivered_rows = frame[frame["member"] == "concept_delivered"]
    concept = frame[frame["member"] == "concept"]
    beyond = "the H1 deliverable" if h1_cfg == DEFAULTS["h1"] else "the H1 member"
    left_out = _left_out(name, h1_cfg)
    L = []
    w = L.append
    dp = {d.id: d for d in cat.DATA_POINTS}
    w("## Part D: traffic after the anchor (the concept members, findings 10.36)")
    w("")
    w("The owner's concept (12 Oct): the operational model should also know the traffic situation around the planned "
      "off-block, after H1 too. In operations the airport's planning systems hold it: the arrival manager (AMAN) the "
      "landing sequence, the departure manager (DMAN) the start-up and takeoff sequence and the runway plan, the "
      "surface manager (A-SMGCS) the pushbacks, routes, taxi times and holding. The challenge data hold none of these "
      "plans, only what happened. The concept members therefore read:")
    w("- the other flights' **plans** after the anchor where the data hold them (filed off-blocks, the schedule): "
      "tier O* (O for the arrivals scheduled at the airport). F1 reads only these and has been the H1 deliverable "
      "since 13 Oct (findings 10.37)" + (", so Part B lists them;" if h1_reads_plans else ";"))
    w("- the other flights' **realised movements** after the anchor, as the stand-in for the planning systems' "
      "forecasts: tier **F**, *planned in the concept, realised in the challenge data*. An F input is not available as "
      "computed; a gain from it is an upper bound for what perfect plans would give.")
    w("")
    w("| member | configuration | reads after the anchor |")
    w("|---|---|---|")
    for label, cfg, what in CONCEPT_RUNS:
        w(f"| {label} | `{cfg}` | {what} |")
    w("")
    text = ("The placebo `op_h1_f4_shift7` reads every F block a week earlier, and `op_h1_f4_norwy` drops the own "
            "runway's sequence; rule T and its result are in findings 10.36 and the model card. The members after F1 "
            "read tier F columns, and none of them is the H1 deliverable. Two are delivered beside it as separate tier "
            f"F files, never instead of it: rule T's member, `{Path(DELIVERED_CONCEPT).stem}` (`h1f`), and, at the "
            f"owner's request, the full concept F5, `{Path(CONCEPT_DEFAULT).stem}` (`h1f5`, findings 10.37). F5 also "
            "reads the realised departure side, whose traces of the own slot rule T could not credit (findings 10.36): "
            "its gain is an upper bound even for perfect plans.")
    if len(concept):
        where = " in this part's families" if left_out else ""
        text += (f" The tables below list the columns of {_concept_name(name)} beyond {beyond} (`{h1_cfg}`){where}. "
                 "Columns are listed once for their three read times (`f15`: the planned off-block, `f30`, `f45`: 15 "
                 "and 30 min after it), and in their definitions 'the last N min' and 'the anchor' mean the read time; "
                 "the tier is read at H1.")
    w(text)
    w("")
    for fam in cat.FAMILIES:
        sub = concept[concept["family"] == fam.id]
        if sub.empty:
            continue
        w(f"### {fam.title}")
        w("")
        ops = [dp[x] for x in fam.sources if dp[x].known["h1"].startswith("F")]
        if ops:
            w("**In operations, and the stand-in here.** " + " ".join(
                f"*{d.field}*: {d.known['h1'][3:].strip()}." for d in ops))
            w("")
        w(f"**Why it matters.** {fam.rationale}")
        w("")
        w(f"**Literature.** {fam.literature}")
        w("")
        if fam.caveats:
            w(f"**Caveats.** {fam.caveats}")
            w("")
        w("| column (read times) | definition | tier at H1 |")
        w("|---|---|---|")
        seen = set()
        for key in fam.columns:
            rows = sub[sub["key"] == key]
            if rows.empty:
                continue
            parts = key.split("_", 2)
            if len(parts) == 3 and parts[1] in cat.READS:
                base = f"{parts[0]}_{{{'|'.join(cat.READS)}}}_{parts[2]}"
                if base in seen:
                    continue
                seen.add(base)
                definition = fam.columns[key].split(". ", 1)[1] if ". " in fam.columns[key] else fam.columns[key]
                shown = base
            else:
                definition, shown = fam.columns[key], key
            w(f"| `{shown}` | {definition} | {rows['tier'].iloc[0]} |")
        w("")
    t = concept["tier"]
    counts = (f"**Counts.** {_capital(_concept_name(name))} reads {len(t)} columns beyond {beyond}{_scope(left_out)}: "
              f"{_counts(t)}.")
    if delivered:
        d = delivered_rows["tier"]
        counts += (f" {_capital(_concept_name(delivered))} reads {len(d)}{_scope(_left_out(delivered, h1_cfg))}: "
                   f"{_counts(d)}.")
    w(counts)
    w("")
    return L


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for h in cat.HORIZONS:
        ap.add_argument(f"--{h}", help=f"configuration of the {h} member (default: the deliverable)")
    ap.add_argument("--gain", nargs="*",
                    help="MOMENT=RUN: gain shares from that run, a run of the moment's member configuration")
    ap.add_argument("--perm", nargs="*", help="MOMENT=RUN: block permutation from that run (any configuration)")
    ap.add_argument("--out", default=str(paths.ROOT / "docs" / "operational_features"))
    ap.add_argument("--concept", default=CONCEPT_DEFAULT,
                    help="the configuration whose columns beyond the H1 member Part D lists (default: the full concept "
                         "F5, delivered as the tier F file h1f5; findings 10.36, 10.37); '' leaves Part D out")
    a = ap.parse_args(argv)
    cfgs = member_configs(a)
    gain_runs, gains = {}, {}
    for h, r in parse_pairs(a.gain).items():
        gain_runs[h], gains[h] = gain_of(r)
        tag = f"__{load_config(cfgs[h]).name}__"  # the run id's name field (a month pair's carries a suffix)
        if tag not in gain_runs[h]:
            raise SystemExit(f"--gain {h}={r}: {gain_runs[h]} is not a run of the {h} member {cfgs[h]} (no {tag} in "
                             "its id); pass one of that configuration's runs, e.g. a refit")
    perm_runs, perms = {}, {}
    for h, r in parse_pairs(a.perm).items():
        perm_runs[h], perms[h] = perm_of(r)
    concept = a.concept or None
    frame = build(cfgs, gains, perms, concept, DELIVERED_CONCEPT)
    missing = frame[frame["family"].isna()]
    if len(missing):
        raise SystemExit(f"columns without a catalogue entry: {sorted(set(missing['column']))}")
    out = Path(a.out)
    frame.to_csv(out.with_suffix(".csv"), index=False)
    out.with_suffix(".md").write_text(markdown(frame, cfgs, gain_runs, perm_runs, concept, DELIVERED_CONCEPT),
                                      encoding="utf-8", newline="\n")
    print(f"wrote {out.with_suffix('.md')} and .csv: {len(frame)} rows; members {cfgs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
