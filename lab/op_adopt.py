"""F1's adoption as the H1 deliverable (findings 10.37): the superseded records kept, and the pre-registered checks.

    python lab/op_adopt.py archive   # keep H1's records (op_h1_r4_nodoy) before F1's are written; run it first
    python lab/op_adopt.py checks    # F1's month pairs and forward split against H1's, and rule F with F1's references
    python lab/op_adopt.py rule-m    # rule M: F1's median member against F1's calibrated median (C_h1)

The owner adopted F1 (`op_h1_f1_plan`: `op_h1_r4_nodoy`, the H1 deliverable until 13 Oct, plus the other flights'
plans after the anchor, tier O*) on 13 Oct. Its checks were fixed in findings 10.37 before their runs, with the bars of
10.35 and 10.36 unchanged. Like lab/op_rules.py they read holdout predictions only, on the OP population
(`op_rules.op_metrics`), and record their decisions in runs/op_decisions.json (`op_rules.record`). Each run they read
is the newest finished seed-42 run of its config at the config's current hash, never a predict run: what
`op_rules.find_run(current=True)` reads, at the pre-registered seed, which the hash does not carry (F1's three refits
share one). F1 stays adopted whatever the checks show: a failed check goes into the model card as a limit, and to the
owner. In the records and the read-out, H1 is `op_h1_r4_nodoy`, as in rule T's.

- ``archive`` copies the decisions ``C_h1`` and ``M`` to ``C_h1_r4_nodoy`` and ``M_r4_nodoy``, and moves the records
  ``h1`` and ``h1_calibrated_median`` of the delivery manifest (submissions/operational/manifest.json) to
  ``h1_r4_nodoy`` and ``h1_r4_nodoy_calibrated_median``, so that F1's calibration check (`lab/op_calibrate.py check
  --member h1`), its rule M and its delivery (`lab/op_deliver.py --member h1`) overwrite nothing on record; the files
  themselves stay in submissions/operational/. It keeps only op_h1_r4_nodoy's records and never overwrites a key it
  wrote before: a second call, or one after F1's delivery (``h1`` then F1's), changes nothing and says so. Each
  manifest key that still holds H1's record is moved, also when the other was moved before (by an earlier call or by
  hand). Where a record to keep is gone or another model's, or a live key and its archive key both hold H1's records,
  it refuses before writing anything. It never touches submissions/submitting.parquet.
- ``checks``: rule T's criteria 4 and 5 (10.36) for F1, OP MAE. c4: F1's five month pairs (`op_h1_f1_plan_<pair>`)
  better than `op_h1_r4_nodoy_<pair>` in at least 8 of the 10 validation months, all ten measured. c5:
  `op_h1_f1_plan_fwd` (fit January-October, validate November and December) better than `op_h1_r4_nodoy_fwd`. Rule F
  (10.35) with F1's own references: its forward November and December against the same months of its pairs (November
  from `_marnov`, December from `_febdec`), tripped above +8 s; the excess over H1's fixed references
  (`op_rules.F_REF`) is reported beside it. A missing or unfinished run fails its criterion and is named, never a
  crash. Recorded as ``A_F1``; the read-out is runs/op_adopt_f1.md.
- ``rule-m``: `op_h1_f1_plan_mae_exact_cpu` (H1's rule M settings on F1's inputs: MAE loss, exact leaves, CPU,
  refitted) replaces F1's calibrated median only if its holdout beats it by at least 1 s of OP MAE and 1 point within
  +-2 min (10.35). An OP MAE gain within the seed floor (0.25 s) of its bar is reported as unresolved and keeps the
  rule's default, the calibrated median (10.35, Measurement). The reference is ``C_h1``, which must be F1's: the check
  of the three refitted seeds that ship (`op_rules.refits`, as `lab/op_deliver.py --config op_h1_f1_plan.yaml --n 3`
  takes them), with tables fitted on F1's five month pairs (their meta file, written by `lab/op_calibrate.py fit`).
  H1's ``M`` must be archived first. Recorded as ``M``; one line states the decision and what follows (if rule C did
  not pass for F1, the file holds the mean only and there is no median to replace).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lab"))

import op_rules  # noqa: E402
from op_rule_t import MIN_MONTHS, PAIRS  # noqa: E402
from op_rules import F_MARGIN, F_REF, SEED_FLOOR_MAE, op_metrics, record, refits  # noqa: E402

from prc import paths  # noqa: E402
from prc.config import load_config  # noqa: E402

H1 = "op_h1_r4_nodoy"
"""The superseded H1 deliverable (findings 10.35): the reference of every check, and the owner of the records kept."""
F1 = "op_h1_f1_plan"
"""The adopted one (findings 10.37)."""
MEDIAN = f"{F1}_mae_exact_cpu"
"""F1's rule M member."""
SEED = 42
"""The pre-registered seed of the month pairs, the forward split and the median member (findings 10.37)."""
N_SEEDS = 3
"""F1's refitted seeds that ship (`lab/op_deliver.py --n 3`): rule M's reference (C_h1) is their mean."""
N_MONTHS = 2 * len(PAIRS)
"""c4 needs every validation month of the five pairs measured."""
F_PAIR = {"2025-11": "marnov", "2025-12": "febdec"}
"""Rule F's months (those of `op_rules.F_REF`) and the pair whose F1 run gives each month's reference."""
M_MAE_GAIN_S = 1.0
"""Rule M: the median member's OP MAE at least 1 s below the calibrated median's ..."""
M_WITHIN_120_GAIN_PP = 1.0
"""... and its share within +-2 min at least 1 point above it. An MAE gain within `op_rules.SEED_FLOOR_MAE` (0.25 s) of
its bar is unresolved and keeps the default, the calibrated median (findings 10.35)."""
KEY = "A_F1"
OUT_MD = "op_adopt_f1.md"
"""The read-out of ``checks``, written under runs/."""
ARCHIVE_DECISIONS = (("C_h1", "C_h1_r4_nodoy"), ("M", "M_r4_nodoy"))
ARCHIVE_MANIFEST = (("h1", "h1_r4_nodoy"), ("h1_calibrated_median", "h1_r4_nodoy_calibrated_median"))
GONE = {
    "C_h1": ("Rebuild it as C_h1_r4_nodoy with `python lab/op_calibrate.py check --tables "
             f"runs/op_calib/h1_nodoy_pairs_lr08.parquet --runs <{H1}'s three refits> --member h1_r4_nodoy`."),
    "M": "Findings 10.35 (rule M) holds its numbers.",
    "h1": "docs/operational_manifest.json holds its tracked copy.",
}
"""Where a superseded record that is gone can be found again (the refusal says it)."""


# --- files and records -----------------------------------------------------------------------------------------------

def _decisions() -> Path:
    """runs/op_decisions.json. `op_rules.record` writes to the path op_rules bound when imported: it must still be the
    one under ``paths.RUNS``, from which this script reads, or nothing is read or written."""
    want = paths.RUNS / "op_decisions.json"
    if Path(op_rules.DECISIONS) != want:
        raise SystemExit(f"op_rules.DECISIONS is {op_rules.DECISIONS}, not {want}: nothing read or written")
    return want


def _manifest() -> Path:
    return paths.SUBMISSIONS / "operational" / "manifest.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write(path: Path, text: str) -> None:
    """Atomically: a temporary file beside it, then replaced."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _config_of(run_id) -> str:
    """The config name in a run id (``<stamp>__<config>__<hash>__s<seed>``), or the id itself."""
    parts = str(run_id).split("__")
    return parts[1] if len(parts) >= 3 else str(run_id)


def _runs_of(runs, name: str) -> bool:
    """Whether ``runs`` is a non-empty list of run ids of config ``name``."""
    return isinstance(runs, list) and bool(runs) and all(f"__{name}__" in str(r) for r in runs)


def _check_of(rec, name: str) -> bool:
    """A rule C check (`lab/op_calibrate.py check`) of config ``name``'s seeds."""
    return isinstance(rec, dict) and _runs_of(rec.get("runs"), name)


def _rule_m_of(rec, name: str) -> bool:
    """A rule M record whose candidate is a run of ``name``'s median member."""
    return isinstance(rec, dict) and name in str((rec.get("candidate") or {}).get("run", ""))


def _file_of(rec, name: str) -> bool:
    """A delivery manifest record (`lab/op_deliver.py`) made from config ``name``'s refits."""
    return isinstance(rec, dict) and _runs_of(rec.get("source_runs"), name)


def _whose(rec) -> str:
    """What a record holds, for a refusal: absent, or the configs of its runs."""
    if rec is None:
        return "absent"
    if not isinstance(rec, dict):
        return "not a record"
    runs = rec.get("runs") or rec.get("source_runs") or [(rec.get("candidate") or {}).get("run")]
    names = sorted({_config_of(r) for r in runs if r})
    return "the record of " + ", ".join(names) if names else "a record without runs"


def _run(name: str) -> Path | None:
    """The newest finished (metrics.json) seed-42 run of config ``name`` at the config's current hash, never a predict
    run; None if there is none. A newer run still training or refitting (holdout predictions written, metrics.json not
    yet) is passed over for the finished one, as `op_rules.find_run` does."""
    tag = f"__{name}__{load_config(f'{name}.yaml').hash()}__s{SEED}"
    hits = sorted(p for p in paths.RUNS.glob(f"*{tag}")
                  if (p / "metrics.json").exists() and not (p / "predict.json").exists())
    return hits[-1] if hits else None


def _tables_pairs(tables) -> list[str] | None:
    """The configs of the month-pair runs rule C's tables were fitted on, sorted, from the meta file that
    `lab/op_calibrate.py fit` writes beside them; None without one. A relative path is read from the repository root,
    where the lab scripts run."""
    if not tables:
        return None
    path = Path(str(tables))
    meta = (path if path.is_absolute() else ROOT / path).with_suffix(".json")
    if not meta.exists():
        return None
    return sorted(_config_of(r) for r in json.loads(meta.read_text(encoding="utf-8")).get("pairs") or [])


# --- archive ---------------------------------------------------------------------------------------------------------

def archive() -> dict:
    """Keep H1's records under their archive keys; returns the keys copied, moved and found kept already."""
    dec_path, man_path = _decisions(), _manifest()
    if not man_path.exists():
        raise SystemExit(f"archive: no delivery manifest {man_path}: nothing written")
    doc, man = _read(dec_path), _read(man_path)
    copies, kept = {}, []
    for (src, dst), of in zip(ARCHIVE_DECISIONS, (_check_of, _rule_m_of)):
        if dst in doc:
            if not of(doc[dst], H1):
                raise SystemExit(f"archive: {dst} in {dec_path} is {_whose(doc[dst])}, not {H1}'s: nothing written")
            kept.append(dst)
        elif of(doc.get(src), H1):
            copies[dst] = doc[src]
        else:
            raise SystemExit(f"archive: {src} in {dec_path} is {_whose(doc.get(src))}, not {H1}'s, and {dst} is "
                             f"absent: the superseded record is gone; nothing written. {GONE[src]}")
    # the manifest: an archive key holds H1's record or is absent; a live key holds H1's record (moved now), F1's
    # (written after an earlier call) or nothing, and never H1's while its archive key holds one already
    (h1, h1_dst), (cal, _) = ARCHIVE_MANIFEST
    for src, dst in ARCHIVE_MANIFEST:
        if dst in man and not _file_of(man[dst], H1):
            raise SystemExit(f"archive: {dst} in {man_path} is {_whose(man[dst])}, not {H1}'s: nothing written")
        if src in man and not (_file_of(man[src], H1) or _file_of(man[src], F1)):
            raise SystemExit(f"archive: {src} in {man_path} is {_whose(man[src])}, neither {H1}'s nor {F1}'s: "
                             "nothing written")
        if dst in man and _file_of(man.get(src), H1):
            raise SystemExit(f"archive: {src} and {dst} in {man_path} both hold {H1}'s records, and a move would "
                             f"overwrite {dst}: keep one of them by hand; nothing written")
    if h1_dst not in man and not _file_of(man.get(h1), H1):
        raise SystemExit(f"archive: {h1} in {man_path} is {_whose(man.get(h1))}, not {H1}'s, and {h1_dst} is absent: "
                         f"the superseded record is gone; nothing written. {GONE[h1]}")
    if _file_of(man.get(h1), H1) and _file_of(man.get(cal), F1):
        raise SystemExit(f"archive: {cal} in {man_path} is {_whose(man[cal])} while {h1} is still {H1}'s: "
                         "nothing written")
    moves = [(src, dst) for src, dst in ARCHIVE_MANIFEST if _file_of(man.get(src), H1)]
    kept += [dst for _, dst in ARCHIVE_MANIFEST if dst in man]

    source = {d: s for s, d in ARCHIVE_DECISIONS}
    for dst, value in copies.items():
        record(dst, value)
        print(f"archive: {source[dst]} copied to {dst} ({dec_path})")
    if moves:
        for src, dst in moves:
            man[dst] = man.pop(src)
        _write(man_path, json.dumps(man, indent=2))
        for src, dst in moves:
            print(f"archive: {src} moved to {dst} ({man_path})")
    if not copies and not moves:
        print(f"archive: nothing to do, the superseded H1 records are kept already ({', '.join(kept)})"
              + (f"; h1 is {F1}'s" if _file_of(man.get(h1), F1) else ""))
    return {"copied": list(copies), "moved": [dst for _, dst in moves], "kept": kept}


# --- checks ----------------------------------------------------------------------------------------------------------

def _num(v) -> float | None:
    """A metric, or None where a run or a month is missing."""
    return None if v is None or not math.isfinite(v) else float(v)


def _diff(a, b) -> float | None:
    """``a - b`` rounded as op_rules rounds (3 decimals), or None when either is missing."""
    a, b = _num(a), _num(b)
    return None if a is None or b is None else round(a - b, 3)


def _f(v, signed: bool = False) -> str:
    return "n/a" if v is None else (f"{v:+.2f}" if signed else f"{v:.2f}")


def checks() -> dict:
    """Generalisation (c4, c5) and rule F for F1; recorded as A_F1 and read out in runs/op_adopt_f1.md."""
    _decisions()  # before any metric is read: the record must go where it is read from
    months, pairs, pair_missing, f1_pairs = {}, [], [], {}
    for p in PAIRS:
        names = {"h1": f"{H1}_{p}", "f1": f"{F1}_{p}"}
        runs = {k: _run(n) for k, n in names.items()}
        gone = [n for k, n in names.items() if runs[k] is None]
        pair_missing += gone
        if runs["f1"] is not None:
            f1_pairs[p] = op_metrics(runs["f1"])  # rule F's references come from F1's pairs even without H1's
        if gone:
            pairs.append({"pair": p, "missing": gone})
            continue
        h, f = op_metrics(runs["h1"])["months"], f1_pairs[p]["months"]
        for m in sorted(set(h) & set(f)):
            d = _diff(f[m], h[m])
            if d is not None:
                months[m] = {"pair": p, "h1": h[m], "f1": f[m], "delta": d}
        pairs.append({"pair": p, "h1": runs["h1"].name, "f1": runs["f1"].name})
    months = {m: months[m] for m in sorted(months)}
    deltas = {m: v["delta"] for m, v in months.items()}
    better = sum(1 for v in deltas.values() if v < 0)
    c4 = len(deltas) == N_MONTHS and better >= MIN_MONTHS

    fwd_names = {"h1": f"{H1}_fwd", "f1": f"{F1}_fwd"}
    fwd_runs = {k: _run(n) for k, n in fwd_names.items()}
    fwd_missing = [n for k, n in fwd_names.items() if fwd_runs[k] is None]
    fwd = {k: op_metrics(r) if r is not None else None for k, r in fwd_runs.items()}
    forward = {"h1": fwd["h1"]["mae"] if fwd["h1"] else None, "f1": fwd["f1"]["mae"] if fwd["f1"] else None,
               "h1_run": fwd_runs["h1"].name if fwd_runs["h1"] else None,
               "f1_run": fwd_runs["f1"].name if fwd_runs["f1"] else None}
    c5 = not fwd_missing and forward["f1"] < forward["h1"]

    ref = {m: _num((f1_pairs.get(p) or {}).get("months", {}).get(m)) for m, p in F_PAIR.items()}
    fwd_months = {m: _num((fwd["f1"] or {}).get("months", {}).get(m)) for m in F_PAIR}
    excess = {m: _diff(fwd_months[m], ref[m]) for m in F_PAIR}
    known = [v for v in excess.values() if v is not None]
    tripped = True if any(v > F_MARGIN for v in known) else (False if len(known) == len(excess) else None)
    rule_f = {
        "reference": ref, "reference_runs": {m: (f1_pairs.get(p) or {}).get("run") for m, p in F_PAIR.items()},
        "forward_run": forward["f1_run"], "forward_months": fwd_months, "excess": excess, "margin": F_MARGIN,
        "tripped": tripped, "excess_vs_h1_ref": {m: _diff(fwd_months[m], F_REF[m]) for m in F_PAIR},
        "h1_reference": dict(F_REF),
        "action": {True: "the model card states F1's forward MAE as the expected out-of-time level beside the Jan+Jul "
                         "figure and lists the excess as a limit",
                   False: "none: the forward months stay within F1's month pairs' level plus the margin",
                   None: "none until the missing values exist"}[tripped],
    }
    if tripped is None:
        gaps = [f"forward {m} ({F1}_fwd)" for m in F_PAIR if fwd_months[m] is None]
        gaps += [f"reference {m} ({F1}_{p})" for m, p in F_PAIR.items() if ref[m] is None]
        rule_f["reason"] = "unresolved: no " + ", no ".join(gaps)

    reasons = {
        "c4": (f"better in {better} of {len(deltas)} months measured, at least {MIN_MONTHS} of {N_MONTHS} needed"
               + (f"; missing: {', '.join(pair_missing)}" if pair_missing else "")),
        "c5": (f"F1 {forward['f1']:.2f} s against H1 {forward['h1']:.2f} s" if not fwd_missing
               else f"missing: {', '.join(fwd_missing)}"),
    }
    dec = {"configs": {"h1": f"{H1} (the H1 deliverable until 13 Oct)", "f1": F1}, "seed": SEED,
           "months_delta_mae": deltas, "months": months, "months_better": better, "c4": bool(c4),
           "forward": forward, "c5": bool(c5), "rule_f": rule_f, "pairs": pairs,
           "missing": pair_missing + fwd_missing, "reasons": reasons, "passed": bool(c4 and c5),
           "note": "F1 is adopted either way (owner, 13 Oct): a failed check goes into the model card as a limit and "
                   "to the owner (findings 10.37)"}
    record(KEY, dec)
    text = _md(dec)
    _write(paths.RUNS / OUT_MD, text)
    print(text)
    return dec


def _md(dec: dict) -> str:
    rf = dec["rule_f"]
    rows = [f"| {m} | {v['pair']} | {_f(v['h1'])} | {_f(v['f1'])} | {_f(v['delta'], signed=True)} |"
            for m, v in dec["months"].items()]
    own = ", ".join(f"{m} {_f(rf['excess'][m], signed=True)} s ({_f(rf['forward_months'][m])} against "
                    f"{_f(rf['reference'][m])} from _{p})" for m, p in F_PAIR.items())
    fixed = ", ".join(f"{m} {_f(v, signed=True)} s" for m, v in rf["excess_vs_h1_ref"].items())
    return "\n".join([
        "# F1's adoption checks (findings 10.37)", "",
        f"OP MAE (s) per validation month of the 2025 month pairs, seed {dec['seed']}: F1 (`{F1}_<pair>`) against H1 "
        f"(`{H1}_<pair>`, the H1 deliverable until 13 Oct).",
        "",
        "| month | pair | H1 | F1 | F1 - H1 |", "|---|---|---|---|---|", *rows, "",
        f"- Month pairs (c4): {dec['reasons']['c4']}: {dec['c4']}.",
        f"- Forward, fit January-October, validate November and December (c5): {dec['reasons']['c5']}: {dec['c5']}.",
        f"- Rule F with F1's own references (tripwire +{rf['margin']:g} s): {own}: tripped {rf['tripped']}"
        + (f" ({rf['reason']})" if rf.get("reason") else "") + ".",
        f"- Against H1's fixed references (`op_rules.F_REF`, for information): {fixed}.",
        f"- Missing runs: {', '.join(dec['missing']) or 'none'}.",
        f"- Passed (c4 and c5): {dec['passed']}. {dec['note']}.", ""])


# --- rule M ----------------------------------------------------------------------------------------------------------

def _reference(c) -> None:
    """Refuse unless ``C_h1`` is F1's calibrated median as 10.37 defines it: tables fitted on F1's five month pairs,
    checked on the mean of the three refitted seeds that ship."""
    if not _check_of(c, F1):
        raise SystemExit(f"rule-m: C_h1 is {_whose(c)}, not F1's calibration check: run `python lab/op_calibrate.py "
                         f"check --tables <F1's tables> --runs <F1's three refits> --member h1` first")
    try:
        ship = refits(F1, N_SEEDS)
    except SystemExit as e:
        raise SystemExit(f"rule-m: {e}: F1's calibrated median (C_h1) is the mean of the {N_SEEDS} refitted seeds "
                         "that ship") from None
    runs = [str(r) for r in c["runs"]]
    if sorted(runs) != sorted(ship):
        raise SystemExit(f"rule-m: C_h1 is the check of {', '.join(runs)}, not of the {N_SEEDS} seeds of {F1} that "
                         f"ship ({', '.join(ship)}): run `python lab/op_calibrate.py check --tables <F1's tables> "
                         f"--runs {' '.join(ship)} --member h1` first")
    want, got = sorted(f"{F1}_{p}" for p in PAIRS), _tables_pairs(c.get("tables"))
    if got is None:
        raise SystemExit(f"rule-m: C_h1's tables ({c.get('tables')}) have no meta file beside them "
                         "(`lab/op_calibrate.py fit` writes one), so the month pairs they were fitted on are unknown")
    if got != want:
        raise SystemExit(f"rule-m: C_h1's tables ({c.get('tables')}) were fitted on {', '.join(got) or 'no pairs'}, "
                         f"not on F1's five month pairs ({', '.join(want)}): fit F1's tables (`python "
                         "lab/op_calibrate.py fit --pairs ...`) and check them on F1's seeds first")


def rule_m() -> dict:
    """Rule M for F1's median member against F1's calibrated median; recorded as M."""
    doc = _read(_decisions())
    if "M_r4_nodoy" not in doc:
        raise SystemExit("rule-m: M_r4_nodoy is absent: run `python lab/op_adopt.py archive` first, so that H1's rule "
                         "M record is kept before F1's replaces it")
    c = doc.get("C_h1")
    _reference(c)
    run = _run(MEDIAN)
    if run is None:
        raise SystemExit(f"rule-m: no finished seed-{SEED} run of {MEDIAN} at its current hash under {paths.RUNS} "
                         "(metrics.json is written after the refit)")
    cand, ref = op_metrics(run), c["median"]
    # rounded to the inputs' precision (C_h1 four decimals, op_metrics three; shares four), so that the bar compares
    # the recorded values without float noise: 100 * (0.4816 - 0.4716) is 0.99999... in floats
    mae_gain = round(ref["mae"] - cand["mae"], 4)
    w_pp = round(100 * (cand["within_120"] - ref["within_120"]), 2)
    # 10.35: a result within the seed floor of its threshold is unresolved and keeps the rule's default (the median)
    unresolved = abs(round(mae_gain - M_MAE_GAIN_S, 4)) <= SEED_FLOOR_MAE
    passed = mae_gain >= M_MAE_GAIN_S and w_pp >= M_WITHIN_120_GAIN_PP and not unresolved
    if not c.get("passed"):  # no band in the file, and lab/op_deliver.py --median-config needs it
        action = "rule C did not pass for F1 (C_h1): the H1 file holds the mean only, no median to replace"
    elif passed:
        action = f"its refit replaces F1's calibrated median (y_med): lab/op_deliver.py --median-config {MEDIAN}.yaml"
    elif unresolved:
        action = (f"the OP MAE gain lies within the seed floor ({SEED_FLOOR_MAE:g} s) of the {M_MAE_GAIN_S:g} s bar: "
                  "unresolved, so the rule keeps its default (findings 10.35): the H1 file carries F1's calibrated "
                  "median, lab/op_deliver.py without --median-config")
    else:
        action = "the H1 file carries F1's calibrated median: lab/op_deliver.py without --median-config"
    dec = {"passed": bool(passed), "unresolved": bool(unresolved), "candidate": cand,
           "reference": {"what": "F1's calibrated median (C_h1, three seeds)", "mae": ref["mae"],
                         "within_120": ref["within_120"]},
           "reference_runs": c["runs"], "reference_tables": c.get("tables"), "rule_c_passed": bool(c.get("passed")),
           "mae_gain": mae_gain, "within_120_gain_pp": w_pp,
           "bar": {"mae_gain": M_MAE_GAIN_S, "within_120_gain_pp": M_WITHIN_120_GAIN_PP},
           "seed_floor_mae": SEED_FLOOR_MAE, "action": action}
    record("M", dec)
    word = "passed" if passed else ("not passed (unresolved)" if unresolved else "not passed")
    print(f"rule M for F1: {word}: {run.name} OP MAE {cand['mae']:.2f} s, {100 * cand['within_120']:.2f} % within "
          f"+-2 min, against F1's calibrated median {ref['mae']:.2f} s, {100 * ref['within_120']:.2f} % (gains "
          f"{mae_gain:+.3f} s and {w_pp:+.2f} points, bars {M_MAE_GAIN_S:g} s and {M_WITHIN_120_GAIN_PP:g} point, "
          f"seed floor {SEED_FLOOR_MAE:g} s): {action}")
    return dec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("archive", help="keep H1's records (op_h1_r4_nodoy) before F1's are written")
    sub.add_parser("checks", help="F1's month pairs and forward split against H1's, and rule F (key A_F1)")
    sub.add_parser("rule-m", help="rule M for F1's median member against F1's calibrated median (key M)")
    a = ap.parse_args(argv)
    {"archive": archive, "checks": checks, "rule-m": rule_m}[a.cmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
