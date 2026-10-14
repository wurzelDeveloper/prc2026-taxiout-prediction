"""lab/op_adopt.py: the checks of F1's adoption as the H1 deliverable (findings 10.37), on synthetic runs and records.

No competition data, no GPU, no network. The run directories hold nothing but placeholder files and are named at each
configuration's current hash, read from the real configs/, so the script's run lookups (its seed-42 one and
`op_rules.refits`) run as they run on runs/; `op_metrics` (which reads the movement frame) is a stand-in keyed by run
name. Everything the script reads and writes lives under a temporary root: runs/ and submissions/ (`paths.RUNS`,
`paths.SUBMISSIONS`, and `op_rules.DECISIONS`, which op_rules binds when imported) and the script's ROOT, from which
the relative table paths of rule C's records are read; the real runs/op_decisions.json and delivery manifest are never
touched.

* ``checks``: c4 needs all ten month-pair months measured and F1 better in at least 8 (8 pass, 7 fail); c5 a lower
  forward OP MAE than H1's (equal fails); rule F reads F1's own pair months (November from _marnov, December from
  _febdec), trips above +8 s and not at it, and reports the excess over H1's fixed references beside; only the newest
  finished seed-42 run of each config at its current hash is read (never a later run of another seed, a later seed-42
  run still training, or a predict run); a missing or unfinished run fails its criterion and is named, never a crash;
  the read-out runs/op_adopt_f1.md.
* ``rule-m``: passes beyond the seed floor of the 1 s bar at exactly 1 point (also where the float difference of the
  shares falls short of 1.0) and fails just below either; a gain within the floor (0.75 to 1.25 s) is unresolved and
  keeps the calibrated median; it reads the seed-42 candidate only, and the tables' meta by a relative or an absolute
  path; it refuses without the archived M_r4_nodoy, without F1's C_h1 (the check of the three refitted seeds that
  ship, with tables fitted on F1's five month pairs) or without a finished seed-42 candidate run at its current hash,
  writing nothing.
* ``archive``: copies C_h1 and M, moves h1 and h1_calibrated_median (the latter also when h1 was moved before), and
  leaves every other key and the scored file alone; a second call, and one after F1's delivery, changes nothing and
  says so; it refuses a foreign record, a record already gone, or H1's records under a live key and its archive key
  both, writing nothing.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from prc import paths
from prc.config import load_config


def _load():
    spec = importlib.util.spec_from_file_location("op_adopt", paths.ROOT / "lab" / "op_adopt.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


oa = _load()

STAMP = "2026-10-13T12-00-00"
LATER = "2026-10-13T23-00-00"
"""A later stamp: a run made after the one the script must read."""
PAIR_MONTHS = {"febdec": ("2025-02", "2025-12"), "marnov": ("2025-03", "2025-11"), "aproct": ("2025-04", "2025-10"),
               "maysep": ("2025-05", "2025-09"), "junaug": ("2025-06", "2025-08")}
MONTHS = sorted(m for ms in PAIR_MONTHS.values() for m in ms)
H1, F1 = "op_h1_r4_nodoy", "op_h1_f1_plan"
_HASHES: dict[str, str] = {}


def _hash(name: str) -> str:
    """The configuration's current hash (configs/<name>.yaml), as the script's run lookups require it."""
    if name not in _HASHES:
        _HASHES[name] = load_config(f"{name}.yaml").hash()
    return _HASHES[name]


H1_SEEDS = [f"2026-10-11T12-02-25__{H1}__4e884d3b__s42", f"2026-10-11T12-35-59__{H1}__4e884d3b__s43",
            f"2026-10-11T13-09-32__{H1}__4e884d3b__s44"]
F1_SEEDS = [f"2026-10-13T06-12-27__{F1}__{_hash(F1)}__s42", f"2026-10-13T07-14-31__{F1}__{_hash(F1)}__s43",
            f"2026-10-13T07-54-55__{F1}__{_hash(F1)}__s44"]
"""F1's three refits at its current hash (40ac5f36 on 13 Oct): the seeds `op_rules.refits` returns, those that ship."""
F1_STAGE1 = f"2026-10-12T22-26-06__{F1}__13468ab0__s42"
"""Rule T's stage-1 run of F1, at the hash before the refit flag (holdout only)."""
F4_SEEDS = ["2026-10-13T03-48-19__op_h1_f4_nodepthru__8946614f__s42",
            "2026-10-13T04-23-37__op_h1_f4_nodepthru__8946614f__s43"]
H1_M_RUN = f"2026-10-12T05-10-45__{H1}_mae_exact_cpu__94e24e8c__s42"
F1_M_RUN = f"2026-10-13T18-00-00__{F1}_mae_exact_cpu__d7b90aac__s42"
H1_TABLES, F1_TABLES = "runs/op_calib/h1_nodoy_pairs_lr08.parquet", "runs/op_calib/h1_f1_pairs.parquet"
"""Rule C's tables as `lab/op_calibrate.py check` records them, relative to the repository root (here the temporary
root); their meta files list H1's and F1's five month-pair runs (`rule_m_world`)."""


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A temporary root with runs/ and submissions/ (with a scored file), and the stand-in `op_metrics` reading
    ``table``."""
    runs, subs = tmp_path / "runs", tmp_path / "submissions"
    runs.mkdir()
    (subs / "operational").mkdir(parents=True)
    monkeypatch.setattr(paths, "RUNS", runs)
    monkeypatch.setattr(paths, "SUBMISSIONS", subs)
    monkeypatch.setattr(oa.op_rules, "DECISIONS", runs / "op_decisions.json")
    monkeypatch.setattr(oa, "ROOT", tmp_path)
    table: dict[str, dict] = {}
    monkeypatch.setattr(oa, "op_metrics", lambda run: json.loads(json.dumps(table[Path(run).name])))
    scored = subs / "submitting.parquet"
    scored.write_bytes(b"the scored challenge file")
    return SimpleNamespace(root=tmp_path, runs=runs, subs=subs, table=table, scored=scored,
                           decisions=runs / "op_decisions.json", manifest=subs / "operational" / "manifest.json")


def run_id(name: str) -> str:
    """The id `add_run` gives a run of config ``name`` at its current hash, seed 42."""
    return f"{STAMP}__{name}__{_hash(name)}__s42"


def add_run(w, name: str, mae: float, months: dict | None = None, within_120: float = 0.47, finished: bool = True,
            h: str | None = None, seed: int = 42, stamp: str = STAMP, predict: bool = False) -> Path:
    """A run of config ``name`` (at its current hash unless ``h``, seed 42 and STAMP unless given) and its OP metrics in
    the stand-in. Its holdout predictions exist from the start, as a training run writes them before its refit;
    ``finished=False``: no metrics.json yet (still training or refitting); ``predict``: a `prc predict` run."""
    d = w.runs / f"{stamp}__{name}__{h or _hash(name)}__s{seed}"
    d.mkdir()
    (d / "oof_predictions.parquet").write_bytes(b"")
    if finished:
        (d / "metrics.json").write_text("{}", encoding="utf-8")
    if predict:
        (d / "predict.json").write_text("{}", encoding="utf-8")
    w.table[d.name] = {"run": d.name, "n": 1000, "mae": mae, "rmse": 2 * mae, "within_120": within_120, "bias": 0.0,
                       "months": dict(months or {}), "airports": {"EDDF": mae}}
    return d


def checks_world(w, h1: dict[str, float], f1: dict[str, float], fwd: tuple[float, float] = (178.0, 175.0),
                 f1_fwd_months: dict[str, float] | None = None, skip: tuple[str, ...] = (),
                 unfinished: tuple[str, ...] = ()) -> None:
    """H1's and F1's five month-pair runs, whose OP MAE in each validation month is ``h1[month]`` and ``f1[month]``,
    and their forward runs with OP MAE ``fwd`` (H1, F1); F1's forward November and December ``f1_fwd_months``
    (default: its overall MAE). A config in ``skip`` has no run, one in ``unfinished`` no metrics.json yet."""
    specs = []
    for p, ms in PAIR_MONTHS.items():
        for name, values in ((f"{H1}_{p}", h1), (f"{F1}_{p}", f1)):
            months = {m: values[m] for m in ms}
            specs.append((name, sum(months.values()) / 2, months))
    specs.append((f"{H1}_fwd", fwd[0], {"2025-11": fwd[0], "2025-12": fwd[0]}))
    specs.append((f"{F1}_fwd", fwd[1], f1_fwd_months or {"2025-11": fwd[1], "2025-12": fwd[1]}))
    for name, mae, months in specs:
        if name not in skip:
            add_run(w, name, mae, months, finished=name not in unfinished)


def _put(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def _get(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _bytes(w) -> tuple:
    """The decisions, the manifest and the scored file as they are on disk (None where absent)."""
    return tuple(p.read_bytes() if p.exists() else None for p in (w.decisions, w.manifest, w.scored))


def _files(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


GOOD_H1 = {m: 180.0 for m in MONTHS}
GOOD_F1 = {m: 178.0 for m in MONTHS}
"""F1 2 s better than H1 in every month; its forward 175 s against H1's 178 s (the `checks_world` default)."""


# --- the bars ---------------------------------------------------------------------------------------------------------

def test_the_checks_reuse_the_pre_registered_bars_and_the_configs_exist():
    assert oa.PAIRS == tuple(PAIR_MONTHS) and oa.N_MONTHS == 10 and oa.MIN_MONTHS == 8
    assert oa.F_MARGIN == 8.0 and oa.F_REF == {"2025-11": 173.46, "2025-12": 180.25}
    assert (oa.M_MAE_GAIN_S, oa.M_WITHIN_120_GAIN_PP, oa.SEED_FLOOR_MAE) == (1.0, 1.0, 0.25)
    assert (oa.SEED, oa.N_SEEDS) == (42, 3)
    assert (oa.H1, oa.F1, oa.MEDIAN) == (H1, F1, f"{F1}_mae_exact_cpu")
    assert set(oa.F_PAIR) == set(oa.F_REF)
    for c in (H1, F1):
        for p, ms in PAIR_MONTHS.items():  # each pair validates the months the test world (and rule F) gives it
            assert tuple(load_config(f"{c}_{p}.yaml").validation.holdout_months) == ms
        fwd = load_config(f"{c}_fwd.yaml").validation
        assert (fwd.scheme, tuple(fwd.holdout_months)) == ("forward_holdout", ("2025-11", "2025-12"))
    for m, p in oa.F_PAIR.items():
        assert m in PAIR_MONTHS[p]
    assert load_config(f"{oa.MEDIAN}.yaml").model.refit_on_all is True
    assert load_config(f"{F1}.yaml").model.refit_on_all is True  # its refits carry the config's current hash


# --- checks: c4, c5, rule F -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("worse, c4", [(0, True), (2, True), (3, False)])
def test_checks_c4_needs_eight_of_the_ten_months(world, worse, c4):
    delta = {m: (1.5 if i < worse else -2.0) for i, m in enumerate(MONTHS)}
    checks_world(world, GOOD_H1, {m: 180.0 + delta[m] for m in MONTHS})
    assert oa.main(["checks"]) == 0
    dec = _get(world.decisions)[oa.KEY]
    assert dec["months_delta_mae"] == delta and list(dec["months_delta_mae"]) == MONTHS
    assert dec["months_better"] == 10 - worse and dec["c4"] is c4
    assert dec["c5"] is True and dec["passed"] is c4
    assert dec["missing"] == [] and dec["reasons"]["c4"].startswith(f"better in {10 - worse} of 10 months")
    assert dec["months"]["2025-12"] == {"pair": "febdec", "h1": 180.0, "f1": 180.0 + delta["2025-12"],
                                        "delta": delta["2025-12"]}
    assert dec["pairs"] == [{"pair": p, "h1": run_id(f"{H1}_{p}"), "f1": run_id(f"{F1}_{p}")} for p in PAIR_MONTHS]
    assert dec["seed"] == 42 and dec["configs"]["f1"] == F1 and dec["configs"]["h1"].startswith(f"{H1} (")


@pytest.mark.parametrize("f1_fwd, c5", [(177.9, True), (178.0, False), (178.1, False)])
def test_checks_c5_needs_a_lower_forward_mae(world, f1_fwd, c5):
    checks_world(world, GOOD_H1, GOOD_F1, fwd=(178.0, f1_fwd))
    assert oa.main(["checks"]) == 0
    dec = _get(world.decisions)[oa.KEY]
    assert dec["forward"] == {"h1": 178.0, "f1": f1_fwd, "h1_run": run_id(f"{H1}_fwd"), "f1_run": run_id(f"{F1}_fwd")}
    assert dec["c5"] is c5 and dec["c4"] is True and dec["passed"] is c5
    assert dec["reasons"]["c5"] == f"F1 {f1_fwd:.2f} s against H1 178.00 s"


@pytest.mark.parametrize("nov, dec_, tripped", [
    (178.0, 176.0, False),  # November +8 s on F1's own reference: at the tripwire, not above (+18 s on H1's pair)
    (178.5, 176.0, True),   # November +8.5 s
    (178.0, 183.01, True),  # December +8.01 s on F1's own reference (+2.76 s on H1's fixed one)
])
def test_rule_f_reads_f1s_own_pair_months(world, nov, dec_, tripped):
    """F1's references are its _marnov November (170 s) and its _febdec December (175 s); H1's pair months (160 and
    190 s) and H1's fixed references (173.46 and 180.25 s) would decide otherwise."""
    h1 = {**GOOD_H1, "2025-11": 160.0, "2025-12": 190.0}
    f1 = {**GOOD_F1, "2025-11": 170.0, "2025-12": 175.0}
    checks_world(world, h1, f1, fwd=(178.0, 177.0), f1_fwd_months={"2025-11": nov, "2025-12": dec_})
    assert oa.main(["checks"]) == 0
    rf = _get(world.decisions)[oa.KEY]["rule_f"]
    assert rf["reference"] == {"2025-11": 170.0, "2025-12": 175.0}
    assert rf["reference_runs"] == {"2025-11": run_id(f"{F1}_marnov"), "2025-12": run_id(f"{F1}_febdec")}
    assert rf["forward_run"] == run_id(f"{F1}_fwd")
    assert rf["forward_months"] == {"2025-11": nov, "2025-12": dec_}
    assert rf["excess"] == {"2025-11": round(nov - 170.0, 3), "2025-12": round(dec_ - 175.0, 3)}
    assert rf["tripped"] is tripped and rf["margin"] == 8.0 and "reason" not in rf
    assert rf["excess_vs_h1_ref"] == {"2025-11": round(nov - 173.46, 3), "2025-12": round(dec_ - 180.25, 3)}
    assert rf["h1_reference"] == {"2025-11": 173.46, "2025-12": 180.25}
    assert ("limit" in rf["action"]) is tripped


def test_checks_read_the_newest_finished_seed_42_run_only(world):
    """The seed is not part of the config hash (F1's three refits share one). A later run of another seed, a later
    seed-42 run still training (its holdout predictions written, metrics.json not yet) and a predict run are never
    read: each of them would turn a criterion or a reference."""
    checks_world(world, GOOD_H1, GOOD_F1)
    for name, mae in ((f"{F1}_febdec", 400.0), (f"{F1}_marnov", 400.0), (f"{H1}_aproct", 100.0),
                      (f"{H1}_fwd", 100.0), (f"{F1}_fwd", 400.0)):
        ms = ("2025-11", "2025-12") if name.endswith("_fwd") else PAIR_MONTHS[name.rsplit("_", 1)[1]]
        add_run(world, name, mae, {m: mae for m in ms}, seed=43, stamp=LATER)
    add_run(world, f"{F1}_junaug", 400.0, {m: 400.0 for m in PAIR_MONTHS["junaug"]}, stamp=LATER, finished=False)
    add_run(world, f"{F1}_maysep", 400.0, {m: 400.0 for m in PAIR_MONTHS["maysep"]}, stamp=LATER, predict=True)
    assert oa.main(["checks"]) == 0
    dec = _get(world.decisions)[oa.KEY]
    assert dec["pairs"] == [{"pair": p, "h1": run_id(f"{H1}_{p}"), "f1": run_id(f"{F1}_{p}")} for p in PAIR_MONTHS]
    assert (dec["forward"]["h1_run"], dec["forward"]["f1_run"]) == (run_id(f"{H1}_fwd"), run_id(f"{F1}_fwd"))
    assert dec["months_better"] == 10 and dec["c4"] is True and dec["c5"] is True and dec["missing"] == []
    rf = dec["rule_f"]
    assert rf["reference_runs"] == {"2025-11": run_id(f"{F1}_marnov"), "2025-12": run_id(f"{F1}_febdec")}
    assert rf["reference"] == {"2025-11": 178.0, "2025-12": 178.0} and rf["forward_run"] == run_id(f"{F1}_fwd")


@pytest.mark.parametrize("gone, unfinished, rule_f", [
    (f"{F1}_junaug", False, False),
    (f"{H1}_marnov", False, False),   # F1's November reference is still there
    (f"{F1}_aproct", True, False),    # holdout predictions without metrics.json: still training, not finished
    (f"{F1}_marnov", False, None),    # no November reference: rule F unresolved, not passed
])
def test_checks_a_missing_pair_run_fails_c4_and_is_named(world, gone, unfinished, rule_f):
    checks_world(world, GOOD_H1, GOOD_F1, skip=() if unfinished else (gone,), unfinished=(gone,) if unfinished else ())
    assert oa.main(["checks"]) == 0
    dec = _get(world.decisions)[oa.KEY]
    assert dec["missing"] == [gone] and dec["c4"] is False and dec["passed"] is False and dec["c5"] is True
    assert len(dec["months_delta_mae"]) == 8 and dec["months_better"] == 8
    assert dec["reasons"]["c4"] == f"better in 8 of 8 months measured, at least 8 of 10 needed; missing: {gone}"
    assert {"pair": gone.rsplit("_", 1)[1], "missing": [gone]} in dec["pairs"]
    rf = dec["rule_f"]
    assert rf["tripped"] is rule_f
    if rule_f is None:
        assert rf["reference"]["2025-11"] is None and rf["excess"]["2025-11"] is None
        assert rf["reason"] == f"unresolved: no reference 2025-11 ({gone})"
    assert (world.runs / oa.OUT_MD).exists()


@pytest.mark.parametrize("gone", [f"{H1}_fwd", f"{F1}_fwd"])
def test_checks_a_missing_forward_run_fails_c5_and_is_named(world, gone):
    checks_world(world, GOOD_H1, GOOD_F1, skip=(gone,))
    assert oa.main(["checks"]) == 0
    dec = _get(world.decisions)[oa.KEY]
    assert dec["missing"] == [gone] and dec["c5"] is False and dec["c4"] is True and dec["passed"] is False
    assert dec["reasons"]["c5"] == f"missing: {gone}"
    side = "h1" if gone.startswith(H1) else "f1"
    assert dec["forward"][side] is None and dec["forward"][f"{side}_run"] is None
    rf = dec["rule_f"]
    if side == "f1":
        assert rf["tripped"] is None and rf["forward_run"] is None
        assert rf["forward_months"] == {"2025-11": None, "2025-12": None}
        assert rf["reason"] == f"unresolved: no forward 2025-11 ({gone}), no forward 2025-12 ({gone})"
    else:
        assert rf["tripped"] is False


def test_checks_without_any_run_records_every_gap(world):
    assert oa.main(["checks"]) == 0
    dec = _get(world.decisions)[oa.KEY]
    assert dec["c4"] is False and dec["c5"] is False and dec["passed"] is False and dec["rule_f"]["tripped"] is None
    assert dec["months_delta_mae"] == {} and len(dec["missing"]) == 12
    md = (world.runs / oa.OUT_MD).read_text(encoding="utf-8")
    assert "- Missing runs: op_h1_r4_nodoy_febdec, op_h1_f1_plan_febdec, op_h1_r4_nodoy_marnov," in md
    assert "2025-11 n/a s (n/a against n/a from _marnov)" in md


def test_checks_merge_the_record_and_write_the_read_out(world, capsys):
    before = {"L": {"choice": "op_h1_r4_nodoy.yaml"}, "T": {"passed": True, "member": "op_h1_f4_nodepthru"}}
    _put(world.decisions, before)
    checks_world(world, GOOD_H1, {**GOOD_F1, "2025-08": 181.0}, f1_fwd_months={"2025-11": 176.0, "2025-12": 174.5})
    capsys.readouterr()
    assert oa.main(["checks"]) == 0
    doc = _get(world.decisions)
    assert {k: v for k, v in doc.items() if k != oa.KEY} == before and doc[oa.KEY]["passed"] is True
    md = (world.runs / oa.OUT_MD).read_text(encoding="utf-8")
    lines = md.splitlines()
    assert lines[0] == "# F1's adoption checks (findings 10.37)"
    assert lines[2].startswith("OP MAE (s) per validation month of the 2025 month pairs, seed 42: ")
    assert f"(`{H1}_<pair>`, the H1 deliverable until 13 Oct)." in lines[2]
    rows = [ln for ln in lines if ln.startswith("| 2025-")]
    assert [r.split(" | ")[0][2:] for r in rows] == MONTHS
    assert "| 2025-03 | marnov | 180.00 | 178.00 | -2.00 |" in rows
    assert "| 2025-08 | junaug | 180.00 | 181.00 | +1.00 |" in rows
    assert "- Month pairs (c4): better in 9 of 10 months measured, at least 8 of 10 needed: True." in lines
    assert ("- Forward, fit January-October, validate November and December (c5): F1 175.00 s against H1 178.00 s: "
            "True.") in lines
    assert ("- Rule F with F1's own references (tripwire +8 s): 2025-11 -2.00 s (176.00 against 178.00 from _marnov), "
            "2025-12 -3.50 s (174.50 against 178.00 from _febdec): tripped False.") in lines
    assert ("- Against H1's fixed references (`op_rules.F_REF`, for information): 2025-11 +2.54 s, 2025-12 -5.75 s."
            in lines)
    assert "- Missing runs: none." in lines
    assert any(ln.startswith("- Passed (c4 and c5): True. F1 is adopted either way") for ln in lines)
    assert md.rstrip() in capsys.readouterr().out
    assert not list(world.runs.rglob("*.tmp")) and _files(world.subs) == ["submitting.parquet"]


# --- rule M -----------------------------------------------------------------------------------------------------------

REF_MAE, REF_W120 = 184.3125, 0.4716
"""F1's calibrated median in C_h1 (OP MAE, share within +-2 min)."""


def _c(seeds: list[str], mae: float = 183.1015, w120: float = 0.4916, tables: str | None = None) -> dict:
    """A rule C check as `lab/op_calibrate.py check` records it (abridged); H1's or F1's tables after the seeds."""
    if tables is None:
        tables = H1_TABLES if f"__{H1}__" in seeds[0] else F1_TABLES
    return {"n": 338277, "mean": {"mae": 186.6846, "within_120": 0.4662}, "median": {"mae": mae, "within_120": w120},
            "passed": True, "runs": list(seeds), "tables": tables}


def _m(run: str) -> dict:
    """A rule M record (abridged)."""
    return {"passed": True, "candidate": {"run": run, "mae": 181.16, "within_120": 0.5023}, "mae_gain": 1.94,
            "within_120_gain_pp": 1.06, "bar": {"mae_gain": 1.0, "within_120_gain_pp": 1.0}}


def _decisions_doc() -> dict:
    """runs/op_decisions.json before the archive (abridged): H1's C_h1 and M among the other rules' keys."""
    return {"C_h1_preview_s42": {"runs": H1_SEEDS[:1]}, "L": {"choice": "op_h1_r4_nodoy.yaml"}, "C_h1": _c(H1_SEEDS),
            "F": {"tripped": False}, "R_hl3": {"choice": "op_hl3_r4_rwyblind.yaml"}, "C_hl3": {"passed": False},
            "M": _m(H1_M_RUN), "T13": {"passed": True}, "T": {"passed": True, "member": "op_h1_f4_nodepthru"}}


def _file(seeds: list[str], name: str, median: dict | None = None) -> dict:
    """A delivery manifest record as `lab/op_deliver.py` writes it (abridged)."""
    return {"file": f"submissions\\operational\\{name}.parquet", "sha1": "0" * 40, "rows": 670790,
            "source_runs": list(seeds), "blend": name, "median_member": median}


def _manifest_doc() -> dict:
    """submissions/operational/manifest.json before the archive (abridged)."""
    return {"h1": _file(H1_SEEDS, "h1__b__mae", {"source_run": "x", "config": f"{H1}_mae_exact_cpu.yaml"}),
            "hl3": _file(["2026-10-11T15-20-52__op_hl3_r4_rwyblind__c93d69e3__s42"], "hl3__b"),
            "hl24": _file(["2026-10-11T15-57-44__op_hl24_r3_rwyblind__930c7b15__s42"], "hl24__b"),
            "h1_calibrated_median": _file(H1_SEEDS, "h1__b"),
            "h1f": {**_file(F4_SEEDS, "h1f__b"), "tier": "F: ..."}}


def rule_m_world(w, cand: tuple[float, float] | None = (181.0, 0.52), c_h1: dict | None = None,
                 archived: bool = True, n_refits: int = 3, **run_kw) -> dict:
    """The decisions after the archive and F1's calibration check; F1's first ``n_refits`` refitted seeds; the meta
    files of H1's and F1's tables (`lab/op_calibrate.py fit`: their five month-pair runs); and the median member's run
    with OP MAE and share within +-2 min ``cand`` (None: no run)."""
    doc = {**_decisions_doc(), "C_h1": c_h1 if c_h1 is not None else _c(F1_SEEDS, REF_MAE, REF_W120),
           "C_h1_r4_nodoy": _c(H1_SEEDS)}
    if archived:
        doc["M_r4_nodoy"] = _m(H1_M_RUN)
    _put(w.decisions, doc)
    for r in F1_SEEDS[:n_refits]:  # what `op_rules.refits` reads: refit.json and test_predictions.parquet
        d = w.runs / r
        d.mkdir()
        for f in ("oof_predictions.parquet", "test_predictions.parquet"):
            (d / f).write_bytes(b"")
        for f in ("metrics.json", "refit.json"):
            (d / f).write_text("{}", encoding="utf-8")
    for tables, cfg in ((H1_TABLES, H1), (F1_TABLES, F1)):
        meta = (w.root / tables).with_suffix(".json")
        meta.parent.mkdir(exist_ok=True)
        _put(meta, {"pairs": [run_id(f"{cfg}_{p}") for p in PAIR_MONTHS], "months": MONTHS, "n_fit": 1721301})
    if cand is not None:
        add_run(w, oa.MEDIAN, cand[0], within_120=cand[1], **run_kw)
    return doc


@pytest.mark.parametrize("mae, w120, passed, unresolved", [
    (183.3125, 0.4816, False, True),   # exactly 1 s and 1 point: within the seed floor (0.25 s) of the 1 s bar
    (183.3135, 0.4816, False, True),   # 0.999 s
    (183.0625, 0.4816, False, True),   # 1.25 s: on the floor's upper edge, still unresolved
    (183.5625, 0.55, False, True),     # 0.75 s: on its lower edge
    (183.0624, 0.4816, True, False),   # 1.2501 s, beyond the floor, and exactly 1 point
    (183.0525, 0.4816, True, False),   # 1.26 s
    (183.0525, 0.4815, False, False),  # 1.26 s and 0.99 point
    (180.0, 0.55, True, False),
    (184.0, 0.47, False, False),       # 0.3125 s
])
def test_rule_m_bars_and_seed_floor(world, capsys, mae, w120, passed, unresolved):
    assert 100 * (0.4816 - REF_W120) < 1.0, "in floats the exact point falls short: the script compares rounded values"
    doc = rule_m_world(world, (mae, w120))
    capsys.readouterr()
    assert oa.main(["rule-m"]) == 0
    got = _get(world.decisions)
    m = got["M"]
    assert m["passed"] is passed and m["unresolved"] is unresolved
    assert m["mae_gain"] == round(REF_MAE - mae, 4) and m["within_120_gain_pp"] == round(100 * (w120 - REF_W120), 2)
    assert m["reference"] == {"what": "F1's calibrated median (C_h1, three seeds)", "mae": REF_MAE,
                              "within_120": REF_W120}
    assert m["reference_runs"] == F1_SEEDS and m["reference_tables"] == F1_TABLES
    assert m["bar"] == {"mae_gain": 1.0, "within_120_gain_pp": 1.0} and m["seed_floor_mae"] == 0.25
    assert m["candidate"] == world.table[run_id(oa.MEDIAN)] and m["rule_c_passed"] is True
    assert {k: v for k, v in got.items() if k != "M"} == {k: v for k, v in doc.items() if k != "M"}
    assert ("seed floor" in m["action"]) is unresolved
    line = capsys.readouterr().out.strip().splitlines()[-1]
    word = "passed" if passed else ("not passed (unresolved)" if unresolved else "not passed")
    assert line.startswith(f"rule M for F1: {word}: ")
    assert (f"--median-config {oa.MEDIAN}.yaml" in line) is passed and line.endswith(m["action"])


def test_rule_m_without_rule_c_has_no_median_to_replace(world, capsys):
    """Rule C failed for F1: the file holds the mean only (lab/op_deliver.py --median-config needs the band)."""
    rule_m_world(world, (180.0, 0.55), c_h1={**_c(F1_SEEDS, REF_MAE, REF_W120), "passed": False})
    capsys.readouterr()
    assert oa.main(["rule-m"]) == 0
    m = _get(world.decisions)["M"]
    assert m["passed"] is True and m["rule_c_passed"] is False and m["action"].startswith("rule C did not pass")
    assert "--median-config" not in capsys.readouterr().out.strip().splitlines()[-1]


def test_rule_m_reads_the_seed_42_candidate_only(world):
    """A later run of another seed, a later seed-42 run still refitting and a predict run are not the candidate."""
    rule_m_world(world, (181.0, 0.52))
    add_run(world, oa.MEDIAN, 170.0, within_120=0.60, seed=43, stamp=LATER)
    add_run(world, oa.MEDIAN, 170.0, within_120=0.60, stamp=LATER, finished=False)
    add_run(world, oa.MEDIAN, 170.0, within_120=0.60, stamp="2026-10-13T23-30-00", predict=True)
    assert oa.main(["rule-m"]) == 0
    m = _get(world.decisions)["M"]
    assert m["candidate"]["run"] == run_id(oa.MEDIAN) and m["mae_gain"] == round(REF_MAE - 181.0, 4)


@pytest.mark.parametrize("absolute", [False, True])
def test_rule_m_reads_the_tables_meta_by_a_relative_or_an_absolute_path(world, absolute):
    tables = str(world.root / F1_TABLES) if absolute else F1_TABLES
    rule_m_world(world, (180.0, 0.55), c_h1=_c(F1_SEEDS, REF_MAE, REF_W120, tables))
    assert oa.main(["rule-m"]) == 0
    m = _get(world.decisions)["M"]
    assert m["passed"] is True and m["reference_tables"] == tables


RULE_M_REFUSALS = {
    "M is not archived": ({"archived": False}, "op_adopt.py archive"),
    "C_h1 is H1's": ({"c_h1": _c(H1_SEEDS)}, "lab/op_calibrate.py check"),
    "C_h1 is absent": ({}, "lab/op_calibrate.py check"),
    "C_h1 of one seed": ({"c_h1": _c(F1_SEEDS[:1], REF_MAE, REF_W120)}, "not of the 3 seeds"),
    "C_h1 of one seed three times": ({"c_h1": _c(F1_SEEDS[:1] * 3, REF_MAE, REF_W120)}, "not of the 3 seeds"),
    "C_h1 of rule T's stage-1 run": ({"c_h1": _c([F1_STAGE1], REF_MAE, REF_W120)}, "not of the 3 seeds"),
    "F1's seeds with H1's tables": ({"c_h1": _c(F1_SEEDS, REF_MAE, REF_W120, H1_TABLES)}, "five month pairs"),
    "tables without a meta file": ({"c_h1": _c(F1_SEEDS, REF_MAE, REF_W120, "runs/op_calib/f1_other.parquet")},
                                   "no meta file"),
    "two of F1's seeds refitted": ({"n_refits": 2}, "only 2 refitted runs"),
    "no candidate run": ({"cand": None}, oa.MEDIAN),
    "the candidate is still training": ({"finished": False}, oa.MEDIAN),
    "the candidate is of another hash": ({"h": "deadbeef"}, oa.MEDIAN),
    "the candidate is of another seed": ({"seed": 43}, oa.MEDIAN),
}


@pytest.mark.parametrize("case", list(RULE_M_REFUSALS))
def test_rule_m_refuses_and_writes_nothing(world, case):
    kw, match = RULE_M_REFUSALS[case]
    doc = rule_m_world(world, **kw)
    if case == "C_h1 is absent":
        _put(world.decisions, {k: v for k, v in doc.items() if k != "C_h1"})
    before = _bytes(world)
    with pytest.raises(SystemExit, match=match):
        oa.main(["rule-m"])
    assert _bytes(world) == before and not list(world.runs.rglob("*.tmp"))


# --- archive ----------------------------------------------------------------------------------------------------------

def test_archive_keeps_the_superseded_records_and_a_second_call_changes_nothing(world, capsys):
    doc, man = _decisions_doc(), _manifest_doc()
    _put(world.decisions, doc)
    _put(world.manifest, man)
    scored = world.scored.read_bytes()
    capsys.readouterr()
    assert oa.main(["archive"]) == 0
    # copied: C_h1 and M stay until F1's calibration check and rule M replace them; every other key as it was
    assert _get(world.decisions) == {**doc, "C_h1_r4_nodoy": doc["C_h1"], "M_r4_nodoy": doc["M"]}
    assert _get(world.manifest) == {"hl3": man["hl3"], "hl24": man["hl24"], "h1f": man["h1f"], "h1_r4_nodoy": man["h1"],
                                    "h1_r4_nodoy_calibrated_median": man["h1_calibrated_median"]}
    assert world.scored.read_bytes() == scored
    assert _files(world.subs) == ["operational/manifest.json", "submitting.parquet"]  # no temporary file left
    assert _files(world.runs) == ["op_decisions.json"]
    out = capsys.readouterr().out
    for line in ("archive: C_h1 copied to C_h1_r4_nodoy",
                 "archive: M copied to M_r4_nodoy",
                 "archive: h1 moved to h1_r4_nodoy",
                 "archive: h1_calibrated_median moved to h1_r4_nodoy_calibrated_median"):
        assert line in out
    before = _bytes(world)
    assert oa.main(["archive"]) == 0
    assert _bytes(world) == before and _files(world.runs) == ["op_decisions.json"]
    assert "archive: nothing to do" in capsys.readouterr().out


def test_archive_after_f1s_delivery_changes_nothing(world, capsys):
    """Once F1's records replaced C_h1, M, h1 and h1_calibrated_median, a call keeps everything as it is."""
    _put(world.decisions, {**_decisions_doc(), "C_h1": _c(F1_SEEDS), "M": _m(F1_M_RUN), "C_h1_r4_nodoy": _c(H1_SEEDS),
                           "M_r4_nodoy": _m(H1_M_RUN)})
    man = _manifest_doc()
    _put(world.manifest, {**{k: man[k] for k in ("hl3", "hl24", "h1f")},
                          "h1": _file(F1_SEEDS, "h1__f1__mae", {"config": f"{oa.MEDIAN}.yaml"}),
                          "h1_calibrated_median": _file(F1_SEEDS, "h1__f1"), "h1_r4_nodoy": man["h1"],
                          "h1_r4_nodoy_calibrated_median": man["h1_calibrated_median"]})
    before = _bytes(world)
    capsys.readouterr()
    assert oa.main(["archive"]) == 0
    assert _bytes(world) == before
    out = capsys.readouterr().out
    assert "archive: nothing to do" in out and f"h1 is {F1}'s" in out


@pytest.mark.parametrize("h1", ["absent", "F1's"])
def test_archive_moves_a_calibrated_median_left_behind(world, capsys, h1):
    """h1 was moved before (by hand) and H1's calibrated median was not: the archive moves it, so that F1's two-step
    delivery cannot overwrite it and a one-step delivery cannot leave it beside F1's file; a second call changes
    nothing."""
    doc, man = _decisions_doc(), _manifest_doc()
    man["h1_r4_nodoy"] = man.pop("h1")
    if h1 == "F1's":
        man["h1"] = _file(F1_SEEDS, "h1__f1")
    _put(world.decisions, doc)
    _put(world.manifest, man)
    capsys.readouterr()
    assert oa.main(["archive"]) == 0
    cal = man.pop("h1_calibrated_median")
    assert _get(world.manifest) == {**man, "h1_r4_nodoy_calibrated_median": cal}
    assert _get(world.decisions) == {**doc, "C_h1_r4_nodoy": doc["C_h1"], "M_r4_nodoy": doc["M"]}
    out = capsys.readouterr().out
    assert "archive: h1_calibrated_median moved to h1_r4_nodoy_calibrated_median" in out
    assert "archive: h1 moved" not in out
    before = _bytes(world)
    assert oa.main(["archive"]) == 0
    assert _bytes(world) == before and "archive: nothing to do" in capsys.readouterr().out


def _move_h1(man: dict) -> None:
    man["h1_r4_nodoy"] = man.pop("h1")
    man["h1_r4_nodoy_calibrated_median"] = man.pop("h1_calibrated_median")


REFUSALS = {
    "a foreign h1": (lambda doc, man: man.update(h1=_file(F4_SEEDS, "h1__f4")), "op_h1_f4_nodepthru"),
    "F1's h1 before the archive": (lambda doc, man: man.update(h1=_file(F1_SEEDS, "h1__f1")), "is gone"),
    "h1 gone": (lambda doc, man: man.pop("h1"), "h1 in .* is absent"),
    "a foreign calibrated median": (lambda doc, man: man.update(h1_calibrated_median=_file(F1_SEEDS, "h1__f1")),
                                    "h1_calibrated_median in"),
    "a foreign h1 after the archive": (lambda doc, man: (_move_h1(man), man.update(h1=_file(F4_SEEDS, "h1__f4"))),
                                       "neither"),
    "a foreign calibrated median after the archive": (
        lambda doc, man: (_move_h1(man), man.update(h1_calibrated_median=_file(F4_SEEDS, "h1__f4"))), "neither"),
    "a foreign archived h1": (lambda doc, man: (_move_h1(man), man.update(h1_r4_nodoy=_file(F4_SEEDS, "h1__f4"))),
                              "h1_r4_nodoy in .* is the record of op_h1_f4_nodepthru, not"),
    "a foreign archived calibrated median": (
        lambda doc, man: (_move_h1(man), man.update(h1_r4_nodoy_calibrated_median=_file(F1_SEEDS, "h1__f1"))),
        f"h1_r4_nodoy_calibrated_median in .* is the record of {F1}, not"),
    "a foreign archived calibrated median before the move": (
        lambda doc, man: man.update(h1_r4_nodoy_calibrated_median=_file(F1_SEEDS, "h1__f1")),
        f"h1_r4_nodoy_calibrated_median in .* is the record of {F1}, not"),
    "H1's h1 under both keys": (lambda doc, man: man.update(h1_r4_nodoy=dict(man["h1"])), "both hold"),
    "H1's calibrated median under both keys": (
        lambda doc, man: (_move_h1(man),
                          man.update(h1_calibrated_median=dict(man["h1_r4_nodoy_calibrated_median"]))), "both hold"),
    "C_h1 rewritten before the archive": (lambda doc, man: doc.update(C_h1=_c(F1_SEEDS)), "--member h1_r4_nodoy"),
    "M gone": (lambda doc, man: doc.pop("M"), "M in .* is absent"),
    "a foreign archive": (lambda doc, man: doc.update(C_h1_r4_nodoy=_c(F1_SEEDS)), "C_h1_r4_nodoy in"),
    "no manifest": (None, "no delivery manifest"),
}


@pytest.mark.parametrize("case", list(REFUSALS))
def test_archive_refuses_and_writes_nothing(world, case):
    change, match = REFUSALS[case]
    doc, man = _decisions_doc(), _manifest_doc()
    if change is not None:
        change(doc, man)
        _put(world.manifest, man)
    _put(world.decisions, doc)
    before, files = _bytes(world), _files(world.runs.parent)
    with pytest.raises(SystemExit, match=match):
        oa.main(["archive"])
    assert _bytes(world) == before and _files(world.runs.parent) == files


@pytest.mark.parametrize("cmd", ["archive", "checks", "rule-m"])
def test_a_decisions_file_outside_the_runs_directory_is_refused(world, monkeypatch, cmd):
    """op_rules binds its decisions file when imported: a runs directory moved without it reads and writes nothing."""
    elsewhere = world.runs.parent / "elsewhere" / "op_decisions.json"
    monkeypatch.setattr(oa.op_rules, "DECISIONS", elsewhere)
    rule_m_world(world)
    _put(world.manifest, _manifest_doc())
    checks_world(world, GOOD_H1, GOOD_F1)
    before = _bytes(world)
    with pytest.raises(SystemExit, match="op_rules.DECISIONS"):
        oa.main([cmd])
    assert _bytes(world) == before and not elsewhere.parent.exists() and not (world.runs / oa.OUT_MD).exists()
