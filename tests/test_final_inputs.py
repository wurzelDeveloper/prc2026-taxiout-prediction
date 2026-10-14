"""`lab/final_inputs.py`: the final data tree, the released-file check and the runbook script, on synthetic files only.

`validate` must accept a ranking file shaped like the delivered one and reject each defect a released file could have
(columns, types, ids, months, arrivals, airports, emptied columns, renamed labels, the template, missing weather or
ERA5); `init` copies what predicting needs into a new tree and records it, `install` only into such a tree; the runbook
expands a blend of blends into its trained runs and nested blends and writes a script that sets its own data tree,
stops on the first failure and re-uses what an earlier start made; `latest`, `derive` and `check` are what that script
calls. No competition data.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import synthetic_tree as st

from prc import paths

FINAL = ["2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]


@pytest.fixture(scope="module")
def fi():
    spec = importlib.util.spec_from_file_location("final_inputs_t", paths.ROOT / "lab" / "final_inputs.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # its dataclasses resolve their annotations through sys.modules
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop(spec.name, None)


def _era5(path, airports, start="2024-12-24", end="2026-07-31 23:00"):
    hours = pd.date_range(start, end, freq="h", tz="UTC")
    frame = pd.DataFrame({"apt": np.repeat(list(airports), len(hours)), "time": np.tile(hours, len(airports))})
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


@pytest.fixture(scope="module")
def released(tmp_path_factory):
    """A source tree (training files plus the additional data `init` copies) and a clean Feb-Jun ranking file."""
    root = tmp_path_factory.mktemp("final_inputs")
    source = root / "source"
    st.write_training(source, per_month=1, wrong_date=0)
    frame = st.ranking_frame(FINAL, seed=11, id_start=3_000_000, per_month=1)
    out = root / "released"
    st.write_ranking(out, frame)
    add = source / "additional_data"
    for kind in ("metar", "taf"):
        for station in st.AIRPORTS:
            for month in ["2025-12", *st.TRAIN_MONTHS[:1], "2026-01", *FINAL, "2026-07", "2026-08"]:   # + July's look-ahead
                f = add / "weather" / "raw" / kind / f"{station}_{month}.csv"
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(f"{kind},{station},{month}\n", encoding="utf-8")
    (add / "weather" / "manifest.json").write_text("{}", encoding="utf-8")
    (add / "weather" / "metar.parquet").write_bytes(b"normalised: rebuilt in the new tree, not copied")
    for sub, name in (("airport_geometry/raw/osm", "EDDF.json"), ("airport_geometry", "stands.parquet"),
                      ("reanalysis/raw", "EDDF.json"), ("pru", "pru_monthly.parquet"), ("adsb/points", "2026-02-01.parquet")):
        f = add / sub / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(f"{sub}/{name}", encoding="utf-8")
    (add / "airport_geometry" / "manifest.json").write_text(json.dumps({"stands_file": None}), encoding="utf-8")
    _era5(add / "reanalysis" / "era5_hourly.parquet", st.AIRPORTS)
    (source / "competition_files" / "ranking.parquet").write_bytes(b"the old ranking file: never copied")
    return {"root": root, "source": source, "frame": frame, "ranking": out / "competition_files" / "ranking.parquet",
            "template": out / "competition_files" / "submitting.parquet"}


def _write(frame: pd.DataFrame, path) -> object:
    return st.write_parquet(frame, path)


def _check(fi, released, path, **kw):
    return fi.validate(path, months=kw.pop("months", FINAL), training_dir=released["source"] / "competition_files", **kw)


# --- validate -------------------------------------------------------------------------------------------------------

def test_a_file_shaped_like_the_delivered_one_passes(fi, released):
    report = _check(fi, released, released["ranking"], template=released["template"])
    assert report.ok, report.errors
    frame = released["frame"]
    assert report.info["rows"] == len(frame) and report.info["departures"] == int(frame["PHASE_mvt"].eq("DEP").sum())
    assert sorted(report.info["by_month"]) == FINAL and report.info["airports"] == list(st.AIRPORTS)
    assert report.info["expected_rank_rows_env"] == f"PRC_EXPECTED_RANK_ROWS={len(frame)}"
    assert not report.warnings, report.warnings
    assert 0.5 < report.info["departures_with_lobt"] < 1.0
    ctx = report.info["context"]
    assert ctx["departures_within_lookback_of_start"] > 0 and "2026-02-02" in ctx["missing_days"]
    assert set(report.info["stale_suspect_share"]) == {"release", "training", "baseline"}


def test_the_delivered_schema_constant_matches_the_synthetic_writer(fi):
    """Both encode ranking.parquet's columns; the test files are only meaningful if they agree."""
    assert list(fi.RANKING_SCHEMA) == [(n, str(t)) for n, t in zip(st.SCHEMA.names, st.SCHEMA.types)]


@pytest.mark.parametrize(
    "defect, match",
    [
        ("missing_column", "missing columns \\['STAND_mvt'\\]"),
        ("extra_column", "unexpected columns \\['NOTE'\\]"),
        ("wrong_type", "MVT_ID_mvt is int64, expected double"),
        ("duplicate_ids", "duplicated ids"),
        ("training_ids", "are training ids"),
        ("outside_months", "outside the expected months"),
        ("no_arrivals", "no arrivals"),
        ("no_departures", "no departures"),
        ("bad_phase", "neither DEP nor ARR"),
        ("no_airport", "without a reporting airport"),
        ("fractional_ids", "not integral"),
        ("new_airport", "airports not in the training data"),
        ("no_nm_times", "LOBT_flt empty on 100.0 %"),
        ("no_stand_runway", "STAND_mvt empty on 100.0 %"),
        ("no_types", "AIRCRAFT_TYPE_mvt empty on 100.0 %"),
        ("arrivals_blank", "TAXITIME_SEC_mvt empty on 100.0 % of the arr rows"),
        ("renamed_rule", "FLIGHT_RULE_mvt: labels the training data never had"),
        ("renamed_runway", "RUNWAY_mvt at LIRF: designators the training data never had"),
    ],
)
def test_each_defect_is_rejected(fi, released, tmp_path, defect, match):
    frame = released["frame"].copy()
    path = tmp_path / f"{defect}.parquet"
    dep = frame["PHASE_mvt"].eq("DEP")
    if defect == "missing_column":
        table = pa.Table.from_pandas(frame[st.SCHEMA.names], schema=st.SCHEMA, preserve_index=False).drop(["STAND_mvt"])
        pq.write_table(table, path)
    elif defect == "extra_column":
        table = pa.Table.from_pandas(frame[st.SCHEMA.names], schema=st.SCHEMA, preserve_index=False)
        pq.write_table(table.append_column("NOTE", pa.array(["x"] * len(frame))), path)
    elif defect == "wrong_type":
        table = pa.Table.from_pandas(frame[st.SCHEMA.names], schema=st.SCHEMA, preserve_index=False)
        pq.write_table(table.set_column(0, "MVT_ID_mvt", pa.array(frame["MVT_ID_mvt"].astype("int64"))), path)
    else:
        if defect == "duplicate_ids":
            frame.loc[frame.index[1], "MVT_ID_mvt"] = frame["MVT_ID_mvt"].iloc[0]
        elif defect == "training_ids":
            frame.loc[frame.index[:3], "MVT_ID_mvt"] = [1.0, 2.0, 3.0]  # the synthetic training ids start at 1
        elif defect == "outside_months":
            frame.loc[frame.index[:5], "MVT_TIME_UTC_mvt"] = pd.Timestamp("2026-07-02 10:00", tz="UTC")
        elif defect == "no_arrivals":
            frame = frame.loc[dep]
        elif defect == "no_departures":
            frame = frame.loc[~dep]
        elif defect == "bad_phase":
            frame.loc[frame.index[:2], "PHASE_mvt"] = "OVF"
        elif defect == "no_airport":
            frame.loc[frame.index[dep][:2], "ADEP_mvt"] = None
        elif defect == "fractional_ids":
            frame.loc[frame.index[0], "MVT_ID_mvt"] = 3_000_000.5
        elif defect == "new_airport":
            frame.loc[frame.index[dep][:4], "ADEP_mvt"] = "LTAI"
        elif defect == "no_nm_times":
            for col in ("LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt"):
                frame[col] = pd.NaT
        elif defect == "no_stand_runway":
            frame["STAND_mvt"] = None
            frame["RUNWAY_mvt"] = None
        elif defect == "no_types":
            frame["AIRCRAFT_TYPE_mvt"] = None
        elif defect == "arrivals_blank":
            frame.loc[~dep, "TAXITIME_SEC_mvt"] = pd.NA
            frame.loc[~dep, "BLOCK_TIME_UTC_mvt"] = pd.NaT
        elif defect == "renamed_rule":
            frame["FLIGHT_RULE_mvt"] = "IFR"
        elif defect == "renamed_runway":
            lirf = frame["ADEP_mvt"].where(dep, frame["ADES_mvt"]).eq("LIRF")
            frame.loc[lirf, "RUNWAY_mvt"] = frame.loc[lirf, "RUNWAY_mvt"].map({"16L": "16L/34R", "25": "25R"})
        _write(frame, path)
    report = _check(fi, released, path)
    assert not report.ok
    assert any(re.search(match, e) for e in report.errors), report.errors
    if defect in ("no_nm_times", "renamed_rule"):   # content findings can be demoted, structural ones never
        assert _check(fi, released, path, content_as_warnings=True).ok
    if defect == "new_airport":
        assert not _check(fi, released, path, content_as_warnings=True).ok


def test_template_defects_are_rejected(fi, released, tmp_path):
    dep_ids = released["frame"].loc[released["frame"]["PHASE_mvt"].eq("DEP"), "MVT_ID_mvt"].to_numpy()
    schema = pa.schema([("MVT_ID_mvt", pa.float64()), ("TAXITIME_SEC_mvt", pa.int32())])

    def template(ids, name, schema=schema):
        path = tmp_path / name
        pq.write_table(pa.table({"MVT_ID_mvt": pa.array(ids, pa.float64()),
                                 "TAXITIME_SEC_mvt": pa.array([None] * len(ids), schema.field("TAXITIME_SEC_mvt").type)},
                                schema=schema), path)
        return path

    stray = template(np.append(dep_ids, 9_999_999.0), "stray.parquet")
    report = _check(fi, released, released["ranking"], template=stray)
    assert any("not departures" in e for e in report.errors)
    report = _check(fi, released, released["ranking"], template=template(dep_ids[:-2], "short.parquet"))
    assert report.ok and any("not in short.parquet" in w for w in report.warnings)
    wide = pa.schema([("MVT_ID_mvt", pa.float64()), ("TAXITIME_SEC_mvt", pa.int64())])
    report = _check(fi, released, released["ranking"], template=template(dep_ids, "int64.parquet", wide))
    assert any("TAXITIME_SEC_mvt is int64, expected int32" in e for e in report.errors)


def test_soft_findings_are_warnings(fi, released, tmp_path):
    frame = released["frame"].copy()
    frame = frame.loc[~frame["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m").eq("2026-06")]   # one month absent
    dep = frame["PHASE_mvt"].eq("DEP")
    frame.loc[dep, "TAXITIME_SEC_mvt"] = 600                                            # a departure target left in
    path = _write(frame, tmp_path / "soft.parquet")
    report = _check(fi, released, path)
    assert report.ok
    assert any("2026-06" in w for w in report.warnings)
    assert any("departures carry a taxi time" in w for w in report.warnings)
    unreadable = tmp_path / "broken.parquet"
    unreadable.write_bytes(b"not parquet")
    assert not _check(fi, released, unreadable).ok


def test_against_the_delivered_file_a_moved_share_is_a_warning(fi, released, tmp_path):
    """--baseline: the delivered Jan+Jul file is the reference for the empty shares and the stale segment."""
    base = st.ranking_frame(["2026-01", "2026-07"], seed=12, id_start=6_000_000, per_month=8)
    base_path = _write(base, tmp_path / "jan_jul.parquet")
    frame = st.ranking_frame(FINAL, seed=13, id_start=7_000_000, per_month=8)
    dep = frame.index[frame["PHASE_mvt"].eq("DEP")]
    frame.loc[dep[: len(dep) // 6], "LOBT_flt"] = pd.NaT                     # some 17 % of the departures lose LOBT
    report = _check(fi, released, _write(frame, tmp_path / "moved.parquet"), baseline=base_path)
    assert report.ok, report.errors
    assert any(w.startswith("LOBT_flt empty") and "delivered Jan+Jul" in w for w in report.warnings), report.warnings
    assert report.info["stale_suspect_share"]["baseline"] is not None


def test_with_the_tree_the_weather_and_era5_coverage_is_checked(fi, released, tmp_path):
    tree = tmp_path / "final"
    fi.init(tree, released["source"])
    assert _check(fi, released, released["ranking"], tree=tree).ok
    (tree / "additional_data" / "weather" / "raw" / "metar" / "LSZH_2026-04.csv").unlink()
    report = _check(fi, released, released["ranking"], tree=tree)
    assert any("raw METAR missing for ['LSZH_2026-04']" in e for e in report.errors), report.errors
    fi.init(tree, released["source"])                                      # the copy completes again
    _era5(tree / "additional_data" / "reanalysis" / "era5_hourly.parquet", st.AIRPORTS, end="2026-05-31 23:00")
    report = _check(fi, released, released["ranking"], tree=tree)
    assert any(e.startswith("ERA5 does not cover") for e in report.errors), report.errors


def test_cli_validate_and_install(fi, released, tmp_path, capsys):
    tree = tmp_path / "tree"
    fi.init(tree, released["source"])
    capsys.readouterr()
    args = ["validate", "--file", str(released["ranking"]), "--template", str(released["template"]), "--months", *FINAL]
    assert fi.main([*args, "--install", "--tree", str(tree)]) == 0
    out = capsys.readouterr().out
    assert "OK" in out and f'$env:PRC_EXPECTED_RANK_ROWS = "{len(released["frame"])}"' in out
    assert (tree / "competition_files" / "ranking.parquet").read_bytes() == released["ranking"].read_bytes()
    assert (tree / "competition_files" / "submitting.parquet").exists()
    assert not list((tree / "competition_files").glob("*.part"))
    bad = released["frame"].loc[released["frame"]["PHASE_mvt"].eq("DEP")]
    bad_path = _write(bad, tmp_path / "bad.parquet")
    assert fi.main(["validate", "--file", str(bad_path), "--months", *FINAL, "--install", "--tree", str(tree)]) == 1
    assert "REJECTED" in capsys.readouterr().out
    assert (tree / "competition_files" / "ranking.parquet").read_bytes() == released["ranking"].read_bytes()


def test_install_goes_only_into_a_tree_init_made(fi, released, tmp_path):
    bare = tmp_path / "bare"
    (bare / "competition_files").mkdir(parents=True)
    with pytest.raises(SystemExit, match="init"):
        fi.install(released["ranking"], released["template"], bare)
    selfish = tmp_path / "selfish"
    (selfish / "competition_files").mkdir(parents=True)
    (selfish / fi.MANIFEST).write_text(json.dumps({"source": str(selfish)}), encoding="utf-8")
    with pytest.raises(SystemExit, match="source tree"):
        fi.install(released["ranking"], released["template"], selfish)


# --- init -----------------------------------------------------------------------------------------------------------

def test_init_copies_what_predicting_needs_and_records_it(fi, released, tmp_path):
    tree = tmp_path / "final"
    fi.init(tree, released["source"], dry_run=True)
    assert not tree.exists()
    manifest = fi.init(tree, released["source"])
    for sub in ("competition_files", "additional_data", "interim"):
        assert (tree / sub).is_dir()
    copied = set(manifest["files"])
    assert len([f for f in copied if f.startswith("competition_files/training_")]) == 12
    assert "additional_data/weather/raw/metar/EDDF_2026-04.csv" in copied
    assert "additional_data/weather/metar.parquet" not in copied              # normalised tables are rebuilt
    assert {"additional_data/airport_geometry/stands.parquet", "additional_data/reanalysis/era5_hourly.parquet",
            "additional_data/reanalysis/raw/EDDF.json", "additional_data/pru/pru_monthly.parquet"} <= copied
    assert not any(f.startswith("additional_data/adsb") for f in copied)
    assert not (tree / "competition_files" / "ranking.parquet").exists()
    assert not list(tree.rglob("*.part"))
    recorded = json.loads((tree / fi.MANIFEST).read_text(encoding="utf-8"))
    one = recorded["files"]["additional_data/reanalysis/raw/EDDF.json"]
    assert one["sha1"] == fi._sha1(released["source"] / "additional_data/reanalysis/raw/EDDF.json")
    assert recorded["weather_coverage"]["metar"]["stations"] == list(st.AIRPORTS)
    assert recorded["weather_coverage"]["metar"]["missing"] == []

    again = fi.init(tree, released["source"])                                   # re-run: identical files skipped
    assert again["copied"] == 0 and again["skipped_identical"] == len(copied)
    rebuilt = tree / "additional_data/airport_geometry/stands.parquet"          # step 4 rebuilds the normalised tables
    rebuilt.write_bytes(b"rebuilt in the tree")
    again = fi.init(tree, released["source"])                                   # a restart keeps them, no conflict
    assert again["copied"] == 0 and again["kept_regenerated"] == 1
    assert again["files"]["additional_data/airport_geometry/stands.parquet"]["regenerated_in_tree"]
    assert rebuilt.read_bytes() == b"rebuilt in the tree"
    assert fi.regenerated_in_tree(Path("additional_data/weather/manifest.json"))
    assert not fi.regenerated_in_tree(Path("additional_data/airport_geometry/raw/osm/EDDF.json"))
    assert not fi.regenerated_in_tree(Path("additional_data/weather/raw/metar/EDDF_2026-04.csv"))
    target = tree / "additional_data/pru/pru_monthly.parquet"
    target.write_text("edited", encoding="utf-8")
    with pytest.raises(SystemExit, match="differ"):
        fi.init(tree, released["source"])
    assert fi.init(tree, released["source"], force=True)["copied"] == 2          # and --force replaces the rebuilt table
    with pytest.raises(SystemExit, match="contain"):
        fi.init(released["source"] / "inside", released["source"])


def test_weather_coverage_names_the_missing_station_months(fi, tmp_path):
    raw = tmp_path / "additional_data" / "weather" / "raw" / "taf"
    raw.mkdir(parents=True)
    need = fi.needed_months(fi.FINAL_MONTHS)   # the release's four months, each padded by its look-back and look-ahead
    assert need == ["2025-12", "2026-01", "2026-02", "2026-03", "2026-05", "2026-06", "2026-07", "2026-08"]
    for month in need:
        if month != "2026-06":
            (raw / f"LSZH_{month}.csv").write_text("x", encoding="utf-8")
    cov = fi.weather_coverage(tmp_path)["taf"]
    assert cov["stations"] == ["LSZH"] and cov["missing"] == ["LSZH_2026-06"]
    assert fi.weather_coverage(tmp_path, stations=["LSZH", "EDDF"])["taf"]["missing"][:2] == ["EDDF_2025-12", "EDDF_2026-01"]
    assert fi.needed_months(["2026-03"]) == ["2026-02", "2026-03", "2026-04"]


# --- runbook --------------------------------------------------------------------------------------------------------

def _fake_run(runs_dir, run_id, blend=None, *, mixture=True, refit=True, predict=None, stack=None, preds=None):
    """A run directory with what the runbook and `latest` read: metrics.json (blend / stack), config.resolved.json,
    feature_list.json, the refit files, predict.json and predictions."""
    d = runs_dir / run_id
    d.mkdir(parents=True)
    metrics = {"blend": blend} if blend else ({"stack": stack} if stack else {"overall": {"rmse": 1.0}})
    (d / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    cfg = {"model": {"type": "catboost", "refit_on_all": True, "per_airport": False,
                     "mixture": {"enabled": mixture, "bases": ["sched_delta_full"], "universe_column": "stale_suspect"}},
           "seed": 42}
    (d / "config.resolved.json").write_text(json.dumps(cfg), encoding="utf-8")
    (d / "feature_list.json").write_text(json.dumps({"hashes": {}}), encoding="utf-8")
    (d / "model").mkdir()
    if refit and not blend and not stack:
        (d / "model" / "refit.cbm").write_bytes(b"x")
        if mixture:
            (d / "model" / "refit__classifier.cbm").write_bytes(b"x")
    if predict is not None:
        (d / "predict.json").write_text(json.dumps(predict), encoding="utf-8")
    if preds is not None:
        for name, values in preds.items():
            pd.DataFrame({"y_pred": values}, index=pd.Index(range(len(values)), name="mvt_id")).to_parquet(d / name)
    return run_id


@pytest.fixture
def blend_world(tmp_path, monkeypatch):
    """Trained runs a, b, c; ``inner`` = equal-weight blend of a and b; ``outer`` = 0.2 inner + 0.8 c, clipped."""
    runs = tmp_path / "runs"
    monkeypatch.setattr(paths, "RUNS", runs)
    a, b, c = (_fake_run(runs, f"2026-10-01T00-00-0{i}__m{i}__0000000{i}__s42") for i in range(3))
    inner = _fake_run(runs, "2026-10-01T01-00-00__inner__aaaaaaaa", {"runs": [a, b], "weights": None, "lobt_bound": None})
    outer = _fake_run(runs, "2026-10-01T02-00-00__outer__bbbbbbbb",
                      {"runs": [inner, c], "weights": [0.2, 0.8], "lobt_bound": {"rank": {}}})
    return {"runs": runs, "a": a, "b": b, "c": c, "inner": inner, "outer": outer, "tmp": tmp_path}


def test_runbook_expands_a_blend_of_blends(fi, blend_world):
    w = blend_world
    tree = fi.blend_tree(w["outer"])
    assert [n.run_id for n in tree.children] == [w["inner"], w["c"]] and tree.lobt_bound and not tree.children[0].lobt_bound
    assert dict(fi.leaves(tree)) == pytest.approx({w["a"]: 0.1, w["b"]: 0.1, w["c"]: 0.8})
    plan = fi.plan_runbook(w["tmp"] / "final_tree", from_blend=w["outer"])
    assert plan.members == [w["a"], w["b"], w["c"]] and plan.final == "final_b02" and plan.stack
    assert plan.blends == [("final_b01", ["final_m01", "final_m02"], [1.0, 1.0], False, w["inner"]),
                           ("final_b02", ["final_b01", "final_m03"], [0.2, 0.8], True, w["outer"])]

    text = fi.runbook(w["tmp"] / "final_tree", from_blend=w["outer"])
    lines = [line.strip() for line in text.splitlines()]
    assert '$env:PRC_DATA_DIR = $Tree' in lines and "Invoke-Py -m prc prepare --force" in lines
    assert any(line.startswith("$Version = Get-Py lab/final_inputs.py derive version") for line in lines)
    assert any(line.startswith("$env:PRC_EXPECTED_RANK_ROWS = Get-Py lab/final_inputs.py derive rank-rows") for line in lines)
    warm = [i for i, line in enumerate(lines) if "--features-only" in line]
    named = [i for i, line in enumerate(lines) if line.startswith("$final_m")]
    assert len(warm) == len(named) == 3 and max(warm) < min(named)          # every warm-up before any prediction
    assert all('--tree "${Tree}"' in lines[i] and '"data.version=${Version}"' in lines[i] for i in warm + named)
    assert all('--threads "4"' in lines[i] and "@PredictArgs" in lines[i] for i in warm + named)
    m3 = next(line for line in lines if line.startswith("$final_m03 = Use-Run"))
    assert f'-Check @("--source", "{w["c"]}", "--data-version", "${{Version}}")' in m3 and "--name final_m03" in m3
    b1 = next(line for line in lines if line.startswith("$final_b01 = Use-Run"))
    assert '--weights "1.0" "1.0"' in b1 and "--lobt-bound" not in b1   # equal weights, but explicit
    b2 = next(line for line in lines if line.startswith("$final_b02 = Use-Run"))
    assert '--runs "${final_b01}" "${final_m03}" --weights "0.2" "0.8" --lobt-bound --name final_b02' in b2
    assert '"--blend-of", "${final_b01}", "${final_m03}", "--blend-weights", "0.2", "0.8"' in b2
    stack = next(line for line in lines if line.startswith("$Stack = Use-Run"))
    assert 'lab/final_stack.py "${final_b02}" --inwindow "0.5" --write final_stack' in stack
    assert lines.index("Invoke-Py -m prc submit --run \"${Stack}\"") > lines.index(stack)
    assert any(line.startswith("# Not done here") for line in lines) and "} finally {" in lines


def test_runbook_refuses_what_would_not_run(fi, blend_world):
    w = blend_world
    with pytest.raises(fi.RunbookError, match="no run directory"):
        fi.plan_runbook(w["tmp"] / "t", from_blend="2026-10-01T03-00-00__gone__cccccccc")
    shutil.rmtree(w["runs"] / w["b"] / "model")
    (w["runs"] / w["b"] / "model").mkdir()
    with pytest.raises(fi.RunbookError, match="no refit model"):
        fi.plan_runbook(w["tmp"] / "t", from_blend=w["outer"])
    plan = fi.plan_runbook(w["tmp"] / "t", from_blend=w["outer"], allow_warnings=True)
    assert any("no refit model" in x for x in plan.warnings)
    assert "# WARNING" in fi.render(plan)
    with pytest.raises(fi.RunbookError, match="weights"):
        fi.plan_runbook(w["tmp"] / "t", runs=[w["a"], w["c"]], weights=[1.0])
    with pytest.raises(fi.RunbookError, match="data/"):
        fi.plan_runbook(paths.ROOT / "data" / "final", runs=[w["a"]])


def test_a_stack_needs_mixture_runs(fi, tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    monkeypatch.setattr(paths, "RUNS", runs)
    a = _fake_run(runs, "2026-10-01T00-00-01__plain__00000001__s42", mixture=False)
    b = _fake_run(runs, "2026-10-01T00-00-02__mix__00000002__s42")
    with pytest.raises(fi.RunbookError, match="mixture"):
        fi.plan_runbook(tmp_path / "t", runs=[a, b])
    plan = fi.plan_runbook(tmp_path / "t", runs=[a, b], allow_warnings=True)
    assert not plan.stack and "-Name final_stack" not in fi.render(plan)
    flat = fi.render(fi.plan_runbook(tmp_path / "t", runs=[b], weights=[1.0], stack=False))
    assert '--weights "1.0" --lobt-bound --name final_blend' in flat and 'submit --run "${final_blend}"' in flat


def test_runbook_reads_the_stands_file_from_the_source_trees_manifest(fi, blend_world, tmp_path, monkeypatch):
    """The geometry manifest records ``stands_file`` as {path, sha1}; the runbook passes the path (no dict text)."""
    source = tmp_path / "source_tree"
    manifest = source / "additional_data" / "airport_geometry" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"stands_file": {"path": "C:/stands/extra.csv", "sha1": "abc"}}), encoding="utf-8")
    monkeypatch.setattr(fi, "_source_tree_of", lambda run_id: source)
    text = fi.runbook(tmp_path / "final_tree", from_blend=blend_world["outer"])
    assert 'fetch-geometry --offline --stands-file "C:/stands/extra.csv"' in text and "{'path'" not in text
    assert '$SourceTree = "' + source.as_posix() + '"' in text


def test_the_rehearsal_checks_instead_of_submitting(fi, blend_world, tmp_path, monkeypatch):
    w = blend_world
    source = tmp_path / "source_tree"
    (source / "competition_files").mkdir(parents=True)
    (source / "competition_files" / "ranking.parquet").write_bytes(b"x")
    monkeypatch.setattr(fi, "_source_tree_of", lambda run_id: source)
    record = {"source": w["outer"], "steps": ["lobt_window", "linked_exclusion"], "rates": [0.0, 1.0]}
    tag = hashlib.sha1(f"{w['outer']}|outer_stack".encode()).hexdigest()[:8]     # as lab/final_stack.py names it
    stack = _fake_run(w["runs"], f"2026-10-01T02-00-00__outer_stack__{tag}", stack=record)
    _fake_run(w["runs"], "2026-10-01T02-00-00__winter_base__deadbeef", stack=record)   # a copy of the record, not a stack
    plan = fi.plan_runbook(tmp_path / "rehearse_tree", from_blend=w["outer"], rehearse=True)
    assert plan.months == ["2026-01", "2026-07"] and plan.prefix == "rehearse" and plan.source_stack == stack
    text = fi.render(plan)
    assert "prc submit" not in text and "prc levelcheck" not in text
    check = next(line.strip() for line in text.splitlines() if "final_inputs.py check" in line)
    assert f'--source-blend {w["outer"]}' in check and f"--source-stack {stack}" in check
    assert '--months 2026-01 2026-07' in text and "--prefix rehearse" in text
    assert f'[string]$Ranking = "{(source / "competition_files" / "ranking.parquet").as_posix()}"' in text


@pytest.mark.skipif(shutil.which("powershell") is None and shutil.which("pwsh") is None, reason="no PowerShell")
def test_the_powershell_script_parses(fi, blend_world, tmp_path):
    script = tmp_path / "runbook.ps1"
    script.write_text(fi.runbook(tmp_path / "final_tree", from_blend=blend_world["outer"]), encoding="utf-8")
    shell = shutil.which("powershell") or shutil.which("pwsh")
    probe = ("$e = $null; $t = $null; [System.Management.Automation.Language.Parser]::ParseFile("
             f"'{script}', [ref]$t, [ref]$e) | Out-Null; if ($e.Count) {{ $e | ForEach-Object {{ $_.Message }}; exit 1 }}")
    done = subprocess.run([shell, "-NoProfile", "-Command", probe], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")
def test_the_bash_script_parses(fi, blend_world, tmp_path):
    script = tmp_path / "runbook.sh"
    text = fi.runbook(tmp_path / "final_tree", from_blend=blend_world["outer"], shell="sh")
    script.write_text(text, encoding="utf-8", newline="\n")
    assert text.startswith("#!/usr/bin/env bash") and "set -euo pipefail" in text and 'export PRC_DATA_DIR="$Tree"' in text
    assert 'final_m03="$(use_run final_m03 --source ' in text and "inherit_errexit" in text
    done = subprocess.run(["bash", "-n", script.name], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr


def test_derive_names_the_version_after_the_installed_file(fi, released, tmp_path):
    tree = tmp_path / "tree"
    with pytest.raises(SystemExit, match="validate --install"):
        fi.derive("version", tree)
    (tree / "competition_files").mkdir(parents=True)
    shutil.copyfile(released["ranking"], tree / "competition_files" / "ranking.parquet")
    version = fi.derive("version", tree, "final26")
    assert re.fullmatch(r"final26_[0-9a-f]{8}", version) and version == f"final26_{fi._sha1(released['ranking'])[:8]}"
    assert fi.derive("rank-rows", tree) == str(len(released["frame"]))
    with pytest.raises(SystemExit, match="plain name"):
        fi.derive("version", tree, "final 26")


def test_latest_picks_the_newest_run_that_passes_the_checks(fi, tmp_path, monkeypatch, capsys):
    runs = tmp_path / "runs"
    monkeypatch.setattr(paths, "RUNS", runs)
    for run_id in ("2026-10-10T10-00-00__final_m01__1234abcd__s42", "2026-10-10T11-00-00__final_m01__1234abcd__s42",
                   "2026-10-10T12-00-00__final_m01x__1234abcd__s42", "2026-10-10T12-30-00__final_b01__abcd1234"):
        (runs / run_id).mkdir(parents=True)
    assert fi.latest_run("final_m01") == "2026-10-10T11-00-00__final_m01__1234abcd__s42"
    assert fi.latest_run("final_b01") == "2026-10-10T12-30-00__final_b01__abcd1234"
    assert fi.latest_run("final_m99") is None
    assert fi.main(["latest", "final_m99"]) == 1
    assert fi.main(["latest", "final_m99", "--missing-ok"]) == 0 and capsys.readouterr().out == ""

    old = "2026-10-10T10-00-00__final_m01__1234abcd__s42"
    (runs / old / "predict.json").write_text(json.dumps({"source_run": "SRC", "data_version": "final26_aaaa"}))
    assert fi.latest_run("final_m01", source="SRC", data_version="final26_aaaa") == old   # the newer one is no predict run
    assert fi.latest_run("final_m01", source="SRC", data_version="rehearse_bbbb") is None
    assert fi.latest_run("final_m01", source="OTHER") is None

    _fake_run(runs, "2026-10-10T13-00-00__final_b02__ffffffff", {"runs": ["x", "y"], "weights": [0.25, 0.75]})
    assert fi.latest_run("final_b02", blend_of=["x", "y"], blend_weights=[1, 3]) is not None   # normalised alike
    assert fi.latest_run("final_b02", blend_of=["x", "y"], blend_weights=[1, 1]) is None
    assert fi.latest_run("final_b02", blend_of=["y", "x"]) is None
    _fake_run(runs, "2026-10-10T14-00-00__final_stack__eeeeeeee",
              stack={"source": "B", "steps": ["lobt_window", "linked_exclusion"], "inwindow": 0.5})
    assert fi.latest_run("final_stack", stack_of="B", inwindow=0.5) == "2026-10-10T14-00-00__final_stack__eeeeeeee"
    assert fi.latest_run("final_stack", stack_of="B", inwindow=0.25) is None


def test_check_passes_an_exact_rehearsal_and_fails_a_drifted_one(fi, tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    monkeypatch.setattr(paths, "RUNS", runs)
    good = {"source_run": "S", "reproduction": {"comparable": True, "within_tol": True, "max_abs_diff": 0.0},
            "count_rescale": {"columns": {"stand_freq": {"rows_changed": 0}}},
            "parity": {"wx_now": {"status": "identical"}}}
    m1 = _fake_run(runs, "2026-10-10T10-00-00__final_m01__11111111__s42", predict=good)
    same = {"test_predictions.parquet": [1.0, 2.0], "oof_predictions.parquet": [3.0, 4.0]}
    b = _fake_run(runs, "2026-10-10T10-00-01__final_b01__22222222", preds=same)
    src = _fake_run(runs, "2026-10-01T00-00-00__blend11__33333333", preds=same)
    problems, notes = fi.check_rehearsal([m1], blend=b, source_blend=src)
    assert problems == [] and any("max |diff| 0" in n for n in notes)
    assert fi.main(["check", "--members", m1, "--blend", b, "--source-blend", src]) == 0

    bad = json.loads(json.dumps(good))
    bad["reproduction"] = {"comparable": True, "within_tol": False, "max_abs_diff": 3.5}
    bad["count_rescale"]["columns"]["stand_freq"]["rows_changed"] = 12
    bad["parity"]["wx_now"] = {"status": "changed", "max_changed_share": 0.02}
    m2 = _fake_run(runs, "2026-10-10T10-00-02__final_m02__44444444__s42", predict=bad)
    off = _fake_run(runs, "2026-10-10T10-00-03__final_b02__55555555",
                    preds={"test_predictions.parquet": [1.0, 2.5], "oof_predictions.parquet": [3.0, 4.0]})
    problems, notes = fi.check_rehearsal([m2], blend=off, source_blend=src)
    assert len(problems) == 3 and any("3.5" in p for p in problems) and any("rescale" in p for p in problems)
    assert any("parity wx_now changed" in n for n in notes)
    assert fi.main(["check", "--members", m2]) == 1
