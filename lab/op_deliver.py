"""The operational deliverables on the final ranking set (findings 10.35, step B4): one file per decision moment.

    python lab/op_deliver.py --tree data_final --member h1 --runs RUN [RUN ...] [--tables T.parquet] [--threads 4]
    python lab/op_deliver.py --tree data_final --member h1 --config op_h1_r4_nodoy.yaml --n 3 --tables T.parquet --rule-c

Steps for one member, each re-used when an earlier start already made it:
1. **Predict.** Each refitted source run is applied to the final tree without retraining: `prc predict --run SRC
   -o data.version=<the tree's version> --tree DIR --name final_<config>`. The version is the one the runbook derived,
   read from the tree's `interim/prepare.json` frame and `lab/final_inputs.py derive version`.
2. **Blend.** `prc blend` takes the equal-weight mean of the predict runs, i.e. of the seeds. Never `--lobt-bound`: the
   NM join window is a challenge artefact (tier C).
3. **Calibrate.** With ``--tables``, `lab/op_calibrate.py apply` adds the calibrated median and the P10/P90 band.
   Without it, the file holds the mean only. ``--rule-c`` applies the tables only if rule C passed for this member
   (`C_<member>` in runs/op_decisions.json, `lab/op_calibrate.py check`); a missing decision stops the script.
4. **Write the file.** The outputs are mapped onto the final template's ids and written to
   `submissions/operational/<member>__<blend run>.parquet`. Columns: ``MVT_ID_mvt``, ``y_mean`` and, if calibrated,
   ``y_med``, ``y_p10``, ``y_p90``; integer seconds, clipped to [30, 200000] as `prc submit` does.
5. **Record it.** `submissions/operational/manifest.json` gets one record per member: the source runs, the predict
   runs, the blend, the tables and rule C's decision, the data version, the sha1 of the template and of the file, the
   row count, and the medians.
6. **Level check.** `prc levelcheck --run <blend>` on the final tree compares each airport-month's predicted mean
   with the PRU's published monthly mean taxi-out, the only out-of-time check that needs no target. It writes
   level_check.csv into the blend run.
7. **Rule M** (``--median-config CFG``). The newest refitted run of that MAE-loss config is predicted on the tree,
   and its predictions replace ``y_med``, clamped into the calibrated P10-P90 band. The file is then
   ``<member>__<blend>__mae.parquet``, and the manifest keeps the earlier record as ``<member>_calibrated_median``.

Member ``h1f`` is rule T's concept member (findings 10.36): H1 plus the traffic after the anchor, tier F. It has no
calibration (mean only), and its manifest record names the tier.

Member ``h1f5`` is the full concept F5 (`op_h1_f5_full`), delivered at the owner's request (13 Oct, findings 10.37): H1
plus every input of rule T after the anchor (the plans, the realised records, NM readings and ADS-B picture) and the
ADS-B picture before it; tier F, mean only. Its realised departure side carries traces of the own slot, which rule T
could not credit (findings 10.36), so its accuracy is an upper bound even for perfect plans; the record says so.

It never calls `prc submit` and never writes `submissions/submitting.parquet`. That file is the scored challenge file,
and the script checks at the end that its sha1 has not changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_TIER_F = ("F: the other flights' realised movements after the anchor stand in for AMAN/DMAN/A-SMGCS plans; an "
           "upper bound for perfect plans, not available as computed (findings 10.36)")
TIER = {
    "h1f": _TIER_F,
    "h1f5": _TIER_F + ". The full concept F5 (findings 10.37): it also reads the realised departure side, whose gain "
                      "carries traces of the own slot and cannot be credited to planning data (rule T, findings 10.36)",
}
"""The manifest's tier note per concept member; the strict deliverables (h1, hl3, hl24) have none."""


def refits_any_hash(name: str, n: int) -> list[str]:
    """The newest ``n`` refitted runs (refit.json and test_predictions.parquet) named ``name``, whatever their hash;
    never a predict run. For the concept members, refitted through an override (``--any-hash``)."""
    from prc import paths

    hits = sorted(d.name for d in paths.RUNS.glob(f"*__{name}__*__s*")
                  if d.name.split("__")[1] == name and (d / "refit.json").exists()
                  and (d / "test_predictions.parquet").exists() and not (d / "predict.json").exists())
    if len(hits) < n:
        raise SystemExit(f"only {len(hits)} refitted runs of {name}, {n} wanted")
    return hits[-n:]


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: list[str], env: dict, log: Path) -> None:
    with open(log, "a", encoding="utf-8") as out:
        out.write(f"\n$ {' '.join(cmd)}\n")
        out.flush()
        rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
    if rc:
        raise SystemExit(f"failed ({rc}): {' '.join(cmd)}; see {log}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tree", required=True, help="the final data tree (data_final)")
    ap.add_argument("--member", required=True, choices=("h1", "hl3", "hl24", "h1f", "h1f5"),
                    help="h1f: rule T's concept member (findings 10.36; tier F, mean only); h1f5: the full concept F5 "
                         "(findings 10.37; tier F, mean only)")
    ap.add_argument("--runs", nargs="+", default=None, help="the refitted source runs (one per seed)")
    ap.add_argument("--config", default=None, help="instead of --runs: the newest --n refitted runs of this config")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--any-hash", action="store_true",
                    help="with --config: refits of any hash. A concept member's config stays holdout-only (no config "
                         "with refit_on_all may resolve a tier F block), so its refits, trained with "
                         "-o model.refit_on_all=true, carry another hash than the config file")
    ap.add_argument("--rule-c", action="store_true", help="apply --tables only if rule C passed for this member")
    ap.add_argument("--tables", default=None, help="calibration tables (lab/op_calibrate.py fit); only if rule C passed")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out-dir", default=None, help="default submissions/operational (another folder for a dry run)")
    ap.add_argument("--median-config", default=None,
                    help="rule M: a refitted MAE-loss config whose predictions replace y_med (clamped into P10-P90)")
    a = ap.parse_args(argv)

    if bool(a.runs) == bool(a.config):
        raise SystemExit("give either --runs or --config")
    tree = Path(a.tree).resolve()
    env = dict(os.environ, PRC_DATA_DIR=str(tree), PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    env.pop("PRC_EXPECTED_RANK_ROWS", None)
    os.environ["PRC_DATA_DIR"] = str(tree)
    sys.path.insert(0, str(ROOT / "lab"))
    import pandas as pd

    from final_inputs import latest_run  # noqa: E402
    from op_rules import refits  # noqa: E402
    from prc import data, paths  # noqa: E402

    if a.config:
        a.runs = (refits_any_hash if a.any_hash else refits)(Path(a.config).stem, a.n)
    decision = None
    if a.rule_c:
        doc = json.loads((paths.RUNS / "op_decisions.json").read_text(encoding="utf-8"))
        decision = doc.get(f"C_{a.member}")
        if decision is None:
            raise SystemExit(f"--rule-c: no decision C_{a.member} in runs/op_decisions.json")
        if not decision.get("passed"):
            print(f"rule C did not pass for {a.member}: the file holds the mean only")
            a.tables = None

    py = sys.executable
    scored = paths.SUBMISSIONS / "submitting.parquet"
    scored_sha = _sha1(scored) if scored.exists() else None
    out_dir = Path(a.out_dir) if a.out_dir else paths.SUBMISSIONS / "operational"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / f"{a.member}.log"
    version = subprocess.check_output([py, "lab/final_inputs.py", "derive", "version", "--tree", str(tree)],
                                      cwd=ROOT, env=env, text=True).strip().splitlines()[-1]

    # 1. predict every source run on the final tree
    preds = []
    for src in a.runs:
        cfg_name = src.split("__")[1]
        name = f"final_{cfg_name}"
        run = latest_run(name, source=src, data_version=version)
        if run is None:
            _run([py, "-m", "prc", "predict", "--run", src, "-o", f"data.version={version}", "--tree", str(tree),
                  "--threads", str(a.threads), "--name", name], env, log)
            run = latest_run(name, source=src, data_version=version)
        if run is None:
            raise SystemExit(f"no predict run for {src}")
        preds.append(run)
        print(f"predict {src} -> {run}")

    # 2. the equal-weight blend of the seeds (a single run is used as it is)
    if len(preds) > 1:
        bname = f"final_op_{a.member}_blend"
        weights = [1.0] * len(preds)
        blend = latest_run(bname, blend_of=preds, blend_weights=weights)
        if blend is None:
            _run([py, "-m", "prc", "blend", "--runs", *preds, "--weights", *map(str, weights), "--name", bname], env, log)
            blend = latest_run(bname, blend_of=preds, blend_weights=weights)
        if blend is None:
            raise SystemExit("the blend run was not found after `prc blend`")
    else:
        blend = preds[0]
    print(f"blend -> {blend}")

    # 3. calibrated median and band, or the mean only
    pred = pd.read_parquet(paths.run_dir(blend) / "test_predictions.parquet")["y_pred"]
    if a.tables:
        cal = out_dir / f"{a.member}__{blend}__calibrated.parquet"
        _run([py, "lab/op_calibrate.py", "apply", "--tables", a.tables, "--run", blend, "--out", str(cal)], env, log)
        frame = pd.read_parquet(cal).drop(columns=["apt"])
    else:
        frame = pd.DataFrame({"y_mean": pred})

    # 3b. rule M: the median member's predictions replace the calibrated median (the band stays the calibration's)
    median_rec = None
    if a.median_config:
        if "y_p10" not in frame.columns:
            raise SystemExit("--median-config needs the calibrated band: give --tables (and rule C passed)")
        m_src = refits(Path(a.median_config).stem, 1)[0]
        m_name = f"final_{m_src.split('__')[1]}"
        m_run = latest_run(m_name, source=m_src, data_version=version)
        if m_run is None:
            _run([py, "-m", "prc", "predict", "--run", m_src, "-o", f"data.version={version}", "--tree", str(tree),
                  "--threads", str(a.threads), "--name", m_name], env, log)
            m_run = latest_run(m_name, source=m_src, data_version=version)
        if m_run is None:
            raise SystemExit(f"no predict run for the median member {m_src}")
        med = pd.read_parquet(paths.run_dir(m_run) / "test_predictions.parquet")["y_pred"].reindex(frame.index)
        if med.isna().any():
            raise SystemExit(f"{int(med.isna().sum())} rows have no prediction of the median member")
        clipped = med.clip(lower=frame["y_p10"], upper=frame["y_p90"])
        median_rec = {"source_run": m_src, "predict_run": m_run, "config": a.median_config,
                      "clamped_into_band_share": round(float((clipped != med).mean()), 4)}
        frame["y_med"] = clipped.clip(lower=30.0)
        print(f"median member {m_src} -> {m_run}; {100 * median_rec['clamped_into_band_share']:.2f} % clamped into P10-P90")

    # 4. onto the template's ids
    template = data.load_submission_template()
    ids = template["MVT_ID_mvt"].astype("int64")
    out = pd.DataFrame({"MVT_ID_mvt": template["MVT_ID_mvt"].to_numpy()})
    for c in frame.columns:
        mapped = ids.map(frame[c])
        if mapped.isna().any():
            raise SystemExit(f"{int(mapped.isna().sum())} template rows have no {c}")
        out[c] = mapped.clip(30, 200000).round().astype("int32").to_numpy()
    dst = out_dir / f"{a.member}__{blend}{'__mae' if median_rec else ''}.parquet"
    out.to_parquet(dst, index=False)
    back = pd.read_parquet(dst)
    if len(back) != len(template) or back["MVT_ID_mvt"].tolist() != template["MVT_ID_mvt"].tolist():
        raise SystemExit("the written file does not match the template's ids")

    # 5. the manifest
    man_path = out_dir / "manifest.json"
    manifest = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {}
    tables_rec = None
    if a.tables:
        meta = Path(a.tables).with_suffix(".json")
        tables_rec = {"file": str(a.tables), "sha1": _sha1(Path(a.tables)),
                      "fit": json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else None}
    if decision is not None:
        tables_rec = {**(tables_rec or {}), "rule_c": {k: decision.get(k) for k in
                      ("passed", "mae_gain", "within_120_gain", "worst_airport_delta", "coverage_p10_p90", "runs")}}
    if median_rec and a.member in manifest and not manifest[a.member].get("median_member"):
        manifest[f"{a.member}_calibrated_median"] = manifest[a.member]  # the file before rule M, kept on record
    manifest[a.member] = {
        "file": str(dst.relative_to(ROOT)) if dst.is_relative_to(ROOT) else str(dst), "sha1": _sha1(dst), "rows": int(len(out)),
        "columns": list(out.columns), "medians_s": {c: float(out[c].median()) for c in out.columns if c != "MVT_ID_mvt"},
        "source_runs": list(a.runs), "predict_runs": preds, "blend": blend, "data_version": version,
        "tree": str(tree.relative_to(ROOT)) if tree.is_relative_to(ROOT) else str(tree),
        "template_sha1": _sha1(paths.SUBMITTING_FILE), "calibration": tables_rec,
        "median_member": median_rec,
        "tier": TIER.get(a.member),
    }
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {dst} ({len(out):,} rows; medians {manifest[a.member]['medians_s']})")

    _run([py, "-m", "prc", "levelcheck", "--run", blend], env, log)
    print(f"level check: {paths.run_dir(blend) / 'level_check.csv'}")

    if scored_sha is not None and _sha1(scored) != scored_sha:
        raise SystemExit("submissions/submitting.parquet changed during this run: restore the scored file")
    print(f"the scored file is unchanged (sha1 {scored_sha[:12] if scored_sha else 'absent'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
