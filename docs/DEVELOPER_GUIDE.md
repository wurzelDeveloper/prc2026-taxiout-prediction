# Developer guide

This guide is for developers who continue this work. It assumes Python and machine learning, but no knowledge of air
traffic data or of this repository. It covers the architecture and the guarantees behind it, how to work in the code
and extend it, and where to continue. For depth it links [README.md](../README.md) (the executive summary and the
documentation map), [docs/QUICKSTART.md](QUICKSTART.md) (install, data, reproduction paths), Appendix A below (the
harness: commands, configuration keys, tests, notes) and [docs/findings.md](findings.md) (every measurement and
decision with run ids, cited as "findings 10.35"). Each subsystem is described in its module and field docstrings,
and `python -m prc <command> --help` lists every flag. The reports are [docs/report.md](report.md) (the main report)
and [docs/operational_model.md](operational_model.md) with [docs/operational_features.md](operational_features.md)
(the operational model card and its inputs).

The state described is that of 13 October 2026, with the documents as revised together with this guide. Figures
marked "local" come from `runs/` or `submissions/` on the development machine, which this repository does not hold.

## 1. What the repository is

**The task.** The PRC Data Challenge 2026 asks for the departure taxi-out time: the seconds from off-block (the
aircraft leaves its stand) to takeoff, at ten European airports: EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH
and LTFM. The training data are every movement of 2025 with its answer (4,167,797 departures and arrivals). The board
scores RMSE on January and July 2026 (344,841 departures); the final submission covers January, February, June and
July 2026 (670,790 departures).

**One harness, two model families.** This is an experiment harness, not one model: the package `prc/` (run as
`python -m prc <command>`), the configurations in `configs/`, the lab scripts in `lab/` and the tests in `tests/`.
Every change is a tracked run, and its effect is read against seed noise. The scored model is the post-operations
analysis model (the challenge model in the code, the configurations and the supporting documents; in this guide's code
paths, "challenge" names its pipeline). It estimates a flight's taxi-out after the flight, from what was recorded about
it, the use that the challenge's rationale names (`docs/report.md` Part 1). The operational models are the planning
models.

| | post-operations model (code: challenge) | operational models (planning) |
|---|---|---|
| question | reconstruct a past flight's recorded taxi-out, after the flight | predict the taxi-out before the aircraft moves |
| reads | all challenge data, including the actual takeoff, the NM join window and the flight's own ADS-B track | what exists at the moment: 24 h and 3 h before the schedule; H1, 15 min before the last filed off-block; H0, at the off-block (2025 only) |
| model | 50 trained CatBoost runs, a blend, an ADS-B second stage and post-processing (`lab/`) | one global CatBoost per moment, averaged over seeds; at H1 also a median (a second model with the MAE loss, rule M) and a calibrated 80 % band |
| result | board RMSE 220.34 (v70, the best upload) | OP MAE on the Jan+Jul 2025 holdout: H1 184.3 s (mean) and 178.8 s (median member; the calibrated median it replaced: 180.8 s); 3 h 214.5 s; 24 h 218.2 s; the stand × runway table 207.4 s |
| status | finished; its final-set file is the scored one | the deliverable since 11 Oct; at H1 `op_h1_f1_plan` (F1) since 13 Oct (findings 10.35, 10.37) |

**OP** is the population the operational models are read on: NM-matched departures with a target below one hour. The
Jan+Jul 2025 holdout has 338,277 such rows.

**Why the turn to the operational models.** The post-operations model's big board steps came from things that do not
exist before the aircraft moves: the stale records (422.4 to 329.5, v2 to v3), the NM join window (303.0 to 263.2, v29
to v31) and the flight's own ADS-B track (260.7 to 228.2, v32 to v35). After v57 (220.99), each step was worth tenths
of a second (findings section 10). It is a post-operations model, not a planning model.

**Terms.** The challenge's [data page](https://prc-data-challenge-2026.netlify.app/data.html) defines every column. It lists 11 airports; the data hold 10.

| term | meaning |
|---|---|
| movement | a departure (`PHASE_mvt = DEP`) or an arrival (`ARR`), with id `MVT_ID_mvt` |
| `MVT_TIME_UTC_mvt` / `BLOCK_TIME_UTC_mvt` | the takeoff or landing / the off-block (AOBT) or in-block |
| target | `TAXITIME_SEC_mvt`; for a departure, exactly `MVT_TIME - BLOCK_TIME` |
| NM; SOBT, IOBT, LOBT | EUROCONTROL's Network Manager, whose flight record fills the `_flt` columns. SOBT is the airport record's schedule, `SCHED_TIME_UTC_mvt`; `IOBT_flt` and `LOBT_flt` are NM's initial and last known off-block times |
| `AOBT_3_flt` | NM's actual off-block; at LTFM, and partly elsewhere, back-computed from the takeoff (findings 10.25) |
| A-CDM | the airport process that predicts taxi times before the push (the H1 use case) |
| stale / wrong-date record | a block time that equals the schedule (most multi-hour targets, nearly all at Rome) / one that falls on the previous day (15 a year) (findings 2.2, 2.4) |
| NM join window | on every NM-matched departure, LOBT minus the block time lies within ±3,606 s, so the target is bounded (findings 10.3) |

## 2. Architecture

### 2.1 Data flow

```text
data/competition_files/   12 training months, ranking.parquet, submitting.parquet (the template)
  |  prc prepare                         prc/data.py:prepare
  v
data/interim/movements.parquet          every movement, training and ranking, departures and arrivals; target and
  |                                     off-block blanked on every departure (target.parquet keeps them apart)
  |  feature blocks                      prc/features/*.py, @feature; external inputs from data/additional_data/
  |  build.build_matrix                  (prc fetch-*) and cache/adsb_surface/<snapshot>/ (prc adsb-surface);
  |                                      cached per block in cache/features/<data.version>/
  v
feature matrix                          one row per departure, training and ranking
  |  validation.make_splits              by month or day, no RNG; cleaning.apply_to_splits on the fit side only
  v
per fold                                fold-dependent blocks rebuilt from the fold's fit rows; CategoryUniverse;
  |                                     make_model, fit, predict, metrics; optional refit on every training row
  v                                     (model.refit_on_all) for the ranking predictions
runs/<id>/                              config, metrics, holdout (OOF) and ranking predictions, models (tracking.Run);
  |                                     runs/index.csv, runs/feature_effects.csv
  +--> challenge:    prc blend --lobt-bound -> lab/final_stack.py -> lab/adsb_apply.py (ADS-B stage) -> lab/base_swap.py
  |                  -> lab/forced_long_shift.py -> lab/ltfm_partner_candidate.py -> prc submit -> submissions/
  +--> operational:  configs/op_*.yaml through the same harness -> lab/op_rules.py, op_calibrate.py, op_rule_t.py,
                     op_adopt.py (pre-registered rules) -> lab/op_deliver.py -> submissions/operational/
```

- **Blocks see the whole frame:** departures and arrivals, training and ranking. Most of the signal is cross-row
  context: queues, ground occupancy and stand turnaround. The block, not the column, is the unit of selection,
  caching, importance and ablation.
- **Lab steps write runs in the harness's format,** so `prc blend`, `prc submit` and the next step all accept them.
  The challenge steps are listed in Appendix A ("The second stage and post-processing") and, as commands, in
  QUICKSTART path C; [lab/README.md](../lab/README.md) lists every lab script. The lineage of v70 is in
  [docs/submission_manifest.md](submission_manifest.md).

**The predict path** handles a new ranking set without retraining (findings 10.32, 10.33):

```text
released ranking file  -> lab/final_inputs.py init --tree data_final --source data   (a new tree; data/ is untouched)
                       -> lab/final_inputs.py validate ... --tree data_final --install
environment: PRC_DATA_DIR=data_final, PRC_EXPECTED_RANK_ROWS=<rows>
                       -> prc prepare --force; prc fetch-weather --offline; prc fetch-geometry --offline
                       -> prc predict --run <refit run> -o data.version=final26_<sha8> --tree data_final
                            blocks rebuilt (fold-dependent ones on the refit's rows), counts rescaled, parity and
                            unseen-level reports, refit model loaded; refuses drift and foreign trees (exit code 2)
                       -> prc blend, lab steps -> prc submit (670,790 rows)
```

`lab/final_inputs.py runbook` writes this path as a script. `lab/final_run.sh` runs the whole final chain (QUICKSTART
path D).

### 2.2 Package map

| module | purpose | entry points |
|---|---|---|
| `prc/cli.py`, `paths.py` | the 18 subcommands; every filesystem location | `main`; `DATA`, `feature_cache_dir` |
| `prc/data.py` | turns the raw files into the masked frame | `prepare`, `build_movements`, `_assert_masked` |
| `prc/config.py` | loads YAML into frozen dataclasses; handles `extends`, `-o` and the hashes | `load_config`, `ExperimentConfig.hash`, `split_hash` |
| `prc/features/registry.py`, `build.py` | registers, selects, runs, validates, hashes and caches the blocks | `feature`, `resolve`; `build_matrix`, `block_hash` |
| `prc/features/*.py`, `_*.py` | 76 blocks in 38 modules (47 groups, 929 columns); shared helpers, hashed into the modules that import them. `_catalogue` and `_long_tiers` describe the operational inputs; no block imports them | `_oplib.anchor_h1`, `_horizon.anchor_long`, `_long_tiers.tier_at` |
| `prc/validation.py`, `cleaning.py`, `train.py` | splits; fit-side cleaning; one experiment and its refit | `make_splits`; `apply_to_splits`; `run_experiment`, `_refit_on_all` |
| `prc/models/` | CatBoost, LightGBM, HistGB, per airport, regime mixture, baselines, persistence | `make_model`, `load_model`, `feature_contract` |
| `prc/metrics.py`, `importance.py`, `ablate.py`, `queue.py` | metrics and masks, block permutation, ablation, config queues | `importance.measure`, `run_ablation` |
| `prc/tracking.py`, `leaderboard.py`, `progress.py`, `audit.py` | run records and read-outs | `Run`, `rebuild_index` |
| `prc/blend.py`, `postprocess.py`, `submit.py` | averages, the join window, submission files | `blend_runs`, `apply_lobt_bound`, `build_submission` |
| `prc/predict.py`, `frame_shift.py`, `levelcheck.py` | a refit model on another data tree; monthly means against the PRU's | `predict_run`, `rescale_counts`, `parity_report` |
| `prc/weather/`, `geometry/`, `pru/`, `reanalysis/`, `adsb/` | external data, never imported by feature code; ADS-B points, links, surface snapshot, stand archive | `fetch`, `normalise`, `check`; `surface.build` |
| `lab/` | the challenge stage and post-processing, the final-set path, the operational rules and delivery (`lab/README.md` lists every script) | `final_stack.py`, `adsb_apply.py`, `final_inputs.py`, `op_rules.py`, `op_deliver.py` |

### 2.3 Data layout

None of these paths is in git, and neither is `configs/_generated/`.

| path | holds | follows `PRC_DATA_DIR` |
|---|---|---|
| `data/` | `competition_files/`; `interim/` (`movements`, `target`, `prepare.json`); `additional_data/` (weather, geometry, ERA5, PRU, ADS-B points) | yes |
| `data_final/`, `data_rehearse/`, `data_datefix/` | other data trees: the final set, the rehearsal, the wrong-date experiment | they are trees |
| `data_release/2026-10-09/` | the final release as delivered; installed only into `data_final/` | no |
| `cache/features/<data.version>/` | one parquet per block and hash; `fold_<split>__<split_hash>/`; `TREE.json` | no |
| `cache/adsb/`, `cache/lab/`, `cache/adsb_surface/`, `cache/adsb_archive/` | the lab stage tables; the frozen surface snapshot (s1); the stand archive | no |
| `runs/`, `submissions/` | runs, `index.csv`, `op_decisions.json` (the rule decisions), queues; files, `uploads.csv` (board scores, written by the upload tool, which is not part of this repository), `operational/` | no |

The cache follows the checkout and is keyed by `data.version` alone, so every data tree needs a version of its own.
`build.check_tree_stamp` binds a version to one tree and one frame. Among the versions on the development machine are
`v1` (`data/`), `final26_7a06db85` and `rehearse_211b61a7`. Its `runs/` and `submissions/` are not published: the
rule decisions recorded in `runs/op_decisions.json` are transcribed in findings 10.35-10.37 and
[docs/operational_model.md](operational_model.md), and the delivered operational files' sha1s are in
[docs/operational_manifest.json](operational_manifest.json).

### 2.4 Leakage guarantees

**Target masking.** `prc/data.py:build_movements` copies the training targets to `target.parquet`, then blanks
`TAXITIME_SEC_mvt` and `BLOCK_TIME_UTC_mvt` on every departure, training and ranking alike. `_assert_masked` fails the
build if any departure keeps them. So feature code cannot see an answer, and training rows and 2026 rows share both
code and information. Arrivals keep their times, because 2026 has them too. Taxi-out is exactly takeoff minus block
time, so any departure block time is a perfect leak. The baselines score 693.6 (global median), 636.1 (stand × runway
median) and 672.8 (NM's off-block proxy) RMSE on the holdout; a run far below them is a leak, not a breakthrough.

**Fold-dependent blocks** declare `uses_target=True` and receive `ctx.target` and `ctx.fit_mask`. Each is rebuilt per
fold from the fold's fit rows and cached under `fold_<split>__<split_hash>/`; the refit uses `fold_refit__...`. There
are twelve, for example `stand_runway_target_stats` and `op_h1_nm`.

**Splits** (`validation.make_splits`) are deterministic and go by month or day. The default, `seasonal_holdout`,
validates on January and July 2025.

**Quarantine.** The lab rungs that read the airport's recorded off-blocks are every H0 rung, which anchors on the
flight's own off-block, and H1 r5. Their nine blocks, the eight `op_h0_*` blocks (`op_h0_hind_lab` among them) and
`op_h1_feed_lab`, declare `paths.TARGET_FILE` as external, and tests keep them out of every refit configuration.

**Tiers.** Every input has a tier, recorded in `docs/feature_table.md` for the post-operations model, and in
`docs/operational_features.md` and `_long_tiers.py` for the operational models. The operational deliverables read O
and O* inputs, plus the C-tagged NM-join level of the flight-plan columns, which is constant on OP. Of the challenge
submission's 58 inputs, 44 do not exist before the movement.

| tier | the input |
|---|---|
| O / O* | is known before the off-block / is known only under an assumption the data cannot check (e.g. the planned runway) |
| T / A | is anchored on the actual takeoff / is produced at or after an off-block (e.g. NM's actual off-block) |
| M / P | observes the flight's own movement (ADS-B; challenge only) / is published later (PRU daily data, ERA5) |
| C | exploits how the challenge data were built (the join window, stale and wrong-date records) |
| F | uses realised movements as a stand-in for missing AMAN, DMAN or A-SMGCS plans: an upper bound, never in a configuration file that sets `refit_on_all` |

**Gates:**

| gate | holds |
|---|---|
| `tests/test_features.py` (needs `prepare`) | every block on 120,000 training rows: declared columns; no feature with abs(r) > 0.99 against the target; `uses_target` matches the function's source; target statistics follow the fit mask |
| `tests/test_operational.py`, `test_operational_long.py`, `test_operational_future.py` | T1: moving the own takeoff, NM actuals or off-block leaves every input bit-identical; T2: censoring later events changes nothing; the refit list is pinned; `TIER_F_REFIT` is empty |
| `tests/test_adsb_surface.py`, `test_operational_adsb.py`, `lab/op_adsb_audit.py` | causal surface events; own-airframe contamination below 0.2 % at H1 (0.054 % on s1) |
| `tests/test_feature_catalogue.py` | a catalogue entry and tier for every deliverable and sensitivity column |

## 3. How to work

### 3.1 Setup

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m prc --help        # 18 subcommands
```

- **Versions.** The runs' `env.json` records Python 3.13.13, CatBoost 1.2.10, pandas 3.0.5, numpy 2.5.3,
  scikit-learn 1.9.1 and pyarrow 25.0.1; scipy 1.18.1 is the development venv's version (not recorded).
  `requirements.txt` pins nothing. QUICKSTART section 2 has the pinned line, and section 3 the data.
- **Platform.** Development ran on Windows 11 with PowerShell, and Git Bash for `lab/*.sh`. The chains call
  `.venv/Scripts/python.exe`, and `lab/final_worktree.py` makes Windows junctions; adjust both on Linux or macOS.
- **GPU.** Most configurations train on the GPU. Without one, pass `-o model.params.task_type=CPU`, plus
  `model.mixture.classifier_params.task_type=CPU` for a mixture; this changes the configuration hash. GPU fits are not
  bit-reproducible.

### 3.2 Commands

| command | what it does |
|---|---|
| `python -m prc prepare [--force]` | builds `data/interim/` and asserts the row counts |
| `python -m prc fetch-weather`, `fetch-geometry`, `fetch-era5`, `fetch-pru` | downloads and normalises one source; `--offline` re-parses, `--check` reports coverage |
| `python -m prc fetch-adsb [--days ...] [--workers N] [--kind ...]` | streams adsb.lol days into `data/additional_data/adsb/points/` (`--kind` prod, the default, or staging, into `points_staging/`) and skips days that exist; no `--offline` or `--check` |
| `python -m prc adsb-surface` | builds, checks, supports and freezes an ADS-B surface snapshot (section 4.7) |
| `python -m prc features -c <cfg> --list` | lists the resolved blocks, their hashes and their cache state |
| `python -m prc train -c <cfg> [-o k=v] [--seeds N] [--no-test] [--keep-test]` | runs one experiment; `-o data.sample_frac=0.2 --no-test` for sweeps |
| `python -m prc ablate -c <cfg> --mode add-to-base --blocks a,b --emit-configs` | writes the variants and prints the `prc queue` line that trains them |
| `python -m prc leaderboard`, `progress`, `audit --run <id>` | compares runs; shows a fit's ETA; shows where a run's error lives |
| `python -m prc blend`, `submit`, `predict` | averages runs; writes the submission file and overwrites `submissions/submitting.parquet` (section 3.5); applies a refit to another data tree |

`--run` takes the bare directory name under `runs/`: train runs end in `__s<seed>`, blends and lab runs do not, and
`prc predict` also accepts a path. `-o` values are parsed as YAML, and unknown keys raise.
`python -m prc <command> --help` lists every flag. Uploads to the challenge's scorer went through a separate tool,
which is not part of this repository.

### 3.3 Tests

Run `python -m pytest -q` from the repository root: 46 test files with 598 test functions (985 tests with their
parameters). In a clean clone without the data, a run on 14 October gave 956 passed, 26 skipped and 3 xfailed in
15 minutes. Under heavy load on Windows, `test_run_index.py::test_concurrent_appends_keep_every_row` (four threads
appending to one index) can fail with a `PermissionError`: `prc/tracking.py`'s lock catches only `FileExistsError`,
and Windows reports a lock file that its holder is deleting as access denied. It passes when rerun on its own.

Most tests need neither data nor network. `tests/synthetic_tree.py` writes a tiny competition tree, and the predict
path runs three CPU fits on it; `tests/adsb_synth.py` builds ADS-B worlds. `test_features.py` needs `data/interim/`,
`test_adsb_stage_tables.py` needs `cache/adsb/`, and a few single tests elsewhere skip without data. The real-data
slices run only with `OPLAD_FULL=1`, `OPLAD_REAL=1` or `HORIZON_REAL=1`; the lab chains count a skipped slice as a
failure.

| you changed | run at least |
|---|---|
| a block, or `build.py` | its synthetic test, `test_feature_cache.py`, `test_features.py` |
| an operational block or configuration | `test_operational.py`, `test_operational_long.py`, `test_operational_future.py`, `test_feature_catalogue.py` |
| the models, the training or the config schema | `test_pipeline.py`, `test_mixture.py`, `test_model_persist.py` |
| the new-data path | `test_predict.py`, `test_frame_shift.py`, `test_final_inputs.py` |

Never run tests, or a second fit, beside a CatBoost fit. CatBoost takes every core, so everything slows about
threefold and looks hung. `python -m prc progress` shows whether the fit still moves.

### 3.4 Measuring a change

- **Seed floors.** Read every effect against the seed floor of your line (`--seeds 3`); an effect inside it is not a
  finding. Recorded floors: `rmse_bulk` ±0.15 s on exp030 but about ±1 s on the mixture line (findings 9.10);
  `rmse_ex_dayoffset` about ±2 s. For OP MAE, 0.25 s between two GPU seeds of `op_h1_r4` is only a lower bound; the
  three refitted seeds of `op_h1_r4_nodoy` span 0.85 s (186.72-187.57 s, findings 10.35). Read single-seed OP MAE
  steps below about 1 s as unresolved, or run three seeds.
- **Populations.** Beside `rmse`, read the one your change targets: `rmse_ex_dayoffset` (without the wrong-date rows,
  which carry over a third of the squared error), `rmse_bulk` (targets below 3,600 s), `rmse_disrupted`, or OP MAE
  (`lab/op_rules.py metrics`). Judge winter features on `exp029_winter_check.yaml`, the Feb+Dec holdout.
- **Generalisation.** The holdout is mildly optimistic, because the GBDTs early-stop on it; the forward split (fit on
  Jan-Oct, validate on Nov+Dec) and the five 2025 month pairs check it.
- **Pre-registration.** Since 11 Oct each decision rule has been written into findings.md before its runs. The rule
  scripts record each decision in `runs/op_decisions.json`; the recorded decisions are transcribed in findings
  10.35-10.37 and `docs/operational_model.md`.

### 3.5 Conventions

- **Paths** come from `prc/paths.py`. `PRC_DATA_DIR` moves only the data tree; `cache/`, `runs/` and `submissions/`
  stay in place.
- **Cache keys.** `build.block_hash` covers the name, `version=`, outputs, params, `data.version`, the parents' hashes
  and the sha1 of any external file. Its source part covers the function, its module and the `prc.features` modules
  that module references directly, one level deep: the defining modules of the functions, classes and module objects
  the block's module holds as globals. A from-imported constant and a module reached only through another helper are
  not covered, so import such modules as module objects (as `operational_long.py` does). Editing a module re-keys
  every block of every module that references it in that way. The delivered refits (F1 and its median member, 3 h,
  24 h, h1f, h1f5) are keyed on these modules: `operational`, `operational_future`, `operational_long`,
  `operational_adsb` (h1f5), `aircraft`, `core`, `geometry`, `stand_runway`, `congestion`, `temporal`, `turnaround`,
  `weather`, `weather_fcst`, `_adsblib`, `_arrlib`, `_geolib`, `_horizon`, `_oplib`, `_proxylib` and `_wxlib`. An edit
  to any of them makes `prc predict` refuse those refits as drifted. For that reason the docstring of `operational.py`
  still names `report/feature_table.md`, the folder's name until 14 October; the file is `docs/feature_table.md`.
  Likewise the docstring of `partner_runway.py` (keyed into the Istanbul specialists exp071 and exp076) names
  `lab/pretests/dependent_runways/`, exploration scripts that are not part of this repository (findings 10.28 holds
  their results). Logic outside `prc/features/` needs a `version=` bump; a change to `prc/data.py` needs a new
  `data.version`.
- **Inputs of trained runs.** Never rebuild an input table that trained runs read, not even with `--offline`:
  `prc predict` would refuse those runs as drifted.
- **The scored file.** `prc submit` always overwrites `submissions/submitting.parquet`, the copy of the scored file
  that `lab/final_run.sh` compares with the final run's file at its end and that `lab/op_deliver.py` checks for
  changes. After submitting any other run, restore it: run
  `python -m prc submit --run 2026-10-11T03-22-21__final_v70_5c8bf3__5c8bf33e` with `PRC_DATA_DIR` set to `data_final`
  (section 4.6 gives the form for each shell), and check that the file's sha1 is
  `6cdf67a406177e819e083847fab7c48febc41854`. `lab/final_run.sh` submits it last. Operational files go through
  `lab/op_deliver.py`, which never submits.
- **Configurations.** `extends:` merges dictionaries but replaces lists, so restate `features.include`, `exclude` and
  `drop_columns`. A new key must drop out of `ExperimentConfig.hash()` while it holds its default, as `block_dropout`
  does. `devices` goes in YAML only, as the string `"0"`.
- **Splits** go by month or day, never row-level random KFold. Congestion features make neighbouring flights
  near-duplicates.
- **Run records.** Within `prc`, `tracking.Run` is the only writer to `runs/`. Seven lab scripts that make runs
  (`final_stack.py`, `adsb_apply.py`, `base_swap.py` and four others) write them directly, and their runs are absent
  from `index.csv`. Locally on 13 Oct, 166 run directories were not in it: 157 with a "stack" record in `metrics.json`
  (their lineage) and 9 made by an exploration script that is not part of this repository.
  `prc leaderboard --rebuild` would scan them in, with empty names and metrics.
- **Line endings are LF.** From Python, write files with `newline="\n"`. Never edit a `.sh` while it runs.
- **Habits.** One GPU job at a time (`lab/op_queue.sh` reads `runs/op_queue.txt`). Notebooks import `prc`, never the
  reverse. findings.md is updated whenever a finding changes.

## 4. Extension points

### 4.1 A feature block

The shape of a block, from `prc/features/congestion.py`. A new module imports `feature` and `FeatureContext` from
`prc.features.registry`, `epoch_seconds` from `prc.features.core` and `window_count` from `prc.features.congestion`.

```python
@feature(
    group="congestion",
    outputs=("pier_dep_back_15m", "pier_dep_back_60m"),
    depends_on=("core_keys", "stand_runway_keys"),
    doc="Departures pushing back from the same pier/terminal around this takeoff.",
)
def pier_load(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """Local apron congestion, which airport-wide counts wash out."""
    keys = ctx.dep("stand_runway_keys")
    q_t = epoch_seconds(mvt.loc[ctx.dep_index, "MVT_TIME_UTC_mvt"])
    q_g = keys["stand_prefix"].astype("string")
    return pd.DataFrame(
        {
            "pier_dep_back_15m": window_count(q_t, q_g, q_t, q_g, back=900, fwd=0),
            "pier_dep_back_60m": window_count(q_t, q_g, q_t, q_g, back=3600, fwd=0),
        },
        index=ctx.dep_index,
    )
```

1. **Place it** in a new module of `prc/features/`, with a group of its own. Modules without a leading underscore are
   imported automatically. Adding it to a shared module such as `congestion.py` would re-key every block whose module
   imports that one, including the delivered operational blocks (section 3.5).
2. **Return the declared `outputs`** in order, indexed by `ctx.dep_index`. Output names must be unique across blocks.
   Read parents with `ctx.dep` and parameters with `ctx.params`.
3. **Targets.** A block that reads targets or fit rows must use `ctx.target` or `ctx.fit_mask` in the decorated
   function itself, and set `uses_target=True`.
4. **Primitives.** Use `core.epoch_seconds` (never cast times through int64), `core.bucket_key`, and the brute-force
   tested `congestion.window_count`, `count_upto`, `window_reduce` and `neighbour_gap`.
5. **Test and measure.** Write a synthetic test, following `tests/test_context_blocks.py`. Then run
   `prc ablate -c <base> --mode add-to-base --blocks <block> --emit-configs`, run the printed `prc queue` line, and
   read the result against the seed floor.
6. **Record the tier.** If the block enters a challenge submission configuration, add its tier and evidence to the
   availability section of `docs/feature_table.md` (between its `availability` markers). An operational block needs
   tags in `_long_tiers.py` and a `_catalogue.py` entry; then regenerate `docs/operational_features.md` with
   `python lab/feature_catalogue.py`. It also needs T1 and T2 tests.

### 4.2 A configuration, rung or key

1. **The file.** Write `configs/<name>.yaml` with `extends:`, `name:` and `notes:`. Check it with
   `prc features -c <name>.yaml --list`, then smoke-run it with `-o data.sample_frac=0.2 --no-test`.
2. **Operational rungs** extend `op_base.yaml` and list their blocks explicitly. `op_base.yaml` sets a raw target, a
   fit without targets above 7,200 s or wrong-date rows, `model.eval_max_target_sec` at 7,200, and the GPU with 8,000
   rounds, learning rate 0.08 and depth 7. A month pair sets `validation.holdout_months`; the forward check sets
   `scheme: forward_holdout` with Nov+Dec (`configs/op_h1_f1_plan_febdec.yaml`, `_fwd.yaml`).
3. **Pinned lists.** Add refit and permutation configurations to
   `tests/test_operational.py::test_permutation_and_refit_per_rung`, and deliverables to `MEMBERS` in
   `tests/test_feature_catalogue.py`. Every `configs/op_*.yaml` must also pass the rung invariants of
   `tests/test_operational.py` (the `OP_CONFIGS` glob): the GPU with devices `"0"`, depth 7, border count 128, learning
   rate 0.08 with 8,000 rounds, the RMSE loss, at most one anchor family, only `op_*` blocks plus `ALLOWED_BASE`
   (`core_keys`, `stand_runway_keys`, `aircraft_identity`, `taxi_distance`, `taxi_route`, `stand_geometry`), and no
   lab block in a refit configuration. A deliberate exception goes into the test with its reason: the lists
   `MAE_CONFIGS` and `H1_MEDIAN`, or a named case such as `op_h1_r4_nodoy_lr04.yaml`.
4. **A new key** needs a field with a default and a docstring in `prc/config.py`, validation in `__post_init__`, and
   removal from `hash()` while it holds its default. Keys under `data`, `validation` and `target` also enter
   `split_hash()`, which has no default-dropping: give the new key the same pop-while-default treatment there, or
   every fold-dependent cache, and `prc predict`'s parity lookup of a source run's refit cache, goes cold.

### 4.3 A data source with an external file

1. **Paths.** In `prc/paths.py`, add the raw directory, the parquet files and a manifest. Add the raw directory to
   `ensure_dirs`.
2. **The package** follows `prc/pru/` and `prc/reanalysis/`: `fetch`, `normalise` and `check`; a resumable, atomic
   download built on `prc.weather.download.http_get_bytes`; a normaliser that raises on a missing variable, a unit
   change or a gap; and a `fetch-<source>` subcommand with `--force`, `--offline` and `--check`.
3. **The block** declares `external=(paths.<FILE>,)` and reads only the parquet. The file's sha1 then joins this
   block's key directly and, through the parent hashes, the keys of the blocks that depend on it; unrelated blocks keep
   their keys. Read values at or before the anchor.
4. **Wiring.** Add the source to the missing-input hint in `build.build_matrix` (which lacks `fetch-era5` today) and
   the directory to `COPY_SPEC` in `lab/final_inputs.py`. Add a test on cut-down real files, and the licence to
   `docs/data_sources.md`.

### 4.4 A model backend

1. **The class** copies the interface of `prc/models/gbdt.py:CatBoostModel`: `encoding` (`"string"` or `"codes"`),
   `ITERATIONS_KEY`, `fit(X, y, *, cat_features, eval_set, sample_weight)`, `prepare`, `predict_prepared`, `predict`,
   `native_importance`, `refit_iterations` and `save`.
2. **Register it** in `GBDTS` in `prc/models/__init__.py`; `model.type` then selects it. `PerAirportModel` wraps any
   backend. The mixture's classifier stays CatBoost.
3. **For `prc predict`,** add it to `LOADABLE`, give it a `load`, and extend `load_model` and `feature_contract`. A
   `codes` backend must also save its `CategoryUniverse` levels.
4. **Tests.** Cover it in `tests/test_pipeline.py` and `tests/test_model_persist.py`. LightGBM and HistGB have no
   runs and no tests of their fit yet: locally on 13 Oct, every indexed run but four baselines was CatBoost. `hgb`
   appears only as the refused type in `tests/test_model_persist.py::test_only_catboost_runs_load` (`load_model`) and
   `tests/test_predict.py` (`prc predict`), which a loader for it must change. The lab's XGBoost trials are listed in
   section 5, item 8.

### 4.5 A new operational horizon

The long members anchor at `a = P - lead`, where `P` is the published schedule.
- **Lead-free blocks.** `op_hl_static`, `op_hl_plan` and `op_hl_demand` serve every lead.
- **Lead blocks.** `op_hl3_wx`, `op_hl3_hist` and their 24 h twins live in `operational_long.py`. They call
  `_horizon.wx_long`, `nm_baseline_long` and `history_at` with the lead from `_horizon.LEADS_S`.

**Mind the hashes.** `_horizon.py` is imported by `operational_long.py`, `operational_future.py` and
`adsb_archive.py`, so adding a lead to `LEADS_S` re-keys the delivered F1, h1f, h1f5, 3 h and 24 h blocks. A block
added to `operational_long.py` re-keys the 3 h and 24 h blocks. Their refits then fail `prc predict`'s drift check:
either retrain them, or put the new lead and its blocks in a new module. `_oplib.py` is imported by every operational
module, by `adsb_archive.py` and by `_horizon.py`, so an edit to it re-keys every delivered member at every horizon;
`operational.py` re-keys the H1 family. At H1 the anchor is the planned off-block minus `H1_LEAD_S` (900 s).

Then add rungs r0-r4, a runway-blind candidate (it drops what `_long_tiers.with_tag(columns, "own_runway")` returns)
and its `_fwd` and month-pair twins. Extend `_long_tiers.py` and the `HORIZONS`, `PREFIXES` and `DataPoint.known`
entries of `_catalogue.py`, and `tests/test_operational_long.py`, which pins `LEADS == {hl3, hl24}`. On the lab side,
extend `op_rules.py R --lead`, `op_calibrate.py`, `op_deliver.py --member`, `feature_catalogue.py` and
`op_ladder_report.py`.

### 4.6 Predicting a new data set

1. **Rehearse** on the delivered files:
   `python lab/final_inputs.py runbook --tree <dir> --from-blend <blend> --rehearse --out r.ps1`. Its `check` must
   reproduce every run within 1e-6 s.
2. **The real set.** Run `runbook --tree data_final --from-blend <blend> --months ...` and start the script with
   `-Ranking` and `-Template`. In order, it runs `init` and `validate --install`, derives `final26_<sha1[:8]>` and the
   row count, prepares, predicts every run, blends, stacks, then runs `submit` (which replaces the scored file, step 5)
   and `levelcheck`.
3. **One run by hand:** `python -m prc predict --run <id> -o data.version=<v> --tree <tree>`, with `PRC_DATA_DIR`
   pointing at the tree. In bash, put `PRC_DATA_DIR=<tree>` before the command. In PowerShell, run
   `$env:PRC_DATA_DIR = "<tree>"` first and `Remove-Item Env:PRC_DATA_DIR` afterwards; the bash form fails there.
   It works only for CatBoost runs trained with `refit_on_all` (in the configuration or as
   `-o model.refit_on_all=true`), without `--no-test`, and with `--keep-test` when `--seeds` is above 1. The
   seed-averaged deliverables were trained with `--seeds N --keep-test`; the median members are single runs. A refusal
   exits with code 2 and gives its reason; `--allow-unseen` and `--allow-drift` override two of them.
4. **What moves.** `frame_shift` rescales two count blocks, and the parity report names everything else that moved.
   For `op_h1_r4` on the final set, 1 block of 11 moved: the arrivals' anomaly columns of `op_h1_traffic` (local log).
5. **Submit with `PRC_DATA_DIR` set.** Without it, `prc submit` maps the run onto the delivered Jan+Jul template and
   writes 344,841 rows without any error. The final set needs 670,790 rows. Every `prc submit`, the runbook's
   included, overwrites `submissions/submitting.parquet`, the copy of the scored file; restore it afterwards as
   section 3.5 describes.
6. **The challenge stage on new months** runs `lab/final_challenge.py` in an isolated checkout
   (`lab/final_worktree.py`); `lab/final_run.sh` runs the whole chain.
7. **The operational files** come from `lab/op_deliver.py --tree <tree> --member <member> --config <cfg> --n <seeds>`
   (members `h1`, `hl3`, `hl24`, `h1f`, `h1f5`). `h1` adds `--tables runs/op_calib/h1_f1_pairs_lr08.parquet --rule-c`,
   then runs again with `--median-config op_h1_f1_plan_mae_exact_cpu.yaml`; `h1f5` needs `--any-hash`, and `h1f`
   `--runs <its two refits>` in place of `--config`, because the concept members were refitted through
   `-o model.refit_on_all=true` and carry another hash than their configurations (QUICKSTART path B lists every line).
   It never clips to the join window and never calls `prc submit`.

### 4.7 Rebuilding the ADS-B snapshot

1. **Fetch.** The geometry tables already exist in `data/`, and `fetch-adsb` needs only `airports.parquet`. Run
   `python -m prc fetch-geometry` (after `prepare`) only in a tree without them, or, as the runbook does in
   `data_final/`, in a new tree whose stands must be re-matched. Never run it in the runs' data tree: with or without
   `--offline` it re-normalises every table. If you must, compare the tables' sha1 before and after. Then run
   `python -m prc fetch-adsb --days ... --workers 3`.
2. **Build.** Run `python -m prc adsb-surface --snapshot s2 [--days A..B] --workers N`. It writes causal events per
   day and resumes where it stopped; each day is keyed by `BUILDER_VERSION` (3), the geometry, the inputs of days D-1
   to D+1 and the vehicle filter's history. The first three days are a warm-up.
3. **Finish** with `--check`, `--support` (needs `prepare`) and `--freeze`. A frozen snapshot refuses changes.
4. **Select it** by setting `PRC_ADSB_SNAPSHOT=s2` before the import. Its events' and support table's sha1 then join
   the keys of `op_h1_adsb`, `op_h0_adsb` and `op_h1_fut_adsb`, so in such a session `prc predict` refuses the
   delivered h1f and h1f5 refits as drifted. Unset `PRC_ADSB_SNAPSHOT` before predicting h1f or h1f5.
5. **Gates.** `lab/op_ladder_run.sh` runs them itself before the first ADS-B rung, once per frozen snapshot content:
   the three ADS-B test files, the real slice and `lab/op_adsb_audit.py --snapshot s2`. Only if they pass does it
   write the marker `runs/op_gates_adsb_<snapshot>_<events sha1>_<support sha1>.passed`, which the ADS-B rungs
   require. Snapshot s1 (561 days, 9,255,993 events, built in 22 min on 6 workers) passed with own-airframe
   contamination of 0.054 % at H1 and 0.227 % at H0 (findings 10.31).
6. **Separate inputs:** the stand archive (`python -m prc.adsb.archive`) and the challenge stage's `cache/adsb/`
   tables (`lab/rebuild_adsb_extra.sh`, `lab/adsb_new_days.sh`).

## 5. Where to continue

The post-operations model is finished, so these items are ranked by expected value for the operational goal. Each
state is the one recorded on 13 Oct (item 1: 14 Oct).

| # | item | state on record | first step |
|---|---|---|---|
| 1 | Close the 13 Oct records | Rule M passed for F1 (`op_h1_f1_plan_mae_exact_cpu`, 178.82 s; decision `M`, findings 10.37), and the manifest's `h1` is `h1__2026-10-13T13-54-07__final_op_h1_blend__aa1cd03b__mae.parquet`; F5 is delivered as `h1f5` (two seeds, OP MAE 177.43 s). The ladder report, the catalogue and notebook 07 were regenerated. F5's month pairs and forward split ran for information (better than F1 in 10 of 10 months; forward 164.95 s against 173.36 s; findings 10.37). The board checks of 14 Oct are done (the new H1 median 573.55, v84; F5 551.52, v85, re-uploaded after its rejection on 13 Oct). The owner uploaded the scored file as `adventurous-radio_final.parquet` on 14 Oct at 07:15 UTC: accepted on all 670,790 rows, no score returned | Findings 10.37 holds the records |
| 2 | Real planning feeds | Rule T, seed 42, OP MAE changes against `op_h1_r4_nodoy` (the H1 of the time), cumulative: the other flights' plans −2.6 s (F1); with the realised arrivals −4.0 s (1.3 s more); with the realised departure side −10.2 s (6.3 s more, not creditable: it traces the own slot); placebo −1.7 s. The data hold no AMAN, DMAN or A-SMGCS plans | Replace the realised reads of `op_h1_fut_rec`, `op_h1_fut_nm` and `op_h1_fut_adsb` (`prc/features/operational_future.py`) with an airport's plan feeds; rerun rule T. Any edit of `operational_future.py`, `_adsblib.py` or `_horizon.py` re-keys F1's `op_h1_fut_plan` and the h1f and h1f5 blocks (an edit of `_horizon.py` also the 3 h and 24 h blocks). Make it for the next model version, and retrain the affected refits afterwards (F1's three and its median member, h1f, h1f5; after a `_horizon.py` edit also the 3 h and 24 h refits) |
| 3 | Deployable inference | Batch only: blocks need the whole frame (on the final tree `op_h1_nm` took 150 s). The frame-wide tables are transductive; `frame_shift` corrects two count blocks only. The drift check pins the commit. Single seeds predict negative taxi times (`op_h1_r4_nodoy` seed 42's refit down to −265 s on the final set). The delivered means reach −126 s (the superseded `op_h1_r4_nodoy`, 36 of 670,790 rows below zero) and −102 s (F1, 33 rows); `lab/op_deliver.py` clips to [30, 200000] s | Freeze the tables at fit time; build a feature service per anchor; latency was never measured |
| 4 | A median at 3 h and 24 h | Rule C failed only through Istanbul: +2.164 s (3 h) and +2.175 s (24 h) against the 2 s bar, for overall gains of 2.921 s and 2.806 s | Pre-register a per-airport fallback, then run `lab/op_calibrate.py` |
| 5 | The ADS-B rungs that never ran | `op_h0_r4_adsb` (can public ADS-B replace the push feed?) and `op_h1_r2_adsb` have no runs, although the s1 gates passed. At H1, R6 (R4 plus the public surface picture) changed the OP MAE by +0.05 s on the five month pairs | Run `bash lab/op_ladder_run.sh "<deadline>" op_h0_r4_adsb`. Compare it with `op_h0_r5`, or add a `_nodoy` twin and compare it with `op_h0_r4_nodoy` (178.2 s) |
| 6 | Three strict xfails | `tests/test_operational_future.py` holds `WRONG_LINK`, `ADSB_SHIFT_PARAMS` and `PLACEBO_REACH`; the file explains each | Fix one, watch its strict xfail fail, and remove the mark. A fix edits `operational_future.py`, `_adsblib.py` or `_horizon.py` and so re-keys F1's `op_h1_fut_plan` and the h1f and h1f5 blocks, and a `_horizon.py` edit also the 3 h and 24 h blocks (item 2): make it for the next model version, and retrain those refits afterwards |
| 7 | Heathrow's 2026 ADS-B | The 2025 support table NaNs every EGLL family, since the taxi gate held on only 6.6 % of 2025 flights. Yet the 2026 taxi-out seen share is 0.73-0.99, and no 2026 targets exist to validate a new regime | Try the rule as block parameters first (`features.params: {op_h1_adsb: {min_support: ..., support_margin: ...}}` in a new configuration); edit `_adsblib.DEFAULTS` only for a new model version. Or build a new snapshot (section 4.7). Then validate it on data |
| 8 | A second learner | The LightGBM and HistGB backends have no runs, no tests of their fit and no loader. XGBoost was tried in the lab: worse as a global regressor (findings 10.2) and a tie as an ADS-B stage member (10.20, `v56x`); LightGBM served as a stand-in in the 2 × 2 weather analysis (10.29). `lightgbm` and `xgboost` are commented out in `requirements.txt`. Findings 10.25 found no literature lever worth 0.1 s | Write the tests first, then persist the category levels |
| 9 | Licence and hindsight limits | The 3 h and 24 h members were never measured without the TAF, whose archive (Ogimet) has no open licence. The long members' scheduled demand, and F1's plan counts, hold only the flights that operated (no cancellations, a mild hindsight). The own-stand trace in h1f is unmeasured (`docs/operational_model.md` sections 3b and 5) | Drop the TAF columns of `op_hl3_r4_rwyblind` and `op_hl24_r3_rwyblind` as `op_h1_r4_nodoy_notaf` did; for a new model version, rebuild `op_hl_demand` and `op_h1_fut_plan` from a schedule with cancellations (an edit of `operational_long.py` re-keys the 3 h and 24 h refits, and one of `operational_future.py` F1's, h1f's and h1f5's) |
| 10 | The challenge backlog | The 15 snow encodings (`docs/snow/SYNTHESIS.md`) need the 27-member base retrained. The review items read 0-0.04 s each, and the stage already holds 350 of the 365 days | See findings 10.15 and 10.25. Worth doing only if the challenge metric matters again |

**Where the records and the documents disagree** (the records win):

| where | says | records |
|---|---|---|
| comments of `configs/op_hl3_r4_rwyblind.yaml`, `op_hl24_r3_rwyblind.yaml`; docstrings of `lab/op_ladder_report.py`, `prc/pru/__init__.py`; the hint of `build.build_matrix` | runway knowable for 60.5 % and 56.4 %; early stopping "not fixable"; ansperformance.eu; the fetch commands | 53.3 % and 53.4 % (`docs/op_runway_knowability.txt`); `eval_max_target_sec` exists; eurocontrol.int; `fetch-era5` missing |

---

## Appendix A. The experiment harness (reference)

This appendix describes the harness that trains and measures the base models. It is built as a
**harness**, not a model: the point is that adding one feature function gets you a measured, comparable answer to
*"did that actually help, beyond seed noise?"* [findings.md](findings.md) records what the data turned out to be,
what moved the score, what did not, and the improvement backlog.

It moved here from the README on 14 October 2026 and keeps the state and figures of the
harness's development (mid-September to early October) unless marked. Current results:
[docs/report.md](report.md); the package map: section 2.2 above;
every subsystem and key as they are now: the module and field docstrings of `prc/` (the keys in `prc/config.py`)
and `python -m prc <command> --help`.

### The commands

```powershell
python -m prc fetch-weather                                  # once: METAR + TAF archives (see Notes)
python -m prc fetch-geometry                                 # once: OSM stands/taxiways + OurAirports runways (see Notes)
python -m prc fetch-era5                                     # once, after fetch-geometry: ERA5 hourly surface data per airport (Open-Meteo)
python -m prc fetch-adsb [--days YYYY-MM-DD ...] [--kind prod|staging] [--workers 3]   # adsb.lol daily history, kept near the ten airports (resumable)
python -m prc adsb-surface --snapshot s1 [--check | --support | --freeze]   # build the ADS-B surface snapshot of other aircraft (resumable), then check, support and freeze it (DEVELOPER_GUIDE.md section 4.7)
python -m prc fetch-pru [--force] [--offline] [--check]      # Eurocontrol PRU open CSVs: daily delays as features, monthly taxi-out for `levelcheck` (see Notes)
python -m prc features -c exp002_catboost_core.yaml --list   # resolved blocks + cache status
python -m prc train    -c exp001_baselines.yaml -o model.type=aobt3_proxy
python -m prc train    -c exp002_catboost_core.yaml
python -m prc queue    -c exp005_catboost_raw_target.yaml exp013_raw_geometry.yaml --seeds 3   # several configs in a row, live output, summary at the end
python -m prc ablate   -c exp002_catboost_core.yaml --blocks turnaround
python -m prc ablate   -c exp028_final_stale_only.yaml --mode add-to-base --emit-configs --blocks arrival_conditions,nm_cross_deltas   # "the config plus one new block", as queue-able configs
python -m prc leaderboard --features
python -m prc audit [-c exp011_clean_fit.yaml] [--run <run_id>]   # data quality: extreme rows, the untyped segment, what rules would flag, disrupted airport-days
python -m prc train    -c exp065_lean2_window_long.yaml      # a base member of the submission: folds for the score, then a refit on all months
python -m prc train    -c exp051_lean2.yaml --seeds 3 --keep-test   # three seeds that also predict the ranking set, for `blend`
python -m prc blend    --runs <run_id> <run_id> <run_id>     # average finished runs into a new run directory (`submit` takes it)
python -m prc blend    --runs <run_id> ... --weights 0.2 0.1 ... --lobt-bound --name <name>   # weighted, clipped to the NM join window
python -m prc progress [--run <run_id>]                      # iteration, rate and ETA of an in-flight CatBoost fit
python -m prc levelcheck --run <run_id>                      # predicted mean per airport-month against the PRU's published mean taxi-out
python -m prc submit   --run <run_id>                        # the bare run-directory name, not runs\<run_id>
python -m prc predict  --run <run_id> -o data.version=<v> --tree <dir>   # a refit model applied to the current data tree's ranking rows, no retraining (QUICKSTART.md path D; DEVELOPER_GUIDE.md section 4.6)
```

Any config value can be overridden from the command line:

```powershell
python -m prc train -c exp002_catboost_core.yaml -o model.per_airport=true model.params.depth=10
python -m prc train -c exp002_catboost_core.yaml -o data.sample_frac=0.2 --no-test   # fast sweep
```

---

### What the data actually is

| | |
|---|---|
| Training | 12 monthly parquets, all of 2025 — 4,167,797 movements (2,085,047 departures with a known taxi time) |
| Ranking | `ranking.parquet` — Jan + Jul **2026**, 344,841 departures to predict |
| Airports | **10**, not the 11 the challenge's data page lists — LTAI (Antalya) is absent from the delivered files |

Current counts, the final release and the NM fields: [docs/report.md](report.md) section 1.1 (Table 1).

Four facts drive the whole design:

1. **The task is really "infer the off-block time".** `TAXITIME == MVT_TIME - BLOCK_TIME` exactly,
   and ranking blanks only `BLOCK_TIME` and `TAXITIME`. The actual **takeoff time is given**, for
   every flight — so congestion windows may look *forward* as well as backward. That is unusual and
   worth exploiting.
2. **`AOBT_3_flt` is the strongest single feature.** NM's own actual off-block time survives in the
   ranking set for ~98% of departures, so `MVT_TIME - AOBT_3_flt` is an independent estimate of the
   answer. The two sources disagree with a std of ~390s, which is the room a model has to work in.
3. **Ranking arrivals are fully populated.** Their block times and taxi-in times are all there,
   which is what makes ground-occupancy and stand-turnaround features legal for the 2026 months.
4. **The airline is recoverable.** `AIRCRAFT_OPERATOR_flt` is SHA-256 hashed, but `CALLSIGN_flt`
   (`DLH400`) and `FLIGHT_mvt` (`LH400`) are in the clear — and `FLIGHT_mvt` survives the ~1.5% of
   rows where the NM join failed.

#### Reference scores

Holdout = Jan + Jul 2025, the seasonal analogue of the test set.

| model | RMSE | MAE |
|---|---|---|
| global median | 693.6 | 315.9 |
| `(stand, runway)` median | 636.1 | 222.4 |
| NM off-block proxy | 672.8 | 258.7 |

**RMSE here is dominated by a small tail.** Targets run to 131,167s (36 hours); the top 0.05% of
rows carry a large share of the total squared error. That is why the group-median baseline beats the
NM proxy on RMSE despite the proxy being far better correlated — and it means outlier handling is
worth as much attention as feature engineering. [docs/report.md](report.md) section 1.4 quantifies it.

---

### Configuration

An experiment is one YAML file in `configs/`, loaded into frozen dataclasses by `prc/config.py`.
Unknown keys and bad enum values raise at load time, so a typo cannot silently train the wrong
thing. Three mechanics apply to every key:

- **`extends: <file>`** names a parent (relative to the file's own directory). The child is
  deep-merged over it, and **lists replace rather than merge**: a child that adds one feature group
  repeats the whole `include` list. `base.yaml` holds the defaults; the experiments inherit it directly
  or through `exp002_catboost_core.yaml`.
- **`-o key=value`** on the command line overrides any key by dotted path; values are parsed as YAML
  scalars (`true` is a bool, `0.2` a float, `"0"` a string). Unknown keys raise here too.
- **Hashing.** Everything except `notes` goes into the 8-character config hash in the run id
  `runs/<timestamp>__<name>__<hash>__s<seed>`. The resolved config is archived as
  `config.resolved.json` in every run directory, and `prc ablate` writes its generated variants to
  `configs/_generated/`.

The tables give the keys of the September harness. The later keys `features.drop_columns`, `model.mixture.*`,
`model.month_weights`, `model.eval_max_target_sec` and `model.block_dropout` are described in the field docstrings of
`prc/config.py`. The later validation scheme `forward_holdout` (fit only on the months before the holdout) is
described in `prc/validation.py`. The registered groups have grown from the list below to 47;
`python -m prc features -c <cfg> --list` shows a config's blocks with their groups.

#### Top level

| key | default | meaning |
|---|---|---|
| `name` | `unnamed` | The experiment's name, used in the run id and the leaderboard (`--name` filters on it). |
| `seed` | `42` | CatBoost's `random_seed` and the seed for `data.sample_frac`. `train --seeds N` runs `seed`, `seed+1`, … and records the spread as the noise floor. Fold membership never depends on it. |
| `notes` | `""` | Free text, archived with the run and shown in `index.csv`. The only key left out of the config hash. |

#### `data`

| key | default | meaning |
|---|---|---|
| `version` | `v1` | Partitions the feature cache (`cache/features/<version>/`). Bump it when `prc/data.py` changes what the movement frame contains. |
| `airports` | `all` | `all`, or a list of ICAO codes that restricts the **training** rows. Ranking predictions are still produced for every airport; a per-airport model answers an airport it never saw with the global median. |
| `sample_frac` | `1.0` | Fraction of training rows kept, drawn with `seed` *before* the split, so the validation months are thinned too. RMSE from a sample is therefore not comparable with a full run (the metric is dominated by rare extreme taxi times); MAE is. Use `0.2` with `--no-test` for sweeps. |
| `clean.max_taxi_sec` | `null` | Flag training departures with a taxi-out above this. Flags apply to **training rows only**: a flagged row can leave the fit, never the score (validation and ranking rows are untouched), and a flag is never a feature. |
| `clean.min_taxi_sec` | `null` | Flag taxi-outs below this, except on a helicopter pad (`RUNWAY_mvt` `H`/`HEL`): those few-second, sometimes negative taxis are real and recur in 2026. |
| `clean.day_offset_tol_sec` | `null` | Flag taxi-outs within this many seconds of a whole number of days, i.e. a block time on the wrong date. |
| `clean.untyped_airports` | `[]` | Flag departures with no `AIRCRAFT_TYPE_mvt` at these airports. `[LIRF]` names the Rome tail segment. |
| `clean.drop_from_fit` | `true` | Remove flagged rows from every fold's training index. `false` keeps them and only reports the clean metrics. Either way `rmse_clean` / `mae_clean` / `n_clean` (metrics on the unflagged holdout rows) appear next to `rmse` in `metrics.json`, `index.csv` and the leaderboard, and `metrics_by_airport.csv` gains `rmse_clean` / `n_clean`. `data.clean` sits under `data`, so the rules are part of the fold-cache key. |

#### `features`

| key | default | meaning |
|---|---|---|
| `include` | `[group:core]` | Block names and `group:<name>` references. Dependencies are pulled in transitively, so `group:congestion` alone also brings `core_keys` and `calendar`. Registered groups: `core`, `temporal`, `nm_schedule`, `stand_runway`, `aircraft`, `congestion`, `turnaround`, `weather`, `weather_fcst`, `geometry`, `stale`, and the one-block groups of the 2026 sweep (`turnaround_ext`, `flight_number`, `runway_sequence`, `taxi_windows`, `pier`, `zone`, `arrivals_live`, `traffic`, `nm_context`, `stale_context`, `key_stats`, `weather_load`, `pru`; see Notes, "The context blocks"). `prc features -c <cfg> --list` shows the resolved blocks. |
| `exclude` | `[]` | Blocks to drop. Excluding one that an included block still depends on is an error, not a silent drop. |
| `params` | `{}` | Per-block parameters, `{block: {param: value}}`; they enter that block's cache key. The blocks that read any: `stand_runway_target_stats.min_support` (20 — observations a (stand, runway) pairing needs before its median is trusted); `prev_stand_arrival.max_age_sec` (86400 — how old the previous arrival at the stand may be); `max_age_min` on every weather and forecast block (90 — how long a report stays in force without a newer one); `taf_planning.lead_hours` (6 — the TAF must have been issued at least this long before takeoff). |

#### `target`

| key | default | meaning |
|---|---|---|
| `column` | `TAXITIME_SEC_mvt` | The target column in `target.parquet`. |
| `transform` | `none` | `none`, `log1p` or `sqrt`, applied to the fitting target; predictions are inverse-transformed. Metrics are always computed on raw seconds. `exp002` uses `log1p` because the target is right-skewed. |
| `clip` | `null` | `[lo, hi]` applied to **training** targets only (`exp002`: `[60, 7200]`), to keep the tail from dominating the fit. Ignored when `residual_of` is set. |
| `residual_of` | `null` | A feature column subtracted before fitting and added back after, so the model learns a correction to it. Requires `transform: none`. Where the base is NaN the model learns the raw target, and the block's indicator column tells it which regime it is in. `exp020` uses `stale_base` (see Notes, "Stale block times"); `aobt3_proxy` is the other candidate. |

#### `validation`

| key | default | meaning |
|---|---|---|
| `scheme` | `seasonal_holdout` | `seasonal_holdout`: train on the other months, validate on `holdout_months`. `strict_forward`: train only on months **after** the holdout (the year-over-year drift check). `group_kfold_day`: `n_folds` folds by calendar day. `month_forward`: expanding window, one fold per month. None of them uses a random number. |
| `holdout_months` | `["2025-01", "2025-07"]` | The validation months for `seasonal_holdout` and `strict_forward`; the default is the seasonal analogue of the Jan/Jul 2026 test set. `exp029_winter_check.yaml` uses `["2025-02", "2025-12"]`, the months with disrupted days, for winter features (see Notes, "Disrupted days"). |
| `n_folds` | `5` | `group_kfold_day` only. |

#### `model`

| key | default | meaning |
|---|---|---|
| `type` | `catboost` | `catboost`, `lightgbm` (optional package, not in `requirements.txt`; no recorded run uses it), `hgb` (sklearn HistGradientBoosting; demotes categoricals above 255 levels), or a baseline: `median`, `group_median`, `aobt3_proxy`. |
| `per_airport` | `false` | One model per airport behind the same interface (needs `group:core` for the `apt` column). Gain importances are n-weighted so the run stays comparable with a global one. |
| `refit_on_all` | `false` | After the folds are scored, fit one more model on **every** training row, holdout months included, and take the ranking predictions from it instead of the fold mean. The metrics are still the folds'. A GBDT has no early-stopping set then, so it runs the fold models' best iteration scaled by the row ratio (all rows / a fold's rows; per airport for a per-airport model); `refit.json` in the run records the counts, `test_predictions_folds.parquet` keeps the fold mean and the log compares the two. Skipped by `--no-test`, `--seeds` and `ablate`. See Notes. |
| `params` | `{}` | Passed straight to the backend. **CatBoost**: any `CatBoostRegressor` key on top of the harness defaults (`loss_function`/`eval_metric` RMSE, `od_type` Iter, `od_wait` 200, `iterations` 5000, `learning_rate` 0.05, `depth` 8, `l2_leaf_reg` 3, `one_hot_max_size` 16, `thread_count` -1, `verbose` 200); `random_seed` and `train_dir` are set by the harness and cannot be overridden. `exp002` sets `task_type: GPU`, `iterations`, `learning_rate`, `depth`, `l2_leaf_reg`, `one_hot_max_size`, `border_count`, `od_wait`, `verbose`; `exp003` adds `depth: 7`, `iterations: 3000` for the smaller per-airport fits. **LightGBM**: `LGBMRegressor` keys plus `early_stopping_rounds` (200). **hgb**: `HistGradientBoostingRegressor` keys. **group_median**: `keys` (list of key lists, default stand→runway→airport hierarchy) and `min_support` (20). **aobt3_proxy**: `column` (`aobt3_proxy`). **median**: none. |

#### `evaluation`

| key | default | meaning |
|---|---|---|
| `primary_metric` | `rmse` | Which metric becomes `primary_value` in `metrics.json`: `rmse`, `mae`, `medae`, `r2`, `bias` or `p90_abs_err`. All of them are always computed. |
| `breakdowns` | `[apt, month]` (`base.yaml`: `[apt, month, hour_local, split]`) | Per-slice metrics written to `metrics_by_breakdown.csv`: any column of the movement frame, plus `month` and `hour_local` derived from the takeoff time. Unknown names are skipped. The per-airport table is always written regardless. |
| `permutation.enabled` | `true` | Block permutation importance on the last fold's validation rows, recorded into `feature_effects.csv`. |
| `permutation.granularity` | `block` | `block` shuffles all of a block's columns with one shared permutation; `column` shuffles each column alone (and lets correlated siblings cover for each other). |
| `permutation.repeats` | `3` | Shuffles per unit; the spread is recorded as `std`. |
| `permutation.sample` | `200000` | Validation rows used for the measurement; `0` means all of them. |
| `shap` | `false` | Reserved; nothing reads it yet. |

#### `geometry`

| key | default | meaning |
|---|---|---|
| `stands_file` | `null` | CSV of hand-placed stands (`apt, stand_mvt, lat, lon`, optional `ref`, `note`) that `fetch-geometry -c <cfg>` merges over the OSM stands when it builds the geometry tables; path relative to the repo root. The tables are shared by every experiment, so this is a build setting, not an experiment setting: change the file, then `fetch-geometry --offline`. See Notes. |

---

### Adding a feature

Drop a decorated function into any module under `prc/features/`. It is imported and registered
automatically — there is no central list to edit.

Section 4.1 above gives the current recipe (where a block goes, its tests, its
tier records).

```python
@feature(
    group="congestion",
    outputs=("rwy_dep_pm10m",),
    depends_on=("core_keys",),
    doc="Departures on the same runway in the +-10 minutes around takeoff.",
)
def my_block(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    dep = mvt.loc[ctx.dep_index]
    ...
    return pd.DataFrame({"rwy_dep_pm10m": values}, index=ctx.dep_index)
```

Then select it in a config by name or by `group:`:

```yaml
features:
  include: [group:core, group:congestion, my_block]
  exclude: [days_since_epoch_block]
```

#### The contract

A block receives the **whole movement frame** — departures *and* arrivals, training *and* ranking,
time-sorted — and returns a DataFrame indexed by `mvt_id` with exactly its declared `outputs`. It is
not a row-wise transform: most of the signal here is cross-row context, and a January 2026 departure
can only see its queue if the January 2026 arrivals are in the same frame.

If a block reads the target it must declare `uses_target=True`. Those blocks are recomputed inside
every CV fold from that fold's training rows only (`ctx.fit_mask`), and cached per fold. Everything
else is computed once for the whole corpus.

#### Why leakage is structurally impossible

`prc prepare` masks `TAXITIME_SEC_mvt` and `BLOCK_TIME_UTC_mvt` on **every** departure in
`movements.parquet` — training rows included — and stores the targets separately. Feature code
literally cannot see the answer, and the feature distributions a model is fitted on are identical to
the ones it predicts on. Arrival rows keep theirs, which is both legal and necessary.

If a run ever scores dramatically below the baselines above, that is a leak, not a breakthrough.

#### Caching

One parquet per block, keyed by a hash of the block's **source code** (plus its version, params and
parent hashes). Edit a feature function and its cache invalidates itself; edit a dependency and
everything downstream invalidates too. Per-block rather than per-feature-set is what makes ablation
sweeps cheap — toggling one block reuses every other block's parquet untouched.

Helpers outside `prc/features/` are not tracked by the source hash; bump `version=` on the spec if
you move logic there.

---

### Judging whether a feature was worth it

Four measurements, recorded automatically into `runs/feature_effects.csv`:

| method | cost | trust |
|---|---|---|
| `gain` | free | ranking hint only — correlated features steal each other's gain |
| `permutation` | seconds | solid; shuffles a **whole block** at once, since columns within a block mask each other individually |
| `ablation` | a retrain per block | the honest number |
| `seed_noise` | N retrains | the reference line that makes the other three interpretable |

```powershell
python -m prc train  -c exp002_catboost_core.yaml --seeds 3        # establish the noise floor
python -m prc ablate -c exp002_catboost_core.yaml --mode leave-one-out
python -m prc ablate -c exp002_catboost_core.yaml --mode add-one-in --blocks my_block
python -m prc leaderboard --features
```

`leave-one-out` under-credits a new feature whose signal an existing block already carries — which
tells you about redundancy, not about the idea. `add-one-in` measures the idea. Both hold the seed
and fold assignment fixed so the delta is attributable to the block.

A block cannot be removed while something downstream still depends on it, so `ablate` removes its
dependents too and prints what actually went:

```
nm_deltas: also removed ['nm_proxy_context']
```

The delta is then "what are this block and everything built on it worth", and the blocks that moved
together are recorded in the `columns` field of `feature_effects.csv` so the number is never read as
belonging to one block alone.

**A delta smaller than the seed-noise floor is not a finding.** `leaderboard --features` prints the
floor at the top of the table for that reason.

---

### Validation

Default is `seasonal_holdout`: train on 2025-02…06 + 08…12, validate on **2025-01 + 2025-07** — the
seasonal analogue of the Jan/Jul 2026 test set.

It does *not* capture year-over-year drift, so run `strict_forward` alongside it as an honest
extrapolation check before trusting any feature that encodes absolute time.

**Never use row-level random KFold here.** Congestion features make flights in the same
fifteen-minute window near-duplicates; a random split puts near-copies on both sides and reports a
score the leaderboard will not reproduce. Day-level grouping is the minimum, and every scheme in
`prc/validation.py` respects it.

---

### Tests

```powershell
python -m pytest -q
```

`tests/test_pipeline.py` runs in seconds without the competition data. `tests/test_features.py`
checks every registered block against its contract and asserts the leakage properties — including
that no feature correlates with the target above 0.99, and that target-using blocks actually respond
to `ctx.fit_mask`. It skips blocks whose downloaded weather tables are missing, and says so.

The weather tests need neither data nor network:
- `test_weather_parse.py` covers the METAR parser and the download-response checks;
- `test_weather_taf.py` covers TAF time resolution, forecast state and which TAF answers a query;
- `test_weather_windows.py` checks every METAR window lookup against brute force;
- `test_feature_cache.py` checks that the external-file digest leaves other cache keys untouched.

The geometry tests need neither data nor network either:
- `test_geometry_parse.py` covers stand-ref normalisation, the stand matching rules, the OurAirports
  filter, the Overpass parser and the download-response checks;
- `test_geometry_routing.py` routes a synthetic seven-node airport: runway penalty, true-metre
  reporting, runway crossings, snapping, gap bridging, and what happens to an unreachable stand;
- `test_geometry_manual.py` covers the hand-placed stands file: validation, and how a placed or
  blanked row overrides the OSM match;
- `test_cleaning.py` covers the cleaning rules on a synthetic frame: each rule, the helicopter
  exemption, that ranking rows never get a flag, and that only the fit side of a split shrinks.
- `test_disruption.py` covers the arrival taxi-in anomaly and the disrupted-day report mask on a
  two-day airport; `test_stand_asof.py` the shared stand join; `test_context_blocks.py` every block
  of the 2026 sweep on one hand-built airport morning, value by value.
- `test_pru_normalise.py` covers the PRU CSV readers (Latin-1, bzip2, missing columns, bad dates,
  duplicates), the two tables, the download-response check and the `pru_daily` block;
  `test_levelcheck.py` the level-check arithmetic and its flag.
- `test_congestion_windows.py` also checks the per-row windows, `window_reduce` and `neighbour_gap`
  against brute force.

The model, post-processing and later feature tests need neither data nor network either:
- `test_mixture.py` covers the regime mixture on synthetic frames (candidates, labels, arithmetic, model
  contracts; one small CPU smoke test); `test_stale_schedule.py` the stale-schedule residual base and the
  fold-only stale rates; `test_delay_stale.py` the fold-safe stale rates among delayed departures.
- `test_postprocess.py` covers the NM join-window clip; `test_blend.py` `prc blend`'s arithmetic.
- `test_era5.py` covers the ERA5 normaliser's validation and the block's causality; `test_deicing.py` the
  de-icing context; `test_live_state.py` the NM proxy anomaly, `dep_live_state` and `deice_location`;
  `test_wake_queue.py` wake minima and runway queues on a hand-made takeoff sequence.

The final-set path needs neither data nor network either:
- `test_predict.py` trains three tiny CPU runs (global, per airport, regime mixture) on a synthetic data tree
  (`tests/synthetic_tree.py`) and checks that `prc predict` reproduces their refit predictions (from the cache, under
  a new `data.version`, from a run without state files), predicts a second tree with new stands (counts rescaled,
  parity measured, unseen levels judged), feeds `blend` and `submit`, and refuses runs without a refit model, the
  wrong data tree or frame, other training targets, feature drift (blocks with input files included), unseen levels
  and other overrides;
- `test_frame_shift.py` the count rescale against the two blocks, the parity report and the unseen verdict;
- `test_model_persist.py` covers saving and loading the CatBoost, per-airport and mixture models (the day-offset
  regime and isotonic calibration included), the state replay, the feature order and unseen categorical levels;
  `test_prepare_rows.py` the `PRC_EXPECTED_RANK_ROWS` override and `prepare.json`; `test_final_inputs.py`
  `lab/final_inputs.py` (each rejection of `validate`, `init`'s copy and record, `install`'s guard, the runbook
  script, which PowerShell and bash must parse, `derive`, `latest` and `check`); `test_run_index.py` the locked
  index appends and run ids that never collide.

`test_adsb_stage_tables.py` checks the cached ADS-B stage tables for leakage (every own-movement timing
before the ADS-B takeoff, no input with |r| >= 0.99 against the target); it is skipped unless `cache/adsb/`
and the target exist.

The later gates (`tests/test_operational.py`, `test_operational_long.py`, `test_operational_future.py`,
`test_adsb_surface.py`, `test_operational_adsb.py`, `test_feature_catalogue.py`) are described in
sections 2.4 and 3.3 above.

---

### The second stage and post-processing (`lab/`)

The harness trains and measures the base models. The submissions add a second stage that reads ADS-B surface
surveillance, plus post-processing; both live in `lab/`. Every lab script reads finished run directories and writes
a new one in the same format, so `prc blend`, `prc submit` and the leaderboard accept its output, and each records
its inputs and parameters in the new run's `metrics.json`. `python lab/manifest.py <run>` turns those records into
`docs/submission_manifest.md`; QUICKSTART path C lists the commands in order.

| step | script | what it does |
|---|---|---|
| stack | `lab/final_stack.py <blend> --inwindow 0.5 --write <stack>` | join-window clip, in-window estimate for the hedged rows, linked-row regime exclusion |
| base for all of 2025 | `lab/build_winter_base.py <stack> <group> ...` (one argument per group, its runs comma-separated) | the stack's out-of-fold prediction on Jan + Jul, month-pair members' on the other ten months |
| ADS-B features | `bash lab/rebuild_adsb_extra.sh` | links, four feature sets, per-stand archive, time-to-location map (per day, cached in `cache/adsb/`) |
| ADS-B stage | `lab/adsb_apply.py <base> ...` | residual formulation (truth minus base) or, with `--direct`, the taxi time itself; cross-fitted by day on the holdout, refitted on all covered days for 2026 |
| member averages | `prc blend --runs ...` | equal-weight averages of stage runs over seeds and settings |
| formulation blend | `lab/blend_direct.py <resid> <direct> --w 0.3` | 0.7 x residual + 0.3 x direct on the rows both cover |
| base swap | `lab/base_swap.py --swap CFG=RUN ...` | retrained base members into the stack; the new base on the rows no stage corrects |
| forced-long shift | `lab/forced_long_shift.py <run>` | shrunk shift where the join window forces a taxi above 30 minutes |
| one candidate | `lab/build_candidate.py --resid ... --direct ... --new-stack ... --name ...` | member average, formulation blend, base swap and shift in one command |
| manifest | `lab/manifest.py <run>` | the lineage of a submitted run, from the run records |

The ADS-B own-movement features observe the flight's pushback and taxi, so they are challenge-only (tier M in
`docs/feature_table.md`): they reconstruct the recorded off-block from the takeoff the dataset provides, which an
operational prediction made before the aircraft moves would not have. ADS-B data: adsb.lol globe history, made
available under the ODbL 1.0 (attribution and ODbL notice: see `docs/data_sources.md`). `lab/ideas/` holds the
scripts that build the third and fourth feature sets (pushback, first reception, track quality, stop classes, the
stand archive, the time-to-location map, the inbound turnaround); `lab/README.md` lists every lab script.

---

### Notes

- **CatBoost speed.** Measured on the full 1.74M-row training split at `depth: 8`,
  `border_count: 128`, 20 threads: **0.57 s/iteration**, so ~28 min for 3000 iterations. `depth: 6`
  halves that to 0.29 s/iteration. These figures come from a short benchmark, in which CatBoost's
  quantisation caches stay warm between fits in one process: real fits on the same split measured
  1.3-2.2 s/iteration on the CPU (the comment in `configs/exp002_catboost_core.yaml`). The GPU figures
  below are the ones most runs used.
  - The size of the eval set is *not* a factor — 344k vs 60k rows measured identical (0.565 vs
    0.587 s/iter), so there is nothing to gain by thinning it. `metric_period` does not help either:
    CatBoost ignores it for the evaluation metric whenever the overfitting detector is on, and says
    so in a warning.
  - CatBoost saturates all cores, so **anything else you run competes directly with it**. A run left
    alone finishes in the time above; the same run sharing the machine with a test suite and an
    ablation sweep took three times as long. Worth remembering before concluding a run has hung.
  - For ablation sweeps use `-o data.sample_frac=0.2 --no-test`, or `model.per_airport=true`
    (10 models of ~208k rows). `prc/models/gbdt.py` keeps LightGBM and HistGB adapters behind the
    same interface if a faster backend is wanted for sweeps.
- **CatBoost on the GPU.** `-o model.params.task_type=GPU` is the whole change: every `model.params`
  key goes straight to `CatBoostRegressor`. Measured on an RTX 4070 Laptop (8 GB) with `exp002`
  (depth 8, 1500 iterations, `border_count` 128):

  | data | CPU (14 cores) | GPU | RMSE CPU / GPU |
  |---|---|---|---|
  | 20 % sample | 398 s fit, 483 s wall | 89 s fit, 143 s wall | 440.5 / 439.7 |
  | full 1.74 M rows | ~1.34 s/iteration | 298 s fit, 388 s wall (~0.2 s/iteration) | in the development machine's run index |

  Two caveats. GPU training is **non-deterministic** (the docs say so: the order of floating-point
  summation varies), so a seed-noise floor measured on GPU includes GPU noise, and an ablation pair
  must be read against that floor. And `random_strength`, `rsm` and `posterior_sampling` are
  CPU-only; nothing here sets them. Put `devices` in YAML only, as the string `"0"`: through `-o`
  it would parse as an int.
  - RMSE on a 20 % sample is not comparable with RMSE on the full data: the sample thins the
    validation months too, and the metric is dominated by rare extreme taxi times. MAE is the
    number to compare across sample sizes (156 vs 153 above).
- **External data: weather.** `python -m prc fetch-weather` downloads METARs (Iowa Environmental
  Mesonet) and TAFs (Ogimet) once into `data/additional_data/weather/raw/`, one file per
  station-month, and normalises them into `metar.parquet`, `taf_issues.parquet` and
  `taf_segments.parquet`. The download is resumable: re-running it fetches only what is missing.
  `--check` reports coverage, parse quality and a weather-regime table (hours of snow, LVP, de-icing
  conditions per airport-month); `--offline` rebuilds the tables after a parser change, no network.
  - `group:weather` holds the METAR blocks: the weather in force at takeoff (`wx_now`), over the
    hours before it (`wx_past`), and in the hour after it (`wx_future`).
  - `group:weather_fcst` holds the forecast blocks: the TAF in force (`taf_now`), the TAF issued at
    least 6 h earlier plus the weather it failed to forecast (`taf_planning`), and the METAR TREND
    (`wx_trend`). Configs: `exp006_weather.yaml`, `exp007_weather_taf.yaml`.
  - The blocks declare the tables as `external=` inputs, so the tables' content digest is part of
    the cache key: re-downloading or re-parsing invalidates exactly the weather caches.
  - Observed weather *after* takeoff is hindsight. It is legal here only because the test months
    are historical.
- **External data: airport geometry.** `python -m prc fetch-geometry` (after `prepare`) downloads one
  OpenStreetMap extract per airport through the Overpass API (stands as `aeroway=parking_position`,
  gates, and every taxiway/taxilane/runway way) and the OurAirports `airports.csv` / `runways.csv`
  (reference points, runway thresholds with headings) into `data/additional_data/airport_geometry/raw/`,
  then normalises them into five tables: `airports`, `runway_ends`, `stands` (position, apron zone
  from k-means), `stand_matches` (which OSM stand each `STAND_mvt` value means, and by which rule) and
  `stand_runway_routes` (great-circle and shortest-taxiway-route metres from every stand to every
  runway end, with the runways the route crosses). The download is resumable; `--offline` rebuilds the
  tables after a rule change; `--check` prints the match rate per airport and split, the unmatched
  stands carrying the most departures, the runway designators without a threshold, and the shape of
  the routed distances, ending in an ok/weak verdict per airport.
  - `group:geometry` holds three blocks so each idea is measured separately: `stand_geometry`
    (metres east/north of the reference point, apron zone, match flag and rule), `taxi_distance` (the
    straight-line heuristic to the departure threshold, plus the stand's bearing relative to the
    runway heading) and `taxi_route` (routed metres, route/straight-line ratio, runway crossings).
    NaN means "unknown" and the flag columns say why. Configs: `exp008_geometry.yaml`,
    `exp009_geometry_per_airport.yaml`, `exp010_all.yaml`.
  - Stand IDs are matched exact-first, then normalised (case, punctuation, leading zeros), then
    without the position suffix ("A21" against "A21L"/"A21R", resolved to the one nearest the apron
    centroid and flagged), then against gate refs. Routing runs `scipy.sparse.csgraph.dijkstra` on
    the OSM way graph with runway edges penalised x3, and reports true metres.
  - `stand_match_report.csv` next to the tables lists every stand ID in the movement data with its
    departure counts, match rule, OSM object and coordinates; unmatched stands have `match_source`
    `none`. To fix or add stands by hand, put a CSV with `apt, stand_mvt, lat, lon` (optional `ref`,
    `note`) somewhere in the repo, name it as `geometry.stands_file` in the config (or pass
    `fetch-geometry --stands-file PATH`) and rebuild with `fetch-geometry --offline`. A row with
    coordinates becomes a stand of its own, matched to that ID and routed like any other; a row with
    both coordinates empty marks the stand unknown, which drops a wrong OSM match.
  - Attribution: map data (c) OpenStreetMap contributors, ODbL 1.0, via the Overpass API; runway and
    airport data from OurAirports (public domain).
- **Data cleaning.** The data is uncleaned operational data, and RMSE is a tail metric: on the
  Jan+Jul 2025 holdout the worst 10 rows carry 59 % of the squared error, the worst 0.1 % carry
  82 %, and Rome alone 73 %. Most of that is one segment: LIRF departures with no aircraft type
  (1,487 training rows, 29.5 % of them with taxi-out above two hours). The 2026 ranking set holds
  383 of their twins, and they are scored, so rows can leave the **fit** but never the **score**.
  `data.clean` (see Configuration) flags training rows by named rules and drops them from every
  fold's fit; validation keeps every row, so `rmse` stays honest, and `rmse_clean` shows the same
  predictions on the unflagged rows. `prc audit` prints what the data contains per airport, what
  a config's rules would flag, and, with `--run`, where a finished run's squared error lives and
  what its RMSE is without the flagged rows, so any past run can be scored under new rules without
  retraining. Two things the audit makes visible: helicopters at LSZH and EHAM taxi for seconds,
  sometimes negative seconds, and those rows are real (the minimum rule exempts runway `H`/`HEL`);
  and predicting a constant for the Rome segment does not beat the model, because the segment is
  mostly noise. `rmse` decides what is submitted; `rmse_clean` compares only across runs with the
  same rules. Configs: `exp011_clean_fit.yaml` (impossible rows only), `exp012_clean_rome.yaml`
  (also the Rome segment, expected to worsen `rmse` while improving `rmse_clean`). Measured:
  both do exactly that, on the log1p base (exp011 558 / clean 244, exp012 567 / clean 240 against
  exp002's 536) and on the raw base (exp016 549 / clean 237 against exp005's 430; scored on the
  same unflagged rows, exp005 gets 258). Dropping the tail from the fit improves the bulk by
  about 20 s and costs about 115 s on the score that counts. The day-offset rule on its own is
  different: it flags the **wrong-date rows**, 15 departures a year whose block time carries the
  previous day's date, so the target is 86,400 s plus an ordinary taxi (13 at LIRF, 1 LFPG,
  1 LSZH). Seven of them fall in the Jan+Jul holdout and carry 37 % of its squared error, which
  is why every run also reports `rmse_ex_dayoffset`. Twelve sit in the untyped Rome segment,
  where a schedule delta above 8 h means a stale or a wrong-date block and never a genuine taxi,
  so the regime mixture (`model.mixture`, `day_offset: true`) hedges them; the Paris and Zürich
  ones are typed flights with half-hour schedule deltas and nothing announces them. Dropping
  them from the fit cost `exp017` 15 s (their 2026 twins are scored); leaving them in a
  regressor's fit as 87,000 s "taxi times" is worse (the first mixture version), so the mixture
  labels them and its regressor skips them (`regressor_max_target_sec`). The Zürich one, at
  87,341 s, had been stopping Zürich's model at iteration 31–49 with the validation curve
  rising; with it out of the regressor's fit LSZH goes 234 → 209.
- **Stale block times (the Rome tail).** The multi-hour "taxi times" are not taxi times. Of the
  584 training departures above two hours, 438 have a target equal to takeoff minus
  `SCHED_TIME_UTC_mvt` to within two minutes (507 within thirty): the recorded block time is the
  scheduled off-block time, and the target is a departure delay, up to 131,167 s. The regime
  lives in one segment, departures with no aircraft type and no NM join (1,487 training rows,
  all but 20 at LIRF; 383 twins in 2026, 181 of them with a delta above two hours), and about
  half of that segment is stale while the other half taxis normally after a delay of hours.
  The model had the delta as a feature and could not use it: `nm_deltas.sched_delta` is NaN
  beyond a day, and CatBoost puts every delta above two hours into one quantisation bin. The
  `stale` group fixes both: `stale_schedule` provides the unclipped delta and, on the suspect
  segment with a non-negative delta, `stale_base`, which `exp020` uses as `target.residual_of`,
  so the exact delta is added back outside the trees and a stale row needs a residual of zero;
  `stale_target_stats` adds the fold's stale rate per carrier, destination and both, which is
  what separates the halves (some carriers are stale in every row, others in none). Measured on
  LIRF alone: 1,101 → 779; on the whole holdout 411.2 → 349.2, with the segment still 11 % of
  the squared error because the halves are not fully separable. Two consequences: the
  day-offset cleaning rule must stay off (those rows are stale blocks the model now predicts;
  dropping them cost 15 s in `exp017`), and `prc submit` clips at 200,000 s, not 20,000, so
  the deliberate multi-hour predictions survive.
- **The regime mixture** (`model.mixture`, `prc/models/mixture.py`) replaces the residual base
  with the explicit MSE-optimal answer: a CatBoost classifier gives each candidate row the
  probability of each regime (taxi, stale schedule, and with `day_offset` a block time on the
  wrong date, i.e. a day plus a taxi), and the prediction is the probability-weighted mixture
  of the candidate deltas and the regressor's taxi time. The regressor is fitted without the
  rows labelled stale or wrong-date, which matters more than it sounds: left in as 87,000 s
  "taxi times" they pulled every neighbouring prediction up by hours. `exp023` runs it on the
  untyped segment, `exp023b` on every departure with a delta of two hours or more (typed
  stale rows at London, Paris and Rome included), and `exp024_final_mixture.yaml` was the
  submission config of mid-September (v4, 16 Sep, board 329.0; exp028 replaced it on 18 Sep); every base
  member of the final submission is a regime mixture (docs/submission_manifest.md). Every run now also
  reports `rmse_ex_dayoffset`, the holdout without the seven wrong-date rows that carry over a third of its
  squared error.
- **Fitting target.** The single largest effect measured in this harness: fitting raw seconds
  without clipping (`exp005`, RMSE 430 ± 2 over three seeds) instead of `log1p` on a
  `[60, 7200]` clip (`exp002`, 536 ± 1). Isolated, the clip costs about 40 s and the transform
  about 60 s: both hide the tail from the fit, and the tail is what RMSE scores. Per-airport
  models on top (`exp015`) give 411. Every feature comparison has to be re-measured on the raw
  base; there, the geometry and weather groups are within noise (`exp013`, `exp014`).
- **Submission runs and the refit.** The fold model that scores the holdout has never seen
  January or July 2025, the seasonal twins of the test months. `model.refit_on_all: true`
  (`exp017_final_raw_per_airport.yaml`) keeps the folds for the score, then fits once more on
  all twelve months and predicts 2026 with that model. There is no holdout left to stop on, so
  the refit runs the folds' best iteration scaled by the row ratio (about ×1.2 here), a
  standard heuristic rather than a measurement; the log's comparison against the fold mean is
  the sanity check. `prc submit` reads `test_predictions.parquet`, which is the refit's; the
  fold mean stays in `test_predictions_folds.parquet` if it is ever needed instead.
- **Disrupted days.** Departure block times are blanked in 2026, but every arrival's taxi-in time
  is there, and on a snow day arrivals taxi in minutes longer than their stand/runway pairing
  normally takes. `prc/disruption.py` calls an airport-day *disrupted* when its arrivals' mean
  taxi-in anomaly (against the pairing's median, `prc/features/_arrlib.py`) exceeds 150 s, and
  every run reports `rmse_disrupted` / `n_disrupted_val` on the departures of those days next to
  `rmse` and `rmse_ex_dayoffset` (`metrics.json`, `index.csv`, the leaderboard, the queue summary,
  the OOF frame's `disrupted` column). A report mask, never a rule. `prc audit` lists the days:
  34 in 2025, 14 in the 2026 ranking months (EHAM 5 Jan 2026 at +1,600 s, LFPG 7 Jan, LTFM
  18-19 Jan, EDDF 26 Jan). Jan+Jul 2025 holds almost none, so winter features are judged on
  `exp029_winter_check.yaml` (Feb + Dec 2025): exp028 scores 248 there overall and 789 on its
  disrupted days, with a bias of -160 s.
- **The context blocks (2026 sweep).** One block per idea, each in its own group so `group:turnaround`
  and friends keep meaning what exp028 trained on: `stand_occupancy` (the previous occupant's
  delay, wake and carrier, the stand's traffic, the *next* occupant), `flight_number_history`
  (fold-dependent long-taxi and stale rates, median and p90 per flight number) and
  `flight_number_context` (rows, months active, the day before, duplicates, schedule spread),
  `runway_sequence` (ordinal in the hour, preceding wake, mixed-mode arrival gaps, saturation,
  throughput against the runway's p99, minutes since the dominant runway changed),
  `taxi_window_counts` (traffic during the flight's own NM taxi window, queue overtakes),
  `pier_pushback` and `zone_load` (apron traffic forwards and by NM push-back time, by pier and by
  OSM zone), `arrival_conditions` (the live taxi-in anomaly of the arrivals that just landed),
  `traffic_deviation` (the day's movements against the weekday median: cancellations),
  `nm_cross_deltas` and `nm_hour_context` (NM's times against each other, the airport-hour's NM
  picture), `stale_day_context` (suspect records clustered by day), `key_target_stats`
  (fold-dependent medians per carrier, type and local hour, tail statistics per stand/runway),
  `wx_deice_load` (departures and arrivals under de-icing conditions), `pru_daily` (below).
  Measure them with `prc ablate --mode add-to-base --emit-configs --blocks ...` followed by
  `prc queue`; read `rmse_ex_dayoffset` and `rmse_disrupted` against the seed floor.
- **External data: Eurocontrol PRU.** `python -m prc fetch-pru` downloads four open CSV datasets
  from the Performance Review Unit (https://www.eurocontrol.int/performance/data/download/csv/, the URL
  `prc/pru/download.py` uses: taxi-out additional time,
  ATC and all pre-departure delays, airport arrival ATFM delays; 2025 and 2026, republished
  monthly, so `--force` before a submission run) into `data/additional_data/pru/raw/` and
  normalises them into `pru_daily.parquet` (per airport-day: delay minutes per movement, incl.
  the weather-attributed ATFM minutes) and `pru_monthly.parquet` (per airport-month: the
  indicator's counts and minutes, and the mean taxi-out they imply). `--check` reports which
  series exist for which airport (EGLL, EHAM and LTFM report no ATC pre-departure delay of the
  first kind). The `pru_daily` block (group `pru`) reads the daily table. The monthly table is
  **not** a feature: it is the answer's monthly mean, which an operational model would not have,
  so it serves `prc levelcheck --run <id>` only, which compares a run's predicted mean per
  airport-month with it (for 2025 the two agree within ~7 s; where the PRU's reference share fell
  against 2025 its mean is a floor, because the excluded flights are the long ones) and writes
  `level_check.csv` into the run. Attribution: EUROCONTROL Performance Review Unit open data.
- **MLflow** would attach as an `MlflowRun(Run)` subclass in `prc/tracking.py`; nothing else changes,
  which is the point of routing every write through one object.
