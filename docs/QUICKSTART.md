# Quick start

From a clean clone (no `data/`, `runs/` or `cache/`; all three are gitignored) to the files of team adventurous-radio's
solution to the PRC Data Challenge 2026: the departure taxi-out time at ten European airports, scored by RMSE. For
depth: [README.md](../README.md) (the executive summary and the documentation map), [docs/report.md](report.md) (the
main report), [docs/DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) (the architecture and conventions; its Appendix A: the
harness's commands, configuration keys, tests), [docs/submission_manifest.md](submission_manifest.md) (v70's lineage)
and [docs/findings.md](findings.md) (every measurement, with run ids).

## 1. What you get

- **The post-operations analysis model** (the challenge model in the code, the configurations and the supporting
  documents), the only scored file. It estimates a flight's taxi-out after the flight from what was recorded about it,
  the use that the challenge's rationale names (docs/report.md Part 1): a post-operations model, not a planning
  model. v70's recipe, board RMSE **220.34** on January + July 2026 (50 trained runs, an ADS-B second stage,
  post-processing). The final set it is applied to covers **January, February, June and July 2026**: 1,341,197
  movements, 670,790 departures to predict (findings 10.33). That file is the final submission, which the owner
  uploaded once, on 14 Oct, as `adventurous-radio_final.parquet` (section 7).
- **The operational model**, the owner's deliverable of 11 Oct (findings 10.35), not scored: the planning model. It
  predicts the taxi-out before the aircraft moves, from data available at the moment: 24 h and 3 h before the
  scheduled off-block, and 15 min before the last filed off-block (H1). Each moment has one global CatBoost for the
  mean; at H1 the file adds a median (a second model, fitted on the absolute error) and an 80 % band. Since 13 Oct the
  H1 model is F1 (findings 10.37): the H1 model of 11 Oct plus the other flights' plans in the 15, 30 and 60 min after
  the moment (O*: the other departures' last filed off-blocks and the arrivals at the own pier, read as known at the
  moment though they can still change; O: the arrivals scheduled at the airport). F1 reads no tier F input.
  Beside the deliverable, never instead of it, come two tier F files (planned in the concept, realised in the challenge
  data): the concept member h1f (H1 + planning data, findings 10.36), which after the moment also reads the other
  flights' realised arrivals, and F5 (h1f5, the owner's request of 13 Oct, findings 10.37), the full concept. Among
  the operational models F5 reads the most context, but it still lacks the post-operations model's decisive inputs: the
  flight's own takeoff, its own ADS-B track and the NM join window. Both files are upper bounds for perfect plans, not
  available as computed. F5's realised departure side also carries traces of the flight's own slot, so its gain is an
  upper bound even for perfect plans.
  The model card is [docs/operational_model.md](operational_model.md). Every input, with its source,
  availability and rationale, is in [docs/operational_features.md](operational_features.md). The read-out is
  [docs/operational_ladder.md](operational_ladder.md). OP MAE on the Jan + Jul 2025 holdout (NM-matched
  departures below one hour):
  - H1 (F1): 184.3 s for the mean of three seeds, 178.8 s for the median (50.9 % within ±2 min); the H1 model until
    13 Oct: 186.7 s and 181.1 s;
  - 3 h: 214.5 s; 24 h: 218.2 s, both the mean of two seeds;
  - the tier F files, upper bounds, each the mean of two seeds: h1f 183.6 s, F5 177.4 s;
  - for scale, the stand × runway table: 207.4 s.
- Every run lands in `runs/<run id>/` (resolved config, metrics, holdout and ranking predictions, models;
  `python -m prc leaderboard` compares them), every submission file in `submissions/<run id>__submitting.parquet`.
- **Paths**: A, one post-operations model (about 2 h of GPU); B, the operational model (GPU: 146 min for H1's three
  refitted seeds, 35 min for each long member's two seeds, 17-46 min for each of H1's five month pairs and 41 min for
  its forward run, plus the superseded H1's comparator runs: 106 min for its three refits, about 90 min for its five
  pairs, and its forward run; CPU: 2.7 h of folds and 3.5 h of refit for the H1 median); C, v70 from scratch (about
  69 h of GPU); D, the final files from trained runs (about 4 h of CPU after the GPU refits).

## 2. Install and a five-minute check

Development and every recorded run: Windows 11, PowerShell, Python 3.13.13, CatBoost 1.2.10 on an RTX 4070 Laptop GPU
(8 GB). Paths C and D, and path B's optional R6, also need Git for Windows (Git Bash) for the `lab/*.sh` scripts.
The package sits at the repository root, so `python -m prc ...` runs from there without an install.

```powershell
git clone https://github.com/wurzelDeveloper/prc2026-taxiout-prediction.git
cd prc2026-taxiout-prediction
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1           # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

bash (Linux, macOS; in Git Bash on Windows activate with `source .venv/Scripts/activate`):

```bash
python3.13 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
```

- `requirements.txt` pins nothing. The versions of the recorded runs (each run's `env.json`; scipy, which `env.json`
  does not record, as in the development venv):
  `python -m pip install catboost==1.2.10 numpy==2.5.3 pandas==3.0.5 pyarrow==25.0.1 scikit-learn==1.9.1 scipy==1.18.1`.
- The development venv was made with uv (`uv venv --python 3.13 .venv`); a uv venv has no pip, use `uv pip install`.
- CatBoost trains on the GPU with the NVIDIA driver alone (the development machine has no CUDA toolkit). Without an
  NVIDIA GPU see "GPU or CPU" in section 8.

**Five-minute check** (no data, no network):

```powershell
python -m prc --help    # 18 subcommands
python -m pytest -q tests/test_pipeline.py tests/test_congestion_windows.py tests/test_feature_cache.py tests/test_cleaning.py tests/test_weather_parse.py tests/test_weather_taf.py tests/test_weather_windows.py tests/test_geometry_parse.py tests/test_geometry_routing.py tests/test_geometry_manual.py
```

- In bash, `tests/test_weather_*.py tests/test_geometry_*.py` can stand for the six weather and geometry files
  (PowerShell does not expand `*` for native commands).
- Next, the final-set path on a synthetic data tree (tiny CPU fits in a temporary directory):
  `python -m pytest -q tests/test_predict.py tests/test_frame_shift.py tests/test_model_persist.py tests/test_prepare_rows.py tests/test_final_inputs.py tests/test_run_index.py`.
  The whole suite is `python -m pytest -q`; files that need the data skip.
- Run `python -m pytest` from the repository root: bare `pytest` leaves the root off `sys.path`.

## 3. Data

**The organisers' files.** Put the twelve monthly training parquets (`training_2025-01-01_2025-02-01.parquet` to
`training_2025-12-01_2026-01-01.parquet`), `ranking.parquet` (January + July 2026) and `submitting.parquet` into
`data/competition_files/`. The organisers distribute them through their OpenSky MinIO buckets (see the challenge's [data page](https://prc-data-challenge-2026.netlify.app/data.html)). Then:

```powershell
python -m prc prepare      # ~11 s: data/interim/; asserts 4,167,797 training and 689,534 ranking movements
```

**The final release.** `final_ranking.parquet` (sha1 `7a06db8582f12186b832aecc8ef4e76bf11dfcb9`) and
`final_submitting.parquet` (sha1 `4006396e963f12ae93198b5e8dd93478adb35023`) go into `data_release/2026-10-09/`
(gitignored; the folder `lab/final_run.sh` reads). They come from the organisers' final-release zip of 8 Oct 2026,
distributed through the [PRC Data Challenge 2026](https://prc-data-challenge-2026.netlify.app/) (there is no direct
link to the files); the zip also repeats `ranking.parquet` and `submitting.parquet`, byte-identical to the
delivered ones. Never copy them over `data/competition_files/`: the trained runs, their feature cache (`data.version`
v1) and the checks of `prc predict` belong to the delivered frame.
`lab/final_inputs.py` installs the release into a data tree of its own (paths A, B and D).

**Downloads**, in this order. Each command resumes and writes under `data/additional_data/<source>/`; sources,
licences and attributions are in [docs/data_sources.md](data_sources.md). Times as recorded on the
development machine:

| # | command | after | recorded | needed by |
|---|---|---|---|---|
| 1 | `python -m prc fetch-weather --only metar` | `prepare` | 14 min (IEM, 180 station-months) | A, B, C, D |
| 2 | `python -m prc fetch-weather --only taf` | `prepare` | 2 h 14 min (Ogimet, 20 s per request; no open licence) | B, D's operational files |
| 3 | `python -m prc fetch-geometry` | `prepare` | 12 min (OpenStreetMap Overpass, OurAirports) | A, B, C, D |
| 4 | `python -m prc fetch-era5` | step 3 | minutes (10 requests to Open-Meteo) | A, C, D; B on a new set (`validate` checks it) |
| 5 | `python -m prc fetch-pru` | `prepare` | 18 s | the final sets of A, B and D (the runbook and `lab/op_deliver.py` end with `prc levelcheck`; fetch it before `lab/final_inputs.py init`, which copies it into the final tree) |
| 6 | raw METAR and TAF of March to May 2026 (below) | steps 1, 2 | not recorded | the final sets of A, B and D (`validate` refuses a missing METAR month) |
| 7 | ADS-B (below) | step 3 | about 13 MB/s: 9.5 h for 124 days with 3 workers | C, D; B's tier F files h1f and h1f5 (all of 2025 and January, February, June and July 2026, before the snapshot s1 is built) |

Step 6. `fetch-weather` takes its months from the movement frame, which does not reach March or May 2026; the final
set's look-back and look-ahead read them, and `validate` stops on a missing METAR month. This writes raw files only:

```powershell
python -c "from prc import data; from prc.weather.download import fetch_chunks; st = data.airports(data.load_movements(columns=['apt'])); [fetch_chunks(k, st, ['2026-03', '2026-04', '2026-05']) for k in ('metar', 'taf')]"
```

Step 7 (adsb.lol, ODbL 1.0). v70's stage was built on exactly 412 days: the 350 published days of 2025 (no release
for 28 May to 10 June, 31 December unreadable) and the 62 of January + July 2026, with adsb.lol's staging release
merged on the default days. The final set also needs every February and June 2026 day; with path C, fetch those after
its ADS-B tables (step C4), as on the development machine. Path B's tier F files (rule T's member h1f and F5) read the
surface snapshot s1 on the final set; the H1 deliverable reads no ADS-B. Build and freeze s1 (section 5) only after
all of 2025 and January, February, June and July 2026 are downloaded, else the missing days' rows get NaN ADS-B
columns and nothing reports it.

```powershell
python -m prc fetch-adsb --workers 3                   # default days: Jan + Jul 2025 and 2026 (124)
$d = 0..364 | ForEach-Object { ([datetime]'2025-01-01').AddDays($_).ToString('yyyy-MM-dd') }
python -m prc fetch-adsb --workers 3 --days $d         # the rest of 2025; existing days are skipped, unpublished ones log FAILED
python -m prc fetch-adsb --kind staging --workers 3    # staging, default days (~400 GB streamed to keep ~2.3 GB)
python lab/adsb_merge_staging.py                       # union into points/; the prod originals go to points_prod/
# the final set's new months, prod only (with path C: after step C4; for path B's s1: before it is built)
$d = (1..28 | ForEach-Object { "2026-02-{0:d2}" -f $_ }) + (1..30 | ForEach-Object { "2026-06-{0:d2}" -f $_ })
python -m prc fetch-adsb --workers 3 --days $d
```

In bash the day lists are
`$(python -c "import pandas as pd; print(*pd.date_range('2025-01-01', '2025-12-31').strftime('%Y-%m-%d'))")`
and `$(seq -f 2026-02-%02g 1 28) $(seq -f 2026-06-%02g 1 30)`. `data/` takes about 16 GB on the development machine,
almost all of it ADS-B points.

Once runs are trained, never rebuild a table they read, online or `--offline`: the feature blocks hash their input
tables, and `prc predict` refuses a run whose tables changed as feature drift. Every run reads `metar.parquet` and the
geometry tables, paths A and C also `era5_hourly.parquet`, path B the TAF tables. In particular, never rebuild METAR
(`fetch-weather` without `--only taf`) after step 6: the raw March to May 2026 files would enter `metar.parquet`, and
`prc predict` would refuse every trained run. Adding a table no trained run reads is safe: `fetch-weather --only taf`
(it writes the TAF tables only) after path A, `fetch-era5` after path B, and `fetch-pru` at any time.

## 4. Path A: one post-operations model to a submission file

`exp066`, one of v70's base members: a regime mixture (a classifier weighs a global taxi-time regressor against the
stale-schedule and wrong-date readings of a record; docs/report.md section 1.2), refitted on all twelve months,
then clipped to the NM join window and stacked. Needs `prepare` and downloads 1, 3 and 4.

```powershell
python -m prc train -c exp066_lean2_global_window_long.yaml -o evaluation.permutation.enabled=false   # GPU 2 h 10 min with the refit; ends "artifacts: runs\<id>"
python -m prc blend --runs <train id> --lobt-bound --name exp066_lobt     # the join-window clip (findings 10.3)
python lab/final_stack.py <blend id> --inwindow 0.5 --write exp066_stack  # prints the holdout RMSE, then "wrote <id>"
python -m prc submit --run <stack id>                                     # submissions\<stack id>__submitting.parquet, 344,841 rows
```

- The override is the recorded run's (`6d1b7355`, the manifest's): without it the run adds about 15 min of
  permutation importance (a diagnostic that does not change the model) and gets another configuration hash.
- Expected level, Jan + Jul 2025 holdout before the clip and the stack (recorded run): RMSE 313.99, 259.29 without
  the seven wrong-date rows, 214.79 on targets below one hour. exp066 alone was never uploaded; on the board the
  clipped 17-run blend scored 263.17 (v31), with the stack 260.72 (v32), v70 220.34.
- Faster: `exp056_lean2_window.yaml` with the same override, 58 min (315.61 / 260.45 / 214.38). Without a GPU add
  `-o model.params.task_type=CPU model.mixture.classifier_params.task_type=CPU` (the CPU was 4.5-6x slower on exp002).
- The final set from this one model (needs downloads 4 to 6 and the release; writes 670,790 rows):
  `python lab/final_inputs.py runbook --tree "$PWD\data_final" --runs <train id> --months 2026-01 2026-02 2026-06 2026-07 --out runs\final_single.ps1`,
  then `powershell -ExecutionPolicy Bypass -File runs\final_single.ps1 -Ranking "$PWD\data_release\2026-10-09\final_ranking.parquet" -Template "$PWD\data_release\2026-10-09\final_submitting.parquet"`.

## 5. Path B: the operational model (the deliverable of findings 10.35 and 10.37)

One model per decision moment, every input available at that moment (`docs/operational_model.md`; every input with
its source, availability and rationale: `docs/operational_features.md`). The exceptions are the two tier F files
(end of this section): upper bounds, beside the deliverable, never instead of it.

| moment | configuration | seeds, refitted |
|---|---|---|
| 24 h before the schedule | `op_hl24_r3_rwyblind` (rule R chose r3; findings 10.35) | 2 |
| 3 h before the schedule | `op_hl3_r4_rwyblind` (rule R chose r4; findings 10.35) | 2 |
| H1, 15 min before the filed off-block (headline) | `op_h1_f1_plan` (F1, adopted 13 Oct; findings 10.37) | 3 |
| H0, at the off-block (2025 comparison only) | `op_h0_r4_nodoy` (the inputs of the H1 model until 13 Oct) | holdout only |

F1 is `op_h1_r4_nodoy`, the H1 deliverable until 13 Oct (rule L kept its learning rate, 0.08), plus the block
`op_h1_fut_plan`: the other flights' plans in the 15, 30 and 60 min after the anchor, 15 columns. Twelve are O*: the
other departures' last filed off-blocks (else their schedule) per airport, own runway and own pier, and the arrivals
scheduled at the own pier. Three are O: the arrivals scheduled at the airport. Every plan is read as known at the
anchor, and the runways and the other flights' piers are the recorded ones (O*). F1 has no tier F input and reads no
realised movement of another flight after the anchor. The superseded H1 stays as the comparator (below, before rule T).

Needs `prepare` and downloads 1, 2, 3 and 5 (no ERA5 for the training; PRU because `lab/op_deliver.py` ends with
`prc levelcheck`: fetch it before `lab/final_inputs.py init`, which copies it into the final tree). The two tier F
files also need download 7: the ADS-B points of all of 2025 and of January, February, June and July 2026. The
deliverables refit on all twelve months after the holdout fit, so never pass `--no-test` to them.

```powershell
python -m pytest -q tests/test_operational.py tests/test_operational_long.py tests/test_operational_future.py tests/test_feature_catalogue.py tests/test_op_adopt.py
python -m prc train -c op_h1_f1_plan.yaml --seeds 3 --keep-test           # H1 (F1): seeds 42-44, each with its refit
python -m prc train -c op_hl3_r4_rwyblind.yaml --seeds 2 --keep-test      # 3 h (rule R chose r4)
python -m prc train -c op_hl24_r3_rwyblind.yaml --seeds 2 --keep-test     # 24 h (rule R chose r3)
python -m prc train -c op_h0_r4_nodoy.yaml --no-test                     # optional: the H1 inputs until 13 Oct, at the push (2025 only)
```

- `tests/test_operational_future.py` holds the gates of F1's plan block (and of the tier F blocks);
  `tests/test_op_adopt.py` those of `lab/op_adopt.py`, the adoption's checks. Neither needs data.
- `op_h1_f1_plan.yaml` sets the refit flag itself, so its three runs refit (config hash 40ac5f36, as on the recorded
  runs).

The 80 % band comes from tables fitted on the five 2025 month pairs, never on January or July (rule C). The same
tables give a calibrated median, which ships at no moment: at H1 a second model replaced it (rule M, below), and at 3 h
and 24 h rule C failed. For H1 the pairs are configs (`op_h1_f1_plan_febdec.yaml` and so on), and so is its forward run
(`op_h1_f1_plan_fwd.yaml`: fit January to October, validate November and December; rule F's tripwire). For a long
member the pairs are overrides of its configuration, for example `-o "validation.holdout_months=[2025-02,2025-12]"
model.refit_on_all=false name=op_hl3_r4_rwyblind_febdec`. They serve only to re-check rule C at 3 h and 24 h
(`fit --out runs/op_calib/hl3.parquet` and `check --member hl3` as below, likewise hl24; 40-50 min of GPU per member
for the five pairs): those files hold the mean either way.

If `runs/op_decisions.json` and `submissions/operational/manifest.json` already hold the superseded H1's records
(`C_h1`, `M`, the manifest's `h1` and `h1_calibrated_median`), as on the development machine on 13 Oct, run
`python lab/op_adopt.py archive` before F1's check and delivery. It keeps them as `C_h1_r4_nodoy`, `M_r4_nodoy`,
`h1_r4_nodoy` and `h1_r4_nodoy_calibrated_median`, so that F1's records overwrite nothing; where one of them is missing
or another model's, it refuses and writes nothing. The files themselves stay in `submissions/operational/`.

```powershell
python -m prc train -c op_h1_f1_plan_febdec.yaml --no-test                # and _marnov, _aproct, _maysep, _junaug
python -m prc train -c op_h1_f1_plan_fwd.yaml --no-test                   # forward: fit Jan-Oct, validate Nov+Dec
python lab/op_adopt.py checks                                             # F1 against the superseded H1 (its runs: below), and rule F
python lab/op_calibrate.py fit --pairs op_h1_f1_plan_febdec op_h1_f1_plan_marnov op_h1_f1_plan_aproct op_h1_f1_plan_maysep op_h1_f1_plan_junaug --out runs/op_calib/h1_f1_pairs_lr08.parquet
python lab/op_calibrate.py check --tables runs/op_calib/h1_f1_pairs_lr08.parquet --runs <the three F1 refits> --member h1
```

- `python lab/op_rules.py refits op_h1_f1_plan` prints the three refits: the newest at the configuration's current
  hash, the ones `lab/op_deliver.py --config op_h1_f1_plan.yaml --n 3` delivers. Rule M (below) needs `C_h1` to be the
  check of exactly these runs, with tables fitted on F1's five pairs.
- `checks` compares F1's month pairs and forward run with the superseded H1's (`op_h1_r4_nodoy_<pair>`,
  `op_h1_r4_nodoy_fwd`; below, before rule T): without those runs its generalisation criteria fail and name the
  missing runs. Rule F reads F1's runs only. It records `A_F1` in `runs/op_decisions.json` and writes
  `runs/op_adopt_f1.md`. On the recorded runs F1 was better in 10 of 10 validation months and forward by 4.79 s
  (173.36 s against 178.15 s), and rule F did not trip.
- Rule C passed for F1 on the recorded runs: the calibrated median scores 180.79 s and 49.70 % within ±2 min against
  the mean's 184.29 s, with no airport worse and a P10-P90 coverage of 77.7 %.

The H1 median is a model of its own, fitted on the absolute error with exact leaves (rule M). CatBoost's GPU fails with
exact leaves, so it trains on the CPU; on the recorded run, beside the GPU queue, the folds took 2.7 h (best round
5,576) and the refit 3.5 h (6,679 rounds):

```powershell
python -m prc train -c op_h1_f1_plan_mae_exact_cpu.yaml                   # H1 median member, seed 42, with its refit
python lab/op_adopt.py rule-m                                             # rule M against F1's calibrated median (C_h1)
```

- Rule M (findings 10.35, re-run for F1 in 10.37): the member replaces the calibrated median only if its holdout beats
  it by at least 1 s of OP MAE and 1 point within ±2 min. On the recorded runs: 178.82 s and 50.88 % against 180.79 s
  and 49.70 %, so it passed.
- `rule-m` records the decision as `M` in `runs/op_decisions.json`. It refuses, writing nothing, until the superseded
  H1's rule M record is archived as `M_r4_nodoy` (`archive`, above), so it stops on a checkout without that record:
  no script writes it (it was written by hand on the development machine, findings 10.35). Then compare by hand:
  `python lab/op_rules.py metrics op_h1_f1_plan_mae_exact_cpu` prints the member's OP metrics, and the output of
  `op_calibrate.py check` holds the calibrated median's (`median`). `lab/op_deliver.py --median-config` does not read
  the record.

Read the holdout (Jan + Jul 2025; OP = NM-matched departures with a target below one hour, 338,277 rows):

```powershell
python lab/op_rules.py metrics op_h1_f1_plan op_hl3_r4_rwyblind op_hl24_r3_rwyblind    # OP MAE, RMSE, +-2 min, per month and airport
Set-Content runs\op_decision_R_hl3.cfg op_hl3_r4_rwyblind.yaml; Set-Content runs\op_decision_R_hl24.cfg op_hl24_r3_rwyblind.yaml
python lab/op_ladder_report.py --out runs/op_ladder_check    # the whole read-out; without --out it overwrites docs/operational_ladder.*
python lab/feature_catalogue.py --hl24 op_hl24_r3_rwyblind.yaml --gain h1=<an op_h1_f1_plan refit> hl3=<3 h seed 43> hl24=<24 h seed 43> --perm h1=<an op_h1_r4 run> --out runs/op_features_check
```

- Given configuration names, `metrics` prints the newest run of each: on the recorded runs 184.8 s (H1, F1's seed 44),
  215.0 s (3 h, seed 43) and 218.5 s (24 h, seed 43); pass run ids for the other seeds. The seeds span 184.8-185.1 s,
  215.0-215.1 s and 218.5-218.7 s; the seed means are 184.3, 214.5 and 218.2 s. The H1 median is 178.8 s
  (`op_h1_f1_plan_mae_exact_cpu`, clamped into the band; the calibrated median it replaced: 180.8 s). The superseded
  H1's seeds spanned 186.7-187.6 s, their mean 186.7 s and its median 181.1 s.
- `lab/op_ladder_report.py` reads F1 as the H1 deliverable (row D-H1). Where the superseded H1's three refits exist,
  it keeps them as row D-H1-R4 with the adoption step: −2.39 s of OP MAE on the recorded runs, day-block interval
  [−2.73, −2.06].
- `lab/op_ladder_report.py` takes the 3 h and 24 h configurations from rule R's files `runs/op_decision_R_hl3.cfg` and
  `runs/op_decision_R_hl24.cfg`. Rule R writes them (`python lab/op_rules.py R --lead hl3|hl24`, run after the r3 and
  r4 runway-blind runs and their `_fwd` twins and before the refits); the `Set-Content` line writes the recorded
  choices instead.
- `lab/feature_catalogue.py` takes F1 as the H1 member, and rule R's choice from `runs/op_decisions.json`, hence
  `--hl24`. `--gain h1=` must be a run of `op_h1_f1_plan`: the generator stops on a run of another configuration. The
  tracked catalogue was written with the newest refit of each deliverable for `--gain` (at H1 F1's seed 44,
  `2026-10-13T07-54-55__op_h1_f1_plan__40ac5f36__s44`), and for `--perm` with v68's run
  `2026-10-08T02-37-29__op_h1_r4__3fcdc779__s42`: the deliverables' configurations measure no block permutation,
  `op_h1_r4` does (`python -m prc train -c op_h1_r4.yaml`, about 40 min of GPU with its refit). Without `--out` the
  script overwrites `docs/operational_features.md` and `.csv`.

Predict the final set (January, February, June and July 2026). The release goes into a data tree of its own; path D's
runbook builds it (`data_final/`). Then `lab/op_deliver.py` does the rest for one member:
1. it applies each refit to the tree (`prc predict`, no retraining);
2. it averages the seeds (`prc blend`, never `--lobt-bound`);
3. it adds the calibrated median and band if rule C passed, and with `--median-config` replaces the median by the
   median member's prediction, clamped into the band (rule M);
4. it writes `submissions/operational/<member>__<run>.parquet` (`__mae.parquet` with rule M's median) and
   `manifest.json` (sha1 of every file); where it applies tables, the folder also gets its calibration intermediate
   `<member>__<run>__calibrated.parquet`, which is not a deliverable; a call with `--median-config` keeps the
   member's earlier record, the calibrated median's file, as `<member>_calibrated_median`;
5. it runs the PRU level check (download 5).

It never calls `prc submit`, so `submissions/submitting.parquet`, the scored challenge file, stays as it is.
`submissions/` is local and gitignored; the tracked [`docs/operational_manifest.json`](operational_manifest.json)
records every delivered file with its full sha1.

```powershell
python lab/op_deliver.py --tree data_final --member h1 --config op_h1_f1_plan.yaml --n 3 --tables runs/op_calib/h1_f1_pairs_lr08.parquet --rule-c
python lab/op_deliver.py --tree data_final --member h1 --config op_h1_f1_plan.yaml --n 3 --tables runs/op_calib/h1_f1_pairs_lr08.parquet --rule-c --median-config op_h1_f1_plan_mae_exact_cpu.yaml
python lab/op_deliver.py --tree data_final --member hl3 --config op_hl3_r4_rwyblind.yaml --n 2
python lab/op_deliver.py --tree data_final --member hl24 --config op_hl24_r3_rwyblind.yaml --n 2
```

H1 takes two calls, as on 13 Oct (findings 10.37). The first writes the file with the calibrated median
(`h1__<blend>.parquet`). The second re-uses its predict runs and blend, predicts the median member, writes
`h1__<blend>__mae.parquet`, the H1 deliverable (the manifest's `h1`), and keeps the first file's record as
`h1_calibrated_median`; a single call with `--median-config` would leave no such record. On the recorded runs the H1
file is `h1__2026-10-13T13-54-07__final_op_h1_blend__aa1cd03b__mae.parquet` (sha1
4153b4ea3196de18211df8d6f492ffd0c0c7adb4, 670,790 rows; the median member's prediction clamped into the band on 0.79 %
of them). On the development machine the folder also keeps the superseded H1's files, under the manifest keys
`h1_r4_nodoy` and `h1_r4_nodoy_calibrated_median`.

The 3 h and 24 h lines take no tables: rule C failed at both leads (findings 10.35), so their files hold `MVT_ID_mvt`
and `y_mean` only, as the delivered ones do.

Without `data_final/`, build the tree first (downloads 4, 5 and 6 are needed too: `init` copies the PRU tables,
`validate` checks ERA5 and the raw METAR months):

```powershell
$Tree = "$PWD\data_final"                    # beside data\, never inside it
python lab/final_inputs.py init --tree $Tree --source "$PWD\data"     # copies 475 files, 403 MB
python lab/final_inputs.py validate --file data_release\2026-10-09\final_ranking.parquet --template data_release\2026-10-09\final_submitting.parquet --months 2026-01 2026-02 2026-06 2026-07 --baseline data\competition_files\ranking.parquet --tree $Tree --install --force
$env:PRC_EXPECTED_RANK_ROWS = python lab/final_inputs.py derive rank-rows --tree $Tree    # 1341197
$env:PRC_DATA_DIR = $Tree
python -m prc prepare --force
python -m prc fetch-weather --offline
python -m prc fetch-geometry --offline
Remove-Item Env:PRC_DATA_DIR, Env:PRC_EXPECTED_RANK_ROWS
```

The queue that ran most of path B on 11-13 October, rule T included, is `lab/op_queue.sh` (on the GPU; the superseded
H1's median member in two CPU steps). It reads one step per line (`ID | BUDGET_MIN | gate|run | COMMAND`) from
`runs/op_queue.txt`, a local file; this section lists the commands. F1's month pairs and forward run (steps `AF_*`)
and F5's refits, delivery, month pairs and forward run (`AF5_*`) went to the same queue on 13 Oct. F1's median member
did not: it ran on the CPU beside the queue. The rules that chose the learning rate (L), r3 or r4 (R), the median
output (C) and the forward tripwire (F) are fixed in findings 10.35 and computed by `lab/op_rules.py` and
`lab/op_calibrate.py`; findings 10.35 transcribes their decisions, findings 10.36 rule T's, and findings 10.37 those
of F1's adoption (the generalisation check, rule F, rule C and rule M, computed by `lab/op_adopt.py` and
`lab/op_calibrate.py`). The scripts write each decision into `runs/op_decisions.json`; this repository holds no
`runs/`, and the decisions recorded on the development machine are in findings 10.35-10.37 and
[docs/operational_model.md](operational_model.md). The superseded H1's rule M record was written outside these
scripts: `python lab/op_rules.py metrics op_h1_r4_nodoy_mae_exact_cpu` prints that candidate's OP metrics, to set
against the calibrated median of `op_calibrate.py check` and the bar of findings 10.35.

**The superseded H1 (the H1 deliverable from 11 to 13 Oct, findings 10.35).** `op_h1_r4_nodoy` is F1 without the
plan block. On the recorded runs the mean of its three seeds scored OP MAE 186.7 s on the holdout and its median
181.1 s (the member `op_h1_r4_nodoy_mae_exact_cpu`, clamped into the band; the calibrated median it replaced:
183.1 s). On the board the mean's January and July rows scored 560.98 (v75), F1's 559.77 (v82). Against it F1 gains
2.39 s of OP MAE on the holdout (day-block interval [−2.73, −2.06]). The owner adopted F1 on 13 Oct, and the
pre-registered checks that followed found F1 better in 10 of 10 validation months and forward by 4.79 s (findings
10.37). The superseded files stay in `submissions/operational/` on the development machine:
`h1__2026-10-11T13-55-49__final_op_h1_blend__58d3474d__mae.parquet` (sha1 1dca8c80175d..., the manifest's
`h1_r4_nodoy`) and its file before rule M, `h1__2026-10-11T13-55-49__final_op_h1_blend__58d3474d.parquet`
(46e4da46b0e1..., `h1_r4_nodoy_calibrated_median`). The superseded H1's runs remain the reference of
`lab/op_adopt.py checks` (the month pairs and the forward run), of the ladder's row D-H1-R4 (the three refits) and of
rule T (below):

```powershell
python -m prc train -c op_h1_r4_nodoy.yaml --seeds 3 --keep-test          # seeds 42-44, each with its refit (106 min of GPU)
python -m prc train -c op_h1_r4_nodoy_febdec.yaml --no-test               # and _marnov, _aproct, _maysep, _junaug
python -m prc train -c op_h1_r4_nodoy_fwd.yaml --no-test                  # forward: fit Jan-Oct, validate Nov+Dec
```

The superseded files were made by the same steps as F1's, with `op_h1_r4_nodoy` in place of `op_h1_f1_plan`: the
tables `runs/op_calib/h1_nodoy_pairs_lr08.parquet`, the median member `op_h1_r4_nodoy_mae_exact_cpu` (395 min of CPU
with its refit) and `lab/op_deliver.py --config op_h1_r4_nodoy.yaml`. Those steps write `C_h1` and the manifest's
`h1`, which F1's steps write too: on the development machine they ran on 11 and 12 Oct, before F1's, and `archive`
kept their records (above). Never run them after F1's check and delivery.

**Traffic after the anchor (rule T, findings 10.36).** Rule T's runs add to the superseded H1 the other flights' plans
after the anchor (O*: F1, the H1 deliverable since 13 Oct) and their realised movements after it (tier F: planned in
the concept, realised in the challenge data; a stand-in for the AMAN, DMAN and A-SMGCS plans the data lack). A tier F
model is an upper bound for perfect plans, not available as computed. The member that passed, `op_h1_f4_nodepthru`
(the plans and the realised arrivals only), is written as `submissions/operational/h1f__<run>.parquet` beside the
deliverable, never instead of it; findings 10.36 records the decision. On the recorded runs each fit took 15-35 min of
GPU and each refit 35-48 min, plus about 55 min of feature building on a cold cache.
- Rule T measures every run against the superseded H1 (above): stage 1 against its seed-42 run, stage 2 against its
  month pairs and forward run. F1 needs no run of its own: stage 1 reads the newest seed-42 run at a configuration's
  current hash, and the deliverable's seed-42 refit is one. A re-run therefore scores that refit (hash 40ac5f36), not
  the holdout-only run behind `T13`'s F1 row (hash 13468ab0, made before the configuration set the refit flag; 182.66 s
  on rule T's rows). On all OP rows the refit's holdout phase scores 184.89 s and that run 184.92 s, so a re-run's F1
  row can differ slightly from the record.
- The ADS-B blocks read the frozen surface snapshot s1 (built as for R6, below, once the ADS-B points of all of 2025
  and of January, February, June and July 2026 are downloaded; section 3, step 7).
- F5 also reads R6's block `op_h1_adsb`, so it needs the ADS-B gates of s1 that `lab/op_ladder_run.sh` runs before R6:
  `tests/test_adsb_surface.py`, `tests/test_operational_adsb.py` and `tests/test_op_adsb_audit.py`, the real slice
  (`OPLAD_REAL=1`, `-k real_slice`; it must pass, not skip) and `python lab/op_adsb_audit.py --snapshot s1`, which
  reads the challenge links of `lab/adsb_links.py` (below).

```powershell
python -m pytest -q tests/test_operational_future.py tests/test_op_rule_t.py   # the gates of the tier F blocks and of rule T, before any F run
foreach ($c in "op_h1_f2_rec","op_h1_f3_nm","op_h1_f4_adsb","op_h1_f4_shift7","op_h1_f4_norwy","op_h1_f4_nodepthru","op_h1_f5_full") { python -m prc train -c "$c.yaml" --no-test }   # op_h1_f4_norwy is informative: reported, not decided on
python lab/op_rule_t.py stage1        # criteria 1-3 and the member: runs/op_rule_t.md, runs/op_decisions.json (T13)
python -m prc train -c op_h1_f4_nodepthru.yaml -o validation.scheme=forward_holdout "validation.holdout_months=[2025-11,2025-12]" evaluation.permutation.enabled=false name=op_h1_f4_nodepthru_fwd --no-test
python -m prc train -c op_h1_f4_nodepthru.yaml -o "validation.holdout_months=[2025-02,2025-12]" evaluation.permutation.enabled=false name=op_h1_f4_nodepthru_febdec --no-test   # likewise marnov, aproct, maysep, junaug
python lab/op_rule_t.py stage2        # criteria 4-5 (T)
python -m prc train -c op_h1_f4_nodepthru.yaml -o model.refit_on_all=true evaluation.permutation.enabled=false --seeds 2 --keep-test
python lab/op_deliver.py --tree data_final --member h1f --runs <the two refitted runs: python lab/op_rule_t.py refits>
python -m prc train -c op_h1_f5_full.yaml -o model.refit_on_all=true evaluation.permutation.enabled=false --seeds 2 --keep-test   # F5 (below)
python lab/op_deliver.py --tree data_final --member h1f5 --config op_h1_f5_full.yaml --n 2 --any-hash
```

Run both stages as the block orders them, before the member's refits and before any other seed-42 run of a
configuration they name (the superseded H1's and F1's included, and the member's pair and forward runs): a later run of
the same name could replace the one scored. A re-run overwrites `T13` and `T` in `runs/op_decisions.json` and
`runs/op_rule_t.md`; the run ids in `T13`'s rows show which runs were scored.

**F5** (`op_h1_f5_full`, findings 10.37), the owner's request of 13 Oct, is written beside the deliverable as
`submissions/operational/h1f5__<run>.parquet`: tier F, the mean only. It is the full concept: F1 plus the other
flights' realised records, NM readings and ADS-B picture after the anchor, and the ADS-B picture before it. Among the
operational models it reads the most context, but it still lacks the post-operations model's decisive inputs: the
flight's own takeoff, its own ADS-B track and the NM join window. Its realised departure side carries traces of the
flight's own slot, which rule T could not credit, so its gain is an upper bound even for perfect plans.
- No configuration with `refit_on_all` resolves a tier F block (`tests/test_operational_future.py`), so its two seeds
  refit through the override, at another hash than the configuration's; `--any-hash` lets `lab/op_deliver.py` find
  them. On the recorded runs the refits took 192 min of GPU, shared with the CPU median member.
- The two seeds' mean scores OP MAE 177.43 s on the holdout: 6.86 s below F1, and 0.8 s below H0, which knows the
  flight's own off-block. The recorded file is `h1f5__2026-10-13T17-23-14__final_op_h1f5_blend__f7393913.parquet`
  (sha1 bac06dfdc0bf8399f17eee6baf9425edb3e6fceb, 670,790 rows).
- Its month pairs and forward split, overrides of its configuration like the member's lines above, follow for
  information only: the owner's decision stands either way.

**Earlier versions, kept for comparison.**
- `op_h1_r4_nodoy`, the H1 deliverable from 11 to 13 Oct (v75: 560.98), superseded by F1: the note before rule T.
- `op_h1_r4` (with the day of year, the v68 board run: 560.9; it neither hedges the stale and wrong-date records nor
  clips to the join window).
- The 3 h and 24 h rungs on the IOBT clock (10 Oct, findings 10.31). That clock assumed the filed plan known hours
  ahead, and was superseded on 11 Oct.

R6, the H1 rung with the public ADS-B surface picture of other aircraft, is not a deliverable. On the five 2025 month
pairs it changed the OP MAE by +0.05 s against R4 (findings 10.35). To reproduce it anyway, it needs the ADS-B points
of 2025 and of the months to predict, and the challenge links of those days, which the ADS-B gate's audit uses to find
each flight's own airframe (tier T, audit only). Then (Git Bash):

```bash
python -m prc adsb-surface --snapshot s1 --workers 6     # 22 min for 561 days on the development machine
python -m prc adsb-surface --snapshot s1 --check
python -m prc adsb-surface --snapshot s1 --support
python -m prc adsb-surface --snapshot s1 --freeze
python lab/adsb_links.py --workers 4                     # cache/adsb/links_<day>.parquet (path C's C4 skips the days that exist)
bash lab/op_ladder_run.sh "2026-10-12 06:00" op_h1_r6    # the leakage and ADS-B gates, then op_h1_r6 with its refit
```

The first argument is the deadline, a UTC time (`YYYY-MM-DD HH:MM`) a few hours ahead: after the gates (on a first
run also pytest files, two feature-cache warms and two smoke fits) a rung starts only if its budget (130 min for
`op_h1_r6`) ends before it, else `runs/op_ladder.log` says "skip".

## 6. Path C: the full submitted pipeline (v70) from scratch

v70 = 50 trained runs (the 27 base members of the blend `blend11_window_lobt`; 15 month-pair members that give the
ADS-B stage its 2025 base; three longer-trained base members, their three depth-9 retrains and two Istanbul
specialists), five ADS-B stage runs and post-processing on the CPU. The manifest lists every run with its
configuration, seed and weight.

- **Budget** (development machine, `wall_seconds` of each run): base members 22.7 h of GPU (26 of 27 recorded), month
  pairs 11.2 h, swap members and specialists 15.9 h, the four GPU stage members 2 to 8 h each while the CPU stage
  member (13.2 h of CPU) ran beside them; the ADS-B tables 2 h 10 min of CPU for 288 days. About 69 h of GPU in all,
  plus the ADS-B download (about 30 h for 412 days). A refitted base member takes 0.3 to 0.7 GB in `runs/` (the
  longer-trained and depth-9 members 0.8 to 2.2 GB); v70's 68 run directories take about 26 GB.
- **Reproducibility**: GPU steps come back within the seed spreads (base members 0.6-1.3 s, stage 0.06-0.11 s on the
  holdout without wrong-date rows), CPU steps exactly.
- **Run ids**: with the overrides below, today's configurations give the manifest's run name and config hash for all
  50 trained runs (checked with `prc.config.load_config`), so your ids differ from the manifest's in the timestamp
  only. Permutation importance is a diagnostic and does not change a model.

Git Bash from the repository root, venv active; `<x>` is your run id of x.

```bash
P=evaluation.permutation.enabled=false
# C1. the 27 base members (GPU)
for s in 42 43 44; do python -m prc train -c exp043_final_screen.yaml -o seed=$s $P; done
for s in 42 43 44 45 46; do python -m prc train -c exp050_lean.yaml -o seed=$s $P; python -m prc train -c exp051_lean2.yaml -o seed=$s $P; done
for c in exp042_final_live_state exp046_classifier_seeds5 exp047_month_weights exp057_lean2_window_anchored; do python -m prc train -c $c.yaml -o $P; done
python -m prc train -c exp037_final_era5_mixture.yaml
python -m prc train -c exp054_lean2_global.yaml
for s in 42 43; do for c in exp056_lean2_window exp059_lean2_window_lobt_anchor exp060_lean2_global_window exp061_lean2_window_b254_eham_anchor; do
  python -m prc train -c $c.yaml -o seed=$s $P; done; done
# C2. the base blend and its stack (CPU, minutes); blend17_wide: these 17 runs in this order, equal weights, no clip
python -m prc blend --name blend17_wide --runs <exp043 s42> <exp043 s43> <exp043 s44> <exp050 s42> ... <exp050 s46> \
  <exp051 s42> ... <exp051 s46> <exp037> <exp042> <exp046> <exp047>
python -m prc blend --name blend11_window_lobt --lobt-bound --runs <blend17_wide> <exp054> <exp056 s42> <exp056 s43> <exp057> \
  <exp059 s42> <exp059 s43> <exp060 s42> <exp060 s43> <exp061 s42> <exp061 s43> --weights 0.2 0.05 0.05 0.05 0.1 0.1 0.1 0.125 0.125 0.05 0.05
python lab/final_stack.py <blend11_window_lobt> --inwindow 0.5 --write blend11_stack
# C3. the 15 month-pair members (GPU, holdout only)
for c in exp056w_lean2_window_winter exp059_lean2_window_lobt_anchorw exp060_lean2_global_windoww \
         exp056n_lean2_window_marnov exp059n_lean2_window_lobt_anchor_marnov exp060n_lean2_global_window_marnov \
         exp056s_lean2_window_junaug exp059s_lean2_window_lobt_anchor_junaug exp060s_lean2_global_window_junaug \
         exp056a_lean2_window_aproct exp059a_lean2_window_lobt_anchor_aproct exp060a_lean2_global_window_aproct \
         exp056m_lean2_window_maysep exp059m_lean2_window_lobt_anchor_maysep exp060m_lean2_global_window_maysep; do
  python -m prc train -c $c.yaml --no-test -o $P; done
```

```bash
# C4. the ADS-B per-day tables and the stage's 2025 base (CPU)
python lab/build_lab_matrix.py exp051_lean2.yaml   # cache/lab/lean2/: the stage's NM clocks, the direct members' columns
python lab/adsb_links.py --workers 4
python lab/adsb_features.py --workers 6
python lab/adsb_features2.py --workers 6
#   set BASE_RUN and V37_RUN in lab/idea_harness.py:32-33 to <blend11_stack> (only its holdout index is used downstream)
python lab/idea_harness.py --build                  # cache/lab/adsb_resid.parquet, read by first reception's assembly
python lab/ideas/first_reception.py graph           # once: the taxi-graph distances
python lab/ideas/inbound_turn.py days --workers 2
python lab/ideas/inbound_turn.py assemble           # lab/adsb_features4.py reads it
#   set STACK in lab/rebuild_adsb_extra.sh:10 to <blend11_stack>; MONTHS must list every month (pushback runs on them)
MONTHS="2025-01 2025-02 2025-03 2025-04 2025-05 2025-06 2025-07 2025-08 2025-09 2025-10 2025-11 2025-12 2026-01 2026-07" \
SKIP_REFIT=1 LOG=runs/rebuild_412 bash lab/rebuild_adsb_extra.sh \
  <exp056w>,<exp059w>,<exp060w> <exp056n>,<exp059n>,<exp060n> <exp056s>,<exp059s>,<exp060s> <exp056a>,<exp059a>,<exp060a> <exp056m>,<exp059m>,<exp060m>
#   prints "combined base: <base_plus_winter>"; keep the group order (Feb+Dec, Mar+Nov, Jun+Aug, Apr+Oct, May+Sep)
# C5. the five stage members (v55c on the CPU, beside the GPU ones)
B=<base_plus_winter>; RES="--regime --v5 --drop-prefix tarr_,it_"; DIR="--direct --rich --v5 --drop-prefix tarr_,it_"
python lab/adsb_apply.py $B --depth 10 --iters 5000 --lr 0.02 $RES --gpu --write v55g_resid_360d_d10
python lab/adsb_apply.py $B --depth 8 --iters 6000 --lr 0.02 $RES --write v55c_resid_360d_d8_6000
python lab/adsb_apply.py $B --depth 10 --iters 5000 --lr 0.02 $RES --gpu --seed-offset 100 --write v58g_resid_360d_d10_s142
python lab/adsb_apply.py $B $DIR --depth 10 --iters 8000 --lr 0.02 --gpu --write v55d_direct_360d_d10_8000
python lab/adsb_apply.py $B $DIR --depth 10 --iters 8000 --lr 0.02 --gpu --seed-offset 100 --write v58d_direct_360d_d10_8000_s142
# C6. the stage candidate v60pre (CPU)
python -m prc blend --runs <v55g> <v55c> <v58g> --name resid360_x3
python -m prc blend --runs <v55d> <v58d> --name direct_360x2
cp runs/<v55d>/corrected_rows.parquet runs/<direct_360x2>/     # blend_direct.py reads it; prc blend writes none
python lab/blend_direct.py <resid360_x3> <direct_360x2> --write v60pre_resid360x3_direct360x2 --w 0.3
# C7. the swap members and the Istanbul specialists (GPU, refitted; any time after the data, e.g. during C4)
for c in exp065_lean2_window_long exp066_lean2_global_window_long exp067_lean2_window_lobt_anchor_long exp071_ltfm_spec_long_partner; do
  python -m prc train -c $c.yaml -o $P; done
for c in exp073_lean2_window_deep exp074_lean2_global_window_deep exp075_lean2_window_lobt_anchor_deep exp076_ltfm_spec_deep_partner; do
  python -m prc train -c $c.yaml; done
# C8. v70 = base variant A (each swap member averaged with its depth-9 retrain) + specialist S79 (CPU, minutes)
python -m prc blend --runs <exp071> <exp076> --name dc_S79
python lab/base_swap.py --swap lean2_window=<exp065>,<exp073> --swap lean2_global_window=<exp066>,<exp074> \
  --swap lean2_window_lobt_anchor=<exp067>,<exp075> --candidate <v60pre> --resid-rows <v55g> --direct-rows <v55d> \
  --name dc_A_swap --old-blend <blend11_window_lobt> --old-stack <blend11_stack>
python lab/forced_long_shift.py <dc_A_swap> --min-l 1800 --k 30 --write dc_A_shift
python lab/ltfm_partner_candidate.py <dc_A_shift> <dc_S79> --write dc_A_S79 --w 0.3
python -m prc submit --run <dc_A_S79>
```

- How v70 was chosen: `lab/deep_candidates.py` rebuilds v69 as a control and applies a pre-registered rule to the
  variants (findings 10.30). It needs your v62 and v69 (their recipes: findings 10.24 and 10.28) and its constants
  edited (table below); C8 builds the chosen candidate directly.
- `python lab/manifest.py <dc_A_S79> --out docs/my_manifest.md` writes the lineage (the default output is the
  tracked manifest). It reads `submissions/uploads.csv`, the log of the upload tool, which is not part of this
  repository; create the file with the header line
  `uploaded_at_utc,object,run_id,bytes,status,score,used_pairs,scored_at_utc`.
- Checkpoints, RMSE on the Jan + Jul 2025 holdout without the wrong-date rows (`prc leaderboard`, `final_stack.py` and
  `base_swap.py` print them): blend17_wide 262.55, blend11_window_lobt 257.76, blend11_stack 257.11; v55g / v55c /
  v58g 235.85 / 235.86 / 235.91, v55d / v58d 237.40 / 237.46; v60pre 235.65; v70 235.20 (bulk 187.30).

**Run ids to replace.** These files pin the development machine's run ids; replace each with your run of the same
name (and seed):

| file:line | constant: what it holds | needed for |
|---|---|---|
| `lab/idea_harness.py:32-33` | `BASE_RUN` (blend11_stack), `V37_RUN` (an older stage run); both may be your blend11_stack | C4 |
| `lab/rebuild_adsb_extra.sh:10` | `STACK`: blend11_stack | C4 |
| `lab/base_swap.py:44-45` | `OLD_BLEND`, `OLD_STACK`: the defaults of `--old-blend` / `--old-stack` (C8 passes both) | C8 |
| `lab/deep_candidates.py:44-48, 50-52, 59` | `V60PRE`, `RESID_ROWS`, `DIRECT_ROWS`, `V62_STACK`, `V69`, `LONG`, `S7` | the alternative to C8 |
| `lab/final_run.sh:40-42` | `BLEND11` (blend11_window_lobt), `BASE` (base_plus_winter), `OP_H1_R4` (the op_h1_r4 refit) | D |
| `lab/final_run.sh:80, 82-85` | the `--oof-from` of the five `v70s_*` members: v55c, v55g, v58g, v55d, v58d | D |
| `lab/final_challenge.py:79-80` | `V60PRE` (the new stage candidate's holdout must equal it, else it stops), `V70_DIRECT_BLEND` (direct_360x2) | D |
| `lab/final_challenge.py:86-92` | `STAGE`: the five cross-fitted stage members | D |
| `lab/final_challenge.py:95-106` | `WINTER_GROUPS`: the 15 month-pair members, in this order | D |
| `lab/final_challenge.py:107-117` | `SWAP` (exp065/073, exp066/074, exp067/075), `SPECIALISTS` (exp071, exp076), `LEAN2_SOURCE` (exp051, seed 42) | D |
| `lab/final_challenge.py:78, 81-85` | `V70`, `V70_CHAIN`: v70 and its chain | `--rehearse` only |
| `lab/op_ladder_report.py:202-211` | `COMPARISON`: B0-B2 (baselines), C1 (exp066), C2 (v32 stack), C3 (v62); or pin per call with `--run C1=<your exp066 run>` | path B's read-out (optional; missing rows are skipped) |

`lab/final_inputs.py:839` (`KNOWN_LEVELS`) only selects a header text. To find your ids: every command prints its
run (`artifacts: runs\<id>` from `prc train`, `blend` and `predict`; `wrote <id>` from the lab scripts). Afterwards
`ls -d runs/*__lean2_window_long__16c2b7dd__s42`
(PowerShell: `Get-ChildItem runs -Directory -Filter *__lean2_window_long__16c2b7dd__s42`), or for the newest run of a
name `python lab/final_inputs.py latest lean2_window_long`. An override that changes the configuration (a CPU run,
for one) changes the hash: match `runs/*__<name>__*__s<seed>` instead.

## 7. Path D: the final files from trained runs

This path starts from trained runs, which are not part of this repository: path C trains them, except the `op_h1_r4`
refit (`python -m prc train -c op_h1_r4.yaml`), and prerequisite 4 below lists them.

`lab/final_run.sh` produces every final file in one resumable chain (findings 10.32, 10.33): the scored file, v70's
recipe on the final set, and beside it the files of the operational approaches of 10-11 Oct, which path B's
deliverable superseded (not scored).

Prerequisites:

1. The release in `data_release/2026-10-09/` (section 3).
2. The isolated checkout, made once: `python lab/final_worktree.py create --worktree ../taxiout-finalrun`. It is a git
   worktree whose `runs/`, `cache/features`, `.venv`, `cache/adsb_surface` and `cache/adsb_archive` are junctions to
   this checkout and whose `cache/adsb`, `cache/lab/ideas` and `cache/lab/adsb_resid.parquet` are private copies
   (about 3.8 GB; they exist after step C4).
3. No uncommitted change under `prc/`, `lab/` or `configs/` (commit your run-id edits locally first): the isolated
   checkout runs the committed code, and the chain moves it to this checkout's HEAD at every start.
4. The trained runs of v70 (or yours, with the ids of the table in section 6 replaced): the 27 base members with their
   refit models and their two blends (blend17_wide, blend11_window_lobt), base_plus_winter, the 15 month-pair members,
   the five cross-fitted stage members, v60pre and direct_360x2, the swap members and specialists, the lean2 run and
   the `op_h1_r4` refit. `runs/` is not in git.
5. Section 3 complete: downloads 1 to 7, with PRU, the March to May 2026 weather and the February and June 2026 ADS-B
   days. `data_final/` is made beside `data/` and must be on the same volume (the ADS-B points are hard-linked).
6. Windows with Git Bash, and a free GPU for phase G.

Launch (PowerShell, repository root):

```powershell
# optional, before the launch: $env:CHAIN_START = "2026-10-11 19:00" (UTC, delays the start); $env:FINAL_R6 = "1" (G also trains op_h1_r6, op_h1_r6_s43 and op_h1_r6_shift7; needs the frozen ADS-B snapshot s1 of section 5)
Start-Process "C:\Program Files\Git\bin\bash.exe" -ArgumentList "lab/final_run.sh" -WorkingDirectory $PWD.Path -WindowStyle Hidden -RedirectStandardOutput runs\final_run.out -RedirectStandardError runs\final_run.err
Get-Content runs\final_run.log -Wait -Tail 20
```

| phase | what | time |
|---|---|---|
| G | GPU, in the background, one GPU fit at a time: v70's five stage members refitted on every covered 2025 day and saved (`v70s_*`, `--no-cv --save-models`; v55c on the CPU beside them), then the `op_hl3_r4_rwyblind` and `op_hl24_r4_rwyblind` refits of the superseded long-horizon approach; a stage member is skipped when its complete run exists, a refit when one exists at the configuration's current hash | budgeted 50-70 min per GPU member, 45 min per refit |
| C1 | CPU: the runbook `runs/final_runbook.sh` (data tree `data_final/`, the release validated and installed, `prepare`, the 27 base members through `prc predict`, the blends, the stack, a base-only fallback file, `levelcheck`) | 47 min in the Jan + Jul rehearsal (4 threads) |
| C2 | CPU, isolated checkout: ADS-B points linked for 2025 and the four months, the four months' tables stripped whenever the release changes, `lab/adsb_new_days.sh` for their 120 days against the frozen pools | 50 min per 62 days in the rehearsal |
| F | CPU, isolated checkout, once G's members exist: `lab/final_challenge.py --stage-models ...` (the swap members and specialists predicted, the stage from the saved models, the base swap, the forced-long shift, the Istanbul specialist), then `prc submit` on the final template | under an hour (estimate) |
| O | CPU, once G's refits exist: `prc predict` and `prc submit` of `op_h1_r4`, `op_hl3_r4_rwyblind`, `op_hl24_r4_rwyblind` on the final tree (the superseded approaches, not the deliverable); then the scored file is submitted again | minutes per run |

About 4 h of CPU from the release to the file (findings 10.32). It writes:

- `submissions/submitting.parquet`: the scored file, 670,790 rows, a copy of
  `submissions/<final run>__submitting.parquet` (the chain compares their sha1 at the end).
- `submissions/<run>__submitting.parquet` of `final_op_h1_r4`, `final_op_hl3_r4_rwyblind` and
  `final_op_hl24_r4_rwyblind` (not scored), and of the runbook's `final_stack` (the base alone, the v32 level). The
  three operational files are the approaches of 10-11 Oct, superseded by path B's deliverable (`lab/op_deliver.py`,
  `submissions/operational/`). Run again with today's configurations, G takes the newest `op_hl3_r4_rwyblind` refit
  (path B's 3 h deliverable, where it exists) and trains a new `op_hl24_r4_rwyblind` (path B's 24 h member is r3; on the
  development machine no refit exists at r4's current hash), so these files are not those of 10-11 Oct either.
- `runs/final_run.log` (every phase, and the final run's level check), `runs/final_runbook.out`,
  `runs/adsb_new_days_final.out` and `runs/final_challenge_final.out`.

After a failure, start it again: every step re-uses what an earlier start made, and a second start never runs a second
G beside a live one. Stop it with `taskkill /T /F /PID <windows pid>` (the log's "armed" line names it). With
`FINAL_R6=1`, R6 is trained but not predicted on the final set; without the frozen snapshot, or when the ADS-B gates
fail, the ladder stops before the R6 rungs and only `runs/final_run.log` says so.

**The same, step by step** (Git Bash, main checkout, venv active; as `lab/final_run.sh` runs it):

```bash
TREE="$(pwd -W)/data_final"; REL="$(pwd -W)/data_release/2026-10-09"; M="2026-01 2026-02 2026-06 2026-07"
# G: the five stage members refitted and saved (v55c on the CPU), and the superseded long-horizon refits; skip what exists
BASE=<base_plus_winter>; RES="--regime --v5 --drop-prefix tarr_,it_"; DIR="--direct --rich --v5 --drop-prefix tarr_,it_"
python lab/adsb_apply.py $BASE --depth 8 --iters 6000 --lr 0.02 $RES --no-cv --save-models --oof-from <v55c> --write v70s_v55c
python lab/adsb_apply.py $BASE --depth 10 --iters 5000 --lr 0.02 $RES --gpu --no-cv --save-models --oof-from <v55g> --write v70s_v55g
python lab/adsb_apply.py $BASE --depth 10 --iters 5000 --lr 0.02 $RES --gpu --seed-offset 100 --no-cv --save-models --oof-from <v58g> --write v70s_v58g
python lab/adsb_apply.py $BASE $DIR --depth 10 --iters 8000 --lr 0.02 --gpu --no-cv --save-models --oof-from <v55d> --write v70s_v55d
python lab/adsb_apply.py $BASE $DIR --depth 10 --iters 8000 --lr 0.02 --gpu --seed-offset 100 --no-cv --save-models --oof-from <v58d> --write v70s_v58d
python -m prc train -c op_hl3_r4_rwyblind.yaml; python -m prc train -c op_hl24_r4_rwyblind.yaml
# C1: the runbook; its output names "final_b02 = <id>" and "final_stack = <id>"
python lab/final_inputs.py runbook --tree "$TREE" --from-blend <blend11_window_lobt> --months $M --out runs/final_runbook.sh --shell sh --threads 4
bash runs/final_runbook.sh "$REL/final_ranking.parquet" "$REL/final_submitting.parquet"
V=$(python lab/final_inputs.py derive version --tree "$TREE")             # final26_7a06db85
# C2: the four months' ADS-B tables, in the isolated checkout (strip once per release); it runs committed code
git -C ../taxiout-finalrun checkout -q --detach "$(git rev-parse HEAD)"
python lab/final_worktree.py points --tree "$TREE" --months $M
python lab/final_worktree.py strip --worktree ../taxiout-finalrun --months $M
(cd ../taxiout-finalrun && PRC_DATA_DIR="$TREE" bash lab/adsb_new_days.sh)
# F: v70's recipe on the predict runs, in the isolated checkout; it ends with "final <id>: ..."
(cd ../taxiout-finalrun && python lab/final_challenge.py --tree "$TREE" --data-version "$V" --blend <final_b02> --stack <final_stack> \
  --name final_v70 --stage-models v55g=<v70s_v55g> v55c=<v70s_v55c> v58g=<v70s_v58g> v55d=<v70s_v55d> v58d=<v70s_v58d> --threads 4)
# O: the superseded approaches' files (likewise final_op_hl3_r4_rwyblind and final_op_hl24_r4_rwyblind), each then submitted; the deliverable is path B's
PRC_DATA_DIR="$TREE" python -m prc predict --run <op_h1_r4 refit> -o data.version=$V --tree "$TREE" --threads 4 --name final_op_h1_r4
PRC_DATA_DIR="$TREE" python -m prc submit --run <final_op_h1_r4 predict run>
# the scored file last: every prc submit overwrites submissions/submitting.parquet
PRC_DATA_DIR="$TREE" python -m prc submit --run <final run>              # must report 670,790 rows
PRC_DATA_DIR="$TREE" python -m prc levelcheck --run <final run>
```

**The final upload.** The scored file was uploaded on 14 October 2026 as `adventurous-radio_final.parquet`; the upload
tool is not part of this repository.

## 8. Troubleshooting

- **Git Bash.** Start `lab/*.sh` chains with `C:\Program Files\Git\bin\bash.exe`;
  `C:\Program Files\Git\usr\bin\bash.exe` starts without the MSYS PATH (no `date`, `grep` or `tee`). The scripts call
  `.venv/Scripts/python.exe` themselves: `lab/final_run.sh` and `lab/op_queue.sh` fall back to `.venv/bin/python`,
  while `lab/adsb_new_days.sh`, `lab/rebuild_adsb_extra.sh`, `lab/op_ladder_run.sh`, `lab/base_swap.py` and
  `lab/build_candidate.py` do not. `pwd -W` prints the Windows form of the current directory.
- **Windows paths.** `--run` takes the bare run id, not `runs\<id>`; `-c` takes a configuration file name from
  `configs/`. The longest path under `runs/` is 163 characters relative to the checkout: keep the checkout's own path
  under about 95 characters (the isolated checkout sits beside it) or enable Windows long paths. Before redirecting
  output to a file in PowerShell, set `$env:PYTHONUTF8 = "1"`, as the lab chains do.
- **`.sh` line endings.** `.gitattributes` keeps `*.sh` at LF. If bash stops with `$'\r': command not found`, the file
  got CRLF: `sed -i 's/\r$//' <file>.sh`, then check with `bash -n <file>.sh`. When writing a `.sh` from Python on
  Windows, pass `newline="\n"`. Never edit a `.sh` chain while it runs: bash reads it as it goes.
- **GPU or CPU.** From `exp002` on, every experiment configuration sets `task_type: GPU`, almost all regime-mixture
  configurations also for their classifier, and the `op_*` configurations add `devices: "0"`. Without an NVIDIA GPU
  (macOS included) pass `-o model.params.task_type=CPU`, plus `model.mixture.classifier_params.task_type=CPU` for a
  mixture; this changes the configuration hash and so the run ids. `lab/adsb_apply.py` runs on the CPU without
  `--gpu`. Run one GPU job at a time (8 GB card; two fits sharing it stalled). GPU fits are not bit-reproducible.
  CatBoost takes every core: `python -m prc progress` shows whether a fit still moves before you call it hung.
- **`PRC_DATA_DIR`.** It moves `competition_files/`, `interim/` and `additional_data/`; `cache/` and `runs/` stay in
  the checkout, so every data tree has its own `data.version` (`final26_<sha1[:8]>`) and a cache version belongs to one
  tree. A variable left set in a PowerShell session redirects every later command: `Remove-Item Env:PRC_DATA_DIR`.
  `prc submit` maps a run onto the template of the current tree: without `PRC_DATA_DIR` a final run goes onto the
  delivered January + July template and writes 344,841 rows without an error, so check for 670,790. `prepare` on the
  final tree needs `PRC_EXPECTED_RANK_ROWS=1341197`. The runbook sets and restores both itself, `final_challenge.py`
  sets the tree from `--tree`, `lab/final_run.sh` unsets both at its start.
- **`prc predict refused`** (exit code 2): the message says why. `--allow-unseen` accepts far more unseen categorical
  levels than the run's own ranking rows had; `--allow-drift` accepts feature code or input tables that changed since
  the run (for example `data/` re-normalised after training). The runbook takes them as `--allow-unseen` /
  `--allow-drift` (bash) or `-AllowUnseen` / `-AllowDrift` (PowerShell), `final_challenge.py` as flags.
  `lab/final_run.sh` passes neither: run the failing step by hand with the flag, then start the chain again, which
  re-uses the finished runs.
- **The isolated checkout's junctions.** `../taxiout-finalrun/runs` and the other junctions point at this checkout.
  Remove the isolated checkout only after `python lab/final_worktree.py unlink --worktree ../taxiout-finalrun` (it
  removes the junctions, never their targets), then `git worktree remove --force ../taxiout-finalrun`: a forced remove
  of a checkout that still holds the junctions could delete this checkout's `runs/`. `create` refuses an existing path.
  The junctions come from `cmd /c mklink /J`; Linux would need symlinks, which are not implemented.
- **Linux and macOS.** Paths A and B use `prc/` and portable Python lab scripts (`lab/final_stack.py`,
  `lab/final_inputs.py`, `lab/op_ladder_report.py`) and should run there (untested: all development was on Windows
  11). Write the final-set runbook with `--out runs/final_single.sh` and run it with `bash runs/final_single.sh
  <ranking> <template>`; translate the PowerShell lines (`$Tree`, `$env:...`) to `export`. R6's `lab/op_ladder_run.sh`
  and path C's scripts call `.venv/Scripts/python.exe` (adjust them); path D needs Windows.
