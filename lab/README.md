# lab/

The scripts that turn the harness's trained runs (`python -m prc train`) into the delivered files: the post-operations
analysis model's second stage and post-processing (the code calls this model the challenge model), the replay of that
recipe on the final ranking set, and the operational models' rules and delivery. The harness itself is the `prc`
package; [docs/DEVELOPER_GUIDE.md](../docs/DEVELOPER_GUIDE.md) describes both, its Appendix A the second stage.
[docs/QUICKSTART.md](../docs/QUICKSTART.md) gives the full command sequences with the arguments of the recorded runs:
path A (one model), path B (the operational model), path C (v70 from scratch) and path D (the final files).

- Run every script from the repository root with the virtual environment active: `python lab/<script>.py ...`. The
  `.sh` chains need bash (Git Bash on Windows, where they were run). The chains, `base_swap.py` and
  `build_candidate.py` start the interpreter of a Windows virtual environment, `.venv/Scripts/python.exe`; only
  `final_run.sh` and `op_queue.sh` fall back to `.venv/bin/python` (docs/QUICKSTART.md section 8).
- The stage and post-processing scripts read finished run directories under `runs/` and write a new one in the same
  format, so `prc blend`, `prc submit` and the leaderboard accept it; each records its inputs and parameters in the
  new run's `metrics.json`.
- Some scripts pin run ids of the development machine as constants; docs/QUICKSTART.md section 6 ("Run ids to
  replace") lists them.
- `lab/` lies outside `prc/features/`, so nothing here is covered by the feature-cache hash.

## The post-operations analysis model: v70's recipe

v70 is the upload whose recipe the final file applies to the final ranking set; its lineage is
[docs/submission_manifest.md](../docs/submission_manifest.md). The scripts follow the training and averaging of the
base members (`prc train`, `prc blend --lobt-bound`), in the order of the submitted pipeline. The ADS-B stage reads
the flight's own pushback and taxi, so it is tier M, challenge-only ([docs/feature_table.md](../docs/feature_table.md)).

| script | what it does | how it is run |
|---|---|---|
| `final_stack.py` | the stack on a blend: the NM join-window clip, an in-window estimate for the rows the mixture hedges, the regime exclusion for linked untyped rows; prints the holdout check | `python lab/final_stack.py <blend> --inwindow 0.5 --write NAME` |
| `linked_untyped.py` | the regime exclusion for unmatched, untyped departures whose flight is NM-joined at its destination; imported by `final_stack.py`, alone it prints the holdout effect | `python lab/linked_untyped.py [run]` |
| `build_lab_matrix.py` | one configuration's feature matrix in `cache/lab/<name>/`: the stage's NM clocks and the direct members' columns; `--rescale-from` for another data tree | `python lab/build_lab_matrix.py exp051_lean2.yaml [-o k=v ...] [--rescale-from RUN --out NAME]` |
| `build_winter_base.py` | the stage's base for all of 2025: the stack's holdout predictions on January and July, the month-pair runs' on the other ten months | `python lab/build_winter_base.py <stack> <group> ...` (a group: runs sharing a holdout, comma-separated) |
| `adsb_apply.py` | the ADS-B second stage: CatBoost on the covered departures, the residual of the base or (`--direct`) the taxi time itself, cross-fitted by day on the holdout and refitted for 2026 | `python lab/adsb_apply.py <base> [flags] --write NAME` |
| `blend_direct.py` | 0.7 × the residual stage + 0.3 × the direct stage on the rows both cover | `python lab/blend_direct.py <resid> <direct> --w 0.3 --write NAME` |
| `base_swap.py` | longer-trained members swapped into the base blend; the new stack on the rows no stage run corrects | `python lab/base_swap.py --swap CFG=RUN[,RUN] ... --candidate RUN --resid-rows RUN --direct-rows RUN --name NAME` |
| `forced_long_shift.py` | a shrunk shift where the join window forces a taxi above 30 min, estimated by day-level cross-validation on the holdout (tier C) | `python lab/forced_long_shift.py <run> --min-l 1800 --k 30 --write NAME` |
| `ltfm_partner_candidate.py` | Istanbul's rows blended with the partner-runway specialist, re-clipped to the window | `python lab/ltfm_partner_candidate.py <reference> <specialist> [--control RUN] --write NAME --w 0.3` |
| `build_candidate.py` | the member average, the formulation blend, the base swap and the shift in one command | `python lab/build_candidate.py --resid RUN ... --direct RUN --new-stack RUN --name NAME` |
| `deep_candidates.py` | v69's recipe rebuilt with the depth-9 retrains: the base variants, the specialists, the control and the pre-registered choice that gave v70 (findings 10.30) | `python lab/deep_candidates.py [--control-only]` |
| `common.py` | shared helpers: the lab matrix with the movements and the target, the seasonal split, the regime labels, the holdout metrics | imported by `final_stack.py` and `linked_untyped.py` |

## The ADS-B stage tables

The per-flight tables the stage reads, built day by day into `cache/adsb/` and `cache/lab/ideas/` from the points of
`python -m prc fetch-adsb` (adsb.lol, ODbL 1.0; [docs/data_sources.md](../docs/data_sources.md)). No raw data is part
of this repository.

| script | what it does | how it is run |
|---|---|---|
| `adsb_merge_staging.py` | the union of adsb.lol's staging release with the prod points the features read | `python lab/adsb_merge_staging.py [--restore]` |
| `adsb_links.py` | the departure-to-aircraft links of every day (`cache/adsb/links_<day>.parquet`) | `python lab/adsb_links.py [--workers N] [--force]` |
| `adsb_features.py` | feature set 1: the own movement (family B) and the surface context at the planned off-block (family A) | `python lab/adsb_features.py [--days D ...] [--workers N]` |
| `adsb_features2.py` | feature set 2: the ground history read against the flight's own stand and its stops | `python lab/adsb_features2.py [--days D ...] [--workers N] [--force]` |
| `adsb_features3.py` | feature set 3, assembled from the builders below: pushback, first reception, track quality, stop classes | `python lab/adsb_features3.py` |
| `adsb_features4.py` | feature set 4, assembled likewise: inbound and turnaround, stand-type archive, time-to-location map | `python lab/adsb_features4.py` |
| `idea_harness.py` | the quick screen of a candidate feature against the stage's residual; `--build` writes `cache/lab/adsb_resid.parquet`, which first reception's assembly reads | `python lab/idea_harness.py --build`; `python lab/idea_harness.py cand.parquet` |
| `rebuild_adsb_extra.sh` | the per-day steps for the months in `MONTHS`, the pooled archives rebuilt, the combined base and, without `SKIP_REFIT`, the stage | `MONTHS="..." SKIP_REFIT=1 bash lab/rebuild_adsb_extra.sh <group> ...` |
| `adsb_new_days.sh` | the per-day tables of new days against frozen pools (the final set), in the isolated checkout | `PRC_DATA_DIR=<tree> bash lab/adsb_new_days.sh` |
| `ideas/pushback.py` | the reverse-motion pushback and engine-start detector | `python lab/ideas/pushback.py --holdout` or `--days D ...` |
| `ideas/first_reception.py` | the aircraft's state at its first ground sighting and the share of its route already covered | `python lab/ideas/first_reception.py graph`, then `days`, `assemble` |
| `ideas/track_quality.py` | reception density, gaps, stand jitter, implausible jumps, the inbound landing, relocations | `python lab/ideas/track_quality.py [--workers N]` |
| `ideas/stop_class.py` | the stops between the stand and the takeoff, classed by location (stand, runway area, remote, hold) | `python lab/ideas/stop_class.py [--days D ...] [--workers N]` |
| `ideas/stop_class_screen.py` | the stop-class columns and their location map (imported by `adsb_features3.py`), screened with the harness | `python lab/ideas/stop_class_screen.py v1` (or `v2`, `v3`, `avail`) |
| `ideas/inbound_turn.py` | the aircraft's own inbound arrival and turnaround, 24 h back | `python lab/ideas/inbound_turn.py days`, then `assemble` |
| `ideas/stand_archive.py` | a target-free archive per stand of the fully observed off-blocks, and off-block estimates from it | `python lab/ideas/stand_archive.py obs`, then `archive`, `days` |
| `ideas/timemap.py` | a target-free time-to-location map: push start to leaving a surface cell; for arrivals, a cell to the in-block | `python lab/ideas/timemap.py extract`, then `map`, `arr`, `arrmap`, `days`, `assemble` |

## The final-set replay

The trained runs applied to the final ranking set (January, February, June and July 2026) without retraining.

| script | what it does | how it is run |
|---|---|---|
| `final_inputs.py` | the data tree of a new ranking set: `init`, `validate` (with `--install`), `runbook` (writes the script that predicts the set from finished runs), `derive`, `latest`, `check` | `python lab/final_inputs.py runbook --tree DIR --from-blend RUN --out final.ps1` |
| `final_worktree.py` | the isolated checkout the stage runs in on a new set: `create`, `points`, `strip`, `compare`, `unlink` (before `git worktree remove`) | `python lab/final_worktree.py create --worktree ../taxiout-finalrun` |
| `final_challenge.py` | v70's additions on top of the runbook's stack, each step the submitted script with the submitted parameters; `--rehearse` reproduces v70 on the delivered set | `python lab/final_challenge.py --tree DIR --data-version V --blend RUN --stack RUN --name NAME --stage-models ...` |
| `final_run.sh` | the whole chain, resumable: phases G (GPU refits), C1 (the runbook), C2 (the ADS-B tables), F (the scored file), O (the operational files of 10-11 October) | detached Git Bash (docs/QUICKSTART.md section 7) |

## The operational models

Models that predict before the aircraft moves: the pre-registered rules, the calibration, the delivery, the ladder's
read-out and runner, and the input catalogue.

| script | what it does | how it is run |
|---|---|---|
| `op_rules.py` | the OP metrics of runs, rules L, F and R from holdout predictions, the refits to deliver | `python lab/op_rules.py metrics RUN_OR_NAME ...` (or `L`, `F`, `R --lead hl3`, `refits NAME`) |
| `op_calibrate.py` | rule C: a calibrated median and an 80 % band from the 2025 month pairs | `python lab/op_calibrate.py fit --pairs RUN ... --out T.parquet`, then `check`, `apply` |
| `op_adopt.py` | the adoption of F1 as the H1 deliverable: the superseded records kept, the generalisation checks with rule F, rule M | `python lab/op_adopt.py archive`, then `checks`, `rule-m` |
| `op_rule_t.py` | rule T: whether the traffic after the H1 anchor helps, and the member it chooses | `python lab/op_rule_t.py stage1`, then `stage2` |
| `op_deliver.py` | one file per decision moment on the final set: the refits predicted and averaged, rule C's band, rule M's median, the manifest, the PRU level check; never `prc submit` | `python lab/op_deliver.py --tree data_final --member h1 --config op_h1_f1_plan.yaml --n 3 ...` |
| `op_adsb_audit.py` | the gate before any fit with the ADS-B surface picture: own-airframe contamination, reception per airport and month, the LTFM placebo | `python lab/op_adsb_audit.py --snapshot s1` |
| `op_clock_checks.py` | the ladder's data evidence: the clocks around the off-block, how well the own runway is known at each decision moment (section 7 is `docs/op_runway_knowability.txt`), the queue | `python lab/op_clock_checks.py [--section N ...] [--out FILE]` |
| `op_ladder_report.py` | the ladder's read-out, every rung scored the same way: `docs/operational_ladder.md` and its csv files | `python lab/op_ladder_report.py [--out STEM]` |
| `op_ladder_run.sh` | the ladder on real data: the leakage gates, a cache warm, a smoke fit, then each rung that ends before a deadline | `bash lab/op_ladder_run.sh "YYYY-MM-DD HH:MM" [rung ...]` |
| `op_queue.sh` | the GPU queue: the steps of `runs/op_queue.txt`, once each, in order | `bash lab/op_queue.sh` (detached) |
| `feature_catalogue.py` | every operational input with its definition, source, availability, tier, rationale and evidence: `docs/operational_features.md` and `.csv` | `python lab/feature_catalogue.py [--gain ...] [--perm ...] [--out STEM]` |

## Documentation generators

| script | what it does | how it is run |
|---|---|---|
| `manifest.py` | the lineage of a submitted run, read from the run records: `docs/submission_manifest.md` | `python lab/manifest.py <run> [--out FILE]` |
| `report_data_findings.py` | the numbers of `docs/report.md` section 1.4: the records' defects and each airport's strongest disrupted day of 2025; prints only | `python lab/report_data_findings.py` |
