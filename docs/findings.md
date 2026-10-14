# Findings and improvement backlog

What the data turned out to be, what moved the score, what did not, and where the next
gains are. Every number below was measured in this repository on the Jan+Jul 2025 holdout
(`seasonal_holdout`, GPU CatBoost) unless it says "leaderboard". Run ids and configs are
named so each figure can be reproduced with `prc leaderboard` or the config in `configs/`.

> **Standing priority (project owner, 5 October 2026): the leaderboard is only part of the final ranking.** A big part
> of the final evaluation goes to the model design and its documentation, so the documentation must be excellent on
> both sides: technical (reproduction, pipeline, code, configs, a submission manifest) and scientific (`docs/paper.md`:
> method, validation, results, limitations, operational availability). Results are written up as they land, and the
> last days before the 11 October deadline are reserved for documentation passes, not only uploads.

## 1. Where we stand

> **Current state (14 Oct 2026).** The table below is the history of the early submissions; the later ones are in
> section 10. The scored final file is the challenge model on the final set, January, February, June and July 2026
> (10.34): v73, its January and July rows, scored 220.393; the full file was uploaded as v72 and v74 and, after the
> board checks of the operational deliverables on 12 Oct (v75-v77: H1 561.0, 3 h 589.1, 24 h 596.1), again as v78 and,
> after those of 13 Oct (v79-v82, 10.36), as v83. On 14 Oct the owner uploaded it once more as the team's unique final
> submission, `adventurous-radio_final.parquet`; the scorer accepted all 670,790 rows and returned no score (10.37).
> Since the owner's goal of 11 Oct, the deliverable is the operational model family, which reads only data available
> before the off-block: H1, 3 h and 24 h ahead, with H0 as a comparison (10.35, `docs/operational_model.md`,
> `docs/operational_features.md`). Beside it, the concept member reads the other flights' traffic after the H1 anchor as
> a stand-in for planning data (tier F, an upper bound; 10.36). The best board score stays v70, 220.3396 (10.30).

| submission | config | what it is | holdout RMSE | leaderboard |
|---|---|---|---|---|
| v2 | `exp017_final_raw_per_airport.yaml` | raw target, one CatBoost per airport, day-offset rule, refit on all months | 426.8 | 422.4 |
| v3 | `exp022_final_stale.yaml` | the same without the rule, plus the stale-schedule group as a residual base | 349.1 | **329.5** |
| v4 | `exp024_final_mixture.yaml`, two seeds blended | the regime mixture over the wide universe (section 3), refit on all months, seeds 42 + 43 averaged | 344.3 | **329.0** |
| v5 | `exp027_final_datefix.yaml` | exp024 trained on a copy of the data with the wrong-date block times corrected, day-offset regime off: a test of "2026 has no wrong-date rows" | 337.4 (not comparable: its holdout is corrected too) | 337.3 |
| v6 | `exp028_final_stale_only.yaml` | exp024 with the wrong-date rows out of the fit and the day-offset regime off: long-delta rows priced at the schedule delta | 364.2 (275.8 without the wrong-date rows) | 318.0 |
| v7 | `exp028`, three seeds blended (42–44) | the same, seed-averaged | 367.6 (274.8) | **317.0** |
| v8 | `exp028`, seed 45 alone | a fourth seed, for the blend | 378.2 (278.4) | 318.5 |
| v9 | `exp030_final_context.yaml`, three seeds blended (42–44) | exp028 plus the five winning context blocks of the 2026 sweep (7.5b) | 344.4 (274.8) | **311.8** |
| v10 | `exp037_final_era5_mixture.yaml`, seed 42 alone | exp030 + the winter blocks (`wx_deice_load`, `arrival_conditions`), the mixture from 30 min schedule deltas at a 60 s tolerance, ERA5 (section 8) | 337.3 (268.2; bulk 217.7) | **306.6** |
| v11 | `exp038_final_nodates_mixture.yaml`, seed 42 alone | exp037 without the four time-position columns (section 9.5) | 352.1 (268.9; bulk 218.8) | 309.1 |

Top teams sit near 240 (section 7 is the analysis of that gap). The holdout tracks the board for bulk changes (v2 within 5 s), but
not for the tail: v3 came in 20 s under its holdout figure, and v4's 4.8 s holdout gain
(most of it the hedging of Rome's wrong-date rows and one Zürich model that finally trains)
became 0.5 s on the board. 2026 has fewer of the rows the holdout is decided by, so the
board is now mostly a bulk score: at ~329 against a wrong-date-free holdout of 273, the
2026 rows we cannot see still cost about 55 s of RMSE, and what moves the board from here
is the bulk (section 5, item 1) and whatever 2026-specific tail those rows are.

v5 was a deliberate probe of that tail. Trained on data with the fifteen wrong-date rows
corrected (section 2.4), it stopped hedging the eleven untyped 2026 Rome rows with a
schedule delta above 12 h toward a day, and, because the corrected rows now looked like
genuine taxis to the classifier, hedged six of them toward a taxi instead (14,000–44,000 s
against v4's 60,000–82,000). The board moved 329.0 → 337.3, a loss of 1.9·10⁹ in squared
error. Against the three readings of those six rows: stale schedules (target = delta)
predict a loss of 2.6·10⁹; wrong-date records (target ≈ 87,600) a loss of 2.0·10¹⁰; genuine
taxis a *gain* of 2.4·10¹⁰. Only the first fits, and a single wrong-date row among the six
would have doubled the loss. So 2026's long-delta Rome rows are stale schedules, and the
right prediction for them is the delta itself, which v4 misses by about 6,700 MSE, ~10 s of
board RMSE (`exp028_final_stale_only.yaml`: original data, wrong-date rows out of the fit,
day-offset regime off).

## 2. What the data is

### 2.1 The target is not always a taxi time

`TAXITIME_SEC_mvt` is `MVT_TIME − BLOCK_TIME`. For a minority of departures the recorded
block time is not a measurement but the **scheduled** off-block time, so the "taxi time" is
a departure delay:

| training departures | count |
|---|---|
| taxi-out above 2 h | 584 |
| of those, target equals takeoff − `SCHED_TIME_UTC_mvt` within 2 min | 438 |
| within 30 min | 507 |
| largest target | 131,167 s (36 h) |

On a stale record the target matches the schedule delta to a few seconds (median gap 3 s,
p90 5 s). This is the whole reason RMSE is a tail metric here: the worst 0.1 % of holdout
rows carried 70–80 % of the squared error before the fix.

### 2.2 The stale segment

Stale records live almost entirely in one segment: departures with **no
`AIRCRAFT_TYPE_mvt` and no NM join** (every `_flt` column null, `FLIGHT_ID_mvt` null).

| | training | 2026 ranking |
|---|---|---|
| segment rows | 1,487 (1,466 at LIRF; 7 LSZH, 7 LTFM, 5 LFPG, 1 EHAM) | 383 LIRF, 4 LTFM, 1 LSZH |
| with schedule delta above 2 h | 627 | 181 |
| stale share (target within 10 min of the delta) | 49.8 % | scored, unknown |

The other half of the segment is genuine: a normal taxi of about 1,000 s after a delay of
hours. What separates the halves, in order of the classifier's importance (month-grouped
CV, AUC 0.915 with strict labels): the carrier prefix of `FLIGHT_mvt`, the destination, the
schedule delta and its fraction of a day, the month, the stand area, the runway.

| carrier prefix | segment rows | stale |
|---|---|---|
| ISR, LAV, ETH | 59, 23, 24 | 98 %, 100 %, 92 % |
| EXS, AEZ, NOS, WZZ | 24, 51, 31, 59 | 79 %, 77 %, 74 %, 71 % |
| WMT, RYR, VLG | 183, 239, 21 | 49 %, 34 %, 33 % |
| ITY, EJU, EZY, BEL | 94, 40, 25, 23 | 16 %, 10 %, 8 %, 9 % |

Weaker fingerprints, all already visible to the model: runway `NA` (11 rows, 100 % stale),
missing flight number (59 rows, 86 %), destination LIRF (20 rows, 95 %), takeoff 20–23 UTC
(314 rows, 69 %). The schedule delta itself: 37 % stale below 2 h, 63 % above. A **negative**
delta (schedule after takeoff, i.e. another leg's schedule) is never stale. There is no
data-path fingerprint: takeoff seconds, schedule minutes and stand presence are the same in
both halves.

### 2.3 Typed rows with stale blocks

Outside the segment the phenomenon exists but is rare and not predictable as a class:

| rows with taxi above 2 h (training) | n | stale by schedule | matches takeoff − `LOBT_flt` |
|---|---|---|---|
| LIRF, typed and NM-joined | 41 | 37 | — |
| EGLL, typed | 37 | 18 | 8 of 20 holdout rows |
| LFPG, typed | 37 | 10 | — |
| LTFM, typed | 21 | 4 | — |

Among the 2,748 typed Rome rows with a delta of 2–12 h only 1.3 % are stale, so the
MSE-optimal prediction for them is close to a normal taxi, which the model already gives.
The one typed Rome row with a one-day delta in the whole year is stale, sits in the holdout,
and alone carries 17 % of the holdout's squared error.

They do have a fingerprint, in NM's own timestamps. On a flight delayed by hours NM normally
carries the delayed off-block time; when `LOBT_flt` (or `IOBT_flt`) still equals
`SCHED_TIME_UTC_mvt`, the movement record's block time tends to be a schedule too:

| NM-joined rows with a schedule delta ≥ 2 h | rows | stale |
|---|---|---|
| `LOBT_flt` = schedule (within 2 min) | 103 | 42 (41 %) |
| `LOBT_flt` ≠ schedule | 34,804 | 24 (0.07 %) |

On the stale rows `AOBT_3_flt` sits about two hours after the schedule (NM's actual
off-block is the real one; the airport's record is the schedule). `stale_schedule` now
outputs `lobt_minus_sched` and `iobt_minus_sched` unclipped (`nm_deltas` blanks anything
beyond a day, so the one-day row above was invisible). The typed one-day Rome row has both
equal to zero.

### 2.4 Wrong-date block times

Fifteen training departures have a target within an hour of exactly one day (13 LIRF,
1 LFPG, 1 LSZH): the block time carries the previous day's date, so the target is
86,400 s plus an ordinary taxi. Seven of them sit in the holdout (six at Rome: four in the
untyped segment, the typed one-day stale row above, one more; and the Paris row at 84,240 s
with a 29-minute schedule delta) and carry 37 % of exp022's holdout squared error. Without
them the holdout reads 277 instead of 349 (`rmse_ex_dayoffset`, reported by every run now).

They are announced after all, inside the untyped segment: twelve of the fifteen sit there,
and there a large schedule delta means a stale or a wrong-date block, never a genuine taxi:

| untyped, NM-less rows by schedule delta | rows | stale | wrong date | genuine |
|---|---|---|---|---|
| 2–4 h | 503 | 298 | 1 | 204 |
| 4–8 h | 77 | 62 | 0 | 15 |
| 8–16 h | 37 | 35 | 2 | 0 |
| above 16 h | 13 | 4 | 9 | 0 |

Outside the segment a delta above a day is an ordinary rescheduled flight (150 typed rows,
median taxi 1,080 s, one stale, one wrong-date), so the three typed wrong-date rows (Paris,
Zürich, and the Rome one, which is stale by schedule) stay unannounced. `model.mixture` with
`day_offset: true` treats "a day plus a taxi" as a third regime of the classifier; on the
Rome holdout the four untyped ones get p ≈ 0.7 from seven training examples.

Whether they recur in 2026 was settled by the board, not the holdout (section 1, v5): the
2026 untyped Rome rows with long schedule deltas are stale schedules, not wrong-date
records, so the day-offset hedge that the 2025 holdout rewards costs about 10 s on the
board. The holdout's seven wrong-date rows are a 2025 artefact. `exp017`'s 15 s loss with
the day-offset rule was the residual-base model losing its way of hedging them, which the
board never paid for. Leaving them in a regressor's fit as 87,000 s "taxi times" is wrong
either way (section 3): they pull every neighbouring prediction up by hours.

What does *not* announce a stale or wrong-date record: the movement id. Ingestion order
(id against the local median id of the surrounding 200 movements), last digit and parity
all give a flat stale rate across the untyped segment, as do the takeoff seconds and the
schedule minutes noted before.

### 2.5 Other things that look wrong but are real

- Helicopters at LSZH and EHAM (runway `H`/`HEL`) taxi for seconds, sometimes negative
  seconds; 313 of the 388 non-positive training targets are Zürich helicopters, and 90
  helicopter rows are in 2026. The `min_taxi_sec` cleaning rule exempts them.
- Paris's 65 non-positive targets are genuine errors, but too few to matter.
- LSZH has one typed Citation departure at 87,341 s (a wrong-date row). In a per-airport
  fit on the raw target it is ~40 % of Zürich's training squared error; the fold model
  early-stops at iteration 31 with the validation curve rising. Costs about 1 s overall.
  Unresolved.

### 2.6 Where the squared error sits now (exp023b fold models, delta-weighted classifier)

| slice | share of squared error | RMSE-equivalent |
|---|---|---|
| three typed rows nothing announces: Rome 87,002 s (stale by a schedule a day out), Paris 84,240 s and 58,206 s (schedule deltas of 29 and 34 min) | 42 % | — |
| the classifier's universe minus the Rome row above (9,316 rows: 165 stale, 5 wrong-date, the rest genuine after a long delay) | 15 % | 800 |
| everything else, the bulk (335,000 rows) | 41 % | 225 |

Per airport: EDDF 182, EDDM 179, EGLL 254, EHAM 190, LEBL 219, LEMD 173, LFPG 569, LIRF 758,
LSZH 234, LTFM 255. Without the seven wrong-date rows the holdout reads 275 (Rome 524, the
rest 243). Inside the universe the mixture scores 1,228 against 2,942 for the regressor
alone and 512 for the oracle that knows every row's regime, so the classifier's remaining
uncertainty costs about a third of the universe's error and the Rome one-day row most of
the rest: it gets p ≈ 0.02 because the 150 typed training rows with a delta above a day
are all genuine, and no fingerprint outranks that split.

## 3. What moved the score

| change | config | effect | status |
|---|---|---|---|
| Raw target, no clip, instead of log1p on a [60, 7200] clip | `exp005` vs `exp002` | 536 → 430 (clip ~40 s, transform ~60 s, isolated) | in |
| One CatBoost per airport | `exp015` vs `exp005` | 430 → 411 | in |
| Stale-schedule group as residual base + carrier/destination stale rates | `exp020` vs `exp015` | Rome 1,101 → 779; overall 411 → 349 | in |
| GPU training | `exp002` | same RMSE, 4.5–11× faster; noise floor ±2 s over 3 seeds | in |
| Refit on all twelve months for the ranking predictions | `exp022` | not measurable locally; predictions correlate 0.985 with the fold mean | in |
| The regime mixture: a classifier over taxi / stale schedule / wrong-date regimes times the candidate deltas, the regressor fitted without those rows, over every departure with a delta ≥ 2 h (`model.mixture`) | `exp023b` vs `exp022` | 349.1 → 344.9; without wrong-date rows 277.2 → 274.8; EGLL 256 → 254, LSZH 245 → 234, LIRF 777 → 758; Rome-only 779 → 764 ± 16 (ex-dayoffset 537 → 521 ± 3) | in |

The mixture's first version was *worse* than the residual base (Rome-only 827 vs 779), and
the reason is worth keeping: the regressor was still fitted on the wrong-date rows (87,000 s
targets that match no schedule) and on the typed stale rows outside the suspect universe,
and it answered 20,000–85,000 s on suspect rows whose real taxi is 1,100 s. Bounding its
targets made things worse still (1,173), because those wild answers had been hedging the
six wrong-date rows in Rome's holdout. The fix was to make the hedge explicit: the
day-offset regime (section 2.4) and the wide universe, which hands the typed stale rows to
the classifier. Weighting the classifier's rows by the candidate delta (what a wrong regime
costs) is worth about 3 s on the headline and nothing on the honest number.

The regressor's fit still held ~70 rows a year with a multi-hour target that matches no
regime (`regressor_max_target_sec: 7200` removes them). At full scale the headline stays at
345.0 and the wrong-date-free number goes 274.8 → 274.3, but the airports move unevenly:
LSZH 234 → 209, because its model finally trains past iteration 49 (one typed 87,341 s row
had been stopping it), while EGLL, EHAM, LFPG and LIRF give back 1–5 s each (the rows'
2026 twins are scored, and the fit had been hedging them a little). A bound of 20,000 s
removes only three rows and does not cure Zürich (iteration 139, 220). Per-airport numbers
are reproducible to 1e-6 between runs with identical training sets, so those shifts are
the data, not GPU noise; the mixture's own three-seed floor is measured overnight.

Blending the two full mixture runs (with and without the bound) gives 344.1 / 273.5: even
two nearly identical models average into a second of gain, which is what the three-seed
blend of the submission run is for.

Why the stale fix needed a residual base and not just a feature: `nm_deltas.sched_delta` is
NaN beyond one day, and CatBoost quantises a feature into `border_count` = 128 bins, so every
delta above two hours shares one bin. With `target.residual_of: stale_base` the exact delta
is added back outside the trees and a stale row needs a residual of zero. The block's
features alone got Rome to 811 (vs 806 with the residual, first version); the stale-rate
block added another 9 s at Rome; excluding negative deltas from the base another 18 s.

## 4. What did not move it

| idea | config | result |
|---|---|---|
| Airport geometry: OSM stand positions, taxiway routing distance, runway length | `exp013` (raw base), `exp008/009` (log1p base) | within noise (436 vs 430 ± 2); on the log1p base it substituted for the learned stand→runway target statistics rather than adding to them. The pipeline and the coverage report are kept; 9 of 10 airports match ≥ 90 % of stands, EDDM is weak |
| METAR/TAF weather, observed and forecast | `exp014` | within noise (432) |
| Cleaning: drop impossible rows (below 30 s, above 2 h, wrong-date) from the fit | `exp016`, `exp011/012` | bulk improves 21 s (258 → 237 on unflagged rows) but the honest score loses 115 s, because the dropped rows' 2026 twins are scored |
| Day-offset rule alone | `exp017` | −15 s |
| Higher iteration cap or more rounds | — | eight of ten fold models stop on flat curves; the two capped airports carry 8 % of the squared error; a cap above 4000 is worthless |
| Predicting a constant for the Rome segment | analysis | 541 vs 537 at the time; the segment is a mixture, not a constant |
| Mixture variants on Rome alone (wide universe + day-offset regime, 764 ± 16 / ex-dayoffset 521 ± 3 over three seeds) | `exp023b` + `-o` | relative label tolerance `tol_frac=0.05`: 763 / 521; regressor without targets above 7,200 s: 752 / 524; classifier rows weighted by the candidate delta (`classifier_weight_power` 1, 2): 740 / 517, 750 / 517; the NM fingerprint columns: 742 / 518. All within the seed spread |
| A data-path fingerprint for stale records in the movement id | analysis | ingestion order, last digit and parity all give a flat stale rate (section 2.4) |

## 5. Ideas to improve further, by expected value

What the mixture settled: the two-stage model, the relative tolerance (no effect), the
typed stale rows and London's LOBT rows (in the wide universe; a LOBT base itself is a
trap, NM's LOBT is the real off-block on ordinary rows), the validation hygiene
(`rmse_ex_dayoffset`), and the stale-rate keys (the classifier has the raw columns). What
is left, by expected value:

1. **The bulk (225 RMSE-equivalent on 335,000 rows, 41 % of the squared error).** This is
   now the largest learnable slice, and the leaderboard weighs it more than the holdout
   does if 2026 has fewer unannounced rows.
   - Re-measure the geometry and weather groups on the mixture base (`exp025`, `exp026`).
     Their "within noise" verdicts were read on a headline where the tail carried most of
     the squared error; a 5 s bulk gain moved a 430 headline by 2 s. Read
     `rmse_ex_dayoffset` and the per-airport columns.
   - Seed averaging (`--seeds 3 --keep-test`, `prc blend`): typically 2–5 s.
   - The hyperparameter sweep (`configs/_generated/sw_*.yaml`): learning rate 0.03 with
     8,000 rounds, depth 6 / 9, `l2_leaf_reg` 1 / 10, `border_count` 254, Bernoulli 0.8.
     Judge against the mixture's own three-seed floor, on `rmse_ex_dayoffset`.
   - The regressor's fit still holds ~70 typed rows a year with multi-hour targets that
     match no schedule (`regressor_max_target_sec: 7200` removes them; Zürich's model
     early-stops at iteration 49 because of one).
2. **The classifier's uncertainty (a third of the universe's error).** The genuine rows of
   carriers that are usually stale (PGT, WZZ, NBT) get p ≈ 1 and cost 15,000–20,000 s
   each; the stale rows of carriers that are usually genuine get p ≈ 0. Nothing in the
   row separates them today (section 2.4 lists what was checked). Ideas not yet tried:
   the same flight number's other movements (did it also depart on the scheduled day, or
   arrive from the schedule's origin?), and the stand's other departures the same day.
3. **The three unannounced typed rows (42 %).** Not a modelling target: the Rome one-day
   row is the only stale typed row with a delta above a day in the whole year, and the two
   Paris rows have half-hour schedule deltas. Their weight in any single holdout is why
   `rmse_ex_dayoffset` and a second holdout pair (`-o 'validation.holdout_months=["2025-04","2025-10"]'`)
   decide tail work, and why the board can sit well under the holdout.

## 6. How to work on this

- Rome-only runs reproduce the LIRF sub-model of a per-airport run exactly and take about a
  minute: `python -m prc train -c exp023b_raw_per_airport_mixture_wide.yaml -o "data.airports=[LIRF]" --no-test`.
  With `model.mixture` the classifier of a Rome-only run is Rome-only too, whereas the full
  run's is global; the suspect segment is 98 % Rome so the loop is faithful for it, less so
  for the typed rows of the wide universe. Rome-only seed noise is ±16 s on `rmse` and ±3 s
  on `rmse_ex_dayoffset`.
- Compare against the seed-noise floor (±2 s on the full holdout) and read `rmse`, never
  `rmse_clean`, for submission decisions; for a tail change read `rmse_ex_dayoffset` next
  to it, because seven wrong-date rows carry over a third of the holdout's squared error
  and their hedging fluctuates between seeds.
- `prc audit --run <id>` shows where a run's squared error lives; a mixture run's
  `metrics.json` has a `mixture` block (universe size, label counts, AUC, and the universe
  RMSE of the mixture, of the regressor alone and of the oracle, which is the ceiling given
  the regressor) and `oof_predictions.parquet` carries `p_stale` per row. `prc leaderboard`
  shows the online score (`board`) next to the holdout once `upload_submission.py --result`
  has fetched it.
- Submission runs: `exp024_final_mixture.yaml --seeds 3 --keep-test`, then
  `prc blend --runs <the three ids>`, `prc submit --run <blend id>`, then
  `upload_submission.py --run <blend id> --version N`.

## 7. The gap to the top of the board (analysis of 2026-09-19)

The board reads 317 (v7) against a leader near 240. Everything below was measured on the
out-of-fold predictions of `2026-09-18T08-22-25__final_stale_only__77f71f79__s42` (exp028,
seed 42, holdout RMSE 364.2, 344,419 rows, total squared error 4.57·10¹⁰) unless it says
otherwise. The probe scripts were one-off pandas jobs against `data/interim/`; the numbers
are reproducible from the OOF parquet and the movement frame.

### 7.1 The squared-error budget

| holdout rows by absolute error | rows | share of squared error | note |
|---|---|---|---|
| above 50,000 s | 3 | 37.4 % | the three unannounced rows of section 2.6 (Paris 84,240 and 58,206; Rome 87,002, typed, stale by a schedule a day out, p_stale 0.04) |
| 20,000–50,000 | 5 | 12.4 % | four untyped wrong-date rows hedged at ~52,000 s, one stale row (45,791 s) at p 0.5 |
| 7,200–20,000 | 25 | 6.4 % | classifier misses in both directions |
| 3,600–7,200 | 95 | 4.8 % | |
| 1,200–3,600 | 1,352 | 10.1 % | genuine long taxis after a delay, under-predicted |
| up to 1,200 | 343,000 | 28.9 % | the bulk |

Read the other way: the classifier's universe (9,317 rows with a delta ≥ 2 h) carries 41.5 %,
the two Paris rows outside it 22.5 %, and the remaining 335,000 rows 36 % at an
RMSE-equivalent of 221. If every row with |error| above 3,600 s were perfect the holdout
would read 227.5; above 1,200 s, 195.8. The bulk (y below 7,200 and |error| below 3,600)
sits at 226.7: EDDM/LEMD/LSZH 172–175, EDDF/EHAM 176–181, LEBL 219, EGLL 233, LFPG 245,
LTFM 249, LIRF 382 (Rome's target has a standard deviation of 631 s even below two hours,
the other airports 300–470).

Inside the universe the classifier's false positives (p above 0.5 on a genuine row: 89 rows,
PGT/NBT/WZZ/RYR flights predicted at 9,000–20,000 s for a 500–1,400 s taxi) cost 16.9 % of the
total, and its false negatives (26 rows) 17.2 %, of which the Rome one-day row alone is 15 %.
Replacing the classifier by the true regime on every universe row except the wrong-date ones
moves the holdout from 364.2 to 352.6: **12 s is the classifier's whole ceiling** on this
holdout. It is badly calibrated in the honest sense (p 0.9–0.98 → 58 % stale; p above 0.98 →
63 %; p 0.5–0.7 → 24 %), but re-mapping p to the observed rate per bin makes the holdout
*worse* (384), because the delta-weighted p already carries the cost asymmetry: within a
bin, the rows with the largest deltas are the stale ones. A calibration that helps has to
condition on the delta (isotonic within delta bands, on nested out-of-fold probabilities).

The board's arithmetic. 317 on 344,841 rows is 3.47·10¹⁰ of squared error. A bulk at 225 is
1.75·10¹⁰ of it, so 2026's tail costs us about 1.7·10¹⁰: two rows at 85,000 s, or 170 rows
at 10,000 s, or 1,000 rows at 4,000 s. A score of 240 is 1.99·10¹⁰ in total. **No single
lever reaches it**: with the bulk unchanged, a perfect tail gives 222 at best on paper and
nothing like it in practice; with the tail unchanged, a perfect bulk gives 222 as well. The
leaders have both a bulk near 200 and a tail costing under 0.6·10¹⁰.

Where the bulk error is: it grows with the prediction (RMSE 119 in the lowest predicted
decile, 438 in the highest), it is largest at 22–00 UTC (250–320) and smallest at 05–06 UTC
(190), and July is worse than January (240 vs 209). Residuals of departures at the same
airport within ±15 min correlate at 0.19 (LIRF) to 0.38 (LTFM); the between-(airport,
hour) share of the bulk residual variance is 13.6 % against 3.7 % under independence, the
between-(airport, day) share 2.5 % against 0.2 %. So about a tenth of the bulk residual
variance is a state shared by the hour that no feature carries, worth up to 11 s on the
bulk if it were fully explained, realistically 3–5.

### 7.2 What the residual tests ruled out

Each idea was scored as a correlation with exp028's bulk residual (a feature with no
residual signal cannot help a model that already has these features).

| idea | result |
|---|---|
| NM's `AOBT_3_flt` as a near-copy of the block time (a leak the leaders might use) | no: the two disagree by a per-airport offset (LTFM +296 s, LIRF −118, EHAM +107 median) with a median absolute deviation of 120–190 s; only 9–26 % of rows agree within a minute; the model's bulk error correlates 0.53 with the proxy's error, so the model already leans on it as far as it is reliable |
| takeoffs on the same runway / at the airport, arrivals landed, during the flight's own taxi window `[AOBT_3, takeoff]` (queue position) | corr with the residual −0.006 to −0.013 (with the target 0.33, with the proxy 0.61: it is the proxy's length) |
| overtakes: flights that pushed back later (per NM) but took off earlier, and the reverse | −0.013 / 0.002 |
| neighbours' NM-proxy anomaly within ±15 min (local congestion as NM sees it) | −0.03 |
| arrivals' taxi-in anomaly within ±30 min | −0.02 on the Jan+Jul 2025 bulk (but see 7.3: that holdout has no disrupted day) |
| stand re-occupation as a lower bound on the taxi (an arrival blocks in at my stand before I take off; 9.7 % of departures, 26 % at EGLL; 40 % of the rows above 3,600 s) | the model already knows it: 0.9 % of such rows are predicted below the bound, and clipping them to it makes the holdout worse (364.2 → 364.7). The rule "gap below 20 min ⇒ a different aircraft" holds 96–99.9 % of the time, and the model's RMSE on those rows is 236 |
| block time = the inbound's in-block (a fourth regime) | 0.15 % of rows, below chance |
| wrong-date rows announced by the flight number's history (a missing previous-day record, a matching previous-day schedule) | nothing: they are rare flight numbers or the same day's flight delayed 19 h; the block simply carries the previous date |
| a quantised, default `AOBT_3` at LTFM/EDDM/LSZH (67/52/57 % of rows on three minute values) | the modal rows are the *better* measured ones (proxy error RMSE 191 vs 508 at EDDM); the modes are the unimpeded taxi, not a default. The modes per runway are unchanged between 2025 and 2026 |

Two things did show residual structure. `AOBT_3 − LOBT` (not a feature today) moves the
proxy's error from +1,137 s (AOBT more than 30 min before the last planned off-block) to
−900 s (more than an hour after); the model has both deltas but not their difference. And
long taxis recur by flight number: 69 flight numbers have three or more rows above 3,600 s
that are not stale (UAE25K, ETD11K, DAL113, DAL67, BAW354N, BAW348K…, rates of 2–5 %
against a base of 0.07 %); 29 % of such rows have a flight number that is long in two or
more months; the holdout's badly missed long rows have a flight-number long-rate four times
the base. 79 % of 2026 rows carry a 2025 flight number.

### 7.3 What 2026 looks like from the outside

The tail structure of the ranking set is the holdout's: 388 untyped NM-less rows (402 in
Jan+Jul 2025), 181 of them with a delta above 2 h (182), 22 above 8 h (16), 4 above 16 h (6);
9,987 typed rows above 2 h (8,912), 258 above 8 h (201), 47 above 20 h (33); 45 typed rows
with `LOBT == SCHED` and a delta above 2 h (37), 17 of them at EHAM, 10 at LFPG. Runway use
shifted (EGLL 09R 42 % of departures against 16 %; EHAM runways 09 and 18C appear; LTFM
16R), 21 % of flight numbers are new, stands and types are not.

**January 2026 had winter.** The arrivals, whose taxi-in times are complete in 2026, and the
METARs both show it, and Jan+Jul 2025 have no day like it:

| airport-day | arrivals' mean taxi-in anomaly | departures | METAR | our predicted departure anomaly |
|---|---|---|---|---|
| EHAM 2026-01-05 | +1,508 s (197 arrivals) | 208 (a normal day has ~600) | snow all of 4 Jan | +731 s (mean 1,455) |
| EHAM 2026-01-04 | +545 | 336 | snow | +838 |
| LFPG 2026-01-07 | +524 | 504 | snow 06–10 UTC | +917 |
| EHAM 2026-01-03 | +380 | 391 | | +744 |
| EGLL 2026-01-04 | +249 | 634 | | +407 |
| EDDF 2026-01-26 | +226 | 465 | snow overnight | +669 |
| LTFM 2026-01-18/19 | within normal | | continuous snow for two days | predicted daily means 1,770 and 2,012 (normal 1,220) |

The only 2025 analogue is LTFM on 23 Feb 2025, a training month: departures +1,468 s for an
arrival anomaly of +264. On the 33 airport-days of 2025 with an arrival anomaly above 150 s,
the departure anomaly ran at about 2.5 times the arrival anomaly; applied to the table it
says EHAM on 5 Jan should sit thousands of seconds above normal, far beyond anything the
data contains. The model has no weather features in exp028 and infers these days from the NM
delay deltas only. Whatever the truth, a few hundred departures with a mean error of a
thousand seconds and a spread as large is 10⁹ of squared error, several seconds of board
RMSE, and the holdout cannot see it.

**The PRU publishes the answer's monthly mean.** Eurocontrol's Performance Review Unit
posts the *additional taxi-out time* per airport-month as open CSV
(`https://www.eurocontrol.int/performance/data/download/csv/taxi_out_additional_time_2026.csv`,
2025 likewise; columns `TOTAL_REF_NB_FL`, `TOTAL_REF_TIME_MIN`, `TOTAL_ADD_TIME_MIN`).
Reference plus additional time over the referenced flights is the mean taxi-out of the
month, computed from the same airport data flow this challenge uses: for 2025 it matches the
training data's monthly mean (rows below 7,200 s) within a median of 7.5 s over the 120
airport-months, within 2 s at EGLL every month. The caveat is `TOTAL_REF_NB_FL / TF`, the
share of flights with a reference: it drops in disrupted months and the excluded flights are
the long ones (LTFM Feb 2025: PRU 990 s on 79 % of flights against a data mean of 1,260).
For 2026:

| airport-month | PRU mean (ref share) | 2025 same month, data | our prediction mean | reading |
|---|---|---|---|---|
| EGLL Jan | 1,335 (0.99) | 1,327 | 1,316 | right |
| EGLL Jul | 1,344 (0.99) | 1,426 | 1,348 | right, and the model followed a −83 s year-over-year change |
| EHAM Jan | 982 (0.86) | 811 | 896 | **under-predicted by at least 90 s over 16,232 rows**; the true mean is above 982 because the excluded 14 % are the long ones |
| LTFM Jan | 1,089 (0.83) | 1,041 | 1,247 | unverifiable: 17 % excluded (7 % in Jan 2025); consistent with the snow days |
| EDDM Jan | 755 (0.67) | 870 | 1,023 | unverifiable: 33 % excluded |
| EDDF Jan | 854 (0.85) | 885 | 1,008 | unverifiable |
| LEBL Jan / Jul | 883 / 967 (0.95) | 887 / 1,008 | 926 / 1,007 | 30–40 s high |
| LEMD Jan | 987 (0.85) | 986 | 1,051 | 50 s high |
| LIRF Jul | 1,214 (0.93) | 1,297 | 1,277 | 50 s high |

Across the twenty airport-months the year-over-year change of our prediction mean correlates
0.69 with the PRU's and 0.67 with the change of the NM proxy's mean; the PRU's change
correlates 0.76 with the proxy's. So the level drift is mostly real and mostly followed. A
level check of every submission run against this file (predicted mean per airport-month
against the PRU mean, flagged where the reference share is 0.9 or more) costs nothing.

### 7.4 The validation blind spot

Jan+Jul 2025 is the seasonal analogue of the test months and it decides the tail correctly
(section 1), but it contains no winter disruption, and the same configurations score 205–208
on an Apr+Oct holdout (`-o 'validation.holdout_months=["2025-04","2025-10"]'`) against
275–278 without the wrong-date rows on Jan+Jul: the January half is decided by a handful of
rows, the July half by heat and traffic. Any feature meant for snow, de-icing, or
cancellations has to be judged on a holdout that contains them: Feb 2025 (LTFM), or the
union of the 2025 airport-days whose arrival anomaly exceeds 150 s. `exp026` (weather and
geometry on the mixture) reading 8 s worse on the Jan+Jul headline and 1.3 s worse on the
honest number is not evidence about snow days; it is seed noise on a holdout without them.

### 7.5 Ideas to check, by expected value

1. **Disrupted days in 2026 (EHAM 3–7 Jan, LFPG 7 Jan, LTFM 18–19 Jan, EDDF 26 Jan).**
   Build the winter holdout first, then measure, in this order: the weather group
   (`wx_snow`, `wx_deice`, `wx_snowfrz_hours_24h` exist); a live ground-condition block from
   the arrivals (mean and max taxi-in anomaly of the arrivals landed in the last 60 min, the
   count above twice their stand-runway median; hourly correlation with the departure anomaly
   is 0.25–0.37 over 2025, and the top 1 % of hours carry a +250 s departure anomaly); a
   cancellation index from the movement frame itself (departures in the hour and day against
   the same weekday's median: 208 departures on 5 Jan at EHAM is a signal no other feature
   gives). Consider sample weights for disrupted hours or a separate disrupted-conditions
   model: 2025 holds about one severe snow day, and a GBDT fitted on two million rows will
   not extrapolate from it. Expected value: unmeasurable locally, several seconds on the
   board if EHAM 5 Jan is what the arrivals say it is.
2. **The classifier (ceiling 12 s on the holdout, probably a comparable share of 2026's tail).**
   Calibrate conditional on the delta with nested out-of-fold probabilities; add the
   flight-number-level stale rate and the count of rows the flight number has in the year
   (the wrong-date and stale rows are mostly flight numbers seen 1–20 times); test whether
   stale records cluster by airport-day (13–14 July 2025 at Rome holds five of the worst
   rows: a feed outage would announce itself as the day's count of untyped NM-less rows,
   which is computable for 2026); the same stand's other departures that day.
3. **The bulk's shared hourly state (up to 11 s, realistically 3–5).** Airport-hour aggregates
   of NM's own delays (mean `aobt3_minus_eobt1` and share of `AOBT_3 > LOBT` among the hour's
   departures), runway-configuration change indicators (time since the dominant departure
   runway changed, share of the hour's departures on it), the live arrivals block from item 1.
   The daily PRU *ATC pre-departure delay* file
   (`atc_pre_departure_delays_2026.csv`, daily per airport, empty for EGLL/EHAM/LTFM) correlates
   0.4–0.6 with the daily mean taxi-out at EDDF/EDDM/LIRF but the day level is only 2.5 % of
   the residual variance: cheap, small.
4. **Explicit NM cross-deltas**: `aobt3_minus_lobt`, `lobt_minus_eobt1`, `aobt3_minus_iobt`
   (unclipped), and a flight-number long-rate encoding (fold-dependent, like
   `stale_target_stats`: rate of y above 3,600 and median y per `FLIGHT_mvt`, smoothed). Both
   target the 1,200–7,200 s error band (15 % of the squared error). One ablation each.
5. **Seed and config averaging**: the three-seed blend gave 1 s on the board (318.0 → 317.0);
   a blend across configs (exp022's residual base, exp028, and a winter-aware run) is the
   cheapest remaining 2–4 s.
6. **The level check** against the PRU monthly means after every submission run (7.3).

External data, assessed:

- **Eurocontrol PRU open CSVs** (`https://ansperformance.eu/csv/`): taxi-out additional time
  (monthly, the level check above); ATC pre-departure delays and all pre-departure delays
  (daily per airport, partial coverage); airport arrival ATFM delays `apt_dly` (daily). All
  cover Jan and Jul 2026. Public domain-style open data, cite the source.
- **ATC industrial action.** The worst LFPG rows of July 2025 fall on 3–5 July, the French
  ATC strike; a hand-made calendar of strike days for 2025 and Jan/Jul 2026 is an hour's work
  and a legitimate day-level feature.
- **NOTAMs.** No free historical archive exists (the FAA API serves active NOTAMs; the
  Eurocontrol EAD needs an account), and the effect a NOTAM would explain, a closed runway or
  taxiway, is already visible in the data as runway use by hour. Low value for the cost.
- **ADS-B surface data (OpenSky).** Ground positions would show the actual pushback of every
  2026 departure and collapse the bulk error. The challenge page asks for open data and an
  open-sourced solution and says nothing else; whether observing the target's proxy is within
  the spirit is a question for the organisers before any work is spent, and it would not touch
  the stale-schedule tail at all.
- Public and school holidays per country: cheap, tiny; the calendar block already carries
  month and weekday.

### 7.5a Status of the backlog (2026-09-20)

Implemented and tested (`pytest -q`: 260 tests, plus `test_features.py` on the data), the
sweep running unattended over the night of 20 September:

- **Winter visibility.** `prc/disruption.py` marks an airport-day disrupted when its arrivals'
  mean taxi-in anomaly (against the (airport, runway, stand) median, `prc/features/_arrlib.py`)
  exceeds 150 s with at least 20 arrivals; every run reports `rmse_disrupted` /
  `n_disrupted_val`, `prc audit` lists the days (34 in 2025, 14 in Jan/Jul 2026; EHAM 2026-01-05
  at +1,624 s, LFPG 2026-01-07 +562, EHAM 2026-01-04 +551). On Jan+Jul 2025 the mask holds
  5,290 rows (Heathrow's congestion days): exp028 reads 417 there against 363 elsewhere. The
  winter holdout `exp029_winter_check.yaml` (Feb + Dec 2025) reads **248.2 overall and 789 on its
  4,509 disrupted rows with a bias of −161 s** (seed 42): the model under-predicts by
  minutes on exactly the kind of day the 2026 ranking months contain.
- **PRU open data.** `prc fetch-pru` downloads and normalises the four Eurocontrol datasets;
  `pru_daily` is a feature block; `prc levelcheck --run <id>` compares predicted monthly means
  with the PRU's (7.3) and, on v7, flags only LIRF July (+62 s at a reference share of 0.93).
  The monthly mean is a benchmark only, never a feature (user decision).
- **The context blocks**, one per idea in its own group: `stand_occupancy`,
  `flight_number_history`, `flight_number_context`, `runway_sequence`, `taxi_window_counts`,
  `pier_pushback`, `zone_load`, `arrival_conditions`, `traffic_deviation`, `nm_cross_deltas`,
  `nm_hour_context`, `stale_day_context`, `key_target_stats`, `wx_deice_load`, `pru_daily`.
  The window primitives gained per-row windows, `window_reduce` and `neighbour_gap`.
- **The measurement.** `prc ablate --mode add-to-base --emit-configs` writes "exp028 plus one
  block" variants; `prc queue` runs them as child processes. The night's queue: the winter
  seed floor (3 seeds of exp029), then 16 winter variants (`exp029b_winter_calibrated` first),
  then 16 Jan+Jul variants (`exp029_calibrated` first). Read with
  `prc leaderboard --name plus_ --sort rmse_ex_dayoffset` and `--name calibrated`, against
  the floors: Jan+Jul ±0.5 s on `rmse_ex_dayoffset`, winter floor from the three seeds.
- **Calibration.** `model.mixture.calibration: isotonic` fits one isotonic curve per band of the
  row's largest candidate delta on nested out-of-fold probabilities of the fit universe (rows
  weighted like the classifier); `p_stale_raw` is kept next to `p_stale`, and the `mixture`
  diagnostics carry `rmse_universe_mixture_uncalibrated` and the per-band log-losses.

Results go into 7.5b once the queue has run; the submission config `exp030` is composed from
the winners, checked on both holdouts, trained with three seeds, blended, level-checked and
submitted (`DEVELOPER_GUIDE.md` Appendix A, Notes, "The context blocks"; README until 14 Oct).

### 7.5b Results of the sweep, and v9 (2026-09-23)

Every block was measured as "exp028 plus one block", seed 42, on `rmse_ex_dayoffset` against the
seed-42 floors: Jan+Jul 275.8, the Feb+Dec winter check 248.1, and 789.1 on the winter check's
disrupted rows (table: `runs/sweep_2026-09-20b_decision.csv`). The rule: a block is in when it
beats a floor by more than 0.5 s on one holdout without losing more than 0.5 s on the other.
Deltas in seconds, negative is better:

| block | Jan+Jul | winter | winter disrupted | verdict |
|---|---|---|---|---|
| `nm_hour_context` | −0.1 | −7.6 | −95 | in |
| `taxi_window_counts` | −0.3 | −1.0 | −10 | in |
| `nm_cross_deltas` | +0.1 | −1.8 | −0.3 | in |
| `runway_sequence` | +0.3 | −1.3 | −3.9 | in |
| `key_target_stats` | −0.8 | +0.2 | +5 | in |
| `wx_deice_load` | +0.8 | −8.4 | −51 | out: loses on Jan+Jul |
| `arrival_conditions` | +1.0 | −3.8 | −38 | out: loses on Jan+Jul |
| `pru_daily` | +0.2 | −5.3 | −62 | out by default: published after the day |
| `flight_number_context` | +2.0 | −2.3 | −4.3 | out |
| `pier_pushback` | +2.2 | −0.6 | +0.9 | out |
| `zone_load` | +0.4 | −0.3 | −4.5 | out |
| `stale_day_context` | +0.2 | −0.3 | +1.5 | out |
| `traffic_deviation` | +0.9 | +1.0 | −1.7 | out |
| `stand_occupancy` | +4.0 | +0.4 | +5 | out |
| `flight_number_history` | +6.5 | +5.6 | +8 | out: hurts on both |

Calibration (`exp029_calibrated`) lost 2.8 s on Jan+Jul and gained nothing on winter, so it is out.

`exp030_final_context.yaml` is exp028 plus the five winners, trained with refit on all months:

| run | rmse | without wrong-date rows | Jan+Jul disrupted rows |
|---|---|---|---|
| seed 42 | 349.0 | 277.5 | 408.2 |
| seed 43 | 340.6 | 273.2 | 409.0 |
| seed 44 | 350.1 | 278.3 | 409.7 |
| blend (v9) | 344.4 | 274.8 | 406.7 |
| exp028 blend (v7) | 367.6 | 274.8 | 417 (seed 42) |

The Jan+Jul holdout sees no gain: the blend's 274.8 equals v7's, and the seeds spread over 5 s.
The board moved 317.0 → **311.8**, the largest step since v6. The gain therefore sits in 2026 rows
the Jan+Jul holdout does not resemble. January 2026's snow days (7.4) are the likeliest place,
since the winners' largest effects were on the winter check's disrupted rows, but nothing here
isolates it. The level check flags only LIRF July (+69 s; v7 +62 s). The January biases of
+126 to +248 s at EDDF, EDDM, LFPG, LSZH and LTFM sit where the PRU's reference share fell below
0.9, so the PRU mean is a floor there, as for v7.

Next: `wx_deice_load` and `arrival_conditions` were kept out only for losing about 1 s on
Jan+Jul, a holdout that has just proved blind to a 5 s board gain. exp030 plus those two is the
direct test of whether the winter side keeps paying on the board.

### 7.6 Is gradient boosting the right tool

Yes, as the core. The task is tabular regression on two million rows whose signal is
cross-row context, and section 7.2 shows that the physically motivated features (queue
position, overtakes, stand occupancy, local NM congestion) have already been absorbed by
CatBoost through the windows and keys it has. What the trees cannot do has been handled
outside them, and that pattern should continue:

- **Regimes with exact answers** (a block time that is the schedule, the wrong date) are a
  classification problem times an exact base, not a regression target; the mixture does the
  MSE-optimal hedging and its remaining cost is the classifier's discrimination (7.1). A
  distributional or quantile model would not help: RMSE rewards the conditional mean, which
  the mixture computes.
- **Joint inference of all off-block times per airport-day** (the stand-occupancy chain, the
  FIFO order of the runway, one aircraft per stand) is the structured alternative to
  row-wise prediction. Its constraints are already respected by the row-wise model (0.9 %
  bound violations, no overtake signal), so the marginal value is small for a large cost.
- **A sequence model over the movement stream** (a transformer per airport-day) could learn
  the shared hourly state of 7.1, worth at most 11 s on the bulk. Hourly aggregate features
  and the live arrivals block reach most of that for a fraction of the effort; a sequence
  model is a stacking component for later, not a replacement.
- **Queue simulation** is subsumed: the takeoff sequence is known, and counts in windows
  around it are what a simulation would consume.
- **Rare conditions** (snow, cancellations) are the one place the GBDT's inductive bias
  fails: it has seen one severe day. Reweighting, a dedicated disrupted-conditions model, or
  hand-built rules calibrated on Feb 2025 are all more promising than a bigger model.
- The highest-value non-modelling work is what this section did: understanding what the
  recorded block time *is* per airport (its offset against NM differs by airport, which means
  different definitions), and building a holdout that contains the conditions of the test
  months.

## 8. The push of 2026-09-24: public data, the feature set, the pipeline

Goal: close the gap from v9 (311.8) toward the leaders (~235). The board judges 2026; locally, the
new **bulk metric** (`rmse_bulk`: holdout rows with a target below 3,600 s) is the instrument. Over
exp030's three GPU seeds it reads 223.49 / 223.42 / 223.70, a noise of ±0.15 s, against ±2 s for
`rmse_ex_dayoffset`, whose 149 multi-hour rows carry half the squared error. Every run and blend now
reports it (`metrics.json` `overall_bulk`, `index.csv`, leaderboard, queue summary, ablation
`__bulk`). Uploads to the scoring bucket need the user's permission in auto mode; the submission files
are built locally.

### 8.1 Public operational data, assessed

| source | what it adds | status |
|---|---|---|
| METAR (IEM) | weather at and before takeoff | in exp031 (with `wx_deice_load`, `arrival_conditions`): bulk 223.49 → 222.33 on Jan+Jul, where the sweep's noisy metric saw nothing. The archive strips runway-state groups: 0 of 263,018 reports carry one, so there is no runway-contamination source in METAR |
| ERA5 reanalysis (Open-Meteo, CC BY 4.0) | snow on the ground, snowfall over 3/24 h, freezing precipitation, ground temperature, overnight minimum, frost hours | new: `python -m prc fetch-era5`, block `era5_surface`, exp032. January 2026 snowfall 32 cm at EHAM (13 cm on the ground; 2025's maximum was 3 cm), 19.5 cm EDDM, 16 cm LSZH, 12 cm LEMD and LTFM |
| OSM / OurAirports geometry | stand-runway distances | exists, not in the submission line. 2026-specific: Frankfurt's Terminal 3 (piers J and H, 1,700 July 2026 departures, matched in OSM) and Munich stands 103-108 (1,200, not in OSM) had under ten departures in 2025; the model predicts them 45 s (EDDF) and 167 s (EDDM) above the airport's other stands. Worth at most ~1 s of board RMSE even at a 300 s miss |
| PRU daily delays at the airport | ATC pre-departure and airport ATFM minutes | measured in the sweep (winter −5.3 s); published after the day, out by policy |
| PRU `apt_dly` at the destination | arrival ATFM delay of the destination that day (290 airports, 67 % of 2026 departures) | residual test on exp030: correlation −0.03 with the bulk residual; the same signal built live from NM (mean `AOBT_3 − EOBT_1` of the hour's flights to that destination) −0.024. Small |
| OpenSky ADS-B | the actual pushback, observed | anonymous access to historical flights is refused (HTTP 403); needs an account, and observing the target is a rules question for the organisers |
| NOTAMs | closures | no free archive; runway closures show in the data as runway use |

### 8.2 What the feature review found

- **Stale block times are not only a multi-hour tail.** Among departures 30-120 min behind schedule,
  the target equals the schedule delta within a minute on 9.1 % at LIRF, 3.4 % EGLL, 2.3 % LTFM,
  under 1.1 % elsewhere. At Rome it is a carrier trait (one carrier 99.7 %, several 24-32 %,
  KLM/Swiss/TAP under 1 %) and a pier trait (pier 9 22 %, pier 4 2 %), peaking in July (44 per 1,000
  departures at LIRF against 19 in January; LEBL 9.6 against 0.6). The regressor meets these rows as
  taxi times: in exp030, 572 LIRF bulk rows with |error| > 900 s carry 65 % of LIRF's bulk squared
  error, stale rows predicted as taxis and genuine taxis hedged up by an hour, and LIRF July reads
  506 s on the bulk against 150-260 elsewhere. Perfect handling of just the stale-matching 30-120 min
  rows would move the holdout 344.4 → 336.6 and the bulk 222.2 → 216.7. The tolerance must be tight:
  41,000 rows a year sit 5-10 min short of the delta (a slightly late pushback and a 20 min+ genuine
  taxi), so the mixture's 600 s would mislabel them; 60 s keeps ~10,000. exp034 widens the mixture to
  30 min deltas with a 60 s tolerance; exp034b adds the LOBT delta as a second candidate (1.5-2.2 % of
  delayed rows at EGLL, LEBL, LFPG, LIRF, LTFM match it). At LEBL it is the main one: of the 255 July
  bulk rows exp034 misses by more than 900 s (38 % of LEBL's July squared error, 223 of them
  under-predicted), 55 % have a target equal to the LOBT delta within a minute and 40 % the schedule
  delta; exp034 gave them a stale probability of zero, having only the schedule candidate.
- **The time-position columns extrapolate.** `takeoff_epoch` (core_keys, 14th of 149 by gain) and
  `days_since_epoch` put all of 2026 beyond the training range, so every split on them sends July 2026
  down the late-December-2025 branch; `day_of_year` and `week_of_year` let the refit copy a 2025
  date onto the same 2026 date. The calendar block's permutation importance is negative (−3.9 s).
  `features.drop_columns` (new) removes columns from the matrix while the blocks keep computing them;
  exp033 drops the four. Judged on the board and on a backward-extrapolation holdout
  (`strict_forward`, train Mar-Dec, validate Jan+Feb).
- **Wake separation and the runway queue** (`wake_queue`, exp035): the ICAO minimum to the previous
  takeoff (2 min behind a Heavy, 3 min Medium/Light behind a Super, 2 min Light behind Medium), the
  slack against it, and the continuous queue each takeoff sits in; a flight cannot have waited at the
  holding point longer than its queue has lasted. Residual test: the queue's age and depth move the
  bulk residual ~20 s across deciles; the slack alone shows nothing, since `runway_sequence` already
  names the preceding wake and `taxi_window_counts` counts the queue by NM off-block order.
- **Weather hours are a small share of the bulk error** on Jan+Jul: thunderstorm hours 2.4 %, fog 2.7 %,
  snow 0.7 %. The long under-predicted taxis at LTFM and LFPG (bias −600 s on the worst rows) have an
  NM `AOBT_3` implying an ordinary taxi and fall on days without thunderstorms (LTFM 13 July 2025:
  none of 48 METARs): flow or ATC holds after pushback that no public source records per flight.
- **No extrapolation term for 2026's snow days.** Per airport-hour in 2025 the departure anomaly
  grows about one-to-one with the previous hour's arrival taxi-in anomaly at small values and at half
  that beyond 300 s (arrival +621 → departure +316). exp031 already predicts EHAM 5 Jan 2026 (arrivals
  2,050 s mean taxi-in) at 1,870 s against v9's 1,511, above what that relation implies.

### 8.3 Results

Jan+Jul holdout, seed 42 (bulk = rows with a target below 3,600 s, noise ±0.15 s):

| run | what | rmse | without wrong-date rows | bulk | LIRF July bulk |
|---|---|---|---|---|---|
| exp030 | v9's config | 349.0 | 277.5 | 223.49 | 506.0 |
| exp031 | + `wx_deice_load`, `arrival_conditions` (with `wx_now`, `wx_past`) | 349.2 | 273.8 | 222.33 | 516.2 |
| exp034 | exp030 + mixture from 30 min deltas, 60 s tolerance | 347.4 | 270.1 | **219.68** | 474.0 |
| exp036 | exp031 + the same mixture change | 352.9 | 268.0 | **218.56** | |
| exp036 + ERA5 | + `era5_surface` | 335.7 | 267.9 | **217.68** | |

The weather gain sits at EGLL, LTFM and LFPG in January (207 against 212, 208 against 217, 232 against
237), the mixture's at LIRF in July; exp036 combines them and is the base the rest is measured on.
exp031's level check: year-over-year correlation with the PRU 0.75 (v9 0.71), LIRF July +50 s (v9 +69),
EHAM January mean 929 s (v9 896, PRU at least 982); EHAM 5 Jan 2026 predicted at 1,870 s (v9 1,511).

exp036's headline `rmse` (352.9) is the wrong-date rows' seed noise (±5-8 s on seed 42 alone); the wrong-date-free
number and the bulk both improve on every earlier run, so the weather and the mixture gains add.

ERA5 improves the bulk by 0.9 s even on Jan+Jul (its target is January 2026's snow, which this holdout lacks).

**v10 (exp037 = exp036 + ERA5, seed 42 alone): board 306.6**, 5.2 s under v9's three-seed blend (311.8), with
about 1 s of single-seed penalty still in it. Holdout: bulk 217.7, without wrong-date rows 268.2. The level
check flags nothing for the first time (Rome July, +50 to +69 s in every earlier candidate, is gone); the
year-over-year correlation with the PRU is 0.76 (v9 0.71).

(pending: the other additions, the date columns, the parameter tests and the diagnostics are
queued in `runs/goal_C_resume_launcher.ps1`, paused on 24 Sep at the user's request to free the machine)

## 9. Literature, competitors and the feature plan (2026-09-24)

Written after v10 (306.6) to decide what to test next without spending GPU hours on guesses.

### 9.1 What the literature says drives taxi-out time

| factor | evidence | our coverage |
|---|---|---|
| Distance stand → runway | "the dominant factor that determines the unimpeded taxi-out time" (NASA, Charlotte); distance and the sum of turning angles lead the regression of Ravizza, Atkin, Maathuis and Burke (Arlanda, Zürich, Heathrow) | per stand×runway target stats (median, p90); OSM route length, runway crossings and stand zones exist (`group:geometry`) but are not in the submission line; turning angles are not computed |
| Unimpeded reference | EUROCONTROL's indicator: P10 per (airport, stand, departure runway) over a rolling year, valid with at least 10 flights at or below it; NASA uses the same P10 per gate×spot×runway as a feature | median and p90 only |
| Runway queue | Idris, Clarke et al. (Boston): runway configuration, airline/terminal, downstream restrictions and takeoff queue size are the main causes; the queue is estimated from departures on the surface at pushback minus passing | `runway_queue`, `taxi_window_counts` (takeoffs between NM off-block and takeoff, overtakes), `runway_sequence`, `wake_queue` (built, untested) |
| Departure fix / downstream restrictions | NASA Charlotte: departure fix is a selected variable because miles-in-trail restrictions hold departures on the ground; Idris: downstream restrictions | **not covered**: nothing encodes the departure direction |
| Surface traffic | NASA: departures on the surface by runway, arrivals on the surface by concourse; SESAR GNN (Lim et al., Atlanta): congestion along the flight's own route beats airport-wide counts ("congestion in a localised area might not translate to increased taxi times" elsewhere) | airport and pier counts, `zone_load` (measured: no gain); nothing route-level |
| Weather, de-icing | visibility, wet tarmac, wind, precipitation (all reviews); EUROCONTROL lists de-icing positions among its factor categories. Its indicator **excludes flights de-iced after off-block** (ATXOT documentation, step 1), so the PRU's falling reference share in January 2026 (EDDM 0.67, LSZH 0.74, LFPG 0.79, LTFM 0.83, EDDF/EHAM 0.85-0.86, against 0.77-0.95 in January 2025) counts, above all, flights de-iced during the taxi-out | METAR, ERA5 and `wx_deice_load` are in the line since v10 |
| What no data holds | controllers' tactical actions: hardstand holds, EDCT/call-for-release, intersection departures, re-routes. NASA's machine-learning models placed only 65-74 % of departures within ±5 min at Charlotte | the long under-predicted taxis at LTFM/LFPG of section 8.2 |

### 9.2 What the competition looks like (public leaderboard and GitHub, 24 Sep 09:48 UTC)

186 teams, 2,629 submissions. First 234.5, tenth 248.4, 25th 268.6; **adventurous-radio 306.6, 94th**. The
ranking uses each team's best submission, so a worse upload never costs anything.

- `javidmardanov/PRC-Data-Challenge-2026` (268.6, 14th on 15 Sep): TimesFM airport-level forecasts, seasonal
  CatBoost blends, a mixture of experts for missing timestamps, arrival stand/runway context, "targeted
  corrections for extreme timestamp estimates"; its latest step added **a LOBT residual expert with
  airport-specific blend weights** and a winter-only Rome flight-identity correction. Local tuning 260 on Jul+Nov
  2025, board 268.6.
- `ChandanHegde07/OpenAir` (277.0): per-airport fits of two clocks (`MVT − AOBT` and `AOBT − EOBT`), stand×runway
  geometry, a 30-min neighbour-delay "disruption state", for unmatched LIRF flights a decomposition
  taxi = schedule delay − gate hold, and a pull toward `MVT − AOBT` on ~12 % of matched rows. On the same Jan+Jul
  2025 holdout: NM-matched rows ~235, unmatched LIRF ~3,450.
- `ahmetabdullahgultekin/prc-taxiout-2026`: EUROCONTROL's P10 reference plus a residual model, METAR, OurAirports.
- `victoralcadi/prc-data-challenge-2026`: LightGBM; "weather observations are the highest-value addition".

**Ours on the same split** (exp037 seed 42 OOF): NM-matched 246.0, **222.6 without its one wrong-date row**; unmatched
LIRF 4,172 (2,829 without its five wrong-date rows); unmatched rows elsewhere are far worse than matched ones
(EGLL 1,050, EDDM/LSZH ~600, EHAM/LTFM ~565, LFPG 1,989 without its wrong-date row). So the bulk model is as good
as that of teams scoring 268-277, yet the board reads 306.6: **the 30-40 s we lose sit in 2026 behaviour the
Jan+Jul holdout cannot show.** The candidates, in order of suspicion:

1. the time-position columns (`takeoff_epoch`, `days_since_epoch`, `day_of_year`, `week_of_year`), which put every
   2026 row outside the training range (section 8.2); no competitor mentions absolute-time features;
2. the tail regimes on 2026 rows: the classifier's stale probability on the unmatched and untyped segments
   (in the holdout it gives p < 0.5 to untyped Rome rows whose target equals the schedule delta within 5 s,
   predicting 30-50 % of it);
3. de-icing in January 2026 (9.1), now addressed by METAR + ERA5 but not validated on a month like it.

### 9.3 The plan, ranked by expected value per GPU hour

Tier 1: 2026-specific; only the board can judge; the configs exist.

1. **exp037 without the four time-position columns**
   (`-o features.drop_columns=[takeoff_epoch,days_since_epoch,day_of_year,week_of_year]`), seed 42 with refit,
   upload as v11. One hour and one upload; the cheapest test of the biggest suspect.
2. **LOBT as a second stale candidate** (`-o model.mixture.bases=[sched_delta_full,lobt_delta_full]`) on exp037:
   Barcelona's July misses match the LOBT delta (8.2) and the 14th-placed team's last step was a LOBT expert.
   One local run (LEBL July bulk), then the board.

Tier 2: new features, judged on the bulk metric first (±0.15 s), then the board.

3. **Departure direction** (the literature's departure fix): bearing and great-circle distance from the airport to
   the destination (OurAirports coordinates, already downloaded), and counts of same-sector departures within
   ±15/30 min (routes sharing a SID and its in-trail spacing). Residual test on the OOF first (minutes, no GPU).
4. **Unimpeded P10 per stand×runway** (fold-safe, EUROCONTROL's definition, with its count), next to the existing
   median and p90.
5. **De-icing specifics**: first departure of an aircraft after an overnight stand (stand-arrival gap ≥ 6 h) under
   ERA5 frost; aircraft size under de-icing conditions; OSM de-icing pads (a new Overpass query) and the detour
   stand → pad → runway.
6. The built and untested blocks: `delay_stale_stats` (carrier/pier stale rates), `wake_queue` (ICAO minima and
   runway queue runs).
7. **An expert for NM-unmatched rows**: their errors are three to five times the matched rows' at every airport;
   the 14th-placed team runs a separate expert for missing timestamps.

Tier 3: larger builds with uncertain value.

8. Route-level congestion along the OSM path (the SESAR GNN result): occupancy of shared taxiway edges by the
   other flights' projected routes.
9. The sum of turning angles along the OSM route (Ravizza et al.); mostly absorbed by stand×runway statistics.
10. An `MVT − AOBT3` regime: 10.2 % of bulk rows lie within 30 s of it (our RMSE on them 136); a perfect
    identification would move the bulk 217.7 → 213.3, but nothing marks those rows (the AOBT3 seconds field is
    zero on 97 % of them and on 96 % of the rest), so it is probably the centre of a continuous error distribution.

Not recommended: destination ATFM delays (residual correlation −0.03), flight-number history (+6.5 s in the sweep),
OpenSky (needs an account and a rules decision), NOTAMs (no archive).

Sources: [NASA, Taxi-out time prediction at Charlotte](https://ntrs.nasa.gov/citations/20180002162);
[Ravizza, Atkin, Maathuis, Burke, JORS 2013](https://link.springer.com/article/10.1057/jors.2012.123);
[Atkin, Burke, Ravizza, Heathrow (MAPSP 2011)](https://people.cs.nott.ac.uk/pszja/papers/MAPSP2011.pdf);
[Idris, Clarke et al., Queuing model for taxi-out time estimation](https://arc.aiaa.org/doi/10.2514/atcq.10.1.1);
[Lim, Tan, Lilith, Alam, Variable taxi-out time prediction using GNNs (SIDs 2021)](https://www.sesarju.eu/sites/default/files/documents/sid/2021/papers/SIDs_2021_paper_78.pdf);
[EUROCONTROL, Additional taxi-out time indicator (2023)](https://ansperformance.eu/library/ATXOT_indicator_documentation_mar23.pdf);
[Herrema et al., CDG taxi-out model](https://doi.org/10.2514/1.I010502);
[leaderboard data](https://prc-leaderboard.fly.dev/data.json);
[javidmardanov](https://github.com/javidmardanov/PRC-Data-Challenge-2026), [OpenAir](https://github.com/ChandanHegde07/OpenAir),
[gultekin](https://github.com/ahmetabdullahgultekin/prc-taxiout-2026), [victoralcadi](https://github.com/victoralcadi/prc-data-challenge-2026).

### 9.4 Pre-tests (2026-09-25, CPU only; `runs/pretest_2026-09-25.py`)

Residual tests on the holdout's non-wrong-date rows outside the mixture universe: HistGradientBoosting on the
clipped residual, 5-fold grouped by day, control columns against control plus the candidate. Gate: 0.15 s.

| candidate | holdout | rows | control CV | + candidate | gain | verdict |
|---|---|---|---|---|---|---|
| A departure direction (bearing, distance, sector, turn off the runway, same-sector departures ±15 min) | Jan+Jul (exp037) | 206,599 | 148.37 | 148.31 | +0.06 s | fail: not built |
| B unimpeded P10 per stand×runway (EUROCONTROL rule, hierarchy fallback, spread to the median) | Jan+Jul (exp037) | 206,599 | 148.37 | 148.31 | +0.05 s | fail: not built |
| C cold soak (stand idle 6 h+ × frost, snow, METAR de-icing; pad queue; heavies) | Feb+Dec (exp028 + wx_deice_load), all rows | 303,564 | 208.50 | 209.36 | −0.86 s | fail overall |
| C, disrupted airport-days only | Feb+Dec | 4,165 | 443.74 | 442.54 | +1.19 s | pass on the days it is meant for |

Every candidate column was populated (99.7-100 %); their correlations with the residual stay below 0.03.
Direction and P10 are already carried by `ades`/`ades_country` and the stand×runway median (P10 correlates
0.976 with it). C helps only on disrupted days and costs elsewhere in a small residual model; it is built
(`prc/features/deicing.py`, group `deicing`) and judged by the GPU winter-holdout step, where the full model
decides whether the gain on snow days survives.

E, the unmatched-row expert, computed exactly offline (exp037's seasonal matrix from cache; a CPU CatBoost on
the mixture's 16,380 regressor rows without an NM match, early-stopped on 5,199 holdout ones; the mixture
recomputed as y_pred + p_taxi · w · (y_expert − y_reg) on those rows):

| variant | rmse | without wrong-date rows | bulk | unmatched bulk, not LIRF |
|---|---|---|---|---|
| exp037 | 337.32 | 268.20 | 217.69 | 323.9 |
| expert, early-stopped, w 0.5 | 337.06 | 267.88 | 217.40 | 312.5 |
| expert, early-stopped, w 1.0 | 337.20 | 268.04 | 217.62 | 324.0 |
| expert, 1,000 rounds, w 0.5 | 337.13 | 267.94 | 217.41 | 313.1 |

At most −0.32 s without wrong-date rows and −0.29 s on the bulk, with the unmatched non-LIRF bulk down 3.5 %:
under the gate (−0.5 s, or −0.3 s with −10 %), and optimistic, since it early-stops on the holdout. Not built.
The unmatched rows' error is not a model-capacity problem; they lack the NM times that carry the signal.

D, the de-icing pads probe (Overpass, `runs/probe_deicing_pads_2026-09-25.py`): pads are mapped as `aeroway=apron`
ways at 3 of the 6 winter airports only: EDDM six ("Flugzeug Enteisungsfläche 13/14/15, DA 1/2/3"), LSZH two
("De-Icing Pad C/F"), LTFM five ("Deicing 1-5 Apron"); none at EDDF, EHAM, LFPG (nor EGLL, LEMD). Under the gate
(4 of 6): not built. Hand-placing the missing pads from the AIP charts would be the way to complete it.

### 9.5 GPU results (2026-09-24)

| step | run | rmse | without wrong-date rows | bulk | board |
|---|---|---|---|---|---|
| base | exp037 s42 (v10) | 337.3 | 268.2 | 217.7 | 306.6 |
| G1 | exp038 = exp037 without `takeoff_epoch`, `days_since_epoch`, `day_of_year`, `week_of_year` (v11) | 352.1 | 268.9 | 218.8 | 309.1 |
| G2 | exp039 = exp037 + `lobt_delta_full` as a second mixture base (local) | 347.8 | 273.4 | 218.8 | not run |

**The time-position columns are not the 2026 problem.** Without them the holdout loses 1.1 s on the bulk
(expected: inside 2025 they help) and the board loses 2.5 s as well (single seeds, noise about ±1 s). Whatever
they encode (the late-2025 level, the seasonal position) transfers to 2026 better than nothing; exp037 stays the
base. The 2026 gap to the teams at 268-277 has to be looked for elsewhere (9.2, candidates 2 and 3).

**The LOBT delta is not a usable second stale candidate.** exp039 loses 5.2 s without wrong-date rows and 1.2 s on
the bulk; LEBL July stays at 258.5 (258.2), and the 377 holdout rows whose target equals the LOBT delta (and not
the schedule delta) improve only from 740 to 736. The classifier cannot tell a record stamped with the last
planned off-block from a genuine pushback at that time, and the third class costs 5 s everywhere else. The
14th-placed team's "LOBT residual expert" is a residual model on the LOBT clock, not a stale regime; our regressor
already has `lobt_delta` among its features. No board run.


### 9.6 Reading 2026 from its own rows: Phase A (2026-09-24, CPU only; `runs/phaseA_2026-09-24.py`)

The second plan targets features that describe 2026 from the 2026 rows (other flights' NM times, the arrivals,
the counts) rather than from 2025 history, and de-icing wherever it is done. Phase A tested three things before
any code.

**A1: does v10 under-predict where NM says taxis are long?** Not detectable that way. Per airport-day the median of
(target − NM proxy) varies from day to day by 18 s (EHAM) to 97 s (LTFM), and on the Jan+Jul holdout the
"proxy-implied bias" (prediction − proxy against the airport's typical offset) tracks the actual bias with a
correlation of 0.15 (slope 0.06). By proxy-anomaly bin, exp037 shows no under-prediction on Jan+Jul even
600-1,800 s above the stand×runway norm; the older winter-holdout model (exp028 + `wx_deice_load`) does
(+14 s at 600-900 s, +33 s at 900-1,800 s). A proxy-anchored regressor (plan phase D) is not supported.

**A2: where de-icing happens.** De-icing hours (METAR `wx_deice`) against the other hours of the same winter
months, stratified by UTC hour, in seconds:

| airport | taxi 2025 | NM proxy 2025 | NM proxy Jan 2026 | off-block delay 2025 | off-block delay Jan 2026 |
|---|---|---|---|---|---|
| EDDF | +152 | +107 | +194 | +172 | +393 |
| EDDM | +247 | +59 | +66 | +281 | +353 |
| EGLL | +179 | +300 | +142 | +201 | +97 |
| EHAM | +210 | +241 | +337 | +148 | +237 |
| LEMD | +211 | +114 | +86 | +109 | +206 |
| LFPG | +414 | +383 | +580 | +240 | +403 |
| LSZH | +143 | +94 | +128 | +162 | +289 |
| LTFM | +445 | +466 | +298 | +117 | +213 |

- De-icing lengthens the recorded taxi at every winter airport, with or without mapped pads, and delays the
  off-block as well.
- It differs by pier: EHAM D/E/A/C +240-250 s against H +39 s; LSZH general aviation 0; LFPG +300-500 s on every
  pier (remote pads); EDDF C/F/V +170-190 s against D/E +60-80 s. EDDM and EGLL have numeric stands without a pier
  prefix, so they appear as one pier.
- Across 55 piers the NM proxy's uplift tracks the taxi uplift (correlation 0.92), so the other flights' proxies
  see de-icing live. **EDDM is the exception**: the taxi rises 247 s, the proxy 59 s, i.e. NM's off-block moves
  later under de-icing there.
- Pier uplifts are stable into January 2026 (correlation 0.64 across piers, 0.48 within airports), and January
  2026 put more de-icing into the taxi than 2025 at EDDF, EHAM, LFPG and LSZH (the PRU reference-share drop of 9.1).

**A3: residual pre-tests** (gate 0.15 s; rows and method of 9.4). B1 = the other departures' NM proxy anomaly
(against the proxy's own stand×runway median, target-free) in the last 30 and 60 min and within ±15 min, per
airport, runway and pier, self excluded: mean, count, share above +5 min (29 columns with the off-block delay).
B2 = de-icing uplift per pier and per stand (fold-safe from the fit rows' targets, and a target-free twin from the
proxy over all rows), and the same times the current de-icing condition. The only winter-holdout OOF is exp028 +
`wx_deice_load`, so the strict version adds the 47 columns of exp037's extra blocks (`nm_hour_context`,
`nm_cross_deltas`, `runway_sequence`, `key_target_stats`, `taxi_window_counts`, `arrival_conditions`,
`era5_surface`) to the controls.

| test | rows | control | + candidate | gain |
|---|---|---|---|---|
| B1, Jan+Jul (exp037 residual) | 206,599 | 148.37 | 148.53 | −0.16 s |
| B2, Jan+Jul | 206,599 | 148.37 | 148.32 | +0.05 s |
| B1, Feb+Dec | 303,564 | 208.50 | 204.18 | +4.32 s |
| B2, Feb+Dec | 303,564 | 208.50 | 207.67 | +0.83 s |
| B1 + B2, Feb+Dec | 303,564 | 208.50 | 202.64 | +5.86 s |
| **B1 beyond exp037, Feb+Dec** | 303,564 | 202.06 | 199.32 | **+2.74 s** |
| **B2 beyond exp037, Feb+Dec** | 303,564 | 202.06 | 200.90 | **+1.17 s** |
| B1 beyond exp037, de-icing hours | 28,540 | 314.51 | 311.69 | +2.82 s |
| B2 beyond exp037, de-icing hours | 28,540 | 314.51 | 312.90 | +1.61 s |
| B1 / B2 beyond exp037, disrupted days | 4,165 | 451.11 | 454.0 / 456.3 | −2.9 / −5.1 s |

Both are winter features: nothing on Jan+Jul, a pass on Feb+Dec even beyond exp037's blocks, largest in de-icing
hours. The disrupted-day subset cannot judge (a few days; the control model alone is already worse than no
correction there, 451 against 447). The live means correlate 0.10 with the winter residual against 0.005 on
Jan+Jul; the count columns and the off-block delay carry nothing (|r| < 0.03). Built next as `dep_live_state`
(group `live_state`) and `deice_location` (group `deice_location`), judged on the GPU winter holdout, with the
Jan+Jul bulk as the do-no-harm check.

### 9.7 The winter holdout on the GPU (2026-09-24, seed 42, no refit; chain `runs/goal_F.log`)

exp040 = exp037 on Feb+Dec 2025 (the first winter-holdout run of the exp037 line; exp028's line read 248.1 and
789.1 disrupted there):

| run | rmse | without wrong-date rows | bulk | disrupted (4,509 rows) | disrupted bias | NM-unmatched |
|---|---|---|---|---|---|---|
| exp040 | 234.6 | 234.4 | 189.7 | 668.2 | −95.0 | 1,207.6 |
| + `dep_live_state` | 231.3 | **230.9** | **188.4** | 650.5 | −96.8 | 1,168.8 |
| + `deice_location` | 234.7 | 234.5 | 189.6 | 681.8 | −108.3 | 1,202.2 |
| + `deicing_context` | 232.3 | **232.1** | **189.0** | **647.7** | −81.3 | 1,183.2 |

- **The live state wins** (bulk −1.3 s; −3.5 s without wrong-date rows; −18 s on the disrupted rows), as the
  pre-test said. (The "nine times the seed noise" first written here used exp030's ±0.15 s; on this line the bulk
  noise is about ±1 s, see 9.10, so the bulk part is suggestive, the disrupted part the stronger evidence.)
- **The cold soak wins as well** (bulk −0.7 s, −2.3 s, −20 s disrupted with the bias −95 → −81 s), although its
  CPU pre-test failed overall (9.4): the small residual model could not use a rare three-way interaction the full
  model can.
- **The de-icing uplift per pier does not survive** the full model (bulk −0.1 s, disrupted +14 s), although it
  passed the pre-test beyond exp037's blocks: the pier×de-icing pattern is learnable from `stand_prefix`,
  `wx_deice` and the live state. Out.

exp042 (`configs/exp042_final_live_state.yaml`) = exp037 + `dep_live_state` + `deicing_context` is the next
candidate; its Jan+Jul folds are the do-no-harm check.

**exp042 (v12) on the board: 306.99**, against v10's 306.63 (single seeds; board noise about ±1 s). Jan+Jul holdout,
seed 42: bulk 217.9 (exp037 217.68), without wrong-date rows 269.3 (268.2), rmse 343.8. Level check: nothing
flagged, year-over-year correlation with the PRU 0.74 (v10 0.76); EHAM January predicted at 926 s against a PRU
floor of 982 s. The winter gains of the Feb+Dec holdout (bulk −1.3 and −0.7 s, disrupted −18 and −20 s) did not
show on the board: whatever separates us from the leaders is not in the bulk's winter behaviour. RMSE here is a
tail metric (the worst 0.1 % of holdout rows carry ~80 % of the squared error), and the board has moved most when
the tail handling changed (v9 → v10: the mixture from 30 min deltas, 5.2 s).

### 9.8 The screen of the unused blocks (2026-09-25, `exp041_screen_unused.yaml`, Jan+Jul, seed 42, no refit)

exp037 plus all 17 built blocks outside the line (and the two Phase A blocks), with block permutation. The 2026-09-20
sweep had judged most of them against the ±2 s floor of `rmse_ex_dayoffset`.

| run | rmse | without wrong-date rows | bulk | disrupted | NM-unmatched bulk |
|---|---|---|---|---|---|
| exp037 | 337.3 | 268.2 | 217.69 | 407.9 | (not recorded) |
| exp042 | 343.8 | 269.3 | 217.93 | 409.3 | 672.9 |
| exp041 screen | 338.9 | **267.5** | **216.05** | 404.8 | **644.1** |

Block permutation (holdout RMSE rise when the block is shuffled) of the unused blocks: `delay_stale_stats` +75.3
(±11.1; it feeds the stale classifier), `dep_live_state` +9.0, `stand_occupancy` +6.4 (±1.7), `taxi_route` +3.1,
`stand_geometry` +2.8, `wake_queue` +2.5 (±0.7), `taf_now` +1.0 (±1.5), `taxi_distance` +0.6, `deicing_context`
+0.06 (a winter block on Jan+Jul); zero or negative: `wx_trend`, `taf_planning`, `zone_load`, `deice_location`,
`flight_number_context`, `traffic_deviation`, `pier_pushback`, `stale_day_context`. The screen's 1.6 s bulk
lead over exp037 is within this line's seed noise (±1 s, 9.10); the stale rates among 30-120 min delays are used heavily once the mixture reaches down to
30 min deltas (exp034 onward; the sweep ran on exp028's two-hour universe). exp043
(`configs/exp043_final_screen.yaml`) = exp042 + the geometry group, `stand_occupancy`, `wake_queue`,
`delay_stale_stats`.

exp043 (the screen's blocks with a positive permutation only) on Jan+Jul: bulk 217.8, without wrong-date rows 268.5,
which first read as "the trimming lost the screen's gain"; exp043's own seeds (215.8 / 217.8 / 216.9, 9.10) show
it was seed noise. Level check: LIRF July +53 s flagged, year-over-year correlation 0.72. **v13 = board 306.13**, the
best so far by 0.5 s (v10 306.63, v12 306.99; single seeds). exp044 (`configs/exp044_final_screen_full.yaml`, the
full screen set with refit) follows.

exp044 (the full screen set with refit) reproduced the screen on Jan+Jul (bulk 216.1, without wrong-date rows 267.6;
level check clean, year-over-year correlation 0.76) and scored **v14 = 309.04 on the board**, 2.9 s worse than v13.

| version | config | Jan+Jul bulk | without wrong-date rows | board |
|---|---|---|---|---|
| v10 | exp037 | 217.69 | 268.2 | 306.63 |
| v11 | exp038 (no date columns) | 218.8 | 268.9 | 309.12 |
| v12 | exp042 = exp037 + `dep_live_state` + `deicing_context` | 217.93 | 269.3 | 306.99 |
| v13 | exp043 = exp042 + geometry, `stand_occupancy`, `wake_queue`, `delay_stale_stats` | 217.8 | 268.5 | **306.13** |
| v14 | exp044 = exp037 + every built block | **216.1** | **267.6** | 309.04 |

**The Jan+Jul holdout does not rank these candidates the way the board does.** The best local candidate is the
worst on the board of the last three; the blocks v14 adds over v13 (`taf_now`, `taf_planning`, `wx_trend`,
`zone_load`, `pier_pushback`, `traffic_deviation`, `flight_number_context`, `stale_day_context`, `deice_location`)
describe 2025 well and 2026 worse: flight-number regularity, day-level traffic against the weekday median, and the
day's stale-suspect counts are exactly the kind of history that shifts between years. Board differences below
about 1 s are single-seed noise; v13's 0.5 s over v10 is one of them. What the board has rewarded so far is the
tail (v9 → v10, the mixture from 30 min deltas: 5.2 s), not the bulk: the 70 s to the leaders is unlikely to sit in
features judged on the Jan+Jul bulk.

### 9.9 Where the squared error sits, and what 2026 looks like (2026-09-25; `runs/tail_2026-09-25.py`, v13's run)

Target-free segments, so that 2026 has them too: U = the mixture's universe (every departure 30 min or more behind
schedule since exp034; 40 % of rows), N = outside U without an NM match, M = outside U with one.

| segment | holdout rows | holdout RMSE | share of holdout SSE | 2026 rows |
|---|---|---|---|---|
| M matched, not delayed | 205,673 (59.7 %) | 147 | 11.8 % | 207,319 (60.1 %) |
| N unmatched, not delayed | 927 | 2,774 | 18.8 % (one LFPG wrong-date row: 18.4 %) | 977 |
| U delayed ≥ 30 min | 137,819 (40.0 %) | 437 | 69.4 % | 136,545 (39.6 %) |

Inside U, by what the target turned out to be (shares of the whole holdout SSE): delayed flights with an ordinary
taxi below an hour 22.9 % (111,751 rows, RMSE 279 against 147 for flights on time; at LIRF the hedge doubles it,
600 against the regressor's 320), wrong-date rows 16.0 % (6), other targets above 2 h 10.6 % (26), near-stale
targets 1-15 min off the schedule delta 9.5 % (23,707, RMSE 389), exact stale 7.5 % (1,987; mean p 0.28, predicted
2,589 against 3,270). The classifier is calibrated in every probability band (stale rate 0.005 / 0.098 / 0.211 /
0.572 / 0.857 / 0.978 against mean p 0.004 / 0.099 / 0.275 / 0.652 / 0.892 / 0.985); what it lacks is discrimination.

- **The bulk cannot close the gap.** A perfect model of the not-delayed matched rows would take the holdout from 332
  to about 311. This is why bulk features (9.7, 9.8) do not show on the board.
- **2026 has the holdout's segment sizes** (U 39.6 %, N 0.28 %, 29.5 per 1,000 rows with a schedule delta above 2 h
  against 26.4). At the holdout's per-segment MSE, 2026 would read 332; the board reads 306, so 2026 has fewer or
  milder extremes than Jan+Jul 2025 (its long-delta Rome rows are stale, not wrong-date: 16 Sep board test).
- **Delayed flights are the largest attackable block.** 75 % of U's rows below 2 h match no clock (schedule, EOBT,
  LOBT, IOBT, NM AOBT within 60 s) and carry 30 % of the holdout SSE. For them the airport's off-block is often
  earlier than NM's: the target exceeds the NM proxy by more than 5 min on 18.7 % of delayed flights against 6.8 %
  on time (IQR 385 against 295 s), consistent with pushing back and then holding (slot waits, remote holds), which
  no column records per flight. Planned clocks explain little: 5-6 % of delayed targets equal takeoff minus
  EOBT/LOBT/IOBT (RMSE 317-389, under-predicted by 126-151 s), 1.4 % the schedule delta.

### 9.10 Seed noise on this line, and the three-seed blend (2026-09-25; chain `runs/goal_J.log`)

exp043 under seeds 42 / 43 / 44 (Jan+Jul, each with refit):

| seed | rmse | without wrong-date rows | bulk |
|---|---|---|---|
| 42 (v13) | 331.7 | 268.5 | 217.8 |
| 43 | 349.0 | 267.8 | 215.8 |
| 44 | 346.1 | 268.1 | 216.9 |
| blend of the three (v15) | 340.6 | 267.0 | 215.7 |

**The bulk's seed noise on the mixture-30 line is about ±1 s, not the ±0.15 s measured on exp030** (before the
mixture reached down to 30 min deltas, which puts 40 % of the rows under the classifier's hedge). Every bulk
difference of this night below ~2 s is therefore noise: exp037 / exp042 / exp043 / exp044 (217.69 / 217.93 / 217.8 /
216.1) are indistinguishable locally, and the winter bulk gains of 9.7 (−1.3, −0.7 s) are suggestive at best; the
disrupted-day gains (−18, −20 s) and the board are the stronger evidence. Local comparisons on this line need three
seeds, or a metric outside the hedge (the matched, not-delayed rows).

**v15 = the three-seed blend of exp043: board 304.85, the best so far** (v13 306.13 single seed, v10 306.63). Level
check clean, year-over-year correlation 0.74. The blend's 1.3 s matches the ~1 s expected from averaging seeds.

### 9.11 The hedging cost on the 2026 rows tracks the board (2026-09-25)

The mixture's prediction on a universe row is p·Δ + (1 − p)·reg, with Δ the schedule delta; if p is calibrated,
the row's expected squared error from the regime uncertainty is p(1 − p)(Δ − reg)². Summed over the universe and
divided by all rows, this **hedging cost** needs no target, so it exists for the ranking rows:

| run | board | board MSE | hedging cost, 2026 | hedging cost, holdout | holdout U < 2 h RMSE |
|---|---|---|---|---|---|
| v13 exp043 s42 | 306.13 | 93,716 | **16,314** | 31,893 | 325.2 |
| v10 exp037 | 306.63 | 94,022 | 16,793 | 30,424 | 324.5 |
| v12 exp042 | 306.99 | 94,243 | 16,777 | 31,474 | 324.9 |
| v14 exp044 | 309.04 | 95,506 | 17,256 | 32,452 | **321.9** |
| v11 exp038 | 309.12 | 95,555 | 17,054 | 33,975 | 325.8 |
| exp043 s43 | | | 16,397 | 35,063 | 322.2 |
| exp043 s44 | | | 15,334 | 32,880 | 323.5 |

- **Across the five submissions the 2026 hedging cost correlates 0.90 with the board MSE**, and differences in it
  cover about half of the board differences; the holdout's own numbers do not (v14 has the best holdout U RMSE and
  the worst board). Five points and a seed spread of ±500 make this a lead, not a law. It can be gamed by
  overconfidence (p near 0 or 1 lowers p(1 − p) whatever the truth), so it must be read next to the holdout's
  calibration by probability band.
- It sizes the prize: the expected hedging cost is 16,300 of the board's 93,700 MSE (17 %). Perfect discrimination
  would read about 278 on the board; each 10 % less hedging cost is worth about 2.7 s.
- Stale recording leaves no trace on the arrivals: of the arrivals landing 30 min or more late, none has an in-block
  time equal to its schedule, at any airport, in either year. The fingerprint has to come from the departures' own
  columns.

### 9.12 What the stale classifier misses (2026-09-25; `runs/stale_misses_2026-09-25.py`)

Holdout universe without wrong-date rows (137,813 rows, 1,987 stale = 1.44 %), p averaged over exp043's three
seeds: AUC 0.947, hedging cost 15,838 per holdout row.

- **84 % of the hedging cost is LIRF**: delays of 4 h or more 30.6 % (151 rows), 2-4 h 18.7 % (769), 60-90 min
  14.3 %, 90-120 min 10.2 %, 45-60 min 5.9 %, 30-45 min 4.2 %. Outside Rome only LFPG 4 h+, LTFM and EGLL 30-45 min
  reach 1 % each.
- **The classifier is already sharp where it matters.** On LIRF's long delays the stale rows get a median p of
  0.96-0.97 (q10 0.52-0.61) and the genuine ones 0.000 (q99 0.91-0.95); AUC 0.996 (4 h+) and 0.989 (2-4 h). The
  band means (0.22 against a stale rate of 0.24) only looked moderate because they average the two groups. A
  cross-fitted isotonic calibration per (LIRF | other) × delta band makes the holdout worse (without wrong-date
  rows 268.5 → 271.1; LIRF universe 732 → 759): what is left is a few genuinely ambiguous rows with huge deltas.
- **No unused signal found.** Among rows with p < 0.05 (0.45 % stale) the columns that still separate
  (`eobt1_minus_sched`, `aobt3_minus_sched`, `lobt_eq_sched`, the `delay_stale_stats` rates; univariate AUC
  0.70-0.75) are ones the classifier already uses (its own AUC inside the stratum 0.906). Among 0.05 ≤ p < 0.5
  nothing exceeds 0.54. Raw candidates (schedule minute, 5-minute schedules, seconds fields, takeoff hour, delta
  modulo 5 min, callsign = flight number) add nothing inside the strata.

### 9.13 Two classifier changes on the GPU, and the instrument's limit (2026-09-25; chain `runs/goal_K.log`)

The hedging report is now part of every mixture run (`metrics.json` `hedging`: `holdout`, `rank_folds`,
`rank_refit`, `ece`, `brier`, the calibration table by probability band; `hedge_rank` / `hedge_holdout` in the
index, the leaderboard and the queue summary; OOF columns `hedge_var` and `regime`). `model.mixture.classifier_seeds`
averages the classifier over several seeds (hash-neutral at 1).

| run | without wrong-date rows | bulk | hedging, holdout | hedging, 2026 refit | ece | board |
|---|---|---|---|---|---|---|
| exp043 s42 / s43 / s44 | 268.5 / 267.8 / 268.1 | 217.8 / 215.8 / 216.9 | 31,893 / 35,063 / 32,880 | 16,314 / 16,397 / 15,334 | | 306.13 (s42) |
| exp045 = exp043 + `classifier_weight_power: 2` | 272.7 | 220.7 | 29,377 | **13,742** | 0.0013 | **308.01** (v16) |
| exp046 = exp043 + `classifier_seeds: 5` | 269.5 | 216.3 | 33,960 | 17,078 | 0.0009 | 306.62 (v17) |

**Δ² weights hurt** (4-5 s on the holdout, 1.9 s on the board): they move the classifier's capacity to the rare
huge deltas and cost it on the many 30-60 min ones. **And they refute the hedging cost as a selection
instrument**: exp045 has the lowest 2026 hedging cost of any run and a better calibration error, and a worse
board. The hedging cost measures the classifier's confidence; a classifier that says p ≈ 0 on more rows lowers
it even where some of those rows are stale, and a calibration error averaged over mostly low-probability rows
does not show that. The correlation of 9.11 held among runs with the same classifier setup only. Keep it as a
diagnostic; choose on the holdout (three seeds) and the board.

**Five averaged classifiers are neutral** (holdout within exp043's seed range, board 306.62 against v13's 306.13
with the same seed): the classifier's seed variance is not where the loss is. The **Brier score** over the
universe is the classifier instrument that works: exp043's seeds 0.01061-0.01063, exp046 0.0106, exp045 0.01082,
the one that lost on the board.

**Conclusion of the delayed-flight work (9.9-9.13).** The stale classifier is at the limit of the information
the data holds: it is calibrated, sharp where the deltas are large (9.12), no unused column separates its misses,
and neither loss weighting, calibration nor seed averaging improves it. What remains in the delayed flights'
error is a few genuinely ambiguous stale rows with huge deltas and the delayed ordinary taxis' larger scatter
(pushed back, then held), for which no public column exists. Best board: v15 (three-seed blend of exp043) 304.85.


### 9.14 LIRF's disruption days (2026-09-25; `runs/lirf_days_2026-09-25.py`, `runs/ground_hold_2026-09-25.py`)

**Stale records are not a disruption-day phenomenon.** In 2025, 9 % of LIRF's departures 30 min or more late carry a
stale block time on every weekday (7.6-9.5 %) and in every month (7.8-10.3 %); the 10 % of days with the most stale
records hold 22 % of them. The count follows the number of delayed flights (July most). One outlier: 8 June 2025,
26.5 % of 204 delayed departures stale.

**LIRF's large errors do cluster on a few days.** In the holdout, the wrong-date rows dominate their days (15, 3, 30,
21 July, 25 January). Without them, four days carry 36 % of LIRF's squared error: 14 July 2025 (13.7 %), 28 July
(9.7 %), 13 July (7.0 %), 27 July (5.9 %), with mean delays of 72-83 min and 74-85 departures two hours or more late.

- 13 July 2025: a severe thunderstorm from about 20:00 local; ENAV limited traffic, takeoffs and landings stopped
  for about 30 minutes, 16 arrivals diverted, delays accumulated through the day (ANSA, Corriere di Maremma). The
  METAR reports thunderstorms at 12:20-13:50 and 18:20-20:50, the hours with the largest errors. Delayed ordinary
  flights had a mean recorded taxi of 35 min against the 17 min NM's off-block implies: pushed back, then held.
- 14 July: no thunderstorm; the errors sit at 00-02 and 08 local, flights delayed overnight from the 13th, and 57 %
  of the day's squared error is stale records with a mean target of 2 h.
- 28 July: thunderstorms 13:50-14:20 and 19 CB/TCU reports; errors at 15, 16 and 20 local. 27 July: no storm and no
  news found; errors at 10 and 23 local.
- 2026 has such days: 2 July 2026 (416 delayed departures, mean delay 70 min, 68 two hours late; thunderstorms
  07:20-08:20 local, storms over central Italy) and 5 January 2026 (thunderstorms 00:50-02:50). Only 41 of 549 LIRF
  days carry any thunderstorm report.
- Strike days do not stand out: the handling strike of 10 July 2025 and the national strike of 26 July 2025 each
  hold about 1 % of LIRF's holdout squared error.

**Disruption signals mark the size of the error, not its direction.** On the holdout's delayed, non-stale
departures (135,777 rows, 31 % of the squared error), the longest gap between takeoffs at the airport while the
flight was due out raises the RMSE from 220 s (under 3 min) to 718 s (over 30 min), and METAR thunderstorms in the
same window from 290 to 905 s; the mean residual moves only from −16 to +56 s (+114 s with four or more reports),
correlations 0.00-0.03. The model is about unbiased on those days; the scatter is what grows, and a feature that
flags variance does not lower an RMSE. At LIRF the delayed genuine departures are over-predicted by about 100 s
(about 200 s during long holds): the stale hedge pulls them toward their large schedule deltas, the price of a
classifier that is calibrated overall.

### 9.15 Weighting the fit toward the scored months (2026-09-25; `configs/exp047_month_weights.yaml`)

`model.month_weights` (new; row weights by the takeoff's calendar month, normalised to a mean of one, for the
regressor and the stale classifier, in the folds and the refit; hash-neutral while empty). exp047 = exp043 with
January and July 2.0, December, February, June and August 1.5, the rest 1.0 (normalised in the refit: 1.505 /
1.129 / 0.752).

| run | without wrong-date rows | bulk | Brier | board |
|---|---|---|---|---|
| exp043 s42 / s43 / s44 | 268.5 / 267.8 / 268.1 | 217.8 / 215.8 / 216.9 | 0.01061-0.01063 | 306.13 (s42) |
| exp047 (v18) | 267.9 | 217.1 | 0.0106 | 306.09 |

Neutral everywhere: the holdout (which sees only the neighbours' weight, since January and July are its own months)
and the board (which sees the full weighting in the refit), 0.04 s apart. The model already carries the season
through `month`, `day_of_year` and the weather; giving the scored months more weight adds nothing. The option
stays for later use.

### 9.16 A forward validation, and why v14 lost on the board (2026-09-25)

**Forward validation** (`validation.scheme: forward_holdout`, new): fit only on the months before the validation
months. exp048 = exp043 fit on January-October 2025 and validated on November and December; exp048b = exp044 the
same. Seed noise here is small: exp048 reads 200.9 / 201.2 without wrong-date rows and 186.6 / 186.9 on the bulk
(seeds 42 / 43).

| run | without wrong-date rows | bulk | board of the same config |
|---|---|---|---|
| exp048 (exp043's blocks) s42 / s43 | 200.9 / 201.2 | 186.6 / 186.9 | 306.13 (v13) |
| exp048b (exp044, every block) s42 / s43 | 199.8 / 202.2 | 186.5 / 187.1 | 309.04 (v14) |

**The calibration fails:** the forward validation ties the two sets (means 201.05 against 201.0 without wrong-date
rows, 186.75 against 186.8 on the bulk; exp048b's tail spreads 2.4 s between its seeds), while the board ranked the
full set 2.9 s worse. No 2025-based split shows the loss.

**Adversarial validation finds it** (`runs/adversarial_2026-09-25.py`): per block, a small classifier telling
January and July 2026 from January and July 2025 departures (60,000 each, folds by calendar day). AUC 0.5 = the
block looks the same in both years.

| block | AUC | in exp043 |
|---|---|---|
| `flight_number_context` | **0.80** | no |
| `traffic_deviation` | **0.74** | no |
| `runway_sequence` | 0.62 | yes |
| `era5_surface`, `wx_now` | 0.61 | yes |
| `stand_runway_keys`, `key_target_stats` | 0.605 | yes |
| `taf_now`, `operator_presence`, `stand_runway_target_stats` | 0.59 | taf no, others yes |
| all other blocks | 0.50-0.59 | |

The two blocks that separate the years are exactly the ones v14 added over v13. Single columns:
`fn_n_months_active` 0.77 (a 2026 flight number can appear in at most two months of the frame, a 2025 one in up to
twelve), `fn_n_rows_apt` 0.71, `fn_rows_per_active_month` 0.67, `apt_day_mvt/arr/dep_ratio` 0.64-0.66 (the day's
traffic against the weekday median, which 2026's traffic level moves). These are artefacts of how frame-wide
counts are built, so their meaning differs on every 2026 row, and a validation inside 2025 cannot see it. exp043's
blocks separate the years only moderately (0.55-0.62), mostly through real differences (weather, runway use,
stands). The calendar's 0.285 is an artefact of the calendar-day folds (the same date falls on another weekday).

**Rule from this:** a feature computed over the whole frame (counts, months active, ratios to a median) must be
checked by adversarial validation before it goes into a submission; the holdout cannot judge it.

### 9.17 Trimming exp043 under the forward validation (2026-09-26; chain `runs/goal_M.log`, `runs/trim_decision_2026-09-26.py`)

Leave-one-out from exp048 (exp043 under the forward validation; two-seed base: bulk 186.76, without wrong-date
rows 201.07). Seed spread: bulk ±0.3 s; without wrong-date rows up to ±1.2 s (exp048b's seeds 2.4 s apart), so the
bulk is read first. Positive = removing the block makes it worse. AUC = the block's adversarial score (9.16).

| removed block | bulk | without wrong-date rows | disrupted days | AUC | verdict |
|---|---|---|---|---|---|
| `nm_hour_context` | +1.32 | +1.58 | +4.71 | 0.58 | keep |
| `dep_live_state` | +1.10 | +1.63 | +4.24 | 0.55 | keep |
| `runway_sequence` | +0.37 | +0.58 | −1.48 | 0.62 | keep |
| `load_normalised` | +0.22 | +0.42 | −4.44 | 0.53 | keep |
| `taxi_window_counts` | +0.17 | +0.48 | +1.00 | 0.53 | keep |
| `stand_runway_frequency` | +0.14 | +0.91 | +1.02 | 0.59 | keep |
| `nm_cross_deltas` | +0.08 | +0.44 | +1.21 | 0.52 | keep |
| `operator_presence` | +0.07 | +0.75 | +2.45 | 0.59 | keep |
| `stand_occupancy` | +0.05 | +1.03 | +0.42 | 0.52 | keep |
| `wake_queue` | −0.10 | +0.43 | −1.39 | 0.50 | keep |
| `deicing_context` | −0.20 | +0.80 | +3.53 | 0.50 | keep |
| `delay_stale_stats` | −0.24 | +1.32 | +0.57 | 0.55 | keep |
| `taxi_route` | +0.12 | +0.23 | −3.31 | 0.56 | trim |
| `key_target_stats` | −0.14 | −0.15 | +1.67 | 0.60 | trim (neutral, drifts) |
| `stand_geometry` | −0.17 | −0.05 | −0.89 | 0.55 | trim |
| `stale_target_stats` | **−0.47** | **−1.48** | +1.93 | 0.52 | trim (helps) |

exp049 (depth 6, `l2_leaf_reg` 10): bulk +0.14, without wrong-date rows −0.17: neutral, not adopted.
Every block of the live NM and arrival picture earns its place; the fold-dependent history statistics
(`stale_target_stats`, `key_target_stats`) and the static geometry do not. exp050 (`configs/exp050_lean.yaml`) =
exp043 without the four trimmed blocks (30 blocks, 234 columns).

**exp050 (the lean set) on Jan+Jul, seed 42: rmse 322.1, without wrong-date rows 264.3, bulk 215.7** (exp043's three
seeds 267.8-268.5 and 215.8-217.8), Brier 0.0106, level check clean (year-over-year correlation 0.76).
**v19 = board 304.39, the best so far**, 1.7 s under v13 (the same model with the four blocks, same seed) and under
v15 (exp043's three-seed blend, 304.85) with a single seed. Trimming the history statistics and the static geometry
transferred to the board. A three-seed blend of exp050 follows (chain `runs/goal_O.log`).

Three seeds of exp050 (Jan+Jul): without wrong-date rows 264.3 / 264.5 / 264.1, bulk 215.7 / 215.6 / 214.7 (exp043:
267.8-268.5 and 215.8-217.8), so the gain is consistent across seeds, not one lucky run. Their blend: 263.0 and
214.1, level check clean. **v20 = the three-seed blend of exp050: board 304.08, the best so far** (v19 304.39 single
seed; the blend adds 0.3 s here against 1.3 s for exp043's).

| version | what | board |
|---|---|---|
| v13 | exp043, seed 42 | 306.13 |
| v15 | exp043, three-seed blend | 304.85 |
| v19 | exp050 = exp043 minus four blocks, seed 42 | 304.39 |
| v20 | exp050, three-seed blend | **304.08** |

### 9.18 Classifying delayed flights around the known clocks (2026-09-26; `runs/clock_classes*_2026-09-26.py`)

Idea: for a delayed flight the airport's recorded off-block sits near one of a few clocks, so classify which and
predict the probability-weighted value. Classes on the holdout's universe rows without wrong-date rows (137,813),
in priority order: stale (|y − Δ| ≤ 60 s, 1.4 %), near-stale (block 1-15 min after the schedule, 16.4 %; median
offset 655 s: pushed about on time, then held), NM clock (|y − proxy| ≤ 120 s, 29.7 %), planned (y within 60 s of
the LOBT/EOBT/IOBT delta, 3.5 %), other (49.0 %; 63.8 % of these rows' squared error).

| model (holdout universe rows) | universe rmse | holdout without wrong-date rows |
|---|---|---|
| exp050 (current) | 375.5 | **264.34** |
| clock classes, cross-fitted inside the holdout (the whole regime model replaced) | 383.9 | 269.14 |
| control: one sklearn regressor for the taxi side, fitted on the fit months, exp050's stale probability kept | 385.4 | 269.95 |
| clock classes, the same setting (classifier and "other" regressor on 300,000 fit rows) | 392.9 | 274.27 |
| oracle (the true class's value) | 309.8 | 227.71 |

**The clock split loses** (4.3 s against its same-learner control, on every class). The classes are not
identifiable enough: mean probability of the true class 0.67 for near-stale, 0.45 NM clock, 0.16 planned, 0.62
other; and a near-stale row keeps its 1-15 min offset spread even when recognised. A single regressor already
learns the conditional mean across these regimes better than a mixture of clock values with fixed offsets. The
ceiling (227.7) is real structure the data cannot resolve per flight. Not built.

### 9.19 Second trimming round, from exp050 (2026-09-29; chain `runs/goal_P.log`, `runs/trim_decision_2026-09-29.py`)

Leave-one-out from exp050b (exp050 under the forward validation, fit Jan-Oct 2025, validate Nov+Dec) over the eleven
exp050 blocks round 1 did not test. Two-seed base: bulk 186.24 / 186.09, without wrong-date rows 199.48 / 198.93
(means 186.17 / 199.20; round 1's exp048 base was 186.76 / 201.07, so the lean set holds under this validation too).
Bands: bulk ±0.30 (round 1's; this base's seeds are 0.15 apart), without wrong-date rows ±0.56 (this base's spread).
Positive = removing the block makes it worse. Variants that take dependents with them are measured as a group.

| removed block (and what went with it) | bulk | without wrong-date rows | disrupted days | AUC | verdict |
|---|---|---|---|---|---|
| `prev_stand_arrival` (+ `deicing_context`) | +1.77 | +1.80 | +4.06 | 0.57 | keep |
| `pier_load` | +0.42 | −0.18 | −0.03 | 0.51 | keep |
| `arrival_conditions` | +0.31 | −0.06 | −0.75 | 0.57 | keep (on the band edge) |
| `taxi_distance` | +0.30 | +0.36 | −1.64 | 0.58 | keep (on the band edge) |
| `wx_now` (+ `deicing_context`, `wx_deice_load`) | +0.27 | +0.19 | −3.85 | 0.61 | neutral, mixed |
| `wx_deice_load` | +0.25 | +0.05 | +1.14 | 0.50 | neutral |
| `era5_surface` (+ `deicing_context`) | +0.22 | +0.39 | +3.59 | 0.61 | neutral, worse on every metric |
| `calendar` (+ `load_normalised`) | +0.19 | −0.36 | −5.89 | 0.29 | neutral, not judgeable here |
| `stand_runway_target_stats` | +0.01 | +0.01 | −2.49 | 0.59 | trim |
| `wx_past` (+ `wx_deice_load`) | −0.03 | −0.04 | −3.69 | 0.53 | trim |
| `nm_proxy_context` | **−0.39** | **−0.52** | −2.84 | 0.52 | trim (helps) |

- `nm_proxy_context` (the AOBT3 proxy against its airport-hour peers) is the one block whose removal helps on every
  metric; `dep_live_state` and `nm_hour_context` carry the same NM picture better.
- `stand_runway_target_stats` is exactly neutral and drifts (0.59): the last fold-dependent history statistic, the
  kind whose removal transferred to the board in round 1.
- `wx_past` goes together with `wx_deice_load`: neutral on the bulk, better on the disrupted days.
- `calendar` cannot be judged by this validation: it holds `month`, `day_of_year`, `week_of_year` and
  `days_since_epoch`, whose November-December values are never seen in a Jan-Oct fit (the 2026 months are), and
  removing three of those columns cost 2.5 s on the board (v11). Kept.
- `era5_surface` and `wx_now` drift (0.61) but are neutral-to-worse here; kept for now (January 2026 had several
  times 2025's snow).

**The candidate, 2026-09-29 (chain `runs/goal_Q.log`).** exp051 (`configs/exp051_lean2.yaml` = exp050 without the
four) on Jan+Jul, seed 42: rmse 323.6, without wrong-date rows 263.4, bulk 214.8 (exp050 s42: 322.1 / 264.3 / 215.7),
level check clean (year-over-year correlation 0.75). Under the forward validation (exp051b, seeds 42 / 43): bulk
186.17 / 186.13, without wrong-date rows 199.30 / 198.76, means 186.15 / 199.03 against exp050b's 186.17 / 199.20:
**the four removals are neutral together** (`nm_proxy_context`'s −0.4 does not add up). **v21 = exp051 seed 42:
board 305.37**, on exp050's single-seed mean (9.20). Round 2 does not move the board.

### 9.20 The board's own seed noise (2026-09-29)

exp050's three seeds, each submitted alone (their predictions came from the blend chain `runs/goal_O.log`):

| version | run | board |
|---|---|---|
| v19 | exp050 seed 42 | 304.39 |
| v22 | exp050 seed 43 | 305.54 |
| v23 | exp050 seed 44 | 306.10 |
| | mean of the three single seeds | 305.34 (spread 1.7 s) |
| v20 | their three-seed blend | **304.08** |
| v21 | exp051 seed 42 | 305.37 |

- **A single seed moves the board by up to 1.7 s** (SD about 0.9 s). Single-seed board gaps below about 2 s are
  noise: v10, v12, v13, v17 and v18 (306.1-307.0) cannot be told apart, and v19's 1.7 s under v13 was partly the
  lucky seed. Only v11 (+2.5), v14 (+2.9) and perhaps v16 (+1.9) were real losses.
- **The three-seed blend is 1.3 s under the single-seed mean**, the same gain as exp043's blend (1.3 s). Candidates
  are compared blend against blend; the blend is also the one lever that has always paid.

**Wider blends (2026-09-30; chains `runs/goal_R.log`, blends by hand with `prc blend`).** exp051 seeds 43/44 and
seeds 45/46 of exp050 and exp051 were trained for blending; every blend below is an equal-weight mean of the runs'
refit predictions.

| version | blend | runs | holdout without wrong-date rows / bulk | board |
|---|---|---|---|---|
| v20 | exp050 × 3 | 3 | 263.0 / 214.1 | 304.08 |
| v24 | exp051 × 3 | 3 | 262.1 / 213.3 | 304.15 |
| v25 | exp050 × 3 + exp051 × 3 | 6 | 262.2 / 213.2 | 303.63 |
| v27 | + exp043 × 3 | 9 | 262.4 / 213.2 | 303.15 |
| v28 | exp043 × 3 + exp050 × 5 + exp051 × 5 | 13 | 262.1 / 213.0 | 303.16 |
| v29 | v28 + exp037, exp042, exp046, exp047 (seed 42 each) | 17 | 262.6 / 213.1 | **303.02** |

- Three seeds to six gained 0.45 s; adding a different block set (exp043) another 0.5 s; four more seeds of the same
  two sets nothing (v28 = v27); four differently configured single runs 0.13 s. **The blend gains from different
  models now, not from more seeds of one.** The holdout does not see any of this (262.1-262.6 throughout).
- v26 was rejected: **the scorer takes five submissions per UTC day** (`DAILY_LIMIT_REACHED`, resets 00:00 UTC).
- For the next blend (chain `runs/goal_S.log`): exp052 (exp050, depth 6, `l2_leaf_reg` 10; alone 264.98 / 216.18) and exp053
  (exp050, depth 9; 265.77 / 215.49), seed 42 each with refit, both a little weaker alone. The 19-run blend (v29 + both,
  `2026-09-30T05-47-06__blend19_wide__9f4c048e`, holdout 262.5 / 213.1, level check clean) = **v30, board 303.03**: level with
  v29, so the blend has plateaued around 303.0.

## 10. The `new_approach_testing` branch (from 2026-09-30)

Goal of the branch: review the approach, try new ones, and document the result for a JOAS-style report. Board
before the branch: v29 303.02 (108th of 209; 1st 219.64, 10th 235.92, 25th 247.54 on 30 Sep 17:45 UTC).

### 10.1 Rules, timeline and the board (official pages and API, 30 Sep)

- The competition closes on **11 October 2026, 23:59:59 CET** (safe cut-off 21:59 UTC). Five uploads per UTC day.
- The ranking is the best RMSE on all 344,841 Jan+Jul 2026 departures (`used_pairs` = 344,841 on every upload); no
  phase 2 is announced, the organisers "monitor submissions for attempts to learn from or exploit the ranking
  process". Prize eligibility: code public on GitHub under GPLv3, external data openly licensed and documented,
  original solution. A JOAS paper is encouraged, not required.
- The full submission history is public (`datacomp.opensky-network.org/api/competitions/<id>/leaderboard`, 3,646
  scored submissions, 209 teams). The leaders improved by 15-50 s between 23 and 30 September.
- According to a competitor's notes of the organisers' Discord (kind-mango, `RECAP.md`, not verified by us), the
  organiser ruled on 17-18 September that open trajectory data (ADSB.lol, ODbL) may be used to extract off-block
  times, and several public repos started pulling adsb.lol surface tracks on 23 September. **This branch uses no
  position data** (user rule of 25 September); whether to lift that rule is the user's decision.

### 10.2 Checks that found nothing (30 Sep)

- **An XGBoost (CUDA) taxi regressor** on the lean2 matrix (`lab/xgb_regressor.py`, one global model, early-stopped
  on the holdout's taxi rows): 227.2 on the holdout's taxi rows against 210.0 for the per-airport CatBoost regressor;
  a 50/50 mean 213.3. Worse alone and in a blend; CatBoost's ordered target statistics on the high-cardinality
  keys (stand, stand x runway, destination) carry the difference.
- **The filed block time against its route median** (`ARVT_1 - EOBT_1` minus the (airport, destination, type)
  median; a competitor's `plan_nm_taxi`, -1.2 s for them): correlation with the lean2 holdout residual -0.008 on the
  not-delayed matched rows. The model already has `m1_flight_dur` and the destination.
- **The recording process did not change in 2026.** On the arrivals, whose block times are published in 2026 too,
  the taxi-in level and shape, the share of block times on a whole minute (1.7 % where seconds are recorded, 8.3 %
  where they come in 5 s steps), the share of in-block times equal to the schedule and every missing-value rate are
  the same in Jan/Jul 2025 and 2026, per airport; so are the departures' takeoff-time properties and the NM proxy's
  median.
- **Fallback (stale) rates are seasonal but not trending**: per 1,000 departures 30 min or more late, LIRF 78-103 in
  every month, EGLL 16-30, LTFM 6-30; the German airports and LFPG are 5-20 times higher in winter than in summer
  (EDDM 10.3 in January, 0.4 in July), LEBL peaks in July-September (19.4). Jan+Jul 2025 carries both seasons.
- **Where 2026's extreme predictions sit**: every 2026 prediction above 7,200 s is an untyped, NM-less Rome row with
  p_stale 0.95-0.9997 (the holdout's p > 0.95 band was stale 97.8 % of the time); no other airport has a prediction
  above 6,340 s. There is nothing to cap.
- **The regime hedge's expected cost** (sum of p(1-p)(delta - reg)^2 per row, target-free): 14.0k of MSE on the
  2026 rows against 24.7k on the holdout. If the classifier were as calibrated in 2026 as in 2025, 2026 would be
  easier than the holdout; the board sits ~23k of MSE above the holdout without wrong-date rows (v6-v30, constant
  offset of 38-42 s). A handful of rows can carry that: one universe row with a 60,000 s delta and p = 0.95 that
  turns out genuine costs 9,400 of board MSE.

### 10.3 The NM join window: a hard bound on every matched target (30 Sep)

Found by the competitor-research agent in other teams' public code (merry-quicksand's "M1", zestful-fountain's
clock overlay, kind-mango's ITY340 rule) and verified on our data: on **all 2,062,577 NM-matched 2025 departures,
`LOBT_flt - BLOCK_TIME_UTC_mvt` lies in [-3,606 s, +3,606 s]**, at every airport (each airport reaches the edge),
and every departure with `LOBT_flt` has an NM match. This is the organisers' join criterion: a movement gets NM
data only when its block time is within about an hour of NM's last off-block time. The ranking set was joined the
same way before its block times were blanked, so every matched 2026 target lies in
`[MVT - LOBT - 3,606, MVT - LOBT + 3,606]` (339,551 of 344,841 ranking rows carry LOBT).

- Projecting a prediction onto an interval that holds the truth can only shorten its distance to the truth, so the
  clip cannot raise any row's error (`prc/postprocess.py`, `prc blend --lobt-bound`, tests in
  `tests/test_postprocess.py`).
- The model breaks the bound in two ways: the mixture hedges matched rows toward a schedule more than an hour
  outside the window (a stale block the join would not have matched), and it predicts ordinary taxis for rows whose
  takeoff came hours after NM's last off-block time (the join implies the block was at most an hour after LOBT, so
  the taxi lasted hours: Amsterdam's snow days).
- Where the bound binds hard in training (takeoff 2 h or more after LOBT, 144 rows), the target sits well inside
  the window: y - lower bound has a median of 2,600-2,800 s. Clipping to the edge is the safe first step; an
  estimate inside the window is the refinement.
- **Holdout (v29's 17-run blend):** 98 rows raised, 16 lowered; rmse 328.13 -> 312.25, without wrong-date rows
  262.55 -> 260.29, bulk 213.08 -> 212.99 (lean2 s42: 323.59 -> 311.30).
- **Ranking rows (v29):** 159 raised, 5 lowered, sum of squared shifts 6.94e9, so if the rule holds in 2026 the
  board MSE falls by at least 20,124 (303.02 -> at most 267.8). Almost all of it is one row: `mvt_id 213147416`
  (ITY340, LIRF, 17 Jul 2026 21:51), whose NM clocks all sit a day before its takeoff (window
  [83,214, 90,426] s) and which we predicted at 1,683 s. The only 2025 row with that fingerprint (AEZ146R, LIRF,
  15 Jul 2025) had a target of 87,002 s. The window, not a board reading, decides this row.
- Upload: `2026-09-30T20-18-18__blend17_wide_lobt__a41a47d5` (v29 clipped) as v31.
- Where the 2026 clip acted: LIRF 3 rows (the ITY340 row alone 6.65e9 of squared shift), EHAM 90 rows all raised
  (2.2e8; 2-9 January, the snow days), LFPG 23 (6.3e7), the other airports together under 1e7. The board gained
  22.6k of MSE against the guaranteed 20.1k: the raised rows' targets lie deeper inside the window than its edge.
- **v31 = board 263.17** (from 303.02; -22,564 MSE, more than the guaranteed 20,124). The window holds in 2026.
- The mixture had already learned the window for its stale candidates (mean p_stale 0.0000 on the 13,772 holdout
  universe rows whose schedule lies outside the window); the gain comes from rows whose window forces a taxi longer
  than the model predicts. The regressor could not learn those: `regressor_max_target_sec: 7200` drops 80 of the
  143 training rows whose window forces a taxi above an hour (their median target is 7,442 s). exp056 builds the
  window into the mixture (`model.mixture.window_column: lobt_delta_full`): windowed rows stay in the regressor's
  fit above the cap, candidates outside the window are invalid, predictions are clipped.
- Inside the window, where the lower edge exceeds NM's proxy by 900 s or more, 2025 targets sit a median
  2,100-2,900 s above the edge. On the holdout, raising the rows the model puts below the edge to edge + half the
  band's mean offset gains 0.5 s (312.25 -> 311.75; the full offset loses: the rows the model under-predicts sit
  nearer the edge); tuned on 98 rows, kept as an option only.

### 10.4 Smaller checks of the evening (30 Sep)

- **Forward windows** (`dep_following`: NM proxy mean, anomaly and count of the departures taking off in the next
  15 and 30 min, per airport and runway; two competitors report -0.7 / -1.4 s on the board): residual pre-test on
  lean2's not-delayed holdout rows +0.02 s (gate 0.15). `dep_live_state`'s +-15 min window already carries it. The
  block is registered (group `following`, `configs/exp058_lean2_window_following.yaml`) but not run.
- **Unmatched departures and the window.** 1,122 NM-unmatched departures (901 training, 221 ranking) can be linked
  to the NM-joined arrival of the same flight at another of the ten airports; in training only 2.6 % of their
  targets fall inside that arrival record's LOBT window, which confirms that "unmatched" means "block time more than
  an hour from NM's last off-block time" (typically a delay past LOBT with an ordinary taxi). The arrival record's
  AOBT_3 as a proxy has a median error of 183 s but outliers (RMSE 3,338 against our 825 on the 273 linked holdout
  rows), and those rows carry 0.55 % of the holdout's squared error: not pursued.
- **A second-stage corrector** on the holdout's out-of-fold predictions (`lab/oof_corrector.py`: gradient boosting
  on the clipped residual from the base prediction, stale probability and eleven NM/live features, rows selected
  on the base prediction <= 7,200 s, cross-fitted by day inside the holdout): 260.29 -> 259.74 without wrong-date
  rows at half weight. Small; a cheap option for the final stack.
- **The global regressor** (exp054, `per_airport: false`): holdout 324.6, without wrong-date rows 264.1, bulk 217.0
  (per-airport lean2 s42 323.6 / 263.4 / 214.8): a little weaker alone, a blend member of another shape.
- **An upper cap for unmatched typed rows** (in training, 95.7-100 % of unmatched typed departures 1 h or more late
  have their block time at least an hour after the schedule: NM's LOBT stayed at the schedule and the join
  failed; they are almost never stale): capping those rows' predictions at `schedule delta - 3,606` loses 0.2 s
  on the holdout (the 2 % of cap rows above the cap are the long targets). Not used.
- **Regime exclusion for linked untyped rows** (`lab/linked_untyped.py`). An unmatched departure's block lies
  outside NM's LOBT +- 3,606 s; for untyped Rome departures whose flight is NM-joined at its destination, a regime
  whose value would fall inside that window is impossible. In training (109 linked rows, 24 stale): stale
  impossible -> 66 of 66 genuine; a ~1,000 s taxi impossible -> 21 of 24 stale. With the rates from the fit
  months (p_stale 0.02 / 0.85) the 30 linked holdout rows go from 136e6 to 79e6 of squared error (about 0.3 s);
  36 ranking rows are linkable (17 must be genuine, 12 are likely stale). An option for the final stack.
- **Why the holdout hid the window.** AEZ146R (the only 2025 departure with all NM clocks a day before its takeoff,
  target 87,002 s) sits in the holdout but is within an hour of a whole day, so `rmse_ex_dayoffset`, the metric
  read for tail work since 16 Sep, excluded it; the headline metric shows the clip (328 -> 312).

### 10.5 Candidates after the window (30 Sep evening)

Since the window, the board and the holdout without wrong-date rows agree within 3 s (v31 263.17 against 260.29),
so holdout differences can be read as board differences again (with the board's seed noise).

| candidate | runs | holdout rmse / without wrong-date / bulk | board |
|---|---|---|---|
| v31 | v29 (17-run blend), window clip | 312.25 / 260.29 / 212.99 | 263.17 |
| v32 | 0.7 v29 + 0.3 exp054 (global regressor), window clip (`2026-09-30T20-48-11__blend17_global30_lobt__22b2a7f5`) | 310.6 / 259.2 / 212.4 | (1 Oct) |

The GPU queue of the night: exp056 (window in the mixture), exp057 (+ AOBT_3 anchor), exp059 (+ LOBT anchor),
exp060 (global regressor + window) seeds 42 and 43.
- **v32 prepared** (upload 1 Oct 00:00 UTC): `2026-09-30T20-48-11__global30_lobt_inwin_linked__stack` = 0.7 v29 +
  0.3 exp054, window clip, rows raised to the window's lower edge moved to edge + 0.5 x the fit-month mean of
  (target - edge) in their band of (edge - NM proxy) (factor 0.5 best on the holdout's 98 raised rows: 0.25 / 0.5
  / 0.75 -> 258.75 / 258.69 / 259.00 before the next step), and the linked-untyped regime exclusion
  (`lab/final_stack.py`). Holdout 309.93, without wrong-date rows 258.39 (v31 312.25 / 260.29). 163 ranking rows
  differ from the plain clipped blend (133 in-window, 30 linked Rome rows).
- **exp056** (exp051 + the window inside the mixture, `2026-09-30T20-47-41__lean2_window__4068fe60__s42`): holdout
  315.6, without wrong-date rows 260.4, bulk 214.4 (exp051 s42 clipped after the fact: 311.3 / 261.2 / 214.65).
  The headline difference is one untyped Rome wrong-date row (classifier noise); AEZ146R, the 2025 twin of
  ITY340, is now predicted at 84,619 s (exp051: 39,066). Inside the model the window adds 0.8 s over the post-hoc
  clip without wrong-date rows (single seed, within noise). In the blend: 0.45 v29 + 0.3 exp054 + 0.25 exp056,
  clipped, 258.75 without wrong-date rows, bulk 212.22 (v32's base 259.19 / 212.36).
- **What the largest remaining holdout errors are** (after the window, without wrong-date rows): almost all NM-unmatched
  rows, which have no window. Untyped Rome rows where the classifier picked the wrong regime (in both directions),
  and typed long-delay rows at EGLL, LFPG, LSZH and LTFM whose target sits near the schedule delta (block near the
  schedule, takeoff hours later: 2-4 % of unmatched typed rows 1 h or more late, which the model hedges at
  ~1,500-2,300 s). The linked-arrival exclusion does not carry over to typed rows: where a 1,000 s taxi would lie
  inside the linked window, most targets are still ordinary taxis (median 717 s), so that arrival record's LOBT
  is evidently not the one the departure's join failed on.

### 10.6 ADS-B surface data (from 30 Sep evening; user decision)

The user lifted the no-position-data rule on 30 Sep ("ADS-B would be fine") and asked what exactly it is useful for:
the information available *before* the off-block, or the live surveillance of the flight's own taxi. Source:
adsb.lol globe history (GitHub releases, one tar per day, ~2-4 GB, ODbL 1.0, "adsb.lol contributors"). The
organisers allowed ADSB.lol on 17-18 Sep (Discord, relayed in kind-mango's RECAP.md); the same relay says OpenSky
Trino access is not allowed and the ruling does not extend to OSN state vectors, so the user's OpenSky account is
not used.

- `python -m prc fetch-adsb` streams each day's tar over HTTPS (nothing but the kept points touches the disk) and keeps
  every point inside a +-0.10 deg x +-0.15 deg box around each airport at or below 5,000 ft; one parquet per day in
  `data/additional_data/adsb/points/`. Link ~13 MB/s: about 9.5 h for Jan+Jul 2025 and 2026 (124 days, three workers).
- Ground coverage differs sharply by airport (share of kept points on the ground in a pilot slice of 15 Jul 2025):
  EHAM 72 %, LSZH 46 %, LEBL 33 %, EDDF 28 %, LIRF 25 %, LEMD 19 %, EGLL 10 %, LFPG 2 %, EDDM 0 %.
- Two feature families, kept apart (`prc/adsb/events.py`): **B, own movement** (the linked aircraft's first ground
  sighting, first movement off its parked position, taxi start, stationary time after taxi start); **A, before
  off-block** (at NM's LOBT, without watching the flight: other aircraft on the ground and taxiing, and when this
  aircraft's inbound landed). Measured as a second stage on the Jan+Jul 2025 holdout, cross-fitted by day: A, B, A+B.
- **exp057** (exp056 + regressor anchored on NM's proxy at every airport): without wrong-date rows 260.8, bulk 215.6
  (exp056 260.4 / 214.4). Per airport on the bulk: EHAM -1.2, LSZH -0.3, but LIRF +5.6, EGLL +2.6, LFPG +1.0; the
  anchor helps where long proxies are real (EHAM) and hurts elsewhere, as the model review's CPU test said. exp061
  anchors EHAM only.
- **First measurement (5 holdout days, 28,281 departures, 14,846 ADS-B-covered;** `lab/adsb_stack.py`: a CatBoost second
  stage on the residual of the clipped v32 blend, cross-fitted by day). Without wrong-date rows / bulk: base 240.2 / 216.4;
  second stage without ADS-B 247.8 / 220.2 (on five days it overfits the rare tail: LIRF 939 -> 1,002); **A (before
  off-block) 248.8 / 220.4: nothing**; **B (own movement) 234.3 / 204.5**, per airport EDDF 183 -> 140, EHAM 157 -> 123,
  LEBL 212 -> 188, LSZH 272 -> 246, EDDM 158 -> 141, LEMD 192 -> 179; A+B 234.1 / 204.9. **ADS-B is useful for watching
  the flight's own pushback and taxi; what it shows before the off-block adds nothing the model does not already know.**
  At EDDF, LIRF, LSZH and LEMD the first ground sighting comes a median 5-11 min after the airport's off-block (stands are
  shielded; the aircraft is first received on the taxiway), at EHAM and LEBL within a minute; the second stage learns the
  stand-specific gap. LTFM has no low-level coverage at all; LFPG and EGLL little; coverage per airport differs between
  2025 and 2026 (2026 links 68-71 % of departures, 2025 51-53 %).
- **10 holdout days (56,024 departures, 29,405 covered):** without wrong-date rows / bulk: base 251.4 / 225.5; second stage
  without ADS-B 253.0 / 224.5; **B (own movement) 237.8 / 207.6** (EDDF 191 -> 152, EDDM 172 -> 137, EHAM 208 -> 171, LEBL
  205 -> 178, LSZH 251 -> 215, LEMD 188 -> 176, EGLL 235 -> 219; LFPG, LIRF, LTFM unchanged); B applied to covered rows
  only (the others keep the base) 238.7 / 208.8. About -14 s without wrong-date rows, -18 s on the bulk.
- **14 holdout days (78,561 departures, 41,484 covered), with the ADS-B times also given as differences to NM's proxy and to
  the base:** base 253.7 / 223.8; B 238.7 / 205.7; B on covered rows only 238.1 / 205.6 (EDDF 195 -> 150, EDDM 169 -> 134,
  EHAM 225 -> 178, LSZH 235 -> 193, LEBL 205 -> 174, LEMD 181 -> 171, EGLL 279 -> 258; LIRF 681 -> 697 through tail rows,
  which the apply step leaves alone by skipping rows with a stale probability above 0.05).
- **exp059** (exp056 + regressor anchored on LOBT, whose residual the window bounds to +-3,606 s): 260.8 / 216.2 (exp056
  260.4 / 214.4); worse alone, but it earns weight in a blend. **exp060 s42** (global regressor + window, 12,000-round
  cap): 260.8 / 216.6 (exp054, 4,000 cap, clipped after the fact: 262.4 / 217.0).
- **Six-member base** (weights searched on the clipped holdout in steps of 0.1: v29 0.2, exp054 0.1, exp056 0.1, exp057 0.1,
  exp059 0.2, exp060 0.3): 257.54 without wrong-date rows, bulk 212.21, both months better (Jan 250.6, Jul 263.0 against
  252.3 / 263.9). With the stack (in-window x0.5, linked exclusion): 257.04 (`2026-10-01T00-35-08__blend6_stack__stack`).
- **What the correction reads** (covered-rows model on the 14 days, CatBoost gain): the flight's own first sighting, taxi
  start and first movement relative to the base prediction (10.1, 9.7, 7.6), the distance from its stand to where it was
  first seen (9.4: how much of the taxi happened before reception), then the base itself, the stale probability and the
  schedule delay. The time stationary after taxi start (holds) 2.9, path length 2.3. Nothing of family A is in the model.
- **exp061** (exp056 + border_count 254 + proxy anchor at EHAM only): 260.7 / 214.7 (exp056 260.4 / 214.4); per airport on
  the bulk EHAM -2.2, LFPG -1.0, EDDF -0.7, LIRF +4.6 (Rome's bulk swings with the stale classifier between runs). Level
  overall; a blend member.

### 10.7 Uploads of 1 October (pre-registered before upload)

| version | run | what | holdout (all / without wrong-date rows) |
|---|---|---|---|
| v32 | `2026-09-30T21-48-31__v32b_stack__stack` | 0.45 v29 + 0.30 exp054 + 0.25 exp056, window clip, in-window x0.5, linked-untyped exclusion; no ADS-B | 310.26 / 257.97 |
| v33 | `2026-09-30T21-48-31__v33_adsb__stack` | v32 + the ADS-B own-movement correction (covered rows, p_stale <= 0.05), fitted on the 26 holdout days crawled by 23:50 UTC and applied to the 26 ranking days crawled (97,960 of 344,841 rows) | on the 26 days: 263.7 -> 246.9, without wrong-date rows 243.9 -> 225.6, bulk 217.3 -> 198.8 |

v33 - v32 is the board's reading of the ADS-B correction on 28 % of the rows.

**Board, 1 Oct 00:01 UTC: v32 = 260.72** (v31 263.17: the window-aware members and the stack, -2.45 s, against -2.3 on
the holdout); **v33 = 249.83** (-10.9 s from v32 with the ADS-B own-movement correction on 97,960 rows, 28 % of the
ranking set; board MSE 67,976 -> 62,417). The holdout predicted the size: on the 26 crawled days -18.3 s without
wrong-date rows. With every ranking day covered (the crawl finishes about 05:00 UTC) the same per-row gain would take
board MSE down by roughly another 7-8k.
- **A richer correction model** (the main model's whole 211-column matrix next to the ADS-B family, `--rich`): 225.39
  without wrong-date rows against 225.57 on the same 26 days. Not worth 200 more columns; the lean model stays.
- **Linking and features, second version** (1 Oct night): a departure without a callsign link takes the one ADS-B takeoff at
  its airport within 45 s whose aircraft is not linked yet (time-only links are as accurate as callsign links where
  coverage is good: EDDF median gap 376 against 384 s, EHAM -5 against -1 s, same spread; +~6 % coverage, most at EHAM, LFPG,
  EGLL); and "parked until" (end of the last stationary period of 2 min or more before takeoff, which at airports where
  the transponder comes on at pushback is the last hold, not the stand). On 38 holdout days (211,343 departures, 111,359
  covered): without wrong-date rows 237.6 -> 218.3, bulk 213.4 -> 194.2, covered rows 209.6 -> 164.9; weight 1.0 optimal
  (0.8 / 1.0 / 1.2: 219.2 / 218.3 / 218.9).
| v34 | `2026-10-01T00-35-08__v34_adsb__703d2e21` | six-member stack (`blend6_stack`) + ADS-B correction, second version (time-only links, parked-until), fitted on 41 holdout days, applied to 162,502 ranking rows (47 %) | on the 41 days: without wrong-date rows 238.2 -> 219.8, bulk 213.9 -> 194.8 |

**Board: v34 = 235.52** (v33 249.83, v32 260.72): the six-member base and the second-version ADS-B correction on 47 % of
the ranking rows. From the branch's start (v29 303.02) the board has moved 67.5 s: the join window ~40 s, the
window-aware blend and stack ~2.5 s, ADS-B own movement ~25 s so far.
- **Learning curve of the correction** (`lab/adsb_learning_curve.py`, fixed 11 evaluation days): trained on 10 / 20 / 30 other
  days, 237.8 / 234.5 / 234.1 without wrong-date rows (base 253.9). Close to saturation: crawling more 2025 months for
  training is not worth the bandwidth.
- **No stale signal in ADS-B.** Among covered departures 30 min or more late, the ADS-B features separate stale from genuine
  records with univariate AUCs of 0.50-0.61 (LIRF, EGLL, all airports) against 0.91-0.95 for the stale classifier; the
  aircraft's real movement says nothing about whether the airport recorded the schedule.
- **Where the error is after the correction** (41 holdout days, without wrong-date rows): LIRF about 40 % of the squared
  error (its covered rows read 532: the stale/delayed tail the correction leaves alone), LTFM about 20 % (no coverage),
  EGLL and LFPG most of the rest (partial coverage; EGLL is first received ~1.9 km from the stand).
- exp056 / exp059 / exp061 seed 43: 261.6 / 261.0 / 260.6 without wrong-date rows, bulk 215.5 / 217.0 / 215.2 (seed 42:
  260.4 / 260.8 / 260.7; 214.4 / 216.2 / 214.7).
- **Family C, the others' ADS-B taxis** (`lab/adsb_family_c.py`: mean and count of the ADS-B taxi readings of the other
  covered departures at the airport in the 30 min before this takeoff, for departures ADS-B does not see itself):
  on 38,843 uncovered holdout rows a second stage gives 223.2 -> 224.9 without it and 224.5 with it, so nothing over
  the base. The live airport state is already in the base through the other flights' NM proxies (`dep_live_state`).
  **ADS-B helps only where it observes the flight itself.**
| v35 | `2026-10-01T04-48-58__v35_adsb_full__e597f033` | eleven-run seed-averaged base (`blend11_stack`, holdout 311.81 / 257.11) + ADS-B correction fitted on all 62 holdout days (188,623 covered rows), applied to 237,291 ranking rows (69 %) | full holdout: 311.81 -> 297.99, without wrong-date rows 257.11 -> 240.16, bulk 211.99 -> 193.22 |

**Board: v35 = 228.20** (v34 235.52, v33 249.83, v32 260.72, v31 263.17; branch start v29 303.02). The full-coverage
ADS-B correction on the eleven-run base. Over the branch the board moved 74.8 s: join window ~40 s, window-aware blend and
stack ~2.5 s, ADS-B own movement ~32.5 s (v32 -> v35). On the 30 Sep standings 228.2 would be about 5th (1st 219.6).
- **Correction-model settings** (full holdout, without wrong-date rows; v35's default depth 6 / 2,000 rounds / lr 0.05 =
  240.16): depth 8 / 3,000 / 0.03 239.82; depth 6 / 4,000 / 0.03 240.06; three seeds averaged 240.06. **Taxi-regime
  correction for hedged rows** (`--regime`: rows with a stale probability above 0.05 get (1 - p) x the correction computed
  from the regressor's taxi answer, i.e. the ADS-B observation enters only the taxi regime's value): 5,200 holdout rows,
  240.16 -> 239.52, bulk 193.22 -> 192.24; 4,296 ranking rows.

| v36 | `2026-10-01T04-48-58__v36_adsb_d8_regime__beac2bc1` | v35's base and coverage; correction depth 8 / 3,000 rounds / lr 0.03, plus the taxi-regime correction on hedged rows (5,200 holdout, 4,296 ranking) | full holdout: 297.99 -> 297.14, without wrong-date rows 240.16 -> 239.08, bulk 193.22 -> 191.80; January 233.29 -> 232.06, July 245.56 -> 244.59 |

**Board: v36 = 226.40** (v35 228.20): -1.80 on the board against -1.08 on the holdout without wrong-date rows.
The depth-8 correction and the taxi-regime correction of hedged rows both carry over.
- **ADS-B and the stale decision, second test** (`lab/adsb_hedged.py`): on the 5,192 hedged covered holdout rows (stale
  probability above 0.05), a second model cross-fitted by day re-estimates the mixture weight between the stale value
  implied by the base and the ADS-B-corrected taxi value, from logit(p), the two values, the schedule delta and ADS-B's
  timing (observed taxi, and when the aircraft moved relative to its schedule). As a classifier on the nearer regime the
  log loss improves (0.519 -> 0.466) but the RMSE on those rows worsens (794 -> 817-827); fitted under squared error
  (p* = (y - taxi) / (stale - taxi), weight = gap squared) p alone gives 794 -> 790 (holdout -0.2 s, noise), and the
  values and ADS-B make it worse (800, 827). Together with the univariate AUCs of 10.7: **ADS-B carries no usable
  stale signal.**
- **Where the 2026 error sits** (holdout MSE per airport x covered cell, weighted to the 2026 cell counts; projects
  224.5 against the board's 226.4): with targets above 4 h trimmed, LTFM 18.5 % (no coverage; ordinary regression to
  the mean, day bias 3 % of its MSE), LIRF 29.5 % (stale tail), EGLL covered 11.7 % (overstated: in 2026 61 % of its
  covered departures are first seen within 100 m of the stand, against 27 % in 2025), LFPG uncovered 9.4 % (its
  holdout share is two rows above 4 h), LEBL covered 6.6 %. Coverage quality differs between the years: EDDF is seen
  later in 2026 (14 % first seen beyond 3 km, 2 % in 2025), LFPG's covered rows have a median of three ground points.
- **The correction under-corrects long observed taxis.** On holdout rows first seen at the stand whose ADS-B first
  movement implies a taxi 10-40 min above the base, the correction delivers about half the gap (residual +441, correction
  +395 at a mean gap of 1,700 s), because the same signature also marks bogus readings (a stray point at the stand hours
  earlier, an earlier tow or taxi-in in a continuous track). On EHAM's snow days in January 2026 (4-6 Jan) the
  correction averages +450 s. Second feature set (`stand_movement`: last point near the flight's own stand, the end of
  the last stop at the stand and of the last long stop anywhere, the stops after leaving the stand, the final
  continuously tracked stretch) built to separate the two.

### 10.8 Uploads of 2 October (pre-registered before upload)

| upload | run | what | holdout evidence |
|---|---|---|---|
| v37 | `2026-10-01T04-48-58__v37c_adsb_v2__6175aff9` | v36 + the second ADS-B feature set (`stand_movement`) and the signed takeoff offset in the correction (`--v2`), same settings (depth 8 / 3,000 / 0.03, taxi-regime correction) | full holdout 297.14 -> 296.78, without wrong-date rows 239.08 -> 238.64, bulk 191.80 -> 191.24; January 232.06 -> 231.52, July 244.59 -> 244.22; EDDF 121.0 -> 119.5, EDDM 123.1 -> 120.7, EHAM 116.7 -> 114.9, LSZH 143.4 -> 140.7, others within 0.5; long-gap rows (first seen at the stand, first movement > 10 min above the base; 629) 407 -> 380 |
| v38 | `2026-10-01T04-48-58__v38a_adsb_v4__6233174f` | v37 + the third ADS-B feature set (`--v4`: pushback detector, first-reception state, track quality, stop classes; all tier M, challenge-only), same settings | full holdout 296.78 -> 295.87, without wrong-date rows 238.64 -> 237.50, bulk 191.24 -> 190.00, projected 2026 222.32; January 231.52 -> 230.47, July 244.22 -> 243.03; every covered airport better (EDDF -4.3, EDDM -3.8, EGLL -2.8, LSZH -2.1, EHAM -1.9, LEBL -1.4, LIRF -0.7, LEMD -0.3, LFPG -0.3), LTFM unchanged (no coverage) |
| v39 | `2026-10-02T00-03-35__v39_blend_d8_d10__db293077` | equal blend of v38 (depth 8, CPU) and the same correction at depth 10 (GPU, `2026-10-01T04-48-58__v38b_adsb_v4_d10__b2ab474d`, alone 237.53 / bulk 189.75) | without wrong-date rows 237.50 -> 237.37, bulk 190.00 -> 189.71, January 230.47 -> 230.37, July 243.03 -> 242.87 (weights not tuned: 0.3 / 0.5 / 0.7 give 237.39 / 237.37 / 237.39) |
| v40 | `2026-10-02T03-11-48__v40_blend_v5__ffa9424c` | equal blend of the correction with the fourth ADS-B set (`--v5`: + inbound/turnaround, stand-type archive, time-to-location map; tier M) at depth 8 (CPU, `..._v40a_adsb_v5__38008f81`, 237.09 / bulk 189.54) and depth 10 (GPU, `..._v40b_adsb_v5_d10__c85f874f`, 237.13 / 189.33) | without wrong-date rows 237.37 (v39) / 237.50 (v38) -> 236.98, bulk 189.71 / 190.00 -> 189.29, January 230.37 -> 230.02, July 242.87 -> 242.44; against v39 per airport EDDF -2.4, LSZH -2.1, LEBL -1.5, EDDM -0.7, EHAM -0.7, LIRF -0.7, LEMD -0.4, but EGLL +0.65 and LFPG +0.16 (the time map on far-first-sighted EGLL rows; in 2026 most EGLL departures are seen at the stand) |
| v41 | `2026-10-01T04-48-58__v41a_adsb_v4_archive__9ee161ad` | v38 + the stand-type archive only (`--v5 --drop-prefix tmap_,tarr_,it_`; target-free per-stand ADS-B statistics, tier M as used), depth 8, CPU | without wrong-date rows 237.50 -> 237.34, bulk 190.00 -> 189.68, January 230.47 -> 230.36, July 243.03 -> 242.82; per airport EDDF -1.1, LSZH -0.9, EHAM -0.8, LEMD -0.6, LEBL -0.5, EGLL +0.4; 2026 changes against v38 rms 16-35 s (v40: 20-62). v41b (+ inbound without callsign columns, GPU) 237.39 / 189.80: inbound adds nothing without its callsign columns |
- **The stale decision is at its information limit** (1 Oct): at LIRF the classifier is calibrated in every probability
  band (late departures: observed stale rate 0.002 / 0.021 / 0.120 / 0.253 / 0.652 / 0.967 against mean p 0.003 / 0.024 /
  0.103 / 0.287 / 0.642 / 0.954) and per carrier; the error of the bands 0.05-0.8 is the hedging cost of honest
  uncertainty. Searched for a 2026-visible fingerprint without success: arrivals never carry stale block times (block =
  schedule on 0 % of late arrivals at every airport), and the seconds of the recorded times do not differ between stale
  and genuine records (takeoff time on a full minute 8.8 % against 8.5 %). Typed stale records at Rome cluster by carrier
  (TWB, LAV, TKJ, ISR 95-100 % stale), which the carrier stale rates already carry.

### 10.9 Pause of 1 October (~10:50 UTC): state and next steps

Stopped at the user's request to free the machine. Nothing is running; all code and findings are committed.

**State.** Board best v36 = 226.40. v37 (`2026-10-01T04-48-58__v37c_adsb_v2__6175aff9`) is built, its submission file
written (`submissions/2026-10-01T04-48-58__v37c_adsb_v2__6175aff9__submitting.parquet`) and pre-registered in 10.8.
Five uploads per UTC day; deadline 11 Oct 2026 21:59 UTC (safe cut-off).

**Next steps, in order.**
1. Upload v37 in the first slot after 00:00 UTC (2 Oct), then fetch and record the score (findings 10.8, the results
   table of `docs/paper.md`):
   `python upload_submission.py --run 2026-10-01T04-48-58__v37c_adsb_v2__6175aff9 --version 37`, about a minute later
   `python upload_submission.py --result --version 37`.
2. Finish the interrupted correction screens (GPU, ~30-50 min each; `--gpu` reads within ~0.1 s of CPU: v37c's settings
   238.75 on GPU against 238.64 on CPU, holdout without wrong-date rows). Base run for all:
   `B=2026-10-01T04-48-58__blend11_stack__b942ea3d`.
   - depth 10: `python lab/adsb_apply.py $B --depth 10 --iters 3000 --lr 0.03 --regime --v2 --gpu` (interrupted at
     ~9 min)
   - depth 10, slower: `... --depth 10 --iters 5000 --lr 0.02 --regime --v2 --gpu`
   - routed distance (`--v3`: `taxi_route` route length, runway crossings, route minus observed path):
     `... --depth 8 --iters 3000 --lr 0.03 --regime --v3 --gpu`
   A variant that beats 238.64 by more than ~0.2 s with both months improving: rerun on CPU with `--write v38_<name>`,
   `python -m prc submit --run <id>`, pre-register in findings, upload.
3. Cheap follow-ups on the correction: average two or three correction models of different depth (seeds alone gave
   nothing); weight the training rows to the 2026 airport x coverage mix and judge it with the projected 2026 metric
   (holdout MSE per airport x coverage cell weighted to the 2026 counts, 224.5 for v36 against 226.4 on the board),
   not the plain holdout.
4. Report: insert the final numbers in the abstract and section 4 after the last upload; the LICENSE (GPLv3, needed for
   prize eligibility) is the user's decision.

**Do not repeat** (measured on 1 Oct, see 10.7-10.8): ADS-B as a stale signal (two tests), an arrival block = schedule
fingerprint, a fingerprint in the seconds of the recorded times, per-airport blend weights of the base members,
snow-day under-correction (the medians agree), the midnight file boundary (~500 departures), more training days for
the correction (the learning curve saturates at 20-30 days).

**Run costs on this machine.** `lab/adsb_features2.py --workers 6`: ~50 min for 124 days (CPU); `lab/adsb_apply.py`
on CPU (12 threads): ~35 min per variant; with `--gpu`: ~30 min at depth 8, ~6.8 GB VRAM.

### 10.10 Local trials of 1 October afternoon (no uploads left today)

- **Routed distance in the correction** (`--v3`: `taxi_route` route length, runway crossings, route minus observed
  path), GPU screen, v37c settings: holdout without wrong-date rows 238.75 -> 238.69, bulk 191.29 -> 191.20: noise.
  The distance from the stand to the first sighting already carries what the route adds.
- **Feature screening harness** (`lab/idea_harness.py`): a small second stage, cross-fitted by day, on v37c's residual
  over the covered holdout rows without a stale candidate, with a fixed context (control) and context + candidate.
  Calibration: the control itself reads 238.89 (v37c 238.64: the second stage overfits slightly), a pure-noise column
  +0.12 against the control, an already-used feature (left_stand_50) +0.07. Bar for a candidate: -0.10 or better with
  both months improving. Links of every departure to its ADS-B aircraft are cached per day (`lab/adsb_links.py`).
- **Operational availability of every input** (user requirement, 1 Oct): a code audit (workflow: five readers, one
  critic re-checking every operational claim) classified all 70 model inputs into tiers O (known before off-block),
  T (anchored on the actual takeoff, operational once re-anchored), A (after-off-block information: the flight's NM
  actual off-block, other flights' later actual events), M (the flight's own movement, ADS-B), P (published later:
  PRU daily, ERA5) and C (challenge artefacts: join window, stale/wrong-date regimes, the NM-join-failure level of
  categoricals). Of the 51 inputs reaching the submission: 7 operational as computed, 7 after re-anchoring, 37 not
  before movement. The critic corrected two tiers (operator_presence and key_target_stats carry the untyped level
  that isolates Rome's stale-suspect segment: C) and eight submission flags (stand_runway_target_stats,
  nm_proxy_context, wx_past, wx_deice_load, stale_target_stats, key_target_stats, stand_geometry, taxi_route reach the
  base through the older 17-run blend at weight 0.20). Table: `docs/feature_table.md` (generated by
  `lab/availability_table.py` from `docs/availability_audit.json`); report section 3.6. Depth-10 correction (GPU):
  238.74 against 238.75 at depth 8 (bulk 191.06 against 191.29): no gain.
- **The final correction model's training rows** (`--cv-with-hedged`): v36/v37c's final 2026 model is fitted on the
  covered rows plus the 5,200 hedged rows that `--regime` adds (residuals against the mixture prediction), the
  cross-fitted models on the covered rows only. Cross-fitting with the hedged rows included, as shipped: holdout without
  wrong-date rows 238.54 against 238.75 (GPU), bulk 191.35 against 191.29, projected 2026 224.05. The inconsistency
  costs nothing (if anything -0.2 s): v37c stays as built and `--fit-fix` is not an improvement.
- **Challenge-only blocks never measured on the current line** (user, 1 Oct: challenge-only features stay in):
  `pru_daily` (tier P; in the 20 Sep sweep on exp028: +0.2 Jan+Jul, -5.3 winter, -62 disrupted, kept out only by the
  published-after-the-day policy; PRU daily covers Jan and Jul 2026 at 88-92 % non-null) and `wx_future` (tier A,
  never measured). Configs exp062 / exp063 = exp056 + the block, each with a Feb+Dec twin (`...w.yaml`) and the winter
  baseline exp056w; queued on the GPU with `--no-test`, permutation off. The PRU *monthly* table stays excluded (the
  user's earlier decision).
- **Exploration workflow, new ADS-B features** (five agents, one idea family each, screened with the harness, each
  winner re-checked by a skeptic for leakage, 2026 availability, duplication and a second seed). All tier M
  (challenge-only: the flight's own movement after off-block). Gain against the harness control (seed 42 / seed 7):
  - **pushback detector** (`lab/ideas/pushback.py`): on the ground most transponders report the heading in the track
    field, so a push is motion ~180 deg off the reported track (median 158 deg on near-stand rows; taxi ~1 deg). Push
    start: median error against the recorded off-block +6 s, MAE 29 s on 2025 push rows (first_move 52 s, first_seen
    53 s, left_stand_50 50 s); engine-start pause median 178 s. Found on 31 % of linked 2025 and 39 % of 2026
    departures. -0.72 / -0.70, every airport better (EDDM -8.8, LSZH -3.5, EGLL -3.0, EHAM -2.5).
  - **first-reception state** (`lab/ideas/first_reception.py`): speed and direction at the first ground point, route
    already covered, deviation from the stand's usual reception boundary. -0.39 / -0.30 (two pace columns dropped:
    they change meaning in 2026).
  - **track quality** (`lab/ideas/track_quality.py`): density, gaps, stand jitter, jumps, inbound landing, relocations.
    -0.35 / -0.24 (gap300_s, first_gs dropped as duplicates).
  - **stop classes** (`lab/ideas/stop_class.py`): the last long stop read as part of the taxi, its class and location
    statistics, the de-icing-like stop, the runway-area queue. -0.29 / -0.19 (ll_excl_gap dropped: duplicate).
  - analyst (residual analysis -> a second pushback signature plus carrier and aircraft type): -0.45 / -0.26, the
    identity half unstable alone; LIRF worse in every variant. Not taken (the pushback detector covers its signal).
  Assembled for both years in `cache/adsb/features3.parquet` (`lab/adsb_features3.py`); `lab/adsb_apply.py --v4`
  adds them to the correction (`--v4-only` for ablations).
- **Upload plan for 2 Oct (UTC):** v37 first (the stand/stop features alone), then v38 (+ the third feature set), so the board shows each step; later slots for a `pru_daily` member if its winter test passes. Mix weighting (`--mixweight`, rows weighted to the 2026 airport mix): 238.90 against 238.75, projected 2026 224.25 against 224.05: dropped.
- **`pru_daily` and `wx_future` on the current line** (exp056 + block, seed 42, GPU, `--no-test`; Jan+Jul baseline
  exp056 s42/s43: without wrong-date rows 260.45 / 261.58, bulk 214.38 / 215.48, disrupted 383.96 / 384.78; winter
  baseline exp056w: 225.0 / bulk 187.0 / disrupted 616.0):
  - `pru_daily` (tier P): Jan+Jul 261.26, bulk 213.34, disrupted 392.77; winter 225.6, bulk 187.6, disrupted 586.4.
    Mixed: the Jan+Jul bulk and the winter disrupted days improve, the winter holdout and its bulk and the Jan+Jul
    disrupted days get worse. The -5.3 s winter gain of the 20 Sep sweep (on exp028) is gone: ERA5, de-icing and
    live-state features now carry that signal, and on the board ADS-B observes the snow days' actual taxis at the
    covered airports. Not adopted.
  - `wx_future` (tier A): Jan+Jul 261.47, bulk 215.50, disrupted 385.65; winter 226.9, bulk 187.8, disrupted 637.3.
    Worse on the winter holdout, neutral on Jan+Jul. Not adopted.
- **Attribution inside v38** (CPU, same settings): the pushback detector alone (`--v4-only pushback`) reads 238.07
  without wrong-date rows (v37c 238.64), bulk 190.68 (191.24), projected 2026 223.18; all four families 237.50 /
  190.00 / 222.32. About half of the gain is the detector, half the other three families.
- **Second exploration round** launched against the v38a reference (harness repointed, frame includes the third
  set; calibration: control 237.85, v38a 237.505, noise +0.04): time-to-location map for the unseen part, stand-type
  archive, neighbour interactions, inbound/turnaround, residual analyst. Every feature computed for all 124 days and
  carrying its operational tier.
- v37's first upload (22:04 UTC, 1 Oct) was rejected by the scorer (DAILY_LIMIT_REACHED: local time 00:04 CEST was read as
  UTC). Nothing was scored; v37 and v38 are re-uploaded after 00:00 UTC.
- **Second exploration round, results** (against v38a; gain vs control at seed 42 / seed 7; all tier M):
  - **inbound and turnaround** (`lab/ideas/inbound_turn.py`): the departing aircraft followed back up to 24 h (inbound
    touchdown, extrapolated where ground reception is missing; the end of its inbound track; the link to the
    airport's arrival record: recorded in-block, arrival stand, unseen taxi-in; park locations, tows and callsign
    switches over the turnaround; off-block estimates from them). -0.63 / -0.59, both months (Jan 121.28 -> 119.47,
    Jul 182.23 -> 180.48). Caveat from the reviewer: the outbound-callsign field behaves differently in some 2026
    airport-months (EGLL and LSZH in July 2026).
  - **stand-type archive** (`lab/ideas/stand_archive.py`): per stand, from all 124 days of ADS-B: push-back vs
    power-out share, push and engine-start durations, switch-on lead, typical first-sighting distance; the best
    archive-based off-block estimate for what the flight shows. -0.29 / -0.30 (EDDM -6.1, LSZH -2.3, LEMD -2.1).
  - **time-to-location map** (`lab/ideas/timemap.py`): the typical time from push start to leaving each 75 m cell
    (per stand, zone, airport; 61,782 clean observed pushes) and from entering a cell to the recorded in-block
    (404,704 linked arrivals), applied to the flight's own cell exits. -0.24 / -0.25 (EGLL +2.4: watch).
  - neighbour interactions (`lab/ideas/interactions.py`): other aircraft moving or pushing near the stand, the observed
    runway queue: +0.02, nothing; its pooled unseen-time map duplicated fr_taxi_est. Rejected.
  - analyst, second round (`lab/ideas/analyst2.py`): the observed off-block against the schedule/LOBT/EOBT clocks:
    +0.01; with neighbour state +0.38 (day fingerprints). Negative.
  Assembled in `cache/adsb/features4.parquet` (`lab/adsb_features4.py`); `lab/adsb_apply.py --v5` = v4 + this set.

**Board: v37 = 225.65, v38 = 223.75** (2 Oct 00:02 / 00:04 UTC; v36 226.40). The stand/stop features: -0.75 on the board against -0.44 on the holdout; the third set (pushback detector, first reception, track quality, stop classes): -1.89 against -1.14. Board gains keep exceeding holdout gains by about 1.6x, consistent with the higher and closer ADS-B coverage of 2026 (EGLL push coverage 58 % against 20 %).

**Board: v40 = 224.05** (2 Oct 01:13 UTC; v38 223.75): the fourth ADS-B set does not transfer, although it gained 0.52 s on
the holdout. Where v40 departs from v38 in 2026: airport-months without a 2025 analogue (LFPG, LEMD, LIRF in January had
almost no 2025 coverage; holdout change rms 0.0-0.7 s against 33-52 s on the 2026 rows) and the inbound family's
callsign columns, whose meaning shifts in 2026 (reviewer's check: EGLL July 2026 76 % of departures first move under the
outbound callsign more than 60 s after their first ADS-B movement, at most 38 % in any 2025 airport-month; EDDF July 2026
median callsign-switch gap ~8,000 s against 7 s; LSZH July 2026 share taxiing under the old callsign 0.150 against
0.025-0.040). The time map's estimates also have a heavier tail in 2026 (EDDF, LIRF, LFPG) and were already worse at
EGLL. Lesson: a feature that depends on transponder behaviour (callsign setting) or on reception patterns can shift
between the years in a way the 2025 holdout cannot see. Next: v41a = v38 + stand archive only; v41b = v41a + the
inbound family without its callsign columns (it_cs_*, it_out_move*, it_old_cs_*).

**Board: v39 = 223.75, v41 = 223.59** (2 Oct ~03:12 UTC). The depth-8/depth-10 blend of the correction changes nothing on
the board (v38 223.75; holdout -0.13); the stand-type archive transfers: -0.17 on the board against -0.16 on the holdout.
New best v41. Of the fourth set, the inbound family's holdout gain sat in its callsign columns, which shift in 2026
(v41b without them: no holdout gain), and the time map is untested alone.

### 10.11 Uploads of 3 October (pre-registered before upload)

| upload | run | what | holdout evidence |
|---|---|---|---|
| v42 | `2026-10-01T04-48-58__v42a_adsb_archive_tmap__82dc1246` | v41 + the time-to-location map without its arrival-map half (`--v5 --drop-prefix tarr_,it_`), depth 8, CPU | without wrong-date rows 237.34 (v41) -> 237.08, bulk 189.68 -> 189.49, January 230.36 -> 230.20, July 242.82 -> 242.47, projected 2026 222.05 -> 221.77; per airport LEBL -0.9, EDDF -0.7, EDDM -0.7, LIRF -0.6, EGLL -0.4, EHAM -0.2, LEMD +0.1. With the arrival map (v42b, GPU): 237.35, EGLL +0.4: the arrival map is the harmful half |
| v43 | `2026-10-01T04-48-58__v43a_adsb_union__6ec50fde` | v42's configuration on the merged prod + staging ADS-B points (all features rebuilt, `lab/rebuild_adsb_union.sh`), depth 8, CPU | holdout without wrong-date rows 237.08 (v42) -> 237.12, bulk 189.49 -> 189.35, projected 2026 221.77 -> 221.70, January 230.20 -> 230.20, July 242.47 -> 242.55; airports within +-0.45; covered rows holdout +883, 2026 +1,193. A holdout tie: the board tests the extra 2026 coverage |
| v44 | `2026-10-01T04-48-58__v44_direct_blend__9c0fcf2b` | v43 blended on the covered rows without a stale candidate with a direct-formulation ADS-B model (`adsb_apply.py --direct --rich --v5`: CatBoost on the taxi time itself from the base matrix's 211 columns + the ADS-B features + the base prediction, GPU, depth 8 / 5,000 / 0.03; alone 238.70 with hedged rows uncorrected), weight 0.3 on the direct model (`lab/blend_direct.py`) | without wrong-date rows 237.12 (v43) / 237.08 (v42) -> 236.83, bulk 189.35 -> 189.00, January 230.20 -> 230.15, July 242.55 -> 242.07; blend weight curve 0.0 / 0.2 / 0.3 / 0.4 / 0.6 / 1.0: 237.12 / 236.87 / 236.83 / 236.83 / 236.99 / 237.93; per airport against v43 LEBL -1.1, LIRF -0.8, EDDF -0.7, LSZH -0.5, EDDM -0.3, LFPG -0.2, LEMD -0.1, EGLL / EHAM 0.0; 2026 change rms 13.4 s |

- **3 Oct plan (user):** upload only the top 1-2 candidates, keep the other slots. Gap to first place ~4 s (223.59 vs
  219.64) = 3.6 % of the squared error. Projected 2026 error budget at v42: LFPG uncovered 22 % (mostly two rows above
  4 h), LTFM uncovered 16 %, LIRF hedged (stale) 15 %, EGLL covered 9 %, LEBL covered 6 %, LFPG covered 5 %, LEMD
  covered 3.5 %. Istanbul and the uncovered Paris rows (37 %) have no surface data: an external-data scouting workflow
  looks for open, historically archived operational data for them (OpenSky excluded); the correction's regularisation
  sweep is the fine-tuning track.
- **External-data scouting** (workflow, four scouts; full list with licences and evidence in
  `docs/external_data_scout.json`): no open, historically archived, fair source has ground-level data for Istanbul,
  the uncovered Paris rows or Rome. Other ADS-B networks are proprietary, paid, evaluation-only or behind bot
  protection (ADS-B Exchange samples show LTFM ground coverage in July 2026 but are licensed for internal evaluation
  only; adsb.fi and airplanes.live have no open archive). EUROCONTROL's OPDI flight events (built from OpenSky, so under
  the reported exclusion) have essentially no ground events at LTFM, LFPG, LIRF either (11-19 Jan 2026: parking exits
  0 / 1 / 0, against 4,399 at EDDF and 9,611 at LSZH). Airline/airport APIs with actual off-block times (Schiphol, AF-KLM,
  FlightAware, ACARS OOOI) would be the label itself and are restricted or proprietary; the CC0 adsb.lol datalink
  archive (ACARS/VDL2) has no stations near LTFM or LIRF and no CDG OOOI, and its OUT times match the training targets
  to the minute (ground-truth-like; interception-law caveats). Actionable: adsb.lol's own `staging` release of
  globe_history_2026 (same source and licence; Jan 1-12, Jan 23-31 and all of July), not read by our crawl.
- **adsb.lol staging release** (same source, licence, format; a second aggregator instance with partly different
  feeders): one-day check 2026-07-11 (`lab/adsb_staging_check.py`): departures covered 67.1 % with prod, 70.4 % with
  prod + staging (staging only: LEMD +34, LFPG +33, EDDF +30, LEBL +30, EGLL +25, EDDM +17, EHAM +17, LSZH +11; LTFM
  and LIRF nothing that day). Staging exists for Jan/Jul 2025 and 2026 except 13-22 Jan 2026. Crawl started 2 Oct
  ~06:45 UTC (`prc fetch-adsb --kind staging`, ~400 GB streamed, ~2.3 GB kept; the streamer finds the release parts by
  their naming pattern when the GitHub API is rate-limited). Then `lab/adsb_merge_staging.py` (union into points/,
  originals kept in points_prod/) and `lab/rebuild_adsb_union.sh` (every ADS-B feature in dependency order, then the
  v42 correction as v43a).
- **Timetable regularity and schedule pressure** (user question, 2 Oct; v42a holdout residuals, bulk rows):
  - flight-level persistence is weak: a row's residual against the same (airport, flight number)'s mean residual on
    the other days of the month: r = 0.08 (a heavily shrunk adjustment: bulk 189.49 -> 188.97, but it needs same-month
    targets, which 2026 does not have); across seasons the flight means correlate at r = 0.10 and carrying January's
    shrunk mean to July makes July worse (208.28 -> 209.09 at k = 10, 208.52 at k = 30). Consistent with
    `flight_number_history` hurting in the 20 Sep sweep: stands, slots and aircraft change with the schedule season.
  - schedule pressure is already captured: mean residual by lateness at the planned off-block (LOBT - schedule) stays
    within -1 to -11 s on covered rows and -7 to -28 s on uncovered rows; very late uncovered flights (> 1 h) taxi
    ~20-30 s faster than predicted ("rushing"), but on 5,700 rows with RMSE ~280 that is worth < 0.1 s. The error's
    spread grows with lateness, not its bias.
- **Unfamiliar operators taxi longer** (user hypothesis, 2 Oct): valid once aircraft type is held fixed. Over 2025, the
  excess taxi against the median of the same airport x stand-runway route-distance decile x hour x aircraft type falls
  with the carrier's 2025 departures at the airport: <= 20 +23.5 s, 21-100 +23.0, 101-500 +7.0, 501-2k +2.5, 2k-10k 0.0,
  > 10k -1.5 (median); for an A320 +61 / +92 s for the two rarest bands, for an A321 +116 s. Without the type control
  the effect is masked (rare carriers and one-off flight numbers are faster: business jets, charters, quiet hours). The
  model already carries it: `operator_presence` (carrier and type frequency at the airport) with the aircraft type;
  v42's holdout residual within narrow-body types is ~0 on covered rows and slightly negative on uncovered rows for the
  rare carriers (mean -49 s on 238 rows). Carriers with <= 100 departures a year are ~3 % of 2026 departures. Nothing
  left to gain.
- **Day-to-day variability of identical flights** (user question, 2 Oct; 2025 bulk departures, typed): the same
  flight number from the same stand to the same runway at the same scheduled hour (groups of >= 5 days, 827,137
  departures) deviates from its own median by a median 103 s (75th pct 200, 90th 354; SD of the leave-one-out mean
  ~270 s). Per airport (median |dev|): EGLL 160, LIRF 156, LTFM 114, LFPG 112, LEBL 90, LEMD 88, EDDF 86, LSZH 82,
  EDDM 80, EHAM 74. On the same holdout rows v42's median absolute error is 63 s (75th 137, 90th 245), and a
  leave-one-out group-mean predictor reads RMSE 299 against v42's 176: static identity explains little, the day's
  dynamic information (NM clocks, traffic, weather, ADS-B) most. Spread already enters the base as features
  (stand-runway IQR, p90 by key); for RMSE only the conditional mean matters.
- **Correction hyperparameters on the v42 feature set** (GPU; v42a on CPU 237.08 / bulk 189.49, GPU reads ~0.1 higher):
  l2_leaf_reg 10: 237.44 / 189.73; lr 0.02 with 5,000 rounds: 237.22 / 189.54; depth 6 with 4,000 rounds: 237.39 /
  189.90. No setting beats depth 8 / 3,000 / 0.03: the correction's hyperparameters are exhausted.
- **Staging merge, actual coverage gain** (all 114 days with a staging release): departures covered 2026 70.06 % ->
  70.4 % (+1,193), holdout 56.4 % -> 56.5 %; +0.4-0.6 points per airport. The one-day check (2026-07-11: +3.3 points)
  was an unrepresentative day on which the prod release had a gap. Merged points per day rise ~1-2 % (2026-07-31: 1.63 M
  prod + 1.63 M staging -> 1.65 M). Expected value from coverage alone ~0.2 s; v43 (rebuilt features on the merged
  points) shows the combined effect of coverage and denser tracks.
- 3 Oct plan (user: top 1-2 only): v42, then v43; the other three slots unused.
- **Istanbul specialist** (night of 2-3 Oct, option 2; exp064 = exp056 on LTFM only, GPU, `--no-test`; LTFM holdout rows):
  baseline 243.24 (bulk 229.44); depth 9 / lr 0.05 241.39 (227.25); lr 0.03 / 12,000 rounds 241.94; border 254 / l2 10
  242.08; depth 10 / lr 0.03 / 12,000 242.03. The submitted 27-run stack reads 240.69 on LTFM; adding the depth-9 specialist
  at weight 0.3-0.4 gives 240.34 (both months slightly better), i.e. ~0.05 s overall; the baseline specialist adds
  nothing. More capacity in the base does not move Istanbul: not uploaded. The direct-formulation blend (v44) is the
  night's candidate.

**Board: v42 = 223.55, v43 = 223.43** (3 Oct 00:02 UTC; v41 223.59). The time map without its arrival half transfers weakly
(-0.04 on the board against -0.26 on the holdout); the staging-merged ADS-B points help a little (-0.12 on the board
against a holdout tie), consistent with ~1,200 more covered 2026 departures. New best v43.

**Board: v44 = 223.09** (3 Oct 00:03 UTC; v43 223.43): the direct-formulation blend transfers fully (-0.34 on the board
against -0.29 on the holdout). New best. Three uploads on 3 Oct (v42, v43, v44) as agreed; two unused.
- **Standings 3 Oct 08:15 UTC:** 1 jolly-lobster 219.07, 2 strong-crane 219.64, 3 vigorous-whistle 221.33, 4
  versatile-watermelon 221.85, 5 quick-boat 222.92, 6 adventurous-radio 223.09 (v44). Next steps: (1) more diverse
  second-stage models of the covered rows, since the direct formulation transferred fully (seeds of the direct model,
  a direct model with a different learner or target, the direct formulation for the hedged rows' taxi regime), blended
  with weights fitted on the holdout; (2) GPU use only with the user's go; uploads 3 Oct: 3 of 5 used (v42, v43, v44).

### 10.12 Where the 2026 error could sit: outliers, bias, disrupted days (user question, 3 Oct)

The user asked whether the remaining error at the error-prone airports comes from a few outlier rows or is spread,
whether 2026 carries the same outliers, and whether a per-airport bias change would pay; and to check for disruptions
in the 2026 months. A board probe (changing one airport's rows by a known amount and reading the RMSE) would reveal
test-label aggregates on the set the final score uses; we answer from legal data instead and keep uploads to genuine
model variants.
- **Concentration on the 2025 holdout (v44, without wrong-date rows).** LFPG: 28.7 % of the total squared error, 65 % of
  it in 0.1 % of its rows (the two records above 4 h); without its top 1 % its RMSE is 192 instead of 374. LIRF: 28.6 %,
  61 % in the top 1 % (the stale tail); 288 without it. **LTFM: 14.0 %, spread**: top 1 % 31 %, top 5 % 54 %, median
  absolute error 115 s (the highest after Rome), RMSE 167 even without its worst 5 %; bias -1.7 s. EGLL 10.5 % (top 1 %
  47 %). A constant shift per airport gains at most 0.3 s on that airport's rows in 2025; a scale around the airport mean
  at most 0.3 s (LTFM, k = 1.05).
- **2026 bias, read from the PRU monthly mean (the sanctioned level check).** Where the PRU reference share is >= 0.9
  and stable against 2025, the implied biases are small: LEBL Jul +38 s, LEBL Jan +19, EGLL -6/-7, LTFM Jul +8, EDDM
  Jul -1, EHAM Jul -1, LIRF Jan 0; upper bound of a perfect calibration 223.09 -> 222.87. LIRF Jul reads +60 but its
  share fell (0.97 -> 0.93: the PRU mean is a floor). The January gaps at the snow airports (LTFM +151, EDDF +173, LSZH
  +195, EDDM +258, LFPG +155) coincide with reference-share drops of 0.04-0.10 (de-iced flights excluded) and cannot be
  read. Calibrating the 2025 holdout to the PRU 2025 means hurts (236.83 -> 237.07): the populations differ. Not used.
- **Day-level bias.** On 2025 holdout airport-days, departures' mean (truth - prediction) is -4 to -18 s at every level
  of the arrivals' taxi-in anomaly (0..60 s: -4; 60..150: -10; 150..300: -6; 338: -18); a day-level uplift fitted on the
  anomaly moves the holdout 205.63 -> 205.60. v44 has no disrupted-day bias left to correct.
- **2026 target-free indicators.** LTFM: the NM proxy's share above 30 min tripled (0.9 % -> 2.7 %), p99 1,795 -> 2,401 s;
  v44's predicted January mean +199 s against January 2025; arrivals' taxi-in tail unchanged (p99 1,746 -> 1,729).
  By day, the rise sits on 12, 18 and 19 January (ERA5 snowfall 3.0 / 3.6 / 4.4 cm; predicted means 1,522 / 1,822 /
  2,109 s against ~1,150 normal; NM proxy means 1,504 / 1,719 / 1,827). LFPG: 2-7 January (snow, predicted 1,250-1,986
  s against ~920) and 29 January (1,482; arrival anomaly day). The model tracks the proxy on these days.
- **Disrupted 2026 airport-days by our detector** (arrivals' taxi-in anomaly > 150 s, >= 20 arrivals): EHAM 2-7 Jan
  (5 Jan +1,624 s) and 29 Jan, LFPG 7 Jan (+562), EDDF 26 Jan, EGLL 4 and 6 Jan and 13 Jul, LSZH 22 Jan, LIRF 5 Jul.
  PRU daily airport ATFM delays agree (EHAM 2-7 Jan 19-84 min per arrival, LFPG 5 and 7 Jan 18 and 24, EDDF 26 Jan 8.6,
  EGLL 8 and 29 Jan 8.7 and 8.6, LSZH many January and July days 3-11, EDDM 1 Jul 18.6). Event research (web) running.
- **Istanbul's snow days.** The base model (exp056, Feb+Dec 2025 holdout) under-predicts LTFM's heavy days: 19 Feb 2025
  truth 2,721 / predicted 2,055 s, 22 Feb 2,147 / 2,053, 23 Feb 2,844 / 2,342 (629 departures, RMSE 1,232 that day),
  11 Feb 1,906 / 1,526; the NM proxy's day mean was 1,963-2,259 s on those days. January 2026's snow days are milder:
  proxy means 1,504 (12 Jan), 1,719 (18 Jan), 1,827 (19 Jan), against ~1,200 normal. The 2025 relation between the
  day's truth and proxy means (truth 220-310 s below the proxy on normal days, +298 s above it on the four days with
  proxy > 1,800, SD 536) implies 1,371 / 1,625 / 2,125 s for those days against predictions of 1,522 / 1,822 / 2,109:
  no consistent direction (two over, one even), and the relation rests on 2-4 days per bin. No defensible adjustment;
  the February 2025 under-prediction occurred at proxy levels January 2026 did not reach.
- **Verdict on the probe idea.** Nothing found by legal means suggests a fixable per-airport bias or a few 2026 outlier
  rows we could target; LTFM's error is spread and its snow days are predicted in the range the proxy implies. A board
  probe would therefore mostly measure noise, at the cost of a practice the organisers watch for. Not done.
- **Events in the 2026 months** (web research workflow, five researchers + consolidation; `docs/events_2026.json`,
  crossed with our data in `docs/events_2026_crossed.csv`): 191 airport-days with a dated event, 95 of them expected
  to lengthen taxi-out at medium/high confidence. January 2026: the 2-9 Jan European cold spell (EHAM 2-7 Jan with 385-800
  cancellations a day, a de-icing fluid shortage on 6 Jan and a 70 % schedule cut on 7 Jan; LFPG 5 and 7 Jan with DGAC
  cuts of 15 % and 40 %; EDDF 5, 7, 12, 26, 29 Jan; EDDM 8-12 Jan freezing rain; LSZH 2 and 10 Jan de-icing crisis; EGLL
  de-icing without snow 3-8 Jan and a drone pause on 14 Jan), Istanbul's wind storm on 8 Jan (64 cancellations), its
  snow days 12 Jan (MADKOM restriction, 54 THY cancellations) and 18-19 Jan (15 % capacity cut) and fog/LVP on 23-24 Jan,
  WEF Davos at LSZH 19-23 Jan (one-hour closure for Air Force One on 22 Jan), the Tehran FIR closure on 14/15 Jan. July
  2026: thunderstorm days (EDDM 1 Jul with a 30-min handling stop, EDDF 13-17 Jul, LEMD 7 Jul, LTFM 22 and 25 Jul, LFPG
  16 Jul), LSZH's Bise low-capacity configuration on 10-13 and 20-24 Jul, a ground-handling strike at LEBL on 20 Jul, and
  LFPG's nightly runway-pair closures for taxiway works from 6 Jul to 14 Sep. Rome had no event in January.
  **The model reacts to these days**: on the 95 days its prediction uplift against the airport-month median correlates
  0.94 with the NM proxy's uplift; it rises by more than 100 s on 47 days (the proxy on 30), and no day shows a proxy
  uplift above 150 s with a prediction uplift below 60 s. Istanbul 12 / 18 / 19 Jan: predictions +335 / +635 / +922 s
  against the proxy's +56 / +271 / +379. Days our data flags without a researched event: EGLL 12-13 Jul (arrival anomaly
  137 / 179 s), EHAM 10, 25, 28 Jan, LFPG 2-4 Jan (the spell's onset), LTFM 27 Jul (proxy +269, prediction +241).

### 10.13 Towards 219: levers tested on 3 October (user: resources free, local tests first, then the best candidates)

- **Base blend weights refitted** (27 leaves, holdout OOF, window-clipped, targets clipped at 7,200 s for the fit, 5 day-folds):
  the current hand-set weights read 257.76 (bulk 212.23); a global non-negative least-squares fit 259.14 / 212.62 (it puts
  0.41 on one classifier_seeds5 leaf and 0.27 on one global_window leaf); per-airport fits 259.07 / 212.56 with Rome
  +9.6 s. Not a lever.
- **Per-airport weights between the residual correction and the direct model** (v43 / v44d, 11-point grid, day-CV): 236.93
  against 236.83 with 0.3 everywhere; in-sample optima 0.1 (EGLL) to 0.6 (LEBL) do not generalise. Not a lever.
- **Hypothesis with size: the ADS-B stage has never seen a snow day.** It is trained on the 62 crawled days (Jan+Jul 2025),
  which hold no de-icing operations, while January 2026 has ~5,000 covered departures on snow days (EHAM 2-7 Jan, LFPG 5-7,
  EDDF, EDDM, LSZH). February and December 2025 contain the data's snow days; crawl of those 59 days started 3 Oct
  (`prc fetch-adsb --days <Feb+Dec 2025>`, prod release). Plan: links and all four feature sets for the new days; the base
  for Feb/Dec training rows from the winter OOF runs (the seasonal stack has no out-of-fold prediction there); retrain the
  residual correction and the direct model on 121 days with cross-fitting by day; evaluate on the Jan+Jul days (comparable
  with 236.83) and on the Feb/Dec snow days separately.
- **The ADS-B stage is weakest on unusual days** (v44 vs base on covered holdout rows without a stale candidate, by the
  airport-day's arrival taxi-in anomaly): gain 29.8 % on calm days (< 30 s), 28.0 % (30-60), 13.6 % (60-100), 16.9 %
  (100-150), 22.4 % (> 150, 3,126 rows, RMSE 302); by the day's mean taxi-out: 35.5 % below 900 s, 16-20 % from 1,100 s
  up, 16.0 % above 1,600 s (2,148 rows, RMSE 499, bias -38 s). January 2026's snow days are such days. Base-blend
  re-weightings (older leaves at 0 / 0.1 / 0.3 instead of 0.2, equal weights): all within +0.1 / -0.1 s. Closed.
- **Pipeline for the extra training days** (prepared 3 Oct while the crawl runs): `lab/build_winter_base.py` builds a
  combined base run (the stack's Jan+Jul OOF, `src` 0, plus the mean of winter-split runs' OOF for Feb+Dec, window-clipped,
  `src` 1; smoke-tested with exp056w: 344,419 + 309,409 rows); `lab/adsb_apply.py` takes `base_src` as an input, falls back
  to the OOF's own p_stale / y_reg for winter rows, reports the Jan+Jul days and the extra days separately (the projection
  uses Jan+Jul rows) and writes only the stack's rows into the candidate's OOF; `lab/rebuild_adsb_extra.sh` extends every
  ADS-B feature to the new days (pooled archives rebuilt over all days) and refits. Winter base members: exp056w (done),
  exp059w and exp060w queued on the GPU after the depth-10 direct model (v45d). A data exploration of the 2025 snow /
  extreme-weather days (five analysts, synthesis in `docs/snow/SYNTHESIS.md`) runs in parallel (user request: what
  happened to traffic on those days, to inform the models).

### 10.14 The 2025 snow and extreme-weather days (user request: what happened to traffic, to inform the models)

Pooled pass over all airports (`cache/lab/snow/event_days_2025.csv`; per-airport detail by the exploration workflow in
`docs/snow/`). Event days from ERA5 (snowfall >= 1 cm, or snow on the ground below 1 C; >= 3 cm = heavy), PRU weather
ATFM delay >= 3 min per arrival, or our arrival-anomaly detector; compared with the median of the same airport's normal
days of the same month and weekday.
- **Impact by kind (2025):** heavy snow (16 days): departures 92 % of normal (cancellations), taxi-out mean +320 s, p99
  +1,232 s, share above 30 min 15.6 % against 1.9 %; snow (106 days): +159 s, p99 +682, 7.5 % against 1.9 %, no thinning;
  weather-ATFM days (255): +66 s; arrival-anomaly days (20, mostly Heathrow congestion): +96 s. The strongest 2025 days:
  LTFM 23 Feb +2,029 s (67 % of departures above 30 min, 10 % cancelled), 19 Feb +1,756, 22 Feb +1,086 (34 % cancelled),
  11 Feb +994; EDDM 24 Dec +1,059 (34 % cancelled), 27 Dec +565, 16 Feb +553, 30 Dec +538; LFPG 22 Nov +813 (freezing,
  no snow), 21 Jan +761, 3-4 Jan +400; EDDF 5 Jan +412; LSZH 4 Jan +364; EHAM 18 Jan +403.
- **The recorded off-block moves against NM's on snow days.** The day's mean (target - NM proxy) exceeds its normal value
  by +50 s on snow days and +102 s on heavy ones (EDDM 24 Dec +632, LTFM 23 Feb +702, EDDM 30 Dec +436): the airport
  records the off-block earlier relative to NM's AOBT_3 when de-icing happens, so the NM clocks under-state de-icing
  days by that much on top of the ordinary taxi uplift.
- **Arrivals do not show snow days.** Arrivals' taxi-in rises by +1 s (snow) and +6 s (heavy) while departures rise by
  +159 / +320 s: de-icing is a departure-side phenomenon. Our arrival-anomaly disruption detector (and the
  `arrival_conditions` features) therefore miss de-icing days; the "disrupted days" of the Jan+Jul holdout were Heathrow
  congestion days. A departure-side live signal is needed for snow days.
- Stale-looking records (target = schedule delta within 2 min) double on snow days (0.8 % against 0.3 %; heavy 1.9 %).
- **Course and models.** The uplift against the airport-month's normal hourly mean peaks in the 6 h after the day's first
  ERA5 snowfall hour (+371 / +378 s at 0-3 / 3-6 h), is +295 / +271 at 6-12 h and +180 at 12-24 h; the night hours (00-03
  UTC) carry the extremes (+1,000-1,800 s: snow clearance, few departures). The off-block shift against NM is airport-
  specific: LTFM +168 s, EDDM +122, EDDF +45, LFPG +34, LSZH +30, EHAM -16, EGLL -52. Models on these days: the winter
  base (Feb+Dec OOF, exp056w) reads bias -39 s / RMSE 400 on snow days and -162 / 727 on heavy ones (against 186 normal;
  partly an artefact of holding out the only heavy-snow month); on the Jan+Jul holdout v44 reads +8 / 136 on snow days
  and +20 / 158 on heavy ones, and **the ADS-B stage is strong on snow days** (covered rows: base 187 -> v44 104, heavy
  241 -> 149; the "weak on unusual days" reading above concerned congestion days). What remains on January 2025's snow
  days is the uncovered rows (base bias +48 s on 3,378 snow rows, +99 on 154 heavy ones: over-prediction) and the absence
  of any day as heavy as EHAM's January 2026 week in the ADS-B stage's training. Expected value of the Feb+Dec crawl
  revised down to a few tenths of a second. Immediate test: weather inputs for the residual correction (`--wx`: ERA5
  columns + hours since the day's first snowfall; it had none).

### 10.15 The snow-day synthesis, the day-level check, reception limits, and the model tests of 3 October

- **Exploration result** (five analyst teams, ten accounts in `docs/snow/<APT>.md`, synthesis `docs/snow/SYNTHESIS.md`,
  tables under `cache/lab/snow/`). Readings that matter for the model: (i) five airports had a 2025 day with more than
  half of the departures above 30 min (LTFM twice, LIRF, EDDM, LFPG); (ii) the worst days were not always snow (freezing
  fog at LFPG +1,031 s, EGLL +408, EHAM +355; thunderstorms at LIRF +1,228, LEBL +390; a runway closure at EDDF +263);
  (iii) the excess sits in different timing records per airport: at LIRF, LEBL, EDDF and the Spanish airports NM's proxy
  barely moves (gate holds), at LTFM the target is 1.5 x the proxy excess + 112 s, at LSZH 1.41 x, at EGLL 0.63 x (the
  proxy over-reads, since NM's AOBT comes before on-stand de-icing); (iv) **the base model holds the day level at eight
  airports (bias within +-65 s) and loses 1.5-7x the matched-day RMSE inside the day**: a 1-2 h lag at each surge and
  compression toward the mean (short flights +40..+200 over, the 30-60 min band -100..-330 under, LTFM's 1-2 h band
  -1,700..-2,100 = 57-81 % of its storm-day squared error); (v) in 2025 the airports delayed pushbacks instead of
  cancelling, so the "stale-looking" share falls on event days everywhere but Rome; (vi) the arrivals' detector is blind
  to de-icing (as 10.14 found); (vii) January 2026 against 2025 support: inside the range at EGLL, LTFM (19 Jan = the
  20-22 Feb 2025 level), LEBL and LIRF; at the maximum at LFPG (5 / 7 Jan, evening-weighted); beyond it at EHAM (3-7 Jan
  at 2.7-4.4x the 2025 proxy maximum, schedule 34-57 %: no analogue), LSZH (10 Jan 2.5x), EDDF (26 / 29 Jan 1.6-2.3x),
  EDDM (30 Jan fog; 19 deep-frost days whose 2025 twins ran +200 s with a quiet proxy) and LEMD (5 and 28 Jan). The
  synthesis closes with 15 testable encodings (hour-of-event features, bank x de-icing state, per-airport proxy
  correction under de-icing, backlog / hold-and-release counts, rare-runway share, METAR hours over ERA5 snowfall, a
  monotone dry-frost term, graded stand groups, carry-over, tail-aware live state, a storm-day regime, Rome's stale
  rates by day, no winter mass at the Spanish airports, daily level checks). All but the level check are base-model
  features, i.e. a retrain of the 27-member stack, which the remaining time does not allow; they are the backlog for a
  successor.
- **Day-level check of v44 on the strong 2026 days** (`lab/snow/level_2026.py`, synthesis item 15; `cache/lab/snow/level_2026.csv`).
  Per airport the day's mean target excess was regressed on the day's mean NM-proxy anomaly over 2025 (all days, and the
  days above +150 s: EHAM slope 0.98 r 0.87, LFPG 1.32 r 0.92, LTFM 1.82 r 0.85, LSZH 1.60, EGLL 0.62); the relation
  applied to the 2026 proxy gives the level a day "should" have. On the 92 airport-days of 2026 above +150 s (52,906
  departures) v44 sits on average 27 s **above** that level, the per-day gaps scatter +-300 s both ways inside the
  relation's own error, and the EHAM week reads +1,128..+1,948 s against implied +1,047..+1,731 (covered rows: the
  direct model sees the taxi). On the 22 strongest 2025 holdout days v44's OOF misses the day mean by -26..+64 s. **No
  systematic under-reaction exists to be corrected**; the day-level lever stays closed (10.13's negative uplift test
  agrees). The only airports with a negative mean gap are LTFM (-81 s over 23 days, v44 above the 2025 relation) and
  LIRF / EDDM (4-day fits, unreliable).
- **Reception limits of the allowed source** (ground points per day on 2026-01-07 / 2026-07-15, `on_ground` rows in
  `data/additional_data/adsb/points/`): EHAM 239k / 314k, EDDM 103k / 118k, LEMD 99k / 24k, LEBL 63k / 150k, LSZH 61k /
  38k, EGLL 54k / 137k, EDDF 23k / 17k, LIRF 12k / 0, **LFPG 2.4k / 12k, LTFM 0 / 17**. Coverage of the direct model's
  rows (linked with a first-seen point) in 2026: EHAM 97-98 %, LSZH 96, LEBL 96, EDDM 96-97, EDDF 87-88, EGLL 87-93, LEMD
  82-97, LIRF 78 % in January and 0 in July, **LFPG 39 %, LTFM 0**. Paris's share of the projected 2026 error (22 %,
  10.13) is therefore a reception limit, not a linking defect; nothing more can be fetched for it from adsb.lol.
- **Model tests on the Jan+Jul days** (reference v44: ex_wd 236.83; harness noise +-0.04):
  - weather inputs for the residual correction (`--wx`, ERA5 columns + hours since the day's first snowfall; run
    `v47w_resid_wx`): 237.07 ex_wd / 189.37 bulk against v42a's 237.08 / 189.49, i.e. neutral; blended with v44d at
    0.3: 236.81. The Jan+Jul days have too few snow rows to decide this; the 121-day refit's winter days are the judge.
  - direct model at depth 10, 5,000 rounds, GPU (`v45d_direct_d10`): 238.68 / 190.45 alone against v44d's 238.70 /
    190.85; with v43a at 0.3: 236.78; at 0.4: 236.78.
  - the mean of the two direct models (`prc blend` of v44d and v45d, `direct_avg2`) as the direct member at 0.3: 236.79 /
    188.92, January 230.10, July 242.05: equal to either alone, better founded. Written as `v45_direct_avg_blend`,
    **uploaded as v45** at 17:16 UTC (pre-registered: 223.0-223.1, a noise-level change against v44's 223.09; one of the
    day's two remaining slots, the other left unused as agreed). **Board 223.11** (v44 223.09): the tie the holdout
    predicted; v44 stays the best submission, and the board's own noise at this level is about +-0.03 s.
  Three structurally different variants inside +-0.05 s of v44 say the covered rows are at their information limit on
  these days; the day's remaining expected gain is the 121-day stage.
- **Winter base members** for the combined base: exp059w (`lean2_window_lobt_anchor_winter`) 223.0 and exp060w
  (`lean2_global_window_winter`) 220.4 on Feb+Dec 2025 (`rmse_disrupted` 580 / 531 on 4,509 rows), next to exp056w.
  The crawl delivered 58 of the 59 Feb+Dec days (31 Dec failed twice at once: no readable assets in its release);
  `lab/rebuild_adsb_extra.sh` started 17:09 UTC with the three winter runs.
- **Are the splits varied?** (user question.) No, by design: the base runs all use the same deterministic seasonal
  holdout (fit on ten months, validate on Jan+Jul 2025; `prc/validation.py` has no RNG), and the ADS-B stage assigns
  day i of the sorted holdout days to fold i % 5. Two runs therefore differ only in what was changed, which is what
  the pairwise deltas need; the price is that every selection decision since September was taken on the same 62 days,
  and decisions inside +-0.1 s are at risk of fitting them. What has varied: the model seeds (GPU fits are
  non-deterministic as well), the winter split (Feb+Dec holdout) for every snow-related judgement, `strict_forward`
  for drift, and now the 121-day stage, which changes the training composition. The board has moved with the holdout
  at every step from v36 to v44 (holdout -1.6 s, board -3.3 s), which is the evidence that the holdout is not yet
  over-fitted at the scale of the changes that were kept. `lab/adsb_apply.py --fold-seed N` now permutes the days
  before the interleave; three GPU runs of the v43a configuration (fold seeds 0, 1, 2) measure how far the holdout
  number moves with the fold assignment alone: [pending].

### 10.16 Checklist item (user, 3 Oct): a pushback-time prediction, per airport?

Idea: the off-block starts with the pushback clearance, the movement a little later; literature gives 2-5 min from
off-block to uncoupling; the component may differ between airports, so predicting it might add to the model. Measured
with the ADS-B pushback detector on the 50,294 covered 2025 departures with a detected, uncensored push
(`lab/ideas/pushback_prior.py`; seconds, medians):

| apt | n | recorded off-block -> push start | push (reverse motion) | push end -> first forward move | off-block -> own-power taxi | IQR | of the median taxi-out |
|---|---|---|---|---|---|---|---|
| EHAM | 19,420 | -25 | 63 | 174 | 217 | 141 | 28 % |
| LSZH | 3,983 | +5 | 92 | 168 | 276 | 102 | 40 % |
| LEBL | 7,801 | +4 | 95 | 173 | 284 | 116 | 33 % |
| EDDM | 10,282 | +13 | 92 | 174 | 289 | 91 | 37 % |
| EDDF | 3,659 | -5 | 98 | 213 | 332 | 185 | 38 % |
| LEMD | 1,285 | +93 | 91 | 199 | 411 | 137 | 32 % |
| LIRF | 614 | +75 | 40 | 312 | 480 | 339 | 47 % |
| EGLL | 3,244 | +121 | 84 | 236 | 494 | 208 | 41 % |

- The literature's 2-5 min holds: 275 s overall (IQR 209-352; push 82 s, engine-start pause 178 s), with a 4.5-minute
  spread between the airport medians. The recorded off-block is the motion start at Amsterdam (the push begins 25 s
  before it), Zürich, Barcelona, Munich and Frankfurt (within +-15 s), but precedes the push by 1.5-2 min at Madrid,
  Rome and Heathrow (a ready / clearance time rather than brake release), which is the airport-specific offset the
  snow synthesis saw in Heathrow's block times under on-stand de-icing.
- **Predictable before the movement?** Only weakly: smoothed group means fitted on half of the days and scored on the
  other half explain R2 0.084 of the off-block -> own-power interval from airport x stand, 0.042 from airport x
  aircraft type, 0.015 from the operator, 0.011 from wake class or hour, 0.094 all together (residual sd 875 s of
  919: the variance is a tail of long engine-start waits and holds after the push, not the push itself).
- **Already in the model?** Yes, both parts. On covered rows the detector's push start, push end, first forward move,
  durations and pause are inputs of the ADS-B stage since v38 (findings 10.10). For the rows without ADS-B the base's
  stand and type statistics absorb the predictable part completely: the base stack's residual regressed on the
  predicted interval has slope -0.01 (r -0.02; per airport -0.05..+0.07), and even the *observed* interval would move
  the base residual's RMSE only from 186.2 to 185.8 s (r +0.07): the residual is queue and holding, not pushback.
- Verdict: no RMSE to gain from a separate pushback prediction, per airport or per stand; the decomposition is a
  useful operational fact for the report (section 5) and the airport offsets explain part of the proxy's behaviour.
  Closed.

### 10.17 Literature review (user request, 3 Oct): protocol, search, synthesis

- **Protocol** `docs/background/literature_review_questions.md`: one main question and RQ1-RQ10 (definitions and horizons,
  drivers, data sources and ADS-B, methods, evaluation, disruptions, data quality, operational practice, competitions,
  gaps), each tied to the decision it informs; search strategy, extraction table with our availability tiers, synthesis
  plan (accuracy ladder, driver-by-feature matrix, methods table).
- **Search** through the Paperguide connector: eight analyst agents (one per question cluster), 187 distinct papers
  screened, 137 included, 131 in `docs/literature_extraction.csv`; mostly abstract-level, full texts read for Hotle
  2016, Pickup 2016, Jung 2019 and Wu 2026. 118 references with DOIs filed in the Paperguide folder "Taxi-out
  prediction (PRC 2026)" (78 by the workflow, 40 by hand after its call cap).
- **Synthesis** `docs/background/literature_review.md` (379 lines): per-question evidence with citations, a 29-row accuracy
  ladder with our three figures placed on it, the driver-by-feature matrix against `docs/feature_table.md` with our
  measured deltas, a 60-row methods table, and 14 gap claims each marked "supported by the search" or "absence of
  evidence only". Headline: single-airport predictors report 55-174 s on cleaned or undisclosed populations with
  undeclared horizons; our 260 s (pre-movement) and 223 s (own ADS-B) over uncleaned ten-airport year-ahead records are
  a different quantity; the availability tiers, the two-family ADS-B split with its reception bound, the regime mixture
  over record meanings and disrupted-day error reporting have no counterpart in the retrieved set. A related-work
  paragraph with the key citations now sits in the report's introduction.
- **Open points** the synthesis flags: several ladder figures are abstract-level or disputed between analysts (Idris
  2002, Balakrishna 2008, He 2022); the ladder could not be normalised by each airport's taxi-out spread (no abstract
  reports it); grey literature (A-CDM Implementation Manual, PRU ATXOT documentation, SESAR deliverables) is not
  indexed by Paperguide and is being read directly for an addendum; the previous challenge editions (2024 take-off
  weight, 2025 fuel burn) are deliberately left out, since their rules and targets differ from this task (user decision,
  3 Oct); a few excluded items
  (ATMAP weather scoring, Taszarek 2020 ERA5 climatology, Dalmau 2024 / Beltman 2025 distributional delay models,
  Verkerk 2018 TOBT definition, Hoffmann 2009 Munich de-icing pads) may deserve re-admission.
- **Grey-literature addendum** (web, 3 Oct; `docs/background/literature_review.md` section before the references): the A-CDM
  Implementation Manual v5.0 gives a three-horizon taxi-time tolerance of +-7 / +-5 / +-2 min (3 h-2 h, 2 h-30 min,
  30 min-AOBT before off-block; "still tentative") and states that remote de-icing is included in EXOT; the PRU's
  additional-taxi-out reference is the median of congestion-filtered unimpeded flights per runway-by-stand grouping (the
  20th percentile is only an intermediate; Kühn 2017's reading corrected in RQ8); the closest European comparator to our
  setup is Swatowska & Gabagnou 2025 (SESAR Innovation Days): one CatBoost over EBBR, LKPR, LFPO, LOWW, LSZH, LTFM,
  trained 2022-23, tested 2024, departure MAE 1:42-3:32 against standard taxi-time tables 2:03-3:45 (gains of 1-6 % at
  Zurich and Istanbul), pooled RMSE 4.05 min over both phases with 0.1 % trimming; Lim 2021 (KATL GNN 76 s RMSE
  pre-departure, the PRU method 199 s as a predictor); no deployed European ML EXOT with before-and-after figures was
  found. Five rows added to the extraction table and the ladder; a wanted list of full texts we could not retrieve is in
  `docs/background/literature_wanted.md` (user may supply PDFs).

### 10.18 The ADS-B stage on 121 training days (4 Oct): the first gain since v44

- **Rebuild** (`lab/rebuild_adsb_extra.sh`, 3 Oct 17:20-20:42 UTC): 58 Feb+Dec 2025 days crawled (31 Dec unreadable), every
  ADS-B feature family extended to them, the pooled archives (stand archive, time-to-location map) rebuilt over 182 days,
  the combined base built from the stack's Jan+Jul OOF plus the mean of three winter-split members (exp056w, exp059w,
  exp060w; `2026-10-01T04-48-58__base_plus_winter__8f857b25`, 344,419 + 304,694 rows).
- **v46a residual correction on 121 days** (`--regime --v5 --drop-prefix tarr_,it_`, depth 8, 3,000 rounds, CPU; run
  `v46a_resid_121d__3a39ad8b`): Jan+Jul holdout ex_wd **236.48** (v43a 237.12, i.e. -0.64), bulk **188.83** (189.35),
  covered 236.53 (237.71); projected 2026 220.74 (v43 line 221.7). On the extra winter days the base reads 219.43 ex_wd and
  the corrected 194.93 (covered rows 176.78 -> 124.95), which is the same relative gain as on Jan+Jul. The hypothesis of
  10.13 held in its revised form: not a snow-day effect, but more and more varied training days for the covered-row model
  (62 -> 121), worth about 0.6 s on the holdout.
- **v46 = v46a + mean(v44d, v45d) at 0.3** (`lab/blend_direct.py`, run `v46_resid121_direct_blend__ff7a34cf`): ex_wd
  **236.30**, bulk 188.49, January 229.61 (v44 230.10), July 241.55 (242.05); with v44d alone at 0.3: 236.33. **Uploaded as
  v46 at 06:17 UTC on 4 Oct** (slot 1 of 5), pre-registered expectation 222.3-222.7 on the board (holdout -0.53 against
  v44, both months improving; board gains have run 1.0-1.6x the holdout). **Board 222.11** (06:20 UTC): -0.98 against
  v44, 1.8x the holdout gain, the new best submission.
- **GPU jobs stalled overnight.** The fold-seed chain (seed 1) and the 121-day direct model (v46d) started within ten
  minutes of each other on the one GPU (the waiter's process check did not see the other job), shared 8 GB and both
  wrote to the same `catboost_info` directory; after 12 h neither had finished (3,000-round fits took 17 min instead of
  3). The fold-seed chain was stopped at 06:20 UTC; v46d continues alone and is restarted if it has not written within
  the hour. Fold seeds 1 and 2 are re-run afterwards (seed 0 on the GPU reproduced the CPU reference: 237.14 vs 237.12).
- Next: v46d (direct model on 121 days) as a second member -> v47 candidate; a `--wx` refit on the 121 days (the winter
  days judge the weather inputs); fold-seed spread.
- **Standings after v46** (leaderboard API, best score per team, 06:25 UTC 4 Oct): 1 strong-crane 216.57, 2 vigorous-whistle
  217.86, 3 jolly-lobster 218.99, 4 charming-eel 219.94, 5 versatile-watermelon 221.85, **6 adventurous-radio 222.11**,
  7 quick-boat 222.92, 8 zesty-puzzle 223.96; 41 teams, 1,000 scored submissions. The leader moved from 219.07 to 216.57
  within a day. Note on the day count: "121 days" is the label of the runs; the stage's training days are the 62 Jan+Jul
  holdout days plus the 58 crawled Feb+Dec days = 120.
- **Next extension** (started 06:30 UTC): crawl of Mar+Nov 2025 (61 days, 4 workers; `runs/fetch_marnov.out`) and then
  Jun+Aug 2025 (`runs/fetch_junaug.out`); month-pair base members `exp056n/059n/060n` (Mar+Nov) and `exp056s/059s/060s`
  (Jun+Aug) queued on the GPU after v46d (`runs/queue_monthpair_base.out`); `lab/build_winter_base.py` now takes several
  comma-separated groups and `lab/rebuild_adsb_extra.sh` a MONTHS list, so the next refit can use about 240 training days.
- **v46d, the direct model on the 121 days** (GPU, depth 8 / 5,000 / 0.03, `v46d_direct_121d__eea53f5e`; 12 h wall time
  because it shared the GPU with the fold-seed run until 06:20 UTC, then 1 h alone): Jan+Jul holdout alone 238.23 ex_wd
  (v44d 238.70, -0.47), bulk 190.38 (190.85); winter days 219.43 -> 196.16. Blends with v46a: v46d alone at 0.3 =
  236.30 (Jan 229.52, Jul 241.62: January better, July worse than v46); mean(v44d, v45d, v46d) at 0.3 = **236.27**,
  bulk 188.49, Jan 229.56, Jul 241.54 (both months at or below v46's 229.61 / 241.55). Written as
  `v47_resid121_direct_avg3__*` but **not uploaded**: 0.03 s against v46 is inside the harness noise (+-0.04) and the
  board noise (+-0.03); the slot is kept for the month-pair refit. The three-model direct average is the direct member
  for the next candidates; a direct model on the ~240 days is worth training when the GPU is free (the direct member
  gained 0.47 alone from 62 -> 121 days).

### 10.19 Independent data-science review (4 Oct; `docs/background/datascience_review.md`)

Two fresh reviewer agents (no project history; lens A practice and tuning, lens B levers and ideas) read
`docs/background/approach_report.md` and the repository, an editor checked every recommendation against this ledger. Verdict:
practice strong (structural leakage control, deterministic time splits, measured noise floors, pre-registration,
negatives recorded); the score came from three data discoveries, not tuning; 0.5-1.5 s more is realistic this week,
not the 5.5 s to the leader. Of 23 distinct recommendations 10 are new, 12 partly tried, 1 a do-not list.
- **Adopted now (no compute):** per-process CatBoost `train_dir` in `lab/adsb_apply.py` (the shared `catboost_info`
  stalled two GPU jobs for 12 h); the resolved flag set, parameters, seeds and fold seed are written into every stage
  run's metrics.json. **Queued on the CPU behind v46w:** fold seeds 1 and 2 (`runs/foldseed_cpu_{1,2}.out`), so the
  stage's fold-assignment noise is measured before the next 0.1 s decision (prerequisite (a) of the review).
- **Planned, in the order of expected gain per hour:** (2) finish the 240-day stage for both members with two day-kind
  ablations (62 + Feb/Dec only; 62 + Mar/Nov only), in progress; (3) re-tune the stage at the larger size incl. the
  residual clip +-3,600 -> +-7,200 s and the direct target clip 7,200 -> 10,800 s; (1) seed-average the stage members;
  (5) the stacking-mismatch diagnostic (apply the shipped stage to a fold-mean 2026 base; the refit and the fold mean
  differ by rms 64 s on 2026); (4) a shrunk shift for rows whose window forces a taxi above 30 min (day-CV -0.039 s);
  (6)/(8) one structurally different base member (Lossguide or a per-airport-tuned member) if the GPU allows.
- **Not to spend time on** (the ledger's numbers): regressor loss changes, target transforms, global classifier
  recalibration, a meta-learner over the leaves, monotone constraints, more seeds of the same base, hedges for the typed
  NM-unmatched rows, day-level uplifts.
- **Red flags for the project owner:** no LICENSE file while GPLv3 is a prize condition; the adsb.lol permission known
  only via a competitor's relay ("not verified by us"); one 2026 row moved the board by 40 s; the stage trained on
  ten-month fold residuals is applied on a twelve-month refit (undiagnosed); the Amsterdam January 2026 near-stand rows
  with implied taxis of 4,000-6,000 s lie outside the 2025 support; Barcelona's July stale cell may be under-called
  (reviewer B; A dissents); documentation drift (120 vs 121 days, -1.6 vs -2.25 s, two snow-day stale-share readings).
- **v46w, weather inputs on the 121 days** (`--wx`, run `v46w_resid_121d_wx__760464aa`, CPU 06:20-09:35 UTC 4 Oct): Jan+Jul
  236.49 / 188.83 against v46a's 236.48 / 188.83; winter days 195.00 against 194.93 (covered rows 125.11 against
  124.95); projected 220.75 against 220.74. The winter days were the judge the Jan+Jul test lacked, and they say the
  same: the correction has no use for ERA5 snow depth, snowfall, frost hours or the hours since the first snowfall on
  top of what the base and the ADS-B timings carry. **Closed.**
- **Crawl of the shoulder months (4 Oct):** Mar+Nov 2025 complete (61 days); Jun+Aug 51 of 61 days: adsb.lol has no
  release for 1-10 June 2025 on either aggregator (`no release v2025.06.0N-planes-readsb-{prod,staging}-0`), so the
  archive now holds 294 day files (124 Jan+Jul of both years, 58 Feb+Dec, 61 Mar+Nov, 51 Jun+Aug). Month-pair base
  members: Mar+Nov three done (exp056n/059n/060n), Jun+Aug exp056s/059s done, exp060s training (started 14:30 UTC after
  the user's GPU window). Fold-seed noise: permuted assignment 1 reads 237.10 against the default's 237.12 (CPU);
  seed 2 running.
- **Fold-assignment noise of the ADS-B stage, measured** (review prerequisite (a); CPU, v43a configuration on the 62-day
  base): default interleave 237.12 ex_wd / 189.35 bulk, permuted assignment 1 237.10 / 189.49, assignment 2 237.02 /
  189.32; projected 2026 221.70 / 221.66 / 221.64. Spread 0.10 s ex_wd, 0.17 s bulk. Rule adopted: a stage change
  must move ex_wd by at least 0.2 s (twice the spread) with both months improving before it is uploaded alone; v46's
  -0.53 clears it, v47's -0.03 does not (kept as a fallback blend, not uploaded).
- **Rebuild over all extra months started 14:55 UTC** with the two finished Jun+Aug members (exp056s, exp059s; exp060s
  still training on the GPU, its folds running past 6,000 rounds) so that the 240-day refit (`v48a_resid_240d`,
  CPU) can be blended and uploaded within the 4 Oct UTC slots; the GPU chain runs the re-tune screen after exp060s
  and then the 240-day direct model (`v48d`).
- **Upload policy for the last week (project owner, 4 Oct 15:40 UTC):** slots that would expire unused go to the best
  available candidate; the 0.2 s bar applies only when a slot has a competing use. **v47 uploaded 15:43 UTC** (slot 2 of
  4 Oct; v46a + mean(v44d, v45d, v46d) at 0.3; holdout 236.27 against v46's 236.30, January 229.56, July 241.54;
  pre-registered as a noise-level change, 222.0-222.2). **Board 221.96: -0.15 against v46, the new best.** A change to
  the direct member moved the board five times as far as the holdout, as at v44 (holdout -0.29, board -0.34): the
  121-day direct model carries winter days the Jan+Jul holdout cannot score, and January 2026 is a snow month. The
  holdout therefore under-reads direct-member changes trained on more days; for those the extra (winter) days'
  numbers are the better guide (v46d: winter days 219.43 -> 196.16, covered rows 165.83 -> 111.05).
- **Stage re-tune screen at the 121-day size** (review item 3; GPU, base `base_plus_winter__8f857b25`, ex_wd / bulk /
  winter days ex_wd): control d8 / 3,000 / 0.03 236.66 / 188.94 / 195.02; residual clip +-7,200 s 236.72 / 189.22 /
  195.07 (worse: closed); **d8 / 6,000 / 0.02 236.42 / 188.69 / 194.79 (-0.24 / -0.25 / -0.23)**; **d10 / 5,000 / 0.02
  236.46 / 188.49 / 194.71 (-0.20 / -0.45 / -0.31)**; direct target clip 10,800 running. At twice the training days
  the stage wants more rounds at a lower rate, as the review predicted; both settings clear the 0.2 s bar on all three
  readings. Device caveat: the GPU control reads 236.66 against the CPU's 236.48 for the same setting (v46a), a 0.18 s
  gap where seed 0 at 62 days showed 0.02, so the tuned settings are refitted on the CPU (d8 / 6,000 / 0.02, about 10 h
  at 240 days) for 5 Oct, with a GPU d10 / 5,000 / 0.02 member alongside as a second residual model.
- **v48, the ADS-B stage on 240 days** (4 Oct): rebuild over all 294 crawled days 14:51-16:13 UTC (170 new days:
  58 Feb+Dec, 61 Mar+Nov, 51 Jun+Aug); combined base `base_plus_winter__1d136cf8` from three month-pair groups (Feb+Dec
  three members, Mar+Nov three, Jun+Aug two: exp060s finished after the rebuild had started); residual refit
  `v48a_resid_240d__f099928a` (CPU, d8 / 3,000 / 0.03, 16:13-21:12 UTC): Jan+Jul **236.16** ex_wd (v46a 236.48, -0.32),
  bulk 188.44 (188.83), projected 220.34 (220.74). Blended with mean(v44d, v45d, v46d) at 0.3
  (`v48_resid240_direct_avg3__0a20dc54`): **235.99**, bulk 188.15, January 229.53, July 241.06 (v47 236.27 / 188.49 /
  229.56 / 241.54). Uploaded 21:13 UTC (slot 3 of 4 Oct); **board 221.52: -0.44 against v47, the new best** (1.6x the
  holdout gain). 121 -> 240 days gave about half the board gain of 62 -> 121 days (-0.44 against -0.98), as the review
  expected for less varied shoulder months. Standings 21:16 UTC: 1 strong-crane 213.92, 2 vigorous-whistle 216.88,
  3 jolly-lobster 218.99, 4 charming-eel 219.94, 5 versatile-watermelon 220.08, **6 adventurous-radio 221.52**; 56 teams.
- Running overnight: the 240-day direct model (`v48d`, GPU, due about 22:15 UTC) for a v48 variant with four direct
  models in the average; then a d10 / 5,000 / 0.02 residual member on the GPU (`v49g`); the tuned residual member
  d8 / 6,000 / 0.02 on the CPU (`v49c`, about 10 h). The 62 + Mar/Nov ablation was stopped for the tuned refit; its
  base `base_plus_winter__a4ae3681` is kept.
- **v48d, the direct model on 240 days** (GPU, d8 / 5,000 / 0.03, `v48d_direct_240d__f6d84ab5`): alone 238.00 ex_wd
  (v46d 238.23), bulk 189.83 (190.38), projected 222.93. Direct target clip 10,800 at 121 days (screen): 238.05 against
  238.23, bulk 190.24 against 190.38, winter days 196.12 against 196.16: -0.18, inside the fold spread, not adopted yet.
- **v49 = v48a + mean(v44d, v45d, v46d, v48d) at 0.3** (`v49_resid240_direct_avg4__fff52299`): holdout 235.98 / bulk
  188.13 / Jan 229.53 / Jul 241.05, a tie with v48 (235.99); uploaded 21:50 UTC as the best available candidate for an
  expiring slot. **Board 221.40: -0.12 against v48, the new best.** Third time a direct-member change trained on more
  days moved the board further than the holdout (v44 -0.29 / -0.34, v47 -0.03 / -0.15, v49 -0.01 / -0.12).
- **v50 = v49 with direct weight 0.4** (single-factor test in the last expiring slot; holdout 236.01 / 188.10):
  **board 221.50, +0.10 against v49.** The board does not want more direct weight; v49's gain came from the 240-day
  direct model in the average. 0.3 stays. All five 4 Oct UTC slots used (v46 222.11, v47 221.96, v48 221.52, v49
  221.40, v50 221.50); best v49.
- **v49g, a depth-10 residual member on 240 days** (GPU, d10 / 5,000 / 0.02, `v49g_resid_240d_d10__a11aad79`): 236.15
  ex_wd, bulk 188.24, extra days 189.88, projected 220.31 (CPU default v48a: 236.16 / 188.44 / 190.05 / 220.34). A tie on
  the holdout, better on the bulk and the extra days, and run on the GPU, which read 0.18 worse than the CPU at 121 days.
  **v51 = mean(v48a, v49g) + mean of four direct models at 0.3** (`v51_resid240x2_direct_avg4__b11cd630`): 235.94 /
  187.99 / Jan 229.47 / Jul 241.02 (v49 235.98 / 188.13 / 229.53 / 241.05): both months better; armed for upload at
  00:01 UTC on 5 Oct (first slot of the day), pre-registered 221.30-221.40.
- **v51 uploaded 00:01 UTC 5 Oct** (slot 1 of 5 Oct): **board 221.36**, -0.04 against v49, as the holdout said
  (-0.04). The new best; a residual-member average transfers one to one, a direct-member change more than that.
- **v50d, direct model on 240 days with the target clipped at 10,800 s** (GPU, `v50d_direct_240d_clip10800__a1998738`):
  238.01 / bulk 190.07 against v48d's 238.00 / 189.83 at the default clip: no effect at 240 days. **Closed** (the
  121-day screen's -0.18 was noise). As a fifth direct model in the average (`direct_avg5__a7295a1e`) the v51 blend reads
  235.94 / 188.00, identical to v51; held back, not uploaded.
- Queued on the GPU overnight: a depth-10 direct model on 240 days (`v51d`) and the tuned residual setting d8 / 6,000 /
  0.02 on the GPU (`v52g`), for a morning blend of up to four residual members (v48a, v49g, v49c, v52g) and six direct
  models.
- **Overnight GPU members (5 Oct):** v51d, a depth-10 direct model on 240 days (d10 / 5,000 / 0.03,
  `v51d_direct_240d_d10__0c3e222a`): **237.75** alone, bulk 189.34, projected 222.67, the best single direct model
  (v48d at depth 8: 238.00 / 189.83). v52g, the tuned residual setting on the GPU (d8 / 6,000 / 0.02,
  `v52g_resid_240d_d8_6000__cdfc4223`): 236.16 / 188.40, a tie with the CPU default v48a (236.16 / 188.44). Blends
  against v51 (235.94): residual pair + all six direct models 235.92; three residual members + six direct 235.93; with
  only the three 240-day direct models 235.97-235.98. All within the noise.
- **v52 = mean(v48a, v49g) + mean of six direct models (v44d, v45d, v46d, v48d, v50d, v51d) at 0.3**
  (`v52_resid240x2_direct_avg6__56da37b2`; 235.92 / 187.97 / Jan 229.46 / Jul 241.01), uploaded 04:26 UTC 5 Oct as a
  single-factor step from v51 (direct side only): **board 221.28, -0.09, the new best.** Fourth direct-member change in
  a row that moved the board further than the holdout (v44 -0.29 / -0.34, v47 -0.03 / -0.15, v49 -0.01 / -0.12, v52
  -0.02 / -0.09). Working hypothesis: the direct models trained on more days carry winter and shoulder-month structure
  that January 2026 rewards and the Jan+Jul 2025 holdout cannot score; more and stronger direct models are the cheapest
  lever left. GPU next: a depth-10 direct model at lr 0.02 with 8,000 rounds (`v53d`).
- **v49c, the tuned residual member on the CPU** (d8 / 6,000 / 0.02, 240 days, `v49c_resid_240d_d8_6000__00ab754d`,
  21:13-07:00 UTC): **236.00** ex_wd, bulk 188.29, extra days 189.86, projected 220.15: the best single residual model
  (CPU default d8 / 3,000 / 0.03: 236.16; the same setting on the GPU, v52g: 236.16). The review's re-tune at size paid
  -0.16, and the CPU is worth 0.16 at this setting. Blends with the six direct models at 0.3: v49c alone 235.84 / 187.98
  / Jan 229.44 / Jul 240.88; mean(v49c, v49g) 235.87 / 187.92; three or four residual members 235.89.
- **v53 = mean(v49c, v49g) + mean of six direct models at 0.3** (`v53_resid240tuned2_direct_avg6__6193232a`), uploaded
  07:06 UTC (slot 3 of 5 Oct), pre-registered 221.15-221.25: **board 221.21, -0.07 against v52, the new best.**
- **Seed averaging started** (review item 1; `lab/adsb_apply.py --seed-offset N`): a seed-142 copy of the tuned
  residual member on the CPU (`v54c`, about 10 h) and, after `v53d` (direct d10 / 8,000 / 0.02), a seed-142 copy of the
  depth-10 direct model on the GPU (`v54d`). Candidate this evening: residual mean(v49c, v54c, v49g) + the direct
  models.
- **v53d, a depth-10 direct model at a lower rate** (d10 / 8,000 / 0.02, 240 days, `v53d_direct_240d_d10_8000__08b23845`):
  237.69 / 189.27 / extra 190.73 / projected 222.54, the best single direct model (v51d at 5,000 / 0.03: 237.75).
  **v54 = mean(v49c, v49g) + mean of seven direct models at 0.3** (`v54_resid240tuned2_direct_avg7__c5745c26`; 235.86 /
  187.90 / Jan 229.42 / Jul 240.93; with only the four 240-day direct models 235.90), uploaded 07:51 UTC (slot 4 of
  5 Oct): **board 221.18, -0.02 against v53, the new best.** Fifth direct-side change in a row at or beyond the holdout
  change, but the step has shrunk to the board's noise: the direct average is close to saturated at seven models.
- **Full-year extension of the ADS-B stage started** (5 Oct ~08:00 UTC): crawl of Apr, May, Sep and Oct 2025 (122 days,
  `runs/fetch_fullyear.out`); month-pair base members `exp056a/059a/060a` (Apr+Oct) and `exp056m/059m/060m` (May+Sep)
  queued on the GPU after `v54d` (`runs/queue_fullyear_base.out`). Then one rebuild over all ten extra months, a
  combined base from five month-pair groups and the residual and direct refits on about 360 days. Expected value by the
  two earlier steps (62 -> 121: board -0.98; 121 -> 240: -0.44 with the direct member): about -0.2 s.
- **v54d, seed-142 copy of the depth-10 direct model** (`v54d_direct_240d_d10_s142__7863471e`): 237.77 / 189.31 /
  extra 190.79 / projected 222.60 against seed 42's 237.75 / 189.34 / 190.85 / 222.67: seed spread 0.02 s for this
  model. Eighth member of the direct average for the evening candidate.

### 10.20 Review items 4-13: status and the forced-long-taxi shift (5 Oct)

Status of the review's 13 recommendations (`docs/background/datascience_review.md`): 1 (seed averaging) running; 2 (more
ADS-B days) done for 240 days, full-year extension running (the day-kind ablation is moot once every month is in);
3 (re-tune at size) done: d8 / 6,000 / 0.02 on the CPU -0.16, residual clip 7,200 worse, direct clip 10,800 neutral;
4 (forced-long shift) done, below; 5-13 not started as of 5 Oct 10:30 UTC.
- **Item 4, a shrunk shift for rows whose NM window forces a taxi above 30 min** (`lab/forced_long_shift.py`; rows with
  L = takeoff - LOBT - 3,606 s above 1,800 s; mean residual per band of (prediction - L) shrunk with k = 30, day-CV on
  the holdout). On v54: 351 holdout rows, truth 1,689 s above the edge, prediction 1,469 s, mean residual +219 s, positive
  in every band (+154 / +413 / +213 / +83 for 0-300 / 300-900 / 900-1,800 / above 1,800 s). Day-CV: ex_wd -0.095 (Jan
  -0.075, Jul -0.111), four of five folds better; robust to k (10: -0.098, 100: -0.074) and the threshold (900: -0.065,
  1,200: -0.052, 2,400: -0.199). Covered rows carry the larger shortfall (+287 s against +153 s uncovered: the stage's
  corrections are shrunk on very long taxis); gated to covered rows -0.058, to uncovered -0.023. By airport the gain sits
  at EGLL, LFPG and LIRF; EHAM's 56 holdout rows are neutral (+0.48 dMSE/N) while 221 of the 526 2026 rows are EHAM's
  snow week. **v55 = v54 + the shift** (reviewers' pre-specified threshold 1,800 s, not the CV-best 2,400;
  `v55_forced_long_shift__8f21533c`; 2026 mean shift +174 s), uploaded 10:25 UTC (slot 5 of 5 Oct), pre-registered
  221.03-221.13: **board 221.15, -0.04 against v54, the new best**, below the range, consistent with the EHAM caveat.
- **Item 11, a per-airport level shift** (day-CV on v55, eligible rows: prediction < 3,600 s, p_stale < 0.001): -0.05 on
  ex_wd but January -0.13 and July +0.01; a shift fitted on January and applied to July costs +0.09, the reverse +0.04.
  The airport level is month-specific (LFPG -22.5, LIRF -26.8, LTFM -18.7, LEMD -15.7 s in January, near zero in July),
  so nothing says it transfers to January 2026. **Rejected**, as reviewer A expected.
- **Item 5, the stacking mismatch:** all 27 leaves kept their fold-mean ranking predictions; the rebuilt refit blend
  matches the saved one exactly; on 2026 the fold-mean blend differs from the refit blend by 39.5 s rms (mean -1.7 s,
  correlation 0.998; 3,586 rows above 120 s). Base copy `base_plus_winter_foldmean2026__1d136cf8` (2026 predictions =
  stack + that difference, window-clipped; holdout OOF unchanged); the depth-10 residual is refitted on it in the GPU
  window (`v56f`) to compare its 2026 output with v49g's.
- **Item 7, a stage member from another learner family:** `lab/adsb_apply.py --xgb` (XGBoost hist, native categoricals
  with frozen training levels; same cross-fit, hedged-row treatment and 2026 application); smoke test running; the
  240-day XGBoost residual member (`v56x`, d8 / 3,000 / 0.03, GPU) runs first in the window after the six base members.
- **Full-year base members done** (5 Oct 10:03-14:14 UTC, 6 x 37-42 min): Apr+Oct holdout 198.8 / 200.1 / 198.8 (bulk
  187.1-188.6), May+Sep 260.5-261.2 with wrong-date rows, 208.9-209.8 without (bulk 191.2-192.6). Crawl: adsb.lol has no
  release for 28 May to 10 June 2025 on either aggregator (the June gap extends back four days), so the archive ends at
  412 day files once Apr/May/Sep/Oct finish (118 of 122 new days readable).
- **Item 7 result, an XGBoost residual member on 240 days** (GPU, hist, d8 / 3,000 / 0.03, `v56x_resid_240d_xgb__443ed3cb`):
  237.38 ex_wd / 189.36 bulk / projected 222.00 alone (CatBoost CPU tuned v49c 236.00). In the residual average with the
  two tuned CatBoost members, blended with the seven direct models at 0.3: equal thirds 235.88 / 187.82, XGBoost at 0.2
  235.85 / 187.82, against v54's 235.86 / 187.90. The bulk gains 0.08, the holdout 0.01: below the reviewers' own
  falsification bar (gain under 0.08 s on the holdout). Kept as an optional member, not a priority.
- **GPU handed to the project owner at 14:47 UTC 5 Oct** (needed from 15:45 UTC): the fold-mean mismatch test (`v56f`,
  item 5) was stopped two minutes into its fit and must be re-run (base `base_plus_winter_foldmean2026__1d136cf8`); the
  360-day GPU chain is held until `runs/gpu_window.done` exists. The CPU keeps the residual seed copy (`v54c`) and then
  the 360-day rebuild and tuned refit.
- **Seed averaging, result** (review item 1): the seed-142 copy of the tuned CPU residual member (`v54c_resid_240d_d8_6000_s142__633ad95d`)
  reads 236.11 against seed 42's 236.00 (seed spread 0.11 s for the residual model; 0.02 s for the depth-10 direct model).
  Blends at 0.3: residual (v49c, v54c, v49g) + eight direct models 235.86 / 187.91; two CPU seeds + eight direct 235.85;
  v54's residual side 235.86. A holdout tie, as variance reduction tends to be. **v56 = three residual members + eight
  direct models at 0.3 + the forced-long shift** (`v56_seedavg_shift__5ab258dc`; shift day-CV -0.098; 2026 mean shift
  +171 s), armed for upload at 00:01 UTC 6 Oct (slot 1), pre-registered 221.10-221.15.
- 360-day rebuild started 17:23 UTC (288 new days over ten extra months; `runs/rebuild_360.out`).
- **360-day base built** (5 Oct 17:23-19:33 UTC; rebuild over 412 crawled days, 288 new): `base_plus_winter__80cbb34d`,
  344,419 holdout rows plus 1,740,628 rows of the ten other 2025 months from five month-pair groups of three members
  each (Feb+Dec 309,409; Mar+Nov 326,781; Jun+Aug 374,324; Apr+Oct 360,962; May+Sep 369,152). The tuned CPU residual refit
  on it (`v55c`, d8 / 6,000 / 0.02) started 19:35 UTC (about 18 h); the GPU models on it are held while the project
  owner uses the GPU.

### 10.21 The ADS-B stage on 360 days (6 Oct): v57 under 221

- **360-day members** on base `base_plus_winter__80cbb34d` (all twelve 2025 months, 412 crawled days): v55g, a depth-10
  residual model (GPU, d10 / 5,000 / 0.02, `v55g_resid_360d_d10__f7e949fa`): **235.85** ex_wd, bulk 188.02, projected
  219.95 (the 240-day equivalent v49g: 236.15 / 188.24 / 220.31; the best 240-day member, CPU v49c: 236.00). v55d, a
  depth-10 direct model (GPU, d10 / 8,000 / 0.02, `v55d_direct_360d_d10_8000__3ca2df95`): **237.40** / 189.08 / 222.25
  (240 days: 237.69). Blends at 0.3: v55g + eight direct models 235.71; + nine (with v55d) 235.70; v55g + v55d alone
  235.69; mixing v55g with the 240-day residual members is worse (235.75-235.78): the 360-day model dominates.
- **The fold-mean mismatch test (item 5)**: the depth-10 residual on the 240-day base with fold-mean 2026 base
  predictions (`v56f_resid_240d_d10_foldmean__6e0b129c`) has the same holdout as v49g by construction (236.15) and a
  2026 correction of mean +0.6 s / rms 149.3 s against v49g's -0.9 s / 147.1 s; the comparison of the two 2026 outputs
  is in progress.
- **v56 was never uploaded**: the timed upload at 00:01 UTC failed on a DNS outage (the scorer's endpoint did not
  resolve; the notebook workflow's fix pass failed on the same outage). Superseded by v57.
- **v57 = v55g + mean of nine direct models at 0.3 + the forced-long shift** (`v57_resid360_direct9_shift__6f5a85fc`;
  holdout 235.70 before the shift, shift day-CV -0.091 with Jan -0.075 / Jul -0.103; 2026 mean shift +177 s), uploaded
  04:31 UTC (slot 1 of 6 Oct): **board 220.99, -0.16 against v55, the new best**, the first submission under 221.
  The steps of the stage's training days on the board: 62 -> 121 days -0.98, 121 -> 240 -0.44 (+ the direct member
  -0.12), 240 -> 360 -0.16 (both members together, with tuning and seeds already in v55).
- **Item 5 closed (stacking mismatch).** Pass-through of a base change on 2026 (fold-mean against refit base, depth-10
  residual on 240 days): covered rows base difference 30.2 s rms, final difference 22.9 s, slope 0.41 (the stage
  absorbs about 60 % of a base change); uncovered rows 54.9 / 55.0 s, slope 1.00. Board test **v58 = v57 with the
  uncovered rows moved halfway to the fold-mean base** (`v58_uncov_half_foldmean__b4d5a70c`; 100,548 rows changed,
  rms 15.3 s over all rows), uploaded 04:35 UTC (slot 2 of 6 Oct), pre-registered 220.94-221.04: **board 221.79, +0.79
  against v57.** The refit on all twelve months (January and July 2025 included) is far better for 2026 than the fold
  models trained on ten months, and the stage is rightly applied on the refit base. The seasonal analogue months
  carry real information for the scored months: a lesson for any future base retrain (never drop them from the
  final fit).
- **More 360-day residual models** (6 Oct): v55c, the tuned CPU model (d8 / 6,000 / 0.02, `v55c_resid_360d_d8_6000__59e75cb5`,
  19:34-08:45 UTC): 235.86 / 188.15 / projected 219.98; v58g, the seed-142 copy of the depth-10 GPU model
  (`v58g_resid_360d_d10_s142__e6924d0b`): 235.91 / 188.06 / 220.01 (seed spread 0.06 s at 360 days). Blends with the nine
  direct models at 0.3: v55g alone 235.70, mean(v55g, v55c) 235.66, mean of all three 235.67.
- **v59 = mean(v55g, v55c, v58g) + nine direct models at 0.3 + the forced-long shift**
  (`v59_resid360x3_direct9_shift__5945c633`; shift day-CV -0.093), uploaded 08:48 UTC (slot 3 of 6 Oct), pre-registered
  220.90-221.00: **board 220.85, -0.14 against v57, the new best.** A residual average over two learner shapes and two
  seeds moved the board 4.5 times the holdout change (-0.03), the largest such ratio so far: variance reduction that a
  same-season holdout barely sees.
- **v58d, the seed-142 copy of the 360-day direct model** (`v58d_direct_360d_d10_8000_s142__15a9c9a9`, 7.8 h on the GPU
  while the CPU ran a 12-thread fit): 237.46 / 189.11 / projected 222.28 against seed 42's 237.40 / 189.08 / 222.25.
  Blends with the three 360-day residual models at 0.3: nine direct models (v59) 235.67; ten (+ v58d) 235.66; **only the
  two 360-day direct seeds 235.65** / bulk 187.66 / Jan 229.17 / Jul 240.74.
- **v60 = mean of three 360-day residual models + mean of the two 360-day direct seeds at 0.3 + the forced-long shift**
  (`v60_all360_shift__60cbc9bb`; every member trained on all twelve months of 2025), uploaded 14:09 UTC (slot 4 of 6 Oct),
  pre-registered 220.78-220.88: **board 220.68, -0.18 against v59, the new best**, on a holdout change of -0.02. The
  direct models trained on 62 to 240 days had turned from diversity into dilution once 360-day direct models existed:
  the direct average should hold only models trained on the most days. Next: more 360-day direct seeds.

### 10.22 A longer-trained base (project owner's request, 6 Oct)

The ADS-B stage gained from more rounds at each increase of its training days; the owner asked to try the same on the
base model. Plan: (1) exp065 = exp056 (the per-airport window member, weight 0.10 in the stack) at learning rate 0.04,
round cap 8,000, patience 400, with the refit on all twelve months; bar = beat the mean of exp056's two seeds (260.45 /
261.58 without wrong-date rows, 214.38 / 215.48 bulk) by more than their 0.6 s spread. (2) Only if it passes: exp066
(global regressor, weight 0.25; cap 24,000) and exp067 (LOBT anchor, weight 0.20), a new stack with the long members,
and the new base used first on the rows the ADS-B stage does not correct (about 30 % of 2026: all of Istanbul, most of
Paris), where v58 showed a base change passes straight to the board; a stage refit on the new base only if time allows.
The GPU residual seed copies and the CPU seed copy were stopped for it (the lowest expected value in the queue).
- **exp065 passes** (`2026-10-06T16-27-57__lean2_window_long__16c2b7dd__s42`, 14:36-16:17 UTC with the refit): 259.63 ex_wd /
  213.79 bulk against exp056's two-seed mean 261.02 / 214.93 (seed 42: 260.45 / 214.38): -1.4 / -1.1, twice the seed
  spread. exp066 + exp067 started 16:17 UTC (`runs/queue_base_long.out`).
- **The swap pipeline** (`lab/base_swap.py`): the submitted stack is reproduced exactly from its blend by
  `lab/final_stack.py <blend> --inwindow 0.5` (max difference 0.0 s on the holdout and 2026); the new blend keeps the
  submitted runs and weights with each retrained configuration swapped in at its total weight; the new stack's
  difference to the old one is added on the rows neither ADS-B stage corrects (holdout and 2026 alike), clipped to the
  window. Test with exp065 alone (weight 0.10): stack 257.11 -> 257.04 ex_wd; candidate v60pre 235.650 -> 235.603 ex_wd,
  bulk 187.657 -> 187.604, Jan 229.173 -> 229.123, Jul 240.743 -> 240.698; 2026 uncovered rows (102,061) move by
  4.1 s rms.
- **exp066, the global regressor trained longer** (`2026-10-06T18-17-11__lean2_global_window_long__6d1b7355__s42`, 130 min):
  259.29 ex_wd / 214.79 bulk against exp060's two-seed mean 261.45 / 217.25: -2.2 / -2.5, the larger gain.
- **v61 = v60 with the two finished long members swapped into the base on the uncovered rows** (exp065 for exp056 at
  0.10, exp066 for exp060 at 0.25; `v61_swap2_shift__411dc8da`): new stack applied to 149,713 holdout and 102,061 2026
  rows not corrected by either ADS-B stage (2026 rms change 13.2 s); holdout 235.650 -> 235.492 ex_wd, bulk 187.657 ->
  187.448, Jan 229.17 -> 228.98, Jul 240.74 -> 240.62; with the forced-long shift (day-CV -0.092). Uploaded 22:15 UTC
  (slot 5 of 6 Oct), pre-registered 220.33-220.53: **board 220.63, -0.05 against v60, the new best**, a third of the
  holdout gain. The three-member swap (with exp067, the LOBT-anchored member at 0.20) is armed for 7 Oct.
- **exp067, the LOBT-anchored member trained longer** (`2026-10-06T20-26-54__lean2_window_lobt_anchor_long__8b4e2ed3__s42`,
  3.9 h while the CPU built the v61 swap): 259.93 ex_wd / 215.32 bulk against exp059's two-seed mean 260.94 / 216.62:
  -1.0 / -1.3. All three retrained members pass; the gain grows with the member's round count (global regressor, 24,000
  cap, largest).
- **v62 = the three-member swap** (exp065 0.10, exp066 0.25, exp067 0.20: 55 % of the base weight;
  `v62_swap3_shift__50d0d544`, stack `v62pre_swap3_stack_4aa895__6f97365e`): 2026 uncovered rows move by 17.3 s rms;
  holdout 235.650 -> 235.403 ex_wd (-0.09 against v61), bulk 187.347, Jan 228.87, Jul 240.54; forced-long shift on top
  (day-CV 235.403 -> 235.318). Armed for the first slot of 7 Oct (00:00 UTC). Expected from v61's transfer (a third of
  the holdout gain): about -0.03 on the board.
- **Carrying the new base onto the corrected rows without a refit** (holdout check, residual member's share 0.7 of the
  covered-row blend): adding alpha x 0.7 x (new stack - old stack) on the 194,706 corrected holdout rows gives ex_wd
  235.403 (alpha 0) -> 235.360 (0.25) -> 235.329 (0.5) -> 235.306 (1.0); July improves by up to 0.23, January worsens
  by up to 0.07. Below the 0.2 s bar and split by month: not a submission step by itself. The clean test is a residual
  stage refitted on the new base: `lab/build_winter_base.py` with the new stack and the same fifteen month-pair members
  gives `base_plus_winter__7a0f84e9` (same 2,085,047 rows); the depth-10 residual on it (seed and settings of v55g, so
  the pair isolates the base) is the first job of the second overnight schedule (`lab/overnight_1007.sh`), followed by a
  second seed of exp066 (its original seeds were 1.3 s apart) and the stage's learning-curve check.

### 10.23 Documentation and housekeeping (night of 6-7 Oct, while the GPU trained)

The data-science review's zero-cost items with real stakes (prerequisite (b)), done except the two that are the
project owner's call (GPLv3 LICENSE, the archived adsb.lol statement):
- **Submission manifest** (`lab/manifest.py <run>` -> `docs/submission_manifest.md`): follows the run records from a
  submitted run to the trained configurations (v61: 60 runs, 44 trained) and writes the upload with its board score
  and the file's SHA-256, the lineage tree with the step, parameters and command at every node, and the trained runs
  with configuration file, seed, validation months, holdout score and git commit. The base weights the review asked
  to have recorded are in it, read from the blend runs.
- **ADS-B stage leakage tests** (`tests/test_adsb_stage_tables.py`, 25 s, skipped without the cache): every
  own-movement timing lies before the ADS-B takeoff (minimum 0.1 s) with the link inside the 600 s tolerance; no
  numeric input of the four stage tables reaches |r| >= 0.99 with the 2025 taxi time (largest 0.60, `park2_est`).
- **Availability table re-marked** (`lab/availability_table.py`): the ADS-B families added after the 1 Oct audit
  (pushback detector, first reception, track quality, stop classes, stand archive, time-to-location map) and the whole
  base matrix of the direct formulation are in the submitted stage; the inbound family and the map's arrival half are
  dropped by `--drop-prefix tarr_,it_`. Inputs reaching the submission: 58 (7 operational as computed, 7 after
  re-anchoring, 44 not before movement), was 51 (7 / 7 / 37); paper section 3.6 updated.
- **Paper**: section 3.2 now describes the base ensemble (members, weights, refit, stack step, training length);
  section 6 lists the submitted pipeline's commands and points to the manifest; the board-gain table runs to v61.
- **README**: a section on the submitted solution (`lab/`), and the layout tree completed (adsb, geometry,
  reanalysis, postprocess, report, notebooks 04-06).

### 10.24 Board readings of 7 Oct (first slots)

- **v62 = the three-member swap** (`v62_swap3_shift__50d0d544`, holdout 235.318 after the shift), uploaded 00:00:36
  UTC (slot 1 of 7 Oct): **board 220.5756, -0.051 against v61, the new best**; holdout -0.09 against v61, so about half
  reached the board (v61: a third). The longer-trained base members are worth -0.10 on the board in two steps.
- **v60g_s242, a fourth 360-day residual model** (seed offset 200, GPU d10 / 5,000 / 0.02,
  `v60g_resid_360d_d10_s242__92d11841`, 22:19-00:01 UTC): 235.86 / bulk 188.00 / projected 219.97, inside the seed
  spread of v55g (235.85) and v58g (235.91).
- **v63 = v62 with the residual member averaged over four models** (`v63_resid4_swap3__bb4e326b`, built by
  `lab/build_candidate.py`; residual member 235.787 -> 235.782, candidate 235.318 -> 235.313), uploaded 00:02:42 UTC
  (slot 2): **board 220.5799, +0.004 against v62: a tie.** A fourth seed of the same residual model adds nothing on
  either scale; the residual average is saturated at three models. v62 stays the best. Further stage seeds are not
  worth GPU time; the residual model on the new base (a different base, not a seed) is the next stage test.
- **v62g_newbase, the depth-10 residual model on the new 360-day base** (`v62g_resid_360d_d10_newbase__e91e8c92`, seed
  and settings of v55g, 00:01-01:44 UTC): 235.43 ex_wd / bulk 187.57 against v55g's 235.85 / 188.02. Split by rows: on
  the 194,706 corrected holdout rows 200.05 against 200.40 (v55g) and 200.43 (v60g, the seed copy): -0.35, ten times the
  seed spread; the new stack itself moves those rows 242.28 -> 241.87, so the refitted stage carries about 85 % of the
  base change (a refit beats the carry without refit, which reached about half). On the uncovered rows the run equals
  its base (274.72 against 275.21).
- **v64 = v62 with the residual member replaced by the new-base model alone** (`v64b_newbase_only__cfa82e23`; holdout
  235.167 after the shift, -0.15 against v62; Jan +0.05, Jul -0.30), uploaded 01:47 UTC (slot 3): **board 220.6812,
  +0.106 against v62.** The single model gives up the residual average, which was worth -0.14 on the board at 360 days
  (v57 single -> v59 three models) at a holdout change of -0.03; the new base's own share is then about -0.03 on the
  board. The five-model average (four old-base + the new-base model, `v64_resid5_newbase__ac2463a4`) reads 235.261.
  Consequence: the remaining GPU hours go to two more new-base residual models (`lab/overnight_1007c.sh`: a depth-10
  seed copy and a depth-8 / 6,000-round model), replacing the learning-curve check and exp068; the candidate is the
  three-model new-base average on the newest stack (watcher builds it with and without the old-base models).
- **exp066 seed 43** (`2026-10-07T03-44-11__lean2_global_window_long__e60b8b51__s43`, 01:44-04:08 UTC): 260.33 / bulk
  215.69 against the original seed 43's 262.10 / 217.88 (-1.8 / -2.2, the same gain as seed 42); seed 43 is the weaker
  seed of this configuration at both learning rates (1.0 s behind seed 42). **v65_swap3s2** = v62 with exp066 averaged
  over both seeds (0.125 each; `v65_swap3s2__b9b485f8`, stack `v65_swap3s2pre_stack_*`): stack 256.74 against v62's
  256.67 ex_wd; candidate 235.338 after the shift against 235.318 (Jan 228.96 against 228.87, Jul 240.50 against
  240.54). A tie within noise; not uploaded on its own. The new-base residual candidates are built on v62's stack (and
  one variant on this one), so the two changes stay apart.
- **On hold (project owner, 04:52 UTC 7 Oct: "keep the new base model on hold for now, so the work finishes sooner").**
  Stopped: the new-base residual seed copy (`v65g_newbase_s142`, 43 of about 100 minutes) and the depth-8 new-base model
  (`v65e_newbase_d8`, started by the chain as the first one ended), the chains `lab/overnight_1007b.sh` (exp068/exp069
  never started) and `lab/overnight_1007c.sh`, and the candidate watcher; no partial run directory was written; GPU idle.
  Also not run: the learning-curve check (`curve_d8`). What exists for the line: the new 360-day base
  (`base_plus_winter__7a0f84e9`, three retrained members in the stack), one new-base residual model
  (`v62g_resid_360d_d10_newbase__e91e8c92`), the two-seed global stack (`v65_swap3s2pre_stack_*`). To resume: the two
  `stage` commands of `lab/overnight_1007c.sh` (about 1 h 45 min each on the GPU), then
  `lab/build_candidate.py --resid <the three new-base models> --direct 2026-10-06T16-08-35__direct_360x2__58632d82
  --new-stack 2026-10-07T00-19-23__v62pre_swap3_stack_4aa895__6f97365e --name <name>`. The best board score stays v62
  (220.58); two upload slots of 7 Oct are unused.

### 10.25 The final days: decision analysis, plan and the uploads of 7 October evening

The project owner asked on 7 Oct (14:30 UTC) whether to keep training the models longer and on more data, or to test
one last approach from the literature, and wants the documentation to show which data sources are feasible for an
operational prediction made at or before pushback without ADS-B of the flight itself. A workflow of nine agents (six
analysts, three adversarial verifiers; read-only) answered from the records:
- **More data is exhausted.** The stage holds 350 of the 365 days of 2025 (adsb.lol has no release for 28 May-10 June,
  31 December is unreadable); the base is refitted on all twelve months. Each doubling of the stage's training days
  halved its board gain: -1.18 (62 -> 120 days), -0.59 (120 -> 232), -0.26 per doubling (232 -> 350).
- **Training longer has a short tail.** exp068 + exp069 (0.20 of the base weight): -0.03 to -0.05 on the board, from the
  measured -6.4 to -11.2 board MSE per 0.1 of weight (v61, v62). A three-model new-base residual average: about -0.03.
  A coherent compound of every remaining item (five long members, the stage refitted on that base, new-base direct
  models): ceiling about -0.15, realistic -0.05. More seeds are saturated (v63 +0.004).
- **No literature approach clears 0.1 s in the time left.** An MLP / TabM member needs torch (not installed) and 6-8 h
  of code; in the only taxi-out stack with per-learner figures the MLP was the weakest learner (Wu 2026, Table 4: test
  RMSE MLP 173.4 s, kNN 166.6, SVR 145.8, CatBoost 143.5, ridge stack 140.5 on 5,952 flights, random ten-fold). An
  XGBoost stage member read -0.01 at 240 days (10.20). Pseudo-labels cannot reach the rows that need help (Istanbul has
  no coverage). The largest conditional lever, shifting Amsterdam's January 2026 snow-week rows toward NM's clock (up
  to 0.4 s), is contradicted by 2025: covered EHAM rows with a 3,000-5,000 s proxy had their truth 1,119-1,227 s below
  the proxy and 42-57 s below the prediction. The open review items (Lossguide member, LTFM capacity, Barcelona cell
  recalibration, Rome reoccupation fingerprint, forward window on the winter holdout) read 0 to 0.04 s each.
- **Competitors' public code:** no team of the top ten has a public repository; the public ones (36th downwards) use a
  subset of our methods. Their measured negatives agree with ours (queueing features -0.1 s, seed averaging 0.2 s).
- **Rank:** 6th at 220.58 (7 Oct 14:32 UTC); 5th 218.77 (1.8 s ahead, the top four still improving); 7th 221.46
  (bold-ladder, -8.5 s in four days). The remaining train-longer line moves the expected final rank by about 0.1-0.2
  places.

Facts found on the way (for the documentation):
- EOBT_1 differs from LOBT on 29.9 % of matched 2025 departures and is then closer to the recorded off-block (median
  |error| 305 s against 483 s; LIRF 186 against 302 s): it behaves like a late-updated estimate, not a pre-movement plan.
- NM's AOBT_3 is back-computed from the takeoff at LTFM (54-56 % of a runway's values in one 60-s window of
  takeoff - AOBT_3) and partly at EDDM, LEBL, LSZH and LEMD.
- The actual departure runway equals the dominant departure runway of the hour before the planned off-block on 66.3 %
  of departures (EHAM 50.7 %, EGLL 91.2 %) and is in the set of runways in use on 93.1 %.
- OPDI was on the organisers' allowed list (relayed 9 Sep); we left it out for coverage, not permission (LTFM 0, LFPG 1,
  LIRF 0 parking exits over 11-19 Jan 2026), which corrects 10.11. EUROCONTROL PRU's OpenSky airport-coverage index
  (June 2026) rates LTFM 0.146, LIRF 0.287 and LFPG 0.305, independent evidence for the reception limit of 3.5.
- Paper section 4 lists "XGBoost or LightGBM members" as not helping, but no LightGBM run exists and the XGBoost stage
  member tied; lightgbm 4.7.0 is installed (CLAUDE.md says otherwise).
- A Paperguide systematic screening (121 records, 9 studies included; report of 7 Oct) finds no study that isolates the
  value of each information source under a temporal or year-ahead validation. The closest ladder is Wang et al. 2020
  (MAN / ZRH, random split, pushback to line-up): route-informed static features RMSE 3.50 / 3.27 min, + aircraft
  heading to stand or runway at pushback 3.43 / 3.16, + recent speeds and weather 3.37 / 3.10; congestion counted over
  the aircraft's own taxi (excluded there as impractical) 2.96 / 2.15. The one temporal test is Lim et al. 2024 at
  Atlanta (last 10 %, ~57,000 departures): RMSE 1.78 min, MAE 1.33 min, 78.0 % within 2 min. Wu 2026 / 2025 and Tang
  2025 count queues over the realised taxi (hindsight).

**Plan (approved 7 Oct ~16:00 UTC).** The GPU runs the scripted train-longer tail in the free window to 05:00 UTC
8 Oct (`lab/overnight_1008.sh`: two more new-base residual models, the new-base candidates, exp068, exp069, the
five-member stack and its swap); the operational ladder (tiers O, two horizons, nested data sources, no own ADS-B) is
coded in a separate worktree and trained from the next free GPU window; the weekend goes to documentation.

**Pre-registered uploads of 7 Oct (slots 4 and 5 of the UTC day, expiring-slot policy of 10.19):**
- **v65 = `v64_resid5_newbase__ac2463a4`** (residual member = mean of v55g, v55c, v58g, v60g_s242 and v62g_newbase;
  v62's stack on the uncovered rows; forced-long shift): holdout ex_wd 235.261 against v62's 235.318 (-0.057), bulk
  187.472 (-0.057), January 228.801 (-0.005), July 240.340 (-0.099). Expected board 220.55-220.60: one fifth of the
  residual average is new-base (worth about -0.03 in full) and four old-base models tied (v63). Best available
  candidate on the holdout, uploaded because the slot expires.
- **v66 = the better of `v66n_resid_newbase` (three new-base residual models) and `v66m_resid_mix` (three old + three
  new)**, chosen on the holdout ex_wd with both months not worse, if it is built before 23:55 UTC. Expected board -0.03
  +- 0.03 against v62 (220.52-220.58).
- **v65 uploaded 16:13 UTC 7 Oct** (slot 4): **board 220.5768**, +0.001 against v62 (220.5756): the tie the
  pre-registration expected (220.55-220.60). v62 stays the best.

### 10.26 The full texts (project owner's PDFs, 7 Oct evening): what they change

The project owner placed 16 PDFs in `docs/literature_pdfs/` (gitignored: publisher copyright, local reading copies
only). A workflow of five readers, a synthesis and two verifiers read 15 papers in full (one file is a duplicate of
Wu 2026); the notes with page references are in `docs/background/literature_fulltext.md`. A number check against the PDFs
found 40 of 49 claims correct, 6 wrong and 3 unclear, all corrected.
- **Idris 2002, settled:** the operational input is N, the departures on the surface at the flight's pushback (R² 0.19,
  about 1.5 min per aircraft, BOS 1998); the strong R² 0.59 belongs to Q, the takeoffs during the flight's OWN taxi
  (hindsight). The model predicts T(N) through an empirical distribution of passing; tested one month ahead (Aug 1998):
  MAE 4.56 min against 5.69 for a flight-history average, unbiased (pp. 11-18). N is exactly our H0 feed rung (R5).
- **Most strong published figures use hindsight or random splits, or both:** Wu 2026, Tang 2025, Yin 2018, He 2022 and
  Jiao 2022 count traffic over the own [off-block, takeoff]; Herrema 2018 (LFPG) uses AOBT and an ATOT-anchored count
  on a random split; Wang 2021's best group (E, RMSE 2.96 / 2.15 / 2.07 min at MAN / ZRH / HKG) is hindsight and was
  excluded by its authors, its operational group D reads 3.37 / 3.10 / 3.10 min (random 70/30, arrivals pooled,
  ending at line-up).
- **Corrections to our own records:** Swatowska 2025's 1-6 % gains are Paris-Orly and Istanbul; Zurich gained 35 %
  (Table V), which corrects 10.17 / the review; Diana 2018 is not a year-over-year test; Lordan 2016 has no hour
  variable; Herrema 2018 is volume 15(3) and its 1.6 min is runway 08L on a random split; Wang's Group A is the own
  realised route + runway used, Group B adds other aircraft taxiing at pushback (10.25 and the ladder spec described B
  loosely). Applied to `docs/background/literature_review.md`, `docs/literature_extraction.csv` and `docs/paper.md`
  (Idris N vs Q; the published whole-taxi-out RMSE range is about 1.6-3.4 min; Lim 2024 and Günther 2011 stay
  unverified, their PDFs were not in the folder).
- **Board model: no change** (no credible literature lever; the MLP was the weakest learner in Wu 2026's own stack).
- **Operational ladder, added before the GPU window (branch operational_ladder, 56371cb):** the report's literature
  rows rebuilt from the full texts (Idris 2002 as the H0 comparator through MAE/SD: Idris 0.41, our stand x runway table
  0.50); spread-normalised errors, +-240/900 s shares, bias change with a day-block bootstrap CI and per-airport rung
  steps; computed baselines B-HAT (stand x planned local hour) and B3 (the flight number's previous 14 days, the
  Idris/ETMS-style history average); a lab HINDSIGHT rung `op_h0_r5_hind` (the traffic during the flight's own taxi,
  never operational) that measures on our ten airports how much of the literature's accuracy comes from own-taxi counts;
  an MAE-loss variant `op_h1_r4_mae` for comparability with Swatowska 2025. `model.eval_max_target_sec` (hash-neutral)
  makes a plain regressor early-stop on the population its fit keeps.
- **The three new-base residual models** (`lab/overnight_1008.sh`, 16:12-19:22 UTC 7 Oct): the depth-10 seed copy
  `v65g_resid_360d_d10_newbase_s142__0f73e083` 235.47 ex_wd / bulk 187.60 and the depth-8 / 6,000-round model
  `v65e_resid_360d_d8_6000_newbase__fdc5af6e` 235.47 / 187.75, next to v62g_newbase's 235.43 / 187.57. Candidates on
  v62's stack with the forced-long shift (ex_wd / bulk / January / July): **v66n** = the three new-base models
  (`v66n_resid_newbase__77ab7e81`) 235.146 / 187.394 / 228.847 / 240.104; **v66m** = three old-base + three new-base
  models (`v66m_resid_mix__0004cd6d`) 235.198 / 187.436 / 228.817 / 240.218; v62 235.318 / 187.529 / 228.806 / 240.439.
- **v66 pre-registered (slot 5 of 7 Oct): v66n.** The better ex_wd (-0.172 against v62). Deviation from the
  pre-registered condition stated here: January is +0.041 against v62 (v66m +0.011), inside the stage's fold spread
  of 0.10 s (10.19); July -0.335. v66n keeps the three-model averaging (two depth-10 seeds and the depth-8 / 6,000
  model, the structure of v62's residual member) and changes only the base the stage was refitted on, so it isolates
  the new base that v64 confounded with the loss of averaging. Expected board 220.52-220.58 (new-base share about
  -0.03, findings 10.24).
- **v66 uploaded 19:25 UTC 7 Oct (slot 5): board 220.6483, +0.073 against v62**, above the pre-registered range
  (220.52-220.58). With v64 (+0.106) this is the second board loss of the stage refitted on the new base, now with the
  three-model averaging kept: holdout -0.15 and -0.17, board +0.11 and +0.07. **The new-base stage line is closed.**
  The stage stays on the old base; the longer-trained members enter only through the uncovered-row swap, the route that
  gained twice (v61, v62). Consequence for 8 Oct: the chain's v67n / v67m swaps are built on the new-base candidates
  and are not upload candidates; the candidate is **v67o = v62's lineage (`v60pre_resid360x3_direct360x2__1c2367be`,
  old-base residual average) + the five-member stack (exp065-exp069, exp066 seed 42 only) on the uncovered rows +
  the forced-long shift**, built after exp069 (reusing the chain's five-member stack).
- **The last two base members trained longer** (lr 0.04, cap 8,000, patience 400, refit): **exp068** (exp057, per-airport
  proxy-anchored, `2026-10-07T21-23-24__lean2_window_anchored_long__3802a29a__s42`, 2 h 13 min) 260.12 ex_wd / 214.76
  bulk against exp057 seed 42's 260.82 / 215.63 (-0.70 / -0.86); **exp069** (exp061, border 254, EHAM-only anchor,
  `2026-10-07T23-36-29__lean2_window_b254_eham_anchor_long__18921f28__s42`, 2 h 08 min) 258.93 / 213.42 against exp061's
  two seeds 260.75 / 260.61 and 214.69 / 215.23 (-1.75 / -1.54 against their mean). All five retrained members pass
  (gains 0.7-1.8 s without wrong-date rows on their own holdout).
- **v67o pre-registered for the first slot of 8 Oct** (`2026-10-06T10-47-37__v67o_swap5__37b3f05c`): v62's lineage
  (old-base residual average + direct models, `v60pre_resid360x3_direct360x2__1c2367be`) with all five longer-trained
  members swapped into the base on the uncovered rows (five-member stack `v67n_swap5pre_stack_17ebe3__4bcca99c`, exp066
  seed 42 only) and the forced-long shift. Holdout ex_wd 235.297 / bulk 187.486 / January 228.749 / July 240.444
  against v62's 235.318 / 187.529 / 228.806 / 240.439 (-0.021, July flat); 2026 rows changed: 91,354 (rms 3.4 s).
  Expected board 220.53-220.58: two more members at 0.20 of the base weight, at the measured -6 to -11 board MSE per
  0.1 of weight (v61, v62). The chain's v67n / v67m (on the closed new-base line) are not upload candidates.
- **v67 uploaded 00:00 UTC 8 Oct (slot 1 of 8 Oct): board 220.6012, +0.026 against v62**, just above the pre-registered
  range (220.53-220.58): a tie to a small loss at the board's step noise. The first three longer-trained members gained
  -0.10 in two swaps (v61, v62); the last two (exp068 for exp057 at 0.10, exp069 for exp061's two seeds at 0.10) add
  nothing on the board although both passed on their own holdout and the candidate's holdout moved -0.021. **The
  train-longer line is closed; v62 (220.5756) stays the best submission.** With the new-base line (v64, v66) also
  closed, no board candidate with a positive expectation is left; the remaining slots go to the operational model
  (pre-registered separately) and to nothing else unless a new measured lever appears.

### 10.27 The operational ladder on real data (night of 7-8 Oct)

`lab/op_ladder_run.sh "2026-10-08 05:00"` (merged branch operational_ladder, c78d537): every gate passed (120 tests;
full-frame anchor sanity; real-slice T1/T1b/T2; cache warm H1 2,429,888 x 113 and H0 x 115; smoke fit OP MAE 199 s on
a 5 % sample; MAE smoke). Rungs on the 2025 Jan+Jul holdout, OP population (target < 3,600 s, NM-matched, 338,277 rows),
one global CatBoost each, early stopping on holdout rows <= 7,200 s:

| rung (H1 = 15 min before the planned off-block) | OP MAE | OP RMSE | within +-2 min | within +-5 min | MAE/SD | best iteration |
|---|---|---|---|---|---|---|
| B1 stand x runway median table (comparison) | 207.4 | 330.2 | 45.5 % | 81.0 % | 0.50 | - |
| R0 flight record and plan | 207.3 | 310.7 | 43.0 % | 79.9 % | 0.50 | 846 |
| R1 + airport layout (OSM routes) | 206.8 | 308.9 | 42.3 % | 80.0 % | 0.50 | 523 |
| R2 + weather at the anchor (METAR/TAF) | 192.8 | 289.0 | 45.5 % | 82.1 % | 0.47 | 3,470 |
| R3 + other flights' movement records up to the anchor | 191.3 | 286.1 | 45.8 % | 82.3 % | 0.46 | 5,592 |
| R4 + NM data of departed flights, plan backlog | 188.1 | 280.4 | 46.2 % | 82.8 % | 0.46 | 4,377 |
| C1 exp066, challenge member (hindsight T/A/C inputs) | 137.1 | 203.2 | 58.7 % | 90.5 % | 0.33 | - |

First readings: static information matches the airport-style table on MAE (and beats it by 20 s RMSE); layout adds
almost nothing once the stand x runway identity is in the model; weather at the anchor is the largest operational step
(-14 s MAE, -20 s RMSE), more than in the challenge model where traffic and NM clocks already carry the day's state;
the movement records and NM data before the anchor add -1.5 and -3.2 s MAE. R4's block permutation (holdout RMSE
increase): stand x runway keys 53.2 s, aircraft identity 38.8, airport 21.3, NM data 19.5, weather 14.0, route 11.2,
traffic 8.2. Level check of R4's refit on 2026 against the PRU monthly means: year-over-year change correlation 0.71,
one flag (LIRF July +79 s). (Noted 13 Oct: the 0.71 covers all 20 airport-months of Jan+Jul 2026. On the 11 with a
PRU reference share of at least 0.9 it is 0.88. The deliverables' correlations in 10.35 and 10.36 use that share rule
on 23 of the final set's 40 airport-months, so neither figure compares directly with them.) Lab rungs (the airport's
own off-block feed, the push horizon H0, the hindsight rung) follow.
- **v68 pre-registered: the operational model op_h1_r4 on the board** (raw refit predictions, no mixture, no join window,
  no forced-long shift; `2026-10-08T02-37-29__op_h1_r4__3fcdc779__s42`). Holdout ALL 585.3, without wrong-date rows
  444.9, bulk 283.4 (v62: 294.1 / 235.3 / 187.5). It cannot know the stale and wrong-date records that dominate the
  board's squared error, so its board score is not an operational accuracy; it gives one year-ahead number for an
  operational model on the challenge's scoring. Expected board 420-500 (wide: the 2026 artefact rows decide it).
  The best submission stays v62 (the ranking takes the best upload).
- **v68 uploaded 01:18 UTC 8 Oct (slot 2): board 560.92**, above the pre-registered 420-500. The range was anchored on
  the holdout without wrong-date rows (444.9); the board instead sits near the holdout on all rows (585.3), because the
  2026 stale and wrong-date records and the rows the NM join window would bound carry most of the squared error of a model
  that neither hedges the record's meanings nor clips to the window. On the challenge's own metric an operational
  pre-movement model scores 560.9 against the challenge model's 220.6 (v62); the gap is mostly record handling (tiers
  C/T), not the taxi time itself (the OP-population gap is 188.1 against 137.1 s MAE for C1, 103.9 s for v62). v62
  stays the best submission; `submissions/submitting.parquet` points to v62 again.
- **The lab rungs (2025 only; the airport's own off-block records of other flights):** H1-R5 (+ the taxiing-out queue
  and recent realised taxi-outs at the anchor, `op_h1_r5__5600710a`) OP MAE 187.3 / RMSE 278.5 / 46.2 % / 83.0 %
  (-0.8 / -1.9 s against R4): 15 min before the planned off-block the true queue barely helps, because it is not the
  queue the flight will meet. **H0-R5 (everything at the recorded off-block, `op_h0_r5__e3225a55`) OP MAE 171.4 / RMSE
  257.4 / 50.2 % within +-2 min / 85.5 % within +-5 min, MAE/SD 0.415**: waiting until the push is worth -15.9 s MAE and
  -21.1 s RMSE, and the spread-normalised error equals Idris 2002's 0.41 with the same information set (BOS 1998,
  temporal test). H0-R5's block permutation: the off-block feed 51.1 s (the queue at the push, Idris's N), aircraft
  identity 50.0, stand x runway 42.2, the delay against plan and schedule at the push 36.7, airport 24.5, route 13.0,
  weather 11.7, movement records 9.5. The 34 s of MAE from H0-R5 (171.4) to the challenge member C1 (137.1) is the value
  of information that exists only after the push; the lab hindsight rung measures its own-taxi part.
- **The hindsight inflation, measured like for like** (`op_h0_r5_hind__bcd6403c`: H0-R5 + the traffic during the
  flight's OWN taxi [recorded off-block, own takeoff]: other takeoffs at the airport and on the own runway, landings,
  departures taxiing at the same time; lab, never operational): **OP MAE 85.1 s / RMSE 122.2 s / 77.3 % within +-2 min
  / 97.6 % within +-5 min, MAE/SD 0.206**, against H0-R5's 171.4 / 257.4 / 50.2 % / 85.5 % (0.415). The own-taxi counts
  correlate 0.60-0.70 with the bulk taxi-out because they scale with the taxi window itself. They halve the error and
  put a model on our ten airports and strict seasonal holdout into the range of the literature's headline figures
  (Yin 2018 81.4 % within 2 min; Tang 2025 94.4 % within 3 min; Wang 2021 Group E), and ahead of our own post-operational
  reconstruction with the flight's ADS-B (v62: 103.9 s). Most published accuracy therefore rests on information that
  exists only after the takeoff (findings 10.26), which no operational system has.
- **Runway-blind sensitivity of R4** (`op_h1_r4_rwyblind__8f2244ea`, every own-runway column dropped, airport-level
  runway-in-use kept): OP MAE 204.9 / RMSE 297.8 / 41.8 % (+16.8 / +17.4 s against R4): the per-flight departure runway
  is worth as much as weather. The H1 rungs assume the planned runway is known 15 min before the off-block (tier O*);
  the data holds the actual runway, which matches the hour's dominant departure runway on 66 % of departures.
- **MAE-loss r4** (`op_h1_r4_mae__3461a185`): hit its 8,000-round cap with the eval MAE still falling (OP MAE 222.2,
  under-trained); not reported. `configs/op_h1_r4_mae_fast.yaml` (lr 0.25, cap 30,000) is queued for the next window.
- **MAE-loss r4 at a larger step** (`op_h1_r4_mae_fast__5ae3b6db`, lr 0.25, cap 30,000): also ran to its cap (still
  not converged), OP MAE 190.3 / RMSE 299.8 / **48.1 % within +-2 min** / 83.3 % within +-5 min against the RMSE model's
  188.1 / 280.4 / 46.2 % / 82.8 %: the MAE loss buys +1.9 pp within the A-CDM +-2 min tolerance at an unchanged MAE.
  A sensitivity row, never on the board. W1 ended 04:31 UTC; nothing runs on the machine.

### 10.28 Dependent runways at Paris and Istanbul (project owner's question, 8 Oct)

Prompted by Herrema 2018 (CDG: "congestion level" on the dependent runways ranks second) and the owner's note that
Istanbul's paired runways cannot be operated independently. Two CPU pre-tests with skeptics (scripts in
`lab/pretests/dependent_runways/`):
- **The structure is real and confirmed in the data.** CDG: departures use the inner runway of each doublet (08L/26R,
  09R/27L; 96.8-99.1 % of their 2026 movements are departures) and every landing on the outer runway crosses it (AIP
  LFPG 20.3.2.3; our OSM routes give 1 crossing to every outer runway). LTFM: 16L/16R and 17L/17R are about 205-212 m
  apart (ICAO dependent below 760 m), 18/36 is 2.4 km away and independent; the centre pair runs segregated (35L
  departs while 35R lands), and **takeoffs on 35L occur at 0.46 times their normal rate in the 30-90 s before a landing
  on 35R** (independent combinations: 1.00). No feature of ours counts traffic on the partner runway.
- **It is not a board lever.** Day-blocked residual models on the 2025 holdout: Paris, on bulk rows the ADS-B stage
  leaves alone, the partner-runway increment over controls is -0.14 s of 220 s RMSE (paired CI95 [-0.34, +0.03]),
  projected board about -0.01 s; Istanbul (all rows uncovered), -0.19 to -0.35 % of LTFM bulk MSE beyond own-runway and
  airport-wide controls (same sign in three fold assignments, permutation-confirmed, past-only features negative in both
  cross-month tests, CI reaching zero in two of three), projected board -0.04 to -0.07 s. The airport-wide and
  runway-queue congestion counts already carry most of the effect. Not built (an LTFM specialist would cost the GPU
  window for less than the board's resolution); a paper paragraph instead.
- Side finding: 9 % of Paris's trimmed holdout squared error comes from July 2025 departures under a works-imposed
  single-runway mixed mode on the north doublet (09L/27R in mixed use), a regime absent from January and July 2026, so
  the holdout overstates Paris's difficulty slightly.
- **Full-model A/B after all (project owner, 8 Oct ~11:40 UTC: "so it is really not worth to test the approach?
  Resources are now free until 17 utc").** Block `partner_runway` (983a71c, group `partner_runway`, tier T): for a
  takeoff on a runway with a close parallel partner (LTFM 16L/16R, 17L/17R and their reciprocals; LFPG's doublets),
  landings on the partner in the last 10 / 20 min, takeoffs on it in the last 10 min, seconds since the last landing
  and the last movement on it, and all movements on the pair in the last 10 min; windows [t - w, t), every other
  runway 0 / NaN with `prw_has_partner` = 0 (`tests/test_partner_runway.py`, brute force). Arms: **A = exp070** (exp065
  on LTFM only, round cap 16,000), **B = exp071** (A + `partner_runway`, nothing else); seeds 42/43/44 holdout-only, then
  one refit per arm (`lab/ltfm_partner_ab.sh`, log `runs/ltfm_partner_ab.log`, deadline guard 16:50 UTC).
  **Pre-registered at launch (11:52 UTC, stated to the owner; written here at 12:10 UTC, before any paired read-out):**
  B passes if its LTFM bulk RMSE (target < 3,600 s) is at least 0.3 s below A's as the mean of the three paired seed
  differences, every paired difference negative (`lab/pretests/dependent_runways/ltfm_ab_readout.py`, day-block
  bootstrap per seed and on the seed-averaged predictions; the centre pair 35L/17R is where the effect should sit, the
  independent 36/18 is a placebo). If it passes, the candidate is v62 with its LTFM rows blended with exp071's refit at
  a weight chosen on the holdout and re-clipped to the join window (`lab/ltfm_partner_candidate.py`), uploaded as v69
  if its holdout improves in both months; expected board -0.01 to -0.04 (LTFM is 13 % of the rows and uncovered by
  ADS-B; arm A alone blended at 0.1-0.2 moves v62's LTFM holdout by -0.02).
- **A/B result (12:59 UTC): the bar FAILS; no candidate, no upload.** LTFM bulk RMSE (target < 3,600 s, 46,499 rows),
  A → B per seed: s42 227.87 → 227.40 (-0.47, day-block CI95 -1.81 to +0.90), s43 228.67 → 227.99 (-0.68, -1.81 to
  +0.43), **s44 227.42 → 227.50 (+0.08, -0.88 to +1.04)**; mean of the paired changes -0.36 (the -0.3 part of the bar
  holds, the every-seed-negative part does not); seed-averaged predictions 227.27 → 226.88 (-0.39, -1.22 to +0.44). By
  runway: the centre pair 35L/17R (18,054 rows), where the partner effect should sit, -0.74 / -0.77 / -0.47 (mean
  -0.66; seed-averaged -0.76, CI95 -2.68 to +1.25); the independent 36/18 (placebo, 26,774 rows) -0.22 / -0.80 / +0.34
  (mean -0.23); the west pair 34L/16R (1,215 rows, rarely run as a pair) +0.23 / +3.30 / +7.06 (mean +3.5, CI95 -1.0 to
  +7.9). Arm A's own seed spread on all LTFM rows is 1.1 s (241.4-242.5), the size of the effect. Reading: the centre
  pair moves the right way in every seed, the west pair the wrong way, and the total is inside the seed noise. A blend
  preview on seed 42 (v62's LTFM rows with B at weight 0.3: LTFM ex_wd 240.23 → 239.76, whole holdout 235.318 →
  235.253, January -0.11, July -0.03; A at the same weight +0.06) looked like a candidate, but building one after the
  bar failed would select on one seed's noise; not pursued. The two refits were stopped (an aborted run directory
  `2026-10-08T14-57-34__ltfm_spec_long_partner__6e4c6c09__s42` holds no outputs). The sham control (exp072:
  `partner_runway_sham`, the partner columns permuted among the paired-runway departures, 5f54e83) runs on the same
  three seeds to tell structure from partner traffic on the centre pair.
- **Sham control, seeds 42-43, and a post-hoc observation (13:20 UTC).** Standalone, the sham (C) does at least as well
  as the real block: LTFM bulk C - A -0.31 / -1.43 (mean -0.87), B - C -0.16 / +0.76. Different feature sets do not pair
  on a seed number (GPU, different trees), so the standalone arms are unpaired draws with a seed spread of about 1.1 s:
  **standalone, the partner traffic adds nothing measurable.** Read as a blend member of v62 instead (LTFM rows,
  weight 0.3, re-clipped; change of v62's LTFM holdout ex_wd 240.233): B -0.47 / -0.28 / -0.45, C -0.14 / -0.16, A +0.06
  / +0.24 / -0.05. Every B seed is ahead of every C and A seed. That metric was chosen after the bar failed, so it
  cannot justify an upload by itself.
- **Confirmation pre-registered (13:25 UTC, before any of its runs exist):** B (exp071) and C (exp072) on the fresh
  seeds 45/46/47, holdout only (`lab/ltfm_partner_confirm.sh`), then exp071 seed 42 with the refit. Metric: the change
  of v62's LTFM holdout ex_wd when the specialist is blended in at the fixed weight 0.3 and re-clipped
  (`lab/ltfm_partner_candidate.py`). **Pass: every new B seed negative, B's mean change at most -0.25 s, and B's mean
  at least 0.15 s below C's.** If it passes, the candidate is v62 with its LTFM rows blended with the exp071 seed-42
  refit at 0.3 and re-clipped, uploaded as v69 if the refit run's fold OOF also improves the whole holdout in both
  months; expected board -0.02 to -0.06 (holdout about -0.06 on the whole ex_wd at the seed-42 preview). If it fails,
  the line is closed and nothing is uploaded.
- **Confirmation result (14:42 UTC): PASS by the pre-registered rule, narrowly.** Change of v62's LTFM holdout ex_wd
  (240.233) with the run blended in at 0.3 (`lab/pretests/dependent_runways/ltfm_blend_confirm.py`):

  | seed | A (exp070) | B (exp071, partner) | C (exp072, sham) |
  |---|---|---|---|
  | 42 | +0.061 | -0.474 | -0.136 |
  | 43 | +0.244 | -0.276 | -0.159 |
  | 44 | -0.054 | -0.447 | -0.200 |
  | 45 | | -0.280 | -0.231 |
  | 46 | | -0.218 | -0.285 |
  | 47 | | -0.533 | +0.057 |

  Fresh seeds 45-47: B -0.344 (all negative), C -0.153, B - C -0.191 (bar -0.15): pass, carried by C's seed 47 (+0.057,
  outside its other five seeds' -0.14 to -0.29). Over all six seeds B -0.371, C -0.159, difference -0.21 (about three
  standard errors of the seed spread): the exploratory and the fresh halves agree on about -0.2. **Standalone the real
  block is no better than the sham** (seed-averaged LTFM bulk over six seeds: C 226.42, B 226.72; A 227.27 over three).
  Reading: the partner traffic carries something the submission's ensemble lacks (about -0.2 s on LTFM when blended),
  but a model built on it is not better on its own than one with the columns permuted; the block's standalone gain
  over A is structural. Next, as pre-registered: the candidate from the exp071 seed-42 refit (started 14:42).
- **v69 pre-registered (15:05 UTC 8 Oct, slot 3 of the UTC day)** = `2026-10-06T10-47-37__v69_ltfm_partner__6e1bc8f1`: v62 with
  its 47,581 LTFM ranking rows blended with the exp071 seed-42 refit (`2026-10-08T16-42-42__ltfm_spec_long_partner__6e4c6c09__s42`)
  at the fixed weight 0.3 and re-clipped to the join window; every other row is v62's (LTFM change mean -3.0 s, rms
  14.5 s). The refit run's fold OOF gives: holdout ex_wd 235.318 → 235.260 (-0.058), January 228.806 → 228.692,
  July 240.439 → 240.422, LTFM 240.233 → 239.809. Both months improve, as the rule requires. **Expected board 220.52-220.56**
  (-0.02 to -0.06 against v62's 220.5756: uncovered-row base changes reached the board at a third to a half of their
  holdout gain in v61/v62, at slope 1.00 in v58). If v69 is not below 220.5756, v62 stays the best and
  `submissions/submitting.parquet` goes back to v62.
- **v69 uploaded 15:04 UTC 8 Oct (slot 3): board 220.4541, -0.1215 against v62 (220.5756): new best**, below the
  pre-registered range (220.52-220.56). The board gain is about twice the holdout gain (-0.058 on the whole ex_wd), the
  same direction as earlier changes on the uncovered rows of Istanbul and Paris, which the Jan+Jul holdout under-reads.
  The partner-runway information on Istanbul's close pairs, which a standalone model could not resolve from a sham,
  helps as a complementary member of the ensemble. `submissions/submitting.parquet` is v69. `docs/submission_manifest.md`
  regenerated for v69 (`lab/manifest.py` knows the new step). 8 Oct slots: 3 used, 2 left.

### 10.29 Do the NM clocks carry the weather? (project owner's question, 8 Oct)

The owner asked whether "the NM data already reflects the weather" (paper 3.7, findings 10.27) is true, and whether
the best model uses actual weather data. Workflow of four investigations and three independent checkers (code, data,
logic), 15:40-16:40 UTC; scripts in `lab/pretests/weather_nm/`, outputs in `cache/lab/weather_nm/`.
- **v69 uses actual weather data.** All 46 trained runs of its lineage read the METAR in force at takeoff (`wx_now`,
  15 columns decoded from IEM's archived METAR/SPECI text: wind, gust, visibility, RVR, ceiling, temperature,
  precipitation intensity, snow/freezing, thunderstorm, fog; derived from them head/crosswind, dew-point spread, the
  LVP and de-icing judgements) and the Open-Meteo reanalysis documented as ERA5/ERA5-Land (`era5_surface`, 9: snow
  depth, snowfall 3/24 h, precipitation, 2 m / soil / 12 h minimum temperature, frost and freezing-precipitation
  hours; the request pins no model). 45 add `deicing_context` (5 of its 6 columns combine weather with the aircraft
  or stand; `cold_soak` itself carries none), the 12 older blend members also `wx_past` and `wx_deice_load`. No TAF,
  TREND, after-takeoff weather or PRU. ADS-B stage: the direct models carry the same columns, the residual models none
  (weather only through their base). Weather is 2.2-6.9 % of the regressors' gain (median 5 %), the NM blocks 38-72 %.
- **The statement was an interpretation, never a measurement.** Every challenge-model weather test had the NM clocks;
  the ladder's R1 → R2 differs in anchor, features and population. "Within noise" holds on the Jan+Jul tests (which
  could not resolve about 1 s on the bulk; ERA5's -0.88 s on exp036 is inside that line's ±1 s, 9.10), not on the
  Feb+Dec holdout (exp028 + wx_deice_load -8.4 ex_wd / -51 disrupted; exp040 + deicing_context -2.3 / -20).
- **Measured now (2 x 2 on the cached exp051 lean2 matrix, LightGBM stand-in, three seeds, NM-matched rows below
  3,600 s; `nm_weather_2x2.py`, `analyse_2x2.py`):** the weather columns are worth 14.0 s RMSE without the 55
  NM-derived columns and 1.0 s with them on Feb+Dec 2025 (7.7 vs 0.6 s on Jan+Jul); de-icing hours 66 vs 5 s (34 vs
  5); 2025 snow days 49 vs 1 s. About 92 % of weather's value is redundant with NM on all rows, 80-86 % on disrupted
  days and in de-icing hours, about 63 % at EHAM and EDDF in de-icing hours. The flight's own NM times and the other
  flights' live NM readings each absorb most of it (decomposition, `analyse_decomp.py`; AOBT_3 alone not isolated).
  **The traffic counts absorb only 10-16 %**: the "traffic" half of the sentence is wrong. Untested: the tail
  (targets ≥ 3,600 s, 37 times as frequent under snow as in calm weather), NM-unmatched rows, the CatBoost mixture
  itself, January 2026's heavier snow.
- **NM's own reading does not equal the weather uplift** (`weather_nm.py`, 2025, within airport x month x local
  hour band): takeoff - AOBT_3 rises 55 s where the target rises 108 s in adverse weather (ratio 0.50), 159 vs 235 s
  under de-icing (0.68), 322 vs 520 s in snow (0.62), 0.55 under LVP, 0.32 in thunderstorms; by airport (de-icing)
  EHAM 1.11, EGLL 1.48 (de-icing on stand after NM's off-block), LFPG 0.87, LTFM 0.72, LSZH 0.68, EDDF 0.61, EDDM 0.25
  (AOBT_3 re-stamped at the pad). The model learns these relations per airport, which is why weather adds little on top.
  Residual check (`weather_nm_residuals.py`, Jan+Jul OOF): exp034 (NM, no weather) leaves +12 s of adverse-weather bias
  where the operational r1 leaves +81 s, and its residual is not learnable from weather (0.18 % of the MSE against 6.25 %).
- Paper 3.7's sentence corrected accordingly; the 10.27 phrase "traffic and NM clocks already carry the day's state"
  is superseded by this entry.

### 10.30 The best configuration at depth 9 (project owner's request, night of 8-9 Oct)

Owner's request (16:50 UTC): retrain the best configuration with an increased depth tonight; chosen: the three
longer-trained base members swapped into v69's base and the Istanbul specialist, depth 7 -> 9, machine free 19:00 UTC to
Fri 10:00 UTC. Configs exp073 (`lean2_window_deep` = exp065), exp074 (`lean2_global_window_deep` = exp066), exp075
(`lean2_window_lobt_anchor_deep` = exp067), exp076 (`ltfm_spec_deep_partner` = exp071); each differs from its parent
only in the regressor's depth (the mixture classifier keeps its defaults), same folds (split hashes d406ebc0 / 69e1ac58),
refit on all twelve months. Earlier depth evidence: exp053 (exp050 at depth 9) was a little weaker alone (265.77 /
215.49) and moved nothing in a blend (v30 = v29 on the board); a depth-9 Istanbul specialist (exp064, 3 Oct) was 1.9 s
better on LTFM than its depth-7 baseline and worth -0.35 on LTFM when blended (findings 10.13).
- Chain `lab/overnight_1009.sh` (waits until 19:00 UTC; exp076, exp074, exp075, exp073; a watchdog stops it at 10:00
  UTC; log `runs/overnight_1009.log`), then `lab/deep_candidates.py` on the CPU.
- **Candidates, pre-registered before any deep run exists (17:05 UTC):** v69's recipe (v60pre → base swap on the rows
  no ADS-B stage corrects → forced-long shift → LTFM specialist at 0.3) with base variants B0 (v62's base), R (the deep
  members replace the long ones) and A (each configuration's weight split between its long and deep run), and
  specialists S7 (exp071), S9 (exp076) and S79 (their mean): 8 candidates besides the control B0+S7, which reproduces v69
  exactly (checked 17:00 UTC: 2026 max |diff| 0.000000 s, holdout 235.2597). **Choice:** the lowest holdout RMSE without
  wrong-date rows among the candidates whose January and July are no worse than v69's; upload as v70 on 9 Oct only if it
  is at least 0.02 s better than v69 (235.260), with the expected board range written here before the upload.
- Standalone, each deep run is read against its depth-7 parent (single seeds; base-member seed spread 0.6-1.3 s on the
  holdout without wrong-date rows, so a difference below about 1 s is not a finding).
- **Night results (chain 19:00-02:44 UTC, no failures).** Each depth-9 run against its depth-7 parent (holdout without
  wrong-date rows / bulk / January / July; single seeds, base-member seed spread 0.6-1.3 s):
  exp074 (global) 259.62 / 214.23 / 252.48 / 265.24 against exp066's two seeds 259.29-260.33 / 214.79-215.69;
  exp075 (LOBT anchor) 260.12 / 215.38 / 252.49 / 266.12 against exp067 259.93 / 215.32 / 252.79 / 265.55;
  exp073 (per-airport window) 260.30 / 213.18 / 253.14 / 265.92 against exp065 259.63 / 213.79 / 253.07 / 264.79;
  exp076 (Istanbul specialist) LTFM 241.34 / bulk 226.63 against exp071's 241.23 / 227.48. **Depth 9 alone is a tie**
  (bulk -0.6 s in two of three base members, never better without wrong-date rows); the deeper trees stop earlier
  (exp074 13,062 rounds against 16,503; exp075's Istanbul model 6,357 against the 8,000 cap).
- **Candidates** (`lab/deep_candidates.py`, `runs/deep_candidates.out`; change against v69 on the holdout without
  wrong-date rows / January / July / bulk / LTFM): B0+S9 -0.012 / -0.017 / -0.009 / -0.061 / -0.089; B0+S79 -0.014 /
  -0.015 / -0.014 / -0.041 / -0.104; R+S7 -0.059 / -0.166 / +0.022; R+S9 -0.064 / -0.177 / +0.022; R+S79 -0.070 / -0.178
  / +0.013; A+S7 -0.051 / -0.107 / -0.008; A+S9 -0.060 / -0.121 / -0.012; **A+S79 -0.064 / -0.121 / -0.020 / -0.156 /
  -0.224**. The R variants (deep members alone) make July worse and are excluded by the rule; the pre-registered choice
  is **A+S79** (`2026-10-06T10-47-37__dc_A_S79_3289fc__3289fcc3`): each base configuration as the mean of its depth-7
  and depth-9 run, the Istanbul specialist as the mean of exp071 and exp076; gain 0.064 ≥ 0.02: passes. 2026 rows
  changed 102,513 (the uncovered rows and Istanbul's), rms 5.7 s, median |change| 3.4 s, two rows above 300 s (hedged
  rows moved to the lower window bound, as v62 moved the same Paris row).
- **v70 pre-registered (02:45 UTC 9 Oct, slot 1 of the UTC day) = A+S79.** Expected board **220.40-220.46** (-0.00 to
  -0.05 against v69's 220.4541): the base part (-0.051 on the holdout) reached the board at a third to a half in v61/v62
  and not at all in v67; the specialist part (-0.013) at about twice its holdout gain in v69.
- **v70 uploaded 02:45 UTC 9 Oct (slot 1): board 220.3396, -0.1145 against v69 (220.4541): new best**, below the
  pre-registered range (220.40-220.46), about 1.8 times the holdout gain (-0.064). Depth 9 tied depth 7 alone; averaged
  with the depth-7 members it adds diversity that the Jan+Jul holdout under-reads, like the earlier changes on the rows no
  stage corrects. `submissions/submitting.parquet` is v70; `docs/submission_manifest.md` regenerated for v70 (68 runs,
  50 trained). 9 Oct slots: 1 used, 4 left.
- **v71 pre-registered (05:43 UTC 9 Oct, slot 2; project owner's request "submit only the depth 9 model") = R+S9**
  (`2026-10-06T10-47-37__dc_R_S9_9f5a0a__9f5a0a5a`): the depth-9 members replace the depth-7 ones in the base swap and
  the depth-9 Istanbul specialist exp076 replaces exp071. The pre-registered rule had excluded it (July +0.022 against
  v69); it is uploaded as the owner's test of replacing against averaging. Holdout without wrong-date rows 235.196
  (v70 235.196), January 228.515 (v70 228.572), July 240.444 (v70 240.402), bulk 187.235 (v70 187.295), LTFM 239.578
  (v70 239.585); 2026 change against v70 rms 4.7 s on 102,513 rows; no NaN, nothing outside the join window.
  **Expected board 220.33-220.42**, most likely at or slightly above v70 (220.3396): the same holdout gain, and on the
  board averaging has beaten single replacements before (v64 against v63/v65). v70 stays the best unless v71 is lower.
- **v71 uploaded 05:42 UTC 9 Oct (slot 2): board 220.3505, +0.011 against v70 (220.3396)**, inside the pre-registered
  range (220.33-220.42) and where it was expected, at or slightly above v70. Replacing the depth-7 members by their depth-9
  runs gives the same holdout as averaging them (235.196 both), and on the board the average is 0.011 better: the third
  time averaging has beaten a single replacement (v63-v65, v70-v71). v70 stays the best; `submissions/submitting.parquet`
  is v70 again. 9 Oct slots: 2 used, 3 left.

### 10.31 A pre-movement model for the final dataset (project owner's direction, 9 Oct)

Challenge update relayed by the owner (9 Oct): deadline extended to 14 Oct, and a final submission is scored on unseen
data (Feb-Jun 2026). Owner's direction: use only data available before the taxi movement (schedule and plan, weather,
ADS-B before the movement), focus on applicability beyond the challenge; for the final submission the pre-movement model
plus a record adapter that uses only the planned delay, if it closes enough of the board gap. Downloads started 06:30
UTC: adsb.lol prod 1 Feb-30 Jun 2026 (`runs/fetch_febjun2026.sh`), raw METAR/TAF Mar-May 2026 (done). Design
workflow (four readers, a spec, a critic): `cache/lab`-free; the critic's main points: an H1 ADS-B gain screened at
0-0.25 s per airport without training (the airport's own off-block feed at H1 was worth -0.8 s); most departures are
first received when they move; satellite airfields (Le Bourget, Egelsbach, Northolt, Dübendorf, Torrejón) inside the
boxes; reception regimes in Feb-Jun 2026 without a 2025 counterpart (CDG newly covered, LIRF lost from May, EDDF about a
quarter); final-data ingestion is on the critical path of any final submission.
- **The record adapter cannot close the gap (upper bound, 09:00 UTC).** On the Jan+Jul 2025 holdout, op_h1_r4 reads ALL
  585.3 / without wrong-date rows 444.9 (C1 314.0 / 259.3). Knowing exactly which rows are stale (2,936, 0.85 %) and
  pricing them pre-movement at planned delay (LOBT - schedule) + R4's taxi moves it only to 582.8 / 441.7; even perfect
  stale rows give 448.4 / 314.8. On stale rows the actual delay exceeds planned delay + taxi by a median 679 s (IQR
  307-1,120 s): how late a flight really leaves is not known before it moves. R4's squared error: wrong-date rows 42.2 %
  (7 rows), stale rows 28.8 %, other targets above an hour 7.5 %, the bulk 21.5 %. A pre-movement model therefore stays
  near v68's 560.9 on the challenge metric; the adapter is not worth building.
- **Generalisation of the pre-movement model (night 9-10 Oct, `lab/overnight_1010.sh`, holdout-only, OP population =
  NM-matched rows below 3,600 s).** Seed floor on Jan+Jul: op_h1_r4 s42 / s43 OP MAE 188.14 / 187.89 (RMSE 280.41 /
  280.32). Calendar-free op_h1_r4_nodoy (op_h1_day_of_year dropped): 187.52 / 279.61. Month pairs, each fitted on the
  other ten months of 2025 (OP MAE in s per validation month; R4 = 15 min before the planned off-block with other
  flights' records and NM data; R2 = public data only: plan, layout, weather):

  | month | R2 | R4 | R4 calendar-free |
  |---|---|---|---|
  | Feb | 188.7 | 180.4 | 179.7 |
  | Mar | 173.8 | 165.7 | 165.0 |
  | Apr | 165.0 | 159.8 | 159.6 |
  | May | 169.2 | 164.7 | 164.5 |
  | Jun | 173.2 | 167.3 | 168.2 |
  | Aug | 180.7 | 174.9 | 174.4 |
  | Sep | 174.8 | 170.9 | 170.7 |
  | Oct | 174.9 | 167.6 | 167.9 |
  | Nov | 181.8 | 174.1 | 173.5 |
  | Dec | 187.7 | 179.8 | 180.2 |

  Within +-2 min (R4): Feb 48.1, Mar 49.7, Apr 50.5, May 49.4, Jun 49.1 %. Every Feb-Jun analogue is at least as good
  as Jan+Jul (188 s); the public-data model loses 3.9-8.3 s (largest in winter); dropping day-of-year changes the MAE
  by -0.7 to +0.9 s (mean -0.15), within the seed floor, so the calendar-free variant is the safer choice for 2026
  (Easter 20 Apr 2025 against 5 Apr 2026). Caveat: the pairs interpolate inside 2025 and early-stop on the holdout;
  the year ahead (2026) will read somewhat worse.
- **Long horizons and the ADS-B archive (night 9-10 Oct, `lab/overnight_1010b.sh`, Jan+Jul 2025 holdout, OP
  population).** Anchor: the initial off-block time IOBT, else the schedule, minus 3 h or 24 h (never LOBT, which is
  refiled until the push); every input is cut at the anchor. Long-lead rungs: r0 = record and plan (lead-free), r2 =
  + layout and the TAF in force at the anchor for the planned time, r3 = + scheduled demand around the planned time,
  r4 = + recent NM history (other flights' records up to the anchor). H1 rungs for comparison: R1 = + layout, R2 = +
  the METAR in force, R3 = + other flights' movement records, R4 = + NM readings and plan backlog. "+ archive" = the
  ADS-B archive block (stand, pier and runway statistics of past tracks, built from fit days only). OP MAE in s, RMSE
  in brackets:

  | rung | H1 (planned off-block - 15 min) | 3 h ahead | 24 h ahead |
  |---|---|---|---|
  | record and plan | 207.3 (310.7) | 206.6 (310.0) | 206.6 (310.0) |
  | + layout | 206.8 (308.9) | | |
  | + weather (H1 observed, 3 h / 24 h forecast) | 192.8 (289.0) | 196.7 (292.4) | 202.8 (302.6) |
  | + demand (H1 other flights' records, long leads the schedule) | 191.3 (286.1) | 193.9 (289.6) | 199.7 (300.2) |
  | + NM data (H1 readings and backlog, long leads history) | 188.1 (280.4) | 194.9 (290.1) | 200.9 (299.6) |
  | + archive | 187.9 (280.1) | 195.0 (289.8) | 200.2 (298.6) |
  | runway-blind | 204.9 (297.8) | 215.6 (312.7) | 222.1 (324.3) |
  | runway-blind + archive | | 214.3 (311.2) | 221.5 (322.3) |
  | runway- and stand-blind (pier only) + archive | | | 223.7 (324.2) |

  (H1 R4 runway-blind is op_h1_r4_rwyblind of 8 Oct.) Readings:
  - Weather is the largest input after the plan. The METAR in force takes 14.0 s at H1 (layout 0.5 s); the TAF
    recovers about two thirds of that 3 h ahead (about 9.4 s after the layout) and a quarter 24 h ahead (3.3 s).
  - Scheduled demand takes 2.8 s at 3 h and 3.1 s at 24 h.
  - Recent NM history does not help at the long leads: MAE +1.1 s at both, RMSE +0.5 s at 3 h and -0.7 s at 24 h. At
    H1, where the records are minutes old, the NM data take 3.2 s.
  - The departure runway is worth 16.8 s at H1, 19.3-20.7 s at 3 h and 21.3 s at 24 h. At H1 the other flights'
    records show the configuration in use; a day ahead nothing does. The deliverables are runway-blind, because few
    airports plan a flight's runway hours ahead; their runway-known twins bound what a runway plan would add.
  - The archive on the Jan+Jul holdout: MAE -0.2 s at H1 (within the seed floor), 0.0 at 3 h and -0.7 s at 24 h with
    the runway known; runway-blind -1.3 s at 3 h and -0.7 s at 24 h (RMSE -1.5 and -1.9 s); at the start of the
    movement (H0, lab) +0.1 s (171.42 against 171.36). **The forward check reverses it**: fitted on Jan-Oct 2025 and
    validated on Nov+Dec, so that every archive record lies before its query, the 24 h runway-blind model reads
    216.4 s (RMSE 307.3) with the archive and 213.3 s (304.0) without (Nov 213.7 / 211.3, Dec 219.1 / 215.3). The holdout gain
    came from records of later months (January's queries read an archive of February-December); forward, the
    month-out tables mislead.
  - Deliverables, refitted for the final dataset (which lies forward in time): `op_hl3_r4_rwyblind` (3 h) and
    `op_hl24_r4_rwyblind` (24 h), runway-blind, without the archive (the refit flag moved from the `_arch` twins).
    The archive is kept as a documented negative result. Open: whether r3 (without the NM history) is the better
    deliverable; on the seasonal holdout the history costs about 1 s of MAE at both leads, but forward it may track a
    level shift that the static inputs cannot.
- **The ADS-B surface snapshot s1 and its gates (10 Oct, morning, CPU).** `prc adsb-surface --snapshot s1 --workers 6`
  over every downloaded day (2025 and Jan-Jul 2026, 561 days; the Feb-Jun 2026 crawl ended 9 Oct with 149 of 150
  days, 2026-05-06 has no release): 9,255,993 events in 22 min, then `--check`, `--support`, `--freeze` (events sha1
  27fa580698cb, support e9b70de19065). Reception in the final months (taxi-out seen share; ADS-B takeoffs against
  2025): EDDM, EHAM, LEBL and LSZH complete; EGLL far better than in 2025 (0.73-0.99 against 0.2-0.6); EDDF thin in
  Feb-Mar (3-6 k takeoffs a month against 14-17 k), recovered by May; LEMD lower in Apr-May; LIRF lost from late April
  (no ADS-B day in May); LFPG poor as in 2025; LTFM none. The support table (2025 training departures) NaNs a family
  where its gate held on fewer than 20 % of them: every family at EGLL (taxi gate on 6.6 % in 2025), LFPG and LTFM,
  the stand family at LIRF; so R6 reads no ADS-B at Heathrow even where 2026's reception is complete (2026 regimes
  are not extrapolated, by design). Gates: the three ADS-B test files (83 passed, 2 skipped), the real slice (passed)
  and `lab/op_adsb_audit.py`: own-airframe contamination of the model's columns on the linked exposed OP rows 0.054 %
  at H1 (gate 0.2 %; worst LSZH 0.097 %, gate 1 %) and 0.227 % at H0 (gate 1 %; worst EHAM 0.559 %, gate 3 %).

### 10.32 The final Feb-Jun set (the release turned out to be Jan, Feb, Jun, Jul 2026: 10.33): the submitted recipe on new months (10 Oct)

The final submission is scored on Feb-Jun 2026 (owner's decision: both models if the rules allow two files, else the
challenge model). Until 10 Oct only the base of the challenge model could be produced on new months:
`lab/final_inputs.py runbook` predicts the base members with their saved refit models (`prc predict`), blends and
stacks them, the v32 level (260.72 on Jan+Jul). What v70 adds on top (220.34: the ADS-B stage, about 32 s; the base
swap, the forced-long shift and the Istanbul specialist, about 8 s) existed only as lab runs whose stage models were
never saved.
- **The stage, saved** (`lab/adsb_apply.py`). `--no-cv` fits only the final models on every covered 2025 day (one
  fit per seed instead of the eleven of the cross-fit and its regime twin: tens of minutes on the GPU instead of 4-7
  h); `--save-models` writes them into the run (`stage_models/`: one CatBoost file per seed, the column order, the
  categoricals, the frame flags, sha1s); `--load-models RUN` applies them to another base run's ranking rows, nothing
  fitted, refusing other frame flags, a changed model file or a missing column; `--oof-from RUN` carries the
  cross-fitted member's holdout output, so every step after the stage that estimates on the holdout (the shift)
  estimates exactly what v70's did. Smoke test (depth 4, 50 rounds, CPU, on v55g's combined base): the loaded models
  reproduce the saved run's 344,841 ranking predictions exactly; the corrected rows and the carried OOF equal v55g's.
  The five v70 members are refitted and saved in the night of 10-11 Oct (`lab/overnight_1011.sh`, `v70s_*`).
- **The recipe on predict runs** (`lab/final_challenge.py`): the swap members (exp065-067, exp073-075) and the two
  Istanbul specialists predicted on the new set; `build_winter_base.py <new stack>` (the 2025 OOF of v70's fifteen
  month-pair runs, the new stack's ranking rows); the lab matrix of the tree; the five members from their saved
  models; the two blends and `blend_direct.py --w 0.3`; `base_swap.py --old-blend <runbook blend> --old-stack
  <runbook stack>` (new: the old pair as arguments, and a predict run counts as its source's configuration); the
  shift; S79 at 0.3.
- **The lab matrix on another tree** (`build_lab_matrix.py -o data.version=V --rescale-from <lean2 run> --out
  lean2__V`, `adsb_apply.py --lab-matrix`): the direct members read the whole lean2 matrix, whose stand/runway and
  carrier/type counts grow by about a fifth in a frame with five ranking months (the release's four months: about
  13 %, 10.33); the matrix is built with `prc
  predict`'s own count rescale, under its own name, so the delivered set's matrix stays.
- **New ADS-B days against frozen pools** (`lab/adsb_new_days.sh`): the per-day steps of `rebuild_adsb_extra.sh` for
  every day that has points and no table (links, sets 1-2, pushback, first reception, track quality, stop classes, the
  stand archive's and the map's per-day estimates), then the assembled tables, without rebuilding the stand archive
  or the time-to-location map: every existing day keeps the values the stage was trained on, and the new days are
  read against the archives of the 412 training days (v70 had its own test months in the pools; of the final set,
  January and July are in them as they were for v70, February and June are not: 10.33). First reception's stand
  statistics are pooled at assembly, over the tree's days. It runs in an isolated
  checkout (`lab/final_worktree.py`: a git worktree whose runs/, cache/features, .venv and the ADS-B snapshot and
  archive are junctions to the main checkout's, and whose cache/adsb, cache/lab/ideas and the idea harness's holdout
  frame are private copies, because the assembled tables are rewritten over the tree's days; the tree's ADS-B points
  are hard links for exactly 2025 and the ranking months, since a day with points but no movements would get empty
  tables that every later start skips as done; `unlink` before `git worktree remove`).
- **Rehearsals on the delivered Jan+Jul files** (tree `data_rehearse`, version `rehearse_211b61a7`):
  1. The runbook (the 27 trained members of blend11_window_lobt): every predict run reproduces its source exactly (max
     |diff| 0 s on 344,841 rows; parity on the training rows: every block identical; count rescale w = 1 at every
     airport); the blend and the stack equal blend11's to 7e-12 s (47 min, CPU, 4 threads). It found two bugs, both
     fixed: `fetch-weather --check` crashed in the TAF coverage report under pandas 3 (a GroupBy `.ne`), and a
     restarted runbook refused its own tree because `init` compared the weather manifest and the normalised geometry
     tables, which the runbook's step 4 rewrites (now kept on a restart).
  2. The recipe (`final_challenge.py --rehearse`: v70's own stage candidate, everything after it from predict runs):
     the base swap, the shift, S79 and the final run equal v70's chain to 0.000000 s on every ranking and holdout
     row (12 min).
  3. The new-days chain: the 744 per-day tables of the 62 Jan+Jul 2026 days removed from the isolated copy
     (`final_worktree.py strip`), rebuilt by `adsb_new_days.sh` against the rehearsal tree, compared with the main
     checkout's (`compare`): 692 identical; the other 52 are the map's arrival visits of those days, which in the main
     checkout predate the rewrite of the points files on 2 Oct (the staging merge) and feed only the frozen arrival
     map; the stage's own inputs, features3 (242,926 rows) and features4 (242,980 rows) of those days, are identical.
     About 50 min of CPU for 62 days. Found: first reception's assembly reads lab/idea_harness.py's holdout frame,
     now copied into the isolated checkout.
  4. The stage from the saved models (`final_challenge.py --stage-models ... --expect-saved`, run on 11 Oct in the
     main checkout, whose Jan+Jul tables are the delivered set's): each of the five members reproduces its saved run
     to 0.000000 s, and the stage candidate's holdout equals v70's (findings 10.34).
  5. The pre-movement model: `prc predict` of op_h1_r4's refit (the v68 file) on the rehearsal tree reproduces its
     ranking predictions exactly (max |diff| 0 s; parity: 11 blocks identical), so a pre-movement file for the final
     set is one predict run (plus R6's or the calendar-free refit's, if chosen after the night).
- **After the release** (CPU only; the GPU work is done beforehand): the runbook in the main checkout (about 50 min:
  init, validate, prepare, the inputs, 27 predict runs, the blends, the stack); `final_worktree.py points --tree
  <final tree> --months 2026-01 2026-02 2026-06 2026-07`, `strip` of the same four months (the delivered Jan+Jul
  tables were linked to the unjittered times, 10.33) and `adsb_new_days.sh` in the isolated checkout (120 days, about
  2 h); `final_challenge.py --stage-models ...` there (8 predict runs, the lab matrix, the five stage
  members, the post-processing: under an hour); `prc submit` with the final template. About 4 h from the release to
  the file, all of it rehearsed on the delivered Jan+Jul files but the stage from the saved models (rehearsal 4,
  pending until the members are saved).

### 10.33 The final ranking file (released 8 Oct, examined 10 Oct)

The organisers' final files (`final_ranking.parquet`, `final_submitting.parquet`; the zip also repeats the delivered
`ranking.parquet` and `submitting.parquet`, byte-identical to ours) are kept in `data_release/2026-10-09/`
(gitignored; `data/competition_files/` is untouched, since the rehearsals and v70 depend on it).
- **Months**: January, February, June and July 2026 (1,341,197 movements; Jan 305,590, Feb 288,432, Jun 363,231, Jul
  383,944), not February to June. The template has 670,790 departures: the 344,841 of Jan+Jul and 325,949 new ones.
  Only one file is scored (owner, 10 Oct): the challenge model's.
- **Same schema**, column for column and type for type; the Jan+Jul rows carry the delivered `MVT_ID_mvt` values.
- **The times are jittered.** On the Jan+Jul rows, `MVT_TIME_UTC_mvt` differs from the delivered value on 92 % of the
  movements by an integer number of seconds in [-12, +12], triangular (55,795 at 0, about 900 at each end; mean 0, sd
  4.9): the sum of two uniform draws. On arrivals the block time moves the other way (correlation -0.99 with the
  movement's shift), and `TAXITIME_SEC_mvt` is recomputed from the shifted pair (taxi-in changes of up to 24 s). Every
  other column, the NM times (LOBT, IOBT, EOBT_1, AOBT_3) included, is identical. A departure's hidden target is
  presumably recomputed the same way, which adds a noise of about 7-10 s sd to it: about 0.2 s on an RMSE of 220.
- **The join window**: with a block time shifted by at most 12 s, a target can lie up to 12 s outside the window
  computed from the shifted takeoff and LOBT. In 2025 only 565 of 2,062,577 matched departures (0.03 %) have their
  target within 12 s of the window's edge, so the clip's guarantee loses at most 12 s on a few hundred rows (well
  below 0.01 s of RMSE); the clip and every trained step stay as they are.

### 10.34 The final run (night of 10-11 Oct): the scored file and the two operational approaches

`lab/final_run.sh`, started 22:09 UTC on 10 Oct (the owner freed the machine for 7 h), finished 01:36 UTC on 11 Oct
after one restart at 00:39.
- **G (GPU).** The five stage members, each refitted once on every covered 2025 day and saved: v70s_v55g 12 min,
  v70s_v58g 13 min, v70s_v55d 50 min, v70s_v58d 50 min (GPU); v70s_v55c 99 min on the CPU beside them. The CPU member
  reproduces the original v55c's 344,841 Jan+Jul ranking predictions exactly (max |diff| 0.000000 s): `--no-cv` fits
  what the cross-fitted run's final fit did. A GPU member moves them by 7.2 s rms on the corrected rows (v70s_v55g
  against v55g, mean +0.01 s): GPU noise. Then the long-horizon deliverables refitted: op_hl3_r4_rwyblind (holdout OP
  MAE 215.63 s, as before) and op_hl24_r4_rwyblind (222.12 s).
- **C1 (runbook, 70 min).** Data tree `data_final`, version final26_7a06db85, 1,341,197 ranking movements; 27 predict
  runs (parity on the training rows: 16 blocks identical, 10 changed, the whole-frame statistics moving with the two
  new months; the count blocks rescaled, uncorrected x1.14-1.16 on the ranking rows); blend, stack, the base-only
  fallback file (670,790 rows) and its level check (only LIRF July flagged, +60 s).
- **C2 (ADS-B tables).** Points for 470 days, the four months' 744 old tables stripped, the per-day tables of the 120
  days against the frozen pools (about 1 h 15 min of CPU). It stopped once: 2026-06-11 is a near-empty adsb.lol day
  (3.8 MB of points against a median of 23.8 MB; 83 links, one detected pushback), on which no departure showed a push
  or power-out, so the stand archive's observation frame had no `kind` column. Fixed in lab/ideas/stand_archive.py
  (ecc49cd); the restart found the live GPU phase and did not start another, and the runbook and the tables resumed.
- **F (20 min).** `final_challenge.py --stage-models`: the eight extra members predicted, the combined base, the
  final tree's lab matrix (counts rescaled), the five stage members from their saved models: 468,265 of the 670,790
  departures corrected by the residual members (69.8 %; 70.4 % on Jan+Jul), 460,568 by the direct ones; the stage
  candidate's holdout equals v70's v60pre to 0.000000 s; then the base swap (202,525 uncovered ranking rows), the
  forced-long shift (estimated on v70's holdout) and S79. Final run `2026-10-11T03-22-21__final_v70_5c8bf3__5c8bf33e`.
- **The scored file**, `submissions/submitting.parquet` (sha1 6cdf67a40617...): 670,790 rows, the final template's id
  set, no NaN, 30 to 111,569 s, median 940 s. Level check against the PRU: where its reference share is at least 0.9,
  the predicted monthly means of February and June lie within -6 and +8 s of the PRU's, except EDDF February (+38 s)
  and LEBL June (+28 s); year-over-year correlation 0.65 (the base alone 0.62); flag LIRF July (+59 s), as for v68.
  (Noted 13 Oct: both correlations cover all 40 airport-months. On the 23 with a reference share of at least 0.9, the
  basis of the operational files' correlations in 10.35 and 10.36, they are 0.91 and 0.89.)
  On the Jan+Jul rows it differs from v70 by 14.7 s rms (mean +0.19 s, median 5.2 s, 99th percentile 43 s); the stage
  rehearsal (v70's recipe with the retrained stage on the delivered inputs, `rh_stage_5bad7d`) splits that into 2.6 s
  from retraining the stage (median 0.5 s) and 14.5 s from the release's jittered times and the larger frame.
- **O (the operational approaches, not scored)**: `final_op_h1_r4` (approach 2, 15 min before the planned off-block;
  median 961 s), `final_op_hl3_r4_rwyblind` and `final_op_hl24_r4_rwyblind` (approach 1, 3 h and 24 h ahead; medians
  971 and 974 s), each `submissions/<run>__submitting.parquet` with 670,790 rows.
- **R6** (approach 2 with the ADS-B surface picture of other aircraft, trained on the free GPU after G, with its seed
  twin and the shift-7 placebo, which reads the surface picture of a week earlier; `lab/op_ladder_run.sh`, deadline
  05:00 UTC). Jan+Jul 2025 OP MAE / RMSE / within 2 min:

  | model | seed 42 | seed 43 | mean |
  |---|---|---|---|
  | R4 | 188.14 / 280.41 / 46.17 % | 187.89 / 280.32 / 46.40 % | 188.02 / 280.37 |
  | R6 | 188.24 / 280.29 / 46.05 % | 186.65 / 279.61 / 46.92 % | 187.45 / 279.95 |
  | R6 shift-7 placebo | 188.42 / 280.49 / 46.08 % | | |

  R6 is 0.6 s of MAE and 0.4 s of RMSE better than R4 on the two-seed means, and the placebo is 1.0 s worse than R6's
  mean, but R6's seeds differ by 1.6 s (R4's by 0.25 s; the block dropout makes R6's fits vary more): a small gain at
  most, not resolved by two seeds. Approach 2 stays R4: a gain of that size does not justify a live ADS-B dependency
  (and R6 reads nothing at Heathrow, Paris and Istanbul by the support table, findings 10.31).
- **On the board (11 Oct).** v72, the full final file (670,790 rows), was rejected by the live scorer ("325949 rows not
  in truth": the board's truth still covers January and July only); the object stays in the bucket. v73, the same
  file's January and July rows (`2026-10-11T03-22-21__final_v70_janjul__0a247cd0`, 344,841 rows), scored **220.393**
  against v70's 220.340: the final pipeline, on the release's jittered inputs, with the larger frame and the retrained
  stage, lands where v70 did. The final submission goes through the same uploads, then an e-mail with the repository's
  URL to the organisers (owner, 11 Oct); since the latest object may count, the full final file is uploaded again as the
  last version before the deadline, after any board check of a January and July subset.

### 10.35 The operational model as the deliverable (owner's goal of 11 Oct): plan and pre-registered rules

**Goal.** The owner set this on 11 Oct: a taxi-out model that uses only data available before or at the off-block, with
every input documented together with its operational rationale.

**The deliverable.** It is one model family, one global CatBoost per decision moment:
- **H1**, the headline: 15 min before the last filed off-block time (IOBT, else LOBT, else the schedule).
- **3 h and 24 h ahead**, the planning companions.
- **H0**, at the off-block. It is evaluated on 2025 only, because 2026 blanks every off-block.

**The scored file.** It stays the challenge model. v74, the full final file again (07:29 UTC), was rejected by the live
scorer as v72 was. After the board checks of 12 Oct it went up once more as v78, and after those of 13 Oct as v83 (10.36), the newest
object in the bucket.

**Corrections decided before any run**, whatever they cost:
- **The long horizons' clock.** It becomes the schedule, not the IOBT:
  - IOBT equals LOBT on 99.45 % of the NM-matched 2025 departures.
  - It carries most of a late flight's delay.
  - The data hold no filing time.

  The IOBT-clock runs of 10.31 stay as the sensitivity "filed plan known" (O*).
- **The day of year** is dropped in every deliverable. Kept, it maps 2025's moving holidays onto 2026 dates.

**Measurement.**
- **Population:** OP, the NM-matched departures with a target below 3,600 s (338,277 rows in Jan+Jul 2025). It is a
  selection on the outcome (C).
- **Metrics:** MAE is the headline, beside RMSE and the share within ±2 min.
- **Reference numbers** (seed 42, `lab/op_rules.py metrics`):

  | run | OP MAE | OP RMSE | within ±2 min |
  |---|---|---|---|
  | op_h1_r4_nodoy | 187.52 s | 279.61 s | 46.3 % |
  | op_h1_r4_nodoy_marnov, November only | 173.46 s | | |
  | op_h1_r4_nodoy_febdec, December only | 180.25 s | | |

- **Seed floor:** 0.25 s of OP MAE between op_h1_r4's seeds 42 and 43 (10.34). Two GPU seeds give only a lower bound of
  the noise. A result within 0.25 s of a threshold is reported as unresolved and keeps the rule's default.

**The rules.** They are fixed here, before their runs, and computed by `lab/op_rules.py` from holdout predictions only.
The decisions go to `runs/op_decisions.json`.
- **L (learning rate of the H1 deliverable).** `op_h1_r4_nodoy_lr04` (lr 0.04, cap 20,000 rounds, od_wait 400)
  replaces `op_h1_r4_nodoy` (lr 0.08) only if both hold:
  - its seed-42 OP MAE is at most 186.77 s, which is 187.52 minus three seed floors;
  - its OP RMSE is at most 279.61 s.

  The queue runs it before the refits, and the refit step reads the choice.
- **R (r3 or r4 per long lead, on the schedule clock).** Choose r3, without the NM history of other flights, if its OP
  MAE is within +0.5 s of r4's both on Jan+Jul and on the forward check (fit Jan-Oct, validate Nov+Dec). Otherwise
  choose r4. The schedule-clock models replace the IOBT-clock files whatever their level; the gap between the two is
  reported as the value of a filed plan.
- **C (median output, per member).** Ship a median with P10 and P90, calibrated on the 2025 month pairs alone, only if
  all of these hold on Jan+Jul:
  - OP MAE improves by at least 1 s;
  - the share within ±2 min improves by at least 1 point;
  - no airport gets worse by more than 2 s;
  - the P10-P90 interval covers 75 to 85 %.

  Otherwise ship the mean only.
- **M (MAE loss with exact leaves; a sensitivity).** `op_h1_r4_nodoy_mae_exact` replaces the calibrated median only if it
  beats it by at least 1 s of OP MAE and 1 point within ±2 min. Its refit would then follow.
- **F (forward tripwire).** It trips if `op_h1_r4_nodoy_fwd` (fit Jan-Oct 2025, validate Nov+Dec) has a November OP
  MAE above 181.46 s or a December one above 188.25 s, that is, the month pairs' level plus 8 s.
  - The action, if tripped: the model card states the forward MAE as the expected out-of-time level beside the Jan+Jul
    figure, and lists the excess as a limit. The likely cause is that the target-free tables are built over the whole
    frame, i.e. transductively; the target statistics come from fit rows only.
  - The files ship either way. The deadline leaves no time for a redesign.
- **Documentation only**, deciding nothing:
  - `op_h1_r4_nodoy_noplan`: without the planned delay and the plan backlog. That is the O* bound of the H1 anchor:
    the last filed plan is assumed known 15 min ahead, as an A-CDM TOBT is.
  - `op_h1_r4_nodoy_notaf`: without the 13 TAF columns. The TAF archive is Ogimet's; in operations the TAF comes from
    the met service.
  - `op_h1_r4_nodoy_rwyblind`: the like-for-like runway-blind comparator of the headline.
  - `op_h0_r4`.
  - The long ladder on the schedule clock.
  - R6 on the month pairs.

**Runs.** They go one at a time through `lab/op_queue.sh`, a restartable queue that reads its steps from
`runs/op_queue.txt` (noted 13 Oct: `runs/` is gitignored; the queue of 11-13 October, with its steps, exit codes and
times, is tracked in `docs/operational_queue_2026-10-11.txt` and `docs/operational_queue_2026-10-11_done.txt`):
1. Rule L's run.
2. Rule L.
3. The H1 deliverable: three seeds, each refitted (`--seeds 3 --keep-test`).
4. The forward check and rule F.
5. The long deliverables, once the schedule clock is merged and its gates pass.
6. The sensitivities.

**Results, as they come in (11 Oct).**
- **Rule L: keep lr 0.08.** `op_h1_r4_nodoy_lr04` (seed 42, stopped at its best round after 35 min) scored:

  | | lr 0.04 | lr 0.08 | threshold |
  |---|---|---|---|
  | OP MAE | 187.00 s | 187.52 s | 186.77 s |
  | OP RMSE | 279.12 s | 279.61 s | 279.61 s |
  | within ±2 min | 46.49 % | 46.32 % | |

  The MAE gain of 0.52 s misses the threshold by 0.23 s, within one seed floor, so it is unresolved and keeps the
  default. B1 refits `op_h1_r4_nodoy` (lr 0.08) with seeds 42, 43 and 44.
- **The long horizons on the schedule clock** (commit 1817b0a):
  - P is the schedule on every row. `op_hl_plan` keeps only `has_stand`, the demand counts no IOBT, and the calendar
    has no day of year.
  - New gate T1e: every flight's IOBT shifted, blanked or removed leaves every long block and the archive
    bit-identical, with the perturbed rows kept in the fit.
  - New runway-blind r3 rungs and forward twins for rule R.
  - The deliverable candidates keep 86 (r4) and 78 (r3) columns at each lead. On the schedule clock the hour at the
    planned off-block equals the scheduled hour; both columns stay.
- **Rule hygiene.** The ladder configs r0-r4 changed only in comments, so the schedule-clock runs share their config
  hashes with the IOBT-clock runs of 10 Oct. The rules and the delivery therefore read only runs of each config's
  current hash.
- **The H0 comparator** is `op_h0_r4_nodoy` (day of year dropped as in every deliverable), so that H1 against H0
  differs by the moment alone.
- **The H1 deliverable (B1).** `op_h1_r4_nodoy`, lr 0.08, seeds 42-44, each refitted on all twelve months; 106 min of
  GPU. OP figures on Jan+Jul 2025:

  | run | MAE | RMSE | within ±2 min |
  |---|---|---|---|
  | seed 42 | 187.57 s | 279.65 s | 46.27 % |
  | seed 43 | 186.72 s | 279.40 s | 46.79 % |
  | seed 44 | 187.50 s | 279.77 s | 46.42 % |
  | **mean of the seeds** | **186.68 s** | **278.74 s** | **46.6 %** (within ±5 min 83.0 %) |
  | calibrated median (rule C) | **183.10 s** | 281.43 s | **49.2 %** (within ±5 min 83.6 %) |

  - The seed-42 run reproduces the holdout-only run of 9 Oct (187.52 s) within GPU noise.
  - The seeds span 0.85 s of MAE (186.72-187.57 s), more than the 0.25 s floor assumed from two seeds of R4. As 10.35 said,
    two seeds give only a lower bound of the noise. Rule L's lr 0.04 run (187.00 s) lies inside the range of the lr 0.08
    seeds.
- **Rule C passed for H1.** Tables from the five month pairs of `op_h1_r4_nodoy` (lr 0.08, one seed each), applied to
  the seed mean:
  - OP MAE 186.68 → 183.10 s (−3.58 s);
  - within ±2 min +2.55 points;
  - every airport improves, by 0.74 s (LTFM) to 8.44 s (EGLL);
  - the P10-P90 band covers 77.7 % (11.6 % below P10, 10.7 % above P90; 76.8-79.1 % per airport), median width
    472 s;
  - RMSE rises by 2.7 s, as a median must.

  The H1 files therefore carry the median and the band beside the mean.
- **The H1 file of the final set** (D_h1, 8 min).
  - **Runs.** Each seed's refit predicted on `data_final` (final26_7a06db85, no retraining):
    `2026-10-11T13-49-06__final_op_h1_r4_nodoy__00e26ad8__s42`, `..._060ef556__s43`, `..._3a749054__s44`. Their
    blend is `2026-10-11T13-55-49__final_op_h1_blend__58d3474d`.
  - **The file.** `submissions/operational/h1__2026-10-11T13-55-49__final_op_h1_blend__58d3474d.parquet`, sha1
    46e4da46b0e1...: 670,790 rows of the final template with `y_mean`, `y_med`, `y_p10`, `y_p90`. Medians: 962 s
    (mean), 920 s (median), 723 s (P10) and 1,214 s (P90).
  - **Level check against the PRU's monthly mean taxi-out**, on the 23 airport-months where its reference share is at
    least 0.9:
    - the predicted mean lies above the PRU's by +1 to +71 s, +18 s on average;
    - year-over-year changes correlate at 0.90;
    - only LIRF July is flagged (+71 s), as for the challenge model and v68.

    Where the reference share is lower, mostly in January, the PRU mean is a floor. EHAM January lies 77 s below it:
    the snow days of 4-5 Jan 2026 are the kind of disruption no pre-movement input foresees.
  - `submissions/submitting.parquet` is unchanged (sha1 6cdf67a4...).
- **Rule F: not tripped.** `op_h1_r4_nodoy_fwd` (`2026-10-11T13-56-03__op_h1_r4_nodoy_fwd__6cc53695__s42`) is fitted on
  January-October 2025 and validated on November and December. OP MAE 178.15 s, RMSE 258.32 s, 46.6 % within ±2 min:

  | month | forward | month pair | excess | tripwire |
  |---|---|---|---|---|
  | November | 176.65 s | 173.46 s | +3.19 s | +8 s |
  | December | 179.62 s | 180.25 s | −0.63 s | +8 s |

  - The month pair is fitted on the other ten months, which include the months after the query.
  - The out-of-time cost is a few seconds at most. The target-free tables built over the whole frame do not flatter the
    holdout measurably.
  - Munich is the weakest airport there (191.0 s), in its de-icing season.
- **The long horizons on the schedule clock, and rule R.** Seed 42 each, OP MAE / RMSE / within ±2 min:

  | lead | r4 runway-blind | r3 runway-blind | r4 on the IOBT clock, with the day of year (10-11 Oct) |
  |---|---|---|---|
  | 3 h, Jan+Jul | 215.05 / 313.73 / 40.2 % | 212.85 / 311.97 / 40.7 % | 215.63 / 312.38 / 39.9 % |
  | 3 h, forward Nov+Dec | 208.18 / 294.17 / 39.4 % | 209.27 / 295.36 / 38.9 % | |
  | 24 h, Jan+Jul | 220.33 / 323.70 / 39.5 % | 218.59 / 322.23 / 40.1 % | 222.12 / 324.25 / 39.2 % |
  | 24 h, forward Nov+Dec | 214.98 / 306.12 / 38.1 % | 214.79 / 305.27 / 38.2 % | |

  - **3 h: r4.** r3 − r4 is −2.20 s on Jan+Jul but +1.09 s on the forward check, beyond the +0.5 s margin. NM's
    history of the hours before helps where the holdout follows the fit in time.
  - **24 h: r3.** r3 − r4 is −1.74 s and −0.19 s: a day-old history adds nothing.
  - **The value of a filed plan.** Moving from the IOBT clock to the schedule clock (and dropping the day of year) costs
    nothing measurable: −0.6 s of MAE at 3 h and −1.8 s at 24 h, both in favour of the schedule clock and inside the
    seed spread of 10.35 (0.85 s at H1). (Corrected 13 Oct: only the 3 h change lies inside that spread. The 24 h
    change is twice it, and its paired day-block interval excludes zero: the step back to the IOBT clock, `L24-IOBT` in
    `docs/operational_ladder_steps.csv`, is +1.78 s [+0.12, +3.50], both runs seed 42.) Read hours ahead, the IOBT
    carried no usable information beyond the schedule. The correction of 10.35 thus removes an O* assumption at no cost.
  - The deliverables are `op_hl3_r4_rwyblind` and `op_hl24_r3_rwyblind`. B2 refits them with seeds 42 and 43. Both
    candidates of each lead keep their refit flag, so that the rule's runs keep their config hashes.
- **The long deliverables (B2).** Each is refitted on all twelve months with seeds 42 and 43 (37 and 36 min of GPU).
  Single-seed OP MAE / RMSE / within ±2 min on Jan+Jul:

  | deliverable | run | MAE | RMSE | within ±2 min |
  |---|---|---|---|---|
  | `op_hl3_r4_rwyblind` | `2026-10-11T15-20-52__op_hl3_r4_rwyblind__c93d69e3__s42` | 215.11 s | 313.76 s | 40.15 % |
  | | `..._s43` (`2026-10-11T15-38-22`) | 214.95 s | 313.39 s | 40.11 % |
  | `op_hl24_r3_rwyblind` | `2026-10-11T15-57-44__op_hl24_r3_rwyblind__930c7b15__s42` | 218.73 s | 322.22 s | 40.10 % |
  | | `..._s43` (`2026-10-11T16-13-52`) | 218.54 s | 322.19 s | 40.12 % |

  The seed means and the calibrated medians follow from the month pairs (B3) and rule C.
- **Rule C for the long members: not passed, by Istanbul alone.** Tables come from each deliverable's own five month
  pairs (B3: `op_hl3_r4_rwyblind_<pair>`, `op_hl24_r3_rwyblind_<pair>`). They are applied to the two-seed means:

  | member | MAE mean → median | ±2 min mean → median | worst airport | P10-P90 coverage |
  |---|---|---|---|---|
  | 3 h | 214.52 → 211.60 s | 40.2 → 42.7 % | LTFM +2.16 s | 77.7 % |
  | 24 h | 218.19 → 215.38 s | 40.1 → 42.6 % | LTFM +2.18 s | 77.7 % |

  - At each lead the median improves nine of the ten airports (up to −6.9 s at EGLL, 3 h), but Istanbul gets worse
    beyond the pre-registered 2 s.
  - The 3 h and 24 h files therefore hold the mean only. The rule was fixed before the runs and is not revisited.
  - Istanbul's January and July residuals evidently do not follow its other months' hours ahead (at H1 the median
    improved Istanbul by 0.7 s).
  - Two-seed means: 3 h RMSE 312.83 s, within ±5 min 78.0 %; 24 h RMSE 321.60 s, within ±5 min 77.7 %.
- **The 3 h file** (D_hl3). The refits were predicted on `data_final` (`..._final_op_hl3_r4_rwyblind__344fa3e3__s42`,
  `..._8864ceeb__s43`) and blended into `2026-10-11T18-09-14__final_op_hl3_blend__4d74dbe5`.
  - **The file.** `submissions/operational/hl3__2026-10-11T18-09-14__final_op_hl3_blend__4d74dbe5.parquet`, sha1
    e2252da943f5..., 670,790 rows, `y_mean` only, median 973 s.
  - **Level check.** On the 23 reliable airport-months the predicted mean lies +27 s above the PRU's on average (−56 to
    +86 s), with a year-over-year correlation of 0.56 (H1: 0.90). Four July airport-months are flagged: EHAM −56,
    LEBL +61, LIRF +86 and LTFM +77 s. EHAM January lies 98 s below its floor. Three hours ahead, without the
    traffic and weather of the moment, the model tracks the typical month, not this one.
- **The 24 h file** (D_hl24). The refits were predicted on `data_final` (`..._final_op_hl24_r3_rwyblind__2d102067__s42`,
  `..._83e0fac8__s43`) and blended into `2026-10-11T18-11-54__final_op_hl24_blend__13b5a89d`.
  - **The file.** `submissions/operational/hl24__2026-10-11T18-11-54__final_op_hl24_blend__13b5a89d.parquet`, sha1
    a395355fae0f..., 670,790 rows, `y_mean` only, median 972 s.
  - **Level check.** +25 s above the PRU's on average on the 23 reliable airport-months (−49 to +92 s), with a
    year-over-year correlation of 0.51. Flagged: EGLL July +63, LIRF July +92 and LTFM July +77 s.
- **The deliverable files of 11 Oct, all in `submissions/operational/` with `manifest.json`** (the sha1 of every file,
  the source, predict and blend runs, the data version, the template's sha1 and rule C's record):

  | moment | file | sha1 | columns |
  |---|---|---|---|
  | H1 | `h1__2026-10-11T13-55-49__final_op_h1_blend__58d3474d.parquet` | 46e4da46b0e1... | mean, median, P10, P90 |
  | 3 h | `hl3__2026-10-11T18-09-14__final_op_hl3_blend__4d74dbe5.parquet` | e2252da943f5... | mean |
  | 24 h | `hl24__2026-10-11T18-11-54__final_op_hl24_blend__13b5a89d.parquet` | a395355fae0f... | mean |

  The scored file `submissions/submitting.parquet` is unchanged (6cdf67a4...).

  Noted 13 Oct: `submissions/` is local and gitignored, so a clone has neither these files nor the manifest.
  `docs/operational_manifest.json` is the manifest's tracked copy (as of 13 Oct): it records every delivered file
  with its full sha1. QUICKSTART path B regenerates the files with `lab/op_deliver.py`.
- **No TAF at H1 (D7).** `op_h1_r4_nodoy_notaf` (`2026-10-11T18-12-08__op_h1_r4_nodoy_notaf__c882830a__s42`), without
  the 13 TAF columns, scores OP MAE 187.30 s, RMSE 279.54 s and 46.4 % within ±2 min.
  - That is inside the deliverable's seed range (186.72-187.57 s), so the TAF is not needed at H1, where the METAR of
    the moment carries the weather.
  - The licence caveat of Ogimet's archive therefore does not touch the headline. The 3 h and 24 h models were not
    tested without it; hours ahead the TAF is their only weather for the scheduled hour.
- **No plan at H1.** `op_h1_r4_nodoy_noplan` (`2026-10-11T18-31-39__op_h1_r4_nodoy_noplan__0796a5c6__s42`) drops the
  planned delay and the plan backlog of airport, runway and pier.
  - It scores OP MAE 189.80 s, RMSE 283.24 s and 45.75 % within ±2 min, against 187.57 / 279.65 / 46.27 % for the
    same seed with them.
  - The plan's columns are worth 2.2 s of MAE, 3.6 s of RMSE and half a point within ±2 min. This bounds H1's O* plan
    assumption for its inputs; the anchor itself is the plan time by definition.
  - Without them the model is still 17.6 s better than the stand × runway table.
- **H1 runway-blind.** `op_h1_r4_nodoy_rwyblind` (`2026-10-11T18-45-17__op_h1_r4_nodoy_rwyblind__51329d01__s42`) scores
  OP MAE 204.96 s, RMSE 298.00 s and 41.8 % within ±2 min. That is the same as `op_h1_r4_rwyblind` with the day of
  year (204.91 s). The planned runway is worth 17.4 s at H1 (same seed). Runway-blind, H1 still beats the 3 h model by
  about 10 s: the weather and traffic of the moment are worth that much over the forecast and the schedule.
- **H0 with the deliverable's inputs.** `op_h0_r4_nodoy` (`2026-10-11T18-59-51__op_h0_r4_nodoy__9b630cdd__s42`): the
  same inputs as H1, read at the flight's own off-block, without the airport's off-block feed. OP MAE 178.22 s, RMSE
  265.42 s, 48.5 % within ±2 min, 84.3 % within ±5 min, MAE/SD 0.431.
  - Waiting for the push is worth 9.4 s over H1 (187.57 s, same seed).
  - With the feed (H0 R5 of 10.27) the push reaches 171.4 s, so the other flights' off-block records add about 7 s
    there.
- **Rule M on the GPU crashed.** `op_h1_r4_nodoy_mae_exact` (MAE loss, exact leaves, plain boosting) died after its
  first round with CatBoost's "CUDA error 700: an illegal memory access" (exit 127). The synthetic smoke fit of the
  morning had passed.
  - The next GPU run started cleanly in a new process, so the queue was not disturbed.
  - Rule M is a sensitivity. It runs last in the night's queue on the CPU, where exact leaves are supported
    (`-o model.params.task_type=CPU name=op_h1_r4_nodoy_mae_exact_cpu`).
- **The day of year overfits most where inputs are few.** The long r0 rung (record and plan, both leads, runway-aware)
  scores OP MAE 196.89 s on the schedule clock (`2026-10-11T19-18-12__op_hl_r0__3f261824__s42`), against 206.61 s on
  the IOBT clock with the day of year (`2026-10-10T03-24-28`, same config hash).
  - Its inputs are a strict subset of the old ones: no day of year, no IOBT, the calendar at the schedule. So the gain
    cannot be new information.
  - The old run gave `op_hl_day_of_year` 11.3 % of its gain, its second input. It sits at Paris (224 against 266 s)
    and Amsterdam (148 against 182 s), the airports where the weather step of 10.27 was largest.
  - Without weather or traffic, the day of year memorises the fit months' day-level conditions: Paris's works in the
    summer of 2025, Amsterdam's winter days. A seasonal holdout puts them on the wrong days.
  - **Consequence.** The H1 ladder of 10.27 (r0-r3) carried the day of year in every rung, so its steps may be
    inflated. Its weather step (r1 → r2, −14 s) is the suspect: weather replaces what the day of year faked. The
    calendar-free rungs `op_h1_r0_nodoy` … `op_h1_r3_nodoy` re-measure the ladder (queue steps A8). Until then the
    ladder's step sizes are provisional.
- **The H1 ladder without the day of year (A8): the step sizes of 10.27 corrected.** Seed 42, OP MAE / RMSE /
  within ±2 min; with the day of year as in 10.27 in brackets:

  | rung | calendar-free | step (MAE / RMSE) | with the day of year (10.27) |
  |---|---|---|---|
  | r0 record and plan | 196.00 / 302.94 / 45.5 % | | (207.27 / 310.71) |
  | r1 + layout | 195.33 / 302.01 / 45.7 % | −0.7 / −0.9 | (206.8 / 308.9) |
  | r2 + weather at the anchor | 192.80 / 288.92 / 45.5 % | −2.5 / **−13.1** | (192.8 / 289.0) |
  | r3 + other flights' movement records | 190.90 / 286.31 / 45.9 % | −1.9 / −2.6 | (191.3 / 286.1) |
  | r4 + NM data of departed flights, plan backlog | 187.57 / 279.65 / 46.3 % | −3.3 / −6.7 | (188.1 / 280.4) |

  - **The day of year cost 11.3 s at r0 and r1 and nothing from r2 on.** With weather in the model it no longer stands
    in for day-level conditions.
  - **The weather step was mostly that artefact.** It is −2.5 s of MAE (not −14 s), but −13.1 s of RMSE. Weather
    decides the tail, not the typical flight: Munich −16 s, Istanbul −10 s, Paris −9 s, Heathrow +7 s of MAE.
  - **What carries the signal.** The record, stand, runway and calendar alone reach 196.0 s, 11.4 s better than the
    stand × runway table. Everything known 15 min ahead adds 8.4 s of MAE and 23 s of RMSE. The largest MAE step is
    NM's data of the departed flights with the plan backlog (−3.3 s); the largest RMSE step is weather (−13.1 s).
  - **Hours ahead** (the long ladder on the schedule clock, runway-aware, an O* upper bound):

    | rung | 3 h | 24 h |
    |---|---|---|
    | r0 (shared) | 196.89 / 304.22 | 196.89 / 304.22 |
    | r2 + layout, forecast weather | 193.72 / 292.02 | 198.87 / 301.38 |
    | r3 + scheduled demand | 191.20 / 289.43 | 196.73 / 299.37 |
    | r4 + NM history | 193.49 / 291.26 | 198.73 / 300.82 |

    The forecast weather is worth 3.2 s of MAE and 12 s of RMSE at 3 h and nothing a day ahead. The scheduled demand
    is worth 2.5 s at 3 h and 2.1 s at 24 h. NM's history helps neither runway-aware rung on Jan+Jul.
  - The 10.27 figures stay as measured; the papers and README now quote the calendar-free ones.
- **R6 on the month pairs: no gain (10.34's open question settled).** `op_h1_r6_<pair>` (R4 plus the public ADS-B
  surface picture, seed 42; 12 Oct 01:40-03:10 UTC) is compared with `op_h1_r4_<pair>` of 9-10 Oct, the same inputs
  without ADS-B and both with the day of year. OP MAE change per validation month:

  | Feb | Mar | Apr | May | Jun | Aug | Sep | Oct | Nov | Dec |
  |---|---|---|---|---|---|---|---|---|---|
  | +0.06 | +0.43 | +0.09 | −0.24 | +0.16 | +0.24 | +0.03 | +0.04 | −0.08 | −0.24 |

  - Weighted by rows: +0.05 s of MAE, +0.01 s of RMSE, +0.03 points within ±2 min.
  - R6 is better in three months of ten, and no month moves by half a second. The 0.6 s of the two Jan+Jul seeds (10.34)
    was noise.
  - The surface picture of other aircraft adds nothing to R4 before the movement, so no deliverable reads ADS-B.
- **The night of 11-12 Oct.** The machine went into Modern Standby from 19:38 to 01:03 UTC. The queue's CPU step
  (CLK7) was suspended and resumed, and nothing was lost. Its runway table equals the vectorised rerun's
  (`lab/op_clock_checks.py` section 7, now merged rather than a set per departure: 4 min instead of hours).
- **Rule M: passed.** `op_h1_r4_nodoy_mae_exact` ran on the CPU (`2026-10-12T05-10-45__op_h1_r4_nodoy_mae_exact_cpu__94e24e8c__s42`;
  MAE loss and early stopping, exact leaves, plain boosting, seed 42), stopping at round 4,942 after 135 min.
  - It scores OP MAE 181.16 s, RMSE 284.07 s, 50.23 % within ±2 min and 84.2 % within ±5 min, with bias −40 s.
  - Against the shipped calibrated median (183.10 s, 49.16 %) that is −1.94 s and +1.06 points, so both pre-registered
    criteria are met (≥ 1 s and ≥ 1 point). The ±2 min criterion is met by 0.06 points only, within the seeds' spread of
    that share (0.5 points between seeds 42-44 of the RMSE model). Against seed 42's own calibrated median: −2.60 s and
    +1.26 points.
  - Noted 13 Oct: rule M's record in `runs/op_decisions.json` (key `M`) was written outside the lab scripts.
    `python lab/op_rules.py metrics 2026-10-12T05-10-45__op_h1_r4_nodoy_mae_exact_cpu__94e24e8c__s42` prints the
    candidate's OP metrics; the bar of 10.35 (1 s and 1 point) sets them against `C_h1`'s calibrated median.
  - Per the rule, the MAE model, refitted on all twelve months, replaces the calibrated median (`y_med`) of the H1 file.
    The mean (three RMSE seeds) and the P10-P90 band (the calibration tables) stay.
  - **Clamped into the band**, as the file holds it (`lab/op_deliver.py --median-config`; ladder row `D-H1-MAE`), so
    that the median never leaves its own band. Not part of the pre-registered rule. It moves 0.54 % of the OP rows
    (0.53 % up to P10, 0.01 % down to P90): OP MAE 181.12 s, RMSE 283.63 s, 50.21 % within ±2 min, 84.19 % within
    ±5 min, bias −39.7 s. Against the calibrated median per airport: −0.9 s (EDDM) to −3.3 s (EHAM); Istanbul equal
    (207.06 s both, on different predictions).
  - The refit ran on the CPU (the GPU fails with exact leaves) from 12:00 to 18:34 UTC on 12 Oct
    (`2026-10-12T14-00-49__op_h1_r4_nodoy_mae_exact_cpu__00a32933__s42`, 395 min). Its holdout phase reproduced the
    morning's run exactly (max |difference| 0.0 s on 344,419 rows: CatBoost on the CPU is deterministic).
  - The H1 file: `submissions/operational/h1__2026-10-11T13-55-49__final_op_h1_blend__58d3474d__mae.parquet`, sha1
    1dca8c80175d..., 670,790 rows; the member's prediction clamped into the band on 0.81 % of them; median 921 s
    against 962 s for the mean. The calibrated-median file stays, recorded as `h1_calibrated_median` in the manifest.
    (Noted 13 Oct: the folder also holds `h1__2026-10-11T13-55-49__final_op_h1_blend__58d3474d__calibrated.parquet`,
    `lab/op_deliver.py`'s calibration intermediate: float64 values, an `apt` column, indexed by `mvt_id`. It is not a
    deliverable, and no manifest record lists it.)
  - An absolute-error loss with exact leaves converges where the gradient-leaf MAE rungs of 10.27 did not (190.3 s at
    their caps). It is the better point forecast for the A-CDM tolerance shares.
- **The deliverables on the board (12 Oct, 06:30 UTC).** The means' January and July rows of the final files, cut by
  `lab/board_subset.py` (it never writes the scored file). Three independent verifiers passed the files first: the
  values against the source blends, the format against v73's accepted file, and the upload log.

  | upload | file | board RMSE | 2025 holdout, RMSE on all rows |
  |---|---|---|---|
  | v75 | H1, `op_h1_r4_nodoy`, three seeds | 560.98 | 584.5 |
  | v76 | 3 h, `op_hl3_r4_rwyblind`, two seeds | 589.09 | 606.2 |
  | v77 | 24 h, `op_hl24_r3_rwyblind`, two seeds | 596.10 | 611.0 |
  | v68 (8 Oct) | H1 R4 with the day of year, seed 42 | 560.92 | 585.3 |

  - The board's RMSE runs over every January and July departure, so the stale and wrong-date records set it. On the
    2025 holdout the 1.8 % of rows outside OP carry 78 % of H1's squared error. The board confirms the files and the
    order of the moments, not the operational accuracy.
  - H1 equals v68 (+0.06). Dropping the day of year and averaging three seeds moved the holdout's all-rows RMSE by
    −0.8 s and its OP MAE by −1.5 s; the board's tail hides that. The final tree alone moved the challenge model by
    +0.05 (v70 220.340, v73 220.393).
  - The hours-ahead members' gap to H1 is wider on 2026 than on the holdout: +28.1 and +35.1 s against +21.7 and
    +26.5 s. The board cannot say whether the typical flight or the tail moved. The level check points to the typical
    flight: the long members track the typical month (correlation 0.51-0.56 with the PRU's year-over-year changes,
    against 0.90 at H1; all three on the same 23 reliable airport-months, noted 13 Oct).
  - v78 is the full final file again, rejected as v72 and v74 were ("325949 rows not in truth"), so that the newest
    object in the bucket is the scored file. That was four of the day's five uploads.
  - The verifiers also found the H1 file's mean 1 s off rint(blend mean) on 31 of 670,790 rows:
    `lab/op_calibrate.py apply` stored its outputs as float32 before `lab/op_deliver.py` rounded them. It keeps float64
    now, so the rule M rebuild rounds the exact mean.

### 10.36 Traffic awareness after the anchor: the owner's concept and rule T (12 Oct)

**The owner's question (12 Oct, evening).** The operational model should know the traffic situation around the
planned off-block, on both sides of it:
- the actual traffic before it: the other aircraft's ADS-B movements and the runway records over a window;
- the planned traffic after it. In operations the arrival manager (AMAN) holds the landing sequence, the departure
  manager (DMAN) the start-up and takeoff sequence (TSAT, TTOT), and the surface management system (A-SMGCS planning)
  the routes and taxi times. The challenge data hold none of these plans. In our setup the realised movements of the
  other aircraft in a window after the anchor stand in for them.

**What the H1 deliverable reads today** (an inventory of every H1 input by four readers, 12 Oct):
- Before the anchor: the other flights' takeoffs, landings, arrivals taxiing, runway mix and configuration, and the
  arrivals' taxi-in anomaly (airport records, `op_h1_traffic`); NM's readings of the flights that took off in the last
  30-60 min and the plan backlog (`op_h1_nm`). The ADS-B picture of other aircraft before the anchor exists
  (`op_h1_adsb`, rung R6), but it added nothing (+0.05 s on the five month pairs, 10.35), so no deliverable reads it.
- After the anchor: nothing about traffic. Every H1 window ends at the anchor. The only forward-looking inputs are the
  TAF for the next hour and the O* groupings of the plan backlog. The 3 h and 24 h members read the other flights'
  scheduled departures and arrivals around the planned off-block (`op_hl_demand`); H1 does not.

**What is added** (`prc/features/operational_future.py`), each block describing the OTHER flights after the H1 anchor
`a` (15 min before the planned off-block):
- `op_h1_fut_plan`, tier O*: their plans, where the data hold them: planned off-blocks (IOBT, else LOBT, else the
  schedule) and scheduled arrivals in `(a, a + 15/30/60 min]` per airport, own runway and pier.
- `op_h1_fut_rec`, `op_h1_fut_nm`, `op_h1_fut_adsb`, tier F: their realised movements, read with the H1 machinery
  at `a + 15` (the planned off-block), `a + 30` and `a + 45` min: the airport's records, NM's readings of the flights
  that took off, and the public ADS-B surface picture.
- Three past-window columns are left out after the anchor because they would locate the own slot: the gaps since the
  last other takeoff and landing on the own runway, and the minutes since the last in-block at the own stand.
- **The ADS-B own-airframe rule had to change.** Before the anchor the callsign and own-stand rules hold (audit:
  0.04-0.08 % contaminated). After it they do not: the own taxi-out often still carries the inbound leg's callsign and
  starts at a first sighting away from the stand. At `a + 30` min, 10-17 % of the exposed departures carried an own
  event (gate 0.2 %).
  - Fix: the own airframe is identified through its own ADS-B takeoff (callsign within 600 s of the recorded
    takeoff, else the only free takeoff within 45 s). Its whole ground stay at the airport (from its last landing, or
    just after its earlier takeoff) is removed through a per-departure callsign token. A departure whose own takeoff
    is not found is NaN in the block.
  - Two bugs found on the way: stays matched by airframe only (an EGLL stay swallowed the same aircraft's earlier EDDF
    departure), and a start pad that let a later rotation's stay overlap the earlier takeoff.
  - Result on 15 Jan 2025, 15 Jul 2025 and 15 Jul 2026: 1 of about 7,400 exposed linked departures contaminated at
    `a + 30` or `a + 45` min; the unlinked ones are NaN.
  - The recorded takeoff is read only to identify the airframe whose events are removed.

**Tier F.** A new tier: *planned in the concept, realised in the challenge data*. Its inputs come, in operations,
from a planning system (AMAN, DMAN, A-SMGCS) whose plans the challenge data lack; the realised movements of the other
flights stand in. An F model's accuracy is therefore an upper bound for what perfect plans would give, and it is not
available as computed. Two traces of the own movement remain through shared resources, and no test can remove them:
the own takeoff takes one runway slot from the others' counts, and the next arrival at the own stand waits for the
own push. In a plan these traces are the own TSAT and TTOT, not the realised ones. The placebo and the own-runway-free
variant below bound them. (As first planned. Noted 13 Oct: the amendment below replaced both before any run. The
placebo bounds only the airport's era and the flight's identity, and F4-norwy kept the airport-level departure columns,
which carry the own runway's sequence where one runway is in use. Rule T instead removes the departure-side traces by
construction: its member, F4-nodepthru, has no realised departure-side column after the anchor, and nothing from that
side is credited. The next arrival at the own stand stays in the member, unmeasured. So do the ADS-B taxi-in columns'
missingness, which depends on finding the own ADS-B takeoff (about 0.15 s for F4), and, on a mixed-mode runway, the
landings on the own runway (unmeasured).)

**Rule T, pre-registered before any F run** (12 Oct 2026, 18:30 UTC; amended at 19:30 UTC, below). Seed 42, Jan+Jul 2025 holdout, OP population,
against `op_h1_r4_nodoy` seed 42 (OP MAE 187.57 s, RMSE 279.65 s).
- **The runs:**
  - F1 = H1 + plans (`op_h1_f1_plan`, O*);
  - F2 = F1 + records (`op_h1_f2_rec`, F);
  - F3 = F2 + NM readings (`op_h1_f3_nm`, F);
  - F4 = F3 + ADS-B (`op_h1_f4_adsb`, F), the concept member;
  - F4-shift7 (`op_h1_f4_shift7`), the placebo: every F block read 7 days earlier;
  - F4-norwy (`op_h1_f4_norwy`): without the other flights' sequence on the own runway after the anchor;
  - F5 = F4 + the ADS-B picture before the anchor (`op_h1_f5_full`): actual plus planned;
  - F4 forward (`op_h1_f4_adsb_fwd`: fit Jan-Oct, validate Nov+Dec, against `op_h1_r4_nodoy_fwd`) and F4 on the five
    month pairs (against `op_h1_r4_nodoy_<pair>`).
- **F4 counts as a real gain only if all of these hold:**
  1. OP MAE at least 2 s below H1, with the 95 % day-block bootstrap interval of the paired difference below zero;
  2. the shift-7 placebo keeps at most a quarter of that gain;
  3. F4-norwy keeps at least three quarters of it;
  4. F4 beats H1 in at least 8 of the 10 month-pair validation months;
  5. the forward check improves too (Nov+Dec OP MAE below `op_h1_r4_nodoy_fwd`'s).
- **If F4 passes**, the concept member *H1 + planning data* is refitted with two seeds and applied to the final set
  as a separate file labelled tier F. The member is F4, or F5 if F5 beats F4 by at least 0.5 s of OP MAE. **If it
  fails**, F is reported as measured and nothing is delivered from it.
- **F1** (plans only, O*: available in operations) is reported with its interval. The strict H1 deliverable stays as
  it is either way: its files, rules and calibration are frozen. F1 is recorded as a candidate for a later version if
  it gains at least 1 s with the interval below zero.
- **The order of the runs:** F1, F2, F3, F4, F4-shift7, F4-norwy, F5; then, only if F4 passes 1-3, the forward check
  and the month pairs; then the refits.

**Rule T amended before any run** (12 Oct, 19:30 UTC; no F model had been trained). Two adversarial reviews of the
module and a gate writer (66 synthetic tests) found no exploitable leak, but showed that the two bounds above could not
do their jobs:
- **The shift-7 placebo bounds the airport's era and the flight's identity only.** The own movement is never in its
  window, so it cannot bound the own traces. As built it also had two biases in opposite directions:
  - its reads for 1-7 January reached before the data and returned zeros (an empty airport), which makes the placebo
    worse and criterion 2 easier;
  - the same flight a week earlier counted as other traffic, which makes the placebo better and criterion 2 harder.
  Fixed: shifted reads that reach before the records are NaN (`_horizon.window_covered`), and the ADS-B stays now
  carry the departure's own callsign, so the same flight a week earlier is removed by the callsign rule. In the records
  and NM blocks the week-earlier flight still counts as one other flight among about 10-40 in a window; the remaining
  bias is conservative. Every rule T comparison on January and July is scored on the rows from 8 January on, for all
  members alike. (Noted 13 Oct: moved to 9 January before any run, in commit f8fd891 at 20:15 UTC. The placebo's
  week-earlier reads, and the 15 h the runway configuration age looks back from them, must reach the data.)
- **F4-norwy did not remove the own runway's sequence.** It kept the backlog on the own runway after the anchor and
  every airport-level departure column, which equal the own-runway columns wherever one runway is in use (the modal
  departure runway carries 99.8 % of EGLL's departures in a half hour, 98.3 % LEBL, 94.3 % LIRF). A decoy test by the
  leak reviewer showed that airport-level counts of other takeoffs after the anchor carry a trace of the own takeoff
  slot at all six airports tested. The shared resources go beyond the runway: the next arrival at the own stand, the
  pier's pushback lanes, the de-icing pads and the runway holding queue.
- **Criterion 3 is replaced** by a variant without any realised departure-side column after the anchor
  (`op_h1_f4_nodepthru`: the plans, O*, and the realised arrivals, F, only). Its gain is free of the departure-side own
  traces by construction; the next arrival at the own stand remains.
  - If F4-nodepthru keeps at least three quarters of F4's gain, the member is F4 (or F5 if F5 is 0.5 s better).
  - Else the member is F4-nodepthru, if it passes criterion 1 on its own.
  - Else there is no member: the gain cannot be told apart from the own movement's traces.
  - F4-norwy (now without the own-runway backlog too) is reported, not decided on.
- **The month pairs and the forward check run for the chosen member**, through overrides of its configuration.
- **Code fixes** (pinned by the gates): the ADS-B stay is relabelled with the departure's own callsign rather than a
  token, so the callsign rule and the stay rule both act (the token stays for departures without a callsign); the
  time-only link requires mutual uniqueness; events without an airframe never link; a pandas 3 merge-key mismatch;
  every helper module the blocks reach is now hashed; the plan windows are named `w15/w30/w60` (read times stay
  `f15/f30/f45`).
- **A remaining observation.** Whether the ADS-B block is NaN depends on finding the own ADS-B takeoff, an
  observation of the own movement. The leak reviewer measured its value at about 0.15 s of MAE (shorter taxis are a
  little more often unlinked at LSZH, LEMD and EDDF, mostly through stand reception). The placebo keeps the same
  mask, so this part of a gain would survive it and count against criterion 2.

**Rule T, stage 1 (12 Oct, 20:26-23:29 UTC; `runs/op_rule_t.md`, `runs/op_decisions.json` T13).** Seed 42, the
Jan+Jul 2025 OP rows from 9 January on, paired against `op_h1_r4_nodoy` seed 42 (185.30 s on these rows), day-block
bootstrap 95 % intervals. H0, the same inputs at the recorded off-block (2025 lab), is shown for scale.

| run | adds after the anchor | OP MAE | change [95 % CI] | OP RMSE | ±2 min |
|---|---|---|---|---|---|
| H1 deliverable | nothing | 185.30 | | 276.08 | 46.57 % |
| F1 | the other flights' plans (O*) | 182.66 | −2.64 [−3.08, −2.17] | 273.65 | 47.56 % |
| F2 | + their realised movement records (F) | 179.43 | −5.87 [−6.89, −4.82] | 267.60 | 48.15 % |
| F3 | + NM's readings of the flights taking off (F) | 176.50 | −8.80 [−9.95, −7.78] | 264.00 | 48.85 % |
| F4 | + the public ADS-B surface picture (F) | 175.07 | **−10.23** [−11.16, −9.39] | 262.35 | 49.31 % |
| F4-shift7 | placebo: every F block a week earlier | 183.61 | −1.69 [−2.20, −1.08] | 274.91 | 47.29 % |
| F4-nodepthru | plans and the realised arrivals only | 181.33 | **−3.97** [−4.43, −3.50] | 272.11 | 48.01 % |
| F5 | F4 + the ADS-B picture before the anchor | 174.86 | −10.44 [−11.37, −9.56] | 262.22 | 49.36 % |
| H0 (lab, 2025) | the own off-block known | 175.47 | | 261.43 | 48.94 % |

- **Criteria.** c1 holds (10.2 s, interval below zero); c2 holds (the placebo keeps 1.7 s, a sixth of the gain, and less
  than the plans alone give: the gain is the traffic of the day). Under c3 the variant without any realised
  departure-side column keeps 4.0 s, 39 % of F4's gain, short of three quarters: 6.3 s of F4's gain come from the
  realised departure side, where the own takeoff, hold and push leave their traces. F4-nodepthru passes c1 on its own,
  so **the member is `op_h1_f4_nodepthru`**; stage 2 checks it on the month pairs and forward.
- **What each layer is worth.** The other flights' plans, which an airport holds at H1 (O*): 2.6 s, at every airport
  (1.6 s at EDDF to 4.4 s at LIRF). The realised arrivals on top (what a perfect AMAN would give): 1.3 s more. The
  realised departure side: 6.3 s more, but the most at EGLL, LIRF, LTFM and LEBL (EGLL −16.2 s, LIRF −16.0 s, LTFM
  −15.1 s, LEBL −11.5 s for F4, against −6.2 to −8.9 s elsewhere). The departure side alone (F4 against F4-nodepthru,
  same rows): EGLL 12.2, LTFM 9.4, LIRF 9.2, LEBL 7.7 s, 2.1-5.3 s elsewhere; on every OP row from 1 January
  (`docs/operational_ladder_steps.csv`) 7.1-12.5 and 1.9-5.0 s. At EGLL, LEBL and LIRF one runway carries nearly
  every departure (the hour's busiest departure runway takes 99.7, 97.5 and 94.6 % of the departures of January and
  July 2025, against 62-87 % at the other seven airports), which is where the own slot leaves the strongest trace in
  the others' counts; LTFM (65 %) is not explained by that. (Corrected 13 Oct: the first version of this paragraph
  counted LTFM among the single-runway airports.) This side cannot be credited to planning data. ADS-B adds 1.4 s
  after the anchor (F3 to F4) and 0.2 s before it (F4 to F5).
- **The upper bound equals knowing the own off-block.** With everything the other flights did in the next 45 min, H1
  reaches 175.1 s, as close as H0 (175.5 s), which knows the flight's own off-block.
- **F4's block permutation** (RMSE rise when the block is shuffled): NM's readings after the anchor 66.1 s, the stand
  × runway keys 52.4 s, the aircraft 36.5 s, the records after the anchor 27.4 s, the plans after it 3.8 s, the ADS-B
  after it 3.1 s; NM's readings before the anchor fall to 0.9 s (their information is in the later reads).
- **On the owner's hypothesis** that the traffic after the anchor is most of the gap ADS-B opened in the challenge
  model: it is not. The gap between H1 (185 s) and the challenge models (C1 137 s; v62, with the flight's own ADS-B
  track, 104 s) is the flight's own movement: its off-block and its track. The other flights' traffic after the
  anchor closes at most 10 s of it, as an upper bound, and 4 s that can be credited to plans.
- **F1** (plans only, O*) is a candidate for a later version of the strict deliverable (2.6 s, interval below zero).
  Its three seeds are refitted at the end of the night's queue as an option for the owner; the deliverable stays as it
  is unless the owner decides otherwise.

**Rule T, stage 2 (12-13 Oct, 23:29-01:48 UTC; `runs/op_decisions.json` T): passed.** The member `op_h1_f4_nodepthru`
(H1 + the other flights' plans and their realised arrivals after the anchor) was run on the five month pairs and the
forward split, through overrides of its configuration, against `op_h1_r4_nodoy`'s pair and forward runs (seed 42):
- **Month pairs: better in 10 of 10 validation months** (c4). OP MAE change: Feb −1.64, Mar −4.31, Apr −3.87, May
  −3.45, Jun −3.92, Aug −1.26, Sep −3.40, Oct −4.11, Nov −3.69, Dec −5.06 s.
- **Forward** (fit January-October, validate November and December): 172.61 s against 178.15 s, −5.54 s (c5).
- The member is refitted with two seeds (`-o model.refit_on_all=true --seeds 2 --keep-test`) and applied to the final
  set as `submissions/operational/h1f__...parquet` (`lab/op_deliver.py --member h1f`: mean only, the manifest names
  tier F).

**Delivered (13 Oct, 01:48-03:41 UTC).** The member was refitted with two seeds
(`2026-10-13T03-48-19__op_h1_f4_nodepthru__8946614f__s42`, `..._04-23-37_..._s43`; their holdout phases score
184.2 and 183.8 s on the full Jan+Jul holdout, 183.6 s as a mean, against 186.7 s for H1's three seeds; added 13 Oct:
these are OP MAEs on all 338,277 OP rows from 1 January, and the mean's OP RMSE is 275.4 s, with 47.6 % within ±2 min
and 83.5 % within ±5 min) and applied to the final set by `lab/op_deliver.py --member h1f`:
`submissions/operational/h1f__2026-10-13T05-41-39__final_op_h1f_blend__08a90f35.parquet`, sha1 1cae0824c3bb...,
670,790 rows, the mean only, the manifest naming tier F. The level check: +16 s on average against the PRU's monthly
means on the 23 reliable airport-months (−3 to +82 s), year-over-year correlation 0.87, LIRF July flagged (H1: +18 s,
0.90, the same flag). The scored file is unchanged.

**F4-norwy** (informative, run after the decision): 176.56 s, −8.74 [−9.60, −7.94] against H1 on the rule T rows. It
keeps 85 % of F4's gain without the own runway's sequence after the anchor, because the airport-level departure columns
carry the same information where one runway is in use: the own-runway bound of the first rule T could not have bounded
the own slot, as the reviews said.

**Corrections of 13 Oct** (a cross-check of the documentation against the run records), made in place:
- Stage 2's date is 12-13 Oct, not 13 Oct: the queue ran it from 23:29 UTC on 12 Oct to 01:48 UTC on 13 Oct.
- Three roundings now follow their sources. H1's three-seed mean in 10.35 is 186.68 s, not 186.69 s
  (`runs/op_decisions.json` C_h1: 186.6846). The calendar-free weather step is −16 s at Munich and +7 s at Heathrow,
  not −15 and +8 s (`docs/operational_ladder_steps.csv`: −15.93 and +7.45). F4's stand × runway keys are worth
  52.4 s, not 52.5 s (52.45 in the run's `importance_perm.csv`).
- The other changes of that day are annotations, marked where they stand.

**F1's refits and the board checks (13 Oct, 04:12-06:44 UTC).** F1 (H1 + the other flights' plans after the
anchor, O*) was refitted with three seeds as an option for the owner (`-o model.refit_on_all=true --seeds 3
--keep-test`, 146 min of GPU; `2026-10-13T06-12-27__op_h1_f1_plan__40ac5f36__s42`, `..._07-14-31_..._s43`,
`..._07-54-55_..._s44`). Their holdout phases score 184.9, 185.1 and 184.8 s of OP MAE on the full Jan+Jul holdout;
the mean 184.3 s, OP RMSE 276.3 s, 47.4 % within ±2 min, against 186.7 s, 278.7 s and 46.6 % for H1's three seeds
(−2.4 s). It was applied to the final tree as the deliverables are (`prc predict` per seed, `prc blend` without
`--lobt-bound`: `2026-10-13T08-42-02__final_op_h1f1_blend__cab07f2b`) but not delivered: no file in
`submissions/operational/`, the manifest unchanged. Its level check on the 23 reliable airport-months (the PRU's
reference share at least 0.90): +17 s above the PRU's monthly means on average (−1 to +83 s), year-over-year
correlation 0.85, LIRF July flagged (H1: +18 s, 0.90, the same flag).

The owner asked for every candidate on the live board. Each upload is the January and July rows of a final-set file
(`lab/board_subset.py`; its `--file` option reads the medians, which are columns of a delivered file and of no run).
The full final file went up again last:

| upload | candidate | holdout RMSE (all / without wrong-date rows) | board |
|---|---|---|---|
| v75 (12 Oct) | the H1 deliverable's mean, three seeds | 584.5 / 444.0 | 560.98 |
| v79 | the H1 file's median (rule M, its `y_med`) | 595.2 / 455.8 | 575.06 |
| v81 | the calibrated median of rule C, which rule M replaced | 588.6 / 448.8 | 564.87 |
| v82 | F1: H1 + the other flights' plans after the anchor (O*), three seeds | 583.5 / 442.3 | 559.77 |
| v80 | the concept member `op_h1_f4_nodepthru` (tier F), two seeds | 583.5 / 441.6 | 557.53 |
| v83 | the full final file again, rejected ("rows not in truth"); the newest object in the bucket | | |

- The board orders the candidates as the holdout's all rows do. Its RMSE runs over every January and July departure,
  so the stale and wrong-date records set it, as for v75 (10.35).
- The medians lose on the RMSE board, as a median should: rule M's by 14.1 s (10.7 s on the holdout), rule C's by
  3.9 s (4.1 s). They serve the absolute error, which the board does not measure.
- F1 gains 1.2 s on the board (1.0 s of all-row RMSE on the holdout), the concept member 3.4 s (1.0 s on the holdout).
  Why the concept member gains more on 2026 than on the holdout, the board cannot say; it is tier F, an upper bound.
- Four candidates and the final file used the day's five uploads.

### 10.37 F1 adopted as the H1 deliverable (13 Oct)

**The owner's decision (13 Oct, about 08:15 UTC).** F1 (`op_h1_f1_plan`: H1 + the other flights' plans after the
anchor, 10.36) becomes the H1 deliverable, and whatever that requires is re-run; the deadline is 14 Oct, 10:00 UTC. F1
reads no tier F input. The plans it adds are O*, like the own runway: the other flights' last filed off-blocks and
their schedule, which can still change after the anchor.

**What carries over.** The three refits (`op_h1_f1_plan__40ac5f36__s42/s43/s44`, made with `-o
model.refit_on_all=true`; the config now sets the flag itself, which leaves its resolved configuration and its hash
unchanged), their final-set predictions and blend (`2026-10-13T08-42-02__final_op_h1f1_blend__cab07f2b`), the level
check (+17 s, correlation 0.85; 10.36) and rule L (F1 inherits the H1 deliverable's learning rate, 0.08).

**Pre-registered before the runs** (the bars of 10.35 and 10.36, unchanged):
- **Generalisation** (rule T's c4 and c5, applied to F1). F1's month pairs (`op_h1_f1_plan_febdec` to `_junaug`, seed
  42) against `op_h1_r4_nodoy`'s: better in at least 8 of the 10 validation months (OP MAE). Forward
  (`op_h1_f1_plan_fwd`, fit January-October, validate November and December): better than `op_h1_r4_nodoy_fwd`. The
  owner's decision stands either way; a failed check goes into the model card as a limit and to the owner.
- **Rule F** (the tripwire). F1's forward November and December OP MAE against F1's own pair months (November from
  `_marnov`, December from `_febdec`) plus 8 s, as for H1.
- **Rule C.** Tables fitted on F1's five pairs (OP rows), checked on the mean of F1's three seeds on January and July
  (`C_h1`; the earlier record is kept as `C_h1_r4_nodoy`): the median at least 1 s of OP MAE and 1 point within ±2 min
  better than the mean, no airport worse by more than 2 s, P10-P90 coverage within 75-85 %.
- **Rule M.** `op_h1_f1_plan_mae_exact_cpu` (H1's rule M settings on F1's features; one CPU run with its refit, seed
  42) replaces F1's calibrated median only if its holdout beats it by at least 1 s of OP MAE and 1 point within ±2 min.
  Otherwise the file carries the calibrated median.
- **Delivery.** `lab/op_deliver.py --member h1 --config op_h1_f1_plan.yaml --n 3 --tables <F1's tables> --rule-c
  [--median-config op_h1_f1_plan_mae_exact_cpu.yaml]`. The earlier H1 records stay in the manifest as `h1_r4_nodoy`
  and `h1_r4_nodoy_calibrated_median`, and their files stay in `submissions/operational/`.

The pairs and the forward run go on the GPU (queue steps `AF_*`), the median member on the CPU beside them (about 7 h
for its folds and refit, as H1's took).

**F5 delivered as well (the owner's request, 13 Oct, about 09:00 UTC).** The owner wants a file of F5
(`op_h1_f5_full`), the full concept: H1 + every input of rule T after the anchor (the plans; the realised records, NM
readings and ADS-B picture) + the ADS-B picture before it. Among the operational models it reads the most context, the
closest to the challenge model's configuration. It still lacks the challenge model's decisive inputs, the flight's own
takeoff, its own track and the join window. Its file ships beside the deliverable like h1f, tier F, mean only, and
carries a second caveat: its realised departure side holds traces of the own slot, which rule T could not credit
(10.36), so its gain is an upper bound even for perfect plans. Its configuration stays holdout-only (no configuration
with `refit_on_all` resolves a tier F block), so two seeds are refitted through `-o model.refit_on_all=true`
(`lab/op_deliver.py --member h1f5 --config op_h1_f5_full.yaml --n 2 --any-hash`). Its month pairs and forward split run
afterwards, for information only: the owner's decision stands either way.

**The final submission's name (the challenge website, relayed by the owner on 13 Oct).** The unique final submission
goes up as `<team-name>_final.parquet`, i.e. `adventurous-radio_final.parquet`, not as a numbered version.
`upload_submission.py --final` uploads only the scored file: byte-identical to `submissions/submitting.parquet`, the
final release template's 670,790 ids once each, int32 seconds without a gap; a second final upload is refused once one
is logged (`tests/test_upload_final.py`). The dry run passed on the scored file (sha1
6cdf67a406177e819e083847fab7c48febc41854) and refused a board subset. The owner uploads it; the board checks of 14 Oct
(the new H1 median and F5) take numbered versions as before.

**F1's checks (13 Oct, 08:25-12:00 UTC; `runs/op_decisions.json` A_F1 and C_h1, `runs/op_adopt_f1.md`).** The superseded
H1 records were archived first (`lab/op_adopt.py archive`: `C_h1_r4_nodoy`, `M_r4_nodoy`, and the manifest's
`h1_r4_nodoy`, `h1_r4_nodoy_calibrated_median`).
- **Generalisation: passed.** F1's month pairs beat `op_h1_r4_nodoy`'s in 10 of 10 validation months (OP MAE: Feb
  −1.17, Mar −3.00, Apr −2.98, May −2.73, Jun −2.34, Aug −1.20, Sep −2.25, Oct −2.96, Nov −2.66, Dec −3.10 s), and the
  forward split by 4.79 s (173.36 s against 178.15 s).
- **Rule F: not tripped.** F1's forward November lies +0.95 s above its own `_marnov` November, its December −2.21 s
  below its own `_febdec` December (tripwire +8 s); against H1's fixed references −1.72 and −5.31 s.
- **Rule C: passed** (tables `runs/op_calib/h1_f1_pairs_lr08.parquet`, fitted on F1's five pairs, 1,721,301 OP rows;
  checked on the mean of F1's three refitted seeds). The calibrated median scores OP MAE 180.79 s against the mean's
  184.29 s (−3.51 s; H1: −3.58 s), 49.70 % within ±2 min (+2.33 points), no airport worse (worst −0.69 s), P10-P90
  coverage 77.7 % (median width 465 s). Rule M's reference for F1 is therefore 180.79 s and 49.70 %.
- **Rule M: passed** (`runs/op_decisions.json` M; `lab/op_adopt.py rule-m`). F1's median member
  (`2026-10-13T10-24-22__op_h1_f1_plan_mae_exact_cpu__d7b90aac__s42`, CPU beside the GPU queue: folds 2.7 h, best round
  5,576; refit 3.5 h, 6,679 rounds) scores OP MAE 178.82 s, RMSE 281.64 s, 50.88 % within ±2 min, bias −39.7 s:
  against F1's calibrated median +1.97 s and +1.18 points (bars 1 s and 1 point; outside the 0.25 s seed floor).
  H1's member had +1.94 s and +1.06 points.

**F1 delivered as the H1 deliverable (13 Oct, 11:54 and 14:41 UTC).** `lab/op_deliver.py --member h1 --config
op_h1_f1_plan.yaml --n 3 --tables runs/op_calib/h1_f1_pairs_lr08.parquet --rule-c`, then the same with
`--median-config op_h1_f1_plan_mae_exact_cpu.yaml`:
- `submissions/operational/h1__2026-10-13T13-54-07__final_op_h1_blend__aa1cd03b__mae.parquet`, sha1
  4153b4ea3196de18211df8d6f492ffd0c0c7adb4, 670,790 rows: the mean of the three seeds, the P10-P90 band and the median
  member's prediction (0.79 % clamped into the band). The manifest's `h1`.
- `h1__2026-10-13T13-54-07__final_op_h1_blend__aa1cd03b.parquet` (sha1 f7f4f2ac87f3...), the file before rule M:
  `h1_calibrated_median`.
- The blend's mean equals the board-checked F1 blend (v82, 559.77) on every row; the level check is the same (+17.4 s
  on the 23 reliable airport-months, −1 to +83 s, correlation 0.85, LIRF July flagged).
- The superseded H1 files stay in the folder under `h1_r4_nodoy` and `h1_r4_nodoy_calibrated_median`; the scored file
  is unchanged (sha1 6cdf67a40617).

**F5 delivered (13 Oct, 15:23 UTC).** Two seeds refitted through the override in 192 min of GPU shared with the CPU
median member (`2026-10-13T13-51-55__op_h1_f5_full__4fae29d6__s42`, `2026-10-13T15-42-58__op_h1_f5_full__4fae29d6__s43`;
OP MAE 177.84 and 177.90 s), delivered by `lab/op_deliver.py --member h1f5 --config op_h1_f5_full.yaml --n 2 --any-hash`
as `submissions/operational/h1f5__2026-10-13T17-23-14__final_op_h1f5_blend__f7393913.parquet` (sha1
bac06dfdc0bf8399f17eee6baf9425edb3e6fceb, 670,790 rows, the mean only; the manifest's `h1f5` names tier F and the
own-slot caveat).
- The mean of the two seeds on the Jan+Jul 2025 holdout: OP MAE 177.43 s, OP RMSE 266.06 s, 48.93 % within ±2 min,
  84.48 % within ±5 min; all rows: RMSE 578.69 s. That is 6.86 s below the H1 deliverable F1 (184.29 s), 0.8 s below
  H0, which knows the flight's own off-block (178.22 s), and 40.3 s above the challenge model's member C1 (137.1 s).
- The level check on the final set: +12.1 s on the 23 reliable airport-months (−9 to +60 s), correlation 0.887, LIRF
  July flagged (+60 s).
- Its month pairs and forward split run for information from 15:23 UTC.
- **F5's month pairs and forward split** (for information; queue `AF5_*`, 15:23-18:39 UTC; overrides of
  `op_h1_f5_full.yaml`, seed 42): F5 beats F1 in 10 of 10 validation months (OP MAE −7.06 to −9.83 s) and the
  superseded H1 in 10 of 10 (−9.73 to −12.68 s). Forward (fit January-October, validate November and December): F5
  164.95 s, F1 173.36 s, H1 178.15 s. The gain holds out of season and forward, but it carries the same caveat as on the
  Jan+Jul holdout: F5's realised departure side holds traces of the own slot, which rule T could not credit (10.36).

**The board checks of 14 Oct (00:02 UTC, after the daily limit's reset).** v84, the January and July rows of the H1
deliverable's median (F1's median member, clamped into the band), scored 573.55 (holdout all-row RMSE 594.3; the
superseded H1's median, v79, 575.06). v85, F5's mean (tier F), scored 551.52 (holdout 578.7; F1's mean 559.77, v82;
the concept member h1f 557.53, v80): its rejected upload of 13 Oct went up again under the same name. On the board F5
gains 8.3 s on F1, against 4.8 s of all-row RMSE on the holdout; like F5's other gains it includes the own slot's
traces (10.36). Two of the day's five uploads; three remain for the owner's final upload, `adventurous-radio_final.parquet`.

**The final upload (14 Oct, 07:15 UTC).** The owner uploaded the scored file as `adventurous-radio_final.parquet`
(`upload_submission.py --run 2026-10-11T03-22-21__final_v70_5c8bf3__5c8bf33e --final`; sha1
6cdf67a406177e819e083847fab7c48febc41854, 4,080,955 bytes). The scorer answered at 07:15:47 UTC: status Succeeded,
670,790 used pairs (every departure of the final set), no score. It was the third of the day's uploads, after v84 and
v85. The tracked copy of the upload log, `docs/board_history.csv` (rewritten by `notebooks/build/build_08.py`), holds
it as family `final-file` without a version; `notebooks/build/build_06.py` now keeps only the numbered uploads, since it
read every scored row's version.

### 10.38 The scored model framed as a post-operations analysis model (owner, 14 Oct)

**The decision.** On 14 Oct the owner reframed the scored model, called the challenge model until then, as the
post-operations analysis model, after the challenge's own rationale. The Rationale page
(<https://prc-data-challenge-2026.netlify.app/rationale.html>, accessed 14 October 2026) says: "A model to estimate
taxi-out time can be used in post-operations analysis to identify constrained operations intervals and measure excess
fuel burnt/CO2 production compared to unconstrained status." Taxi-out time is of interest there because "it is often
difficult to obtain and challenging to predict accurately", and "The reasons for variability of taxi-out time are
various and interlinked. These include airline-specific constraints, airport procedures and load, and Air Traffic Flow
Management (ATFM)-related reasons." The organisers' line relayed by another team, "the model is for post-ops, not for
tactical use" (docs/data_sources.md section 4), points the same way; the page itself does not mention tactical use.

**The argument (the owner's).** After the flight, the data recorded about it exist: the actual takeoff time, NM's
flight data (and, where a data set joins NM's flight list to the airport's movements with a tolerance, the window that
join leaves), the flight's own ADS-B track, reanalyses and EUROCONTROL's published data. The scored model's
after-the-fact inputs (tiers T, A, M and P) are therefore legitimate for post-operations analysis: estimating the
taxi-out where it is missing or hard to obtain. The operational models (10.27, 10.35-10.37) remain the planning models:
they predict before the movement, from what is known 24 h, 3 h or 15 min before the planned off-block. Both are useful,
for different purposes: the post-operations model serves the use the rationale names, the operational models serve
planning, which the post-operations model cannot.

**The limits of the framing.**
1. Tier C stays a property of how the challenge's records were made. The stale records (the schedule logged as the
   off-block, 2.2) and the wrong-date records (2.4) are recording defects; a post-operations analysis would flag them
   rather than reproduce them. The regime mixture does this: its stale probability (`p_stale`) flags the records whose
   recorded taxi-out is the schedule's artefact, and its regressor estimates the taxi time (`y_reg`). Both are in the
   mixture members' run records (`oof_predictions.parquet`, `refit_detail.parquet`).
2. The join window (10.3): after the flight, an analyst holding NM's flight data and the takeoff time has the same
   bound wherever the data set is joined with that tolerance. It remains specific to how the data set was assembled
   and says nothing before the flight.
3. No analysis in this work used the model to identify constrained intervals or to measure fuel or CO2. The model is
   built for that use; it has not been applied to it.

**Documents changed (framing and terms only).** README.md, docs/report.md, the fig08 labels
(`notebooks/build/build_08.py`), QUICKSTART.md, DEVELOPER_GUIDE.md, docs/operational_model.md, docs/paper.md (a
sentence in the Abstract, a paragraph in section 1; the paper keeps its terms and declares the alias once),
docs/data_sources.md (the Rationale page as a first-hand source in section 4) and CLAUDE.md. In reader-facing text
the scored model is "the post-operations analysis model" at its first mention, then "the post-operations model". The
code, the configurations, the run, config, file and column names, the `board_history.csv` families, this file and the
other supporting documents keep "challenge model". Nothing was rerun; every number, run id, score, version and file
name is unchanged.

**The data findings of report.md 1.4 (14 Oct; the owner's choice: a short table, no new figure).** Two paragraphs and
Table 6 now open section 1.4, and one sentence each sits in the report's Summary and the README's main findings.
- The records' defects in 2025: 584 targets above 2 h, 438 of them with the stale fingerprint (target = takeoff -
  schedule within 120 s), 480 at Rome; 1,507 departures without an aircraft type and without an NM match (1,487 at
  Rome), 711 of them with a target above 10 min that equals takeoff - schedule; 15 wrong-date records (13 at Rome). On
  v70's holdout the 981 targets above 1 h (0.28 % of the rows) carry 59.6 % of the squared error, the worst 0.1 % of
  the rows 62.9 %, the seven wrong-date rows 36.0 %, the 26 targets above 20,000 s 47.3 %.
- Each airport's strongest day of 2025 (local day; mean and share above 30 min over the departures with a target below
  2 h; days with at least half the airport's median number of departures, which drops the first local hours of
  1 January 2026 at Paris; against the airport's median day): Istanbul 23 Feb +1,822 s (x2.8; 65.6 % above 30 min
  against 2.8 %), Munich 24 Dec +1,164 s, Rome 13 Jul +1,093 s, Paris 17 Jan +966 s, Frankfurt 5 Jan +410 s, Heathrow
  31 Dec +389 s, Amsterdam 5 Jan +358 s, Zürich 4 Jan +347 s, Barcelona 9 Oct +328 s, Madrid 2 May +176 s. Snow or
  freezing fog on seven, thunderstorms on three (Madrid's from the METAR: +TSRA at 05:51 and 15:53 UTC; the others from
  docs/snow/). Istanbul's February snow holds five of the ten strongest airport-days; 15 of the 3,644 airport-days
  rose more than 10 min above their airport's median day, 53 more than 5 min.
- More than half of the departures above 30 min on five days at three airports (Istanbul 19, 22 and 23 February,
  Munich 24 December, Rome 13 July) over the departures below 2 h; Paris's 17 January reaches 49.3 % so, and 50.2 % over
  all departures, which is why the figure index counts six days at four airports.
- `lab/report_data_findings.py` recomputes every number from data/interim and v70's run record.

**The documentation moved into docs/ (14 Oct; the owner's choice of one folder).** `report/` became `docs/`, and
`DEVELOPER_GUIDE.md`, `QUICKSTART.md` and `findings.md` moved into it; `README.md`, `CLAUDE.md` and
`LICENSE` stay at the root. Every markdown link was re-targeted from its file's new location (141 relative links, none
broken), and the path mentions `report/...` became `docs/...` in 65 tracked files: the documents, the lab scripts and
notebook builders that write there (their output paths included), the notebooks' text, the tests and the data files.
`.gitignore` keeps the literature PDFs out (`docs/literature_pdfs/`). One mention stays: the docstring of
`prc/features/operational.py`, a hashed feature module, whose edit would re-key the delivered refits (developer guide,
cache keys). No feature hash, configuration or number changed.
