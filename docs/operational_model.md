# The operational taxi-out model: model card

**Prediction moment.** This model predicts how long a departing aircraft will take from its off-block to its takeoff.
It predicts before the aircraft moves, using only data that exist at that moment. It is the repository's answer to
the owner's goal of 11 October 2026: a taxi-out model that is useful in operations, with every input documented
together with its operational rationale.

**The H1 deliverable since 13 October.** On 13 October the owner adopted F1 (`op_h1_f1_plan`) as the H1 deliverable:
the H1 model until then (`op_h1_r4_nodoy`) plus the other flights' plans for the next hour (section 3). F1 reads no
tier F input, and nothing from after the flight's own off-block. The superseded H1 stays in this card, as history and
as the comparison that justified the adoption: 2.39 s less OP MAE on the holdout, better in 10 of 10 validation
months, and 4.79 s better on the forward check (findings 10.37).

**Scope.** The models of this card are the planning models. The scored challenge file comes from a different model,
the post-operations analysis model (the challenge model in the code, the configurations and the supporting documents):
it estimates a flight's taxi-out after the flight, from what was recorded about it (`docs/report.md` Part 1). It is
a post-operations model, not a planning model, and its file alone is the final submission (section 6). Two concept
files (section 3b) read the other flights' realised movements after the moment, as a stand-in for the plans of an
airport's planning systems. `h1f` reads their realised arrivals (an arrival manager's plan), F5 (`h1f5`) all of their
movements. That is tier F: planned in the concept, realised in the challenge data. Both files are upper bounds, not
available as computed. F5's is one even for perfect plans, because its realised departure side carries traces of the
flight's own slot. They sit beside the deliverable, never instead of it.

**Companion documents.**
- [`docs/operational_features.md`](operational_features.md): every data point and model input, with its
  definition, source, availability, tier, rationale and evidence.
- [`docs/operational_ladder.md`](operational_ladder.md): every run's holdout read-out and the paired steps.
- [`findings.md`](findings.md) 10.27, 10.31, 10.34, 10.35, 10.36 and 10.37: the lab notebook.

## 1. What it predicts, and when

The target is the departure taxi-out time in seconds: from the recorded off-block (AOBT) to the takeoff (ATOT), for
IFR departures at ten European airports. The airports are EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH and
LTFM.

The model family has one deliverable member per decision moment, plus two tier F concept members at H1 (section 3b).
Each member is a single global CatBoost model over the ten airports, fitted on 2025:

| member | predicts at | who would use it | inputs, in short |
|---|---|---|---|
| **24 h ahead** (`op_hl24_r3_rwyblind`) | the published schedule (SOBT) minus 24 h | day-ahead planning: stand and resource plans, the day's expected taxi load | the flight's schedule and plan record, its stand, the airport layout, the METAR then and the TAF for the scheduled hour, the scheduled demand around the flight's slot; no runway, and no NM history (rule R: a day-old history adds nothing) |
| **3 h ahead** (`op_hl3_r4_rwyblind`) | SOBT minus 3 h | same-day (tactical) planning, before the flight plan's last updates | as 24 h ahead, read 3 h before the schedule, plus NM's history of the flights that departed in the 3 h and 24 h before (rule R) |
| **H1, the headline** (`op_h1_f1_plan`, F1, since 13 Oct; until then `op_h1_r4_nodoy`) | 15 min before the last filed off-block time (IOBT, else LOBT, else SOBT); the median flight pushes 19.2 min later | the variable taxi time an A-CDM airport adds to the target off-block time to get a target takeoff time | the flight's plan and record, its stand and planned runway, the airport layout, the METAR and the TAF at the moment, the other flights' takeoffs, landings and taxi-ins up to the moment, NM's readings of the flights that departed in the last 30-60 min, the backlog of flights planned off but not yet airborne, and (since 13 Oct) the other flights' plans for the next 15, 30 and 60 min: departures by their last filed off-block, arrivals by their schedule |
| **H1 + planning data** (concept, tier F; `op_h1_f4_nodepthru`, file `h1f`, section 3b) | as H1 | what an airport with an arrival manager would add: an upper bound for perfect plans, not available as computed; beside the deliverable, never instead of it | the deliverable's inputs (the other flights' plans after the anchor included, O*), plus their realised arrivals after it (F, standing in for the arrival manager's plan) |
| **F5, the full concept** (tier F; `op_h1_f5_full`, file `h1f5`, section 3b; the owner's request of 13 Oct) | as H1 | the most context among the operational models; an upper bound even for perfect plans, because its realised departure side carries traces of the own slot; beside the deliverable, never instead of it | the deliverable's inputs, plus the other flights' realised records, NM readings and ADS-B surface picture after the anchor (F), and the ADS-B picture before it |
| **H0** (comparison only; `op_h0_r4_nodoy`) | the recorded off-block itself | the taxi time at the push | the inputs of the H1 deliverable until 13 Oct at the push (without F1's plans); evaluated on 2025 only, because 2026 blanks every off-block |

**Why these moments.** Tactical surface management during the taxi is well researched: A-SMGCS, taxi trajectories
and the aircraft's own track. What is less known is how precise an estimate can be *before* the information it needs
exists. The members answer that at three distances from the movement.

**What the outputs are.** The mean, the RMSE-optimal point. At H1 also a median, which is better for absolute error
and for the share within a tolerance, and an 80 % band (P10-P90).
- **The band** comes from calibration tables fitted on the 2025 month pairs (rule C).
- **The median** first came from the same tables. Rule M then replaced it by a second model fitted on the absolute
  error (`op_h1_f1_plan_mae_exact_cpu`; until 13 Oct `op_h1_r4_nodoy_mae_exact_cpu`). Its prediction is clamped into
  the band, so that the file's median never leaves its own band. The clamp was not part of the pre-registered rule.
  For the superseded H1 it moved 0.5 % of the OP rows (0.7 % of all holdout rows) and changed the MAE by −0.04 s. In
  F1's file it moves the median on 0.79 % of the final set's rows.
- **At 3 h and 24 h** rule C failed: the calibrated median would have improved nine airports but made Istanbul 2.2 s
  worse, beyond the pre-registered 2 s. Those files carry the mean only.

## 2. Results

**How they are measured.**
- **Holdout.** The seasonal holdout, January and July 2025. Each model is fitted on the other ten months.
- **Population (OP).** NM-matched departures with a target below one hour: 338,277 rows, 98.2 % of the 344,419
  Jan+Jul 2025 holdout departures. The rest hold the challenge's stale and wrong-date block times (section 5).
- **Spread and uncertainty.** The seed spread is reported. The paired steps carry 95 % day-block bootstrap intervals.

Every figure below, except the two concept rows, is from `docs/operational_ladder.md` (with the paired steps and
their intervals). The figures `docs/figures/fig07_01_moments.png`, `fig07_02_airports.png` and
`fig07_03_band_coverage.png` (notebook 07) draw the deliverables' rows, F1 at H1, beside H0; the first of them also
draws the stand × runway table as a reference line. The superseded H1's rows and the tier F rows are not drawn. The
concept rows score the mean of their two refits' holdout predictions as the ladder scores the deliverables' seed means
(findings 10.36 and 10.37). The forward figure of `h1f` is the seed-42 forward run of rule T's stage 2 (findings
10.36). F5's month pairs and forward split were started after its delivery, for information only, so its row has no
forward figure.

| member | output | OP MAE | OP RMSE | within ±2 min | within ±5 min | P10-P90 coverage | forward check (Nov+Dec) MAE |
|---|---|---|---|---|---|---|---|
| 24 h | mean (shipped) | 218.2 | 321.6 | 40.1 % | 77.7 % | | 214.8 |
| 24 h | median (not shipped: rule C, Istanbul +2.2 s) | 215.4 | 327.5 | 42.6 % | 78.5 % | 77.7 % | |
| 3 h | mean (shipped) | 214.5 | 312.8 | 40.2 % | 78.0 % | | 208.2 |
| 3 h | median (not shipped: rule C, Istanbul +2.2 s) | 211.6 | 318.5 | 42.7 % | 78.9 % | 77.7 % | |
| H1 = F1 (`op_h1_f1_plan`, three seeds) | mean (shipped) | 184.3 | 276.3 | 47.4 % | 83.4 % | | 173.4 (seed 42) |
| H1 = F1 | calibrated median (rule C; replaced by rule M) | 180.8 | 278.8 | 49.7 % | 84.0 % | 77.7 % | |
| H1 = F1 | median member, clamped into the band (shipped; rule M) | **178.8** | 281.1 | **50.9 %** | 84.6 % | 77.7 % | |
| H1 until 13 Oct (superseded; `op_h1_r4_nodoy`, three seeds) | mean (shipped until 13 Oct) | 186.7 | 278.7 | 46.6 % | 83.0 % | | 178.1 (Nov 176.6, Dec 179.6) |
| H1 until 13 Oct | calibrated median (rule C; replaced by rule M) | 183.1 | 281.4 | 49.2 % | 83.6 % | 77.7 % | |
| H1 until 13 Oct | median member, clamped into the band (shipped until 13 Oct; rule M) | 181.1 | 283.6 | 50.2 % | 84.2 % | 77.7 % | |
| H0 (2025 only; `op_h0_r4_nodoy`, seed 42) | mean | 178.2 | 265.4 | 48.5 % | 84.3 % | | |
| H1 + planning data (concept, tier F, section 3b; `op_h1_f4_nodepthru`, two seeds) | mean (shipped beside the deliverable) | 183.6 | 275.4 | 47.6 % | 83.5 % | | 172.6 (seed 42) |
| F5, the full concept (tier F, section 3b; `op_h1_f5_full`, two seeds) | mean (shipped beside the deliverable) | 177.4 | 266.1 | 48.9 % | 84.5 % | | |
| stand × runway table (the A-CDM variable-taxi-time practice) | median table | 207.4 | 330.2 | 45.5 % | 81.0 % | | |

**The adoption step.** F1 against the superseded H1, paired on the same rows (95 % day-block bootstrap over 62 days):
−2.39 s of OP MAE [−2.73, −2.06] and +0.75 points within ±2 min [+0.58, +0.91]. On the forward check F1 scores
173.36 s against 178.15 s (−4.79 s).

**Comparison with published practice.** None of these use our protocol (`docs/operational_ladder.md`, literature
section).
- **A-CDM manual** (EUROCONTROL 2017): an accuracy target of ±2 min for taxi times; the table is marked tentative.
- **Frankfurt** (Günther 2011, from the abstract): static stand-runway tables place 42 % of departures within 2 min,
  and a traffic-aware calculation at the off-block 66 %.
- **Six A-CDM airports** (Swatowska 2025): a model with post-operational runway and stand reached a departure MAE of
  2:18 min at Zurich and 3:32 min at Istanbul.

## 3. What each source of information is worth

The ladder adds one source at a time (`docs/operational_ladder.md`; findings 10.27).

**At H1** (every rung without the day of year; findings 10.35 and 10.37):
- **The flight's record.** Its record, stand, runway and calendar alone reach 196.0 s, 11 s better than the stand ×
  runway table.
- **What else is known at the moment**, up to the H1 model until 13 Oct (rung r4), adds 8.4 s of MAE and 23 s of
  RMSE.
  - NM's data of the departed flights and the plan backlog: −3.3 s.
  - Weather at the moment: −2.5 s of MAE but −13 s of RMSE. It decides the tail and the disrupted winter days, not
    the typical flight.
  - The other flights' movement records: −1.9 s.
  - The layout: −0.7 s.
- **The other flights' plans for the next hour** (F1, the deliverable since 13 Oct): −2.39 s more [−2.73, −2.06] and
  +0.75 points within ±2 min, paired on the same rows; 184.3 s against 186.7 s on the three seeds' means. They are 15
  columns for the next 15, 30 and 60 min: the other departures by their last filed off-block, else the schedule, per
  airport, own runway and own pier (O*), and the arrivals scheduled at the own pier (O*) and at the airport (O).
- **The planned runway** is worth 17.4 s, measured on the superseded H1: runway-blind it scores 205.0 s (seed 42).
- **A caveat.** The first ladder (findings 10.27) carried the day of year in every rung. There it faked day-level
  conditions in the rungs without weather, and weather looked like a −14 s step.

**Hours ahead** (the ladder on the schedule clock, without the day of year):
- **The runway.** Without the planned runway, which is unknown hours ahead, the models lose about 22 s: 191.2 against
  212.8 s at 3 h, 196.7 against 218.6 s at 24 h.
- **What the runway-aware rungs show.** With the runway, three hours ahead reaches 191.2 s, within 4 s of rung r4 at
  H1 (187.6 s, seed 42: the H1 model until 13 Oct, without F1's plans). The information of the moment is worth only
  that much.
- **Forecast weather.** With the layout it adds 3.2 s at 3 h and nothing a day ahead.
- **Scheduled demand** around the slot adds 2-2.5 s.
- **NM's history** of the flights that departed in the hours before helps 3 h ahead, by 1.1 s on the forward check
  (rule R), and not a day ahead.
- **The filed plan.** It carried nothing usable beyond the schedule: on the IOBT clock the models scored no better.

## 3b. Traffic after the anchor: what planning data would add (the concept, findings 10.36)

**The question** (the owner's concept, 12 Oct). An operational model should know the traffic situation around the
planned off-block on both sides of the moment it predicts: the actual traffic before it, and the planned traffic after
it. An A-CDM airport's planning systems hold the second part:
- the arrival manager (AMAN): the landing sequence;
- the departure manager (DMAN): the start-up and takeoff sequence (TSAT, TTOT) and the runway plan;
- the surface manager (A-SMGCS planning): the pushbacks, routes, taxi times and holding.

The challenge data hold none of these plans, only what happened. Until 13 Oct the H1 deliverable therefore read the
traffic before the anchor only. Since then (F1, section 3) it also reads the other flights' plans after the anchor
where the data hold them (O*), but none of their realised movements.

**What was tested** (`prc/features/operational_future.py`; every input in `docs/operational_features.md`, Part D):
- the other flights' **plans** after the anchor where the data hold them: their last filed off-blocks and the arrivals'
  schedule in the next 15/30/60 min (tier O*);
- their **realised movements** after the anchor, read at the planned off-block and 15 and 30 min after it: the
  airport's records, NM's readings of the flights taking off, the public ADS-B surface picture. These stand in for the
  planning systems' forecasts, so they form a new tier, **F**: *planned in the concept, realised in the challenge
  data*. An F input is not available as computed; what it gains is an upper bound for perfect plans.

**The own movement stays out.** The own takeoff leaves every count. After the anchor the ADS-B identity rules of H1 no
longer held: the own taxi-out often carries the inbound leg's callsign. The own airframe is therefore identified
through its own ADS-B takeoff and its whole ground stay removed, which brought the share of contaminated departures at
H1 + 30 min from 10-17 % to 1 in about 7,400. Columns that would locate the own slot are left out. Traces through the
shared resources remain: the own takeoff takes a runway slot from the others' counts, the next arrival at the own stand
waits for the own push, and pushback lanes, de-icing pads and the holding queue are shared. Rule T removes the
departure-side traces by construction, through a variant without any realised departure-side column. That variant is
rule T's member, delivered as `h1f`, so nothing from the departure side is credited; the next arrival at the own stand
stays in it, unmeasured. F5 (`h1f5`), delivered as well at the owner's request (below), reads the realised departure
side: its gain is an upper bound even for perfect plans.

**Rule T** (pre-registered before any run, amended before any run after two adversarial reviews; findings 10.36). Seed
42, the January and July 2025 holdout from 9 January on, OP population, paired against the configuration of the H1
deliverable until 13 Oct (`op_h1_r4_nodoy`):

| run | adds after the anchor | OP MAE | change [95 % CI] | within ±2 min |
|---|---|---|---|---|
| H1 deliverable until 13 Oct | nothing | 185.3 | | 46.6 % |
| F1 (the H1 deliverable since 13 Oct) | the other flights' plans (O*) | 182.7 | −2.6 [−3.1, −2.2] | 47.6 % |
| F4-nodepthru | + the realised arrivals (F) | 181.3 | −4.0 [−4.4, −3.5] | 48.0 % |
| F4 | + the realised departure side: records, NM, ADS-B (F) | 175.1 | −10.2 [−11.2, −9.4] | 49.3 % |
| F5 | F4 + the ADS-B picture before the anchor | 174.9 | −10.4 [−11.4, −9.6] | 49.4 % |
| placebo | F4's F inputs read a week earlier | 183.6 | −1.7 [−2.2, −1.1] | 47.3 % |
| H0 (lab) | the flight's own off-block known | 175.5 | | 48.9 % |

**What it shows.**
- **Plans that exist today are worth 2.6 s** (O*), at every airport (1.6-4.4 s). Rule T recorded them as a candidate
  for the next version of the strict deliverable, and F1's three seeds were refitted on 13 Oct as an option (findings
  10.36). The owner adopted F1 the same day: it is the H1 deliverable since then (findings 10.37). Its three seeds
  score 184.3 s of OP MAE on the full holdout against the superseded H1's 186.7 s (−2.39 s [−2.73, −2.06], paired),
  and 559.8 on the live board against 561.0.
- **Perfect arrival planning adds 1.3 s** (4.0 s with the plans). This part is free of the departure-side traces by
  construction. Two traces of the own movement remain in it: the next arrival at the own stand (unmeasured), and the
  missingness of the ADS-B taxi-in columns, which depends on finding the own ADS-B takeoff (about 0.15 s of MAE for F4,
  findings 10.36). On a mixed-mode runway the landings on the own runway may also carry a trace of the own slot
  (unmeasured).
- **The realised departure side adds 6.3 s more, but cannot be credited to planning data.** It gains most at EGLL,
  LTFM, LIRF and LEBL (8-12 s from the departure side, against 2-5 s elsewhere; 11-16 s for F4 as a whole; findings
  10.36). At EGLL, LEBL and LIRF one runway carries 95-100 % of each hour's departures, so the own slot leaves the
  strongest trace in the others' counts; LTFM (65 %) is not explained by that. With it, H1 reaches the accuracy of
  knowing the own off-block (H0).
- **The placebo** keeps less than the plans alone: the gain is the traffic of the day, not the airport or the flight.
- **The gap to the post-operations models is the flight's own movement, not the others' traffic.** Knowing everything
  the other flights did in the next 45 min closes at most 10 s of the 50-80 s between H1 and the post-operations
  models (C1 137 s; with the flight's own ADS-B track, 104 s).

**Stage 2 and the member.** The member is the variant without the realised departure side, `op_h1_f4_nodepthru`: the
H1 model until 13 Oct plus the other flights' plans and their realised arrivals after the anchor, that is, F1 plus the
realised arrivals. On the 2025 month pairs it beat the superseded H1 in all 10 validation months (by 1.3-5.1 s), and
on the forward split by 5.5 s (172.6 against 178.1 s). It is refitted with two seeds and applied to the final set as a
separate file beside the deliverable (`h1f`, section 6). On the full holdout its two refits score 183.6 s against the
deliverable F1's 184.3 s. It shows what an airport with a perfect arrival manager would gain on top of the plans it
already holds, not what the strict deliverable can do today.

**F5 delivered as well (13 Oct, the owner's request; findings 10.37).** F5 (`op_h1_f5_full`) is the full concept: the
deliverable's inputs, plus every realised movement of the other flights after the anchor (their records, NM's
readings and the ADS-B surface picture) and the ADS-B picture before it. It is not rule T's member, because rule T
could not credit its realised departure side. The owner asked for its file anyway, as a second tier F file beside the
deliverable (`h1f5`, mean only).
- **What it is.** Among the operational models it reads the most context. It still lacks the post-operations
  model's decisive inputs: the flight's own takeoff, its own ADS-B track and the NM join window.
- **What it is not.** Its realised departure side carries traces of the own slot, which rule T could not credit
  (above). Its gain is therefore an upper bound even for perfect plans, not the value of a planning system.
- **Its figures** (the mean of two refitted seeds, the January and July 2025 holdout): OP MAE 177.43 s, OP RMSE
  266.06 s, 48.93 % within ±2 min, 84.48 % within ±5 min; RMSE over all rows 578.69 s. That is 6.86 s below the
  deliverable F1 (184.29 s), 0.8 s below H0, which knows the flight's own off-block (178.22 s), and 40.3 s above the
  post-operations model's member C1 (137.1 s). In rule T, seed 42 on its rows from 9 January, F5 scored 174.86 s
  (−10.44 s).
- **How it was made.** No configuration with `refit_on_all` resolves a tier F block, so its configuration stays
  holdout-only and its two seeds were refitted through `-o model.refit_on_all=true`. Its month pairs and forward split
  were started after the delivery, for information only: the owner's decision stands either way.

The rule and the figures above are in findings 10.36 (rule T, with the departure side's per-airport gains:
F4-nodepthru's OP MAE minus F4's on the same seed-42 runs and rows) and 10.37 (F1's adoption, F5). The adoption step's
interval is the ladder's (`docs/operational_ladder.md`, D-H1-R4 to D-H1). The lab scripts' decision records (keys T13
and T) are not published; findings 10.36 transcribes them. `docs/operational_ladder.md` has the paired steps per
airport, but scored on every OP row from 1 January; rule T's own figures start on 9 January, so the two differ
slightly. F1's configuration hash moved with its refit flag (13468ab0 to 40ac5f36), so the ladder's rule T table
reads F1 at seed 42 of its three refits (184.89 s on every OP row; rule T's holdout-only run: 184.92 s); rule T's own
record is unchanged.
`docs/operational_features.md` (Part D) has every input with its planning-system source and its stand-in.

## 4. The information contract

What every member obeys. The leakage gates in `tests/test_operational.py` and `tests/test_operational_long.py` enforce
it, and for the plans F1 adds, `tests/test_operational_future.py`:
- **Never the flight's own outcome.** The model never reads the flight's own takeoff, its own off-block record, or its
  own NM actual times (AOBT_3, ARVT_3, EOBT_1). Only H0 reads the off-block, and only as its anchor.
- **Other flights up to the anchor.** Other flights' events count only when they happened at or before the anchor.
  The tests rebuild each block with every later event removed and require identical values. Plans are not events:
  since 13 Oct H1 also reads the other flights' plans for the next hour (their last filed off-blocks and the
  schedule), as the airport holds them at the anchor (O*), but none of their realised movements after it.
- **Weather as published.** A METAR counts from 5 min after its observation. A TAF counts only when issued at or
  before the anchor.
- **The schedule as the long clock.** The long members run on the published schedule. No long member reads any
  flight's IOBT; the tests shift, blank and remove it and require identical values.
- **Statistics from fit rows only.** Statistics that read targets or NM readings come from the fit rows only, and
  leave out the query's own month. Target-free tables (the arrivals' taxi-in medians) are built over the whole frame:
  transductive in time, with no departure target.

The concept members of section 3b (`h1f` and F5) break the second rule on purpose: they read the other flights'
realised movements after the anchor as a stand-in for plans (tier F). They are never the deliverable, and their own
gates are in `tests/test_operational_future.py` (the own flight and airframe never enter; nothing after the last read
time; a tier F block in no refit configuration).

**Tiers** (per column and moment in the catalogue):
- **O**: an airport or NM holds the value at the moment.
- **O***: available under an assumption the data cannot check:
  - the planned departure runway is assumed to be the one recorded (H1);
  - hours ahead, the stand plan is assumed final;
  - at H1, the last filed off-block time is assumed known 15 min ahead;
  - at H1, the other flights' last filed off-blocks are assumed known at the anchor (F1's plans; they can still change
    after it);
  - other flights' recorded runway and stand stand in for their planned ones.
- **C**: an artefact of the challenge data, the NM-join level of the flight-plan columns. It is constant on the
  population the results are read on.
- **F**: planned in the concept, realised in the challenge data (section 3b): the other flights' realised movements
  after the anchor, standing in for the AMAN, DMAN and A-SMGCS plans the data lack. An upper bound for perfect plans,
  not available as computed; beside the deliverable, never instead of it.

## 5. Applicability and limits

- **Population.** The results hold for NM-joined departures: 98.9 % of 2025 and 98.5 % of the January and July 2026
  departures. The rest carry the challenge's stale and wrong-date block times. The models still predict them, but
  nothing is claimed there.
- **The runway at H1 is an assumption (O*).** The data hold the runway used. It equals the dominant departure runway
  of the hour before the planned off-block on 66.3 % of departures, and is in the set of runways in use on 93.1 %.
  The runway-blind comparator bounds the cost.
- **The plan at H1 is an assumption (O*).** The anchor and the planned delay use the last filed off-block time, whose
  update time is not in the data. F1's plans of the other flights rest on the same assumption: their last filed
  off-blocks are taken as known at the anchor, and they can still change after it. The no-plan sensitivity, measured
  on the superseded H1, bounds the plan columns' share: without the planned delay and the plan backlog it scores
  189.8 s (seed 42; 187.6 s with them), 2.2 s worse and still 17.6 s better than the stand × runway table. The anchor
  itself still rests on the plan, which is H1's definition.
- **Transductive tables.** The arrivals' taxi-in medians and the stand, route and geometry tables are built over the
  whole data frame, 2026 included. They read no departure target, but they are not strictly causal in time. The
  forward check (fit January-October, validate November and December) shows the cost. For F1 the forward months lie
  +0.95 s (November) and −2.21 s (December) from its own month pairs, which also saw the months after them (the
  superseded H1: +3.2 s and −0.6 s). Rule F, a tripwire at +8 s, did not trip.
- **The adoption generalises.** Checked as pre-registered before the runs (findings 10.37): F1's month pairs beat the
  superseded H1's in 10 of 10 validation months (by 1.17-3.10 s of OP MAE), and the forward split by 4.79 s (173.36
  against 178.15 s).
- **Early stopping.** Each model stops early on its own holdout, which makes the holdout figures mildly optimistic.
  The forward check and the month pairs are read the same way.
- **No cancellations.** Which flights operate is taken from the record. The scheduled demand of the long members
  therefore carries a mild hindsight.
- **TAF licence.** The TAF archive is Ogimet's, which has no open licence. In operations the TAF comes from the met
  service. At H1 the model does not need it: the superseded H1 without its 13 TAF columns scored 187.3 s, inside its
  three seeds' range (186.7-187.6 s), because the METAR of the moment carries the weather. F1 reads the same weather
  and was not re-tested. Hours ahead the TAF is the only weather for the scheduled hour; there the dependency was not
  measured.
- **Seeds and GPU.** GPU training is not bit-reproducible. The deliverables average three (H1) or two (3 h, 24 h)
  seeds. The seed spread is in `docs/operational_ladder.md`.
- **Out of time.** The final set's February and June have no analogue in the Jan+Jul holdout. Their 2025 analogues
  come from the month pairs, and the predicted monthly means are checked against the PRU's published monthly mean
  taxi-out (section 6).

## 6. The deliverables and the concept files

Predictions for the final set (January, February, June and July 2026; 670,790 departures, the organisers' final
template). Each file is `submissions/operational/<member>__<run>.parquet` with `MVT_ID_mvt`, `y_mean` and, where rule
C passed, `y_med`, `y_p10`, `y_p90`, in integer seconds. At H1 `y_med` is rule M's median member (file suffix `__mae`).
`lab/op_deliver.py` writes the files and `submissions/operational/manifest.json`. `submissions/` is local and
gitignored, so a clone holds neither. [`docs/operational_manifest.json`](operational_manifest.json) records every
delivered file with its full sha1, and QUICKSTART path B regenerates the files with `lab/op_deliver.py`. These files
are not scored; the scored file is the post-operations model's, and it is the unique final submission (end of this
section).

| member | configuration and refitted seeds | file in `submissions/operational/` | sha1 | outputs | level check against the PRU (23 reliable airport-months) |
|---|---|---|---|---|---|
| H1 (F1, since 13 Oct) | `op_h1_f1_plan`, seeds 42-44 (`..._40ac5f36__s42/s43/s44`); the median member `op_h1_f1_plan_mae_exact_cpu`, seed 42 (`..._d7b90aac__s42`) | `h1__2026-10-13T13-54-07__final_op_h1_blend__aa1cd03b__mae.parquet` | 4153b4ea3196... | mean, median (rule M), P10, P90 | +17.4 s on average (−1 to +83 s), year-over-year correlation 0.85; LIRF July flagged |
| H1 until 13 Oct (superseded) | `op_h1_r4_nodoy`, seeds 42-44 (`..._4e884d3b__s42/s43/s44`); the median member `op_h1_r4_nodoy_mae_exact_cpu`, seed 42 (`..._00a32933__s42`) | `h1__2026-10-11T13-55-49__final_op_h1_blend__58d3474d__mae.parquet` | 1dca8c80175d... | mean, median (rule M), P10, P90 | +18 s on average (+1 to +71 s), year-over-year correlation 0.90; LIRF July flagged |
| 3 h | `op_hl3_r4_rwyblind`, seeds 42-43 (`..._c93d69e3__s42/s43`) | `hl3__2026-10-11T18-09-14__final_op_hl3_blend__4d74dbe5.parquet` | e2252da943f5... | mean | +27 s (−56 to +86 s), correlation 0.56; four July airport-months flagged |
| 24 h | `op_hl24_r3_rwyblind`, seeds 42-43 (`..._930c7b15__s42/s43`) | `hl24__2026-10-11T18-11-54__final_op_hl24_blend__13b5a89d.parquet` | a395355fae0f... | mean | +25 s (−49 to +92 s), correlation 0.51; three July airport-months flagged |
| H1 + planning data (concept, tier F) | `op_h1_f4_nodepthru`, seeds 42-43 refitted with `-o model.refit_on_all=true` (`..._8946614f__s42/s43`) | `h1f__2026-10-13T05-41-39__final_op_h1f_blend__08a90f35.parquet` | 1cae0824c3bb... | mean | +16 s (−3 to +82 s), correlation 0.87; LIRF July flagged |
| F5, the full concept (tier F) | `op_h1_f5_full`, seeds 42-43 refitted with `-o model.refit_on_all=true` (`..._4fae29d6__s42/s43`) | `h1f5__2026-10-13T17-23-14__final_op_h1f5_blend__f7393913.parquet` | bac06dfdc0bf... | mean | +12.1 s (−9 to +60 s), correlation 0.887; LIRF July flagged |

On the final set the H1 median is F1's median member, clamped into the band on 0.79 % of the rows. Over the final set
the file's medians are 960 s (mean), 922 s (median), 723 s (P10) and 1,207 s (P90). The file before rule M
(`h1__..._aa1cd03b.parquet`, sha1 f7f4f2ac87f3..., the calibrated median) stays in the folder and in the manifest as
`h1_calibrated_median`.

The superseded H1's files stay in the folder too. `lab/op_adopt.py archive` moved their manifest records to
`h1_r4_nodoy` (the file of the table: its median member clamped on 0.81 % of the rows, medians 962 s for the mean and
921 s for the median) and `h1_r4_nodoy_calibrated_median` (`h1__..._58d3474d.parquet`, sha1 46e4da46b0e1..., the
calibrated median that rule M replaced). The folder also holds `lab/op_deliver.py`'s two calibration intermediates,
`h1__..._aa1cd03b__calibrated.parquet` and `h1__..._58d3474d__calibrated.parquet` (float64 values, an `apt` column,
indexed by `mvt_id`): not deliverables, and in no manifest record.

The concept member's file (`h1f`) reads the other flights' realised arrivals after the anchor (tier F: planned in the
concept, realised in the challenge data). It is an upper bound for perfect plans, here a perfect arrival manager, and
not available as computed: beside the deliverable, never instead of it (section 3b).

F5's file (`h1f5`, the owner's request) reads every realised movement of the other flights after the anchor, and the
ADS-B picture before it. Among the operational models it reads the most context, but not the post-operations model's
decisive inputs: the flight's own takeoff, its own ADS-B track and the NM join window. Its realised departure side
carries traces of the own slot, which rule T could not credit, so it is an upper bound even for perfect plans. Its
manifest record names tier F and this caveat. It sits beside the deliverable, never instead of it (section 3b).

The six files of the table hold 670,790 rows, the final template's ids (template sha1 4006396e963f...), data version
final26_7a06db85.
The level check compares each airport-month's predicted mean with EUROCONTROL's published monthly mean taxi-out. It is
the only out-of-time check that needs no target. Where the PRU's reference share is below 0.9 (mostly January), its
mean is a floor. Hours ahead the models track the typical month rather than this one, so their year-over-year
correlation is lower than H1's.

**On the live board** (12-13 Oct; findings 10.35-10.37). The board scores RMSE over every January and July 2026
departure, so the stale and wrong-date records that no operational model can predict (section 5) set its value. The
means' January and July rows scored 561.0 (the H1 deliverable until 13 Oct, v75), 589.1 (3 h, v76) and 596.1 (24 h,
v77), against 584.5, 606.2 and 611.0 on all rows of the 2025 holdout. On 13 Oct the superseded H1 file's median (rule
M) scored 575.1 (v79) and the calibrated median it replaced 564.9 (v81), as a median must on a squared error. F1, the
H1 deliverable since that day, scored 559.8 with its mean (v82; 583.5 on all rows of the holdout), and the concept
file `h1f` 557.5 (v80). On 14 Oct, after the daily upload limit reset, the median of the new H1 file scored 573.5
(v84) and F5's mean 551.5 (v85; tier F, an upper bound). The earlier H1 rung with the day of year (v68) scored
560.9, the post-operations model's final file 220.4 (v73). The board confirms the files and the order of the moments;
it cannot measure their operational accuracy.

**The final submission** (the challenge website's rule, relayed by the owner on 13 Oct; findings 10.37). The team's
unique final submission goes up as `<team-name>_final.parquet`, here `adventurous-radio_final.parquet`, not as a
numbered version. It is the scored challenge file, never one of the files above: run
`2026-10-11T03-22-21__final_v70_5c8bf3__5c8bf33e`, 670,790 rows, sha1 6cdf67a406177e819e083847fab7c48febc41854.
The upload script (not part of this repository) uploads that file and no other: it must be byte-identical to
`submissions/submitting.parquet` and hold the final template's ids once each, and a second final upload is refused
once one is logged. The owner uploaded it on 14 Oct at 07:15 UTC (accepted on all 670,790 rows, no score returned).
The numbered uploads v72, v74, v78 and v83 were the same file. The live scorer, which holds the truth of January and
July only, rejected each ("rows not in truth"); the file's January and July rows scored 220.39 (v73).

## 7. How it was decided

| date | decision | where |
|---|---|---|
| 1 Oct | Every input classified as available before the movement or not (tiers O, T, A, M, P, C) | `docs/feature_table.md` |
| 8 Oct | The operational ladder at H1 (R0-R5, H0, a hindsight rung) with leakage gates | findings 10.27 |
| 9 Oct | The long horizons (3 h, 24 h), runway-blind deliverables, the ADS-B archive rejected by its forward check | findings 10.31 |
| 10-11 Oct | R6 (the public ADS-B surface picture) unresolved after two seeds: not a deliverable | findings 10.34 |
| 11 Oct | The owner's goal: the operational model is the deliverable; rules L, R, C, M and F pre-registered before their runs; the day of year dropped; the long horizons moved to the schedule clock | findings 10.35 |
| 11 Oct | Rule L: learning rate 0.08 kept (0.04 missed the bar by 0.23 s) | findings 10.35 |
| 11 Oct | Rule F: not tripped (H1 forward November +3.2 s and December −0.6 s against the month pairs) | findings 10.35 |
| 11 Oct | Rule R: 3 h keeps NM's history (r4; dropping it cost 1.1 s on the forward check), 24 h drops it (r3) | findings 10.35 |
| 11 Oct | The schedule clock costs nothing against the IOBT clock (3 h 215.0 against 215.6 s, 24 h 220.3 against 222.1 s) | findings 10.35 |
| 11 Oct | Rule C: the calibrated median ships at H1 (183.1 s, every airport better); at 3 h and 24 h it would make Istanbul 2.2 s worse, so the mean only | findings 10.35 |
| 11 Oct | Sensitivities at H1: no TAF 187.3 s (inside the seed range), no plan 189.8 s (+2.2 s), runway-blind 205.0 s (+17.4 s), H0 at the push 178.2 s (−9.4 s) | findings 10.35 |
| 11 Oct | The day of year inflated the lower rungs of the first H1 ladder: re-measured calendar-free, weather is worth −2.5 s of MAE (−13 s of RMSE), not −14 s | findings 10.35 |
| 12 Oct | R6 on the five month pairs: +0.05 s of MAE against R4, so the public ADS-B surface picture adds nothing before the movement | findings 10.35 |
| 12 Oct | Rule M: the median member (MAE loss, exact leaves, on the CPU because the GPU fails with exact leaves) replaces the calibrated median at H1: 181.2 s and 50.2 % within ±2 min against 183.1 s and 49.2 %, both criteria met (the ±2 min one by 0.06 points); clamped into the band as shipped, 181.1 s, no airport worse | findings 10.35 |
| 12 Oct | The owner's concept: traffic after the anchor. Rule T pre-registered (18:30 UTC) and amended before any run after two adversarial reviews (19:30 UTC): the own-slot bound is a variant without any realised departure-side column | findings 10.36 |
| 12 Oct | Rule T stage 1 (20:26-23:29 UTC): the plans after the anchor −2.6 s; with the realised traffic −10.2 s, whose departure side is not creditable; the member: plans and realised arrivals (−4.0 s) | findings 10.36 |
| 12-13 Oct | Rule T stage 2 (23:29-01:48 UTC): the member better in 10 of 10 month-pair months and forward (−5.5 s); refitted with two seeds and applied to the final set as a separate file (tier F) | findings 10.36 |
| 13 Oct | F1 (the other flights' plans after the anchor, O*) refitted with three seeds as an option, not delivered: 184.3 s against the H1 deliverable's 186.7 s on the full holdout, 559.8 against 561.0 on the board | findings 10.36 |
| 13 Oct | The owner adopts F1 as the H1 deliverable (about 08:15 UTC). Its checks are fixed before their runs, with the bars of 10.35 and 10.36: the generalisation (rule T's criteria 4 and 5, against the superseded H1), rule F with F1's own month pairs as references, rules C and M for F1. Rule L carries over (learning rate 0.08) | findings 10.37 |
| 13 Oct | The owner asks for F5, the full concept, as well (about 09:00 UTC): a second tier F file beside the deliverable, the mean only; an upper bound even for perfect plans, because its realised departure side carries traces of the own slot | findings 10.37 |
| 13 Oct | The final submission (the challenge website's rule, relayed by the owner about 09:15 UTC): the scored challenge file goes up once as `adventurous-radio_final.parquet`; the owner uploads it | findings 10.37 |
| 13 Oct | F1 generalises: better than the superseded H1 in 10 of 10 month-pair months (by 1.17-3.10 s) and on the forward split (173.36 against 178.15 s). Rule F not tripped: November +0.95 s and December −2.21 s against F1's own month pairs (tripwire +8 s) | findings 10.37 |
| 13 Oct | Rule C passed for F1: the calibrated median 180.79 s against the mean's 184.29 s (−3.51 s), +2.33 points within ±2 min, no airport worse, coverage 77.7 %; F1's file delivered with it (11:54 UTC) | findings 10.37 |
| 13 Oct | Rule M passed for F1: its median member 178.82 s and 50.88 % within ±2 min, 1.97 s and 1.18 points better than the calibrated median; the file rebuilt with it (14:41 UTC) is the H1 deliverable. The superseded H1's files and records are kept | findings 10.37 |
| 13 Oct | F5 delivered as `h1f5` (15:23 UTC; two refitted seeds): OP MAE 177.43 s, 6.86 s below F1 and 0.8 s below H0 | findings 10.37 |
| 14 Oct | Board checks (00:02 UTC): the H1 deliverable's median 573.55 (v84), F5's mean 551.52 (v85). The owner uploaded the scored challenge file as `adventurous-radio_final.parquet` (07:15 UTC): accepted on all 670,790 rows, no score returned | findings 10.37 |

## 8. Reproduce

QUICKSTART path B gives every step, with the data, the downloads and the final tree `data_final/` they need. For the
three deliverable files the steps are, in this order:

```powershell
python -m pytest -q tests/test_operational.py tests/test_operational_long.py tests/test_operational_future.py tests/test_feature_catalogue.py tests/test_op_adopt.py
# H1 (F1): three refitted seeds, the five 2025 month pairs behind the band (rule C), the forward run (rule F), the median member (rule M)
python -m prc train -c op_h1_f1_plan.yaml --seeds 3 --keep-test
python -m prc train -c op_h1_f1_plan_febdec.yaml --no-test                # and _marnov, _aproct, _maysep, _junaug
python -m prc train -c op_h1_f1_plan_fwd.yaml --no-test                   # fit Jan-Oct, validate Nov+Dec
python lab/op_adopt.py checks                                             # against the superseded H1's pairs and forward run (below); rule F
python lab/op_calibrate.py fit --pairs op_h1_f1_plan_febdec op_h1_f1_plan_marnov op_h1_f1_plan_aproct op_h1_f1_plan_maysep op_h1_f1_plan_junaug --out runs/op_calib/h1_f1_pairs_lr08.parquet
python lab/op_calibrate.py check --tables runs/op_calib/h1_f1_pairs_lr08.parquet --runs (python lab/op_rules.py refits op_h1_f1_plan --n 3).Split() --member h1
python -m prc train -c op_h1_f1_plan_mae_exact_cpu.yaml                   # CPU, folds and refit: 2.7 h and 3.5 h recorded
python lab/op_adopt.py rule-m                                             # rule M against F1's calibrated median
python lab/op_deliver.py --tree data_final --member h1 --config op_h1_f1_plan.yaml --n 3 --tables runs/op_calib/h1_f1_pairs_lr08.parquet --rule-c
python lab/op_deliver.py --tree data_final --member h1 --config op_h1_f1_plan.yaml --n 3 --tables runs/op_calib/h1_f1_pairs_lr08.parquet --rule-c --median-config op_h1_f1_plan_mae_exact_cpu.yaml
# 3 h and 24 h: two refitted seeds each, the mean only
python -m prc train -c op_hl3_r4_rwyblind.yaml --seeds 2 --keep-test      # rule R chose r4
python -m prc train -c op_hl24_r3_rwyblind.yaml --seeds 2 --keep-test     # rule R chose r3
python lab/op_deliver.py --tree data_final --member hl3 --config op_hl3_r4_rwyblind.yaml --n 2
python lab/op_deliver.py --tree data_final --member hl24 --config op_hl24_r3_rwyblind.yaml --n 2
```

- **H1 takes two delivery calls**, as on 13 Oct (findings 10.37). The first writes the file with the calibrated
  median. The second predicts the median member, writes the `__mae` file, the H1 deliverable (the manifest's `h1`),
  and keeps the first file's record as `h1_calibrated_median`.
- **The refit flag.** `op_h1_f1_plan.yaml` sets it itself. The recorded refits were made with
  `-o model.refit_on_all=true` before the flag moved into the file; the resolved configuration and its hash
  (40ac5f36) are the same.
- **The adoption's comparator.** `lab/op_adopt.py checks` reads the superseded H1's month pairs
  (`python -m prc train -c op_h1_r4_nodoy_febdec.yaml --no-test`, likewise the other four) and its forward run
  (`python -m prc train -c op_h1_r4_nodoy_fwd.yaml --no-test`). Without them its generalisation criteria fail and
  name the missing runs. On a machine that already holds the superseded H1's records (`C_h1`, `M`, the manifest's `h1`
  and `h1_calibrated_median`), `python lab/op_adopt.py archive` keeps them as `C_h1_r4_nodoy`, `M_r4_nodoy`,
  `h1_r4_nodoy` and `h1_r4_nodoy_calibrated_median` before F1's check and delivery. `rule-m` refuses until the
  superseded H1's rule M record is archived as `M_r4_nodoy`. No script writes that record, so on a fresh checkout
  `rule-m` stops and the comparison is made by hand (QUICKSTART path B).
- **The superseded H1's files** follow from the same steps, without `lab/op_adopt.py`, with `op_h1_r4_nodoy` in place
  of `op_h1_f1_plan`: its five pairs, the tables `runs/op_calib/h1_nodoy_pairs_lr08.parquet` and its median member
  `op_h1_r4_nodoy_mae_exact_cpu` (395 min on the CPU, folds and refit). Those steps write `C_h1` and the manifest's
  `h1` and `h1_calibrated_median`, as F1's do. On the development machine they ran on 11 and 12 Oct, before F1's, and
  `lab/op_adopt.py archive` kept their records (above). Never run them after F1's check and delivery (QUICKSTART
  path B).
- **3 h and 24 h without calibration tables.** Rule C failed at both leads (section 1), so their files hold the mean
  only, and `lab/op_deliver.py` without `--tables` writes exactly that. The queue passed the tables with `--rule-c`,
  which dropped them on rule C's recorded decision. Reproducing that decision needs the long members' own month pairs,
  `lab/op_calibrate.py fit` and `check` (findings 10.35).
- **The concept file `h1f`** follows the rule T block of QUICKSTART path B: its gates
  (`tests/test_operational_future.py`, `tests/test_op_rule_t.py`), the seed-42 runs, `lab/op_rule_t.py stage1`, the
  member's forward run and month pairs, `lab/op_rule_t.py stage2`, the two refits and `lab/op_deliver.py --member h1f`.
  Besides the superseded H1's month pairs and forward run (above), the comparators of criteria 4 and 5, it needs the
  frozen ADS-B surface snapshot `s1`, built from the points of 2025 and of the final months.
- **The order of rule T's steps matters.** `stage1` and `stage2` read the newest seed-42 run of each configuration
  they name, at the configuration's current hash. Run them before the member's refits and before any other seed-42
  run of those configurations (findings 10.36). A re-run overwrites T13 and T in `runs/op_decisions.json` and
  `runs/op_rule_t.md`. Since F1's configuration carries its refit flag, a re-run would also read F1 at seed 42 of its
  refits, as the ladder's rule T table does (section 3b). To re-check the record, compare the run ids with those of
  T13's rows; the rule T table of `docs/operational_ladder.md` lists the same runs, F1's excepted.
- **The concept file `h1f5` (F5)** is refitted through the override, because its configuration stays holdout-only (no
  configuration with `refit_on_all` resolves a tier F block). Its refits therefore carry another hash than the
  configuration file, which `lab/op_deliver.py --any-hash` accepts. F5 also reads the ADS-B picture before the anchor
  (R6's block), so it needs the ADS-B gates of `s1` that `lab/op_ladder_run.sh` runs before R6 (QUICKSTART path B):

  ```powershell
  python -m prc train -c op_h1_f5_full.yaml -o model.refit_on_all=true evaluation.permutation.enabled=false --seeds 2 --keep-test
  python lab/op_deliver.py --tree data_final --member h1f5 --config op_h1_f5_full.yaml --n 2 --any-hash
  ```

**How the delivered runs were made.** `lab/op_queue.sh` ran the queue of 11-13 October one step at a time (the
superseded H1's median member on the CPU; F1's ran on the CPU beside the queue).
- **The steps.** Each line of the queue is `ID | budget | gate or run | command`. The queue's record (its steps, exit
  codes and times in UTC) stayed on the development machine and is not published; findings 10.35-10.37 describe the
  steps. After the adoption the same queue ran F1's month pairs and forward run (steps `AF_p_*` and `AF_fwd`), F5's
  refits and file (`AF5_refit` and `AF5_deliver`, 11:51-15:23 UTC), and then F5's month pairs and forward split, for
  information. F1's median member, F1's checks, rules C and M and its two delivery calls ran outside the queue.
- **The decisions.** The lab scripts wrote them to `runs/op_decisions.json`, which a clone lacks (`runs/` is
  gitignored); findings 10.35, 10.36 and 10.37 transcribe them. F1's are `A_F1` (the generalisation and rule F, with
  the read-out `runs/op_adopt_f1.md`), `C_h1` and `M`, written by `lab/op_adopt.py` and `lab/op_calibrate.py`. The
  superseded H1's are kept as `C_h1_r4_nodoy` and `M_r4_nodoy` (`lab/op_adopt.py archive`). Its rule M record was
  written outside the lab scripts.
  `python lab/op_rules.py metrics 2026-10-12T05-10-45__op_h1_r4_nodoy_mae_exact_cpu__94e24e8c__s42` prints that
  candidate's OP metrics; findings 10.35 sets them against the calibrated median and rule M's bar (at least 1 s of OP
  MAE and 1 point within ±2 min).
- **The commits.** Each run records the HEAD at its start in `env.json`; the commits named in this section are those
  of the development repository, whose history is not published. Eight runs make the three deliverable files:
  F1's seeds 42-44 at `3fc34f3`, `5950f1e` and `b820a37` (the `env.json` of seeds 42 and 43 records a dirty tree),
  F1's median member at `38df172`, 3 h at `e9d5609` and `b160e46`, and 24 h at `b160e46`. The superseded H1's seeds
  were at `1817b0a`, `d1372a0` and `e6b0ae3` (seed 42's tree dirty), its median member at `611aaae`. The concept files'
  refits: `h1f` at `c277e9c` and `3fc34f3`, F5 at `5a63d3e` and `fe99492` (seed 43's tree dirty). No feature block
  changed. Each member's runs record the same block hashes in their `feature_list.json`: F1's four runs the same 12,
  F5's two the same 16, and the 12 blocks the two share carry the same hashes in both. The only changes under
  `prc/features/` from `1817b0a` to `611aaae` (`_catalogue.py`, `_long_tiers.py`) and from `3fc34f3` to `fe99492`
  (`_catalogue.py`) are imported by no block; none changed between `c277e9c` and `3fc34f3`. `prc predict` re-checked
  every block's hash when it applied the refits to the final set and found no drift.
- **The gates.** The leakage gates ran in the queue before the long runs:
  - the synthetic suite: 225 passed, 3 skipped (the real-data slices);
  - the real slice of the long horizons: 1 passed.
- **The freeze** (13 Oct, commit `b820a37`, before the adoption). The whole suite, `python -m pytest -q`, in a clean
  worktree without `data/`, `cache/` and `runs/`, as a fresh clone runs it: 879 passed, 26 skipped, 3 xfailed, in
  13 min. The skips are the files that need the competition data or the lab's ADS-B tables and the real-data slices
  (`OPLAD_REAL`, `OPLAD_FULL`, `HORIZON_REAL`). The three expected failures are strict: they pin limits of the tier F
  blocks that `tests/test_operational_future.py` documents (the placebo guard's reach, a time-only link to the wrong
  airframe, a boolean accepted as a day shift).
- **After the freeze** the adoption changed configurations and code, but no feature block: F1's month-pair, forward
  and median-member configurations and its refit flag (`38df172`); `lab/op_deliver.py`'s member `h1f5` and option
  `--any-hash` (`dc68b06`); the upload script's guard for the final submission and its test (`0142ceb`; neither is
  part of this repository);
  `lab/op_adopt.py` with `tests/test_op_adopt.py`, and the ladder report and the catalogue reading F1 as the H1 member
  (`5a63d3e`; under `prc/features/` only `_catalogue.py`, which no block imports); the catalogue with F5 delivered
  (`c9c3578`).
- **The final suite** (13 Oct, commit `c9c3578`, after the adoption; every code change of the day included), run the
  same way: 976 passed, 26 skipped (the same files and slices), 3 xfailed (the same three), in 12 min. The freeze and
  this run were of the development repository's suite, which also tested the upload tool and a lab script that are
  not published; this repository's run is in `docs/DEVELOPER_GUIDE.md` section 3.3.
