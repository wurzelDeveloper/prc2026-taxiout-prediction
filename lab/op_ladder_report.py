"""The operational ladder's read-out: every rung and every comparison row scored the same way.

    python lab/op_ladder_report.py                       # newest run of each op config under runs/
    python lab/op_ladder_report.py --runs-root <main checkout>/runs --runs-root runs
    python lab/op_ladder_report.py --run op_h1_r4=<run_id> --run op_h0_r5=<run_id>

Reads ``oof_predictions.parquet`` of the op runs (``op_h1_r0`` .. ``op_h1_r5``, ``op_h0_r5``, the runway-blind and
winter variants; the newest run directory per config name *whose config hash is the current config's*, so a
smoke run with ``-o`` overrides or a run of an older edit is never picked up silently, unless ``--run name=<id>``
pins one; the rows labelled rule R read the runs rule R decided on, as runs/op_decisions.json records them) and of the
comparison runs below, and writes ``docs/operational_ladder.md``,
``docs/operational_ladder.csv`` (every row, population and slice), ``operational_ladder_steps.csv`` (the paired
rung steps, overall and per airport) and ``operational_ladder_unimpeded.csv``. Every Jan+Jul rung must have scored exactly C1's holdout rows (else a
WARNING); the ladder table prints n.

Early-stopping asymmetry (not fixable without a harness change): an op rung is a plain CatBoost whose
``use_best_model`` stops on the WHOLE uncleaned holdout at raw targets (stale multi-hour records and the
wrong-date rows near 87,000 s included), while C1's mixture regressor stops on holdout rows with target
<= 7,200 s that are not stale-labelled. The tail can move an op rung's best iteration (and through
``plan_refit_iterations`` the board refit's rounds). The report prints every run's best iteration (from its
log) so the reader can see it; a hash-neutral ``model.eval_max_target_sec`` in ``prc/train.py`` would remove it.

Populations (all on the holdout rows each run scored):

* ALL       -- the challenge metric;
* EX_WD     -- without the rows within 3,600 s of whole days (wrong-date block times; `rmse_ex_dayoffset`);
* BULK      -- target below 3,600 s;
* OP        -- BULK and NM-matched: flights with a flight plan whose record is a taxi time (the operational population);
* TRIM      -- per airport the top and bottom 0.1 % of targets removed (Swatowska 2025's population);
* DISRUPTED -- the airport-days the harness marks as disrupted (the runs' ``disrupted`` column);
* H1_SENS   -- OP without the rows whose recorded off-block precedes the H1 anchor (the flight was already moving).

Metrics: RMSE, MAE, MedAE, bias (mean prediction minus target), P90 |error|, the shares within +-60, 120, 180, 240,
300, 420 and 900 s (A-CDM tolerances +-2 / +-5 / +-7 min; +-1 / +-3 / +-4 / +-15 min for the literature), and the
spread-normalised errors MAE/SD and RMSE/SD (SD = the target's standard deviation on the same slice; the only fair
bridge to Idris 2002 and Herrema 2018); per airport and per month on OP. No MAPE: it is dominated by short taxis.

Table baselines computed here (no run directory; from the 2025 departures with target < 7,200 s outside the
holdout months, i.e. Feb-Jun + Aug-Dec): B1-fit (B1's stand x runway chain refitted on those rows, so B-HAT's
hour term can be read against it), B-HAT (stand x local hour of the planned off-block, falling back through the B1
chain) and B3 (mean taxi of the same flight number, ``FLIGHT_mvt`` at the same airport, over the 14 days before the
H1 anchor: realised taxis of earlier days, holdout days included, are known operationally; Idris 2002's comparator).
``FLIGHT_ID_mvt`` is a per-movement id, not a flight number (checked on two January weeks: no id on two days, while
``FLIGHT_mvt`` repeats on 92-100 % of rows' keys per airport). The stand x runway 20th percentile is printed per
airport as an unimpeded reference (additional time = target minus it), never as a scored row.

Lab rungs outside the operational reading: ``op_h0_r5_hind`` adds the traffic during the flight's own taxi
(hindsight, NOT operational; it measures the literature's hindsight inflation like for like), ``op_h1_r4_mae`` is
r4 refitted with MAE loss (for MAE and within-tolerance comparisons; never on the board). Neither enters the
tripwire; the hindsight rung gets a paired step from op_h0_r5.

Run-level tripwire: an H1 rung that is at or better than C1 (exp066 s42) on OP is treated as a LEAK until its
top permutation blocks explain it; so is an H0 rung that beats C2 (v32 stack) at every airport. Internal
ordering, which catches leaks far smaller than "as good as a hindsight model": an H1 rung (r4, r5, runway-blind)
at or better than op_h0_r5 (everything at pushback) on OP MAE is a LEAK; an H1 rung better than op_h0_r5 at
any airport is FLAGGED. The step from each rung to the next is printed with a paired day-block bootstrap
interval (OP MAE, +-2 min share and mean signed error, identical rows), overall and per airport. The report
prints the verdict per rung.

H0 is reported on OP and H1_SENS only (for stale records the H0 anchor is the schedule); the winter rung has
its own table (other holdout rows, not comparable with the Jan+Jul ladder).

Reads columns of the whole movement frame and the target file: run it when no GPU fit is running.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this checkout's `prc`, also when run as a script

from prc import cleaning, data, metrics, paths  # noqa: E402
from prc.config import load_config  # noqa: E402
from prc.features import _oplib  # noqa: E402

OP_CONFIGS = (
    ("op_h1_r0", "H1 r0: record and plan", "O (runway O*)"),
    ("op_h1_r1", "H1 r1: + airport layout", "O (runway O*)"),
    ("op_h1_r2", "H1 r2: + weather at the anchor", "O"),
    ("op_h1_r3", "H1 r3: + other flights' movement records", "O (runway O*)"),
    ("op_h1_r4", "H1 r4: + NM completed readings, plan backlog (board run)", "O (runway O*)"),
    ("op_h1_r5", "H1 r5: + airport off-block feed (LAB, 2025)", "O; lab"),
    ("op_h0_r5", "H0 r5: everything at pushback (LAB, 2025)", "O at push; lab"),
    ("op_h1_r4_rwyblind", "H1 r4 runway-blind (sensitivity)", "O"),
    ("op_h1_r1_winter", "H1 r1 on Feb+Dec 2025 (other holdout; weather-free twin of r2_winter)", "O"),
    ("op_h1_r2_winter", "H1 r2 on Feb+Dec 2025 (other holdout)", "O"),
    ("op_h0_r5_hind", "H0 r5 + the traffic during the flight's own taxi (LAB HINDSIGHT, NOT operational: "
     "measures the literature's hindsight inflation like for like)", "O at push + T over the own taxi; lab"),
    ("op_h1_r4_mae", "H1 r4 refitted with MAE loss (MAE and within-tolerance comparisons; never on the board)",
     "O (runway O*)"),
    ("op_h1_r4_mae_fast", "H1 r4 with MAE loss at lr 0.25 / cap 30,000 (still at its cap: not converged; never on "
     "the board)", "O (runway O*)"),
    ("op_h1_r6", "H1 r6: r4 + the ADS-B surface picture of other aircraft (board run; an upper bound for 2026's "
     "reception regimes)", "O (runway O*; public ADS-B)"),
    ("op_h1_r6_shift7", "H1 r6 placebo: the ADS-B events of 7 days earlier (a gain here is era or coverage identity)",
     "O (placebo)"),
    ("op_h1_r2_adsb", "H1 r2 + ADS-B: schedule, weather, layout and public ADS-B only (no movement records, no NM)",
     "O (public data)"),
    ("op_h0_r4", "H0 r4: everything at pushback but the airport's off-block feed (LAB, 2025)", "O at push; lab"),
    ("op_h0_r4_adsb", "H0 r4 + ADS-B: can public ADS-B replace the off-block feed at the push? (LAB, 2025)",
     "O at push (public ADS-B); lab"),
    # the deliverable family of findings 10.35 (single seeds; the deliverables themselves are the DELIVERABLES rows).
    # op_h1_r4_nodoy was the H1 deliverable until 13 Oct, when F1 (op_h1_f1_plan, rule T's table) replaced it (findings
    # 10.37); its sensitivities below stay as measured
    ("op_h1_r0_nodoy", "H1 r0 without the day of year: record and plan", "O (runway O*)"),
    ("op_h1_r1_nodoy", "H1 r1 without the day of year: + airport layout", "O (runway O*)"),
    ("op_h1_r2_nodoy", "H1 r2 without the day of year: + weather at the anchor", "O (runway O*)"),
    ("op_h1_r3_nodoy", "H1 r3 without the day of year: + other flights' movement records", "O (runway O*)"),
    ("op_h1_r4_nodoy", "H1 r4 without the day of year, seed 42: the H1 deliverable configuration until 13 Oct "
     "(superseded by F1, findings 10.37)", "O (runway O*; own and others' plans O*)"),
    ("op_h1_r4_nodoy_lr04", "H1 deliverable until 13 Oct at learning rate 0.04 (rule L: not adopted)",
     "O (runway O*; own and others' plans O*)"),
    ("op_h1_r4_nodoy_noplan", "H1 deliverable until 13 Oct without the planned delay and the plan backlog (what the "
     "filed plan adds)", "O (runway O*)"),
    ("op_h1_r4_nodoy_notaf", "H1 deliverable until 13 Oct without the 13 TAF columns",
     "O (runway O*; own and others' plans O*)"),
    ("op_h1_r4_nodoy_rwyblind", "H1 deliverable until 13 Oct, runway-blind (its like-for-like comparator)",
     "O (own and others' plans O*)"),
    ("op_h1_r4_nodoy_mae_exact_cpu", "Median member of the H1 deliverable until 13 Oct: MAE loss with exact leaves, on "
     "the CPU (rule M passed)", "O (runway O*; own and others' plans O*)"),
    ("op_h1_f1_plan_mae_exact_cpu", "F1 median member: MAE loss with exact leaves, on the CPU (rule M re-run for F1, "
     "findings 10.37)", "O (runway O*; own and others' plans O*)"),
    ("op_h0_r4_nodoy", "H0 r4 without the day of year: the inputs of the H1 deliverable until 13 Oct at the push "
     "(LAB, 2025)", "O at push; lab"),
    ("op_hl_r0", "3 h and 24 h r0: record and plan at the scheduled off-block (one model for both leads)",
     "O (runway O*, stand O*)"),
    ("op_hl3_r2", "3 h r2: + layout and weather (METAR persistence, TAF for the planned hour)", "O (runway O*, stand O*)"),
    ("op_hl3_r3", "3 h r3: + scheduled demand around the planned off-block", "O (runway O*, stand O*)"),
    ("op_hl3_r4", "3 h r4: + NM history of departed flights", "O (runway O*, stand O*)"),
    ("op_hl3_r3_rwyblind", "3 h r3 runway-blind (deliverable candidate, rule R)", "O (stand O*)"),
    ("op_hl3_r4_rwyblind", "3 h r4 runway-blind (deliverable candidate, rule R)", "O (stand O*)"),
    ("op_hl24_r2", "24 h r2: + layout and weather (METAR persistence, TAF for the planned hour)",
     "O (runway O*, stand O*)"),
    ("op_hl24_r3", "24 h r3: + scheduled demand around the planned off-block", "O (runway O*, stand O*)"),
    ("op_hl24_r4", "24 h r4: + NM history of departed flights", "O (runway O*, stand O*)"),
    ("op_hl24_r3_rwyblind", "24 h r3 runway-blind (deliverable candidate, rule R)", "O (stand O*)"),
    ("op_hl24_r4_rwyblind", "24 h r4 runway-blind (deliverable candidate, rule R)", "O (stand O*)"),
    ("op_h1_r4_nodoy_fwd", "H1 deliverable configuration until 13 Oct, forward (rule F)",
     "O (runway O*; own and others' plans O*)"),
    ("op_h1_f1_plan_fwd", "F1, the H1 deliverable from 13 Oct, forward (rule F and the generalisation check, findings "
     "10.37)", "O (runway O*; own and others' plans O*)"),
    ("op_hl3_r3_rwyblind_fwd", "3 h r3 runway-blind, forward (rule R)", "O (stand O*)"),
    ("op_hl3_r4_rwyblind_fwd", "3 h r4 runway-blind, forward (rule R)", "O (stand O*)"),
    ("op_hl24_r3_rwyblind_fwd", "24 h r3 runway-blind, forward (rule R)", "O (stand O*)"),
    ("op_hl24_r4_rwyblind_fwd", "24 h r4 runway-blind, forward (rule R)", "O (stand O*)"),
    # rule T (findings 10.36): the H1 deliverable until 13 Oct (op_h1_r4_nodoy) plus traffic after the anchor; F = the
    # other flights' realised movements standing in for AMAN/DMAN/A-SMGCS plans (an upper bound, not available as
    # computed)
    ("op_h1_f1_plan", "F1: op_h1_r4_nodoy (the H1 deliverable until 13 Oct) + the other flights' plans after the "
     "anchor (filed off-blocks, scheduled arrivals); adopted as the H1 deliverable on 13 Oct (findings 10.37); read at "
     "seed 42 of its three refits, not rule T's holdout-only run of 12 Oct", "O* (others' plans)"),
    ("op_h1_f2_rec", "F2: F1 + their realised movement records after the anchor", "O* + F"),
    ("op_h1_f3_nm", "F3: F2 + NM's readings of the flights taking off after the anchor", "O* + F"),
    ("op_h1_f4_adsb", "F4: F3 + the public ADS-B surface picture after the anchor (the full concept after the anchor)",
     "O* + F"),
    ("op_h1_f4_shift7", "F4's placebo: every F block read 7 days earlier", "O* + F (placebo)"),
    ("op_h1_f4_nodepthru", "F4 without any realised departure-side column: plans and realised arrivals (rule T's member)",
     "O* + F (arrivals)"),
    ("op_h1_f4_norwy", "F4 without the own runway's sequence after the anchor", "O* + F"),
    ("op_h1_f5_full", "F5: F4 + the ADS-B picture before the anchor (actual and planned)", "O + O* + F"),
)
LONG_SINCE = "2026-10-11T12-10-00"
"""The long rungs moved to the schedule clock on 11 Oct (findings 10.35). The ladder configs r0-r4 changed only in
comments, so their hashes equal the IOBT-clock runs'; a long rung's run counts only from this run-id stamp (local time)."""
SEED42 = ("op_h1_r4_nodoy", "op_h1_f1_plan", "op_hl3_r3_rwyblind", "op_hl3_r4_rwyblind", "op_hl24_r3_rwyblind",
          "op_hl24_r4_rwyblind")
"""Rungs read at seed 42 (several seeds share the name and hash; the sensitivities they are compared with, and the runs
rule R decided on, are seed 42). F1's config carries its refits' hash since its adoption (findings 10.37), so its row is
the seed-42 refit's holdout phase, not rule T's holdout-only run of 12 Oct (another hash; runs/op_rule_t.md has it)."""
RULE_R_ROWS = {f"op_{lead}_{rung}_rwyblind{fwd}": (f"R_{lead}", part, rung)
               for lead in ("hl3", "hl24") for rung in ("r3", "r4")
               for fwd, part in (("", "main"), ("_fwd", "forward"))}
"""The rows labelled rule R and where runs/op_decisions.json records the run rule R decided on (key, part, rung). The
deliverables' seed-42 refits share these names, hashes and seed and are newer, so a row reads the recorded run when the
record holds it (`rung_dir`)."""
NOT_IN_TRIPWIRE = ("op_h0_r5_hind", "op_h1_r4_mae")
"""Lab rungs read outside the leak tripwire: a hindsight rung is better by design, a MAE-loss rung is another metric."""

COMPUTED = (
    ("B1-fit", "B1 chain refitted on the report's fit rows (stand x runway, stand, runway, airport medians)",
     "O (runway O*)"),
    ("B-HAT", "B-HAT stand x local-hour median (hour of the planned off-block), B1 chain as fallback",
     "O (runway O* in the fallback)"),
    ("B3", "B3 mean taxi of the same flight number over the 14 days before H1 (Idris-style), B1 chain as fallback",
     "O (earlier days' realised taxis; runway O* in the fallback)"),
)
"""Table baselines computed by this script from the fit rows; scored like the runs, no run directory."""
HOLDOUT_MONTHS = (1, 7)
FIT_MAX_TARGET_S = 7200.0
HISTORY_DAYS = 14
MIN_SUPPORT = 20
"""Rows a table cell needs before it is used (as ``GroupMedian``)."""
UNIMPEDED_QUANTILE = 0.2

COMPARISON = (
    ("B0", "2026-09-14T16-05-59__baselines__ec76a3db__s42", "B0 global median", "O", "no fit"),
    ("B1", "2026-09-14T16-06-25__baselines__8ead7253__s42", "B1 stand-runway median table (A-CDM VTT practice)", "O (runway O*)", "no fit"),
    ("B2", "2026-09-14T16-06-40__baselines__0eb24067__s42", "B2 NM proxy takeoff - AOBT_3", "A; T at LTFM", "no fit"),
    ("C1", "2026-10-06T18-17-11__lean2_global_window_long__6d1b7355__s42", "C1 exp066 member (challenge model)", "O+T+A+C", "trained on the other ten months"),
    ("C2", "2026-09-30T21-48-31__v32b_stack__faf18912", "C2 v32 stack, no ADS-B (challenge reconstruction)", "O+T+A+C", "holdout-tuned stack"),
    ("C3", "2026-10-06T10-47-37__v62_swap3_shift__50d0d544", "C3 v62 final, + own ADS-B (challenge reconstruction)", "O+T+A+C+M", "ADS-B stage cross-fitted by day on the holdout"),
    ("L3-IOBT", "2026-10-11T02-14-34__op_hl3_r4_rwyblind__863fff84__s42", "3 h r4 runway-blind on the IOBT clock (10-11 Oct, superseded: the filed plan known 3 h ahead)", "O* (filed plan; stand O*)", "trained on the other ten months"),
    ("L24-IOBT", "2026-10-11T02-42-26__op_hl24_r4_rwyblind__3ef42e10__s42", "24 h r4 runway-blind on the IOBT clock (10-11 Oct, superseded: the filed plan known 24 h ahead)", "O* (filed plan; stand O*)", "trained on the other ten months"),
)

LITERATURE = """\
Verified against the full texts (with a page-level number check);
paraphrased, pages in brackets. Every row's last column is its protocol caveat: none of them is our protocol.

| Study | Airport(s) | Horizon | Information set | Validation | Population | Result | Caveat (like for like?) |
|---|---|---|---|---|---|---|---|
| Idris et al. 2002 | BOS, ten carriers (about half the traffic; p.5) | at pushback | others' surface state at pushback: N = departures pushed back and not yet airborne, from the other flights' actual Out/Off (pp.14-17); runway configuration from PRAS records (p.6, fn 7); airline | temporal: fit May-Jul 1998, test August 1998 (pp.14, 17) | top 1 % removed from the fit (p.15, fn 20) | queue model MAE 4.56 min (274 s) vs 5.69 min for the mean of the flight's fourteen preceding occurrences; within 5 min 65.6 vs 53.7 % (Table 2, p.17); MAE/SD 0.41 vs 0.52 (SD about 11 min, p.2 fn 3) | The H0 comparator: information set of op_h0_r5; our B3 is a 14-day analogue of his running average (days, not occurrences). Other airport, era and spread: compare MAE/SD only (the SD is the whole analysed sample's). The strong single factor in its abstract is the hindsight queue Q (takeoffs during the own taxi; p.13 and Fig. 8, p.14), not N |
| Wang et al. 2021 (TR-C 124; cited by us as Wang 2020) | MAN / ZRH / HKG | at pushback (Groups A-D); post-taxi (E) | A: the own realised ADS-B route, the runway used and weight class; B: + other aircraft taxiing at pushback; D: + recent speeds of other aircraft and weather; E: + aircraft reaching runway or stand during the own taxi (hindsight; the authors exclude it as unobtainable in practice, pp.6-7) | random 70/30 row split, 10-fold CV on the 70 % (p.10) | arrivals and departures pooled; departures end at line-up, not wheels-off; complete tracks only (69 / 57 / 78 %), taxis over 45 min removed (p.5) | test RMSE in min (Tables 11-13, pp.12-13): A 3.50 / 3.27 / 4.06; B 3.43 / 3.16 / 3.28; D 3.37 / 3.10 / 3.10 (MAE 2.29 / 1.78 / 1.96; MAN within 1 / 3 / 5 min 34.1 / 74.6 / 89.8 %, Table 11); E 2.96 / 2.15 / 2.07 | No: pooled arrivals, line-up endpoint, random split, own realised route. Steps: A to B -2 / -3 / -19 % RMSE; D to E -12 / -31 / -33 % (the published hindsight inflation; ours is op_h0_r5 to op_h0_r5_hind). Recent speeds rank 4th at MAN and HKG, 8th at ZRH (Table 14, p.17). The paper's Table 8 gives other MAN shares than Table 11 |
| Swatowska and Gabagnou 2025 | EBBR, LFPO, LKPR, LOWW, LSZH, LTFM | not declared (calendar at movement time) | assigned runway and stand and the de-icing status from post-operational APDF records (Table I, p.4), weather, calendar; no traffic inputs | year-ahead: fit 2022-23, test 2024; tuning by random stratified 5-fold (p.3) | top and bottom 0.1 % per airport and phase removed (p.3) | departure MAE, model vs the airport's taxi-time table (Table V, p.5): LSZH 2:18 vs 3:34 (-35 %), LFPO 2:02 vs 2:03 (-1 %), LTFM 3:32 vs 3:45 (-6 %); CatBoost with MAE loss (Table II, p.4); the only RMSE pools taxi-in (4.05 min, Table III) | Closest: same target, two shared airports, year-ahead. Compare departure MAE on TRIM at LSZH / LTFM (and against op_h1_r4_mae for the loss). Their runway and stand are post-operational records |
| Herrema et al. 2018 | LFPG, runway 08L, stand groups A-C | stated as before off-block; evaluated at or after it | inputs include the actual AOBT and the departures in the last 20 min counted at the flight's own ATOT (Table 1, p.122; top 10, p.124); an unimpeded flag derived from the observed taxi (p.125) | random 70/15/15, random 4-fold CV for tuning (pp.124-125) | one runway; outliers kept (p.124) | regression tree RMSE 1:36 min (96 s; Table 3, p.125); RMSE/spread about 0.43 against the root node's 3.7 min (Fig. 7, p.126); 94.2 % within 3 min with arrivals pooled (Table 5, p.128) | No: hindsight inputs (ATOT-anchored count, AOBT), random split, one runway |
| Wu et al. 2026 | ZGSZ, runway 15 departures, 14 days of 2019 | post-taxi | takeoffs and landings counted over the own [AOBT, ATOT] (Table 1, p.6); airline and stand as clusters of their mean taxi-out (p.7) | random 9:1, random 10-fold (pp.8, 10) | taxi in [0, 3,600] s (p.5) | stacking RMSE 140.5 s, MAE 99.7 s; 41.0 / 86.3 / 96.5 % within 60 / 180 / 300 s (Table 4, p.10; p.11) | No: hindsight counts over the own taxi, random split, 14 days, one runway |
| Tang et al. 2025 | ZSPD, December 2023 | post-taxi | surface flows classified against the own AOBT and ATOT (Table 5, p.11); route-sum taxi estimate learned from realised link times (pp.6-7); CTOT flag | random 8:2, 5-fold CV for tuning (p.12) | cleaning rule not given (p.4) | all features: RMSE 1.561 min (about 94 s), MAE 1.178 min (about 71 s); 54.4 / 94.4 / 99.4 % within 1 / 3 / 5 min (Table 8, p.13) | No: hindsight counts over the own taxi, random split, one month |
| Yin 2018, He 2022, Jiao 2022 | ZSPD / ZPPP / HKG | post-taxi | counts over the own taxi | Yin one test day; He random 20 %; Jiao not described | various | RMSE about 115 s (Yin, Table 3, p.7), 146 s (He, Table 5, p.12), about 213 s (Jiao, Table 14, p.26) | No: hindsight reconstructions |
| Guenther 2011 (not verified: full text not read) | EDDF | static table / at off-block | - | - | - | abstract only: 42 % within 2 min (static table), 66 % traffic-aware at off-block | Not citable as a result until read; the planned EDDF +-2 min comparison |
| Lim et al. 2024 (not verified: full text not read) | KATL | - | - | reportedly temporal, last 10 % | - | no figure quoted: our notes carry conflicting abstract-level numbers | Not citable until read; the protocol description resembles Lim 2021 |
| EUROCONTROL 2017 (A-CDM) | - | - | - | - | - | taxi-time tolerances +-7 / +-5 / +-2 min | Definitions, not results: the +-420 / 300 / 120 s shares |

Our protocol: seasonal Jan+Jul 2025 holdout, one global model trained on the other ten months, ten airports,
departures only, AOBT to ATOT, the operational population OP. Raw shares within a tolerance do not travel across
studies (+-5 min is 0.45 SD at Boston); compare spread-normalised (MAE/SD, RMSE/SD) and on the same population
definition: the EDDF +-2 min share with Guenther 2011 once read, the LSZH / LTFM MAE on TRIM with Swatowska 2025,
and op_h0_r5's MAE/SD with Idris 2002 (B3 with its flight-history average). Our B1 table is an empirical stand x
runway median, likely stronger than an airport's published taxi-time table."""

H0_RUNGS = ("op_h0_r5", "op_h0_r5_hind", "op_h0_r4", "op_h0_r4_adsb", "op_h0_r4_nodoy")
H0_POPULATIONS = ("OP", "H1_SENS")
"""H0 is reported on these populations only: for a stale record the H0 anchor is the schedule, not the push."""
RULE_T = ("op_h1_f1_plan", "op_h1_f2_rec", "op_h1_f3_nm", "op_h1_f4_adsb", "op_h1_f4_shift7", "op_h1_f4_nodepthru",
          "op_h1_f4_norwy", "op_h1_f5_full")
"""Rule T's runs (findings 10.36): the same holdout, but tier F inputs from F2 on (F1 adds the other flights' plans
only, O*, and is the H1 deliverable since 13 Oct, findings 10.37). This report scores them, and their steps, on every OP
row from 1 January; rule T itself scores them from 9 January (lab/op_rule_t.py, `FIRST_DAY`)."""
FORWARD = ("op_h1_r4_nodoy_fwd", "op_h1_f1_plan_fwd", "op_hl3_r3_rwyblind_fwd", "op_hl3_r4_rwyblind_fwd",
           "op_hl24_r3_rwyblind_fwd", "op_hl24_r4_rwyblind_fwd")
OTHER_TABLES = (
    ("Other holdout: Feb+Dec 2025 (not comparable with the ladder above)", ("op_h1_r1_winter", "op_h1_r2_winter")),
    ("Forward check: fit January-October 2025, validate November+December (rules F and R; not comparable with the "
     "ladder above)", FORWARD),
    ("Traffic after the anchor (rule T, findings 10.36): the same holdout and population, but the F columns read the "
     "other flights' realised movements after H1 as a stand-in for AMAN/DMAN/A-SMGCS plans (tier F: planned in the "
     "concept, realised in the challenge data; an upper bound for perfect plans, not available as computed). This "
     "table and its rung steps below use every OP row from 1 January, so they differ from rule T's own figures, "
     "which are scored from 9 January (runs/op_rule_t.md, findings 10.36)", RULE_T),
)
OTHER_HOLDOUT = tuple(n for _, names in OTHER_TABLES for n in names)
"""Rungs in a table of their own, never next to the Jan+Jul ladder: those validated on another holdout, and rule T's
(`OWN_TABLE_JANJUL`)."""
OWN_TABLE_JANJUL = RULE_T
"""Rungs with a table of their own that still score the Jan+Jul holdout (rule T's; F1, the H1 deliverable since 13 Oct,
among them): the row-set check and the tripwire read them like the ladder's rungs."""
STEPS = (("op_h1_r0", "op_h1_r1"), ("op_h1_r1", "op_h1_r2"), ("op_h1_r2", "op_h1_r3"), ("op_h1_r3", "op_h1_r4"),
         ("op_h1_r4", "op_h1_r5"), ("op_h1_r5", "op_h0_r5"), ("op_h1_r4", "op_h1_r4_rwyblind"), ("op_h1_r4", "C1"),
         ("op_h0_r5", "op_h0_r5_hind"), ("op_h1_r1_winter", "op_h1_r2_winter"),
         ("op_h1_r4", "op_h1_r6"), ("op_h1_r4", "op_h1_r6_shift7"), ("op_h1_r2", "op_h1_r2_adsb"),
         ("op_h0_r4", "op_h0_r4_adsb"), ("op_h0_r4_adsb", "op_h0_r5"),
         # findings 10.35: the long ladder on the schedule clock, rule R, the value of a filed plan, the horizons,
         # the H1 sensitivities and the calibrated median
         ("op_hl_r0", "op_hl3_r2"), ("op_hl3_r2", "op_hl3_r3"), ("op_hl3_r3", "op_hl3_r4"),
         ("op_hl3_r4", "op_hl3_r4_rwyblind"), ("op_hl3_r3_rwyblind", "op_hl3_r4_rwyblind"),
         ("op_hl_r0", "op_hl24_r2"), ("op_hl24_r2", "op_hl24_r3"), ("op_hl24_r3", "op_hl24_r4"),
         ("op_hl24_r4", "op_hl24_r4_rwyblind"), ("op_hl24_r3_rwyblind", "op_hl24_r4_rwyblind"),
         ("op_hl3_r4_rwyblind", "L3-IOBT"), ("op_hl24_r4_rwyblind", "L24-IOBT"),
         ("D-24H", "D-3H"), ("D-3H", "D-H1"),
         ("op_h1_r4_nodoy", "op_h1_r4_nodoy_noplan"), ("op_h1_r4_nodoy", "op_h1_r4_nodoy_notaf"),
         ("op_h1_r4_nodoy", "op_h1_r4_nodoy_rwyblind"), ("op_h1_r4_nodoy", "op_h1_r4_nodoy_lr04"),
         ("op_h1_r4_nodoy", "op_h1_r4_nodoy_mae_exact_cpu"), ("op_h1_r4", "op_h1_r4_nodoy"),
         ("D-H1", "D-H1-MED"), ("D-3H", "D-3H-MED"), ("D-24H", "D-24H-MED"), ("D-H1-MED", "D-H1-MAE"),
         # the H1 ladder without the day of year, and what the day of year did on the fewest inputs
         ("op_h1_r0_nodoy", "op_h1_r1_nodoy"), ("op_h1_r1_nodoy", "op_h1_r2_nodoy"), ("op_h1_r2_nodoy", "op_h1_r3_nodoy"),
         ("op_h1_r3_nodoy", "op_h1_r4_nodoy"), ("op_h1_r0", "op_h1_r0_nodoy"),
         # rule T (findings 10.36): traffic after the anchor
         ("op_h1_r4_nodoy", "op_h1_f1_plan"), ("op_h1_f1_plan", "op_h1_f2_rec"), ("op_h1_f2_rec", "op_h1_f3_nm"),
         ("op_h1_f3_nm", "op_h1_f4_adsb"), ("op_h1_f4_adsb", "op_h1_f5_full"), ("op_h1_f4_adsb", "op_h1_f4_shift7"),
         ("op_h1_f4_adsb", "op_h1_f4_nodepthru"), ("op_h1_f4_adsb", "op_h1_f4_norwy"),
         ("op_h1_r4_nodoy", "op_h1_f4_nodepthru"), ("op_h1_f1_plan", "op_h1_f4_nodepthru"),
         # findings 10.37: F1 adopted as the H1 deliverable. The adoption's gain (the superseded deliverable's seed mean
         # to F1's); the superseded deliverable's own calibration steps and its H0 comparator, which differs from it by
         # the moment alone (10.35; F1 to H0 would also differ by F1's plans after the anchor, which H0 lacks, so that
         # step is not drawn); F1's rule M candidate and its forward check
         ("D-H1-R4", "D-H1"), ("D-H1-R4", "D-H1-R4-MED"), ("D-H1-R4-MED", "D-H1-R4-MAE"),
         ("D-H1-R4", "op_h0_r4_nodoy"), ("op_h1_f1_plan", "op_h1_f1_plan_mae_exact_cpu"),
         ("op_h1_r4_nodoy_fwd", "op_h1_f1_plan_fwd"))
"""Rung pairs whose OP difference gets a paired day-block bootstrap interval, overall and per airport."""

WITHIN = (60, 120, 180, 240, 300, 420, 900)
POPULATIONS = ("ALL", "EX_WD", "BULK", "OP", "TRIM", "DISRUPTED", "H1_SENS")
MVT_COLUMNS = ["apt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "MVT_TIME_UTC_mvt", "FLIGHT_ID_mvt",
               "IOBT_flt", "LOBT_flt", "SCHED_TIME_UTC_mvt"]
BASELINE_COLUMNS = ["apt", "split", "is_dep", "STAND_mvt", "RUNWAY_mvt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt",
                    "IOBT_flt", "LOBT_flt", "SCHED_TIME_UTC_mvt"]


# --- metrics --------------------------------------------------------------------------------------

def metric_row(y_true, y_pred) -> dict[str, float]:
    """RMSE, MAE, MedAE, bias, P90 |error|, target SD, MAE/SD, RMSE/SD and the within-tolerance shares.

    NaN for an empty slice; the two ratios are NaN when the slice's target has no spread.
    """
    y = np.asarray(y_true, dtype="float64")
    p = np.asarray(y_pred, dtype="float64")
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    out: dict[str, float] = {"n": float(len(y))}
    if not len(y):
        return {**out, "rmse": np.nan, "mae": np.nan, "medae": np.nan, "bias": np.nan, "p90_abs": np.nan,
                "sd_y": np.nan, "mae_sd": np.nan, "rmse_sd": np.nan, **{f"within_{w}": np.nan for w in WITHIN}}
    e = p - y
    a = np.abs(e)
    sd = float(np.std(y))  # population SD of the target on this slice
    out.update(
        rmse=float(np.sqrt(np.mean(e**2))),
        mae=float(a.mean()),
        medae=float(np.median(a)),
        bias=float(e.mean()),
        p90_abs=float(np.quantile(a, 0.9)),
        sd_y=sd,
    )
    out["mae_sd"] = out["mae"] / sd if sd > 0 else np.nan
    out["rmse_sd"] = out["rmse"] / sd if sd > 0 else np.nan
    for w in WITHIN:
        out[f"within_{w}"] = float((a <= w).mean())
    return out


def trim_mask(y: pd.Series, apt: pd.Series, frac: float = 0.001) -> pd.Series:
    """Per airport, drop the top and bottom ``frac`` of targets."""
    lo = y.groupby(apt).transform(lambda s: s.quantile(frac))
    hi = y.groupby(apt).transform(lambda s: s.quantile(1 - frac))
    return (y >= lo) & (y <= hi)


def population_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean masks over the holdout rows; ``frame`` has y_true, apt, nm_matched, dayoffset, disrupted, h1_early."""
    y = frame["y_true"]
    bulk = y < metrics.BULK_MAX_SEC
    op = bulk & frame["nm_matched"]
    return {
        "ALL": pd.Series(True, index=frame.index),
        "EX_WD": ~frame["dayoffset"],
        "BULK": bulk,
        "OP": op,
        "TRIM": trim_mask(y, frame["apt"]),
        "DISRUPTED": frame["disrupted"],
        "H1_SENS": op & ~frame["h1_early"],
    }


# --- the deliverables (findings 10.35) --------------------------------------------------------------

DELIVERABLES = (
    ("D-24H", "hl24", 2, "24 h deliverable"),
    ("D-3H", "hl3", 2, "3 h deliverable"),
    ("D-H1", "h1", 3, "H1 deliverable (headline)"),
    ("D-H1-R4", "h1_r4_nodoy", 3, "H1 deliverable until 13 Oct, superseded by F1 (findings 10.37)"),
)
"""(row label, member, seeds averaged, title): the seed mean of the member's configuration (`deliverable_config`), and
with the member's calibration tables the calibrated median as the row ``<label>-MED``. The H1 deliverable is F1 from
13 Oct (findings 10.37); ``D-H1-R4`` keeps the H1 deliverable until then on record, and its step to ``D-H1`` is the
adoption's gain."""
MEDIAN_MEMBER = {"h1": "op_h1_f1_plan_mae_exact_cpu", "h1_r4_nodoy": "op_h1_r4_nodoy_mae_exact_cpu"}
"""Rule M's candidate per member (findings 10.35, passed for the H1 deliverable until 13 Oct; re-run for F1, findings
10.37, where it replaces the calibrated median only if it beats it by 1 s of OP MAE and 1 point within +-2 min): the
configuration whose prediction, clamped into the calibrated P10-P90 band as `lab/op_deliver.py --median-config` does,
is the row ``<label>-MAE``. The row is drawn when the candidate's run and the member's tables exist, whether or not
rules C and M passed (the file carries it only if both did)."""
DELIVERABLE_TIERS = {"h1": "O (runway O*; own and others' plans O*, before and after the anchor)",
                     "h1_r4_nodoy": "O (runway O*; own and others' plans O*)", "hl3": "O (stand O*)",
                     "hl24": "O (stand O*)"}
DEFAULT_CONFIG = {"h1": "op_h1_f1_plan", "h1_r4_nodoy": "op_h1_r4_nodoy", "hl3": "op_hl3_r4_rwyblind",
                  "hl24": "op_hl24_r3_rwyblind"}
"""The delivered configurations: F1 at H1 (findings 10.37), the H1 deliverable until 13 Oct (rule L's choice), and rule
R's choices hours ahead (findings 10.35; r4 at 3 h, r3 at 24 h), there the fallback when runs/ holds no decision
file."""
DEFAULT_TABLES = {"h1": "runs/op_calib/h1_f1_pairs_lr08.parquet",
                  "h1_r4_nodoy": "runs/op_calib/h1_nodoy_pairs_lr08.parquet", "hl3": "runs/op_calib/hl3.parquet",
                  "hl24": "runs/op_calib/hl24.parquet"}
"""Each member's rule C tables, fitted on its own five month pairs (F1's: findings 10.37). A missing file means no
``-MED`` and no ``-MAE`` row for that member (printed), never a failure."""
PAIRS = ("febdec", "marnov", "aproct", "maysep", "junaug")


def deliverable_config(member: str) -> str:
    """The configuration a deliverable row averages.

    Hours ahead (``hl3``, ``hl24``) the one rule R chose (runs/op_decision_R_<member>.cfg), else the delivered one
    (`DEFAULT_CONFIG`). At H1 the configuration is fixed: ``h1`` is F1 (`op_h1_f1_plan`), the owner's choice of 13 Oct
    (findings 10.37). F1 extends rule L's choice (`op_h1_r4_nodoy`, learning rate 0.08, runs/op_decision_L.cfg) by the
    other flights' plans after the anchor and keeps its learning rate, so rule L's file no longer names the H1
    configuration and is not read. ``h1_r4_nodoy`` is rule L's choice, the H1 deliverable until 13 Oct.
    """
    if member in ("hl3", "hl24"):
        f = paths.RUNS / f"op_decision_R_{member}.cfg"
        if f.exists():
            return Path(f.read_text(encoding="utf-8").strip()).stem
    return DEFAULT_CONFIG[member]


def seed_runs(roots: list[Path], name: str, n: int, since: str | None = None) -> list[Path]:
    """The newest run per seed of config ``name`` at its current hash; the newest ``n`` seeds, or [] if fewer."""
    h = config_hash(name)
    per_seed: dict[str, Path] = {}
    for root in roots:
        for d in root.glob(f"*__{name}__*"):
            parts = d.name.split("__")
            if (len(parts) >= 4 and parts[1] == name and parts[2] == h and (d / "oof_predictions.parquet").exists()
                    and (since is None or d.name >= since)):
                if parts[3] not in per_seed or d.name > per_seed[parts[3]].name:
                    per_seed[parts[3]] = d
    found = sorted(per_seed.values(), key=lambda d: d.name)
    return found[-n:] if len(found) >= n else []


def seed_mean(dirs: list[Path]) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """The mean prediction of several runs on the same holdout rows, and each run's frame."""
    frames = [read_oof(d) for d in dirs]
    idx = frames[0].index
    for f in frames[1:]:
        if not f.index.sort_values().equals(idx.sort_values()):
            raise ValueError("the seed runs scored different holdout rows")
    mean = frames[0][["y_true"]].copy()
    mean["y_pred"] = np.mean([f["y_pred"].reindex(idx).to_numpy(dtype="float64") for f in frames], axis=0)
    return mean, frames


def op_frame(oof: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    """y_true, y_pred, apt and the OP flag of a holdout frame (the input `op_calibrate.rule_c` reads)."""
    frame = ctx.reindex(oof.index).copy()
    frame["y_true"] = oof["y_true"]
    frame["y_pred"] = oof["y_pred"]
    frame["op"] = population_masks(frame)["OP"].fillna(False).to_numpy(dtype=bool)
    return frame[["y_true", "y_pred", "apt", "op"]]


def month_pair_mae(roots: list[Path], name: str) -> dict[str, dict]:
    """OP MAE and the share within +-2 min per month of the five month-pair runs ``<name>_<pair>`` (newest each)."""
    dirs = [find_run(roots, f"{name}_{p}", by_name=True) for p in PAIRS]
    dirs = [d for d in dirs if d is not None]
    if not dirs:
        return {}
    oof = pd.concat([read_oof(d) for d in dirs])
    mvt = data.load_movements(columns=["MVT_TIME_UTC_mvt", metrics.NM_MATCH_COLUMN]).reindex(oof.index)
    y, p = oof["y_true"], oof["y_pred"]
    op = (y < metrics.BULK_MAX_SEC) & mvt[metrics.NM_MATCH_COLUMN].notna()
    month = mvt["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
    out = {}
    for m in sorted(month[op].unique()):
        r = metric_row(y[op & (month == m)], p[op & (month == m)])
        out[m] = {"mae": r["mae"], "within_120": r["within_120"], "n": r["n"], "runs": len(dirs)}
    return out


# --- inputs ---------------------------------------------------------------------------------------

def find_run(roots: list[Path], name_or_id: str, by_name: bool, expected_hash: str | None = None,
             since: str | None = None, seed: int | None = None) -> Path | None:
    """A run directory: by exact id, or the newest whose config name is ``name_or_id``.

    With ``expected_hash`` (the current config's ``cfg.hash()``), runs of the same name with another hash -- a
    smoke run with ``-o`` overrides, a run of an older edit of the config -- are skipped, and each skip printed.
    ``since`` skips runs whose id (its local-time stamp) sorts before it; ``seed`` keeps that seed's runs only.
    """
    for root in roots:
        if not by_name and (root / name_or_id).is_dir():
            return root / name_or_id
    if not by_name:
        return None
    hits = []
    for root in roots:
        for d in root.glob(f"*__{name_or_id}__*"):
            parts = d.name.split("__")
            if d.is_dir() and len(parts) >= 3 and parts[1] == name_or_id and (d / "oof_predictions.parquet").exists():
                if expected_hash is not None and parts[2] != expected_hash:
                    print(f"op rung {name_or_id}: skipped {d.name} (config hash {parts[2]} != current {expected_hash})")
                    continue
                if since is not None and d.name < since:
                    print(f"op rung {name_or_id}: skipped {d.name} (before {since})")
                    continue
                if seed is not None and (len(parts) < 4 or parts[3] != f"s{seed}"):
                    continue
                hits.append(d)
    return max(hits, key=lambda d: d.name) if hits else None


def rule_r_runs() -> dict[str, str]:
    """Config name -> the run id rule R decided on, for the `RULE_R_ROWS` that runs/op_decisions.json records."""
    f = paths.RUNS / "op_decisions.json"
    if not f.exists():
        return {}
    doc = json.loads(f.read_text(encoding="utf-8"))
    out = {}
    for name, (key, part, rung) in RULE_R_ROWS.items():
        run = (((doc.get(key) or {}).get(part) or {}).get(rung) or {}).get("run")
        if run:
            out[name] = run
    return out


def rung_dir(roots: list[Path], name: str, pinned: dict[str, str], recorded: dict[str, str]) -> Path | None:
    """The run an op rung's row reads: a ``--run`` pin; else, for a row labelled rule R, the run rule R decided on
    (``recorded``, from `rule_r_runs`); else the newest run at the config's current hash (seed 42 for `SEED42`, from
    `LONG_SINCE` on for a long rung)."""
    if name in pinned:
        return find_run(roots, pinned[name], by_name=False)
    if name in recorded:
        d = find_run(roots, recorded[name], by_name=False)
        if d is not None:
            return d
        print(f"op rung {name}: rule R's run {recorded[name]} not found under {roots}; the newest run instead")
    return find_run(roots, name, by_name=True, expected_hash=config_hash(name),
                    since=LONG_SINCE if name.startswith("op_hl") else None, seed=42 if name in SEED42 else None)


def config_hash(name: str) -> str | None:
    """The current hash of ``configs/<name>.yaml`` (None when the config does not load)."""
    try:
        return load_config(f"{name}.yaml").hash()
    except Exception as exc:  # noqa: BLE001 -- reported; the rung is then matched by name only
        print(f"op rung {name}: config not loadable ({exc}); matched by name only")
        return None


def best_iterations(run: Path) -> str | None:
    """The fold best iteration(s) the run logged (``<fold>: best iteration N``), comma-separated."""
    try:
        text = (run / "log.txt").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    found = re.findall(r"best iteration (\d+)", text)
    return ", ".join(found) if found else None


def read_oof(run: Path) -> pd.DataFrame:
    oof = pd.read_parquet(run / "oof_predictions.parquet")
    if "mvt_id" in oof.columns:
        oof = oof.set_index("mvt_id")
    return oof


def holdout_context(ids: pd.Index, disrupted: pd.Series | None) -> pd.DataFrame:
    """apt, month, NM match, day-offset, disruption and H1-early flags for the scored rows."""
    mvt = data.load_movements(columns=MVT_COLUMNS).reindex(ids)
    y = data.load_target().reindex(ids)
    a = _oplib.anchor_h1(mvt)
    block = _oplib.load_offblocks().reindex(ids).to_numpy(dtype="float64")
    with np.errstate(invalid="ignore"):
        early = np.isfinite(block) & np.isfinite(a) & (block < a)
    if disrupted is None:
        from prc import disruption

        full = data.load_movements(columns=list(disruption.MVT_COLUMNS))
        disrupted = disruption.disrupted_rows(full, ids)
    return pd.DataFrame(
        {
            "y_true": y,
            "apt": mvt["apt"].astype(str),
            "month": mvt["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m"),
            "day": mvt["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d"),
            "nm_matched": mvt[metrics.NM_MATCH_COLUMN].notna(),
            "dayoffset": cleaning.day_offset_rows(mvt, y).reindex(ids).fillna(False).astype(bool),
            "disrupted": disrupted.reindex(ids).fillna(False).astype(bool),
            "h1_early": pd.Series(early, index=ids),
        },
        index=ids,
    )


# --- table baselines computed here ----------------------------------------------------------------

B1_CHAIN = (("apt", "stand", "runway"), ("apt", "stand"), ("apt", "runway"), ("apt",))
"""B1's lookup hierarchy (``prc.models.baselines.GroupMedian``)."""


def _key(frame: pd.DataFrame, cols) -> pd.Series:
    parts = [frame[c].astype("string").fillna("__NA__") for c in cols]
    out = parts[0]
    for p in parts[1:]:
        out = out + "\x1f" + p
    return out


def chain_quantile(fit: pd.DataFrame, query: pd.DataFrame, levels, quantile: float = 0.5,
                   min_support: int = MIN_SUPPORT) -> pd.Series:
    """The ``quantile`` of ``fit['y']`` per key, first key level with at least ``min_support`` rows wins.

    The same hierarchy as ``GroupMedian`` (whose median this reproduces at ``quantile=0.5``); the last resort is the
    quantile over every fit row.
    """
    out = pd.Series(np.nan, index=query.index, dtype="float64")
    for cols in levels:
        need = out.isna().to_numpy()
        if not need.any():
            break
        g = fit["y"].groupby(_key(fit, cols))
        table = g.quantile(quantile)[g.size() >= min_support]
        out.iloc[need] = _key(query.loc[need], cols).map(table).to_numpy(dtype="float64", na_value=np.nan)
    return out.fillna(float(fit["y"].quantile(quantile)))


def flight_history_mean(frame: pd.DataFrame, ids: pd.Index, max_target: float = FIT_MAX_TARGET_S,
                        days: int = HISTORY_DAYS, min_obs: int = 1) -> pd.Series:
    """Mean target of the same (airport, flight number) over takeoffs in [anchor - ``days``, anchor).

    History is every row of ``frame`` with a finite target below ``max_target`` (holdout days included: an
    earlier day's realised taxi is known at the anchor); the query row itself never counts. NaN with fewer than
    ``min_obs`` earlier occurrences or no flight number.
    """
    h = frame[frame["y"].notna() & (frame["y"] < max_target) & frame["flight"].notna() & np.isfinite(frame["t"])]
    q = frame.reindex(ids)
    key_h = h["apt"].astype(str) + "\x1f" + h["flight"].astype(str)
    codes, uniques = pd.factorize(key_h)
    key_q = q["apt"].astype(str) + "\x1f" + q["flight"].astype(str)
    code_q = pd.Index(uniques).get_indexer(key_q)
    a = q["anchor"].to_numpy(dtype="float64")
    ok = (code_q >= 0) & np.isfinite(a) & q["flight"].notna().to_numpy()
    scale = np.int64(10**10)  # above any epoch second of the data
    th = h["t"].to_numpy(dtype="float64").astype("int64")
    comp = codes.astype("int64") * scale + th
    order = np.argsort(comp, kind="stable")
    comp = comp[order]
    cum = np.concatenate([[0.0], np.cumsum(h["y"].to_numpy(dtype="float64")[order])])
    base = np.where(ok, code_q, 0).astype("int64") * scale
    a_i = np.where(ok, np.floor(a), 0).astype("int64")
    lo = np.searchsorted(comp, base + a_i - days * 86400, side="left")
    hi = np.searchsorted(comp, base + a_i, side="left")
    tot = cum[hi] - cum[lo]
    n = (hi - lo).astype("float64")
    # the query row itself, when it is a history row inside its own window (a takeoff before its own H1)
    qt = q["t"].to_numpy(dtype="float64")
    qy = q["y"].to_numpy(dtype="float64")
    with np.errstate(invalid="ignore"):
        self_in = ok & ids.isin(h.index) & (qt >= a_i - days * 86400) & (qt < a_i)
    tot = tot - np.where(self_in, qy, 0.0)
    n = n - self_in
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(ok & (n >= max(min_obs, 1)), tot / np.where(n > 0, n, 1.0), np.nan)
    return pd.Series(mean, index=ids, dtype="float64")


def baseline_frame() -> pd.DataFrame:
    """2025 training departures with what the table baselines need (reads the movement frame and the target)."""
    from prc.features.temporal import local_naive

    mvt = data.load_movements(columns=BASELINE_COLUMNS)
    dep = mvt[(mvt["split"] == "train").to_numpy() & mvt["is_dep"].astype(bool).to_numpy()]
    planned = _oplib.planned_offblock(dep)
    hour = local_naive(_oplib.to_timestamps(planned, dep.index), dep["apt"].astype(str)).dt.hour
    return pd.DataFrame(
        {
            "apt": dep["apt"].astype(str),
            "stand": dep["STAND_mvt"].astype("string"),
            "runway": dep["RUNWAY_mvt"].astype("string"),
            "flight": dep["FLIGHT_mvt"].astype("string"),
            "hour": hour.astype("Int64"),
            "t": _oplib.secs(dep["MVT_TIME_UTC_mvt"]),
            "anchor": _oplib.anchor_h1(dep),
            "month": dep["MVT_TIME_UTC_mvt"].dt.month.to_numpy(),
            "y": data.load_target().reindex(dep.index),
        },
        index=dep.index,
    )


def table_baselines(frame: pd.DataFrame, ids: pd.Index, holdout_months=HOLDOUT_MONTHS,
                    max_target: float = FIT_MAX_TARGET_S, min_support: int = MIN_SUPPORT,
                    days: int = HISTORY_DAYS) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """B1-fit, B-HAT and B3 predictions for ``ids`` (as OOF frames), and the stand x runway unimpeded reference.

    ``frame`` (``baseline_frame``): apt, stand, runway, flight, hour, t (takeoff), anchor (H1), month, y. The fit
    rows are those with a target below ``max_target`` outside ``holdout_months``; B3's history is any earlier row.
    """
    fit = frame[frame["y"].notna() & (frame["y"] < max_target) & ~frame["month"].isin(holdout_months)]
    q = frame.reindex(ids)
    b1 = chain_quantile(fit, q, B1_CHAIN, 0.5, min_support)
    bhat = chain_quantile(fit, q, (("apt", "stand", "hour"), *B1_CHAIN), 0.5, min_support)
    b3 = flight_history_mean(frame, ids, max_target, days).fillna(b1)
    unimpeded = chain_quantile(fit, q, B1_CHAIN, UNIMPEDED_QUANTILE, min_support)
    y = q["y"].astype("float64")
    preds = {"B1-fit": b1, "B-HAT": bhat, "B3": b3}
    return {k: pd.DataFrame({"y_true": y, "y_pred": v}, index=ids) for k, v in preds.items()}, unimpeded


def unimpeded_table(ref: pd.Series, ctx: pd.DataFrame) -> pd.DataFrame:
    """Per airport on OP: the target's median, the stand x runway p20 reference and the additional time over it."""
    frame = ctx.reindex(ref.index).copy()
    frame["ref"] = ref
    f = frame[population_masks(frame)["OP"].fillna(False).to_numpy(dtype=bool) & frame["ref"].notna().to_numpy()]
    f = f.assign(add=f["y_true"] - f["ref"])
    out = f.groupby("apt").agg(n=("y_true", "size"), median_target=("y_true", "median"), mean_ref=("ref", "mean"),
                               mean_additional=("add", "mean"), median_additional=("add", "median"))
    total = pd.DataFrame({"n": [len(f)], "median_target": [f["y_true"].median()], "mean_ref": [f["ref"].mean()],
                          "mean_additional": [f["add"].mean()], "median_additional": [f["add"].median()]},
                         index=pd.Index(["all"], name="apt"))
    return pd.concat([out, total])


# --- the report -----------------------------------------------------------------------------------

def score(label: str, oof: pd.DataFrame, ctx: pd.DataFrame) -> list[dict]:
    rows = []
    frame = ctx.reindex(oof.index)
    frame["y_pred"] = oof["y_pred"]
    frame["y_true"] = oof["y_true"]
    masks = population_masks(frame)
    if label in H0_RUNGS:
        masks = {k: v for k, v in masks.items() if k in H0_POPULATIONS}
    for pop, m in masks.items():
        f = frame.loc[m.fillna(False).to_numpy(dtype=bool)]
        rows.append({"row": label, "population": pop, "slice": "all", "level": "all", **metric_row(f["y_true"], f["y_pred"])})
        if pop in ("OP", "TRIM"):
            for col in (("apt", "month") if pop == "OP" else ("apt",)):
                for level, g in f.groupby(col):
                    rows.append({"row": label, "population": pop, "slice": col, "level": level, **metric_row(g["y_true"], g["y_pred"])})
    return rows


def tripwire(table: pd.DataFrame) -> list[str]:
    out = []
    op = table[(table["population"] == "OP") & (table["slice"] == "all")].set_index("row")
    if "C1" not in op.index:
        return ["C1 (exp066) not found: tripwire not evaluated."]
    c1 = op.loc["C1"]
    for name, *_ in OP_CONFIGS:
        if (name not in op.index or not name.startswith("op_h1")
                or (name in OTHER_HOLDOUT and name not in OWN_TABLE_JANJUL) or name in NOT_IN_TRIPWIRE):
            continue
        r = op.loc[name]
        leak = r["mae"] <= c1["mae"] or r["within_120"] >= c1["within_120"]
        out.append(f"- {name}: OP MAE {r['mae']:.1f} s vs C1 {c1['mae']:.1f} s, +-2 min {100 * r['within_120']:.1f} % vs "
                   f"{100 * c1['within_120']:.1f} % -> {'LEAK until explained' if leak else 'ok (below C1)'}")
    apt = table[(table["population"] == "OP") & (table["slice"] == "apt")]
    if "op_h0_r5" in op.index:
        h0 = op.loc["op_h0_r5"]
        h0_apt = apt[apt["row"] == "op_h0_r5"].set_index("level")["mae"]
        for name in ("op_h1_r4", "op_h1_r5", "op_h1_r4_rwyblind"):
            if name not in op.index:
                continue
            r = op.loc[name]
            r_apt = apt[apt["row"] == name].set_index("level")["mae"]
            better = sorted(lvl for lvl in r_apt.index if lvl in h0_apt.index and r_apt[lvl] < h0_apt[lvl])
            if r["mae"] <= h0["mae"]:
                verdict = "LEAK until explained (an H1 rung at or better than everything at pushback)"
            elif better:
                verdict = "FLAG: better than H0 at " + ", ".join(better)
            else:
                verdict = "ok"
            out.append(f"- {name} vs op_h0_r5 (everything at pushback): OP MAE {r['mae']:.1f} s vs {h0['mae']:.1f} s -> {verdict}")
    if {"op_h0_r5", "C2"} <= set(apt["row"]):
        h0 = apt[apt["row"] == "op_h0_r5"].set_index("level")["mae"]
        c2 = apt[apt["row"] == "C2"].set_index("level")["mae"]
        beats = (h0 < c2.reindex(h0.index)).all()
        out.append(f"- op_h0_r5: beats C2 (v32) on OP MAE at {(h0 < c2.reindex(h0.index)).sum()} of {len(h0)} airports -> "
                   f"{'LEAK until explained' if beats else 'ok'}")
    return out


def run_info(run: Path) -> dict:
    info = {}
    try:
        info["n_features"] = json.loads((run / "metrics.json").read_text()).get("n_features")
    except (OSError, ValueError):
        pass
    return info


def fmt(v, pct=False, nd=1) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return f"{100 * v:.{nd}f} %" if pct else f"{v:,.{nd}f}"


LADDER_HEADER = (
    "| Row | Run | Tiers | Features | Best iter. | OP n | OP RMSE | OP MAE | OP MedAE | OP bias | OP MAE/SD | OP RMSE/SD | "
    "+-2 min | +-5 min | +-7 min | BULK RMSE | EX_WD RMSE | ALL RMSE | H1_SENS MAE |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
)


def _ladder_lines(allrows: pd.DataFrame, meta: dict, labels: list[str]) -> list[str]:
    lines = list(LADDER_HEADER)
    for label in labels:
        m = meta[label]
        sub = allrows[allrows["row"] == label].set_index("population")
        if "OP" not in sub.index:
            continue
        v = lambda pop, col: float(sub.loc[pop, col]) if pop in sub.index and col in sub.columns else np.nan  # noqa: E731
        lines.append(
            f"| {m['title']} | `{m['run']}` | {m['tiers']} | {m.get('n_features') or '-'} | {m.get('best_iteration') or '-'} | "
            f"{fmt(v('OP', 'n'), nd=0)} | {fmt(v('OP', 'rmse'))} | {fmt(v('OP', 'mae'))} | "
            f"{fmt(v('OP', 'medae'))} | {fmt(v('OP', 'bias'))} | {fmt(v('OP', 'mae_sd'), nd=3)} | {fmt(v('OP', 'rmse_sd'), nd=3)} | "
            f"{fmt(v('OP', 'within_120'), True)} | {fmt(v('OP', 'within_300'), True)} | "
            f"{fmt(v('OP', 'within_420'), True)} | {fmt(v('BULK', 'rmse'))} | {fmt(v('EX_WD', 'rmse'))} | {fmt(v('ALL', 'rmse'))} | "
            f"{fmt(v('H1_SENS', 'mae'))} |"
        )
    return lines


def _ci(st: dict, key: str, lo: str, hi: str, scale: float = 1.0) -> tuple[str, str]:
    """``(change, [lo, hi])`` formatted, or dashes when the step carries no such key."""
    if key not in st:
        return "-", "-"
    return f"{scale * st[key]:+.2f}", f"[{scale * st[lo]:+.2f}, {scale * st[hi]:+.2f}]"


def _step_cells(st: dict) -> str:
    cells = [*_ci(st, "d_mae", "mae_lo", "mae_hi"), *_ci(st, "d_w120", "w120_lo", "w120_hi", 100.0),
             *_ci(st, "d_bias", "bias_lo", "bias_hi")]
    return " | ".join(cells)


STEP_HEADER = "OP MAE change (s) | 95 % CI | +-2 min change (pp) | 95 % CI | mean signed error change (s) | 95 % CI |"


def deliverable_lines(seed_frames: dict, ctx: pd.DataFrame, calib: dict, pairs: dict, oofs: dict) -> list[str]:
    """The deliverables' sections: the seeds (the floor), rule C's calibration, the twelve months."""
    lines: list[str] = []
    if seed_frames:
        lines += ["", "## The deliverables' seeds (OP; the spread is the seed floor)", "",
                  "| Deliverable | run | OP MAE | OP RMSE | +-2 min |", "|---|---|---|---|---|"]
        for label, items in seed_frames.items():
            maes = []
            for d, f in items:
                fr = op_frame(f, ctx)
                fr = fr[fr["op"]]
                r = metric_row(fr["y_true"], fr["y_pred"])
                maes.append(r["mae"])
                lines.append(f"| {label} | `{d.name}` | {fmt(r['mae'], nd=2)} | {fmt(r['rmse'], nd=2)} | "
                             f"{fmt(r['within_120'], True, 2)} |")
            fr = op_frame(oofs[label], ctx)
            fr = fr[fr["op"]]
            r = metric_row(fr["y_true"], fr["y_pred"])
            lines.append(f"| {label} | the mean of the seeds | {fmt(r['mae'], nd=2)} | {fmt(r['rmse'], nd=2)} | "
                         f"{fmt(r['within_120'], True, 2)} |")
            lines.append(f"| {label} | spread of the seeds' MAE (max - min) | {fmt(max(maes) - min(maes), nd=2)} | | |")
    if calib:
        lines += ["", "## Rule C: the calibrated median and the P10-P90 band (tables from the 2025 month pairs only)", "",
                  "| Deliverable | MAE mean -> median | +-2 min mean -> median | worst airport change | P10-P90 coverage | "
                  "below P10 | above P90 | median width | rule C |", "|---|---|---|---|---|---|---|---|---|"]
        for label, c in calib.items():
            verdict = "passed" if c["passed"] else "not passed: mean only"
            lines.append(f"| {label} | {c['mean']['mae']:.2f} -> {c['median']['mae']:.2f} s | "
                         f"{100 * c['mean']['within_120']:.2f} -> {100 * c['median']['within_120']:.2f} % | "
                         f"{c['worst_airport_delta']:+.2f} s | {100 * c['coverage_p10_p90']:.1f} % | "
                         f"{100 * c['below_p10']:.1f} % | {100 * c['above_p90']:.1f} % | {c['median_width_s']:.0f} s | "
                         f"{verdict} |")
        lines += ["", "Coverage of the P10-P90 band per airport:", "",
                  "| airport | " + " | ".join(calib) + " |", "|---" * (len(calib) + 1) + "|"]
        apts = sorted(set().union(*(c["airports"] for c in calib.values())))
        for a in apts:
            cells = [f"{100 * c['airports'][a]['coverage']:.1f} %" if a in c["airports"] else "-" for c in calib.values()]
            lines.append(f"| {a} | " + " | ".join(cells) + " |")
    rule_m = {label: c["rule_m"] for label, c in calib.items() if c.get("rule_m")}
    if rule_m:
        lines += ["", "## Rule M: the median member (MAE loss, exact leaves) in place of the calibrated median", "",
                  "Where rule C passed, the file's median is the calibrated median, or, where rule M also passes, the "
                  "member's prediction clamped into the calibrated P10-P90 band; the mean and the band are unchanged. "
                  "Where rule C did not pass, the file holds the mean only (`lab/op_deliver.py` refuses "
                  "`--median-config` without the band). The clamped share counts every holdout row.", "",
                  "| Deliverable | median | run | OP MAE | OP RMSE | +-2 min | clamped into the band |",
                  "|---|---|---|---|---|---|---|"]
        for label, m in rule_m.items():
            for row, what, run, share in ((f"{label}-MED", "calibrated median (rule C)", "calibration tables", ""),
                                          (m["member"], "the member, unclamped", f"`{m['run']}`", ""),
                                          (f"{label}-MAE", "the member clamped: the file's median if rules C and M "
                                           "passed", f"`{m['run']}`", f"{100 * m['clamped_share']:.2f} %")):
                if row not in oofs:
                    continue
                fr = op_frame(oofs[row], ctx)
                fr = fr[fr["op"]]
                r = metric_row(fr["y_true"], fr["y_pred"])
                lines.append(f"| {label} | {what} | {run} | {fmt(r['mae'], nd=2)} | {fmt(r['rmse'], nd=2)} | "
                             f"{fmt(r['within_120'], True, 2)} | {share} |")
    if any(pairs.values()):
        labels = [k for k, v in pairs.items() if v]
        months = sorted(set().union(*(v.keys() for v in pairs.values())) | {"2025-01", "2025-07"})
        lines += ["", "## Every month of 2025, each predicted by a model fitted on the other ten (OP MAE in s / within "
                  "+-2 min)", "",
                  "The month pairs (Feb+Dec, Mar+Nov, Apr+Oct, May+Sep, Jun+Aug) are single seeds of each deliverable's "
                  "configuration; January and July are the seasonal holdout of the deliverable row (the seed mean). "
                  "February and June are the analogues of the final set's new months.", "",
                  "| month | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
        for m in months:
            cells = []
            for label in labels:
                v = pairs[label].get(m)
                if v is None and m in ("2025-01", "2025-07") and label in oofs:
                    fr = op_frame(oofs[label], ctx)
                    keep = fr["op"].to_numpy() & (ctx.reindex(fr.index)["month"] == m).to_numpy()
                    fr = fr[keep]
                    r = metric_row(fr["y_true"], fr["y_pred"])
                    v = {"mae": r["mae"], "within_120": r["within_120"]}
                cells.append("-" if v is None else f"{v['mae']:.1f} / {100 * v['within_120']:.1f} %")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return lines


def markdown(table: pd.DataFrame, meta: dict, trip: list[str], steps: list[dict] | None = None,
             warnings: list[str] | None = None, apt_steps: list[dict] | None = None,
             unimpeded: pd.DataFrame | None = None, extra: list[str] | None = None) -> str:
    lines = ["# Operational ladder: holdout read-out", "",
             "Generated by `lab/op_ladder_report.py` from each run's `oof_predictions.parquet`. Op rungs: one global "
             "CatBoost each, trained on the other ten months (seasonal Jan+Jul 2025 holdout). "
             "C2 and C3 are challenge reconstructions, tuned or cross-fitted on the holdout: not like-for-like.", "",
             "Early stopping: each op rung stops on the holdout rows with target <= 7,200 s (`model.eval_max_target_sec`, "
             "the population its fit keeps), C1's regressor on holdout rows <= 7,200 s that are not stale-labelled; "
             "every holdout row is scored either way. `Best iter.` shows where each run stopped (8,000 = the cap). "
             "H0 (pushback) rows are scored on OP and H1_SENS only.", "",
             "B1-fit, B-HAT and B3 are table baselines computed by this script from the 2025 departures with target "
             "< 7,200 s outside January and July (B3's flight history also uses holdout days before the anchor). "
             "`op_h0_r5_hind` is a LAB HINDSIGHT rung (the traffic during the flight's own taxi), not an operational "
             "result; `op_h1_r4_mae` is r4 with MAE loss, read for MAE and the tolerance shares only. MAE/SD and "
             "RMSE/SD divide by the target's SD on the same slice.", ""]
    if warnings:
        lines += ["**Warnings**", "", *[f"- {w}" for w in warnings], ""]
    allrows = table[table["slice"] == "all"]
    main_labels = [k for k in meta if k not in OTHER_HOLDOUT]
    lines += ["## Ladder (Jan+Jul 2025 holdout; OP population: target < 3,600 s and NM-matched)", "",
              *_ladder_lines(allrows, meta, main_labels)]
    for title, names in OTHER_TABLES:
        other = [k for k in meta if k in names]
        if other:
            lines += ["", f"## {title}", "", *_ladder_lines(allrows, meta, other)]
    if extra:
        lines += extra
    if steps:
        lines += ["", "## Rung steps (paired on identical OP rows; 95 % day-block bootstrap)", "",
                  "| From | To | n | days | " + STEP_HEADER,
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for st in steps:
            lines.append(f"| {st['from']} | {st['to']} | {st['n']:,} | {st['days']} | {_step_cells(st)} |")
    if apt_steps:
        lines += ["", "## Rung steps per airport (paired OP rows of that airport; 95 % day-block bootstrap)", "",
                  "| From | To | Airport | n | days | " + STEP_HEADER,
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for st in apt_steps:
            lines.append(f"| {st['from']} | {st['to']} | {st['apt']} | {st['n']:,} | {st['days']} | {_step_cells(st)} |")
    lines += ["", "## Every population", "",
              "| Row | Population | n | RMSE | MAE | MedAE | bias | P90 | SD | MAE/SD | RMSE/SD | "
              + " | ".join(f"+-{w} s" for w in WITHIN) + " |",
              "|---" * (11 + len(WITHIN)) + "|"]
    for label in meta:
        for pop in POPULATIONS:
            if label in H0_RUNGS and pop not in H0_POPULATIONS:
                continue
            r = allrows[(allrows["row"] == label) & (allrows["population"] == pop)]
            if r.empty:
                continue
            r = r.iloc[0]
            g = lambda c: r[c] if c in r.index else np.nan  # noqa: E731
            lines.append(f"| {label} | {pop} | {int(r['n']):,} | {fmt(r['rmse'])} | {fmt(r['mae'])} | {fmt(r['medae'])} | {fmt(r['bias'])} | "
                         f"{fmt(r['p90_abs'])} | {fmt(g('sd_y'))} | {fmt(g('mae_sd'), nd=3)} | {fmt(g('rmse_sd'), nd=3)} | "
                         + " | ".join(fmt(g(f'within_{w}'), True) for w in WITHIN) + " |")
    for col, title in (("apt", "airport"), ("month", "month")):
        sub = table[(table["population"] == "OP") & (table["slice"] == col)]
        if sub.empty:
            continue
        stats = [("mae", "MAE (s)", False, 1), ("within_120", "share within +-2 min", True, 1)]
        if col == "apt" and "mae_sd" in sub.columns:
            stats += [("sd_y", "target SD (s)", False, 1), ("mae_sd", "MAE/SD", False, 3), ("rmse_sd", "RMSE/SD", False, 3)]
        for stat, name, pct, nd in stats:
            piv = sub.pivot_table(index="level", columns="row", values=stat, aggfunc="first")
            piv = piv[[c for c in meta if c in piv.columns]]
            lines += ["", f"## OP {name} per {title}", "", "| " + title + " | " + " | ".join(piv.columns) + " |",
                      "|---" * (len(piv.columns) + 1) + "|"]
            for level, row in piv.iterrows():
                lines.append(f"| {level} | " + " | ".join(fmt(x, pct, nd) for x in row.to_numpy()) + " |")
        if col == "apt":
            piv = sub.pivot_table(index="level", columns="row", values="mae", aggfunc="first")
            if "B1" in piv.columns:
                cols = [c for c in meta if c in piv.columns and c != "B1"]
                d = piv[cols].sub(piv["B1"], axis=0)
                lines += ["", "## OP MAE minus B1 per airport (s; negative = better than the stand x runway table)", "",
                          "| airport | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
                for level, row in d.iterrows():
                    lines.append(f"| {level} | "
                                 + " | ".join("-" if not np.isfinite(x) else f"{x:+.1f}" for x in row.to_numpy()) + " |")
    if unimpeded is not None and len(unimpeded):
        lines += ["", f"## Unimpeded reference (not a scored row): stand x runway {round(100 * UNIMPEDED_QUANTILE)}th "
                  "percentile of the fit rows, on OP", "",
                  "Additional time = target minus the row's reference (the PRU split into unimpeded and additional "
                  "time, here from our own fit rows). As a predictor a low quantile is biased low by construction.", "",
                  "| airport | n | median target (s) | mean reference (s) | mean additional (s) | median additional (s) |",
                  "|---|---|---|---|---|---|"]
        for level, r in unimpeded.iterrows():
            lines.append(f"| {level} | {int(r['n']):,} | {fmt(r['median_target'])} | {fmt(r['mean_ref'])} | "
                         f"{fmt(r['mean_additional'])} | {fmt(r['median_additional'])} |")
    lines += ["", "## Comparability rows (same population definition, different protocol)", "",
              "| Row | EDDF +-2 min (OP) | vs Guenther 2011 (abstract, not verified) | LSZH MAE (TRIM) | LTFM MAE (TRIM) | "
              "vs Swatowska 2025 (Table V) | OP MAE/SD | vs Idris 2002 (Table 2) |",
              "|---|---|---|---|---|---|---|---|"]
    for label in meta:
        def at(pop, apt, col):
            r = table[(table["row"] == label) & (table["population"] == pop) & (table["slice"] == "apt") & (table["level"] == apt)]
            return float(r[col].iloc[0]) if len(r) else np.nan

        r_op = allrows[(allrows["row"] == label) & (allrows["population"] == "OP")]
        mae_sd = float(r_op["mae_sd"].iloc[0]) if len(r_op) and "mae_sd" in r_op.columns else np.nan
        lines.append(f"| {label} | {fmt(at('OP', 'EDDF', 'within_120'), True)} | 42 % static / 66 % at off-block | "
                     f"{fmt(at('TRIM', 'LSZH', 'mae'))} s | {fmt(at('TRIM', 'LTFM', 'mae'))} s | "
                     f"LSZH 138 s (table 214 s), LTFM 212 s (table 225 s) | {fmt(mae_sd, nd=3)} | "
                     "0.41 queue at pushback, 0.52 flight-history mean |")
    lines += ["", "## Tripwire", "", *trip, "", "## Literature anchors (fixed text)", "", LITERATURE, ""]
    return "\n".join(lines)


def paired_step(a: pd.DataFrame, b: pd.DataFrame, ctx: pd.DataFrame, n_boot: int = 1000, seed: int = 0,
                apt: str | None = None) -> dict | None:
    """OP MAE, +-2 min share and bias of ``b`` minus ``a`` on their common OP rows, with a day-block bootstrap 95 % CI.

    Days are resampled with replacement; the pairing keeps both runs' errors on a row together. The bias change is
    the mean signed error change (e_b - e_a, e = prediction - target), negative when ``b`` predicts lower. With
    ``apt`` only that airport's OP rows (and days) take part.
    """
    if "day" not in ctx.columns:
        return None
    ids = a.index.intersection(b.index)
    frame = ctx.reindex(ids)
    frame["y_true"] = a.loc[ids, "y_true"]
    op = population_masks(frame)["OP"].fillna(False).to_numpy(dtype=bool, copy=True)
    if apt is not None:
        op &= (frame["apt"] == apt).to_numpy(dtype=bool)
    ids = ids[op]
    if not len(ids):
        return None
    y = a.loc[ids, "y_true"].to_numpy(dtype="float64")
    sa = a.loc[ids, "y_pred"].to_numpy(dtype="float64") - y
    sb = b.loc[ids, "y_pred"].to_numpy(dtype="float64") - y
    ea, eb = np.abs(sa), np.abs(sb)
    per = pd.DataFrame({"day": ctx.loc[ids, "day"].to_numpy(), "d_abs": eb - ea,
                        "d_w": (eb <= 120).astype(float) - (ea <= 120).astype(float), "n": 1.0,
                        "d_signed": sb - sa}).groupby("day").sum()
    s = per[["d_abs", "d_w", "n", "d_signed"]].to_numpy()
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(per), np.full(len(per), 1.0 / len(per)), size=n_boot).astype("float64")
    tot = w @ s  # (n_boot, 4): sum of |e| changes, sum of within-2-min changes, rows, sum of signed changes
    d_mae, d_w, d_b = tot[:, 0] / tot[:, 2], tot[:, 1] / tot[:, 2], tot[:, 3] / tot[:, 2]
    n_rows = s[:, 2].sum()
    return {"n": int(len(ids)), "days": int(len(per)), "d_mae": float(s[:, 0].sum() / n_rows),
            "mae_lo": float(np.quantile(d_mae, 0.025)), "mae_hi": float(np.quantile(d_mae, 0.975)),
            "d_w120": float(s[:, 1].sum() / n_rows),
            "w120_lo": float(np.quantile(d_w, 0.025)), "w120_hi": float(np.quantile(d_w, 0.975)),
            "d_bias": float(s[:, 3].sum() / n_rows),
            "bias_lo": float(np.quantile(d_b, 0.025)), "bias_hi": float(np.quantile(d_b, 0.975))}


def rung_steps(oofs: dict[str, pd.DataFrame], ctx: pd.DataFrame, n_boot: int = 1000) -> tuple[list[dict], list[dict]]:
    """Every STEPS pair present in ``oofs``: the overall paired step and one per airport (``apt`` key)."""
    overall, per_apt = [], []
    for a_, b_ in STEPS:
        if a_ not in oofs or b_ not in oofs:
            continue
        st = paired_step(oofs[a_], oofs[b_], ctx, n_boot=n_boot)
        if st is None:
            continue
        overall.append({"from": a_, "to": b_, **st})
        common = oofs[a_].index.intersection(oofs[b_].index)
        for apt in sorted(ctx.reindex(common)["apt"].dropna().unique()):
            sa = paired_step(oofs[a_], oofs[b_], ctx, n_boot=n_boot, apt=apt)
            if sa is not None:
                per_apt.append({"from": a_, "to": b_, "apt": apt, **sa})
    return overall, per_apt


def row_set_warnings(oofs: dict[str, pd.DataFrame]) -> list[str]:
    """Every Jan+Jul op rung must have scored exactly C1's holdout rows."""
    if "C1" not in oofs:
        return ["C1 not found: the op rungs' scored rows were not checked against it."]
    ref = oofs["C1"].index
    out = []
    for name, *_ in OP_CONFIGS:
        if (name in OTHER_HOLDOUT and name not in OWN_TABLE_JANJUL) or name not in oofs:
            continue
        idx = oofs[name].index
        if not idx.sort_values().equals(ref.sort_values()):
            out.append(f"{name} scored {len(idx):,} rows, C1 {len(ref):,} ({len(idx.difference(ref)):,} not in C1, "
                       f"{len(ref.difference(idx)):,} of C1 missing): not the same holdout rows")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs-root", action="append", type=Path, help="run directories to search (default: runs/ of this checkout)")
    ap.add_argument("--run", action="append", default=[], help="pin a run: <config name or comparison label>=<run id>")
    ap.add_argument("--out", type=Path, default=paths.ROOT / "docs" / "operational_ladder")
    ap.add_argument("--tables", nargs="*", default=[],
                    help="MEMBER=FILE, MEMBER one of " + ", ".join(DEFAULT_TABLES) + ": calibration tables (default "
                    "runs/op_calib/..., see DEFAULT_TABLES)")
    ap.add_argument("--no-table-baselines", action="store_true",
                    help="skip B1-fit, B-HAT, B3 and the unimpeded reference (they read the whole 2025 departure frame)")
    args = ap.parse_args(argv)
    roots = [r.resolve() for r in (args.runs_root or [paths.RUNS])]
    pinned = dict(item.split("=", 1) for item in args.run)
    args.tables = dict(item.split("=", 1) for item in args.tables)

    runs: dict[str, dict] = {}
    for label, run_id, title, tiers, protocol in COMPARISON:
        d = find_run(roots, pinned.get(label, run_id), by_name=False)
        if d is None:
            print(f"comparison {label}: {run_id} not found under {roots}; skipped")
            continue
        runs[label] = {"dir": d, "run": d.name, "title": title, "tiers": tiers, "protocol": protocol}
    recorded = rule_r_runs()
    for name, title, tiers in OP_CONFIGS:
        d = rung_dir(roots, name, pinned, recorded)
        if d is None:
            print(f"op rung {name}: no run found; skipped")
            continue
        runs[name] = {"dir": d, "run": d.name, "title": title, "tiers": tiers, "protocol": "trained on the other ten months",
                      **run_info(d)}
    for v in runs.values():
        v["best_iteration"] = best_iterations(v["dir"])
    if not runs:
        print("nothing to report")
        return 1

    oofs = {k: read_oof(v["dir"]) for k, v in runs.items()}
    seed_frames: dict[str, list] = {}
    for label, member, n, title in DELIVERABLES:
        cfg = deliverable_config(member)
        dirs = seed_runs(roots, cfg, n, since=LONG_SINCE if cfg.startswith("op_hl") else None)
        if not dirs:
            print(f"deliverable {label}: fewer than {n} seed runs of {cfg}; skipped")
            continue
        oofs[label], frames = seed_mean(dirs)
        seed_frames[label] = list(zip(dirs, frames))
        runs[label] = {"run": ", ".join(d.name for d in dirs), "title": f"{title}: `{cfg}`, {n} seeds averaged (mean)",
                       "tiers": DELIVERABLE_TIERS[member], "protocol": "trained on the other ten months", "member": member,
                       "config": cfg, "best_iteration": None}
    ids = pd.Index(sorted(set().union(*(o.index for o in oofs.values()))), name="mvt_id")
    disrupted = next((o["disrupted"] for o in oofs.values() if "disrupted" in o.columns), None)
    if disrupted is not None:
        # the column covers one holdout; rows of another holdout (winter) are computed
        missing = ids.difference(disrupted.index)
        if len(missing):
            from prc import disruption

            full = data.load_movements(columns=list(disruption.MVT_COLUMNS))
            disrupted = pd.concat([disrupted, disruption.disrupted_rows(full, missing)])
    ctx = holdout_context(ids, disrupted)

    calib: dict[str, dict] = {}
    for label, member, n, title in DELIVERABLES:
        tpath = Path(args.tables.get(member, DEFAULT_TABLES[member]))
        if not tpath.is_absolute():
            tpath = paths.ROOT / tpath
        if label not in oofs:
            continue
        if not tpath.exists():
            print(f"deliverable {label}: no calibration tables at {tpath}; no {label}-MED or {label}-MAE row")
            continue
        import op_calibrate  # noqa: E402  (lab/, beside this script)

        tables = pd.read_parquet(tpath)
        frame = op_frame(oofs[label], ctx)
        out = op_calibrate.apply_tables(tables, frame["y_pred"].to_numpy(), frame["apt"].astype(str).to_numpy())
        med = oofs[label][["y_true"]].copy()
        med["y_pred"] = out["y_med"].to_numpy()
        oofs[f"{label}-MED"] = med
        runs[f"{label}-MED"] = {**runs[label], "title": runs[label]["title"].replace("(mean)", "(calibrated median)")}
        calib[label] = {"tables": str(tpath), **op_calibrate.rule_c(tables, frame)}
        mm = MEDIAN_MEMBER.get(member)
        if mm in oofs:
            raw = oofs[mm]["y_pred"].reindex(oofs[label].index).to_numpy(dtype="float64")
            if not np.isfinite(raw).all():
                print(f"rule M: {mm} did not score every row of {label}; no {label}-MAE row")
                continue
            clamped = np.maximum(np.clip(raw, out["y_p10"].to_numpy(), out["y_p90"].to_numpy()), op_calibrate.FLOOR_S)
            med_m = oofs[label][["y_true"]].copy()
            med_m["y_pred"] = clamped
            oofs[f"{label}-MAE"] = med_m
            runs[f"{label}-MAE"] = {**runs[mm], "member": member,
                                    "title": f"{title}: rule M's median, `{mm}` clamped into the calibrated P10-P90 band"}
            calib[label]["rule_m"] = {"member": mm, "run": runs[mm]["run"], "clamped_share": float((clamped != raw).mean())}
    pairs = {label: month_pair_mae(roots, runs[label]["config"]) for label, *_ in DELIVERABLES if label in runs}

    unimpeded = None
    if not args.no_table_baselines:
        months = [f"2025-{m:02d}" for m in HOLDOUT_MONTHS]
        hold = oofs["C1"].index if "C1" in oofs else ctx.index[ctx["month"].isin(months).to_numpy()]
        computed, ref = table_baselines(baseline_frame(), hold)
        for label, title, tiers in COMPUTED:
            oofs[label] = computed[label]
            runs[label] = {"run": "computed in this report", "title": title, "tiers": tiers,
                           "protocol": "table from the fit rows", "best_iteration": None}
        unimpeded = unimpeded_table(ref, ctx)

    rows = []
    for label, oof in oofs.items():
        rows += score(label, oof, ctx)
    table = pd.DataFrame(rows)
    order = [lbl for lbl in ("B0", "B1", *(c for c, *_ in COMPUTED), "B2") if lbl in runs] + [
        n for n, *_ in OP_CONFIGS if n in runs] + [lbl for lbl in ("L3-IOBT", "L24-IOBT") if lbl in runs] + [
        x for label, *_ in DELIVERABLES for x in (label, f"{label}-MED", f"{label}-MAE") if x in runs] + [
        lbl for lbl in ("C1", "C2", "C3") if lbl in runs]
    meta = {k: {kk: vv for kk, vv in runs[k].items() if kk != "dir"} for k in order}
    trip = tripwire(table)
    warnings = row_set_warnings(oofs)
    for w in warnings:
        print("WARNING:", w)
    steps, apt_steps = rung_steps(oofs, ctx)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.insert(1, "run_id", table["row"].map({k: v["run"] for k, v in runs.items()}))
    table.to_csv(args.out.with_suffix(".csv"), index=False)
    stem = args.out.with_suffix("")
    pd.DataFrame([*({"apt": "all", **st} for st in steps), *apt_steps]).to_csv(
        stem.with_name(stem.name + "_steps.csv"), index=False)
    if unimpeded is not None:
        unimpeded.to_csv(stem.with_name(stem.name + "_unimpeded.csv"))
    if calib:  # rule C per deliverable and airport, for the figures (notebooks/build/build_07.py)
        pd.DataFrame([{"deliverable": label, "apt": a, **v} for label, c in calib.items()
                      for a, v in [("all", {"coverage": c["coverage_p10_p90"], "mae_mean": c["mean"]["mae"],
                                            "mae_med": c["median"]["mae"]}), *c["airports"].items()]]
                     ).to_csv(stem.with_name(stem.name + "_calibration.csv"), index=False)
    extra = deliverable_lines(seed_frames, ctx, calib, pairs, oofs)
    args.out.with_suffix(".md").write_text(markdown(table, meta, trip, steps, warnings, apt_steps, unimpeded, extra),
                                           encoding="utf-8")
    print("\n".join(trip))
    print(f"wrote {args.out.with_suffix('.md')} and .csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
