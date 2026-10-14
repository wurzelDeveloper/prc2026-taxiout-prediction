# Feature blocks of the submission line (exp051 lean2: 26 blocks, 211 columns)

Scope: the post-operations analysis model (the challenge model in the code), whose file is scored. The operational
models, which read only what exists before the aircraft moves, have their own catalogue:
[`operational_features.md`](operational_features.md) (every data point and input with its availability per decision
moment, tier, rationale and evidence) and the model card [`operational_model.md`](operational_model.md).

| block | columns | what it measures |
|---|---|---|
| `core_keys` | 2 | Reporting airport and takeoff time as epoch seconds. |
| `aircraft_identity` | 12 | Aircraft type / wake category / operator identity. |
| `runway_queue` | 8 | Departure pressure on the same runway around takeoff. |
| `airport_load` | 8 | Airport-wide traffic load and ground occupancy at takeoff time. |
| `arrival_conditions` | 7 | Taxi-in anomaly of the arrivals that landed at the airport in the last 30 min, 60 min and 6 h. |
| `calendar` | 13 | Calendar / time-of-day features in airport local time. |
| `stand_runway_keys` | 11 | Stand/runway identity, composite keys and parsed stand structure. |
| `prev_stand_arrival` | 7 | The arrival that last occupied this stand. |
| `era5_surface` | 9 | ERA5 at the last full hour before takeoff: snow depth, snowfall over 3 h and 24 h, freezing precipitation, ground temperature, overnight minimum and frost hours. |
| `wx_now` | 15 | The METAR in force at takeoff: runway wind, visibility, ceiling, weather. |
| `deicing_context` | 6 | Cold-soaked aircraft (stand idle 6 h+ or overnight) under frost, snow or METAR de-icing conditions, the queue of such departures in the last hour, and heavies under de-icing conditions. |
| `stale_schedule` | 9 | Takeoff minus schedule as a residual base on the untyped, NM-less segment; the unclipped schedule and LOBT deltas for everyone. |
| `delay_stale_stats` | 8 | Smoothed share of block times equal to the schedule (and to NM's last off-block time) among the fold's departures 30-120 min late, by carrier, pier and both, per airport. |
| `dep_live_state` | 18 | The other departures' NM taxi-out anomaly (mean, share above +5 min) in the last 30/60 min and +-15 min, per airport, runway and pier, self excluded. |
| `load_normalised` | 3 | Traffic load relative to what is normal for this airport at this hour. |
| `nm_cross_deltas` | 7 | NM's actual, last-planned, initial and scheduled off-block times against each other, plus data-source fingerprints. |
| `nm_deltas` | 15 | Deltas between takeoff time and the NM/scheduled timestamps. |
| `nm_hour_context` | 5 | The airport-hour's NM picture: mean push-back delay, share of actual off-blocks past the plan, share without an NM match. |
| `operator_presence` | 4 | How established this carrier / aircraft type is at this airport. |
| `pier_load` | 2 | Departures pushing back from the same pier/terminal around this takeoff. |
| `runway_sequence` | 10 | Position in the runway's takeoff sequence, the preceding wake, mixed-mode arrivals, saturation and the runway configuration's age. |
| `stand_occupancy` | 9 | The previous occupant's delay, wake and carrier, the stand's recent traffic, and the next occupant. |
| `stand_runway_frequency` | 4 | How common this stand / runway / pairing is at the airport. |
| `taxi_distance` | 4 | Straight-line distance from the stand to the departure runway threshold. |
| `taxi_window_counts` | 6 | Takeoffs, departures and landings between NM's off-block time and this takeoff; queue overtakes by NM off-block order. |
| `wake_queue` | 9 | ICAO wake minimum to the previous takeoff and its slack, and the continuous runway queue (position, age, followers, length) the takeoff belongs to. |

<!-- availability:start -->
## Operational availability of every input

The challenge gives the actual takeoff time and hides the recorded off-block time: it asks for a retrospective
reconstruction. The real-world use of a taxi-out prediction is the opposite direction: a prediction made **before
the aircraft moves** (departure management), when neither the off-block nor the takeoff has happened. Each input is
therefore classified by whether it would exist at that moment (audit of 1 Oct 2026: five readers over the code, a
critic re-checking every operational claim; the table below was generated from that audit during development).

| tier | meaning |
|---|---|
| **O** operational | everything used is known before this flight's off-block (schedule, flight plan, NM planned times, stand, aircraft, statistics of past data, weather and traffic before the prediction time) |
| **T** takeoff-anchored | uses the flight's actual takeoff time or takeoff order as an anchor; the same idea is operational once re-anchored on a planned or predicted time |
| **A** after off-block | uses information produced at or after an off-block: the flight's own NM actual off-block (AOBT_3) or other flights' actual events after the prediction time |
| **M** own movement | observes the flight's own pushback and taxi (ADS-B): exists only after its off-block; challenge-only |
| **P** published later | not available in real time (EUROCONTROL PRU daily data, ERA5 reanalysis; a weather forecast model would substitute for ERA5) |
| **C** challenge artefact | exploits how the challenge data were built: the NM join window (off-block within LOBT ± 3,606 s), stale or wrong-date recorded off-block times, the NM-join-failure level of categorical columns |
| **O*** operational under an assumption | as O, but resting on an assumption the data cannot check: the planned departure runway taken as the recorded one, the stand plan taken as final hours ahead, other flights' recorded runway or stand standing in for their planned ones, the last filed off-block time known 15 min ahead. Used by the operational models (`docs/operational_features.md`); in this table O* inputs are counted as O |

A block carries every tier that applies, the most restrictive first. "Operational as computed" means the tiers are
O only; "after re-anchoring" means T(+O): computing the same quantity at the planned off-block instead of the actual
takeoff makes it operational. "In base" marks the inputs that reach the submission: the blocks of the base model (the
27 runs of the eleven-run stack, including the older 17-run blend at weight 0.20, and the same configurations
retrained longer or on other month pairs), and the ADS-B stage's inputs as the submitted stage runs use them (`--v5
--drop-prefix tarr_,it_`: the third and fourth feature sets without the inbound and arrival-map families; the whole
base matrix in the direct formulation).

**Inputs that reach the submission** (base blocks and the ADS-B correction stage): 44 not before movement, 7 after re-anchoring, 7 operational as computed.

| input | tiers | in base | operational status | note |
|---|---|---|---|---|
| `ADS-B takeoff detection and departure link (takeoffs, link; link columns adsb_linked, adsb_takeoff_dt, adsb_link_by_time)` | M+T | yes | not before movement | Challenge-only plumbing. |
| `adsb_apply coverage gate and training rows (adsb_linked & adsb_first_seen; --fit-fix)` | M | yes | not before movement | A departure is corrected only if ADS-B saw its own ground track before takeoff: 237,291 ranking rows in v35 and 241,587 in v36/v37c, about 69-70 %. |
| `adsb_apply derived ADS-B timings (*_m_base, *_m_proxy, *_obs_m_base)` | M+C+A+P+T | yes | not before movement | These turn the flight's own observed taxi into gaps to the base prediction and to NM's actual off-block proxy. |
| `adsb_apply stale gating and taxi-regime correction (p_stale input, MAX_P_STALE gate, --regime with y_reg)` | M+C+A+P+T | yes | not before movement | Rows whose recorded block time may be a schedule (p_stale > 0.05) are not corrected directly. |
| `first-reception state (fr_*: speed, direction, route covered at the first ground point)` | M | yes | not before movement | Challenge-only: the flight's own first ADS-B ground sighting after its off-block; the per-stand reception boundary pools all 124 days of ADS-B (no target). |
| `own_movement (ADS-B family B, first set) + adsb_first_dist_stand` | M+T | yes | not before movement | Tier M, challenge-only. |
| `pushback detector (pb_*: reverse-motion push start/end, engine-start pause, stand cluster)` | M | yes | not before movement | Challenge-only: the flight's own pushback seen in ADS-B (motion opposite to the reported heading); in operations it would detect the actual off-block in real time, not predict it. |
| `stand-type archive (ar_*, sa_best_est, sa_8kt_est, sa_best_src)` | M | yes | not before movement | Challenge-only as used: per-stand ADS-B statistics (target-free, all 124 days) applied to the flight's own observed movement; the archive alone would be an operational O input. |
| `stand_movement (ADS-B family B, second set; FAM_B2 incl. adsb_takeoff_offset)` | M+T | yes | not before movement | Tier M, challenge-only: the flight's own departure from its stand and its stops after leaving, observed after its off-block. |
| `stop classes (ll_*, deice_*, dz_s, rwy_queue_est, park2_est)` | M | yes | not before movement | Challenge-only: classifies the flight's own stops after leaving the stand by location (runway area, remote/de-icing, hold); location statistics pool all 124 days of ADS-B stops (no target). |
| `time-to-location map (tmap_*)` | M | yes | not before movement | Challenge-only: a target-free map of typical elapsed times per surface cell applied to the flight's own cell exits after its off-block. |
| `track quality (adsbQ_*: reception density, gaps, stand jitter, jumps, inbound landing, relocations)` | M | yes | not before movement | Challenge-only: describes the reliability of the flight's own ground track up to its takeoff. |
| `NM join-window clip (prc/postprocess.py)` | C+T | yes | not before movement | Challenge artefact, worth about 40 s on the board (v29 303.02 to v31 263.17). |
| `RegimeMixtureModel regressor_anchor (aobt3_proxy; lobt_delta_full)` | C+A+T | yes | not before movement | The regressor learns the target minus an anchor. |
| `RegimeMixtureModel window_column (NM join window inside the mixture)` | C+T | yes | not before movement | The join window inside the model. |
| `RegimeMixtureModel: stale/schedule base, universe, regime labels, classifier, regressor fit rows` | C+A+P+T | yes | not before movement | Challenge artefact: it predicts which recorded block times are schedules (stale records) and hedges between takeoff - schedule and a taxi time. |
| `adsb_apply --rich (main model's lean2 matrix as x_* columns)` | C+A+P+T+O | yes | not before movement | Variant that gives the correction the whole main-model matrix. |
| `adsb_apply base (base run prediction, window-clipped)` | C+A+P+T+O | yes | not before movement | The correction is a second stage on the residual of the full submission model, so it inherits all of that model's tiers: the stale mixture, the join-window clip and the linked exclusion (C); |
| `adsb_apply window inputs and output clip (win_lo, win_hi)` | C+T | yes | not before movement | The join-window bounds are given to the correction as inputs, and the corrected prediction is clipped to them, the same challenge artefact as the post-processing clip. |
| `aircraft_identity` | C+O | yes | not before movement | The content is operational (flight-plan and schedule identity). |
| `delay_stale_stats` | C+O | yes | not before movement | In the submission line and its strongest classifier input (block permutation +75 s, exp043 notes). |
| `final_stack in-window estimate (inwindow_offsets, inwindow)` | C+A+T | yes | not before movement | Challenge artefact: a prediction below the window's lower edge would be clipped onto the edge. |
| `final_stack.exclusion (linked-untyped regime exclusion)` | C+A+T | yes | not before movement | Challenge artefact: it rules out the regime whose value would put the block inside the window that the departure's failed NM join excludes, and moves the stale weight to the fit-month rate (0.02 when stale is impossible, 0.85 whe… |
| `key_target_stats` | C+T+O | yes | not before movement | These are fold-safe medians, long-taxi rates and p90s by carrier, type, local hour and stand-runway pair, computable from an operational archive. |
| `linked_untyped.linked_frame (destination-arrival LOBT link for untyped, NM-unmatched departures)` | C+A+T | yes | not before movement | Recovers NM's LOBT for an untyped, NM-unmatched departure from the same flight's arrival record at another of the ten airports. |
| `nm_cross_deltas` | C+A+O | yes | not before movement | The main signal (actual off-block against the last plan) is after-off-block information (A). |
| `nm_deltas` | C+A+T+O | yes | not before movement | The headline column aobt3_proxy (takeoff minus NM's actual off-block) is after-off-block information (A) with no pre-movement equivalent. |
| `nm_hour_context` | C+A+T | yes | not before movement | The hour's push-back delay picture uses other flights' actual off-blocks in a clock-hour bucket that runs up to 59 min past this takeoff, including this flight's own off-block (A). |
| `operator_presence` | C+O | yes | not before movement | Operational: it measures how established the carrier and aircraft type are at the airport and how concentrated the carrier's piers are. |
| `stale_schedule` | C+T+O | yes | not before movement | A challenge-artefact block: it models records whose block time is the schedule. |
| `stale_target_stats` | C+O | yes | not before movement | A fold-safe rate of stale block records per carrier and destination: a challenge-artefact statistic (C), although its inputs are available before off-block. |
| `adsb_apply NM list (aobt3_proxy, lobt_delta, sched_delta, eobt1_delta)` | A+T | yes | not before movement | NM clocks given to the correction model. |
| `airport_load` | A+T | yes | not before movement | Airport-wide load: the backward counts and arrivals-still-taxiing become operational once anchored at the prediction time. |
| `arrival_conditions` | A+T | yes | not before movement | Live ground conditions from the taxi-in anomaly of recently landed arrivals. |
| `deicing_context` | A+P+T | yes | not before movement | The idea is operational: an operator knows how long the aircraft has been parked and what the de-icing weather is. |
| `dep_live_state` | A+T | yes | not before movement | How long other departures are currently taking to taxi, by airport, runway and pier. |
| `nm_proxy_context` | A+T | yes | not before movement | It inherits the after-off-block proxy (A) and centres it on a clock-hour bucket that includes other flights' later takeoffs and actual off-blocks, so it is not operational. |
| `prev_stand_arrival` | A+T | yes | not before movement | The idea is operational (turnaround so far, the inbound's taxi-in), but anchored at the actual takeoff the stand join also picks up aircraft that blocked in at the same stand during this flight's taxi-out. |
| `runway_queue` | A+T | yes | not before movement | These columns count runway departure pressure from other flights' actual takeoffs around this flight's actual takeoff. |
| `runway_sequence` | A+T | yes | not before movement | These columns read this takeoff's place in the realised runway sequence: the preceding wake, mixed-mode arrivals, saturation and configuration age. |
| `stand_occupancy` | A+T | yes | not before movement | The ideas are mostly operational (previous occupant's delay, wake and carrier, the stand's recent traffic, planned ground time), but the block is built on the actual takeoff. |
| `taxi_window_counts` | A+T | yes | not before movement | These columns count traffic during the flight's own taxi, bounded by NM's actual off-block and the actual takeoff, and queue overtakes by NM actual off-block order. |
| `wake_queue` | A+T | yes | not before movement | This block reconstructs the holding-point queue the flight actually sat in from the realised takeoff times. |
| `era5_surface` | P+T | yes | not before movement | Not available in real time (ERA5T about 5 days late, final ERA5 about 2-3 months late). |
| `calendar` | T+O | yes | after re-anchoring | Re-anchoring is trivial, since time of day and calendar at the actual takeoff can be computed at the planned off-block instead; |
| `core_keys` | T+O | yes | after re-anchoring | The airport key is operational, but takeoff_epoch is the actual takeoff time to the second, which a predictor working before off-block does not have; |
| `load_normalised` | T | yes | after re-anchoring | The last hour's departure count relative to its usual weekday-hour level. |
| `pier_load` | T | yes | after re-anchoring | Local apron load from same-pier departures in the hour before takeoff. |
| `wx_deice_load` | T | yes | after re-anchoring | Operational after re-anchoring: it counts other flights' takeoffs and landings in look-back windows ending at this flight's takeoff, whose last minutes fall after its off-block, and ending them at the prediction time removes that… |
| `wx_now` | T | yes | after re-anchoring | Operational once re-anchored: at off-block, the METAR in force at takeoff is either the same report or the next half-hourly one, so the operational version reads the latest METAR before the prediction, or TREND/TAF for the planne… |
| `wx_past` | T | yes | after re-anchoring | Operational after re-anchoring: every window looks back from takeoff, so ending them at off-block loses only the minutes of the taxi itself. |
| `adsb_apply keys (apt, stand, runway)` | O | yes | operational as computed | Airport, stand and departure runway as categories. |
| `stand_geometry` | O | yes | operational as computed | Operational: static stand position, distance to the airport reference point, k-means apron zone and the quality of the OSM match, all of which an airport knows from its own layout data. |
| `stand_runway_frequency` | O | yes | operational as computed | Operational: how common the stand, the runway and the pairing are at the airport. |
| `stand_runway_keys` | O | yes | operational as computed | Operational: stand, departure runway, destination and their combinations. |
| `stand_runway_target_stats` | O | yes | operational as computed | Operational: a fold-safe historical median and IQR of taxi-out per (stand, runway) pairing, with fallback to stand, runway and airport. |
| `taxi_distance` | O | yes | operational as computed | Operational: the straight-line distance from the stand to the departure threshold, plus runway length and relative bearing, from static layout data. |
| `taxi_route` | O | yes | operational as computed | Operational: the shortest taxiway route length, its ratio to the straight-line distance, and the number of runway crossings from stand to threshold, all from static network data. |
| `adsb_apply --mixweight (row weights from 2026 coverage counts)` | M+C |  | not before movement | Training rows are weighted per airport by the 2026 ranking set's ADS-B coverage count over 2025's. |
| `adsb_apply --v3 route features (geo_route_dist_m, geo_route_rwy_crossings, route_unseen_m)` | M+O |  | not before movement | Geometry is O. |
| `inbound and turnaround (it_*: the aircraft's inbound, arrival-record link, parks, tows over 24 h)` | M |  | not before movement | Challenge-only, and left out of the submitted stage (--drop-prefix tarr_,it_): the departing aircraft's own track up to 24 h before its takeoff, including its movements after the off-block; the arrival record's in-block time is an allowed movement column. |
| `time-to-location map, arrival half (tarr_*)` | M |  | not before movement | Challenge-only; left out of the submitted stage (--drop-prefix tarr_,it_) after it did not transfer to the board (v40). |
| `RegimeMixtureModel day-offset regime` | C |  | not before movement | Challenge artefact: a block time carrying the previous day's date, so the target is 86,400 s plus a taxi. |
| `flight_number_history` | C+O |  | not before movement | The fold-safe history of the flight number (long-taxi rate, median and p90 taxi-out) is operational from an archive. |
| `stale_day_context` | C+A+T |  | not before movement | Excluded from the submission line (screened out in exp043). |
| `deice_location` | A+T |  | not before movement | The target-based uplift is an archive statistic and so operational. |
| `dep_following` | A+T |  | not before movement | These columns read the NM taxi-out proxies of the departures that take off in the next 15/30 min after this one. |
| `family_c (others' ADS-B taxi readings: c_mean30, c_n30, c_mean30_m_proxy)` | A+T |  | not before movement | Live surveillance of the other flights' completed taxis. |
| `pier_pushback` | A+T |  | not before movement | Challenge-only as implemented: it counts same-pier departures taking off up to 15 min after this one, arrivals blocking in up to 15 min after takeoff, and push-back conflicts read from this flight's own NM actual off-block. |
| `pru_daily` | A+P+T |  | not before movement | Not in the submission line (default out). |
| `traffic_deviation` | A+T |  | not before movement | The whole day's (and clock hour's) realised movement count against its weekday median. |
| `wx_future` | A+T |  | not before movement | Hindsight: weather observed in the hour after the actual takeoff does not exist before off-block, so this block serves the challenge only unless forecasts replace it. |
| `zone_load` | A+T |  | not before movement | Challenge-only as implemented for the same reasons as pier_pushback (forward 15-minute windows on other flights' actual takeoffs and in-blocks, and this flight's NM actual off-block). |
| `flight_number_context` | T+O |  | after re-anchoring | The idea, how regular the flight number is, can be read from the season schedule, but the block works on actual takeoff dates and times: fn_hours_since_prev_dep is this takeoff minus the previous same-number takeoff, so it contai… |
| `surface_context (ADS-B family A: adsbA_ground_ac, adsbA_taxiing_ac, adsbA_since_inbound)` | T+O |  | after re-anchoring | The operational ADS-B family once anchored on the planned off-block: other aircraft on the ground and taxiing in the 5 min before LOBT, and the time since this aircraft's inbound landed. |
| `taf_now` | T |  | after re-anchoring | Operational once re-anchored: a TAF is a forecast issued before it is used, so only the issue cutoff (now the actual takeoff) and the valid time need to move, to the prediction time and the planned takeoff. |
| `taf_planning` | T |  | after re-anchoring | Mostly operational: the planning TAF predates off-block by hours, and only the 'unforecast weather' columns, which compare it with METARs up to the actual takeoff, need re-anchoring to the prediction time. |
| `wx_trend` | T |  | after re-anchoring | Operational once re-anchored: TREND is a nowcast attached to a METAR already issued, so the report in force at off-block replaces the one in force at takeoff. |
<!-- availability:end -->
