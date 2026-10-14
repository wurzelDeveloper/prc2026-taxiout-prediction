"""Operational tiers of the inputs of the operational models: the long-horizon rungs (configs/op_hl*.yaml), the ADS-B
archive and, since findings 10.35, the H1 and H0 deliverables (`tier_at`, read by the feature catalogue).

A declaration as code, so that the configs and the tests can check it (tests/test_operational_long.py). No feature
module imports it, so editing it changes no cache key. A column carries every tag of every pattern it matches; a
column without a tag is tier O (plan data, the airport's own static facts, weather and other flights' records known
at the anchor).

The clock of these rungs is the published schedule (``P`` = ``SCHED_TIME_UTC_mvt``, `_horizon.planned_offblock_long`;
findings 10.35): plan data every departure carries, known months ahead, the same on NM-joined and unjoined rows. The
IOBT is not read, of the flight or of any other flight, so no column rests on a filed-plan assumption (until 11 Oct
the IOBT was the clock, its minutes against the schedule an output and the other flights' IOBTs a demand count).

* ``own_runway`` -- reads the flight's own recorded departure runway (``RUNWAY_mvt``). At H1 the configuration in use
  is current and the runway was accepted as O* (findings 10.27). Hours ahead the recorded runway is the realised
  outcome of a future runway configuration: on the 2025 departures it matches the airport's dominant departure runway
  of the hour before ``a = SOBT - 24 h`` on 53.4 % and before ``a = SOBT - 3 h`` on 53.3 %, against 66.3 % before H1
  and 70.6 % before the own takeoff (`lab/op_clock_checks.py` section 7, docs/op_runway_knowability.txt; it is among
  the runways in use on 75.1 %, 75.8 %, 93.1 % and 97.9 %). The long deliverables (the refit rungs)
  are therefore runway-blind (``*_rwyblind``: every ``own_runway`` column dropped); the runway-aware long rungs are an
  O* upper bound.
* ``own_stand`` -- reads the flight's own recorded stand beyond its pier: O* at both long leads (a departure's stand
  depends on where its inbound parks, decided on the day). ``op_hl24_r4_arch_rwyblind_pier`` drops them.
* ``own_pier`` -- reads the pier (the stand prefix, the OSM apron zone): O*, but a weak assumption: the airline's
  terminal and pier allocation is stable and taken as known a day ahead. The pier-only rung keeps these.
* ``others_future`` -- groups OTHER flights that move after the anchor by their recorded runway or stand (their
  planned allocation assumed known; `_horizon.O_STAR_FUTURE_ATTRIBUTES_LONG`, test T2b).
* ``join_c`` -- tier C, the challenge's NM join: the join needs ``|LOBT - block| <= 3,606 s``, i.e. the flight's own
  actual off-block, so whether a record joined is not knowable before the movement. Its missing level marks an
  unjoined flight; the carriers are only columns of `aircraft_identity`: every ``_flt`` column (and the operator code
  read from the NM callsign) and the aircraft type of the record (every untyped departure is unjoined, and the
  untyped segment holds the stale block times). The unjoined 1.1 % of 2025 departures hold 42 % of the target's
  squared deviation. The flags themselves (``has_iobt``, ``has_actype``) are not outputs, and no output of the long
  blocks carries the join; the carriers are constant on the OP population (NM-joined), so the long ladder is read
  there.

Two tags serve the H1 and H0 blocks only (findings 10.35):

* ``own_plan`` -- the flight's own last filed off-block time against the schedule (``op_h1_plan_minus_sched_min``).
  The H1 anchor sits 15 min before that time, so the column assumes the last filed plan is known 15 min ahead, as an
  A-CDM TOBT is; the data hold its last value without the time of its last update (O* at H1; at H0, the push, the
  plan is final and the column is O).
* ``others_plan`` -- counts OTHER flights by their last filed off-block (the plan backlog), assuming their plans final
  at the anchor (O* at H1 and H0).

One tag serves the traffic after the H1 anchor (`prc/features/operational_future.py`; findings 10.36):

* ``others_realised`` -- tier **F**: the OTHER flights' realised movements after the anchor (records, NM readings,
  public ADS-B read at ``a + 15/30/45`` min). In operations these values come from the airport's planning systems
  (AMAN landing sequence, DMAN start-up and takeoff sequence, A-SMGCS surface plan); the challenge data hold only what
  happened, so the realised movements stand in. An upper bound for perfect plans, not available as computed. The plan
  counts of ``op_h1_fut_plan`` are not F: they read the other flights' filed plans and the schedule (``others_plan``,
  O*).

The moment matters for the stand: at H1 and H0 the aircraft is on its stand, so ``own_stand`` and ``own_pier`` are
facts there (`tier_at` drops them); hours ahead they are the stand plan (O*).

Declared: the outputs of the blocks the long, archive and H1/H0 deliverable rungs resolve (core_keys,
stand_runway_keys, aircraft_identity, taxi_distance, taxi_route, stand_geometry, op_hl*, op_adsb_archive, op_h1_*,
op_h0_*).
"""

from __future__ import annotations

import re

TAGS = ("own_runway", "own_stand", "own_pier", "others_future", "join_c", "own_plan", "others_plan", "others_realised")

PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # stand_runway_keys
    (r"^runway$", ("own_runway",)),
    (r"^apt_runway$", ("own_runway",)),
    (r"^(stand|apt_stand|stand_number|stand_suffix)$", ("own_stand",)),
    (r"^apt_stand_runway$", ("own_stand", "own_runway")),
    (r"^stand_prefix$", ("own_pier",)),
    # aircraft_identity: the NM-join level (carrier_flightno / carrier_missing come from the movement record)
    (r"^(aircraft_type|aircraft_type_flt|aircraft_type_mismatch|wake_category|market_segment|flight_type"
     r"|flight_rule_flt|operator_hash|carrier_icao)$", ("join_c",)),
    # geometry: the route and distances to the recorded runway from the recorded stand
    (r"^geo_gc_dist_m$", ("own_stand", "own_runway")),
    (r"^geo_rwy_", ("own_runway",)),
    (r"^geo_route_", ("own_stand", "own_runway")),
    (r"^geo_stand_(east_m|north_m|arp_dist_m|matched|match_source)$", ("own_stand",)),
    (r"^geo_stand_zone$", ("own_pier",)),
    # the long-horizon blocks
    (r"^op_hl_has_stand$", ("own_stand",)),
    (r"^op_hl_dem_\w+_rwy_w\d+$", ("own_runway", "others_future")),
    (r"^op_hl_dem_\w+_pier_w\d+$", ("own_pier", "others_future")),
    (r"^op_hl(3|24)_wx_(headwind|crosswind)_kt$", ("own_runway",)),
    (r"^op_hl(3|24)_hist_rwy_", ("own_runway",)),
    # the ADS-B archive
    (r"^op_arch_stand_", ("own_stand",)),
    (r"^op_arch_pier_", ("own_pier",)),
    (r"^op_arch_sr_", ("own_stand", "own_runway")),
    (r"^op_arch_rwy_", ("own_runway",)),
    # H1 and H0 (prc/features/operational.py; findings 10.35)
    (r"^op_h[01]_wx_(headwind|crosswind)_kt$", ("own_runway",)),
    (r"^op_h[01]_(dep|arr)_rwy_b\d+$", ("own_runway",)),
    (r"^op_h[01]_gap_last_(dep|arr)_rwy_s$", ("own_runway",)),
    (r"^op_h[01]_rwy_own_", ("own_runway",)),
    (r"^op_h[01]_nm_rwy_", ("own_runway",)),
    (r"^op_h[01]_backlog_apt$", ("others_plan",)),
    (r"^op_h[01]_backlog_rwy$", ("own_runway", "others_future", "others_plan")),
    (r"^op_h[01]_backlog_pier$", ("own_pier", "others_future", "others_plan")),
    (r"^op_h[01]_stand_min_since_inblock$", ("own_stand",)),
    (r"^op_h1_plan_minus_sched_min$", ("own_plan",)),
    (r"^op_h[01]_adsb_rwy_own_hold_now$", ("own_runway",)),
    # traffic after the H1 anchor (prc/features/operational_future.py; findings 10.36)
    (r"^op_h1_fplan_dep_plan_apt_w\d+$", ("others_plan",)),
    (r"^op_h1_fplan_dep_plan_rwy_w\d+$", ("own_runway", "others_future", "others_plan")),
    (r"^op_h1_fplan_dep_plan_pier_w\d+$", ("own_pier", "others_future", "others_plan")),
    (r"^op_h1_fplan_arr_sched_pier_w\d+$", ("own_pier", "others_future")),
    (r"^op_h1_f(rec|nm|adsb)_f\d+_", ("others_realised",)),
    (r"^op_h1_frec_f\d+_((dep|arr)_rwy_b\d+|rwy_own_)", ("own_runway",)),
    (r"^op_h1_fnm_f\d+_(nm_rwy_|backlog_rwy$)", ("own_runway", "others_future")),
    (r"^op_h1_fnm_f\d+_backlog_pier$", ("own_pier", "others_future")),
    (r"^op_h1_fnm_f\d+_backlog_apt$", ("others_plan",)),
    (r"^op_h1_fadsb_f\d+_rwy_own_hold_now$", ("own_runway",)),
)
"""(regular expression, tags) pairs; a column takes the union over every pattern it matches."""

_COMPILED = tuple((re.compile(p), tags) for p, tags in PATTERNS)


def tags_of(column: str) -> frozenset[str]:
    """Every tag of ``column`` (empty: tier O)."""
    return frozenset(t for rx, tags in _COMPILED if rx.search(column) for t in tags)


def with_tag(columns, *tags: str) -> list[str]:
    """The columns (in order) that carry any of ``tags``."""
    unknown = set(tags) - set(TAGS)
    if unknown:
        raise ValueError(f"unknown tags {sorted(unknown)}")
    return [c for c in columns if tags_of(c) & set(tags)]


FACTS_AT = {"h1": frozenset({"own_stand", "own_pier"}), "h0": frozenset({"own_stand", "own_pier", "own_plan"})}
"""Tags that are facts at a moment: at H1 and H0 the aircraft is on its stand; at H0 (the push) its plan is final."""


def tier_at(column: str, horizon: str) -> str:
    """The tier label of ``column`` at a decision moment (``hl24``, ``hl3``, ``h1``, ``h0``)."""
    if horizon not in ("hl24", "hl3", "h1", "h0"):
        raise ValueError(f"unknown moment {horizon!r}")
    return _label(tags_of(column) - FACTS_AT.get(horizon, frozenset()))


def tier_of(column: str) -> str:
    """A short tier label for reports: ``O``, or ``O*`` / ``C`` with the reasons (as read hours ahead)."""
    return _label(tags_of(column))


def _label(tags) -> str:
    if not tags:
        return "O"
    notes = []
    if "own_runway" in tags:
        notes.append("own runway")
    if "own_stand" in tags:
        notes.append("own stand")
    if "own_pier" in tags:
        notes.append("own pier")
    if "others_future" in tags:
        notes.append("others' allocation")
    if "own_plan" in tags:
        notes.append("own plan final 15 min ahead")
    if "others_plan" in tags:
        notes.append("others' plans final")
    star = f"O* ({', '.join(notes)})" if notes else ""
    if "others_realised" in tags:
        return "F (others' realised movements standing in for AMAN/DMAN/A-SMGCS plans)" + (f" + {star}" if star else "")
    if "join_c" in tags:
        return "C (NM join level)" + (f" + {star}" if star else "")
    return star
