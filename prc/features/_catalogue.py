"""The operational feature catalogue: every input of the operational models, where it comes from, when it is known and
why it should move the taxi-out time (findings 10.35; the owner's goal of 11 Oct 2026).

`lab/feature_catalogue.py` turns these hand-written facts into docs/operational_features.md and .csv. It adds three
things this module does not hold:
- the column lists of the deliverable configurations;
- the tier of each column at each horizon (`prc.features._long_tiers.tier_at`);
- the measured evidence: gain shares of the refitted runs and the block permutation.

`tests/test_feature_catalogue.py` checks the catalogue:
- every column a deliverable model reads has exactly one entry;
- every entry is complete;
- every data point an entry names exists.

No feature block imports this module (underscore: not auto-imported either), so editing it moves no cache key.

**The decision moments.** One model per moment, each a global CatBoost over the ten airports:
- ``hl24``, 24 h ahead: the published schedule minus 24 h.
- ``hl3``, 3 h ahead: the published schedule minus 3 h.
- ``h1``, the headline: 15 min before the last filed off-block time (IOBT, else LOBT, else the schedule). Both the A-CDM
  pre-departure sequence and the target start-up approval time are computed around this moment. Its deliverable since
  13 Oct is F1 (`op_h1_f1_plan`): the H1 model of findings 10.35 plus the other flights' plans after the anchor
  (findings 10.37).
- ``h0``: at the recorded off-block, the moment of the A-CDM actual off-block time. 2025 only, because 2026 blanks the
  off-blocks.

A column's name carries its moment as a prefix (``op_h1_``, ``op_h0_``, ``op_hl3_``, ``op_hl24_``; ``op_hl_`` for the
inputs both long moments share) or no prefix (static inputs every model reads). The catalogue keys the families on
the name without that prefix, so one entry serves the same quantity at every moment.
"""
from __future__ import annotations

from dataclasses import dataclass, field

HORIZONS = ("hl24", "hl3", "h1", "h0")
"""The decision moments, earliest first."""

HORIZON_NAMES = {
    "hl24": "24 h ahead (schedule - 24 h)",
    "hl3": "3 h ahead (schedule - 3 h)",
    "h1": "H1, 15 min before the last filed off-block",
    "h0": "H0, at the off-block (2025 only)",
}

PREFIXES = ("op_hl24_", "op_hl3_", "op_hl_", "op_h1_", "op_h0_")
"""Column-name prefixes of the moment-specific blocks; a column without one is a static input."""


def strip_prefix(column: str) -> str:
    """The column's name without its moment prefix (the catalogue's key)."""
    for p in PREFIXES:
        if column.startswith(p):
            return column[len(p):]
    return column


@dataclass(frozen=True)
class DataPoint:
    """One raw field or external source and when an operator holds it."""

    id: str
    field: str
    what: str
    system: str
    known: dict[str, str]
    """Availability per moment (`HORIZONS`): ``yes``, ``O*: <the assumption>``, ``no`` or ``not used``."""
    use: str
    """How the models use it: for the flight itself, for other flights, or both."""


@dataclass(frozen=True)
class Family:
    """A family of model inputs that share one source, one definition pattern and one rationale."""

    id: str
    title: str
    columns: dict[str, str]
    """Column name without its moment prefix -> its definition."""
    sources: tuple[str, ...]
    """`DataPoint` ids."""
    rationale: str
    literature: str
    caveats: str = ""
    anchor_note: dict[str, str] = field(default_factory=dict)
    """Per moment, what "the anchor" means for this family where it differs from the general rule."""


# --- the data points ------------------------------------------------------------------------------

ALL = {"hl24": "yes", "hl3": "yes", "h1": "yes", "h0": "yes"}

DATA_POINTS: tuple[DataPoint, ...] = (
    DataPoint(
        "airport", "apt (the reporting airport)", "The departure airport (ICAO code).", "the movement record",
        ALL, "the flight itself",
    ),
    DataPoint(
        "sobt", "SCHED_TIME_UTC_mvt", "Scheduled off-block time (SOBT); on an arrival's record, the scheduled in-block "
        "time.",
        "airline schedule and slot coordination; published months ahead",
        ALL,
        "the flight itself: the clock of the 3 h and 24 h models, the scheduled local hour everywhere, the planned delay "
        "(H1, H0) and, at H0, the off-block's delay against the schedule; other flights: the scheduled demand around "
        "the planned off-block (3 h, 24 h), the push delay of departed flights (3 h) and the arrivals scheduled in the "
        "hour after the H1 anchor; at H1 and H0 the schedule stands in for the plan of a departure without a filed one",
    ),
    DataPoint(
        "iobt", "IOBT_flt (LOBT_flt where it is missing)",
        "NM's off-block time of the filed flight plan. On the NM-matched 2025 departures it equals NM's last known "
        "off-block time on 99.45 % and carries most of a late flight's delay. The data hold its last value only, "
        "without the time of its last update.",
        "NM flight plan (IFPS); A-CDM TOBT/EOBT updates",
        {"hl24": "not used", "hl3": "not used",
         "h1": "O*: the last filed values assumed known at the anchor (the flight's own 15 min ahead, the other "
               "flights' up to an hour ahead)",
         "h0": "yes"},
        "the flight itself: the H1 anchor, the planned delay (H1, H0) and, at H0, the off-block's delay against the "
        "plan; other flights: the plan backlog (H1, H0), the departures planned in the hour after the H1 anchor and "
        "the push delay of departed flights (H1, H0)",
    ),
    DataPoint(
        "actype", "AIRCRAFT_TYPE_mvt", "ICAO aircraft type of the movement record.",
        "airline schedule (type assigned months ahead, swaps on the day)", ALL, "the flight itself",
    ),
    DataPoint(
        "fpl_type", "AIRCRAFT_TYPE_flt, WK_TBL_CAT_flt", "Aircraft type and ICAO wake category of the flight plan.",
        "NM flight plan (repetitive plans for scheduled flights, individual plans up to 120 h ahead)",
        {"hl24": "yes (repetitive plan, else the schedule's type)", "hl3": "yes", "h1": "yes", "h0": "yes"},
        "the flight itself",
    ),
    DataPoint(
        "fpl_class", "MARKET_SEGMENT_flt, FLIGHT_TYPE_flt, FLIGHT_RULE_flt, FLIGHT_RULE_mvt",
        "NM market segment, flight type and flight rules (and the record's flight rules).",
        "NM flight plan and its classification", ALL, "the flight itself",
    ),
    DataPoint(
        "operator", "AIRCRAFT_OPERATOR_flt, CALLSIGN_flt, FLIGHT_mvt", "Aircraft operator, callsign and flight number.",
        "airline schedule and flight plan", ALL, "the flight itself",
    ),
    DataPoint(
        "stand", "STAND_mvt", "The departure stand of the record; on an arrival's record, the arrival stand.",
        "airport stand allocation (planned the day before, revised on the day)",
        {"hl24": "O*: stand plan assumed final", "hl3": "O*: stand plan assumed final",
         "h1": "yes (the aircraft is on its stand)", "h0": "yes"},
        "the flight itself; other flights' stands group per pier the scheduled demand (3 h, 24 h), the plan backlog "
        "(H1, H0) and the plans after the anchor (H1)",
    ),
    DataPoint(
        "runway", "RUNWAY_mvt", "The departure runway of the record.",
        "ATC runway assignment within the configuration in use",
        {"hl24": "no (the long deliverables are runway-blind)", "hl3": "no (runway-blind)",
         "h1": "O*: the planned runway assumed known (66.3 % equal to the hour's dominant departure runway)",
         "h0": "O*: as at H1"},
        "the flight itself (H1, H0); other flights' recorded runways: per runway, the counts of those that have taken "
        "off and, standing in for their planned runway, the plan backlog (H1, H0) and the departures planned after the "
        "anchor (H1)",
    ),
    DataPoint(
        "ades", "ADES_mvt", "The destination airport.", "airline schedule and flight plan", ALL, "the flight itself",
    ),
    DataPoint(
        "dep_events", "MVT_TIME_UTC_mvt of OTHER departures", "Takeoff times of other departures.",
        "the airport's movement records (A-CDM ATOT), in real time",
        {"hl24": "yes, up to the anchor", "hl3": "yes, up to the anchor", "h1": "yes, up to the anchor",
         "h0": "yes, up to the anchor"},
        "other flights only, and only events at or before the anchor; the flight's own takeoff is never read",
    ),
    DataPoint(
        "arr_events", "MVT_TIME_UTC_mvt and BLOCK_TIME_UTC_mvt of arrivals", "Landing and in-block times of arrivals.",
        "the airport's movement records (A-CDM ALDT, AIBT), in real time",
        {"hl24": "not used", "hl3": "not used", "h1": "yes, up to the anchor", "h0": "yes, up to the anchor"},
        "other flights: landings, arrivals still taxiing, the taxi-in anomaly; the flight's own inbound: the stand's "
        "last in-block",
    ),
    DataPoint(
        "nm_actuals", "AOBT_3_flt of OTHER departed flights", "NM's actual off-block time of flights already airborne.",
        "NM (from the airport's departure messages); the airport holds its own off-block records",
        {"hl24": "yes, for flights airborne by the anchor", "hl3": "yes, for flights airborne by the anchor",
         "h1": "yes, for flights airborne by the anchor", "h0": "yes, for flights airborne by the anchor"},
        "other flights only: their taxi-out reading (takeoff - AOBT_3) and their push delay; never the flight's own",
    ),
    DataPoint(
        "metar", "METAR (IEM archive)", "Routine and special weather observations at the airport.",
        "met service, public in real time; counted from 5 min after observation",
        {"hl24": "yes (persistence at the anchor)", "hl3": "yes (persistence at the anchor)",
         "h1": "yes", "h0": "yes"},
        "the airport's weather",
    ),
    DataPoint(
        "taf", "TAF (Ogimet archive)", "Terminal aerodrome forecasts with their change groups.",
        "met service, public at issue (the archive used here is Ogimet's, which has no open licence)",
        {"hl24": "yes (issued by the anchor, valid at the planned off-block where it reaches it)",
         "hl3": "yes (issued by the anchor, valid at the planned off-block)", "h1": "yes (issued by the anchor)",
         "h0": "yes (issued by the off-block)"},
        "the airport's forecast weather",
    ),
    DataPoint(
        "osm", "OpenStreetMap aerodrome objects", "Stand positions, taxiways, runways and holding points.",
        "OSM (ODbL); in operations the AIP aerodrome chart and the airport's stand database", ALL,
        "the airport's layout: stand position, route, distance",
    ),
    DataPoint(
        "ourairports", "OurAirports airports and runways", "Airport reference points, runway ends and headings.",
        "OurAirports (public domain); in operations the AIP", ALL, "the airport's layout",
    ),
    DataPoint(
        "offblock_feed", "BLOCK_TIME_UTC_mvt of departures (the airport's off-block records)",
        "The recorded off-block of every 2025 departure.",
        "the airport's A-CDM actual off-block times (AOBT)",
        {"hl24": "not used", "hl3": "not used", "h1": "not used (lab rung R5 only)",
         "h0": "yes: the flight's own off-block is the H0 anchor (2025 only)"},
        "the H0 anchor; the lab feed of other flights' off-blocks (R5) is not part of a deliverable",
    ),
)

NEVER_USED: tuple[tuple[str, str], ...] = (
    ("the flight's own takeoff time (MVT_TIME_UTC_mvt)",
     "it is the outcome the taxi-out leads to; the challenge gives it, an operator does not have it before the movement"),
    ("the flight's own recorded off-block (BLOCK_TIME_UTC_mvt)",
     "it is the target's other end; it is read only as the H0 anchor in the 2025 lab member, never as a value"),
    ("the flight's own AOBT_3_flt and ARVT_3_flt",
     "NM actuals of the flight itself; AOBT_3 is back-computed from the takeoff at LTFM and partly at EDDM, LEBL, "
     "LSZH and LEMD (findings 10.25)"),
    ("EOBT_1_flt", "it behaves like a late-updated estimate (closer to the recorded off-block than the plan), and its "
     "update time is not in the data"),
    ("LOBT_flt as an input of its own", "it stands in for the IOBT where that is missing (the iobt data point: the H1 "
     "anchor, the planned delay, the other flights' plans), and the long models do not read it at all"),
    ("IOBT_flt in the 3 h and 24 h models", "its value hours ahead is unknown (see the data point); since findings "
     "10.35 the long models run on the schedule"),
    ("ADS-B positions (adsb.lol)", "the flight's own track is tier M. Other aircraft's positions up to the anchor "
     "need a live feed; the lab rung R6 reads them and gained nothing on the five month pairs (+0.05 s of MAE, "
     "findings 10.35), so no deliverable reads them. After the anchor the tier F concept members of Part D read them. "
     "Of the two delivered as tier F files beside the deliverable, rule T's member (`op_h1_f4_nodepthru`, `h1f`) "
     "reads only the arrivals' taxi-ins; the full concept F5 (`op_h1_f5_full`, `h1f5`) reads all of them, up to the "
     "anchor as well"),
    ("ERA5 reanalysis and the PRU daily delay data", "published after the fact (tier P)"),
    ("the NM join window, stale and wrong-date block times", "artefacts of how the challenge data were built (tier C); "
     "the operational models are fitted without targets above 2 h or wrong-date rows and read on the NM-joined "
     "population"),
)


# --- the families ---------------------------------------------------------------------------------

FAMILIES: tuple[Family, ...] = (
    Family(
        "airport", "Airport",
        {"apt": "The departure airport (categorical)."},
        ("airport",),
        "Layout, runway system, procedures and traffic level set each airport's taxi-out level and how it reacts to "
        "every other input. One global model over the ten airports learns those differences through this key and "
        "its interactions with the others.",
        "Swatowska 2025: one model over six A-CDM airports.",
    ),
    Family(
        "aircraft", "Aircraft type and wake category",
        {
            "aircraft_type": "ICAO type designator of the movement record (categorical).",
            "aircraft_type_flt": "ICAO type designator of the flight plan (categorical).",
            "aircraft_type_mismatch": "1 where both types exist and differ (a swap after filing, or a coding difference).",
            "wake_category": "ICAO wake turbulence category of the flight plan (categorical).",
        },
        ("actype", "fpl_type"),
        "Larger aircraft take longer to push back and start, taxi more slowly, often use their own stands, and need "
        "larger wake separation at the runway.",
        "Robinson 2013: equipment type is a systematic shifter. Lee 2016: weight class is selected. Durubi 2025: type is "
        "the second static predictor.",
        "The `_flt` columns have a missing level for records NM did not join. That marks the challenge's join (tier C) "
        "and is constant on the NM-joined population the models are read on.",
    ),
    Family(
        "operation", "Kind of operation",
        {
            "market_segment": "NM market segment, e.g. mainline, low-cost, regional, business aviation, all-cargo "
                              "(categorical).",
            "flight_type": "NM flight type, e.g. scheduled, non-scheduled, general aviation (categorical).",
            "flight_rule_mvt": "Flight rules of the movement record (categorical).",
            "flight_rule_flt": "Flight rules of the flight plan (categorical).",
        },
        ("fpl_class",),
        "Business, cargo and charter operations use other aprons and handlers than the scheduled carriers. Low-cost "
        "operations favour fast turnarounds at their own piers. Both change where the taxi starts and how long the "
        "push takes.",
        "No study isolates the market segment. Robinson 2013: carrier is a systematic shifter. Hotle 2016: carrier group "
        "in the unimpeded-time key.",
        "Same missing-level caveat as the aircraft family (tier C, constant on the NM-joined population).",
    ),
    Family(
        "operator", "Operator and flight number",
        {
            "operator_hash": "NM's aircraft operator code, hashed in the data release (categorical).",
            "carrier_icao": "The three-letter ICAO prefix of NM's callsign (categorical).",
            "carrier_flightno": "The airline designator in the record's flight number (categorical).",
            "carrier_missing": "1 where the flight number has no airline designator.",
        },
        ("operator",),
        "Airlines differ in their pushback and start-up procedures, de-icing contracts, terminals and piers. The "
        "operator is a stratifier in every taxi-time study that tests it.",
        "Idris 2002: airline or terminal, univariate R² 0.02. Robinson 2013: carrier is a shifter. Hotle 2016: carrier "
        "group in the unimpeded key.",
    ),
    Family(
        "stand", "Stand and pier",
        {
            "stand": "The departure stand of the record (categorical).",
            "apt_stand": "Airport x stand (categorical).",
            "stand_prefix": "The pier or apron of the stand: the leading letters or the numbering block of its "
                            "identifier, per airport (categorical).",
            "stand_number": "The numeric part of the stand identifier.",
            "stand_suffix": "The position suffix of the stand identifier, e.g. L, R, A (categorical).",
            "has_stand": "1 where the record has a stand (the long models only).",
        },
        ("stand",),
        "The stand fixes where the taxi starts, how far the runways are, and whether the aircraft needs a pushback "
        "or leaves under its own power. Pier and terminal shift the unimpeded time by minutes.",
        "Hotle 2016: the terminal shifts a group's unimpeded time by up to 5.5 min. Lordan 2016: (stand, runway) "
        "dummies explain about 90 % of the log variance at Barcelona (arrivals pooled). Swatowska 2025: runway and stand carry 33 % of "
        "the departure SHAP. Durubi 2025: the stand is the main static predictor.",
        "Known as a fact at H1 and H0, when the aircraft is on its stand. Hours ahead it is the stand plan, assumed "
        "final (O*).",
    ),
    Family(
        "runway", "Departure runway",
        {
            "runway": "The departure runway of the record (categorical).",
            "apt_runway": "Airport x runway (categorical).",
            "apt_stand_runway": "Airport x stand x runway, the route key (categorical).",
        },
        ("runway", "stand"),
        "The runway fixes the end of the route: its length, its crossings and the holding point where the queue "
        "forms. Runway configurations shift the mean taxi-out by minutes.",
        "Idris 2002: configuration means from about 15 to 23 min at Boston. Park & Kim 2022: operating direction is "
        "significant. Robinson 2013: runway assignment is a variance source.",
        "Tier O* everywhere. The data hold the runway actually used, and the models assume the planned one is known. "
        "At H1 the actual runway equals the dominant departure runway of the hour before the planned off-block on "
        "66.3 % of departures, and lies in the set of runways in use on 93.1 %. The H1 model before the plans after "
        "the anchor (`op_h1_r4_nodoy`) is shown beside its runway-blind twin (`op_h1_r4_nodoy_rwyblind`). The 3 h and "
        "24 h models are runway-blind: hours ahead the recorded runway matches the dominant runway of the hour before "
        "the anchor on only 53 % (3 h and 24 h), and lies among the runways in use on 75 % "
        "(docs/op_runway_knowability.txt).",
    ),
    Family(
        "destination", "Destination",
        {
            "ades": "The destination airport (categorical).",
            "ades_country": "Its ICAO nationality prefix, the first two letters (categorical).",
            "is_domestic": "1 where the destination shares the departure airport's prefix.",
        },
        ("ades",),
        "The destination sets the departure direction, and with it the SID and often the runway. It also sets the "
        "aircraft's fuel weight and whether the flight leaves from a Schengen, non-Schengen or international pier.",
        "Lee 2016: the departure fix is selected at Charlotte.",
    ),
    Family(
        "stand_position", "Stand position (map)",
        {
            "geo_stand_east_m": "Stand position, metres east of the airport reference point (OSM parking position).",
            "geo_stand_north_m": "Stand position, metres north of the airport reference point.",
            "geo_stand_arp_dist_m": "Distance from the stand to the airport reference point, metres.",
            "geo_stand_zone": "The apron zone of the stand: k-means over the OSM parking positions (categorical).",
            "geo_stand_matched": "1 where the record's stand was matched to a mapped position.",
            "geo_stand_match_source": "How it was matched: exact, normalised, without suffix, gate, manual, none "
                                      "(categorical).",
        },
        ("stand", "osm", "ourairports"),
        "A position generalises where the stand identifier does not. Stands that are rare in the training data, or "
        "new, inherit the behaviour of their neighbours. The zone groups stands that share an apron exit.",
        "Wang 2020: distance ranks high in random-forest importance. Bekkers 2025: OSM routing reproduces stand-runway "
        "distances.",
        "NaN means not matched; OSM coverage differs by airport (docs/data_sources.md). Same stand tier as the stand "
        "family.",
    ),
    Family(
        "runway_distance", "Distance to the runway (map)",
        {
            "geo_gc_dist_m": "Great-circle distance from the stand to the departure runway's threshold, metres.",
            "geo_rwy_matched": "1 where the runway designator was matched to a threshold.",
            "geo_rwy_length_m": "Length of the departure runway, metres.",
            "geo_rwy_threshold_bearing_deg": "Bearing from the stand to the threshold, relative to the runway heading "
                                             "(near 0: the stand lies behind the threshold; near 180: beyond it, so "
                                             "the aircraft taxis back along the runway's length).",
        },
        ("stand", "runway", "osm", "ourairports"),
        "Distance is the floor of the taxi time: the unimpeded time is mostly distance over taxi speed.",
        "Ravizza 2012: distance and turns rank high. Wang 2020: distance is second in random-forest importance. "
        "Günther 2011: static stand-runway tables place 42 % of departures within 2 min at Frankfurt.",
        "Reads the own runway, tier O*. The long models drop it.",
    ),
    Family(
        "route", "Taxi route (map)",
        {
            "geo_route_dist_m": "Length of the shortest route from the stand to the threshold over the OSM taxiway "
                                "graph, metres; runway edges are penalised x3.",
            "geo_route_ratio": "Route length over the great-circle distance (detours).",
            "geo_route_rwy_crossings": "Runways the route crosses.",
            "geo_route_found": "1 where a route was found.",
        },
        ("stand", "runway", "osm", "ourairports"),
        "The route's length and its runway crossings set the unimpeded taxi time. Each crossing waits for a clearance "
        "between arrivals and departures.",
        "Jeong 2020: the unimpeded time is predictable to about 1 min RMSE given the route. Tang 2024 isolates runway "
        "crossings (on the arrival side).",
        "Reads the own runway, tier O*. The long models drop it.",
    ),
    Family(
        "calendar", "Calendar",
        {
            "hour_local": "Local hour at the anchor.",
            "minute_of_day": "Local minute of the day at the anchor.",
            "dow": "Local day of the week at the anchor (0 = Monday).",
            "is_weekend": "1 on Saturday and Sunday.",
            "month": "Local calendar month.",
            "sched_hour_local": "Local hour of the scheduled off-block.",
        },
        ("sobt", "iobt", "airport"),
        "Waves of departures, staffing and night procedures follow the clock. The season carries the winter "
        "operations and the summer peak.",
        "Goldberg 2008: summer peak. Durubi 2025: hour of day as a table key. Lordan 2016: weekday dummies.",
        "The day of year is dropped in every deliverable (findings 10.35). Kept, it would map 2025's moving holidays "
        "onto 2026 dates. On the schedule clock of the 3 h and 24 h models, the hour at the planned off-block equals "
        "the scheduled hour.",
        {"hl24": "the planned off-block (the schedule), not the anchor: 24 h earlier the weekday is another",
         "hl3": "the planned off-block (the schedule)", "h1": "the H1 anchor", "h0": "the recorded off-block"},
    ),
    Family(
        "planned_delay", "Planned and realised delay",
        {
            "plan_minus_sched_min": "Last filed off-block time (IOBT, else LOBT) minus the schedule, minutes "
                                    "(positive = planned late).",
            "delay_vs_plan_min": "H0 only: the recorded off-block minus the planned one, whole minutes.",
            "delay_vs_sched_min": "H0 only: the recorded off-block minus the schedule, whole minutes.",
        },
        ("iobt", "sobt", "offblock_feed"),
        "A late flight can be pushed with priority, and late flights cluster in disrupted periods, so the delay at the "
        "push says something about both the flight and the airport's state.",
        "Lian 2018: the departure delay at the push is a top factor. Groppe 2011 and Jung 2019: the reliability of the "
        "TOBT limits every pre-departure estimate.",
        "At H1 the column assumes the last filed plan is known 15 min ahead, as an A-CDM TOBT is (O*). The data hold "
        "its last value without the time of its last update. The sensitivity `op_h1_r4_nodoy_noplan` drops it "
        "together with the plan backlog.",
    ),
    Family(
        "metar_now", "Weather now (METAR)",
        {
            "wx_wind_kt": "Mean wind speed, kt.",
            "wx_gust_kt": "Gust speed, kt (the mean wind where no gust is reported).",
            "wx_headwind_kt": "Headwind component on the own departure runway, kt.",
            "wx_crosswind_kt": "Crosswind component on the own departure runway, kt.",
            "wx_vis_m": "Minimum visibility, m (capped at 10 km).",
            "wx_rvr_min_m": "Minimum runway visual range, m (2,000 m where none is reported).",
            "wx_ceiling_ft": "Ceiling, ft (capped at 5,000 ft).",
            "wx_temp_c": "Air temperature, °C.",
            "wx_dewpt_spread_c": "Temperature minus dew point, °C.",
            "wx_precip_int": "Precipitation intensity: 0 none, 1 light, 2 moderate, 3 heavy.",
            "wx_snowfrz": "1 with snow, freezing or ice precipitation.",
            "wx_ts": "1 with a thunderstorm or cumulonimbus/towering cumulus.",
            "wx_fog": "1 with fog.",
            "wx_lvp": "1 under low-visibility conditions: RVR below 600 m (else visibility below 600 m) or ceiling "
                      "below 200 ft.",
            "wx_deice": "1 under likely de-icing conditions: at most 3 °C with precipitation, fog, mist or visibility "
                        "below 1,500 m; any freezing or ice precipitation; or frost (at most 0 °C, dew point within "
                        "2 °C).",
        },
        ("metar",),
        "Low-visibility procedures reduce runway and taxiway capacity and slow the taxi. De-icing adds a stop on a "
        "remote pad, or a wait at the stand. Snow and freezing precipitation slow the taxi and close taxiways. Wind "
        "decides the runway configuration.",
        "Park & Kim 2022: wind, pressure and precipitation are significant at two Korean airports. Kim 2021: METAR "
        "elements in an A-CDM random forest. Pejovic 2009: delay probability rises with fog, snow and storms at "
        "Heathrow.",
        "Head- and crosswind read the own runway (O*). A report is in force until the next one, for at most 90 min. "
        "NaN means no report in force.",
        {"hl24": "the anchor minus 5 min, as persistence", "hl3": "the anchor minus 5 min, as persistence",
         "h1": "the anchor minus 5 min", "h0": "the off-block minus 5 min"},
    ),
    Family(
        "metar_recent", "Recent weather (METAR windows)",
        {
            "wx_b60_min_vis_m": "Lowest visibility in the last 60 min, m.",
            "wx_b60_min_ceiling_ft": "Lowest ceiling in the last 60 min, ft.",
            "wx_b60_max_gust_kt": "Highest gust in the last 60 min, kt.",
            "wx_b60_any_precip": "1 with any precipitation in the last 60 min.",
            "wx_b180_any_snowfrz": "1 with any snow or freezing precipitation in the last 3 h.",
            "wx_b180_lvp_min": "Minutes under low-visibility conditions in the last 3 h.",
            "wx_deice_min_6h": "Minutes under de-icing conditions in the last 6 h.",
            "wx_snowfrz_hours_24h": "Hours of snow or freezing precipitation in the last 24 h.",
            "wx_hours_since_snowfrz": "Hours since the last snow or freezing report (up to 72 h).",
        },
        ("metar",),
        "Snow and freezing weather leave work behind after the precipitation stops: contaminated surfaces, closed "
        "taxiways and de-icing queues. Recent low visibility means procedures may still be in force.",
        "Pickup 2016: de-icing is handled by separate A-CDM modules. Scheicher 2023: de-icing usage is predictable at "
        "Vienna.",
        "Windows end at the METAR-side reading time of the moment (see the weather-now family).",
    ),
    Family(
        "taf", "Forecast weather (TAF)",
        {
            "taf_issue_age_h": "3 h and 24 h only: hours from the TAF's issue to the anchor.",
            "taf_is_amd": "1 when the TAF read is an amendment.",
            "taf_amd_12h": "Amendments issued in the 12 h up to the anchor.",
            "taf_vis_m": "Forecast prevailing visibility, m.",
            "taf_ceiling_ft": "Forecast prevailing ceiling, ft.",
            "taf_wind_kt": "Forecast prevailing wind, kt.",
            "taf_snowfrz": "1 when snow or freezing precipitation is forecast as prevailing.",
            "taf_worst_vis_m": "Worst forecast visibility, including TEMPO and PROB groups, m.",
            "taf_worst_ceiling_ft": "Worst forecast ceiling, including TEMPO and PROB groups, ft.",
            "taf_worst_snowfrz": "1 when any group forecasts snow or freezing precipitation.",
            "taf_worst_ts": "1 when any group forecasts a thunderstorm.",
            "taf_temp_prob": "Highest probability (%) of the active TEMPO and PROB groups (a TEMPO without one counts "
                             "50; 0 when none is active).",
            "taf_f60_worst_vis_m": "Worst forecast visibility over the following hour of the same TAF, m.",
            "taf_f60_worst_snowfrz": "1 when snow or freezing precipitation is forecast in the following hour.",
        },
        ("taf",),
        "Runway configurations, de-icing crews and flow measures are planned from the forecast. A busy amendment "
        "record marks weather that is hard to plan for. Hours ahead, the forecast is the only weather for the hour "
        "of the flight.",
        "Prevailing and worst-case state follow the TAF rules: FM replaces, BECMG applies at the end of its window, "
        "TEMPO and PROB add a worst case (module prc.features.weather_fcst).",
        "The TAF archive is Ogimet's, which has no open licence; in operations the TAF comes from the met service. The "
        "sensitivity `op_h1_r4_nodoy_notaf` drops the 13 TAF columns of H1. At 24 h a 24 h TAF (Barcelona) mostly "
        "does not reach the planned off-block, so it is NaN.",
        {"hl24": "the TAF issued by the anchor that covers the planned off-block",
         "hl3": "the TAF issued by the anchor that covers the planned off-block",
         "h1": "the TAF issued by the anchor that covers the anchor", "h0": "the TAF in force at the off-block"},
    ),
    Family(
        "traffic_counts", "Traffic up to the anchor",
        {
            "dep_apt_b15": "Takeoffs at the airport in the last 15 min (the flight itself excluded).",
            "dep_apt_b60": "Takeoffs at the airport in the last 60 min.",
            "dep_rwy_b15": "Takeoffs on the own departure runway in the last 15 min.",
            "dep_rwy_b30": "Takeoffs on the own departure runway in the last 30 min.",
            "arr_apt_b15": "Landings at the airport in the last 15 min.",
            "arr_apt_b60": "Landings at the airport in the last 60 min.",
            "arr_rwy_b15": "Landings on the own departure runway in the last 15 min (mixed-mode use).",
            "arr_taxiing": "Arrivals landed and not yet in block at the anchor, at most an hour after landing.",
            "gap_last_dep_rwy_s": "Seconds since the last takeoff on the own runway (capped at 2 h).",
            "gap_last_arr_rwy_s": "Seconds since the last landing on the own runway (capped at 2 h).",
        },
        ("dep_events", "arr_events", "runway"),
        "Runway throughput and the surface load right now set the queue the flight will join. Taxi-out grows once "
        "the departures on the surface exceed what the runway can take. Arrivals compete for the same runway and "
        "taxiways where runways interact.",
        "Idris 2002: the departures on the surface at the push explain R² 0.19, about 1.5 min per aircraft. "
        "Simaiakis & Balakrishnan 2015: queuing model of the departure process. Clewlow 2010: arrivals during the "
        "taxi correlate with taxi-out where runways interact.",
        "Counts per own runway read the own runway (O*). Only events at or before the anchor enter.",
    ),
    Family(
        "runway_use", "Runway configuration in use",
        {
            "rwy_distinct_b60": "Distinct departure runways used in the last 60 min.",
            "rwy_own_share_b60": "Share of the last 60 min's takeoffs on the own runway.",
            "rwy_own_dominant_b60": "1 where the own runway was the most used departure runway of the last 60 min.",
            "rwy_config_age_min": "Minutes since the airport's dominant departure runway last changed (15-min "
                                  "buckets complete by the anchor; capped at 12 h).",
        },
        ("dep_events", "runway"),
        "The configuration in use decides routes and queues. A recent change means re-sequencing and longer routes "
        "while the traffic adapts. A runway outside the dominant configuration is a special case: a long route, or "
        "a heavy aircraft needing the long runway.",
        "Idris 2002: configuration. Maslovara 2021: at Zurich the tailwind forces a less favourable configuration.",
        "The distinct count and the configuration age are airport-level (O). The own share and dominance read the own "
        "runway (O*).",
    ),
    Family(
        "arrival_anomaly", "Arrivals' taxi-in anomaly",
        {
            "arr_anom_b30_mean": "Mean taxi-in anomaly of the arrivals that blocked in during the last 30 min: "
                                 "taxi-in minus its (airport, runway, stand) median.",
            "arr_anom_b30_n": "How many arrivals that mean covers.",
            "arr_anom_b60_mean": "The same over the last 60 min.",
            "arr_anom_b60_n": "How many arrivals that mean covers.",
        },
        ("arr_events",),
        "Arrivals that just finished their taxi measure the surface state directly: snow, de-icing traffic, closed "
        "taxiways and congestion slow them as they will slow the departure.",
        "Clewlow 2010: impact of arrivals on departure taxi operations.",
        "The medians are a target-free table over every arrival in the frame, the 2026 months included, so they are "
        "transductive in time. No departure target enters them.",
    ),
    Family(
        "turnaround", "The flight's own stand",
        {"stand_min_since_inblock": "Minutes since the last arrival blocked in at the flight's stand, at or before "
                                    "the anchor (capped at a day)."},
        ("arr_events", "stand"),
        "A tight turnaround pushes late and under pressure. An aircraft parked for many hours is cold-soaked and more "
        "likely to need de-icing in winter.",
        "Okwir 2024: turnaround milestones as predictors.",
    ),
    Family(
        "nm_history_short", "Recent taxi-outs of departed flights (NM, last 30-60 min)",
        {
            f"nm_{lvl}_{win}_{stat}": text
            for lvl, where in (("apt", "at the airport"), ("rwy", "on the own runway"))
            for win, span in (("b30", "30"), ("b60", "60"))
            for stat, text in (
                ("anom_mean", f"Mean taxi-out anomaly of the flights that took off {where} in the last {span} min: "
                              "NM's reading (takeoff - AOBT_3) minus its (airport, stand, runway) median."),
                ("anom_share5", f"Share of those flights ({where}, last {span} min) whose anomaly exceeds +5 min."),
                ("push_delay_mean", f"Their mean push delay ({where}, last {span} min): AOBT_3 minus their planned "
                                    "off-block."),
            )
        },
        ("nm_actuals", "dep_events", "iobt"),
        "The taxi-outs that just ended are the most direct measure of the surface state at the moment: if they ran "
        "long, the next ones will. The push delay of the others shows whether the airport is running late.",
        "Günther 2011: at Frankfurt a traffic-aware calculation at the off-block places 66 % of departures within "
        "2 min, against 42 % for static tables.",
        "The medians come from the fold's fit rows only. They leave out the fit rows of the query's own planned month, "
        "so a fit row's own reading never enters them. AOBT_3 is back-computed from the takeoff at LTFM and partly "
        "at EDDM, LEBL, LSZH and LEMD, so the anomaly carries little there. Per-runway columns read the own runway "
        "(O*).",
    ),
    Family(
        "plan_backlog", "Plan backlog",
        {
            "backlog_apt": "Other departures planned off (last filed plan) at or before the anchor and not yet airborne, "
                           "at the airport; each counted for at most 3 h after its plan.",
            "backlog_rwy": "The same on the own runway (the others grouped by their recorded runway).",
            "backlog_pier": "The same at the own pier (the others grouped by their recorded stand).",
        },
        ("iobt", "dep_events", "runway", "stand"),
        "The departures planned off but not yet airborne are the queue that forms ahead of the flight. This is the "
        "plan-based version of the count of departures on the surface.",
        "Idris 2002: departures on the surface at the push. Simaiakis & Balakrishnan 2015.",
        "O* twice. The others' last filed plans are assumed known at the anchor. The per-runway and per-pier counts "
        "group not-yet-airborne flights by their recorded runway and stand. The sensitivity "
        "`op_h1_r4_nodoy_noplan` drops the three backlog columns together with the planned delay.",
    ),
    Family(
        "fut_plan", "Other flights' plans after the anchor",
        {
            **{f"fplan_dep_plan_{lvl}_{w}": f"Other departures planned off (last filed plan, else the schedule) {where} "
                                            f"in the {span} min after the anchor (the own plan excluded)."
               for lvl, where in (("apt", "at the airport"), ("rwy", "on the own runway"), ("pier", "at the own pier"))
               for w, span in (("w15", "15"), ("w30", "30"), ("w60", "60"))},
            **{f"fplan_arr_sched_{lvl}_{w}": f"Arrivals scheduled (in-block) {where} in the {span} min after the anchor."
               for lvl, where in (("apt", "at the airport"), ("pier", "at the own pier"))
               for w, span in (("w15", "15"), ("w30", "30"), ("w60", "60"))},
        },
        ("iobt", "sobt", "runway", "stand"),
        "The departures planned off and the arrivals scheduled in the next hour are the traffic the flight meets on "
        "its way to the runway: the departures ahead of it in the runway queue, the pushbacks on its pier, the "
        "arrivals on the same aprons and runways. At H1 an airport holds their last filed plans and the schedule, "
        "which a DMAN or an AMAN turns into a sequence, so the H1 deliverable reads them (F1, findings 10.37): the "
        "forward-looking twin of the plan backlog, and at H1 the counterpart of the long models' scheduled demand.",
        "Simaiakis & Balakrishnan 2015: a queue model fed with the departure demand. EUROCONTROL 2017: the A-CDM "
        "pre-departure sequence is built from these plans.",
        "O*: the other flights' last filed plans are assumed known at the anchor, up to an hour before their planned "
        "off-block. The data hold each plan's last value without the time of its last update, so a plan revised after "
        "the anchor enters with its revised value. The other flights' runway and stand are the recorded ones, standing "
        "in for the planned allocation. Cancellations are not in the data, so the flights that operated are taken from "
        "the record (a mild hindsight, as in the scheduled demand). In the H1 deliverable since 13 Oct (findings "
        "10.37). Rule T measured what they add to the H1 model before them (`op_h1_r4_nodoy`): 2.6 s of OP MAE at "
        "seed 42 on the January and July 2025 holdout from 9 January on, at every airport, from 1.6 s at EDDF to 4.4 s "
        "at LIRF; 2.4 s for the three refitted seeds on the full holdout (findings 10.36).",
    ),
    Family(
        "scheduled_demand", "Scheduled demand around the planned off-block",
        {
            f"dem_{ref}_{lvl}_{w}": f"{what} {where} within ±{span} min of the own scheduled off-block."
            for ref, what in (("dep_sched", "Other departures scheduled"), ("arr_sched", "Arrivals scheduled (in-block)"))
            for lvl, where in (("apt", "at the airport"), ("rwy", "on the own runway"), ("pier", "at the own pier"))
            for w, span in (("w30", "30"), ("w60", "60"))
        },
        ("sobt", "runway", "stand"),
        "Hours ahead, no traffic has been observed yet. The schedule's own peaks are the best guide to the congestion "
        "a flight will meet: a slot in the middle of a departure wave queues longer than one in a lull.",
        "Simaiakis & Balakrishnan 2015: taxi-out grows with the departures on the surface. The schedule is their "
        "planned version.",
        "The per-runway and per-pier counts group the other flights by their recorded runway and stand (O*). The "
        "runway-blind deliverables drop the per-runway counts. Cancellations are not in the data, so the flights "
        "that operate are taken from the record: a mild hindsight that every schedule-based count shares.",
    ),
    Family(
        "nm_history_long", "Taxi-outs of departed flights before a long anchor (NM, last 3-24 h)",
        {
            f"hist_{lvl}_{win}_{stat}": text
            for lvl, where in (("apt", "at the airport"), ("rwy", "on the own runway"))
            for win, span in (("b3h", "3 h"), ("b24h", "24 h"))
            for stat, text in (
                ("n", f"Flights with a plausible NM reading that took off {where} in the {span} before the anchor."),
                ("anom_mean", f"Their mean taxi-out anomaly ({where}, {span}): takeoff - AOBT_3 minus its (airport, "
                              "stand, runway) median."),
                ("anom_share5", f"Share of them ({where}, {span}) with an anomaly above +5 min."),
                ("push_delay_mean", f"Their mean push delay ({where}, {span}): AOBT_3 minus their schedule."),
            )
        },
        ("nm_actuals", "dep_events", "sobt"),
        "Persistence. A day that runs slow (snow, works, a closed taxiway) tends to keep running slow. The signal is "
        "weaker a day ahead than 3 h ahead.",
        "Zhou 2020: recovery speed after a disruption differs by event and airport.",
        "Fold-dependent like the short history: the medians come from the fit rows, month-out. A window that reaches "
        "before the frame's records is NaN, not empty (`_horizon.window_covered`). Per-runway columns read the own "
        "runway (O*); the runway-blind deliverables drop them.",
    ),
)


# --- traffic after the H1 anchor: the concept members' inputs (findings 10.36) ----------------------------------
# The other flights' plans after the anchor (`fut_plan`, O*) are above: the H1 deliverable reads them since findings
# 10.37. What follows is tier F, read only by the concept members, and the ADS-B picture before the anchor
# (`adsb_now`), which no deliverable reads: the lab rungs R6 and R2 + ADS-B and the full concept F5. Two concept
# members are delivered beside the H1 deliverable as tier F files (findings 10.37): rule T's member
# `op_h1_f4_nodepthru` (`h1f`) and F5 (`h1f5`), whose realised departure side carries traces of the own slot.

READS = {"f15": "the planned off-block (H1 + 15 min)", "f30": "H1 + 30 min", "f45": "H1 + 45 min"}
"""The read times of the tier F blocks (`prc.features.operational_future.SHIFTS`)."""
_LATER = {"hl24": "not used", "hl3": "not used", "h0": "not used"}

DATA_POINTS = DATA_POINTS + (
    DataPoint(
        "aman", "(not in the data) the arrival manager's landing sequence (ELDT, TLDT) and stand allocation",
        "The planned landing and in-block times of the arrivals in the hour after the anchor, and their stands.",
        "AMAN (arrival management); the A-CDM ELDT; the airport's stand allocation",
        {**_LATER, "h1": "F: planned by the AMAN; the challenge data hold the realised landings and in-blocks of the "
                         "arrivals, which stand in"},
        "other flights: the arrivals after the anchor (rule T's F2-F5 and its member `op_h1_f4_nodepthru`, so both "
        "delivered tier F files: the member's `h1f` and F5's `h1f5`)",
    ),
    DataPoint(
        "dman", "(not in the data) the departure manager's pre-departure sequence (TSAT, TTOT) and runway plan",
        "The planned start-up and takeoff times of the departures in the hour after the anchor, and the runway "
        "configuration they are planned on.",
        "DMAN (departure management); A-CDM TSAT and TTOT",
        {**_LATER, "h1": "F: planned by the DMAN; the realised takeoffs, runway use and NM readings of the other "
                         "departures stand in"},
        "other flights: the departures after the anchor (rule T's F2-F5, so the delivered F5 file `h1f5`, whose gain "
        "from them carries traces of the own slot; not rule T's member, delivered as `h1f`, which reads no realised "
        "departure-side column)",
    ),
    DataPoint(
        "asmgcs", "(not in the data) the surface manager's plan (A-SMGCS planning and routing)",
        "The planned pushbacks, routes, taxi times and runway holding of the aircraft on the surface after the anchor.",
        "A-SMGCS planning function (surface management)",
        {**_LATER, "h1": "F: planned by the surface manager; the realised ADS-B surface events of the other aircraft "
                         "stand in"},
        "other aircraft: the surface after the anchor (rule T's F4 and F5, so the delivered F5 file `h1f5`; rule T's "
        "member, delivered as `h1f`, reads only the arrivals' taxi-ins)",
    ),
    DataPoint(
        "adsb_surface", "public ADS-B surface events of every aircraft (adsb.lol; prc adsb-surface, snapshot s1)",
        "Pushes, taxi-outs and taxi-ins, runway holds, remote stops, takeoffs and landings, each with the second it "
        "became knowable to a receiver.",
        "adsb.lol (ODbL 1.0), received by volunteer feeders; airport coverage varies (none at LTFM, little at LFPG)",
        {**_LATER, "h1": "yes up to the anchor (other aircraft's public positions: rung R6 and F5's actual part; no "
                         "deliverable reads them, since R6 gained nothing, and of the tier F files only F5's `h1f5`); "
                         "F after it, as the stand-in for the surface plan"},
        "other aircraft, the own airframe removed (up to the anchor the events under its callsign or originating "
        "within 100 m of its stand, after it also its whole ground stay; rule T's F4 and F5, so the delivered F5 file "
        "`h1f5`, before the anchor too; rule T's member, delivered as `h1f`, reads only the arrivals' taxi-ins after "
        "the anchor)",
    ),
)


def _reread(ids: tuple[str, ...], prefix: str, skip: tuple[str, ...] = ()) -> dict[str, str]:
    """The columns of the families ``ids`` (minus ``skip``) as the F blocks read them at each of `READS`."""
    base = {k: v for f in FAMILIES if f.id in ids for k, v in f.columns.items() if k not in skip}
    return {f"{prefix}_{tag}_{k}": f"Read at {when} instead of at the anchor. {v}" for tag, when in READS.items()
            for k, v in base.items()}


_ADSB = {
    "taxi_out_now": "Other aircraft taxiing out (ADS-B) at the read time.",
    "taxi_out_now_norm": "The same, divided by the airport's ADS-B detection share at that time.",
    "taxi_in_now": "Arrivals taxiing in (ADS-B) at the read time.",
    "taxi_out_start_15": "Taxi-outs of other aircraft begun (from a push, a stop or a first sighting near a stand) in "
                         "the 15 min before the read time.",
    "remote_now": "Other aircraft stopped at a remote position (de-icing pads, remote holding) at the read time.",
    "remote_60": "Remote stops begun in the 60 min before the read time.",
    "taxi_out_elapsed_mean_now": "Mean time since their taxi-out began of the aircraft taxiing out at the read time, s.",
    "taxi_stop_s_mean_60": "Mean stationary time during the taxi of the takeoffs in the 60 min before the read time, s.",
    "taxi_stop_share_60": "Share of those taxi times spent stationary.",
    "remote_dur_mean_60": "Mean duration of the remote stops that ended in the 60 min before the read time, s.",
    "rwy_hold_now": "Aircraft holding at a runway at the read time (the departure queue).",
    "rwy_own_hold_now": "The same at the own runway's ends (where the holds' runway labels agree with the takeoffs "
                        "that follow them).",
    "rwy_hold_dur_mean_30": "Mean duration of the runway holds that ended in the 30 min before the read time, s.",
    "push_15": "Pushbacks of other aircraft in the 15 min before the read time.",
    "push_15_norm": "The same, divided by the airport's push detection share.",
    "taxi_dur_mean_30": "Mean taxi-out time of the push-origin takeoffs in the 30 min before the read time, s (the "
                        "other flights' realised taxi times).",
    "taxi_dur_n_30": "How many takeoffs that mean covers.",
    "taxi_dur_mean_60": "The same over 60 min, s.",
    "taxi_dur_n_60": "How many takeoffs that mean covers.",
}

_F_CAVEAT = ("Tier F: not available as computed. In operations the value comes from a plan with its own errors, so the "
             "gain is an upper bound for perfect plans. The own movement is excluded, but traces of it remain through "
             "the resources it shares with the others (one runway slot, the next arrival at the own stand, the pier's "
             "pushback lanes, the de-icing pads, the holding queue). Rule T removes the departure-side traces by "
             "construction: its member, `op_h1_f4_nodepthru` (delivered as `h1f`), reads no realised departure-side "
             "column, and nothing from that side is credited. Three traces stay in the member: the next arrival at the "
             "own stand (unmeasured); the missingness of its ADS-B taxi-in columns, which depends on finding the own "
             "ADS-B takeoff (about 0.15 s of MAE for F4); and, where a runway is in mixed mode, the landings on the own "
             "runway (unmeasured; findings 10.36). The full concept F5, delivered at the owner's request as the second "
             "tier F file (`h1f5`, findings 10.37), reads the realised departure side as well: its gain carries traces "
             "of the own slot, which rule T could not credit, so it is an upper bound even for perfect plans.")

FAMILIES = FAMILIES + (
    Family(
        "fut_records", "Realised traffic after the anchor: the airport's records",
        _reread(("traffic_counts", "runway_use", "arrival_anomaly"), "frec",
                skip=("gap_last_dep_rwy_s", "gap_last_arr_rwy_s")),
        ("dman", "aman", "dep_events", "arr_events", "runway"),
        "The takeoffs, landings and arrivals taxiing in during the flight's own off-block and taxi window are the "
        "traffic it actually meets. A DMAN plans the takeoff sequence and the runway configuration, an AMAN the "
        "landings; with perfect plans these counts would be known at H1.",
        "Idris 2002 and Wang 2020: the traffic during the own taxi is the strongest single factor in the literature, "
        "read there in hindsight. Here the window is fixed relative to the anchor, never tied to the own movement.",
        _F_CAVEAT + " The gaps since the last takeoff and landing on the own runway are left out: read after the own "
        "takeoff they would locate it.",
    ),
    Family(
        "fut_nm", "Realised taxi-outs and plan backlog after the anchor: NM",
        _reread(("nm_history_short", "plan_backlog"), "fnm"),
        ("dman", "nm_actuals", "dep_events", "iobt"),
        "How long the departures of the next 45 min taxi, and how late they push, describe the surface the flight "
        "taxis on. In operations the surface manager and the DMAN estimate these times for the flights in the sequence.",
        "Günther 2011: a traffic-aware calculation at the off-block. Wang 2020: recent speeds of other aircraft rank "
        "among the strongest inputs.",
        _F_CAVEAT + " Fold-dependent: the NM proxy medians come from the fit rows only, month-out.",
    ),
    Family(
        "fut_adsb", "Realised surface picture after the anchor: public ADS-B",
        {f"fadsb_{tag}_{k}": f"Read at {when}. {v}" for tag, when in READS.items() for k, v in _ADSB.items()},
        ("asmgcs", "adsb_surface"),
        "The aircraft pushing, taxiing, de-icing and holding around the flight's own taxi are the surface picture an "
        "A-SMGCS plans. Public ADS-B shows it where reception is good.",
        "Wang 2020: other aircraft taxiing at the push (group B) and their recent speeds (group D).",
        _F_CAVEAT + " The own airframe is identified through its own ADS-B takeoff and its whole ground stay is "
        "removed; a departure whose own takeoff is not found is NaN. Coverage-gated (NaN, never zero, where reception "
        "is not established); no reception at LTFM and little at LFPG.",
    ),
    Family(
        "adsb_now", "Surface picture up to the anchor: public ADS-B",
        {f"adsb_{k}": v.replace("the read time", "the anchor (minus the feed's 30 s)") for k, v in _ADSB.items()},
        ("adsb_surface",),
        "What the surface looks like when the prediction is made: the queue the flight will join, the aircraft "
        "already taxiing, the pushes around it. An airport with an A-SMGCS sees this directly; public ADS-B shows it "
        "where reception is good.",
        "Wang 2020: other aircraft taxiing at the push (group B) add 2-19 % of RMSE over the flight's own attributes.",
        "Tier O (other aircraft's public positions up to the anchor), the own-runway hold O*. Rung R6 (findings 10.31, "
        "10.34): no gain over R4 on the five month pairs (+0.05 s, findings 10.35), so no deliverable reads it. The "
        "full concept F5 reads it beside the traffic after the anchor (0.2 s of OP MAE from F4 to F5 in rule T's "
        "stage 1, findings 10.36) and is delivered at the owner's request as the second tier F file (`h1f5`, findings "
        "10.37); rule T's member (`h1f`) does not read it. The coverage columns gate it and are never inputs.",
    ),
)


def family_of(column: str) -> Family | None:
    """The family whose columns hold ``column`` (by its name without the moment prefix), or None."""
    key = strip_prefix(column)
    hits = [f for f in FAMILIES if key in f.columns]
    return hits[0] if len(hits) == 1 else None


def definition_of(column: str) -> str | None:
    fam = family_of(column)
    return None if fam is None else fam.columns[strip_prefix(column)]
