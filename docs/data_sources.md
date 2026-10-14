# Data sources and licences

This document lists every data source the repository uses or can fetch. For each source it gives what the source is
used for, whether the submitted pipeline and the operational models (`docs/operational_model.md`, findings 10.35) use
it, its licence and the attribution it requires, and how to obtain it.
"Submitted pipeline" means the lineage of the best upload, v70 (board 220.3396; `docs/submission_manifest.md`,
findings 10.30), whose recipe the scored final file applies to the final release (findings 10.34); v70 reads no source
beyond v62's (220.5756, findings 10.24). That is the base model (the blocks listed in `docs/feature_table.md`) plus
the ADS-B correction stage: the post-operations analysis model of `docs/report.md` Part 1 (the challenge model in the
code, the configurations and the supporting documents; section 4). The tiers come from the availability table in
`docs/feature_table.md`:

- **O**: operational;
- **T**: anchored on the takeoff;
- **A**: after off-block;
- **M**: own movement;
- **P**: published later;
- **C**: challenge artefact;
- **O***: operational under an assumption (the planned runway, the stand plan, the last filed off-block or the other
  flights' plans taken as known; `docs/feature_table.md`);
- **F**: planned in the concept, realised in the challenge data: in the operational models, the other flights' realised
  movements after the H1 anchor, standing in for AMAN/DMAN/A-SMGCS plans; only the two tier F files beside the H1
  deliverable read them (`docs/operational_features.md` Part D; findings 10.36, 10.37).

The organisers' eligibility page (<https://prc-data-challenge-2026.netlify.app/eligibility.html>, read 7 Oct 2026)
asks that "All used external datasets are openly accessible/usable and documented" and that "All additional datasets
used are openly available under an open source license." The repository distributes no data: `data/`, `cache/` and
`runs/` are gitignored. Every source the repository uses is fetched by the command in the last column; the
exceptions are the competition files, which come from the organisers, and OPDI, which is listed but not used.

The repository's own code, documents and figures are licensed under `GPL-3.0-or-later` (see `LICENSE`).

Every licence statement in section 2 was read on the provider's own page on **7 October 2026**. The URL is given
with each statement.

## 1. Data Overview

| source | what we use it for | used by the submitted pipeline (tier) | used by the operational models (tier) | licence | required attribution | download URL / access path | command |
|---|---|---|---|---|---|---|---|
| **Competition files** (organisers, PRC Data Challenge 2026): 12 monthly training parquets for 2025, `ranking.parquet` (Jan+Jul 2026) and the template `submitting.parquet` | Everything: the movements, the target (2025 departures only), stands, runways, schedule; the frame every feature block reads | **yes**, every tier (O, T, A, C); all blocks | **yes**: the flight's schedule, plan record, stand and runway, the other flights' movement records up to the anchor (O; the planned runway O*) and their schedules after it (the 3 h and 24 h models' scheduled demand around the planned off-block: O, per own runway or pier O*; the plans in the hour after the H1 anchor, read by the H1 deliverable F1 since 13 Oct, findings 10.37: O for the arrivals scheduled at the airport, O* for the rest, since the other departures' plans are taken as final and the other flights' recorded runway and stand as their planned ones). The two tier F files beside the deliverable (findings 10.36, 10.37) also read the other flights' realised records up to 45 min after the H1 anchor (F): `h1f` the arrivals' only, `h1f5` the departures' too | **None stated.** The organisers' data page gives no licence or terms of use. Use is governed by the challenge rules. | None stated | OpenSky Network MinIO buckets, with access keys the team generates for its OSN account (data page <https://prc-data-challenge-2026.netlify.app/data.html>, which points to MinIO client instructions on an earlier challenge page). The files go into `data/competition_files/` under the names the data page lists. | manual download with the MinIO client, then `python -m prc prepare` |
| **EUROCONTROL Network Manager flight list** (the `_flt` columns inside the competition files: `IOBT_flt`, `EOBT_1_flt`, `LOBT_flt`, `AOBT_3_flt`, `ARVT_1_flt`, `ARVT_3_flt`, aircraft type, wake category, operator, callsign) | Planned and actual NM off-block times and their deltas; the NM join window (target within LOBT ± 3,606 s, findings 10.3); aircraft identity | **yes**: O (planned times, aircraft), A (`AOBT_3_flt`), C (join window, join-failure level), T. Blocks `nm_deltas` (C+A+T+O), `nm_cross_deltas` (C+A+O), `nm_hour_context` (C+A+T), `dep_live_state` (A+T), the join-window clip (C+T) | **yes**: IOBT/LOBT as the H1 anchor and the planned delay (O*: the last filed value assumed known 15 min ahead), the other departures' filed off-blocks in the hour after the H1 anchor (F1, the H1 deliverable since 13 Oct, findings 10.37; O*), aircraft identity (O), `AOBT_3_flt` of flights already departed (O); never the flight's own NM actual times; the 3 h and 24 h models read no IOBT (schedule clock). Of the two tier F files beside the deliverable, `h1f5` also reads NM's readings of the other flights up to 45 min after the H1 anchor (the `AOBT_3_flt` of those airborne by then, and the departures planned off by then but not yet airborne; F, findings 10.37); `h1f` reads none | Part of the competition files; none stated | None stated | inside the competition files | as above |
| **METAR**, Iowa Environmental Mesonet (IEM) ASOS/AWOS global METAR archive, Iowa State University | Weather in force at takeoff and in look-back windows: wind on the runway, visibility, ceiling, present weather, de-icing conditions | **yes**: T (`wx_now`, `wx_past`, `wx_deice_load`), and through `wx_now` in `deicing_context` (A+P+T) | **yes**: every model, read 5 min after observation at the anchor (O) | **No named licence.** IEM states: "The materials found on this website are in the public domain and may be used freely by anyone for any lawful purpose." The underlying reports come from national meteorological services. IEM names its feeds as Unidata IDD, NCEI ISD and MADIS. | Requested, not required: "Attributing the Iowa Environmental Mesonet of Iowa State University would be appreciated." | `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py`, one request per station and month (`prc/weather/download.py`) → `data/additional_data/weather/raw/` → `metar.parquet` | `python -m prc fetch-weather --only metar` |
| **TAF**, Ogimet (G. Ballester Valor) | Forecast weather: the TAF a planner holds at the moment (`op_h1_wx`, `op_hl3_wx`, `op_hl24_wx`); in the challenge model the blocks `taf_now`, `taf_planning`, not in the submission. | **no** (`taf_now`, `taf_planning`: T, not in base; `docs/feature_table.md`) | **yes**: every model (`op_h1_wx`, `op_hl3_wx`, `op_hl24_wx`; O). The licence caveat applies. At H1 the model does not need it: `op_h1_r4_nodoy_notaf` scores inside the seed range (findings 10.35). In operations the TAF comes from the met service | **No open licence.** Ogimet's licence page allows reuse of its text tables with a link to Ogimet, and says the coded reports are "copyright del organismo o institución de cada país" (copyright of each country's agency), to be used with regard to WMO Resolution 40. A TAF is a coded report. | A link to Ogimet for its text tables. For coded reports, the issuing national service's rights apply. | `https://www.ogimet.com/cgi-bin/gettafor`, per station and month, about 20 s between requests (quota-limited) → `data/additional_data/weather/raw/` → `taf_issues.parquet`, `taf_segments.parquet` | `python -m prc fetch-weather --only taf` |
| **OpenStreetMap** via the Overpass API | Stand positions, gates, the taxiway/taxilane/runway way graph; stand matching, apron zones, stand-to-threshold distances and routes; in the ADS-B stage, stand positions for the own-movement features | **yes**: O (`stand_geometry`, `taxi_distance`, `taxi_route`); M (the ADS-B stage's stand-relative features, e.g. `adsb_first_dist_stand`) | **yes**: stand position (O), distance and route to the runway (O*: the planned runway) | **Open Data Commons Open Database License (ODbL) 1.0** | Credit "OpenStreetMap and its contributors" with the attribution notice ("© OpenStreetMap contributors"), and state that the data is available under the ODbL (link to <https://www.openstreetmap.org/copyright>) | Overpass endpoints `https://overpass-api.de/api/interpreter`, `https://overpass.kumi.systems/api/interpreter`, `https://overpass.private.coffee/api/interpreter` (rotated; `prc/geometry/download.py`), one extract per airport, 7 km box → `data/additional_data/airport_geometry/raw/osm/` | `python -m prc fetch-geometry --only osm` (needs `prepare` first) |
| **OurAirports** (`airports.csv`, `runways.csv`) | Airport reference points; runway ends with true headings (thresholds for distances and routes). The reference points also centre the ADS-B boxes and the ERA5 grid points. | **yes**: O (thresholds in `taxi_distance`, `taxi_route`; reference point in `stand_geometry`); also plumbing for the ERA5 fetch and the ADS-B boxes | **yes**: thresholds and reference points (O) | **Public domain**. The data repository is marked The Unlicense. | Not required: "We'd love you to give us credit, like we give credit to our sources, but you're not required to." | `https://davidmegginson.github.io/ourairports-data/airports.csv` and `.../runways.csv` → `data/additional_data/airport_geometry/raw/ourairports/` | `python -m prc fetch-geometry --only ourairports` |
| **ERA5 / ERA5-Land reanalysis** via the Open-Meteo historical weather API | Snow depth, snowfall over 3 h and 24 h, freezing precipitation, ground and soil temperature, overnight minimum, frost hours, read at the last full hour before takeoff | **yes**: P (`era5_surface` P+T; `deicing_context` A+P+T) | **no** (tier P) | Open-Meteo data: **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Underlying ERA5: the **Licence to Use Copernicus Products** (European Union, represented by ECMWF; use "for any purpose in so far as it is lawful"), as Open-Meteo names it; from the Copernicus Climate Data Store, ERA5 is CC BY 4.0 (section 6). Open-Meteo's terms limit the *free API* to non-commercial use. | Open-Meteo: a link "Weather data by Open-Meteo.com" to <https://open-meteo.com/>. Copernicus: "Contains modified Copernicus Climate Change Service information 2026" (for modified products; "Generated using Copernicus Climate Change Service information [Year]" otherwise). Citations Open-Meteo asks for: Hersbach, H. et al. (2023), ERA5 hourly data on single levels from 1940 to present, ECMWF, doi:10.24381/cds.adbb2d47; Muñoz Sabater, J. (2019), ERA5-Land hourly data from 2001 to present, ECMWF, doi:10.24381/CDS.E2161BAC; Zippenfenig, P. (2023), Open-Meteo.com Weather API, Zenodo, doi:10.5281/ZENODO.7970649. | `https://archive-api.open-meteo.com/v1/archive`, one request per airport, 2024-12-24 to 2026-07-31, UTC, no key, no `models` parameter (Open-Meteo's default selection) → `data/additional_data/reanalysis/raw/` → `era5_hourly.parquet` | `python -m prc fetch-era5` (needs `fetch-geometry` first) |
| **EUROCONTROL Performance Review Unit / Aviation Intelligence Unit open CSVs**: `taxi_out_additional_time`, `atc_pre_departure_delays`, `all_pre_departure_delays`, `apt_dly` (2025 and 2026) | `pru_daily`: the day's and the previous day's pre-departure and arrival ATFM delay minutes per movement. `pru_monthly`: a level check of predicted monthly means (`prc levelcheck`), never a feature. | **no**: `pru_daily` (A+P+T) is not in the base; the monthly table serves only `levelcheck` (`docs/DEVELOPER_GUIDE.md` Appendix A, `docs/feature_table.md`) | **no** as an input; the monthly mean taxi-out is the level check of the final-set files (`lab/op_deliver.py`, `prc levelcheck`) | **No open licence.** The portal's disclaimer says: "This data is published by EUROCONTROL for information purposes. It may be copied in whole or in part, provided that EUROCONTROL is mentioned as the source and it is not used for commercial purposes (i.e. for financial gain)." The relayed ruling of 17-18 Sep (section 4) treats the AIU daily airport files as open data for the challenge. | "EUROCONTROL is mentioned as the source" | `https://www.eurocontrol.int/performance/data/download/csv/<dataset>_<year>.csv` (`apt_dly` as `.csv.bz2`; `prc/pru/download.py`) → `data/additional_data/pru/raw/` → `pru_daily.parquet`, `pru_monthly.parquet` | `python -m prc fetch-pru` |
| **adsb.lol globe history**, `prod` release | The ADS-B correction stage: takeoff detection and the departure link, then the flight's own first ground sighting, pushback, taxi start, stops and track quality (families B and later, findings 10.6-10.21). Points inside a ±0.10° × ±0.15° box per airport at or below 5,000 ft. | **yes**: M (with T, A, C, P, O inherited from the base, as the availability table lists) | **no deliverable**: the surface picture of other aircraft before the anchor (rung R6) gained nothing (findings 10.34, 10.35). **Yes** for the two tier F files beside the deliverable (findings 10.36, 10.37): `h1f` reads three counts of the arrivals taxiing in, at 15, 30 and 45 min after the H1 anchor (F); `h1f5` the surface picture of the other aircraft after the anchor (F) and before it (O, its own-runway hold O*; R6's block). Both read the snapshot s1, built from the `prod` points only | **Open Data Commons Open Database License (ODbL) 1.0**: "This database is made available under the Open Database License: http://opendatacommons.org/licenses/odbl/1.0/." Data sent to adsb.lol's feed is waived under CC0 by the feeders. | adsb.lol names no wording. ODbL 1.0 section 4.3 requires a notice on a publicly used Produced Work; the licence's own example is "Contains information from DATABASE NAME, which is made available here under the Open Database License (ODbL)". The wording "adsb.lol contributors" in README.md is ours. | GitHub releases `https://github.com/adsblol/globe_history_<year>/releases`, tag `v<YYYY.MM.DD>-planes-readsb-prod-0`, tar split in ~2 GB parts, streamed and filtered on the fly (`prc/adsb/stream.py`) → `data/additional_data/adsb/points/` (one parquet per day). The stage trains on 350 days of 2025 (findings 10.25). | `python -m prc fetch-adsb [--days ...]` (default Jan+Jul 2025 and 2026) |
| **adsb.lol globe history**, `staging` release (a second aggregator instance with partly different feeders) | Extra points for the same days, merged into the `prod` points | **yes**, from v43 on (findings 10.11): M. Merged into the Jan+Jul 2025 and 2026 days that have a staging release (114 day files); +1,193 covered 2026 departures, −0.12 on the board | **no** | **ODbL 1.0** (same repositories and licence statement as `prod`) | as `prod` | tag `v<YYYY.MM.DD>-planes-readsb-staging-0` in the same repositories → `data/additional_data/adsb/points_staging/`, then merged into `points/` (the `prod` originals are kept in `points_prod/`) | `python -m prc fetch-adsb --kind staging`, then `python lab/adsb_merge_staging.py` |
| **OPDI**, EUROCONTROL Open Performance Data Initiative (flight events built from OpenSky Network data) | Nothing. It was scouted as a second own-movement source. | **no** | **no** | No data licence named on <https://www.opdi.aero/data.html>. The about page defines open data in general terms ("Open data can be freely used, reused, and distributed provided that the data source is attributed") but does not grant a licence for the OPDI tables. | n/a | `https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_events/flight_events_<YYYYMMDD>_<YYYYMMDD>.parquet` | none (not implemented) |

## 2. Where each licence was verified (7 October 2026)

- **Competition files.** <https://prc-data-challenge-2026.netlify.app/data.html>: the page describes access through
  OpenSky MinIO buckets and the three file groups. It states no licence or terms of use.
- **IEM METAR.**
  - <https://mesonet.agron.iastate.edu/disclaimer.php>, section "Usage of IEM Products": "The materials found on
    this website are in the public domain and may be used freely by anyone for any lawful purpose. Attributing the
    Iowa Environmental Mesonet of Iowa State University would be appreciated."
  - The download page (<https://mesonet.agron.iastate.edu/request/download.phtml>) and the dataset pages
    (<https://mesonet.agron.iastate.edu/info/datasets/>, <https://mesonet.agron.iastate.edu/info/datasets/metar.html>)
    name the feeds (Unidata IDD, NCEI ISD, MADIS; the METAR dataset page adds NCEI GHCNh, spelled "GHCHh" there and
    linked to <https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly>) and state no licence
    of their own.
  - What we can confirm is therefore IEM's public-domain statement for the materials on its site, not a named open
    licence. The reports themselves originate with national meteorological services.
- **Ogimet TAF.** <https://www.ogimet.com/licencia.phtml> (Spanish, "© Copyright G. Ballester Valor 2005-2026"):
  - the graphical maps are free to use if "(c) Ogimet ..." stays visible and the origin is linked;
  - the text tables are free to use "siempre que se incluya el enlace al sitio de origen Ogimet";
  - "Los partes meteorológicos codificados son copyright del organismo o institución de cada país. Para utilizarlos,
    hay que tener en cuenta la Resolución 40 de la OMM."

  The front page adds "Por favor, no abuse solicitando grandes cantidades de información". No open licence can be
  confirmed for TAF text obtained from Ogimet. The submitted pipeline does not use it.
- **OpenStreetMap.** <https://www.openstreetmap.org/copyright>:
  - "licensed under the Open Data Commons Open Database License (ODbL) by the OpenStreetMap Foundation (OSMF)";
  - "You are free to copy, distribute, transmit and adapt our data, as long as you credit OpenStreetMap and its
    contributors. If you alter or build upon our data, you may distribute the result only under the same license.";
  - "Provide credit to OpenStreetMap by displaying our attribution notice. Make clear that the data is available
    under the Open Database License."
- **OurAirports.**
  - <https://ourairports.com/data/>: "All data is released to the Public Domain, and comes with no guarantee of
    accuracy or fitness for use."
  - The GitHub data repository `davidmegginson/ourairports-data`, which `fetch-geometry` downloads from, carries the
    licence "The Unlicense" (GitHub API, same day).
- **Open-Meteo and ERA5.**
  - <https://open-meteo.com/en/licence>: data under "Attribution 4.0 International (CC BY 4.0)", with the link text
    "Weather data by Open-Meteo.com". It names the Copernicus Climate Change Service (C3S) licence for the
    reanalysis.
  - <https://open-meteo.com/en/terms>: "You may only use the free API services for non-commercial purposes."
  - <https://open-meteo.com/en/docs/historical-weather-api>: the citations given in the table and "Generated using
    Copernicus Climate Change Service information 2022."
  - <https://apps.ecmwf.int/datasets/licences/copernicus/> (Licence to Use Copernicus Products): use "for any
    purpose in so far as it is lawful". The notices are "Generated using Copernicus Climate Change Service
    information [Year]" and, for modified products, "Contains modified Copernicus Climate Change Service information
    [Year]".
  - `era5_hourly.parquet` is normalised and resampled from the API answer, so the second notice is the fitting one.
- **EUROCONTROL PRU / AIU CSVs.** <https://www.ansperformance.eu/about/disclaimer/>:
  - "This data is published by EUROCONTROL for information purposes. It may be copied in whole or in part, provided
    that EUROCONTROL is mentioned as the source and it is not used for commercial purposes (i.e. for financial
    gain)."
  - "The information in this document may not be modified without prior written permission from EUROCONTROL."

  The data catalogue page <https://www.ansperformance.eu/data/> states no licence of its own. The download host was
  checked the same day: `https://www.eurocontrol.int/performance/data/download/csv/taxi_out_additional_time_2025.csv`
  answers 200 `text/csv`.
- **adsb.lol.**
  - The README of <https://github.com/adsblol/globe_history_2025> and of `globe_history_2026`: "This database is
    made available under the Open Database License: http://opendatacommons.org/licenses/odbl/1.0/."
  - GitHub API licence field for both repositories: "Open Data Commons Open Database License v1.0".
  - On prod and staging, the same README says: "Download both files from the prod or staging release for a day
    you're interested in. (if prod or staging for that day is significantly smaller, ignore it)" and "Prod is used,
    unless it is down then staging is used."
  - <https://www.adsb.lol/docs/open-data/historical/>: "License: ODbL 1.0".
  - <https://www.adsb.lol/privacy-license/>: feeders "waive all copyright and related or neighboring rights to the
    data you are sharing, under the CC0 license". This covers the incoming feed, not the published database.
- **ODbL 1.0 obligations** (<https://opendatacommons.org/licenses/odbl/1-0/>), which apply to OSM and adsb.lol:
  - Section 4.3: a publicly used Produced Work (our predictions, figures and tables) needs a notice that content
    was obtained from the database and is available under the licence.
  - Section 4.4: a Derivative Database that is publicly used must be under the ODbL. By 4.4 c, it counts as
    publicly used when a Produced Work made from it is publicly used.
  - Section 4.6: the user must then offer either the whole Derivative Database or "a file containing all of the
    alterations made to the Database or the method of making the alterations to the Database (such as an
    algorithm)", free of charge over the internet.
  - The derived tables here (`stands.parquet`, `stand_runway_routes.parquet`, the ADS-B point and stage tables)
    are not published; the code that fetches and builds them is. Our reading is that this public code serves as the
    4.6 b "method". This is not legal advice.
- **OPDI.**
  - <https://www.opdi.aero/data.html>: "The FLIGHT, EVENT and MEASUREMENT tables are published under the Open
    Performance Data Initiative (OPDI) in collaboration with the OpenSky Network (OSN). The TRAJECTORY data comes
    from OSN's State Vector data."
  - <https://www.opdi.aero/about.html> has only the general definition of open data quoted in the table.
  - Another team's README calls OPDI "Open (CC-BY 4.0, confirmed by organiser)" (kind-mango, README.md). We have
    not seen that licence on an OPDI page.

## 3. What is not used, and why

- **OPDI.** The organisers listed OPDI as allowed on 9 September (section 4, relayed). We left it out because of
  coverage, not permission. This corrects findings 10.11, which called it excluded as OpenSky-derived (findings
  10.25). In the sampled row groups of the 11-19 January 2026 file (three row groups, about 40 MB of 271.8 MB, read
  with HTTP range requests), OPDI holds stand exits (`exit-parking_position`) only at EDDF (4,399) and LSZH (9,611);
  the other eight airports have 0-7 (LTFM 0, LFPG 1, LIRF 0). The three airports with the fewest are also where
  adsb.lol links the fewest 2026 departures (LTFM 0 %, LIRF 36 %, LFPG 40 %; section 5).
- **OpenSky Network state vectors and Trino.** Not used. Per the relay, Trino access is not allowed (11 Sep), and
  the adsb.lol ruling does not extend to OSN state vectors (17-18 Sep). Our OpenSky account is not used
  (findings 10.6).
- **Other surface sources** were scouted and rejected as proprietary, paid, evaluation-only or without an open
  archive: ADS-B Exchange samples, adsb.fi, airplanes.live, airline and airport APIs, and the adsb.lol ACARS/VDL2
  archive (findings 10.11).


## 5. Independent evidence on ground reception: the PRU OpenSky airport-coverage index

EUROCONTROL's Performance Review Unit publishes an OpenSky airport-coverage ranking
(<https://euctrl-pru.github.io/opensky-airport-coverage/>, read 7 Oct 2026; output files produced 28 Aug 2026 per
its about page). Its inputs are OpenSky state vectors, OPDI flights, APDF airport records and NM tables. The sample
is three days, 5-7 June 2026 (with the same days of 2024 and 2025 on the aerodrome pages).

The site's metrics page defines the coverage index as "the share of movements seen at all, multiplied by the share
of a typical ground movement received (the average of the taxi-out and taxi-in medians)". Its bands are:

| band | coverage index |
|---|---|
| Excellent | ≥ 0.90 |
| Good | 0.60 to < 0.90 |
| Partial | 0.30 to < 0.60 |
| Poor | 0.05 to < 0.30 |
| None | < 0.05 |

The ten challenge airports in its measured 2026 table
(`downloads/opensky-airport-coverage-measured-2026.csv`, downloaded 7 Oct 2026) are below. Next to them is our own
adsb.lol link rate, the share of the January and July 2026 departures linked to an ADS-B takeoff:

| airport | PRU OpenSky coverage index, June 2026 | band | adsb.lol linked, Jan+Jul 2026 |
|---|---|---|---|
| LSZH | 0.996 | Excellent | 96 % |
| LEBL | 0.892 | Good | 97 % |
| EDDF | 0.803 | Good | 88 % |
| EHAM | 0.640 | Good | 98 % |
| LEMD | 0.545 | Partial | 89 % |
| EDDM | 0.498 | Partial | 97 % |
| EGLL | 0.374 | Partial | 95 % |
| LFPG | 0.305 | Partial | 40 % |
| LIRF | 0.287 | Poor | 36 % |
| LTFM | 0.146 | Poor | 0 % |

The two measures differ in network, period and definition:

- **Network:** OpenSky versus adsb.lol.
- **Period:** three June days versus January and July.
- **Definition:** ground continuity versus a linked takeoff.

Even so, both put Istanbul, Rome and Paris at the bottom. This is organiser-published evidence, independent of our
pipeline, that the uncovered rows reflect a limit of the open receiver networks, not of our crawl. 

The networks also differ by airport. Munich reads 0.498 on OpenSky, but adsb.lol links 97 % of its 2026 departures.
This is why the adsb.lol stage reaches airports that OPDI (built from OpenSky) does not.

## 6. Open points

- **ERA5 through Open-Meteo.** Open-Meteo's terms limit the free API to "non-commercial purposes"
  (<https://open-meteo.com/en/terms>), and the challenge carries a prize. ERA5 itself is distributed by Copernicus
  under CC BY 4.0 and can be fetched directly from the Copernicus Climate Data Store
  (<https://cds.climate.copernicus.eu/>); `prc fetch-era5` reads it through Open-Meteo.
- **The competition files carry no stated licence.** This is the organisers' data and does not affect eligibility.
- **Discrepancies in other files** (not changed here):
  - The `prc/pru/__init__.py` docstring and manifest string give `https://ansperformance.eu/csv/` as the PRU
    download URL; `prc/pru/download.py` downloads from `https://www.eurocontrol.int/performance/data/download/csv/`
    (README corrected 7 Oct).
  - Findings 10.11 describes OPDI as excluded under the OpenSky ruling. Findings 10.25 corrects this to "left out
    for coverage".
