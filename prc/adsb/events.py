"""From adsb.lol points at an airport to per-departure events.

For every departure of a day: find the aircraft that took off from its airport near the recorded takeoff time
with the same callsign, then read what ADS-B saw of that aircraft on the ground before takeoff. Two families
of quantities, kept apart on purpose (the user's question: what is ADS-B useful for?):

* **own movement** (family B): the flight's first ground sighting, its first movement off the parked position
  (pushback), its taxi start, the time it spent stationary after taxi start, its takeoff as ADS-B saw it. These
  observe the off-block itself.
* **before off-block** (family A, `surface_context`): what could be known at the flight's *planned* off-block
  time (NM's LOBT) without watching the flight: aircraft moving on the surface, aircraft waiting near the
  runways, and when this aircraft arrived (its inbound landing), if ADS-B saw it.

**Operational availability** (tiers of `docs/feature_table.md`; the real-world use case is a prediction made
before the aircraft moves):

* own movement (`own_movement`, `stand_movement`): **tier M, challenge-only.** They observe the flight's own pushback
  and taxi, i.e. information that exists only after its off-block. They answer the challenge's question (when did the
  recorded off-block happen, given the takeoff) and would not exist at the time of an operational prediction; in
  operations the same observation would serve to detect the actual off-block in real time, not to predict it.
* before off-block (`surface_context`): **tier O** when computed at the planned off-block time it is anchored on (NM's
  LOBT): it counts other aircraft's positions up to that time only. The flight's own inbound landing is O as well
  (it precedes the off-block), but finding the aircraft through `link` uses its takeoff, which an operational system
  would replace by the flight plan's ICAO 24-bit address.
* the link (`takeoffs`, `link`): uses the recorded takeoff time (challenge construction).

Takeoff detection needs no ground flag: an aircraft's takeoff is the first point of a climb (altitude rising
through the airport's lowest seen altitude + 300 ft at more than 100 kt) after at least one slow point
(< 60 kt) at the airport. Ground points are ``on_ground`` or slower than 40 kt below the takeoff altitude.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MATCH_TOL_S = 600.0
TIME_TOL_S = 45.0
"""Time-only fallback link: the one unlinked ADS-B takeoff at the airport within this of the recorded takeoff."""
"""|ADS-B takeoff - recorded takeoff| allowed for a link."""
LOOKBACK_S = 4 * 3600.0
"""How far before takeoff the ground segment is searched."""
MOVE_M = 25.0
"""Displacement from the parked position that counts as the first movement."""
TAXI_KT = 8.0
PARK_RADIUS_M = 15.0
PARK_MIN_S = 120.0


def _norm_callsign(s: pd.Series) -> pd.Series:
    return s.astype("string").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)


def _metres(lat0, lon0, lat, lon):
    k = 111_320.0
    return np.hypot((lat - lat0) * k, (lon - lon0) * k * np.cos(np.radians(lat0)))


def takeoffs(points: pd.DataFrame) -> pd.DataFrame:
    """One row per detected takeoff: icao, apt, t_takeoff, callsign at takeoff."""
    out = []
    for (icao, apt), g in points.groupby(["icao", "apt"], observed=True, sort=False):
        g = g.sort_values("t")
        t = g["t"].to_numpy()
        alt = g["alt_ft"].to_numpy(dtype="float64")
        gs = g["gs_kt"].to_numpy(dtype="float64")
        ground = g["on_ground"].to_numpy(bool)
        air_alt = np.where(ground, np.nan, alt)
        if np.all(np.isnan(air_alt)):
            continue
        floor = np.nanmin(np.where(np.isnan(air_alt), np.inf, air_alt))
        slow = ground | (gs < 60)
        climb = (~ground) & (alt > floor + 300) & (gs > 100)
        i = 0
        n = len(t)
        while i < n:
            j = np.argmax(slow[i:]) + i if slow[i:].any() else n
            if j >= n:
                break
            k_rel = np.argmax(climb[j:]) if climb[j:].any() else -1
            if k_rel < 0:
                break
            k = j + k_rel
            if t[k] - t[j] > 3 * 3600:  # slow point too long before: not the same departure
                i = j + 1
                continue
            # the takeoff: last slow point before the climb, plus the roll (first point above 100 kt after it)
            last_slow = j + np.nonzero(slow[j:k])[0].max() if slow[j:k].any() else j
            roll = last_slow + 1 + (np.argmax(gs[last_slow + 1:k + 1] > 100) if (gs[last_slow + 1:k + 1] > 100).any() else 0)
            cs = g["callsign"].iloc[max(0, k - 5):k + 5].dropna()
            out.append((icao, apt, float(t[min(roll, k)]), cs.iloc[-1] if len(cs) else None, int(last_slow), int(k)))
            # skip past this departure's airborne segment
            nxt = np.nonzero(slow[k:])[0]
            i = k + (nxt[0] if len(nxt) else n)
    return pd.DataFrame(out, columns=["icao", "apt", "t_takeoff", "callsign", "i_last_slow", "i_climb"])


def link(departures: pd.DataFrame, tko: pd.DataFrame) -> pd.DataFrame:
    """Map departures (index mvt_id; apt, callsign, t_mvt) to detected takeoffs by callsign and time."""
    d = departures.reset_index()[["mvt_id", "apt", "callsign", "t_mvt"]].copy()
    d["cs"] = _norm_callsign(d["callsign"])
    k = tko.copy()
    k["cs"] = _norm_callsign(k["callsign"])
    k["apt"] = k["apt"].astype(str)
    d["apt"] = d["apt"].astype(str)
    m = d.merge(k, on=["apt", "cs"], how="inner")
    m["dt"] = (m["t_takeoff"] - m["t_mvt"]).abs()
    m = m[m["dt"] <= MATCH_TOL_S].sort_values("dt").drop_duplicates("mvt_id")
    out = m.set_index("mvt_id")[["icao", "t_takeoff", "dt"]]
    out["by_time"] = 0
    # second pass, by time alone: a departure without a callsign link takes the one ADS-B takeoff at its airport
    # within TIME_TOL_S whose aircraft has no link yet (several candidates: no link)
    used = set(zip(out["icao"], out["t_takeoff"].round()))
    rest = d[~d["mvt_id"].isin(out.index)]
    extra = []
    for apt, g in rest.groupby("apt"):
        t = k[k["apt"] == apt]
        t = t[[(ic, round(tt)) not in used for ic, tt in zip(t["icao"], t["t_takeoff"])]]
        if t.empty:
            continue
        tt = t["t_takeoff"].to_numpy()
        for _, r in g.iterrows():
            c = np.nonzero(np.abs(tt - r["t_mvt"]) <= TIME_TOL_S)[0]
            if len(c) == 1:
                extra.append((r["mvt_id"], t["icao"].iloc[c[0]], float(tt[c[0]]), float(abs(tt[c[0]] - r["t_mvt"])), 1))
    if extra:
        ex = pd.DataFrame(extra, columns=["mvt_id", "icao", "t_takeoff", "dt", "by_time"])
        ex = ex[~ex.duplicated(["icao", "t_takeoff"], keep=False)].set_index("mvt_id")
        out = pd.concat([out, ex])
    return out


def own_movement(points: pd.DataFrame, links: pd.DataFrame, t_mvt: pd.Series, apt: pd.Series) -> pd.DataFrame:
    """Family B for each linked departure: the aircraft's ground segment before its takeoff.

    Tier M (challenge-only): observes the flight's own movement after its off-block."""
    by = {key: g.sort_values("t") for key, g in points.groupby(["icao", "apt"], observed=True, sort=False)}
    rows = {}
    for mvt_id, r in links.iterrows():
        g = by.get((r["icao"], apt.loc[mvt_id]))
        if g is None:
            continue
        tk = r["t_takeoff"]
        seg = g[(g["t"] < tk) & (g["t"] >= tk - LOOKBACK_S)]
        slowmask = seg["on_ground"].to_numpy(bool) | (seg["gs_kt"].to_numpy(dtype="float64") < 40)
        seg = seg[slowmask]
        if seg.empty:
            rows[mvt_id] = {"adsb_n_ground": 0}
            continue
        # parked until: the end of the last stationary period (>= PARK_MIN_S within PARK_RADIUS_M) before takeoff,
        # searched over the whole ground history in the lookback (the arrival's taxi-in and the gate time included)
        t_all = seg["t"].to_numpy()
        la, lo_ = seg["lat"].to_numpy(), seg["lon"].to_numpy()
        parked_until, parked_dur = np.nan, np.nan
        i = 0
        while i < len(t_all):
            j = i
            while j + 1 < len(t_all) and _metres(la[i], lo_[i], la[j + 1], lo_[j + 1]) <= PARK_RADIUS_M:
                j += 1
            if t_all[j] - t_all[i] >= PARK_MIN_S:
                parked_until, parked_dur = float(t_all[j]), float(t_all[j] - t_all[i])
            i = j + 1
        # the segment after the aircraft's last airborne point before takeoff (i.e. after its inbound landing)
        t = seg["t"].to_numpy()
        gaps = np.diff(t)
        cut = np.nonzero(gaps > 1800)[0]
        if len(cut):
            seg = seg.iloc[cut[-1] + 1:]
            t = seg["t"].to_numpy()
        lat = seg["lat"].to_numpy()
        lon = seg["lon"].to_numpy()
        gs = seg["gs_kt"].to_numpy(dtype="float64")
        disp = _metres(lat[0], lon[0], lat, lon)
        moved = np.nonzero(disp > MOVE_M)[0]
        taxi = np.nonzero(gs >= TAXI_KT)[0]
        first_move = float(t[moved[0]]) if len(moved) else np.nan
        taxi_start = float(t[taxi[0]]) if len(taxi) else np.nan
        # stationary time after taxi start (holds on the way out), from gaps between slow points
        hold = np.nan
        if len(taxi):
            after = slice(taxi[0], None)
            dt = np.diff(t[after], append=tk)
            hold = float(np.sum(dt[gs[after] < 1.0]))
        step = _metres(lat[:-1], lon[:-1], lat[1:], lon[1:]) if len(lat) > 1 else np.zeros(0)
        rows[mvt_id] = {
            "adsb_first_lat": float(lat[0]),
            "adsb_first_lon": float(lon[0]),
            "adsb_path_m": float(np.sum(step[step < 500])),
            "adsb_n_ground": int(len(seg)),
            "adsb_parked_until": float(tk - parked_until) if np.isfinite(parked_until) else np.nan,
            "adsb_parked_dur": parked_dur,
            "adsb_first_seen": float(tk - t[0]),
            "adsb_first_move": float(tk - first_move) if np.isfinite(first_move) else np.nan,
            "adsb_taxi_start": float(tk - taxi_start) if np.isfinite(taxi_start) else np.nan,
            "adsb_hold_s": hold,
            "adsb_takeoff_offset": float(tk - t_mvt.loc[mvt_id]),
            "adsb_max_gap": float(np.max(np.diff(t, append=tk))),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


STOP_MIN_S = 60.0
"""Shortest stationary period (within PARK_RADIUS_M) that `stand_movement` records."""
AT_STAND_M = 60.0
"""A stationary period anchored within this of the flight's stand is a stop at the stand."""


def stand_movement(points: pd.DataFrame, links: pd.DataFrame, apt: pd.Series, stand_lat: pd.Series,
                   stand_lon: pd.Series) -> pd.DataFrame:
    """Family B, second set: the ground history read against the flight's own stand and its stops.

    Tier M (challenge-only): observes the flight's own movement after its off-block.

    The first set (`own_movement`) times the first movement from the first sighting, which a tow, an earlier
    taxi-in or position noise at the stand can place hours before the off-block. Here, over the ground points in
    the lookback (seconds before the ADS-B takeoff, like the first set):

    * ``adsb_left_stand_50`` / ``_100``: the last ground point within 50 / 100 m of the stand's coordinates;
      ``adsb_n_near_stand``: how many ground points lie within 100 m of it;
    * ``adsb_stand_stop_end``: the end of the last stationary period anchored within `AT_STAND_M` of the stand;
    * ``adsb_stop_end_300`` / ``_600`` / ``_1200``: the end of the last stationary period of at least that long,
      anywhere (a long stop away from the stand is a remote stand, a de-icing pad or a hold);
    * after the last stop at the stand (or, without one, after the first sighting): the number and the total
      length of the stops (``adsb_n_stops_after``, ``adsb_stop_s_after``) and the longest (``adsb_max_stop_after``);
    * ``adsb_tracked_from``: the start of the final stretch with no gap above 120 s that ends at the takeoff.
    """
    by = {key: g.sort_values("t") for key, g in points.groupby(["icao", "apt"], observed=True, sort=False)}
    rows = {}
    for mvt_id, r in links.iterrows():
        g = by.get((r["icao"], apt.loc[mvt_id]))
        if g is None:
            continue
        tk = r["t_takeoff"]
        seg = g[(g["t"] < tk) & (g["t"] >= tk - LOOKBACK_S)]
        seg = seg[seg["on_ground"].to_numpy(bool) | (seg["gs_kt"].to_numpy(dtype="float64") < 40)]
        if seg.empty:
            continue
        t = seg["t"].to_numpy()
        la, lo = seg["lat"].to_numpy(), seg["lon"].to_numpy()
        out = {}
        # stationary periods: maximal runs of points within PARK_RADIUS_M of the run's first point
        stops = []
        i = 0
        while i < len(t):
            j = i
            while j + 1 < len(t) and _metres(la[i], lo[i], la[j + 1], lo[j + 1]) <= PARK_RADIUS_M:
                j += 1
            if t[j] - t[i] >= STOP_MIN_S:
                stops.append((t[i], t[j], la[i], lo[i]))
            i = j + 1
        for s in (300, 600, 1200):
            ends = [e for b, e, _, _ in stops if e - b >= s]
            out[f"adsb_stop_end_{s}"] = tk - max(ends) if ends else np.nan
        slat, slon = stand_lat.get(mvt_id, np.nan), stand_lon.get(mvt_id, np.nan)
        since = t[0]
        if np.isfinite(slat) and np.isfinite(slon):
            d = _metres(slat, slon, la, lo)
            for rad in (50, 100):
                near = np.nonzero(d <= rad)[0]
                out[f"adsb_left_stand_{rad}"] = tk - t[near[-1]] if len(near) else np.nan
            out["adsb_n_near_stand"] = int(np.sum(d <= 100))
            at_stand = [e for b, e, a, o in stops if _metres(slat, slon, a, o) <= AT_STAND_M]
            if at_stand:
                out["adsb_stand_stop_end"] = tk - max(at_stand)
                since = max(at_stand)
        after = [(b, e) for b, e, _, _ in stops if b >= since]
        out["adsb_n_stops_after"] = len(after)
        out["adsb_stop_s_after"] = float(sum(e - b for b, e in after))
        out["adsb_max_stop_after"] = float(max((e - b for b, e in after), default=0.0))
        gaps = np.nonzero(np.diff(np.append(t, tk)) > 120)[0]
        out["adsb_tracked_from"] = tk - (t[gaps[-1] + 1] if len(gaps) and gaps[-1] + 1 < len(t) else (t[0] if not len(gaps) else tk))
        rows[mvt_id] = out
    return pd.DataFrame.from_dict(rows, orient="index")


def surface_context(points: pd.DataFrame, when: pd.Series, apt: pd.Series, links: pd.DataFrame) -> pd.DataFrame:
    """Family A: what ADS-B shows of the airport at each departure's planned off-block time ``when`` (epoch s).

    Counted over the other aircraft (the flight's own aircraft excluded when it is linked): aircraft taxiing
    (ground, 3-40 kt) and aircraft seen on the ground at all in the 5 minutes up to ``when``; and the flight's
    own inbound, if ADS-B saw this aircraft land at the airport before: seconds from that landing to ``when``.
    Nothing after ``when`` is used, and ``when`` does not depend on the flight's own movement.

    Tier O (operational) when computed at ``when`` = the planned off-block (LOBT): it reads other aircraft up to that
    time only. Measured value for the challenge: none over the NM-based features (findings 10.7).
    """
    ground = points[points["on_ground"] | (points["gs_kt"] < 40)]
    out = {}
    for a, rows in when.groupby(apt.astype(str)):
        g = ground[ground["apt"].astype(str) == a]
        if g.empty:
            continue
        t = g["t"].to_numpy()
        order = np.argsort(t)
        t = t[order]
        icao = g["icao"].to_numpy()[order]
        gs = g["gs_kt"].to_numpy(dtype="float64")[order]
        for mvt_id, w in rows.items():
            if not np.isfinite(w):
                continue
            lo, hi = np.searchsorted(t, w - 300.0), np.searchsorted(t, w, side="right")
            own = links["icao"].get(mvt_id) if mvt_id in links.index else None
            ic = icao[lo:hi]
            mv = gs[lo:hi]
            keep = ic != own if own is not None else np.ones(len(ic), bool)
            out[mvt_id] = {
                "adsbA_ground_ac": int(len(set(ic[keep]))),
                "adsbA_taxiing_ac": int(len(set(ic[keep & (mv >= 3) & (mv < 40)]))),
            }
    frame = pd.DataFrame.from_dict(out, orient="index")
    # the inbound: last time the linked aircraft was airborne-and-slowing into this airport before `when`
    inbound = {}
    air = points[~points["on_ground"] & (points["gs_kt"] >= 40)]
    by = {k: g["t"].to_numpy() for k, g in air.groupby(["icao", "apt"], observed=True, sort=False)}
    for mvt_id, r in links.iterrows():
        ts = by.get((r["icao"], apt.loc[mvt_id]))
        w = when.get(mvt_id)
        if ts is None or w is None or not np.isfinite(w):
            continue
        before = ts[ts < min(w, r["t_takeoff"]) - 600]
        if len(before):
            inbound[mvt_id] = float(w - before.max())
    frame["adsbA_since_inbound"] = pd.Series(inbound)
    return frame
