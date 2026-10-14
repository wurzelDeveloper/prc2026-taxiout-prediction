"""Disrupted airport-days: a report mask read off the arrivals.

The Jan+Jul 2025 holdout has no snow day; January 2026 has several (EHAM 4-5
Jan, LFPG 7 Jan, LTFM 18-19 Jan, EDDF 26 Jan). Departure block times are
blanked in 2026, but arrivals keep theirs, and on a disrupted day their taxi-in
times run minutes above what their stand/runway pairing normally takes. An
airport-day whose arrivals' mean taxi-in anomaly exceeds `DISRUPTION_THRESHOLD_S`
is "disrupted", and every run reports its RMSE on the departures of those days
(``overall_disrupted`` / ``rmse_disrupted``) next to the headline number, so
winter features are judged where they matter.

Like `prc.cleaning.day_offset_rows` this is a report mask, never a rule: it
leaves neither the fit nor the score. It imports only `prc.features._arrlib`
(the anomaly definition the `arrival_conditions` block shares), the way
`prc.weather.coverage` imports `_wxlib`.
"""

from __future__ import annotations

import pandas as pd

from prc.features import _arrlib

DISRUPTION_THRESHOLD_S = 150.0
"""Mean taxi-in anomaly (s) above which an airport-day counts as disrupted. In 2025 about thirty
airport-days cross it; EHAM on 2026-01-05 sits at +1,500 s."""
MVT_COLUMNS = _arrlib.ARR_COLUMNS
"""What `disrupted_rows` reads from the movement frame (the departure side needs ``apt`` and takeoff time)."""


def _day_key(apt: pd.Series, when: pd.Series) -> pd.Series:
    return apt.astype(str) + "|" + when.dt.floor("D").dt.strftime("%Y-%m-%d")


def disrupted_days(
    mvt: pd.DataFrame,
    threshold_s: float = DISRUPTION_THRESHOLD_S,
    *,
    min_arrivals: int = _arrlib.MIN_ARRIVALS_PER_DAY,
    min_support: int = _arrlib.MIN_SUPPORT,
) -> pd.DataFrame:
    """Airport-days above the threshold: ``apt, day, n_arr, mean_anomaly, n_dep``, worst first."""
    days = _arrlib.airport_day_anomaly(mvt, min_support)
    dep = mvt.loc[mvt["is_dep"].to_numpy(dtype=bool), ["apt", "MVT_TIME_UTC_mvt"]]
    n_dep = _day_key(dep["apt"], dep["MVT_TIME_UTC_mvt"]).value_counts()
    key = days["apt"].astype(str) + "|" + pd.to_datetime(days["day"]).dt.strftime("%Y-%m-%d")
    days["n_dep"] = key.map(n_dep).fillna(0).astype(int).to_numpy()
    hit = days.loc[(days["n_arr"] >= min_arrivals) & (days["mean_anomaly"] > threshold_s)]
    return hit.sort_values("mean_anomaly", ascending=False).reset_index(drop=True)


def disrupted_rows(
    mvt: pd.DataFrame,
    idx: pd.Index,
    threshold_s: float = DISRUPTION_THRESHOLD_S,
    *,
    min_arrivals: int = _arrlib.MIN_ARRIVALS_PER_DAY,
    min_support: int = _arrlib.MIN_SUPPORT,
) -> pd.Series:
    """True for the rows of ``idx`` (departures) whose airport-day is disrupted. Works for ranking rows too."""
    days = disrupted_days(mvt, threshold_s, min_arrivals=min_arrivals, min_support=min_support)
    hit = set((days["apt"].astype(str) + "|" + pd.to_datetime(days["day"]).dt.strftime("%Y-%m-%d")).tolist())
    rows = mvt.loc[idx, ["apt", "MVT_TIME_UTC_mvt"]]
    mask = _day_key(rows["apt"], rows["MVT_TIME_UTC_mvt"]).isin(hit)
    return pd.Series(mask.to_numpy(dtype=bool), index=idx, name="disrupted")
