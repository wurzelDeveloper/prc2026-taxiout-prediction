"""NM's taxi-out reading of every departure, and how far it sits from what is normal for its stand and runway.

``takeoff - AOBT_3_flt`` is NM's measurement of the taxi-out (the airport's own off-block time, which defines the
target, is blanked; NM's is not). Its anomaly against the proxy's own (airport, stand, runway) median needs no
target, so it is computed over every departure in the frame, 2026 included, like the arrivals' taxi-in anomaly in
`_arrlib`. The other flights' anomalies around a takeoff are a live reading of the airport: slow taxis, a queue,
de-icing after pushback (findings 9.6: across 55 piers the proxy's de-icing uplift tracks the taxi's at 0.92).

Shared by `live_state` and `deice_location`. Not auto-imported (underscore module), but hashed into every feature
module that imports it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc.features.core import epoch_seconds

PROXY_MAX_S = 7_200.0
"""A proxy outside [0, 2 h] is a mis-join or a stale NM record, not a reading of the taxi; it is left out."""
MIN_SUPPORT = 20
"""Proxies a (stand, runway) pairing needs before its own median is trusted as the baseline."""
ANOMALY_CLIP = (-900.0, 1800.0)
"""Clip for anomalies that are averaged over other flights, so one stale record cannot move a window."""
WIDE_CLIP = (-900.0, 3600.0)
"""Clip for anomalies whose tail is the point (de-icing adds up to half an hour)."""


def proxy_seconds(dep: pd.DataFrame) -> pd.Series:
    """``MVT_TIME_UTC_mvt - AOBT_3_flt`` in seconds; NaN where NM has no off-block time or the gap is implausible."""
    t = np.asarray(epoch_seconds(dep["MVT_TIME_UTC_mvt"]), dtype="float64")
    aobt = np.asarray(epoch_seconds(dep["AOBT_3_flt"]), dtype="float64")
    with np.errstate(invalid="ignore"):
        p = t - aobt
        p[~((p >= 0) & (p <= PROXY_MAX_S))] = np.nan
    return pd.Series(p, index=dep.index, dtype="float64")


def proxy_anomaly(dep: pd.DataFrame, min_support: int = MIN_SUPPORT) -> pd.Series:
    """Each departure's proxy minus the median proxy of its (airport, stand, runway), unclipped.

    Pairings with fewer than ``min_support`` proxies fall back to (airport, stand), then (airport, runway), then the
    airport. Computed over the departures given, which for a feature block is every departure of both splits.
    """
    proxy = proxy_seconds(dep)
    apt = dep["apt"].astype("string")
    stand = dep["STAND_mvt"].astype("string").fillna("__NA__")
    rwy = dep["RUNWAY_mvt"].astype("string").fillna("__NA__")
    valid = proxy.notna().to_numpy()
    base = pd.Series(np.nan, index=dep.index, dtype="float64")
    for key in (apt + "|" + stand + "|" + rwy, apt + "|" + stand, apt + "|" + rwy, apt):
        grouped = proxy[valid].groupby(key[valid].to_numpy())
        median = grouped.median()[grouped.size() >= min_support]
        need = base.isna()
        base[need] = key[need].map(median).astype("float64")
    return (proxy - base).rename("proxy_anomaly")
