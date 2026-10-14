"""Forward windows: the departures that take off right after this one, as NM saw them.

A competitor's "forward windows" (-0.7 s on their board): a departure queued behind slow ones shows in the NM
proxies of the flights that take off right after it. `live_state.dep_live_state` has a symmetric +-15 min window;
these look forward only. Residual pre-test +0.02 s on lean2 (findings 10.4): registered, not in the line.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc.features import _proxylib
from prc.features.congestion import window_count, window_reduce
from prc.features.core import epoch_seconds
from prc.features.registry import FeatureContext, feature
FOLLOWING_WINDOWS = (("f15", 900), ("f30", 1800))
FOLLOWING_LEVELS = ("apt", "rwy")
FOLLOWING = tuple(
    f"follow_{lvl}_{win}_{stat}" for lvl in FOLLOWING_LEVELS for win, _ in FOLLOWING_WINDOWS for stat in ("proxy", "anom", "n")
)


@feature(
    group="following",
    outputs=FOLLOWING,
    depends_on=("core_keys",),
    doc="The departures that take off in the next 15 and 30 min, per airport and runway: their mean NM proxy, its anomaly and their count, self excluded.",
)
def dep_following(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """What is coming after this takeoff, as NM saw it (a competitor's "forward windows", -0.7 s on their board).

    A departure queued behind slow ones shows in the NM proxies of the flights that take off right after it: they
    were in the same queue. `dep_live_state`'s +-15 min window mixes both sides; these windows look forward only,
    ``(t, t + w]``, over the other flights' recorded takeoff times (historical test months, like the arrivals).
    """
    min_support = int(ctx.params.get("min_support", _proxylib.MIN_SUPPORT))
    dep = mvt.loc[ctx.dep_index]
    proxy = _proxylib.proxy_seconds(dep)
    anomaly = _proxylib.proxy_anomaly(dep, min_support).clip(*_proxylib.ANOMALY_CLIP)
    p = proxy.to_numpy(dtype="float64")
    valid = np.isfinite(p) & np.isfinite(anomaly.to_numpy(dtype="float64"))
    t = pd.Series(np.asarray(epoch_seconds(dep["MVT_TIME_UTC_mvt"]), dtype="float64"), index=dep.index)
    apt = dep["apt"].astype("string")
    groups = {"apt": apt, "rwy": apt + "|" + dep["RUNWAY_mvt"].astype("string").fillna("__NA__")}
    # (t, t + w]: a window from t + 1 s so the flight itself and anything at the same second are left out
    t_after = t + 1.0
    out = {}
    for lvl in FOLLOWING_LEVELS:
        g = groups[lvl]
        for win, fwd in FOLLOWING_WINDOWS:
            n = window_count(t[valid], g[valid], t_after, g, back=0, fwd=fwd - 1)
            s_p = np.nan_to_num(window_reduce(t[valid], g[valid], proxy[valid], t_after, g, back=0, fwd=fwd - 1, how="sum"))
            s_a = np.nan_to_num(window_reduce(t[valid], g[valid], anomaly[valid], t_after, g, back=0, fwd=fwd - 1, how="sum"))
            others = np.maximum(n, 1)
            out[f"follow_{lvl}_{win}_proxy"] = np.where(n > 0, s_p / others, np.nan)
            out[f"follow_{lvl}_{win}_anom"] = np.where(n > 0, s_a / others, np.nan)
            out[f"follow_{lvl}_{win}_n"] = n
    frame = pd.DataFrame(out, index=ctx.dep_index)
    return frame.astype({c: ("float32" if not c.endswith("_n") else "int32") for c in frame.columns})
