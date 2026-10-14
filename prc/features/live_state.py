"""The departures around this one, as NM saw them: how long taxis are taking right now.

The literature's "taxi times of recent departures" cannot be read here, since every departure's block time is
blanked; NM's off-block time is not, so each other flight's ``takeoff - AOBT_3`` against its stand×runway norm
(`_proxylib`) stands in for it. Averaged over the last 30 and 60 minutes and over +-15 minutes of takeoffs, per
airport, per runway and per pier, with the flight itself left out.

Findings 9.6: nothing on the Jan+Jul holdout (exp037 already carries its summer share through `nm_hour_context`,
`taxi_window_counts` and the arrivals), but +2.7 s on the Feb+Dec residual beyond exp037's blocks and +2.8 s in
its de-icing hours: de-icing after pushback shows in the other flights' proxies (across piers at 0.92), and
January 2026 put more of it into the taxi than any 2025 month. The windows' counts and the other flights'
off-block delay carried nothing and are left out. Windows reaching past the takeoff use the other flights'
recorded times, legal here as for `arrival_conditions` because the test months are historical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc.features import _proxylib
from prc.features.congestion import window_count, window_reduce
from prc.features.core import epoch_seconds
from prc.features.registry import FeatureContext, feature
from prc.features.stand_runway import stand_prefix_key

HIGH_ANOMALY_S = 300.0
"""A proxy this far above its stand×runway norm counts as a slow taxi in the ``share5`` columns."""
WINDOWS = (("b30", 1800, 0), ("b60", 3600, 0), ("pm15", 900, 900))
LEVELS = ("apt", "rwy", "pier")

LIVE_STATE = tuple(f"live_{lvl}_{win}_{stat}" for lvl in LEVELS for win, _, _ in WINDOWS for stat in ("mean", "share5"))


@feature(
    group="live_state",
    outputs=LIVE_STATE,
    depends_on=("core_keys",),
    doc="The other departures' NM taxi-out anomaly (mean, share above +5 min) in the last 30/60 min and +-15 min, per airport, runway and pier, self excluded.",
)
def dep_live_state(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    min_support = int(ctx.params.get("min_support", _proxylib.MIN_SUPPORT))
    dep = mvt.loc[ctx.dep_index]
    anomaly = _proxylib.proxy_anomaly(dep, min_support).clip(*_proxylib.ANOMALY_CLIP)
    a = anomaly.to_numpy(dtype="float64")
    valid = np.isfinite(a)
    high = valid & (a > HIGH_ANOMALY_S)
    self_value = np.where(valid, a, 0.0)

    t = pd.Series(np.asarray(epoch_seconds(dep["MVT_TIME_UTC_mvt"]), dtype="float64"), index=dep.index)
    apt = dep["apt"].astype("string")
    groups = {
        "apt": apt,
        "rwy": apt + "|" + dep["RUNWAY_mvt"].astype("string").fillna("__NA__"),
        "pier": stand_prefix_key(dep["apt"], dep["STAND_mvt"]),
    }
    out = {}
    for lvl in LEVELS:
        g = groups[lvl]
        for win, back, fwd in WINDOWS:
            total = np.nan_to_num(window_reduce(t[valid], g[valid], anomaly[valid], t, g, back=back, fwd=fwd, how="sum"))
            n = window_count(t[valid], g[valid], t, g, back=back, fwd=fwd) - valid.astype(np.int64)
            n_high = window_count(t[high], g[high], t, g, back=back, fwd=fwd) - high.astype(np.int64)
            others = np.maximum(n, 1)
            out[f"live_{lvl}_{win}_mean"] = np.where(n > 0, (total - self_value) / others, np.nan)
            out[f"live_{lvl}_{win}_share5"] = np.where(n > 0, n_high / others, np.nan)
    return pd.DataFrame(out, index=ctx.dep_index).astype("float32")
