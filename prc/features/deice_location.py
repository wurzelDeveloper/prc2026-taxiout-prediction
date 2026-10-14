"""Where de-icing costs taxi time: the uplift per pier and per stand in de-icing hours.

Pads are mapped in OSM at three of the six winter airports only (findings 9.4), and many airports de-ice on
dedicated stands or on the departure stand itself. The data says where it lands in the recorded taxi without
knowing where the pads are (findings 9.6): de-icing hours lengthen the taxi at every winter airport, by pier from
+39 s (EHAM pier H) to +250 s (EHAM D/E/A/C) and +300-500 s (every LFPG pier), and the pier pattern holds into
January 2026 (0.64 across piers).

Two estimates per pier and per stand, each the mean anomaly in de-icing hours minus the mean in the other hours of
the same winter months (Nov-Mar), shrunk toward the airport's uplift with `shrink` de-icing rows:

- ``deice_uplift_*`` from the fit rows' targets (taxi-out minus the fit rows' stand×runway median);
- ``deice_proxy_uplift_*`` from NM's proxy anomaly (`_proxylib`) over every departure, 2026 included, which needs
  no target. It misses EDDM, where NM's off-block moves later under de-icing (taxi +247 s, proxy +59 s).

The ``_now`` columns multiply each by the current METAR de-icing condition (`wx_now`). Findings 9.6: +1.2 s on the
Feb+Dec residual beyond exp037's blocks, +1.6 s in its de-icing hours, nothing on Jan+Jul.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc.features import _proxylib
from prc.features.registry import FeatureContext, feature
from prc.features.stand_runway import stand_prefix_key

WINTER_MONTHS = (1, 2, 3, 11, 12)
TAXI_MAX_S = 3600.0
"""Fit rows above an hour (stale records, wrong dates, genuine extremes) stay out of the uplift."""
SHRINK = 30.0

DEICE_LOCATION = tuple(
    f"{name}_{lvl}{now}" for name in ("deice_uplift", "deice_proxy_uplift") for lvl in ("pier", "stand") for now in ("", "_now")
)


def _hier_median(values: pd.Series, keys: list[pd.Series], mask: np.ndarray, min_support: int) -> pd.Series:
    out = pd.Series(np.nan, index=values.index, dtype="float64")
    for key in keys:
        grouped = values[mask].groupby(key[mask].to_numpy())
        median = grouped.median()[grouped.size() >= min_support]
        need = out.isna()
        out[need] = key[need].map(median).astype("float64")
    return out


def _uplift(values: pd.Series, key: pd.Series, apt: pd.Series, deice: np.ndarray, mask: np.ndarray, shrink: float) -> np.ndarray:
    """Per key: mean(values | de-icing) - mean(values | not), over ``mask`` rows, shrunk toward the airport's uplift."""
    m = mask & values.notna().to_numpy()
    frame = pd.DataFrame({"v": values[m].to_numpy(), "k": key[m].to_numpy(), "a": apt[m].to_numpy(), "d": deice[m]})

    def per(col: str) -> tuple[pd.Series, pd.Series]:
        g = frame.groupby([col, "d"])["v"].agg(["mean", "size"]).unstack("d")
        idx = g.index
        mean_on = g[("mean", True)] if ("mean", True) in g.columns else pd.Series(np.nan, index=idx)
        mean_off = g[("mean", False)] if ("mean", False) in g.columns else pd.Series(np.nan, index=idx)
        n_on = g[("size", True)].fillna(0) if ("size", True) in g.columns else pd.Series(0.0, index=idx)
        return mean_on - mean_off, n_on

    u_apt, _ = per("a")
    u_key, n_key = per("k")
    prior = frame.drop_duplicates("k").set_index("k")["a"].reindex(u_key.index).map(u_apt).astype("float64")
    shrunk = ((n_key * u_key.fillna(0) + shrink * prior) / (n_key + shrink)).where(u_key.notna(), prior)
    looked_up = key.map(shrunk).astype("float64")
    return looked_up.fillna(apt.map(u_apt).astype("float64")).to_numpy()


@feature(
    group="deice_location",
    outputs=DEICE_LOCATION,
    depends_on=("core_keys", "wx_now"),
    uses_target=True,
    doc="De-icing-hour taxi uplift per pier and per stand (fold-safe from the fit rows, and target-free from NM's proxy), alone and times the current de-icing condition.",
)
def deice_location(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    min_support = int(ctx.params.get("min_support", _proxylib.MIN_SUPPORT))
    shrink = float(ctx.params.get("shrink", SHRINK))
    dep = mvt.loc[ctx.dep_index]
    apt = dep["apt"].astype("string")
    stand = dep["STAND_mvt"].astype("string").fillna("__NA__")
    rwy = dep["RUNWAY_mvt"].astype("string").fillna("__NA__")
    keys = {"pier": stand_prefix_key(dep["apt"], dep["STAND_mvt"]), "stand": apt + "|" + stand}
    winter = dep["MVT_TIME_UTC_mvt"].dt.month.isin(WINTER_MONTHS).to_numpy()
    deice = np.nan_to_num(ctx.dep("wx_now")["wx_deice"].to_numpy(dtype="float64")) > 0

    y = ctx.target.reindex(ctx.dep_index).astype("float64")
    fit = ctx.fit_mask.reindex(ctx.dep_index, fill_value=False).to_numpy(dtype=bool) & (y < TAXI_MAX_S).to_numpy()
    baseline = _hier_median(y, [apt + "|" + stand + "|" + rwy, apt + "|" + stand, apt + "|" + rwy, apt], fit, min_support)
    taxi_anomaly = (y - baseline).where(pd.Series(fit, index=y.index)).clip(*_proxylib.WIDE_CLIP)
    proxy_anomaly = _proxylib.proxy_anomaly(dep, min_support).clip(*_proxylib.WIDE_CLIP)

    out = {}
    for name, values, mask in (("deice_uplift", taxi_anomaly, fit & winter), ("deice_proxy_uplift", proxy_anomaly, winter)):
        for lvl, key in keys.items():
            u = _uplift(values, key, apt, deice, mask, shrink)
            out[f"{name}_{lvl}"] = u
            out[f"{name}_{lvl}_now"] = u * deice
    return pd.DataFrame(out, index=ctx.dep_index).astype("float32")
