"""The hindsight rung of the operational ladder: traffic counted over the flight's OWN taxi (lab only).

Tier: **lab only, HINDSIGHT**. The block below uses the flight's own takeoff (``MVT_TIME_UTC_mvt``) together with
its own recorded off-block, so its window is the flight's own taxi ``[b, t]``. That is known only once the
aircraft is airborne: it is never operational, never in a board or refit config, and never in an ``op_h1_*``
config (tests/test_operational_hind.py, tests/test_operational.py). It exists for one measurement: how much of
the accuracy that published taxi-out models report comes from counts taken over the flight's own taxi, with the
same definitions on our ten airports and our holdout. Such counts drive, among others, the queue size Q of
Idris et al. (others taking off between the own pushback and the own takeoff), the hindsight congestion group
("Group E") of Wang et al., and the departure/arrival queue-length and simultaneous-flow indicators (AQLI,
SCFI) of the Wu, Tang, Yin and He line of work (the report's literature notes). The rung is
``configs/op_h0_r5_hind.yaml`` = ``op_h0_r5`` plus this block; ``H0 r5 hind - H0 r5`` is the hindsight share.

Definitions (whole seconds, `_oplib.secs`), for a training departure with recorded off-block ``b`` and takeoff
``t``, the window start capped at one hour before the takeoff, ``b' = max(b, t - HIND_CAP_S)``:

* ``dep_apt`` / ``dep_rwy`` -- OTHER takeoffs at the airport / on the own runway in ``[b', t]`` (D_AQLI);
* ``arr_apt`` -- landings at the airport in ``[b', t]`` (A_AQLI);
* ``dep_taxiing`` -- OTHER departures whose own taxi ``[b_o, t_o]`` overlaps ``[b', t]`` (D_SCFI): recorded
  off-block at or before ``t`` and takeoff at or after ``b'``. Only records with ``0 <= t_o - b_o <= HIND_CAP_S``
  are taxi intervals (a stale schedule or a wrong-date block time is not).

The own flight is removed from every count. NaN where there is no recorded off-block (every ranking row) or the
takeoff precedes it. The window *length* is never an output: it is the target.

Auto-imported by the registry: no work at import time. ``_oplib`` and ``congestion`` are imported as modules so
that `build._related_modules` hashes them into this block's cache key; `_oplib` and `operational` are not edited
by this module, so the existing op blocks keep their hashes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc.features import _oplib, congestion  # noqa: F401  (module objects: hashed into this block's cache key)
from prc.features.registry import FeatureContext, feature

HIND_CAP_S = 3600.0
"""Longest own-taxi window counted: the window starts at most an hour before the own takeoff."""

HIND = ("dep_apt", "dep_rwy", "arr_apt", "dep_taxiing")


def own_taxi_window(b: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(start, back)`` of the own-taxi window ``[start, t]``: ``start = max(b, t - HIND_CAP_S)``, ``back = t - start``.

    Both NaN where ``b`` or ``t`` is missing or ``t < b``.
    """
    b = np.asarray(b, dtype="float64")
    t = np.asarray(t, dtype="float64")
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(b) & np.isfinite(t) & (t >= b)
        start = np.where(ok, np.maximum(b, t - HIND_CAP_S), np.nan)
    return start, t - start


def hind_counts(mvt: pd.DataFrame, q_index: pd.Index, offblocks: pd.Series) -> pd.DataFrame:
    """The four own-taxi counts of `HIND` for the departures ``q_index`` (see the module docstring)."""
    dep = _oplib.departures(mvt)
    arr = _oplib.arrivals(mvt)
    q = mvt.loc[q_index]

    dep_t = _oplib.secs(dep["MVT_TIME_UTC_mvt"])
    dep_b = offblocks.reindex(dep.index).to_numpy(dtype="float64")
    dep_apt = dep["apt"].astype("string").to_numpy(dtype=object)
    dep_rwy = _oplib.runway_key(dep).to_numpy(dtype=object)
    arr_t = _oplib.secs(arr["MVT_TIME_UTC_mvt"])
    arr_apt = arr["apt"].astype("string").to_numpy(dtype=object)

    t = _oplib.secs(q["MVT_TIME_UTC_mvt"])
    b = offblocks.reindex(q_index).to_numpy(dtype="float64")
    q_apt = q["apt"].astype("string").to_numpy(dtype=object)
    q_rwy = _oplib.runway_key(q).to_numpy(dtype=object)
    start, back = own_taxi_window(b, t)

    out = {
        # the own takeoff sits at the window's end: backward_count removes it (own_t = t)
        "dep_apt": _oplib.backward_count(dep_t, dep_apt, t, q_apt, back, own_t=t),
        "dep_rwy": _oplib.backward_count(dep_t, dep_rwy, t, q_rwy, back, own_t=t),
        "arr_apt": _oplib.backward_count(arr_t, arr_apt, t, q_apt, back),
    }

    # D_SCFI: taxi intervals [b_o, t_o] with b_o <= t and t_o >= start, i.e. #(b_o <= t) - #(t_o <= start - 1)
    # (every interval with t_o < start also has b_o <= t, so the second set is inside the first).
    with np.errstate(invalid="ignore"):
        taxi = dep_t - dep_b
        is_iv = np.isfinite(taxi) & (taxi >= 0) & (taxi <= HIND_CAP_S)
    iv_g = _oplib._series(np.where(is_iv, dep_apt, None))
    q_g = _oplib._series(q_apt)
    began = congestion.count_upto(_oplib._series(np.where(is_iv, dep_b, np.nan)), iv_g, _oplib._series(t), q_g)
    ended = congestion.count_upto(_oplib._series(np.where(is_iv, dep_t, np.nan)), iv_g, _oplib._series(start - 1.0), q_g)
    with np.errstate(invalid="ignore"):
        own_iv = np.isfinite(t - b) & (t - b >= 0) & (t - b <= HIND_CAP_S)
    out["dep_taxiing"] = began.astype("int64") - ended.astype("int64") - own_iv.astype("int64")

    frame = pd.DataFrame(out, index=q_index)[list(HIND)].astype("float32")
    return _oplib._mask(frame, start)


@feature(
    group="op_lab_h0_hind",
    outputs=tuple(f"op_h0_hind_{c}" for c in HIND),
    external=_oplib.LAB_EXTERNAL,
    doc="LAB, HINDSIGHT, 2025 only: other takeoffs, landings and taxiing departures during the flight's own taxi [off-block, takeoff].",
)
def op_h0_hind_lab(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """Traffic during the flight's OWN taxi: the window is ``[recorded off-block, own takeoff]``, capped at an hour.

    Other takeoffs at the airport and on the own runway, landings at the airport, and other departures whose
    taxi overlaps the window; the own flight excluded throughout. NaN for ranking rows (no recorded off-block).

    Tier: lab only, HINDSIGHT: uses the flight's own takeoff; never operational, never in a board or refit
    config; exists to measure how much of the literature's accuracy comes from own-taxi counts (Idris Q,
    Wang Group E, Wu/Tang/Yin/He AQLI/SCFI).
    """
    frame = hind_counts(mvt, ctx.dep_index, _oplib.load_offblocks())
    frame.columns = [f"op_h0_hind_{c}" for c in frame.columns]
    return frame
