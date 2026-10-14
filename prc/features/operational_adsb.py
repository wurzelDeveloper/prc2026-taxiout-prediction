"""The operational ladder's ADS-B surface picture: what public ADS-B showed of OTHER aircraft on the airport before the
flight moves (findings 10.31; rung R6 = R4 + `op_h1_adsb`).

* ``op_h1_adsb`` -- at H1 (15 min before the planned off-block, `_oplib.anchor_h1`), read ``ADSB_LAG_S`` earlier.
  Board- and final-scorable. Tier **O** (other aircraft's public positions up to the anchor; adsb.lol, ODbL);
  ``rwy_own_hold_now`` is **O*** (the planned runway assumed known).
* ``op_h0_adsb`` -- the same at H0 (the recorded off-block, `_oplib.anchor_h0`): lab only, 2025 only, NaN for ranking
  rows; it declares the lab external (`_oplib.LAB_EXTERNAL`) and sits in its own group (``op_lab_h0_adsb``), so
  ``group:op_lab_h0`` does not pull it in. Question: can public ADS-B replace the airport's own off-block feed at the push?

Both read the consolidated event table of an ADS-B surface snapshot (`prc adsb-surface`, prc/adsb/surface.py) and its
frozen training support table (``prc adsb-surface --support``) as external inputs, and the geometry's stands, stand
matches (the own stand and its neighbours) and runway ends (the own runway). The own airframe is removed at event level
(its callsign, or an origin near its own stand), so moving or deleting its points leaves every value bit-identical
(tests/test_operational_adsb.py, T1d); every event is used only once it was knowable at the query time ``a - 30 s``
(T2, the table rebuilt from points censored there, on random and on adversarial anchors). Coverage columns (``cov_*``)
gate the three column families (taxi, hold, stand; `_adsblib`) -- NaN, never zero, where a family's reception is not
established or lies outside the airport's 2025 training regime -- and are meant for gating and strata: the configs
drop them from the model matrix (``features.drop_columns``; critique F6). Parameters (``features.params.op_h1_adsb``):
`_adsblib.DEFAULTS`, among them ``shift_days`` (the shifted placebo, >= 0) and ``coverage_ref`` (``scheduled`` for
the public-data rung).

The flight's own takeoff is read only to remove it from the recorded reference movements (as ``op_h1_traffic``); the own
NM actuals, ``EOBT_1`` and (at H1) the own off-block are never read. Auto-imported by the registry: no work at import
time; the helper modules are imported as module objects so that `build._related_modules` hashes them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc import paths
from prc.features import _adsblib, _geolib, _oplib, congestion, core  # noqa: F401  (module objects: hashed into these blocks' keys)
from prc.features.registry import FeatureContext, feature

ADSB_EXTERNAL = (paths.ADSB_SURFACE_EVENTS_FILE, paths.ADSB_SURFACE_SUPPORT_FILE, paths.GEO_STANDS_FILE,
                 paths.GEO_STAND_MATCHES_FILE, paths.GEO_RUNWAY_ENDS_FILE)


def _names(prefix: str) -> tuple[str, ...]:
    return tuple(f"{prefix}{c}" for c in _adsblib.COLUMNS)


def _surface(mvt: pd.DataFrame, ctx: FeatureContext, horizon: str) -> pd.DataFrame:
    q = mvt.loc[ctx.dep_index]
    a = _oplib.anchor_h1(q) if horizon == "h1" else _oplib.anchor_h0(q, _oplib.load_offblocks())
    frame = _adsblib.surface_at(mvt, ctx.dep_index, a, ctx.params)
    frame.columns = [f"op_{horizon}_adsb_{c}" for c in frame.columns]
    return _oplib._mask(frame, a).reindex(ctx.dep_index).astype("float32")


@feature(
    group="op_h1_adsb",
    outputs=_names("op_h1_adsb_"),
    external=ADSB_EXTERNAL,
    doc="ADS-B surface picture of other aircraft at H1 (taxiing out and in, runway queue, remote stops, pushes, recent taxi times), coverage-gated.",
)
def op_h1_adsb(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """Other aircraft on the surface at H1 - 30 s, from public ADS-B: taxiing out (and normalised by coverage), taxiing
    in, taxi-outs begun in the last 15 min, runway holds (all, own runway), remote stops now and in the last hour, the
    mean elapsed taxi of those moving out, stationary time and share of recent taxi-outs, remote-stop and hold
    durations; pushes in the last 15 min and the taxi time of recent push-origin takeoffs (stand gate); coverage.

    Operational availability: tier O (public ADS-B positions of other aircraft up to the anchor; the reference movements
    are the airport's records, or the schedule with ``coverage_ref: scheduled``); ``rwy_own_hold_now`` is O* (the
    planned runway assumed known).
    """
    return _surface(mvt, ctx, "h1")


@feature(
    group="op_lab_h0_adsb",
    outputs=_names("op_h0_adsb_"),
    external=(*ADSB_EXTERNAL, *_oplib.LAB_EXTERNAL),
    doc="LAB, 2025 only: the ADS-B surface picture of other aircraft at the recorded off-block (H0), coverage-gated.",
)
def op_h0_adsb(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """`op_h1_adsb` at H0 (the recorded off-block - 30 s). The own push starts at the anchor, so the own-stand rule of
    the exclusion matters here (critique F2).

    Operational availability: tier O at pushback (own-runway hold O*). Lab only: the anchor is the airport's own
    off-block record, 2025 only (NaN for ranking rows).
    """
    return _surface(mvt, ctx, "h0")
