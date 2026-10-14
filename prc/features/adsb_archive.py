"""The historical ADS-B archive: how past departures left their stand, as a lookup table.

The archive the owner approved on 9 Oct (lab/ideas/stand_archive.py, ported as a pre-movement input): per stand, how
often the aircraft are pushed back rather than powering out, the push and engine-start durations, and the typical time
from the off-block to reach given distances from the off-block position. At prediction time the model needs only this
table, never live ADS-B: per (airport, stand), (airport, pier), (airport, stand, runway) and (airport, runway), what
the fit set's archived tracks showed of departures that already happened.

Input (`paths.ADSB_ARCHIVE_FILE`, consolidated by `python -m prc.adsb.archive`; its digest joins the cache key): one
row per linked departure of the lab's pass 1 (`paths.ADSB_STAND_OBS`, ``obs_day``: the off-block kind and its timings
where the aircraft's own track shows the off-block fully) and the push cluster's distance to the matched OSM stand from
the pushback detector (``features3.parquet``). Fully observed off-blocks, the lab's rule:

* ``push``: a detected reverse-motion push, not censored, its start timed within 60 s, its cluster within 80 m of the
  stand where the stand is matched; the off-block is the push start;
* ``power``: no push, parked >= 120 s within 80 m of the matched stand and left with a gap <= 30 s; the off-block is
  the last point at the stand.

Per archived flight (`flight_values`; ADS-B clock, seconds, nothing relative to the recorded takeoff):

* ``power`` -- 1 for a power-out, 0 for a push, and only where the stand is matched (the push cluster's distance to the
  OSM stand is known): the power-out rule needs a matched stand, so at an unmatched stand every classified off-block
  would be a push and the share would read 0 by construction;
* ``push_dur_s`` -- the duration of a push (0 to 15 min; kept at unmatched stands too);
* ``es_dur_s`` -- the engine-start pause of a push: from the end of the push to the first forward move (0 to 15 min);
* ``start_8kt_s`` -- from the off-block to the first point at 8 kt or faster (push and power-out; 0 to 30 min);
* ``tau_100_s``, ``tau_300_s``, ``tau_1000_s`` -- from the off-block until the track first lies 100 / 300 / 1,000 m
  from the off-block position (the lab's radial profile; NaN where the crossing step spans more than 120 s; 0 to 1 h).

Deliberately not in the table: the whole taxi-out (ADS-B takeoff minus an off-block). Every archived flight is a fit
row with a target, so a per-stand taxi median is a lower-support copy of the target statistics the model already
learns on the stand and stand x runway keys (the first-movement version measured r = 0.80 with a fit-rows target
median on the same key and -0.03 with the residual against it), and measured from the first ADS-B movement it misses
the push and the engine-start pause wherever the reception starts late (median 55 s short, 347 s at LIRF). Nor the
stationary time after the last stand stop: a push ending beyond 60 m of the OSM stand put the engine-start pause into
it (median 0 s for pushes under 70 m, 169-192 s above); the pause is ``es_dur_s`` now.

Per key (`archive_at`): the median of each timing and the power-out share (the mean of ``power``), each shrunk toward
its parent with weight ``n / (n + K)`` (K = ``shrink_k``, 10; n = the records behind that statistic at that level):
stand -> pier -> airport; pier -> airport; (stand, runway) -> (pier, runway) -> (airport, runway) -> airport;
(airport, runway) -> airport. A missing stand or runway falls back to the parent level. The pier is the stand prefix
(`_oplib.pier_key`); at airports with numeric stand IDs (EDDM, EGLL, LEBL, LIRF and most of LEMD) it is empty, so
there the pier level is the whole airport and a stand without records reads the airport value. Coverage follows the
reception: Istanbul has none (NaN), Paris almost none (airport constants). Support counts are not outputs: a fit row
reads a month-out table, so a count would name its month and lie beyond every fit value at serving time.

Fold-dependent (``uses_target=True``) because it reads ``ctx.fit_mask``; it never reads the target. Only the fit rows'
records enter (rows, not days), and these are the cleaned fit rows: `data.clean` (targets above 2 h, wrong-date rows)
removes rows from the fit before the mask is built, so those flights' records never enter (a selection by the target
of a few hundred 2025 departures: negligible, but the table is not target-free in the strict sense). A query reads
the table that leaves out its own planned month (the schedule's month, `_horizon.plan_month`; no IOBT, findings
10.35), as `_oplib.NmBaseline` does, so a fit row's own record never enters what it reads. Holdout and 2026 queries read the full fit table. On the
seasonal holdout that table pools the ten fit months, some after the holdout month: transductive in time, so the
holdout gain is optimistic. The forward pair `op_hl24_r4_rwyblind_fwd` / `op_hl24_r4_arch_rwyblind_fwd` (fit Jan-Oct,
validate Nov+Dec, every record before its query) turned the 24 h gain of 0.7 s into a loss of 3.1 s of MAE, so no
deliverable reads the archive (findings 10.31). For 2026 it is all of 2025, strictly past.

Operational availability (`prc.features._long_tiers`): tier O as a table of past departures' tracks, refreshed from
archived ADS-B; the stand-keyed statistics are O* hours or more ahead (the stand allocation assumed known; the pier
level is O), the runway-keyed ones O* (the own planned runway assumed known; the long-horizon deliverables drop
them). The ADS-B link to a recorded departure used that past flight's recorded takeoff (allowed: it had happened).
Not lab-only: the table exists for 2026 queries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc import paths
from prc.features import _horizon, _oplib, core, stand_runway  # noqa: F401  (module objects: hashed into this block's cache key)
from prc.features.registry import FeatureContext, feature

ARCHIVE_EXTERNAL = (paths.ADSB_ARCHIVE_FILE,)

RADII = (100, 300, 1000)
"""Radii (m from the off-block position) of the time profile the table keeps (the lab's pass 1 has 10 to 3,000 m)."""
INPUT_COLUMNS = {
    "obs": ("apt", "kind", "push_dur", "es_dur", "start_8kt", *(f"tau_{r}" for r in RADII)),
    "features3": ("pb_cl_dist_stand",),
}
"""The columns `prc.adsb.archive` consolidates (index ``mvt_id``): the lab's pass-1 observations, the pushback detector."""
RAW = tuple(c for cols in INPUT_COLUMNS.values() for c in cols)

KINDS = ("push", "power")
PUSH_DUR_MAX_S = 900.0
ES_DUR_MAX_S = 900.0
START_8KT_MAX_S = 1800.0
TAU_MAX_S = 3600.0
SHRINK_K = 10.0

FLIGHT = ("power", "push_dur_s", "es_dur_s", "start_8kt_s", *(f"tau_{r}_s" for r in RADII))
STATS = (
    ("power_share", "power", "mean"),
    ("push_dur_s", "push_dur_s", "median"),
    ("es_dur_s", "es_dur_s", "median"),
    ("start_8kt_s", "start_8kt_s", "median"),
    *((f"tau_{r}_s", f"tau_{r}_s", "median") for r in RADII),
)
"""(output statistic, per-flight value, reduction)."""
LEVELS = ("apt", "pier", "stand", "rwy", "pier_rwy", "stand_rwy")
CHAINS = {
    "stand": ("apt", "pier", "stand"),
    "pier": ("apt", "pier"),
    "sr": ("apt", "rwy", "pier_rwy", "stand_rwy"),
    "rwy": ("apt", "rwy"),
}
"""Output level -> its shrinkage chain, root first."""
ARCH = tuple(f"{chain}_{name}" for chain in CHAINS for name, _, _ in STATS)


def _names(prefix: str, cols) -> tuple[str, ...]:
    return tuple(f"{prefix}{c}" for c in cols)


# --- the archived flights ----------------------------------------------------------------------------------

_FLIGHTS: dict = {}


def load_flights() -> pd.DataFrame:
    """The consolidated archive input (index ``mvt_id``, the `RAW` columns). Memoised on the file's mtime."""
    path = paths.ADSB_ARCHIVE_FILE
    key = (str(path), path.stat().st_mtime_ns)
    if _FLIGHTS.get("key") != key:
        frame = pd.read_parquet(path)
        missing = [c for c in RAW if c not in frame.columns]
        if missing:
            raise ValueError(f"{path} lacks {missing}: rebuild it with `python -m prc.adsb.archive`")
        if frame.index.has_duplicates:
            raise ValueError(f"{path} holds a departure twice")
        _FLIGHTS["key"], _FLIGHTS["value"] = key, frame
    return _FLIGHTS["value"]


def _col(raw: pd.DataFrame, name: str) -> np.ndarray:
    return pd.to_numeric(raw[name], errors="coerce").to_numpy(dtype="float64", na_value=np.nan)


def _within(values: np.ndarray, ok: np.ndarray, lo: float, hi: float) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.where(ok & (values >= lo) & (values <= hi), values, np.nan)


def flight_values(raw: pd.DataFrame) -> pd.DataFrame:
    """Per archived flight: power-out flag, push duration, engine-start pause, time to 8 kt and to the radii (`FLIGHT`)."""
    kind = raw["kind"].astype("string")
    push = kind.eq("push").fillna(False).to_numpy(dtype=bool)
    power = kind.eq("power").fillna(False).to_numpy(dtype=bool)
    seen = push | power
    matched = np.isfinite(_col(raw, "pb_cl_dist_stand"))
    duration = _col(raw, "push_dur")
    with np.errstate(invalid="ignore"):
        positive = duration > 0
    out = {
        "power": np.where(matched & push, 0.0, np.where(matched & power, 1.0, np.nan)),
        "push_dur_s": _within(duration, push & positive, 0.0, PUSH_DUR_MAX_S),
        "es_dur_s": _within(_col(raw, "es_dur"), push, 0.0, ES_DUR_MAX_S),
        "start_8kt_s": _within(_col(raw, "start_8kt"), seen, 0.0, START_8KT_MAX_S),
    }
    for r in RADII:
        out[f"tau_{r}_s"] = _within(_col(raw, f"tau_{r}"), seen, 0.0, TAU_MAX_S)
    return pd.DataFrame(out, index=raw.index)[list(FLIGHT)]


# --- the table -----------------------------------------------------------------------------------------------

def level_codes(dep: pd.DataFrame) -> dict[str, np.ndarray]:
    """Integer codes of every departure's key at each level of `LEVELS`; -1 where the stand / runway is missing."""
    apt = dep["apt"].astype("string")
    stand = dep["STAND_mvt"].astype("string")
    rwy = dep["RUNWAY_mvt"].astype("string")
    pier = _oplib.pier_key(dep).astype("string")
    has_s, has_r = stand.notna().to_numpy(), rwy.notna().to_numpy()

    def code(key: pd.Series, ok: np.ndarray) -> np.ndarray:
        values = key.to_numpy(dtype=object, na_value=None)
        values[~ok] = None
        return pd.factorize(values, use_na_sentinel=True)[0].astype("int64")

    return {
        "apt": code(apt, apt.notna().to_numpy()),
        "pier": code(pier, has_s),
        "stand": code(apt + "|" + stand, has_s),
        "rwy": code(apt + "|" + rwy, has_r),
        "pier_rwy": code(pier + "|" + rwy, has_s & has_r),
        "stand_rwy": code(apt + "|" + stand + "|" + rwy, has_s & has_r),
    }


def _tables(codes: np.ndarray, v: dict[str, np.ndarray], use: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per code of one level, for every statistic of `STATS`: its value on the finite ``use`` rows and its count."""
    size = int(codes.max()) + 1 if len(codes) else 0
    ok = use & (codes >= 0)
    frame = pd.DataFrame({c: v[c][ok] for c in FLIGHT})
    grouped = frame.groupby(codes[ok], sort=False)
    count = grouped.count()
    reduced = {
        "median": grouped[[c for _, c, how in STATS if how == "median"]].median(),
        "mean": grouped[[c for _, c, how in STATS if how == "mean"]].mean(),
    }
    out = {}
    for name, col, how in STATS:
        value, n = np.full(size, np.nan), np.zeros(size)
        value[reduced[how].index.to_numpy(dtype="int64")] = reduced[how][col].to_numpy(dtype="float64")
        n[count.index.to_numpy(dtype="int64")] = count[col].to_numpy(dtype="float64")
        out[name] = (value, n)
    return out


def _lookup(table: np.ndarray, codes: np.ndarray, missing: float = np.nan) -> np.ndarray:
    if not len(table):
        return np.full(len(codes), missing)
    return np.where(codes >= 0, table[np.clip(codes, 0, len(table) - 1)], missing)


def archive_at(dep: pd.DataFrame, q_index: pd.Index, values: pd.DataFrame, fit: np.ndarray, shrink_k: float = SHRINK_K) -> pd.DataFrame:
    """The archive statistics (`ARCH`) of each query departure, from the fit rows' records, month-out.

    ``dep``: every departure of the frame (keys and planned months); ``values``: `flight_values` of the archived
    flights (any index; reindexed onto ``dep``); ``fit``: the fit rows, aligned to ``dep``. A record enters only when
    its departure is a fit row with a planned month; a query reads the table without its own planned month when that
    month holds records, else the full table. The counts weigh the shrinkage only; they are not outputs.
    """
    codes = level_codes(dep)
    vals = values.reindex(dep.index)
    v = {c: vals[c].to_numpy(dtype="float64", na_value=np.nan) for c in FLIGHT}
    month = _horizon.plan_month(dep)
    has = np.zeros(len(dep), dtype=bool)
    for c in FLIGHT:
        has |= np.isfinite(v[c])
    rec = np.asarray(fit, dtype=bool) & has & np.isfinite(month)
    q_pos = dep.index.get_indexer(q_index)
    if (q_pos < 0).any():
        raise ValueError("archive_at: every query must be a departure of the frame")
    keys = _horizon.month_keys(month[q_pos], {int(m) for m in np.unique(month[rec])})
    out = {c: np.full(len(q_index), np.nan) for c in ARCH}
    for key in np.unique(keys):
        rows = np.flatnonzero(keys == key)
        use = rec if key < 0 else rec & (month != key)
        tables = {lvl: _tables(codes[lvl], v, use) for lvl in LEVELS}
        qp = q_pos[rows]
        for level, chain in CHAINS.items():
            for name, _, _ in STATS:
                value = None
                for lvl in chain:
                    table, count = tables[lvl][name]
                    c = codes[lvl][qp]
                    raw, n = _lookup(table, c), _lookup(count, c, 0.0)
                    if value is None:
                        value = raw
                        continue
                    w = n / (n + shrink_k)
                    with np.errstate(invalid="ignore"):
                        value = np.where(np.isfinite(raw), np.where(np.isfinite(value), w * raw + (1 - w) * value, raw), value)
                out[f"{level}_{name}"][rows] = value
    return pd.DataFrame(out, index=q_index)[list(ARCH)].astype("float32")


@feature(
    group="op_archive",
    outputs=_names("op_arch_", ARCH),
    uses_target=True,
    external=ARCHIVE_EXTERNAL,
    doc="Historical ADS-B archive per stand, pier, stand x runway and runway (fit rows only, month-out): power-out share, push and engine-start durations, time to 8 kt and to 100/300/1,000 m from the off-block.",
)
def op_adsb_archive(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """How departures from this stand (pier, stand x runway, runway) left it, from the fit rows' archived tracks.

    The power-out share, the median push duration, engine-start pause, time from the off-block to 8 kt and to 100 /
    300 / 1,000 m from the off-block position, shrunk toward pier and airport; from the fold's fit rows only
    (``ctx.fit_mask``; the target is never read), leaving out the query's own planned month. Horizon-free: a table of
    past departures, valid at any anchor; no live ADS-B at prediction time.

    Operational availability: tier O as a table of past tracks; stand-keyed columns O* hours ahead (stand allocation
    assumed known), runway-keyed columns O* (the own planned runway assumed known). Not lab-only.
    """
    dep = _oplib.departures(mvt)
    mask = ctx.fit_mask
    fit = np.zeros(len(dep), dtype=bool) if mask is None else mask.reindex(dep.index).fillna(False).to_numpy(dtype=bool)
    values = flight_values(load_flights())
    frame = archive_at(dep, ctx.dep_index, values, fit, float(ctx.params.get("shrink_k", SHRINK_K)))
    frame.columns = [f"op_arch_{c}" for c in frame.columns]
    return frame.reindex(ctx.dep_index)
