"""Whole-frame statistics when a finished run is applied to another data tree (`prc predict`).

Most feature blocks look at one row, or at the hours around it, but some summarise the *whole* movement frame: every
departure, training and ranking alike. At training time that frame held 2025 plus the source run's ranking set (Jan+Jul
2026, 14 months); a tree holding the final ranking set (January, February, June and July 2026; findings 10.33) holds
16. Two consequences, two tools:

* The counts of `stand_runway_frequency` and `operator_presence` (departures from this stand, on this runway, on this
  pairing; this carrier's and this type's departures at the airport) grow by about 13 % on every row, and the model
  would read every stand as 13 % busier than the stands it was fitted on. `rescale_counts` puts the ranking rows back
  on the scale the model knows: a key the source frame had takes the source frame's own count (transplanted: the same
  value the training rows with that key carried); a key new in the new ranking set takes ``n_train + w_apt *
  n_rank_new``, with ``w_apt`` the source ranking set's departures at that airport over the new set's (1 on the source's
  own rows, so a reproduction stays exact); the shares are recomputed from the corrected counts. The key logic mirrors
  the two blocks; `check_counts` holds it against the source run's cached values and against the new tree's uncorrected
  ones every time, so an edit to either block cannot leave it behind unnoticed.
* Everything else that reads the whole frame (`load_normalised`, `runway_sequence`'s p99, the proxy and taxi-in
  baselines of `_proxylib` / `_arrlib`, the turnaround medians, the fold-dependent statistics) moves less and is not
  corrected; `parity_report` measures it: each block's values on the training rows -- the same rows in both trees --
  against the source run's cache. In a rehearsal on the delivered Jan+Jul file every column must come out identical.

`unseen_report` counts the ranking rows whose categorical level the model never saw (per airport for per-airport
models) next to the same share on the source's own ranking rows, the baseline a released file is judged against.

The source run's values come from the shared feature cache, by the block hashes the run recorded (`SourceFrame`); a
block whose cache file is gone is reported as such, and the count keys then come from the source tree's movement frame.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from prc import paths
from prc.models.base import NA_LEVEL

COUNT_BLOCKS = ("stand_runway_frequency", "operator_presence")
"""The blocks whose columns `rescale_counts` corrects."""
COUNT_COLUMNS: dict[str, tuple[str, str]] = {
    # column: (block, counted key) -- `prc.features.stand_runway.stand_runway_frequency`, `aircraft.operator_presence`
    "stand_freq": ("stand_runway_frequency", "apt_stand"),
    "runway_freq": ("stand_runway_frequency", "apt_runway"),
    "stand_runway_freq": ("stand_runway_frequency", "apt_stand_runway"),
    "carrier_apt_freq": ("operator_presence", "carrier_apt"),
    "type_apt_freq": ("operator_presence", "type_apt"),
}
RATIO_COLUMNS: dict[str, tuple[str, str, str]] = {
    # column: (block, numerator key, denominator key)
    "runway_share_at_stand": ("stand_runway_frequency", "apt_stand_runway", "apt_stand"),
    "carrier_apt_share": ("operator_presence", "carrier_apt", "apt"),
    "carrier_stand_concentration": ("operator_presence", "carrier_pier", "carrier_apt"),
}
KEY_BLOCKS: dict[str, tuple[str, ...]] = {
    # the outputs the counted keys are made of, and the block that makes each
    "core_keys": ("apt",),
    "stand_runway_keys": ("apt_stand", "apt_runway", "apt_stand_runway", "stand_prefix"),
    "aircraft_identity": ("carrier_flightno", "aircraft_type"),
}

UNSEEN_FACTOR = 3.0
"""A categorical column is out of line when its unseen share exceeds this multiple of the source's own share ..."""
UNSEEN_MARGIN = 0.02
"""... and the source's share plus this (overall) ..."""
UNSEEN_MARGIN_AIRPORT = 0.05
"""... or, at one airport with at least `UNSEEN_MIN_ROWS` ranking rows, that airport's source share plus this."""
UNSEEN_MIN_ROWS = 200
UNSEEN_NO_BASELINE = 0.10
"""The limit where the source's own share is unknown (its cache is gone): only a wholesale change trips it."""
EXAMPLES = 5


# --- the source run's own feature values ---------------------------------------------------------------------------

@dataclass
class SourceFrame:
    """The source run's feature blocks as it built them: the shared cache, by the hashes in its ``feature_list.json``."""

    data_version: str
    hashes: dict[str, str]
    refit_fold: str
    """``refit__<split_hash>``: where its fold-dependent blocks were cached for the refit."""
    tree: Path
    """The data tree the run read (its movement frame is the fallback for the count keys)."""

    def path(self, spec) -> Path | None:
        digest = self.hashes.get(spec.name)
        if digest is None:
            return None
        fold = self.refit_fold if spec.uses_target else None
        return paths.feature_cache_dir(self.data_version, fold) / f"{spec.name}__{digest}.parquet"

    def has(self, spec) -> bool:
        p = self.path(spec)
        return p is not None and p.exists()

    def read(self, spec, columns: list[str] | None = None) -> pd.DataFrame | None:
        """The cached block (only ``columns`` if given), or None when the run did not build it or the file is gone."""
        return pd.read_parquet(self.path(spec), columns=columns) if self.has(spec) else None


def read_cached(spec, data_version: str, digest: str, columns: list[str] | None = None, fold: str | None = None):
    """A block from the cache of ``data_version`` (the predict run's own, just built), or None."""
    p = paths.feature_cache_dir(data_version, fold if spec.uses_target else None) / f"{spec.name}__{digest}.parquet"
    return pd.read_parquet(p, columns=columns) if p.exists() else None


# --- counts ----------------------------------------------------------------------------------------------------------

def count_keys(keys: pd.DataFrame) -> dict[str, pd.Series]:
    """The counted keys, built exactly as the two blocks build them (strings; no key is ever missing there)."""
    apt = keys["apt"].astype("string")
    out = {"apt": apt}
    if "apt_stand" in keys.columns:
        for col in ("apt_stand", "apt_runway", "apt_stand_runway"):
            out[col] = keys[col].astype("string")
    if "carrier_flightno" in keys.columns:
        carrier_apt = (apt + "_" + keys["carrier_flightno"].astype("string")).astype("string")
        out["carrier_apt"] = carrier_apt
        out["type_apt"] = (apt + "_" + keys["aircraft_type"].astype("string")).astype("string")
        out["carrier_pier"] = (carrier_apt + "_" + keys["stand_prefix"].astype("string")).astype("string")
    return out


def needed_keys(blocks) -> list[str]:
    """The counted keys the columns of ``blocks`` need."""
    out: list[str] = []
    for _, (b, key) in COUNT_COLUMNS.items():
        if b in blocks:
            out.append(key)
    for _, (b, num, den) in RATIO_COLUMNS.items():
        if b in blocks:
            out += [num, den]
    return list(dict.fromkeys(out))


def key_columns(blocks) -> dict[str, list[str]]:
    """Per key block, the outputs the counted keys of ``blocks`` are made of."""
    want = {"apt"}
    if "stand_runway_frequency" in blocks:
        want |= {"apt_stand", "apt_runway", "apt_stand_runway"}
    if "operator_presence" in blocks:
        want |= {"carrier_flightno", "aircraft_type", "stand_prefix"}
    return {b: [c for c in cols if c in want] for b, cols in KEY_BLOCKS.items() if any(c in want for c in cols)}


def block_columns_from_counts(block: str, n: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The block's columns from per-row counts of its keys, with the block's own dtypes (float32 throughout)."""
    out: dict[str, np.ndarray] = {}
    for col, (b, key) in COUNT_COLUMNS.items():
        if b == block:
            out[col] = np.asarray(n[key], dtype=np.float32)
    for col, (b, num, den) in RATIO_COLUMNS.items():
        if b == block:
            with np.errstate(divide="ignore", invalid="ignore"):
                out[col] = (np.asarray(n[num], dtype=np.float32) / np.asarray(n[den], dtype=np.float32)).astype(np.float32)
    return out


def frame_values(keys: pd.DataFrame, blocks) -> dict[str, np.ndarray]:
    """What the count blocks compute over the frame whose departures ``keys`` describes (every column of ``blocks``)."""
    k = count_keys(keys)
    n = {key: k[key].map(k[key].value_counts()).to_numpy(dtype="float64", na_value=np.nan) for key in needed_keys(blocks)}
    out: dict[str, np.ndarray] = {}
    for block in blocks:
        out |= block_columns_from_counts(block, n)
    return out


def check_counts(keys: pd.DataFrame, values: pd.DataFrame, blocks) -> dict:
    """``values`` (a frame's count columns as the blocks built them, indexed like ``keys``) against `frame_values`
    recomputed from ``keys``. Returns ``{"checked": [...], "mismatch": [...]}``; a column ``values`` lacks is skipped."""
    expect = frame_values(keys, blocks)
    checked, mismatch = [], []
    for col, want in expect.items():
        if col not in values.columns:
            continue
        got = values[col].reindex(keys.index).to_numpy(dtype=np.float32)
        checked.append(col)
        if not np.array_equal(got, want, equal_nan=True):
            mismatch.append(col)
    return {"checked": checked, "mismatch": mismatch}


def rescale_counts(new_keys: pd.DataFrame, src_keys: pd.DataFrame, rank_idx: pd.Index, train_ids: pd.Index, blocks):
    """The count columns of ``blocks`` for the ranking rows ``rank_idx`` of the new frame, on the source frame's scale.

    ``new_keys`` / ``src_keys``: the key columns (`KEY_BLOCKS`) over every departure of the new and of the source frame;
    ``train_ids``: the training departures (in both frames). Returns ``({column: float32 values over rank_idx}, record,
    {column: whether the row's key(s) were in the source frame})``.
    """
    kn, ks = count_keys(new_keys), count_keys(src_keys)
    src_rank = src_keys.index.difference(train_ids)
    apt_rank = kn["apt"].reindex(rank_idx)
    n_new_apt = apt_rank.value_counts()
    n_src_apt = ks["apt"].reindex(src_rank).value_counts()
    w = (n_src_apt.reindex(n_new_apt.index).astype("float64") / n_new_apt.astype("float64"))
    w_row = apt_rank.map(w).astype("float64").fillna(1.0).to_numpy()

    n: dict[str, np.ndarray] = {}
    seen_key: dict[str, np.ndarray] = {}
    rows: dict[str, dict] = {}
    for key in needed_keys(blocks):
        src_counts = ks[key].value_counts()
        k_rank = kn[key].reindex(rank_idx)
        transplanted = k_rank.map(src_counts).to_numpy(dtype="float64", na_value=np.nan)
        n_train = k_rank.map(kn[key].reindex(train_ids).value_counts()).to_numpy(dtype="float64", na_value=np.nan)
        n_rank = k_rank.map(k_rank.value_counts()).to_numpy(dtype="float64", na_value=np.nan)
        formula = np.where(np.isfinite(n_train), n_train, 0.0) + w_row * n_rank
        seen = np.isfinite(transplanted)
        n[key] = np.where(seen, transplanted, formula)
        seen_key[key] = seen
        rows[key] = {"seen": int(seen.sum()), "new": int((~seen).sum()), "new_levels": int(k_rank[~seen].nunique())}
    corrected: dict[str, np.ndarray] = {}
    for block in blocks:
        corrected |= block_columns_from_counts(block, n)
    seen_col = {col: seen_key[key] for col, (b, key) in COUNT_COLUMNS.items() if b in blocks}
    seen_col |= {col: seen_key[num] & seen_key[den] for col, (b, num, den) in RATIO_COLUMNS.items() if b in blocks}
    record = {
        "w": {str(a): (float(v) if np.isfinite(v) else None) for a, v in w.items()},
        "source_rank_departures": {str(a): int(v) for a, v in n_src_apt.items()},
        "rank_departures": {str(a): int(v) for a, v in n_new_apt.items()},
        "keys": rows,
    }
    return corrected, record, seen_col


def column_summary(before: np.ndarray, after: np.ndarray, seen: np.ndarray | None = None) -> dict:
    """Means before and after, and on rows whose key the source had, the uncorrected value over the corrected one
    (how much the new frame had inflated that key)."""
    b = np.asarray(before, dtype="float64")
    a = np.asarray(after, dtype="float64")
    out = {"mean_before": float(np.nanmean(b)) if len(b) else None, "mean_after": float(np.nanmean(a)) if len(a) else None,
           "rows_changed": int(np.sum(~np.isclose(a, b, rtol=0, atol=0, equal_nan=True)))}
    mask = np.isfinite(a) & np.isfinite(b) & (a != 0)
    if seen is not None:
        mask &= seen
    if mask.any():
        r = b[mask] / a[mask]
        out["inflation_seen"] = {"mean": float(np.mean(r)), "p10": float(np.quantile(r, 0.1)),
                                 "p50": float(np.quantile(r, 0.5)), "p90": float(np.quantile(r, 0.9))}
    return out


# --- parity on the training rows -------------------------------------------------------------------------------------

def _is_categorical(s: pd.Series) -> bool:
    return isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype)


def compare_columns(new: pd.Series, old: pd.Series) -> dict:
    """One column on the same rows in two frames: the share of rows that differ, and how."""
    n = len(new)
    if n == 0:
        return {"rows": 0, "changed": 0.0}
    if _is_categorical(new) or _is_categorical(old):
        a = new.astype("string").fillna(NA_LEVEL).to_numpy()
        b = old.astype("string").fillna(NA_LEVEL).to_numpy()
        changed = a != b
        return {"rows": n, "kind": "categorical", "changed": float(changed.mean())}
    x = pd.to_numeric(new, errors="coerce").to_numpy(dtype="float64")
    y = pd.to_numeric(old, errors="coerce").to_numpy(dtype="float64")
    nx, ny = np.isnan(x), np.isnan(y)
    both = ~nx & ~ny
    close = np.zeros(n, dtype=bool)
    close[both] = np.isclose(x[both], y[both], rtol=1e-6, atol=1e-9)
    changed = (nx != ny) | (both & ~close)
    out = {"rows": n, "kind": "numeric", "changed": float(changed.mean()), "nan_changed": float((nx != ny).mean())}
    if changed.any():
        out |= {
            "mean_new": float(np.nanmean(x)) if (~nx).any() else None,
            "mean_source": float(np.nanmean(y)) if (~ny).any() else None,
            "max_abs_diff": float(np.max(np.abs(x[both] - y[both]))) if both.any() else None,
        }
        if out["mean_source"]:
            out["mean_ratio"] = out["mean_new"] / out["mean_source"] if out["mean_new"] is not None else None
    return out


def parity_report(specs, block_columns: dict[str, list[str]], new_hashes: dict[str, str], X: pd.DataFrame,
                  train_ids: pd.Index, source: SourceFrame, new_version: str, new_fold: str, *, log=print) -> dict:
    """Per block: its columns on the training rows in the new matrix ``X`` against the source run's cached block.

    The training rows are the same rows in both frames, so any difference is the frame around them: the ranking months
    exchanged, an input file rebuilt. Results are kept per (block, source hash, new hash, folds) in the new version's
    cache (``_parity/``), so the runs of one blend share the work. A block whose source cache is gone is reported as
    ``unavailable``; one read from the very same cache file (same tree and version) as ``identical`` unread."""
    memo_dir = paths.feature_cache_dir(new_version) / "_parity"
    memo_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict] = {}
    for spec in specs:
        cols = [c for c in block_columns.get(spec.name, []) if c in X.columns]
        src_hash, new_hash = source.hashes.get(spec.name), new_hashes.get(spec.name)
        if not cols or src_hash is None or new_hash is None:
            continue
        new_file = paths.feature_cache_dir(new_version, new_fold if spec.uses_target else None) / f"{spec.name}__{new_hash}.parquet"
        if source.path(spec) == new_file:
            out[spec.name] = {"status": "identical", "how": "the same cache file", "columns": len(cols)}
            continue
        folds = f"__{source.refit_fold}__{new_fold}" if spec.uses_target else ""
        memo = memo_dir / f"{spec.name}__{src_hash}__{new_hash}{folds}.json"
        if memo.exists():
            out[spec.name] = json.loads(memo.read_text(encoding="utf-8"))
            continue
        old = source.read(spec, columns=cols)
        if old is None:
            out[spec.name] = {"status": "unavailable", "reason": f"no cached source block {spec.name}__{src_hash}"}
            continue
        rows = train_ids.intersection(old.index)
        new = X.loc[rows, cols]
        old = old.loc[rows, cols]
        per_col = {c: compare_columns(new[c], old[c]) for c in cols}
        changed = {c: v for c, v in per_col.items() if v["changed"] > 0}
        entry = {
            "status": "identical" if not changed else "changed",
            "rows": int(len(rows)),
            "columns": len(cols),
            "max_changed_share": max((v["changed"] for v in per_col.values()), default=0.0),
            "changed_columns": changed,
        }
        memo.write_text(json.dumps(entry, indent=1), encoding="utf-8")
        out[spec.name] = entry
        del new, old
    same = [b for b, e in out.items() if e.get("status") == "identical"]
    moved = {b: e for b, e in out.items() if e.get("status") == "changed"}
    missing = [b for b, e in out.items() if e.get("status") == "unavailable"]
    log(f"    parity on the training rows: {len(same)} block(s) identical, {len(moved)} changed"
        + (f", {len(missing)} without a source cache {missing}" if missing else ""))
    for block, e in sorted(moved.items(), key=lambda kv: -kv[1]["max_changed_share"]):
        worst = sorted(e["changed_columns"].items(), key=lambda kv: -kv[1]["changed"])[:3]
        desc = ", ".join(
            f"{c} {100 * v['changed']:.1f} %" + (f" (mean x{v['mean_ratio']:.3f})" if v.get("mean_ratio") else "")
            for c, v in worst
        )
        log(f"      {block}: {desc}")
    return out


# --- unseen categorical levels -----------------------------------------------------------------------------------------

def level_codes(values: pd.Series) -> np.ndarray:
    """One integer per row and level as CatBoost sees it: the string, missing values as ``__NA__`` (so a missing value
    and a literal ``__NA__`` are the same level). Works on the categories, not on every row, for a categorical."""
    if isinstance(values.dtype, pd.CategoricalDtype):
        names = values.cat.categories.astype("string").to_numpy(dtype=object, na_value=NA_LEVEL)
        canon, _ = pd.factorize(np.append(names, NA_LEVEL))
        codes = values.cat.codes.to_numpy()
        return np.where(codes >= 0, canon[np.maximum(codes, 0)], canon[-1]).astype("int64")
    codes, _ = pd.factorize(values.astype("string").fillna(NA_LEVEL))
    return codes.astype("int64")


def unseen_rows(values: pd.Series, apt: pd.Series, fit_ids: pd.Index, rank_ids: pd.Index, per_airport: bool) -> pd.Series:
    """Per ranking row: whether its level (missing = ``__NA__``) never occurs among the fit rows -- of its own airport
    when ``per_airport`` (each airport's model saw only its own rows)."""
    key = level_codes(values)
    if per_airport:
        apt_codes = level_codes(apt.reindex(values.index))
        key = apt_codes * (int(key.max()) + 2 if len(key) else 1) + key
    fit_pos = values.index.get_indexer(fit_ids)
    rank_pos = values.index.get_indexer(rank_ids)
    if (rank_pos < 0).any():
        raise ValueError(f"{int((rank_pos < 0).sum())} ranking rows are not rows of the column")
    seen = np.unique(key[fit_pos[fit_pos >= 0]])
    return pd.Series(~np.isin(key[rank_pos], seen), index=rank_ids)


def unseen_summary(unseen: pd.Series, values: pd.Series, apt: pd.Series) -> dict:
    """Shares overall and per airport, and the most frequent unseen levels."""
    rank_apt = apt.reindex(unseen.index).astype("string")
    by = unseen.groupby(rank_apt.to_numpy()).agg(["mean", "size"])
    levels = values.reindex(unseen.index[unseen.to_numpy()]).astype("string").fillna(NA_LEVEL)
    counts = levels.value_counts().head(EXAMPLES)
    return {
        "rank_rows": int(unseen.sum()),
        "rank_share": float(unseen.mean()) if len(unseen) else 0.0,
        "levels": int(levels.nunique()),
        "examples": {str(k): int(v) for k, v in counts.items()},
        "by_airport": {str(a): {"share": float(r["mean"]), "rows": int(r["size"])} for a, r in by.iterrows()},
    }


def unseen_limit(base: float | None, margin: float) -> float:
    if base is None:
        return UNSEEN_NO_BASELINE
    return max(UNSEEN_FACTOR * base, base + margin)


def unseen_verdict(report: dict) -> list[str]:
    """The (column, airport) pairs whose unseen share is out of line with the source's own ranking rows."""
    out = []
    for col, info in report.items():
        base = info.get("baseline") or {}
        overall = base.get("rank_share")
        limit = unseen_limit(overall, UNSEEN_MARGIN)
        info["limit"] = limit
        if info["rank_share"] > limit:
            out.append(f"{col}: {100 * info['rank_share']:.1f} % of the ranking rows (limit {100 * limit:.1f} %, "
                       f"source {('%.1f %%' % (100 * overall)) if overall is not None else 'unknown'})")
        base_apt = base.get("by_airport") or {}
        for apt, a in info["by_airport"].items():
            if a["rows"] < UNSEEN_MIN_ROWS:
                continue
            b = (base_apt.get(apt) or {}).get("share") if base else None
            lim = unseen_limit(b if base else None, UNSEEN_MARGIN_AIRPORT)
            if a["share"] > lim:
                out.append(f"{col} at {apt}: {100 * a['share']:.1f} % of {a['rows']:,} ranking rows (limit {100 * lim:.1f} %, "
                           f"source {('%.1f %%' % (100 * b)) if b is not None else 'unknown'})")
    return out


def unseen_report(X: pd.DataFrame, cat_cols: list[str], fit_ids: pd.Index, rank_ids: pd.Index, apt: pd.Series,
                  per_airport: bool, *, specs=(), block_columns=None, source: SourceFrame | None = None,
                  train_ids: pd.Index | None = None) -> dict:
    """Per categorical column of the model: the new ranking rows whose level is unseen, and as ``baseline`` the same on
    the source's own ranking rows against the same fit rows, from the source's cached blocks (None when gone)."""
    out = {}
    for col in cat_cols:
        u = unseen_rows(X[col], apt, fit_ids, rank_ids, per_airport)
        out[col] = unseen_summary(u, X[col], apt) | {"baseline": None}
    if source is None or train_ids is None or not block_columns:
        return out
    by_name = {s.name: s for s in specs}
    owner = {c: b for b, cols in block_columns.items() for c in cols}
    core = by_name.get("core_keys")
    src_apt = source.read(core, columns=["apt"]) if core is not None else None
    if src_apt is None:
        return out
    src_apt = src_apt["apt"]
    src_rank = src_apt.index.difference(train_ids)
    for block in dict.fromkeys(owner[c] for c in cat_cols if c in owner):
        cols = [c for c in cat_cols if owner.get(c) == block]
        spec = by_name.get(block)
        old = source.read(spec, columns=cols) if spec is not None else None
        if old is None:
            continue
        for col in cols:
            u = unseen_rows(old[col], src_apt, fit_ids, src_rank, per_airport)
            out[col]["baseline"] = unseen_summary(u, old[col], src_apt)
    return out
