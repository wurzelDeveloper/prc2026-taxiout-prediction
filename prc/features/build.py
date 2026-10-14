"""Executing feature blocks: hashing, caching, validating, assembling.

Caching is **per block**, not per feature-set.  That is the decision that makes
iteration cheap: a leave-one-out ablation sweep over 8 blocks toggles one block
at a time and reuses the other seven's parquet files untouched, so the sweep
costs one model fit per block rather than one full feature build per block.

A block's cache key hashes its own source code, so editing a feature function
invalidates it automatically -- no stale results after a tweak -- and parent
hashes fold in, so editing a dependency invalidates everything downstream.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from dataclasses import dataclass

import pandas as pd

from prc import paths
from prc.features.registry import FeatureContext, FeatureSpec, resolve

__all__ = [
    "build_matrix", "block_hash", "cache_status", "check_tree_stamp", "code_hash", "FeatureMatrix", "resolve",
]


@dataclass
class FeatureMatrix:
    X: pd.DataFrame
    cat_cols: list[str]
    block_columns: dict[str, list[str]]
    block_hashes: dict[str, str]
    seconds: dict[str, float]

    @property
    def columns_to_block(self) -> dict[str, str]:
        return {c: b for b, cols in self.block_columns.items() for c in cols}


def _source_hash(spec: FeatureSpec) -> str:
    """Hash the block's code *and* the feature-package code it relies on.

    Hashing only ``spec.fn`` would miss module-level helpers (a block calling a
    shared `epoch_seconds` would keep a stale cache after that helper changed),
    so we also fold in the block's own module and any other `prc.features`
    module it imports from.  Helpers living outside `prc.features` are not
    tracked -- bump ``version=`` on the spec if you move logic there.
    """
    parts: list[str] = []
    try:
        parts.append(inspect.getsource(spec.fn))
        module = inspect.getmodule(spec.fn)
    except OSError:  # pragma: no cover - interactively defined blocks
        return hashlib.sha1(spec.fn.__name__.encode("utf-8")).hexdigest()[:12]

    seen: set[str] = set()
    for mod in _related_modules(module):
        if mod.__name__ in seen:
            continue
        seen.add(mod.__name__)
        try:
            parts.append(inspect.getsource(mod))
        except OSError:  # pragma: no cover
            continue
    return hashlib.sha1("".join(sorted(parts)).encode("utf-8")).hexdigest()[:12]


def _related_modules(module):
    if module is None:  # pragma: no cover
        return []
    out = [module]
    for value in vars(module).values():
        other = inspect.getmodule(value)
        name = getattr(other, "__name__", "")
        if other is not module and name.startswith("prc.features.") and name.rsplit(".", 1)[-1] not in {"registry", "build"}:
            out.append(other)
    return out


_DIGESTS: dict[tuple[str, int, int], str] = {}


def _file_digest(path: paths.Path) -> str:
    """sha1 of a file's bytes, or ``"missing"`` if it does not exist.

    Memoised on (path, size, mtime) so that `features --list` and every fold of
    a run do not re-read the same weather table just to recompute its key.
    """
    try:
        st = path.stat()
    except FileNotFoundError:
        return "missing"
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _DIGESTS:
        h = hashlib.sha1()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _DIGESTS[key] = h.hexdigest()[:12]
    return _DIGESTS[key]


def external_path(path: paths.Path, external_root: paths.Path | None = None) -> paths.Path:
    """Where an external input of a block is read from: ``path`` itself, or with ``external_root`` the same file in
    another data tree (``path`` relative to the current one, ``paths.DATA``, re-rooted there). A file outside the
    current data tree is not re-rooted."""
    if external_root is None:
        return path
    try:
        return paths.Path(external_root) / paths.Path(path).relative_to(paths.DATA)
    except ValueError:
        return path


def block_hash(
    spec: FeatureSpec,
    data_version: str,
    params: dict,
    parent_hashes: dict[str, str],
    *,
    external_root: paths.Path | None = None,
) -> str:
    """The block's cache key. ``external_root`` digests the external inputs as found in another data tree (what the
    key was, or would be, there): `prc.predict` checks a run's recorded keys against the tree it was trained in."""
    payload = {
        "name": spec.name,
        "version": spec.version,
        "src": _source_hash(spec),
        "outputs": list(spec.outputs),
        "data_version": data_version,
        "params": params,
        "parents": {k: parent_hashes[k] for k in sorted(spec.depends_on) if k in parent_hashes},
    }
    # Blocks that read files outside the movement frame fold in those files'
    # content, keyed by file name only (a checkout elsewhere shares caches). The
    # key is added only when there is something to add, so every other block's
    # hash -- and the cache file it points at -- is exactly what it was before.
    if spec.external:
        payload["external"] = {p.name: _file_digest(external_path(p, external_root)) for p in spec.external}
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def code_hash(spec: FeatureSpec, params: dict) -> str:
    """The part of a block's key that is its code alone (source, version, outputs, params): no data version, no
    parents, no input files. Recorded by `prc predict`, so that a later check needs no data tree."""
    payload = {"name": spec.name, "version": spec.version, "src": _source_hash(spec), "outputs": list(spec.outputs),
               "params": params}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]


# --- which data tree a cache directory belongs to ---------------------------------------------------------------------

TREE_STAMP = "TREE.json"
"""``cache/features/<data.version>/TREE.json``: the data tree (and movement frame) the version's blocks were built from."""


class CacheTreeMismatch(ValueError):
    """A feature-cache directory built from another data tree or another movement frame than this process reads."""


def frame_identity() -> dict:
    """The data tree this process reads (``PRC_DATA_DIR`` or the repository's ``data/``) and the frame `prepare` built
    there (`prc.data.frame_id` of its inputs; None for a frame built before ``prepare.json`` existed)."""
    from prc import data

    record = data.prepare_record()
    return {"data_tree": str(paths.DATA), "frame_id": (record or {}).get("frame_id")}


def check_tree_stamp(data_version: str, *, write: bool = True) -> dict:
    """Bind ``data_version`` to one data tree and one movement frame.

    The cache is shared by every data tree and partitioned by ``data.version`` only, so the first build under a
    version records the tree it read (and the frame id from ``prepare.json``); a later build from another tree -- a
    session where ``PRC_DATA_DIR`` was not set, or the other way round -- or from a frame re-prepared from other inputs
    under the same version (a corrected re-release with the same ids) is refused instead of being served the other
    tree's blocks. A stamp without a frame id (written before the frame had one) takes the current one. A missing stamp
    is written (``write``) and passes. Returns the stamp."""
    stamp_file = paths.feature_cache_dir(data_version) / TREE_STAMP
    current = frame_identity()
    if not stamp_file.exists():
        stamp = {"data_version": data_version, **current}
        if write:
            stamp_file.parent.mkdir(parents=True, exist_ok=True)
            stamp_file.write_text(json.dumps(stamp | {"created": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1),
                                  encoding="utf-8")
        return stamp
    stamp = json.loads(stamp_file.read_text(encoding="utf-8"))
    if not paths.same_dir(stamp["data_tree"], current["data_tree"]):
        raise CacheTreeMismatch(
            f"data.version '{data_version}' belongs to the data tree {stamp['data_tree']} ({stamp_file}), but this "
            f"process reads {current['data_tree']}: another data tree under the same data.version would be served "
            f"that tree's cached blocks. Point PRC_DATA_DIR at {stamp['data_tree']}, or give this tree its own "
            f"data.version (delete {stamp_file.parent} only if nothing needs that cache)."
        )
    if stamp.get("frame_id") and current["frame_id"] and stamp["frame_id"] != current["frame_id"]:
        raise CacheTreeMismatch(
            f"data.version '{data_version}' was built from another movement frame of {current['data_tree']} "
            f"(frame {stamp['frame_id']}, now {current['frame_id']}: the competition files changed and `prepare "
            f"--force` rebuilt the frame), so its cached blocks describe rows that are no longer there. Give the new "
            f"frame its own data.version."
        )
    if not stamp.get("frame_id") and current["frame_id"] and write:
        stamp["frame_id"] = current["frame_id"]
        stamp_file.write_text(json.dumps(stamp, indent=1), encoding="utf-8")
    return stamp


def _cache_path(spec: FeatureSpec, digest: str, data_version: str, fold: str | None) -> paths.Path:
    """Where the block is cached; the directory is made when a file is written (`cache_status` and an uncached build
    leave the cache untouched)."""
    d = paths.feature_cache_dir(data_version, fold if spec.uses_target else None)
    return d / f"{spec.name}__{digest}.parquet"


def _validate(spec: FeatureSpec, out: object, dep_index: pd.Index) -> pd.DataFrame:
    if not isinstance(out, pd.DataFrame):
        raise TypeError(f"block '{spec.name}' returned {type(out).__name__}, expected DataFrame")
    got, want = list(out.columns), list(spec.outputs)
    if got != want:
        missing, extra = sorted(set(want) - set(got)), sorted(set(got) - set(want))
        raise ValueError(
            f"block '{spec.name}' column mismatch.\n"
            f"  declared: {want}\n  returned: {got}\n"
            f"  missing: {missing}  unexpected: {extra}"
        )
    if out.index.has_duplicates:
        raise ValueError(f"block '{spec.name}' returned a duplicated index")
    if not out.index.isin(dep_index).all():
        raise ValueError(f"block '{spec.name}' returned rows outside the departure index")
    return out.reindex(dep_index)


def _read_cached(path: paths.Path, dep_index: pd.Index, data_version: str) -> pd.DataFrame:
    """A cached block, refused when its rows are not this movement frame's departures.

    A block is cached over the departures of the frame that computed it, and the cache directory is keyed by
    ``data.version`` only. A frame with other rows under the same version -- another data tree (``PRC_DATA_DIR``)
    holding a new ranking set -- would otherwise read the old file and reindex it into NaN features for every new
    row without a word. The same rows in another order are reindexed as before."""
    out = pd.read_parquet(path)
    if out.index.equals(dep_index):
        return out
    if len(out) == len(dep_index) and out.index.isin(dep_index).all():
        return out.reindex(dep_index)
    raise ValueError(
        f"cached block {path.name} holds {len(out):,} rows that are not this movement frame's {len(dep_index):,} "
        f"departures: it was built from another data tree under data.version '{data_version}'. Give this data tree "
        f"its own data.version (e.g. -o data.version=<new>) so its features are computed, not read from that cache."
    )


def compute_block(
    spec: FeatureSpec,
    mvt: pd.DataFrame,
    ctx: FeatureContext,
) -> pd.DataFrame:
    """Run one block and validate its contract. No caching."""
    out = spec.fn(mvt, ctx)
    return _validate(spec, out, ctx.dep_index)


def build_matrix(
    specs: list[FeatureSpec],
    mvt: pd.DataFrame,
    dep_index: pd.Index,
    *,
    data_version: str,
    params: dict[str, dict] | None = None,
    fold: str | None = None,
    fit_mask: pd.Series | None = None,
    target: pd.Series | None = None,
    cfg=None,
    use_cache: bool = True,
    verbose: bool = True,
) -> FeatureMatrix:
    """Compute (or load) every block in order and concatenate into one matrix.

    ``fold``/``fit_mask``/``target`` only matter for blocks with
    ``uses_target=True``: those are recomputed inside each CV fold from training
    rows only, and cached under a fold-scoped directory.  Everything else is
    fold-independent and computed once for the whole corpus.  That split is the
    entire leakage story for target encodings.
    """
    if use_cache:
        check_tree_stamp(data_version)
    missing = {s.name: [str(p) for p in s.missing_external()] for s in specs if s.missing_external()}
    if missing:
        files = [p for ps in missing.values() for p in ps]
        sources = ((paths.WEATHER, "fetch-weather"), (paths.GEOMETRY, "fetch-geometry"), (paths.PRU, "fetch-pru"),
                   (paths.ADSB_SURFACE, "adsb-surface"))
        commands = [
            cmd for root, cmd in sources if any(f.startswith(str(root)) for f in files)
        ] or [cmd for _, cmd in sources]
        raise FileNotFoundError(
            f"external inputs missing for blocks {sorted(missing)}: {missing}\n"
            "download them with " + " / ".join(f"`python -m prc {c}`" for c in commands)
        )
    params = params or {}
    outputs: dict[str, pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    seconds: dict[str, float] = {}
    block_columns: dict[str, list[str]] = {}
    cat_cols: list[str] = []

    for spec in specs:
        p = params.get(spec.name, {})
        digest = block_hash(spec, data_version, p, hashes)
        hashes[spec.name] = digest
        path = _cache_path(spec, digest, data_version, fold)

        t0 = time.perf_counter()
        if use_cache and path.exists():
            out = _read_cached(path, dep_index, data_version)
            source = "cache"
        else:
            if spec.uses_target and (fit_mask is None or target is None):
                raise ValueError(f"block '{spec.name}' uses the target but no fit_mask/target was given")
            ctx = FeatureContext(
                dep_index=dep_index,
                params=p,
                deps={d: outputs[d] for d in spec.depends_on},
                fit_mask=fit_mask if spec.uses_target else None,
                target=target if spec.uses_target else None,
                cfg=cfg,
            )
            out = compute_block(spec, mvt, ctx)
            if use_cache:
                path.parent.mkdir(parents=True, exist_ok=True)
                out.to_parquet(path, index=True)
            source = "computed"
        dt = time.perf_counter() - t0
        seconds[spec.name] = dt

        outputs[spec.name] = out
        block_columns[spec.name] = list(out.columns)
        cat_cols.extend(spec.categorical)
        if verbose:
            print(f"  [{source:>8}] {spec.name:<24} {out.shape[1]:>3} cols  {dt:6.2f}s")

    X = pd.concat([outputs[s.name] for s in specs], axis=1)
    if X.columns.has_duplicates:
        dupes = sorted(X.columns[X.columns.duplicated()].unique())
        raise ValueError(f"duplicate feature column names across blocks: {dupes}")
    drop = list(getattr(getattr(cfg, "features", None), "drop_columns", None) or [])
    if drop:
        if unknown := sorted(set(drop) - set(X.columns)):
            raise ValueError(f"features.drop_columns names columns no selected block outputs: {unknown}")
        X = X.drop(columns=drop)
        cat_cols = [c for c in cat_cols if c not in drop]
        block_columns = {b: [c for c in cols if c not in drop] for b, cols in block_columns.items()}
        if verbose:
            print(f"  [dropped ] {', '.join(drop)}")
    return FeatureMatrix(X, cat_cols, block_columns, hashes, seconds)


def cache_status(
    specs: list[FeatureSpec],
    *,
    data_version: str,
    params: dict[str, dict] | None = None,
    fold: str | None = None,
) -> pd.DataFrame:
    """One row per block: outputs, dependencies, and whether its cache is warm."""
    params = params or {}
    hashes: dict[str, str] = {}
    rows = []
    for spec in specs:
        p = params.get(spec.name, {})
        digest = block_hash(spec, data_version, p, hashes)
        hashes[spec.name] = digest
        path = _cache_path(spec, digest, data_version, fold)
        rows.append(
            {
                "block": spec.name,
                "group": spec.group,
                "n_outputs": len(spec.outputs),
                "uses_target": spec.uses_target,
                "depends_on": ",".join(spec.depends_on),
                "hash": digest,
                "cached": path.exists(),
                "external": ("missing" if spec.missing_external() else "ok") if spec.external else "",
                "doc": spec.doc,
            }
        )
    return pd.DataFrame(rows)
