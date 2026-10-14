"""The block cache key: stable for ordinary blocks, content-sensitive for external files.

Blocks that read downloaded data (the weather tables) cannot rely on their
source hash alone -- re-fetching or re-parsing changes what they compute
without changing a line of their code. Folding the files' digest into the key
fixes that, but only if it leaves every *other* block's key untouched: a key
that moves turns every warm cache in `cache/features/` cold without a word.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from prc.features.build import _file_digest, _source_hash, block_hash, build_matrix
from prc.features.registry import FeatureSpec


def _fn(mvt, ctx):
    return pd.DataFrame({"x": 0.0}, index=ctx.dep_index)


def _spec(**kw) -> FeatureSpec:
    return FeatureSpec(name="probe", fn=_fn, group="probe", outputs=("x",), **kw)


def test_blocks_without_external_inputs_keep_the_legacy_hash():
    spec = _spec(depends_on=("core_keys",))
    legacy = json.dumps(
        {
            "name": "probe",
            "version": 1,
            "src": _source_hash(spec),
            "outputs": ["x"],
            "data_version": "v1",
            "params": {"a": 1},
            "parents": {"core_keys": "abc"},
        },
        sort_keys=True,
        default=str,
    )
    want = hashlib.sha1(legacy.encode("utf-8")).hexdigest()[:12]
    assert block_hash(spec, "v1", {"a": 1}, {"core_keys": "abc"}) == want


def test_rewriting_an_external_file_changes_the_hash(tmp_path):
    f = tmp_path / "wx.parquet"
    f.write_bytes(b"first")
    spec = _spec(external=(f,))
    before = block_hash(spec, "v1", {}, {})
    f.write_bytes(b"second, and a different size")
    assert block_hash(spec, "v1", {}, {}) != before


def test_missing_external_file_hashes_without_raising(tmp_path):
    """`features --list` must still show the block (as missing), not crash."""
    f = tmp_path / "absent.parquet"
    assert _file_digest(f) == "missing"
    spec = _spec(external=(f,))
    assert spec.missing_external() == [f]
    block_hash(spec, "v1", {}, {})


def test_external_digest_ignores_the_directory(tmp_path):
    """Only the file name enters the key, so a checkout at another path shares caches."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "wx.parquet").write_bytes(b"same")
    (b / "wx.parquet").write_bytes(b"same")
    assert block_hash(_spec(external=(a / "wx.parquet",)), "v1", {}, {}) == block_hash(
        _spec(external=(b / "wx.parquet",)), "v1", {}, {}
    )


def test_build_fails_fast_when_an_external_input_is_missing(tmp_path):
    """Before minutes of other blocks are built, not after."""
    spec = _spec(external=(tmp_path / "absent.parquet",))
    with pytest.raises(FileNotFoundError, match="fetch-weather"):
        build_matrix([spec], pd.DataFrame(), pd.Index([]), data_version="t", use_cache=False, verbose=False)


def _fn2(mvt, ctx):
    return pd.DataFrame({"x": 0.0, "y": 1.0}, index=ctx.dep_index)


def test_drop_columns_leaves_the_matrix_and_the_block_lists():
    from types import SimpleNamespace

    spec = FeatureSpec(name="probe2", fn=_fn2, group="probe", outputs=("x", "y"))
    idx = pd.Index([1, 2, 3])
    cfg = SimpleNamespace(features=SimpleNamespace(drop_columns=["y"]))
    fm = build_matrix([spec], pd.DataFrame(index=idx), idx, data_version="t", cfg=cfg, use_cache=False, verbose=False)
    assert list(fm.X.columns) == ["x"]
    assert fm.block_columns == {"probe2": ["x"]}
    cfg = SimpleNamespace(features=SimpleNamespace(drop_columns=["nope"]))
    with pytest.raises(ValueError, match="nope"):
        build_matrix([spec], pd.DataFrame(index=idx), idx, data_version="t", cfg=cfg, use_cache=False, verbose=False)
