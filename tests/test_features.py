"""Contract tests for the feature registry and every registered block.

The leakage tests are the point of this file. A model that scores well because
the off-block time found its way back into a feature looks exactly like a model
that scores well, right up until the leaderboard disagrees -- so the properties
that make that impossible are asserted rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prc import data, paths
from prc.features import all_specs, resolve
from prc.features.build import build_matrix
from prc.features.registry import _toposort, dependents

pytestmark = pytest.mark.skipif(
    not paths.MOVEMENTS_FILE.exists(),
    reason="run `python -m prc prepare` first",
)

SAMPLE_ROWS = 120_000


@pytest.fixture(scope="module")
def frame():
    """A contiguous slice of the *training* period.

    Contiguous rather than random: the congestion and turnaround blocks are
    cross-row, so a random sample would leave them full of holes and the values
    would not be the ones production computes.

    Training rather than the tail of the corpus: the latest rows are all 2026
    ranking rows, which carry no targets at all -- a fixture that quietly has an
    empty target makes every leakage assertion below vacuously true.
    """
    mvt = data.load_movements()
    mvt = mvt.loc[mvt["split"].eq("train")].sort_values("MVT_TIME_UTC_mvt", kind="stable").tail(SAMPLE_ROWS)
    y = data.load_target().reindex(mvt.index).dropna()
    assert len(y) > SAMPLE_ROWS // 4, "fixture has too few targets to test anything"
    return mvt, y


def _buildable_specs():
    """Every registered block whose external inputs exist -- and the names of those dropped.

    Weather blocks read downloaded files. A checkout without them must still run
    the contract and leakage suite over everything else, and say what it left out.
    """
    specs = resolve([f"group:{g}" for g in {s.group for s in all_specs().values()}])
    unavailable = [s.name for s in specs if s.missing_external()]
    drop = set().union(*(dependents(n) for n in unavailable))
    return [s for s in specs if s.name not in drop], sorted(drop)


def test_blocks_with_external_inputs_are_present():
    _, dropped = _buildable_specs()
    if dropped:
        pytest.skip(f"not built, external inputs missing (run `python -m prc fetch-weather` / `fetch-geometry`): {dropped}")


@pytest.fixture(scope="module")
def built(frame):
    mvt, y = frame
    specs, _ = _buildable_specs()
    dep_idx = data.dep_index(mvt)
    fit_mask = pd.Series(False, index=mvt.index)
    fit_mask.loc[y.index] = True
    fm = build_matrix(
        specs, mvt, dep_idx,
        data_version="test", fold="test",
        fit_mask=fit_mask, target=y.reindex(dep_idx),
        use_cache=False, verbose=False,
    )
    return mvt, y, fm


# --- registry ---------------------------------------------------------------

def test_every_block_declares_a_known_group():
    for spec in all_specs().values():
        assert spec.group, f"{spec.name} has no group"
        assert spec.outputs, f"{spec.name} declares no outputs"


def test_dependencies_exist_and_are_acyclic():
    specs = all_specs()
    for spec in specs.values():
        for dep in spec.depends_on:
            assert dep in specs, f"{spec.name} depends on unknown block '{dep}'"
    _toposort(list(specs.values()))


def test_resolve_pulls_in_transitive_dependencies():
    names = [s.name for s in resolve(["load_normalised"])]
    assert {"core_keys", "calendar", "airport_load", "runway_queue"} <= set(names)
    assert names.index("core_keys") < names.index("load_normalised")


def test_excluding_a_required_block_is_an_error():
    with pytest.raises(ValueError, match="required by"):
        resolve(["load_normalised"], ["airport_load"])


def test_unknown_names_are_rejected():
    with pytest.raises(KeyError):
        resolve(["no_such_block"])
    with pytest.raises(KeyError):
        resolve(["group:no_such_group"])


def test_output_names_are_unique_across_blocks():
    seen: dict[str, str] = {}
    for spec in all_specs().values():
        for col in spec.outputs:
            assert col not in seen, f"'{col}' declared by both {seen[col]} and {spec.name}"
            seen[col] = spec.name


# --- block contract ---------------------------------------------------------

def test_blocks_return_declared_columns_on_the_departure_index(built):
    mvt, _, fm = built
    dep_idx = data.dep_index(mvt)
    for block, columns in fm.block_columns.items():
        spec = all_specs()[block]
        assert columns == list(spec.outputs), f"{block} returned {columns}, declared {list(spec.outputs)}"
    assert fm.X.index.equals(dep_idx)
    assert not fm.X.index.has_duplicates


def test_categorical_declarations_match_the_built_matrix(built):
    _, _, fm = built
    for col in fm.cat_cols:
        assert col in fm.X.columns
        assert isinstance(fm.X[col].dtype, pd.CategoricalDtype), f"{col} declared categorical but is {fm.X[col].dtype}"


# --- leakage ----------------------------------------------------------------

def test_canonical_frame_masks_the_target_on_every_departure():
    mvt = data.load_movements(columns=["is_dep", "TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt"])
    dep = mvt.loc[mvt["is_dep"]]
    assert dep["TAXITIME_SEC_mvt"].isna().all()
    assert dep["BLOCK_TIME_UTC_mvt"].isna().all()


def test_arrival_block_times_are_preserved():
    """Arrival-side data must survive: it is what makes turnaround features legal."""
    mvt = data.load_movements(columns=["is_dep", "split", "TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt"])
    arr = mvt.loc[~mvt["is_dep"] & mvt["split"].eq("rank")]
    assert arr["TAXITIME_SEC_mvt"].notna().mean() > 0.99
    assert arr["BLOCK_TIME_UTC_mvt"].notna().mean() > 0.99


def test_no_feature_is_a_near_copy_of_the_target(built):
    _, y, fm = built
    train = fm.X.loc[fm.X.index.intersection(y.index)]
    target = y.reindex(train.index).to_numpy(dtype="float64")
    offenders = []
    for col in train.columns:
        if isinstance(train[col].dtype, pd.CategoricalDtype):
            continue
        values = pd.to_numeric(train[col], errors="coerce").to_numpy(dtype="float64")
        ok = np.isfinite(values) & np.isfinite(target)
        if ok.sum() < 1000 or np.std(values[ok]) == 0:
            continue
        r = abs(np.corrcoef(values[ok], target[ok])[0, 1])
        if r > 0.99:
            offenders.append((col, round(float(r), 4)))
    assert not offenders, f"features almost perfectly correlated with the target: {offenders}"


def test_target_using_blocks_are_declared_as_such():
    """A block reading `ctx.target` without `uses_target=True` would be cached
    globally instead of per fold, silently leaking across the validation split."""
    import inspect

    for spec in all_specs().values():
        source = inspect.getsource(spec.fn)
        reads_target = "ctx.target" in source or "ctx.fit_mask" in source
        assert reads_target == spec.uses_target, (
            f"{spec.name}: uses_target={spec.uses_target} but "
            f"{'does' if reads_target else 'does not'} read the target"
        )


def test_target_stats_respect_the_fit_mask(frame):
    """Rows outside the fit mask must not contribute to their own encoding."""
    mvt, y = frame
    dep_idx = data.dep_index(mvt)
    specs = resolve(["stand_runway_target_stats"])

    half = y.index[: len(y) // 2]
    fit_mask = pd.Series(False, index=mvt.index)
    fit_mask.loc[half] = True

    narrow = build_matrix(
        specs, mvt, dep_idx, data_version="test", fold="half",
        fit_mask=fit_mask, target=y.reindex(dep_idx), use_cache=False, verbose=False,
    ).X

    full_mask = pd.Series(False, index=mvt.index)
    full_mask.loc[y.index] = True
    wide = build_matrix(
        specs, mvt, dep_idx, data_version="test", fold="full",
        fit_mask=full_mask, target=y.reindex(dep_idx), use_cache=False, verbose=False,
    ).X

    assert not narrow["taxi_med_hier"].equals(wide["taxi_med_hier"]), (
        "target statistics did not change when the fit mask did -- the block is "
        "ignoring ctx.fit_mask and leaking across folds"
    )


# --- ablation dependency closure --------------------------------------------

def test_dependents_closure_is_transitive():
    from prc.features.registry import dependents

    closure = dependents("core_keys")
    assert "core_keys" in closure
    # load_normalised depends on airport_load, which depends on core_keys.
    assert {"airport_load", "load_normalised", "calendar"} <= set(closure)


def test_dependents_of_a_leaf_is_just_itself():
    from prc.features.registry import dependents

    assert dependents("load_normalised") == ["load_normalised"]


def test_dependents_can_be_scoped_to_the_blocks_in_a_config():
    from prc.features.registry import dependents

    scope = ["core_keys", "nm_deltas", "nm_proxy_context"]
    assert dependents("nm_deltas", scope) == ["nm_deltas", "nm_proxy_context"]
    # calendar is outside the scope, so it must not appear
    assert "calendar" not in dependents("core_keys", scope)


def test_removing_a_block_with_its_dependents_resolves_cleanly():
    """Leave-one-out ablation must produce a valid feature set, not a
    dependency error -- this is what `prc ablate` relies on."""
    from prc.features.registry import dependents

    include = ["group:core", "group:nm_schedule", "group:congestion", "group:temporal",
               "group:stand_runway"]
    in_config = [s.name for s in resolve(include)]
    group = [b for b in dependents("nm_deltas", in_config) if b != "core_keys"]
    remaining = resolve(include, group)
    assert "nm_deltas" not in {s.name for s in remaining}
    assert "nm_proxy_context" not in {s.name for s in remaining}
    assert "core_keys" in {s.name for s in remaining}
