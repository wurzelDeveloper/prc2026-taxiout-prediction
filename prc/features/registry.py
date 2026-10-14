"""The feature registry: `@feature`-decorated blocks, and how they resolve.

A *block* is one function producing a small group of related columns.  Blocks --
not individual columns -- are the unit of selection, caching, ablation and
importance, because columns inside a block are usually correlated and mask each
other when measured individually.

The contract, and why it looks like this
----------------------------------------
A block function receives the **whole** movement frame: departures *and*
arrivals, training *and* ranking, time-sorted.  It is not a row-wise transform.
That is deliberate -- most of the signal in taxi-out time is cross-row context
(runway queues, ground occupancy, stand turnaround), and a January 2026
departure can only see its context if the January 2026 arrivals are in the same
frame.  The frame it receives has the target and off-block time masked on every
departure (see `prc.data`), so no amount of creativity can leak the answer.

Blocks return a DataFrame indexed by ``mvt_id``, carrying exactly the columns
declared in ``outputs``.  `prc.features.build` enforces both.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from prc.config import ExperimentConfig

FeatureFn = Callable[[pd.DataFrame, "FeatureContext"], pd.DataFrame]


@dataclass
class FeatureContext:
    """Everything a block may use besides the movement frame itself."""

    dep_index: pd.Index
    """Canonical departure index. Every block reindexes its result onto this."""

    params: dict[str, Any] = field(default_factory=dict)
    """Per-block parameters from the experiment config (`features.params.<block>`)."""

    deps: dict[str, pd.DataFrame] = field(default_factory=dict)
    """Outputs of the blocks named in ``depends_on``, already computed."""

    fit_mask: pd.Series | None = None
    """Rows whose target may be used for statistics. Only set for target-using blocks."""

    target: pd.Series | None = None
    """Taxi-out seconds on ``fit_mask`` rows. Only set if ``spec.uses_target``."""

    cfg: "ExperimentConfig | None" = None

    def dep(self, name: str) -> pd.DataFrame:
        try:
            return self.deps[name]
        except KeyError:
            raise KeyError(f"block '{name}' was not declared in depends_on") from None


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    fn: FeatureFn
    group: str
    outputs: tuple[str, ...]
    depends_on: tuple[str, ...] = ()
    categorical: tuple[str, ...] = ()
    uses_target: bool = False
    version: int = 1
    doc: str = ""
    external: tuple[Path, ...] = ()
    """Files outside the movement frame that the block reads (the weather tables).
    Their content digest is part of the cache key, so re-downloaded or re-parsed
    data invalidates the block instead of being served stale from cache."""

    def __post_init__(self) -> None:
        unknown = set(self.categorical) - set(self.outputs)
        if unknown:
            raise ValueError(f"block '{self.name}': categorical cols not in outputs: {sorted(unknown)}")

    def missing_external(self) -> list[Path]:
        return [p for p in self.external if not p.exists()]


_REGISTRY: dict[str, FeatureSpec] = {}


def feature(
    *,
    group: str,
    outputs: tuple[str, ...] | list[str],
    depends_on: tuple[str, ...] | list[str] = (),
    categorical: tuple[str, ...] | list[str] = (),
    uses_target: bool = False,
    version: int = 1,
    name: str | None = None,
    doc: str = "",
    external: tuple[Path, ...] | list[Path] = (),
) -> Callable[[FeatureFn], FeatureFn]:
    """Register a feature block. The block name defaults to the function name."""

    def decorator(fn: FeatureFn) -> FeatureFn:
        block = name or fn.__name__
        if block in _REGISTRY:
            raise ValueError(f"duplicate feature block '{block}'")
        spec = FeatureSpec(
            name=block,
            fn=fn,
            group=group,
            outputs=tuple(outputs),
            depends_on=tuple(depends_on),
            categorical=tuple(categorical),
            uses_target=uses_target,
            version=version,
            doc=doc or (fn.__doc__ or "").strip().split("\n")[0],
            external=tuple(Path(p) for p in external),
        )
        _REGISTRY[block] = spec
        return fn

    return decorator


def all_specs() -> dict[str, FeatureSpec]:
    _load_feature_modules()
    return dict(_REGISTRY)


def get_spec(name: str) -> FeatureSpec:
    specs = all_specs()
    if name not in specs:
        raise KeyError(f"unknown feature block '{name}'. Known: {sorted(specs)}")
    return specs[name]


def groups() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for spec in all_specs().values():
        out.setdefault(spec.group, []).append(spec.name)
    return {g: sorted(v) for g, v in sorted(out.items())}


_loaded = False


def _load_feature_modules() -> None:
    """Import every module in `prc.features` so decorators run.

    This is what makes adding a feature a one-file operation: drop a decorated
    function into any module here and it becomes selectable by name -- there is
    no central list to remember to edit.
    """
    global _loaded
    if _loaded:
        return
    import importlib
    import pkgutil

    import prc.features as pkg

    _loaded = True  # set first: the imports below re-enter via all_specs()
    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.name.startswith("_") or mod.name in {"registry", "build"}:
            continue
        importlib.import_module(f"{pkg.__name__}.{mod.name}")


def resolve(include: list[str], exclude: list[str] | None = None) -> list[FeatureSpec]:
    """Expand a config selection into an ordered, dependency-closed block list.

    ``include`` accepts block names and ``group:<name>`` references.  Transitive
    ``depends_on`` blocks are pulled in automatically.  Excluding a block that
    another included block depends on is an error rather than a silent drop --
    silently dropping a dependency would produce a model that trains fine and is
    quietly wrong.
    """
    specs = all_specs()
    by_group = groups()
    exclude = list(exclude or [])

    def expand(names: list[str]) -> list[str]:
        out: list[str] = []
        for n in names:
            if n.startswith("group:"):
                g = n.split(":", 1)[1]
                if g not in by_group:
                    raise KeyError(f"unknown feature group '{g}'. Known: {sorted(by_group)}")
                out.extend(by_group[g])
            elif n in specs:
                out.append(n)
            else:
                raise KeyError(f"unknown feature block '{n}'. Known: {sorted(specs)}")
        return out

    wanted = dict.fromkeys(expand(include))
    excluded = set(expand(exclude))

    # Close over dependencies.
    stack = list(wanted)
    while stack:
        spec = specs[stack.pop()]
        for dep in spec.depends_on:
            if dep not in specs:
                raise KeyError(f"block '{spec.name}' depends on unknown block '{dep}'")
            if dep not in wanted:
                wanted[dep] = None
                stack.append(dep)

    for name in list(wanted):
        if name in excluded:
            needed_by = [s.name for s in specs.values() if name in s.depends_on and s.name in wanted and s.name not in excluded]
            if needed_by:
                raise ValueError(f"cannot exclude '{name}': required by {needed_by}")
            del wanted[name]

    return _toposort([specs[n] for n in wanted])


def dependents(name: str, within: list[str] | None = None) -> list[str]:
    """Every block that transitively depends on ``name``, itself included.

    Ablation needs this: a block cannot be removed while something downstream
    still requires it, so "what is this block worth" really means "what are this
    block and everything built on it worth".
    """
    specs = all_specs()
    scope = set(within) if within is not None else set(specs)
    closure = {name}
    changed = True
    while changed:
        changed = False
        for candidate in scope:
            if candidate in closure:
                continue
            if closure & set(specs[candidate].depends_on):
                closure.add(candidate)
                changed = True
    return sorted(closure)


def _toposort(specs: list[FeatureSpec]) -> list[FeatureSpec]:
    by_name = {s.name: s for s in specs}
    ordered: list[FeatureSpec] = []
    state: dict[str, int] = {}  # 0 = visiting, 1 = done

    def visit(name: str, trail: tuple[str, ...]) -> None:
        if state.get(name) == 1:
            return
        if state.get(name) == 0:
            raise ValueError(f"circular feature dependency: {' -> '.join((*trail, name))}")
        state[name] = 0
        for dep in by_name[name].depends_on:
            if dep in by_name:
                visit(dep, (*trail, name))
        state[name] = 1
        ordered.append(by_name[name])

    for name in sorted(by_name):
        visit(name, ())
    return ordered
