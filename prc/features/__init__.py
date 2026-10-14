"""Feature blocks.

To add a feature: drop a `@feature`-decorated function into any module in this
package (or a new module here). It is imported and registered automatically, and
becomes selectable by name -- or via `group:<its group>` -- in an experiment
config. There is no central list to update.
"""

from prc.features.registry import (
    FeatureContext,
    FeatureSpec,
    all_specs,
    feature,
    get_spec,
    groups,
    resolve,
)

__all__ = [
    "FeatureContext",
    "FeatureSpec",
    "all_specs",
    "feature",
    "get_spec",
    "groups",
    "resolve",
]
