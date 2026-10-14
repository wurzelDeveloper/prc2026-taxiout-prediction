"""Experiment configuration: YAML in, validated frozen dataclasses out.

YAML rather than Python because configs get diffed in git, generated in bulk by
`prc ablate`, and archived verbatim next to each run's metrics. Dataclasses
rather than raw dicts because a typo in a config should fail immediately, not
silently train a model with a default you did not intend.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from prc import paths


@dataclass(frozen=True)
class CleanConfig:
    """Rules that flag training departures by their target value.

    A flag reads the target, so it can never be a feature, and it is only ever
    computed for training rows: a flagged row can leave the *fit*, never the
    *score*. Validation keeps every row (so ``rmse`` stays comparable with the
    leaderboard, which scores uncleaned 2026 data) and a second metric set is
    reported on the unflagged rows.
    """

    max_taxi_sec: float | None = None
    """Flag taxi times above this."""
    min_taxi_sec: float | None = None
    """Flag taxi times below this -- except on a helicopter pad (RUNWAY_mvt H/HEL),
    where a taxi of a few seconds, even negative, is real and recurs in 2026."""
    day_offset_tol_sec: float | None = None
    """Flag taxi times within this many seconds of a whole number of days: a block time on the wrong date."""
    untyped_airports: list[str] = field(default_factory=list)
    """Flag departures with no AIRCRAFT_TYPE_mvt at these airports (LIRF's tail segment)."""
    drop_from_fit: bool = True
    """Remove flagged rows from every fold's training index. Off: flag for the clean metrics only."""

    def __post_init__(self) -> None:
        if self.min_taxi_sec is not None and self.max_taxi_sec is not None and self.min_taxi_sec >= self.max_taxi_sec:
            raise ValueError(f"data.clean: min_taxi_sec ({self.min_taxi_sec}) must be below max_taxi_sec ({self.max_taxi_sec})")
        if self.day_offset_tol_sec is not None and self.day_offset_tol_sec < 0:
            raise ValueError("data.clean.day_offset_tol_sec must not be negative")
        if not isinstance(self.untyped_airports, list) or not all(isinstance(a, str) for a in self.untyped_airports):
            raise ValueError(f"data.clean.untyped_airports must be a list of ICAO codes, got {self.untyped_airports!r}")

    @property
    def is_active(self) -> bool:
        return (
            any(v is not None for v in (self.max_taxi_sec, self.min_taxi_sec, self.day_offset_tol_sec))
            or bool(self.untyped_airports)
        )

    def summary(self) -> str:
        """A short label for index.csv: ``"min30|max7200|day3600|untyped:LIRF"``, or ``""``."""
        parts = []
        if self.min_taxi_sec is not None:
            parts.append(f"min{self.min_taxi_sec:g}")
        if self.max_taxi_sec is not None:
            parts.append(f"max{self.max_taxi_sec:g}")
        if self.day_offset_tol_sec is not None:
            parts.append(f"day{self.day_offset_tol_sec:g}")
        if self.untyped_airports:
            parts.append("untyped:" + ",".join(self.untyped_airports))
        if parts and not self.drop_from_fit:
            parts.append("flag-only")
        return "|".join(parts)


@dataclass(frozen=True)
class DataConfig:
    version: str = "v1"
    airports: str | list[str] = "all"
    sample_frac: float = 1.0
    """Fraction of training rows to keep. Ablation sweeps use this to trade a
    little precision for a lot of wall time; leave at 1.0 for scoring runs."""
    clean: CleanConfig = field(default_factory=CleanConfig)
    """Training-row cleaning rules. Under ``data`` so that `split_hash` covers
    them: dropping rows from the fit changes the fold-dependent target statistics."""


@dataclass(frozen=True)
class FeatureConfig:
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    params: dict[str, dict] = field(default_factory=dict)
    drop_columns: list[str] = field(default_factory=list)
    """Feature columns removed from the model matrix after every block is built. The
    blocks still compute them, so dependents and caches are unaffected; a name that no
    selected block outputs raises. Left out of `hash()` while empty, so existing run
    names are unchanged."""


@dataclass(frozen=True)
class TargetConfig:
    column: str = "TAXITIME_SEC_mvt"
    transform: str = "none"  # none | log1p | sqrt
    clip: tuple[float, float] | None = None
    residual_of: str | None = None  # e.g. "aobt3_proxy"

    def __post_init__(self) -> None:
        if self.transform not in {"none", "log1p", "sqrt"}:
            raise ValueError(f"unknown target.transform '{self.transform}'")
        if self.residual_of and self.transform != "none":
            # Residuals are signed; log1p and sqrt of a negative are NaN, which
            # would train on an all-NaN target and fail far downstream.
            raise ValueError(
                f"target.transform must be 'none' when residual_of is set "
                f"(got '{self.transform}' with residual_of='{self.residual_of}')"
            )


@dataclass(frozen=True)
class ValidationConfig:
    scheme: str = "seasonal_holdout"
    holdout_months: list[str] = field(default_factory=lambda: ["2025-01", "2025-07"])
    n_folds: int = 5

    def __post_init__(self) -> None:
        known = {"seasonal_holdout", "forward_holdout", "strict_forward", "group_kfold_day", "month_forward"}
        if self.scheme not in known:
            raise ValueError(f"unknown validation.scheme '{self.scheme}'. Known: {sorted(known)}")


@dataclass(frozen=True)
class MixtureConfig:
    """The regime mixture (`prc.models.mixture`): a classifier decides whether a
    departure's target is a taxi time or one of several candidate block-time
    deltas already present as feature columns, and the prediction is the
    probability-weighted mixture of the candidates and the regressor.

    Class 0 is "taxi" (the regressor); class k is ``bases[k - 1]``. A base is a
    candidate on a row when it is finite, at least ``base_min_sec``, and either
    the row is marked by ``universe_column`` or the base is at least
    ``min_base_sec``. Rows with at least one candidate form the classifier's
    universe; every other row is the regressor's alone. A training row in the
    universe is labelled with the nearest candidate when the target is within
    ``max(tol_sec, tol_frac * base)`` of it, otherwise "taxi".
    """

    enabled: bool = False
    bases: list[str] = field(default_factory=list)
    universe_column: str | None = None
    """A 0/1 feature column; its rows are always in the universe (e.g. ``stale_suspect``)."""
    min_base_sec: float | None = 3600.0
    """A base this large puts any row in the universe. ``null``: the universe column alone decides."""
    base_min_sec: float = 0.0
    """A base below this is never a candidate (a negative schedule delta is another leg's schedule)."""
    tol_sec: float = 600.0
    tol_frac: float = 0.0
    day_offset: bool = False
    """Add the regime "a whole day plus a taxi": a block time recorded on the wrong
    date puts the target at 86,400 s plus an ordinary taxi. Its base is a day plus the
    regressor's own answer, so it is not a feature column; a universe row is labelled
    with it when the target is within ``day_tol_sec`` of a whole number of days and no
    candidate base is nearer. Only universe rows can carry it."""
    day_tol_sec: float = 3600.0
    exclude_stale_from_regressor: bool = True
    """Fit the regressor without the rows labelled as a base regime, so it learns taxi times only."""
    regressor_max_target_sec: float | None = None
    """Also fit the regressor without any row whose target is above this, universe or not:
    a multi-hour target that matches no candidate (a block time on the wrong date, a
    stale block outside the universe) is not a taxi time the regressor could learn, and
    left in the fit it drags every neighbouring row's prediction up by hours. The rows
    stay scored; only the fit loses them."""
    classifier_params: dict[str, Any] = field(default_factory=dict)
    """CatBoostClassifier parameters over `prc.models.mixture.CLASSIFIER_DEFAULTS`."""
    classifier_weight_power: float = 0.0
    """Row weight of the classifier's fit: the row's largest candidate (a day for the
    day-offset regime) to this power, normalised to a mean of one. The squared-error
    cost of a wrong regime grows with the candidate, so 1 or 2 makes the classifier
    spend its capacity where the score is decided. The weight depends on features
    only, never on the label, so the probabilities stay calibrated."""
    classifier_seeds: int = 1
    """How many classifiers (seeds ``seed``, ``seed + 1000``, ...) are fitted on the same universe; their
    probabilities are averaged. The hedge a row pays is p(1 - p)(base - taxi)^2, so the classifier's seed
    variance lands on the score directly (findings 9.10-9.11); averaging it costs one classifier fit per seed
    and leaves the regressor alone. Left out of the config hash while 1."""
    classifier_eval: bool = True
    """Early-stop the classifier on the fold's universe rows (the same mild optimism as the regressor's)."""
    regressor_anchor: str | None = None
    """A feature column the regressor is anchored on: it is fitted on ``target - anchor`` and predicts
    ``anchor + correction``. ``aobt3_proxy`` (takeoff minus NM's actual off-block) is the natural anchor: trees
    approximate the near-identity between the proxy and the target with steps, a residual does not need to.
    The anchor is clipped to `anchor_clip`; where it is missing the per-airport median of the fit rows' target
    stands in. Left out of the config hash while unset."""
    anchor_clip: list[float] = field(default_factory=lambda: [0.0, 7200.0])
    """[lo, hi] the anchor is clipped to (a proxy of several hours is a mis-join, not a taxi)."""
    anchor_airports: list[str] | None = None
    """Anchor only these airports' rows (``null``: all). The proxy anchor helps where long proxies are real
    (EHAM's snow days) and hurts where they are artefacts (EGLL, LEBL, LTFM)."""
    window_column: str | None = None
    """The NM join window (findings 10.3): a feature column ``w`` (``lobt_delta_full``, takeoff minus LOBT) such
    that every target lies in [w - `window_half_width`, w + `window_half_width`] where ``w`` is finite. With it
    set, a candidate base outside the window is invalid, the regressor keeps the rows it covers even above
    `regressor_max_target_sec` (their long targets are real: the join puts the block within the window), and the
    regressor's and the mixture's predictions are clipped to the window. Left out of the config hash while unset."""
    window_half_width: float = 3606.0
    calibration: str = "none"
    """``none`` or ``isotonic``: re-map the classifier's stale probability with an isotonic
    regression fitted on nested out-of-fold probabilities of the fit universe, one curve per
    band of the row's largest candidate delta (`calibration_delta_bands`), rows weighted like
    the classifier's. The mixture weight is the probability, and the raw classifier is
    over-confident in both directions (findings 7.1); a plain re-mapping ignoring the delta
    made things worse, hence the bands."""
    calibration_delta_bands: list[float] = field(default_factory=lambda: [7200.0, 14_400.0, 28_800.0])
    """Band edges (seconds) on the row's largest valid candidate; bands with fewer than
    `calibration_min_rows` fit rows are pooled into the band below."""
    calibration_folds: int = 3
    """Nested folds (by takeoff day) that produce the out-of-fold probabilities the curves are fitted on."""
    calibration_min_rows: int = 200

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        if self.calibration not in ("none", "isotonic"):
            raise ValueError(f"model.mixture.calibration must be 'none' or 'isotonic', got {self.calibration!r}")
        if self.calibration_folds < 2:
            raise ValueError("model.mixture.calibration_folds must be at least 2")
        edges = list(self.calibration_delta_bands)
        if any(e <= 0 for e in edges) or edges != sorted(edges) or len(set(edges)) != len(edges):
            raise ValueError("model.mixture.calibration_delta_bands must be distinct positive seconds in increasing order")
        if self.calibration_min_rows < 1:
            raise ValueError("model.mixture.calibration_min_rows must be at least 1")
        if not isinstance(self.bases, list) or not self.bases or len(set(self.bases)) != len(self.bases):
            raise ValueError(f"model.mixture.bases must be a non-empty list of distinct feature columns, got {self.bases!r}")
        if self.universe_column is None and self.min_base_sec is None:
            raise ValueError("model.mixture needs universe_column or min_base_sec (or both) to define the universe")
        if self.tol_sec < 0 or not (0 <= self.tol_frac < 1) or self.base_min_sec < 0:
            raise ValueError("model.mixture: tol_sec and base_min_sec must be >= 0 and tol_frac in [0, 1)")
        if self.regressor_max_target_sec is not None and self.regressor_max_target_sec <= 0:
            raise ValueError("model.mixture.regressor_max_target_sec must be positive")
        if self.day_tol_sec < 0:
            raise ValueError("model.mixture.day_tol_sec must not be negative")
        if self.classifier_weight_power < 0:
            raise ValueError("model.mixture.classifier_weight_power must not be negative")
        if not isinstance(self.classifier_seeds, int) or self.classifier_seeds < 1:
            raise ValueError("model.mixture.classifier_seeds must be a positive integer")
        clip = list(self.anchor_clip)
        if len(clip) != 2 or not clip[0] < clip[1]:
            raise ValueError(f"model.mixture.anchor_clip must be [lo, hi] with lo < hi, got {self.anchor_clip!r}")


@dataclass(frozen=True)
class ModelConfig:
    type: str = "catboost"
    per_airport: bool = False
    refit_on_all: bool = False
    """After the folds are scored, fit one more model on every training row
    (holdout months included) and take the ranking predictions from it instead
    of the fold mean. There is no early-stopping set then, so a GBDT runs a fixed
    number of rounds: the fold models' best iteration, scaled by the row ratio
    (all rows / a fold's rows). Only a submission run needs it; ``--no-test``,
    seed and ablation runs skip the refit, so their scores are unaffected."""
    params: dict[str, Any] = field(default_factory=dict)
    mixture: MixtureConfig = field(default_factory=MixtureConfig)
    month_weights: dict[int, float] = field(default_factory=dict)
    """Row weight of every fit by the takeoff's calendar month (UTC), e.g. ``{1: 2.0, 7: 2.0, 12: 1.5}``; a month
    left out weighs 1. Normalised to a mean of one over the fit rows. The scored months are January and July 2026,
    and the fit otherwise weighs October like July. Applies to the regressor (and the mixture's classifier, times
    its delta weights), in the folds and in the refit; the eval set stays unweighted. Left out of the hash while
    empty."""
    eval_max_target_sec: float | None = None
    """Early stopping of a plain GBDT (no mixture) reads only the holdout rows whose raw target is at or below this
    many seconds; the metrics still score every holdout row. With the fit cleaned of the same targets
    (``data.clean.max_taxi_sec``), the stopping rule then sees the population the model is fitted on rather than the
    multi-hour stale and wrong-date records it cannot predict (the mixture's regressor already stops that way). The
    refit's round count follows the fold's best iteration as usual. Left out of the hash while None."""
    block_dropout: dict[str, float] = field(default_factory=dict)
    """``{block: share}``: every fit (folds and refit) sees the named blocks' columns blanked (NaN) on that share of its
    rows' (airport, day) groups, picked by a hash of (block, airport, day) -- deterministic, no seed. The early-stopping
    set, the metrics and the ranking rows keep every value. For a feed that may be missing at inference (the ADS-B
    surface picture, critique F6): NaN then means "no feed" at every airport, not only in the few airport-periods of
    2025 without reception, and the outage behaviour is learned. Float columns only. Left out of the hash while
    empty."""

    def __post_init__(self) -> None:
        bad = {k: v for k, v in dict(self.month_weights).items() if not (isinstance(k, int) and 1 <= k <= 12) or not float(v) > 0}
        if bad:
            raise ValueError(f"model.month_weights needs months 1-12 with positive weights, got {bad}")
        if self.eval_max_target_sec is not None and not float(self.eval_max_target_sec) > 0:
            raise ValueError(f"model.eval_max_target_sec must be positive, got {self.eval_max_target_sec}")
        bad = {k: v for k, v in dict(self.block_dropout).items() if not isinstance(k, str) or not 0.0 < float(v) < 1.0}
        if bad:
            raise ValueError(f"model.block_dropout needs block names with shares in (0, 1), got {bad}")


@dataclass(frozen=True)
class PermutationConfig:
    enabled: bool = True
    granularity: str = "block"  # block | column
    repeats: int = 3
    sample: int = 200_000


@dataclass(frozen=True)
class EvaluationConfig:
    primary_metric: str = "rmse"
    breakdowns: list[str] = field(default_factory=lambda: ["apt", "month"])
    permutation: PermutationConfig = field(default_factory=PermutationConfig)
    shap: bool = False


@dataclass(frozen=True)
class GeometryConfig:
    stands_file: str | None = None
    """Hand-maintained stand positions (CSV: apt, stand_mvt, lat, lon) that
    `fetch-geometry -c <config>` merges over the OSM stands when it builds the
    geometry tables. Relative to the repo root. The tables are shared by every
    experiment, so change the file and rebuild them (`fetch-geometry --offline`)
    rather than pointing different experiments at different files."""


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "unnamed"
    seed: int = 42
    notes: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)

    def __post_init__(self) -> None:
        needed = {self.target.residual_of, self.model.mixture.universe_column, *self.model.mixture.bases} - {None}
        if clash := sorted(needed & set(self.features.drop_columns)):
            raise ValueError(f"features.drop_columns removes columns the target or the mixture reads: {clash}")
        if self.model.mixture.enabled:
            # The mixture adds the candidate deltas back itself, in raw seconds.
            if self.target.residual_of:
                raise ValueError(
                    f"model.mixture and target.residual_of are exclusive (got residual_of='{self.target.residual_of}')"
                )
            if self.target.transform != "none":
                raise ValueError(f"target.transform must be 'none' with model.mixture (got '{self.target.transform}')")
            if self.model.eval_max_target_sec is not None:
                # The mixture's classifier needs the long rows in its eval set; its regressor has its own cap.
                raise ValueError("model.eval_max_target_sec is for a plain regressor; the mixture uses "
                                 "model.mixture.regressor_max_target_sec")

    def to_dict(self) -> dict:
        return _to_plain(asdict(self))

    def hash(self) -> str:
        """Digest over everything that changes the result -- `notes` excluded."""
        payload = self.to_dict()
        payload.pop("notes", None)
        if not payload["features"].get("drop_columns"):
            payload["features"].pop("drop_columns", None)
        if payload["model"]["mixture"].get("classifier_seeds") == 1:
            payload["model"]["mixture"].pop("classifier_seeds", None)
        if payload["model"]["mixture"].get("regressor_anchor") is None:
            payload["model"]["mixture"].pop("regressor_anchor", None)
            payload["model"]["mixture"].pop("anchor_clip", None)
        if payload["model"]["mixture"].get("anchor_airports") is None:
            payload["model"]["mixture"].pop("anchor_airports", None)
        if payload["model"]["mixture"].get("window_column") is None:
            payload["model"]["mixture"].pop("window_column", None)
            payload["model"]["mixture"].pop("window_half_width", None)
        if not payload["model"].get("month_weights"):
            payload["model"].pop("month_weights", None)
        if payload["model"].get("eval_max_target_sec") is None:
            payload["model"].pop("eval_max_target_sec", None)
        if not payload["model"].get("block_dropout"):
            payload["model"].pop("block_dropout", None)
        return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:8]

    def split_hash(self, seed: int | None = None) -> str:
        """Digest over just the things that determine fold membership.

        Fold-dependent feature caches are keyed on this rather than on the full
        config hash, so an ablation sweep -- which changes only the feature list
        and leaves the folds identical -- reuses one set of target statistics
        instead of recomputing them once per variant.
        """
        payload = {
            "validation": _to_plain(asdict(self.validation)),
            "data": _to_plain(asdict(self.data)),
            "target": _to_plain(asdict(self.target)),
            "seed": self.seed if seed is None else seed,
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:8]

    def replace(self, **overrides: Any) -> "ExperimentConfig":
        merged = _deep_merge(self.to_dict(), overrides)
        return from_dict(merged)


def _to_plain(obj):
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _build(cls, payload: Any):
    """Recursively construct a config dataclass, rejecting unknown keys."""
    if not is_dataclass(cls) or not isinstance(payload, dict):
        return payload
    known = {f.name for f in fields(cls)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"unknown key(s) in {cls.__name__}: {sorted(unknown)}. Known: {sorted(known)}")
    kwargs = {}
    for key, value in payload.items():
        nested = _NESTED.get((cls, key))
        kwargs[key] = _build(nested, value) if nested is not None else value
    return cls(**kwargs)


# Explicit nesting map: `dataclasses.fields` reports string annotations under
# `from __future__ import annotations`, so the types cannot be introspected.
_NESTED = {
    (ExperimentConfig, "data"): DataConfig,
    (DataConfig, "clean"): CleanConfig,
    (ExperimentConfig, "features"): FeatureConfig,
    (ExperimentConfig, "target"): TargetConfig,
    (ExperimentConfig, "validation"): ValidationConfig,
    (ExperimentConfig, "model"): ModelConfig,
    (ModelConfig, "mixture"): MixtureConfig,
    (ExperimentConfig, "evaluation"): EvaluationConfig,
    (ExperimentConfig, "geometry"): GeometryConfig,
    (EvaluationConfig, "permutation"): PermutationConfig,
}


def from_dict(payload: dict) -> ExperimentConfig:
    if "target" in payload and isinstance(payload["target"], dict):
        clip = payload["target"].get("clip")
        if isinstance(clip, list):
            payload = _deep_merge(payload, {"target": {"clip": tuple(clip)}})
    model = payload.get("model")
    weights = model.get("month_weights") if isinstance(model, dict) else None
    if isinstance(weights, dict) and any(isinstance(k, str) for k in weights):
        # JSON (a run's config.resolved.json) has string keys only; a month written "1" means month 1.
        fixed = {int(k) if isinstance(k, str) and k.strip().isdigit() else k: v for k, v in weights.items()}
        payload = copy.deepcopy(payload)
        payload["model"]["month_weights"] = fixed
    return _build(ExperimentConfig, payload)


def apply_overrides(payload: dict, overrides: list[str] | None) -> dict:
    """Dotted assignments -- ``model.params.depth=10`` -- applied to a config payload in place, values parsed as YAML
    scalars so types come out right (``true`` is a bool, ``0.05`` a float). Returns the payload."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got '{item}'")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        node = payload
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return payload


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    """Load a config, resolving `extends:` chains, then apply CLI overrides (`apply_overrides`)."""
    payload = _load_with_extends(Path(path))
    return from_dict(apply_overrides(payload, overrides))


def _load_with_extends(path: Path, _seen: set[Path] | None = None) -> dict:
    path = path if path.is_absolute() else (paths.CONFIGS / path if not path.exists() else path)
    path = path.resolve()
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"circular `extends` chain at {path}")
    _seen.add(path)

    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent_name = payload.pop("extends", None)
    if parent_name:
        parent = _load_with_extends(path.parent / parent_name, _seen)
        payload = _deep_merge(parent, payload)
    return payload


def dump_config(cfg: ExperimentConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
