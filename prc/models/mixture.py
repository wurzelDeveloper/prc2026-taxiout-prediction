"""A regime mixture: which of several candidate block times a departure's target follows.

``TAXITIME_SEC_mvt`` is takeoff minus the recorded block time, and for a minority
of departures the recorded block time is not a measurement but a schedule (the
scheduled off-block time; more rarely NM's last off-block time), so the target
is a delay of hours rather than a taxi of minutes. Each such candidate is a
feature column already (`stale_schedule.sched_delta_full`, ``lobt_delta_full``),
so on a row that carries one the target sits in one of ``K + 1`` regimes: a
taxi time, or takeoff minus candidate ``k``.

The MSE-optimal prediction for such a row is the mixture

    E[y | x] = sum_k p_k(x) * base_k(x) + p_0(x) * taxi(x)

and this model computes it explicitly: ``p`` from a small CatBoost classifier
fitted on the rows that carry a candidate (the *universe*), ``taxi`` from the
wrapped regressor (the per-airport CatBoost of the config), fitted without the
rows whose target follows a candidate. The residual-base alternative
(``target.residual_of``) asks one regressor to learn ``-(1 - p) * (base - taxi)``,
a product of a probability and a row-varying delta, which trees quantise
badly; here the two factors stay apart.

Calibration is the point: the mixture weight *is* the probability, so the
classifier is never fitted with ``auto_class_weights``, ``scale_pos_weight`` or
``class_weights``, which would inflate the stale share on purpose. Even so the
raw classifier is over-confident in both directions (findings 7.1: p above
0.98 was stale 63 % of the time), so ``calibration: isotonic`` re-maps its
stale probability with isotonic curves fitted on *nested* out-of-fold
probabilities of the fit universe, one curve per band of the row's largest
candidate delta -- a single curve ignoring the delta made the holdout worse,
because within a band the rows with the largest deltas are the stale ones and
the delta-weighted fit already leans on them.

Everything except the CatBoost fits is plain numpy and unit-tested with stub
models (`tests/test_mixture.py`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from prc.config import MixtureConfig
from prc.models.base import CategoryUniverse, prepare_matrix

TAXI = 0
"""Class 0: the target is a taxi time (the regressor's). Class ``k >= 1`` follows ``bases[k - 1]``;
with ``day_offset`` the last class is a whole day plus the regressor's taxi time."""
DAY = 86_400.0
NESTED_DAY_COLUMN = "takeoff_epoch"
"""The feature the nested calibration split groups by (a day at a time); rows are split round-robin without it."""

CLASSIFIER_DEFAULTS = {
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 4,
    "od_type": "Iter",
    "od_wait": 100,
    "thread_count": -1,
    "verbose": 100,
}
"""CPU by default: the suspect universe is ~1,500 rows, where a GPU fit is slower than a CPU one."""

FORBIDDEN_CLASSIFIER_PARAMS = ("auto_class_weights", "scale_pos_weight", "class_weights")
DETAIL_COLUMNS = ("y_reg", "p_stale", "p_stale_raw", "in_universe", "hedge_var")
"""What `predict_detail` adds next to ``y_pred``; `train` copies them into the OOF frame.
``p_stale`` is the mixture weight actually used, ``p_stale_raw`` the classifier's before calibration
(equal when ``calibration`` is ``none``). ``hedge_var`` is the variance of the row's regime distribution
around the prediction, sum_k p_k (value_k - y_pred)^2 (p(1 - p)(base - taxi)^2 with one base): the squared
error the hedge is expected to cost on that row if the probabilities are calibrated. It needs no target, so
its mean over the ranking rows is a reading of the 2026 tail (findings 9.11)."""
SEED_STRIDE = 1000
"""Classifier ``i`` of `classifier_seeds` uses ``seed + i * SEED_STRIDE``."""
P_BANDS = (0.05, 0.2, 0.5, 0.8, 0.95)
"""Edges of the stale-probability bands the calibration table of `hedging_report` uses."""


def _rmse(y_true, y_pred) -> float:
    err = np.asarray(y_pred, dtype="float64") - np.asarray(y_true, dtype="float64")
    return float(np.sqrt(np.mean(err**2)))


def _counts(labels: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(labels, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


class RegimeMixtureModel:
    encoding = "string"
    predict_threads = -1
    """CatBoost threads for the classifiers' predictions (`prc predict --threads`); -1 is CatBoost's own default."""

    def __init__(
        self,
        regressor,
        cfg: MixtureConfig,
        universe: CategoryUniverse,
        *,
        seed: int = 42,
        train_dir: Path | None = None,
        classifier_iterations: int | None = None,
        classifier_factory=None,
    ):
        self.regressor = regressor
        self.cfg = cfg
        self.universe = universe
        self.seed = seed
        self.train_dir = train_dir
        self.classifier_iterations = classifier_iterations
        """A fixed budget for the classifier (the refit, which has no eval set)."""
        self.classifier_factory = classifier_factory
        """Test seam: ``params -> classifier``; default is ``CatBoostClassifier``."""
        self.n_bases = len(cfg.bases)
        self.day_class: int | None = self.n_bases + 1 if cfg.day_offset else None
        self.n_classes = self.n_bases + 1 + (1 if cfg.day_offset else 0)
        self.classifier = None
        """The first classifier (importance, refit iterations and the tests read it)."""
        self.classifiers: list = []
        """All `classifier_seeds` classifiers; their probabilities are averaged."""
        self.classes_: list[int] = []
        self.constant_class_: int | None = None
        self.feature_names: list[str] = []
        self.diagnostics_: dict | None = None
        self.calibrators_: list[tuple[tuple[int, ...], object | None]] = []
        """Per pooled band group: the band indices it covers and its isotonic curve (None = identity)."""
        self.calibration_info_: list[dict] = []
        self.anchor_fallback_: dict[str, float] = {}
        """Per-airport median of the regressor's fit targets: the anchor where `regressor_anchor` is missing."""
        self.anchor_fallback_global_: float = 0.0
        self.state_source_: str = "fit"
        """Where the fit-time state came from: ``fit``, ``saved`` (`load` with the state file), ``missing`` (`load`
        without it: `restore_fit_state` must run before predicting) or ``restored``."""

    # --- regimes --------------------------------------------------------------
    def candidates(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per row: the candidate deltas ``B`` (n x K), which of them are valid, and universe membership.

        A base is valid when it is finite, at least ``base_min_sec``, and either the
        row is marked by ``universe_column`` or the base is at least ``min_base_sec``.
        The universe is every row with at least one valid base.
        """
        cfg = self.cfg
        missing = [c for c in cfg.bases if c not in X.columns]
        if missing:
            raise KeyError(f"model.mixture.bases {missing} are not features; include the blocks that make them")
        B = np.column_stack([pd.to_numeric(X[c], errors="coerce").to_numpy(dtype="float64") for c in cfg.bases])
        with np.errstate(invalid="ignore"):
            valid = np.isfinite(B) & (B >= cfg.base_min_sec)
            if cfg.universe_column is not None:
                if cfg.universe_column not in X.columns:
                    raise KeyError(f"model.mixture.universe_column '{cfg.universe_column}' is not a feature")
                marked = pd.to_numeric(X[cfg.universe_column], errors="coerce").fillna(0).to_numpy(dtype="float64") > 0
            else:
                marked = np.zeros(len(X), dtype=bool)
            large = (B >= cfg.min_base_sec) if cfg.min_base_sec is not None else np.zeros_like(valid)
        valid &= marked[:, None] | large
        lo, hi = self.window(X)
        with np.errstate(invalid="ignore"):
            # a candidate outside the NM join window cannot be the recorded block time
            valid &= ~(np.isfinite(lo)[:, None] & ((B < lo[:, None]) | (B > hi[:, None])))
        return B, valid, valid.any(axis=1)

    def window(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Per row: the target bounds of the NM join window (`window_column`), NaN where it is missing or unset."""
        col = self.cfg.window_column
        if col is None:
            nan = np.full(len(X), np.nan)
            return nan, nan
        if col not in X.columns:
            raise KeyError(f"model.mixture.window_column '{col}' is not a feature; include the block that makes it")
        w = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype="float64")
        return w - self.cfg.window_half_width, w + self.cfg.window_half_width

    def _clip_to_window(self, values: np.ndarray, X: pd.DataFrame) -> np.ndarray:
        lo, hi = self.window(X)
        out = np.where(np.isfinite(lo), np.maximum(values, lo), values)
        return np.where(np.isfinite(hi), np.minimum(out, hi), out)

    def labels(self, X: pd.DataFrame, y) -> np.ndarray:
        """Per row: -1 outside the universe, 0 for a taxi-time target, ``k`` for a target that follows ``bases[k-1]``."""
        B, valid, universe = self.candidates(X)
        y = np.asarray(y, dtype="float64")
        n = len(y)
        if n == 0:
            return np.zeros(0, dtype="int64")
        with np.errstate(invalid="ignore"):
            dist = np.where(valid, np.abs(y[:, None] - B), np.inf)
        k = dist.argmin(axis=1)
        rows = np.arange(n)
        best = dist[rows, k]
        nearest = np.where(valid[rows, k], B[rows, k], 0.0)
        tol = np.maximum(self.cfg.tol_sec, self.cfg.tol_frac * nearest)
        lab = np.where(best <= tol, k + 1, TAXI)
        if self.day_class is not None:
            # A block time on the wrong date: the target is whole days plus a
            # taxi. An exact candidate match (a stale block a day out) wins.
            days = np.round(y / DAY)
            wrong_date = (days >= 1) & (np.abs(y - days * DAY) <= self.cfg.day_tol_sec)
            lab = np.where((lab == TAXI) & wrong_date, self.day_class, lab)
        return np.where(universe, lab, -1).astype("int64")

    def row_weights(self, X: pd.DataFrame) -> np.ndarray | None:
        """The classifier's row weights (`classifier_weight_power`), or None when unweighted."""
        power = self.cfg.classifier_weight_power
        if power <= 0:
            return None
        B, valid, _ = self.candidates(X)
        # The largest candidate delta: what a wrong regime costs on this row. The
        # day-offset regime is left out on purpose, or every row would weigh the same.
        ref = np.where(valid, B, 0.0).max(axis=1)
        return np.maximum(ref, 1.0) ** power

    def _anchor(self, X: pd.DataFrame) -> np.ndarray:
        """The regressor's anchor per row (`regressor_anchor`), or zeros when the regressor is not anchored.

        The column is clipped to `anchor_clip`; a missing value takes the per-airport median of the fit rows'
        target (the global median for an airport the fit never saw)."""
        col = self.cfg.regressor_anchor
        if col is None:
            return np.zeros(len(X), dtype="float64")
        if col not in X.columns:
            raise KeyError(f"model.mixture.regressor_anchor '{col}' is not a feature; include the block that makes it")
        lo, hi = self.cfg.anchor_clip
        a = np.clip(pd.to_numeric(X[col], errors="coerce").to_numpy(dtype="float64"), lo, hi)
        missing = ~np.isfinite(a)
        if missing.any():
            fill = np.full(int(missing.sum()), self.anchor_fallback_global_, dtype="float64")
            if "apt" in X.columns:
                apt = X["apt"].astype("string").to_numpy()[missing]
                fill = np.array([self.anchor_fallback_.get(str(k), self.anchor_fallback_global_) for k in apt])
            a[missing] = fill
        if self.cfg.anchor_airports is not None:
            if "apt" not in X.columns:
                raise KeyError("model.mixture.anchor_airports needs the 'apt' column")
            keep = X["apt"].astype("string").isin(list(self.cfg.anchor_airports)).to_numpy(dtype=bool)
            a = np.where(keep, a, 0.0)
        return a

    def _fit_anchor_fallback(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self.anchor_fallback_ = {}
        self.anchor_fallback_global_ = float(np.median(y)) if len(y) else 0.0
        if self.cfg.regressor_anchor is None or "apt" not in X.columns:
            return
        s = pd.Series(y, index=X.index).groupby(X["apt"].astype("string").to_numpy()).median()
        self.anchor_fallback_ = {str(k): float(v) for k, v in s.items()}

    def _regressor_rows(self, lab: np.ndarray, y: np.ndarray, X: pd.DataFrame | None = None) -> np.ndarray:
        """Which rows the regressor is fitted (and early-stopped) on.

        With `window_column`, a row inside a finite join window is exempt from `regressor_max_target_sec`: the
        join put its block time within the window, so a long target there is a real (held) taxi, not a record
        error."""
        keep = lab <= 0 if self.cfg.exclude_stale_from_regressor else np.ones(len(lab), dtype=bool)
        if self.cfg.regressor_max_target_sec is not None:
            with np.errstate(invalid="ignore"):
                long = np.asarray(y, dtype="float64") > self.cfg.regressor_max_target_sec
            if X is not None and self.cfg.window_column is not None:
                lo, hi = self.window(X)
                yv = np.asarray(y, dtype="float64")
                with np.errstate(invalid="ignore"):
                    long &= ~(np.isfinite(lo) & (yv >= lo) & (yv <= hi))
            keep &= ~long
        return keep

    # --- fitting ----------------------------------------------------------------
    def _fit_state(self, X: pd.DataFrame, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """What `fit` derives from the fit rows before fitting anything: the regime labels, the regressor's rows and
        the anchor per row (setting the anchor's fallback medians). `restore_fit_state` replays it for a loaded model."""
        lab = self.labels(X, y)
        keep = self._regressor_rows(lab, y, X)
        # The anchor's fallback comes from the regressor's own rows (taxi times), never from stale targets.
        self._fit_anchor_fallback(X.loc[keep] if not keep.all() else X, y[keep])
        return lab, keep, self._anchor(X)

    def fit(self, X, y, *, cat_features, eval_set=None, sample_weight=None):
        y = np.asarray(y, dtype="float64")
        w = None if sample_weight is None else np.asarray(sample_weight, dtype="float64")
        lab, keep, anchor = self._fit_state(X, y)
        reg_eval = None
        val_lab = None
        if eval_set is not None:
            Xv, yv = eval_set
            yv = np.asarray(yv, dtype="float64")
            val_lab = self.labels(Xv, yv)
            vkeep = self._regressor_rows(val_lab, yv, Xv)
            anchor_v = self._anchor(Xv)
            reg_eval = (Xv if vkeep.all() else Xv.loc[vkeep], (yv - anchor_v)[vkeep])
        anchored = "" if self.cfg.regressor_anchor is None else f", anchored on {self.cfg.regressor_anchor}"
        print(
            f"  [mixture] universe {int((lab >= 0).sum()):,} of {len(y):,} rows, labels {_counts(lab[lab >= 0])}; "
            f"regressor on {int(keep.sum()):,} rows{anchored}",
            flush=True,
        )
        extra = {} if w is None else {"sample_weight": w[keep]}
        self.regressor.fit(
            X if keep.all() else X.loc[keep], (y - anchor)[keep], cat_features=cat_features, eval_set=reg_eval, **extra
        )
        self._fit_classifier(X, lab, cat_features, eval_set, val_lab, sample_weight=w)
        if eval_set is not None:
            self.diagnostics_ = self._diagnostics(lab, Xv, yv, val_lab)
        return self

    def _fit_classifier(self, X, lab, cat_features, eval_set, val_lab, sample_weight=None) -> None:
        from catboost import Pool

        u = lab >= 0
        self.feature_names = list(X.columns)
        cats = [c for c in cat_features if c in X.columns]
        present = np.unique(lab[u])
        if len(present) < 2:
            # One regime only (a tiny or single-carrier universe): nothing to classify.
            self.constant_class_ = int(present[0]) if len(present) else TAXI
            self.classifier = None
            self.classifiers = []
            self.classes_ = [self.constant_class_]
            return

        weights = self.row_weights(X)
        if sample_weight is not None:
            # the fit's own row weights (`model.month_weights`) times the delta weights
            weights = sample_weight if weights is None else weights * sample_weight
        scale = None if weights is None else float(weights[u].mean())
        frame_u = prepare_matrix(X.loc[u], self.universe, encoding=self.encoding)
        weights_u = None if weights is None else weights[u] / scale
        train_pool = Pool(frame_u, lab[u], cat_features=cats, weight=weights_u)
        valid_pool = None
        if eval_set is not None and self.cfg.classifier_eval and val_lab is not None:
            Xv, _ = eval_set
            uv = val_lab >= 0
            if uv.any():
                # CatBoost cannot score a class the fit never saw; for early
                # stopping only, such rows count as the fit's first class.
                vl = np.where(np.isin(val_lab[uv], present), val_lab[uv], int(present[0]))
                vw = None if self.row_weights(Xv) is None else self.row_weights(Xv)[uv] / scale
                valid_pool = Pool(
                    prepare_matrix(Xv.loc[uv], self.universe, encoding=self.encoding), vl, cat_features=cats, weight=vw
                )

        params = self._classifier_params(has_eval=valid_pool is not None)
        self.classifiers = []
        for i in range(self.cfg.classifier_seeds):
            p_i = dict(params)
            if i:
                p_i["random_seed"] = self.seed + i * SEED_STRIDE
                if "train_dir" in p_i:
                    p_i["train_dir"] = str(Path(p_i["train_dir"]) / f"seed{i}")
                    Path(p_i["train_dir"]).mkdir(parents=True, exist_ok=True)
            clf = self._make_classifier(p_i)
            clf.fit(train_pool, eval_set=valid_pool, use_best_model=valid_pool is not None)
            self.classifiers.append(clf)
        self.classifier = self.classifiers[0]
        self.classes_ = [int(c) for c in self.classifier.classes_]
        if self.cfg.calibration != "none":
            self._fit_calibration(X.loc[u], frame_u, lab[u], weights_u, cats, params)

    # --- calibration ------------------------------------------------------------
    def _reference(self, X: pd.DataFrame) -> np.ndarray:
        """The row's largest valid candidate delta: what a wrong regime costs, and the band key."""
        B, valid, _ = self.candidates(X)
        return np.where(valid, B, 0.0).max(axis=1)

    def _nested_folds(self, X_u: pd.DataFrame) -> np.ndarray:
        k = self.cfg.calibration_folds
        if NESTED_DAY_COLUMN in X_u.columns:
            day = pd.to_numeric(X_u[NESTED_DAY_COLUMN], errors="coerce").fillna(0).to_numpy(dtype="float64") // DAY
            return pd.factorize(day.astype("int64"), sort=True)[0] % k
        return np.arange(len(X_u)) % k

    def _band_groups(self, bands: np.ndarray) -> list[tuple[int, ...]]:
        """Bands with fewer than `calibration_min_rows` rows pooled into the band below (the lowest into the one above)."""
        n_bands = len(self.cfg.calibration_delta_bands) + 1
        groups: list[list[int]] = [[b] for b in range(n_bands)]
        counts = [int((bands == b).sum()) for b in range(n_bands)]
        i = n_bands - 1
        while i >= 0 and len(groups) > 1:
            if counts[i] < self.cfg.calibration_min_rows:
                j = i - 1 if i > 0 else 1
                groups[j] = sorted(groups[j] + groups[i]) if j < i else sorted(groups[i] + groups[j])
                counts[j] += counts[i]
                del groups[i]
                del counts[i]
                if j > i:
                    # merged upward: the survivor now sits at index i
                    i -= 1
                    continue
            i -= 1
        return [tuple(g) for g in groups]

    def _fit_calibration(self, X_u, frame_u, lab_u, weights_u, cats, params) -> None:
        """Isotonic curves from nested out-of-fold probabilities, one per pooled delta band."""
        from catboost import Pool
        from sklearn.isotonic import IsotonicRegression
        from sklearn.metrics import log_loss

        nested = dict(params)
        for key in ("od_type", "od_wait", "train_dir"):
            nested.pop(key, None)
        nested["allow_writing_files"] = False
        best = self.classifier_refit_iterations()
        if best is not None:
            nested["iterations"] = best

        folds = self._nested_folds(X_u)
        P = np.zeros((len(X_u), self.n_classes), dtype="float64")
        for f in range(self.cfg.calibration_folds):
            test = folds == f
            train = ~test
            if not test.any():
                continue
            present = np.unique(lab_u[train])
            if len(present) < 2:
                P[test, int(present[0]) if len(present) else TAXI] = 1.0
                continue
            pool = Pool(frame_u.loc[train], lab_u[train], cat_features=cats, weight=None if weights_u is None else weights_u[train])
            clf = self._make_classifier(nested)
            clf.fit(pool)
            raw = np.asarray(clf.predict_proba(frame_u.loc[test]), dtype="float64")
            if raw.ndim == 1:
                raw = np.column_stack([1.0 - raw, raw])
            for j, cls in enumerate(int(c) for c in clf.classes_):
                if 0 <= cls < self.n_classes:
                    P[test, cls] = raw[:, j]

        p_raw = 1.0 - P[:, TAXI]
        stale = (lab_u != TAXI).astype("float64")
        bands = np.searchsorted(np.asarray(self.cfg.calibration_delta_bands, dtype="float64"), self._reference(X_u), side="right")
        self.calibrators_ = []
        self.calibration_info_ = []
        for group in self._band_groups(bands):
            rows = np.isin(bands, group)
            info = {"bands": list(group), "n": int(rows.sum()), "stale_share": float(stale[rows].mean()) if rows.any() else None}
            iso = None
            if rows.sum() >= 2 and 0 < stale[rows].mean() < 1:
                iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                iso.fit(p_raw[rows], stale[rows], sample_weight=None if weights_u is None else weights_u[rows])
                cal = np.clip(iso.predict(p_raw[rows]), 1e-6, 1 - 1e-6)
                info["logloss_raw"] = float(log_loss(stale[rows], np.clip(p_raw[rows], 1e-6, 1 - 1e-6)))
                info["logloss_calibrated"] = float(log_loss(stale[rows], cal))
            self.calibrators_.append((group, iso))
            self.calibration_info_.append(info)
        print(f"  [mixture] calibration: {self.calibration_info_}", flush=True)

    def _calibrate(self, P: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Re-map the stale mass of each row by its band's curve; the non-taxi classes keep their proportions."""
        if not self.calibrators_:
            return P
        bands = np.searchsorted(np.asarray(self.cfg.calibration_delta_bands, dtype="float64"), ref, side="right")
        out = P.copy()
        p_raw = 1.0 - P[:, TAXI]
        for group, iso in self.calibrators_:
            rows = np.isin(bands, group)
            if iso is None or not rows.any():
                continue
            g = np.clip(iso.predict(p_raw[rows]), 0.0, 1.0)
            non_taxi = P[rows, 1:]
            mass = non_taxi.sum(axis=1)
            share = np.where(mass[:, None] > 0, non_taxi / np.where(mass > 0, mass, 1.0)[:, None], 1.0 / non_taxi.shape[1])
            out[rows, 1:] = share * g[:, None]
            out[rows, TAXI] = 1.0 - g
        return out

    def _classifier_params(self, *, has_eval: bool) -> dict:
        params = CLASSIFIER_DEFAULTS | dict(self.cfg.classifier_params) | {"random_seed": self.seed}
        bad = [k for k in FORBIDDEN_CLASSIFIER_PARAMS if k in params]
        if bad:
            raise ValueError(f"model.mixture.classifier_params {bad}: the mixture needs calibrated probabilities")
        loss = "Logloss" if self.n_classes == 2 else "MultiClass"
        params["loss_function"] = loss
        params.setdefault("eval_metric", loss)
        if self.classifier_iterations is not None:
            params["iterations"] = int(self.classifier_iterations)
        if not has_eval:
            params.pop("od_type", None)
            params.pop("od_wait", None)
        if self.train_dir is not None:
            Path(self.train_dir).mkdir(parents=True, exist_ok=True)
            params["train_dir"] = str(self.train_dir)
            params.setdefault("allow_writing_files", True)
        else:
            params.setdefault("allow_writing_files", False)
        return params

    def _make_classifier(self, params: dict):
        if self.classifier_factory is not None:
            return self.classifier_factory(params)
        from catboost import CatBoostClassifier

        return CatBoostClassifier(**params)

    # --- prediction -------------------------------------------------------------
    def _proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Class probabilities on the ``K + 1`` axis, averaged over the classifiers; a class the fit never saw gets 0."""
        P = np.zeros((len(frame), self.n_classes), dtype="float64")
        if self.classifier is None:
            P[:, self.constant_class_ if self.constant_class_ is not None else TAXI] = 1.0
            return P
        classifiers = self.classifiers or [self.classifier]
        threads = {} if self.predict_threads == -1 else {"thread_count": self.predict_threads}
        for clf in classifiers:
            raw = np.asarray(clf.predict_proba(frame, **threads), dtype="float64")
            if raw.ndim == 1:
                raw = np.column_stack([1.0 - raw, raw])
            for j, cls in enumerate(int(c) for c in clf.classes_):
                if 0 <= cls < self.n_classes:
                    P[:, cls] += raw[:, j]
        return P / len(classifiers)

    def _mix(
        self, P: np.ndarray, valid_u: np.ndarray, B_u: np.ndarray, reg_u: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The mixture prediction, the stale mass and the hedge variance, after masking invalid candidates and renormalising."""
        P = P.copy()
        bases = slice(1, self.n_bases + 1)
        # A candidate that is invalid on this row gets no mass; what is left is
        # renormalised, and a row with no mass left is the regressor's.
        P[:, bases] *= valid_u
        total = P.sum(axis=1)
        none = total <= 0
        P[none] = 0.0
        P[none, TAXI] = 1.0
        total[none] = 1.0
        P /= total[:, None]
        # The value each regime stands for on this row: the regressor's taxi, each candidate, a day plus the taxi.
        V = np.zeros_like(P)
        V[:, TAXI] = reg_u
        V[:, bases] = np.where(valid_u, B_u, 0.0)
        if self.day_class is not None:
            V[:, self.day_class] = DAY + reg_u
        mixed = (P * V).sum(axis=1)
        hedge = (P * (V - mixed[:, None]) ** 2).sum(axis=1)
        return mixed, 1.0 - P[:, TAXI], hedge

    def predict_detail(self, X: pd.DataFrame, *, calibrated: bool = True) -> pd.DataFrame:
        """``y_pred`` plus what made it: the regressor's ``y_reg``, ``p_stale`` (the mixture weight, 1 - p_taxi),
        ``p_stale_raw`` (the classifier's before calibration) and ``in_universe``."""
        if self.state_source_ == "missing":
            raise RuntimeError("this mixture was loaded without its fit-time state; call restore_fit_state on its fit rows first")
        reg = self._clip_to_window(np.asarray(self.regressor.predict(X), dtype="float64") + self._anchor(X), X)
        B, valid, universe = self.candidates(X)
        out = pd.DataFrame(
            {
                "y_pred": reg, "y_reg": reg, "p_stale": np.zeros(len(X)), "p_stale_raw": np.zeros(len(X)),
                "in_universe": universe, "hedge_var": np.zeros(len(X)),
            },
            index=X.index,
        )
        if not universe.any():
            return out
        frame = prepare_matrix(X.loc[universe], self.universe, encoding=self.encoding)
        P_raw = self._proba(frame)
        ref = np.where(valid[universe], B[universe], 0.0).max(axis=1)
        P = self._calibrate(P_raw, ref) if calibrated else P_raw
        mixed, p_stale, hedge = self._mix(P, valid[universe], B[universe], reg[universe])
        _, p_stale_raw, _ = self._mix(P_raw, valid[universe], B[universe], reg[universe])
        out.loc[universe, "y_pred"] = self._clip_to_window(mixed, X.loc[universe])
        out.loc[universe, "p_stale"] = p_stale
        out.loc[universe, "p_stale_raw"] = p_stale_raw
        out.loc[universe, "hedge_var"] = hedge
        return out

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_detail(X)["y_pred"].to_numpy()

    # --- diagnostics --------------------------------------------------------------
    def _diagnostics(self, lab, Xv, yv, val_lab) -> dict:
        info = {
            "n_universe_train": int((lab >= 0).sum()),
            "n_universe_val": int((val_lab >= 0).sum()),
            "labels_train": _counts(lab[lab >= 0]),
            "labels_val": _counts(val_lab[val_lab >= 0]),
            "classifier_best_iteration": self.classifier_refit_iterations(),
        }
        u = val_lab >= 0
        if not u.any():
            return info
        Xu = Xv.loc[u]
        detail = self.predict_detail(Xu)
        yu = yv[u]
        lu = val_lab[u]
        yp = detail["y_pred"].to_numpy()
        yr = detail["y_reg"].to_numpy()
        ps = detail["p_stale"].to_numpy()
        B, _, _ = self.candidates(Xu)
        rows = np.arange(len(lu))
        is_base = (lu > 0) & (lu <= self.n_bases)
        oracle = np.where(is_base, B[rows, np.clip(lu - 1, 0, self.n_bases - 1)], yr)
        if self.day_class is not None:
            oracle = np.where(lu == self.day_class, np.round(yu / DAY) * DAY + yr, oracle)
        info |= {
            "rmse_universe_mixture": _rmse(yu, yp),
            "rmse_universe_regressor": _rmse(yu, yr),
            "rmse_universe_oracle": _rmse(yu, oracle),
            "p_stale_mean_by_label": {str(int(k)): float(ps[lu == k].mean()) for k in np.unique(lu)},
        }
        if self.calibrators_:
            info["rmse_universe_mixture_uncalibrated"] = _rmse(yu, self.predict_detail(Xu, calibrated=False)["y_pred"].to_numpy())
            info["calibration"] = self.calibration_info_
        stale = (lu > 0).astype(int)
        if stale.min() != stale.max():
            from sklearn.metrics import log_loss, roc_auc_score

            info["auc"] = float(roc_auc_score(stale, ps))
            info["logloss"] = float(log_loss(stale, np.clip(ps, 1e-6, 1 - 1e-6)))
        return info

    # --- contracts ----------------------------------------------------------------
    def refit_iterations(self):
        """The regressor's early-stopped count (an int, or a per-airport map), unchanged."""
        fn = getattr(self.regressor, "refit_iterations", None)
        return fn() if callable(fn) else None

    def classifier_refit_iterations(self) -> int | None:
        """The classifiers' early-stopped count (their mean with `classifier_seeds` > 1)."""
        bests = [getattr(c, "best_iteration_", None) for c in (self.classifiers or [self.classifier])]
        bests = [int(b) + 1 for b in bests if b is not None]
        return int(round(float(np.mean(bests)))) if bests else None

    def native_importance(self) -> pd.DataFrame:
        return self.regressor.native_importance()

    def classifier_importance(self) -> pd.DataFrame:
        """Gain importance, averaged over the classifiers."""
        tables = []
        for clf in self.classifiers or [self.classifier]:
            fn = getattr(clf, "get_feature_importance", None)
            if not callable(fn):
                continue
            values = np.asarray(fn(), dtype="float64")
            if len(values) == len(self.feature_names):
                tables.append(values)
        if not tables:
            return pd.DataFrame(columns=["feature", "gain"])
        return pd.DataFrame({"feature": self.feature_names, "gain": np.mean(tables, axis=0)}).sort_values("gain", ascending=False)

    def save(self, path: Path) -> None:
        """The regressor (its own files), each classifier (``<stem>__classifier[_s<i>].cbm``), the calibration curves
        (``<stem>__calibration.joblib``) and the fit-time state prediction needs besides them (``<stem>__mixture.json``:
        the anchor's fallback medians, the class of a single-regime universe, the classifier count). Runs saved before
        the state file existed are restored from their training rows instead (`restore_fit_state`)."""
        self.regressor.save(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n_saved = 0
        for i, clf in enumerate(self.classifiers or [self.classifier]):
            save = getattr(clf, "save_model", None)
            if callable(save):
                save(str(classifier_file(path, i)))
                n_saved += 1
        if self.calibrators_:
            import joblib

            joblib.dump(
                {"delta_bands": list(self.cfg.calibration_delta_bands), "calibrators": self.calibrators_, "info": self.calibration_info_},
                path.with_name(f"{path.stem}__calibration.joblib"),
            )
        state = {
            "n_classifiers": n_saved,
            "classes": [int(c) for c in self.classes_],
            "constant_class": None if self.constant_class_ is None else int(self.constant_class_),
            "anchor_fallback": {str(k): float(v) for k, v in self.anchor_fallback_.items()},
            "anchor_fallback_global": float(self.anchor_fallback_global_),
            "feature_names": list(self.feature_names),
            "calibrated": bool(self.calibrators_),
        }
        path.with_name(f"{path.stem}{STATE_SUFFIX}").write_text(json.dumps(state, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, cfg: MixtureConfig, universe: CategoryUniverse, regressor, *, seed: int = 42) -> "RegimeMixtureModel":
        """The mixture `save` wrote at ``path`` around an already loaded ``regressor``, ready to predict.

        With ``<stem>__mixture.json`` every piece of state comes from the files. Without it (runs saved before it
        existed) the classifiers are the ``__classifier*.cbm`` files present (none: a single-regime universe), and
        the anchor's fallback medians and that single regime are left unset (None) for `restore_fit_state`, which
        recomputes them from the training rows; predicting before that raises.
        """
        from catboost import CatBoostClassifier

        path = Path(path)
        model = cls(regressor, cfg, universe, seed=seed)
        state_file = path.with_name(f"{path.stem}{STATE_SUFFIX}")
        state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else None
        if state is not None:
            n = int(state["n_classifiers"])
            missing = [classifier_file(path, i).name for i in range(n) if not classifier_file(path, i).exists()]
            if missing:
                raise FileNotFoundError(f"{state_file.name} lists {n} classifier(s); missing {missing} in {path.parent}")
        else:
            present = [classifier_file(path, i).exists() for i in range(max(cfg.classifier_seeds, 1))]
            n = sum(present)
            if any(present) and not all(present):
                raise FileNotFoundError(
                    f"model.mixture.classifier_seeds={cfg.classifier_seeds}, but only classifier files "
                    f"{[classifier_file(path, i).name for i, ok in enumerate(present) if ok]} exist in {path.parent}"
                )
        model.classifiers = []
        for i in range(n):
            clf = CatBoostClassifier()
            clf.load_model(str(classifier_file(path, i)))
            model.classifiers.append(clf)
        model.classifier = model.classifiers[0] if model.classifiers else None
        if model.classifier is not None:
            model.classes_ = [int(c) for c in model.classifier.classes_]
            model.feature_names = list(model.classifier.feature_names_)
        calibration_file = path.with_name(f"{path.stem}__calibration.joblib")
        if calibration_file.exists():
            import joblib

            payload = joblib.load(calibration_file)
            if [float(e) for e in payload["delta_bands"]] != [float(e) for e in cfg.calibration_delta_bands]:
                raise ValueError(
                    f"{calibration_file.name} was fitted on delta bands {payload['delta_bands']}, "
                    f"the config says {list(cfg.calibration_delta_bands)}"
                )
            model.calibrators_ = [(tuple(int(b) for b in group), iso) for group, iso in payload["calibrators"]]
            model.calibration_info_ = list(payload.get("info") or [])
        elif cfg.calibration != "none" and model.classifier is not None:
            raise FileNotFoundError(f"model.mixture.calibration is '{cfg.calibration}' but {calibration_file.name} is missing")
        if state is not None:
            model.constant_class_ = None if state["constant_class"] is None else int(state["constant_class"])
            if model.classifier is None:
                model.classes_ = [int(c) for c in state["classes"]]
            model.anchor_fallback_ = {str(k): float(v) for k, v in state["anchor_fallback"].items()}
            model.anchor_fallback_global_ = float(state["anchor_fallback_global"])
            model.feature_names = list(state.get("feature_names") or model.feature_names)
        else:
            model.constant_class_ = None
            model.anchor_fallback_ = None
            model.anchor_fallback_global_ = None
        model.state_source_ = "saved" if state is not None else "missing"
        return model

    def restore_fit_state(self, X: pd.DataFrame, y) -> tuple[np.ndarray, np.ndarray, dict]:
        """Replay `fit`'s bookkeeping on the fit rows ``X``/``y`` for a model loaded by `load`.

        State the files lacked (a run saved before ``__mixture.json``) is set from the replay: the anchor's fallback
        medians and, without a classifier, the single regime of the universe. State the files had is kept and the
        replay only checked against it. Returns the regressor's rows (a mask over ``X``), its fit target on them
        (the target minus the anchor) and a report. The replay reads only feature columns of the fit rows and their
        targets -- the same rows and columns `fit` read when the refit is restored over the refit's own rows.
        """
        y = np.asarray(y, dtype="float64")
        saved_fallback, saved_global = self.anchor_fallback_, self.anchor_fallback_global_
        lab, keep, anchor = self._fit_state(X, y)
        report: dict = {"recomputed": [], "checked": {}}
        if saved_fallback is None:
            report["recomputed"].append("anchor_fallback")
        else:
            same = set(saved_fallback) == set(self.anchor_fallback_) and all(
                np.isclose(saved_fallback[k], self.anchor_fallback_[k], rtol=0, atol=1e-9) for k in saved_fallback
            ) and bool(np.isclose(saved_global, self.anchor_fallback_global_, rtol=0, atol=1e-9))
            report["checked"]["anchor_fallback"] = bool(same)
            self.anchor_fallback_, self.anchor_fallback_global_ = saved_fallback, saved_global
            anchor = self._anchor(X)
        present = np.unique(lab[lab >= 0])
        if self.classifier is None:
            if len(present) >= 2:
                raise ValueError(
                    f"no classifier was saved, but the fit universe holds {len(present)} regimes {present.tolist()}: "
                    "the classifier file is missing, or these are not the rows the model was fitted on"
                )
            single = int(present[0]) if len(present) else TAXI
            if self.constant_class_ is None:
                self.constant_class_ = single
                self.classes_ = [single]
                report["recomputed"].append("constant_class")
            else:
                report["checked"]["constant_class"] = self.constant_class_ == single
        else:
            report["checked"]["universe_has_several_regimes"] = bool(len(present) >= 2)
        report |= {
            "n_fit": int(len(y)),
            "n_universe_fit": int((lab >= 0).sum()),
            "labels_fit": _counts(lab[lab >= 0]),
            "n_regressor_fit": int(keep.sum()),
        }
        if self.state_source_ == "missing":
            self.state_source_ = "restored"
        return keep, (y - anchor)[keep], report

    def state_columns(self) -> list[str]:
        """The feature columns `restore_fit_state` reads (candidates, universe marker, window, anchor, airport)."""
        cfg = self.cfg
        cols = [*cfg.bases, cfg.universe_column, cfg.window_column, cfg.regressor_anchor, "apt"]
        return list(dict.fromkeys(c for c in cols if c is not None))


STATE_SUFFIX = "__mixture.json"
"""`RegimeMixtureModel.save` writes the fit-time state prediction needs to ``<stem>__mixture.json``."""


def classifier_file(path: Path, i: int) -> Path:
    """Where `RegimeMixtureModel.save` puts classifier ``i`` of the model saved at ``path``."""
    path = Path(path)
    return path.with_name(f"{path.stem}__classifier{'' if i == 0 else f'_s{i}'}.cbm")


def hedging_report(detail: pd.DataFrame, regime=None) -> dict:
    """The hedge's expected cost and, with the regime labels, how calibrated the stale probability is.

    ``detail`` carries `DETAIL_COLUMNS` for every scored row (universe or not). ``hedge`` is the mean of
    ``hedge_var`` over *all* rows: the hedge's share of the MSE, in the MSE's units. With ``regime`` (the labels
    of `RegimeMixtureModel.labels` on the same rows: -1 outside the universe, 0 taxi, k > 0 a base or the day),
    the universe rows are grouped by `P_BANDS` of ``p_stale``: per band the rows, the mean probability and the
    stale share; ``ece`` is their row-weighted absolute gap, ``brier`` the mean squared gap per row. A model can
    lower ``hedge`` by being confident; only with ``ece`` beside it does a lower hedge mean better discrimination.
    """
    n = len(detail)
    report: dict = {
        "hedge": float(detail["hedge_var"].sum() / n) if n else None,
        "n_universe": int(np.asarray(detail["in_universe"], dtype=bool).sum()),
    }
    if regime is None:
        return report
    lab = np.asarray(regime, dtype="int64")
    u = np.asarray(detail["in_universe"], dtype=bool) & (lab >= 0)
    if not u.any():
        return report
    p = np.asarray(detail["p_stale"], dtype="float64")[u]
    stale = (lab[u] > 0).astype("float64")
    band = np.searchsorted(np.asarray(P_BANDS), p, side="right")
    edges = (0.0, *P_BANDS, 1.0)
    table, gap = [], 0.0
    for b in range(len(P_BANDS) + 1):
        rows = band == b
        if not rows.any():
            continue
        mean_p, rate = float(p[rows].mean()), float(stale[rows].mean())
        gap += rows.sum() * abs(mean_p - rate)
        table.append({"band": f"{edges[b]:.2f}-{edges[b + 1]:.2f}", "n": int(rows.sum()), "mean_p": mean_p, "stale_rate": rate})
    report |= {"calibration": table, "ece": float(gap / u.sum()), "brier": float(np.mean((p - stale) ** 2))}
    return report
