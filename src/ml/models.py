"""Per-number probability models and the model registry.

Every model maps a feature tensor ``(m, 90, F)`` to probabilities ``(m, 90)``
normalised to sum to 5 (the number of balls drawn). ML models use one shared
binary classifier over flattened (draw, number) rows; number identity enters
through the ``number_norm``/``is_odd``/``decade`` features.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from dataclasses import dataclass

import numpy as np

from src.features.engineering import BASE_RATE, FEATURE_NAMES, N_NUMBERS, NON_ML_FEATURES, PICK

EPS = 1e-6
OPTIONAL_DISABLE_ENV = "OTOSLOTTO_DISABLE_OPTIONAL_MODELS"


def normalize_probabilities(p) -> np.ndarray:
    """Proportionally allocate five marginal probabilities with valid bounds."""
    p = np.asarray(p, dtype=np.float64)
    if p.ndim not in (1, 2) or p.shape[-1] != N_NUMBERS:
        raise ValueError("Scores must have shape (90,) or (draws,90)")
    if not np.isfinite(p).all() or (p < 0).any() or (p.max(axis=-1) <= 0).any():
        raise ValueError("Scores must be finite, nonnegative and have positive row totals")
    weights = np.maximum(p / p.max(axis=-1, keepdims=True), 1e-12)
    scaled = weights * (PICK / weights.sum(axis=-1, keepdims=True))
    if np.all((scaled >= EPS) & (scaled <= 1 - EPS)):
        return scaled
    output = np.empty_like(weights)
    for row, result in zip(weights.reshape(-1, N_NUMBERS), output.reshape(-1, N_NUMBERS)):
        # Saturated numbers retain their cap; remaining mass is redistributed.
        active = np.ones(N_NUMBERS, dtype=bool)
        result.fill(EPS)
        remaining = PICK - N_NUMBERS * EPS
        while active.any():
            allocation = remaining * row[active] / row[active].sum()
            capped = allocation > 1 - 2 * EPS
            if not capped.any():
                result[active] += allocation
                break
            indices = np.flatnonzero(active)[capped]
            result[indices] = 1 - EPS
            remaining -= len(indices) * (1 - 2 * EPS)
            active[indices] = False
    return output


def binary_log_loss(p, y) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    y = np.asarray(y, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


class BaseProbabilityModel:
    name = "base"
    kind = "baseline"

    def __init__(self, feature_names=None, seed: int = 42, **params):
        self.feature_names = list(feature_names or FEATURE_NAMES)
        self.seed = int(seed)
        unknown = set(params) - set(self.default_params())
        if unknown and not self.accepts_any_params:
            raise TypeError(f"{self.name}: unknown parameters {sorted(unknown)}")
        self.params = {**self.default_params(), **params}
        self.fitted_ = False
        self.info_: dict = {}

    accepts_any_params = False

    @classmethod
    def default_params(cls) -> dict:
        return {}

    def _col(self, name: str) -> int:
        return self.feature_names.index(name)

    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y)
        if X.ndim != 3 or X.shape[1] != N_NUMBERS or Y.shape != X.shape[:2]:
            raise ValueError(f"expected X (T, 90, F) and Y (T, 90); got {X.shape} and {Y.shape}")
        self._fit(X, Y)
        self.fitted_ = True
        return self

    def _fit(self, X, Y):
        pass

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        single = X.ndim == 2
        if single:
            X = X[None]
        if len(X) == 0:
            return np.empty((0, N_NUMBERS))
        p = normalize_probabilities(self._predict(X))
        return p[0] if single else p

    def _predict(self, X):
        raise NotImplementedError


class UniformModel(BaseProbabilityModel):
    name = "uniform"

    def _predict(self, X):
        return np.full(X.shape[:2], BASE_RATE)


class RandomBaselineModel(BaseProbabilityModel):
    """Exact uniform marginals with a separate seeded random ticket ranking."""

    name = "random"

    def _predict(self, X):
        return np.full(X.shape[:2], BASE_RATE)

    def ranking_scores(self, X):
        X = np.asarray(X)
        single = X.ndim == 2
        if single:
            X = X[None]
        hist = X[:, 0, self._col("history")].astype(np.int64)
        out = np.empty(X.shape[:2])
        for i, t in enumerate(hist):
            out[i] = np.random.default_rng([self.seed, int(t)]).random(N_NUMBERS)
        return out[0] if single else out


class FrequencyModel(BaseProbabilityModel):
    """Lightly smoothed all-history frequency (count + 0.5) / (draws + 9)."""

    name = "frequency"
    kind = "statistical"

    def _predict(self, X):
        count = X[:, :, self._col("count_all")]
        hist = X[:, :, self._col("history")]
        return (count + 0.5) / (hist + 0.5 / BASE_RATE)


class EWFrequencyModel(BaseProbabilityModel):
    """Exponentially weighted frequency; half-life chosen on training draws."""

    name = "ew_frequency"
    kind = "statistical"
    candidates = ("ew_fast", "ew_mid", "ew_slow")

    @classmethod
    def default_params(cls):
        return {"feature": None}

    def _fit(self, X, Y):
        feature = self.params["feature"]
        if feature is None:
            losses = {c: binary_log_loss(normalize_probabilities(X[:, :, self._col(c)]), Y) for c in self.candidates}
            feature = min(losses, key=losses.get)
            self.info_["training_log_loss"] = losses
        self.feature_ = feature
        self.info_["feature"] = feature

    def _predict(self, X):
        return X[:, :, self._col(getattr(self, "feature_", self.params["feature"] or "ew_mid"))]


class BayesianModel(BaseProbabilityModel):
    """Beta-binomial posterior mean with prior Beta(a*5/90, a*85/90).

    The prior strength ``a`` is chosen by training-window log loss (empirical
    Bayes); large ``a`` shrinks to uniform.
    """

    name = "bayesian"
    kind = "statistical"

    @classmethod
    def default_params(cls):
        return {"prior_strength": None, "grid": (10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)}

    def _posterior(self, X, a):
        count = X[:, :, self._col("count_all")]
        hist = X[:, :, self._col("history")]
        return (count + a * BASE_RATE) / (hist + a)

    def _fit(self, X, Y):
        a = self.params["prior_strength"]
        if a is None:
            losses = {float(g): binary_log_loss(normalize_probabilities(self._posterior(X, g)), Y) for g in self.params["grid"]}
            a = min(losses, key=losses.get)
            self.info_["training_log_loss"] = losses
        self.prior_strength_ = float(a)
        self.info_["prior_strength"] = float(a)

    def _predict(self, X):
        return self._posterior(X, getattr(self, "prior_strength_", self.params["prior_strength"] or 1000.0))


class SharedClassifierModel(BaseProbabilityModel):
    """One binary classifier shared across all 90 numbers."""

    kind = "ml"
    accepts_any_params = True

    def _make_estimator(self):
        raise NotImplementedError

    def _ml_columns(self):
        return [i for i, n in enumerate(self.feature_names) if n not in NON_ML_FEATURES]

    def _fit(self, X, Y):
        cols = self._ml_columns()
        flat_x = X[:, :, cols].reshape(-1, len(cols))
        flat_y = np.asarray(Y, dtype=np.int64).reshape(-1)
        self.constant_ = None
        self.estimator_ = None
        if np.unique(flat_y).size < 2:
            # Degenerate labels: no classifier can be fitted; predict uniform.
            self.constant_ = BASE_RATE
            self.info_["constant_labels"] = True
            return
        est = self._make_estimator()
        est.fit(flat_x, flat_y)
        self.estimator_ = est
        self.positive_index_ = list(est.classes_).index(1)

    def _predict(self, X):
        if self.estimator_ is None:
            return np.full(X.shape[:2], self.constant_ if self.constant_ is not None else BASE_RATE)
        cols = self._ml_columns()
        flat = X[:, :, cols].reshape(-1, len(cols))
        proba = self.estimator_.predict_proba(flat)[:, self.positive_index_]
        return proba.reshape(X.shape[0], X.shape[1])


class LogisticModel(SharedClassifierModel):
    name = "logistic"

    @classmethod
    def default_params(cls):
        return {"C": 0.1, "max_iter": 1000}

    def _make_estimator(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), LogisticRegression(**self.params))


class RandomForestModel(SharedClassifierModel):
    name = "random_forest"

    @classmethod
    def default_params(cls):
        return {"n_estimators": 100, "max_depth": 6, "min_samples_leaf": 20, "n_jobs": 1}

    def _make_estimator(self):
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(random_state=self.seed, **self.params)


class ExtraTreesModel(SharedClassifierModel):
    name = "extra_trees"

    @classmethod
    def default_params(cls):
        return {"n_estimators": 100, "max_depth": 6, "min_samples_leaf": 20, "n_jobs": 1}

    def _make_estimator(self):
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(random_state=self.seed, **self.params)


class GradientBoostingModel(SharedClassifierModel):
    name = "gradient_boosting"

    @classmethod
    def default_params(cls):
        return {"n_estimators": 60, "max_depth": 2, "learning_rate": 0.05, "subsample": 0.8}

    def _make_estimator(self):
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(random_state=self.seed, **self.params)


class HistGradientBoostingModel(SharedClassifierModel):
    name = "hist_gradient_boosting"

    @classmethod
    def default_params(cls):
        return {"max_iter": 100, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 40,
                "l2_regularization": 1.0, "early_stopping": False}

    def _make_estimator(self):
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(random_state=self.seed, **self.params)


class XGBoostModel(SharedClassifierModel):
    name = "xgboost"

    @classmethod
    def default_params(cls):
        return {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8,
                "tree_method": "hist", "n_jobs": 1, "verbosity": 0, "eval_metric": "logloss"}

    def _make_estimator(self):
        return importlib.import_module("xgboost").XGBClassifier(random_state=self.seed, **self.params)


class LightGBMModel(SharedClassifierModel):
    name = "lightgbm"

    @classmethod
    def default_params(cls):
        return {"n_estimators": 100, "num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 40,
                "n_jobs": 1, "verbose": -1}

    def _make_estimator(self):
        return importlib.import_module("lightgbm").LGBMClassifier(random_state=self.seed, **self.params)


class CatBoostModel(SharedClassifierModel):
    name = "catboost"

    @classmethod
    def default_params(cls):
        return {"iterations": 100, "depth": 4, "learning_rate": 0.05, "verbose": False,
                "thread_count": 1, "allow_writing_files": False}

    def _make_estimator(self):
        return importlib.import_module("catboost").CatBoostClassifier(random_seed=self.seed, **self.params)


@dataclass(frozen=True)
class ModelSpec:
    cls: type
    description: str
    package: str | None = None


MODEL_SPECS: dict[str, ModelSpec] = {
    "uniform": ModelSpec(UniformModel, "Analytical uniform baseline: every number 5/90."),
    "random": ModelSpec(RandomBaselineModel, "Reproducible random-ranking control (uniform probabilities)."),
    "frequency": ModelSpec(FrequencyModel, "Smoothed all-history frequency."),
    "ew_frequency": ModelSpec(EWFrequencyModel, "Exponentially weighted recent frequency."),
    "bayesian": ModelSpec(BayesianModel, "Beta-binomial posterior with empirical-Bayes prior strength."),
    "logistic": ModelSpec(LogisticModel, "Shared logistic regression over per-number features."),
    "random_forest": ModelSpec(RandomForestModel, "Shared random forest classifier."),
    "extra_trees": ModelSpec(ExtraTreesModel, "Shared extremely randomized trees."),
    "gradient_boosting": ModelSpec(GradientBoostingModel, "Shared sklearn gradient boosting."),
    "hist_gradient_boosting": ModelSpec(HistGradientBoostingModel, "Shared sklearn histogram gradient boosting."),
    "xgboost": ModelSpec(XGBoostModel, "Optional XGBoost classifier.", "xgboost"),
    "lightgbm": ModelSpec(LightGBMModel, "Optional LightGBM classifier.", "lightgbm"),
    "catboost": ModelSpec(CatBoostModel, "Optional CatBoost classifier.", "catboost"),
}
BASELINE_MODELS = ("uniform", "random")


def _optional_disabled() -> bool:
    return os.environ.get(OPTIONAL_DISABLE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def available_models(include_optional: bool = True) -> dict[str, dict]:
    """Registry: ``name -> {available, reason, kind, optional, package, description}``.

    ``available`` is a bool, ``reason`` a human-readable string ("available",
    "optional models disabled", "optional package 'x' not installed").
    Optional packages are discovered with ``importlib.util.find_spec`` only and
    can be disabled via ``include_optional=False`` or the environment variable
    ``OTOSLOTTO_DISABLE_OPTIONAL_MODELS=1``.
    """
    out = {}
    for name, spec in MODEL_SPECS.items():
        available, reason = True, "available"
        if spec.package:
            if not include_optional or _optional_disabled():
                available, reason = False, "optional models disabled"
            else:
                try:
                    found = importlib.util.find_spec(spec.package) is not None
                except (ImportError, ValueError):
                    found = False
                if not found:
                    available, reason = False, f"optional package '{spec.package}' not installed"
        out[name] = {
            "available": available,
            "reason": reason,
            "kind": spec.cls.kind,
            "optional": spec.package is not None,
            "package": spec.package,
            "description": spec.description,
        }
    return out


def create_model(name: str, feature_names=None, seed: int = 42, **params) -> BaseProbabilityModel:
    if name not in MODEL_SPECS:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODEL_SPECS)}")
    return MODEL_SPECS[name].cls(feature_names=feature_names, seed=seed, **params)
