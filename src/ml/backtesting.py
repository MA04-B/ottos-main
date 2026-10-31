"""Chronological walk-forward backtesting (expanding start, bounded window)."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.features.engineering import FeatureSet


class InsufficientDataError(ValueError):
    pass


@dataclass
class WalkForwardConfig:
    validation_draws: int = 20
    max_train_draws: int = 250
    refit_every: int | None = None
    n_folds: int = 4
    min_train_draws: int = 30
    warmup_draws: int = 10

    def __post_init__(self):
        if self.validation_draws < 1 or self.max_train_draws < 0 or self.n_folds < 1:
            raise ValueError("validation_draws/n_folds must be >= 1; max_train_draws must be >= 0")
        if self.warmup_draws < 0 or self.min_train_draws < 1:
            raise ValueError("warmup_draws must be >= 0 and min_train_draws >= 1")
        if self.refit_every is not None and self.refit_every < 1:
            raise ValueError("refit_every must be >= 1 when set")

    def refit_interval(self, n_eval: int) -> int:
        if self.refit_every:
            return max(1, int(self.refit_every))
        return max(1, math.ceil(n_eval / self.n_folds))


def evaluation_indices(n_draws: int, cfg: WalkForwardConfig) -> np.ndarray:
    start = max(cfg.min_train_draws, cfg.warmup_draws + 1, n_draws - cfg.validation_draws)
    if start >= n_draws:
        raise InsufficientDataError(
            f"insufficient history: {n_draws} draws, need more than {max(cfg.min_train_draws, cfg.warmup_draws + 1)}"
        )
    return np.arange(start, n_draws)


def training_window(fit_end: int, cfg: WalkForwardConfig) -> tuple[int, int]:
    """Rows [lo, hi) usable for a model that predicts draw ``fit_end``."""
    start = fit_end - cfg.max_train_draws if cfg.max_train_draws else cfg.warmup_draws
    return max(cfg.warmup_draws, start), fit_end


def walk_forward(fs: FeatureSet, factory: Callable, cfg: WalkForwardConfig, *, in_sample_draws: int = 50,
                 progress: Callable[[float], None] | None = None) -> dict:
    """Refit every ``refit_interval`` draws using only strictly earlier labels."""
    idx = evaluation_indices(fs.n_draws, cfg)
    step = cfg.refit_interval(len(idx))
    probs = np.empty((len(idx), fs.X.shape[1]))
    rankings = np.empty_like(probs)
    folds, ins_p, ins_y = [], [], []
    starts = list(range(0, len(idx), step))
    for f, offset in enumerate(starts):
        block = idx[offset:offset + step]
        lo, hi = training_window(int(block[0]), cfg)
        if hi > block[0]:
            raise AssertionError("training window overlaps evaluation block")
        model = factory()
        model.fit(fs.X[lo:hi], fs.Y[lo:hi])
        probs[offset:offset + len(block)] = model.predict_proba(fs.X[block])
        rankings[offset:offset + len(block)] = (
            model.ranking_scores(fs.X[block]) if hasattr(model, "ranking_scores")
            else probs[offset:offset + len(block)])
        tail = max(lo, hi - in_sample_draws)
        ins_p.append(model.predict_proba(fs.X[tail:hi]))
        ins_y.append(fs.Y[tail:hi])
        folds.append({"fold": f + 1, "train_start": lo, "train_end": hi, "eval_start": int(block[0]),
                      "eval_end": int(block[-1]) + 1})
        if progress:
            progress((f + 1) / len(starts))
    return {"indices": idx, "probs": probs, "ranking_scores": rankings, "actual": fs.Y[idx], "folds": folds,
            "in_sample_probs": np.vstack(ins_p), "in_sample_actual": np.vstack(ins_y)}
