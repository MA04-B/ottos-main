"""Out-of-sample metrics, exact hypergeometric null tests and Holm correction."""

from __future__ import annotations

import math
from functools import lru_cache
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.stats import hypergeom

from src.features.engineering import BASE_RATE, N_NUMBERS, PICK
from src.ml.models import EPS

EXPECTED_MATCHES = PICK * PICK / N_NUMBERS  # 25/90
UNIFORM_BRIER = BASE_RATE * (1.0 - BASE_RATE)
UNIFORM_LOG_LOSS = -(BASE_RATE * math.log(BASE_RATE) + (1.0 - BASE_RATE) * math.log(1.0 - BASE_RATE))


def match_pmf() -> np.ndarray:
    """P(k matches) for a fixed 5-number pick against a uniform 5-of-90 draw."""
    return hypergeom.pmf(np.arange(PICK + 1), N_NUMBERS, PICK, PICK)


@lru_cache(maxsize=32)
def total_matches_null_pmf(n_draws: int) -> np.ndarray:
    """Exact distribution of total matches over independent draws (convolution)."""
    base = match_pmf()
    out = np.array([1.0])
    for _ in range(int(n_draws)):
        out = np.convolve(out, base)
    out.setflags(write=False)
    return out


def exact_pvalue(total_matches: int, n_draws: int) -> float:
    """One-sided exact P(S >= total_matches) under the uniform null."""
    if n_draws <= 0:
        return 1.0
    pmf = total_matches_null_pmf(n_draws)
    total = int(total_matches)
    if total <= 0:
        return 1.0
    if total >= len(pmf):
        return 0.0
    return float(min(1.0, max(0.0, pmf[total:].sum())))


def critical_total(n_draws: int, alpha: float = 0.05) -> int:
    """Smallest total matches whose exact one-sided p-value is <= alpha."""
    pmf = total_matches_null_pmf(n_draws)
    tail = np.cumsum(pmf[::-1])[::-1]
    hits = np.flatnonzero(tail <= alpha)
    return int(hits[0]) if len(hits) else len(pmf)


def holm_adjust(pvalues: Mapping[str, float | None]) -> dict[str, float | None]:
    """Holm-Bonferroni step-down adjusted p-values (None entries are skipped)."""
    valid = sorted(((p, k) for k, p in pvalues.items() if p is not None and np.isfinite(p)), key=lambda x: x[0])
    m = len(valid)
    adjusted: dict[str, float | None] = {k: None for k in pvalues}
    running = 0.0
    for i, (p, key) in enumerate(valid):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[key] = running
    return adjusted


def wilson_interval(successes: float, trials: float, confidence: float = 0.95) -> tuple[float, float]:
    if trials <= 0:
        return (0.0, 1.0)
    from scipy.stats import norm

    z = float(norm.ppf(0.5 + confidence / 2.0))
    phat = successes / trials
    denom = 1.0 + z * z / trials
    centre = (phat + z * z / (2 * trials)) / denom
    half = z * math.sqrt(phat * (1 - phat) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def calibration_table(probs, actual, n_bins: int = 10) -> dict:
    p = np.asarray(probs, dtype=np.float64).ravel()
    y = np.asarray(actual, dtype=np.float64).ravel()
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        edges = np.array([p.min(), p.max()])
    which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    bins, ece = [], 0.0
    for b in range(len(edges) - 1):
        mask = which == b
        if not mask.any():
            continue
        mp, obs, cnt = float(p[mask].mean()), float(y[mask].mean()), int(mask.sum())
        ece += cnt / len(p) * abs(mp - obs)
        bins.append({"bin_low": float(edges[b]), "bin_high": float(edges[b + 1]), "mean_predicted": mp,
                     "observed_rate": obs, "count": cnt})
    return {"bins": bins, "expected_calibration_error": float(ece)}


def top_k_matches(probs, actual, k: int = PICK) -> np.ndarray:
    top = np.argsort(-np.asarray(probs), axis=1, kind="stable")[:, :k]
    return np.take_along_axis(np.asarray(actual), top, axis=1).sum(axis=1)


def evaluate_predictions(probs, actual, *, seed: int = 42, n_bootstrap: int = 1000, n_periods: int = 4,
                         labels: Sequence[str] | None = None, analytic_uniform: bool = False,
                         confidence: float = 0.95, random_uniform: bool = False, ranking_scores=None,
                         alpha: float = 0.05) -> dict:
    """Score out-of-sample probabilities against realised draws.

    ``analytic_uniform`` replaces tie-dependent top-5 statistics with their
    exact uniform-null values (used for the uniform baseline).
    """
    probs = np.asarray(probs, dtype=np.float64)
    actual = np.asarray(actual)
    if probs.ndim != 2 or probs.shape != actual.shape or probs.shape[1] != N_NUMBERS:
        raise ValueError("Prediction and label arrays must have matching (draws,90) shapes")
    if not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
        raise ValueError("Probabilities must be finite and between 0 and 1")
    if not np.isin(actual, [0, 1]).all() or (actual.sum(axis=1) != PICK).any():
        raise ValueError("Each drawing must have five binary positive labels")
    actual = actual.astype(np.int64)
    if not 0 < confidence < 1 or not 0 < alpha < 1 or n_bootstrap < 1 or n_periods < 1:
        raise ValueError("Invalid confidence, bootstrap count or period count")
    ranking_probs = probs if ranking_scores is None else np.asarray(ranking_scores)
    if ranking_probs.shape != probs.shape or not np.isfinite(ranking_probs).all():
        raise ValueError("Ranking scores must be finite and match prediction shape")
    probs = np.full_like(probs, BASE_RATE) if random_uniform else np.clip(probs, EPS, 1.0 - EPS)
    V = int(len(probs))
    if V == 0:
        raise ValueError("no evaluation draws")
    lo_q, hi_q = (1 - confidence) / 2 * 100, (1 + confidence) / 2 * 100
    rng = np.random.default_rng(seed)
    cell_brier = (probs - actual) ** 2
    cell_ll = -(actual * np.log(probs) + (1 - actual) * np.log(1 - probs))
    draw_brier = cell_brier.mean(axis=1)
    draw_ll = cell_ll.mean(axis=1)
    brier, log_loss = float(draw_brier.mean()), float(draw_ll.mean())
    boot = rng.integers(0, V, size=(max(1, n_bootstrap), V))
    bd = (draw_brier - UNIFORM_BRIER)[boot].mean(axis=1)
    pmf = match_pmf()
    null = total_matches_null_pmf(V)
    if analytic_uniform:
        matches = None
        mean_matches = EXPECTED_MATCHES
        histogram = (pmf * V).tolist()
        total = None
        p_value = None
        cdf = np.cumsum(null)
        ci = (float(np.searchsorted(cdf, (1 - confidence) / 2) / V), float(np.searchsorted(cdf, (1 + confidence) / 2) / V))
        top10_recall = 10 / N_NUMBERS
        hit_ci = (BASE_RATE, BASE_RATE)
    else:
        matches = top_k_matches(ranking_probs, actual)
        mean_matches = float(matches.mean())
        histogram = np.bincount(matches, minlength=PICK + 1)[: PICK + 1].astype(int).tolist()
        total = int(matches.sum())
        p_value = exact_pvalue(total, V)
        bm = matches[boot].mean(axis=1)
        ci = (float(np.percentile(bm, lo_q)), float(np.percentile(bm, hi_q)))
        top10_recall = float(top_k_matches(ranking_probs, actual, 10).mean() / PICK)
        hit_ci = (ci[0] / PICK, ci[1] / PICK)
    periods = []
    for i, part in enumerate(np.array_split(np.arange(V), min(max(1, n_periods), V))):
        entry = {"period": i + 1, "start": int(part[0]), "end": int(part[-1]), "draws": int(len(part)),
                 "brier": float(draw_brier[part].mean()), "log_loss": float(draw_ll[part].mean()),
                 "mean_matches": EXPECTED_MATCHES if matches is None else float(matches[part].mean())}
        if labels is not None:
            entry["first_label"], entry["last_label"] = labels[part[0]], labels[part[-1]]
        periods.append(entry)
    crit = critical_total(V, alpha)
    return {
        "n_draws": V,
        "mean_matches": mean_matches,
        "expected_matches": EXPECTED_MATCHES,
        "lift": mean_matches / EXPECTED_MATCHES,
        "total_matches": total,
        "match_histogram": histogram,
        "null_match_histogram": (pmf * V).tolist(),
        "precision_at_5": mean_matches / PICK,
        "recall_at_5": mean_matches / PICK,
        "recall_at_10": top10_recall,
        "hit_rate_ci": [float(hit_ci[0]), float(hit_ci[1])],
        "hit_rate_ci_kind": "known uniform marginal" if analytic_uniform else "draw-level bootstrap percentile",
        "mean_matches_ci": [ci[0], ci[1]],
        "mean_matches_ci_kind": "uniform null band" if analytic_uniform else "bootstrap percentile",
        "ci_caveat": ("Bootstrap intervals are descriptive and unstable for small samples or constant outcomes. "
                      "The analytical uniform interval is a null prediction band, not an estimated-skill interval."),
        "brier": brier,
        "uniform_brier": UNIFORM_BRIER,
        "brier_skill": 1.0 - brier / UNIFORM_BRIER,
        "brier_diff_ci": [float(np.percentile(bd, lo_q)), float(np.percentile(bd, hi_q))],
        "log_loss": log_loss,
        "uniform_log_loss": UNIFORM_LOG_LOSS,
        "log_loss_diff": log_loss - UNIFORM_LOG_LOSS,
        "calibration": calibration_table(probs, actual),
        "periods": periods,
        "p_value": p_value,
        "test": "exact one-sided hypergeometric (convolution) test of total top-5 matches vs uniform",
        "critical_total_matches": crit,
        "critical_mean_matches": crit / V,
        "confidence": confidence,
        "critical_alpha": alpha,
    }
