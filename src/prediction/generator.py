"""Generate 10 unique 5-number combinations from (possibly uniform) scores.

These are presentation strategies, not predictions with demonstrated skill:
every combination has the same 1 / C(90, 5) jackpot probability.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral

import numpy as np
import pandas as pd

from src.features.engineering import BASE_RATE, N_NUMBERS, PICK, number_statistics

STRATEGIES = ("mixed", "ranked", "weighted", "historical", "recent", "bayesian", "balanced", "ensemble", "diversified")
MIXED_SEQUENCE = ("ranked", "weighted", "historical", "recent", "bayesian", "balanced", "ensemble", "diversified",
                  "weighted", "balanced")
N_COMBINATIONS = 10
TOTAL_COMBINATIONS = math.comb(N_NUMBERS, PICK)
SUM_RANGE = (171, 284)  # about mean 227.5 +/- 1 sd of a uniform 5-of-90 sum
BAYES_PRIOR_STRENGTH = 50.0
DISCLAIMER = (f"Draws are random: this combination, like every other, has a 1 in {TOTAL_COMBINATIONS:,} jackpot "
              "chance; the strategy does not improve the odds.")
DESCRIPTIONS = {
    "ranked": "highest model scores (first) or samples from the top-ranked pool",
    "weighted": "sampled using smoothed model scores, temperature adjustment and diversity penalties",
    "historical": "sampled proportionally to smoothed all-history frequency",
    "recent": "sampled proportionally to exponentially weighted recent frequency",
    "bayesian": "sampled from a posterior blending model scores (prior) with the last draws' counts",
    "balanced": "score-weighted sample constrained to 2-3 odd, 2-3 low (<=45), >=3 decades and a central sum",
    "diversified": "score-weighted sample with a strong penalty on numbers already used, to maximise coverage",
}


def _as_scores(values, what: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.shape != (N_NUMBERS,):
        raise ValueError(f"{what} must contain exactly {N_NUMBERS} values")
    if not np.all(np.isfinite(arr)) or (arr < 0).any():
        raise ValueError(f"{what} must be finite and non-negative")
    if arr.max() <= 0:
        raise ValueError(f"{what} must not be all zero")
    return arr


def _unit(w) -> np.ndarray:
    w = np.clip(np.asarray(w, dtype=np.float64), 0.0, None)
    w = w / max(float(w.max()), 1e-12) + 1e-12
    return w / w.sum()


def _balance_violations(nums) -> int:
    nums = np.asarray(nums)
    odd = int((nums % 2).sum())
    low = int((nums <= 45).sum())
    total = int(nums.sum())
    decades = len(set(((nums - 1) // 10).tolist()))
    return (odd not in (2, 3)) + (low not in (2, 3)) + (not SUM_RANGE[0] <= total <= SUM_RANGE[1]) + (decades < 3)


class _Context:
    def __init__(self, model, stats, ensemble, ensemble_label, diversity, rng):
        self.model = model
        self.stats = stats
        self.diversity = diversity
        self.rng = rng
        self.usage = np.zeros(N_NUMBERS)
        self.ranked_calls = 0
        W = stats["recent_window"]
        prior = _unit(model) * PICK
        self.weights = {
            "ranked": _unit(model),
            "weighted": _unit(model),
            "balanced": _unit(model),
            "diversified": _unit(model),
            "historical": _unit(stats["frequency"]),
            "recent": _unit(stats["ew_frequency"]),
            "bayesian": _unit((BAYES_PRIOR_STRENGTH * prior + stats["recent_counts"]) / (BAYES_PRIOR_STRENGTH + W)),
            "ensemble": _unit(ensemble),
        }
        self.ensemble_label = ensemble_label

    def _sample(self, w, penalty=1.0, pool=None):
        temperature = 0.5 + 1.5 * self.diversity
        p = np.power(w, 1.0 / temperature) * np.exp(-self.diversity * penalty * self.usage)
        if pool is not None:
            mask = np.zeros(N_NUMBERS, dtype=bool)
            mask[pool] = True
            p = np.where(mask, p, 0.0)
        if pool is not None:
            p /= p.sum()
        else:
            p = _unit(p)
        return np.sort(self.rng.choice(N_NUMBERS, PICK, replace=False, p=p) + 1)

    def draw(self, strategy: str) -> np.ndarray:
        w = self.weights[strategy]
        if strategy == "ranked":
            order = np.lexsort((self.rng.random(N_NUMBERS), -w))
            self.ranked_calls += 1
            if self.ranked_calls == 1:
                return np.sort(order[:PICK] + 1)
            size = min(N_NUMBERS, 10 + int(round(30 * self.diversity)) + 2 * self.ranked_calls)
            return self._sample(w, pool=order[:size])
        if strategy == "balanced":
            best, best_v = None, 99
            for _ in range(500):
                cand = self._sample(w)
                v = _balance_violations(cand)
                if v < best_v:
                    best, best_v = cand, v
                if v == 0:
                    break
            if best_v:
                # Highly concentrated weights can make rejection sampling
                # miss valid balanced tickets. Retry with uniform proposals.
                for _ in range(2000):
                    cand = np.sort(self.rng.choice(N_NUMBERS, PICK, replace=False) + 1)
                    if _balance_violations(cand) == 0:
                        return cand
                raise RuntimeError("Could not generate a balanced combination")
            return best
        if strategy == "diversified":
            return self._sample(w, penalty=4.0)
        return self._sample(w)


def generate_combinations(frame=None, scores=None, seed: int = 42, diversity: float = 0.5, strategy: str = "mixed",
                          model_name: str = "uniform", ensemble_scores=None, count: int = N_COMBINATIONS) -> list[dict]:
    """Return ``count`` (default 10) unique, sorted 5-number combinations.

    ``scores``: 90 non-negative per-number scores (index 0 = number 1); None
    means uniform. ``ensemble_scores``: optional mapping name -> 90 scores (or
    list of score lists); with >= 2 sources the "ensemble" strategy averages
    them, otherwise it is an explicitly labelled heuristic blend.
    """
    strategy = str(strategy).strip().lower()
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose one of {STRATEGIES}")
    diversity = float(diversity)
    if not math.isfinite(diversity):
        raise ValueError("diversity must be finite")
    if not 0 <= diversity <= 1:
        raise ValueError("diversity must be between 0 and 1")
    if not isinstance(count, Integral) or isinstance(count, bool):
        raise ValueError("count must be an integer")
    count = int(count)
    if not 1 <= count <= 100:
        raise ValueError("count must be within 1..100")
    if not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    notes = []
    if scores is None:
        model = np.full(N_NUMBERS, BASE_RATE)
        if model_name != "uniform":
            notes.append(f"No scores were supplied for '{model_name}', so uniform scores were used.")
        model_name = "uniform"
    else:
        model = _as_scores(scores, "scores")
    is_uniform = bool(np.allclose(model, model.mean()))
    stats = number_statistics(frame)

    sources = {}
    if ensemble_scores is not None:
        items = ensemble_scores.items() if isinstance(ensemble_scores, Mapping) else (
            (f"source_{i + 1}", v) for i, v in enumerate(ensemble_scores))
        for name, values in items:
            sources[str(name)] = _unit(_as_scores(values, f"ensemble_scores[{name}]"))
    if len(sources) >= 2:
        ensemble = np.mean(list(sources.values()), axis=0)
        ensemble_label = f"ensemble({'+'.join(sources)})"
        ensemble_desc = f"average of {len(sources)} model score vectors: {', '.join(sources)}"
    else:
        ensemble = (_unit(model) + _unit(stats["frequency"]) + _unit(stats["ew_frequency"])
                    + _unit((BAYES_PRIOR_STRENGTH * _unit(model) * PICK + stats["recent_counts"])
                            / (BAYES_PRIOR_STRENGTH + stats["recent_window"]))) / 4.0
        ensemble_label = None
        ensemble_desc = (f"heuristic blend of '{model_name}' scores with historical, recent and bayesian weights "
                         "(not a multi-model ensemble)")

    rng = np.random.default_rng(seed)
    ctx = _Context(model, stats, ensemble, ensemble_label, diversity, rng)
    plan = [MIXED_SEQUENCE[i % len(MIXED_SEQUENCE)] if strategy == "mixed" else strategy for i in range(count)]
    model_rank = np.empty(N_NUMBERS, dtype=np.int64)
    model_rank[np.argsort(-model, kind="stable")] = np.arange(1, N_NUMBERS + 1)
    seen: set[tuple[int, ...]] = set()
    out = []
    for i, sub in enumerate(plan):
        combo = None
        for _ in range(200):
            cand = tuple(int(x) for x in ctx.draw(sub))
            if cand not in seen:
                combo = cand
                break
        fallback = combo is None
        if fallback and sub == "balanced":
            raise RuntimeError("Could not generate ten unique balanced combinations")
        while combo is None:
            cand = tuple(sorted(int(x) for x in rng.choice(N_NUMBERS, PICK, replace=False) + 1))
            if cand not in seen:
                combo = cand
        seen.add(combo)
        idx = np.asarray(combo) - 1
        ctx.usage[idx] += 1
        weights = ctx.weights[sub]
        label = ensemble_label if (sub == "ensemble" and ensemble_label) else model_name
        desc = ensemble_desc if sub == "ensemble" else DESCRIPTIONS[sub]
        parts = [f"{sub.capitalize()} strategy: {desc}."]
        if strategy == "mixed":
            parts.append("Part of a mixed set cycling through strategies.")
        if is_uniform and sub in ("ranked", "weighted", "balanced", "diversified", "bayesian"):
            parts.append("Model scores are uniform (no validated predictive signal), so ordering among numbers is "
                         "effectively random.")
        elif sub not in ("historical", "recent"):
            parts.append(f"Scores come from model '{model_name}'.")
        if fallback:
            parts.append("Uniform random fallback was used to keep combinations unique.")
        if sub == "balanced":
            parts.append("Balanced constraints were verified; uniform proposals may be used if model weights are too concentrated.")
        parts.extend(notes)
        parts.append(DISCLAIMER)
        details = []
        for n in combo:
            k = n - 1
            since = int(stats["draws_since_seen"][k])
            details.append({
                "number": n,
                "score": float(model[k]),
                "rank": int(model_rank[k]),
                "strategy_weight": float(weights[k]),
                "historical_frequency": float(stats["frequency"][k]),
                "historical_count": int(stats["counts"][k]),
                "recent_frequency": float(stats["ew_frequency"][k]),
                "draws_since_seen": since if since >= 0 else None,
                "selection_reason": (f"Selected by {sub}; strategy weight {weights[k]:.6f}, "
                                     f"model rank {model_rank[k]}, {int(stats['counts'][k])} historical "
                                     "appearances. This is a heuristic explanation, not evidence of higher odds."),
            })
        out.append({
            "numbers": list(combo),
            "strategy": sub,
            "requested_strategy": strategy,
            "scores": [float(model[n - 1]) for n in combo],
            "explanation": " ".join(parts),
            "number_details": details,
            "sum": int(sum(combo)),
            "odd": int(sum(n % 2 for n in combo)),
            "spread": int(combo[-1] - combo[0]),
            "model": label,
            "seed": seed,
            "index": i + 1,
            "jackpot_probability": 1.0 / TOTAL_COMBINATIONS,
        })
    return out


def combinations_frame(combinations) -> pd.DataFrame:
    """Flat, reusable CSV shape: one row per ticket, one column per number."""
    return pd.DataFrame([{
        "combination": i, "model": c["model"], "strategy": c["strategy"], "seed": c["seed"],
        **{f"number_{j}": n for j, n in enumerate(c["numbers"], 1)},
        **{f"score_{j}": score for j, score in enumerate(c["scores"], 1)},
        "sum": c["sum"], "odd": c["odd"], "spread": c["spread"],
        "jackpot_probability": c.get("jackpot_probability", 1 / TOTAL_COMBINATIONS),
    } for i, c in enumerate(combinations, 1)])
