import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prediction.generator import STRATEGIES, generate_combinations  # noqa: E402

KEYS = {"numbers", "strategy", "scores", "explanation", "number_details", "sum", "odd", "spread", "model", "seed"}


def make_frame(n=60, seed=0, undated=20):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        d = dt.date(2021, 1, 2) + dt.timedelta(days=7 * i)
        iso = d.isocalendar()
        nums = sorted(rng.choice(np.arange(1, 91), 5, replace=False).tolist())
        rows.append({"year": iso[0], "week": iso[1], "draw_date": None if i < undated else d.isoformat(),
                     **{f"number_{j + 1}": nums[j] for j in range(5)}})
    return pd.DataFrame(rows)


def skewed_scores():
    s = np.full(90, 0.04)
    s[:10] = 0.2
    return s.tolist()


def test_strategy_names():
    assert set(STRATEGIES) == {"mixed", "ranked", "weighted", "historical", "recent", "bayesian", "balanced",
                               "ensemble", "diversified"}


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("diversity", [0.0, 0.5, 1.0])
def test_valid_unique_combinations(strategy, diversity):
    combos = generate_combinations(make_frame(), scores=skewed_scores(), strategy=strategy, diversity=diversity,
                                   model_name="frequency")
    assert len(combos) == 10
    assert len({tuple(c["numbers"]) for c in combos}) == 10
    for c in combos:
        nums = c["numbers"]
        assert KEYS <= set(c)
        assert nums == sorted(nums) and len(set(nums)) == 5 and all(1 <= n <= 90 for n in nums)
        assert all(isinstance(n, int) for n in nums)
        assert c["sum"] == sum(nums) and c["odd"] == sum(n % 2 for n in nums) and c["spread"] == nums[-1] - nums[0]
        assert len(c["scores"]) == 5 and len(c["number_details"]) == 5
        assert [d["number"] for d in c["number_details"]] == nums
        assert c["seed"] == 42 and "1 in 43,949,268" in c["explanation"]
        if strategy != "mixed":
            assert c["strategy"] == strategy


def test_mixed_cycles_strategies():
    combos = generate_combinations(make_frame(), strategy="mixed")
    assert {c["strategy"] for c in combos} == set(STRATEGIES) - {"mixed"}
    assert all(c["requested_strategy"] == "mixed" for c in combos)


def test_reproducible():
    frame = make_frame()
    a = generate_combinations(frame, scores=skewed_scores(), seed=5)
    b = generate_combinations(frame, scores=skewed_scores(), seed=5)
    c = generate_combinations(frame, scores=skewed_scores(), seed=6)
    assert [x["numbers"] for x in a] == [x["numbers"] for x in b]
    assert [x["numbers"] for x in a] != [x["numbers"] for x in c]


def test_uniform_default_and_honesty():
    combos = generate_combinations(None, scores=None, model_name="random_forest", strategy="ranked")
    assert all(c["model"] == "uniform" for c in combos)
    assert all(s == pytest.approx(5 / 90) for c in combos for s in c["scores"])
    assert "uniform scores were used" in combos[0]["explanation"]


def test_ranked_first_is_top_scores():
    combos = generate_combinations(make_frame(), scores=skewed_scores(), strategy="ranked", diversity=0.0)
    assert set(combos[0]["numbers"]) <= set(range(1, 11))


def test_ensemble_labelling():
    single = generate_combinations(make_frame(), scores=skewed_scores(), strategy="ensemble", model_name="logistic")
    assert all(c["model"] == "logistic" for c in single)
    assert "not a multi-model ensemble" in single[0]["explanation"]
    multi = generate_combinations(make_frame(), scores=skewed_scores(), strategy="ensemble", model_name="logistic",
                                  ensemble_scores={"logistic": skewed_scores(), "bayesian": [1.0] * 90})
    assert all(c["model"] == "ensemble(logistic+bayesian)" for c in multi)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        generate_combinations(None, scores=[1.0] * 89)
    with pytest.raises(ValueError):
        generate_combinations(None, scores=[-1.0] + [1.0] * 89)
    with pytest.raises(ValueError):
        generate_combinations(None, scores=[float("nan")] * 90)
    with pytest.raises(ValueError):
        generate_combinations(None, strategy="magic")
