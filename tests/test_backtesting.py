import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import hypergeom

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.engineering import BASE_RATE, build_features  # noqa: E402
from src.ml.backtesting import InsufficientDataError, WalkForwardConfig, walk_forward  # noqa: E402
from src.ml.metrics import (  # noqa: E402
    EXPECTED_MATCHES,
    UNIFORM_BRIER,
    UNIFORM_LOG_LOSS,
    critical_total,
    evaluate_predictions,
    exact_pvalue,
    holm_adjust,
    match_pmf,
    total_matches_null_pmf,
)
from src.ml.models import BaseProbabilityModel, create_model  # noqa: E402
from src.ml.training import train_experiment  # noqa: E402


def frame_from(draws, undated=0):
    start = dt.date(2015, 1, 3)
    rows = []
    for i, nums in enumerate(draws):
        d = start + dt.timedelta(days=7 * i)
        iso = d.isocalendar()
        rows.append({"year": iso[0], "week": iso[1], "draw_date": None if i < undated else d.isoformat(),
                     **{f"number_{j + 1}": int(n) for j, n in enumerate(sorted(nums))}})
    return pd.DataFrame(rows)


def random_frame(n=80, seed=0):
    rng = np.random.default_rng(seed)
    return frame_from([rng.choice(np.arange(1, 91), 5, replace=False) for _ in range(n)])


def cyclic_frame(n=80, undated=0):
    """Deterministic cycle of 18 blocks: numbers with the longest gap come next."""
    return frame_from([np.arange(5 * (i % 18) + 1, 5 * (i % 18) + 6) for i in range(n)], undated=undated)


class SpyModel(BaseProbabilityModel):
    name = "spy"
    log: list = []

    def _fit(self, X, Y):
        self.train_max = int(X[:, 0, self._col("history")].max())
        self.train_rows = len(X)

    def _predict(self, X):
        SpyModel.log.append((self.train_max, int(X[:, 0, self._col("history")].min()), self.train_rows))
        return np.full(X.shape[:2], BASE_RATE)


def test_match_pmf_and_null():
    pmf = match_pmf()
    assert pmf.sum() == pytest.approx(1.0)
    assert (np.arange(6) * pmf).sum() == pytest.approx(EXPECTED_MATCHES)
    null = total_matches_null_pmf(7)
    assert null.sum() == pytest.approx(1.0)
    assert (np.arange(len(null)) * null).sum() == pytest.approx(7 * 25 / 90)
    assert exact_pvalue(1, 1) == pytest.approx(1 - hypergeom.pmf(0, 90, 5, 5))
    assert exact_pvalue(0, 10) == 1.0
    assert exact_pvalue(51, 10) == 0.0
    crit = critical_total(20)
    assert exact_pvalue(crit, 20) <= 0.05 < exact_pvalue(crit - 1, 20)


def test_holm_adjust():
    adj = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03, "d": None})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)
    assert adj["d"] is None


def test_walk_forward_no_leakage_and_bounded_window():
    SpyModel.log = []
    fs = build_features(random_frame(90))
    cfg = WalkForwardConfig(validation_draws=20, max_train_draws=25, n_folds=4)
    out = walk_forward(fs, lambda: SpyModel(fs.feature_names), cfg, in_sample_draws=0)
    assert len(out["folds"]) == 4 and len(out["probs"]) == 20
    assert list(out["indices"]) == list(range(70, 90))
    for fold in out["folds"]:
        assert fold["train_end"] == fold["eval_start"]
        assert fold["train_end"] - fold["train_start"] <= 25
    for train_max, pred_min, rows in SpyModel.log:
        if rows:
            assert train_max < pred_min


def test_insufficient_history():
    fs = build_features(random_frame(20))
    with pytest.raises(InsufficientDataError):
        walk_forward(fs, lambda: create_model("uniform"), WalkForwardConfig(validation_draws=5))


def test_uniform_metrics_are_analytic():
    actual = build_features(random_frame(30)).Y[10:]
    probs = np.full(actual.shape, BASE_RATE)
    m = evaluate_predictions(probs, actual, analytic_uniform=True, n_bootstrap=50)
    assert m["mean_matches"] == pytest.approx(25 / 90)
    assert m["brier"] == pytest.approx(UNIFORM_BRIER)
    assert m["log_loss"] == pytest.approx(UNIFORM_LOG_LOSS, rel=1e-6)
    assert m["p_value"] is None
    assert sum(m["match_histogram"]) == pytest.approx(len(actual))
    lo, hi = m["mean_matches_ci"]
    assert lo <= 25 / 90 <= hi


def test_perfect_predictions_metrics():
    actual = build_features(random_frame(30)).Y[10:]
    probs = np.where(actual == 1, 0.9, 0.01)
    m = evaluate_predictions(probs, actual, n_bootstrap=50, n_periods=3)
    assert m["mean_matches"] == 5.0 and m["match_histogram"][5] == len(actual)
    assert m["p_value"] < 1e-20 and m["brier_skill"] > 0
    assert len(m["periods"]) == 3 and m["calibration"]["bins"]
    assert m["precision_at_5"] == m["recall_at_5"] == 1.0
    assert m["mean_matches_ci"] == [5.0, 5.0]


def test_random_baseline_reproducible():
    fs = build_features(random_frame(60))
    cfg = WalkForwardConfig(validation_draws=10)
    a = walk_forward(fs, lambda: create_model("random", fs.feature_names, 7), cfg)["ranking_scores"]
    b = walk_forward(fs, lambda: create_model("random", fs.feature_names, 7), cfg)["ranking_scores"]
    c = walk_forward(fs, lambda: create_model("random", fs.feature_names, 8), cfg)["ranking_scores"]
    assert np.array_equal(a, b)
    assert not np.array_equal(np.argsort(-a, axis=1)[:, :5], np.argsort(-c, axis=1)[:, :5])


def test_anti_predictive_model_not_selected():
    result = train_experiment(cyclic_frame(80), ["ew_frequency"], validation_draws=20, n_bootstrap=100, save=False)
    by = {r["model"]: r for r in result["results"]}
    assert by["ew_frequency"]["mean_matches"] == 0.0
    assert by["ew_frequency"]["significant"] is False
    assert result["selected_model"] == "uniform"
    assert result["scores"] == pytest.approx([5 / 90] * 90)
    assert any("defaulting to uniform" in w for w in result["warnings"])


def test_genuine_signal_selected_with_holm():
    result = train_experiment(cyclic_frame(80, undated=40), ["ew_frequency", "logistic"], validation_draws=20,
                              n_bootstrap=100, save=False, min_selection_draws=5)
    by = {r["model"]: r for r in result["results"]}
    assert (by["logistic"]["significant"]
            and by["logistic"]["p_value_holm"] >= by["logistic"]["selection_metrics"]["p_value"])
    assert result["selected_model"] == "logistic"
    top = np.argsort(result["scores"])[::-1][:5] + 1
    assert sorted(top.tolist()) == [41, 42, 43, 44, 45]  # draw 80 -> block 80 % 18 = 8
