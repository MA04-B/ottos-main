import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.stats import binomtest, hypergeom

from src.analysis.statistics import analyze
from src.data.audit import audit_html
from src.data.database import Database
from src.data.parser import ParseError, parse_html
from src.data.scraper import sync_database
from src.features.engineering import BASE_RATE, build_features, validate_frame
from src.ml.backtesting import WalkForwardConfig, training_window
from src.ml.metrics import UNIFORM_BRIER, UNIFORM_LOG_LOSS, critical_total, evaluate_predictions, holm_adjust
from src.ml.models import create_model, normalize_probabilities
from src.ml.persistence import load_latest
from src.ml.training import train_experiment
from src.prediction.generator import STRATEGIES, _balance_violations, generate_combinations
from src.prediction.generator import combinations_frame
from test_data import html, record
from test_models import make_frame


def test_all_features_equal_prefix_prediction_and_ignore_target_metadata():
    frame = make_frame(65, undated=15)
    full = build_features(frame)
    for t in (1, 10, 16, 35, 64):
        prefix = build_features(frame.iloc[:t])
        np.testing.assert_array_equal(full.X[t], prefix.X[-1])
    changed = frame.copy()
    changed.loc[35:, "draw_date"] = None
    np.testing.assert_array_equal(full.X[:36], build_features(changed).X[:36])


def test_every_historical_feature_matches_manual_past():
    frame = make_frame(65)
    fs = build_features(frame)
    t = 40
    y = fs.Y[:t]
    counts = y.sum(axis=0)
    np.testing.assert_allclose(fs.X[t, :, fs.column("count_all")], counts)
    for window in (10, 25, 50):
        np.testing.assert_allclose(fs.X[t, :, fs.column(f"freq_w{window}")], y[-window:].mean(axis=0))
    for name, half in (("ew_fast", 10), ("ew_mid", 30), ("ew_slow", 100)):
        ew = np.full(90, BASE_RATE)
        decay = .5 ** (1 / half)
        for draw in y:
            ew = decay * ew + (1 - decay) * draw
        np.testing.assert_allclose(fs.X[t, :, fs.column(name)], ew)
    previous = np.flatnonzero(y[-1])
    pair = y.T.astype(float) @ y.astype(float)
    expected = []
    for number in range(90):
        partners = [p for p in previous if p != number]
        expected.append(np.mean([pair[p, number] / counts[p] for p in partners]))
    np.testing.assert_allclose(fs.X[t, :, fs.column("pair_score")], expected)
    for number in range(90):
        positions = np.flatnonzero(y[:, number])
        gap = t - 1 - positions[-1] if len(positions) else t
        assert fs.X[t, number, fs.column("gap")] == pytest.approx(min(gap, 300) / 18)


def test_conflicting_frame_keys_never_enter_features():
    frame = make_frame(5)
    duplicate = frame.iloc[[0]].copy()
    duplicate["number_1"] = 1 if duplicate.iloc[0].number_1 != 1 else 2
    import pandas as pd
    with pytest.raises(ValueError):
        validate_frame(pd.concat([frame, duplicate]))


@pytest.mark.parametrize("weights", [
    np.r_[1., np.zeros(89)], np.r_[np.ones(4), np.zeros(86)],
    np.geomspace(1e-200, 1e200, 90), np.ones(90),
])
def test_capped_probabilities_sum_to_five(weights):
    p = normalize_probabilities(weights)
    assert p.sum() == pytest.approx(5., abs=1e-10)
    assert ((p > 0) & (p < 1)).all()
    assert np.isfinite(p).all()


@pytest.mark.parametrize("weights", [[float("nan")] * 90, [-1.] * 90, [0.] * 90, [1.] * 89])
def test_invalid_probabilities_are_not_silently_repaired(weights):
    with pytest.raises(ValueError):
        normalize_probabilities(weights)


def test_full_history_window_and_final_test_independence():
    assert training_window(100, WalkForwardConfig(max_train_draws=0)) == (10, 100)
    original = make_frame(100, seed=9)
    changed = original.copy()
    changed.loc[90:, [f"number_{i}" for i in range(1, 6)]] = [1, 2, 3, 4, 5]
    options = dict(validation_draws=20, training_window=0, n_bootstrap=30, save=False,
                   min_selection_draws=5)
    a = train_experiment(original, ["frequency", "logistic"], **options)
    b = train_experiment(changed, ["frequency", "logistic"], **options)
    assert a["selection_end_index"] == 89
    assert a["final_test_start_index"] == 90
    assert a["selected_model"] == b["selected_model"]
    for left, right in zip(a["results"], b["results"]):
        assert left["selection_metrics"] == right["selection_metrics"]
        assert left["p_value_holm"] == right["p_value_holm"]
        assert left["eligible"] == right["eligible"]
    assert any(x["final_test_metrics"] != y["final_test_metrics"] for x, y in zip(a["results"], b["results"]))


def test_random_control_uses_exact_uniform_probability_metrics():
    fs = build_features(make_frame(30))
    model = create_model("random", fs.feature_names, 17).fit(fs.X[:10], fs.Y[:10])
    np.testing.assert_allclose(model.predict_proba(fs.X[10:20]), BASE_RATE, rtol=0, atol=1e-16)
    rng = np.random.default_rng(17)
    ranks = BASE_RATE * (1 + 1e-6 * rng.random(fs.Y.shape))
    result = evaluate_predictions(ranks, fs.Y, random_uniform=True, n_bootstrap=50)
    assert result["brier"] == pytest.approx(UNIFORM_BRIER)
    assert result["log_loss"] == pytest.approx(UNIFORM_LOG_LOSS)
    assert result["hit_rate_ci"] == pytest.approx(np.array(result["mean_matches_ci"]) / 5)
    assert result["hit_rate_ci_kind"] == "draw-level bootstrap percentile"


def test_statistics_against_independent_combinatorics():
    frame = make_frame(60)
    stats = analyze(frame)
    draws = frame[[f"number_{i}" for i in range(1, 6)]].to_numpy()
    assert stats["pairs"]["count"].sum() == 60 * 10
    assert stats["triples"]["count"].sum() == 60 * 10
    assert stats["pairs"]["expected"].iloc[0] == pytest.approx(60 * hypergeom.pmf(2, 90, 2, 5))
    freq = stats["frequency"]
    expected_p = {str(n): binomtest(int(c), 60, 5 / 90).pvalue for n, c in zip(freq.number, freq["count"])}
    corrected = holm_adjust(expected_p)
    np.testing.assert_allclose(freq.holm_p_value, [corrected[str(n)] for n in freq.number])
    np.testing.assert_array_equal(stats["distribution"]["sum"], draws.sum(axis=1))
    np.testing.assert_array_equal(stats["distribution"]["odd"], (draws % 2).sum(axis=1))
    for n in range(1, 91):
        pos = np.flatnonzero((draws == n).any(axis=1))
        gaps = np.diff(pos) - 1
        row = stats["gaps"].iloc[n - 1]
        assert row.completed_gap_distribution == gaps.tolist()
        assert row.since_last == (59 - pos[-1] if len(pos) else 60)


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("diversity", [0., 1.])
def test_all_strategies_extreme_scores_and_constraints(strategy, diversity):
    scores = np.r_[1., np.zeros(89)]
    a = generate_combinations(make_frame(30), scores, strategy=strategy, diversity=diversity, seed=7)
    b = generate_combinations(make_frame(30), scores, strategy=strategy, diversity=diversity, seed=7)
    assert a == b
    assert len(a) == len({tuple(p["numbers"]) for p in a}) == 10
    for p in a:
        assert len(set(p["numbers"])) == 5 and p["numbers"] == sorted(p["numbers"])
        assert min(p["numbers"]) >= 1 and max(p["numbers"]) <= 90
        if p["strategy"] == "balanced":
            assert _balance_violations(p["numbers"]) == 0
        assert all("selection_reason" in n and "historical_count" in n for n in p["number_details"])


def test_prize_errors_do_not_discard_valid_drawing():
    records, issues = parse_html(html([[2024, 10, "2024.03.09.", 0, "broken Ft", 1, 5, 7, 20, 90]]))
    assert len(records) == 1
    assert records[0]["prizes"]["5 találat (Ft)"] is None
    assert any("Malformed prize" in issue["message"] for issue in issues)


def test_same_numbers_with_conflicting_dates_are_not_duplicates():
    row = [2024, 10, "2024.03.09.", 0, "0 Ft", 1, 5, 7, 20, 90]
    other = [2024, 11, "2024.03.09.", *row[3:]]
    with pytest.raises(ParseError, match="Conflicting"):
        parse_html(html([row, other]))


def test_single_missing_saved_row_aborts_source_reconciliation(tmp_path, monkeypatch):
    db = Database(tmp_path / "test.db")
    db.import_records([record(), record(week=11, draw_date="2024-03-16")], [])
    response = MagicMock(status_code=200, headers={})
    response.content = html([[2024, 10, "2024.03.09.", 0, "0 Ft", 1, 5, 7, 20, 90]]).encode()
    monkeypatch.setattr("src.data.scraper.RAW", tmp_path / "raw")
    with patch("requests.Session.get", return_value=response), pytest.raises(ValueError, match="omits 1 saved"):
        sync_database(db, force=True)
    assert len(db.draws()) == 2
    assert "omits 1 saved" in db.state()["last_error"]


def test_corrupt_artifact_is_reported_and_old_engine_is_stale(tmp_path):
    frame = make_frame(60)
    result = train_experiment(frame, [], validation_draws=10, n_bootstrap=10, models_dir=tmp_path)
    metadata = json.loads(Path(result["metadata_path"]).read_text())
    metadata["feature_version"] = 2
    Path(result["metadata_path"]).write_text(json.dumps(metadata))
    assert load_latest(frame, models_dir=tmp_path)["stale"]
    Path(result["artifact"]).write_bytes(b"broken pickle")
    with pytest.raises(ValueError, match="No loadable"):
        load_latest(frame, models_dir=tmp_path)


def test_prediction_batch_content_conflicts_and_partial_batches_fail(tmp_path):
    db = Database(tmp_path / "test.db")
    predictions = generate_combinations(seed=42)
    with pytest.raises(ValueError, match="exactly ten"):
        db.save_predictions(predictions[:1], "short")
    db.save_predictions(predictions, "batch")
    changed = generate_combinations(seed=43)
    with pytest.raises(ValueError, match="different contents"):
        db.save_predictions(changed, "batch")
    assert db.prediction_count() == 10
    assert len(db.latest_predictions()) == 10


def test_preferences_concurrent_writes_are_atomic(tmp_path, monkeypatch):
    from src.preferences import DEFAULTS, read_preferences, save_preferences
    monkeypatch.setattr("src.preferences.DATA", tmp_path)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda seed: save_preferences({**DEFAULTS, "seed": seed}), range(20)))
    assert read_preferences()["seed"] in range(20)
    assert not list(tmp_path.glob("*.tmp"))


def test_independent_source_audit_detects_mismatch(tmp_path):
    db = Database(tmp_path / "test.db")
    db.import_records([record()], [])
    headers = ["Év", "Hét", "Húzásdátum", *[f"{n} találat ({unit})" for n in (5, 4, 3, 2) for unit in ("db", "Ft")],
               *[f"Szám {i}" for i in range(1, 6)]]
    row = [2024, 10, "2024.03.09.", *[0] * 8, 1, 5, 7, 20, 90]
    source = "<table><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr><tr>"
    source += "".join(f"<td>{cell}</td>" for cell in row) + "</tr></table>"
    result = audit_html(source, db)
    assert result["source_unique_draws"] == 1
    assert not result["missing_from_database"]
    assert result["field_mismatches"][0]["field"] == "prizes"
    assert result["passed"] is False


def test_prediction_csv_has_ten_rows_and_five_numeric_columns():
    from io import StringIO
    import pandas as pd
    predictions = generate_combinations(seed=17)
    restored = pd.read_csv(StringIO(combinations_frame(predictions).to_csv(index=False)))
    assert len(restored) == 10
    columns = [f"number_{i}" for i in range(1, 6)]
    assert restored[columns].to_numpy().tolist() == [p["numbers"] for p in predictions]
    assert all(pd.api.types.is_integer_dtype(restored[c]) for c in columns)


def test_local_data_change_bypasses_interval_and_conditional_headers(tmp_path, monkeypatch):
    db = Database(tmp_path / "test.db")
    response = MagicMock(status_code=200, headers={"ETag": "source-version"})
    response.content = html([[2024, 10, "2024.03.09.", 0, "0 Ft", 1, 5, 7, 20, 90]]).encode()
    monkeypatch.setattr("src.data.scraper.RAW", tmp_path / "raw")
    with patch("requests.Session.get", return_value=response) as get:
        assert sync_database(db, force=True)["imported"] == 1
        with db.connect() as conn:
            conn.execute("DELETE FROM lottery_draws")
        assert sync_database(db)["imported"] == 1
        assert "If-None-Match" not in get.call_args.kwargs["headers"]


def test_failed_download_is_throttled_without_hiding_the_error(tmp_path):
    import requests
    db = Database(tmp_path / "test.db")
    with patch("requests.Session.get", side_effect=requests.Timeout("Offline")) as get:
        with pytest.raises(requests.Timeout):
            sync_database(db, force=True)
        assert sync_database(db)["status"] == "cached"
        assert get.call_count == 1
        assert db.state()["last_error"] == "Offline"


def test_critical_threshold_uses_configured_alpha():
    fs = build_features(make_frame(30))
    result = evaluate_predictions(np.full(fs.Y.shape, BASE_RATE), fs.Y,
                                  analytic_uniform=True, alpha=.01, n_bootstrap=10)
    assert result["critical_total_matches"] == critical_total(30, .01)
    assert result["critical_alpha"] == .01
