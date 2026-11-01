import threading
from copy import deepcopy

import pandas as pd
import pytest

from src.prediction.generator import generate_combinations
from src.prediction.search import MAX_SEARCH_ATTEMPTS, MagicSearchJob
from test_models import make_frame


def matching_fixture(seed=42, dated=True):
    frame = make_frame(40, undated=0 if dated else 40)
    portfolio = generate_combinations(frame, seed=seed, strategy="weighted")
    columns = [f"number_{i}" for i in range(1, 6)]
    frame.loc[0, columns] = portfolio[0]["numbers"]
    return frame


@pytest.mark.parametrize("dated", [True, False])
def test_real_engine_exact_match_returns_portfolio_and_official_metadata(dated):
    frame = matching_fixture(dated=dated)
    original = frame.copy(deep=True)
    options = {"seed": 42, "strategy": "weighted", "diversity": .5}
    job = MagicSearchJob()
    search_id = job.start(frame, options, max_attempts=2)
    job.future.result(timeout=10)
    snapshot = job.snapshot()
    assert snapshot.search_id == search_id and snapshot.status == "matched"
    assert snapshot.attempts == 1 and snapshot.checked_combinations == 1
    assert snapshot.matched_index == 0 and len(snapshot.portfolio) == 10
    assert snapshot.portfolio[0]["numbers"] == frame.iloc[0][[f"number_{i}" for i in range(1, 6)]].tolist()
    historical = snapshot.historical_draws[0]
    assert historical["draw_date"] == frame.iloc[0].draw_date
    assert (historical["year"], historical["week"]) == (frame.iloc[0].year, frame.iloc[0].week)
    pd.testing.assert_frame_equal(frame, original)
    assert options == {"seed": 42, "strategy": "weighted", "diversity": .5}
    snapshot.portfolio[0]["numbers"][0] = -1
    assert job.snapshot().portfolio[0]["numbers"][0] >= 1


def test_repeat_numbers_report_all_matching_historical_draws():
    frame = matching_fixture()
    frame.loc[1, [f"number_{i}" for i in range(1, 6)]] = frame.iloc[0][[f"number_{i}" for i in range(1, 6)]].tolist()
    job = MagicSearchJob()
    job.start(frame, {"seed": 42, "strategy": "weighted"}, max_attempts=1)
    job.future.result(timeout=10)
    assert job.snapshot().status == "matched"
    assert len(job.snapshot().historical_draws) == 2


def test_exhaustion_counts_every_generated_combination_and_advances_seed(monkeypatch):
    frame = make_frame(40)
    portfolio = generate_combinations(frame, seed=123, strategy="weighted")
    history = {tuple(row) for row in frame[[f"number_{i}" for i in range(1, 6)]].to_numpy()}
    assert not any(tuple(p["numbers"]) in history for p in portfolio)
    seeds = []

    def generate(data, **options):
        seeds.append(options["seed"])
        return deepcopy(portfolio)

    monkeypatch.setattr("src.prediction.search.generate_combinations", generate)
    job = MagicSearchJob()
    job.start(frame, {"seed": 8, "strategy": "weighted", "diversity": .2}, max_attempts=3)
    job.future.result(timeout=5)
    snapshot = job.snapshot()
    assert snapshot.status == "exhausted"
    assert snapshot.attempts == 3 and snapshot.checked_combinations == 30
    assert snapshot.last_seed == 10 and seeds == [8, 9, 10]
    assert not snapshot.portfolio and not snapshot.historical_draws


def test_four_matches_do_not_qualify(monkeypatch):
    frame = matching_fixture()
    portfolio = generate_combinations(frame, seed=42, strategy="weighted")
    winning = portfolio[0]["numbers"]
    replacement = next(n for n in range(1, 91) if n not in winning)
    portfolio[0]["numbers"] = sorted([*winning[:4], replacement])
    assert len(set(winning) & set(portfolio[0]["numbers"])) == 4
    monkeypatch.setattr("src.prediction.search.generate_combinations", lambda *a, **kw: portfolio)
    job = MagicSearchJob()
    job.start(frame, {"seed": 42}, max_attempts=1)
    job.future.result(timeout=5)
    assert job.snapshot().status == "exhausted"


def test_cancellation_is_nonblocking_and_prevents_another_call(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    frame = matching_fixture()
    portfolio = generate_combinations(frame, seed=42, strategy="weighted")
    calls = []

    def blocked_generator(data, **options):
        calls.append(options["seed"])
        entered.set()
        assert release.wait(timeout=10)
        return portfolio

    monkeypatch.setattr("src.prediction.search.generate_combinations", blocked_generator)
    job = MagicSearchJob()
    try:
        job.start(frame, {"seed": 42}, max_attempts=10)
        assert entered.wait(timeout=5)
        assert not job.future.done()
        assert job.snapshot().status == "running" and job.snapshot().attempts == 1
        with pytest.raises(RuntimeError, match="already running"):
            job.start(frame, {"seed": 42}, max_attempts=10)
        job.stop()
        assert job.snapshot().status == "cancelling"
        release.set()
        job.future.result(timeout=5)
        assert job.snapshot().status == "cancelled"
        assert not job.snapshot().portfolio
        assert calls == [42]
    finally:
        release.set()
        if job.future:
            job.future.result(timeout=5)


def test_failed_generator_reports_error_and_logs_traceback(monkeypatch, caplog):
    def failed(*args, **kwargs):
        raise ValueError("Invalid model scores")

    monkeypatch.setattr("src.prediction.search.generate_combinations", failed)
    job = MagicSearchJob()
    job.start(make_frame(40), {"seed": 42}, max_attempts=10)
    job.future.result(timeout=5)
    assert job.snapshot().status == "failed"
    assert "Invalid model scores" in job.snapshot().error
    assert job.snapshot().attempts == 1
    assert "Historical-match search failed" in caplog.text
    assert not job.snapshot().portfolio


def test_empty_history_fails_without_generating(monkeypatch):
    called = threading.Event()
    monkeypatch.setattr("src.prediction.search.generate_combinations", lambda *a, **kw: called.set())
    job = MagicSearchJob()
    job.start(make_frame(40).iloc[:0], {"seed": 42}, max_attempts=1)
    job.future.result(timeout=5)
    assert job.snapshot().status == "failed"
    assert "Historical drawings are required" in job.snapshot().error
    assert not called.is_set()


@pytest.mark.parametrize("limit", [0, -1, MAX_SEARCH_ATTEMPTS + 1, 1.5, True])
def test_invalid_limit_does_not_start_a_worker(limit):
    job = MagicSearchJob()
    with pytest.raises(ValueError, match="Maximum attempts"):
        job.start(make_frame(40), {"seed": 42}, max_attempts=limit)
    assert job.snapshot().status == "idle" and job.future is None


def test_restart_after_completion_has_new_search_id():
    job = MagicSearchJob()
    frame = matching_fixture()
    first_id = job.start(frame, {"seed": 42, "strategy": "weighted"}, max_attempts=1)
    job.future.result(timeout=5)
    second_id = job.start(frame, {"seed": 42, "strategy": "weighted"}, max_attempts=1)
    job.future.result(timeout=5)
    assert first_id != second_id
    assert job.snapshot().status == "matched" and job.snapshot().attempts == 1


def test_sessions_have_independent_search_and_stop_state(monkeypatch):
    frame = matching_fixture()
    portfolio = generate_combinations(frame, seed=42, strategy="weighted")
    entered, release = threading.Barrier(3), threading.Event()

    def blocked(*args, **kwargs):
        entered.wait(timeout=5)
        assert release.wait(timeout=5)
        return portfolio

    monkeypatch.setattr("src.prediction.search.generate_combinations", blocked)
    first, second = MagicSearchJob(), MagicSearchJob()
    try:
        first.start(frame, {"seed": 42}, max_attempts=1)
        second.start(frame, {"seed": 42}, max_attempts=1)
        entered.wait(timeout=5)
        first.stop()
        assert second.snapshot().status == "running"
        release.set()
        first.future.result(timeout=5)
        second.future.result(timeout=5)
        assert first.snapshot().status == "cancelled"
        assert second.snapshot().status == "matched"
    finally:
        release.set()
        for job in (first, second):
            if job.future:
                job.future.result(timeout=5)
