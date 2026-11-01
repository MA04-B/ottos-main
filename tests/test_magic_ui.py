import threading
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from src.data.database import Database
from src.prediction.generator import generate_combinations
from src.preferences import DEFAULTS
from test_app import _prediction_navigation
from test_magic_search import matching_fixture


@contextmanager
def prediction_app(tmp_path):
    frame = matching_fixture()
    db = Database(tmp_path / "magic-ui.db")
    db.import_records([{
        "year": int(row.year), "week": int(row.week), "draw_date": row.draw_date,
        "numbers": [int(getattr(row, f"number_{i}")) for i in range(1, 6)], "prizes": {},
    } for row in frame.itertuples()], [])
    st.cache_resource.clear()
    st.cache_data.clear()
    try:
        with patch("src.data.database.Database", return_value=db), \
                patch("src.preferences.read_preferences",
                      return_value={**DEFAULTS, "auto_update": False, "strategy": "weighted"}), \
                patch("src.ml.training.load_latest", return_value=None), \
                patch("streamlit.navigation", side_effect=_prediction_navigation):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
            app.run(timeout=30)
            yield app, db, frame
    finally:
        st.cache_resource.clear()
        st.cache_data.clear()


def test_magic_match_updates_existing_cards_saves_once_and_retains_csv(tmp_path):
    with prediction_app(tmp_path) as (app, db, frame):
        app.button(key="magic_generate").click().run(timeout=30)
        job = app.session_state["magic_search_job"]
        job.future.result(timeout=10)
        app.run(timeout=30)
        assert not app.exception
        assert job.snapshot().status == "matched"
        predictions = app.session_state["predictions"]
        match = predictions[job.snapshot().matched_index]
        assert match["numbers"] == frame.iloc[0][[f"number_{i}" for i in range(1, 6)]].tolist()
        assert match["historical_search"]["attempts"] == 1
        assert match["historical_search"]["historical_draws"][0]["draw_date"] == frame.iloc[0].draw_date
        assert db.prediction_count() == 10
        assert any("Exact 5/5 historical match" in element.value for element in app.success)
        assert app.get("download_button")[0].label == "Download CSV"
        app.run(timeout=30)
        assert db.prediction_count() == 10
        reloaded = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
        reloaded.run(timeout=30)
        assert not reloaded.exception
        assert any(f"Historical draw: {frame.iloc[0].draw_date}" in element.value for element in reloaded.markdown)
        assert db.prediction_count() == 10
        app.button(key="generate_again").click().run(timeout=30)
        assert not app.exception
        assert app.session_state["predictions"][0]["seed"] == 43
        assert db.prediction_count() == 20


def test_live_search_disables_generation_and_can_be_stopped(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    with prediction_app(tmp_path) as (app, db, frame):
        portfolio = generate_combinations(frame, seed=100, strategy="weighted")

        def blocked(*args, **kwargs):
            entered.set()
            assert release.wait(timeout=15)
            return portfolio

        monkeypatch.setattr("src.prediction.search.generate_combinations", blocked)
        try:
            app.button(key="magic_generate").click().run(timeout=30)
            assert entered.wait(timeout=5)
            job = app.session_state["magic_search_job"]
            assert job.snapshot().status == "running"
            assert app.button(key="generate_initial").disabled
            assert app.button(key="generate_again").disabled
            assert app.button(key="magic_generate").disabled
            assert "Attempts: 1" in app.get("progress")[0].text
            app.button(key="magic_stop").click().run(timeout=30)
            assert job.snapshot().status == "cancelling"
            release.set()
            job.future.result(timeout=5)
            app.run(timeout=30)
            assert not app.exception
            assert job.snapshot().status == "cancelled"
            assert not app.button(key="generate_initial").disabled
            assert db.prediction_count() == 0
            assert any("Search stopped" in element.value for element in app.info)
        finally:
            release.set()
            if "magic_search_job" in app.session_state:
                app.session_state["magic_search_job"].stop()
                if app.session_state["magic_search_job"].future:
                    app.session_state["magic_search_job"].future.result(timeout=5)


def test_limit_exhaustion_and_worker_failure_leave_old_portfolio_unchanged(tmp_path, monkeypatch):
    with prediction_app(tmp_path) as (app, db, frame):
        app.button(key="generate_initial").click().run(timeout=30)
        original = app.session_state["predictions"]
        no_match = generate_combinations(frame, seed=100, strategy="weighted")
        historical = {tuple(row) for row in frame[[f"number_{i}" for i in range(1, 6)]].to_numpy()}
        assert not any(tuple(p["numbers"]) in historical for p in no_match)
        monkeypatch.setattr("src.prediction.search.generate_combinations", lambda *a, **kw: no_match)
        app.number_input(key="magic_max_attempts").set_value(2).run(timeout=30)
        app.button(key="magic_generate").click().run(timeout=30)
        job = app.session_state["magic_search_job"]
        job.future.result(timeout=5)
        app.run(timeout=30)
        assert not app.exception
        assert job.snapshot().status == "exhausted" and job.snapshot().attempts == 2
        assert app.session_state["predictions"] == original
        assert db.prediction_count() == 10
        assert any("within the attempt limit" in element.value for element in app.warning)

        def failed(*a, **kw):
            raise ValueError("Test generator failure")

        monkeypatch.setattr("src.prediction.search.generate_combinations", failed)
        app.button(key="magic_generate").click().run(timeout=30)
        job.future.result(timeout=5)
        app.run(timeout=30)
        assert not app.exception
        assert job.snapshot().status == "failed"
        assert app.session_state["predictions"] == original
        assert db.prediction_count() == 10
        assert any("Test generator failure" in element.value for element in app.error)
