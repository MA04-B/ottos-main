from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.data.database import Database
from src.ml.training import train_experiment
from src.preferences import DEFAULTS
from test_models import make_frame


def offline_database(tmp_path, name):
    db = Database(tmp_path / name)
    frame = make_frame(65, undated=15)
    records = [{
        "year": int(row.year),
        "week": int(row.week),
        "draw_date": row.draw_date,
        "numbers": [int(getattr(row, f"number_{i}")) for i in range(1, 6)],
        "prizes": {},
    } for row in frame.itertuples()]
    db.import_records(records, [])
    return db, frame


def run_page(db, page_index, model_result=None):
    def select_page(pages):
        navigation = MagicMock()
        navigation.run.side_effect = pages[page_index]._page
        return navigation

    st.cache_resource.clear()
    st.cache_data.clear()
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    with patch("src.data.database.Database", return_value=db), \
            patch("src.preferences.read_preferences", return_value={**DEFAULTS, "auto_update": False}), \
            patch("src.ml.training.load_latest", return_value=model_result), \
            patch("streamlit.navigation", side_effect=select_page):
        app.run(timeout=30)
    return app


def test_offline_application_startup(tmp_path):
    db = Database(tmp_path / "offline.db")
    db.import_records([{"year": 2024, "week": 10, "draw_date": "2024-03-09",
                        "numbers": [1, 5, 20, 45, 90], "prizes": {}}], [])
    app = run_page(db, 0)
    assert not app.exception
    assert app.metric[0].value == "1"
    assert any("43,949,268" in caption.value for caption in app.caption)


@pytest.mark.parametrize("page_index", range(8))
def test_all_dashboard_callbacks_with_real_metrics(tmp_path, page_index):
    db, frame = offline_database(tmp_path, "pages.db")
    experiment = train_experiment(frame, ["frequency"], validation_draws=10, n_bootstrap=10, save=False)
    app = run_page(db, page_index, experiment)
    assert not app.exception


def test_generate_and_generate_again_share_fresh_generation_flow(tmp_path):
    db, _ = offline_database(tmp_path, "generate.db")
    st.cache_resource.clear()
    st.cache_data.clear()
    try:
        with patch("src.data.database.Database", return_value=db), \
                patch("src.preferences.read_preferences", return_value={**DEFAULTS, "auto_update": False}), \
                patch("src.ml.training.load_latest", return_value=None), \
                patch("streamlit.navigation", side_effect=lambda pages: _prediction_navigation(pages)):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
            app.run(timeout=30)
            assert not app.exception
            next(button for button in app.button if button.label == "Generate 10 combinations").click().run()
            assert not app.exception
            first = app.session_state["predictions"]
            assert len(first) == 10 and db.prediction_count() == 10
            labels = [button.label for button in app.button]
            assert labels.index("Save predictions") < labels.index("Generate Again")
            assert app.get("download_button")
            assert app.get("download_button")[0].label == "Download CSV"
            next(button for button in app.button if button.label == "Generate Again").click().run()
            assert not app.exception
            second = app.session_state["predictions"]
            assert len(second) == 10 and db.prediction_count() == 20
            assert second[0]["seed"] == first[0]["seed"] + 1
            assert [p["numbers"] for p in second] != [p["numbers"] for p in first]
    finally:
        st.cache_resource.clear()
        st.cache_data.clear()


def _prediction_navigation(pages):
    navigation = MagicMock()
    navigation.run.side_effect = pages[4]._page
    return navigation
