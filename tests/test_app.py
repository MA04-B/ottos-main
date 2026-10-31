from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.data.database import Database
from src.preferences import DEFAULTS
from src.ml.training import train_experiment
from test_models import make_frame


def test_offline_application_startup(tmp_path):
    db = Database(tmp_path / "offline.db")
    db.import_records([{"year": 2024, "week": 10, "draw_date": "2024-03-09",
                        "numbers": [1, 5, 20, 45, 90], "prizes": {}}], [])
    st.cache_resource.clear()
    st.cache_data.clear()
    with patch("src.data.database.Database", return_value=db), \
            patch("src.preferences.read_preferences", return_value={**DEFAULTS, "auto_update": False}), \
            patch("src.ml.training.load_latest", return_value=None):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
        app.run(timeout=30)
        assert not app.exception
        assert app.metric[0].value == "1"
        assert any("43,949,268" in caption.value for caption in app.caption)
    st.cache_resource.clear()
    st.cache_data.clear()


@pytest.mark.parametrize("page_index", range(8))
def test_all_dashboard_callbacks_with_real_metrics(tmp_path, page_index):
    db = Database(tmp_path / "pages.db")
    frame = make_frame(65, undated=15)
    records = [{"year": int(r.year), "week": int(r.week), "draw_date": r.draw_date,
                "numbers": [int(getattr(r, f"number_{i}")) for i in range(1, 6)], "prizes": {}}
               for r in frame.itertuples()]
    db.import_records(records, [])
    experiment = train_experiment(frame, ["frequency"], validation_draws=10, n_bootstrap=10, save=False)

    def select_page(pages):
        navigation = MagicMock()
        navigation.run.side_effect = pages[page_index]._page
        return navigation

    st.cache_resource.clear()
    st.cache_data.clear()
    try:
        with patch("src.data.database.Database", return_value=db), \
                patch("src.preferences.read_preferences", return_value={**DEFAULTS, "auto_update": False}), \
                patch("src.ml.training.load_latest", return_value=experiment), \
                patch("streamlit.navigation", side_effect=select_page):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
            app.run(timeout=30)
            assert not app.exception
    finally:
        st.cache_resource.clear()
        st.cache_data.clear()
