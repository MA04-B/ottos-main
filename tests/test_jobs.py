import threading
from unittest.mock import patch

import pandas as pd
import pytest

from src.data.database import Database
from src.jobs import TrainingJob


def test_training_runs_in_worker_and_rejects_overlap():
    entered, release = threading.Event(), threading.Event()
    job = TrainingJob()

    def bounded_work(*args):
        entered.set()
        assert release.wait(timeout=5)
        return {"selected_model": "uniform"}

    try:
        with patch.object(job, "run", side_effect=bounded_work):
            job.start(pd.DataFrame(), ["uniform"], 20, 42, 250)
            assert entered.wait(timeout=5)
            with pytest.raises(RuntimeError, match="already running"):
                job.start(pd.DataFrame(), ["uniform"], 20, 42, 250)
            release.set()
            assert job.future.result(timeout=5)["selected_model"] == "uniform"
    finally:
        release.set()
        job.executor.shutdown(wait=True)


def test_progress_is_bounded_and_failure_is_logged(tmp_path):
    db = Database(tmp_path / "test.db")
    job = TrainingJob()
    try:
        job.update(3, 10, "Fold 3")
        assert job.progress == .3
        assert job.message == "Fold 3"
        job.update(2.)
        assert job.progress == 1.
        with patch("src.jobs.Database", return_value=db), \
                patch("src.ml.training.train_experiment", side_effect=ValueError("Invalid features")):
            with pytest.raises(ValueError, match="Invalid features"):
                job.run(pd.DataFrame(), ["uniform"], 20, 42, 250)
        events = db.table("system_logs")
        assert events.iloc[0].event == "training_failed"
        assert "Invalid features" in events.iloc[0].details
    finally:
        job.executor.shutdown(wait=True)
