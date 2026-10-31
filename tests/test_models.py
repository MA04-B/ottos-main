import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.engineering import (  # noqa: E402
    BASE_RATE,
    FEATURE_NAMES,
    build_features,
    dataset_fingerprint,
    validate_frame,
)
from src.ml.models import MODEL_SPECS, available_models, create_model  # noqa: E402
from src.ml.persistence import load_latest  # noqa: E402
from src.ml.training import train_experiment  # noqa: E402

CORE = ["uniform", "random", "frequency", "ew_frequency", "bayesian", "logistic", "random_forest", "extra_trees",
        "gradient_boosting", "hist_gradient_boosting"]
FAST = {"random_forest": {"n_estimators": 10}, "extra_trees": {"n_estimators": 10},
        "gradient_boosting": {"n_estimators": 10}, "hist_gradient_boosting": {"max_iter": 10},
        "xgboost": {"n_estimators": 10}, "lightgbm": {"n_estimators": 10}, "catboost": {"iterations": 10}}


def make_frame(n=80, seed=0, undated=0):
    rng = np.random.default_rng(seed)
    start = dt.date(2020, 1, 4)
    rows = []
    for i in range(n):
        d = start + dt.timedelta(days=7 * i)
        iso = d.isocalendar()
        nums = sorted(rng.choice(np.arange(1, 91), 5, replace=False).tolist())
        rows.append({"year": iso[0], "week": iso[1], "draw_date": None if i < undated else d.isoformat(),
                     **{f"number_{j + 1}": nums[j] for j in range(5)}})
    return pd.DataFrame(rows)


def test_validate_rejects_invalid():
    frame = make_frame(5)
    bad = frame.copy()
    bad.loc[0, "number_2"] = bad.loc[0, "number_1"]
    with pytest.raises(ValueError):
        validate_frame(bad)
    bad = frame.copy()
    bad.loc[1, "number_3"] = 91
    with pytest.raises(ValueError):
        validate_frame(bad)
    with pytest.raises(ValueError):
        validate_frame(frame.drop(columns=["week"]))


def test_validate_orders_by_year_week_and_keeps_null_dates():
    frame = make_frame(30, undated=12)
    frame.loc[3, "draw_date"] = ""
    frame.loc[4, "draw_date"] = float("nan")
    shuffled = frame.sample(frac=1.0, random_state=1)
    out = validate_frame(shuffled)
    assert len(out) == 30
    assert list(zip(out.year, out.week)) == sorted(zip(out.year, out.week))
    assert out["draw_date"].isna().sum() == 12
    assert all(d is None or isinstance(d, str) for d in out["draw_date"])
    assert dataset_fingerprint(shuffled) == dataset_fingerprint(frame)
    frame2 = frame.copy()
    frame2.loc[3, "draw_date"] = None
    assert dataset_fingerprint(frame2) == dataset_fingerprint(frame)


def test_undated_history_builds_time_features():
    frame = make_frame(40, undated=40)
    fs = build_features(frame)
    days = fs.X[:, 0, fs.column("days_since_prev")]
    assert np.all(np.isfinite(fs.X))
    assert np.allclose(days[1:40], 1.0)  # consecutive ISO weeks -> 7 days


def test_features_are_prior_only():
    frame = make_frame(60, seed=3)
    k = 35
    changed = frame.copy()
    rng = np.random.default_rng(9)
    for i in range(k, len(changed)):
        nums = sorted(rng.choice(np.arange(1, 91), 5, replace=False).tolist())
        changed.loc[i, [f"number_{j}" for j in range(1, 6)]] = nums
    a, b = build_features(frame), build_features(changed)
    assert np.array_equal(a.X[: k + 1], b.X[: k + 1])
    assert not np.array_equal(a.X[k + 1:], b.X[k + 1:])


def test_feature_values_match_manual_counts():
    frame = make_frame(30, seed=4)
    fs = build_features(frame)
    t = 20
    counts = fs.Y[:t].sum(axis=0)
    assert np.allclose(fs.X[t, :, fs.column("freq_all")], counts / t)
    assert np.allclose(fs.X[t, :, fs.column("prev_draw")], fs.Y[t - 1])
    assert np.allclose(fs.X[0, :, fs.column("freq_all")], BASE_RATE)
    assert fs.X.shape == (31, 90, len(FEATURE_NAMES))


def test_registry_shape_and_optional_disabling(monkeypatch):
    reg = available_models()
    assert set(reg) == set(MODEL_SPECS)
    for name in CORE:
        assert reg[name]["available"] is True and reg[name]["optional"] is False
    for info in reg.values():
        assert {"available", "reason", "kind", "optional", "package", "description"} <= set(info)
    off = available_models(include_optional=False)
    for name in ("xgboost", "lightgbm", "catboost"):
        assert off[name]["available"] is False and "disabled" in off[name]["reason"]
    monkeypatch.setenv("OTOSLOTTO_DISABLE_OPTIONAL_MODELS", "1")
    assert not available_models()["xgboost"]["available"]


@pytest.mark.parametrize("name", CORE + ["xgboost", "lightgbm", "catboost"])
def test_models_trainable(name):
    if not available_models()[name]["available"]:
        pytest.skip(available_models()[name]["reason"])
    fs = build_features(make_frame(70, seed=5))
    model = create_model(name, fs.feature_names, 1, **FAST.get(name, {})).fit(fs.X[10:60], fs.Y[10:60])
    p = model.predict_proba(fs.X[60])
    assert p.shape == (90,) and np.all(np.isfinite(p))
    assert np.all((p > 0) & (p < 1)) and np.isclose(p.sum(), 5.0, atol=1e-3)
    assert model.predict_proba(fs.X[60:63]).shape == (3, 90)


def test_constant_labels_handled():
    fs = build_features(make_frame(30))
    model = create_model("logistic", fs.feature_names).fit(fs.X[10:20], np.zeros((10, 90), dtype=int))
    assert np.allclose(model.predict_proba(fs.X[20]), BASE_RATE)


def test_train_experiment_and_persistence(tmp_path, monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "MODELS", tmp_path / "models")
    frame = make_frame(70, seed=6, undated=30)
    calls = []
    result = train_experiment(frame, ["frequency", "logistic", "nope"], validation_draws=8, seed=3,
                              progress=lambda *a: calls.append(a), n_bootstrap=50, training_window=40)
    assert calls and calls[-1][0] == 1.0
    assert len(result["scores"]) == 90 and np.isclose(sum(result["scores"]), 5.0, atol=1e-3)
    by = {r["model"]: r for r in result["results"]}
    assert by["uniform"]["status"] == "ok" and by["random"]["status"] == "ok"
    assert by["nope"]["status"] == "unavailable"
    assert by["uniform"]["mean_matches"] == pytest.approx(25 / 90)
    assert set(result["model_scores"]) >= {"uniform", "frequency", "logistic"}
    assert result["ensemble_validated"] is False and result["ensemble_members"] == ["frequency", "logistic"]
    artifact = Path(result["artifact"])
    assert artifact.exists() and artifact.parent == tmp_path / "models"
    assert not list(artifact.parent.glob("*.tmp"))
    loaded = load_latest(frame)
    assert loaded is not None and loaded["stale"] is False
    assert loaded["dataset_fingerprint"] == result["dataset_fingerprint"]
    assert loaded["scores"] == pytest.approx(result["scores"])
    assert loaded["selected_model"] == result["selected_model"]
    assert loaded["fitted_model"] is not None
    stale = load_latest(make_frame(71, seed=6))
    assert stale["stale"] is True and stale["stale_models"] and "retrain" in stale["stale_reason"]
    assert load_latest(make_frame(71, seed=6), allow_stale=False) is None
    assert load_latest(frame, models_dir=tmp_path / "empty") is None


def test_progress_tolerates_any_signature(tmp_path):
    frame = make_frame(50)
    for cb in (lambda: None, lambda f: None, lambda f, m: None, lambda *a, **k: None):
        train_experiment(frame, [], validation_draws=5, progress=cb, n_bootstrap=20, save=False)


def test_optional_disabled_in_training():
    result = train_experiment(make_frame(50), ["xgboost"], validation_draws=5, n_bootstrap=20, save=False,
                              include_optional=False)
    by = {r["model"]: r for r in result["results"]}
    assert by["xgboost"]["status"] == "unavailable"
    assert result["selected_model"] == "uniform"


def test_insufficient_data_defaults_to_uniform():
    result = train_experiment(make_frame(12), ["frequency"], validation_draws=5, save=False, n_bootstrap=20)
    assert result["selected_model"] == "uniform"
    assert all(s == pytest.approx(5 / 90) for s in result["scores"])
    assert {r["status"] for r in result["results"]} == {"insufficient_data"}


def test_unknown_option_rejected():
    with pytest.raises(TypeError):
        train_experiment(make_frame(40), [], bogus=1, save=False)
