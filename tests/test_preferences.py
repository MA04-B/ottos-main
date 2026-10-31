import json

import pytest

from src.preferences import DEFAULTS, read_preferences, save_preferences, validate_preferences


def test_settings_persist(tmp_path, monkeypatch):
    monkeypatch.setattr("src.preferences.DATA", tmp_path)
    assert read_preferences() == DEFAULTS
    save_preferences({**DEFAULTS, "seed": 123, "auto_update": False})
    assert read_preferences()["seed"] == 123
    assert not read_preferences()["auto_update"]
    assert not (tmp_path / "settings.tmp").exists()


@pytest.mark.parametrize("changes", [{"seed": -1}, {"diversity": float("nan")},
                                      {"analysis_window": 17}, {"validation_draws": 0},
                                      {"auto_update": "yes"}, {"strategy": "invalid"}])
def test_invalid_settings_fail_explicitly(changes):
    with pytest.raises(ValueError):
        validate_preferences({**DEFAULTS, **changes})


def test_malformed_json_is_not_silently_reset(tmp_path, monkeypatch):
    monkeypatch.setattr("src.preferences.DATA", tmp_path)
    (tmp_path / "settings.json").write_text("{broken")
    with pytest.raises(json.JSONDecodeError):
        read_preferences()
