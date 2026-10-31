import json
import uuid

from config.settings import DATA

DEFAULTS = {"auto_update": True, "interval_hours": 24, "seed": 42, "diversity": .5,
            "strategy": "mixed", "analysis_window": 0, "validation_draws": 20,
            "training_window": 250, "selected_model": "automatic"}
STRATEGIES = ("mixed", "ranked", "weighted", "historical", "recent", "bayesian",
              "balanced", "ensemble", "diversified")


def validate_preferences(values):
    if not isinstance(values, dict):
        raise ValueError("Settings must be a JSON object")
    result = {**DEFAULTS, **values}
    if not isinstance(result["auto_update"], bool):
        raise ValueError("auto_update must be true or false")
    for key, lower, upper in (("interval_hours", 1, 168), ("seed", 0, 2**31 - 1),
                              ("validation_draws", 10, 250), ("training_window", 0, 3500)):
        if type(result[key]) is not int or not lower <= result[key] <= upper:
            raise ValueError(f"{key} must be an integer between {lower} and {upper}")
    diversity = result["diversity"]
    if not isinstance(diversity, (int, float)) or isinstance(diversity, bool) or not 0 <= diversity <= 1:
        raise ValueError("diversity must be between 0 and 1")
    if result["strategy"] not in STRATEGIES:
        raise ValueError("Unknown generation strategy")
    if result["analysis_window"] not in (0, 10, 25, 50, 100, 250):
        raise ValueError("Unknown historical analysis window")
    if not isinstance(result["selected_model"], str) or not result["selected_model"]:
        raise ValueError("Model selection must be a nonempty name")
    return result


def read_preferences():
    path = DATA / "settings.json"
    if not path.exists():
        return DEFAULTS.copy()
    values = json.loads(path.read_text(encoding="utf-8"))
    return validate_preferences(values)


def save_preferences(values):
    values = validate_preferences(values)
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "settings.json"
    temporary = path.with_name(f"settings.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
