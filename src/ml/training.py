"""Walk-forward experiment runner with honest, uniform-by-default selection."""

from __future__ import annotations

import datetime as dt
import inspect
import logging
import time
import warnings as _warnings
from functools import partial

import numpy as np

from src.features.engineering import BASE_RATE, FEATURE_VERSION, N_NUMBERS, build_features, dataset_fingerprint, validate_frame
from src.ml.backtesting import InsufficientDataError, WalkForwardConfig, evaluation_indices, training_window, walk_forward
from src.ml.metrics import EXPECTED_MATCHES, UNIFORM_LOG_LOSS, evaluate_predictions, holm_adjust, top_k_matches
from src.ml.models import BASELINE_MODELS, available_models, binary_log_loss, create_model, normalize_probabilities
from src.ml.persistence import ENGINE_VERSION, load_latest, save_experiment

LOG = logging.getLogger(__name__)
DISCLAIMER = (
    "Lottery draws are designed to be independent and uniformly random. Every 5-number combination has the "
    "same 1 in 43,949,268 jackpot chance; no model can be expected to change that. Scores are research output."
)
DEFAULT_OPTIONS = {
    "training_window": 250,
    "refit_every": None,
    "n_folds": 4,
    "min_train_draws": 30,
    "warmup_draws": 10,
    "alpha": 0.05,
    "n_bootstrap": 1000,
    "n_periods": 4,
    "model_params": None,
    "include_optional": True,
    "models_dir": None,
    "save": True,
    "final_fit": True,
    "require_calibration": True,
    "in_sample_draws": 50,
    "overfit_gap": 0.5,
    "holdout_fraction": 0.5,
    "min_selection_draws": 30,
}
OPTION_ALIASES = {"max_train_draws": "training_window"}

__all__ = ["train_experiment", "load_latest", "available_models", "DEFAULT_OPTIONS", "ENGINE_VERSION"]


def _make_reporter(progress):
    if progress is None:
        return lambda fraction, message: None
    try:
        params = list(inspect.signature(progress).parameters.values())
        varargs = any(p.kind == p.VAR_POSITIONAL for p in params)
        positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        arity = 2 if varargs else min(len(positional), 2)
    except (TypeError, ValueError):
        arity = 2

    def report(fraction, message):
        args = (float(min(1.0, max(0.0, fraction))), str(message))[:arity]
        try:
            progress(*args)
        except Exception:
            LOG.exception("Training progress callback failed")

    return report


def _resolve_options(options: dict) -> dict:
    resolved = dict(DEFAULT_OPTIONS)
    for key, value in options.items():
        key = OPTION_ALIASES.get(key, key)
        if key not in DEFAULT_OPTIONS:
            raise TypeError(f"unknown train_experiment option {key!r}; valid: {sorted(DEFAULT_OPTIONS)}")
        resolved[key] = value
    resolved["model_params"] = dict(resolved["model_params"] or {})
    return resolved


def _normalize_names(names) -> list[str]:
    if isinstance(names, str):
        names = [names]
    out = []
    for name in names or []:
        key = str(name).strip().lower()
        if key and key not in out:
            out.append(key)
    return out


def _in_sample(wf) -> dict:
    p, y = wf["in_sample_probs"], wf["in_sample_actual"]
    if len(p) == 0:
        return {}
    return {"mean_matches": float(top_k_matches(p, y).mean()), "log_loss": binary_log_loss(p, y), "draws": int(len(p))}


def _flatten(result: dict) -> None:
    m = result.get("metrics") or {}
    for key in ("mean_matches", "lift", "brier", "brier_skill", "log_loss", "log_loss_diff", "p_value",
                "mean_matches_ci", "match_histogram", "n_draws"):
        result[key] = m.get(key)


def train_experiment(frame, names, validation_draws: int = 20, seed: int = 42, progress=None, **options) -> dict:
    """Run a chronological walk-forward experiment and persist the outcome.

    See module docs / parent API notes for the returned dictionary shape.
    """
    opts = _resolve_options(options)
    if not 0 < float(opts["holdout_fraction"]) < 1 or int(opts["min_selection_draws"]) < 1:
        raise ValueError("holdout_fraction must be between 0 and 1; min_selection_draws must be positive")
    if not 0 < float(opts["alpha"]) < 1:
        raise ValueError("alpha must be between 0 and 1")
    from numbers import Integral
    if not isinstance(validation_draws, Integral) or isinstance(validation_draws, bool) or validation_draws < 1:
        raise ValueError("validation_draws must be >= 1")
    if not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    validation_draws, seed = int(validation_draws), int(seed)
    report = _make_reporter(progress)
    report(0.0, "validating data")
    data = validate_frame(frame)
    fingerprint = dataset_fingerprint(data, validated=True)
    fs = build_features(data, validated=True)
    n = fs.n_draws
    cfg = WalkForwardConfig(validation_draws=validation_draws, max_train_draws=int(opts["training_window"]),
                            refit_every=opts["refit_every"], n_folds=int(opts["n_folds"]),
                            min_train_draws=int(opts["min_train_draws"]), warmup_draws=int(opts["warmup_draws"]))
    registry = available_models(include_optional=bool(opts["include_optional"]))
    requested = _normalize_names(names)
    order = list(BASELINE_MODELS) + [m for m in requested if m not in BASELINE_MODELS]
    notes: list[str] = []
    try:
        eval_idx = evaluation_indices(n, cfg)
    except InsufficientDataError as exc:
        eval_idx = None
        notes.append(f"Insufficient labelled history for out-of-sample evaluation: {exc}.")
    labels = [fs.labels[i] for i in eval_idx] if eval_idx is not None else None
    # The final suffix is sealed from model selection. All choices use only
    # the preceding development segment; final metrics are diagnostic only.
    selection_count = (len(eval_idx) - max(1, int(np.ceil(len(eval_idx) * opts["holdout_fraction"])))
                       if eval_idx is not None else 0)
    selection_count = max(0, selection_count)

    results, factories = [], {}
    for i, name in enumerate(order):
        base = i / (len(order) + 1)
        report(base, f"backtesting {name}")
        result = {"model": name, "role": "baseline" if name in BASELINE_MODELS else "candidate",
                  "requested": name in requested, "status": "ok", "reason": "", "warnings": [],
                  "significant": False, "eligible": False, "selected": False, "p_value_holm": None}
        info = registry.get(name)
        if info is None:
            result.update(status="unavailable", reason=f"unknown model '{name}'")
        elif not info["available"]:
            result.update(status="unavailable", reason=info["reason"])
        elif eval_idx is None:
            result.update(status="insufficient_data", reason=notes[-1])
        else:
            result["kind"] = info["kind"]
            factory = partial(create_model, name, fs.feature_names, seed, **opts["model_params"].get(name, {}))
            try:
                start = time.perf_counter()
                with _warnings.catch_warnings(record=True) as caught:
                    wf = walk_forward(fs, factory, cfg, in_sample_draws=int(opts["in_sample_draws"]),
                                      progress=lambda f, b=base: report(b + f / (len(order) + 1), f"backtesting {name}"))
                result["warnings"].extend(str(w.message) for w in caught)
                result["fit_seconds"] = time.perf_counter() - start
                metric_options = dict(seed=seed, n_bootstrap=int(opts["n_bootstrap"]),
                                      n_periods=int(opts["n_periods"]), analytic_uniform=(name == "uniform"),
                                      random_uniform=(name == "random"), alpha=float(opts["alpha"]))
                result["metrics"] = evaluate_predictions(wf["probs"], wf["actual"], labels=labels,
                                                          ranking_scores=wf["ranking_scores"], **metric_options)
                result["selection_metrics"] = (
                    evaluate_predictions(wf["probs"][:selection_count], wf["actual"][:selection_count],
                                         labels=labels[:selection_count],
                                         ranking_scores=wf["ranking_scores"][:selection_count], **metric_options)
                    if selection_count else None)
                result["final_test_metrics"] = evaluate_predictions(
                    wf["probs"][selection_count:], wf["actual"][selection_count:],
                    labels=labels[selection_count:], ranking_scores=wf["ranking_scores"][selection_count:],
                    **metric_options)
                result["folds"] = wf["folds"]
                result["in_sample"] = _in_sample(wf) if name not in BASELINE_MODELS else {}
                factories[name] = factory
            except Exception as exc:  # keep the experiment alive; report honestly
                result.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        _flatten(result)
        results.append(result)

    alpha = float(opts["alpha"])
    candidates = [r for r in results if r["role"] == "candidate" and r["status"] == "ok"]
    adjusted = holm_adjust({r["model"]: (r.get("selection_metrics") or {}).get("p_value") for r in candidates})
    for r in candidates:
        m = r.get("selection_metrics")
        if not m:
            r["warnings"].append("No development draws available for model selection")
            continue
        r["p_value_holm"] = adjusted[r["model"]]
        r["significant"] = bool(r["p_value_holm"] is not None and r["p_value_holm"] < alpha
                                and m["mean_matches"] > EXPECTED_MATCHES)
        calibrated = m["brier_skill"] > 0 and m["log_loss"] <= UNIFORM_LOG_LOSS
        r["eligible"] = (r["significant"] and (calibrated or not opts["require_calibration"])
                         and selection_count >= int(opts["min_selection_draws"]))
        if r["significant"] and not r["eligible"]:
            if not calibrated:
                r["warnings"].append("Significant development hits but probability losses exceed uniform.")
            if selection_count < int(opts["min_selection_draws"]):
                r["warnings"].append(f"Only {selection_count} development draws; at least "
                                     f"{opts['min_selection_draws']} are required for model selection.")
        ins = r.get("in_sample") or {}
        if ins and ins["mean_matches"] - m["mean_matches"] > float(opts["overfit_gap"]):
            r["warnings"].append(
                f"Likely overfitting: in-sample mean matches {ins['mean_matches']:.2f} vs out-of-sample "
                f"{m['mean_matches']:.2f}.")
        if m["n_draws"] < 50:
            r["warnings"].append(
                f"Low statistical power: {m['n_draws']} evaluation draws; p<{alpha} requires mean matches >= "
                f"{m['critical_mean_matches']:.2f} (chance {EXPECTED_MATCHES:.3f}).")

    eligible = [r for r in candidates if r["eligible"]]
    if eligible:
        best = max(eligible, key=lambda r: (r["selection_metrics"]["mean_matches"], -r["selection_metrics"]["log_loss"]))
        selected = best["model"]
        selection_reason = (f"'{selected}' was provisionally selected on development draws only (Holm-adjusted p="
                            f"{best['p_value_holm']:.4g} < {alpha}). Final-test metrics were not used to choose "
                            "or tune models; prospective confirmation is still required.")
    else:
        selected = "uniform"
        selection_reason = ("No model met corrected development-evidence and minimum sample requirements; "
                            "defaulting to uniform scores. Final-test metrics are not used for selection.")
    for r in results:
        r["selected"] = r["model"] == selected
    notes.append(selection_reason)

    report(len(order) / (len(order) + 1), "fitting final models")
    model_scores: dict[str, list[float]] = {}
    fitted = create_model("uniform", fs.feature_names, seed).fit(fs.X[:0], fs.Y[:0])
    final_names = [r["model"] for r in results if r["status"] == "ok" and r["model"] != "uniform"
                   and (opts["final_fit"] or r["model"] == selected)]
    lo, hi = training_window(n, cfg)
    for name in final_names:
        if hi - lo < 1:
            break
        try:
            with _warnings.catch_warnings(record=True) as caught:
                model = factories[name]().fit(fs.X[lo:hi], fs.Y[lo:hi])
            for r in results:
                if r["model"] == name:
                    r["warnings"].extend(str(w.message) for w in caught)
            model_scores[name] = [float(v) for v in model.predict_proba(fs.X[n])]
            if name == selected:
                fitted = model
        except Exception as exc:
            for r in results:
                if r["model"] == name:
                    r["warnings"].append(f"final fit failed: {type(exc).__name__}: {exc}")
    model_scores["uniform"] = [BASE_RATE] * N_NUMBERS
    if selected not in model_scores:
        notes.append(f"Final fit of selected '{selected}' failed; uniform used instead.")
        selection_reason = f"Selected model could not be fitted for generation; defaulting to uniform scores ({selected})."
        selected, fitted = "uniform", create_model("uniform", fs.feature_names, seed).fit(fs.X[:0], fs.Y[:0])
        for r in results:
            r["selected"] = r["model"] == "uniform"
    scores = list(model_scores[selected])
    members = [k for k in model_scores if k not in BASELINE_MODELS]
    ensemble_scores = None
    if len(members) >= 2:
        ensemble_scores = [float(v) for v in normalize_probabilities(np.mean([model_scores[k] for k in members], axis=0))]

    summary = {
        "engine_version": ENGINE_VERSION,
        "feature_version": FEATURE_VERSION,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_fingerprint": fingerprint,
        "n_draws": n,
        "first_draw": fs.labels[0] if n else None,
        "last_draw": fs.labels[-1] if n else None,
        "validation_draws": int(len(eval_idx)) if eval_idx is not None else 0,
        "requested_validation_draws": validation_draws,
        "evaluation_range": ({"start_index": int(eval_idx[0]), "end_index": int(eval_idx[-1]),
                              "first_label": labels[0], "last_label": labels[-1]} if eval_idx is not None else None),
        "seed": seed,
        "config": {k: v for k, v in opts.items() if k not in ("models_dir",)},
        "feature_names": fs.feature_names,
        "results": results,
        "selected_model": selected,
        "selection_reason": selection_reason,
        "selection_protocol": "Chronological development selection followed by sealed final test; no final-test optimization",
        "selection_draws": selection_count,
        "final_test_draws": len(eval_idx) - selection_count if eval_idx is not None else 0,
        "selection_end_index": int(eval_idx[selection_count - 1]) if selection_count else None,
        "final_test_start_index": int(eval_idx[selection_count]) if eval_idx is not None else None,
        "selection_is_provisional": selected != "uniform",
        "alpha": alpha,
        "multiple_testing": "Holm-Bonferroni over candidate (non-baseline) models",
        "expected_matches_uniform": EXPECTED_MATCHES,
        "scores": scores,
        "model_scores": model_scores,
        "ensemble_scores": ensemble_scores,
        "ensemble_members": members if ensemble_scores is not None else [],
        "ensemble_validated": False,
        "warnings": notes,
        "disclaimer": DISCLAIMER,
        "stale": False,
        "stale_reason": "",
        "stale_models": [],
    }
    artifact, metadata_path = "", ""
    if opts["save"]:
        report(0.98, "saving artifact")
        artifact, metadata_path = save_experiment(summary, fitted, opts["models_dir"])
    summary.update(artifact=artifact, metadata_path=metadata_path, fitted_model=fitted)
    report(1.0, f"done: selected {selected}")
    return summary
