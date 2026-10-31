"""Atomic artifact persistence and offline resume of the latest experiment."""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import uuid
from pathlib import Path

import joblib
import numpy as np

from src.features.engineering import FEATURE_VERSION, dataset_fingerprint

ARTIFACT_FORMAT = 1
ARTIFACT_PREFIX = "experiment_"
ENGINE_VERSION = "2.0.0"
LOG = logging.getLogger(__name__)


def models_directory(models_dir=None) -> Path:
    if models_dir is None:
        from config import settings

        models_dir = settings.MODELS
    path = Path(models_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_jsonable(value):
    """Recursively convert numpy/pathlib values into JSON-safe Python objects."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_replace(path: Path, write) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "wb") as fh:
            write(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_json(path: Path, data) -> None:
    text = json.dumps(to_jsonable(data), indent=2, sort_keys=True).encode("utf-8")
    _atomic_replace(Path(path), lambda fh: fh.write(text))


def atomic_write_joblib(path: Path, obj) -> None:
    _atomic_replace(Path(path), lambda fh: joblib.dump(obj, fh))


def save_experiment(summary: dict, fitted_model, models_dir=None) -> tuple[str, str]:
    """Write ``<stem>.joblib`` (summary + fitted model) then ``<stem>.json``.

    The JSON metadata is written last, so it only ever references a complete
    artifact; readers discover experiments through the JSON files.
    """
    directory = models_directory(models_dir)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = f"{ARTIFACT_PREFIX}{stamp}_{summary['dataset_fingerprint'][:12]}"
    artifact = directory / f"{stem}.joblib"
    metadata = directory / f"{stem}.json"
    clean = to_jsonable({k: v for k, v in summary.items() if k not in ("artifact", "metadata_path")})
    atomic_write_joblib(artifact, {"format": ARTIFACT_FORMAT, "summary": clean, "fitted_model": fitted_model})
    atomic_write_json(metadata, {**clean, "format": ARTIFACT_FORMAT, "artifact": artifact.name})
    return str(artifact), str(metadata)


def _candidates(directory: Path, errors: list[str]):
    found = []
    for meta_path in directory.glob(f"{ARTIFACT_PREFIX}*.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            message = f"Cannot read model metadata {meta_path.name}: {exc}"
            LOG.warning(message)
            errors.append(message)
            continue
        if meta.get("format") != ARTIFACT_FORMAT or not meta.get("artifact"):
            errors.append(f"Unsupported or incomplete model metadata {meta_path.name}")
            continue
        found.append((str(meta.get("created_at", "")), meta_path.name, meta_path, meta))
    found.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return found


def _load(meta_path: Path, meta: dict, errors: list[str]) -> dict | None:
    artifact = meta_path.with_name(meta["artifact"])
    try:
        stored = joblib.load(artifact)
    except Exception as exc:
        message = f"Cannot load artifact {artifact.name}: {type(exc).__name__}: {exc}"
        LOG.warning(message)
        errors.append(message)
        return None
    if not isinstance(stored, dict) or stored.get("format") != ARTIFACT_FORMAT:
        errors.append(f"Invalid model artifact format in {artifact.name}")
        return None
    summary = dict(stored.get("summary") or {})
    if summary.get("dataset_fingerprint") != meta.get("dataset_fingerprint"):
        errors.append(f"Artifact/metadata dataset mismatch in {artifact.name}")
        return None
    scores = np.asarray(summary.get("scores"), dtype=float)
    if scores.shape != (90,) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        errors.append(f"Invalid number scores in {artifact.name}")
        return None
    summary.update(artifact=str(artifact), metadata_path=str(meta_path), fitted_model=stored.get("fitted_model"))
    return summary


def load_latest(frame, models_dir=None, allow_stale: bool = True) -> dict | None:
    """Return the newest loadable experiment, or None.

    Prefers an experiment whose ``dataset_fingerprint`` matches ``frame``
    (``stale=False``). Otherwise, if ``allow_stale``, returns the newest
    loadable experiment for a different dataset with ``stale=True``, a
    ``stale_reason`` and ``stale_models`` (names whose scores are outdated);
    its scores must not be presented as current.
    """
    directory = models_directory(models_dir)
    current = dataset_fingerprint(frame) if frame is not None else None
    errors = []
    candidates = _candidates(directory, errors)
    for _, _, meta_path, meta in candidates:
        if (current is not None and meta.get("dataset_fingerprint") == current
                and meta.get("feature_version") == FEATURE_VERSION
                and meta.get("engine_version") == ENGINE_VERSION):
            loaded = _load(meta_path, meta, errors)
            if loaded is not None:
                loaded.update(stale=False, stale_reason="", stale_models=[], current_fingerprint=current,
                              load_warnings=errors)
                return loaded
    if not allow_stale:
        if errors:
            raise ValueError("Model load failed: " + "; ".join(errors))
        return None
    for _, _, meta_path, meta in candidates:
        loaded = _load(meta_path, meta, errors)
        if loaded is None:
            continue
        names = sorted(set(loaded.get("model_scores", {})) | {loaded.get("selected_model", "uniform")})
        loaded.update(
            stale=True,
            stale_reason=(
                "dataset or feature/engine version differs from the saved experiment "
                f"(saved feature {meta.get('feature_version')}, current {FEATURE_VERSION}; "
                f"saved engine {meta.get('engine_version')}, current {ENGINE_VERSION}); retrain before use"
            ),
            stale_models=names,
            current_fingerprint=current,
            load_warnings=errors,
        )
        return loaded
    if errors:
        raise ValueError("No loadable model artifacts: " + "; ".join(errors))
    return None
