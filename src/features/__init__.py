"""Leakage-free feature engineering for 5-of-90 lottery draws."""

from src.features.engineering import (
    BASE_RATE,
    FEATURE_NAMES,
    N_NUMBERS,
    NUMBER_COLUMNS,
    PICK,
    FeatureSet,
    build_features,
    dataset_fingerprint,
    draw_matrix,
    draws_array,
    number_statistics,
    validate_frame,
)

__all__ = [
    "BASE_RATE",
    "FEATURE_NAMES",
    "N_NUMBERS",
    "NUMBER_COLUMNS",
    "PICK",
    "FeatureSet",
    "build_features",
    "dataset_fingerprint",
    "draw_matrix",
    "draws_array",
    "number_statistics",
    "validate_frame",
]
