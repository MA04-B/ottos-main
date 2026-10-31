"""Prior-only, per-number feature engineering for the 5-of-90 lottery.

Data contract: a DataFrame with ``year``, ``week``, ``draw_date`` (ISO string or
None/empty) and ``number_1`` .. ``number_5``. Draws are ordered by
(year, week) with explicit dates used only as a tie-breaker. Missing dates are
never invented; time features fall back to ISO year/week arithmetic.

Feature row ``X[t]`` describes every number *before* draw ``t`` happens: it is
computed exclusively from draws ``0 .. t-1`` plus the schedule of draw ``t``
(its year/week and, when published, its date). ``X[n]`` is the feature row for
the next, not yet drawn, draw.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.validator import validate_draw

N_NUMBERS = 90
PICK = 5
BASE_RATE = PICK / N_NUMBERS
NUMBER_COLUMNS = [f"number_{i}" for i in range(1, PICK + 1)]
REQUIRED_COLUMNS = ["year", "week", "draw_date", *NUMBER_COLUMNS]
FEATURE_VERSION = 3
WINDOWS = (10, 25, 50)
EW_HALFLIVES = {"ew_fast": 10.0, "ew_mid": 30.0, "ew_slow": 100.0}
GAP_CAP = 300
EXPECTED_GAP = N_NUMBERS / PICK

FEATURE_NAMES = [
    "number_norm",
    "is_odd",
    "decade",
    "freq_all",
    *[f"freq_w{w}" for w in WINDOWS],
    "gap",
    *EW_HALFLIVES,
    "prev_draw",
    "prev2_draw",
    "pair_score",
    "days_since_prev",
    "week_sin",
    "week_cos",
    "log_history",
    "count_all",
    "history",
]
# Raw, monotonically growing counters are used by the statistical models only.
NON_ML_FEATURES = ("count_all", "history")


def _clean_date(value) -> str | None:
    """Return an ISO date string or None. Never fabricates a date."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (dt.date, pd.Timestamp)):
        return pd.Timestamp(value).date().isoformat()
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "nat", "null"}:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"unparseable draw_date {text!r}")
    return parsed.date().isoformat()


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and chronologically order draws.

    Returns a new frame with integer year/week, ``draw_date`` as ISO string or
    None, and per-row ascending ``number_1..number_5``. Ordering is by
    (year, week, explicit date if any, original position). Exact duplicate
    rows are dropped; invalid rows raise ``ValueError``.
    """
    if frame is None:
        raise ValueError("frame is required")
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing required columns: {missing}")
    data = frame[REQUIRED_COLUMNS].copy().reset_index(drop=True)
    if data.empty:
        return data.assign(year=data["year"].astype("int64"), week=data["week"].astype("int64"))
    for col in ["year", "week", *NUMBER_COLUMNS]:
        values = pd.to_numeric(data[col], errors="coerce")
        bad = values.isna() | (values != np.round(values))
        if bad.any():
            raise ValueError(f"column {col!r} has missing/non-integer values at rows {list(np.flatnonzero(bad))[:10]}")
        data[col] = values.astype("int64")
    if ((data["week"] < 1) | (data["week"] > 53)).any():
        raise ValueError("week must be within 1..53")
    numbers = data[NUMBER_COLUMNS].to_numpy()
    if ((numbers < 1) | (numbers > N_NUMBERS)).any():
        raise ValueError(f"numbers must be within 1..{N_NUMBERS}")
    numbers = np.sort(numbers, axis=1)
    duplicated = (np.diff(numbers, axis=1) == 0).any(axis=1)
    if duplicated.any():
        raise ValueError(f"draws must contain {PICK} distinct numbers; bad rows {list(np.flatnonzero(duplicated))[:10]}")
    data[NUMBER_COLUMNS] = numbers
    data["draw_date"] = [_clean_date(v) for v in data["draw_date"]]
    for row in data.itertuples(index=False):
        validate_draw({"year": int(row.year), "week": int(row.week), "draw_date": row.draw_date,
                       "numbers": [int(getattr(row, c)) for c in NUMBER_COLUMNS]})
    data = data.drop_duplicates().reset_index(drop=True)
    if data.duplicated(["year", "week"]).any():
        raise ValueError("Conflicting duplicate year/week drawings")
    known_dates = data["draw_date"].dropna()
    if known_dates.duplicated().any():
        raise ValueError("Duplicate drawing dates")
    data["_date_key"] = data["draw_date"].fillna("")
    data["_pos"] = np.arange(len(data))
    data = data.sort_values(["year", "week", "_date_key", "_pos"], kind="mergesort")
    data = data.drop(columns=["_date_key", "_pos"]).reset_index(drop=True)
    data["draw_date"] = data["draw_date"].astype(object).where(data["draw_date"].notna(), None)
    return data


def draws_array(frame: pd.DataFrame) -> np.ndarray:
    """(n, 5) int array of sorted numbers from an already validated frame."""
    if len(frame) == 0:
        return np.empty((0, PICK), dtype=np.int64)
    return frame[NUMBER_COLUMNS].to_numpy(dtype=np.int64)


def draw_matrix(draws: np.ndarray) -> np.ndarray:
    """(n, 90) 0/1 indicator matrix."""
    draws = np.asarray(draws, dtype=np.int64)
    matrix = np.zeros((len(draws), N_NUMBERS), dtype=np.int8)
    if len(draws):
        rows = np.repeat(np.arange(len(draws)), draws.shape[1])
        matrix[rows, draws.ravel() - 1] = 1
    return matrix


def dataset_fingerprint(frame: pd.DataFrame, validated: bool = False) -> str:
    """SHA-256 over a canonical JSON rendering of the validated draws.

    Null dates are encoded as JSON ``null`` so the fingerprint is independent of
    whether the source used None, NaN or an empty string.
    """
    data = frame if validated else validate_frame(frame)
    rows = [
        [int(r.year), int(r.week), r.draw_date if r.draw_date else None, *[int(getattr(r, c)) for c in NUMBER_COLUMNS]]
        for r in data.itertuples(index=False)
    ]
    payload = json.dumps({"version": 1, "columns": REQUIRED_COLUMNS, "rows": rows}, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _week_ordinal(year: int, week: int) -> int:
    """Ordinal of the Monday of ISO (year, week); used only for gap arithmetic."""
    try:
        return dt.date.fromisocalendar(int(year), int(week), 1).toordinal()
    except ValueError:
        return dt.date.fromisocalendar(int(year), 1, 1).toordinal() + 7 * (int(week) - 1)


def _time_features(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (days_since_prev, week_of_year) arrays of length n + 1.

    Only the previous draw's schedule is used. Published target drawing dates
    are retrospective metadata and are not assumed known before the draw.
    The next week is a schedule assumption, not an invented historical date.
    """
    n = len(frame)
    days = np.full(n + 1, 7.0)
    weeks = np.ones(n + 1)
    if n == 0:
        return days, weeks
    years = frame["year"].to_numpy(dtype=np.int64)
    wk = frame["week"].to_numpy(dtype=np.int64)
    ords = np.array([_week_ordinal(y, w) for y, w in zip(years, wk)], dtype=np.int64)
    for t in range(1, n + 1):
        weeks[t] = dt.date.fromordinal(int(ords[t - 1]) + 7).isocalendar()[1]
    return np.clip(days, 0.0, 365.0), weeks


@dataclass
class FeatureSet:
    """Feature tensor ``X`` (n+1, 90, F) and labels ``Y`` (n, 90)."""

    X: np.ndarray
    Y: np.ndarray
    feature_names: list[str]
    draws: np.ndarray
    labels: list[str]

    @property
    def n_draws(self) -> int:
        return int(self.Y.shape[0])

    def column(self, name: str) -> int:
        return self.feature_names.index(name)

    @property
    def ml_indices(self) -> list[int]:
        return [i for i, name in enumerate(self.feature_names) if name not in NON_ML_FEATURES]


def _draw_labels(frame: pd.DataFrame) -> list[str]:
    labels = []
    for r in frame.itertuples(index=False):
        label = f"{int(r.year)}-W{int(r.week):02d}"
        labels.append(f"{label} ({r.draw_date})" if r.draw_date else label)
    return labels


def build_features(frame: pd.DataFrame, validated: bool = False) -> FeatureSet:
    """Build prior-only features for every draw plus the upcoming draw."""
    data = frame if validated else validate_frame(frame)
    draws = draws_array(data)
    n = len(draws)
    M = draw_matrix(draws).astype(np.float64)
    cs = np.vstack([np.zeros((1, N_NUMBERS)), np.cumsum(M, axis=0)])
    col = {name: i for i, name in enumerate(FEATURE_NAMES)}
    X = np.zeros((n + 1, N_NUMBERS, len(FEATURE_NAMES)), dtype=np.float64)
    numbers = np.arange(1, N_NUMBERS + 1, dtype=np.float64)
    X[:, :, col["number_norm"]] = numbers / N_NUMBERS
    X[:, :, col["is_odd"]] = numbers % 2
    X[:, :, col["decade"]] = ((numbers - 1) // 10) / 8.0
    days, weeks = _time_features(data)
    last = np.full(N_NUMBERS, -1, dtype=np.int64)
    ew = {k: np.full(N_NUMBERS, BASE_RATE) for k in EW_HALFLIVES}
    decay = {k: 0.5 ** (1.0 / hl) for k, hl in EW_HALFLIVES.items()}
    pair = np.zeros((N_NUMBERS, N_NUMBERS))
    for t in range(n + 1):
        row = X[t]
        count = cs[t]
        row[:, col["freq_all"]] = count / t if t else BASE_RATE
        for w in WINDOWS:
            span = min(w, t)
            row[:, col[f"freq_w{w}"]] = (cs[t] - cs[t - span]) / span if span else BASE_RATE
        gap = np.where(last >= 0, t - last - 1, t)
        row[:, col["gap"]] = np.minimum(gap, GAP_CAP) / EXPECTED_GAP
        for k in EW_HALFLIVES:
            row[:, col[k]] = ew[k]
        if t >= 1:
            row[:, col["prev_draw"]] = M[t - 1]
            prev = draws[t - 1] - 1
            cond = pair[prev] / np.maximum(count[prev], 1.0)[:, None]
            in_prev = M[t - 1]
            # Mean P(k | j) over the other numbers j of the previous draw.
            row[:, col["pair_score"]] = (cond.sum(axis=0) - in_prev) / (PICK - in_prev)
        else:
            row[:, col["pair_score"]] = (PICK - 1) / (N_NUMBERS - 1)
        if t >= 2:
            row[:, col["prev2_draw"]] = M[t - 2]
        row[:, col["days_since_prev"]] = days[t] / 7.0
        angle = 2.0 * np.pi * (weeks[t] - 1) / 53.0
        row[:, col["week_sin"]] = np.sin(angle)
        row[:, col["week_cos"]] = np.cos(angle)
        row[:, col["log_history"]] = np.log1p(t)
        row[:, col["count_all"]] = count
        row[:, col["history"]] = t
        if t < n:
            idx = draws[t] - 1
            last[idx] = t
            for k in EW_HALFLIVES:
                ew[k] = decay[k] * ew[k] + (1.0 - decay[k]) * M[t]
            pair[np.ix_(idx, idx)] += 1.0
    return FeatureSet(X=X, Y=M.astype(np.int8), feature_names=list(FEATURE_NAMES), draws=draws, labels=_draw_labels(data))


def number_statistics(frame: pd.DataFrame | None, halflife: float = 10.0, recent_window: int = 50) -> dict:
    """Descriptive per-number statistics over the full validated history."""
    if frame is None or len(frame) == 0:
        draws = np.empty((0, PICK), dtype=np.int64)
    else:
        draws = draws_array(validate_frame(frame))
    n = len(draws)
    M = draw_matrix(draws).astype(np.float64)
    counts = M.sum(axis=0) if n else np.zeros(N_NUMBERS)
    window = min(recent_window, n)
    recent = M[n - window:].sum(axis=0) if window else np.zeros(N_NUMBERS)
    decay = 0.5 ** (1.0 / halflife)
    ew = np.full(N_NUMBERS, BASE_RATE)
    for t in range(n):
        ew = decay * ew + (1.0 - decay) * M[t]
    since = np.full(N_NUMBERS, -1, dtype=np.int64)
    for k in range(N_NUMBERS):
        hits = np.flatnonzero(M[:, k]) if n else np.empty(0, dtype=np.int64)
        if len(hits):
            since[k] = n - 1 - hits[-1]
    return {
        "n_draws": n,
        "counts": counts,
        "frequency": (counts + 0.5) / (n + 0.5 / BASE_RATE),
        "recent_counts": recent,
        "recent_window": window,
        "ew_frequency": ew,
        "draws_since_seen": since,
    }
