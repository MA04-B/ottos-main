"""Bounded, cancellable historical-match search using the existing generator."""

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, replace
from numbers import Integral

from src.features.engineering import NUMBER_COLUMNS, validate_frame
from src.prediction.generator import generate_combinations

LOG = logging.getLogger(__name__)
MAX_SEARCH_ATTEMPTS = 10_000
ACTIVE_STATUSES = {"running", "cancelling"}


@dataclass(frozen=True)
class SearchSnapshot:
    search_id: str = ""
    status: str = "idle"
    attempts: int = 0
    checked_combinations: int = 0
    max_attempts: int = 0
    initial_seed: int = 0
    last_seed: int | None = None
    started_at: float = 0.
    elapsed_seconds: float = 0.
    matched_index: int | None = None
    historical_draws: tuple[dict, ...] = ()
    portfolio: tuple[dict, ...] = ()
    error: str = ""


class MagicSearchJob:
    """One isolated search per browser session; workers never access Streamlit."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._state = SearchSnapshot()
        self.future = None

    def snapshot(self) -> SearchSnapshot:
        with self._lock:
            state = self._state
            if state.status in ACTIVE_STATUSES:
                state = replace(state, elapsed_seconds=time.monotonic() - state.started_at)
            return deepcopy(state)

    def start(self, frame, generation_options, max_attempts=1000):
        if (not isinstance(max_attempts, Integral) or isinstance(max_attempts, bool)
                or not 1 <= max_attempts <= MAX_SEARCH_ATTEMPTS):
            raise ValueError(f"Maximum attempts must be an integer between 1 and {MAX_SEARCH_ATTEMPTS:,}")
        options = deepcopy(generation_options)
        seed = options.get("seed", 42)
        if not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0:
            raise ValueError("Search seed must be a nonnegative integer")
        if "count" in options and options["count"] != 10:
            raise ValueError("Magic Generate must use ten-combination portfolios")
        data = frame.copy(deep=True)
        with self._lock:
            if self._state.status in ACTIVE_STATUSES:
                raise RuntimeError("A historical-match search is already running")
            self._cancel.clear()
            self._state = SearchSnapshot(
                search_id=str(uuid.uuid4()), status="running", max_attempts=int(max_attempts),
                initial_seed=int(seed), started_at=time.monotonic())
            # Shutting down without waiting keeps the UI nonblocking, while
            # releasing the worker thread as soon as this bounded task ends.
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lottery-magic")
            try:
                self.future = executor.submit(self._run, data, options)
            except RuntimeError as exc:
                self._state = replace(self._state, status="failed", error=f"Could not start search: {exc}")
                LOG.exception("Could not start historical-match worker")
                raise
            finally:
                executor.shutdown(wait=False)
            return self._state.search_id

    def stop(self):
        with self._lock:
            if self._state.status == "running":
                self._cancel.set()
                self._state = replace(self._state, status="cancelling")

    def _update(self, **values):
        with self._lock:
            self._state = replace(self._state, **values)

    def _finish(self, status, **values):
        with self._lock:
            if self._cancel.is_set() and status in ("matched", "exhausted"):
                status, values = "cancelled", {}
            self._state = replace(self._state, status=status,
                                  elapsed_seconds=time.monotonic() - self._state.started_at, **values)
            LOG.info("Historical-match search %s: %s after %s attempts, %s combinations checked",
                     self._state.search_id, status, self._state.attempts, self._state.checked_combinations)

    def _run(self, frame, options):
        try:
            data = validate_frame(frame)
            if data.empty:
                raise ValueError("Historical drawings are required for Magic Generate")
            history = {}
            for row in data.itertuples(index=False):
                numbers = tuple(int(getattr(row, column)) for column in NUMBER_COLUMNS)
                history.setdefault(numbers, []).append({
                    "year": int(row.year), "week": int(row.week),
                    "draw_date": row.draw_date, "numbers": list(numbers),
                })
            snapshot = self.snapshot()
            checked = 0
            LOG.info("Starting historical-match search %s (limit %s, seed %s)",
                     snapshot.search_id, snapshot.max_attempts, snapshot.initial_seed)
            for attempt in range(1, snapshot.max_attempts + 1):
                if self._cancel.is_set():
                    self._finish("cancelled")
                    return
                seed = snapshot.initial_seed + attempt - 1
                self._update(attempts=attempt, last_seed=seed)
                portfolio = generate_combinations(data, **{**options, "seed": seed})
                if self._cancel.is_set():
                    self._finish("cancelled")
                    return
                if len(portfolio) != 10:
                    raise ValueError("The generation engine did not return a ten-combination portfolio")
                for index, combination in enumerate(portfolio):
                    checked += 1
                    # Five-of-five is exact set equality, not a partial match.
                    matches = history.get(tuple(sorted(combination["numbers"])))
                    if matches:
                        self._update(checked_combinations=checked)
                        self._finish("matched", matched_index=index,
                                     historical_draws=tuple(matches), portfolio=tuple(portfolio))
                        return
                self._update(checked_combinations=checked)
            self._finish("exhausted")
        except Exception as exc:
            LOG.exception("Historical-match search failed")
            self._finish("failed", error=f"{type(exc).__name__}: {exc}")
