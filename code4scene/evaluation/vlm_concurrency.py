"""Process-wide concurrency, congestion control, and retries for VLM calls."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import os
import threading
import time
from typing import Any, TypeVar
import urllib.error


MAX_SUPPORTED_CONCURRENCY = 32
DEFAULT_MAX_CONCURRENCY = 32
DEFAULT_MAX_RETRIES = 3
_MAX_RETRIES = 8
_CONFIG_ENV = "CODE4SCENE_VLM_MAX_CONCURRENCY"
_RETRIES_ENV = "CODE4SCENE_VLM_MAX_RETRIES"
_ADAPTIVE_ENV = "CODE4SCENE_VLM_ADAPTIVE_CONCURRENCY"
_POLICY_ID = "vlm-adaptive-concurrency-v1"
_T = TypeVar("_T")
_R = TypeVar("_R")


def _bounded_env_int(name: str, default: int, *, upper: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if not 1 <= value <= upper:
        raise ValueError(f"{name} must be between 1 and {upper}")
    return value


def _bounded_env_nonnegative_int(name: str, default: int, *, upper: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if not 0 <= value <= upper:
        raise ValueError(f"{name} must be between 0 and {upper}")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True)
class VLMConcurrencyConfig:
    """Frozen operational policy, read once from the environment."""

    policy_id: str = _POLICY_ID
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    max_retries: int = DEFAULT_MAX_RETRIES
    adaptive: bool = True
    retry_backoff_initial_s: float = 0.5
    retry_backoff_max_s: float = 4.0

    def __post_init__(self) -> None:
        if not 1 <= self.max_concurrency <= MAX_SUPPORTED_CONCURRENCY:
            raise ValueError(
                "max_concurrency must be between 1 and "
                f"{MAX_SUPPORTED_CONCURRENCY}"
            )
        if not 0 <= self.max_retries <= _MAX_RETRIES:
            raise ValueError(f"max_retries must be between 0 and {_MAX_RETRIES}")
        if self.retry_backoff_initial_s <= 0:
            raise ValueError("retry_backoff_initial_s must be positive")
        if self.retry_backoff_max_s < self.retry_backoff_initial_s:
            raise ValueError(
                "retry_backoff_max_s must be at least retry_backoff_initial_s"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def runtime_config() -> VLMConcurrencyConfig:
    """Read the concurrency settings from the environment."""

    return VLMConcurrencyConfig(
        max_concurrency=_bounded_env_int(
            _CONFIG_ENV,
            DEFAULT_MAX_CONCURRENCY,
            upper=MAX_SUPPORTED_CONCURRENCY,
        ),
        max_retries=_bounded_env_nonnegative_int(
            _RETRIES_ENV,
            DEFAULT_MAX_RETRIES,
            upper=_MAX_RETRIES,
        ),
        adaptive=_env_bool(_ADAPTIVE_ENV, True),
    )


class _AdaptiveGate:
    def __init__(self, endpoint: str, config: VLMConcurrencyConfig) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.config = config
        self._condition = threading.Condition()
        self._current_limit = config.max_concurrency
        self._active = 0
        self._peak_active = 0
        self._success_streak = 0
        self._next_decrease_at = 0.0
        self._request_count = 0
        self._retry_count = 0
        self._congestion_count = 0
        self._decrease_count = 0
        self._increase_count = 0

    @contextmanager
    def slot(self) -> Iterator[None]:
        with self._condition:
            while self._active >= self._current_limit:
                self._condition.wait()
            self._active += 1
            self._request_count += 1
            self._peak_active = max(self._peak_active, self._active)
        try:
            yield
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    def record_retry(self) -> None:
        with self._condition:
            self._retry_count += 1

    def record_congestion(self) -> None:
        with self._condition:
            self._congestion_count += 1
            self._success_streak = 0
            now = time.monotonic()
            if (
                self.config.adaptive
                and self._current_limit > 1
                and now >= self._next_decrease_at
            ):
                self._current_limit = max(1, self._current_limit // 2)
                self._decrease_count += 1
                self._next_decrease_at = now + self.config.retry_backoff_max_s
            self._condition.notify_all()

    def record_success(self) -> None:
        with self._condition:
            if self._current_limit >= self.config.max_concurrency:
                self._success_streak = 0
                return
            self._success_streak += 1
            # Additive recovery avoids immediately recreating the overload
            # that caused multiplicative decrease.
            if self._success_streak >= max(4, self._current_limit):
                self._current_limit += 1
                self._increase_count += 1
                self._success_streak = 0
                self._condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "endpoint": self.endpoint,
                "configured_max_concurrency": self.config.max_concurrency,
                "current_concurrency_limit": self._current_limit,
                "peak_active_requests": self._peak_active,
                "request_attempt_count": self._request_count,
                "retry_count": self._retry_count,
                "congestion_event_count": self._congestion_count,
                "multiplicative_decrease_count": self._decrease_count,
                "additive_increase_count": self._increase_count,
            }


_REGISTRY_LOCK = threading.Lock()
_GATES: dict[tuple[str, VLMConcurrencyConfig], _AdaptiveGate] = {}


def _gate(endpoint: str, config: VLMConcurrencyConfig) -> _AdaptiveGate:
    key = (endpoint.rstrip("/"), config)
    with _REGISTRY_LOCK:
        if key not in _GATES:
            _GATES[key] = _AdaptiveGate(key[0], config)
        return _GATES[key]


def request_with_adaptive_concurrency(
    endpoint: str,
    operation: Callable[[], _R],
    *,
    retryable: Callable[[Exception], bool],
    config: VLMConcurrencyConfig | None = None,
) -> _R:
    """Run one HTTP request under the shared gate and retry congestion only."""

    selected = config or runtime_config()
    gate = _gate(endpoint, selected)
    for attempt in range(selected.max_retries + 1):
        try:
            with gate.slot():
                result = operation()
        except Exception as error:
            if not retryable(error):
                raise
            gate.record_congestion()
            if attempt >= selected.max_retries:
                raise
            gate.record_retry()
            delay = min(
                selected.retry_backoff_initial_s * (2**attempt),
                selected.retry_backoff_max_s,
            )
            time.sleep(delay)
            continue
        gate.record_success()
        return result
    raise AssertionError("adaptive VLM retry loop exhausted without returning")


def retryable_http_error(error: Exception) -> bool:
    """Whether a wire error represents transient endpoint congestion."""

    if isinstance(error, urllib.error.HTTPError):
        return error.code in {408, 409, 425, 429, 500, 502, 503, 504}
    return isinstance(
        error,
        (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            OSError,
        ),
    )


def parallel_map(
    values: Sequence[_T],
    function: Callable[[_T], _R],
    *,
    max_workers: int | None = None,
    thread_name_prefix: str = "scenebench-vlm",
) -> tuple[_R, ...]:
    """Evaluate independent VLM work concurrently while preserving input order."""

    items = tuple(values)
    if not items:
        return ()
    configured = runtime_config().max_concurrency
    requested = configured if max_workers is None else min(max_workers, configured)
    workers = min(requested, len(items))
    if workers <= 1:
        return tuple(function(value) for value in items)
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix=thread_name_prefix,
    ) as executor:
        return tuple(executor.map(function, items))


def runtime_snapshot() -> dict[str, Any]:
    config = runtime_config()
    with _REGISTRY_LOCK:
        gates = [
            gate.snapshot()
            for (endpoint, gate_config), gate in sorted(
                _GATES.items(), key=lambda item: item[0][0]
            )
            if gate_config == config
        ]
    return {
        "schema_version": "1.0",
        "config": config.to_dict(),
        "endpoints": gates,
    }


def _reset_for_tests() -> None:
    with _REGISTRY_LOCK:
        _GATES.clear()


__all__ = [
    "DEFAULT_MAX_CONCURRENCY",
    "DEFAULT_MAX_RETRIES",
    "MAX_SUPPORTED_CONCURRENCY",
    "VLMConcurrencyConfig",
    "parallel_map",
    "request_with_adaptive_concurrency",
    "retryable_http_error",
    "runtime_config",
    "runtime_snapshot",
]
