from __future__ import annotations

import threading

import pytest

from code4scene.evaluation import vlm_concurrency


@pytest.fixture(autouse=True)
def reset_runtime(monkeypatch: pytest.MonkeyPatch):
    for name in (
        "CODE4SCENE_VLM_MAX_CONCURRENCY",
        "CODE4SCENE_VLM_MAX_RETRIES",
        "CODE4SCENE_VLM_ADAPTIVE_CONCURRENCY",
    ):
        monkeypatch.delenv(name, raising=False)
    vlm_concurrency._reset_for_tests()
    yield
    vlm_concurrency._reset_for_tests()


def test_default_runtime_is_explicit_32_way_adaptive():
    assert vlm_concurrency.runtime_config().to_dict() == {
        "policy_id": "vlm-adaptive-concurrency-v1",
        "max_concurrency": 32,
        "max_retries": 3,
        "adaptive": True,
        "retry_backoff_initial_s": 0.5,
        "retry_backoff_max_s": 4.0,
    }


def test_parallel_map_starts_all_32_workers_and_preserves_order():
    barrier = threading.Barrier(32)
    values = tuple(reversed(range(32)))

    def work(value: int) -> int:
        barrier.wait(timeout=5.0)
        return value * 10

    assert vlm_concurrency.parallel_map(values, work) == tuple(
        value * 10 for value in values
    )


def test_congestion_halves_limit_and_retries_transient_failure():
    config = vlm_concurrency.VLMConcurrencyConfig(
        max_concurrency=8,
        max_retries=2,
        retry_backoff_initial_s=0.001,
        retry_backoff_max_s=0.002,
    )
    attempts = 0

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("busy")
        return "ok"

    assert (
        vlm_concurrency.request_with_adaptive_concurrency(
            "http://vlm.test/v1",
            operation,
            retryable=vlm_concurrency.retryable_http_error,
            config=config,
        )
        == "ok"
    )
    snapshot = vlm_concurrency.runtime_snapshot()
    # runtime_snapshot reports the env-selected default controller only; query
    # the test controller directly to validate its adaptive state.
    assert snapshot["endpoints"] == []
    gate = vlm_concurrency._gate("http://vlm.test/v1", config)
    assert gate.snapshot() == {
        "endpoint": "http://vlm.test/v1",
        "configured_max_concurrency": 8,
        "current_concurrency_limit": 4,
        "peak_active_requests": 1,
        "request_attempt_count": 2,
        "retry_count": 1,
        "congestion_event_count": 1,
        "multiplicative_decrease_count": 1,
        "additive_increase_count": 0,
    }


def test_non_retryable_error_is_not_replayed():
    attempts = 0

    def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise ValueError("malformed structured output")

    with pytest.raises(ValueError, match="malformed"):
        vlm_concurrency.request_with_adaptive_concurrency(
            "http://vlm.test/v1",
            operation,
            retryable=vlm_concurrency.retryable_http_error,
        )
    assert attempts == 1
