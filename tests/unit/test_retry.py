import time
from unittest.mock import MagicMock

import pytest

from genmedia.retry import RetryWrapper, RetryableError, NonRetryableError


def test_succeeds_first_try():
    fn = MagicMock(return_value="result")
    wrapper = RetryWrapper(max_retries=3, base_delay=0.01, max_delay=0.1)
    result = wrapper.execute(fn)
    assert result == "result"
    assert fn.call_count == 1
    assert wrapper.attempts == 1


def test_retries_on_retryable_error():
    fn = MagicMock(side_effect=[RetryableError("429"), RetryableError("429"), "result"])
    wrapper = RetryWrapper(max_retries=5, base_delay=0.01, max_delay=0.1)
    result = wrapper.execute(fn)
    assert result == "result"
    assert fn.call_count == 3
    assert wrapper.attempts == 3


def test_exhausts_retries():
    fn = MagicMock(side_effect=RetryableError("429"))
    wrapper = RetryWrapper(max_retries=3, base_delay=0.01, max_delay=0.1)
    with pytest.raises(RetryableError):
        wrapper.execute(fn)
    assert fn.call_count == 3
    assert wrapper.attempts == 3


def test_does_not_retry_non_retryable():
    fn = MagicMock(side_effect=NonRetryableError("400 bad request"))
    wrapper = RetryWrapper(max_retries=5, base_delay=0.01, max_delay=0.1)
    with pytest.raises(NonRetryableError):
        wrapper.execute(fn)
    assert fn.call_count == 1


def test_respects_retry_after():
    err = RetryableError("429")
    err.retry_after = 0.05
    fn = MagicMock(side_effect=[err, "result"])
    wrapper = RetryWrapper(max_retries=3, base_delay=0.01, max_delay=10.0)
    start = time.monotonic()
    result = wrapper.execute(fn)
    elapsed = time.monotonic() - start
    assert result == "result"
    assert elapsed >= 0.04


def test_backoff_increases():
    errors = [RetryableError("429") for _ in range(3)]
    fn = MagicMock(side_effect=errors + ["result"])
    wrapper = RetryWrapper(max_retries=5, base_delay=1.0, max_delay=100.0)
    wrapper.execute(fn)
    assert len(wrapper.delays) == 3
    assert wrapper.delays[1] > wrapper.delays[0]


def test_max_delay_cap():
    wrapper = RetryWrapper(max_retries=10, base_delay=1.0, max_delay=5.0)
    delay = wrapper._calculate_delay(attempt=10)
    assert delay <= 5.0 * 1.5


def test_env_var_overrides(monkeypatch):
    monkeypatch.setenv("GENMEDIA_MAX_RETRIES", "2")
    monkeypatch.setenv("GENMEDIA_RETRY_BASE_DELAY", "0.05")
    wrapper = RetryWrapper()
    assert wrapper.max_retries == 2
    assert wrapper.base_delay == 0.05


def test_retryable_error_preserves_status_code():
    err = RetryableError("429 Too Many Requests", status_code=429)
    assert err.status_code == 429


def test_retryable_error_preserves_status_code_500():
    err = RetryableError("500 Internal Server Error", status_code=500)
    assert err.status_code == 500
