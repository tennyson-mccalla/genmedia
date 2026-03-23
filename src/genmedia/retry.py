import os
import random
import time


class RetryableError(Exception):
    def __init__(self, message: str, retry_after: float | None = None, status_code: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after
        self.status_code = status_code


class NonRetryableError(Exception):
    pass


def classify_sdk_error(exc: Exception) -> RetryableError | NonRetryableError:
    """Translate a google-genai SDK exception into RetryableError or NonRetryableError."""
    from google.genai import errors as genai_errors

    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    message = str(exc)

    if isinstance(exc, genai_errors.APIError):
        status_code = getattr(exc, "status", status_code)

    retry_after = None
    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", {})
        ra = headers.get("Retry-After") or headers.get("retry-after")
        if ra:
            try:
                retry_after = float(ra)
            except ValueError:
                pass

    if status_code in (429, 500, 503):
        return RetryableError(message, retry_after=retry_after, status_code=status_code)

    return NonRetryableError(message)


class RetryWrapper:
    def __init__(
        self,
        max_retries: int | None = None,
        base_delay: float | None = None,
        max_delay: float = 60.0,
    ):
        self.max_retries = max_retries if max_retries is not None else int(os.environ.get("GENMEDIA_MAX_RETRIES", "5"))
        self.base_delay = base_delay if base_delay is not None else float(os.environ.get("GENMEDIA_RETRY_BASE_DELAY", "2.0"))
        self.max_delay = max_delay
        self.attempts = 0
        self.delays: list[float] = []

    def _calculate_delay(self, attempt: int) -> float:
        delay = self.base_delay * (2 ** (attempt - 1))
        delay = min(delay, self.max_delay)
        jitter = random.uniform(0, delay * 0.5)
        return delay + jitter

    def execute(self, fn):
        self.attempts = 0
        self.delays = []
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            self.attempts = attempt
            try:
                return fn()
            except NonRetryableError:
                raise
            except RetryableError as e:
                last_error = e
                if attempt == self.max_retries:
                    raise
                if e.retry_after is not None:
                    delay = e.retry_after
                else:
                    delay = self._calculate_delay(attempt)
                self.delays.append(delay)
                time.sleep(delay)

        raise last_error
