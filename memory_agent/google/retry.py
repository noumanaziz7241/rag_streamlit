"""Retry helpers for transient Google Gemini / Vertex errors."""

from __future__ import annotations

import random
import time
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")

RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}
RETRYABLE_MESSAGE_MARKERS = (
    "UNAVAILABLE",
    "RESOURCE_EXHAUSTED",
    "high demand",
    "overloaded",
    "try again later",
)


def is_retryable_google_error(exc: BaseException) -> bool:
    """True for rate limits and temporary capacity errors (e.g. HTTP 503)."""
    message = str(exc).lower()
    if any(marker.lower() in message for marker in RETRYABLE_MESSAGE_MARKERS):
        return True
    if any(f"{code}" in str(exc) for code in RETRYABLE_HTTP_CODES):
        return True

    for attr in ("code", "status", "status_code", "http_status"):
        value = getattr(exc, attr, None)
        if value in RETRYABLE_HTTP_CODES:
            return True
        if isinstance(value, str) and value.upper() in {"UNAVAILABLE", "RESOURCE_EXHAUSTED"}:
            return True

    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
        if status in RETRYABLE_HTTP_CODES:
            return True

    return False


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 20.0,
) -> T:
    """Call ``fn`` with exponential backoff on retryable Google API errors."""
    last_exc: BaseException | None = None

    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts - 1 or not is_retryable_google_error(exc):
                raise
            delay = min(max_delay, base_delay * (2**attempt) + random.uniform(0, 0.5))
            time.sleep(delay)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("call_with_retry exhausted without result")


def call_with_model_fallback(
    models: Iterable[str],
    fn: Callable[[str], T],
    *,
    max_attempts_per_model: int = 3,
) -> T:
    """Try each model in order; retry transient errors before moving to the next."""
    model_list = list(dict.fromkeys(models))
    if not model_list:
        raise ValueError("At least one model name is required.")

    last_exc: BaseException | None = None
    for model_name in model_list:
        try:
            return call_with_retry(
                lambda model=model_name: fn(model),
                max_attempts=max_attempts_per_model,
            )
        except Exception as exc:
            last_exc = exc
            if not is_retryable_google_error(exc):
                raise

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("call_with_model_fallback exhausted without result")
