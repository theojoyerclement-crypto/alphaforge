"""Retries with exponential backoff, for transient venue errors only.

Only errors that are safe to retry are retried (network failures, timeouts, rate limits).
A rejected order or an authentication error will fail the same way every time; retrying
those would only hide the problem and, for orders, risk duplicates.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from alphaforge.utils.logging import get_logger

logger = get_logger(__name__)
T = TypeVar("T")


def call_with_retries(
    fn: Callable[[], T],
    *,
    retry_on: tuple[type[BaseException], ...],
    attempts: int = 5,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    sleep: Callable[[float], None] = time.sleep,
    label: str = "",
) -> T:
    """Call ``fn``; on an error in ``retry_on``, wait ``base_delay * 2**k`` (capped) and retry.

    Args:
        fn: Zero-argument callable (wrap arguments with a lambda).
        retry_on: Exception types considered transient.
        attempts: Total calls, including the first.
        sleep: Injected for tests, so retries run instantly.

    Raises:
        The last transient error after ``attempts`` calls, or any other error immediately.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except retry_on as exc:
            if attempt == attempts:
                raise
            delay = min(max_delay, base_delay * 2 ** (attempt - 1))
            logger.warning(
                "%s failed (%s); retry %d/%d in %.1fs", label, exc, attempt, attempts - 1, delay
            )
            sleep(delay)
    raise RuntimeError("unreachable")
