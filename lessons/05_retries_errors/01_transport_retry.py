"""Lesson 05 — transport-layer retries with backoff and jitter.

We wrap the chat() call in a hand-rolled retry decorator. The point is to
SEE the exponential backoff and the classification of retryable vs.
terminal errors. The SDK does this automatically with `max_retries`, but
knowing the shape lets you override intelligently.

Run:
    uv run python lessons/05_retries_errors/01_transport_retry.py
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import anthropic  # for exception types

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import chat, LLMResponse


# --- Classify exceptions ----------------------------------------------------


def is_retryable(exc: BaseException) -> bool:
    """Decide whether the SDK exception is worth retrying.

    Retryable: rate-limited (429), server hiccup (5xx), connection problems.
    Terminal: auth (401/403), bad request (400), content policy refusals.
    """
    if isinstance(exc, anthropic.RateLimitError):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return 500 <= exc.status_code < 600
    if isinstance(exc, anthropic.APIConnectionError):
        return True
    # AuthenticationError, PermissionError, BadRequestError → no retry.
    return False


# --- Retry helper -----------------------------------------------------------


def chat_with_retry(
    *args,
    max_attempts: int = 4,
    base_delay: float = 0.8,
    max_delay: float = 16.0,
    **kwargs,
) -> LLMResponse:
    """Exponential backoff with jitter. Caps at max_attempts."""
    for attempt in range(1, max_attempts + 1):
        try:
            return chat(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if not is_retryable(exc) or attempt == max_attempts:
                # Either we shouldn't retry, or we've run out of tries.
                raise

            # Exponential backoff: 0.8, 1.6, 3.2, 6.4 ... capped at max_delay.
            # Jitter: ± 30% so retries don't synchronize across clients.
            delay = min(base_delay * 2 ** (attempt - 1), max_delay)
            jitter = delay * random.uniform(-0.3, 0.3)
            sleep_for = delay + jitter
            print(
                f"  ! {type(exc).__name__} on attempt {attempt}/{max_attempts}; "
                f"sleeping {sleep_for:.2f}s before retry"
            )
            time.sleep(sleep_for)

    raise RuntimeError("unreachable")  # for type checkers


# --- The SDK's built-in retry control --------------------------------------
# The Anthropic SDK has its own retry logic. You can configure it with:
#
#   from anthropic import Anthropic
#   client = Anthropic(max_retries=5)        # default is 2
#   client.messages.create(..., timeout=30)  # per-request timeout
#
# Use the built-in retry for ordinary cases. Hand-roll only when you need
# per-call control (e.g., longer backoff for known-flaky tools, or you
# want to log/measure each retry).


if __name__ == "__main__":
    response = chat_with_retry(
        messages=[{"role": "user", "content": "In one sentence: what is HTTP 429?"}]
    )
    print(response.text)
