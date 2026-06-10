"""Lesson 02b — prompt caching, the silent-invalidation failure.

A common production mistake: a "static" system prompt that actually
includes the current timestamp, request id, or other per-request
content. The cache_control marker is present, but every call writes a
fresh cache entry instead of reading an existing one. The metrics show
`created` climbing every turn while `read` stays at zero.

We reproduce the failure here by stamping the system prompt with a
high-precision timestamp on every call. Then we fix it by moving the
volatile content into the user message, below the cache breakpoint.

Run:
    uv run python lessons/02b_prompt_caching/02_cache_invalidation.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import chat

CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"


def load_handbook() -> str:
    docs: list[str] = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        docs.append(f"## {path.name}\n\n{path.read_text(encoding='utf-8')}")
    return "\n\n---\n\n".join(docs)


HANDBOOK_3X = "\n\n---\n\n".join([load_handbook()] * 3)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def call_broken(user_text: str) -> tuple[int, int]:
    """The broken pattern: timestamp baked into the cached prefix.

    Every call has a different timestamp → every prefix is unique →
    every call writes a new cache entry instead of reading one.
    """
    prefix = (
        f"You are Acme's handbook assistant. Request time: {now_iso()}.\n\n"
        f"=== HANDBOOK ===\n\n{HANDBOOK_3X}\n\n=== END HANDBOOK ==="
    )
    response = chat(
        system=[{
            "type": "text",
            "text": prefix,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": user_text}],
    )
    return response.cache_creation_input_tokens, response.cache_read_input_tokens


def call_fixed(user_text: str) -> tuple[int, int]:
    """The fixed pattern: prefix is genuinely static; volatile content
    moves to the user message, which is below the cache breakpoint."""
    prefix = (
        f"You are Acme's handbook assistant.\n\n"
        f"=== HANDBOOK ===\n\n{HANDBOOK_3X}\n\n=== END HANDBOOK ==="
    )
    response = chat(
        system=[{
            "type": "text",
            "text": prefix,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{
            "role": "user",
            "content": f"(request time: {now_iso()}) {user_text}",
        }],
    )
    return response.cache_creation_input_tokens, response.cache_read_input_tokens


def report(label: str, callable_) -> None:
    print(f"\n--- {label} ---")
    for i, q in enumerate(["vacation days?", "wellness benefit?", "remote work policy?"], start=1):
        created, read = callable_(q)
        print(f"  turn {i}: created={created:5d}  read={read:5d}")


if __name__ == "__main__":
    report("BROKEN: timestamp in the cached prefix", call_broken)
    report("FIXED:  timestamp moved to the user message", call_fixed)

    print(
        "\nBroken pattern: `created` is high every turn, `read` is always 0 — "
        "you're paying the write price on every call. "
        "Fixed pattern: turn 1 writes, turns 2-3 read."
    )
