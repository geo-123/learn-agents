"""Lesson 02b: prompt caching, basic hit demo.

Two calls. Same large cache marked system prompt, different user
questions. Turn 1 writes the cache; turn 2 reads it. The hit shows up
as `cache_read_input_tokens > 0`.

Run:
    uv run python lessons/02b_prompt_caching/01_cache_hits.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import chat

CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"


def load_handbook() -> str:
    """Concatenate every doc in the corpus to form the stable prefix.

    The minimum cacheable prefix depends on the model: 512 tokens on the
    current generation (Haiku 5.5, Sonnet 5.5, Opus 5.5), up to 4096 on
    some older models. The Acme handbook (~3K tokens) clears the current one.
    """
    docs: list[str] = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        docs.append(f"## {path.name}\n\n{path.read_text(encoding='utf-8')}")
    return "\n\n---\n\n".join(docs)


HANDBOOK = load_handbook()

# The cache minimum is model dependent and not monotonic across
# generations: 512 tokens on Haiku 5.5, but 4096 on Haiku 4.5 and Opus 4.6.
# We repeat the handbook three times so the prefix clears every threshold,
# whichever model LEARN_AGENTS_MODEL selects.
SYSTEM_PREFIX = (
    "You are Acme's internal handbook assistant. Answer ONLY from the "
    "handbook below. If the handbook does not cover the question, say so. "
    "Cite the doc name when you can.\n\n"
    "=== HANDBOOK (canonical) ===\n\n" + HANDBOOK + "\n\n"
    "=== HANDBOOK (clarification copy) ===\n\n" + HANDBOOK + "\n\n"
    "=== HANDBOOK (escalation reference) ===\n\n" + HANDBOOK + "\n\n"
    "=== END HANDBOOK ==="
)


def ask(user_text: str) -> None:
    """One LLM call with cache_control marking the system prompt as
    the stable prefix. Prints the cache token accounting."""
    response = chat(
        system=[{
            "type": "text",
            "text": SYSTEM_PREFIX,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": user_text}],
    )
    print(f"  Q: {user_text}")
    print(f"     created={response.cache_creation_input_tokens:5d}  "
          f"read={response.cache_read_input_tokens:5d}  "
          f"non-cached-in={response.input_tokens:4d}  "
          f"out={response.output_tokens:3d}")
    print(f"     answer: {response.text[:100]}")


if __name__ == "__main__":
    print("\n--- Turn 1: cache miss (first time the model sees this prefix) ---")
    ask("How many vacation days do full time employees get?")

    print("\n--- Turn 2: cache hit (same prefix, different question) ---")
    ask("What's the wellness benefit?")

    # The point: on turn 2, `read` should be roughly equal to the prefix
    # length and `created` should be 0. Without caching, turn 2 would
    # have repaid the full input cost.
    print(
        "\nIf turn 2 shows `read` ≈ the prefix size and `created` = 0,"
        " the cache is doing its job."
    )
