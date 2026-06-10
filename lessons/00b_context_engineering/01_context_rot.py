"""Lesson 00b — Context rot, hands-on.

A small needle-in-a-haystack demo. We insert one fact ("PURPLE-WHALE-42")
into a body of filler text and ask the model to retrieve it. Two
variations:

  1. Position — needle at the start vs. middle vs. end of the haystack.
  2. Distractor — adding a plausible-looking but wrong "secret code"
     elsewhere in the haystack.

On the small corpus shipped here (~3K tokens), modern frontier models
usually pass all variants. The point is to see the *pattern* and the
plumbing, then crank up the haystack size (or switch to a less capable
model) to reproduce Chroma's published context-rot findings yourself.

Run:
    uv run python lessons/00b_context_engineering/01_context_rot.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import chat

CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"

NEEDLE = "ANNOUNCEMENT: Acme's 2026 annual company offsite will be held in Lisbon from June 14 to June 16."
DISTRACTOR = "ANNOUNCEMENT: Acme's 2025 annual company offsite was held in Berlin from August 22 to August 24."

QUESTION = (
    "Where and when is Acme's 2026 annual company offsite? "
    "Answer in one short sentence."
)


def build_haystack(position: str, *, with_distractor: bool = False) -> str:
    """Concatenate the demo corpus, then splice the needle at the requested
    position. Optionally insert a similarly-shaped distractor on the
    opposite side of the document.
    """
    docs: list[str] = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        docs.append(f"## {path.name}\n\n{path.read_text(encoding='utf-8')}")
    haystack = "\n\n---\n\n".join(docs)
    mid = len(haystack) // 2

    if position == "start":
        body = NEEDLE + "\n\n---\n\n" + haystack
    elif position == "end":
        body = haystack + "\n\n---\n\n" + NEEDLE
    elif position == "middle":
        body = haystack[:mid] + "\n\n---\n\n" + NEEDLE + "\n\n---\n\n" + haystack[mid:]
    else:
        raise ValueError(f"unknown position: {position}")

    if with_distractor:
        # Drop the distractor in the *opposite* region from the needle so
        # they compete for attention rather than sit side-by-side.
        if position == "start":
            body = body + "\n\n---\n\n" + DISTRACTOR
        elif position == "end":
            body = DISTRACTOR + "\n\n---\n\n" + body
        else:
            body = body + "\n\n---\n\n" + DISTRACTOR
    return body


def ask(haystack: str) -> tuple[str, int, int]:
    """One LLM call. Returns (answer_text, input_tokens, output_tokens)."""
    messages = [{
        "role": "user",
        "content": (
            f"Below is the Acme internal knowledge base.\n\n"
            f"{haystack}\n\n---\n\n"
            f"{QUESTION}"
        ),
    }]
    response = chat(messages)
    return response.text.strip(), response.input_tokens, response.output_tokens


def evaluate(answer: str) -> str:
    """Crude correctness check: must mention Lisbon AND June (the needle's
    city and month). Distractor would surface Berlin or August."""
    a = answer.lower()
    if "lisbon" in a and "june" in a:
        return "CORRECT"
    if "berlin" in a or "august" in a:
        return "WRONG (took the distractor)"
    return "WRONG"


def run_trial(label: str, position: str, *, with_distractor: bool = False) -> None:
    haystack = build_haystack(position, with_distractor=with_distractor)
    answer, in_tok, out_tok = ask(haystack)
    verdict = evaluate(answer)
    print(f"  {label:35s} {verdict:30s} in={in_tok:5d} out={out_tok:3d}")
    print(f"    answer: {answer[:80]}")


if __name__ == "__main__":
    print("\n--- Trial 1: position effect (no distractor) ---")
    run_trial("needle at start", "start")
    run_trial("needle at middle", "middle")
    run_trial("needle at end", "end")

    print("\n--- Trial 2: distractor effect (needle in middle) ---")
    run_trial("middle, no distractor", "middle")
    run_trial("middle, with distractor", "middle", with_distractor=True)

    print(
        "\nNote: on this ~3K-token haystack modern models usually pass all "
        "variants. To reproduce the published context-rot findings, pad the "
        "haystack to 50K+ tokens (e.g., concatenate the corpus 20x) or "
        "switch to a less capable model in src/_llm.py."
    )
