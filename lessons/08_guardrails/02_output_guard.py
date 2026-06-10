"""Lesson 08 — output guard.

Runs on the LLM's response BEFORE returning to the user. Two layers:
  1. Regex redaction (emails, phone-like patterns).
  2. LLM-as-guard: is the response on-topic for an Acme handbook agent?
     If off-topic, replace with a polite refusal.

Run:
    uv run python lessons/08_guardrails/02_output_guard.py
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import chat


EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")


@dataclass
class GuardResult:
    allowed: bool
    text: str
    reasons: list[str]


def redact_pii(text: str) -> tuple[str, list[str]]:
    reasons: list[str] = []
    n_email = len(EMAIL_RE.findall(text))
    if n_email:
        reasons.append(f"redacted {n_email} email(s)")
        text = EMAIL_RE.sub("[email redacted]", text)
    n_phone = len(PHONE_RE.findall(text))
    if n_phone:
        reasons.append(f"redacted {n_phone} phone-like number(s)")
        text = PHONE_RE.sub("[phone redacted]", text)
    return text, reasons


# --- LLM-as-guard: rubric-based on-topic check ----------------------------


class OnTopicVerdict(BaseModel):
    on_topic: bool = Field(
        description=(
            "True if the answer is plausibly about Acme company policies "
            "(vacation, benefits, security, incidents, onboarding, expenses, "
            "remote work, code review). False otherwise."
        ),
    )
    reason: str = Field(description="One short sentence.")


GUARD_TOOL = {
    "name": "submit_verdict",
    "description": "Submit the on-topic verdict.",
    "input_schema": OnTopicVerdict.model_json_schema(),
}

GUARD_SYSTEM = (
    "You are an output guardrail. You are given an AI assistant's response. "
    "Decide if the response is plausibly about Acme's company policies — "
    "things you'd expect to find in an employee handbook. Be strict: small "
    "talk, jokes, or general knowledge are NOT on-topic."
)


def is_on_topic(text: str) -> OnTopicVerdict:
    response = chat(
        messages=[{"role": "user", "content": f"Response to check:\n\n{text}"}],
        system=GUARD_SYSTEM,
        tools=[GUARD_TOOL],
        tool_choice={"type": "tool", "name": "submit_verdict"},
    )
    return OnTopicVerdict.model_validate(response.tool_uses[0]["input"])


# --- Composite output guard ----------------------------------------------


def output_guard(text: str) -> GuardResult:
    text, redact_reasons = redact_pii(text)
    verdict = is_on_topic(text)
    if not verdict.on_topic:
        return GuardResult(
            allowed=False,
            text=(
                "I can only answer questions about Acme company policies. "
                f"(off-topic — {verdict.reason})"
            ),
            reasons=[*redact_reasons, f"off-topic: {verdict.reason}"],
        )
    return GuardResult(allowed=True, text=text, reasons=redact_reasons)


# --- Demo: chat with output guardrails -----------------------------------


def safe_chat(user_text: str) -> str:
    print(f"\nuser   : {user_text!r}")
    raw = chat(
        messages=[{"role": "user", "content": user_text}],
        system="You are a helpful assistant. Answer concisely.",
    ).text
    print(f"raw    : {raw[:120]!r}")
    g = output_guard(raw)
    print(f"guard  : allowed={g.allowed}  reasons={g.reasons}")
    return g.text


if __name__ == "__main__":
    samples = [
        "What is Acme's vacation policy in one sentence?",       # on-topic + PII-free
        "What is the capital of France?",                          # off-topic
        "Tell me a joke.",                                         # off-topic
        "How can I reach IT? Their number is +1-555-867-5309.",   # phone in response, on-topic
    ]
    for s in samples:
        print(f"final  : {safe_chat(s)}")
