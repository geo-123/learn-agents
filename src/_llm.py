"""Tiny vendor-isolating wrapper around the LLM call.

Every lesson imports from here, so swapping providers later means
changing one file. The wrapper deliberately stays thin: no retries,
no streaming, no caching. Those concerns get their own lessons.

Public surface:
    chat(messages, *, system=None, tools=None, tool_choice=None, model=..., max_tokens=...) -> LLMResponse
    LLMResponse  (with .text, .tool_uses, .content, .stop_reason, .usage)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# `or` handles both unset AND empty-string values from .env.
DEFAULT_MODEL = os.getenv("LEARN_AGENTS_MODEL") or "claude-haiku-4-5"
DEFAULT_MAX_TOKENS = 2048

_client: Anthropic | None = None


def client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic()
    return _client


@dataclass
class LLMResponse:
    """A normalized response from the LLM.

    Holds raw content *blocks* (text, tool_use), not a flat string —
    tool_use blocks need to round-trip back to the model on the next
    turn, which means the agent loop has to see them structurally.

    Cache fields are 0 when prompt caching is not in use; when it is,
    `cache_read_input_tokens` is the number of tokens served from cache
    and `cache_creation_input_tokens` is the number of tokens written
    to a fresh cache entry on this call.
    """
    content: list[dict[str, Any]]
    stop_reason: str
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def text(self) -> str:
        """Concatenate text blocks; empty string if there are none."""
        return "".join(
            block["text"] for block in self.content
            if block.get("type") == "text"
        )

    @property
    def tool_uses(self) -> list[dict[str, Any]]:
        """Every tool_use block in the response (empty list if none)."""
        return [b for b in self.content if b.get("type") == "tool_use"]


def chat(
    messages: list[dict[str, Any]],
    *,
    system: str | list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: dict[str, Any] | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float | None = None,
    top_p: float | None = None,
) -> LLMResponse:
    """One LLM call. Returns plain dicts — no SDK types leak out.

    Sampling knobs (set None to use provider default):
      temperature  — Anthropic range 0.0-1.0. Use 0 for extraction/structured
                     output; ~0.7 for creative work.
      top_p        — Nucleus sampling 0.0-1.0. Don't set this AND temperature
                     simultaneously; pick one. Anthropic recommends temperature
                     for most cases.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system is not None:
        kwargs["system"] = system
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p

    msg = client().messages.create(**kwargs)

    content: list[dict[str, Any]] = []
    for block in msg.content:
        if block.type == "text":
            content.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            content.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })

    return LLMResponse(
        content=content,
        stop_reason=msg.stop_reason,
        input_tokens=msg.usage.input_tokens,
        output_tokens=msg.usage.output_tokens,
        cache_creation_input_tokens=getattr(msg.usage, "cache_creation_input_tokens", 0) or 0,
        cache_read_input_tokens=getattr(msg.usage, "cache_read_input_tokens", 0) or 0,
    )
