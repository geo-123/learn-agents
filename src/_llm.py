"""Tiny vendor isolating wrapper around the LLM call.

Every lesson imports from here, so swapping providers later means
changing one file. The wrapper deliberately stays thin: no retries,
no streaming, no tool loop. Those concerns get their own lessons.

Public surface:
    chat(messages, *, system=None, tools=None, tool_choice=None, model=..., max_tokens=...,
         effort=None, output_schema=None, thinking=None, cache_control=None) -> LLMResponse
    LLMResponse  (with .text, .tool_uses, .thinking, .parsed, .content, .stop_reason, .usage)

Three things changed in 2026 that this file now handles for every lesson:

  * Current models think by default. A response can start with `thinking`
    blocks, so we read blocks by type, never by position, and we keep every
    block (thinking included, even with empty text) so the agent loop can
    send it back unchanged on the next turn.
  * `effort` (low, medium, high, xhigh, max) is the knob for how hard the
    model works. It replaces thinking budgets and, for most uses, sampling
    settings: `temperature`, `top_p` and `top_k` are gone from the API
    (version 2026-10-21) and from the SDK.
  * Structured output is native. Pass `output_schema=SomePydanticModel` and
    the API constrains the answer to that schema; `.parsed` holds the
    validated object. Forcing a tool call with `tool_choice` to get JSON is
    the older trick, and the newest models reject forced tool choice.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

# `or` handles both unset AND empty string values from .env.
DEFAULT_MODEL = os.getenv("LEARN_AGENTS_MODEL") or "claude-haiku-5-5"
# Thinking counts toward max_tokens, so the cap leaves room for it.
DEFAULT_MAX_TOKENS = 8000
# Unset means the model's own default (medium on Haiku 5.5).
DEFAULT_EFFORT = os.getenv("LEARN_AGENTS_EFFORT") or None

_client: Anthropic | None = None


def client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic()
    return _client


@dataclass
class LLMResponse:
    """A normalized response from the LLM.

    Holds raw content *blocks* (thinking, text, tool_use, ...), not a flat
    string: tool_use and thinking blocks need to round trip back to the
    model on the next turn, which means the agent loop has to see them
    structurally.

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
    parsed: Any = None

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

    @property
    def thinking(self) -> str:
        """Thinking summaries, if requested with display "summarized" (empty otherwise)."""
        return "".join(
            block.get("thinking", "") for block in self.content
            if block.get("type") == "thinking"
        )

    @property
    def refused(self) -> bool:
        """True when a safety classifier declined the request (stop_reason "refusal")."""
        return self.stop_reason == "refusal"


def strip_thinking(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A copy of `messages` with thinking blocks removed from assistant turns.

    Current models bind each thinking block to the exact history before it
    (the system prompt, the tools, every earlier message). If your code
    trims or rewrites the start of the history (a sliding window, a client
    side summary), the thinking blocks of the turns you keep no longer match,
    and the API rejects them (400 on accounts created since 31 August 2026).
    Stripping them from the kept turns is the documented way out. Never do
    this in a plain append only loop: there the blocks must go back unchanged.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if message.get("role") == "assistant" and isinstance(content, list):
            kept = [b for b in content if b.get("type") not in ("thinking", "redacted_thinking")]
            message = {**message, "content": kept or [{"type": "text", "text": ""}]}
        out.append(message)
    return out


def _block_to_dict(block: Any) -> dict[str, Any]:
    """An SDK content block as a plain dict that can be sent back unchanged."""
    data = block.model_dump(mode="json", exclude_none=True)
    data.pop("parsed_output", None)  # a client side convenience field, not part of the API
    return data


def chat(
    messages: list[dict[str, Any]],
    *,
    system: str | list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: dict[str, Any] | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    effort: str | None = DEFAULT_EFFORT,
    output_schema: type[BaseModel] | dict[str, Any] | None = None,
    thinking: dict[str, Any] | None = None,
    cache_control: dict[str, Any] | None = None,
) -> LLMResponse:
    """One LLM call. Returns plain dicts: no SDK types leak out.

    effort: "low" | "medium" | "high" | "xhigh" | "max". How much the model
        thinks and how thorough it is. Lower is faster and cheaper.
    output_schema: a Pydantic model class (validated object in .parsed) or a
        JSON schema dict (parsed JSON in .parsed). The answer is constrained
        to the schema by the API itself.
    thinking: e.g. {"type": "adaptive", "display": "summarized"} to receive
        readable thinking summaries. Leave None for the model's default.
    tool_choice: {"type": "auto"} (default) or {"type": "none"}. Forcing a
        call ({"type": "any"} or {"type": "tool", ...}) returns a 400 on
        Claude Opus 5.5, Sonnet 5.5 and Fable 5.1; steer with the prompt and
        `strict: true` tool schemas instead.
    cache_control: {"type": "ephemeral"} caches the request prefix
        automatically (Lesson 02b).
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
    if thinking is not None:
        kwargs["thinking"] = thinking
    output_config: dict[str, Any] = {}
    if effort is not None:
        output_config["effort"] = effort

    parsed: Any = None
    if isinstance(output_schema, type) and issubclass(output_schema, BaseModel):
        if output_config:
            kwargs["output_config"] = output_config
        msg = client().messages.parse(output_format=output_schema, **kwargs)
        parsed = msg.parsed_output
    else:
        if output_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": output_schema}
        if output_config:
            kwargs["output_config"] = output_config
        if cache_control is not None:
            kwargs["cache_control"] = cache_control
        msg = client().messages.create(**kwargs)

    content = [_block_to_dict(block) for block in msg.content]
    if isinstance(output_schema, dict) and msg.stop_reason == "end_turn":
        text = "".join(b["text"] for b in content if b.get("type") == "text")
        parsed = json.loads(text) if text else None

    usage = msg.usage
    return LLMResponse(
        content=content,
        stop_reason=msg.stop_reason or "",
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        parsed=parsed,
    )
