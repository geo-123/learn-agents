"""Capstone: the course's modules wired into one end to end agent.

One question goes through:
  input guard (08, 01)  ->  the loop (00, 01)  ->  tools:
      search_docs from the Lesson 09 MCP server, over stdio
      memory, the 04b memory tool backend, behind a path and PII guard
      send_email, simulated, behind the 08, 03 tool argument guard
  every tool result scanned for injected instructions (08, 05) before it
  enters the history, then the output guard (08, 02) before the answer is shown.
The Lesson 07 tracer records one span per run, model call, tool call and guard.

Run:
    uv run python capstone/agent.py "What is the expense policy for client dinners?"
    uv run python capstone/agent.py --plant "What is the expense policy for client dinners?"
    uv run python capstone/agent.py --dry-run      # offline, no API key, scripted model
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capstone import wiring as w
from src._llm import LLMResponse, chat

SYSTEM = w.injection.DEFENDED_SYSTEM + (  # Lesson 08, 05 defense 1
    " You also have a memory directory: look in it before you start, and save"
    " anything that will help with the next question."
)


async def run(question: str, *, chat_fn: Callable[..., LLMResponse], plant: bool,
              max_steps: int = 6) -> tuple[str, Path]:
    tracer = w.tracing.Tracer(path=w.TRACES_DIR / f"capstone_{int(time.time())}.jsonl")
    with tracer.span("agent_run", question=question[:80], plant=plant) as out:
        with tracer.span("guard", name="input") as g:
            verdict = w.input_guard.input_guard(question)
            g.update(allowed=verdict.allowed, reasons=verdict.reasons)
        print(f"[guard] input allowed={verdict.allowed} {verdict.reasons or ''}")
        if verdict.allowed:
            answer = await loop(verdict.cleaned_text, tracer, chat_fn, plant, max_steps)
        else:  # refused before any model call or tool
            answer = "(input rejected: " + "; ".join(verdict.reasons) + ")"
        out["answer_preview"] = answer[:80]
    return answer, tracer.path


async def loop(question: str, tracer: Any, chat_fn: Callable[..., LLMResponse],
               plant: bool, max_steps: int) -> str:
    w.memory.ROOT.mkdir(exist_ok=True)
    # Entering the Client spawns the Lesson 09 server; leaving it stops it.
    async with w.mcp_client.Client(w.mcp_client.SERVER_PARAMS) as client:
        tools = await w.mcp_tool_specs(client) + w.LOCAL_TOOLS
        print(f"[mcp] tools: {[t['name'] for t in tools]}")
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        for step in range(1, max_steps + 1):
            with tracer.span("llm_call", step=step, n_messages=len(messages)) as out:
                response = chat_fn(messages, system=SYSTEM, tools=tools)
                out.update(stop_reason=response.stop_reason, input_tokens=response.input_tokens,
                           output_tokens=response.output_tokens,
                           tool_uses=[tu["name"] for tu in response.tool_uses])
            # Append only, content unchanged: thinking blocks go back as they came.
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use":
                return guard_output(tracer, response.text)
            results = [await run_tool(client, tracer, tu, plant) for tu in response.tool_uses]
            messages.append({"role": "user", "content": results})
    return guard_output(tracer, f"(stopped after {max_steps} steps)")


async def run_tool(client: Any, tracer: Any, tu: dict[str, Any], plant: bool) -> dict[str, Any]:
    name, args = tu["name"], tu["input"]
    print(f"  -> {name}({json.dumps(args)[:90]})")
    with tracer.span("tool_call", tool=name, args=json.dumps(args)[:120]) as out:
        if name in w.MCP_TOOLS:
            pieces, is_error = await w.call_mcp(client, name, args, plant=plant)
        elif name == "memory":
            pieces, is_error = memory_tool(tracer, tu)
        elif name == "send_email":
            pieces, is_error = send_email(tracer, args)
        else:
            pieces, is_error = [f"unknown tool {name!r}"], True
        out.update(is_error=is_error, pieces=len(pieces))
        # Every result is scanned before it enters the history, whatever its source:
        # a memory file written in an earlier run can carry an injection too.
        with tracer.span("guard", name="injection_scan", tool=name) as g:
            content, flagged = w.screen(name, pieces)
            g.update(allowed=not flagged, quarantined=flagged)
    if flagged:
        print(f"[guard] injection scan quarantined {flagged}")
    print(f"     <- {content.splitlines()[-2][:90]}")  # last line inside the delimiters
    result = {"type": "tool_result", "tool_use_id": tu["id"], "content": content}
    return {**result, "is_error": True} if is_error else result


def memory_tool(tracer: Any, tu: dict[str, Any]) -> tuple[list[str], bool]:
    with tracer.span("guard", name="memory_args", command=tu["input"].get("command")) as g:
        reason = w.memory_args_guard(tu["input"])
        g.update(allowed=reason is None, reason=reason or "")
    if reason:
        print(f"[guard] memory blocked: {reason}")
        return [f"BLOCKED by memory guard: {reason}"], True
    result = w.memory.handle(tu)  # the 04b backend runs the command
    return [result["content"]], bool(result.get("is_error"))


def send_email(tracer: Any, args: dict[str, Any]) -> tuple[list[str], bool]:
    try:
        email = w.arg_guard.SendEmailArgs.model_validate(args)
    except ValidationError as e:
        return [f"invalid args: {e.errors()}"], True
    with tracer.span("guard", name="tool_args", to=email.to) as g:
        ok, reason = w.arg_guard.guard_send_email(email)  # recipient allowlist
        g.update(allowed=ok, reason=reason)
    if not ok:
        print(f"[guard] send_email blocked: {reason}")
        return [f"BLOCKED by tool guard: {reason}"], True
    return [json.dumps(w.arg_guard.send_email(email))], False


def guard_output(tracer: Any, text: str) -> str:
    with tracer.span("guard", name="output") as g:
        result = w.output_guard.output_guard(text)  # redaction, then the topic check
        g.update(allowed=result.allowed, reasons=result.reasons)
    print(f"[guard] output allowed={result.allowed} {result.reasons or ''}")
    return result.text


def main() -> None:
    parser = argparse.ArgumentParser(description="The course modules wired into one agent.")
    parser.add_argument("question", nargs="*")
    parser.add_argument("--dry-run", action="store_true",
                        help="scripted model, no API key; implies --plant")
    parser.add_argument("--plant", action="store_true",
                        help="add the Lesson 08 poisoned chunk to search results")
    opts = parser.parse_args()
    question = " ".join(opts.question) or "What is the expense policy for client dinners?"
    chat_fn: Callable[..., LLMResponse] = chat
    if opts.dry_run:
        chat_fn = w.scripted_chat(w.DRY_RUN_SCRIPT)
        w.output_guard.chat = chat_fn  # the 02 topic check calls the model too
    print(f"USER: {question}")
    answer, trace = asyncio.run(run(question, chat_fn=chat_fn, plant=opts.plant or opts.dry_run))
    print(f"\nANSWER: {answer}")
    print(f"emails actually sent: {w.arg_guard.SENT_EMAILS or 'none'}")
    print(f"memory folder: {w.memory.ROOT}")
    print(f"trace: {trace}")
    print(f"view it: uv run python lessons/07_eval_tracing/02_trace_viewer.py {trace}")


if __name__ == "__main__":
    main()
