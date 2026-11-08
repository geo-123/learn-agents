"""Lesson 09d: sub agents as tools, and handoffs between agents.

Two ways to let several agents work on one request, side by side.

Part 1, a sub agent wrapped as a tool. The parent agent has one tool,
`research_agent(task)`. Its implementation is not a lookup: it runs a
whole agent loop of its own (the Lesson 01 loop) with a fresh `messages`
list, its own system prompt and its own tool, `search_handbook` (the
Lesson 03c BM25 index over the handbook). The parent never sees that
loop. The tool result is the sub agent's final answer and nothing else.

Part 2, handoffs. Three agents (triage, billing, hr) are the states of a
small state machine. Each turn the active agent either answers or hands
off to another agent, as structured output (Lesson 02). The router moves
control and passes a short handoff note, never the whole history.

Things to notice in the output:
  * Part 1: the parent's requests stay small however much the sub agent
    reads. The search results live and die in the sub agent's context.
  * Part 2: every hop starts from the request plus one note. Compare the
    size of what was sent with what carrying the whole history would cost.
  * The difference in control: a sub agent returns to its parent, which
    keeps the conversation. A handoff gives the conversation away.

`--dry-run` replaces `chat` with a small scripted fake, so the whole
wiring runs offline. Token counts are then estimated as characters / 4.

Run:
    uv run python lessons/09d_multi_agent_systems/01_subagents_and_handoffs.py
    uv run python lessons/09d_multi_agent_systems/01_subagents_and_handoffs.py --dry-run
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import LLMResponse, chat

LESSONS = Path(__file__).resolve().parents[1]
CORPUS_DIR = LESSONS.parent / "data" / "corpus"


def _load(name: str, path: Path) -> ModuleType:
    """Load a lesson file by path: its numeric prefix blocks a plain `import`."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # before exec, so @dataclass can resolve its module
    spec.loader.exec_module(module)
    return module


_bm25 = _load("_hybrid_bm25", LESSONS / "03c_agentic_rag" / "01_hybrid_bm25.py")
INDEX = _bm25.BM25Index(_bm25.build_corpus())


# --- Bookkeeping: what each model call was sent -----------------------------

CALLS: list[tuple[str, int]] = []  # (agent, input tokens)


def record(agent: str, response: LLMResponse) -> None:
    CALLS.append((agent, response.input_tokens))
    print(f"    [{agent:8s}] model call: {response.input_tokens:5d} input tokens")


def peak(agent: str) -> int:
    return max((n for a, n in CALLS if a == agent), default=0)


# --- One agent loop, used by the parent AND by the sub agent -----------------


def run_loop(agent: str, system: str, tools: list[dict[str, Any]], user: str,
             dispatch: Callable[[str, dict[str, Any]], str],
             *, max_steps: int = 6) -> tuple[str, list[dict[str, Any]]]:
    """The Lesson 01 loop. Returns the final text and the full transcript."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
    for _ in range(max_steps):
        r = chat(messages, system=system, tools=tools)
        record(agent, r)
        messages.append({"role": "assistant", "content": r.content})  # thinking included
        if r.stop_reason != "tool_use":
            return r.text, messages
        results = [
            {"type": "tool_result", "tool_use_id": tu["id"], "content": dispatch(tu["name"], tu["input"])}
            for tu in r.tool_uses
        ]
        messages.append({"role": "user", "content": results})
    return "(step limit reached)", messages


# --- Part 1: a sub agent wrapped as a tool ----------------------------------

SEARCH_TOOL = {
    "name": "search_handbook",
    "description": "Search the Acme employee handbook. Returns the best matching paragraphs with their file names.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "A few key words."}},
        "required": ["query"],
    },
}

RESEARCH_TOOL = {
    "name": "research_agent",
    "description": (
        "Hand a self contained research task to a sub agent that searches the Acme "
        "handbook as often as it needs. Returns only its final answer, with sources."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"task": {"type": "string", "description": "The full task. The sub agent sees nothing else."}},
        "required": ["task"],
    },
}

RESEARCH_SYSTEM = (
    "You are a research sub agent. Search the handbook until you can answer the task, "
    "then reply with a short answer that names the source files. No preamble."
)
PARENT_SYSTEM = (
    "You are Acme's employee assistant. For any question about company policy, delegate "
    "the research to research_agent with a self contained task, then answer the employee briefly."
)

SUB_TRANSCRIPTS: list[list[dict[str, Any]]] = []


def research_agent(task: str) -> str:
    """The tool's implementation: a whole agent loop with a fresh context."""
    print(f"    research_agent starts with a fresh context: {task!r}")

    def dispatch(name: str, args: dict[str, Any]) -> str:
        return search_handbook(args["query"]) if name == "search_handbook" else f"unknown tool {name}"

    answer, transcript = run_loop("research", RESEARCH_SYSTEM, [SEARCH_TOOL], task, dispatch)
    SUB_TRANSCRIPTS.append(transcript)
    return answer  # the transcript stays here: the parent gets only this


def search_handbook(query: str, k: int = 3) -> str:
    hits = INDEX.search(query, k=k)
    return "\n\n".join(f"[{c.doc}] {c.text}" for c, _ in hits) or "no matches"


def part1(question: str) -> None:
    print(f"\n=== Part 1: a sub agent as a tool ===\nquestion: {question}")

    def dispatch(name: str, args: dict[str, Any]) -> str:
        return research_agent(args["task"]) if name == "research_agent" else f"unknown tool {name}"

    answer, parent_transcript = run_loop("parent", PARENT_SYSTEM, [RESEARCH_TOOL], question, dispatch)
    print(f"\nanswer: {answer}")
    hidden = sum(len(json.dumps(t)) for t in SUB_TRANSCRIPTS)
    print(f"\nparent peak request : {peak('parent'):5d} tokens ({len(json.dumps(parent_transcript))} chars of history)")
    print(f"sub agent peak      : {peak('research'):5d} tokens")
    print(f"sub agent transcript the parent never saw: {hidden} chars")


# --- Part 2: handoffs as a state machine -------------------------------------

AgentName = Literal["triage", "billing", "hr"]


class Turn(BaseModel):
    """What the active agent returns on every turn: an answer or a handoff."""
    action: Literal["answer", "handoff"]
    answer: str = Field(description="The reply to the employee when action is answer, else empty.")
    handoff_to: Literal["none", "triage", "billing", "hr"] = Field(
        description="The agent to hand off to when action is handoff, else none.")
    note: str = Field(description="For a handoff: one or two sentences the next agent needs. Else empty.")


def _doc(name: str) -> str:
    return (CORPUS_DIR / name).read_text(encoding="utf-8")


_RULES = (
    " Answer only questions your documents cover. Otherwise hand off to the agent that owns the "
    "topic, with a short note: billing owns expenses and reimbursements, hr owns vacation and benefits."
)
AGENTS: dict[str, str] = {
    "triage": "You are the triage agent. You never answer: hand off every request to billing or hr." + _RULES,
    "billing": "You are the billing agent." + _RULES + "\n\n" + _doc("expense_policy.md"),
    "hr": "You are the hr agent." + _RULES + "\n\n" + _doc("vacation_policy.md") + "\n\n" + _doc("benefits.md"),
}


def route(request: str, *, start: str = "triage", max_hops: int = 4) -> str:
    """The router: about 30 lines are the whole handoff mechanism."""
    active, note = start, ""
    history_chars = 0  # what a "carry everything" design would resend
    for hop in range(1, max_hops + 1):
        content = f"Employee request:\n{request}"
        if note:
            content += f"\n\nHandoff note from the previous agent:\n{note}"
        response = chat([{"role": "user", "content": content}], system=AGENTS[active], output_schema=Turn)
        record(active, response)
        print(f"             sent {len(content)} chars; passing the whole history would add {history_chars} more")
        history_chars += len(content) + len(response.text)
        turn = response.parsed
        if turn is None:
            return f"({active} returned no turn, stop_reason={response.stop_reason})"
        if turn.action == "answer":
            return f"[{active}] {turn.answer}"
        if turn.handoff_to not in AGENTS or turn.handoff_to == active:
            return f"({active} made an invalid handoff to {turn.handoff_to!r}; stopping)"
        print(f"  hop {hop}: {active} -> {turn.handoff_to}   note: {turn.note}")
        active, note = turn.handoff_to, turn.note
    return f"(no answer after {max_hops} hops; escalate to a person)"


def part2(requests: list[str]) -> None:
    print("\n=== Part 2: handoffs between agents ===")
    for request in requests:
        print(f"\nrequest: {request}")
        print(f"final  : {route(request)}")


# --- Dry run: a scripted fake model ------------------------------------------


def _estimate_tokens(*parts: Any) -> int:
    return len(json.dumps(parts, default=str)) // 4


def _reply(text: str, tokens: int, *, tool: tuple[str, dict[str, Any]] | None = None,
           parsed: Any = None) -> LLMResponse:
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    if tool is not None:
        content.append({"type": "tool_use", "id": f"toolu_fake_{tokens}", "name": tool[0], "input": tool[1]})
    return LLMResponse(content=content, stop_reason="tool_use" if tool else "end_turn",
                       input_tokens=tokens, output_tokens=len(text) // 4, parsed=parsed)


def fake_chat(messages: list[dict[str, Any]], *, system: Any = None,
              tools: list[dict[str, Any]] | None = None, output_schema: Any = None, **_: Any) -> LLMResponse:
    """Plays every agent by simple rules, so the wiring runs without the API."""
    tokens = _estimate_tokens(system, tools, messages)
    results = [b for m in messages if isinstance(m["content"], list)
               for b in m["content"] if b.get("type") == "tool_result"]
    first = messages[0]["content"]
    tool_name = tools[0]["name"] if tools else ""
    if tool_name == "research_agent":
        if not results:
            return _reply("", tokens, tool=("research_agent", {"task": first}))
        return _reply(f"Here is what the handbook says. {results[-1]['content']}", tokens)
    if tool_name == "search_handbook":
        queries = ["hotel London cap", "receipts meals"]
        if len(results) < len(queries):
            return _reply("", tokens, tool=("search_handbook", {"query": queries[len(results)]}))
        return _reply("Hotels are capped at $400 a night in London; meals over $25 need an "
                      "itemized receipt with attendee names (expense_policy.md).", tokens)
    if output_schema is Turn:
        name = re.search(r"You are the (\w+) agent", str(system))
        agent = name.group(1) if name else "triage"
        money = re.search(r"expense|hotel|receipt|reimburs", first, re.I)
        people = re.search(r"vacation|pto|leave|benefit", first, re.I)
        if agent == "triage":
            target = "billing" if money or "billing" in first.lower() else "hr"
            asked = first.split("\n", 1)[-1][:80]
            turn = Turn(action="handoff", answer="", handoff_to=target, note=f"Employee asks: {asked}")
        elif agent == "billing" and people and not money:
            turn = Turn(action="handoff", answer="", handoff_to="hr", note="Vacation payout question, not billing.")
        elif agent == "billing":
            turn = Turn(action="answer", answer="Expenses must be filed within 30 days; late ones need VP approval.",
                        handoff_to="none", note="")
        else:
            turn = Turn(action="answer", answer="Up to 5 unused vacation days can be paid out at year end.",
                        handoff_to="none", note="")
        return _reply(turn.model_dump_json(), tokens, parsed=turn)
    return _reply("(fake: no rule for this call)", tokens)


def main() -> None:
    global chat
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--dry-run", action="store_true", help="use a scripted fake instead of the Claude API")
    args = parser.parse_args()
    if args.dry_run:
        chat = fake_chat
        print("(dry run: a scripted fake plays every agent; token counts are estimates)")

    part1("How much can I spend on a hotel in London for a conference, and do I need receipts for meals?")
    part2([
        "My hotel expense from 45 days ago was rejected. Why?",
        "Billing told me to ask here: are unused vacation days paid out at year end?",
    ])

    print("\n=== Context per agent (peak input tokens of one request) ===")
    for agent in ("parent", "research", "triage", "billing", "hr"):
        print(f"  {agent:8s} {peak(agent):5d}")


if __name__ == "__main__":
    main()
