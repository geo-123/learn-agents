"""Lesson 09d: tracing across agents, circuit breakers, and routing by agent cards.

Three things a multi agent system needs once it runs unattended.

1. Distributed tracing. One `trace_id` per user request; a span per agent
   and per model call; every span names its parent. Lesson 07's Tracer
   finds the parent through a contextvar, which works inside one call
   stack. A handoff leaves that stack (in production the next agent may be
   another process), so the trace context travels in the handoff envelope,
   the way a `traceparent` header travels with an HTTP request. The tracer
   here extends the Lesson 07 Tracer with an explicit parent, writes the
   same JSONL fields to `traces/` and prints the tree with the Lesson 07
   viewer.

2. Circuit breakers, shared by every agent of one request:
     * `max_hops`: a handoff chain longer than this stops;
     * ping pong: A to B to A to B stops at once, long before `max_hops`;
     * a token budget across all agents: once the request has spent it,
       the next model call never happens.
   A tripped breaker fails closed: the request ends with a message that a
   person takes over, and the span that tripped records why.

3. Routing by agent cards, A2A style (Lesson 09 showed the full A2A 1.0
   card). Each agent publishes a small card: name, description, skills.
   The router ranks the cards against the request with the course's BM25
   (Lesson 03c). A clear winner is taken; a weak or close result goes to
   a model choice with structured output, which sees only the cards.
   `--model-route` sends every request to the model choice.

The demo runs three requests: one handoff that ends in an answer, one
ping pong between billing and it, and one under a small token budget.

`--dry-run` replaces `chat` with a scripted fake. The ping pong is scripted
there; live, it happens only if the agents really bounce the request (the
home office stipend sits between two documents, so they often do). The
token budget trips either way.

Run:
    uv run python lessons/09d_multi_agent_systems/04_multi_agent_tracing_and_budgets.py
    uv run python lessons/09d_multi_agent_systems/04_multi_agent_tracing_and_budgets.py --dry-run
    uv run python lessons/09d_multi_agent_systems/04_multi_agent_tracing_and_budgets.py --model-route
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Literal

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


_trace07 = _load("_trace_jsonl", LESSONS / "07_eval_tracing" / "01_trace_jsonl.py")
_viewer = _load("_trace_viewer", LESSONS / "07_eval_tracing" / "02_trace_viewer.py")
_bm25 = _load("_hybrid_bm25", LESSONS / "03c_agentic_rag" / "01_hybrid_bm25.py")


# --- 1. Distributed tracing --------------------------------------------------------


@dataclass(frozen=True)
class SpanContext:
    """What must travel with a handoff so the next agent can join the trace."""
    trace_id: str
    span_id: str


class DistributedTracer(_trace07.Tracer):
    """The Lesson 07 Tracer, with the parent passed in instead of found."""

    @contextlib.contextmanager
    def span_in(self, kind: str, parent: SpanContext | None, **attrs: Any) -> Iterator[tuple[SpanContext, dict[str, Any]]]:
        ctx = SpanContext(trace_id=parent.trace_id if parent else uuid.uuid4().hex[:12], span_id=uuid.uuid4().hex[:8])
        start = time.monotonic()
        status = "ok"
        outputs: dict[str, Any] = {}
        try:
            yield ctx, outputs
        except Exception as e:  # noqa: BLE001
            status = f"error: {type(e).__name__}: {e}"
            raise
        finally:
            self._write({  # the Lesson 07 fields, so its viewer reads them, plus trace_id
                "trace_id": ctx.trace_id,
                "run_id": ctx.trace_id,
                "span_id": ctx.span_id,
                "parent_id": parent.span_id if parent else None,
                "kind": kind,
                "ms": round((time.monotonic() - start) * 1000, 1),
                "status": status,
                "attrs": attrs,
                "outputs": outputs,
            })


class Handoff(BaseModel):
    """The envelope one agent sends the next: the task note and the trace context."""
    to: str
    note: str
    trace_id: str
    parent_span_id: str


# --- 2. Circuit breakers -------------------------------------------------------------


class BreakerTripped(Exception):
    pass


@dataclass
class Budget:
    """One per user request, shared by every agent that works on it."""
    max_hops: int = 4
    max_tokens: int = 20_000
    hops: int = 0
    tokens: int = 0
    path: list[str] = field(default_factory=list)

    def charge(self, agent: str, response: LLMResponse) -> None:
        # The call is already paid for; the breaker can only stop the next one.
        self.tokens += response.input_tokens + response.output_tokens
        if self.tokens > self.max_tokens:
            raise BreakerTripped(f"token budget: {self.tokens} > {self.max_tokens} after {agent}")

    def hop(self, to: str) -> None:
        self.hops += 1
        self.path.append(to)
        p = self.path
        if len(p) >= 4 and p[-1] == p[-3] and p[-2] == p[-4]:
            raise BreakerTripped(f"ping pong between {p[-2]} and {p[-1]}: {' -> '.join(p)}")
        if self.hops > self.max_hops:
            raise BreakerTripped(f"max_hops: {self.hops} > {self.max_hops}: {' -> '.join(p)}")


# --- 3. Agent cards and the router ----------------------------------------------------

CARDS: list[dict[str, Any]] = [
    {"name": "billing", "description": "Answers questions on expenses and reimbursement.",
     "skills": [{"id": "expenses", "name": "Expense policy",
                 "description": "Travel, hotels, meals, receipts, approval and reimbursement deadlines.",
                 "tags": ["expense", "reimbursement", "receipt", "hotel", "travel", "billing"]}]},
    {"name": "hr", "description": "Answers questions on time off and benefits.",
     "skills": [{"id": "time_off", "name": "Vacation and leave",
                 "description": "Vacation days, holidays, payout of unused days, parental leave.",
                 "tags": ["vacation", "pto", "leave", "holiday"]},
                {"id": "benefits", "name": "Benefits",
                 "description": "Health insurance, retirement plan, wellness.",
                 "tags": ["benefits", "insurance", "401k"]}]},
    {"name": "it", "description": "Answers questions on laptops, accounts and security.",
     "skills": [{"id": "security", "name": "Security and accounts",
                 "description": "Laptops, equipment, MFA, passwords, phishing, onboarding accounts.",
                 "tags": ["laptop", "equipment", "mfa", "password", "security"]}]},
]
AGENT_NAMES = [card["name"] for card in CARDS]


def card_text(card: dict[str, Any]) -> str:
    skills = " ".join(f"{s['name']} {s['description']} {' '.join(s['tags'])}" for s in card["skills"])
    return f"{card['name']} {card['description']} {skills}"


CARD_INDEX = _bm25.BM25Index([_bm25.Chunk(c["name"], i, card_text(c)) for i, c in enumerate(CARDS)])


class RouteChoice(BaseModel):
    agent: Literal["billing", "hr", "it"]
    reason: str = Field(description="One short sentence.")


def route(request: str, *, use_model: bool) -> tuple[str, dict[str, Any]]:
    """BM25 over the cards first; the model only when BM25 is unsure (or asked)."""
    ranked = CARD_INDEX.search(request, k=len(CARDS))
    scores = {c.doc: round(s, 2) for c, s in ranked}
    clear = len(ranked) == 1 or (len(ranked) > 1 and ranked[0][1] >= 1.5 * ranked[1][1])
    if ranked and clear and not use_model:
        return ranked[0][0].doc, {"by": "bm25", "scores": scores}
    response = chat([{"role": "user", "content": f"Agent cards:\n{json.dumps(CARDS)}\n\nRequest:\n{request}"}],
                    system="Pick the one agent whose card best matches the request.", output_schema=RouteChoice)
    choice = response.parsed
    if choice is None:
        return ranked[0][0].doc if ranked else "hr", {"by": "fallback", "scores": scores}
    return choice.agent, {"by": "model", "scores": scores, "reason": choice.reason}


# --- The agents ------------------------------------------------------------------------


class Turn(BaseModel):
    action: Literal["answer", "handoff"]
    answer: str = Field(description="The reply when action is answer, else empty.")
    handoff_to: Literal["none", "billing", "hr", "it"]
    note: str = Field(description="For a handoff: what the next agent needs, in one or two sentences. Else empty.")


def _doc(*names: str) -> str:
    return "\n\n".join((CORPUS_DIR / n).read_text(encoding="utf-8") for n in names)


_CARD_LINES = "\n".join(f"- {c['name']}: {c['description']}" for c in CARDS)
_RULES = f"Answer only what your documents cover. Otherwise hand off, using these cards:\n{_CARD_LINES}\n\n"
SYSTEMS = {
    "billing": "You are the billing agent. " + _RULES + _doc("expense_policy.md"),
    "hr": "You are the hr agent. " + _RULES + _doc("vacation_policy.md", "benefits.md"),
    "it": "You are the it agent. " + _RULES + _doc("security.md", "onboarding.md"),
}


def handle(request: str, tracer: DistributedTracer, budget: Budget, *, use_model_route: bool = False) -> str:
    """One user request: one trace, one budget, any number of agents."""
    with tracer.span_in("request", None, request=request[:60]) as (root, out):
        with tracer.span_in("router", root) as (_, router_out):
            agent, how = route(request, use_model=use_model_route)
            router_out.update(agent=agent, **how)
        budget.path.append(agent)
        parent, note = root, ""
        try:
            while True:
                with tracer.span_in("agent", parent, agent=agent, hop=budget.hops) as (agent_ctx, agent_out):
                    content = f"Request:\n{request}" + (f"\n\nHandoff note:\n{note}" if note else "")
                    with tracer.span_in("llm_call", agent_ctx, agent=agent) as (_, llm_out):
                        response = chat([{"role": "user", "content": content}], system=SYSTEMS[agent], output_schema=Turn)
                        llm_out.update(input_tokens=response.input_tokens, output_tokens=response.output_tokens,
                                       stop_reason=response.stop_reason)
                        budget.charge(agent, response)
                    turn = response.parsed
                    if turn is None or turn.action == "answer" or turn.handoff_to not in SYSTEMS:
                        answer = turn.answer if turn else f"(no turn, stop_reason={response.stop_reason})"
                        agent_out["answer"] = answer[:60]
                        out.update(answer=answer[:60], tokens=budget.tokens, hops=budget.hops)
                        return f"[{agent}] {answer}"
                    agent_out["handoff_to"] = turn.handoff_to
                    budget.hop(turn.handoff_to)
                    envelope = Handoff(to=turn.handoff_to, note=turn.note,
                                       trace_id=agent_ctx.trace_id, parent_span_id=agent_ctx.span_id)
                # The next agent rebuilds its parent from the envelope alone,
                # as an agent in another process would.
                parent = SpanContext(envelope.trace_id, envelope.parent_span_id)
                agent, note = envelope.to, envelope.note
        except BreakerTripped as e:
            out.update(breaker=str(e), tokens=budget.tokens, hops=budget.hops)
            return f"(stopped by a circuit breaker: {e}. A person takes over.)"


# --- Dry run: a scripted fake model ------------------------------------------------------


def fake_chat(messages: list[dict[str, Any]], *, system: Any = None,
              output_schema: Any = None, **_: Any) -> LLMResponse:
    """Plays the router's model choice and the three agents by keyword rules."""
    user = messages[0]["content"]
    tokens = len(str(system)) // 4 + len(user) // 4
    parsed: BaseModel
    if output_schema is RouteChoice:
        request = user.rsplit("Request:\n", 1)[-1].lower()
        pick = "billing" if re.search(r"reimburs|expense", request) else "it" if "laptop" in request else "hr"
        parsed = RouteChoice(agent=pick, reason="keyword match in the fake")
    else:
        found = re.search(r"You are the (\w+) agent", str(system))
        agent = found.group(1) if found else "hr"
        text = user.lower()
        if agent == "billing" and "stipend" in text:
            parsed = Turn(action="handoff", answer="", handoff_to="it", note="Home office equipment: it owns equipment.")
        elif agent == "it" and "stipend" in text:
            parsed = Turn(action="handoff", answer="", handoff_to="billing", note="A reimbursement: billing owns those.")
        elif agent == "billing" and "vacation" in text and "handoff note" not in text:
            parsed = Turn(action="handoff", answer="", handoff_to="hr",
                          note="Told them: file the hotel expense within 30 days. Still open: vacation balance.")
        else:
            parsed = Turn(action="answer", answer="File the hotel expense within 30 days. You accrue 18 vacation "
                          "days a year, 1.5 a month; PeopleHQ shows what is left.", handoff_to="none", note="")
    body = parsed.model_dump_json()
    return LLMResponse(content=[{"type": "text", "text": body}], stop_reason="end_turn",
                       input_tokens=tokens, output_tokens=len(body) // 4, parsed=parsed)


def main() -> None:
    global chat
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--dry-run", action="store_true", help="use a scripted fake instead of the Claude API")
    parser.add_argument("--model-route", action="store_true", help="route every request with a model choice")
    args = parser.parse_args()
    if args.dry_run:
        chat = fake_chat
        print("(dry run: a scripted fake plays the router and the agents; token counts are estimates)")

    print("=== Routing by agent cards ===")
    for request in ("I lost my laptop and need to reset MFA", "How many vacation days do I get?"):
        agent, how = route(request, use_model=args.model_route)
        print(f"  {request!r:45s} -> {agent:8s} by {how['by']}, scores {how['scores']}")

    path = _trace07.TRACES_DIR / f"multi_agent_{int(time.time())}.jsonl"
    tracer = DistributedTracer(path=path)
    two_topics = ("Back from the London trip: how long do I have to file the hotel expense, "
                  "and how many vacation days do I have left?")
    demos = [
        ("a handoff chain", two_topics, Budget()),
        ("a ping pong", "Who reimburses the home office stipend for my new monitor?", Budget()),
        ("a small token budget", two_topics, Budget(max_tokens=300)),
    ]
    for title, request, budget in demos:
        print(f"\n=== {title} ===\nrequest: {request}")
        print(f"result : {handle(request, tracer, budget, use_model_route=args.model_route)}")
        print(f"budget : hops {budget.hops}/{budget.max_hops}, tokens {budget.tokens}/{budget.max_tokens}, "
              f"path {' -> '.join(budget.path)}")
        spans = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        trace_id = spans[-1]["trace_id"]  # the request span closes last
        print(f"trace {trace_id}:")
        _viewer.render([s for s in spans if s["trace_id"] == trace_id])

    print(f"\nall spans: {path}")
    first_call = next(line for line in path.read_text(encoding="utf-8").splitlines() if '"llm_call"' in line)
    print(f"one llm_call span as written:\n{first_call}")


if __name__ == "__main__":
    main()
