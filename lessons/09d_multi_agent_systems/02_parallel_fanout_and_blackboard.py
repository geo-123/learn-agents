"""Lesson 09d: parallel fan out over a typed blackboard.

Lesson 09b's orchestrator handed each worker a subquestion and collected
plain strings. Here the agents coordinate through a shared, typed
board instead: a Pydantic `SharedState` with one field (or a few) per
specialist.

  * Three specialists (it, policy, expenses) run at the same time with
    `asyncio.gather`. Each reads the board, sees the request, and proposes
    writes for the fields it owns.
  * The board guards every write. An agent may set only its own fields,
    and the merged board must still validate against `SharedState`. A
    write to someone else's field, or of the wrong type, is rejected and
    logged; the rest of the board is untouched.
  * A writer agent runs last and reads the completed board.

`chat()` is synchronous (it blocks on one HTTP request), so each agent call
goes through `asyncio.to_thread`: the event loop hands the blocking call to
a worker thread and waits for all of them together. The board itself is
only touched on the event loop thread, between awaits.

The script runs the specialists once one after the other and once with
`gather`, and prints both times. Live, that means the specialists are
called twice: the comparison is the point.

`--dry-run` replaces `chat` with a scripted fake that sleeps 0.5 seconds
per call, so the timing gain shows offline too.

Run:
    uv run python lessons/09d_multi_agent_systems/02_parallel_fanout_and_blackboard.py
    uv run python lessons/09d_multi_agent_systems/02_parallel_fanout_and_blackboard.py --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import LLMResponse, chat

CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"

REQUEST = (
    "Plan the first week for a remote engineer who starts on Monday, needs a laptop, "
    "and flies to the London office in week two."
)


# --- The blackboard ----------------------------------------------------------


class SharedState(BaseModel):
    """The board. Every field after `request` belongs to exactly one agent."""
    request: str
    equipment: list[str] = Field(default_factory=list)       # it
    security_steps: list[str] = Field(default_factory=list)  # it
    remote_rules: list[str] = Field(default_factory=list)    # policy
    expense_rules: list[str] = Field(default_factory=list)   # expenses
    plan: str = ""                                            # writer


OWNERS: dict[str, tuple[str, ...]] = {
    "it": ("equipment", "security_steps"),
    "policy": ("remote_rules",),
    "expenses": ("expense_rules",),
    "writer": ("plan",),
}


class Blackboard:
    """Holds the state and enforces who may write what."""

    def __init__(self, request: str) -> None:
        self._state = SharedState(request=request)
        self._lock = asyncio.Lock()
        self.log: list[str] = []

    def read(self) -> SharedState:
        return self._state.model_copy(deep=True)  # agents get a copy, never the live object

    async def write(self, agent: str, updates: dict[str, Any]) -> None:
        # Writes run on the event loop thread, so two of them cannot interleave
        # as long as nothing inside awaits. The lock keeps that true the day a
        # write awaits something (a database, a file).
        async with self._lock:
            accepted: dict[str, Any] = {}
            for name, value in updates.items():
                if name in OWNERS.get(agent, ()):
                    accepted[name] = value
                else:
                    self.log.append(f"REJECTED {agent} -> {name}: not its field")
            if not accepted:
                return
            try:
                self._state = SharedState.model_validate({**self._state.model_dump(), **accepted})
            except ValidationError as e:
                problem = e.errors()[0]
                self.log.append(f"REJECTED {agent} -> {list(accepted)}: {problem['msg']}")
                return
            self.log.append(f"ok       {agent} -> {', '.join(accepted)}")


# --- The agents --------------------------------------------------------------


class FieldWrite(BaseModel):
    field: str = Field(description="The board field to set.")
    items: list[str] = Field(description="Short, concrete items for that field.")


class Proposal(BaseModel):
    """What a specialist returns: the writes it wants to make."""
    writes: list[FieldWrite]


def _doc(name: str) -> str:
    return (CORPUS_DIR / name).read_text(encoding="utf-8")


SPECIALISTS: dict[str, str] = {
    "it": "You are the it specialist.\n\n" + _doc("security.md") + "\n\n" + _doc("onboarding.md"),
    "policy": "You are the policy specialist.\n\n" + _doc("remote_work.md"),
    "expenses": "You are the expenses specialist.\n\n" + _doc("expense_policy.md"),
}
WRITER_SYSTEM = (
    "You are the writer. Turn the completed board into a short first week plan, day by day, "
    "in plain prose. Use only what is on the board. Say which fields are empty, if any."
)


async def specialist(name: str, board: Blackboard) -> None:
    snapshot = board.read()
    fields = ", ".join(OWNERS[name])
    user = (
        f"The board:\n{snapshot.model_dump_json(indent=2)}\n\n"
        f"You own these fields: {fields}. Propose writes for them only, "
        "three items at most per field, grounded in your documents."
    )
    # chat() blocks, so it runs in a worker thread while the loop waits for the others.
    response = await asyncio.to_thread(
        chat, [{"role": "user", "content": user}], system=SPECIALISTS[name], output_schema=Proposal,
    )
    proposal = response.parsed
    if proposal is None:
        raise RuntimeError(f"{name} returned no proposal (stop_reason={response.stop_reason})")
    await board.write(name, {w.field: w.items for w in proposal.writes})


async def writer(board: Blackboard) -> None:
    snapshot = board.read()
    response = await asyncio.to_thread(
        chat, [{"role": "user", "content": f"The board:\n{snapshot.model_dump_json(indent=2)}"}],
        system=WRITER_SYSTEM,
    )
    await board.write("writer", {"plan": response.text.strip()})


async def run(parallel: bool) -> tuple[Blackboard, float]:
    board = Blackboard(REQUEST)
    start = time.perf_counter()
    if parallel:
        # return_exceptions: one failing specialist leaves its fields empty
        # instead of cancelling the others.
        results = await asyncio.gather(*(specialist(n, board) for n in SPECIALISTS), return_exceptions=True)
        for name, result in zip(SPECIALISTS, results):
            if isinstance(result, BaseException):
                board.log.append(f"FAILED   {name}: {result}")
    else:
        for name in SPECIALISTS:
            await specialist(name, board)
    await writer(board)  # the final agent always waits for the full board
    return board, time.perf_counter() - start


# --- Dry run: a scripted fake model ------------------------------------------

FAKE_LATENCY = 0.5


def fake_chat(messages: list[dict[str, Any]], *, system: Any = None,
              output_schema: Any = None, **_: Any) -> LLMResponse:
    """Plays each agent by rules. Sleeps like a network call, in its thread."""
    time.sleep(FAKE_LATENCY)
    found = re.search(r"You are the (\w+) specialist", str(system))
    agent = found.group(1) if found else "writer"
    canned: dict[str, list[FieldWrite]] = {
        "it": [FieldWrite(field="equipment", items=["Laptop shipped before Monday"]),
               FieldWrite(field="security_steps", items=["Enroll in MFA on day one", "Security training in week one"])],
        "policy": [FieldWrite(field="remote_rules", items=["Core hours 10:00 to 15:00 local time"]),
                   # A stray write to another agent's field: the board must reject it.
                   FieldWrite(field="expense_rules", items=["Book business class"])],
        "expenses": [FieldWrite(field="expense_rules", items=["Economy for flights under 6 hours",
                                                              "London hotels up to $400 a night"])],
    }
    if output_schema is Proposal:
        proposal = Proposal(writes=canned.get(agent, []))
        text = proposal.model_dump_json()
        parsed: Any = proposal
    else:
        board = json.loads(messages[0]["content"].split("\n", 1)[1])
        text = (f"Day 1: {board['equipment'][0]}; {board['security_steps'][0]}. "
                f"Every day: {board['remote_rules'][0]}. Week 2 travel: {'; '.join(board['expense_rules'])}.")
        parsed = None
    return LLMResponse(content=[{"type": "text", "text": text}], stop_reason="end_turn",
                       input_tokens=len(json.dumps(messages)) // 4, output_tokens=len(text) // 4, parsed=parsed)


async def main_async() -> None:
    sequential_board, sequential_s = await run(parallel=False)
    print(f"sequential: {sequential_s:5.2f} s")
    board, parallel_s = await run(parallel=True)
    print(f"parallel  : {parallel_s:5.2f} s   ({sequential_s / parallel_s:.1f}x faster)")

    print("\n=== Write log of the parallel run ===")
    for line in board.log:
        print(f"  {line}")

    print("\n=== The guard on two more writes ===")
    await board.write("policy", {"plan": "Skip the security training."})  # not its field
    await board.write("it", {"equipment": "one laptop"})                 # wrong type
    for line in board.log[-2:]:
        print(f"  {line}")

    print("\n=== The completed board ===")
    print(board.read().model_dump_json(indent=2))
    assert sequential_board.read().plan, "the sequential run must also produce a plan"


def main() -> None:
    global chat
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--dry-run", action="store_true", help="use a scripted fake instead of the Claude API")
    args = parser.parse_args()
    if args.dry_run:
        chat = fake_chat
        print(f"(dry run: a scripted fake plays every agent, {FAKE_LATENCY} s per call)")
    print(f"request: {REQUEST}\n")
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
