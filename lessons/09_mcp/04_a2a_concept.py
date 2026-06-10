"""Lesson 09 — A2A on the wire, by hand.

We build A2A messages as plain dicts so you can see the protocol's
shape end to end: AgentCard, task submission, streamed status updates,
artifact attachment. No SDK dependency — the point is what the wire
format looks like, not which library you use to send it.

The scenario: a "research" agent (this script) delegates a
fact-extraction sub-task to a notional "extractor" agent, watches
status updates stream in, and receives the final artifact. We invoke
our LLM wrapper at the moment the extractor would actually do work,
so the demo produces a real answer — but the *transport* (HTTP, SSE,
JSON-RPC) is mocked out as Python function calls.

Run:
    uv run python lessons/09_mcp/04_a2a_concept.py
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import chat


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- 1. The AgentCard --------------------------------------------------
# The "business card" an A2A agent serves at /.well-known/agent.json. A
# caller fetches this once to learn what the agent can do, how to reach
# it, and how to authenticate.

EXTRACTOR_AGENT_CARD = {
    "name": "acme-extractor",
    "version": "1.2.0",
    "description": "Extracts structured facts from free-text passages.",
    "url": "https://agents.acme.internal/extractor",
    "skills": [
        {
            "id": "extract_facts",
            "description": "Pull named entities and numeric facts from a passage.",
            "input_modes": ["text"],
            "output_modes": ["application/json"],
        }
    ],
    "authentication": {"schemes": ["bearer"]},
    "capabilities": {"streaming": True, "push_notifications": False},
}


# --- 2. The task lifecycle ---------------------------------------------
# A2A tasks have an id and a state machine:
#   submitted -> working -> (completed | failed | canceled)
# Status updates and intermediate messages stream as the task runs.


def submit_task(caller_id: str, skill: str, prompt: str) -> dict:
    """The JSON-RPC request body a caller agent sends to the receiver."""
    return {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "tasks/send",
        "params": {
            "id": f"task-{uuid.uuid4().hex[:8]}",
            "session_id": f"session-{uuid.uuid4().hex[:8]}",
            "skill": skill,
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": prompt}],
            },
            "metadata": {"caller": caller_id, "submitted_at": now()},
        },
    }


def stream_status(task_id: str, state: str, note: str = "") -> dict:
    """One event in the server-sent-events stream the receiver emits."""
    return {
        "jsonrpc": "2.0",
        "method": "tasks/status",
        "params": {
            "id": task_id,
            "status": {"state": state, "timestamp": now(), "note": note},
        },
    }


def attach_artifact(task_id: str, name: str, payload: dict) -> dict:
    """The completion event with a typed artifact attached."""
    return {
        "jsonrpc": "2.0",
        "method": "tasks/status",
        "params": {
            "id": task_id,
            "status": {"state": "completed", "timestamp": now()},
            "artifacts": [{
                "name": name,
                "mime_type": "application/json",
                "data": payload,
            }],
        },
    }


# --- 3. The "extractor" agent's actual work ----------------------------
# In a real A2A deployment this would be a different process behind a
# different URL. Here it's just a function — the protocol is what we're
# showing, not the deployment.


def extractor_handle(task_request: dict) -> Iterator[dict]:
    """Yield status events as the extractor processes the task.

    A real implementation would write these to an SSE stream.
    """
    task_id = task_request["params"]["id"]
    user_text = task_request["params"]["message"]["parts"][0]["text"]

    yield stream_status(task_id, "submitted", "queued")
    yield stream_status(task_id, "working", "calling LLM")

    response = chat(
        system="You extract structured facts. Reply with ONLY a JSON object.",
        messages=[{"role": "user", "content": user_text}],
    )
    # Crude parse — the real extractor would use a forced tool call (Lesson 02).
    try:
        facts = json.loads(response.text)
    except json.JSONDecodeError:
        facts = {"raw": response.text}

    yield attach_artifact(task_id, "facts", facts)


# --- 4. The "researcher" agent's call ----------------------------------


def researcher_delegate(prompt: str) -> dict:
    """Send a task to the extractor and consume its status stream until
    the task reaches a terminal state. Returns the final artifact."""
    request = submit_task(
        caller_id="acme-researcher",
        skill="extract_facts",
        prompt=prompt,
    )
    print(f"-> POST {EXTRACTOR_AGENT_CARD['url']}/tasks/send")
    print(f"   {json.dumps(request, indent=2)[:300]}...\n")

    final = None
    for event in extractor_handle(request):
        state = event["params"]["status"]["state"]
        print(f"<- {state:10s}  {event['params']['status'].get('note', '')}")
        if state in ("completed", "failed", "canceled"):
            final = event
            break

    assert final is not None
    artifact = final["params"].get("artifacts", [{}])[0]
    return artifact


if __name__ == "__main__":
    print("=== AgentCard served at /.well-known/agent.json ===")
    print(json.dumps(EXTRACTOR_AGENT_CARD, indent=2))

    print("\n=== Researcher delegates a task to Extractor ===\n")
    artifact = researcher_delegate(
        "From: 'Acme Q2 revenue was $14.2M, up 22% YoY, driven by EU expansion.' "
        "Extract revenue, growth_pct, and primary_driver."
    )

    print(f"\n=== Final artifact ({artifact.get('name')}) ===")
    print(json.dumps(artifact.get("data"), indent=2))
