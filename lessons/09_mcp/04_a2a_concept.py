"""Lesson 09: A2A on the wire, by hand.

We build A2A messages as plain dicts so you can see the protocol's
shape end to end: AgentCard, a streamed message send, task status
updates, an artifact. No SDK dependency: the point is what the wire
format looks like, not which library you use to send it.

The shapes follow A2A 1.0 with the JSON-RPC binding: PascalCase method
names (`SendStreamingMessage`), camelCase fields, enum values such as
`ROLE_USER` and `TASK_STATE_WORKING`, and parts that hold exactly one of
`text`, `raw`, `url` or `data`.

The scenario: a "research" agent (this script) delegates a
fact extraction subtask to a notional "extractor" agent, watches
status updates stream in, and receives the final artifact. We invoke
our LLM wrapper at the moment the extractor would actually do work,
so the demo produces a real answer: but the *transport* (HTTP, server
sent events, JSON-RPC) is mocked out as Python function calls.

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
# The "business card" an A2A agent serves at /.well-known/agent-card.json.
# A caller fetches this once to learn what the agent can do, how to reach
# it (one entry per protocol binding), and how to authenticate.

EXTRACTOR_URL = "https://agents.acme.internal/extractor/a2a/v1"

EXTRACTOR_AGENT_CARD = {
    "name": "acme-extractor",
    "description": "Extracts structured facts from free text passages.",
    "version": "1.2.0",
    "supportedInterfaces": [
        {"url": EXTRACTOR_URL, "protocolBinding": "JSONRPC", "protocolVersion": "1.0"},
    ],
    "capabilities": {"streaming": True, "pushNotifications": False},
    "securitySchemes": {
        "bearer": {"httpAuthSecurityScheme": {"scheme": "Bearer", "bearerFormat": "JWT"}},
    },
    "securityRequirements": [{"schemes": {"bearer": {"list": []}}}],
    "defaultInputModes": ["text/plain"],
    "defaultOutputModes": ["application/json"],
    "skills": [
        {
            "id": "extract_facts",
            "name": "Fact extractor",
            "description": "Pull named entities and numeric facts from a passage.",
            "tags": ["extraction", "facts"],
            "examples": ["Extract revenue and growth from this earnings note."],
            "inputModes": ["text/plain"],
            "outputModes": ["application/json"],
        }
    ],
}


# --- 2. The task lifecycle ---------------------------------------------
# The caller sends a *message*; the receiver decides whether to answer
# with a message or open a *task*. A task has an id, a context id and a
# state machine:
#   SUBMITTED -> WORKING -> (COMPLETED | FAILED | CANCELED | REJECTED)
# with INPUT_REQUIRED and AUTH_REQUIRED as pauses along the way.
# Streamed events are JSON-RPC responses whose `result` holds one of
# `task`, `message`, `statusUpdate` or `artifactUpdate`.

TERMINAL = {"TASK_STATE_COMPLETED", "TASK_STATE_FAILED",
            "TASK_STATE_CANCELED", "TASK_STATE_REJECTED"}


def send_streaming_message(prompt: str, caller_id: str) -> dict:
    """The JSON-RPC request body a caller agent sends to the receiver."""
    return {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "SendStreamingMessage",
        "params": {
            "message": {
                "messageId": str(uuid.uuid4()),
                "role": "ROLE_USER",
                "parts": [{"text": prompt}],
                "metadata": {"caller": caller_id, "sentAt": now()},
            },
            "configuration": {"acceptedOutputModes": ["application/json"]},
        },
    }


def stream_event(request_id: str, payload: dict) -> dict:
    """Wrap one StreamResponse in the JSON-RPC envelope of the request."""
    return {"jsonrpc": "2.0", "id": request_id, "result": payload}


def task_created(task_id: str, context_id: str) -> dict:
    return {"task": {
        "id": task_id,
        "contextId": context_id,
        "status": {"state": "TASK_STATE_SUBMITTED", "timestamp": now()},
    }}


def status_update(task_id: str, context_id: str, state: str, note: str = "") -> dict:
    status: dict = {"state": state, "timestamp": now()}
    if note:
        status["message"] = {
            "messageId": str(uuid.uuid4()),
            "role": "ROLE_AGENT",
            "parts": [{"text": note}],
        }
    return {"statusUpdate": {"taskId": task_id, "contextId": context_id, "status": status}}


def artifact_update(task_id: str, context_id: str, name: str, payload: dict) -> dict:
    """A typed output attached to the task: here one JSON `data` part."""
    return {"artifactUpdate": {
        "taskId": task_id,
        "contextId": context_id,
        "artifact": {
            "artifactId": str(uuid.uuid4()),
            "name": name,
            "parts": [{"data": payload, "mediaType": "application/json"}],
        },
        "lastChunk": True,
    }}


# --- 3. The "extractor" agent's actual work ----------------------------
# In a real A2A deployment this would be a different process behind a
# different URL. Here it's just a function: the protocol is what we're
# showing, not the deployment.


def extractor_handle(request: dict) -> Iterator[dict]:
    """Yield stream events as the extractor processes the message.

    A real implementation would write each one as a `data:` line of a
    server sent events stream.
    """
    request_id = request["id"]
    user_text = request["params"]["message"]["parts"][0]["text"]
    task_id = f"task-{uuid.uuid4().hex[:8]}"
    context_id = f"ctx-{uuid.uuid4().hex[:8]}"

    yield stream_event(request_id, task_created(task_id, context_id))
    yield stream_event(request_id, status_update(task_id, context_id, "TASK_STATE_WORKING", "calling LLM"))

    response = chat(
        system="You extract structured facts. Reply with ONLY a JSON object.",
        messages=[{"role": "user", "content": user_text}],
    )
    # Crude parse: the real extractor would pass `output_schema` to chat()
    # (Lesson 02) so the API itself guarantees valid JSON.
    try:
        facts = json.loads(response.text)
    except json.JSONDecodeError:
        facts = {"raw": response.text}

    yield stream_event(request_id, artifact_update(task_id, context_id, "facts", facts))
    yield stream_event(request_id, status_update(task_id, context_id, "TASK_STATE_COMPLETED"))


# --- 4. The "researcher" agent's call ----------------------------------


def researcher_delegate(prompt: str) -> dict | None:
    """Send a message to the extractor and consume its event stream until
    the task reaches a terminal state. Returns the artifact, if any."""
    request = send_streaming_message(prompt, caller_id="acme-researcher")
    print(f"-> POST {EXTRACTOR_URL}  (method {request['method']})")
    print(f"   {json.dumps(request, indent=2)[:300]}...\n")

    artifact = None
    for event in extractor_handle(request):
        result = event["result"]
        if "task" in result:
            print(f"<- task {result['task']['id']} {result['task']['status']['state']}")
        elif "statusUpdate" in result:
            status = result["statusUpdate"]["status"]
            parts = status.get("message", {}).get("parts", [{}])
            print(f"<- {status['state']:22s}  {parts[0].get('text', '')}")
            if status["state"] in TERMINAL:
                break
        elif "artifactUpdate" in result:
            artifact = result["artifactUpdate"]["artifact"]
            print(f"<- artifact '{artifact['name']}'")
    return artifact


if __name__ == "__main__":
    print("=== AgentCard served at /.well-known/agent-card.json ===")
    print(json.dumps(EXTRACTOR_AGENT_CARD, indent=2))

    print("\n=== Researcher delegates a task to Extractor ===\n")
    artifact = researcher_delegate(
        "From: 'Acme Q2 revenue was $14.2M, up 22% YoY, driven by EU expansion.' "
        "Extract revenue, growth_pct, and primary_driver."
    )

    if artifact is None:
        print("\n(no artifact returned)")
    else:
        print(f"\n=== Final artifact ({artifact['name']}) ===")
        print(json.dumps(artifact["parts"][0]["data"], indent=2))
