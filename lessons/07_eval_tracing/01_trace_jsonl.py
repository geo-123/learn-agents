"""Lesson 07: JSONL tracer with nested spans.

A `Tracer` writes one span per line. `with tracer.span(kind, ...):` opens
a span; nested `with` blocks become children via a contextvar. The result
is a tree, written linearly. The viewer in 02_ rebuilds the tree.

Run:
    uv run python lessons/07_eval_tracing/01_trace_jsonl.py
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import chat


TRACES_DIR = Path(__file__).resolve().parents[2] / "traces"
TRACES_DIR.mkdir(exist_ok=True)


# A contextvar holds the *current* span id, so nested `with` blocks
# automatically pick up the right parent.
_current_span: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_span", default=None,
)


@dataclass
class Tracer:
    path: Path
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def _write(self, span: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(span) + "\n")

    @contextlib.contextmanager
    def span(self, kind: str, **attrs: Any):
        span_id = uuid.uuid4().hex[:8]
        parent = _current_span.get()
        start = time.monotonic()
        token = _current_span.set(span_id)
        status = "ok"
        outputs: dict[str, Any] = {}
        try:
            yield outputs
        except Exception as e:  # noqa: BLE001
            status = f"error: {type(e).__name__}: {e}"
            raise
        finally:
            _current_span.reset(token)
            self._write({
                "run_id": self.run_id,
                "span_id": span_id,
                "parent_id": parent,
                "kind": kind,
                "ms": round((time.monotonic() - start) * 1000, 1),
                "status": status,
                "attrs": attrs,
                "outputs": outputs,
            })


def traced_chat(tracer: Tracer, messages: list[dict], **kwargs):
    """Wrap a chat() call in an llm_call span."""
    with tracer.span("llm_call", n_messages=len(messages)) as out:
        response = chat(messages, **kwargs)
        out["input_tokens"] = response.input_tokens
        out["output_tokens"] = response.output_tokens
        out["stop_reason"] = response.stop_reason
        out["text_preview"] = response.text[:120]
        out["tool_uses"] = [tu["name"] for tu in response.tool_uses]
        return response


# --- Demo: trace a simple loop -------------------------------------------


def demo() -> None:
    out_path = TRACES_DIR / f"demo_{int(time.time())}.jsonl"
    tracer = Tracer(path=out_path)
    print(f"writing trace to {out_path}")

    with tracer.span("agent_run", task="explain RAG in two sentences"):
        messages: list[dict] = [
            {"role": "user", "content": "Explain RAG in two sentences."}
        ]
        for i in range(2):
            with tracer.span("step", n=i + 1):
                r = traced_chat(tracer, messages, max_tokens=200)
                messages.append({"role": "assistant", "content": r.content})
                if r.stop_reason != "tool_use":
                    break
                messages.append({"role": "user", "content": "Go on."})

    print(f"trace finished. View with: uv run python "
          f"lessons/07_eval_tracing/02_trace_viewer.py {out_path}")


if __name__ == "__main__":
    demo()
