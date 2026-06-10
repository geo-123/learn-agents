"""Lesson 10 — capstone research agent.

Synthesizes every prior topic into one runnable program. Reads like a
synthesis of every file before it; intentionally denser than any one
of them in isolation.

Run:
    uv run python lessons/10_capstone/capstone_agent.py "How many vacation days do I get?"
"""
from __future__ import annotations

import contextlib
import contextvars
import importlib.util
import json
import random
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import anthropic
from pydantic import BaseModel, Field, ValidationError

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from src._llm import chat, LLMResponse


# -------------------------------------------------------------------------
# Reuse the TF-IDF index from Lesson 03a. Same retrieval pattern.
# -------------------------------------------------------------------------
rag_path = HERE.parents[0] / "03a_retrieval_basics" / "02_tfidf_search.py"
spec = importlib.util.spec_from_file_location("_tfidf", rag_path)
assert spec and spec.loader
_tfidf = importlib.util.module_from_spec(spec)
sys.modules["_tfidf"] = _tfidf  # Python 3.13: @dataclass needs the module registered before exec.
spec.loader.exec_module(_tfidf)
INDEX = _tfidf.build_index()
CHUNK_TEXTS: dict[str, str] = {f"{c.doc}#{c.idx}": c.text for c in INDEX.chunks}


# -------------------------------------------------------------------------
# Tracing (Lesson 07). One JSONL file per agent run.
# -------------------------------------------------------------------------
TRACES_DIR = HERE.parents[1] / "traces"
TRACES_DIR.mkdir(exist_ok=True)
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
    def span(self, kind: str, **attrs):
        sid = uuid.uuid4().hex[:8]
        parent = _current_span.get()
        start = time.monotonic()
        token = _current_span.set(sid)
        status = "ok"
        out: dict[str, Any] = {}
        try:
            yield out
        except Exception as e:  # noqa: BLE001
            status = f"error: {type(e).__name__}: {e}"
            raise
        finally:
            _current_span.reset(token)
            self._write({
                "run_id": self.run_id,
                "span_id": sid,
                "parent_id": parent,
                "kind": kind,
                "ms": round((time.monotonic() - start) * 1000, 1),
                "status": status,
                "attrs": attrs,
                "outputs": out,
            })


# -------------------------------------------------------------------------
# Input guard (Lesson 08). Refuses off-topic / PII.
# -------------------------------------------------------------------------
INJECTION = re.compile(r"ignore (the )?(previous|prior|above)", re.I)
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def input_guard(text: str) -> tuple[bool, str, str]:
    """Returns (allowed, cleaned, reason)."""
    if SSN.search(text):
        return False, "", "refused: input contains an SSN"
    cleaned = INJECTION.sub("[removed]", text)
    return True, cleaned, "ok"


# -------------------------------------------------------------------------
# Tools (Lesson 01 dispatch pattern, with arg guards from Lesson 08).
# -------------------------------------------------------------------------
MEMORY_PATH = HERE / "capstone_memory.json"


def _load_mem() -> dict:
    return json.loads(MEMORY_PATH.read_text(encoding="utf-8")) if MEMORY_PATH.exists() else {}


def _save_mem(d: dict) -> None:
    MEMORY_PATH.write_text(json.dumps(d, indent=2), encoding="utf-8")


class SearchArgs(BaseModel):
    query: str
    k: int = Field(default=3, ge=1, le=6)


def search_docs(args: SearchArgs) -> dict:
    return {
        "hits": [
            {"id": f"{c.doc}#{c.idx}", "text": c.text, "score": round(s, 3)}
            for c, s in INDEX.search(args.query, k=args.k)
        ]
    }


class RememberArgs(BaseModel):
    key: str
    value: str


def remember(args: RememberArgs) -> dict:
    store = _load_mem()
    store[args.key] = args.value
    _save_mem(store)
    return {"stored": {args.key: args.value}}


class RecallArgs(BaseModel):
    key: str


def recall(args: RecallArgs) -> dict:
    store = _load_mem()
    return {"key": args.key, "value": store.get(args.key)}


class AskUserArgs(BaseModel):
    question: str


def ask_user(args: AskUserArgs) -> dict:
    print(f"\n[agent asks] {args.question}")
    return {"answer": input("you> ").strip()}


class ResearchReport(BaseModel):
    """Structured final output (Lesson 02)."""
    answer: str = Field(description="The answer in plain prose.")
    citations: list[str] = Field(description="Chunk ids you used.")
    confidence: float = Field(ge=0.0, le=1.0)
    used_memory: bool = Field(description="True if you read from long-term memory.")


def submit_report(args: ResearchReport) -> dict:
    return {"acknowledged": True}


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[BaseModel], Any]
    guard: Callable[[BaseModel], tuple[bool, str]] | None = None

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.args_model.model_json_schema(),
        }


REGISTRY: dict[str, Tool] = {t.name: t for t in [
    Tool("search_docs",
         "Search the Acme handbook. Returns hits with id, text, score. "
         "Cite ids you actually used.",
         SearchArgs, search_docs),
    Tool("remember",
         "Save a fact about the user for future sessions. Use sparingly.",
         RememberArgs, remember),
    Tool("recall",
         "Look up a previously-saved fact about the user.",
         RecallArgs, recall),
    Tool("ask_user",
         "Ask the user ONE clarifying question when their request is ambiguous. "
         "Use this at most once.",
         AskUserArgs, ask_user),
    Tool("submit_report",
         "Submit the final research report. Call this exactly once when you're done.",
         ResearchReport, submit_report),
]}


def dispatch(tracer: Tracer, tool_use: dict) -> dict:
    name, use_id = tool_use["name"], tool_use["id"]
    with tracer.span("tool_call", name=name, input=tool_use.get("input")) as out:
        tool = REGISTRY.get(name)
        if tool is None:
            out["error"] = f"unknown tool {name!r}"
            return _err(use_id, out["error"])
        try:
            args = tool.args_model.model_validate(tool_use.get("input", {}) or {})
        except ValidationError as e:
            out["error"] = f"invalid args: {e.errors()}"
            return _err(use_id, out["error"])
        if tool.guard:
            ok, why = tool.guard(args)
            if not ok:
                out["blocked"] = why
                return _err(use_id, f"BLOCKED: {why}")
        try:
            result = tool.fn(args)
            out["result_preview"] = json.dumps(result)[:200]
        except Exception as e:  # noqa: BLE001
            out["error"] = f"{type(e).__name__}: {e}"
            return _err(use_id, out["error"])
        return {"type": "tool_result", "tool_use_id": use_id, "content": json.dumps(result)}


def _err(use_id: str, msg: str) -> dict:
    return {"type": "tool_result", "tool_use_id": use_id, "content": msg, "is_error": True}


# -------------------------------------------------------------------------
# Transport retry (Lesson 05). Wraps chat() with exponential backoff.
# -------------------------------------------------------------------------
def chat_with_retry(*args, max_attempts: int = 4, **kwargs) -> LLMResponse:
    for attempt in range(1, max_attempts + 1):
        try:
            return chat(*args, **kwargs)
        except (anthropic.RateLimitError, anthropic.APIConnectionError) as exc:
            if attempt == max_attempts:
                raise
            delay = min(0.8 * 2 ** (attempt - 1), 16.0)
            time.sleep(delay + delay * random.uniform(-0.3, 0.3))
            print(f"  ! transport retry {attempt} ({type(exc).__name__})")
        except anthropic.APIStatusError as exc:
            if 500 <= exc.status_code < 600 and attempt < max_attempts:
                delay = min(0.8 * 2 ** (attempt - 1), 16.0)
                time.sleep(delay + delay * random.uniform(-0.3, 0.3))
                print(f"  ! transport retry {attempt} (5xx)")
                continue
            raise
    raise RuntimeError("unreachable")


# -------------------------------------------------------------------------
# Budget (Lesson 00 / 05).
# -------------------------------------------------------------------------
@dataclass
class Budget:
    max_steps: int = 8
    max_seconds: float = 60.0
    max_input_tokens: int = 30_000
    steps: int = 0
    tokens: int = 0
    started: float = 0.0

    def start(self) -> None:
        self.started = time.monotonic()

    def exhausted(self) -> tuple[bool, str]:
        if self.steps >= self.max_steps:
            return True, "max_steps"
        if time.monotonic() - self.started >= self.max_seconds:
            return True, "max_seconds"
        if self.tokens >= self.max_input_tokens:
            return True, "max_input_tokens"
        return False, ""


# -------------------------------------------------------------------------
# Citation verification (Lesson 03b idea).
# -------------------------------------------------------------------------
def verify(report: ResearchReport) -> tuple[bool, str]:
    if not report.citations:
        return False, "no citations"
    for cid in report.citations:
        if cid not in CHUNK_TEXTS:
            return False, f"hallucinated citation: {cid}"
    return True, "ok"


# -------------------------------------------------------------------------
# Main agent loop. The thing that ties it all together.
# -------------------------------------------------------------------------
SYSTEM = (
    "You are a research assistant for Acme employees. Answer ONLY using the "
    "company handbook. Rules:\n"
    "- Start by calling recall('user_name') to see if you know the user.\n"
    "- Use search_docs to look things up. Make multiple searches if needed.\n"
    "- If the question is ambiguous, call ask_user ONCE to clarify.\n"
    "- If the user mentions their name or a preference, call remember.\n"
    "- When you have your answer, call submit_report. End the loop there.\n"
    "- If the handbook genuinely doesn't cover the topic, say so in the report "
    "  with confidence < 0.3."
)


def run_agent(question: str) -> tuple[ResearchReport | None, Path]:
    trace_path = TRACES_DIR / f"capstone_{int(time.time())}_{uuid.uuid4().hex[:4]}.jsonl"
    tracer = Tracer(path=trace_path)
    budget = Budget()

    with tracer.span("agent_run", question=question) as run_out:
        # Input guard runs first; nothing reaches the LLM if it fails.
        allowed, cleaned, reason = input_guard(question)
        if not allowed:
            run_out["refused"] = reason
            print(f"[input_guard] refused: {reason}")
            return None, trace_path

        tools = [t.schema() for t in REGISTRY.values()]
        messages: list[dict] = [{"role": "user", "content": cleaned}]
        budget.start()
        report: ResearchReport | None = None

        while True:
            done, why = budget.exhausted()
            if done:
                print(f"[budget] exhausted: {why}")
                run_out["budget_exhausted"] = why
                break

            with tracer.span("llm_call", step=budget.steps + 1) as turn_out:
                response = chat_with_retry(messages, system=SYSTEM, tools=tools)
            budget.steps += 1
            budget.tokens += response.input_tokens
            messages.append({"role": "assistant", "content": response.content})

            print(f"\n--- step {budget.steps}  stop={response.stop_reason} ---")
            for tu in response.tool_uses:
                print(f"  -> {tu['name']}({list(tu['input'].keys())})")

            # If the model called submit_report, parse it and end.
            for tu in response.tool_uses:
                if tu["name"] == "submit_report":
                    try:
                        report = ResearchReport.model_validate(tu["input"])
                    except ValidationError as e:
                        # Schema-repair retry (Lesson 02/05).
                        messages.append({
                            "role": "user",
                            "content": (
                                "submit_report failed validation: "
                                f"{e.errors()}. Try again with valid args."
                            ),
                        })
                        continue
                    return report, trace_path

            if response.stop_reason != "tool_use":
                # Model gave a final text without calling submit_report.
                # Treat as an error — we expected a report.
                run_out["error"] = "model did not call submit_report"
                break

            # Otherwise, dispatch the tool calls and loop.
            results = [dispatch(tracer, tu) for tu in response.tool_uses]
            messages.append({"role": "user", "content": results})

    return report, trace_path


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "How many vacation days do I get?"
    print(f"Q: {q}")
    report, trace_path = run_agent(q)

    if report is None:
        print("\n(no report)")
    else:
        ok, why = verify(report)
        print(f"\n=== report ===")
        print(f"answer:     {report.answer}")
        print(f"citations:  {report.citations}")
        print(f"confidence: {report.confidence:.2f}")
        print(f"verified:   {'PASS' if ok else 'FAIL'} ({why})")
    print(f"\ntrace: {trace_path}")
