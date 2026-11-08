"""Lesson 10: capstone research agent.

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
# Input guard (Lesson 08). Refuses PII and off topic questions.
# -------------------------------------------------------------------------
# Same patterns as Lesson 08, 01_input_guard.py.
INJECTION_PATTERNS = [
    re.compile(r"ignore (the )?(previous|prior|above)( instructions)?", re.I),
    re.compile(r"you are now (?:a |an )?", re.I),
    re.compile(r"(disregard|forget) (your|the) (system|prior|previous)", re.I),
    re.compile(r"<<\s*system\s*>>", re.I),
]
PII_PATTERNS = {
    "us_ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
}
# The input guard blocks the high risk classes; an email address in a
# question is fine ("where do I forward phishing?" may name one).
INPUT_BLOCKED_PII = ("us_ssn", "credit_card")

# Off topic check, rule based (Lesson 08: no LLM as guard where a cheap rule
# does). A question is on topic when it shares at least one content word with
# the handbook, so the vocabulary comes from the corpus itself, not from a
# hand written topic list. Function words and generic request words do not
# count: every question has them. The rule is deliberately loose: "pets in
# the office" passes (the agent then reports that the handbook does not cover
# it), "the capital of Portugal" does not. It is a quality guard, so a miss
# only costs one wasted run.
HANDBOOK_VOCAB = set(INDEX.vocab_index)
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "for", "with", "from", "by", "as", "about", "into", "is", "are", "was",
    "were", "be", "been", "do", "does", "did", "can", "could", "should",
    "would", "will", "may", "might", "must", "i", "me", "my", "we", "our",
    "you", "your", "it", "its", "this", "that", "these", "those", "there",
    "what", "who", "whom", "where", "when", "why", "how", "which", "s", "t",
    "please", "help", "tell", "know", "get", "give", "want", "need", "any",
    "some", "all", "no", "not", "so", "than", "then", "very", "just",
}


def content_words(text: str) -> set[str]:
    return {t for t in _tfidf.tokenize(text) if t not in STOPWORDS}


def input_guard(text: str) -> tuple[bool, str, str]:
    """Returns (allowed, cleaned, reason)."""
    for name in INPUT_BLOCKED_PII:
        if PII_PATTERNS[name].search(text):
            return False, "", f"refused: input contains {name}"
    cleaned, bare = text, text
    for pat in INJECTION_PATTERNS:
        cleaned = pat.sub("[removed]", cleaned)
        bare = pat.sub(" ", bare)
    # Checked without the stripped phrases, so the words of an injection
    # cannot make a question look on topic.
    if not content_words(bare) & HANDBOOK_VOCAB:
        return False, "", "refused: off topic (no word in common with the handbook)"
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


MAX_QUERY_CHARS = 300


def guard_search_docs(args: SearchArgs) -> tuple[bool, str]:
    """k is bounded by the schema (1 to 6); the guard bounds the query."""
    if not args.query.strip():
        return False, "empty query"
    if len(args.query) > MAX_QUERY_CHARS:
        return False, f"query too long (>{MAX_QUERY_CHARS} chars)"
    return True, ""


class RememberArgs(BaseModel):
    key: str
    value: str


MEMORY_KEY = re.compile(r"[a-z][a-z0-9_]{0,39}")
MAX_MEMORY_VALUE_CHARS = 200


def guard_memory_key(key: str) -> tuple[bool, str]:
    if not MEMORY_KEY.fullmatch(key):
        return False, f"key {key!r} must be snake_case, at most 40 chars"
    return True, ""


def guard_remember(args: RememberArgs) -> tuple[bool, str]:
    """Long term memory outlives the run, so it is stricter than the input
    guard: no PII of any class (emails included) and no large blobs."""
    ok, why = guard_memory_key(args.key)
    if not ok:
        return ok, why
    if len(args.value) > MAX_MEMORY_VALUE_CHARS:
        return False, f"value too long (>{MAX_MEMORY_VALUE_CHARS} chars)"
    for name, pat in PII_PATTERNS.items():
        if pat.search(args.value) or pat.search(args.key):
            return False, f"refusing to store {name} in long term memory"
    return True, ""


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


def guard_recall(args: RecallArgs) -> tuple[bool, str]:
    return guard_memory_key(args.key)


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
    used_memory: bool = Field(description="True if you read from long term memory.")


def submit_report(args: ResearchReport) -> dict:
    return {"acknowledged": True}


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[BaseModel], Any]
    # Runs after pydantic validation, before fn (Lesson 08, 03_tool_arg_guard.py).
    guard: Callable[[Any], tuple[bool, str]] | None = None

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
         SearchArgs, search_docs, guard_search_docs),
    Tool("remember",
         "Save a fact about the user for future sessions. Use sparingly. "
         "Keys are snake_case; never store contact details or ids.",
         RememberArgs, remember, guard_remember),
    Tool("recall",
         "Look up a previously saved fact about the user.",
         RecallArgs, recall, guard_recall),
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


MAX_SCHEMA_REPAIRS = 2  # invalid submit_report calls fed back before giving up

PARTIAL_PROMPT = (
    "The run is out of budget, so no more tools. Give your best answer from "
    "what you found so far. Cite only ids you saw in search results, and set "
    "the confidence to match what is still missing."
)


def _add_user_text(messages: list[dict], text: str) -> None:
    """Add a text block to the next user message.

    If the history already ends with a user message that is not sent yet (the
    tool results of the last turn), the text joins it after the tool_result
    blocks: two user messages in a row would be merged anyway, and tool
    results must come first in their message.
    """
    last = messages[-1]
    if last["role"] != "user":
        messages.append({"role": "user", "content": text})
        return
    content = last["content"]
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else list(content)
    last["content"] = blocks + [{"type": "text", "text": text}]


def partial_report(tracer: Tracer, messages: list[dict], tools: list[dict]) -> ResearchReport | None:
    """Graceful stop: one last call with tools switched off.

    tool_choice none keeps the tool definitions (the history holds tool_use
    blocks) but lets the model only write text, and output_schema makes that
    text a ResearchReport (Lesson 02). None if even this call yields nothing.
    """
    _add_user_text(messages, PARTIAL_PROMPT)
    with tracer.span("llm_call", step="partial") as out:
        try:
            response = chat_with_retry(
                messages, system=SYSTEM, tools=tools,
                tool_choice={"type": "none"}, output_schema=ResearchReport,
            )
        except ValidationError as e:
            # The API enforces the JSON shape; bounds such as confidence <= 1
            # are checked by pydantic afterwards and can still fail.
            out["error"] = f"partial report invalid: {e.errors()}"
            return None
        out["stop_reason"] = response.stop_reason
    messages.append({"role": "assistant", "content": response.content})
    return response.parsed


def run_agent(question: str) -> tuple[ResearchReport | None, Path, str]:
    """Returns (report, trace_path, outcome).

    outcome is "report", "partial: <budget limit>", "refused: <reason>" or
    "error: <reason>". A partial report is the best answer the model could
    give when the budget ran out; treat it as such.
    """
    trace_path = TRACES_DIR / f"capstone_{int(time.time())}_{uuid.uuid4().hex[:4]}.jsonl"
    tracer = Tracer(path=trace_path)
    budget = Budget()

    with tracer.span("agent_run", question=question) as run_out:
        # Input guard runs first; nothing reaches the LLM if it fails.
        allowed, cleaned, reason = input_guard(question)
        if not allowed:
            run_out["refused"] = reason
            print(f"[input_guard] {reason}")
            return None, trace_path, reason

        tools = [t.schema() for t in REGISTRY.values()]
        messages: list[dict] = [{"role": "user", "content": cleaned}]
        budget.start()
        schema_failures = 0

        while True:
            done, why = budget.exhausted()
            if done:
                print(f"[budget] exhausted: {why}; asking for a partial answer")
                run_out["budget_exhausted"] = why
                partial = partial_report(tracer, messages, tools)
                if partial is None:
                    run_out["error"] = "no partial report"
                    return None, trace_path, f"error: budget exhausted ({why}), no partial report"
                return partial, trace_path, f"partial: {why}"

            with tracer.span("llm_call", step=budget.steps + 1):
                response = chat_with_retry(messages, system=SYSTEM, tools=tools)
            budget.steps += 1
            budget.tokens += response.input_tokens
            messages.append({"role": "assistant", "content": response.content})

            print(f"\n--- step {budget.steps}  stop={response.stop_reason} ---")
            for tu in response.tool_uses:
                print(f"  -> {tu['name']}({list(tu['input'].keys())})")

            if response.stop_reason != "tool_use":
                # Model gave a final text without calling submit_report.
                # Treat as an error: we expected a report.
                run_out["error"] = "model did not call submit_report"
                return None, trace_path, "error: model did not call submit_report"

            # Every tool_use of this turn gets its tool_result, in ONE user
            # message, submit_report included: the API rejects a request in
            # which a tool_use is not answered by the very next user message.
            report: ResearchReport | None = None
            results: list[dict] = []
            for tu in response.tool_uses:
                result = dispatch(tracer, tu)
                if tu["name"] == "submit_report":
                    if result.get("is_error"):
                        # Schema repair (Lesson 02/05): the validation errors
                        # go back as an is_error tool_result.
                        schema_failures += 1
                        result["content"] += " Fix these fields and call submit_report again."
                    else:
                        report = ResearchReport.model_validate(tu["input"])
                results.append(result)

            if report is not None:
                # Done. The other tools of this turn ran; their results are
                # not sent because there is no next request.
                return report, trace_path, "report"

            messages.append({"role": "user", "content": results})
            if schema_failures > MAX_SCHEMA_REPAIRS:
                run_out["error"] = f"submit_report invalid {schema_failures} times"
                return None, trace_path, f"error: schema repair cap ({MAX_SCHEMA_REPAIRS} retries) reached"


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "How many vacation days do I get?"
    print(f"Q: {q}")
    report, trace_path, outcome = run_agent(q)

    if report is None:
        print(f"\n(no report: {outcome})")
    else:
        ok, why = verify(report)
        print(f"\n=== report ({outcome}) ===")
        print(f"answer:     {report.answer}")
        print(f"citations:  {report.citations}")
        print(f"confidence: {report.confidence:.2f}")
        print(f"verified:   {'PASS' if ok else 'FAIL'} ({why})")
    print(f"\ntrace: {trace_path}")
