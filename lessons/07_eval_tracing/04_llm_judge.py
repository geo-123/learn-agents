"""Lesson 07: LLM as judge with a rubric.

We rerun the fixtures from 03_ but layer on a semantic judge: a
separate chat() call that scores each answer 1 to 5 on faithfulness and
helpfulness. The judge is a different "kind" of agent: its prompt is
a rubric, not an open task.

Caveats (in the lesson README): position bias, verbosity bias,
self preference. Always document the judge model.

Run:
    uv run python lessons/07_eval_tracing/04_llm_judge.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(HERE.parents[1]))
from src._llm import chat

rag_path = HERE.parents[0] / "03b_rag_in_loop" / "01_rag_tool.py"
spec = importlib.util.spec_from_file_location("_rag", rag_path)
assert spec and spec.loader
_rag = importlib.util.module_from_spec(spec)
sys.modules["_rag"] = _rag  # Python 3.13: @dataclass needs the module registered before exec.
spec.loader.exec_module(_rag)


JUDGE_MODEL = "claude-sonnet-5-5"  # explicitly different from (and stronger than) the default Haiku.


class JudgeScore(BaseModel):
    faithfulness: int = Field(
        ge=1, le=5,
        description=(
            "1=fabricated, 3=partly supported, 5=fully grounded in the corpus. "
            "If the agent says 'I don't know' for a genuinely out of scope question, "
            "that's a 5."
        ),
    )
    helpfulness: int = Field(
        ge=1, le=5,
        description="1=useless, 3=okay, 5=directly answers the question.",
    )
    rationale: str = Field(description="One short sentence.")


JUDGE_SYSTEM = (
    "You are evaluating an AI assistant's answer to a question about the "
    "Acme company handbook. Score the answer 1 to 5 on TWO axes: "
    "faithfulness (is the answer supported by the handbook?) and "
    "helpfulness (does it actually answer the question?). Be strict. "
    "If you don't have the handbook, judge based on whether the answer "
    "looks specific, concrete, and likely correct given a policy doc context."
)


def judge(question: str, answer: str) -> JudgeScore:
    prompt = (
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Score the answer."
    )
    response = chat(
        messages=[{"role": "user", "content": prompt}],
        system=JUDGE_SYSTEM,
        output_schema=JudgeScore,  # structured output: the scores come back typed
        model=JUDGE_MODEL,
    )
    if response.parsed is None:
        raise RuntimeError(f"judge gave no score (stop_reason={response.stop_reason})")
    return response.parsed


def main() -> None:
    fixtures = [
        json.loads(line)
        for line in (HERE / "fixtures.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    rows = []
    for fx in fixtures:
        print(f"\n--- {fx['id']} ---")
        answer = _rag.run_agent(fx["input"])
        score = judge(fx["input"], answer)
        print(f"Q: {fx['input']}")
        print(f"A: {answer[:120]}...")
        print(f"  faithfulness={score.faithfulness}  helpfulness={score.helpfulness}")
        print(f"  rationale: {score.rationale}")
        rows.append((fx["id"], score.faithfulness, score.helpfulness))

    print(f"\n{'='*60}\nJudge: {JUDGE_MODEL}")
    print(f"{'id':<20} {'faith':>6} {'help':>6}")
    for rid, f, h in rows:
        print(f"{rid:<20} {f:>6} {h:>6}")
    avg_f = sum(f for _, f, _ in rows) / len(rows)
    avg_h = sum(h for _, _, h in rows) / len(rows)
    print(f"{'AVERAGE':<20} {avg_f:>6.2f} {avg_h:>6.2f}")


if __name__ == "__main__":
    main()
