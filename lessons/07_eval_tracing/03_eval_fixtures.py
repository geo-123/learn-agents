"""Lesson 07: deterministic eval harness over fixtures.

Loads fixtures.jsonl, runs the RAG agent from Lesson 03b on each input,
checks the answer with must_contain / must_not_contain assertions, and
prints a pass/fail summary.

Run:
    uv run python lessons/07_eval_tracing/03_eval_fixtures.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Import the RAG agent from 03b/01_rag_tool.py
rag_path = HERE.parents[0] / "03b_rag_in_loop" / "01_rag_tool.py"
spec = importlib.util.spec_from_file_location("_rag", rag_path)
assert spec and spec.loader
_rag = importlib.util.module_from_spec(spec)
sys.modules["_rag"] = _rag  # Python 3.13: @dataclass needs the module registered before exec.
spec.loader.exec_module(_rag)


def load_fixtures() -> list[dict]:
    return [
        json.loads(line)
        for line in (HERE / "fixtures.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def check(answer: str, fixture: dict) -> tuple[bool, list[str]]:
    """Return (passed, [reasons_failed])."""
    failures: list[str] = []
    a_lower = answer.lower()

    must_contain = [s.lower() for s in fixture.get("must_contain", [])]
    must_not_contain = [s.lower() for s in fixture.get("must_not_contain", [])]
    match_any = fixture.get("match_any_of_contain", False)

    if must_contain:
        if match_any:
            if not any(s in a_lower for s in must_contain):
                failures.append(f"none of {must_contain} found in answer")
        else:
            missing = [s for s in must_contain if s not in a_lower]
            if missing:
                failures.append(f"missing required substrings: {missing}")

    forbidden = [s for s in must_not_contain if s in a_lower]
    if forbidden:
        failures.append(f"contains forbidden substrings: {forbidden}")

    return (not failures), failures


def main() -> None:
    fixtures = load_fixtures()
    print(f"running {len(fixtures)} fixtures...")

    results = []
    for fx in fixtures:
        print(f"\n--- {fx['id']} ---")
        print(f"INPUT: {fx['input']}")
        answer = _rag.run_agent(fx["input"])
        print(f"ANSWER: {answer[:200]}{'...' if len(answer) > 200 else ''}")
        passed, reasons = check(answer, fx)
        status = "PASS" if passed else "FAIL"
        print(f"{status}  " + ("; ".join(reasons) if reasons else ""))
        results.append({"id": fx["id"], "passed": passed, "reasons": reasons})

    passed = sum(1 for r in results if r["passed"])
    print(f"\n{'='*60}\nTOTAL: {passed}/{len(results)} passed")
    if passed < len(results):
        print("Failures:")
        for r in results:
            if not r["passed"]:
                print(f"  {r['id']}: {r['reasons']}")


if __name__ == "__main__":
    main()
