"""Lesson 10: capstone eval runner.

Runs the capstone agent against fixtures.jsonl. Each fixture asserts:
  - The answer contains required substrings (or any of them, if match_any).
  - Citations include at least one expected doc.
  - Confidence is within an expected band.

Prints per fixture pass/fail, then a summary. Traces land in traces/.

Run:
    uv run python lessons/10_capstone/run_evals.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Import the capstone agent dynamically (numbered files aren't packages).
spec = importlib.util.spec_from_file_location("_cap", HERE / "capstone_agent.py")
assert spec and spec.loader
_cap = importlib.util.module_from_spec(spec)
sys.modules["_cap"] = _cap  # Python 3.13: @dataclass needs the module registered before exec.
spec.loader.exec_module(_cap)


def check(report, fx: dict) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if report is None:
        return False, ["agent returned no report"]

    text = report.answer.lower()
    must = [s.lower() for s in fx.get("must_contain", [])]
    if must:
        if fx.get("match_any_of_contain"):
            if not any(s in text for s in must):
                failures.append(f"none of {must} present in answer")
        else:
            missing = [s for s in must if s not in text]
            if missing:
                failures.append(f"missing: {missing}")

    cite_any = fx.get("must_cite_any", [])
    if cite_any and not any(any(c.startswith(d) for c in report.citations) for d in cite_any):
        failures.append(f"no citation matched any of {cite_any}; got {report.citations}")

    if "min_confidence" in fx and report.confidence < fx["min_confidence"]:
        failures.append(f"confidence {report.confidence:.2f} < {fx['min_confidence']}")
    if "max_confidence" in fx and report.confidence > fx["max_confidence"]:
        failures.append(f"confidence {report.confidence:.2f} > {fx['max_confidence']}")

    return (not failures), failures


def main() -> None:
    fixtures = [
        json.loads(line)
        for line in (HERE / "fixtures.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"running {len(fixtures)} fixtures...\n")

    rows = []
    for fx in fixtures:
        print(f"--- {fx['id']} ---")
        print(f"Q: {fx['input']}")
        report, trace = _cap.run_agent(fx["input"])
        ok, reasons = check(report, fx)
        print(f"{'PASS' if ok else 'FAIL'}  " + ("; ".join(reasons) if reasons else ""))
        print(f"(trace: {trace.name})\n")
        rows.append((fx["id"], ok, reasons))

    passed = sum(1 for _, ok, _ in rows if ok)
    print("=" * 60)
    print(f"TOTAL: {passed}/{len(rows)} passed")
    for rid, ok, reasons in rows:
        if not ok:
            print(f"  {rid}: {reasons}")


if __name__ == "__main__":
    main()
