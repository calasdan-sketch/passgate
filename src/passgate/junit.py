"""JUnit XML output so any CI system (GitHub, GitLab, Jenkins, Azure) shows passgate results as test cases."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any, List, Tuple

from .safe import safe_text


def _suite(name: str, cases: List[Tuple[str, bool, str]]) -> str:
    failures = sum(1 for _, ok, _ in cases if not ok)
    root = ET.Element("testsuites")
    suite = ET.SubElement(
        root, "testsuite", name=safe_text(name, 80), tests=str(len(cases)), failures=str(failures), errors="0"
    )
    for cname, ok, msg in cases:
        tc = ET.SubElement(suite, "testcase", classname="passgate", name=safe_text(cname, 120))
        if not ok:
            ET.SubElement(tc, "failure", message=safe_text(msg, 300)).text = safe_text(msg, 2000)
    if hasattr(ET, "indent"):
        ET.indent(root)
    # ASCII with numeric character references: valid under the declared encoding, and it survives
    # any console or redirect encoding (a Windows code page would otherwise mangle non-ASCII labels)
    body = ET.tostring(root, encoding="ascii").decode("ascii")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + body.split("?>", 1)[-1].lstrip("\n") + "\n"


def to_junit(obj: Any) -> str:
    """Render any passgate result object as JUnit XML. Unknown objects raise TypeError."""
    from .cluster import ItemsResult
    from .gate import Comparison, GateResult, PairedComparison
    from .multi import CasesResult
    from .noninf import NonInferiority
    from .sequential import SequentialResult

    if isinstance(obj, GateResult):
        cases = [
            (obj._name(i, r), r.meets, f"{r.round.passes}/{r.round.trials}, lower bound {r.lower:.3f} < {obj.threshold:g}; {obj._hint(r)}")
            for i, r in enumerate(obj.rounds)
        ]
        if len(obj.rounds) < obj.min_rounds:
            cases.append(("round-count", False, f"only {len(obj.rounds)} round(s), gate requires {obj.min_rounds}"))
        return _suite("passgate gate", cases)
    if isinstance(obj, CasesResult):
        cases = [
            (r.case.name, not r.regressed, f"worse than baseline: {r.case.cand_passes}/{r.case.cand_trials} vs {r.case.base_passes}/{r.case.base_trials}, adjusted p={r.adj_worse:.4g}")
            for r in obj.rows
        ]
        return _suite("passgate cases", cases)
    if isinstance(obj, (NonInferiority, ItemsResult, SequentialResult)):
        return _suite(f"passgate {type(obj).__name__}", [("verdict", obj.passed, str(obj))])
    if isinstance(obj, (Comparison, PairedComparison)):
        return _suite("passgate compare", [("candidate-better", obj.better, str(obj))])
    raise TypeError("cannot render this object as JUnit")
