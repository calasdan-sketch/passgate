"""Many eval cases at once: find the regressions without drowning in false alarms.

Run 50 cases and test each at 5% and you expect 2.5 false alarms even when nothing changed
(about a 92% chance of at least one). Here each case is tested (one-sided Fisher exact: is the
candidate WORSE than the baseline on this case?) and the p-values are adjusted:

* ``holm``  - Holm-Bonferroni step-down, controls the family-wise error rate (any false alarm) at alpha.
* ``bh``    - Benjamini-Hochberg, controls the false discovery rate (share of flags that are false) at alpha.
* ``none``  - raw p-values, for comparison only.

Note the contrast with ``gate``: demanding that EVERY round clear a bar is an intersection-union test, which
is already valid at level alpha with no correction. Corrections matter when you *search* many cases for a failure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from .safe import md_cell, safe_text
from .stats import FISHER_MAX_TRIALS, fisher_exact_2x2

ADJUSTMENTS = ("holm", "bh", "none")
MAX_CASES = 100_000


@dataclass(frozen=True)
class Case:
    name: str
    cand_passes: int
    cand_trials: int
    base_passes: int
    base_trials: int


def holm_adjust(pvals: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values (monotone step-down), same order as input."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def bh_adjust(pvals: Sequence[float]) -> List[float]:
    """Benjamini-Hochberg adjusted p-values (q-values), same order as input."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i], reverse=True)
    adj = [0.0] * m
    running = 1.0
    for pos, i in enumerate(order):
        rank = m - pos
        running = min(running, pvals[i] * m / rank)
        adj[i] = min(1.0, running)
    return adj


@dataclass
class CaseRow:
    case: Case
    p_worse: float  # raw one-sided p: candidate is worse than baseline
    p_better: float
    adj_worse: float
    adj_better: float
    regressed: bool
    improved: bool


@dataclass
class CasesResult:
    rows: List[CaseRow]
    alpha: float
    adjust: str

    @property
    def regressions(self) -> List[CaseRow]:
        return [r for r in self.rows if r.regressed]

    @property
    def improvements(self) -> List[CaseRow]:
        return [r for r in self.rows if r.improved]

    @property
    def passed(self) -> bool:
        """No case shows a statistically supported regression."""
        return not self.regressions

    def as_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "adjust": self.adjust,
            "alpha": self.alpha,
            "cases": len(self.rows),
            "regressions": len(self.regressions),
            "improvements": len(self.improvements),
            "rows": [
                {
                    "case": r.case.name,
                    "candidate": f"{r.case.cand_passes}/{r.case.cand_trials}",
                    "baseline": f"{r.case.base_passes}/{r.case.base_trials}",
                    "p_worse": round(r.p_worse, 6),
                    "adjusted_p_worse": round(r.adj_worse, 6),
                    "regressed": r.regressed,
                    "improved": r.improved,
                }
                for r in self.rows
            ],
        }

    def __str__(self) -> str:
        lines = [
            f"cases: {len(self.rows)} case(s), {len(self.regressions)} regression(s), "
            f"{len(self.improvements)} improvement(s) ({self.adjust}-adjusted, alpha {self.alpha:g})"
        ]
        for r in sorted(self.rows, key=lambda r: r.p_worse)[:20]:
            tag = "REGRESSED" if r.regressed else ("improved" if r.improved else "ok")
            lines.append(
                f"  {tag:9} {safe_text(r.case.name, 40):>20}: cand {r.case.cand_passes}/{r.case.cand_trials} "
                f"vs base {r.case.base_passes}/{r.case.base_trials}  p={r.p_worse:.4g} adj={r.adj_worse:.4g}"
            )
        if len(self.rows) > 20:
            lines.append(f"  ... {len(self.rows) - 20} more case(s) not shown (use --json for all)")
        lines.append(f"verdict: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)

    def explain(self) -> str:
        n = len(self.rows)
        reg = self.regressions
        if not reg:
            extra = f" {len(self.improvements)} case(s) got better." if self.improvements else ""
            return (
                f"We compared the candidate with the baseline on each of {n} cases. After allowing for the fact that "
                f"checking many cases produces some false alarms by luck, none got reliably worse.{extra}"
            )
        names = ", ".join(safe_text(r.case.name, 30) for r in reg[:5])
        return (
            f"{len(reg)} of {n} cases got reliably worse even after allowing for checking many cases at once "
            f"(first: {names}). Treat these as real regressions, not noise."
        )

    def to_markdown(self) -> str:
        out = [
            f"### passgate cases: {'PASS' if self.passed else 'FAIL'}",
            "",
            self.explain(),
            "",
            "| | case | candidate | baseline | p (worse) | adjusted p |",
            "|---|---|---|---|---|---|",
        ]
        for r in sorted(self.rows, key=lambda r: r.p_worse)[:50]:
            tag = "REGRESSED" if r.regressed else ("improved" if r.improved else "ok")
            out.append(
                f"| {tag} | {md_cell(r.case.name, 60)} | {r.case.cand_passes}/{r.case.cand_trials} | "
                f"{r.case.base_passes}/{r.case.base_trials} | {r.p_worse:.4g} | {r.adj_worse:.4g} |"
            )
        if len(self.rows) > 50:
            out.append(f"\n{len(self.rows) - 50} more case(s) not shown.")
        return "\n".join(out) + "\n"


def compare_cases(cases: Sequence[Case], alpha: float = 0.05, adjust: str = "holm") -> CasesResult:
    """Per-case regression search with multiplicity control. Raises ValueError on empty or oversized input."""
    if not cases:
        raise ValueError("cases needs at least one case")
    if len(cases) > MAX_CASES:
        raise ValueError(f"too many cases (limit {MAX_CASES:,})")
    if not 0.0 < alpha <= 0.5:
        raise ValueError("alpha must be in (0, 0.5], e.g. 0.05")
    if adjust not in ADJUSTMENTS:
        raise ValueError(f"adjust must be one of {', '.join(ADJUSTMENTS)}")
    p_worse: List[float] = []
    p_better: List[float] = []
    for c in cases:
        if min(c.cand_trials, c.base_trials) <= 0 or not (
            0 <= c.cand_passes <= c.cand_trials and 0 <= c.base_passes <= c.base_trials
        ):
            raise ValueError(f"case {safe_text(c.name, 40)!r}: passes must be within [0, trials] and trials positive")
        if c.cand_trials + c.base_trials > FISHER_MAX_TRIALS:
            raise ValueError(f"case {safe_text(c.name, 40)!r}: too many trials for the exact test")
        table = (c.cand_passes, c.cand_trials - c.cand_passes, c.base_passes, c.base_trials - c.base_passes)
        p_worse.append(fisher_exact_2x2(*table, alternative="less"))
        p_better.append(fisher_exact_2x2(*table, alternative="greater"))
    fn = {"holm": holm_adjust, "bh": bh_adjust, "none": lambda p: list(p)}[adjust]
    aw, ab = fn(p_worse), fn(p_better)
    rows = [
        CaseRow(c, pw, pb, w, b, w < alpha and w <= b, b < alpha and b < w)
        for c, pw, pb, w, b in zip(cases, p_worse, p_better, aw, ab)
    ]
    return CasesResult(rows, alpha, adjust)
