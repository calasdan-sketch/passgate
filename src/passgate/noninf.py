"""Non-inferiority ("did we regress?") tests for candidate vs baseline pass rates.

``compare`` asks "is the candidate *better*?". In CI the usual question is the opposite:
"is the candidate *not meaningfully worse*?". A superiority test answers that badly, since two
identical prompts fail it about 95% of the time. Here the candidate passes when the lower end of a
confidence interval for (candidate - baseline) stays above ``-margin``.

Intervals are Newcombe's hybrid score intervals (Newcombe 1998, method 10): independent samples for
different items, and the paired version (with a correlation correction) when both arms ran the
same items. The verdict uses a one-sided level ``alpha``, i.e. a two-sided ``1 - 2*alpha`` interval.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from .gate import Round, _check_alpha
from .stats import _wilson, z_for_confidence

VERDICTS = ("superior", "non-inferior", "inconclusive", "inferior")


def newcombe_diff(c1: int, n1: int, c2: int, n2: int, confidence: float = 0.9) -> Tuple[float, float]:
    """Two-sided Newcombe-Wilson interval for p1 - p2 (independent samples)."""
    if n1 <= 0 or n2 <= 0:
        raise ValueError("sample sizes must be positive")
    if not (0 <= c1 <= n1 and 0 <= c2 <= n2):
        raise ValueError("passes must be within [0, trials]")
    z = z_for_confidence(confidence)
    p1, p2 = c1 / n1, c2 / n2
    l1, u1 = _wilson(p1, n1, z)
    l2, u2 = _wilson(p2, n2, z)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return max(-1.0, lo), min(1.0, hi)


def newcombe_paired_diff(both: int, cand_only: int, base_only: int, neither: int, confidence: float = 0.9) -> Tuple[float, float]:
    """Two-sided Newcombe (1998, method 10) interval for p1 - p2 from paired data."""
    a, b, c, d = both, cand_only, base_only, neither
    n = a + b + c + d
    if min(a, b, c, d) < 0 or n <= 0:
        raise ValueError("paired counts must be non-negative with at least one item")
    z = z_for_confidence(confidence)
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = _wilson(p1, n, z)
    l2, u2 = _wilson(p2, n, z)
    big_a = (a + b) * (c + d) * (a + c) * (b + d)
    if big_a == 0:
        phi = 0.0
    else:
        bb = a * d - b * c
        if bb - n / 2.0 > 0:
            cc = bb - n / 2.0
        elif bb >= -n / 2.0:
            cc = 0.0
        else:
            cc = bb + n / 2.0
        phi = cc / math.sqrt(big_a)
    diff = p1 - p2
    lo = diff - math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    hi = diff + math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return max(-1.0, lo), min(1.0, hi)


def _verdict(lower: float, upper: float, margin: float) -> str:
    if lower > 0.0:
        return "superior"
    if lower > -margin:
        return "non-inferior"
    if upper < -margin:
        return "inferior"
    return "inconclusive"


@dataclass
class NonInferiority:
    """Result of a non-inferiority test (independent or paired)."""

    candidate_rate: float
    baseline_rate: float
    diff: float
    lower: float
    upper: float
    margin: float
    alpha: float
    paired: bool = False
    items: Optional[int] = None

    @property
    def verdict(self) -> str:
        return _verdict(self.lower, self.upper, self.margin)

    @property
    def passed(self) -> bool:
        """True when the candidate is superior or non-inferior (not meaningfully worse)."""
        return self.verdict in ("superior", "non-inferior")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "design": "paired" if self.paired else "independent",
            "candidate_rate": round(self.candidate_rate, 6),
            "baseline_rate": round(self.baseline_rate, 6),
            "difference": round(self.diff, 6),
            "ci_lower": round(self.lower, 6),
            "ci_upper": round(self.upper, 6),
            "ci_confidence": round(1 - 2 * self.alpha, 6),
            "margin": self.margin,
            "alpha": self.alpha,
            "verdict": self.verdict,
            "passed": self.passed,
            **({"items": self.items} if self.items is not None else {}),
        }

    def __str__(self) -> str:
        return (
            f"regress ({'paired' if self.paired else 'independent'}): candidate {self.candidate_rate:.3f} vs "
            f"baseline {self.baseline_rate:.3f}, difference {self.diff:+.3f}, "
            f"{1 - 2 * self.alpha:.0%} interval [{self.lower:+.3f}, {self.upper:+.3f}], "
            f"margin {self.margin:g} - verdict: {self.verdict}"
        )

    def explain(self) -> str:
        pts = lambda x: f"{abs(x) * 100:.1f}"  # noqa: E731
        v = self.verdict
        head = (
            f"The candidate passes {self.candidate_rate:.1%} of the time and the baseline "
            f"{self.baseline_rate:.1%}. Allowing for luck, the true gap (candidate minus baseline) is "
            f"somewhere between {self.lower * 100:+.1f} and {self.upper * 100:+.1f} percentage points."
        )
        if v == "superior":
            tail = "Even the pessimistic end is above zero, so the candidate is genuinely better."
        elif v == "non-inferior":
            tail = (
                f"The pessimistic end is no worse than -{pts(self.margin)} points, which is inside the "
                f"{pts(self.margin)}-point drop you said you can tolerate. No meaningful regression."
            )
        elif v == "inferior":
            tail = (
                f"Even the optimistic end is more than {pts(self.margin)} points below the baseline, "
                "so the candidate is meaningfully worse. Block it."
            )
        else:
            tail = (
                f"The data cannot rule out a drop bigger than {pts(self.margin)} points. Either run more trials, "
                "or accept the risk explicitly. Treated as not passing."
            )
        return f"{head} {tail}"

    def to_markdown(self) -> str:
        return (
            f"### passgate regress: {self.verdict}\n\n"
            "| | candidate | baseline | difference | interval | margin |\n|---|---|---|---|---|---|\n"
            f"| pass rate | {self.candidate_rate:.3f} | {self.baseline_rate:.3f} | {self.diff:+.3f} | "
            f"\\[{self.lower:+.3f}, {self.upper:+.3f}\\] | {self.margin:g} |\n\n"
            f"{self.explain()}\n"
        )


def non_inferiority(
    candidate: Sequence[Round], baseline: Sequence[Round], margin: float, alpha: float = 0.05
) -> NonInferiority:
    """Is the candidate no worse than the baseline by more than ``margin`` (independent samples, pooled)?"""
    if not candidate or not baseline:
        raise ValueError("regress needs at least one candidate round and one baseline round")
    _check_alpha(alpha)
    _check_margin(margin)
    c1, n1 = sum(r.passes for r in candidate), sum(r.trials for r in candidate)
    c2, n2 = sum(r.passes for r in baseline), sum(r.trials for r in baseline)
    lo, hi = newcombe_diff(c1, n1, c2, n2, 1 - 2 * alpha)
    return NonInferiority(c1 / n1, c2 / n2, c1 / n1 - c2 / n2, lo, hi, margin, alpha)


def non_inferiority_paired(
    pairs: Iterable[Tuple[bool, bool]], margin: float, alpha: float = 0.05
) -> NonInferiority:
    """Non-inferiority when both arms ran on the SAME items (Newcombe paired interval)."""
    _check_alpha(alpha)
    _check_margin(margin)
    a = b = c = d = 0
    for cand, base in pairs:
        if not isinstance(cand, bool) or not isinstance(base, bool):
            raise ValueError("each pair must be (bool, bool)")
        if cand and base:
            a += 1
        elif cand:
            b += 1
        elif base:
            c += 1
        else:
            d += 1
    n = a + b + c + d
    if n == 0:
        raise ValueError("regress needs at least one item")
    lo, hi = newcombe_paired_diff(a, b, c, d, 1 - 2 * alpha)
    p1, p2 = (a + b) / n, (a + c) / n
    return NonInferiority(p1, p2, p1 - p2, lo, hi, margin, alpha, paired=True, items=n)


def _check_margin(margin: float) -> None:
    if not 0.0 < margin < 1.0:
        raise ValueError("margin must be between 0 and 1, e.g. 0.03 for 'at most 3 points worse'")
