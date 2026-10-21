"""The pass^N gate and the candidate-vs-baseline comparison."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .stats import fisher_exact_2x2, two_proportion_z, wilson_lower


@dataclass(frozen=True)
class Round:
    """One evaluation round: how many trials, how many passed."""

    trials: int
    passes: int
    label: str = ""

    def __post_init__(self) -> None:
        if self.trials <= 0:
            raise ValueError("trials must be positive")
        if not 0 <= self.passes <= self.trials:
            raise ValueError(f"passes {self.passes} outside [0, {self.trials}]")

    @property
    def rate(self) -> float:
        return self.passes / self.trials


@dataclass
class RoundEval:
    round: Round
    wilson_lower: float
    meets: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.round.label,
            "trials": self.round.trials,
            "passes": self.round.passes,
            "rate": round(self.round.rate, 6),
            "wilson_lower": round(self.wilson_lower, 6),
            "meets": self.meets,
        }


@dataclass
class GateResult:
    threshold: float
    confidence: float
    min_rounds: int
    rounds: List[RoundEval] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            len(self.rounds) >= self.min_rounds
            and all(r.meets for r in self.rounds)
        )

    def __bool__(self) -> bool:
        return self.passed

    def as_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "threshold": self.threshold,
            "confidence": self.confidence,
            "min_rounds": self.min_rounds,
            "rounds": [r.as_dict() for r in self.rounds],
        }

    def __str__(self) -> str:
        head = (
            f"passgate: {sum(r.meets for r in self.rounds)}/{len(self.rounds)} round(s) "
            f"clear Wilson-lower >= {self.threshold:.3f} "
            f"({self.confidence:.0%} confidence, min {self.min_rounds} rounds)"
        )
        lines = [
            f"  {'PASS' if r.meets else 'FAIL'} "
            f"{(r.round.label or 'round ' + str(i + 1)):>12}: "
            f"{r.round.passes}/{r.round.trials} = {r.round.rate:.3f} "
            f"(lower {r.wilson_lower:.3f})"
            for i, r in enumerate(self.rounds)
        ]
        verdict = "PASS" if self.passed else "FAIL"
        return "\n".join([head] + lines + [f"verdict: {verdict}"])


def evaluate_rounds(
    rounds: Sequence[Round],
    threshold: float = 0.9,
    confidence: float = 0.95,
    min_rounds: int = 1,
) -> GateResult:
    """Gate: EVERY round's Wilson lower bound must clear ``threshold``.

    A flaky candidate that spikes on average but has one weak round fails —
    the "pass^N" property.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if min_rounds < 1:
        raise ValueError("min_rounds must be >= 1")
    result = GateResult(threshold=threshold, confidence=confidence, min_rounds=min_rounds)
    for r in rounds:
        lower = wilson_lower(r.passes, r.trials, confidence)
        result.rounds.append(RoundEval(round=r, wilson_lower=lower, meets=lower >= threshold))
    return result


@dataclass
class Comparison:
    candidate_rate: float
    baseline_rate: float
    z: float
    p_one_sided: float
    fisher_p: float
    alpha: float

    @property
    def better(self) -> bool:
        return self.candidate_rate > self.baseline_rate and self.p_one_sided < self.alpha

    @property
    def verdict(self) -> str:
        if self.p_one_sided < self.alpha:
            return "candidate-better" if self.candidate_rate > self.baseline_rate else "baseline-better"
        return "inconclusive"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "candidate_rate": round(self.candidate_rate, 6),
            "baseline_rate": round(self.baseline_rate, 6),
            "z": round(self.z, 6),
            "p_one_sided": round(self.p_one_sided, 6),
            "fisher_p": round(self.fisher_p, 6),
            "alpha": self.alpha,
            "verdict": self.verdict,
        }

    def __str__(self) -> str:
        return (
            f"candidate {self.candidate_rate:.3f} vs baseline {self.baseline_rate:.3f} — "
            f"z={self.z:.3f}, one-sided p={self.p_one_sided:.4g}, "
            f"fisher p={self.fisher_p:.4g}, verdict: {self.verdict}"
        )


def compare(
    candidate: Sequence[Round],
    baseline: Sequence[Round],
    alpha: float = 0.05,
) -> Comparison:
    """Compare candidate vs baseline on pooled totals, with Fisher as a check."""
    c1 = sum(r.passes for r in candidate)
    n1 = sum(r.trials for r in candidate)
    c2 = sum(r.passes for r in baseline)
    n2 = sum(r.trials for r in baseline)
    z, p = two_proportion_z(c1, n1, c2, n2)
    fisher = fisher_exact_2x2(c1, n1 - c1, c2, n2 - c2)
    return Comparison(
        candidate_rate=c1 / n1,
        baseline_rate=c2 / n2,
        z=z,
        p_one_sided=p,
        fisher_p=fisher,
        alpha=alpha,
    )
