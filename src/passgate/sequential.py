"""Always-valid ("peek as often as you like") pass-rate certification.

A fixed-size gate (``gate``) is valid only if you look once, after the planned number of trials.
Checking a Wilson bound after every trial and stopping when it clears is a classic mistake: the
false-pass rate balloons far above the nominal level. This module implements a *mixture
likelihood-ratio test* whose Type I error is controlled over **all** stopping times (Ville's
inequality; see Robbins 1970, Howard et al. 2021 "Time-uniform, nonparametric, nonasymptotic
confidence sequences"). It lets a run stop the moment the evidence is conclusive.

Test of ``H0: p <= theta``. With ``k`` passes in ``n`` trials, the mixture over alternatives
``p ~ Uniform(theta, 1]`` is

    M(k, n) = [ integral_theta^1 p^k (1-p)^(n-k) dp / (1-theta) ] / ( theta^k (1-theta)^(n-k) )

``M`` is a supermartingale under every ``p <= theta``, so ``P(M ever >= 1/alpha) <= alpha``.
Certify "pass" the first time ``M >= 1/alpha``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .stats import betainc

__all__ = ["log_evidence", "pass_boundary", "anytime_lower", "SequentialResult", "run_sequential"]


def log_evidence(passes: int, trials: int, theta: float) -> float:
    """log M(k, n) for H0: p <= theta against a uniform mixture on (theta, 1]. -inf when numerically zero."""
    k, n = passes, trials
    if not 0.0 < theta < 1.0:
        raise ValueError("theta must be between 0 and 1")
    if n < 0 or not 0 <= k <= n:
        raise ValueError("passes must be within [0, trials]")
    tail = betainc(n - k + 1.0, k + 1.0, 1.0 - theta)  # = integral_theta^1 posterior mass
    if tail <= 0.0:
        return -math.inf
    log_beta = math.lgamma(k + 1.0) + math.lgamma(n - k + 1.0) - math.lgamma(n + 2.0)
    return log_beta + math.log(tail) - math.log1p(-theta) - k * math.log(theta) - (n - k) * math.log1p(-theta)


def pass_boundary(theta: float, alpha: float, max_trials: int) -> List[Optional[int]]:
    """``out[n]`` = fewest passes in ``n`` trials that certify ``p > theta`` at any-time level ``alpha``.

    ``None`` when even n/n passes is not enough yet. Index 0 is unused (None). Evidence is monotone
    in the pass count, so each entry is a binary search.
    """
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")
    if max_trials < 1:
        raise ValueError("max_trials must be >= 1")
    target = math.log(1.0 / alpha)
    out: List[Optional[int]] = [None] * (max_trials + 1)
    lo_start = 0
    for n in range(1, max_trials + 1):
        if log_evidence(n, n, theta) < target:
            continue
        lo, hi = lo_start, n
        while lo < hi:
            mid = (lo + hi) // 2
            if log_evidence(mid, n, theta) >= target:
                hi = mid
            else:
                lo = mid + 1
        out[n] = lo
        lo_start = max(0, lo - 1)  # boundary k*(n) never drops by more than 1 as n grows by 1
    return out


def anytime_lower(passes: int, trials: int, alpha: float = 0.05) -> float:
    """Time-uniform lower confidence bound for the pass rate (holds simultaneously at every n).

    The largest ``theta`` that the mixture test still rejects. Wider than a fixed-n Wilson bound;
    that gap is the price of being allowed to look after every trial.
    """
    if trials <= 0:
        return 0.0
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")
    if not 0 <= passes <= trials:
        raise ValueError("passes must be within [0, trials]")
    target = math.log(1.0 / alpha)
    lo, hi = 1e-12, 1.0 - 1e-12
    if log_evidence(passes, trials, lo) < target:
        return 0.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if log_evidence(passes, trials, mid) >= target:
            lo = mid
        else:
            hi = mid
    return lo


@dataclass
class SequentialResult:
    """Outcome of replaying a stream of pass/fail outcomes through the any-time test."""

    decision: str  # "pass", "fail", "undecided"
    reason: str
    trials: int
    passes: int
    threshold: float
    alpha: float
    max_trials: int
    lower: float  # time-uniform lower bound at the stopping point

    @property
    def passed(self) -> bool:
        return self.decision == "pass"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "trials_used": self.trials,
            "passes": self.passes,
            "threshold": self.threshold,
            "alpha": self.alpha,
            "max_trials": self.max_trials,
            "anytime_lower": round(self.lower, 6),
            "passed": self.passed,
        }

    def __str__(self) -> str:
        return (
            f"sequential: {self.decision.upper()} after {self.trials} trial(s) "
            f"({self.passes} passes), any-time lower bound {self.lower:.3f} vs bar {self.threshold:.3f} "
            f"- {self.reason}"
        )

    def explain(self) -> str:
        if self.decision == "pass":
            return (
                f"After {self.trials} tries ({self.passes} passed) we can already say, with the error rate "
                f"you chose ({self.alpha:.0%}) even though we peeked after every try, that the true success rate "
                f"is above {self.threshold:.0%}. Safe to stop and ship."
            )
        if self.decision == "fail":
            return (
                f"After {self.trials} tries ({self.passes} passed) it is no longer possible to prove the "
                f"success rate is above {self.threshold:.0%} within the {self.max_trials}-try budget. Stop and fix the candidate."
            )
        return (
            f"The stream ended after {self.trials} tries ({self.passes} passed) before the evidence was "
            "conclusive either way. Treated as not passing; run more trials."
        )

    def to_markdown(self) -> str:
        return (
            f"### passgate sequential: {self.decision.upper()}\n\n"
            f"{self.trials} trial(s), {self.passes} passes; any-time lower bound {self.lower:.3f}; "
            f"bar {self.threshold:g}; alpha {self.alpha:g}.\n\n{self.explain()}\n"
        )


def run_sequential(
    outcomes: Iterable[bool], threshold: float = 0.9, alpha: float = 0.025, max_trials: int = 1000
) -> SequentialResult:
    """Replay ``outcomes`` in order and stop at the first conclusive point.

    Stops with ``pass`` when the any-time test certifies p > threshold. Stops with ``fail`` as soon as
    certification is impossible inside ``max_trials`` (even if every remaining trial passed). If the
    stream ends first the decision is ``undecided`` (never a pass). Uses at most ``max_trials`` outcomes.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if max_trials < 1 or max_trials > 100_000:
        raise ValueError("max_trials must be between 1 and 100000")
    table = pass_boundary(threshold, alpha, max_trials)
    floor = _futility_floor(table, max_trials)
    k = n = 0
    for o in outcomes:
        if not isinstance(o, bool):
            raise ValueError("outcomes must be booleans")
        n += 1
        k += o
        need = table[n]
        if need is not None and k >= need:
            return SequentialResult("pass", "evidence crossed the any-time bound", n, k, threshold, alpha, max_trials, anytime_lower(k, n, alpha))
        if k - n < floor[n]:
            return SequentialResult("fail", "cannot be certified within the trial budget", n, k, threshold, alpha, max_trials, anytime_lower(k, n, alpha))
        if n >= max_trials:
            return SequentialResult("fail", "trial budget used up without certification", n, k, threshold, alpha, max_trials, anytime_lower(k, n, alpha))
    return SequentialResult("undecided", "outcome stream ended first", n, k, threshold, alpha, max_trials, anytime_lower(k, n, alpha))


def _futility_floor(table: List[Optional[int]], max_trials: int) -> List[float]:
    """floor[n] = min over m > n of (table[m] - m); (k, n) can still certify iff k - n >= floor[n]."""
    floor = [math.inf] * (max_trials + 2)
    best = math.inf
    for n in range(max_trials, -1, -1):
        floor[n] = best
        if n >= 1 and table[n] is not None:
            best = min(best, table[n] - n)  # type: ignore[operator]
    return floor
