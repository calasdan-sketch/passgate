"""Power-aware planning: how likely is a round (or N rounds) to clear the gate at a given true rate?

``stats.trials_needed`` returns the *break-even* size: a round that passes at exactly
``rate`` clears only about half the time. This module answers the question teams
actually have: "how many trials until the gate passes a genuinely good candidate
with probability >= ``power``?" All numbers are exact binomial probabilities.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .stats import (
    EXACT_MAX_TRIALS,
    METHODS,
    binom_sf,
    min_passes,
    trials_needed,
    z_for_confidence,
)

EXACT_SCAN = 20_000  # sizes up to this are scanned one by one, so 'trials' is the true minimum


def pass_probability(
    trials: int, rate: float, threshold: float, confidence: float = 0.95, method: str = "wilson"
) -> float:
    """Exact probability that one round of ``trials`` at true pass rate ``rate`` clears the gate."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be between 0 and 1")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if method not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)}")
    z_for_confidence(confidence)
    k = min_passes(trials, threshold, confidence, method)
    if k is None:
        return 0.0
    return binom_sf(k, trials, rate)


def oc_curve(
    trials: int,
    threshold: float,
    rates: Optional[Sequence[float]] = None,
    confidence: float = 0.95,
    method: str = "wilson",
    rounds: int = 1,
) -> List[Dict[str, float]]:
    """Operating-characteristic curve: P(gate passes) as a function of the true rate.

    With ``rounds`` > 1 every round must clear, so the probability is raised to that power.
    """
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    if rates is None:
        rates = [round(x / 100, 2) for x in range(50, 101, 2)]
    return [
        {"rate": r, "pass_probability": pass_probability(trials, r, threshold, confidence, method) ** rounds}
        for r in rates
    ]


def plan_power(
    rate: float,
    threshold: float,
    power: float = 0.8,
    confidence: float = 0.95,
    method: str = "wilson",
    rounds: int = 1,
    max_trials: int = 200_000,
) -> Optional[Dict[str, Any]]:
    """Smallest round size at which a candidate with true rate ``rate`` passes with probability >= ``power``.

    ``power`` is the probability that ALL ``rounds`` clear, so each round needs ``power ** (1/rounds)``.
    The pass probability is a sawtooth in the trial count (integer pass thresholds), so two sizes are
    returned: ``trials`` is the first size that reaches the target and ``stable_trials`` is the first
    size from which every larger size scanned also reaches it. Returns None when ``rate <= threshold``
    or the target cannot be met within ``max_trials`` (exact method: capped at the exact-method limit).
    """
    if not 0.0 < power < 1.0:
        raise ValueError("power must be between 0 and 1")
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    if method not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)}")
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be between 0 and 1")
    if rate <= threshold:
        return None
    cap = min(max_trials, EXACT_MAX_TRIALS) if method == "exact" else max_trials
    per_round = power ** (1.0 / rounds)
    be = trials_needed(rate, threshold, confidence, method, max_trials=cap)
    if be is None:
        return None
    n = be[0]

    def ok(m: int) -> bool:
        return pass_probability(m, rate, threshold, confidence, method) >= per_round

    first = None
    for m in range(1, min(cap, EXACT_SCAN) + 1):  # exact minimum: every size is checked
        if ok(m):
            first = m
            break
    if first is None:  # beyond the exhaustive range: geometric search, then walk back one growth step
        hi = max(n, EXACT_SCAN)
        while not ok(hi):
            hi = int(hi * 1.15) + 1
            if hi > cap:
                return None
        first = hi
        for m in range(max(EXACT_SCAN + 1, int(hi / 1.15) - 1), hi + 1):
            if ok(m):
                first = m
                break
    # stability: scan forward a window and find the last miss
    end = min(cap, max(first + 200, int(first * 1.3)))
    last_miss = first - 1
    for m in range(first, end + 1):
        if not ok(m):
            last_miss = m
    stable = last_miss + 1
    k = min_passes(first, threshold, confidence, method)
    return {
        "trials": first,
        "min_passes": k,
        "stable_trials": stable,
        "power_at_trials": pass_probability(first, rate, threshold, confidence, method) ** rounds,
        "break_even_trials": n,
        "per_round_power": per_round,
        "rounds": rounds,
        "total_trials": first * rounds,
    }
