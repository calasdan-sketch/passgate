"""Small, dependency-free statistics used by the gate.

All bounds are exact closed forms; ``z`` comes from the standard normal
quantile (``statistics.NormalDist``), so results match any reference
implementation to floating-point precision.
"""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import Tuple

_NORMAL = NormalDist()


def z_for_confidence(confidence: float) -> float:
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1.0")
    return _NORMAL.inv_cdf(0.5 + confidence / 2.0)


def wilson_interval(passes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= passes <= trials:
        raise ValueError("passes must be within [0, trials]")
    z = z_for_confidence(confidence)
    phat = passes / trials
    denom = 1.0 + z * z / trials
    center = (phat + z * z / (2 * trials)) / denom
    half = (z * math.sqrt((phat * (1 - phat) + z * z / (4 * trials)) / trials)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def wilson_lower(passes: int, trials: int, confidence: float = 0.95) -> float:
    """Lower bound of the Wilson interval — the conservative pass-rate estimate."""
    return wilson_interval(passes, trials, confidence)[0]


def two_proportion_z(c1: int, n1: int, c2: int, n2: int) -> Tuple[float, float]:
    """Pooled two-proportion z-test: candidate (c1/n1) vs baseline (c2/n2).

    Returns (z, one_sided_p). Positive z means the candidate is better.
    """
    if n1 <= 0 or n2 <= 0:
        raise ValueError("sample sizes must be positive")
    if not (0 <= c1 <= n1 and 0 <= c2 <= n2):
        raise ValueError("passes must be within [0, trials]")
    p1, p2 = c1 / n1, c2 / n2
    pooled = (c1 + c2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0.0:
        return (0.0, 1.0) if p1 == p2 else (math.inf if p1 > p2 else -math.inf, 0.0)
    z = (p1 - p2) / se
    return z, 1.0 - _NORMAL.cdf(z)


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher's exact p-value for [[a, b], [c, d]].

    Exact hypergeometric enumeration — right for small eval rounds where the
    normal approximation lies.
    """
    for v in (a, b, c, d):
        if v < 0:
            raise ValueError("counts must be non-negative")
    n = a + b + c + d
    row1, col1 = a + b, a + c

    def prob(x: int) -> float:
        return math.comb(col1, x) * math.comb(n - col1, row1 - x) / math.comb(n, row1)

    lo = max(0, row1 + col1 - n)
    hi = min(row1, col1)
    observed = prob(a)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= observed + 1e-12))
