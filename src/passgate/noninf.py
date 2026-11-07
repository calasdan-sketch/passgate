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


# --- sample-size planning for `regress` (added in 0.3.0) ---

def _log_pmf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return (
        math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)
        + k * math.log(p) + (n - k) * math.log1p(-p)
    )


def _min_cand_passes(n: int, kb: int, margin: float, conf: float) -> Optional[int]:
    """Fewest candidate passes (out of n) that make `regress` pass against kb baseline passes (None: impossible)."""
    def ok(kc: int) -> bool:
        return newcombe_diff(kc, n, kb, n, conf)[0] > -margin
    if not ok(n):
        return None
    lo, hi = 0, n
    while lo < hi:  # the lower end of the interval rises with kc
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def regress_power(
    trials_per_arm: int, candidate_rate: float, baseline_rate: float, margin: float, alpha: float = 0.05
) -> float:
    """Exact probability that `regress` (independent samples, equal arm sizes) passes at these true rates.

    Sums the binomial probabilities of every (candidate passes, baseline passes) outcome for which the
    Newcombe interval clears -margin. Outcomes with probability below about 1e-13 are skipped.
    """
    from .stats import binom_sf

    _check_alpha(alpha)
    _check_margin(margin)
    n = int(trials_per_arm)
    if n < 1:
        raise ValueError("trials_per_arm must be positive")
    for r in (candidate_rate, baseline_rate):
        if not 0.0 <= r <= 1.0:
            raise ValueError("rates must be between 0 and 1")
    conf = 1 - 2 * alpha
    sd = math.sqrt(max(baseline_rate * (1 - baseline_rate), 1e-12) * n)
    lo_b = max(0, int(n * baseline_rate - 9 * sd) - 1)
    hi_b = min(n, int(n * baseline_rate + 9 * sd) + 2)
    total = 0.0
    for kb in range(lo_b, hi_b + 1):
        w = _log_pmf(kb, n, baseline_rate)
        if w < -32.0:
            continue
        kmin = _min_cand_passes(n, kb, margin, conf)
        if kmin is not None:
            total += math.exp(w) * binom_sf(kmin, n, candidate_rate)
    return min(1.0, total)


def plan_noninferiority(
    candidate_rate: float,
    baseline_rate: float,
    margin: float,
    power: float = 0.8,
    alpha: float = 0.05,
    max_trials: int = 20_000,
) -> Optional[Dict[str, Any]]:
    """Trials per arm so that `regress` (independent samples) passes with probability >= ``power``.

    ``candidate_rate`` / ``baseline_rate`` are the true pass rates you assume. Returns None when the
    candidate's assumed rate is itself ``margin`` or more below the baseline (no sample size helps) or
    the size would exceed ``max_trials`` per arm. The power is a sawtooth in the size, so the answer is
    found by a geometric search then bisection: it is a size that reaches the target, and a slightly
    smaller one may too. ``power_at_trials`` is the exact power at the returned size.
    """
    if not 0.0 < power < 1.0:
        raise ValueError("power must be between 0 and 1")
    _check_alpha(alpha)
    _check_margin(margin)
    for r in (candidate_rate, baseline_rate):
        if not 0.0 <= r <= 1.0:
            raise ValueError("rates must be between 0 and 1")
    if candidate_rate - baseline_rate <= -margin:
        return None

    def ok(m: int) -> bool:
        return regress_power(m, candidate_rate, baseline_rate, margin, alpha) >= power

    lo, hi = 1, 8
    while not ok(hi):
        lo, hi = hi, int(hi * 1.25) + 1
        if hi > max_trials:
            return None
    while lo + 1 < hi:  # lo fails, hi passes
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return {
        "trials_per_arm": hi,
        "total_trials": 2 * hi,
        "power_at_trials": regress_power(hi, candidate_rate, baseline_rate, margin, alpha),
        "candidate_rate": candidate_rate,
        "baseline_rate": baseline_rate,
        "margin": margin,
        "alpha": alpha,
        "power": power,
    }


def min_provable_margin(
    trials_per_arm: int, rate: float, power: float = 0.8, alpha: float = 0.05, candidate_rate: Optional[float] = None
) -> Optional[float]:
    """Smallest margin ``regress`` can prove with ``trials_per_arm`` trials per arm (independent samples).

    The answer to "I can only afford N trials: how small a drop can this check rule out?". ``rate`` is the
    baseline's true pass rate; the candidate is assumed equal unless ``candidate_rate`` is given. Power rises with
    the margin, so this bisects it. Returns None when even a margin of 0.5 cannot be proved with that power.
    """
    if not 0.0 < power < 1.0:
        raise ValueError("power must be between 0 and 1")
    cand = rate if candidate_rate is None else candidate_rate

    def ok(m: float) -> bool:
        return regress_power(trials_per_arm, cand, rate, m, alpha) >= power

    lo, hi = 1e-4, 0.5
    if not ok(hi):
        return None
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi
