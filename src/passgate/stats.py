"""Small, dependency-free statistics used by the gate.

Wilson bounds are closed forms; Clopper-Pearson ("exact") bounds invert the
regularized incomplete beta function by bisection; Fisher and McNemar use
exact integer arithmetic for small tables and log-space sums for large ones.
``z`` comes from the standard normal quantile (``statistics.NormalDist``).
"""
from __future__ import annotations

import math
import time
from functools import lru_cache
from statistics import NormalDist
from typing import Optional, Tuple

_NORMAL = NormalDist()
METHODS = ("wilson", "exact")
_EXACT_INT_LIMIT = 5000  # tables up to this size use exact big-integer arithmetic
# Above this many trials the exact methods (Fisher, McNemar, Clopper-Pearson) are refused:
# lgamma-based sums lose precision (about 1e-7 relative at 1e8, 1e-6 at 1e9) and Fisher is O(sqrt n).
EXACT_MAX_TRIALS = 10**8
FISHER_MAX_TRIALS = EXACT_MAX_TRIALS


def z_for_confidence(confidence: float) -> float:
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1.0")
    return _NORMAL.inv_cdf(0.5 + confidence / 2.0)


def _check_counts(passes: int, trials: int) -> None:
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= passes <= trials:
        raise ValueError("passes must be within [0, trials]")


def _wilson(phat: float, trials: float, z: float) -> Tuple[float, float]:
    denom = 1.0 + z * z / trials
    center = (phat + z * z / (2 * trials)) / denom
    half = (z * math.sqrt(max(0.0, phat * (1 - phat) + z * z / (4 * trials)) / trials)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def wilson_interval(passes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    _check_counts(passes, trials)
    return _wilson(passes / trials, trials, z_for_confidence(confidence))


def wilson_lower(passes: int, trials: int, confidence: float = 0.95) -> float:
    """Lower bound of the Wilson interval - the conservative pass-rate estimate."""
    return wilson_interval(passes, trials, confidence)[0]


# --- regularized incomplete beta (Numerical Recipes continued fraction, Lentz) ---

def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 100_000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            return h
    raise ValueError("counts too large for the exact method; use --method wilson")  # pragma: no cover


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) for a, b > 0."""
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return min(1.0, max(0.0, bt * _betacf(a, b, x) / a))
    return min(1.0, max(0.0, 1.0 - bt * _betacf(b, a, 1.0 - x) / b))


def _bisect_increasing(f, target: float) -> float:
    """Solve f(p) = target on [0, 1] for increasing f with f(0) <= target <= f(1).

    Illinois (bracketed regula falsi): superlinear like the secant method, but
    the root always stays bracketed, so it cannot diverge.
    """
    lo, hi = 0.0, 1.0
    flo, fhi = f(lo) - target, f(hi) - target
    side = 0
    for _ in range(200):
        if hi - lo < 1e-15:
            break
        x = (lo * fhi - hi * flo) / (fhi - flo) if fhi != flo else (lo + hi) / 2.0
        if not lo < x < hi:
            x = (lo + hi) / 2.0
        fx = f(x) - target
        if fx == 0.0:
            return x
        if fx < 0.0:
            lo, flo = x, fx
            if side == -1:
                fhi /= 2.0
            side = -1
        else:
            hi, fhi = x, fx
            if side == 1:
                flo /= 2.0
            side = 1
        if abs(fx) < 1e-17:
            break
    return lo if abs(flo) <= abs(fhi) else hi


@lru_cache(maxsize=4096)
def _cp_lower(k: float, n: float, alpha: float) -> float:
    if k <= 0:
        return 0.0
    if k >= n:
        return (alpha / 2.0) ** (1.0 / n)
    # P(X >= k | p) = I_p(k, n - k + 1), increasing in p
    return _bisect_increasing(lambda p: betainc(k, n - k + 1.0, p), alpha / 2.0)


@lru_cache(maxsize=4096)
def _cp_upper(k: float, n: float, alpha: float) -> float:
    if k >= n:
        return 1.0
    if k <= 0:
        return 1.0 - (alpha / 2.0) ** (1.0 / n)
    # P(X <= k | p) = 1 - I_p(k + 1, n - k), decreasing in p
    return _bisect_increasing(lambda p: betainc(k + 1.0, n - k, p), 1.0 - alpha / 2.0)


def clopper_pearson_interval(passes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Two-sided Clopper-Pearson ("exact") interval.

    Its coverage is never below nominal. It is usually wider than Wilson, but
    not always (near k = n at large n its lower bound can sit above Wilson's).
    """
    _check_counts(passes, trials)
    z_for_confidence(confidence)
    if trials > EXACT_MAX_TRIALS:
        raise ValueError(f"the exact method is limited to {EXACT_MAX_TRIALS:,} trials; use --method wilson")
    alpha = 1.0 - confidence
    return _cp_lower(passes, trials, alpha), _cp_upper(passes, trials, alpha)


def lower_bound(passes: int, trials: int, confidence: float = 0.95, method: str = "wilson") -> float:
    """Conservative pass-rate estimate under ``method`` ('wilson' or 'exact')."""
    if method == "wilson":
        return wilson_lower(passes, trials, confidence)
    if method == "exact":
        return clopper_pearson_interval(passes, trials, confidence)[0]
    raise ValueError(f"method must be one of {', '.join(METHODS)}")


def _lower_real(rate: float, trials: int, confidence: float, method: str) -> float:
    """Lower bound for a (possibly fractional) observed rate - used only for planning."""
    if method == "wilson":
        return _wilson(rate, trials, z_for_confidence(confidence))[0]
    return _cp_lower(rate * trials, trials, 1.0 - confidence)


def min_passes(trials: int, threshold: float, confidence: float = 0.95, method: str = "wilson") -> Optional[int]:
    """Fewest passes out of ``trials`` whose lower bound clears ``threshold`` (None if even all-pass fails)."""
    if lower_bound(trials, trials, confidence, method) < threshold:
        return None
    lo, hi = 0, trials
    while lo < hi:
        mid = (lo + hi) // 2
        if lower_bound(mid, trials, confidence, method) >= threshold:
            hi = mid
        else:
            lo = mid + 1
    return lo


def trials_needed(
    rate: float,
    threshold: float,
    confidence: float = 0.95,
    method: str = "wilson",
    max_trials: int = 10**8,
    deadline: Optional[float] = None,
) -> Optional[Tuple[int, int]]:
    """Smallest round size N at which a round passing at ``rate`` clears the gate.

    Returns ``(N, K)``: run N trials, the round clears if at least K pass, and
    K <= floor(rate * N). None when ``rate <= threshold`` (no sample size can
    certify it) or N would exceed ``max_trials``.

    Note: this is the break-even size. If the true rate is exactly ``rate``,
    a round of size N clears only about half the time - plan with headroom.

    ``deadline`` (a ``time.monotonic()`` value) makes the search raise ``TimeoutError`` once passed.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be between 0 and 1")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if method not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)}")
    z_for_confidence(confidence)
    if rate <= threshold:
        return None
    if method == "exact":
        max_trials = min(max_trials, EXACT_MAX_TRIALS)

    def ok(n: int) -> bool:
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError("trials_needed deadline passed")
        return _lower_real(rate, n, confidence, method) >= threshold

    hi = 1
    while not ok(hi):
        hi *= 2
        if hi > max_trials * 2:
            return None
    lo = hi // 2 + 1 if hi > 1 else 1
    while lo < hi:  # continuous-rate bound is monotone in n
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid + 1
    # integer passes: step up until floor(rate * n) passes is enough
    for n in range(lo, min(max_trials, lo + 2000) + 1):  # usually 1-3 steps
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError("trials_needed deadline passed")
        k = min_passes(n, threshold, confidence, method)
        if k is not None and k <= math.floor(rate * n + 1e-9):
            return n, k
    return None


# --- comparisons ---

def two_proportion_z(c1: int, n1: int, c2: int, n2: int) -> Tuple[float, float]:
    """Pooled two-proportion z-test: candidate (c1/n1) vs baseline (c2/n2).

    Returns (z, one_sided_p) for H1: candidate > baseline. Positive z means the
    candidate is better; the opposite one-sided p is ``1 - p`` (``NormalDist().cdf(z)``).
    """
    if n1 <= 0 or n2 <= 0:
        raise ValueError("sample sizes must be positive")
    if not (0 <= c1 <= n1 and 0 <= c2 <= n2):
        raise ValueError("passes must be within [0, trials]")
    p1, p2 = c1 / n1, c2 / n2
    pooled = (c1 + c2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0.0:
        return (0.0, 1.0) if p1 == p2 else (math.inf if p1 > p2 else -math.inf, 0.0 if p1 > p2 else 1.0)
    z = (p1 - p2) / se
    return z, _NORMAL.cdf(-z)


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_exact_2x2(a: int, b: int, c: int, d: int, alternative: str = "two-sided") -> float:
    """Fisher's exact p-value for [[a, b], [c, d]].

    ``alternative``: 'two-sided' (default), 'greater' (row 1 has a higher first-column
    rate, i.e. P(X >= a)) or 'less' (P(X <= a)). Exact hypergeometric enumeration -
    right for small eval rounds where the normal approximation lies.
    """
    for v in (a, b, c, d):
        if v < 0:
            raise ValueError("counts must be non-negative")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative must be 'two-sided', 'greater' or 'less'")
    n = a + b + c + d
    if n == 0:
        raise ValueError("table is empty")
    if n > FISHER_MAX_TRIALS:
        raise ValueError(f"Fisher's exact test is limited to {FISHER_MAX_TRIALS:,} total trials; use the z-test")
    row1, col1 = a + b, a + c
    lo, hi = max(0, row1 + col1 - n), min(row1, col1)
    if alternative == "greater":
        xs = range(a, hi + 1)
    elif alternative == "less":
        xs = range(lo, a + 1)
    else:
        xs = range(lo, hi + 1)

    if n <= _EXACT_INT_LIMIT:
        def w(x: int) -> int:
            return math.comb(col1, x) * math.comb(n - col1, row1 - x)

        total = math.comb(n, row1)
        if alternative == "two-sided":
            obs = w(a)
            return min(1.0, sum(v for v in (w(x) for x in xs) if v <= obs) / total)
        return min(1.0, sum(w(x) for x in xs) / total)

    # Large tables: hypergeometric terms are unimodal, so walk outward from each
    # starting point and stop once terms are negligible - O(sqrt(n)), not O(n).
    log_total = _log_comb(n, row1)

    def lp(x: int) -> float:
        return _log_comb(col1, x) + _log_comb(n - col1, row1 - x) - log_total

    mode = min(hi, max(lo, (row1 + 1) * (col1 + 1) // (n + 2)))

    def tail(start: int, step: int) -> float:
        """Sum of terms from ``start`` moving away from the mode (start must be on that side)."""
        total, x = 0.0, start
        while lo <= x <= hi:
            t = math.exp(lp(x))
            total += t
            if t < total * 1e-17 or t == 0.0:
                break
            x += step
        return total

    def upper_tail(x: int) -> float:  # P(X >= x)
        if x > hi:
            return 0.0
        if x <= lo:
            return 1.0
        return tail(x, 1) if x > mode else max(0.0, 1.0 - tail(x - 1, -1))

    def lower_tail(x: int) -> float:  # P(X <= x)
        if x < lo:
            return 0.0
        if x >= hi:
            return 1.0
        return tail(x, -1) if x < mode else max(0.0, 1.0 - tail(x + 1, 1))

    if alternative == "greater":
        return min(1.0, upper_tail(a))
    if alternative == "less":
        return min(1.0, lower_tail(a))
    cut = lp(a) + 1e-7  # relative tolerance for floating-point ties (as scipy does)
    # other side of the mode: first x whose term is <= the observed one (terms are monotone there)
    if a <= mode:
        same = lower_tail(a)
        left, right = mode, hi + 1
        while left < right:
            mid = (left + right) // 2
            if lp(mid) <= cut:
                right = mid
            else:
                left = mid + 1
        other = upper_tail(left) if left <= hi else 0.0
    else:
        same = upper_tail(a)
        left, right = lo - 1, mode
        while left < right:
            mid = (left + right + 1) // 2
            if lp(mid) <= cut:
                left = mid
            else:
                right = mid - 1
        other = lower_tail(left) if left >= lo else 0.0
    return min(1.0, same + other)


def binom_sf_half(k: int, m: int) -> float:
    """P(X >= k) for X ~ Binomial(m, 1/2)."""
    if k <= 0:
        return 1.0
    if k > m:
        return 0.0
    if m <= _EXACT_INT_LIMIT:
        term, total = math.comb(m, k), 0
        for x in range(k, m + 1):  # C(m, x+1) = C(m, x) * (m - x) / (x + 1), exact in integers
            total += term
            term = term * (m - x) // (x + 1)
        return total / (1 << m)
    return betainc(k, m - k + 1.0, 0.5)


def mcnemar_exact(cand_only: int, base_only: int) -> Tuple[float, float]:
    """Exact (binomial) McNemar test on the discordant pairs of a paired eval.

    ``cand_only``: items only the candidate passed; ``base_only``: items only the
    baseline passed. Returns (p_greater, p_less), one-sided p-values for
    H1: candidate better / baseline better.
    """
    if cand_only < 0 or base_only < 0:
        raise ValueError("counts must be non-negative")
    m = cand_only + base_only
    if m == 0:
        return 1.0, 1.0
    if m > EXACT_MAX_TRIALS:
        raise ValueError(f"exact McNemar is limited to {EXACT_MAX_TRIALS:,} discordant items")
    return binom_sf_half(cand_only, m), binom_sf_half(base_only, m)


# --- Student t and binomial tail helpers (added in 0.3.0) ---

def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), via the regularized incomplete beta."""
    if k <= 0:
        return 1.0
    if k > n or p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    return betainc(float(k), float(n - k + 1), p)


def t_cdf(t: float, df: float) -> float:
    """Student t cumulative distribution function (df > 0)."""
    if df <= 0:
        raise ValueError("df must be positive")
    if t == 0.0:
        return 0.5
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0 else tail


def t_ppf(q: float, df: float) -> float:
    """Student t quantile (inverse CDF) by bracketed bisection; df >= 1e6 uses the normal quantile."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    if df >= 1e6:
        return _NORMAL.inv_cdf(q)
    if q == 0.5:
        return 0.0
    if q < 0.5:
        return -t_ppf(1.0 - q, df)
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < q:
        hi *= 2.0
        if hi > 1e12:
            break
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13 * max(1.0, hi):
            break
    return (lo + hi) / 2.0
