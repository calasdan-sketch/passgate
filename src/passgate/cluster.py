"""Item-level (clustered) pass rates and pass^k, for evals that run each item several times.

When 50 items are each sampled 10 times, the 500 trials are NOT 500 independent coin flips: an item
the model "mostly knows" passes almost every time and a hard one almost never does. Treating the
trials as independent (what a plain Wilson bound on the pooled counts does) makes the interval far too
narrow. Here the unit of independence is the item:

* the pooled rate gets a cluster-robust standard error (ratio estimator, items as clusters);
* the interval is Wilson's, evaluated at the *effective* sample size ``n_eff = n / DEFF``,
  with a Student-t critical value on ``items - 1`` degrees of freedom;
* ``pass^k`` (every one of k tries succeeds, tau-bench) and ``pass@k`` (at least one of k succeeds)
  use the unbiased per-item estimators ``C(c,k)/C(m,k)`` and ``1 - C(m-c,k)/C(m,k)`` (Chen et al. 2021).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

from .safe import md_cell, safe_text
from .stats import _wilson, t_ppf, z_for_confidence

MAX_ITEMS = 5_000_000


@dataclass(frozen=True)
class Item:
    """One test item: how many times it was tried and how many tries passed."""

    trials: int
    passes: int
    name: str = ""

    def __post_init__(self) -> None:
        for field_name in ("trials", "passes"):
            v = getattr(self, field_name)
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(f"{field_name} must be a whole number, got {v!r}")
        if self.trials <= 0:
            raise ValueError("trials must be positive")
        if not 0 <= self.passes <= self.trials:
            raise ValueError(f"passes {self.passes} outside [0, {self.trials}]")


def _check_k(trials: int, passes: int, k: int) -> None:
    if k < 1 or trials < k:
        raise ValueError("need trials >= k >= 1")
    if not 0 <= passes <= trials:
        raise ValueError(f"passes {passes} outside [0, {trials}]")


def pass_hat_k(trials: int, passes: int, k: int) -> float:
    """Unbiased estimate that k independent tries ALL pass, from ``passes`` of ``trials`` (pass^k).

    Equal to C(passes,k)/C(trials,k), computed as a running product so a huge trial count with a big k
    costs O(k) float steps instead of building enormous integers.
    """
    _check_k(trials, passes, k)
    if passes < k:
        return 0.0
    out = 1.0
    for i in range(k):
        out *= (passes - i) / (trials - i)
        if out == 0.0:
            break
    return out


def pass_at_k(trials: int, passes: int, k: int) -> float:
    """Unbiased estimate that at least one of k tries passes (pass@k, Chen et al. 2021).

    Equal to 1 - C(trials-passes,k)/C(trials,k). Evaluated as ``-expm1(sum(log1p(...)))`` so a tiny
    answer keeps its digits instead of being lost in ``1 - 0.999999999``.
    """
    _check_k(trials, passes, k)
    fails = trials - passes
    if fails < k:
        return 1.0
    if passes == 0:
        return 0.0
    total = 0.0
    for i in range(k):
        total += math.log1p(-passes / (trials - i))
    return min(1.0, max(0.0, -math.expm1(total)))


@dataclass
class ItemsResult:
    """Cluster-aware pass rate / pass^k with a lower bound and a gate verdict."""

    items: int
    trials: int
    passes: int
    metric: str  # "pass-rate", "pass^k", "pass@k"
    k: int
    estimate: float
    lower: float
    upper: float
    threshold: float
    confidence: float
    deff: float  # design effect (1.0 = trials behave independently)
    naive_lower: float  # plain Wilson on the pooled counts, shown for contrast
    skipped_items: int = 0  # items with fewer than k trials (pass^k / pass@k only)
    worst: List[Tuple[str, int, int]] = field(default_factory=list)  # (name, passes, trials), weakest first

    @property
    def passed(self) -> bool:
        return self.lower >= self.threshold

    def as_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "metric": self.metric,
            "k": self.k,
            "items": self.items,
            "trials": self.trials,
            "passes": self.passes,
            "estimate": round(self.estimate, 6),
            "lower": round(self.lower, 6),
            "upper": round(self.upper, 6),
            "naive_pooled_wilson_lower": round(self.naive_lower, 6),
            "design_effect": round(self.deff, 4),
            "threshold": self.threshold,
            "confidence": self.confidence,
            "skipped_items": self.skipped_items,
            "weakest_items": [{"item": n, "passes": p, "trials": t} for n, p, t in self.worst],
        }

    def __str__(self) -> str:
        lines = [
            f"items: {self.metric} = {self.estimate:.3f} over {self.items} item(s), {self.trials} trial(s); "
            f"{self.confidence:.0%} cluster-aware lower bound {self.lower:.3f} vs bar {self.threshold:.3f}",
            f"  design effect {self.deff:.2f} (1.00 = trials independent); a plain pooled Wilson bound would "
            f"claim {self.naive_lower:.3f}"
            if self.k == 1
            else f"  per-item unbiased {self.metric} estimates (k={self.k}), Wilson interval over items at an effective sample size",
        ]
        if self.skipped_items:
            lines.append(f"  {self.skipped_items} item(s) with fewer than k={self.k} trials were left out")
        for name, p, t in self.worst:
            lines.append(f"  weakest: {safe_text(name, 40)} {p}/{t}")
        lines.append(f"verdict: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)

    def explain(self) -> str:
        what = {
            "pass-rate": "pass rate",
            "pass^k": f"chance that all {self.k} tries in a row succeed",
            "pass@k": f"chance that at least one of {self.k} tries succeeds",
        }[self.metric]
        s = (
            f"Across {self.items} test items the {what} is {self.estimate:.1%}. Because tries on the same item "
            f"are alike, they carry less information than the same number of independent tries "
            f"(design effect {self.deff:.1f}). Allowing for that, we are {self.confidence:.0%} sure the true value "
            f"is at least {self.lower:.1%}"
        )
        if self.passed:
            return s + f", which clears your bar of {self.threshold:.0%}."
        gap = self.k == 1 and self.naive_lower >= self.threshold
        return s + f", short of your bar of {self.threshold:.0%}." + (
            " A naive calculation that treats every try as independent would have wrongly passed this." if gap else ""
        )

    def to_markdown(self) -> str:
        return (
            f"### passgate items: {'PASS' if self.passed else 'FAIL'}\n\n"
            "| metric | items | trials | estimate | lower bound | bar | design effect | naive lower |\n"
            "|---|---|---|---|---|---|---|---|\n"
            f"| {md_cell(self.metric)} | {self.items} | {self.trials} | {self.estimate:.3f} | {self.lower:.3f} | "
            f"{self.threshold:g} | {self.deff:.2f} | {self.naive_lower:.3f} |\n\n{self.explain()}\n"
        )


def _crit(confidence: float, df: int) -> float:
    q = 0.5 + confidence / 2.0
    return t_ppf(q, df) if df >= 1 else z_for_confidence(confidence)


def cluster_wilson(items: Sequence[Item], confidence: float = 0.95) -> Tuple[float, float, float, float, float]:
    """Cluster-aware Wilson interval for the pooled pass rate.

    Returns ``(rate, lower, upper, deff, naive_lower)``. ``n_eff`` is floored at the number of items
    (each item counts for at least one independent observation) and capped at the total trials
    (the design effect is never allowed below 1, so this is never less cautious than plain Wilson).
    """
    g = len(items)
    if g == 0:
        raise ValueError("need at least one item")
    m_tot = sum(i.trials for i in items)
    k_tot = sum(i.passes for i in items)
    p = k_tot / m_tot
    crit = _crit(confidence, g - 1)
    naive_lower = _wilson(p, m_tot, z_for_confidence(confidence))[0]
    if g == 1:
        n_eff = 1.0
    else:
        ss = sum((i.passes - p * i.trials) ** 2 for i in items)
        var = (g / (g - 1.0)) * ss / (m_tot * m_tot)
        if var > 0.0 and 0.0 < p < 1.0:
            n_eff = p * (1.0 - p) / var
        else:
            n_eff = float(g)  # degenerate (all items identical or p at 0/1): fall back to one observation per item
        n_eff = min(float(m_tot), max(float(g), n_eff))
    lo, hi = _wilson(p, n_eff, crit)
    return p, lo, hi, m_tot / n_eff, naive_lower


def _mean_t_interval(vals: Sequence[float], confidence: float, max_neff: float) -> Tuple[float, float, float]:
    """Interval for the mean of per-item values in [0, 1].

    A plain Student-t interval collapses to zero width when every item has the same value (three items
    that all pass 10 of 10 would "prove" pass^2 = 1.000). So, like the k = 1 path, the interval is
    Wilson's evaluated at an effective sample size taken from the spread of the item values, never fewer
    than one observation per item and never more than ``max_neff`` (the usable trials).
    """
    g = len(vals)
    mean = sum(vals) / g
    if g < 2:
        return mean, 0.0, 1.0
    var_of_mean = sum((v - mean) ** 2 for v in vals) / (g - 1) / g
    if var_of_mean > 0.0 and 0.0 < mean < 1.0:
        n_eff = mean * (1.0 - mean) / var_of_mean
    else:
        n_eff = float(g)
    n_eff = min(max_neff, max(float(g), n_eff))
    lo, hi = _wilson(mean, n_eff, _crit(confidence, g - 1))
    return mean, lo, hi


def evaluate_items(
    items: Sequence[Item],
    threshold: float = 0.9,
    confidence: float = 0.95,
    k: int = 1,
    metric: str = "pass^k",
) -> ItemsResult:
    """Gate on an item-clustered pass rate (k = 1) or on pass^k / pass@k (k >= 2).

    For ``k == 1`` the bound is the cluster-aware Wilson interval. For ``k >= 2`` it is a Wilson interval, with a
    Student-t critical value, over per-item unbiased estimates (approximate; items with fewer than k trials are left out
    and counted). Fails closed: zero usable items raises ``ValueError``.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    z_for_confidence(confidence)
    if k < 1:
        raise ValueError("k must be >= 1")
    if metric not in ("pass^k", "pass@k"):
        raise ValueError("metric must be 'pass^k' or 'pass@k'")
    if not items:
        raise ValueError("no items to evaluate")
    if len(items) > MAX_ITEMS:
        raise ValueError(f"too many items (limit {MAX_ITEMS:,})")
    trials = sum(i.trials for i in items)
    passes = sum(i.passes for i in items)
    weakest = sorted(items, key=lambda i: (i.passes / i.trials, -i.trials))[:3]
    worst = [(i.name or "(unnamed)", i.passes, i.trials) for i in weakest if i.passes < i.trials]
    if k == 1:
        p, lo, hi, deff, naive = cluster_wilson(items, confidence)
        return ItemsResult(len(items), trials, passes, "pass-rate", 1, p, lo, hi, threshold, confidence, deff, naive, 0, worst)
    usable = [i for i in items if i.trials >= k]
    if not usable:
        raise ValueError(f"no item has at least k={k} trials")
    fn = pass_hat_k if metric == "pass^k" else pass_at_k
    vals = [fn(i.trials, i.passes, k) for i in usable]
    est, lo, hi = _mean_t_interval(vals, confidence, float(sum(i.trials for i in usable)))
    naive = _wilson(est, sum(i.trials for i in usable), z_for_confidence(confidence))[0]
    return ItemsResult(
        len(usable), sum(i.trials for i in usable), sum(i.passes for i in usable), metric, k, est, lo, hi,
        threshold, confidence, 1.0, naive, len(items) - len(usable), worst,
    )
