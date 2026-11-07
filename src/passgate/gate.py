"""The pass^N gate and the candidate-vs-baseline comparisons."""
from __future__ import annotations

import math
import operator
import time
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .safe import md_cell, safe_text
from .stats import (
    FISHER_MAX_TRIALS,
    METHODS,
    fisher_exact_2x2,
    lower_bound,
    mcnemar_exact,
    trials_needed,
    two_proportion_z,
    wilson_lower,
    z_for_confidence,
)

_METHOD_NAME = {"wilson": "Wilson", "exact": "Clopper-Pearson"}
HINT_BUDGET = {"wilson": 1000, "exact": 25}  # failing rounds that get a "trials needed" hint
HINT_SECONDS = 2.0  # and a wall-clock cap on hint computation; hints never change the verdict


def _check_alpha(alpha: float) -> None:
    # alpha > 0.5 would let both one-sided tails fall under alpha, so a worse or equal
    # candidate could be called better (a common slip: typing 0.95 meaning confidence)
    if not 0.0 < alpha <= 0.5:
        raise ValueError("alpha must be in (0, 0.5], e.g. 0.05 (it is the false-positive rate, not the confidence)")


@dataclass(frozen=True)
class Round:
    """One evaluation round: how many trials, how many passed."""

    trials: int
    passes: int
    label: str = ""

    def __post_init__(self) -> None:
        for name in ("trials", "passes"):
            v = getattr(self, name)
            if isinstance(v, float) and v.is_integer():
                v = int(v)  # 93.0 is fine; 93.5 is not
            try:
                if isinstance(v, bool):
                    raise TypeError
                object.__setattr__(self, name, operator.index(v))  # accepts numpy ints too
            except TypeError:
                raise ValueError(f"{name} must be a whole number, got {v!r}") from None
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
    lower: float
    meets: bool
    method: str = "wilson"
    confidence: float = 0.95
    needed: Optional[Tuple[int, int]] = None  # (trials, passes) that would clear, for failing rounds
    hint_skipped: bool = False  # True when the hint budget ran out before this round

    @property
    def wilson_lower(self) -> float:
        """The Wilson lower bound (whatever ``method`` the gate used)."""
        if self.method == "wilson":
            return self.lower
        return wilson_lower(self.round.passes, self.round.trials, self.confidence)

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "label": self.round.label,
            "trials": self.round.trials,
            "passes": self.round.passes,
            "rate": round(self.round.rate, 6),
            "lower": round(self.lower, 6),
            "wilson_lower": round(self.wilson_lower, 6),
            "meets": self.meets,
        }
        if not self.meets:
            d["trials_needed"] = (
                {"trials": self.needed[0], "passes": self.needed[1]} if self.needed else None
            )
            if self.hint_skipped:
                d["trials_needed_skipped"] = True
        return d


@dataclass
class GateResult:
    threshold: float
    confidence: float
    min_rounds: int
    rounds: List[RoundEval] = field(default_factory=list)
    method: str = "wilson"

    @property
    def passed(self) -> bool:
        return len(self.rounds) >= max(1, self.min_rounds) and all(r.meets for r in self.rounds)

    def __bool__(self) -> bool:
        return self.passed

    def _name(self, i: int, r: RoundEval) -> str:
        return r.round.label or f"round {i + 1}"

    def _hint(self, r: RoundEval) -> str:
        if r.meets:
            return ""
        if r.hint_skipped:
            return "hint skipped (too many failing rounds)"
        if r.needed:
            return f"needs {r.needed[0]} trials at this rate ({r.needed[1]}+ passes)"
        if r.round.rate > self.threshold:
            return "needs more than 10,000,000 trials at this rate"
        return "rate is at or below the bar; more trials cannot help"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "threshold": self.threshold,
            "confidence": self.confidence,
            "method": self.method,
            "min_rounds": self.min_rounds,
            "rounds": [r.as_dict() for r in self.rounds],
        }

    def __str__(self) -> str:
        name = _METHOD_NAME[self.method]
        head = (
            f"passgate: {sum(r.meets for r in self.rounds)}/{len(self.rounds)} round(s) "
            f"clear {name}-lower >= {self.threshold:.3f} "
            f"({self.confidence:.0%} confidence, min {self.min_rounds} rounds)"
        )
        lines = []
        for i, r in enumerate(self.rounds):
            hint = self._hint(r)
            lines.append(
                f"  {'PASS' if r.meets else 'FAIL'} {safe_text(self._name(i, r), 40):>12}: "
                f"{r.round.passes}/{r.round.trials} = {r.round.rate:.3f} (lower {r.lower:.3f})"
                + (f" - {hint}" if hint else "")
            )
        if len(self.rounds) < self.min_rounds:
            lines.append(f"  FAIL only {len(self.rounds)} round(s), gate requires {self.min_rounds}")
        verdict = "PASS" if self.passed else "FAIL"
        return "\n".join([head] + lines + [f"verdict: {verdict}"])

    def explain(self) -> str:
        """Plain-English verdict for people who do not read confidence intervals."""
        n = len(self.rounds)
        bar = f"{self.threshold:.0%}" if abs(self.threshold * 100 - round(self.threshold * 100)) < 1e-9 else f"{self.threshold:.1%}"
        if n == 0:
            return "There were no test rounds, so nothing can be vouched for. That is a failure by design."
        if self.passed:
            return (
                f"All {n} round(s) show, with {self.confidence:.0%} confidence, that the true success rate is at least "
                f"{bar}. Each round was judged on its own pessimistic estimate, not on the average, so one lucky round "
                "cannot carry the others."
            )
        parts = []
        if n < self.min_rounds:
            parts.append(f"Only {n} round(s) were run but {self.min_rounds} are required.")
        bad = [(i, r) for i, r in enumerate(self.rounds) if not r.meets]
        if bad:
            i, r = bad[0]
            parts.append(
                f"{len(bad)} of {n} round(s) could not prove a success rate of {bar}. The weakest evidence is "
                f"{safe_text(self._name(i, r), 40)!r}: it passed {r.round.passes} of {r.round.trials} "
                f"({r.round.rate:.1%}), but allowing for luck the true rate could be as low as {r.lower:.1%}. "
            )
            if r.needed:
                parts[-1] += f"At the same pass rate, about {r.needed[0]} trials would be enough."
            elif not r.hint_skipped:
                parts[-1] += "At this pass rate, no number of trials can prove it; the candidate itself needs to improve."
        return " ".join(parts)

    def to_markdown(self) -> str:
        """GitHub-flavoured Markdown report (e.g. for $GITHUB_STEP_SUMMARY or a PR comment)."""
        name = _METHOD_NAME[self.method]
        out = [
            f"### passgate: {'PASS' if self.passed else 'FAIL'}",
            "",
            f"{sum(r.meets for r in self.rounds)}/{len(self.rounds)} round(s) clear "
            f"{name} lower bound >= {self.threshold:g} at {self.confidence:.0%} confidence "
            f"(min {self.min_rounds} rounds).",
            "",
            "| | round | passes / trials | rate | lower bound | note |",
            "|---|---|---|---|---|---|",
        ]
        for i, r in enumerate(self.rounds):
            out.append(
                f"| {'PASS' if r.meets else 'FAIL'} | {md_cell(self._name(i, r), 60)} | "
                f"{r.round.passes} / {r.round.trials} | {r.round.rate:.3f} | {r.lower:.3f} | {self._hint(r)} |"
            )
        if len(self.rounds) < self.min_rounds:
            out += ["", f"**FAIL:** only {len(self.rounds)} round(s); the gate requires {self.min_rounds}."]
        return "\n".join(out) + "\n"


def evaluate_rounds(
    rounds: Sequence[Round],
    threshold: float = 0.9,
    confidence: float = 0.95,
    min_rounds: int = 1,
    method: str = "wilson",
    hint_budget: Optional[int] = None,
) -> GateResult:
    """Gate: EVERY round's lower confidence bound must clear ``threshold``.

    A flaky candidate that spikes on average but has one weak round fails -
    the "pass^N" property. ``method`` is 'wilson' (default) or 'exact'
    (Clopper-Pearson, never undercovers). Zero rounds never pass.

    Failing rounds get a "trials needed" hint; ``hint_budget`` caps how many
    (default: 1000 for Wilson, 25 for exact, whose search is far slower), and
    hint computation also stops after ``HINT_SECONDS``. Skipped rounds are
    marked ``hint_skipped``. The verdict never depends on hints.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    z_for_confidence(confidence)  # validates 0.5 < confidence < 1
    if min_rounds < 1:
        raise ValueError("min_rounds must be >= 1")
    if method not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)}")
    budget = HINT_BUDGET[method] if hint_budget is None else hint_budget
    if budget < 0:
        raise ValueError("hint_budget must be >= 0")
    result = GateResult(threshold=threshold, confidence=confidence, min_rounds=min_rounds, method=method)
    deadline = time.monotonic() + HINT_SECONDS
    for r in rounds:
        lower = lower_bound(r.passes, r.trials, confidence, method)
        meets = lower >= threshold
        needed, skipped = None, False
        if not meets:
            if budget > 0 and time.monotonic() < deadline:
                budget -= 1
                try:
                    needed = trials_needed(r.rate, threshold, confidence, method, max_trials=10**7, deadline=deadline)
                except TimeoutError:
                    skipped = True
            else:
                skipped = True
        result.rounds.append(
            RoundEval(
                round=r, lower=lower, meets=meets, method=method, confidence=confidence,
                needed=needed, hint_skipped=skipped,
            )
        )
    return result


def _verdict(p_greater: float, p_less: float, alpha: float) -> str:
    if p_greater < alpha and p_less < alpha:  # cannot happen for alpha <= 0.5; fail closed anyway
        return "inconclusive"
    if p_greater < alpha:
        return "candidate-better"
    if p_less < alpha:
        return "baseline-better"
    return "inconclusive"


@dataclass
class Comparison:
    candidate_rate: float
    baseline_rate: float
    z: float
    p_one_sided: float
    fisher_p: Optional[float]  # None when the table exceeds FISHER_MAX_TRIALS
    alpha: float
    fisher_p_one_sided: Optional[float] = 1.0
    test: str = "z"
    p_less: float = 1.0  # one-sided p for "baseline better" under the chosen test

    @property
    def p_value(self) -> float:
        """One-sided p (candidate better) under the chosen ``test``."""
        if self.test == "fisher":
            return 1.0 if self.fisher_p_one_sided is None else self.fisher_p_one_sided
        return self.p_one_sided

    @property
    def better(self) -> bool:
        return self.verdict == "candidate-better"

    @property
    def verdict(self) -> str:
        return _verdict(self.p_value, self.p_less, self.alpha)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "candidate_rate": round(self.candidate_rate, 6),
            "baseline_rate": round(self.baseline_rate, 6),
            "test": self.test,
            "z": round(self.z, 6) if math.isfinite(self.z) else None,
            "p_one_sided": round(self.p_one_sided, 6),
            "fisher_p": None if self.fisher_p is None else round(self.fisher_p, 6),
            "fisher_p_one_sided": None if self.fisher_p_one_sided is None else round(self.fisher_p_one_sided, 6),
            "alpha": self.alpha,
            "verdict": self.verdict,
        }

    def _fisher_text(self) -> str:
        if self.fisher_p is None or self.fisher_p_one_sided is None:
            return f"fisher p=n/a (over {FISHER_MAX_TRIALS:,} trials)"
        return f"fisher p={self.fisher_p:.4g} (one-sided {self.fisher_p_one_sided:.4g})"

    def __str__(self) -> str:
        return (
            f"candidate {self.candidate_rate:.3f} vs baseline {self.baseline_rate:.3f} - "
            f"z={self.z:.3f}, one-sided p={self.p_one_sided:.4g}, "
            f"{self._fisher_text()}, test: {self.test}, verdict: {self.verdict}"
        )

    def explain(self) -> str:
        gap = (self.candidate_rate - self.baseline_rate) * 100
        base = f"The candidate passes {self.candidate_rate:.1%} and the baseline {self.baseline_rate:.1%} ({gap:+.1f} points)."
        if self.verdict == "candidate-better":
            return base + " The gap is too big to be luck at your chosen error rate: the candidate is genuinely better."
        if self.verdict == "baseline-better":
            return base + " The baseline is genuinely better; the candidate got worse."
        return (
            base + " That gap is within what luck alone could produce with this many trials, so nothing is proven. "
            "Note this is a test for 'better'; to check 'not worse', use the regress command."
        )

    def to_markdown(self) -> str:
        return (
            f"### passgate compare: {self.verdict}\n\n"
            "| | candidate | baseline |\n|---|---|---|\n"
            f"| pass rate | {self.candidate_rate:.3f} | {self.baseline_rate:.3f} |\n\n"
            f"z = {self.z:.3f}, one-sided p = {self.p_one_sided:.4g}; {self._fisher_text()}; "
            f"decided by **{self.test}** at alpha = {self.alpha:g}.\n"
        )


def compare(
    candidate: Sequence[Round],
    baseline: Sequence[Round],
    alpha: float = 0.05,
    test: str = "z",
) -> Comparison:
    """Compare candidate vs baseline on pooled totals (independent samples).

    ``test`` picks what decides the verdict: 'z' (pooled two-proportion z-test,
    the default) or 'fisher' (one-sided Fisher exact - use it for small rounds).
    ``alpha`` must be in (0, 0.5]. Fisher is reported only up to
    ``FISHER_MAX_TRIALS`` total trials (None above that).
    If both arms ran on the SAME items, use :func:`compare_paired` instead.
    """
    if not candidate or not baseline:
        raise ValueError("compare needs at least one candidate round and one baseline round")
    _check_alpha(alpha)
    if test not in ("z", "fisher"):
        raise ValueError("test must be 'z' or 'fisher'")
    c1 = sum(r.passes for r in candidate)
    n1 = sum(r.trials for r in candidate)
    c2 = sum(r.passes for r in baseline)
    n2 = sum(r.trials for r in baseline)
    z, p = two_proportion_z(c1, n1, c2, n2)
    table = (c1, n1 - c1, c2, n2 - c2)
    f_two: Optional[float] = None
    f_greater: Optional[float] = None
    if n1 + n2 <= FISHER_MAX_TRIALS:
        f_two = fisher_exact_2x2(*table)
        f_greater = fisher_exact_2x2(*table, alternative="greater")
    elif test == "fisher":
        raise ValueError(
            f"--test fisher is limited to {FISHER_MAX_TRIALS:,} total trials; use the z-test (default) at this size"
        )
    if test == "fisher":
        p_less = fisher_exact_2x2(*table, alternative="less")
    else:
        p_less = NormalDist().cdf(z) if math.isfinite(z) else (1.0 if z > 0 else 0.0)
        if z == 0.0:
            p_less = 1.0 if p == 1.0 else 0.5
    return Comparison(
        candidate_rate=c1 / n1,
        baseline_rate=c2 / n2,
        z=z,
        p_one_sided=p,
        fisher_p=f_two,
        alpha=alpha,
        fisher_p_one_sided=f_greater,
        test=test,
        p_less=p_less,
    )


@dataclass
class PairedComparison:
    """Paired (same items, both arms) comparison via the exact McNemar test."""

    n: int
    both: int
    candidate_only: int
    baseline_only: int
    neither: int
    p_one_sided: float
    p_less: float
    alpha: float

    @property
    def candidate_rate(self) -> float:
        return (self.both + self.candidate_only) / self.n

    @property
    def baseline_rate(self) -> float:
        return (self.both + self.baseline_only) / self.n

    @property
    def verdict(self) -> str:
        return _verdict(self.p_one_sided, self.p_less, self.alpha)

    @property
    def better(self) -> bool:
        return self.verdict == "candidate-better"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "items": self.n,
            "candidate_rate": round(self.candidate_rate, 6),
            "baseline_rate": round(self.baseline_rate, 6),
            "both_pass": self.both,
            "candidate_only": self.candidate_only,
            "baseline_only": self.baseline_only,
            "both_fail": self.neither,
            "test": "mcnemar-exact",
            "p_one_sided": round(self.p_one_sided, 6),
            "alpha": self.alpha,
            "verdict": self.verdict,
        }

    def __str__(self) -> str:
        return (
            f"paired: {self.n} items, candidate {self.candidate_rate:.3f} vs baseline "
            f"{self.baseline_rate:.3f} - fixed {self.candidate_only}, broke {self.baseline_only}, "
            f"exact McNemar one-sided p={self.p_one_sided:.4g}, verdict: {self.verdict}"
        )

    def explain(self) -> str:
        base = (
            f"On the same {self.n} items, the candidate fixed {self.candidate_only} that the baseline failed and "
            f"broke {self.baseline_only} that the baseline passed."
        )
        if self.verdict == "candidate-better":
            return base + " That imbalance is too lopsided to be luck: the candidate is genuinely better."
        if self.verdict == "baseline-better":
            return base + " That imbalance is too lopsided to be luck: the candidate is genuinely worse."
        return base + " That is within what luck produces when only the changed items matter, so nothing is proven."

    def to_markdown(self) -> str:
        return (
            f"### passgate paired: {self.verdict}\n\n"
            "| items | candidate rate | baseline rate | fixed (cand only) | broke (base only) | McNemar p |\n"
            "|---|---|---|---|---|---|\n"
            f"| {self.n} | {self.candidate_rate:.3f} | {self.baseline_rate:.3f} | {self.candidate_only} | "
            f"{self.baseline_only} | {self.p_one_sided:.4g} |\n"
        )


def compare_paired(pairs: Iterable[Tuple[bool, bool]], alpha: float = 0.05) -> PairedComparison:
    """Candidate vs baseline when both ran on the same items: ``pairs`` of (candidate_pass, baseline_pass).

    Uses only the items where the two disagree (exact binomial McNemar), which
    is far more sensitive than pooling when item difficulty varies.
    """
    _check_alpha(alpha)
    both = c_only = b_only = neither = 0
    for cand, base in pairs:
        if not isinstance(cand, bool) or not isinstance(base, bool):
            raise ValueError("each pair must be (bool, bool)")
        if cand and base:
            both += 1
        elif cand:
            c_only += 1
        elif base:
            b_only += 1
        else:
            neither += 1
    n = both + c_only + b_only + neither
    if n == 0:
        raise ValueError("compare_paired needs at least one item")
    p_g, p_l = mcnemar_exact(c_only, b_only)
    return PairedComparison(n, both, c_only, b_only, neither, p_g, p_l, alpha)
