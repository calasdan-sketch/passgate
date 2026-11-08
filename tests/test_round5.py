"""Round 5 tests: author line, differential checks against exact arithmetic, properties, golden files.

Everything here is deterministic (fixed seeds). ``hypothesis`` is used when installed and skipped when
not; the seeded generators below cover the same ground without it.
"""
import itertools
import json
import math
import os
import pathlib
import random
import re
from decimal import Decimal, getcontext
from fractions import Fraction
from statistics import NormalDist

import pytest

import passgate
from passgate import (
    Item,
    Round,
    cli,
    clopper_pearson_interval,
    evaluate_items,
    evaluate_rounds,
    fisher_exact_2x2,
    mcnemar_exact,
    min_passes,
    pass_at_k,
    pass_hat_k,
    wilson_interval,
)
from passgate.cluster import _mean_t_interval

ROOT = pathlib.Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden"
AUTHOR = "Calas Automations Inc."


# ---- author line ----------------------------------------------------------------------------

def test_author_string_is_pinned_everywhere():
    assert passgate.__author__ == AUTHOR
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert pyproject.count(f'name = "{AUTHOR}"') == 2  # authors and maintainers
    assert "calasdan-sketch" not in re.sub(r"https://github\.com/calasdan-sketch/passgate\S*", "", pyproject)
    assert (ROOT / "LICENSE").read_text(encoding="utf-8").splitlines()[2] == f"Copyright (c) 2026 {AUTHOR}"


def test_version_output_names_the_author(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith(f"passgate {passgate.__version__} (")
    assert AUTHOR in out


def test_no_handle_used_as_author_label_in_docs():
    for path in ["README.md", "CHANGELOG.md", "PUBLISH-CHECKLIST.md", "LICENSE"]:
        p = ROOT / path
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if "calasdan-sketch" in line and "uses: calasdan-sketch/passgate" not in line:
                # allowed: URLs, and the sentences that explain the old handle was replaced
                assert "github.com/calasdan-sketch" in line or "handle" in line.lower() or "Round 5" in line, (path, line)


# ---- exact reference implementations --------------------------------------------------------

def _wilson_decimal(c, n, conf):
    getcontext().prec = 60
    z = Decimal(NormalDist().inv_cdf(0.5 + conf / 2))  # z itself is a double; the algebra is exact
    p, nn = Decimal(c) / Decimal(n), Decimal(n)
    z2 = z * z
    denom = 1 + z2 / nn
    centre = (p + z2 / (2 * nn)) / denom
    half = z * ((p * (1 - p) + z2 / (4 * nn)) / nn).sqrt() / denom
    return float(max(Decimal(0), centre - half)), float(min(Decimal(1), centre + half))


def test_wilson_matches_60_digit_decimal_arithmetic():
    rng = random.Random(51)
    for _ in range(400):
        n = rng.choice([1, 2, 3, 7, 20, 100, 1000, 10**5, 10**7])
        c = rng.choice([0, 1, n // 2, n - 1, n, rng.randint(0, n)])
        conf = rng.choice([0.8, 0.9, 0.95, 0.99, 0.999])
        lo, hi = wilson_interval(c, n, conf)
        rlo, rhi = _wilson_decimal(c, n, conf)
        assert lo == pytest.approx(rlo, abs=1e-12)
        assert hi == pytest.approx(rhi, abs=1e-12)


def _binom_tail_ge(k, n, p):
    """P(X >= k) as an exact Fraction."""
    p = Fraction(p)
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def test_clopper_pearson_endpoints_satisfy_their_defining_equations_exactly():
    # lower L solves P(X >= k | L) = alpha/2; upper U solves P(X <= k | U) = alpha/2
    for n, k in [(5, 3), (12, 12), (12, 1), (30, 29), (40, 20), (60, 59)]:
        for conf in (0.9, 0.95):
            lo, hi = clopper_pearson_interval(k, n, conf)
            a2 = (1 - conf) / 2
            if k > 0:
                assert float(_binom_tail_ge(k, n, lo)) == pytest.approx(a2, abs=1e-9)
            if k < n:
                assert float(1 - _binom_tail_ge(k + 1, n, hi)) == pytest.approx(a2, abs=1e-9)


def test_pass_at_k_and_pass_hat_k_match_brute_force_enumeration():
    for m in range(1, 9):
        for c in range(0, m + 1):
            for k in range(1, m + 1):
                combos = list(itertools.combinations(range(m), k))
                all_pass = Fraction(sum(1 for s in combos if all(i < c for i in s)), len(combos))
                any_pass = Fraction(sum(1 for s in combos if any(i < c for i in s)), len(combos))
                assert pass_hat_k(m, c, k) == pytest.approx(float(all_pass), abs=1e-15)
                assert pass_at_k(m, c, k) == pytest.approx(float(any_pass), abs=1e-15)


def test_pass_k_estimators_match_the_old_binomial_formula_and_stay_fast_when_huge():
    rng = random.Random(52)
    for _ in range(300):
        m = rng.randint(1, 400)
        c = rng.randint(0, m)
        k = rng.randint(1, m)
        assert pass_hat_k(m, c, k) == pytest.approx(math.comb(c, k) / math.comb(m, k), rel=1e-11, abs=1e-300)
        assert pass_at_k(m, c, k) == pytest.approx(1 - math.comb(m - c, k) / math.comb(m, k), rel=1e-9, abs=1e-12)
    # a tiny answer keeps its digits (1 - 0.9999999 style cancellation used to lose them)
    assert pass_at_k(10**9, 1, 3) == pytest.approx(3e-9, rel=1e-6)
    # a billion trials with k = 100,000 used to take seconds (huge integers); now a float loop
    assert pass_hat_k(10**9, 5 * 10**8, 10**5) == 0.0
    assert 0.0 < pass_hat_k(10**9, 10**9 - 1, 10**5) < 1.0


def test_pass_k_estimators_reject_impossible_counts():
    for fn in (pass_hat_k, pass_at_k):
        with pytest.raises(ValueError):
            fn(5, 6, 2)
        with pytest.raises(ValueError):
            fn(5, -1, 2)
        with pytest.raises(ValueError):
            fn(5, 3, 6)


def _binom_half_sf(k, m):
    return Fraction(sum(math.comb(m, i) for i in range(k, m + 1)), 2**m)


def test_mcnemar_matches_exact_fractions():
    for b in range(0, 14):
        for c in range(0, 14):
            if b + c == 0:
                continue
            m = b + c
            pg, pl = mcnemar_exact(b, c)
            assert pg == pytest.approx(float(_binom_half_sf(b, m)), abs=1e-12)
            assert pl == pytest.approx(float(_binom_half_sf(c, m)), abs=1e-12)


def test_fisher_matches_exact_hypergeometric_enumeration():
    rng = random.Random(53)
    for _ in range(60):
        a, b, c, d = (rng.randint(0, 12) for _ in range(4))
        if a + b + c + d == 0 or a + b == 0 or c + d == 0 or a + c == 0 or b + d == 0:
            continue
        n, r1, c1 = a + b + c + d, a + b, a + c
        denom = math.comb(n, r1)
        probs = {x: Fraction(math.comb(c1, x) * math.comb(n - c1, r1 - x), denom) for x in range(max(0, r1 + c1 - n), min(r1, c1) + 1)}
        p_obs = probs[a]
        two = sum(p for p in probs.values() if p <= p_obs * (1 + Fraction(1, 10**9)))
        assert fisher_exact_2x2(a, b, c, d) == pytest.approx(float(min(Fraction(1), two)), abs=1e-9)
        assert fisher_exact_2x2(a, b, c, d, "greater") == pytest.approx(float(sum(p for x, p in probs.items() if x >= a)), abs=1e-9)
        assert fisher_exact_2x2(a, b, c, d, "less") == pytest.approx(float(sum(p for x, p in probs.items() if x <= a)), abs=1e-9)


# ---- properties (seeded; hypothesis versions below) -----------------------------------------

def test_gate_verdict_agrees_with_min_passes_for_every_count():
    for method, nmax in (("wilson", 150), ("exact", 40)):
        for thr in (0.5, 0.8, 0.9, 0.95, 0.99):
            for n in range(1, nmax + 1):
                need = min_passes(n, thr, 0.95, method)
                for c in range(0, n + 1):
                    r = evaluate_rounds([Round(n, c)], thr, 0.95, 1, method, hint_budget=0)
                    assert r.passed == (need is not None and c >= need), (method, thr, n, c)


def test_no_verdict_sits_within_float_noise_of_the_bar():
    # If a lower bound landed within 1e-9 of the bar, a different libm could flip the verdict.
    close = []
    for thr in (0.5, 0.8, 0.9, 0.95, 0.99):
        for n in range(1, 400):
            for c in range(0, n + 1):
                lo = wilson_interval(c, n, 0.95)[0]
                if abs(lo - thr) < 1e-9:
                    close.append((thr, n, c, lo))
    assert close == []


def test_wilson_bounds_are_monotone_and_ordered():
    rng = random.Random(54)
    for _ in range(300):
        n = rng.randint(1, 2000)
        c = rng.randint(0, n)
        lo, hi = wilson_interval(c, n, 0.95)
        assert 0.0 <= lo <= c / n <= hi <= 1.0
        if c < n:
            assert wilson_interval(c + 1, n, 0.95)[0] > lo
        assert wilson_interval(c, n, 0.99)[0] <= lo
        assert wilson_interval(c * 2, n * 2, 0.95)[0] >= lo - 1e-12  # same rate, twice the evidence


def test_hypothesis_properties():
    hyp = pytest.importorskip("hypothesis")
    st = pytest.importorskip("hypothesis.strategies")

    @hyp.settings(max_examples=150, deadline=None, derandomize=True)
    @hyp.given(st.integers(1, 3000), st.data(), st.floats(0.51, 0.999))
    def prop(n, data, conf):
        c = data.draw(st.integers(0, n))
        lo, hi = wilson_interval(c, n, conf)
        assert 0 <= lo <= c / n <= hi <= 1
        elo, ehi = clopper_pearson_interval(c, n, conf)
        assert elo <= lo + 1e-9 or c == n or n > 1500  # exact is wider except near c == n (documented)
        assert elo <= c / n <= ehi

    prop()


def test_a_bar_exactly_equal_to_the_lower_bound_passes():
    # '>=' not '>': evidence exactly at the bar clears it (round-trips the bound as its own threshold)
    from passgate.stats import lower_bound
    for c, n in [(93, 100), (990, 1000), (5, 5), (47, 50)]:
        thr = lower_bound(c, n, 0.95)
        if 0.0 < thr < 1.0:
            assert evaluate_rounds([Round(n, c)], thr, 0.95, 1).passed
            assert min_passes(n, thr, 0.95) == c or min_passes(n, thr, 0.95) < c
            assert not evaluate_rounds([Round(n, c - 1)], thr, 0.95, 1).passed
    r = evaluate_items([Item(10, 9)] * 30, threshold=0.5, k=1)
    assert evaluate_items([Item(10, 9)] * 30, threshold=r.lower, k=1).passed


def test_min_passes_is_exact_at_its_own_boundary():
    from passgate.stats import lower_bound
    for n in (20, 57, 100):
        for thr in (0.8, 0.9):
            need = min_passes(n, thr, 0.95)
            if need is not None:
                assert lower_bound(need, n, 0.95) >= thr
                assert lower_bound(need - 1, n, 0.95) < thr
                # a bar equal to the all-pass bound is reachable only with all passes
                top = lower_bound(n, n, 0.95)
                assert min_passes(n, top, 0.95) == n


# ---- the item-level fix ---------------------------------------------------------------------

def test_identical_items_no_longer_prove_a_perfect_pass_k():
    # Before round 5, three items that each passed 10/10 gave lower bound 1.000 for pass^2 (zero spread).
    r = evaluate_items([Item(10, 10)] * 3, threshold=0.9, k=2)
    assert r.estimate == 1.0 and r.lower < 0.5 and not r.passed
    r = evaluate_items([Item(10, 10)] * 3, threshold=0.9, k=2, metric="pass@k")
    assert not r.passed


def test_more_identical_items_do_earn_a_pass():
    r = evaluate_items([Item(10, 10)] * 200, threshold=0.9, k=2)
    assert r.passed and r.lower > 0.9


def test_item_interval_is_never_tighter_than_one_observation_per_item_and_bounded_by_trials():
    rng = random.Random(55)
    for _ in range(200):
        g = rng.randint(2, 40)
        vals = [rng.random() for _ in range(g)]
        mean, lo, hi = _mean_t_interval(vals, 0.95, float(g * 10))
        assert 0.0 <= lo <= mean <= hi <= 1.0
        _, lo_g, hi_g = _mean_t_interval(vals, 0.95, float(g))  # n_eff pinned at g: widest allowed
        assert lo_g <= lo + 1e-12 and hi_g >= hi - 1e-12


# ---- golden files: every output format ------------------------------------------------------

ROUNDS = '[{"label": "r1", "trials": 100, "passes": 97}, {"label": "r2", "trials": 100, "passes": 93}]'
ITEMS = "item,trials,passes\nalpha,10,10\nbeta,10,9\ngamma,10,6\ndelta,10,10\n"


def _golden(name, text):
    path = GOLDEN / name
    text = text.replace(passgate.__version__, "X.Y.Z")
    if os.environ.get("UPDATE_GOLDEN") == "1":
        path.parent.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    assert path.read_text(encoding="utf-8") == text, f"{name} changed; run UPDATE_GOLDEN=1 pytest tests/test_round5.py and review the diff"


@pytest.mark.parametrize("fmt", ["text", "json", "markdown", "junit"])
def test_golden_gate(fmt, tmp_path, capsys):
    f = tmp_path / "rounds.json"
    f.write_text(ROUNDS, encoding="utf-8")
    rc = cli.main(["gate", str(f), "--threshold", "0.9", "--min-rounds", "2", "--format", fmt, "--explain"])
    assert rc == 1
    _golden(f"gate.{fmt}.txt", capsys.readouterr().out)


@pytest.mark.parametrize("fmt", ["text", "json", "markdown", "junit"])
@pytest.mark.parametrize("k", [1, 2])
def test_golden_items(fmt, k, tmp_path, capsys):
    f = tmp_path / "items.csv"
    f.write_text(ITEMS, encoding="utf-8")
    rc = cli.main(["items", str(f), "--threshold", "0.8", "--k", str(k), "--format", fmt, "--explain"])
    assert rc in (0, 1)
    _golden(f"items.k{k}.{fmt}.txt", capsys.readouterr().out)


def test_json_output_is_strict_json_with_no_nan_or_infinity(tmp_path, capsys):
    f = tmp_path / "rounds.json"
    f.write_text(ROUNDS, encoding="utf-8")
    cli.main(["gate", str(f), "--json"])
    json.loads(capsys.readouterr().out, parse_constant=lambda c: pytest.fail(f"non-JSON constant {c}"))


# ---- reproducibility ------------------------------------------------------------------------

def test_same_input_gives_byte_identical_output_without_hints(tmp_path, capsys):
    f = tmp_path / "rounds.json"
    f.write_text(ROUNDS, encoding="utf-8")
    outs = []
    for _ in range(3):
        cli.main(["gate", str(f), "--json"])
        outs.append(capsys.readouterr().out)
    assert outs[0] == outs[1] == outs[2]


def test_source_uses_no_unseeded_randomness_or_wall_clock_in_statistics():
    for mod in ("stats", "cluster", "gate", "noninf", "sequential", "power", "multi"):
        src = (ROOT / "src" / "passgate" / f"{mod}.py").read_text(encoding="utf-8")
        assert "import random" not in src and "random." not in src, mod
    # the only wall-clock use is the hint deadline, which never changes a verdict
    assert "time.monotonic" in (ROOT / "src/passgate/gate.py").read_text(encoding="utf-8")
