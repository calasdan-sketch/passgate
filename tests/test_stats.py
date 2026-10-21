"""Statistics tests — references are derived, not remembered."""
import math

import pytest

from passgate.stats import (
    fisher_exact_2x2,
    two_proportion_z,
    wilson_interval,
    wilson_lower,
    z_for_confidence,
)


def test_wilson_perfect_score_exact_form():
    # for p_hat == 1 the Wilson lower bound collapses to 1 / (1 + z^2/n)
    z = z_for_confidence(0.95)
    assert wilson_lower(100, 100, 0.95) == pytest.approx(1.0 / (1.0 + z * z / 100), abs=1e-12)


def test_wilson_zero_score_symmetric():
    z = z_for_confidence(0.95)
    lo, hi = wilson_interval(0, 100, 0.95)
    assert lo == pytest.approx(0.0, abs=1e-12)
    assert hi == pytest.approx(1.0 - 1.0 / (1.0 + z * z / 100), abs=1e-12)


def test_wilson_bounds_bracket_the_rate():
    lo, hi = wilson_interval(93, 100, 0.95)
    assert 0.0 <= lo <= 0.93 <= hi <= 1.0


def test_wilson_monotone_in_passes():
    lowers = [wilson_lower(p, 100, 0.95) for p in range(0, 101)]
    assert lowers == sorted(lowers)


def test_wilson_tighter_with_more_data():
    # same observed rate, larger sample -> higher lower bound
    assert wilson_lower(90, 100, 0.95) < wilson_lower(900, 1000, 0.95)


def test_wilson_input_validation():
    with pytest.raises(ValueError):
        wilson_lower(5, 0)
    with pytest.raises(ValueError):
        wilson_lower(11, 10)
    with pytest.raises(ValueError):
        z_for_confidence(0.4)


def test_two_proportion_z_direction_and_magnitude():
    # derived: pooled p = 178/200 = 0.89, se = sqrt(0.89*0.11*0.02),
    # z = (0.93 - 0.85) / se
    se = math.sqrt(0.89 * 0.11 * (1 / 100 + 1 / 100))
    z, p = two_proportion_z(93, 100, 85, 100)
    assert z == pytest.approx(0.08 / se, rel=1e-9)
    assert 0.03 < p < 0.04


def test_two_proportion_z_degenerate():
    z, p = two_proportion_z(50, 100, 50, 100)
    assert z == 0.0 and p == 0.5  # one-sided p at z=0 is 0.5
    z, p = two_proportion_z(60, 100, 40, 100)
    assert z > 0 and p < 0.01


def test_fisher_perfect_separation():
    # [[10,0],[0,10]]: only x=0 and x=10 are as extreme, each 1/C(20,10)
    expected = 2 / math.comb(20, 10)
    assert fisher_exact_2x2(10, 0, 0, 10) == pytest.approx(expected, rel=1e-9)


def test_fisher_identical_groups_not_significant():
    p = fisher_exact_2x2(8, 2, 8, 2)
    assert p > 0.5


def test_fisher_symmetry():
    assert fisher_exact_2x2(1, 9, 11, 1) == pytest.approx(fisher_exact_2x2(11, 1, 1, 9), rel=1e-12)


def test_fisher_validation():
    with pytest.raises(ValueError):
        fisher_exact_2x2(-1, 2, 3, 4)
