"""Cross-check passgate's statistics against SciPy (a dev-only oracle).

    pip install -e ".[dev]"
    python scripts/crosscheck_scipy.py            # full run
    python scripts/crosscheck_scipy.py --quick    # what the test suite runs

Prints the worst error per function and exits 1 if any exceeds its tolerance.
passgate itself never imports SciPy.
"""
from __future__ import annotations

import argparse
import random
import sys
from typing import Dict

from scipy import stats as S

from passgate.stats import (
    clopper_pearson_interval,
    fisher_exact_2x2,
    mcnemar_exact,
    wilson_interval,
)

# tolerances: absolute for interval bounds, relative for p-values (absolute below 1e-12)
TOL = {"clopper_pearson": 1e-9, "wilson": 1e-12, "fisher": 1e-8, "mcnemar": 1e-8}


def _rel(ref: float, got: float) -> float:
    return abs(ref - got) / ref if ref > 1e-12 else abs(ref - got)


def run(samples: int = 2000, seed: int = 0) -> Dict[str, float]:
    rng = random.Random(seed)
    worst = dict.fromkeys(TOL, 0.0)
    for _ in range(samples):
        n = rng.choice([1, 2, 3, 5, 10, 37, 100, 1000, 20000, 10**6])
        k = rng.choice([0, 1, n - 1, n]) if rng.random() < 0.25 else rng.randint(0, n)
        k = min(max(k, 0), n)
        conf = rng.choice([0.8, 0.9, 0.95, 0.99, 0.999])
        ref = S.binomtest(k, n).proportion_ci(conf, method="exact")
        lo, hi = clopper_pearson_interval(k, n, conf)
        worst["clopper_pearson"] = max(worst["clopper_pearson"], abs(lo - ref.low), abs(hi - ref.high))
        ref = S.binomtest(k, n).proportion_ci(conf, method="wilson")
        lo, hi = wilson_interval(k, n, conf)
        worst["wilson"] = max(worst["wilson"], abs(lo - ref.low), abs(hi - ref.high))

        n1, n2 = rng.choice([3, 10, 50, 400, 3000, 8000, 50000]), rng.choice([3, 10, 50, 400, 3000, 8000])
        a = rng.randint(0, n1)
        c = rng.randint(0, n2) if rng.random() < 0.6 else min(n2, max(0, round(a * n2 / n1) + rng.randint(-3, 3)))
        for alt in ("two-sided", "greater", "less"):
            ref_p = S.fisher_exact([[a, n1 - a], [c, n2 - c]], alternative=alt).pvalue
            worst["fisher"] = max(worst["fisher"], _rel(ref_p, fisher_exact_2x2(a, n1 - a, c, n2 - c, alt)))

        m = rng.choice([1, 5, 30, 4000, 9000, 10**5])
        b = rng.randint(0, m)
        ref_p = S.binomtest(b, m, 0.5, alternative="greater").pvalue
        worst["mcnemar"] = max(worst["mcnemar"], _rel(ref_p, mcnemar_exact(b, m - b)[0]))
    return worst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    worst = run(200 if args.quick else 2000, args.seed)
    bad = False
    for name, err in worst.items():
        flag = "OK " if err <= TOL[name] else "BAD"
        bad |= err > TOL[name]
        print(f"{flag} {name:16} worst error {err:.2e} (tolerance {TOL[name]:.0e})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
