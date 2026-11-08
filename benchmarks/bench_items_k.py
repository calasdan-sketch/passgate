"""Coverage of the pass^2 lower bound from `items --k 2`: the round-4 Student-t interval versus the round-5 Wilson interval.

True pass^2 is E[p^2] for per-item pass probabilities p ~ Beta(a, b); 10 tries per item; 1,500 simulated evals per row.
A 95% two-sided interval should have its lower end below the truth about 97.5% of the time.
Run: python benchmarks/bench_items_k.py  (needs numpy; about 1 minute)
"""
import math

import numpy as np

from passgate.cluster import _crit, _mean_t_interval, pass_hat_k

rng = np.random.default_rng(7)


def old_lower(vals, conf):
    g = len(vals)
    mean = sum(vals) / g
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (g - 1))
    return mean - _crit(conf, g - 1) * sd / math.sqrt(g)


for a, b in [(9, 1), (19, 1), (2, 2)]:
    true = (a * (a + 1)) / ((a + b) * (a + b + 1))  # E[p^2] for Beta(a, b)
    for g in (5, 10, 30, 100):
        m, reps, cov_old, cov_new = 10, 1500, 0, 0
        for _ in range(reps):
            c = rng.binomial(m, rng.beta(a, b, g))
            vals = [pass_hat_k(m, int(x), 2) for x in c]
            cov_old += old_lower(vals, 0.95) <= true
            cov_new += _mean_t_interval(vals, 0.95, float(g * m))[1] <= true
        print(f"Beta({a},{b}) true pass^2={true:.3f} g={g:3d} coverage of 95%-interval lower end: "
              f"old {cov_old / reps:.3f} new {cov_new / reps:.3f} (nominal 0.975)")
