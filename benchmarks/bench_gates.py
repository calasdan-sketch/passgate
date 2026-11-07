"""Committed Monte Carlo benchmark: false-pass and false-fail rates of passgate vs naive gates.

Re-run:   pip install -e ".[dev]" && python benchmarks/bench_gates.py
Writes:   benchmarks/results.json and benchmarks/RESULTS.md (seeded, so the numbers reproduce exactly
          for a given NumPy version; the NumPy version is recorded in the output).

"False pass" = the gate passes although the true rate is at or below the bar (the dangerous error).
"False fail" = the gate blocks although the true rate is comfortably above the bar (the expensive error).
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from passgate import Round, compare, min_passes  # noqa: E402
from passgate.cluster import Item, evaluate_items  # noqa: E402
from passgate.multi import bh_adjust, holm_adjust  # noqa: E402
from passgate.noninf import (  # noqa: E402
    newcombe_diff,
    newcombe_paired_diff,
    non_inferiority,
)
from passgate.power import pass_probability, plan_power  # noqa: E402
from passgate.sequential import _futility_floor, pass_boundary  # noqa: E402
from passgate.stats import fisher_exact_2x2  # noqa: E402

SEED = 20261107
OUT = os.path.dirname(os.path.abspath(__file__))
REPS = int(os.environ.get("PASSGATE_BENCH_REPS", "20000"))


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


# ---------------------------------------------------------------- 1. fixed-size gates
def bench_fixed() -> Dict[str, Any]:
    rng = np.random.default_rng(SEED + 1)
    thr, rounds = 0.9, 3
    out: List[Dict[str, Any]] = []
    plan = plan_power(0.95, thr, 0.8, rounds=rounds)
    sizes = [30, 100, 300, plan["trials"]]
    for n in sizes:
        kw = min_passes(n, thr, 0.95, "wilson")
        ke = min_passes(n, thr, 0.95, "exact")
        kw = n + 1 if kw is None else kw
        ke = n + 1 if ke is None else ke
        for p in (0.85, 0.89, 0.95, 0.97):
            k = rng.binomial(n, p, size=(REPS, rounds))
            res = {
                "naive: one run, rate >= bar": (k[:, 0] / n >= thr).mean(),
                "naive: mean of 3 rounds >= bar": (k.sum(axis=1) / (n * rounds) >= thr).mean(),
                "passgate wilson (every round)": (k >= kw).all(axis=1).mean(),
                "passgate exact (every round)": (k >= ke).all(axis=1).mean(),
            }
            exact_w = pass_probability(n, p, thr, 0.95, "wilson") ** rounds
            out.append({"n_per_round": n, "true_rate": p, "pass_rate": {a: float(b) for a, b in res.items()},
                        "exact_wilson_pass_probability": exact_w})
    return {"threshold": thr, "rounds": rounds, "reps": REPS, "planned_n": plan["trials"], "rows": out}


# ---------------------------------------------------------------- 2. peeking vs any-time
def bench_sequential() -> Dict[str, Any]:
    rng = np.random.default_rng(SEED + 2)
    thr, alpha, big_n, start = 0.9, 0.025, 500, 10
    seq = pass_boundary(thr, alpha, big_n)
    kseq = np.array([10**9 if (n == 0 or seq[n] is None) else seq[n] for n in range(big_n + 1)])
    kwil = np.array([10**9 if n < start else (min_passes(n, thr, 0.95, "wilson") or 10**9) for n in range(big_n + 1)])
    floor = _futility_floor(seq, big_n)
    floor_arr = np.array(floor[: big_n + 1], dtype=float)
    ns = np.arange(big_n + 1)
    rows = []
    for p in (0.88, 0.90, 0.95, 0.97):
        x = rng.random((REPS, big_n)) < p
        cum = np.concatenate([np.zeros((REPS, 1), dtype=np.int64), x.cumsum(axis=1)], axis=1)
        fixed = (cum[:, big_n] >= kwil[big_n]).mean()
        peek_hit = cum >= kwil[None, :]
        peek = peek_hit.any(axis=1)
        s_hit = cum >= kseq[None, :]
        s_pass = s_hit.any(axis=1)
        # trials used by the any-time test: first pass, else first futility stop, else the budget
        first_pass = np.where(s_pass, s_hit.argmax(axis=1), big_n)
        fut = (cum - ns[None, :]) < floor_arr[None, :]
        fut[:, 0] = False
        first_fut = np.where(fut.any(axis=1), fut.argmax(axis=1), big_n)
        stop = np.where(s_pass & (first_pass <= first_fut), first_pass, np.minimum(first_fut, big_n))
        rows.append({
            "true_rate": p,
            "fixed_wilson_at_500": float(fixed),
            "naive_peeking_wilson": float(peek.mean()),
            "anytime_sequential": float(s_pass.mean()),
            "anytime_mean_trials": float(stop.mean()),
        })
    return {"threshold": thr, "alpha": alpha, "max_trials": big_n, "reps": REPS, "rows": rows}


# ---------------------------------------------------------------- 3. clustered items
def bench_cluster() -> Dict[str, Any]:
    rng = np.random.default_rng(SEED + 3)
    thr, reps, m = 0.8, 4000, 10
    rows = []
    for g in (10, 30, 100):
        for icc in (0.05, 0.3):
            for mu in (0.74, 0.80, 0.9):
                s = 1.0 / icc - 1.0
                pi = rng.beta(mu * s, (1 - mu) * s, size=(reps, g))
                k = rng.binomial(m, pi)
                naive = cl = 0
                for r in range(reps):
                    items = [Item(m, int(c)) for c in k[r]]
                    res = evaluate_items(items, thr, 0.95, 1)
                    naive += res.naive_lower >= thr
                    cl += res.lower >= thr
                rows.append({"items": g, "trials_per_item": m, "icc": icc, "true_rate": mu,
                             "naive_pooled_wilson_pass": naive / reps, "cluster_aware_pass": cl / reps})
    return {"threshold": thr, "reps": reps, "rows": rows}


# ---------------------------------------------------------------- 4. regression gate
def bench_regress() -> Dict[str, Any]:
    rng = np.random.default_rng(SEED + 4)
    n, margin, reps = 200, 0.05, 6000
    base_p = 0.90
    rows = []
    for diff in (0.0, -0.03, -0.05, -0.08):
        c = rng.binomial(n, base_p + diff, size=reps)
        b = rng.binomial(n, base_p, size=reps)
        naive = sup = ni = 0
        for i in range(reps):
            cand, base = [Round(n, int(c[i]))], [Round(n, int(b[i]))]
            naive += (c[i] / n >= b[i] / n - margin)
            sup += compare(cand, base).verdict == "candidate-better"
            ni += non_inferiority(cand, base, margin).passed
        rows.append({"true_diff": diff, "naive_point_estimate_rule": naive / reps,
                     "superiority_compare": sup / reps, "regress_non_inferiority": ni / reps})
    # interval coverage against the simulated truth (nominal 90% two-sided)
    cov = []
    for n_i in (30, 100, 400):
        for p1, p2 in ((0.9, 0.9), (0.8, 0.7), (0.97, 0.9)):
            c1, c2 = rng.binomial(n_i, p1, size=3000), rng.binomial(n_i, p2, size=3000)
            hit = sum(newcombe_diff(int(a), n_i, int(b), n_i, 0.9)[0] <= p1 - p2 <= newcombe_diff(int(a), n_i, int(b), n_i, 0.9)[1]
                      for a, b in zip(c1, c2))
            cov.append({"design": "independent", "n": n_i, "p1": p1, "p2": p2, "coverage": hit / 3000})
        for p11, p10, p01 in ((0.8, 0.05, 0.05), (0.85, 0.1, 0.02), (0.6, 0.15, 0.1)):
            cells = np.array([p11, p10, p01, 1 - p11 - p10 - p01])
            tab = rng.multinomial(n_i, cells, size=3000)
            truth = p10 - p01
            hit = 0
            for a, b, c_, d in tab:
                lo, hi = newcombe_paired_diff(int(a), int(b), int(c_), int(d), 0.9)
                hit += lo <= truth <= hi
            cov.append({"design": "paired", "n": n_i, "p_both": p11, "p_cand_only": p10, "p_base_only": p01, "coverage": hit / 3000})
    return {"n_per_arm": n, "margin": margin, "reps": reps, "rows": rows, "coverage_nominal_0.90": cov}


# ---------------------------------------------------------------- 5. many cases
def bench_cases() -> Dict[str, Any]:
    rng = np.random.default_rng(SEED + 5)
    cases, n, reps, alpha = 50, 20, 4000, 0.05
    # one-sided Fisher p (candidate worse) for every (cand passes, base passes) pair at n per arm
    table = np.array([[fisher_exact_2x2(a, n - a, b, n - b, "less") for b in range(n + 1)] for a in range(n + 1)])
    out = {}
    for label, n_bad in (("no true regression", 0), ("3 real regressions (0.9 -> 0.5)", 3)):
        flagged = {"none (raw p < 0.05)": [], "holm": [], "bh": []}
        found = {"none (raw p < 0.05)": [], "holm": [], "bh": []}
        for _ in range(reps):
            p_c = np.full(cases, 0.9)
            p_c[:n_bad] = 0.5
            a = rng.binomial(n, p_c)
            b = rng.binomial(n, 0.9, size=cases)
            p = table[a, b]
            for name, adj in (("none (raw p < 0.05)", p), ("holm", np.array(holm_adjust(p))), ("bh", np.array(bh_adjust(p)))):
                sig = adj < alpha
                flagged[name].append(bool(sig.any()) if n_bad == 0 else bool(sig[n_bad:].any()))
                found[name].append(float(sig[:n_bad].mean()) if n_bad else 0.0)
        out[label] = {
            k: {"any_false_alarm": float(np.mean(flagged[k])), "share_of_real_regressions_found": float(np.mean(found[k]))}
            for k in flagged
        }
    return {"cases": cases, "trials_per_arm": n, "alpha": alpha, "reps": reps, "scenarios": out}


def render(res: Dict[str, Any]) -> str:
    L: List[str] = ["# passgate benchmark results", "",
                    f"Seed base {SEED}, NumPy {np.__version__}, {REPS} repetitions for sections 1-2 "
                    "(fewer where noted). Generated by `benchmarks/bench_gates.py`; do not edit by hand.", ""]
    f = res["fixed"]
    L += [f"## 1. Fixed-size gates (bar {f['threshold']}, {f['rounds']} rounds, 95% confidence)", "",
          "Share of simulated runs that PASS. Rows below the bar are false passes (bad); rows above are correct passes.",
          "", "| trials/round | true rate | naive: one run | naive: mean of 3 | passgate wilson | passgate exact | exact calc (wilson) |",
          "|---|---|---|---|---|---|---|"]
    for r in f["rows"]:
        pr = r["pass_rate"]
        L.append(f"| {r['n_per_round']} | {r['true_rate']} | {pct(pr['naive: one run, rate >= bar'])} | "
                 f"{pct(pr['naive: mean of 3 rounds >= bar'])} | {pct(pr['passgate wilson (every round)'])} | "
                 f"{pct(pr['passgate exact (every round)'])} | {pct(r['exact_wilson_pass_probability'])} |")
    L += ["", f"The last block uses `plan --rate 0.95 --threshold 0.9 --power 0.8 --rounds 3` = {f['planned_n']} trials per round.", ""]
    s = res["sequential"]
    L += [f"## 2. Peeking after every trial (bar {s['threshold']}, up to {s['max_trials']} trials)", "",
          "Share of runs that end in PASS at any point. Checking a plain Wilson bound after every trial inflates false passes; "
          "the any-time test stays under its alpha.", "",
          "| true rate | one look at 500 (wilson) | peeking every trial (wilson) | any-time sequential | sequential mean trials used |",
          "|---|---|---|---|---|"]
    for r in s["rows"]:
        L.append(f"| {r['true_rate']} | {pct(r['fixed_wilson_at_500'])} | {pct(r['naive_peeking_wilson'])} | "
                 f"{pct(r['anytime_sequential'])} | {r['anytime_mean_trials']:.0f} |")
    c = res["cluster"]
    L += ["", f"## 3. Items tried several times (bar {c['threshold']}, 10 trials per item, {c['reps']} repetitions)", "",
          "Share of runs that PASS. True rate below 0.80 = false pass. ICC = how alike repeated tries of one item are.", "",
          "| items | ICC | true rate | plain pooled Wilson | cluster-aware |", "|---|---|---|---|---|"]
    for r in c["rows"]:
        L.append(f"| {r['items']} | {r['icc']} | {r['true_rate']} | {pct(r['naive_pooled_wilson_pass'])} | {pct(r['cluster_aware_pass'])} |")
    g = res["regress"]
    L += ["", f"## 4. Regression gate ({g['n_per_arm']} trials per arm, baseline 0.90, tolerated drop {g['margin']}, {g['reps']} repetitions)", "",
          "Share of runs that PASS the candidate. A drop of 0.05 or more is a true regression (false pass if it passes).", "",
          "| true difference | point-estimate rule | superiority (`compare`) | `regress` |", "|---|---|---|---|"]
    for r in g["rows"]:
        L.append(f"| {r['true_diff']:+.2f} | {pct(r['naive_point_estimate_rule'])} | {pct(r['superiority_compare'])} | {pct(r['regress_non_inferiority'])} |")
    L += ["", "Newcombe interval coverage (nominal 90% two-sided, 3000 repetitions each):", "",
          "| design | n | setting | coverage |", "|---|---|---|---|"]
    for r in g["coverage_nominal_0.90"]:
        setting = (f"p1={r['p1']}, p2={r['p2']}" if r["design"] == "independent"
                   else f"both={r['p_both']}, cand-only={r['p_cand_only']}, base-only={r['p_base_only']}")
        L.append(f"| {r['design']} | {r['n']} | {setting} | {pct(r['coverage'])} |")
    m = res["cases"]
    L += ["", f"## 5. Searching {m['cases']} cases for regressions ({m['trials_per_arm']} trials per arm, alpha {m['alpha']}, {m['reps']} repetitions)", "",
          "| scenario | method | any false alarm | real regressions found |", "|---|---|---|---|"]
    for scen, d in m["scenarios"].items():
        for meth, v in d.items():
            L.append(f"| {scen} | {meth} | {pct(v['any_false_alarm'])} | {pct(v['share_of_real_regressions_found'])} |")
    return "\n".join(L) + "\n"


def main() -> None:
    t0 = time.time()
    res = {"seed": SEED, "numpy": np.__version__, "python": sys.version.split()[0]}
    for name, fn in (("fixed", bench_fixed), ("sequential", bench_sequential), ("cluster", bench_cluster),
                     ("regress", bench_regress), ("cases", bench_cases)):
        t = time.time()
        res[name] = fn()
        print(f"{name}: {time.time() - t:.1f}s", flush=True)
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    with open(os.path.join(OUT, "RESULTS.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(render(res))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
