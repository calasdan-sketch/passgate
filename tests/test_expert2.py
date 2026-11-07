"""Expert round 2: power planning, non-inferiority, any-time sequential test, clustered items,
multiple comparisons, ingestion, JUnit, plain-English explanations."""
import json
import math
import random
import xml.etree.ElementTree as ET
import zipfile

import pytest

import passgate.ingest as ing
from passgate import Round
from passgate.cli import main
from passgate.cluster import Item, cluster_wilson, evaluate_items, pass_at_k, pass_hat_k
from passgate.gate import compare_paired, evaluate_rounds
from passgate.junit import to_junit
from passgate.multi import Case, bh_adjust, compare_cases, holm_adjust
from passgate.noninf import (
    newcombe_diff,
    newcombe_paired_diff,
    non_inferiority,
    non_inferiority_paired,
)
from passgate.power import oc_curve, pass_probability, plan_power
from passgate.sequential import (
    anytime_lower,
    log_evidence,
    pass_boundary,
    run_sequential,
)
from passgate.stats import binom_sf, min_passes, t_cdf, t_ppf

scipy_stats = pytest.importorskip("scipy.stats", reason="scipy is the dev-only oracle")


# ---------------------------------------------------------------- stats helpers
@pytest.mark.parametrize("df", [1, 2, 5, 30, 400])
@pytest.mark.parametrize("q", [0.6, 0.9, 0.975, 0.995])
def test_t_ppf_matches_scipy(df, q):
    assert t_ppf(q, df) == pytest.approx(scipy_stats.t.ppf(q, df), rel=1e-9)
    assert t_cdf(t_ppf(q, df), df) == pytest.approx(q, abs=1e-10)
    assert t_ppf(1 - q, df) == pytest.approx(-t_ppf(q, df), rel=1e-12)


def test_binom_sf_matches_scipy():
    for n, k, p in [(20, 5, 0.3), (200, 190, 0.95), (1000, 480, 0.5), (50, 0, 0.1), (50, 51, 0.9)]:
        assert binom_sf(k, n, p) == pytest.approx(scipy_stats.binom.sf(k - 1, n, p), rel=1e-9, abs=1e-14)


# ---------------------------------------------------------------- power
def test_pass_probability_matches_brute_force():
    for n, rate in [(40, 0.97), (100, 0.95), (250, 0.93)]:
        k = min_passes(n, 0.9)
        brute = sum(math.comb(n, i) * rate**i * (1 - rate) ** (n - i) for i in range(k, n + 1))
        assert pass_probability(n, rate, 0.9) == pytest.approx(brute, rel=1e-9)


def test_pass_probability_edges():
    assert pass_probability(10, 0.99, 0.9) == 0.0  # all-pass cannot clear the bar at n=10
    assert pass_probability(500, 1.0, 0.9) == 1.0
    assert pass_probability(500, 0.0, 0.9) == 0.0
    with pytest.raises(ValueError):
        pass_probability(0, 0.5, 0.9)
    with pytest.raises(ValueError):
        pass_probability(10, 1.5, 0.9)


@pytest.mark.parametrize("rate,thr,power", [(0.95, 0.9, 0.8), (0.97, 0.9, 0.9), (0.8, 0.7, 0.8)])
def test_plan_power_is_first_size_reaching_target(rate, thr, power):
    plan = plan_power(rate, thr, power)
    n = plan["trials"]
    assert pass_probability(n, rate, thr) >= power
    assert all(pass_probability(m, rate, thr) < power for m in range(1, n))  # brute-force minimality
    assert plan["stable_trials"] >= n
    assert all(pass_probability(m, rate, thr) >= power for m in range(plan["stable_trials"], plan["stable_trials"] + 50))
    assert plan["break_even_trials"] <= n


def test_plan_power_rounds_and_limits():
    one, three = plan_power(0.95, 0.9, 0.8), plan_power(0.95, 0.9, 0.8, rounds=3)
    assert three["trials"] > one["trials"]
    assert three["power_at_trials"] >= 0.8
    assert three["total_trials"] == 3 * three["trials"]
    assert plan_power(0.9, 0.9, 0.8) is None  # at the bar: never
    assert plan_power(0.95, 0.9, 0.8, max_trials=50) is None
    for bad in (dict(power=0), dict(power=1), dict(rounds=0), dict(method="x")):
        with pytest.raises(ValueError):
            plan_power(0.95, 0.9, **{"power": 0.8, **bad})


def test_oc_curve_monotone_and_rounds():
    c1 = oc_curve(200, 0.9)
    assert all(a["pass_probability"] <= b["pass_probability"] + 1e-12 for a, b in zip(c1, c1[1:]))
    c3 = oc_curve(200, 0.9, rounds=3)
    assert all(b["pass_probability"] <= a["pass_probability"] + 1e-12 for a, b in zip(c1, c3))


# ---------------------------------------------------------------- non-inferiority
def test_newcombe_independent_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.stats.proportion")
    for c1, n1, c2, n2 in [(93, 100, 85, 100), (0, 10, 5, 10), (50, 50, 40, 60), (3, 7, 1, 9), (400, 400, 380, 400)]:
        for conf in (0.8, 0.9, 0.95):
            ref = sm.confint_proportions_2indep(c1, n1, c2, n2, method="newcomb", compare="diff", alpha=1 - conf)
            got = newcombe_diff(c1, n1, c2, n2, conf)
            assert got[0] == pytest.approx(ref[0], abs=1e-12)
            assert got[1] == pytest.approx(ref[1], abs=1e-12)


def test_newcombe_paired_known_value_and_symmetry():
    # reduces to the independent-style width when there is no discordance information (phi = 0)
    lo, hi = newcombe_paired_diff(40, 0, 0, 60, 0.95)
    assert lo < 0 < hi
    lo1, hi1 = newcombe_paired_diff(70, 20, 5, 5, 0.9)
    lo2, hi2 = newcombe_paired_diff(70, 5, 20, 5, 0.9)
    assert lo1 == pytest.approx(-hi2, abs=1e-12) and hi1 == pytest.approx(-lo2, abs=1e-12)
    assert 0 < lo1  # 20 fixed vs 5 broken is a real gain
    with pytest.raises(ValueError):
        newcombe_paired_diff(0, 0, 0, 0)


def test_newcombe_coverage_monte_carlo():
    rng = random.Random(7)
    reps, hits = 1500, 0
    for _ in range(reps):
        c1 = sum(rng.random() < 0.8 for _ in range(60))
        c2 = sum(rng.random() < 0.7 for _ in range(60))
        lo, hi = newcombe_diff(c1, 60, c2, 60, 0.9)
        hits += lo <= 0.1 <= hi
    assert 0.86 <= hits / reps <= 0.95
    hits = 0
    for _ in range(reps):  # paired: p10 = 0.1, p01 = 0.02
        counts = [0, 0, 0, 0]
        for _ in range(80):
            u = rng.random()
            counts[0 if u < 0.85 else 1 if u < 0.95 else 2 if u < 0.97 else 3] += 1
        lo, hi = newcombe_paired_diff(*counts, 0.9)
        hits += lo <= 0.10 - 0.02 <= hi
    assert 0.86 <= hits / reps <= 0.97


def test_non_inferiority_verdicts():
    same = non_inferiority([Round(1000, 900)], [Round(1000, 900)], margin=0.05)
    assert same.verdict == "non-inferior" and same.passed
    sup = non_inferiority([Round(1000, 950)], [Round(1000, 900)], margin=0.05)
    assert sup.verdict == "superior" and sup.passed
    worse = non_inferiority([Round(1000, 800)], [Round(1000, 900)], margin=0.05)
    assert worse.verdict == "inferior" and not worse.passed
    vague = non_inferiority([Round(20, 17)], [Round(20, 18)], margin=0.05)
    assert vague.verdict == "inconclusive" and not vague.passed
    assert "Treated as not passing" in vague.explain()
    d = same.as_dict()
    assert d["design"] == "independent" and d["passed"] is True and d["ci_confidence"] == 0.9
    assert "regress" in same.to_markdown()


def test_non_inferiority_paired_and_validation():
    pairs = [(True, True)] * 90 + [(True, False)] * 5 + [(False, True)] * 3 + [(False, False)] * 2
    r = non_inferiority_paired(pairs, 0.05)
    assert r.paired and r.items == 100 and r.passed
    for bad_margin in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            non_inferiority([Round(10, 9)], [Round(10, 9)], bad_margin)
    for bad_alpha in (0.0, 0.7):
        with pytest.raises(ValueError):
            non_inferiority([Round(10, 9)], [Round(10, 9)], 0.05, bad_alpha)
    with pytest.raises(ValueError):
        non_inferiority([], [Round(10, 9)], 0.05)
    with pytest.raises(ValueError):
        non_inferiority_paired([], 0.05)
    with pytest.raises(ValueError):
        non_inferiority_paired([(1, 0)], 0.05)


def test_regress_false_pass_rate_at_margin_is_bounded():
    rng = random.Random(11)
    n, reps, passes = 150, 1500, 0
    for _ in range(reps):  # true difference exactly -margin: false passes must stay <= ~alpha
        c = sum(rng.random() < 0.80 for _ in range(n))
        b = sum(rng.random() < 0.85 for _ in range(n))
        passes += non_inferiority([Round(n, c)], [Round(n, b)], 0.05).passed
    assert passes / reps <= 0.08


# ---------------------------------------------------------------- sequential
def test_log_evidence_matches_numeric_integration():
    for k, n, th in [(9, 10, 0.5), (45, 50, 0.9), (30, 60, 0.4)]:
        steps = 200000
        s = 0.0
        for i in range(steps):
            p = th + (1 - th) * (i + 0.5) / steps
            s += (p / th) ** k * ((1 - p) / (1 - th)) ** (n - k)
        assert log_evidence(k, n, th) == pytest.approx(math.log(s / steps), abs=1e-5)


def test_boundary_monotone_and_inverts_evidence():
    b = pass_boundary(0.9, 0.025, 300)
    target = math.log(40)
    last = 0
    for n in range(1, 301):
        if b[n] is None:
            assert log_evidence(n, n, 0.9) < target
            continue
        assert log_evidence(b[n], n, 0.9) >= target
        assert b[n] == 0 or log_evidence(b[n] - 1, n, 0.9) < target
        assert b[n] >= last - 1
        last = b[n]
    assert b[300] / 300 > 0.9  # certifying needs a rate above the bar


def test_sequential_type_one_error_under_continuous_monitoring():
    rng = random.Random(5)
    n_max, reps, alpha = 300, 3000, 0.025
    b = pass_boundary(0.9, alpha, n_max)
    naive = [10**9 if n < 10 else (min_passes(n, 0.9) or 10**9) for n in range(n_max + 1)]
    seq_hits = naive_hits = 0
    for _ in range(reps):
        k = 0
        s_hit = w_hit = False
        for n in range(1, n_max + 1):
            k += rng.random() < 0.9  # true rate exactly at the bar: worst case
            if b[n] is not None and k >= b[n]:
                s_hit = True
            if k >= naive[n]:
                w_hit = True
        seq_hits += s_hit
        naive_hits += w_hit
    assert seq_hits / reps <= alpha + 0.012
    assert naive_hits / reps > 2 * alpha  # the classic peeking inflation is real


def test_anytime_lower_bound_coverage_and_width():
    rng = random.Random(9)
    miss = 0
    reps, n_max, p = 400, 150, 0.8
    for _ in range(reps):
        k = 0
        for n in range(1, n_max + 1):
            k += rng.random() < p
            if n % 5 == 0 and anytime_lower(k, n, 0.05) > p:
                miss += 1
                break
    assert miss / reps <= 0.05 + 0.03
    assert anytime_lower(950, 1000, 0.025) < 0.95
    from passgate.stats import wilson_lower
    assert anytime_lower(95, 100, 0.025) < wilson_lower(95, 100, 0.95)  # the price of peeking
    assert anytime_lower(0, 50) == 0.0 and anytime_lower(1, 0) == 0.0


def test_run_sequential_decisions():
    assert run_sequential([True] * 80, 0.9).decision == "pass"
    early = run_sequential([False] * 20, 0.9, max_trials=100)
    assert early.decision == "fail" and early.trials < 20
    assert run_sequential([True] * 5, 0.9).decision == "undecided"  # too short: never a pass
    capped = run_sequential(([True] * 9 + [False]) * 10, 0.9, max_trials=100)
    assert capped.decision == "fail" and not capped.passed
    assert "Treated" in run_sequential([True] * 5, 0.9).explain()
    assert run_sequential([True] * 80, 0.9).as_dict()["passed"] is True
    with pytest.raises(ValueError):
        run_sequential([1, 0], 0.9)
    with pytest.raises(ValueError):
        run_sequential([True], 1.2)
    with pytest.raises(ValueError):
        run_sequential([True], 0.9, max_trials=0)
    with pytest.raises(ValueError):
        pass_boundary(0.9, 0.7, 10)


# ---------------------------------------------------------------- clustered items / pass^k
def test_pass_hat_k_and_pass_at_k_known_values():
    assert pass_hat_k(10, 7, 3) == pytest.approx(math.comb(7, 3) / math.comb(10, 3))
    assert pass_hat_k(10, 10, 4) == 1.0 and pass_hat_k(10, 3, 4) == 0.0
    assert pass_at_k(10, 0, 3) == 0.0 and pass_at_k(10, 10, 3) == 1.0
    assert pass_at_k(10, 3, 1) == pytest.approx(0.3)
    assert pass_hat_k(10, 3, 1) == pytest.approx(0.3)
    with pytest.raises(ValueError):
        pass_hat_k(3, 3, 5)


def test_pass_hat_k_unbiased_by_enumeration():
    # E over Binomial(m, p) of C(c,k)/C(m,k) equals p**k exactly
    m, p, k = 8, 0.7, 3
    e = sum(math.comb(m, c) * p**c * (1 - p) ** (m - c) * pass_hat_k(m, c, k) for c in range(m + 1))
    assert e == pytest.approx(p**k, rel=1e-12)


def test_cluster_bound_is_never_less_cautious_than_plain_wilson():
    rng = random.Random(1)
    for _ in range(200):
        items = [Item(10, rng.choice([0, 5, 10, 10, 9, 3])) for _ in range(rng.randint(2, 40))]
        _, lo, _, deff, naive = cluster_wilson(items)
        assert lo <= naive + 1e-9 and deff >= 1.0 - 1e-12


def test_cluster_aware_catches_what_plain_wilson_passes():
    items = [Item(10, 10)] * 22 + [Item(10, 0)] * 8  # 73% pass rate, items are all-or-nothing
    r = evaluate_items(items, threshold=0.65)
    assert r.naive_lower >= 0.65 > r.lower and not r.passed
    assert r.deff > 5
    assert "wrongly passed" in r.explain()


def test_cluster_degenerate_inputs_fail_closed():
    assert not evaluate_items([Item(10, 10)] * 3, threshold=0.99).passed  # 3 items cannot certify 99%
    assert evaluate_items([Item(5, 5)] * 400, threshold=0.9).passed
    assert not evaluate_items([Item(1, 1)], threshold=0.5).passed
    with pytest.raises(ValueError):
        evaluate_items([], 0.9)
    with pytest.raises(ValueError):
        evaluate_items([Item(2, 1)], 0.9, k=3)
    with pytest.raises(ValueError):
        Item(0, 0)
    with pytest.raises(ValueError):
        Item(5, 6)
    with pytest.raises(ValueError):
        Item(5.0, 2)
    with pytest.raises(ValueError):
        evaluate_items([Item(5, 5)], 0.9, k=0)
    with pytest.raises(ValueError):
        evaluate_items([Item(5, 5)], 0.9, k=2, metric="x")


def test_cluster_false_pass_rate_monte_carlo():
    rng = random.Random(3)
    reps, g, m, mu, icc, thr = 800, 30, 10, 0.80, 0.3, 0.80
    s = 1 / icc - 1
    naive = clus = 0
    for _ in range(reps):
        items = [Item(m, sum(rng.random() < p for _ in range(m))) for p in (rng.betavariate(mu * s, (1 - mu) * s) for _ in range(g))]
        r = evaluate_items(items, thr)
        naive += r.naive_lower >= thr
        clus += r.passed
    assert naive / reps > 0.10  # plain Wilson is badly anti-conservative here
    assert clus / reps < 0.07


def test_pass_k_gate_and_skipped_items():
    items = [Item(4, 4)] * 30 + [Item(2, 2)]
    r = evaluate_items(items, 0.5, k=3)
    assert r.skipped_items == 1 and r.items == 30 and r.passed
    r2 = evaluate_items(items, 0.5, k=3, metric="pass@k")
    assert r2.metric == "pass@k" and r2.estimate == 1.0


# ---------------------------------------------------------------- multiple comparisons
def test_holm_bh_match_statsmodels():
    mt = pytest.importorskip("statsmodels.stats.multitest")
    rng = random.Random(4)
    for _ in range(20):
        p = [rng.random() ** rng.choice([1, 3, 6]) for _ in range(rng.randint(1, 60))]
        for ours, name in ((holm_adjust, "holm"), (bh_adjust, "fdr_bh")):
            ref = mt.multipletests(p, method=name)[1]
            assert max(abs(a - b) for a, b in zip(ours(p), ref)) < 1e-12


def test_compare_cases_flags_and_controls():
    cases = [Case("a", 10, 20, 19, 20), Case("b", 18, 20, 19, 20), Case("c", 20, 20, 12, 20)]
    res = compare_cases(cases)
    assert [r.case.name for r in res.regressions] == ["a"]
    assert [r.case.name for r in res.improvements] == ["c"]
    assert not res.passed and "1 of 3" in res.explain()
    quiet = compare_cases([Case(str(i), 18, 20, 18, 20) for i in range(30)])
    assert quiet.passed and "none got reliably worse" in quiet.explain()


def test_compare_cases_family_wise_error_control():
    rng = random.Random(8)
    reps, flagged_raw, flagged_holm = 300, 0, 0
    for _ in range(reps):
        cs = [Case(str(i), sum(rng.random() < 0.9 for _ in range(20)), 20, sum(rng.random() < 0.9 for _ in range(20)), 20) for i in range(40)]
        flagged_raw += not compare_cases(cs, adjust="none").passed
        flagged_holm += not compare_cases(cs, adjust="holm").passed
    assert flagged_raw / reps > 0.2
    assert flagged_holm / reps <= 0.08


def test_compare_cases_validation():
    with pytest.raises(ValueError):
        compare_cases([])
    with pytest.raises(ValueError):
        compare_cases([Case("x", 5, 4, 1, 2)])
    with pytest.raises(ValueError):
        compare_cases([Case("x", 1, 2, 1, 2)], adjust="nope")
    with pytest.raises(ValueError):
        compare_cases([Case("x", 1, 2, 1, 2)], alpha=0.9)


# ---------------------------------------------------------------- ingestion
def _w(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return str(p)


def _promptfoo_rows(n_tests=6, reps=3, prompts=1):
    rows = []
    for pi in range(prompts):
        for t in range(n_tests):
            for r in range(reps):
                rows.append({"testIdx": t, "promptIdx": pi, "provider": {"id": f"prov{pi}"}, "success": (t + r + pi) % 3 != 0})
    return rows


def test_promptfoo_v3_json_and_jsonl(tmp_path):
    rows = _promptfoo_rows()
    p = _w(tmp_path, "r.json", {"version": 3, "results": {"version": 3, "results": rows}})
    arms = ing.load_arms(p)
    assert list(arms) == ["prov0|prompt0"]
    assert len(arms["prov0|prompt0"]) == 6 and all(t == 3 for t, _ in arms["prov0|prompt0"].values())
    pj = _w(tmp_path, "r.jsonl", "\n".join(json.dumps(r) for r in rows))
    assert ing.load_arms(pj) == arms
    assert sum(i.trials for i in ing.load_items(p)) == 18


def test_promptfoo_fails_closed(tmp_path):
    with pytest.raises(ing.IngestError):
        ing.load_arms(_w(tmp_path, "a.json", {"results": {"results": [{"testIdx": 0}]}}), "promptfoo")
    with pytest.raises(ing.IngestError):
        ing.load_arms(_w(tmp_path, "b.json", {"nothing": 1}), "promptfoo")
    with pytest.raises(ing.IngestError):
        ing.load_arms(_w(tmp_path, "c.json", "{not json"), "promptfoo")
    with pytest.raises(ing.IngestError):
        ing.load_arms(str(tmp_path / "missing.json"), "promptfoo")


def test_multi_arm_requires_choice(tmp_path):
    p = _w(tmp_path, "m.json", {"results": {"results": _promptfoo_rows(prompts=2, reps=1)}})
    arms = ing.load_arms(p)
    assert len(arms) == 2
    with pytest.raises(ing.IngestError, match="--arm"):
        ing.load_items(p)
    assert len(ing.load_items(p, arm="prov1|prompt1")) == 6
    with pytest.raises(ing.IngestError, match="not found"):
        ing.load_items(p, arm="nope")
    pairs = ing.pairs_from_arms(arms["prov0|prompt0"], arms["prov1|prompt1"])
    assert len(pairs) == 6


def test_pairs_from_arms_guards(tmp_path):
    with pytest.raises(ing.IngestError, match="same items"):
        ing.pairs_from_arms({"a": (1, 1)}, {"b": (1, 1)})
    with pytest.raises(ing.IngestError, match="repeated trials"):
        ing.pairs_from_arms({"a": (3, 2)}, {"a": (3, 1)})


def _inspect_log(scores_per_sample, scorer="match"):
    samples = []
    for sid, epochs in scores_per_sample.items():
        for e, v in enumerate(epochs, start=1):
            samples.append({"id": sid, "epoch": e, "scores": {scorer: {"value": v}}})
    return {"version": 2, "status": "success", "eval": {"task": "t"}, "samples": samples}


def test_inspect_json_and_eval_archive(tmp_path):
    log = _inspect_log({1: ["C", "C", "I"], 2: ["I", "I", "N"], "x": [1, 1.0, True]})
    p = _w(tmp_path, "l.json", log)
    arms = ing.load_arms(p)
    assert arms["default"] == {"1": (3, 2), "2": (3, 0), "x": (3, 3)}
    z = tmp_path / "l.eval"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("header.json", json.dumps({"status": "success"}))
        for s in log["samples"]:
            zf.writestr(f"samples/{s['id']}_epoch_{s['epoch']}.json", json.dumps(s))
    assert ing.load_arms(str(z)) == arms


def test_inspect_refuses_partial_credit_and_ambiguous_scorers(tmp_path):
    with pytest.raises(ing.IngestError, match="partial"):
        ing.load_arms(_w(tmp_path, "p.json", _inspect_log({1: ["P"]})))
    with pytest.raises(ing.IngestError, match="fractional"):
        ing.load_arms(_w(tmp_path, "f.json", _inspect_log({1: [0.5]})))
    both = _inspect_log({1: ["C"]})
    both["samples"][0]["scores"]["other"] = {"value": "I"}
    p = _w(tmp_path, "b.json", both)
    with pytest.raises(ing.IngestError, match="--scorer"):
        ing.load_arms(p)
    assert ing.load_arms(p, scorer="other")["default"] == {"1": (1, 0)}
    with pytest.raises(ing.IngestError, match="not found"):
        ing.load_arms(p, scorer="zzz")
    unscored = {"samples": [{"id": 1, "epoch": 1}]}
    with pytest.raises(ing.IngestError, match="scores"):
        ing.load_arms(_w(tmp_path, "u.json", unscored), "inspect")


def test_eval_archive_guards(tmp_path, monkeypatch):
    bad = tmp_path / "bad.eval"
    bad.write_bytes(b"not a zip")
    with pytest.raises(ing.IngestError):
        ing.load_arms(str(bad))
    z = tmp_path / "big.eval"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("samples/1_epoch_1.json", json.dumps({"id": 1, "scores": {"m": {"value": "C"}}}))
    monkeypatch.setattr(ing, "MAX_ZIP_MEMBER_BYTES", 5)
    with pytest.raises(ing.IngestError, match="too large"):
        ing.load_arms(str(z))


def test_generic_formats_and_errors(tmp_path):
    csvp = _w(tmp_path, "g.csv", "item,pass,arm\na,1,x\na,0,x\nb,true,y\n")
    assert ing.load_arms(csvp) == {"x": {"a": (2, 1)}, "y": {"b": (1, 1)}}
    counted = _w(tmp_path, "c.json", [{"id": "q1", "trials": 5, "passes": 4}])
    assert ing.load_arms(counted) == {"default": {"q1": (5, 4)}}
    with pytest.raises(ing.IngestError):
        ing.load_arms(_w(tmp_path, "e.csv", "item,pass\na,maybe\n"))
    with pytest.raises(ing.IngestError):
        ing.load_arms(_w(tmp_path, "e2.json", [{"id": "q", "trials": 5, "passes": 6}]))
    with pytest.raises(ing.IngestError):
        ing.load_arms(_w(tmp_path, "e3.json", [{"id": "q"}]))
    with pytest.raises(ing.IngestError):
        ing.load_arms(_w(tmp_path, "e4.csv", "item,pass\n"))
    with pytest.raises(ing.IngestError):
        ing.load_arms(counted, "bogus")
    deep = _w(tmp_path, "deep.json", "[" * 5000 + "]" * 5000)
    with pytest.raises(ing.IngestError):
        ing.load_arms(deep)


# ---------------------------------------------------------------- JUnit and explanations
def test_junit_is_valid_xml_even_for_hostile_labels():
    evil = "a\x1b[31m\x00<b>&\"'\ud800\u202e"
    res = evaluate_rounds([Round(100, 70, evil), Round(1000, 990, "ok")], 0.9, min_rounds=3)
    xml = to_junit(res)
    root = ET.fromstring(xml.encode("utf-8"))
    suite = root.find("testsuite")
    assert suite.get("tests") == "3" and suite.get("failures") == "2"
    assert "\x1b" not in xml and "\x00" not in xml and "\u202e" not in xml


def test_junit_for_each_result_type():
    pc = compare_paired([(True, False)] * 12 + [(True, True)] * 20)
    ni = non_inferiority([Round(100, 90)], [Round(100, 90)], 0.05)
    cs = compare_cases([Case("z", 10, 20, 19, 20)])
    seq = run_sequential([True] * 80, 0.9)
    it = evaluate_items([Item(5, 5)] * 50, 0.9)
    for obj in (pc, ni, cs, seq, it):
        ET.fromstring(to_junit(obj).encode("utf-8"))
    with pytest.raises(TypeError):
        to_junit(object())


def test_gate_explanations_in_plain_language():
    res = evaluate_rounds([Round(100, 93, "core"), Round(100, 70, "adv")], 0.9, min_rounds=3)
    text = res.explain()
    assert "Only 2 round(s)" in text and "could not prove" in text and "86.3%" in text
    ok = evaluate_rounds([Round(1000, 990)] * 2, 0.9)
    assert "at least 90%" in ok.explain()
    assert "no number of trials" in evaluate_rounds([Round(100, 50)], 0.9).explain()
    assert "no test rounds" in evaluate_rounds([], 0.9).explain()


# ---------------------------------------------------------------- CLI
def _run(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_cli_plan_power_and_oc(capsys):
    code, out, _ = _run(capsys, "plan", "--rate", "0.95", "--threshold", "0.9", "--power", "0.8", "--rounds", "3")
    assert code == 0 and "327 trials per round" in out and "981 trials" in out
    code, out, _ = _run(capsys, "plan", "--rate", "0.95", "--threshold", "0.9", "--power", "0.8", "--json")
    assert json.loads(out)["trials"] == 217
    code, out, _ = _run(capsys, "plan", "--rate", "0.85", "--threshold", "0.9", "--power", "0.8")
    assert code == 1 and "cannot certify" in out
    assert _run(capsys, "plan", "--rate", "0.95", "--power", "1.5")[0] == 2
    code, out, _ = _run(capsys, "oc", "--trials", "200", "--threshold", "0.9")
    assert code == 0 and "true rate 0.96" in out
    assert json.loads(_run(capsys, "oc", "--trials", "200", "--json")[1])["curve"]
    assert _run(capsys, "oc", "--trials", "0")[0] == 2


def test_cli_regress(capsys, tmp_path):
    code, out, _ = _run(capsys, "regress", "--margin", "0.05", "--candidate", "900/1000", "--baseline", "905/1000")
    assert code == 0 and "non-inferior" in out
    code, out, _ = _run(capsys, "regress", "--margin", "0.02", "--candidate", "800/1000", "--baseline", "900/1000", "--explain")
    assert code == 1 and "inferior" in out and "In plain words" in out
    p = _w(tmp_path, "pairs.csv", "item,candidate,baseline\n" + "\n".join(f"q{i},1,{int(i % 10 != 0)}" for i in range(100)))
    code, out, _ = _run(capsys, "regress", "--margin", "0.05", "--paired", p, "--json")
    assert code == 0 and json.loads(out)["design"] == "paired"
    assert _run(capsys, "regress", "--margin", "0.05")[0] == 2
    assert _run(capsys, "regress", "--margin", "0.05", "--paired", p, "--candidate", "1/2")[0] == 2
    assert _run(capsys, "regress", "--margin", "0", "--candidate", "9/10", "--baseline", "9/10")[0] == 2
    assert _run(capsys, "regress", "--margin", "0.05", "--candidate", "bad", "--baseline", "9/10")[0] == 2


def test_cli_sequential(capsys, tmp_path):
    good = _w(tmp_path, "s.txt", "\n".join(["1"] * 80))
    code, out, _ = _run(capsys, "sequential", good, "--threshold", "0.9", "--explain")
    assert code == 0 and "PASS" in out and "peeked" in out
    bad = _w(tmp_path, "b.txt", "0 0 0 0 0 0 0 0 0 0 0 0")
    assert _run(capsys, "sequential", bad, "--max-trials", "50")[0] == 1
    assert _run(capsys, "sequential", _w(tmp_path, "j.json", [True, False, True]))[0] == 1
    assert _run(capsys, "sequential", _w(tmp_path, "x.txt", "1 0 maybe"))[0] == 2
    assert _run(capsys, "sequential", _w(tmp_path, "e.txt", "   "))[0] == 2
    assert _run(capsys, "sequential", str(tmp_path / "nope.txt"))[0] == 2
    assert _run(capsys, "sequential", good, "--alpha", "0.9")[0] == 2
    code, out, _ = _run(capsys, "sequential", good, "--json")
    assert json.loads(out)["decision"] == "pass"


def test_cli_items_and_arms(capsys, tmp_path):
    rows = [{"testIdx": t, "promptIdx": 0, "provider": "p", "success": t % 4 != 0 or r == 0} for t in range(40) for r in range(5)]
    p = _w(tmp_path, "pf.json", {"results": {"results": rows}})
    code, out, _ = _run(capsys, "arms", p)
    assert code == 0 and "40 item(s), 200 trial(s)" in out
    code, out, _ = _run(capsys, "items", p, "--threshold", "0.7", "--explain")
    assert "design effect" in out and "In plain words" in out and code in (0, 1)
    code, out, _ = _run(capsys, "items", p, "--k", "3", "--threshold", "0.5", "--format", "junit")
    ET.fromstring(out.encode("utf-8"))
    assert _run(capsys, "items", p, "--k", "9")[0] == 2  # more tries per item than exist
    assert _run(capsys, "items", str(tmp_path / "nope.json"))[0] == 2
    assert _run(capsys, "items", p, "--threshold", "2")[0] == 2


def test_cli_cases(capsys, tmp_path):
    p = _w(tmp_path, "c.csv", "case,cand_passes,cand_trials,base_passes,base_trials\na,10,20,19,20\nb,18,20,19,20\n")
    code, out, _ = _run(capsys, "cases", p, "--explain")
    assert code == 1 and "REGRESSED" in out
    code, out, _ = _run(capsys, "cases", p, "--format", "junit")
    assert ET.fromstring(out.encode("utf-8")).find("testsuite").get("failures") == "1"
    dup = _w(tmp_path, "d.csv", "case,cand_passes,cand_trials,base_passes,base_trials\na,1,2,1,2\na,1,2,1,2\n")
    assert _run(capsys, "cases", dup)[0] == 2
    assert _run(capsys, "cases", _w(tmp_path, "m.csv", "case,cand_passes\na,1\n"))[0] == 2
    assert _run(capsys, "cases", _w(tmp_path, "n.json", [1, 2]))[0] == 2
    ok = _w(tmp_path, "ok.json", [{"case": "z", "cand_passes": 5, "cand_trials": 5, "base_passes": 5, "base_trials": 5}])
    assert _run(capsys, "cases", ok)[0] == 0


def test_cli_paired_from_promptfoo_and_baseline_file(capsys, tmp_path):
    rows = []
    for t in range(60):
        rows.append({"testIdx": t, "promptIdx": 0, "provider": {"id": "m"}, "success": t % 5 != 0})
        rows.append({"testIdx": t, "promptIdx": 1, "provider": {"id": "m"}, "success": True})
    p = _w(tmp_path, "pf.json", {"results": {"results": rows}})
    code, out, _ = _run(capsys, "paired", p, "--candidate-arm", "m|prompt1", "--baseline-arm", "m|prompt0")
    assert code == 0 and "candidate-better" in out
    assert _run(capsys, "paired", p, "--candidate-arm", "m|prompt1")[0] == 2
    a = _w(tmp_path, "a.json", {"results": {"results": [r for r in rows if r["promptIdx"] == 1]}})
    b = _w(tmp_path, "b.json", {"results": {"results": [r for r in rows if r["promptIdx"] == 0]}})
    code, out, _ = _run(capsys, "paired", a, "--baseline-file", b)
    assert code == 0 and "candidate-better" in out
    short = _w(tmp_path, "s.json", {"results": {"results": [r for r in rows if r["promptIdx"] == 0][:30]}})
    assert _run(capsys, "paired", a, "--baseline-file", short)[0] == 2


def test_cli_explain_and_junit_on_existing_commands(capsys, tmp_path):
    p = _w(tmp_path, "r.json", [{"label": "x\x1b[2J", "trials": 100, "passes": 70}])
    code, out, _ = _run(capsys, "gate", p, "--explain")
    assert code == 1 and "In plain words" in out and "\x1b" not in out
    code, out, _ = _run(capsys, "gate", p, "--format", "junit")
    ET.fromstring(out.encode("utf-8"))
    code, out, _ = _run(capsys, "gate", p, "--json", "--explain")
    assert "explanation" in json.loads(out)
    code, out, _ = _run(capsys, "compare", "--candidate", "93/100", "--baseline", "85/100", "--explain")
    assert "In plain words" in out
    code, out, _ = _run(capsys, "gate", p, "--format", "markdown", "--explain")
    assert "could not prove" in out


def test_github_summary_gets_explanation_free_table(capsys, tmp_path, monkeypatch):
    target = tmp_path / "sum.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(target))
    p = _w(tmp_path, "r.json", [{"trials": 100, "passes": 99}])
    _run(capsys, "regress", "--margin", "0.05", "--candidate", "99/100", "--baseline", "98/100", "--github-summary")
    assert "passgate regress" in target.read_text()
    assert p
