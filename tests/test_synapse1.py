"""Round 1 features: exact bounds, planning, paired McNemar, one-sided Fisher, reports, safety."""
import json
import math

import pytest

import passgate.stats as ps
from passgate import Round, compare, compare_paired, evaluate_rounds
from passgate.cli import load_pairs, main
from passgate.safe import md_cell, safe_text
from passgate.stats import (
    betainc,
    clopper_pearson_interval,
    fisher_exact_2x2,
    lower_bound,
    mcnemar_exact,
    min_passes,
    trials_needed,
    wilson_lower,
    z_for_confidence,
)

# ---------- incomplete beta / Clopper-Pearson (references derived from closed forms) ----------


@pytest.mark.parametrize("x", [0.01, 0.3, 0.5, 0.77, 0.999])
def test_betainc_closed_forms(x):
    assert betainc(1, 1, x) == pytest.approx(x, abs=1e-13)
    assert betainc(3, 1, x) == pytest.approx(x**3, rel=1e-12)
    assert betainc(1, 4, x) == pytest.approx(1 - (1 - x) ** 4, rel=1e-12)
    assert betainc(2.5, 7, x) == pytest.approx(1 - betainc(7, 2.5, 1 - x), abs=1e-13)


def _binom_sf(k, n, p):
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def test_clopper_pearson_matches_its_definition():
    lo, hi = clopper_pearson_interval(7, 20, 0.95)
    assert _binom_sf(7, 20, lo) == pytest.approx(0.025, abs=1e-10)  # P(X >= 7 | lo) = alpha/2
    assert 1 - _binom_sf(8, 20, hi) == pytest.approx(0.025, abs=1e-10)  # P(X <= 7 | hi) = alpha/2


def test_clopper_pearson_edges_and_symmetry():
    n = 50
    assert clopper_pearson_interval(n, n, 0.95)[0] == pytest.approx(0.025 ** (1 / n), rel=1e-12)
    assert clopper_pearson_interval(0, n, 0.95) == (0.0, pytest.approx(1 - 0.025 ** (1 / n), rel=1e-12))
    for k in (1, 13, 49):
        lo, hi = clopper_pearson_interval(k, n)
        lo2, hi2 = clopper_pearson_interval(n - k, n)
        assert lo == pytest.approx(1 - hi2, abs=1e-12) and hi == pytest.approx(1 - lo2, abs=1e-12)


def _lower_coverage(n, p, method):
    """P(lower bound <= p) when X ~ Binomial(n, p): how often the one-sided claim holds."""
    return sum(
        math.comb(n, k) * p**k * (1 - p) ** (n - k)
        for k in range(n + 1)
        if lower_bound(k, n, 0.95, method) <= p
    )


def test_exact_lower_bound_never_undercovers_but_wilson_can():
    n = 30
    grid = [i / 200 for i in range(1, 200)]
    exact = [_lower_coverage(n, p, "exact") for p in grid]
    wilson = [_lower_coverage(n, p, "wilson") for p in grid]
    assert min(exact) >= 0.975 - 1e-9  # one-sided 97.5% guarantee holds for every p
    assert min(wilson) < 0.975  # Wilson's coverage dips below nominal for some p


def test_lower_bound_rejects_unknown_method():
    with pytest.raises(ValueError):
        lower_bound(5, 10, method="wald")


# ---------- planning ----------


def test_trials_needed_zero_failure_closed_forms():
    z = z_for_confidence(0.95)
    # Wilson all-pass lower = n / (n + z^2) >= t  <=>  n >= z^2 t / (1 - t)
    assert trials_needed(1.0, 0.9) == (math.ceil(z * z * 0.9 / 0.1), math.ceil(z * z * 0.9 / 0.1))
    # Clopper-Pearson all-pass lower = (alpha/2)^(1/n) >= t  <=>  n >= ln(alpha/2) / ln(t)
    n = math.ceil(math.log(0.025) / math.log(0.9))
    assert trials_needed(1.0, 0.9, method="exact") == (n, n)


@pytest.mark.parametrize("method", ["wilson", "exact"])
@pytest.mark.parametrize("rate,threshold", [(0.95, 0.9), (0.8, 0.7), (0.99, 0.95), (0.6, 0.5)])
def test_trials_needed_is_valid_and_minimal(method, rate, threshold):
    n, k = trials_needed(rate, threshold, method=method)
    assert k <= math.floor(rate * n + 1e-9)
    assert lower_bound(k, n, method=method) >= threshold
    assert k == min_passes(n, threshold, method=method)
    for m in range(1, n):  # brute force: no smaller round size works at this rate
        assert lower_bound(math.floor(rate * m + 1e-9), m, method=method) < threshold


def test_trials_needed_impossible_and_validation():
    assert trials_needed(0.9, 0.9) is None
    assert trials_needed(0.5, 0.9) is None
    with pytest.raises(ValueError):
        trials_needed(1.5, 0.9)
    with pytest.raises(ValueError):
        trials_needed(0.95, 0.9, method="nope")


def test_min_passes_none_when_all_pass_cannot_clear():
    assert min_passes(5, 0.9) is None


# ---------- Fisher one-sided and large tables ----------


def _hyper(x, row1, col1, n):
    return math.comb(col1, x) * math.comb(n - col1, row1 - x) / math.comb(n, row1)


def test_fisher_one_sided_matches_direct_sum():
    a, b, c, d = 9, 1, 4, 6
    row1, col1, n = a + b, a + c, a + b + c + d
    greater = sum(_hyper(x, row1, col1, n) for x in range(a, min(row1, col1) + 1))
    less = sum(_hyper(x, row1, col1, n) for x in range(max(0, row1 + col1 - n), a + 1))
    assert fisher_exact_2x2(a, b, c, d, "greater") == pytest.approx(greater, rel=1e-12)
    assert fisher_exact_2x2(a, b, c, d, "less") == pytest.approx(less, rel=1e-12)
    assert greater + less - _hyper(a, row1, col1, n) == pytest.approx(1.0, rel=1e-12)
    with pytest.raises(ValueError):
        fisher_exact_2x2(a, b, c, d, "bigger")


@pytest.mark.parametrize("table", [(150, 50, 120, 80), (10, 290, 3, 297), (200, 0, 190, 10), (60, 60, 61, 59)])
def test_fisher_large_table_path_agrees_with_exact_integers(monkeypatch, table):
    exact = {alt: fisher_exact_2x2(*table, alt) for alt in ("two-sided", "greater", "less")}
    monkeypatch.setattr(ps, "_EXACT_INT_LIMIT", 0)
    for alt, want in exact.items():
        assert fisher_exact_2x2(*table, alt) == pytest.approx(want, rel=1e-9, abs=1e-300)


def test_fisher_huge_table_is_fast_and_sane():
    p = fisher_exact_2x2(900_000, 100_000, 899_000, 101_000, "greater")
    assert 0.0 <= p < 0.01


# ---------- McNemar / paired ----------


def test_mcnemar_exact_values():
    assert mcnemar_exact(7, 0) == (pytest.approx(1 / 128), 1.0)
    assert mcnemar_exact(0, 0) == (1.0, 1.0)
    g, less = mcnemar_exact(3, 5)
    assert g == pytest.approx(sum(math.comb(8, i) for i in range(3, 9)) / 256)
    assert less == pytest.approx(sum(math.comb(8, i) for i in range(5, 9)) / 256)
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 2)


def test_mcnemar_large_counts_use_beta_path():
    g, _ = mcnemar_exact(6000, 5000)
    assert 0 < g < 1e-15


def test_paired_beats_pooled_when_items_overlap():
    # 200 shared items; candidate fixes 12 and breaks 2. Pooled rates 0.86 vs 0.81 are not
    # significant as independent samples, but the paired test sees the 12-vs-2 split.
    pairs = [(True, True)] * 160 + [(True, False)] * 12 + [(False, True)] * 2 + [(False, False)] * 26
    pc = compare_paired(pairs)
    assert pc.verdict == "candidate-better" and pc.p_one_sided < 0.01
    pooled = compare([Round(200, 172)], [Round(200, 162)])
    assert pooled.verdict == "inconclusive"
    d = pc.as_dict()
    assert d["candidate_only"] == 12 and d["baseline_only"] == 2 and d["items"] == 200


def test_paired_validation():
    with pytest.raises(ValueError):
        compare_paired([])
    with pytest.raises(ValueError):
        compare_paired([(1, 0)])
    assert compare_paired([(False, True)] * 9).verdict == "baseline-better"


# ---------- compare: chosen test decides, baseline-better reachable ----------


def test_compare_fisher_test_is_stricter_on_small_rounds():
    by_z = compare([Round(10, 9)], [Round(10, 4)], alpha=0.02)
    by_f = compare([Round(10, 9)], [Round(10, 4)], alpha=0.02, test="fisher")
    assert by_z.verdict == "candidate-better"
    assert by_f.verdict == "inconclusive" and 0.02 < by_f.p_value < 0.05
    with pytest.raises(ValueError):
        compare([Round(10, 9)], [Round(10, 4)], test="t")


@pytest.mark.parametrize("test", ["z", "fisher"])
def test_compare_reports_baseline_better(test):
    assert compare([Round(100, 40)], [Round(100, 90)], test=test).verdict == "baseline-better"


# ---------- gate ----------


def test_gate_exact_method_and_hints():
    res = evaluate_rounds([Round(100, 93), Round(100, 70)], threshold=0.9, method="exact")
    assert res.method == "exact" and not res.passed
    d = res.as_dict()
    assert d["method"] == "exact"
    need = d["rounds"][0]["trials_needed"]
    assert need["passes"] <= 0.93 * need["trials"]
    assert lower_bound(need["passes"], need["trials"], method="exact") >= 0.9
    assert d["rounds"][1]["trials_needed"] is None  # 0.70 can never certify 0.9
    text = str(res)
    assert "Clopper-Pearson" in text and "needs" in text and "cannot help" in text
    assert d["rounds"][0]["wilson_lower"] == pytest.approx(wilson_lower(93, 100), abs=1e-6)


def test_gate_rejects_unknown_method():
    with pytest.raises(ValueError):
        evaluate_rounds([Round(10, 9)], method="wald")


def test_gate_reports_missing_rounds():
    res = evaluate_rounds([Round(1000, 990)], threshold=0.9, min_rounds=3)
    assert "only 1 round(s), gate requires 3" in str(res)
    assert "requires 3" in res.to_markdown()


def test_round_accepts_integer_like_values_only():
    class NumpyLikeInt:
        def __init__(self, v):
            self.v = v

        def __index__(self):
            return self.v

    r = Round(trials=NumpyLikeInt(10), passes=9.0)
    assert (r.trials, r.passes) == (10, 9) and type(r.trials) is int
    for bad in (9.5, True, "9", None):
        with pytest.raises(ValueError):
            Round(trials=10, passes=bad)


# ---------- untrusted text ----------


def test_safe_text_escapes_controls_bidi_zero_width():
    evil = "ok\x1b[2J‮gnp.exe​\r\n\x00"
    out = safe_text(evil)
    assert out.isprintable() and "\x1b" not in out and "‮" not in out
    assert "\\x1b" in out and "\\u202e" in out and "\\u200b" in out
    assert safe_text("x" * 500, 20) == "x" * 17 + "..."
    assert safe_text("café \U0001f680") == "café \U0001f680"  # ordinary text untouched


def test_md_cell_cannot_break_table_or_inject_html():
    cell = md_cell("a|b <img src=x onerror=alert(1)> [x](http://e)")
    assert "|" not in cell.replace("\\|", "")
    assert "<" not in cell.replace("\\<", "") and "[" not in cell.replace("\\[", "")


def test_hostile_label_is_neutralised_in_every_format(tmp_path, capsys):
    f = tmp_path / "r.json"
    f.write_text(json.dumps([{"label": "x\x1b]0;pwned\x07‮|y", "trials": 10, "passes": 10}]), encoding="utf-8")
    for fmt in ("text", "markdown", "json"):
        main(["gate", str(f), "--threshold", "0.5", "--format", fmt])
        out = capsys.readouterr().out
        assert "\x1b" not in out and "\x07" not in out and "‮" not in out, fmt


def test_error_messages_are_sanitised(tmp_path, capsys):
    assert main(["gate", str(tmp_path / "evil\x1b[31m.json")]) == 2
    assert "\x1b" not in capsys.readouterr().err


# ---------- CLI: paired, plan, formats, step summary ----------


def test_load_pairs_formats(tmp_path):
    c = tmp_path / "p.csv"
    c.write_bytes(b"\xef\xbb\xbfItem,Candidate,Baseline\r\nq1,1,0\r\nq2,PASS,fail\r\nq3,true,True\r\n")
    assert load_pairs(str(c)) == [(True, False), (True, False), (True, True)]
    j = tmp_path / "p.json"
    j.write_text(json.dumps([{"id": 1, "candidate": True, "baseline": 0}]), encoding="utf-8")
    assert load_pairs(str(j)) == [(True, False)]
    jl = tmp_path / "p.jsonl"
    jl.write_text('{"candidate": "yes", "baseline": "no"}\n\n{"candidate": 0, "baseline": 1}\n', encoding="utf-8")
    assert load_pairs(str(jl)) == [(True, False), (False, True)]


@pytest.mark.parametrize(
    "name,content",
    [
        ("dup.csv", "item,candidate,baseline\nq1,1,0\nq1,1,0\n"),
        ("empty.csv", "item,candidate,baseline\n"),
        ("blank.jsonl", "\n\n"),
        ("bad.csv", "item,candidate,baseline\nq1,maybe,0\n"),
        ("two.csv", "item,candidate,baseline\nq1,2,0\n"),
        ("missing.csv", "item,candidate\nq1,1\n"),
        ("notlist.json", '{"candidate": 1, "baseline": 0}'),
        ("line.jsonl", "[1, 0]\n"),
        ("deep.json", "[" * 100_000 + "]" * 100_000),
    ],
)
def test_paired_bad_input_exits_2(tmp_path, capsys, name, content):
    f = tmp_path / name
    f.write_text(content, encoding="utf-8")
    assert main(["paired", str(f)]) == 2
    assert capsys.readouterr().err.startswith("passgate: ")


def test_paired_cli_exit_codes(tmp_path, capsys):
    f = tmp_path / "p.jsonl"
    f.write_text("".join('{"candidate": 1, "baseline": 0}\n' for _ in range(8)), encoding="utf-8")
    assert main(["paired", str(f), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "candidate-better"
    f.write_text('{"candidate": 1, "baseline": 0}\n', encoding="utf-8")
    assert main(["paired", str(f)]) == 1


def test_gate_deeply_nested_json_exits_2(tmp_path):
    f = tmp_path / "r.json"
    f.write_text("[" * 100_000 + "]" * 100_000, encoding="utf-8")
    assert main(["gate", str(f)]) == 2


def test_plan_cli(capsys):
    assert main(["plan", "--rate", "0.95", "--threshold", "0.9", "--rounds", "3", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["achievable"] and d["total_trials"] == 3 * d["trials_per_round"]
    assert main(["plan", "--rate", "0.85", "--threshold", "0.9"]) == 1
    assert "never certify" in capsys.readouterr().out
    assert main(["plan", "--rate", "1.5"]) == 2
    assert main(["plan", "--rate", "0.95", "--rounds", "0"]) == 2


def test_markdown_format_and_step_summary(tmp_path, capsys, monkeypatch):
    f = tmp_path / "r.json"
    f.write_text("[[99, 100], [80, 100]]", encoding="utf-8")
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert main(["gate", str(f), "--format", "markdown", "--github-summary"]) == 1
    out = capsys.readouterr().out
    assert out.startswith("### passgate: FAIL") and "| FAIL |" in out
    assert summary.read_text(encoding="utf-8").startswith("### passgate: FAIL")
    assert main(["compare", "--candidate", "93/100", "--baseline", "85/100", "--github-summary"]) == 0
    assert "passgate compare" in summary.read_text(encoding="utf-8")


def test_step_summary_without_env_warns_but_keeps_exit_code(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    assert main(["compare", "--candidate", "93/100", "--baseline", "85/100", "--github-summary"]) == 0
    assert "GITHUB_STEP_SUMMARY is not set" in capsys.readouterr().err


def test_compare_cli_test_flag(capsys):
    args = ["compare", "--candidate", "9/10", "--baseline", "4/10", "--alpha", "0.02"]
    assert main(args) == 0
    assert main(args + ["--test", "fisher"]) == 1
    capsys.readouterr()
