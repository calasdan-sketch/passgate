"""Regression tests for the synapse-1 panel review findings."""
import pathlib
import re
import time
import unicodedata

import pytest

from passgate import Round, compare, compare_paired, evaluate_rounds
from passgate.cli import main
from passgate.safe import md_cell, safe_text
from passgate.stats import (
    EXACT_MAX_TRIALS,
    clopper_pearson_interval,
    fisher_exact_2x2,
    mcnemar_exact,
    trials_needed,
    wilson_lower,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


# 1. alpha above 0.5 must never let a worse/equal candidate pass
@pytest.mark.parametrize("alpha", [0.51, 0.6, 0.95, 0.999, 0.0, -0.1, 1.0])
def test_alpha_outside_range_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        compare([Round(100, 50)], [Round(100, 60)], alpha=alpha)
    with pytest.raises(ValueError, match="alpha"):
        compare_paired([(True, False)] * 5 + [(False, True)] * 5, alpha=alpha)


def test_alpha_typo_is_exit_2_not_a_pass(tmp_path):
    assert main(["compare", "--candidate", "50/100", "--baseline", "60/100", "--alpha", "0.95"]) == 2
    assert main(["compare", "--candidate", "50/100", "--baseline", "50/100", "--alpha", "0.6"]) == 2
    f = tmp_path / "p.jsonl"
    f.write_text('{"candidate": 1, "baseline": 0}\n{"candidate": 0, "baseline": 1}\n', encoding="utf-8")
    assert main(["paired", str(f), "--alpha", "0.95"]) == 2


def test_alpha_half_is_allowed_and_verdict_fails_closed_when_both_tails_small():
    assert compare([Round(100, 50)], [Round(100, 60)], alpha=0.5).verdict == "baseline-better"
    from passgate.gate import _verdict

    assert _verdict(0.01, 0.01, 0.05) == "inconclusive"


# 2. huge counts: compare stays fast, Fisher is skipped (or refused when it decides)
def test_compare_huge_counts_fast_and_fisher_skipped():
    t = time.perf_counter()
    c = compare([Round(10**16, 6 * 10**15)], [Round(10**16, 6 * 10**15 + 10**6)])
    assert time.perf_counter() - t < 1.0
    assert c.fisher_p is None and c.fisher_p_one_sided is None
    d = c.as_dict()
    assert d["fisher_p"] is None and d["verdict"] == "inconclusive"
    assert "n/a" in str(c) and "n/a" in c.to_markdown()
    with pytest.raises(ValueError, match="z-test"):
        compare([Round(10**9, 6 * 10**8)], [Round(10**9, 5 * 10**8)], test="fisher")
    assert main(["compare", "--candidate", f"{6 * 10**8}/{10**9}", "--baseline", f"{5 * 10**8}/{10**9}",
                 "--test", "fisher"]) == 2


def test_exact_methods_refuse_counts_beyond_their_accuracy():
    n = EXACT_MAX_TRIALS + 1
    with pytest.raises(ValueError):
        fisher_exact_2x2(n, 0, 0, 1)
    with pytest.raises(ValueError):
        clopper_pearson_interval(n - 1, n)
    with pytest.raises(ValueError):
        mcnemar_exact(n, 0)
    assert wilson_lower(n - 1, n) > 0.99  # Wilson has no such limit
    assert trials_needed(0.9000001, 0.9, method="exact") is None


def test_mcnemar_exact_path_is_linear():
    t = time.perf_counter()
    mcnemar_exact(2600, 2400)
    assert time.perf_counter() - t < 0.5


# 3. hint budget keeps the exact gate fast on many failing rounds
def test_exact_gate_hint_budget():
    rounds = [Round(100_000, 90_001 + i) for i in range(200)]
    t = time.perf_counter()
    res = evaluate_rounds(rounds, threshold=0.9, method="exact")
    assert time.perf_counter() - t < 6  # 2 s hint time budget + bound computation
    failing = [r for r in res.rounds if not r.meets]
    hinted = [r for r in failing if not r.hint_skipped]
    assert 1 <= len(hinted) <= 25 and len(failing) > 25
    skipped = [r for r in failing if r.hint_skipped]
    assert all(r.needed is None for r in skipped)
    assert "hint skipped" in str(res)
    d = res.as_dict()["rounds"]
    assert any(x.get("trials_needed_skipped") for x in d)
    # the verdict never depends on hints
    assert evaluate_rounds(rounds, threshold=0.9, method="exact", hint_budget=0).passed == res.passed
    with pytest.raises(ValueError):
        evaluate_rounds(rounds, hint_budget=-1)


def test_hint_count_budget_is_exact_when_time_allows():
    rounds = [Round(100, 80)] * 30  # cheap hints (rate below bar, nothing to search)
    res = evaluate_rounds(rounds, threshold=0.9, hint_budget=10)
    assert sum(not r.hint_skipped for r in res.rounds) == 10


# 4. docs and sources must not carry invisible / bidi characters
def _invisible(text):
    allowed = {"\n", "\t", "\r"}
    return [
        (i, f"U+{ord(ch):04X}")
        for i, ch in enumerate(text)
        if ch not in allowed and (unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp") or safe_text(ch) != ch)
    ]


def test_repo_text_files_have_no_invisible_characters():
    files = [ROOT / "README.md", ROOT / "CHANGELOG.md", ROOT / "pyproject.toml"]
    files += sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "tests").rglob("*.py"))
    files += sorted((ROOT / "scripts").rglob("*.py")) + sorted((ROOT / ".github").rglob("*.yml"))
    bad = {}
    for f in files:
        text = f.read_text(encoding="utf-8")
        if text.startswith("\ufeff"):
            text = text[1:]  # a leading BOM is an encoding marker, not content
        found = _invisible(text)
        if found:
            bad[str(f.relative_to(ROOT))] = found[:5]
    assert not bad, bad


# 5. SciPy cross-check is committed and runs when SciPy is installed (dev extra only)
def test_scipy_crosscheck_quick():
    pytest.importorskip("scipy")
    import importlib.util

    spec = importlib.util.spec_from_file_location("crosscheck", ROOT / "scripts" / "crosscheck_scipy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    worst = mod.run(samples=60, seed=7)
    for name, err in worst.items():
        assert err <= mod.TOL[name], (name, err)


def test_scipy_is_not_a_runtime_dependency():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    deps_line = [ln for ln in text.splitlines() if ln.strip().startswith("dependencies")]
    assert not deps_line or "scipy" not in deps_line[0]
    for f in (ROOT / "src").rglob("*.py"):
        src = f.read_text(encoding="utf-8")
        assert "import scipy" not in src and "from scipy" not in src, f


# 6. docstring no longer claims CP is always wider than Wilson (it is not)
def test_cp_docstring_and_the_counterexample():
    assert "wider than Wilson" not in (clopper_pearson_interval.__doc__ or "").replace("usually wider", "")
    assert clopper_pearson_interval(500, 500)[0] > wilson_lower(500, 500)


# 7. Markdown: mentions, autolinks, emoji shortcodes, entities and blank fillers
@pytest.mark.parametrize(
    "label,forbidden",
    [
        ("@security-team please look", "@s"),
        ("see https://evil.example/x", "://"),
        ("www.evil.example", "www."),
        ("ship it :rocket: :tada:", ":rocket:"),
        ("issue #123", "#1"),
        ("&lt;script&gt;", "&lt;"),
        ("a@b.example", "a@b"),
    ],
)
def test_md_cell_defuses_github_extras(label, forbidden):
    cell = md_cell(label)
    # the dangerous token's punctuation is backslash-escaped, so GitHub cannot act on it
    assert forbidden in label
    assert re.search(r"(?<!\\)[@:#&.]", cell) is None, cell
    # every special char is preceded by a backslash, so the text still reads the same
    assert cell.replace("\\", "") == label.replace("\\", "")


@pytest.mark.parametrize("ch", ["\u3164", "\u115f", "\u1160", "\uffa0", "\u2800", "\u180e"])
def test_blank_filler_letters_are_escaped(ch):
    out = safe_text(f"a{ch}b")
    assert ch not in out and "\\u" in out


def test_markdown_report_escapes_mentions_end_to_end(tmp_path, capsys):
    f = tmp_path / "r.json"
    f.write_text('[{"label": "@org/team :fire: https://x.example", "trials": 10, "passes": 2}]', encoding="utf-8")
    main(["gate", str(f), "--format", "markdown"])
    out = capsys.readouterr().out
    assert "\\@org" in out and "\\:fire\\:" in out and "https\\:" in out
