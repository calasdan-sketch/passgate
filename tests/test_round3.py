"""Round 3 tests: input hardening, stdin, per-trial rounds, regress planning, hint deadline."""
import io
import json
import os
import random
import sys
import time

import pytest

from passgate import (
    Round,
    cli,
    evaluate_rounds,
    ingest,
    plan_noninferiority,
    regress_power,
)
from passgate.noninf import newcombe_diff
from passgate.safe import safe_text
from passgate.stats import trials_needed


@pytest.fixture(autouse=True)
def _fresh_stdin_cache():
    ingest._STDIN_CACHE.clear()
    yield
    ingest._STDIN_CACHE.clear()


def run(args, capsys):
    rc = cli.main(args)
    out = capsys.readouterr()
    return rc, out.out, out.err


def feed_stdin(monkeypatch, text):
    class FakeStdin:
        buffer = io.BytesIO(text.encode("utf-8"))
    monkeypatch.setattr(sys, "stdin", FakeStdin())


# --- hostile / odd input never looks like a regression (exit 1) -----------------------------

def test_huge_csv_cell_is_exit_2_not_a_traceback(tmp_path, capsys):
    f = tmp_path / "h.csv"
    f.write_text("item,candidate,baseline\na," + "1" * 2_000_000 + ",1\n")
    rc, _, err = run(["paired", str(f)], capsys)
    assert rc == 2 and "Traceback" not in err


def test_ragged_csv_row_is_exit_2(tmp_path, capsys):
    f = tmp_path / "r.csv"
    f.write_text("trials,passes\n100,90,extra,cells\n")
    rc, _, err = run(["gate", str(f)], capsys)
    assert rc == 2 and "more cells" in err


@pytest.mark.skipif(not os.path.exists("/dev/zero"), reason="POSIX special file")
def test_endless_special_file_is_capped(capsys):
    rc, _, err = run(["paired", "/dev/zero"], capsys)
    assert rc == 2 and "larger than" in err


def test_nul_bytes_rejected(tmp_path, capsys):
    f = tmp_path / "n.json"
    f.write_bytes(b'[[93, 100]]\x00')
    rc, _, err = run(["gate", str(f)], capsys)
    assert rc == 2 and "NUL" in err


def test_unexpected_crash_exits_2(monkeypatch, tmp_path, capsys):
    f = tmp_path / "r.json"
    f.write_text("[[9, 10]]")

    def boom(*a, **k):
        raise RuntimeError("synthetic")
    monkeypatch.setattr(cli, "evaluate_rounds", boom)
    rc, _, err = run(["gate", str(f)], capsys)
    assert rc == 2 and "internal error" in err


def test_long_bad_value_is_truncated_in_message(tmp_path, capsys):
    f = tmp_path / "p.csv"
    f.write_text("item,candidate,baseline\na," + "x" * 5000 + ",1\n")
    rc, _, err = run(["paired", str(f)], capsys)
    assert rc == 2 and len(err) < 400


def test_deeply_nested_json_still_exit_2(tmp_path, capsys):
    f = tmp_path / "d.json"
    f.write_text("[" * 100000)
    rc, _, _ = run(["gate", str(f)], capsys)
    assert rc == 2


# --- formats: separators, no extension, stdin ---------------------------------------------

def test_semicolon_and_tab_csv(tmp_path, capsys):
    a = tmp_path / "a.csv"
    a.write_text("trials;passes;label\n100;99;x\n")
    b = tmp_path / "b.csv"
    b.write_text("trials\tpasses\n100\t99\n")
    for f in (a, b):
        rc, out, _ = run(["gate", str(f), "--threshold", "0.9", "--json"], capsys)
        assert rc == 0 and json.loads(out)["rounds"][0]["passes"] == 99


def test_extensionless_files_are_sniffed(tmp_path, capsys):
    c = tmp_path / "rounds"
    c.write_text("trials,passes\n100,99\n")
    j = tmp_path / "rounds2"
    j.write_text('[{"trials": 100, "passes": 99}]')
    for f in (c, j):
        rc, _, _ = run(["gate", str(f), "--threshold", "0.9"], capsys)
        assert rc == 0


def test_gate_reads_stdin(monkeypatch, capsys):
    feed_stdin(monkeypatch, '[{"trials": 100, "passes": 99, "label": "s"}]')
    rc, out, _ = run(["gate", "-", "--threshold", "0.9"], capsys)
    assert rc == 0 and "PASS" in out


def test_paired_and_sequential_read_stdin(monkeypatch, capsys):
    feed_stdin(monkeypatch, "item,candidate,baseline\n" + "".join(f"i{i},1,0\n" for i in range(12)))
    rc, out, _ = run(["paired", "-", "--json"], capsys)
    assert rc == 0 and json.loads(out)["verdict"] == "candidate-better"
    ingest._STDIN_CACHE.clear()
    feed_stdin(monkeypatch, "1\n" * 80)
    rc, out, _ = run(["sequential", "-", "--threshold", "0.9"], capsys)
    assert rc == 0 and "PASS" in out


def test_wrong_columns_message_points_to_per_trial(tmp_path, capsys):
    f = tmp_path / "t.csv"
    f.write_text("item,pass\na,1\nb,0\n")
    rc, _, err = run(["gate", str(f)], capsys)
    assert rc == 2 and "--per-trial" in err


def test_items_on_a_rounds_file_explains(tmp_path, capsys):
    f = tmp_path / "r.json"
    f.write_text("[[93, 100]]")
    rc, _, err = run(["items", str(f)], capsys)
    assert rc == 2 and "passgate gate" in err


# --- gate --per-trial: one raw results file = one round --------------------------------------

def _trial_file(path, passes, trials):
    path.write_text("item,pass\n" + "".join(f"q{i},{1 if i < passes else 0}\n" for i in range(trials)))


def test_gate_per_trial_files_become_rounds(tmp_path, capsys):
    paths = []
    for i, p in enumerate((98, 99, 100)):
        f = tmp_path / f"run{i}.csv"
        _trial_file(f, p, 100)
        paths.append(str(f))
    rc, out, _ = run(["gate", *paths, "--per-trial", "--threshold", "0.9", "--min-rounds", "3", "--json"], capsys)
    d = json.loads(out)
    assert rc == 0 and [r["passes"] for r in d["rounds"]] == [98, 99, 100]
    assert d["rounds"][0]["label"] == "run0.csv"


def test_gate_per_trial_fails_on_one_weak_round(tmp_path, capsys):
    paths = []
    for i, p in enumerate((100, 80, 100)):
        f = tmp_path / f"r{i}.csv"
        _trial_file(f, p, 100)
        paths.append(str(f))
    rc, out, _ = run(["gate", *paths, "--per-trial", "--threshold", "0.9"], capsys)
    assert rc == 1 and "FAIL" in out


def test_per_trial_flags_need_per_trial(tmp_path, capsys):
    f = tmp_path / "r.json"
    f.write_text("[[99, 100]]")
    rc, _, err = run(["gate", str(f), "--arm", "x"], capsys)
    assert rc == 2 and "--per-trial" in err


def test_multiple_rounds_files_are_concatenated(tmp_path, capsys):
    a = tmp_path / "a.json"
    a.write_text("[[99, 100]]")
    b = tmp_path / "b.json"
    b.write_text("[[98, 100]]")
    rc, out, _ = run(["gate", str(a), str(b), "--threshold", "0.9", "--min-rounds", "2", "--json"], capsys)
    assert rc == 0 and len(json.loads(out)["rounds"]) == 2


def test_stdin_only_once_per_trial(monkeypatch, capsys):
    feed_stdin(monkeypatch, "item,pass\na,1\n")
    rc, _, err = run(["gate", "-", "-", "--per-trial"], capsys)
    assert rc == 2 and "only once" in err


# --- output hygiene ---------------------------------------------------------------------

def test_safe_text_escapes_lookalike_and_invisible_characters():
    s = safe_text("a\u00a0b\u2003c\ufe0fd\ue000e\u0378f")
    for bad in ("\u00a0", "\u2003", "\ufe0f", "\ue000", "\u0378"):
        assert bad not in s
    assert safe_text("plain ok é 日本") == "plain ok é 日本"


def test_json_output_carries_version(tmp_path, capsys):
    f = tmp_path / "r.json"
    f.write_text("[[99, 100]]")
    rc, out, _ = run(["gate", str(f), "--json"], capsys)
    assert json.loads(out)["passgate_version"]
    rc, out, _ = run(["plan", "--rate", "0.95", "--json"], capsys)
    assert json.loads(out)["passgate_version"]


# --- speed / budgets -------------------------------------------------------------------------

def test_trials_needed_deadline():
    with pytest.raises(TimeoutError):
        trials_needed(0.9001, 0.9, method="exact", deadline=time.monotonic() - 1)
    assert trials_needed(0.95, 0.9, deadline=time.monotonic() + 60) == (140, 133)


def test_exact_gate_with_many_failing_huge_rounds_is_fast():
    rounds = [Round(100_000, 90_001 + i) for i in range(200)]
    t = time.process_time()  # CPU time, so a busy CI machine does not flake it
    res = evaluate_rounds(rounds, threshold=0.9, method="exact")
    assert time.process_time() - t < 8
    assert sum(1 for r in res.rounds if r.hint_skipped) > 100


def test_cases_limit_checked_before_parsing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "MAX_CASES", 3)
    f = tmp_path / "c.csv"
    f.write_text("case,cand_passes,cand_trials,base_passes,base_trials\n" + "".join(f"c{i},1,2,1,2\n" for i in range(5)))
    rc, _, err = run(["cases", str(f)], capsys)
    assert rc == 2 and "too many cases" in err


# --- plan-regress ----------------------------------------------------------------------------

def test_regress_power_matches_simulation():
    rng = random.Random(20261107)
    n, p, margin, alpha = 150, 0.9, 0.05, 0.05
    exact = regress_power(n, p, p, margin, alpha)
    reps, wins = 6000, 0
    for _ in range(reps):
        kc = sum(rng.random() < p for _ in range(n))
        kb = sum(rng.random() < p for _ in range(n))
        wins += newcombe_diff(kc, n, kb, n, 1 - 2 * alpha)[0] > -margin
    assert abs(wins / reps - exact) < 0.025  # 3.5 binomial standard errors at this size


def test_plan_noninferiority_size_reaches_power_and_is_near_minimal():
    res = plan_noninferiority(0.9, 0.9, 0.05, power=0.8)
    n = res["trials_per_arm"]
    assert regress_power(n, 0.9, 0.9, 0.05) >= 0.8
    assert regress_power(int(n * 0.9), 0.9, 0.9, 0.05) < 0.8
    assert 0.9 * 0.1 * 2 / 0.05**2 * (1.645 + 0.8416) ** 2 * 0.9 < n < 0.9 * 0.1 * 2 / 0.05**2 * (1.645 + 0.8416) ** 2 * 1.15


def test_plan_noninferiority_impossible_and_validation():
    assert plan_noninferiority(0.80, 0.90, 0.05) is None  # candidate already 10 points worse
    assert plan_noninferiority(0.9, 0.9, 0.0001, max_trials=500) is None
    for bad in ({"power": 1.0}, {"alpha": 0.9}, {"margin": 0.0}, {"margin": float("nan")}):
        kw = {"candidate_rate": 0.9, "baseline_rate": 0.9, "margin": 0.05, **bad}
        with pytest.raises(ValueError):
            plan_noninferiority(**kw)
    with pytest.raises(ValueError):
        plan_noninferiority(1.5, 0.9, 0.05)


def test_plan_regress_cli(capsys):
    rc, out, _ = run(["plan-regress", "--margin", "0.05", "--baseline-rate", "0.9", "--json"], capsys)
    d = json.loads(out)
    assert rc == 0 and d["achievable"] and 400 < d["trials_per_arm"] < 520
    rc, _, _ = run(["plan-regress", "--margin", "0.05", "--baseline-rate", "0.9", "--candidate-rate", "0.8"], capsys)
    assert rc == 1
    rc, _, err = run(["plan-regress", "--margin", "2", "--baseline-rate", "0.9"], capsys)
    assert rc == 2


# --- sequential: lazy boundary equals the full table, and long streams are cheap ---------------

def test_lazy_boundary_matches_full_table():
    from passgate.sequential import _LazyBoundary, pass_boundary
    for theta in (0.5, 0.9, 0.99):
        for alpha in (0.01, 0.025, 0.2):
            table = pass_boundary(theta, alpha, 300)
            seq = _LazyBoundary(theta, alpha)
            assert [seq.need(n) for n in range(1, 301)] == table[1:]
            order = list(range(1, 301))
            random.Random(1).shuffle(order)
            shuffled = _LazyBoundary(theta, alpha)
            assert all(shuffled.need(n) == table[n] for n in order)


def test_run_sequential_matches_table_implementation():
    from passgate.sequential import _futility_floor, pass_boundary, run_sequential

    def reference(outcomes, th, al, m):
        table = pass_boundary(th, al, m)
        floor = _futility_floor(table, m)
        k = n = 0
        for o in outcomes:
            n += 1
            k += o
            need = table[n]
            if need is not None and k >= need:
                return ("pass", n, k)
            if k - n < floor[n] or n >= m:
                return ("fail", n, k)
        return ("undecided", n, k)

    rng = random.Random(7)
    for _ in range(150):
        th, al = rng.choice([0.5, 0.8, 0.9, 0.95]), rng.choice([0.025, 0.1])
        m, p = rng.choice([5, 30, 100, 300]), rng.choice([0.6, 0.85, 0.93, 0.97, 1.0])
        stream = [rng.random() < p for _ in range(rng.choice([3, 40, m, m + 50]))]
        r = run_sequential(stream, th, al, m)
        assert (r.decision, r.trials, r.passes) == reference(stream, th, al, m)


def test_sequential_with_big_budget_is_fast():
    from passgate.sequential import run_sequential
    t = time.process_time()
    r = run_sequential([True] * 200, 0.9, 0.025, 100_000)
    assert r.passed and time.process_time() - t < 3


# --- bound and the margin you can afford --------------------------------------------------------

def test_bound_clean_sweep_matches_rule(capsys):
    rc, out, _ = run(["bound", "30/30", "--json"], capsys)
    d = json.loads(out)["results"][0]
    assert rc == 0
    assert abs(d["exact_lower"] - 0.025 ** (1 / 30)) < 1e-9  # closed form for k = n
    assert d["exact_upper"] == 1.0
    rc, _, err = run(["bound", "31/30"], capsys)
    assert rc == 2


def test_min_provable_margin_is_consistent_with_power():
    from passgate import min_provable_margin
    m = min_provable_margin(300, 0.9)
    assert 0.04 < m < 0.08
    assert regress_power(300, 0.9, 0.9, m) >= 0.8 - 1e-9
    assert regress_power(300, 0.9, 0.9, m * 0.97) < 0.8 + 0.02
    assert min_provable_margin(3, 0.9) is None
    big = min_provable_margin(3000, 0.9)
    assert big < m  # more trials, smaller provable margin


def test_plan_regress_trials_mode(capsys):
    rc, out, _ = run(["plan-regress", "--baseline-rate", "0.9", "--trials", "300", "--json"], capsys)
    assert rc == 0 and 0.04 < json.loads(out)["min_margin"] < 0.08
    rc, _, err = run(["plan-regress", "--baseline-rate", "0.9"], capsys)
    assert rc == 2 and "exactly one" in err
