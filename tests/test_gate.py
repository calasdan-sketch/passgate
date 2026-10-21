"""Gate logic and CLI tests."""
import json

import pytest

from passgate import Round, evaluate_rounds, compare
from passgate.cli import main


def rounds(*ratios, label=""):
    return [Round(passes=k, trials=n, label=f"{label}{i+1}") for i, (k, n) in enumerate(ratios)]


def test_strong_candidate_passes():
    # wilson_lower(93, 100, 0.95) ~= 0.863 -> clears 0.85 on every round
    result = evaluate_rounds(rounds((93, 100), (95, 100), (92, 100)), threshold=0.85)
    assert result.passed, str(result)


def test_flaky_candidate_fails_even_with_high_average():
    # average pass rate 89%, but one weak round kills the run — the pass^N property
    result = evaluate_rounds(rounds((99, 100), (99, 100), (70, 100)), threshold=0.85)
    assert not result.passed
    assert not result.rounds[2].meets


def test_threshold_too_high_for_sample_size():
    # 93/100 is genuinely good but 100 trials cannot certify >= 0.9 at 95%
    assert not evaluate_rounds(rounds((93, 100)), threshold=0.9).passed
    # ... while 1000 trials with the same rate can
    assert evaluate_rounds(rounds((930, 1000)), threshold=0.9).passed


def test_min_rounds_enforced():
    result = evaluate_rounds(rounds((95, 100)), threshold=0.85, min_rounds=3)
    assert not result.passed
    assert "min 3" in str(result)


def test_round_validation():
    with pytest.raises(ValueError):
        Round(trials=0, passes=0)
    with pytest.raises(ValueError):
        Round(trials=10, passes=11)
    with pytest.raises(ValueError):
        evaluate_rounds([], threshold=1.5)


def test_report_str_and_dict():
    result = evaluate_rounds(rounds((99, 100), (70, 100), label="r"), threshold=0.85)
    text = str(result)
    assert "verdict: FAIL" in text and "r2" in text
    data = result.as_dict()
    assert data["passed"] is False and len(data["rounds"]) == 2


def test_compare_candidate_better():
    cmp_ = compare(rounds((93, 100)), rounds((85, 100)), alpha=0.05)
    assert cmp_.verdict == "candidate-better"
    assert cmp_.better and cmp_.p_one_sided < 0.05
    assert 0.0 < cmp_.fisher_p < 0.2  # exact test is stricter than the z-test


def test_compare_inconclusive_on_tiny_samples():
    cmp_ = compare(rounds((3, 3)), rounds((2, 3)), alpha=0.05)
    assert cmp_.verdict == "inconclusive"


def test_cli_gate_json_pass(tmp_path, capsys):
    f = tmp_path / "rounds.json"
    f.write_text(json.dumps([{"trials": 100, "passes": 93}] * 3), encoding="utf-8")
    code = main(["gate", str(f), "--threshold", "0.85", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["passed"] is True


def test_cli_gate_csv_fail(tmp_path, capsys):
    f = tmp_path / "rounds.csv"
    f.write_text("trials,passes\n100,70\n100,99\n", encoding="utf-8")
    assert main(["gate", str(f), "--threshold", "0.85"]) == 1
    assert "verdict: FAIL" in capsys.readouterr().out


def test_cli_pairs_format(tmp_path, capsys):
    f = tmp_path / "rounds.json"
    f.write_text(json.dumps([[99, 100], [98, 100]]), encoding="utf-8")
    assert main(["gate", str(f), "--threshold", "0.9", "--min-rounds", "2"]) == 0


def test_cli_compare(capsys):
    code = main(["compare", "--candidate", "93/100", "--baseline", "85/100"])
    assert code == 0
    assert "candidate-better" in capsys.readouterr().out


def test_cli_missing_file(capsys):
    with pytest.raises(SystemExit):
        main(["gate", "no-such-file.json"])
