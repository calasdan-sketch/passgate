"""Edge cases found in the SOP review: exit codes, BOM/CRLF, bad input, API validation."""
import json
import subprocess
import sys

import pytest

from passgate import Round, compare, evaluate_rounds
from passgate.cli import load_rounds, main
from passgate.stats import fisher_exact_2x2, two_proportion_z


def test_bad_ratio_exits_2_not_1(capsys):
    # exit 1 means "gate failed" in CI; bad input must be distinguishable
    assert main(["compare", "--candidate", "oops", "--baseline", "5/10"]) == 2
    assert "bad round" in capsys.readouterr().err


def test_missing_file_exits_2(capsys):
    assert main(["gate", "no-such-file.json"]) == 2
    assert "not found" in capsys.readouterr().err


def test_invalid_json_exits_2(tmp_path, capsys):
    f = tmp_path / "r.json"
    f.write_text("{not json", encoding="utf-8")
    assert main(["gate", str(f)]) == 2
    assert "not valid JSON" in capsys.readouterr().err


def test_json_not_a_list(tmp_path):
    f = tmp_path / "r.json"
    f.write_text('{"trials": 10, "passes": 9}', encoding="utf-8")
    assert main(["gate", str(f)]) == 2


def test_missing_keys_exit_2(tmp_path, capsys):
    f = tmp_path / "r.json"
    f.write_text('[{"trials": 10}]', encoding="utf-8")
    assert main(["gate", str(f)]) == 2
    assert "passes" in capsys.readouterr().err


def test_passes_over_trials_exit_2(tmp_path):
    f = tmp_path / "r.json"
    f.write_text("[[11, 10]]", encoding="utf-8")
    assert main(["gate", str(f)]) == 2


def test_bom_and_crlf_csv(tmp_path):
    f = tmp_path / "r.csv"
    f.write_bytes(b"\xef\xbb\xbftrials,passes,label\r\n100,99,a\r\n100,98,b\r\n")
    rounds = load_rounds(str(f))
    assert [r.label for r in rounds] == ["a", "b"] and rounds[0].passes == 99


def test_bom_json(tmp_path):
    f = tmp_path / "r.json"
    f.write_bytes(b"\xef\xbb\xbf" + json.dumps([[9, 10]]).encode())
    assert load_rounds(str(f))[0].trials == 10


def test_csv_alt_columns_n_k(tmp_path):
    f = tmp_path / "r.csv"
    f.write_text("n,k\n50,49\n", encoding="utf-8")
    r = load_rounds(str(f))[0]
    assert (r.trials, r.passes) == (50, 49)


def test_csv_missing_column_exit_2(tmp_path):
    f = tmp_path / "r.csv"
    f.write_text("trials\n50\n", encoding="utf-8")
    assert main(["gate", str(f)]) == 2


def test_bad_threshold_exit_2(tmp_path, capsys):
    f = tmp_path / "r.json"
    f.write_text("[[9, 10]]", encoding="utf-8")
    assert main(["gate", str(f), "--threshold", "1.5"]) == 2
    assert main(["gate", str(f), "--confidence", "2"]) == 2


def test_bad_alpha_exit_2():
    assert main(["compare", "--candidate", "9/10", "--baseline", "5/10", "--alpha", "0"]) == 2


def test_empty_rounds_file_fails_gate(tmp_path):
    f = tmp_path / "r.json"
    f.write_text("[]", encoding="utf-8")
    assert main(["gate", str(f)]) == 1  # zero rounds can never satisfy min_rounds


def test_compare_json_is_valid_json_when_infinite(capsys):
    # zero-variance case used to emit the non-standard token Infinity
    assert main(["compare", "--candidate", "10/10", "--baseline", "0/10", "--json"]) == 0
    out = capsys.readouterr().out
    json.loads(out, parse_constant=lambda c: pytest.fail(f"non-standard JSON constant {c}"))


def test_compare_empty_inputs_raise():
    with pytest.raises(ValueError):
        compare([], [Round(5, 10)])


def test_evaluate_rounds_validates_confidence():
    with pytest.raises(ValueError):
        evaluate_rounds([], confidence=1.5)


def test_p_value_precise_for_large_z():
    _, p = two_proportion_z(1000, 1000, 500, 1000)
    assert 0.0 <= p < 1e-100  # 1 - cdf(z) would have rounded to exactly 0.0 or lost precision


def test_fisher_empty_table():
    with pytest.raises(ValueError):
        fisher_exact_2x2(0, 0, 0, 0)


def test_output_is_ascii_safe():
    r = evaluate_rounds([Round(100, 93)], threshold=0.5)
    assert str(r).isascii()
    c = compare([Round(100, 93)], [Round(100, 85)])
    assert str(c).isascii()


def test_python_dash_m_entry():
    proc = subprocess.run(
        [sys.executable, "-m", "passgate", "compare", "--candidate", "93/100", "--baseline", "85/100"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0 and "candidate-better" in proc.stdout
