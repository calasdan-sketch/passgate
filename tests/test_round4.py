"""Round 4 tests: what a stranger who installs the package actually meets.

Covers Windows-shaped input (UTF-16 from PowerShell, unexpanded globs), console encodings, Ctrl-C,
write errors, JUnit encoding, the `example` command, and checks that README commands and metadata
agree with the real program (so the docs cannot drift unnoticed).
"""
import io
import pathlib
import re
import shlex
import sys
import xml.etree.ElementTree as ET

import pytest

import passgate
from passgate import Round, cli, evaluate_rounds, ingest
from passgate.junit import to_junit

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _fresh_stdin_cache():
    ingest._STDIN_CACHE.clear()
    yield
    ingest._STDIN_CACHE.clear()


def run(args, capsys):
    rc = cli.main(args)
    out = capsys.readouterr()
    return rc, out.out, out.err


GOOD = '[{"label": "a", "trials": 100, "passes": 99}]'


# ---- Windows-shaped input -------------------------------------------------------------------

@pytest.mark.parametrize("codec", ["utf-16", "utf-16-be"])
def test_utf16_with_bom_is_read(tmp_path, capsys, codec):
    """Windows PowerShell 5 writes UTF-16 with a BOM for `cmd > file`."""
    raw = GOOD.encode(codec)
    if codec == "utf-16-be":
        raw = b"\xfe\xff" + raw
    f = tmp_path / "r.json"
    f.write_bytes(raw)
    rc, out, _ = run(["gate", str(f), "--min-rounds", "1", "--threshold", "0.9"], capsys)
    assert rc == 0 and "PASS" in out


def test_utf16_csv_with_crlf(tmp_path, capsys):
    f = tmp_path / "r.csv"
    f.write_bytes("trials,passes,label\r\n100,99,x\r\n".encode("utf-16"))
    rc, out, _ = run(["gate", str(f), "--min-rounds", "1"], capsys)
    assert rc == 0


def test_utf32_bom_and_garbage_still_refused(tmp_path, capsys):
    f = tmp_path / "r.json"
    f.write_bytes(b"\xff\xfe\x00\x00" + GOOD.encode("utf-32-le"))
    rc, _, err = run(["gate", str(f)], capsys)
    assert rc == 2 and "Traceback" not in err
    f.write_bytes(b"\xff\xfe\xff\xff\xff")  # BOM, then an odd/invalid stream
    rc, _, err = run(["gate", str(f)], capsys)
    assert rc == 2 and "UTF" in err


def test_glob_pattern_is_expanded_when_the_shell_did_not(tmp_path, capsys, monkeypatch):
    for n in ("b", "a"):
        (tmp_path / f"{n}.json").write_text(GOOD, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    rc, out, _ = run(["gate", "*.json", "--min-rounds", "2", "--threshold", "0.9"], capsys)
    assert rc == 0 and "2/2 round(s)" in out
    rc, _, err = run(["gate", "nope*.json"], capsys)
    assert rc == 2 and "no file matches" in err


def test_literal_name_with_brackets_wins_over_glob(tmp_path, capsys, monkeypatch):
    (tmp_path / "a[1].json").write_text(GOOD, encoding="utf-8")
    (tmp_path / "a1.json").write_text(GOOD.replace("99", "5"), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    rc, out, _ = run(["gate", "a[1].json", "--min-rounds", "1"], capsys)
    assert rc == 0 and "1/1 round(s)" in out


def test_stdin_errors_say_standard_input_not_dash(monkeypatch, capsys):
    class Fake:
        buffer = io.BytesIO(b"[[5,3]]")
    monkeypatch.setattr(sys, "stdin", Fake())
    rc, _, err = run(["gate", "-"], capsys)
    assert rc == 2 and "standard input round 1" in err and "- round" not in err


# ---- console and file encodings -------------------------------------------------------------

def test_junit_is_ascii_and_round_trips():
    res = evaluate_rounds([Round(10, 10, label="café 日本")], threshold=0.5, min_rounds=1)
    xml = to_junit(res)
    xml.encode("ascii")
    assert xml.count("<?xml") == 1
    root = ET.fromstring(xml.split("\n", 1)[1])
    assert root.find(".//testcase").get("name").endswith("café 日本")


def test_legacy_console_keeps_the_character_code(monkeypatch):
    buf = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(buf, encoding="cp1252", errors="strict", newline=""))
    cli._emit("label café 日")
    sys.stdout.flush()
    assert buf.getvalue().decode("cp1252").strip() == "label café \\u65e5"


# ---- Ctrl-C and write errors ------------------------------------------------------------------

def test_ctrl_c_is_a_clean_130(monkeypatch, capsys):
    def boom(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(cli, "_gate_rounds", boom)
    rc, _, err = run(["gate", "x.json"], capsys)
    assert rc == 130 and "Traceback" not in err and "interrupted" in err


def test_write_failure_is_exit_2_and_plain(monkeypatch, capsys):
    import errno

    def boom(*a, **k):
        raise OSError(errno.ENOSPC, "No space left on device")
    monkeypatch.setattr(cli, "_emit", boom)
    rc, _, err = run(["plan", "--rate", "0.95", "--threshold", "0.9"], capsys)
    assert rc == 2 and "No space left" in err and "internal error" not in err


# ---- the example command ----------------------------------------------------------------------

@pytest.mark.parametrize("kind,cmd", [
    ("rounds", ["gate", "{f}", "--min-rounds", "3"]),
    ("paired", ["paired", "{f}"]),
    ("cases", ["cases", "{f}"]),
    ("outcomes", ["sequential", "{f}", "--threshold", "0.9"]),
])
def test_every_example_is_accepted_by_its_command(kind, cmd, tmp_path, capsys):
    f = tmp_path / "x.dat"
    rc, _, err = run(["example", kind, "-o", str(f)], capsys)
    assert rc == 0 and f.exists() and "synthetic" in err
    assert b"\r" not in f.read_bytes()
    rc, out, err = run([a.replace("{f}", str(f)) for a in cmd], capsys)
    assert rc in (0, 1), err


def test_example_never_overwrites(tmp_path, capsys):
    f = tmp_path / "keep.json"
    f.write_text("mine", encoding="utf-8")
    rc, _, err = run(["example", "rounds", "-o", str(f)], capsys)
    assert rc == 2 and f.read_text(encoding="utf-8") == "mine" and "will not overwrite" in err


def test_example_prints_to_screen_without_output_flag(capsys):
    rc, out, _ = run(["example", "rounds"], capsys)
    assert rc == 0 and out.lstrip().startswith("[")


# ---- docs and metadata cannot drift -----------------------------------------------------------

def _readme():
    return (ROOT / "README.md").read_text(encoding="utf-8")


def test_version_is_the_same_everywhere():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    v = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M).group(1)
    assert passgate.__version__ == v
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert re.search(rf"^## {re.escape(v)}\b", changelog, re.M), "CHANGELOG has no heading for this version"
    short = ".".join(v.split(".")[:2])
    assert f"passgate {short}" in _readme()


def test_metadata_points_somewhere_real():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[project.urls]" in pyproject
    for key in ("Homepage", "Issues", "Changelog"):
        assert re.search(rf"^{key}\s*=\s*\"https://", pyproject, re.M)


def _blocks(text):
    return re.findall(r"^```(\w*)\n(.*?)^```", text, re.M | re.S)


def _fixtures(tmp):
    (tmp / "rounds.json").write_text(cli._EXAMPLES["rounds"], encoding="utf-8")
    (tmp / "results.csv").write_text(cli._EXAMPLES["paired"], encoding="utf-8")
    (tmp / "cases.csv").write_text(cli._EXAMPLES["cases"], encoding="utf-8")
    (tmp / "outcomes.txt").write_text(cli._EXAMPLES["outcomes"], encoding="utf-8")


def _readme_commands():
    """(command, expected output lines) for every one-line `passgate ...` in a bash block that is followed by # comments."""
    found = []
    for lang, body in _blocks(_readme()):
        if lang != "bash":
            continue
        lines = body.splitlines()
        for i, line in enumerate(lines):
            if not line.startswith("passgate ") or any(c in line.split("#")[0] for c in "|<>"):
                continue
            exp = []
            for nxt in lines[i + 1:]:
                if not nxt.startswith("# "):
                    break
                exp.append(nxt[2:].rstrip())
            if exp:
                found.append((line, exp))
    return found


def test_readme_has_commands_to_check():
    assert len(_readme_commands()) >= 8


@pytest.mark.parametrize("command,expected", _readme_commands(), ids=lambda v: v if isinstance(v, str) else "out")
def test_readme_output_comments_match_the_program(command, expected, tmp_path, monkeypatch, capsys):
    _fixtures(tmp_path)
    monkeypatch.chdir(tmp_path)
    rc = cli.main(shlex.split(command, comments=True)[1:])
    got = capsys.readouterr().out.splitlines()
    assert rc in (0, 1), command
    first = got[0].rstrip()
    ok = False
    for exp in expected:
        if exp.startswith("... "):
            ok = ok or exp[4:] in first
        else:
            ok = ok or exp == first
    assert ok, f"README shows {expected!r} but the program printed {first!r}"


def test_readme_quick_start_output_block_matches(tmp_path, monkeypatch, capsys):
    blocks = _blocks(_readme())
    idx = next(i for i, (lang, body) in enumerate(blocks) if lang == "bash" and "passgate gate rounds.json --threshold 0.9 --min-rounds 3" in body)
    shown = blocks[idx + 1][1].rstrip("\n")
    _fixtures(tmp_path)
    monkeypatch.chdir(tmp_path)
    cli.main(["gate", "rounds.json", "--threshold", "0.9", "--min-rounds", "3"])
    assert capsys.readouterr().out.rstrip("\n") == shown


def test_readme_heredoc_matches_the_example_command():
    body = next(b for lang, b in _blocks(_readme()) if lang == "bash" and "cat > rounds.json" in b)
    shown = body.split("<<'EOF'\n", 1)[1].split("EOF", 1)[0]
    assert shown == cli._EXAMPLES["rounds"]


def test_example_dash_means_screen_and_missing_folder_is_plain(tmp_path, capsys):
    rc, out, _ = run(["example", "cases", "-o", "-"], capsys)
    assert rc == 0 and out.startswith("case,") and not (tmp_path / "-").exists()
    rc, _, err = run(["example", "rounds", "-o", str(tmp_path / "no" / "such" / "x.json")], capsys)
    assert rc == 2 and "Traceback" not in err and "cannot read or write" in err


def test_lone_surrogate_utf16_is_refused_cleanly(tmp_path, capsys):
    f = tmp_path / "r.json"
    f.write_bytes(b"\xff\xfe[\x00\x00\xd8")
    rc, _, err = run(["gate", str(f)], capsys)
    assert rc == 2 and "Traceback" not in err
