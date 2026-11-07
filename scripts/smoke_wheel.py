"""Install a built wheel into a brand-new virtual environment and run what a stranger would run.

    python scripts/smoke_wheel.py dist/passgate-X.Y.Z-py3-none-any.whl

Offline (``--no-index``): passgate has no dependencies. Works on Windows, macOS and Linux. Run it from
any directory: the check that ``import passgate`` comes from the venv (not a checkout) catches a wheel
that only works because the source tree happens to be next to it.
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import tempfile
import venv

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    hits = sorted(glob.glob(argv[1])) or [argv[1]]  # cmd.exe and PowerShell do not expand *.whl
    wheel = os.path.abspath(hits[-1])
    failures = 0
    with tempfile.TemporaryDirectory(prefix="passgate smoke ") as tmp:  # a space in the path on purpose
        env_dir = os.path.join(tmp, "venv")
        venv.EnvBuilder(with_pip=True, clear=True).create(env_dir)
        bindir = os.path.join(env_dir, "Scripts" if os.name == "nt" else "bin")
        py = os.path.join(bindir, "python" + (".exe" if os.name == "nt" else ""))
        exe = os.path.join(bindir, "passgate" + (".exe" if os.name == "nt" else ""))
        work = os.path.join(tmp, "work dir")
        os.mkdir(work)

        def run(*cmd, stdin=None):
            return subprocess.run(
                list(cmd), cwd=work, input=stdin, capture_output=True, text=True, encoding="utf-8", errors="replace"
            )

        r = run(py, "-m", "pip", "install", "--no-index", "--no-deps", "--quiet", wheel)
        if r.returncode:
            print("install failed:\n" + r.stdout + r.stderr)
            return 1

        def expect(name, res, code, *needles):
            nonlocal failures
            ok = res.returncode == code and all(n in res.stdout + res.stderr for n in needles)
            ok = ok and "Traceback" not in res.stderr
            print(("ok   " if ok else "FAIL ") + name + ("" if ok else f"  (exit {res.returncode})\n{res.stdout}{res.stderr}"))
            failures += not ok

        r = run(py, "-I", "-c", "import passgate, sys; print(passgate.__file__)")
        expect("imports from the venv", r, 0, "site-packages")
        r = run(py, "-I", "-c", "import importlib.resources as r; print(r.files('passgate').joinpath('py.typed').is_file())")
        expect("py.typed is installed", r, 0, "True")
        expect("--version", run(exe, "--version"), 0, "passgate ")
        expect("python -m passgate --version", run(py, "-m", "passgate", "--version"), 0, "passgate ")
        expect("--help", run(exe, "--help"), 0, "usage: passgate", "Exit codes")
        expect("example writes a file", run(exe, "example", "rounds", "-o", "rounds.json"), 0, "wrote")
        expect(
            "quick start: gate fails the weak rounds (exit 1)",
            run(exe, "gate", "rounds.json", "--threshold", "0.9", "--min-rounds", "3"), 1,
            "1/3 round(s) clear", "verdict: FAIL", "needs 398 trials",
        )
        expect("plan", run(exe, "plan", "--rate", "0.95", "--threshold", "0.9", "--rounds", "3"), 0, "run 140 trials per round")
        expect("compare", run(exe, "compare", "--candidate", "93/100", "--baseline", "85/100"), 0, "candidate-better")
        expect("bound", run(exe, "bound", "30/30"), 0, "11.6%")
        expect("json output", run(exe, "gate", "rounds.json", "--min-rounds", "3", "--json"), 1, '"passgate_version"')
        expect("junit output", run(exe, "gate", "rounds.json", "--min-rounds", "3", "--format", "junit"), 1, "<testsuites>")
        expect("stdin", run(exe, "gate", "-", "--min-rounds", "1", stdin='[[85, 100]]'), 1, "FAIL")
        expect("bad input is exit 2, not a traceback", run(exe, "gate", "missing.json"), 2, "not found")
        expect("missing command is exit 2", run(exe), 2)
    print("smoke: " + ("all checks passed" if not failures else f"{failures} check(s) FAILED"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
