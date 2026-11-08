"""Mutation-style sanity check: break key lines of the source one at a time and confirm the tests notice.

Usage: python scripts/mutation_check.py [--all-tests]

Copies ``src/passgate`` to a temporary folder, applies ONE edit at a time, and runs a fast subset of the
test suite against the edited copy. A mutant the tests fail on is "killed" (good); one that still passes is
"survived" (a gap in the tests, or an edit that does not change behaviour). Nothing in the repository is
modified. Exit code 1 if any mutant survives.
"""
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
FAST = ["tests/test_stats.py", "tests/test_gate.py", "tests/test_round5.py", "tests/test_synapse1.py"]

# (file, text that must occur exactly once, replacement, what the edit breaks)
MUTANTS = [
    ("gate.py", "meets = lower >= threshold", "meets = lower > threshold", "gate bar: >= becomes >"),
    ("gate.py", "meets = lower >= threshold", "meets = True", "gate always passes"),
    ("cluster.py", "return self.lower >= self.threshold", "return self.lower > self.threshold", "items bar: >= becomes >"),
    ("stats.py", "if lower_bound(trials, trials, confidence, method) < threshold:", "if lower_bound(trials, trials, confidence, method) <= threshold:", "min_passes edge"),
    ("stats.py", "return _NORMAL.inv_cdf(0.5 + confidence / 2.0)", "return _NORMAL.inv_cdf(confidence)", "z uses one-sided level"),
    ("stats.py", "denom = 1.0 + z * z / trials", "denom = 1.0", "Wilson without its correction"),
    ("stats.py", "return max(0.0, min(phat, center - half)), min(1.0, max(phat, center + half))", "return center - half, center + half", "Wilson clamps removed"),
    ("stats.py", "if k >= n:\n        return (alpha / 2.0) ** (1.0 / n)", "if k >= n:\n        return (alpha) ** (1.0 / n)", "Clopper-Pearson k=n lower uses alpha, not alpha/2"),
    ("cluster.py", "if passes < k:\n        return 0.0", "if passes <= k:\n        return 0.0", "pass^k edge"),
    ("cluster.py", "if fails < k:\n        return 1.0", "if fails <= k:\n        return 1.0", "pass@k edge"),
    ("cluster.py", "n_eff = float(g)\n    n_eff = min(max_neff, max(float(g), n_eff))", "n_eff = max_neff\n    n_eff = min(max_neff, max(float(g), n_eff))", "zero-spread items trusted again"),
    ("cluster.py", "n_eff = min(float(m_tot), max(float(g), n_eff))", "n_eff = float(m_tot)", "design effect ignored"),
    ("cluster.py", "var = (g / (g - 1.0)) * ss / (m_tot * m_tot)", "var = ss / (m_tot * m_tot)", "cluster variance loses g/(g-1)"),
    ("stats.py", "return 1.0, 1.0\n    if m > EXACT_MAX_TRIALS:", "return 0.0, 0.0\n    if m > EXACT_MAX_TRIALS:", "McNemar with no discordant pairs"),
    ("gate.py", "return len(self.rounds) >= max(1, self.min_rounds) and", "return len(self.rounds) > max(1, self.min_rounds) and", "min_rounds edge (verdict)"),
]


def main() -> int:
    tests = ["tests"] if "--all-tests" in sys.argv else FAST
    work = pathlib.Path(tempfile.mkdtemp(prefix="passgate-mut-"))
    survived = []
    try:
        for i, (fname, old, new, what) in enumerate(MUTANTS):
            dst = work / f"m{i}"
            shutil.copytree(ROOT / "src" / "passgate", dst / "passgate", ignore=shutil.ignore_patterns("__pycache__"))
            path = dst / "passgate" / fname
            text = path.read_text(encoding="utf-8")
            if text.count(old) != 1:
                print(f"SKIP  {what}: expected one match in {fname}, found {text.count(old)}")
                survived.append(what + " (pattern drifted)")
                continue
            path.write_text(text.replace(old, new), encoding="utf-8")
            env = dict(os.environ, PYTHONPATH=str(dst), PYTHONDONTWRITEBYTECODE="1")
            probe = subprocess.run([sys.executable, "-c", "import passgate;print(passgate.__file__)"], env=env, capture_output=True, text=True, cwd=ROOT)
            if str(dst) not in probe.stdout:
                print("ERROR the mutant copy is not the one being imported:", probe.stdout.strip())
                return 2
            r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *tests], env=env, cwd=ROOT, capture_output=True, text=True)
            killed = r.returncode != 0
            print(("killed   " if killed else "SURVIVED ") + what)
            if not killed:
                survived.append(what)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"\n{len(MUTANTS) - len(survived)} of {len(MUTANTS)} mutants killed")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
