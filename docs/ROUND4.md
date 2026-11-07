# Round 4: the stranger's first hour

Question for this round: what does a person who has never seen this repository get when they type `pip install passgate` and follow the README? Rounds 1 to 3 made the statistics and the hostile-input handling solid; round 4 looked at packaging, first-run experience, Windows and the docs.

## How the review was done (honest version)

One agent played every specialist role below, one after the other, and ran real commands for each. No independent sub-agents or human reviewers were available in this run, so the "panel" is one reviewer wearing different hats. Treat the findings as one careful pass, not as independent confirmation. Everything not listed under "what was run" is marked untested.

Numbers were measured on a shared two-core machine under heavy load (load average 13 to 16), so times are upper bounds.

## Baseline (before any change)

| check | result |
|---|---|
| `pytest -q` | 241 passed, 92 s |
| `ruff check .` | clean |
| `mypy src/passgate` (run once in a scratch venv, not configured in the repo) | no issues in 13 files |
| `python benchmarks/bench_gates.py` | 128 s; the committed tables reproduce exactly (only the NumPy version string in the header differs: 2.5.3 to 2.5.4) |
| build | `python -m build` gives a 61 KB wheel and a 106 KB sdist; `twine check --strict` passes on both |
| wheel contents | package modules, `py.typed`, `LICENSE`, `METADATA`, entry point; no tests, no personal paths (grepped for home directories, `claude`, `session_`) |
| installed in fresh venvs (not editable) | Python 3.11.18, 3.12.3, 3.13.18: `passgate --version`, `--help`, every README quick-start command ran |
| declared dependencies | none; importing in a clean venv works, so the list is complete and minimal |
| PyPI (read-only `pip index versions passgate`) | only 0.1.1 is published |

## Specialists and what each found

1. **First-time user (README only, clean venv).** Ran the quick start from the wheel in a directory with a space in its name (`passgate gate rounds.json --threshold 0.9 --min-rounds 3`, `plan`, `compare`, `regress`, `plan-regress`, `cases`, `paired`, `sequential`, `bound`): all output matched the README. Then installed the *published* 0.1.1 into a fresh venv and ran the same commands: `gate` works, `passgate plan` fails with `invalid choice: 'plan' (choose from 'gate', 'compare')`. So the README's `pip install passgate` leads to a broken first hour today. Fixed with a release-status note. The `cat > rounds.json <<'EOF'` heredoc does not work in cmd.exe or PowerShell; fixed with `passgate example`.
2. **Release/packaging engineer.** `tar tzf` on the sdist: no `CHANGELOG.md`, `scripts/`, `benchmarks/`. Extracted the sdist and ran its tests: `test_review_fixes.py` needs CHANGELOG and scripts. Fixed with `MANIFEST.in`; the sdist tests (62 selected tests) pass afterwards. `project.urls` was missing, so the PyPI page would have no link to the repository. Fixed. Version agrees in `pyproject.toml`, `__version__`, CHANGELOG heading and the README table; now enforced by a test. Classifiers claim Python 3.9 to 3.13; 3.9 and 3.10 are not installed here, so only the syntax was checked for 3.9 (`ast.parse(feature_version=(3, 9))` on all sources, plus a grep for 3.10-only features) and 3.11 to 3.13 were run.
3. **Supply-chain reviewer.** Zero runtime dependencies; build backend is `setuptools>=77` only. Standard-library imports only (no `subprocess`, `socket`, `urllib`, `eval`, `pickle` in `src/`). CI already uses `permissions: contents: read`. Not fixed: the workflow and `action.yml` pin actions by tag (`actions/checkout@v4`, `actions/setup-python@v5`), not by commit hash; the hashes cannot be looked up offline. No Trusted Publishing workflow exists (see ideas). The `LICENSE` and package author are the GitHub handle `calasdan-sketch`; that is the owner's call (see PUBLISH-CHECKLIST).
4. **Docs/DX writer.** The `--help` epilog listed three exit codes and omitted that `compare`, `paired` and `regress` can use another code for "inconclusive". Several README examples had no output shown. Added exit codes to `--help` and the README, added output lines for `bound`, `paired` and both `plan-regress` forms (the old README put the 451-trial output under the wrong command), and a test that compares every README output line with the real program.
5. **CI/CD integrator.** CI tested only Python 3.9 and 3.12 and never installed the *built* wheel. Added Python 3.10, 3.11, 3.13, a ruff step and `scripts/smoke_wheel.py` (fresh venv, path with a space, runs the quick start). Not run on GitHub runners from here. `passgate gate runs\*.json` fails on Windows shells because they never expand `*`; fixed inside `gate`.
6. **Hostile-input fuzzer.** `fuzz.py` (scratch, not committed): 7,924 command runs over about 340 payloads (empty, truncated, NUL, invalid UTF-8, nested 100,000 deep, lone surrogates, escape codes, bidi, random bytes, UTF-16) times 14 command shapes and two extensions. Result: no traceback, no "internal error", exit codes only 0, 1 or 2, no raw control characters in output (5 reports were the fuzzer's own false positive: a literal `U+202E (as escaped text)` text). By hand: Ctrl-C printed a traceback (fixed, exit 130); `passgate plan ... > /dev/full` printed "internal error ... please report it" (fixed, plain message, exit 2); UTF-16 files from PowerShell were refused (fixed).
7. **Performance engineer.** 200,000 rounds through `gate`: about 2.1 s CPU (10 to 14 s wall under load); output is one line per round. Start-up of `passgate --version`: about 60 ms CPU per run, 200 ms wall under load; `import passgate` loads everything (about 97 ms in `-X importtime` under load). Not changed: lazy imports would save little and complicate the API. The Monte Carlo benchmark is unchanged because no statistics code changed.
8. **Competitor's maintainer.** Re-checked the two strongest competitors from the README table with `pip index versions`: errorbars 0.2.4 and evalpower 0.1.0 are still the newest versions, so the sourced comparison table (checked 2026-11-07) is still current. The pages and source were not re-read in this round (marked unverified for anything newer than that date). What they do better is still the same: errorbars has plots and bootstrap intervals; evalpower audits whether a model-task cell is stochastic.
9. **Windows reviewer (read the code, nothing was run on Windows).** `grep` for `shell=True`, `os.replace`, `signal`, `os.sep`, `encoding` in `src/`: none of the first three; files are read as bytes and decoded explicitly; the step summary is opened with `encoding="utf-8"`. Gaps found and fixed: UTF-16 input, unexpanded globs, heredoc in the quick start, JUnit mojibake and `?` replacement on code-page consoles (simulated with `PYTHONIOENCODING=ascii` and `cp1252` on Linux). Not verifiable here: real console behaviour, `windows-latest` CI, the `os.path.basename` rule on backslash paths.

## What was built or fixed

All listed in `CHANGELOG.md` under "Round 4". Short form: README release-status note and working Windows quick start; `passgate example`; UTF-16 and glob handling; Ctrl-C and write errors; ASCII-safe JUnit; character codes instead of `?`; `standard input` in messages; `MANIFEST.in`; project URLs; CI matrix, lint and wheel smoke; 32 new tests (241 to 273) including README-versus-program checks.

## Scored ideas

Scores 1 (low) to 5 (high) for user value and differentiation; effort and risk 1 (low) to 5 (high). Status DONE or NOT-DONE (reason).

| # | idea | value | diff. | effort | risk | status |
|---|---|---|---|---|---|---|
| 1 | README says plainly that PyPI is older than the repo | 5 | 1 | 1 | 1 | DONE |
| 2 | `passgate example` (no heredoc needed on Windows) | 4 | 3 | 1 | 1 | DONE |
| 3 | Test that README output comments equal real output | 4 | 3 | 2 | 1 | DONE |
| 4 | Fresh-venv wheel smoke script, run in CI | 4 | 2 | 2 | 1 | DONE (CI run untested) |
| 5 | `MANIFEST.in` so the sdist can run its own tests | 3 | 1 | 1 | 1 | DONE |
| 6 | Project URLs in package metadata | 3 | 1 | 1 | 1 | DONE |
| 7 | Ctrl-C exits 130 without a traceback | 3 | 1 | 1 | 1 | DONE |
| 8 | Write failures (full disk) are plain exit 2 | 3 | 1 | 1 | 1 | DONE |
| 9 | UTF-16 (PowerShell 5 redirect) input | 4 | 2 | 1 | 2 | DONE |
| 10 | Expand `*.json` where the shell does not | 4 | 2 | 2 | 2 | DONE |
| 11 | ASCII-safe JUnit XML | 3 | 1 | 1 | 1 | DONE |
| 12 | Character code instead of `?` on narrow consoles | 2 | 1 | 1 | 1 | DONE |
| 13 | Exit codes in `--help` and README | 3 | 1 | 1 | 1 | DONE |
| 14 | CI on 3.10 to 3.13 plus ruff | 3 | 1 | 1 | 1 | DONE (untested on runners) |
| 15 | Version-agreement test (pyproject, `__version__`, CHANGELOG, README) | 3 | 1 | 1 | 1 | DONE |
| 16 | Docs folder and invisible-character scan cover new files | 2 | 1 | 1 | 1 | DONE |
| 17 | `mypy` configured and run in CI | 2 | 1 | 2 | 2 | NOT-DONE (ran clean once by hand; adding a pinned tool to CI is the owner's call) |
| 18 | Pin GitHub Actions by commit hash | 3 | 1 | 2 | 2 | NOT-DONE (cannot look up hashes offline; do it with Dependabot) |
| 19 | Trusted Publishing release workflow | 4 | 1 | 3 | 4 | NOT-DONE (publishing was off limits; needs owner's PyPI setup) |
| 20 | SARIF output | 2 | 2 | 3 | 2 | NOT-DONE (no code locations in a statistical gate; low fit) |
| 21 | Per-cell stochasticity audit (evalpower's feature) | 3 | 3 | 4 | 3 | NOT-DONE (needs design of the statistic) |
| 22 | Bootstrap intervals and plots (errorbars' feature) | 2 | 2 | 4 | 3 | NOT-DONE (would break the no-dependency promise for plots) |
| 23 | Paired-design planning for `regress` | 3 | 3 | 4 | 3 | NOT-DONE (needs a model of the discordant rate) |
| 24 | Judge-noise modelling | 3 | 4 | 5 | 4 | NOT-DONE (research problem) |
| 25 | Lazy imports to cut start-up | 1 | 1 | 2 | 2 | NOT-DONE (gain too small to measure under load) |
| 26 | Shell completion | 1 | 1 | 2 | 1 | NOT-DONE |
| 27 | Windows run of the whole suite | 5 | 1 | 1 | 1 | NOT-DONE (no Windows here; the CI job exists) |
| 28 | Python 3.9 and 3.10 run | 4 | 1 | 1 | 1 | NOT-DONE (not installed; CI covers them) |

## Could not test

- Anything on Windows (real console, PowerShell 5 redirect, `windows-latest` CI). The UTF-16 and glob fixes are tested on Linux with synthetic bytes.
- Python 3.9 and 3.10 (only syntax was checked for 3.9).
- The GitHub Action and the new CI steps on GitHub's runners.
- Whether the GitHub repository is public; the README install line `pip install git+https://github.com/calasdan-sketch/passgate.git` assumes it is.
- Whether the PyPI name `passgate` will accept the upload (only the read-only listing was consulted).
- Competitor pages and source were not re-read, only their latest version numbers.
