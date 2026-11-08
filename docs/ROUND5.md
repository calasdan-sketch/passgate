# Round 5: statistics and reproducibility audit

Question for this round: are the numbers right, and would two machines give the same verdict? Rounds 1 to 4 covered the statistics, hostile input and the first hour; round 5 attacked the maths itself, the upgrade from the published 0.1.1, and what a release manager wires into CI.

## How the review was done (honest version)

One agent played four specialists in turn (ML-evaluation statistician, LLM-evaluation engineer, release-gating/CI engineer, numerical-reproducibility reviewer) and ran real commands for each. No independent reviewers were available, so treat this as one careful pass. Timings come from a shared, loaded machine.

## Baseline (before any change)

| check | result |
|---|---|
| `pytest -q` | 273 passed, 92 s |
| `ruff check .` | clean |
| `mypy src/passgate` | no issues in 13 files (run by hand; not configured in the repo) |
| `python benchmarks/bench_gates.py` | tables identical to the committed ones (re-run after the changes: also identical, so no committed number moved) |
| PyPI (read-only `pip index versions`) | passgate 0.1.1 only; errorbars 0.2.4 and evalpower 0.1.0 still newest |

After: 307 passed (34 new), ruff clean, mypy clean, mutation check 15 of 15 killed.

## Specialists and what each found

1. **ML-evaluation statistician.**
   - Command: `python -c "from passgate import *; print(evaluate_items([Item(10,10)]*3, k=2, threshold=.9))"`. Result: lower bound 1.000, PASS. Three items that all passed prove nothing, but the t-interval over identical numbers has zero width. Fail-open. Fixed (Wilson at an effective sample size from the spread of item values, floored at one observation per item, capped at the usable trials).
   - Command: `python benchmarks/bench_items_k.py`. Coverage of the lower end of the pass^2 interval (nominal 0.975): the old t-interval gave 0.869 to 0.965 with skewed item difficulty (Beta(19,1), 5 to 100 items); the new one gives 0.973 to 1.000. With Beta(2,2) both are at or above nominal. Table in `benchmarks/RESULTS-items-k.md`. Not tested: other tries-per-item than 10, other difficulty shapes.
   - The Wilson and Clopper-Pearson "95%" bounds are two-sided, so each is a 97.5% one-sided bound. This was already stated in the README limits; unchanged.
2. **LLM-evaluation engineer.**
   - Command: `passgate items big.csv --k 100000` with one item of 10^12 trials. Before the fix, `evaluate_items([Item(10**9, 5*10**8)], k=10**5)` took 5.4 s wall (huge integers in `math.comb`); now 0.8 s for the whole CLI call, and the estimator itself takes microseconds. `pass_at_k` also loses digits when the answer is tiny (`1 - 0.999999997`); it now uses `expm1`/`log1p` (3.0e-9 for 1 pass in 10^9 trials with k=3).
   - `pass_hat_k(5, 6, 2)` (more passes than trials) silently returned a number; it now raises. Brute-force enumeration of every subset for m up to 8 matches both estimators to 1e-15 (`test_pass_at_k_and_pass_hat_k_match_brute_force_enumeration`).
3. **Release-gating/CI engineer.**
   - The GitHub Action had no outputs, so a later workflow step could not branch on the result without parsing the log. Added `exit-code` and `report-path` (written before the step exits, so a failing gate still sets them). The YAML parses; not run on a GitHub runner.
   - Upgrade path: installed 0.1.1 from the registry into a scratch venv and ran 120 random `gate --json` inputs (list, `{"rounds": ...}`, CSV, BOM, CRLF) through both versions: 0 differences in exit code or per-round verdict. 100 random `compare` pairs: 29 differ, all of them the documented `inconclusive` to `baseline-better` change (e.g. candidate 9/152 against baseline 60/60 was "inconclusive" in 0.1.1). New output only adds fields (`passgate_version`, `method`, `lower`, `trials_needed`). No migration is needed: 0.1.1 had no config or state files.
   - Wheel and sdist: the sdist did not contain the new `tests/golden` files, so its own tests failed. Fixed in `MANIFEST.in`; the sdist's tests now pass when run from an extracted sdist against a separately installed wheel (96 passed, 2 skipped).
4. **Numerical-reproducibility reviewer.**
   - Hypothesis (`derandomize=True`) found `wilson_interval(1, 1)` returning an upper bound of 0.9999999999999999, below the observed rate. Fixed by clamping so the observed rate is always inside its interval.
   - Differential tests: Wilson against 60-digit `decimal` arithmetic (400 random cases, 1e-12); Clopper-Pearson endpoints against their defining equation in exact `Fraction` binomial sums; McNemar and Fisher against exact enumeration. All agree.
   - Boundary stability: `test_no_verdict_sits_within_float_noise_of_the_bar` scans five bars and every (passes, trials) with trials below 400 and finds no Wilson lower bound within 1e-9 of a bar, so a different libm cannot flip those verdicts. Larger trial counts were not scanned.
   - Locale and parallelism: the same `items --json` output (identical md5) under `LC_ALL=C`, `tr_TR.UTF-8` and `de_DE.UTF-8` (if those locales are missing the machine falls back to C, so this proves less than it seems), and from six parallel runs. No random numbers in the library (`random` is never imported in the statistics modules; a test checks this). The only wall-clock input is the 2 s "trials needed" hint deadline, which can mark a hint "skipped" on a very slow machine but never changes a verdict.
   - Mutation check: `python scripts/mutation_check.py` breaks 15 lines (bar `>=` to `>`, wrong z level, missing Wilson correction, wrong edge cases, and so on). First run: 3 survived. They were all "evidence exactly equal to the bar passes", which no test covered. Added tests; now 15 of 15 killed.
   - Golden files: text, JSON, Markdown and JUnit for `gate` and for `items` at k=1 and k=2 are pinned in `tests/golden/`.

## Author line

Command: `unzip -p dist/passgate-0.3.0-py3-none-any.whl '*/METADATA' | grep -E '^(Author|Maintainer)'` printed `Author: Calas Automations Inc.` and `Maintainer: Calas Automations Inc.`; the sdist `PKG-INFO` says the same; the wheel's `LICENSE` line 3 is `Copyright (c) 2026 Calas Automations Inc.`; `passgate --version` prints `passgate 0.3.0 (Calas Automations Inc., MIT licence)`. No e-mail is shipped (none was present). Remaining `calasdan-sketch` text is real GitHub URLs, and the `uses:` line in the README. Whether that repository is public could not be checked from here (GitHub access was refused), so the project URLs are kept as in round 4 and are marked unverified. Pinned by `tests/test_round5.py`.

## Scored ideas

Scores 1 (low) to 5 (high); effort and risk 1 (low) to 5 (high).

| # | idea | value | diff. | effort | risk | status |
|---|---|---|---|---|---|---|
| 1 | Fix zero-width interval for pass^k / pass@k | 5 | 3 | 2 | 2 | DONE |
| 2 | Coverage simulation for the item interval | 4 | 4 | 2 | 1 | DONE |
| 3 | Float-loop pass^k / pass@k (fast, no digit loss) | 3 | 2 | 1 | 1 | DONE |
| 4 | Clamp Wilson so the estimate is inside its interval | 2 | 1 | 1 | 1 | DONE |
| 5 | Author line and pin test | 4 | 1 | 1 | 1 | DONE |
| 6 | Differential tests against exact arithmetic | 4 | 2 | 2 | 1 | DONE |
| 7 | Property tests (seeded, plus hypothesis when installed) | 3 | 2 | 2 | 1 | DONE |
| 8 | Mutation check script | 3 | 3 | 2 | 1 | DONE |
| 9 | Golden files for every output format | 3 | 2 | 2 | 1 | DONE |
| 10 | Action outputs `exit-code`, `report-path` | 3 | 3 | 1 | 1 | DONE (untested on runners) |
| 11 | Upgrade-path test from 0.1.1 | 4 | 1 | 2 | 1 | DONE |
| 12 | sdist carries golden files | 2 | 1 | 1 | 1 | DONE |
| 13 | Verdict-stability scan near the bar | 3 | 3 | 1 | 1 | DONE (trials below 400 only) |
| 14 | Judge-noise modelling | 3 | 4 | 5 | 4 | NOT-DONE (research problem; carried over) |
| 15 | Per-cell stochasticity audit (evalpower's feature) | 3 | 3 | 4 | 3 | NOT-DONE (carried over) |
| 16 | Bootstrap intervals and plots | 2 | 2 | 4 | 3 | NOT-DONE (breaks the no-dependency promise) |
| 17 | Paired-design planning for `regress` | 3 | 3 | 4 | 3 | NOT-DONE (needs a discordant-rate model) |
| 18 | SARIF output | 2 | 2 | 3 | 2 | NOT-DONE (no code locations in a statistical gate) |
| 19 | Pin Actions by commit hash | 3 | 1 | 2 | 2 | NOT-DONE (cannot look hashes up offline) |
| 20 | Trusted Publishing workflow | 4 | 1 | 3 | 4 | NOT-DONE (publishing off limits; owner's PyPI setup) |
| 21 | `mypy` in CI | 2 | 1 | 2 | 2 | NOT-DONE (ran clean by hand again) |
| 22 | Beta-binomial (overdispersed) planning for `items` | 4 | 4 | 4 | 3 | NOT-DONE (needs a validated model of item difficulty) |
| 23 | Bayesian credible-interval option | 2 | 2 | 3 | 3 | NOT-DONE (prior choice becomes a trust question) |
| 24 | `gate --baseline-file` (fail if worse than the last release's saved result) | 4 | 4 | 3 | 3 | NOT-DONE (needs a state-file format design) |
| 25 | Stable `--hint-budget 0` flag in the Action for byte-identical logs | 2 | 1 | 1 | 1 | NOT-DONE (hints are already off the verdict path) |
| 26 | Windows and Python 3.9/3.10 runs | 5 | 1 | 1 | 1 | NOT-DONE (not available here; CI covers it) |

## What I could not test

- Anything on Windows, and Python 3.9 and 3.10 (only 3.13 was run in this session; earlier rounds ran 3.11 to 3.13).
- The GitHub Action (including the new outputs) and CI on GitHub's runners.
- Whether the GitHub repository is public; whether PyPI accepts the name.
- Competitor pages and source were not re-read; only the registry's newest version numbers (errorbars 0.2.4, evalpower 0.1.0) were checked, and they are unchanged, so the comparison table dated 2026-11-07 stands.
- Coverage of the item interval beyond the three difficulty shapes and ten tries per item in `bench_items_k.py`.
- The locale test: only proves something if the machine has those locales installed.
