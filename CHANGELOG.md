# Changelog

## 0.3.0 (unreleased)

Expert round 2 (statistician, LLM-eval practitioner, CI/DevEx lead, data engineer, skeptical ML engineer, plain-language explainer).

### Added
- `plan --power P [--rounds N]` and `oc`: power-aware sizing and operating curves. `plan` without `--power` only gave the break-even size, where a good candidate passes about half the time. Exact binomial arithmetic; the returned size is the true minimum (checked one by one up to 20,000 trials), plus the size from which the target holds for every larger size.
- `regress` / `non_inferiority` / `non_inferiority_paired`: non-inferiority test ("not worse by more than a margin") with Newcombe hybrid-score intervals, independent and paired. Verdicts `superior`, `non-inferior`, `inconclusive`, `inferior`. Matches statsmodels to 1e-12; coverage checked by simulation.
- `sequential` / `run_sequential` / `pass_boundary` / `anytime_lower`: always-valid mixture likelihood-ratio test and confidence sequence. Controls the false-pass rate over all stopping times; also stops with `fail` when certification is impossible in the budget.
- `items` / `evaluate_items`: cluster-aware pass rate (design effect, effective sample size, Student-t) and unbiased pass^k / pass@k per item.
- `cases` / `compare_cases`: per-case one-sided Fisher tests with Holm or Benjamini-Hochberg adjustment (match statsmodels).
- Ingestion of promptfoo results (`.json`/`.jsonl`), Inspect AI logs (`.json`, `.eval`) and generic CSV/JSON/JSONL; `arms` lists prompts/providers; `paired` accepts `--candidate-arm/--baseline-arm/--baseline-file`. Fails closed on partial credit, several scorers, mismatched items and oversized archives.
- `--format junit` and `--explain` (plain-English verdict) on the judging commands; `explain()` methods on all results; `action.yml` composite GitHub Action with optional PR comment.
- `benchmarks/bench_gates.py` with committed `RESULTS.md` / `results.json`: seeded Monte Carlo of false-pass and false-fail rates against naive gates.
- Student t quantile/CDF and `binom_sf` helpers in `passgate.stats` (tested against SciPy).
- 69 new tests (134 -> 203), including simulation tests with fixed seeds.

### Changed
- README: the claim that no multiple-testing correction is applied is clarified (all-rounds is an intersection-union test); comparison table extended and sourced; benchmark numbers added.
- `safe_text` also escapes lone surrogates (so XML and terminal output stay well-formed).
- dev extra now includes `statsmodels` and `numpy` (oracle and benchmark only; runtime stays dependency-free).

## 0.2.0 (unreleased)

### Fixed after panel review
- `alpha` must be in (0, 0.5] for `compare` and `paired` (CLI exit 2 otherwise). Before, `--alpha 0.95` (typed as a confidence level) could report a worse or equal candidate as `candidate-better`. The verdict is also inconclusive if both one-sided tails fall under alpha.
- Huge counts: Fisher is skipped (reported as `null` / n/a) above 10^8 total trials, so `compare` stays instant. `--test fisher` is refused at that size, and so are exact McNemar and Clopper-Pearson, whose float precision degrades there.
- Exact McNemar on up to 5,000 discordant items is now linear (an incremental binomial coefficient), not quadratic.
- Gate hints have a budget: 1000 failing rounds for Wilson, 25 for exact, and 2 s of wall-clock time. Skipped rounds say "hint skipped" (`trials_needed_skipped` in JSON). New `hint_budget=` argument. The verdict is unaffected.
- README contained a literal U+202E. Test files contained literal bidi and filler characters; they are now `\uXXXX` escapes. A new test scans docs, sources, tests and CI config for invisible characters.
- Added `scripts/crosscheck_scipy.py` and a `dev` extra (`pytest`, `ruff`, `scipy`). The test suite runs a quick cross-check when SciPy is present. README accuracy claims now point at it.
- The `clopper_pearson_interval` docstring no longer claims it is always wider than Wilson; it is not near k = n.
- Markdown cells now escape all Markdown-significant ASCII punctuation, defusing `@mentions`, `:emoji:`, bare-URL/`www.` autolinks, `#123` references and HTML entities. Blank "filler" letters (U+3164, U+115F, U+1160, U+FFA0, U+2800, U+17B4/5, U+180E) are escaped in all output.

### Added
- `--method exact` on `gate`: Clopper-Pearson lower bound. Its coverage is never below nominal (checked by an exact coverage test). Stdlib only: regularized incomplete beta plus a bracketed Illinois root finder, cross-checked against SciPy to about 1e-12.
- `passgate plan --rate R --threshold T [--rounds N]`: the smallest round size that certifies `T` at rate `R`, plus the passes needed. Minimality is verified by brute force in the tests.
- Failing rounds now show what would fix them ("needs 398 trials at this rate (370+ passes)", or "more trials cannot help"). `--json` carries the same data as `trials_needed`.
- `passgate paired FILE`: exact McNemar test for candidate vs baseline on the same items (CSV, JSON or JSONL). Duplicate item ids and empty files are rejected (exit 2).
- `compare --test fisher`: the one-sided Fisher exact p decides the verdict. `fisher_p_one_sided` is now reported.
- `--format markdown` and `--github-summary` (appends to `$GITHUB_STEP_SUMMARY`) on `gate`, `compare` and `paired`.
- API: `compare_paired`, `trials_needed`, `min_passes`, `lower_bound`, `clopper_pearson_interval`, `mcnemar_exact`, `fisher_exact_2x2(..., alternative=)`, `GateResult.to_markdown()`.

### Changed
- Untrusted text (labels, item ids, file paths in errors) is escaped before printing. Control, bidi and zero-width characters show as `\xNN`/`\uNNNN`; Markdown and HTML are escaped in Markdown reports; text labels are capped in length.
- `compare` verdict `baseline-better` is now reachable. Before, it effectively never fired because only the candidate-better tail was tested. Exit codes are unchanged (0 only for `candidate-better`).
- Fisher's exact test uses exact integer comparison for tables up to 5,000 trials. Above that it uses log-space sums that walk outward from the mode (O(sqrt n) instead of O(n)). The old version used an absolute tie tolerance of 1e-12, which over-counted tiny-probability tables.
- `Round` validates its counts: integer-like values (including NumPy ints) and whole floats are accepted; fractions and booleans raise `ValueError`.
- Gate text output names the method and reports "only N round(s), gate requires M" when rounds are missing.
- `RoundEval` now stores `lower` (the bound for the chosen method). `wilson_lower` remains as a property.
- Deeply nested JSON input exits 2 instead of crashing with a traceback.

## 0.1.1
- Exit code 2 for bad input; BOM/CRLF-safe loaders; validation; `python -m passgate`; ASCII-safe output; valid JSON for infinite z.

## 0.1.0
- Initial release.
