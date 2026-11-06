# Changelog

## 0.2.0 (unreleased)

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
