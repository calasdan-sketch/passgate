# passgate

**Pass once is luck. Pass^N is a gate.** A zero-dependency CI gate for LLM evals: every evaluation round must clear a *conservative* pass-rate bar, not just the average. It also tells you how many trials you need, and whether a candidate really beats your baseline.

Eval flakiness is the quiet killer. A prompt scores 91% once, ships, and regresses. Averaging hides the weak round. `passgate` requires each round's lower confidence bound (Wilson, or exact Clopper-Pearson) to clear your bar. A candidate that needs 3 good rounds to ship must produce 3 good rounds, with enough trials behind each one.

```bash
pip install passgate          # Python 3.9+, standard library only
```

Also runnable as `python -m passgate ...`.

## Quick start (60 seconds)

Write the round results your eval runner produced. This example is synthetic:

```bash
cat > rounds.json <<'EOF'
[
  { "label": "core-set",    "trials": 100, "passes": 93  },
  { "label": "adversarial", "trials": 100, "passes": 70  },
  { "label": "regression",  "trials": 120, "passes": 116 }
]
EOF
passgate gate rounds.json --threshold 0.9 --min-rounds 3
```

```
passgate: 1/3 round(s) clear Wilson-lower >= 0.900 (95% confidence, min 3 rounds)
  FAIL     core-set: 93/100 = 0.930 (lower 0.863) - needs 398 trials at this rate (370+ passes)
  FAIL  adversarial: 70/100 = 0.700 (lower 0.604) - rate is at or below the bar; more trials cannot help
  PASS   regression: 116/120 = 0.967 (lower 0.917)
verdict: FAIL
```

Exit code `1` fails CI. Every failing round says what would fix it. **93/100 does not certify ≥ 0.9** with 100 trials, but 398 trials at the same rate would. A 70% round can never certify 90%, no matter how many trials you run. The bar scales with evidence, not vibes.

## Commands

| command | answers | exit 0 when |
|---|---|---|
| `gate FILE` | Does every round clear the bar? | all rounds clear and there are at least `--min-rounds` |
| `plan --rate R` | How many trials per round do I need? | the target is achievable |
| `compare --candidate K/N --baseline K/N` | Is the candidate better? (independent samples) | candidate is significantly better |
| `paired FILE` | Is the candidate better on the **same items**? | candidate is significantly better |

All commands exit `2` on bad input (missing file, bad JSON/CSV, invalid number, duplicate item, empty file). A typo is never mistaken for a model regression, and an empty input never passes.

### gate

```bash
passgate gate rounds.json --threshold 0.9 --min-rounds 3 [--method exact] [--confidence 0.95]
```

- Input: a JSON list of `{trials, passes, label?}` objects, or `[passes, trials]` pairs, or a CSV with `trials,passes[,label]` columns (`n,k` also accepted). UTF-8 with or without BOM, LF or CRLF.
- `--method wilson` (default) uses the Wilson score bound. `--method exact` uses the Clopper-Pearson bound, which never undercovers: its one-sided coverage is at least nominal for every true rate (the test suite checks this). Wilson can dip slightly below nominal for some rates.
- `--json` (or `--format json`) gives machine-readable output, including `trials_needed` for each failing round.

### plan: size your eval before you burn tokens

```bash
passgate plan --rate 0.95 --threshold 0.9 --rounds 3
```

```
plan: run 140 trials per round; a round clears >= 0.900 (95% confidence, wilson) with 133+ passes.
      3 round(s) = 420 trials. This is break-even: at a true rate of exactly 0.950 a round clears only about half the time, so add headroom.
```

The result is the smallest round size at which a round passing at `--rate` clears the gate. The test suite checks this minimality by brute force.

### compare: candidate vs baseline (independent samples)

```bash
passgate compare --candidate 93/100 --candidate 95/100 --baseline 85/100 --baseline 88/100
# candidate 0.940 vs baseline 0.865 - z=2.528, one-sided p=0.00573, fisher p=0.01734 (one-sided 0.008671), test: z, verdict: candidate-better
```

This runs a pooled two-proportion z-test plus Fisher's exact test (two-sided and one-sided). `--test fisher` lets the exact one-sided Fisher p decide the verdict. Use it for small rounds, where the normal approximation is too optimistic:

```bash
passgate compare --candidate 9/10 --baseline 4/10 --alpha 0.02 --test fisher
# ... one-sided p=0.009538, fisher p=0.05728 (one-sided 0.02864), test: fisher, verdict: inconclusive
```

The verdict is `candidate-better`, `baseline-better` or `inconclusive`.

### paired: same items, both arms (exact McNemar)

If the candidate and the baseline ran on the **same** test items, pooling throws away information. `paired` looks only at the items where the two disagree (the ones fixed and the ones broken). It is much more sensitive when item difficulty varies:

```bash
cat > results.csv <<'EOF'
item,candidate,baseline
q001,1,1
q002,1,0
q003,0,1
EOF
passgate paired results.csv
```

Input can be CSV, a JSON list, or JSONL (`.jsonl`/`.ndjson`) with `candidate` and `baseline` fields. Values may be `1/0`, `true/false`, `pass/fail` or `yes/no`. The `item` (or `id`) field is optional, but a duplicate id is rejected so an item cannot be counted twice. Example from the test suite (200 shared items, 12 fixed, 2 broken): pooled `compare` says *inconclusive*, while `paired` says *candidate-better* (p < 0.01).

## CI

```yaml
- run: pip install passgate
- run: python run_evals.py --out rounds.json   # your eval runner
- run: passgate gate rounds.json --threshold 0.9 --min-rounds 3 --github-summary
```

`--github-summary` appends a Markdown table to the job summary (`$GITHUB_STEP_SUMMARY`). `--format markdown` prints the same table, for a PR comment. Labels and ids from your files are treated as untrusted. Terminal control codes, bidi overrides and zero-width characters are shown escaped (`\x1b`, `‮`). Markdown, HTML and table syntax is escaped too, so a hostile label cannot rewrite your logs or your job summary.

## Python API

```python
from passgate import Round, evaluate_rounds, compare, compare_paired, trials_needed

result = evaluate_rounds(
    [Round(trials=100, passes=93), Round(trials=100, passes=95)],
    threshold=0.85, confidence=0.95, min_rounds=2, method="wilson",  # or "exact"
)
assert result.passed, str(result)
print(result.to_markdown())

print(compare([Round(100, 93)], [Round(100, 85)], test="fisher").verdict)
print(compare_paired([(True, False), (True, True), (False, False)]).verdict)
print(trials_needed(rate=0.95, threshold=0.9))   # (140, 133)
```

`Round` accepts any integer-like value (including NumPy integers) and whole-number floats such as `93.0`. It rejects fractions and booleans.

## How it compares

Each statement about another tool links to the public page it comes from. If something has changed since, please open an issue.

| | passgate | [errorbars](https://pypi.org/project/errorbars/) | [evalpower](https://pypi.org/project/evalpower/) | [flakelens](https://pypi.org/project/flakelens/) | [scorio](https://scorio.readthedocs.io/en/latest/api/eval.html) |
|---|---|---|---|---|---|
| Runtime dependencies | none | numpy | none | none | NumPy (used throughout its API docs) |
| Python | 3.9+ | 3.10+ | 3.11+ | 3.10+ | not checked |
| Pass/fail threshold gate with CI exit codes | yes (0 / 1 / 2) | not stated on its page | no (always exits 0, by design) | yes (non-zero when a case is broken) | no |
| Per-round lower bound | Wilson or Clopper-Pearson | - | Clopper-Pearson | Wilson | normal-approx. Bayesian |
| "How many trials do I need?" | yes (`plan`, hints on failure) | yes (power analysis) | yes (runs needed) | - | - |
| Paired test on shared items | exact McNemar | exact McNemar | - (not a between-model test) | - | - |
| Clustered standard errors | no | yes | - | - | - |
| pass^k estimator | no | - | yes | - | yes |
| Runs your prompts for you | no | no | no | yes | no |

What others do better: errorbars handles clustered questions and multi-model leaderboards. evalpower and scorio estimate pass^k ([τ-bench](https://arxiv.org/abs/2406.12045) defines pass^k). flakelens runs your test cases itself. If you need those, use them. passgate is the small, strict, dependency-free gate you put at the end of a pipeline.

## Honest limits

- The Wilson and Clopper-Pearson intervals are two-sided, so a 95% interval gives a one-sided 97.5% lower bound per round. Requiring every round to clear is conservative, but passgate applies no formal multiple-testing correction.
- Rounds are treated as independent samples of one pass rate. If rounds reuse the same test items, or items come in correlated clusters, the real uncertainty is larger than the bound shows. See [Miller 2024, "Adding Error Bars to Evals"](https://arxiv.org/abs/2411.00640) on clustered standard errors.
- `compare` pools all rounds and does not model variance from one round to the next. Use `paired` when both arms share items.
- `plan` and the failure hints give the *break-even* size. If your true rate equals `--rate`, a round of that size clears only about half the time.
- Fisher p-values are exact (integer arithmetic) up to 5,000 total trials. Above that they use log-space sums, checked against SciPy to about 1e-7 relative error up to 10^8 total trials. Beyond that, accuracy degrades; use the z-test.
- If an LLM judge decides pass/fail, the judge's own noise is inside your counts. passgate cannot separate it out.

## License

MIT. See `LICENSE`.
