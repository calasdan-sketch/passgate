# passgate

**Pass once is luck. Pass^N is a gate.** A zero-dependency CI gate for LLM evals: every evaluation round must clear a *conservative* pass-rate bar, not just the average. It tells you how many trials you need (with a power target), whether a candidate is *not worse* than the baseline, when you can safely stop early, and it explains the verdict in plain English.

Eval flakiness is the quiet killer. A prompt scores 91% once, ships, and regresses. Averaging hides the weak round. `passgate` requires each round's lower confidence bound (Wilson, or exact Clopper-Pearson) to clear your bar. A candidate that needs 3 good rounds to ship must produce 3 good rounds, with enough trials behind each one.

```bash
pip install passgate          # Python 3.9+, standard library only
```

> **Release status.** The newest version on PyPI at the time of writing is 0.1.1, which has only `gate` and `compare`. This README describes 0.3.0 (the version in this repository), which is not on PyPI yet. Until it is, install it from GitHub, `pip install git+https://github.com/calasdan-sketch/passgate.git`, or from a checkout, `pip install .`. Check with `passgate --version`.

Also runnable as `python -m passgate ...`.

## Quick start (60 seconds)

Write the round results your eval runner produced. This example is synthetic. On Windows, where the `cat <<EOF` line below does not work, run `passgate example rounds -o rounds.json` instead; it writes the identical file.

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
| `gate FILE...` | Does every round clear the bar? | all rounds clear and there are at least `--min-rounds` |
| `plan --rate R [--power P]` | How many trials per round do I need? | the target is achievable |
| `plan-regress --margin M --baseline-rate R` | How many trials per arm so `regress` can prove "not worse"? (or `--trials N`: what margin can N trials prove?) | the target is achievable |
| `bound 30/30 ...` | What does this result actually prove? | always |
| `oc --trials N` | How likely is a good (or bad) candidate to pass? | always |
| `compare --candidate K/N --baseline K/N` | Is the candidate *better*? (independent samples) | candidate is significantly better |
| `paired FILE` | Is the candidate better on the **same items**? | candidate is significantly better |
| `regress --margin M ...` | Is the candidate **not worse** than the baseline by more than M? | superior or non-inferior |
| `sequential FILE` | Can I stop early, peeking after every trial? | certified above the bar |
| `items FILE` | Pass rate / pass^k when each item is tried several times | cluster-aware lower bound clears the bar |
| `cases FILE` | Which of many cases regressed? (multiple-testing control) | no supported regression |
| `arms FILE` | Which prompts / providers does this promptfoo or Inspect file hold? | always |
| `example KIND` | Print (or `-o FILE` write) a small synthetic input file: `rounds`, `paired`, `cases` or `outcomes` | always |

Exit codes: `0` pass, `1` did not pass, `2` bad input or any unexpected error, `130` interrupted with Ctrl-C. All commands exit `2` on bad input (missing file, bad JSON/CSV, invalid number, duplicate item, empty file, a file over 256 MiB, NUL bytes) and on any unexpected internal error, so a typo or a crash is never mistaken for a model regression, and an empty input never passes. Every file argument also accepts `-` for standard input (`cat rounds.json | passgate gate -`), and `gate` expands `*`/`?` patterns itself when the shell does not (cmd.exe and PowerShell never do: `passgate gate runs\*.json`; a file whose real name contains those characters still works). Files may be UTF-8 (BOM or not) or UTF-16 with a byte-order mark, which is what Windows PowerShell 5 writes for `command > file`. CSV may use `,`, `;`, tab or `|` as the separator. Commands that judge a result (`gate`, `compare`, `paired`, `regress`, `sequential`, `items`, `cases`) accept `--format text|json|markdown|junit`, `--explain` (a plain-English paragraph for people who do not read confidence intervals) and `--github-summary`. JSON output carries `passgate_version`. `compare`, `paired` and `regress` also take `--inconclusive-exit CODE` (1, or 3 to 125) so CI can tell "needs more data" from "worse".

### gate

```bash
passgate gate rounds.json --threshold 0.9 --min-rounds 3 [--method exact] [--confidence 0.95]
```

- Input: a JSON list of `{trials, passes, label?}` objects, or `[passes, trials]` pairs, or a CSV with `trials,passes[,label]` columns (`n,k` also accepted). UTF-8 with or without BOM, LF or CRLF.
- `--method wilson` (default) uses the Wilson score bound. `--method exact` uses the Clopper-Pearson bound, which never undercovers: its one-sided coverage is at least nominal for every true rate (the test suite checks this). Wilson can dip slightly below nominal for some rates.
- `--json` (or `--format json`) gives machine-readable output, including `trials_needed` for each failing round.
- Several files are joined into one list of rounds. With `--per-trial`, each file is a *raw results file* (promptfoo, Inspect, or CSV/JSONL with one outcome per row) and becomes ONE round, so three eval runs need no glue script: `passgate gate run1.json run2.json run3.json --per-trial --min-rounds 3`. Pick a prompt or provider with `--arm`. If items inside a file were tried more than once, `gate` warns on stderr that its bound is too optimistic and points to `items`.

### bound: what does "30 out of 30" actually prove?

```bash
passgate bound 30/30
# 30/30 = 1.000: the true pass rate is between 0.884 and 1.000 (95%, exact); Wilson says 0.886 to 1.000. The failure rate could still be as high as 11.6%.
```

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

The verdict is `candidate-better`, `baseline-better` or `inconclusive`. `--alpha` must be in (0, 0.5]. It is the false-positive rate, not the confidence, so a mistyped `--alpha 0.95` is rejected (exit 2) instead of letting a worse candidate pass.

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
# paired: 3 items, candidate 0.667 vs baseline 0.667 - fixed 1, broke 1, exact McNemar one-sided p=0.75, verdict: inconclusive
```

Input can be CSV, a JSON list, or JSONL (`.jsonl`/`.ndjson`) with `candidate` and `baseline` fields. Values may be `1/0`, `true/false`, `pass/fail` or `yes/no`. The `item` (or `id`) field is optional, but a duplicate id is rejected so an item cannot be counted twice. Example from the test suite (200 shared items, 12 fixed, 2 broken): pooled `compare` says *inconclusive*, while `paired` says *candidate-better* (p < 0.01).

### plan --power: size the eval so a good candidate actually passes

`plan` without `--power` gives the break-even size. With a power target it gives the size at which a candidate whose true rate is `--rate` passes the gate with at least that probability (exact binomial arithmetic, no simulation). With `--rounds 3` all three rounds must clear.

```bash
passgate plan --rate 0.95 --threshold 0.9 --power 0.8 --rounds 3
# plan: run 327 trials per round (305+ passes to clear); a candidate whose true rate is 0.950 then passes 81.8% of the time (all 3 rounds).
passgate oc --trials 327 --threshold 0.9 --rounds 3      # the whole operating curve
```

The pass probability is a sawtooth in the trial count (pass counts are whole numbers), so the output also reports the size from which the target holds for every larger size. The returned size is the true minimum (every size up to 20,000 is checked one by one).

### regress: "did we get worse?" is not the same question as "are we better?"

`compare` asks whether the candidate is *better*. Two identical prompts fail that test about 95% of the time, which makes it useless as a CI regression check. `regress` asks the right question: with a tolerated drop of `--margin`, is the lower end of the confidence interval for (candidate - baseline) above `-margin`? It uses Newcombe's hybrid score interval (independent items) or its paired version (same items), validated against statsmodels and by simulation.

```bash
passgate regress --margin 0.03 --candidate 188/200 --baseline 190/200 --explain
passgate regress --margin 0.03 --paired results.csv        # both arms ran the same items
```

Verdicts: `superior`, `non-inferior` (exit 0), `inconclusive`, `inferior` (exit 1). Inconclusive means the data cannot rule out a drop bigger than the margin; it never passes (`--inconclusive-exit 3` gives it its own exit code).

How many trials is enough? Ask before you run:

```bash
passgate plan-regress --margin 0.05 --baseline-rate 0.9              # trials per arm
# plan-regress: run 451 trials per arm (902 in total). A candidate whose true rate is 0.900 against a baseline at 0.900 is then declared not worse than 0.05 80.2% of the time (one-sided alpha 0.05).
passgate plan-regress --baseline-rate 0.9 --trials 300               # or: what margin can 300 trials per arm prove?
# plan-regress: with 300 trials per arm, 'regress' can prove the candidate is not worse than about 6.1 points (80% chance if the true rates are 0.900 and 0.900; one-sided alpha 0.05). A smaller margin needs more trials.
```

The power is exact binomial arithmetic over every outcome (checked against simulation in the benchmark, section 6). The size comes from a geometric search plus bisection, so a slightly smaller one may also reach the target. It assumes independent samples (different items per arm) and searches up to 20,000 trials per arm. At a true drop of exactly the margin, `regress` passes about 5.1% of the time, a hair over the nominal 5%, because the Newcombe interval is an approximation.

### sequential: stop the moment the evidence is conclusive

Checking a Wilson bound after every trial and stopping when it clears is a common mistake: at a true rate exactly on the bar it "passes" about 14% of the time instead of at most 2.5% (see the benchmark). `sequential` uses a mixture likelihood-ratio test whose false-pass rate is controlled over *all* stopping times (Ville's inequality), so you can look after every trial.

```bash
passgate sequential outcomes.txt --threshold 0.9 --max-trials 1000    # one 1/0 per line
# for 80 lines with a single 0 near the start:
# sequential: PASS after 75 trial(s) (74 passes), any-time lower bound 0.901 vs bar 0.900 - evidence crossed the any-time bound
```

It also stops with `fail` as soon as certification is impossible within `--max-trials`. A stream that ends early is `undecided` and never passes. The default `--alpha 0.025` matches the one-sided level of a 95% gate. `pass_boundary()` returns the pass-count table if you want to build the stopping rule into your own runner. The price: at a horizon of 500 trials a candidate at 0.95 is certified 91.5% of the time versus 98.8% for a single fixed look, in exchange for needing about half as many trials on average.

### items: when each test item is tried several times

50 items sampled 10 times are not 500 independent coin flips. `items` treats the item as the unit: a cluster-robust standard error gives a design effect, and the Wilson bound is evaluated at the effective sample size with a Student-t critical value. For `--k 1` it never reports a bound less cautious than plain Wilson. With `--k 3` it reports unbiased **pass^k** (all of k tries succeed, tau-bench) or `--metric pass@k` per item, with a Wilson interval over the per-item values at an effective sample size taken from their spread (approximate; never narrower than one observation per item).

```bash
passgate items results.json --threshold 0.8 --explain         # promptfoo, Inspect, CSV or JSONL
passgate items results.json --k 4 --threshold 0.6             # chance that 4 tries in a row succeed
```

### cases: find the regressions without drowning in false alarms

Test 50 cases at 5% each and you get at least one false alarm 34% of the time (benchmark section 5) even when nothing changed. `cases` runs a one-sided Fisher exact test per case and adjusts with Holm (any false alarm <= alpha) or Benjamini-Hochberg (`--adjust bh`). Adjusted p-values match statsmodels. By contrast, `gate` needs no correction: requiring *every* round to clear is an intersection-union test, already valid at its level.

```bash
printf 'case,cand_passes,cand_trials,base_passes,base_trials\na,10,20,19,20\nb,18,20,19,20\n' > cases.csv
passgate cases cases.csv --explain
```

### Reading promptfoo and Inspect AI results

`items`, `arms`, `paired` and `regress` read results straight from other tools. Formats follow each tool's public docs; every reader fails closed (exit 2) on anything it cannot interpret instead of guessing.

- **promptfoo** `results.json` / `.jsonl`: rows with `success`, `testIdx`, `promptIdx`, `provider`. `--repeat` runs become trials of one item. Provider errors count as failures. Use `passgate arms FILE` to list prompt/provider arms, then `--arm`, or `--candidate-arm/--baseline-arm` for `paired`.
- **Inspect AI** `.json` logs and `.eval` archives: `epoch` repeats become trials of one sample. Scores `C`/`1` pass; `I`/`N`/`0` fail; partial credit (`P`, fractions) is refused. Pick a scorer with `--scorer`. Compare two runs with `paired run_a.eval --baseline-file run_b.eval`.
- **CSV / JSON / JSONL**: one outcome per row (`pass`/`passed`/`success`/`correct`/`ok`) or `trials` + `passes`, plus optional `item` and `arm`.

## CI

```yaml
- run: pip install passgate
- run: python run_evals.py --out rounds.json   # your eval runner
- run: passgate gate rounds.json --threshold 0.9 --min-rounds 3 --github-summary --explain
```

`--github-summary` appends a Markdown report to the job summary (`$GITHUB_STEP_SUMMARY`). `--format markdown` prints the same report, for a PR comment, and `--format junit` emits JUnit XML so GitLab, Jenkins, Azure DevOps and GitHub test reporters show each round (or case) as a test.

A composite GitHub Action is included ([`action.yml`](action.yml)). It installs passgate from the action's own checkout, runs one command, writes the Markdown verdict to the job summary, optionally posts or updates a pull-request comment (`comment: "true"`, needs `pull-requests: write`, skipped for forks), and keeps passgate's exit code:

```yaml
- uses: calasdan-sketch/passgate@main        # pin a tag or commit in real use
  with:
    command: regress
    args: --margin 0.03 --paired results.csv
    comment: "true"
```

The action's shell script was exercised locally (summary written, exit codes kept); it has not been run on GitHub's runners. Labels and ids from your files are treated as untrusted. Terminal control codes, bidi overrides and zero-width characters are shown escaped (`\x1b`, `\u202e`). In Markdown output, every Markdown-significant punctuation mark is escaped too. That covers tables, HTML, `@mentions`, `:emoji:` codes, bare-URL autolinks and `#123` references, so a hostile label cannot rewrite your logs, fill your job summary or ping your team. JUnit output escapes the same characters, so it is always well-formed XML.

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

from passgate import non_inferiority, plan_power, run_sequential, evaluate_items, Item, compare_cases, Case
print(plan_power(0.95, 0.9, power=0.8)["trials"])                           # 217
print(non_inferiority([Round(1000, 900)], [Round(1000, 905)], margin=0.05).verdict)
print(run_sequential([True] * 80, threshold=0.9).decision)                   # pass
print(evaluate_items([Item(10, 10)] * 20 + [Item(10, 0)] * 5, threshold=0.6).passed)
```

`Round` accepts any integer-like value (including NumPy integers) and whole-number floats such as `93.0`. It rejects fractions and booleans.

## How it compares

Each statement about another tool comes from the public page it links to, or from a text search of that tool's published source (versions below, checked 2026-11-07; installed and read, **not executed**). "Not found" means the search found nothing, which can miss things. If something has changed since, please open an issue.

| | passgate 0.3 | [errorbars](https://pypi.org/project/errorbars/) 0.2.4 | [evalpower](https://pypi.org/project/evalpower/) 0.1.0 | [flakelens](https://pypi.org/project/flakelens/) 0.1.1 | [scorio](https://scorio.readthedocs.io/en/latest/api/eval.html) 0.2.3 | [Inspect AI](https://inspect.aisi.org.uk/) 0.3.288 | [promptfoo](https://www.promptfoo.dev/docs/configuration/outputs/) 0.125.1 |
|---|---|---|---|---|---|---|---|
| Runtime dependencies | none | numpy | none | none | NumPy | many (full eval framework) | Node.js app |
| Python | 3.9+ | 3.10+ | 3.11+ | 3.10+ | 3.10+ | n/a | n/a |
| Pass/fail threshold gate with CI exit codes | yes (0 / 1 / 2) | not stated on its page | no (always exits 0, by design) | yes (non-zero when a case is broken) | no | no gate found | pass-rate threshold only (`PROMPTFOO_PASS_RATE_THRESHOLD`); no confidence bound found |
| Per-round lower bound | Wilson or Clopper-Pearson | - | Clopper-Pearson | Wilson | normal-approx. Bayesian | Wilson CI metric available | - |
| Sample-size planning | `plan` (power target, operating curve), `plan-regress` (trials, or the margin N trials can prove) | power analysis (min. detectable effect) | runs needed | - | - | - | - |
| Paired test on shared items | exact McNemar | exact McNemar | - | - | - | - | - |
| Non-inferiority ("not worse by more than M") | yes (`regress`) | not found | not found | not found | not found | not found | not found |
| Early stopping with Type I control | yes (`sequential`) | not found | not found | not found | sequential-inference helpers (`sinf`) | not found | not found |
| Multiple-testing control across cases | Holm / BH (`cases`) | Holm on leaderboard pairs | - | - | - | - | - |
| Clustered items | cluster-aware Wilson (`items`) | clustered standard errors | - | - | - | clustered standard error metric | - |
| pass^k / pass@k | unbiased per item (`items --k`) | not found | yes | - | yes | pass@k reducer | - |
| Reads promptfoo / Inspect results | yes | Inspect adapter (extra) | - | - | not checked | native | native |
| JUnit XML / PR-comment action | yes / yes (untested on runners) | not found | not found | not found | not found | not found | not checked |
| Plain-English verdict | `--explain` | - | - | - | - | - | - |
| Runs your prompts for you | no | no | no | yes | no | yes | yes |

What others do better: errorbars has richer clustered/multi-model leaderboards, bootstrap intervals and SVG plots. evalpower and scorio estimate pass^k from run counts without needing item-level files, and evalpower audits each model-task cell for whether it is demonstrably stochastic (passgate has no such per-cell audit; `bound` only states what a count proves). Both were re-read on 2026-11-07 at the same versions as in the table (PyPI index and package source, not executed). Inspect AI and promptfoo run the evals themselves and have large ecosystems. flakelens runs your test cases itself. If you need those, use them; passgate is the small, strict, dependency-free gate you put at the end of the pipeline, and it reads their output.

### Measured: how often does each gate make the wrong call?

Committed Monte Carlo benchmark with fixed seeds: [`benchmarks/bench_gates.py`](benchmarks/bench_gates.py), full tables in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md). Re-run it with `pip install -e ".[dev]" && python benchmarks/bench_gates.py` (one to two minutes). Headlines (bar 0.9 unless noted):

| situation | naive rule | passgate |
|---|---|---|
| true rate 0.89 (below the bar), 3 rounds of 327 trials: false pass | 27.7% (single run) / 17.2% (mean of 3) | 0.0% |
| true rate 0.95, same setup: correct pass | ~100% | 81.4% (the 80% power you asked `plan` for; 8.5% at 100 trials, so size your eval) |
| peeking after every trial, true rate exactly 0.90: false pass | 13.7% (Wilson checked every trial) | 1.5% (`sequential`; nominal limit 2.5%) |
| 30 items x 10 tries, ICC 0.3, true rate 0.80, bar 0.80: false pass | 16.4% (plain pooled Wilson) | 3.1% (`items`) |
| candidate equal to baseline, 200 trials each: passes the CI check | 5.4% (`compare` demands "better") | 50.2% (`regress`, margin 0.05); a true 0.05 drop passes 4.9% |
| 50 cases, nothing changed: any false alarm | 34.0% (raw p < 0.05) | 0.1% (Holm) |
| `plan-regress` size (451/arm, margin 0.05): equal prompts pass | n/a | 80.5% simulated vs 80.2% exact |

The honest cost: passgate is strict. With few trials it blocks good candidates (0% pass at 30 trials per round, 8.5% at 100), and the sequential test certifies a 0.95 candidate within 500 trials 91.5% of the time versus 98.8% for one fixed look. Cluster-aware bounds are still slightly liberal with only 10 items (4.5% false pass at the boundary in the table). Use `plan --power` to size the eval.

## Honest limits

- The Wilson and Clopper-Pearson intervals are two-sided, so a 95% interval gives a one-sided 97.5% lower bound per round. Requiring every round to clear is an intersection-union test, which needs no multiple-testing correction for the claim "every round is above the bar". Searching many *cases* for a regression is a different question; use `cases` for that.
- `gate` treats trials in a round as independent. If rounds reuse the same test items, or items are tried several times, the real uncertainty is larger than the bound shows. Use `items` for repeated trials; see [Miller 2024, "Adding Error Bars to Evals"](https://arxiv.org/abs/2411.00640). The `items` bound uses an estimated design effect, so it is itself approximate with few items (under about 20) and the pass^k / pass@k interval is a Wilson interval over per-item estimates at an effective sample size taken from their spread, not exact. Simulated coverage of its lower end is at or above nominal in the cases tried ([`benchmarks/RESULTS-items-k.md`](benchmarks/RESULTS-items-k.md)); the round-4 t-interval fell short (87% to 95% against 97.5%) and could report a perfect bound from items that all scored the same.
- `sequential` and `regress` need real sample sizes. Tiny streams never certify, and equal prompts pass `regress` only about half the time at 200 trials per arm and a 5-point margin. `plan-regress` sizes the independent-samples design only (not the paired one) and stops at 20,000 trials per arm.
- Files are read whole: the cap is 256 MiB, a CSV cell may be 1 MiB, and about a million rows take around ten seconds to parse. Parsing speed was measured on one shared machine, not benchmarked.
- Windows: nothing in passgate has been run on Windows. CI is set up to run the tests and a fresh-venv wheel smoke test on `windows-latest`, but that workflow has not run on GitHub yet. What was checked by reading the code: no `shell=True`, no signals, no file replacement, files opened in binary and decoded as UTF-8 or UTF-16 (BOM) so a PowerShell 5 redirect works; CRLF, BOM and `;`-separated CSV are tested; console encodings that cannot show a character (cp1252, ASCII) were simulated on Linux and print `\xe9` or `\u65e5` instead of failing; file patterns such as `runs\*.json` are expanded by passgate itself. Labels from `--per-trial` use the OS's own basename rule, which is untested on Windows.
- The `sequential` any-time guarantee is for a fixed bar and a stream of independent trials. It does not cover changing the bar after seeing data, and its bound (`anytime_lower`) is wider than a fixed-size Wilson bound by design.
- The promptfoo and Inspect readers follow public docs and the published log schema; they are tested on synthetic files in that shape, not on every version of those tools. Unrecognised data fails closed. JUnit and the GitHub Action cover the common cases; there is no SARIF output.
- `compare` pools all rounds and does not model variance from one round to the next. Use `paired` when both arms share items.
- `plan` and the failure hints give the *break-even* size. If your true rate equals `--rate`, a round of that size clears only about half the time.
- Fisher p-values are exact (integer arithmetic) up to 5,000 total trials. Above that they use log-space sums. The exact methods (Fisher, McNemar, Clopper-Pearson) are refused above 10^8 trials, where float precision degrades: `compare` then reports Fisher as n/a and decides by the z-test, and `--test fisher`, `paired` or `--method exact` exit 2. Wilson and the z-test have no such limit.
- Accuracy is checked against SciPy (and statsmodels for Newcombe intervals and p-value adjustments) by [`scripts/crosscheck_scipy.py`](scripts/crosscheck_scipy.py) (`pip install -e ".[dev]"`; SciPy is a dev-only oracle, never a runtime dependency). The test suite runs a quick version when SciPy is installed. A recent full run gave worst errors of about 5e-13 (Clopper-Pearson), 2e-10 (Fisher, up to about 58,000 trials) and 2e-10 (McNemar, up to 10^5 items). Accuracy between that and 10^8 was spot-checked by hand at about 1e-7 relative, but not by the script.
- With `--method exact`, "trials needed" hints are computed for at most 25 failing rounds and 2 seconds in total. The rest say "hint skipped". Hints never affect the verdict.
- If an LLM judge decides pass/fail, the judge's own noise is inside your counts. passgate cannot separate it out.

## License

MIT. Copyright (c) 2026 Calas Automations Inc. See `LICENSE`.
