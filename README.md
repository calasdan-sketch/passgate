# passgate

**Pass once is luck. Pass^N is a gate.** A statistical reliability gate for LLM evals: a candidate must clear a Wilson-bound pass bar on **every** evaluation round, not just on average — then optionally prove it beats your baseline.

Eval flakiness is the quiet killer: a prompt scores 91% once, ships, and regresses. Averaging hides the weak round. `passgate` requires each round's *conservative* pass-rate estimate (the lower bound of the Wilson score interval) to clear your bar, so a candidate that needs 3 good rounds to ship must produce 3 good rounds.

## Install

```bash
pip install passgate
```

## The gate

```bash
passgate gate rounds.json --threshold 0.9 --min-rounds 3
```

`rounds.json` (or `.csv`) holds one entry per round:

```json
[
  { "label": "baseline-eval-set", "trials": 100, "passes": 93 },
  { "label": "adversarial-set",   "trials": 100, "passes": 95 },
  { "label": "regression-set",    "trials": 120, "passes": 116 }
]
```

Output is a per-round scoreboard and a verdict; exit code `1` fails CI:

```
passgate: 2/3 round(s) clear Wilson-lower >= 0.900 (95% confidence, min 3 rounds)
         PASS round 1: 93/100 = 0.930 (lower 0.863)
         FAIL round 2: 70/100 = 0.700 (lower 0.604)
verdict: FAIL
```

Note what the gate does: **93/100 does not certify ≥ 0.9** at 95% confidence with 100 trials — you need more data or a lower bar. That is the point: the bar scales with evidence, not vibes. Add `--json` for machines.

## Candidate vs baseline

```bash
passgate compare --candidate 93/100 --candidate 95/100 \
                 --baseline  85/100 --baseline  88/100
```

Pooled two-proportion z-test with a one-sided p-value, plus Fisher's exact p (exact hypergeometric enumeration — right for small rounds where the normal approximation lies). Exit `0` only when the improvement is statistically significant at `--alpha` (default 0.05).

## Python API

```python
from passgate import Round, evaluate_rounds, compare

result = evaluate_rounds(
    [Round(trials=100, passes=93), Round(trials=100, passes=95)],
    threshold=0.9, confidence=0.95, min_rounds=2,
)
assert result.passed, str(result)

cmp = compare([Round(100, 93)], [Round(100, 85)])
print(cmp.verdict)  # candidate-better | baseline-better | inconclusive
```

## CI

```yaml
- run: pip install passgate
- run: python run_evals.py --out rounds.json   # your eval runner
- run: passgate gate rounds.json --threshold 0.9 --min-rounds 3
```

## Notes

- Wilson interval (not Wald): behaves correctly near 0% and 100%, where the naive interval explodes.
- `gate` fails when rounds < `--min-rounds`, so "we only ran it once" cannot pass a 3-round gate.
- Pure stdlib — no scipy, no pandas.

## License

MIT-style: use it, ship it, no warranty.
