"""passgate CLI: gate eval rounds, compare a candidate against a baseline, or plan sample sizes.

Exit codes: 0 = pass, 1 = gate/comparison did not pass, 2 = bad input or usage.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from . import __version__
from .cluster import evaluate_items
from .gate import Round, compare, compare_paired, evaluate_rounds
from .ingest import (
    _STDIN_CACHE,
    FORMATS,
    csv_records,
    kind_of,
    load_arms,
    load_items,
    pairs_from_arms,
    pick_arm,
    read_text,
)
from .junit import to_junit
from .multi import ADJUSTMENTS, MAX_CASES, Case, compare_cases
from .noninf import (
    min_provable_margin,
    non_inferiority,
    non_inferiority_paired,
    plan_noninferiority,
)
from .power import oc_curve, plan_power
from .safe import safe_text
from .sequential import run_sequential
from .stats import METHODS, trials_needed


class InputError(ValueError):
    """Bad user input (file, format, or number). Reported with exit code 2."""


def parse_ratio(text: str) -> Round:
    """'93/100' -> Round(trials=100, passes=93)."""
    try:
        passes_s, trials_s = text.split("/", 1)
        return Round(trials=int(trials_s), passes=int(passes_s))
    except (ValueError, AttributeError):
        raise InputError(f"bad round {text!r}, expected PASSES/TRIALS e.g. 93/100") from None


def _to_int(value: Any, what: str, where: str) -> int:
    if value is None or isinstance(value, bool):
        raise InputError(f"{where}: missing or invalid {what}")
    try:
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError
            return int(value)
        return int(str(value).strip())
    except ValueError:
        raise InputError(f"{where}: {what} must be a whole number, got {safe_text(repr(value), 40)}") from None


def _make_round(trials: Any, passes: Any, label: str, where: str) -> Round:
    t = _to_int(trials, "trials", where)
    p = _to_int(passes, "passes", where)
    try:
        return Round(trials=t, passes=p, label=label)
    except ValueError as e:
        raise InputError(f"{where}: {e}") from None


def load_rounds(path: str) -> List[Round]:
    """JSON list of {trials, passes} (or [passes, trials] pairs) or CSV (trials,passes[,label]).

    ``-`` reads standard input. Files may carry a UTF-8 BOM and CRLF line endings; CSV may use
    ``,``, ``;``, tab or ``|`` as the separator. The format comes from the extension, or from the
    content when the name has no known extension.
    """
    text = read_text(path)
    rounds: List[Round] = []
    if not text.strip():
        raise InputError(f"{'standard input' if path == '-' else safe_text(path)} is empty")
    if kind_of(path, text) == "csv":
        for i, row in enumerate(csv_records(text, safe_text(path)), start=1):
            trials = row.get("trials") if row.get("trials") not in (None, "") else row.get("n")
            passes = row.get("passes") if row.get("passes") not in (None, "") else row.get("k")
            if trials is None and passes is None:
                raise InputError(
                    f"{safe_text(path)}: no 'trials' and 'passes' columns (found: "
                    + ", ".join(safe_text(k, 20) for k in list(row)[:6]) + "). A per-trial results file? "
                    "Give it to gate with --per-trial."
                )
            rounds.append(
                _make_round(trials, passes, (row.get("label") or "").strip(), f"{safe_text(path)} row {i}")
            )
        return rounds
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise InputError(f"{safe_text(path)} is not valid JSON: {safe_text(e)}") from None
    except RecursionError:
        raise InputError(f"{safe_text(path)}: JSON is nested too deeply") from None
    if not isinstance(data, list):
        raise InputError(f"{safe_text(path)}: expected a JSON list of rounds")
    for i, item in enumerate(data, start=1):
        where = f"{safe_text(path)} round {i}"
        if isinstance(item, dict):
            rounds.append(
                _make_round(item.get("trials"), item.get("passes"), str(item.get("label", "")), where)
            )
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            rounds.append(_make_round(item[1], item[0], f"round {i}", where))
        else:
            raise InputError(f"{where}: expected {{trials, passes}} or [passes, trials]")
    return rounds


_TRUE = {"1", "true", "pass", "passed", "yes", "y", "t"}
_FALSE = {"0", "false", "fail", "failed", "no", "n", "f"}


def _to_bool(value: Any, what: str, where: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _TRUE:
            return True
        if v in _FALSE:
            return False
    raise InputError(f"{where}: {what} must be pass/fail (1/0, true/false), got {safe_text(repr(value), 40)}")


def _read_text(path: str) -> str:
    return read_text(path)


def load_pairs(path: str) -> List[Tuple[bool, bool]]:
    """Per-item results for a paired comparison.

    CSV with ``candidate`` and ``baseline`` columns (optional ``item``), a JSON
    list of ``{"item", "candidate", "baseline"}`` objects, or JSONL (one object
    per line). Values: 1/0, true/false, pass/fail. Duplicate item ids are
    rejected so an item cannot be double-counted.
    """
    text = _read_text(path)
    records: List[Tuple[str, Dict[str, Any]]] = []
    kind = kind_of(path, text)
    if kind == "csv":
        for i, row in enumerate(csv_records(text, safe_text(path)), start=1):
            records.append((f"{safe_text(path)} row {i}", row))
    elif kind == "jsonl":
        for i, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise InputError(f"{path} line {i} is not valid JSON: {e}") from None
            except RecursionError:
                raise InputError(f"{path} line {i}: JSON is nested too deeply") from None
            if not isinstance(obj, dict):
                raise InputError(f"{path} line {i}: expected a JSON object")
            records.append((f"{path} line {i}", obj))
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise InputError(f"{path} is not valid JSON: {e}") from None
        except RecursionError:
            raise InputError(f"{path}: JSON is nested too deeply") from None
        if not isinstance(data, list):
            raise InputError(f"{path}: expected a JSON list of items")
        for i, obj in enumerate(data, start=1):
            if not isinstance(obj, dict):
                raise InputError(f"{path} item {i}: expected an object with candidate and baseline")
            records.append((f"{path} item {i}", obj))
    pairs: List[Tuple[bool, bool]] = []
    seen = set()
    for where, rec in records:
        item = rec.get("item", rec.get("id"))
        if item not in (None, ""):
            key = str(item)
            if key in seen:
                raise InputError(f"{where}: duplicate item {key!r}")
            seen.add(key)
        pairs.append(
            (_to_bool(rec.get("candidate"), "candidate", where), _to_bool(rec.get("baseline"), "baseline", where))
        )
    if not pairs:
        raise InputError(f"{path}: no items found")
    return pairs


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="passgate",
        description="Require an eval candidate to clear a Wilson-bound pass bar on EVERY round.",
        epilog="Exit codes: 0 pass, 1 fail / not significant, 2 bad input.",
    )
    ap.add_argument("--version", action="version", version=f"passgate {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gate", help="pass^N gate over eval rounds; exit 1 if any round is weak")
    g.add_argument("rounds", nargs="+", metavar="FILE", help="JSON or CSV rounds file(s); '-' reads standard input")
    g.add_argument(
        "--per-trial", action="store_true",
        help="each FILE is a raw results file (promptfoo, Inspect, CSV/JSONL with one outcome per row) and becomes ONE round",
    )
    g.add_argument("--from", dest="src", choices=FORMATS, default="auto", help="input format for --per-trial files")
    g.add_argument("--arm", help="with --per-trial: which arm (prompt/provider) when a file holds several")
    g.add_argument("--scorer", help="with --per-trial: Inspect scorer name when a log has several")
    g.add_argument("--threshold", type=float, default=0.9, help="required Wilson lower bound (default 0.9)")
    g.add_argument("--confidence", type=float, default=0.95, help="interval confidence (default 0.95)")
    g.add_argument("--min-rounds", type=int, default=1, help="fail if fewer rounds than this")
    g.add_argument(
        "--method", choices=METHODS, default="wilson",
        help="lower bound: wilson (default) or exact (Clopper-Pearson, never undercovers)",
    )
    _add_output(g)

    c = sub.add_parser("compare", help="candidate vs baseline: is the candidate actually better?")
    c.add_argument("--candidate", required=True, action="append", help="PASSES/TRIALS (repeatable, one per round)")
    c.add_argument("--baseline", required=True, action="append", help="PASSES/TRIALS (repeatable)")
    c.add_argument("--alpha", type=float, default=0.05)
    c.add_argument(
        "--test", choices=("z", "fisher"), default="z",
        help="what decides the verdict: z (default) or fisher (one-sided exact; best for small rounds)",
    )
    _add_output(c)
    _add_inconclusive(c)

    pr = sub.add_parser("paired", help="candidate vs baseline on the SAME items (exact McNemar)")
    pr.add_argument("results", help="CSV/JSON/JSONL with per-item candidate and baseline pass/fail")
    pr.add_argument("--alpha", type=float, default=0.05)
    _add_output(pr)
    _add_inconclusive(pr)

    pr.add_argument("--from", dest="src", choices=FORMATS, default="auto", help="input format for promptfoo / Inspect files")
    pr.add_argument("--candidate-arm", help="arm (prompt/provider) to treat as the candidate in a multi-arm file")
    pr.add_argument("--baseline-arm", help="arm to treat as the baseline in a multi-arm file")
    pr.add_argument("--baseline-file", help="a second results file holding the baseline run (items matched by id)")
    pr.add_argument("--scorer", help="Inspect scorer name when a log has several")

    rg = sub.add_parser("regress", help="non-inferiority: is the candidate NOT meaningfully worse than the baseline?")
    rg.add_argument("--margin", type=float, required=True, help="largest drop you accept, e.g. 0.03 = 3 points")
    rg.add_argument("--candidate", action="append", help="PASSES/TRIALS (repeatable); different items from the baseline")
    rg.add_argument("--baseline", action="append", help="PASSES/TRIALS (repeatable)")
    rg.add_argument("--paired", help="per-item results file where both arms ran the SAME items (see 'paired')")
    rg.add_argument("--alpha", type=float, default=0.05, help="one-sided false-pass rate (default 0.05)")
    rg.add_argument("--from", dest="src", choices=FORMATS, default="auto")
    rg.add_argument("--candidate-arm")
    rg.add_argument("--baseline-arm")
    rg.add_argument("--baseline-file")
    rg.add_argument("--scorer")
    _add_output(rg)
    _add_inconclusive(rg)

    sq = sub.add_parser("sequential", help="stop early: certify a pass rate while peeking after every trial")
    sq.add_argument("outcomes", help="file of outcomes in run order: 1/0, true/false or pass/fail, one per line (or a JSON list)")
    sq.add_argument("--threshold", type=float, default=0.9)
    sq.add_argument("--alpha", type=float, default=0.025, help="false-pass rate over ALL stopping times (default 0.025, the one-sided level of a 95% gate)")
    sq.add_argument("--max-trials", type=int, default=1000, help="trial budget; gives up (fails) beyond this")
    _add_output(sq)

    it = sub.add_parser("items", help="cluster-aware gate when each item is tried several times (also pass^k)")
    it.add_argument("results", help="CSV/JSON/JSONL, promptfoo results, or Inspect log (.json / .eval)")
    it.add_argument("--threshold", type=float, default=0.9)
    it.add_argument("--confidence", type=float, default=0.95)
    it.add_argument("--k", type=int, default=1, help="1 = pooled pass rate; k>=2 = pass^k (default metric)")
    it.add_argument("--metric", choices=("pass^k", "pass@k"), default="pass^k")
    it.add_argument("--from", dest="src", choices=FORMATS, default="auto")
    it.add_argument("--arm", help="which arm (prompt/provider) when the file holds several")
    it.add_argument("--scorer", help="Inspect scorer name when a log has several")
    _add_output(it)

    cs = sub.add_parser("cases", help="per-case regression search with multiple-testing control")
    cs.add_argument("results", help="CSV/JSON: case, cand_passes, cand_trials, base_passes, base_trials")
    cs.add_argument("--alpha", type=float, default=0.05)
    cs.add_argument("--adjust", choices=ADJUSTMENTS, default="holm", help="holm (any false alarm <= alpha), bh (FDR), none")
    _add_output(cs)

    ar = sub.add_parser("arms", help="list the arms (prompt/provider/model) and item counts in a results file")
    ar.add_argument("results")
    ar.add_argument("--from", dest="src", choices=FORMATS, default="auto")
    ar.add_argument("--scorer")

    oc = sub.add_parser("oc", help="operating characteristic: chance the gate passes at each true pass rate")
    oc.add_argument("--trials", type=int, required=True, help="trials per round")
    oc.add_argument("--threshold", type=float, default=0.9)
    oc.add_argument("--confidence", type=float, default=0.95)
    oc.add_argument("--method", choices=METHODS, default="wilson")
    oc.add_argument("--rounds", type=int, default=1, help="all of this many rounds must clear")
    oc.add_argument("--json", action="store_true")

    bd = sub.add_parser("bound", help="what a result like 30/30 actually proves (plain bounds, no verdict)")
    bd.add_argument("result", nargs="+", metavar="PASSES/TRIALS")
    bd.add_argument("--confidence", type=float, default=0.95)
    bd.add_argument("--json", action="store_true")

    pr2 = sub.add_parser("plan-regress", help="how many trials per arm so that 'regress' can actually prove no regression")
    pr2.add_argument("--margin", type=float, default=None, help="largest drop you accept, e.g. 0.03 (omit with --trials)")
    pr2.add_argument("--trials", type=int, default=None, help="instead: trials per arm you can afford; prints the smallest margin you can prove")
    pr2.add_argument("--baseline-rate", type=float, required=True, help="the baseline's true pass rate you expect, e.g. 0.9")
    pr2.add_argument("--candidate-rate", type=float, default=None, help="the candidate's true pass rate (default: same as the baseline)")
    pr2.add_argument("--power", type=float, default=0.8, help="chance a candidate at --candidate-rate passes (default 0.8)")
    pr2.add_argument("--alpha", type=float, default=0.05, help="one-sided false-pass rate (default 0.05)")
    pr2.add_argument("--json", action="store_true")

    pl = sub.add_parser("plan", help="how many trials per round to certify a threshold at an expected rate")
    pl.add_argument("--rate", type=float, required=True, help="pass rate you expect per round, e.g. 0.95")
    pl.add_argument("--threshold", type=float, default=0.9)
    pl.add_argument("--confidence", type=float, default=0.95)
    pl.add_argument("--method", choices=METHODS, default="wilson")
    pl.add_argument("--rounds", type=int, default=1, help="rounds you will run (for the total)")
    pl.add_argument("--power", type=float, default=None, help="size so a candidate at --rate passes with at least this probability (e.g. 0.8)")
    pl.add_argument("--json", action="store_true", help="machine-readable output")
    return ap


def _exit_for(args: argparse.Namespace, passed: bool, verdict: str) -> int:
    """0 pass, 1 fail; an 'inconclusive' verdict may use its own code (--inconclusive-exit) so CI can tell it apart."""
    if passed:
        return 0
    code = getattr(args, "inconclusive_exit", 1)
    return code if verdict == "inconclusive" else 1


def _exit_code(text: str) -> int:
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if n != 1 and not 3 <= n <= 125:
        raise argparse.ArgumentTypeError("must be 1 or between 3 and 125 (0 would let unproven changes pass; 2 means bad input)")
    return n


def _add_inconclusive(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--inconclusive-exit", type=_exit_code, default=1, metavar="CODE",
        help="exit code for an 'inconclusive' verdict (default 1, same as a failure); use e.g. 3 to treat it as 'needs more data' in CI",
    )


def _add_output(p: argparse.ArgumentParser) -> None:
    p.add_argument("--json", action="store_true", help="machine-readable output (same as --format json)")
    p.add_argument("--format", choices=("text", "json", "markdown", "junit"), default=None, help="output format")
    p.add_argument("--explain", action="store_true", help="add a plain-English explanation of the verdict")
    p.add_argument(
        "--github-summary", action="store_true",
        help="also append a Markdown report to $GITHUB_STEP_SUMMARY when it is set",
    )


def _gate_rounds(args: argparse.Namespace) -> List[Round]:
    files = args.rounds
    if files.count("-") > 1:
        raise InputError("standard input ('-') can be used only once")
    if not args.per_trial:
        if args.src != "auto" or args.arm or args.scorer:
            raise InputError("--from, --arm and --scorer only apply with --per-trial")
        return [r for f in files for r in load_rounds(f)]
    rounds: List[Round] = []
    for f in files:
        chosen = pick_arm(load_arms(f, args.src, args.scorer), args.arm)
        trials = sum(v[0] for v in chosen.values())
        passes = sum(v[1] for v in chosen.values())
        repeated = sum(1 for v in chosen.values() if v[0] > 1)
        if repeated:
            print(
                f"passgate: warning: {safe_text(f)}: {repeated} item(s) were tried more than once. gate treats every "
                "trial as independent, so its bound is too optimistic here; use 'passgate items' for a cluster-aware bound.",
                file=sys.stderr,
            )
        label = "stdin" if f == "-" else os.path.basename(f)
        rounds.append(_make_round(trials, passes, label, safe_text(f)))
    return rounds


def _render(args: argparse.Namespace, obj: Any) -> None:
    fmt = "json" if args.json else (args.format or "text")
    explain = getattr(args, "explain", False) and hasattr(obj, "explain")
    if fmt == "json":
        d = {"passgate_version": __version__, **obj.as_dict()}
        if explain:
            d["explanation"] = obj.explain()
        _emit(json.dumps(d, indent=2))
    elif fmt == "junit":
        _emit(to_junit(obj).rstrip("\n"))
    elif fmt == "markdown":
        md = obj.to_markdown().rstrip("\n")
        _emit(md + (f"\n\n{obj.explain()}" if explain and obj.explain() not in md else ""))
    else:
        _emit(str(obj) + (f"\nIn plain words: {obj.explain()}" if explain else ""))
    if getattr(args, "github_summary", False):
        target = os.environ.get("GITHUB_STEP_SUMMARY")
        if not target:
            print("passgate: --github-summary ignored, GITHUB_STEP_SUMMARY is not set", file=sys.stderr)
            return
        try:
            with open(target, "a", encoding="utf-8") as fh:
                fh.write(obj.to_markdown() + "\n")
        except OSError as e:
            print(f"passgate: could not write step summary: {safe_text(e.strerror or e)}", file=sys.stderr)


def _plan(args: argparse.Namespace) -> int:
    if args.rounds < 1:
        raise InputError("--rounds must be >= 1")
    if args.power is not None:
        return _plan_power(args)
    res = trials_needed(args.rate, args.threshold, args.confidence, args.method)
    if args.json:
        _emit(json.dumps({
            "passgate_version": __version__, "rate": args.rate, "threshold": args.threshold, "confidence": args.confidence,
            "method": args.method, "rounds": args.rounds, "achievable": res is not None,
            "trials_per_round": res[0] if res else None, "min_passes_per_round": res[1] if res else None,
            "total_trials": res[0] * args.rounds if res else None,
        }, indent=2))
    elif res is None:
        _emit(
            f"plan: a {args.rate:.3f} pass rate can never certify >= {args.threshold:.3f}; "
            "raise the rate or lower the bar"
        )
    else:
        n, k = res
        _emit(
            f"plan: run {n} trials per round; a round clears >= {args.threshold:.3f} "
            f"({args.confidence:.0%} confidence, {args.method}) with {k}+ passes.\n"
            f"      {args.rounds} round(s) = {n * args.rounds} trials. This is break-even: at a true rate "
            f"of exactly {args.rate:.3f} a round clears only about half the time, so add headroom."
        )
    return 0 if res is not None else 1


def _plan_power(args: argparse.Namespace) -> int:
    res = plan_power(args.rate, args.threshold, args.power, args.confidence, args.method, args.rounds)
    if args.json:
        _emit(json.dumps({"passgate_version": __version__, "rate": args.rate, "threshold": args.threshold, "power": args.power,
                          "achievable": res is not None, **(res or {})}, indent=2))
    elif res is None:
        _emit(
            f"plan: a {args.rate:.3f} pass rate cannot certify >= {args.threshold:.3f} "
            f"with {args.power:.0%} power within the search limit; raise the rate or lower the bar"
        )
    else:
        _emit(
            f"plan: run {res['trials']} trials per round ({res['min_passes']}+ passes to clear); a candidate whose true "
            f"rate is {args.rate:.3f} then passes {res['power_at_trials']:.1%} of the time"
            + (f" (all {args.rounds} rounds)" if args.rounds > 1 else "")
            + f".\n      {args.rounds} round(s) = {res['total_trials']} trials. "
            f"Break-even is only {res['break_even_trials']}. Sizes below {res['stable_trials']} can dip under the target "
            "again because pass counts are whole numbers."
        )
    return 0 if res is not None else 1


def _bound(args: argparse.Namespace) -> int:
    from .stats import clopper_pearson_interval, wilson_interval

    rows = []
    for text in args.result:
        r = parse_ratio(text)
        if r.trials > 100_000_000:
            raise InputError("bound handles up to 100,000,000 trials")
        w = wilson_interval(r.passes, r.trials, args.confidence)
        c = clopper_pearson_interval(r.passes, r.trials, args.confidence)
        rows.append({"passes": r.passes, "trials": r.trials, "rate": r.passes / r.trials,
                     "wilson_lower": w[0], "wilson_upper": w[1], "exact_lower": c[0], "exact_upper": c[1]})
    if args.json:
        _emit(json.dumps({"passgate_version": __version__, "confidence": args.confidence, "results": rows}, indent=2))
    else:
        for x in rows:
            _emit(
                f"{x['passes']}/{x['trials']} = {x['rate']:.3f}: the true pass rate is between {x['exact_lower']:.3f} and "
                f"{x['exact_upper']:.3f} ({args.confidence:.0%}, exact); Wilson says {x['wilson_lower']:.3f} to {x['wilson_upper']:.3f}. "
                f"The failure rate could still be as high as {1 - x['exact_lower']:.1%}."
            )
    return 0


def _plan_regress(args: argparse.Namespace) -> int:
    cand = args.baseline_rate if args.candidate_rate is None else args.candidate_rate
    if (args.margin is None) == (args.trials is None):
        raise InputError("give exactly one of --margin (find the trials) or --trials (find the margin)")
    if args.trials is not None:
        if not 1 <= args.trials <= 200_000:
            raise InputError("--trials must be between 1 and 200,000")
        m = min_provable_margin(args.trials, args.baseline_rate, args.power, args.alpha, cand)
        if args.json:
            _emit(json.dumps({"passgate_version": __version__, "trials_per_arm": args.trials, "min_margin": m, "power": args.power}, indent=2))
        elif m is None:
            _emit(f"plan-regress: with {args.trials} trials per arm even a 50-point margin cannot be proved with {args.power:.0%} power")
        else:
            _emit(
                f"plan-regress: with {args.trials} trials per arm, 'regress' can prove the candidate is not worse "
                f"than about {m * 100:.1f} points ({args.power:.0%} chance if the true rates are {cand:.3f} and "
                f"{args.baseline_rate:.3f}; one-sided alpha {args.alpha:g}). A smaller margin needs more trials."
            )
        return 0 if m is not None else 1
    res = plan_noninferiority(cand, args.baseline_rate, args.margin, args.power, args.alpha)
    if args.json:
        _emit(json.dumps({"passgate_version": __version__, "achievable": res is not None, **(res or {})}, indent=2))
    elif res is None:
        if cand - args.baseline_rate <= -args.margin:
            why = f"is already {args.margin:g} or more worse, so no sample size can prove it is not worse"
        else:
            why = "would need over 20,000 trials per arm (the search limit)"
        _emit(f"plan-regress: a candidate at {cand:.3f} versus a baseline at {args.baseline_rate:.3f} {why}")
    else:
        n = res["trials_per_arm"]
        _emit(
            f"plan-regress: run {n} trials per arm ({res['total_trials']} in total). A candidate whose true rate is "
            f"{cand:.3f} against a baseline at {args.baseline_rate:.3f} is then declared not worse than {args.margin:g} "
            f"{res['power_at_trials']:.1%} of the time (one-sided alpha {args.alpha:g}).\n"
            "      Assumes independent samples (different items per arm); a paired run on the same items can need fewer, but is not planned here."
        )
    return 0 if res is not None else 1


def _oc(args: argparse.Namespace) -> int:
    curve = oc_curve(args.trials, args.threshold, None, args.confidence, args.method, args.rounds)
    if args.json:
        _emit(json.dumps({"passgate_version": __version__, "trials": args.trials, "threshold": args.threshold, "rounds": args.rounds, "curve": curve}, indent=2))
    else:
        _emit(f"oc: P(gate passes) with {args.trials} trials x {args.rounds} round(s), bar {args.threshold:.3f}")
        for pt in curve:
            bar = "#" * int(round(pt["pass_probability"] * 40))
            _emit(f"  true rate {pt['rate']:.2f}  {pt['pass_probability']:6.1%}  {bar}")
    return 0


def _load_stream(path: str) -> List[bool]:
    text = _read_text(path)
    toks: List[Any]
    stripped = text.strip()
    if stripped.startswith("["):
        try:
            toks = json.loads(stripped)
        except (json.JSONDecodeError, RecursionError):
            raise InputError(f"{safe_text(path)}: not valid JSON") from None
        if not isinstance(toks, list):
            raise InputError(f"{safe_text(path)}: expected a JSON list")
    else:
        toks = [t for t in stripped.replace(",", " ").split() if t]
    if len(toks) > 5_000_000:
        raise InputError("too many outcomes")
    if not toks:
        raise InputError(f"{safe_text(path)}: no outcomes found")
    fast = {"1": True, "0": False, "true": True, "false": False, "pass": True, "fail": False}
    out: List[bool] = []
    for i, t in enumerate(toks, start=1):
        v = fast.get(t) if isinstance(t, str) else None
        out.append(v if v is not None else _to_bool(t, "outcome", f"{safe_text(path)} #{i}"))
    return out


def load_cases(path: str) -> List[Case]:
    """CSV/JSON of per-case counts. Columns: case, cand_passes, cand_trials, base_passes, base_trials."""
    text = _read_text(path)
    recs: List[Dict[str, Any]]
    try:
        if kind_of(path, text) == "csv":
            recs = csv_records(text, safe_text(path))
        else:
            data = json.loads(text)
            if not isinstance(data, list) or not all(isinstance(r, dict) for r in data):
                raise InputError(f"{safe_text(path)}: expected a JSON list of objects")
            recs = [{str(k).lower(): v for k, v in r.items()} for r in data]
    except (json.JSONDecodeError, RecursionError):
        raise InputError(f"{safe_text(path)} is not valid JSON") from None
    if len(recs) > MAX_CASES:
        raise InputError(f"too many cases (limit {MAX_CASES:,})")
    cases: List[Case] = []
    seen = set()
    for i, r in enumerate(recs, start=1):
        where = f"{safe_text(path)} row {i}"
        name = str(r.get("case", r.get("name", r.get("item", f"case{i}"))))
        if name in seen:
            raise InputError(f"{where}: duplicate case {safe_text(name, 40)!r}")
        seen.add(name)
        vals = [_to_int(r.get(k), k, where) for k in ("cand_passes", "cand_trials", "base_passes", "base_trials")]
        cases.append(Case(name, *vals))
    if not cases:
        raise InputError(f"{safe_text(path)}: no cases found")
    return cases


def _paired_data(args: argparse.Namespace) -> List[Tuple[bool, bool]]:
    path = args.paired if args.cmd == "regress" else args.results
    if args.baseline_file or args.candidate_arm or args.baseline_arm or args.src != "auto":
        if args.baseline_file == "-" and path == "-":
            raise InputError("standard input ('-') can be used only once")
        arms_c = load_arms(path, args.src, args.scorer)
        if args.baseline_file:
            cand = pick_arm(arms_c, args.candidate_arm)
            base = pick_arm(load_arms(args.baseline_file, args.src, args.scorer), args.baseline_arm)
        else:
            if not (args.candidate_arm and args.baseline_arm):
                raise InputError("give both --candidate-arm and --baseline-arm (see 'passgate arms FILE'), or --baseline-file")
            cand, base = pick_arm(arms_c, args.candidate_arm), pick_arm(arms_c, args.baseline_arm)
        return pairs_from_arms(cand, base)
    return load_pairs(path)


def _arms(args: argparse.Namespace) -> int:
    arms = load_arms(args.results, args.src, args.scorer)
    for name in sorted(arms):
        items = arms[name]
        t = sum(v[0] for v in items.values())
        p = sum(v[1] for v in items.values())
        _emit(f"arm {safe_text(name, 60)!r}: {len(items)} item(s), {t} trial(s), {p} pass(es)")
    return 0


def _emit(text: str) -> None:
    """Print without ever crashing on a legacy Windows console encoding."""
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(enc, errors="replace").decode(enc))


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    _STDIN_CACHE.clear()  # standard input is read once per invocation, never reused across calls
    try:
        if args.cmd == "gate":
            rounds = _gate_rounds(args)
            result = evaluate_rounds(
                rounds, threshold=args.threshold, confidence=args.confidence,
                min_rounds=args.min_rounds, method=args.method,
            )
            _render(args, result)
            return 0 if result.passed else 1
        if args.cmd == "plan":
            return _plan(args)
        if args.cmd == "oc":
            return _oc(args)
        if args.cmd == "bound":
            return _bound(args)
        if args.cmd == "plan-regress":
            return _plan_regress(args)
        if args.cmd == "arms":
            return _arms(args)
        if args.cmd == "sequential":
            res = run_sequential(_load_stream(args.outcomes), args.threshold, args.alpha, args.max_trials)
            _render(args, res)
            return 0 if res.passed else 1
        if args.cmd == "items":
            its = load_items(args.results, args.src, args.arm, args.scorer)
            ir = evaluate_items(its, args.threshold, args.confidence, args.k, args.metric)
            _render(args, ir)
            return 0 if ir.passed else 1
        if args.cmd == "cases":
            cr = compare_cases(load_cases(args.results), args.alpha, args.adjust)
            _render(args, cr)
            return 0 if cr.passed else 1
        if args.cmd == "regress":
            if args.paired:
                if args.candidate or args.baseline:
                    raise InputError("use either --paired FILE or --candidate/--baseline, not both")
                ni = non_inferiority_paired(_paired_data(args), args.margin, args.alpha)
            else:
                if not args.candidate or not args.baseline:
                    raise InputError("give --candidate and --baseline (PASSES/TRIALS), or --paired FILE")
                ni = non_inferiority([parse_ratio(x) for x in args.candidate], [parse_ratio(x) for x in args.baseline], args.margin, args.alpha)
            _render(args, ni)
            return _exit_for(args, ni.passed, ni.verdict)
        if args.cmd == "paired":
            pc = compare_paired(_paired_data(args), alpha=args.alpha)
            _render(args, pc)
            return _exit_for(args, pc.verdict == "candidate-better", pc.verdict)

        cand = [parse_ratio(x) for x in args.candidate]
        base = [parse_ratio(x) for x in args.baseline]
        cmp_ = compare(cand, base, alpha=args.alpha, test=args.test)
        _render(args, cmp_)
        return _exit_for(args, cmp_.verdict == "candidate-better", cmp_.verdict)
    except (ValueError, OverflowError) as e:  # includes InputError and library validation errors
        print(f"passgate: {safe_text(e, 400)}", file=sys.stderr)
        return 2
    except BrokenPipeError:  # e.g. `passgate ... | head`; the reader left, nothing more to say
        try:
            sys.stdout = open(os.devnull, "w")  # keep the interpreter's exit flush quiet
        except OSError:
            pass
        return 2
    except MemoryError:
        print("passgate: out of memory; the input is too large", file=sys.stderr)
        return 2
    except RecursionError:
        print("passgate: input is nested too deeply", file=sys.stderr)
        return 2
    except Exception as e:  # never let a crash look like a model regression (exit 1)
        print(f"passgate: internal error ({type(e).__name__}: {safe_text(e, 200)}); please report it", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
