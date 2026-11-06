"""passgate CLI: gate eval rounds, compare a candidate against a baseline, or plan sample sizes.

Exit codes: 0 = pass, 1 = gate/comparison did not pass, 2 = bad input or usage.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from . import __version__
from .gate import Round, compare, compare_paired, evaluate_rounds
from .safe import safe_text
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
        raise InputError(f"{where}: {what} must be a whole number, got {value!r}") from None


def _make_round(trials: Any, passes: Any, label: str, where: str) -> Round:
    t = _to_int(trials, "trials", where)
    p = _to_int(passes, "passes", where)
    try:
        return Round(trials=t, passes=p, label=label)
    except ValueError as e:
        raise InputError(f"{where}: {e}") from None


def load_rounds(path: str) -> List[Round]:
    """JSON list of {trials, passes} (or [passes, trials] pairs) or CSV (trials,passes[,label]).

    Files may carry a UTF-8 BOM (common from Windows tools) and CRLF line endings.
    """
    rounds: List[Round] = []
    try:
        if path.lower().endswith(".csv"):
            with open(path, newline="", encoding="utf-8-sig") as fh:
                for i, row in enumerate(csv.DictReader(fh), start=1):
                    row = {(k or "").strip().lower(): v for k, v in row.items()}
                    trials = row.get("trials") if row.get("trials") not in (None, "") else row.get("n")
                    passes = row.get("passes") if row.get("passes") not in (None, "") else row.get("k")
                    rounds.append(
                        _make_round(trials, passes, (row.get("label") or "").strip(), f"{path} row {i}")
                    )
            return rounds
        with open(path, encoding="utf-8-sig") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        raise InputError(f"rounds file not found: {path}") from None
    except OSError as e:
        raise InputError(f"cannot read {path}: {e.strerror or e}") from None
    except UnicodeDecodeError:
        raise InputError(f"{path} is not valid UTF-8 text") from None
    except json.JSONDecodeError as e:
        raise InputError(f"{path} is not valid JSON: {e}") from None
    except RecursionError:
        raise InputError(f"{path}: JSON is nested too deeply") from None
    if not isinstance(data, list):
        raise InputError(f"{path}: expected a JSON list of rounds")
    for i, item in enumerate(data, start=1):
        where = f"{path} round {i}"
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
    raise InputError(f"{where}: {what} must be pass/fail (1/0, true/false), got {value!r}")


def _read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8-sig", newline="") as fh:
            return fh.read()
    except FileNotFoundError:
        raise InputError(f"results file not found: {path}") from None
    except OSError as e:
        raise InputError(f"cannot read {path}: {e.strerror or e}") from None
    except UnicodeDecodeError:
        raise InputError(f"{path} is not valid UTF-8 text") from None


def load_pairs(path: str) -> List[Tuple[bool, bool]]:
    """Per-item results for a paired comparison.

    CSV with ``candidate`` and ``baseline`` columns (optional ``item``), a JSON
    list of ``{"item", "candidate", "baseline"}`` objects, or JSONL (one object
    per line). Values: 1/0, true/false, pass/fail. Duplicate item ids are
    rejected so an item cannot be double-counted.
    """
    text = _read_text(path)
    records: List[Tuple[str, Dict[str, Any]]] = []
    low = path.lower()
    if low.endswith(".csv"):
        reader = csv.DictReader(text.splitlines())
        for i, row in enumerate(reader, start=1):
            records.append((f"{path} row {i}", {(k or "").strip().lower(): v for k, v in row.items()}))
    elif low.endswith(".jsonl") or low.endswith(".ndjson"):
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
    g.add_argument("rounds", help="JSON or CSV rounds file")
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

    pr = sub.add_parser("paired", help="candidate vs baseline on the SAME items (exact McNemar)")
    pr.add_argument("results", help="CSV/JSON/JSONL with per-item candidate and baseline pass/fail")
    pr.add_argument("--alpha", type=float, default=0.05)
    _add_output(pr)

    pl = sub.add_parser("plan", help="how many trials per round to certify a threshold at an expected rate")
    pl.add_argument("--rate", type=float, required=True, help="pass rate you expect per round, e.g. 0.95")
    pl.add_argument("--threshold", type=float, default=0.9)
    pl.add_argument("--confidence", type=float, default=0.95)
    pl.add_argument("--method", choices=METHODS, default="wilson")
    pl.add_argument("--rounds", type=int, default=1, help="rounds you will run (for the total)")
    pl.add_argument("--json", action="store_true", help="machine-readable output")
    return ap


def _add_output(p: argparse.ArgumentParser) -> None:
    p.add_argument("--json", action="store_true", help="machine-readable output (same as --format json)")
    p.add_argument("--format", choices=("text", "json", "markdown"), default=None, help="output format")
    p.add_argument(
        "--github-summary", action="store_true",
        help="also append a Markdown report to $GITHUB_STEP_SUMMARY when it is set",
    )


def _render(args: argparse.Namespace, obj: Any) -> None:
    fmt = "json" if args.json else (args.format or "text")
    if fmt == "json":
        _emit(json.dumps(obj.as_dict(), indent=2))
    elif fmt == "markdown":
        _emit(obj.to_markdown().rstrip("\n"))
    else:
        _emit(str(obj))
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
    res = trials_needed(args.rate, args.threshold, args.confidence, args.method)
    if args.json:
        _emit(json.dumps({
            "rate": args.rate, "threshold": args.threshold, "confidence": args.confidence,
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


def _emit(text: str) -> None:
    """Print without ever crashing on a legacy Windows console encoding."""
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(enc, errors="replace").decode(enc))


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.cmd == "gate":
            rounds = load_rounds(args.rounds)
            result = evaluate_rounds(
                rounds, threshold=args.threshold, confidence=args.confidence,
                min_rounds=args.min_rounds, method=args.method,
            )
            _render(args, result)
            return 0 if result.passed else 1
        if args.cmd == "plan":
            return _plan(args)
        if args.cmd == "paired":
            pc = compare_paired(load_pairs(args.results), alpha=args.alpha)
            _render(args, pc)
            return 0 if pc.verdict == "candidate-better" else 1

        cand = [parse_ratio(x) for x in args.candidate]
        base = [parse_ratio(x) for x in args.baseline]
        cmp_ = compare(cand, base, alpha=args.alpha, test=args.test)
        _render(args, cmp_)
        return 0 if cmp_.verdict == "candidate-better" else 1
    except (ValueError, OverflowError) as e:  # includes InputError and library validation errors
        print(f"passgate: {safe_text(e, 400)}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
