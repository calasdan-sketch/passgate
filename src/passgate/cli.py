"""passgate CLI: gate eval rounds, or compare a candidate against a baseline.

Exit codes: 0 = pass, 1 = gate/comparison did not pass, 2 = bad input or usage.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from typing import Any, List, Optional

from . import __version__
from .gate import Round, compare, evaluate_rounds


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
    g.add_argument("--json", action="store_true", help="machine-readable output")

    c = sub.add_parser("compare", help="candidate vs baseline: is the candidate actually better?")
    c.add_argument("--candidate", required=True, action="append", help="PASSES/TRIALS (repeatable, one per round)")
    c.add_argument("--baseline", required=True, action="append", help="PASSES/TRIALS (repeatable)")
    c.add_argument("--alpha", type=float, default=0.05)
    c.add_argument("--json", action="store_true", help="machine-readable output")
    return ap


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
                rounds, threshold=args.threshold, confidence=args.confidence, min_rounds=args.min_rounds
            )
            _emit(json.dumps(result.as_dict(), indent=2) if args.json else str(result))
            return 0 if result.passed else 1

        cand = [parse_ratio(x) for x in args.candidate]
        base = [parse_ratio(x) for x in args.baseline]
        cmp_ = compare(cand, base, alpha=args.alpha)
        _emit(json.dumps(cmp_.as_dict(), indent=2) if args.json else str(cmp_))
        return 0 if cmp_.verdict == "candidate-better" else 1
    except ValueError as e:  # includes InputError and validation errors from the library
        print(f"passgate: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
