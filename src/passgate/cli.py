"""passgate CLI: gate eval rounds, or compare a candidate against a baseline."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from typing import List

from . import __version__
from .gate import Round, compare, evaluate_rounds


def parse_ratio(text: str) -> Round:
    """'93/100' -> Round(trials=100, passes=93)."""
    try:
        passes_s, trials_s = text.split("/", 1)
        return Round(trials=int(trials_s), passes=int(passes_s))
    except (ValueError, AttributeError):
        raise SystemExit(f"passgate: bad round {text!r}, expected PASSES/TRIALS e.g. 93/100")


def load_rounds(path: str) -> List[Round]:
    """JSON list of {trials, passes} (or [passes, trials] pairs) or CSV."""
    try:
        if path.endswith(".csv"):
            rounds = []
            with open(path, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    rounds.append(
                        Round(
                            trials=int(row.get("trials") or row.get("n")),
                            passes=int(row.get("passes") or row.get("k")),
                            label=row.get("label", ""),
                        )
                    )
            return rounds
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        raise SystemExit(f"passgate: rounds file not found: {path}")
    rounds = []
    for i, item in enumerate(data):
        if isinstance(item, dict):
            rounds.append(
                Round(
                    trials=int(item["trials"]),
                    passes=int(item["passes"]),
                    label=str(item.get("label", "")),
                )
            )
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            rounds.append(Round(passes=int(item[0]), trials=int(item[1]), label=f"round {i+1}"))
        else:
            raise SystemExit(f"passgate: cannot read round {i+1} from {path}")
    return rounds


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="passgate",
        description="Require an eval candidate to clear a Wilson-bound pass bar on EVERY round.",
    )
    ap.add_argument("--version", action="version", version=f"passgate {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gate", help="pass^N gate over eval rounds; exit 1 if any round is weak")
    g.add_argument("rounds", help="JSON or CSV rounds file")
    g.add_argument("--threshold", type=float, default=0.9, help="required Wilson lower bound (default 0.9)")
    g.add_argument("--confidence", type=float, default=0.95, help="interval confidence (default 0.95)")
    g.add_argument("--min-rounds", type=int, default=1, help="fail if fewer rounds than this")
    g.add_argument("--json", action="store_true")

    c = sub.add_parser("compare", help="candidate vs baseline: is the candidate actually better?")
    c.add_argument("--candidate", required=True, action="append", help="PASSES/TRIALS (repeatable, one per round)")
    c.add_argument("--baseline", required=True, action="append", help="PASSES/TRIALS (repeatable)")
    c.add_argument("--alpha", type=float, default=0.05)
    c.add_argument("--json", action="store_true")

    args = ap.parse_args(argv)

    if args.cmd == "gate":
        try:
            rounds = load_rounds(args.rounds)
        except ValueError as e:
            print(f"passgate: {e}", file=sys.stderr)
            return 2
        result = evaluate_rounds(
            rounds, threshold=args.threshold, confidence=args.confidence, min_rounds=args.min_rounds
        )
        print(json.dumps(result.as_dict(), indent=2) if args.json else result)
        return 0 if result.passed else 1

    cand = [parse_ratio(x) for x in args.candidate]
    base = [parse_ratio(x) for x in args.baseline]
    cmp_ = compare(cand, base, alpha=args.alpha)
    print(json.dumps(cmp_.as_dict(), indent=2) if args.json else cmp_)
    return 0 if cmp_.verdict == "candidate-better" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
