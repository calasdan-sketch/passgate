"""Read per-trial eval results from common tools into item-level counts.

Supported (best effort; formats follow each tool's public docs, and every reader fails closed with a
clear message instead of guessing):

* ``generic``   CSV / JSON / JSONL rows. One outcome per row (``pass``, ``passed``, ``success``, ``correct`` or
                ``ok``) or pre-counted rows (``trials`` + ``passes``). Optional ``item`` / ``id`` / ``case`` and
                ``arm`` / ``variant`` / ``model`` columns.
* ``promptfoo`` ``promptfoo eval -o results.json`` (or ``.jsonl``): rows carry ``success``, ``testIdx``, ``promptIdx``
                and ``provider``. Repeated runs (``--repeat``) share a ``testIdx`` and become trials of one item.
                Provider errors count as failures (never silently dropped).
* ``inspect``   Inspect AI logs: ``.json`` logs and ``.eval`` archives. ``epoch`` repeats of one sample id become
                trials of one item. Scores ``C`` / ``1`` / ``true`` pass; ``I`` / ``N`` / ``0`` / ``false`` fail;
                partial credit (``P`` or fractions) is refused rather than rounded.
"""
from __future__ import annotations

import csv
import io
import json
import os
import zipfile
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .cluster import Item
from .safe import safe_text

FORMATS = ("auto", "generic", "promptfoo", "inspect")
MAX_FILE_BYTES = 1 << 30  # 1 GiB
MAX_ZIP_MEMBER_BYTES = 256 << 20
MAX_ZIP_TOTAL_BYTES = 2 << 30
MAX_ROWS = 20_000_000

Arms = Dict[str, Dict[str, Tuple[int, int]]]  # arm -> item -> (trials, passes)


class IngestError(ValueError):
    """Unreadable or unsupported results file (reported with exit code 2)."""


_PASS_KEYS = ("pass", "passed", "success", "correct", "ok")
_ITEM_KEYS = ("item", "id", "case", "name", "question")
_ARM_KEYS = ("arm", "variant", "model", "prompt")
_TRUE = {"1", "true", "pass", "passed", "yes", "y", "t", "c", "correct"}
_FALSE = {"0", "false", "fail", "failed", "no", "n", "f", "i", "incorrect"}


def _bool(value: Any, where: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _TRUE:
            return True
        if v in _FALSE:
            return False
    raise IngestError(f"{where}: cannot read {value!r} as pass/fail")


def _size_ok(path: str) -> None:
    try:
        if os.path.getsize(path) > MAX_FILE_BYTES:
            raise IngestError(f"{safe_text(path)} is larger than {MAX_FILE_BYTES >> 20} MiB")
    except FileNotFoundError:
        raise IngestError(f"results file not found: {safe_text(path)}") from None
    except OSError as e:
        raise IngestError(f"cannot read {safe_text(path)}: {safe_text(e.strerror or e)}") from None


def _add(arms: Arms, arm: str, item: str, trials: int, passes: int) -> None:
    d = arms.setdefault(arm, {})
    t, p = d.get(item, (0, 0))
    d[item] = (t + trials, p + passes)


def _rows_from_text(path: str, text: str) -> List[Tuple[str, Dict[str, Any]]]:
    low = path.lower()
    rows: List[Tuple[str, Dict[str, Any]]] = []
    try:
        if low.endswith(".csv"):
            for i, row in enumerate(csv.DictReader(io.StringIO(text, newline="")), start=1):
                rows.append((f"row {i}", {(k or "").strip().lower(): v for k, v in row.items()}))
        elif low.endswith((".jsonl", ".ndjson")):
            for i, line in enumerate(text.splitlines(), start=1):
                if line.strip():
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        raise IngestError(f"line {i}: expected a JSON object")
                    rows.append((f"line {i}", obj))
        else:
            data = json.loads(text)
            if not isinstance(data, list):
                raise IngestError("expected a JSON list of rows")
            for i, obj in enumerate(data, start=1):
                if not isinstance(obj, dict):
                    raise IngestError(f"item {i}: expected an object")
                rows.append((f"item {i}", obj))
    except json.JSONDecodeError as e:
        raise IngestError(f"not valid JSON: {safe_text(e)}") from None
    except RecursionError:
        raise IngestError("JSON is nested too deeply") from None
    if len(rows) > MAX_ROWS:
        raise IngestError("too many rows")
    return rows


def _read(path: str) -> str:
    _size_ok(path)
    try:
        with open(path, encoding="utf-8-sig", newline="") as fh:
            return fh.read()
    except UnicodeDecodeError:
        raise IngestError(f"{safe_text(path)} is not valid UTF-8 text") from None
    except OSError as e:
        raise IngestError(f"cannot read {safe_text(path)}: {safe_text(e.strerror or e)}") from None


def _generic(path: str) -> Arms:
    arms: Arms = {}
    for n, (where, rec) in enumerate(_rows_from_text(path, _read(path))):
        rec = {str(k).strip().lower(): v for k, v in rec.items()}
        item = next((str(rec[k]) for k in _ITEM_KEYS if rec.get(k) not in (None, "")), f"#{n + 1}")
        arm = next((str(rec[k]) for k in _ARM_KEYS if rec.get(k) not in (None, "")), "default")
        if rec.get("trials") not in (None, "") and rec.get("passes") not in (None, ""):
            try:
                t, p = int(str(rec["trials"]).strip()), int(str(rec["passes"]).strip())
            except ValueError:
                raise IngestError(f"{where}: trials and passes must be whole numbers") from None
            if t <= 0 or not 0 <= p <= t:
                raise IngestError(f"{where}: passes must be within [0, trials] and trials positive")
            _add(arms, arm, item, t, p)
            continue
        key = next((k for k in _PASS_KEYS if rec.get(k) not in (None, "")), None)
        if key is None:
            raise IngestError(f"{where}: need a pass/passed/success/correct/ok value or trials+passes")
        _add(arms, arm, item, 1, int(_bool(rec[key], where)))
    return arms


def _provider_name(p: Any) -> str:
    if isinstance(p, dict):
        return str(p.get("id") or p.get("label") or "provider")
    return str(p) if p not in (None, "") else "provider"


def _promptfoo(path: str) -> Arms:
    text = _read(path)
    rows: List[Dict[str, Any]]
    if path.lower().endswith((".jsonl", ".ndjson")):
        rows = [r for _, r in _rows_from_text(path, text)]
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise IngestError(f"not valid JSON: {safe_text(e)}") from None
        except RecursionError:
            raise IngestError("JSON is nested too deeply") from None
        res = data.get("results") if isinstance(data, dict) else data
        if isinstance(res, dict):
            res = res.get("results")
        if not isinstance(res, list):
            raise IngestError("not a promptfoo results file (expected results.results[] or a list of rows)")
        rows = res
    arms: Arms = {}
    for i, r in enumerate(rows, start=1):
        if not isinstance(r, dict):
            raise IngestError(f"row {i}: expected an object")
        ok = r.get("success", r.get("pass"))
        if not isinstance(ok, bool):
            raise IngestError(f"row {i}: missing boolean 'success'")
        if "testIdx" in r:
            item = f"test{r['testIdx']}"
        else:
            tc = r.get("testCase")
            desc = r.get("description") or (tc.get("description") if isinstance(tc, dict) else None)
            item = str(desc or f"row{i}")
        arm = f"{_provider_name(r.get('provider'))}|prompt{r.get('promptIdx', 0)}"
        _add(arms, arm, item, 1, int(ok))
    if not arms:
        raise IngestError("no results found")
    return arms


def _inspect_value(v: Any, where: str) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip()
        if s in ("C", "c", "true", "True"):
            return True
        if s in ("I", "N", "i", "n", "false", "False"):
            return False
        raise IngestError(f"{where}: score {safe_text(s, 20)!r} is not a plain pass/fail (partial credit is not guessed)")
    if isinstance(v, (int, float)):
        if v == 1:
            return True
        if v == 0:
            return False
        raise IngestError(f"{where}: fractional score {v!r} is not a plain pass/fail; binarise it in the scorer first")
    raise IngestError(f"{where}: unsupported score value")


def _inspect_samples(path: str) -> Iterable[Dict[str, Any]]:
    _size_ok(path)
    if path.lower().endswith(".eval"):
        try:
            zf = zipfile.ZipFile(path)
        except zipfile.BadZipFile:
            raise IngestError(f"{safe_text(path)} is not a valid .eval archive") from None
        with zf:
            infos = [i for i in zf.infolist() if i.filename.startswith("samples/") and i.filename.endswith(".json")]
            if len(infos) > MAX_ROWS or sum(i.file_size for i in infos) > MAX_ZIP_TOTAL_BYTES:
                raise IngestError("archive is too large")
            for info in infos:
                if info.file_size > MAX_ZIP_MEMBER_BYTES:
                    raise IngestError("archive member is too large")
                try:
                    yield json.loads(zf.read(info).decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError, RecursionError):
                    raise IngestError(f"unreadable sample {safe_text(info.filename, 60)}") from None
        return
    text = _read(path)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise IngestError(f"not valid JSON: {safe_text(e)}") from None
    except RecursionError:
        raise IngestError("JSON is nested too deeply") from None
    samples = data.get("samples") if isinstance(data, dict) else None
    if not isinstance(samples, list):
        raise IngestError("not an Inspect log (no samples list; was it written with --log-format json?)")
    yield from samples


def _inspect(path: str, scorer: Optional[str]) -> Arms:
    arms: Arms = {}
    for n, s in enumerate(_inspect_samples(path), start=1):
        if not isinstance(s, dict):
            raise IngestError(f"sample {n}: expected an object")
        scores = s.get("scores")
        if not isinstance(scores, dict) or not scores:
            raise IngestError(f"sample {n}: no scores (was the run scored?)")
        name = scorer
        if name is None:
            if len(scores) > 1:
                raise IngestError("several scorers present (" + ", ".join(safe_text(k, 30) for k in scores) + "); pick one with --scorer")
            name = next(iter(scores))
        if name not in scores:
            raise IngestError(f"sample {n}: scorer {safe_text(name, 30)!r} not found")
        sc = scores[name]
        value = sc.get("value") if isinstance(sc, dict) else sc
        _add(arms, "default", str(s.get("id", f"#{n}")), 1, int(_inspect_value(value, f"sample {n}")))
    if not arms:
        raise IngestError("no samples found")
    return arms


def sniff(path: str) -> str:
    low = path.lower()
    if low.endswith(".eval"):
        return "inspect"
    if low.endswith((".csv", ".jsonl", ".ndjson")):
        head = _read(path)[:4000]
        return "promptfoo" if ("testIdx" in head and "success" in head) else "generic"
    head = _read(path)[:20000]
    if '"samples"' in head and ('"eval"' in head or '"epoch"' in head):
        return "inspect"
    if '"testIdx"' in head or '"promptIdx"' in head or '"prompts"' in head:
        return "promptfoo"
    return "generic"


def load_arms(path: str, fmt: str = "auto", scorer: Optional[str] = None) -> Arms:
    """Parse ``path`` into ``{arm: {item: (trials, passes)}}``. Raises IngestError on anything unclear."""
    if fmt not in FORMATS:
        raise IngestError(f"format must be one of {', '.join(FORMATS)}")
    if fmt == "auto":
        fmt = sniff(path)
    if fmt == "promptfoo":
        return _promptfoo(path)
    if fmt == "inspect":
        return _inspect(path, scorer)
    arms = _generic(path)
    if not arms:
        raise IngestError(f"{safe_text(path)}: no rows found")
    return arms


def pick_arm(arms: Arms, arm: Optional[str]) -> Dict[str, Tuple[int, int]]:
    if arm is not None:
        if arm not in arms:
            raise IngestError(f"arm {safe_text(arm, 40)!r} not found; available: " + ", ".join(safe_text(a, 40) for a in sorted(arms)))
        return arms[arm]
    if len(arms) > 1:
        raise IngestError("several arms in this file (" + ", ".join(safe_text(a, 40) for a in sorted(arms)) + "); choose one with --arm")
    return next(iter(arms.values()))


def load_items(path: str, fmt: str = "auto", arm: Optional[str] = None, scorer: Optional[str] = None) -> List[Item]:
    chosen = pick_arm(load_arms(path, fmt, scorer), arm)
    return [Item(t, p, name) for name, (t, p) in chosen.items()]


def pairs_from_arms(
    cand: Dict[str, Tuple[int, int]], base: Dict[str, Tuple[int, int]]
) -> List[Tuple[bool, bool]]:
    """Match two arms on item id for a paired test. Needs the same items, one trial each (fail closed)."""
    if set(cand) != set(base):
        only_c, only_b = len(set(cand) - set(base)), len(set(base) - set(cand))
        raise IngestError(f"the two arms do not cover the same items ({only_c} only in candidate, {only_b} only in baseline)")
    out: List[Tuple[bool, bool]] = []
    for name in sorted(cand):
        (tc, pc), (tb, pb) = cand[name], base[name]
        if tc != 1 or tb != 1:
            raise IngestError(
                "paired needs one result per item per arm; this data has repeated trials. "
                "Use 'items' (cluster-aware) or 'regress' on the totals instead"
            )
        out.append((bool(pc), bool(pb)))
    return out
