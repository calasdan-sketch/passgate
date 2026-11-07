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
import sys
import zipfile
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .cluster import Item
from .safe import safe_text

FORMATS = ("auto", "generic", "promptfoo", "inspect")
MAX_FILE_BYTES = 256 << 20  # 256 MiB
MAX_CSV_FIELD = 1 << 20  # one CSV cell may be 1 MiB (the csv module default is 128 KiB)
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
    if path == "-":
        return
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


def kind_of(path: str, text: str) -> str:
    """'csv', 'jsonl' or 'json': by extension when it is known, else by looking at the content."""
    low = path.lower()
    if low.endswith(".csv"):
        return "csv"
    if low.endswith((".jsonl", ".ndjson")):
        return "jsonl"
    if low.endswith(".json"):
        return "json"
    stripped = text.lstrip()
    if not stripped or stripped[0] == "[":
        return "json"
    if stripped[0] == "{":
        lines = [ln for ln in stripped.splitlines() if ln.strip()]
        if len(lines) >= 2:
            try:
                json.loads(lines[0])
                return "jsonl"
            except (json.JSONDecodeError, RecursionError):
                pass
        return "json"
    return "csv"


def csv_records(text: str, where: str = "file") -> List[Dict[str, Any]]:
    """CSV rows as dicts with lower-cased, stripped keys. Detects ',', ';', tab or '|' separators.

    Raises IngestError (never a bare csv.Error) on oversized cells or ragged rows.
    """
    lines = text.splitlines()
    first = lines[0] if lines else ""
    delim = max((",", ";", "\t", "|"), key=first.count)
    if first.count(delim) == 0:
        delim = ","
    out: List[Dict[str, Any]] = []
    old_limit = csv.field_size_limit(MAX_CSV_FIELD)
    try:
        for row in csv.DictReader(io.StringIO(text, newline=""), delimiter=delim, strict=True):
            if None in row:  # more cells than header columns
                raise IngestError(f"{where} row {len(out) + 1}: more cells than header columns")
            out.append({(k or "").strip().lower(): v for k, v in row.items()})
    except csv.Error as e:
        raise IngestError(f"{where}: not valid CSV ({safe_text(e)})") from None
    finally:
        csv.field_size_limit(old_limit)  # do not leave a process-wide setting behind
    return out


def _rows_from_text(path: str, text: str) -> List[Tuple[str, Dict[str, Any]]]:
    kind = kind_of(path, text)
    rows: List[Tuple[str, Dict[str, Any]]] = []
    try:
        if kind == "csv":
            for i, row in enumerate(csv_records(text, safe_text(path)), start=1):
                rows.append((f"row {i}", row))
        elif kind == "jsonl":
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
                    hint = (
                        "; this looks like a rounds file, which is for 'passgate gate'"
                        if isinstance(obj, list) and len(obj) == 2 else ""
                    )
                    raise IngestError(f"item {i}: expected an object{hint}")
                rows.append((f"item {i}", obj))
    except json.JSONDecodeError as e:
        raise IngestError(f"not valid JSON: {safe_text(e)}") from None
    except RecursionError:
        raise IngestError("JSON is nested too deeply") from None
    if len(rows) > MAX_ROWS:
        raise IngestError("too many rows")
    return rows


_STDIN_CACHE: List[str] = []


def read_text(path: str) -> str:
    """Whole text of ``path`` ('-' = standard input), capped at MAX_FILE_BYTES even for special files."""
    if path == "-":
        if not _STDIN_CACHE:
            try:
                data = sys.stdin.buffer.read(MAX_FILE_BYTES + 1)
            except (OSError, ValueError) as e:
                raise IngestError(f"cannot read standard input: {safe_text(e)}") from None
            _STDIN_CACHE.append(_decode(data, "standard input"))
        return _STDIN_CACHE[0]
    _size_ok(path)
    try:
        with open(path, "rb") as fh:
            data = fh.read(MAX_FILE_BYTES + 1)
    except OSError as e:
        raise IngestError(f"cannot read {safe_text(path)}: {safe_text(e.strerror or e)}") from None
    return _decode(data, safe_text(path))


def _decode(data: bytes, what: str) -> str:
    if len(data) > MAX_FILE_BYTES:
        raise IngestError(f"{what} is larger than {MAX_FILE_BYTES >> 20} MiB")
    # Windows PowerShell 5 writes UTF-16 (with a byte-order mark) for `command > file`; accept that too
    codec = "utf-16" if data[:2] in (b"\xff\xfe", b"\xfe\xff") and data[:4] != b"\xff\xfe\x00\x00" else "utf-8-sig"
    try:
        text = data.decode(codec)
    except UnicodeDecodeError:
        raise IngestError(f"{what} is not valid UTF-8 (or UTF-16 with a byte-order mark) text") from None
    if "\x00" in text:
        raise IngestError(f"{what} contains NUL bytes; it is not a text results file")
    return text


def _read(path: str) -> str:
    return read_text(path)


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
    if kind_of(path, text) in ("jsonl", "csv"):
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
                    with zf.open(info) as member:  # read at most the cap, whatever the header claims
                        raw = member.read(MAX_ZIP_MEMBER_BYTES + 1)
                    if len(raw) > MAX_ZIP_MEMBER_BYTES:
                        raise IngestError("archive member is too large")
                    yield json.loads(raw.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError, RecursionError, zipfile.BadZipFile, OSError, EOFError):
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
    if kind_of(path, _read(path)[:4000]) in ("csv", "jsonl"):
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
