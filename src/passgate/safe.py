"""Neutralise untrusted text (labels, paths, item ids) before it reaches a terminal or report."""
from __future__ import annotations

import unicodedata

# bidi overrides/isolates/marks and zero-width characters: invisible, and able to
# reorder or hide what a reviewer sees ("Trojan Source", CVE-2021-42574)
_INVISIBLE = {
    0x061C, 0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x202A, 0x202B, 0x202C, 0x202D,
    0x202E, 0x2060, 0x2061, 0x2062, 0x2063, 0x2064, 0x2066, 0x2067, 0x2068, 0x2069, 0xFEFF,
}


def safe_text(text: object, limit: int = 80) -> str:
    """Escape control, bidi and zero-width characters as \\uXXXX and cap the length."""
    out = []
    for ch in str(text):
        cp = ord(ch)
        if cp in _INVISIBLE or unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp"):
            out.append(f"\\x{cp:02x}" if cp < 0x100 else f"\\u{cp:04x}" if cp < 0x10000 else f"\\U{cp:08x}")
        else:
            out.append(ch)
    s = "".join(out)
    return s if len(s) <= limit else s[: max(0, limit - 3)] + "..."


def md_cell(text: object, limit: int = 80) -> str:
    """safe_text, plus escaping that keeps a value inside one Markdown table cell as literal text."""
    s = safe_text(text, limit)
    for ch in ("\\", "|", "`", "*", "_", "[", "]", "<", ">", "#", "~"):
        s = s.replace(ch, "\\" + ch)
    return s
