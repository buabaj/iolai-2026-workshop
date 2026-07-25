from __future__ import annotations

import re

_ITEM_LINE = re.compile(
    r"(?m)^\s*(?:\(?\s*\d+\s*\)|\d+\s*[.)]|\[\s*\d+\s*\])\s+\S"
)


def count_items(query: str) -> int:
    q = (query or "").strip()
    if not q:
        return 0

    matches = _ITEM_LINE.findall(q)
    if matches:
        return len(matches)

    parts = re.split(r"\n\s*\n", q, maxsplit=1)
    if len(parts) == 2:
        body = [ln.strip() for ln in parts[1].splitlines() if ln.strip()]
        if body:
            return len(body)

    lines = [ln.strip() for ln in q.splitlines() if ln.strip()]
    return 1 if len(lines) <= 1 else len(lines) - 1
