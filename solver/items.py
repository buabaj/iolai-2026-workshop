from __future__ import annotations

import re

_LINE_NUM = re.compile(r"^[ \t]*(\d{1,3})[.)\]]", re.M)
_PAREN_NUM = re.compile(r"\((\d{1,3})\)")
_RANGE = re.compile(r"\(?(\d{1,3})\s*(?:[-–—]|to)\s*(\d{1,3})\)?")
_LINE_LETTER = re.compile(r"^[ \t]*([A-Z])[.)\]]\s", re.M)
_PAREN_LETTER = re.compile(r"\(([A-Z])\)")


def detect_n_items(query: str, context: str = "") -> int:
    q = query or ""
    line_nums = [int(m) for m in _LINE_NUM.findall(q)]
    paren_nums = [int(m) for m in _PAREN_NUM.findall(q)]

    range_n = 0
    for a, b in _RANGE.findall(q):
        a_i, b_i = int(a), int(b)
        if 0 < b_i - a_i < 60:
            range_n = max(range_n, b_i - a_i + 1)

    cand = max(len(set(line_nums)), len(set(paren_nums)))
    if range_n and cand and range_n != cand:
        return cand
    cand = max(
        cand,
        len(set(_LINE_LETTER.findall(q))),
        len(set(_PAREN_LETTER.findall(q))),
    )

    n = max(range_n, cand)
    if n > 1:
        return n

    lines = [ln.strip() for ln in q.splitlines() if ln.strip()]
    if len(lines) > 1:
        head = lines[0]
        body = lines[1:] if head.endswith((":", ".")) else lines
        if body:
            return len(body)

    if context:
        c_nums = len(set(int(m) for m in _LINE_NUM.findall(context)))
        if c_nums > 1:
            return c_nums
        c_lets = len(set(_LINE_LETTER.findall(context)))
        if c_lets > 1:
            return c_lets

    return max(n, 1)


def extract_item_sources(query: str, n: int) -> list[str]:
    q = query or ""
    out: list[str] = []
    for ln in q.splitlines():
        s = ln.strip()
        if not s:
            continue
        m = re.match(r"^\(?(\d{1,3})\)?[.):\]]\s*(.+)$", s)
        if m:
            out.append(m.group(2).strip())
    if not out:
        lines = [ln.strip() for ln in q.splitlines() if ln.strip()]
        if len(lines) > 1 and lines[0].endswith((":", ".")):
            out = lines[1:]
    out = [o.split("|")[0].strip() if "|" in o else o for o in out]
    out = [o for o in out if o]
    while len(out) < n:
        out.append(out[-1] if out else "?")
    return out[:n]


def fit_to_n(
    items: list[str],
    n: int,
    fallback: list[str] | None = None,
) -> list[str]:
    n = max(1, int(n))
    cleaned = [i for i in items if i and str(i).strip()]
    if len(cleaned) > n:
        cleaned = cleaned[-n:]
    while len(cleaned) < n:
        if fallback and len(cleaned) < len(fallback):
            cleaned.append(str(fallback[len(cleaned)]).strip() or "?")
        else:
            cleaned.append(cleaned[-1] if cleaned else "?")
    return [c if c.strip() else "?" for c in cleaned[:n]]
