from __future__ import annotations

import re

_RANGE = re.compile(r"(\d+)\s*[-–—]\s*(\d+)")
_PAREN_NUM = re.compile(r"\((\d+)\)")
_LINE_NUM = re.compile(r"(?m)^\s*(\d+)\s*[.)]")


def _from_ranges(text: str) -> int | None:
    spans = [(int(a), int(b)) for a, b in _RANGE.findall(text or "")]
    if not spans:
        return None
    start, end = max(spans, key=lambda ab: ab[1] - ab[0])
    if end >= start:
        return end - start + 1
    return None


def _from_paren_markers(text: str) -> int | None:
    nums = {int(x) for x in _PAREN_NUM.findall(text or "")}
    if not nums:
        return None
    return max(nums)


def _from_line_numbers(text: str) -> int | None:
    vals = [int(m.group(1)) for m in _LINE_NUM.finditer(text or "")]
    if not vals:
        return None
    run = [vals[0]]
    for n in vals[1:]:
        if n == run[-1] + 1:
            run.append(n)
        elif n in run:
            continue
        else:
            break
    return len(run)


def _query_body_lines(query: str) -> list[str]:
    query = (query or "").strip()
    if not query:
        return []
    parts = re.split(r"\n\s*\n", query, maxsplit=1)
    if len(parts) == 2:
        return [ln.strip() for ln in parts[1].splitlines() if ln.strip()]
    lines = [ln.strip() for ln in query.splitlines() if ln.strip()]
    return lines[1:] if len(lines) > 1 else []


def count_items(query: str, context: str = "") -> int:
    query = query or ""
    context = context or ""

    paren_n = _from_paren_markers(query)
    line_n = _from_line_numbers(query)
    range_n = _from_ranges(query)

    # Numbered answer lines beat note ranges like "coincide with (1–15)".
    if line_n:
        return line_n
    # Header ranges beat incomplete inline blank lists in short excerpts.
    if paren_n and range_n:
        return max(paren_n, range_n)
    if paren_n:
        return paren_n
    if range_n:
        return range_n

    body = _query_body_lines(query)
    if body and not (len(body) == 1 and len(body[0].split()) > 8):
        if any(len(line.split()) <= 6 for line in body):
            return len(body)

    q_lines = [ln.strip() for ln in query.splitlines() if ln.strip()]
    if len(q_lines) <= 2:
        n = _from_line_numbers(context)
        if n:
            return n

    if body:
        return len(body)
    return 1
