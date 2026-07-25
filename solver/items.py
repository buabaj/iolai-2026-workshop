from __future__ import annotations

import re

_RANGE = re.compile(r"(\d+)\s*[-–—]\s*(\d+)")
_PAREN_NUM = re.compile(r"\((\d+)\)")
_LINE_NUM = re.compile(r"(?m)^\s*(\d+)\s*[.)]\s*(.*)$")
_LETTER_LINE = re.compile(r"(?m)^\s*([A-Z])\s*[.)]\s*(.*)$")


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


def _numbered_line_run(text: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    for match in _LINE_NUM.finditer(text or ""):
        num = int(match.group(1))
        payload = (match.group(2) or "").strip()
        if not rows:
            rows.append((num, payload))
            continue
        prev = rows[-1][0]
        if num == prev + 1:
            rows.append((num, payload))
        elif num in {n for n, _ in rows}:
            continue
        else:
            break
    return rows


def _from_line_numbers(text: str) -> int | None:
    rows = _numbered_line_run(text)
    return len(rows) if rows else None


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
    return len(extract_items(query, context)) or 1


def extract_option_letters(context: str) -> list[str]:
    letters: list[str] = []
    for match in _LETTER_LINE.finditer(context or ""):
        letter = match.group(1)
        if letter not in letters:
            letters.append(letter)
    return letters


def extract_items(query: str, context: str = "") -> list[str]:
    query = query or ""
    context = context or ""

    query_lines = _numbered_line_run(query)
    if query_lines:
        return [payload or f"item {num}" for num, payload in query_lines]

    paren_nums = sorted({int(x) for x in _PAREN_NUM.findall(query)})
    range_n = _from_ranges(query)
    if paren_nums:
        n = max(paren_nums)
        if range_n:
            n = max(n, range_n)
        return [f"blank ({k})" for k in range(1, n + 1)]
    if range_n:
        return [f"item {k}" for k in range(1, range_n + 1)]

    body = _query_body_lines(query)
    if body and not (len(body) == 1 and len(body[0].split()) > 8):
        if any(len(line.split()) <= 6 for line in body):
            return body

    q_lines = [ln.strip() for ln in query.splitlines() if ln.strip()]
    if len(q_lines) <= 2:
        ctx_lines = _numbered_line_run(context)
        if ctx_lines:
            return [payload or f"item {num}" for num, payload in ctx_lines]

    if body:
        return body
    return ["item 1"]
