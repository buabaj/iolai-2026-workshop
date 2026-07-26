from __future__ import annotations

import re

_LINE_NUMBER = re.compile(r"^[ \t]*(\d{1,3})[.)\]]", re.M)
_PAREN_NUMBER = re.compile(r"\((\d{1,3})\)")
_ITEM_RANGE = re.compile(r"\(?(\d{1,3})\s*(?:[-–—]|to)\s*(\d{1,3})\)?")
_LINE_LETTER = re.compile(r"^[ \t]*([A-Z])[.)\]]\s", re.M)
_PAREN_LETTER = re.compile(r"\(([A-Z])\)")
_NUMBERED_ANSWER = re.compile(r"^\s*\(?(\d{1,3})\)?[.):\]]\s*(.+)$")


def count_answer_slots(query: str, context: str = "") -> int:
    """How many answers this problem expects. Always >= 1."""
    query = query or ""
    line_nums = [int(m) for m in _LINE_NUMBER.findall(query)]
    paren_nums = [int(m) for m in _PAREN_NUMBER.findall(query)]

    range_count = 0
    for start, end in _ITEM_RANGE.findall(query):
        start_i, end_i = int(start), int(end)
        if 0 < end_i - start_i < 60:
            range_count = max(range_count, end_i - start_i + 1)

    marker_count = max(len(set(line_nums)), len(set(paren_nums)))
    if range_count and marker_count and range_count != marker_count:
        return marker_count
    marker_count = max(
        marker_count,
        len(set(_LINE_LETTER.findall(query))),
        len(set(_PAREN_LETTER.findall(query))),
    )

    slot_count = max(range_count, marker_count)
    if slot_count > 1:
        return slot_count

    lines = [line.strip() for line in query.splitlines() if line.strip()]
    if len(lines) > 1:
        head = lines[0]
        body = lines[1:] if head.endswith((":", ".")) else lines
        if body:
            return len(body)

    if context:
        context_nums = len(set(int(m) for m in _LINE_NUMBER.findall(context)))
        if context_nums > 1:
            return context_nums
        context_letters = len(set(_LINE_LETTER.findall(context)))
        if context_letters > 1:
            return context_letters

    return max(slot_count, 1)


def source_fallbacks(query: str, slot_count: int) -> list[str]:
    """Per-item source text used when the model returns too few lines.

    Empty predictions score zero on EM and chrF. Echoing the query item is
    usually wrong on EM but recovers chrF on fill-blank / transcription tasks.
    """
    query = query or ""
    sources: list[str] = []
    for line in query.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _NUMBERED_ANSWER.match(stripped)
        if match:
            sources.append(match.group(2).strip())
    if not sources:
        lines = [line.strip() for line in query.splitlines() if line.strip()]
        if len(lines) > 1 and lines[0].endswith((":", ".")):
            sources = lines[1:]
    sources = [s.split("|")[0].strip() if "|" in s else s for s in sources]
    sources = [s for s in sources if s]
    while len(sources) < slot_count:
        sources.append(sources[-1] if sources else "?")
    return sources[:slot_count]


def pad_short_answers(
    answers: list[str],
    slot_count: int,
    fallbacks: list[str] | None = None,
) -> list[str]:
    """Pad undersized answer lists only. Never truncate — the grader keeps the first N."""
    slot_count = max(1, int(slot_count))
    padded = [str(a).strip() if a and str(a).strip() else "?" for a in answers]
    while len(padded) < slot_count:
        if fallbacks and len(padded) < len(fallbacks):
            fill = str(fallbacks[len(padded)]).strip() or "?"
        else:
            fill = "?"
        padded.append(fill)
    return padded
