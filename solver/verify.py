from __future__ import annotations

import re
from typing import Any, Mapping

_LETTER = re.compile(r"^[A-Za-z]$")
_DIGITS = re.compile(r"^\d+$")
_PROSE = re.compile(r"(?i)^(here is|here are|the answer|i think|```|final answers?)")


def verify_answers(
    answers: list[str],
    n: int,
    row: Mapping[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    task_type = ""
    if row is not None:
        task_type = str(row.get("task_type", "") or "").strip().lower()

    if len(answers) != n:
        reasons.append(f"length {len(answers)} != {n}")

    for i, answer in enumerate(answers):
        s = (answer or "").strip()
        if not s:
            reasons.append(f"item {i}: empty")
            continue
        if _PROSE.match(s) or "```" in s:
            reasons.append(f"item {i}: malformed")
        if s in {str(i), str(i + 1)}:
            reasons.append(f"item {i}: index leak")
        if task_type == "match_letters" and not _LETTER.match(s):
            reasons.append(f"item {i}: not a letter")
        if task_type == "text_to_num" and not _DIGITS.match(re.sub(r"[\s,]", "", s)):
            reasons.append(f"item {i}: not digits")
        if task_type == "num_to_text" and _DIGITS.match(s):
            reasons.append(f"item {i}: expected text")

    return len(reasons) == 0, reasons
