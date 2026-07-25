from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping

_OUTER_QUOTE = re.compile(r"^[`'\"“”‘’]+|[`'\"“”‘’]+$")
_LETTER = re.compile(r"^[A-Z]$")
_DIGITS = re.compile(r"^\d+$")


def normalize_answer(answer: str, task_type: str = "") -> str:
    """Standardize valid answers only. Never invent a plausible answer from prose."""
    s = unicodedata.normalize("NFC", answer or "").strip()
    s = _OUTER_QUOTE.sub("", s).strip()
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n+", " ", s).strip()
    if not s:
        return ""
    task = (task_type or "").strip().lower()

    if task == "match_letters":
        letter = s.upper()
        return letter if _LETTER.fullmatch(letter) else ""
    if task == "text_to_num":
        compact = re.sub(r"[\s,]", "", s)
        return compact if _DIGITS.fullmatch(compact) else ""
    return s


def normalize_answers(
    answers: list[str],
    row: Mapping[str, Any] | None = None,
) -> list[str]:
    task_type = ""
    if row is not None:
        task_type = str(row.get("task_type", "") or "")
    return [normalize_answer(a, task_type) for a in answers]
