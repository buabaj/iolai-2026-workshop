from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping

_OUTER_QUOTE = re.compile(r"^[`'\"“”‘’]+|[`'\"“”‘’]+$")


def normalize_answer(answer: str, task_type: str = "") -> str:
    s = unicodedata.normalize("NFC", answer or "").strip()
    s = _OUTER_QUOTE.sub("", s).strip()
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n+", " ", s).strip()
    task = (task_type or "").strip().lower()

    if task == "match_letters":
        match = re.search(r"[A-Za-z]", s)
        return match.group(0).upper() if match else s.upper()[:1]
    if task == "text_to_num":
        return re.sub(r"[^\d]", "", s)
    return s


def normalize_answers(
    answers: list[str],
    row: Mapping[str, Any] | None = None,
) -> list[str]:
    task_type = ""
    if row is not None:
        task_type = str(row.get("task_type", "") or "")
    return [normalize_answer(a, task_type) for a in answers]
