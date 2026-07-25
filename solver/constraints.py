from __future__ import annotations

import re
from typing import Any, Mapping

from .items import extract_option_letters
from .normalize import normalize_answer

_BRACKETED = re.compile(r"^\[.*\]$")
_STYLE_MAJORITY = 0.85
_LETTER = re.compile(r"^[A-Z]$")


def _example_forms(context: str) -> list[str]:
    forms: list[str] = []
    for line in (context or "").splitlines():
        line = line.strip()
        if not line or len(line) > 80:
            continue
        if "|" in line:
            for part in line.split("|"):
                part = part.strip()
                if part and len(part.split()) <= 4:
                    forms.append(part)
        elif len(line.split()) <= 3 and not line[0].isdigit():
            forms.append(line)
    return forms[:40]


def mirror_style(context: str, answer: str, task_type: str = "") -> str:
    """Conservative style mirror: only wrap/strip brackets when examples are nearly unanimous."""
    answer = (answer or "").strip()
    if not answer:
        return answer
    forms = _example_forms(context)
    if len(forms) < 3:
        return answer

    bracket_share = sum(1 for f in forms if _BRACKETED.match(f)) / len(forms)
    if bracket_share >= _STYLE_MAJORITY and not answer.startswith("["):
        return f"[{answer.strip('[]')}]"
    if bracket_share <= (1.0 - _STYLE_MAJORITY) and answer.startswith("[") and answer.endswith("]"):
        inner = answer[1:-1].strip()
        if inner:
            return inner
    return answer


def normalize_option_letters(answers: list[str], options: list[str]) -> list[str]:
    """Keep answer only if it is exactly one allowed option letter. Never invent."""
    option_set = {o.upper() for o in options} if options else set()
    out: list[str] = []
    for ans in answers:
        letter = (ans or "").strip().upper()
        if _LETTER.fullmatch(letter) and (not option_set or letter in option_set):
            out.append(letter)
        else:
            out.append("")
    return out


def duplicate_letter_indices(answers: list[str]) -> list[int]:
    seen: dict[str, int] = {}
    conflicts: list[int] = []
    for i, ans in enumerate(answers):
        letter = (ans or "").strip().upper()
        if not _LETTER.fullmatch(letter):
            continue
        if letter in seen:
            conflicts.append(i)
            if seen[letter] not in conflicts:
                conflicts.append(seen[letter])
        else:
            seen[letter] = i
    return sorted(set(conflicts))


def apply_constraints(
    answers: list[str],
    row: Mapping[str, Any],
    items: list[str] | None = None,
    *,
    enabled: bool = True,
) -> list[str]:
    task_type = str(row.get("task_type", "") or "").lower()
    context = str(row.get("context", "") or "")
    out = [normalize_answer(a, task_type) for a in answers]
    if not enabled:
        return out

    if task_type == "match_letters":
        out = normalize_option_letters(out, extract_option_letters(context))
    elif task_type in {"fill_blanks", "translation"}:
        out = [mirror_style(context, a, task_type) for a in out]

    return out
