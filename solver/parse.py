from __future__ import annotations

import json
import re
from typing import Any

_FINAL_MARKER = re.compile(r"(?im)^\s*final answers?\s*:?\s*$")
_LINE_PREFIX = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*")
_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _strip_line(line: str) -> str:
    return _LINE_PREFIX.sub("", line).strip()


def _as_str_list(obj: Any) -> list[str] | None:
    if isinstance(obj, list):
        return ["" if x is None else str(x).strip() for x in obj]
    return None


def _extract_json(text: str) -> Any | None:
    cleaned = text.strip()
    fence = _FENCE.search(cleaned)
    if fence:
        cleaned = fence.group(1).strip()

    candidates: list[str] = []
    for opener, closer in (("[", "]"), ("{", "}")):
        start = cleaned.rfind(opener)
        while start != -1:
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(cleaned)):
                ch = cleaned[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        candidates.append(cleaned[start : i + 1])
                        break
            start = cleaned.rfind(opener, 0, start)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def _from_final_answers(text: str) -> list[str] | None:
    markers = list(_FINAL_MARKER.finditer(text))
    if not markers:
        return None
    block = text[markers[-1].end() :]
    answers = [_strip_line(ln) for ln in block.splitlines() if _strip_line(ln)]
    return answers or None


def _from_lines(text: str) -> list[str]:
    answers = []
    for ln in text.splitlines():
        s = _strip_line(ln)
        if not s or s.startswith("```"):
            continue
        low = s.lower()
        if low.startswith("here is") or low.startswith("the answers"):
            continue
        answers.append(s)
    return answers


def parse_answers(text: str, n: int | None = None) -> list[str]:
    text = (text or "").strip()
    answers: list[str] | None = None

    value = _extract_json(text)
    if value is not None:
        if isinstance(value, dict) and "answers" in value:
            answers = _as_str_list(value["answers"])
        else:
            answers = _as_str_list(value)

    if answers is None:
        answers = _from_final_answers(text)
    if answers is None:
        answers = _from_lines(text)

    if n is None:
        return answers
    if len(answers) < n:
        return answers + [""] * (n - len(answers))
    return answers[:n]
