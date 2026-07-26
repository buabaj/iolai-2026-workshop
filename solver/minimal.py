from __future__ import annotations

import re
from typing import Any, Mapping

from .items import count_answer_slots, pad_short_answers, source_fallbacks
from .matching import solve_matching
from .model import GenStats, ModelBundle, generate_with_stats
from .normalize import safe_normalize_answers

SYSTEM_PROMPT = (
    "You solve International Linguistics Olympiad problems. "
    "Answer every numbered item. Put each answer on its own line, "
    "in order, with no numbering and no extra text."
)
MAX_NEW_TOKENS = 512

_LEADING_MARKER = re.compile(r"^\s*(?:\(?\d{1,3}\)?[.):\]]\s*|[-*•]\s+)")
_CODE_FENCE = re.compile(r"^```[a-zA-Z]*\s*$")
_PREAMBLE = re.compile(
    r"^\s*(?:here (?:are|is)\b|answers?\s*:?\s*$|explanation\b|note\b|okay\b|"
    r"solution\b|reasoning\b|analysis\b|translations?\s*:?\s*$|the answers?\b|"
    r"let me\b|first,|so,|therefore\b|thus\b)",
    re.I,
)
_NUMBERED_LINE = re.compile(r"^\s*\(?(\d{1,3})\)?[.):\]]\s*(.+)$")


def clean_answer_text(text: str) -> str:
    text = (text or "").strip()
    text = _LEADING_MARKER.sub("", text)
    text = text.strip().strip("`").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'“”":
        text = text[1:-1].strip()
    return text.strip()


def extract_answer_lines(model_text: str, slot_count: int) -> list[str]:
    labeled: dict[int, str] = {}
    unlabeled: list[str] = []

    for line in (model_text or "").splitlines():
        if not line.strip() or _CODE_FENCE.match(line):
            continue
        if _PREAMBLE.match(line):
            continue

        numbered = _NUMBERED_LINE.match(line.strip())
        if numbered:
            label = int(numbered.group(1))
            value = clean_answer_text(numbered.group(2))
            if value and not _PREAMBLE.match(value):
                labeled[label] = value
            continue

        cleaned = clean_answer_text(line)
        if cleaned and not _PREAMBLE.match(cleaned):
            unlabeled.append(cleaned)

    if labeled:
        slots: list[str | None] = [None] * slot_count
        for label, value in labeled.items():
            if 1 <= label <= slot_count:
                slots[label - 1] = value
        fill_from = 0
        for index in range(slot_count):
            if slots[index] is None and fill_from < len(unlabeled):
                slots[index] = unlabeled[fill_from]
                fill_from += 1
        ordered = [s for s in slots if s is not None]
        leftover = unlabeled[fill_from:]
        return ordered + leftover

    return unlabeled


def _greedy_answers(
    row: Mapping[str, Any],
    bundle: ModelBundle,
    *,
    max_new_tokens: int,
    generate_fn,
) -> tuple[list[str], str, GenStats]:
    context = str(row.get("context", "") or "").strip()
    query = str(row.get("query", "") or "").strip()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{context}\n\n{query}"},
    ]
    if generate_fn is not None:
        raw = generate_fn(bundle, messages, max_new_tokens)
        stats = GenStats(
            prompt_tokens=0,
            new_tokens=0,
            hit_max_new=False,
            eos_limited=True,
        )
    else:
        raw, stats = generate_with_stats(
            bundle, messages, max_new_tokens=max_new_tokens
        )
    slot_count = count_answer_slots(query, context)
    fallbacks = source_fallbacks(query, slot_count)
    answers = pad_short_answers(
        extract_answer_lines(raw, slot_count),
        slot_count,
        fallbacks,
    )
    return answers, raw, stats


def solve_row(
    row: Mapping[str, Any],
    bundle: ModelBundle,
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
    generate_fn=None,
) -> tuple[list[str], str, GenStats]:
    context = str(row.get("context", "") or "").strip()
    query = str(row.get("query", "") or "").strip()
    task_type = str(row.get("task_type", "") or "").strip().lower()
    slot_count = count_answer_slots(query, context)

    if task_type == "match_letters" and generate_fn is None:
        try:
            matched = solve_matching(bundle, row, slot_count)
            if matched and len(matched) == slot_count:
                stats = GenStats(
                    prompt_tokens=0,
                    new_tokens=0,
                    hit_max_new=False,
                    eos_limited=True,
                )
                return (
                    safe_normalize_answers(matched, task_type),
                    "",
                    stats,
                )
        except Exception:
            pass

    answers, raw, stats = _greedy_answers(
        row, bundle, max_new_tokens=max_new_tokens, generate_fn=generate_fn
    )
    return safe_normalize_answers(answers, task_type), raw, stats
