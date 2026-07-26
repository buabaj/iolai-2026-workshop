from __future__ import annotations

from typing import Any, Mapping

from .items import detect_n_items, extract_item_sources
from .matching import repair_bijection, solve_matching
from .model import GenStats, ModelBundle, generate_with_stats
from .parse import build_user_content, parse_answers

SYSTEM = (
    "You solve International Linguistics Olympiad problems. "
    "Answer every numbered item. Put each answer on its own line, "
    "in order, with no numbering and no extra text."
)
MAX_NEW_TOKENS = 256


def parse_lines(text: str) -> list[str]:
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]


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
    n = detect_n_items(query, context)
    fallback = extract_item_sources(query, n)

    if task_type == "match_letters" and generate_fn is None:
        try:
            matched = solve_matching(bundle, row, n)
            if matched and len(matched) == n:
                stats = GenStats(
                    prompt_tokens=0,
                    new_tokens=0,
                    hit_max_new=False,
                    eos_limited=True,
                )
                return repair_bijection(matched), "", stats
        except Exception:
            pass

    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": build_user_content(context, query, n, task_type),
        },
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

    pred = parse_answers(raw, n, fallback)
    if task_type == "match_letters":
        pred = repair_bijection(pred)
    return pred, raw, stats
