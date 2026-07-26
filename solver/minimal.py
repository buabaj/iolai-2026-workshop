from __future__ import annotations

from typing import Any, Mapping

from .model import GenStats, ModelBundle, generate_with_stats

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
    messages = [
        {"role": "system", "content": SYSTEM},
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
    return parse_lines(raw), raw, stats
