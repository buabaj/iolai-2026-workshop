from __future__ import annotations

import re
from typing import Any, Mapping

from .model import GenStats, ModelBundle, generate_with_stats

SYSTEM = (
    "You solve International Linguistics Olympiad problems. "
    "Answer every numbered item. Put each answer on its own line, "
    "in order, with no numbering and no extra text."
)
MAX_NEW_TOKENS = 512

_STRIP_PREFIX = re.compile(r"^\s*(?:\(?\d{1,3}\)?[.):\]]\s*|[-*•]\s+)")
_FENCE = re.compile(r"^```[a-zA-Z]*\s*$")
_CHATTY = re.compile(
    r"^\s*(?:here (?:are|is)\b|answers?\s*:?\s*$|explanation\b|note\b|okay\b|"
    r"solution\b|reasoning\b|analysis\b|translations?\s*:?\s*$|the answers?\b|"
    r"let me\b|first,|so,|therefore\b|thus\b)",
    re.I,
)


def clean_line(s: str) -> str:
    s = (s or "").strip()
    s = _STRIP_PREFIX.sub("", s)
    s = s.strip().strip("`").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'“”":
        s = s[1:-1].strip()
    return s.strip()


def parse_lines(text: str) -> list[str]:
    out: list[str] = []
    for ln in (text or "").splitlines():
        if not ln.strip() or _FENCE.match(ln):
            continue
        if _CHATTY.match(ln):
            continue
        cleaned = clean_line(ln)
        if cleaned and not _CHATTY.match(cleaned):
            out.append(cleaned)
    return out


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
