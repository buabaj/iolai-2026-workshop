from __future__ import annotations

import re

_STRIP_PREFIX = re.compile(r"^\s*(?:\(?\d{1,3}\)?[.):\]]\s*|[-*•]\s+)")
_FENCE = re.compile(r"^```[a-zA-Z]*\s*$")
_CHATTY = re.compile(
    r"^\s*(?:here (?:are|is)\b|answers?\s*:?\s*$|explanation\b|note\b|okay\b|"
    r"solution\b|reasoning\b|analysis\b|translations?\s*:?\s*$|the answers?\b|"
    r"let me\b|first,|so,|therefore\b|thus\b)",
    re.I,
)

TASK_HINTS = {
    "translation": (
        "Each answer is the translation alone -- no source text, no gloss, "
        "no notes, no quotation marks."
    ),
    "match_letters": (
        "Each answer is a single capital letter identifying the match for that "
        "numbered item. Every letter is used exactly once, so no letter may repeat."
    ),
    "fill_blanks": (
        "Each answer is only the missing form that belongs in that blank -- "
        "not the whole line, not the gloss."
    ),
    "text_to_num": "Each answer is written in digits only (e.g. 111).",
    "num_to_text": (
        "Each answer is the number written out in the problem language, words only."
    ),
}


def clean_line(s: str) -> str:
    s = (s or "").strip()
    s = _STRIP_PREFIX.sub("", s)
    s = s.strip().strip("`").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'“”":
        s = s[1:-1].strip()
    return s.strip()


def task_hint(task_type: str) -> str:
    return TASK_HINTS.get((task_type or "").strip().lower(), "")


def build_user_content(context: str, query: str, n: int, task_type: str = "") -> str:
    hint = task_hint(task_type)
    parts = [
        context.strip(),
        "",
        query.strip(),
        "",
        f"There are exactly {n} item{'s' if n != 1 else ''} to answer.",
    ]
    if hint:
        parts.append(hint)
    parts.append(
        f"Put exactly {n} answer line{'s' if n != 1 else ''}, one per item, "
        "in order, with no numbering and no extra text."
    )
    return "\n".join(parts)


def parse_answers(text: str, n: int, fallback: list[str] | None = None) -> list[str]:
    from .items import fit_to_n

    if not text:
        return list(fallback[:n]) if fallback else ["?"] * n

    m = None
    for m2 in re.finditer(r"(?:^|\n)\s*(?:final\s+)?answers?\s*:\s*\n?", text, re.I):
        m = m2
    body = text[m.end() :] if m else text

    numbered: list[tuple[int, str]] = []
    raw: list[str] = []
    for ln in body.splitlines():
        if _FENCE.match(ln):
            continue
        mm = re.match(r"^\s*\(?(\d{1,3})\)?[.):\]]\s*(.+)$", ln.strip())
        if mm:
            val = clean_line(mm.group(2))
            if val and not _CHATTY.match(val):
                numbered.append((int(mm.group(1)), val))
        c = clean_line(ln)
        if c and not _CHATTY.match(c):
            raw.append(c)

    if len(numbered) >= n:
        by_label: dict[int, str] = {}
        for lab, val in numbered:
            by_label[lab] = val
        labs = sorted(by_label)
        if len(labs) >= n:
            return [by_label[lab] for lab in labs[:n]]

    return fit_to_n(raw, n, fallback)
