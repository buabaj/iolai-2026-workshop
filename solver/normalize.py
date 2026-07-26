"""Per-line surface normalizers (Lipas/Hul). No arity force / pad / truncate."""

from __future__ import annotations

import re


def normalize_match_letter(ans: str) -> str:
    ans = ans.strip()
    m = re.fullmatch(r"[\(\[]?([A-Za-z])[\)\]]?[.)]?", ans)
    if m:
        return m.group(1).upper()
    tokens = re.findall(r"\b([A-Za-z])\b", ans)
    if tokens:
        return tokens[-1].upper()
    m = re.search(r"[A-Za-z]", ans)
    return m.group(0).upper() if m else ans


def normalize_text_to_num(ans: str) -> str:
    a = re.sub(r"(?i)^(answer|ans|result)\s*[:=]\s*", "", ans.strip()).strip()
    if re.fullmatch(r"[\d\s+\-*/^=()]+", a.replace(",", "")):
        a = a.replace(",", "").replace(" ", "")
        if "=" in a and " = " not in a:
            a = a.replace("=", " = ")
        return a.strip()
    m = re.search(r"\d+", a)
    return m.group(0) if m and len(a) < 40 else a


def safe_normalize_answers(answers: list[str], task_type: str) -> list[str]:
    """Per-line only. Does not pad, truncate, or reorder."""
    task_type = (task_type or "").strip().lower()
    out: list[str] = []
    for a in answers:
        a = a.strip()
        if task_type == "match_letters":
            a = normalize_match_letter(a)
        elif task_type == "text_to_num":
            a = normalize_text_to_num(a)
        out.append(a)
    return out
