from __future__ import annotations

from typing import Any, Mapping

TASK_SUFFIXES = {
    "translation": (
        "Each string must be the requested translation only. "
        "Preserve the target language asked by the query. "
        "Prefer the wording pattern seen in the examples."
    ),
    "fill_blanks": (
        "Each string must be the single missing form only. "
        "Do not repeat the prompt or item number."
    ),
    "match_letters": "Each string must be one capital letter only (A, B, C, ...).",
    "text_to_num": "Each string must contain digits only, with no spaces or commentary.",
    "num_to_text": (
        "Each string must be the number written in the puzzle language, "
        "matching spacing and hyphenation patterns seen in the examples."
    ),
}

BASELINE_SYSTEM = (
    "You solve International Linguistics Olympiad problems by reasoning from the "
    "data you are given. You may meet a task type you have never seen: read the "
    "instruction and the examples, and answer in the same form they use. "
    "Common task types and what to give -- "
    "translation: the translated form only, in the language the task asks for; "
    "fill_blanks: only the missing form for each blank; "
    "match_letters: only the option letter (for example A, B, C); "
    "text_to_num: the number in digits; "
    "num_to_text: the number written out in words, in the language asked; "
    "any other type: give exactly what the instruction asks, nothing else. "
    "Reason step by step first. Then write a line that says exactly FINAL ANSWERS: "
    "and, below it, one answer per line in the order the items are asked -- the "
    "bare answer only, no numbering, no quotes, no extra text."
)

JSON_SYSTEM = (
    "You solve International Linguistics Olympiad problems.\n"
    "Use ONLY the data and hints in the problem. Do not use outside knowledge of the language.\n"
    "Infer the system from the examples, then answer every numbered item.\n"
    "Return ONLY a JSON array of strings with exactly {n} answers, in item order.\n"
    "No prose, no numbering, no markdown fences."
)

RESCUE_SYSTEM = (
    "You solve International Linguistics Olympiad problems.\n"
    "Use ONLY the data and hints in the problem. Do not use outside knowledge of the language.\n"
    "Infer rules from the examples, test them mentally against the given data, then answer.\n"
    "Return a JSON object with keys:\n"
    '- "rules": short bullet-style strings describing the inferred mapping\n'
    '- "tests": short checks against given examples\n'
    '- "answers": a JSON array of exactly {n} strings, in item order\n'
    "Keep every field concise. No markdown fences."
)

EXPLAIN_SYSTEM = (
    "Write 2-4 short bullet points explaining the answer: the inferred pattern, "
    "one supporting example from the data, and any ambiguity. "
    "Be human-readable and concise. Not a chain of thought. Not raw reasoning."
)


def _get(row: Mapping[str, Any], key: str, default: str = "") -> str:
    value = row.get(key, default) if hasattr(row, "get") else default
    return default if value is None else str(value).strip()


def _task_suffix(task_type: str, eval_type: str) -> str:
    parts = [TASK_SUFFIXES.get(task_type.lower(), "Give exactly what the instruction asks for each item, nothing else.")]
    if eval_type.lower() == "multi":
        parts.append(
            "Return one best valid answer per item, not multiple alternatives. "
            "Prefer the most canonical order or wording supported by the examples."
        )
    return " ".join(parts)


def _user_block(row: Mapping[str, Any], n: int) -> str:
    header = (
        f"N={n}\n"
        f"task_type={_get(row, 'task_type')}\n"
        f"work_lang={_get(row, 'work_lang')}\n"
        f"task_lang={_get(row, 'task_lang')}\n"
        f"eval_type={_get(row, 'eval_type')}\n"
    )
    return f"{header}\n{_get(row, 'context')}\n\n{_get(row, 'query')}".strip()


def build_messages(
    row: Mapping[str, Any],
    n: int,
    strategy: str,
    *,
    rescue: bool = False,
) -> list[dict[str, str]]:
    strategy = (strategy or "baseline").strip().lower()
    task_type = _get(row, "task_type")
    eval_type = _get(row, "eval_type")

    if strategy == "baseline":
        return [
            {"role": "system", "content": BASELINE_SYSTEM},
            {"role": "user", "content": f"{_get(row, 'context')}\n\n{_get(row, 'query')}"},
        ]

    if rescue:
        system = RESCUE_SYSTEM.format(n=n) + "\n" + _task_suffix(task_type, eval_type)
    else:
        system = JSON_SYSTEM.format(n=n) + "\n" + _task_suffix(task_type, eval_type)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _user_block(row, n)},
    ]


def build_rescue_messages(row: Mapping[str, Any], n: int) -> list[dict[str, str]]:
    return build_messages(row, n, "structured_verify_v1", rescue=True)


def build_explain_messages(
    row: Mapping[str, Any],
    answers: list[str],
    raw: str = "",
) -> list[dict[str, str]]:
    body = (
        f"CONTEXT:\n{_get(row, 'context')}\n\n"
        f"QUERY:\n{_get(row, 'query')}\n\n"
        f"ANSWERS:\n{answers}\n"
    )
    if raw:
        body += f"\nMODEL TRACE (for summarising only):\n{raw[:2000]}"
    return [
        {"role": "system", "content": EXPLAIN_SYSTEM},
        {"role": "user", "content": body},
    ]
