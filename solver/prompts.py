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
        "Do not repeat the prompt or item number. "
        "Match brackets, colons, and spelling style used in the data."
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

ANALYSIS_SYSTEM = (
    "You analyze International Linguistics Olympiad problems.\n"
    "Use ONLY the data and hints in the problem.\n"
    "Return ONLY a compact JSON object with keys:\n"
    '"mappings", "rules", "constraints", "unresolved".\n'
    "mappings: short strings like \"form → meaning\".\n"
    "rules: objects {\"rule\": \"...\", \"evidence\": [\"example cite\", ...]} "
    "or short strings if needed.\n"
    "constraints: hard restrictions implied by the data (e.g. bijection).\n"
    "unresolved: open questions.\n"
    "Prefer reusable hypotheses tied to evidence over linguistic taxonomy.\n"
    "No markdown fences, no prose outside JSON."
)

ITEM_SYSTEM = (
    "You solve one item from an International Linguistics Olympiad problem.\n"
    "Use ONLY the provided context, analysis, and hints.\n"
    "Match the orthography and answer style of the examples exactly.\n"
    "{task_suffix}\n"
    "Reply with a brief justification if needed, then end with exactly one line:\n"
    "FINAL: <answer>"
)

BATCH_SYSTEM = (
    "You solve several items from an International Linguistics Olympiad problem.\n"
    "Use ONLY the provided context, analysis, and hints.\n"
    "Match the orthography and answer style of the examples exactly.\n"
    "{task_suffix}\n"
    "Return ONLY a JSON array of exactly {n} strings, in item order.\n"
    "No markdown fences."
)

REPAIR_ITEM_SYSTEM = (
    "Extract the single answer for this IOL item.\n"
    "{task_suffix}\n"
    "Reply with exactly one line: FINAL: <answer>"
)

MATCH_RESOLVE_SYSTEM = (
    "Resolve conflicting letter assignments for a matching problem.\n"
    "Each answer must be a distinct option letter from the allowed set.\n"
    "Return ONLY a JSON array of letters for the listed item indices, in order.\n"
    "No markdown fences."
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
    parts = [
        TASK_SUFFIXES.get(
            task_type.lower(),
            "Give exactly what the instruction asks for each item, nothing else.",
        )
    ]
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


def build_analysis_messages(row: Mapping[str, Any]) -> list[dict[str, str]]:
    body = (
        f"task_type={_get(row, 'task_type')}\n"
        f"work_lang={_get(row, 'work_lang')}\n"
        f"task_lang={_get(row, 'task_lang')}\n"
        f"eval_type={_get(row, 'eval_type')}\n\n"
        f"{_get(row, 'context')}\n\n{_get(row, 'query')}"
    )
    return [
        {"role": "system", "content": ANALYSIS_SYSTEM},
        {"role": "user", "content": body.strip()},
    ]


def build_item_messages(
    row: Mapping[str, Any],
    *,
    item_index: int,
    item_text: str,
    n: int,
    analysis: str,
    prior_answers: list[str],
) -> list[dict[str, str]]:
    task_type = _get(row, "task_type")
    eval_type = _get(row, "eval_type")
    system = ITEM_SYSTEM.format(task_suffix=_task_suffix(task_type, eval_type))
    prior = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(prior_answers) if a)
    user = (
        f"Item {item_index} of {n}\n"
        f"task_type={task_type}\n"
        f"work_lang={_get(row, 'work_lang')}\n"
        f"task_lang={_get(row, 'task_lang')}\n\n"
        f"CONTEXT:\n{_get(row, 'context')}\n\n"
        f"QUERY INSTRUCTION:\n{_get(row, 'query')}\n\n"
        f"ANALYSIS:\n{analysis}\n\n"
        f"PRIOR ANSWERS:\n{prior or '(none)'}\n\n"
        f"SOLVE THIS ITEM ONLY:\n{item_text}\n"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user.strip()},
    ]


def build_batch_messages(
    row: Mapping[str, Any],
    *,
    items: list[str],
    analysis: str,
) -> list[dict[str, str]]:
    task_type = _get(row, "task_type")
    eval_type = _get(row, "eval_type")
    n = len(items)
    system = BATCH_SYSTEM.format(n=n, task_suffix=_task_suffix(task_type, eval_type))
    listed = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(items))
    user = (
        f"task_type={task_type}\n"
        f"work_lang={_get(row, 'work_lang')}\n"
        f"task_lang={_get(row, 'task_lang')}\n\n"
        f"CONTEXT:\n{_get(row, 'context')}\n\n"
        f"QUERY INSTRUCTION:\n{_get(row, 'query')}\n\n"
        f"ANALYSIS:\n{analysis}\n\n"
        f"SOLVE THESE {n} ITEMS IN ORDER:\n{listed}\n"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user.strip()},
    ]


def build_repair_item_messages(
    row: Mapping[str, Any],
    *,
    item_text: str,
    failed_raw: str,
) -> list[dict[str, str]]:
    task_type = _get(row, "task_type")
    eval_type = _get(row, "eval_type")
    system = REPAIR_ITEM_SYSTEM.format(task_suffix=_task_suffix(task_type, eval_type))
    user = (
        f"CONTEXT (excerpt):\n{_get(row, 'context')[:1500]}\n\n"
        f"ITEM:\n{item_text}\n\n"
        f"PREVIOUS OUTPUT:\n{(failed_raw or '')[:800]}\n\n"
        "Emit FINAL only."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user.strip()},
    ]


def build_match_resolve_messages(
    row: Mapping[str, Any],
    *,
    items: list[str],
    answers: list[str],
    conflict_indices: list[int],
    options: list[str],
    analysis: str,
) -> list[dict[str, str]]:
    lines = []
    for i in conflict_indices:
        item = items[i] if i < len(items) else f"item {i + 1}"
        prev = answers[i] if i < len(answers) else ""
        lines.append(f"{i + 1}. item={item!r} previous={prev!r}")
    user = (
        f"Allowed letters: {', '.join(options)}\n"
        f"ANALYSIS:\n{analysis}\n\n"
        f"CONTEXT:\n{_get(row, 'context')}\n\n"
        f"Resolve these conflicting items (return {len(conflict_indices)} letters):\n"
        + "\n".join(lines)
    )
    return [
        {"role": "system", "content": MATCH_RESOLVE_SYSTEM},
        {"role": "user", "content": user.strip()},
    ]


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


def explanation_from_analysis(analysis: str, answers: list[str]) -> str:
    text = (analysis or "").strip()
    if not text:
        return "- Inferred pattern from the given examples.\n- Answers: " + ", ".join(answers[:8])
    compact = text.replace("\n", " ")
    if len(compact) > 500:
        compact = compact[:497] + "..."
    return (
        f"- Analysis summary: {compact}\n"
        f"- Produced {len(answers)} answers from the inferred rules."
    )
