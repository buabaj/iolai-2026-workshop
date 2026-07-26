from __future__ import annotations

import re
from typing import Any

from .model import ModelBundle

_OPT_LINE = re.compile(r"^[ \t]*([A-Za-z])[.)]\s+(.+)$", re.M)
_ITEM_LINE = re.compile(r"^[ \t]*(\d{1,3})[.)]\s+(.+)$", re.M)


def parse_matching_block(
    context: str,
) -> tuple[list[tuple[int, str]], list[tuple[str, str]]]:
    items = [(int(a), b.strip()) for a, b in _ITEM_LINE.findall(context or "")]
    opts = [(a.upper(), b.strip()) for a, b in _OPT_LINE.findall(context or "")]
    seen_items: set[int] = set()
    items = [x for x in items if not (x[0] in seen_items or seen_items.add(x[0]))]
    seen_opts: set[str] = set()
    opts = [x for x in opts if not (x[0] in seen_opts or seen_opts.add(x[0]))]
    return items, opts


def best_assignment(score: list[list[float]]) -> list[int]:
    n = len(score)
    m = len(score[0]) if score else 0
    if n == 0 or m == 0:
        return []
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        _, cols = linear_sum_assignment(-np.array(score, dtype=float))
        return list(cols)
    except Exception:
        pass

    used: set[int] = set()
    out = [0] * n
    order = sorted(
        range(n),
        key=lambda i: -(
            max(score[i]) - sorted(score[i])[-2] if m > 1 else max(score[i])
        ),
    )
    for i in order:
        j = max(
            (jj for jj in range(m) if jj not in used),
            key=lambda jj: score[i][jj],
            default=0,
        )
        used.add(j)
        out[i] = j
    for _ in range(4):
        improved = False
        for a in range(n):
            for b in range(a + 1, n):
                cur = score[a][out[a]] + score[b][out[b]]
                alt = score[a][out[b]] + score[b][out[a]]
                if alt > cur + 1e-9:
                    out[a], out[b] = out[b], out[a]
                    improved = True
        if not improved:
            break
    return out


def repair_letter_bijection(answers: list[str]) -> list[str]:
    if len(answers) < 3:
        return answers
    if not all(re.fullmatch(r"[A-Z]", a or "") for a in answers):
        return answers
    n = len(answers)
    universe = [chr(ord("A") + i) for i in range(n)]
    if len(set(answers)) == n:
        return answers
    unused = [letter for letter in universe if letter not in set(answers)]
    if not unused:
        return answers
    seen: set[str] = set()
    out: list[str] = []
    for answer in answers:
        if answer in seen and unused:
            out.append(unused.pop(0))
        else:
            seen.add(answer)
            out.append(answer)
    return out


def solve_matching(
    bundle: ModelBundle,
    row: Any,
    slot_count: int,
    *,
    batch_size: int = 4,
) -> list[str] | None:
    """Score (item, option) next-token logprobs and take a 1-1 assignment.

    Returns None on any failure so callers can fall back to greedy decode.
    """
    import torch

    items, opts = parse_matching_block(str(row.get("context", "") or ""))
    if len(items) < 3 or len(opts) < 3 or len(items) != slot_count:
        return None

    letters = [opt[0] for opt in opts]
    candidate_ids: list[list[int]] = []
    for letter in letters:
        ids: set[int] = set()
        for form in (letter, " " + letter):
            tokens = bundle.tok.encode(form, add_special_tokens=False)
            if tokens:
                ids.add(int(tokens[0]))
        if not ids:
            return None
        candidate_ids.append(sorted(ids))

    context = str(row.get("context", "") or "").strip()
    prompts: list[str] = []
    for number, item_text in items:
        messages = [
            {
                "role": "system",
                "content": (
                    "You match items to their correct counterparts in a "
                    "linguistics problem. Reply with one option letter only."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{context}\n\nWhich lettered option corresponds to item "
                    f"{number} ({item_text})? Reply with the option letter only."
                ),
            },
        ]
        prompts.append(
            bundle.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )

    score: list[list[float]] = []
    try:
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            encoded = bundle.tok(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=6144,
            )
            encoded = {k: v.to(bundle.model.device) for k, v in encoded.items()}
            with torch.no_grad():
                logits = bundle.model(**encoded).logits[:, -1, :].float()
            logprobs = torch.log_softmax(logits, dim=-1)
            for batch_index in range(len(chunk)):
                score.append(
                    [
                        max(float(logprobs[batch_index, token_id].item()) for token_id in ids)
                        for ids in candidate_ids
                    ]
                )
    except Exception:
        return None

    columns = best_assignment(score)
    if len(columns) != slot_count:
        return None
    return repair_letter_bijection([letters[col] for col in columns])
