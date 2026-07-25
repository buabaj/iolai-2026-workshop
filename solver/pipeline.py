from __future__ import annotations

import time
from typing import Any, Callable, Mapping

from .items import count_items
from .model import (
    EXPLAIN_MAX_NEW_TOKENS,
    ModelBundle,
    generate,
    primary_max_new_tokens,
    rescue_max_new_tokens,
)
from .normalize import normalize_answers
from .parse import parse_answers
from .prompts import build_explain_messages, build_messages, build_rescue_messages
from .verify import verify_answers

STRATEGIES = ("baseline", "task_json_v1", "structured_verify_v1")
DEFAULT_STRATEGY = "structured_verify_v1"

SOFT_DEADLINE_SEC = 26 * 60
RESCUE_RATE_CAP = 0.15
RESCUE_MIN_REMAINING_SEC = 30
EXPLAIN_MIN_REMAINING_SEC = 15
EXPLAIN_RESERVE_BASE_SEC = 4 * 60
EXPLAIN_RESERVE_PER_ROW_SEC = 3


class DeadlineClock:
    def __init__(self, budget_sec: float = SOFT_DEADLINE_SEC):
        self.start = time.monotonic()
        self.budget_sec = budget_sec

    def elapsed(self) -> float:
        return time.monotonic() - self.start

    def remaining(self) -> float:
        return self.budget_sec - self.elapsed()

    def expired(self) -> bool:
        return self.remaining() <= 0


def _as_mapping(row: Any) -> Mapping[str, Any]:
    return row.to_dict() if hasattr(row, "to_dict") else row


def solve_row(
    row: Any,
    strategy: str,
    bundle: ModelBundle,
    *,
    clock: DeadlineClock | None = None,
    explain: bool = False,
    allow_rescue: bool = True,
    rescue_used: int = 0,
    rescue_budget_rows: int | None = None,
    generate_fn: Callable[..., str] | None = None,
) -> dict[str, Any]:
    gen = generate_fn or (
        lambda b, m, max_new_tokens: generate(b, m, max_new_tokens=max_new_tokens)
    )
    row = _as_mapping(row)
    strategy = (strategy or DEFAULT_STRATEGY).strip().lower()
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")

    n = count_items(str(row.get("query", "") or "")) or 1
    meta: dict[str, Any] = {
        "n": n,
        "strategy": strategy,
        "rescued": False,
        "verify_ok": False,
        "verify_reasons": [],
        "raw": "",
    }

    if clock is not None and clock.expired():
        return {
            "pred": [""] * n,
            "explanation": None,
            "meta": {**meta, "skipped": True, "reason": "deadline"},
        }

    raw = gen(bundle, build_messages(row, n, strategy), primary_max_new_tokens(n, strategy))
    meta["raw"] = raw
    answers = parse_answers(raw, n=n)
    ok, reasons = verify_answers(answers, n, row)
    meta["verify_ok"] = ok
    meta["verify_reasons"] = reasons

    can_rescue = (
        strategy == "structured_verify_v1"
        and allow_rescue
        and not ok
        and (clock is None or clock.remaining() > RESCUE_MIN_REMAINING_SEC)
        and (rescue_budget_rows is None or rescue_used < rescue_budget_rows)
    )
    if can_rescue:
        rescue_raw = gen(bundle, build_rescue_messages(row, n), rescue_max_new_tokens(n))
        rescue_answers = parse_answers(rescue_raw, n=n)
        rescue_ok, rescue_reasons = verify_answers(rescue_answers, n, row)
        primary_filled = sum(1 for a in answers if a)
        rescue_filled = sum(1 for a in rescue_answers if a)
        if rescue_ok or rescue_filled >= primary_filled:
            answers = rescue_answers
            ok, reasons = rescue_ok, rescue_reasons
            meta.update(
                rescued=True,
                verify_ok=ok,
                verify_reasons=reasons,
                raw=rescue_raw,
            )

    answers = normalize_answers(answers, row)
    ok, reasons = verify_answers(answers, n, row)
    meta["verify_ok"] = ok
    meta["verify_reasons"] = reasons

    explanation = None
    if explain and (clock is None or clock.remaining() > EXPLAIN_MIN_REMAINING_SEC):
        explanation = gen(
            bundle,
            build_explain_messages(row, answers, raw=meta.get("raw", "")),
            EXPLAIN_MAX_NEW_TOKENS,
        )

    return {"pred": answers, "explanation": explanation, "meta": meta}
