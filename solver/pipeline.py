from __future__ import annotations

import time
from typing import Any, Callable, Mapping

from .constraints import apply_constraints, duplicate_letter_indices, normalize_option_letters
from .items import extract_items, extract_option_letters
from .model import (
    ANALYSIS_MAX_NEW_TOKENS,
    EXPLAIN_MAX_NEW_TOKENS,
    ITEM_MAX_NEW_TOKENS,
    ModelBundle,
    generate,
    primary_max_new_tokens,
    rescue_max_new_tokens,
)
from .normalize import normalize_answers
from .parse import parse_answers, parse_single_final
from .prompts import (
    build_analysis_messages,
    build_batch_messages,
    build_explain_messages,
    build_item_messages,
    build_match_resolve_messages,
    build_messages,
    build_repair_item_messages,
    build_rescue_messages,
    explanation_from_analysis,
)
from .verify import verify_answers

STRATEGIES = (
    "baseline",
    "task_json_v1",
    "structured_verify_v1",
    "per_item_v1",
    "analyze_adaptive_v1",
    "analyze_constrained_v1",
    "analyze_per_item_v1",
)
DEFAULT_STRATEGY = "analyze_constrained_v1"

SOFT_DEADLINE_SEC = 26 * 60
RESCUE_RATE_CAP = 0.15
RESCUE_MIN_REMAINING_SEC = 30
EXPLAIN_MIN_REMAINING_SEC = 15
EXPLAIN_RESERVE_BASE_SEC = 4 * 60
EXPLAIN_RESERVE_PER_ROW_SEC = 3

ITEM_MIN_REMAINING_SEC = 1.5
PRESSURE_REMAINING_SEC = 20
BATCH_ITEM_CAP = 3
REPAIR_MAX_NEW_TOKENS = 48

Mode = str


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


def _normalize_strategy(strategy: str) -> str:
    strategy = (strategy or DEFAULT_STRATEGY).strip().lower()
    if strategy == "analyze_per_item_v1":
        return "analyze_constrained_v1"
    return strategy


def choose_mode(n: int, task_type: str) -> Mode:
    task = (task_type or "").lower()
    if n <= 1:
        return "direct"
    if task in {"match_letters", "fill_blanks"}:
        return "per_item"
    if n <= BATCH_ITEM_CAP:
        return "batch"
    return "per_item"


def _item_token_budget(clock: DeadlineClock | None, items_left: int) -> int:
    if clock is None:
        return ITEM_MAX_NEW_TOKENS
    remaining = clock.remaining()
    if remaining <= 0 or items_left <= 0:
        return 32
    per_item = remaining / max(1, items_left)
    return int(max(32, min(ITEM_MAX_NEW_TOKENS, per_item * 24)))


def _can_rescue(
    *,
    allow_rescue: bool,
    parse_or_verify_failed: bool,
    clock: DeadlineClock | None,
    rescue_used: int,
    rescue_budget_rows: int | None,
) -> bool:
    return (
        allow_rescue
        and parse_or_verify_failed
        and (clock is None or clock.remaining() > RESCUE_MIN_REMAINING_SEC)
        and (rescue_budget_rows is None or rescue_used < rescue_budget_rows)
    )


def _stage_meta(
    meta: dict[str, Any],
    *,
    parsed: list[str],
    normalized: list[str],
    final: list[str],
    parse_status: list[str],
) -> None:
    meta["parsed"] = list(parsed)
    meta["normalized"] = list(normalized)
    meta["final"] = list(final)
    meta["parse_status"] = list(parse_status)


def _solve_oneshot(
    row: Mapping[str, Any],
    strategy: str,
    bundle: ModelBundle,
    *,
    n: int,
    clock: DeadlineClock | None,
    allow_rescue: bool,
    rescue_used: int,
    rescue_budget_rows: int | None,
    gen: Callable[..., str],
    meta: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    oneshot = (
        strategy
        if strategy in {"baseline", "task_json_v1", "structured_verify_v1"}
        else "task_json_v1"
    )
    raw = gen(bundle, build_messages(row, n, oneshot), primary_max_new_tokens(n, oneshot))
    meta["raw"] = raw
    meta["raw_generation"] = raw
    meta["generations"] = meta.get("generations", 0) + 1
    answers = parse_answers(raw, n=n)
    parse_status = ["ok" if a else "reject" for a in answers]
    ok, reasons = verify_answers(answers, n, row)
    meta["verify_ok"] = ok
    meta["verify_reasons"] = reasons
    parse_failed = (not ok) or any(s == "reject" for s in parse_status)

    if _can_rescue(
        allow_rescue=allow_rescue,
        parse_or_verify_failed=parse_failed,
        clock=clock,
        rescue_used=rescue_used,
        rescue_budget_rows=rescue_budget_rows,
    ):
        rescue_raw = gen(bundle, build_rescue_messages(row, n), rescue_max_new_tokens(n))
        meta["generations"] = meta.get("generations", 0) + 1
        rescue_answers = parse_answers(rescue_raw, n=n)
        rescue_ok, rescue_reasons = verify_answers(rescue_answers, n, row)
        primary_filled = sum(1 for a in answers if a)
        rescue_filled = sum(1 for a in rescue_answers if a)
        if rescue_ok or rescue_filled >= primary_filled:
            answers = rescue_answers
            ok, reasons = rescue_ok, rescue_reasons
            parse_status = ["rescued" if a else "reject" for a in rescue_answers]
            meta.update(
                rescued=True,
                verify_ok=ok,
                verify_reasons=reasons,
                raw=rescue_raw,
                raw_generation=rescue_raw,
            )

    parsed = list(answers)
    normalized = normalize_answers(parsed, row)
    final = list(normalized)
    ok, reasons = verify_answers(final, n, row)
    meta["verify_ok"] = ok
    meta["verify_reasons"] = reasons
    _stage_meta(meta, parsed=parsed, normalized=normalized, final=final, parse_status=parse_status)
    return final, meta


def _maybe_analyze(
    row: Mapping[str, Any],
    bundle: ModelBundle,
    *,
    n: int,
    clock: DeadlineClock | None,
    gen: Callable[..., str],
    meta: dict[str, Any],
    do_analysis: bool,
) -> str:
    if not do_analysis:
        return ""
    under_pressure = clock is not None and clock.remaining() < PRESSURE_REMAINING_SEC
    if under_pressure or (
        clock is not None and clock.remaining() < ITEM_MIN_REMAINING_SEC * max(n, 1) * 1.2
    ):
        meta["analysis_skipped"] = True
        return ""
    tokens = ANALYSIS_MAX_NEW_TOKENS
    if clock is not None:
        tokens = min(ANALYSIS_MAX_NEW_TOKENS, max(64, int(clock.remaining() * 8)))
    analysis = gen(bundle, build_analysis_messages(row), tokens)
    meta["generations"] = meta.get("generations", 0) + 1
    meta["analysis"] = analysis
    return analysis


def _repair_item(
    row: Mapping[str, Any],
    bundle: ModelBundle,
    *,
    item_text: str,
    failed_raw: str,
    clock: DeadlineClock | None,
    gen: Callable[..., str],
    meta: dict[str, Any],
) -> str | None:
    if clock is not None and clock.remaining() < 3:
        return None
    raw = gen(
        bundle,
        build_repair_item_messages(row, item_text=item_text, failed_raw=failed_raw),
        REPAIR_MAX_NEW_TOKENS,
    )
    meta["generations"] = meta.get("generations", 0) + 1
    meta.setdefault("repairs", 0)
    meta["repairs"] += 1
    meta.setdefault("item_raw", []).append(raw)
    return parse_single_final(raw)


def _solve_per_item(
    row: Mapping[str, Any],
    bundle: ModelBundle,
    *,
    items: list[str],
    analysis: str,
    clock: DeadlineClock | None,
    gen: Callable[..., str],
    meta: dict[str, Any],
    allow_rescue: bool,
    rescue_used: int,
    rescue_budget_rows: int | None,
) -> tuple[list[str], list[str]]:
    n = len(items)
    answers: list[str] = []
    parse_status: list[str] = []
    item_rescues = 0
    for i, item_text in enumerate(items):
        items_left = n - i
        if clock is not None and clock.remaining() < ITEM_MIN_REMAINING_SEC * items_left:
            answers.extend([""] * items_left)
            parse_status.extend(["reject"] * items_left)
            meta["budget_truncated"] = True
            break
        messages = build_item_messages(
            row,
            item_index=i + 1,
            item_text=item_text,
            n=n,
            analysis=analysis or "(none)",
            prior_answers=answers,
        )
        raw = gen(bundle, messages, _item_token_budget(clock, items_left))
        meta["generations"] = meta.get("generations", 0) + 1
        meta.setdefault("item_raw", []).append(raw)
        ans = parse_single_final(raw)
        status = "ok" if ans else "reject"
        if ans is None and _can_rescue(
            allow_rescue=allow_rescue,
            parse_or_verify_failed=True,
            clock=clock,
            rescue_used=rescue_used + item_rescues,
            rescue_budget_rows=rescue_budget_rows,
        ):
            rescued = _repair_item(
                row,
                bundle,
                item_text=item_text,
                failed_raw=raw,
                clock=clock,
                gen=gen,
                meta=meta,
            )
            if rescued is not None:
                ans = rescued
                status = "rescued"
                item_rescues += 1
                meta["rescued"] = True
        answers.append(ans or "")
        parse_status.append(status)
    if len(answers) < n:
        answers.extend([""] * (n - len(answers)))
        parse_status.extend(["reject"] * (n - len(parse_status)))
    return answers[:n], parse_status[:n]


def _solve_batch(
    row: Mapping[str, Any],
    bundle: ModelBundle,
    *,
    items: list[str],
    analysis: str,
    clock: DeadlineClock | None,
    gen: Callable[..., str],
    meta: dict[str, Any],
    allow_rescue: bool,
    rescue_used: int,
    rescue_budget_rows: int | None,
) -> tuple[list[str], list[str]]:
    n = len(items)
    tokens = primary_max_new_tokens(n, "task_json_v1")
    if clock is not None:
        tokens = min(tokens, max(64, int(clock.remaining() * 16)))
    raw = gen(bundle, build_batch_messages(row, items=items, analysis=analysis or "(none)"), tokens)
    meta["generations"] = meta.get("generations", 0) + 1
    meta["raw"] = raw
    meta["raw_generation"] = raw
    answers = parse_answers(raw, n=n)
    parse_status = ["ok" if a else "reject" for a in answers]
    item_rescues = 0
    for i, ans in enumerate(answers):
        if ans:
            continue
        if not _can_rescue(
            allow_rescue=allow_rescue,
            parse_or_verify_failed=True,
            clock=clock,
            rescue_used=rescue_used + item_rescues,
            rescue_budget_rows=rescue_budget_rows,
        ):
            continue
        rescued = _repair_item(
            row,
            bundle,
            item_text=items[i],
            failed_raw=raw,
            clock=clock,
            gen=gen,
            meta=meta,
        )
        if rescued is not None:
            answers[i] = rescued
            parse_status[i] = "rescued"
            item_rescues += 1
            meta["rescued"] = True
    return answers, parse_status


def _resolve_match_conflicts(
    row: Mapping[str, Any],
    bundle: ModelBundle,
    *,
    items: list[str],
    answers: list[str],
    analysis: str,
    clock: DeadlineClock | None,
    gen: Callable[..., str],
    meta: dict[str, Any],
) -> list[str]:
    options = extract_option_letters(str(row.get("context", "") or ""))
    answers = normalize_option_letters(answers, options)
    conflicts = duplicate_letter_indices(answers)
    if not conflicts:
        return answers
    if clock is not None and clock.remaining() < 5:
        return answers
    raw = gen(
        bundle,
        build_match_resolve_messages(
            row,
            items=items,
            answers=answers,
            conflict_indices=conflicts,
            options=options,
            analysis=analysis or "(none)",
        ),
        max(48, 24 * len(conflicts)),
    )
    meta["generations"] = meta.get("generations", 0) + 1
    meta["match_resolve"] = True
    resolved = parse_answers(raw, n=len(conflicts))
    resolved = normalize_option_letters(resolved, options)
    out = list(answers)
    for idx, letter in zip(conflicts, resolved):
        if letter:
            out[idx] = letter
    return out


def _solve_adaptive(
    row: Mapping[str, Any],
    bundle: ModelBundle,
    *,
    items: list[str],
    clock: DeadlineClock | None,
    gen: Callable[..., str],
    meta: dict[str, Any],
    do_analysis: bool,
    use_constraints: bool,
    allow_rescue: bool,
    rescue_used: int,
    rescue_budget_rows: int | None,
) -> tuple[list[str], dict[str, Any]]:
    n = len(items)
    task_type = str(row.get("task_type", "") or "")
    mode = choose_mode(n, task_type)
    meta["mode"] = mode
    parse_status: list[str] = ["reject"] * n

    if mode == "direct":
        answers, meta = _solve_oneshot(
            row,
            "task_json_v1",
            bundle,
            n=n,
            clock=clock,
            allow_rescue=allow_rescue,
            rescue_used=rescue_used,
            rescue_budget_rows=rescue_budget_rows,
            gen=gen,
            meta=meta,
        )
        # oneshot already staged; optionally apply constraints on final
        parsed = list(meta.get("parsed") or answers)
        if use_constraints:
            final = apply_constraints(parsed, row, items=items, enabled=True)
        else:
            final = list(meta.get("normalized") or answers)
        _stage_meta(
            meta,
            parsed=parsed,
            normalized=list(meta.get("normalized") or normalize_answers(parsed, row)),
            final=final,
            parse_status=list(meta.get("parse_status") or parse_status),
        )
        ok, reasons = verify_answers(final, n, row)
        meta["verify_ok"] = ok
        meta["verify_reasons"] = reasons
        meta["constraints"] = use_constraints
        return final, meta

    analysis = _maybe_analyze(
        row, bundle, n=n, clock=clock, gen=gen, meta=meta, do_analysis=do_analysis
    )
    if mode == "batch":
        answers, parse_status = _solve_batch(
            row,
            bundle,
            items=items,
            analysis=analysis,
            clock=clock,
            gen=gen,
            meta=meta,
            allow_rescue=allow_rescue,
            rescue_used=rescue_used,
            rescue_budget_rows=rescue_budget_rows,
        )
    else:
        answers, parse_status = _solve_per_item(
            row,
            bundle,
            items=items,
            analysis=analysis,
            clock=clock,
            gen=gen,
            meta=meta,
            allow_rescue=allow_rescue,
            rescue_used=rescue_used,
            rescue_budget_rows=rescue_budget_rows,
        )
    if task_type.lower() == "match_letters":
        answers = _resolve_match_conflicts(
            row,
            bundle,
            items=items,
            answers=answers,
            analysis=str(meta.get("analysis") or ""),
            clock=clock,
            gen=gen,
            meta=meta,
        )

    parsed = list(answers)
    normalized = normalize_answers(parsed, row)
    if use_constraints:
        final = apply_constraints(parsed, row, items=items, enabled=True)
    else:
        final = list(normalized)
    ok, reasons = verify_answers(final, n, row)
    meta["verify_ok"] = ok
    meta["verify_reasons"] = reasons
    meta["constraints"] = use_constraints
    _stage_meta(meta, parsed=parsed, normalized=normalized, final=final, parse_status=parse_status)
    return final, meta


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
    strategy = _normalize_strategy(strategy)
    known = {
        "baseline",
        "task_json_v1",
        "structured_verify_v1",
        "per_item_v1",
        "analyze_adaptive_v1",
        "analyze_constrained_v1",
    }
    if strategy not in known:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")

    items = extract_items(
        str(row.get("query", "") or ""),
        str(row.get("context", "") or ""),
    )
    n = len(items) or 1
    if not items:
        items = ["item 1"]

    meta: dict[str, Any] = {
        "n": n,
        "strategy": strategy,
        "rescued": False,
        "verify_ok": False,
        "verify_reasons": [],
        "raw": "",
        "raw_generation": "",
        "analysis": "",
        "generations": 0,
        "mode": "direct",
        "parsed": [],
        "normalized": [],
        "final": [],
        "parse_status": [],
    }

    if clock is not None and clock.expired():
        empty = [""] * n
        _stage_meta(
            meta,
            parsed=empty,
            normalized=empty,
            final=empty,
            parse_status=["reject"] * n,
        )
        return {
            "pred": empty,
            "explanation": None,
            "meta": {**meta, "skipped": True, "reason": "deadline"},
        }

    # Ablation C: per_item / adaptive without rescue. D: constrained with rescue.
    if strategy in {"per_item_v1", "analyze_adaptive_v1"}:
        allow_rescue = False

    adaptive = strategy in {
        "per_item_v1",
        "analyze_adaptive_v1",
        "analyze_constrained_v1",
    }

    if adaptive:
        if clock is not None and clock.remaining() < PRESSURE_REMAINING_SEC and n > 8:
            meta["fallback"] = "task_json_v1"
            answers, meta = _solve_oneshot(
                row,
                "task_json_v1",
                bundle,
                n=n,
                clock=clock,
                allow_rescue=allow_rescue,
                rescue_used=rescue_used,
                rescue_budget_rows=rescue_budget_rows,
                gen=gen,
                meta=meta,
            )
            parsed = list(meta.get("parsed") or answers)
            if strategy == "analyze_constrained_v1":
                answers = apply_constraints(parsed, row, items=items, enabled=True)
            _stage_meta(
                meta,
                parsed=parsed,
                normalized=list(meta.get("normalized") or normalize_answers(parsed, row)),
                final=answers,
                parse_status=list(meta.get("parse_status") or ["ok"] * n),
            )
        else:
            answers, meta = _solve_adaptive(
                row,
                bundle,
                items=items,
                clock=clock,
                gen=gen,
                meta=meta,
                do_analysis=strategy != "per_item_v1",
                use_constraints=strategy == "analyze_constrained_v1",
                allow_rescue=allow_rescue,
                rescue_used=rescue_used,
                rescue_budget_rows=rescue_budget_rows,
            )
    else:
        answers, meta = _solve_oneshot(
            row,
            strategy,
            bundle,
            n=n,
            clock=clock,
            allow_rescue=allow_rescue,
            rescue_used=rescue_used,
            rescue_budget_rows=rescue_budget_rows,
            gen=gen,
            meta=meta,
        )

    explanation = None
    if explain and (clock is None or clock.remaining() > EXPLAIN_MIN_REMAINING_SEC):
        analysis = str(meta.get("analysis") or "")
        if analysis:
            explanation = explanation_from_analysis(analysis, answers)
        else:
            explanation = gen(
                bundle,
                build_explain_messages(row, answers, raw=meta.get("raw", "")),
                EXPLAIN_MAX_NEW_TOKENS,
            )
            meta["generations"] = meta.get("generations", 0) + 1

    meta["final"] = list(answers)
    return {"pred": answers, "explanation": explanation, "meta": meta}
