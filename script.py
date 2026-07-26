from __future__ import annotations

import json
import os
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd

from solver.minimal import MAX_NEW_TOKENS, solve_row
from solver.model import DEFAULT_MODEL_ID, apply_greedy_decoding, assert_gpu_resident, load_model

MODEL_ID = os.environ.get("IOL_MODEL_ID", DEFAULT_MODEL_ID)
WRITE_EXPLANATIONS = os.environ.get("IOL_EXPLAIN", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
TEST_CSV_PATH = os.environ.get("IOL_TEST_CSV", "/tmp/data/test.csv")
SUBMISSION_CSV_PATH = os.environ.get("IOL_OUT_CSV", "submission.csv")
HARD_LIMIT_SEC = float(os.environ.get("IOL_HARD_LIMIT_SEC", str(30 * 60)))
SAFETY_MARGIN_SEC = float(os.environ.get("IOL_SAFETY_SEC", "150"))
EXPLAIN_RESERVE_SEC = float(os.environ.get("IOL_EXPLAIN_RESERVE_SEC", "300"))
EXPLAIN_MAX_NEW_TOKENS = 96


def write_submission(rows: list[dict], path: str) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def submission_row(row, answers: list[str], explanation: str | None) -> dict:
    if not answers:
        answers = ["?"]
    record = {"id": row["id"], "pred": json.dumps(answers, ensure_ascii=False)}
    if WRITE_EXPLANATIONS:
        record["explanation"] = explanation or ""
    return record


def is_placeholder_prediction(pred_json: str) -> bool:
    try:
        return json.loads(pred_json) == ["?"]
    except Exception:
        return False


def main() -> None:
    started = time.monotonic()
    problems = pd.read_csv(TEST_CSV_PATH, dtype=str).fillna("")
    submission = [submission_row(row, ["?"], "") for _, row in problems.iterrows()]
    write_submission(submission, SUBMISSION_CSV_PATH)
    print(f"placeholder {SUBMISSION_CSV_PATH} ({len(submission)} rows)", flush=True)

    bundle = load_model(MODEL_ID, offline=True)
    assert_gpu_resident(bundle)
    apply_greedy_decoding(bundle.model)
    try:
        repetition_penalty = float(bundle.model.generation_config.repetition_penalty)
    except Exception:
        repetition_penalty = float("nan")
    print(f"generation_config.repetition_penalty={repetition_penalty}", flush=True)

    hard_stop = started + HARD_LIMIT_SEC - SAFETY_MARGIN_SEC
    answer_deadline = hard_stop - (EXPLAIN_RESERVE_SEC if WRITE_EXPLANATIONS else 0.0)
    print(
        f"loaded {bundle.model_id} in {time.monotonic() - started:.1f}s "
        f"explain={WRITE_EXPLANATIONS} answer_deadline_reserve="
        f"{EXPLAIN_RESERVE_SEC if WRITE_EXPLANATIONS else 0:.0f}s",
        flush=True,
    )

    rows_completed = 0
    hit_token_cap = 0
    soft_deadline_row: int | None = None
    completed: list[tuple[object, list[str], str]] = []

    for row_index, row in problems.iterrows():
        if time.monotonic() >= answer_deadline:
            soft_deadline_row = int(row_index)
            print(f"soft deadline at {row_index}/{len(problems)}", flush=True)
            break
        answers, raw_text, stats = solve_row(
            row, bundle, max_new_tokens=MAX_NEW_TOKENS
        )
        hit_token_cap += int(stats.hit_max_new)
        submission[row_index] = submission_row(row, answers, None)
        completed.append((row_index, answers, raw_text))
        rows_completed += 1
        write_submission(submission, SUBMISSION_CSV_PATH)
        print(
            f"{row_index + 1}/{len(problems)} n={len(answers)} "
            f"new={stats.new_tokens} "
            f"{'HIT_CAP' if stats.hit_max_new else 'eos'} "
            f"elapsed={time.monotonic() - started:.0f}s",
            flush=True,
        )

    explanations_written = 0
    if WRITE_EXPLANATIONS:
        for row_index, answers, raw_text in completed:
            if time.monotonic() >= hard_stop:
                break
            explanation = write_explanation(
                bundle, problems.loc[row_index], answers, raw_text
            )
            if explanation.strip():
                explanations_written += 1
            submission[row_index] = submission_row(
                problems.loc[row_index], answers, explanation
            )
            write_submission(submission, SUBMISSION_CSV_PATH)

    write_submission(submission, SUBMISSION_CSV_PATH)
    placeholder_rows = sum(
        1 for record in submission if is_placeholder_prediction(record["pred"])
    )
    explanation_rate = explanations_written / max(1, len(problems))
    print(
        f"summary n_rows={len(problems)} n_done={rows_completed} "
        f"n_qmark={placeholder_rows} hit_cap={hit_token_cap} "
        f"soft_deadline_at={soft_deadline_row} rp={repetition_penalty} "
        f"explanation_rate={explanation_rate:.3f} "
        f"total={time.monotonic() - started:.0f}s",
        flush=True,
    )
    print(
        f"wrote {SUBMISSION_CSV_PATH} total={time.monotonic() - started:.0f}s",
        flush=True,
    )


def write_explanation(bundle, row, answers: list[str], raw_text: str) -> str:
    from solver.model import generate

    messages = [
        {
            "role": "system",
            "content": (
                "Write 2-4 short bullet points explaining the answer. "
                "Human-readable, not a chain of thought."
            ),
        },
        {
            "role": "user",
            "content": (
                f"CONTEXT:\n{row.get('context', '')}\n\n"
                f"QUERY:\n{row.get('query', '')}\n\n"
                f"ANSWERS:\n{answers}\n\n"
                f"TRACE:\n{(raw_text or '')[:2000]}"
            ),
        },
    ]
    try:
        return generate(bundle, messages, max_new_tokens=EXPLAIN_MAX_NEW_TOKENS)
    except Exception:
        return ""


if __name__ == "__main__":
    main()
