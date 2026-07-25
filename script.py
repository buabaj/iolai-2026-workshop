from __future__ import annotations

import json
import os
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd

from solver.items import count_items
from solver.model import DEFAULT_MODEL_ID, load_model
from solver.pipeline import (
    DEFAULT_STRATEGY,
    EXPLAIN_RESERVE_BASE_SEC,
    EXPLAIN_RESERVE_PER_ROW_SEC,
    RESCUE_RATE_CAP,
    SOFT_DEADLINE_SEC,
    DeadlineClock,
    solve_row,
)

MODEL_ID = os.environ.get("IOL_MODEL_ID", DEFAULT_MODEL_ID)
STRATEGY = os.environ.get("IOL_STRATEGY", DEFAULT_STRATEGY)
WRITE_EXPLANATIONS = os.environ.get("IOL_EXPLAIN", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
TEST_CSV_PATH = os.environ.get("IOL_TEST_CSV", "/tmp/data/test.csv")
SUBMISSION_CSV_PATH = os.environ.get("IOL_OUT_CSV", "submission.csv")
SOFT_DEADLINE_SECONDS = float(
    os.environ.get("IOL_SOFT_DEADLINE_SEC", str(SOFT_DEADLINE_SEC))
)


def _write_submission(rows: list[dict], path: str) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _empty_record(row, *, with_explanation: bool) -> dict:
    n = count_items(str(row.get("query", "") or "")) or 1
    record = {"id": row["id"], "pred": json.dumps([""] * n, ensure_ascii=False)}
    if with_explanation:
        record["explanation"] = ""
    return record


def main() -> None:
    started = time.monotonic()
    print(f"loading model from {MODEL_ID!r}", flush=True)
    bundle = load_model(MODEL_ID, offline=True)
    print(
        f"loaded {bundle.model_id} in {time.monotonic() - started:.1f}s "
        f"| strategy={STRATEGY} explain={WRITE_EXPLANATIONS}",
        flush=True,
    )

    df = pd.read_csv(TEST_CSV_PATH, dtype=str).fillna("")
    clock = DeadlineClock(SOFT_DEADLINE_SECONDS)
    rescue_budget = max(1, int(RESCUE_RATE_CAP * len(df)))
    rescue_used = 0
    rows: list[dict] = []

    for _, row in df.iterrows():
        if clock.expired():
            print(f"soft deadline at {len(rows)}/{len(df)}; flushing", flush=True)
            for _, remaining in df.iloc[len(rows) :].iterrows():
                rows.append(_empty_record(remaining, with_explanation=WRITE_EXPLANATIONS))
            break

        remaining_rows = len(df) - len(rows)
        explain_budget = EXPLAIN_RESERVE_BASE_SEC + EXPLAIN_RESERVE_PER_ROW_SEC * remaining_rows
        explain_this = WRITE_EXPLANATIONS and clock.remaining() > explain_budget

        result = solve_row(
            row,
            STRATEGY,
            bundle,
            clock=clock,
            explain=explain_this,
            allow_rescue=True,
            rescue_used=rescue_used,
            rescue_budget_rows=rescue_budget,
        )
        if result["meta"].get("rescued"):
            rescue_used += 1

        record = {
            "id": row["id"],
            "pred": json.dumps(result["pred"], ensure_ascii=False),
        }
        if WRITE_EXPLANATIONS:
            record["explanation"] = result.get("explanation") or ""
        rows.append(record)
        _write_submission(rows, SUBMISSION_CSV_PATH)
        print(
            f"{len(rows)}/{len(df)} done | rescued={rescue_used} | "
            f"elapsed={clock.elapsed():.0f}s remaining={clock.remaining():.0f}s",
            flush=True,
        )

    _write_submission(rows, SUBMISSION_CSV_PATH)
    print(f"wrote {SUBMISSION_CSV_PATH} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
