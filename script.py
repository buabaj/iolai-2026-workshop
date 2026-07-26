from __future__ import annotations

import json
import os
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd

from solver.minimal import MAX_NEW_TOKENS, solve_row
from solver.model import DEFAULT_MODEL_ID, assert_gpu_resident, load_model

MODEL_ID = os.environ.get("IOL_MODEL_ID", DEFAULT_MODEL_ID)
WRITE_EXPLANATIONS = os.environ.get("IOL_EXPLAIN", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
TEST_CSV_PATH = os.environ.get("IOL_TEST_CSV", "/tmp/data/test.csv")
SUBMISSION_CSV_PATH = os.environ.get("IOL_OUT_CSV", "submission.csv")
HARD_LIMIT_SEC = float(os.environ.get("IOL_HARD_LIMIT_SEC", str(30 * 60)))
SAFETY_SEC = float(os.environ.get("IOL_SAFETY_SEC", "150"))


def _write(rows: list[dict], path: str) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _row(row, pred: list[str], explanation: str | None) -> dict:
    if not pred:
        pred = ["?"]
    out = {"id": row["id"], "pred": json.dumps(pred, ensure_ascii=False)}
    if WRITE_EXPLANATIONS:
        out["explanation"] = explanation or ""
    return out


def main() -> None:
    t0 = time.monotonic()
    df = pd.read_csv(TEST_CSV_PATH, dtype=str).fillna("")
    rows = [_row(r, ["?"], "") for _, r in df.iterrows()]
    _write(rows, SUBMISSION_CSV_PATH)
    print(f"placeholder {SUBMISSION_CSV_PATH} ({len(rows)} rows)", flush=True)

    bundle = load_model(MODEL_ID, offline=True)
    assert_gpu_resident(bundle)
    from solver.model import _force_greedy

    _force_greedy(bundle.model)

    deadline = t0 + HARD_LIMIT_SEC - SAFETY_SEC
    print(
        f"loaded {bundle.model_id} in {time.monotonic() - t0:.1f}s "
        f"explain={WRITE_EXPLANATIONS}",
        flush=True,
    )

    for index, row in df.iterrows():
        if time.monotonic() >= deadline:
            print(f"soft deadline at {index}/{len(df)}", flush=True)
            break
        pred, raw, _ = solve_row(row, bundle, max_new_tokens=MAX_NEW_TOKENS)
        explanation = None
        if WRITE_EXPLANATIONS and time.monotonic() < deadline - 60:
            explanation = _explain(bundle, row, pred, raw)
        rows[index] = _row(row, pred, explanation)
        _write(rows, SUBMISSION_CSV_PATH)
        print(
            f"{index + 1}/{len(df)} n={len(pred)} "
            f"elapsed={time.monotonic() - t0:.0f}s",
            flush=True,
        )

    _write(rows, SUBMISSION_CSV_PATH)
    print(f"wrote {SUBMISSION_CSV_PATH} total={time.monotonic() - t0:.0f}s", flush=True)


def _explain(bundle, row, pred: list[str], raw: str) -> str:
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
                f"ANSWERS:\n{pred}\n\n"
                f"TRACE:\n{(raw or '')[:2000]}"
            ),
        },
    ]
    try:
        return generate(bundle, messages, max_new_tokens=128)
    except Exception:
        return ""


if __name__ == "__main__":
    main()
