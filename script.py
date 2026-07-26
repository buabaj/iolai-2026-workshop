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


def _is_qmark(pred_json: str) -> bool:
    try:
        value = json.loads(pred_json)
        return value == ["?"]
    except Exception:
        return False


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
    try:
        rp = float(bundle.model.generation_config.repetition_penalty)
    except Exception:
        rp = float("nan")
    print(f"generation_config.repetition_penalty={rp}", flush=True)

    deadline = t0 + HARD_LIMIT_SEC - SAFETY_SEC
    print(
        f"loaded {bundle.model_id} in {time.monotonic() - t0:.1f}s "
        f"explain={WRITE_EXPLANATIONS}",
        flush=True,
    )

    n_done = 0
    hit_cap = 0
    soft_at: int | None = None
    solved: list[tuple[object, list[str], str]] = []

    for index, row in df.iterrows():
        if time.monotonic() >= deadline:
            soft_at = int(index)
            print(f"soft deadline at {index}/{len(df)}", flush=True)
            break
        pred, raw, stats = solve_row(row, bundle, max_new_tokens=MAX_NEW_TOKENS)
        hit_cap += int(stats.hit_max_new)
        rows[index] = _row(row, pred, None)
        solved.append((index, pred, raw))
        n_done += 1
        _write(rows, SUBMISSION_CSV_PATH)
        print(
            f"{index + 1}/{len(df)} n={len(pred)} "
            f"new={stats.new_tokens} "
            f"{'HIT_CAP' if stats.hit_max_new else 'eos'} "
            f"elapsed={time.monotonic() - t0:.0f}s",
            flush=True,
        )

    if WRITE_EXPLANATIONS:
        for index, pred, raw in solved:
            if time.monotonic() >= deadline - 60:
                break
            explanation = _explain(bundle, df.loc[index], pred, raw)
            rows[index] = _row(df.loc[index], pred, explanation)
            _write(rows, SUBMISSION_CSV_PATH)

    _write(rows, SUBMISSION_CSV_PATH)
    n_qmark = sum(1 for r in rows if _is_qmark(r["pred"]))
    print(
        f"summary n_rows={len(df)} n_done={n_done} n_qmark={n_qmark} "
        f"hit_cap={hit_cap} soft_deadline_at={soft_at} "
        f"rp={rp} total={time.monotonic() - t0:.0f}s",
        flush=True,
    )
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
