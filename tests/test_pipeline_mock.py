from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solver.pipeline import solve_row


class _FakeBundle:
    model_id = "fake"


def _fake_generate(bundle, messages, max_new_tokens=512):
    system = next(m["content"] for m in messages if m["role"] == "system")
    user = next(m["content"] for m in messages if m["role"] == "user")
    n = 1
    for line in user.splitlines():
        if line.startswith("N="):
            n = int(line.split("=", 1)[1])
            break
    if "FINAL ANSWERS" in system:
        return "reasoning\nFINAL ANSWERS:\n" + "\n".join(f"ans{i}" for i in range(n))
    if '"rules"' in system:
        return json.dumps(
            {"rules": ["r"], "tests": ["t"], "answers": [f"ans{i}" for i in range(n)]}
        )
    return json.dumps([f"ans{i}" for i in range(n)])


def test_pipeline_strategies_return_n_answers():
    row = {
        "context": "DATA: a→1",
        "query": "Convert:\n1. a\n2. a",
        "task_type": "text_to_num",
        "eval_type": "single",
    }
    for strategy in ("baseline", "task_json_v1", "structured_verify_v1"):
        result = solve_row(row, strategy, _FakeBundle(), generate_fn=_fake_generate)
        assert len(result["pred"]) == 2


def test_pipeline_accepts_valid_digits():
    def gen_digits(bundle, messages, max_new_tokens=512):
        return '["10", "20"]'

    row = {"context": "c", "query": "1. x\n2. y", "task_type": "text_to_num"}
    result = solve_row(row, "task_json_v1", _FakeBundle(), generate_fn=gen_digits)
    assert result["pred"] == ["10", "20"]
    assert result["meta"]["verify_ok"] is True
