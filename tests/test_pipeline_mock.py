from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solver.pipeline import DEFAULT_STRATEGY, solve_row


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
        if line.startswith("Item ") and " of " in line:
            parts = line.split()
            idx = int(parts[1])
            return f"FINAL: {10 * idx}"
        if line.startswith("SOLVE THESE"):
            # batch path
            count = sum(1 for ln in user.splitlines() if ln[:1].isdigit() and ". " in ln)
            return json.dumps([str(10 * (i + 1)) for i in range(count or n)])
    if "mappings" in system and "rules" in system:
        return json.dumps(
            {
                "mappings": ["a→1"],
                "rules": [{"rule": "a maps to 1", "evidence": ["DATA"]}],
                "constraints": [],
                "unresolved": [],
            }
        )
    if "Resolve conflicting" in system or "distinct option letter" in system:
        return '["A", "B"]'
    if "Emit FINAL only" in user or "Extract the single answer" in system:
        return "FINAL: 99"
    if "FINAL ANSWERS" in system:
        return "reasoning\nFINAL ANSWERS:\n" + "\n".join(f"ans{i}" for i in range(n))
    if '"rules"' in system and '"answers"' in system:
        return json.dumps(
            {"rules": ["r"], "tests": ["t"], "answers": [f"{i + 1}" for i in range(n)]}
        )
    return json.dumps([f"{i + 1}" for i in range(n)])


def test_default_strategy_is_constrained_adaptive():
    assert DEFAULT_STRATEGY == "analyze_constrained_v1"


def test_pipeline_strategies_return_n_answers():
    row = {
        "context": "DATA: a→1",
        "query": "Convert:\n1. a\n2. a",
        "task_type": "text_to_num",
        "eval_type": "single",
    }
    for strategy in (
        "baseline",
        "task_json_v1",
        "structured_verify_v1",
        "per_item_v1",
        "analyze_adaptive_v1",
        "analyze_constrained_v1",
        "analyze_per_item_v1",
    ):
        result = solve_row(row, strategy, _FakeBundle(), generate_fn=_fake_generate)
        assert len(result["pred"]) == 2


def test_pipeline_accepts_valid_digits():
    def gen_digits(bundle, messages, max_new_tokens=512):
        return '["10", "20"]'

    row = {"context": "c", "query": "1. x\n2. y", "task_type": "text_to_num"}
    result = solve_row(row, "task_json_v1", _FakeBundle(), generate_fn=gen_digits)
    assert result["pred"] == ["10", "20"]
    assert result["meta"]["verify_ok"] is True


def test_adaptive_batch_for_small_n():
    calls = {"n": 0}

    def gen(bundle, messages, max_new_tokens=512):
        calls["n"] += 1
        return _fake_generate(bundle, messages, max_new_tokens)

    row = {"context": "c", "query": "1. x\n2. y", "task_type": "text_to_num"}
    result = solve_row(row, "analyze_constrained_v1", _FakeBundle(), generate_fn=gen)
    assert result["meta"]["mode"] == "batch"
    assert result["pred"] == ["10", "20"]
    assert calls["n"] >= 2  # analysis + batch


def test_match_letters_uses_per_item():
    def gen(bundle, messages, max_new_tokens=512):
        system = next(m["content"] for m in messages if m["role"] == "system")
        user = next(m["content"] for m in messages if m["role"] == "user")
        if "mappings" in system:
            return '{"mappings":[],"rules":[],"constraints":[],"unresolved":[]}'
        for line in user.splitlines():
            if line.startswith("Item "):
                idx = int(line.split()[1])
                return f"FINAL: {chr(ord('A') + idx - 1)}"
        return "FINAL: A"

    row = {
        "context": "1. aaa\n2. bbb\n\nA. one\nB. two\n",
        "query": "Determine the correct correspondences.",
        "task_type": "match_letters",
    }
    result = solve_row(row, "analyze_constrained_v1", _FakeBundle(), generate_fn=gen)
    assert result["meta"]["mode"] == "per_item"
    assert result["pred"] == ["A", "B"]
    assert result["meta"]["parsed"] == ["A", "B"]
