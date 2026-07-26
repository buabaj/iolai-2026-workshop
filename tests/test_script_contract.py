from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_script_contract():
    source = (ROOT / "script.py").read_text()
    tree = ast.parse(source)
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "main" in names
    assert 'os.environ["HF_HUB_OFFLINE"] = "1"' in source
    assert 'os.environ["TRANSFORMERS_OFFLINE"] = "1"' in source
    assert "/tmp/data/test.csv" in source
    assert "submission.csv" in source
    assert "trust_remote_code" not in source
    assert "n_qmark" in source
    assert "generation_config.repetition_penalty" in source
    model = (ROOT / "solver" / "model.py").read_text()
    assert '"repetition_penalty": 1.0' in model
    assert "tokenize=False" in model
    assert "max_length=6144" in model
    assert "raise RuntimeError" not in model


def test_parse_hygiene():
    from solver.minimal import parse_lines

    assert parse_lines("a\n\nb\n  c  ") == ["a", "b", "c"]
    assert parse_lines("") == []
    assert parse_lines("Here are the answers:\n1. foo\n2. bar") == ["foo", "bar"]
    assert parse_lines('```\n"baz"\n```') == ["baz"]
    assert parse_lines("1. 111\n2. 222") == ["111", "222"]


def test_solve_row_naive():
    from solver.minimal import MAX_NEW_TOKENS, solve_row
    from solver.model import ModelBundle

    assert MAX_NEW_TOKENS == 512

    def fake_gen(bundle, messages, max_new_tokens):
        assert messages[1]["content"] == "ctx\n\nq"
        return "Here are the answers:\n1. a\n2. b"

    pred, raw, stats = solve_row(
        {"context": "ctx", "query": "q", "task_type": "translation"},
        ModelBundle(tok=None, model=None, model_id="x"),
        generate_fn=fake_gen,
    )
    assert pred == ["a", "b"]
    assert stats.prompt_tokens == 0


def test_solver_surface():
    import solver

    assert hasattr(solver, "solve_row")
    assert hasattr(solver, "generate_with_stats")
