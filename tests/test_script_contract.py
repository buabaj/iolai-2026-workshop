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
    assert "from solver.minimal import" in source
    model = (ROOT / "solver" / "model.py").read_text()
    assert '"repetition_penalty": 1.0' in model


def test_parse_lines():
    from solver.minimal import parse_lines

    assert parse_lines("a\n\nb\n  c  ") == ["a", "b", "c"]
    assert parse_lines("") == []


def test_solver_surface():
    import solver

    assert hasattr(solver, "solve_row")
    assert hasattr(solver, "load_model")
    assert hasattr(solver, "generate")
