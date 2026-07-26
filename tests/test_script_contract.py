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


def test_detect_n_and_fit_to_n():
    from solver.items import detect_n_items, extract_item_sources, fit_to_n

    q = "1. foo\n2. bar\n3. baz"
    assert detect_n_items(q) == 3
    assert detect_n_items("Translate:\nx\ny") == 2
    assert detect_n_items("Match.", "1. a\n2. b\n3. c") == 3

    assert fit_to_n(["a"], 3, ["x", "y", "z"]) == ["a", "y", "z"]
    assert fit_to_n(["r1", "r2", "a", "b", "c"], 3) == ["a", "b", "c"]
    assert fit_to_n([], 2) == ["?", "?"]
    assert extract_item_sources(q, 3) == ["foo", "bar", "baz"]


def test_matching_parse_and_assignment():
    from solver.matching import best_assignment, parse_matching_block, repair_bijection

    ctx = "1. red\n2. blue\n3. green\n\nA. azul\nB. rojo\nC. verde"
    items, opts = parse_matching_block(ctx)
    assert [i[0] for i in items] == [1, 2, 3]
    assert [o[0] for o in opts] == ["A", "B", "C"]

    cols = best_assignment([[0.1, 0.9, 0.0], [0.8, 0.1, 0.0], [0.0, 0.1, 0.7]])
    assert cols == [1, 0, 2]
    assert repair_bijection(["A", "A", "C"]) == ["A", "B", "C"]
    assert repair_bijection(["A", "B", "C"]) == ["A", "B", "C"]


def test_solve_row_fit_repairs():
    from solver.minimal import solve_row
    from solver.model import ModelBundle

    def fake_gen(bundle, messages, max_new_tokens):
        return "only-one"

    pred, raw, stats = solve_row(
        {
            "context": "",
            "query": "1. aaa\n2. bbb\n3. ccc",
            "task_type": "fill_blanks",
        },
        ModelBundle(tok=None, model=None, model_id="x"),
        generate_fn=fake_gen,
    )
    assert pred == ["only-one", "bbb", "ccc"]
    assert raw == "only-one"
    assert stats.prompt_tokens == 0


def test_solver_surface():
    import solver

    assert hasattr(solver, "solve_row")
    assert hasattr(solver, "detect_n_items")
    assert hasattr(solver, "fit_to_n")
    assert hasattr(solver, "generate_with_stats")
