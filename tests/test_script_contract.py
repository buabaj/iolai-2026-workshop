from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_script_contract():
    source = (ROOT / "script.py").read_text()
    assert "main" in {
        node.name for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
    }
    assert 'IOL_EXPLAIN", "1"' in source
    assert "EXPLAIN_RESERVE_SEC" in source
    assert "explanation_rate" in source
    assert "ensure_runtime" not in source
    assert "naive_splitlines" not in source
    model = (ROOT / "solver" / "model.py").read_text()
    assert '"repetition_penalty": 1.0' in model
    assert "tokenize=False" in model
    assert "max_length=6144" in model
    minimal = (ROOT / "solver" / "minimal.py").read_text()
    assert "pad_short_answers" in minimal
    assert "solve_matching" in minimal
    assert "safe_normalize_answers" in minimal
    assert "extract_answer_lines_naive" not in minimal
    assert (ROOT / "solver" / "normalize.py").exists()


def test_normalize_surface():
    from solver.normalize import (
        normalize_match_letter,
        normalize_text_to_num,
        safe_normalize_answers,
    )

    assert normalize_match_letter("A.") == "A"
    assert normalize_match_letter("(B)") == "B"
    assert normalize_match_letter("option C") == "C"
    assert normalize_text_to_num("Answer: 42") == "42"
    assert normalize_text_to_num("1,234") == "1234"
    assert safe_normalize_answers(["(A)", "b."], "match_letters") == ["A", "B"]
    assert safe_normalize_answers(["Answer: 42", "x"], "fill_blanks") == [
        "Answer: 42",
        "x",
    ]
    assert len(safe_normalize_answers(["a", "b", "c"], "text_to_num")) == 3


def test_extract_and_pad():
    from solver.items import count_answer_slots, pad_short_answers
    from solver.minimal import extract_answer_lines

    assert count_answer_slots("1. a\n2. b\n3. c") == 3
    assert extract_answer_lines("Here are the answers:\n1. foo\n2. bar", 2) == [
        "foo",
        "bar",
    ]
    assert extract_answer_lines("2. second\n1. first", 2) == ["first", "second"]
    assert pad_short_answers(["x"], 3, ["a", "b", "c"]) == ["x", "b", "c"]


def test_solve_row_pads_and_normalizes_match_greedy():
    from solver.minimal import solve_row
    from solver.model import ModelBundle

    def fake_gen(bundle, messages, max_new_tokens):
        return "A.\n(B)\nC"

    answers, raw, _ = solve_row(
        {
            "context": "1. x\n2. y\n3. z\nA. a\nB. b\nC. c",
            "query": "1. ?\n2. ?\n3. ?",
            "task_type": "match_letters",
        },
        ModelBundle(tok=None, model=None, model_id="x"),
        generate_fn=fake_gen,
    )
    assert answers == ["A", "B", "C"]
    assert raw == "A.\n(B)\nC"


def test_solve_row_text_to_num_normalize():
    from solver.minimal import solve_row
    from solver.model import ModelBundle

    def fake_gen(bundle, messages, max_new_tokens):
        return "Answer: 42\n1,234"

    answers, raw, _ = solve_row(
        {"context": "", "query": "1. a\n2. b", "task_type": "text_to_num"},
        ModelBundle(tok=None, model=None, model_id="x"),
        generate_fn=fake_gen,
    )
    assert answers == ["42", "1234"]
    assert raw == "Answer: 42\n1,234"


def test_solve_row_fill_blanks_unchanged():
    from solver.minimal import solve_row
    from solver.model import ModelBundle

    def fake_gen(bundle, messages, max_new_tokens):
        return "2. beta\n1. alpha"

    answers, raw, _ = solve_row(
        {"context": "", "query": "1. aaa\n2. bbb", "task_type": "fill_blanks"},
        ModelBundle(tok=None, model=None, model_id="x"),
        generate_fn=fake_gen,
    )
    assert answers == ["alpha", "beta"]
    assert raw == "2. beta\n1. alpha"
