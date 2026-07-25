from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solver.constraints import (
    apply_constraints,
    duplicate_letter_indices,
    mirror_style,
    normalize_option_letters,
)
from solver.items import count_items, extract_items, extract_option_letters
from solver.normalize import normalize_answer
from solver.parse import parse_single_final
from solver.pipeline import choose_mode, solve_row


class _FakeBundle:
    model_id = "fake"


def test_extract_items_numbered_translation():
    query = "Translate into English:\n17. foo bar\n18. baz\n19. qux"
    items = extract_items(query)
    assert items == ["foo bar", "baz", "qux"]
    assert count_items(query) == 3


def test_extract_items_fill_blanks_paren():
    query = (
        "Fill the blanks (1-4):\n\n"
        "her | (1)\n"
        "foo | (2)\n"
        "bar | (3)\n"
        "(4) | it chose\n"
    )
    items = extract_items(query)
    assert items == ["blank (1)", "blank (2)", "blank (3)", "blank (4)"]


def test_extract_items_match_letters_from_context():
    query = "Determine the correct correspondences."
    context = "1. aaa\n2. bbb\n3. ccc\n\nA. one\nB. two\nC. three\n"
    items = extract_items(query, context)
    assert items == ["aaa", "bbb", "ccc"]
    assert extract_option_letters(context) == ["A", "B", "C"]


def test_extract_items_num_range():
    query = "Write the equalities (1-9) in numerals."
    assert extract_items(query) == [f"item {k}" for k in range(1, 10)]


def test_strict_parse_accepts_final_and_json():
    assert parse_single_final("because X\nFINAL: [bø:va]") == "[bø:va]"
    assert parse_single_final("Answer: C") == "C"
    assert parse_single_final('{"answer": "42"}') == "42"
    assert parse_single_final('["A"]') == "A"
    assert parse_single_final("FINAL: anguls") == "anguls"
    assert parse_single_final("FINAL: 4") == "4"


def test_strict_parse_rejects_prose():
    assert parse_single_final("The correct match is C") is None
    assert parse_single_final("1 + 3 = 4") is None
    assert parse_single_final("Therefore the answer is anguls") is None
    assert parse_single_final("") is None
    assert parse_single_final("thinking\nanguls") is None


def test_normalize_never_guesses():
    assert normalize_answer("The correct match is C", "match_letters") == ""
    assert normalize_answer("C", "match_letters") == "C"
    assert normalize_answer("b", "match_letters") == "B"
    assert normalize_answer("1 + 3 = 4", "text_to_num") == ""
    assert normalize_answer("4", "text_to_num") == "4"
    assert normalize_answer("1,111", "text_to_num") == "1111"


def test_choose_mode_routing():
    assert choose_mode(1, "translation") == "direct"
    assert choose_mode(5, "match_letters") == "per_item"
    assert choose_mode(5, "fill_blanks") == "per_item"
    assert choose_mode(2, "translation") == "batch"
    assert choose_mode(8, "translation") == "per_item"


def test_normalize_option_letters_rejects_invalid():
    assert normalize_option_letters(["A", "ZZ", "A"], ["A", "B", "C"]) == ["A", "", "A"]


def test_duplicate_letter_indices():
    assert duplicate_letter_indices(["A", "B", "A"]) == [0, 2]


def test_mirror_style_requires_strong_majority():
    weak = "[a]\nb\nc\nd"
    assert mirror_style(weak, "xyz", "fill_blanks") == "xyz"
    strong = "\n".join(f"[{c}]" for c in "abcdefgh") + "\nplain"
    assert mirror_style(strong, "xyz", "fill_blanks") == "[xyz]"


def test_apply_constraints_no_guess_fill():
    row = {
        "task_type": "match_letters",
        "context": "1. x\nA. one\nB. two\nC. three\n",
    }
    out = apply_constraints(["A", ""], row)
    assert out[0] == "A"
    assert out[1] == ""


def test_rescue_on_none_parse():
    calls = {"n": 0}

    def gen(bundle, messages, max_new_tokens=512):
        calls["n"] += 1
        system = next(m["content"] for m in messages if m["role"] == "system")
        user = next(m["content"] for m in messages if m["role"] == "user")
        if "mappings" in system:
            return '{"mappings":[],"rules":[],"constraints":[],"unresolved":[]}'
        if "Extract the single answer" in system or "Emit FINAL only" in user:
            return "FINAL: 42"
        if any(ln.startswith("Item ") for ln in user.splitlines()):
            return "The answer must be forty two somehow"
        return '["42"]'

    row = {"context": "c", "query": "1. x", "task_type": "text_to_num"}
    # N=1 uses direct oneshot; force multi-item per-item path
    row = {"context": "c", "query": "1. x\n2. y\n3. z\n4. w", "task_type": "fill_blanks"}
    result = solve_row(
        row,
        "analyze_constrained_v1",
        _FakeBundle(),
        generate_fn=gen,
        allow_rescue=True,
    )
    assert result["meta"]["mode"] == "per_item"
    assert result["meta"].get("rescued") is True
    assert all(a == "42" for a in result["pred"])
    assert "rescued" in result["meta"]["parse_status"]
