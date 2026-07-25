from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solver.items import count_items
from solver.normalize import normalize_answer, normalize_answers
from solver.parse import parse_answers
from solver.prompts import build_messages
from solver.verify import verify_answers


def test_count_items_numbered_dot():
    query = "Translate into English:\n17. foo bar\n18. baz\n19. qux"
    assert count_items(query) == 3


def test_count_items_paren():
    assert count_items("Fill in:\n1) aaa\n2) bbb") == 2


def test_count_items_fallback_blank():
    assert count_items("Write the forms.\n\nalpha\nbeta\ngamma") == 3


def test_parse_json_array():
    assert parse_answers('Some noise\n["a", "b", "c"]\n', n=3) == ["a", "b", "c"]


def test_parse_json_object_answers():
    text = '{"rules": ["r1"], "tests": ["t1"], "answers": ["x", "y"]}'
    assert parse_answers(text, n=2) == ["x", "y"]


def test_parse_final_answers():
    text = "reasoning...\nFINAL ANSWERS:\n1. hello\n2. world"
    assert parse_answers(text, n=2) == ["hello", "world"]


def test_parse_pad_truncate():
    assert parse_answers('["only"]', n=3) == ["only", "", ""]
    assert parse_answers('["a","b","c","d"]', n=2) == ["a", "b"]


def test_parse_fenced_json():
    assert parse_answers('```json\n["A", "B"]\n```', n=2) == ["A", "B"]


def test_verify_match_letters():
    ok, _ = verify_answers(["A", "B"], 2, {"task_type": "match_letters"})
    assert ok
    ok, _ = verify_answers(["AB", "B"], 2, {"task_type": "match_letters"})
    assert not ok


def test_verify_text_to_num():
    ok, _ = verify_answers(["111", "42"], 2, {"task_type": "text_to_num"})
    assert ok
    ok, _ = verify_answers(["111", "forty"], 2, {"task_type": "text_to_num"})
    assert not ok


def test_verify_empty():
    ok, reasons = verify_answers(["a", ""], 2, {"task_type": "translation"})
    assert not ok
    assert any("empty" in reason for reason in reasons)


def test_normalize_match_letters():
    assert normalize_answer(" (b) ", "match_letters") == "B"


def test_normalize_text_to_num():
    assert normalize_answer("1,111", "text_to_num") == "1111"


def test_normalize_quotes():
    assert normalize_answer('"café"', "translation") == "café"


def test_normalize_answers_row():
    assert normalize_answers(["a", "b"], {"task_type": "match_letters"}) == ["A", "B"]


def test_build_messages_baseline():
    row = {"context": "CTX", "query": "1. x", "task_type": "translation"}
    messages = build_messages(row, 1, "baseline")
    assert "FINAL ANSWERS" in messages[0]["content"]
    assert "CTX" in messages[1]["content"]


def test_build_messages_task_json():
    row = {
        "context": "CTX",
        "query": "1. x\n2. y",
        "task_type": "text_to_num",
        "eval_type": "single",
        "work_lang": "eng_Latn",
        "task_lang": "xyz_Latn",
    }
    messages = build_messages(row, 2, "task_json_v1")
    assert "JSON array" in messages[0]["content"]
    assert "digits only" in messages[0]["content"]
    assert "N=2" in messages[1]["content"]


def test_build_messages_rescue():
    row = {"context": "CTX", "query": "1. x", "task_type": "translation"}
    messages = build_messages(row, 1, "structured_verify_v1", rescue=True)
    assert '"answers"' in messages[0]["content"]
