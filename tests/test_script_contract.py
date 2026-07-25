from __future__ import annotations

import ast
from pathlib import Path


def test_script_contract():
    source = Path(__file__).resolve().parents[1].joinpath("script.py").read_text()
    tree = ast.parse(source)
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "main" in names
    assert 'os.environ["HF_HUB_OFFLINE"] = "1"' in source
    assert 'os.environ["TRANSFORMERS_OFFLINE"] = "1"' in source
    assert "/tmp/data/test.csv" in source
    assert "submission.csv" in source
    assert "trust_remote_code" not in source
