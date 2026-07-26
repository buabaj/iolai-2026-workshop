#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _check_generation_config(cfg_path: Path, errors: list[str]) -> None:
    if not cfg_path.exists():
        errors.append(f"missing {cfg_path.name}")
        return
    cfg = json.loads(cfg_path.read_text())
    rp = float(cfg.get("repetition_penalty", 0))
    if abs(rp - 1.0) > 1e-9:
        errors.append(f"repetition_penalty={rp} (want 1.0)")
    if cfg.get("do_sample", True) is not False:
        errors.append(f"do_sample={cfg.get('do_sample')} (want false)")


def verify_qwen25(root: Path) -> list[str]:
    """Final burn: 0.178 ATB + Lipas normalize + explain."""
    required = [
        "script.py",
        "solver/__init__.py",
        "solver/model.py",
        "solver/minimal.py",
        "solver/items.py",
        "solver/matching.py",
        "solver/normalize.py",
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
    ]
    errors = [f"missing {r}" for r in required if not (root / r).exists()]
    shards = sorted(root.glob("model-*.safetensors"))
    if len(shards) < 3:
        errors.append(f"expected >=3 shards, found {len(shards)}")
    _check_generation_config(root / "generation_config.json", errors)

    model_py = (root / "solver" / "model.py").read_text()
    if "repetition_penalty" not in model_py or "1.0" not in model_py:
        errors.append("solver/model.py missing RP=1.0")
    if "tokenize=False" not in model_py or "max_length=6144" not in model_py:
        errors.append("solver/model.py missing winner encode")

    minimal_py = (root / "solver" / "minimal.py").read_text()
    if "pad_short_answers" not in minimal_py or "extract_answer_lines" not in minimal_py:
        errors.append("solver/minimal.py missing pad/hygiene path")
    if "solve_matching" not in minimal_py:
        errors.append("solver/minimal.py missing match assignment")
    if "safe_normalize_answers" not in minimal_py:
        errors.append("solver/minimal.py missing safe_normalize_answers")
    if "extract_answer_lines_naive" in minimal_py or "IOL_NAIVE" in minimal_py:
        errors.append("solver/minimal.py must not ship naive experiment")

    normalize_py = (root / "solver" / "normalize.py").read_text()
    if "normalize_match_letter" not in normalize_py:
        errors.append("solver/normalize.py missing normalize_match_letter")
    if "normalize_text_to_num" not in normalize_py:
        errors.append("solver/normalize.py missing normalize_text_to_num")

    script_py = (root / "script.py").read_text()
    if 'IOL_EXPLAIN", "1"' not in script_py:
        errors.append("script.py must default IOL_EXPLAIN to 1")
    if "EXPLAIN_RESERVE_SEC" not in script_py or "explanation_rate" not in script_py:
        errors.append("script.py missing explain reserve / rate")
    if "ensure_runtime" in script_py:
        errors.append("2.5 pack must not use Qwen3 runtime bootstrap")
    if "naive_splitlines" in script_py:
        errors.append("script.py must not advertise naive push parse")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pack_dir", type=Path)
    parser.add_argument("--qwen3", action="store_true")
    args = parser.parse_args()
    root = args.pack_dir
    if args.qwen3 or (root / "wheelhouse").is_dir():
        print("FAIL")
        print("  - use 2.5 ATB pack for this burn, not qwen3")
        return 1
    errors = verify_qwen25(root)
    if errors:
        print("FAIL")
        for e in errors:
            print(f"  - {e}")
        return 1
    shards = len(list(root.glob("model-*.safetensors")))
    print(f"OK {root} atb+normalize+explain shards={shards} rp=1.0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
