#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED = [
    "script.py",
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "solver/__init__.py",
    "solver/model.py",
    "solver/minimal.py",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pack_dir", type=Path)
    args = parser.parse_args()
    root = args.pack_dir
    errors = [f"missing {r}" for r in REQUIRED if not (root / r).exists()]
    shards = sorted(root.glob("model-*.safetensors"))
    if len(shards) < 3:
        errors.append(f"expected >=3 shards, found {len(shards)}")
    cfg_path = root / "generation_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        rp = float(cfg.get("repetition_penalty", 0))
        if abs(rp - 1.0) > 1e-9:
            errors.append(f"repetition_penalty={rp} (want 1.0)")
        if cfg.get("do_sample", True) is not False:
            errors.append(f"do_sample={cfg.get('do_sample')} (want false)")
    model_py = (root / "solver" / "model.py").read_text()
    if "repetition_penalty" not in model_py or "1.0" not in model_py:
        errors.append("solver/model.py missing RP=1.0 override")
    if "tokenize=False" not in model_py or "max_length=6144" not in model_py:
        errors.append("solver/model.py missing winner encode path")
    minimal_py = (root / "solver" / "minimal.py").read_text()
    if "_CHATTY" not in minimal_py:
        errors.append("solver/minimal.py missing length-preserving hygiene")
    script_py = (root / "script.py").read_text()
    if "n_qmark" not in script_py:
        errors.append("script.py missing run summary counters")
    if errors:
        print("FAIL")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"OK {root} shards={len(shards)} rp=1.0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
