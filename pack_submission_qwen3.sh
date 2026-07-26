#!/usr/bin/env bash
# Pack offline Qwen3 HF submission: model/ + wheelhouse/ + script.py + solver/
# Usage: ./pack_submission_qwen3.sh [out_dir]

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${1:-./submit_build_qwen3_14b}"
HF_REPO="${IOL_HF_REPO:-jbuaba/iolai-2026-qwen3-14b}"
HUB_MODEL_ID="${IOL_HUB_MODEL:-Qwen/Qwen3-14B-AWQ}"

mkdir -p "${OUT_DIR}/model" "${OUT_DIR}/wheelhouse" "${OUT_DIR}/solver"

echo "==> weights ${HUB_MODEL_ID} -> ${OUT_DIR}/model"
if command -v hf >/dev/null 2>&1; then
  hf download "${HUB_MODEL_ID}" --local-dir "${OUT_DIR}/model"
else
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download('${HUB_MODEL_ID}', local_dir='${OUT_DIR}/model')"
fi

echo "==> wheelhouse"
WH="${OUT_DIR}/wheelhouse"
python3 -m pip download --no-deps -d "${WH}" \
  transformers==4.51.3 huggingface_hub==0.30.2 2>/dev/null || true
python3 -m pip download --no-deps --only-binary=:all: \
  --python-version 39 --platform manylinux2014_x86_64 --implementation cp --abi cp39 \
  -d "${WH}" tokenizers==0.21.1
# autoawq 0.2.9 wheel is py3-none-any; prefer copying a known-good wheel if present
if [[ ! -f "${WH}/autoawq-0.2.9-py3-none-any.whl" ]]; then
  if [[ -f "${ROOT}/submit_build_qwen3_14b/wheelhouse/autoawq-0.2.9-py3-none-any.whl" ]]; then
    cp "${ROOT}/submit_build_qwen3_14b/wheelhouse/autoawq-0.2.9-py3-none-any.whl" "${WH}/"
  else
    echo "ERROR: missing autoawq-0.2.9-py3-none-any.whl in ${WH}" >&2
    echo "Place the wheel there (PyPI may only offer an sdist)." >&2
    exit 1
  fi
fi
rm -f "${WH}"/*.tar.gz "${WH}"/*.metadata 2>/dev/null || true

echo "==> solver + script"
for name in __init__ model minimal items matching runtime; do
  cp "${ROOT}/solver/${name}.py" "${OUT_DIR}/solver/${name}.py"
done
cp "${ROOT}/script.py" "${OUT_DIR}/script.py"

python3 - "${OUT_DIR}/model" <<'PY'
import json, sys
from pathlib import Path
cfg_path = Path(sys.argv[1]) / "generation_config.json"
if cfg_path.exists():
    cfg = json.loads(cfg_path.read_text())
    cfg["repetition_penalty"] = 1.0
    cfg["do_sample"] = False
    for k in ("temperature", "top_p", "top_k", "typical_p"):
        cfg.pop(k, None)
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
PY

cat > "${OUT_DIR}/README.md" <<EOF
---
license: apache-2.0
base_model: ${HUB_MODEL_ID}
tags:
  - iol-ai-2026
  - qwen3
---

# IOL-AI 2026 — Qwen3-14B-AWQ

Our greedy + short-pad + match_letters assignment stack on Qwen3.
Offline \`wheelhouse/\` bootstraps transformers 4.51.3 so Qwen3 loads in the Space.
EOF

echo "packed ${OUT_DIR}"
echo "python tools/verify_pack.py ${OUT_DIR} --qwen3"
echo "hf upload ${HF_REPO} ${OUT_DIR} --repo-type model"
