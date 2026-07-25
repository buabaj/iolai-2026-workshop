#!/usr/bin/env bash
# Build a public HF model repo for IOL-AI 2026.
# Usage:
#   ./pack_submission.sh [hub_model_id] [out_dir]
#   IOL_EXPLAIN=1 ./pack_submission.sh [hub_model_id] [out_dir]
#
# Default stays on 7B until the stratified-48 timing gate passes on 14B.

set -euo pipefail

HUB_MODEL_ID="${1:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
OUT_DIR="${2:-./submit_build}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
WRITE_EXPLANATIONS="${IOL_EXPLAIN:-0}"
if [[ "${HUB_MODEL_ID}" == *"14B"* ]]; then
  HF_REPO="${IOL_HF_REPO:-jbuaba/iolai-2026-qwen25-14b}"
else
  HF_REPO="${IOL_HF_REPO:-jbuaba/iolai-2026-qwen25-7b}"
fi

mkdir -p "${OUT_DIR}"

if command -v hf >/dev/null 2>&1; then
  hf download "${HUB_MODEL_ID}" --local-dir "${OUT_DIR}"
else
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download('${HUB_MODEL_ID}', local_dir='${OUT_DIR}')"
fi

rm -rf "${OUT_DIR}/solver"
mkdir -p "${OUT_DIR}/solver"
for name in __init__ items prompts parse verify normalize model pipeline constraints; do
  cp "${ROOT}/solver/${name}.py" "${OUT_DIR}/solver/${name}.py"
done
cp "${ROOT}/script.py" "${OUT_DIR}/script.py"

if [[ "${WRITE_EXPLANATIONS}" == "1" ]]; then
  python3 - "${OUT_DIR}/script.py" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
text = path.read_text()
old = 'os.environ.get("IOL_EXPLAIN", "0")'
new = 'os.environ.get("IOL_EXPLAIN", "1")'
if old not in text:
    raise SystemExit("could not set IOL_EXPLAIN default")
path.write_text(text.replace(old, new, 1))
PY
fi

cat > "${OUT_DIR}/README.md" <<EOF
---
license: apache-2.0
tags:
  - iol-ai-2026
---

# IOL-AI 2026 submission

- Weights: \`${HUB_MODEL_ID}\`
- Entrypoint: \`script.py\`
- Loads from \`.\`, reads \`/tmp/data/test.csv\`, writes \`submission.csv\`
- Strategy: \`analyze_constrained_v1\` (analysis → per-item → strict FINAL parse → rescue → conservative normalize)
- Ablations: \`structured_verify_v1\` (A), \`per_item_v1\` (C), \`analyze_constrained_v1\` (D)
- Invariant: parse may reject; normalize never guesses
- Explanations: \`IOL_EXPLAIN=1\` (analysis summary when available)
- Quant fallback: \`IOL_LOAD=bnb\`
EOF

echo "packed ${OUT_DIR}"
echo "upload with: hf upload ${HF_REPO} ${OUT_DIR} --repo-type model"
