#!/usr/bin/env bash
# Pack offline HF submission: weights + script.py + solver/
# Usage: ./pack_submission.sh [hub_model_id] [out_dir]

set -euo pipefail

HUB_MODEL_ID="${1:-Qwen/Qwen2.5-14B-Instruct-AWQ}"
ROOT="$(cd "$(dirname "$0")" && pwd)"

if [[ "${HUB_MODEL_ID}" == *"14B"* ]]; then
  OUT_DIR="${2:-./submit_build_14b}"
  HF_REPO="${IOL_HF_REPO:-jbuaba/iolai-2026-qwen25-14b}"
else
  OUT_DIR="${2:-./submit_build}"
  HF_REPO="${IOL_HF_REPO:-jbuaba/iolai-2026-qwen25-7b}"
fi

if [[ "${HUB_MODEL_ID}" == *"14B"* && "${OUT_DIR}" == "./submit_build" ]]; then
  echo "refuse: 14B must pack to ./submit_build_14b, not ./submit_build" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
if command -v hf >/dev/null 2>&1; then
  hf download "${HUB_MODEL_ID}" --local-dir "${OUT_DIR}"
else
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download('${HUB_MODEL_ID}', local_dir='${OUT_DIR}')"
fi

rm -rf "${OUT_DIR}/solver"
mkdir -p "${OUT_DIR}/solver"
for name in __init__ model minimal items matching normalize; do
  cp "${ROOT}/solver/${name}.py" "${OUT_DIR}/solver/${name}.py"
done
cp "${ROOT}/script.py" "${OUT_DIR}/script.py"

# Auto+jury ship path defaults explanations on in script.py already.
# IOL_EXPLAIN=0 can still force auto-only packs if needed.
python3 - "${OUT_DIR}" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
cfg_path = out / "generation_config.json"
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
tags:
  - iol-ai-2026
---

# IOL-AI 2026

- Weights: \`${HUB_MODEL_ID}\`
- \`script.py\` → \`/tmp/data/test.csv\` → \`submission.csv\`
- Minimal greedy decode with \`repetition_penalty=1.0\`
EOF

echo "packed ${OUT_DIR}"
echo "python tools/verify_pack.py ${OUT_DIR}"
echo "hf upload ${HF_REPO} ${OUT_DIR} --repo-type model"
