# IOL-AI 2026

Offline HF submission for [IOL-AI 2026](https://iolai.org).

`script.py` reads `/tmp/data/test.csv`, writes `submission.csv`.
Decode: greedy, `repetition_penalty=1.0`, `max_new_tokens=256`, line-split answers.

## Layout

| Path | Role |
|---|---|
| `script.py` | Competition entry |
| `solver/model.py` | Load + generate |
| `solver/minimal.py` | Prompt + parse |
| `evaluate.py` | Linguini proxy only (≠ Space) |
| `pack_submission.sh` | Build `./submit_build_14b` |
| `tools/verify_pack.py` | Pre-upload checks |

## Pack / upload

```bash
./pack_submission.sh
python tools/verify_pack.py ./submit_build_14b
hf upload jbuaba/iolai-2026-qwen25-14b ./submit_build_14b --repo-type model
```

## Local gate (Colab T4, `transformers==4.44.1`)

```bash
python evaluate.py --n 48 --seed 0 --stratified \
  --model_id "Qwen/Qwen2.5-14B-Instruct-AWQ"
```

Do not Space-submit until that gate looks healthy.
