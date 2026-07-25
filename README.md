# IOL-AI 2026 solver

Competitive offline solver for [IOL-AI 2026](https://iolai.org).

## Layout

| Path | Purpose |
|---|---|
| `script.py` | Competition entrypoint |
| `solver/` | Prompting, parse, verify, normalize, inference |
| `evaluate.py` | Local Linguini harness (Hub access; not used in sandbox) |
| `pack_submission.sh` | Snapshot AWQ weights + code for a public HF model repo |
| `linguini_eval_T4.ipynb` | Workshop Colab template |

## Sandbox contract

- Public HF model repo; weights + code in the same repo
- `script.py` loads from `.`, reads `/tmp/data/test.csv`, writes `submission.csv`
- Offline T4, 30 minutes, libraries pinned by the Space (`torch` 2.4, `transformers` 4.44.1, `autoawq`, `bitsandbytes`, …)
- `pred` is a JSON list aligned to numbered query items

## Local / Colab checks

```bash
pip install -U transformers accelerate datasets sacrebleu gptqmodel bitsandbytes pytest
python -m pytest tests/ -q
python evaluate.py --strategy structured_verify_v1 --n 16 --seed 0 \
  --model_id "Qwen/Qwen2.5-7B-Instruct-AWQ"
```

If AWQ import fails on Colab, either `pip install -U gptqmodel` or:

```bash
python evaluate.py --strategy structured_verify_v1 --n 16 --seed 0 \
  --model_id "Qwen/Qwen2.5-7B-Instruct" --load bnb
```

## Package and submit

```bash
./pack_submission.sh Qwen/Qwen2.5-7B-Instruct-AWQ ./submit_build
hf upload YOURUSER/iolai-2026-qwen25-7b ./submit_build --repo-type model
```

Human track:

```bash
IOL_EXPLAIN=1 ./pack_submission.sh Qwen/Qwen2.5-7B-Instruct-AWQ ./submit_build_explain
```

Do not re-host Linguini problems as plaintext.
