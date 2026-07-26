# IOL-AI 2026

Offline HF submission for [IOL-AI 2026](https://iolai.org).

`script.py` reads `/tmp/data/test.csv`, writes `submission.csv`.
Decode: greedy, `repetition_penalty=1.0`, `max_new_tokens=256`.
Post-repair: fit-to-N (never empty) + `match_letters` logit assignment.

## Layout

| Path | Role |
|---|---|
| `script.py` | Competition entry |
| `solver/model.py` | Load + generate |
| `solver/minimal.py` | Prompt + parse + repairs |
| `solver/items.py` | detect-N / fit-to-N |
| `solver/matching.py` | match_letters assignment |
| `evaluate.py` | Linguini proxy only (≠ Space) |
| `pack_submission.sh` | Build `./submit_build_14b` |
| `tools/verify_pack.py` | Pre-upload checks |

## Pack / upload

```bash
./pack_submission.sh
python tools/verify_pack.py ./submit_build_14b
hf upload jbuaba/iolai-2026-qwen25-14b ./submit_build_14b --repo-type model
```

## Gate before Space (Colab T4, `transformers==4.44.1`)

Compare against prior baseline (`score≈0.050`, `wrong_n=0.250`, `empty=0.083`):

```bash
python evaluate.py --n 16 --seed 0 --stratified \
  --model_id "Qwen/Qwen2.5-14B-Instruct-AWQ"
```

Ship only if `empty`/`wrong_n` drop, `score_proxy` does not regress, and `proj_90_items` ≪ 1800s.
