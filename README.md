# IOL-AI 2026

Offline HF submission for [IOL-AI 2026](https://iolai.org).

`script.py` reads `/tmp/data/test.csv`, writes `submission.csv`.

- Greedy decode, `repetition_penalty=1.0`, `max_new_tokens=512`
- Winner encode: chat_template string → `tok(..., max_length=6144)`
- Length-preserving line hygiene (no fit-to-N)
- Auto track: `IOL_EXPLAIN=0` (explanations only after answers if enabled)

## Pack / upload

```bash
# sync code into existing pack (no re-download)
python tools/verify_pack.py ./submit_build_14b
hf upload jbuaba/iolai-2026-qwen25-14b ./submit_build_14b/script.py script.py --repo-type model
hf upload jbuaba/iolai-2026-qwen25-14b ./submit_build_14b/solver solver --repo-type model
```

## Gate before Space

```bash
python evaluate.py --n 16 --seed 0 --stratified \
  --model_id "Qwen/Qwen2.5-14B-Instruct-AWQ"
```
