# IOL-AI 2026 solver

Competitive offline solver for [IOL-AI 2026](https://iolai.org).

## Invariant

> Parsing may reject uncertain output. Normalization may standardize **valid** output. Neither may guess.

## Layout

| Path | Purpose |
|---|---|
| `script.py` | Competition entrypoint |
| `solver/` | Items, adaptive solve, strict parse, conservative normalize |
| `evaluate.py` | Stratified Linguini harness; EM(parsed/normalized/final) |
| `pack_submission.sh` | Snapshot AWQ weights + code (default **7B**) |

## Default path (D)

`analyze_constrained_v1`: shared analysis → per-item / batch → **strict** `FINAL:` parse → rescue on reject → conservative normalize.

Rollback: `structured_verify_v1` (A).

## Local ablation (7B only — no 14B until D ≥ A)

```bash
# A — one-shot rollback (local regression only; do not burn Space to re-prove 0.08)
python evaluate.py --strategy structured_verify_v1 --n 48 --seed 0 --stratified \
  --model_id "Qwen/Qwen2.5-7B-Instruct-AWQ"

# C — per-item + strict parse, no rescue
python evaluate.py --strategy per_item_v1 --n 48 --seed 0 --stratified \
  --model_id "Qwen/Qwen2.5-7B-Instruct-AWQ"

# D — analysis + per-item + strict parse + rescue + constraints
python evaluate.py --strategy analyze_constrained_v1 --n 48 --seed 0 --stratified \
  --model_id "Qwen/Qwen2.5-7B-Instruct-AWQ"
```

Ship D only if gate reports `ship=True`: empty_rate &lt; 2%, wrong_item_count_rate ≈ 0, `EM(final) ≥ EM(parsed)`, projected 90-row time &lt; 22 min.

Compare `EMp` / `EMn` / `EMf` in the FINAL line — if normalize destroys EM, rip constraints/normalize further.

## Package

```bash
./pack_submission.sh
hf upload jbuaba/iolai-2026-qwen25-7b ./submit_build --repo-type model
```

Do not re-host Linguini problems as plaintext.
