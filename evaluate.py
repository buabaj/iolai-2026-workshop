#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import time
from pathlib import Path

from solver.minimal import MAX_NEW_TOKENS, solve_row
from solver.model import load_model

DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"
PUBLIC_ROWS = 90


def gold_alts(reference) -> list[list[str]]:
    value = ast.literal_eval(str(reference))
    out: list[list[str]] = []
    for item in value:
        if isinstance(item, (list, tuple)):
            out.append([str(a) for a in item])
        else:
            out.append([str(item)])
    return out


def score_problem(pred: list[str], reference) -> dict:
    try:
        from sacrebleu.metrics import CHRF

        chrf_m = CHRF()
    except ImportError as exc:
        raise SystemExit("sacrebleu required for evaluate.py") from exc

    gold = gold_alts(reference)
    preds = (pred + [""] * len(gold))[: len(gold)]
    em_hits = 0
    chrf_sum = 0.0
    empty = 0
    for prediction, alts in zip(preds, gold):
        stripped = (prediction or "").strip()
        if not stripped:
            empty += 1
        hit = any(stripped.lower() == a.strip().lower() for a in alts)
        em_hits += int(hit)
        chrf_sum += max(chrf_m.sentence_score(prediction, [a]).score for a in alts)
    n = max(1, len(gold))
    em = em_hits / n
    chrf = chrf_sum / n
    return {
        "em": em,
        "chrf": chrf,
        "score": math.sqrt(max(0.0, em) * max(0.0, chrf / 100.0)),
        "n_items": n,
        "empty": empty,
        "len_error": abs(len(pred) - len(gold)),
    }


def stratified_sample(df, n: int, seed: int):
    import numpy as np
    import pandas as pd

    rng = np.random.RandomState(seed)
    groups = {k: g.reset_index(drop=True) for k, g in df.groupby("task_type", sort=True)}
    types = sorted(groups)
    base, rem = n // len(types), n % len(types)
    picks, leftover = [], []
    for i, task in enumerate(types):
        take = base + (1 if i < rem else 0)
        g = groups[task]
        if len(g) <= take:
            picks.append(g)
            continue
        idx = rng.choice(len(g), size=take, replace=False)
        picks.append(g.iloc[sorted(idx)])
        rest = sorted(set(range(len(g))) - set(idx.tolist()))
        leftover.append(g.iloc[rest])
    out = pd.concat(picks, ignore_index=True) if picks else df.head(0)
    if len(out) < n and leftover:
        need = n - len(out)
        pool = pd.concat(leftover, ignore_index=True)
        if len(pool) > need:
            idx = rng.choice(len(pool), size=need, replace=False)
            out = pd.concat([out, pool.iloc[sorted(idx)]], ignore_index=True)
        else:
            out = pd.concat([out, pool], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_id", default=DEFAULT_MODEL)
    parser.add_argument("--load", default="awq", choices=["awq", "bnb"])
    parser.add_argument("--stratified", action="store_true")
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    from datasets import load_dataset

    full = load_dataset("facebook/linguini")["test"].to_pandas()
    df = (
        stratified_sample(full, args.n, args.seed)
        if args.stratified
        else full.sample(min(args.n, len(full)), random_state=args.seed).reset_index(
            drop=True
        )
    )
    print(
        f"Linguini n={len(df)} seed={args.seed} stratified={args.stratified} "
        f"| {dict(df['task_type'].value_counts())}",
        flush=True,
    )

    bundle = load_model(args.model_id, offline=args.model_id == ".", load_mode=args.load)
    started = time.monotonic()
    em_num = chrf_num = empty = items = 0
    wrong_n = 0
    per_problem = []

    for index, row in df.iterrows():
        t0 = time.monotonic()
        pred, _ = solve_row(row, bundle, max_new_tokens=MAX_NEW_TOKENS)
        scored = score_problem(pred, row["answer"])
        em_num += scored["em"] * scored["n_items"]
        chrf_num += scored["chrf"] * scored["n_items"]
        empty += scored["empty"]
        items += scored["n_items"]
        wrong_n += int(scored["len_error"] > 0)
        per_problem.append(
            {
                "id": str(row.get("id", index)),
                "task_type": str(row.get("task_type", "")),
                **scored,
                "pred": pred,
                "runtime_sec": time.monotonic() - t0,
            }
        )
        print(
            f"[{index + 1}/{len(df)}] {row.get('task_type')} "
            f"EM={scored['em']:.2f} chrF={scored['chrf']:.1f} "
            f"empty={scored['empty']} len_err={scored['len_error']} "
            f"{time.monotonic() - t0:.1f}s",
            flush=True,
        )

    runtime = time.monotonic() - started
    em = em_num / max(1, items)
    chrf = chrf_num / max(1, items)
    score = math.sqrt(max(0.0, em) * max(0.0, chrf / 100.0))
    proj90 = (runtime / max(1, len(df))) * PUBLIC_ROWS
    summary = {
        "model_id": args.model_id,
        "n": args.n,
        "seed": args.seed,
        "stratified": args.stratified,
        "em": em,
        "chrf": chrf,
        "score_proxy": score,
        "empty_rate": empty / max(1, items),
        "wrong_item_count_rate": wrong_n / max(1, len(df)),
        "runtime_sec": runtime,
        "projected_90_sec": proj90,
        "per_problem": per_problem,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    tag = "strat" if args.stratified else "rand"
    out = args.results_dir / f"minimal_n{args.n}_{tag}_seed{args.seed}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(
        f"FINAL EM={em:.3f} chrF={chrf:.1f} score≈{score:.3f} "
        f"empty={empty / max(1, items):.3f} wrong_n={wrong_n / max(1, len(df)):.3f} "
        f"runtime={runtime:.0f}s proj90={proj90:.0f}s",
        flush=True,
    )
    print(f"wrote {out}", flush=True)
    print(
        "NOTE: Linguini proxy ≠ Space score. Gate on T4 with transformers==4.44.1.",
        flush=True,
    )


if __name__ == "__main__":
    main()
