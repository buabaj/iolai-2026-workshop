from __future__ import annotations

import argparse
import ast
import json
import math
import time
from pathlib import Path

from solver import STRATEGIES
from solver.model import load_model
from solver.pipeline import RESCUE_RATE_CAP, solve_row

DEFAULT_EVAL_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
DEFAULT_N = 16
DEFAULT_SEED = 0
RESULTS_DIR = Path("results")


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

        chrf = CHRF()
    except ImportError:
        chrf = None

    gold = gold_alts(reference)
    preds = (pred + [""] * len(gold))[: len(gold)]
    em_hits = 0
    chrf_sum = 0.0
    items = []
    for index, (prediction, alts) in enumerate(zip(preds, gold), 1):
        hit = any(prediction.strip().lower() == alt.strip().lower() for alt in alts)
        if chrf is not None:
            score = max(chrf.sentence_score(prediction, [alt]).score for alt in alts)
        else:
            score = 100.0 if hit else 0.0
        em_hits += int(hit)
        chrf_sum += score
        items.append(
            {
                "item": index,
                "em": hit,
                "chrf": score,
                "pred": prediction,
                "gold": alts,
            }
        )
    n_items = max(1, len(gold))
    em = em_hits / n_items
    chrf_avg = chrf_sum / n_items
    return {
        "em": em,
        "chrf": chrf_avg,
        "score_proxy": math.sqrt(max(0.0, em) * max(0.0, chrf_avg / 100.0)),
        "n_items": n_items,
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate solver strategies on Linguini")
    parser.add_argument("--strategy", choices=STRATEGIES, default="baseline")
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model_id", default=DEFAULT_EVAL_MODEL)
    parser.add_argument("--load", default="awq", choices=["awq", "bnb"])
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"loading Linguini n={args.n} seed={args.seed}", flush=True)
    df = (
        load_dataset("facebook/linguini")["test"]
        .to_pandas()
        .sample(args.n, random_state=args.seed)
        .reset_index(drop=True)
    )
    print(f"{len(df)} problems | {list(df['task_type'].unique())}", flush=True)

    offline = args.model_id == "."
    load_started = time.monotonic()
    bundle = load_model(args.model_id, offline=offline, load_mode=args.load)
    print(f"model ready in {time.monotonic() - load_started:.1f}s", flush=True)

    per_problem = []
    em_total = 0.0
    chrf_total = 0.0
    item_count = 0
    rescue_used = 0
    rescue_budget = max(1, int(RESCUE_RATE_CAP * len(df)))
    started = time.monotonic()

    for index, row in df.iterrows():
        row_started = time.monotonic()
        result = solve_row(
            row,
            args.strategy,
            bundle,
            explain=args.explain,
            rescue_used=rescue_used,
            rescue_budget_rows=rescue_budget,
        )
        if result["meta"].get("rescued"):
            rescue_used += 1
        scored = score_problem(result["pred"], row["answer"])
        em_total += scored["em"] * scored["n_items"]
        chrf_total += scored["chrf"] * scored["n_items"]
        item_count += scored["n_items"]
        per_problem.append(
            {
                "idx": int(index),
                "id": str(row.get("id", index)),
                "task_type": str(row.get("task_type", "")),
                "em": scored["em"],
                "chrf": scored["chrf"],
                "score_proxy": scored["score_proxy"],
                "runtime_sec": time.monotonic() - row_started,
                "pred": result["pred"],
                "rescued": result["meta"].get("rescued", False),
                "verify_ok": result["meta"].get("verify_ok", False),
                "items": scored["items"],
            }
        )
        print(
            f"[{index + 1}/{len(df)}] {row.get('task_type')} "
            f"EM={scored['em']:.2f} chrF={scored['chrf']:.1f} "
            f"{time.monotonic() - row_started:.1f}s",
            flush=True,
        )

    runtime = time.monotonic() - started
    em = em_total / max(1, item_count)
    chrf_avg = chrf_total / max(1, item_count)
    score = math.sqrt(max(0.0, em) * max(0.0, chrf_avg / 100.0))
    summary = {
        "strategy": args.strategy,
        "model_id": args.model_id,
        "n": args.n,
        "seed": args.seed,
        "em": em,
        "chrf": chrf_avg,
        "score_proxy": score,
        "runtime_sec": runtime,
        "rescue_used": rescue_used,
        "per_problem": per_problem,
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / f"{args.strategy}_n{args.n}_seed{args.seed}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"FINAL strategy={args.strategy} EM={em:.3f} chrF={chrf_avg:.1f} "
        f"score≈{score:.3f} runtime={runtime:.0f}s rescued={rescue_used}",
        flush=True,
    )
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
