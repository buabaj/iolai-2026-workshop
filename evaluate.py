from __future__ import annotations

import argparse
import ast
import json
import math
import time
from collections import defaultdict
from pathlib import Path

from solver import STRATEGIES
from solver.model import load_model
from solver.pipeline import DEFAULT_STRATEGY, RESCUE_RATE_CAP, solve_row
from solver.verify import verify_answers

DEFAULT_EVAL_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
DEFAULT_N = 16
DEFAULT_SEED = 0
RESULTS_DIR = Path("results")
PUBLIC_ROWS = 90
TIME_BUDGET_SEC = 22 * 60


def gold_alts(reference) -> list[list[str]]:
    value = ast.literal_eval(str(reference))
    out: list[list[str]] = []
    for item in value:
        if isinstance(item, (list, tuple)):
            out.append([str(a) for a in item])
        else:
            out.append([str(item)])
    return out


def score_em(pred: list[str], gold: list[list[str]]) -> float:
    if not gold:
        return 0.0
    preds = (pred + [""] * len(gold))[: len(gold)]
    hits = 0
    for prediction, alts in zip(preds, gold):
        stripped = (prediction or "").strip()
        if any(stripped.lower() == alt.strip().lower() for alt in alts):
            hits += 1
    return hits / len(gold)


def score_problem(pred: list[str], reference, row=None, meta=None) -> dict:
    try:
        from sacrebleu.metrics import CHRF

        chrf = CHRF()
    except ImportError:
        chrf = None

    gold = gold_alts(reference)
    preds = (pred + [""] * len(gold))[: len(gold)]
    em_hits = 0
    chrf_sum = 0.0
    empty = 0
    nonempty_em = 0
    nonempty = 0
    items = []
    for index, (prediction, alts) in enumerate(zip(preds, gold), 1):
        stripped = (prediction or "").strip()
        if not stripped:
            empty += 1
        else:
            nonempty += 1
        hit = any(stripped.lower() == alt.strip().lower() for alt in alts)
        if stripped:
            nonempty_em += int(hit)
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
                "empty": not bool(stripped),
            }
        )
    n_items = max(1, len(gold))
    em = em_hits / n_items
    chrf_avg = chrf_sum / n_items
    invalid = 0
    if row is not None:
        ok, _ = verify_answers(preds, len(gold), row)
        invalid = 0 if ok else 1

    meta = meta or {}
    parsed = list(meta.get("parsed") or pred)
    normalized = list(meta.get("normalized") or pred)
    final = list(meta.get("final") or pred)
    return {
        "em": em,
        "chrf": chrf_avg,
        "score_proxy": math.sqrt(max(0.0, em) * max(0.0, chrf_avg / 100.0)),
        "n_items": n_items,
        "empty_items": empty,
        "em_given_nonempty": (nonempty_em / nonempty) if nonempty else 0.0,
        "nonempty_items": nonempty,
        "len_error": abs(len(pred) - len(gold)),
        "wrong_item_count": int(len(pred) != len(gold)),
        "invalid_format": invalid,
        "em_parsed": score_em(parsed, gold),
        "em_normalized": score_em(normalized, gold),
        "em_final": score_em(final, gold),
        "items": items,
    }


def stratified_sample(df, n: int, seed: int):
    import numpy as np
    import pandas as pd

    rng = np.random.RandomState(seed)
    groups = {k: g.reset_index(drop=True) for k, g in df.groupby("task_type", sort=True)}
    types = sorted(groups)
    if not types:
        return df.head(0)

    base = n // len(types)
    rem = n % len(types)
    picks = []
    leftover = []
    for i, task in enumerate(types):
        take = base + (1 if i < rem else 0)
        g = groups[task]
        if len(g) <= take:
            picks.append(g)
            continue
        idx = rng.choice(len(g), size=take, replace=False)
        picks.append(g.iloc[sorted(idx)])
        rest_idx = sorted(set(range(len(g))) - set(idx.tolist()))
        leftover.append(g.iloc[rest_idx])

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


def _by_task_metrics(per_problem: list[dict]) -> dict[str, dict]:
    buckets: dict[str, dict] = defaultdict(
        lambda: {
            "em_num": 0.0,
            "chrf_num": 0.0,
            "items": 0,
            "empty": 0,
            "nonempty_em": 0.0,
            "nonempty": 0,
            "len_err": 0.0,
            "invalid": 0,
            "wrong_count": 0,
            "rows": 0,
            "runtime": 0.0,
            "generations": 0,
        }
    )
    for row in per_problem:
        task = row.get("task_type") or "unknown"
        b = buckets[task]
        n_items = max(1, int(row.get("n_items", 1)))
        b["em_num"] += float(row["em"]) * n_items
        b["chrf_num"] += float(row["chrf"]) * n_items
        b["items"] += n_items
        b["empty"] += int(row.get("empty_items", 0))
        b["nonempty"] += int(row.get("nonempty_items", 0))
        b["nonempty_em"] += float(row.get("em_given_nonempty", 0)) * max(
            1, int(row.get("nonempty_items", 0))
        )
        b["len_err"] += float(row.get("len_error", 0))
        b["invalid"] += int(row.get("invalid_format", 0))
        b["wrong_count"] += int(row.get("wrong_item_count", 0))
        b["rows"] += 1
        b["runtime"] += float(row.get("runtime_sec", 0))
        b["generations"] += int(row.get("generations", 0))

    out = {}
    for task, b in buckets.items():
        items = max(1, b["items"])
        em = b["em_num"] / items
        chrf = b["chrf_num"] / items
        nonempty = max(1, b["nonempty"])
        out[task] = {
            "rows": b["rows"],
            "em": em,
            "chrf": chrf,
            "score_proxy": math.sqrt(max(0.0, em) * max(0.0, chrf / 100.0)),
            "empty_rate": b["empty"] / items,
            "em_given_nonempty": b["nonempty_em"] / nonempty,
            "invalid_format_rate": b["invalid"] / max(1, b["rows"]),
            "wrong_item_count_rate": b["wrong_count"] / max(1, b["rows"]),
            "mean_len_error": b["len_err"] / max(1, b["rows"]),
            "time_per_problem": b["runtime"] / max(1, b["rows"]),
            "time_per_generated_item": b["runtime"] / max(1, b["generations"] or b["items"]),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate solver strategies on Linguini")
    parser.add_argument("--strategy", choices=STRATEGIES, default=DEFAULT_STRATEGY)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model_id", default=DEFAULT_EVAL_MODEL)
    parser.add_argument("--load", default="awq", choices=["awq", "bnb"])
    parser.add_argument("--explain", action="store_true")
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="Sample evenly across task_type (recommended for gate runs)",
    )
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"loading Linguini n={args.n} seed={args.seed} stratified={args.stratified}", flush=True)
    full = load_dataset("facebook/linguini")["test"].to_pandas()
    if args.stratified:
        df = stratified_sample(full, args.n, args.seed)
    else:
        df = full.sample(min(args.n, len(full)), random_state=args.seed).reset_index(drop=True)
    print(f"{len(df)} problems | {dict(df['task_type'].value_counts())}", flush=True)

    offline = args.model_id == "."
    load_started = time.monotonic()
    bundle = load_model(args.model_id, offline=offline, load_mode=args.load)
    print(f"model ready in {time.monotonic() - load_started:.1f}s", flush=True)

    per_problem = []
    em_total = 0.0
    em_parsed_total = 0.0
    em_norm_total = 0.0
    em_final_total = 0.0
    chrf_total = 0.0
    item_count = 0
    empty_total = 0
    nonempty_em_total = 0.0
    nonempty_total = 0
    invalid_total = 0
    wrong_count_total = 0
    generations_total = 0
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
        scored = score_problem(result["pred"], row["answer"], row=row, meta=result["meta"])
        gens = int(result["meta"].get("generations", 0))
        generations_total += gens
        em_total += scored["em"] * scored["n_items"]
        em_parsed_total += scored["em_parsed"] * scored["n_items"]
        em_norm_total += scored["em_normalized"] * scored["n_items"]
        em_final_total += scored["em_final"] * scored["n_items"]
        chrf_total += scored["chrf"] * scored["n_items"]
        item_count += scored["n_items"]
        empty_total += scored["empty_items"]
        nonempty_total += scored["nonempty_items"]
        nonempty_em_total += scored["em_given_nonempty"] * max(1, scored["nonempty_items"])
        invalid_total += scored["invalid_format"]
        wrong_count_total += scored["wrong_item_count"]
        runtime = time.monotonic() - row_started
        per_problem.append(
            {
                "idx": int(index),
                "id": str(row.get("id", index)),
                "task_type": str(row.get("task_type", "")),
                "em": scored["em"],
                "em_parsed": scored["em_parsed"],
                "em_normalized": scored["em_normalized"],
                "em_final": scored["em_final"],
                "chrf": scored["chrf"],
                "score_proxy": scored["score_proxy"],
                "n_items": scored["n_items"],
                "empty_items": scored["empty_items"],
                "nonempty_items": scored["nonempty_items"],
                "em_given_nonempty": scored["em_given_nonempty"],
                "len_error": scored["len_error"],
                "wrong_item_count": scored["wrong_item_count"],
                "invalid_format": scored["invalid_format"],
                "runtime_sec": runtime,
                "generations": gens,
                "mode": result["meta"].get("mode"),
                "parse_status": result["meta"].get("parse_status"),
                "pred": result["pred"],
                "parsed": result["meta"].get("parsed"),
                "normalized": result["meta"].get("normalized"),
                "rescued": result["meta"].get("rescued", False),
                "verify_ok": result["meta"].get("verify_ok", False),
                "items": scored["items"],
            }
        )
        print(
            f"[{index + 1}/{len(df)}] {row.get('task_type')} mode={result['meta'].get('mode')} "
            f"EM={scored['em']:.2f} EMp={scored['em_parsed']:.2f} EMn={scored['em_normalized']:.2f} "
            f"chrF={scored['chrf']:.1f} empty={scored['empty_items']} gens={gens} {runtime:.1f}s",
            flush=True,
        )

    runtime = time.monotonic() - started
    em = em_total / max(1, item_count)
    em_parsed = em_parsed_total / max(1, item_count)
    em_normalized = em_norm_total / max(1, item_count)
    em_final = em_final_total / max(1, item_count)
    chrf_avg = chrf_total / max(1, item_count)
    score = math.sqrt(max(0.0, em) * max(0.0, chrf_avg / 100.0))
    empty_rate = empty_total / max(1, item_count)
    em_given_nonempty = nonempty_em_total / max(1, nonempty_total)
    projected_90 = (runtime / max(1, len(df))) * PUBLIC_ROWS
    by_task = _by_task_metrics(per_problem)
    wrong_rate = wrong_count_total / max(1, len(df))
    gate = {
        "score_ok": score >= 0.16,
        "empty_ok": empty_rate < 0.02,
        "wrong_count_ok": wrong_rate < 0.01,
        "normalize_ok": em_final + 1e-9 >= em_parsed,
        "timing_ok": projected_90 < TIME_BUDGET_SEC,
        "ship": (
            score >= 0.16
            and empty_rate < 0.02
            and wrong_rate < 0.01
            and em_final + 1e-9 >= em_parsed
            and projected_90 < TIME_BUDGET_SEC
        ),
    }
    summary = {
        "strategy": args.strategy,
        "model_id": args.model_id,
        "n": args.n,
        "seed": args.seed,
        "stratified": args.stratified,
        "em": em,
        "em_parsed": em_parsed,
        "em_normalized": em_normalized,
        "em_final": em_final,
        "chrf": chrf_avg,
        "score_proxy": score,
        "empty_rate": empty_rate,
        "em_given_nonempty": em_given_nonempty,
        "invalid_format_rate": invalid_total / max(1, len(df)),
        "wrong_item_count_rate": wrong_rate,
        "time_per_problem": runtime / max(1, len(df)),
        "time_per_generated_item": runtime / max(1, generations_total or item_count),
        "generations_total": generations_total,
        "runtime_sec": runtime,
        "projected_90_sec": projected_90,
        "rescue_used": rescue_used,
        "by_task_type": by_task,
        "gate": gate,
        "per_problem": per_problem,
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    tag = "strat" if args.stratified else "rand"
    out_path = args.results_dir / f"{args.strategy}_n{args.n}_{tag}_seed{args.seed}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"FINAL strategy={args.strategy} EM={em:.3f} EMp={em_parsed:.3f} EMn={em_normalized:.3f} "
        f"EMf={em_final:.3f} chrF={chrf_avg:.1f} score≈{score:.3f} empty={empty_rate:.3f} "
        f"runtime={runtime:.0f}s proj90={projected_90:.0f}s ship={gate['ship']}",
        flush=True,
    )
    for task, metrics in sorted(by_task.items()):
        print(
            f"  {task}: EM={metrics['em']:.3f} chrF={metrics['chrf']:.1f} "
            f"empty={metrics['empty_rate']:.3f} EM|ne={metrics['em_given_nonempty']:.3f} "
            f"rows={metrics['rows']}",
            flush=True,
        )
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
