"""
Profile DDD beam confidence H = logsumexp(logprobsum) without early stopping.

This is E3.1 in experiment_plan_v2. It runs EAGLE with DDD enabled only to
force a deeper draft expansion and to record H at configured check steps.
The threshold is set very low by default, so no early stop should happen.
"""

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence


PROJECT_ROOT = "/home/hzliu/AD/Homework_haozhe/MLsys_final"
EAGLE_ROOT = os.path.join(PROJECT_ROOT, "EAGLE")
for _path in (PROJECT_ROOT, EAGLE_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from experiments.common import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    build_chat_input,
    load_prompt_records,
    trim_generated_ids,
)
from experiments.config import BASE_MODEL_PATH, EA_MODEL_PATH, EXPERIMENTS_ROOT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile DDD H distribution")
    parser.add_argument("--prompt-source", choices=["toy", "mt_bench"], default="toy")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--question-begin", type=int, default=None)
    parser.add_argument("--question-end", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--total-token", type=int, default=60)
    parser.add_argument("--base-depth", type=int, default=5)
    parser.add_argument("--tree-top-k", type=int, default=10)
    parser.add_argument("--ddd-max-depth", type=int, default=11)
    parser.add_argument("--ddd-check-steps", default="5,7,9")
    parser.add_argument("--ddd-threshold", type=float, default=-1e9,
                        help="Very low by default: profile H without early stopping")
    parser.add_argument("--torch-dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--no-system-prompt", action="store_true")
    parser.add_argument("--output-dir", default=os.path.join(EXPERIMENTS_ROOT, "E3_ddd"))
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> Dict:
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    import torch
    from eagle.model.ea_model import EaModel

    records = load_prompt_records(
        args.prompt_source,
        limit=args.limit,
        question_begin=args.question_begin,
        question_end=args.question_end,
    )
    if not records:
        raise RuntimeError("No prompt records loaded.")

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = args.output_file or default_output_file(args)
    output_path = os.path.join(args.output_dir, output_file)
    if os.path.exists(output_path) and not args.force:
        raise FileExistsError(f"{output_path} exists. Pass --force to overwrite.")

    check_steps = parse_check_steps(args.ddd_check_steps)
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }

    print("=" * 76)
    print("E3.1 DDD H distribution profiling")
    print(f"  prompt_source: {args.prompt_source}")
    print(f"  records:       {len(records)}")
    print(f"  max_new_tokens:{args.max_new_tokens}")
    print(f"  ddd_max_depth: {args.ddd_max_depth}")
    print(f"  check_steps:   {check_steps}")
    print(f"  threshold:     {args.ddd_threshold}")
    print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unchanged>')}")
    print("=" * 76)

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False. Run on a GPU-visible node.")

    print("\n[1/4] Loading EaModel with DDD profiling config...")
    t_load = time.time()
    model = EaModel.from_pretrained(
        base_model_path=BASE_MODEL_PATH,
        ea_model_path=EA_MODEL_PATH,
        total_token=args.total_token,
        depth=args.base_depth,
        top_k=args.tree_top_k,
        torch_dtype=dtype_map[args.torch_dtype],
        low_cpu_mem_usage=True,
        device_map=args.device_map,
        use_eagle3=True,
        use_ddd=True,
        ddd_max_depth=args.ddd_max_depth,
        ddd_check_steps=check_steps,
        ddd_threshold=args.ddd_threshold,
        use_opt_tree=False,
    )
    model.eval()
    tokenizer = model.get_tokenizer()
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    print(f"  Loaded in {time.time() - t_load:.1f}s")

    system_prompt = None if args.no_system_prompt else DEFAULT_SYSTEM_PROMPT

    print("\n[2/4] Warmup...")
    run_warmup(model, tokenizer, records[0], args, system_prompt, stop_token_ids)
    print("  Warmup done")

    print("\n[3/4] Profiling H...")
    profiled_records = []
    all_call_records = []
    for idx, record in enumerate(records, start=1):
        result = profile_record(model, tokenizer, record, args, system_prompt, stop_token_ids)
        profiled_records.append(result)
        all_call_records.extend(result["ddd_call_records"])
        print(
            f"  [{idx}/{len(records)}] {record['question_id']} "
            f"turns={result['num_turns']} calls={len(result['ddd_call_records'])}"
        )

    summary = summarize(all_call_records)
    payload = {
        "metadata": {
            "base_model_path": BASE_MODEL_PATH,
            "ea_model_path": EA_MODEL_PATH,
            "prompt_source": args.prompt_source,
            "num_records": len(records),
            "max_new_tokens": args.max_new_tokens,
            "total_token": args.total_token,
            "base_depth": args.base_depth,
            "tree_top_k": args.tree_top_k,
            "ddd_max_depth": args.ddd_max_depth,
            "ddd_check_steps": check_steps,
            "ddd_threshold": args.ddd_threshold,
            "torch_dtype": args.torch_dtype,
            "device_map": args.device_map,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "use_system_prompt": not args.no_system_prompt,
            "early_stop_should_be_disabled": args.ddd_threshold <= -1e8,
        },
        "summary": summary,
        "records": profiled_records,
    }

    print("\n[4/4] Saving results...")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"[SAVED] {output_path}")
    print_summary(summary)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def run_warmup(model, tokenizer, record: Dict, args, system_prompt, stop_token_ids) -> None:
    if args.warmup <= 0:
        return
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": record["turns"][0]})
    input_ids = build_chat_input(tokenizer, messages=messages)
    for _ in range(args.warmup):
        reset_ddd_stats(model)
        generate_eagle(model, input_ids, args, stop_token_ids)
    reset_ddd_stats(model)


def profile_record(model, tokenizer, record: Dict, args, system_prompt, stop_token_ids) -> Dict:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    turns = []
    record_call_rows = []
    for turn_idx, user_text in enumerate(record["turns"]):
        messages.append({"role": "user", "content": user_text})
        input_ids, prompt_text = build_chat_input(tokenizer, messages=messages, return_prompt=True)

        reset_ddd_stats(model)
        generated = generate_eagle(model, input_ids, args, stop_token_ids)
        stats = model.ea_layer._ddd_stats
        call_records = enrich_call_records(
            stats.get("call_records", []),
            question_id=record["question_id"],
            category=record.get("category"),
            turn_index=turn_idx,
        )
        record_call_rows.extend(call_records)

        assistant_text = tokenizer.decode(
            generated["trimmed_token_ids"],
            spaces_between_special_tokens=False,
        ).strip()
        messages.append({"role": "assistant", "content": assistant_text})

        turns.append(
            {
                "turn_index": turn_idx,
                "input_tokens": len(input_ids[0]),
                "prompt_chars": len(prompt_text),
                "generated_trimmed_tokens": generated["trimmed_tokens"],
                "generated_raw_tokens": generated["raw_tokens"],
                "ddd_calls": len(call_records),
                "wall_time_s": generated["wall_time_s"],
            }
        )

    return {
        "question_id": record["question_id"],
        "source": record.get("source"),
        "category": record.get("category"),
        "num_turns": len(record["turns"]),
        "turns": turns,
        "ddd_call_records": record_call_rows,
    }


def generate_eagle(model, input_ids, args, stop_token_ids: Sequence[Optional[int]]) -> Dict:
    import torch

    input_tensor = torch.as_tensor(input_ids).cuda()
    sync_cuda()
    start = time.time()
    output_ids, reported_new_token, loop_idx = model.eagenerate(
        input_tensor.clone(),
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
        max_new_tokens=args.max_new_tokens,
        log=True,
        is_llama3=True,
    )
    sync_cuda()
    wall_time = time.time() - start

    raw_ids = output_ids[0][len(input_ids[0]):].detach().cpu().tolist()
    trimmed_ids = trim_generated_ids(raw_ids, stop_token_ids, max_new_tokens=args.max_new_tokens)
    return {
        "raw_tokens": len(raw_ids),
        "trimmed_tokens": len(trimmed_ids),
        "trimmed_token_ids": trimmed_ids,
        "reported_new_token": int(reported_new_token),
        "loop_count": int(loop_idx) + 1 if loop_idx is not None else None,
        "wall_time_s": round(wall_time, 4),
    }


def reset_ddd_stats(model) -> None:
    model.ea_layer._ddd_stats = {
        "calls": 0,
        "early_stops": 0,
        "total_checks": 0,
        "depths": [],
        "actual_depths": [],
        "early_stop_steps": [],
        "checked_H": [],
        "call_records": [],
    }


def enrich_call_records(call_records: List[Dict], question_id, category, turn_index: int) -> List[Dict]:
    enriched = []
    for call_index, item in enumerate(call_records):
        row = {
            "question_id": question_id,
            "category": category,
            "turn_index": turn_index,
            "call_index": call_index,
            "actual_depth": item.get("actual_depth"),
            "early_stopped": item.get("early_stopped"),
            "early_stop_step": item.get("early_stop_step"),
            "checks": item.get("checks", []),
        }
        enriched.append(row)
    return enriched


def summarize(call_records: List[Dict]) -> Dict:
    all_checks = []
    h_by_step = defaultdict(list)
    for call in call_records:
        for check in call.get("checks", []):
            row = {
                "step": int(check["step"]),
                "H": float(check["H"]),
                "question_id": call.get("question_id"),
                "turn_index": call.get("turn_index"),
                "call_index": call.get("call_index"),
            }
            all_checks.append(row)
            h_by_step[row["step"]].append(row["H"])

    actual_depths = [int(x["actual_depth"]) for x in call_records if x.get("actual_depth") is not None]
    summary = {
        "num_calls": len(call_records),
        "num_checks": len(all_checks),
        "early_stops": sum(1 for x in call_records if x.get("early_stopped")),
        "actual_depth": describe_values(actual_depths),
        "H_global": describe_values([x["H"] for x in all_checks]),
        "H_by_step": {
            str(step): describe_values(values)
            for step, values in sorted(h_by_step.items())
        },
        "suggested_threshold_grid": suggested_thresholds([x["H"] for x in all_checks]),
    }
    return summary


def describe_values(values: Sequence[float]) -> Dict:
    values = [float(x) for x in values]
    if not values:
        return {"count": 0}
    sorted_values = sorted(values)
    result = {
        "count": len(sorted_values),
        "min": round(sorted_values[0], 6),
        "max": round(sorted_values[-1], 6),
        "mean": round(statistics.mean(sorted_values), 6),
        "p05": round(percentile(sorted_values, 0.05), 6),
        "p10": round(percentile(sorted_values, 0.10), 6),
        "p25": round(percentile(sorted_values, 0.25), 6),
        "p50": round(percentile(sorted_values, 0.50), 6),
        "p75": round(percentile(sorted_values, 0.75), 6),
        "p90": round(percentile(sorted_values, 0.90), 6),
        "p95": round(percentile(sorted_values, 0.95), 6),
    }
    if len(sorted_values) > 1:
        result["std"] = round(statistics.pstdev(sorted_values), 6)
    else:
        result["std"] = 0.0
    return result


def suggested_thresholds(values: Sequence[float]) -> List[float]:
    values = sorted(float(x) for x in values)
    if not values:
        return []
    qs = [0.05, 0.10, 0.25, 0.50, 0.75]
    return sorted({round(percentile(values, q), 3) for q in qs})


def percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("percentile() needs at least one value")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def print_summary(summary: Dict) -> None:
    print("\n" + "=" * 76)
    print("SUMMARY")
    print(f"  calls:        {summary['num_calls']}")
    print(f"  checks:       {summary['num_checks']}")
    print(f"  early_stops:  {summary['early_stops']}")
    print(f"  actual_depth: {compact_desc(summary['actual_depth'])}")
    print(f"  H global:     {compact_desc(summary['H_global'])}")
    print("  H by step:")
    for step, desc in summary["H_by_step"].items():
        print(f"    step {step}: {compact_desc(desc)}")
    print(f"  suggested threshold grid: {summary['suggested_threshold_grid']}")
    print("=" * 76)


def compact_desc(desc: Dict) -> str:
    if not desc or desc.get("count", 0) == 0:
        return "count=0"
    return (
        f"count={desc['count']}, mean={desc['mean']}, "
        f"p10={desc['p10']}, p50={desc['p50']}, p90={desc['p90']}"
    )


def parse_check_steps(text: str) -> List[int]:
    steps = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        steps.append(int(chunk))
    return sorted(set(steps))


def sync_cuda() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def default_output_file(args) -> str:
    limit = "all" if args.limit is None else str(args.limit)
    steps = args.ddd_check_steps.replace(",", "-")
    return (
        f"ddd_h_{args.prompt_source}"
        f"_limit-{limit}_max-{args.max_new_tokens}"
        f"_depth-{args.ddd_max_depth}_steps-{steps}.json"
    )


if __name__ == "__main__":
    main()
