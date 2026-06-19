"""
Run fixed-depth EAGLE baselines for DDD ablation.

This is E3.2 in experiment_plan_v2. It measures how changing the draft
expansion depth affects throughput and accepted tokens per EAGLE step before
enabling dynamic early stop.
"""

import argparse
import json
import os
import statistics
import sys
import time
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
    parser = argparse.ArgumentParser(description="Fixed-depth EAGLE sweep")
    parser.add_argument("--prompt-source", choices=["toy", "mt_bench"], default="mt_bench")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--question-begin", type=int, default=None)
    parser.add_argument("--question-end", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--depths", default="5,7,9,11")
    parser.add_argument("--total-token", type=int, default=60)
    parser.add_argument("--tree-top-k", type=int, default=10)
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

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False. Run on a GPU-visible node.")

    records = load_prompt_records(
        args.prompt_source,
        limit=args.limit,
        question_begin=args.question_begin,
        question_end=args.question_end,
    )
    if not records:
        raise RuntimeError("No prompt records loaded.")

    depths = parse_int_list(args.depths)
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = args.output_file or default_output_file(args, depths)
    output_path = os.path.join(args.output_dir, output_file)
    if os.path.exists(output_path) and not args.force:
        raise FileExistsError(f"{output_path} exists. Pass --force to overwrite.")

    print("=" * 76)
    print("E3.2 Fixed-depth EAGLE sweep")
    print(f"  prompt_source: {args.prompt_source}")
    print(f"  records:       {len(records)}")
    print(f"  max_new_tokens:{args.max_new_tokens}")
    print(f"  depths:        {depths}")
    print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unchanged>')}")
    print("=" * 76)

    results = []
    for depth in depths:
        results.append(run_depth(depth, records, args))

    payload = {
        "metadata": {
            "base_model_path": BASE_MODEL_PATH,
            "ea_model_path": EA_MODEL_PATH,
            "prompt_source": args.prompt_source,
            "num_records": len(records),
            "max_new_tokens": args.max_new_tokens,
            "depths": depths,
            "total_token": args.total_token,
            "tree_top_k": args.tree_top_k,
            "torch_dtype": args.torch_dtype,
            "device_map": args.device_map,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "use_system_prompt": not args.no_system_prompt,
        },
        "summary": summarize_depths(results),
        "depth_results": results,
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {output_path}")
    print_summary(payload["summary"])
    return payload


def run_depth(depth: int, records: List[Dict], args) -> Dict:
    import torch
    from eagle.model.ea_model import EaModel

    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }

    print(f"\n[depth={depth}] Loading model...")
    t_load = time.time()
    model = EaModel.from_pretrained(
        base_model_path=BASE_MODEL_PATH,
        ea_model_path=EA_MODEL_PATH,
        total_token=args.total_token,
        depth=depth,
        top_k=args.tree_top_k,
        torch_dtype=dtype_map[args.torch_dtype],
        low_cpu_mem_usage=True,
        device_map=args.device_map,
        use_eagle3=True,
        use_ddd=False,
        use_opt_tree=False,
    )
    model.eval()
    tokenizer = model.get_tokenizer()
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    system_prompt = None if args.no_system_prompt else DEFAULT_SYSTEM_PROMPT
    print(f"  Loaded in {time.time() - t_load:.1f}s")

    print("  Warmup...")
    run_warmup(model, tokenizer, records[0], args, system_prompt, stop_token_ids)

    print("  Running records...")
    record_results = []
    for idx, record in enumerate(records, start=1):
        result = run_record(model, tokenizer, record, args, system_prompt, stop_token_ids)
        record_results.append(result)
        print(f"    [{idx}/{len(records)}] {record['question_id']} turns={result['num_turns']}")

    summary = summarize_records(record_results)
    print(
        f"  depth={depth}: tok/s={summary['mean_tok_per_s_trimmed']:.2f}, "
        f"accept/step={summary['mean_accept_per_step_trimmed']:.3f}, "
        f"loops={summary['mean_loop_count']:.2f}"
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "depth": depth,
        "summary": summary,
        "records": record_results,
    }


def run_warmup(model, tokenizer, record: Dict, args, system_prompt, stop_token_ids) -> None:
    if args.warmup <= 0:
        return
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": record["turns"][0]})
    input_ids = build_chat_input(tokenizer, messages=messages)
    for _ in range(args.warmup):
        generate_eagle(model, input_ids, args, stop_token_ids)


def run_record(model, tokenizer, record: Dict, args, system_prompt, stop_token_ids) -> Dict:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    turns = []
    for turn_idx, user_text in enumerate(record["turns"]):
        messages.append({"role": "user", "content": user_text})
        input_ids, prompt_text = build_chat_input(tokenizer, messages=messages, return_prompt=True)
        result = generate_eagle(model, input_ids, args, stop_token_ids)
        assistant_text = tokenizer.decode(
            result["trimmed_token_ids"],
            spaces_between_special_tokens=False,
        ).strip()
        messages.append({"role": "assistant", "content": assistant_text})
        result.pop("trimmed_token_ids", None)
        turns.append(
            {
                "turn_index": turn_idx,
                "input_tokens": len(input_ids[0]),
                "prompt_chars": len(prompt_text),
                "user_prompt_preview": user_text[:240],
                **result,
            }
        )

    return {
        "question_id": record["question_id"],
        "source": record.get("source"),
        "category": record.get("category"),
        "num_turns": len(record["turns"]),
        "turns": turns,
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
    loop_count = int(loop_idx) + 1 if loop_idx is not None else None
    return {
        "raw_tokens": len(raw_ids),
        "trimmed_tokens": len(trimmed_ids),
        "trimmed_token_ids": trimmed_ids,
        "raw_exceeds_max_new_tokens": len(raw_ids) > args.max_new_tokens,
        "reported_new_token": int(reported_new_token),
        "loop_count": loop_count,
        "accept_per_step_trimmed": round(len(trimmed_ids) / loop_count, 6) if loop_count else 0.0,
        "wall_time_s": round(wall_time, 4),
        "tok_per_s_trimmed": round(len(trimmed_ids) / wall_time, 4) if wall_time > 0 else 0.0,
    }


def summarize_records(records: List[Dict]) -> Dict:
    turns = [turn for record in records for turn in record["turns"]]
    return {
        "turns": len(turns),
        "total_trimmed_tokens": sum(x["trimmed_tokens"] for x in turns),
        "total_raw_tokens": sum(x["raw_tokens"] for x in turns),
        "total_wall_time_s": round(sum(x["wall_time_s"] for x in turns), 4),
        "raw_overshoot_turns": sum(1 for x in turns if x["raw_exceeds_max_new_tokens"]),
        "mean_tok_per_s_trimmed": round(statistics.mean([x["tok_per_s_trimmed"] for x in turns]), 4),
        "mean_accept_per_step_trimmed": round(statistics.mean([x["accept_per_step_trimmed"] for x in turns]), 6),
        "mean_loop_count": round(statistics.mean([x["loop_count"] for x in turns]), 4),
        "mean_trimmed_tokens": round(statistics.mean([x["trimmed_tokens"] for x in turns]), 4),
    }


def summarize_depths(results: List[Dict]) -> List[Dict]:
    rows = []
    baseline_speed = results[0]["summary"]["mean_tok_per_s_trimmed"] if results else 0.0
    for item in results:
        summary = item["summary"]
        rows.append(
            {
                "depth": item["depth"],
                **summary,
                "speed_vs_first_depth": round(summary["mean_tok_per_s_trimmed"] / baseline_speed, 4)
                if baseline_speed > 0 else 0.0,
            }
        )
    return rows


def print_summary(summary: List[Dict]) -> None:
    print("\n" + "=" * 76)
    print("SUMMARY")
    for row in summary:
        print(
            f"  depth={row['depth']:>2}  tok/s={row['mean_tok_per_s_trimmed']:>7.2f}  "
            f"vs-first={row['speed_vs_first_depth']:>5.3f}  "
            f"accept/step={row['mean_accept_per_step_trimmed']:>5.3f}  "
            f"loops={row['mean_loop_count']:>6.2f}"
        )
    print("=" * 76)


def parse_int_list(text: str) -> List[int]:
    values = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if chunk:
            values.append(int(chunk))
    return values


def sync_cuda() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def default_output_file(args, depths: List[int]) -> str:
    limit = "all" if args.limit is None else str(args.limit)
    depth_text = "-".join(str(x) for x in depths)
    return (
        f"fixed_depth_{args.prompt_source}"
        f"_limit-{limit}_max-{args.max_new_tokens}"
        f"_depths-{depth_text}.json"
    )


if __name__ == "__main__":
    main()
