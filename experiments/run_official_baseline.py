"""
Run an official-style EAGLE baseline comparison.

This script compares:
  - EaModel.naivegenerate(): vanilla AR through the EAGLE patched model
  - EaModel.eagenerate(): EAGLE-3 speculative decoding

Both methods use the same EaModel instance, chat template, dtype, stop tokens,
and max_new_tokens. This is the baseline口径 we should use before tuning DDD
or OPT-Tree.
"""

import argparse
import json
import os
import statistics
import sys
import time
from typing import Dict, List, Tuple


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
    parser = argparse.ArgumentParser(description="Official-style EAGLE baseline comparison")
    parser.add_argument("--prompt-source", choices=["toy", "mt_bench"], default="toy")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of prompt records")
    parser.add_argument("--question-begin", type=int, default=None)
    parser.add_argument("--question-end", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--sample-top-k", type=float, default=0.0)
    parser.add_argument("--total-token", type=int, default=60)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--tree-top-k", type=int, default=10)
    parser.add_argument("--torch-dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--cuda-visible-devices", default=None,
                        help="Example: '0' or '0,1,2,3,4'. If omitted, keep current environment.")
    parser.add_argument("--no-system-prompt", action="store_true")
    parser.add_argument("--output-dir", default=os.path.join(EXPERIMENTS_ROOT, "E1_official_baseline"))
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only validate prompt loading and chat-template tokenization")
    return parser.parse_args()


def main() -> Dict:
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    import torch
    from eagle.model.ea_model import EaModel

    dtype = torch.float16 if args.torch_dtype == "fp16" else torch.bfloat16
    records = load_prompt_records(
        args.prompt_source,
        limit=args.limit,
        question_begin=args.question_begin,
        question_end=args.question_end,
    )
    if not records:
        raise RuntimeError("No prompt records loaded.")

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = args.output_file or _default_output_file(args)
    output_path = os.path.join(args.output_dir, output_file)
    if os.path.exists(output_path) and not args.force:
        raise FileExistsError(f"{output_path} exists. Pass --force to overwrite.")

    print("=" * 70)
    print("Official-style EAGLE baseline")
    print(f"  prompt_source: {args.prompt_source}")
    print(f"  records:       {len(records)}")
    print(f"  max_new_tokens:{args.max_new_tokens}")
    print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unchanged>')}")
    print("=" * 70)

    if args.dry_run:
        return dry_run(records, args)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() is False. Run this script in the project "
            "environment on a GPU-visible node, for example with "
            "`.venv/bin/python experiments/run_official_baseline.py ...`."
        )

    print("\n[1/3] Loading EaModel...")
    t_load = time.time()
    model = EaModel.from_pretrained(
        base_model_path=BASE_MODEL_PATH,
        ea_model_path=EA_MODEL_PATH,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.tree_top_k,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=args.device_map,
        use_eagle3=True,
    )
    model.eval()
    tokenizer = model.get_tokenizer()
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    print(f"  Loaded in {time.time() - t_load:.1f}s")

    print("\n[2/3] Warmup...")
    system_prompt = None if args.no_system_prompt else DEFAULT_SYSTEM_PROMPT
    _run_warmup(model, tokenizer, records[0], args, system_prompt)
    print("  Warmup done")

    print("\n[3/3] Running comparison...")
    all_results = []
    for idx, record in enumerate(records, start=1):
        print(f"  [{idx}/{len(records)}] {record['question_id']}")
        all_results.append(
            run_record(model, tokenizer, record, args, system_prompt, stop_token_ids)
        )

    summary = summarize_results(all_results)
    payload = {
        "metadata": {
            "base_model_path": BASE_MODEL_PATH,
            "ea_model_path": EA_MODEL_PATH,
            "prompt_source": args.prompt_source,
            "num_records": len(records),
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "total_token": args.total_token,
            "depth": args.depth,
            "tree_top_k": args.tree_top_k,
            "torch_dtype": args.torch_dtype,
            "device_map": args.device_map,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "use_system_prompt": not args.no_system_prompt,
        },
        "summary": summary,
        "records": all_results,
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {output_path}")
    print_summary(summary)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def dry_run(records: List[Dict], args) -> Dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False)
    system_prompt = None if args.no_system_prompt else DEFAULT_SYSTEM_PROMPT
    previews = []

    print("\n[DRY RUN] Validating prompt records and chat template...")
    for record in records:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": record["turns"][0]})
        input_ids, prompt = build_chat_input(tokenizer, messages=messages, return_prompt=True)
        item = {
            "question_id": record["question_id"],
            "num_turns": len(record["turns"]),
            "input_tokens_first_turn": len(input_ids[0]),
            "prompt_preview": prompt[:300],
        }
        previews.append(item)
        print(
            f"  {record['question_id']}: turns={item['num_turns']}, "
            f"first_turn_tokens={item['input_tokens_first_turn']}"
        )

    return {
        "dry_run": True,
        "prompt_source": args.prompt_source,
        "num_records": len(records),
        "records": previews,
    }


def run_record(model, tokenizer, record: Dict, args, system_prompt, stop_token_ids) -> Dict:
    method_outputs = {}
    for method in ("naive", "eagle"):
        method_outputs[method] = run_conversation(
            model,
            tokenizer,
            turns=record["turns"],
            method=method,
            args=args,
            system_prompt=system_prompt,
            stop_token_ids=stop_token_ids,
        )

    turn_matches = []
    for naive_turn, eagle_turn in zip(method_outputs["naive"]["turns"], method_outputs["eagle"]["turns"]):
        turn_matches.append(naive_turn["trimmed_token_ids"] == eagle_turn["trimmed_token_ids"])

    return {
        "question_id": record["question_id"],
        "source": record.get("source"),
        "category": record.get("category"),
        "num_turns": len(record["turns"]),
        "all_turns_match": all(turn_matches),
        "turn_matches": turn_matches,
        "methods": method_outputs,
    }


def run_conversation(
    model,
    tokenizer,
    turns: List[str],
    method: str,
    args,
    system_prompt,
    stop_token_ids,
) -> Dict:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    outputs = []
    for turn_idx, user_text in enumerate(turns):
        messages.append({"role": "user", "content": user_text})
        input_ids, prompt = build_chat_input(tokenizer, messages=messages, return_prompt=True)
        result = generate_once(model, tokenizer, input_ids, method, args, stop_token_ids)
        assistant_text = tokenizer.decode(
            result["trimmed_token_ids"],
            spaces_between_special_tokens=False,
        ).strip()
        messages.append({"role": "assistant", "content": assistant_text})
        result.update(
            {
                "turn_index": turn_idx,
                "prompt_chars": len(prompt),
                "input_tokens": len(input_ids[0]),
                "user_prompt": user_text,
                "decoded_text_preview": assistant_text[:300],
            }
        )
        outputs.append(result)

    return {
        "turns": outputs,
        "total_trimmed_tokens": sum(x["trimmed_tokens"] for x in outputs),
        "total_raw_tokens": sum(x["raw_tokens"] for x in outputs),
        "total_wall_time_s": round(sum(x["wall_time_s"] for x in outputs), 4),
    }


def generate_once(model, tokenizer, input_ids, method: str, args, stop_token_ids) -> Dict:
    import torch

    input_tensor = torch.as_tensor(input_ids).cuda()
    generate_fn = model.naivegenerate if method == "naive" else model.eagenerate

    _sync_cuda()
    start = time.time()
    output_ids, reported_new_token, loop_idx = generate_fn(
        input_tensor.clone(),
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.sample_top_k,
        max_new_tokens=args.max_new_tokens,
        log=True,
        is_llama3=True,
    )
    _sync_cuda()
    wall = time.time() - start

    raw_ids = output_ids[0][len(input_ids[0]):].detach().cpu().tolist()
    trimmed_ids = trim_generated_ids(raw_ids, stop_token_ids, max_new_tokens=args.max_new_tokens)
    trimmed_count = len(trimmed_ids)
    raw_count = len(raw_ids)
    step_count = int(loop_idx) + 1 if loop_idx is not None else None

    return {
        "method": method,
        "raw_token_ids": raw_ids,
        "trimmed_token_ids": trimmed_ids,
        "raw_tokens": raw_count,
        "trimmed_tokens": trimmed_count,
        "reported_new_token": int(reported_new_token),
        "loop_count": step_count,
        "wall_time_s": round(wall, 4),
        "tok_per_s_raw": round(raw_count / wall, 4) if wall > 0 else 0.0,
        "tok_per_s_trimmed": round(trimmed_count / wall, 4) if wall > 0 else 0.0,
    }


def summarize_results(records: List[Dict]) -> Dict:
    method_names = ("naive", "eagle")
    method_summary = {}
    for method in method_names:
        turn_rows = [
            turn
            for record in records
            for turn in record["methods"][method]["turns"]
        ]
        method_summary[method] = {
            "turns": len(turn_rows),
            "total_trimmed_tokens": sum(x["trimmed_tokens"] for x in turn_rows),
            "total_raw_tokens": sum(x["raw_tokens"] for x in turn_rows),
            "total_wall_time_s": round(sum(x["wall_time_s"] for x in turn_rows), 4),
            "mean_tok_per_s_trimmed": round(statistics.mean([x["tok_per_s_trimmed"] for x in turn_rows]), 4),
            "mean_tok_per_s_raw": round(statistics.mean([x["tok_per_s_raw"] for x in turn_rows]), 4),
            "mean_loop_count": round(statistics.mean([x["loop_count"] for x in turn_rows]), 4),
        }

    naive_speed = method_summary["naive"]["mean_tok_per_s_trimmed"]
    eagle_speed = method_summary["eagle"]["mean_tok_per_s_trimmed"]
    return {
        "all_records_match": all(x["all_turns_match"] for x in records),
        "matching_records": sum(1 for x in records if x["all_turns_match"]),
        "total_records": len(records),
        "methods": method_summary,
        "speedup_eagle_vs_naive_trimmed_mean": round(eagle_speed / naive_speed, 4) if naive_speed > 0 else 0.0,
    }


def print_summary(summary: Dict) -> None:
    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  token match: {summary['matching_records']}/{summary['total_records']} records")
    for method, item in summary["methods"].items():
        print(
            f"  {method:<5}  {item['mean_tok_per_s_trimmed']:.2f} tok/s "
            f"(trimmed mean), total_time={item['total_wall_time_s']:.2f}s"
        )
    print(f"  speedup: {summary['speedup_eagle_vs_naive_trimmed_mean']:.3f}x")
    print("=" * 70)


def _run_warmup(model, tokenizer, record: Dict, args, system_prompt) -> None:
    if args.warmup <= 0:
        return
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": record["turns"][0]})
    input_ids = build_chat_input(tokenizer, messages=messages)
    for _ in range(args.warmup):
        for method in ("naive", "eagle"):
            generate_once(model, tokenizer, input_ids, method, args, stop_token_ids)


def _sync_cuda() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _default_output_file(args) -> str:
    limit = "all" if args.limit is None else str(args.limit)
    return (
        f"official_baseline_{args.prompt_source}"
        f"_limit-{limit}_max-{args.max_new_tokens}.json"
    )


if __name__ == "__main__":
    main()
