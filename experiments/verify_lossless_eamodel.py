"""
Token-level greedy consistency check for EaModel.naivegenerate vs eagenerate.

This is the E1.4 correctness gate. It intentionally uses the same EaModel,
chat template, dtype, stop-token trimming, and input ids for both methods.
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
    parser = argparse.ArgumentParser(description="Verify greedy token-level EaModel consistency")
    parser.add_argument("--prompt-source", choices=["toy", "mt_bench"], default="toy")
    parser.add_argument("--limit", type=int, default=None)
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
    parser.add_argument("--use-ddd", action="store_true")
    parser.add_argument("--ddd-max-depth", type=int, default=11)
    parser.add_argument("--ddd-check-steps", default="5,7,9")
    parser.add_argument("--ddd-threshold", type=float, default=-2.0)
    parser.add_argument("--torch-dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--disable-tf32", action="store_true",
                        help="Disable TF32 matmul/cudnn paths for numerical diagnosis")
    parser.add_argument("--deterministic", action="store_true",
                        help="Ask PyTorch to use deterministic algorithms where available")
    parser.add_argument("--no-system-prompt", action="store_true")
    parser.add_argument("--history-policy", choices=["naive", "eagle"], default="naive",
                        help="Assistant history to reuse after each turn so both methods keep identical inputs")
    parser.add_argument("--output-dir", default=os.path.join(EXPERIMENTS_ROOT, "E1_lossless"))
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--omit-token-ids", action="store_true",
                        help="Do not save full generated token ids in the JSON output")
    parser.add_argument("--no-diagnose-first-diff", action="store_true",
                        help="Skip an extra base-model forward at mismatch positions")
    return parser.parse_args()


def main() -> Dict:
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch
    from eagle.model.ea_model import EaModel

    if args.disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

    if args.temperature > 1e-5:
        print("[WARN] temperature > 0: token-exact greedy matching is not expected.")

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

    print("=" * 76)
    print("E1.4 EaModel greedy lossless check")
    print(f"  prompt_source: {args.prompt_source}")
    print(f"  records:       {len(records)}")
    print(f"  max_new_tokens:{args.max_new_tokens}")
    print(f"  history_policy:{args.history_policy}")
    print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unchanged>')}")
    print("=" * 76)

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False. Run on a GPU-visible node.")

    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.torch_dtype]
    ddd_check_steps = parse_int_list(args.ddd_check_steps)
    print("\n[1/4] Loading EaModel...")
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
        use_ddd=args.use_ddd,
        ddd_max_depth=args.ddd_max_depth,
        ddd_check_steps=ddd_check_steps,
        ddd_threshold=args.ddd_threshold,
    )
    model.eval()
    tokenizer = model.get_tokenizer()
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    print(f"  Loaded in {time.time() - t_load:.1f}s")

    system_prompt = None if args.no_system_prompt else DEFAULT_SYSTEM_PROMPT
    print("\n[2/4] Warmup...")
    run_warmup(model, tokenizer, records[0], args, system_prompt, stop_token_ids)
    print("  Warmup done")

    print("\n[3/4] Running token-level comparison...")
    result_records = []
    for idx, record in enumerate(records, start=1):
        result = run_record(model, tokenizer, record, args, system_prompt, stop_token_ids)
        result_records.append(result)
        status = "OK" if result["all_turns_match"] else "MISMATCH"
        print(f"  [{idx}/{len(records)}] {record['question_id']} {status}")

    summary = summarize(result_records, args.max_new_tokens)
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
            "use_ddd": args.use_ddd,
            "ddd_max_depth": args.ddd_max_depth if args.use_ddd else None,
            "ddd_check_steps": ddd_check_steps if args.use_ddd else None,
            "ddd_threshold": args.ddd_threshold if args.use_ddd else None,
            "torch_dtype": args.torch_dtype,
            "disable_tf32": args.disable_tf32,
            "deterministic": args.deterministic,
            "device_map": args.device_map,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "use_system_prompt": not args.no_system_prompt,
            "history_policy": args.history_policy,
            "same_input_for_methods": True,
            "diagnose_first_diff": not args.no_diagnose_first_diff,
        },
        "summary": summary,
        "records": result_records,
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
        for method in ("naive", "eagle"):
            generate_once(model, input_ids, method, args, stop_token_ids)


def run_record(model, tokenizer, record: Dict, args, system_prompt, stop_token_ids) -> Dict:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    turns = []
    for turn_idx, user_text in enumerate(record["turns"]):
        messages.append({"role": "user", "content": user_text})
        input_ids, prompt_text = build_chat_input(tokenizer, messages=messages, return_prompt=True)
        naive = generate_once(model, input_ids, "naive", args, stop_token_ids)
        eagle = generate_once(model, input_ids, "eagle", args, stop_token_ids)
        comparison = compare_outputs(
            tokenizer=tokenizer,
            model=model,
            input_ids=input_ids[0],
            naive_ids=naive["trimmed_token_ids"],
            eagle_ids=eagle["trimmed_token_ids"],
            diagnose=not args.no_diagnose_first_diff,
        )

        assistant_ids = naive["trimmed_token_ids"] if args.history_policy == "naive" else eagle["trimmed_token_ids"]
        assistant_text = tokenizer.decode(assistant_ids, spaces_between_special_tokens=False).strip()

        if args.omit_token_ids:
            naive.pop("raw_token_ids", None)
            naive.pop("trimmed_token_ids", None)
            eagle.pop("raw_token_ids", None)
            eagle.pop("trimmed_token_ids", None)

        messages.append({"role": "assistant", "content": assistant_text})

        turns.append(
            {
                "turn_index": turn_idx,
                "same_input_for_methods": True,
                "input_tokens": len(input_ids[0]),
                "prompt_chars": len(prompt_text),
                "user_prompt_preview": user_text[:300],
                "match": comparison["match"],
                "comparison": comparison,
                "methods": {
                    "naive": naive,
                    "eagle": eagle,
                },
            }
        )

    return {
        "question_id": record["question_id"],
        "source": record.get("source"),
        "category": record.get("category"),
        "num_turns": len(record["turns"]),
        "all_turns_match": all(turn["match"] for turn in turns),
        "turns": turns,
    }


def generate_once(model, input_ids, method: str, args, stop_token_ids: Sequence[Optional[int]]) -> Dict:
    import torch

    input_tensor = torch.as_tensor(input_ids).cuda()
    generate_fn = model.naivegenerate if method == "naive" else model.eagenerate

    sync_cuda()
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
    sync_cuda()
    wall_time = time.time() - start

    raw_ids = output_ids[0][len(input_ids[0]):].detach().cpu().tolist()
    trimmed_ids = trim_generated_ids(raw_ids, stop_token_ids, max_new_tokens=args.max_new_tokens)
    return {
        "method": method,
        "raw_token_ids": raw_ids,
        "trimmed_token_ids": trimmed_ids,
        "raw_tokens": len(raw_ids),
        "trimmed_tokens": len(trimmed_ids),
        "raw_exceeds_max_new_tokens": len(raw_ids) > args.max_new_tokens,
        "reported_new_token": int(reported_new_token),
        "loop_count": int(loop_idx) + 1 if loop_idx is not None else None,
        "wall_time_s": round(wall_time, 4),
        "tok_per_s_trimmed": round(len(trimmed_ids) / wall_time, 4) if wall_time > 0 else 0.0,
    }


def compare_outputs(tokenizer, model, input_ids: List[int], naive_ids: List[int], eagle_ids: List[int],
                    diagnose: bool) -> Dict:
    match = naive_ids == eagle_ids
    first_diff = first_difference(naive_ids, eagle_ids)
    result = {
        "match": match,
        "same_length": len(naive_ids) == len(eagle_ids),
        "naive_trimmed_tokens": len(naive_ids),
        "eagle_trimmed_tokens": len(eagle_ids),
        "first_diff_pos": first_diff,
    }
    if match:
        return result

    naive_token = token_at(naive_ids, first_diff)
    eagle_token = token_at(eagle_ids, first_diff)
    result.update(
        {
            "naive_token_at_diff": naive_token,
            "eagle_token_at_diff": eagle_token,
            "naive_token_text": decode_token(tokenizer, naive_token),
            "eagle_token_text": decode_token(tokenizer, eagle_token),
            "common_prefix_text": tokenizer.decode(naive_ids[:first_diff], spaces_between_special_tokens=False)[-240:],
            "naive_window_text": decode_window(tokenizer, naive_ids, first_diff),
            "eagle_window_text": decode_window(tokenizer, eagle_ids, first_diff),
        }
    )
    if diagnose and naive_token is not None and eagle_token is not None:
        result["base_model_next_token_at_diff"] = diagnose_first_diff(
            model=model,
            tokenizer=tokenizer,
            prefix_ids=input_ids + naive_ids[:first_diff],
            naive_token=naive_token,
            eagle_token=eagle_token,
        )
    return result


def diagnose_first_diff(model, tokenizer, prefix_ids: List[int], naive_token: int, eagle_token: int) -> Dict:
    import torch

    if hasattr(model.base_model, "model") and hasattr(model.base_model.model, "tree_mask"):
        model.base_model.model.tree_mask = None

    with torch.no_grad():
        tensor = torch.as_tensor([prefix_ids]).cuda()
        outputs = model.base_model(tensor, use_cache=False)
        logits = outputs.logits[:, -1, :]
        top_values, top_indices = torch.topk(logits, k=5, dim=-1)

    top_tokens = []
    for token_id, logit in zip(top_indices[0].detach().cpu().tolist(), top_values[0].detach().cpu().tolist()):
        top_tokens.append(
            {
                "token_id": int(token_id),
                "text": decode_token(tokenizer, int(token_id)),
                "logit": float(logit),
                "is_naive_token": int(token_id) == int(naive_token),
                "is_eagle_token": int(token_id) == int(eagle_token),
            }
        )

    return {
        "argmax_token_id": int(top_indices[0, 0].item()),
        "argmax_text": decode_token(tokenizer, int(top_indices[0, 0].item())),
        "naive_token_is_argmax": int(naive_token) == int(top_indices[0, 0].item()),
        "eagle_token_is_argmax": int(eagle_token) == int(top_indices[0, 0].item()),
        "top5": top_tokens,
    }


def first_difference(left: List[int], right: List[int]) -> Optional[int]:
    for idx, (left_token, right_token) in enumerate(zip(left, right)):
        if int(left_token) != int(right_token):
            return idx
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def parse_int_list(text: str) -> List[int]:
    values = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if chunk:
            values.append(int(chunk))
    return values


def token_at(tokens: List[int], pos: Optional[int]) -> Optional[int]:
    if pos is None or pos >= len(tokens):
        return None
    return int(tokens[pos])


def decode_token(tokenizer, token_id: Optional[int]) -> Optional[str]:
    if token_id is None:
        return None
    return tokenizer.decode([int(token_id)], spaces_between_special_tokens=False)


def decode_window(tokenizer, token_ids: List[int], pos: Optional[int], before: int = 12, after: int = 40) -> str:
    if pos is None:
        return ""
    start = max(0, pos - before)
    end = min(len(token_ids), pos + after)
    return tokenizer.decode(token_ids[start:end], spaces_between_special_tokens=False)


def summarize(records: List[Dict], max_new_tokens: int) -> Dict:
    turns = [turn for record in records for turn in record["turns"]]
    mismatches = [turn for turn in turns if not turn["match"]]
    method_names = ("naive", "eagle")
    method_summary = {}
    for method in method_names:
        rows = [turn["methods"][method] for turn in turns]
        method_summary[method] = {
            "total_trimmed_tokens": sum(row["trimmed_tokens"] for row in rows),
            "total_raw_tokens": sum(row["raw_tokens"] for row in rows),
            "raw_overshoot_turns": sum(1 for row in rows if row["raw_exceeds_max_new_tokens"]),
            "mean_trimmed_tokens": round(statistics.mean([row["trimmed_tokens"] for row in rows]), 4) if rows else 0.0,
            "mean_tok_per_s_trimmed": round(statistics.mean([row["tok_per_s_trimmed"] for row in rows]), 4) if rows else 0.0,
            "mean_loop_count": round(statistics.mean([row["loop_count"] for row in rows]), 4) if rows else 0.0,
        }

    return {
        "all_records_match": all(record["all_turns_match"] for record in records),
        "matching_records": sum(1 for record in records if record["all_turns_match"]),
        "total_records": len(records),
        "matching_turns": sum(1 for turn in turns if turn["match"]),
        "total_turns": len(turns),
        "mismatch_turns": len(mismatches),
        "mismatch_locations": [
            {
                "question_id": record["question_id"],
                "turn_index": turn["turn_index"],
                "first_diff_pos": turn["comparison"]["first_diff_pos"],
            }
            for record in records
            for turn in record["turns"]
            if not turn["match"]
        ],
        "max_new_tokens": max_new_tokens,
        "methods": method_summary,
    }


def print_summary(summary: Dict) -> None:
    print("\n" + "=" * 76)
    print("SUMMARY")
    print(
        f"  record match: {summary['matching_records']}/{summary['total_records']} | "
        f"turn match: {summary['matching_turns']}/{summary['total_turns']}"
    )
    for method, item in summary["methods"].items():
        print(
            f"  {method:<5} tok/s={item['mean_tok_per_s_trimmed']:.2f}, "
            f"raw_overshoot_turns={item['raw_overshoot_turns']}"
        )
    if summary["mismatch_locations"]:
        print("  mismatches:")
        for item in summary["mismatch_locations"][:12]:
            print(
                f"    question={item['question_id']} turn={item['turn_index']} "
                f"first_diff={item['first_diff_pos']}"
            )
    print("=" * 76)


def sync_cuda() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _default_output_file(args) -> str:
    limit = "all" if args.limit is None else str(args.limit)
    return (
        f"lossless_{args.prompt_source}"
        f"_limit-{limit}_max-{args.max_new_tokens}.json"
    )


if __name__ == "__main__":
    main()
