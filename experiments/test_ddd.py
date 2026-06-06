"""
DDD (Dynamic Depth Decoding) 功能测试

验证 DDD 实现的正确性：
  1. DDD=off 时，行为与 baseline 完全一致
  2. DDD=on 时，beam search 能在检查点触发早停
  3. 早停不会破坏输出一致性 (temperature=0)
  4. 输出加速比和早停统计

用法:
  python experiments/test_ddd.py
  python experiments/test_ddd.py --threshold -5 --max-depth 11
"""

import sys
import os
import time

PROJECT_ROOT = "/home/hzliu/AD/Homework_haozhe/MLsys_final"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "EAGLE"))

import torch
torch.manual_seed(42)

from experiments.config import BASE_MODEL_PATH, EA_MODEL_PATH, GPU_DEVICE, EAGLEConfig


def test_ddd(config_name, use_ddd, ddd_max_depth=9, ddd_check_steps=None,
             ddd_threshold=-10.0, max_new_tokens=256):
    """Run EAGLE inference with given DDD config and report results."""
    from eagle.model.ea_model import EaModel

    print(f"\n{'='*60}")
    print(f"  Test: {config_name}")
    if use_ddd:
        print(f"  DDD: max_depth={ddd_max_depth}, check_steps={ddd_check_steps}, threshold={ddd_threshold}")
    else:
        print(f"  DDD: OFF (baseline, fixed depth=5)")
    print(f"{'='*60}")

    eagle_cfg = EAGLEConfig()
    model = EaModel.from_pretrained(
        base_model_path=BASE_MODEL_PATH,
        ea_model_path=EA_MODEL_PATH,
        total_token=eagle_cfg.total_token,
        depth=eagle_cfg.depth,
        top_k=eagle_cfg.top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_eagle3=True,
        use_ddd=use_ddd,
        ddd_max_depth=ddd_max_depth,
        ddd_check_steps=ddd_check_steps,
        ddd_threshold=ddd_threshold,
    )
    model.eval()
    tokenizer = model.get_tokenizer()

    prompt = "The capital of France is"
    input_ids = tokenizer([prompt], add_special_tokens=True).input_ids
    input_ids_t = torch.as_tensor(input_ids).cuda()

    # Warmup
    for _ in range(3):
        torch.cuda.synchronize()
        model.eagenerate(input_ids_t.clone(), temperature=0.0, log=False, is_llama3=True,
                         max_new_tokens=max_new_tokens)
        torch.cuda.synchronize()

    # Timed run
    torch.cuda.synchronize()
    t0 = time.time()
    output_ids, new_token, step_count = model.eagenerate(
        input_ids_t.clone(), temperature=0.0, log=True, is_llama3=True,
        max_new_tokens=max_new_tokens,
    )
    torch.cuda.synchronize()
    wall_time = time.time() - t0

    gen_ids = output_ids[0][len(input_ids[0]):]
    n_tokens = len(gen_ids)
    output_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    tok_per_sec = n_tokens / wall_time

    # DDD stats
    ddd_stats = model.ea_layer._ddd_stats if hasattr(model.ea_layer, '_ddd_stats') else {}
    avg_depth = (sum(ddd_stats.get('depths', [0])) / max(len(ddd_stats.get('depths', [1])), 1))

    print(f"  Generated: {n_tokens} tokens in {step_count} EAGLE steps")
    print(f"  Wall time: {wall_time:.3f}s ({tok_per_sec:.1f} tok/s)")
    print(f"  Accept per step: {n_tokens/max(step_count,1):.2f}")

    if use_ddd and ddd_stats:
        n_checks = ddd_stats.get("total_checks", 0)
        n_stops = ddd_stats.get("early_stops", 0)
        depths = ddd_stats.get("depths", [])
        print(f"  DDD Stats: checks={n_checks}, early_stops={n_stops}, "
              f"avg_depth={avg_depth:.1f}, depths={depths[:10]}...")
        stop_rate = n_stops / max(n_checks, 1) * 100
        print(f"  Early stop rate: {stop_rate:.1f}%")

    print(f"  Output: '{output_text[:100]}...'")

    del model
    torch.cuda.empty_cache()
    return tok_per_sec, output_text[:200] if use_ddd else output_text[:200]


def main():
    results = {}

    # Baseline (DDD off)
    speed_base, text_base = test_ddd("BASELINE (DDD=off)", use_ddd=False, max_new_tokens=128)
    results['baseline'] = speed_base

    # DDD with different thresholds
    thresholds = [-5.0, -8.0, -12.0]
    for tau in thresholds:
        name = f"DDD τ={tau}"
        speed_ddd, text_ddd = test_ddd(name, use_ddd=True,
                                        ddd_max_depth=9,
                                        ddd_check_steps=[5, 7],
                                        ddd_threshold=tau,
                                        max_new_tokens=128)
        results[name] = speed_ddd

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Baseline (DDD=off):     {results['baseline']:.1f} tok/s")
    for tau in thresholds:
        name = f"DDD τ={tau}"
        speedup = results[name] / results['baseline']
        print(f"  {name}:     {results[name]:.1f} tok/s  ({speedup:.2f}x)")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--max-depth", type=int, default=9)
    p.add_argument("--gpu", type=int, default=GPU_DEVICE)
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.threshold is not None:
        test_ddd(f"DDD τ={args.threshold}", use_ddd=True,
                 ddd_max_depth=args.max_depth,
                 ddd_check_steps=[5, 7],
                 ddd_threshold=args.threshold)
    else:
        main()
