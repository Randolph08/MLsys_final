"""
E3.1: DDD 消融实验

对比 4 组配置在 5 个 prompt 上的吞吐量和接受长度:
  Baseline(depth=5): 当前默认
  Baseline(depth=7): 更深但无早停
  DDD(τ=-10, check=[5,7]):  保守早停
  DDD(τ=-6,  check=[5,7]):  激进早停

输出:
  experiments/E3_ablation/ddd_ablation.json
  experiments/E3_ablation/ddd_raw.json
"""

import sys, os, time, json, statistics
from collections import defaultdict

PROJECT_ROOT = "/home/hzliu/AD/Homework_haozhe/MLsys_final"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "EAGLE"))

import torch
torch.manual_seed(42)

from experiments.config import BASE_MODEL_PATH, EA_MODEL_PATH, GPU_DEVICE, EAGLEConfig


PROMPTS = [
    "The capital of France is",
    "Explain the concept of machine learning in simple terms:",
    "Write a Python function to find the nth Fibonacci number:",
    "What are the main causes of climate change?",
    "Translate the following to French: Hello, how are you?",
]
MAX_NEW_TOKENS = 256


def run_config(name, model_kwargs, prompts, max_new_tokens):
    """Run one EAGLE config on all prompts, return stats."""
    from eagle.model.ea_model import EaModel

    print(f"\n{'='*60}")
    print(f"  Config: {name}")
    for k, v in model_kwargs.items():
        if k not in ('use_ddd', 'ddd_max_depth', 'ddd_check_steps', 'ddd_threshold', 'depth'):
            pass
    depth_val = model_kwargs.get('depth', 5)
    use_ddd = model_kwargs.get('use_ddd', False)
    if use_ddd:
        print(f"  DDD: max_depth={model_kwargs.get('ddd_max_depth')}, "
              f"check={model_kwargs.get('ddd_check_steps')}, τ={model_kwargs.get('ddd_threshold')}")
    else:
        print(f"  Fixed depth: {depth_val}")
    print(f"{'='*60}")

    model = EaModel.from_pretrained(
        base_model_path=BASE_MODEL_PATH,
        ea_model_path=EA_MODEL_PATH,
        total_token=60,
        depth=model_kwargs.get('depth', 5),
        top_k=10,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_eagle3=True,
        use_ddd=use_ddd,
        ddd_max_depth=model_kwargs.get('ddd_max_depth', 9),
        ddd_check_steps=model_kwargs.get('ddd_check_steps', None),
        ddd_threshold=model_kwargs.get('ddd_threshold', -10.0),
    )
    model.eval()
    tokenizer = model.get_tokenizer()

    results = []
    for pi, prompt in enumerate(prompts):
        input_ids = tokenizer([prompt], add_special_tokens=True).input_ids
        input_ids_t = torch.as_tensor(input_ids).cuda()

        # Warmup
        for _ in range(2):
            torch.cuda.synchronize()
            model.eagenerate(input_ids_t.clone(), temperature=0.0, log=False,
                             is_llama3=True, max_new_tokens=max_new_tokens)
            torch.cuda.synchronize()

        # Timed
        torch.cuda.synchronize()
        t0 = time.time()
        output_ids, new_token, step_count = model.eagenerate(
            input_ids_t.clone(), temperature=0.0, log=True, is_llama3=True,
            max_new_tokens=max_new_tokens,
        )
        torch.cuda.synchronize()
        wall = time.time() - t0

        n_tok = len(output_ids[0]) - len(input_ids[0])
        tok_s = n_tok / wall if wall > 0 else 0

        # DDD stats
        ddd_s = {}
        if use_ddd and hasattr(model.ea_layer, '_ddd_stats'):
            s = model.ea_layer._ddd_stats
            ddd_s = {
                'early_stops': s.get('early_stops', 0),
                'total_checks': s.get('total_checks', 0),
                'avg_depth': sum(s.get('depths', [0])) / max(len(s.get('depths', [1])), 1),
            }

        results.append({
            'prompt': prompt[:50],
            'n_tokens': n_tok,
            'n_steps': step_count,
            'wall_time_s': round(wall, 3),
            'tok_per_s': round(tok_s, 1),
            'accept_per_step': round(n_tok / max(step_count, 1), 2),
            'ddd_stats': ddd_s,
        })
        print(f"  [{pi+1}/{len(prompts)}] {n_tok} tok, {step_count} steps, "
              f"{wall:.2f}s, {tok_s:.1f} tok/s")

    del model
    torch.cuda.empty_cache()

    # Aggregate
    tok_s_list = [r['tok_per_s'] for r in results]
    acc_list = [r['accept_per_step'] for r in results]
    return {
        'name': name,
        'config': {k: v for k, v in model_kwargs.items() if not k.startswith('_')},
        'results': results,
        'mean_tok_per_s': round(statistics.mean(tok_s_list), 1),
        'std_tok_per_s': round(statistics.stdev(tok_s_list), 1) if len(tok_s_list) > 1 else 0,
        'mean_accept_per_step': round(statistics.mean(acc_list), 2),
        'total_tokens': sum(r['n_tokens'] for r in results),
        'total_time_s': round(sum(r['wall_time_s'] for r in results), 3),
    }


def main():
    out_dir = os.path.join(PROJECT_ROOT, "experiments", "E3_ablation")
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "ddd_ablation.json")
    raw_path = os.path.join(out_dir, "ddd_raw.json")

    # Check if already cached
    if os.path.exists(summary_path):
        print("[E3.1] 已有消融数据，跳过运行。删除 ddd_ablation.json 可重新运行。")
        with open(summary_path) as f:
            all_results = json.load(f)
        for r in all_results:
            print(f"  {r['name']}: {r['mean_tok_per_s']:.1f} tok/s, "
                  f"accept={r['mean_accept_per_step']:.2f}")
        return all_results

    configs = [
        ("Baseline depth=5", {
            'depth': 5, 'use_ddd': False,
        }),
        ("Baseline depth=7", {
            'depth': 7, 'use_ddd': False,
        }),
        ("DDD τ=-10 check=[5,7]", {
            'depth': 5, 'use_ddd': True,
            'ddd_max_depth': 9, 'ddd_check_steps': [5, 7],
            'ddd_threshold': -10.0,
        }),
        ("DDD τ=-6 check=[5,7]", {
            'depth': 5, 'use_ddd': True,
            'ddd_max_depth': 9, 'ddd_check_steps': [5, 7],
            'ddd_threshold': -6.0,
        }),
    ]

    all_results = []
    for name, kwargs in configs:
        r = run_config(name, kwargs, PROMPTS, MAX_NEW_TOKENS)
        all_results.append(r)

    # Save
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    with open(raw_path, 'w') as f:
        json.dump([r['results'] for r in all_results], f, indent=2, default=str)

    # Summary table
    print(f"\n{'='*70}")
    print("E3.1 DDD ABLATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<35} {'tok/s':>8} {'accept':>8} {'total_tok':>10} {'time':>8}")
    print(f"{'-'*35} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
    baseline_speed = None
    for r in all_results:
        if baseline_speed is None:
            baseline_speed = r['mean_tok_per_s']
        speedup = r['mean_tok_per_s'] / baseline_speed if baseline_speed > 0 else 0
        su_str = f"{speedup:.2f}x"
        print(f"{r['name']:<35} {r['mean_tok_per_s']:>7.1f} {r['mean_accept_per_step']:>7.2f} "
              f"{r['total_tokens']:>10} {r['total_time_s']:>7.1f}s  ({su_str})")
    print(f"{'='*70}")
    print(f"  Data saved to: {out_dir}/")
    print(f"{'='*70}")

    return all_results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=GPU_DEVICE)
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    main()
