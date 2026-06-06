"""
E3.2 + E3.3: OPT-Tree & Joint Ablation

对比 4 组配置:
  Baseline(depth=5)
  DDD (τ=-6, check=[5,7])
  OPT-Tree (expand=1.5)
  DDD + OPT-Tree (joint)

输出: experiments/E3_ablation/full_ablation.json
"""
import sys, os, time, json, statistics

PROJECT_ROOT = "/home/hzliu/AD/Homework_haozhe/MLsys_final"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "EAGLE"))

import torch
torch.manual_seed(42)

from experiments.config import BASE_MODEL_PATH, EA_MODEL_PATH, GPU_DEVICE

PROMPTS = [
    "The capital of France is",
    "Explain the concept of machine learning in simple terms:",
    "Write a Python function to find the nth Fibonacci number:",
    "What are the main causes of climate change?",
    "Translate the following to French: Hello, how are you?",
]
MAX_NEW_TOKENS = 256


def run_config(name, model_kwargs):
    from eagle.model.ea_model import EaModel

    print(f"\n{'='*60}")
    ddd_on = model_kwargs.get('use_ddd', False)
    opt_on = model_kwargs.get('use_opt_tree', False)
    flags = []
    if ddd_on: flags.append(f"DDD")
    if opt_on: flags.append(f"OPT-Tree")
    print(f"  {name}: {'+'.join(flags) if flags else 'Baseline'}")
    print(f"{'='*60}")

    model = EaModel.from_pretrained(
        base_model_path=BASE_MODEL_PATH, ea_model_path=EA_MODEL_PATH,
        total_token=60, depth=5, top_k=10,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
        device_map="auto", use_eagle3=True,
        use_ddd=ddd_on,
        ddd_max_depth=model_kwargs.get('ddd_max_depth', 9),
        ddd_check_steps=model_kwargs.get('ddd_check_steps', None),
        ddd_threshold=model_kwargs.get('ddd_threshold', -10.0),
        use_opt_tree=opt_on,
        opt_expand_factor=model_kwargs.get('opt_expand_factor', 2.0),
    )
    model.eval()
    tokenizer = model.get_tokenizer()

    results = []
    for pi, prompt in enumerate(PROMPTS):
        input_ids = tokenizer([prompt], add_special_tokens=True).input_ids
        input_ids_t = torch.as_tensor(input_ids).cuda()

        for _ in range(2):
            torch.cuda.synchronize()
            model.eagenerate(input_ids_t.clone(), temperature=0.0, log=False,
                             is_llama3=True, max_new_tokens=MAX_NEW_TOKENS)
            torch.cuda.synchronize()

        torch.cuda.synchronize()
        t0 = time.time()
        output_ids, new_token, step_count = model.eagenerate(
            input_ids_t.clone(), temperature=0.0, log=True, is_llama3=True,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        torch.cuda.synchronize()
        wall = time.time() - t0

        n_tok = len(output_ids[0]) - len(input_ids[0])
        tok_s = n_tok / wall if wall > 0 else 0
        results.append({
            'prompt': prompt[:50], 'n_tokens': n_tok, 'n_steps': step_count,
            'wall_time_s': round(wall, 3), 'tok_per_s': round(tok_s, 1),
            'accept_per_step': round(n_tok / max(step_count, 1), 2),
        })
        print(f"  [{pi+1}/5] {n_tok} tok, {step_count} steps, {wall:.2f}s, {tok_s:.1f} tok/s")

    del model
    torch.cuda.empty_cache()

    tok_s_list = [r['tok_per_s'] for r in results]
    acc_list = [r['accept_per_step'] for r in results]
    return {
        'name': name,
        'results': results,
        'mean_tok_per_s': round(statistics.mean(tok_s_list), 1),
        'mean_accept_per_step': round(statistics.mean(acc_list), 2),
        'total_tokens': sum(r['n_tokens'] for r in results),
        'total_time_s': round(sum(r['wall_time_s'] for r in results), 3),
        'lossless_vs_baseline': None,  # will be filled in
    }


def main():
    out_dir = os.path.join(PROJECT_ROOT, "experiments", "E3_ablation")
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "full_ablation.json")

    if os.path.exists(summary_path):
        print("[E3.2/3.3] 已有消融数据，跳过。删除 full_ablation.json 可重新运行。")
        with open(summary_path) as f:
            data = json.load(f)
        for r in data:
            print(f"  {r['name']}: {r['mean_tok_per_s']:.1f} tok/s, accept={r['mean_accept_per_step']:.2f}")
        return data

    configs = [
        ("Baseline", {'use_ddd': False, 'use_opt_tree': False}),
        ("DDD", {'use_ddd': True, 'ddd_max_depth': 9,
                 'ddd_check_steps': [5, 7], 'ddd_threshold': -6.0,
                 'use_opt_tree': False}),
        ("OPT-Tree", {'use_ddd': False, 'use_opt_tree': True,
                      'opt_expand_factor': 1.5}),
        ("DDD+OPT-Tree", {'use_ddd': True, 'ddd_max_depth': 9,
                          'ddd_check_steps': [5, 7], 'ddd_threshold': -6.0,
                          'use_opt_tree': True, 'opt_expand_factor': 1.5}),
    ]

    all_results = []
    baseline_tokens = None
    for name, kwargs in configs:
        r = run_config(name, kwargs)
        # Check lossless vs baseline
        cur_tokens = sum(x['n_tokens'] for x in r['results'])
        if baseline_tokens is None:
            baseline_tokens = cur_tokens
            r['lossless_vs_baseline'] = True
        else:
            r['lossless_vs_baseline'] = (cur_tokens == baseline_tokens)
        all_results.append(r)

    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    baseline_speed = all_results[0]['mean_tok_per_s']
    print(f"\n{'='*70}")
    print("E3.2 + E3.3 FULL ABLATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<20} {'tok/s':>8} {'vs Base':>8} {'accept':>8} {'lossless':>10}")
    print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for r in all_results:
        su = r['mean_tok_per_s'] / baseline_speed
        ll = '✓' if r['lossless_vs_baseline'] else '✗'
        print(f"{r['name']:<20} {r['mean_tok_per_s']:>7.1f} {su:>7.2f}x {r['mean_accept_per_step']:>7.2f} {ll:>10}")
    print(f"{'='*70}")
    print(f"  Data: {out_dir}/")
    print(f"{'='*70}")

    # Also show per-prompt breakdown
    print(f"\n{'Prompt':<55} {'Base':>8} {'DDD':>8} {'OPT':>8} {'Joint':>8}")
    for pi, p in enumerate(PROMPTS):
        speeds = [r['results'][pi]['tok_per_s'] for r in all_results]
        print(f"  {p[:52]:<52} {speeds[0]:>7.1f} {speeds[1]:>7.1f} {speeds[2]:>7.1f} {speeds[3]:>7.1f}")

    return all_results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=GPU_DEVICE)
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    main()
