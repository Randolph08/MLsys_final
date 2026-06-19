"""
DDD 超参数网格搜索

搜索空间:
  τ ∈ [-4, -6, -8, -10, -12]
  check_steps ∈ [[3,5], [5,7], [4,6,8]]
  max_depth ∈ [7, 9, 11]

共 5 × 3 × 3 = 45 组配置
"""
import sys, os, time, json, statistics, itertools

PROJECT_ROOT = "/home/hzliu/AD/Homework_haozhe/MLsys_final"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "EAGLE"))

import torch
torch.manual_seed(42)

from experiments.config import BASE_MODEL_PATH, EA_MODEL_PATH, GPU_DEVICE

PROMPTS = [
    "What is the capital of Japan?",             # 事实问答
    "Write a recursive function to compute factorial in Python:",  # 代码生成
    "Write a short poem about autumn leaves:",   # 创意写作
]
MAX_NEW_TOKENS = 128

TAU_VALUES = [-4, -6, -8, -10, -12]
CHECK_STEPS_LIST = [[3, 5], [5, 7], [4, 6, 8]]
MAX_DEPTH_VALUES = [7, 9, 11]


def run_config(tau, check_steps, max_depth):
    from eagle.model.ea_model import EaModel

    label = f"τ={tau} check={check_steps} depth={max_depth}"
    model = EaModel.from_pretrained(
        base_model_path=BASE_MODEL_PATH, ea_model_path=EA_MODEL_PATH,
        total_token=60, depth=5, top_k=10,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
        device_map="auto", use_eagle3=True,
        use_ddd=True, ddd_max_depth=max_depth,
        ddd_check_steps=check_steps, ddd_threshold=float(tau),
        use_opt_tree=False,
    )
    model.eval()
    tokenizer = model.get_tokenizer()

    tok_s_list, acc_list, depth_list = [], [], []
    for prompt in PROMPTS:
        input_ids = tokenizer([prompt], add_special_tokens=True).input_ids
        input_t = torch.as_tensor(input_ids).cuda()
        for _ in range(2):
            torch.cuda.synchronize()
            model.eagenerate(input_t.clone(), temperature=0.0, log=False,
                             is_llama3=True, max_new_tokens=MAX_NEW_TOKENS)
            torch.cuda.synchronize()

        torch.cuda.synchronize()
        t0 = time.time()
        _, new_tok, steps = model.eagenerate(
            input_t.clone(), temperature=0.0, log=True, is_llama3=True,
            max_new_tokens=MAX_NEW_TOKENS)
        torch.cuda.synchronize()
        wall = time.time() - t0

        n = len(input_ids[0])  # just for reference
        tok_s_list.append(MAX_NEW_TOKENS / wall)
        acc_list.append(MAX_NEW_TOKENS / max(steps, 1))

        if hasattr(model.ea_layer, '_ddd_stats'):
            depths = model.ea_layer._ddd_stats.get('depths', [0])
            depth_list.append(sum(depths) / max(len(depths), 1))

    del model; torch.cuda.empty_cache()

    return {
        'tau': tau, 'check_steps': check_steps, 'max_depth': max_depth,
        'mean_tok_s': round(statistics.mean(tok_s_list), 1),
        'mean_accept': round(statistics.mean(acc_list), 2),
        'mean_actual_depth': round(statistics.mean(depth_list), 1) if depth_list else 0,
    }


def main():
    out_dir = os.path.join(PROJECT_ROOT, "experiments", "hparam_search")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "ddd_grid.json")

    if os.path.exists(path):
        print("[hparam] 已有数据，加载中...")
        with open(path) as f:
            results = json.load(f)
    else:
        total = len(TAU_VALUES) * len(CHECK_STEPS_LIST) * len(MAX_DEPTH_VALUES)
        results = []
        count = 0
        for tau, cs, md in itertools.product(TAU_VALUES, CHECK_STEPS_LIST, MAX_DEPTH_VALUES):
            count += 1
            print(f"[{count}/{total}] τ={tau}, check={cs}, depth={md}")
            r = run_config(tau, cs, md)
            results.append(r)
            print(f"  -> {r['mean_tok_s']:.1f} tok/s, accept={r['mean_accept']:.2f}, "
                  f"actual_depth={r['mean_actual_depth']:.1f}")
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)

    # Also run baseline once for reference
    print("\n[Baseline]")
    from eagle.model.ea_model import EaModel
    model = EaModel.from_pretrained(
        base_model_path=BASE_MODEL_PATH, ea_model_path=EA_MODEL_PATH,
        total_token=60, depth=5, top_k=10,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
        device_map="auto", use_eagle3=True,
        use_ddd=False, use_opt_tree=False,
    )
    model.eval()
    tok = model.get_tokenizer()
    base_speeds = []
    for prompt in PROMPTS:
        ids = torch.as_tensor(tok([prompt], add_special_tokens=True).input_ids).cuda()
        for _ in range(2):
            torch.cuda.synchronize()
            model.eagenerate(ids.clone(), temperature=0.0, log=False, is_llama3=True, max_new_tokens=MAX_NEW_TOKENS)
            torch.cuda.synchronize()
        torch.cuda.synchronize()
        t0 = time.time()
        model.eagenerate(ids.clone(), temperature=0.0, log=False, is_llama3=True, max_new_tokens=MAX_NEW_TOKENS)
        torch.cuda.synchronize()
        base_speeds.append(MAX_NEW_TOKENS / (time.time() - t0))
    del model; torch.cuda.empty_cache()
    baseline_tok_s = statistics.mean(base_speeds)
    print(f"  Baseline: {baseline_tok_s:.1f} tok/s")

    # Find best config
    best = max(results, key=lambda r: r['mean_tok_s'])
    speedup = best['mean_tok_s'] / baseline_tok_s

    # Summary
    print(f"\n{'='*75}")
    print("DDD HYPERPARAMETER SEARCH RESULTS")
    print(f"{'='*75}")
    print(f"  Baseline: {baseline_tok_s:.1f} tok/s")
    print(f"  Best DDD: {best['mean_tok_s']:.1f} tok/s ({speedup:.2f}x)")
    print(f"  Best params: τ={best['tau']}, check_steps={best['check_steps']}, "
          f"max_depth={best['max_depth']}")
    print(f"  Best actual_depth: {best['mean_actual_depth']:.1f}")
    print(f"  Total configs tested: {len(results)}")
    print(f"{'='*75}")

    # Top-5
    print(f"\n  Top-5 configs:")
    top5 = sorted(results, key=lambda r: r['mean_tok_s'], reverse=True)[:5]
    for i, r in enumerate(top5):
        su = r['mean_tok_s'] / baseline_tok_s
        print(f"  {i+1}. τ={r['tau']:>4}, check={str(r['check_steps']):>10}, "
              f"depth={r['max_depth']:>2} -> {r['mean_tok_s']:.1f} tok/s ({su:.2f}x), "
              f"accept={r['mean_accept']:.2f}")

    # Per-τ summary
    print(f"\n  Per-τ averages:")
    for tau in TAU_VALUES:
        vals = [r['mean_tok_s'] for r in results if r['tau'] == tau]
        print(f"  τ={tau:>4}: avg={statistics.mean(vals):.1f}, best={max(vals):.1f}, "
              f"worst={min(vals):.1f} tok/s")

    # Per-check_steps summary
    print(f"\n  Per-check_steps averages:")
    for cs in CHECK_STEPS_LIST:
        vals = [r['mean_tok_s'] for r in results if r['check_steps'] == cs]
        print(f"  {str(cs):>10}: avg={statistics.mean(vals):.1f}, best={max(vals):.1f} tok/s")

    # Per-max_depth summary
    print(f"\n  Per-max_depth averages:")
    for md in MAX_DEPTH_VALUES:
        vals = [r['mean_tok_s'] for r in results if r['max_depth'] == md]
        print(f"  depth={md:>2}: avg={statistics.mean(vals):.1f}, best={max(vals):.1f} tok/s")

    print(f"\n  Data: {out_dir}/")
    print(f"{'='*75}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=GPU_DEVICE)
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    main()
