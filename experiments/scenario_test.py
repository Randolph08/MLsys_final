"""
E4.2: 场景泛化性分析

测试 Baseline 和 DDD 在 4 类 prompt 上的效果差异：
  事实问答 / 代码生成 / 创意写作 / 翻译
"""
import sys, os, time, json, statistics

PROJECT_ROOT = "/home/hzliu/AD/Homework_haozhe/MLsys_final"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "EAGLE"))

import torch
from experiments.config import BASE_MODEL_PATH, EA_MODEL_PATH, GPU_DEVICE

SCENARIOS = {
    "事实问答": [
        "What is the capital of Japan?",
        "How many planets are in our solar system?",
        "Who wrote the novel '1984'?",
    ],
    "代码生成": [
        "Write a Python function to check if a string is a palindrome:",
        "Write a recursive function to compute factorial in Python:",
        "Write a function to merge two sorted lists in Python:",
    ],
    "创意写作": [
        "Write a short poem about autumn leaves:",
        "Write a haiku about technology:",
        "Begin a short story with: The old house on the hill had been empty for years...",
    ],
    "翻译": [
        "Translate to French: The weather is beautiful today.",
        "Translate to German: I would like a cup of coffee please.",
        "Translate to Spanish: Where is the nearest train station?",
    ],
}
MAX_NEW_TOKENS = 128


def test_scenario(name, prompts, use_ddd):
    from eagle.model.ea_model import EaModel

    label = "DDD" if use_ddd else "Baseline"
    print(f"\n  [{label}] {name} ({len(prompts)} prompts)")

    model = EaModel.from_pretrained(
        base_model_path=BASE_MODEL_PATH, ea_model_path=EA_MODEL_PATH,
        total_token=60, depth=5, top_k=10,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
        device_map="auto", use_eagle3=True,
        use_ddd=use_ddd,
        ddd_max_depth=9 if use_ddd else None,
        ddd_check_steps=[5, 7] if use_ddd else None,
        ddd_threshold=-6.0 if use_ddd else None,
        use_opt_tree=False,
    )
    model.eval()
    tokenizer = model.get_tokenizer()

    results = []
    for prompt in prompts:
        input_ids = tokenizer([prompt], add_special_tokens=True).input_ids
        input_t = torch.as_tensor(input_ids).cuda()
        for _ in range(2):
            torch.cuda.synchronize()
            model.eagenerate(input_t.clone(), temperature=0.0, log=False,
                             is_llama3=True, max_new_tokens=MAX_NEW_TOKENS)
            torch.cuda.synchronize()

        torch.cuda.synchronize()
        t0 = time.time()
        out_ids, new_tok, steps = model.eagenerate(
            input_t.clone(), temperature=0.0, log=True, is_llama3=True,
            max_new_tokens=MAX_NEW_TOKENS)
        torch.cuda.synchronize()
        wall = time.time() - t0

        n = len(out_ids[0]) - len(input_ids[0])
        results.append({'n_tokens': n, 'n_steps': steps, 'wall_s': round(wall, 3),
                        'tok_s': round(n / wall, 1), 'accept': round(n / max(steps, 1), 2)})

    del model; torch.cuda.empty_cache()

    tok_s = [r['tok_s'] for r in results]
    acc = [r['accept'] for r in results]
    return {
        'scenario': name, 'use_ddd': use_ddd,
        'results': results,
        'mean_tok_s': round(statistics.mean(tok_s), 1),
        'mean_accept': round(statistics.mean(acc), 2),
    }


def main():
    out_dir = os.path.join(PROJECT_ROOT, "experiments", "E4_scenarios")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "scenario_results.json")

    if os.path.exists(path):
        print("[E4.2] 已有数据，跳过。")
        with open(path) as f:
            data = json.load(f)
    else:
        data = []
        for use_ddd in [False, True]:
            for name, prompts in SCENARIOS.items():
                data.append(test_scenario(name, prompts, use_ddd))
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    # Summary table
    print(f"\n{'='*75}")
    print("E4.2: SCENARIO GENERALIZATION")
    print(f"{'='*75}")
    print(f"{'Scenario':<15} {'Base tok/s':>12} {'DDD tok/s':>12} {'Speedup':>10} {'Base acc':>10} {'DDD acc':>10}")
    print(f"{'-'*15} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")

    for name in SCENARIOS:
        b = next(r for r in data if r['scenario'] == name and not r['use_ddd'])
        d = next(r for r in data if r['scenario'] == name and r['use_ddd'])
        su = d['mean_tok_s'] / b['mean_tok_s']
        print(f"{name:<15} {b['mean_tok_s']:>11.1f} {d['mean_tok_s']:>11.1f} {su:>9.2f}x "
              f"{b['mean_accept']:>9.2f} {d['mean_accept']:>9.2f}")

    print(f"{'='*75}")
    print(f"  Data: {out_dir}/")
    print(f"{'='*75}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=GPU_DEVICE)
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    main()
