"""
E4.1: 分布一致性验证 (Lossless Verification)

验证 temperature=0 时 Pure AR / Baseline / DDD / OPT-Tree 输出完全一致。
"""
import sys, os, time, json

PROJECT_ROOT = "/home/hzliu/AD/Homework_haozhe/MLsys_final"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "EAGLE"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.config import BASE_MODEL_PATH, EA_MODEL_PATH, GPU_DEVICE

PROMPTS = [
    "The capital of France is",
    "Explain the concept of machine learning in simple terms:",
    "Write a Python function to find the nth Fibonacci number:",
    "What are the main causes of climate change?",
    "Translate the following to French: Hello, how are you?",
]
MAX_TOKENS = 128


def run_pure_ar(prompts):
    """Pure AR baseline with model.generate()"""
    print("\n[Pure AR] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    model.eval()

    outputs = {}
    for prompt in prompts:
        input_ids = tokenizer([prompt], add_special_tokens=True).input_ids
        input_t = torch.as_tensor(input_ids).cuda()
        out = model.generate(input_t, max_new_tokens=MAX_TOKENS, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        gen_ids = out[0][len(input_ids[0]):].tolist()
        outputs[prompt[:50]] = gen_ids
        print(f"  '{prompt[:40]}...' -> {len(gen_ids)} tokens")
    del model
    torch.cuda.empty_cache()
    return outputs


def run_eagle(name, model_kwargs, prompts):
    """Run one EAGLE config"""
    from eagle.model.ea_model import EaModel

    print(f"\n[{name}] Loading model...")
    model = EaModel.from_pretrained(
        base_model_path=BASE_MODEL_PATH,
        ea_model_path=EA_MODEL_PATH,
        total_token=60, depth=model_kwargs.get('depth', 5), top_k=10,
        torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto",
        use_eagle3=True,
        use_ddd=model_kwargs.get('use_ddd', False),
        ddd_max_depth=model_kwargs.get('ddd_max_depth', 9),
        ddd_check_steps=model_kwargs.get('ddd_check_steps', None),
        ddd_threshold=model_kwargs.get('ddd_threshold', -10.0),
        use_opt_tree=model_kwargs.get('use_opt_tree', False),
        opt_expand_factor=model_kwargs.get('opt_expand_factor', 2.0),
    )
    model.eval()
    tokenizer = model.get_tokenizer()

    outputs = {}
    for prompt in prompts:
        input_ids = tokenizer([prompt], add_special_tokens=True).input_ids
        input_t = torch.as_tensor(input_ids).cuda()
        out_ids = model.eagenerate(input_t.clone(), temperature=0.0, log=False,
                                   is_llama3=True, max_new_tokens=MAX_TOKENS)
        gen_ids = out_ids[0][len(input_ids[0]):].tolist()
        outputs[prompt[:50]] = gen_ids
        print(f"  '{prompt[:40]}...' -> {len(gen_ids)} tokens")
    del model
    torch.cuda.empty_cache()
    return outputs


def main():
    print("=" * 60)
    print("E4.1: Lossless Verification")
    print("=" * 60)

    # Run all configs
    ref = run_pure_ar(PROMPTS)
    baseline = run_eagle("Baseline EAGLE-3", {'depth': 5}, PROMPTS)
    ddd = run_eagle("DDD", {'depth': 5, 'use_ddd': True,
                             'ddd_max_depth': 9, 'ddd_check_steps': [5, 7],
                             'ddd_threshold': -6.0}, PROMPTS)
    opt = run_eagle("OPT-Tree", {'depth': 5, 'use_opt_tree': True,
                                  'opt_expand_factor': 1.5}, PROMPTS)

    # Compare
    print(f"\n{'='*60}")
    print("COMPARISON (token-by-token identity check)")
    print(f"{'='*60}")

    configs = {'Pure AR': ref, 'Baseline': baseline, 'DDD': ddd, 'OPT-Tree': opt}
    all_match = True

    for prompt_key in ref:
        ref_tokens = ref[prompt_key]
        print(f"\n  Prompt: '{prompt_key}...'")
        print(f"  Reference (Pure AR): {len(ref_tokens)} tokens")
        for name, output in configs.items():
            if name == 'Pure AR':
                continue
            tokens = output[prompt_key]
            match = (tokens == ref_tokens)
            status = "✓ MATCH" if match else "✗ DIFF"
            if not match:
                all_match = False
                # Find first diff
                for i, (a, b) in enumerate(zip(ref_tokens, tokens)):
                    if a != b:
                        print(f"    First diff at pos {i}: AR={a}, {name}={b}")
                        break
            print(f"  {name:>12}: {len(tokens)} tokens — {status}")

    print(f"\n{'='*60}")
    if all_match:
        print("CONCLUSION: All configs produce IDENTICAL output.")
        print("           Lossless property is PRESERVED. ✓")
    else:
        print("CONCLUSION: MISMATCH detected — investigate!")
    print(f"{'='*60}")

    # Save
    out_dir = os.path.join(PROJECT_ROOT, "experiments", "E4_lossless")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "lossless_result.json"), 'w') as f:
        json.dump({
            'all_match': all_match,
            'prompts': list(ref.keys()),
            'results': {k: {'tokens': v} for k, v in configs.items()},
        }, f, indent=2)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=GPU_DEVICE)
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    main()
