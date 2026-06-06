"""
OPT-Tree 功能测试
"""
import sys, os, time
PROJECT_ROOT = "/home/hzliu/AD/Homework_haozhe/MLsys_final"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "EAGLE"))

import torch
torch.manual_seed(42)

from experiments.config import BASE_MODEL_PATH, EA_MODEL_PATH, GPU_DEVICE, EAGLEConfig


def test_opt(config_name, use_opt_tree, opt_expand_factor=2.0, max_new_tokens=256):
    from eagle.model.ea_model import EaModel

    print(f"\n{'='*60}")
    print(f"  Test: {config_name}")
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
        use_opt_tree=use_opt_tree,
        opt_expand_factor=opt_expand_factor,
    )
    model.eval()
    tokenizer = model.get_tokenizer()

    prompt = "The capital of France is"
    input_ids = tokenizer([prompt], add_special_tokens=True).input_ids
    input_ids_t = torch.as_tensor(input_ids).cuda()

    for _ in range(3):
        torch.cuda.synchronize()
        model.eagenerate(input_ids_t.clone(), temperature=0.0, log=False, is_llama3=True,
                         max_new_tokens=max_new_tokens)
        torch.cuda.synchronize()

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

    print(f"  Generated: {n_tokens} tokens in {step_count} steps")
    print(f"  Wall time: {wall_time:.3f}s  ({tok_per_sec:.1f} tok/s)")
    print(f"  Accept/step: {n_tokens/max(step_count,1):.2f}")
    print(f"  Output: '{output_text[:120]}...'")

    del model
    torch.cuda.empty_cache()
    return tok_per_sec, output_text[:200], n_tokens, step_count


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=GPU_DEVICE)
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Baseline
    s_base, t_base, n_base, steps_base = test_opt("BASELINE", use_opt_tree=False, max_new_tokens=128)

    # OPT-Tree with different expand factors
    for factor in [1.5, 2.0, 3.0]:
        s_opt, t_opt, n_opt, steps_opt = test_opt(
            f"OPT-Tree ×{factor}", use_opt_tree=True,
            opt_expand_factor=factor, max_new_tokens=128)

        print(f"  vs baseline: {s_opt:.1f} tok/s ({s_opt/s_base:.2f}x), "
              f"steps: {steps_base}→{steps_opt}, tokens: {n_base}→{n_opt}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"  Baseline: {s_base:.1f} tok/s ({n_base} tokens, {steps_base} steps)")
    print(f"{'='*60}")
