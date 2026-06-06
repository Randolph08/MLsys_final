"""
Diagnostic script: Compare EAGLE-3 vs pure AR performance
to identify if the EAGLE pipeline is working correctly.
"""
import torch
import sys
import time

sys.path.insert(0, '/home/hzliu/AD/Homework_haozhe/MLsys_final/EAGLE')

from eagle.model.ea_model import EaModel
from eagle.model.utils import prepare_logits_processor

BASE_MODEL_PATH = "/home/hzliu/AD/Homework_haozhe/MLsys_final/models/DeepSeek-R1-Distill-Llama-8B"
EA_MODEL_PATH = "/home/hzliu/AD/Homework_haozhe/MLsys_final/EAGLE_checkpoints/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B"

def test_config(name, total_token, depth, top_k, prompt, max_new=256):
    """Run EAGLE-3 with given tree parameters and measure speed."""
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"  total_token={total_token}, depth={depth}, top_k={top_k}")
    print(f"  Prompt: '{prompt[:80]}...'")

    model = EaModel.from_pretrained(
        base_model_path=BASE_MODEL_PATH,
        ea_model_path=EA_MODEL_PATH,
        total_token=total_token,
        depth=depth,
        top_k=top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_eagle3=True,
    )
    model.eval()
    tokenizer = model.get_tokenizer()

    input_ids = tokenizer([prompt], add_special_tokens=True).input_ids

    # Warmup
    for _ in range(2):
        torch.cuda.synchronize()
        model.eagenerate(torch.as_tensor(input_ids).cuda(), temperature=0.0, log=False, is_llama3=True)
        torch.cuda.synchronize()

    # Timed run
    torch.cuda.synchronize()
    t0 = time.time()
    output_ids, new_token, step_count = model.eagenerate(
        torch.as_tensor(input_ids).cuda(), temperature=0.0, log=True, is_llama3=True,
        max_new_tokens=max_new,
    )
    torch.cuda.synchronize()
    t1 = time.time()

    gen_ids = output_ids[0][len(input_ids[0]):]
    n_tokens = len(gen_ids)
    walltime = t1 - t0

    print(f"  Generated {n_tokens} tokens in {step_count} EAGLE steps")
    print(f"  Wall time: {walltime:.3f}s")
    print(f"  Tokens/sec: {n_tokens/walltime:.1f}")
    print(f"  Accepts/step: {n_tokens/max(step_count,1):.2f}")
    print(f"  Avg step time: {1000*walltime/step_count:.1f}ms")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return n_tokens/walltime


def test_pure_ar(prompt, max_new=256):
    """Run pure AR decoding as baseline."""
    print(f"\n{'='*60}")
    print(f"Pure AR Baseline")
    print(f"  Prompt: '{prompt[:80]}...'")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

    input_ids = tokenizer([prompt], add_special_tokens=True).input_ids
    input_ids_t = torch.as_tensor(input_ids).cuda()

    # Warmup
    for _ in range(2):
        torch.cuda.synchronize()
        model.generate(input_ids_t, max_new_tokens=32, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()

    # Timed run
    torch.cuda.synchronize()
    t0 = time.time()
    output = model.generate(
        input_ids_t, max_new_tokens=max_new, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    torch.cuda.synchronize()
    t1 = time.time()

    gen_ids = output[0][len(input_ids[0]):]
    n_tokens = len(gen_ids)
    walltime = t1 - t0

    print(f"  Generated {n_tokens} tokens")
    print(f"  Wall time: {walltime:.3f}s")
    print(f"  Tokens/sec: {n_tokens/walltime:.1f}")

    del model
    torch.cuda.empty_cache()

    return n_tokens/walltime


if __name__ == "__main__":
    prompt = "The capital of France is"

    # First run pure AR baseline
    ar_speed = test_pure_ar(prompt, max_new=128)

    # Test different EAGLE configs
    configs = [
        # (name, total_token, depth, top_k)
        ("Default", 60, 5, 10),
        ("Deeper tree", 60, 7, 10),
        ("Larger tree", 100, 5, 10),
        ("Wider beam", 60, 5, 12),
    ]

    results = []
    for name, tt, d, tk in configs:
        try:
            speed = test_config(name, tt, d, tk, prompt, max_new=128)
            results.append((name, speed))
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n{'='*60}")
    print("SUMMARY:")
    print(f"  Pure AR: {ar_speed:.1f} tok/s")
    for name, speed in results:
        print(f"  EAGLE-3 ({name}): {speed:.1f} tok/s (speedup: {speed/ar_speed:.2f}x)")
    print("=" * 60)
