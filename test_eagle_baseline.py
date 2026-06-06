"""
Minimal test script: Load EAGLE-3 + LLaMA-3-8B and run a single inference.
Verifies that the baseline pipeline works end-to-end.
"""
import torch
import sys
import os
import time

# Add EAGLE to path
sys.path.insert(0, '/home/hzliu/AD/Homework_haozhe/MLsys_final/EAGLE')

from eagle.model.ea_model import EaModel
from eagle.model.utils import prepare_logits_processor

# Configuration
BASE_MODEL_PATH = "/home/hzliu/AD/Homework_haozhe/MLsys_final/models/Llama-3.1-8B-Instruct"
EA_MODEL_PATH = "/home/hzliu/AD/Homework_haozhe/MLsys_final/EAGLE_checkpoints/EAGLE3-LLaMA3.1-Instruct-8B"

# EAGLE tree parameters
TOTAL_TOKEN = 60   # total draft tokens in tree + 1
DEPTH = 5          # max draft length - 1
TOP_K = 10         # beam size at each step

def main():
    print("=" * 60)
    print("Loading EAGLE-3 model...")
    print(f"  Base model: {BASE_MODEL_PATH}")
    print(f"  EAGLE checkpoint: {EA_MODEL_PATH}")
    print(f"  Tree params: total_token={TOTAL_TOKEN}, depth={DEPTH}, top_k={TOP_K}")
    print("=" * 60)

    # Load model
    t0 = time.time()
    model = EaModel.from_pretrained(
        base_model_path=BASE_MODEL_PATH,
        ea_model_path=EA_MODEL_PATH,
        total_token=TOTAL_TOKEN,
        depth=DEPTH,
        top_k=TOP_K,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_eagle3=True,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")
    model.eval()

    tokenizer = model.get_tokenizer()
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")

    # Prepare a simple test prompt
    prompt = "The capital of France is"
    input_ids = tokenizer([prompt], add_special_tokens=True).input_ids
    print(f"Prompt: '{prompt}'")
    print(f"Input token count: {len(input_ids[0])}")

    # Warmup runs
    print("\nWarming up (3 runs)...")
    for i in range(3):
        torch.cuda.synchronize()
        output_ids = model.eagenerate(
            torch.as_tensor(input_ids).cuda(),
            temperature=0.0,
            log=False,
            is_llama3=True,
        )
        torch.cuda.synchronize()

    # Timed run
    print("\nRunning timed inference...")
    torch.cuda.synchronize()
    t_start = time.time()

    output_ids, new_token, step_count = model.eagenerate(
        torch.as_tensor(input_ids).cuda(),
        temperature=0.0,
        log=True,
        is_llama3=True,
    )

    torch.cuda.synchronize()
    wall_time = time.time() - t_start

    # Decode output
    generated_ids = output_ids[0][len(input_ids[0]):]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    new_tokens_count = len(generated_ids)

    # Results
    print("=" * 60)
    print("RESULTS:")
    print(f"  Generated text: '{output_text[:200]}...'" if len(output_text) > 200 else f"  Generated text: '{output_text}'")
    print(f"  New tokens: {new_tokens_count}")
    print(f"  EAGLE steps: {step_count}")
    print(f"  Wall time: {wall_time:.3f}s")
    print(f"  Tokens/sec: {new_tokens_count / wall_time:.1f}")
    print("=" * 60)
    print("\n✓ Baseline inference pipeline works!")

if __name__ == "__main__":
    main()
