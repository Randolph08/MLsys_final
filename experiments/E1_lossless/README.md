# E1.4 Greedy Lossless / Token Sequence Check

## Purpose

This experiment verifies whether `EaModel.naivegenerate()` and `EaModel.eagenerate()` produce identical greedy token sequences under the official-style evaluation setup:

- same `EaModel` instance
- same LLaMA3 chat template
- same input ids for both methods at every turn
- `temperature=0.0`
- same stop-token trimming
- strict comparison after trimming to `max_new_tokens`

For multi-turn MT-Bench records, both methods use the same input history. If a previous turn mismatches, the next turn still uses the naive answer as the shared assistant history, so later mismatches are not caused by divergent conversation histories.

## Code Changes

The `max_new_tokens` boundary in `EAGLE/eagle/model/ea_model.py` was changed from:

```python
if new_token > max_new_tokens:
    break
```

to:

```python
if new_token >= max_new_tokens:
    break
```

This removes the simple off-by-one issue in sequential generation. EAGLE can still return more raw tokens than `max_new_tokens` because one speculative step may accept multiple tokens before the loop condition is checked; the verification script therefore compares strictly trimmed token sequences.

## Script

```text
experiments/verify_lossless_eamodel.py
```

The script records:

- raw and trimmed token ids
- raw length and trimmed length
- first different token position
- decoded text around the first difference
- an optional full-prefix base-model argmax diagnosis at the first difference

## Commands

```bash
.venv/bin/python experiments/verify_lossless_eamodel.py \
  --prompt-source toy \
  --limit 5 \
  --max-new-tokens 128 \
  --warmup 1 \
  --force \
  --cuda-visible-devices 0,1,2,3,4
```

```bash
.venv/bin/python experiments/verify_lossless_eamodel.py \
  --prompt-source mt_bench \
  --limit 20 \
  --max-new-tokens 128 \
  --warmup 1 \
  --force \
  --cuda-visible-devices 0,1,2,3,4
```

## Results

| Dataset | Records | Turns | Record Match | Turn Match | Naive tok/s | EAGLE tok/s | Naive Raw Overshoot | EAGLE Raw Overshoot |
|---------|---------|-------|--------------|------------|-------------|-------------|---------------------|--------------------|
| toy | 5 | 5 | 5/5 | 5/5 | 20.35 | 80.40 | 0 | 3 |
| MT-Bench | 5 | 10 | 3/5 | 8/10 | 21.29 | 82.59 | 0 | 5 |
| MT-Bench | 20 | 40 | 13/20 | 33/40 | 21.17 | 76.36 | 0 | 27 |

Saved outputs:

- `experiments/E1_lossless/lossless_toy_limit-5_max-128.json`
- `experiments/E1_lossless/lossless_mt_bench_limit-5_max-128.json`
- `experiments/E1_lossless/lossless_mt_bench_limit-20_max-128.json`

## Numerical Precision Diagnosis

To check whether the MT-Bench mismatches are caused by fp16 numerical path differences, we reran the same MT-Bench first-5 slice under two diagnostic settings:

| Setting | Record Match | Turn Match | Naive tok/s | EAGLE tok/s | Mismatch Locations |
|---------|--------------|------------|-------------|-------------|--------------------|
| fp16 default | 3/5 | 8/10 | 21.29 | 82.59 | q82 turn1, q84 turn0 |
| fp16 + deterministic + no TF32 | 3/5 | 8/10 | 12.50 | 50.53 | q82 turn1, q84 turn0 |
| fp32 + deterministic + no TF32 | 5/5 | 10/10 | 13.26 | 47.99 | none |

Saved diagnostic outputs:

- `experiments/E1_lossless/lossless_mt_bench_limit-5_max-128_fp16_det_no_tf32.json`
- `experiments/E1_lossless/lossless_mt_bench_limit-5_max-128_fp32_det_no_tf32.json`

This confirms that deterministic settings alone do not remove the mismatches. The same mismatch locations remain in fp16, while fp32 removes them completely on this slice. The most likely cause is therefore fp16 numerical differences between sequential KV-cache decoding and tree/full-prefix verification, amplified by near-tie logits.

## MT-Bench 20 Mismatch Summary

| Question | Category | Turn | First Diff | Naive Token | EAGLE Token | Full-Prefix Argmax |
|----------|----------|------|------------|-------------|-------------|--------------------|
| 82 | writing | 1 | 6 | ` here` | ` I` | EAGLE |
| 84 | writing | 0 | 49 | `  ` | ` **` | EAGLE |
| 89 | writing | 1 | 1 | `'s` | ` are` | Naive |
| 91 | roleplay | 0 | 8 | ` question` | ` no` | Naive |
| 92 | roleplay | 0 | 97 | `:` | `.` | Naive |
| 96 | roleplay | 0 | 63 | ` create` | ` converse` | EAGLE |
| 98 | roleplay | 1 | 68 | ` sophisticated` | ` sophistication` | EAGLE |

Full-prefix base-model diagnosis supports EAGLE in 4/7 mismatch turns and naive in 3/7 mismatch turns. In all seven cases, the two competing tokens are top-2 candidates under the full-prefix forward pass, with very small logit differences. This strongly suggests the remaining greedy mismatches come from fp16 numerical differences between sequential KV-cache decoding and tree/full-prefix verification, rather than from prompt formatting, stop-token handling, or a simple `max_new_tokens` bug.

## Interpretation

E1.4 resolves the major evaluation-format issues:

1. The prompt format is now official-style chat template.
2. The speed baseline uses `EaModel.naivegenerate()`.
3. The simple `max_new_tokens` off-by-one issue is fixed.
4. Multi-turn comparisons now use identical input ids for both methods.

The remaining exact-token mismatches are real but localized. They occur when greedy logits are very close, so tiny fp16 path differences can flip the argmax. For the final report, greedy exact-match should be reported as an empirical correctness check rather than assumed to be 100% in this fp16 RTX 3090 setting.

The fp32 diagnostic slice shows that the ideal 100% greedy exact match can be recovered for the previously mismatching MT-Bench first-5 slice, but at a large performance cost. Therefore, fp32 should be used only as a correctness diagnosis, while the main speed experiments should remain fp16 and report turn-level match rate.

For subsequent DDD and OPT-Tree experiments, use the same verification script and report:

- turn-level match rate
- mismatch locations
- speedup after strict trimming
- raw overshoot counts
