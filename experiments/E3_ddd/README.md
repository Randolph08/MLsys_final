# E3.1 DDD H Distribution Profiling

## Purpose

This experiment profiles the DDD confidence signal

```text
H = logsumexp(logprobsum)
```

at DDD check steps without triggering early stop. The goal is to choose a meaningful threshold search range before running DDD ablations.

## Code Changes

`EAGLE/eagle/model/cnets.py` now keeps richer DDD stats while preserving legacy fields:

- `calls`
- `early_stops`
- `total_checks`
- `depths`
- `actual_depths`
- `early_stop_steps`
- `checked_H`
- `call_records`

For E3.1, the model is loaded with `use_ddd=True`, `ddd_max_depth=11`, `ddd_check_steps=[5,7,9]`, and `ddd_threshold=-1e9`, so DDD records H but should never early-stop.

## Script

```text
experiments/profile_ddd_h.py
```

## Commands

Toy sanity:

```bash
.venv/bin/python experiments/profile_ddd_h.py \
  --prompt-source toy \
  --limit 2 \
  --max-new-tokens 64 \
  --warmup 0 \
  --ddd-max-depth 11 \
  --ddd-check-steps 5,7,9 \
  --force \
  --cuda-visible-devices 0,1,2,3,4
```

MT-Bench slice:

```bash
.venv/bin/python experiments/profile_ddd_h.py \
  --prompt-source mt_bench \
  --limit 5 \
  --max-new-tokens 128 \
  --warmup 1 \
  --ddd-max-depth 11 \
  --ddd-check-steps 5,7,9 \
  --force \
  --cuda-visible-devices 0,1,2,3,4
```

## Saved Outputs

- `experiments/E3_ddd/ddd_h_toy_limit-2_max-64_depth-11_steps-5-7-9.json`
- `experiments/E3_ddd/ddd_h_mt_bench_limit-5_max-128_depth-11_steps-5-7-9.json`

## MT-Bench Results

| Metric | Value |
|--------|-------|
| records | 5 |
| turns | 10 |
| DDD calls | 205 |
| H checks | 615 |
| early stops | 0 |
| actual depth | 12.0 for all calls |

### H Distribution

| Step | Count | Mean | P10 | P25 | P50 | P75 | P90 |
|------|-------|------|-----|-----|-----|-----|-----|
| 5 | 205 | -1.384 | -2.870 | -1.872 | -1.124 | -0.598 | -0.212 |
| 7 | 205 | -2.110 | -3.655 | -2.789 | -1.969 | -1.205 | -0.524 |
| 9 | 205 | -3.234 | -5.306 | -4.164 | -3.121 | -2.020 | -1.201 |

The H value decreases clearly as the check step gets deeper, which supports DDD's basic assumption: deeper beam expansions carry less total confidence mass.

### Threshold Trigger Fractions

Fraction of checks where `H < threshold`:

| Threshold | Global | Step 5 | Step 7 | Step 9 |
|-----------|--------|--------|--------|--------|
| -8.0 | 0.33% | 0.00% | 0.00% | 0.98% |
| -6.0 | 2.76% | 0.00% | 0.98% | 7.32% |
| -5.0 | 6.83% | 0.00% | 6.34% | 14.15% |
| -4.0 | 13.17% | 2.93% | 8.78% | 27.80% |
| -3.0 | 27.64% | 9.76% | 20.98% | 52.20% |
| -2.0 | 48.13% | 20.49% | 48.29% | 75.61% |
| -1.0 | 75.45% | 56.59% | 77.56% | 92.20% |
| -0.5 | 89.59% | 79.51% | 90.73% | 98.54% |
| -0.3 | 93.17% | 86.83% | 93.66% | 99.02% |

## Interpretation

The old grid `[-4, -6, -8, -10, -12]` is mostly too conservative for this implementation. Values below `-6` almost never trigger early stop, especially at step 5/7. Conversely, the paper-like `-0.3` is extremely aggressive under this code's H definition and would stop at step 5 in most calls.

Recommended DDD threshold sweep:

```text
[-5.0, -4.0, -3.0, -2.0, -1.0, -0.5, -0.3]
```

For the first DDD ablation, use paper-like check steps `[5,7,9]` and `ddd_max_depth=11`, then report:

- tok/s
- accept/step
- avg actual depth
- early stop rate
- early stop step histogram
- turn-level lossless match rate

---

# E3.2 Fixed-Depth Sweep

## Purpose

This experiment measures the trade-off between deeper fixed draft expansion and end-to-end throughput before enabling DDD. It answers whether a deeper tree actually improves accepted tokens per EAGLE step, and whether that improvement pays for the extra drafting cost.

## Script

```text
experiments/run_fixed_depth_sweep.py
```

## Command

```bash
.venv/bin/python experiments/run_fixed_depth_sweep.py \
  --prompt-source mt_bench \
  --limit 5 \
  --max-new-tokens 128 \
  --warmup 1 \
  --depths 5,7,9,11 \
  --force \
  --cuda-visible-devices 0,1,2,3,4
```

## Saved Output

- `experiments/E3_ddd/fixed_depth_mt_bench_limit-5_max-128_depths-5-7-9-11.json`

## Results

| Fixed Depth | tok/s | Speed vs Depth 5 | Accept / Step | Mean Loop Count | Total Wall Time |
|-------------|-------|------------------|---------------|-----------------|-----------------|
| 5 | 83.51 | 1.000x | 5.286 | 22.20 | 14.07s |
| 7 | 83.99 | 1.006x | 5.803 | 20.30 | 14.04s |
| 9 | 82.06 | 0.983x | 6.017 | 19.60 | 14.38s |
| 11 | 77.81 | 0.932x | 6.051 | 19.50 | 15.18s |

## Interpretation

Increasing fixed depth improves `accept/step`, but the gains saturate quickly:

- depth 5 to 7 improves accept/step from 5.286 to 5.803 with almost no throughput loss.
- depth 9 improves accept/step further to 6.017, but throughput begins to drop.
- depth 11 barely improves accept/step over depth 9, while throughput falls to 0.932x of depth 5.

This is exactly the trade-off DDD is supposed to exploit: deeper expansion can help on some steps, but a fixed depth of 11 pays the drafting cost on every step. The next experiment should run DDD with `ddd_max_depth=11`, check steps `[5,7,9]`, and thresholds from the E3.1 grid so it can keep the useful deeper exploration while early-stopping low-confidence steps.

---

# E3.3/E3.4 DDD Threshold Sweep

## Purpose

This experiment runs paper-like DDD with `ddd_max_depth=11` and `check_steps=[5,7,9]`, using the threshold range selected from E3.1. It measures both end-to-end throughput and whether DDD actually reduces draft depth.

## Script

```text
experiments/run_ddd_sweep.py
```

## Command

```bash
.venv/bin/python experiments/run_ddd_sweep.py \
  --prompt-source mt_bench \
  --limit 5 \
  --max-new-tokens 128 \
  --warmup 1 \
  --ddd-max-depth 11 \
  --ddd-check-steps 5,7,9 \
  --thresholds=-5,-4,-3,-2,-1,-0.5,-0.3 \
  --force \
  --cuda-visible-devices 0,1,2,3,4
```

## Saved Output

- `experiments/E3_ddd/ddd_sweep_mt_bench_limit-5_max-128_depth-11_tau-m5p0-m4p0-m3p0-m2p0-m1p0-m0p5-m0p3.json`

## Results

| Threshold | tok/s | Speed vs tau=-5 | Accept / Step | Avg Actual Depth | Early Stop Rate | Early Stop Step Hist |
|-----------|-------|-----------------|---------------|------------------|-----------------|----------------------|
| -5.0 | 78.85 | 1.000x | 6.051 | 11.590 | 14.15% | 7:13, 9:16 |
| -4.0 | 79.23 | 1.005x | 6.051 | 11.210 | 27.80% | 5:6, 7:12, 9:39 |
| -3.0 | 81.10 | 1.028x | 6.051 | 10.322 | 53.17% | 5:20, 7:23, 9:66 |
| -2.0 | 84.09 | 1.067x | 6.035 | 9.087 | 75.24% | 5:44, 7:57, 9:54 |
| -1.0 | 86.52 | 1.097x | 5.884 | 7.526 | 91.00% | 5:118, 7:44, 9:30 |
| -0.5 | 88.13 | 1.118x | 5.714 | 6.664 | 97.23% | 5:172, 7:24, 9:15 |
| -0.3 | 87.84 | 1.114x | 5.617 | 6.400 | 98.64% | 5:194, 7:11, 9:12 |

## Interpretation

There are two useful operating points:

- `tau=-2.0` is the conservative trade-off. It keeps almost the same acceptance as fixed depth 11 (`6.035` vs `6.051` accept/step), reduces average actual depth from 12 to about 9.09, and reaches `84.09 tok/s`.
- `tau=-0.5` is the speed-oriented point. It reaches the best throughput (`88.13 tok/s`), but early-stops in 97.23% of calls and loses acceptance (`5.714` accept/step).

Compared with the fixed-depth baselines:

| Method | tok/s | Accept / Step | Note |
|--------|-------|---------------|------|
| Fixed depth 5 | 83.51 | 5.286 | default shallow baseline |
| Fixed depth 7 | 83.99 | 5.803 | best fixed-depth speed in E3.2 |
| Fixed depth 11 | 77.81 | 6.051 | highest fixed-depth acceptance, slow |
| DDD tau=-2.0 | 84.09 | 6.035 | keeps depth-11 acceptance with depth-7-like speed |
| DDD tau=-0.5 | 88.13 | 5.714 | fastest, but more aggressive |

For the final DDD-only row, `tau=-2.0` is the safest paper-style operating point, while `tau=-0.5` can be reported as the speed-optimized variant.

---

# E3.5 DDD Candidate Validation

## Token-Level Correctness

The selected DDD candidates were checked with the same greedy lossless script used in E1.4. Both runs use MT-Bench first 5 questions, `max_new_tokens=128`, fp16, chat-template prompts, and shared naive history.

| Method | Record Match | Turn Match | Mismatches | Naive tok/s | EAGLE/DDD tok/s |
|--------|--------------|------------|------------|-------------|-----------------|
| DDD `tau=-2.0` | 4/5 | 9/10 | q84 turn0 pos49 | 21.28 | 83.78 |
| DDD `tau=-0.5` | 3/5 | 8/10 | q82 turn1 pos6, q84 turn0 pos49 | 21.14 | 85.26 |

`tau=-2.0` has slightly better token-level agreement than the more aggressive `tau=-0.5`. The remaining mismatches overlap with the fp16 numerical divergences found in E1.4/E1.5, so they do not by themselves indicate a new DDD-specific logic error.

Saved outputs:

- `experiments/E1_lossless/lossless_mt_bench_limit-5_max-128_ddd_tau-m2.json`
- `experiments/E1_lossless/lossless_mt_bench_limit-5_max-128_ddd_tau-m0p5.json`

## Larger MT-Bench Slice

To reduce small-sample noise, fixed-depth baselines and the two DDD candidates were rerun on MT-Bench first 20 questions, i.e. 40 turns.

Fixed-depth baselines:

| Method | tok/s | Speed vs Depth 5 | Accept / Step | Mean Loop Count |
|--------|-------|------------------|---------------|-----------------|
| Fixed depth 5 | 76.05 | 1.000x | 4.872 | 25.32 |
| Fixed depth 7 | 75.16 | 0.988x | 5.187 | 24.12 |
| Fixed depth 9 | 73.66 | 0.969x | 5.349 | 23.50 |
| Fixed depth 11 | 70.19 | 0.923x | 5.372 | 23.45 |

DDD candidates:

| Method | tok/s | Accept / Step | Avg Actual Depth | Early Stop Rate | Early Stop Step Hist |
|--------|-------|---------------|------------------|-----------------|----------------------|
| DDD `tau=-2.0` | 76.35 | 5.357 | 8.226 | 85.52% | 5:380, 7:252, 9:207 |
| DDD `tau=-0.5` | 78.60 | 5.165 | 6.421 | 97.93% | 5:881, 7:72, 9:39 |

Saved outputs:

- `experiments/E3_ddd/fixed_depth_mt_bench_limit-20_max-128_depths-5-7-9-11.json`
- `experiments/E3_ddd/ddd_sweep_mt_bench_limit-20_max-128_depth-11_tau-m2-m0p5.json`

## Interpretation

On the 20-question MT-Bench slice, the fixed-depth trend remains clear: deeper fixed draft trees improve acceptance but reduce throughput. DDD `tau=-2.0` keeps almost the same acceptance as fixed depth 11 (`5.357` vs `5.372` accept/step) while recovering throughput to depth-5 level (`76.35` vs `76.05 tok/s`). This is a small but coherent DDD gain: the mechanism works, but end-to-end speedup is bounded because target verification still dominates runtime.

DDD `tau=-0.5` is faster (`78.60 tok/s`) but gives up more acceptance. It is useful as a speed-oriented variant, while `tau=-2.0` should remain the safer paper-style configuration for the final DDD row.

---

# Full MT-Bench Results

After the small-slice diagnosis, the selected fixed-depth and DDD configurations were rerun on the full MT-Bench set used by this project: 80 questions, 160 turns, `max_new_tokens=128`.

## Saved Outputs

- `experiments/E3_ddd/fixed_depth_mt_bench_limit-80_max-128_depths-5-11.json`
- `experiments/E3_ddd/ddd_sweep_mt_bench_limit-80_max-128_depth-11_tau-m2p0-m0p5.json`

## Fixed-Depth Baselines

| Method | tok/s | Speed vs Depth 5 | Accept / Step | Mean Loop Count |
|--------|-------|------------------|---------------|-----------------|
| Fixed depth 5 | 65.23 | 1.000x | 5.341 | 22.41 |
| Fixed depth 11 | 77.03 | 1.181x | 6.392 | 19.46 |

Unlike the 20-question slice, full MT-Bench favors deeper fixed expansion: depth 11 improves both acceptance and throughput.

## DDD Candidates

| Method | tok/s | Accept / Step | Avg Actual Depth | Early Stop Rate | Early Stop Step Hist |
|--------|-------|---------------|------------------|-----------------|----------------------|
| DDD `tau=-2.0` | 68.51 | 6.365 | 9.121 | 71.05% | 5:820, 7:754, 9:760 |
| DDD `tau=-0.5` | 70.74 | 6.041 | 6.991 | 93.18% | 5:2505, 7:383, 9:307 |

## Full-Set Interpretation

DDD works mechanistically: it reduces average actual draft depth from the fixed depth-11 maximum to `9.121` for `tau=-2.0` and `6.991` for `tau=-0.5`, with high early-stop rates. However, on full MT-Bench this does not translate into an end-to-end speedup over fixed depth 11.

The best full-set throughput among these rows is fixed depth 11 (`77.03 tok/s`). DDD `tau=-2.0` preserves almost the same acceptance (`6.365` vs `6.392`) but drops to `68.51 tok/s`. DDD `tau=-0.5` is faster than `tau=-2.0` but sacrifices acceptance and still remains below fixed depth 11.

Final DDD reporting should therefore be cautious:

- Small slices suggested DDD could recover depth-11 acceptance at depth-5-like speed.
- Full MT-Bench shows that fixed depth 11 is already a strong setting.
- DDD still provides useful mechanism evidence, but not a positive full-set throughput result under the current implementation and hardware.
