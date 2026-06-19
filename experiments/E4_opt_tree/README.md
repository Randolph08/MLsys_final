# E4.1 Baseline vs OPT Tree Set Diff

## Purpose

This experiment checks whether OPT-Tree actually changes the draft tree selected by the current EAGLE-3 dynamic tree baseline.

The key question is not throughput yet. The first question is structural:

```text
Does OPT-Tree select a different node set from the original EAGLE top-score rerank?
```

If the selected node sets are identical, OPT-Tree cannot improve acceptance length on this baseline, and later throughput experiments should be interpreted as overhead measurements.

## Instrumentation

`EAGLE/eagle/model/cnets.py` now records tree-selection diagnostics only when the experiment script explicitly creates `model.ea_layer._tree_diff_records`.

For each draft-tree construction call, it records:

- baseline selected candidate indices
- OPT over-expanded candidate indices
- OPT final selected candidate indices
- baseline and OPT depth histograms
- Jaccard similarity between baseline and OPT final selected sets
- baseline-only and OPT-only node counts

Normal EAGLE/DDD runs are unaffected because `_tree_diff_records` is absent by default.

## Script

```text
experiments/analyze_tree_selection.py
```

## Commands

Smoke test:

```bash
.venv/bin/python experiments/analyze_tree_selection.py \
  --prompt-source mt_bench \
  --limit 2 \
  --max-new-tokens 64 \
  --warmup 1 \
  --opt-expand-factors 1.5,2.0 \
  --cuda-visible-devices 0,1,2,3,4 \
  --force
```

Larger small slice:

```bash
.venv/bin/python experiments/analyze_tree_selection.py \
  --prompt-source mt_bench \
  --limit 5 \
  --max-new-tokens 128 \
  --warmup 1 \
  --opt-expand-factors 1.5 \
  --cuda-visible-devices 0,1,2,3,4 \
  --force
```

## Saved Outputs

- `experiments/E4_opt_tree/tree_diff_mt_bench_limit-2_max-64_expand-1p5-2p0.json`
- `experiments/E4_opt_tree/tree_diff_mt_bench_limit-2_max-64_expand-4p0-8p0.json`
- `experiments/E4_opt_tree/tree_diff_mt_bench_limit-5_max-128_expand-1p5.json`

## Results

| Run | Turns | Tree Calls | Expand | Mean Jaccard | Identical Rate | Mean Baseline-only | Mean OPT-only | Mean Accept / Step |
|-----|-------|------------|--------|--------------|----------------|--------------------|---------------|--------------------|
| MT-Bench 2, max64 | 4 | 50 | 1.5 | 1.000 | 100.00% | 0.00 | 0.00 | 5.619 |
| MT-Bench 2, max64 | 4 | 50 | 2.0 | 1.000 | 100.00% | 0.00 | 0.00 | 5.619 |
| MT-Bench 2, max64 | 4 | 50 | 4.0 | 1.000 | 100.00% | 0.00 | 0.00 | 5.619 |
| MT-Bench 2, max64 | 4 | 50 | 8.0 | 1.000 | 100.00% | 0.00 | 0.00 | 5.619 |
| MT-Bench 5, max128 | 10 | 232 | 1.5 | 1.000 | 100.00% | 0.00 | 0.00 | 5.286 |

For the MT-Bench 5 run, the aggregate depth histograms are also identical:

| Depth | Baseline Nodes | OPT Nodes |
|-------|----------------|-----------|
| 1 | 1022 | 1022 |
| 2 | 2055 | 2055 |
| 3 | 2586 | 2586 |
| 4 | 2675 | 2675 |
| 5 | 2759 | 2759 |
| 6 | 2591 | 2591 |

## Interpretation

On all tested calls, OPT-Tree selects exactly the same final candidate set as the original EAGLE-3 dynamic tree selection. This remains true even when `opt_expand_factor` is increased to `4.0` and `8.0` on the MT-Bench 2 smoke run, so the result is not explained simply by the original `1.5/2.0` expand factors being too conservative. This strongly supports the negative-result hypothesis:

```text
OPT-Tree does not create a new tree on top of the current EAGLE-3 dynamic baseline.
```

Therefore, in this codebase, OPT-Tree is unlikely to improve acceptance length on the current EAGLE-3 dynamic baseline.

The next OPT experiment should be E4.2, but its purpose is now narrower: confirm that OPT-only throughput is not better once the tree set is known to be identical, and separate algorithmic gain from implementation overhead.

---

# E4.2 OPT-Only Dynamic Baseline

## Purpose

This experiment measures normal EAGLE-3 vs EAGLE-3 + OPT throughput without tree-diff instrumentation. Its main purpose is to confirm whether OPT changes `accept/step` or loop count under the corrected chat-template protocol.

## Script

```text
experiments/run_opt_dynamic_baseline.py
```

## Saved Outputs

- `experiments/E4_opt_tree/opt_dynamic_mt_bench_limit-5_max-128_expand-1p5-2p0.json`
- `experiments/E4_opt_tree/opt_dynamic_mt_bench_limit-5_max-128_expand-1p5-2p0_single_repeat.json`
- `experiments/E4_opt_tree/opt_dynamic_mt_bench_limit-80_max-128_expand-1p5_single_repeat.json`

## Results

Separate model loads:

| Method | tok/s | Speed vs EAGLE-3 | Accept / Step | Mean Loop Count |
|--------|-------|------------------|---------------|-----------------|
| EAGLE-3 | 52.44 | 1.000x | 5.286 | 22.20 |
| OPT-1.5 | 66.67 | 1.272x | 5.286 | 22.20 |
| OPT-2.0 | 63.90 | 1.219x | 5.286 | 22.20 |

Single shared model with EAGLE-3 repeat:

| Method | tok/s | Speed vs First EAGLE-3 | Accept / Step | Mean Loop Count |
|--------|-------|------------------------|---------------|-----------------|
| EAGLE-3 | 52.20 | 1.000x | 5.286 | 22.20 |
| OPT-1.5 | 50.14 | 0.961x | 5.286 | 22.20 |
| OPT-2.0 | 58.84 | 1.127x | 5.286 | 22.20 |
| EAGLE-3-repeat | 78.19 | 1.498x | 5.286 | 22.20 |

Full MT-Bench 80 records / 160 turns, single shared model with EAGLE-3 repeat:

| Method | tok/s | Speed vs First EAGLE-3 | Accept / Step | Mean Loop Count |
|--------|-------|------------------------|---------------|-----------------|
| EAGLE-3 | 61.66 | 1.000x | 5.341 | 22.41 |
| OPT-1.5 | 47.18 | 0.765x | 5.341 | 22.41 |
| EAGLE-3-repeat | 80.50 | 1.306x | 5.341 | 22.41 |

## Interpretation

The stable signal is not throughput; it is that all methods have exactly the same acceptance behavior:

```text
accept/step = 5.286
mean loop count = 22.20
total trimmed tokens = 1191
total raw tokens = 1211
```

This matches E4.1: OPT selects the same final tree, so it cannot change acceptance length.

The small-slice tok/s numbers are not reliable enough to claim an OPT speedup or slowdown. In the single-model run, the repeated EAGLE-3 baseline at the end reaches `78.19 tok/s`, while the first EAGLE-3 pass is only `52.20 tok/s`. That drift is larger than the OPT-vs-baseline differences, so these E4.2 speed numbers should be treated as warmup/order-sensitive diagnostics rather than final performance claims.

The full MT-Bench run gives the same algorithmic conclusion: `accept/step=5.341` and `mean loop count=22.41` for all three rows. OPT-1.5 is slower than the first EAGLE-3 pass in this run, while the repeated EAGLE-3 pass is much faster than the first pass. Therefore the robust conclusion remains "no acceptance/tree-selection gain"; throughput should be discussed as implementation/order-sensitive overhead rather than a clean algorithmic comparison.

Final OPT reporting should therefore emphasize:

- no tree-set difference;
- no accept/step difference;
- no loop-count difference;
- no demonstrated algorithmic gain on EAGLE-3 dynamic rerank.

## Negative-Result Explanation for the Final Report

If the final OPT-vs-EAGLE-3 throughput comparison remains poor, the result should be explained as a baseline-mismatch issue rather than simply as an implementation failure.

The current EAGLE-3 baseline is already a strong dynamic-tree baseline. It expands a candidate pool, scores nodes by cumulative path log-probability, and globally keeps the highest-scoring nodes. Because cumulative log-probability is monotonic along a path,

```text
score(child) = score(parent) + log p(child | parent)
log p(child | parent) <= 0
```

a high-scoring child almost always implies that its ancestors have even higher scores and are already selected by the baseline. The selected top-score set is therefore naturally ancestor-closed in practice. OPT-Tree also optimizes high-probability paths under an ancestor-closure constraint, so on this baseline it collapses to the same final node set.

This is exactly what E4.1 observes: all tested calls have `mean Jaccard = 1.000`, `identical rate = 100%`, and zero baseline-only or OPT-only nodes. Thus, if OPT has worse tok/s later, the most plausible explanation is:

```text
OPT-Tree has no algorithmic tree-selection gain on EAGLE-3 dynamic rerank,
while its extra over-expansion / selection logic adds implementation overhead.
```

This is a valid negative result. It means OPT-Tree is not orthogonal to EAGLE-3 dynamic tree selection.

## Baseline Decision

We decided not to replace EAGLE-3 with EAGLE-1/EAGLE-2 and not to add a weaker comparison baseline for OPT-Tree in the remaining experiments. The final report will keep EAGLE-3 as the main baseline and explain OPT-Tree as a negative result on this strong dynamic-tree baseline.
