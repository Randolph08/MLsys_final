# E6 Optimization Experiments

本文档记录在现有 DDD / OPT-Tree 复现实验基础上，对若干优化思路进行的补充验证。核心目标是区分：

- 哪些配置能带来可复现的性能提升；
- 哪些方向看起来合理，但在当前 EAGLE-3 框架下没有收益；
- 后续最终报告中应该采用哪组结果作为优化后方法。

## 1. 实验口径

除特别说明外，实验口径与 `final_experiment_results.md` 保持一致：

| 项目 | 设置 |
|------|------|
| Base model | `models/Llama-3.1-8B-Instruct` |
| EAGLE checkpoint | `EAGLE_checkpoints/LLAMA3-Instruct-8B/` |
| 主数据集 | MT-Bench 本地 80 records / 160 turns |
| 生成长度 | `max_new_tokens=128` |
| dtype | fp16 |
| GPU | `CUDA_VISIBLE_DEVICES=0,1,2,3,4` |
| 主要指标 | macro tok/s、global tok/s、accept/step、mean loop、DDD avg depth、early-stop rate、tree Jaccard |

其中 macro tok/s 是各 turn 的 tok/s 平均值，global tok/s 是 `total_trimmed_tokens / total_wall_time_s`。

## 2. 有效优化：DDD 阈值调参

此前全量 DDD 只验证了 `tau=-2.0` 和 `tau=-0.5`。这次补充了中间阈值 `tau=-1.0` 和更激进阈值 `tau=-0.3`。

| Method | macro tok/s | global tok/s | accept/step | mean loop | avg depth | early stop |
|--------|-------------|--------------|-------------|-----------|-----------|------------|
| fixed depth 11 | 77.03 | 72.84 | 6.392 | 19.46 | fixed 11 | - |
| DDD tau=-2.0 | 68.51 | 65.43 | 6.365 | 19.53 | 9.121 | 71.05% |
| DDD tau=-0.5 | 70.74 | 67.39 | 6.041 | 20.43 | 6.991 | 93.18% |
| DDD tau=-1.0 | 88.27 | 86.86 | 6.252 | 19.81 | 7.695 | 86.60% |
| DDD tau=-0.3 | 65.99 | 63.24 | 5.907 | 20.84 | 6.714 | 94.99% |

关键结论：

- `tau=-1.0` 是当前最明显的性能提升点。
- 相比此前最佳 DDD `tau=-0.5`，`tau=-1.0` 的 macro tok/s 从 `70.74` 提升到 `88.27`，提升约 `24.8%`。
- 相比 fixed depth 11，`tau=-1.0` 的 macro tok/s 提升约 `14.6%`，global tok/s 提升约 `19.3%`。
- `tau=-0.3` 过于激进，虽然 avg depth 更低，但 accept/step 下降、loop 增加，最终速度变差。

解释：

DDD 的收益不是简单来自“越早停越好”。`tau=-0.5` / `tau=-0.3` 停得更浅，但候选质量下降，导致每轮接受 token 变少。`tau=-1.0` 保留了足够的 draft 深度，同时避免多数无必要的深层 draft，因此在当前设置下形成了更好的平衡。

推荐配置：

```bash
.venv/bin/python experiments/run_ddd_sweep.py \
  --prompt-source mt_bench \
  --limit 80 \
  --max-new-tokens 128 \
  --warmup 1 \
  --thresholds=-1.0 \
  --ddd-max-depth 11 \
  --ddd-check-steps 5,7,9 \
  --cuda-visible-devices 0,1,2,3,4
```

## 3. 无效方向：固定深度缩短

为了确认 DDD 的提升是否只是来自更浅的固定深度，补充了 fixed depth 9，并在一次被中断的全量运行中观察到 fixed depth 7 的完整结果。

| Method | macro tok/s | global tok/s | accept/step | mean loop | 说明 |
|--------|-------------|--------------|-------------|-----------|------|
| fixed depth 5 | 65.23 | 63.42 | 5.341 | 22.41 | 已有 E3 full |
| fixed depth 7 | 60.98 | - | 5.926 | 20.49 | 控制台完整输出，进程在后续 depth9 被手动中断，未落 JSON |
| fixed depth 9 | 60.47 | 57.39 | 6.270 | 19.68 | E6 full |
| fixed depth 11 | 77.03 | 72.84 | 6.392 | 19.46 | 已有 E3 full |

结论：

- 简单缩短 fixed depth 不是有效优化。
- depth 9 的 accept/step 接近 depth 11，但 per-step 成本/整体 wall time 表现更差。
- DDD `tau=-1.0` 的价值在于按位置动态停层，而不是固定使用更浅深度。

## 4. 无效方向：DDD 动态 tree budget

本轮实现并测试了一个 prototype：当 DDD 提前停止时，将 tree verification 的节点预算按实际深度同步缩小，试图把浅层停止转化为更小 verification tree。

小样本 MT-Bench 5 records / 10 turns 结果如下：

| Method | macro tok/s | global tok/s | accept/step | avg depth | mean budget | early stop |
|--------|-------------|--------------|-------------|-----------|-------------|------------|
| DDD tau=-2.0, no dynamic budget | 84.09 | 85.01 | 6.035 | 9.087 | 59.00 | 75.24% |
| DDD tau=-2.0, dynamic budget min32 | 52.95 | 53.52 | 5.897 | 9.014 | 45.30 | 76.78% |
| DDD tau=-0.5, no dynamic budget | 88.13 | 88.98 | 5.714 | 6.664 | 59.00 | 97.24% |
| DDD tau=-0.5, dynamic budget min32 | 51.09 | 51.64 | 5.348 | 6.710 | 35.07 | 96.54% |

结论：

- 动态减少 tree budget 是明显负优化。
- 节点数减少后，候选覆盖变差，accept/step 下降；同时动态构造更小树没有带来足够的端到端 wall-time 收益。
- 该方向不建议进入 full-set 实验，也不建议作为最终优化点。

实现备注：

- 代码中保留了 `use_ddd_dynamic_budget` 和 `ddd_min_budget` 参数，方便后续复查；
- 默认值为关闭，不影响现有 DDD / EAGLE 路径。

## 5. 无效方向：扩大 OPT-Tree 候选池

为了确认 OPT-Tree 无效是否只是候选池太小，补充测试了 `tree_top_k=20`，并使用 `opt_expand_factor=2.0,4.0`。

MT-Bench 2 records / 4 turns / max_new_tokens=64：

| tree_top_k | expand | tree calls | mean Jaccard | identical rate | base-only | opt-only | macro tok/s |
|------------|--------|------------|--------------|----------------|-----------|----------|-------------|
| 20 | 2.0 | 49 | 1.000 | 100.00% | 0.00 | 0.00 | 62.64 |
| 20 | 4.0 | 49 | 1.000 | 100.00% | 0.00 | 0.00 | 19.36 |

结论：

- 即使显著扩大候选池，OPT-Tree 和 EAGLE-3 baseline 的最终节点集合仍完全相同。
- expand=4.0 只增加了选择开销，吞吐明显下降。
- OPT-Tree 的问题不是简单调大候选集能解决，而是它与 EAGLE-3 dynamic tree selection 的目标高度重叠。

## 6. 最终建议

当前基础上性能明显提升的优化方式只有一个：

| 推荐项 | 配置 |
|--------|------|
| 使用 DDD | `use_ddd=True` |
| 最大深度 | `ddd_max_depth=11` |
| 检查层 | `ddd_check_steps=5,7,9` |
| 阈值 | `ddd_threshold=-1.0` |
| OPT-Tree | 不启用 |
| dynamic budget | 不启用 |

最终报告建议写法：

> 在 EAGLE-3 strong baseline 上，OPT-Tree 未产生有效 tree-set 差异；DDD 的原始阈值设置也没有超过 fixed depth 11。进一步阈值调参发现，`tau=-1.0` 能在平均 draft depth 和 accept length 之间取得更好平衡，使 full MT-Bench macro tok/s 达到 `88.27`，相比 fixed depth 11 提升约 `14.6%`，相比此前 DDD best `tau=-0.5` 提升约 `24.8%`。

## 7. 结果文件

| 内容 | 文件 |
|------|------|
| DDD tau=-1.0 / -0.3 full | `experiments/E6_optimizations/ddd_sweep_mt_bench_limit-80_max-128_depth-11_tau-m1p0-m0p3.json` |
| fixed depth 9 full | `experiments/E6_optimizations/fixed_depth_mt_bench_limit-80_max-128_depths-9.json` |
| DDD dynamic budget smoke | `experiments/E6_optimizations/ddd_sweep_mt_bench_limit-5_max-128_depth-11_tau-m2p0-m0p5_budget-min32.json` |
| OPT expanded candidate smoke | `experiments/E6_optimizations/tree_diff_mt_bench_limit-2_max-64_expand-2p0-4p0.json` |
