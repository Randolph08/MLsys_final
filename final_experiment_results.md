# Final Experiment Results and Analysis

本文档汇总当前仓库中已经完成的主要复现实验、补充诊断实验和最终消融实验。实验主线是：在 EAGLE-3 框架上复现并分析 DDD 和 OPT-Tree，并解释为什么当前结果没有达到论文中理想的正向提升。

## 1. 实验口径

统一口径如下：

| 项目 | 设置 |
|------|------|
| Base model | `models/Llama-3.1-8B-Instruct` |
| EAGLE checkpoint | `EAGLE_checkpoints/LLAMA3-Instruct-8B/` |
| 主数据集 | MT-Bench 本地 80 records / 160 turns |
| 生成长度 | `max_new_tokens=128` |
| Prompt | tokenizer chat template + system prompt |
| dtype | 性能实验使用 fp16；正确性诊断补充 fp32 |
| GPU | `CUDA_VISIBLE_DEVICES=0,1,2,3,4`，`device_map=auto` |
| 主要指标 | tok/s、speedup、accept/step、mean loop count、DDD avg depth、DDD early-stop rate、tree Jaccard、turn-level exact match |

说明：最终主表和后续新增的对比实验均采用 full MT-Bench 80 records / 160 turns。少量小样本结果只作为诊断或筛查实验保留，不作为最终性能结论。

## 2. 总体结论

1. EAGLE-3 baseline 已经在 full MT-Bench 上复现出明显加速：`68.55 tok/s` vs naive `17.59 tok/s`，约 `3.90x`。
2. Greedy exact match 没有在 fp16 下达到 100%。小实验表明主要原因是 fp16 数值路径导致的 near-tie/token 分叉：fp16 deterministic 仍 mismatch，fp32 deterministic 在 5 条 MT-Bench 上达到 `10/10 turn match`。
3. DDD 机制本身成立，并且经过补充阈值调参后取得当前最好的性能：`tau=-1.0` 在 full MT-Bench 上达到 `88.27 tok/s`，超过 fixed depth 11 的 `77.03 tok/s`。
4. OPT-Tree 在当前 EAGLE-3 dynamic tree baseline 上基本无效。E4.1 直接观察到 OPT 与 baseline 的最终树节点集合完全相同：`mean Jaccard=1.000`，`identical rate=100%`。
5. DDD+OPT 组合也没有带来额外收益：在 `tau=-2.0/-0.5/-1.0` 三个 full-set 配置下，accept/step、avg depth、early-stop rate 均与 DDD-only 完全一致，tok/s 不提升。
6. 不同任务类别差异明显。MT-Bench 8 个 category 的 EAGLE speedup 从 roleplay 的 `3.39x` 到 math 的 `4.26x`，说明只看总体均值会掩盖任务差异。

## 3. 最终主表

以下表格使用 full MT-Bench 80 records / 160 turns。`speedup vs EAGLE` 以 official EAGLE-3 default/depth5 的 `68.55 tok/s` 为参考。

| Method | tok/s | speedup vs naive | speedup vs EAGLE | accept/step | mean loop | avg depth | early stop |
|--------|-------|------------------|------------------|-------------|-----------|-----------|------------|
| Naive AR | 17.59 | 1.00x | - | - | 117.95 | - | - |
| EAGLE-3 default/depth5 | 68.55 | 3.90x | 1.00x | 5.341 | 22.41 | fixed 5 | - |
| EAGLE-3 fixed depth11 | 77.03 | 4.38x | 1.12x | 6.392 | 19.46 | fixed 11 | - |
| EAGLE-3 + DDD tau=-2.0 | 68.51 | 3.89x | 1.00x | 6.365 | 19.53 | 9.121 | 71.05% |
| EAGLE-3 + DDD tau=-0.5 | 70.74 | 4.02x | 1.03x | 6.041 | 20.43 | 6.991 | 93.18% |
| EAGLE-3 + DDD tau=-1.0 | 88.27 | 5.02x | 1.29x | 6.252 | 19.81 | 7.695 | 86.60% |
| EAGLE-3 + OPT-1.5 | 47.18 | 2.68x | 0.69x | 5.341 | 22.41 | fixed 5 | - |
| EAGLE-3 + DDD tau=-0.5 + OPT-1.5 | 69.35 | 3.94x | 1.01x | 6.041 | 20.43 | 6.991 | 93.18% |
| EAGLE-3 + DDD tau=-1.0 + OPT-1.5 | 77.56 | 4.41x | 1.13x | 6.252 | 19.81 | 7.695 | 86.60% |

关键解读：

- 早期 DDD `tau=-0.5` 只比 default EAGLE-3 略快，且不如 fixed depth11；补充调参后，`tau=-1.0` 成为当前最佳配置。
- `tau=-1.0` 说明 DDD 的收益不是“越早停越好”，而是要在 draft depth 和 accept/step 之间取得平衡。
- OPT-only 和 DDD+OPT 都没有改变 accept/step。OPT 的主要问题不是调参没调好，而是它与 EAGLE-3 dynamic tree selection 不正交。
- tok/s 存在顺序/预热波动，尤其是 OPT-only full run 中 EAGLE-3 repeat 达到 `80.50 tok/s`。因此 OPT 的判断应以 tree-set 和 acceptance 证据为主。

## 4. 正确性与 Lossless 诊断

### 4.1 已修正的问题

前期脚本中的主要风险是统计和停止边界口径不统一。当前已做如下修正：

- 统一使用 `tokenizer.apply_chat_template(..., add_generation_prompt=True)`。
- baseline 使用 `EaModel.naivegenerate()`，EAGLE 使用 `EaModel.eagenerate()`。
- 修正 `max_new_tokens` 边界，避免 raw output overshoot 被误当作 lossless mismatch。
- 所有主要结果都记录 actual generated token、trimmed token、wall time 和 loop count。

### 4.2 Greedy exact match 结果

MT-Bench 20 records / 40 turns：

| Setting | record match | turn match | mismatch turns |
|---------|--------------|------------|----------------|
| fp16 normal | 13/20 | 33/40 | 7 |
| fp16 deterministic no TF32, 5 records | 3/5 | 8/10 | 2 |
| fp32 deterministic no TF32, 5 records | 5/5 | 10/10 | 0 |

结论：当前 fp16 + tree verification 路径存在真实 token 分叉，但 fp32 诊断说明主要是数值精度/near-tie 问题，而不是 prompt 构造、历史拼接或停止边界错误。性能实验继续使用 fp16 是合理的，但最终报告应单独说明 exact match 没有达到 100%。

## 5. EAGLE-3 Baseline

Full MT-Bench official baseline：

| Method | turns | trimmed tokens | wall time | tok/s | mean loop |
|--------|-------|----------------|-----------|-------|-----------|
| Naive AR | 160 | 18845 | 1124.00s | 17.59 | 117.95 |
| EAGLE-3 | 160 | 18903 | 294.06s | 68.55 | 22.41 |

EAGLE-3 speedup 为 `3.8966x`。这是当前项目里复现最成功、最稳定的一部分。

需要注意：record-level exact match 为 `63/80`。这与第 4 节的 fp16 数值诊断一致，不能简单解释为算法错误。

## 6. DDD 结果

### 6.1 H 分布与阈值选择

小样本 H profiling 显示，MT-Bench 上 H 值比最初预期更负：

| step | median H |
|------|----------|
| 5 | -1.124 |
| 7 | -1.969 |
| 9 | -3.121 |

因此原先接近 `-0.3` 的 paper-like threshold 在当前实现和模型上过于激进；`tau=-2.0` 和 `tau=-0.5` 被选为最终全量对照。

### 6.2 Fixed depth 对照

Full MT-Bench fixed depth：

| depth | tok/s | accept/step | mean loop |
|-------|-------|-------------|-----------|
| 5 | 65.23 | 5.341 | 22.41 |
| 11 | 77.03 | 6.392 | 19.46 |

小样本上 depth5 曾经看起来更快，但 full MT-Bench 上 fixed depth11 更强。这说明小样本结论会被任务分布和 warmup 方差影响，最终报告必须优先使用 full-set 结果。

### 6.3 DDD 全量结果

| threshold | tok/s | accept/step | mean loop | avg depth | early stop |
|-----------|-------|-------------|-----------|-----------|------------|
| -2.0 | 68.51 | 6.365 | 19.53 | 9.121 | 71.05% |
| -0.5 | 70.74 | 6.041 | 20.43 | 6.991 | 93.18% |
| -1.0 | 88.27 | 6.252 | 19.81 | 7.695 | 86.60% |
| -0.3 | 65.99 | 5.907 | 20.84 | 6.714 | 94.99% |

DDD 的正向部分：

- 动态深度确实生效，平均 depth 从 fixed 11 降到 9.12、7.70 或 6.99 等不同水平。
- `tau=-0.5` 几乎每次都会 early stop，说明机制层面可以控制 draft 深度。
- 补充实验表明 `tau=-1.0` 是更好的折中点，平均 depth 为 `7.695`，但 accept/step 仍保持 `6.252`，因此 end-to-end tok/s 明显提升。

DDD 调参后的结论：

- `tau=-0.5` / `tau=-0.3` 过于激进，虽然深度更浅，但 accept/step 下降、loop 增加，最终速度变差。
- `tau=-2.0` 过于保守，平均 depth 偏深，省下的 draft 计算不足。
- `tau=-1.0` 在当前 EAGLE-3 + LLaMA3-8B + MT-Bench 设置下是推荐配置。

### 6.4 E6 优化补充实验

E6 进一步验证了四个优化思路：

| Direction | Result | Conclusion |
|-----------|--------|------------|
| DDD threshold tuning | `tau=-1.0` 达到 `88.27 tok/s` | 有效，作为最终推荐配置 |
| fixed depth 9 | `60.47 tok/s`，低于 depth11 | 无效，简单缩短固定深度不是优化 |
| DDD dynamic tree budget | MT-Bench 5 条从 `88.13 tok/s` 掉到 `51.09 tok/s` | 无效，候选覆盖下降大于节点预算收益 |
| OPT expanded candidates | `tree_top_k=20` 仍 `Jaccard=1.000` | 无效，扩大候选池不能改变 tree selection |

其中 DDD threshold tuning 和 fixed depth 9 是 full-set 结果；dynamic tree budget 与 OPT expanded candidates 是早期筛查诊断，不进入最终主表。详细记录见 `experiments/E6_optimizations/README.md`。

## 7. OPT-Tree 结果

### 7.1 Tree-set diff

E4.1 在 MT-Bench small slice 上记录了 baseline tree 与 OPT tree 的节点集合：

| Run | turns | tree calls | expand | mean Jaccard | identical rate | accept/step |
|-----|-------|------------|--------|--------------|----------------|-------------|
| MT-Bench 5, max128 | 10 | 232 | 1.5 | 1.000 | 100.00% | 5.286 |

所有观测调用中：

- baseline-only nodes = 0
- OPT-only nodes = 0
- final selected node set 完全一致

这说明 OPT-Tree 没有在当前 EAGLE-3 dynamic rerank baseline 上产生新的候选树。

### 7.2 OPT-only full run

| Method | tok/s | accept/step | mean loop |
|--------|-------|-------------|-----------|
| EAGLE-3 | 61.66 | 5.341 | 22.41 |
| OPT-1.5 | 47.18 | 5.341 | 22.41 |
| EAGLE-3-repeat | 80.50 | 5.341 | 22.41 |

accept/step 和 loop 完全相同，所以 OPT 没有算法层面的 acceptance gain。tok/s 在同一进程内出现较大顺序漂移，因此不能只凭第一轮 EAGLE 和 OPT 的速度差下结论；但结合 tree-set diff，可以稳健地说：OPT 没有带来可验证的算法收益。

### 7.3 DDD+OPT full run

| Method | DDD-only tok/s | DDD+OPT tok/s | accept delta | depth delta | early-stop delta |
|--------|----------------|---------------|--------------|-------------|------------------|
| tau=-2.0 | 68.51 | 60.56 | 0.000 | 0.000 | 0.00pp |
| tau=-0.5 | 70.74 | 69.35 | 0.000 | 0.000 | 0.00pp |
| tau=-1.0 | 88.27 | 77.56 | 0.000 | 0.000 | 0.00pp |

DDD+OPT 也不改变 acceptance、depth 和 early-stop 行为。补跑优化后的 `tau=-1.0` 后，OPT 仍然没有带来算法收益，反而从 `88.27 tok/s` 降到 `77.56 tok/s`。最终报告中建议把 OPT-Tree 写成 negative result：

> OPT-Tree 与 EAGLE-3 dynamic tree selection 的目标高度重叠。在当前 strong baseline 上，OPT-Tree 选择的最终节点集合与 baseline 完全相同，因此没有提升 acceptance length；额外选择逻辑只表现为实现开销或顺序敏感的吞吐波动。

## 8. E5.2 多任务类别分析

本项目没有额外下载外部数据集。为了回应“不同数据集/任务结果差异很大”的观察，E5.2 使用 full MT-Bench 的 8 个 category 作为不同任务子集，复用同一批全量实验日志离线聚合。

EAGLE baseline by category：

| category | naive tok/s | EAGLE tok/s | speedup | turn match |
|----------|-------------|-------------|---------|------------|
| writing | 17.49 | 66.58 | 3.806x | 80.0% |
| roleplay | 13.47 | 45.64 | 3.387x | 65.0% |
| reasoning | 20.62 | 79.16 | 3.839x | 85.0% |
| math | 20.97 | 89.37 | 4.262x | 95.0% |
| coding | 15.09 | 63.23 | 4.190x | 90.0% |
| extraction | 20.17 | 78.06 | 3.869x | 100.0% |
| stem | 14.73 | 57.02 | 3.870x | 80.0% |
| humanities | 16.51 | 61.84 | 3.745x | 80.0% |

主要观察：

- roleplay 最难，speedup 和 exact match 都最低。
- math/coding/extraction 更接近受约束输出，EAGLE speedup 更高。
- 优化后的 DDD `tau=-1.0` 在 8 个 category 上都高于 fixed depth11，但提升幅度差异很大。
- OPT 在所有 category 上 `accept delta=0`，这与 tree-set diff 完全一致。

因此最终报告不要只给一个全局平均值，应该保留 per-category 表格，说明任务分布会显著影响投机解码收益。

优化后 DDD `tau=-1.0` by category：

| category | fixed d11 tok/s | DDD tau=-1 tok/s | tau=-1/d11 |
|----------|-----------------|------------------|------------|
| writing | 74.13 | 83.98 | 1.133x |
| roleplay | 60.58 | 71.59 | 1.182x |
| reasoning | 75.88 | 86.55 | 1.141x |
| math | 86.66 | 94.39 | 1.089x |
| coding | 95.01 | 103.01 | 1.084x |
| extraction | 82.48 | 87.69 | 1.063x |
| stem | 77.58 | 88.84 | 1.145x |
| humanities | 52.60 | 85.13 | 1.618x |

## 9. 为什么结果没有达到理想状态

### 9.1 正确性不满 100%

理想状态下 greedy exact match 应该是 100%。当前 fp16 下不是 100%，但 fp32 小实验达到 100%，所以主要原因是 fp16 数值路径分叉。这个问题可以通过 fp32 诊断确认原因，但不能在 fp16 性能实验中无代价消除。

### 9.2 DDD 需要重新调阈值

最初选择的 `tau=-2.0` 和 `tau=-0.5` 没有超过 fixed depth11，因此早期结论是 DDD 表现不理想。补充调参后，`tau=-1.0` 证明 DDD 可以超过 fixed depth11。原因是 `tau=-1.0` 不像 `tau=-0.5` / `tau=-0.3` 那样过早停止，能够保留足够高的 accept/step，同时又避免大部分不必要的深层 draft。

因此最终报告应把 DDD 写成“经过阈值调参后有效”，而不是简单写成 negative result。

### 9.3 OPT-Tree 与 baseline 不正交

OPT-Tree 原本更适合相对弱的 static/binary tree baseline。EAGLE-3 dynamic rerank 已经按 cumulative path confidence 做全局节点选择，并且这个选择在实践中天然接近 ancestor-closed。OPT 的 ancestor-closure 优化在这里退化为同一个节点集合。

### 9.4 小样本与全量差异

MT-Bench small slice 中出现过 depth5、DDD 或 OPT tok/s 看似更优的情况，但 full MT-Bench 和 repeat run 显示这些结果对样本和顺序敏感。最终结论应以 full-set、tree-set、acceptance 等更稳健证据为准。

## 10. 最终报告建议写法

建议报告叙事如下：

1. 先说明修正后的 official baseline：EAGLE-3 在 MT-Bench 上达到约 `3.90x` 加速，baseline 复现成功。
2. 再说明 correctness：fp16 下 exact match 不是 100%，但 fp32 诊断确认主要是数值 near-tie。
3. DDD 写成“机制成立且阈值调参后有效”：早期 `tau=-0.5` 不如 fixed depth11，但 `tau=-1.0` 在 full-set 上达到当前最佳 tok/s。
4. OPT 写成 negative result：tree-set 完全重合，acceptance 完全不变，说明 OPT-Tree 与 EAGLE-3 dynamic baseline 不正交。
5. 加入 category split：不同任务子集差异很大，其他组在不同数据集上结果不同是合理现象。

## 11. 结果文件索引

| 内容 | 文件 |
|------|------|
| 实验计划 v2 | `experiment_plan_v2.md` |
| 修正说明 | `reproduction_fix_notes.md` |
| Official baseline full | `experiments/E1_official_baseline/official_baseline_mt_bench_limit-80_max-128.json` |
| Lossless / precision diagnosis | `experiments/E1_lossless/` |
| DDD full | `experiments/E3_ddd/ddd_sweep_mt_bench_limit-80_max-128_depth-11_tau-m2p0-m0p5.json` |
| DDD optimized full | `experiments/E6_optimizations/ddd_sweep_mt_bench_limit-80_max-128_depth-11_tau-m1p0-m0p3.json` |
| Fixed-depth full | `experiments/E3_ddd/fixed_depth_mt_bench_limit-80_max-128_depths-5-11.json` |
| Fixed-depth 9 full | `experiments/E6_optimizations/fixed_depth_mt_bench_limit-80_max-128_depths-9.json` |
| OPT tree diff | `experiments/E4_opt_tree/tree_diff_mt_bench_limit-5_max-128_expand-1p5.json` |
| OPT expanded tree diff | `experiments/E6_optimizations/tree_diff_mt_bench_limit-2_max-64_expand-2p0-4p0.json` |
| OPT full | `experiments/E4_opt_tree/opt_dynamic_mt_bench_limit-80_max-128_expand-1p5_single_repeat.json` |
| DDD+OPT full | `experiments/E5_ablation/ddd_sweep_mt_bench_limit-80_max-128_depth-11_tau-m2p0-m0p5_opt-1p5.json` |
| DDD optimized + OPT full | `experiments/E5_ablation/ddd_sweep_mt_bench_limit-80_max-128_depth-11_tau-m1p0_opt-1p5.json` |
| E5.1 README | `experiments/E5_ablation/README.md` |
| E5.2 category summary | `experiments/E5_scenarios/mt_bench_category_summary.json` |
| E5.2 README | `experiments/E5_scenarios/README.md` |
