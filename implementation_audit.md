# Implementation Audit for DDD and OPT-Tree

本文档记录对当前 DDD / OPT-Tree 复现代码的再次检查结论。结论优先：

- 没有发现“开关没生效”“统计完全错位”“baseline 没对齐”这类会直接推翻已有实验的大 bug。
- DDD 和 OPT-Tree 的不理想结果不完全是调参问题，更主要来自 baseline 很强、任务分布、fp16 数值分叉，以及当前实现相对论文算法的若干口径差异。
- 仍然存在几处会削弱复现效果或影响报告严谨性的实现/评估风险，建议在报告里说明，若时间允许可做小规模补充。

## 1. 已确认基本正确的部分

### 1.1 Prompt 和 history 口径已经修正

`experiments/common.py` 使用 tokenizer chat template：

- `apply_chat_template(..., add_generation_prompt=True)`
- `add_special_tokens=False`
- 多轮实验会把上一轮 assistant 输出加入 history

这说明当前 baseline / DDD / OPT 实验不是早期那种 raw prompt 口径。

### 1.2 EAGLE 和 naive baseline 使用同一个 EaModel

`experiments/run_official_baseline.py` 在同一个 `EaModel` 内比较：

- `EaModel.naivegenerate()`
- `EaModel.eagenerate()`

这比直接调用外部 HF `generate()` 更接近 EAGLE 官方评测口径。

### 1.3 DDD 开关确实生效

`EAGLE/eagle/model/cnets.py` 中：

- `depth = self.ddd_max_depth if self.use_ddd else self.depth`
- 在 draft expansion 中根据 `ddd_check_steps` 检查 `H`
- 结果 JSON 中 `avg depth`、`early_stop_rate` 明显变化

因此 DDD 不是“没有跑起来”，而是跑起来后 end-to-end tok/s 没有超过 fixed depth 11。

### 1.4 OPT 开关也确实进入最终树构造

`use_opt_tree=True` 时走 `EAGLE/eagle/model/cnets.py` 的 OPT branch，重新构造：

- `draft_tokens`
- `tree_mask`
- `tree_position_ids`
- `retrieve_indices`

并不是单纯记录诊断值。

## 2. 可能削弱效果的实现问题

### 2.1 DDD 的检查位置偏晚，会少省一次 draft forward

当前位置在 `topK_genrate()` 的 expansion loop 中，先完成当前 step 的 drafter forward，再计算 H 并决定是否 `break`：

```text
out_hidden = self(...)
...
if self.use_ddd and (i + 1) in self.ddd_check_steps:
    H = logsumexp(scores)
    if H < threshold:
        break
```

因此当 `check_step=5` 触发 early stop 时，当前实现的实际 depth 记录为 6。这说明它已经完成了第 5 次循环对应的计算，只是阻止了后续更深扩展。

影响：

- DDD 机制指标仍然有效，确实减少了后续 depth。
- 但节省的计算比理想 DDD 少一截。
- 这会显著削弱 DDD 的 tok/s 提升，尤其当 target verify 本来就是主要耗时部分时。

建议：

- 若继续优化 DDD，应实现一个 pre-check 或 early-planning 版本：在进入下一次 drafter expansion 前决定是否继续。
- 报告中应说明当前 DDD 是 post-step check，可能低估论文算法潜在收益。

### 2.2 DDD depth 命名存在 off-by-one 语义

当前 `depth=5` 实际包含：

- 初始 top-k proposal
- 再执行 5 次 expansion loop

所以统计中的 `actual_depth` 最小值是 6，而不是 5。fixed depth 和 DDD 都用同一套语义，因此实验内部可比；但和论文或其他组结果对齐时容易混淆。

建议：

- 文档中写清楚本实现的 depth 是 expansion loop count，实际树深度约为 `depth + 1`。

### 2.3 OPT-Tree 的候选空间已经被 EAGLE beam prune 过

当前 OPT 不是从一个完整大树中重新优化树形，而是在 EAGLE dynamic expansion 已经逐层保留 top-k path 后，再从 `scores_list` 中 over-expand / rerank：

```text
topk_cs = topk(cu_scores.view(-1), top_k)
...
scores_list = concat(...)
...
top_over = topk(scores_list, over_N)
```

也就是说，很多潜在节点在 OPT 之前已经被 EAGLE 的 beam pruning 丢掉。OPT 只能在这个已经很强、已经高度排序过的候选集合上做后处理。

影响：

- 当前 OPT 实现更准确地说是 “OPT-style rerank on EAGLE dynamic candidates”，不是完整重构 OPT-Tree。
- 这会让 OPT 很容易退化为 baseline，因为 EAGLE-3 dynamic baseline 本来就在按 cumulative path confidence 选节点。

### 2.4 OPT 的目标函数实现较简化

当前 OPT greedy 使用 `exp(cumulative logprob)` 作为单节点贡献，并加 ancestor closure。若论文中的 OPT-Tree 使用更复杂的 expected acceptance length / path probability / DP 形式，则当前实现并不是严格复现。

但注意：即便把 `opt_expand_factor` 从 1.5/2.0 提到 4.0/8.0，小样本 tree diff 仍然完全一致：

| expand | tree calls | mean Jaccard | identical rate | baseline-only | OPT-only |
|--------|------------|--------------|----------------|---------------|----------|
| 4.0 | 50 | 1.000 | 100.00% | 0.00 | 0.00 |
| 8.0 | 50 | 1.000 | 100.00% | 0.00 | 0.00 |

结果文件：

- `experiments/E4_opt_tree/tree_diff_mt_bench_limit-2_max-64_expand-4p0-8p0.json`

因此“OPT 无效只是 expand factor 太小”的解释不太成立。

### 2.5 tok/s 统计当前是 macro average

脚本中的 `mean_tok_per_s_trimmed` 是每个 turn 的 tok/s 再取平均，而不是：

```text
global tok/s = total_trimmed_tokens / total_wall_time_s
```

例如 full MT-Bench：

| Method | macro tok/s | global tok/s |
|--------|-------------|--------------|
| Naive AR | 17.59 | 16.77 |
| EAGLE-3 | 68.55 | 64.28 |
| fixed depth11 | 77.03 | 72.84 |
| DDD tau=-0.5 | 70.74 | 67.39 |

趋势没有反转，fixed depth11 仍然强于 DDD，但报告中应明确使用哪种 tok/s。正式报告更建议主表使用 global tok/s，macro tok/s 可作为补充。

## 3. 调参是否不充分

### 3.1 DDD full-set threshold grid 还不算充分

已有实验：

- MT-Bench 5 records：`tau ∈ {-5,-4,-3,-2,-1,-0.5,-0.3}`
- MT-Bench 20 records：`tau ∈ {-2,-0.5}`
- MT-Bench 80 records：`tau ∈ {-2,-0.5}`

从小样本看，`tau=-0.5` 附近最好；full-set 上它也比 `tau=-2` 快，但仍不如 fixed depth11。

结论：

- 当前调参足以说明 DDD 没有稳定超过 fixed depth11。
- 但若要追求最佳数值，full-set 还可以补 `tau=-1.0,-0.3`，以及 fixed depth `7,9`。
- 这可能改善 DDD 的最好数字，但不太可能改变 OPT 的结论。

### 3.2 OPT expand factor 不是主要瓶颈

已有：

- expand 1.5 / 2.0：tree set 完全一致
- 新增 expand 4.0 / 8.0 小样本复核：tree set 仍完全一致

因此 OPT 不理想更可能来自 baseline overlap 和候选空间已经被 EAGLE prune，而不是 expand factor 没调够。

## 4. 其他导致结果不理想的原因

### 4.1 EAGLE-3 baseline 太强

fixed depth11 在 full MT-Bench 上达到：

- macro tok/s: 77.03
- global tok/s: 72.84
- accept/step: 6.392

DDD tau=-0.5：

- macro tok/s: 70.74
- global tok/s: 67.39
- accept/step: 6.041

也就是说，DDD 减少了 depth，但 fixed depth11 得到了更好的 acceptance 和更少 loop。这个 baseline 很难超过。

### 4.2 MT-Bench 任务分布不利于统一阈值

按 category 看，writing、roleplay、math、coding、humanities 等差异很大。统一 threshold 很可能无法同时适配所有类别。

这也是为什么其他组在不同数据集上可能得到差异很大的结果。

### 4.3 fp16 数值分叉影响 exact match

fp16 下 MT-Bench greedy exact match 不是 100%，而 fp32 deterministic 小样本可以达到 100%。这说明 correctness 差距主要不是 prompt 或停止边界错误，而是 fp16 tree verification 的数值路径问题。

## 5. 建议的后续优先级

若时间有限，不建议继续围绕 OPT expand factor 做大规模调参。更值得做的是：

1. 把最终报告的 tok/s 主表改成 global tok/s，并保留 macro tok/s 作为附录。
2. 在 full MT-Bench 上补一个小 DDD grid：`tau=-1.0,-0.3`，并补 fixed depth `7,9`，确认最优 DDD/固定深度点。
3. 若想真正追求 DDD 提升，修改 DDD 为 pre-check 版本，或至少记录每次 early stop 实际节省的 drafter forward 数。
4. 若想证明 OPT-Tree 的论文式正向收益，需要换更弱 baseline：static tree / binary tree / EAGLE-1/2，而不是继续只在 EAGLE-3 dynamic baseline 上调 expand factor。

当前结果最合理的解释是：

> EAGLE-3 dynamic baseline 已经很强；DDD 当前实现能降低深度但检查位置偏晚且统一阈值难以覆盖所有任务；OPT-Tree 在 EAGLE-3 已 pruning 的候选集合上退化为相同树。因此结果不理想主要是 baseline/实现口径/任务分布共同造成，不是单一代码 bug。
