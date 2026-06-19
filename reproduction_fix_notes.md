# EAGLE-3 + DDD/OPT-Tree 复现实验修改说明

> 本文档基于当前仓库代码和已有实验结果整理，目标是把后续修改方向固定下来，避免继续在不公平或不可解释的实验口径上迭代。

## 1. 当前状态概览

当前项目选择的技术路线是：

- Baseline: SafeAILab/EAGLE 仓库中的 EAGLE-3 推理框架
- Target model: `models/Llama-3.1-8B-Instruct`
- EAGLE drafter: `EAGLE_checkpoints/EAGLE3-LLaMA3.1-Instruct-8B`
- 改进方向: DDD 和 OPT-Tree

已有实验包括：

- `experiments/E2.1_timing/`: EAGLE 单步时延分解
- `experiments/E2.2_acceptance/`: 按树深度统计接受率
- `experiments/E2.3_tree_util/`: 树节点利用率分析
- `experiments/E3_ablation/`: DDD、OPT-Tree、joint 消融
- `experiments/E4_lossless/`: 输出一致性验证
- `experiments/E4_scenarios/`: 多 prompt 类型场景实验
- `experiments/hparam_search/`: DDD 阈值网格搜索

关键现象：

- EAGLE-3 baseline 有加速，但总体加速比偏低。
- DDD 在部分实验中能略微改善时延，但效果不稳定。
- OPT-Tree 的 `accept_per_step` 基本与 baseline 完全一致，甚至 tokens/s 下降。
- 当前 `verify_lossless.py` 的结果为 `all_match=false`，无损性验证口径还没有完全理顺。

## 2. 核心诊断

### 2.1 Baseline 复现口径不公平

当前很多实验脚本直接使用裸 prompt：

- `experiments/config.py`
- `experiments/ablation_ddd.py`
- `experiments/ablation_full.py`
- `experiments/scenario_test.py`
- `experiments/hparam_search.py`
- `experiments/verify_lossless.py`

例如：

```python
input_ids = tokenizer([prompt], add_special_tokens=True).input_ids
```

但 LLaMA-3.1-8B-Instruct 是 chat model。EAGLE 官方 README 明确说明，Vicuna、LLaMA2-Chat、LLaMA3-Instruct 这类 chat model 必须使用正确 chat template，否则会影响输出质量和 EAGLE 性能。

官方 LLaMA3 评测脚本使用的是：

```python
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
input_ids = tokenizer([prompt], add_special_tokens=False).input_ids
```

因此，当前裸 prompt 实验可以作为“简单 sanity check”，但不适合作为最终复现论文或对比其他组的主结果。

### 2.2 Pure AR baseline 与官方口径不同

当前 `verify_lossless.py` 中的 Pure AR baseline 使用 HuggingFace 的 `AutoModelForCausalLM.generate()`：

```python
out = model.generate(input_t, max_new_tokens=MAX_TOKENS, do_sample=False)
```

而 EAGLE 官方评测中的 vanilla baseline 使用的是同一个 `EaModel` 内部的 `naivegenerate()`，也就是同一份改过 KV cache 的 model implementation。

这两个 baseline 不完全等价：

- 使用的模型类不同
- KV cache 路径不同
- 停止条件不同
- 计时开销不同

后续 speedup ratio 应统一为：

```text
EAGLE eagenerate / EaModel naivegenerate
```

而不是：

```text
EAGLE eagenerate / HuggingFace generate
```

### 2.3 `max_new_tokens` 存在 off-by-one

`EAGLE/eagle/model/ea_model.py` 中，`eagenerate()` 和 `naivegenerate()` 都使用：

```python
if new_token > max_new_tokens:
    break
```

这会导致输出经常超过 `max_new_tokens` 1 到数个 token。当前实验结果中常见 `n_tokens=257/258/259`，就是这个现象的体现。

这会影响：

- tokens/s 统计
- lossless token-by-token 对齐
- 不同配置之间的停止一致性

建议改为更严格的边界控制：

```python
if new_token >= max_new_tokens:
    break
```

或在记录指标时统一截断到前 `max_new_tokens` 个 token。

### 2.4 DDD 检查位置与论文算法存在偏差

当前 DDD 逻辑位于 `EAGLE/eagle/model/cnets.py` 的 `topK_genrate()` 中。实现思路是：

1. 先执行本轮 draft model forward
2. 更新 `scores`
3. 如果当前步在 `ddd_check_steps` 中，再计算 `H`
4. 若 `H < threshold`，break

问题是：这一步的 draft forward 已经执行完了，开销已经付掉了。DDD 论文的核心收益来自“在下一次 draft model call 之前”判断是否继续，因此检查应尽量放到下一次扩展之前。

另外当前配置与论文默认设置也不一致：

- 论文推荐 `n=11`
- 论文推荐 `S={5,7,9}`
- 论文推荐 threshold `x=-0.3`
- 论文使用 EAGLE-2 + MT-Bench + temperature 0

当前网格搜索主要扫 `[-4, -6, -8, -10, -12]`，与论文阈值数量级不同。由于不同代码中 `H` 的定义可能略有差异，可以继续扫阈值，但应先打印并保存实际 `H` 分布，再用分位数决定合理搜索区间。

### 2.5 DDD 统计数据被污染

当前 early stop 时会执行：

```python
self._ddd_stats["depths"].append(i + 1)
break
```

循环结束后又会执行：

```python
actual_depth = len(ss_token)
self._ddd_stats["depths"].append(actual_depth)
```

这会让一次 draft 被记录两次，导致 `avg_depth` 不可信。

后续应只在每次 `topK_genrate()` 结束时记录一次实际 draft call 数，建议记录字段包括：

- `actual_depth`
- `checked_steps`
- `H_values`
- `early_stop_step`
- `early_stopped`

### 2.6 OPT-Tree 在当前 baseline 上几乎退化为等价实现

当前 baseline 的 EAGLE-2/3 dynamic tree 已经做了：

1. beam search over-expand 出大量候选
2. 按累计 path logprob 全局取 top-N
3. 用 ancestor closure 构造连通树

当前 OPT-Tree 分支的做法是：

1. 先按 `scores_list` 取 top-over-N
2. 再按 `exp(score)` 从高到低选择节点并补祖先

由于子节点累计概率不会高于祖先，按累计 logprob top-N 通常已经天然包含必要祖先。因此当前 OPT-Tree 和 baseline 选择出的有效树非常接近。

这解释了 `experiments/E3_ablation/full_ablation.json` 中的现象：

- OPT-Tree 的 `accept_per_step` 与 baseline 完全相同
- joint 配置的接受长度也几乎没有变化
- OPT-Tree 只增加 Python 侧 tree construction 开销，导致 tokens/s 下降

因此，后续不建议在报告中声称“当前 OPT-Tree 成功提升了 EAGLE-3”。更合理的表述是：

> OPT-Tree 在 static EAGLE tree 或 binary tree 上有明确收益，但在 EAGLE-2/3 的 dynamic tree baseline 上与原有 rerank 目标高度重叠，当前实现退化为近似 baseline，因此没有带来额外接受长度提升。

这可以作为失败分析写入报告。

### 2.7 Tree utilization 统计不能直接支撑 OPT-Tree

`experiments/profiling_tree_util.py` 中原本计划 monkey patch `topK_genrate()` 记录 `N_expand`，但实际 patch 没有安装，最终统计主要来自外部可观测量：

- `N_verify = 60`
- `N_accepted = accept_length + 1`

这可以说明“最终路径只使用了树上的少量节点”，但这是 tree verification 的天然性质。它不能证明：

- baseline rerank 选错了节点
- 高 path confidence 与 target acceptance 不相关
- OPT-Tree 会比 EAGLE-2/3 dynamic tree 更好

若要支撑 OPT-Tree，需要额外记录：

- 被选中节点的 cumulative logprob
- 未被选中但候选池中节点的 cumulative logprob
- 每个节点是否位于最终 accepted path
- baseline tree 与 OPT tree 的节点集合差异
- 两者的 retrieve paths 差异

## 3. 修改原则

后续修改遵循以下原则：

1. 先修正实验口径，再调算法。
2. 先复现官方 baseline，再验证自定义改进。
3. 所有配置必须共享同一个 prompt formatting、dtype、停止条件和测速方式。
4. DDD 和 OPT-Tree 的效果应分别用直接指标验证：
   - DDD: draft call 数是否下降，tokens/s 是否提升，acceptance 是否没有明显下降
   - OPT-Tree: 树节点集合是否变化，mean acceptance length 是否提升
5. 不再用“总 token 数相同”作为 lossless 证明。

## 4. 建议修改项

### P0. 增加统一 prompt builder

新增一个公共函数，例如放在 `experiments/common.py`：

```python
def build_chat_input(tokenizer, user_prompt, system_prompt=None):
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer([prompt], add_special_tokens=False).input_ids
```

所有主实验都改成使用该函数。

裸 prompt 版本可以保留，但应明确命名为：

```text
sanity / raw-prompt / toy prompt
```

不要作为最终主表。

### P0. 新增官方口径 baseline 脚本

建议新增：

```text
experiments/run_official_baseline.py
```

功能：

- 复用 `EaModel.from_pretrained()`
- 对同一批 chat-formatted prompts 同时跑：
  - `model.naivegenerate()`
  - `model.eagenerate()`
- 统一记录：
  - generated token ids
  - new_token
  - wall_time
  - tokens/s
  - EAGLE steps
  - accept_per_step

这样得到的 baseline 才能作为后续 DDD/OPT-Tree 消融对照。

### P0. 修正 lossless 验证

建议将 lossless 验证拆成两层：

1. Greedy deterministic check:
   - `naivegenerate()` vs `eagenerate()`
   - 同一 `EaModel`
   - 同一 input ids
   - 同一 max token 截断
   - 比较完整 token 序列

2. Sampling distribution check:
   - temperature > 0
   - 固定多组 random seeds
   - 验证不是逐 token 完全相等，而是检查分布统计或至少确认 rejection sampling 路径没有出错

当前阶段优先完成第一层。

### P0. 修正 `max_new_tokens` 边界

建议在 `EAGLE/eagle/model/ea_model.py` 中统一改为：

```python
if new_token >= max_new_tokens:
    break
```

同时所有统计脚本中用实际生成 token 数：

```python
n_tok = len(output_ids[0]) - len(input_ids[0])
tok_s = n_tok / wall
```

不要使用固定 `MAX_NEW_TOKENS / wall` 估算。

### P1. 重写 DDD 统计与检查逻辑

建议在 `topK_genrate()` 内部整理如下字段：

```python
stats = {
    "calls": 0,
    "actual_depths": [],
    "early_stops": 0,
    "checked_H": [],
    "early_stop_steps": [],
}
```

每次 `topK_genrate()`：

- 只记录一次 `actual_depth`
- 保存每个检查点的 H
- 保存是否 early stop

同时建议先运行一个 H profiling：

```text
experiments/profile_ddd_h.py
```

输出：

```text
experiments/E3_ablation/ddd_h_distribution.json
```

再根据实际分布选择阈值，而不是直接沿用当前 `[-4, -12]`。

### P1. DDD 实验重新设计

建议主表包含：

| 配置 | max_depth | check_steps | threshold | tok/s | accept/step | avg draft calls | early stop rate |
|------|-----------|-------------|-----------|-------|-------------|-----------------|-----------------|
| Baseline depth=5 | 5 | - | - | | | | |
| Fixed depth=7 | 7 | - | - | | | | |
| Fixed depth=11 | 11 | - | - | | | | |
| DDD paper-like | 11 | 5,7,9 | tuned | | | | |
| DDD best | tuned | tuned | tuned | | | | |

注意：DDD 论文相对 EAGLE-2 的提升本来就是几个百分点，不应期待数量级提升。

### P1. OPT-Tree 改为失败分析或换对照对象

当前有两条可选路线：

#### 路线 A: 把 OPT-Tree 写成失败分析

保留当前实现和实验，补充说明：

- EAGLE-2/3 dynamic tree baseline 已经接近 OPT-Tree 的 path cumulative probability 目标
- OPT-Tree 没有改变 mean acceptance length
- Python 侧树构造额外开销反而拉低 tokens/s
- 这说明在强 dynamic-tree baseline 上，OPT-Tree 不是一个正交优化点

这条路线风险低，适合最终报告里体现“失败尝试也分析清楚”。

#### 路线 B: 改成与 static tree 对比

若仍想复现 OPT-Tree 的正向收益，应新增一个 static EAGLE tree baseline 或 binary tree baseline，然后比较：

- static EAGLE tree
- binary tree
- OPT-Tree
- EAGLE-2/3 dynamic tree

这样 OPT-Tree 才有更清晰的复现对象。

### P2. 修正 tree utilization 实验

如果保留 E2.3，应重命名为：

```text
Tree verification budget utilization
```

不要直接叫 rerank failure。

若要支撑 rerank/OPT-Tree，应真正 hook `topK_genrate()`，记录每轮：

```python
{
    "candidate_scores": ...,
    "selected_indices_baseline": ...,
    "selected_indices_opt": ...,
    "accepted_node_indices": ...,
    "tree_position_ids": ...,
}
```

然后计算：

- baseline 与 OPT 的 Jaccard similarity
- selected path score 分布
- accepted path 中节点是否被两个方法同时选中
- OPT 是否引入更深路径

### P2. 模型 dtype 记录

LLaMA-3.1-8B-Instruct 原始 config 中 `torch_dtype` 是 `bfloat16`，当前实验统一强制 `torch.float16`。RTX 3090 对 bf16 支持有限，因此使用 fp16 可以接受，但报告中必须明确：

```text
由于 RTX 3090 对 bf16 支持不如 A100/H100，本实验统一使用 fp16。
这可能导致与官方报告存在数值和性能差异。
```

## 5. 建议实验重跑顺序

### Step 1: 官方口径 baseline

先只跑：

- `naivegenerate()`
- `eagenerate()`

使用：

- chat template
- temperature 0
- same max_new_tokens
- same prompts
- same dtype
- same model implementation

目标：

- 确认 baseline speedup
- 确认 greedy token 序列一致
- 得到可靠的 accept/step

### Step 2: DDD 修正版

只比较：

- baseline depth=5
- fixed depth=7
- fixed depth=11
- DDD with tuned threshold

目标：

- 证明 DDD 确实减少 draft calls
- 证明 tokens/s 有提升或解释为什么没有
- 证明 acceptance 没有明显下降

### Step 3: OPT-Tree 定性分析

先不急着追求提升，先验证：

- OPT 选出的树和 baseline 选出的树是否不同
- 如果不同，accept/step 是否改变
- 如果几乎相同，写成失败分析

### Step 4: 最终消融表

如果 DDD 有稳定收益，最终主表可以是：

| 方法 | tok/s | speedup vs naive | accept/step | avg draft calls | 备注 |
|------|-------|------------------|-------------|-----------------|------|
| Naive AR | | 1.00x | - | - | official baseline |
| EAGLE-3 | | | | fixed depth |
| EAGLE-3 + DDD | | | | lower | main improvement |
| EAGLE-3 + OPT-Tree | | | | | likely no gain |
| EAGLE-3 + DDD + OPT | | | | | analyze interaction |

## 6. 报告中建议的表述

### 对 baseline 差距的表述

可以写：

> 我们初始复现中仅得到约 1.6x 加速，低于论文报告。后续排查发现，差异主要来自评测口径和系统环境：初始实验使用裸 prompt 而非 chat template，Pure AR baseline 使用 HuggingFace generate 而非 EAGLE 官方 naivegenerate，同时硬件为单卡 RTX 3090，模型为 LLaMA-3.1-8B-Instruct，与论文中 Vicuna/LLaMA2-chat、MT-Bench 和 A40/多卡设置不同。因此我们重新统一了 prompt template、baseline 实现和停止条件。

### 对 DDD 的表述

可以写：

> DDD 的核心收益来自减少不必要的 drafter forward。我们将其实现为基于 beam logprob mass 的动态深度控制，并额外记录每轮实际 draft calls、H 分布和 early stop rate。实验重点不只看 tokens/s，还看 draft calls 是否下降以及 acceptance length 是否保持稳定。

### 对 OPT-Tree 的表述

如果最终仍无提升，可以写：

> OPT-Tree 在 static tree baseline 上目标明确，但我们使用的 EAGLE-3 已包含 EAGLE-2 风格 dynamic tree rerank，其节点选择已经按 cumulative path confidence 全局排序并保证连通性。我们的实现发现 OPT-Tree 与该 baseline 的节点选择高度重合，因此 mean acceptance length 几乎不变，而额外 tree construction 开销降低了吞吐。该结果说明 OPT-Tree 与 EAGLE-2/3 dynamic tree 并不完全正交，是本项目中的一个 negative result。

## 7. 推荐文件改动清单

建议新增：

- `experiments/common.py`
- `experiments/run_official_baseline.py`
- `experiments/profile_ddd_h.py`
- `experiments/verify_lossless_eamodel.py`
- `experiments/analyze_tree_selection.py`

建议修改：

- `EAGLE/eagle/model/ea_model.py`
  - 修正 `new_token >= max_new_tokens`
- `EAGLE/eagle/model/cnets.py`
  - 修正 DDD 检查位置
  - 修正 DDD stats
  - 暂时保留 OPT-Tree，但加入 tree diff 记录
- `experiments/ablation_ddd.py`
  - 使用 chat template
  - 记录实际 draft calls
- `experiments/ablation_full.py`
  - 删除“总 token 数相同即 lossless”的判断
  - 增加 token sequence compare
- `experiments/hparam_search.py`
  - 用实际生成 token 数计算 tok/s
  - 根据 H 分布重新设置 threshold search
- `experiments/profiling_tree_util.py`
  - 明确当前指标含义，或改为真正记录 candidate selection

## 8. 修改后的预期结论

更合理的最终结论可能是：

1. EAGLE-3 baseline 在我们的 RTX 3090 + LLaMA-3.1-8B-Instruct 环境下能提供稳定加速，但加速比低于论文强硬件和特定 benchmark 设置。
2. DDD 在部分场景下能减少 drafter 侧开销，但收益受 lazy evaluation、检查开销和硬件特性影响，预期提升为小幅提升。
3. OPT-Tree 与 EAGLE-2/3 dynamic tree 的目标高度重叠，在当前 baseline 上没有稳定收益，是一个有价值的 negative result。
4. 代码生成类 prompt 的 acceptance 更高，翻译和创意写作类 prompt 较低，说明投机解码收益强依赖任务分布。

## 9. 最小可交付版本

如果时间紧，建议优先完成以下最小闭环：

1. 用 chat template + `EaModel.naivegenerate()` 重跑 baseline。
2. 修正 `max_new_tokens` 和 lossless 验证。
3. 修正 DDD stats，并重跑 DDD-only 消融。
4. 将 OPT-Tree 写成 negative result，不再强行包装成正向提升。

这样最终报告仍然是完整的：

- 有真实系统瓶颈 profiling
- 有 baseline 复现
- 有 DDD 改进和消融
- 有 OPT-Tree 失败分析
- 有局限性讨论

