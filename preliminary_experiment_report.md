# 投机解码推理优化 —— 前期实验汇报

> **项目**：USTC ML System 2026 Final Project
> **方案**：基于 EAGLE-3 的 DDD（动态深度）+ OPT-Tree（最优树形状）推理优化
> **实验日期**：2026-06-04
> **本文件用途**：为 6.9 开题报告提供完整的背景论述、实验数据和瓶颈分析

---

## 目录

1. [问题背景：LLM 推理为什么需要投机解码](#1-问题背景llm-推理为什么需要投机解码)
2. [实验环境与 Baseline](#2-实验环境与-baseline)
3. [前期实验全景：逻辑关系](#3-前期实验全景逻辑关系)
4. [实验一：EAGLE 管线时延分解](#4-实验一eagle-管线时延分解)
5. [实验二：逐深度 Token 触及率与接受率分析](#5-实验二逐深度-token-触及率与接受率分析)
6. [实验三：树节点利用效率分析](#6-实验三树节点利用效率分析)
7. [瓶颈总览与改进方案](#7-瓶颈总览与改进方案)

---

## 1. 问题背景：LLM 推理为什么需要投机解码

### 1.1 Memory-Bound 瓶颈

大语言模型的自回归生成过程是一个逐 token 串行的过程。每生成一个 token，都需要将整套模型权重从 GPU 显存（HBM）搬运到计算单元。这个过程的瓶颈特性是：

- **算术强度极低**：每次 decode step 的计算量与数据搬运量之比远小于 1
- **GPU 利用率极低**：batch=1 解码时，GPU 算力利用率（MFU）通常只有几个百分点
- **硬件趋势恶化**：GPU 算力增速远超显存带宽增速，memory wall 问题日趋严重

**这个结论不是理论推测**——在 Lab2（Roofline 分析）中，我们通过实测数据验证了 RTX 3090 上 Decode 阶段位于 memory-bound 区域。这构成了本项目的理论基础。

### 1.2 投机解码原理

投机解码（Speculative Decoding）是一种**无损加速**方法：

1. **Draft（草稿）**：用一个极便宜的 drafter（1 层 Transformer，~1% 参数量）自回归猜出未来 K 个 token
2. **Verify（验证）**：将"前缀 + 草稿"一次性送入完整 target 模型，并行计算 K+1 个位置的 logits
3. **Accept/Reject（决策）**：用 modified rejection sampling 逐位置验证，接受最长正确前缀

**加速原理**：一次 target forward 原本只出 1 个 token，现在可以验证 K 个——将搬运权重的带宽成本摊薄到多个 token 上。且 rejection sampling 保证输出分布与纯 target 解码完全等价（lossless）。

### 1.3 EAGLE-3 技术路线

EAGLE 系列是投机解码领域的代表性工作：

| 版本 | 核心贡献 | 我们如何使用 |
|------|---------|------------|
| EAGLE-1 (ICML'24) | Feature-level AR drafter | 理论奠基 |
| EAGLE-2 (EMNLP'24) | Context-aware dynamic draft tree（expand + rerank） | **改进直接作用于其树构造流程** |
| EAGLE-3 (ICLR'25) | 更强的 drafter（token prediction + multi-layer fusion + TTT） | **复用其预训练 checkpoint 作为 baseline** |

我们的改进定位在 **D6 维度（draft 长度/形状决策）**：DDD 自适应调整 beam search 扩展深度，OPT-Tree 最大化期望接受长度选取树节点。两者均在推理循环的树构造阶段，**不需要重训 drafter**。

---

## 2. 实验环境与 Baseline

### 2.1 硬件与软件

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA GeForce RTX 3090 (24 GB) × 1（实验使用 GPU 4） |
| CUDA | 12.8 |
| PyTorch | 2.6.0+cu124 |
| Transformers | 4.57.6 |

### 2.2 模型

| 组件 | 路径 |
|------|------|
| Target Model | `meta-llama/Llama-3.1-8B-Instruct`（从 ModelScope 下载） |
| EAGLE-3 Drafter | `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`（官方 checkpoint，从 hf-mirror 下载） |

### 2.3 EAGLE 推理参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `total_token` | 60 | Draft tree 中总节点数（含 root） |
| `depth` | 5 | Beam search 最大扩展步数 |
| `top_k` | 10 | 每步保留的 top-k 候选 |
| `temperature` | 0.0 | Greedy decoding |

### 2.4 Baseline 性能

| 配置 | 吞吐量 | 相对加速比 |
|------|--------|----------|
| Pure AR（HuggingFace `model.generate()`） | **40.0 tok/s** | 1.00×（性能下界） |
| **EAGLE-3 Baseline** | **63.8 tok/s** | **1.60×**（改进起点） |

Baseline 加速比 1.60× 虽然验证了投机解码的有效性，但明显低于 EAGLE-3 论文报告的 3-5×。这个差距意味着在特定硬件和模型配置下存在未被充分利用的优化空间——这正是我们后续瓶颈定位实验要探索的。

---

## 3. 前期实验全景：逻辑关系

三个实验不是孤立的测量，而是构成了一条从宏观到微观、从现象到归因的完整分析链：

```
实验零 (Lab2)     "Decode 阶段是 memory-bound"
    │                   │
    │    ┌──────────────┴──────────────┐
    │    │                             │
    ▼    ▼                             ▼
实验一 (E2.1)              实验二 (E2.2)          实验三 (E2.3)
管线时延分解               逐深度接受/触及分析      节点利用效率
    │                         │                      │
    │ "Drafter 占 26.6%"      │ "92.4% step 不走    │ "95.8% 验证节点
    │                         │  到 depth 5"         │  未被接受"
    │                         │                      │
    ▼                         ▼                      ▼
┌─────────────────────────────────────────────────────────┐
│              两个优化目标（均在 D6 维度）                  │
│                                                         │
│  DDD: 自适应深度 — 在 beam 信心不足时提前停止扩展         │
│  OPT-Tree: 最优节点选择 — 最大化 E[acceptance length]    │
└─────────────────────────────────────────────────────────┘
```

**实验间的依赖关系**：

- **实验一 → 实验二/三**：实验一确认 Drafter 侧（26.6%）值得优化，为后续实验框定范围
- **实验二 → DDD**：揭示固定扩展深度在 >90% step 中存在系统性浪费 → DDD 的立论依据
- **实验三 → OPT-Tree**：量化启发式 rerank 的低效率（4.2% 利用率）→ OPT-Tree 的立论依据
- **实验二 ↔ 实验三**：两个瓶颈相互正交 —— DDD 控制树深度，OPT-Tree 控制树形状，不存在冲突

---

## 4. 实验一：EAGLE 管线时延分解

### 4.1 实验动机

在决定优化什么之前，必须先知道时间花在哪里。EAGLE 的推理循环比纯 AR 解码复杂得多：它包含 drafter 的前向传播、tree mask 构建、target verify、rejection sampling、KV cache 更新等多个步骤。这个实验将每个 inference step 分解为关键阶段，量化各自的 GPU 时延占比。

### 4.2 实现方法

**插桩策略**：在 `eagenerate` 主循环中插入 `torch.cuda.Event` 计时点，将每个 step 分解为三个阶段：

```
一个 EAGLE step 的计时点布局：

  [循环体开始]
      │
      ├─ ev_start ──→ tree_decoding() ──→ ev_end    ← 阶段② Target Verify
      │
      ├─ ev_start ──→ evaluate_posterior() ──→ ev_end ← 阶段③ Rejection
      │
      ├─ ev_start ──→ update_inference_inputs() ──→ ev_end ← 阶段① Drafter+KV
      │                 └── 内含 topK_genrate()
      │                     (expand + rerank + KV update)
      │
      └─ torch.cuda.synchronize() → 读取三个阶段的 elapsed_time

注：阶段①(Drafter)和阶段③(Rejection)在代码中为相邻调用，
    但 rejection 实际仅占 0.2 ms，因此报告中将 ①③ 合并。
```

**为什么没有拆分 expand 和 rerank**：两者在 `topK_genrate` 内部紧密耦合，拆分需要完全重写该函数（~120 行），引入 bug 的风险较大。当前三阶段分解已足够支撑 DDD 和 OPT-Tree 的论证（两者分别优化 Drafter 的深度和形状）。

**代码文件**：`experiments/profiling_timing.py`（321 行）

### 4.3 实验结果

| 阶段 | 平均时延 (ms) | 占比 | P95 (ms) | Std (ms) |
|------|-------------|------|----------|----------|
| ①+③ Drafter Construction + Rejection/KV | 12.46 | **26.6%** | 12.64 | 0.18 |
| ② Target Verify (32-layer tree forward) | 34.24 | **73.4%** | 35.04 | 0.29 |
| **总计** | **46.70** | **100%** | — | — |

> 测试条件：5 组 prompt × 80+ profiling steps = **512 个 inference steps**

**解读**：

- Target Verify 占 73.4% 是预期的——32 层完整 forward 是每个 step 中最重的操作
- Drafter 侧占 **26.6%**（12.46 ms）是一个有意义的优化目标：如果能通过 DDD 减少无效 beam search 步数，这部分的时延可以降低，进而提升整体吞吐

### 4.4 关键结论

> Drafter 的树构造和 KV 更新占据了每个 step 超过 1/4 的时间（12.46 ms，26.6%）。如果能在部分 step 中减少无效扩展（DDD 的目标），这部分开销可以被压缩。这框定了后续实验的分析范围——聚焦于 Drafter 侧的深度决策和节点选择。

---

## 5. 实验二：逐深度 Token 触及率与接受率分析

### 5.1 实验动机

EAGLE-2 的 beam search 固定扩展 depth=5 步，但这隐含了一个假设——"每一步的扩展都具有类似的价值"。如果深层节点只有在极少数 step 中才会被验证路径触及（即前面的 token 全被接受，后面才有机会被验证），那么多余的扩展步数就是在浪费算力。

这个实验测量两个关键指标：
- **条件接受率**：到达某深度的 token，被 target 接受的概率
- **触及率**：在所有 inference step 中，验证路径能走到该深度的比例

### 5.2 实现方法

**核心思路**：每次 verify 后，通过 `tree_position_ids`（每个树节点到根的距离）和 `retrieve_indices`（候选路径到节点索引的映射），追溯被接受和被拒绝的 token 在树中的深度位置。

**数据采集流程**：

```
for each inference step:
    1. tree_decoding() + evaluate_posterior() → best_candidate, accept_length
    2. 利用 tree_position_ids[retrieve_indices[best_candidate, pos]]
       获取接受路径上每个 token 的树深度
    3. 对于 pos = 0..accept_length: 记录为 "accepted at depth d"
    4. 对于 pos = accept_length + 1: 记录为 "rejected at depth d"（d 即是最远被测试的深度）
    5. 聚合到 depth_stats[depth].accepted 和 depth_stats[depth].tested
```

**代码文件**：`experiments/profiling_acceptance.py`（307 行）

### 5.3 实验结果

| 树深度 | 被测试次数 | 被接受次数 | **条件接受率** | **Step 触及率** |
|--------|----------|----------|--------------|---------------|
| 0 (root) | 512 | 512 | 100.0% | 100.0% |
| 1 | 512 | 348 | 68.0% | 100.0% |
| 2 | 333 | 223 | 67.0% | **65.0%** |
| 3 | 156 | 104 | 66.7% | **30.5%** |
| 4 | 75 | 61 | 81.3% | 14.6% |
| 5 | 39 | 26 | 66.7% | **7.6%** |
| 6 | 16 | 14 | 87.5% | 3.1% |

> 测试条件：5 组 prompt × 512 个 inference steps，总计 1643 个 token 被测试

### 5.4 关键发现

**发现一：条件接受率在各深度相对稳定（~67%）**，而非预期中的单调递减。这是因为 EAGLE 的 beam search 在每个深度都通过 top-k 筛选保证了候选质量——能进入某深度的节点，本身就经过了质量过滤。

**发现二（核心）：Step 触及率随深度急剧下降**——仅有 7.6% 的 step 中验证路径能走到 depth 5，仅有 3.1% 能走到 depth 6。这意味着：

> 固定扩展 depth=5/6 的 beam search，在 **92.4% 的 step 中为永远不会被触及的深层节点白白消耗了 drafter 算力**。

### 5.5 与 DDD 改进的因果关系

DDD 的早停机制是：在 beam search 扩展过程中检查 beam 整体置信度 $H = \log\sum_i \exp(\text{logprobsum}[i])$，当 $H < \tau$ 时提前停止。这个机制的有效性由本实验的数据直接支撑：

- 92.4% 的 step 中 depth 5 未被触及 → beam 在大多数 step 中缺乏深层扩展的信心
- $H$ 作为 beam 整体质量的代理指标，正好能检测到这种"信心不足"
- DDD 在这些 step 中提前终止扩展 → 节省了固定深度方案中浪费的算力

> **因果关系**：因为 E2.2 证明 depth ≥ 5 仅在 7.6% 的 step 中被触及，固定扩展深度存在系统性浪费，所以使用 DDD 检测 beam 置信度并自适应早停可以消除这种浪费。

---

## 6. 实验三：树节点利用效率分析

### 6.1 实验动机

EAGLE-2 在 rerank 阶段采用启发式策略：对 expand 出的所有候选节点按"根到该节点的累积 drafter 概率（path confidence）"排序，取 top-N 送入 target verify。但这个策略有一个根本缺陷：

> path confidence 是 drafter 的主观置信度，不等于 target 实际会接受的概率。

如果实测数据表明送入 verify 的 N 个节点中只有极少部分被接受，就说明启发式排序的区分能力有限——即"高 path confidence"不能有效区分"会被接受"和"会被拒绝"的节点。这正是 OPT-Tree 将节点选择形式化为全局优化问题的立论依据。

### 6.2 实现方法

**核心思路**：在每个 inference step 中，统计：
- `N_verify`：送入 target verify 的节点数（= `total_tokens` = 60）
- `N_accepted`：实际被接受的节点数（= `accept_length + 1`）
- 逐深度的 verified 和 accepted 节点计数（通过 `tree_position_ids` 聚合）

**数据采集流程**：

```
for each inference step:
    1. 获取 tree_position_ids（每个树节点的深度）
    2. 统计各深度的节点数 → 累积到 depth_nodes[d].verify
    3. evaluate_posterior → accept_length
    4. 追溯接受路径上每个 token 的深度 → 累积到 depth_nodes[d].accepted
    5. 计算利用率 = N_accepted / N_verify
```

**代码文件**：`experiments/profiling_tree_util.py`（364 行）

### 6.3 实验结果

#### 6.3.1 节点利用效率总览

| 指标 | 数值 |
|------|------|
| 每步送入验证的节点数 (N_verify) | 60 |
| 每步实际接受的节点数 (N_accepted) | **2.52** |
| **节点验证利用率** | **4.2%** |
| 利用率中位数 | 3.3% |
| 利用率 P95 | 9.1% |

#### 6.3.2 逐深度节点分布

| 树深度 | 验证节点数 (总计) | 接受节点数 (总计) | 利用率 |
|--------|-----------------|-----------------|--------|
| 0 (root) | 512 | 512 | 100.0% |
| 1 | 4,817 | 348 | **7.2%** |
| 2 | 15,705 | 223 | 1.4% |
| 3 | 6,444 | 104 | 1.6% |
| 4 | 2,145 | 61 | 2.8% |
| 5 | 798 | 26 | 3.3% |
| 6 | 299 | 14 | 4.7% |
| **合计** | **30,720** | **1,288** | **4.2%** |

> 合计 30,720 = 60 节点/step × 512 steps（一致性校验通过）

### 6.4 关键发现

**60 个验证节点中仅 2.5 个被接受**，意味着 **95.8% 的 target verify 计算资源浪费在不会被接受的节点上**。

虽然 target verify 的并行性使得多验证几个节点的边际成本很低（通过 tree attention 在一个 batch 中完成），但低利用率仍然暗示了问题：如果选入 verify 的节点能更精准地命中"真正会被接受的"节点（这正是 OPT-Tree 的目标），有效接受长度可以提升。

### 6.5 与 OPT-Tree 改进的因果关系

当前启发式 rerank 的局限在于：它按 path confidence 局部排序，无法考虑"某个节点虽然自身置信度稍低但拥有多个高置信度后代"的全局价值。

OPT-Tree 将节点选择形式化为最大化整棵树期望接受长度的问题：

$$\mathbb{E}[\text{acceptance length}] = \sum_{v \in T} \prod_{u \in \text{path}(v)} q(u)$$

通过 over-expand → 全局贪心选 top-N → 保证祖先连通性，OPT-Tree 在相同的验证预算下选出期望收益更高的节点子集。

> **因果关系**：因为 E2.3 证明启发式 rerank 的节点验证利用率仅 4.2%，验证预算分配效率低下，所以使用 OPT-Tree 以最大化全局期望接受长度为目标选择节点。

---

## 7. 瓶颈总览与改进方案

### 7.1 三个实验的发现汇总

```
┌─────────────────────────────────────────────────────────────────┐
│                    前期实验：瓶颈发现总览                           │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 实验一 (E2.1)：时延分解                                    │    │
│  │   Drafter 占 26.6% 总时延 → 值得优化                       │    │
│  │   Target Verify 占 73.4% → 节点选择影响 verify 效率        │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           │                                     │
│          ┌────────────────┴────────────────┐                    │
│          ▼                                 ▼                    │
│  ┌──────────────────────┐    ┌──────────────────────────┐      │
│  │ 实验二 (E2.2)：       │    │ 实验三 (E2.3)：           │      │
│  │ 深度触及率分析         │    │ 节点利用效率              │      │
│  │                      │    │                          │      │
│  │ Depth 5 仅在 7.6%    │    │ 利用率仅 4.2%            │      │
│  │ 的 step 中被触及      │    │ (60 节点 → 2.5 接受)     │      │
│  │                      │    │                          │      │
│  │ → 固定深度系统性浪费   │    │ → 启发式 rerank 低效     │      │
│  └──────────┬───────────┘    └────────────┬─────────────┘      │
│             │                              │                    │
│             ▼                              ▼                    │
│  ┌──────────────────────┐    ┌──────────────────────────┐      │
│  │ 改进一：DDD           │    │ 改进二：OPT-Tree          │      │
│  │ 自适应扩展深度         │    │ 最大化 E[acc_len] 选节点  │      │
│  │ Check beam 置信度 H   │    │ Over-expand → 贪心 →     │      │
│  │ H < τ → 提前停止      │    │ 保证连通子树              │      │
│  │ 均在推理循环内，不重训  │    │ 均在推理循环内，不重训     │      │
│  └──────────────────────┘    └──────────────────────────┘      │
│                                                                 │
│  两个改进正交：DDD 控深度，OPT-Tree 控形状 → 可联合叠加           │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 改进方案概要

| 改进 | DDD | OPT-Tree |
|------|-----|----------|
| **全称** | Dynamic Depth Decoding | Optimal Tree Node Selection |
| **D6 子维度** | 深度决策 | 形状决策 |
| **作用阶段** | Expand（beam search 循环） | Rerank（节点筛选） |
| **核心机制** | 检查 beam logprobsum $H$，$H < \tau$ 时早停 | 最大化 $\mathbb{E}[\text{acc\_len}]$ 贪心选节点 |
| **需要重训？** | 否 | 否 |
| **参考论文** | arXiv 2409.00142 | arXiv 2406.17276 |
| **实现位置** | `cnets.py` `topK_genrate` 的 `for i in range(depth)` 循环内 | `cnets.py` `topK_genrate` 的 `top_scores` 选择逻辑处 |

### 7.3 下一阶段：消融实验（6.9 后）

| 配置 | DDD | OPT-Tree | 说明 |
|------|-----|----------|------|
| Baseline (EAGLE-3) | ✗ | ✗ | 已有：63.8 tok/s |
| DDD-only | ✓ | ✗ | 验证自适应深度的单独贡献 |
| OPT-Tree-only | ✗ | ✓ | 验证最优节点选择的单独贡献 |
| DDD + OPT-Tree | ✓ | ✓ | 验证两者的联合/协同效果 |

---

## 附录 A：实验数据文件索引

| 文件 | 内容 |
|------|------|
| `experiments/E2.1_timing/timing_summary.json` | 时延分解汇总 |
| `experiments/E2.1_timing/timing_raw.json` | 512 step 逐步时延 |
| `experiments/E2.1_timing/figures/timing_breakdown.png` | 时延饼图 + 柱状图 |
| `experiments/E2.1_timing/figures/timing_stacked.png` | 逐步堆叠柱状图 |
| `experiments/E2.1_timing/figures/timing_timeseries.png` | 时延 + 接受长度时间序列 |
| `experiments/E2.2_acceptance/acceptance_summary.json` | 接受率汇总 |
| `experiments/E2.2_acceptance/acceptance_raw.json` | 512 step 逐步记录 |
| `experiments/E2.2_acceptance/figures/acceptance_rate.png` | 接受率 vs 深度柱状图 |
| `experiments/E2.2_acceptance/figures/acceptance_cdf.png` | 接受长度分布直方图 + CDF |
| `experiments/E2.3_tree_util/tree_util_summary.json` | 节点利用率汇总 |
| `experiments/E2.3_tree_util/tree_util_raw.json` | 512 step 逐步记录 |
| `experiments/E2.3_tree_util/figures/tree_utilization.png` | 逐深度 verified vs accepted |
| `experiments/E2.3_tree_util/figures/utilization_distribution.png` | 利用率分布 + 时间序列 |

## 附录 B：实验代码文件索引

| 文件 | 功能 |
|------|------|
| `experiments/config.py` | 所有实验共享的配置、路径、参数管理 |
| `experiments/profiling_timing.py` | E2.1：时延分解，`torch.cuda.Event` 四阶段计时 |
| `experiments/profiling_acceptance.py` | E2.2：逐深度接受率分析，tree_position_ids 深度追溯 |
| `experiments/profiling_tree_util.py` | E2.3：节点利用效率，verified/accepted 逐深度聚合 |
