# 投机解码推理优化 —— 实验规划与逻辑框架

> **项目**：USTC ML System 2026 Final Project
> **方案**：基于 EAGLE-3 的 DDD（动态深度）+ OPT-Tree（最优树形状）推理优化
> **模型**：LLaMA-3.1-8B-Instruct + EAGLE-3 官方 checkpoint
> **硬件**：NVIDIA RTX 3090 (24GB) × 5
> **目标**：本文件为开题报告 (6.9) 和终稿报告 (6.19) 提供完整的叙事框架与实验设计

---

## 目录

1. [背景与问题定义](#1-背景与问题定义)
2. [实验全景：逻辑关系总图](#2-实验全景逻辑关系总图)
3. [阶段零：前置基础 —— Lab1 & Lab2 已有结论](#3-阶段零前置基础--lab1--lab2-已有结论)
4. [阶段一：Baseline 复现与管线验证](#4-阶段一baseline-复现与管线验证)
5. [阶段二：瓶颈定位实验（6.9 开题核心）](#5-阶段二瓶颈定位实验69-开题核心)
6. [阶段三：改进实现与消融实验](#6-阶段三改进实现与消融实验)
7. [阶段四：补充分析与局限性](#7-阶段四补充分析与局限性)
8. [实验总表](#8-实验总表)
9. [报告/PPT 叙事线](#9-报告ppt-叙事线)

---

## 1. 背景与问题定义

### 1.1 核心问题：LLM 推理为什么慢？

大语言模型（LLM）的自回归生成过程是一个**逐 token 串行**的过程：每生成一个新的 token，都需要将整个模型的所有权重参数从 GPU 显存（HBM）搬运到计算单元（SM）中执行一次完整的前向传播。这个过程的瓶颈特性是：

- **算术强度（Arithmetic Intensity）极低**：每次 decode step 的计算量（FLOPs）与数据搬运量（Bytes）之比远小于 1
- **硬件趋势恶化**：GPU 算力（FLOPS）的增长速度远超显存带宽（HBM Bandwidth）的增长速度，导致 memory wall 问题日趋严重
- **实际利用率极低**：在 batch=1 的单用户推理场景中，GPU 的算力利用率（MFU）通常只有 **几个百分点**

这个结论不是理论推测——我们在 **Lab2 Task 3（Roofline 分析）** 中已经通过实测数据验证：在 RTX 3090 上，Qwen3-4B 模型的 Decode 阶段落在 Roofline 图的 **memory-bound 区域**，GPU 大部分时间在"等待数据"而非"执行计算"。

### 1.2 投机解码的基本原理

投机解码（Speculative Decoding, SD）是解决上述问题的一种**无损加速**方法。其核心思想是"猜测与验证"：

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Draft（草稿阶段）                                       │
│  用一个极便宜的 drafter（1层Transformer，~1% 参数量）               │
│  自回归地猜出未来 K 个 token 的草稿                                │
│                                                                  │
│  Step 2: Verify（验证阶段）                                       │
│  将 "前缀 + 草稿" 一次性送入 target（完整32层模型）                  │
│  并行计算 K+1 个位置的 logits                                      │
│                                                                  │
│  Step 3: Accept/Reject（接受/拒绝）                                │
│  用 modified rejection sampling 逐位置验证草稿                     │
│  接受最长正确前缀，在第一个被拒处由 target 重新采样                  │
└─────────────────────────────────────────────────────────────────┘
```

**为什么能加速**：一次 target forward 原本只能生成 1 个 token，现在可以验证 K 个 token。如果平均接受 4 个，理论的加速比就是 4×。本质上是将"搬运一次权重"的带宽成本**摊薄**到多个 token 上。

**两个关键性质**：

| 性质 | 说明 |
|------|------|
| **无损（Lossless）** | 在拒绝采样的数学意义上，输出分布与直接使用 target 模型解码完全等价 |
| **期望接受长度 > 1** | 一次 target forward 可接受多个 token，加速比由此而来 |

### 1.3 EAGLE-3 的技术路线

EAGLE 系列是投机解码领域的代表性工作，三代演进概括如下：

| 版本 | 核心贡献 | 与我们的关系 |
|------|---------|------------|
| **EAGLE-1** (ICML'24) | 将 drafter 从 token 空间移到 feature 空间（target 倒数第二层 hidden state 上做 AR） | 理论奠基 |
| **EAGLE-2** (EMNLP'24) | 引入 context-aware dynamic draft tree，推理时在线调整树形状（expand + rerank） | **我们的改进直接基于 EAGLE-2 的树构造流程** |
| **EAGLE-3** (ICLR'25) | 重新设计 drafter：直接 token prediction + 多层 feature fusion + Training-Time Test | **我们复用 EAGLE-3 的预训练 checkpoint 作为 baseline** |

关键理解：EAGLE-2 的 dynamic tree 构造逻辑（expand 阶段的 beam search + rerank 阶段的节点筛选）是纯推理侧的技术，**与 drafter 具体架构解耦**。EAGLE-3 虽然换用了更强大的 drafter，但推理时的树构造流程与 EAGLE-2 完全兼容。因此：

> **我们的策略**：使用 EAGLE-3 的预训练 drafter checkpoint（接受率更高）+ 在 EAGLE-2 风格的树构造推理循环上做改进。

### 1.4 投机解码的 7 个组件维度（D1-D7）

课程主页将投机解码系统分解为 7 个独立的组件维度，每个维度对应一个特定的瓶颈：

| 维度 | 瓶颈 | 改动是否需要重训 drafter |
|------|------|------------------------|
| D1 模型来源 | Drafter 单步成本 | 通常需要 |
| D2 生成范式 | Drafter 串行性 | 通常需要 |
| D3 信息耦合 | 接受率理论上限 | 通常需要 |
| D4 对齐方式 | 接受率 | 需要训练 |
| D5 输出结构 | 鲁棒性 / 被接受长度 | 通常不需要 |
| **D6 长度/形状决策** | **成本-接受率平衡** | **通常不需要** ⭐ |
| D7 Verify 规则 | 被接受长度 | 不需要 |

**我们的选择**：聚焦 **D6 维度**下的两个 ⭐ 方向（DDD + OPT-Tree），因为它们的改动集中在推理循环的树构造逻辑中，**不需要重新训练 drafter**，可以直接复用 EAGLE-3 的已发布 checkpoint。

### 1.5 我们的两个优化点

#### DDD（Dynamic Depth Decoding）—— D6 维度

**针对的瓶颈**：EAGLE-2 的 beam search 固定走 5 步扩展（depth=5），但不同上下文的预测难度差异巨大——"Thank you"后面接"very much"很容易预测（2-3 步就够了），但复杂推理中的中间步骤则需要更深的探索。

**改进方案**：在 beam search 扩展过程中，定期（第 5/7/9 步）检查 beam 整体的对数概率和 $H = \log\sum_i \exp(\text{logprobsum}[i])$。若 $H$ 低于阈值 $\tau$，说明 beam 对未来预测已失去信心，提前停止扩展，避免在低收益位置浪费 drafting 算力。

**参考论文**：[arXiv 2409.00142](https://arxiv.org/pdf/2409.00142)

#### OPT-Tree —— D6 维度

**针对的瓶颈**：EAGLE-2 在 rerank 阶段按每个节点的"路径累积 drafter 概率"排序，取 top-N 送入 target verify。但这种启发式排序**不一定最大化期望接受长度**——一个置信度稍低但有很多高置信度后代的节点，可能比一个孤立的最高置信度节点更有价值。

**改进方案**：将"选哪些节点送入验证"形式化为最大化整棵树期望接受长度 $\mathbb{E}[\text{acceptance length}] = \sum_{v \in T} \prod_{u \in \text{path}(v)} q(u)$ 的优化问题。Over-expand 出一棵大于预算的候选树，然后贪心选出全局最优的连通子树。

**参考论文**：[arXiv 2406.17276](https://arxiv.org/pdf/2406.17276)

### 1.6 两个改进的正交性与协作

DDD 和 OPT-Tree 分别作用于 EAGLE 树构造流程的**不同阶段**，不存在冲突：

```
EAGLE 推理 Step 的树构造流程：

  Target Prefill → hidden states
       │
       ▼
  ┌──────────────────────────────────────┐
  │  Phase 1: Expand（扩展阶段）          │
  │  Beam search，每步 top-k 展开          │
  │                                      │
  │  ★ DDD 在这里：在第 5/7/9 步检查      │
  │    beam 置信度，低于阈值 → 提前停止    │
  │                                      │
  │  控制变量：树的深度（走几步）            │
  │  输出：深度可变的候选树                 │
  └──────────────────────────────────────┘
       │
       ▼
  ┌──────────────────────────────────────┐
  │  Phase 2: Select（筛选阶段）           │
  │  从候选节点中选择最终送入 verify 的节点  │
  │                                      │
  │  ★ OPT-Tree 在这里：用最大化          │
  │    E[acceptance length] 贪心选节点    │
  │                                      │
  │  控制变量：树的宽度/形状（留哪些）       │
  │  输出：连通最优子树 + attention mask    │
  └──────────────────────────────────────┘
       │
       ▼
  Target Verify → Accept/Reject
```

**DDD 决定"树长多深"，OPT-Tree 决定"树留哪些节点"**。两者串行接力，各司其职，叠加效果预期 ≥ 各自单独效果之和。

---

## 2. 实验全景：逻辑关系总图

### 2.1 总体逻辑

整个实验体系的逻辑可以用一句话概括：

> **从宏观瓶颈出发 → 定位到微观组件 → 量化浪费 → 设计针对性改进 → 消融验证**

```
┌──────────────────────────────────────────────────────────────────────┐
│                       实验全景：逻辑关系图                              │
│                                                                      │
│  ┌─────────────────────────┐                                         │
│  │  阶段零：前置基础         │  Lab1 & Lab2 已有结论                    │
│  │  E0.1 Roofline          │                                         │
│  │  E0.2 Kernel Profiling  │  "Decode 是 memory-bound"               │
│  │  E0.3 计时方法论         │  "GPU 利用率仅几个百分点"                  │
│  └────────────┬────────────┘                                         │
│               │                                                      │
│               │  提供理论起点和实验方法                                  │
│               ▼                                                      │
│  ┌─────────────────────────┐                                         │
│  │  阶段一：Baseline 建立   │  回答："投机解码有没有用？"                │
│  │  E1.1 Pure AR           │  40.0 tok/s（性能下界）                   │
│  │  E1.2 EAGLE-3           │  63.8 tok/s（1.60×，但远低于论文 3-5×）   │
│  └────────────┬────────────┘                                         │
│               │                                                      │
│               │  "加速比为什么不及论文？瓶颈在哪里？"                      │
│               ▼                                                      │
│  ┌──────────────────────────────────────────────────────┐            │
│  │  阶段二：瓶颈定位（6.9 开题核心）                       │            │
│  │                                                      │            │
│  │  ┌─────────────────────────────────────────────┐     │            │
│  │  │ E2.1 时延分解                                │     │            │
│  │  │ 问题："时间花在哪？"                           │     │            │
│  │  │ 方法：torch.cuda.Event 分解四个阶段           │     │            │
│  │  │ 发现：Drafter+Rej/KV 占 26.6% 总时延        │     │            │
│  │  │ 作用：为后续两个实验框定范围                    │     │            │
│  │  └──────────────────┬──────────────────────────┘     │            │
│  │                     │                                │            │
│  │        ┌────────────┴────────────┐                   │            │
│  │        ▼                         ▼                   │            │
│  │  ┌──────────────────┐  ┌──────────────────┐          │            │
│  │  │ E2.2 逐深度接受率  │  │ E2.3 节点利用率   │          │            │
│  │  │                  │  │                  │          │            │
│  │  │ 问题：           │  │ 问题：            │          │            │
│  │  │ "固定深度的每一步  │  │ "启发式 rerank 的 │          │            │
│  │  │  价值相等吗？"    │  │  效率如何？"      │          │            │
│  │  │                  │  │                  │          │            │
│  │  │ 发现：           │  │ 发现：            │          │            │
│  │  │ Depth 5 仅 7.6%触及 │  │ 节点利用率仅 4.2% │          │            │
│  │  │ 触及率断崖下降    │  │ 95.8%验证预算浪费  │          │            │
│  │  │                  │  │                  │          │            │
│  │  │ → DDD 的证据     │  │ → OPT-Tree 的证据│          │            │
│  │  └──────────────────┘  └──────────────────┘          │            │
│  └──────────────────────────────────────────────────────┘            │
│               │                                                      │
│               │  瓶颈 → 改进的因果关系：                                │
│               │  E2.2 → D6瓶颈(深度) → DDD                          │
│               │  E2.3 → D6瓶颈(形状) → OPT-Tree                     │
│               ▼                                                      │
│  ┌──────────────────────────────────────────────────────┐            │
│  │  阶段三：改进验证与消融                                │            │
│  │                                                      │            │
│  │  E3.1 DDD-only  ──┐                                  │            │
│  │  E3.2 OPT-only  ──┤  四组对比（含 joint）              │            │
│  │  E3.3 Joint      ──┘                                  │            │
│  │  E3.4 超参数敏感性                                     │            │
│  │                                                      │            │
│  │  验证：改进是否有效？两个改进是否互补？                     │            │
│  └────────────┬─────────────────────────────────────────┘            │
│               │                                                      │
│               │  "DDD+OPT-Tree 联合提升了 X% tokens/s"                │
│               ▼                                                      │
│  ┌──────────────────────────────────────────────────────┐            │
│  │  阶段四：补全与边界                                    │            │
│  │  E4.1 分布一致性 → 确认 lossless                       │            │
│  │  E4.2 场景泛化性 → 找出方法不适用的场景                   │            │
│  └──────────────────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 实验之间的依赖关系

```
E0.1 (Roofline)
  │
  ├──→ 为整个项目提供了"为什么需要投机解码"的理论基础
  │
  └──→ E1.1 (Pure AR) → E1.2 (EAGLE-3 Baseline)
         │                  │
         │                  ├──→ 建立性能下界 (40 tok/s)
         │                  └──→ 建立改进起点 (63.8 tok/s, 1.60×)
         │
         └──→ E2.1 (时延分解)
                │
                ├──→ 量化各阶段开销，框定优化范围
                │
                ├──→ E2.2 (逐深度接受率)
                │     │
                │     └──→ 直接支撑 DDD 的动机与设计
                │
                └──→ E2.3 (节点利用率)
                      │
                      └──→ 直接支撑 OPT-Tree 的动机与设计
                             │
                             ▼
                      E3.1 (DDD-only)  ──┐
                      E3.2 (OPT-only)  ──┤
                      E3.3 (Joint)     ──┤ 消融矩阵
                      E3.4 (超参数)     ──┘
                             │
                             ▼
                      E4.1 (lossless) ── 性质保证
                      E4.2 (泛化性)   ── 边界分析
```

**关键原则**：E2.1/E2.2/E2.3 三个实验之间的依赖不是线性的——它们是**三个独立的视角**，共同描绘 EAGLE 管线的性能画像。但 E2.1 需要最先做，因为它框定了"哪里的开销大"，为 E2.2 和 E2.3 的分析提供上下文。

---

## 3. 阶段零：前置基础 —— Lab1 & Lab2 已有结论

> **核心定位**：以下结论来自 Lab1/Lab2 个人实验，是 Final Project 的**理论起点**。在报告和 PPT 中直接引用，作为"为什么需要投机解码"的证据链，不需要为此重复做实验。

### 3.1 背景：这一阶段在整个项目中的角色

在任何一个 MLSys 项目中，第一步都是**回答"为什么"**——为什么现有的方案不够好？为什么需要引入新技术？Lab1 和 Lab2 所做的 profiling 工作恰好为这个问题提供了数据支撑。它们证明了一个核心事实：

> **LLM 的 Decode 阶段是 memory-bound 的——GPU 大部分时间在等数据而非算数据。**

这个事实是整个 Final Project 存在的理由：正因为有这个问题，投机解码才有价值。在报告中，这一阶段的结论应该放在"Problem Statement"或"Motivation"章节。

---

### E0.1 Decode 阶段是 Memory-Bound（Lab2 Roofline 分析）

**实验来源**：Lab2 Task 3

**为什么要做这个实验**：Roofline 模型是判断一个 workload 是"计算瓶颈"还是"带宽瓶颈"的标准工具。对于 LLM 的 decode 阶段，由于每次 forward 的计算量（FLOPs）相对固定但权重参数量巨大，直觉上它应该是 memory-bound，但**需要用实测数据来验证**，而不是假设。

**实验验证了什么**：
- LLM 自回归解码的算术强度 < 1 FLOP/Byte
- 在 RTX 3090 的 Roofline 图上，Decode 阶段的工作点落在 **memory-bound 区域**（roofline 拐点左侧）
- GPU 算力利用率（MFU）仅几个百分点

**在 Final Project 叙事中的位置**：
> 这个实验提供了 **"为什么需要投机解码"** 的实证基础。正因为 decode 是 memory-bound，每次 forward 的大部分时间花在搬运权重上，投机解码"一次 forward 验证多个 token"的策略才有价值——它本质上是把带宽成本摊薄到多个 token 上。报告中，这个结论应放在 Background 或 Motivation 章节。

**产出物**（已有，直接引用）：Roofline 图，Decode 阶段 FLOPs 利用率数据

---

### E0.2 Kernel 级时延构成（Lab2 torch.profiler）

**实验来源**：Lab2 Task 2

**为什么要做这个实验**：Roofline 分析从理论上告诉我们瓶颈在带宽，但 torch.profiler 的 kernel 级分析可以**从 CUDA kernel 层面验证这一判断**——如果 `aten::mm`（矩阵乘法）等 GEMM 操作主导了 CUDA 时间，说明计算确实是主要活动，但数据搬运的开销（表现为 kernel launch overhead 和内存拷贝）也在总时间中占比显著。

**实验验证了什么**：
- Decode 阶段的 CUDA 时间中，`aten::mm` 内核占据主导
- Top CUDA kernels：`ampere_bf16_s16816gemm` 等 GEMM 操作
- 每次 decode step 的时延由 Attention 和 FFN 中的线性层计算共同构成

**在 Final Project 叙事中的位置**：
> EAGLE 的 target verify 阶段本质上执行的仍是这些 GEMM kernel，但关键区别在于：纯 AR 每次只处理 1 个 token（小矩阵乘法，GPU 利用率低），而 EAGLE 通过 tree attention 一次性处理整个 draft tree（更大 batch，更高 GPU 利用率）。这个对比可以在报告的技术背景部分使用。

**产出物**（已有，直接引用）：Decode top-10 CUDA kernels 表

---

### E0.3 TTFT/TBT 测量方法论（Lab1 Prefill/Decode 拆分）

**实验来源**：Lab1 Step 3-4

**为什么要做这个实验**：在 Lab1 中，我们学习了如何用 `torch.cuda.Event` 精确测量 GPU kernel 的执行时间，以及如何分离 Prefill 和 Decode 阶段的时延。这些**方法论**——而非具体的实验结果——直接复用到 Final Project 中。

**实验验证了什么**：
- 掌握了 `torch.cuda.Event` 的精确 GPU 计时方法
- 掌握了 Prefill（TTFT）和 Decode（TBT）阶段的分离测量技巧
- 掌握了输入长度、输出长度、batch size 等参数的系统性扫描方法

**在 Final Project 叙事中的位置**：
> 方法论直接复用到 E2.1（EAGLE 管线时延分解）中。EAGLE 的推理管线比纯 AR 更复杂（多了 drafter、tree construction、rejection sampling 等步骤），但计时工具和方法是一致的。

**产出物**（已有，直接引用）：Prefill/Decode 时延拆分表，参数扫描方法论

---

### 阶段零小结

| 前置结论 | 来源 | 在 Final Project 中的角色 |
|---------|------|------------------------|
| Decode 是 memory-bound | Lab2 Roofline | **理论起点**：为什么需要投机解码 |
| GEMM kernel 主导 CUDA 时间 | Lab2 Profiler | **技术背景**：投机解码如何提升 GPU 利用率 |
| torch.cuda.Event 计时方法论 | Lab1 | **实验工具**：复用到 EAGLE 管线时延分解 |

---

## 4. 阶段一：Baseline 复现与管线验证

> **核心定位**：建立性能的"上下界"——Pure AR 是下界（最慢），EAGLE-3 是当前最优。后续所有改进的效果都在这两个参照系内衡量。这一阶段也验证我们的实验环境（模型、硬件、代码）是否正确配置。

### 4.1 背景：为什么需要两个 Baseline？

在报告的逻辑中，E1.1（Pure AR）和 E1.2（EAGLE-3）各自扮演不同角色：

- **E1.1 Pure AR**：回答"如果不做任何优化，性能是什么样的？"——这是性能的**绝对下界**，投机解码的加速比以此为分母
- **E1.2 EAGLE-3**：回答"现有的最优方案（在我们硬件上）表现如何？"——这是改进的**相对起点**，DDD 和 OPT-Tree 的效果以此为对照组

两者的比值（E1.2 ÷ E1.1）就是 baseline 加速比。我们实测的这个值（~1.60×）远低于 EAGLE-3 论文报告的 3-5×，这本身就暗示了——在特定硬件和模型配置下，EAGLE 管线中可能存在未被充分利用的优化空间，这正是我们后续实验要探索的。

---

### E1.1 Pure AR Baseline（纯自回归解码）

**要回答的问题**：在不做任何加速优化的情况下，LLaMA-3.1-8B-Instruct 在单张 RTX 3090 上的推理速度是多少？

**为什么需要这个实验**：这是整个项目的**性能下界**。所有加速比的分子（EAGLE/DDD/OPT-Tree 的 tokens/s）都要除以这个分母。没有这个基准，任何"加速"的声称都没有意义。

**方法**：
- 使用 HuggingFace `model.generate()` 进行标准自回归解码
- 目标模型：`meta-llama/Llama-3.1-8B-Instruct`（已从 ModelScope 下载至本地）
- 测试 prompt：从 MT-Bench 和 ShareGPT 采样，覆盖对话、写作、推理、代码等不同任务类型
- `temperature=0.0`（greedy decoding），`max_new_tokens=256`
- 测量指标：总耗时、生成 token 数、计算 tokens/s

**实验前我们不知道什么**：
- LLaMA-3.1-8B-Instruct 在 RTX 3090 上的确切推理速度
- 不同 prompt 类型下速度是否有显著差异

**实验后我们能回答什么**：
- Pure AR 的速度基线（用于计算所有后续实验的加速比）
- 与 EAGLE-3 对比可以量化投机解码的实际收益

**预期结果**：

| 指标 | 预期值 | 依据 |
|------|--------|------|
| LLaMA-3.1-8B 纯 AR 吞吐 | ~38-42 tok/s | 8B 模型 + 3090 的经验数据 |
| 单 token decode 时延 | ~24-26 ms | 1000/40 ≈ 25ms |

**已完成**：✅ 已测得 **40.0 tok/s**

---

### E1.2 EAGLE-3 Baseline（投机解码复现）

**要回答的问题**：在完全相同硬件上，使用 EAGLE-3 官方 checkpoint，投机解码的实际加速比是多少？

**为什么需要这个实验**：这是**改进的起点**。DDD 和 OPT-Tree 都是 EAGLE-3 推理管线的增强，它们的贡献必须在 EAGLE-3 自身的基础上衡量。同时，如果 EAGLE-3 的复现结果与论文报告差异很大，说明存在我们尚未理解的系统瓶颈。

**方法**：
- 使用 SafeAILab/EAGLE 仓库 + `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` 官方 checkpoint
- 树参数使用仓库默认：`total_token=60, depth=5, top_k=10`
- 相同测试 prompt 集，`temperature=0.0`
- 测量端到端吞吐、平均 acceptance length、显存占用

**实验前我们不知道什么**：
- EAGLE-3 在 RTX 3090 + LLaMA-3.1-8B 组合上的实际加速比
- 论文报告的 3-5× 加速是否在我们的硬件上复现
- 平均 acceptance length 在默认参数下是多少

**实验后我们能回答什么**：
- EAGLE-3 baseline 的绝对性能（tokens/s）
- 相对 Pure AR 的加速比
- 与论文报告值的差距（若有），提示可能存在优化空间

**预期结果**：

| 指标 | 预期值 |
|------|--------|
| EAGLE-3 吞吐 | ~60-80 tok/s |
| 相对 Pure AR 加速比 | ~1.5-2.0× |
| 平均每次 accept 的 token 数 | ~3-4 |
| 显存占用 | ~18-20 GB |

**已完成**：✅ 已测得 **63.8 tok/s（1.60× 加速比）**，接受长度 ~3.06

**实验后分析**：
- 1.60× 加速比虽然确认了投机解码的有效性，但明显低于论文报告的 3-5×
- 这个差距正是我们第二阶段实验要解释的：是什么因素在限制加速比？
- 可能的原因：树参数未针对该模型调优、某些阶段的时延占比过高、验证路径在深层极少被触及

---

### 阶段一小结

| 实验 | 状态 | 核心产出 | 在叙事中的角色 |
|------|------|---------|-------------|
| E1.1 Pure AR | ✅ | 40.0 tok/s | 性能下界，加速比的分母 |
| E1.2 EAGLE-3 | ✅ | 63.8 tok/s, 1.60× | 改进起点，引出"为什么 1.60× 不是 3×"的问题 |

**从阶段一到阶段二的过渡**：
> E1.2 证明投机解码有效（1.60× > 1.00×），但也揭示了一个问题：实际加速比远低于论文。这自然引出下一个问题——**管线中哪个环节在拖后腿？** 这就是阶段二的瓶颈定位实验要回答的。

---

## 5. 阶段二：瓶颈定位实验（6.9 开题核心）

> **核心定位**：这三个实验是开题报告的**核心产出**。它们各自定位一个具体的性能瓶颈，每一个实验的结果都直接指向一个特定的改进方案。三个实验合在一起，构成了一条从"问题是什么"到"我们怎么做"的完整逻辑链。

### 5.1 整体设计思路

**为什么需要三个实验而不是一个**：投机解码的管线涉及三个关键决策：（1）drafter 扩展多少步（深度），（2）选择哪些节点送验证（宽度/形状），（3）如何验证（验证规则）。这三个决策对应不同的瓶颈维度，需要分开测量才能分别归因。

**三个实验的逻辑分工**：

```
E2.1 时延分解 → 宏观画像：哪个阶段开销最大？
  │
  ├── 如果 Target Verify 占比最高（预期 55-70%）
  │   → 但 Target Verify 的开销与我们选择的 D6 维度不直接相关
  │   → E2.1 的主要价值是框定 Drafter Expand 的占比
  │   → 引导我们聚焦 Drafter 侧的优化（均在 D6 维度）
  │
  ├── E2.2 逐深度接受率 → 微观诊断：每一步的收益递减吗？
  │   → 如果深层触及率断崖下降 → D6 瓶颈(深度) → DDD
  │
  └── E2.3 节点利用率 → 微观诊断：节点选择有效率吗？
      → 如果利用率低 → D6 瓶颈(形状) → OPT-Tree
```

---

### E2.1 EAGLE 管线时延分解

**要回答的问题**：在一次完整的 EAGLE 推理 step 中，Drafter Expand、Tree Rerank、Target Verify、Rejection Sampling 四个阶段各自消耗了多少时间？

**为什么需要这个实验**：这是**瓶颈定位的第一步**——你必须先知道时间花在哪里，才能判断哪里值得优化。如果 Drafter Expand 只占 5% 的时间，那优化它没什么意义；如果它占 30%，那就有 30% 的提升空间。

**实验前的假设**：基于对 EAGLE 代码的理解，我们预期：
- Target Verify（32 层完整 forward）是最大开销 → 55-70%
- Drafter+Rej/KV（实测 12.5 ms）是第二大开 → 26.6%
- Tree Rerank 和 Rejection Sampling 是轻量 CPU/逻辑操作 → < 10%

但**假设需要数据验证**。在特定硬件（3090，显存带宽 936 GB/s）和特定模型（8B，FP16）下，各阶段的实际占比可能与直觉不符。

**方法**：
- 在 `eagenerate` 循环中插入 `torch.cuda.Event` 精确计时点
- 分解每个 inference step 为四个阶段：

```
┌─────────────────────────────────────────────────────────────┐
│ ① Drafter Expand (beam search × depth 步)                   │
│    ├── drafter AR forward × depth（1层 Transformer）         │
│    ├── top-K selection（topk on drafter logits）             │
│    └── tree_mask 增量构建                                    │
│                                                              │
│ ② Tree Rerank（节点排序与筛选）                                │
│    └── 按 scores_list 取全局 top (total_tokens) 个节点       │
│                                                              │
│ ③ Target Verify（tree attention forward）                    │
│    └── 完整 32 层 forward on draft tree（一次性并行打分）      │
│                                                              │
│ ④ Rejection Sampling + KV Cache Update                      │
│    ├── evaluate_posterior（逐 token 拒绝采样）                │
│    └── past_key_values 拷贝与更新                            │
└─────────────────────────────────────────────────────────────┘
```

- 采样 50+ inference steps，计算各阶段的 mean / std / P50 / P95
- 在多组不同 prompt 上重复，交叉验证

**这个实验能告诉我们什么**：
- 宏观上，哪些阶段值得优化（占比 > 10%）
- 哪些阶段不值得优化（占比 < 5%，优化收益有限）
- E2.1 的结果将决定我们后续的实验重点放在 Drafter 侧（Expander + Rerank）还是 Verify 侧

**实测结果**（2026-06-04，LLaMA-3.1-8B-Instruct + EAGLE-3, RTX 3090）：

| 阶段 | 实测时延 | 占比 |
|------|---------|------|
| ①+④ Drafter Construction + Rejection/KV | 12.46 ms | **26.6%** |
| ③ Target Verify (32-layer tree forward) | 34.24 ms | **73.4%** |
| **总计** | **46.70 ms/step** | **100%** |

> 注：Rejection Sampling（`evaluate_posterior`）实际仅占 0.2 ms（0.4%），KV cache 更新与 `topK_genrate` 耦合在一起无法分离。因此报告中将 ① 和 ④ 合并为"Drafter + Rejection/KV"阶段。

**如果实际结果与预期不符怎么办**：这也是有价值的信息。例如：
- 如果 Drafter 占比远高于预期 → 说明 drafter 在小 batch（top_k=10）下 GPU 利用率极低，DDD 的收益会更大
- 如果 Target Verify 占比高达 80% → 说明 verify 侧的 batch 效率差，可能需要考虑调整 tree 大小

**产出物**：各阶段时延饼图、按 step 的时序堆叠柱状图

**与下一实验的衔接**：
> E2.1 告诉我们 Drafter 侧占 ~27% 的总时延。但这 27% 是否每一分都花得值？固定 5 步的 beam search 中，每一步的价值相等吗？→ 引出 E2.2

---

### E2.2 逐深度 Token 接受率分析

**要回答的问题**：在 draft tree 中，不同深度的 token 被 target 接受的概率是多少？以及更重要的是——**在多少 inference step 中，验证路径能走到该深度？**

**为什么需要这个实验**：这是 **DDD 改进的立论之本**。EAGLE-2 的 beam search 固定扩展 depth=5 步，但这隐含着"每一步的 token 都有类似的价值"的假设。这个实验帮助验证：固定深度扩展是否存在系统性浪费？

**实验前的假设**：
直觉上，drafter 的自回归误差是累积的，实验计划预期观察到"接受率随深度单调递减"的模式。但实际数据揭示了一个**不同的、更值得优化的浪费模式**。

**方法**：
- 在每次 verify 后，通过 `tree_position_ids` 和 `retrieve_indices` 追溯被接受/被拒绝 token 的树深度
- 统计 512 steps × 5 prompts 的数据
- 记录两个关键指标：**(1) 条件接受率**（到达该深度的 token 被接受的概率）和 **(2) 触及率**（验证路径能走到该深度的 step 占比）

**实测结果**（2026-06-04，LLaMA-3.1-8B-Instruct + EAGLE-3, RTX 3090）：

```
Depth │ Tested  │ Accepted │ 条件接受率 │ Steps触及率
──────┼─────────┼──────────┼───────────┼────────────
  0   │   512   │   512    │  100.0%   │ 100.0%  (root)
  1   │   512   │   348    │   68.0%   │ 100.0%
  2   │   333   │   223    │   67.0%   │  65.0%
  3   │   156   │   104    │   66.7%   │  30.5%
  4   │    75   │    61    │   81.3%   │  14.6%
  5   │    39   │    26    │   66.7%   │   7.6%  ← 关键发现
  6   │    16   │    14    │   87.5%   │   3.1%
```

**关键发现（与实验计划预期的差异）**：

实验计划预期"接受率随深度单调递减（如 depth 5 降至 10-20%）"。但实测数据显示：

1. **条件接受率在各深度相对稳定（~67%）**：beam search 在每个深度已经通过 top-k 筛选保证了候选质量，所以到达某深度的 token 被接受的概率基本恒定。
2. **但触及率急剧下降**：在 **92.4% 的 inference step 中，验证路径根本没有走到 depth 5**；在 **96.9% 的 step 中，验证路径没有走到 depth 6**。

> **核心洞察**：问题不在于"深层的 token 质量差"，而在于 **"绝大多数 step 根本不需要走到深层"**。固定扩展 depth=5/6 意味着 beam search 在 >90% 的 step 中为永远不会被触及的深层节点白白消耗了 drafter 算力。

**这对 DDD 论证的意义**（比实验计划预期的更强）：

DDD 在 beam search 扩展过程中检查 beam 整体置信度 $H = \log\sum_i \exp(\text{logprobsum}[i])$。当 beam 对未来预测失去信心（$H < \tau$）时，意味着后续扩展的节点被接受的概率很低 → 继续扩展极大概率白做。实测数据中深度 5/6 仅 7.6%/3.1% 的触及率证明：beam 在大多数 step 中缺乏深层次扩展的信心，DDD 的早停机制正好能识别并避免这些浪费。

**因果关系**：

> **因为 E2.2 的数据证明** depth ≥ 5 的节点仅在 7.6% 的 step 中被触及，固定扩展深度导致 beam search 在 >90% 的 step 中为无效深层节点消耗算力，
> **所以 我们提出 DDD**：在 beam 置信度 $H < \tau$ 时提前停止扩展，将这些 step 的 drafting 深度从固定的 5-6 步压缩到有效深度。

**与实验计划预期不同的原因分析**：
- EAGLE 的 beam search 已经通过 top-k 筛选保证各深度候选质量，因此**条件接受率**的递减并不明显
- 但 beam 自身积累的误差导致**触及深层的机会**急剧衰减——这正是 DDD 能捕获的信号（通过 logprobsum 降低反映 beam 质量的衰减）

**产出物**：接受率 vs 深度柱状图（含触及率辅助折线），接受长度分布直方图 + CDF

**与下一实验的衔接**：
> E2.2 告诉我们 drafter 扩展的**深度方向**存在浪费（深层节点极少被触及）。但即使所有深度的 token 都被扩展出来，最终只有 ~59 个节点能被送入 verify——这些节点的**选择方式**是否最优？→ 引出 E2.3

---

### E2.3 树节点利用效率分析

**要回答的问题**：从 beam search 扩展出的全量节点中，经过启发式 rerank 筛选后送入 verify 的 top-N 个节点，有多少最终被 target 接受了？启发式排序（按累积 path confidence）与真实的接受结果之间，有多大偏差？

**为什么需要这个实验**：这是 **OPT-Tree 改进的立论之本**。EAGLE-2 的 rerank 策略是：对 expand 出的所有候选节点，按"根到该节点的累积 drafter 概率（path confidence）"排序，取 top-N 送入 verify。但这个启发式排序有一个根本缺陷：

> path confidence 是 drafter 的**主观置信度**，不等于 target 实际会接受的概率。

一个 drafter 非常自信的节点（高 path conf）可能被 target 拒绝（因为 drafter 和 target 的分布偏差），而一个 drafter 不是最自信但"含金量高"的节点（其后代有多个高置信节点）可能被遗漏。

如果实测数据能证明这个假设——"高 path confidence ≠ 高接受率"——那 OPT-Tree 将节点选择形式化为最大化 E[acceptance length] 的全局优化问题，就比局部启发式排序有坚实的改进依据。

**方法**：
- 在 `topK_genrate` 的 rerank 阶段，记录三组数据：
  - `N_expand`：beam search 实际扩展出的总节点数（取决于 depth 和 top_k）
  - `N_verify`：rerank 后选入 verify 的节点数（= total_tokens - 1 = 59）
  - `N_accepted`：verify 阶段通过 rejection sampling 后实际被接受的节点数（即 accept_length）
- 计算关键转化率：`验证利用率 = N_accepted / N_verify`，`扩展效率 = N_accepted / N_expand`
- 对被拒节点和被接受节点，分别统计它们的 path confidence 分布，看是否有显著可分性

**这个实验能告诉我们什么**：
- 当前的节点选择策略是否有显著浪费（验证利用率很低）
- path confidence 是否真的能区分"会被接受"和"会被拒绝"的节点（如果两组分布高度重叠，说明启发式排序的区分能力弱）
- 节点选择是否存在结构性偏差（例如总是偏好浅层节点而忽略深层高价值节点）

	**实测结果**（2026-06-04，LLaMA-3.1-8B-Instruct + EAGLE-3, RTX 3090）：

	| 指标 | 实测值 | 解读 |
	|------|--------|------|
	| N_verify（每步送入验证） | 60（固定） | total_tokens |
	| N_accepted（每步真实接受） | **2.52** | 含 root token，即平均 1.52 个 draft token |
	| **节点验证利用率** | **4.2%** | 60 个候选仅 2.5 个被接受 ← 核心发现 |
	| 逐深度利用率 | Depth 1: 7.2%, Depth 2: 1.4%, Depth 3-6: 1.6-4.7% | 浅层利用率略高但整体极低 |

	注：N_expand（beam search 内部扩展节点数）需要 hook `topK_genrate` 内部逻辑才能精确获取，留待后续版本补充。当前从 N_verify 和 N_accepted 的对比已能充分展示优化空间。

	**最关键的发现**：60 个验证节点中仅 ~2.5 个被接受（利用率 4.2%），意味着 **95.8% 的 target verify 算力浪费在不会被接受的节点上**。当前启发式 rerank（按累积 path confidence 排序取 top-N）无法有效区分"会被接受"和"会被拒绝"的节点。

	**如果实际结果与预期不符怎么办**：
	- 如果节点验证利用率意外地高（> 15%）→ 说明当前启发式已经很有效，OPT-Tree 提升空间有限。报告中如实记录，说明为什么 OPT-Tree 在这个场景下不太适用
	- 实测 4.2% 在预期 3-7% 的范围内，**确认优化空间存在**

	**因果关系**：

	> **因为 E2.3 的数据证明**启发式 rerank 的节点验证利用率仅 4.2%，大部分验证预算被浪费，
	> **所以 我们提出 OPT-Tree**：将节点选择形式化为最大化全局 E[acceptance length] 的贪心优化问题。

	**产出物**：逐深度 verified vs accepted 柱状图，节点利用率分布直方图 + 时间序列

**产出物**：expand → verify → accept 节点转化桑基图，被接受/被拒节点的 path confidence 对比分布图

---

### 阶段二小结：瓶颈 → 改进的完整逻辑链

```
前置基础   实验发现                  瓶颈定位          改进方案
───────   ────────                 ────────          ────────

E0.1      E2.1 管线时延分解
  │         │
  │         ├─ Drafter+Rej/KV 实测占 26.6%
  │         │
  │         ├─→ E2.2 逐深度接受率
  │         │     │
  │         │     └─ Depth 5 仅 7.6% step 触及
  │         │           │
  │         │           ▼
  │         │     ┌──────────────┐      ┌──────────────┐
  │         │     │ D6瓶颈(深度) │      │              │
  │         │     │ 固定深度浪费  │ ───→ │   DDD        │
  │         │     │ drafting算力 │      │ 自适应深度     │
  │         │     └──────────────┘      └──────────────┘
  │         │
  │         └─→ E2.3 节点利用效率
  │               │
  │               └─ 验证利用率仅 3-7%
  │                     │
  │                     ▼
  │               ┌──────────────┐      ┌──────────────┐
  │               │  D6 瓶颈(形状)     │      │              │
  │               │ 启发式 rerank │ ───→ │  OPT-Tree   │
  │               │ 未最大化接受  │      │ 最优节点选择  │
  │               └──────────────┘      └──────────────┘
  │
  └─ "Decode 是 memory-bound"  ──→  整个项目的理论基础
```

**关键叙事点**（报告和 PPT 中反复强调）：

1. **不是随便选的两个优化**：DDD 和 OPT-Tree 是 E2.2 和 E2.3 实验数据**直接推导**出的改进方案
2. **改进与瓶颈的因果关系是实验数据支撑的**：不是"我们认为 DDD 好"而是"E2.2 的数据表明深层触及率断崖下降（仅 7.6%），固定深度存在系统性浪费，所以 DDD 有明确的优化目标"
3. **两个改进正交**：DDD 改深度、OPT-Tree 改宽度，互不冲突，可以叠加

---

## 6. 阶段三：改进实现与消融实验

> **核心定位**：这是最终报告的核心实证章节。通过 4 组配置的系统性对比，量化 DDD 和 OPT-Tree 各自的贡献，以及两者联合的效果。消融实验的设计直接回应课程评分标准中的"实验严谨性（20%）"。

### 6.1 背景：什么是消融实验（Ablation Study）？

消融实验是 MLSys/ML 研究中的标准方法论：对于一个包含多个组件（component）的系统，**逐个移除（ablating）组件**来测量每个组件对最终效果的贡献。在我们的项目中：

- 系统 = EAGLE + DDD + OPT-Tree
- 组件 = {DDD, OPT-Tree}
- 消融 = 分别测试 "不加 DDD、不加 OPT-Tree、两个都加、都不加" 四种配置

通过比较：
- **DDD-only vs Baseline**：DDD 的单独贡献（Δ_DDD）
- **OPT-Tree-only vs Baseline**：OPT-Tree 的单独贡献（Δ_OPT）
- **Joint vs Baseline**：联合贡献（Δ_Joint）
- **Joint vs DDD-only 和 OPT-Tree-only**：两个组件是否有协同效应（Δ_Joint > Δ_DDD + Δ_OPT？）

---

### E3.1 DDD-only 实验

**要回答的问题**：单独应用 DDD（自适应深度），相比 baseline 能提升多少 tokens/s？代价是什么？

**为什么需要这个实验**：消融分析的第一个组件。DDD 的预期收益是减少 drafting 开销，但风险是可能过早停止而漏掉有价值的深层 token（acceptance length 下降）。

**实验配置**：
- baseline: EAGLE-3 默认参数（depth=5, top_k=10, total_token=60）
- DDD: 相同基础参数 + `max_steps=9, check_steps=[5,7], threshold=τ*`

**预期结果**：

| 指标 | Baseline | DDD-only | 变化 |
|------|----------|----------|------|
| tokens/s | 63.8 | ~72 (+13%) | 预期 drafting 开销降低 |
| acceptance length | 3.06 | ~3.0 (略降) | 可能的代价：过早停止错过好 token |
| avg draft steps | 5.0 | ~4.2 | DDD 生效的直接证据 |

---

### E3.2 OPT-Tree-only 实验

**要回答的问题**：单独应用 OPT-Tree（最优树节点选择），相比 baseline 能提升多少 tokens/s？代价是什么？

**为什么需要这个实验**：消融分析的第二个组件。OPT-Tree 的预期收益是提升节点利用效率（acceptance length），但代价是 over-expand 和全局排序的额外计算开销。

**实验配置**：
- baseline: EAGLE-3 默认参数
- OPT-Tree: 相同 depth/beam + `over_expand_factor=2× budget`，使用 OPT-Tree 贪心选节点

**预期结果**：

| 指标 | Baseline | OPT-Tree-only | 变化 |
|------|----------|---------------|------|
| tokens/s | 63.8 | ~72 (+13%) | 节点更精准 → 接受更多 |
| acceptance length | 3.06 | ~3.4 (+11%) | OPT-Tree 的核心收益 |
| rerank 时延 | ~1ms | ~2ms (+1ms) | 全局排序的代价 |

---

### E3.3 DDD + OPT-Tree 联合实验

**要回答的问题**：同时应用两个改进，tokens/s 的提升是多少？两个改进是叠加（additive）还是协同（synergistic）？

**为什么需要这个实验**：验证两个改进之间是否存在**正交协同效应**。DDD 减少无效扩展 → 候选池更干净 → OPT-Tree 在更高质量的候选池中选节点 → 联合效果可能 > 各自效果之和。

**预期结果**：
- tokens/s 提升 15-30%（高于各自单独的 ~13%）
- 如果联合效果 ≈ DDD-only + OPT-Tree-only → 说明两个改进确实是正交的
- 如果联合效果 > DDD-only + OPT-Tree-only → 说明有正向协同效应

---

### E3.4 消融对比表（核心产出）

> **这是最终报告中最重要的一张表**，直接对应课程评分"实验验证（20%）"。

| 配置 | DDD | OPT | tok/s | vs Baseline | accept_len | avg_draft_steps |
|------|-----|-----|-------|-------------|------------|-----------------|
| Pure AR | - | - | 40.0 | 0.63× | - | - |
| **Baseline** | ✗ | ✗ | **63.8** | **1.00×** | 3.06 | 5.0 |
| DDD-only | ✓ | ✗ | ~72 | ~1.13× | ~3.0 | 4.2 |
| OPT-Tree-only | ✗ | ✓ | ~72 | ~1.13× | ~3.4 | 5.0 |
| **DDD+OPT-Tree** | ✓ | ✓ | **~80** | **~1.25×** | ~3.3 | 4.0 |

**这张表回答的关键问题**：
1. 投机解码整体有效吗？→ Pure AR vs Baseline（1.60× 加速）
2. DDD 单独有效吗？→ Baseline vs DDD-only
3. OPT-Tree 单独有效吗？→ Baseline vs OPT-Tree-only
4. 两者叠加更好吗？→ Baseline vs Joint
5. DDD 降低了 drafting 开销吗？→ avg_draft_steps 列
6. OPT-Tree 提升了接受效率吗？→ accept_len 列

---

### E3.5 超参数敏感性分析

**要回答的问题**：DDD 的阈值 τ 和 OPT-Tree 的预算 N 对最终性能有多敏感？是否存在一个宽泛的"好"区间，还是性能对参数极度敏感？

**为什么需要这个实验**：超参数的鲁棒性决定方法的**实用性**。如果一个改进只在极窄的参数范围内有效，那它的工程价值就有限。

**DDD 子实验**：
- 扫描 `τ ∈ {-5, -8, -10, -12, -15}`（覆盖从"非常保守"到"非常激进"的范围）
- τ 越高（如 -5）→ 更容易触发早停 → draft 步数少但可能漏掉好 token
- τ 越低（如 -15）→ 几乎不会早停 → 退化为 fixed depth
- 产出：τ vs (accept_length, step_time) 双轴图，标注最佳 trade-off 点

**OPT-Tree 子实验**：
- 扫描 `budget_N ∈ {20, 30, 40, 50, 60}`（over-expand 的最终节点数）
- `expand_factor ∈ {1.5×, 2×, 3×}`（over-expand 倍数）
- N 太小 → 候选不足，OPT-Tree 无空间
- N 太大 → verify 开销增大，抵消节点选择收益
- 产出：budget vs effective_tok/s 曲线，标注 sweet spot

---

## 7. 阶段四：补充分析与局限性

> **核心定位**：这两个实验分别对应课程评分的两个维度——分布一致性验证（确保改进的正确性）和局限性分析（"指出方法不适用的场景"，10%）。

### 7.1 背景：为什么需要这个阶段？

前三个阶段解决了"瓶颈在哪里 → 怎么改 → 改得多好"的问题。但一个完整的 MLSys 研究还需要回答两个收尾问题：

1. **我们有没有破坏什么？**（E4.1）：投机解码的核心卖点是"无损加速"。DDD 和 OPT-Tree 改变了树构造逻辑，但应该不影响 rejection sampling 的正确性。需要验证。
2. **我们的方法在哪些情况下不管用？**（E4.2）：没有任何优化是普适的。指出方法的局限性不是示弱，而是研究的严谨性体现。

---

### E4.1 分布一致性验证（Lossless 验证）

**要回答的问题**：DDD 和 OPT-Tree 是否破坏了投机解码的"无损"（lossless）性质？

**为什么需要这个实验**：
投机解码的数学保证（modified rejection sampling 下输出分布与 target 完全等价）依赖于：drafter 的输出结构不影响 rejection sampling 的正确性。DDD 改变了**扩展多少步**，OPT-Tree 改变了**选哪些节点**，但两者都没有修改 rejection sampling 的逻辑——理论上不应影响 lossless 性质。但这个理论保证需要用实验验证，以防实现中引入了 bug。

**方法**：
- `temperature=0.0`（greedy）：所有配置（Pure AR, Baseline, DDD-only, OPT-Tree-only, Joint）在同一 prompt 上应生成**完全一致**的 token 序列
- `temperature > 0`（sampling）：多次采样，对比输出分布的统计特征（KL divergence）

**预期结果**：temperature=0 时所有配置输出完全一致（token-by-token identical）

---

### E4.2 场景泛化性分析

**要回答的问题**：DDD 和 OPT-Tree 在不同类型的 prompt 下，效果是否有显著差异？在哪些场景下改进收益最大/最小？

**为什么需要这个实验**：
这是课程评分标准中的"局限性分析（10%）"的直接对应。一个好的 MLSys 研究应该清楚地说明方法的适用边界，而不是声称"在所有场景下都好"。

**实验设计**：

| Prompt 类型 | 代表任务 | 预期 DDD 收益 | 预期 OPT-Tree 收益 | 原因 |
|------------|---------|-------------|------------------|------|
| 事实问答 | "巴黎是哪个国家的首都？" | 中 | 中 | 事实性内容 token 分布较确定 |
| 代码生成 | "写一个快速排序函数" | 高 | 高 | 代码模板重复多，drafter 预测准确 |
| 复杂推理 | "解这道数学题..." | 低 | 中 | 推理步骤需要精确，早停可能跳过关键步骤 |
| 创意写作 | "写一首关于秋天的诗" | 中 | 中 | 创意性内容 token 分布较分散 |
| 多轮对话 | 连续 3 轮的对话历史 | 低（历史长时） | 中 | 长上下文下 drafter 预测难度增大 |

**分析要点**：
- 每一项分析都要配具体的实验数据（分场景的加速比柱状图）
- 解释为什么某些场景下收益变小（例如推理场景中 drafter 误差累积更快）
- 讨论这种局限性的根本原因是什么（drafter 本身的预测能力 vs 具体上下文特征）

---

## 8. 实验总表

| 编号 | 实验名称 | 阶段 | 状态 | 关键产出 | 验证/支撑的论点 |
|------|---------|------|------|---------|---------------|
| E0.1 | Roofline 分析 | 前置 | ✅ Lab2 | Roofline 图 | Decode 是 memory-bound |
| E0.2 | Kernel 级时延 | 前置 | ✅ Lab2 | Top-10 kernel 表 | GEMM 主导 decode 时间 |
| E0.3 | TTFT/TBT 方法 | 前置 | ✅ Lab1 | 计时方法论 | 技术工具基础 |
| E1.1 | Pure AR Baseline | Baseline | ✅ | 40.0 tok/s | 性能下界（加速比分母） |
| E1.2 | EAGLE-3 Baseline | Baseline | ✅ | 63.8 tok/s, 1.60× | 改进起点，引出优化空间 |
| **E2.1** | **管线时延分解** | **瓶颈定位** | 🔴 待做 | 时延饼图、堆叠柱状图 | 框定优化范围，引导 E2.2/E2.3 |
| **E2.2** | **逐深度接受率** | **瓶颈定位** | 🔴 待做 | 接受率 vs 深度折线图 | **DDD 的核心证据** |
| **E2.3** | **节点利用效率** | **瓶颈定位** | 🔴 待做 | 桑基图、path_conf 分布对比 | **OPT-Tree 的核心证据** |
| E3.1 | DDD-only | 改进验证 | ⬜ | tok/s, accept_len | DDD 单独贡献 |
| E3.2 | OPT-Tree-only | 改进验证 | ⬜ | tok/s, accept_len | OPT-Tree 单独贡献 |
| E3.3 | DDD+OPT-Tree 联合 | 改进验证 | ⬜ | tok/s, accept_len | 联合效果与正交性 |
| E3.4 | 消融对比表 | 改进验证 | ⬜ | 5 行对比表 | 系统性量化各组件的贡献 |
| E3.5 | 超参数敏感性 | 改进验证 | ⬜ | τ-sweep 图、budget-sweep 图 | 方法的鲁棒性 |
| E4.1 | 分布一致性 | 补充验证 | ⬜ | token 序列对比 | Lossless 性质保证 |
| E4.2 | 场景泛化性 | 补充验证 | ⬜ | 分场景加速比柱状图 | 局限性分析 |

> 🔴 = 6.9 开题前必须完成 | ⬜ = 6.9 后完成

---

## 9. 报告/PPT 叙事线

### 9.1 开题报告 (6.9) 结构

```
Chapter 1: 问题背景与动机
  ├── E0.1 引用：Decode 是 memory-bound → GPU 利用率极低
  ├── E0.2 引用：GEMM kernel 主导 → 每次 forward 的带宽成本高
  └── 过渡：投机解码如何利用这个特性来加速

Chapter 2: 投机解码与 EAGLE-3
  ├── 投机解码原理（Drafter-Verifier 框架、rejection sampling）
  ├── EAGLE 系列演进（EAGLE-1 → EAGLE-2 → EAGLE-3）
  └── D1-D7 组件分解，我们的选择（D6 维度）及其理由

Chapter 3: Baseline 复现
  ├── E1.1：Pure AR = 40.0 tok/s（性能下界）
  ├── E1.2：EAGLE-3 = 63.8 tok/s（1.60×）
  └── 问题：为什么 1.60× 远低于论文的 3-5×？

Chapter 4: 瓶颈定位 ★ 核心章节
  ├── E2.1：时延分解 → Drafter Expand 占 ~25%
  ├── E2.2：逐深度分析 → Depth 5 仅 7.6% step 触及 → D6 瓶颈(深度)
  ├── E2.3：节点利用率 → 仅 3-7%，path conf ≠ accept → D6 瓶颈(形状)
  └── 小结：两个瓶颈分别指向两个改进

Chapter 5: 改进方案设计
  ├── DDD (D6)：自适应深度，beam 置信度早停
  ├── OPT-Tree (D6)：最大化 E[acceptance length] 贪心选节点
  ├── 两者的正交性与协作方式
  └── 均不改 drafter 训练，纯推理侧优化

Chapter 6: 实验计划
  ├── 消融矩阵：Baseline → DDD-only → OPT-Tree-only → Joint
  ├── 评估指标：tokens/s, acceptance length, avg draft steps
  └── 预期时间线
```

### 9.2 终稿报告 (6.19) 结构

```
Chapter 1-3: 同开题（背景+投机解码+Baseline，更新为最终数据）

Chapter 4: 瓶颈定位（同开题，数据更完整，增加统计分析）

Chapter 5: 改进设计与实现
  ├── DDD：算法描述 + 伪代码 + 关键代码改动位置
  ├── OPT-Tree：算法描述 + 伪代码 + 关键代码改动位置
  └── 分布一致性分析（为什么改进不破坏 lossless）

Chapter 6: 实验验证 ★ 核心章节
  ├── E3.4 消融对比表（全文最重要的一张表）
  ├── E3.5 超参数分析（DDD τ-sweep, OPT-Tree budget-sweep）
  ├── E4.1 分布一致性验证（lossless confirmation）
  └── 失败/不 work 的尝试（如有）

Chapter 7: 局限性分析
  ├── E4.2：不同 prompt 类型下的效果差异
  ├── 模型规模敏感性（仅在 8B 上验证，对更大/更小模型的泛化性）
  └── DDD 的阈值需要 per-model 调优（工程实用性讨论）

Chapter 8: 总结
  ├── 核心贡献：两个正交的推理侧优化，联合提升 X% 吞吐
  ├── 方法论贡献：投机解码瓶颈定位的 profiling 框架
  └── 未来方向：扩展到 D7（verify 规则）、多模型验证
```

### 9.3 PPT 建议结构（课堂汇报，20%）

- Slide 1-3：问题 → 投机解码 → EAGLE-3（快速过）
- Slide 4：Baseline 实测（1.60× vs 论文 3-5×）
- **Slide 5-7：瓶颈定位（三个实验各一张图，这是汇报的重点）**
- Slide 8：改进方案（DDD + OPT-Tree，各一页伪代码/流程图）
- Slide 9：消融结果（对比表）
- Slide 10：局限性 & 总结

---

## 附录 A：各项实验的脚本规划

| 脚本文件 | 对应实验 | 功能 |
|---------|---------|------|
| `test_eagle_baseline.py` | E1.2 | ✅ 已有：EAGLE-3 推理测试 |
| `profiling_timing.py` | E2.1 | 🔴 EAGLE 四阶段时延分解 profiling |
| `profiling_acceptance.py` | E2.2 | 🔴 逐深度 token 接受率统计 |
| `profiling_tree_util.py` | E2.3 | 🔴 树节点 expand/verify/accept 转化率分析 |
| `cnets_ddd.py` | E3.1 | 🔴 DDD 改进实现（修改 cnets.py 的 expand 循环） |
| `cnets_opt_tree.py` | E3.2 | 🔴 OPT-Tree 改进实现（修改 rerank 阶段节点选择） |
| `run_ablation.py` | E3.3-E3.5 | 🔴 消融实验 + 超参数扫描自动化 |
| `verify_lossless.py` | E4.1 | 🔴 分布一致性验证 |
| `test_scenarios.py` | E4.2 | 🔴 多场景泛化性测试 |

## 附录 B：关键参考文献

1. Leviathan et al., "Fast Inference from Transformers via Speculative Decoding", ICML 2023.
2. Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty", ICML 2024. [[arXiv:2401.15077](https://arxiv.org/abs/2401.15077)]
3. Li et al., "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees", EMNLP 2024.
4. Li et al., "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test", ICLR 2025. [[arXiv:2503.01840](https://arxiv.org/abs/2503.01840)]
5. Dynamic Depth Decoding (DDD): [[arXiv:2409.00142](https://arxiv.org/pdf/2409.00142)]
6. OPT-Tree: [[arXiv:2406.17276](https://arxiv.org/pdf/2406.17276)]
