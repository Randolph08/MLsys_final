# 投机解码推理优化：DDD + OPT-Tree 项目方案设计

> **课程**：USTC ML System 2026 Final Project
> **团队**：3 人
> **方向**：投机解码（Speculative Decoding）推理侧优化
> **Baseline**：EAGLE-2 / EAGLE-3（SafeAILab/EAGLE 开源仓库）
> **改进点**：DDD（动态深度）+ OPT-Tree（最优树形状），均属于 D6 维度

---

## 目录

1. [项目背景与动机](#1-项目背景与动机)
2. [Baseline 选择与环境](#2-baseline-选择与环境)
3. [Profiling 计划：定位真实瓶颈](#3-profiling-计划定位真实瓶颈)
4. [优化点一：DDD（Dynamic Depth Decoding）](#4-优化点一ddd动态深度解码)
5. [优化点二：OPT-Tree（最优树形状选择）](#5-优化点二opt-tree最优树形状选择)
6. [实验设计](#6-实验设计)
7. [时间线与里程碑](#7-时间线与里程碑)
8. [团队分工](#8-团队分工)
9. [风险与应对](#9-风险与应对)

---

## 1. 项目背景与动机

### 1.1 问题：LLM 推理的 Memory-Bound 瓶颈

大语言模型（LLM）的自回归解码阶段是典型的 **memory-bound** 场景：每生成一个 token，需要将整套模型权重从 HBM 加载到计算单元。在 batch=1 的单用户推理场景中，算术强度极低，GPU 利用率常常只有峰值算力的几个百分点。随着硬件算力增速远超显存带宽增速，这一瓶颈持续加剧。

### 1.2 投机解码的核心思想

投机解码（Speculative Decoding）用一个小而便宜的 **drafter（草稿模型）** 预先猜测未来 K 个 token，然后将"前缀 + 草稿"一次性送入大模型（verifier/target）并行验证，通过拒绝采样确保输出分布无损。其本质是**用并行性换取带宽效率**——一次 target forward 接受多个 token，将带宽成本摊薄。

### 1.3 为什么选 D5-D7 维度

投机解码系统可分解为 7 个组件维度（D1-D7）。其中 D1-D4 涉及 drafter 模型架构和训练，而 **D5（draft 输出结构）、D6（draft 长度决策）、D7（verify 规则）是纯推理侧的优化**，可以：

- 直接复用已有的 drafter 预训练权重，**不需要额外训练**
- 改动集中在推理循环的树构造逻辑，**代码改动面小**
- 改进效果可以通过消融实验清晰验证

### 1.4 我们的优化目标

在 EAGLE 框架的推理循环中，同时引入两个正交的优化：

| 优化点 | 所属维度 | 作用阶段 | 核心改进 |
|--------|---------|---------|---------|
| **DDD** | D6（长度决策） | Expand（扩展） | 根据 beam 置信度自适应决定扩展深度，替代固定步数 |
| **OPT-Tree** | D6（形状决策） | Rerank（筛选） | 用最大化期望接受长度的贪心算法选节点，替代启发式排序 |

---

## 2. Baseline 选择与环境

### 2.1 硬件环境

| 资源 | 规格 |
|------|------|
| GPU | 约 5× NVIDIA GeForce RTX 3090 (24GB)，单卡 24GB 显存 |
| CUDA | 12.8 |
| Driver | 570.211.01 |
| OS | Linux 5.15.0 |

> **资源利用策略**：单卡即可运行 LLaMA-3-8B + EAGLE drafter（~17GB）。多卡优势在于：(1) 可并行跑多个消融实验配置；(2) 多 batch size 实验可跨卡并行；(3) 必要时可尝试 LLaMA-3-8B 同级别但不同家族的 target model 做 cross-model 验证。

### 2.2 软件与模型选择

| 组件 | 选择 | 说明 |
|------|------|------|
| **Baseline 仓库** | [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) | EAGLE-1/2/3 官方实现，Apache 2.0 |
| **Drafter Checkpoint** | EAGLE-3（优先）/ EAGLE-2（备选） | 开源预训练权重，HuggingFace 下载 |
| **Target Model** | LLaMA-3-8B（优先）/ LLaMA-2-7B-Chat（备选） | 8B 在 FP16 下约 16GB，3090 单卡可运行 |
| **推理框架** | PyTorch + Transformers | 与 Lab1/Lab2 环境一致 |
| **Profiling 工具** | PyTorch Profiler + torch.cuda.Event | 沿用 Lab2 的 profiling 经验 |

### 2.3 EAGLE 版本选择说明

EAGLE-1/2/3 同属一个代码仓库，区别如下：

| | EAGLE-1 | EAGLE-2 | EAGLE-3 |
|---|---|---|---|
| Drafter 架构 | Feature-level AR | 复用 EAGLE-1 | Token pred + multi-layer fusion |
| 草稿结构 | 链式 | 动态树 | 动态树（兼容 EAGLE-2） |
| 接受长度 | ~3.5 | ~4.2 | 可达 6+ |
| 加速比 | ~3× | ~4× | 最高 6.5× |

**推荐策略**：使用 EAGLE-3 的 drafter checkpoint（接受率更高，baseline 更强），在 EAGLE-2 的树构造推理循环上做 DDD 和 OPT-Tree 改进。两者的树构造接口兼容。

### 2.4 环境搭建步骤

```bash
# 1. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 2. 安装 PyTorch（CUDA 12.8）
pip install torch torchvision torchaudio

# 3. 安装依赖
pip install transformers accelerate

# 4. 克隆 EAGLE 仓库
git clone https://github.com/SafeAILab/EAGLE.git
cd EAGLE
pip install -e .

# 5. 下载预训练权重（以 LLaMA-3-8B + EAGLE-3 为例）
# - Target model: meta-llama/Meta-Llama-3-8B (HuggingFace)
# - Drafter: yuhuili/EAGLE-3-Llama-3-8B (HuggingFace)
```

---

## 3. Profiling 计划：定位真实瓶颈

> **目标**：在 6.9 开题前完成 baseline profiling，用数据回答"瓶颈在哪"。

### 3.1 Profiling 维度

| 维度 | 测量内容 | 工具 | 预期发现 |
|------|---------|------|---------|
| **时延分解** | drafter forward、target forward、tree construction、rejection sampling 各阶段耗时 | `torch.cuda.Event` + 自定义 timer | 树构造（expand + rerank）占 drafting 阶段相当比例；固定深度导致尾部步骤浪费 |
| **接受率分析** | 按 draft 深度位置统计逐位置接受率 | 自定义 hook 在推理循环中记录 | 深层位置（step 5+）接受率显著下降，固定深度 6 步存在浪费 |
| **显存追踪** | peak memory、KV cache 占用、drafter/target 权重占用 | `torch.cuda.max_memory_allocated` | 确认 3090 24GB 余量，评估是否可增加 over-expand 节点预算 |
| **树结构分析** | 每轮 expand 出的节点数、实际被接受的节点数、树宽度分布 | 自定义统计 | 评估当前 rerank 策略是否最优，是否有效利用了 verify 预算 |

### 3.2 Profiling 实验设计

**实验配置**：
- Target: LLaMA-3-8B
- Drafter: EAGLE-3 checkpoint
- 测试集: MT-Bench 或 ShareGPT 的前 100 条 prompt
- 生成长度: 每个 prompt 生成 256 token

**输出产物**（用于开题报告）：
1. **各阶段时延饼图/柱状图** — 展示 drafter forward、target forward、树构造、其他各占多少比例
2. **逐深度位置接受率折线图** — 展示 step 1-6 各自接受率的变化趋势
3. **树节点利用率统计** — expand 出的节点 vs 最终被 verify 接受的节点比例
4. **Roofline 模型分析** — 确认 decode 阶段处于 memory-bound 区域

### 3.3 Profiling 代码框架

```python
# profiling_eagle.py — 核心 profiling 逻辑框架

import torch
import time
from collections import defaultdict

class EAGLEProfiler:
    def __init__(self):
        self.stats = defaultdict(list)

    @torch.no_grad()
    def profile_step(self, model, input_ids, max_new_tokens=256):
        """对 EAGLE 推理的一个完整 step 做 profiling"""
        timings = {}

        # 1. Target forward (prefill)
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        hidden_states = model.target_model(input_ids, output_hidden_states=True)
        t1.record()
        torch.cuda.synchronize()
        timings['target_prefill'] = t0.elapsed_time(t1)

        # 2. Draft tree construction (expand + rerank)
        t2 = torch.cuda.Event(enable_timing=True)
        t3 = torch.cuda.Event(enable_timing=True)
        t2.record()
        draft_tree, tree_mask = model.build_draft_tree(hidden_states)
        t3.record()
        torch.cuda.synchronize()
        timings['tree_construction'] = t2.elapsed_time(t3)

        # 3. Target forward (verify)
        t4 = torch.cuda.Event(enable_timing=True)
        t5 = torch.cuda.Event(enable_timing=True)
        t4.record()
        logits = model.target_model.verify(draft_tree, tree_mask)
        t5.record()
        torch.cuda.synchronize()
        timings['target_verify'] = t4.elapsed_time(t5)

        # 4. Acceptance rate per depth
        # ... (在 rejection sampling 后统计各深度接受率)

        return timings
```

---

## 4. 优化点一：DDD（动态深度解码）

### 4.1 动机：固定深度的浪费

EAGLE-2 的 beam search 固定走 6 步扩展。但不同上下文的预测难度差异很大：
- **简单位置**（如 "Thank you" 之后接 "very much"）：前几步接受率极高，后几步已偏离，继续扩展是浪费
- **困难位置**（如复杂推理的中间步骤）：前几步接受率低，但 beam 整体还保持高置信度，值得走更远

### 4.2 算法描述

> 参考论文：[Dynamic Depth Decoding (arXiv 2409.00142)](https://arxiv.org/pdf/2409.00142)

**核心思想**：在 beam search 扩展过程中，定期检查 beam 整体的置信度。如果置信度低于阈值，说明后续扩展收益很低，提前停止。

**具体流程**：

```
Input:  max_steps = 11, check_steps = [5, 7, 9], threshold τ
Output: draft tree of dynamic depth

for step in 1..max_steps:
    # 正常 beam search 扩展一步
    expand_one_step(beam)

    if step in check_steps:
        # 计算 beam 整体置信度
        H = log Σ_i exp(logprobsum[i])

        if H < τ:
            break  # 提前停止
```

**关键公式**：

$$
H = \log\sum_{i \in \text{beam}} \exp(\text{logprobsum}[i])
$$

其中 `logprobsum[i]` 是 beam 中第 i 条路径的累积对数概率。当多条路径的累积概率都较低时，H 也会偏低，表示 beam 整体对未来预测缺乏信心。

### 4.3 实现计划

**改动位置**：EAGLE 推理循环中 `expand` 阶段的 beam search 循环

**改动量**：约 30-50 行代码

**超参数**：
- `max_steps = 11`（最大扩展步数，从原来的 6 放宽）
- `check_steps = [5, 7, 9]`（检查点）
- `threshold τ`：需要通过网格搜索在验证集上确定最优值

**伪代码**：

```python
def expand_with_ddd(hidden_states, max_steps=11,
                     check_steps=[5, 7, 9], threshold=-10.0):
    beam = initialize_beam(hidden_states)
    actual_depth = 0

    for step in range(1, max_steps + 1):
        beam = expand_one_step(beam)
        actual_depth = step

        if step in check_steps:
            logprobsums = [path.cumulative_logprob for path in beam.paths]
            # H = logsumexp of all path logprobsums
            H = torch.logsumexp(torch.stack(logprobsums), dim=0)

            if H < threshold:
                break  # early stop

    return beam.to_tree(), actual_depth
```

### 4.4 预期效果

- 简单 prompt：提前在 step 5 或 7 停止，减少 drafting 开销
- 困难 prompt：走到 step 11，充分探索
- 总体：在保持或略提升 acceptance length 的同时，减少 drafting 阶段平均耗时

---

## 5. 优化点二：OPT-Tree（最优树形状选择）

### 5.1 动机：启发式 Rerank 的次优性

EAGLE-2 在 rerank 阶段按每个节点的路径累积 confidence 排序，取 top-N 送入 verify。但这种启发式排序不一定最大化期望接受长度——一个置信度稍低但有很多高置信度后代的节点，可能比一个孤立的最高置信度节点更有价值。

### 5.2 算法描述

> 参考论文：[OPT-Tree (arXiv 2406.17276)](https://arxiv.org/pdf/2406.17276)

**核心思想**：将"选哪些节点送入验证"形式化为最大化整棵树的期望接受长度的优化问题。

**整棵树的期望接受长度**：

$$
\mathbb{E}[\text{acceptance length}] = \sum_{v \in T} \prod_{u \in \text{path}(v)} q(u)
$$

其中 $q(u)$ 是节点 $u$ 的 drafter 预测概率（作为其被接受概率的近似）。每个节点对期望接受长度的贡献 = 根到该节点路径上所有概率的乘积。

**OPT-Tree 贪心算法**：

```
Input:  已 over-expand 的大树（节点数 M > 预算 N），每节点有 drafter 概率 q
Output: 连通子树 T，|T| ≤ N，最大化 E[acceptance length]

Step 1: 对整棵树，计算每个节点的路径累积概率
        score(v) = Π_{u in path(v)} q(u)

Step 2: 按 score 全局降序排列所有节点

Step 3: 贪心取前 N 个节点，但需保证连通性——
        对每个选中节点 v，其所有祖先也必须被选中

Step 4: 构建 tree attention mask（祖先链可见），送入 verify
```

### 5.3 实现计划

**改动位置**：EAGLE 推理循环中 `rerank` 阶段，节点选择逻辑

**改动量**：约 50-80 行代码

**实现要点**：

1. **Over-expand**：在 expand 阶段多扩展一些节点（超过 verify 预算），为 OPT-Tree 提供更丰富的候选池
2. **贪心选择 + 连通性保证**：排序后取节点时，自动补上该节点的所有祖先
3. **预算控制**：最终送入 verify 的节点数不超过 N_max（由硬件和 attention mask 上限约束）

**伪代码**：

```python
def opt_tree_selection(tree_nodes, budget_N):
    """
    tree_nodes: List[TreeNode], 每个节点有:
        - .parent: 父节点引用
        - .q: drafter 预测概率
        - .path_score: 路径累积概率
    budget_N: int, verify 预算（最大节点数）
    """
    # Step 1: 计算每个节点的 path_score
    for node in tree_nodes:
        node.path_score = compute_path_score(node)

    # Step 2: 按 path_score 降序排列
    sorted_nodes = sorted(tree_nodes, key=lambda n: n.path_score, reverse=True)

    # Step 3: 贪心选择 + 连通性保证
    selected = set()
    for node in sorted_nodes:
        if len(selected) >= budget_N:
            break

        # 将当前节点及其所有祖先加入选中集合
        current = node
        ancestors_to_add = []
        while current is not None and current not in selected:
            ancestors_to_add.append(current)
            current = current.parent

        # 检查加入后是否超预算
        if len(selected) + len(ancestors_to_add) <= budget_N:
            selected.update(ancestors_to_add)

    # Step 4: 构建连通子树（保证 attention mask 有效）
    return build_tree_from_selected(selected)
```

### 5.4 与 DDD 的协作关系

```
┌──────────────────────────────────────────────────┐
│              EAGLE 推理循环（改进后）              │
│                                                  │
│  Target Prefill → hidden states                  │
│       │                                          │
│       ▼                                          │
│  ┌─────────────────────────────────────┐         │
│  │  Phase 1: Expand (DDD)              │         │
│  │  - beam search, max 11 steps        │         │
│  │  - 在第 5/7/9 步检查 H              │         │
│  │  - H < τ → early stop               │         │
│  │  - over-expand 更多节点              │         │
│  │  输出: 深度可变的候选树              │         │
│  └─────────────────────────────────────┘         │
│       │                                          │
│       ▼                                          │
│  ┌─────────────────────────────────────┐         │
│  │  Phase 2: Select (OPT-Tree)         │         │
│  │  - 计算每个节点 path_score           │         │
│  │  - 全局排序 → 贪心选 top-N           │         │
│  │  - 保证祖先连通性                    │         │
│  │  输出: 最优子树 + attention mask     │         │
│  └─────────────────────────────────────┘         │
│       │                                          │
│       ▼                                          │
│  Tree Attention Verify → Accept/Reject           │
│                                                  │
└──────────────────────────────────────────────────┘
```

两者**串行协作，互不冲突**：
- DDD 控制"树长多深" — 决定候选池的规模和深度
- OPT-Tree 控制"留哪些节点" — 从候选池中选出最优子集

---

## 6. 实验设计

### 6.1 实验配置矩阵

| 配置 | DDD | OPT-Tree | 说明 |
|------|-----|----------|------|
| **Baseline** | ✗ | ✗ | EAGLE-2 原始：固定深度 6 + 启发式 rerank |
| **DDD-only** | ✓ | ✗ | 仅动态深度，rerank 不变 |
| **OPT-Tree-only** | ✗ | ✓ | 固定深度 + 最优树选择 |
| **DDD+OPT-Tree** | ✓ | ✓ | 最终方案 |

### 6.2 评估指标

| 类别 | 指标 | 定义 |
|------|------|------|
| **吞吐** | tokens/s | 每秒生成 token 数（含 drafting + verify 全部耗时） |
| **加速比** | speedup | 相对纯 AR 解码的端到端加速比 |
| **接受效率** | acceptance length | 每次 verify 平均接受的 token 数 |
| **drafting 效率** | avg draft steps | 平均实际扩展步数（DDD 专属） |
| **精度** | lossless 验证 | 确认输出分布与 target 一致 |

### 6.3 多卡并行实验策略

5 张 3090 允许我们将消融实验的多个配置同时跑在不同 GPU 上，大幅缩短实验周期：

| GPU | 用途 | 配置 |
|-----|------|------|
| GPU 0 | Baseline（EAGLE-2 原始） | 无改动 |
| GPU 1 | DDD-only | 动态深度 |
| GPU 2 | OPT-Tree-only | 最优树 |
| GPU 3 | DDD + OPT-Tree | 联合方案 |
| GPU 4 | 超参数搜索 / 补充实验 | 灵活分配 |

> 每张卡独立运行一个完整配置，所有配置在相同测试集上评估，确保结果可比。实验脚本统一管理，通过 `CUDA_VISIBLE_DEVICES` 指定 GPU。

### 6.4 测试数据集

- **MT-Bench**：80 条多轮对话，覆盖写作、推理、数学等 8 类
- **ShareGPT 子集**：随机采样 100 条单轮对话
- **HumanEval**（可选）：代码生成场景，验证不同 domain 下的效果

### 6.5 消融实验

**主实验**：4 个配置 × 3 个数据集，比较 tokens/s 和 acceptance length

**超参数敏感性**（DDD）：
- `threshold τ ∈ {-5, -8, -10, -12, -15}`
- `check_steps` 的不同设置

**预算敏感性**（OPT-Tree）：
- `budget_N ∈ {20, 30, 40, 50}`（对照 EAGLE-2 默认约 25-30）

**场景分析**：
- 按 prompt 难度（perplexity）分组，分析改进在不同难度下的增益差异

---

## 7. 时间线与里程碑

```
Week 1 (6.2 - 6.8) ─── 开题准备
│
├── 6.2-6.3: 环境搭建、模型下载、跑通 baseline
├── 6.4-6.5: Profiling（时延分解 + 接受率分析 + 显存追踪）
├── 6.6-6.7: 开题报告撰写（瓶颈分析 + 方案设计）
├── 6.8:    开题报告内部 review
└── 6.9:    ★ 开题报告

Week 2 (6.9 - 6.15) ── 核心实现
│
├── 6.9-6.10:  实现 DDD（expand 循环改造）
├── 6.10-6.11: DDD 超参搜索 + 初步实验
├── 6.12-6.13: 实现 OPT-Tree（rerank 逻辑改造）
├── 6.13-6.14: OPT-Tree 预算参数调优
└── 6.14-6.15: DDD+OPT-Tree 联合调试

Week 3 (6.15 - 6.19) ── 实验与报告
│
├── 6.15-6.16: 完整消融实验（4 配置 × 3 数据集，多卡并行）
├── 6.16-6.17: 局限性分析 + 失败案例分析
├── 6.17-6.18: 终稿报告撰写
└── 6.19:     ★ 最终提交
```

### 6.9 开题报告内容框架

| 章节 | 内容 | 预期产出 |
|------|------|---------|
| 问题背景 | 投机解码 memory-bound 瓶颈 | 前人工作简述 + 动机 |
| 瓶颈识别 | profiling 数据 | 时延饼图、逐深度接受率折线图、roofline 分析 |
| 改进方案 | DDD + OPT-Tree 算法设计 | 算法流程图、伪代码、与瓶颈的因果关系论证 |
| 实验计划 | 配置矩阵 + 消融设计 | 4 配置对比表、评估指标定义 |
| 预期结果 | 基于论文数据的合理预估 | 加速比预估、接受率提升预估 |

---

## 8. 团队分工

| 成员 | 主要职责 | 6.9 前重点 | 6.9 后重点 |
|------|---------|-----------|-----------|
| **A (环境+Profiling)** | 环境搭建、baseline 跑通、profiling 全流程 | 环境搭建 + profiling 数据产出 | DDD 实现 + 超参搜索 |
| **B (DDD 负责人)** | DDD 算法实现、DDD 相关实验 | 算法理解 + profiling 配合 + 方案撰写 | DDD 核心实现 + 实验 |
| **C (OPT-Tree 负责人)** | OPT-Tree 算法实现、OPT-Tree 相关实验 | 算法理解 + profiling 配合 + 方案撰写 | OPT-Tree 核心实现 + 实验 |
| **全员** | 消融实验、报告撰写 | 开题报告 | 联合实验 + 终稿 |

> 注：成员 A 在 6.9 后环境工作收尾后，可接手联合调试和消融实验的协调工作。

---

## 9. 风险与应对

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|---------|
| EAGLE-3 checkpoint 在 3090 上 OOM | 低 | 高 | LLaMA-3-8B (~16GB) + EAGLE-3 drafter (~1GB) 约 17GB，3090 24GB 完全够用。如遇意外可降级到 LLaMA-2-7B (~14GB) |
| DDD 阈值难以调优，效果不如预期 | 中 | 中 | 先做宽泛网格搜索；如果效果微弱，报告中详细分析原因也是加分项 |
| OPT-Tree 贪心选择开销抵消收益 | 低 | 中 | 节点排序是纯 CPU 操作，O(M log M) 在 M ≤ 50 时基本无感；监控并报告中讨论 |
| 分布一致性被破坏（DDD/OPT-Tree 改变了树结构） | 低 | 高 | DDD 只改深度不改验证规则；OPT-Tree 不改采样逻辑——两者都不影响 lossless 性质，但需手动验证 |
| GitHub 仓库代码结构复杂，改动位置难定位 | 中 | 中 | 先用 1-2 天通读 EAGLE 推理循环代码，标注 expand/rerank 两个入口函数 |

---

## 附录：关键参考文献

1. Leviathan et al., "Fast Inference from Transformers via Speculative Decoding", ICML 2023.
2. Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty", ICML 2024. [[arXiv](https://arxiv.org/abs/2401.15077)]
3. Li et al., "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees", EMNLP 2024.
4. Li et al., "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test", ICLR 2025. [[arXiv](https://arxiv.org/abs/2503.01840)]
5. Dynamic Depth Decoding (DDD): [[arXiv 2409.00142](https://arxiv.org/pdf/2409.00142)]
6. OPT-Tree: [[arXiv 2406.17276](https://arxiv.org/pdf/2406.17276)]
7. Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads", ICML 2024.
8. Elhoushi et al., "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding", ACL 2024.
