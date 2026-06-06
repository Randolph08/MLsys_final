# 投机解码推理优化：DDD + OPT-Tree

> **课程**: USTC ML System 2026 Final Project  
> **方向**: 投机解码（Speculative Decoding）推理侧优化  
> **Baseline**: [EAGLE-2 / EAGLE-3](https://github.com/SafeAILab/EAGLE) (SafeAILab)  
> **改进点**: DDD（动态深度）+ OPT-Tree（最优树形状），均属于 D6 维度  
> **硬件**: NVIDIA RTX 3090 (24GB) × 5  

---

## 项目概述

本项目针对大语言模型推理中的 **memory-bound 瓶颈**，在 EAGLE-3 投机解码框架上进行两项正交改进：

| 优化点 | 所属维度 | 作用阶段 | 核心改进 |
|--------|---------|---------|---------|
| **DDD** | D6（长度决策） | Expand（扩展） | 根据 beam 置信度自适应决定扩展深度，替代固定步数 |
| **OPT-Tree** | D6（形状决策） | Rerank（筛选） | 用最大化期望接受长度的贪心算法选节点，替代启发式排序 |

### 投机解码原理

投机解码用一个小型 drafter（草稿模型）预先猜测未来 K 个 token，将"前缀 + 草稿"一次性送入大模型并行验证，通过拒绝采样确保输出分布无损。其本质是**用并行性换取带宽效率**。

---

## 目录结构

```
MLsys_final/
├── EAGLE/                        # EAGLE baseline 源码 (来自 SafeAILab/EAGLE)
│   ├── eagle/
│   │   ├── model/                # 模型架构 (ea_model, cnets, kv_cache)
│   │   ├── evaluation/           # 评测脚本 (speed, alpha, gen_answer)
│   │   ├── train/                # 训练配置与脚本
│   │   └── application/          # WebUI
│   ├── requirements.txt          # EAGLE 官方依赖
│   └── setup.py
├── experiments/                  # 实验脚本与结果
│   ├── config.py                 # 公共配置模块（路径、参数）
│   ├── profiling_timing.py       # E2.1: 管线时延分解
│   ├── profiling_acceptance.py   # E2.2: 逐深度接受率分析
│   ├── profiling_tree_util.py    # E2.3: 树节点利用效率
│   ├── ablation_ddd.py           # E3: DDD 消融实验
│   ├── ablation_full.py          # E3: 完整消融实验
│   ├── verify_lossless.py        # E4: 无损性验证
│   ├── scenario_test.py          # E4: 多场景测试
│   ├── test_ddd.py               # DDD 单元测试
│   ├── test_opt_tree.py          # OPT-Tree 单元测试
│   ├── E2.1_timing/              # 时延分析结果（含图表）
│   ├── E2.2_acceptance/          # 接受率分析结果（含图表）
│   ├── E2.3_tree_util/           # 树利用率分析结果（含图表）
│   ├── E3_ablation/              # 消融实验结果
│   └── E4_*/                     # 其他实验结果
├── models/                       # 预训练模型 (需下载，见下方)
│   ├── Llama-3.1-8B-Instruct/    # Target 模型 (meta-llama)
│   └── DeepSeek-R1-Distill-Llama-8B/  # Drafter 可选模型 (deepseek-ai)
├── EAGLE_checkpoints/            # EAGLE-3 drafter 权重 (需下载)
│   ├── EAGLE3-LLaMA3.1-Instruct-8B/
│   └── EAGLE3-DeepSeek-R1-Distill-LLaMA-8B/
├── project_design.md             # 项目方案设计文档
├── experiment_plan.md            # 实验规划与逻辑框架
├── preliminary_experiment_report.md  # 前期实验汇报
├── test_diagnostic.py            # 环境诊断脚本
├── test_eagle_baseline.py        # EAGLE baseline 测试
├── requirements.txt              # Python 依赖
├── download_models.sh            # 模型下载脚本
└── .gitignore
```

---

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 下载模型与 Checkpoint

```bash
# 运行自动下载脚本
bash download_models.sh
```

或者手动从 HuggingFace 下载：

| 模型 | HuggingFace 地址 |
|------|-----------------|
| Llama-3.1-8B-Instruct | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| DeepSeek-R1-Distill-Llama-8B | [deepseek-ai/DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) |
| EAGLE3-LLaMA3.1-Instruct-8B | [SafeAILab/EAGLE3-LLaMA3.1-Instruct-8B](https://huggingface.co/SafeAILab/EAGLE3-LLaMA3.1-Instruct-8B) |
| EAGLE3-DeepSeek-R1-Distill-LLaMA-8B | [SafeAILab/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B](https://huggingface.co/SafeAILab/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B) |

### 3. 验证环境

```bash
# 检查依赖和环境
python test_diagnostic.py

# 运行 EAGLE baseline 测试
python test_eagle_baseline.py
```

### 4. 运行实验

```bash
# E2.1 - 管线时延分解
python experiments/profiling_timing.py

# E2.2 - 逐深度接受率分析
python experiments/profiling_acceptance.py

# E2.3 - 树节点利用效率
python experiments/profiling_tree_util.py

# E3 - DDD 消融实验
python experiments/ablation_ddd.py

# E3 - 完整消融实验
python experiments/ablation_full.py

# E4 - 无损性验证
python experiments/verify_lossless.py

# E4 - 多场景测试
python experiments/scenario_test.py
```

---

## 实验概览

| 阶段 | 实验 | 目的 | 关键结论 |
|------|------|------|---------|
| E2.1 | 管线时延分解 | 定位 EAGLE 管线各阶段耗占比 | 识别瓶颈阶段 |
| E2.2 | 逐深度接受率 | 分析不同 draft 深度的接受率 | 深层 token 接受率显著下降 |
| E2.3 | 树节点利用 | 分析 draft tree 的节点利用效率 | 部分节点利用率低 |
| E3 | DDD + OPT-Tree 消融 | 验证两项改进的效果 | 待补充 |
| E4 | 无损性验证 + 多场景 | 验证加速不损害输出质量 | 待补充 |

---

## 关键文档

- [项目方案设计](project_design.md) — 动机、架构与改进点详解
- [实验规划](experiment_plan.md) — 完整实验框架与叙事线
- [前期实验报告](preliminary_experiment_report.md) — 6.4 前期实验结果与瓶颈分析

---

## 致谢

- [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) — EAGLE-2/3 投机解码框架
- [meta-llama](https://huggingface.co/meta-llama) — LLaMA-3.1 模型
- [deepseek-ai](https://huggingface.co/deepseek-ai) — DeepSeek-R1 模型
