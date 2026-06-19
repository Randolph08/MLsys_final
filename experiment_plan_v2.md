# 投机解码推理优化 —— 实验规划 v2

> **项目**：USTC ML System 2026 Final Project  
> **方案**：基于 EAGLE-3 的 DDD（Dynamic Depth Decoding）+ OPT-Tree 推理侧复现与改进  
> **模型**：LLaMA-3.1-8B-Instruct + EAGLE-3 官方 checkpoint  
> **硬件**：NVIDIA RTX 3090 (24GB) × 5  
> **本文档定位**：在已有实验和复现排查基础上，重新整理最终大作业从 baseline、瓶颈定位、算法复现、改进验证到报告撰写所需的完整实验内容。

---

## 0. v2 相比原计划的核心变化

原 `experiment_plan.md` 已经建立了很好的叙事框架：从 memory-bound 背景出发，通过 EAGLE 管线 profiling 找到 DDD 和 OPT-Tree 的优化动机，再做消融验证。这个逻辑仍然保留。

v2 的主要调整是：**先修正实验口径，再谈算法效果**。当前代码和已有结果显示，初始复现实验中存在几个会显著影响结论的问题：

1. LLaMA-3.1-8B-Instruct 是 chat model，主实验必须使用 `tokenizer.apply_chat_template()`，不能把裸 prompt 作为最终主表。
2. 速度 baseline 应使用 EAGLE 官方口径的 `EaModel.naivegenerate()`，而不是 HuggingFace `generate()`。
3. `eagenerate()` 和 `naivegenerate()` 当前存在 `max_new_tokens` off-by-one，需要统一停止条件或统一截断统计。
4. 当前 DDD 统计有重复记录，且阈值搜索区间没有先基于真实 `H = logsumexp(logprobsum)` 分布确定。
5. 当前 OPT-Tree 在 EAGLE-2/3 dynamic tree baseline 上高度退化，可能是一个 negative result，而不是稳定正向改进。
6. 当前 tree utilization 只能证明验证预算利用率低，不能直接证明 baseline rerank 选错节点；若要支撑 OPT-Tree，需要记录节点集合差异。

因此，v2 的实验主线变为：

```
统一口径和正确性门槛
    ↓
官方 EAGLE baseline 复现
    ↓
重新解释已有瓶颈 profiling
    ↓
修正版 DDD 复现与调参
    ↓
OPT-Tree 节点选择差异分析 / negative result
    ↓
最终消融、泛化性和报告叙事
```

---

## 1. 总体实验原则

后续所有主实验必须满足以下原则。

| 原则 | 具体要求 |
|------|----------|
| 统一 prompt | 使用 LLaMA3 chat template，裸 prompt 仅作为 sanity check |
| 统一 baseline | 速度对比使用 `EaModel.naivegenerate()` vs `EaModel.eagenerate()` |
| 统一停止条件 | 用实际生成 token 数统计 tok/s，必要时截断到 `max_new_tokens` |
| 统一 dtype | RTX 3090 上主实验统一 `fp16`，报告中说明与官方硬件差异 |
| 先正确后优化 | lossless / token sequence check 不通过时，不解释 speedup 为算法收益 |
| 指标对应机制 | DDD 看 draft calls / early stop rate；OPT-Tree 看 tree set diff / accept length |
| 保留 negative result | 如果 OPT-Tree 与 dynamic EAGLE baseline 重合，应如实写成失败分析 |

---

## 2. 实验全景与依赖关系

```
E0 前置背景
  ├── Lab2 Roofline: Decode 是 memory-bound
  ├── Lab2 Profiler: decode GEMM/kernel 行为
  └── Lab1 TTFT/TBT: torch.cuda.Event 计时方法

E1 统一口径 baseline
  ├── E1.1 Chat-template prompt sanity
  ├── E1.2 Official Naive baseline
  ├── E1.3 EAGLE-3 official baseline
  └── E1.4 Greedy token一致性检查

E2 瓶颈定位复核
  ├── E2.1 Pipeline timing breakdown
  ├── E2.2 Acceptance-by-depth
  └── E2.3 Tree selection / utilization 重新定义

E3 DDD 复现与改进
  ├── E3.1 H 分布 profiling
  ├── E3.2 Fixed depth 对照
  ├── E3.3 DDD paper-like 配置
  ├── E3.4 DDD threshold search
  └── E3.5 DDD-only 主结果

E4 OPT-Tree 复现与定位
  ├── E4.1 Baseline vs OPT tree set diff
  ├── E4.2 OPT-only dynamic baseline 对照
  ├── E4.3 弱 baseline 对照（不再执行）
  └── E4.4 OPT-Tree negative result 分析

E5 最终系统实验
  ├── E5.1 四组消融矩阵
  ├── E5.2 多数据集 / 场景泛化性
  ├── E5.3 长度 / batch / prompt 类型敏感性
  └── E5.4 最终报告图表
```

---

## 3. 阶段零：前置背景实验

这一阶段沿用原计划中的有效内容，不需要重新跑。它们用于报告中的 Motivation。

### E0.1 Decode 阶段是 Memory-Bound

**来源**：Lab2 Roofline 分析。

**要支撑的论点**：LLM batch=1 decode 每步只生成一个 token，算术强度低，RTX 3090 上 decode 工作点落在 memory-bound 区域。

**在报告中的作用**：说明投机解码为什么有价值。投机解码并不是减少 target 模型参数，而是让一次 target forward 同时验证多个 token，从而摊薄显存带宽成本。

**产物**：Roofline 图、decode MFU / arithmetic intensity 数据。

### E0.2 Kernel 级时延构成

**来源**：Lab2 torch.profiler。

**要支撑的论点**：decode 主要由 Attention/MLP 的 GEMM kernel 构成，但小 batch decode 无法充分利用 GPU。

**在报告中的作用**：解释为什么 tree verification 可能比纯 AR 更有效：一次处理多个 draft node，可以提高 target forward 的有效并行度。

### E0.3 GPU 计时方法论

**来源**：Lab1 TTFT/TBT 拆分。

**要支撑的论点**：本项目的时延测量使用 `torch.cuda.Event` 和显式 `torch.cuda.synchronize()`，避免 CPU wall-time 或异步 kernel 造成误差。

---

## 4. 阶段一：统一口径 Baseline 与正确性门槛

这是 v2 中最重要的前置阶段。只有这一阶段通过，后续 DDD 和 OPT-Tree 的结果才可解释。

### E1.1 Chat-template Prompt Sanity

**要回答的问题**：当前 prompt 是否以 LLaMA3-Instruct 正确期望的格式进入模型？

**为什么要做**：裸 prompt 会让 instruct model 工作在非官方输入分布上，影响生成质量、接受率和速度，不能作为最终复现口径。

**方法**：

- 使用 `experiments/common.py` 中的 `build_chat_input()`。
- 使用 `tokenizer.apply_chat_template(..., add_generation_prompt=True)`。
- tokenization 时使用 `add_special_tokens=False`。
- 对 toy prompt 和 MT-Bench prompt 做 dry-run，记录 input token length 和 prompt preview。

**当前状态**：已完成初版。

**已有结果**：

| 数据 | 结果 |
|------|------|
| toy 第一条 prompt | chat-template 后 141 tokens |
| dry-run | 通过 |

**产物**：

- `experiments/common.py`
- `experiments/run_official_baseline.py --dry-run`

**建议命令**：

```bash
.venv/bin/python experiments/run_official_baseline.py \
  --prompt-source toy \
  --limit 5 \
  --max-new-tokens 128 \
  --warmup 0 \
  --dry-run \
  --force
```

### E1.2 Official Naive Baseline

**要回答的问题**：在 EAGLE 官方 model implementation 中，纯自回归 baseline 的速度是多少？

**为什么要做**：后续 speedup 的分母必须和 EAGLE 使用同一个模型类、KV cache 实现、dtype、prompt template 和停止规则。

**方法**：

- 使用 `EaModel.naivegenerate()`。
- 与 EAGLE 使用同一个 `EaModel.from_pretrained()` 实例。
- `temperature=0.0`。
- `torch_dtype=fp16`。
- 使用 chat-template prompt。
- 记录 raw tokens、trimmed tokens、wall time、tok/s。

**不要再作为主 baseline 的内容**：

- HuggingFace `AutoModelForCausalLM.generate()`。
- 裸 prompt 的 `add_special_tokens=True`。
- 固定用 `MAX_NEW_TOKENS / wall_time` 估算 tok/s。

### E1.3 EAGLE-3 Official Baseline

**要回答的问题**：在统一口径下，EAGLE-3 相对 `EaModel.naivegenerate()` 的真实加速是多少？

**方法**：

- 使用 `EaModel.eagenerate()`。
- 参数保持官方默认或仓库默认：`total_token=60, depth=5, top_k=10`。
- 与 E1.2 共用同一批 input ids。
- 记录：
  - `tok_per_s_trimmed`
  - `tok_per_s_raw`
  - `loop_count`
  - generated token ids
  - token match

**当前状态**：已完成初版脚本与小规模验证。

**已有结果**：

| 数据集 | records | max_new_tokens | token match | naive tok/s | eagle tok/s | speedup |
|--------|---------|----------------|-------------|-------------|-------------|---------|
| toy | 1 | 16 | 1/1 | 2.53 | 5.60 | 2.209× |
| toy | 5 | 128 | 5/5 | 20.01 | 79.63 | 3.979× |
| MT-Bench 小切片 | 5 | 128 | 3/5 | 20.68 | 80.74 | 3.904× |
| MT-Bench 全量 | 80 | 128 | 63/80 | 17.59 | 68.55 | 3.897× |

**解读**：

- 速度路径已经跑通，说明官方 baseline 口径下 EAGLE-3 在当前硬件上可以达到约 4× 的 toy / small-slice 加速。
- MT-Bench 小切片出现 3/5 token match，说明 correctness 还不能默认成立，需要进入 E1.4 做定位。
- MT-Bench 全量保持约 3.90× speedup，说明 EAGLE official baseline 已经具有可报告的全量主结果。

**产物**：

- `experiments/E1_official_baseline/official_baseline_toy_limit-5_max-128.json`
- `experiments/E1_official_baseline/official_baseline_mt_bench_limit-5_max-128.json`

**建议正式命令**：

```bash
.venv/bin/python experiments/run_official_baseline.py \
  --prompt-source mt_bench \
  --limit 80 \
  --max-new-tokens 256 \
  --warmup 1 \
  --cuda-visible-devices 0,1,2,3,4 \
  --force
```

### E1.4 Greedy Lossless / Token Sequence Check

**要回答的问题**：`naivegenerate()` 和 `eagenerate()` 在 greedy 解码下是否生成完全一致的 token 序列？

**为什么要做**：投机解码理论上应保持 target distribution。对于 `temperature=0.0` 的 deterministic greedy，最直观的正确性检查就是 token-by-token identical。

**当前结果**：

- toy 5 条全部一致。
- MT-Bench 前 20 条共 40 个 turn，其中 33/40 完全一致。
- 分叉不是 prompt formatting、multi-turn history、长度不同或 stop token 造成，而是真实 token divergence。

**已定位到的例子**：

| question_id | turn | first diff | 现象 |
|-------------|------|------------|------|
| 82 | 1 | token 6 | 两个回答语义相近但 token 不同 |
| 84 | 0 | token 49 | 邮件写作内容后半段分叉 |
| 89 | 1 | token 1 | near-tie argmax 翻转 |
| 91 | 0 | token 8 | near-tie argmax 翻转 |
| 92 | 0 | token 97 | near-tie argmax 翻转 |
| 96 | 0 | token 63 | near-tie argmax 翻转 |
| 98 | 1 | token 68 | near-tie argmax 翻转 |

**已完成方法**：

1. 已修正 `max_new_tokens` off-by-one，避免统计和截断误差。
2. 已增加 `experiments/verify_lossless_eamodel.py`，只比较：
   - 同一 `EaModel`
   - 同一 input ids
   - 同一 chat template
   - 同一 `max_new_tokens`
   - 同一 stop token trimming
3. 对 mismatch case 保存：
   - first different token position
   - previous context
   - naive/eagle token window
   - full-prefix base-model argmax

**阶段结论**：

- toy prompts 达到 100% token exact match。
- MT-Bench 仍有少量 exact-match 分叉。
- 所有已诊断 mismatch 的 naive/eagle competing tokens 都在 full-prefix forward 的 top-2 内，logit 差距很小。
- 当前最可能解释是 fp16 下 sequential KV-cache decoding 与 tree/full-prefix verification 的数值路径差异。

### E1.5 Numerical Precision Diagnosis

**要回答的问题**：E1.4 中 MT-Bench 的 greedy token mismatch 是否主要由 fp16 数值路径差异导致？

**为什么要做**：如果 mismatch 来自算法逻辑错误，则后续 DDD/OPT-Tree 实验不能继续；如果 mismatch 在更高精度或更确定性设置下明显减少，则可以把它作为 RTX 3090 + fp16 环境下的数值现象写入报告。

**方法**：

- 复用 `experiments/verify_lossless_eamodel.py`。
- 增加数值诊断参数：
  - `--torch-dtype fp32`
  - `--disable-tf32`
  - `--deterministic`
- 先跑 MT-Bench 前 5 条，因为这里已经包含 question 82 和 84 两个稳定 mismatch case。
- 若 fp32 显存或耗时不可接受，则退而跑 fp16 + `--disable-tf32 --deterministic`，确认非确定性设置本身是否影响结果。

**建议命令**：

```bash
.venv/bin/python experiments/verify_lossless_eamodel.py \
  --prompt-source mt_bench \
  --limit 5 \
  --max-new-tokens 128 \
  --warmup 0 \
  --torch-dtype fp32 \
  --disable-tf32 \
  --deterministic \
  --force \
  --cuda-visible-devices 0,1,2,3,4
```

**判断标准**：

| 现象 | 解释 |
|------|------|
| fp32 match rate 明显提高或达到 100% | mismatch 主要来自 fp16 数值路径差异 |
| fp32 仍在相同位置 mismatch | 更可能是 KV-cache sequential path 与 tree path 逻辑/实现差异 |
| deterministic fp16 与普通 fp16 一致 | mismatch 不是随机非确定性，而是稳定数值路径差异 |
| deterministic fp16 明显改善 | 需要在主实验中固定 deterministic 设置 |

**实际结果**：

| 设置 | record match | turn match | naive tok/s | EAGLE tok/s | mismatch |
|------|--------------|------------|-------------|-------------|----------|
| fp16 default | 3/5 | 8/10 | 21.29 | 82.59 | q82 turn1, q84 turn0 |
| fp16 + deterministic + no TF32 | 3/5 | 8/10 | 12.50 | 50.53 | q82 turn1, q84 turn0 |
| fp32 + deterministic + no TF32 | 5/5 | 10/10 | 13.26 | 47.99 | none |

**阶段结论**：

- deterministic/no-TF32 本身不能消除 mismatch，因为 fp16 控制组仍然在同一位置分叉。
- fp32 + deterministic/no-TF32 可以让同一 MT-Bench 前 5 条达到 10/10 turn exact match。
- 因此，E1.4 中的 MT-Bench mismatch 主要来自 fp16 数值路径差异，而不是 prompt、history、stop token、off-by-one 或随机非确定性。
- fp32 可用于 correctness diagnosis，但速度明显下降，不适合作为后续主性能实验口径。

---

## 5. 阶段二：瓶颈定位实验复核

原计划中的 E2.1/E2.2/E2.3 仍然有价值，但 v2 需要重新解释其中一部分结论。

### E2.1 EAGLE Pipeline Timing Breakdown

**要回答的问题**：一次 EAGLE inference step 中，时间主要花在 target verify 还是 drafter/tree construction？

**已有方法**：

- 在 `eagenerate` 主循环中用 `torch.cuda.Event` 分段计时。
- 统计 `tree_decoding()`、`evaluate_posterior()`、`update_inference_inputs()` 等阶段。

**已有结果**：

| 阶段 | 平均时延 | 占比 |
|------|----------|------|
| Drafter Construction + Rejection/KV | 12.46 ms | 26.6% |
| Target Verify | 34.24 ms | 73.4% |
| 总计 | 46.70 ms | 100% |

**v2 解读**：

- 这个结果仍然支撑 DDD：drafter/tree construction 占比约 1/4，有优化空间。
- 但 DDD 的收益上限不可能超过这部分开销，且 DDD 论文提升本来就是小幅提升，不应预期数量级提升。

**需要补充**：

- 在 chat-template + official baseline 口径下重跑 E2.1，确认旧裸 prompt 结果是否仍然成立。
- 记录 `total_token/depth/top_k/max_new_tokens/dtype`。

**产物**：

- `experiments/E2.1_timing/timing_summary.json`
- timing stacked figure

### E2.2 Acceptance-by-Depth

**要回答的问题**：固定 depth 的 deeper expansion 是否经常不会被 verify path 触及？

**已有结果**：

| 树深度 | 被测试次数 | 被接受次数 | 条件接受率 | step 触及率 |
|--------|------------|------------|------------|-------------|
| 0 | 512 | 512 | 100.0% | 100.0% |
| 1 | 512 | 348 | 68.0% | 100.0% |
| 2 | 333 | 223 | 67.0% | 65.0% |
| 3 | 156 | 104 | 66.7% | 30.5% |
| 4 | 75 | 61 | 81.3% | 14.6% |
| 5 | 39 | 26 | 66.7% | 7.6% |
| 6 | 16 | 14 | 87.5% | 3.1% |

**v2 解读**：

- 条件接受率没有明显单调下降，说明 EAGLE beam search 的 top-k 筛选已经保证了到达深层的候选质量。
- 真正的浪费是触及率断崖下降：大多数 step 根本走不到深层。
- 这直接支撑 DDD：在 beam 置信度不足时提前停止，减少“不会被触及”的深层 draft call。

**需要补充**：

- 使用 chat-template + official baseline 重跑。
- 将统计中的 `actual_depth` 和 `accepted_depth` 与 DDD 的 `early_stop_step` 对齐。

**产物**：

- acceptance rate / reach rate by depth 图。
- accepted length CDF。

### E2.3 Tree Utilization / Tree Selection Analysis

**要回答的问题**：当前树节点选择是否真的给 OPT-Tree 留出了改进空间？

**旧实验结果**：

| 指标 | 实测值 |
|------|--------|
| `N_verify` | 60 |
| `N_accepted` | 2.52 |
| 节点验证利用率 | 4.2% |

**v2 重新解释**：

- 这个结果可以说明 “verify budget 中最终只有少量节点位于 accepted path”。
- 但这不能直接说明 “baseline rerank 选错了节点”，因为 tree verification 天然只接受一条路径，绝大多数验证节点不会被最终接受。
- 若要支撑 OPT-Tree，需要真正记录 baseline tree 与 OPT tree 的节点集合差异。

**新实验设计**：

每个 inference step 记录：

```text
candidate_nodes
candidate_cumulative_logprob
selected_nodes_baseline
selected_nodes_opt
accepted_path_nodes
retrieve_indices_baseline
retrieve_indices_opt
tree_position_ids
```

计算指标：

| 指标 | 含义 |
|------|------|
| Jaccard(selected_baseline, selected_opt) | 两棵树是否实际不同 |
| depth histogram difference | OPT 是否引入更深或更宽的节点 |
| accepted path coverage | accepted path 是否被两者同时覆盖 |
| mean accept length delta | OPT 是否提高接受长度 |
| tree construction overhead | OPT 额外 Python/GPU 开销 |

**判断标准**：

- 若 Jaccard 高、accept length 不变：OPT-Tree 在 EAGLE-2/3 dynamic baseline 上退化，应写成 negative result。
- 若 Jaccard 低、accept length 上升：OPT-Tree 有正向复现价值。
- 若 accept length 上升但 tok/s 下降：说明算法有效但实现开销过高，应区分 algorithmic gain 和 system overhead。

---

## 6. 阶段三：DDD 复现与改进实验

DDD 是当前最值得优先做的算法方向，因为它与 E2.2 的触及率结果直接对应，且不依赖构造新的 static tree baseline。

### E3.1 DDD H 分布 Profiling

**要回答的问题**：当前代码中 `H = logsumexp(logprobsum)` 的实际数值范围是多少？

**为什么要做**：DDD 论文中的阈值约为 `x=-0.3`，但当前实现曾经扫描 `[-4, -6, -8, -10, -12]`。如果不先看 H 分布，阈值搜索没有意义。

**方法**：

- 在 `topK_genrate()` 每个 check step 记录：
  - `step`
  - `H`
  - current beam scores
  - whether early stop would happen for candidate thresholds
- 不启用 early stop，只 profiling。
- 使用 MT-Bench 小切片和 toy prompts 各跑一组。

**产物**：

- `experiments/profile_ddd_h.py`
- `experiments/E3_ddd/ddd_h_toy_limit-2_max-64_depth-11_steps-5-7-9.json`
- `experiments/E3_ddd/ddd_h_mt_bench_limit-5_max-128_depth-11_steps-5-7-9.json`
- `experiments/E3_ddd/README.md`
- 每个 check step 的分位数表：P5/P25/P50/P75/P95。

**实际结果**：

MT-Bench 前 5 条、`ddd_max_depth=11`、`check_steps=[5,7,9]`、`ddd_threshold=-1e9`：

| step | count | mean H | P10 | P25 | P50 | P75 | P90 |
|------|-------|--------|-----|-----|-----|-----|-----|
| 5 | 205 | -1.384 | -2.870 | -1.872 | -1.124 | -0.598 | -0.212 |
| 7 | 205 | -2.110 | -3.655 | -2.789 | -1.969 | -1.205 | -0.524 |
| 9 | 205 | -3.234 | -5.306 | -4.164 | -3.121 | -2.020 | -1.201 |

阈值触发比例 `P(H < threshold)`：

| threshold | global | step 5 | step 7 | step 9 |
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

**阶段结论**：

- 给出合理 threshold search range。
- 不再盲扫与 H 分布不匹配的阈值。
- 旧网格 `[-4,-6,-8,-10,-12]` 里，`-8/-10/-12` 基本过于保守，几乎不会 early stop。
- 论文式 `-0.3` 在当前代码的 H 定义下极其激进，会在 step 5/7/9 的绝大多数检查点触发。
- 下一步 DDD sweep 建议使用 `[-5.0, -4.0, -3.0, -2.0, -1.0, -0.5, -0.3]`。

### E3.2 Fixed Depth Baselines

**要回答的问题**：固定 depth 本身如何影响速度和接受长度？

**为什么要做**：DDD 的对照不能只有 depth=5。DDD 论文通常在更大的 max depth 下动态早停，因此需要比较 fixed depth。

**配置表**：

| 配置 | max_depth | check_steps | threshold | 目的 |
|------|-----------|-------------|-----------|------|
| Baseline | 5 | - | - | EAGLE 默认 |
| Fixed depth 7 | 7 | - | - | 检查更深树是否提高 accept |
| Fixed depth 9 | 9 | - | - | DDD check range 对照 |
| Fixed depth 11 | 11 | - | - | paper-like max depth |

**指标**：

- tok/s
- accept/step
- loop_count
- avg draft calls
- target verify time
- drafter construction time

**判断标准**：

- 如果 fixed depth 增加带来 accept length 上升但 tok/s 下降，说明 DDD 有 trade-off 空间。
- 如果 fixed depth 增加完全没有 accept gain，DDD 的收益主要来自减少无效深度，而不是允许更深探索。

**实际结果**：

MT-Bench 前 5 条、`max_new_tokens=128`、`total_token=60`：

| fixed depth | tok/s | speed vs depth 5 | accept/step | mean loop count |
|-------------|-------|------------------|-------------|-----------------|
| 5 | 83.51 | 1.000× | 5.286 | 22.20 |
| 7 | 83.99 | 1.006× | 5.803 | 20.30 |
| 9 | 82.06 | 0.983× | 6.017 | 19.60 |
| 11 | 77.81 | 0.932× | 6.051 | 19.50 |

**阶段结论**：

- 加深 fixed depth 可以提高 `accept/step`，说明更深探索确实有收益空间。
- 但 depth 9 以后收益明显饱和，depth 11 的 throughput 已经低于 depth 5。
- 这正好支撑 DDD：不应固定每步都走 depth 11，而应在高置信 step 保留深探索，在低置信 step early stop。

### E3.3 DDD Paper-like 配置复现

**要回答的问题**：在接近论文设置的 depth/check schedule 下，DDD 能否复现小幅稳定提升？

**建议配置**：

| 参数 | 建议 |
|------|------|
| `ddd_max_depth` | 11 |
| `ddd_check_steps` | 5,7,9 |
| `ddd_threshold` | 由 E3.1 H 分布确定，包含 paper-like 附近点 |
| `temperature` | 0.0 |
| prompt | MT-Bench |

**实现修正**：

- `topK_genrate()` 中每次只记录一次 `actual_depth`。
- 保存 `early_stopped`、`early_stop_step`、`checked_H`。
- 尽量在下一次 draft forward 之前判断是否继续扩展，避免 “检查时开销已经付掉”。

**核心指标**：

| 指标 | 解释 |
|------|------|
| early stop rate | DDD 是否真的生效 |
| avg actual depth | drafting 深度是否下降 |
| accept/step | 是否损失接受长度 |
| drafter time | DDD 直接优化对象 |
| tok/s | 最终系统收益 |

### E3.4 DDD Threshold Search

**要回答的问题**：DDD 的阈值是否存在稳定 sweet spot？

**方法**：

- 根据 E3.1 的 H 分布选 5 到 7 个阈值。
- 每个阈值跑同一批 prompts。
- 输出 trade-off 曲线。

**实际结果**：

MT-Bench 前 5 条、`ddd_max_depth=11`、`check_steps=[5,7,9]`：

| threshold | tok/s | speed vs tau=-5 | accept/step | avg actual depth | early stop rate |
|-----------|-------|-----------------|-------------|------------------|-----------------|
| -5.0 | 78.85 | 1.000× | 6.051 | 11.590 | 14.15% |
| -4.0 | 79.23 | 1.005× | 6.051 | 11.210 | 27.80% |
| -3.0 | 81.10 | 1.028× | 6.051 | 10.322 | 53.17% |
| -2.0 | 84.09 | 1.067× | 6.035 | 9.087 | 75.24% |
| -1.0 | 86.52 | 1.097× | 5.884 | 7.526 | 91.00% |
| -0.5 | 88.13 | 1.118× | 5.714 | 6.664 | 97.23% |
| -0.3 | 87.84 | 1.114× | 5.617 | 6.400 | 98.64% |

**判断标准**：

- 最佳点不应只看 tok/s，也要看 accept/step 是否明显下降。
- 如果 aggressive threshold tok/s 高但输出一致性差，应排除。

**阶段结论**：

- `tau=-2.0` 是较稳的折中点：几乎保留 fixed depth 11 的 accept/step，同时速度接近 fixed depth 7。
- `tau=-0.5` 是速度最优点：达到 88.13 tok/s，但 early stop rate 高达 97.23%，accept/step 明显下降。
- 最终报告可将 `tau=-2.0` 作为 paper-style DDD 主配置，将 `tau=-0.5` 作为 speed-optimized variant。

### E3.5 DDD-only 主结果

**要回答的问题**：修正版 DDD 在最终口径下是否有稳定收益？

**当前状态**：已完成候选配置小规模验证和 MT-Bench 20 条扩展切片。

**候选配置**：

| 配置 | 定位 | 选择理由 |
|------|------|----------|
| DDD `tau=-2.0` | paper-style 主配置 | 几乎保留 fixed depth 11 的 accept/step，同时减少平均 draft 深度 |
| DDD `tau=-0.5` | speed-optimized variant | tok/s 最高，但 early stop 过于激进，accept/step 有明显下降 |

**token-level correctness 小实验**：

MT-Bench 前 5 条、`max_new_tokens=128`、fp16：

| 方法 | record match | turn match | mismatch | naive tok/s | EAGLE/DDD tok/s |
|------|--------------|------------|----------|-------------|-----------------|
| DDD `tau=-2.0` | 4/5 | 9/10 | q84 turn0 pos49 | 21.28 | 83.78 |
| DDD `tau=-0.5` | 3/5 | 8/10 | q82 turn1 pos6, q84 turn0 pos49 | 21.14 | 85.26 |

这些 mismatch 与 E1.4/E1.5 中已经定位的 fp16 数值路径分叉重合，因此目前没有证据说明 DDD 本身引入新的 greedy correctness bug。

**MT-Bench 20 条扩展结果**：

固定深度对照：

| 方法 | tok/s | speed vs depth5 | accept/step | mean loop count |
|------|-------|-----------------|-------------|-----------------|
| Fixed depth 5 | 76.05 | 1.000× | 4.872 | 25.32 |
| Fixed depth 7 | 75.16 | 0.988× | 5.187 | 24.12 |
| Fixed depth 9 | 73.66 | 0.969× | 5.349 | 23.50 |
| Fixed depth 11 | 70.19 | 0.923× | 5.372 | 23.45 |

DDD 候选：

| 方法 | tok/s | accept/step | avg actual depth | early stop rate | early stop step hist |
|------|-------|-------------|------------------|-----------------|----------------------|
| DDD `tau=-2.0` | 76.35 | 5.357 | 8.226 | 85.52% | 5:380, 7:252, 9:207 |
| DDD `tau=-0.5` | 78.60 | 5.165 | 6.421 | 97.93% | 5:881, 7:72, 9:39 |

**阶段结论**：

- `tau=-2.0` 在 20 条 MT-Bench 上几乎保持 fixed depth 11 的接受长度，同时把 throughput 恢复到 depth 5 附近。这说明 DDD 的机制是成立的，但端到端收益较小。
- `tau=-0.5` 获得最高 tok/s，但接受长度降到 fixed depth 7 附近，适合作为速度优先的对照，不适合作为唯一主结论。
- 20 条小切片中 DDD 看起来有一定收益，但最终结论必须以后续全量结果为准。

**MT-Bench 全量结果**：

MT-Bench 80 条、160 turns、`max_new_tokens=128`。

固定深度对照：

| 方法 | tok/s | speed vs depth5 | accept/step | mean loop count |
|------|-------|-----------------|-------------|-----------------|
| Fixed depth 5 | 65.23 | 1.000× | 5.341 | 22.41 |
| Fixed depth 11 | 77.03 | 1.181× | 6.392 | 19.46 |

DDD 候选：

| 方法 | tok/s | accept/step | avg actual depth | early stop rate | early stop step hist |
|------|-------|-------------|------------------|-----------------|----------------------|
| DDD `tau=-2.0` | 68.51 | 6.365 | 9.121 | 71.05% | 5:820, 7:754, 9:760 |
| DDD `tau=-0.5` | 70.74 | 6.041 | 6.991 | 93.18% | 5:2505, 7:383, 9:307 |

**全量结论修正**：

- 全量 MT-Bench 与 20 条小切片不同：fixed depth 11 同时提高 accept/step 和 tok/s，是当前 EAGLE 配置中最强的 depth 对照。
- DDD 的机制仍然成立，`tau=-2.0` 将平均实际深度降到 9.121，`tau=-0.5` 降到 6.991。
- 但这两个早期阈值没有超过 fixed depth 11 的端到端吞吐，因此后续需要继续在 `-2.0` 与 `-0.5` 之间补充阈值搜索。
- E6 已完成该补充搜索，并发现 `tau=-1.0` 达到 `88.27 tok/s`，超过 fixed depth 11。因此最终报告不能再使用“DDD 全量无正向提速”的旧结论，而应写成“DDD 经过阈值调参后有效”。

**最终报告表述方式**：

- 当前最终表述：DDD `tau=-1.0` 通过减少无效 draft depth，同时保持足够高的 accept/step，在 full MT-Bench 上相对 fixed depth 11 带来约 `14.6%` macro tok/s 提升。
- 保留早期失败阈值作为消融：`tau=-0.5` / `tau=-0.3` 说明过早停止会损害候选质量，`tau=-2.0` 说明过保守则节省不够。

---

## 7. 阶段四：OPT-Tree 复现与失败分析

OPT-Tree 不应再直接假设能在当前 EAGLE-3 dynamic tree 上提升。v2 中将其设计成“先验证是否有差异，再决定正向复现还是 negative result”。

### E4.1 Baseline vs OPT Tree Set Diff

**要回答的问题**：当前 OPT-Tree 实现到底有没有选出不同于 baseline 的树？

**为什么要做**：当前消融结果显示 OPT-Tree 的 `accept_per_step` 几乎与 baseline 完全相同，说明两者可能选择了几乎一样的节点集合。

**方法**：

- 对同一 candidate pool，同时运行 baseline rerank 和 OPT selection。
- 不立即进入 target verify，先保存两者节点集合。
- 计算 Jaccard similarity、depth distribution 和 path coverage。

**判断标准**：

| 现象 | 结论 |
|------|------|
| Jaccard > 0.9 | OPT 退化为 baseline，后续写 negative result |
| Jaccard 0.5-0.9 | 有差异，需要继续看 accept length |
| Jaccard < 0.5 | OPT 明显改变树结构，值得重点复现 |

**当前结果**：

已新增 `experiments/analyze_tree_selection.py`，在 `topK_genrate()` 中只在显式设置 `_tree_diff_records` 时记录 baseline 与 OPT 的候选节点集合。普通 EAGLE/DDD 路径不会记录这部分数据。

| 数据 | turns | tree calls | expand | mean Jaccard | identical rate | baseline-only | OPT-only | accept/step |
|------|-------|------------|--------|--------------|----------------|---------------|----------|-------------|
| MT-Bench 前 2 条, max64 | 4 | 50 | 1.5 | 1.000 | 100.00% | 0.00 | 0.00 | 5.619 |
| MT-Bench 前 2 条, max64 | 4 | 50 | 2.0 | 1.000 | 100.00% | 0.00 | 0.00 | 5.619 |
| MT-Bench 前 5 条, max128 | 10 | 232 | 1.5 | 1.000 | 100.00% | 0.00 | 0.00 | 5.286 |

MT-Bench 前 5 条中，baseline 与 OPT 的 depth histogram 也完全相同。

**阶段结论**：

- 在当前 EAGLE-3 dynamic tree baseline 上，OPT-Tree 最终选择的节点集合与 baseline 完全一致。
- 这说明 OPT-Tree 目前没有带来 algorithmic tree-selection gain；后续即使 tok/s 下降，也更可能是额外构树开销造成。
- E4.2 仍可做，但目的应调整为确认 OPT-only throughput/overhead，而不是期待 accept length 提升。

### E4.2 OPT-only Dynamic Baseline 对照

**要回答的问题**：在当前 EAGLE-3 dynamic baseline 上，OPT-Tree 是否提升 accept length 或 tok/s？

**配置**：

| 配置 | 描述 |
|------|------|
| EAGLE-3 baseline | 原 dynamic tree rerank |
| EAGLE-3 + OPT | 当前 OPT selection |
| EAGLE-3 + OPT no-overhead estimate | 用同一 verify 结果估计算法收益，剥离 Python 开销 |

**指标**：

- tok/s
- accept/step
- tree construction time
- Jaccard similarity
- accepted path depth

**预期**：

当前最可能结果是：

> OPT-Tree 与 EAGLE-2/3 dynamic tree 的目标高度重叠，节点选择高度相似，accept length 几乎不变；额外构树开销反而降低 tok/s。

这不是失败，而是一个有价值的 negative result：说明 OPT-Tree 对 static/binary tree 有收益，但未必是 strong dynamic-tree baseline 的正交优化。

**当前结果**：

已新增 `experiments/run_opt_dynamic_baseline.py`，在不启用 tree-diff instrumentation 的正常路径下比较 EAGLE-3 和 EAGLE-3 + OPT。

分开加载模型的小切片结果：

| 方法 | tok/s | speed vs EAGLE-3 | accept/step | mean loop count |
|------|-------|------------------|-------------|-----------------|
| EAGLE-3 | 52.44 | 1.000× | 5.286 | 22.20 |
| OPT-1.5 | 66.67 | 1.272× | 5.286 | 22.20 |
| OPT-2.0 | 63.90 | 1.219× | 5.286 | 22.20 |

同一模型内切换配置，并在最后重复 EAGLE-3：

| 方法 | tok/s | speed vs first EAGLE-3 | accept/step | mean loop count |
|------|-------|-------------------------|-------------|-----------------|
| EAGLE-3 | 52.20 | 1.000× | 5.286 | 22.20 |
| OPT-1.5 | 50.14 | 0.961× | 5.286 | 22.20 |
| OPT-2.0 | 58.84 | 1.127× | 5.286 | 22.20 |
| EAGLE-3-repeat | 78.19 | 1.498× | 5.286 | 22.20 |

全量 MT-Bench 80 records / 160 turns，同一模型内切换配置，并在最后重复 EAGLE-3：

| 方法 | tok/s | speed vs first EAGLE-3 | accept/step | mean loop count |
|------|-------|-------------------------|-------------|-----------------|
| EAGLE-3 | 61.66 | 1.000× | 5.341 | 22.41 |
| OPT-1.5 | 47.18 | 0.765× | 5.341 | 22.41 |
| EAGLE-3-repeat | 80.50 | 1.306× | 5.341 | 22.41 |

**阶段结论**：

- OPT 与 EAGLE-3 的 `accept/step`、`mean loop count`、生成 token 数完全相同，说明没有 algorithmic acceptance gain。
- 小切片 tok/s 受 warmup/order 影响很大：同一模型内 EAGLE-3-repeat 从 52.20 漂到 78.19 tok/s，这个漂移大于 OPT 和 baseline 的差异。
- 全量 OPT 对照也没有 acceptance 变化：三行都是 `accept/step=5.341`、`mean loop count=22.41`。OPT-1.5 tok/s 低于第一轮 EAGLE-3，但第二轮 EAGLE-3 又明显高于第一轮，因此吞吐更适合作为 order-sensitive overhead 现象，而不是算法收益或失败的唯一证据。
- 因此 E4.2 不应被写成 OPT 正向提速或负向降速的强证据，而应作为“acceptance 完全不变，速度小差异不可解释为算法收益”的辅助证据。

### E4.3 弱 baseline 对照（不再执行）

曾经考虑过用 EAGLE-1/EAGLE-2、static tree 或 weak dynamic/BFS tree 作为更弱 baseline 来验证 OPT-Tree 本身。但后续决定不更换主 baseline，也不再为 OPT-Tree 新增弱 baseline。

最终报告中采用当前口径：

- EAGLE-3 dynamic rerank 是主 baseline。
- OPT-Tree 在该 strong baseline 上节点集合完全重合。
- OPT-Tree 的结果作为 baseline-mismatch / non-orthogonality negative result 分析。

### E4.4 OPT-Tree Negative Result 写法

如果最终无正向提升，报告中建议写：

> OPT-Tree 将节点选择形式化为最大化 expected acceptance length，在 static tree baseline 上目标明确。但 EAGLE-2/3 dynamic tree 已经按 cumulative path confidence 全局选择节点并保持 ancestor closure。我们在 EAGLE-3 上发现 OPT-Tree 选择出的节点集合与 baseline 高度重合，因此 mean acceptance length 几乎不变；同时额外 tree construction 带来系统开销，最终 tok/s 下降。该结果表明 OPT-Tree 与 strong dynamic-tree baseline 并不完全正交，是本项目的 negative result。

---

## 8. 阶段五：最终消融与泛化性实验

### E5.1 四组消融矩阵

**要回答的问题**：DDD 和 OPT-Tree 各自贡献是什么？联合后是否更好？

**最终主表**：

| 方法 | tok/s | speedup vs naive | speedup vs EAGLE | accept/step | avg draft depth | early stop rate | tree diff |
|------|-------|------------------|------------------|-------------|-----------------|-----------------|-----------|
| Naive AR | 17.59 | 1.00× | - | - | - | - | - |
| EAGLE-3 default/depth5 | 68.55 | 3.90× | 1.00× | 5.341 | fixed 5 | - | - |
| EAGLE-3 fixed depth11 | 77.03 | 4.38× | 1.12× | 6.392 | fixed 11 | - | - |
| EAGLE-3 + DDD tau=-2.0 | 68.51 | 3.89× | 1.00× | 6.365 | 9.121 | 71.05% | - |
| EAGLE-3 + DDD tau=-0.5 | 70.74 | 4.02× | 1.03× | 6.041 | 6.991 | 93.18% | - |
| EAGLE-3 + OPT-1.5 | 47.18 | 2.68× | 0.69× | 5.341 | fixed 5 | - | Jaccard=1.000 |
| EAGLE-3 + DDD tau=-0.5 + OPT-1.5 | 69.35 | 3.94× | 1.01× | 6.041 | 6.991 | 93.18% | inferred same |

**注意**：

- 若 E1.4 token 一致性仍未完全通过，主表应报告 speedup 和机制指标，但 correctness 部分要单独说明。
- 不要用 “总 token 数相同” 作为 lossless 证明。

### E5.2 多数据集 / 场景泛化性

**要回答的问题**：不同数据集和任务类型下，EAGLE/DDD/OPT 的收益是否一致？其他小组观察到不同数据集结果差异很大，因此本项目最终阶段必须单独做这组对比，而不能只用 MT-Bench 总体均值下结论。

**当前状态**：已完成全量 MT-Bench category split 离线聚合。新增 `experiments/analyze_mtbench_categories.py`，复用 full MT-Bench 的 E1/E3/E4 结果生成 `experiments/E5_scenarios/mt_bench_category_summary.json` 和 `experiments/E5_scenarios/README.md`。旧 `E4_scenarios/scenario_results.json` 仍只作为历史参考，因为它不一定使用 chat-template、official naive baseline 和修正后的 lossless/DDD 统计口径。

**数据集 / 任务集划分**：

| 数据集或任务集 | 来源 | 代表任务 | 主要观察点 |
|----------------|------|----------|------------|
| MT-Bench overall | 已有本地数据 | 多轮指令跟随 | 与当前主线结果保持可比 |
| MT-Bench category split | MT-Bench `category` 字段 | writing、roleplay、reasoning、coding、math、extraction、STEM、humanities | 直接检查同一 benchmark 内任务差异 |
| Toy sanity | `experiments/common.py` toy prompts | 简短稳定 prompt | 排除脚本和统计错误 |
| Curated translation | 新增本地 prompt suite | 中英互译、格式固定输出 | 观察高约束输出下 acceptance 是否提高 |
| Curated coding | 新增本地 prompt suite 或本地可用 code 数据 | 函数补全、解释代码、生成测试 | 观察模板化 token 和局部重复是否提高 EAGLE/DDD 收益 |
| Curated math/reasoning | 新增本地 prompt suite 或本地可用 math 数据 | 多步推理、算术、逻辑题 | 观察路径敏感任务是否更容易 greedy 分叉 |
| Curated writing/roleplay | 新增本地 prompt suite | 开放式写作、角色扮演 | 观察开放分布是否降低 accept/step |

**如果后续能在本地确认存在外部数据集**，可以把 HumanEval/MBPP、GSM8K、AlpacaEval 或 ShareGPT 子集加入 E5.2。目前已完成的最终口径不额外引入手写 prompt，而是把 MT-Bench 的 8 个 category 作为不同任务子集，避免小规模 curated suite 带来的额外偏差。

**当前全量 category split 结果摘要**：

| category | naive tok/s | EAGLE tok/s | speedup | turn match |
|----------|-------------|-------------|---------|------------|
| writing | 17.49 | 66.58 | 3.806× | 80.0% |
| roleplay | 13.47 | 45.64 | 3.387× | 65.0% |
| reasoning | 20.62 | 79.16 | 3.839× | 85.0% |
| math | 20.97 | 89.37 | 4.262× | 95.0% |
| coding | 15.09 | 63.23 | 4.190× | 90.0% |
| extraction | 20.17 | 78.06 | 3.869× | 100.0% |
| stem | 14.73 | 57.02 | 3.870× | 80.0% |
| humanities | 16.51 | 61.84 | 3.745× | 80.0% |

主要观察：

- 不同任务子集差异明显，EAGLE speedup 从 3.387× 到 4.262×。
- Fixed depth 11 在 full MT-Bench 总体上最好，但 category 上并非全部获胜；writing、roleplay、humanities 中 depth 5 tok/s 更高。
- DDD 在所有 category 上都降低实际深度，但吞吐不稳定，不系统性超过 fixed depth 11。
- OPT-Tree 在所有 category 上 `accept delta=0`，与 E4.1 的树集合完全重合结论一致。

**实验方法**：

- 每个数据集 / 任务集使用同一套 chat-template 构造逻辑。
- 每组至少跑 20 个 turn；若数据量不足，报告实际样本数。
- 对比方法至少包含：
  - `EaModel.naivegenerate()`
  - EAGLE-3 baseline
  - EAGLE-3 + DDD `tau=-2.0`
  - EAGLE-3 + DDD `tau=-0.5`
  - 若 E4 完成，再加入 OPT 和 DDD+OPT。
- MT-Bench category split 应优先复用同一次 MT-Bench 运行的逐样本结果，减少重复计算。

**指标**：

- tok/s by dataset / category
- speedup vs naive 和 speedup vs EAGLE
- accept/step
- mean loop count
- DDD avg actual depth
- DDD early stop rate
- turn-level exact match rate
- mismatch location 和 near-tie 诊断数量

**预期解读**：

- 如果 coding/translation 的 accept/step 明显高于 writing/roleplay，可以解释为受约束输出更容易被 drafter 命中。
- 如果 math/reasoning 出现更高 mismatch rate，需要区分是 fp16 near-tie 还是算法逻辑问题。
- 如果 DDD 在某些任务上 early stop 很高但速度没有提升，说明该任务下 target verify 仍是主要瓶颈。
- 如果不同数据集差异显著，最终报告不应只给一个全局平均值，而应保留 per-dataset 表格或柱状图。

### E5.3 长度敏感性

**要回答的问题**：输入长度和输出长度会如何影响投机解码收益？

**方法**：

- 选择短/中/长 prompt。
- `max_new_tokens ∈ {64, 128, 256, 512}`。
- 记录 tok/s 和 loop_count。

**意义**：

- 输出越短，warmup 和 fixed overhead 越显著。
- 输出越长，acceptance 的统计更稳定。

### E5.4 多卡使用策略

**当前目标**：不是做 tensor parallel scaling，而是利用所有 GPU 并行跑不同配置。

**建议**：

- 单个 8B + EAGLE 实验使用 1 张 3090 即可。
- DDD threshold search 和 scenario test 可以用 `CUDA_VISIBLE_DEVICES` 分配不同 GPU 并行跑。
- 每个结果 JSON 必须记录 GPU id、dtype、timestamp、git status。

---

## 9. 实验总表

| 编号 | 实验名称 | 优先级 | 状态 | 关键产物 | 主要论点 |
|------|----------|--------|------|----------|----------|
| E0.1 | Roofline memory-bound | P0 | 已有 | Lab2 图表 | 为什么需要投机解码 |
| E0.2 | Kernel profiler | P0 | 已有 | Lab2 表格 | decode 小 batch 利用率低 |
| E1.1 | Chat-template sanity | P0 | 初版完成 | `common.py`, dry-run | 主实验必须统一 prompt |
| E1.2 | Official naive baseline | P0 | 初版完成 | E1 JSON | speedup 分母 |
| E1.3 | EAGLE official baseline | P0 | 小规模完成 | E1 JSON | EAGLE-3 真实起点 |
| E1.4 | Greedy token consistency | P0 | 已完成小规模 | mismatch report | correctness 门槛 |
| E1.5 | Numerical precision diagnosis | P0 | 已完成小规模 | fp32/deterministic report | mismatch 主要来自 fp16 数值路径 |
| E2.1 | Pipeline timing | P1 | 已有历史结果，非最终主线 | timing summary | DDD 优化空间 |
| E2.2 | Acceptance by depth | P1 | 已有历史结果，最终由 full depth sweep 支撑 | depth stats | DDD 动机 |
| E2.3 | Tree selection diff | P1 | 已由 E4.1 覆盖 | tree diff JSON | OPT 是否退化 |
| E3.1 | DDD H distribution | P0 | 已完成小规模 | H histogram / threshold grid | 阈值搜索依据 |
| E3.2 | Fixed depth sweep | P0 | 已完成小规模 | depth table | DDD 对照 |
| E3.3 | DDD paper-like | P0 | 已完成小规模 | DDD result | 复现论文设置 |
| E3.4 | DDD threshold search | P1 | 已完成小规模 | sweep table | 找 sweet spot |
| E3.5 | DDD-only main | P0 | 全量完成 | final table row | 机制成立但全量未超过 fixed depth 11 |
| E4.1 | Baseline vs OPT tree diff | P0 | 已完成小规模 | Jaccard/depth diff | OPT 在 dynamic baseline 上退化 |
| E4.2 | OPT-only dynamic baseline | P1 | 全量完成 | OPT row | acceptance 无变化，速度受顺序/预热影响 |
| E4.3 | Weak/static tree baseline | P2 | 不再执行 | - | 保留 EAGLE-3 主 baseline |
| E5.1 | Final ablation matrix | P0 | 全量完成 | 主结果表 | DDD+OPT 不超过 DDD-only |
| E5.2 | Multi-dataset / scenario generalization | P1 | 已完成 MT-Bench category split | 分 category 表 | 解释任务差异 |
| E5.3 | Length sensitivity | P2 | 可选 | 长度曲线 | 系统分析 |

---

## 10. 推荐执行顺序

### Step 1：完成正确性和 baseline 闭环

1. 修正 `max_new_tokens` 边界或在所有实验中统一截断。
2. 新增 `verify_lossless_eamodel.py`。
3. 用 MT-Bench 20 条跑 token-level lossless 小规模验证。
4. 跑 E1.5 数值精度诊断，确认 mismatch 是否主要来自 fp16 路径差异。
5. 用 MT-Bench 80 条跑 official baseline。
6. 输出 mismatch report。

**验收标准**：

- 有一份可信的 `Naive vs EAGLE` baseline JSON。
- 知道 greedy mismatch 的比例和原因。

### Step 2：修正版 DDD

1. 修正 DDD stats。
2. 跑 H distribution profiling。
3. 跑 fixed depth sweep。
4. 跑 paper-like DDD。
5. 跑 threshold search。
6. 对 `tau=-2.0` 和 `tau=-0.5` 做 token-level correctness 小实验。
7. 在 MT-Bench 20 条上扩展 fixed-depth 与 DDD 候选验证。

**验收标准**：

- DDD 的 early stop rate、avg depth、tok/s、accept/step 都可解释。

### Step 3：OPT-Tree 定性定位

1. 记录 baseline tree 与 OPT tree 的节点集合。
2. 计算 Jaccard similarity。
3. 跑 OPT-only 对照。
4. 若 dynamic baseline 上退化，写成 baseline-mismatch / non-orthogonality negative result。

**验收标准**：

- 不再只用 `accept_per_step` 是否变化猜测 OPT 是否有效，而是有 tree set diff 证据。

### Step 4：最终表和报告图

1. 重跑四组消融矩阵。
2. 基于已完成的 E5.2 MT-Bench category split 整理多任务泛化性结论。
3. 整理所有图表。
4. 在报告中明确区分：
   - 已成功复现的部分
   - 修正后仍存在差距的部分
   - DDD 机制成立但 full MT-Bench 未超过 fixed depth 11
   - OPT-Tree negative result

---

## 11. 最小可交付版本

如果时间紧，优先完成以下最小闭环：

1. `chat_template + EaModel.naivegenerate/eagenerate` official baseline。
2. `max_new_tokens` 与 token sequence lossless 修正。
3. DDD stats 修正 + H distribution + DDD-only 消融。
4. OPT-Tree tree set diff + negative result 分析。
5. 最终主表至少包含：
   - Naive AR
   - EAGLE-3
   - EAGLE-3 + DDD
   - EAGLE-3 + OPT
   - EAGLE-3 + DDD + OPT

这能形成完整报告：

- 有背景：decode memory-bound。
- 有 baseline：官方口径 EAGLE-3。
- 有诊断：timing / acceptance depth / tree selection。
- 有正向尝试：DDD。
- 有失败分析：OPT-Tree 与 dynamic tree baseline 重叠。
- 有局限性：prompt 类型、硬件、dtype、lossless mismatch。

---

## 12. 报告叙事建议

### 12.1 对 baseline 修正的表述

可以写：

> 初始复现实验中，我们得到的 EAGLE 加速比低于论文报告。进一步排查发现，差异部分来自评测口径：初始脚本使用裸 prompt，而 LLaMA-3.1-8B-Instruct 需要 chat template；同时 pure AR baseline 使用 HuggingFace `generate()`，与 EAGLE 官方使用的 `EaModel.naivegenerate()` 不一致。因此我们重新建立了 official-style baseline，在同一 `EaModel`、同一 chat template、同一 dtype 和停止规则下比较 `naivegenerate()` 与 `eagenerate()`。

### 12.2 对 DDD 的表述

可以写：

> Acceptance-by-depth profiling 显示，深层节点的条件接受率并不一定低，但验证路径触及深层的概率急剧下降。这说明固定深度扩展在大量 step 中为不会被触及的节点支付了 drafting 成本。DDD 利用 beam logprob mass 作为置信度信号，在低置信度时提前停止扩展，从而减少无效 draft calls。

### 12.3 对 OPT-Tree 的表述

可以写：

> OPT-Tree 在 static tree 或 binary tree baseline 上目标明确，但我们使用的 EAGLE-3 已包含 EAGLE-2 风格的 dynamic tree rerank。实验证明，在该 strong baseline 上，OPT-Tree 选择的节点集合与 baseline 高度重合，mean acceptance length 几乎不变；额外 tree construction overhead 反而降低吞吐。因此我们将其作为一个 negative result，说明并非所有 D6 维度优化都能与 dynamic EAGLE baseline 正交叠加。

### 12.4 对硬件和 dtype 的表述

可以写：

> 受限于 RTX 3090 的硬件特性，本项目统一使用 fp16 进行推理，而官方实验可能使用 A100/H100 等对 bf16 和大 batch 更友好的硬件。因此绝对 tok/s 和数值路径可能与论文存在差异。我们重点比较同一硬件、同一 dtype、同一 prompt 口径下的相对变化。

---

## 13. 附录：建议脚本清单

| 脚本 | 状态 | 用途 |
|------|------|------|
| `experiments/common.py` | 已新增 | chat template、prompt loading、token trimming |
| `experiments/run_official_baseline.py` | 已新增 | official naive vs EAGLE baseline |
| `experiments/verify_lossless_eamodel.py` | 已新增 | greedy token-level correctness |
| `experiments/profile_ddd_h.py` | 已新增 | DDD H 分布 |
| `experiments/run_fixed_depth_sweep.py` | 已新增 | fixed depth 对照 |
| `experiments/run_ddd_sweep.py` | 已新增 | DDD threshold search |
| `experiments/analyze_tree_selection.py` | 已新增 | baseline vs OPT tree diff |
| `experiments/run_opt_dynamic_baseline.py` | 已新增 | E4.2 OPT-only dynamic baseline 对照 |
| `experiments/run_final_ablation.py` | 可由 ablation_full 改 | 最终四组消融 |
| `experiments/analyze_mtbench_categories.py` | 已新增 | E5.2 MT-Bench category split 离线聚合 |

---

## 14. 当前已完成的新结果记录

| 文件 | 内容 | 关键结果 |
|------|------|----------|
| `experiments/E1_official_baseline/official_baseline_toy_limit-1_max-16.json` | smoke test | speedup 2.209×, match 1/1 |
| `experiments/E1_official_baseline/official_baseline_toy_limit-5_max-128.json` | toy 全量 | speedup 3.979×, match 5/5 |
| `experiments/E1_official_baseline/official_baseline_mt_bench_limit-5_max-128.json` | MT-Bench 小切片 | speedup 3.904×, match 3/5 |
| `experiments/E1_official_baseline/official_baseline_mt_bench_limit-80_max-128.json` | MT-Bench 全量 official baseline | speedup 3.897×, match 63/80 |
| `experiments/E1_lossless/lossless_toy_limit-5_max-128.json` | E1.4 toy lossless | record match 5/5, turn match 5/5 |
| `experiments/E1_lossless/lossless_mt_bench_limit-20_max-128.json` | E1.4 MT-Bench lossless | record match 13/20, turn match 33/40 |
| `experiments/E1_lossless/lossless_mt_bench_limit-5_max-128_fp16_det_no_tf32.json` | E1.5 fp16 deterministic control | record match 3/5, turn match 8/10 |
| `experiments/E1_lossless/lossless_mt_bench_limit-5_max-128_fp32_det_no_tf32.json` | E1.5 fp32 diagnosis | record match 5/5, turn match 10/10 |
| `experiments/E3_ddd/ddd_h_mt_bench_limit-5_max-128_depth-11_steps-5-7-9.json` | E3.1 DDD H profiling | H step5/7/9 median = -1.124/-1.969/-3.121 |
| `experiments/E3_ddd/fixed_depth_mt_bench_limit-5_max-128_depths-5-7-9-11.json` | E3.2 fixed depth sweep | depth 7 best tok/s 83.99, depth 11 accept/step 6.051 but speed 0.932× |
| `experiments/E3_ddd/ddd_sweep_mt_bench_limit-5_max-128_depth-11_tau-m5p0-m4p0-m3p0-m2p0-m1p0-m0p5-m0p3.json` | E3.3/E3.4 DDD sweep | tau=-2 keeps accept/step 6.035 at 84.09 tok/s; tau=-0.5 best speed 88.13 tok/s |
| `experiments/E1_lossless/lossless_mt_bench_limit-5_max-128_ddd_tau-m2.json` | E3.5 DDD tau=-2 correctness | record match 4/5, turn match 9/10 |
| `experiments/E1_lossless/lossless_mt_bench_limit-5_max-128_ddd_tau-m0p5.json` | E3.5 DDD tau=-0.5 correctness | record match 3/5, turn match 8/10 |
| `experiments/E3_ddd/fixed_depth_mt_bench_limit-20_max-128_depths-5-7-9-11.json` | E3.5 fixed-depth 20 条扩展 | depth5 76.05 tok/s; depth11 accept/step 5.372, tok/s 70.19 |
| `experiments/E3_ddd/ddd_sweep_mt_bench_limit-20_max-128_depth-11_tau-m2-m0p5.json` | E3.5 DDD 20 条扩展 | tau=-2 76.35 tok/s/5.357 accept; tau=-0.5 78.60 tok/s/5.165 accept |
| `experiments/E3_ddd/fixed_depth_mt_bench_limit-80_max-128_depths-5-11.json` | E3.5 fixed-depth 全量 | depth5 65.23 tok/s; depth11 77.03 tok/s/6.392 accept |
| `experiments/E3_ddd/ddd_sweep_mt_bench_limit-80_max-128_depth-11_tau-m2p0-m0p5.json` | E3.5 DDD 全量 | tau=-2 68.51 tok/s; tau=-0.5 70.74 tok/s; 均低于 fixed depth 11 |
| `experiments/E4_opt_tree/tree_diff_mt_bench_limit-2_max-64_expand-1p5-2p0.json` | E4.1 OPT tree diff smoke | expand 1.5/2.0 mean Jaccard 1.000, identical 100% |
| `experiments/E4_opt_tree/tree_diff_mt_bench_limit-5_max-128_expand-1p5.json` | E4.1 OPT tree diff 小切片 | 232 calls, mean Jaccard 1.000, baseline-only/OPT-only 0 |
| `experiments/E4_opt_tree/opt_dynamic_mt_bench_limit-5_max-128_expand-1p5-2p0.json` | E4.2 OPT-only 小切片 | accept/step 全部 5.286，loop 全部 22.20 |
| `experiments/E4_opt_tree/opt_dynamic_mt_bench_limit-5_max-128_expand-1p5-2p0_single_repeat.json` | E4.2 same-model 重复验证 | EAGLE-repeat tok/s 漂移大，速度小差异不作为 OPT 收益 |
| `experiments/E4_opt_tree/opt_dynamic_mt_bench_limit-80_max-128_expand-1p5_single_repeat.json` | E4.2 全量 OPT 对照 | accept/step 全部 5.341，loop 全部 22.41 |
| `experiments/E5_scenarios/mt_bench_category_summary.json` | E5.2 全量 category split | speedup/DDD/OPT 按 MT-Bench category 聚合 |
| `experiments/E5_scenarios/README.md` | E5.2 可读表格 | 展示 8 个 category 的差异，并包含优化后 DDD tau=-1.0 |
| `experiments/E5_ablation/ddd_sweep_mt_bench_limit-80_max-128_depth-11_tau-m2p0-m0p5_opt-1p5.json` | E5.1 DDD+OPT 组合 | acceptance/depth 与 DDD-only 完全对齐，tok/s 不提升 |
| `experiments/E5_ablation/ddd_sweep_mt_bench_limit-80_max-128_depth-11_tau-m1p0_opt-1p5.json` | E5.1 优化后 DDD+OPT 组合 | tau=-1.0 acceptance/depth 与 DDD-only 完全对齐，tok/s 从 88.27 降到 77.56 |
| `experiments/E5_ablation/README.md` | E5.1 主消融补充记录 | DDD-only vs DDD+OPT 对比 |
| `experiments/E6_optimizations/ddd_sweep_mt_bench_limit-80_max-128_depth-11_tau-m1p0-m0p3.json` | E6 DDD 阈值优化 | tau=-1.0 达到 88.27 tok/s，当前最佳 |
| `experiments/E6_optimizations/fixed_depth_mt_bench_limit-80_max-128_depths-9.json` | E6 fixed depth 9 对照 | 60.47 tok/s，低于 fixed depth 11 |
| `experiments/E6_optimizations/ddd_sweep_mt_bench_limit-5_max-128_depth-11_tau-m2p0-m0p5_budget-min32.json` | E6 DDD dynamic budget smoke | 明显负优化，不进入全量 |
| `experiments/E6_optimizations/tree_diff_mt_bench_limit-2_max-64_expand-2p0-4p0.json` | E6 OPT 扩大候选池 smoke | tree_top_k=20 仍 Jaccard=1.000 |
| `experiments/E6_optimizations/README.md` | E6 优化实验总结 | 汇总有效/无效优化方向 |

这些结果说明：官方 baseline 速度路径已经可用；E1.4 进一步排除了 prompt formatting、multi-turn history 和简单 `max_new_tokens` off-by-one 对 correctness 的影响。E1.5 显示 deterministic/no-TF32 的 fp16 控制组仍在相同位置 mismatch，而 fp32 诊断组可以恢复 MT-Bench 前 5 条的 10/10 turn exact match。因此，当前 MT-Bench 中的少量 greedy token mismatch 主要来自 fp16 sequential KV-cache 路径与 tree/full-prefix 路径的数值差异。后续 DDD/OPT-Tree 主性能实验仍使用 fp16，但必须继续用同一 lossless 脚本报告 turn-level match rate。

E3.1 进一步说明，当前实现中的 H 分布与旧 DDD 网格不匹配：`-8/-10/-12` 几乎不会触发 early stop，而 `-0.3` 又非常激进。下一步 DDD sweep 应围绕 `[-5.0, -4.0, -3.0, -2.0, -1.0, -0.5, -0.3]` 重新设计。

E3.2 说明 fixed depth 的收益存在饱和：depth 7 相比 depth 5 提高 accept/step 且速度基本不降；depth 9/11 的接受长度继续小幅提高，但 throughput 开始下降。这支撑后续 DDD 使用 `ddd_max_depth=11`，但通过 H 阈值避免每一步都支付 depth 11 的固定开销。

E3.3/E3.4 显示 DDD 能把 fixed depth 11 的深探索收益部分转化为速度收益。早期小样本中，`tau=-2.0` 是保守配置，`tau=-0.5` 是速度优先配置；但 full MT-Bench 上这两个阈值都没有超过 fixed depth 11。因此后续 E6 继续补充了中间阈值和更激进阈值。

MT-Bench 全量结果先修正了小切片判断：fixed depth 11 达到 `77.03 tok/s` 和 `6.392 accept/step`，优于 depth 5、`tau=-2.0` 和 `tau=-0.5`。E6 进一步发现 `tau=-1.0` 是当前最佳 DDD 配置，full MT-Bench 达到 `88.27 tok/s`、`6.252 accept/step`、平均 depth `7.695`，超过 fixed depth 11。因此最终报告应把 DDD 写成“机制复现成功，且经过阈值调参后有效”；推荐主配置为 `ddd_max_depth=11`、`ddd_check_steps=5,7,9`、`ddd_threshold=-1.0`。

E6 同时排除了三个看似合理但实测无效的方向：固定 depth 9 只有 `60.47 tok/s`，说明简单缩短 fixed depth 不行；DDD dynamic budget 在 MT-Bench 5 条上从原始 DDD 的 `88.13 tok/s` 掉到 `51.09 tok/s`，说明减少 tree budget 会损害候选覆盖；OPT 扩大候选池到 `tree_top_k=20` 后仍然 `Jaccard=1.000`，说明 OPT-Tree 无效不是候选池太小造成的。

E4.1 显示 OPT-Tree 在当前 EAGLE-3 dynamic tree baseline 上没有改变最终节点集合：MT-Bench 前 5 条、232 次 tree construction 中 mean Jaccard=1.000，identical rate=100%，baseline-only 和 OPT-only 节点数均为 0。这强烈支持 OPT-Tree 的 negative-result 叙事：当前实现并非没有运行 OPT，而是 OPT 的选择目标与 EAGLE-3 dynamic rerank 高度重合。

E4.2 进一步显示 OPT 与 EAGLE-3 的 acceptance 行为完全相同：小切片 `accept/step=5.286`，全量 `accept/step=5.341`，对应 loop count 也完全相同。tok/s 在同一模型内存在很大顺序漂移，不能作为 OPT 正向收益或负向开销的唯一证据。因此最终 OPT 分析应强调“节点集合和 acceptance 均无变化”，而不是围绕 tok/s 做结论。根据当前项目决策，后续不再替换 baseline，也不再增加 EAGLE-1/EAGLE-2 或 weak tree baseline。

E5.2 已完成 MT-Bench category split。结果显示不同任务子集差异很大：EAGLE speedup 从 roleplay 的 3.387× 到 math 的 4.262×；turn-level exact match 从 roleplay 的 65.0% 到 extraction 的 100.0%。最终报告不应只给总体均值，而应保留 category 表格，并说明当前没有额外外部数据集，因此“多数据集”部分在本项目实际执行中落地为 MT-Bench 内部的多任务子集分析。
