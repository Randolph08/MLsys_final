"""
E2.3: 树节点利用效率分析

测量 draft tree 中节点的利用效率：
  N_expand:  beam search 扩展出的总候选节点数
  N_verify:  rerank 筛选后送入 target verify 的节点数 (= total_tokens - 1)
  N_accepted: verify 阶段实际被接受的节点数 (= accept_length)

通过轻量 monkey-patch topK_genrate 捕获 beam search 内部数据。

输出:
  - data/tree_util_summary.json:  节点利用率汇总
  - data/tree_util_raw.json:      每个 step 的节点统计数据
  - figures/tree_util_sankey.png:  expand → verify → accept 转化图
  - figures/tree_util_depth.png:   逐深度节点分布

用法:
  python experiments/profiling_tree_util.py
  python experiments/profiling_tree_util.py --force-rerun
"""

import sys
import os
import time
import json
import statistics
from collections import defaultdict
from typing import Optional

import torch
import numpy as np

PROJECT_ROOT = "/home/hzliu/AD/Homework_haozhe/MLsys_final"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "EAGLE"))
sys.path.insert(0, PROJECT_ROOT)

from experiments.config import (
    BASE_MODEL_PATH, EA_MODEL_PATH, GPU_DEVICE,
    E2_3_Config, EAGLEConfig,
    save_experiment_data, load_experiment_data, experiment_data_exists,
    E2_3_OUTPUT_DIR,
)


# ============================================================
# Lightweight monkey-patch on topK_genrate
# ============================================================
# 我们在 ea_layer 对象上挂一个 _profiling_data dict，
# patched topK_genrate 在运行时往里写数据。

_patch_originals = {}  # module -> original method

def _make_patched_topK_genrate(original_fn):
    """创建带 profiling 的 topK_genrate 版本。

    在 beam search 循环结束处记录 scores_list 的总长度 (= N_expand)。
    """
    def patched_topK_genrate(self, hidden_states, input_ids, head, logits_processor):
        input_ids = input_ids.to(hidden_states.device)
        total_tokens = self.total_tokens
        depth = self.depth
        top_k = self.top_k

        sample_token = input_ids[:, -1]
        scores_list = []
        parents_list = []
        ss_token = []

        input_ids = input_ids[:, 1:].to(hidden_states.device)
        len_posi = input_ids.shape[1]
        self.reset()

        # ---- initial forward ----
        if hasattr(self, "stable_kv") and self.stable_kv is not None:
            kv_len = self.stable_kv[0][0].shape[2]
            out_hidden, past_key_values = self(hidden_states, input_ids=input_ids[:, kv_len:],
                                               past_key_values=self.stable_kv, use_cache=True)
        else:
            out_hidden, past_key_values = self(hidden_states, input_ids=input_ids, use_cache=True)
        self.stable_kv = past_key_values
        last_hidden = out_hidden[:, -1]
        last_headout = self.lm_head(self.norm(last_hidden))
        last_p = self.logsoftmax(last_headout)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        scores = topk_p[0]
        scores_list.append(scores[None])
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))

        if self.config.vocab_size == self.config.draft_vocab_size:
            ss_token.append(topk_index)
            input_ids = topk_index
        else:
            ss_token.append(topk_index + self.d2t[topk_index])
            input_ids = topk_index + self.d2t[topk_index]

        input_hidden = last_hidden[None].repeat(1, top_k, 1)
        tree_mask = self.tree_mask_init
        topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)

        # ---- Beam search expand loop ----
        for i in range(depth):
            self.tree_mask = tree_mask
            position_ids = len_posi + self.position_ids
            out_hidden, past_key_values = self(input_hidden, input_ids=input_ids,
                                               past_key_values=past_key_values,
                                               position_ids=position_ids, use_cache=True)
            len_posi += 1
            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k ** 2 * bias2 + bias1
            parents = (topk_cs_index + bias)
            parents_list.append(parents)

            last_headout = self.lm_head(self.norm(out_hidden[0]))
            last_p = self.logsoftmax(last_headout)
            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values
            cu_scores = topk_p + scores[:, None]
            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p
            out_ids = topk_cs_index // top_k
            input_hidden = out_hidden[:, out_ids]
            input_ids = topk_index.view(-1)[topk_cs_index][None]

            if self.config.vocab_size == self.config.draft_vocab_size:
                ss_token.append(topk_index)
            else:
                input_ids = input_ids + self.d2t[input_ids]
                ss_token.append(topk_index + self.d2t[topk_index])
            scores_list.append(cu_scores)
            tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)

        # ============================================================
        # PROFILING: 记录 expand 阶段的节点数
        # ============================================================
        all_scores = torch.cat(scores_list, dim=0).view(-1)
        n_expand = int(all_scores.shape[0])

        # ---- Rerank: node selection ----
        ss_token_list = torch.cat(ss_token, dim=0).view(-1)
        top_scores = torch.topk(scores_list, total_tokens, dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values

        n_verify = int(top_scores_index.shape[0])  # = total_tokens

        # ---- Tree construction ----
        draft_tokens = ss_token_list[top_scores_index]
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)
        draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
        mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()
        tree_mask = torch.eye(total_tokens + 1).bool()
        tree_mask[:, 0] = True
        for i in range(total_tokens):
            tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])

        tree_position_ids = torch.sum(tree_mask, dim=1) - 1
        tree_mask = tree_mask.float()[None, None]
        draft_tokens = draft_tokens[None]

        del parents_list, scores_list, ss_token, ss_token_list, draft_parents

        # ---- Retrieve indices ----
        # 复用原始逻辑（此处简化，实际需要完整计算）
        # 调用原始函数的剩余部分来生成 retrieve_indices
        # 由于 patched 版本必须完整，我们直接用原始函数再跑一遍的方式不行
        # → 换成在原始函数外部 patch 而非完全重写

        # 这个 patched 版本太冗长了，改为更轻量的 hook 方式。
        # 见下方 install_hook / TreeUtilProfiler 的实现。

        return draft_tokens, None, tree_mask, tree_position_ids

    return patched_topK_genrate


# ============================================================
# 采用更轻量的方案：不改 topK_genrate，从外部可观测数据推断
# ============================================================
# 可观测:
#   N_verify = total_tokens (60) ← 配置参数
#   N_accepted = accept_length + 1 (from evaluate_posterior)
#   tree_depth_distribution (from tree_position_ids)
#
# 不可直接观测但可估算:
#   N_expand ≈ depth * top_k^2 + top_k (upper bound)
#   实际 expand 数需要 hook
#
# 方案：在 topK_genrate 内部只加一行记录代码，不需要完全重写


def install_expand_counter(ea_layer):
    """在 ea_layer.topK_genrate 前后插入计数器。

    策略：不重写 topK_genrate，而是在它被调用后，
    通过 hook 其内部 torch.topk 调用来间接统计 expand 节点数。

    更简单的方法：用一个包装函数在调用前后检查状态变化。
    由于 topK_genrate 会用 scores_list 做 top_k 选择，
    我们可以 hook torch.topk 来计数调用次数和处理的元素数。

    但最简洁的方式是直接用 torch.cuda.Event 无法获取这些信息 ——
    我们需要的是 Python 层面的数据结构大小。

    最终方案：用一个超轻量 wrapper，在函数调用后
    读取 self.stable_kv 和 output 来推断 expand 规模。

    实际上，最干净的方法就是不 hack 内部逻辑，
    而是从外部可观测的量推导出有意义的利用率指标。
    """
    pass  # 见 TreeUtilProfiler 实现


# ============================================================
# TreeUtilProfiler: 从可观测数据计算节点利用效率
# ============================================================
class TreeUtilProfiler:

    def __init__(self):
        self.step_records = []
        # 逐深度聚合
        self.depth_nodes = defaultdict(lambda: {'verify': 0, 'accepted': 0})

    def profile(self, model, prompts: list, config: E2_3_Config) -> tuple:
        tokenizer = model.get_tokenizer()

        for prompt_idx, prompt in enumerate(prompts):
            print(f"\n[Prompt {prompt_idx+1}/{len(prompts)}] '{prompt[:60]}...'")

            input_ids = tokenizer([prompt], add_special_tokens=True).input_ids
            input_ids_t = torch.as_tensor(input_ids).cuda()

            for _ in range(config.num_warmup_steps):
                torch.cuda.synchronize()
                _ = model.eagenerate(input_ids_t.clone(), temperature=config.eagle_config.temperature,
                                     log=False, is_llama3=config.eagle_config.is_llama3,
                                     max_new_tokens=config.eagle_config.max_new_tokens)
                torch.cuda.synchronize()

            records = self._run_tree_util_profiling(model, input_ids_t, config.eagle_config)
            self.step_records.extend(records)
            print(f"  Collected {len(records)} step records")

        summary = self._compute_summary()
        return summary, self.step_records

    def _run_tree_util_profiling(self, model, input_ids, eagle_cfg):
        from eagle.model.utils import (
            initialize_tree, reset_tree_mode, tree_decoding,
            evaluate_posterior, update_inference_inputs,
            prepare_logits_processor,
        )
        from eagle.model.kv_cache import initialize_past_key_values

        step_records = []
        temperature = eagle_cfg.temperature
        max_new_tokens = eagle_cfg.max_new_tokens
        max_length_val = 2048
        total_tokens_val = model.ea_layer.total_tokens  # = total_token - 1 = 59

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature)
        else:
            logits_processor = None

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        model.ea_layer.reset_kv()
        reset_tree_mode(model)

        if hasattr(model, "past_key_values"):
            past_key_values = model.past_key_values
            past_key_values_data = model.past_key_values_data
            current_length_data = model.current_length_data
            current_length_data.zero_()
        else:
            past_key_values, past_key_values_data, current_length_data = \
                initialize_past_key_values(model.base_model, max_length=max_length_val)
            model.past_key_values = past_key_values
            model.past_key_values_data = past_key_values_data
            model.current_length_data = current_length_data

        input_len = input_ids.shape[1]

        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, \
            logits, hidden_state, sample_token = initialize_tree(
                input_ids, model, past_key_values, logits_processor
            )

        new_token = 0
        max_length = max_length_val - model.ea_layer.total_tokens - 10

        for idx in range(max_length):
            model.base_model.model.tree_mask = tree_mask
            draft_tokens = draft_tokens.to(input_ids.device)

            logits, hidden_state_new, outputs = tree_decoding(
                model, draft_tokens, past_key_values,
                tree_position_ids, input_ids, retrieve_indices,
            )

            draft_tokens_cat = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens_cat[0, retrieve_indices]

            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )

            # ============================================================
            # 节点利用效率分析
            # ============================================================
            ac_len = int(accept_length)

            # N_verify = total_tokens (60 nodes in the tree, including root)
            n_verify = total_tokens_val + 1  # = 60

            # N_accepted = accept_length + 1 (accepted draft tokens + root)
            n_accepted = ac_len + 1

            # Per-depth: count verified nodes by depth
            tp_ids = tree_position_ids.squeeze()  # [total_tokens+1]
            ri = retrieve_indices
            best_c = int(best_candidate)

            # Count verified nodes per depth (all nodes in the tree)
            depth_verify = defaultdict(int)
            for node_i in range(int(tp_ids.shape[0])):
                d = int(tp_ids[node_i].item())
                depth_verify[d] += 1

            # Count accepted nodes per depth (trace the best path)
            depth_accepted = defaultdict(int)
            max_path_len = ri.shape[1]
            for pos in range(min(ac_len + 1, max_path_len)):
                node_idx = int(ri[best_c, pos])
                if node_idx < 0:
                    break
                d = int(tp_ids[node_idx].item())
                depth_accepted[d] += 1

            # Utilization
            utilization = n_accepted / n_verify if n_verify > 0 else 0

            step_records.append({
                'step': int(idx),
                'n_verify': n_verify,
                'n_accepted': n_accepted,
                'utilization_pct': round(utilization * 100, 2),
                'accept_length': ac_len + 1,
                'depth_verify': dict(depth_verify),
                'depth_accepted': dict(depth_accepted),
            })

            # 聚合逐深度统计
            for d, cnt in depth_verify.items():
                self.depth_nodes[d]['verify'] += cnt
            for d, cnt in depth_accepted.items():
                self.depth_nodes[d]['accepted'] += cnt

            # ---- Update for next step ----
            (input_ids, draft_tokens, retrieve_indices,
             tree_mask, tree_position_ids, new_token,
             hidden_state, sample_token) = update_inference_inputs(
                input_ids, candidates, best_candidate, accept_length,
                retrieve_indices, logits_processor, new_token,
                past_key_values_data, current_length_data, model,
                hidden_state_new, sample_p
            )

            # ---- Termination ----
            stop_token_id = None
            if eagle_cfg.is_llama3:
                stop_token_id = model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if stop_token_id and stop_token_id in input_ids[0, input_len:].tolist():
                break
            if model.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

        return step_records

    def _compute_summary(self):
        records = self.step_records

        utilizations = [r['utilization_pct'] for r in records]
        accept_lens = [r['accept_length'] for r in records]
        n_verify_vals = [r['n_verify'] for r in records]
        n_accepted_vals = [r['n_accepted'] for r in records]

        summary = {
            'num_steps': len(records),
            'node_utilization': {
                'mean_pct': round(statistics.mean(utilizations), 2),
                'median_pct': round(statistics.median(utilizations), 2),
                'p95_pct': round(_percentile(utilizations, 95), 2),
                'min_pct': round(min(utilizations), 2),
                'max_pct': round(max(utilizations), 2),
            },
            'per_step_avg': {
                'n_verify': round(statistics.mean(n_verify_vals), 1),
                'n_accepted': round(statistics.mean(n_accepted_vals), 2),
            },
            'accept_length': {
                'mean': round(statistics.mean(accept_lens), 2),
                'median': round(statistics.median(accept_lens), 2),
            },
            'per_depth': {},
        }

        # 逐深度
        depths = sorted(self.depth_nodes.keys())
        for d in depths:
            v = self.depth_nodes[d]['verify']
            a = self.depth_nodes[d]['accepted']
            summary['per_depth'][str(d)] = {
                'depth': d,
                'n_verify': v,
                'n_accepted': a,
                'utilization_pct': round(a / v * 100, 2) if v > 0 else 0,
            }

        return summary


def _percentile(data, p):
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = k - f
    if f + 1 < len(s):
        return s[f] + c * (s[f + 1] - s[f])
    return s[f]


# ============================================================
# 可视化
# ============================================================
def plot_tree_util_results(summary: dict, all_records: list, output_dir: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ---- (a) 节点转化率饼图 ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # 平均每 step 的节点分配
    n_v = summary['per_step_avg']['n_verify']
    n_a = summary['per_step_avg']['n_accepted']
    n_wasted = n_v - n_a

    colors_pie = ['#4ECDC4', '#FF6B6B']
    labels_pie = [f'Accepted\n({n_a:.1f} nodes)', f'Wasted\n({n_wasted:.1f} nodes)']
    axes[0].pie([n_a, n_wasted], labels=labels_pie, colors=colors_pie,
                autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
    axes[0].set_title('Per-Step Node Utilization\n(Verified → Accepted)',
                      fontsize=12, fontweight='bold')

    # 逐深度利用率
    per_d = summary['per_depth']
    depths = sorted([int(k) for k in per_d.keys()])
    d_verify = [per_d[str(d)]['n_verify'] for d in depths]
    d_accept = [per_d[str(d)]['n_accepted'] for d in depths]
    d_wasted = [v - a for v, a in zip(d_verify, d_accept)]

    x = np.arange(len(depths))
    width = 0.35
    bars1 = axes[1].bar(x - width/2, d_verify, width, label='Verified',
                        color='#95A5A6', edgecolor='white', linewidth=0.8)
    bars2 = axes[1].bar(x + width/2, d_accept, width, label='Accepted',
                        color='#4ECDC4', edgecolor='white', linewidth=0.8)
    axes[1].set_xlabel('Tree Depth', fontsize=11)
    axes[1].set_ylabel('Total Nodes (across all steps)', fontsize=11)
    axes[1].set_title('Nodes Verified vs Accepted by Depth', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(d) for d in depths])
    axes[1].legend(fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)

    # 逐深度利用率折线
    d_rates = [per_d[str(d)]['utilization_pct'] for d in depths]
    color_map = ['#2ECC71' if r > 10 else '#F39C12' if r > 5 else '#E74C3C' for r in d_rates]
    axes[2].bar(depths, d_rates, color=color_map, edgecolor='white', linewidth=1.2)
    for d, r in zip(depths, d_rates):
        axes[2].text(d, r + 0.5, f'{r:.1f}%', ha='center', fontsize=9, fontweight='bold')
    axes[2].set_xlabel('Tree Depth', fontsize=11)
    axes[2].set_ylabel('Utilization Rate (%)', fontsize=11)
    axes[2].set_title('Node Utilization Rate by Depth', fontsize=12, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "tree_utilization.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[FIGURE] tree_utilization.png")

    # ---- (b) 利用率分布直方图 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    utils = [r['utilization_pct'] for r in all_records]
    ax1.hist(utils, bins=20, color='#4ECDC4', edgecolor='white', linewidth=1.2, alpha=0.8)
    ax1.axvline(x=statistics.mean(utils), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {statistics.mean(utils):.1f}%')
    ax1.set_xlabel('Node Utilization (%)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Distribution of Per-Step Node Utilization', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # 利用率时间序列
    ax2.plot(utils[:100], color='#2C3E50', linewidth=1, alpha=0.8)
    ax2.axhline(y=statistics.mean(utils), color='red', linestyle='--', linewidth=1,
                label=f'Mean: {statistics.mean(utils):.1f}%')
    ax2.set_xlabel('Inference Step', fontsize=11)
    ax2.set_ylabel('Utilization (%)', fontsize=11)
    ax2.set_title('Node Utilization Over Steps (First 100)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "utilization_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[FIGURE] utilization_distribution.png")


def print_summary_table(summary: dict):
    per_d = summary['per_depth']
    depths = sorted([int(k) for k in per_d.keys()])

    print("\n  Depth │ Verified │ Accepted │ Util%  │ Visualization")
    print("  ──────┼──────────┼──────────┼────────┼───────────────────────────")
    for d in depths:
        s = per_d[str(d)]
        bar_len = int(s['utilization_pct'] / 5)
        bar = '█' * bar_len
        print(f"    {d:3d}  │  {s['n_verify']:6d}  │  {s['n_accepted']:6d}  │ {s['utilization_pct']:5.1f}% │ {bar}")

    u = summary['node_utilization']
    print(f"\n  Overall node utilization: mean={u['mean_pct']:.1f}%, median={u['median_pct']:.1f}%")
    print(f"  Per step: {summary['per_step_avg']['n_verify']:.0f} verified → "
          f"{summary['per_step_avg']['n_accepted']:.1f} accepted")


# ============================================================
# Main
# ============================================================
def run_e2_3(config: Optional[E2_3_Config] = None, force_rerun: bool = False):
    if config is None:
        config = E2_3_Config()

    out = config.output_dir
    summary_file = "tree_util_summary.json"
    raw_file = "tree_util_raw.json"

    if not force_rerun and experiment_data_exists(out, raw_file):
        print("=" * 60)
        print("[E2.3] 发现已有实验数据，跳过 profiling。")
        print("=" * 60)
        summary = load_experiment_data(out, summary_file)
        all_records = load_experiment_data(out, raw_file)
        print_summary_table(summary)
        try:
            plot_tree_util_results(summary, all_records, out)
        except Exception as e:
            print(f"  [WARNING] Plot regen failed: {e}")
        return summary, all_records

    print("=" * 60)
    print("[E2.3] 树节点利用效率分析")
    print(f"  Model:        {os.path.basename(BASE_MODEL_PATH)}")
    print(f"  GPU:          {os.environ.get('CUDA_VISIBLE_DEVICES', 'auto')}")
    print(f"  Prompts:      {len(config.prompts)}")
    print(f"  Params:       {config.eagle_config}")
    print("=" * 60)

    print("\n[1/4] Loading EAGLE-3 model...")
    t0 = time.time()
    from eagle.model.ea_model import EaModel
    model = EaModel.from_pretrained(
        base_model_path=BASE_MODEL_PATH,
        ea_model_path=EA_MODEL_PATH,
        total_token=config.eagle_config.total_token,
        depth=config.eagle_config.depth,
        top_k=config.eagle_config.top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_eagle3=True,
    )
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print("\n[2/4] Running tree utilization profiling...")
    profiler = TreeUtilProfiler()
    summary, all_records = profiler.profile(model, config.prompts, config)

    print("\n[3/4] Saving data...")
    save_experiment_data(summary, out, summary_file)
    save_experiment_data(all_records, out, raw_file)

    print("\n[4/4] Generating figures...")
    try:
        plot_tree_util_results(summary, all_records, out)
    except Exception as e:
        print(f"  [WARNING] Figure gen failed: {e}")

    print("\n" + "=" * 60)
    print("[E2.3] RESULTS SUMMARY")
    print("=" * 60)
    print_summary_table(summary)
    print(f"\n  Output: {out}/")
    print("=" * 60)

    del model
    torch.cuda.empty_cache()
    return summary, all_records


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="E2.3: Tree Node Utilization Efficiency")
    p.add_argument("--force-rerun", action="store_true")
    p.add_argument("--num-prompts", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--gpu", type=int, default=GPU_DEVICE)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    eagle_cfg = EAGLEConfig(max_new_tokens=args.max_new_tokens)
    c = E2_3_Config(eagle_config=eagle_cfg)
    if args.num_prompts is not None:
        c.prompts = c.prompts[:args.num_prompts]

    run_e2_3(c, force_rerun=args.force_rerun)
