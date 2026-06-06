"""
E2.2: 逐深度 Token 接受率分析

对每个 inference step，记录 draft tree 中各深度位置的 token 被 target 接受的概率。
这是 DDD（动态深度解码）改进的核心证据 —— 如果深层接受率骤降，
说明固定深度在浪费 drafting 算力。

方法：
  - 每次 tree verify 后，通过 tree_position_ids 定位被接受/被拒绝 token 的深度
  - 按深度位置聚合 → 计算 acceptance_rate[depth] = accepted / tested

输出:
  - data/acceptance_summary.json:  逐深度接受率 + 总体统计
  - data/acceptance_raw.json:     每个 step 的逐深度记录 (可选)
  - figures/acceptance_rate.png:  接受率 vs 深度折线图
  - figures/acceptance_cdf.png:   接受长度 CDF 图

用法:
  python experiments/profiling_acceptance.py
  python experiments/profiling_acceptance.py --force-rerun
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
    E2_2_Config, EAGLEConfig,
    save_experiment_data, load_experiment_data, experiment_data_exists,
    E2_2_OUTPUT_DIR,
)


# ============================================================
# AcceptanceProfiler: 跟踪每步的接受/拒绝深度
# ============================================================
class AcceptanceProfiler:

    def __init__(self):
        # 逐深度聚合: depth -> {'accepted': N, 'tested': N}
        self.depth_stats = defaultdict(lambda: {'accepted': 0, 'tested': 0})
        # 每个 step 的详细记录
        self.step_records = []

    def profile(self, model, prompts: list, config: E2_2_Config) -> tuple:
        """运行 profiling。返回 (summary_dict, step_records_list)"""
        tokenizer = model.get_tokenizer()

        for prompt_idx, prompt in enumerate(prompts):
            print(f"\n[Prompt {prompt_idx+1}/{len(prompts)}] '{prompt[:60]}...'")

            input_ids = tokenizer([prompt], add_special_tokens=True).input_ids
            input_ids_t = torch.as_tensor(input_ids).cuda()

            # Warmup
            for _ in range(config.num_warmup_steps):
                torch.cuda.synchronize()
                _ = model.eagenerate(
                    input_ids_t.clone(),
                    temperature=config.eagle_config.temperature,
                    log=False,
                    is_llama3=config.eagle_config.is_llama3,
                    max_new_tokens=config.eagle_config.max_new_tokens,
                )
                torch.cuda.synchronize()

            # Profiling run
            records = self._run_acceptance_profiling(model, input_ids_t, config.eagle_config)
            self.step_records.extend(records)

            print(f"  Collected {len(records)} step records")

        summary = self._compute_summary()
        return summary, self.step_records

    def _run_acceptance_profiling(self, model, input_ids: torch.Tensor,
                                   eagle_cfg: EAGLEConfig) -> list:
        """
        手动展开 eagenerate 主循环，在每次 verify 后记录被接受 token 的树深度。
        """
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

        # ---- Prefill + First Draft Tree ----
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, \
            logits, hidden_state, sample_token = initialize_tree(
                input_ids, model, past_key_values, logits_processor
            )

        new_token = 0
        max_length = max_length_val - model.ea_layer.total_tokens - 10

        # ---- Main Loop ----
        for idx in range(max_length):
            model.base_model.model.tree_mask = tree_mask
            draft_tokens = draft_tokens.to(input_ids.device)

            # ---- Target Verify ----
            logits, hidden_state_new, outputs = tree_decoding(
                model, draft_tokens, past_key_values,
                tree_position_ids, input_ids, retrieve_indices,
            )

            # ---- Rejection Sampling ----
            draft_tokens_cat = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens_cat[0, retrieve_indices]

            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )

            # ============================================================
            # 记录接受/拒绝的深度信息
            #   使用当前步的 tree_position_ids 和 retrieve_indices
            #   （与 evaluate_posterior 使用的是同一份，确保索引一致）
            # ============================================================
            tp_ids = tree_position_ids.squeeze()  # [total_tokens+1] 每节点的深度
            ri = retrieve_indices                   # [num_paths, max_path_len]

            ac_len = int(accept_length)
            best_c = int(best_candidate)

            # 遍历 best_candidate 路径上的每个位置
            # 位置 0 是 sample_token (root)，从位置 1 开始是 draft token
            max_path_len = ri.shape[1]

            for pos in range(max_path_len):
                node_idx = int(ri[best_c, pos])
                if node_idx < 0:
                    # padding position
                    break

                depth = int(tp_ids[node_idx].item())

                if pos <= ac_len:
                    # Accepted
                    self.depth_stats[depth]['accepted'] += 1
                    self.depth_stats[depth]['tested'] += 1
                elif pos == ac_len + 1:
                    # First rejected position
                    self.depth_stats[depth]['tested'] += 1
                    # rejected (implicitly: tested but not accepted)
                    break
                else:
                    # Beyond the rejection point — not tested
                    break

            # Record per-step info
            accepted_depths = []
            for pos in range(min(ac_len + 1, max_path_len)):
                node_idx = int(ri[best_c, pos])
                if node_idx < 0:
                    break
                accepted_depths.append(int(tp_ids[node_idx].item()))

            rejected_depth = None
            if ac_len + 1 < max_path_len:
                rej_node_idx = int(ri[best_c, ac_len + 1])
                if rej_node_idx >= 0:
                    rejected_depth = int(tp_ids[rej_node_idx].item())

            step_records.append({
                'step': int(idx),
                'accept_length': ac_len + 1,  # +1 counts the root token
                'accepted_depths': accepted_depths,
                'rejected_depth': rejected_depth,
            })

            # ---- Update tree for next step ----
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

    # ---- 汇总统计 ----
    def _compute_summary(self) -> dict:
        depths = sorted(self.depth_stats.keys())

        per_depth = {}
        for d in depths:
            s = self.depth_stats[d]
            rate = s['accepted'] / s['tested'] if s['tested'] > 0 else 0.0
            per_depth[str(d)] = {
                'depth': d,
                'accepted': s['accepted'],
                'tested': s['tested'],
                'acceptance_rate': round(rate, 4),
            }

        # 总体统计
        total_accepted = sum(s['accepted'] for s in self.depth_stats.values())
        total_tested = sum(s['tested'] for s in self.depth_stats.values())
        accept_lengths = [r['accept_length'] for r in self.step_records]
        num_steps_with_data = len([r for r in self.step_records if r['accept_length'] > 0])

        summary = {
            'num_steps': len(self.step_records),
            'num_steps_with_accepts': num_steps_with_data,
            'total_tokens_accepted': total_accepted,
            'total_tokens_tested': total_tested,
            'overall_acceptance_rate': round(total_accepted / total_tested, 4) if total_tested > 0 else 0,
            'mean_accept_length': round(statistics.mean(accept_lengths), 2) if accept_lengths else 0,
            'median_accept_length': round(statistics.median(accept_lengths), 2) if accept_lengths else 0,
            'per_depth': per_depth,
        }

        return summary


# ============================================================
# 可视化
# ============================================================
def plot_acceptance_results(summary: dict, all_records: list, output_dir: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    per_depth = summary['per_depth']
    depths = sorted([int(k) for k in per_depth.keys()])
    rates = [per_depth[str(d)]['acceptance_rate'] * 100 for d in depths]
    tested_counts = [per_depth[str(d)]['tested'] for d in depths]

    # ---- (a) 接受率 vs 深度折线图 + 柱状图 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # 左：接受率折线图
    color_map = []
    for d in depths:
        if d <= 1:
            color_map.append('#2ECC71')      # 浅层高接受 → 绿色
        elif d <= 3:
            color_map.append('#F39C12')      # 中层中接受 → 橙色
        else:
            color_map.append('#E74C3C')      # 深层低接受 → 红色

    bars = ax1.bar(depths, rates, color=color_map, edgecolor='white', linewidth=1.2)
    ax1.plot(depths, rates, 'o-', color='#2C3E50', linewidth=2, markersize=8, zorder=5)

    # 标注数值
    for d, r in zip(depths, rates):
        ax1.text(d, r + 1.5, f'{r:.1f}%', ha='center', fontsize=10, fontweight='bold')

    ax1.set_xlabel('Depth in Draft Tree', fontsize=12)
    ax1.set_ylabel('Acceptance Rate (%)', fontsize=12)
    ax1.set_title('Token Acceptance Rate by Tree Depth\n(Lower = DDD early-stop opportunity)',
                  fontsize=13, fontweight='bold')
    ax1.set_xticks(depths)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, max(rates) * 1.2)

    # 右：每个深度的 tested 数量
    ax2.bar(depths, tested_counts, color=color_map, edgecolor='white', linewidth=1.2)
    for d, c in zip(depths, tested_counts):
        ax2.text(d, c + max(tested_counts) * 0.02, str(c), ha='center', fontsize=9)
    ax2.set_xlabel('Depth in Draft Tree', fontsize=12)
    ax2.set_ylabel('Number of Tokens Tested', fontsize=12)
    ax2.set_title('Sample Size per Depth', fontsize=13, fontweight='bold')
    ax2.set_xticks(depths)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "acceptance_rate.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[FIGURE] acceptance_rate.png")

    # ---- (b) 接受长度分布 (CDF) ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    accept_lengths = [r['accept_length'] for r in all_records if r['accept_length'] > 0]

    # 直方图
    max_al = max(accept_lengths) if accept_lengths else 10
    bins = range(1, int(max_al) + 3)
    ax1.hist(accept_lengths, bins=bins, color='#4ECDC4', edgecolor='white',
             linewidth=1.2, alpha=0.8, density=True)
    ax1.axvline(x=statistics.mean(accept_lengths), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {statistics.mean(accept_lengths):.2f}')
    ax1.axvline(x=statistics.median(accept_lengths), color='orange', linestyle='--',
                linewidth=2, label=f'Median: {statistics.median(accept_lengths):.2f}')
    ax1.set_xlabel('Accept Length (tokens per step)', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('Accept Length Distribution', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # CDF
    sorted_al = sorted(accept_lengths)
    y_cdf = np.arange(1, len(sorted_al) + 1) / len(sorted_al)
    ax2.step(sorted_al, y_cdf, where='post', color='#2C3E50', linewidth=2)
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(y=0.9, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Accept Length (tokens per step)', fontsize=11)
    ax2.set_ylabel('CDF', fontsize=11)
    ax2.set_title('Accept Length CDF', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "acceptance_cdf.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[FIGURE] acceptance_cdf.png")


# ============================================================
# 命令行报告打印
# ============================================================
def print_summary_table(summary: dict):
    """打印逐深度接受率表格"""
    per_depth = summary['per_depth']
    depths = sorted([int(k) for k in per_depth.keys()])

    print("\n  Depth │ Accepted │ Tested │ Rate   │ Visualization")
    print("  ──────┼──────────┼────────┼────────┼─────────────────────────")
    for d in depths:
        s = per_depth[str(d)]
        bar_len = int(s['acceptance_rate'] * 40)
        bar = '█' * bar_len + '░' * (40 - bar_len)
        print(f"    {d:3d}  │  {s['accepted']:6d}  │ {s['tested']:5d}  │ {s['acceptance_rate']*100:5.1f}% │ {bar}")

    print(f"\n  Overall: {summary['num_steps']} steps, "
          f"mean accept_len = {summary['mean_accept_length']}, "
          f"overall rate = {summary['overall_acceptance_rate']*100:.1f}%")


# ============================================================
# Main
# ============================================================
def run_e2_2(config: Optional[E2_2_Config] = None, force_rerun: bool = False):
    if config is None:
        config = E2_2_Config()

    out = config.output_dir
    summary_file = "acceptance_summary.json"
    raw_file = "acceptance_raw.json"

    # 检查缓存
    if not force_rerun and experiment_data_exists(out, raw_file):
        print("=" * 60)
        print("[E2.2] 发现已有实验数据，跳过 profiling。")
        print("      设置 force_rerun=True 强制重新运行")
        print("=" * 60)
        summary = load_experiment_data(out, summary_file)
        all_records = load_experiment_data(out, raw_file)
        print_summary_table(summary)
        try:
            plot_acceptance_results(summary, all_records, out)
        except Exception as e:
            print(f"  [WARNING] Plot regeneration failed: {e}")
        return summary, all_records

    print("=" * 60)
    print("[E2.2] 逐深度 Token 接受率分析")
    print(f"  Model:        {os.path.basename(BASE_MODEL_PATH)}")
    print(f"  Checkpoint:   {os.path.basename(EA_MODEL_PATH)}")
    print(f"  GPU:          {os.environ.get('CUDA_VISIBLE_DEVICES', 'auto')}")
    print(f"  Prompts:      {len(config.prompts)}")
    print(f"  Params:       {config.eagle_config}")
    print("=" * 60)

    # [1] Load model
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

    # [2] Profile
    print("\n[2/4] Running acceptance profiling...")
    profiler = AcceptanceProfiler()
    summary, all_records = profiler.profile(model, config.prompts, config)

    # [3] Save
    print("\n[3/4] Saving data...")
    save_experiment_data(summary, out, summary_file)
    save_experiment_data(all_records, out, raw_file)

    # [4] Plot
    print("\n[4/4] Generating figures...")
    try:
        plot_acceptance_results(summary, all_records, out)
    except Exception as e:
        print(f"  [WARNING] Figure generation failed: {e}")

    # Print
    print("\n" + "=" * 60)
    print("[E2.2] RESULTS SUMMARY")
    print("=" * 60)
    print_summary_table(summary)
    print(f"\n  Output: {out}/")
    print("=" * 60)

    del model
    torch.cuda.empty_cache()
    return summary, all_records


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="E2.2: Per-Depth Acceptance Rate Analysis")
    p.add_argument("--force-rerun", action="store_true")
    p.add_argument("--num-prompts", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--gpu", type=int, default=GPU_DEVICE)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    eagle_cfg = EAGLEConfig(max_new_tokens=args.max_new_tokens)
    from experiments.config import E2_2_Config as E2_2Cfg
    c = E2_2Cfg(eagle_config=eagle_cfg)
    if args.num_prompts is not None:
        c.prompts = c.prompts[:args.num_prompts]

    run_e2_2(c, force_rerun=args.force_rerun)
