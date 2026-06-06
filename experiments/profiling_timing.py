"""
E2.1: EAGLE 管线时延分解实验

将每个 inference step 分解为 3 个阶段，测量各自的 GPU 时延：
  ① Drafter Tree Construction: topK_genrate (expand + rerank)
  ② Target Verify:             tree_decoding (32-layer tree forward)
  ③ Rejection + KV Update:     evaluate_posterior + update_inference_inputs

注：阶段①内部的 expand 和 rerank 在当前实现中合并测量。
    如需细分，可在 cnets.py 的 topK_genrate 中插入 CUDA event 分界点。

输出:
  - data/timing_summary.json:  各阶段均值/中位数/占比
  - data/timing_raw.json:      每个 step 的原始时延 (可选)
  - figures/timing_breakdown.png:  饼图 + 柱状图
  - figures/timing_stacked.png:    堆叠柱状图
  - figures/timing_timeseries.png: 时延 + 接受长度时间序列

用法:
  python experiments/profiling_timing.py
  python experiments/profiling_timing.py --force-rerun
  python experiments/profiling_timing.py --num-prompts 1 --num-profile-steps 40
"""

import sys
import os
import time
import json
import statistics
from typing import Optional

import torch
import numpy as np

PROJECT_ROOT = "/home/hzliu/AD/Homework_haozhe/MLsys_final"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "EAGLE"))
sys.path.insert(0, PROJECT_ROOT)

from experiments.config import (
    BASE_MODEL_PATH, EA_MODEL_PATH, GPU_DEVICE,
    E2_1_Config, EAGLEConfig,
    save_experiment_data, load_experiment_data, experiment_data_exists,
    E2_1_OUTPUT_DIR,
)


# ============================================================
# PhaseProfiler: 在 eagenerate 循环中插入 GPU 计时
# ============================================================
class PhaseProfiler:

    def __init__(self):
        self.raw_timings = []

    def profile(self, model, prompts: list, config: E2_1_Config) -> tuple:
        """
        运行 profiling。对每个 prompt 运行 warmup + profiling steps。
        返回 (summary_dict, raw_timings_list)
        """
        tokenizer = model.get_tokenizer()
        all_step_timings = []

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
            step_timings = self._run_profiled_generate(model, input_ids_t, config.eagle_config)
            all_step_timings.extend(step_timings)

            print(f"  Collected {len(step_timings)} step timings")

        summary = self._compute_summary(all_step_timings)
        return summary, all_step_timings

    def _run_profiled_generate(self, model, input_ids: torch.Tensor,
                               eagle_cfg: EAGLEConfig) -> list:
        """
        手动展开 eagenerate 主循环，在 3 个关键 phase 前后插入 torch.cuda.Event。

        计时点布局:
          [Prefill + initial tree: NOT timed]

          for each step:
            ╔══════════════════════════════════════╗
            ║ Phase ③: Target Verify              ║  tree_decoding()
            ╚══════════════════════════════════════╝
            ╔══════════════════════════════════════╗
            ║ Phase ④: Rejection + KV Update      ║  evaluate_posterior()
            ║           (rejection sampling part)  ║  + part of
            ╚══════════════════════════════════════╝  update_inference_inputs
            ╔══════════════════════════════════════╗
            ║ Phase ①: Drafter Construction       ║  topK_genrate() inside
            ║           (expand + rerank)          ║  update_inference_inputs
            ╚══════════════════════════════════════╝
        """
        from eagle.model.utils import (
            initialize_tree, reset_tree_mode, tree_decoding,
            evaluate_posterior, update_inference_inputs,
            prepare_logits_processor,
        )
        from eagle.model.kv_cache import initialize_past_key_values

        step_timings = []
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

        # Initialize KV cache
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

        # ---- Prefill + First Draft Tree (not timed) ----
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

            # ============================================
            # Phase ③: Target Verify
            # ============================================
            ev_verify_s = torch.cuda.Event(enable_timing=True)
            ev_verify_e = torch.cuda.Event(enable_timing=True)
            ev_verify_s.record()

            logits, hidden_state_new, outputs = tree_decoding(
                model, draft_tokens, past_key_values,
                tree_position_ids, input_ids, retrieve_indices,
            )

            ev_verify_e.record()

            # ============================================
            # Phase ④: Rejection Sampling
            # ============================================
            draft_tokens_cat = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens_cat[0, retrieve_indices]

            ev_reject_s = torch.cuda.Event(enable_timing=True)
            ev_reject_e = torch.cuda.Event(enable_timing=True)
            ev_reject_s.record()

            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )

            ev_reject_e.record()

            # ============================================
            # Phase ①: Drafter Tree Construction
            # ============================================
            ev_draft_s = torch.cuda.Event(enable_timing=True)
            ev_draft_e = torch.cuda.Event(enable_timing=True)
            ev_draft_s.record()

            (input_ids, draft_tokens, retrieve_indices,
             tree_mask, tree_position_ids, new_token,
             hidden_state, sample_token) = update_inference_inputs(
                input_ids, candidates, best_candidate, accept_length,
                retrieve_indices, logits_processor, new_token,
                past_key_values_data, current_length_data, model,
                hidden_state_new, sample_p
            )

            ev_draft_e.record()

            # ---- Read GPU timings ----
            torch.cuda.synchronize()

            phase_verify_ms = ev_verify_s.elapsed_time(ev_verify_e)
            phase_reject_ms = ev_reject_s.elapsed_time(ev_reject_e)
            phase_draft_ms = ev_draft_s.elapsed_time(ev_draft_e)
            total_ms = phase_draft_ms + phase_verify_ms + phase_reject_ms

            step_timings.append({
                'step': int(idx),
                'phase_draft_ms': round(phase_draft_ms, 3),       # ①
                'phase_verify_ms': round(phase_verify_ms, 3),     # ③
                'phase_reject_kv_ms': round(phase_reject_ms, 3),  # ④
                'total_ms': round(total_ms, 3),
                'accept_length': int(accept_length) + 1,
            })

            # ---- Termination checks ----
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

        return step_timings

    # ---- 统计分析 ----
    def _compute_summary(self, all_timings: list) -> dict:
        phases = ['phase_draft_ms', 'phase_verify_ms', 'phase_reject_kv_ms']
        phase_labels = {
            'phase_draft_ms': '① Drafter Construction (Expand + Rerank)',
            'phase_verify_ms': '② Target Verify (32-layer tree forward)',
            'phase_reject_kv_ms': '③ Rejection Sampling + KV Update',
        }

        summary = {'num_steps': len(all_timings), 'phases': {}}

        for phase in phases:
            values = [t[phase] for t in all_timings]
            if not values:
                continue

            totals = [t['total_ms'] for t in all_timings]
            mean_v = statistics.mean(values)
            p95_v = _percentile(values, 95)
            std_v = statistics.stdev(values) if len(values) > 1 else 0.0
            ratios = [v / t for v, t in zip(values, totals) if t > 0]
            mean_ratio = statistics.mean(ratios) * 100 if ratios else 0

            summary['phases'][phase] = {
                'label': phase_labels[phase],
                'mean_ms': round(mean_v, 3),
                'median_ms': round(statistics.median(values), 3),
                'p95_ms': round(p95_v, 3),
                'std_ms': round(std_v, 3),
                'mean_ratio_pct': round(mean_ratio, 1),
            }

        totals = [t['total_ms'] for t in all_timings]
        accepts = [t['accept_length'] for t in all_timings]
        summary['overall'] = {
            'mean_total_ms': round(statistics.mean(totals), 3),
            'median_total_ms': round(statistics.median(totals), 3),
            'mean_accept_length': round(statistics.mean(accepts), 2),
        }

        return summary


def _percentile(data: list, p: float) -> float:
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
def plot_timing_results(summary: dict, all_timings: list, output_dir: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    phases = ['phase_draft_ms', 'phase_verify_ms', 'phase_reject_kv_ms']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    labels = ['Drafter\nConstruction', 'Target Verify\n(32L forward)', 'Rejection\n+ KV Update']

    # ---- (a) 饼图 + 柱状图 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    means = [summary['phases'][p]['mean_ms'] for p in phases]
    ratios = [summary['phases'][p]['mean_ratio_pct'] for p in phases]
    stds = [summary['phases'][p]['std_ms'] for p in phases]

    wedges, _, autotexts = ax1.pie(
        means, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 10}
    )
    for at in autotexts:
        at.set_fontweight('bold')
    ax1.set_title('EAGLE Step Timing Breakdown\n(Mean per Phase)', fontsize=13, fontweight='bold')

    bars = ax2.bar(range(3), means, color=colors, edgecolor='white', linewidth=1.2)
    ax2.errorbar(range(3), means, yerr=stds, fmt='none', ecolor='#333333',
                 capsize=8, capthick=1.5, linewidth=1.5)
    for i, (bar, m, r) in enumerate(zip(bars, means, ratios)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i] + 0.5,
                 f'{m:.1f} ms\n({r:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(['Drafter\nConstruction', 'Target\nVerify', 'Rejection\n+ KV Update'], fontsize=10)
    ax2.set_ylabel('Time (ms)', fontsize=12)
    ax2.set_title('Phase Timing (Mean ± Std)', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "timing_breakdown.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[FIGURE] timing_breakdown.png")

    # ---- (b) 堆叠柱状图 (前 30 步) ----
    n = min(30, len(all_timings))
    display = all_timings[:n]
    fig, ax = plt.subplots(figsize=(14, 5))
    x = range(n)
    bottom = np.zeros(n)
    for phase, c, lbl in zip(phases, colors, labels):
        vals = [t[phase] for t in display]
        ax.bar(x, vals, bottom=bottom, color=c, label=lbl.replace('\n', ' '),
               edgecolor='white', linewidth=0.5)
        bottom += np.array(vals)
    ax.set_xlabel('Inference Step', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title(f'Per-Step Timing Breakdown (First {n} Steps)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "timing_stacked.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[FIGURE] timing_stacked.png")

    # ---- (c) 时延 + 接受长度时间序列 ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax1.plot([t['total_ms'] for t in all_timings], color='#333333', linewidth=1, alpha=0.8)
    ax1.set_ylabel('Total Time (ms)', fontsize=11)
    ax1.set_title('Per-Step Total Latency & Acceptance Length', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3)

    accepts = [t['accept_length'] for t in all_timings]
    ax2.fill_between(range(len(accepts)), accepts, alpha=0.3, color='#4ECDC4')
    ax2.plot(accepts, color='#4ECDC4', linewidth=1.5)
    ax2.set_xlabel('Inference Step', fontsize=11)
    ax2.set_ylabel('Accept Length', fontsize=11)
    ax2.grid(alpha=0.3)
    mean_acc = statistics.mean(accepts)
    ax2.axhline(y=mean_acc, color='red', linestyle='--', linewidth=1,
                label=f'Mean: {mean_acc:.1f}')
    ax2.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "timing_timeseries.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[FIGURE] timing_timeseries.png")


# ============================================================
# Main
# ============================================================
def run_e2_1(config: Optional[E2_1_Config] = None, force_rerun: bool = False):
    if config is None:
        config = E2_1_Config()

    out = config.output_dir
    summary_file = "timing_summary.json"
    raw_file = "timing_raw.json"

    # 检查缓存
    if not force_rerun and experiment_data_exists(out, raw_file):
        print("=" * 60)
        print("[E2.1] 发现已有实验数据，跳过 profiling。")
        print(f"      设置 force_rerun=True 强制重新运行")
        print("=" * 60)
        summary = load_experiment_data(out, summary_file)
        all_timings = load_experiment_data(out, raw_file)
        # 仍然生成图表（可能是新增的）
        try:
            plot_timing_results(summary, all_timings, out)
        except Exception:
            pass
        return summary, all_timings

    print("=" * 60)
    print("[E2.1] EAGLE 管线时延分解实验")
    print(f"  Model:        {os.path.basename(BASE_MODEL_PATH)}")
    print(f"  Checkpoint:   {os.path.basename(EA_MODEL_PATH)}")
    print(f"  GPU:          {os.environ.get('CUDA_VISIBLE_DEVICES', 'auto')}")
    print(f"  Prompts:      {len(config.prompts)}")
    print(f"  Profile steps per prompt: {config.num_profile_steps}")
    print(f"  Params:       {config.eagle_config}")
    print("=" * 60)

    # [1] Load model
    print("\n[1/4] Loading EAGLE-3 model...")
    t0 = time.time()
    model = __import__('eagle.model.ea_model', fromlist=['EaModel']).EaModel.from_pretrained(
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
    print("\n[2/4] Running profiling...")
    profiler = PhaseProfiler()
    summary, all_timings = profiler.profile(model, config.prompts, config)

    # [3] Save
    print("\n[3/4] Saving data...")
    save_experiment_data(summary, out, summary_file)
    save_experiment_data(all_timings, out, raw_file)

    # [4] Plot
    print("\n[4/4] Generating figures...")
    try:
        plot_timing_results(summary, all_timings, out)
    except Exception as e:
        print(f"  [WARNING] Figure generation failed: {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("[E2.1] RESULTS SUMMARY")
    print("=" * 60)
    for pk, pi in summary['phases'].items():
        print(f"  {pi['label']}:")
        print(f"    {pi['mean_ms']:.2f} ms  ({pi['mean_ratio_pct']:.1f}%)  [P95: {pi['p95_ms']:.2f} ms]")
    print(f"  Overall: {summary['overall']['mean_total_ms']:.2f} ms/step, "
          f"accept_len={summary['overall']['mean_accept_length']:.2f}")
    print(f"  Output: {out}/")
    print("=" * 60)

    del model
    torch.cuda.empty_cache()
    return summary, all_timings


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="E2.1: EAGLE Pipeline Timing Breakdown")
    p.add_argument("--force-rerun", action="store_true")
    p.add_argument("--num-prompts", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--num-profile-steps", type=int, default=80)
    p.add_argument("--gpu", type=int, default=GPU_DEVICE)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    eagle_cfg = EAGLEConfig(max_new_tokens=args.max_new_tokens)
    c = E2_1_Config(
        eagle_config=eagle_cfg,
        num_profile_steps=args.num_profile_steps,
    )
    if args.num_prompts is not None:
        c.prompts = c.prompts[:args.num_prompts]

    run_e2_1(c, force_rerun=args.force_rerun)
