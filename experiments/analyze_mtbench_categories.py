#!/usr/bin/env python3
"""Aggregate full MT-Bench runs by category for E5.2."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "E5_scenarios"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def round6(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 6)


def safe_div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return num / den


def category_order_from_records(records: list[dict[str, Any]]) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()
    for record in records:
        category = record.get("category", "unknown")
        if category not in seen:
            seen.add(category)
            order.append(category)
    return order


def empty_turn_bucket() -> dict[str, Any]:
    return {
        "records": set(),
        "turns": 0,
        "trimmed_tokens": 0,
        "raw_tokens": 0,
        "wall_time_s": 0.0,
        "loop_count": 0,
        "input_tokens": 0,
        "raw_overshoot_turns": 0,
    }


def add_turn(bucket: dict[str, Any], record: dict[str, Any], turn: dict[str, Any]) -> None:
    bucket["records"].add(record.get("question_id"))
    bucket["turns"] += 1
    bucket["trimmed_tokens"] += int(turn.get("trimmed_tokens", 0))
    bucket["raw_tokens"] += int(turn.get("raw_tokens", 0))
    bucket["wall_time_s"] += float(turn.get("wall_time_s", 0.0))
    bucket["loop_count"] += int(turn.get("loop_count", 0))
    bucket["input_tokens"] += int(turn.get("input_tokens", 0))
    if turn.get("raw_exceeds_max_new_tokens"):
        bucket["raw_overshoot_turns"] += 1


def finalize_turn_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    records = len(bucket["records"])
    turns = bucket["turns"]
    tokens = bucket["trimmed_tokens"]
    wall = bucket["wall_time_s"]
    loops = bucket["loop_count"]
    return {
        "records": records,
        "turns": turns,
        "trimmed_tokens": tokens,
        "raw_tokens": bucket["raw_tokens"],
        "wall_time_s": round6(wall),
        "tok_per_s": round6(safe_div(tokens, wall)),
        "accept_per_step": round6(safe_div(tokens, loops)),
        "mean_loop_count": round6(safe_div(loops, turns)),
        "mean_input_tokens": round6(safe_div(bucket["input_tokens"], turns)),
        "raw_overshoot_turns": bucket["raw_overshoot_turns"],
    }


def summarize_method_records(records: list[dict[str, Any]], method: str) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = defaultdict(empty_turn_bucket)
    for record in records:
        category = record.get("category", "unknown")
        method_data = record["methods"][method]
        for turn in method_data["turns"]:
            add_turn(buckets[category], record, turn)
    return {category: finalize_turn_bucket(bucket) for category, bucket in buckets.items()}


def summarize_plain_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = defaultdict(empty_turn_bucket)
    for record in records:
        category = record.get("category", "unknown")
        for turn in record["turns"]:
            add_turn(buckets[category], record, turn)
    return {category: finalize_turn_bucket(bucket) for category, bucket in buckets.items()}


def summarize_match_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "records": 0,
            "record_matches": 0,
            "turns": 0,
            "turn_matches": 0,
        }
    )
    for record in records:
        category = record.get("category", "unknown")
        bucket = buckets[category]
        bucket["records"] += 1
        bucket["record_matches"] += int(bool(record.get("all_turns_match")))
        turn_matches = record.get("turn_matches", [])
        bucket["turns"] += len(turn_matches)
        bucket["turn_matches"] += sum(1 for matched in turn_matches if matched)

    finalized: dict[str, Any] = {}
    for category, bucket in buckets.items():
        finalized[category] = {
            **bucket,
            "record_match_rate": round6(safe_div(bucket["record_matches"], bucket["records"])),
            "turn_match_rate": round6(safe_div(bucket["turn_matches"], bucket["turns"])),
        }
    return finalized


def summarize_ddd_calls(records: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "ddd_calls": 0,
            "early_stops": 0,
            "actual_depth_sum": 0.0,
            "early_stop_step_hist": defaultdict(int),
        }
    )
    for record in records:
        category = record.get("category", "unknown")
        for call in record.get("ddd_call_records", []):
            bucket = buckets[category]
            bucket["ddd_calls"] += 1
            bucket["actual_depth_sum"] += float(call.get("actual_depth", 0.0))
            if call.get("early_stopped"):
                bucket["early_stops"] += 1
                step = call.get("early_stop_step")
                if step is not None:
                    bucket["early_stop_step_hist"][str(step)] += 1

    finalized: dict[str, Any] = {}
    for category, bucket in buckets.items():
        calls = bucket["ddd_calls"]
        finalized[category] = {
            "ddd_calls": calls,
            "early_stops": bucket["early_stops"],
            "early_stop_rate": round6(safe_div(bucket["early_stops"], calls)),
            "mean_actual_depth": round6(safe_div(bucket["actual_depth_sum"], calls)),
            "early_stop_step_hist": dict(sorted(bucket["early_stop_step_hist"].items())),
        }
    return finalized


def merge_ddd_turns_and_calls(records: list[dict[str, Any]]) -> dict[str, Any]:
    turns = summarize_plain_records(records)
    calls = summarize_ddd_calls(records)
    merged: dict[str, Any] = {}
    for category, turn_summary in turns.items():
        merged[category] = {**turn_summary, **calls.get(category, {})}
    return merged


def summarize_official_baseline(path: Path) -> dict[str, Any]:
    data = load_json(path)
    records = data["records"]
    naive = summarize_method_records(records, "naive")
    eagle = summarize_method_records(records, "eagle")
    match = summarize_match_records(records)
    categories = category_order_from_records(records)
    by_category: dict[str, Any] = {}
    for category in categories:
        naive_tok = naive[category]["tok_per_s"]
        eagle_tok = eagle[category]["tok_per_s"]
        by_category[category] = {
            "naive": naive[category],
            "eagle": eagle[category],
            "speedup": round6(safe_div(eagle_tok or 0.0, naive_tok or 0.0)),
            "match": match[category],
        }
    return {
        "summary": data["summary"],
        "category_order": categories,
        "by_category": by_category,
    }


def summarize_fixed_depth(path: Path) -> dict[str, Any]:
    data = load_json(path)
    by_depth: dict[str, Any] = {}
    category_order: list[str] = []
    for result in data["depth_results"]:
        depth = str(result["depth"])
        if not category_order:
            category_order = category_order_from_records(result["records"])
        by_depth[depth] = {
            "summary": result["summary"],
            "by_category": summarize_plain_records(result["records"]),
        }
    return {
        "summary": data["summary"],
        "category_order": category_order,
        "by_depth": by_depth,
    }


def summarize_ddd(path: Path) -> dict[str, Any]:
    data = load_json(path)
    by_threshold: dict[str, Any] = {}
    category_order: list[str] = []
    for result in data["threshold_results"]:
        threshold = str(result["threshold"])
        if not category_order:
            category_order = category_order_from_records(result["records"])
        by_threshold[threshold] = {
            "summary": result["summary"],
            "by_category": merge_ddd_turns_and_calls(result["records"]),
        }
    return {
        "summary": data["summary"],
        "category_order": category_order,
        "by_threshold": by_threshold,
    }


def summarize_opt(path: Path) -> dict[str, Any]:
    data = load_json(path)
    by_config: dict[str, Any] = {}
    category_order: list[str] = []
    for result in data["config_results"]:
        name = result["name"]
        if not category_order:
            category_order = category_order_from_records(result["records"])
        by_config[name] = {
            "summary": result["summary"],
            "use_opt_tree": result.get("use_opt_tree"),
            "opt_expand_factor": result.get("opt_expand_factor"),
            "by_category": summarize_plain_records(result["records"]),
        }
    return {
        "summary": data["summary"],
        "category_order": category_order,
        "by_config": by_config,
    }


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def num(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def table(lines: list[list[str]]) -> str:
    header = "| " + " | ".join(lines[0]) + " |"
    sep = "| " + " | ".join(["---"] * len(lines[0])) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in lines[1:]]
    return "\n".join([header, sep, *rows])


def render_readme(summary: dict[str, Any]) -> str:
    order = summary["official_baseline"]["category_order"]
    official = summary["official_baseline"]["by_category"]
    fixed = summary["fixed_depth"]["by_depth"]
    ddd = summary["ddd"]["by_threshold"]
    ddd_optimized = summary.get("ddd_optimized", {}).get("by_threshold", {})
    opt = summary["opt_dynamic"]["by_config"]
    tau_m2 = next(key for key in ddd if float(key) == -2.0)
    tau_m05 = next(key for key in ddd if float(key) == -0.5)
    tau_m1 = None
    if ddd_optimized:
        tau_m1 = next((key for key in ddd_optimized if float(key) == -1.0), None)

    lines: list[str] = [
        "# E5.2 MT-Bench Category Analysis",
        "",
        "本目录汇总全量 MT-Bench 80 records / 160 turns 在不同 category 上的结果。",
        "这里把 MT-Bench 的 8 个 category 作为不同任务子集，用同一批全量实验日志做离线聚合；没有额外引入小样本手写 prompt。",
        "",
        "## EAGLE-3 baseline vs naive",
        "",
    ]
    baseline_rows = [["category", "turns", "naive tok/s", "EAGLE tok/s", "speedup", "turn match"]]
    for category in order:
        item = official[category]
        baseline_rows.append(
            [
                category,
                str(item["eagle"]["turns"]),
                num(item["naive"]["tok_per_s"]),
                num(item["eagle"]["tok_per_s"]),
                num(item["speedup"], 3),
                pct(item["match"]["turn_match_rate"]),
            ]
        )
    lines.extend([table(baseline_rows), ""])

    lines.extend(["## Fixed depth: depth 5 vs depth 11", ""])
    fixed_rows = [["category", "d5 tok/s", "d5 accept", "d11 tok/s", "d11 accept", "d11/d5 tok/s"]]
    for category in order:
        d5 = fixed["5"]["by_category"][category]
        d11 = fixed["11"]["by_category"][category]
        ratio = safe_div(d11["tok_per_s"] or 0.0, d5["tok_per_s"] or 0.0)
        fixed_rows.append(
            [
                category,
                num(d5["tok_per_s"]),
                num(d5["accept_per_step"], 3),
                num(d11["tok_per_s"]),
                num(d11["accept_per_step"], 3),
                num(ratio, 3),
            ]
        )
    lines.extend([table(fixed_rows), ""])

    lines.extend(["## DDD threshold sweep by category", ""])
    ddd_rows = [
        [
            "category",
            "fixed d11 tok/s",
            "tau=-2 tok/s",
            "tau=-2 depth",
            "tau=-2 early",
            "tau=-0.5 tok/s",
            "tau=-0.5 depth",
            "tau=-0.5 early",
        ]
    ]
    for category in order:
        d11 = fixed["11"]["by_category"][category]
        t2 = ddd[tau_m2]["by_category"][category]
        t05 = ddd[tau_m05]["by_category"][category]
        ddd_rows.append(
            [
                category,
                num(d11["tok_per_s"]),
                num(t2["tok_per_s"]),
                num(t2["mean_actual_depth"], 3),
                pct(t2["early_stop_rate"]),
                num(t05["tok_per_s"]),
                num(t05["mean_actual_depth"], 3),
                pct(t05["early_stop_rate"]),
            ]
        )
    lines.extend([table(ddd_rows), ""])

    if tau_m1 is not None:
        lines.extend(["## Optimized DDD tau=-1.0 by category", ""])
        opt_ddd_rows = [
            [
                "category",
                "fixed d11 tok/s",
                "tau=-1 tok/s",
                "tau=-1 accept",
                "tau=-1 depth",
                "tau=-1 early",
                "tau=-1/d11 tok/s",
            ]
        ]
        for category in order:
            d11 = fixed["11"]["by_category"][category]
            t1 = ddd_optimized[tau_m1]["by_category"][category]
            ratio = safe_div(t1["tok_per_s"] or 0.0, d11["tok_per_s"] or 0.0)
            opt_ddd_rows.append(
                [
                    category,
                    num(d11["tok_per_s"]),
                    num(t1["tok_per_s"]),
                    num(t1["accept_per_step"], 3),
                    num(t1["mean_actual_depth"], 3),
                    pct(t1["early_stop_rate"]),
                    num(ratio, 3),
                ]
            )
        lines.extend([table(opt_ddd_rows), ""])

    lines.extend(["## OPT-Tree dynamic baseline by category", ""])
    opt_rows = [["category", "EAGLE tok/s", "OPT tok/s", "OPT/EAGLE", "EAGLE accept", "OPT accept", "accept delta"]]
    for category in order:
        base = opt["EAGLE-3"]["by_category"][category]
        opt15 = opt["OPT-1.5"]["by_category"][category]
        ratio = safe_div(opt15["tok_per_s"] or 0.0, base["tok_per_s"] or 0.0)
        delta = None
        if opt15["accept_per_step"] is not None and base["accept_per_step"] is not None:
            delta = opt15["accept_per_step"] - base["accept_per_step"]
        opt_rows.append(
            [
                category,
                num(base["tok_per_s"]),
                num(opt15["tok_per_s"]),
                num(ratio, 3),
                num(base["accept_per_step"], 3),
                num(opt15["accept_per_step"], 3),
                num(delta, 3),
            ]
        )
    lines.extend([table(opt_rows), ""])

    lines.extend(
        [
            "## Main observations",
            "",
            "- MT-Bench category 差异很明显：EAGLE baseline 的 category speedup 约在 3.39x 到 4.26x 之间，说明不同任务子集本身会显著改变复现结论。",
            "- Full MT-Bench 上 depth 11 是比 depth 5 更强的固定深度点；按 category 看，大多数子集也呈现 depth 11 更优，但幅度不一致。",
            "- 早期 DDD 阈值 tau=-2.0 / tau=-0.5 在 full MT-Bench 总体上没有超过 fixed depth 11；补充的 tau=-1.0 则在全量总体上成为当前最佳配置。",
            "- tau=-1.0 的 category 表显示，不同任务子集仍有明显差异；在当前全量结果中，8 个 category 均高于 fixed depth 11，但提升幅度从 1.063x 到 1.618x 不等。",
            "- OPT-Tree 的 accept delta 在所有 category 上都是 0，和 E4.1 的树集合 Jaccard=1.0 一致：在当前 EAGLE-3 框架里 OPT-Tree 没有改变最终候选树，也没有带来算法层面的接受长度提升。",
            "",
            "详细机器可读结果见 `mt_bench_category_summary.json`。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official-baseline",
        type=Path,
        default=ROOT / "experiments" / "E1_official_baseline" / "official_baseline_mt_bench_limit-80_max-128.json",
    )
    parser.add_argument(
        "--fixed-depth",
        type=Path,
        default=ROOT / "experiments" / "E3_ddd" / "fixed_depth_mt_bench_limit-80_max-128_depths-5-11.json",
    )
    parser.add_argument(
        "--ddd",
        type=Path,
        default=ROOT / "experiments" / "E3_ddd" / "ddd_sweep_mt_bench_limit-80_max-128_depth-11_tau-m2p0-m0p5.json",
    )
    parser.add_argument(
        "--ddd-optimized",
        type=Path,
        default=ROOT / "experiments" / "E6_optimizations" / "ddd_sweep_mt_bench_limit-80_max-128_depth-11_tau-m1p0-m0p3.json",
    )
    parser.add_argument(
        "--opt-dynamic",
        type=Path,
        default=ROOT / "experiments" / "E4_opt_tree" / "opt_dynamic_mt_bench_limit-80_max-128_expand-1p5_single_repeat.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "inputs": {
            "official_baseline": str(args.official_baseline),
            "fixed_depth": str(args.fixed_depth),
            "ddd": str(args.ddd),
            "ddd_optimized": str(args.ddd_optimized),
            "opt_dynamic": str(args.opt_dynamic),
        },
        "official_baseline": summarize_official_baseline(args.official_baseline),
        "fixed_depth": summarize_fixed_depth(args.fixed_depth),
        "ddd": summarize_ddd(args.ddd),
        "ddd_optimized": summarize_ddd(args.ddd_optimized),
        "opt_dynamic": summarize_opt(args.opt_dynamic),
    }

    json_path = args.output_dir / "mt_bench_category_summary.json"
    readme_path = args.output_dir / "README.md"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    readme_path.write_text(render_readme(summary), encoding="utf-8")

    print(f"[SAVED] {json_path}")
    print(f"[SAVED] {readme_path}")


if __name__ == "__main__":
    main()
