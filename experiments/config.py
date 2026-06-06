"""
实验公共配置模块
所有实验脚本共享的模型路径、参数和输出目录配置。
"""
import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = "/home/hzliu/AD/Homework_haozhe/MLsys_final"
EAGLE_ROOT = os.path.join(PROJECT_ROOT, "EAGLE")
EXPERIMENTS_ROOT = os.path.join(PROJECT_ROOT, "experiments")

# 模型路径
BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Llama-3.1-8B-Instruct")
EA_MODEL_PATH = os.path.join(PROJECT_ROOT, "EAGLE_checkpoints", "EAGLE3-LLaMA3.1-Instruct-8B")

# 输出目录
E2_1_OUTPUT_DIR = os.path.join(EXPERIMENTS_ROOT, "E2.1_timing")
E2_2_OUTPUT_DIR = os.path.join(EXPERIMENTS_ROOT, "E2.2_acceptance")
E2_3_OUTPUT_DIR = os.path.join(EXPERIMENTS_ROOT, "E2.3_tree_util")

# GPU 配置
GPU_DEVICE = 0  # 使用 GPU 0（当前可用）


# ============================================================
# EAGLE 树参数
# ============================================================
@dataclass
class EAGLEConfig:
    """EAGLE 推理参数"""
    total_token: int = 60       # draft tree 中的总 token 数 + 1
    depth: int = 5              # beam search 最大扩展步数
    top_k: int = 10             # 每步保留的 top-k 候选
    temperature: float = 0.0    # 采样温度 (0 = greedy)
    max_new_tokens: int = 256   # 最大生成 token 数
    is_llama3: bool = True      # 使用 Llama-3 的 stop token 逻辑

    def __repr__(self) -> str:
        return (f"EAGLEConfig(total_token={self.total_token}, depth={self.depth}, "
                f"top_k={self.top_k}, temperature={self.temperature}, "
                f"max_new_tokens={self.max_new_tokens})")


# ============================================================
# E2.1 实验参数
# ============================================================
@dataclass
class E2_1_Config:
    """E2.1 管线时延分解实验参数"""
    # 测试 prompt 列表
    prompts: list = field(default_factory=lambda: [
        "The capital of France is",
        "Explain the concept of machine learning in simple terms:",
        "Write a Python function to find the nth Fibonacci number:",
        "What are the main causes of climate change?",
        "Translate the following to French: Hello, how are you?",
    ])

    # profiling 参数
    num_warmup_steps: int = 3   # warmup 步数（不计入统计）
    num_profile_steps: int = 80  # 每个 prompt 的 profiling 步数
    eagle_config: EAGLEConfig = field(default_factory=EAGLEConfig)

    # 输出
    output_dir: str = E2_1_OUTPUT_DIR
    save_raw_data: bool = True   # 是否保存逐 step 的原始数据


# ============================================================
# E2.2 实验参数
# ============================================================
@dataclass
class E2_2_Config:
    """E2.2 逐深度接受率分析实验参数"""
    prompts: list = field(default_factory=lambda: [
        "The capital of France is",
        "Explain the concept of machine learning in simple terms:",
        "Write a Python function to find the nth Fibonacci number:",
        "What are the main causes of climate change?",
        "Translate the following to French: Hello, how are you?",
    ])
    num_warmup_steps: int = 3
    eagle_config: EAGLEConfig = field(default_factory=EAGLEConfig)
    output_dir: str = E2_2_OUTPUT_DIR
    save_raw_data: bool = True


# ============================================================
# E2.3 实验参数
# ============================================================
@dataclass
class E2_3_Config:
    """E2.3 树节点利用效率分析实验参数"""
    prompts: list = field(default_factory=lambda: [
        "The capital of France is",
        "Explain the concept of machine learning in simple terms:",
        "Write a Python function to find the nth Fibonacci number:",
        "What are the main causes of climate change?",
        "Translate the following to French: Hello, how are you?",
    ])
    num_warmup_steps: int = 3
    eagle_config: EAGLEConfig = field(default_factory=EAGLEConfig)
    output_dir: str = E2_3_OUTPUT_DIR
    save_raw_data: bool = True


# ============================================================
# 实验结果保存/加载工具
# ============================================================
def save_experiment_data(data: dict, output_dir: str, filename: str):
    """保存实验结果到 JSON 文件"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"[SAVED] {filepath}")


def load_experiment_data(output_dir: str, filename: str) -> Optional[dict]:
    """加载已保存的实验结果"""
    filepath = os.path.join(output_dir, filename)
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return None


def experiment_data_exists(output_dir: str, filename: str) -> bool:
    """检查实验数据是否已存在"""
    return os.path.exists(os.path.join(output_dir, filename))


# ============================================================
# 确保输出目录存在
# ============================================================
for d in [E2_1_OUTPUT_DIR, E2_2_OUTPUT_DIR, E2_3_OUTPUT_DIR]:
    os.makedirs(os.path.join(d, "data"), exist_ok=True)
    os.makedirs(os.path.join(d, "figures"), exist_ok=True)
