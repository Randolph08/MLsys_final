#!/bin/bash
# ============================================================
# 模型与 Checkpoint 下载脚本
# 以下所有文件均可从 HuggingFace 公开下载
# ============================================================
set -e

echo "========================================"
echo "  MLsys Final Project - 模型下载脚本"
echo "========================================"

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="${PROJECT_ROOT}/models"
CKPT_DIR="${PROJECT_ROOT}/EAGLE_checkpoints"

# -------- 1. Target Model: Llama-3.1-8B-Instruct --------
echo ""
echo "[1/3] 下载 Target Model: Llama-3.1-8B-Instruct"
echo "  来源: meta-llama/Llama-3.1-8B-Instruct"
echo "  大小: ~15GB"
mkdir -p "${MODELS_DIR}/Llama-3.1-8B-Instruct"

# 使用 huggingface_hub 或 git clone
if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
        --local-dir "${MODELS_DIR}/Llama-3.1-8B-Instruct" \
        --exclude "*.pth" "*.bin"
else
    echo "  [INFO] huggingface-cli 未安装，使用 git-lfs clone"
    echo "  请确保已安装 git-lfs: git lfs install"
    git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct \
        "${MODELS_DIR}/Llama-3.1-8B-Instruct"
fi

# -------- 2. Drafter Model (可选): DeepSeek-R1-Distill-Llama-8B --------
echo ""
echo "[2/3] 下载 Drafter Model(可选): DeepSeek-R1-Distill-Llama-8B"
echo "  来源: deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
echo "  大小: ~15GB"
mkdir -p "${MODELS_DIR}/DeepSeek-R1-Distill-Llama-8B"

if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --local-dir "${MODELS_DIR}/DeepSeek-R1-Distill-Llama-8B" \
        --exclude "*.pth" "*.bin"
else
    git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        "${MODELS_DIR}/DeepSeek-R1-Distill-Llama-8B"
fi

# -------- 3. EAGLE-3 Checkpoints --------
echo ""
echo "[3/3] 下载 EAGLE-3 Checkpoints"
echo "  来源: SafeAILab/EAGLE (HuggingFace)"
echo ""

# EAGLE-3 for LLaMA-3.1-8B-Instruct
echo "  [3a] EAGLE3-LLaMA3.1-Instruct-8B"
mkdir -p "${CKPT_DIR}/EAGLE3-LLaMA3.1-Instruct-8B"
if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download SafeAILab/EAGLE3-LLaMA3.1-Instruct-8B \
        --local-dir "${CKPT_DIR}/EAGLE3-LLaMA3.1-Instruct-8B"
else
    git clone https://huggingface.co/SafeAILab/EAGLE3-LLaMA3.1-Instruct-8B \
        "${CKPT_DIR}/EAGLE3-LLaMA3.1-Instruct-8B"
fi

# EAGLE-3 for DeepSeek-R1-Distill-Llama-8B
echo "  [3b] EAGLE3-DeepSeek-R1-Distill-LLaMA-8B"
mkdir -p "${CKPT_DIR}/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B"
if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download SafeAILab/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B \
        --local-dir "${CKPT_DIR}/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B"
else
    git clone https://huggingface.co/SafeAILab/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B \
        "${CKPT_DIR}/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B"
fi

echo ""
echo "========================================"
echo "  下载完成！"
echo "========================================"
echo ""
echo "  模型位置:"
echo "    Target Model:  ${MODELS_DIR}/Llama-3.1-8B-Instruct"
echo "    Drafter Model: ${MODELS_DIR}/DeepSeek-R1-Distill-Llama-8B"
echo "    EAGLE CKPT 1:  ${CKPT_DIR}/EAGLE3-LLaMA3.1-Instruct-8B"
echo "    EAGLE CKPT 2:  ${CKPT_DIR}/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B"
echo ""
echo "  接下来运行: pip install -r requirements.txt"
echo "========================================"
