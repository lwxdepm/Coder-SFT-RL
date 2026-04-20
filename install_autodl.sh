#!/bin/bash
# ============================================================
# Code-RL AutoDL 环境安装脚本 (v3)
# 适配: PyTorch 2.7.0 + CUDA 12.8 + Python 3.12 + RTX 4090
# 注意: 请在 code-rl 环境中运行: conda activate code-rl
# ============================================================
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { echo -e "${CYAN}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "============================================================"
echo "  Code-RL AutoDL 环境安装"
echo "============================================================"
echo ""

# -----------------------------------------------------------
# Step 0: 环境检查
# -----------------------------------------------------------
log_info "检查环境..."

CONDA_ENV=$(conda info --envs 2>/dev/null | grep '*' | awk '{print $1}')
if [ "$CONDA_ENV" = "base" ]; then
    log_warn "当前在 base 环境中运行，建议在 code-rl 环境中执行"
    read -p "是否继续？(y/n，默认n): " CONTINUE_BASE
    if [ "$CONTINUE_BASE" != "y" ] && [ "$CONTINUE_BASE" != "Y" ]; then
        exit 0
    fi
else
    log_ok "当前环境: $CONDA_ENV"
fi

PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\K[0-9]+\.[0-9]+')
log_ok "Python 版本: $PYTHON_VERSION"

if python3 -c "import torch" &>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    log_ok "PyTorch 版本: $TORCH_VERSION"
    log_ok "GPU: $(python3 -c "import torch; print(torch.cuda.get_device_name(0))")"
else
    log_error "PyTorch 未安装"
    log_info "pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"
    exit 1
fi

echo ""

# -----------------------------------------------------------
# Step 1: Firejail
# -----------------------------------------------------------
log_info "安装 Firejail 沙箱..."
if command -v firejail &>/dev/null; then
    log_ok "Firejail 已安装"
else
    apt-get update -qq
    apt-get install -y -qq firejail >/dev/null 2>&1
    if command -v firejail &>/dev/null; then
        log_ok "Firejail 安装成功"
    else
        log_warn "Firejail 安装失败，将使用 subprocess 回退（不影响训练）"
    fi
fi

echo ""

# -----------------------------------------------------------
# Step 2: Python 依赖
# -----------------------------------------------------------
log_info "安装 Python 依赖..."

cd "$PROJECT_DIR"

# 先装 VERL，让它拉它需要的 numpy 版本
log_info "[1/5] 安装 VERL (从 GitHub main 分支)..."
pip install git+https://github.com/volcengine/verl.git
log_ok "VERL 安装完成"

log_info "[2/5] 安装 vLLM 0.9.1..."
pip install "vllm==0.9.1"
log_ok "vLLM 安装完成"

# 修复 numpy 版本冲突（VERL 可能装了 1.26，但 AutoDL 自带的 cupy/opencv 要 2.x）
log_info "[2.5/5] 升级 numpy 解决依赖冲突..."
pip install "numpy>=2.0,<2.5" --quiet
log_ok "numpy 已升级"

log_info "[3/5] 安装基础包..."
pip install \
    datasets \
    huggingface_hub \
    pandas \
    wandb \
    rich \
    tensorboard \
    matplotlib \
    seaborn \
    tqdm \
    pydantic \
    python-Levenshtein \
    peft \
    safetensors \
    gradio
log_ok "基础包安装完成"

log_info "[4/5] 安装 flash-attn (可选，无 nvcc 时会跳过)..."
FLASH_ATTN_OK=false
if command -v nvcc &>/dev/null; then
    if pip install flash-attn --no-build-isolation 2>&1 | tail -3; then
        FLASH_ATTN_OK=true
    fi
else
    log_warn "nvcc 未找到，跳过 flash-attn 编译"
    log_info "pip 尝试安装预编译 wheel，如果有则直接安装"
    if pip install flash-attn --no-cache-dir 2>&1 | tail -3; then
        FLASH_ATTN_OK=true
    fi
fi

if $FLASH_ATTN_OK; then
    log_ok "flash-attn 安装完成"
else
    log_warn "flash-attn 未安装（需要 nvcc 或匹配的预编译 wheel），不影响训练"
fi

echo ""

# -----------------------------------------------------------
# Step 3: 验证安装
# -----------------------------------------------------------
log_info "验证关键依赖..."

FAILURES=0
OPTIONAL_FAILURES=0

check_pkg() {
    local pkg=$1
    local mod=${2:-$1}
    local optional=${3:-false}

    # 显示真实错误，不隐藏 stderr
    ERR_MSG=$(python3 -c "import $mod; print(getattr($mod, '__version__', 'ok'))" 2>&1)
    if [ $? -eq 0 ]; then
        log_ok "$pkg: $ERR_MSG"
    else
        if [ "$optional" = "true" ]; then
            log_warn "$pkg 未安装（可选，不影响）"
            OPTIONAL_FAILURES=$((OPTIONAL_FAILURES + 1))
        else
            log_error "$pkg 未安装"
            echo "  错误详情: $ERR_MSG"
            FAILURES=$((FAILURES + 1))
        fi
    fi
}

check_pkg "torch" "torch" "false"
check_pkg "vllm" "vllm" "false"
check_pkg "transformers" "transformers" "false"
check_pkg "datasets" "datasets" "false"
check_pkg "verl" "verl" "false"
check_pkg "peft" "peft" "false"
check_pkg "gradio" "gradio" "false"
check_pkg "flash_attn" "flash_attn" "true"

echo ""

if [ $FAILURES -eq 0 ]; then
    log_ok "所有关键依赖安装成功！"
    if [ $OPTIONAL_FAILURES -gt 0 ]; then
        log_warn "$OPTIONAL_FAILURES 个可选依赖未安装（不影响训练）"
    fi
else
    log_error "$FAILURES 个关键依赖安装失败"
    exit 1
fi

# -----------------------------------------------------------
# Step 4: 后续步骤
# -----------------------------------------------------------
echo ""
echo "============================================================"
echo "  安装完成！后续步骤:"
echo "============================================================"
echo ""
echo "  1. 准备数据:"
echo "     python data/verify_dataset.py --input data/raw/kodcode_dataset.jsonl --output data/verified/codea1_verify.jsonl"
echo ""
echo "  2. 转换格式:"
echo "     python data/convert_dataset.py --input data/verified/codea1_verify.jsonl --output data/final/codea1_verify.parquet"
echo ""
echo "  3. 过滤超长样本 (修改 filter_overlong.py 路径):"
echo "     python data/filter_overlong.py"
echo ""
echo "  4. 启动训练:"
echo "     bash run_training.sh"
echo ""
echo "============================================================"
