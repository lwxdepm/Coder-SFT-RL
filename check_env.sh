#!/bin/bash
# ============================================================
# Code-RL 环境兼容性检查脚本
# 在 code-rl 环境中运行: bash check_env.sh
# ============================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

pass_count=0
warn_count=0
fail_count=0

pass()  { echo -e "  ${GREEN}[PASS]${NC} $1"; pass_count=$((pass_count+1)); }
warn()  { echo -e "  ${YELLOW}[WARN]${NC} $1"; warn_count=$((warn_count+1)); }
fail()  { echo -e "  ${RED}[FAIL]${NC} $1"; fail_count=$((fail_count+1)); }
info()  { echo -e "  ${CYAN}[INFO]${NC} $1"; }

echo ""
echo "============================================================"
echo "  Code-RL 环境兼容性检查"
echo "============================================================"
echo ""

# -----------------------------------------------------------
# 1. Python 版本
# -----------------------------------------------------------
echo "── Python ──────────────────────────────────────────"
PY_VER=$(python3 --version 2>&1 | grep -oP '\K[0-9]+\.[0-9]+')
if python3 -c "exit(0 if (3,10) <= tuple(map(int, '$PY_VER'.split('.'))) <= (3,12) else 1)"; then
    pass "Python $PY_VER (兼容 3.10-3.12)"
else
    fail "Python $PY_VER (需要 3.10-3.12)"
fi

# -----------------------------------------------------------
# 2. PyTorch + CUDA
# -----------------------------------------------------------
echo "── PyTorch ─────────────────────────────────────────"
PYTORCH_OK=true

if ! python3 -c "import torch" 2>/dev/null; then
    fail "PyTorch 未安装"
    PYTORCH_OK=false
else
    T_VER=$(python3 -c "import torch; print(torch.__version__)")
    CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda)")
    pass "PyTorch $T_VER"
    pass "CUDA $CUDA_VER (from PyTorch)"

    # 检查 GPU 可用
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
        GPU_MEM=$(python3 -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}')")
        pass "GPU: $GPU_NAME ($GPU_MEM GB)"

        # 显存是否够用
        if python3 -c "import torch; exit(0 if torch.cuda.get_device_properties(0).total_memory > 12*1024**3 else 1)" 2>/dev/null; then
            pass "显存 >= 12GB，足够训练"
        else
            warn "显存 < 12GB，训练可能 OOM"
        fi
    else
        fail "CUDA 不可用，无法使用 GPU"
    fi
fi

# -----------------------------------------------------------
# 3. vLLM
# -----------------------------------------------------------
echo "── vLLM ────────────────────────────────────────────"
if python3 -c "import vllm" 2>/dev/null; then
    V_VER=$(python3 -c "import vllm; print(vllm.__version__)")
    pass "vLLM $V_VER"

    # 检查 vLLM 能否检测到 GPU
    VLLM_CUDA_ERR=$(python3 -c "
import vllm
try:
    from vllm.utils import cuda_device_count_stateless
    count = cuda_device_count_stateless()
    print(f'vLLM 检测到 {count} 个 GPU')
except:
    print('unknown')
" 2>/dev/null || echo "unknown")
    info "$VLLM_CUDA_ERR"
else
    ERR=$(python3 -c "import vllm" 2>&1)
    fail "vLLM 未安装或加载失败"
    echo "    错误: $ERR"
fi

# -----------------------------------------------------------
# 4. Transformers
# -----------------------------------------------------------
echo "── Transformers ────────────────────────────────────"
if python3 -c "import transformers" 2>/dev/null; then
    T_VER=$(python3 -c "import transformers; print(transformers.__version__)")
    pass "transformers $T_VER"
else
    fail "transformers 未安装"
fi

# vLLM + Transformers 兼容性
if python3 -c "import vllm, transformers" 2>/dev/null; then
    COMPAT_ERR=$(python3 -c "
import vllm
import transformers
# 尝试触发已知的 aimv2 冲突
from vllm.transformers_utils.configs import OvisConfig
" 2>&1 || true)
    if echo "$COMPAT_ERR" | grep -q "already used"; then
        fail "vLLM 与 transformers 存在 aimv2 命名冲突，降级 transformers: pip install 'transformers<4.54.0'"
    else
        pass "vLLM + transformers 兼容"
    fi
fi

# -----------------------------------------------------------
# 5. VERL
# -----------------------------------------------------------
echo "── VERL ────────────────────────────────────────────"
if python3 -c "import verl" 2>/dev/null; then
    V_VER=$(python3 -c "import verl; print(getattr(verl, '__version__', 'dev'))")
    pass "verl $V_VER"
else
    fail "verl 未安装"
fi

# 检查 VERL 的 PPO trainer 是否可用
if python3 -c "from verl.trainer.main_ppo import main" 2>/dev/null; then
    pass "verl.trainer.main_ppo 可用"
else
    warn "verl.trainer.main_ppo 导入失败 (可能影响训练)"
fi

# -----------------------------------------------------------
# 6. FlashAttention
# -----------------------------------------------------------
echo "── FlashAttention ──────────────────────────────────"
if python3 -c "import flash_attn" 2>/dev/null; then
    FA_VER=$(python3 -c "import flash_attn; print(flash_attn.__version__)")
    pass "flash-attn $FA_VER"
else
    warn "flash-attn 未安装 (不影响训练，仅影响速度)"
fi

# -----------------------------------------------------------
# 7. LoRA / PEFT
# -----------------------------------------------------------
echo "── PEFT ────────────────────────────────────────────"
if python3 -c "import peft" 2>/dev/null; then
    P_VER=$(python3 -c "import peft; print(peft.__version__)")
    pass "peft $P_VER"
else
    fail "peft 未安装 (需要用于 LoRA 导出)"
fi

# -----------------------------------------------------------
# 8. 数据集
# -----------------------------------------------------------
echo "── Datasets ────────────────────────────────────────"
if python3 -c "import datasets" 2>/dev/null; then
    D_VER=$(python3 -c "import datasets; print(datasets.__version__)")
    pass "datasets $D_VER"
else
    warn "datasets 未安装"
fi

# -----------------------------------------------------------
# 9. 其他工具
# -----------------------------------------------------------
echo "── 其他工具 ──────────────────────────────────────────"
for pkg in pandas numpy wandb rich gradio safetensors; do
    if python3 -c "import $pkg" 2>/dev/null; then
        V=$(python3 -c "import $pkg; print(getattr($pkg, '__version__', 'ok'))" 2>/dev/null || echo "ok")
        pass "$pkg $V"
    else
        warn "$pkg 未安装"
    fi
done

# -----------------------------------------------------------
# 10. Firejail 沙箱
# -----------------------------------------------------------
echo "── 沙箱 ──────────────────────────────────────────────"
if command -v firejail &>/dev/null; then
    pass "firejail 已安装"
else
    warn "firejail 未安装，将使用 subprocess 回退"
fi

# -----------------------------------------------------------
# 11. 数据文件检查
# -----------------------------------------------------------
echo "── 项目数据 ──────────────────────────────────────────"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -f "$SCRIPT_DIR/data/raw/kodcode_dataset.jsonl" ]; then
    LINES=$(wc -l < "$SCRIPT_DIR/data/raw/kodcode_dataset.jsonl")
    pass "原始数据存在: kodcode_dataset.jsonl ($LINES 行)"
else
    warn "原始数据不存在: data/raw/kodcode_dataset.jsonl"
    info "需要先准备训练数据"
fi

if [ -f "$SCRIPT_DIR/data/final/codea1_verify.parquet" ]; then
    pass "训练数据存在: codea1_verify.parquet"
else
    warn "训练数据不存在: data/final/codea1_verify.parquet"
fi

# -----------------------------------------------------------
# 汇总
# -----------------------------------------------------------
echo ""
echo "============================================================"
echo "  检查结果汇总"
echo "============================================================"
echo -e "  ${GREEN}通过: $pass_count${NC}"
echo -e "  ${YELLOW}警告: $warn_count${NC}"
echo -e "  ${RED}失败: $fail_count${NC}"
echo ""

if [ $fail_count -eq 0 ]; then
    echo -e "${GREEN}环境就绪！可以开始训练。${NC}"
else
    echo -e "${RED}有 $fail_count 个关键问题需要解决。${NC}"
fi
echo ""
