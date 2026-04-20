#!/bin/bash
# ============================================================
# 从 HuggingFace 下载模型
# ============================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { echo -e "${CYAN}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ============================================================
# 配置区
# ============================================================

# 项目目录下的 models 文件夹
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOWNLOAD_DIR="$PROJECT_DIR/models"

# 要下载的模型列表
# 格式: model_id (HuggingFace repo ID)
MODELS=(
    "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    "Qwen/Qwen2.5-Coder-0.5B-Instruct"
)

# ============================================================
# 脚本开始
# ============================================================

echo ""
echo "============================================================"
echo "  HuggingFace 模型下载"
echo "============================================================"
echo "  下载目录: $DOWNLOAD_DIR"
echo "============================================================"
echo ""

# 检查 huggingface_hub 是否已安装
if ! python3 -c "from huggingface_hub import snapshot_download" 2>/dev/null; then
    log_error "huggingface_hub 未安装"
    echo ""
    read -p "是否现在安装? (y/n): " INSTALL_HF
    if [ "$INSTALL_HF" = "y" ] || [ "$INSTALL_HF" = "Y" ]; then
        pip install huggingface_hub -q
        log_ok "huggingface_hub 安装完成"
    else
        exit 1
    fi
fi

HF_VER=$(python3 -c "from huggingface_hub import __version__; print(__version__)" 2>/dev/null || echo "unknown")
log_info "huggingface_hub 版本: $HF_VER"
echo ""

# 逐个下载
FAILED=0
SUCCESS=0

for MODEL_ID in "${MODELS[@]}"; do
    # 目标路径: models/Qwen/Qwen2.5-Coder-1.5B-Instruct/
    TARGET_DIR="$DOWNLOAD_DIR/$MODEL_ID"

    # 检查是否已存在
    if [ -f "$TARGET_DIR/config.json" ]; then
        log_warn "$MODEL_ID 已存在，跳过"
        SUCCESS=$((SUCCESS + 1))
        continue
    fi

    mkdir -p "$TARGET_DIR"
    log_info "下载: $MODEL_ID"
    log_info "目标: $TARGET_DIR"

    python3 -c "
from huggingface_hub import snapshot_download
model_id = '$MODEL_ID'
local_dir = '$TARGET_DIR'
print(f'正在下载 {model_id} ...')
path = snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)
print(f'下载完成: {path}')
"

    if [ $? -eq 0 ]; then
        log_ok "$MODEL_ID 下载完成"
        SUCCESS=$((SUCCESS + 1))
    else
        log_error "$MODEL_ID 下载失败"
        FAILED=$((FAILED + 1))
    fi

    echo ""
done

# ============================================================
echo "============================================================"
echo "  下载完成"
echo "============================================================"
echo -e "  ${GREEN}成功: $SUCCESS${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "  ${RED}失败: $FAILED${NC}"
fi
echo ""

# 列出已下载的模型
echo "  已下载内容:"
find "$DOWNLOAD_DIR" -maxdepth 3 -name "config.json" 2>/dev/null | while read f; do
    DIR=$(dirname "$f")
    SIZE=$(du -sh "$DIR" 2>/dev/null | awk '{print $1}')
    echo "    $DIR ($SIZE)"
done
echo ""
