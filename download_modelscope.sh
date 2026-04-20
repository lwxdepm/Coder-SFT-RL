#!/bin/bash
# ============================================================
# 从 ModelScope (魔搭) 下载模型
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
# 配置区：在此添加/修改要下载的模型
# ============================================================

# 默认下载目录
DOWNLOAD_DIR="${1:-$HOME/.cache/modelscope/hub}"

# 要下载的模型列表 (每行: model_id repo_type)
# repo_type: model (默认), dataset
MODELS=(
    "Qwen/Qwen2.5-Coder-1.5B-Instruct    model"
    "Qwen/Qwen2.5-Coder-0.5B-Instruct    model"
    "evalplus/mbppplus                     dataset"
)

# ============================================================
# 脚本开始
# ============================================================

echo ""
echo "============================================================"
echo "  ModelScope 模型下载"
echo "============================================================"
echo "  下载目录: $DOWNLOAD_DIR"
echo "============================================================"
echo ""

# 检查 modelscope 是否已安装
if ! python3 -c "import modelscope" 2>/dev/null; then
    log_error "modelscope 未安装"
    echo ""
    read -p "是否现在安装? (y/n): " INSTALL_MS
    if [ "$INSTALL_MS" = "y" ] || [ "$INSTALL_MS" = "Y" ]; then
        pip install modelscope -q
        log_ok "modelscope 安装完成"
    else
        exit 1
    fi
fi

MS_VER=$(python3 -c "import modelscope; print(modelscope.__version__)" 2>/dev/null || echo "unknown")
log_info "modelscope 版本: $MS_VER"
echo ""

# 逐个下载
FAILED=0
SUCCESS=0

for entry in "${MODELS[@]}"; do
    MODEL_ID=$(echo "$entry" | awk '{print $1}')
    REPO_TYPE=$(echo "$entry" | awk '{print $2}')
    [ -z "$REPO_TYPE" ] && REPO_TYPE="model"

    TARGET_DIR="$DOWNLOAD_DIR/$MODEL_ID"

    if [ -f "$TARGET_DIR/config.json" ]; then
        log_warn "$MODEL_ID 已存在，跳过"
        SUCCESS=$((SUCCESS + 1))
        continue
    fi

    log_info "下载 $REPO_TYPE: $MODEL_ID"

    if [ "$REPO_TYPE" = "model" ]; then
        python3 -c "
from modelscope import snapshot_download
model_id = '$MODEL_ID'
cache_dir = '$DOWNLOAD_DIR'
print(f'正在下载 {model_id} ...')
path = snapshot_download(model_id, cache_dir=cache_dir)
print(f'下载完成: {path}')
"
    elif [ "$REPO_TYPE" = "dataset" ]; then
        python3 -c "
from modelscope import dataset_download
dataset_id = '$MODEL_ID'
cache_dir = '$DOWNLOAD_DIR'
print(f'正在下载数据集 {dataset_id} ...')
path = dataset_download(dataset_id, cache_dir=cache_dir)
print(f'下载完成: {path}')
"
    fi

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
echo "  下载目录: $DOWNLOAD_DIR"
echo ""

# 列出已下载的模型
if [ -d "$DOWNLOAD_DIR" ]; then
    echo "  已下载内容:"
    find "$DOWNLOAD_DIR" -maxdepth 3 -name "config.json" -o -name "*.parquet" 2>/dev/null | while read f; do
        echo "    $f"
    done
fi
echo ""
