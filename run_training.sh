#!/bin/bash
set -e

# vLLM 0.10+ V1 引擎（VERL main 分支需要）
export VLLM_USE_V1=1

# ============================================================
# Micro-GRPO 训练脚本
# 适配 AutoDL RTX 4090 (24GB) + VERL 0.8.0.dev0
# ============================================================

# 确保在项目根目录下运行，防止相对路径解析错误
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 关键安全检查: 验证项目目录结构完整
for dir in reward scripts data; do
    if [ ! -d "$dir" ]; then
        echo "❌ 项目目录缺失: $dir"
        echo "   请确认你在正确的项目根目录下运行此脚本"
        exit 1
    fi
done

MODEL="models/Qwen/Qwen2.5-Coder-1.5B-Instruct"
OUTPUT_DIR="checkpoints/micro-grpo-$(date +%Y%m%d-%H%M%S)"

# 数据路径 — 按需修改
TRAIN_DATA="data/final/codea1_filtered.parquet"

# 显存优化开关: true=启用 CPU offload, false=纯 GPU
OFFLOAD=true

echo "========================================"
echo "  Micro-GRPO 训练"
echo "  Model: $MODEL"
echo "  Output: $OUTPUT_DIR"
echo "  Offload: $OFFLOAD"
echo "  Train data: $TRAIN_DATA"
echo "========================================"

# 清理旧进程
pkill -9 -f "python.*verl" 2>/dev/null || true
pkill -9 -f "ray" 2>/dev/null || true
sleep 3

# 检查 GPU
echo ""
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
echo ""

# 检查数据文件
if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ 训练数据不存在: $TRAIN_DATA"
    echo "   请先准备数据: python data/convert_dataset.py ..."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
WANDB_PROJECT="code-rl-micro-grpo"

# W&B 配置: 防止断线导致 crashed 状态
export WANDB_PROJECT="$WANDB_PROJECT"
export WANDB_WATCH="gradients"  # 记录梯度统计（比默认 'all' 更轻量）
export WANDB_LOG_MODEL="checkpoint"

# 防 OOM: 在训练前打印显存基线
echo ""
echo "显存基线:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
echo ""

# ============================================================
# VERL 训练参数说明:
#  - algorithm: GRPO 算法
#  - data: 数据路径和长度
#  - actor_rollout_ref: 模型 + LoRA + rollout + vLLM
#  - custom_reward_function: 自定义奖励函数
#  - trainer: 日志和保存
# ============================================================

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=true \
    \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$TRAIN_DATA" \
    data.train_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.prompt_key=prompt \
    \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=1024 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload="$OFFLOAD" \
    actor_rollout_ref.ref.fsdp_config.param_offload="$OFFLOAD" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="$OFFLOAD" \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.max_num_seqs=24 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.rollout.prompt_length=512 \
    actor_rollout_ref.rollout.response_length=512 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.disable_log_stats=True \
    \
    custom_reward_function.path="reward/reward_func.py" \
    custom_reward_function.name="compute_score" \
    \
    reward_model.enable=false \
    \
    trainer.logger=['console','wandb'] \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="grpo-coder-1.5b" \
    trainer.n_gpus_per_node=1 \
    trainer.total_epochs=1 \
    trainer.val_before_train=false \
    trainer.save_freq=50 \
    trainer.default_local_dir="$OUTPUT_DIR" \
    \
    actor_rollout_ref.actor.optim.lr=3e-5 \
    \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo ""
echo "========================================"
echo "  训练完成!"
echo "  日志=$OUTPUT_DIR/train.log"
echo "========================================"
