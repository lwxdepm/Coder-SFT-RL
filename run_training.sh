#!/bin/bash
set -e
BASE_MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct"
MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct"
OUTPUT_DIR="checkpoints/micro-grpo-$(date +%Y%m%d-%H%M%S)"

echo "========================================"
echo "  Micro-GRPO 训练"
echo "  Model: $BASE_MODEL"
echo "  Output: $OUTPUT_DIR"
echo "========================================"
# 清理旧进程
pkill -9 -f "python.*verl" 2>/dev/null || true
pkill -9 -f "ray" 2>/dev/null || true
sleep 3

echo ""
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
echo ""

mkdir -p $OUTPUT_DIR
WANDB_PROJECT="code-rl-micro-grpo"

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=true \
    \
    data.train_files=/home/featurize/code-r1/data/final/codea1_verify.parquet \
    data.val_files=data/processed/val.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=500 \
    data.max_response_length=400 \
    data.prompt_key=prompt \
    \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=512 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8\
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2\
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.max_num_seqs=24 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=512 \
    actor_rollout_ref.rollout.prompt_length=500 \
    actor_rollout_ref.rollout.response_length=400 \
    actor_rollout_ref.rollout.temperature=0.8\
    actor_rollout_ref.rollout.free_cache_engine=True\
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.disable_log_stats=True \
    \
    custom_reward_function.path=reward/reward_func.py \
    custom_reward_function.name=compute_score \
    \
    reward_model.enable=false \
    \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name="grpo-coder-1.5b" \
    trainer.n_gpus_per_node=1 \
    trainer.total_epochs=1 \
    trainer.val_before_train=false \
    trainer.save_freq=50 \
    trainer.default_local_dir=$OUTPUT_DIR \
    \
    actor_rollout_ref.actor.optim.lr=3e-5 \
    \
    2>&1 | tee $OUTPUT_DIR/train.log

echo ""
echo "========================================"
echo "  训练完成!"
echo "  日志=$OUTPUT_DIR/train.log"
echo "========================================"