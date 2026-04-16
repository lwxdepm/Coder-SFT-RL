# Code-RL: Code Generation via Reinforcement Learning

Code-RL is a reinforcement learning framework for training code generation models using execution-based rewards. The project integrates with VERL (Volcengine Efficient RL) and provides tools for dataset preparation, reward computation via sandboxed code execution, model training, evaluation, and interactive comparison.

## Features

- **Execution-based Rewards**: Uses sandboxed code execution (Firejail/subprocess) to compute pass rates as rewards
- **Production-Grade Reward Manager**: Optimized reward computation with process pool management, warmup, and smart batch scheduling
- **Dataset Support**: Supports multiple datasets including RLVR Code Data Python, HumanEval+, MBPP+
- **Parallel Execution**: Batch reward computation with parallel processing
- **Interactive Demo**: Gradio-based interface for comparing base vs trained models
- **Evaluation Suite**: Benchmark evaluation with pass@1, pass@k metrics
- **Temperature Scheduling**: Configurable temperature scheduling for generation diversity

## Project Structure

```
code-r1/
├── setup.py                    # Package setup
├── requirements.txt            # Dependencies
├── example_usage.py            # Example usage of core components
├── run_training.sh             # Training script
│
├── reward/                     # Reward computation module (Production-Grade)
│   ├── __init__.py
│   ├── reward_func.py         # Main reward manager (GRPO reward computation)
│   ├── executor.py            # Code executor with sandbox
│   ├── metrics.py             # Evaluation metrics
│   └── sandbox.py             # Sandbox implementations
│
├── compare_generate/          # Model comparison UI
│   ├── app.py                 # Gradio interface
│   ├── generate.py            # Code generator
│   └── compare_example.md     # Example comparisons
│
├── data/                      # Dataset handling
│   ├── prepare_dataset.py     # Dataset preparation
│   ├── verify_dataset.py      # Dataset verification
│   ├── raw/                   # Raw data
│   ├── processed/             # Processed data (train/val splits)
│   ├── verified/              # Verified data
│   └── test/                  # Test data
│
├── eval/                      # Evaluation tools
│   ├── benchmark.py           # Benchmark management
│   └── evaluate.py            # Model evaluation
│
├── scripts/                   # Utility scripts
│   ├── quick_test.sh
│   ├── prepare_data.sh
│   ├── test_model.py
│   ├── export_model.py
│   └── convert_to_parquet.py
│
└── wandb/                     # Weights & Biases logs
```

## Installation

### Prerequisites

- Python 3.9+
- Firejail (for sandboxed execution) or use subprocess fallback
- CUDA-capable GPU (recommended for training)

### Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd code-r1

# Install Python dependencies
pip install -r requirements.txt

# Install VERL from source
pip install git+https://github.com/volcengine/verl.git

# Install firejail (optional, for better sandboxing)
# On Ubuntu/Debian:
sudo apt-get install firejail
```

## Quick Start

### 1. Prepare Dataset

```bash
python data/prepare_dataset.py --sources rlvr --max_per_source 2000 --verify
```

This will download and process RLVR Code Data Python, verify solutions, and create train/val splits in `data/processed/`.

### 2. Run Example Usage

```bash
python example_usage.py
```

This demonstrates the code executor, temperature scheduler, and metrics calculation.

### 3. Launch Interactive Demo

```bash
cd compare_generate
python app.py
```

Open `http://localhost:7860` to compare base model (Qwen2.5-Coder-1.5B-Instruct) with trained model outputs.

## Usage

### Reward Computation

The core reward computation is implemented in `reward/reward_func.py`:

```python
from reward.reward_func import get_reward_manager, compute_score

# Using the reward manager directly
manager = get_reward_manager()
code = "def add(a, b): return a + b"
tests = ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"]
result = manager.compute_reward(code, tests)
print(f"Reward: {result['reward']}, Passed: {result['passed']}/{result['total']}")

# Or using the VERL-compatible compute_score function
result = compute_score(
    solution_str="def add(a, b): return a + b",
    extra_info={"tests": ["assert add(1, 2) == 3"]}
)
```

### Model Evaluation

Evaluate a model on benchmarks:

```bash
python eval/evaluate.py --model_path Qwen/Qwen2.5-Coder-1.5B-Instruct --benchmark mbpp --n_samples 1
```

## Configuration

### Sandbox Settings

By default, the system tries to use Firejail for secure code execution. If Firejail is not available, it falls back to subprocess execution. Configure in `reward/reward_func.py`:

```python
from reward.reward_func import CodeRewardManager

manager = CodeRewardManager(
    sandbox_type="firejail",  # or "subprocess"
    timeout=5.0,
    max_workers=None,  # Uses CPU cores - 1
    parallel_threshold=20,
    reward_config={  # Optional reward configuration
        'format_penalty': -1.5,
        'syntax_error_penalty': -1.0,
        'pass_rate_weight': 2.0,
        'full_pass_bonus': 1.0,
    }
)
```

### Reward Scaling

Rewards are scaled to range [-0.5, 1.0] where:
- -0.5: All tests failed or syntax error
- 0.0: No tests provided, valid syntax
- 1.0: All tests passed

## Training

The project uses execution-based rewards for RL training. Typical training workflow:

1. **Prepare dataset** in parquet format with prompt and test columns
2. **Configure reward function** using `reward/reward_func.py`
3. **Run training** with VERL and appropriate hyperparameters

Example training script (`run_training.sh`):
```bash
#!/bin/bash
set -e

MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct"
OUTPUT_DIR="checkpoints/micro-grpo-$(date +%Y%m%d-%H%M%S)"

echo "========================================"
echo "  Micro-GRPO 训练"
echo "  Model: $MODEL"
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
    data.train_files=data/processed/train.parquet \
    data.val_files=data/processed/val.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=1000 \
    data.max_response_length=4000 \
    data.prompt_key=prompt \
    \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=8 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.model.enable_gradient_checkpointing=True\
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload}\
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload}\
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload}\
    actor_rollout_ref.rollout.free_cache_engine=True\
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.prompt_length=1000 \
    actor_rollout_ref.rollout.response_length=4000 \
    actor_rollout_ref.rollout.temperature=0.7 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
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
    2>&1 | tee $OUTPUT_DIR/train.log

echo ""
echo "========================================"
echo "  训练完成!"
echo "  日志: $OUTPUT_DIR/train.log"
echo "========================================"
```

To run the training script:
```bash
chmod +x run_training.sh
./run_training.sh
```

## Contributing

1. **Code Style**: Follow existing code style (PEP 8)
2. **Testing**: Add tests for new functionality
3. **Documentation**: Update relevant documentation
4. **Pull Requests**: Submit PRs with clear descriptions

### Development Setup

```bash
# Install development dependencies
pip install black flake8 pytest

# Format code
black .

# Check code style
flake8

# Run tests (when available)
pytest
```

## License

[Specify License - TODO]

## Citation

If you use this code in your research, please cite:

```bibtex
[Citation information - TODO]
```

## Acknowledgments

- [VERL](https://github.com/volcengine/verl) for the efficient RL framework
- [Qwen Team](https://github.com/QwenLM) for the base models
- Dataset providers: RLVR Code Data Python, HumanEval+, MBPP+