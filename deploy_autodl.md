# Code-RL AutoDL 部署指南

> 目标环境: AutoDL RTX 4090 (24GB) / PyTorch 2.7.0 / CUDA 12.8 / Python 3.12 / Ubuntu 22.04

---

## 快速开始

```bash
# 进入项目目录
cd /path/to/Code-RL

# 一键安装（约 10-30 分钟，取决于网络）
bash install_autodl.sh
```

---

## 分步安装（手动模式）

如果自动脚本失败，可以按以下步骤手动安装：

### 1. 安装 Firejail 沙箱

```bash
sudo apt-get update
sudo apt-get install -y firejail
firejail --version   # 验证安装
```

### 2. 安装 Python 依赖

```bash
# 基础包
pip install datasets huggingface_hub pandas numpy \
    wandb rich tensorboard matplotlib seaborn tqdm \
    pydantic python-Levenshtein peft safetensors gradio

# vLLM（>=0.9.0 匹配 CUDA 12.8）
pip install "vllm>=0.9.0"

# VERL
pip install git+https://github.com/volcengine/verl.git

# flash-attn（可选，加速训练）
pip install flash-attn --no-build-isolation
```

### 3. 验证安装

```bash
python -c "
import torch, vllm, transformers, verl
print(f'PyTorch: {torch.__version__}')
print(f'vLLM: {vllm.__version__}')
print(f'transformers: {transformers.__version__}')
print(f'VERL: {verl.__version__}')
"
```

---

## 完整工作流

### Step 1: 准备数据

```bash
# 验证原始数据（过滤掉参考答案无法通过测试的噪声样本）
python data/verify_dataset.py \
    --input data/raw/kodcode_dataset.jsonl \
    --output data/verified/codea1_verify.jsonl \
    --n_runs 4 --n_workers 64

# 转换为 VERL 训练格式
python data/convert_dataset.py \
    --input data/verified/codea1_verify.jsonl \
    --output data/final/codea1_verify.parquet \
    --data-source codea1
```

### Step 2: 过滤超长样本

需要修改 `data/filter_overlong.py` 中的硬编码路径：

```python
# 将 /home/featurize/code-r1/... 改为你的实际路径
filter_long_samples(
    'data/final/codea1_verify.parquet',
    'Qwen/Qwen2.5-Coder-1.5B-Instruct',
    "prompt",
    500,   # max prompt tokens
    'data/final/codea1_verify.parquet'
)
```

### Step 3: 启动训练

```bash
bash run_training.sh
```

训练参数说明（来自 `run_training.sh`）：

| 参数 | 值 | 说明 |
|------|-----|------|
| 基础模型 | Qwen2.5-Coder-1.5B-Instruct | 3.5B 参数 |
| LoRA rank | 16 | ~76MB 额外显存 |
| batch_size | 32 | 训练批次大小 |
| rollout n | 4 | 每条 prompt生成4条响应 |
| temperature | 0.8 | 生成多样性 |
| max_prompt | 500 tokens | prompt 最大长度 |
| max_response | 400 tokens | 响应最大长度 |
| lr | 3e-5 | 学习率 |
| epochs | 1 | 训练轮数 |

### Step 4: 导出模型

```bash
python scripts/export_model.py \
    --checkpoint_dir checkpoints/micro-grpo-2026xxxx-xxxxxx \
    --output_dir exported_models/grpo_merged \
    --base_model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --test
```

### Step 5: 评估

```bash
# MBPP+ 评估
python eval/evaluate.py \
    --model_path exported_models/grpo_merged \
    --benchmark mbpp \
    --n_samples 1 \
    --temperature 0.2

# HumanEval+ 评估
python eval/evaluate.py \
    --model_path exported_models/grpo_merged \
    --benchmark humaneval \
    --n_samples 1 \
    --temperature 0.2
```

### Step 6: 交互式对比

```bash
cd compare_generate
python app.py
# 浏览器打开 http://localhost:7860
```

---

## 显存估算

| 组件 | 显存占用 |
|------|---------|
| 基础模型 (bf16) | ~3 GB |
| LoRA 适配器 | ~76 MB |
| vLLM KV Cache (n=4) | ~4-6 GB |
| 训练激活值 + 梯度 | ~4-6 GB |
| 预留 | ~2-4 GB |
| **总计** | **~14-18 GB** |

RTX 4090 24GB 显存足够，但如果训练时 OOM 可以：
- 降低 `rollout.gpu_memory_utilization` 从 0.75 → 0.65
- 降低 `data.train_batch_size` 从 32 → 16
- 降低 `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`

---

## 常见问题

### Q: flash-attn 编译失败怎么办？

flash-attn 不是必须的。编译失败不影响训练，只是推理速度会慢一些。跳过方法：

```bash
# 注释掉 requirements_autodl.txt 中的 flash-attn 行即可
# 或者在 install_autodl.sh 中忽略它的错误
```

### Q: vLLM 提示 CUDA 版本不匹配？

确保使用的是 vLLM >= 0.9.0（预编译 wheel 使用 CUDA 12.8）：

```bash
pip install "vllm>=0.9.0" --force-reinstall
```

### Q: 训练中途断掉怎么恢复？

VERL 的 checkpoint 保存在 `checkpoints/micro-grpo-*/` 目录中，重新运行 `run_training.sh` 时可以指定从某个 step 恢复（需查看 VERL 文档的 resume 功能）。

### Q: 沙箱用 subprocess 还是 firejail？

在 AutoDL 这种云环境下两者都可以：
- **Firejail**: 更严格的文件系统隔离 + 网络隔离
- **subprocess**: 轻量级，依赖系统自带权限

如果 Firejail 安装失败，代码会自动回退到 subprocess 模式。

### Q: VERL 训练报错 "xxx parameter not recognized"？

VERL 的 main 分支可能更新了配置参数名。如果训练启动时报告某个参数不识别：
1. 查看 [VERL 文档](https://verl.readthedocs.io/) 确认最新参数名
2. 修改 `run_training.sh` 中对应的配置键

---

## 文件清单

| 文件 | 用途 |
|------|------|
| `requirements_autodl.txt` | AutoDL 适配的依赖声明 |
| `install_autodl.sh` | 一键安装脚本 |
| `deploy_autodl.md` | 本部署文档 |

## 与原项目的差异

| 差异点 | 原项目 | AutoDL 适配 |
|--------|--------|------------|
| vLLM 版本 | 0.7.3 | >=0.9.0 |
| PyTorch | >=2.1.0 | >=2.7.0 |
| Firejail | pip 声明 | apt 安装 |
| 缺失文件 | utils/temperature.py | 需补充或移除引用 |
| 路径 | /home/featurize/code-r1/ | 需改为实际路径 |
