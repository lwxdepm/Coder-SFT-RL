import os
import torch
import matplotlib.pyplot as plt  # ✨ 新增：用于绘图
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType

def main():
    # ==========================================
    # 1. 路径与基础配置
    # ==========================================
    model_path = "/root/autodl-tmp/My_Xcoder/models/Qwen_models/Qwen2.5-Coder-1.5B"
    train_data_path = "/root/autodl-tmp/My_Xcoder/data/sft-kodcode/train.jsonl"
    val_data_path = "/root/autodl-tmp/My_Xcoder/data/sft-kodcode/val.jsonl"
    output_dir = "saves/qwen1.5b_lora_with_loss"
    plot_path = os.path.join(output_dir, "loss_curve.png") # Loss 图保存路径

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    print("🚀 正在加载 Tokenizer 与 Dataset...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files={"train": train_data_path, "test": val_data_path})

    # ==========================================
    # 2. 加载模型
    # ==========================================
    print("🧠 正在加载模型权重...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda"
    )

    # ==========================================
    # 3. LoRA 配置
    # ==========================================
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # ==========================================
    # 4. 配置训练参数
    # ==========================================
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,               # 🔄 修改：放弃绝对步数(100)，改用 10% 比例更科学
        num_train_epochs=1,             # ⏬ 核心降低：4.7万条纯算法数据看1遍足够，看2遍绝对过拟合！
        optim="adamw_8bit",
        max_length=2048,
        
        # 记录与保存设置
        logging_steps=10,             
        eval_strategy="steps",
        eval_steps=100,                 # 整个训练总计约 742 步，每 100 步评估刚好能打出 7 个点，曲线漂亮
        save_strategy="steps",
        save_steps=200,                 # 🔽 降低：多存几个档（大概能存 3-4 个 checkpoints）
        save_total_limit=3,             # 🔼 提升：多保留一个存档点，方便我们后续挑表现最好的回合
        bf16=True,
        report_to="none",
    )

    # ==========================================
    # 5. 启动 SFTTrainer
    # ==========================================
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
        peft_config=peft_config,
    )

    print("🔥 开始训练...")
    train_result = trainer.train()

    # ==========================================
    # 6. 提取日志并绘制 Loss 曲线
    # ==========================================
    print("📊 正在绘制 Loss 曲线...")
    history = trainer.state.log_history
    
    train_loss = [log["loss"] for log in history if "loss" in log]
    train_steps = [log["step"] for log in history if "loss" in log]
    
    eval_loss = [log["eval_loss"] for log in history if "eval_loss" in log]
    eval_steps = [log["step"] for log in history if "eval_loss" in log]

    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_loss, label="Train Loss", color="blue", alpha=0.6)
    if eval_loss:
        plt.plot(eval_steps, eval_loss, label="Eval Loss", color="red", marker='o')
    
    plt.title("Training and Evaluation Loss Curve")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 保存图片
    plt.savefig(plot_path)
    print(f"📈 Loss 曲线已保存至: {plot_path}")

    # 保存最终结果
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("🎉 任务圆满完成！")

if __name__ == "__main__":
    main()