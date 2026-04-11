import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

def main():
    # ==========================================
    # 1. 路径与基础配置
    # ==========================================
    model_path = "/root/autodl-tmp/My_Xcoder/models/Qwen_models/Qwen2.5-Coder-1.5B"
    train_data_path = "/root/autodl-tmp/My_Xcoder/data/sft/train.jsonl"
    val_data_path = "/root/autodl-tmp/My_Xcoder/data/sft/val.jsonl"
    output_dir = "saves/qwen1.5b_native_full"

    print("🚀 正在加载 Tokenizer 与 Dataset...")
    
    # ==========================================
    # 2. 加载数据与 Tokenizer
    # ==========================================
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Qwen 等很多模型没有默认的 pad_token，通常用 eos_token 替代
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        "json", 
        data_files={"train": train_data_path, "test": val_data_path}
    )

    # ==========================================
    # 3. 加载模型 (注入 4090 性能黑科技)
    # ==========================================
    print("🧠 正在加载模型权重...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,               # ✅ 修复警告：将 torch_dtype 改为 dtype
        attn_implementation="sdpa",         # 使用 PyTorch 原生加速，无需配置 flash-attn
        device_map="cuda"                   # 自动映射到当前 GPU
    )

    # ==========================================
    # 4. 配置训练参数 (Training Arguments)
    # ==========================================
    # SFTConfig 继承自 TrainingArguments，专门针对微调优化
    training_args = SFTConfig(
        output_dir=output_dir,
        
        # 批次与显存控制 (2 * 8 = 16 的等效 Batch Size)
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,        # 极限省显存魔法
        
        # 学习率与步数
        learning_rate=2e-5,                 # 全量微调学习率要低
        lr_scheduler_type="cosine",
        warmup_steps=100,                   # ✅ 修复警告：将 warmup_ratio 改为具体的 warmup_steps
        num_train_epochs=3,                 # 训练 3 个 Epoch

        optim="adamw_8bit",                 # ✅ 8-bit AdamW 优化器：将优化器显存从 18GB 压到 4.5GB
        
        # 序列长度
        max_length=2048,                # ✅ 修正：trl 中指定输入截断长度的正确参数名是 max_seq_length
        
        # 评估与保存策略
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,                 # ✅ 磁盘防爆盾：永远只保留最新的 2 个存档，旧的会自动覆盖
        logging_steps=10,
        
        # 精度设置
        bf16=True,                          # 确认开启 bfloat16
    )

    # ==========================================
    # 5. 启动 SFTTrainer
    # ==========================================
    print("⚙️ 正在初始化 SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,         # ✅ 修复报错：新版 trl 中统一改名为了 processing_class
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
    )

    print("🔥 开始训练！")
    trainer.train()
    
    # 保存最终模型和 Tokenizer
    print(f"💾 训练完成，正在保存最终权重至 {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("🎉 All Done!")

if __name__ == "__main__":
    main()