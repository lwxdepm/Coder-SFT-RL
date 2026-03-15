"""
Export VERL checkpoint to HuggingFace format
"""

import argparse
import os
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig
from safetensors.torch import load_file
import shutil
import json


def find_latest_checkpoint(checkpoint_dir):
    """找到最新的 checkpoint"""
    checkpoint_dir = Path(checkpoint_dir)
    
    # 先检查是否有 latest_checkpointed_iteration.txt
    latest_file = checkpoint_dir / "latest_checkpointed_iteration.txt"
    if latest_file.exists():
        with open(latest_file, 'r') as f:
            latest_step = int(f.read().strip())
        checkpoint_path = checkpoint_dir / f"global_step_{latest_step}"
        if checkpoint_path.exists():
            print(f"Found latest checkpoint from file: step {latest_step}")
            return checkpoint_path
    
    # 否则查找所有 checkpoint 文件夹
    checkpoints = []
    for item in checkpoint_dir.iterdir():
        if item.is_dir() and item.name.startswith("global_step_"):
            try:
                step = int(item.name.split("_")[-1])
                if (item / "actor").exists():
                    checkpoints.append((step, item))
            except ValueError:
                continue
    
    if not checkpoints:
        raise ValueError(f"No valid checkpoints found in {checkpoint_dir}")
    
    # 返回最新的 checkpoint
    checkpoints.sort(key=lambda x: x[0])
    latest_step, latest_path = checkpoints[-1]
    print(f"Found latest checkpoint: step {latest_step}")
    return latest_path


def load_verl_lora_weights(actor_path):
    """加载 VERL 保存的 LoRA 权重"""
    lora_adapter_path = actor_path / "lora_adapter"
    
    if not lora_adapter_path.exists():
        raise ValueError(f"LoRA adapter not found at {lora_adapter_path}")
    
    # 加载 adapter config
    with open(lora_adapter_path / "adapter_config.json", 'r') as f:
        adapter_config = json.load(f)
    
    print(f"LoRA config: {adapter_config}")
    
    # 加载 adapter 权重
    adapter_weights_path = lora_adapter_path / "adapter_model.safetensors"
    if adapter_weights_path.exists():
        print(f"Loading LoRA weights from safetensors...")
        lora_state_dict = load_file(str(adapter_weights_path))
    else:
        # 尝试 .bin 格式
        adapter_weights_path = lora_adapter_path / "adapter_model.bin"
        if adapter_weights_path.exists():
            print(f"Loading LoRA weights from .bin file...")
            lora_state_dict = torch.load(adapter_weights_path, map_location="cpu")
        else:
            raise ValueError(f"No adapter weights found in {lora_adapter_path}")
    
    return adapter_config, lora_state_dict


def export_lora_model(checkpoint_path, base_model_path, output_dir, merge=True):
    """导出 LoRA 模型
    
    Args:
        checkpoint_path: VERL checkpoint 路径
        base_model_path: 基础模型路径
        output_dir: 输出目录
        merge: 是否合并 LoRA 权重到基础模型
    """
    actor_path = checkpoint_path / "actor"
    
    if not actor_path.exists():
        raise ValueError(f"Actor model not found at {actor_path}")
    
    print(f"\n{'='*60}")
    print(f"Loading base model from: {base_model_path}")
    print(f"Loading LoRA from: {actor_path}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    # 加载 tokenizer（从 checkpoint 或 base model）
    hf_path = actor_path / "huggingface"
    if hf_path.exists():
        print("Loading tokenizer from checkpoint...")
        tokenizer = AutoTokenizer.from_pretrained(hf_path)
    else:
        print("Loading tokenizer from base model...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # 加载基础模型
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 加载 LoRA adapter
    lora_adapter_path = actor_path / "lora_adapter"
    
    if lora_adapter_path.exists():
        print(f"Loading LoRA adapter...")
        
        try:
            # 使用 PEFT 加载
            model = PeftModel.from_pretrained(
                base_model,
                str(lora_adapter_path),
                is_trainable=False,
            )
            print("✓ Successfully loaded LoRA adapter")
            
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            if merge:
                print("\nMerging LoRA weights into base model...")
                model = model.merge_and_unload()
                print("✓ Weights merged")
                
                # 保存合并后的模型
                print(f"Saving merged model to {output_dir}...")
                model.save_pretrained(output_dir, safe_serialization=True)
                tokenizer.save_pretrained(output_dir)
                print("✓ Merged model saved")
                
            else:
                # 只保存 base model + LoRA adapter
                print(f"Saving base model to {output_dir}...")
                base_model.save_pretrained(output_dir, safe_serialization=True)
                tokenizer.save_pretrained(output_dir)
                
                # 复制 LoRA adapter
                lora_output = Path(output_dir) / "lora_adapter"
                print(f"Copying LoRA adapter to {lora_output}...")
                shutil.copytree(lora_adapter_path, lora_output, dirs_exist_ok=True)
                print("✓ LoRA adapter copied")
            
            # 同时保存一份独立的 LoRA adapter
            lora_only_dir = f"{output_dir}_lora_only"
            print(f"\nSaving standalone LoRA adapter to {lora_only_dir}...")
            shutil.copytree(lora_adapter_path, lora_only_dir, dirs_exist_ok=True)
            print("✓ Standalone LoRA adapter saved")
            
        except Exception as e:
            print(f"✗ Error loading LoRA adapter: {e}")
            print("Saving base model only...")
            os.makedirs(output_dir, exist_ok=True)
            base_model.save_pretrained(output_dir, safe_serialization=True)
            tokenizer.save_pretrained(output_dir)
    
    else:
        print("Warning: No LoRA adapter found, saving base model only")
        os.makedirs(output_dir, exist_ok=True)
        base_model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
    
    return output_dir


def test_model(model_path, test_prompt=None):
    """测试导出的模型"""
    print(f"\n{'='*60}")
    print("Testing exported model...")
    print(f"{'='*60}\n")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    if test_prompt is None:
        test_prompt = "def fibonacci(n):"
    
    print(f"Test prompt: {test_prompt}")
    
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated:\n{generated_text}\n")


def main():
    parser = argparse.ArgumentParser(description="Export VERL checkpoint to HuggingFace format")
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        required=True, 
        help="Directory containing checkpoints (e.g., checkpoints/micro-grpo-20260315-095557)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Output directory for exported model"
    )
    parser.add_argument(
        "--base_model", 
        type=str, 
        required=True, 
        help="Base model path (e.g., Qwen/Qwen2.5-Coder-0.5B-Instruct)"
    )
    parser.add_argument(
        "--checkpoint_step", 
        type=int, 
        default=None, 
        help="Specific checkpoint step (default: latest)"
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Don't merge LoRA weights, keep as adapter"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test the exported model"
    )
    parser.add_argument(
        "--test_prompt",
        type=str,
        default=None,
        help="Custom test prompt"
    )
    
    args = parser.parse_args()
    
    # 找到 checkpoint
    if args.checkpoint_step is not None:
        checkpoint_path = Path(args.checkpoint_dir) / f"global_step_{args.checkpoint_step}"
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")
        print(f"Using specified checkpoint: step {args.checkpoint_step}")
    else:
        checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
    
    # 导出模型
    output_dir = export_lora_model(
        checkpoint_path, 
        args.base_model, 
        args.output_dir,
        merge=not args.no_merge
    )
    
    print(f"\n{'='*60}")
    print("✓ Export completed successfully!")
    print(f"{'='*60}")
    if args.no_merge:
        print(f"Base model + LoRA adapter: {output_dir}")
        print(f"Standalone LoRA adapter: {output_dir}_lora_only")
    else:
        print(f"Merged model: {output_dir}")
        print(f"Standalone LoRA adapter: {output_dir}_lora_only")
    print(f"{'='*60}\n")
    
    # 测试模型
    if args.test:
        test_model(output_dir, args.test_prompt)
    
    print("\nUsage examples:")
    print(f"  # Load merged model:")
    print(f"  from transformers import AutoModelForCausalLM")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{output_dir}')")
    print(f"\n  # Load LoRA adapter:")
    print(f"  from peft import PeftModel")
    print(f"  base = AutoModelForCausalLM.from_pretrained('{args.base_model}')")
    print(f"  model = PeftModel.from_pretrained(base, '{output_dir}_lora_only')")


if __name__ == "__main__":
    main()