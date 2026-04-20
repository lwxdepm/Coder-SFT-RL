import json
import re
import argparse
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from rich.console import Console

console = Console()

# ============================================================
# Unicode Cleaning (保留原有的优秀实践)
# ============================================================
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")

def drop_surrogates_in_str(s: str) -> str:
    if not s: return s
    return _SURROGATE_RE.sub("", s)

def sanitize_obj(obj):
    if isinstance(obj, str): return drop_surrogates_in_str(obj)
    if isinstance(obj, list): return [sanitize_obj(x) for x in obj]
    if isinstance(obj, dict): return {sanitize_obj(k): sanitize_obj(v) for k, v in obj.items()}
    return obj

# ============================================================
# X-Coder SFT 数据集处理逻辑
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="针对 4090 + Qwen 1.5B 优化的 X-Coder 数据预处理脚本")
    parser.add_argument("--output_dir", default="data/sft-Xcoder")
    
    # 采样 50,000 条数据进行实验
    parser.add_argument("--max_samples", type=int, default=50000) 
    parser.add_argument("--system_prompt", type=str, default="You are a helpful and expert programming assistant.")
    
    # 针对 4090 优化的长度控制：
    # 35,000 字符约等于 8,000 - 10,000 Tokens，能保留高质量的长思维链 (Long CoT)
    parser.add_argument("--max_length", type=int, default=35000, help="总字符长度阈值，充分利用 4090 显存")
    
    # 过滤掉 response 过短的样本（确保推理过程的丰富度）
    parser.add_argument("--min_response_length", type=int, default=1000, help="最小回答长度，确保包含足够的推理链")
    
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    REPO_ID = "IIGroup/X-Coder-SFT-376k"
    CONFIG_NAME = "hybrid_376k"

    console.print(f"[cyan]正在从 Hugging Face 加载 {REPO_ID} ({CONFIG_NAME})...[/cyan]")
    
    # 加载数据集并打乱，确保 5 万条数据具有随机分布的难度
    try:
        ds = load_dataset(REPO_ID, CONFIG_NAME, split="train").shuffle(seed=42)
    except Exception as e:
        console.print(f"[red]加载失败，请检查网络或 HF_ENDPOINT 设置。错误: {e}[/red]")
        return

    formatted_samples = []
    skipped_empty = 0
    skipped_long = 0
    skipped_short_resp = 0

    for i, item in enumerate(tqdm(ds, desc="Processing X-Coder Samples")):
        if args.max_samples and len(formatted_samples) >= args.max_samples:
            break

        # X-Coder 数据集的原始字段为 query 和 response
        instruction = item.get("query", "")
        output = item.get("response", "")
        
        # 1. 基础过滤：剔除空数据
        if not instruction.strip() or not output.strip():
            skipped_empty += 1
            continue
            
        # 2. 长度过滤：防止太长导致 OOM
        total_char_len = len(instruction) + len(output)
        if total_char_len > args.max_length:
            skipped_long += 1
            continue
            
        # 3. 质量过滤：剔除缺乏思维链的数据
        if len(output) < args.min_response_length:
            skipped_short_resp += 1
            continue

        # 4. 组装成标准的 Messages 格式 (ChatML)
        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output}
        ]

        # 5. Unicode 清理并保存
        clean_sample = sanitize_obj({"messages": messages})
        formatted_samples.append(clean_sample)

    # 保存为 jsonl
    output_path = output_dir / "x_coder_sft_formatted.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for s in formatted_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    console.print(f"\n[green]处理完成！[/green]")
    console.print(f"成功提取: [bold]{len(formatted_samples)}[/bold] 条")
    console.print(f"跳过空数据: {skipped_empty} 条")
    console.print(f"因超长跳过: {skipped_long} 条 (阈值: {args.max_length} 字符)")
    console.print(f"因推理太短跳过: {skipped_short_resp} 条 (阈值: {args.min_response_length} 字符)")
    console.print(f"[cyan]数据已保存至: {output_path}[/cyan]")

if __name__ == "__main__":
    main()