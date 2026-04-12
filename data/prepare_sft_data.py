import json
import re
import argparse
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from rich.console import Console

console = Console()

# ============================================================
# Unicode Cleaning (保留你之前的优秀实践)
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
# SFT 数据集处理逻辑
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/sft")
    # 建议先采样少量数据跑通流程，比如 10000 条
    parser.add_argument("--max_samples", type=int, default=50000) 
    parser.add_argument("--system_prompt", type=str, default="You are a helpful and expert programming assistant.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print("[cyan]正在从 Hugging Face 加载 KodCode-V1-SFT-R1...[/cyan]")
    
    # 此时会走你刚才设置的 HF_ENDPOINT 镜像
    # 修改后的代码
    console.print("[cyan]正在打乱数据集以进行全局随机采样...[/cyan]")
    
    # 设定 seed 保证每次跑出来的随机子集是一样的，方便实验复现
    ds = load_dataset("KodCode/KodCode-V1-SFT-R1", split="train").shuffle(seed=42)
    
    formatted_samples = []
    skipped = 0

    for i, item in enumerate(tqdm(ds, desc="Processing SFT Samples")):
        if args.max_samples and len(formatted_samples) >= args.max_samples:
            break

        # 1. 字段适配：不同数据集的键名可能不同，做个 fallback
        instruction = item.get("instruction") or item.get("prompt") or item.get("question", "")
        output = item.get("output") or item.get("solution") or item.get("answer", "")
        
        # 2. 基础过滤：剔除空数据
        if not instruction.strip() or not output.strip():
            skipped += 1
            continue
            
        # 3. 长度控制：
        # 否则稍微长一点的上下文就会触发 OOM
        if len(instruction) + len(output) > 3000:
            skipped += 1
            continue

        # 4. 组装成标准的 Messages 格式
        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output}
        ]

        # 5. Unicode 清理
        clean_sample = sanitize_obj({"messages": messages})
        formatted_samples.append(clean_sample)

    # 保存为 jsonl
    output_path = output_dir / "kodcode_sft_formatted.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for s in formatted_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    console.print(f"\n[green]处理完成！成功提取 {len(formatted_samples)} 条，跳过 {skipped} 条。[/green]")
    console.print(f"[cyan]数据已保存至: {output_path}[/cyan]")

if __name__ == "__main__":
    main()