import random
from pathlib import Path

def split_jsonl(
    input_file: str, 
    train_file: str, 
    val_file: str, 
    train_ratio: float = 0.95, 
    seed: int = 42
):
    """
    将 JSONL 文件按比例随机拆分为训练集和验证集。
    """
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"❌ 找不到输入文件: {input_path}")
        return

    # 1. 设定随机种子，保证每次运行划分结果一致（方便复现）
    random.seed(seed)

    print(f"📖 正在读取数据: {input_path} ...")
    with open(input_path, 'r', encoding='utf-8') as f:
        # readlines() 会保留每行末尾的换行符
        lines = f.readlines()

    total_samples = len(lines)
    print(f"✅ 共读取到 {total_samples} 条数据。")

    # 2. 全局打乱数据
    print("🔀 正在全局打乱数据...")
    random.shuffle(lines)

    # 3. 计算切分索引
    split_idx = int(total_samples * train_ratio)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    # 4. 写入训练集
    print(f"💾 正在写入训练集 ({len(train_lines)} 条, {train_ratio*100:.0f}%) -> {train_file}")
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)

    # 5. 写入验证集
    print(f"💾 正在写入验证集 ({len(val_lines)} 条, {(1-train_ratio)*100:.0f}%) -> {val_file}")
    with open(val_file, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)

    print("🎉 数据切分完成！")

if __name__ == "__main__":
    # 路径配置（请根据你的实际文件路径进行修改）
    INPUT_PATH = "sft/kodcode_sft_formatted.jsonl"
    TRAIN_PATH = "sft/train.jsonl"
    VAL_PATH = "sft/val.jsonl"
    
    split_jsonl(
        input_file=INPUT_PATH, 
        train_file=TRAIN_PATH, 
        val_file=VAL_PATH, 
        train_ratio=0.95
    )