#!/usr/bin/env python3
"""
将 JSONL/JSON 文件转换为 VERL 训练格式
支持多种输入格式，自动适配字段
"""
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd


def load_data(input_path: str) -> List[Dict]:
    """
    加载 JSON 或 JSONL 文件
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"文件不存在: {input_path}")
    
    data = []
    
    if input_path.suffix == '.jsonl':
        # JSONL 格式
        with open(input_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"警告：第 {i+1} 行解析失败: {e}")
                    continue
    
    elif input_path.suffix == '.json':
        # JSON 格式
        with open(input_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
            
            # 如果是列表，直接使用
            if isinstance(loaded, list):
                data = loaded
            # 如果是字典，尝试提取数据
            elif isinstance(loaded, dict):
                # 常见的键名
                for key in ['data', 'items', 'examples', 'train', 'test']:
                    if key in loaded and isinstance(loaded[key], list):
                        data = loaded[key]
                        break
                
                # 如果没找到，把字典本身当作单个样本
                if not data:
                    data = [loaded]
    
    elif input_path.suffix == '.parquet':
        # Parquet 格式
        df = pd.read_parquet(input_path)
        data = df.to_dict('records')
    
    else:
        raise ValueError(f"不支持的文件格式: {input_path.suffix}")
    
    print(f"成功加载 {len(data)} 条数据")
    return data


import numpy as np

def extract_field(item: Dict, field_names: List[str], default=None):
    for name in field_names:
        if name in item:
            value = item[name]

            if value is None:
                continue

            # numpy array
            if isinstance(value, np.ndarray):
                if value.size > 0:
                    return value
                continue

            # list
            if isinstance(value, list):
                if len(value) > 0:
                    return value
                continue

            # string
            if isinstance(value, str):
                if value.strip() != "":
                    return value
                continue

            # 其他类型直接返回
            return value

    return default


def make_minimal_prompt(prompt: str) -> str:
    """
    创建最小化的 prompt（可以自定义格式）
    """
    # 如果 prompt 已经很完整，直接返回
    if isinstance(prompt, str):
        return prompt.strip()
    
    # 如果是字典，尝试提取
    if isinstance(prompt, dict):
        # 尝试常见的键名
        for key in ['text', 'content', 'description', 'problem', 'question']:
            if key in prompt:
                return str(prompt[key]).strip()
        
        # 否则转为字符串
        return str(prompt)
    
    return str(prompt)


def convert_to_verl_format(
    item: Dict,
    idx: int,
    data_source: str = "custom",
    ability: str = "code_generation",
    split: str = "train",
) -> Dict:
    """
    将单个数据项转换为 VERL 格式
    
    Args:
        item: 原始数据项
        idx: 索引
        data_source: 数据来源
        ability: 能力类型
        split: 数据集划分
    
    Returns:
        转换后的数据
    """
    # 提取 prompt（支持多种字段名）
    prompt = extract_field(
        item,
        ['prompt', 'problem', 'problem_statement', 'description', 'question', 'input'],
        default=""
    )
    
    # 提取 solution（支持多种字段名）
    solution = extract_field(
        item,
        ['solution', 'code', 'answer', 'response', 'completion', 'output'],
        default=""
    )
    
    # 提取 tests（支持多种字段名）
    tests = extract_field(
        item,
        ['tests', 'test_cases', 'test', 'assertions', 'checks'],
        default=[]
    )
    
   
    # 构建 VERL 格式
    verl_item = {
        "prompt": prompt,
        "data_source": data_source,
        "ability": ability,
        "reward_model": {
            "style": "code_execution",
            "ground_truth": tests,  # 测试用例作为 ground_truth
        },
        "extra_info": {
            "index": idx,
            "split": split,
            "tests": tests,  # 同时保存在 extra_info
            "solution": solution,  # solution 放在 extra_info
        },
    }
    
    # 如果有其他字段，也保存到 extra_info
    extra_fields = set(item.keys()) - {'prompt', 'solution', 'tests', 'problem', 'code', 'answer'}
    for field in extra_fields:
        verl_item["extra_info"][field] = item[field]
    
    return verl_item


def convert_dataset(
    input_path: str,
    output_path: str,
    data_source: str = "custom",
    ability: str = "code_generation",
    split: str = "train",
    output_format: str = "parquet",
):
    """
    转换整个数据集
    
    Args:
        input_path: 输入文件路径 (JSON/JSONL/Parquet)
        output_path: 输出文件路径
        data_source: 数据来源标识
        ability: 能力类型
        split: 数据集划分
        output_format: 输出格式 ('parquet', 'jsonl', 'json')
    """
    print("=" * 80)
    print("JSONL/JSON 到 VERL 格式转换")
    print("=" * 80)
    
    # 加载数据
    print(f"\n📂 加载数据: {input_path}")
    data = load_data(input_path)
    
    if not data:
        print("❌ 没有加载到数据！")
        return
    
    # 转换
    print(f"\n🔄 开始转换...")
    converted = []
    
    for idx, item in enumerate(data):
        try:
            verl_item = convert_to_verl_format(
                item,
                idx=idx,
                data_source=data_source,
                ability=ability,
                split=split
            )
            converted.append(verl_item)
            
            # 打印前几个样本
            if idx < 3:
                print(f"\n样本 {idx+1}:")
                print(f"  Prompt: {verl_item['prompt'][:100]}...")
                print(f"  Tests: {len(verl_item['reward_model']['ground_truth'])} 个")
                print(f"  Solution: {'有' if verl_item['extra_info']['solution'] else '无'}")
        
        except Exception as e:
            print(f"⚠️  样本 {idx} 转换失败: {e}")
            continue
    
    print(f"\n✅ 成功转换 {len(converted)}/{len(data)} 条数据")
    
    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 保存到: {output_path}")
    
    if output_format == 'parquet' or output_path.suffix == '.parquet':
        # 保存为 Parquet
        df = pd.DataFrame(converted)
        df.to_parquet(output_path, index=False)
    
    elif output_format == 'jsonl' or output_path.suffix == '.jsonl':
        # 保存为 JSONL
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in converted:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    elif output_format == 'json' or output_path.suffix == '.json':
        # 保存为 JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(converted, f, ensure_ascii=False, indent=2)
    
    else:
        raise ValueError(f"不支持的输出格式: {output_format}")
    
    print("\n" + "=" * 80)
    print("✅ 转换完成！")
    print("=" * 80)
    
    # 统计信息
    print(f"\n📊 统计信息:")
    print(f"  总样本数: {len(converted)}")
    
    # 统计有 solution 的数量
    with_solution = sum(1 for item in converted if item['extra_info']['solution'])
    print(f"  有 solution: {with_solution} ({with_solution/len(converted)*100:.1f}%)")
    
    # 统计 tests 数量
    avg_tests = sum(len(item['reward_model']['ground_truth']) for item in converted) / len(converted)
    print(f"  平均测试数: {avg_tests:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="将 JSON/JSONL 文件转换为 VERL 训练格式"
    )
    
    parser.add_argument(
        '--input',
        type=str,
        help='输入文件路径 (JSON/JSONL/Parquet)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='输出文件路径'
    )
    
    parser.add_argument(
        '--data-source',
        type=str,
        default='custom',
        help='数据来源标识 (default: custom)'
    )
    
    parser.add_argument(
        '--ability',
        type=str,
        default='code_generation',
        help='能力类型 (default: code_generation)'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val', 'test'],
        help='数据集划分 (default: train)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        default='parquet',
        choices=['parquet', 'jsonl', 'json'],
        help='输出格式 (default: parquet)'
    )
    
    args = parser.parse_args()
    
    convert_dataset(
        input_path=args.input,
        output_path=args.output,
        data_source=args.data_source,
        ability=args.ability,
        split=args.split,
        output_format=args.format
    )


if __name__ == '__main__':
    main()