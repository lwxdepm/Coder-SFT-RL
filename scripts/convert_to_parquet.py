#!/usr/bin/env python3
"""Convert JSONL to Parquet for verl training"""

import json
import pandas as pd
from pathlib import Path

def convert_jsonl_to_parquet(jsonl_file, parquet_file):
    """Convert JSONL to Parquet format"""
    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                
                # 添加必需的字段
                if 'data_source' not in item:
                    item['data_source'] = item.get('source', 'mbpp')
                
                # 添加 reward_model 字段 (verl naive manager 需要)
                if 'reward_model' not in item:
                    item['reward_model'] = {
                        'ground_truth': item.get('solution', ''),
                        'input': item.get('prompt', '')
                    }
                
                data.append(item)
    
    df = pd.DataFrame(data)
    
    # 验证必需字段
    required = ['prompt', 'tests', 'data_source']
    for field in required:
        if field not in df.columns:
            raise ValueError(f"Dataset must have '{field}' column")
    
    # 创建目录
    Path(parquet_file).parent.mkdir(parents=True, exist_ok=True)
    
    # 保存
    df.to_parquet(parquet_file, engine='pyarrow', index=False)
    
    print(f"✓ Converted {jsonl_file} -> {parquet_file}")
    print(f"  Samples: {len(df)}")
    print(f"  Columns: {df.columns.tolist()}")
    
    # 显示样本
    if len(df) > 0:
        print(f"\nSample:")
        sample = df.iloc[0]
        print(f"  data_source: {sample['data_source']}")
        print(f"  has reward_model: {'reward_model' in sample}")
        print(f"  prompt: {sample['prompt'][:80]}...")

if __name__ == "__main__":
    convert_jsonl_to_parquet(
        'data/verified/train.jsonl',
        'data/processed/train.parquet'
    )
    
    convert_jsonl_to_parquet(
        'data/verified/val.jsonl',
        'data/processed/val.parquet'
    )
    
    print("\n✓ All data converted successfully!")
