#!/usr/bin/env python3
"""View Parquet file content"""

import pandas as pd
import argparse
from pathlib import Path

def view_parquet(parquet_file, num_samples=5):
    parquet_path = Path(parquet_file)
    if not parquet_path.exists():
        print(f"❌ File does not exist: {parquet_file}")
        return

    # 读取 Parquet 文件
    df = pd.read_parquet(parquet_file)

    # 基本信息
    print(f"✓ File: {parquet_file}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Total samples: {len(df)}\n")

    # 显示前几条记录
    print(f"Showing first {num_samples} samples:\n")
    print(df.head(num_samples))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View Parquet file content")
    parser.add_argument("parquet_file", type=str, help="Path to Parquet file")
    parser.add_argument("--num", type=int, default=5, help="Number of samples to show")
    args = parser.parse_args()

    view_parquet(args.parquet_file, args.num)