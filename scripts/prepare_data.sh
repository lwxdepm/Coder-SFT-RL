#!/bin/bash
# ============================================================
# Data Preparation Pipeline
# ============================================================

set -e

echo "========================================"
echo "  Code-RL Data Preparation"
echo "========================================"

# Step 1: Download and preprocess
echo "[1/3] Loading datasets..."
python data/prepare_dataset.py \
    --sources rlvr \
    --max_per_source 2000 \
    --output_dir data/raw \
    --seed 42

# Step 2: Verify with reference solutions
echo "[2/3] Verifying dataset quality..."
python data/verify_dataset.py \
    --input data/raw/train.jsonl \
    --output data/verified/train.jsonl \
    --n_runs 3 \
    --n_workers 16 \
    --sandbox subprocess \
    --filter_status pass no_solution \
    --analyze_failures

python data/verify_dataset.py \
    --input data/raw/val.jsonl \
    --output data/verified/val.jsonl \
    --n_runs 3 \
    --n_workers 16 \
    --sandbox subprocess \
    --filter_status pass no_solution

# Step 3: Create final processed split
echo "[3/3] Creating final dataset..."
python data/prepare_dataset.py \
    --sources kodcode mbpp humaneval \
    --max_per_source 2000 \
    --output_dir data/processed \
    --verify \
    --seed 42

echo "========================================"
echo "  Data preparation complete!"
echo "  Train: data/processed/train.jsonl"
echo "  Val:   data/processed/val.jsonl"
echo "========================================"