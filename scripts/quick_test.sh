#!/bin/bash

BASE_MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct"
TRAINED_MODEL="$1"

if [ -z "$TRAINED_MODEL" ]; then
    echo "Usage: $0 <trained_model_path>"
    echo "Example: $0 models/my-trained-model"
    exit 1
fi

echo "========================================"
echo "  Model Comparison Test"
echo "========================================"
echo "Base Model:    $BASE_MODEL"
echo "Trained Model: $TRAINED_MODEL"
echo "========================================"

python scripts/test_model.py \
    --compare \
    --base_model "$BASE_MODEL" \
    --trained_model "$TRAINED_MODEL" \
    --num_samples 2 \
    --temperature 0.7 \
    --max_new_tokens 256