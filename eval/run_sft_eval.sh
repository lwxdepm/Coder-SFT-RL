#!/bin/bash

# ==========================================
# 1. 路径配置 (请根据实际情况确认)
# ==========================================
BASE_MODEL="/root/autodl-tmp/My_Xcoder/models/Qwen_models/Qwen2.5-Coder-1.5B"
SFT_MODEL="/root/autodl-tmp/My_Xcoder/sft/saves/qwen1.5b_merged_2"
RESULT_LOG="eval_results_summary.md"

# 采样参数 (为防止显存溢出，K 从 20 降到了 5)
K_SAMPLES=5
TEMP=0.8
TOP_P=0.95

# 初始化日志文件
echo "# 🚀 模型性能测评报告" > $RESULT_LOG
echo "测评时间: $(date)" >> $RESULT_LOG
echo -e "\n| 模型 | 数据集 | 评价模式 | Base 得分 | Plus 得分 | 状态 |" >> $RESULT_LOG
echo "| :--- | :--- | :--- | :--- | :--- | :--- |" >> $RESULT_LOG

# ==========================================
# 核心修复: 全新升级的分数提取函数
# ==========================================
extract_score() {
    local text="$1"
    local mode="$2"
    
    if [ "$mode" == "base" ]; then
        # 匹配 "(base tests)" 所在行及下一行，筛选出包含 "pass@" 的那一行，提取数字
        echo "$text" | grep -A 1 "(base tests)" | grep -i "pass@" | head -n 1 | awk '{print $2}'
    elif [ "$mode" == "plus" ]; then
        # 匹配 "(base + extra tests)" 所在行及下一行
        echo "$text" | grep -A 1 "(base + extra tests)" | grep -i "pass@" | head -n 1 | awk '{print $2}'
    fi
}

# ==========================================
# 2. 执行测评循环
# ==========================================
for MODEL_PATH in $BASE_MODEL $SFT_MODEL; do
    MODEL_NAME=$(basename $MODEL_PATH)
    
    for DATASET in humaneval mbpp; do
        
        # --- A. 测评 Pass@1 (Greedy Decoding) ---
        echo "正在执行: $MODEL_NAME | $DATASET | Greedy (Pass@1)..."
        
        # 注意：这里直接调用的 evalplus.evaluate。如果你用 uv 跑，请自行加上 uv run
        output=$(evalplus.evaluate --dataset $DATASET --model $MODEL_PATH --backend vllm --greedy --tp 1 2>&1)
        
        if [ $? -eq 0 ]; then
            score_base=$(extract_score "$output" "base")
            score_plus=$(extract_score "$output" "plus")
            
            score_base=${score_base:-"N/A"}
            score_plus=${score_plus:-"N/A"}
            
            echo "| $MODEL_NAME | $DATASET | Greedy (Pass@1) | $score_base | $score_plus | ✅ 成功 |" >> $RESULT_LOG
        else
            echo "❌ 失败: $MODEL_NAME $DATASET Greedy"
            echo "| $MODEL_NAME | $DATASET | Greedy (Pass@1) | Error | Error | ❌ 崩溃 |" >> $RESULT_LOG
        fi

        # --- B. 测评 Pass@k (Sampling) ---
        echo "正在执行: $MODEL_NAME | $DATASET | Sampling (Pass@$K_SAMPLES)..."
        
        output_k=$(evalplus.evaluate --dataset $DATASET --model $MODEL_PATH --backend vllm \
                    --n_samples $K_SAMPLES --top_p $TOP_P --temperature $TEMP --tp 1 2>&1)
        
        if [ $? -eq 0 ]; then
            score_base_k=$(extract_score "$output_k" "base")
            score_plus_k=$(extract_score "$output_k" "plus")
            
            score_base_k=${score_base_k:-"N/A"}
            score_plus_k=${score_plus_k:-"N/A"}
            
            echo "| $MODEL_NAME | $DATASET | Pass@$K_SAMPLES | $score_base_k | $score_plus_k | ✅ 成功 |" >> $RESULT_LOG
        else
            echo "❌ 失败: $MODEL_NAME $DATASET Sampling"
            echo "| $MODEL_NAME | $DATASET | Pass@$K_SAMPLES | Error | Error | ❌ 崩溃 |" >> $RESULT_LOG
        fi

    done
done

echo -e "\n🎉 所有测评已完成！结果已记录至 $RESULT_LOG"