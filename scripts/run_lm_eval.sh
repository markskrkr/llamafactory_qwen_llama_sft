#!/bin/bash
# 使用 lm-evaluation-harness 评估模型的脚本
# 
# 使用方法:
#   bash scripts/run_lm_eval.sh <model_path> [adapter_path] [output_dir]
#
# 示例:
#   # 评估基础模型
#   bash scripts/run_lm_eval.sh meta-llama/Meta-Llama-3-8B-Instruct
#
#   # 评估带 LoRA 的模型
#   bash scripts/run_lm_eval.sh meta-llama/Meta-Llama-3-8B-Instruct saves/llama3-8b/lora/sft

set -e

MODEL_PATH=${1:-"meta-llama/Meta-Llama-3-8B-Instruct"}
ADAPTER_PATH=${2:-""}
OUTPUT_DIR=${3:-"results/lm_eval_$(date +%Y%m%d_%H%M%S)"}

# 评估任务列表
TASKS="arc_easy,arc_challenge,hellaswag,piqa,boolq,winogrande,openbookqa"

echo "=========================================="
echo "LM Evaluation Harness 评估"
echo "=========================================="
echo "模型路径: $MODEL_PATH"
echo "Adapter路径: ${ADAPTER_PATH:-无}"
echo "输出目录: $OUTPUT_DIR"
echo "评估任务: $TASKS"
echo "=========================================="

# 构建命令
if [ -z "$ADAPTER_PATH" ]; then
    python scripts/lm_eval_harness.py \
        --model_path "$MODEL_PATH" \
        --tasks "$TASKS" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size auto \
        --trust_remote_code \
        --log_samples
else
    python scripts/lm_eval_harness.py \
        --model_path "$MODEL_PATH" \
        --adapter_path "$ADAPTER_PATH" \
        --tasks "$TASKS" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size auto \
        --trust_remote_code \
        --log_samples
fi

echo "=========================================="
echo "评估完成！结果保存在: $OUTPUT_DIR"
echo "=========================================="

