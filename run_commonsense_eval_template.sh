#!/bin/bash
# Evaluate SliceFine checkpoints on all commonsense datasets
# Parallel evaluation on 2 GPUs (2, 3)

# Datasets
DATASETS=("boolq" "piqa" "social_i_qa" "hellaswag" "winogrande" "ARC-Challenge" "ARC-Easy" "openbookqa")

# Base models
BASE_MODEL_1_5B="/mnt/workspace/models/open_source/Qwen2.5-1.5B"
BASE_MODEL_7B="/mnt/workspace/models/open_source/Qwen2.5-7B"

# Output log directory
LOG_DIR="eval_logs_template"
mkdir -p $LOG_DIR

echo "=========================================="
echo "Starting Commonsense Evaluation (2 GPUs: 2, 3)"
echo "=========================================="

# Build task list: (checkpoint, dataset, base_model, log_name)
# Format: "ckpt_path|dataset|base_model|log_name|batch_size"
TASKS=()

# Base 1.5B
for ds in "${DATASETS[@]}"; do
    TASKS+=("NONE|$ds|$BASE_MODEL_1_5B|base_1.5b_${ds}|64")
done

# 1.5B checkpoints
for ckpt in "outputs/qwen2_5_1.5b_commonsense_slice_n64_m20" "outputs/qwen2_5_1.5b_commonsense_slice_n128_m20" "outputs/qwen2_5_1.5b_commonsense_slice_n256_m20"; do
    # Find latest checkpoint in directory
    latest_ckpt=$(ls -d ${ckpt}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    if [ -z "$latest_ckpt" ]; then
        latest_ckpt="$ckpt"
    fi
    ckpt_name=$(basename "$ckpt")_$(basename "$latest_ckpt")
    for ds in "${DATASETS[@]}"; do
        TASKS+=("$latest_ckpt|$ds|$BASE_MODEL_1_5B|${ckpt_name}_${ds}|64")
    done
done

# Base 7B
for ds in "${DATASETS[@]}"; do
    TASKS+=("NONE|$ds|$BASE_MODEL_7B|base_7b_${ds}|16")
done

# 7B checkpoints
for ckpt in "outputs/qwen2_5_7b_commonsense_slice_n64_m10" "outputs/qwen2_5_7b_commonsense_slice_n64_m20" "outputs/qwen2_5_7b_commonsense_slice_n128_m20" "outputs/qwen2_5_7b_commonsense_slice_n256_m20"; do
    # Find latest checkpoint in directory
    latest_ckpt=$(ls -d ${ckpt}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    if [ -z "$latest_ckpt" ]; then
        latest_ckpt="$ckpt"
    fi
    ckpt_name=$(basename "$ckpt")_$(basename "$latest_ckpt")
    for ds in "${DATASETS[@]}"; do
        TASKS+=("$latest_ckpt|$ds|$BASE_MODEL_7B|${ckpt_name}_${ds}|16")
    done
done

# Function to run evaluation on a specific GPU
run_eval() {
    local gpu=$1
    local ckpt=$2
    local ds=$3
    local base_model=$4
    local log_name=$5
    local batch_size=$6

    # Handle NONE as empty checkpoint path
    if [ "$ckpt" == "NONE" ]; then
        ckpt_path=""
    else
        ckpt_path="$ckpt"
    fi

    echo "[GPU $gpu] Running: $log_name (batch_size=$batch_size)"
    CUDA_VISIBLE_DEVICES=$gpu python eval_qwen_commonsense_slice_template.py \
        --dataset "$ds" \
        --base_model "$base_model" \
        --checkpoint_path "$ckpt_path" \
        --batch_size "$batch_size" \
        > "$LOG_DIR/${log_name}.log" 2>&1
    echo "[GPU $gpu] Done: $log_name"
}

# Run tasks in parallel on 2 GPUs
NUM_GPUS=2
task_idx=0
total_tasks=${#TASKS[@]}

while [ $task_idx -lt $total_tasks ]; do
    pids=()

    # Launch up to NUM_GPUS tasks in parallel
    for gpu in 2 3; do
        if [ $task_idx -lt $total_tasks ]; then
            task="${TASKS[$task_idx]}"
            ckpt=$(echo "$task" | cut -d'|' -f1)
            ds=$(echo "$task" | cut -d'|' -f2)
            base_model=$(echo "$task" | cut -d'|' -f3)
            log_name=$(echo "$task" | cut -d'|' -f4)
            batch_size=$(echo "$task" | cut -d'|' -f5)

            run_eval $gpu "$ckpt" "$ds" "$base_model" "$log_name" "$batch_size" &
            pids+=($!)
            ((task_idx++))
        fi
    done

    # Wait for all parallel tasks to complete
    for pid in "${pids[@]}"; do
        wait $pid
    done

    echo "Progress: $task_idx / $total_tasks tasks completed"
done

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "Logs saved to: $LOG_DIR/"
echo "=========================================="

# Extract final accuracy from all logs
echo ""
echo ">>> Summary of Results:"
echo "Model,Dataset,Accuracy" > "$LOG_DIR/summary.csv"
for log in "$LOG_DIR"/*.log; do
    name=$(basename "$log" .log)
    acc=$(grep "Final accuracy" "$log" | tail -1 | grep -oP '[0-9]+\.[0-9]+$' || echo "N/A")
    echo "$name,$acc"
    echo "$name,$acc" >> "$LOG_DIR/summary.csv"
done

