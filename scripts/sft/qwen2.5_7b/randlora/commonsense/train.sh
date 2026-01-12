#!/bin/bash
# Qwen2.5-7B RandLoRA Fine-tuning on Commonsense_170k
# 8x H800 GPUs, rank=64, lr=1e-4

set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
DATASET=commonsense_170k
OUTPUT_DIR=saves/qwen2.5-7b/randlora/commonsense_170k

FORCE_TORCHRUN=1 llamafactory-cli train \
    --model_name_or_path ${MODEL_PATH} \
    --trust_remote_code true \
    --stage sft \
    --do_train true \
    --finetuning_type lora \
    --lora_rank 64 \
    --lora_target all \
    --use_randlora true \
    --dataset ${DATASET} \
    --template qwen2.5 \
    --cutoff_len 2048 \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 4 \
    --output_dir ${OUTPUT_DIR} \
    --logging_steps 10 \
    --save_steps 500 \
    --plot_loss true \
    --overwrite_output_dir true \
    --save_only_model false \
    --report_to none \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --num_train_epochs 3.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 true \
    --ddp_timeout 180000000

