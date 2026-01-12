#!/bin/bash
# Qwen2.5-7B Full Fine-tuning on Magicoder-Evol-Instruct-110K
# 8x H800 GPUs with DeepSpeed ZeRO-3 + CPU Offload

set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=1

MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
DATASET=magicoder_evol_instruct
OUTPUT_DIR=saves/qwen2.5-7b/full/magicoder

FORCE_TORCHRUN=1 llamafactory-cli train \
    --model_name_or_path ${MODEL_PATH} \
    --trust_remote_code true \
    --stage sft \
    --do_train true \
    --finetuning_type full \
    --deepspeed examples/deepspeed/ds_z3_offload_config.json \
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
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --num_train_epochs 3.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 true \
    --ddp_timeout 180000000 \
    --gradient_checkpointing true

