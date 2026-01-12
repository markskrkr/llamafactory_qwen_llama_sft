#!/bin/bash
# LLaMA-3.2-3B Full Fine-tuning on MetaMathQA
# 8x H800 GPUs with DeepSpeed ZeRO-3 + CPU Offload

set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=1

MODEL_PATH=meta-llama/Llama-3.2-3B-Instruct
DATASET=metamathqa
OUTPUT_DIR=saves/llama3.2-3b/full/metamathqa

FORCE_TORCHRUN=1 llamafactory-cli train \
    --model_name_or_path ${MODEL_PATH} \
    --trust_remote_code true \
    --stage sft \
    --do_train true \
    --finetuning_type full \
    --deepspeed examples/deepspeed/ds_z3_offload_config.json \
    --dataset ${DATASET} \
    --template llama3 \
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
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --num_train_epochs 3.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 true \
    --ddp_timeout 180000000 \
    --gradient_checkpointing true

