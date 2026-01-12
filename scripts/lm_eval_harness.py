#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 lm-evaluation-harness 评估模型在常见基准测试上的表现。

支持的任务包括:
- ARC (arc_easy, arc_challenge)
- HellaSwag
- PIQA
- BoolQ
- WinoGrande
- OBQA (openbookqa)
- SciQ

使用方法:
    # 评估 HuggingFace 模型
    python scripts/lm_eval_harness.py \
        --model_path meta-llama/Meta-Llama-3-8B-Instruct \
        --tasks arc_easy,arc_challenge,hellaswag,piqa,boolq,winogrande,openbookqa \
        --output_dir results/llama3-8b-eval

    # 评估带有 LoRA adapter 的模型
    python scripts/lm_eval_harness.py \
        --model_path meta-llama/Meta-Llama-3-8B-Instruct \
        --adapter_path saves/llama3-8b/lora/sft \
        --tasks arc_easy,hellaswag \
        --output_dir results/llama3-8b-lora-eval

    # 使用 vLLM 加速推理
    python scripts/lm_eval_harness.py \
        --model_path meta-llama/Meta-Llama-3-8B-Instruct \
        --tasks arc_easy,hellaswag \
        --backend vllm \
        --output_dir results/llama3-8b-vllm-eval
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用 lm-evaluation-harness 评估语言模型"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="模型路径或 HuggingFace 模型名称",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="LoRA adapter 路径 (可选)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="arc_easy,arc_challenge,hellaswag,piqa,boolq,winogrande,openbookqa",
        help="评估任务，用逗号分隔",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=0,
        help="few-shot 样本数量 (默认: 0)",
    )
    parser.add_argument(
        "--batch_size",
        type=str,
        default="auto",
        help="批次大小，可以是数字或 'auto'",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["hf", "vllm"],
        default="hf",
        help="推理后端: hf (HuggingFace) 或 vllm",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="设备 (默认: cuda:0)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/lm_eval",
        help="结果输出目录",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="信任远程代码",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="模型数据类型",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制每个任务的样本数量 (用于测试)",
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        help="记录模型输出样本",
    )
    return parser.parse_args()


def build_command(args):
    """构建 lm_eval 命令"""
    cmd = ["lm_eval", "--model", args.backend]
    
    # 构建 model_args
    model_args = [f"pretrained={args.model_path}"]
    
    if args.adapter_path:
        model_args.append(f"peft={args.adapter_path}")
    
    if args.trust_remote_code:
        model_args.append("trust_remote_code=True")
    
    if args.dtype != "auto":
        model_args.append(f"dtype={args.dtype}")
    
    cmd.extend(["--model_args", ",".join(model_args)])
    cmd.extend(["--tasks", args.tasks])
    cmd.extend(["--num_fewshot", str(args.num_fewshot)])
    cmd.extend(["--batch_size", str(args.batch_size)])
    
    if args.backend == "hf":
        cmd.extend(["--device", args.device])
    
    cmd.extend(["--output_path", args.output_dir])
    
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    
    if args.log_samples:
        cmd.append("--log_samples")
    
    return cmd


def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 构建并执行命令
    cmd = build_command(args)
    print(f"执行命令: {' '.join(cmd)}")
    print("-" * 50)
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"评估失败，返回码: {result.returncode}")
        sys.exit(result.returncode)
    
    print("-" * 50)
    print(f"评估完成！结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()

