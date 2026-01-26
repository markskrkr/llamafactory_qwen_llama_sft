"""
从包含 LoRA 结构的 checkpoint 中提取 base_layer 权重，
转换为标准 HuggingFace 模型格式，可直接用于推理验证。

用法:
    python extract_base_model.py \
        --checkpoint_path /path/to/checkpoint/base_model \
        --original_model_path /path/to/original/model \
        --output_path /path/to/output
"""

import argparse
import os
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import shutil


def extract_base_weights(state_dict):
    """
    从包含 LoRA 结构的 state_dict 中提取 base_layer 权重
    
    转换规则:
    - 'model.layers.0.mlp.gate_proj.base_layer.weight' -> 'model.layers.0.mlp.gate_proj.weight'
    - 跳过所有 lora_A, lora_B 权重
    """
    new_state_dict = {}
    
    for key, value in state_dict.items():
        # 跳过 LoRA 权重
        if 'lora_A' in key or 'lora_B' in key:
            continue
        
        # 提取 base_layer 权重并重命名
        if '.base_layer.' in key:
            new_key = key.replace('.base_layer.', '.')
            new_state_dict[new_key] = value
        else:
            # 非 LoRA 层直接保留
            new_state_dict[new_key] = value
    
    return new_state_dict


def main():
    parser = argparse.ArgumentParser(description="Extract base model from LoRA checkpoint")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the checkpoint directory containing model.safetensors")
    parser.add_argument("--original_model_path", type=str, required=True,
                        help="Path to the original model (for config and tokenizer)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path for the extracted model")
    args = parser.parse_args()
    
    print("=" * 60)
    print("提取 Base Model 从 LoRA Checkpoint")
    print("=" * 60)
    
    # Step 1: 加载 checkpoint 权重
    checkpoint_file = os.path.join(args.checkpoint_path, "model.safetensors")
    if not os.path.exists(checkpoint_file):
        checkpoint_file = args.checkpoint_path  # 可能直接传入了文件路径
    
    print(f"\n[Step 1] 加载 checkpoint: {checkpoint_file}")
    saved_state_dict = load_file(checkpoint_file)
    print(f"  - 原始 keys 数量: {len(saved_state_dict)}")
    
    # 检查是否包含 LoRA 结构
    has_lora = any('lora_A' in k or 'lora_B' in k for k in saved_state_dict.keys())
    has_base_layer = any('.base_layer.' in k for k in saved_state_dict.keys())
    print(f"  - 包含 LoRA 权重: {has_lora}")
    print(f"  - 包含 base_layer: {has_base_layer}")
    
    # Step 2: 提取 base_layer 权重
    print(f"\n[Step 2] 提取 base_layer 权重...")
    extracted_state_dict = extract_base_weights(saved_state_dict)
    print(f"  - 提取后 keys 数量: {len(extracted_state_dict)}")
    
    # Step 3: 加载原始模型获取缺失的权重 (如 lm_head)
    print(f"\n[Step 3] 从原始模型补充缺失权重...")
    original_model = AutoModelForCausalLM.from_pretrained(
        args.original_model_path,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    original_state_dict = original_model.state_dict()
    
    # 找出缺失的 keys
    extracted_keys = set(extracted_state_dict.keys())
    original_keys = set(original_state_dict.keys())
    missing_keys = original_keys - extracted_keys
    
    if missing_keys:
        print(f"  - 补充缺失的 keys ({len(missing_keys)}):")
        for key in missing_keys:
            extracted_state_dict[key] = original_state_dict[key]
            print(f"      + {key}")
    
    # Step 4: 创建输出目录并保存
    print(f"\n[Step 4] 保存提取后的模型到: {args.output_path}")
    os.makedirs(args.output_path, exist_ok=True)
    
    # 保存权重
    output_file = os.path.join(args.output_path, "model.safetensors")
    save_file(extracted_state_dict, output_file)
    print(f"  - 权重已保存: {output_file}")
    
    # 复制 config 和 tokenizer
    config = AutoConfig.from_pretrained(args.original_model_path)
    config.save_pretrained(args.output_path)
    print(f"  - Config 已保存")
    
    tokenizer = AutoTokenizer.from_pretrained(args.original_model_path)
    tokenizer.save_pretrained(args.output_path)
    print(f"  - Tokenizer 已保存")
    
    # Step 5: 验证
    print(f"\n[Step 5] 验证提取的模型...")
    restored_model = AutoModelForCausalLM.from_pretrained(
        args.output_path,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    test_input = "Hello, how are you?"
    inputs = tokenizer(test_input, return_tensors="pt")
    
    with torch.no_grad():
        original_output = original_model(**inputs).logits
        restored_output = restored_model(**inputs).logits
    
    logits_diff = (original_output - restored_output).abs().max().item()
    
    print(f"\n" + "=" * 60)
    print(f"  与原始模型的 Logits 最大差异: {logits_diff}")
    print("=" * 60)
    
    if logits_diff < 1e-5:
        print("✅ 提取成功！模型可用于推理验证")
    else:
        print("⚠️  存在差异，请检查")
    
    print(f"\n输出模型路径: {args.output_path}")
    print("可以使用以下方式加载:")
    print(f'  model = AutoModelForCausalLM.from_pretrained("{args.output_path}")')


if __name__ == "__main__":
    main()

