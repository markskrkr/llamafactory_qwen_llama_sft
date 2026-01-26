"""
验证 Step 0 保存的 base_model 是否与原始模型一致

Step 0 保存的模型包含 LoRA 结构，需要提取 base_layer 权重
"""

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============ 配置 ============
SAVED_MODEL_PATH = "/mnt/workspace/gaoshenghao/LlamaFactory/base_model/model.safetensors"  # Step 0 保存的模型
ORIGINAL_MODEL_PATH = "/mnt/workspace/models/open_source/Qwen2-1.5B-Instruct"  # 原始模型
# ==============================

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
            new_state_dict[key] = value
    
    return new_state_dict


def main():
    print("=" * 60)
    print("验证 Step 0 保存的模型与原始模型是否一致")
    print("=" * 60)
    
    # Step 1: 加载 Step 0 保存的权重并提取 base_layer
    print("\n[Step 1] 加载 Step 0 保存的模型...")
    saved_state_dict = load_file(SAVED_MODEL_PATH)
    print(f"  - 原始 keys 数量: {len(saved_state_dict)}")
    
    # 检查是否包含 LoRA 结构
    has_lora = any('lora_A' in k or 'lora_B' in k for k in saved_state_dict.keys())
    has_base_layer = any('.base_layer.' in k for k in saved_state_dict.keys())
    print(f"  - 包含 LoRA 权重: {has_lora}")
    print(f"  - 包含 base_layer: {has_base_layer}")
    
    # 提取 base_layer 权重
    extracted_state_dict = extract_base_weights(saved_state_dict)
    print(f"  - 提取后 keys 数量: {len(extracted_state_dict)}")
    
    # Step 2: 加载原始模型
    print("\n[Step 2] 加载原始模型...")
    original_model = AutoModelForCausalLM.from_pretrained(
        ORIGINAL_MODEL_PATH,
        torch_dtype=torch.float32,  # 使用 float32 避免精度问题
        device_map="cpu"
    )
    original_state_dict = original_model.state_dict()
    print(f"  - 原始模型 keys 数量: {len(original_state_dict)}")
    
    # Step 3: 对比权重
    print("\n[Step 3] 对比权重...")
    
    # 检查 key 是否匹配
    extracted_keys = set(extracted_state_dict.keys())
    original_keys = set(original_state_dict.keys())
    
    missing_in_extracted = original_keys - extracted_keys
    extra_in_extracted = extracted_keys - original_keys
    
    if missing_in_extracted:
        print(f"  ⚠️  原始模型有但提取后缺失的 keys ({len(missing_in_extracted)}):")
        for k in list(missing_in_extracted)[:5]:
            print(f"      - {k}")
        if len(missing_in_extracted) > 5:
            print(f"      ... 还有 {len(missing_in_extracted) - 5} 个")
    
    if extra_in_extracted:
        print(f"  ⚠️  提取后多出的 keys ({len(extra_in_extracted)}):")
        for k in list(extra_in_extracted)[:5]:
            print(f"      - {k}")
    
    # 对比共同 keys 的权重值
    common_keys = extracted_keys & original_keys
    print(f"\n  - 共同 keys 数量: {len(common_keys)}")
    
    max_diff = 0.0
    diff_keys = []
    
    for key in common_keys:
        extracted_weight = extracted_state_dict[key].float()
        original_weight = original_state_dict[key].float()
        
        if extracted_weight.shape != original_weight.shape:
            print(f"  ❌ Shape 不匹配: {key}")
            print(f"      提取: {extracted_weight.shape}, 原始: {original_weight.shape}")
            continue
        
        diff = (extracted_weight - original_weight).abs().max().item()
        if diff > max_diff:
            max_diff = diff
        
        if diff > 1e-6:
            diff_keys.append((key, diff))
    
    print(f"\n  - 最大权重差异: {max_diff}")
    
    if diff_keys:
        print(f"\n  ⚠️  有差异的 keys ({len(diff_keys)}):")
        diff_keys.sort(key=lambda x: x[1], reverse=True)
        for k, d in diff_keys[:10]:
            print(f"      - {k}: {d}")
    
    # Step 4: 验证推理输出
    print("\n[Step 4] 验证推理输出...")
    tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL_PATH)
    
    # 创建一个新模型并加载提取的权重
    restored_model = AutoModelForCausalLM.from_pretrained(
        ORIGINAL_MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    restored_model.load_state_dict(extracted_state_dict, strict=False)
    
    # 测试输入
    test_input = "Hello, how are you?"
    inputs = tokenizer(test_input, return_tensors="pt")
    
    with torch.no_grad():
        original_output = original_model(**inputs).logits
        restored_output = restored_model(**inputs).logits
    
    logits_diff = (original_output - restored_output).abs().max().item()
    
    print(f"\n" + "=" * 60)
    print(f"  Logits 最大差异: {logits_diff}")
    print("=" * 60)
    
    if logits_diff < 1e-5:
        print("✅ 验证通过！Step 0 保存的模型与原始模型一致")
    elif logits_diff < 0.01:
        print("⚠️  有微小差异，可能是浮点精度问题")
    else:
        print("❌ 存在显著差异，请检查保存逻辑")


if __name__ == "__main__":
    main()

