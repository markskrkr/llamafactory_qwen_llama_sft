"""
Training script for Qwen 2.5 SFT on Commonsense dataset with SliceTrainer (Think-Touch Bandit).
"""

import os
import time
import json
import logging
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import argparse
from typing import Dict, Sequence
import copy

# Import SliceTrainer
from SliceTrainer import SliceTrainer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# =========================================================================
# Custom Dataset and Collator for Commonsense Data with Qwen 2.5 Template
# =========================================================================

class CommonsenseDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        log.info(f"Loading data from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        log.info(f"Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = self.data[i]
        
        # 构建符合 Qwen 2.5 格式的消息列表
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")
        
        if input_text:
            user_content = f"{instruction}\n{input_text}"
        else:
            user_content = instruction

        # 1. 构建完整的对话 (User + Assistant)
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output_text}
        ]
        
        # 2. 应用 Chat Template 获取完整文本 (不进行 tokenize)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # 3. Tokenize 完整文本
        tokenized = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        
        input_ids = tokenized.input_ids[0]
        attention_mask = tokenized.attention_mask[0]
        labels = input_ids.clone()

        # 4. Masking: 我们需要找到 User 部分的长度，将这部分的 Label 设为 -100
        # Qwen 2.5 的 template 通常会在 user 和 assistant 之间插入特定的 token
        # 最稳妥的方法是单独渲染 user prompt (带 generation prompt) 来计算长度
        user_messages = [{"role": "user", "content": user_content}]
        user_text = self.tokenizer.apply_chat_template(
            user_messages, 
            tokenize=False, 
            add_generation_prompt=True # 添加引导 assistant 回答的 token
        )
        
        user_tokens = self.tokenizer(user_text, return_tensors="pt").input_ids[0]
        user_len = len(user_tokens)

        # 确保 user_len 不超过 max_length，否则切片会报错
        safe_user_len = min(user_len, self.max_length)
        
        # 将 User 部分的 labels 设为 -100 (忽略 loss)
        labels[:safe_user_len] = -100
        
        # 将 Padding 部分的 labels 设为 -100
        if self.tokenizer.pad_token_id is not None:
            labels[input_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

class DataCollatorForSupervisedDataset(object):
    """Simple collator that stacks the pre-tokenized tensors"""
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, attention_mask = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "attention_mask"))
        
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        attention_mask = torch.stack(attention_mask)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

# =========================================================================
# Main Training Script
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser()

    # Model and data paths
    p.add_argument("--model_path", type=str, required=True, help="Path to Qwen2.5 model")
    p.add_argument("--dataset_path", type=str, required=True, help="Path to commonsense json file")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--max_length", type=int, default=2048)

    # Training hyperparameters
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_strategy", type=str, default="steps")
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--bf16", type=str2bool, default=True)
    p.add_argument("--gradient_checkpointing", type=str2bool, default=True)

    # SliceTrainer Hyperparameters
    p.add_argument("--move_steps", type=int, default=100)
    p.add_argument("--n_select", type=int, default=1)
    p.add_argument("--slice_train_bias", type=str2bool, default=True)
    p.add_argument("--epsilon", type=float, default=1e-5)
    p.add_argument("--w_max", type=float, default=10.0)
    p.add_argument("--delta", type=float, default=0.1)
    p.add_argument("--kappa", type=float, default=0.1)
    p.add_argument("--c_scale", type=float, default=0.1)

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    log.info(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "right" # Qwen2.5 padding side
    
    # Qwen 2.5 通常使用 eos_token 作为 pad_token，如果 pad_token 未定义
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    log.info(f"Building Commonsense dataset from {args.dataset_path}")
    # 这里使用我们新定义的 Dataset 类
    train_dataset = CommonsenseDataset(
        data_path=args.dataset_path,
        tokenizer=tokenizer,
        max_length=args.max_length
    )

    log.info(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        # device_map="auto" # 自动分配设备，或者移除由 Trainer 处理
    )
    
    # Qwen 配置调整
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.config.use_cache = False
    model.enable_input_require_grads()
    
    # 3. 再次确保 config 中的设置正确
    model.config.gradient_checkpointing = True 
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # 使用简单的 collator
    data_collator = DataCollatorForSupervisedDataset()

    # TrainingArguments for SliceTrainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="no",
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False, # 必须设为 False，否则 Dataset 里的 columns 会被过滤
        report_to=[],
    )

    global_start = time.time()

    log.info("Initializing SliceTrainer...")
    slice_trainer = SliceTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=None,
        compute_metrics=None,
        training_args=training_args,
        data_collator=data_collator,
        move_steps=args.move_steps,
        bias=args.slice_train_bias,
        verbose=True,
        n_select=args.n_select,
        epsilon=args.epsilon,
        w_max=args.w_max,
        delta=args.delta,
        kappa=args.kappa,
        c_scale=args.c_scale,
    )

    # Run training
    log.info("Starting SliceTrainer training on Commonsense data...")
    slice_trainer.run()

    total_time = time.time() - global_start
    log.info(f"Training completed in {total_time:.2f} seconds")

    # Save tokenizer and model
    tokenizer.save_pretrained(args.output_dir)
    slice_trainer.save_model(args.output_dir) # 确保保存最终模型权重

    log.info(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()