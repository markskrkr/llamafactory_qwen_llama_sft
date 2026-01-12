本目录包含为多种模型、算法和数据集组合生成的监督微调(SFT)训练脚本。

## 目录结构

```
scripts/sft/
├── qwen2.5_7b/                    # Qwen2.5-7B 模型
│   ├── full/                      # 全量微调 (lr=1e-5)
│   │   ├── commonsense/train.sh   # Commonsense_170k 数据集
│   │   ├── code/train.sh          # Magicoder 数据集
│   │   └── math/train.sh          # MetaMathQA 数据集
│   ├── lora/                      # 普通LoRA (rank=32, lr=1e-4)
│   │   ├── commonsense/train.sh
│   │   ├── code/train.sh
│   │   └── math/train.sh
│   ├── randlora/                  # RandLoRA (rank=64, lr=1e-4)
│   │   ├── commonsense/train.sh
│   │   ├── code/train.sh
│   │   └── math/train.sh
│   └── vblora_dora/               # VB-LoRA (rank=128, lr=1e-4)
│       ├── commonsense/train.sh
│       ├── code/train.sh
│       └── math/train.sh
└── llama3.2_3b/                   # LLaMA-3.2-3B 模型
    ├── full/                      # 全量微调 (lr=1e-5)
    │   ├── commonsense/train.sh
    │   ├── code/train.sh
    │   └── math/train.sh
    ├── lora/                      # 普通LoRA (rank=32, lr=1e-4)
    │   ├── commonsense/train.sh
    │   ├── code/train.sh
    │   └── math/train.sh
    ├── randlora/                  # RandLoRA (rank=64, lr=1e-4)
    │   ├── commonsense/train.sh
    │   ├── code/train.sh
    │   └── math/train.sh
    └── vblora_dora/               # VB-LoRA (rank=128, lr=1e-4)
        ├── commonsense/train.sh
        ├── code/train.sh
        └── math/train.sh
```

## 数据集与测试集对应关系

| 数据集 | 训练数据 | 测试集 |
|--------|----------|--------|
| commonsense | `data/commonsense_170k.json` | BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA |
| code | `ise-uiuc/Magicoder-Evol-Instruct-110K` | HumanEval, MBPP |
| math | `meta-math/MetaMathQA` | GSM8K, MATH |

## 微调算法参数配置

| 算法 | rank | 学习率 | 特殊参数 |
|------|------|--------|----------|
| Full FT | - | 1e-5 | DeepSpeed ZeRO-3 + CPU Offload |
| LoRA | 32 | 1e-4 | - |
| RandLoRA | 64 | 1e-4 | `--use_randlora true` |
| VB-LoRA | 128 | 1e-4 | `--use_vblora true` |

## 使用方法

bash scripts/sft/qwen2.5_7b/lora/commonsense/train.sh

# 示例：运行LLaMA-3.2-3B在math数据集上的全量微调
bash scripts/sft/llama3.2_3b/full/math/train.sh
```

## 硬件配置

- 8x H800 GPUs (80GB 显存/卡)
- 全量微调使用 DeepSpeed ZeRO-3 + CPU Offload 进行显存优化
- LoRA系列方法使用标准DDP训练

## 输出目录结构

```
saves/
├── qwen2.5-7b/
│   ├── full/
│   │   ├── commonsense_170k/
│   │   ├── magicoder/
│   │   └── metamathqa/
│   ├── lora/
│   ├── randlora/
│   └── vblora/
└── llama3.2-3b/
    ├── full/
    ├── lora/
    ├── randlora/
    └── vblora/
```

