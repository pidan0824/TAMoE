# TAMoE: A Task-Adaptive Mixture-of-Experts Method

How to Pre-train Time-Series Models with Multiple Pretext Tasks:
A Task-Adaptive Mixture-of-Experts Method
---

## Key Components

### Multiple Pretext Tasks Pre-training

Six self-supervised reconstruction tasks:

| Task | Target | Masking Strategy |
|------|--------|------------------|
| **PM** | Masked patches | Random patch masking |
| **MPM** | Masked patches | Structured block masking |
| **RFM** | Full sequence | Random frequency masking |
| **SFM** | Full sequence | Structured frequency band masking |
| **DM** | Masked patches | Decomposition-based (trend/residual) |
| **HM** | Full sequence | Holistic reconstruction |

### Task Token 

Generates task-adaptive routing signals:
- **CR (Contextual Representation)**: Extracts task-relevant context from backbone hidden states via cross-attention
- **VWR (Variable-Wise Representation)**: Extracts per-variable features (periodic/spectral/statistical) from input with gated fusion

### MoE Architecture

```
y = y_shared + alpha * y_routed
```

- Shared expert + routed experts
- Task token guides expert selection
- `alpha` scheduled during pre-training
- Fine-tuning: `aggregation_mode=shared_only`

---

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Data

Download from [Autoformer](https://github.com/thuml/Autoformer), place in `./dataset/`.

---

## Pre-trained Checkpoints

Location: `saved_models/checkpoint/<dataset>/<scheme>/`

Files per checkpoint:
- `<model>.pth` — backbone
- `<model>.json` — config
- `<model>_task_token.pth` — task token

Config: patch_len=12, stride=12, context=512, 100 epochs, 8 experts.

### Fine-tuning

```bash
python tamoe_finetune.py \
    --dset_finetune etth1 \
    --context_points 512 \
    --target_points 96 \
    --patch_len 12 --stride 12 \
    --pretrained_model saved_models/checkpoint/etth1/schemeA/<model> \
    --is_finetune 1 \
    --aggregation_mode shared_only
```

---

## Training from Scratch

### Pre-training

```bash
python tamoe_multitask_pretrain.py \
    --dset etth1 \
    --context_points 512 \
    --patch_len 12 --stride 12 \
    --d_model 16 --d_ff 128 --n_heads 4 --n_layers 3 \
    --n_epochs_pretrain 100 \
    --use_routed_expert 1 --use_shared_expert 1 --use_task_token 1 \
    --num_experts 8 --moe_top_k 2
```

### Fine-tuning

```bash
# Full fine-tuning
python tamoe_finetune.py \
    --dset_finetune etth1 \
    --pretrained_model <path> \
    --is_finetune 1 \
    --n_epochs 30

# Linear probing
python tamoe_finetune.py \
    --dset_finetune etth1 \
    --pretrained_model <path> \
    --is_linear_probe 1 \
    --n_epochs 30
```

---

## Requirements

- Python 3.9+
- PyTorch 2.0.0+cu118
- CUDA 11.8
