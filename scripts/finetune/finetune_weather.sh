#!/bin/bash
# Finetune TAMoE on Weather dataset
set -e

PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ_DIR"

unset PYTHONPATH
unset LD_PRELOAD
unset LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true
conda activate TAMoE

# GPU
GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Dataset
DATASET="weather"
CONTEXT_POINTS=512

# Model architecture (must match pretrained model)
N_HEADS=16
D_MODEL=128
D_FF=512
N_LAYERS=3
PATCH_LEN=12
STRIDE=12

# Pretrained model
PRETRAIN="saved_models/${DATASET}/tamoe_pretrain/tamoe_multitask_cw${CONTEXT_POINTS}_patch${PATCH_LEN}_stride${STRIDE}_d${D_MODEL}_epochs100_all_moe8_tasktoken_model1.pth"

# Finetune settings
N_EPOCHS=30
BATCH_SIZE=128
FINETUNED_MODEL_ID=1
MODEL_TYPE="tamoe_finetune"

# MoE settings
USE_ROUTED_EXPERT=1
USE_SHARED_EXPERT=1
NUM_EXPERTS=8
MOE_TOP_K=2
AGGREGATION_MODE='shared_only'

mkdir -p logs

for PRED in 96 192 336 720; do
    LOG="logs/finetune_${DATASET}_cw${CONTEXT_POINTS}_t${PRED}_e${N_EPOCHS}.log"
    echo "[Finetune] $DATASET cw$CONTEXT_POINTS t$PRED"

    python tamoe_finetune.py \
        --dset_finetune $DATASET \
        --context_points $CONTEXT_POINTS \
        --target_points $PRED \
        --patch_len $PATCH_LEN \
        --stride $STRIDE \
        --n_layers $N_LAYERS \
        --n_heads $N_HEADS \
        --d_model $D_MODEL \
        --d_ff $D_FF \
        --batch_size $BATCH_SIZE \
        --is_finetune 1 \
        --pretrained_model "$PRETRAIN" \
        --n_epochs $N_EPOCHS \
        --use_routed_expert $USE_ROUTED_EXPERT \
        --use_shared_expert $USE_SHARED_EXPERT \
        --num_experts $NUM_EXPERTS \
        --moe_top_k $MOE_TOP_K \
        --aggregation_mode $AGGREGATION_MODE \
        --model_type $MODEL_TYPE \
        --finetuned_model_id $FINETUNED_MODEL_ID \
        --seed 2021 \
        > "$LOG" 2>&1

    grep "Test -> MSE" "$LOG" | tail -1
done

echo ""
echo "=== Finetune completed for $DATASET ==="
