#!/bin/bash
set -e

unset PYTHONPATH
unset LD_PRELOAD
unset LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true
conda activate TAMoE

PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ_DIR"

DATASET="etth2"
GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Model
N_HEADS=4; D_MODEL=16; D_FF=128; N_LAYERS=3
PATCH_LEN=12; STRIDE=12; CONTEXT_POINTS=512

# Training
N_EPOCHS=100; BATCH_SIZE=128; LR=1e-3

# Task-Aware MoE
USE_ROUTED_EXPERT=1; USE_SHARED_EXPERT=1; USE_TASK_TOKEN=1
NUM_EXPERTS=8; MOE_TOP_K=2
MOE_AUX_LOSS_WEIGHT=0.0001; MOE_ROUTER_FUSION='concat'

# Alpha schedule
MOE_ALPHA_SCHEDULE='plateau'
MOE_ALPHA_START=0.1; MOE_ALPHA_END=0.15; MOE_ALPHA_PLATEAU_START=50
MOE_ROUTED_L2_WEIGHT=1e-3

# Task Token Generator
D_TASK=16; VWR_BETA=0.5
USE_CR=1; USE_VWR=1
USE_PERIODIC=1; USE_SPECTRAL=1; USE_STAT=1
USE_GLOBAL_DESC=1; USE_FINE_GRAINED_TASK_ID=0; USE_LEARNABLE_VWR=0

# Task probabilities
PROB_PM=0.16; PROB_MPM=0.16; PROB_RFM=0.18
PROB_SFM=0.18; PROB_DM=0.18; PROB_HM=0.14

MODEL_ID=1
MODEL_NAME="tamoe_pretrain"
mkdir -p "saved_models/${DATASET}/${MODEL_NAME}"
mkdir -p logs

python tamoe_multitask_pretrain.py \
    --dset "$DATASET" \
    --context_points $CONTEXT_POINTS \
    --patch_len $PATCH_LEN \
    --stride $STRIDE \
    --n_layers $N_LAYERS \
    --n_heads $N_HEADS \
    --d_model $D_MODEL \
    --d_ff $D_FF \
    --batch_size $BATCH_SIZE \
    --n_epochs_pretrain $N_EPOCHS \
    --lr $LR \
    --use_routed_expert $USE_ROUTED_EXPERT \
    --use_shared_expert $USE_SHARED_EXPERT \
    --use_task_token $USE_TASK_TOKEN \
    --num_experts $NUM_EXPERTS \
    --moe_top_k $MOE_TOP_K \
    --moe_aux_loss_weight $MOE_AUX_LOSS_WEIGHT \
    --moe_router_fusion_mode "$MOE_ROUTER_FUSION" \
    --moe_alpha_schedule "$MOE_ALPHA_SCHEDULE" \
    --moe_alpha_start $MOE_ALPHA_START \
    --moe_alpha_end $MOE_ALPHA_END \
    --moe_alpha_plateau_start $MOE_ALPHA_PLATEAU_START \
    --moe_routed_l2_weight $MOE_ROUTED_L2_WEIGHT \
    --d_task $D_TASK \
    --vwr_beta $VWR_BETA \
    --use_cr $USE_CR \
    --use_vwr $USE_VWR \
    --use_periodic $USE_PERIODIC \
    --use_spectral $USE_SPECTRAL \
    --use_stat $USE_STAT \
    --use_global_desc $USE_GLOBAL_DESC \
    --use_fine_grained_task_id $USE_FINE_GRAINED_TASK_ID \
    --use_learnable_vwr $USE_LEARNABLE_VWR \
    --pm_prob $PROB_PM \
    --mpm_prob $PROB_MPM \
    --rfm_prob $PROB_RFM \
    --sfm_prob $PROB_SFM \
    --dm_prob $PROB_DM \
    --hm_prob $PROB_HM \
    --model_type "$MODEL_NAME" \
    --pretrained_model_id $MODEL_ID \
    2>&1 | tee "logs/${MODEL_NAME}_${DATASET}.log"
