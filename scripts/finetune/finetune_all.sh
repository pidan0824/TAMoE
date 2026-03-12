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

GPU_ID=${1:-0}
TARGET_POINTS=${2:-720}
export CUDA_VISIBLE_DEVICES=$GPU_ID

IS_FINETUNE=1
IS_LINEAR_PROBE=0
N_EPOCHS=30
FINETUNED_MODEL_ID=1

USE_ROUTED_EXPERT=1
USE_SHARED_EXPERT=1
NUM_EXPERTS=8
MOE_TOP_K=2
AGGREGATION_MODE='shared_only'

declare -A DATASET_D_MODEL
DATASET_D_MODEL["etth1"]=16
DATASET_D_MODEL["etth2"]=16
DATASET_D_MODEL["ettm1"]=128
DATASET_D_MODEL["ettm2"]=128
DATASET_D_MODEL["weather"]=128
DATASET_D_MODEL["electricity"]=128
DATASET_D_MODEL["traffic"]=128

DATASETS=("etth1" "etth2" "ettm1" "ettm2" "weather" "electricity" "traffic")

build_pretrained_model_path() {
    local dataset=$1
    local d_model=$2
    echo "saved_models/${dataset}/tamoe_pretrain/tamoe_multitask_cw512_patch12_stride12_d${d_model}_epochs100_all_moe8_tasktoken_model1.pth"
}

echo "Finetune: target_points=$TARGET_POINTS, epochs=$N_EPOCHS, GPU=$GPU_ID"
echo ""

TOTAL=${#DATASETS[@]}
SUCCESSFUL=0
FAILED=0
SKIPPED=0

for dataset in "${DATASETS[@]}"; do
    d_model=${DATASET_D_MODEL[$dataset]}
    pretrained_model=$(build_pretrained_model_path "$dataset" "$d_model")

    echo "[$((SUCCESSFUL+FAILED+SKIPPED+1))/$TOTAL] $dataset (d_model=$d_model)"

    if [ ! -f "$pretrained_model" ]; then
        echo "  Skipped: model not found"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    if python tamoe_finetune.py \
        --dset_finetune "$dataset" \
        --is_finetune $IS_FINETUNE \
        --pretrained_model "$pretrained_model" \
        --is_linear_probe $IS_LINEAR_PROBE \
        --target_points $TARGET_POINTS \
        --finetuned_model_id $FINETUNED_MODEL_ID \
        --n_epochs $N_EPOCHS \
        --use_routed_expert $USE_ROUTED_EXPERT \
        --use_shared_expert $USE_SHARED_EXPERT \
        --num_experts $NUM_EXPERTS \
        --moe_top_k $MOE_TOP_K \
        --aggregation_mode $AGGREGATION_MODE \
        2>&1 | tee "logs/finetune_${dataset}_t${TARGET_POINTS}.log"; then
        echo "  Success"
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        echo "  Failed"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

echo "Summary: $SUCCESSFUL success, $FAILED failed, $SKIPPED skipped"
[ $FAILED -eq 0 ] && exit 0 || exit 1
