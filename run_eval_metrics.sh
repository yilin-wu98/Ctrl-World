#!/bin/bash
# Ctrl-World evaluation: generate videos and compute FID/FVD/LPIPS metrics
#
# Usage:
#   # Trajectory replay mode (autoregressive rollout per trajectory)
#   bash run_eval_metrics.sh
#
#   # Dataloader mode (single-step predictions via DROID Dataset_mix)
#   bash run_eval_metrics.sh --use_dataloader --num_samples 256
#
#   # Metrics only from saved views
#   bash run_eval_metrics.sh --compute_metrics_only
#
#   # Chunked distributed eval
#   VAL_CHUNK=4 VAL_CHUNK_ID=0 bash run_eval_metrics.sh

# Model paths (adjust to your setup)
SVD_MODEL_PATH=${SVD_MODEL_PATH:-"$SCRATCH/ctrl-world-models/svd_model_ckpt"}
CLIP_MODEL_PATH=${CLIP_MODEL_PATH:-"$SCRATCH/ctrl-world-models/clip_model_ckpt"}
CKPT_PATH=${CKPT_PATH:-"$SCRATCH/ctrl-world-models/ctrl_model_ckpt/checkpoint-10000.pt"}

# Dataset (for trajectory replay mode)
VAL_DATASET_DIR=${VAL_DATASET_DIR:-"$SCRATCH/SAILOR/DROID/world-model-droid-eval-new"}
SPLIT=${SPLIT:-"val"}

# Dataset (for dataloader mode - uses dataset_root_path + dataset_names from config)
DATASET_ROOT_PATH=${DATASET_ROOT_PATH:-""}
DATASET_META_INFO_PATH=${DATASET_META_INFO_PATH:-""}
DATASET_NAMES=${DATASET_NAMES:-""}

# Output
OUTPUT_DIR=${OUTPUT_DIR:-"$SCRATCH/SAILOR/Evals/DROID"}

# Chunking (set VAL_CHUNK and VAL_CHUNK_ID for distributed eval)
CHUNK_ARGS=""
if [ -n "$VAL_CHUNK" ] && [ -n "$VAL_CHUNK_ID" ]; then
    CHUNK_ARGS="--val_chunk $VAL_CHUNK --val_chunk_id $VAL_CHUNK_ID"
fi

# Dataloader mode args
DL_ARGS=""
if [ -n "$DATASET_ROOT_PATH" ]; then
    DL_ARGS="$DL_ARGS --dataset_root_path $DATASET_ROOT_PATH"
fi
if [ -n "$DATASET_META_INFO_PATH" ]; then
    DL_ARGS="$DL_ARGS --dataset_meta_info_path $DATASET_META_INFO_PATH"
fi
if [ -n "$DATASET_NAMES" ]; then
    DL_ARGS="$DL_ARGS --dataset_names $DATASET_NAMES"
fi

python scripts/eval_wm_metrics.py \
    --svd_model_path "$SVD_MODEL_PATH" \
    --clip_model_path "$CLIP_MODEL_PATH" \
    --ckpt_path "$CKPT_PATH" \
    --val_dataset_dir "$VAL_DATASET_DIR" \
    --split "$SPLIT" \
    --output_dir "$OUTPUT_DIR" \
    --task_type replay_val_videos \
    --num_videos 16 \
    --video_fps 4 \
    $CHUNK_ARGS \
    $DL_ARGS \
    "$@"
