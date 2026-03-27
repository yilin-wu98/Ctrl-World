#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=6:00:00
#SBATCH --array=0-7

module load libffi
module load OpenSSL
module load cuda/12.6.0/cudnn/9.3

source /home/mila/a/arnav-kumar.jain/LLM/SAILOR-FM/.venv/bin/activate

export TORCH_HOME=$SCRATCH/SAILOR/.cache/torch
export XDG_CACHE_HOME=$SCRATCH/SAILOR/.cache
export HF_HOME=$SCRATCH/.cache/huggingface

# --- Config ---
NUM_CHUNKS=8
MAX_TRAJS=256
OUTPUT_DIR=$SCRATCH/SAILOR/Evals/DROID/ctrl_world_256

cd /home/mila/a/arnav-kumar.jain/LLM/Ctrl-World

echo "=== Ctrl-World Eval: chunk ${SLURM_ARRAY_TASK_ID}/${NUM_CHUNKS}, max_trajs=${MAX_TRAJS} ==="

python scripts/eval_wm_metrics.py \
    --svd_model_path "$SCRATCH/ctrl-world-models/svd_model_ckpt" \
    --clip_model_path "$SCRATCH/ctrl-world-models/clip_model_ckpt" \
    --ckpt_path "$SCRATCH/ctrl-world-models/ctrl_model_ckpt/checkpoint-10000.pt" \
    --val_dataset_dir "$SCRATCH/SAILOR/DROID/preprocessed_v2" \
    --split val \
    --output_dir "$OUTPUT_DIR" \
    --max_trajs "$MAX_TRAJS" \
    --val_chunk "$NUM_CHUNKS" \
    --val_chunk_id "$SLURM_ARRAY_TASK_ID" \
    --num_videos 4 \
    --skip_metrics \
    --metric_start_frame 8 \
    --metric_num_frames 50

echo "=== Chunk ${SLURM_ARRAY_TASK_ID} done ==="
