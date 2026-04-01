#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=3:00:00
#SBATCH --array=0-3

module load libffi
module load OpenSSL
module load cuda/12.6.0/cudnn/9.3

source /home/mila/a/arnav-kumar.jain/LLM/SAILOR-FM/.venv/bin/activate

export TORCH_HOME=$SCRATCH/SAILOR/.cache/torch
export XDG_CACHE_HOME=$SCRATCH/SAILOR/.cache
export HF_HOME=$SCRATCH/.cache/huggingface

VIEWS_DIR=$SCRATCH/SAILOR/Evals/DROID/ctrl_world_256_start8/views
OUT_DIR=$SCRATCH/SAILOR/Evals/DROID/ctrl_world_256_start8
MAX_TRAJS=256

cd /home/mila/a/arnav-kumar.jain/LLM/Ctrl-World

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    echo "FVD + FID for 16 frames"
    python scripts/compute_metrics.py \
        --views_dir "$VIEWS_DIR" --start_frame 0 --num_frames 16 \
        --max_trajs "$MAX_TRAJS" --skip_lpips \
        --output "$OUT_DIR/metrics_fvd_fid_16f.json"

elif [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    echo "FVD + FID for 32 frames"
    python scripts/compute_metrics.py \
        --views_dir "$VIEWS_DIR" --start_frame 0 --num_frames 32 \
        --max_trajs "$MAX_TRAJS" --skip_lpips \
        --output "$OUT_DIR/metrics_fvd_fid_32f.json"

elif [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
    echo "FVD + FID for 50 frames"
    python scripts/compute_metrics.py \
        --views_dir "$VIEWS_DIR" --start_frame 0 --num_frames 50 \
        --max_trajs "$MAX_TRAJS" --skip_lpips \
        --output "$OUT_DIR/metrics_fvd_fid_50f.json"

elif [ "$SLURM_ARRAY_TASK_ID" -eq 3 ]; then
    echo "FID for all frames"
    python scripts/compute_metrics.py \
        --views_dir "$VIEWS_DIR" --start_frame 0 \
        --max_trajs "$MAX_TRAJS" --skip_fvd --skip_lpips \
        --output "$OUT_DIR/metrics_fid_all.json"
fi

echo "Task $SLURM_ARRAY_TASK_ID done!"
