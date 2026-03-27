#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=1:00:00
#SBATCH --partition=main

module load libffi
module load OpenSSL
module load cuda/12.6.0/cudnn/9.3

source /home/mila/a/arnav-kumar.jain/LLM/SAILOR-FM/.venv/bin/activate

export TORCH_HOME=$SCRATCH/SAILOR/.cache/torch
export XDG_CACHE_HOME=$SCRATCH/SAILOR/.cache
export HF_HOME=$SCRATCH/.cache/huggingface

OUTPUT_DIR=$SCRATCH/SAILOR/Evals/DROID/ctrl_world_256

cd /home/mila/a/arnav-kumar.jain/LLM/Ctrl-World

echo "=== Computing metrics on saved views (frames [8:58]) ==="

python scripts/eval_wm_metrics.py \
    --output_dir "$OUTPUT_DIR" \
    --compute_metrics_only \
    --metric_start_frame 8 \
    --metric_num_frames 50

echo "=== Metrics done ==="
