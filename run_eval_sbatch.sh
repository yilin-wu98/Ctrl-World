#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=64GB
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=18:59:00
#SBATCH --tmp=2T
#module load httpproxy
#module load cuda/12.6
#module load cudnn/9
#
cd ~/Ctrl-World
source .venv/bin/activate

export TORCH_HOME=$SCRATCH/SAILOR/.cache/torch
export XDG_CACHE_HOME=$SCRATCH/SAILOR/.cache
export HF_HOME=$SCRATCH/.cache/huggingface

VAL_CHUNK_ID=${1:-0}
echo "Starting eval script (val_chunk_id=$VAL_CHUNK_ID)"

python scripts/rollout_replay_traj_preprocessed_val_videos.py \
    --svd_model_path $SCRATCH/ctrl-world-models/svd_model_ckpt \
        --clip_model_path $SCRATCH/ctrl-world-models/clip_model_ckpt \
	    --ckpt_path $SCRATCH/ctrl-world-models/ctrl_model_ckpt/checkpoint-10000.pt \
	        --task_type replay_eval \
		    --split val \
		        --val_dataset_dir $SCRATCH/SAILOR/DROID/world-model-droid-eval-new \
			    --val_chunk 7 \
			        --val_chunk_id $VAL_CHUNK_ID
