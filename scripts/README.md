# Ctrl-World Evaluation on DROID

## 1. Video Generation

Generate predicted videos via autoregressive trajectory replay. Uses 16 SLURM array tasks, each processing a chunk of trajectories. Generation starts from frame `start_idx=8`, with history initialized from copies of that frame.

```bash
sbatch scripts/run_eval_ctrl_world.sh
```

Key flags in the script: `MAX_TRAJS=256`, `START_IDX=8`, `NUM_CHUNKS=16`. Outputs per-view `.npy` files to `$SCRATCH/SAILOR/Evals/DROID/ctrl_world_256_start8/views/<traj_id>/gt_view{0,1,2}.npy` and `pred_view{0,1,2}.npy` (uint8, `(T, H, W, 3)`).

To run **without text conditioning**, use `run_eval_ctrl_world_notext.sh` instead (adds `--no_text`).

## 2. Metric Computation

Compute FID (pytorch-fid), FVD (I3D), LPIPS (VGG), PSNR, and SSIM from saved `.npy` files. Runs as SLURM array jobs split by frame horizon.

```bash
sbatch scripts/run_compute_metrics.sh
```

Or run manually for a specific horizon:

```bash
python scripts/compute_metrics.py \
    --views_dir $SCRATCH/SAILOR/Evals/DROID/ctrl_world_256_start8/views \
    --start_frame 0 --num_frames 16 --max_trajs 256 \
    --output metrics_16f.json
```

Use `--skip_fvd`, `--skip_fid`, `--skip_lpips`, `--skip_psnr`, `--skip_ssim` to skip individual metrics. FVD requires `--num_frames` (fixed-length clips); other metrics work with `--num_frames` or without (all frames).

## Camera Views

- `view_0`: exterior_1_left
- `view_1`: exterior_2_left
- `view_2`: wrist_left
