#!/usr/bin/env python3
"""
Ctrl-World Evaluation Script

Replays recorded trajectories through the Ctrl-World model, generates predicted
videos, and computes evaluation metrics (FID, FVD, LPIPS) comparing predictions
against ground truth.

Similar to SAILOR-FM's sailor/dreamer/wm_eval.py but adapted for Ctrl-World's
autoregressive (step-by-step) generation via VAE latent space.

Usage:
    # Generate videos + compute metrics on a dataset split
    python scripts/eval_wm_metrics.py \
        --svd_model_path /path/to/svd \
        --clip_model_path /path/to/clip \
        --ckpt_path /path/to/checkpoint.pt \
        --val_dataset_dir /path/to/preprocessed_data \
        --split val \
        --output_dir ./eval_output \
        --num_videos 16

    # Chunked evaluation for distributed processing
    python scripts/eval_wm_metrics.py \
        --svd_model_path /path/to/svd \
        --clip_model_path /path/to/clip \
        --ckpt_path /path/to/checkpoint.pt \
        --val_dataset_dir /path/to/preprocessed_data \
        --split val \
        --output_dir ./eval_output \
        --val_chunk 4 --val_chunk_id 0

    # Compute metrics only from previously saved videos
    python scripts/eval_wm_metrics.py \
        --output_dir ./eval_output \
        --compute_metrics_only
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math
import os
import sys
import json
import gc
import time
import warnings
from argparse import ArgumentParser
from collections import defaultdict
from typing import Tuple, Optional
from scipy import linalg

import mediapy
from decord import VideoReader, cpu
from accelerate import Accelerator
from tqdm.auto import tqdm

# Add parent directory for model imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.ctrl_world import CrtlWorld


# =============================================================================
# Metrics (adapted from SAILOR-FM sailor/dreamer/metrics.py)
# =============================================================================

def compute_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m} too large")
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


class InceptionV3Features(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import inception_v3, Inception_V3_Weights
        self.inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        self.inception.eval()
        self.inception.fc = nn.Identity()
        for param in self.inception.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        x = x * 2 - 1
        if x.shape[-2:] != (299, 299):
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        return self.inception(x)


class I3DFeatures(nn.Module):
    _WEIGHT_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"

    def __init__(self):
        super().__init__()
        self.model = self._load_i3d_model()
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @staticmethod
    def _load_i3d_model():
        cache_dir = os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "i3d"
        )
        os.makedirs(cache_dir, exist_ok=True)
        filepath = os.path.join(cache_dir, "i3d_torchscript.pt")
        if not os.path.isfile(filepath):
            print(f"Downloading I3D weights to {filepath} ...")
            torch.hub.download_url_to_file(I3DFeatures._WEIGHT_URL, filepath, progress=True)
        return torch.jit.load(filepath, map_location="cpu")

    @torch.no_grad()
    def forward(self, x):
        x = x * 2 - 1
        if x.shape[-2:] != (224, 224):
            B, C, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            x = x.reshape(B, T, C, 224, 224).permute(0, 2, 1, 3, 4).contiguous()
        return self.model(x, return_features=True)


def _sliding_window_clips(videos, window_size=16, stride=8):
    N, C, T, H, W = videos.shape
    clips = []
    for i in range(N):
        vid = videos[i]
        if T < window_size:
            pad = vid[:, -1:].expand(-1, window_size - T, -1, -1)
            clips.append(torch.cat([vid, pad], dim=1).unsqueeze(0))
        else:
            starts = list(range(0, T - window_size + 1, stride))
            if starts[-1] + window_size < T:
                starts.append(T - window_size)
            for s in starts:
                clips.append(vid[:, s:s + window_size].unsqueeze(0))
    return torch.cat(clips, dim=0)


@torch.no_grad()
def compute_fid(real_images, fake_images, batch_size=50, device='cuda'):
    if real_images.ndim == 5:
        real_images = einops.rearrange(real_images, 'b t c h w -> (b t) c h w')
    if fake_images.ndim == 5:
        fake_images = einops.rearrange(fake_images, 'b t c h w -> (b t) c h w')

    feature_extractor = InceptionV3Features().to(device).eval()

    def extract_features(images):
        features = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(device)
            features.append(feature_extractor(batch).cpu().numpy())
        return np.concatenate(features, axis=0)

    real_features = extract_features(real_images)
    fake_features = extract_features(fake_images)
    mu_r, sigma_r = compute_statistics(real_features)
    mu_f, sigma_f = compute_statistics(fake_features)
    fid = calculate_frechet_distance(mu_r, sigma_r, mu_f, sigma_f)

    feature_extractor.cpu()
    torch.cuda.empty_cache()
    return fid


@torch.no_grad()
def compute_fvd(real_videos, fake_videos, batch_size=16, device='cuda', window_size=16, stride=8):
    """
    Compute FVD using the StyleGAN-V sliding-window approach.

    Reference: Skorokhodov et al., "StyleGAN-V: A Continuous Video Generator
    with Arbitrary Video Length", CVPR 2022.
    Uses I3D (Kinetics-400) features with sliding windows of `window_size` frames.
    """
    # Ensure (N, 3, T, H, W)
    if real_videos.ndim == 5 and real_videos.shape[1] != 3:
        real_videos = real_videos.permute(0, 2, 1, 3, 4)
    if fake_videos.ndim == 5 and fake_videos.shape[1] != 3:
        fake_videos = fake_videos.permute(0, 2, 1, 3, 4)

    feature_extractor = I3DFeatures().to(device).eval()
    real_clips = _sliding_window_clips(real_videos, window_size, stride)
    fake_clips = _sliding_window_clips(fake_videos, window_size, stride)
    print(f"FVD: {len(real_clips)} real clips, {len(fake_clips)} fake clips")

    def extract_features(clips):
        features = []
        for i in range(0, len(clips), batch_size):
            batch = clips[i:i + batch_size].to(device)
            features.append(feature_extractor(batch).cpu().numpy())
        return np.concatenate(features, axis=0)

    real_features = extract_features(real_clips)
    fake_features = extract_features(fake_clips)
    mu_r, sigma_r = compute_statistics(real_features)
    mu_f, sigma_f = compute_statistics(fake_features)
    fvd = calculate_frechet_distance(mu_r, sigma_r, mu_f, sigma_f)

    feature_extractor.cpu()
    torch.cuda.empty_cache()
    return fvd


@torch.no_grad()
def compute_lpips(real_images, fake_images, batch_size=256, device='cuda'):
    import lpips
    lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()

    if real_images.ndim == 5:
        real_images = real_images.reshape(-1, *real_images.shape[2:])
    if fake_images.ndim == 5:
        fake_images = fake_images.reshape(-1, *fake_images.shape[2:])

    scores = []
    for i in range(0, len(real_images), batch_size):
        r = real_images[i:i + batch_size].to(device) * 2 - 1
        f = fake_images[i:i + batch_size].to(device) * 2 - 1
        scores.append(lpips_fn(r, f).mean().cpu())

    lpips_fn.cpu()
    torch.cuda.empty_cache()
    return torch.stack(scores).mean().item()


def compute_all_metrics(real_videos_dict, pred_videos_dict, view_keys, device='cuda'):
    """
    Compute FID, FVD, LPIPS for each camera view.

    Args:
        real_videos_dict: {view_key: list of (T, 3, H, W) tensors in [0, 1]}
        pred_videos_dict: {view_key: list of (T, 3, H, W) tensors in [0, 1]}
        view_keys: list of camera view names

    Returns:
        dict of metrics
    """
    metrics = {}

    for key in view_keys:
        if key not in real_videos_dict or key not in pred_videos_dict:
            continue

        real_list = real_videos_dict[key]
        pred_list = pred_videos_dict[key]

        # Pad to max length and stack: (N, T, 3, H, W)
        max_t = max(max(v.shape[0] for v in real_list), max(v.shape[0] for v in pred_list))

        def pad_and_stack(vid_list, max_t):
            padded = []
            for v in vid_list:
                if v.shape[0] < max_t:
                    last = v[-1:].expand(max_t - v.shape[0], -1, -1, -1)
                    v = torch.cat([v, last], dim=0)
                padded.append(v)
            return torch.stack(padded, dim=0)

        real_t = pad_and_stack(real_list, max_t).clamp(0, 1)
        pred_t = pad_and_stack(pred_list, max_t).clamp(0, 1)

        # Truncate to same count and length
        n_min = min(real_t.shape[0], pred_t.shape[0])
        t_min = min(real_t.shape[1], pred_t.shape[1])
        real_t = real_t[:n_min, :t_min]
        pred_t = pred_t[:n_min, :t_min]

        print(f"\nComputing metrics for view '{key}': {real_t.shape}")

        # FVD
        print(f"  Computing FVD...")
        metrics[f"fvd_{key}"] = compute_fvd(real_t, pred_t, batch_size=8, device=device)
        torch.cuda.empty_cache()
        gc.collect()

        # FID
        print(f"  Computing FID...")
        metrics[f"fid_{key}"] = compute_fid(real_t, pred_t, batch_size=8, device=device)
        torch.cuda.empty_cache()
        gc.collect()

        # LPIPS
        print(f"  Computing LPIPS...")
        metrics[f"lpips_{key}"] = compute_lpips(real_t, pred_t, batch_size=8, device=device)
        torch.cuda.empty_cache()
        gc.collect()

        print(f"  {key} | FVD: {metrics[f'fvd_{key}']:.2f} | FID: {metrics[f'fid_{key}']:.2f} | LPIPS: {metrics[f'lpips_{key}']:.4f}")

    return metrics


def print_metrics(metrics, title="Metrics"):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    grouped = {}
    for key, value in sorted(metrics.items()):
        parts = key.split("_", 1)
        metric_name = parts[0]
        camera = parts[1] if len(parts) > 1 else "global"
        grouped.setdefault(camera, {})[metric_name] = value
    for camera, camera_metrics in grouped.items():
        print(f"\n  {camera}:")
        for metric, value in sorted(camera_metrics.items()):
            print(f"    {metric:<8s} {value:.4f}" if isinstance(value, float) else f"    {metric:<8s} {value}")
    print(f"\n{'=' * 60}\n")


# =============================================================================
# Ctrl-World Agent (model loading + trajectory replay)
# =============================================================================

class CtrlWorldAgent:
    """Loads Ctrl-World model and replays trajectories, collecting per-view predictions."""

    def __init__(self, args):
        args.val_model_path = args.ckpt_path
        self.args = args
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.dtype = args.dtype

        # Load Ctrl-World model
        self.model = CrtlWorld(args)
        self.model.load_state_dict(torch.load(args.val_model_path, map_location='cpu'))
        self.model.to(self.device).to(self.dtype)
        self.model.eval()
        print("Ctrl-World model loaded successfully")

        # Load normalization statistics
        with open(args.data_stat_path, 'r') as f:
            data_stat = json.load(f)
            self.state_p01 = np.array(data_stat['state_01'])[None, :]
            self.state_p99 = np.array(data_stat['state_99'])[None, :]

    def normalize_bound(self, data, data_min, data_max, clip_min=-1, clip_max=1, eps=1e-8):
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    def get_traj_info(self, traj_id, start_idx=0, steps=8, split='train', pred_step=1):
        """Load trajectory metadata, actions, and encode video frames to VAE latents."""
        val_dataset_dir = self.args.val_dataset_dir
        args = self.args
        skip = args.skip_step
        num_frames = steps

        annotation_path = f"{val_dataset_dir}/annotations/{split}/{traj_id}.json"
        with open(annotation_path) as f:
            anno = json.load(f)
            try:
                length = len(anno['action'])
            except Exception:
                length = anno["state_length"]

        cut_interaction_num = math.ceil((length - start_idx) / ((pred_step - 1) * skip))
        frames_ids = np.arange(start_idx, start_idx + (cut_interaction_num * pred_step + 8) * skip, skip)
        max_ids = np.ones_like(frames_ids) * (length - 1)
        frames_ids = np.min([frames_ids, max_ids], axis=0).astype(int)

        # Actions and joint positions
        instruction = anno['texts'][0]
        car_action = np.array(anno['states'])[frames_ids]
        joint_pos = np.array(anno['observation.state.joint_position'])
        gripper_pos = np.array(anno['observation.state.gripper_position'])
        if len(gripper_pos.shape) == 1:
            gripper_pos = gripper_pos[:, None]
        joint_pos = np.concatenate([joint_pos, gripper_pos], axis=-1)[frames_ids]

        # Load and split video into views
        num_views = 3
        video_path = f"{val_dataset_dir}/{anno['video_path']}"
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        try:
            concat_video = vr.get_batch(range(length)).asnumpy()
        except Exception:
            concat_video = vr.get_batch(range(length)).numpy()
        concat_video = concat_video[frames_ids]

        video_dict = []
        video_latent = []
        w_per_view = concat_video.shape[2] // num_views
        for view_id in range(num_views):
            start_w = view_id * w_per_view
            end_w = (view_id + 1) * w_per_view
            true_video = concat_video[:, :, start_w:end_w, :]
            video_dict.append(true_video)

            # Encode to VAE latent
            device = self.device
            true_video_t = torch.from_numpy(true_video).to(self.dtype).to(device)
            x = true_video_t.permute(0, 3, 1, 2).to(device) / 255.0 * 2 - 1
            vae = self.model.pipeline.vae
            with torch.no_grad():
                batch_size = 32
                latents = []
                for i in range(0, len(x), batch_size):
                    batch = x[i:i + batch_size]
                    latent = vae.encode(batch).latent_dist.sample().mul_(vae.config.scaling_factor)
                    latents.append(latent)
                video_latent.append(torch.cat(latents, dim=0))

        # Trajectory length from start_idx onward (in skipped-frame space)
        effective_length = len(frames_ids)
        return car_action, joint_pos, video_dict, video_latent, instruction, cut_interaction_num, effective_length

    def forward_wm(self, action_cond, video_latent_true, video_latent_cond, his_cond=None, text=None):
        """Run world model forward pass and decode predictions."""
        args = self.args
        image_cond = video_latent_cond

        action_cond = self.normalize_bound(action_cond, self.state_p01, self.state_p99)
        action_cond = torch.tensor(action_cond).unsqueeze(0).to(self.device).to(self.dtype)

        with torch.no_grad():
            if text is not None:
                text_token = self.model.action_encoder(action_cond, text, self.model.tokenizer, self.model.text_encoder)
            else:
                text_token = self.model.action_encoder(action_cond)
            pipeline = self.model.pipeline

            _, latents = CtrlWorldDiffusionPipeline.__call__(
                pipeline,
                image=image_cond,
                text=text_token,
                width=args.width,
                height=int(args.height * 3),
                num_frames=args.num_frames,
                history=his_cond,
                num_inference_steps=args.num_inference_steps,
                decode_chunk_size=args.decode_chunk_size,
                max_guidance_scale=args.guidance_scale,
                fps=args.fps,
                motion_bucket_id=args.motion_bucket_id,
                mask=None,
                output_type='latent',
                return_dict=False,
                frame_level_cond=True,
            )
        # Split 3 views: (B, F, C, 3*H, W) -> (3*B, F, C, H, W)
        latents = einops.rearrange(latents, 'b f c (m h) (n w) -> (b m n) f c h w', m=3, n=1)

        # Decode ground truth
        true_video = torch.stack(video_latent_true, dim=0)
        decoded_video = []
        bsz, frame_num = true_video.shape[:2]
        true_video_flat = true_video.flatten(0, 1)
        decode_kwargs = {}
        for i in range(0, true_video_flat.shape[0], args.decode_chunk_size):
            chunk = true_video_flat[i:i + args.decode_chunk_size] / pipeline.vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        true_video_dec = torch.cat(decoded_video, dim=0)
        true_video_dec = true_video_dec.reshape(bsz, frame_num, *true_video_dec.shape[1:])
        true_video_dec = ((true_video_dec / 2.0 + 0.5).clamp(0, 1) * 255)
        true_video_dec = true_video_dec.detach().to(torch.float32).cpu().numpy().transpose(0, 1, 3, 4, 2).astype(np.uint8)

        # Decode predicted
        decoded_video = []
        bsz_p, frame_num_p = latents.shape[:2]
        x = latents.flatten(0, 1)
        decode_kwargs = {}
        for i in range(0, x.shape[0], args.decode_chunk_size):
            chunk = x[i:i + args.decode_chunk_size] / pipeline.vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        videos = torch.cat(decoded_video, dim=0)
        videos = videos.reshape(bsz_p, frame_num_p, *videos.shape[1:])
        videos = ((videos / 2.0 + 0.5).clamp(0, 1) * 255)
        videos = videos.detach().to(torch.float32).cpu().numpy().transpose(0, 1, 3, 4, 2).astype(np.uint8)

        # Concatenate for visualization: (3_views, T, H*2, W, 3) stacked horizontally
        videos_cat = np.concatenate([true_video_dec, videos], axis=-3)
        videos_cat = np.concatenate([video for video in videos_cat], axis=-2).astype(np.uint8)

        return videos_cat, true_video_dec, videos, latents

    def replay_trajectory(self, traj_id, start_idx=0, split='train'):
        """
        Replay a full trajectory and return per-view ground truth and predicted frames.

        Returns:
            gt_views: list of 3 arrays, each (T, H, W, 3) uint8
            pred_views: list of 3 arrays, each (T, H, W, 3) uint8
            concat_video: (T, H_cat, W_cat, 3) uint8 for visualization
        """
        args = self.args
        pred_step = args.pred_step
        num_history = args.num_history
        num_frames = args.num_frames

        eef_gt, joint_pos_gt, video_dict, video_latents, instruction, cut_interaction_num, trajectory_length = \
            self.get_traj_info(traj_id, start_idx=start_idx, steps=int(pred_step * args.interact_num + 8), split=split, pred_step=pred_step)

        # Initialize history buffers
        his_cond = []
        his_eef = []
        first_latent = torch.cat([v[0] for v in video_latents], dim=1).unsqueeze(0)
        for _ in range(num_history * 4):
            his_cond.append(first_latent)
            his_eef.append(eef_gt[0:1])

        video_to_save = []
        gt_frames_per_view = [[] for _ in range(3)]
        pred_frames_per_view = [[] for _ in range(3)]

        for i in range(cut_interaction_num):
            start_id = int(i * (pred_step - 1))
            end_id = start_id + pred_step
            video_latent_true = [v[start_id:end_id] for v in video_latents]

            cartesian_pose = eef_gt[start_id:end_id]

            # Build history and action conditioning
            history_idx = [0, 0, -8, -6, -4, -2]
            his_pose = np.concatenate([his_eef[idx] for idx in history_idx], axis=0)
            action_cond = np.concatenate([his_pose, cartesian_pose], axis=0)
            his_cond_input = torch.cat([his_cond[idx] for idx in history_idx], dim=0).unsqueeze(0)
            current_latent = his_cond[-1]

            videos_cat, true_videos, pred_videos_raw, predicted_latents = self.forward_wm(
                action_cond, video_latent_true, current_latent,
                his_cond=his_cond_input,
                text=instruction if args.text_cond else None,
            )

            # Update history
            his_eef.append(cartesian_pose[pred_step - 1:pred_step])
            his_cond.append(torch.cat([v[pred_step - 1] for v in predicted_latents], dim=1).unsqueeze(0))

            # Collect per-view frames (exclude last frame except for final step)
            if i == cut_interaction_num - 1:
                slice_end = pred_step
            else:
                slice_end = pred_step - 1

            for view_id in range(3):
                gt_frames_per_view[view_id].append(true_videos[view_id][:slice_end])
                pred_frames_per_view[view_id].append(pred_videos_raw[view_id][:slice_end])

            if i == cut_interaction_num - 1:
                video_to_save.append(videos_cat)
            else:
                video_to_save.append(videos_cat[:slice_end])

        # Concatenate all frames
        concat_video = np.concatenate(video_to_save, axis=0)[:trajectory_length]

        for view_id in range(3):
            gt_frames_per_view[view_id] = np.concatenate(gt_frames_per_view[view_id], axis=0)[:trajectory_length]
            pred_frames_per_view[view_id] = np.concatenate(pred_frames_per_view[view_id], axis=0)[:trajectory_length]

        return gt_frames_per_view, pred_frames_per_view, concat_video


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = ArgumentParser(description="Ctrl-World evaluation: generate videos and compute metrics")

    # Model paths
    parser.add_argument('--svd_model_path', type=str, default=None)
    parser.add_argument('--clip_model_path', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)

    # Dataset
    parser.add_argument('--val_dataset_dir', type=str, default=None)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--metric_start_frame', type=int, default=0,
                        help="Start frame index for metric computation (skip initial frames)")
    parser.add_argument('--metric_num_frames', type=int, default=None,
                        help="Number of frames to use for metrics (e.g., 50). If None, use all frames.")
    parser.add_argument('--data_stat_path', type=str, default=None)

    # Task config
    parser.add_argument('--task_type', type=str, default='replay_val_videos')
    parser.add_argument('--dataset_root_path', type=str, default=None)
    parser.add_argument('--dataset_meta_info_path', type=str, default=None)
    parser.add_argument('--dataset_names', type=str, default=None)

    # Chunking for distributed evaluation
    parser.add_argument('--val_chunk', type=int, default=None, help="Total number of chunks")
    parser.add_argument('--val_chunk_id', type=int, default=None, help="Which chunk to process (0-indexed)")
    parser.add_argument('--traj_ids_file', type=str, default=None,
                        help="JSON file with list of trajectory IDs. If not provided, reads from annotations dir.")

    # Output
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(os.environ.get('SCRATCH', '.'), 'SAILOR', 'Evals', 'DROID'),
                        help="Directory to save videos and metrics (default: $SCRATCH/SAILOR/Evals/DROID)")
    parser.add_argument('--num_videos', type=int, default=16,
                        help="Max number of comparison videos to save (0 to skip)")
    parser.add_argument('--no_videos', action='store_true', help="Skip saving videos entirely")
    parser.add_argument('--video_fps', type=int, default=4)

    # Metrics
    parser.add_argument('--compute_metrics_only', action='store_true',
                        help="Load pre-saved per-view videos and compute metrics only (no model needed)")
    parser.add_argument('--skip_metrics', action='store_true',
                        help="Skip metric computation, only generate and save videos")

    # Dataloader mode: use DROID Dataset_mix instead of per-trajectory replay
    parser.add_argument('--use_dataloader', action='store_true',
                        help="Use DROID Dataset_mix dataloader for evaluation (single-step predictions)")
    parser.add_argument('--num_samples', type=int, default=256,
                        help="Number of samples for metric computation in dataloader mode")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for dataloader mode")
    parser.add_argument('--num_workers', type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument('--max_trajs', type=int, default=None,
                        help="Maximum number of trajectories to evaluate (limits both dataloader and replay modes)")
    parser.add_argument('--no_text', action='store_true',
                        help="Disable text conditioning (pass blank text to the model)")

    return parser.parse_args()


def get_traj_ids(args):
    """Get trajectory IDs to evaluate."""
    if args.traj_ids_file is not None:
        with open(args.traj_ids_file, 'r') as f:
            all_ids = json.load(f)
    else:
        # List annotation files in the split directory
        anno_dir = os.path.join(args.val_dataset_dir, "annotations", args.split)
        all_ids = sorted([f.replace('.json', '') for f in os.listdir(anno_dir) if f.endswith('.json')])

    # Limit number of trajectories first, then chunk
    if args.max_trajs is not None:
        all_ids = all_ids[:args.max_trajs]

    # Apply chunking
    if args.val_chunk is not None and args.val_chunk_id is not None:
        chunk_size = len(all_ids) // args.val_chunk
        start = args.val_chunk_id * chunk_size
        end = start + chunk_size if args.val_chunk_id < args.val_chunk - 1 else len(all_ids)
        all_ids = all_ids[start:end]

    return all_ids


def load_saved_views(output_dir, view_keys, metric_start_frame=0, metric_num_frames=None):
    """Load previously saved per-view .npy files for metrics-only mode."""
    real_videos = {k: [] for k in view_keys}
    pred_videos = {k: [] for k in view_keys}

    views_dir = os.path.join(output_dir, "views")
    if not os.path.isdir(views_dir):
        raise FileNotFoundError(f"Views directory not found: {views_dir}")

    ms = metric_start_frame
    me = ms + metric_num_frames if metric_num_frames else None

    traj_dirs = sorted([d for d in os.listdir(views_dir) if os.path.isdir(os.path.join(views_dir, d))])
    for traj_dir in traj_dirs:
        traj_path = os.path.join(views_dir, traj_dir)
        for vi, key in enumerate(view_keys):
            gt_path = os.path.join(traj_path, f"gt_view{vi}.npy")
            pred_path = os.path.join(traj_path, f"pred_view{vi}.npy")
            if os.path.exists(gt_path) and os.path.exists(pred_path):
                gt = torch.from_numpy(np.load(gt_path)[ms:me]).float() / 255.0
                pred = torch.from_numpy(np.load(pred_path)[ms:me]).float() / 255.0
                gt = gt.permute(0, 3, 1, 2)
                pred = pred.permute(0, 3, 1, 2)
                real_videos[key].append(gt)
                pred_videos[key].append(pred)

    print(f"Loaded {len(traj_dirs)} trajectories from {views_dir} (frames [{ms}:{me}])")
    return real_videos, pred_videos


def decode_latents_to_pixels(pipeline, latents, decode_chunk_size=7):
    """Decode VAE latents to pixel-space uint8 numpy arrays.

    Args:
        pipeline: SVD pipeline with .vae decoder
        latents: (B, T, C, H, W) tensor of latents
        decode_chunk_size: number of frames to decode at once

    Returns:
        videos: (B, T, H_px, W_px, 3) uint8 numpy array
    """
    bsz, frame_num = latents.shape[:2]
    flat = latents.flatten(0, 1)
    decoded = []
    decode_kwargs = {}
    for i in range(0, flat.shape[0], decode_chunk_size):
        chunk = flat[i:i + decode_chunk_size] / pipeline.vae.config.scaling_factor
        decode_kwargs["num_frames"] = chunk.shape[0]
        decoded.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
    videos = torch.cat(decoded, dim=0)
    videos = videos.reshape(bsz, frame_num, *videos.shape[1:])
    videos = ((videos / 2.0 + 0.5).clamp(0, 1) * 255)
    return videos.detach().to(torch.float32).cpu().numpy().transpose(0, 1, 3, 4, 2).astype(np.uint8)


def encode_frames_to_latent(vae, frames, dtype, device, batch_size=32):
    """Encode RGB frames (T, 3, H, W) in [0,1] to SVD VAE latents (T, 4, H_lat, W_lat).

    Args:
        vae: SVD VAE encoder
        frames: (T, 3, H, W) tensor in [0, 1]
        dtype: torch dtype for inference
        device: torch device

    Returns:
        latents: (T, 4, H_lat, W_lat) tensor
    """
    x = frames.to(dtype).to(device) * 2 - 1  # scale to [-1, 1]
    latents = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = x[i:i + batch_size]
            latent = vae.encode(batch).latent_dist.sample().mul_(vae.config.scaling_factor)
            latents.append(latent)
    return torch.cat(latents, dim=0)


def eval_with_dataloader(model_args, cli_args, device):
    """
    Evaluate using SAILOR's PrecomputedDroid dataloader for single-step predictions.

    Uses the same DROID dataloader as sailor/dreamer/wm_eval.py:
    1. Loads batches from PrecomputedDroid (per-camera video frames + states/actions)
    2. Encodes video frames through SVD VAE to get latents
    3. Stacks 3 camera views vertically for Ctrl-World input format
    4. Extracts cartesian states from annotations for action conditioning
    5. Runs Ctrl-World pipeline, decodes predictions, computes metrics
    """
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset'))
    from droid import PrecomputedDroid

    img_keys = ['exterior_1_left', 'exterior_2_left', 'wrist_left']
    view_keys = ["view_0", "view_1", "view_2"]
    output_dir = cli_args.output_dir
    vid_dir = os.path.join(output_dir, "videos")
    os.makedirs(vid_dir, exist_ok=True)

    # Load Ctrl-World model
    model = CrtlWorld(model_args)
    model.load_state_dict(torch.load(model_args.ckpt_path, map_location='cpu'))
    model.to(device).to(model_args.dtype)
    model.eval()
    pipeline = model.pipeline
    vae = pipeline.vae
    print("Ctrl-World model loaded for dataloader evaluation")

    # Load percentile normalization stats for actions
    with open(model_args.data_stat_path, 'r') as f:
        data_stat = json.load(f)
        state_p01 = np.array(data_stat['state_01'])[None, :]
        state_p99 = np.array(data_stat['state_99'])[None, :]

    def normalize_bound(data, eps=1e-8):
        ndata = 2 * (data - state_p01) / (state_p99 - state_p01 + eps) - 1
        return np.clip(ndata, -1, 1)

    num_history = model_args.num_history
    num_frames = model_args.num_frames
    total_frames = num_history + num_frames

    # Create DROID dataset with video frames
    dataset_path = cli_args.val_dataset_dir
    print(f"Loading PrecomputedDroid from {dataset_path}, split={cli_args.split}")
    val_dataset = PrecomputedDroid(
        root=dataset_path,
        split=cli_args.split,
        horizon=num_frames,
        img_keys=img_keys,
        relabel_actions=False,
        normalize=False,  # We normalize actions ourselves with percentile bounds
        cache_trajectories=False,
        return_language=True,
        load_precomputed_features=False,
        max_trajectories=cli_args.num_samples,
        return_video_frames=True,
        encoder_type='svd',
        n_history=num_history,
        use_fixed_t=False,
        use_fixed_id=True,
        eval_mode=True,
        return_cartesian_states=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cli_args.batch_size,
        shuffle=False,
        num_workers=cli_args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    real_videos = {k: [] for k in view_keys}
    pred_videos = {k: [] for k in view_keys}
    samples_collected = 0
    video_count = 0
    eval_start = time.time()

    print(f"Evaluating {cli_args.num_samples} samples via PrecomputedDroid (batch_size={cli_args.batch_size})...")

    for batch in tqdm(val_dataloader, desc="Evaluating batches"):
        if samples_collected >= cli_args.num_samples:
            break

        bsz = batch['obs'][img_keys[0]].shape[0]
        actual_bsz = min(bsz, cli_args.num_samples - samples_collected)

        # Encode per-camera video frames to VAE latents and stack vertically
        # obs[key]: (B, T, 3, H, W) float [0, 1]
        all_latents = []
        for b_idx in range(actual_bsz):
            per_cam_latents = []
            for key in img_keys:
                frames = batch['obs'][key][b_idx]  # (T, 3, H, W)
                latent = encode_frames_to_latent(vae, frames, model_args.dtype, device)  # (T, 4, H_lat, W_lat)
                per_cam_latents.append(latent)
            # Stack 3 views vertically: (T, 4, 3*H_lat, W_lat) = (T, 4, 72, 40)
            stacked = torch.cat(per_cam_latents, dim=2)  # cat along H dimension
            all_latents.append(stacked)

        latent_all = torch.stack(all_latents, dim=0)  # (B, T, 4, 72, 40)

        his_latent_gt = latent_all[:, :num_history]    # (B, H, 4, 72, 40)
        future_latent_gt = latent_all[:, num_history:]  # (B, F, 4, 72, 40)
        current_latent = future_latent_gt[:, 0]         # (B, 4, 72, 40)

        # Build cartesian action conditioning from states
        # Ctrl-World uses cartesian(6) + gripper(1) = 7D with percentile normalization
        # PrecomputedDroid with return_cartesian_states=True returns obs['cartesian_states']: (B, T, 7)
        action_cond = batch['obs']['cartesian_states'][:actual_bsz].numpy()  # (B, T, 7)

        # Normalize with percentile bounds
        action_cond_norm = np.zeros_like(action_cond)
        for b_idx in range(actual_bsz):
            action_cond_norm[b_idx] = normalize_bound(action_cond[b_idx])
        actions_tensor = torch.tensor(action_cond_norm).to(device).to(model_args.dtype)

        # Get text instructions
        texts = batch['task']['text'][:actual_bsz] if 'task' in batch else [""] * actual_bsz

        with torch.no_grad():
            # Encode actions + text
            if model_args.text_cond:
                action_latent = model.action_encoder(
                    actions_tensor, texts, model.tokenizer, model.text_encoder, model_args.frame_level_cond
                )
            else:
                action_latent = model.action_encoder(actions_tensor)

            # Generate predicted future frames
            _, pred_latents = CtrlWorldDiffusionPipeline.__call__(
                pipeline,
                image=current_latent,
                text=action_latent,
                width=model_args.width,
                height=int(3 * model_args.height),
                num_frames=num_frames,
                history=his_latent_gt,
                num_inference_steps=model_args.num_inference_steps,
                decode_chunk_size=model_args.decode_chunk_size,
                max_guidance_scale=model_args.guidance_scale,
                fps=model_args.fps,
                motion_bucket_id=model_args.motion_bucket_id,
                mask=None,
                output_type='latent',
                return_dict=False,
                frame_level_cond=model_args.frame_level_cond,
                his_cond_zero=model_args.his_cond_zero,
            )

        # Split 3 camera views: (B, F, C, 3*H, W) -> (3*B, F, C, H, W)
        pred_latents_split = einops.rearrange(pred_latents, 'b f c (m h) (n w) -> (b m n) f c h w', m=3, n=1)
        gt_latents_split = einops.rearrange(future_latent_gt, 'b f c (m h) (n w) -> (b m n) f c h w', m=3, n=1)

        # Decode to pixel space
        gt_pixels = decode_latents_to_pixels(pipeline, gt_latents_split, model_args.decode_chunk_size)
        pred_pixels = decode_latents_to_pixels(pipeline, pred_latents_split, model_args.decode_chunk_size)

        # Reshape to (B, 3, F, H, W, 3)
        gt_pixels = gt_pixels.reshape(actual_bsz, 3, *gt_pixels.shape[1:])
        pred_pixels = pred_pixels.reshape(actual_bsz, 3, *pred_pixels.shape[1:])

        for i in range(actual_bsz):
            if samples_collected >= cli_args.num_samples:
                break

            for vi, key in enumerate(view_keys):
                gt_t = torch.from_numpy(gt_pixels[i, vi]).float().permute(0, 3, 1, 2) / 255.0
                pred_t = torch.from_numpy(pred_pixels[i, vi]).float().permute(0, 3, 1, 2) / 255.0
                real_videos[key].append(gt_t)
                pred_videos[key].append(pred_t)

            # Save comparison videos
            if not cli_args.no_videos and video_count < cli_args.num_videos:
                gt_row = np.concatenate([gt_pixels[i, v] for v in range(3)], axis=2)
                pred_row = np.concatenate([pred_pixels[i, v] for v in range(3)], axis=2)
                concat = np.concatenate([gt_row, pred_row], axis=1)
                out_path = os.path.join(vid_dir, f"eval_sample{video_count}.mp4")
                mediapy.write_video(out_path, concat, fps=cli_args.video_fps)
                video_count += 1

            samples_collected += 1

        del latent_all, actions_tensor, pred_latents, pred_latents_split, gt_latents_split
        del gt_pixels, pred_pixels
        torch.cuda.empty_cache()
        gc.collect()

    eval_elapsed = time.time() - eval_start
    print(f"Generation complete: {samples_collected} samples in {eval_elapsed:.1f}s")

    return real_videos, pred_videos, view_keys, eval_elapsed


def main():
    cli_args = parse_args()
    view_keys = ["view_0", "view_1", "view_2"]  # 3 camera views

    os.makedirs(cli_args.output_dir, exist_ok=True)
    vid_dir = os.path.join(cli_args.output_dir, "videos")
    views_dir = os.path.join(cli_args.output_dir, "views")
    os.makedirs(vid_dir, exist_ok=True)
    os.makedirs(views_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Metrics-only mode ----
    if cli_args.compute_metrics_only:
        print("Computing metrics from saved views...")
        real_videos, pred_videos = load_saved_views(
            cli_args.output_dir, view_keys,
            metric_start_frame=cli_args.metric_start_frame,
            metric_num_frames=cli_args.metric_num_frames,
        )
        metrics = compute_all_metrics(real_videos, pred_videos, view_keys, device=device)
        print_metrics(metrics, title="Ctrl-World Eval Metrics")
        metrics_path = os.path.join(cli_args.output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
        print(f"Metrics saved to {metrics_path}")
        return

    # ---- Load config ----
    from config import wm_args
    model_args = wm_args(task_type=cli_args.task_type)

    # Merge CLI args into model args
    for k, v in cli_args.__dict__.items():
        if v is not None:
            model_args.__dict__[k] = v

    # Override text conditioning if --no_text is set
    if cli_args.no_text:
        model_args.text_cond = False

    # ---- Dataloader mode (single-step predictions via Dataset_mix) ----
    if cli_args.use_dataloader:
        real_videos, pred_videos, view_keys, eval_elapsed = eval_with_dataloader(
            model_args, cli_args, device
        )

        # Compute metrics
        if not cli_args.skip_metrics and len(real_videos[view_keys[0]]) > 0:
            print(f"\nComputing metrics on {len(real_videos[view_keys[0]])} samples...")
            metrics = compute_all_metrics(real_videos, pred_videos, view_keys, device=device)
            print_metrics(metrics, title="Ctrl-World Eval Metrics (dataloader mode)")

            metrics_to_save = {
                "config": {
                    "checkpoint": cli_args.ckpt_path,
                    "dataset_root_path": model_args.dataset_root_path,
                    "dataset_names": model_args.dataset_names,
                    "split": cli_args.split,
                    "num_samples": len(real_videos[view_keys[0]]),
                    "mode": "dataloader",
                },
                "metrics": {k: float(v) for k, v in metrics.items()},
                "eval_time_s": round(eval_elapsed, 1),
            }
            metrics_path = os.path.join(cli_args.output_dir, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics_to_save, f, indent=2)
            print(f"Metrics saved to {metrics_path}")

        print("\nEvaluation complete.")
        return

    # ---- Trajectory replay mode (autoregressive rollout) ----
    agent = CtrlWorldAgent(model_args)
    traj_ids = get_traj_ids(cli_args)

    # Filter already-processed trajectories
    remaining_ids = []
    for tid in traj_ids:
        traj_views_dir = os.path.join(views_dir, str(tid))
        if os.path.exists(os.path.join(traj_views_dir, "gt_view2.npy")):
            continue  # Already done
        remaining_ids.append(tid)

    print(f"Evaluating {len(remaining_ids)} trajectories ({len(traj_ids) - len(remaining_ids)} already done)")

    # Collect per-view videos for metrics
    all_gt_views = {k: [] for k in view_keys}
    all_pred_views = {k: [] for k in view_keys}
    video_count = 0
    eval_start = time.time()

    for idx, traj_id in enumerate(tqdm(remaining_ids, desc="Replaying trajectories")):
        try:
            gt_views, pred_views, concat_video = agent.replay_trajectory(
                traj_id, start_idx=cli_args.start_idx, split=cli_args.split
            )
        except Exception as e:
            print(f"Error processing trajectory {traj_id}: {e}")
            continue

        # Save per-view numpy arrays (for metrics-only mode later)
        traj_views_dir = os.path.join(views_dir, str(traj_id))
        os.makedirs(traj_views_dir, exist_ok=True)
        for vi in range(3):
            np.save(os.path.join(traj_views_dir, f"gt_view{vi}.npy"), gt_views[vi])
            np.save(os.path.join(traj_views_dir, f"pred_view{vi}.npy"), pred_views[vi])

        # Collect for metrics (optionally slice to a specific frame range)
        ms = cli_args.metric_start_frame
        me = ms + cli_args.metric_num_frames if cli_args.metric_num_frames else None
        for vi, key in enumerate(view_keys):
            gt_t = torch.from_numpy(gt_views[vi][ms:me]).float().permute(0, 3, 1, 2) / 255.0
            pred_t = torch.from_numpy(pred_views[vi][ms:me]).float().permute(0, 3, 1, 2) / 255.0
            all_gt_views[key].append(gt_t)
            all_pred_views[key].append(pred_t)

        # Save comparison video
        if not cli_args.no_videos and video_count < cli_args.num_videos:
            out_path = os.path.join(vid_dir, f"eval_{traj_id}.mp4")
            mediapy.write_video(out_path, concat_video, fps=cli_args.video_fps)
            video_count += 1

        # Memory management
        del gt_views, pred_views, concat_video
        torch.cuda.empty_cache()
        gc.collect()

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(remaining_ids)} trajectories")

    eval_elapsed = time.time() - eval_start
    print(f"\nGeneration complete in {eval_elapsed:.1f}s")

    # Also load any previously saved views
    print("Loading all saved views for metric computation...")
    all_gt_views_full, all_pred_views_full = load_saved_views(
        cli_args.output_dir, view_keys,
        metric_start_frame=cli_args.metric_start_frame,
        metric_num_frames=cli_args.metric_num_frames,
    )

    # Compute metrics
    if not cli_args.skip_metrics and len(all_gt_views_full[view_keys[0]]) > 0:
        print(f"\nComputing metrics on {len(all_gt_views_full[view_keys[0]])} trajectories...")
        metrics = compute_all_metrics(all_gt_views_full, all_pred_views_full, view_keys, device=device)
        print_metrics(metrics, title="Ctrl-World Eval Metrics")

        metrics_to_save = {
            "config": {
                "checkpoint": cli_args.ckpt_path,
                "val_dataset_dir": cli_args.val_dataset_dir,
                "split": cli_args.split,
                "num_trajectories": len(all_gt_views_full[view_keys[0]]),
                "mode": "trajectory_replay",
            },
            "metrics": {k: float(v) for k, v in metrics.items()},
            "eval_time_s": round(eval_elapsed, 1),
        }
        metrics_path = os.path.join(cli_args.output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_to_save, f, indent=2)
        print(f"Metrics saved to {metrics_path}")
    else:
        print("Skipping metrics (no trajectories or --skip_metrics).")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
