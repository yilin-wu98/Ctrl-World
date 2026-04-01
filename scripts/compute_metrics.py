#!/usr/bin/env python3
"""
Compute FID, FVD, LPIPS, PSNR, SSIM from saved per-view .npy files.

Reads gt_view{0,1,2}.npy and pred_view{0,1,2}.npy from the views/ subdirectory
of the output dir. Each .npy file is (T, H, W, 3) uint8.

Usage:
    python scripts/compute_metrics.py \
        --views_dir $SCRATCH/SAILOR/Evals/DROID/ctrl_world_256/views \
        --start_frame 8 --num_frames 50

    # Skip FVD (slow) for quick checks:
    python scripts/compute_metrics.py \
        --views_dir $SCRATCH/SAILOR/Evals/DROID/ctrl_world_256/views \
        --start_frame 8 --num_frames 50 --skip_fvd
"""

import argparse
import gc
import json
import os
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from PIL import Image
from scipy import linalg
from tqdm import tqdm


# =============================================================================
# Feature extractors
# =============================================================================

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
        if x.shape[-2:] != (299, 299):
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = x * 2 - 1
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


# =============================================================================
# Metrics
# =============================================================================

def compute_statistics(features):
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


def _sliding_window_clips(videos, window_size=15, stride=8):
    """Extract sliding-window clips from videos. (N, C, T, H, W) -> (M, C, window_size, H, W)"""
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


def _save_frames_to_dir(images, dirpath):
    """Save (N, C, H, W) float [0,1] tensor as PNG files."""
    os.makedirs(dirpath, exist_ok=True)
    for i in range(len(images)):
        img = (images[i].permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy()
        Image.fromarray(img).save(os.path.join(dirpath, f"{i:06d}.png"))


def compute_fid_streaming(views_dir, traj_dirs, view_idx, start_frame=0, num_frames=None,
                          batch_size=32, device='cuda'):
    """FID using pytorch-fid, streaming one video at a time to avoid OOM.

    Saves all frames as PNGs to temp dirs, then runs pytorch-fid.
    """
    from pytorch_fid import fid_score

    sf = start_frame
    ef = sf + num_frames if num_frames else None
    total_frames = 0

    with tempfile.TemporaryDirectory() as real_dir, tempfile.TemporaryDirectory() as fake_dir:
        for traj_dir in traj_dirs:
            traj_path = os.path.join(views_dir, traj_dir)
            gt = np.load(os.path.join(traj_path, f"gt_view{view_idx}.npy"))[sf:ef]
            pred = np.load(os.path.join(traj_path, f"pred_view{view_idx}.npy"))[sf:ef]
            for i in range(len(gt)):
                idx = total_frames + i
                Image.fromarray(gt[i]).save(os.path.join(real_dir, f"{idx:06d}.png"))
                Image.fromarray(pred[i]).save(os.path.join(fake_dir, f"{idx:06d}.png"))
            total_frames += len(gt)

        print(f"  FID: {total_frames} frames saved, computing...")
        fid = fid_score.calculate_fid_given_paths(
            [real_dir, fake_dir],
            batch_size=batch_size,
            device=device,
            dims=2048,
        )
    torch.cuda.empty_cache()
    return fid


@torch.no_grad()
def compute_fvd(real_videos, fake_videos, batch_size=16, device='cuda'):
    """FVD using I3D features. One clip per video (no sliding windows).

    Expects (N, T, C, H, W) input. Each video is one clip passed to I3D.
    """
    if real_videos.ndim == 5 and real_videos.shape[1] != 3:
        real_videos = real_videos.permute(0, 2, 1, 3, 4)
    if fake_videos.ndim == 5 and fake_videos.shape[1] != 3:
        fake_videos = fake_videos.permute(0, 2, 1, 3, 4)

    feature_extractor = I3DFeatures().to(device).eval()
    print(f"  FVD: {len(real_videos)} real clips, {len(fake_videos)} fake clips")

    def extract_features(clips):
        features = []
        for i in range(0, len(clips), batch_size):
            batch = clips[i:i + batch_size].to(device)
            features.append(feature_extractor(batch).cpu().numpy())
        return np.concatenate(features, axis=0)

    real_features = extract_features(real_videos)
    fake_features = extract_features(fake_videos)
    mu_r, sigma_r = compute_statistics(real_features)
    mu_f, sigma_f = compute_statistics(fake_features)
    fvd = calculate_frechet_distance(mu_r, sigma_r, mu_f, sigma_f)

    feature_extractor.cpu()
    torch.cuda.empty_cache()
    return fvd


@torch.no_grad()
def compute_lpips_streaming(views_dir, traj_dirs, view_idx, start_frame=0, num_frames=None,
                            batch_size=64, device='cuda'):
    """LPIPS using VGG, computed in streaming fashion (one video at a time).

    Loads each video's frames on the fly to avoid OOM.
    Returns mean LPIPS across all frame pairs from all videos.
    """
    import lpips
    lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()

    sf = start_frame
    ef = sf + num_frames if num_frames else None
    total_score = 0.0
    total_frames = 0

    for traj_dir in traj_dirs:
        traj_path = os.path.join(views_dir, traj_dir)
        gt = np.load(os.path.join(traj_path, f"gt_view{view_idx}.npy"))[sf:ef]
        pred = np.load(os.path.join(traj_path, f"pred_view{view_idx}.npy"))[sf:ef]
        real_t = torch.from_numpy(gt).float().permute(0, 3, 1, 2) / 255.0
        pred_t = torch.from_numpy(pred).float().permute(0, 3, 1, 2) / 255.0
        n = real_t.shape[0]

        for i in range(0, n, batch_size):
            r = real_t[i:i + batch_size].to(device) * 2 - 1
            f = pred_t[i:i + batch_size].to(device) * 2 - 1
            scores = lpips_fn(r, f)
            total_score += scores.sum().item()
            total_frames += scores.numel()

        del real_t, pred_t

    lpips_fn.cpu()
    torch.cuda.empty_cache()
    return total_score / total_frames


@torch.no_grad()
def compute_psnr_streaming(views_dir, traj_dirs, view_idx, start_frame=0, num_frames=None,
                           batch_size=64, device='cuda'):
    """PSNR computed in streaming fashion (one video at a time).

    PSNR = 10 * log10(1 / MSE) for images in [0, 1].
    Returns mean PSNR across all frame pairs from all videos.
    """
    sf = start_frame
    ef = sf + num_frames if num_frames else None
    total_psnr = 0.0
    total_frames = 0

    for traj_dir in traj_dirs:
        traj_path = os.path.join(views_dir, traj_dir)
        gt = np.load(os.path.join(traj_path, f"gt_view{view_idx}.npy"))[sf:ef]
        pred = np.load(os.path.join(traj_path, f"pred_view{view_idx}.npy"))[sf:ef]
        real_t = torch.from_numpy(gt).float().permute(0, 3, 1, 2) / 255.0
        pred_t = torch.from_numpy(pred).float().permute(0, 3, 1, 2) / 255.0
        n = real_t.shape[0]

        for i in range(0, n, batch_size):
            r = real_t[i:i + batch_size].to(device)
            f = pred_t[i:i + batch_size].to(device)
            mse = ((r - f) ** 2).mean(dim=(1, 2, 3))  # per-frame MSE
            # Avoid log(0) for perfect frames
            psnr = 10 * torch.log10(1.0 / mse.clamp(min=1e-10))
            total_psnr += psnr.sum().item()
            total_frames += psnr.numel()

        del real_t, pred_t

    torch.cuda.empty_cache()
    return total_psnr / total_frames


@torch.no_grad()
def compute_ssim_streaming(views_dir, traj_dirs, view_idx, start_frame=0, num_frames=None,
                           batch_size=16, device='cuda'):
    """SSIM computed in streaming fashion (one video at a time).

    Uses the standard SSIM formulation with 11x11 Gaussian window.
    Returns mean SSIM across all frame pairs from all videos.
    """
    sf = start_frame
    ef = sf + num_frames if num_frames else None
    total_ssim = 0.0
    total_frames = 0

    # Build 11x11 Gaussian window
    window_size = 11
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_1d = g.unsqueeze(1)
    window_2d = window_1d @ window_1d.t()
    window_2d = window_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, 11, 11)
    window = window_2d.expand(3, 1, -1, -1).contiguous().to(device)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    pad = window_size // 2

    for traj_dir in traj_dirs:
        traj_path = os.path.join(views_dir, traj_dir)
        gt = np.load(os.path.join(traj_path, f"gt_view{view_idx}.npy"))[sf:ef]
        pred = np.load(os.path.join(traj_path, f"pred_view{view_idx}.npy"))[sf:ef]
        real_t = torch.from_numpy(gt).float().permute(0, 3, 1, 2) / 255.0
        pred_t = torch.from_numpy(pred).float().permute(0, 3, 1, 2) / 255.0
        n = real_t.shape[0]

        for i in range(0, n, batch_size):
            r = real_t[i:i + batch_size].to(device)
            f = pred_t[i:i + batch_size].to(device)

            mu_r = F.conv2d(r, window, padding=pad, groups=3)
            mu_f = F.conv2d(f, window, padding=pad, groups=3)
            mu_r_sq = mu_r ** 2
            mu_f_sq = mu_f ** 2
            mu_rf = mu_r * mu_f

            sigma_r_sq = F.conv2d(r * r, window, padding=pad, groups=3) - mu_r_sq
            sigma_f_sq = F.conv2d(f * f, window, padding=pad, groups=3) - mu_f_sq
            sigma_rf = F.conv2d(r * f, window, padding=pad, groups=3) - mu_rf

            ssim_map = ((2 * mu_rf + C1) * (2 * sigma_rf + C2)) / \
                       ((mu_r_sq + mu_f_sq + C1) * (sigma_r_sq + sigma_f_sq + C2))

            # Mean SSIM per frame, then sum
            ssim_per_frame = ssim_map.mean(dim=(1, 2, 3))
            total_ssim += ssim_per_frame.sum().item()
            total_frames += ssim_per_frame.numel()

        del real_t, pred_t

    torch.cuda.empty_cache()
    return total_ssim / total_frames


# =============================================================================
# Data loading
# =============================================================================

def get_traj_dirs(views_dir, max_trajs=None):
    """Get sorted list of valid trajectory directories."""
    traj_dirs = sorted([d for d in os.listdir(views_dir)
                        if os.path.isdir(os.path.join(views_dir, d))])
    # Filter to those with all 3 views
    valid = []
    for d in traj_dirs:
        traj_path = os.path.join(views_dir, d)
        if all(os.path.exists(os.path.join(traj_path, f"gt_view{vi}.npy"))
               and os.path.exists(os.path.join(traj_path, f"pred_view{vi}.npy"))
               for vi in range(3)):
            valid.append(d)
    if max_trajs is not None:
        valid = valid[:max_trajs]
    return valid


def load_single_view(views_dir, traj_dirs, view_idx, start_frame=0, num_frames=None):
    """Load a single view across all trajectories.

    Returns:
        real_list: list of per-video tensors, each (T_i, C, H, W) in [0,1]
        pred_list: list of per-video tensors, each (T_i, C, H, W) in [0,1]
    """
    sf = start_frame
    ef = sf + num_frames if num_frames else None
    real_list, pred_list = [], []
    for traj_dir in traj_dirs:
        traj_path = os.path.join(views_dir, traj_dir)
        gt = np.load(os.path.join(traj_path, f"gt_view{view_idx}.npy"))[sf:ef]
        pred = np.load(os.path.join(traj_path, f"pred_view{view_idx}.npy"))[sf:ef]
        real_list.append(torch.from_numpy(gt).float().permute(0, 3, 1, 2) / 255.0)
        pred_list.append(torch.from_numpy(pred).float().permute(0, 3, 1, 2) / 255.0)
    return real_list, pred_list


def pad_and_stack(vid_list, max_t):
    padded = []
    for v in vid_list:
        if v.shape[0] < max_t:
            continue
            # last = v[-1:].expand(max_t - v.shape[0], -1, -1, -1)
            # v = torch.cat([v, last], dim=0)
        padded.append(v)
    return torch.stack(padded, dim=0)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compute FID/FVD/LPIPS from saved view .npy files")
    parser.add_argument('--views_dir', type=str, required=True,
                        help="Path to views/ directory containing per-trajectory subdirs")
    parser.add_argument('--start_frame', type=int, default=0,
                        help="Start frame for metric computation")
    parser.add_argument('--num_frames', type=int, default=None,
                        help="Number of frames to use (default: all)")
    parser.add_argument('--max_trajs', type=int, default=None,
                        help="Max trajectories to load")
    parser.add_argument('--fvd_window', type=int, default=15,
                        help="FVD sliding window size (default: 15 to match IRASim)")
    parser.add_argument('--fvd_stride', type=int, default=8,
                        help="FVD sliding window stride")
    parser.add_argument('--skip_fvd', action='store_true', help="Skip FVD computation")
    parser.add_argument('--skip_fid', action='store_true', help="Skip FID computation")
    parser.add_argument('--skip_lpips', action='store_true', help="Skip LPIPS computation")
    parser.add_argument('--skip_psnr', action='store_true', help="Skip PSNR computation")
    parser.add_argument('--skip_ssim', action='store_true', help="Skip SSIM computation")
    parser.add_argument('--output', type=str, default=None,
                        help="Path to save metrics JSON (default: <views_dir>/../metrics.json)")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    view_keys = ["view_0", "view_1", "view_2"]

    # Get valid trajectory dirs
    traj_dirs = get_traj_dirs(args.views_dir, max_trajs=args.max_trajs)
    if len(traj_dirs) == 0:
        print("No trajectories found!")
        return
    sf = args.start_frame
    ef = sf + args.num_frames if args.num_frames else 'end'
    print(f"Found {len(traj_dirs)} trajectories, frames [{sf}:{ef}]")

    metrics = {}

    # Process one view at a time to reduce memory
    for vi, key in enumerate(view_keys):
        print(f"\n{'='*60}")
        print(f"Loading view {vi}...")
        print(f"View '{key}': {len(traj_dirs)} videos")
        print(f"{'='*60}")

        if not args.skip_fvd:
            assert args.num_frames is not None, "FVD requires --num_frames to fix video length"
            real_list, pred_list = load_single_view(
                args.views_dir, traj_dirs, vi,
                start_frame=args.start_frame, num_frames=args.num_frames,
            )
            print(f"  Computing FVD ({args.num_frames} frames per clip)...")
            # Pad all videos to num_frames (fixed length)
            real_stacked = pad_and_stack(real_list, args.num_frames)
            pred_stacked = pad_and_stack(pred_list, args.num_frames)
            metrics[f"fvd_{key}"] = compute_fvd(
                real_stacked, pred_stacked, batch_size=8, device=device,
            )
            print(f"  FVD: {metrics[f'fvd_{key}']:.2f}")
            del real_list, pred_list, real_stacked, pred_stacked
            torch.cuda.empty_cache()
            gc.collect()

        if not args.skip_fid:
            print(f"  Computing FID (streaming)...")
            metrics[f"fid_{key}"] = compute_fid_streaming(
                args.views_dir, traj_dirs, vi,
                start_frame=args.start_frame, num_frames=args.num_frames,
                batch_size=32, device=device,
            )
            print(f"  FID: {metrics[f'fid_{key}']:.2f}")
            torch.cuda.empty_cache()
            gc.collect()

        # LPIPS streams one video at a time — no bulk loading needed
        if not args.skip_lpips:
            print(f"  Computing LPIPS (streaming)...")
            metrics[f"lpips_{key}"] = compute_lpips_streaming(
                args.views_dir, traj_dirs, vi,
                start_frame=args.start_frame, num_frames=args.num_frames,
                batch_size=64, device=device,
            )
            print(f"  LPIPS: {metrics[f'lpips_{key}']:.4f}")
            torch.cuda.empty_cache()
            gc.collect()

        if not args.skip_psnr:
            print(f"  Computing PSNR (streaming)...")
            metrics[f"psnr_{key}"] = compute_psnr_streaming(
                args.views_dir, traj_dirs, vi,
                start_frame=args.start_frame, num_frames=args.num_frames,
                batch_size=64, device=device,
            )
            print(f"  PSNR: {metrics[f'psnr_{key}']:.2f}")
            torch.cuda.empty_cache()
            gc.collect()

        if not args.skip_ssim:
            print(f"  Computing SSIM (streaming)...")
            metrics[f"ssim_{key}"] = compute_ssim_streaming(
                args.views_dir, traj_dirs, vi,
                start_frame=args.start_frame, num_frames=args.num_frames,
                batch_size=16, device=device,
            )
            print(f"  SSIM: {metrics[f'ssim_{key}']:.4f}")
            torch.cuda.empty_cache()
            gc.collect()

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Summary (frames [{args.start_frame}:{args.start_frame + args.num_frames if args.num_frames else 'end'}])")
    print(f"{'='*60}")
    for key in view_keys:
        print(f"\n  {key}:")
        for metric_name in ['fid', 'fvd', 'lpips', 'psnr', 'ssim']:
            mk = f"{metric_name}_{key}"
            if mk in metrics:
                fmt = '.4f' if metric_name in ('lpips', 'ssim') else '.2f'
                print(f"    {metric_name:8s} {metrics[mk]:{fmt}}")
    print(f"{'='*60}")

    # Save
    output_path = args.output or os.path.join(os.path.dirname(args.views_dir), "metrics.json")
    with open(output_path, 'w') as f:
        json.dump({
            "config": {
                "views_dir": args.views_dir,
                "start_frame": args.start_frame,
                "num_frames": args.num_frames,
                "fvd_window": args.fvd_window,
                "num_trajectories": len(traj_dirs),
            },
            "metrics": {k: float(v) for k, v in metrics.items()},
        }, f, indent=2)
    print(f"\nMetrics saved to {output_path}")


if __name__ == "__main__":
    main()
