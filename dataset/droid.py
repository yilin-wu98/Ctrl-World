"""
DROID PyTorch Dataset for World Models
Loads preprocessed DROID data with latent features
Samples trajectory chunks of fixed horizon for world model training
"""

import os
import json
import random
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from einops import rearrange


# rsync -avz --progress





def read_video_frames_range(video_path: str, start_idx: int, end_idx: int, num_cameras: int) -> Dict[str, np.ndarray]:
    """Read a specific range of frames from a stacked video using random access.

    This is much faster than reading the entire video when only a small chunk is needed.
    Uses OpenCV's seek capability to jump directly to the start frame.

    Args:
        video_path: Path to the video file
        start_idx: Starting frame index (inclusive)
        end_idx: Ending frame index (exclusive)
        num_cameras: Number of cameras stacked horizontally in the video

    Returns:
        Dictionary mapping camera names to numpy arrays of shape (num_frames, H, W, C)
    """
    camera_names = ['exterior_1_left', 'exterior_2_left', 'wrist_left'][:num_cameras]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file {video_path}")

    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    # Get video dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    camera_width = frame_width // num_cameras

    # Read only the frames we need
    frames_by_camera = {cam: [] for cam in camera_names}

    for _ in range(end_idx - start_idx):
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Split frame by camera
        for i, cam_name in enumerate(camera_names):
            start_w = i * camera_width
            end_w = (i + 1) * camera_width
            frames_by_camera[cam_name].append(frame[:, start_w:end_w, :])

    cap.release()

    # Stack frames into arrays
    result = {}
    for cam_name, frames in frames_by_camera.items():
        if frames:
            result[cam_name] = np.stack(frames, axis=0)

    return result


class DROIDTrajectory:
    """Container for a single DROID trajectory"""

    def __init__(
        self,
        data_root: str,
        traj_id: int,
        data_type: str,
        load_on_init: bool = False,
        encoder_type: str = "svd",
        load_precomputed_features: bool = False
    ):
        self.data_root = data_root
        self.traj_id = traj_id
        self.data_type = data_type
        self.encoder_type = encoder_type  # "svd" or "sd3"
        self.load_precomputed_features=load_precomputed_features
        self.loaded = False

        self._annotation = None
        self._video = None
        self._latents = None

        # Load annotation to get metadata
        self._load_annotation()

        if load_on_init:
            self.load()

    def _load_annotation(self):
        """Load trajectory annotation file"""
        anno_path = os.path.join(self.data_root, f"annotations/{self.data_type}/{self.traj_id}.json")
        with open(anno_path, 'r') as f:
            self._annotation = json.load(f)

    def load(self, _load_video: bool = False):
        """Load trajectory latent features into memory.

        Args:
            _load_video: Deprecated, ignored. Use get_video_frames() for video access.
        """
        if self.loaded:
            return

        # # Skip trajectories with 2 cameras (inconsistent with 3-camera setup)
        # if self._annotation.get('num_cameras') == 2:
        #     return

        # Load latent features as numpy (better memory handling with DataLoader workers)
        # Shape: [num_cameras, num_frames, C, H, W]
        if self.load_precomputed_features:
            latent_path = os.path.join(self.data_root, self._annotation['latent_path'])

            # Modify path based on encoder type (sd3 uses compressed npz format)
            if self.encoder_type == "sd3":
                latent_path = latent_path.replace('.npy', '_sd3.npz')

            # Support .npy, .npz (compressed), and .pt formats
            if latent_path.endswith('.npz'):
                with np.load(latent_path) as data:
                    stacked_latents = data['latents'].astype(np.float32)
            elif os.path.exists(latent_path):
                # Load from uncompressed npy format (auto-converts fp16 to fp32)
                stacked_latents = np.load(latent_path).astype(np.float32)
            else:
                npz_path = latent_path.replace('.npy', '.npz')
                if os.path.exists(npz_path):
                    with np.load(npz_path) as data:
                        stacked_latents = data['latents'].astype(np.float32)
                else:
                    raise FileNotFoundError(f"Latent file not found: {latent_path}")
        
      
            # Split latents by camera - numpy slicing creates lightweight views
            num_cameras = self._annotation['num_cameras']
            camera_names = ['exterior_1_left', 'exterior_2_left', 'wrist_left'][:num_cameras]
            self._latents = {}
            for i, cam_name in enumerate(camera_names):
                self._latents[cam_name] = stacked_latents[i]

        self.loaded = True

    def get_video_frames(self, start_idx: int, end_idx: int) -> Dict[str, np.ndarray]:
        """Read a specific range of video frames.

        If preprocessed .npy frames exist, uses memory-mapped loading (very fast).
        Otherwise falls back to OpenCV video decoding (slower).

        Args:
            start_idx: Starting frame index (inclusive)
            end_idx: Ending frame index (exclusive)

        Returns:
            Dictionary mapping camera names to numpy arrays of shape (num_frames, H, W, C)
        """
        num_cameras = self._annotation['num_cameras']
        camera_names = ['exterior_1_left', 'exterior_2_left', 'wrist_left'][:num_cameras]

        # Check if preprocessed numpy frames exist (preferred - much faster)
        if 'video_frames_path' in self._annotation:
            npy_path = os.path.join(self.data_root, self._annotation['video_frames_path'])
            if os.path.exists(npy_path):
                # Memory-mapped loading - only reads the requested frames from disk
                # Shape: (num_cameras, T, H, W, C)
                frames_mmap = np.load(npy_path, mmap_mode='r')

                result = {}
                for i, cam_name in enumerate(camera_names):
                    # Slicing a mmap array only reads those bytes from disk
                    # Need to copy to avoid issues with mmap in DataLoader workers
                    result[cam_name] = np.array(frames_mmap[i, start_idx:end_idx])

                return result

        # Fallback to video decoding (slower)
        video_path = os.path.join(self.data_root, self._annotation['video_path'])
        return read_video_frames_range(video_path, start_idx, end_idx, num_cameras)

    def unload(self):
        """Free memory by unloading trajectory data"""
        if self._latents is not None:
            self._latents.clear()
        self._latents = None
        self.loaded = False

    def __len__(self) -> int:
        return self._annotation['video_length']

    @property
    def annotation(self) -> Dict:
        return self._annotation

    @property
    def latents(self) -> Dict[str, torch.Tensor]:
        if not self.loaded:
            self.load()
        return self._latents


class PrecomputedDroid(Dataset):
    """
    PyTorch Dataset for preprocessed DROID data

    Sampling strategy:
    1. Sample a trajectory
    2. Sample a random chunk of length `horizon` from that trajectory
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        horizon: int = 16,
        img_keys: List[str] = ['exterior_1_left', 'exterior_2_left', 'wrist_left'],
        relabel_actions: bool = False,
        normalize: bool = True,
        cache_trajectories: bool = False,
        return_language: bool = True,
        load_precomputed_features: bool = False,
        max_trajectories: Optional[int] = None,
        return_video_frames: bool = False,
        encoder_type: str = "svd",
        n_memory_frames: int = 0,
        t_memory: int = 1,
        n_history: int = 2,
        use_fixed_t: bool = False,
        fixed_t: int = 0,
        use_fixed_id: bool = False,
        eval_mode: bool = False,
        return_cartesian_states: bool = False,

    ):
        """
        Args:
            root: Root directory of preprocessed DROID dataset
            split: 'train' or 'val'
            horizon: Number of frames to generate (not including history)
            n_history: Number of history/context frames before generation
            img_keys: Which camera views to use (0=exterior_1_left, 1=exterior_2_left, 2=wrist_left)
            relabel_actions: Whether to relabel actions (compute from state differences)
            normalize: Whether to normalize states and actions
            cache_trajectories: Keep trajectories in memory (high RAM usage)
            return_language: Whether to return language instructions
            load_precomputed_features: Whether to load precomputed text features
            max_trajectories: Limit number of trajectories (for debugging)
            return_video_frames: Whether to return raw video frames (useful for saving videos)
            encoder_type: Which encoder features to use ("svd" or "sd3")
        """
        self.root = Path(root)
        self.split = split
        self.horizon = horizon
        self.n_history = n_history
        self.img_keys_list = img_keys
        self.relabel_actions = relabel_actions
        self.normalize = normalize
        self.cache_trajectories = cache_trajectories
        self.return_language = return_language
        self.load_precomputed_features = load_precomputed_features
        self.return_video_frames = return_video_frames
        self.encoder_type = encoder_type
        self.n_memory_frames = n_memory_frames
        self.t_memory = t_memory
        self.use_fixed_t = use_fixed_t
        self.fixed_t = fixed_t
        self.use_fixed_id = use_fixed_id
        self.eval_mode = eval_mode
        self.return_cartesian_states = return_cartesian_states
        self.eps_idx = 0
       
        # Map string keys to video indices
        self.camera_map = {
            'exterior_1_left': 0,
            'exterior_2_left': 1,
            'wrist_left': 2
        }

        # Load trajectory list from annotations
        data_type = 'val' if split == 'valid' else split
        anno_dir = self.root / f"annotations/{data_type}"

        if not anno_dir.exists():
            raise ValueError(f"Annotation directory not found: {anno_dir}")

        # Get all trajectory IDs from annotation files
        self.traj_ids = []
        for anno_file in sorted(anno_dir.glob("*.json")):
            traj_id = int(anno_file.stem)
            self.traj_ids.append(traj_id)

        if max_trajectories:
            self.traj_ids = self.traj_ids[:max_trajectories]

        print(f"Found {len(self.traj_ids)} {split} trajectories")

        # Load normalization statistics (filename depends on relabel_actions)
        if self.normalize:
            suffix = 'relabel' if relabel_actions else 'recorded'
            norm_path = self.root / f"norm_stats_{suffix}.json"
            if norm_path.exists():
                with open(norm_path, 'r') as f:
                    norm_stats = json.load(f)['norm_stats']
                self.norm_dict = {
                    'states': {
                        'mean': torch.tensor(norm_stats['state']['mean']),
                        'std': torch.tensor(norm_stats['state']['std']),
                    },
                    'actions': {
                        'mean': torch.tensor(norm_stats['actions']['mean']),
                        'std': torch.tensor(norm_stats['actions']['std']),
                    }
                }
                print(f"Loaded normalization statistics from {norm_path}")
            else:
                print(f"Warning: {norm_path.name} not found at {norm_path}, skipping normalization")
                self.normalize = False

        # Create trajectory objects and preprocess states/actions
        self.trajectories: List[DROIDTrajectory] = []
        self.valid_trajectories: List[int] = []

        # Pre-loaded data for fast access (indexed by position in self.trajectories)
        self._states: List[torch.Tensor] = []      # Each: (T, 8) float32 joint states
        self._actions: List[torch.Tensor] = []     # Each: (T, 8) float32
        self._cartesian_states: List[torch.Tensor] = []  # Each: (T, 7) float32 cartesian(6)+gripper(1)
        self._text_features: List[torch.Tensor] = []  # Each: (feat_dim,) float32
        self._texts: List[str] = []                # Language instructions

        print("Initializing trajectories and preprocessing states/actions...")
        for traj_id in tqdm(self.traj_ids):
            traj = DROIDTrajectory(
                data_root=str(self.root),
                traj_id=traj_id,
                data_type=data_type,
                load_on_init=cache_trajectories,
                encoder_type=self.encoder_type,
                load_precomputed_features=self.load_precomputed_features
            )
            # # Skip trajectories with 2 cameras (inconsistent with 3-camera setup)
            # if traj.annotation.get('num_cameras') == 2:
            #     continue
            # Preprocess and cache states/actions for this trajectory
            states, actions = self._preprocess_states_actions(traj.annotation)
             # Require both video and states to be long enough (lengths can differ)
            # if not self.pad_short_trajectories:
            if (len(traj) >= horizon + 2 and len(states) >= horizon + 2) or self.eval_mode:
                self.trajectories.append(traj)
                self.valid_trajectories.append(len(self.trajectories) - 1)
                self._states.append(states)
                self._actions.append(actions)
                if self.return_cartesian_states:
                    cart_states = self._preprocess_cartesian_states(traj.annotation)
                    self._cartesian_states.append(cart_states)
            # else:
            #     target_length = horizon + 2
            #     # Pad short trajectories by repeating last frame
            #     if len(traj) < target_length or len(states) < target_length:
            #         traj.load()  # Need latents for padding
            #         states, actions, padded_latents = pad_trajectory_to_length(
            #             states, actions, traj._latents, target_length
            #         )
            #         traj._latents = padded_latents
            #         traj._annotation['video_length'] = target_length
            #         print(f"Padded trajectory {traj_id} to length {target_length}")
            #     # Add trajectory (either original or padded)
            #     self.trajectories.append(traj)
            #     self.valid_trajectories.append(len(self.trajectories) - 1)
            #     self._states.append(states)
            #     self._actions.append(actions)

                # Cache text features and text
                if self.return_language:
                    self._texts.append(traj.annotation['texts'][0])
                    if 'text_features' in traj.annotation:
                        self._text_features.append(
                            torch.tensor(traj.annotation['text_features'], dtype=torch.float32)
                        )

        print(f"Loaded {len(self.valid_trajectories)} valid trajectories (length >= {horizon})")

        if len(self.valid_trajectories) == 0:
            raise ValueError(f"No trajectories found with length >= {horizon}")

    def __len__(self) -> int:
        return len(self.valid_trajectories)

    def _preprocess_states_actions(
        self,
        annotation: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess full trajectory states and actions from annotation.

        Called once during __init__ to cache preprocessed data.

        Note: The annotation stores raw (full-resolution) states and actions,
        but video/latents are downsampled by rgb_skip. We need to convert
        video frame indices to raw indices using the skip factor.

        Returns:
            states: (T, 8) float32 tensor of states at video frame rate
            actions: (T, 8) float32 tensor of actions at video frame rate
        """
        rgb_skip = 3  # Hardcoded for now

        # Get joint states (joint_position + gripper_position) at raw resolution
        joint_position = np.array(annotation['observation.state.joint_position'])
        
        gripper_position = np.array(annotation['observation.state.gripper_position'])[:, None]
        
        if len(gripper_position.shape) == 3:
            gripper_position = gripper_position[:, 0, :]
        full_states = np.concatenate([joint_position, gripper_position], axis=-1)

        # Sample states at video frame rate (every rgb_skip frames)
        raw_length = len(joint_position)
        state_indices = np.arange(0, raw_length, rgb_skip)
        states = full_states[state_indices]

        if self.relabel_actions:
            # Compute actions as state differences between consecutive video frames
            next_state_indices = np.clip(state_indices + rgb_skip, 0, len(full_states) - 1)
            actions = full_states[next_state_indices] - full_states[state_indices]
        else:
            # Use recorded actions - sum actions between video frames
            action_joint = np.array(annotation['action.joint_position'])
            action_gripper = np.array(annotation['action.gripper_position'])[:, None]
            full_actions = np.concatenate([action_joint, action_gripper], axis=-1)

            # Sum actions over each rgb_skip interval to get action per video frame
            actions = []
            for i in state_indices:
                end_action_idx = min(i + rgb_skip, len(full_actions))
                action_sum = full_actions[i:end_action_idx].sum(axis=0)
                actions.append(action_sum)
            actions = np.stack(actions, axis=0)

        return (
            torch.from_numpy(states).float(),
            torch.from_numpy(actions).float()
        )

    def _preprocess_cartesian_states(self, annotation: Dict) -> torch.Tensor:
        """Extract cartesian(6) + gripper(1) = 7D states at video frame rate.

        Uses the 'states' field from annotation if available (already at video frame rate),
        otherwise downsamples from raw 'observation.state.cartesian_position'.
        """
        if 'states' in annotation:
            # preprocessed_v2 annotations have 'states' at video frame rate: (T_vid, 7)
            return torch.tensor(annotation['states'], dtype=torch.float32)

        # Fallback: downsample from raw cartesian + gripper
        rgb_skip = 3
        cartesian = np.array(annotation['observation.state.cartesian_position'])  # (raw_T, 6)
        gripper = np.array(annotation['observation.state.gripper_position'])      # (raw_T,)
        if gripper.ndim == 1:
            gripper = gripper[:, None]
        elif len(gripper.shape) == 3:
            gripper = gripper[:, 0, :]
        full_cart = np.concatenate([cartesian, gripper], axis=-1)  # (raw_T, 7)
        indices = np.arange(0, len(cartesian), rgb_skip)
        return torch.from_numpy(full_cart[indices]).float()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Sample a chunk from a trajectory

        Returns:
            Dict containing:
                - obs: Dict with image latents and states
                  - states: (horizon, 8) float32 joint positions (7D) + gripper position (1D)
                - actions: (horizon, 8) float32 actions (joint deltas + gripper delta)
                - task: language instruction (if return_language=True)
                - rewards: (horizon,) rewards (zeros for DROID)
        """
        # Get trajectory index (position in self.trajectories list)
        if self.use_fixed_id:
            idx = self.eps_idx
            self.eps_idx += 1
            if self.eps_idx >= len(self.valid_trajectories):
                self.eps_idx = 0
        traj_idx = self.valid_trajectories[idx]
        traj = self.trajectories[traj_idx]

        # Ensure trajectory latents are loaded
        if not traj.loaded:
            traj.load()

        # Sample t: the last history frame. History = [t-n_history+1, ..., t],
        # generation = [t+1, ..., t+horizon]. Obs chunk = [t-n_history+1, t+horizon].
        n_hist = self.n_history
        
        min_t = n_hist - 1
        max_t = len(traj) - self.horizon - 1
        ## add an option here to use the random t or a fixed t for the generation
        if self.use_fixed_t:
            t = n_hist -1 + self.fixed_t
        else:
            t = random.randint(min_t, max_t) if max_t > min_t else min_t
        
        ## get the max of 0 and t - n_hist + 1
        start_idx = max(0, t - n_hist + 1)
        # start_idx = t - n_hist + 1
        # end_idx = t + self.horizon + 1
        end_idx = min(len(traj), t + self.horizon + 1)

        # Extract latent features for each camera view
        obs = {}
        if self.load_precomputed_features:
            for img_key in self.img_keys_list:
                # Resolve exterior_rand to a random exterior camera
                if img_key == 'exterior_rand':
                    actual_key = random.choice(['exterior_1_left', 'exterior_2_left'])
                else:
                    actual_key = img_key

                if actual_key not in traj.latents:
                    raise ValueError(f"Invalid image key: {actual_key}. Available cameras: {list(traj.latents.keys())}")

                # Extract latent features for this camera view (numpy slice -> tensor)
                latents = torch.from_numpy(traj.latents[actual_key][start_idx:end_idx])  # (horizon, C, H, W)
                obs[f'{img_key}_features'] = latents

        # Load video frames on-demand using random access (only reads needed frames)
        if self.return_video_frames:
            video_frames = traj.get_video_frames(start_idx, end_idx)
            for img_key in self.img_keys_list:
                if img_key == 'exterior_rand':
                    actual_key = random.choice(['exterior_1_left', 'exterior_2_left'])
                else:
                    actual_key = img_key
                if actual_key in video_frames:
                    obs[img_key] = load_and_preprocess_video(video_frames[actual_key])

        # Get pre-loaded states and actions (slice from cached tensors)
        states = self._states[traj_idx][start_idx:end_idx]
        actions = self._actions[traj_idx][start_idx:end_idx]

        # Normalize if enabled
        if self.normalize:
            states = (states - self.norm_dict['states']['mean']) / self.norm_dict['states']['std']
            actions = (actions - self.norm_dict['actions']['mean']) / self.norm_dict['actions']['std']

        obs['states'] = states

        # Add cartesian states if requested (for Ctrl-World action conditioning)
        if self.return_cartesian_states and self._cartesian_states:
            obs['cartesian_states'] = self._cartesian_states[traj_idx][start_idx:end_idx]

        result = {
            'obs': obs,
            'actions': actions,
            'rewards': torch.zeros(self.n_history + self.horizon),
        }

        # Build sparse memory frames before history, spaced by t_memory from t
        if self.n_memory_frames > 0:
            # Clamp to valid range for both latents and states (states_len can be shorter than traj_len)
            memory_indices = [
                max(0, t - (self.n_memory_frames - i) * self.t_memory)
                for i in range(self.n_memory_frames)
            ]

            memory_obs = {}

            # Latent features for each camera
            for img_key in self.img_keys_list:
                if img_key == 'exterior_rand':
                    actual_key = random.choice(['exterior_1_left', 'exterior_2_left'])
                else:
                    actual_key = img_key
                if self.load_precomputed_features:
                    mem_latents = np.stack([traj.latents[actual_key][mi] for mi in memory_indices])
                    memory_obs[f'{img_key}_features'] = torch.from_numpy(mem_latents)

            # Raw video frames if requested
            if self.return_video_frames:
                for img_key in self.img_keys_list:
                    if img_key == 'exterior_rand':
                        actual_key = random.choice(['exterior_1_left', 'exterior_2_left'])
                    else:
                        actual_key = img_key
                    mem_frames = []
                    for mi in memory_indices:
                        frame_dict = traj.get_video_frames(mi, mi + 1)
                        if actual_key in frame_dict:
                            mem_frames.append(frame_dict[actual_key][0])
                    if mem_frames:
                        memory_obs[img_key] = load_and_preprocess_video(np.stack(mem_frames))

            # States at memory indices
            mem_states = torch.stack([self._states[traj_idx][mi] for mi in memory_indices])
            if self.normalize:
                mem_states = (mem_states - self.norm_dict['states']['mean']) / self.norm_dict['states']['std']
            memory_obs['states'] = mem_states

            result['memory'] = memory_obs

        if self.return_language:
            result['task'] = {
                'text': self._texts[traj_idx],
                'features': self._text_features[traj_idx],
            }

        # Unload latents if not caching (states/actions remain cached)
        if not self.cache_trajectories:
            traj.unload()

        return result


def load_and_preprocess_video(images: np.ndarray) -> torch.Tensor:
    """Preprocess video frames: normalize and rearrange to (T, C, H, W)"""
    images = images.astype(np.float32) / 255.0
    images = rearrange(images, 't h w c -> t c h w')
    return torch.tensor(images).contiguous()


def compute_norm_stats(
    data_root: str,
    output_path: Optional[str] = None,
    max_trajectories: Optional[int] = None,
    relabel_actions: bool = True,
) -> Dict:
    """
    Compute normalization statistics (mean, std) for states and actions.

    Args:
        data_root: Path to preprocessed DROID dataset
        output_path: Path to save norm_stats.json (defaults to data_root/norm_stats_{relabel|recorded}.json)
        max_trajectories: Limit number of trajectories for faster computation
        relabel_actions: If True, compute stats for relabeled actions (state differences).
                        If False, compute stats for recorded actions (summed over frame skip).

    Returns:
        Dictionary with normalization statistics
    """
    if output_path is None:
        suffix = 'relabel' if relabel_actions else 'recorded'
        output_path = os.path.join(data_root, f'norm_stats_{suffix}.json')

    # Collect all states and actions
    all_states = []
    all_actions = []

    # Load annotations for train split
    anno_dir = Path(data_root) / "annotations/train"
    if not anno_dir.exists():
        raise ValueError(f"Annotation directory not found: {anno_dir}")

    anno_files = sorted(anno_dir.glob("*.json"))
    if max_trajectories:
        anno_files = anno_files[:max_trajectories]

    print(f"Computing normalization stats from {len(anno_files)} trajectories...")
    print(f"Action mode: {'relabeled (state differences)' if relabel_actions else 'recorded (summed)'}")

    for anno_file in tqdm(anno_files):
        with open(anno_file, 'r') as f:
            annotation = json.load(f)

        # Get joint states (joint_position + gripper_position)
        joint_position = np.array(annotation['observation.state.joint_position'])
        gripper_position = np.array(annotation['observation.state.gripper_position'])[:, None]
        full_states = np.concatenate([joint_position, gripper_position], axis=-1)

        # Compute rgb_skip from annotation
        video_length = annotation['video_length']
        raw_length = annotation['raw_length']
        traj_rgb_skip = raw_length // video_length

        # Sample states at video frame rate
        state_indices = np.arange(0, raw_length, traj_rgb_skip)
        states = full_states[state_indices]
        all_states.append(states)

        if relabel_actions:
            # Compute actions as state differences
            next_state_indices = np.clip(state_indices + traj_rgb_skip, 0, len(full_states) - 1)
            actions = full_states[next_state_indices] - full_states[state_indices]
        else:
            # Use recorded actions - sum actions over each rgb_skip interval
            action_joint = np.array(annotation['action.joint_position'])
            action_gripper = np.array(annotation['action.gripper_position'])[:, None]
            full_actions = np.concatenate([action_joint, action_gripper], axis=-1)

            actions = []
            for i in state_indices:
                end_action_idx = min(i + traj_rgb_skip, len(full_actions))
                action_sum = full_actions[i:end_action_idx].sum(axis=0)
                actions.append(action_sum)
            actions = np.stack(actions, axis=0)

        all_actions.append(actions)

    # Concatenate all data
    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)

    # Compute statistics
    norm_stats = {
        'norm_stats': {
            'state': {
                'mean': all_states.mean(axis=0).tolist(),
                'std': all_states.std(axis=0).tolist(),
            },
            'actions': {
                'mean': all_actions.mean(axis=0).tolist(),
                'std': all_actions.std(axis=0).tolist(),
            }
        }
    }

    print(f"State shape: {all_states.shape}")
    print(f"State mean: {norm_stats['norm_stats']['state']['mean']}")
    print(f"State std: {norm_stats['norm_stats']['state']['std']}")
    print(f"Action shape: {all_actions.shape}")
    print(f"Action mean: {norm_stats['norm_stats']['actions']['mean']}")
    print(f"Action std: {norm_stats['norm_stats']['actions']['std']}")

    # Save to file
    with open(output_path, 'w') as f:
        json.dump(norm_stats, f, indent=2)
    print(f"Saved normalization stats to {output_path}")

    return norm_stats


def encode_text_features(
    data_root: str,
    text_encoder: torch.nn.Module,
    splits: List[str] = ['train', 'val'],
    batch_size: int = 64,
) -> None:
    """
    Encode text instructions to latent features and save them to the annotations.
    Processes and saves incrementally to avoid OOM issues with large datasets.

    Args:
        data_root: Path to preprocessed DROID dataset
        text_encoder: Text encoder model (e.g., ClipEncoder)
        splits: Which splits to process
        batch_size: Batch size for text encoding
    """
    for split in splits:
        anno_dir = Path(data_root) / f"annotations/{split}"
        if not anno_dir.exists():
            print(f"Skipping {split} split - directory not found: {anno_dir}")
            continue

        anno_files = sorted(anno_dir.glob("*.json"))
        print(f"Processing {len(anno_files)} {split} trajectories...")

        # Collect files that need processing
        files_to_process = []
        for anno_file in anno_files:
            with open(anno_file, 'r') as f:
                annotation = json.load(f)
            if 'text_features' not in annotation:
                files_to_process.append(anno_file)

        if not files_to_process:
            print(f"All {split} trajectories already have text features, skipping...")
            continue

        print(f"Encoding text features for {len(files_to_process)} trajectories...")

        # Process in batches and save incrementally
        with torch.no_grad():
            for i in tqdm(range(0, len(files_to_process), batch_size)):
                batch_files = files_to_process[i:i+batch_size]

                # Load texts for this batch
                batch_texts = []
                batch_annotations = []
                for anno_file in batch_files:
                    with open(anno_file, 'r') as f:
                        annotation = json.load(f)
                    batch_texts.append(annotation['texts'][0])
                    batch_annotations.append(annotation)

                # Encode batch
                features = text_encoder(batch_texts)  # (B, feature_dim)
                features = features.cpu().numpy()

                # Save immediately to avoid accumulating in memory
                for anno_file, annotation, feat in zip(batch_files, batch_annotations, features):
                    annotation['text_features'] = feat.tolist()
                    with open(anno_file, 'w') as f:
                        json.dump(annotation, f, indent=2)

        print(f"Saved text features for {len(files_to_process)} {split} trajectories")


def convert_pt_to_npy(
    data_root: str,
    splits: List[str] = ['train', 'val'],
    delete_pt: bool = False,
) -> None:
    """
    Convert latent .pt files to .npy format for faster loading.

    Args:
        data_root: Path to preprocessed DROID dataset
        splits: Which splits to process
        delete_pt: If True, delete original .pt files after conversion
    """
    for split in splits:
        anno_dir = Path(data_root) / f"annotations/{split}"
        if not anno_dir.exists():
            print(f"Skipping {split} split - directory not found: {anno_dir}")
            continue

        anno_files = sorted(anno_dir.glob("*.json"))
        print(f"Converting {len(anno_files)} {split} trajectories from .pt to .npy...")

        converted = 0
        skipped = 0

        for anno_file in tqdm(anno_files):
            with open(anno_file, 'r') as f:
                annotation = json.load(f)

            pt_path = os.path.join(data_root, annotation['latent_path'])
            npy_path = pt_path.replace('.pt', '.npy')

            # Skip if already converted
            if os.path.exists(npy_path):
                skipped += 1
                continue

            if not os.path.exists(pt_path):
                print(f"Warning: {pt_path} not found, skipping")
                continue

            # Load .pt and save as .npy
            latents = torch.load(pt_path, weights_only=True).numpy()
            np.save(npy_path, latents)
            converted += 1

            # Optionally delete original .pt file
            if delete_pt:
                os.remove(pt_path)

            # Update annotation to point to .npy file
            annotation['latent_path'] = annotation['latent_path'].replace('.pt', '.npy')
            with open(anno_file, 'w') as f:
                json.dump(annotation, f, indent=2)

        print(f"Converted {converted} files, skipped {skipped} (already exist)")


def convert_videos_to_npy(
    data_root: str,
    splits: List[str] = ['train', 'val'],
    delete_video: bool = False,
) -> None:
    """
    Convert video files to .npy format for fast memory-mapped loading.

    This preprocessing step converts MP4 videos to numpy arrays, enabling:
    - Memory-mapped loading (only requested frames are read from disk)
    - No video decoding overhead at training time
    - Efficient random access for chunked sampling

    The frames are saved in shape (num_cameras, T, H, W, C) to match the latent format.

    Args:
        data_root: Path to preprocessed DROID dataset
        splits: Which splits to process
        delete_video: If True, delete original video files after conversion
    """
    for split in splits:
        anno_dir = Path(data_root) / f"annotations/{split}"
        if not anno_dir.exists():
            print(f"Skipping {split} split - directory not found: {anno_dir}")
            continue

        anno_files = sorted(anno_dir.glob("*.json"))
        print(f"Converting {len(anno_files)} {split} videos to .npy format...")

        converted = 0
        skipped = 0
        errors = 0

        for anno_file in tqdm(anno_files):
            try:
                with open(anno_file, 'r') as f:
                    annotation = json.load(f)

                video_path = os.path.join(data_root, annotation['video_path'])
                npy_path = video_path.replace('.mp4', '_frames.npy')

                # Skip if already converted
                if os.path.exists(npy_path):
                    skipped += 1
                    continue

                if not os.path.exists(video_path):
                    print(f"  Warning: {video_path} not found, skipping")
                    errors += 1
                    continue

                # Read entire video
                num_cameras = annotation['num_cameras']
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"  Warning: Cannot open {video_path}")
                    errors += 1
                    continue

                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()

                if not frames:
                    print(f"  Warning: No frames in {video_path}")
                    errors += 1
                    continue

                # Stack frames: (T, H, W*num_cameras, C)
                stacked_video = np.stack(frames, axis=0)

                # Split by camera and rearrange to (num_cameras, T, H, W, C)
                camera_width = stacked_video.shape[2] // num_cameras
                video_by_camera = []
                for i in range(num_cameras):
                    start_w = i * camera_width
                    end_w = (i + 1) * camera_width
                    video_by_camera.append(stacked_video[:, :, start_w:end_w, :])

                # Shape: (num_cameras, T, H, W, C)
                video_array = np.stack(video_by_camera, axis=0)

                # Save as numpy array
                np.save(npy_path, video_array)

                # Update annotation
                annotation['video_frames_path'] = annotation['video_path'].replace('.mp4', '_frames.npy')
                with open(anno_file, 'w') as f:
                    json.dump(annotation, f, indent=2)

                # Optionally delete original video
                if delete_video:
                    os.remove(video_path)

                converted += 1

            except Exception as e:
                print(f"  Error processing {anno_file}: {e}")
                errors += 1

        print(f"Converted {converted} videos, skipped {skipped}, errors {errors}")


def encode_sd3_features(
    data_root: str,
    splits: List[str] = ['train', 'val'],
    batch_size: int = 8,
    image_size: Tuple[int, int] = (192, 320),
    device: str = 'cuda',
    chunks: int = 1,
    chunk_id: int = 0,
    use_fp16: bool = True,
) -> None:
    """
    Encode video frames to SD3 VAE latent features and save as compressed .npz files.

    This preprocesses videos using the SD3Encoder (AutoencoderKL from SD3)
    and saves the latent features with '_sd3.npz' suffix to distinguish
    from SVD encoder features.

    Supports parallel processing by splitting trajectories into chunks.

    Args:
        data_root: Path to preprocessed DROID dataset
        splits: Which splits to process
        batch_size: Batch size for encoding (reduce if OOM)
        image_size: Image size (H, W) for the encoder
        device: Device to run encoding on
        chunks: Total number of chunks to split data into (for parallel processing)
        chunk_id: Which chunk to process (0-indexed, must be < chunks)
        use_fp16: If True, save latents as float16 (2x smaller). Default True.
    """
    from encoders import SD3Encoder

    # Initialize SD3 encoder
    print("Loading SD3 encoder...")
    encoder = SD3Encoder(
        model_name="stabilityai/stable-diffusion-3-medium-diffusers",
        image_size=image_size,
        spatial_size=4,
        device=device,
    )
    encoder.eval()
    print(f"SD3 encoder loaded. Scaling factor: {encoder.scaling_factor}, Shift factor: {encoder.shift_factor}")

    for split in splits:
        anno_dir = Path(data_root) / f"annotations/{split}"
        if not anno_dir.exists():
            print(f"Skipping {split} split - directory not found: {anno_dir}")
            continue

        anno_files = sorted(anno_dir.glob("*.json"))
        total_files = len(anno_files)

        # Split into chunks for parallel processing
        if chunks > 1:
            chunk_size = (total_files + chunks - 1) // chunks  # Ceiling division
            start_idx = chunk_id * chunk_size
            end_idx = min(start_idx + chunk_size, total_files)
            anno_files = anno_files[start_idx:end_idx]
            print(f"Processing chunk {chunk_id}/{chunks}: trajectories {start_idx}-{end_idx} of {total_files} {split} trajectories...")
        else:
            print(f"Processing {total_files} {split} trajectories...")

        encoded = 0
        skipped = 0
        errors = 0

        for anno_file in tqdm(anno_files):
            try:
                with open(anno_file, 'r') as f:
                    annotation = json.load(f)

                # Determine output path (handle both .npy and .pt extensions)
                ## if path not in annotation, create one with latents/id.pt 
                if 'latent_path' not in annotation:
                    annotation['latent_path'] = f"latents/val/{annotation['episode_id']}.npy"
                    ## make the key latent_path appears earlier in the file next to video_path 
                    
                    with open(anno_file, 'w') as f:
                        json.dump(annotation, f, indent=2)
                latent_path = os.path.join(data_root, annotation['latent_path'])
                if latent_path.endswith('.npy'):
                    sd3_latent_path = latent_path.replace('.npy', '_sd3.npz')
                else:
                    sd3_latent_path = latent_path.replace('.pt', '_sd3.npz')

                if os.path.exists(sd3_latent_path):
                    skipped += 1
                    continue

                # Load video frames from MP4
                num_cameras = annotation['num_cameras']
                video_path = os.path.join(data_root, annotation['video_path'])
                frames = _load_video_as_array(video_path, num_cameras)

                if frames is None:
                    print(f"  Warning: Could not load frames for {anno_file}")
                    errors += 1
                    continue

                # Encode each camera's frames
                all_camera_latents = []

                for cam_idx in range(num_cameras):
                    # Get frames for this camera: (T, H, W, C)
                    cam_frames = frames[cam_idx]
                    T = cam_frames.shape[0]

                    # Preprocess: normalize to [0, 1] and rearrange to (T, C, H, W)
                    cam_frames = cam_frames.astype(np.float32) / 255.0
                    cam_frames = rearrange(cam_frames, 't h w c -> t c h w')
                    cam_frames = torch.from_numpy(cam_frames).to(device)

                    # Encode in batches
                    latents_list = []
                    for i in range(0, T, batch_size):
                        batch = cam_frames[i:i+batch_size]
                        with torch.no_grad():
                            latents = encoder(batch)  # (B, N, D)
                        latents_list.append(latents.cpu().numpy())

                    # Concatenate all batches: (T, N, D)
                    cam_latents = np.concatenate(latents_list, axis=0)
                    all_camera_latents.append(cam_latents)

                # Stack all cameras: (num_cameras, T, N, D)
                stacked_latents = np.stack(all_camera_latents, axis=0)

                # Convert to fp16 if requested (2x smaller)
                if use_fp16:
                    stacked_latents = stacked_latents.astype(np.float16)

                # Save as compressed npz
                np.savez_compressed(sd3_latent_path, latents=stacked_latents)
                encoded += 1

            except Exception as e:
                print(f"  Error processing {anno_file}: {e}")
                import traceback
                traceback.print_exc()
                errors += 1

        print(f"Encoded {encoded} trajectories, skipped {skipped}, errors {errors}")


def _load_video_as_array(video_path: str, num_cameras: int) -> Optional[np.ndarray]:
    """Load video file and return as numpy array.

    Args:
        video_path: Path to video file
        num_cameras: Number of cameras stacked horizontally

    Returns:
        Array of shape (num_cameras, T, H, W, C) or None if failed
    """
    if not os.path.exists(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        return None

    # Stack frames: (T, H, W*num_cameras, C)
    stacked_video = np.stack(frames, axis=0)

    # Split by camera: (num_cameras, T, H, W, C)
    camera_width = stacked_video.shape[2] // num_cameras
    video_by_camera = []
    for i in range(num_cameras):
        start_w = i * camera_width
        end_w = (i + 1) * camera_width
        video_by_camera.append(stacked_video[:, :, start_w:end_w, :])

    return np.stack(video_by_camera, axis=0)


# Example usage and testing
if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to preprocessed DROID dataset')
    parser.add_argument('--mode', type=str, default='test',
                       choices=['test', 'compute_norm_stats', 'encode_text', 'convert_to_npy', 'convert_videos', 'encode_sd3'],
                       help='Mode: test dataset, compute norm stats, encode text, convert pt to npy, convert videos to npy, or encode sd3 features')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'val', 'valid'])
    parser.add_argument('--horizon', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--max_trajectories', type=int, default=None)
    parser.add_argument('--relabel_actions', action='store_true', default=True,
                       help='Use relabeled actions (state differences)')
    parser.add_argument('--no_relabel_actions', dest='relabel_actions', action='store_false',
                       help='Use recorded actions (summed over frame skip)')
    parser.add_argument('--chunks', type=int, default=1,
                       help='Total number of chunks for parallel processing')
    parser.add_argument('--chunk_id', type=int, default=0,
                       help='Which chunk to process (0-indexed)')
    parser.add_argument('--use_fp16', action='store_true', default=True,
                       help='Save SD3 latents as float16 (2x smaller)')
    parser.add_argument('--no_fp16', dest='use_fp16', action='store_false',
                       help='Save SD3 latents as float32')
    args = parser.parse_args()

    if args.mode == 'compute_norm_stats':
        # Compute normalization statistics
        compute_norm_stats(
            data_root=args.data_root,
            max_trajectories=args.max_trajectories,
            relabel_actions=args.relabel_actions,
        )

    elif args.mode == 'encode_text':
        # Encode text features
        from encoders import get_task_encoder
        text_encoder = get_task_encoder(config={}, device='cuda')
        text_encoder = text_encoder.to('cuda')

        encode_text_features(
            data_root=args.data_root,
            text_encoder=text_encoder,
            splits=['train', 'val'],
            batch_size=args.batch_size,
        )

    elif args.mode == 'convert_to_npy':
        # Convert .pt latent files to .npy for faster loading
        convert_pt_to_npy(
            data_root=args.data_root,
            splits=['train', 'val'],
            delete_pt=False,  # Set to True to delete original .pt files
        )

    elif args.mode == 'convert_videos':
        # Convert video files to .npy for fast memory-mapped loading
        convert_videos_to_npy(
            data_root=args.data_root,
            splits=['train', 'val'],
            delete_video=False,  # Set to True to delete original videos
        )

    elif args.mode == 'encode_sd3':
        # Encode video frames with SD3 encoder
        encode_sd3_features(
            data_root=args.data_root,
            splits=[args.split],
            batch_size=args.batch_size,
            image_size=(192, 320),
            device='cuda',
            chunks=args.chunks,
            chunk_id=args.chunk_id,
            use_fp16=args.use_fp16,
        )

    else:
        # Test dataset loading
        print("Creating DROID dataset...")
        dataset = PrecomputedDroid(
            root=args.data_root,
            split=args.split,
            horizon=args.horizon,
            img_keys=['exterior_1_left', 'wrist_left', 'exterior_2_left'],
            relabel_actions=args.relabel_actions,
            normalize=True,
            cache_trajectories=False,
            return_language=True,
            return_video_frames=True,
            max_trajectories=args.max_trajectories,
        )

        print(f"\nDataset size: {len(dataset)} trajectories")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
    )

    print(f"Dataloader will provide {len(dataloader)} batches per epoch")

    # Test loading
    print("\nTesting data loading...")
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Observation keys: {list(batch['obs'].keys())}")

        for key, value in batch['obs'].items():
            if key != 'states':
                print(f"  {key} shape: {value.shape}")      

        print(f"  States shape: {batch['obs']['states'].shape}")
        print(f"  Actions shape: {batch['actions'].shape}")
        print(f"  Rewards shape: {batch['rewards'].shape}")

        if 'task' in batch:
            print(f"  Language instructions: {batch['task']['text'][:2]}")

        print(f"  Mean states: {batch['obs']['states'].mean((0, 1))}")
        print(f"  Mean actions: {batch['actions'].mean((0, 1))}")

        if batch_idx >= 2:  # Show first 3 batches
            break

    print("\nDataset loading successful!")
