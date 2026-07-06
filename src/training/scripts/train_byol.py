# --- Deterministic DataLoader worker seed function (must be top-level for pickling) ---
def seed_worker(worker_id):
    # Use torch initial seed for reproducibility
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class DeterministicRNG:
    def __init__(self, seed: int):
        self.gen = torch.Generator(device='cpu')
        self.gen.manual_seed(seed)

    def rand(self, *shape):
        return torch.rand(*shape, generator=self.gen)

    def randn_like(self, x):
        return torch.randn(x.shape, generator=self.gen, device='cpu').to(x.device)

    def randint(self, low, high, shape):
        return torch.randint(low, high, shape, generator=self.gen)
"""
BYOL (Bootstrap Your Own Latent) self-supervised training script.

This script reads `config/model_config.yml`, creates dataloaders using
`load_config_and_create_dataloaders`, applies the same normalization pipeline
used in MAE scripts, trains BYOL (momentum encoder with two augmented views), and
stores extracted features from the best checkpoint.

Key behavior:
- Keeps time as batch by reshaping to (B*T, C, H, W)
- Uses BYOL loss (MSE between normalized projections with stop-gradient)
- Momentum encoder updates via EMA
- Flexible embedding dimensions (default: embed_dim=2048, proj_dim=128)

Usage:
    python src/training/scripts/train_byol.py --config config/model_config.yml
"""

import argparse
import math
import random
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
import numpy as np
from tqdm import tqdm

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Add src to path so project imports work when running from repo root
current_file = Path(__file__).resolve()
scripts_dir = current_file.parent
training_dir = scripts_dir.parent
src_dir = training_dir.parent
sys.path.insert(0, str(src_dir))

from data_pipeline.loaders.utils import load_config_and_create_dataloaders
from data_pipeline.preprocessing.normstats.compute_stats import compute_normalization_stats
from data_pipeline.preprocessing.normstats.stats_manager import (
    load_normalization_stats,
    save_normalization_stats,
    stats_exist,
)
from data_pipeline.preprocessing.transformers import NormalizeWithPrecomputedStats
from models.architectures.byol_2 import DinoBYOLCNN
from models.architectures.patch_transformer import ClimateTransformerBackbone


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/model_config_new_45.yml')
    parser.add_argument('--device', type=str, default='cuda:1' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model-name', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--embed-dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--proj-dim', type=int, default=1024, help='Projection dimension')
    parser.add_argument('--momentum-coeff', type=float, default=0.9, help='EMA coefficient for momentum encoder')
    parser.add_argument('--noise-std', type=float, default=0.1, help='If None, uses config augmentation gaussian noise std')
    
    # Augmentation parameters
    parser.add_argument('--spatial-mask', type=float, default=0.4, help='Spatial masking probability')
    parser.add_argument('--channel-dropout', type=float, default=0.0, help='Channel dropout probability')
    parser.add_argument('--blur-prob', type=float, default=0.3, help='Blur probability')
    parser.add_argument('--blur-sigma', type=float, default=0.1, help='Blur sigma')
    parser.add_argument('--crop-prob', type=float, default=0.5, help='Random crop probability')
    parser.add_argument('--crop-scale-min', type=float, default=0.6, help='Minimum crop scale')
    parser.add_argument('--crop-scale-max', type=float, default=1.0, help='Maximum crop scale')

    parser.add_argument(
        '--force-recompute-stats',
        action='store_true',
        help='Force recomputation of normalization statistics even if they exist'
    )
    parser.add_argument(
        '--use-lr-scheduler',
        action='store_true',
        default=True,
        help='Use cosine annealing learning rate scheduler'
    )
    parser.add_argument(
        '--monitor-collapse',
        action='store_true',
        default=True,
        help='Monitor for mode collapse in embeddings'
    )
    return parser.parse_args()

def byol_loss(pred, target):

    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)

    return 2 - 2 * (pred * target).sum(dim=-1).mean()

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def log_run_hyperparams(args: argparse.Namespace, model_name: str) -> Path:
    """Append run args to a per-model YAML log file."""
    output_dir = Path('/home/armaan/MoK_orig_finer/results/hyperparams')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{model_name}_config.yml"
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    run_payload = {
        'timestamp_ist': datetime.now(ist_tz).isoformat(timespec='seconds'),
        'model_name': model_name,
        'args': vars(args),
    }

    existing = {}
    if output_path.exists():
        with open(output_path, 'r') as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                existing = loaded

    runs = existing.get('runs', []) if isinstance(existing, dict) else []
    if not isinstance(runs, list):
        runs = []
    runs.append(run_payload)

    payload = {
        'model_name': model_name,
        'runs': runs,
    }

    with open(output_path, 'w') as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    return output_path

def compute_alignment(pred, target):
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return (pred * target).sum(dim=-1).mean()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False
    cudnn.enabled = False

    # For full determinism (PyTorch >= 1.8)
    torch.use_deterministic_algorithms(True)


def cosine_schedule(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return end
    progress = step / (total_steps - 1)
    return end - (end - start) * 0.5 * (1.0 + math.cos(math.pi * progress))


class BYOLAugmentation:
    """
    BYOL-style augmentation for climate data to create diverse augmented views.
    """
    def __init__(
        self,
        noise_std: float = 0.1,
        spatial_mask_prob: float = 0.15,
        channel_dropout_prob: float = 0.1,
        blur_prob: float = 0.5,
        blur_sigma: float = 1.0,
        crop_prob: float = 0.0,
        crop_scale_min: float = 0.7,
        crop_scale_max: float = 1.0,
        rng = None
    ):
        self.noise_std = noise_std
        self.spatial_mask_prob = spatial_mask_prob
        self.channel_dropout_prob = channel_dropout_prob
        self.blur_prob = blur_prob
        self.blur_sigma = blur_sigma
        self.crop_prob = crop_prob
        self.crop_scale_min = crop_scale_min
        self.crop_scale_max = crop_scale_max
        self.rng = rng

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations to input tensor.
        
        Args:
            x: Input tensor of shape (B*T, C, H, W)
        
        Returns:
            Augmented tensor of same shape
        """
        # 1. Random crop (applied first to maintain spatial coherence)
        device = x.device
        if self.crop_prob > 0 and self.rng.rand(1).item() < self.crop_prob:
            x = self._random_crop(x)
        
        # 2. Gaussian noise
        if self.noise_std > 0:
            x = x + self.rng.randn_like(x) * self.noise_std
        
        # 3. Random spatial masking (mask patches)
        if self.spatial_mask_prob > 0 and self.rng.rand(1).item() < 0.5:
            x = self._spatial_masking(x)
        
        # 4. Random channel dropout
        if self.channel_dropout_prob > 0 and self.rng.rand(1).item() < 0.5:
            x = self._channel_dropout(x)
        
        # 5. Gaussian blur (spatial smoothing)
        if self.blur_prob > 0 and self.rng.rand(1).item() < self.blur_prob:
            x = self._gaussian_blur(x)
        
        return x
    
    def _spatial_masking(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly mask spatial patches."""
        BT, C, H, W = x.shape
        patch_size = 16
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        if num_patches_h == 0 or num_patches_w == 0:
            return x

        # Sample one Bernoulli decision per patch and upsample to pixel-space.
        patch_drop = (self.rng.rand(num_patches_h, num_patches_w) < self.spatial_mask_prob)
        patch_keep = (~patch_drop).to(device=x.device, dtype=x.dtype)

        spatial_mask = torch.ones((H, W), device=x.device, dtype=x.dtype)
        masked_h = num_patches_h * patch_size
        masked_w = num_patches_w * patch_size
        spatial_mask[:masked_h, :masked_w] = (
            patch_keep.repeat_interleave(patch_size, dim=0)
            .repeat_interleave(patch_size, dim=1)
        )

        return x * spatial_mask.view(1, 1, H, W)
    
    def _channel_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly drop entire channels."""
        BT, C, H, W = x.shape
        device = x.device
        mask = torch.ones((1, C, 1, 1), device=device)
        drop = self.rng.rand(C) < self.channel_dropout_prob
        mask[0, drop, 0, 0] = 0
        return x * mask
    
    def _gaussian_blur(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur for spatial smoothing."""
        kernel_size = 5
        sigma = self.blur_sigma
        
        # Create Gaussian kernel
        kernel_range = torch.arange(kernel_size, dtype=torch.float32, device=x.device) - kernel_size // 2
        kernel = torch.exp(-0.5 * (kernel_range / sigma) ** 2)
        kernel = kernel / kernel.sum()
        
        # 2D kernel (separable)
        kernel_2d = kernel.view(1, 1, -1, 1) * kernel.view(1, 1, 1, -1)
        kernel_2d = kernel_2d.repeat(x.shape[1], 1, 1, 1)
        
        # Apply depthwise convolution
        padding = kernel_size // 2
        x_blurred = F.conv2d(x, kernel_2d, padding=padding, groups=x.shape[1])
        
        return x_blurred
    
    def _random_crop(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random crop and resize back to original size.
        Creates different spatial views for BYOL.
        """
        BT, C, H, W = x.shape
        scale = self.rng.rand(1).item() * (self.crop_scale_max - self.crop_scale_min) + self.crop_scale_min
        crop_h = int(H * scale)
        crop_w = int(W * scale)
        max_top = H - crop_h
        max_left = W - crop_w
        top = int(self.rng.randint(0, max_top + 1, (1,)).item()) if max_top > 0 else 0
        left = int(self.rng.randint(0, max_left + 1, (1,)).item()) if max_left > 0 else 0
        x_cropped = x[:, :, top:top+crop_h, left:left+crop_w]
        x_resized = F.interpolate(
            x_cropped,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
        return x_resized


def detect_mode_collapse(embeddings: torch.Tensor, threshold_std: float = 0.01, threshold_cosine: float = 0.98) -> dict:
    """
    Detect mode collapse in embeddings by checking:
    1. Standard deviation (should be high)
    2. Mean cosine similarity (should be low for diversity)
    
    Args:
        embeddings: Tensor of shape (N, D)
        threshold_std: If std < threshold, likely mode collapse
        threshold_cosine: If mean cosine sim > threshold, likely mode collapse
    
    Returns:
        Dictionary with collapse metrics
    """
    with torch.no_grad():
        # Normalize embeddings
        emb_norm = F.normalize(embeddings, dim=-1)
        
        # Compute per-dimension statistics
        std_per_dim = embeddings.std(dim=0)
        mean_std = std_per_dim.mean().item()
        min_std = std_per_dim.min().item()
        max_std = std_per_dim.max().item()
        
        # Compute mean cosine similarity (sample for efficiency)
        sample_size = embeddings.shape[0]
        indices = torch.arange(sample_size)
        emb_sampled = emb_norm[indices]
        
        # Compute pairwise cosine similarity
        cosine_sim = torch.mm(emb_sampled, emb_sampled.t())
        # Zero out diagonal
        cosine_sim.fill_diagonal_(0)
        mean_cosine = cosine_sim.mean().item()
        
        # Check for collapse
        is_collapsed = mean_std < threshold_std or mean_cosine > threshold_cosine
        
        return {
            'mean_std': mean_std,
            'min_std': min_std,
            'max_std': max_std,
            'mean_cosine_sim': mean_cosine,
            'is_collapsed': is_collapsed,
        }


def resolve_num_time_steps(data_cfg: dict) -> int:
    raw_num_time_steps = data_cfg.get('num_time_steps', None)
    if raw_num_time_steps is not None:
        if isinstance(raw_num_time_steps, list):
            value = len(raw_num_time_steps)
        else:
            value = int(raw_num_time_steps)
        if value <= 0:
            raise ValueError(f"Invalid num_time_steps={value}. It must be > 0.")
        return value

    raw_time_steps = data_cfg.get('time_steps', None)
    if isinstance(raw_time_steps, list):
        value = len(raw_time_steps)
        if value <= 0:
            raise ValueError("time_steps list is empty; cannot infer num_time_steps.")
        return value

    if raw_time_steps is not None:
        value = int(raw_time_steps)
        if value <= 0:
            raise ValueError(f"Invalid time_steps={value}. It must be > 0.")
        return value

    return 1


def prepare_input_btchw(
    data: torch.Tensor,
    device: torch.device,
    normalize_transform=None,
    num_time_steps: int = 1,
) -> Tuple[torch.Tensor, int, int]:
    """
    Convert batch tensor into (B*T, C, H, W) while keeping time as batch.
    Applies normalization per-sample before temporal flattening.

    Args:
        data: Input tensor
        device: Target device
        normalize_transform: Optional normalization transform
        num_time_steps: Number of time steps (for 4D input to calculate channels)

    Returns:
        x_btchw: Tensor of shape (B*T, C, H, W)
        B: original batch size
        T: original temporal length (1 if not temporal)
    """
    if data.dim() == 3:
        # (B, H, W) -> (B, 1, H, W); T=1
        B = data.shape[0]
        T = 1
        x = data.unsqueeze(1)
        x = x.float().to(device)
        if normalize_transform is not None:
            normed = [normalize_transform(x[i]) for i in range(x.shape[0])]
            x = torch.stack(normed, dim=0)
        return x, B, T

    if data.dim() == 4:
        # (B, T*C, H, W) where T*C is loaded, divide by num_time_steps to get C
        B, TC, H, W = data.shape
        T = num_time_steps
        
        x = data.float().to(device)
        
        # Handle lat/lon channel replication if needed
        if TC % T != 0:
            # Check if last 2 channels are lat/lon that need replication
            base_channels = TC - 2
            if base_channels > 0 and base_channels % T == 0:
                # Lat/lon case: replicate lat/lon for each time step
                # Split into base temporal channels and lat/lon spatial channels
                x_base = x[:, :base_channels, :, :]  # (B, base_channels, H, W)
                x_latlon = x[:, base_channels:, :, :]  # (B, 2, H, W)
                
                # Reshape base: (B, base_channels, H, W) -> (B, T, C_base, H, W)
                C_base = base_channels // T
                x_base = x_base.reshape(B, T, C_base, H, W)
                
                # Replicate lat/lon for each time step: (B, 2, H, W) -> (B, T, 2, H, W)
                x_latlon = x_latlon.unsqueeze(1).expand(B, T, -1, H, W)
                
                # Combine: (B, T, C_base + 2, H, W)
                x = torch.cat([x_base, x_latlon], dim=2)
                
                if normalize_transform is not None:
                    # Apply normalization to each (T*C, H, W) sample before reshaping
                    # Reshape temporarily back to (B, T*C, H, W)
                    x_temp = x.reshape(B, -1, H, W)
                    normed = [normalize_transform(x_temp[i]) for i in range(B)]
                    x_temp = torch.stack(normed, dim=0)
                    # Reshape back to (B, T, C, H, W)
                    x = x_temp.reshape(B, T, -1, H, W)
                
                # Reshape: (B, T, C, H, W) -> (B*T, C, H, W)
                C = C_base + 2
                x = x.reshape(B * T, C, H, W)
                return x, B, T
            else:
                raise ValueError(
                    f"Cannot reshape: dimension {TC} is not divisible by num_time_steps={T}, "
                    f"and lat/lon replication logic cannot apply (base_channels={base_channels})"
                )
        
        # Normal case: TC is divisible by T
        C = TC // T
        if normalize_transform is not None:
            # Apply normalization to each (T*C, H, W) sample
            normed = [normalize_transform(x[i]) for i in range(B)]
            x = torch.stack(normed, dim=0)
        # Reshape for [chan0_t1, ..., chanC_tN] layout:
        # (B, T*C, H, W) -> (B, C, T, H, W) -> (B, T, C, H, W) -> (B*T, C, H, W)
        x = x.reshape(B, C, T, H, W).permute(0, 2, 1, 3, 4)
        x = x.reshape(B * T, C, H, W)
        return x, B, T

    if data.dim() == 5:
        # (B, T, C, H, W) -> (B*T, C, H, W)
        B, T, C, H, W = data.shape
        x = data.float().to(device).reshape(B * T, C, H, W)
        if normalize_transform is not None:
            normed = [normalize_transform(x[i]) for i in range(x.shape[0])]
            x = torch.stack(normed, dim=0)
        return x, B, T

    raise ValueError(
        f"Unexpected data shape: {data.shape}. Expected (B,H,W), (B,T*C,H,W), or (B,T,C,H,W)."
    )

def covariance_loss(z):
    z = z - z.mean(dim=0)
    N, D = z.shape
    cov = (z.T @ z) / (N - 1)

    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag ** 2).sum() / (D * D)

def main():

    args = parse_args()
    set_seed(args.seed)

    config = load_config(args.config)

    model_name = args.model_name or config.get('model', {}).get('name', 'byol_cnn')
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs

    hyperparam_log_path = log_run_hyperparams(args, model_name)
    print(f"✓ Logged run args to: {hyperparam_log_path}")

    device = torch.device(args.device)


    # --- Deterministic DataLoader setup ---

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader_no_norm, val_loader, test_loader = load_config_and_create_dataloaders(
        config_path=args.config,
        dataloader_kwargs={
            'worker_init_fn': seed_worker,
            'generator': g,
            'num_workers': 0,
            'persistent_workers': False,
        }
    )

    data_cfg = config.get('data', {})
    normalize_strategy = data_cfg.get('normalize_strategy', 1)
    add_per_channel_norm = data_cfg.get('add_per_channel_norm', True)

    print("\n" + "=" * 80)
    print("Normalization Strategy")
    print("=" * 80)
    print(f"normalize_strategy: {normalize_strategy}")
    print(f"add_per_channel_norm: {add_per_channel_norm}")

    normalize_transform = None

    if normalize_strategy == 1:
        print("Strategy: Normalize using training data statistics (spatially-varying)")

        force_recompute = True
        stats_already_exist = stats_exist(model_name)

        if force_recompute and stats_already_exist:
            print("⚠ Forcing recomputation of normalization statistics (--force-recompute-stats flag set)")
            print(f"   Existing statistics for '{model_name}' will be overwritten.")

        if stats_already_exist and not force_recompute:
            print(f"Loading existing normalization statistics for '{model_name}'...")
            norm_stats = load_normalization_stats(model_name, verbose=True)
        else:
            if not stats_already_exist:
                print(f"Computing normalization statistics from training data for '{model_name}'...")
            else:
                print(f"Recomputing normalization statistics from training data for '{model_name}'...")
            norm_stats = compute_normalization_stats(
                train_loader=train_loader_no_norm,
                model_name=model_name,
                device=device,
                verbose=True,
            )
            save_normalization_stats(norm_stats, verbose=True)

        normalize_transform = NormalizeWithPrecomputedStats(
            mean=norm_stats.mean.to(device),
            std=norm_stats.std.to(device),
            static_channel_indices=norm_stats.static_channel_indices,
        )
        print("✓ Normalization transform created (precomputed stats)")

        if add_per_channel_norm:
            from data_pipeline.preprocessing.transformers import Compose, PerChannelMinMaxNormalize

            per_channel_transform = PerChannelMinMaxNormalize(
                static_channel_indices=norm_stats.static_channel_indices
            )
            normalize_transform = Compose([normalize_transform, per_channel_transform])
            print("✓ Added per-channel min-max normalization [0, 1]")

    elif normalize_strategy == 0:
        if add_per_channel_norm:
            print("Strategy: Per-channel min-max normalization only (no precomputed stats)")
            normalize_transform = None
        else:
            print("Strategy: No normalization (using raw data)")
            normalize_transform = None
    else:
        raise ValueError(
            f"Unsupported normalize_strategy: {normalize_strategy}. "
            f"Supported values: 0 (no normalization), 1 (training data stats)"
        )

    # For BYOL with temporal flattening, use the existing dataloaders without dataset-level transforms
    # Normalization will be applied per-batch in prepare_input_btchw
    train_loader = train_loader_no_norm
    print(f"✓ Using dataloaders (normalization applied per-batch for temporal handling)")

    # Get num_time_steps from config to calculate actual channels
    num_time_steps = resolve_num_time_steps(data_cfg)
    print(f"\nnum_time_steps from config: {num_time_steps}")

    # Determine number of channels
    sample = next(iter(train_loader))[0]
    print(f"Sample shape from dataloader: {sample.shape}")
    
    if sample.dim() == 3:
        in_channels = 1
    elif sample.dim() == 4:
        # (B, T*C, H, W) where dim[1] = T*C, so C = T*C / T
        # Handle lat/lon channels that are replicated across all time steps
        TC = sample.shape[1]
        if TC % num_time_steps == 0:
            # Normal case: divisible by num_time_steps
            in_channels = TC // num_time_steps
            print(f"Calculated: {TC} (loaded) / {num_time_steps} (time steps) = {in_channels} channels")
        else:
            # Check for lat/lon case: last 2 channels are spatial, rest are temporal
            base_channels = TC - 2
            if base_channels > 0 and base_channels % num_time_steps == 0:
                # Lat/lon detected: base channels are temporal, 2 are spatial
                channels_per_step = base_channels // num_time_steps
                in_channels = channels_per_step + 2  # After replication
                print(f"✓ Detected lat/lon channels: {TC} = {base_channels} (temporal) + 2 (spatial)")
                print(f"Calculated: {base_channels} / {num_time_steps} = {channels_per_step} temporal channels")
                print(f"After replicating lat/lon for each timestep: {in_channels} channels per timestep")
            else:
                raise ValueError(
                    f"Invalid shape/config combination: sample channels={TC} not divisible by "
                    f"num_time_steps={num_time_steps}, and lat/lon replication logic cannot apply.\n"
                    f"Expected either: (1) TC % {num_time_steps} == 0, or "
                    f"(2) (TC-2) % {num_time_steps} == 0 for lat/lon handling."
                )
        if in_channels <= 0:
            raise ValueError(
                f"Inferred in_channels={in_channels} from TC={TC} and "
                f"num_time_steps={num_time_steps}; expected positive channel count."
            )
    elif sample.dim() == 5:
        in_channels = sample.shape[2]
    else:
        raise ValueError(
            f"Unexpected sample shape: {sample.shape}. Expected (B,H,W), (B,T*C,H,W), or (B,T,C,H,W)."
        )
    print(f"in_channels: {in_channels}")

    training_cfg = config.get('training', {})
    model_cfg = config.get('model', {})
    activation_cfg = model_cfg.get('activation', {})

    embed_dim = int(args.embed_dim)
    proj_dim = int(args.proj_dim)

    backbone_dim = int(model_cfg.get('backbone_dim', 1024))

    # Build patch-transformer backbone.
    spatial_h = int(sample.shape[-2])
    spatial_w = int(sample.shape[-1])
    backbone = ClimateTransformerBackbone(
        in_channels=in_channels,
        embed_dim=backbone_dim,
        patch_size=int(model_cfg.get('patch_size', 16)),
        depth=int(model_cfg.get('depth', 6)),
        num_heads=int(model_cfg.get('num_heads', 8)),
        mlp_ratio=float(model_cfg.get('mlp_ratio', 4.0)),
        input_h=spatial_h,
        input_w=spatial_w,
    )

    model = DinoBYOLCNN(
            backbone,
            backbone_dim=backbone_dim,
            embed_dim=embed_dim,
            proj_dim=proj_dim
        ).to(device)

    # Initialize LazyLinear modules with one forward pass
    with torch.no_grad():
        x_dummy, _, _ = prepare_input_btchw(sample, device, normalize_transform, num_time_steps=num_time_steps)
        print(f"After prepare_input_btchw: {x_dummy.shape} (expected: B*T, {in_channels}, H, W)")
        pred, teacher_proj, student_embed = model(x_dummy, x_dummy)
        print(f"Student embedding dimension: {student_embed.shape[-1]}")

    optimizer = optim.AdamW(
        list(model.student_encoder.parameters()) +
        list(model.predictor.parameters()),
        lr=training_cfg.get('learning_rate', 1e-4),
        weight_decay=training_cfg.get('weight_decay', 0.0),
    )

    epochs = training_cfg.get('epochs', 10)
    total_steps = max(1, epochs * max(1, len(train_loader)))
    
    # Create learning rate scheduler
    scheduler = None
    if args.use_lr_scheduler:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=training_cfg.get('learning_rate', 1e-4) * 0.01,
        )
        print(f"✓ Using CosineAnnealingLR scheduler (T_max={epochs})")

    augmentation_config = data_cfg.get('augmentation', {})
    gaussian_noise_config = augmentation_config.get('gaussian_noise', {})
    gaussian_noise_enabled = gaussian_noise_config.get('enabled', False)
    gaussian_noise_std = gaussian_noise_config.get('std', 0.01) if gaussian_noise_enabled else 0.0

    noise_std = args.noise_std
    if noise_std is None:
        noise_std = gaussian_noise_std if gaussian_noise_enabled else 0.05

    # Create augmentation transform for BYOL (applied independently to each view)
    rng1 = DeterministicRNG(args.seed)
    rng2 = DeterministicRNG(args.seed + 1)

    augmentation1 = BYOLAugmentation(
        noise_std=noise_std,
        spatial_mask_prob=args.spatial_mask,
        channel_dropout_prob=args.channel_dropout,
        blur_prob=args.blur_prob,
        blur_sigma=args.blur_sigma,
        crop_prob=args.crop_prob,
        crop_scale_min=args.crop_scale_min,
        crop_scale_max=args.crop_scale_max,
        rng = rng1
    )

    augmentation2 = BYOLAugmentation(
        noise_std=noise_std,
        spatial_mask_prob=args.spatial_mask,
        channel_dropout_prob=args.channel_dropout,
        blur_prob=args.blur_prob,
        blur_sigma=args.blur_sigma,
        crop_prob=args.crop_prob,
        crop_scale_min=args.crop_scale_min,
        crop_scale_max=args.crop_scale_max,
        rng = rng2
    )

    print("\n" + "=" * 80)
    print("BYOL Training Setup")
    print("=" * 80)
    print(f"input channels: {in_channels}")
    print(f"embed_dim: {embed_dim}")
    print(f"proj_dim: {proj_dim}")
    print(f"momentum coefficient: {args.momentum_coeff}")
    print(f"\nAugmentation:")
    print(f"  noise_std: {noise_std}")
    print(f"  spatial_mask_prob: {args.spatial_mask}")
    print(f"  channel_dropout_prob: {args.channel_dropout}")
    print(f"  blur_prob: {args.blur_prob}")
    print(f"  crop_prob: {args.crop_prob}")
    print(f"  crop_scale: [{args.crop_scale_min}, {args.crop_scale_max}]")

    best_score = - float('inf')
    best_epoch = -1
    global_step = 0

    for epoch in range(epochs):
        model.train()
        model.teacher_encoder.eval()  # Teacher is always in eval mode

        train_loss = 0.0
        train_batches = 0
        train_alignment = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", ncols=100)
        all_embeddings = []
        
        for batch in pbar:
            raw = batch[0]
            x_btchw, _, _ = prepare_input_btchw(raw, device, normalize_transform, num_time_steps=num_time_steps)

            # Create two augmented views
            x_view1 = augmentation1(x_btchw)
            x_view2 = augmentation2(x_btchw)

            # Forward through model (returns pred, teacher_proj, student_embed)
            pred1, teacher_proj2, student_embed1 = model(x_view1, x_view2)
            pred2, teacher_proj1, student_embed2 = model(x_view2, x_view1)

            loss = (byol_loss(pred1, teacher_proj2) + byol_loss(pred2, teacher_proj1)) * 0.5
            std1 = student_embed1.std(dim=0)
            std2 = student_embed2.std(dim=0)

            cov_loss = (covariance_loss(student_embed1) + covariance_loss(student_embed2)) * 0.5

            std_loss = (
                torch.mean(F.relu(1.0 - std1)) +
                torch.mean(F.relu(1.0 - std2))
            ) * 0.5
            loss = loss + 1.0 * std_loss + 1.0 * cov_loss

            align1 = compute_alignment(pred1, teacher_proj2)
            align2 = compute_alignment(pred2, teacher_proj1)
            train_alignment += ((align1 + align2) * 0.5).item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # Update momentum encoder with EMA
                momentum = cosine_schedule(
                    args.momentum_coeff,
                    0.999,
                    global_step,
                    total_steps,
                )

                model.update_teacher(momentum=momentum)
                
                # Collect embeddings for mode collapse monitoring
                if args.monitor_collapse:
                    all_embeddings.append(student_embed1.cpu())
                    all_embeddings.append(student_embed2.cpu())

            global_step += 1
            train_loss += loss.item()
            train_batches += 1
            
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'train_loss': f"{train_loss / max(1, train_batches):.6f}",
                'train_alignment': f"{train_alignment / max(1, train_batches):.6f}",
                'lr': f"{current_lr:.2e}"
            })

        avg_train = train_loss / max(1, train_batches)
        train_alignment = train_alignment / max(1, train_batches)
        
        # Monitor for mode collapse
        collapse_metrics = None
        if args.monitor_collapse and len(all_embeddings) > 0:
            all_embeddings_cat = torch.cat(all_embeddings, dim=0)
            collapse_metrics = detect_mode_collapse(all_embeddings_cat)

        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_alignment_total = 0.0
        val_embeddings = []

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Validation {epoch + 1}/{epochs}", ncols=100)
            for batch in val_pbar:
                raw = batch[0]
                x_btchw, _, _ = prepare_input_btchw(raw, device, normalize_transform, num_time_steps=num_time_steps)

                # Apply augmentations for validation
                x_view1 = x_btchw
                x_view2 = x_btchw

                # Compute loss in both directions (same as training)
                pred1, teacher_proj2, student_embed1 = model(x_view1, x_view2)
                pred2, teacher_proj1, student_embed2 = model(x_view2, x_view1)

                loss = (byol_loss(pred1, teacher_proj2) + byol_loss(pred2, teacher_proj1)) * 0.5
                std1 = student_embed1.std(dim=0)
                std2 = student_embed2.std(dim=0)

                std_loss = (
                    torch.mean(F.relu(1.0 - std1)) +
                    torch.mean(F.relu(1.0 - std2))
                ) * 0.5

                cov_loss = (covariance_loss(student_embed1) + covariance_loss(student_embed2)) * 0.5
                loss = loss + 1.0 * std_loss + 1.0 * cov_loss

                align1 = compute_alignment(pred1, teacher_proj2)
                align2 = compute_alignment(pred2, teacher_proj1)
                val_alignment_total += ((align1 + align2) * 0.5).item()
                
                val_loss += loss.item()
                val_batches += 1
                
                val_pbar.set_postfix({
                    'val_loss': f"{val_loss / max(1, val_batches):.6f}",
                    'val_alignment': f"{val_alignment_total / max(1, val_batches):.6f}"
                })
                
                if args.monitor_collapse:
                    val_embeddings.append(student_embed1.cpu())
                    val_embeddings.append(student_embed2.cpu())

        avg_val = val_loss / max(1, val_batches)
        val_alignment = val_alignment_total / max(1, val_batches)

        # Validation collapse metrics
        val_collapse_metrics = None
        if args.monitor_collapse and len(val_embeddings) > 0:
            val_embeddings_cat = torch.cat(val_embeddings, dim=0)
            val_collapse_metrics = detect_mode_collapse(val_embeddings_cat)
            mean_std = val_collapse_metrics['mean_std']
            mean_cosine = val_collapse_metrics['mean_cosine_sim']

        else:
            mean_std = 0.0
            mean_cosine = 0.0

        score = (
                val_alignment
                - 0.5 * mean_cosine
            )
        
        # Print epoch summary
        epoch_str = f"Epoch {epoch + 1}: train_loss={avg_train:.6f}, val_loss={avg_val:.6f}"
        if train_alignment is not None:
            epoch_str += f" | train_align={train_alignment:.4f}"
        if val_alignment is not None:
            epoch_str += f" | val_align={val_alignment:.4f}"
        if collapse_metrics:
            epoch_str += f" | train_collapse: std={collapse_metrics['mean_std']:.6f}, cosine={collapse_metrics['mean_cosine_sim']:.4f}"
            if collapse_metrics['is_collapsed']:
                epoch_str += " ⚠ COLLAPSED"
        if val_collapse_metrics:
            epoch_str += f" | val_collapse: std={val_collapse_metrics['mean_std']:.6f}, cosine={val_collapse_metrics['mean_cosine_sim']:.4f}"
            if val_collapse_metrics['is_collapsed']:
                epoch_str += " ⚠ COLLAPSED"        
        print(epoch_str)
        print(f"  Validation score (alignment - cosine): {score:.6f}\n")
        
        # Step the learning rate scheduler
        if scheduler is not None:
            scheduler.step()
            print(f"  LR updated to: {scheduler.get_last_lr()[0]:.2e}")

        if score > best_score:
            best_score = score
            best_epoch = epoch + 1

            ckpt_dir = Path('checkpoints')
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"{model_name}_best_byol.pth"

            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_score': best_score,
                    'train_loss': avg_train,
                    'train_alignment': train_alignment,
                    'val_loss': avg_val,
                    'val_alignment': val_alignment,
                    'embed_dim': embed_dim,
                    'proj_dim': proj_dim,
                },
                ckpt_path,
            )
            print(f"✓ Saved best checkpoint (score={best_score:.6f}) to: {ckpt_path}\n")

    print("\n" + "=" * 80)
    print("Training complete")
    print("=" * 80)
    print(f"Best validation score: {best_score:.6f} (epoch {best_epoch})")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("Extracting features from best model")
    print("=" * 80)

    ckpt_path = Path('checkpoints') / f"{model_name}_best_byol.pth"

    if ckpt_path.exists():
        print(f"Loading best checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        features_dir = Path('features') / model_name
        features_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving features to: {features_dir}")

        for split_name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
            print(f"\nExtracting {split_name} features...")
            all_features = []
            all_labels = []

            with torch.no_grad():
                for batch in tqdm(loader, desc=f"Processing {split_name}", ncols=100):
                    raw = batch[0]
                    labels = batch[1] if len(batch) > 1 else None

                    x_btchw, B, T = prepare_input_btchw(raw, device, normalize_transform, num_time_steps=num_time_steps)

                    embeddings, _ = model.student_encoder(x_btchw)
                    # Reshape from (B*T, embed_dim) to (B, T, embed_dim)
                    embeddings = embeddings.reshape(B, T, -1)

                    all_features.append(embeddings.cpu())

                    # Keep labels in (B,) shape (not expanded to B*T)
                    if labels is not None:
                        if not torch.is_tensor(labels):
                            labels = torch.tensor(labels)
                        labels = labels.detach().cpu()
                        if labels.dim() == 0:
                            labels = labels.unsqueeze(0)
                        all_labels.append(labels)

            if len(all_features) == 0:
                print(f"  ⚠ No {split_name} features found; skipping save")
                continue

            all_features = torch.cat(all_features, dim=0)
            print(f"  {split_name} features shape: {all_features.shape} (B, T, feature_dim)")

            # Compute statistics on flattened features
            std_temp = all_features.reshape(-1, all_features.shape[-1]).std(dim=0)
            print(
                f"  {split_name} features per-dim std: "
                f"min={std_temp.min():.6f}, max={std_temp.max():.6f}, mean={std_temp.mean():.6f}"
            )

            feature_path = features_dir / f"{split_name}_features.pt"
            save_dict = {'features': all_features}

            if len(all_labels) > 0:
                try:
                    all_labels = torch.cat(all_labels, dim=0)
                    save_dict['labels'] = all_labels
                    print(f"  {split_name} labels shape: {all_labels.shape}")
                except Exception:
                    print(f"  ⚠ Could not concatenate labels for {split_name}; saving features only")

            torch.save(save_dict, feature_path)
            print(f"  ✓ Saved to: {feature_path}")

        print("\n" + "=" * 80)
        print("Feature extraction complete")
        print("=" * 80)
    else:
        print(f"⚠ Checkpoint not found: {ckpt_path}")
        print("  Skipping feature extraction")


if __name__ == '__main__':
    main()
