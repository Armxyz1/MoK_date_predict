import random
import sys
from pathlib import Path
import os
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import yaml
import numpy as np
from tqdm import tqdm

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Add src to path so project imports work when running from repo root
current_file = Path(__file__).resolve()
exp_dir = current_file.parent
src_dir = exp_dir.parent
sys.path.insert(0, str(src_dir))

from data_pipeline.loaders.utils import load_config_and_create_dataloaders
from data_pipeline.preprocessing.normstats.compute_stats import compute_normalization_stats
from data_pipeline.preprocessing.normstats.stats_manager import (
    load_normalization_stats,
    save_normalization_stats,
    stats_exist,
)
from data_pipeline.preprocessing.transformers import NormalizeWithPrecomputedStats
from combined_model import ClimateTransformerBackbone, DeltaSeqRegressor, DinoBYOLCNN, CombinedModel, ExplainabilityWrapper

# ---- helper: normalize to [0, 1] ----
def normalize_map(a):
    a = np.abs(a)
    return (a - a.min()) / (a.max() - a.min() + 1e-8)


def build_channel_names(data_cfg: dict, channels_per_timestep: int):
    surf_vars = list(data_cfg.get("input_geo_var_surf", []))
    press_vars = list(data_cfg.get("input_geo_var_press", []))
    pressure_levels = list(data_cfg.get("pressure_levels", []))

    channel_names = []
    channel_names.extend([str(v) for v in surf_vars])

    for var_name in press_vars:
        if pressure_levels:
            for level in pressure_levels:
                channel_names.append(f"{var_name}_pl{level}")
        else:
            channel_names.append(str(var_name))

    if data_cfg.get("include_lat", False):
        channel_names.append("lat")
    if data_cfg.get("include_lon", False):
        channel_names.append("lon")
    if data_cfg.get("include_landsea", False):
        channel_names.append("landsea")

    # Guarantee length matches the actual channel dimension used in plotting.
    if len(channel_names) < channels_per_timestep:
        start_idx = len(channel_names)
        channel_names.extend([f"channel_{idx}" for idx in range(start_idx, channels_per_timestep)])
    elif len(channel_names) > channels_per_timestep:
        channel_names = channel_names[:channels_per_timestep]

    return channel_names

# ---- plotting function ----
def plot_timestep_channels_side_by_side(input_maps, attr_maps, channel_names, save_path, title=""):
    num_channels = input_maps.shape[0]

    fig, axes = plt.subplots(num_channels, 2, figsize=(12, 3 * num_channels), squeeze=False)

    for c in range(num_channels):
        input_map = input_maps[c]
        attr_map = normalize_map(attr_maps[c])

        ax_input = axes[c, 0]
        ax_attr = axes[c, 1]

        ax_input.imshow(input_map)
        ax_input.set_ylabel(channel_names[c], fontsize=10)

        # Use a light-to-dark colormap so low values are light and high values are dark.
        ax_attr.imshow(attr_map, cmap="binary", vmin=0, vmax=1)

        if c == 0:
            ax_input.set_title("Input")
            ax_attr.set_title("Attribution")

        for ax in (ax_input, ax_attr):
            xtick_labels = np.linspace(-180, 180, 7)
            ytick_labels = np.linspace(-90, 90, 7)
            ax.set_xticks(np.linspace(0, input_map.shape[1], len(xtick_labels)))
            ax.set_yticks(np.linspace(0, input_map.shape[0], len(ytick_labels)))
            ax.set_xticklabels([f"{int(x)}" for x in xtick_labels])
            ax.set_yticklabels([f"{int(y)}" for y in ytick_labels[::-1]])
            ax.grid(True, linestyle="--", alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=150)
    plt.close()

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

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


def main():
    set_seed(42)

    config = load_config("/home/armaan/MoK_orig_finer/config/model_config.yml")

    model_name = "byol_new_transformer_20"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader_no_norm, val_loader, test_loader = load_config_and_create_dataloaders(
        config_path="/home/armaan/MoK_orig_finer/config/model_config.yml"
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

        stats_already_exist = stats_exist(model_name)
        if stats_already_exist:
            print(f"Loading existing normalization statistics for '{model_name}'...")
            norm_stats = load_normalization_stats(model_name, verbose=True)
        else:
            print(f"Computing normalization statistics from training data for '{model_name}'...")
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
            from src.data_pipeline.preprocessing.transformers import Compose, PerChannelMinMaxNormalize

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

    train_loader = train_loader_no_norm
    print(f"✓ Using dataloaders (normalization applied per-batch for temporal handling)")

    num_time_steps = resolve_num_time_steps(data_cfg)
    print(f"\nnum_time_steps from config: {num_time_steps}")

    sample = next(iter(train_loader))[0]
    print(f"Sample shape from dataloader: {sample.shape}")
    
    if sample.dim() == 3:
        in_channels = 1
    elif sample.dim() == 4:
        TC = sample.shape[1]
        if TC % num_time_steps == 0:
            in_channels = TC // num_time_steps
            print(f"Calculated: {TC} (loaded) / {num_time_steps} (time steps) = {in_channels} channels")
        else:
            base_channels = TC - 2
            if base_channels > 0 and base_channels % num_time_steps == 0:
                channels_per_step = base_channels // num_time_steps
                in_channels = channels_per_step + 2
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

    model_cfg = config.get('model', {})
    embed_dim = 512
    proj_dim = 1024
    backbone_dim = 1024
    spatial_h = int(sample.shape[-2])
    spatial_w = int(sample.shape[-1])

    transformer_backbone = ClimateTransformerBackbone(
        in_channels=in_channels,
        embed_dim=backbone_dim,
        patch_size=int(model_cfg.get('patch_size', 16)),
        depth=int(model_cfg.get('depth', 6)),
        num_heads=int(model_cfg.get('num_heads', 8)),
        mlp_ratio=float(model_cfg.get('mlp_ratio', 4.0)),
        input_h=spatial_h,
        input_w=spatial_w,
    )

    bckbn = DinoBYOLCNN(
            transformer_backbone,
            backbone_dim=backbone_dim,
            embed_dim=embed_dim,
            proj_dim=proj_dim
        ).to(device)

    print("\n" + "=" * 80)
    print("Extracting features from checkpoint")
    print("=" * 80)

    backbn_ckpt_path = Path('/home/armaan/MoK_orig_finer/checkpoints/MoK_byol_new_transformer_45_best_byol.pth')
    backbn_checkpoint = torch.load(backbn_ckpt_path, map_location=device)
    bckbn.load_state_dict(backbn_checkpoint['model_state_dict'])
    bckbn.eval()
    bckbn.student_encoder.eval()

    regressor = DeltaSeqRegressor(
        input_dim=512,
        hidden_dim=16,
        num_layers=1,
        T=13
    )
    regressor_ckpt_path = Path('/home/armaan/MoK_orig_finer/checkpoints/new_delta_v5_grid_MoK_byol_new_transformer_45_h16_a1.300_b1.000_g1.900_nl1_best_checkpoint.pth')
    regressor_checkpoint = torch.load(regressor_ckpt_path, map_location=device)
    regressor.load_state_dict(regressor_checkpoint['state_dict'])
    regressor.eval()

    model = CombinedModel(
        backbone=bckbn,
        regressor=regressor,
        device=device,
        normalize_transform=normalize_transform,
        num_time_steps=num_time_steps
    ).to(device)

    explainer = ExplainabilityWrapper(
        model = model,
        device=device,
        num_time_steps=num_time_steps,
    )

    base_dir = "outputs"
    os.makedirs(base_dir, exist_ok=True)

    sample_counter = 0

    for batch in tqdm(test_loader, desc="Processing Test Set", ncols=100):
        x = batch[0].to(device)
        B, TC, H, W = x.shape

        T = num_time_steps
        C = TC // T
        channel_names = build_channel_names(data_cfg, C)

        # ---- reshape input for visualization ----
        x_reshaped = x.reshape(B, C, T, H, W).permute(0, 2, 1, 3, 4)
        x_reshaped = x_reshaped.detach().cpu().numpy()

        # ---- run explainability ----
        outputs = explainer.attribute(
            x,
            n_steps=32,
        )

        maps = outputs["maps"].detach().cpu().numpy()  # (B, T, C, H, W)

        # ---- process each sample ----
        for i in range(B):
            sample_dir = os.path.join(base_dir, f"sample_{sample_counter:05d}")
            os.makedirs(sample_dir, exist_ok=True)

            for t in range(T):
                input_maps_t = x_reshaped[i][t]  # (C, H, W)
                attr_maps_t = maps[i][t]         # (C, H, W)

                save_path = os.path.join(
                    sample_dir,
                    f"side_by_side_t{t:02d}_all_channels.png"
                )

                plot_timestep_channels_side_by_side(
                    input_maps_t,
                    attr_maps_t,
                    channel_names,
                    save_path,
                    title=f"Timestep {t} | All Channels"
                )

            sample_counter += 1


if __name__ == '__main__':
    main()
