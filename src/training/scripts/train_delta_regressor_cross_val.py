"""
Train DeltaSeqRegressor on precomputed sequence features.

This script:
1) Loads features from features/<feature_subdir>/{train,val,test}_features.pt
2) Slices timesteps 0-12 and preserves the per-timestep feature sequence.
3) Trains DeltaSeqRegressor on train split only.
4) Uses val split for early stopping/generalization.
5) Uses DeltaLossV2 for training only.
6) Uses final-timestep MSE only for validation/test evaluation.
7) Evaluates train/val/test and writes prediction/metric tables to results/tables.
"""


import argparse
import json
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yaml

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.architectures.delta_regressor import DeltaSeqRegressor, DeltaLossV2


DEFAULT_FEATURE_SUBDIR = "MoK_byol_new_transformer_45"
DEFAULT_MODEL_NAME_PREFIX = "delta_grid_search"
ACTIVE_FEATURE_SUBDIR = DEFAULT_FEATURE_SUBDIR
ACTIVE_MODEL_NAME_PREFIX = DEFAULT_MODEL_NAME_PREFIX
ACTIVE_BEST_CHECKPOINT: Optional[Dict[str, Any]] = None
ACTIVE_BEST_VAL_LOSS = float("inf")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_years(year_spec: List) -> List[int]:
    years: List[int] = []
    for item in year_spec:
        if isinstance(item, int):
            years.append(item)
        elif isinstance(item, str) and ":" in item:
            start_str, end_str = item.split(":", 1)
            start = int(start_str)
            end = int(end_str)
            step = 1 if end >= start else -1
            years.extend(list(range(start, end + step, step)))
        elif isinstance(item, str):
            years.append(int(item))
        else:
            raise ValueError(f"Unrecognized year spec item: {item}")
    return years


def maybe_build_years(config: dict, split: str, num_samples: int) -> List[int]:
    data_cfg = config.get("data", {})
    spec = data_cfg.get(f"{split}_years")
    if spec is None:
        return list(range(num_samples))

    years = parse_years(spec)
    if len(years) != num_samples:
        print(
            f"Year count mismatch for {split}: {len(years)} years vs {num_samples} samples. "
            "Falling back to sample indices."
        )
        return list(range(num_samples))
    return years


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_tag(value: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in value)


def slice_time_window(feats: np.ndarray, time_steps: List[int]) -> np.ndarray:
    if feats.ndim == 2:
        return feats
    if feats.ndim == 3:
        return feats[:, time_steps, :]
    if feats.ndim == 4:
        return feats[:, :, time_steps, :]
    raise ValueError(f"Expected 2D, 3D or 4D features, got shape {feats.shape}")


def ensure_sequence_features(feats: np.ndarray, expected_steps: int) -> np.ndarray:
    if feats.ndim == 2:
        if feats.shape[1] % expected_steps != 0:
            raise ValueError(
                f"Cannot reshape flat features of shape {feats.shape} into ({expected_steps}, D)."
            )
        feats = feats.reshape(feats.shape[0], expected_steps, feats.shape[1] // expected_steps)
    elif feats.ndim == 3:
        if feats.shape[1] != expected_steps:
            raise ValueError(
                f"Expected {expected_steps} timesteps, got feature shape {feats.shape}."
            )
    elif feats.ndim == 4:
        if feats.shape[1] == expected_steps:
            feats = feats.reshape(feats.shape[0], expected_steps, -1)
        elif feats.shape[2] == expected_steps:
            feats = np.transpose(feats, (0, 2, 1, 3)).reshape(feats.shape[0], expected_steps, -1)
        else:
            raise ValueError(
                f"Expected one temporal dimension of length {expected_steps}, got shape {feats.shape}."
            )
    else:
        raise ValueError(f"Expected 2D, 3D or 4D features, got shape {feats.shape}")

    return np.asarray(feats, dtype=np.float32)


def load_split_features(
    feature_root: Path,
    feature_subdir: str,
    split: str,
    time_steps: List[int],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    feature_path = feature_root / feature_subdir / f"{split}_features.pt"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    payload = torch.load(feature_path, map_location="cpu")
    if not isinstance(payload, dict) or "features" not in payload:
        raise ValueError(f"Expected dict with 'features' in {feature_path}")

    feats = payload["features"]
    if isinstance(feats, torch.Tensor):
        feats = feats.cpu().numpy()
    feats = np.asarray(feats)

    feats = slice_time_window(feats, time_steps)
    feats = ensure_sequence_features(feats, expected_steps=len(time_steps))

    labels = payload.get("labels")
    if labels is not None:
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        labels = np.asarray(labels).reshape(-1)

    return feats, labels


def compute_train_target_mean(config: dict, y_train: Optional[np.ndarray] = None) -> Optional[float]:
    try:
        data_cfg = config.get("data", {})
        target_file = data_cfg.get("target_file")
        train_years_spec = data_cfg.get("train_years")

        if target_file is None:
            raise ValueError("No target_file specified in config.data")

        targets_df = pd.read_csv(target_file)

        if train_years_spec is not None:
            train_years = (
                parse_years(train_years_spec)
                if isinstance(train_years_spec, list)
                else parse_years([train_years_spec])
            )
            train_targets_df = targets_df[targets_df["Year"].isin(train_years)]
        else:
            train_targets_df = targets_df

        if len(train_targets_df) == 0:
            train_targets_df = targets_df

        target_idx = config.get("model", {}).get("target", None)
        if isinstance(target_idx, int) and target_idx < len(targets_df.columns):
            target_col = targets_df.columns[target_idx]
        elif "DateRelJun01" in train_targets_df.columns:
            target_col = "DateRelJun01"
        else:
            target_col = train_targets_df.columns[1]

        train_target_mean = float(train_targets_df[target_col].mean())

        return train_target_mean
    except Exception:
        if y_train is not None and len(y_train) > 0:
            return float(np.mean(y_train))
        return None


def compute_regression_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    train_target_mean: Optional[float] = None,
) -> Dict[str, float]:
    df = pd.DataFrame({"Target": targets, "Prediction": predictions})
    df["Error"] = df["Prediction"] - df["Target"]
    df["Absolute_Error"] = df["Error"].abs()

    target_mean = df["Target"].mean()
    target_std = df["Target"].std()
    mae = df["Absolute_Error"].mean()
    rmse = (df["Error"] ** 2).mean() ** 0.5

    metrics = {
        "target_mean": float(target_mean),
        "target_std": float(target_std),
        "mae": float(mae),
        "rmse": float(rmse),
        "num_samples": int(len(df)),
    }

    if train_target_mean is not None:
        mse_model = (df["Error"] ** 2).mean()
        baseline_errors = df["Target"] - train_target_mean
        mse_baseline = (baseline_errors ** 2).mean()
        skill_score = 1 - (mse_model / mse_baseline) if mse_baseline > 0 else 0.0
        metrics["skill_score"] = float(skill_score)
    else:
        ss_res = (df["Error"] ** 2).sum()
        ss_tot = ((df["Target"] - target_mean) ** 2).sum()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        metrics["r2"] = float(r2)

    return metrics


def save_predictions_table(
    years: List[int],
    targets: np.ndarray,
    predictions: np.ndarray,
    output_path: Path,
) -> None:
    df = pd.DataFrame(
        {
            "Year": years,
            "Target": targets.tolist(),
            "Prediction": predictions.tolist(),
            "Error": (predictions - targets).tolist(),
            "Absolute_Error": np.abs(predictions - targets).tolist(),
        }
    )

    df = df.sort_values("Year").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved predictions to: {output_path}")


def build_seq_loader(
    feats: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.FloatTensor(feats),
        torch.FloatTensor(targets),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_seq_model(
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    device: str,
    train_mean: Optional[float] = None,
) -> nn.Module:
    model = DeltaSeqRegressor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        T=13,
    ).to(device)

    if train_mean is not None:
        with torch.no_grad():
            model.y0.copy_(torch.tensor(float(train_mean), device=model.y0.device, dtype=model.y0.dtype))

    return model


def build_loss() -> nn.Module:
    return nn.MSELoss()


def train_with_val(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: str,
    patience: int,
    alpha: float,
    beta: float,
    gamma: float,
    verbose: int,
) -> Tuple[nn.Module, float]:

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_criterion = DeltaLossV2(alpha=alpha, beta=beta, gamma=gamma).to(device)
    eval_criterion = build_loss().to(device)

    best_state = None
    best_val_mse = float("inf")
    stale_epochs = 0

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_samples = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred_seq = model(xb)
            loss = train_criterion(pred_seq, yb)

            loss.backward()
            optimizer.step()

            batch_size = len(xb)
            epoch_train_loss += float(loss.item()) * batch_size
            epoch_samples += batch_size

        # ======================================================
        # VALIDATION
        # ======================================================
        model.eval()
        val_preds = []
        val_targets = []
        val_mse_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pred_seq = model(xb)
                batch_loss = eval_criterion(pred_seq[:, -1], yb)

                batch_size = len(xb)
                val_mse_loss += float(batch_loss.item()) * batch_size
                val_samples += batch_size

                val_preds.append(pred_seq[:, -1].cpu().numpy())
                val_targets.append(yb.cpu().numpy())

        val_preds_np = np.concatenate(val_preds)
        val_targets_np = np.concatenate(val_targets)
        val_mse = float(np.mean((val_preds_np - val_targets_np) ** 2))

        # ======================================================
        # EARLY STOPPING
        # ======================================================
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                if verbose >= 1:
                    print(f"Early stopping at epoch {epoch + 1} (best val MSE={best_val_mse:.6f})")
                break

        if verbose >= 2:
            train_loss = epoch_train_loss / max(epoch_samples, 1)
            val_loss = val_mse_loss / max(val_samples, 1)
            print(
                f"    epoch={epoch + 1:04d} "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} "
                f"val_mse={val_mse:.6f} "
                f"best={best_val_mse:.6f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_mse


def predict_scalar(model: nn.Module, feats: np.ndarray, device: str, batch_size: int = 256) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(torch.FloatTensor(feats)), batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            pred_seq = model(xb)
            preds.append(pred_seq[:, -1].cpu().numpy())
    return np.concatenate(preds)


def predict_all_timesteps(model: nn.Module, feats: np.ndarray, device: str, batch_size: int = 256) -> np.ndarray:
    """Get predictions for all timesteps.
    
    Returns:
        2D array of shape (n_samples, n_timesteps) with predictions for each timestep.
    """
    model.eval()
    loader = DataLoader(TensorDataset(torch.FloatTensor(feats)), batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            pred_seq = model(xb)  # (batch_size, T)
            preds.append(pred_seq.cpu().numpy())
    return np.concatenate(preds)


def compute_test_loss(
    model: nn.Module,
    feats: np.ndarray,
    targets: np.ndarray,
    device: str,
    batch_size: int = 256,
) -> float:
    """Compute final-timestep MSE on test set."""
    criterion = build_loss().to(device)
    
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(feats), torch.FloatTensor(targets)),
        batch_size=batch_size,
        shuffle=False,
    )
    
    total_loss = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred_seq = model(xb)

            pred_final = pred_seq[:, -1]
            loss = criterion(pred_final, yb)
            
            batch_size_actual = len(xb)
            total_loss += float(loss.item()) * batch_size_actual
            num_samples += batch_size_actual
    
    return total_loss / max(num_samples, 1)

def evaluate_grid_point(
    hidden_dim: int,
    alpha: float,
    beta: float,
    gamma: float,
    num_layers: int,
) -> Optional[Dict[str, Any]]:
    global ACTIVE_BEST_CHECKPOINT, ACTIVE_BEST_VAL_LOSS

    config_path = "config/model_config.yml"
    feature_subdir = ACTIVE_FEATURE_SUBDIR
    feature_tag = make_tag(Path(feature_subdir).name)
    model_name = (
        f"{ACTIVE_MODEL_NAME_PREFIX}_{feature_tag}_"
        f"h{hidden_dim}_a{alpha:.3f}_b{beta:.3f}_g{gamma:.3f}_nl{num_layers}"
    )
    feature_root = Path("features")
    dropout = 0.0
    lr = 1e-3
    weight_decay = 0
    epochs = 300
    batch_size = 10
    patience = 20
    verbose = 0

    config = load_config(config_path)
    time_steps = list(range(0, 13))
    seed_values = [51, 53, 55, 57]


    X_train, y_train = load_split_features(feature_root, feature_subdir, "train", time_steps)
    X_val, y_val = load_split_features(feature_root, feature_subdir, "val", time_steps)
    X_test, y_test = load_split_features(feature_root, feature_subdir, "test", time_steps)

    train_mask = ~np.isnan(y_train)
    val_mask = ~np.isnan(y_val)
    test_mask = ~np.isnan(y_test)
    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_val, y_val = X_val[val_mask], y_val[val_mask]
    X_test, y_test = X_test[test_mask], y_test[test_mask]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    min_val_loss = float("inf")
    best_seed_payload = None
    train_target_mean = compute_train_target_mean(config, y_train)

    for seed in seed_values:
        set_random_seed(seed)
        train_loader = build_seq_loader(X_train, y_train, batch_size=batch_size, shuffle=False)
        val_loader = build_seq_loader(X_val, y_val, batch_size=batch_size, shuffle=False)
        model = build_seq_model(
            input_dim=X_train.shape[-1],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            device=device,
            train_mean=train_target_mean,
        )
        model, best_val_mse = train_with_val(
            model,
            train_loader,
            val_loader,
            epochs=epochs,
            learning_rate=lr,
            weight_decay=weight_decay,
            device=device,
            patience=patience,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            verbose=verbose,
        )
        train_preds = predict_scalar(model, X_train, device=device)
        val_preds = predict_scalar(model, X_val, device=device)
        test_preds = predict_scalar(model, X_test, device=device)
        train_metrics = compute_regression_metrics(y_train, train_preds, train_target_mean)
        val_metrics = compute_regression_metrics(y_val, val_preds, train_target_mean)
        test_metrics = compute_regression_metrics(y_test, test_preds, train_target_mean)
        test_loss = compute_test_loss(model, X_test, y_test, device)
        payload = {
            "seed": seed,
            "model_name": model_name,
            "feature_subdir": feature_subdir,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "train_preds": train_preds,
            "val_preds": val_preds,
            "test_preds": test_preds,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "test_loss": test_loss,
            "best_val_mse": best_val_mse,
            "train_targets": y_train,
            "val_targets": y_val,
            "test_targets": y_test,
        }
        if best_val_mse < min_val_loss:
            min_val_loss = best_val_mse
            best_seed_payload = payload
        if best_val_mse < ACTIVE_BEST_VAL_LOSS:
            ACTIVE_BEST_VAL_LOSS = best_val_mse
            ACTIVE_BEST_CHECKPOINT = {
                "model_name": model_name,
                "feature_subdir": feature_subdir,
                "seed": seed,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "test_loss": float(test_loss),
                "best_val_mse": float(best_val_mse),
                # Clone tensors to CPU so checkpoint is portable across devices.
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
    return best_seed_payload



def main():
    global ACTIVE_FEATURE_SUBDIR, ACTIVE_MODEL_NAME_PREFIX, ACTIVE_BEST_CHECKPOINT, ACTIVE_BEST_VAL_LOSS

    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-subdir", type=str, default=DEFAULT_FEATURE_SUBDIR)
    parser.add_argument("--model-name-prefix", type=str, default=DEFAULT_MODEL_NAME_PREFIX)
    args = parser.parse_args()

    ACTIVE_FEATURE_SUBDIR = args.feature_subdir
    ACTIVE_MODEL_NAME_PREFIX = args.model_name_prefix
    ACTIVE_BEST_CHECKPOINT = None
    ACTIVE_BEST_VAL_LOSS = float("inf")

    hidden_dim_values = [16]
    alpha_values = [1.3, 1.5, 1.7]
    beta_values = [0.5, 1.0, 1.5]
    gamma_values = [1.5, 1.7, 1.9]
    num_layers_values = [1]
    param_grid = list(product(hidden_dim_values, alpha_values, beta_values, gamma_values, num_layers_values))

    print(f"Running grid search with {len(param_grid)} combinations")

    all_rows: List[Dict[str, Any]] = []
    per_trial_rows: List[Dict[str, Any]] = []
    best_seed_result: Optional[Dict[str, Any]] = None

    for trial_idx, (hidden_dim, alpha, beta, gamma, num_layers) in enumerate(param_grid):
        print(
            f"[{trial_idx + 1}/{len(param_grid)}] "
            f"h={hidden_dim}, a={alpha}, b={beta}, g={gamma}, nl={num_layers}"
        )
        best_seed = evaluate_grid_point(
            hidden_dim=hidden_dim,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            num_layers=num_layers,
        )

        if best_seed is None:
            continue

        print(
            f"  -> best_seed={best_seed['seed']} "
            f"best_val_mse={best_seed['best_val_mse']:.6f} "
            f"test_loss={best_seed['test_loss']:.6f}"
        )
        print("\n")

        for split, metrics in zip(["train", "val", "test"], [best_seed["train_metrics"], best_seed["val_metrics"], best_seed["test_metrics"]]):
            all_rows.append({
                "trial": trial_idx,
                "seed": best_seed["seed"],
                "hidden_dim": best_seed["hidden_dim"],
                "num_layers": best_seed["num_layers"],
                "alpha": best_seed["alpha"],
                "beta": best_seed["beta"],
                "gamma": best_seed["gamma"],
                "split": split,
                "best_val_mse": best_seed["best_val_mse"],
                **metrics,
            })

        per_trial_rows.append({
            "trial": trial_idx,
            "seed": best_seed["seed"],
            "feature_subdir": best_seed.get("feature_subdir", ACTIVE_FEATURE_SUBDIR),
            "hidden_dim": best_seed["hidden_dim"],
            "num_layers": best_seed["num_layers"],
            "alpha": best_seed["alpha"],
            "beta": best_seed["beta"],
            "gamma": best_seed["gamma"],
            "best_val_mse": best_seed["best_val_mse"],
            "test_loss": best_seed["test_loss"],
        })

        if best_seed_result is None or best_seed["best_val_mse"] < best_seed_result["best_val_mse"]:
            best_seed_result = best_seed

    # Use the selected best model name for saved artifact prefixes.
    default_feature_tag = make_tag(Path(ACTIVE_FEATURE_SUBDIR).name)
    model_name = (
        str(best_seed_result.get("model_name"))
        if best_seed_result is not None and best_seed_result.get("model_name")
        else f"{ACTIVE_MODEL_NAME_PREFIX}_{default_feature_tag}"
    )

    # Save metrics summary
    results_dir = project_root / "results" / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(all_rows)
    summary_path = results_dir / f"{model_name}_trial_seed_split_metrics.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved metrics summary to: {summary_path}")

    trial_df = pd.DataFrame(per_trial_rows)
    trial_summary_path = results_dir / f"{model_name}_grid_trial_summary.csv"
    trial_df = trial_df.sort_values(by="test_loss", ascending=True).reset_index(drop=True)
    trial_df.to_csv(trial_summary_path, index=False)
    print(f"Saved grid trial summary to: {trial_summary_path}")

    # Save best hyperparameters and predictions
    if best_seed_result is not None:
        # Save predictions
        config = load_config("config/model_config.yml")
        years_train = maybe_build_years(config, "train", len(best_seed_result["train_preds"]))
        years_val = maybe_build_years(config, "val", len(best_seed_result["val_preds"]))
        years_test = maybe_build_years(config, "test", len(best_seed_result["test_preds"]))
        save_predictions_table(
            years_train,
            np.asarray(best_seed_result["train_targets"]),
            np.asarray(best_seed_result["train_preds"]),
            results_dir / f"{model_name}_best_train_predictions.csv",
        )
        save_predictions_table(
            years_val,
            np.asarray(best_seed_result["val_targets"]),
            np.asarray(best_seed_result["val_preds"]),
            results_dir / f"{model_name}_best_val_predictions.csv",
        )
        save_predictions_table(
            years_test,
            np.asarray(best_seed_result["test_targets"]),
            np.asarray(best_seed_result["test_preds"]),
            results_dir / f"{model_name}_best_test_predictions.csv",
        )
        # Save best hparams JSON
        best_hparams_path = results_dir / f"{model_name}_best_hparams.json"
        best_hparams_payload = {
            "model_name": best_seed_result.get("model_name"),
            "feature_subdir": best_seed_result.get("feature_subdir", ACTIVE_FEATURE_SUBDIR),
            "hidden_dim": best_seed_result["hidden_dim"],
            "num_layers": best_seed_result["num_layers"],
            "alpha": best_seed_result["alpha"],
            "beta": best_seed_result["beta"],
            "gamma": best_seed_result["gamma"],
            "test_loss": float(best_seed_result["test_loss"]),
            "best_val_mse": float(best_seed_result["best_val_mse"]),
            "seed": int(best_seed_result["seed"]),
        }
        with open(best_hparams_path, "w") as f:
            json.dump(best_hparams_payload, f, indent=2)
        print(f"Saved best hyperparameter config to: {best_hparams_path}")

        # Save best checkpoint found across all trials/seeds.
        if ACTIVE_BEST_CHECKPOINT is not None and "state_dict" in ACTIVE_BEST_CHECKPOINT:
            checkpoints_dir = project_root / "checkpoints"
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoints_dir / f"{model_name}_best_checkpoint.pth"
            torch.save(ACTIVE_BEST_CHECKPOINT, checkpoint_path)
            print(f"Saved best model checkpoint to: {checkpoint_path}")

        # Save per-timestep skill scores for best seed
        config = load_config("config/model_config.yml")
        feature_root = Path("features")
        feature_subdir = str(best_seed_result.get("feature_subdir", ACTIVE_FEATURE_SUBDIR))
        time_steps = list(range(0, 13))
        X_train, y_train = load_split_features(feature_root, feature_subdir, "train", time_steps)
        X_val, y_val = load_split_features(feature_root, feature_subdir, "val", time_steps)
        X_test, y_test = load_split_features(feature_root, feature_subdir, "test", time_steps)
        train_mask = ~np.isnan(y_train)
        val_mask = ~np.isnan(y_val)
        test_mask = ~np.isnan(y_test)
        X_train, y_train = X_train[train_mask], y_train[train_mask]
        X_val, y_val = X_val[val_mask], y_val[val_mask]
        X_test, y_test = X_test[test_mask], y_test[test_mask]
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        set_random_seed(int(best_seed_result["seed"]))
        model = build_seq_model(
            input_dim=X_train.shape[-1],
            hidden_dim=best_seed_result["hidden_dim"],
            num_layers=best_seed_result["num_layers"],
            dropout=0.0,
            device=device,
            train_mean=compute_train_target_mean(config, y_train),
        )
        train_loader = build_seq_loader(X_train, y_train, batch_size=10, shuffle=False)
        val_loader = build_seq_loader(X_val, y_val, batch_size=10, shuffle=False)
        model, _ = train_with_val(
            model,
            train_loader,
            val_loader,
            epochs=300,
            learning_rate=1e-3,
            weight_decay=0,
            device=device,
            patience=20,
            alpha=best_seed_result["alpha"],
            beta=best_seed_result["beta"],
            gamma=best_seed_result["gamma"],
            verbose=0,
        )
        train_all_preds = predict_all_timesteps(model, X_train, device=device)
        val_all_preds = predict_all_timesteps(model, X_val, device=device)
        test_all_preds = predict_all_timesteps(model, X_test, device=device)
        train_target_mean = compute_train_target_mean(config, y_train)
        timestep_skill_rows = []
        for t_idx, t_val in enumerate(time_steps):
            train_t_preds = train_all_preds[:, t_idx]
            val_t_preds = val_all_preds[:, t_idx]
            test_t_preds = test_all_preds[:, t_idx]
            train_t_metrics = compute_regression_metrics(y_train, train_t_preds, train_target_mean)
            val_t_metrics = compute_regression_metrics(y_val, val_t_preds, train_target_mean)
            test_t_metrics = compute_regression_metrics(y_test, test_t_preds, train_target_mean)
            for split, metrics in zip(["train", "val", "test"], [train_t_metrics, val_t_metrics, test_t_metrics]):
                timestep_skill_rows.append({
                    "timestep_index": t_idx,
                    "step_value": t_val,
                    "split": split,
                    "skill_score": metrics.get("skill_score", np.nan),
                })
        timestep_skill_df = pd.DataFrame(timestep_skill_rows)
        timestep_skill_path = results_dir / f"{model_name}_best_seed_timestep_skill_scores.csv"
        timestep_skill_df.to_csv(timestep_skill_path, index=False)
        print(f"Saved timestep skill scores to: {timestep_skill_path}")

    print("Best grid-search result:")
    if best_seed_result is not None:
        print(
            {
                "model_name": best_seed_result.get("model_name"),
                "seed": int(best_seed_result["seed"]),
                "hidden_dim": best_seed_result["hidden_dim"],
                "num_layers": best_seed_result["num_layers"],
                "alpha": best_seed_result["alpha"],
                "beta": best_seed_result["beta"],
                "gamma": best_seed_result["gamma"],
                "best_val_mse": float(best_seed_result["best_val_mse"]),
                "test_loss": float(best_seed_result["test_loss"]),
            }
        )
    else:
        print("No valid grid-search result found.")


if __name__ == "__main__":
    main()
