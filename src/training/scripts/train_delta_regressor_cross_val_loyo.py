"""Sequential test-year evaluation with train augmentation.

Workflow:
1) Load precomputed sequence features from features/<feature_subdir>/{train,val,test}_features.pt.
2) Keep timesteps 0..12 and preserve sequence structure.
3) Iterate over test years in original order (no shuffling).
4) For test index i, append test samples [0..i-1] to the end of train data.
5) Train DeltaSeqRegressor with early stopping on the same fixed val split.
6) Evaluate only the current test year i.
7) Save per-year predictions and aggregate MSE/RMSE/skill score.
"""


import argparse
import json
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

from combined_model import DeltaSeqRegressor, DeltaLossV2


DEFAULT_FEATURE_SUBDIR = "MoK_byol_new_transformer_45"
DEFAULT_MODEL_NAME_PREFIX = "delta_loyo"
DEFAULT_HIDDEN_DIM_SEARCH = "16"
DEFAULT_NUM_LAYERS_SEARCH = "1"
DEFAULT_TOP_K_BEST_TRIALS = 50


def parse_search_values(spec: str, cast_type: type) -> List[Any]:
    values: List[Any] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(cast_type(token))
    if not values:
        raise ValueError(f"Search spec is empty: '{spec}'")
    return values


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


def apply_valid_label_mask(
    feats: np.ndarray,
    labels: np.ndarray,
    years: List[int],
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    mask = ~np.isnan(labels)
    filtered_years = [int(y) for y, keep in zip(years, mask) if keep]
    return feats[mask], labels[mask], filtered_years


def compute_target_mean_from_values(values: np.ndarray) -> Optional[float]:
    if values.size == 0:
        return None
    return float(np.mean(values))


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
        print(
            f"Training target mean: {train_target_mean:.4f} "
            f"(from {len(train_targets_df)} samples in column '{target_col}')"
        )
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


def compute_skill_score_from_baseline(
    targets: np.ndarray,
    predictions: np.ndarray,
    baseline_means: np.ndarray,
) -> float:
    mse_model = float(np.mean((predictions - targets) ** 2))
    mse_baseline = float(np.mean((targets - baseline_means) ** 2))
    if mse_baseline <= 0:
        return 0.0
    return 1.0 - (mse_model / mse_baseline)


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


def run_sequential_test_eval(args: argparse.Namespace, save_outputs: bool = True) -> Dict[str, float]:
    config = load_config(args.config_path)
    feature_root = Path(args.feature_root)
    feature_subdir = args.feature_subdir
    feature_tag = make_tag(Path(feature_subdir).name)
    model_name = (
        f"{args.model_name_prefix}_{feature_tag}_"
        f"h{args.hidden_dim}_a{args.alpha:.3f}_b{args.beta:.3f}_g{args.gamma:.3f}_nl{args.num_layers}"
    )

    time_steps = list(range(0, 13))
    X_train, y_train = load_split_features(feature_root, feature_subdir, "train", time_steps)
    X_val, y_val = load_split_features(feature_root, feature_subdir, "val", time_steps)
    X_test, y_test = load_split_features(feature_root, feature_subdir, "test", time_steps)

    train_years = maybe_build_years(config, "train", len(y_train))
    val_years = maybe_build_years(config, "val", len(y_val))
    test_years = maybe_build_years(config, "test", len(y_test))

    X_train, y_train, train_years = apply_valid_label_mask(X_train, y_train, train_years)
    X_val, y_val, val_years = apply_valid_label_mask(X_val, y_val, val_years)
    X_test, y_test, test_years = apply_valid_label_mask(X_test, y_test, test_years)

    if X_test.shape[0] == 0:
        raise ValueError("No valid test samples after removing NaN labels.")

    del val_years
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed_candidates = [51, 53, 55, 57]

    year_rows: List[Dict[str, Any]] = []
    aggregate_targets: List[float] = []
    aggregate_preds: List[float] = []
    aggregate_train_means: List[float] = []
    aggregate_best_val_mse_sum = 0.0

    for i in range(X_test.shape[0]):
        current_year = int(test_years[i])
        X_hist = X_test[:i]
        y_hist = y_test[:i]
        X_train_aug = np.concatenate([X_train, X_hist], axis=0)
        y_train_aug = np.concatenate([y_train, y_hist], axis=0)

        train_target_mean = compute_target_mean_from_values(y_train_aug)
        if train_target_mean is None:
            raise ValueError("Unable to compute train target mean for skill score.")

        train_loader = build_seq_loader(
            X_train_aug,
            y_train_aug,
            batch_size=args.batch_size,
            shuffle=False,
        )
        val_loader = build_seq_loader(
            X_val,
            y_val,
            batch_size=args.batch_size,
            shuffle=False,
        )

        best_model = None
        best_val_mse = float("inf")
        best_seed = None

        if args.verbose >= 2:
            print("Train mean for current iteration:", train_target_mean)

        for seed in seed_candidates:
            set_random_seed(seed)

            candidate_model = build_seq_model(
                input_dim=X_train.shape[-1],
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                device=device,
                train_mean=train_target_mean,
            )
            candidate_model, candidate_val_mse = train_with_val(
                model=candidate_model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=args.epochs,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                device=device,
                patience=args.patience,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                verbose=max(args.verbose - 1, 0),
            )

            if candidate_val_mse < best_val_mse:
                best_val_mse = candidate_val_mse
                best_model = candidate_model
                best_seed = seed

        if best_model is None or best_seed is None:
            raise RuntimeError("Failed to train any seed candidate.")

        current_feat = X_test[i : i + 1]
        current_target = float(y_test[i])
        current_pred = float(predict_scalar(best_model, current_feat, device=device, batch_size=1)[0])
        error = current_pred - current_target
        sq_error = error ** 2
        baseline_sq_error = (current_target - train_target_mean) ** 2
        current_skill = 1.0 - (sq_error / baseline_sq_error) if baseline_sq_error > 0 else 0.0

        year_rows.append(
            {
                "Year": current_year,
                "Test_Index": i,
                "Train_Size_Used": int(X_train_aug.shape[0]),
                "Target": current_target,
                "Prediction": current_pred,
                "Error": error,
                "Squared_Error": sq_error,
                "Train_Target_Mean": train_target_mean,
                "Baseline_Squared_Error": baseline_sq_error,
                "Skill_Score": current_skill,
                "Best_Val_MSE": float(best_val_mse),
                "Best_Seed": int(best_seed),
            }
        )

        aggregate_targets.append(current_target)
        aggregate_preds.append(current_pred)
        aggregate_train_means.append(float(train_target_mean))
        aggregate_best_val_mse_sum += float(best_val_mse)

        if args.verbose >= 1:
            print(
                f"year={current_year} idx={i} train_size={X_train_aug.shape[0]} "
                f"best_seed={best_seed} "
                f"target={current_target:.4f} pred={current_pred:.4f} "
                f"mse={sq_error:.6f} skill={current_skill:.6f}"
            )

    targets_np = np.asarray(aggregate_targets, dtype=np.float32)
    preds_np = np.asarray(aggregate_preds, dtype=np.float32)
    train_means_np = np.asarray(aggregate_train_means, dtype=np.float32)

    mse = float(np.mean((preds_np - targets_np) ** 2))
    rmse = float(np.sqrt(mse))
    skill_score = float(compute_skill_score_from_baseline(targets_np, preds_np, train_means_np))

    summary_payload = {
        "model_name": model_name,
        "feature_subdir": feature_subdir,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "num_test_years": int(len(year_rows)),
        "sum_best_val_mse": float(aggregate_best_val_mse_sum),
        "mse": mse,
        "rmse": rmse,
        "skill_score": skill_score,
    }

    if save_outputs:
        results_dir = project_root / "results" / "tables"
        results_dir.mkdir(parents=True, exist_ok=True)

        predictions_path = results_dir / f"{model_name}_sequential_test_predictions.csv"
        pd.DataFrame(year_rows).to_csv(predictions_path, index=False)
        print(f"Saved sequential test predictions to: {predictions_path}")

        summary_path = results_dir / f"{model_name}_sequential_test_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary_payload, f, indent=2)
        print(f"Saved sequential summary to: {summary_path}")

        summary_table_path = results_dir / f"{model_name}_sequential_test_summary.csv"
        pd.DataFrame([summary_payload]).to_csv(summary_table_path, index=False)
        print(f"Saved sequential summary table to: {summary_table_path}")

    if args.verbose >= 1:
        print("Final Sequential Test Metrics")
        print(f"  MSE: {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  Skill Score: {skill_score:.6f}")
    return summary_payload


def run_hyperparam_search(args: argparse.Namespace) -> Dict[str, Any]:
    hidden_dims = (
        parse_search_values(args.hidden_dim_search, int)
        if args.hidden_dim_search
        else [args.hidden_dim]
    )
    num_layers_values = (
        parse_search_values(args.num_layers_search, int)
        if args.num_layers_search
        else [args.num_layers]
    )
    if not hidden_dims:
        raise ValueError("No hidden_dim choices found for search.")
    if not num_layers_values:
        raise ValueError("No num_layers choices found for search.")

    alpha_choices = [0.6, 1.0, 1.4]
    beta_choices = [0.6, 1.1, 1.6]
    gamma_choices = [1.6, 1.8, 2.0]

    all_combinations = [
        (hidden_dim, num_layers, alpha, beta, gamma)
        for hidden_dim in hidden_dims
        for num_layers in num_layers_values
        for alpha in alpha_choices
        for beta in beta_choices
        for gamma in gamma_choices
    ]
    if not all_combinations:
        raise ValueError("No hyperparameter combinations generated for search.")
    combinations = all_combinations
    total_trials = len(combinations)

    print(
        f"Running discrete grid search with {total_trials} combinations "
        f"(total possible: {len(all_combinations)}). "
        f"hidden_dim choices={hidden_dims}, num_layers choices={num_layers_values}, "
        f"alpha choices={alpha_choices}, "
        f"beta choices={beta_choices}, "
        f"gamma choices={gamma_choices}"
    )

    trial_rows: List[Dict[str, Any]] = []
    best_result: Optional[Dict[str, Any]] = None

    for trial_idx, (hidden_dim, num_layers, alpha, beta, gamma) in enumerate(combinations, start=1):
        trial_args = argparse.Namespace(**vars(args))
        trial_args.hidden_dim = int(hidden_dim)
        trial_args.num_layers = int(num_layers)
        trial_args.alpha = float(alpha)
        trial_args.beta = float(beta)
        trial_args.gamma = float(gamma)
        # Keep output concise: only print per-combination MSE from this function.
        trial_args.verbose = 0

        summary = run_sequential_test_eval(trial_args, save_outputs=False)

        row = {
            "trial": int(trial_idx),
            "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers),
            "alpha": float(alpha),
            "beta": float(beta),
            "gamma": float(gamma),
            "sum_best_val_mse": float(summary["sum_best_val_mse"]),
            "test_mse": float(summary["mse"]),
            "mse": float(summary["mse"]),
            "rmse": float(summary["rmse"]),
            "skill_score": float(summary["skill_score"]),
            "model_name": summary["model_name"],
        }
        trial_rows.append(row)

        if best_result is None or row["sum_best_val_mse"] < best_result["sum_best_val_mse"]:
            best_result = row

        print(
            f"trial={trial_idx} "
            f"hidden_dim={row['hidden_dim']} "
            f"num_layers={row['num_layers']} "
            f"alpha={row['alpha']:.3f} "
            f"beta={row['beta']:.3f} "
            f"gamma={row['gamma']:.3f} "
            f"sum_best_val_mse={row['sum_best_val_mse']:.6f} "
            f"test_mse={row['test_mse']:.6f}"
        )

    if best_result is None:
        raise RuntimeError("Hyperparameter search produced no results.")

    feature_tag = make_tag(Path(args.feature_subdir).name)
    search_name = (
        f"{args.model_name_prefix}_{feature_tag}_"
        f"search"
    )
    results_dir = project_root / "results" / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)

    trials_df = pd.DataFrame(trial_rows)

    top_k = max(1, min(args.top_k_best_trials, len(trial_rows)))
    best_trials_df = trials_df.sort_values("sum_best_val_mse", ascending=True).head(top_k)
    best_trials_csv_path = results_dir / f"{search_name}_best_trials.csv"
    best_trials_df.to_csv(best_trials_csv_path, index=False)
    print(f"Saved top-{top_k} best trials to: {best_trials_csv_path}")

    best_trials_json_path = results_dir / f"{search_name}_best_trials.json"
    with open(best_trials_json_path, "w") as f:
        json.dump(best_trials_df.to_dict(orient="records"), f, indent=2)
    print(f"Saved top-{top_k} best trials JSON to: {best_trials_json_path}")

    best_summary = {
        "search_name": search_name,
        "num_trials": int(len(trial_rows)),
        "best_value": float(best_result["sum_best_val_mse"]),
        "best_trial": int(best_result["trial"]),
        "best_model_name": best_result["model_name"],
        "best_hidden_dim": int(best_result["hidden_dim"]),
        "best_num_layers": int(best_result["num_layers"]),
        "best_alpha": float(best_result["alpha"]),
        "best_beta": float(best_result["beta"]),
        "best_gamma": float(best_result["gamma"]),
        "selection_metric": "sum_best_val_mse",
        "best_sum_best_val_mse": float(best_result["sum_best_val_mse"]),
        "best_mse": float(best_result["mse"]),
        "best_rmse": float(best_result["rmse"]),
        "best_skill_score": float(best_result["skill_score"]),
    }

    best_json_path = results_dir / f"{search_name}_best_summary.json"
    with open(best_json_path, "w") as f:
        json.dump(best_summary, f, indent=2)
    print(f"Saved best-search summary to: {best_json_path}")

    best_csv_path = results_dir / f"{search_name}_best_summary.csv"
    pd.DataFrame([best_summary]).to_csv(best_csv_path, index=False)
    print(f"Saved best-search summary table to: {best_csv_path}")

    print("\nBest Hyperparameter Combination")
    print(f"  Trial: {best_summary['best_trial']}")
    print(
        "  Params: "
        f"hidden_dim={best_summary['best_hidden_dim']}, "
        f"num_layers={best_summary['best_num_layers']}, "
        f"alpha={best_summary['best_alpha']:.6f}, "
        f"beta={best_summary['best_beta']:.6f}, "
        f"gamma={best_summary['best_gamma']:.6f}"
    )
    print(f"  Selection metric: {best_summary['selection_metric']}={best_summary['best_sum_best_val_mse']:.6f}")
    print(f"  Best MSE: {best_summary['best_mse']:.6f}")
    print(f"  Best RMSE: {best_summary['best_rmse']:.6f}")
    print(f"  Best Skill Score: {best_summary['best_skill_score']:.6f}")

    # Save predictions/summary artifacts once, using only the best trial's hyperparameters.
    best_args = argparse.Namespace(**vars(args))
    best_args.hidden_dim = int(best_result["hidden_dim"])
    best_args.num_layers = int(best_result["num_layers"])
    best_args.alpha = float(best_result["alpha"])
    best_args.beta = float(best_result["beta"])
    best_args.gamma = float(best_result["gamma"])
    best_args.verbose = 0
    run_sequential_test_eval(best_args, save_outputs=True)

    return best_summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="config/model_config.yml")
    parser.add_argument("--feature-root", type=str, default="features")
    parser.add_argument("--feature-subdir", type=str, default=DEFAULT_FEATURE_SUBDIR)
    parser.add_argument("--model-name-prefix", type=str, default=DEFAULT_MODEL_NAME_PREFIX)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.3008733912115933)
    parser.add_argument("--beta", type=float, default=1.0738407996082124)
    parser.add_argument("--gamma", type=float, default=1.8981156711075444)
    parser.add_argument(
        "--hidden-dim-search",
        type=str,
        default=DEFAULT_HIDDEN_DIM_SEARCH,
        help="Comma-separated hidden_dim values for grid search, e.g. '8,16,32'",
    )
    parser.add_argument(
        "--num-layers-search",
        type=str,
        default=DEFAULT_NUM_LAYERS_SEARCH,
        help="Comma-separated num_layers values for grid search, e.g. '1,2,3'",
    )
    parser.add_argument("--top-k-best-trials", type=int, default=DEFAULT_TOP_K_BEST_TRIALS)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--verbose", type=int, default=1)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run_hyperparam_search(args)


if __name__ == "__main__":
    main()
