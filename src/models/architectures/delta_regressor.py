import torch
import torch.nn as nn

class DeltaSeqRegressor(nn.Module):
    """
    Predicts deltas and accumulates them:
        y_t = y_0 + sum_{i<=t} delta_i

    Works for negative targets as well.
    """

    def __init__(self, input_dim=512, hidden_dim=128, num_layers=1, T=13):
        super().__init__()

        self.T = T

        self.rnn = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.time_embed = nn.Embedding(T, hidden_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Predict delta instead of absolute value
        self.delta_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # learnable initial prediction
        self.y0 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        x: (B, T, F)
        returns: (B, T)
        """
        B, T, _ = x.shape

        h, _ = self.rnn(x)

        # timestep embedding
        t_idx = torch.arange(T, device=x.device)
        h = h + self.time_embed(t_idx).unsqueeze(0)

        # residual connection
        h = h + self.input_proj(x)

        # predict deltas
        deltas = self.delta_head(h).squeeze(-1)  # (B, T)

        # cumulative sum → predictions
        y_hat = torch.cumsum(deltas, dim=1)

        # add initial bias
        y_hat = y_hat + self.y0

        return y_hat
    
class DeltaLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.3):
        super().__init__()
        self.alpha = alpha  # ranking
        self.beta = beta    # delta regularization
        self.mse = nn.MSELoss()

    def forward(self, preds, target):
        """
        preds: (B, T)
        target: (B,)
        """
        B, T = preds.shape
        target = target.unsqueeze(1)

        # --------------------------------------------------
        # 1. FINAL LOSS (main objective)
        # --------------------------------------------------
        loss_final = self.mse(preds[:, -1], target.squeeze(1))

        # --------------------------------------------------
        # 2. ERROR
        # --------------------------------------------------
        err = (preds - target) ** 2

        # --------------------------------------------------
        # 3. GLOBAL MONOTONICITY (ranking)
        # --------------------------------------------------
        diff = err.unsqueeze(2) - err.unsqueeze(1)
        mask = torch.triu(torch.ones(T, T, device=preds.device), diagonal=1)

        loss_rank = (torch.relu(-diff) * mask).mean()

        # --------------------------------------------------
        # 4. DELTA REGULARIZATION (prevents explosion)
        # --------------------------------------------------
        deltas = preds[:, 1:] - preds[:, :-1]
        loss_delta = (deltas ** 2).mean()

        # --------------------------------------------------
        # TOTAL
        # --------------------------------------------------
        total = (
            loss_final +
            self.alpha * loss_rank +
            self.beta * loss_delta
        ) / (1 + self.alpha + self.beta)

        return total
    
class DeltaLossV2(nn.Module):
    def __init__(self, alpha=1.0, beta=0.3, gamma=0.5, T=13, mode="linear"):
        super().__init__()
        self.alpha = alpha  # ranking
        self.beta = beta    # delta smoothness
        self.gamma = gamma  # time-weighted loss

        # ----------------------------------------
        # build timestep weights
        # ----------------------------------------
        t = torch.arange(1, T + 1).float()

        if mode == "linear":
            weights = t
        elif mode == "exp":
            weights = torch.exp(t / T)   # strong late emphasis
        elif mode == "quadratic":
            weights = (t / T) ** 2
        else:
            raise ValueError("Invalid mode")

        weights = weights / weights.sum()
        self.register_buffer("weights", weights)

        self.mse = nn.MSELoss()

    def forward(self, preds, target):
        """
        preds: (B, T)
        target: (B,)
        """
        B, T = preds.shape
        target = target.unsqueeze(1)

        # --------------------------------------------------
        # 1. FINAL LOSS (still anchor)
        # --------------------------------------------------
        loss_final = self.mse(preds[:, -1], target.squeeze(1))

        # --------------------------------------------------
        # 2. ERROR
        # --------------------------------------------------
        err = (preds - target) ** 2  # (B, T)

        # --------------------------------------------------
        # 3. TIME-WEIGHTED SUPERVISION (NEW)
        # --------------------------------------------------
        loss_time = (err * self.weights.unsqueeze(0)).mean()

        # --------------------------------------------------
        # 4. GLOBAL RANKING (monotonic improvement)
        # --------------------------------------------------
        diff = err.unsqueeze(2) - err.unsqueeze(1)
        mask = torch.triu(torch.ones(T, T, device=preds.device), diagonal=1)

        loss_rank = (torch.relu(-diff) * mask).mean()

        # --------------------------------------------------
        # 5. DELTA REGULARIZATION
        # --------------------------------------------------
        deltas = preds[:, 1:] - preds[:, :-1]
        loss_delta = ((deltas ** 2) * self.weights[1:].unsqueeze(0)).mean()  # time-weighted smoothness

        # --------------------------------------------------
        # TOTAL
        # --------------------------------------------------
        total = (
            loss_final +
            self.alpha * loss_rank +
            self.beta * loss_delta +
            self.gamma * loss_time
        ) / (1 + self.alpha + self.beta + self.gamma)

        return total
    
class DeltaLossV3(nn.Module):
    def __init__(self, alpha=1.0, beta=0.3, gamma=0.5, T=13, mode="linear"):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.T = T

        t = torch.arange(1, T + 1).float()

        if mode == "linear":
            weights = t
        elif mode == "exp":
            weights = torch.exp(t / T)
        elif mode == "quadratic":
            weights = (t / T) ** 2
        else:
            raise ValueError("Invalid mode")

        weights = weights / weights.sum()
        self.register_buffer("weights", weights)

        self.mse = nn.MSELoss()

    def forward(self, preds, target):
        B, T = preds.shape
        target = target.unsqueeze(1)

        # --------------------------------------------------
        # FINAL LOSS
        # --------------------------------------------------
        loss_final = self.mse(preds[:, -1], target.squeeze(1))

        # --------------------------------------------------
        # ERROR
        # --------------------------------------------------
        err = (preds - target) ** 2

        # --------------------------------------------------
        # TIME-WEIGHTED SUPERVISION
        # --------------------------------------------------
        loss_time = (err * self.weights.unsqueeze(0)).mean()

        # --------------------------------------------------
        # TIME-AWARE GLOBAL RANKING
        # --------------------------------------------------
        diff = err.unsqueeze(2) - err.unsqueeze(1)

        mask = torch.triu(torch.ones(T, T, device=preds.device), diagonal=1)

        # weight by importance of later timestep
        w_j = self.weights.unsqueeze(0)  # (1, T)

        # distance weighting (long-range violations matter more)
        dist = torch.arange(T, device=preds.device).float()
        dist_matrix = dist.unsqueeze(0) - dist.unsqueeze(1)
        dist_matrix = torch.relu(dist_matrix)
        dist_matrix = dist_matrix / (T - 1 + 1e-6)

        pair_weights = w_j.unsqueeze(1) * (1 + dist_matrix)

        loss_rank = torch.relu(-diff) * mask * pair_weights
        loss_rank = loss_rank.mean()

        # --------------------------------------------------
        # DELTA SMOOTHNESS (time-weighted)
        # --------------------------------------------------
        deltas = preds[:, 1:] - preds[:, :-1]
        loss_delta = ((deltas ** 2) * self.weights[1:].unsqueeze(0)).mean()

        # --------------------------------------------------
        # TOTAL
        # --------------------------------------------------
        total = (
            loss_final +
            self.alpha * loss_rank +
            self.beta * loss_delta +
            self.gamma * loss_time
        ) / (1 + self.alpha + self.beta + self.gamma)

        return total