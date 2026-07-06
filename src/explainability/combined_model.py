import torch
import torch.nn as nn
import math
import copy
from captum.attr import IntegratedGradients


def prepare_input_btchw(
    data: torch.Tensor,
    device: torch.device,
    normalize_transform=None,
    num_time_steps: int = 13,
):
    
    B, TC, H, W = data.shape
    T = num_time_steps
    
    x = data.float().to(device)

    C = TC // T
    if normalize_transform is not None:
        # Apply normalization to each (T*C, H, W) sample
        normed = [normalize_transform(x[i]) for i in range(B)]
        x = torch.stack(normed, dim=0)

    # Reshape: (B, T*C, H, W) -> (B, T, C, H, W) -> (B*T, C, H, W)
    x = x.reshape(B, C, T, H, W).permute(0, 2, 1, 3, 4)
    x = x.reshape(B * T, C, H, W)
    return x, B, T
    
def get_1d_sincos_pos_embed(embed_dim, positions):
    """
    embed_dim must be even
    positions: (M,)
    returns: (M, embed_dim)
    """
    assert embed_dim % 2 == 0

    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2
    omega = 1.0 / (10000 ** omega)

    positions = positions.float().unsqueeze(1)
    out = positions * omega.unsqueeze(0)

    sin = torch.sin(out)
    cos = torch.cos(out)

    return torch.cat([sin, cos], dim=1)


def get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w):
    """
    Returns:
        (grid_h * grid_w, embed_dim)
    """

    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, torch.arange(grid_h))
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, torch.arange(grid_w))

    emb = torch.zeros(grid_h, grid_w, embed_dim)

    for i in range(grid_h):
        for j in range(grid_w):
            emb[i, j] = torch.cat([emb_h[i], emb_w[j]], dim=0)

    return emb.reshape(grid_h * grid_w, embed_dim)


class ClimateTransformerBackbone(nn.Module):

    def __init__(
        self,
        in_channels,
        embed_dim=256,
        patch_size=16,
        depth=6,
        num_heads=8,
        mlp_ratio=4,
        input_h=481,
        input_w=1440
    ):
        super().__init__()

        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.token_reduction = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            stride=2,
            padding=1
        )

        # compute token grid size
        h = input_h // patch_size
        w = input_w // patch_size

        h = math.ceil(h / 2)
        w = math.ceil(w / 2)

        num_tokens = h * w

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # positional encoding
        pos_embed = get_2d_sincos_pos_embed(embed_dim, h, w)
        pos_embed = torch.cat(
            [torch.zeros(1, embed_dim), pos_embed],
            dim=0
        )

        self.register_buffer(
            "pos_embed",
            pos_embed.unsqueeze(0),
            persistent=False
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dropout=0.0,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

        self.norm = nn.LayerNorm(embed_dim)

        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x):

        x = self.patch_embed(x)
        x = self.token_reduction(x)

        B, C, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = x + self.pos_embed[:, :x.size(1)]

        x = self.transformer(x)

        x = self.norm(x)

        x = x[:, 1:].mean(dim=1)

        return x
    
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
    

class CombinedModel(nn.Module):
    def __init__(self, backbone, regressor, device, normalize_transform=None, num_time_steps=13):
        super().__init__()
        self.backbone = backbone
        self.regressor = regressor
        self.device = device
        self.normalize_transform = normalize_transform
        self.num_time_steps = num_time_steps

    def forward(self, x):
        # x.shape = (B, T*C, H, W)

        x, B, T = prepare_input_btchw(x, self.device, self.normalize_transform, self.num_time_steps) # (B*T, C, H, W)
        
        embeddings, _ = self.backbone.student_encoder(x) # (B*T, embed_dim)
        embeddings = embeddings.reshape(B, T, -1) # (B, T, embed_dim)

        pred_seq = self.regressor(embeddings) # (B, T)

        return pred_seq[:, -1]  # return final timestep prediction for loss computation



# ---------------------------------------------------------
# Projector (BYOL)
# ---------------------------------------------------------

class Projector(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        
        # Standard BYOL Projector: Linear -> BN -> Act -> Linear
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False), # No bias before BN
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim, bias=False) # No norm or act at the end
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------
# Encoder
# ---------------------------------------------------------

class Encoder(nn.Module):

    """
    Generic encoder wrapper.
    Backbone must output (B*T, feature_dim)
    """

    def __init__(self, backbone, backbone_dim, embed_dim=256, proj_dim=1024):

        super().__init__()

        self.backbone = backbone

        # explicit dimension (no LazyLinear)
        self.embedding = nn.Linear(backbone_dim, embed_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        if self.embedding.bias is not None:
            nn.init.constant_(self.embedding.bias, 0)

        self.projector = Projector(
            embed_dim,
            embed_dim * 4,
            proj_dim
        )

    def forward(self, x):

        feats = self.backbone(x)

        embed = self.embedding(feats)

        proj = self.projector(embed)

        return embed, proj


# ---------------------------------------------------------
# Predictor
# ---------------------------------------------------------

class Predictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        # Standard BYOL Predictor: Linear -> BN -> Act -> Linear
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4, bias=False), # No bias before BN
            nn.BatchNorm1d(dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim, bias=False) # No norm or act at the end
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------
# Main BYOL Model
# ---------------------------------------------------------

class DinoBYOLCNN(nn.Module):

    def __init__(self, backbone, backbone_dim, embed_dim=256, proj_dim=1024):

        super().__init__()

        # student
        student_backbone = backbone

        # teacher
        teacher_backbone = copy.deepcopy(backbone)

        self.student_encoder = Encoder(
            student_backbone,
            backbone_dim,
            embed_dim,
            proj_dim
        )

        self.teacher_encoder = Encoder(
            teacher_backbone,
            backbone_dim,
            embed_dim,
            proj_dim
        )

        self.predictor = Predictor(proj_dim)

        # freeze teacher
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False


    @torch.no_grad()
    def update_teacher(self, momentum=0.996):

        for ps, pt in zip(
            self.student_encoder.parameters(),
            self.teacher_encoder.parameters()
        ):
            pt.data = momentum * pt.data + (1 - momentum) * ps.data


    def forward(self, x1, x2):

        # student branch
        student_embed, student_proj = self.student_encoder(x1)
        pred = self.predictor(student_proj)

        # teacher branch
        with torch.no_grad():
            self.teacher_encoder.eval()
            _, teacher_proj = self.teacher_encoder(x2)

        return pred, teacher_proj, student_embed
    
class ExplainabilityWrapper:
    def __init__(self, model: nn.Module, device: torch.device, num_time_steps: int):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.num_time_steps = num_time_steps

        self.ig = IntegratedGradients(self.forward_func)

    def forward_func(self, x):
        return self.model(x)  # (B,)

    def attribute(
        self,
        x: torch.Tensor,
        baseline: torch.Tensor = None,
        n_steps: int = 50,
        normalize: bool = True,
    ):
        x = x.to(self.device)

        # ---- baseline ----
        if baseline is None:
            # better than zero for geospatial data
            baseline = x.mean(dim=(2, 3), keepdim=True)
            baseline = baseline.expand_as(x)

        # ---- IG ----
        attr = self.ig.attribute(
            inputs=x,
            baselines=baseline,
            n_steps=n_steps,
            internal_batch_size=8,
        )
        # (B, T*C, H, W)

        # ---- reshape ----
        B, TC, H, W = attr.shape
        T = self.num_time_steps
        C = TC // T

        attr = attr.reshape(B, C, T, H, W).permute(0, 2, 1, 3, 4)
        # (B, T, C, H, W)

        # ---- optional normalization (IMPORTANT) ----
        if normalize:
            denom = attr.abs().sum(dim=(1, 2, 3, 4), keepdim=True) + 1e-8
            attr = attr / denom

        abs_attr = attr.abs()

        # ---- outputs you care about ----

        # 1. Full maps → EXACTLY what you asked for
        maps = attr  # (B, T, C, H, W)

        # 2. (T, C) importance matrix
        tc_importance = abs_attr.sum(dim=(-1, -2))  # (B, T, C)

        # 3. Temporal importance
        t_importance = tc_importance.sum(dim=-1)  # (B, T)

        # 4. Spatial maps per timestep (optional but useful)
        spatial_maps = abs_attr.sum(dim=2)  # (B, T, H, W)

        return {
            "maps": maps,                        # (B, T, C, H, W)
            "tc_importance": tc_importance,      # (B, T, C)
            "t_importance": t_importance,        # (B, T)
            "spatial_maps": spatial_maps,        # (B, T, H, W)
        }

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