import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


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