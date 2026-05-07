"""
DGCNN with shared encoder + metric head (+ optional confidence head).

Head A: Metric tensor  (s1, s2, c, s) → M ∈ SPD(2)          §3.3
Head B: Confidence     scalar in [0,1]  (anisotropy reliability)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    k = max(1, min(int(k), int(x.size(-1))))
    with torch.no_grad():
        inner  = -2.0 * torch.matmul(x.transpose(2, 1), x)
        xx     = torch.sum(x ** 2, dim=1, keepdim=True)
        dist   = -xx - inner - xx.transpose(2, 1)
        idx    = dist.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x: torch.Tensor, k: int = 20, idx=None) -> torch.Tensor:
    """Edge features: concatenate (neighbour − centre, centre)."""
    batch_size, num_dims, num_points = x.size()
    k = max(1, min(int(k), int(num_points)))
    if idx is None:
        idx = knn(x, k=k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx_flat = (idx + idx_base).view(-1)

    x_t     = x.transpose(2, 1).contiguous()
    feature = x_t.view(batch_size * num_points, -1)[idx_flat, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x_rep   = x_t.view(batch_size, num_points, 1, num_dims).expand_as(feature)

    return torch.cat([feature - x_rep, x_rep], dim=3).permute(0, 3, 1, 2).contiguous()


class _DGCNNEncoder(nn.Module):
    def __init__(self, k: int = 20, emb_dims: int = 256, dropout: float = 0.5,
                 in_dims: int = 3):
        super().__init__()
        self.k = k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        # get_graph_feature doubles in_dims (concat [neighbour−centre, centre])
        self.conv1 = nn.Sequential(nn.Conv2d(in_dims * 2, 64,  1, bias=False), self.bn1, nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(128,          64,  1, bias=False), self.bn2, nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128,          128, 1, bias=False), self.bn3, nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(256,          256, 1, bias=False), self.bn4, nn.LeakyReLU(0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, 1, bias=False), self.bn5, nn.LeakyReLU(0.2))

        self.linear1 = nn.Linear(emb_dims * 2, 512, bias=False)
        self.bn6  = nn.BatchNorm1d(512)
        self.dp1  = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7  = nn.BatchNorm1d(256)
        self.dp2  = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1)[0]

        x = self.conv5(torch.cat([x1, x2, x3, x4], dim=1))
        x_max = F.adaptive_max_pool1d(x, 1).view(x.size(0), -1)
        x_avg = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
        z = torch.cat([x_max, x_avg], dim=1)

        z = F.leaky_relu(self.bn6(self.linear1(z)), 0.2)
        z = self.dp1(z)
        z = F.leaky_relu(self.bn7(self.linear2(z)), 0.2)
        z = self.dp2(z)
        return z


class DGCNN(nn.Module):
    """
    Output tensor:
        (B, 4)  metric only        — predict_confidence=False
        (B, 5)  metric + confidence — predict_confidence=True (default)

        [:, 0:4]  Head A: (s1, s2, c, s)  metric parameters
        [:, 4]    Head B: confidence ∈ [0,1]

    Head A decoding:
        s1, s2 > 0 with s1 ≥ s2 by construction (softplus log-half parameterisation)
        (c, s) normalised (cos 2θ, sin 2θ)
        M = R(θ) diag(s1², s2²) R(θ)ᵀ
    """

    def __init__(self, k: int = 20, emb_dims: int = 256, dropout: float = 0.5,
                 in_dims: int = 3,
                 predict_confidence: bool = True,
                 max_log_half: float = 1.5):
        super().__init__()
        self.k = k
        self.predict_confidence = bool(predict_confidence)
        self.max_log_half = float(max_log_half)
        self.encoder = _DGCNNEncoder(k=k, emb_dims=emb_dims, dropout=dropout,
                                     in_dims=in_dims)

        self.metric_head = nn.Linear(256, 4)

        if self.predict_confidence:
            self.conf_head = nn.Sequential(
                nn.Linear(256, 64),
                nn.LeakyReLU(0.2),
                nn.Linear(64, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)

        raw_m = self.metric_head(z)
        mean_raw, half_raw, c_raw, s_raw = raw_m.unbind(dim=1)

        # Log-space eigenvalue parameterisation; softplus guarantees s1 ≥ s2
        # so params_to_tensor's internal sort is always a no-op. The previous
        # (s1_raw, s2_raw) + sort approach randomly swapped roles when s1≈s2,
        # which was the primary cause of the metric-loss training floor at ~0.35.
        log_mean = mean_raw.clamp(-3.5, 1.5)
        log_half = F.softplus(half_raw).clamp(max=self.max_log_half)
        s1 = torch.exp((log_mean + log_half).clamp(-7.0, 3.0))
        s2 = torch.exp((log_mean - log_half).clamp(-7.0, 3.0))
        norm = torch.sqrt(c_raw ** 2 + s_raw ** 2 + 1e-8)
        c    = c_raw / norm
        s    = s_raw / norm

        parts = [torch.stack([s1, s2, c, s], dim=1)]
        if self.predict_confidence:
            conf = torch.sigmoid(self.conf_head(z)).squeeze(-1)
            parts.append(conf[:, None])
        return torch.cat(parts, dim=1)

    @staticmethod
    def decode_metric(out: torch.Tensor) -> torch.Tensor:
        from src.geometry.metric_utils import params_to_tensor
        return params_to_tensor(out[:, 0], out[:, 1], out[:, 2], out[:, 3])

    @staticmethod
    def decode_confidence(out: torch.Tensor) -> torch.Tensor:
        if out.shape[1] >= 5:
            return out[:, 4].clamp(0.0, 1.0)
        return torch.ones(out.shape[0], device=out.device, dtype=out.dtype)


def infer_in_dims_from_checkpoint(state_dict: dict) -> int:
    """
    Infer in_dims from saved conv1 weight shape:
        conv1.weight.shape[1] = in_dims * 2  (get_graph_feature doubles dims)
    """
    for key in ('encoder.conv1.0.weight', 'conv1.0.weight'):
        if key in state_dict:
            return state_dict[key].shape[1] // 2
    return 3


def infer_predict_confidence_from_checkpoint(state_dict: dict) -> bool:
    return any(k.startswith('conf_head.') for k in state_dict.keys())
