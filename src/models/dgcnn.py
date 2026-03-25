"""
DGCNN with shared encoder + dual prediction heads.

Head A: Metric tensor  (s1, s2, c, s) → M ∈ SPD(2)          §3.3
Head B: Confidence     scalar in [0,1]  (direction reliability)
Head C: Singularity    ternary logits  ∈ ℝ³  → {−1,0,+1}/4  §3.4

The shared DGCNN encoder is unchanged from the baseline.  Both heads decode
from the same global descriptor z_i ∈ ℝ^{2·emb_dims}.

Backward compatibility
──────────────────────
Old checkpoints that only contain metric-head weights can still be loaded
by passing ``load_legacy=True`` to ``load_legacy_checkpoint()``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Graph-feature helpers (unchanged from baseline)
# ---------------------------------------------------------------------------

def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """k-nearest neighbour indices for each point in a batch."""
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

    x_t     = x.transpose(2, 1).contiguous()                       # (B, N, D)
    feature = x_t.view(batch_size * num_points, -1)[idx_flat, :]   # (B*N*k, D)
    feature = feature.view(batch_size, num_points, k, num_dims)
    x_rep   = x_t.view(batch_size, num_points, 1, num_dims).expand_as(feature)

    return torch.cat([feature - x_rep, x_rep], dim=3).permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# Shared DGCNN encoder
# ---------------------------------------------------------------------------

class _DGCNNEncoder(nn.Module):
    def __init__(self, k: int = 20, emb_dims: int = 256, dropout: float = 0.5,
                 in_dims: int = 3):
        """
        Args:
            in_dims: number of feature dimensions per point.
                     3  → coordinates only (original)
                     6  → coordinates + surface normals
        """
        super().__init__()
        self.k = k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        # get_graph_feature doubles in_dims (concatenates [neighbour−centre, centre])
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
        """
        Args:
            x:  (B, in_dims, N)  point features in LCF.
        Returns:
            z:  (B, 256)   global descriptor.
        """
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1)[0]                    # (B, 64, N)

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
        z = torch.cat([x_max, x_avg], dim=1)     # (B, 2·emb_dims)

        z = F.leaky_relu(self.bn6(self.linear1(z)), 0.2)
        z = self.dp1(z)
        z = F.leaky_relu(self.bn7(self.linear2(z)), 0.2)
        z = self.dp2(z)
        return z                                  # (B, 256)


# ---------------------------------------------------------------------------
# DGCNN with metric head (Head A) + singularity head (Head B)
# ---------------------------------------------------------------------------

class DGCNN(nn.Module):
    """
    Dynamic Graph CNN — metric + confidence (+ optional singularity).

    Output tensor shape:
        (B, 4)  metric only (legacy)
        (B, 5)  metric + confidence
        (B, 8)  metric + confidence + singularity logits
        [:, 0:4]  — Head A: (s1, s2, c, s)  metric parameters
        [:, 4]    — Head B: confidence in [0,1]
        [:, 5:8]  — Head C: (l_neg, l_zero, l_pos)  singularity logits

    Head A decoding
    ───────────────
        s1, s2 > 0   shape factors via softplus
        (c, s)       normalised (cos 2θ, sin 2θ) direction vector
        M = R(θ) diag(s1², s2²) R(θ)ᵀ  ∈ SPD(2)

    Head B decoding
    ───────────────
        Ternary logits for classes {−1/4, 0, +1/4} (negative / regular / positive
        singularity).  Use softmax for probabilities; argmax for hard label.
    """

    def __init__(self, k: int = 20, emb_dims: int = 256, dropout: float = 0.5,
                 in_dims: int = 3, predict_singularity: bool = False,
                 predict_confidence: bool = True,
                 max_log_half: float = 1.5):
        """
        Args:
            in_dims: feature dims per point (3=coords only, 6=coords+normals).
        """
        super().__init__()
        self.k = k
        self.predict_singularity = bool(predict_singularity)
        self.predict_confidence = bool(predict_confidence)
        self.max_log_half = float(max_log_half)
        self.encoder = _DGCNNEncoder(k=k, emb_dims=emb_dims, dropout=dropout,
                                     in_dims=in_dims)

        # Head A: metric  (4 outputs)
        self.metric_head = nn.Linear(256, 4)

        # Head B: confidence (1 output; sigmoid applied in forward)
        if self.predict_confidence:
            self.conf_head = nn.Sequential(
                nn.Linear(256, 64),
                nn.LeakyReLU(0.2),
                nn.Linear(64, 1),
            )

        # Head C: singularity  (3 outputs — logits for {-1, 0, +1} classes)
        if self.predict_singularity:
            self.sing_head = nn.Sequential(
                nn.Linear(256, 64),
                nn.LeakyReLU(0.2),
                nn.Linear(64, 3),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:  (B, in_dims, N)  LCF-transformed features (coords [+ normals]).

        Returns:
            out: (B, 4) or (B, 5) or (B, 8)
        """
        z = self.encoder(x)                               # (B, 256)

        # ── Head A: metric ──────────────────────────────────────────────
        raw_m = self.metric_head(z)                        # (B, 4)
        mean_raw, half_raw, c_raw, s_raw = raw_m.unbind(dim=1)

        # Log-space eigenvalue parameterisation (no sort needed):
        #
        #   log√λ_max = mean_raw + softplus(half_raw)
        #   log√λ_min = mean_raw − softplus(half_raw)
        #
        # softplus(half_raw) ≥ 0 guarantees s1 ≥ s2 by construction, so
        # params_to_tensor's internal sort([s1², s2²]) is ALWAYS a no-op.
        #
        # The old (s1_raw, s2_raw) + sort approach oscillated: whenever the
        # network predicted s1≈s2 (common early in training), the sort randomly
        # permuted which variable received which gradient, causing the eigenvalue
        # heads to swap roles back and forth rather than converging.  This was the
        # primary cause of the metric-loss training floor at ~0.35.
        #
        # Ranges: mean_raw clamped to [−3.5, 1.5] → geometric-mean √λ ∈ [e⁻³·⁵, e¹·⁵]
        #         softplus(half_raw) ∈ [0, ∞)     → log-ratio  log(λ_max/λ_min)/2 ≥ 0
        # Combined clamp ensures s1,s2 ∈ [e⁻⁷, e³] as before.
        log_mean = mean_raw.clamp(-3.5, 1.5)
        log_half = F.softplus(half_raw).clamp(max=self.max_log_half)
        s1 = torch.exp((log_mean + log_half).clamp(-7.0, 3.0))   # sqrt(λ_max)
        s2 = torch.exp((log_mean - log_half).clamp(-7.0, 3.0))   # sqrt(λ_min) ≤ s1
        norm = torch.sqrt(c_raw ** 2 + s_raw ** 2 + 1e-8)
        c    = c_raw / norm
        s    = s_raw / norm

        parts = [torch.stack([s1, s2, c, s], dim=1)]      # (B,4)
        if self.predict_confidence:
            conf = torch.sigmoid(self.conf_head(z)).squeeze(-1)  # (B,)
            parts.append(conf[:, None])                    # (B,1)
        if not self.predict_singularity:
            return torch.cat(parts, dim=1)

        # ── Head C: singularity ─────────────────────────────────────────
        sing_logits = self.sing_head(z)                    # (B, 3)
        parts.append(sing_logits)
        return torch.cat(parts, dim=1)

    # ------------------------------------------------------------------
    # Convenience decoders
    # ------------------------------------------------------------------

    @staticmethod
    def decode_metric(out: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct (B, 2, 2) metric tensors from network output (B, 7).
        """
        from src.geometry.metric_utils import params_to_tensor
        s1 = out[:, 0]
        s2 = out[:, 1]
        c  = out[:, 2]
        s  = out[:, 3]
        return params_to_tensor(s1, s2, c, s)             # (B, 2, 2)

    @staticmethod
    def decode_singularity_probs(out: torch.Tensor) -> torch.Tensor:
        """
        Softmax class probabilities for singularity labels.  (B, 3).
        """
        if out.shape[1] >= 8:
            return F.softmax(out[:, 5:8], dim=-1)
        if out.shape[1] >= 7:
            return F.softmax(out[:, 4:7], dim=-1)
        raise ValueError("No singularity logits in output.")

    @staticmethod
    def decode_singularity_index(out: torch.Tensor) -> torch.Tensor:
        """
        Hard singularity index in {−0.25, 0.0, +0.25}.  (B,).
        """
        if out.shape[1] >= 8:
            logits = out[:, 5:8]
        elif out.shape[1] >= 7:
            logits = out[:, 4:7]
        else:
            raise ValueError("No singularity logits in output.")
        classes = logits.argmax(dim=-1)              # 0,1,2
        index_map = torch.tensor([-0.25, 0.0, 0.25], device=out.device, dtype=out.dtype)
        return index_map[classes]

    @staticmethod
    def decode_confidence(out: torch.Tensor) -> torch.Tensor:
        """
        Confidence in [0,1]. If absent (legacy output), return ones.
        """
        if out.shape[1] in (5, 8):
            return out[:, 4].clamp(0.0, 1.0)
        return torch.ones(out.shape[0], device=out.device, dtype=out.dtype)


# ---------------------------------------------------------------------------
# Legacy checkpoint loader
# ---------------------------------------------------------------------------

def infer_in_dims_from_checkpoint(state_dict: dict) -> int:
    """
    Infer in_dims from the saved conv1 weight shape.

    get_graph_feature doubles the feature dim, so:
        conv1.weight.shape[1] = in_dims * 2
    Works regardless of whether the checkpoint config contains 'in_dims'.
    """
    for key in ('encoder.conv1.0.weight', 'conv1.0.weight'):
        if key in state_dict:
            return state_dict[key].shape[1] // 2
    return 3  # safe default: 3D-only input


def infer_predict_singularity_from_checkpoint(state_dict: dict) -> bool:
    """
    Infer whether checkpoint contains a trained singularity head.
    """
    return any(k.startswith('sing_head.') for k in state_dict.keys())


def infer_predict_confidence_from_checkpoint(state_dict: dict) -> bool:
    """
    Infer whether checkpoint contains a trained confidence head.
    """
    return any(k.startswith('conf_head.') for k in state_dict.keys())


def load_legacy_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    k: int = 20,
    emb_dims: int = 256,
    dropout: float = 0.5,
) -> 'DGCNN':
    """
    Load a checkpoint trained with the old 4-output DGCNN.

    The shared encoder and metric-head weights are restored; the singularity
    head is left randomly initialised.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)

    in_dims = infer_in_dims_from_checkpoint(state)
    model = DGCNN(
        k=k, emb_dims=emb_dims, dropout=dropout, in_dims=in_dims,
        predict_singularity=infer_predict_singularity_from_checkpoint(state),
        predict_confidence=infer_predict_confidence_from_checkpoint(state),
    ).to(device)

    # Remap old flat parameter names to new encoder-namespaced names
    new_state = {}
    for key, val in state.items():
        if key.startswith('bn') or key.startswith('conv') or key.startswith('linear') or key.startswith('dp'):
            # Old keys like 'bn1.weight' → 'encoder.bn1.weight'
            new_key = 'encoder.' + key
        else:
            new_key = key
        new_state[new_key] = val

    # Also handle the old linear3 → metric_head rename
    for old, new in [('encoder.linear3.weight', 'metric_head.weight'),
                     ('encoder.linear3.bias',   'metric_head.bias')]:
        if old in new_state:
            new_state[new] = new_state.pop(old)

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"[DGCNN] Missing keys (new head, OK): {missing}")
    if unexpected:
        print(f"[DGCNN] Unexpected keys (ignored): {unexpected}")

    return model
