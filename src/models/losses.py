"""
Loss functions for NPAQ training.

Active terms:
    LogEuclideanLoss    — ||log M_pred − log M_gt||_F²
    DirectionalLoss     — anisotropy-weighted principal-direction alignment
    JacobianLoss        — ||log(Ĵ^T Ĵ) − log(M*^{−1})||_F²
    AntiFlipLoss        — determinant barrier (prevents inverted quads)
    Topology proxy      — metric determinant/condition barriers

Legacy singularity terms (kept for backward compatibility) remain implemented
but are disabled by default in training (lambda_sing=lambda_ph=0).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np

from src.geometry.metric_utils import eigh2x2


# ---------------------------------------------------------------------------
# Helpers: matrix log for 2×2 SPD matrices
# ---------------------------------------------------------------------------

def matrix_log_spd2(M: torch.Tensor) -> torch.Tensor:
    """
    Compute log(M) for a batch of 2×2 SPD matrices via eigendecomposition.

    Args:
        M:  (*, 2, 2) symmetric positive definite matrices.

    Returns:
        logM:  (*, 2, 2)
    """
    eigvals, eigvecs = eigh2x2(M)                               # (*,2), (*,2,2)
    log_eig = torch.log(eigvals.clamp(min=1e-8))               # (*,2)
    return eigvecs @ torch.diag_embed(log_eig) @ eigvecs.transpose(-2, -1)


# ---------------------------------------------------------------------------
# Metric head losses
# ---------------------------------------------------------------------------

class LogEuclideanLoss(nn.Module):
    """
    Log-Euclidean distance on SPD(2):

        L_metric = (1/B) Σ_b ||log M_pred_b − log M_gt_b||_F²
    """
    def forward(self, M_pred: torch.Tensor, M_gt: torch.Tensor) -> torch.Tensor:
        return (matrix_log_spd2(M_pred) - matrix_log_spd2(M_gt)).pow(2).sum(dim=(-2, -1)).mean()


class DirectionalLoss(nn.Module):
    """
    4-RoSy direction alignment loss (§3.5):

        L_dir = (1 − cos(4 Δθ)) / 2   ∈ [0, 1]

    where Δθ = θ_pred − θ_gt and u = exp(4iθ) = (cos 4θ, sin 4θ).

    This is invariant to 90° rotations — the correct symmetry for a
    cross-field: θ and θ+k·90° represent the same field, so the loss
    is zero whenever predictions are correct up to a quarter-turn.

    Previous 2-RoSy formulation (abs-cos) assigned *maximum* penalty
    (loss=1) to a 90°-off prediction, which directly conflicts with
    LogEuclideanLoss (loss=0 for the same prediction on an isotropic
    metric). That gradient conflict was the primary cause of the 0.38
    training floor.

    Inputs must be **2-D unit vectors** in the Local Canonical Frame.
    d2_pred / d2_gt are accepted for backward compatibility but ignored
    (4-RoSy handles the 90° symmetry automatically).

    Optional per-sample ``weights`` (B,) — set to anisotropy degree
    w = (λ_max − λ_min)/(λ_max + λ_min + ε) so that isotropic points
    (sphere: w≈0) contribute nothing to the directional gradient.
    """

    def __init__(self, isotropy_stop_threshold: float = 0.0):
        super().__init__()
        self.isotropy_stop_threshold = float(isotropy_stop_threshold)

    @staticmethod
    def _to_4rosy(d: torch.Tensor) -> torch.Tensor:
        """
        Map 2-D unit vector d = (dx, dy) → u = (cos 4θ, sin 4θ).

        u is invariant to 90° rotations:
            exp(4i(θ + k·π/2)) = exp(4iθ) · exp(2πik) = exp(4iθ)  ∀k∈Z
        """
        dx, dy = d[..., 0], d[..., 1]
        c2 = dx * dx - dy * dy      # cos(2θ)
        s2 = 2.0 * dx * dy          # sin(2θ)
        c4 = c2 * c2 - s2 * s2     # cos(4θ)
        s4 = 2.0 * c2 * s2         # sin(4θ)
        return torch.stack([c4, s4], dim=-1)   # (..., 2)

    def forward(
        self,
        d1_pred: torch.Tensor, d1_gt: torch.Tensor,
        d2_pred: Optional[torch.Tensor] = None,   # kept for API compat, unused
        d2_gt:   Optional[torch.Tensor] = None,   # kept for API compat, unused
        weights: Optional[torch.Tensor] = None,   # (B,) in [0, 1]
    ) -> torch.Tensor:
        u_pred = self._to_4rosy(d1_pred)           # (B, 2)
        u_gt   = self._to_4rosy(d1_gt)             # (B, 2)
        # cos(4Δθ) = Re(u_pred · ū_gt) = dot product of 2-D unit-circle points
        alignment = (u_pred * u_gt).sum(dim=-1)    # (B,), ∈ [−1, 1]
        loss = (1.0 - alignment) * 0.5             # (B,), ∈ [0, 1]
        if weights is not None:
            if self.isotropy_stop_threshold > 0.0:
                weights = weights * (weights >= self.isotropy_stop_threshold).to(weights.dtype)
            w_sum = weights.sum().clamp(min=1e-8)
            return (weights * loss).sum() / w_sum
        return loss.mean()


# ---------------------------------------------------------------------------
# Singularity head losses
# ---------------------------------------------------------------------------

class SingularityLoss(nn.Module):
    """
    Cross-entropy loss for ternary singularity classification (§3.5):

        L_sing = −(1/N) Σ_i log softmax(σ_i)[c_i*]

    Args:
        logits:    (B, 3) raw logits [l_neg, l_zero, l_pos].
        labels:    (B,)   long tensor with class indices {0, 1, 2}
                          (0 = negative singularity, 1 = regular, 2 = positive).
    """
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels)


class PoincareHopfLoss(nn.Module):
    """
    Soft Poincaré–Hopf constraint (§3.5):

        L_PH = (Σ_i ŝ_i · softmax(σ_i) − χ(S)/4)²

    where ŝ_i ∈ {−0.25, 0.0, +0.25} and χ(S) is the Euler characteristic.

    Args:
        logits:      (B, 3)  singularity logits.
        euler_char:  scalar or (1,) — Euler characteristic of the surface.
                     For a closed genus-0 surface: χ = 2.
    """
    def forward(
        self,
        logits: torch.Tensor,
        euler_char: float = 2.0,
    ) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)                  # (B, 3)
        index_vals = torch.tensor(
            [-0.25, 0.0, 0.25], dtype=logits.dtype, device=logits.device
        )
        # Soft expected singularity index per point
        soft_index = (probs * index_vals).sum(dim=-1)      # (B,)
        total_index = soft_index.sum()
        # Cross-field singularity indices are in {-1/4, 0, +1/4}, so the
        # Poincare-Hopf target is chi(S)/4 (not chi(S)).
        target = torch.tensor(float(euler_char) / 4.0, dtype=logits.dtype, device=logits.device)
        return (total_index - target).pow(2)


# ---------------------------------------------------------------------------
# End-to-end losses  (§6.4)
# ---------------------------------------------------------------------------

class JacobianLoss(nn.Module):
    """
    MAIE Jacobian loss (§6.4):

        L_J = (1/M) Σ_i ||log(Ĵ_iᵀ Ĵ_i) − log(M_i*^{−1})||_F²

    where Ĵ_i is the actual Jacobian of the final optimised quad, and
    M_i* is the target metric from the network.

    Args:
        V_final:    (N, 3) final vertex positions after unrolled PD.
        quads_t:    (M, 4) quad indices (long tensor).
        M_targets:  (M, 2, 2) target metric tensors (grad-tracked).
        ref_pinv:   (3, 4) pseudo-inverse of the reference square system.
    """
    def forward(
        self,
        V_final:   torch.Tensor,
        quads_t:   torch.LongTensor,
        M_targets: torch.Tensor,
        ref_pinv:  torch.Tensor,
    ) -> torch.Tensor:
        M_q = quads_t.shape[0]

        # Gather quad vertices and compute Jacobians
        v_q   = V_final[quads_t].double()                  # (M, 4, 3)
        pinv_b = ref_pinv.double().unsqueeze(0).expand(M_q, -1, -1)  # (M, 3, 4)
        X     = torch.bmm(pinv_b, v_q)                     # (M, 3, 3)
        J     = X[:, :2, :].transpose(1, 2)                # (M, 3, 2)
        JtJ   = torch.bmm(J.transpose(1, 2), J)            # (M, 2, 2)

        # log(Ĵᵀ Ĵ)
        log_JtJ = matrix_log_spd2(JtJ.float())             # (M, 2, 2)

        # log(M*^{−1})  — regularise before inversion to avoid singular matrices
        eye     = torch.eye(2, dtype=torch.float32, device=M_targets.device)
        M_reg   = M_targets.float() + 1e-6 * eye
        M_inv   = torch.linalg.inv(M_reg)
        log_Minv = matrix_log_spd2(M_inv)

        diff = log_JtJ - log_Minv
        return diff.pow(2).sum(dim=(-2, -1)).mean()


class AntiFlipLoss(nn.Module):
    """
    Determinant penalty for anti-flip regularisation:

        P_flip = (1/M) Σ_i max(0, ε − det(Ĵ_iᵀ Ĵ_i))²

    Ensures near-zero or negative determinants produce smooth non-NaN gradients
    at the integer-rounding boundary.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        V_final:  torch.Tensor,
        quads_t:  torch.LongTensor,
        ref_pinv: torch.Tensor,
    ) -> torch.Tensor:
        from src.optimization.pd_solver import anti_flip_penalty, _REF_SQUARE_T, _A_REF_PINV_NP
        ref_t = _REF_SQUARE_T.to(V_final.device).to(V_final.dtype)
        return anti_flip_penalty(V_final, quads_t, ref_t, ref_pinv, eps=self.eps)


# ---------------------------------------------------------------------------
# MAIE Alignment Score  (evaluation metric, §9.6)
# ---------------------------------------------------------------------------

def maie_alignment_score(
    V_final:   torch.Tensor,
    quads_t:   torch.LongTensor,
    M_targets: torch.Tensor,
    ref_pinv:  torch.Tensor,
) -> torch.Tensor:
    """
    MAS ∈ (0, 1]:  exp(−||log(Ĵᵀ Ĵ) − log(M*^{−1})||_F) per quad, averaged.
    """
    M_q   = quads_t.shape[0]
    v_q   = V_final[quads_t].double()
    pinv_b = ref_pinv.double().unsqueeze(0).expand(M_q, -1, -1)
    X     = torch.bmm(pinv_b, v_q)
    J     = X[:, :2, :].transpose(1, 2)
    JtJ   = torch.bmm(J.transpose(1, 2), J)
    log_JtJ  = matrix_log_spd2(JtJ.float())
    eye      = torch.eye(2, dtype=torch.float32, device=M_targets.device)
    M_inv    = torch.linalg.inv(M_targets.float() + 1e-6 * eye)
    log_Minv = matrix_log_spd2(M_inv)
    frob = (log_JtJ - log_Minv).pow(2).sum(dim=(-2, -1)).sqrt()
    return torch.exp(-frob).mean()


# ---------------------------------------------------------------------------
# Combined legacy loss  (backward compatible with train.py)
# ---------------------------------------------------------------------------

class CombinedLoss(nn.Module):
    """
    Legacy combined loss: L = L_metric + λ_dir L_dir.
    Returns (total, metric, dir) to match existing train.py unpacking.
    """
    def __init__(self, lambda_dir: float = 0.1, dir_stop_threshold: float = 0.0):
        super().__init__()
        self.lambda_dir = lambda_dir
        self.log_euc    = LogEuclideanLoss()
        self.dir_loss   = DirectionalLoss(isotropy_stop_threshold=dir_stop_threshold)

    def forward(
        self,
        M_pred: torch.Tensor, M_gt: torch.Tensor,
        dir1_pred=None, dir1_gt=None,
        dir2_pred=None, dir2_gt=None,
        aniso_weights: Optional[torch.Tensor] = None,   # (B,) anisotropy weights
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_metric = self.log_euc(M_pred, M_gt)
        loss_dir    = torch.tensor(0.0, device=M_pred.device)
        if dir1_pred is not None and dir1_gt is not None:
            loss_dir = self.dir_loss(dir1_pred, dir1_gt, dir2_pred, dir2_gt,
                                     weights=aniso_weights)
        total = loss_metric + self.lambda_dir * loss_dir
        return total, loss_metric, loss_dir


# ---------------------------------------------------------------------------
# Total end-to-end loss  (§6.4)
# ---------------------------------------------------------------------------

class TotalEndToEndLoss(nn.Module):
    """
    Full training loss:

        L_total = L_metric + λ_dir L_dir
                + λ_sing L_sing + λ_PH L_PH
                + λ_J L_J

    All terms are optional; set λ = 0 to disable.
    """

    def __init__(
        self,
        lambda_dir:  float = 0.1,
        dir_stop_threshold: float = 0.0,
        lambda_sing: float = 0.5,
        lambda_ph:   float = 1.0,
        lambda_j:    float = 0.1,
        lambda_conf: float = 0.1,
        lambda_topo: float = 0.0,
        topo_det_eps: float = 1e-3,
        topo_cond_max: float = 50.0,
        topo_entropy_weight: float = 0.1,
        topo_field_weight: float = 0.0,
        topo_ring_weight: float = 0.0,
        topo_charge_weight: float = 0.0,
        topo_turn_weight: float = 0.0,
        topo_cons_metric_weight: float = 0.0,
        topo_cons_dir_weight: float = 0.0,
        topo_consistency_sigma: float = 0.35,
        topo_ring_min_queries: int = 6,
        topo_warmup_epochs: int = 0,
        topo_warmup_start_epoch: int = 0,
    ):
        super().__init__()
        # Be permissive with YAML scalar types (e.g., quoted scientific notation).
        self.lambda_dir  = float(lambda_dir)
        self.dir_stop_threshold = float(dir_stop_threshold)
        self.lambda_sing = float(lambda_sing)
        self.lambda_ph   = float(lambda_ph)
        self.lambda_j    = float(lambda_j)
        self.lambda_conf = float(lambda_conf)
        self.lambda_topo = float(lambda_topo)

        self.topo_det_eps = float(topo_det_eps)
        self.topo_cond_max = float(topo_cond_max)
        self.topo_entropy_weight = float(topo_entropy_weight)
        self.topo_field_weight = float(topo_field_weight)
        self.topo_ring_weight = float(topo_ring_weight)
        self.topo_charge_weight = float(topo_charge_weight)
        self.topo_turn_weight = float(topo_turn_weight)
        self.topo_cons_metric_weight = float(topo_cons_metric_weight)
        self.topo_cons_dir_weight = float(topo_cons_dir_weight)
        self.topo_consistency_sigma = float(topo_consistency_sigma)
        self.topo_ring_min_queries = int(topo_ring_min_queries)
        self.topo_warmup_epochs = int(topo_warmup_epochs)
        self.topo_warmup_start_epoch = int(topo_warmup_start_epoch)
        self.current_epoch = 0

        self.metric_loss = LogEuclideanLoss()
        self.dir_loss    = DirectionalLoss(isotropy_stop_threshold=self.dir_stop_threshold)
        self.sing_loss   = SingularityLoss()
        self.ph_loss     = PoincareHopfLoss()
        self.jac_loss    = JacobianLoss()
        self.flip_loss   = AntiFlipLoss()
        self.conf_loss   = nn.MSELoss()   # BCELoss has irreducible floor ~0.34 for continuous targets

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def current_topo_multiplier(self) -> float:
        if self.lambda_topo == 0.0:
            return 0.0
        if self.current_epoch < self.topo_warmup_start_epoch:
            return 0.0
        if self.topo_warmup_epochs <= 0:
            return 1.0
        progress = (self.current_epoch - self.topo_warmup_start_epoch + 1) / float(self.topo_warmup_epochs)
        return float(min(1.0, max(0.0, progress)))

    @staticmethod
    def _to_4rosy(d: torch.Tensor) -> torch.Tensor:
        dx, dy = d[..., 0], d[..., 1]
        c2 = dx * dx - dy * dy
        s2 = 2.0 * dx * dy
        c4 = c2 * c2 - s2 * s2
        s4 = 2.0 * c2 * s2
        return torch.stack([c4, s4], dim=-1)

    def _consistency_proxy_loss(
        self,
        dir1_pred: torch.Tensor,
        basis: torch.Tensor,
        anchor_basis: torch.Tensor,
        query_pos: torch.Tensor,
        aniso_weights: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Local field-closure proxy over multiple nearby queries from one region.

        - field term: nearby 4-RoSy orientations should vary smoothly
        - ring term: the net 4θ winding around the local neighborhood should stay
          near zero on ordinary smooth regions (discourages seam/T-junction growth)
        """
        B, Q, _ = dir1_pred.shape
        if Q < 2:
            z = torch.zeros(1, device=dir1_pred.device)[0]
            return z, z, z, z

        Rq = basis[..., :2]                # (B,Q,3,2)
        Ra = anchor_basis[..., :2]         # (B,3,2)
        d_world = torch.einsum('bqij,bqj->bqi', Rq, dir1_pred)
        d_anchor = torch.einsum('bij,bqj->bqi', Ra.transpose(-2, -1), d_world)
        d_anchor = F.normalize(d_anchor, p=2, dim=-1)
        u = self._to_4rosy(d_anchor)       # (B,Q,2)

        pos2 = query_pos[..., :2]
        dist = torch.cdist(pos2, pos2, p=2)
        sigma = self.topo_consistency_sigma * dist.detach().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-4)
        weights = torch.exp(-(dist ** 2) / (2.0 * sigma ** 2))
        eye = torch.eye(Q, device=dir1_pred.device, dtype=weights.dtype).unsqueeze(0)
        weights = weights * (1.0 - eye)
        align = torch.einsum('bik,bjk->bij', u, u).clamp(min=-1.0, max=1.0)
        pair_loss = 0.5 * (1.0 - align)
        if aniso_weights is not None:
            aw = aniso_weights.clamp(min=0.0)
            pair_w = weights * (aw.unsqueeze(-1) * aw.unsqueeze(-2))
        else:
            pair_w = weights
        field = (pair_w * pair_loss).sum() / pair_w.sum().clamp(min=1e-8)

        if Q < max(3, self.topo_ring_min_queries):
            ring = torch.zeros(1, device=dir1_pred.device)[0]
            charge = torch.zeros(1, device=dir1_pred.device)[0]
            turn = torch.zeros(1, device=dir1_pred.device)[0]
            return field, ring, charge, turn

        angles_pos = torch.atan2(pos2[..., 1], pos2[..., 0])      # (B,Q)
        order = torch.argsort(angles_pos, dim=-1)
        phase = torch.atan2(u[..., 1], u[..., 0])                 # 4θ
        phase_ord = torch.gather(phase, 1, order)
        w_ord = None if aniso_weights is None else torch.gather(aniso_weights, 1, order)
        phase_next = torch.roll(phase_ord, shifts=-1, dims=1)
        delta = torch.atan2(torch.sin(phase_next - phase_ord), torch.cos(phase_next - phase_ord))
        total = delta.sum(dim=1)
        ring_loss = 1.0 - torch.cos(total)
        charge_loss = total.abs() / np.pi
        turn_loss = 1.0 - torch.cos(delta)
        if w_ord is not None:
            ring_w = w_ord.mean(dim=1).clamp(min=0.0)
            ring = (ring_w * ring_loss).sum() / ring_w.sum().clamp(min=1e-8)
            charge = (ring_w * charge_loss).sum() / ring_w.sum().clamp(min=1e-8)
            turn_w = 0.5 * (w_ord + torch.roll(w_ord, shifts=-1, dims=1))
            turn = (turn_w * turn_loss).sum() / turn_w.sum().clamp(min=1e-8)
        else:
            ring = ring_loss.mean()
            charge = charge_loss.mean()
            turn = turn_loss.mean()
        return field, ring, charge, turn

    def _topology_proxy_loss(
        self,
        M_pred: torch.Tensor,
        sing_logits: Optional[torch.Tensor],
        aniso_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Differentiable topology proxy used during training.

        Components:
        1) determinant barrier: prevent near-singular metrics;
        2) condition-number barrier: avoid extreme anisotropy blow-ups;
        3) singularity-entropy penalty (weighted by anisotropy): discourage
           uncertain singularity logits where orientation structure is strong.
        """
        eigvals, _ = eigh2x2(M_pred)
        lam_min = eigvals[..., 0].clamp(min=1e-8)
        lam_max = eigvals[..., 1].clamp(min=1e-8)

        det = lam_min * lam_max
        cond = lam_max / lam_min

        det_barrier = torch.relu(self.topo_det_eps - det).pow(2).mean()
        cond_barrier = torch.relu(cond - self.topo_cond_max).pow(2).mean()

        ent_term = torch.zeros(1, device=M_pred.device)[0]
        if sing_logits is not None:
            probs = F.softmax(sing_logits, dim=-1)
            ent = -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=-1)
            ent = ent / np.log(3.0)  # normalize to [0,1]
            if aniso_weights is not None:
                w = aniso_weights.detach().clamp(min=0.0)
                ent_term = (w * ent).sum() / w.sum().clamp(min=1e-8)
            else:
                ent_term = ent.mean()

        return det_barrier + cond_barrier + self.topo_entropy_weight * ent_term

    def forward(
        self,
        # Metric head
        M_pred:      torch.Tensor,          # (B, 2, 2)
        M_gt:        torch.Tensor,          # (B, 2, 2)
        # Singularity head
        sing_logits: Optional[torch.Tensor] = None,  # (B, 3)
        sing_labels: Optional[torch.Tensor] = None,  # (B,)  long
        euler_char:  float = 2.0,
        # Directional
        dir1_pred:   Optional[torch.Tensor] = None,
        dir1_gt:     Optional[torch.Tensor] = None,
        dir2_pred:   Optional[torch.Tensor] = None,
        dir2_gt:     Optional[torch.Tensor] = None,
        aniso_weights: Optional[torch.Tensor] = None,  # (B,) weights for dir loss
        confidence_pred: Optional[torch.Tensor] = None,    # (B,)
        confidence_gt: Optional[torch.Tensor] = None,      # (B,)
        # End-to-end Jacobian
        V_final:     Optional[torch.Tensor] = None,  # (N, 3)
        quads_t:     Optional[torch.LongTensor] = None,
        M_targets:   Optional[torch.Tensor] = None,  # (M, 2, 2)
        ref_pinv:    Optional[torch.Tensor] = None,  # (3, 4)
        consistency_dir1_pred: Optional[torch.Tensor] = None,  # (B,Q,2)
        consistency_dir2_pred: Optional[torch.Tensor] = None,  # (B,Q,2)
        consistency_basis: Optional[torch.Tensor] = None,      # (B,Q,3,3)
        consistency_anchor_basis: Optional[torch.Tensor] = None,  # (B,3,3)
        consistency_query_pos: Optional[torch.Tensor] = None,  # (B,Q,3)
        consistency_aniso_weights: Optional[torch.Tensor] = None,  # (B,Q)
        consistency_M_pred: Optional[torch.Tensor] = None,     # (B,Q,2,2)
        consistency_M_gt: Optional[torch.Tensor] = None,       # (B,Q,2,2)
        consistency_dir1_gt: Optional[torch.Tensor] = None,    # (B,Q,2)
        consistency_dir2_gt: Optional[torch.Tensor] = None,    # (B,Q,2)
    ) -> Tuple[torch.Tensor, dict]:
        terms: dict = {}

        # ── Metric ──────────────────────────────────────────────────────
        terms['metric'] = self.metric_loss(M_pred, M_gt)

        # ── Directional ─────────────────────────────────────────────────
        if dir1_pred is not None and dir1_gt is not None:
            terms['dir'] = self.dir_loss(dir1_pred, dir1_gt, dir2_pred, dir2_gt,
                                          weights=aniso_weights)
        else:
            terms['dir'] = torch.zeros(1, device=M_pred.device)[0]

        # ── Confidence ──────────────────────────────────────────────────
        if confidence_pred is not None and confidence_gt is not None:
            cp = confidence_pred.clamp(0.0, 1.0)
            cg = confidence_gt.clamp(0.0, 1.0)
            terms['conf'] = self.conf_loss(cp, cg)
        else:
            terms['conf'] = torch.zeros(1, device=M_pred.device)[0]

        # ── Singularity ─────────────────────────────────────────────────
        if sing_logits is not None:
            # Cross-entropy needs labels; if labels are unavailable we still keep
            # the soft PH topological regulariser active during training.
            if sing_labels is not None:
                terms['sing'] = self.sing_loss(sing_logits, sing_labels)
            else:
                terms['sing'] = torch.zeros(1, device=M_pred.device)[0]
            terms['ph'] = self.ph_loss(sing_logits, euler_char)
        else:
            terms['sing'] = torch.zeros(1, device=M_pred.device)[0]
            terms['ph'] = torch.zeros(1, device=M_pred.device)[0]

        # ── End-to-end Jacobian ─────────────────────────────────────────
        if (V_final is not None and quads_t is not None
                and M_targets is not None and ref_pinv is not None):
            terms['jac']  = self.jac_loss(V_final, quads_t, M_targets, ref_pinv)
            terms['flip'] = self.flip_loss(V_final, quads_t, ref_pinv)
        else:
            terms['jac']  = torch.zeros(1, device=M_pred.device)[0]
            terms['flip'] = torch.zeros(1, device=M_pred.device)[0]

        # ── Topology proxy (train-time differentiable surrogate) ────────
        topo_mult = self.current_topo_multiplier()

        if topo_mult > 0.0:
            terms['topo'] = self._topology_proxy_loss(
                M_pred=M_pred,
                sing_logits=sing_logits,
                aniso_weights=aniso_weights,
            )
        else:
            terms['topo'] = torch.zeros(1, device=M_pred.device)[0]

        terms['topo_field'] = torch.zeros(1, device=M_pred.device)[0]
        terms['topo_ring'] = torch.zeros(1, device=M_pred.device)[0]
        terms['topo_charge'] = torch.zeros(1, device=M_pred.device)[0]
        terms['topo_turn'] = torch.zeros(1, device=M_pred.device)[0]
        terms['topo_cons_metric'] = torch.zeros(1, device=M_pred.device)[0]
        terms['topo_cons_dir'] = torch.zeros(1, device=M_pred.device)[0]
        if (
            topo_mult > 0.0
            and consistency_dir1_pred is not None
            and consistency_basis is not None
            and consistency_anchor_basis is not None
            and consistency_query_pos is not None
            and (self.topo_field_weight != 0.0 or self.topo_ring_weight != 0.0)
        ):
            field_term, ring_term, charge_term, turn_term = self._consistency_proxy_loss(
                consistency_dir1_pred,
                consistency_basis,
                consistency_anchor_basis,
                consistency_query_pos,
                consistency_aniso_weights,
            )
            terms['topo_field'] = field_term
            terms['topo_ring'] = ring_term
            terms['topo_charge'] = charge_term
            terms['topo_turn'] = turn_term
            if consistency_M_pred is not None:
                flat_M = consistency_M_pred.reshape(-1, 2, 2)
                terms['topo'] = terms['topo'] + 0.5 * self._topology_proxy_loss(
                    M_pred=flat_M,
                    sing_logits=None,
                    aniso_weights=None,
                )
        if (
            topo_mult > 0.0
            and consistency_M_pred is not None
            and consistency_M_gt is not None
            and self.topo_cons_metric_weight != 0.0
        ):
            terms['topo_cons_metric'] = self.metric_loss(
                consistency_M_pred.reshape(-1, 2, 2),
                consistency_M_gt.reshape(-1, 2, 2),
            )
        if (
            topo_mult > 0.0
            and consistency_dir1_pred is not None
            and consistency_dir1_gt is not None
            and self.topo_cons_dir_weight != 0.0
        ):
            flat_w = None
            if consistency_aniso_weights is not None:
                flat_w = consistency_aniso_weights.reshape(-1)
            terms['topo_cons_dir'] = self.dir_loss(
                consistency_dir1_pred.reshape(-1, consistency_dir1_pred.shape[-1]),
                consistency_dir1_gt.reshape(-1, consistency_dir1_gt.shape[-1]),
                None if consistency_dir2_pred is None else consistency_dir2_pred.reshape(-1, consistency_dir2_pred.shape[-1]),
                None if consistency_dir2_gt is None else consistency_dir2_gt.reshape(-1, consistency_dir2_gt.shape[-1]),
                weights=flat_w,
            )

        total = (
            terms['metric']
            + self.lambda_dir  * terms['dir']
            + self.lambda_sing * terms['sing']
            + self.lambda_ph   * terms['ph']
            + self.lambda_j    * (terms['jac'] + terms['flip'])
            + self.lambda_conf * terms['conf']
            + (self.lambda_topo * topo_mult) * (
                terms['topo']
                + self.topo_field_weight * terms['topo_field']
                + self.topo_ring_weight * terms['topo_ring']
                + self.topo_charge_weight * terms['topo_charge']
                + self.topo_turn_weight * terms['topo_turn']
                + self.topo_cons_metric_weight * terms['topo_cons_metric']
                + self.topo_cons_dir_weight * terms['topo_cons_dir']
            )
        )
        return total, terms
