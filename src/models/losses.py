"""
Loss functions for NPAQ training.

Active terms:
    LogEuclideanLoss    — ||log M_pred − log M_gt||_F²
    AnisotropyLoss      — eigenvalue-ratio log-distance (corrects LogEuclidean trace bias)
    JacobianLoss        — optional unrolled-PD term, requires explicit quad inputs
    AntiFlipLoss        — determinant barrier (prevents inverted quads)
    Topology proxy      — det/cond barriers (metric-degeneracy guard)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from src.geometry.metric_utils import eigh2x2


def matrix_log_spd2(M: torch.Tensor) -> torch.Tensor:
    """log(M) for a batch of 2×2 SPD matrices via eigendecomposition."""
    eigvals, eigvecs = eigh2x2(M)
    log_eig = torch.log(eigvals.clamp(min=1e-3))
    return eigvecs @ torch.diag_embed(log_eig) @ eigvecs.transpose(-2, -1)


class LogEuclideanLoss(nn.Module):
    """L_metric = (1/B) Σ ||log M_pred − log M_gt||_F²"""
    def forward(self, M_pred: torch.Tensor, M_gt: torch.Tensor) -> torch.Tensor:
        return (matrix_log_spd2(M_pred) - matrix_log_spd2(M_gt)).pow(2).sum(dim=(-2, -1)).mean()


class AnisotropyLoss(nn.Module):
    """
    Penalise mismatch in eigenvalue ratio λ_max/λ_min in log space.
    LogEuclidean's Frobenius norm under-weights this on weakly anisotropic samples.
    """
    def forward(self, M_pred: torch.Tensor, M_gt: torch.Tensor) -> torch.Tensor:
        eigvals_p, _ = eigh2x2(M_pred)
        eigvals_g, _ = eigh2x2(M_gt)
        ratio_p = (eigvals_p[..., 1] / eigvals_p[..., 0].clamp(min=1e-4)).clamp(min=1e-4)
        ratio_g = (eigvals_g[..., 1] / eigvals_g[..., 0].clamp(min=1e-4)).clamp(min=1e-4)
        return (torch.log(ratio_p) - torch.log(ratio_g)).pow(2).mean()


class JacobianLoss(nn.Module):
    """
    MAIE Jacobian loss (§6.4):
        L_J = (1/M) Σ ||log(Ĵᵀ Ĵ) − log(M*^{−1})||_F²

    Enable only after the metric head has converged (val_metric < ~0.05).
    """
    def forward(
        self,
        V_final:   torch.Tensor,
        quads_t:   torch.LongTensor,
        M_targets: torch.Tensor,
        ref_pinv:  torch.Tensor,
    ) -> torch.Tensor:
        M_q = quads_t.shape[0]
        v_q   = V_final[quads_t].double()
        pinv_b = ref_pinv.double().unsqueeze(0).expand(M_q, -1, -1)
        X     = torch.bmm(pinv_b, v_q)
        J     = X[:, :2, :].transpose(1, 2)
        JtJ   = torch.bmm(J.transpose(1, 2), J)

        log_JtJ = matrix_log_spd2(JtJ.float())
        eye     = torch.eye(2, dtype=torch.float32, device=M_targets.device)
        M_inv   = torch.linalg.inv(M_targets.float() + 1e-6 * eye)
        log_Minv = matrix_log_spd2(M_inv)

        return (log_JtJ - log_Minv).pow(2).sum(dim=(-2, -1)).mean()


class AntiFlipLoss(nn.Module):
    """Gram-determinant barrier preventing rank-degenerate quads."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        V_final:  torch.Tensor,
        quads_t:  torch.LongTensor,
        ref_pinv: torch.Tensor,
    ) -> torch.Tensor:
        from src.optimization.pd_solver import anti_flip_penalty, _REF_SQUARE_T
        ref_t = _REF_SQUARE_T.to(V_final.device).to(V_final.dtype)
        return anti_flip_penalty(V_final, quads_t, ref_t, ref_pinv, eps=self.eps)


def maie_alignment_score(
    V_final:   torch.Tensor,
    quads_t:   torch.LongTensor,
    M_targets: torch.Tensor,
    ref_pinv:  torch.Tensor,
) -> torch.Tensor:
    """MAS ∈ (0, 1]:  exp(−||log(Ĵᵀ Ĵ) − log(M*^{−1})||_F) per quad, averaged."""
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


class TotalLoss(nn.Module):
    """
    L_total = L_metric
            + λ_aniso L_aniso
            + λ_conf  L_conf
            + λ_topo  (det_barrier + cond_barrier)
            + λ_j     (L_J + L_flip)

    All terms with λ=0 are skipped.  If λ_j>0, callers must provide the
    unrolled-PD tensors; otherwise we fail fast instead of silently dropping
    the end-to-end term.
    """

    def __init__(
        self,
        lambda_aniso: float = 0.1,
        lambda_conf:  float = 0.1,
        lambda_topo:  float = 0.01,
        lambda_j:     float = 0.0,
        topo_det_eps: float = 1e-3,
        topo_cond_max: float = 50.0,
    ):
        super().__init__()
        self.lambda_aniso = float(lambda_aniso)
        self.lambda_conf  = float(lambda_conf)
        self.lambda_topo  = float(lambda_topo)
        self.lambda_j     = float(lambda_j)
        self.topo_det_eps = float(topo_det_eps)
        self.topo_cond_max = float(topo_cond_max)

        self.metric_loss = LogEuclideanLoss()
        self.aniso_loss  = AnisotropyLoss()
        self.jac_loss    = JacobianLoss()
        self.flip_loss   = AntiFlipLoss()
        self.conf_loss   = nn.MSELoss()

    def _topology_proxy(self, M_pred: torch.Tensor) -> torch.Tensor:
        """Det + condition-number barriers — keep metric non-degenerate."""
        eigvals, _ = eigh2x2(M_pred)
        lam_min = eigvals[..., 0].clamp(min=1e-8)
        lam_max = eigvals[..., 1].clamp(min=1e-8)
        det_barrier  = torch.relu(self.topo_det_eps - lam_min * lam_max).pow(2).mean()
        cond_barrier = torch.relu(lam_max / lam_min - self.topo_cond_max).pow(2).mean()
        return det_barrier + cond_barrier

    def forward(
        self,
        M_pred:      torch.Tensor,
        M_gt:        torch.Tensor,
        confidence_pred: Optional[torch.Tensor] = None,
        confidence_gt:   Optional[torch.Tensor] = None,
        V_final:     Optional[torch.Tensor] = None,
        quads_t:     Optional[torch.LongTensor] = None,
        M_targets:   Optional[torch.Tensor] = None,
        ref_pinv:    Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        terms: dict = {}
        zero = torch.zeros((), device=M_pred.device)

        terms['metric'] = self.metric_loss(M_pred, M_gt)
        terms['aniso']  = self.aniso_loss(M_pred, M_gt) if self.lambda_aniso > 0.0 else zero

        if confidence_pred is not None and confidence_gt is not None:
            terms['conf'] = self.conf_loss(confidence_pred.clamp(0, 1), confidence_gt.clamp(0, 1))
        else:
            terms['conf'] = zero

        terms['topo'] = self._topology_proxy(M_pred) if self.lambda_topo > 0.0 else zero

        terms['jac'] = zero
        terms['flip'] = zero
        if self.lambda_j > 0.0:
            missing = [
                name for name, value in (
                    ('V_final', V_final),
                    ('quads_t', quads_t),
                    ('M_targets', M_targets),
                    ('ref_pinv', ref_pinv),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    "lambda_j > 0 requires unrolled-PD tensors: "
                    + ", ".join(missing)
                )
            terms['jac'] = self.jac_loss(V_final, quads_t, M_targets, ref_pinv)
            terms['flip'] = self.flip_loss(V_final, quads_t, ref_pinv)

        total = (
            terms['metric']
            + self.lambda_aniso * terms['aniso']
            + self.lambda_conf  * terms['conf']
            + self.lambda_topo  * terms['topo']
            + self.lambda_j     * (terms['jac'] + terms['flip'])
        )
        return total, terms
