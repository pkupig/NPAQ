"""
Utility functions for metric tensor manipulation.
Implements conversions between parameterization and 2x2 SPD matrices,
as well as matrix logarithms, exponentials, and decompositions.
"""

import torch
import numpy as np
from typing import Tuple, Union


def params_to_tensor(
    s1: Union[float, np.ndarray, torch.Tensor],
    s2: Union[float, np.ndarray, torch.Tensor],
    c: Union[float, np.ndarray, torch.Tensor],
    s: Union[float, np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert shape parameters (s1, s2, c, s) to a 2x2 metric tensor M.
    M = R(θ) diag(s1^2, s2^2) R(θ)^T, where (c,s) = (cos 2θ, sin 2θ).

    Args:
        s1, s2: Positive scalars representing sqrt of eigenvalues (shape factors).
        c, s: Cos(2θ) and sin(2θ) representing principal direction.

    Returns:
        M: (..., 2, 2) symmetric positive definite matrix.
    """
    # Determine input type and device
    if isinstance(s1, torch.Tensor):
        # Use atan2 rather than sign-based half-angle recovery.  The sqrt/sign
        # formula is ambiguous at (c, s)=(-1, 0) and can collapse a valid 90 deg
        # direction into a near-zero rotation.
        theta = 0.5 * torch.atan2(s, c)
        cosθ = torch.cos(theta)
        sinθ = torch.sin(theta)
        R = torch.stack([torch.stack([cosθ, -sinθ], dim=-1),
                         torch.stack([sinθ,  cosθ], dim=-1)], dim=-2)  # (..., 2, 2)
        # DGCNN's softplus parameterisation guarantees s1 ≥ s2; no sort needed.
        ev = torch.stack([s1**2, s2**2], dim=-1).clamp(1e-4, 1e4)
        diag = torch.diag_embed(ev)
        M = R @ diag @ R.transpose(-2, -1)
        return M
    else:
        theta = 0.5 * np.arctan2(s, c)
        cosθ = np.cos(theta)
        sinθ = np.sin(theta)
        R = np.stack([
            np.stack([cosθ, -sinθ], axis=-1),
            np.stack([sinθ,  cosθ], axis=-1),
        ], axis=-2)
        ev = np.clip(
            np.stack([np.asarray(s1) ** 2, np.asarray(s2) ** 2], axis=-1),
            1e-4,
            1e4,
        )
        diag = np.zeros(np.shape(ev)[:-1] + (2, 2), dtype=np.asarray(ev).dtype)
        diag[..., 0, 0] = ev[..., 0]
        diag[..., 1, 1] = ev[..., 1]
        M = R @ diag @ np.swapaxes(R, -1, -2)
        return M


def eigh2x2(M: torch.Tensor):
    """
    Analytical eigendecomposition of a batch of 2×2 symmetric matrices.

    Replaces torch.linalg.eigh for 2×2 matrices because LAPACK's routine
    fails (error code 1) when eigenvalues are nearly equal — which happens
    frequently for isotropic metric predictions early in training.

    The closed-form formula is always stable:
        disc = sqrt( ((a-c)/2)^2 + b^2 )   ≥ 0  (regularised with +1e-10)
        λ₁ = (a+c)/2 - disc   (smaller)
        λ₂ = (a+c)/2 + disc   (larger)

    Eigenvector of λ₂ is derived from row 2 of (M - λ₂I)v = 0
    using whichever of two equivalent expressions is numerically larger
    (avoids division by near-zero).

    Args:
        M:  (..., 2, 2)  batch of symmetric matrices.

    Returns:
        eigvals:   (..., 2)     eigenvalues in ascending order.
        eigvecs:   (..., 2, 2)  columns are eigenvectors (eigvecs[..., :, i]
                                corresponds to eigvals[..., i]).
    """
    a    = M[..., 0, 0]
    b    = M[..., 0, 1]
    c    = M[..., 1, 1]
    mid  = (a + c) * 0.5
    diff = (a - c) * 0.5                          # (a-c)/2
    disc = torch.sqrt(diff**2 + b**2 + 1e-10)     # always ≥ 0

    lam1 = mid - disc   # smaller eigenvalue
    lam2 = mid + disc   # larger eigenvalue

    # Eigenvector of lam2.
    # From (M - λ₂I)v = 0, two equivalent expressions for (vx, vy):
    #   A:  (vx, vy) ∝ (b,          disc - diff)   stable when diff < 0
    #   B:  (vx, vy) ∝ (disc + diff, b           )   stable when diff ≥ 0
    # Choose B when diff ≥ 0, A otherwise.
    use_B = (diff >= 0).to(M.dtype)
    vx2 = use_B * (disc + diff) + (1.0 - use_B) * b
    vy2 = use_B * b              + (1.0 - use_B) * (disc - diff)
    v2n  = torch.sqrt(vx2**2 + vy2**2 + 1e-10)
    vx2  = vx2 / v2n
    vy2  = vy2 / v2n

    # Eigenvector of lam1 is orthogonal to that of lam2.
    vx1, vy1 = -vy2, vx2

    eigvecs = torch.stack([
        torch.stack([vx1, vy1], dim=-1),   # col 0 → smaller eigenvalue
        torch.stack([vx2, vy2], dim=-1),   # col 1 → larger eigenvalue
    ], dim=-1)   # (..., 2, 2)

    return torch.stack([lam1, lam2], dim=-1), eigvecs


def tensor_to_params(
    M: Union[np.ndarray, torch.Tensor]
) -> Tuple[Union[np.ndarray, torch.Tensor], ...]:
    """
    Inverse of params_to_tensor: extract (s1, s2, c, s) from a metric tensor.
    s1, s2 are sqrt of eigenvalues. (c,s) = (cos 2θ, sin 2θ) where θ is the angle
    of the first eigenvector.

    Args:
        M: (..., 2, 2) symmetric positive definite matrix.

    Returns:
        s1, s2, c, s: each with shape (...).
    """
    is_torch = isinstance(M, torch.Tensor)
    if is_torch:
        # Eigen decomposition
        eigvals, eigvecs = eigh2x2(M)  # (...,2), (...,2,2)  (ascending order)
        s1 = torch.sqrt(eigvals[..., 1])  # larger eigenvalue
        s2 = torch.sqrt(eigvals[..., 0])  # smaller eigenvalue
        v1 = eigvecs[..., 1]  # eigenvector for larger eigenvalue
        # Compute angle θ from v1: v1 = (cosθ, sinθ) (since it's a unit vector)
        cosθ = v1[..., 0]
        sinθ = v1[..., 1]
        # Compute double-angle
        c = cosθ**2 - sinθ**2  # cos2θ
        s = 2 * cosθ * sinθ     # sin2θ
    else:
        eigvals, eigvecs = np.linalg.eigh(M)
        s1 = np.sqrt(eigvals[..., 1])
        s2 = np.sqrt(eigvals[..., 0])
        v1 = eigvecs[..., 1]
        cosθ = v1[..., 0]
        sinθ = v1[..., 1]
        c = cosθ**2 - sinθ**2
        s = 2 * cosθ * sinθ
    return s1, s2, c, s


def logm_spd(M: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute matrix logarithm of a symmetric positive definite matrix.
    Uses eigenvalue decomposition: log(M) = V diag(log λ) V^T.
    """
    is_torch = isinstance(M, torch.Tensor)
    if is_torch:
        eigvals, eigvecs = eigh2x2(M)
        log_eigvals = torch.log(eigvals.clamp(min=1e-8))
        return eigvecs @ torch.diag_embed(log_eigvals) @ eigvecs.transpose(-2, -1)
    else:
        eigvals, eigvecs = np.linalg.eigh(M)
        log_eigvals = np.log(np.maximum(eigvals, 1e-8))
        return eigvecs @ np.diag(log_eigvals) @ eigvecs.T


def expm_spd(L: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute matrix exponential of a symmetric matrix (result will be SPD).
    Uses eigenvalue decomposition: exp(L) = V diag(exp λ) V^T.
    """
    is_torch = isinstance(L, torch.Tensor)
    if is_torch:
        eigvals, eigvecs = eigh2x2(L)
        exp_eigvals = torch.exp(eigvals)
        return eigvecs @ torch.diag_embed(exp_eigvals) @ eigvecs.transpose(-2, -1)
    else:
        eigvals, eigvecs = np.linalg.eigh(L)
        exp_eigvals = np.exp(eigvals)
        return eigvecs @ np.diag(exp_eigvals) @ eigvecs.T


def interpolate_metric(
    M1: Union[np.ndarray, torch.Tensor],
    M2: Union[np.ndarray, torch.Tensor],
    t: float
) -> Union[np.ndarray, torch.Tensor]:
    """
    Interpolate between two metric tensors using Log-Euclidean interpolation.
    M(t) = exp((1-t) log M1 + t log M2)
    """
    logM1 = logm_spd(M1)
    logM2 = logm_spd(M2)
    if isinstance(M1, torch.Tensor):
        logM_t = (1 - t) * logM1 + t * logM2
    else:
        logM_t = (1 - t) * logM1 + t * logM2
    return expm_spd(logM_t)


def metric_to_ellipse(M: Union[np.ndarray, torch.Tensor]):
    """
    Convert metric tensor to ellipse parameters: axes lengths and orientation.
    Returns: (a, b, angle) where a >= b are the semi-axis lengths, angle in radians.
    """
    s1, s2, c, s = tensor_to_params(M)
    # tensor_to_params returns s1=sqrt(λ_max(M))=1/σ_min(J), s2=sqrt(λ_min(M))=1/σ_max(J).
    # MAIE semi-axes equal the singular values of J: σ_max=1/s2 (longer), σ_min=1/s1 (shorter).
    a = 1.0 / s2  # larger semi-axis
    b = 1.0 / s1  # smaller semi-axis
    angle = 0.5 * np.arctan2(s, c)
    return a, b, angle
