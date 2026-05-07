"""
Unrolled Differentiable Projective Dynamics Solver  (§6 of PAPER.md).

Key design decisions
────────────────────
• The solver is a **PyTorch nn.Module** whose `forward()` method runs K PD
  iterations in a differentiable computational graph.
• **Local step** uses `torch.linalg.svd` (automatic gradient support, §6.2).
• **Global step** solves  A V = P  where A = I + γ Lᵀ L is constant.
  Gradient flows through the solve via **implicit differentiation** (§6.3):
      ∂L/∂P = A^{-1} ∂L/∂V    (A is symmetric, so A^{-T} = A^{-1}).
  The forward pass uses a pre-factorised scipy sparse Cholesky; the backward
  pass calls the same factorised solver.
• **Degeneracy regularisation**: a determinant-of-Gram penalty keeps Jacobians
  away from rank collapse. Signed flips are handled in the inference solver by
  the signed-area regularizer and backtracking (§5.3 / §6.4).
• **Truncated unrolling**: iterations 0…K−K_grad run without gradient;
  only the last K_grad iterations contribute to the backward pass (§6.5).

Usage in training
─────────────────
    solver = UnrolledPDSolver(quads, n_vertices, ...)
    V_final = solver(V_init, M_targets)   # V_init from parametrisation
    loss = jacobian_loss(V_final, M_targets) + anti_flip_penalty(V_final, quads)
    loss.backward()   # gradients reach DGCNN through M_targets

Usage in inference (reconstruct.py)
────────────────────────────────────
    V_opt = solver.optimize_numpy(V_init_np, M_targets_np, n_iter=30)
"""

from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized
from typing import List, Optional, Callable, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Implicit differentiation: differentiable sparse linear solve
# ---------------------------------------------------------------------------

class _SparseCholeskyLinearSolve(torch.autograd.Function):
    """
    Forward:  V = A^{-1} P   (using pre-factorised Cholesky)
    Backward: ∂L/∂P = A^{-1} ∂L/∂V    (A symmetric ⟹ A^{-T} = A^{-1})

    P:  (N, D) right-hand side — tracked tensor.
    solve_fn: callable returned by scipy.sparse.linalg.factorized(A).
    """

    @staticmethod
    def forward(ctx, P: torch.Tensor, solve_fn) -> torch.Tensor:
        P_np = P.detach().cpu().double().numpy()
        N, D = P_np.shape
        V_np = np.empty_like(P_np)
        for d in range(D):
            V_np[:, d] = solve_fn(P_np[:, d])
        ctx.solve_fn = solve_fn
        return torch.tensor(V_np, dtype=P.dtype, device=P.device)

    @staticmethod
    def backward(ctx, grad_V: torch.Tensor):
        solve_fn = ctx.solve_fn
        gV_np = grad_V.detach().cpu().double().numpy()
        N, D = gV_np.shape
        gP_np = np.empty_like(gV_np)
        for d in range(D):
            gP_np[:, d] = solve_fn(gV_np[:, d])
        grad_P = torch.tensor(gP_np, dtype=grad_V.dtype, device=grad_V.device)
        return grad_P, None   # None for solve_fn (not a tensor)


# ---------------------------------------------------------------------------
# Helper: build system matrix A = I + γ Lᵀ L from quad connectivity
# ---------------------------------------------------------------------------

def _build_quad_laplacian(n_vertices: int, quads: np.ndarray) -> sparse.csc_matrix:
    """
    Graph Laplacian L of the quad mesh (edge-based, uniform weights).

    Returns L as a csc_matrix.
    """
    edges: set = set()
    for q in quads:
        for k in range(4):
            a, b = int(q[k]), int(q[(k + 1) % 4])
            if a != b:
                edges.add((min(a, b), max(a, b)))

    row, col, data = [], [], []
    deg = np.zeros(n_vertices)
    for (a, b) in edges:
        row += [a, b]
        col += [b, a]
        data += [-1.0, -1.0]
        deg[a] += 1.0
        deg[b] += 1.0

    L = sparse.csr_matrix(
        (data + deg.tolist(), (row + list(range(n_vertices)), col + list(range(n_vertices)))),
        shape=(n_vertices, n_vertices),
    )
    return L.tocsc()


def _build_global_system(
    n_vertices: int,
    quads: np.ndarray,
    smoothness_weight: float,
) -> Tuple[sparse.csc_matrix, object]:
    """
    Build and factorize A = I + γ Lᵀ L.

    Returns:
        A:          (N, N) csc_matrix.
        A_factor:   callable — the Cholesky factorised solver.
    """
    L = _build_quad_laplacian(n_vertices, quads)
    A = sparse.eye(n_vertices, format='csc') + smoothness_weight * (L.T @ L)
    A_factor = factorized(A.tocsc())
    return A, A_factor


# ---------------------------------------------------------------------------
# Reference square corners (shared across all quads)
# ---------------------------------------------------------------------------

_REF_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
_REF_SQUARE_T = torch.tensor(_REF_SQUARE, dtype=torch.float64)   # (4, 2)
# Extended for lstsq: [ref | 1]  (4, 3)
_A_REF_NP = np.hstack([_REF_SQUARE, np.ones((4, 1))])   # (4, 3)
# Pseudo-inverse of A_ref: (3, 4)
_A_REF_PINV_NP = np.linalg.pinv(_A_REF_NP)


# ---------------------------------------------------------------------------
# Vectorised local step (PyTorch)
# ---------------------------------------------------------------------------

def _local_step_vectorised(
    V: torch.Tensor,
    quads_t: torch.LongTensor,
    S: torch.Tensor,
    ref_t: torch.Tensor,
    A_ref_pinv_t: torch.Tensor,
) -> torch.Tensor:
    """
    Vectorised differentiable local step for all quads.

    For each quad q_i:
        1.  J_i  = A_ref⁺ · v_q → (3, 2)          (current Jacobian)
        2.  K_i  = J_i · S_i^T  → (3, 2)
        3.  SVD(K_i) → U, Σ, Vᵀ
        4.  R*   = U · Vᵀ       → (3, 2)           (optimal rotation)
        5.  scale = ‖J_i‖_F / ‖S_i‖_F
        6.  J*   = R* · (scale · S_i)               (projected Jacobian)
        7.  t*   = mean(v_q − ref · J*^T)           (optimal translation)
        8.  v_proj = ref · J*^T + t*                (projected vertices)

    Args:
        V:              (N, 3) vertex positions.
        quads_t:        (M, 4) long — quad indices.
        S:              (M, 2, 2) shape matrices  S = M^{-1/2}.
        ref_t:          (4, 2)   reference square corners.
        A_ref_pinv_t:   (3, 4)   pseudo-inverse of [ref | 1].

    Returns:
        proj_avg:  (N, 3) averaged projected positions.
    """
    M_q = quads_t.shape[0]
    N   = V.shape[0]
    dtype = V.dtype
    device = V.device

    # v_q[qi, lj, :] = position of local vertex lj of quad qi
    v_q = V[quads_t]       # (M, 4, 3)

    # J_i = A_ref_pinv (3,4) @ v_q_i (4,3) → (M, 3, 2) after slicing first 2 rows
    # Batched: (M, 3, 4) @ (M, 4, 3) via einsum
    # A_ref_pinv_t: (3, 4) → broadcast to (M, 3, 4)
    A_pinv_b = A_ref_pinv_t.unsqueeze(0).expand(M_q, -1, -1)  # (M, 3, 4)
    X = torch.bmm(A_pinv_b, v_q)                               # (M, 3, 3)
    J = X[:, :2, :].transpose(1, 2)                            # (M, 3, 2)

    # K_i = J_i @ S_i^T
    K = torch.bmm(J, S.transpose(1, 2))                        # (M, 3, 2)

    # SVD of K: U (M,3,2), sigma (M,2), Vh (M,2,2)
    U, sigma, Vh = torch.linalg.svd(K, full_matrices=False)    # (M,3,2),(M,2),(M,2,2)
    R_opt = torch.bmm(U, Vh)                                   # (M, 3, 2)

    # Scale correction
    j_frob = torch.linalg.matrix_norm(J, ord='fro')            # (M,)
    s_frob = torch.linalg.matrix_norm(S, ord='fro')            # (M,)
    scale  = j_frob / (s_frob + 1e-10)                        # (M,)

    # J* = R* @ (scale · S)   → (M, 3, 2)
    J_proj = torch.bmm(R_opt, scale.view(M_q, 1, 1) * S)      # (M, 3, 2)

    # ref_t: (4, 2) → (M, 4, 2)
    ref_b = ref_t.unsqueeze(0).expand(M_q, -1, -1)             # (M, 4, 2)
    # v_proj_shape = ref @ J_proj^T  → (M, 4, 3)
    shape_part = torch.bmm(ref_b, J_proj.transpose(1, 2))      # (M, 4, 3)
    t_opt = (v_q - shape_part).mean(dim=1, keepdim=True)       # (M, 1, 3)
    v_proj = shape_part + t_opt                                 # (M, 4, 3)

    # Scatter v_proj into proj_sum / count
    proj_sum = torch.zeros(N, 3, dtype=dtype, device=device)
    count    = torch.zeros(N,    dtype=dtype, device=device)

    # Flatten quad indices and projected vertices for scatter
    qi_flat = quads_t.reshape(-1)                               # (M*4,)
    vp_flat = v_proj.reshape(-1, 3)                             # (M*4, 3)

    proj_sum.index_add_(0, qi_flat, vp_flat)
    count.index_add_(0, qi_flat, torch.ones(M_q * 4, dtype=dtype, device=device))

    proj_avg = proj_sum / count.clamp(min=1.0).unsqueeze(1)    # (N, 3)
    return proj_avg


# ---------------------------------------------------------------------------
# Jacobian degeneracy determinant penalty  (differentiable)
# ---------------------------------------------------------------------------

def anti_flip_penalty(
    V: torch.Tensor,
    quads_t: torch.LongTensor,
    ref_t: torch.Tensor,
    A_ref_pinv_t: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Determinant-of-Gram degeneracy penalty:

        P_deg = (1/M) Σ_i  max(0,  ε − det(J_iᵀ J_i))²

    Quads with det(JᵀJ) ≥ ε contribute zero; rank-deficient quads are pushed
    away from zero area.  Since det(JᵀJ) is non-negative, this term does not
    detect mirror flips; signed flips are handled by the numpy inference
    regularizer and topology certification.

    Returns:
        scalar penalty tensor (grad-tracked).
    """
    M_q = quads_t.shape[0]

    v_q = V[quads_t]                                              # (M, 4, 3)
    A_pinv_b = A_ref_pinv_t.unsqueeze(0).expand(M_q, -1, -1)    # (M, 3, 4)
    X    = torch.bmm(A_pinv_b, v_q)                              # (M, 3, 3)
    J    = X[:, :2, :].transpose(1, 2)                           # (M, 3, 2)
    JtJ  = torch.bmm(J.transpose(1, 2), J)                       # (M, 2, 2)
    det  = torch.det(JtJ)                                         # (M,)
    pen  = torch.clamp(eps - det, min=0.0).pow(2)
    return pen.mean()


# ---------------------------------------------------------------------------
# UnrolledPDSolver
# ---------------------------------------------------------------------------

class UnrolledPDSolver(nn.Module):
    """
    Unrolled K-iteration Projective Dynamics solver.

    The solver is a PyTorch Module.  Its `forward()` method runs K iterations:
      - Warm-up iterations (no gradient):  0 … K − K_grad − 1
      - Differentiable tail:               K − K_grad … K − 1

    Gradient flows from the Jacobian loss (computed outside this module) back
    through the differentiable tail to V_init and M_targets, and ultimately
    to the DGCNN weights.
    """

    def __init__(
        self,
        quads: np.ndarray,
        n_vertices: int,
        smoothness_weight: float = 0.1,
        n_iter: int = 30,
        grad_iter: int = 5,
    ):
        """
        Args:
            quads:             (M, 4) quad face array (numpy, fixed topology).
            n_vertices:        N — number of mesh vertices.
            smoothness_weight: γ in  A = I + γ Lᵀ L.
            n_iter:            Total K PD iterations.
            grad_iter:         K_grad — number of differentiable tail iterations.
        """
        super().__init__()
        self.quads_np   = quads
        self.n_vertices = n_vertices
        self.n_iter     = n_iter
        self.grad_iter  = min(grad_iter, n_iter)

        # Build and factorize global system  A = I + γ Lᵀ L
        A_sp, A_factor = _build_global_system(n_vertices, quads, smoothness_weight)
        self._A_factor = A_factor   # scipy callable — not a tensor

        # Register buffers
        quads_t = torch.from_numpy(quads.astype(np.int64))
        self.register_buffer('quads_t', quads_t)

        ref = torch.tensor(_REF_SQUARE, dtype=torch.float64)
        self.register_buffer('ref_t', ref)

        a_pinv = torch.tensor(_A_REF_PINV_NP, dtype=torch.float64)
        self.register_buffer('A_ref_pinv_t', a_pinv)

    # ------------------------------------------------------------------
    def _target_shape_matrices(self, M_targets: torch.Tensor) -> torch.Tensor:
        """
        Compute S = M^{-1/2} for each quad target metric.

        Args:
            M_targets:  (M, 2, 2) per-quad target metrics (grad-tracked).

        Returns:
            S:          (M, 2, 2) shape matrices.
        """
        eigvals, eigvecs = torch.linalg.eigh(M_targets)          # (M,2),(M,2,2)
        inv_sqrt = 1.0 / eigvals.clamp(min=1e-6).sqrt()          # (M,2)
        S = eigvecs @ torch.diag_embed(inv_sqrt) @ eigvecs.transpose(-2, -1)
        return S

    # ------------------------------------------------------------------
    def _local_step(
        self, V: torch.Tensor, S: torch.Tensor
    ) -> torch.Tensor:
        return _local_step_vectorised(
            V, self.quads_t, S, self.ref_t, self.A_ref_pinv_t
        )

    # ------------------------------------------------------------------
    def _global_step(self, proj: torch.Tensor) -> torch.Tensor:
        """
        Solve  A V_new = proj  via implicit differentiation.
        """
        return _SparseCholeskyLinearSolve.apply(proj, self._A_factor)

    # ------------------------------------------------------------------
    def forward(
        self,
        V_init: torch.Tensor,
        M_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run K unrolled PD iterations.

        Args:
            V_init:     (N, 3) initial vertex positions.
            M_targets:  (M, 2, 2) per-quad target metric tensors (grad-tracked).

        Returns:
            V_final:    (N, 3) optimised vertex positions (grad-tracked).
        """
        S = self._target_shape_matrices(M_targets)   # (M, 2, 2) — grad-tracked

        # Cast to float64 for numerical stability
        V = V_init.double()
        S = S.double()
        if self.ref_t.dtype != torch.float64:
            ref = self.ref_t.double()
            pinv = self.A_ref_pinv_t.double()
        else:
            ref = self.ref_t
            pinv = self.A_ref_pinv_t

        warm = self.n_iter - self.grad_iter

        # ── Warm-up: no gradient ─────────────────────────────────────────
        with torch.no_grad():
            for _ in range(warm):
                proj = _local_step_vectorised(V, self.quads_t, S, ref, pinv)
                V = _SparseCholeskyLinearSolve.apply(proj, self._A_factor)

        # ── Differentiable tail ──────────────────────────────────────────
        V = V.detach().requires_grad_(False)   # detach warm-up result
        for _ in range(self.grad_iter):
            proj = _local_step_vectorised(V, self.quads_t, S, ref, pinv)
            V = _SparseCholeskyLinearSolve.apply(proj, self._A_factor)

        return V.float() if V_init.dtype == torch.float32 else V

    # ------------------------------------------------------------------
    def optimize_numpy(
        self,
        V_init: np.ndarray,
        M_targets: np.ndarray,
        n_iter: Optional[int] = None,
        tol: float = 1e-6,
        callback: Optional[Callable] = None,
    ) -> np.ndarray:
        """
        Pure-numpy optimisation loop (used at inference time).

        This avoids constructing a PyTorch computation graph and uses the same
        pre-factorised solver — much faster for large meshes.

        Args:
            V_init:    (N, 3) initial vertex positions.
            M_targets: (M, 2, 2) per-quad target metric tensors.
            n_iter:    Number of iterations (defaults to self.n_iter).
            tol:       Convergence tolerance on relative vertex change.
            callback:  Optional function(iter, V) called each iteration.

        Returns:
            V_opt:  (N, 3) optimised vertices.
        """
        if n_iter is None:
            n_iter = self.n_iter

        quads_np = self.quads_np
        ref_np   = _REF_SQUARE
        pinv_np  = _A_REF_PINV_NP
        A_factor = self._A_factor

        # Compute S = M^{-1/2} in numpy
        eigvals, eigvecs = np.linalg.eigh(M_targets)          # (M,2),(M,2,2)
        inv_sqrt = 1.0 / np.maximum(eigvals, 1e-6) ** 0.5
        S = np.einsum('mij,mj,mkj->mik', eigvecs, inv_sqrt, eigvecs)  # (M,2,2)

        V = V_init.copy().astype(np.float64)
        M_q = len(quads_np)
        N   = len(V)

        for it in range(n_iter):
            V_old = V.copy()

            # ── Local step ──────────────────────────────────────────────
            v_q  = V[quads_np]                  # (M, 4, 3)
            X    = np.einsum('ab,mbc->mac', pinv_np, v_q)    # (M, 3, 3)
            J    = X[:, :2, :].transpose(0, 2, 1)            # (M, 3, 2)

            K   = np.einsum('mab,mcb->mac', J, S)            # (M, 3, 2)
            U, _, Vt = np.linalg.svd(K, full_matrices=False) # (M,3,2),(M,2),(M,2,2)
            R   = np.einsum('mab,mbc->mac', U, Vt)           # (M, 3, 2)

            j_frob = np.linalg.norm(J.reshape(M_q, -1), axis=1)  # (M,)
            s_frob = np.linalg.norm(S.reshape(M_q, -1), axis=1)
            scale  = j_frob / (s_frob + 1e-10)               # (M,)

            J_proj = np.einsum('mab,mbc->mac', R, scale[:, None, None] * S)
            shape_part = np.einsum('ab,mbc->mac', ref_np, J_proj.transpose(0, 2, 1))
            t_opt  = (v_q - shape_part).mean(axis=1, keepdims=True)
            v_proj = shape_part + t_opt                       # (M, 4, 3)

            proj_sum = np.zeros((N, 3))
            cnt      = np.zeros(N)
            np.add.at(proj_sum, quads_np.reshape(-1), v_proj.reshape(-1, 3))
            np.add.at(cnt,      quads_np.reshape(-1), 1.0)
            proj_avg = proj_sum / np.maximum(cnt[:, None], 1.0)

            # ── Global step ──────────────────────────────────────────────
            V_new = np.empty_like(V)
            for d in range(3):
                V_new[:, d] = A_factor(proj_avg[:, d])
            V = V_new

            if callback is not None:
                callback(it, V)

            diff = np.linalg.norm(V - V_old) / (np.linalg.norm(V_old) + 1e-12)
            if diff < tol:
                break

        return V.astype(V_init.dtype)


# ---------------------------------------------------------------------------
# Backward-compatible numpy-based solver
# ---------------------------------------------------------------------------
# The old ProjectiveDynamicsSolver interface is preserved below so that
# reconstruct.py continues to work before it is updated to use UnrolledPDSolver.

from .constraints import (
    ShapeConstraint,
    FeaturePointConstraint,
    FeatureEdgeConstraint,
    AntiFlipRegularizer,
)


class ProjectiveDynamicsSolver:
    """
    Legacy numpy PD solver (kept for backward compatibility).

    New code should use UnrolledPDSolver for differentiable training.
    This class is used by scripts/reconstruct.py for inference.
    """

    def __init__(
        self,
        vertices: np.ndarray,
        quads: np.ndarray,
        target_metric_field: Callable[[int], np.ndarray],
        ref_square: Optional[np.ndarray] = None,
        smoothness_weight: float = 0.1,
        feature_point_constraints: Optional[List] = None,
        feature_edge_constraints: Optional[List] = None,
        anti_flip_weight: float = 10.0,
        anti_flip_eps: float = 1e-6,
        surface_attach_weight: float = 0.0,
        flip_backtracking: bool = True,
        flip_backtrack_steps: int = 8,
    ):
        self.V    = vertices.copy()
        self.quads = quads
        self.M    = quads.shape[0]
        self.N    = vertices.shape[0]
        self.target_metric_field = target_metric_field
        self.ref_square = (
            ref_square if ref_square is not None
            else _REF_SQUARE.copy()
        )
        self.smoothness_weight         = smoothness_weight
        self.feature_point_constraints = feature_point_constraints or []
        self.feature_edge_constraints  = feature_edge_constraints  or []
        self.surface_attach_weight = float(surface_attach_weight)
        self.flip_backtracking = bool(flip_backtracking)
        self.flip_backtrack_steps = int(flip_backtrack_steps)

        if anti_flip_weight > 0:
            self.anti_flip = AntiFlipRegularizer(quads, eps0=anti_flip_eps, weight=anti_flip_weight)
        else:
            self.anti_flip = None

        self.shape_constraints = [
            ShapeConstraint(self.target_metric_field(i), quads[i].tolist())
            for i in range(self.M)
        ]

        self._build_laplacian()
        self._prefactorize()

    def _build_laplacian(self):
        L = _build_quad_laplacian(self.N, self.quads)
        self.L = L

    def _prefactorize(self):
        reg = self.smoothness_weight * (self.L.T @ self.L)
        A   = sparse.eye(self.N, format='csc') + reg
        try:
            self._solve_factor = factorized(A)
        except Exception:
            self._solve_factor = None

    def local_step(self) -> np.ndarray:
        proj_sum = np.zeros_like(self.V)
        count    = np.zeros(self.N, dtype=np.float64)

        for i, sc in enumerate(self.shape_constraints):
            sc.target_metric = self.target_metric_field(i)
            idx    = sc.indices
            v_proj = sc.project(self.V, self.ref_square)

            for fec in self.feature_edge_constraints:
                if fec.quad_idx == i:
                    v_proj = fec.modify_local_projection(v_proj, self.V, idx)

            for j, gi in enumerate(idx):
                proj_sum[gi] += v_proj[j]
                count[gi]    += 1.0

        if self.anti_flip is not None:
            self.anti_flip.apply(self.V, proj_sum, count)

        proj_avg = proj_sum / np.maximum(count, 1.0)[:, None]

        for fpc in self.feature_point_constraints:
            proj_avg[fpc.idx] = fpc.project(proj_avg[fpc.idx])

        return proj_avg

    def global_step(self, proj_targets: np.ndarray) -> np.ndarray:
        V_new = np.empty_like(self.V)
        if self._solve_factor is not None:
            for c in range(3):
                V_new[:, c] = self._solve_factor(proj_targets[:, c])
        else:
            from scipy.sparse.linalg import cg
            A = sparse.eye(self.N) + self.smoothness_weight * self.L.T @ self.L
            for c in range(3):
                res, info = cg(A, proj_targets[:, c], x0=self.V[:, c])
                V_new[:, c] = res if info == 0 else self.V[:, c]
        return V_new

    def optimize(
        self,
        surface_points: np.ndarray,
        surface_normals: Optional[np.ndarray] = None,
        max_iter: int = 30,
        tol: float = 1e-6,
        callback: Optional[Callable] = None,
    ) -> np.ndarray:
        surface_tree = None
        surf_pts = np.asarray(surface_points, dtype=np.float64)
        surf_nrm = None
        if self.surface_attach_weight > 0.0 and len(surf_pts) > 0:
            from scipy.spatial import KDTree
            surface_tree = KDTree(surf_pts)
            if surface_normals is not None:
                surf_nrm = np.asarray(surface_normals, dtype=np.float64)
                if surf_nrm.shape != surf_pts.shape:
                    raise ValueError(
                        "surface_normals must have the same shape as surface_points: "
                        f"{surf_nrm.shape} != {surf_pts.shape}"
                    )

        # Pre-repair flipped quads
        if self.anti_flip is not None:
            from .constraints import _quad_signed_areas_vectorized
            areas = _quad_signed_areas_vectorized(self.V, self.quads)
            bad   = np.where(areas < self.anti_flip.eps0)[0]
            if len(bad) > 0:
                for bi in bad:
                    verts = self.quads[bi]
                    c = self.V[verts].mean(axis=0)
                    sc = min(5.0, np.sqrt(self.anti_flip.eps0 / max(abs(areas[bi]), 1e-12)))
                    for vi in verts:
                        self.V[vi] = c + sc * (self.V[vi] - c)

        for it in range(max_iter):
            V_old = self.V.copy()
            proj = self.local_step()
            if surface_tree is not None:
                _, nn_idx = surface_tree.query(self.V)
                attach = surf_pts[nn_idx]
                if surf_nrm is not None:
                    n = surf_nrm[nn_idx]
                    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
                    offset = np.sum((proj - attach) * n, axis=1, keepdims=True)
                    attach = proj - offset * n
                w = self.surface_attach_weight
                proj = (proj + w * attach) / (1.0 + w)
            V_prop = self.global_step(proj)
            if self.flip_backtracking and self.anti_flip is not None:
                from .constraints import _quad_signed_areas_vectorized

                def _counts(Vv: np.ndarray) -> tuple[int, int]:
                    areas = _quad_signed_areas_vectorized(Vv, self.quads)
                    n_flip = int((areas < 0.0).sum())
                    n_near = int(((areas >= 0.0) & (areas < self.anti_flip.eps0)).sum())
                    return n_flip, n_near

                old_counts = _counts(V_old)
                prop_counts = _counts(V_prop)
                if (prop_counts[0] > old_counts[0]) or (
                    old_counts[0] == 0 and prop_counts[0] > 0
                ):
                    best_V = V_old
                    best_counts = old_counts
                    accepted = False
                    alpha = 1.0
                    for _ in range(max(1, self.flip_backtrack_steps)):
                        cand = V_old + alpha * (V_prop - V_old)
                        cand_counts = _counts(cand)
                        if (cand_counts[0] < best_counts[0]) or (
                            cand_counts[0] == best_counts[0] and cand_counts[1] <= best_counts[1]
                        ):
                            best_V = cand
                            best_counts = cand_counts
                        if (cand_counts[0] <= old_counts[0]) and (
                            cand_counts[1] <= max(old_counts[1], prop_counts[1])
                        ):
                            self.V = cand
                            accepted = True
                            break
                        alpha *= 0.5
                    if not accepted:
                        self.V = best_V
                else:
                    self.V = V_prop
            else:
                self.V = V_prop
            diff   = np.linalg.norm(self.V - V_old) / (np.linalg.norm(V_old) + 1e-12)
            if callback:
                callback(it, self.V)
            if diff < tol:
                break

        return self.V
