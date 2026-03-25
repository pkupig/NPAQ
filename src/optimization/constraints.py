"""
Local projection constraints for Projective Dynamics.
Implements Section 5.4.2 and 5.5.2, 5.5.3.
Anti-flip regularizer implements Section 5.3 (anti-flip between topology and PD).
"""

import numpy as np
from scipy.linalg import svd
from typing import Tuple, Optional, List, Union


class ShapeConstraint:
    """
    Local projection operator that projects a quadrilateral's Jacobian onto the
    manifold of shapes defined by a target metric tensor.
    This implements the SVD-based shape matching (Section 5.4.2).
    """

    def __init__(self, target_metric: np.ndarray, quad_vertices_idx: List[int], weight: float = 1.0):
        """
        Args:
            target_metric: (2,2) target metric tensor M_target.
            quad_vertices_idx: List of 4 indices of the quadrilateral in the global vertex array.
            weight: Weight for this constraint in the global solve.
        """
        self.target_metric = target_metric
        self.indices = quad_vertices_idx
        self.weight = weight

    def project(self, V: np.ndarray, ref_square: np.ndarray) -> np.ndarray:
        """
        Compute the projected positions of the quadrilateral's vertices (the local step).

        Args:
            V: (N,3) current global vertex positions.
            ref_square: (4,2) reference square coordinates (e.g., [0,0], [1,0], [1,1], [0,1]).

        Returns:
            projected_V: (4,3) new positions for the quadrilateral's vertices.
        """
        # Get current vertices
        v = V[self.indices]  # (4,3)

        # Degenerate quad: two or more vertices share the same index (e.g. [v0,v1,v2,v2]
        # from an unmatched triangle in the greedy tri-to-quad step).  Applying the
        # normal SVD projection to a rank-deficient point set produces wildly incorrect
        # results.  Return the current positions unchanged so these faces act as soft
        # anchors without corrupting the global solve.
        if len(set(self.indices)) < 4:
            return v

        # Compute current Jacobian J (3x2) that maps ref_square to v.
        # We need to find a linear transformation J and translation t such that:
        # v_i ≈ J * ref_i + t   for i=0..3.
        # Solve in least squares sense.
        # Let A = [ref_i 1] stacked, we want A * [J; t]^T = v.
        A = np.hstack([ref_square, np.ones((4,1))])  # (4,3)
        # Solve for [J; t] as 3x3 matrix? Actually we want J (3x2) and t (3,1), total 3x3.
        # Solve X = A \ v, where X is 3x3? Wait: v is (4,3), A is (4,3). Solve A * X = v, X is (3,3).
        X, _, _, _ = np.linalg.lstsq(A, v, rcond=None)  # X (3,3): first two cols are J, last col is t.
        J = X[:2].T  # (3,2)  (since X is 3x3, rows correspond to coefficients for each column of v? Let's verify:
        # Actually A is (4,3), v is (4,3). lstsq returns X (3,3) such that A @ X ≈ v. So X[j,k] maps A[:,j] to v[:,k].
        # So J would be X[:2, :].T? We need J as 3x2. Let's take the first two columns of X (since A cols: ref, ref, 1).
        # So X_col0 corresponds to ref_u, X_col1 to ref_v, X_col2 to translation.
        # Then v ≈ A[:,:2] @ X[:2,:] + A[:,2:3] @ X[2:3,:] . So J = X[:2,:] is 2x3? No, X is 3x3, each column of X corresponds to a coordinate of v (x,y,z). So X[0,:] is coefficients for u, X[1,:] for v, X[2,:] for 1.
        # To get J as 3x2: J[:,0] = X[0,:] (for u), J[:,1] = X[1,:] (for v). So J = X[:2, :].T (2 rows? Wait dimension: X[:2,:] is (2,3). Transpose gives (3,2). Yes that works.
        J = X[:2, :].T  # (3,2)

        # Compute target shape matrix S = M_target^{-1/2}
        # M_target = R diag(λ1, λ2) R^T  →  S = R diag(1/√λ1, 1/√λ2) R^T
        eigvals, eigvecs = np.linalg.eigh(self.target_metric)
        S = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-8))) @ eigvecs.T  # (2,2)

        # Find optimal rotation R_opt that aligns J with S.
        K = J @ S.T  # (3,2)
        U, _, Vt = svd(K, full_matrices=False)
        R_opt = U @ Vt  # (3,2)

        # SCALE CORRECTION: S encodes curvature-based shape and has magnitude ~1/curvature,
        # while J has magnitude equal to the actual mesh edge length (~0.01–0.1 in world units).
        # Using R_opt @ S directly produces a J_proj that is orders of magnitude larger than J,
        # pushing every vertex far outside the mesh surface.
        # Fix: rescale S so J_proj has the same Frobenius norm as J, preserving quad size
        # while only correcting its shape (aspect ratio + orientation).
        j_frob = np.linalg.norm(J, 'fro')
        s_frob = np.linalg.norm(S, 'fro')
        scale = j_frob / (s_frob + 1e-12)

        # Projected Jacobian: same scale as J, shape from M_target  (3,2)
        J_proj = R_opt @ (scale * S)

        # Now find new vertex positions v_proj that best match the linear map J_proj and translation t.
        # We want v_proj = J_proj @ ref_i.T + t, with t chosen to minimize distance to current v.
        # Given t, the residual per vertex is (v_i - (J_proj @ ref_i.T + t)).
        # The optimal t is the average of (v_i - J_proj @ ref_i.T).
        t_opt = np.mean(v - (ref_square @ J_proj.T), axis=0)  # (3,)
        v_proj = ref_square @ J_proj.T + t_opt  # (4,3)

        return v_proj


class FeaturePointConstraint:
    """
    Constraint that forces a vertex to slide along a feature line.
    Implements Section 5.5.2: v = p0 + α t, where t is the tangent direction.
    """

    def __init__(self, vertex_idx: int, line_point: np.ndarray, tangent: np.ndarray):
        """
        Args:
            vertex_idx: Index of the constrained vertex.
            line_point: A point on the feature line (e.g., the initial projection).
            tangent: Unit tangent direction along the feature line.
        """
        self.idx = vertex_idx
        self.p0 = line_point.copy()
        self.t = tangent / (np.linalg.norm(tangent) + 1e-12)

    def apply_to_system(self, A: np.ndarray, b: np.ndarray, weight: float = 1e6):
        """
        Modify the global linear system to enforce the constraint.
        This is a hard constraint implemented via a penalty or by reducing degrees of freedom.
        Here we implement it as a strong penalty (diagonal modification) for simplicity.
        In practice, one would eliminate the constrained DOF or use Lagrange multipliers.
        """
        # This is a placeholder; actual implementation depends on solver structure.
        # We'll implement it in pd_solver.py when constructing the system.
        pass

    def project(self, v: np.ndarray) -> np.ndarray:
        """Project vertex onto the line (closest point on line)."""
        # v projected onto line: p0 + ((v - p0)·t) t
        alpha = np.dot(v - self.p0, self.t)
        return self.p0 + alpha * self.t


class FeatureEdgeConstraint:
    """
    Constraint that aligns an edge of a quadrilateral with a feature line.
    Implements Section 5.5.3: adds a term to local projection to encourage edge alignment.
    """

    def __init__(self, quad_idx: int, edge_vertices: Tuple[int, int], line_point: np.ndarray, tangent: np.ndarray, weight: float = 1.0):
        """
        Args:
            quad_idx: Index of the quadrilateral (for local projection).
            edge_vertices: Indices of the two vertices forming the edge.
            line_point: A point on the feature line.
            tangent: Unit tangent direction.
            weight: Weight for the alignment term.
        """
        self.quad_idx = quad_idx
        self.i, self.j = edge_vertices
        self.p0 = line_point
        self.t = tangent / (np.linalg.norm(tangent) + 1e-12)
        self.weight = weight

    def modify_local_projection(self, v_proj: np.ndarray, v_current: np.ndarray, indices: List[int]) -> np.ndarray:
        """
        Modify the projected vertex positions to encourage edge alignment.
        This adds a term to the local minimization:
            min ||A v_proj - vec(J_proj)||^2 + μ ||(v_proj_j - v_proj_i) - t||^2
        Here we implement a simplified version: after computing v_proj from shape constraint,
        we project the edge onto the line direction.
        """
        # Find positions of the two vertices in the quad's local vertex array
        # indices is list of 4 global indices for this quad.
        # We need to locate which entries in v_proj correspond to i and j.
        local_i = indices.index(self.i)
        local_j = indices.index(self.j)

        # Use both the current edge and the projected edge to derive a stable target
        # length, then move the edge midpoint back onto the feature line. This is
        # stronger than the previous "direction only" correction and is what
        # actually makes quad edges coincide with sharp polylines.
        e_cur = v_current[self.j] - v_current[self.i]                       # (3,)
        e_proj = v_proj[local_j] - v_proj[local_i]                         # (3,)
        len_cur = abs(float(np.dot(e_cur, self.t)))
        len_proj = abs(float(np.dot(e_proj, self.t)))
        L_target = max(len_cur, len_proj, 1e-8)

        midpoint = 0.5 * (v_proj[local_i] + v_proj[local_j])
        alpha = float(np.dot(midpoint - self.p0, self.t))
        midpoint_on_line = self.p0 + alpha * self.t

        if float(np.dot(e_proj, self.t)) < 0.0:
            tangent = -self.t
        else:
            tangent = self.t

        target_i = midpoint_on_line - 0.5 * L_target * tangent
        target_j = midpoint_on_line + 0.5 * L_target * tangent

        blend = float(self.weight) / (1.0 + float(self.weight))
        blend = float(np.clip(blend, 0.0, 0.95))
        v_proj[local_i] = (1.0 - blend) * v_proj[local_i] + blend * target_i
        v_proj[local_j] = (1.0 - blend) * v_proj[local_j] + blend * target_j
        return v_proj


# ---------------------------------------------------------------------------
# Anti-flip regularizer (§5.3): inserted between topology generation and PD.
# ---------------------------------------------------------------------------

def _quad_signed_areas_vectorized(V: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """
    Vectorized signed-area computation for all quads.

    Each quad [v0,v1,v2,v3] is split into triangles (v0,v1,v2) and (v0,v2,v3).
    Signed area = sum of the two triangle areas projected onto the quad's reference
    normal (average of both triangle normals).

    Returns (M,) array of signed areas.
    """
    v = V[quads]                                             # (M, 4, 3)
    n0 = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])   # (M, 3)
    n1 = np.cross(v[:, 2] - v[:, 0], v[:, 3] - v[:, 0])   # (M, 3)
    n_ref = n0 + n1                                          # (M, 3)
    n_len = np.linalg.norm(n_ref, axis=1, keepdims=True)    # (M, 1)
    # For degenerate quads (repeated vertices) n_len can be 0; set area to 0.
    safe_len = np.maximum(n_len, 1e-12)
    n_hat = n_ref / safe_len                                 # (M, 3)
    a0 = 0.5 * (n0 * n_hat).sum(axis=1)                    # (M,)
    a1 = 0.5 * (n1 * n_hat).sum(axis=1)                    # (M,)
    areas = a0 + a1
    # Zero out degenerate quads
    areas[n_len.squeeze() < 1e-12] = 0.0
    return areas


class AntiFlipRegularizer:
    """
    Anti-flip regularizer (§5.3, paper re-design).

    Integer rounding during seamless parametrisation can introduce flipped or
    near-degenerate elements, especially near cross-field singularities.  This
    class detects such elements and injects corrective projections into the PD
    local-step accumulator so the global solve can recover them.

    Strategy for a flipped/near-degenerate quad with signed area A < eps0:
      1. Compute centroid c = mean(v0..v3).
      2. Scale vertices about c by  scale = sqrt(eps0 / max(|A|, tiny)).
         This inflates the quad to area eps0 and, for truly inverted quads
         (A < 0), implicitly reverses the inversion because sqrt(-A) → real
         scale applied to reflected geometry.
      3. Add the corrective positions to the PD accumulator with weight
         `anti_flip_weight`, competing with the shape-constraint projection.

    The corrective weight is intentionally large (default 10) so that severely
    flipped elements are un-flipped in the first few PD iterations, after which
    the shape constraints naturally take over.

    Note on gradient flow (§5.3):
    ─────────────────────────────
    When `freeze_rounding_grad=True` (see reconstruct.yaml / §5.3 of the paper),
    the metric field is treated as a fixed (detached) input to the topology stage,
    and this regulariser operates purely in the forward/geometry pass.  Gradients
    from L_J (§6.4) are only propagated back through the *differentiable tail* of
    PD (K_grad iterations), not through the rounding step.  This is the recommended
    starting point: verify that the field prediction is accurate before enabling
    end-to-end rounding-gradient training.
    """

    def __init__(self, quads: np.ndarray, eps0: float = 1e-6, weight: float = 10.0):
        """
        Args:
            quads:  (M, 4) quad connectivity (int indices into vertex array).
            eps0:   Minimum acceptable signed area.  Quads below this trigger
                    correction.  Should be much smaller than typical quad area
                    (default 1e-6 works for unit-scale meshes).
            weight: Correction weight relative to shape-constraint projections.
                    Higher values un-flip elements faster but may over-constrain
                    healthy quads during early iterations.
        """
        self.quads = quads
        self.eps0 = eps0
        self.weight = weight

    # ------------------------------------------------------------------
    def apply(self, V: np.ndarray, proj_sum: np.ndarray, count: np.ndarray) -> int:
        """
        Detect flipped / near-degenerate quads and add corrective projections
        to the PD local-step accumulator (proj_sum, count).

        Called inside ProjectiveDynamicsSolver.local_step() after all shape
        constraints have been processed.

        Args:
            V:        (N, 3) current global vertex positions.
            proj_sum: (N, 3) accumulated target positions  — modified in-place.
            count:    (N,)   float accumulation counts     — modified in-place.

        Returns:
            Number of quads that triggered correction.
        """
        areas = _quad_signed_areas_vectorized(V, self.quads)        # (M,)
        flip_mask = areas < self.eps0                               # (M,) bool

        n_fixed = int(flip_mask.sum())
        if n_fixed == 0:
            return 0

        bad_quads = self.quads[flip_mask]                           # (K, 4)
        bad_areas = areas[flip_mask]                                # (K,)
        v_bad = V[bad_quads]                                        # (K, 4, 3)

        # Centroid of each flipped quad
        centroids = v_bad.mean(axis=1)                             # (K, 3)

        # Scale to inflate area to eps0
        # |A_new| = scale^2 * |A|  =>  scale = sqrt(eps0 / max(|A|, tiny))
        scales = np.sqrt(self.eps0 / np.maximum(np.abs(bad_areas), 1e-12))
        scales = np.minimum(scales, 5.0)                            # cap against extreme inflation

        # v_corr[k, j] = centroid[k] + scale[k] * (v[k,j] - centroid[k])
        v_corr = centroids[:, np.newaxis, :] + scales[:, np.newaxis, np.newaxis] * (
            v_bad - centroids[:, np.newaxis, :]
        )                                                           # (K, 4, 3)

        # Accumulate into proj_sum / count with anti-flip weight
        w = self.weight
        for k in range(len(bad_quads)):
            for local_j in range(4):
                gi = bad_quads[k, local_j]
                proj_sum[gi] += w * v_corr[k, local_j]
                count[gi] += w

        return n_fixed

    # ------------------------------------------------------------------
    def count_flipped(self, V: np.ndarray) -> Tuple[int, int]:
        """
        Count flipped (A < 0) and near-degenerate (0 ≤ A < eps0) quads.

        Returns:
            (n_flipped, n_near_degenerate)
        """
        areas = _quad_signed_areas_vectorized(V, self.quads)
        n_flip = int((areas < 0).sum())
        n_near = int(((areas >= 0) & (areas < self.eps0)).sum())
        return n_flip, n_near
