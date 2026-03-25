"""
Global parameterization: flatten the mesh to UV space while preserving metric alignment.
Implements Section 6.3: "Inverse Ellipse Rectification".
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Optional, Tuple, List
from ..geometry.metric_utils import logm_spd, expm_spd, tensor_to_params, params_to_tensor


class ParameterizationSolver:
    """
    Solves for UV coordinates of each vertex such that the pullback metric matches the
    3D mesh's metric (or a target metric, e.g., identity for conformal).
    """

    def __init__(
        self,
        vertices: np.ndarray,
        quads: np.ndarray,
        metric_field: np.ndarray,  # (M,2,2) target metric for each quad (in 3D)
        ref_square: Optional[np.ndarray] = None,
    ):
        """
        Args:
            vertices: (N,3) mesh vertices.
            quads: (M,4) quad indices.
            metric_field: target metric tensors for each quad (in 3D).
            ref_square: (4,2) reference square (default unit square).
        """
        self.V = vertices
        self.quads = quads
        self.M = quads.shape[0]
        self.N = vertices.shape[0]
        self.metric_field = metric_field
        if ref_square is None:
            self.ref_square = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=float)
        else:
            self.ref_square = ref_square

    def compute_local_jacobians(self, uv: np.ndarray) -> np.ndarray:
        """
        Given current UV coordinates (N,2), compute per-quad Jacobians J (2x2) mapping ref_square to UV.
        Returns array of shape (M,2,2).
        """
        J = np.zeros((self.M, 2, 2))
        for i, quad in enumerate(self.quads):
            uv_quad = uv[quad]  # (4,2)
            # Solve for affine map: uv ≈ J @ ref + t
            A = np.hstack([self.ref_square, np.ones((4,1))])  # (4,3)
            X, _, _, _ = np.linalg.lstsq(A, uv_quad, rcond=None)
            # J is first two rows? Actually X is (3,2). X[:2, :] is (2,2). Transpose?
            # Let's derive: uv (4,2) = A (4,3) @ X (3,2). X[0,:] is coefficients for ref_u, X[1,:] for ref_v, X[2,:] for translation.
            # So the Jacobian J (2x2) such that J @ ref gives the linear part: J[:,0] = X[0,:], J[:,1] = X[1,:]. So J = X[:2, :].T? Actually J should be 2x2, with columns = partial derivatives. So J[0,0] = X[0,0], J[1,0] = X[0,1]? This is confusing.
            # Simpler: The linear map from ref (2D) to uv (2D) can be represented by a 2x2 matrix L such that L @ ref_i ≈ uv_i - t.
            # We can solve L from (ref_i) as 4x2 system. Let's do that directly:
            ref = self.ref_square  # (4,2)
            uv_c = uv_quad - uv_quad.mean(axis=0)  # center
            ref_c = ref - ref.mean(axis=0)
            # Solve ref_c @ L.T = uv_c  => L.T = ref_c \ uv_c
            L, _, _, _ = np.linalg.lstsq(ref_c, uv_c, rcond=None)
            J[i] = L.T  # (2,2)
        return J

    def compute_metric_from_jacobian(self, J: np.ndarray) -> np.ndarray:
        """
        Compute pullback metric from Jacobian: M = J^{-T} J^{-1}.
        """
        # For each quad, compute metric
        M = np.zeros((self.M, 2, 2))
        for i in range(self.M):
            Jinv = np.linalg.inv(J[i])
            M[i] = Jinv.T @ Jinv
        return M

    def parameterization_energy(self, uv: np.ndarray) -> float:
        """
        Compute total Log-Euclidean distance between current pullback metric and target metric.
        """
        J = self.compute_local_jacobians(uv)
        M_curr = self.compute_metric_from_jacobian(J)
        total = 0.0
        for i in range(self.M):
            log_curr = logm_spd(M_curr[i])
            log_tgt = logm_spd(self.metric_field[i])
            diff = log_curr - log_tgt
            total += np.sum(diff**2)
        return total

    def solve(
        self,
        fixed_vertices: Optional[List[int]] = None,
        fixed_uv: Optional[np.ndarray] = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> np.ndarray:
        """
        Solve for UV coordinates by minimizing the metric distortion energy.
        This is a nonlinear optimization; we use a simple gradient descent or Newton.
        For simplicity, we implement a Gauss-Newton method.

        Args:
            fixed_vertices: list of vertex indices to keep fixed.
            fixed_uv: (len(fixed_vertices),2) fixed UV positions.
            max_iter: maximum iterations.
            tol: convergence tolerance.

        Returns:
            uv: (N,2) UV coordinates.
        """
        if fixed_vertices is None:
            fixed_vertices = []
        if fixed_uv is None:
            fixed_uv = np.zeros((len(fixed_vertices),2))

        # Initial guess: simple harmonic map (solve Laplace equation)
        uv = self._initial_guess(fixed_vertices, fixed_uv)

        # Gauss-Newton iterations
        for it in range(max_iter):
            # Compute Jacobians and residuals
            J_quads = self.compute_local_jacobians(uv)  # (M,2,2)
            # Build system: we need to compute gradient of energy w.r.t. uv.
            # Energy per quad: ||log(M_curr) - log(M_tgt)||_F^2.
            # Deriving analytic gradient is complex; we use numerical gradient for simplicity.
            # For production, one would implement a custom sensitivity analysis.
            # Here we just use a simple gradient descent as placeholder.
            grad = self._compute_energy_gradient(uv, J_quads)
            # Update
            step = 0.01
            uv_new = uv - step * grad
            # Apply fixed constraints
            for idx, uv_fixed in zip(fixed_vertices, fixed_uv):
                uv_new[idx] = uv_fixed
            # Check convergence
            if np.linalg.norm(uv_new - uv) < tol:
                uv = uv_new
                break
            uv = uv_new
        return uv

    def _initial_guess(self, fixed_vertices, fixed_uv):
        """Solve Laplace equation with fixed boundaries for initial UV."""
        # Build Laplacian matrix for vertices (graph Laplacian)
        edges = set()
        for quad in self.quads:
            for i in range(4):
                a, b = quad[i], quad[(i+1)%4]
                edges.add((a,b))
                edges.add((b,a))
        N = self.N
        row, col, data = [], [], []
        for (a,b) in edges:
            row.append(a)
            col.append(b)
            data.append(1.0)
        A = sparse.csr_matrix((data, (row, col)), shape=(N, N))
        deg = np.array(A.sum(axis=1)).flatten()
        D = sparse.diags(deg)
        L = D - A  # Laplacian

        # Set up system for each coordinate with fixed vertices
        b = np.zeros((N,2))
        # Modify L for fixed vertices: set row i to identity and b to fixed value
        L_fixed = L.tolil()
        for i, idx in enumerate(fixed_vertices):
            L_fixed[idx] = 0
            L_fixed[idx, idx] = 1
            b[idx] = fixed_uv[i]
        L_fixed = L_fixed.tocsc()
        # Solve
        uv = np.zeros((N,2))
        for c in range(2):
            uv[:,c] = spsolve(L_fixed, b[:,c])
        return uv

    def _compute_energy_gradient(self, uv, J_quads):
        """Numerical gradient (finite differences) for demonstration."""
        eps = 1e-6
        grad = np.zeros_like(uv)
        E0 = self.parameterization_energy(uv)
        for i in range(self.N):
            for d in range(2):
                uv_plus = uv.copy()
                uv_plus[i, d] += eps
                E_plus = self.parameterization_energy(uv_plus)
                grad[i, d] = (E_plus - E0) / eps
        return grad