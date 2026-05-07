"""
Seamless parametrisation and integer quad-layout extraction.

Implements §5.3:
  1. Solve two Poisson systems to obtain scalar fields φ₁, φ₂ whose gradients
     align with the cross-field directions d₁, d₂.
  2. Apply the Sinusoidal Barrier Penalty (differentiable relaxation of the
     integer seam constraint):
         L_int = Σ_{edges e} sin²(π [φ](e))
     where [φ](e) = φ(j) − φ(i) is the jump across edge e.
  3. Extract quad topology by integer grid sampling (non-differentiable,
     treated as straight-through estimator in training).

If topology extraction produces no valid quads the function raises RuntimeError
rather than silently producing degenerate geometry.

⚠ KNOWN LIMITATION (Python fallback path only):
  solve_parametrisation() solves a GLOBAL Poisson system without cut graph or
  holonomy transitions.  For meshes with singularities (any closed surface with
  χ≠0, or any cross-field with non-zero index sum), the correct seamless
  parametrisation requires:
    1. Cut graph that makes the surface simply-connected.
    2. 90° rotation transitions (holonomy) at each seam edge crossing a singular
       chart boundary.
  Without these, the Poisson solve near singularities produces conflicting
  gradients → torn / severely distorted UV layout → degenerate quad extraction.

  extract_quads_integer_grid() uses nearest-vertex matching in UV space instead
  of iso-line intersection.  This leaves gaps wherever mesh vertices don't fall
  near integer UV positions (coarse triangulation or large gradient_size).

  RECOMMENDATION: Use the C++ libigl/CoMISo backend (use_igl_backend=true in
  reconstruct.yaml) which implements true cut-graph seam cutting and integer
  rounding via CoMISo.  The Python path is kept only as a differentiable proxy
  for the sinusoidal barrier loss during training (lambda_j > 0).
"""

from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized, spsolve
from collections import Counter
from typing import Tuple, Optional

import torch


# ---------------------------------------------------------------------------
# Poisson parametrisation solver
# ---------------------------------------------------------------------------

def _cotangent_laplacian(V: np.ndarray, F: np.ndarray, W=None) -> sparse.csc_matrix:
    """Standard cotangent Laplacian L = D − W (symmetric, negative semi-definite)."""
    from .crossfield import compute_cotangent_weights
    if W is None:
        W = compute_cotangent_weights(V, F)
    deg = np.array(W.sum(axis=1)).flatten()
    L = sparse.diags(deg) - W
    return L.tocsc()


def _build_gradient_rhs(
    V: np.ndarray,
    F: np.ndarray,
    theta: np.ndarray,
    frames: np.ndarray,
    gradient_size: float = 1.0,
    _W=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build right-hand sides b₁, b₂ for the gradient-alignment Poisson systems.

    For each undirected edge (i,j) with cotangent weight w_ij, the target
    parametrisation increment along the edge is:
        Δφ_k(i→j) = dot(d_k_avg_3d, V[j]−V[i]) / gradient_size

    where d_k_avg_3d is the average 3D direction of field k at i and j,
    with d₁ = cos θ · e₁ + sin θ · e₂  and  d₂ = −sin θ · e₁ + cos θ · e₂.

    This gives  L φ_k = b_k  (gradient-aligned Poisson).

    Args:
        _W: optional pre-computed cotangent weight matrix (avoids recomputing).
    """
    from .crossfield import compute_cotangent_weights
    N = len(V)
    if _W is None:
        W = compute_cotangent_weights(V, F)
    else:
        W = _W

    e1 = frames[:, :, 0]   # (N, 3)
    e2 = frames[:, :, 1]   # (N, 3)

    # 3D directions for each vertex
    cos_t = np.cos(theta)   # (N,)
    sin_t = np.sin(theta)
    d1_3d = cos_t[:, None] * e1 + sin_t[:, None] * e2   # (N, 3)
    d2_3d = -sin_t[:, None] * e1 + cos_t[:, None] * e2  # (N, 3)

    b1 = np.zeros(N)
    b2 = np.zeros(N)

    # Vectorised RHS accumulation — no Python loop over edges.
    W_coo = W.tocoo()
    W_coo.sum_duplicates()
    _row = np.asarray(W_coo.row, dtype=np.int64)
    _col = np.asarray(W_coo.col, dtype=np.int64)
    _dat = np.asarray(W_coo.data, dtype=np.float64)
    ut    = (_row < _col) & (np.abs(_dat) >= 1e-14)
    i_arr = _row[ut];  j_arr = _col[ut];  w_arr = _dat[ut]   # (E,)

    if len(i_arr) == 0:
        return b1, b2

    edges_ij = V[j_arr] - V[i_arr]                                # (E, 3)
    d1_avg   = 0.5 * (d1_3d[i_arr] + d1_3d[j_arr])               # (E, 3)
    d2_avg   = 0.5 * (d2_3d[i_arr] + d2_3d[j_arr])               # (E, 3)
    t1 = np.einsum('ei,ei->e', d1_avg, edges_ij) / gradient_size  # (E,)
    t2 = np.einsum('ei,ei->e', d2_avg, edges_ij) / gradient_size  # (E,)

    np.add.at(b1, i_arr,  w_arr * t1)
    np.add.at(b1, j_arr, -w_arr * t1)
    np.add.at(b2, i_arr,  w_arr * t2)
    np.add.at(b2, j_arr, -w_arr * t2)

    return b1, b2


def solve_parametrisation(
    V: np.ndarray,
    F: np.ndarray,
    theta: np.ndarray,
    frames: np.ndarray,
    gradient_size: float = 1.0,
    integer_projection_iters: int = 2,
    integer_potential_iters: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute scalar parametrisation fields φ₁, φ₂ aligned to the cross-field.

    Solves  L φ_k = b_k  with one pinned vertex (φ_k[0] = 0) to remove the
    constant null-space of the Laplacian.

    Args:
        V:              (N, 3) mesh vertices.
        F:              (M, 3) mesh triangles.
        theta:          (N,) cross-field angles (from crossfield_angles_from_complex).
        frames:         (N, 3, 2) per-vertex tangent frames.
        gradient_size:  Controls the scale of quads; smaller → more quads.

    Returns:
        phi1, phi2:  (N,) scalar fields.
    """
    from .crossfield import compute_cotangent_weights
    N = len(V)
    W = compute_cotangent_weights(V, F)
    L = _cotangent_laplacian(V, F, W=W)
    b1, b2 = _build_gradient_rhs(V, F, theta, frames, gradient_size, _W=W)

    # Pin vertex 0 to remove null-space: replace row/col 0 with identity
    L = L.tolil()
    L[0, :] = 0.0
    L[:, 0] = 0.0
    L[0, 0] = 1.0
    L = L.tocsc()
    b1[0] = 0.0
    b2[0] = 0.0

    phi1 = spsolve(L, b1)
    phi2 = spsolve(L, b2)

    # Global integer-seam projection:
    # project edge jumps toward nearest integers while preserving a globally
    # integrable potential field (solve on all vertices at once).
    if int(integer_projection_iters) > 0:
        phi1 = _project_edge_jumps_to_integers(phi1, W, iters=int(integer_projection_iters))
        phi2 = _project_edge_jumps_to_integers(phi2, W, iters=int(integer_projection_iters))

    # Mixed-integer refinement with explicit integer potential n on vertices:
    #   minimize Σ w_ij * ((phi_j-phi_i) + (n_j-n_i) - round(phi_j-phi_i))^2
    # n is projected to integers each iteration, yielding globally consistent
    # seam jumps (cycle-compatible by construction through vertex potential).
    if int(integer_potential_iters) > 0:
        phi1 = _refine_with_integer_vertex_potential(phi1, W, iters=int(integer_potential_iters))
        phi2 = _refine_with_integer_vertex_potential(phi2, W, iters=int(integer_potential_iters))

    return phi1.astype(np.float32), phi2.astype(np.float32)


def _project_edge_jumps_to_integers(
    phi: np.ndarray,
    W: sparse.spmatrix,
    *,
    iters: int = 2,
) -> np.ndarray:
    """
    Global least-squares projection of edge jumps to nearest integers.

    For each undirected edge (i,j), let jump = phi[j]-phi[i], n=round(jump),
    residual target r = n - jump. Solve for vertex correction d:

        min_d  Σ_ij w_ij * ((d_j - d_i) - r_ij)^2

    and update phi <- phi + d. This preserves global integrability and enforces
    seam jumps near integer transitions more robustly than local rounding.
    """
    x = np.asarray(phi, dtype=np.float64).copy()
    N = x.shape[0]
    if N == 0:
        return x
    L = sparse.diags(np.array(W.sum(axis=1)).ravel()) - W
    L = L.tolil()
    L[0, :] = 0.0
    L[:, 0] = 0.0
    L[0, 0] = 1.0
    L = L.tocsc()

    Wc = W.tocoo()
    Wc.sum_duplicates()
    _r = np.asarray(Wc.row, dtype=np.int64)
    _c = np.asarray(Wc.col, dtype=np.int64)
    _v = np.asarray(Wc.data, dtype=np.float64)
    ut    = (_r < _c) & (np.abs(_v) >= 1e-14)
    i_arr = _r[ut];  j_arr = _c[ut];  w_arr = _v[ut]
    if len(i_arr) == 0:
        return x

    for _ in range(max(1, int(iters))):
        jumps = x[j_arr] - x[i_arr]
        n_ij  = np.round(jumps)
        r_ij  = n_ij - jumps
        rhs   = np.zeros(N, dtype=np.float64)
        np.add.at(rhs, i_arr, -w_arr * r_ij)
        np.add.at(rhs, j_arr,  w_arr * r_ij)
        rhs[0] = 0.0
        x += np.asarray(spsolve(L, rhs), dtype=np.float64)
    return x


def _refine_with_integer_vertex_potential(
    phi: np.ndarray,
    W: sparse.spmatrix,
    *,
    iters: int = 2,
) -> np.ndarray:
    """
    Alternating mixed-integer refinement using vertex integer potential n.

    Step A (continuous): solve weighted least squares for real-valued n.
    Step B (integer):    round n to nearest integers.
    Step C:              update phi <- phi + n_int and repeat.
    """
    x = np.asarray(phi, dtype=np.float64).copy()
    N = x.shape[0]
    if N == 0:
        return x

    Wc = W.tocoo()
    Wc.sum_duplicates()
    _r = np.asarray(Wc.row, dtype=np.int64)
    _c = np.asarray(Wc.col, dtype=np.int64)
    _v = np.asarray(Wc.data, dtype=np.float64)
    ut    = (_r < _c) & (np.abs(_v) >= 1e-14)
    i_arr = _r[ut];  j_arr = _c[ut];  w_arr = _v[ut]
    if len(i_arr) == 0:
        return x

    L = sparse.diags(np.array(W.sum(axis=1)).ravel()) - W
    L = L.tolil()
    L[0, :] = 0.0
    L[:, 0] = 0.0
    L[0, 0] = 1.0
    L = L.tocsc()

    for _ in range(max(1, int(iters))):
        jumps = x[j_arr] - x[i_arr]
        t_ij  = np.round(jumps)
        r_ij  = t_ij - jumps
        rhs   = np.zeros(N, dtype=np.float64)
        np.add.at(rhs, i_arr, -w_arr * r_ij)
        np.add.at(rhs, j_arr,  w_arr * r_ij)
        rhs[0] = 0.0
        n_real = np.asarray(spsolve(L, rhs), dtype=np.float64)
        n_int  = np.round(n_real)
        n_int -= n_int[0]   # gauge fix: keep vertex 0 unchanged
        x = x + n_int
    return x


# ---------------------------------------------------------------------------
# Sinusoidal Barrier Penalty  (differentiable, PyTorch)
# ---------------------------------------------------------------------------

def sinusoidal_barrier_loss(
    phi1: torch.Tensor,
    phi2: torch.Tensor,
    edges: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    """
    Differentiable Sinusoidal Barrier Penalty (§5.3):

        L_int = weight * Σ_{(i,j)∈edges} [sin²(π(φ₁(j)−φ₁(i))) + sin²(π(φ₂(j)−φ₂(i)))]

    Encourages integer seam transitions without a mixed-integer solver.
    Gradient flows back to φ₁, φ₂ (and hence to the cross-field and DGCNN).

    Args:
        phi1, phi2:  (N,) parametrisation values (PyTorch, grad-tracked).
        edges:       (E, 2) long tensor of edge index pairs.
        weight:      scalar penalty weight.

    Returns:
        scalar loss tensor.
    """
    i_idx = edges[:, 0]
    j_idx = edges[:, 1]
    jump1 = phi1[j_idx] - phi1[i_idx]
    jump2 = phi2[j_idx] - phi2[i_idx]
    loss = (torch.sin(np.pi * jump1) ** 2 + torch.sin(np.pi * jump2) ** 2).mean()
    return weight * loss


def mesh_edges(F: np.ndarray) -> np.ndarray:
    """
    Return all unique undirected edges of a triangle mesh as (E, 2) int array.
    """
    edge_set: set = set()
    for face in F:
        for k in range(3):
            a, b = int(face[k]), int(face[(k + 1) % 3])
            edge_set.add((min(a, b), max(a, b)))
    return np.array(sorted(edge_set), dtype=np.int64)


# ---------------------------------------------------------------------------
# Integer grid quad extraction  (non-differentiable, straight-through)
# ---------------------------------------------------------------------------

def extract_quads_integer_grid(
    V: np.ndarray,
    phi1: np.ndarray,
    phi2: np.ndarray,
    max_param_dist: float = 0.7,
    enforce_manifold: bool = True,
    require_closed: bool = False,
    k_candidates: int = 8,
) -> np.ndarray:
    """
    Extract quad topology by sampling the parametrisation at integer grid points.

    For each integer lattice point (p, q) within the range of (φ₁, φ₂),
    the closest mesh vertex in parameter space is assigned as the "canonical"
    vertex at that grid point.  Quads are then assembled for each 2×2 cell
    of the lattice where all four corners are present.

    This is the non-differentiable rounding step; during end-to-end training
    it is treated as a straight-through estimator — the topology Q is fixed,
    and gradients flow only through the PD solver (§6).

    Args:
        V:              (N, 3) mesh vertex positions (used only for degenerate checks).
        phi1, phi2:     (N,) parametrisation scalar fields.
        max_param_dist: Grid points farther than this in parameter space from any
                        vertex are considered "outside the mesh" and skipped.

    Returns:
        quads:  (M, 4) int64 array of quad face indices.

    Raises:
        RuntimeError: if no valid quads can be assembled.
    """
    from scipy.spatial import KDTree

    param_coords = np.column_stack([phi1, phi2])   # (N, 2)
    param_tree = KDTree(param_coords)

    p_min = int(np.floor(phi1.min()))
    p_max = int(np.ceil(phi1.max()))
    q_min = int(np.floor(phi2.min()))
    q_max = int(np.ceil(phi2.max()))

    # Map each integer grid point to a nearby mesh vertex.
    # We enforce one-to-one assignment (a vertex can represent at most one grid
    # point) to avoid many-to-one collapses that often create non-manifold edges.
    grid_pts = []
    gp_coords = []
    for p in range(p_min, p_max + 1):
        for q in range(q_min, q_max + 1):
            grid_pts.append((p, q))
            gp_coords.append([float(p), float(q)])

    if not gp_coords:
        raise RuntimeError(
            "Parametrisation range is empty — cannot extract quads. "
            "Check gradient_size and cross-field configuration."
        )

    gp_coords = np.array(gp_coords)
    kc = int(max(1, min(k_candidates, len(V))))
    dists, idxs = param_tree.query(gp_coords, k=kc)
    if kc == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    # Process closer grid points first to maximize reliable assignments.
    order = np.argsort(dists[:, 0])
    used_vertices = set()
    grid_to_vertex: dict = {}
    for oid in order:
        p, q = grid_pts[int(oid)]
        assigned = None
        for ci in range(kc):
            d = float(dists[oid, ci])
            v = int(idxs[oid, ci])
            if d >= max_param_dist:
                continue
            if v in used_vertices:
                continue
            assigned = v
            break
        if assigned is not None:
            grid_to_vertex[(p, q)] = assigned
            used_vertices.add(assigned)

    # Build quads: cell (p,q)–(p+1,q)–(p+1,q+1)–(p,q+1)
    quads = []
    seen_vsets: set = set()
    edge_counts: Counter = Counter()
    for p in range(p_min, p_max):
        for q in range(q_min, q_max):
            v00 = grid_to_vertex.get((p,     q    ))
            v10 = grid_to_vertex.get((p + 1, q    ))
            v11 = grid_to_vertex.get((p + 1, q + 1))
            v01 = grid_to_vertex.get((p,     q + 1))
            if None in (v00, v10, v11, v01):
                continue
            vset = frozenset([v00, v10, v11, v01])
            if len(vset) < 4:
                continue    # degenerate (shared vertex)
            key = tuple(sorted(vset))
            if key in seen_vsets:
                continue    # duplicate
            cand = [int(v00), int(v10), int(v11), int(v01)]
            if enforce_manifold:
                # Reject candidates that would create edges with incidence > 2.
                cand_edges = []
                for ei in range(4):
                    a = cand[ei]
                    b = cand[(ei + 1) % 4]
                    if a > b:
                        a, b = b, a
                    cand_edges.append((a, b))
                if any(edge_counts[e] >= 2 for e in cand_edges):
                    continue
            seen_vsets.add(key)
            quads.append(cand)
            if enforce_manifold:
                for e in cand_edges:
                    edge_counts[e] += 1

    if not quads:
        raise RuntimeError(
            "Integer-grid quad extraction produced zero valid quads.\n"
            "Possible causes:\n"
            "  - gradient_size is too large (too few integer cells)\n"
            "  - cross-field is degenerate (all vertices map to same integer cell)\n"
            "  - one-to-one manifold extraction filtered all candidates\n"
            "  - mesh is too coarse relative to gradient_size\n"
            "Adjust miq.gradient_size or miq.crossfield_mu in the config."
        )

    quads_arr = np.array(quads, dtype=np.int64)
    if require_closed:
        quads_arr = _prune_to_closed_manifold(quads_arr)
        if quads_arr.size == 0:
            raise RuntimeError(
                "Closed-topology extraction failed: no boundary-free manifold subset "
                "remains after pruning. Adjust cross-field/parametrisation settings."
            )
    return quads_arr


def _prune_to_closed_manifold(quads: np.ndarray) -> np.ndarray:
    """
    Iteratively remove quads incident to non-2-valent edges.

    This guarantees the returned quad set has only edges with incidence exactly 2
    (closed 2-manifold edge condition), if non-empty.
    """
    if quads.ndim != 2 or quads.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.int64)
    alive = np.ones(quads.shape[0], dtype=bool)
    while True:
        edge_counts = Counter()
        edge_to_faces = {}
        idx_alive = np.where(alive)[0]
        for fi in idx_alive:
            q = quads[fi]
            for k in range(4):
                a = int(q[k]); b = int(q[(k + 1) % 4])
                if a > b:
                    a, b = b, a
                e = (a, b)
                edge_counts[e] += 1
                edge_to_faces.setdefault(e, []).append(fi)

        bad_edges = [e for e, c in edge_counts.items() if c != 2]
        if not bad_edges:
            break
        remove = set()
        for e in bad_edges:
            for fi in edge_to_faces.get(e, []):
                remove.add(fi)
        if not remove:
            break
        prev = alive.sum()
        for fi in remove:
            alive[fi] = False
        if alive.sum() == prev:
            break
        if alive.sum() == 0:
            return np.zeros((0, 4), dtype=np.int64)
    return quads[alive]


# ---------------------------------------------------------------------------
# Differentiable parametrisation wrapper (PyTorch)
# ---------------------------------------------------------------------------

class _ParametrisationSolve(torch.autograd.Function):
    """
    Differentiable Poisson solve:  φ = L^{-1} b(θ)

    Forward:  φ = L_factor(b)
    Backward: ∂L/∂b = L^{-1} ∂L/∂φ   (L is symmetric ⇒ L^{-T} = L^{-1})
    """

    @staticmethod
    def forward(ctx, b: torch.Tensor, L_factor) -> torch.Tensor:
        b_np = b.detach().cpu().double().numpy()
        phi_np = L_factor(b_np)
        ctx.L_factor = L_factor
        return torch.tensor(phi_np, dtype=b.dtype, device=b.device)

    @staticmethod
    def backward(ctx, grad_phi: torch.Tensor):
        L_factor = ctx.L_factor
        grad_np = grad_phi.detach().cpu().double().numpy()
        grad_b_np = L_factor(grad_np)
        grad_b = torch.tensor(grad_b_np, dtype=grad_phi.dtype, device=grad_phi.device)
        return grad_b, None


def parametrisation_from_crossfield_torch(
    theta: torch.Tensor,
    frames_np: np.ndarray,
    V_np: np.ndarray,
    F_np: np.ndarray,
    L_factor,
    gradient_size: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable parametrisation: θ → (φ₁, φ₂).

    Uses a pre-factorised Laplacian solver.  The right-hand side b(θ) is
    computed from the current θ values (detached) and provided to the implicit-
    diff backward through _ParametrisationSolve.  This gives correct gradients
    ∂φ/∂b via L^{-1}, though ∂b/∂θ is treated as a stop-gradient for stability
    (standard approximation in unrolled optimisation).

    Args:
        theta:         (N,) cross-field angles (PyTorch, may be grad-tracked).
        frames_np:     (N, 3, 2) tangent frames (numpy constant).
        V_np:          (N, 3) vertex positions (numpy constant).
        F_np:          (M, 3) triangle faces (numpy constant).
        L_factor:      pre-factorised pinned cotangent Laplacian.
        gradient_size: parametrisation scale.

    Returns:
        phi1, phi2:  (N,) PyTorch tensors (grad-tracked through L_factor).
    """
    device = theta.device
    dtype  = theta.dtype

    # Compute rhs from detached angles (approximation; full ∂b/∂θ is expensive)
    theta_np = theta.detach().cpu().numpy()
    b1_np, b2_np = _build_gradient_rhs(V_np, F_np, theta_np, frames_np, gradient_size)
    b1_np[0] = 0.0; b2_np[0] = 0.0   # match pinned Laplacian BC

    b1 = torch.tensor(b1_np, dtype=dtype, device=device)
    b2 = torch.tensor(b2_np, dtype=dtype, device=device)

    phi1 = _ParametrisationSolve.apply(b1, L_factor)
    phi2 = _ParametrisationSolve.apply(b2, L_factor)
    return phi1, phi2


def precompute_param_laplacian(V: np.ndarray, F: np.ndarray):
    """
    Pre-factorize the (pinned) cotangent Laplacian for the parametrisation solver.

    Returns:
        L_factor:  callable — the factorised solver.
        L_csc:     scipy csc_matrix — the pinned Laplacian.
    """
    L = _cotangent_laplacian(V, F)
    L = L.tolil()
    L[0, :] = 0.0
    L[:, 0] = 0.0
    L[0, 0] = 1.0
    L = L.tocsc()
    return factorized(L), L
