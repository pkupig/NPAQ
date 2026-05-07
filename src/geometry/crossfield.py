"""
Differentiable 4-RoSy cross-field computation via Ginzburg–Landau relaxation.

Implements §5.1: solve  (L_conn + μI) u = μ u*
where u = e^{4iθ} encodes the local orientation angle θ with 90° symmetry.

The connection Laplacian L_conn discretises the covariant derivative of complex
numbers on the mesh, transporting frames between adjacent vertices via the
parallel-transport angle (Crane et al. 2010).

Gradient flow:  u  differentiable w.r.t.  u* = e^{4iθ*}  (guidance)
via a custom autograd.Function that calls the pre-factorised solver in both
the forward and backward passes.
"""

from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Geometry helpers (numpy)
# ---------------------------------------------------------------------------

def compute_cotangent_weights(V: np.ndarray, F: np.ndarray) -> sparse.csr_matrix:
    """
    Cotangent Laplacian weight matrix W for a triangle mesh.

    W[i,j] = 0.5 * (cot α_ij + cot β_ij) for each interior edge (i,j),
    where α_ij and β_ij are the angles opposite the edge in its two incident
    triangles.  W[i,i] is not filled; use L = diag(W@1) - W for the Laplacian.

    Returns:
        sparse csr_matrix (N, N) with positive off-diagonal cotangent weights.
    """
    N = len(V)
    rows, cols, data = [], [], []

    for k in range(3):
        # Vertex l is opposite edge (i,j); angle at l contributes cot weight
        i = F[:, k]
        j = F[:, (k + 1) % 3]
        l = F[:, (k + 2) % 3]

        vi = V[i] - V[l]   # (F, 3)
        vj = V[j] - V[l]   # (F, 3)

        dot_ij  = np.einsum('fd,fd->f', vi, vj)
        cross_n = np.linalg.norm(np.cross(vi, vj), axis=1)  # (F,)
        cot_w   = 0.5 * dot_ij / (cross_n + 1e-10)
        cot_w   = np.clip(cot_w, -10.0, 10.0)

        rows += i.tolist() + j.tolist()
        cols += j.tolist() + i.tolist()
        data += cot_w.tolist() + cot_w.tolist()

    W = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
    W.sum_duplicates()
    return W


def compute_vertex_frames(
    V: np.ndarray,
    F: np.ndarray,
    normals: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute per-vertex tangent frames (e1, e2).

    Returns:
        frames  (N, 3, 2)  where frames[i, :, 0] = e1_i, frames[i, :, 1] = e2_i.
    """
    N = len(V)

    if normals is None:
        v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        fn /= (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-10)
        vn = np.zeros((N, 3))
        np.add.at(vn, F[:, 0], fn)
        np.add.at(vn, F[:, 1], fn)
        np.add.at(vn, F[:, 2], fn)
        vn /= (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-10)
    else:
        vn = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-10)

    # Build a deterministic, numerically stable tangent frame using the
    # Duff et al. (2017) "Building an Orthonormal Basis, Revisited" formula.
    #
    # The old helper-axis approach (cross with X or Y, switching at |n_x|=0.9)
    # creates an artificial switch line on the mesh.  Vertices straddling that line
    # receive frames that differ by ~45°, inflating the parallel-transport
    # angles in build_connection_laplacian and producing spurious singularities
    # that force excessive mesh cuts → UV fragmentation → too few quads.
    #
    # The Duff et al. formula avoids the helper-axis branch singularity; a
    # global smooth tangent frame still cannot exist on arbitrary closed meshes.
    nx, ny, nz = vn[:, 0], vn[:, 1], vn[:, 2]
    sign = np.where(nz >= 0.0, 1.0, -1.0)          # never zero
    a    = -1.0 / (sign + nz)                        # denominator ≥ 1 in abs
    b    = nx * ny * a
    e1   = np.stack([1.0 + sign * nx**2 * a,  sign * b,  -sign * nx], axis=1)
    e2   = np.stack([b,  sign + ny**2 * a,  -ny], axis=1)
    # Already unit-length by construction; normalize for float safety.
    e1 /= (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-12)
    e2 /= (np.linalg.norm(e2, axis=1, keepdims=True) + 1e-12)

    return np.stack([e1, e2], axis=2)   # (N, 3, 2)


def _parallel_transport_angle(
    e1_i: np.ndarray,
    e2_i: np.ndarray,
    e1_j: np.ndarray,
    e2_j: np.ndarray,
    n_j: np.ndarray,
) -> float:
    """
    Angle r_ij: rotate frame at j to align with parallel-transported frame from i.

    Parallel-transports e1_i to the tangent plane at j (project out n_j),
    then measures the angle to e1_j in the frame (e1_j, e2_j).
    """
    transported = e1_i - np.dot(e1_i, n_j) * n_j
    t_norm = np.linalg.norm(transported)
    if t_norm < 1e-10:
        return 0.0
    transported /= t_norm
    cos_r = np.dot(transported, e1_j)
    sin_r = np.dot(transported, e2_j)
    return float(np.arctan2(sin_r, cos_r))


def build_connection_laplacian(
    V: np.ndarray,
    F: np.ndarray,
    frames: np.ndarray,
) -> Tuple[sparse.csc_matrix, sparse.csc_matrix]:
    """
    Build the 4-RoSy connection Laplacian as two real sparse matrices:

        L_conn[i,j]  = -w_ij * exp(4i r_ij)
                     = -w_ij * cos(4 r_ij)  +  i * (-w_ij * sin(4 r_ij))
        L_conn[i,i]  = Σ_j w_ij

    L_real = real part,  L_imag = imaginary part.

    The full (2N × 2N) real block system is:
        A = [[L_real + μI,  -L_imag],
             [L_imag,        L_real + μI]]

    Returns:
        L_real, L_imag — symmetric and antisymmetric sparse matrices (N, N).
    """
    N = len(V)
    W = compute_cotangent_weights(V, F)

    # Per-vertex normals (derived from frames)
    e1 = frames[:, :, 0]   # (N, 3)
    e2 = frames[:, :, 1]   # (N, 3)
    normals = np.cross(e1, e2)
    normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-10)

    # Extract unique undirected edges with non-trivial cotangent weights.
    # Use COO upper-triangle (a < b) — fully vectorised, no Python edge loop.
    W_coo = W.tocoo()
    W_coo.sum_duplicates()
    _row = np.asarray(W_coo.row, dtype=np.int64)
    _col = np.asarray(W_coo.col, dtype=np.int64)
    _dat = np.asarray(W_coo.data, dtype=np.float64)
    ut    = (_row < _col) & (np.abs(_dat) >= 1e-14)
    a_arr = _row[ut];  b_arr = _col[ut];  w_ab = _dat[ut]   # (E,)

    # Vectorised parallel-transport angles  r_{a→b}
    # transported = e1[a] − (e1[a]·n[b]) n[b],  then measure angle to e1[b]
    e1_a = e1[a_arr]        # (E, 3)
    e1_b = e1[b_arr]        # (E, 3)
    e2_b = e2[b_arr]        # (E, 3)
    n_b  = normals[b_arr]   # (E, 3)

    transported = e1_a - np.einsum('ei,ei->e', e1_a, n_b)[:, None] * n_b
    t_norm = np.linalg.norm(transported, axis=1, keepdims=True)
    transported /= np.maximum(t_norm, 1e-10)

    cos_r = np.einsum('ei,ei->e', transported, e1_b)   # (E,)
    sin_r = np.einsum('ei,ei->e', transported, e2_b)   # (E,)
    r_ab  = np.arctan2(sin_r, cos_r)                   # (E,)

    angle_ab =  4.0 * r_ab    # 4-RoSy transport a→b
    angle_ba = -angle_ab       # antisymmetric by connection 1-form definition

    c_ab = np.cos(angle_ab);  s_ab = np.sin(angle_ab)
    c_ba = np.cos(angle_ba);  s_ba = np.sin(angle_ba)

    # Build sparse matrices: off-diagonal entries for both directions
    rows_r = np.concatenate([a_arr, b_arr])
    cols_r = np.concatenate([b_arr, a_arr])
    data_r = np.concatenate([-w_ab * c_ab, -w_ab * c_ba])

    rows_i = np.concatenate([a_arr, b_arr])
    cols_i = np.concatenate([b_arr, a_arr])
    data_i = np.concatenate([-w_ab * s_ab, -w_ab * s_ba])

    diag = np.zeros(N)
    np.add.at(diag, a_arr, w_ab)
    np.add.at(diag, b_arr, w_ab)

    L_real = sparse.csr_matrix((data_r, (rows_r, cols_r)), shape=(N, N))
    L_real = L_real + sparse.diags(diag)

    L_imag = sparse.csr_matrix((data_i, (rows_i, cols_i)), shape=(N, N))

    return L_real.tocsc(), L_imag.tocsc()


def guidance_field_from_metric(
    M_vert: np.ndarray,
    frames: np.ndarray,
    isotropy_eps: float = 1e-6,
) -> np.ndarray:
    """
    Compute guidance u* = exp(4i θ*) from the per-vertex 2×2 metric tensors.

    θ* = angle of the principal direction (largest-eigenvalue eigenvector)
         of M_i in the local 2D frame.

    Args:
        M_vert:  (N, 2, 2) metric tensors in the LCF 2D frame.
        frames:  (N, 3, 2) tangent frames (unused here; kept for API symmetry).

    Returns:
        u_star   (N,) complex64 array with |u_star[i]| = 1.
    """
    eigvals, eigvecs = np.linalg.eigh(M_vert)   # (N,2), (N,2,2)
    # Largest eigenvalue → last column
    d1 = eigvecs[:, :, -1]   # (N, 2)  principal direction in 2D LCF
    theta = np.arctan2(d1[:, 1], d1[:, 0])      # (N,)
    u_star = np.exp(4j * theta).astype(np.complex64)

    # Near-isotropic tensors have undefined principal direction: eigh may return
    # numerically arbitrary eigenvectors that inject random guidance phases.
    # Clamp such vertices to neutral guidance (theta=0 -> u*=1) to avoid
    # artificial symmetry breaking on near-isotropic regions (e.g. torus tests).
    if float(isotropy_eps) > 0.0:
        aniso = anisotropy_degree_from_metric(M_vert)
        iso_mask = aniso <= float(isotropy_eps)
        if np.any(iso_mask):
            u_star = np.asarray(u_star, dtype=np.complex64).copy()
            u_star[iso_mask] = np.complex64(1.0 + 0.0j)
    return u_star


def anisotropy_degree_from_metric(M_vert: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Per-vertex anisotropy degree in [0,1]:

        a = (lambda_max - lambda_min) / (lambda_max + lambda_min + eps)

    Near-umbilic regions (e.g. sphere) have a≈0 and principal direction is
    numerically unstable, so guidance should be smoothed and down-weighted.
    """
    eigvals = np.linalg.eigvalsh(M_vert)
    lam_min = np.maximum(eigvals[:, 0], 0.0)
    lam_max = np.maximum(eigvals[:, 1], 0.0)
    return (lam_max - lam_min) / (lam_max + lam_min + eps)


def _harmonic_fill_complex(
    W: sparse.spmatrix,
    u_star: np.ndarray,
    anchor_mask: np.ndarray,
) -> np.ndarray:
    """
    Fill non-anchor vertices by harmonic interpolation on the mesh graph.
    """
    N = W.shape[0]
    if anchor_mask.sum() == 0 or anchor_mask.sum() == N:
        return u_star

    L = sparse.diags(np.array(W.sum(axis=1)).ravel()) - W
    anchors = np.where(anchor_mask)[0]
    unknown = np.where(~anchor_mask)[0]
    if len(unknown) == 0:
        return u_star

    L_uu = L[unknown][:, unknown].tocsc()
    L_ua = L[unknown][:, anchors].tocsc()

    rhs_r = -L_ua @ u_star[anchors].real
    rhs_i = -L_ua @ u_star[anchors].imag
    x_r = sparse.linalg.spsolve(L_uu, rhs_r)
    x_i = sparse.linalg.spsolve(L_uu, rhs_i)

    out = u_star.astype(np.complex128).copy()
    out[unknown] = x_r + 1j * x_i
    out /= (np.abs(out) + 1e-12)
    return out.astype(np.complex64)


def _build_singularity_vortex_override(
    V: np.ndarray,
    frames: np.ndarray,
    singularity_mask: np.ndarray,
    singularity_indices: Optional[np.ndarray],
    support_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a local vortex-style guidance field around prescribed singularities.

    For a quarter-turn singularity with integer charge q in the complex 4-RoSy
    field, the complex phase should wind by q * 2π around the centre. We
    approximate that locally by:

        u_s(x) = exp(i * q * alpha(x))

    where alpha is the polar angle of x in the tangent frame at the singularity.
    """
    pts = np.asarray(V, dtype=np.float64)
    mask = np.asarray(singularity_mask, dtype=bool).reshape(-1)
    N = len(pts)
    if mask.shape[0] != N or not np.any(mask) or support_radius <= 0.0:
        return np.zeros(N, dtype=np.complex64), np.zeros(N, dtype=np.float64)

    if singularity_indices is None:
        q_units = np.ones(N, dtype=np.float64)
    else:
        q_units = np.asarray(singularity_indices, dtype=np.float64).reshape(-1)
        if q_units.shape[0] != N:
            raise ValueError(
                f"singularity_indices length mismatch: {q_units.shape[0]} != {N}"
            )

    accum = np.zeros(N, dtype=np.complex128)
    weights = np.zeros(N, dtype=np.float64)
    centers = np.where(mask)[0]
    radius = float(support_radius)
    eps = max(radius * 1e-3, 1e-8)
    for si in centers.tolist():
        q = float(q_units[si])
        if abs(q) < 1e-8:
            continue
        c = pts[si]
        disp = pts - c[None, :]
        dist = np.linalg.norm(disp, axis=1)
        active = (dist > eps) & (dist <= radius)
        if not np.any(active):
            continue
        e1 = frames[si, :, 0]
        e2 = frames[si, :, 1]
        local = disp[active]
        x = local @ e1
        y = local @ e2
        alpha = np.arctan2(y, x)
        u_loc = np.exp(1j * q * alpha)
        w = np.square(np.clip(1.0 - dist[active] / radius, 0.0, 1.0))
        accum[active] += w * u_loc
        weights[active] += w

    valid = weights > 1e-12
    out = np.zeros(N, dtype=np.complex64)
    if np.any(valid):
        blended = accum[valid] / weights[valid]
        out[valid] = (blended / (np.abs(blended) + 1e-12)).astype(np.complex64)
    return out, weights


def _mean_edge_length(V: np.ndarray, F: np.ndarray) -> float:
    edges = set()
    for f in F:
        for k in range(3):
            a = int(f[k])
            b = int(f[(k + 1) % 3])
            if a != b:
                edges.add((min(a, b), max(a, b)))
    if not edges:
        return 1.0
    e = np.asarray(sorted(edges), dtype=np.int64)
    return float(np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1).mean() + 1e-12)


def _vertex_adjacency(F: np.ndarray, n_vertices: int) -> list[list[int]]:
    adj = [set() for _ in range(n_vertices)]
    for tri in np.asarray(F, dtype=np.int64):
        for k in range(3):
            a = int(tri[k])
            b = int(tri[(k + 1) % 3])
            if a != b:
                adj[a].add(b)
                adj[b].add(a)
    return [sorted(v) for v in adj]


def _smooth_u_on_vertex_mask(
    u: np.ndarray,
    adj: list[list[int]],
    vertex_mask: np.ndarray,
    n_iter: int = 4,
) -> np.ndarray:
    out = np.asarray(u, dtype=np.complex128).copy()
    ids = np.where(vertex_mask)[0]
    if len(ids) == 0:
        return out.astype(np.complex64)
    for _ in range(max(1, int(n_iter))):
        prev = out.copy()
        for vi in ids:
            nbrs = adj[int(vi)]
            if not nbrs:
                continue
            avg = prev[nbrs].mean()
            if abs(avg) > 1e-12:
                out[vi] = avg / abs(avg)
        out /= (np.abs(out) + 1e-12)
    return out.astype(np.complex64)


def _expand_vertex_mask(
    seed_mask: np.ndarray,
    adj: list[list[int]],
    rounds: int = 1,
) -> np.ndarray:
    mask = np.asarray(seed_mask, dtype=bool).copy()
    if mask.ndim != 1:
        return mask
    active = np.where(mask)[0].tolist()
    for _ in range(max(0, int(rounds))):
        if not active:
            break
        new_ids = set(active)
        for vi in active:
            new_ids.update(int(nb) for nb in adj[int(vi)])
        active = [vi for vi in new_ids if not mask[vi]]
        if not active:
            break
        mask[active] = True
    return mask


def repair_flow_violations(
    V: np.ndarray,
    F: np.ndarray,
    u: np.ndarray,
    frames: np.ndarray,
    protected_mask: Optional[np.ndarray] = None,
    *,
    max_rounds: int = 2,
    expand_rounds: int = 1,
    smoothing_iters: int = 6,
) -> tuple[np.ndarray, dict]:
    """
    Reduce local flow-conservation violations by smoothing only around violating
    vertices and accepting the change only when the diagnostic improves.
    """
    u_best = np.asarray(u, dtype=np.complex64).copy()
    adj = _vertex_adjacency(F, len(V))
    protected = None if protected_mask is None else np.asarray(protected_mask, dtype=bool).reshape(-1)
    best_flow = verify_flow_conservation(V, F, u_best, frames)
    best_ph = verify_poincare_hopf(V, F, u_best, frames=frames)
    info = {
        'rounds': 0,
        'improved_rounds': 0,
        'num_violations_before': int(best_flow.get('num_violations', 0)),
        'num_violations_after': int(best_flow.get('num_violations', 0)),
        'accepted': False,
    }
    if best_flow.get('conserved', False):
        return u_best, info

    best_holo = verify_holonomy_compatibility(V, F, u_best, frames)
    for _ in range(max(1, int(max_rounds))):
        viol_ids = np.asarray(best_flow.get('violation_ids', []), dtype=np.int64)
        if viol_ids.size == 0:
            break
        mask = np.zeros(len(V), dtype=bool)
        mask[viol_ids] = True
        mask = _expand_vertex_mask(mask, adj, rounds=expand_rounds)
        if protected is not None and protected.shape[0] == len(V):
            mask &= ~protected
            if not np.any(mask):
                break
        u_try = _smooth_u_on_vertex_mask(u_best, adj, mask, n_iter=smoothing_iters)
        flow_try = verify_flow_conservation(V, F, u_try, frames)
        holo_try = verify_holonomy_compatibility(V, F, u_try, frames)
        ph_try = verify_poincare_hopf(V, F, u_try, frames=frames)
        improved = (
            int(flow_try.get('num_violations', 0)) < int(best_flow.get('num_violations', 0))
            or (
                int(flow_try.get('num_violations', 0)) == int(best_flow.get('num_violations', 0))
                and (
                    int(holo_try.get('cotree_violations', 0)) < int(best_holo.get('cotree_violations', 0))
                    or abs(float(ph_try.get('deficit', 0.0))) < abs(float(best_ph.get('deficit', 0.0)))
                )
            )
        )
        info['rounds'] += 1
        if not improved:
            break
        u_best = u_try
        best_flow = flow_try
        best_holo = holo_try
        best_ph = ph_try
        info['improved_rounds'] += 1
        info['accepted'] = True
        if best_flow.get('conserved', False):
            break

    info['num_violations_after'] = int(best_flow.get('num_violations', 0))
    return u_best, info


def repair_high_order_singularities(
    V: np.ndarray,
    F: np.ndarray,
    u: np.ndarray,
    frames: np.ndarray,
    protected_mask: Optional[np.ndarray] = None,
    *,
    max_rounds: int = 2,
    expand_rounds: int = 1,
    smoothing_iters: int = 4,
    tol: float = 0.125,
) -> tuple[np.ndarray, dict]:
    """
    Reduce local high-order singularities (|index| > 1/4) by smoothing only in
    their local support and accepting the change only when the high-order count
    decreases.
    """
    u_best = np.asarray(u, dtype=np.complex64).copy()
    tri = np.asarray(F, dtype=np.int64)
    adj = _vertex_adjacency(F, len(V))
    protected = None if protected_mask is None else np.asarray(protected_mask, dtype=bool).reshape(-1)
    best_holo = verify_holonomy_compatibility(V, F, u_best, frames, tol=tol)
    best_flow = verify_flow_conservation(V, F, u_best, frames, tol=tol)
    best_ph = verify_poincare_hopf(V, F, u_best, frames=frames, tol=tol)
    info = {
        'rounds': 0,
        'improved_rounds': 0,
        'high_order_before': int(best_holo.get('high_order_count', 0)),
        'high_order_after': int(best_holo.get('high_order_count', 0)),
        'accepted': False,
    }
    if int(best_holo.get('high_order_count', 0)) <= 0:
        return u_best, info

    for _ in range(max(1, int(max_rounds))):
        sing = detect_singularities_from_crossfield(V, F, u_best, frames=frames, tol=tol)
        idx = np.asarray(sing.get('indices', []), dtype=np.float32)
        face_ids = np.asarray(sing.get('face_ids', []), dtype=np.int64)
        high = np.where(np.abs(np.abs(idx) - 0.25) > 0.1)[0]
        if high.size == 0:
            break

        face_mask = np.zeros(len(tri), dtype=bool)
        face_mask[face_ids[high]] = True
        active_vertices = np.zeros(len(V), dtype=bool)
        active_vertices[np.unique(tri[face_mask].reshape(-1))] = True
        active_vertices = _expand_vertex_mask(active_vertices, adj, rounds=expand_rounds)
        if protected is not None and protected.shape[0] == len(V):
            active_vertices &= ~protected
            if not np.any(active_vertices):
                break
        u_try = _smooth_u_on_vertex_mask(u_best, adj, active_vertices, n_iter=smoothing_iters)
        holo_try = verify_holonomy_compatibility(V, F, u_try, frames, tol=tol)
        flow_try = verify_flow_conservation(V, F, u_try, frames, tol=tol)
        ph_try = verify_poincare_hopf(V, F, u_try, frames=frames, tol=tol)
        improved = (
            int(holo_try.get('high_order_count', 0)) < int(best_holo.get('high_order_count', 0))
            or (
                int(holo_try.get('high_order_count', 0)) == int(best_holo.get('high_order_count', 0))
                and (
                    int(flow_try.get('num_violations', 0)) < int(best_flow.get('num_violations', 0))
                    or abs(float(ph_try.get('deficit', 0.0))) < abs(float(best_ph.get('deficit', 0.0)))
                )
            )
        )
        info['rounds'] += 1
        if not improved:
            break
        u_best = u_try
        best_holo = holo_try
        best_flow = flow_try
        best_ph = ph_try
        info['improved_rounds'] += 1
        info['accepted'] = True
        if int(best_holo.get('high_order_count', 0)) <= 0:
            break

    info['high_order_after'] = int(best_holo.get('high_order_count', 0))
    return u_best, info


def cancel_close_singularity_pairs(
    V: np.ndarray,
    F: np.ndarray,
    u: np.ndarray,
    frames: Optional[np.ndarray] = None,
    protected_mask: Optional[np.ndarray] = None,
    distance_threshold: Optional[float] = None,
    distance_ratio: float = 2.5,
    smoothing_iters: int = 4,
    max_rounds: int = 2,
    tol: float = 0.125,
) -> Tuple[np.ndarray, dict]:
    """
    Cancel nearby opposite-sign ±1/4 singularity pairs by local smoothing.

    This is a conservative post-process: only close positive/negative pairs of
    the same magnitude are touched, and only their local vertex neighbourhood is
    smoothed. It reduces spurious singularity pairs created by noisy guidance
    without changing the global field construction.
    """
    u_cur = np.asarray(u, dtype=np.complex64).copy()
    if F.ndim != 2 or F.shape[1] != 3 or len(F) == 0:
        return u_cur, {'pairs_cancelled': 0, 'rounds': 0}

    thr = float(distance_threshold) if distance_threshold is not None else (
        float(distance_ratio) * _mean_edge_length(V, F)
    )
    adj = _vertex_adjacency(F, len(V))
    tri = np.asarray(F, dtype=np.int64)
    total_pairs = 0
    rounds_done = 0
    best_sing = detect_singularities_from_crossfield(V, F, u_cur, frames=frames, tol=tol)
    best_units = int(best_sing.get('sum_vertex_units', 0))
    best_ph = verify_poincare_hopf(V, F, u_cur, frames=frames, tol=tol) if frames is not None else None
    protected = None if protected_mask is None else np.asarray(protected_mask, dtype=bool).reshape(-1)

    for _ in range(max(1, int(max_rounds))):
        sing = best_sing
        idx = sing['indices']
        if len(idx) == 0:
            break
        face_ids = sing['face_ids']
        pos = np.where(np.isclose(idx, 0.25, atol=tol))[0]
        neg = np.where(np.isclose(idx, -0.25, atol=tol))[0]
        if len(pos) == 0 or len(neg) == 0:
            break

        used_neg: set[int] = set()
        pairs: list[tuple[int, int]] = []
        for pi in pos:
            p = sing['positions'][pi]
            best = None
            best_d = np.inf
            for ni in neg:
                if int(ni) in used_neg:
                    continue
                d = float(np.linalg.norm(p - sing['positions'][ni]))
                if d < best_d:
                    best_d = d
                    best = int(ni)
            if best is not None and best_d <= thr:
                used_neg.add(best)
                pairs.append((int(pi), best))

        if not pairs:
            break

        face_mask = np.zeros(len(F), dtype=bool)
        for pi, ni in pairs:
            face_mask[int(face_ids[pi])] = True
            face_mask[int(face_ids[ni])] = True

        # Expand to one-ring neighbouring triangles so the cancellation has
        # enough support to smooth the local index pair away.
        active_vertices = np.zeros(len(V), dtype=bool)
        active_vertices[np.unique(tri[face_mask].reshape(-1))] = True
        expanded_face_mask = np.any(active_vertices[tri], axis=1)
        active_vertices[:] = False
        active_vertices[np.unique(tri[expanded_face_mask].reshape(-1))] = True
        if protected is not None and protected.shape[0] == len(V):
            active_vertices &= ~protected
            if not np.any(active_vertices):
                break

        u_try = _smooth_u_on_vertex_mask(
            u_cur, adj, active_vertices, n_iter=smoothing_iters
        )
        sing_try = detect_singularities_from_crossfield(V, F, u_try, frames=frames, tol=tol)
        units_try = int(sing_try.get('sum_vertex_units', 0))
        ph_try = verify_poincare_hopf(V, F, u_try, frames=frames, tol=tol) if frames is not None else None
        before_face_count = int(best_sing.get('num_singular_faces', 0))
        after_face_count = int(sing_try.get('num_singular_faces', 0))
        before_vertex_count = int(best_sing.get('num_singular_vertices', 0))
        after_vertex_count = int(sing_try.get('num_singular_vertices', 0))
        ph_ok = True
        if best_ph is not None and ph_try is not None:
            ph_ok = abs(float(ph_try.get('deficit', 0.0))) <= abs(float(best_ph.get('deficit', 0.0))) + 1e-9
        improved = (
            units_try == best_units
            and ph_ok
            and (
                after_face_count < before_face_count
                or after_vertex_count < before_vertex_count
            )
        )
        if not improved:
            break
        u_cur = u_try
        best_sing = sing_try
        if ph_try is not None:
            best_ph = ph_try
        total_pairs += len(pairs)
        rounds_done += 1

    return u_cur, {
        'pairs_cancelled': int(total_pairs),
        'rounds': int(rounds_done),
        'distance_threshold': float(thr),
    }


def _assemble_gl_system(
    L_real: sparse.spmatrix,
    L_imag: sparse.spmatrix,
    mu: Union[float, np.ndarray],
) -> sparse.csc_matrix:
    """
    Assemble the (2N × 2N) real block system for the Ginzburg–Landau equation:

        [[L_r + μI,  -L_i ],    [u_r]     [μ u*_r]
         [L_i,     L_r + μI]] * [u_i]  =  [μ u*_i]

    Returns the assembled matrix as a csc_matrix.
    """
    N = L_real.shape[0]
    if np.isscalar(mu):
        mu_diag = np.full(N, float(mu), dtype=np.float64)
    else:
        mu_diag = np.asarray(mu, dtype=np.float64).reshape(-1)
        if mu_diag.shape[0] != N:
            raise ValueError(f"mu vector length mismatch: {mu_diag.shape[0]} != {N}")
    mu_I = sparse.diags(mu_diag, format='csc')
    A_top = sparse.hstack([L_real + mu_I, -L_imag], format='csc')
    A_bot = sparse.hstack([L_imag, L_real + mu_I], format='csc')
    return sparse.vstack([A_top, A_bot], format='csc')


def solve_crossfield_gl(
    V: np.ndarray,
    F: np.ndarray,
    M_vert: np.ndarray,
    frames: np.ndarray,
    mu: float = 10.0,
    umbilic_smoothing: bool = True,
    anisotropy_eps: float = 0.03,
    mu_min_ratio: float = 0.05,
    guidance_confidence: Optional[np.ndarray] = None,
    guidance_override: Optional[np.ndarray] = None,
    guidance_override_weight: Optional[np.ndarray] = None,
    protected_mask: Optional[np.ndarray] = None,
    guidance_override_reliability_floor: float = 0.9,
    guidance_override_anchor_eps: float = 0.05,
    filter_singularities: bool = True,
    singularity_cancel_distance_ratio: float = 2.5,
    singularity_cancel_smoothing_iters: int = 4,
    singularity_cancel_max_rounds: int = 2,
    singularity_mask: Optional[np.ndarray] = None,
    singularity_indices: Optional[np.ndarray] = None,
    repair_high_order: bool = True,
    high_order_repair_max_rounds: int = 1,
    high_order_repair_expand_rounds: int = 1,
    high_order_repair_smoothing_iters: int = 3,
    repair_flow: bool = True,
    flow_repair_max_rounds: int = 1,
    flow_repair_expand_rounds: int = 0,
    flow_repair_smoothing_iters: int = 3,
) -> Tuple[np.ndarray, object]:
    """
    Solve the Ginzburg–Landau cross-field equation:

        (L_conn + μI) u = μ u*

    Args:
        V:                 (N, 3) mesh vertices.
        F:                 (M, 3) mesh triangles.
        M_vert:            (N, 2, 2) per-vertex metric tensors (LCF 2D).
        frames:            (N, 3, 2) per-vertex tangent frames.
        mu:                alignment weight (default 10.0).
        singularity_mask:  (N,) bool — vertices with prescribed singularities.
        singularity_indices: (N,) integer quarter-turn charges for prescribed
            singularities. Positive values encode +1/4, +1/2, ... singularities.

    Returns:
        u:          (N,) complex64 — per-vertex cross-field (|u| ≈ 1).
        solve_info: dict with 'L_real', 'L_imag', 'A_factorized', 'mu'
                    for use in the differentiable PyTorch wrapper.
    """
    N = len(V)

    L_real, L_imag = build_connection_laplacian(V, F, frames)
    # Use a small isotropy threshold so numerically isotropic tensors do not
    # introduce arbitrary principal directions via unstable eigenvectors.
    u_star = guidance_field_from_metric(
        M_vert, frames,
        isotropy_eps=max(1e-6, float(anisotropy_eps) * 0.25),
    )

    # Umbilic-aware guidance:
    # principal directions are unstable where anisotropy≈0, so use harmonic
    # interpolation from anisotropic anchors instead of raw noisy eigenvectors.
    aniso = anisotropy_degree_from_metric(M_vert)
    if guidance_confidence is None:
        conf = np.ones_like(aniso, dtype=np.float64)
    else:
        conf = np.asarray(guidance_confidence, dtype=np.float64).reshape(-1)
        if conf.shape[0] != N:
            raise ValueError(f"guidance_confidence length mismatch: {conf.shape[0]} != {N}")
        conf = np.clip(conf, 0.0, 1.0)

    override_active = np.zeros(N, dtype=bool)
    protected = None if protected_mask is None else np.asarray(protected_mask, dtype=bool).reshape(-1)
    if guidance_override is not None:
        u_override = np.asarray(guidance_override, dtype=np.complex128).reshape(-1)
        if u_override.shape[0] != N:
            raise ValueError(f"guidance_override length mismatch: {u_override.shape[0]} != {N}")
        if guidance_override_weight is None:
            ow = np.ones(N, dtype=np.float64)
        else:
            ow = np.asarray(guidance_override_weight, dtype=np.float64).reshape(-1)
            if ow.shape[0] != N:
                raise ValueError(f"guidance_override_weight length mismatch: {ow.shape[0]} != {N}")
            ow = np.clip(ow, 0.0, 1.0)
        blend = np.clip(ow, 0.0, 1.0)
        mixed = (1.0 - blend) * u_star.astype(np.complex128) + blend * u_override
        norm = np.abs(mixed)
        use_override = norm > 1e-12
        override_active = np.asarray(use_override & (blend >= float(guidance_override_anchor_eps)), dtype=bool)
        u_star = u_star.astype(np.complex128)
        u_star[use_override] = mixed[use_override] / norm[use_override]
        u_star = u_star.astype(np.complex64)
        if np.any(override_active):
            floor = float(np.clip(guidance_override_reliability_floor, 0.0, 1.0))
            conf = np.asarray(conf, dtype=np.float64).copy()
            conf[override_active] = np.maximum(conf[override_active], floor * blend[override_active])

    singularity_active = np.zeros(N, dtype=bool)
    singularity_support_radius = 0.0
    singularity_override_weight = np.zeros(N, dtype=np.float64)
    if singularity_mask is not None:
        singularity_mask = np.asarray(singularity_mask, dtype=bool).reshape(-1)
        if singularity_mask.shape[0] != N:
            raise ValueError(f"singularity_mask length mismatch: {singularity_mask.shape[0]} != {N}")
        if np.any(singularity_mask):
            singularity_support_radius = 2.5 * _mean_edge_length(V, F)
            sing_u, sing_w = _build_singularity_vortex_override(
                V=V,
                frames=frames,
                singularity_mask=singularity_mask,
                singularity_indices=singularity_indices,
                support_radius=singularity_support_radius,
            )
            singularity_override_weight = np.clip(sing_w, 0.0, 1.0)
            singularity_active = singularity_override_weight > 1e-6
            if np.any(singularity_active):
                mixed = (
                    (1.0 - singularity_override_weight) * u_star.astype(np.complex128)
                    + singularity_override_weight * sing_u.astype(np.complex128)
                )
                norm = np.abs(mixed)
                use_sing = singularity_active & (norm > 1e-12)
                u_star = u_star.astype(np.complex128)
                u_star[use_sing] = mixed[use_sing] / norm[use_sing]
                u_star = u_star.astype(np.complex64)
                conf = np.asarray(conf, dtype=np.float64).copy()
                conf[use_sing] = np.maximum(
                    conf[use_sing],
                    0.85 * singularity_override_weight[use_sing],
                )

    # Directional saliency controls how strongly the GL solve follows neural
    # guidance.  With a saliency head available, combine it with the current
    # metric anisotropy as two estimates of whether the principal direction is
    # meaningful.  The geometric mean preserves the q scale when both estimates
    # agree (sqrt(q*q)=q), while suppressing unreliable directions if either the
    # head or the metric itself says the region is near-isotropic.
    reliability = np.sqrt(np.clip(aniso * conf, 0.0, 1.0)) if guidance_confidence is not None else aniso
    anchor_mask = reliability >= float(anisotropy_eps)
    if np.any(override_active):
        floor = float(np.clip(guidance_override_reliability_floor, 0.0, 1.0))
        reliability = np.asarray(reliability, dtype=np.float64).copy()
        reliability[override_active] = np.maximum(
            reliability[override_active],
            floor * blend[override_active],
        )
        anchor_mask = np.asarray(anchor_mask | override_active, dtype=bool)
    if np.any(singularity_active):
        reliability = np.asarray(reliability, dtype=np.float64).copy()
        reliability[singularity_active] = np.maximum(
            reliability[singularity_active],
            0.8 * singularity_override_weight[singularity_active],
        )
        anchor_mask = np.asarray(anchor_mask | singularity_active, dtype=bool)
    # Fully isotropic case: there is no trustworthy directional anchor anywhere.
    # Falling back to raw eigenvector guidance would inject random phases.
    # Use a neutral constant guidance field and keep reliability at zero.
    if not np.any(anchor_mask):
        u_star = np.ones(N, dtype=np.complex64)
        reliability = np.zeros(N, dtype=np.float64)

    if umbilic_smoothing and np.any(anchor_mask):
        W = compute_cotangent_weights(V, F)
        u_star = _harmonic_fill_complex(W, u_star, anchor_mask)

    # Spatially varying alignment weight: weak guidance near umbilics, full
    # guidance on anisotropic regions.
    mu_floor = max(0.0, float(mu_min_ratio)) * float(mu)
    mu_vec = mu_floor + (float(mu) - mu_floor) * reliability

    A = _assemble_gl_system(L_real, L_imag, mu_vec)

    b = np.zeros(2 * N)
    b[:N]  = mu_vec * u_star.real.astype(np.float64)
    b[N:]  = mu_vec * u_star.imag.astype(np.float64)

    # Prefactorize once (used by differentiable wrapper too)
    A_factor = factorized(A.tocsc())
    sol = A_factor(b)

    u_r, u_i = sol[:N], sol[N:]
    u = (u_r + 1j * u_i).astype(np.complex64)

    # Normalise to unit circle
    u /= (np.abs(u) + 1e-10)

    cancel_info = {
        'pairs_cancelled': 0,
        'rounds': 0,
        'distance_threshold': 0.0,
    }
    if filter_singularities:
        u, cancel_info = cancel_close_singularity_pairs(
            V=V,
            F=F,
            u=u,
            frames=frames,
            protected_mask=protected,
            distance_ratio=singularity_cancel_distance_ratio,
            smoothing_iters=singularity_cancel_smoothing_iters,
            max_rounds=singularity_cancel_max_rounds,
        )

    import warnings

    flow_repair_info = {
        'rounds': 0,
        'improved_rounds': 0,
        'num_violations_before': 0,
        'num_violations_after': 0,
        'accepted': False,
    }
    high_order_repair_info = {
        'rounds': 0,
        'improved_rounds': 0,
        'high_order_before': 0,
        'high_order_after': 0,
        'accepted': False,
    }

    if repair_high_order:
        u_repaired, high_order_repair_info = repair_high_order_singularities(
            V=V,
            F=F,
            u=u,
            frames=frames,
            protected_mask=protected,
            max_rounds=high_order_repair_max_rounds,
            expand_rounds=high_order_repair_expand_rounds,
            smoothing_iters=high_order_repair_smoothing_iters,
        )
        if high_order_repair_info.get('accepted', False):
            u = u_repaired

    # Poincaré–Hopf check: Σ index(sᵢ) must equal χ(S) for quadrangulability.
    ph = verify_poincare_hopf(V, F, u, frames=frames)
    if not ph['satisfied']:
        warnings.warn(
            f"Poincaré–Hopf violated after GL solve: Σidx={ph['sum_index']:.2f}, "
            f"χ={ph['chi']}. {ph['recommendation']}",
            stacklevel=2,
        )

    # Holonomy compatibility check: high-order singularities + co-tree holonomies.
    holo = verify_holonomy_compatibility(V, F, u, frames)
    if not holo['compatible']:
        warnings.warn(
            f"Holonomy incompatibility after GL solve: {holo['recommendation']}",
            stacklevel=2,
        )

    # Flow conservation check: QuadriFlow-style T-junction detection.
    flow = verify_flow_conservation(V, F, u, frames)
    if repair_flow and not flow['conserved']:
        u_repaired, flow_repair_info = repair_flow_violations(
            V=V,
            F=F,
            u=u,
            frames=frames,
            protected_mask=protected,
            max_rounds=flow_repair_max_rounds,
            expand_rounds=flow_repair_expand_rounds,
            smoothing_iters=flow_repair_smoothing_iters,
        )
        if flow_repair_info.get('accepted', False):
            u = u_repaired
            ph = verify_poincare_hopf(V, F, u, frames=frames)
            holo = verify_holonomy_compatibility(V, F, u, frames)
            flow = verify_flow_conservation(V, F, u, frames)

    if not flow['conserved']:
        warnings.warn(
            f"Flow conservation violated after GL solve: {flow['recommendation']}",
            stacklevel=2,
        )

    solve_info = {
        'L_real': L_real,
        'L_imag': L_imag,
        'A_factorized': A_factor,
        'mu': mu_vec,
        'N': N,
        'anisotropy': aniso,
        'confidence': conf,
        'reliability': reliability,
        'num_override_anchors': int(np.count_nonzero(override_active)),
        'num_singularity_anchors': int(np.count_nonzero(singularity_active)),
        'singularity_support_radius': float(singularity_support_radius),
        'num_umbilic': int((~anchor_mask).sum()),
        'singularity_filter': cancel_info,
        'high_order_repair': high_order_repair_info,
        'flow_repair': flow_repair_info,
        'poincare_hopf': ph,
        'holonomy': holo,
        'flow_conservation': flow,
    }
    return u, solve_info


# ---------------------------------------------------------------------------
# PyTorch differentiable wrapper
# ---------------------------------------------------------------------------

class _CrossFieldSolveFunction(torch.autograd.Function):
    """
    Custom autograd.Function for  u = A^{-1} (μ u*).

    Forward:  u = A_factor(μ u*)                           [numpy solve]
    Backward: ∂L/∂u* = μ A^{-T} ∂L/∂u = μ A^{-1} ∂L/∂u  [same solve, A symmetric]

    The (2N×2N) block system A is symmetric (L_real symmetric, L_imag antisymmetric),
    so A^{-T} = A^{-1}, and the backward pass reuses the same factorised solver.
    """

    @staticmethod
    def forward(ctx, u_star_ri: torch.Tensor, A_factor, mu) -> torch.Tensor:
        # u_star_ri: (N, 2) — [real, imag] columns
        N = u_star_ri.shape[0]
        u_star_np = u_star_ri.detach().cpu().double().numpy()
        if np.isscalar(mu):
            mu_vec = np.full(N, float(mu), dtype=np.float64)
        else:
            mu_vec = np.asarray(mu, dtype=np.float64).reshape(-1)
            if mu_vec.shape[0] != N:
                raise ValueError(f"mu length mismatch: {mu_vec.shape[0]} != {N}")
        b = np.concatenate([mu_vec * u_star_np[:, 0], mu_vec * u_star_np[:, 1]], axis=0)
        sol = A_factor(b)
        u_ri = torch.tensor(
            np.stack([sol[:N], sol[N:]], axis=1),
            dtype=u_star_ri.dtype, device=u_star_ri.device
        )
        ctx.A_factor = A_factor
        ctx.mu_vec = mu_vec
        return u_ri

    @staticmethod
    def backward(ctx, grad_u: torch.Tensor):
        A_factor = ctx.A_factor
        mu_vec = ctx.mu_vec
        N = grad_u.shape[0]
        grad_np = grad_u.detach().cpu().double().numpy().flatten(order='F')
        sol = A_factor(grad_np)
        g0 = mu_vec * sol[:N]
        g1 = mu_vec * sol[N:]
        grad_u_star = torch.tensor(np.stack([g0, g1], axis=1), dtype=grad_u.dtype, device=grad_u.device)
        return grad_u_star, None, None


def crossfield_from_metric_torch(
    M_vert: torch.Tensor,
    solve_info: dict,
) -> torch.Tensor:
    """
    Differentiable cross-field angles θ from per-vertex metrics M_vert.

    The gradient flows:  θ  ←  u = A^{-1}(μ u*)  ←  u* = exp(4iθ*(M_vert))
                             ←  M_vert via eigh

    Args:
        M_vert:     (N, 2, 2) metric tensors (PyTorch, grad-tracked).
        solve_info: dict returned by solve_crossfield_gl.

    Returns:
        theta:  (N,) angle in [−π/4, π/4] representing the cross-field direction.
    """
    # Step 1: guidance field u* from M_vert (differentiable via torch.linalg.eigh)
    from src.geometry.metric_utils import eigh2x2
    eigvals, eigvecs = eigh2x2(M_vert)              # (N,2), (N,2,2)
    d1 = eigvecs[..., -1]                          # (N, 2) largest-eigval direction
    theta_star = torch.atan2(d1[:, 1], d1[:, 0])  # (N,)
    # u* as [cos(4θ*), sin(4θ*)] — (N, 2)
    u_star_ri = torch.stack([torch.cos(4.0 * theta_star),
                              torch.sin(4.0 * theta_star)], dim=1)

    # Step 2: cross-field solve (differentiable via implicit diff)
    A_factor = solve_info['A_factorized']
    mu       = solve_info['mu']
    u_ri = _CrossFieldSolveFunction.apply(u_star_ri, A_factor, mu)

    # Step 3: recover angle θ = angle(u) / 4
    theta = 0.25 * torch.atan2(u_ri[:, 1], u_ri[:, 0])   # (N,)
    return theta


def crossfield_angles_from_complex(u: np.ndarray) -> np.ndarray:
    """
    Extract per-vertex cross-field angles θ ∈ [-π/4, π/4] from complex u = e^{4iθ}.
    """
    return (np.angle(u) / 4.0).astype(np.float32)


def euler_characteristic(V: np.ndarray, F: np.ndarray) -> int:
    """
    Compute Euler characteristic χ = V − E + F for a triangle mesh.
    Works for any triangle mesh (open or closed, connected or not).
    """
    edges = set()
    for f in F:
        for k in range(3):
            a, b = int(f[k]), int(f[(k+1) % 3])
            edges.add((min(a, b), max(a, b)))
    return len(V) - len(edges) + len(F)


def count_boundary_edges(F: np.ndarray) -> int:
    """Count edges incident to exactly one triangle."""
    edge_counts: dict[tuple[int, int], int] = {}
    for f in F:
        for k in range(3):
            a, b = int(f[k]), int(f[(k + 1) % 3])
            key = (min(a, b), max(a, b))
            edge_counts[key] = edge_counts.get(key, 0) + 1
    return int(sum(1 for count in edge_counts.values() if count == 1))


def verify_poincare_hopf(
    V: np.ndarray,
    F: np.ndarray,
    u: np.ndarray,
    frames: Optional[np.ndarray] = None,
    tol: float = 0.125,
) -> dict:
    """
    Verify the Poincaré–Hopf theorem for a 4-RoSy cross-field.

    For a cross-field on a closed orientable surface:
        Σ_i index(s_i) = χ(S)
    where index ∈ {±1/4, ±1/2, …} and χ = V − E + F.

    Returns:
        dict with keys:
            'chi':          Euler characteristic of the mesh.
            'sum_index':    Σ index(sᵢ) over all detected singularities.
            'satisfied':    True if |sum_index − chi| < 0.3 (rounding tolerance).
            'deficit':      sum_index − chi (signed error).
            'singularities': output of detect_singularities_from_crossfield.
            'recommendation': human-readable advice string.
    """
    sing = detect_singularities_from_crossfield(V, F, u, frames=frames, tol=tol)
    chi = euler_characteristic(V, F)
    boundary_edges = count_boundary_edges(F)
    if 'vertex_indices' in sing and len(sing['vertex_indices']) > 0:
        sum_idx = float(np.asarray(sing['vertex_indices'], dtype=np.float64).sum())
    else:
        sum_idx = float(sing['indices'].sum()) if len(sing['indices']) > 0 else 0.0

    if boundary_edges > 0:
        rec = (
            "Strict P-H check skipped for open mesh: boundary contribution is "
            f"not modelled ({boundary_edges} boundary edges)."
        )
        return {
            'chi': chi,
            'sum_index': sum_idx,
            'satisfied': True,
            'strict': False,
            'closed': False,
            'boundary_edges': boundary_edges,
            'deficit': 0.0,
            'singularities': sing,
            'recommendation': rec,
        }

    deficit = sum_idx - float(chi)
    satisfied = abs(deficit) < 0.3

    if satisfied:
        rec = "P-H satisfied — cross-field is topologically consistent."
    elif deficit > 0:
        rec = (f"P-H violated: Σidx={sum_idx:.2f} > χ={chi}. "
               f"Too many positive singularities. "
               f"Try increasing crossfield_mu to suppress spurious singularities.")
    else:
        rec = (f"P-H violated: Σidx={sum_idx:.2f} < χ={chi}. "
               f"Too few singularities. "
               f"Try decreasing crossfield_mu or anisotropy_eps.")

    return {
        'chi': chi,
        'sum_index': sum_idx,
        'satisfied': satisfied,
        'strict': True,
        'closed': True,
        'boundary_edges': 0,
        'deficit': deficit,
        'singularities': sing,
        'recommendation': rec,
    }


def _compute_edge_integer_mismatches(
    V: np.ndarray,
    F: np.ndarray,
    u: np.ndarray,
    frames: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-edge integer mismatch for the 4-RoSy field u.

    For each undirected edge (a, b), the integer mismatch in direction a→b is:
        m_ab = round( wrap(angle(u_b) - angle(u_a) - 4*r_{a→b}) / (π/2) )

    where r_{a→b} is the parallel-transport angle from the frame at a to b.
    The mismatch in direction b→a is -m_ab (antisymmetric).

    Returns:
        a_arr        (E,) — from-vertex index for each edge
        b_arr        (E,) — to-vertex index for each edge
        int_mismatch (E,) — integer mismatch in direction a→b
    """
    e1 = frames[:, :, 0]
    e2 = frames[:, :, 1]
    normals = np.cross(e1, e2)
    normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-10)

    # Unique undirected edges from faces
    edge_set = set()
    for f in F:
        for k in range(3):
            a, b = int(f[k]), int(f[(k + 1) % 3])
            edge_set.add((min(a, b), max(a, b)))
    edges = np.array(sorted(edge_set), dtype=np.int64)  # (E, 2)
    a_arr = edges[:, 0]
    b_arr = edges[:, 1]

    # Parallel-transport angle r_{a→b}
    e1_a = e1[a_arr]
    n_b  = normals[b_arr]
    transported = e1_a - np.einsum('ei,ei->e', e1_a, n_b)[:, None] * n_b
    t_norm = np.linalg.norm(transported, axis=1, keepdims=True)
    transported /= np.maximum(t_norm, 1e-10)
    cos_r = np.einsum('ei,ei->e', transported, e1[b_arr])
    sin_r = np.einsum('ei,ei->e', transported, e2[b_arr])
    r_ab  = np.arctan2(sin_r, cos_r)

    phi = np.angle(u.astype(np.complex128))
    delta = phi[b_arr] - phi[a_arr] - 4.0 * r_ab
    delta = (delta + np.pi) % (2.0 * np.pi) - np.pi   # wrap to [-π, π]
    int_mismatch = np.rint(delta / (np.pi / 2.0)).astype(np.int32)

    return a_arr, b_arr, int_mismatch


def _compute_face_integer_charges(
    V: np.ndarray,
    F: np.ndarray,
    u: np.ndarray,
    frames: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Connection-aware per-face integer charge for a 4-RoSy field.

    With `frames`, each directed edge contributes its transport-corrected
    integer mismatch. Without `frames`, this falls back to raw phase winding.
    """
    tri = np.asarray(F, dtype=np.int64)
    if tri.ndim != 2 or tri.shape[1] != 3 or len(tri) == 0:
        return np.zeros((0,), dtype=np.int64)

    if frames is None:
        phi = np.angle(u.astype(np.complex128))
        p0 = phi[tri[:, 0]]
        p1 = phi[tri[:, 1]]
        p2 = phi[tri[:, 2]]

        def wrap(x):
            return (x + np.pi) % (2.0 * np.pi) - np.pi

        winding = wrap(p1 - p0) + wrap(p2 - p1) + wrap(p0 - p2)
        return np.rint(winding / (2.0 * np.pi)).astype(np.int64)

    a_arr, b_arr, int_mismatch = _compute_edge_integer_mismatches(V, tri, u, frames)
    undirected = {
        (int(a), int(b)): int(m)
        for a, b, m in zip(a_arr.tolist(), b_arr.tolist(), int_mismatch.tolist())
    }

    out = np.zeros(len(tri), dtype=np.int64)
    for fi, (a, b, c) in enumerate(tri.tolist()):
        mab = undirected.get((min(a, b), max(a, b)), 0)
        mbc = undirected.get((min(b, c), max(b, c)), 0)
        mca = undirected.get((min(c, a), max(c, a)), 0)
        if a > b:
            mab = -mab
        if b > c:
            mbc = -mbc
        if c > a:
            mca = -mca
        out[fi] = int(mab + mbc + mca)
    return out


def _compute_vertex_integer_charges(
    V: np.ndarray,
    F: np.ndarray,
    u: np.ndarray,
    frames: np.ndarray,
) -> np.ndarray:
    """
    Connection-aware per-vertex integer charge in quarter-turn units.

    This is the closest Python-side analogue of libigl's singularityIndex:
    sum of outgoing edge mismatches around each vertex, reduced mod 4 to a
    canonical singularity unit in {-2,-1,0,+1}.
    """
    N = len(V)
    a_arr, b_arr, int_mismatch = _compute_edge_integer_mismatches(V, F, u, frames)
    charge = np.zeros(N, dtype=np.int64)
    np.add.at(charge, a_arr, int_mismatch.astype(np.int64))
    np.add.at(charge, b_arr, -int_mismatch.astype(np.int64))
    return ((charge + 2) % 4) - 2


def verify_holonomy_compatibility(
    V: np.ndarray,
    F: np.ndarray,
    u: np.ndarray,
    frames: np.ndarray,
    tol: float = 0.125,
) -> dict:
    """
    Verify holonomy compatibility of a 4-RoSy cross-field for quadrangulation.

    Two conditions are checked:

    1. No high-order singularities — indices must all be ∈ {±1/4}.
       Singularities with |index| > 1/4 (e.g. ±1/2, ±3/4, ±1) violate the
       quadrangulability condition regardless of global topology.

    2. Co-tree loop holonomies ≡ 0 mod 4 (integer mismatch units, π/2 per unit).
       For genus-0 surfaces, Poincaré–Hopf is sufficient (all loops contractible,
       so this check is skipped).  For genus-g > 0, the 2g non-contractible
       generators must each have zero accumulated holonomy mod π/2.

    Algorithm for co-tree check:
        - Accumulate integer mismatches h[v] from root to each vertex via BFS
          spanning tree.
        - For each co-tree edge (a, b) with mismatch m_ab:
            loop_holonomy = m_ab + h[a] - h[b]
          Must be ≡ 0 mod 4.

    Returns:
        dict with keys:
            'compatible'        — bool: both checks passed
            'high_order_count'  — int: number of high-order singularities
            'cotree_violations' — int: co-tree loops with non-zero holonomy
            'genus'             — int: inferred surface genus
            'recommendation'    — str: human-readable advice
    """
    sing = detect_singularities_from_crossfield(V, F, u, frames=frames, tol=tol)
    chi = euler_characteristic(V, F)
    boundary_edges = count_boundary_edges(F)
    closed = boundary_edges == 0
    genus = max(0, (2 - chi) // 2) if closed else None

    # Check 1: high-order singularities
    indices = sing['indices']
    high_order_count = int(np.sum(np.abs(np.abs(indices) - 0.25) > 0.1))

    # Check 2: co-tree holonomies (closed surfaces only; skip for genus-0)
    cotree_violations = 0
    num_cotree_edges = 0
    if closed and genus > 0:
        N = len(V)
        a_arr, b_arr, int_mismatch = _compute_edge_integer_mismatches(V, F, u, frames)
        E = len(a_arr)

        # Build adjacency list
        adj = [[] for _ in range(N)]
        for idx in range(E):
            a, b = int(a_arr[idx]), int(b_arr[idx])
            adj[a].append((b, idx, +1))   # +1: stored direction a→b
            adj[b].append((a, idx, -1))   # -1: traverse b→a = reverse

        # BFS spanning tree: accumulate holonomy h[v] from root
        from collections import deque
        h = np.zeros(N, dtype=np.int64)
        in_tree = np.zeros(E, dtype=bool)
        visited = np.zeros(N, dtype=bool)
        visited[0] = True
        queue = deque([0])
        while queue:
            v = queue.popleft()
            for (nb, eidx, sign) in adj[v]:
                if not visited[nb]:
                    visited[nb] = True
                    in_tree[eidx] = True
                    # sign=+1 means mismatch is in direction v→nb (+m)
                    # sign=-1 means edge stored as nb→v, so mismatch v→nb = -m
                    h[nb] = h[v] + sign * int(int_mismatch[eidx])
                    queue.append(nb)

        # Check each co-tree edge
        for eidx in range(E):
            if in_tree[eidx]:
                continue
            num_cotree_edges += 1
            a, b = int(a_arr[eidx]), int(b_arr[eidx])
            # Loop holonomy: go a→b via co-tree, then back via tree
            loop = int(int_mismatch[eidx]) + int(h[a]) - int(h[b])
            if loop % 4 != 0:
                cotree_violations += 1

    compatible = (high_order_count == 0) and (cotree_violations == 0)

    if compatible:
        if closed:
            rec = "Holonomy compatible — field satisfies quadrangulation conditions."
        else:
            rec = (
                "Strict global holonomy check skipped for open mesh; "
                f"high-order singularity check passed ({boundary_edges} boundary edges)."
            )
    else:
        parts = []
        if high_order_count > 0:
            parts.append(f"{high_order_count} high-order singularities (|index| > 1/4)")
        if cotree_violations > 0:
            parts.append(
                f"{cotree_violations}/{num_cotree_edges} co-tree loops have non-zero holonomy "
                f"mod π/2 (genus={genus})"
            )
        rec = "Holonomy incompatible: " + "; ".join(parts) + ". Quads may not form a valid mesh."

    return {
        'compatible': compatible,
        'high_order_count': high_order_count,
        'cotree_violations': cotree_violations,
        'genus': genus,
        'closed': closed,
        'boundary_edges': boundary_edges,
        'strict_global_check': closed,
        'recommendation': rec,
    }


def verify_flow_conservation(
    V: np.ndarray,
    F: np.ndarray,
    u: np.ndarray,
    frames: np.ndarray,
    tol: float = 0.125,
) -> dict:
    """
    Verify QuadriFlow-style min-cost-flow conservation for a 4-RoSy field.

    For each vertex v, the sum of outgoing integer mismatches over its 1-ring fan
    must equal 4 × (singularity index at v).  For non-singular interior vertices
    this means the sum = 0 mod 4.  T-junction vertices (those that violate this)
    indicate that the integer transition functions are inconsistent and will produce
    holes or overlaps in the extracted quad mesh.

    Per-vertex charge:
        charge[v] = Σ_{e incident to v} sign_v(e) * m_e
    where sign_v(e) = +1 if v = a_arr[e], −1 if v = b_arr[e].

    Expected charge for a singular vertex of index k/4: charge = k mod 4 ∈ {0,±1}.
    (A non-singular vertex has index 0, expected charge 0.)

    Returns:
        dict with keys:
            'conserved'         — bool: no T-junction violations found
            'num_violations'    — int: vertices with flow conservation failure
            'num_singular'      — int: total detected singularities
            'violation_ids'     — (K,) vertex indices that violate conservation
            'recommendation'    — str: human-readable advice
    """
    N = len(V)
    a_arr, b_arr, int_mismatch = _compute_edge_integer_mismatches(V, F, u, frames)

    # Per-vertex outgoing charge
    charge = np.zeros(N, dtype=np.int64)
    np.add.at(charge, a_arr, int_mismatch.astype(np.int64))
    np.add.at(charge, b_arr, -int_mismatch.astype(np.int64))   # reverse direction

    tri = F.astype(np.int64)
    face_charge = _compute_face_integer_charges(V, tri, u, frames=frames)

    # Distribute face charge to vertices (each face contributes 1/3 of its charge)
    # Use integer arithmetic: vertex_expected_charge = sum of face_charge over touching faces
    # For a non-singular vertex all touching faces have charge 0 → sum = 0
    vertex_face_charge = np.zeros(N, dtype=np.int64)
    np.add.at(vertex_face_charge, tri[:, 0], face_charge)
    np.add.at(vertex_face_charge, tri[:, 1], face_charge)
    np.add.at(vertex_face_charge, tri[:, 2], face_charge)

    # Flow violation: charge[v] mod 4 ≠ vertex_face_charge[v] mod 4
    # (both should reduce to the same value for a consistent field)
    flow_mod = charge % 4
    expected_mod = vertex_face_charge % 4
    violation_mask = (flow_mod != expected_mod)
    violation_ids = np.where(violation_mask)[0]
    num_violations = int(len(violation_ids))
    num_singular = int((np.abs(face_charge) > 0).sum())

    conserved = num_violations == 0
    if conserved:
        rec = "Flow conservation satisfied — no T-junctions detected."
    else:
        rate = 100.0 * num_violations / N
        rec = (
            f"{num_violations}/{N} ({rate:.1f}%) vertices violate flow conservation "
            f"(T-junctions). These cause holes/overlaps in quad extraction. "
            f"Consider increasing crossfield_mu or gradient_size."
        )

    return {
        'conserved': conserved,
        'num_violations': num_violations,
        'num_singular': num_singular,
        'violation_ids': violation_ids,
        'recommendation': rec,
    }


def detect_singularities_from_crossfield(
    V: np.ndarray,
    F: np.ndarray,
    u: np.ndarray,
    frames: Optional[np.ndarray] = None,
    tol: float = 0.125,
) -> dict:
    """
    Detect cross-field singularities from winding number of u = exp(4iθ).

    For each triangle, compute:
        charge_u = round( sum(wrap(dphi)) / (2π) )
        index    = charge_u / 4

    Returns a dictionary with non-zero singular triangles and per-sign counts.
    """
    if F.ndim != 2 or F.shape[1] != 3:
        return {
            'num_singular_faces': 0,
            'num_positive': 0,
            'num_negative': 0,
            'indices': np.zeros((0,), dtype=np.float32),
            'face_ids': np.zeros((0,), dtype=np.int64),
            'positions': np.zeros((0, 3), dtype=np.float64),
            'num_singular_vertices': 0,
            'vertex_ids': np.zeros((0,), dtype=np.int64),
            'vertex_units': np.zeros((0,), dtype=np.int32),
            'vertex_indices': np.zeros((0,), dtype=np.float32),
            'vertex_positions': np.zeros((0, 3), dtype=np.float64),
            'sum_vertex_units': 0,
        }

    tri = F.astype(np.int64)
    charge_u = _compute_face_integer_charges(V, tri, u, frames=frames).astype(np.int32)
    idx = charge_u.astype(np.float32) / 4.0

    mask = np.abs(idx) > float(tol)
    face_ids = np.where(mask)[0].astype(np.int64)
    idx_nz = idx[mask]
    pos = V[tri[face_ids]].mean(axis=1) if len(face_ids) > 0 else np.zeros((0, 3), dtype=np.float64)

    out = {
        'num_singular_faces': int(len(face_ids)),
        'num_positive': int((idx_nz > 0).sum()),
        'num_negative': int((idx_nz < 0).sum()),
        'indices': idx_nz.astype(np.float32),
        'face_ids': face_ids,
        'positions': pos.astype(np.float64),
    }
    if frames is not None:
        vertex_units = _compute_vertex_integer_charges(V, tri, u, frames)
        vmask = vertex_units != 0
        vertex_ids = np.where(vmask)[0].astype(np.int64)
        out.update({
            'num_singular_vertices': int(len(vertex_ids)),
            'vertex_ids': vertex_ids,
            'vertex_units': vertex_units[vmask].astype(np.int32),
            'vertex_indices': (vertex_units[vmask].astype(np.float32) / 4.0),
            'vertex_positions': V[vertex_ids].astype(np.float64) if len(vertex_ids) > 0 else np.zeros((0, 3), dtype=np.float64),
            'sum_vertex_units': int(vertex_units[vmask].sum()) if len(vertex_ids) > 0 else 0,
        })
    else:
        out.update({
            'num_singular_vertices': 0,
            'vertex_ids': np.zeros((0,), dtype=np.int64),
            'vertex_units': np.zeros((0,), dtype=np.int32),
            'vertex_indices': np.zeros((0,), dtype=np.float32),
            'vertex_positions': np.zeros((0, 3), dtype=np.float64),
            'sum_vertex_units': 0,
        })
    return out
