"""
Field-aligned quad topology generation.

Replaces the old midpoint-subdivision shortcut with a principled pipeline:

    Triangle mesh  →  Connection Laplacian  →  Ginzburg–Landau cross-field
                   →  Gradient-alignment parametrisation
                   →  Integer-grid quad extraction

All three steps are differentiable (§5.1–5.3 of PAPER.md).

⚠ The midpoint-subdivision (_midpoint_subdivision_to_quad) and greedy-pairing
  (_greedy_tri_to_quad) fallbacks have been removed.  If topology extraction
  fails the pipeline raises RuntimeError with a diagnostic message instead of
  producing field-independent "Voronoi garbage".
"""

from __future__ import annotations
import os
import shutil
import subprocess
import tempfile
import inspect
import re
from collections import Counter
import numpy as np
import warnings
from scipy.spatial import KDTree
from typing import Optional, Tuple

try:
    import open3d as o3d
except ImportError:
    o3d = None

from .crossfield import (
    compute_vertex_frames,
    guidance_field_from_metric,
    solve_crossfield_gl,
    crossfield_angles_from_complex,
    detect_singularities_from_crossfield,
    verify_poincare_hopf,
)
from .parametrization import (
    solve_parametrisation,
    extract_quads_integer_grid,
    mesh_edges,
    precompute_param_laplacian,
)

# Default path of the compiled C++ binary relative to the project root.
_DEFAULT_BINARY = os.path.join(
    os.path.dirname(__file__), '..', '..', 'cpp_miq', 'build', 'run_miq'
)


def _lift_lcf_metric_to_world(metric_lcf: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Lift per-point 2D LCF metrics to 3D tangent-plane tensors."""
    R = np.asarray(basis, dtype=np.float64)[:, :, :2]
    M = np.asarray(metric_lcf, dtype=np.float64)
    if R.shape[0] != M.shape[0] or M.shape[-2:] != (2, 2):
        raise ValueError(
            "metric_basis must align with metric_field: "
            f"metric_field={M.shape}, metric_basis={R.shape}"
        )
    return np.einsum('nia,nab,njb->nij', R, M, R)


def _project_world_metric_to_frames(metric_world: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Project 3D tangent-plane tensors into a mesh vertex frame."""
    R = np.asarray(frames, dtype=np.float64)
    M = np.asarray(metric_world, dtype=np.float64)
    if R.shape[0] != M.shape[0] or M.shape[-2:] != (3, 3):
        raise ValueError(
            "metric_world must align with frames: "
            f"metric_world={M.shape}, frames={R.shape}"
        )
    return np.einsum('nia,nij,njb->nab', R, M, R)


def _parse_run_miq_diagnostics(stdout: str) -> dict:
    diag: dict = {}
    if not stdout:
        return diag

    m = re.search(
        r"UV fold-overs recovered:\s*(\d+), skipped:\s*(\d+), degenerate:\s*(\d+)\s*/\s*(\d+)\s*UV triangles",
        stdout,
    )
    if m:
        diag['uv_extraction'] = {
            'recovered_fold_overs': int(m.group(1)),
            'skipped_fold_overs': int(m.group(2)),
            'degenerate_uv_triangles': int(m.group(3)),
            'uv_triangles': int(m.group(4)),
        }

    m = re.search(r"Winding corrected:\s*(\d+)/(\d+)", stdout)
    if m:
        diag['quad_winding'] = {
            'corrected_quads': int(m.group(1)),
            'quad_count': int(m.group(2)),
        }

    flips = [int(v) for v in re.findall(r"ITERATION\s+\d+\s+FLIPS\s+(\d+)", stdout)]
    if flips:
        diag['miq_rounding'] = {
            'flip_count_max': int(max(flips)),
            'flip_count_final': int(flips[-1]),
            'flip_count_iters': int(len(flips)),
        }

    m = re.search(r"UV valid area:\s*([0-9eE+\-.]+)\s*\(pos=(\d+)\s+neg=(\d+)\)", stdout)
    if m:
        diag['uv_area'] = {
            'valid_area': float(m.group(1)),
            'positive_triangles': int(m.group(2)),
            'negative_triangles': int(m.group(3)),
        }

    m = re.search(
        r"Poincar[ée]\S*\s*Hopf:\s*Σ\s*singularity_index\s*=\s*(-?\d+)\s+expected\s*=\s*(-?\d+)\s*\(χ\s*=\s*(-?\d+)\)",
        stdout,
    )
    if m:
        sum_units = int(m.group(1))
        expected_units = int(m.group(2))
        chi = int(m.group(3))
        diag['poincare_hopf'] = {
            'sum_units': sum_units,
            'expected_units': expected_units,
            'chi': chi,
            'sum_index': float(sum_units) / 4.0,
            'expected_index': float(expected_units) / 4.0,
            'deficit': float(sum_units - expected_units) / 4.0,
            'satisfied': (sum_units == expected_units),
        }

    if 'poincare_hopf' in diag:
        if re.search(r"Poincar[ée]\S*\s*Hopf\s*satisfied", stdout):
            diag['poincare_hopf']['satisfied'] = True
        elif re.search(r"Poincar[ée]\S*\s*Hopf\s*violated", stdout):
            diag['poincare_hopf']['satisfied'] = False

    return diag


def _solve_crossfield_gl_compat(
    *,
    V: np.ndarray,
    F: np.ndarray,
    M_vert: np.ndarray,
    frames: np.ndarray,
    mu: float,
    umbilic_smoothing: bool,
    anisotropy_eps: float,
    mu_min_ratio: float,
    guidance_confidence: Optional[np.ndarray],
    guidance_override: Optional[np.ndarray] = None,
    guidance_override_weight: Optional[np.ndarray] = None,
    singularity_mask: Optional[np.ndarray] = None,
    singularity_indices: Optional[np.ndarray] = None,
    protected_mask: Optional[np.ndarray] = None,
    guidance_override_reliability_floor: Optional[float] = None,
    guidance_override_anchor_eps: Optional[float] = None,
    singularity_cancel_distance_ratio: Optional[float] = None,
    singularity_cancel_smoothing_iters: Optional[int] = None,
    singularity_cancel_max_rounds: Optional[int] = None,
    repair_high_order: Optional[bool] = None,
    high_order_repair_max_rounds: Optional[int] = None,
    high_order_repair_expand_rounds: Optional[int] = None,
    high_order_repair_smoothing_iters: Optional[int] = None,
    repair_flow: Optional[bool] = None,
    flow_repair_max_rounds: Optional[int] = None,
    flow_repair_expand_rounds: Optional[int] = None,
    flow_repair_smoothing_iters: Optional[int] = None,
):
    """
    Compatibility wrapper for crossfield solver signatures.

    Some environments may still have an older `solve_crossfield_gl` that does
    not accept `guidance_confidence`. In that case we silently fall back to the
    legacy call path.
    """
    sig = inspect.signature(solve_crossfield_gl)
    kwargs = dict(
        V=V,
        F=F,
        M_vert=M_vert,
        frames=frames,
        mu=mu,
        umbilic_smoothing=umbilic_smoothing,
        anisotropy_eps=anisotropy_eps,
        mu_min_ratio=mu_min_ratio,
    )
    if 'guidance_confidence' in sig.parameters:
        kwargs['guidance_confidence'] = guidance_confidence
    if 'guidance_override' in sig.parameters:
        kwargs['guidance_override'] = guidance_override
    if 'guidance_override_weight' in sig.parameters:
        kwargs['guidance_override_weight'] = guidance_override_weight
    if 'singularity_mask' in sig.parameters and singularity_mask is not None:
        kwargs['singularity_mask'] = singularity_mask
    if 'singularity_indices' in sig.parameters and singularity_indices is not None:
        kwargs['singularity_indices'] = singularity_indices
    if 'protected_mask' in sig.parameters and protected_mask is not None:
        kwargs['protected_mask'] = protected_mask
    if 'guidance_override_reliability_floor' in sig.parameters and guidance_override_reliability_floor is not None:
        kwargs['guidance_override_reliability_floor'] = guidance_override_reliability_floor
    if 'guidance_override_anchor_eps' in sig.parameters and guidance_override_anchor_eps is not None:
        kwargs['guidance_override_anchor_eps'] = guidance_override_anchor_eps
    if 'singularity_cancel_distance_ratio' in sig.parameters and singularity_cancel_distance_ratio is not None:
        kwargs['singularity_cancel_distance_ratio'] = singularity_cancel_distance_ratio
    if 'singularity_cancel_smoothing_iters' in sig.parameters and singularity_cancel_smoothing_iters is not None:
        kwargs['singularity_cancel_smoothing_iters'] = singularity_cancel_smoothing_iters
    if 'singularity_cancel_max_rounds' in sig.parameters and singularity_cancel_max_rounds is not None:
        kwargs['singularity_cancel_max_rounds'] = singularity_cancel_max_rounds
    if 'repair_high_order' in sig.parameters and repair_high_order is not None:
        kwargs['repair_high_order'] = repair_high_order
    if 'high_order_repair_max_rounds' in sig.parameters and high_order_repair_max_rounds is not None:
        kwargs['high_order_repair_max_rounds'] = high_order_repair_max_rounds
    if 'high_order_repair_expand_rounds' in sig.parameters and high_order_repair_expand_rounds is not None:
        kwargs['high_order_repair_expand_rounds'] = high_order_repair_expand_rounds
    if 'high_order_repair_smoothing_iters' in sig.parameters and high_order_repair_smoothing_iters is not None:
        kwargs['high_order_repair_smoothing_iters'] = high_order_repair_smoothing_iters
    if 'repair_flow' in sig.parameters and repair_flow is not None:
        kwargs['repair_flow'] = repair_flow
    if 'flow_repair_max_rounds' in sig.parameters and flow_repair_max_rounds is not None:
        kwargs['flow_repair_max_rounds'] = flow_repair_max_rounds
    if 'flow_repair_expand_rounds' in sig.parameters and flow_repair_expand_rounds is not None:
        kwargs['flow_repair_expand_rounds'] = flow_repair_expand_rounds
    if 'flow_repair_smoothing_iters' in sig.parameters and flow_repair_smoothing_iters is not None:
        kwargs['flow_repair_smoothing_iters'] = flow_repair_smoothing_iters
    return solve_crossfield_gl(**kwargs)


def _count_foldovers(V: np.ndarray, quadF: np.ndarray) -> int:
    """
    Vectorised count of fold-over quads (no modification).

    A quad (v0,v1,v2,v3) is a fold-over when the two halves of its
    diagonal-split triangulation (0-1-2) and (0-2-3) have normals
    pointing into opposite half-spaces (dot < 0).
    """
    if quadF.ndim != 2 or quadF.shape[1] != 4 or quadF.shape[0] == 0:
        return 0
    v = V[quadF]                                       # (Q, 4, 3)
    n_a = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])  # (Q, 3)
    n_b = np.cross(v[:, 2] - v[:, 0], v[:, 3] - v[:, 0])  # (Q, 3)
    return int(((n_a * n_b).sum(-1) < 0).sum())


def _fix_foldover_quads(
    V: np.ndarray,
    quadF: np.ndarray,
    max_iters: int = 5,
    shrink: float = 0.25,
) -> Tuple[np.ndarray, int]:
    """
    Detect and repair fold-over quads by pulling affected vertices toward
    their quad centroid.

    A quad (v0,v1,v2,v3) is a fold-over when its two triangulations
    (0-1-2) and (0-2-3) have normals pointing in opposite half-spaces
    (dot < 0), meaning the quad self-intersects.

    Repair: each vertex of a bad quad is nudged ``shrink`` fraction toward
    the quad centroid.  Repeated until no fold-overs remain or max_iters
    exhausted.  Vertices are moved in 3-D, so surface fidelity is
    preserved to first order.

    Returns:
        V_fixed:      vertex array (same shape as V) with repaired positions.
        total_fixed:  number of fold-overs resolved across all iterations.
    """
    V = V.copy().astype(np.float64)
    total_fixed = 0

    for _ in range(max_iters):
        fixed_this = 0
        for q in quadF:
            v = V[q]                                   # (4, 3)
            n_a = np.cross(v[1] - v[0], v[2] - v[0])  # tri 0-1-2
            n_b = np.cross(v[2] - v[0], v[3] - v[0])  # tri 0-2-3
            if np.dot(n_a, n_b) < 0.0:
                centroid = v.mean(axis=0)
                V[q] = (1.0 - shrink) * v + shrink * centroid
                fixed_this += 1
        total_fixed += fixed_this
        if fixed_this == 0:
            break

    return V, total_fixed


def _remove_degenerate_quads(
    V: np.ndarray,
    quadF: np.ndarray,
    min_area_ratio: float = 1e-4,
) -> np.ndarray:
    """
    Remove quads whose area is below ``min_area_ratio * mean_quad_area``.

    Degenerate quads (two nearly-coincident vertices) appear visually as
    triangles and arise near singularities when MIQ integer transitions don't
    close perfectly.  Removing them is safe: they contribute zero surface area
    and are already visually invisible.
    """
    if quadF.ndim != 2 or quadF.shape[1] != 4 or quadF.shape[0] == 0:
        return quadF
    v = V[quadF]                                        # (Q, 4, 3)
    # Area via diagonal cross products (two triangle halves)
    area_a = np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=-1)
    area_b = np.linalg.norm(np.cross(v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]), axis=-1)
    area   = 0.5 * (area_a + area_b)                   # (Q,)
    threshold = min_area_ratio * (area.mean() + 1e-30)
    return quadF[area > threshold]


def _compact_quad_mesh(
    V: np.ndarray,
    quadF: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove orphan vertices left after face pruning; remap indices."""
    if quadF.ndim != 2 or quadF.shape[1] != 4 or quadF.shape[0] == 0:
        return V, quadF
    used = np.unique(quadF)
    remap = np.empty(V.shape[0], dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    return V[used], remap[quadF]


def _prune_quads_to_closed_manifold_subset(quads: np.ndarray) -> np.ndarray:
    """
    Keep only a closed manifold subset (all edges incidence exactly 2), if any.
    """
    Q = np.asarray(quads, dtype=np.int64)
    if Q.ndim != 2 or Q.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.int64)
    alive = np.ones(Q.shape[0], dtype=bool)
    while True:
        edge_counts = Counter()
        edge_to_faces = {}
        for fi in np.where(alive)[0]:
            q = Q[fi]
            for k in range(4):
                a = int(q[k]); b = int(q[(k + 1) % 4])
                if a > b:
                    a, b = b, a
                e = (a, b)
                edge_counts[e] += 1
                edge_to_faces.setdefault(e, []).append(fi)
        bad = [e for e, c in edge_counts.items() if c != 2]
        if not bad:
            break
        remove = set()
        for e in bad:
            remove.update(edge_to_faces.get(e, []))
        if not remove:
            break
        for fi in remove:
            alive[fi] = False
        if alive.sum() == 0:
            return np.zeros((0, 4), dtype=np.int64)
    return Q[alive]


# ---------------------------------------------------------------------------
# Poisson surface reconstruction
# ---------------------------------------------------------------------------

def poisson_surface_reconstruction(
    points: np.ndarray,
    normals: np.ndarray,
    depth: int = 6,
    width: float = 0.0,
    scale: float = 1.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Screened Poisson surface reconstruction.

    Returns:
        vertices:  (V, 3)
        faces:     (F, 3) triangle faces.

    Raises:
        RuntimeError if reconstruction fails.
    """
    if o3d is not None:
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points  = o3d.utility.Vector3dVector(points)
            pcd.normals = o3d.utility.Vector3dVector(normals)
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=depth, width=width, scale=scale
            )
            V = np.asarray(mesh.vertices)
            F = np.asarray(mesh.triangles)
            if V.shape[0] > 0 and F.ndim == 2 and F.shape[1] == 3:
                return V, F
            warnings.warn(
                f"Open3D Poisson returned unexpected shapes V={V.shape} F={F.shape}."
            )
        except Exception as exc:
            warnings.warn(f"Open3D Poisson failed: {exc}")

    raise RuntimeError(
        "Poisson reconstruction failed to produce a valid triangle mesh.\n"
        "Ensure Open3D is installed and point normals are consistent."
    )


# ---------------------------------------------------------------------------
# Mesh orientation utilities
# ---------------------------------------------------------------------------

def _orient_mesh_outward(
    V: np.ndarray,
    F: np.ndarray,
    ref_points: Optional[np.ndarray] = None,
    ref_normals: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensure triangle faces have CCW winding (outward-pointing normals).

    Two strategies:
    1. If ref_points + ref_normals are provided (point cloud case): compare
       each face normal with the nearest input point normal; flip globally if
       the mean dot-product is negative.
    2. Otherwise (closed mesh): use the signed-volume test.  Negative signed
       volume ⟹ CW winding ⟹ flip.

    Returns:
        V, F  with F possibly flipped (column order [0,2,1]).
    """
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]

    if ref_points is not None and ref_normals is not None:
        face_centers = (v0 + v1 + v2) / 3.0
        face_n = np.cross(v1 - v0, v2 - v0)
        face_n /= np.linalg.norm(face_n, axis=1, keepdims=True) + 1e-10
        tree = KDTree(ref_points)
        _, idx = tree.query(face_centers)
        mean_dot = float((face_n * ref_normals[idx]).sum(axis=1).mean())
        if mean_dot < 0:
            F = F[:, [0, 2, 1]]
    else:
        # Signed-volume test (works for closed manifold meshes)
        signed_vol = float(np.einsum('fi,fi->f', v0, np.cross(v1, v2)).sum())
        if signed_vol < 0:
            F = F[:, [0, 2, 1]]

    return V, F


# ---------------------------------------------------------------------------
# Metric field transfer (point cloud → mesh vertices)
# ---------------------------------------------------------------------------

def _transfer_metric_to_mesh(
    points: np.ndarray,
    metric_field: np.ndarray,
    V_mesh: np.ndarray,
) -> np.ndarray:
    """
    Nearest-neighbour transfer of per-point metric tensors to mesh vertices.

    Returns:
        M_vert:  (N_mesh, ..., ...) metric tensors.
    """
    tree = KDTree(points)
    _, idx = tree.query(V_mesh)
    return metric_field[idx]


def _transfer_scalar_to_mesh(
    points: np.ndarray,
    values: np.ndarray,
    V_mesh: np.ndarray,
) -> np.ndarray:
    """
    Nearest-neighbour transfer of per-point scalar values to mesh vertices.
    """
    tree = KDTree(points)
    _, idx = tree.query(V_mesh)
    return values[idx]


# ---------------------------------------------------------------------------
# Frame-field construction (legacy helper kept for external callers)
# ---------------------------------------------------------------------------

def compute_frame_field(
    points: np.ndarray,
    metric_field: np.ndarray,
    V: np.ndarray,
    F: np.ndarray,
    normals: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Transfer metric field to mesh vertices and return (N, 3, 2) tangent frames.

    This function is retained for backward compatibility with callers that only
    need the frame field (e.g. visualisation).  The cross-field computation is
    now done in `miq_quadrangulate`.
    """
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)

    if V.shape[0] == 0:
        raise ValueError("Mesh vertex array is empty.")
    if F.shape[0] == 0 or F.shape[1] != 3:
        raise ValueError(f"Expected triangle faces (F,3); got {F.shape}.")

    frames = compute_vertex_frames(V, F, normals)
    return frames


# ---------------------------------------------------------------------------
# Main quad-mesh pipeline
# ---------------------------------------------------------------------------

def miq_quadrangulate(
    V: np.ndarray,
    F: np.ndarray,
    frame_field: np.ndarray,
    metric_field_vert: Optional[np.ndarray] = None,
    guidance_confidence: Optional[np.ndarray] = None,
    gradient_size: float = 1.0,
    crossfield_mu: float = 10.0,
    umbilic_smoothing: bool = True,
    anisotropy_eps: float = 0.03,
    mu_min_ratio: float = 0.05,
    integer_constraints: bool = True,
    require_closed_topology: bool = False,
    anisotropy_schedule: Optional[Tuple[float, ...]] = None,
    integer_projection_iters: int = 2,
    integer_potential_iters: int = 2,
    max_param_dist: float = 0.7,
    return_diagnostics: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a field-aligned quad mesh from a triangle mesh + metric field.

    Pipeline:
        1. Build connection Laplacian on the triangle mesh.
        2. Solve Ginzburg–Landau cross-field:  (L_conn + μI) u = μ u*.
        3. Extract cross-field angles θ = angle(u) / 4.
        4. Solve two gradient-alignment Poisson systems → (φ₁, φ₂).
        5. Extract quad topology via integer-grid sampling.

    Args:
        V:                   (N, 3) mesh vertices.
        F:                   (M, 3) triangle faces.
        frame_field:         (N, 3, 2) per-vertex tangent frames.
        metric_field_vert:   (N, 2, 2) per-vertex metric tensors (2×2 in LCF).
                             If None, an isotropic field is used (θ* = 0 everywhere).
        gradient_size:       Controls quad density: smaller value → more quads.
        crossfield_mu:       Alignment weight μ in the GL equation.
        integer_constraints: If False, skip quad extraction (return triangles as-is).

    Returns:
        quadV:  (N, 3)  — same as input V (topology extraction does not add vertices).
        quadF:  (Q, 4)  — quad faces.

    Raises:
        RuntimeError: if quad extraction produces no valid faces.
    """
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)

    if V.shape[0] == 0 or F.shape[0] == 0:
        raise RuntimeError("Empty mesh passed to miq_quadrangulate.")
    if F.shape[1] != 3:
        raise RuntimeError(
            f"miq_quadrangulate requires a triangle mesh; got faces shape {F.shape}."
        )

    frames = frame_field   # (N, 3, 2)

    # Default metric: isotropic (all principal directions = 0)
    if metric_field_vert is None:
        metric_field_vert = np.stack([np.eye(2)] * len(V), axis=0)

    # Robustness-by-construction: anisotropy continuation.
    # If the fully anisotropic metric is hard to integrate into a valid integer
    # layout, progressively reduce anisotropy while preserving metric positivity.
    if anisotropy_schedule is None:
        anisotropy_schedule = (1.0, 0.7, 0.45, 0.25, 0.0)

    def _blend_metric_anisotropy(M: np.ndarray, alpha: float) -> np.ndarray:
        # M = Q diag(l1,l2) Q^T,  l' = 1 + alpha*(l-1)
        # alpha=1: original metric, alpha=0: isotropic identity.
        evals, evecs = np.linalg.eigh(M)
        lam = 1.0 + float(alpha) * (evals - 1.0)
        lam = np.clip(lam, 1e-6, None)
        return evecs @ np.diag(lam) @ evecs.T

    last_exc = None
    quadF = None
    u = None
    used_alpha = None
    for alpha in anisotropy_schedule:
        try:
            M_try = np.stack(
                [_blend_metric_anisotropy(metric_field_vert[i], alpha) for i in range(len(metric_field_vert))],
                axis=0,
            ).astype(np.float64)

            u, _ = _solve_crossfield_gl_compat(
                V=V,
                F=F,
                M_vert=M_try,
                frames=frames,
                mu=crossfield_mu,
                umbilic_smoothing=umbilic_smoothing,
                anisotropy_eps=anisotropy_eps,
                mu_min_ratio=mu_min_ratio,
                guidance_confidence=guidance_confidence,
            )

            theta = crossfield_angles_from_complex(u)
            phi1, phi2 = solve_parametrisation(
                V, F, theta, frames, gradient_size,
                integer_projection_iters=integer_projection_iters,
                integer_potential_iters=integer_potential_iters,
            )
            quadF = extract_quads_integer_grid(
                V, phi1, phi2,
                max_param_dist=float(max_param_dist),
                enforce_manifold=True,
                require_closed=require_closed_topology,
            )
            used_alpha = float(alpha)
            break
        except Exception as exc:
            last_exc = exc
            continue

    if quadF is None or u is None:
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Failed to generate quads under anisotropy continuation.")

    sing_info = detect_singularities_from_crossfield(V, F, u, frames=frames)

    if return_diagnostics:
        return V.copy(), quadF, {
            'crossfield_singularities': sing_info,
            'anisotropy_alpha': used_alpha,
        }
    return V.copy(), quadF


# ---------------------------------------------------------------------------
# IGL MIQ subprocess backend helpers
# ---------------------------------------------------------------------------

def _face_tangent_frames(V: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-face tangent frames.

    Returns:
        e1:  (M, 3) first tangent direction (along first edge)
        e2:  (M, 3) second tangent direction (perpendicular in tangent plane)
        n:   (M, 3) face normals
    """
    a = V[F[:, 0]]; b = V[F[:, 1]]; c = V[F[:, 2]]
    e1 = b - a
    e1 = e1 / (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-12)
    raw_n = np.cross(b - a, c - a)
    n = raw_n / (np.linalg.norm(raw_n, axis=1, keepdims=True) + 1e-12)
    e2 = np.cross(n, e1)
    e2 = e2 / (np.linalg.norm(e2, axis=1, keepdims=True) + 1e-12)
    return e1, e2, n


def _crossfield_to_face_directions(
    u_complex: np.ndarray,
    V: np.ndarray,
    F: np.ndarray,
    vert_frames: np.ndarray,
    M_vert: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert per-vertex 4-RoSy complex field  u = e^{4iθ}  to per-face
    principal directions PD1, PD2 (both F × 3) for the C++ MIQ binary.

    Steps:
      1. Extract per-vertex angle θ = arg(u) / 4.
      2. Lift to 3-D using vertex tangent frames.
      3. Average the 3-D direction over each face's three vertices.
      4. Project the average onto the face tangent plane and re-normalise.
      5. Build PD2 = 90° rotation of PD1 in the face tangent plane.
      6. If M_vert is supplied, scale PD1/PD2 by the per-face sqrt eigenvalues
         (s1, s2) of the metric tensors so that IGL MIQ receives anisotropy
         information through the vector magnitudes (libigl miq.h: "not unit").

    Anisotropy convention:
        |PD1| = s1 = sqrt(λ_max(M_face))  — high-curvature → short quads
        |PD2| = s2 = sqrt(λ_min(M_face))  — low-curvature  → long  quads

    Both vectors are divided by sqrt(s1 * s2) so their geometric mean magnitude
    equals 1, keeping the global quad density (gradient_size) unchanged while
    encoding the correct aspect ratio s1/s2.

    Args:
        u_complex:    (N,) complex array, u_i = e^{4i θ_i}.
        V:            (N, 3) mesh vertices.
        F:            (M, 3) triangle faces.
        vert_frames:  (N, 3, 2) per-vertex tangent frames [e1_v, e2_v].
        M_vert:       (N, 2, 2) optional per-vertex 2×2 metric tensors.
                      When provided, PD1/PD2 magnitudes encode anisotropy.

    Returns:
        PD1, PD2:  each (M, 3).
    """
    # Vertex-level angles
    theta_v = np.angle(u_complex.astype(complex)) / 4.0          # (N,)
    e1_v = vert_frames[:, :, 0]                                   # (N, 3)
    e2_v = vert_frames[:, :, 1]                                   # (N, 3)
    cos_t = np.cos(theta_v)                                       # (N,)
    sin_t = np.sin(theta_v)
    d1_v = cos_t[:, None] * e1_v + sin_t[:, None] * e2_v         # (N, 3)

    # Average over face corners with 4-RoSy branch disambiguation.
    #
    # Problem: each vertex has 4 equivalent directions {θ, θ+90°, θ+180°, θ+270°}.
    # Naively averaging the 3D vectors can cancel (e.g. (1,0,0)+(0,1,0)+(-1,0,0))
    # if vertices land on different branches, producing near-zero or wrong PD.
    #
    # Fix: use face vertex 0 as reference; for vertices 1 and 2, pick whichever
    # of the 4 branches has the highest dot-product with the reference.
    # The 4 branches of d are {d, R90·d, -d, -R90·d} where R90 = cross(n, ·).
    fe1, fe2, fn = _face_tangent_frames(V, F)

    d0 = d1_v[F[:, 0]]   # (M, 3) reference
    for slot in (1, 2):
        dc = d1_v[F[:, slot]]                        # (M, 3) candidate
        r90 = np.cross(fn, dc)                        # 90° CCW in face plane
        # Dot products for all 4 branches
        dot0 =  (d0 * dc  ).sum(axis=1)   # branch 0:   dc
        dot1 =  (d0 * r90 ).sum(axis=1)   # branch 1:  R90·dc
        dot2 = -dot0                       # branch 2:  -dc
        dot3 = -dot1                       # branch 3: -R90·dc
        # For each face pick the branch with highest alignment to d0
        best = np.stack([dot0, dot1, dot2, dot3], axis=1).argmax(axis=1)  # (M,)
        dirs = np.stack([dc, r90, -dc, -r90], axis=2)   # (M, 3, 4)
        d1_v_slot = dirs[np.arange(len(best)), :, best]  # (M, 3)
        if slot == 1:
            d1_s1 = d1_v_slot
        else:
            d1_s2 = d1_v_slot

    d1_avg = (d0 + d1_s1 + d1_s2) / 3.0  # (M, 3) — branch-consistent average

    # Project d1_avg onto face tangent plane
    c1 = (d1_avg * fe1).sum(axis=1)   # (M,)
    c2 = (d1_avg * fe2).sum(axis=1)
    norm = np.sqrt(c1 ** 2 + c2 ** 2) + 1e-12
    c1 /= norm; c2 /= norm

    PD1 = c1[:, None] * fe1 + c2[:, None] * fe2   # (M, 3) unit
    PD2 = -c2[:, None] * fe1 + c1[:, None] * fe2  # (M, 3) 90° CCW rotation, unit

    # ── Anisotropic scaling from per-vertex metric tensors ────────────────
    # IGL MIQ treats PD1/PD2 as "not unit" — their magnitudes set local quad
    # density per direction.  Scale by sqrt(λ_max), sqrt(λ_min) respectively,
    # then normalise by geometric mean so overall quad density is unchanged.
    if M_vert is not None:
        eigvals_v = np.linalg.eigvalsh(M_vert)          # (N, 2) ascending
        s1_v = np.sqrt(np.maximum(eigvals_v[:, 1], 1e-8))  # sqrt(λ_max) per vertex
        s2_v = np.sqrt(np.maximum(eigvals_v[:, 0], 1e-8))  # sqrt(λ_min) per vertex
        # Per-face mean
        s1_f = s1_v[F].mean(axis=1)   # (M,)
        s2_f = s2_v[F].mean(axis=1)   # (M,)
        # Normalise by geometric mean so |PD1|*|PD2| = 1 (density unchanged)
        gm   = np.sqrt(s1_f * s2_f + 1e-12)
        PD1  = PD1 * (s1_f / gm)[:, None]
        PD2  = PD2 * (s2_f / gm)[:, None]

    return PD1.astype(np.float64), PD2.astype(np.float64)


def _write_tri_obj(path: str, V: np.ndarray, F: np.ndarray) -> None:
    """Write a triangle-mesh OBJ that igl::readOBJ can parse."""
    with open(path, 'w') as fh:
        for v in V:
            fh.write(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n")
        for face in F:
            fh.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def _read_mesh_obj(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimal OBJ reader (v / f lines only).  Returns (V, F) as numpy arrays.
    Handles triangles and quads; face indices are 0-based.
    """
    verts: list = []
    faces: list = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'v':
                verts.append([float(x) for x in parts[1:4]])
            elif parts[0] == 'f':
                face = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                faces.append(face)
    if not faces:
        raise RuntimeError(f"No faces found in OBJ: {path}")
    V = np.array(verts, dtype=np.float64)
    F = np.array(faces, dtype=np.int64)
    return V, F


def miq_quadrangulate_igl(
    V: np.ndarray,
    F: np.ndarray,
    frame_field: np.ndarray,
    u_complex: np.ndarray,
    gradient_size: float = -1.0,
    stiffness: float = 5.0,
    max_param_dist: float = 0.7,
    direct_round: bool = True,
    miq_iter: int = 5,
    binary_path: Optional[str] = None,
    verbose: bool = True,
    M_vert: Optional[np.ndarray] = None,
    return_diagnostics: bool = False,
    stiffness_schedule: Optional[Tuple[float, ...]] = None,
    foldover_threshold: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run igl::copyleft::comiso::miq via the compiled C++ subprocess binary.

    Completely bypasses the Python/pybind11 ABI layer that causes
    std::bad_cast — Python and C++ live in separate processes.

    Args:
        V:             (N, 3) mesh vertices.
        F:             (M, 3) triangle faces.
        frame_field:   (N, 3, 2) per-vertex tangent frames.
        u_complex:     (N,) complex cross-field, u_i = e^{4i θ_i}.
        gradient_size: MIQ gradient scale (larger → fewer quads).
                       ≤0 = auto-compute from average edge length (recommended).
        stiffness:     MIQ stiffness weight.
        max_param_dist: max UV distance for integer-grid vertex snap.
        direct_round:  If True (default), use single-pass direct rounding instead of
                       iterative rounding.  Direct rounding is far more stable — it
                       avoids the FLIP spiral where iterative rounds accumulate UV
                       inversions until the domain collapses.  Set to False only when
                       you need maximum alignment quality on a high-quality cross-field.
        miq_iter:      rounding iterations (only used when direct_round=False).
        binary_path:   path to the compiled run_miq binary.
                       Defaults to cpp_miq/build/run_miq relative to project root.
        verbose:       if True, print C++ stdout.
        M_vert:        (N, 2, 2) optional per-vertex metric tensors (LCF 2D).
                       When provided, PD1/PD2 magnitudes are scaled by the metric
                       sqrt-eigenvalues so IGL MIQ produces anisotropic quads
                       aligned to the predicted curvature field.

    Returns:
        quadV: (N, 3)  — same vertex positions as V.
        quadF: (Q, 4)  — quad face indices.

    Raises:
        FileNotFoundError: if the binary is not found.
        RuntimeError:      if the C++ process exits with an error.
    """
    # Resolve binary
    if binary_path is None:
        binary_path = os.path.abspath(_DEFAULT_BINARY)
    if not os.path.isfile(binary_path):
        raise FileNotFoundError(
            f"run_miq binary not found at: {binary_path}\n"
            f"Build it first:\n"
            f"  bash cpp_miq/build.sh"
        )

    V  = np.asarray(V,  dtype=np.float64)
    F  = np.asarray(F,  dtype=np.int64)
    u_complex = np.asarray(u_complex, dtype=complex)

    # Auto-scale gradient_size from average edge length.
    #
    # Physical meaning: gradient_size = |∇_S φ|, the target UV gradient magnitude.
    # A larger value means φ changes faster across the surface → more integer
    # crossings → more (smaller) quads.
    #
    # Derivation:
    #   desired quad size in 3D = k × avg_el
    #   one integer step in UV  = 1
    #   → gradient_size = 1 / (k × avg_el)
    #
    # k=1.0 → quads ≈ 1 triangle edge (dense), gradient_size ≈ 1/avg_el ≈ 20 for
    # typical unit-box meshes — matches the C++ binary's default of 20.
    # Increase k for coarser (fewer) quads.
    if gradient_size <= 0:
        edge_vecs = np.vstack([
            V[F[:, 1]] - V[F[:, 0]],
            V[F[:, 2]] - V[F[:, 1]],
            V[F[:, 0]] - V[F[:, 2]],
        ])
        avg_el = float(np.linalg.norm(edge_vecs, axis=1).mean())
        k = 1.0          # quads per edge; raise (e.g. 3.0) for coarser output
        gradient_size = 1.0 / (k * avg_el)
        if verbose:
            print(f"[miq_igl] auto gradient_size = {gradient_size:.4g} "
                  f"(avg_edge = {avg_el:.4g}, k={k})")

    # Poincaré–Hopf pre-check on the raw GL field before MIQ combing.
    # This is only an approximate signal: libigl's authoritative check inside
    # run_miq is done after combing and can differ on some meshes.
    ph = verify_poincare_hopf(V, F, u_complex, frames=frame_field)
    if verbose:
        status = "ok" if ph['satisfied'] else "mismatch"
        print(f"[miq_igl] P-H {status}: Σidx={ph['sum_index']:.2f}, χ={ph['chi']}, "
              f"singularities={ph['singularities']['num_singular_faces']}")
        if not ph['satisfied']:
            print("[miq_igl]   (approx pre-check on raw field; continuing to C++ MIQ authoritative check)")

    # ── P3: adaptive stiffness retry ─────────────────────────────────────
    # Build the stiffness schedule to try.  If stiffness_schedule is given,
    # it completely overrides `stiffness`; otherwise a single attempt is made.
    if stiffness_schedule is not None:
        s_tries = tuple(stiffness_schedule)
    else:
        s_tries = (stiffness,)

    def _run_once(s: float) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Run the C++ MIQ binary once with stiffness s; return (V, F, diag)."""
        tmpdir = tempfile.mkdtemp(prefix='npaq_miq_')
        try:
            mesh_path  = os.path.join(tmpdir, 'mesh.obj')
            ureal_path = os.path.join(tmpdir, 'u_real.txt')
            uimag_path = os.path.join(tmpdir, 'u_imag.txt')
            out_path   = os.path.join(tmpdir, 'quad.obj')

            _write_tri_obj(mesh_path, V, F)
            np.savetxt(ureal_path, u_complex.real, fmt='%.9g')
            np.savetxt(uimag_path, u_complex.imag, fmt='%.9g')

            cmd = [
                binary_path,
                mesh_path, ureal_path, uimag_path, out_path,
                str(gradient_size), str(s),
                str(int(direct_round)), str(miq_iter),
            ]

            if M_vert is not None:
                PD1, PD2 = _crossfield_to_face_directions(
                    u_complex, V, F, frame_field, M_vert=M_vert
                )
                pd1_path = os.path.join(tmpdir, 'pd1.txt')
                pd2_path = os.path.join(tmpdir, 'pd2.txt')
                np.savetxt(pd1_path, PD1, fmt='%.9g')
                np.savetxt(pd2_path, PD2, fmt='%.9g')
                cmd += [pd1_path, pd2_path]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    "run_miq timed out after 120 s. The cross-field likely has too many "
                    "singularities. Try a better checkpoint or reduce gradient_size."
                )

            if verbose and result.stdout:
                print(result.stdout, end='')
            if result.returncode != 0:
                raise RuntimeError(
                    f"run_miq failed (exit {result.returncode}):\n{result.stderr}"
                )

            qV, qF = _read_mesh_obj(out_path)
            diag   = _parse_run_miq_diagnostics(result.stdout)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        return qV, qF, diag

    # Try each stiffness; keep the result with the fewest fold-overs.
    best_quadV = best_quadF = best_diag = None
    best_folds = float('inf')
    best_stiffness = s_tries[0]

    for s_try in s_tries:
        try:
            qV, qF, diag = _run_once(s_try)
        except RuntimeError as exc:
            if verbose:
                print(f"[miq_igl] stiffness={s_try} failed: {exc}")
            continue

        n_folds = _count_foldovers(qV, qF)
        if verbose and len(s_tries) > 1:
            print(f"[miq_igl] stiffness={s_try:.1f} → {qF.shape[0]} quads, "
                  f"{n_folds} fold-overs")

        if n_folds < best_folds:
            best_quadV, best_quadF, best_diag = qV, qF, diag
            best_folds = n_folds
            best_stiffness = s_try

        if n_folds <= foldover_threshold:
            break   # good enough — no need to try higher stiffness

    if best_quadV is None:
        raise RuntimeError("All stiffness attempts failed. Check cross-field quality.")

    quadV, quadF, miq_diag = best_quadV, best_quadF, best_diag

    # ── Post-processing chain ────────────────────────────────────────────
    if quadF.ndim == 2 and quadF.shape[1] == 4 and quadF.shape[0] > 0:
        # P2: pull fold-over vertices toward quad centroid
        quadV, n_repaired = _fix_foldover_quads(quadV, quadF)
        if verbose and n_repaired > 0:
            print(f"[miq_igl] fold-over post-fix: repaired {n_repaired} quads")
        miq_diag['postfix_foldovers_repaired'] = n_repaired
        miq_diag['stiffness_used'] = float(best_stiffness)

        # Remove near-zero-area quads (degenerate "triangles" near singularities)
        n_before = quadF.shape[0]
        quadF = _remove_degenerate_quads(quadV, quadF)
        n_degen = n_before - quadF.shape[0]
        if verbose and n_degen > 0:
            print(f"[miq_igl] removed {n_degen} degenerate quads")
        miq_diag['degenerate_quads_removed'] = n_degen

        # Prune to closed manifold subset — only if result keeps ≥50% of faces.
        # (C++ MIQ may output open-boundary meshes; aggressive pruning can empty them.)
        n_before = quadF.shape[0]
        if n_before > 0:
            candidate = _prune_quads_to_closed_manifold_subset(quadF)
            if candidate.shape[0] >= 0.5 * n_before:
                n_pruned = n_before - candidate.shape[0]
                quadF = candidate
                if verbose and n_pruned > 0:
                    print(f"[miq_igl] manifold pruning removed {n_pruned} quads")
                miq_diag['manifold_pruned_quads'] = n_pruned
            else:
                if verbose:
                    print(f"[miq_igl] manifold pruning skipped "
                          f"(would remove {n_before - candidate.shape[0]}/{n_before} quads)")
                miq_diag['manifold_pruned_quads'] = 0

        # Compact: remove orphan vertices
        quadV, quadF = _compact_quad_mesh(quadV, quadF)

    if return_diagnostics:
        return quadV, quadF, miq_diag
    return quadV, quadF


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def initial_quad_mesh_from_pointcloud(
    points: np.ndarray,
    metric_field: np.ndarray,
    metric_basis: Optional[np.ndarray] = None,
    guidance_confidence: Optional[np.ndarray] = None,
    guidance_override: Optional[np.ndarray] = None,
    guidance_override_weight: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    poisson_depth: int = 6,
    gradient_size: float = 1.0,
    crossfield_mu: float = 10.0,
    umbilic_smoothing: bool = True,
    anisotropy_eps: float = 0.03,
    mu_min_ratio: float = 0.05,
    integer_constraints: bool = True,
    require_closed_topology: bool = False,
    anisotropy_schedule: Optional[Tuple[float, ...]] = None,
    integer_projection_iters: int = 2,
    integer_potential_iters: int = 2,
    max_param_dist: float = 0.7,
    return_diagnostics: bool = False,
    use_igl_backend: bool = False,
    igl_binary_path: Optional[str] = None,
    igl_gradient_size: float = 20.0,
    igl_stiffness: float = 5.0,
    igl_direct_round: bool = True,
    igl_miq_iter: int = 5,
    stiffness_schedule: Optional[Tuple[float, ...]] = None,
    foldover_threshold: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full pipeline:  Point cloud  →  Poisson mesh  →  Cross-field  →  Quad mesh.

    Args:
        points:             (N, 3) input point cloud.
        metric_field:       (N, 2, 2) per-point metric tensors. If metric_basis
                            is supplied, tensors are interpreted in the input
                            point LCF and reprojected into the reconstructed
                            Poisson mesh frames before quadrangulation.
        metric_basis:       Optional (N, 3, 3) input LCF basis from prediction.
        normals:            (N, 3) normals (estimated if None).
        poisson_depth:      Octree depth for Poisson reconstruction.
        gradient_size:      Controls quad density for the Python GL+param pipeline.
        crossfield_mu:      GL alignment weight.
        integer_constraints: Whether to run quad extraction (Python path only).
        use_igl_backend:    If True, use the C++ IGL MIQ subprocess instead of the
                            Python GL+param pipeline.  Requires a compiled run_miq
                            binary (see cpp_miq/build.sh).
        igl_binary_path:    Path to the run_miq binary; None = auto-detect.
        igl_gradient_size:  gradient_size passed to C++ MIQ (typically 10–30).
        igl_stiffness:      stiffness passed to C++ MIQ.

    Returns:
        quadV:  (Nv, 3) vertex positions of the quad mesh.
        quadF:  (Nf, 4) quad face indices.
    """
    if normals is None:
        from .feature_lines import estimate_normals
        normals = estimate_normals(points)

    # 1. Surface reconstruction
    V_tri, F_tri = poisson_surface_reconstruction(points, normals, depth=poisson_depth)

    # 1b. Ensure consistent outward winding relative to input point cloud normals
    V_tri, F_tri = _orient_mesh_outward(V_tri, F_tri, points, normals)

    # 2. Tangent frames on the reconstructed mesh
    frames = compute_vertex_frames(V_tri, F_tri)

    # 3. Transfer metric to mesh vertices.  Point predictions live in the input
    # LCF, while Poisson reconstruction creates new vertex frames.  Lift to a
    # world tensor first, transfer, then reproject into the Poisson frames.
    if metric_basis is not None:
        metric_world = _lift_lcf_metric_to_world(metric_field, metric_basis)
        metric_world_vert = _transfer_metric_to_mesh(points, metric_world, V_tri)
        M_vert = _project_world_metric_to_frames(metric_world_vert, frames)
    else:
        M_vert = _transfer_metric_to_mesh(points, metric_field, V_tri)
    conf_vert = None
    if guidance_confidence is not None:
        conf_vert = _transfer_scalar_to_mesh(
            points, np.asarray(guidance_confidence).reshape(-1), V_tri
        )
    override_vert = None
    override_weight_vert = None
    if guidance_override is not None:
        tree = KDTree(points)
        _, idx = tree.query(V_tri)
        override_vert = np.asarray(guidance_override).reshape(-1)[idx]
        if guidance_override_weight is not None:
            override_weight_vert = np.asarray(guidance_override_weight).reshape(-1)[idx]

    if use_igl_backend:
        # 4a. Cross-field (GL solve) then IGL MIQ via C++ subprocess
        u, solve_info = _solve_crossfield_gl_compat(
            V=V_tri,
            F=F_tri,
            M_vert=M_vert.astype(np.float64),
            frames=frames,
            mu=crossfield_mu,
            umbilic_smoothing=umbilic_smoothing,
            anisotropy_eps=anisotropy_eps,
            mu_min_ratio=mu_min_ratio,
            guidance_confidence=conf_vert,
            guidance_override=override_vert,
            guidance_override_weight=override_weight_vert,
        )
        out = miq_quadrangulate_igl(
            V_tri, F_tri,
            frame_field=frames,
            u_complex=u,
            M_vert=M_vert,          # ← pass metric so PD magnitudes encode anisotropy
            gradient_size=igl_gradient_size,
            stiffness=igl_stiffness,
            direct_round=bool(igl_direct_round),
            miq_iter=int(igl_miq_iter),
            binary_path=igl_binary_path,
            stiffness_schedule=stiffness_schedule,
            foldover_threshold=int(foldover_threshold),
            return_diagnostics=return_diagnostics,
        )
        if return_diagnostics:
            quadV, quadF, miq_diag = out
        else:
            quadV, quadF = out
        if require_closed_topology:
            quadF = _prune_quads_to_closed_manifold_subset(quadF)
            if quadF.size == 0:
                raise RuntimeError(
                    "Closed-topology extraction failed on IGL backend: "
                    "no boundary-free manifold subset remains."
                )
        if return_diagnostics:
            sing_info = detect_singularities_from_crossfield(V_tri, F_tri, u, frames=frames)
            ph_diag = miq_diag.get('poincare_hopf', solve_info.get('poincare_hopf', {}))
            return quadV, quadF, {
                'crossfield_singularities': sing_info,
                'crossfield_filter': solve_info.get('singularity_filter', {}),
                'poincare_hopf': ph_diag,
                'holonomy': solve_info.get('holonomy', {}),
                'flow_conservation': solve_info.get('flow_conservation', {}),
                'miq_extraction': miq_diag,
            }
    else:
        # 4b. Python GL cross-field + Poisson parametrisation + integer-grid
        out = miq_quadrangulate(
            V_tri, F_tri,
            frame_field=frames,
            metric_field_vert=M_vert,
            guidance_confidence=conf_vert,
            gradient_size=gradient_size,
            crossfield_mu=crossfield_mu,
            umbilic_smoothing=umbilic_smoothing,
            anisotropy_eps=anisotropy_eps,
            mu_min_ratio=mu_min_ratio,
            integer_constraints=integer_constraints,
            require_closed_topology=require_closed_topology,
            anisotropy_schedule=anisotropy_schedule,
            integer_projection_iters=integer_projection_iters,
            integer_potential_iters=integer_potential_iters,
            max_param_dist=max_param_dist,
            return_diagnostics=return_diagnostics,
        )
        if return_diagnostics:
            return out
        quadV, quadF = out

    return quadV, quadF


def initial_quad_mesh_from_mesh(
    V_tri: np.ndarray,
    F_tri: np.ndarray,
    metric_field_on_vertices: np.ndarray,
    guidance_confidence: Optional[np.ndarray] = None,
    guidance_override: Optional[np.ndarray] = None,
    guidance_override_weight: Optional[np.ndarray] = None,
    singularity_mask: Optional[np.ndarray] = None,
    singularity_indices: Optional[np.ndarray] = None,
    protected_mask: Optional[np.ndarray] = None,
    guidance_override_reliability_floor: float = 0.9,
    guidance_override_anchor_eps: float = 0.05,
    singularity_cancel_distance_ratio: float = 2.5,
    singularity_cancel_smoothing_iters: int = 4,
    singularity_cancel_max_rounds: int = 2,
    repair_high_order: bool = True,
    high_order_repair_max_rounds: int = 1,
    high_order_repair_expand_rounds: int = 1,
    high_order_repair_smoothing_iters: int = 3,
    repair_flow: bool = True,
    flow_repair_max_rounds: int = 1,
    flow_repair_expand_rounds: int = 0,
    flow_repair_smoothing_iters: int = 3,
    gradient_size: float = 1.0,
    crossfield_mu: float = 10.0,
    umbilic_smoothing: bool = True,
    anisotropy_eps: float = 0.03,
    mu_min_ratio: float = 0.05,
    integer_constraints: bool = True,
    require_closed_topology: bool = False,
    anisotropy_schedule: Optional[Tuple[float, ...]] = None,
    integer_projection_iters: int = 2,
    integer_potential_iters: int = 2,
    max_param_dist: float = 0.7,
    return_diagnostics: bool = False,
    use_igl_backend: bool = False,
    igl_binary_path: Optional[str] = None,
    igl_gradient_size: float = 20.0,
    igl_stiffness: float = 5.0,
    igl_direct_round: bool = True,
    igl_miq_iter: int = 5,
    stiffness_schedule: Optional[Tuple[float, ...]] = None,
    foldover_threshold: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mesh-input variant of the quadrangulation pipeline.

    Unlike `initial_quad_mesh_from_pointcloud`, this path skips Poisson
    reconstruction and uses the provided triangle mesh directly. This avoids
    geometry drift when the input is already a clean mesh.
    """
    V_tri = np.asarray(V_tri, dtype=np.float64)
    F_tri = np.asarray(F_tri, dtype=np.int64)
    M_vert = np.asarray(metric_field_on_vertices, dtype=np.float64)

    if F_tri.ndim != 2 or F_tri.shape[1] != 3:
        raise ValueError(f"initial_quad_mesh_from_mesh expects triangle faces (M,3), got {F_tri.shape}")
    if M_vert.shape[0] != V_tri.shape[0]:
        raise ValueError(
            "metric_field_on_vertices must align with mesh vertices: "
            f"{M_vert.shape[0]} != {V_tri.shape[0]}"
        )

    frames = compute_vertex_frames(V_tri, F_tri)

    if use_igl_backend:
        u, solve_info = _solve_crossfield_gl_compat(
            V=V_tri,
            F=F_tri,
            M_vert=M_vert.astype(np.float64),
            frames=frames,
            mu=crossfield_mu,
            umbilic_smoothing=umbilic_smoothing,
            anisotropy_eps=anisotropy_eps,
            mu_min_ratio=mu_min_ratio,
            guidance_confidence=guidance_confidence,
            guidance_override=guidance_override,
            guidance_override_weight=guidance_override_weight,
            singularity_mask=singularity_mask,
            singularity_indices=singularity_indices,
            protected_mask=protected_mask,
            guidance_override_reliability_floor=guidance_override_reliability_floor,
            guidance_override_anchor_eps=guidance_override_anchor_eps,
            singularity_cancel_distance_ratio=singularity_cancel_distance_ratio,
            singularity_cancel_smoothing_iters=singularity_cancel_smoothing_iters,
            singularity_cancel_max_rounds=singularity_cancel_max_rounds,
            repair_high_order=repair_high_order,
            high_order_repair_max_rounds=high_order_repair_max_rounds,
            high_order_repair_expand_rounds=high_order_repair_expand_rounds,
            high_order_repair_smoothing_iters=high_order_repair_smoothing_iters,
            repair_flow=repair_flow,
            flow_repair_max_rounds=flow_repair_max_rounds,
            flow_repair_expand_rounds=flow_repair_expand_rounds,
            flow_repair_smoothing_iters=flow_repair_smoothing_iters,
        )
        out = miq_quadrangulate_igl(
            V_tri, F_tri,
            frame_field=frames,
            u_complex=u,
            M_vert=M_vert,
            gradient_size=igl_gradient_size,
            stiffness=igl_stiffness,
            direct_round=bool(igl_direct_round),
            miq_iter=int(igl_miq_iter),
            binary_path=igl_binary_path,
            stiffness_schedule=stiffness_schedule,
            foldover_threshold=foldover_threshold,
            return_diagnostics=return_diagnostics,
        )
        if return_diagnostics:
            quadV, quadF, miq_diag = out
        else:
            quadV, quadF = out
        if require_closed_topology:
            quadF = _prune_quads_to_closed_manifold_subset(quadF)
            if quadF.size == 0:
                raise RuntimeError(
                    "Closed-topology extraction failed on IGL backend: "
                    "no boundary-free manifold subset remains."
                )
        if return_diagnostics:
            sing_info = detect_singularities_from_crossfield(V_tri, F_tri, u, frames=frames)
            ph_diag = miq_diag.get('poincare_hopf', solve_info.get('poincare_hopf', {}))
            return quadV, quadF, {
                'crossfield_singularities': sing_info,
                'crossfield_filter': solve_info.get('singularity_filter', {}),
                'poincare_hopf': ph_diag,
                'holonomy': solve_info.get('holonomy', {}),
                'flow_conservation': solve_info.get('flow_conservation', {}),
                'miq_extraction': miq_diag,
            }
    else:
        out = miq_quadrangulate(
            V_tri, F_tri,
            frame_field=frames,
            metric_field_vert=M_vert,
            guidance_confidence=guidance_confidence,
            gradient_size=gradient_size,
            crossfield_mu=crossfield_mu,
            umbilic_smoothing=umbilic_smoothing,
            anisotropy_eps=anisotropy_eps,
            mu_min_ratio=mu_min_ratio,
            integer_constraints=integer_constraints,
            require_closed_topology=require_closed_topology,
            anisotropy_schedule=anisotropy_schedule,
            integer_projection_iters=integer_projection_iters,
            integer_potential_iters=integer_potential_iters,
            max_param_dist=max_param_dist,
            return_diagnostics=return_diagnostics,
        )
        if return_diagnostics:
            return out
        quadV, quadF = out

    return quadV, quadF
