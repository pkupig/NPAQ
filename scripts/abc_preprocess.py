#!/usr/bin/env python
"""
Preprocess ABC dataset OBJ/STL meshes into .npz patches for ABCDataset training.

For each mesh:
  1. Sample N_points uniformly from the surface (trimesh)
  2. Estimate normals (open3d, MST-consistent orientation)
  3. Compute principal curvatures k1,k2 and directions d1,d2 via local quadratic fitting
  4. Build 2D LCF metric tensor: M_2d = R^T @ M_3d @ R
     where M_3d = w1*d1*d1^T + w2*d2*d2^T and R = LCF tangent basis
  5. Save .npz: {points(N,3), normals(N,3), metric(N,2,2),
                 principal_dir1(N,3), principal_dir2(N,3)}

Usage:
  # Preprocess chunk 0000 for training, 0001 for validation
  python scripts/abc_preprocess.py \\
      --input  data/abc/0000 \\
      --output data/abc_preprocessed/train \\
      --n_points 4096 --k_fit 20 --max_files 5000

  python scripts/abc_preprocess.py \\
      --input  data/abc/0001 \\
      --output data/abc_preprocessed/val \\
      --n_points 4096 --k_fit 20 --max_files 1000
"""

import argparse
import os
import sys
import json
import traceback
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset.synthesis import construct_target_metric


# ---------------------------------------------------------------------------
# Principal curvature estimation via local quadratic fitting
# ---------------------------------------------------------------------------

def _estimate_principal_curvature(
    points: np.ndarray,
    normals: np.ndarray,
    k: int = 20,
    rho: float = 1.0,
    epsilon: float = 0.05,
):
    """
    Vectorised principal curvature estimation via the Weingarten map
    (shape operator from normal variations).

    Replaces the old height-function quadratic fitting loop with a fully
    vectorised numpy implementation that:
      1. Uses Duff et al. (2017) stable tangent frames — consistent with
         crossfield.py and _build_2d_metric (eliminates frame discontinuities).
      2. Fits the shape operator W = [[a,b],[b,c]] via normal variation:
             W · [ex, ey]^T = -[dnx, dny]^T  (Weingarten map)
         This uses the full normal field, not just the scalar height, giving
         better accuracy for clean meshes (e.g. Stanford models).
      3. No Python loop — O(N·k) vectorised normal equations.

    Returns:
        k1, k2: (N,) principal curvatures  (|k1| ≥ |k2| on average)
        d1, d2: (N,3) unit principal directions in 3D
    """
    N = len(points)
    tree = KDTree(points)
    _, idx = tree.query(points, k=k + 1)
    idx = idx[:, 1:]   # (N, k) — exclude self

    nv = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)

    # ── Duff et al. (2017) stable tangent frames ──────────────────────────
    nx, ny, nz = nv[:, 0], nv[:, 1], nv[:, 2]
    sg = np.where(nz >= 0.0, 1.0, -1.0)
    av = -1.0 / (sg + nz)
    bv = nx * ny * av
    e1 = np.stack([1.0 + sg*nx**2*av,  sg*bv,        -sg*nx], axis=1)  # (N, 3)
    e2 = np.stack([bv,                  sg + ny**2*av, -ny  ], axis=1)  # (N, 3)
    e1 /= (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-12)
    e2 /= (np.linalg.norm(e2, axis=1, keepdims=True) + 1e-12)

    # ── Gather neighbor data ──────────────────────────────────────────────
    neigh_pts = points[idx]   # (N, k, 3)
    neigh_nrm = nv[idx]       # (N, k, 3)

    ni_exp = nv[:, None, :]   # (N, 1, 3)

    # Edge vectors projected to tangent plane
    edge = neigh_pts - points[:, None, :]                                # (N, k, 3)
    edge_t = edge - (edge * ni_exp).sum(-1, keepdims=True) * ni_exp     # (N, k, 3)
    et_len = np.linalg.norm(edge_t, axis=-1) + 1e-12                    # (N, k)

    # Normalised edge direction in tangent frame
    ex = (edge_t * e1[:, None, :]).sum(-1) / et_len   # (N, k)
    ey = (edge_t * e2[:, None, :]).sum(-1) / et_len   # (N, k)

    # Normal variation: dn_t / edge_len  (shape operator estimate)
    dn   = neigh_nrm - nv[:, None, :]                                    # (N, k, 3)
    dn_t = dn - (dn * ni_exp).sum(-1, keepdims=True) * ni_exp           # (N, k, 3)
    dnx  = (dn_t * e1[:, None, :]).sum(-1) / et_len                     # (N, k)
    dny  = (dn_t * e2[:, None, :]).sum(-1) / et_len                     # (N, k)

    # ── Normal equations for W = [[a,b],[b,c]] ───────────────────────────
    # Per-neighbor design rows: [ex, ey, 0] and [0, ex, ey]
    # A^T A = [[Σex²,       Σex·ey,          0      ],
    #          [Σex·ey,     Σ(ex²+ey²),      Σex·ey ],
    #          [0,          Σex·ey,           Σey²   ]]
    # A^T b = [-Σex·dnx,  -(Σey·dnx + Σex·dny),  -Σey·dny]
    Su2 = (ex * ex).sum(-1)   # (N,)
    Sv2 = (ey * ey).sum(-1)   # (N,)
    Suv = (ex * ey).sum(-1)   # (N,)
    z   = np.zeros(N)

    ATA = np.stack([
        np.stack([Su2,       Suv,        z        ], axis=-1),
        np.stack([Suv,       Su2 + Sv2,  Suv      ], axis=-1),
        np.stack([z,         Suv,        Sv2      ], axis=-1),
    ], axis=-2)   # (N, 3, 3)

    ATb = np.stack([
        -(ex * dnx).sum(-1),
        -((ey * dnx) + (ex * dny)).sum(-1),
        -(ey * dny).sum(-1),
    ], axis=-1)   # (N, 3)

    # Solve (ridge regression for numerical stability)
    reg = 1e-8 * np.eye(3, dtype=np.float64)
    coeffs = np.linalg.solve(ATA.astype(np.float64) + reg,
                              ATb.astype(np.float64))   # (N, 3)

    a_c, b_c, c_c = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2]

    # ── Eigendecomposition of shape operator W ────────────────────────────
    II = np.stack([
        np.stack([a_c, b_c], axis=-1),
        np.stack([b_c, c_c], axis=-1),
    ], axis=-2)   # (N, 2, 2)

    eigvals, eigvecs = np.linalg.eigh(II)   # (N, 2), (N, 2, 2); ascending

    k1_arr = eigvals[:, 1]                  # max curvature
    k2_arr = eigvals[:, 0]                  # min curvature

    ev1 = eigvecs[:, :, 1]   # (N, 2) — eigenvector for k1
    ev2 = eigvecs[:, :, 0]   # (N, 2) — eigenvector for k2

    d1_arr = ev1[:, 0:1] * e1 + ev1[:, 1:2] * e2   # (N, 3)
    d2_arr = ev2[:, 0:1] * e1 + ev2[:, 1:2] * e2   # (N, 3)
    d1_arr /= (np.linalg.norm(d1_arr, axis=1, keepdims=True) + 1e-12)
    d2_arr /= (np.linalg.norm(d2_arr, axis=1, keepdims=True) + 1e-12)

    return k1_arr, k2_arr, d1_arr, d2_arr


def _smooth_direction_field(
    d1: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    k_smooth: int = 10,
) -> np.ndarray:
    """
    Smooth a line-field d1 (N, 3) via tensor-voting / outer-product averaging.

    For each point i:
      1. Accumulate M_i = Σ_j d1[j] d1[j]^T over the k nearest neighbors.
      2. Extract the principal eigenvector of M_i as the smoothed direction.
      3. Re-project onto the tangent plane to ensure tangency.

    The 180° ambiguity of d1 is handled automatically (outer products are
    sign-invariant).

    Returns:
        d1_smooth: (N, 3) smoothed unit direction field.
    """
    N = len(points)
    tree = KDTree(points)
    _, idx = tree.query(points, k=k_smooth + 1)
    idx = idx[:, 1:]   # (N, k_smooth)

    # Outer products (N, 3, 3)
    M = d1[:, :, None] * d1[:, None, :]   # (N, 3, 3)

    # Average over KNN including self
    M_neigh = M[idx]                      # (N, k_smooth, 3, 3)
    M_self  = M[:, None, :, :]            # (N, 1, 3, 3)
    M_avg   = np.concatenate([M_self, M_neigh], axis=1).mean(axis=1)  # (N, 3, 3)

    # Extract principal eigenvector
    _, eigvecs = np.linalg.eigh(M_avg)    # (N, 3), (N, 3, 3); ascending
    d1_s = eigvecs[:, :, -1]              # (N, 3) — largest eigenvalue

    # Re-project onto tangent plane
    nv = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
    d1_s = d1_s - (d1_s * nv).sum(-1, keepdims=True) * nv
    d1_s /= (np.linalg.norm(d1_s, axis=1, keepdims=True) + 1e-12)

    return d1_s


def _build_2d_metric(
    points: np.ndarray,
    normals: np.ndarray,
    k1: np.ndarray,
    k2: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
    k_lcf: int = 32,
    rho: float = 1.0,
    epsilon: float = 0.05,
):
    """
    Build per-point 2D LCF metric tensors M_2d = R^T @ M_3d @ R.
    M_3d = w1*d1*d1^T + w2*d2*d2^T (same formula as SyntheticDataset).

    The LCF basis is built from normals (same logic as compute_local_canonical_frame
    when normals are provided), vectorized over all N points.

    Returns:
        metric_2d: (N, 2, 2)
        basis_all: (N, 3, 3) full LCF frames
    """
    N = len(points)

    # 3D metric tensor per point
    M_3d = construct_target_metric(d1, d2, k1, k2, rho=rho, epsilon=epsilon)  # (N, 3, 3)

    # Build LCF basis vectorized from normals (replicates lcf.py logic)
    # e3 = unit normal
    e3 = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)  # (N, 3)

    # Build e1/e2 with Duff et al. (2017) stable ONB formula — same as compute_vertex_frames
    # and lcf.py (normals branch) so preprocessing and inference frames are identical.
    nx_v, ny_v, nz_v = e3[:, 0], e3[:, 1], e3[:, 2]
    sign_v = np.where(nz_v >= 0.0, 1.0, -1.0)
    a_v    = -1.0 / (sign_v + nz_v)
    b_v    = nx_v * ny_v * a_v
    e1 = np.stack([1.0 + sign_v * nx_v**2 * a_v,  sign_v * b_v,  -sign_v * nx_v], axis=1)
    e2 = np.stack([b_v,  sign_v + ny_v**2 * a_v,  -ny_v], axis=1)
    e1 /= (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-12)
    e2 /= (np.linalg.norm(e2, axis=1, keepdims=True) + 1e-12)

    # basis[i] has columns [e1, e2, e3], shape (N, 3, 3)
    basis_all = np.stack([e1, e2, e3], axis=2)

    # R = first two columns of basis: (N, 3, 2)
    R = basis_all[:, :, :2]

    # M_2d[i] = R[i].T @ M_3d[i] @ R[i]   (vectorized einsum)
    metric_2d = np.einsum('nij,njk,nkl->nil', R.transpose(0, 2, 1), M_3d, R)  # (N, 2, 2)

    return metric_2d, basis_all


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(
    mesh_path: str,
    n_points: int,
    k_fit: int,
    k_lcf: int,
    rho: float,
    epsilon: float,
    seed: int,
) -> dict | None:
    """
    Load mesh, sample points, compute normals + curvature + 2D metrics.
    Returns dict with numpy arrays or None on failure.
    """
    try:
        import trimesh
    except ImportError:
        raise RuntimeError("trimesh is required: pip install trimesh")

    # Try loading as a mesh first; fall back to point-cloud loading for PLY/PCD.
    pts = None
    try:
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
        if (hasattr(mesh, 'vertices') and len(mesh.vertices) >= 10 and
                hasattr(mesh, 'faces') and len(mesh.faces) >= 4):
            # True triangle mesh — sample uniformly from surface
            np.random.seed(seed)
            actual_n = min(n_points, len(mesh.vertices))
            try:
                pts, _ = trimesh.sample.sample_surface(mesh, actual_n)
                pts = pts.astype(np.float32)
            except Exception:
                idx = np.random.choice(len(mesh.vertices), actual_n, replace=False)
                pts = mesh.vertices[idx].astype(np.float32)
    except Exception:
        pass

    if pts is None:
        # Point-cloud PLY/PCD file: load raw vertices directly
        try:
            pc = trimesh.load(mesh_path, process=False)
            raw = np.asarray(pc.vertices, dtype=np.float32)
            if len(raw) < 10:
                return None
            np.random.seed(seed)
            actual_n = min(n_points, len(raw))
            idx = np.random.choice(len(raw), actual_n, replace=False)
            pts = raw[idx]
        except Exception:
            return None

    # Strip NaN / Inf points before any further processing
    valid = np.isfinite(pts).all(axis=1)
    if valid.sum() < 10:
        return None
    pts = pts[valid]

    # ── Normalise to unit bounding-box ───────────────────────────────────
    # Curvature k ∝ 1/scale, so unnormalised meshes yield metric values
    # that vary by orders of magnitude across shapes (e.g. armadillo: frob≈0.04,
    # bunny2: frob≈5275 before fix).  Normalising here keeps all curvatures
    # in a comparable range and prevents any single mesh from dominating training.
    pts = pts.astype(np.float64)
    pts -= pts.mean(axis=0)
    bbox_diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) + 1e-12
    pts /= bbox_diag
    pts = pts.astype(np.float32)

    # Estimate normals (open3d fast path)
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_fit))
        try:
            pcd.orient_normals_consistent_tangent_plane(k=min(k_fit, 15))
        except Exception:
            pass   # qhull can fail on near-coplanar/degenerate point sets
        normals = np.asarray(pcd.normals, dtype=np.float32)
        # Replace any NaN normals (can happen if orientation partially failed)
        bad = ~np.isfinite(normals).all(axis=1)
        if bad.any():
            normals[bad] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    except ImportError:
        from src.geometry.feature_lines import estimate_normals
        normals = estimate_normals(pts, k=k_fit).astype(np.float32)

    # Principal curvature estimation (vectorised shape operator)
    k1, k2, d1, d2 = _estimate_principal_curvature(
        pts, normals, k=k_fit, rho=rho, epsilon=epsilon
    )

    # Post-smooth d1 to reduce high-frequency noise in principal directions.
    # d2 is recomputed from d1 × n to preserve orthogonality.
    nv_pts = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
    d1 = _smooth_direction_field(d1, pts, nv_pts, k_smooth=k_fit)
    d2 = np.cross(nv_pts, d1)
    d2 /= (np.linalg.norm(d2, axis=1, keepdims=True) + 1e-12)

    # ── Normalise curvature scale ────────────────────────────────────────
    # Stanford (and other real) meshes have curvatures 5-50× larger than
    # synthetic shapes (cylinder k=1, sphere k=1) even after unit-bbox
    # normalisation, because they have fine geometric detail.
    # Scale k1, k2 so that the 95th-percentile |k| = 1 within each mesh.
    # This keeps the ANISOTROPY RATIO (k1/k2) unchanged while mapping the
    # absolute curvature magnitude to the same range as synthetic data,
    # so that metric tensor Frobenius norms and the Log-Euclidean loss
    # are directly comparable across dataset types.
    k_ref = np.percentile(np.abs(np.concatenate([k1, k2])), 95) + 1e-6
    k1 = k1 / k_ref
    k2 = k2 / k_ref

    # 2D LCF metric tensors
    metric_2d, _ = _build_2d_metric(
        pts, normals, k1, k2, d1, d2, k_lcf=k_lcf, rho=rho, epsilon=epsilon
    )

    return {
        'points':         pts.astype(np.float32),
        'normals':        normals.astype(np.float32),
        'metric':         metric_2d.astype(np.float32),
        'principal_dir1': d1.astype(np.float32),
        'principal_dir2': d2.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess ABC dataset for NPAQ training')
    parser.add_argument('--input',     required=True, help='Directory of raw ABC OBJ/STL files')
    parser.add_argument('--output',    required=True, help='Output directory for .npz files')
    parser.add_argument('--n_points',  type=int, default=4096, help='Surface samples per mesh')
    parser.add_argument('--k_fit',     type=int, default=20,   help='Neighbours for curvature fitting')
    parser.add_argument('--k_lcf',     type=int, default=32,   help='Neighbours for LCF computation')
    parser.add_argument('--rho',       type=float, default=1.0,  help='Metric scale factor ρ')
    parser.add_argument('--epsilon',   type=float, default=0.05, help='Regularisation ε for flat regions')
    parser.add_argument('--max_files', type=int, default=None, help='Max number of meshes to process')
    parser.add_argument('--seed',      type=int, default=42,   help='Random seed for sampling')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Collect all mesh files
    extensions = {'.obj', '.stl', '.off', '.ply'}
    mesh_files = sorted([
        str(p) for p in Path(args.input).rglob('*')
        if p.suffix.lower() in extensions
    ])
    if args.max_files:
        mesh_files = mesh_files[:args.max_files]

    if not mesh_files:
        print(f"No mesh files found in {args.input}")
        return

    print(f"Found {len(mesh_files)} mesh files. Processing...")

    ok_files = []
    failed = 0
    for i, fpath in enumerate(tqdm(mesh_files)):
        out_name = Path(fpath).stem + '.npz'
        out_path = os.path.join(args.output, out_name)
        if os.path.exists(out_path):
            ok_files.append(out_path)
            continue

        try:
            result = process_file(
                fpath,
                n_points=args.n_points,
                k_fit=args.k_fit,
                k_lcf=args.k_lcf,
                rho=args.rho,
                epsilon=args.epsilon,
                seed=args.seed + i,
            )
        except Exception:
            result = None
        if result is None:
            failed += 1
            continue

        np.savez_compressed(out_path, **result)
        ok_files.append(out_path)

    # Write index file for ABCDataset
    index_path = os.path.join(args.output, '..', f'{os.path.basename(args.output)}_index.json')
    index_path = os.path.normpath(index_path)
    with open(index_path, 'w') as f:
        json.dump({'files': ok_files}, f, indent=2)

    print(f"\nDone. {len(ok_files)} files saved to {args.output}, {failed} failed.")
    print(f"Index written to {index_path}")


if __name__ == '__main__':
    main()
