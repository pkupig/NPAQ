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
from sklearn.neighbors import KDTree
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
    Estimate principal curvatures k1, k2 and directions d1, d2 at every point
    by fitting a local quadratic height-function patch in the tangent plane.

    Returns:
        k1, k2: (N,) principal curvatures (|k1| >= |k2|)
        d1, d2: (N,3) principal directions (unit, in 3D)
    """
    N = len(points)
    tree = KDTree(points)
    _, idx = tree.query(points, k=k + 1)   # k+1 because the point itself is included
    idx = idx[:, 1:]                        # exclude self, shape (N, k)

    k1_arr = np.zeros(N)
    k2_arr = np.zeros(N)
    d1_arr = np.zeros((N, 3))
    d2_arr = np.zeros((N, 3))

    for i in range(N):
        n = normals[i]
        n = n / (np.linalg.norm(n) + 1e-12)

        # Build orthonormal tangent frame {e1, e2} at p
        # Pick any vector not parallel to n
        if abs(n[0]) < 0.9:
            tmp = np.array([1.0, 0.0, 0.0])
        else:
            tmp = np.array([0.0, 1.0, 0.0])
        e1 = tmp - np.dot(tmp, n) * n
        e1 /= (np.linalg.norm(e1) + 1e-12)
        e2 = np.cross(n, e1)
        e2 /= (np.linalg.norm(e2) + 1e-12)

        # Neighbors projected onto tangent plane
        neighbors = points[idx[i]]  # (k, 3)
        diffs = neighbors - points[i]
        u = diffs @ e1   # (k,)
        v = diffs @ e2   # (k,)
        h = diffs @ n    # (k,) height above tangent plane

        # Fit quadratic: h = a*u^2 + b*u*v + c*v^2
        # Design matrix A, solve A @ [a,b,c]^T = h in least-squares sense
        A = np.stack([u**2, u * v, v**2], axis=1)   # (k, 3)
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, h, rcond=None)   # [a, b, c]
        except np.linalg.LinAlgError:
            continue
        a, b, c = coeffs

        # Second fundamental form (shape operator in {e1,e2} basis)
        II = np.array([[2*a, b], [b, 2*c]])

        # Principal curvatures and directions in tangent plane
        eigvals, eigvecs = np.linalg.eigh(II)   # ascending order
        # eigvals[1] >= eigvals[0]; use absolute values for anisotropy weighting
        k2_i, k1_i = eigvals[0], eigvals[1]     # k1: larger |eigenvalue|

        # Map 2D eigenvectors back to 3D
        ev1 = eigvecs[:, 1]   # corresponds to k1
        ev2 = eigvecs[:, 0]   # corresponds to k2
        d1_3d = ev1[0] * e1 + ev1[1] * e2
        d2_3d = ev2[0] * e1 + ev2[1] * e2
        d1_3d /= (np.linalg.norm(d1_3d) + 1e-12)
        d2_3d /= (np.linalg.norm(d2_3d) + 1e-12)

        k1_arr[i] = k1_i
        k2_arr[i] = k2_i
        d1_arr[i] = d1_3d
        d2_arr[i] = d2_3d

    return k1_arr, k2_arr, d1_arr, d2_arr


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

    # Build e1/e2 with Duff et al. (2017) smooth formula — same as compute_vertex_frames
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

    # Principal curvature estimation
    k1, k2, d1, d2 = _estimate_principal_curvature(
        pts, normals, k=k_fit, rho=rho, epsilon=epsilon
    )

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
