"""
Local Canonical Frame (LCF) construction for point clouds.
Implements Section 5.1.2 of the paper.
"""

import numpy as np
from typing import Tuple, Optional


def compute_local_canonical_frame(
    points: np.ndarray,
    query_idx: int,
    k: int = 32,
    normals: Optional[np.ndarray] = None,
    return_neighbors: bool = False,
    tree=None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Construct a local canonical frame (LCF) for a query point using PCA on its k-nearest neighbors.

    Args:
        points: (N, 3) array of point coordinates.
        query_idx: Index of the query point.
        k: Number of nearest neighbors to use (including the query point).
        normals: (N, 3) array of normals (optional). If provided, the normal is aligned with the local z-axis.
        return_neighbors: If True, return the indices of the k-nearest neighbors.

    Returns:
        local_coords: (k, 3) coordinates of the k neighbors transformed to the LCF.
        basis: (3, 3) orthonormal basis matrix (columns are e1, e2, e3).
        neighbor_normals: (k, 3) normals of the neighbors transformed to LCF (if normals provided).
        neighbor_indices: (k,) indices of the k neighbors (if return_neighbors=True).
    """
    from scipy.spatial import KDTree

    # Reuse a pre-built tree if provided, otherwise build one.
    # Passing a pre-built tree avoids O(N²logN) cost when called in a loop over all points.
    if tree is None:
        tree = KDTree(points)
    distances, indices = tree.query(points[query_idx], k=k)  # includes the point itself

    # Extract neighbor points
    neighbors = points[indices]  # (k, 3)
    center = points[query_idx]   # (3,)

    # Center the neighbors
    centered = neighbors - center  # (k, 3)

    # PCA via direct 3x3 eigendecomposition of the covariance matrix.
    # np.linalg.eigh avoids PCA object overhead and keeps DataLoader work small.
    cov = centered.T @ centered                        # (3, 3)
    eigvals, eigvecs = np.linalg.eigh(cov)             # ascending
    # Descending order, columns are components.
    basis = eigvecs[:, ::-1]                           # (3, 3)

    # ── Sign disambiguation ────────────────────────────────────────────────────
    # PCA eigenvectors have an arbitrary sign (v and -v are equally valid).
    # Without disambiguation, the same patch processed twice can yield (e1, e2) or
    # (-e1, e2), flipping the off-diagonal terms of the 2D projected metric tensor
    # and corrupting the predicted principal direction angle θ*.
    #
    # Convention: each eigenvector's largest-magnitude component must be positive.
    # This is the standard "max-abs" sign convention.
    for i in range(3):
        col = basis[:, i]
        max_idx = np.argmax(np.abs(col))
        if col[max_idx] < 0:
            basis[:, i] = -col

    # Ensure right-handed frame: e1 × e2 should point in the same half-space as e3.
    e1, e2, e3 = basis[:, 0], basis[:, 1], basis[:, 2]
    if np.dot(np.cross(e1, e2), e3) < 0:
        basis[:, 2] = -basis[:, 2]

    # If normals are provided, align the z-axis with the normal of the query point.
    # Build e1/e2 with Duff et al.'s numerically stable ONB formula — same as
    # compute_vertex_frames — so training and inference share the same frame.
    if normals is not None:
        normal_query = normals[query_idx]  # (3,)
        e3_new = normal_query / (np.linalg.norm(normal_query) + 1e-12)
        nx, ny, nz = e3_new[0], e3_new[1], e3_new[2]
        sign  = 1.0 if nz >= 0.0 else -1.0
        a     = -1.0 / (sign + nz)
        b     = nx * ny * a
        e1_new = np.array([1.0 + sign * nx**2 * a,  sign * b,  -sign * nx])
        e2_new = np.array([b,  sign + ny**2 * a,  -ny])
        e1_new /= np.linalg.norm(e1_new) + 1e-12
        e2_new /= np.linalg.norm(e2_new) + 1e-12
        basis = np.stack([e1_new, e2_new, e3_new], axis=1)

    # ── X-Y axis sign disambiguation via 3rd-order moment (skewness) ──────────
    # After PCA (or Duff ONB), e1 and e2 can still flip 180°.
    # Force e1 to point toward the denser half of the local neighborhood by
    # requiring the skewness (3rd moment) of projected X-coordinates to be positive.
    # This makes the frame consistent across identical patches processed separately.
    coords_tmp = centered @ basis   # (k, 3) temporary projection
    if np.sum(coords_tmp[:, 0] ** 3) < 0:
        basis[:, 0] = -basis[:, 0]
        basis[:, 1] = -basis[:, 1]  # flip both to keep right-handed frame

    # Transform neighbors to LCF: coordinates in the basis
    local_coords = centered @ basis  # (k, 3)   (x,y,z) in local frame

    # Transform normals if provided
    neighbor_normals = None
    if normals is not None:
        neighbor_normals = normals[indices] @ basis  # rotate normals as vectors

    if return_neighbors:
        return local_coords, basis, neighbor_normals, indices
    else:
        return local_coords, basis, neighbor_normals
