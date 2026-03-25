"""
Laplacian smoothing of metric fields on point clouds.
Implements diffusion equation for smoothing metric tensors (Section 5.2).
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.neighbors import KDTree
from typing import Optional


def build_point_cloud_laplacian(
    points: np.ndarray,
    k: int = 10,
    metric: str = 'cotangent'
) -> sparse.csr_matrix:
    """
    Build a graph Laplacian matrix for a point cloud.
    Supports simple graph Laplacian (uniform weights) or cotangent-like weights (if normals available).

    Args:
        points: (N, 3) array of point coordinates.
        k: Number of nearest neighbors for graph construction.
        metric: Type of Laplacian: 'uniform' (binary adjacency) or 'cotangent' (requires normals? not implemented).

    Returns:
        L: (N, N) sparse Laplacian matrix (L = D - A).
    """
    N = int(points.shape[0])
    if N <= 1:
        return sparse.csr_matrix((N, N))
    k = max(1, min(int(k), N - 1))

    tree = KDTree(points)
    distances, indices = tree.query(points, k=k+1)  # include self

    row = []
    col = []
    data = []

    for i in range(N):
        neighbors = indices[i, 1:]  # exclude self
        for j in neighbors:
            row.append(i)
            col.append(j)
            data.append(1.0)
            # make symmetric
            row.append(j)
            col.append(i)
            data.append(1.0)

    # Build adjacency matrix (symmetric, binary)
    A = sparse.csr_matrix((data, (row, col)), shape=(N, N))
    # Binarize: scipy sums duplicate COO entries during CSR construction.
    # When the k-NN graph is symmetric (j in N(i) and i in N(j)), each pair is added
    # twice (once from i's loop, once from j's loop), making A[i,j]=2 instead of 1.
    # Resetting data to 1.0 fixes this without changing sparsity pattern.
    A.data[:] = 1.0
    # Set diagonal to zero (just in case)
    A.setdiag(0)
    A.eliminate_zeros()

    # Degree matrix
    D = sparse.diags(np.array(A.sum(axis=1)).flatten())

    # Laplacian: L = D - A
    L = D - A
    return L.tocsr()


def smooth_metric_field(
    metric_field: np.ndarray,
    points: np.ndarray,
    lambda_smooth: float = 0.1,
    iterations: int = 10,
    laplacian: Optional[sparse.csr_matrix] = None,
    k: int = 10
) -> np.ndarray:
    """
    Smooth a metric field defined on a point cloud using diffusion:
        M_new = M - lambda_smooth * L @ M   (implicitly, but here we solve (I + lambda L) M_new = M)
    or explicitly iterate.

    Args:
        metric_field: (N, 2, 2) array of metric tensors.
        points: (N, 3) point coordinates.
        lambda_smooth: Smoothing strength (time step).
        iterations: Number of explicit smoothing iterations.
        laplacian: Precomputed Laplacian matrix (optional).
        k: Number of neighbors if Laplacian not provided.

    Returns:
        smoothed_field: (N, 2, 2) smoothed metric tensors.
    """
    if laplacian is None:
        laplacian = build_point_cloud_laplacian(points, k=k)

    N = metric_field.shape[0]
    # Flatten metric tensors to vectors of length 4 (since symmetric, we can store 3 components, but for simplicity keep all 4)
    # We'll treat each component independently.
    # Option 1: smooth each component separately.
    components = np.zeros((N, 4))
    components[:, 0] = metric_field[:, 0, 0]  # M_00
    components[:, 1] = metric_field[:, 0, 1]  # M_01
    components[:, 2] = metric_field[:, 1, 0]  # M_10 (symmetric, same as 01)
    components[:, 3] = metric_field[:, 1, 1]  # M_11

    # Explicit diffusion: v_new = v - lambda * L @ v
    # We can do multiple iterations.
    for _ in range(iterations):
        # Compute L @ v for each component
        for c in range(4):
            diff = laplacian @ components[:, c]
            components[:, c] -= lambda_smooth * diff

    # Reconstruct symmetric tensors
    smoothed = np.zeros_like(metric_field)
    smoothed[:, 0, 0] = components[:, 0]
    smoothed[:, 0, 1] = components[:, 1]
    smoothed[:, 1, 0] = components[:, 2]  # Should equal components[:,1] if symmetric, but we'll enforce
    smoothed[:, 1, 1] = components[:, 3]

    # Enforce symmetry and positive definiteness (optional)
    # Symmetrize
    smoothed[:, 0, 1] = smoothed[:, 1, 0] = 0.5 * (smoothed[:, 0, 1] + smoothed[:, 1, 0])
    # Could also project to SPD cone, but maybe not needed.

    return smoothed


def smooth_metric_field_implicit(
    metric_field: np.ndarray,
    points: np.ndarray,
    lambda_smooth: float = 0.1,
    laplacian: Optional[sparse.csr_matrix] = None,
    k: int = 10
) -> np.ndarray:
    """
    Implicit smoothing: solve (I + lambda L) M_new = M.
    This is more stable for large lambda.
    """
    if laplacian is None:
        laplacian = build_point_cloud_laplacian(points, k=k)

    N = metric_field.shape[0]
    I = sparse.identity(N, format='csr')
    A = I + lambda_smooth * laplacian

    components = np.zeros((N, 4))
    components[:, 0] = metric_field[:, 0, 0]
    components[:, 1] = metric_field[:, 0, 1]
    components[:, 2] = metric_field[:, 1, 0]
    components[:, 3] = metric_field[:, 1, 1]

    smoothed_comp = np.zeros_like(components)
    for c in range(4):
        smoothed_comp[:, c] = spsolve(A, components[:, c])

    smoothed = np.zeros_like(metric_field)
    smoothed[:, 0, 0] = smoothed_comp[:, 0]
    smoothed[:, 0, 1] = smoothed_comp[:, 1]
    smoothed[:, 1, 0] = smoothed_comp[:, 2]
    smoothed[:, 1, 1] = smoothed_comp[:, 3]
    smoothed[:, 0, 1] = smoothed[:, 1, 0] = 0.5 * (smoothed[:, 0, 1] + smoothed[:, 1, 0])
    return smoothed
