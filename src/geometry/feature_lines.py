"""
Feature line extraction from point clouds using normal variation and clustering.
Implements Section 5.5.1 with robust methods.
"""

import numpy as np
from scipy.spatial import KDTree
from typing import List, Optional, Tuple, Union
import warnings
from collections import defaultdict


def estimate_normals(points: np.ndarray, k: int = 20, consistent: bool = True) -> np.ndarray:
    """
    Estimate normals via PCA of k-nearest neighbors.
    Uses Open3D when available (fast, with MST-based consistent orientation).
    Falls back to a vectorized SciPy/numpy implementation.
    Returns (N,3) array of unit normals.
    """
    # Fast path: Open3D (vectorised C++ implementation + MST orientation)
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
        if consistent:
            pcd.orient_normals_consistent_tangent_plane(k=min(k, 15))
        return np.asarray(pcd.normals)
    except ImportError:
        pass

    # Fallback: SciPy KDTree with batched query (avoids per-point tree lookups)
    tree = KDTree(points)
    _, idx = tree.query(points, k=k)   # (N, k) — single batched call
    normals = np.zeros_like(points)
    for i in range(len(points)):
        neighbors = points[idx[i]]
        centered = neighbors - neighbors.mean(axis=0)
        cov = centered.T @ centered
        _, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]          # smallest eigenvector
        normals[i] = normal / (np.linalg.norm(normal) + 1e-12)
    return normals


def compute_curvature(points: np.ndarray, normals: np.ndarray, k: int = 20) -> np.ndarray:
    """
    Compute curvature as the variation of normals within neighborhood.
    Returns scalar curvature per point (higher means sharper feature).
    Fully vectorised: single batched KDTree query, no Python loop.
    """
    tree = KDTree(points)
    _, idx = tree.query(points, k=k)            # (N, k)
    neighbor_normals = normals[idx]             # (N, k, 3)
    mean_normal = neighbor_normals.mean(axis=1) # (N, 3)
    mean_norms = np.linalg.norm(mean_normal, axis=1, keepdims=True)
    mean_normal /= np.maximum(mean_norms, 1e-6)
    dots = np.abs((neighbor_normals * mean_normal[:, None, :]).sum(axis=2))  # (N, k)
    curvature = 1.0 - dots.mean(axis=1)        # (N,)
    return curvature


def detect_feature_points(
    points: np.ndarray,
    normals: Optional[np.ndarray] = None,
    k: int = 20,
    threshold: float = 0.2,
    method: str = 'curvature'
) -> np.ndarray:
    """
    Identify points likely on sharp features.
    Args:
        points: (N,3)
        normals: (N,3) or None (will estimate)
        k: neighborhood size for normal estimation/curvature
        threshold: cutoff for feature classification
        method: 'curvature' or 'normal_variance'
    Returns:
        boolean mask (N,) where True indicates feature point.
    """
    if normals is None:
        normals = estimate_normals(points, k=k)
    if method == 'curvature':
        curvature = compute_curvature(points, normals, k=k)
        return curvature > threshold
    elif method == 'normal_variance':
        # Use eigenvalue ratio of covariance of normals (simpler)
        tree = KDTree(points)
        mask = np.zeros(len(points), dtype=bool)
        for i in range(len(points)):
            _, idx = tree.query(points[i:i+1], k=k)
            neighbor_normals = normals[idx[0]]
            cov = neighbor_normals.T @ neighbor_normals / k
            eigvals = np.linalg.eigvalsh(cov)
            # If normals vary a lot, second eigenvalue is significant
            if len(eigvals) >= 2:
                ratio = eigvals[1] / (eigvals[0] + 1e-12)
                mask[i] = ratio > threshold
        return mask
    else:
        raise ValueError(f"Unknown method: {method}")


def cluster_feature_points(
    points: np.ndarray,
    feature_mask: np.ndarray,
    eps: float = 0.05,
    min_samples: int = 5,
    min_cluster_size: int = 10
) -> List[np.ndarray]:
    """
    Cluster feature points using DBSCAN.
    Returns a list of arrays of point indices for each cluster (excluding noise).
    """
    feature_pts = points[feature_mask]
    if len(feature_pts) < min_samples:
        return []

    labels = _dbscan_labels(feature_pts, eps=eps, min_samples=min_samples)
    unique_labels = set(labels)
    clusters = []
    for lab in unique_labels:
        if lab == -1:
            continue  # noise
        cluster_mask = (labels == lab)
        if np.sum(cluster_mask) >= min_cluster_size:
            global_indices = np.where(feature_mask)[0][cluster_mask]
            clusters.append(global_indices)
    return clusters


def _dbscan_labels(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Minimal DBSCAN implementation for small feature-point clusters."""
    n = len(points)
    labels = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return labels

    tree = KDTree(points)
    neighborhoods = tree.query_ball_point(points, r=float(eps))
    is_core = np.array([len(nb) >= int(min_samples) for nb in neighborhoods], dtype=bool)

    cluster_id = 0
    visited = np.zeros(n, dtype=bool)
    for seed in range(n):
        if visited[seed]:
            continue
        visited[seed] = True
        if not is_core[seed]:
            continue

        labels[seed] = cluster_id
        queue = list(neighborhoods[seed])
        while queue:
            j = queue.pop()
            if not visited[j]:
                visited[j] = True
                if is_core[j]:
                    queue.extend(neighborhoods[j])
            if labels[j] < 0:
                labels[j] = cluster_id
        cluster_id += 1

    return labels


def order_points_along_curve(points: np.ndarray) -> np.ndarray:
    """
    Order a set of points that roughly lie on a curve.
    Uses PCA projection and ordering along principal component.
    For more complex shapes, this may fail; consider using graph-based methods.
    """
    if len(points) < 2:
        return np.arange(len(points))

    centered = points - points.mean(axis=0, keepdims=True)
    cov = centered.T @ centered
    _, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, -1]
    proj = centered @ principal
    order = np.argsort(proj)
    return order


def fit_polyline(
    points: np.ndarray,
    cluster_indices: np.ndarray,
    smooth: bool = True,
    smooth_window: int = 5,
    simplify: bool = False,
    simplify_epsilon: float = 0.01
) -> np.ndarray:
    """
    Convert a cluster of points into an ordered polyline.
    Args:
        points: full point cloud (N,3)
        cluster_indices: indices of points in this cluster
        smooth: apply moving average smoothing
        smooth_window: window size for moving average
        simplify: apply Douglas-Peucker simplification
        simplify_epsilon: maximum distance for simplification
    Returns:
        ordered points (M,3) representing the polyline.
    """
    cluster_pts = points[cluster_indices]
    # Order the points
    order = order_points_along_curve(cluster_pts)
    ordered = cluster_pts[order]

    # Smoothing
    if smooth and len(ordered) >= smooth_window:
        from scipy.ndimage import uniform_filter1d
        # Use reflection padding to avoid boundary effects
        smoothed = uniform_filter1d(ordered, size=smooth_window, axis=0, mode='reflect')
        ordered = smoothed

    # Simplify (Douglas-Peucker)
    if simplify and len(ordered) > 2:
        ordered = _douglas_peucker(ordered, simplify_epsilon)

    return ordered


def _douglas_peucker(points: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Ramer-Douglas-Peucker polyline simplification.
    """
    if len(points) < 3:
        return points

    # Find point with maximum distance
    start, end = points[0], points[-1]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-12:
        return points

    distances = []
    for p in points[1:-1]:
        # Distance from point to line segment
        vec = p - start
        t = np.dot(vec, line_vec) / (line_len**2)
        proj = start + np.clip(t, 0, 1) * line_vec
        dist = np.linalg.norm(p - proj)
        distances.append(dist)

    max_dist = max(distances)
    max_idx = np.argmax(distances) + 1  # +1 because we skipped first

    if max_dist > epsilon:
        # Recursively simplify
        left = _douglas_peucker(points[:max_idx+1], epsilon)
        right = _douglas_peucker(points[max_idx:], epsilon)
        return np.vstack([left[:-1], right])
    else:
        return np.array([start, end])


def extract_feature_lines(
    points: np.ndarray,
    normals: Optional[np.ndarray] = None,
    k_normals: int = 20,
    curvature_threshold: float = 0.2,
    cluster_eps: float = 0.05,
    min_cluster_size: int = 10,
    smooth_polyline: bool = True,
    simplify_polyline: bool = True,
    simplify_epsilon: float = 0.01
) -> List[np.ndarray]:
    """
    Main function: detect and extract feature lines from point cloud.
    Returns a list of ordered point arrays, each representing a feature line.
    """
    # Step 1: Feature point detection
    feature_mask = detect_feature_points(
        points, normals, k=k_normals, threshold=curvature_threshold
    )

    # Step 2: Clustering
    clusters = cluster_feature_points(
        points, feature_mask, eps=cluster_eps,
        min_samples=min_cluster_size // 2, min_cluster_size=min_cluster_size
    )

    # Step 3: Fit polylines
    lines = []
    for cl in clusters:
        line = fit_polyline(
            points, cl,
            smooth=smooth_polyline,
            simplify=simplify_polyline,
            simplify_epsilon=simplify_epsilon
        )
        lines.append(line)

    return lines


def extract_feature_lines_from_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    dihedral_threshold_deg: float = 25.0,
    min_polyline_vertices: int = 2,
    include_boundary: bool = True,
) -> List[np.ndarray]:
    """
    Extract sharp feature polylines directly from a triangle mesh using dihedral angles.

    This path is more appropriate than point-cloud curvature clustering when the
    input already has connectivity and sharp CAD-like creases.
    """
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    if F.ndim != 2 or F.shape[1] != 3 or len(F) == 0:
        return []

    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12

    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, tri in enumerate(F):
        for k in range(3):
            a = int(tri[k])
            b = int(tri[(k + 1) % 3])
            if a == b:
                continue
            e = (a, b) if a < b else (b, a)
            edge_faces[e].append(fi)

    threshold_cos = float(np.cos(np.deg2rad(float(dihedral_threshold_deg))))
    feature_edges: list[tuple[int, int]] = []
    for e, incident in edge_faces.items():
        if len(incident) == 1:
            if include_boundary:
                feature_edges.append(e)
            continue
        if len(incident) != 2:
            continue
        n0 = fn[incident[0]]
        n1 = fn[incident[1]]
        if float(np.dot(n0, n1)) <= threshold_cos:
            feature_edges.append(e)

    if not feature_edges:
        return []

    adj: dict[int, set[int]] = defaultdict(set)
    for a, b in feature_edges:
        adj[a].add(b)
        adj[b].add(a)

    visited_edges: set[tuple[int, int]] = set()
    polylines: list[np.ndarray] = []

    def _mark(a: int, b: int):
        key = (a, b) if a < b else (b, a)
        visited_edges.add(key)

    def _seen(a: int, b: int) -> bool:
        key = (a, b) if a < b else (b, a)
        return key in visited_edges

    seeds = [v for v, nbs in adj.items() if len(nbs) != 2]
    seeds += [v for v in adj.keys() if v not in seeds]

    for seed in seeds:
        for nb in list(adj.get(seed, [])):
            if _seen(seed, nb):
                continue
            line = [seed]
            prev = seed
            cur = nb
            _mark(seed, nb)
            line.append(cur)
            while True:
                nbs = [x for x in adj[cur] if x != prev and not _seen(cur, x)]
                if not nbs:
                    break
                if len(adj[cur]) != 2 and cur != seed:
                    break
                nxt = nbs[0]
                prev, cur = cur, nxt
                _mark(prev, cur)
                line.append(cur)
                if cur == seed:
                    break
            if len(line) >= max(2, int(min_polyline_vertices)):
                polylines.append(V[np.asarray(line, dtype=np.int64)])

    return polylines


def extract_feature_corners_from_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    dihedral_threshold_deg: float = 25.0,
    include_boundary: bool = True,
    min_corner_degree: int = 3,
) -> np.ndarray:
    """
    Extract sharp corner vertices from a triangle mesh.

    A vertex is treated as a corner when its incident sharp-edge degree is not 2.
    Degree >= min_corner_degree catches CAD-like junctions; degree 1 catches open
    feature endpoints. Boundary corners are included when include_boundary=True.
    """
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    if F.ndim != 2 or F.shape[1] != 3 or len(F) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12

    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, tri in enumerate(F):
        for k in range(3):
            a = int(tri[k])
            b = int(tri[(k + 1) % 3])
            if a == b:
                continue
            e = (a, b) if a < b else (b, a)
            edge_faces[e].append(fi)

    threshold_cos = float(np.cos(np.deg2rad(float(dihedral_threshold_deg))))
    adj: dict[int, set[int]] = defaultdict(set)
    for e, incident in edge_faces.items():
        is_feature = False
        if len(incident) == 1:
            is_feature = bool(include_boundary)
        elif len(incident) == 2:
            n0 = fn[incident[0]]
            n1 = fn[incident[1]]
            is_feature = float(np.dot(n0, n1)) <= threshold_cos
        if not is_feature:
            continue
        a, b = e
        adj[a].add(b)
        adj[b].add(a)

    corner_ids = [
        vid for vid, nbrs in adj.items()
        if len(nbrs) == 1 or len(nbrs) >= int(min_corner_degree)
    ]
    if not corner_ids:
        return np.zeros((0, 3), dtype=np.float64)
    return V[np.asarray(sorted(set(corner_ids)), dtype=np.int64)]
