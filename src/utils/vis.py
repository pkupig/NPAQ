"""
Visualization utilities using polyscope and matplotlib.
Supports rendering of point clouds, meshes, metric fields (as ellipses), and feature lines.
"""

import numpy as np
try:
    import polyscope as ps
except ImportError:
    ps = None
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, List, Tuple, Union
import warnings


def init_ps(backend: str = "gl", up_direction: str = "z_up"):
    """
    Initialize polyscope with preferred settings.
    """
    ps.init(backend=backend)
    ps.set_up_dir(up_direction)
    ps.set_ground_plane_mode("shadow_only")
    ps.set_SSAA_factor(2)  # Anti-aliasing for prettier screenshots
    return ps


def register_point_cloud(
    name: str,
    points: np.ndarray,
    normals: Optional[np.ndarray] = None,
    color: Optional[Tuple[float, float, float]] = None,
    radius: float = 0.005,
    enabled: bool = True,
):
    """
    Register a point cloud with polyscope.
    """
    ps_cloud = ps.register_point_cloud(name, points, radius=radius, enabled=enabled)
    if normals is not None:
        ps_cloud.add_vector_quantity("normals", normals, enabled=False)
    if color is not None:
        ps_cloud.set_color(color)
    return ps_cloud


def register_mesh(
    name: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    color: Optional[Tuple[float, float, float]] = None,
    enabled: bool = True,
):
    """
    Register a mesh with polyscope.
    Faces can be triangles (N,3) or quads (N,4). Polyscope handles both.
    """
    ps_mesh = ps.register_surface_mesh(name, vertices, faces, enabled=enabled)
    if color is not None:
        ps_mesh.set_color(color)
    return ps_mesh


def add_metric_field_ellipses(
    name: str,
    points: np.ndarray,
    metric_tensors: np.ndarray,
    scale: float = 1.0,
    color: Tuple[float, float, float] = (1.0, 0.5, 0.0),
    resolution: int = 20,
    opacity: float = 0.5,
):
    """
    Visualize metric tensors as ellipses on a point cloud.
    Each ellipse is the MAIE of a quadrilateral (inverse of metric).
    We render them as oriented disks (flattened ellipsoids) in the tangent plane.
    
    Implementation: Generates a single batched mesh containing all ellipses to 
    avoid overhead of registering thousands of individual structures.

    Args:
        name: Name of the structure in Polyscope.
        points: (N, 3) centers of ellipses.
        metric_tensors: (N, 3, 3) global metric tensors. 
                        Eigenvalues correspond to 1/axis_length^2.
                        Smallest eigenvalue typically corresponds to the normal direction.
        scale: Global scaling factor for ellipse size.
        color: RGB tuple.
        resolution: Number of vertices per ellipse disk.
        opacity: Transparency of the ellipses.
    """
    if len(points) == 0 or len(metric_tensors) == 0:
        return

    N = len(points)
    
    # 1. Create a template unit disk in XY plane (Z=0)
    theta = np.linspace(0, 2*np.pi, resolution, endpoint=False)
    disk_verts = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1) # (Res, 3)
    # Add center point at (0,0,0) for triangle fan
    disk_verts = np.vstack([[[0,0,0]], disk_verts]) # (Res+1, 3)
    
    # Create faces (triangle fan)
    # Center is index 0. Vertices are 1..Res
    disk_faces = []
    for i in range(resolution):
        # Triangle: Center, i+1, (i+1)%Res + 1
        v1 = i + 1
        v2 = (i + 1) % resolution + 1
        disk_faces.append([0, v1, v2])
    disk_faces = np.array(disk_faces) # (Res, 3)

    all_verts = []
    all_faces = []
    
    # 2. Transform disk for each point
    for i in range(N):
        center = points[i]
        M = metric_tensors[i]
        
        # M is typically 3x3 for global metric. 
        # Eigen decomposition: M = V * diag(lambda) * V.T
        # Ellipse semi-axes lengths are 1 / sqrt(lambda)
        
        try:
            eigvals, eigvecs = np.linalg.eigh(M)
        except np.linalg.LinAlgError:
            continue
            
        # Filter negative or zero eigenvalues (numerical stability)
        # In our synthesis, the "normal" direction has 0 eigenvalue (infinite length) 
        # OR it has a very large weight if M measures "cost".
        # Let's verify definition: M = J^-T J^-1. 
        # High curvature -> High M -> Small ellipse axis.
        # Normal direction usually has NO deformation cost in surface metric, 
        # but for M to be valid 3D tensor, we usually set normal eigenvalue to something specific 
        # or it naturally comes out.
        # IF M was constructed as sum( w_i d_i d_i^T ), then eigvals are w1, w2, 0.
        # 0 eigenvalue => infinite length. We want to draw a flat disk, so we force that axis to 0.
        
        # We assume the SMALLEST eigenvalue corresponds to the normal direction if it is close to 0,
        # OR the LARGEST eigenvalue corresponds to normal if M is a "projection" metric?
        # Actually, in `synthesis.py`, M = w1 d1 d1^T + w2 d2 d2^T.
        # The eigenvalues are w1, w2, and 0. 
        # The 0 eigenvalue corresponds to the normal direction n = d1 x d2.
        # We want the ellipse to be flat in the normal direction.
        # So we map the eigenvector corresponding to 0 eigenvalue to the disk's Z axis (which has 0 height).
        # And map the other two to X and Y axes of the disk.

        # Sort eigenvalues to identify normal vs tangent
        # eigvals are in ascending order by default in eigh
        # order: [approx_0, w_smaller, w_larger]
        # map to: [Z_axis,  X_axis,    Y_axis   ]
        # lengths:[0,       1/sqrt(w), 1/sqrt(W)]
        
        # Define scale factors
        scales = np.zeros(3)
        # Avoid division by zero
        valid_mask = eigvals > 1e-6
        scales[valid_mask] = 1.0 / np.sqrt(eigvals[valid_mask]) * scale
        
        # If eigenvalue is effectively 0, we want thickness to be 0 (flat disk)
        scales[~valid_mask] = 0.0 
        
        # Construct transformation matrix T = V * Scale
        # Note: Disk is defined in XY plane. 
        # Z (0,0,1) should map to eigenvector 0 (Normal).
        # X (1,0,0) should map to eigenvector 1.
        # Y (0,1,0) should map to eigenvector 2.
        # V columns are [eigvec0, eigvec1, eigvec2]
        
        transform = eigvecs @ np.diag(scales)
        
        # Apply transform: v_new = T @ v_old + center
        # disk_verts is (K, 3), we need (T @ disk_verts.T).T
        v_transformed = (transform @ disk_verts.T).T + center
        
        all_verts.append(v_transformed)
        # Offset faces
        vertex_offset = i * len(disk_verts)
        all_faces.append(disk_faces + vertex_offset)

    if not all_verts:
        return

    # 3. Flatten and register
    total_verts = np.vstack(all_verts)
    total_faces = np.vstack(all_faces)
    
    ps_mesh = ps.register_surface_mesh(f"{name}_ellipses", total_verts, total_faces, enabled=True)
    ps_mesh.set_color(color)
    ps_mesh.set_transparency(opacity)
    ps_mesh.set_smooth_shade(False) # Flat shading makes orientation clearer


def visualize_with_matplotlib(
    points: Optional[np.ndarray] = None,
    mesh_vertices: Optional[np.ndarray] = None,
    mesh_faces: Optional[np.ndarray] = None,
    feature_lines: Optional[List[np.ndarray]] = None,
    metric_tensors: Optional[np.ndarray] = None,
    ellipsoid_scale: float = 1.0,
    show: bool = True,
    title: str = "Visualization",
):
    """
    Fallback visualization using matplotlib (3D). Good for static images.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    if points is not None:
        # Subsample for performance if too many points
        if len(points) > 2000:
            idx = np.random.choice(len(points), 2000, replace=False)
            pts_vis = points[idx]
        else:
            pts_vis = points
        ax.scatter(pts_vis[:, 0], pts_vis[:, 1], pts_vis[:, 2], c='blue', s=1, alpha=0.6, label='Point cloud')

    if mesh_vertices is not None and mesh_faces is not None:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        # Matplotlib is very slow with many faces, so we just plot vertices or a subset
        if len(mesh_faces) > 1000:
             # Just plot wireframe or subset
             pass
        else:
            mesh = Poly3DCollection(mesh_vertices[mesh_faces], alpha=0.3, facecolor='cyan', edgecolor='gray', linewidth=0.1)
            ax.add_collection3d(mesh)

    if feature_lines is not None:
        for line in feature_lines:
            ax.plot(line[:, 0], line[:, 1], line[:, 2], 'r-', linewidth=2)

    if metric_tensors is not None and points is not None:
        # Crude visualization of principal directions
        # Just draw lines along the major axis
        for i in range(0, len(points), max(1, len(points)//200)): # sparse subsample
            p = points[i]
            M = metric_tensors[i]
            if M.shape == (3,3):
                vals, vecs = np.linalg.eigh(M)
                # Smallest eigenvalue of metric => Largest axis of ellipse
                # But in our synthesis, 0 eigenvalue is normal. 
                # Let's look for the smallest NON-ZERO eigenvalue => Largest Tangent Axis
                # Or just plot the eigenvectors corresponding to the two largest eigenvalues of M 
                # (which correspond to smallest axes - high curvature directions)
                # Let's plot the direction of largest M eigenvalue (High curvature direction)
                d = vecs[:, 2] 
                # Scale for vis
                l = 0.05 * ellipsoid_scale
                ax.plot([p[0]-d[0]*l, p[0]+d[0]*l], 
                        [p[1]-d[1]*l, p[1]+d[1]*l], 
                        [p[2]-d[2]*l, p[2]+d[2]*l], 'k-', linewidth=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    if show:
        plt.show()
    return fig, ax