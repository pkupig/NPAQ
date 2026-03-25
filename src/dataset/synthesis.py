"""
Synthetic analytic surface generation for supervised training.
Implements Section 5.1.3 of the paper.
Returns GLOBAL 3x3 metric tensors based on curvature.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Callable, Optional
from dataclasses import dataclass


@dataclass
class SurfaceSpec:
    """Specification of an analytic surface."""
    name: str
    func: Callable[[np.ndarray, np.ndarray], np.ndarray]  # maps (u,v) to (x,y,z)
    # Returns (N, 3, 3) global metric tensor based on curvature
    metric_tensor: Callable[[np.ndarray, np.ndarray], np.ndarray]
    principal_dirs: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
    # Returns (N, 3) outward unit normals — used as extra model input features
    normals_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = None
    # Parameter domain bounds; surfaces with unbounded curvature falloff need a finite range
    u_range: Tuple[float, float] = (0.0, 2 * np.pi)
    v_range: Tuple[float, float] = (0.0, 2 * np.pi)


def construct_target_metric(
    dir1: np.ndarray, 
    dir2: np.ndarray, 
    k1: np.ndarray, 
    k2: np.ndarray,
    rho: float = 1.0,
    epsilon: float = 0.1
) -> np.ndarray:
    """
    Construct global 3x3 target metric tensor from principal directions and curvatures.
    M = (rho * (|k1| + eps))^2 * d1 * d1^T + (rho * (|k2| + eps))^2 * d2 * d2^T
    (High curvature -> High metric value -> Small edge length)
    """
    # Weights proportional to squared curvature (inverse squared edge length)
    w1 = (rho * (np.abs(k1) + epsilon)) ** 2
    w2 = (rho * (np.abs(k2) + epsilon)) ** 2
    
    # Outer products (N, 3, 3)
    d1_outer = dir1[:, :, None] * dir1[:, None, :]
    d2_outer = dir2[:, :, None] * dir2[:, None, :]
    
    M = w1[:, None, None] * d1_outer + w2[:, None, None] * d2_outer
    return M


def cylinder_surface(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Cylinder: radius=1, height=2."""
    x = np.cos(u)
    y = np.sin(u)
    z = v
    return np.stack([x, y, z], axis=-1)

def cylinder_metric(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Target metric for cylinder."""
    # Principal directions
    d1, d2 = cylinder_principal_dirs(u, v)
    # Curvatures: k1=0 (axial), k2=1 (hoop)
    k1 = np.zeros_like(u)
    k2 = np.ones_like(u)
    
    return construct_target_metric(d1, d2, k1, k2)

def cylinder_principal_dirs(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """d1: axial (z), d2: hoop (xy)"""
    zeros = np.zeros_like(u)
    ones = np.ones_like(u)
    # d1 along Z axis (0,0,1)
    d1 = np.stack([zeros, zeros, ones], axis=-1)
    # d2 along circle tangent (-sin u, cos u, 0)
    d2 = np.stack([-np.sin(u), np.cos(u), zeros], axis=-1)
    return d1, d2

def cylinder_normals(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Outward radial normal of cylinder."""
    zeros = np.zeros_like(u)
    return np.stack([np.cos(u), np.sin(u), zeros], axis=-1)


def torus_surface(u: np.ndarray, v: np.ndarray, R: float = 2.0, r: float = 1.0) -> np.ndarray:
    """Torus: major R, minor r."""
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    return np.stack([x, y, z], axis=-1)

def torus_metric(u: np.ndarray, v: np.ndarray, R: float = 2.0, r: float = 1.0) -> np.ndarray:
    d1, d2 = torus_principal_dirs(u, v, R, r)
    # Curvatures
    # k1 (along u, toroidal): cos(v) / (R + r*cos(v))
    k1 = np.cos(v) / (R + r * np.cos(v))
    # k2 (along v, poloidal): 1 / r
    k2 = np.ones_like(u) / r
    
    return construct_target_metric(d1, d2, k1, k2)

def torus_principal_dirs(u: np.ndarray, v: np.ndarray, R: float = 2.0, r: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    # d1: tangent to large circle (u) -> (-sin u, cos u, 0)
    d1 = np.stack([-np.sin(u), np.cos(u), np.zeros_like(u)], axis=-1)
    # d2: tangent to small circle (v)
    # normal to surface is (cos u cos v, sin u cos v, sin v)
    # d2 is (-cos u sin v, -sin u sin v, cos v)
    d2 = np.stack([-np.cos(u)*np.sin(v), -np.sin(u)*np.sin(v), np.cos(v)], axis=-1)
    return d1, d2

def torus_normals(u: np.ndarray, v: np.ndarray, R: float = 2.0, r: float = 1.0) -> np.ndarray:
    """Outward surface normal of torus: points away from the tube centre."""
    return np.stack([np.cos(v)*np.cos(u), np.cos(v)*np.sin(u), np.sin(v)], axis=-1)


def saddle_surface(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Saddle: z = u^2 - v^2."""
    x = u
    y = v
    z = u**2 - v**2
    return np.stack([x, y, z], axis=-1)

def saddle_metric(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    # Use exact differential geometry for z = x^2 - y^2
    # p = (u, v, u^2-v^2)
    # xu = (1, 0, 2u), xv = (0, 1, -2v)
    # E = 1+4u^2, F = -4uv, G = 1+4v^2
    # n = (-2u, 2v, 1) / sqrt(1+4u^2+4v^2)
    # L = 2 / |n|, M = 0, N = -2 / |n|
    # Gaussian K = -4 / (1+4u^2+4v^2)^2
    # Mean H = ...
    
    # For robust training, we can just approximate principal dirs or use a simplified metric 
    # that encourages orthogonality.
    # Here we perform an explicit calculation for high quality data.
    
    xu = np.stack([np.ones_like(u), np.zeros_like(u), 2*u], axis=-1)
    xv = np.stack([np.zeros_like(u), np.ones_like(u), -2*v], axis=-1)
    
    # Normal
    n_unorm = np.cross(xu, xv)
    n_len = np.linalg.norm(n_unorm, axis=-1, keepdims=True)
    n = n_unorm / n_len
    
    # Second fundamental form coeffs
    # xuu = (0,0,2), xuv=(0,0,0), xvv=(0,0,-2)
    L = 2 / n_len.squeeze(-1)
    M = np.zeros_like(u)
    N_coeff = -2 / n_len.squeeze(-1)
    
    # First fundamental form
    E = np.sum(xu*xu, axis=-1)
    F = np.sum(xu*xv, axis=-1)
    G = np.sum(xv*xv, axis=-1)
    
    # Shape operator eigenvalues (Principal Curvatures)
    # k^2 - 2H k + K = 0
    # K = (LN - M^2) / (EG - F^2)
    # H = (EN + GL - 2FM) / 2(EG - F^2)
    denom = E*G - F**2
    K_gauss = (L*N_coeff - M**2) / denom
    H_mean = (E*N_coeff + G*L - 2*F*M) / (2*denom)
    
    discriminant = np.sqrt(np.maximum(H_mean**2 - K_gauss, 0))
    k1 = H_mean + discriminant
    k2 = H_mean - discriminant
    
    # Principal directions
    # direction (du, dv) satisfies (L - kE)du + (M - kF)dv = 0
    # For k1:
    A = L - k1*E
    B = M - k1*F
    # vector (B, -A) or (-B, A) in parameter space
    # d1_uv = (M - k1*F, -(L - k1*E))
    d1_u = M - k1*F
    d1_v = -(L - k1*E)
    
    # Map to 3D: d1 = d1_u * xu + d1_v * xv
    d1 = d1_u[:, None] * xu + d1_v[:, None] * xv
    d1 = d1 / (np.linalg.norm(d1, axis=-1, keepdims=True) + 1e-12)
    
    # d2 is orthogonal to d1 in tangent plane (and usually corresponds to k2)
    d2 = np.cross(n, d1)
    
    return construct_target_metric(d1, d2, k1, k2)

def saddle_normals(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Upward unit normal of saddle z = u^2 - v^2."""
    n = np.stack([-2*u, 2*v, np.ones_like(u)], axis=-1)
    return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12)

def saddle_principal_dirs(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Placeholder to match interface, logic is inside metric function
    # We can just return the ones computed above if needed, but for synthesis loop:
    # We call metric function which does the work.
    # We can re-implement light version here or simply call metric function logic.
    # To save compute, we'll just return approximations or dummy here if not used separately.
    # BUT, synthesis.py calls this to get ground truth vectors for directional loss.
    # So we MUST implement it.
    
    # Reuse logic (copy-paste for independence or helper)
    xu = np.stack([np.ones_like(u), np.zeros_like(u), 2*u], axis=-1)
    xv = np.stack([np.zeros_like(u), np.ones_like(u), -2*v], axis=-1)
    n_unorm = np.cross(xu, xv)
    n_len = np.linalg.norm(n_unorm, axis=-1, keepdims=True)
    L = 2 / n_len.squeeze(-1)
    M = np.zeros_like(u)
    N_coeff = -2 / n_len.squeeze(-1)
    E = np.sum(xu*xu, axis=-1)
    F = np.sum(xu*xv, axis=-1)
    G = np.sum(xv*xv, axis=-1)
    denom = E*G - F**2
    K_gauss = (L*N_coeff - M**2) / denom
    H_mean = (E*N_coeff + G*L - 2*F*M) / (2*denom)
    discriminant = np.sqrt(np.maximum(H_mean**2 - K_gauss, 0))
    k1 = H_mean + discriminant
    d1_u = M - k1*F
    d1_v = -(L - k1*E)
    d1 = d1_u[:, None] * xu + d1_v[:, None] * xv
    d1 = d1 / (np.linalg.norm(d1, axis=-1, keepdims=True) + 1e-12)
    n = n_unorm / n_len
    d2 = np.cross(n, d1)
    return d1, d2


# ---------------------------------------------------------------------------
# Sphere  (isotropic: k1 = k2 = 1/R everywhere)
# ---------------------------------------------------------------------------

def sphere_surface(u: np.ndarray, v: np.ndarray, R: float = 1.0) -> np.ndarray:
    """Unit sphere in spherical coordinates (u=longitude, v=colatitude)."""
    x = R * np.sin(v) * np.cos(u)
    y = R * np.sin(v) * np.sin(u)
    z = R * np.cos(v)
    return np.stack([x, y, z], axis=-1)

def sphere_metric(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Isotropic metric: k1 = k2 = 1 everywhere."""
    d1, d2 = sphere_principal_dirs(u, v)
    k1 = np.ones_like(u)
    k2 = np.ones_like(u)
    return construct_target_metric(d1, d2, k1, k2)

def sphere_principal_dirs(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """d1: longitude tangent, d2: meridian tangent."""
    zeros = np.zeros_like(u)
    # d1 = d/du normalised = (-sin u, cos u, 0)  [unit vector]
    d1 = np.stack([-np.sin(u), np.cos(u), zeros], axis=-1)
    # d2 = d/dv normalised = (cos v cos u, cos v sin u, -sin v)
    d2 = np.stack([np.cos(v) * np.cos(u), np.cos(v) * np.sin(u), -np.sin(v)], axis=-1)
    return d1, d2

def sphere_normals(u: np.ndarray, v: np.ndarray, R: float = 1.0) -> np.ndarray:
    """Outward unit normal of unit sphere = position vector."""
    return np.stack([np.sin(v)*np.cos(u), np.sin(v)*np.sin(u), np.cos(v)], axis=-1)


# ---------------------------------------------------------------------------
# Ellipsoid  (variable anisotropy: curvatures differ by location and axes)
# ---------------------------------------------------------------------------

def ellipsoid_surface(u: np.ndarray, v: np.ndarray,
                      a: float = 2.0, b: float = 1.0, c: float = 0.5) -> np.ndarray:
    """Ellipsoid x²/a² + y²/b² + z²/c² = 1 in spherical parametrisation."""
    x = a * np.sin(v) * np.cos(u)
    y = b * np.sin(v) * np.sin(u)
    z = c * np.cos(v)
    return np.stack([x, y, z], axis=-1)

def ellipsoid_metric(u: np.ndarray, v: np.ndarray,
                     a: float = 2.0, b: float = 1.0, c: float = 0.5) -> np.ndarray:
    """Compute principal curvatures and directions via differential geometry."""
    sv, cv = np.sin(v), np.cos(v)
    su, cu = np.sin(u), np.cos(u)

    # Tangent vectors
    xu = np.stack([-a * sv * su,  b * sv * cu, np.zeros_like(u)], axis=-1)
    xv = np.stack([ a * cv * cu,  b * cv * su, -c * sv           ], axis=-1)

    # Surface normal (unnormalised)
    n_un = np.cross(xu, xv)
    n_len = np.linalg.norm(n_un, axis=-1, keepdims=True) + 1e-12
    n = n_un / n_len

    # Second-order partials
    xuu = np.stack([-a * sv * cu, -b * sv * su, np.zeros_like(u)], axis=-1)
    xuv = np.stack([-a * cv * su,  b * cv * cu, np.zeros_like(u)], axis=-1)
    xvv = np.stack([-a * sv * cu, -b * sv * su, -c * sv           ], axis=-1)

    # First fundamental form coefficients
    E = (xu * xu).sum(-1)
    F = (xu * xv).sum(-1)
    G = (xv * xv).sum(-1)

    # Second fundamental form coefficients
    L = (n * xuu).sum(-1)
    M = (n * xuv).sum(-1)
    N = (n * xvv).sum(-1)

    denom = E * G - F ** 2 + 1e-12
    K = (L * N - M ** 2) / denom
    H = (E * N + G * L - 2 * F * M) / (2 * denom)

    disc = np.sqrt(np.maximum(H ** 2 - K, 0.0))
    k1 = H + disc
    k2 = H - disc

    # Principal direction corresponding to k1
    A = L - k1 * E
    B = M - k1 * F
    d1_u = B;  d1_v = -A
    d1 = d1_u[:, None] * xu + d1_v[:, None] * xv
    norm1 = np.linalg.norm(d1, axis=-1, keepdims=True) + 1e-12
    d1 = d1 / norm1
    d2 = np.cross(n, d1)

    return construct_target_metric(d1, d2, k1, k2)

def ellipsoid_normals(u: np.ndarray, v: np.ndarray,
                      a: float = 2.0, b: float = 1.0, c: float = 0.5) -> np.ndarray:
    """Outward unit normal of ellipsoid = cross(xu, xv) / |cross(xu, xv)|."""
    sv, cv = np.sin(v), np.cos(v)
    su, cu = np.sin(u), np.cos(u)
    xu = np.stack([-a * sv * su,  b * sv * cu, np.zeros_like(u)], axis=-1)
    xv = np.stack([ a * cv * cu,  b * cv * su, -c * sv           ], axis=-1)
    n = np.cross(xu, xv)
    return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12)

def ellipsoid_principal_dirs(u: np.ndarray, v: np.ndarray,
                              a: float = 2.0, b: float = 1.0, c: float = 0.5
                              ) -> Tuple[np.ndarray, np.ndarray]:
    sv, cv = np.sin(v), np.cos(v)
    su, cu = np.sin(u), np.cos(u)
    xu = np.stack([-a * sv * su,  b * sv * cu, np.zeros_like(u)], axis=-1)
    xv = np.stack([ a * cv * cu,  b * cv * su, -c * sv           ], axis=-1)
    n_un = np.cross(xu, xv)
    n_len = np.linalg.norm(n_un, axis=-1, keepdims=True) + 1e-12
    n = n_un / n_len
    xuu = np.stack([-a * sv * cu, -b * sv * su, np.zeros_like(u)], axis=-1)
    xuv = np.stack([-a * cv * su,  b * cv * cu, np.zeros_like(u)], axis=-1)
    xvv = np.stack([-a * sv * cu, -b * sv * su, -c * sv           ], axis=-1)
    E = (xu * xu).sum(-1); F = (xu * xv).sum(-1); G = (xv * xv).sum(-1)
    L = (n * xuu).sum(-1); M = (n * xuv).sum(-1); Nf = (n * xvv).sum(-1)
    denom = E * G - F ** 2 + 1e-12
    K = (L * Nf - M ** 2) / denom
    H = (E * Nf + G * L - 2 * F * M) / (2 * denom)
    disc = np.sqrt(np.maximum(H ** 2 - K, 0.0))
    k1 = H + disc
    A = L - k1 * E; B = M - k1 * F
    d1 = B[:, None] * xu + (-A)[:, None] * xv
    d1 = d1 / (np.linalg.norm(d1, axis=-1, keepdims=True) + 1e-12)
    d2 = np.cross(n, d1)
    return d1, d2


# ---------------------------------------------------------------------------
# Box / Cube  (piecewise planar sharp CAD proxy)
# ---------------------------------------------------------------------------

def _box_face_params(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    face = np.floor(u).astype(np.int64) % 6
    s = 2.0 * (u - np.floor(u)) - 1.0
    t = np.clip(v, -1.0, 1.0)
    return face, s, t


def box_surface(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    face, s, t = _box_face_params(u, v)
    pts = np.zeros((len(u), 3), dtype=np.float64)

    m = face == 0  # +X
    pts[m] = np.stack([np.ones_like(s[m]), s[m], t[m]], axis=-1)
    m = face == 1  # -X
    pts[m] = np.stack([-np.ones_like(s[m]), s[m], t[m]], axis=-1)
    m = face == 2  # +Y
    pts[m] = np.stack([s[m], np.ones_like(s[m]), t[m]], axis=-1)
    m = face == 3  # -Y
    pts[m] = np.stack([s[m], -np.ones_like(s[m]), t[m]], axis=-1)
    m = face == 4  # +Z
    pts[m] = np.stack([s[m], t[m], np.ones_like(s[m])], axis=-1)
    m = face == 5  # -Z
    pts[m] = np.stack([s[m], t[m], -np.ones_like(s[m])], axis=-1)
    return pts


def box_principal_dirs(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    face, _s, _t = _box_face_params(u, v)
    d1 = np.zeros((len(u), 3), dtype=np.float64)
    d2 = np.zeros((len(u), 3), dtype=np.float64)

    m = (face == 0) | (face == 1)  # yz face
    d1[m] = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    d2[m] = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    m = (face == 2) | (face == 3)  # xz face
    d1[m] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    d2[m] = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    m = (face == 4) | (face == 5)  # xy face
    d1[m] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    d2[m] = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return d1, d2


def box_normals(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    face, _s, _t = _box_face_params(u, v)
    n = np.zeros((len(u), 3), dtype=np.float64)
    n[face == 0] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    n[face == 1] = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    n[face == 2] = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    n[face == 3] = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    n[face == 4] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n[face == 5] = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return n


def box_metric(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    d1, d2 = box_principal_dirs(u, v)
    # Keep a stable but moderate anisotropy so the network actually learns
    # face-aligned directions on sharp CAD instead of treating the face as
    # nearly isotropic.
    k1 = np.full_like(u, 1.0, dtype=np.float64)
    k2 = np.full_like(u, 0.45, dtype=np.float64)
    return construct_target_metric(d1, d2, k1, k2, epsilon=0.05)


# Register available surfaces
SURFACES = {
    'cylinder': SurfaceSpec(
        name='cylinder',
        func=cylinder_surface,
        metric_tensor=cylinder_metric,
        principal_dirs=cylinder_principal_dirs,
        normals_fn=cylinder_normals,
        u_range=(0.0, 2 * np.pi),
        v_range=(-np.pi, np.pi),
    ),
    'torus': SurfaceSpec(
        name='torus',
        func=torus_surface,
        metric_tensor=torus_metric,
        principal_dirs=torus_principal_dirs,
        normals_fn=torus_normals,
        u_range=(0.0, 2 * np.pi),
        v_range=(0.0, 2 * np.pi),
    ),
    'saddle': SurfaceSpec(
        name='saddle',
        func=saddle_surface,
        metric_tensor=saddle_metric,
        principal_dirs=saddle_principal_dirs,
        normals_fn=saddle_normals,
        u_range=(-2.0, 2.0),
        v_range=(-2.0, 2.0),
    ),
    'sphere': SurfaceSpec(
        name='sphere',
        func=sphere_surface,
        metric_tensor=sphere_metric,
        principal_dirs=sphere_principal_dirs,
        normals_fn=sphere_normals,
        u_range=(0.0, 2 * np.pi),
        v_range=(0.15, np.pi - 0.15),
    ),
    'ellipsoid': SurfaceSpec(
        name='ellipsoid',
        func=ellipsoid_surface,
        metric_tensor=ellipsoid_metric,
        principal_dirs=ellipsoid_principal_dirs,
        normals_fn=ellipsoid_normals,
        u_range=(0.0, 2 * np.pi),
        v_range=(0.15, np.pi - 0.15),
    ),
    'box': SurfaceSpec(
        name='box',
        func=box_surface,
        metric_tensor=box_metric,
        principal_dirs=box_principal_dirs,
        normals_fn=box_normals,
        u_range=(0.0, 6.0),
        v_range=(-1.0, 1.0),
    ),
}


def sample_surface(
    spec: SurfaceSpec,
    n_points: int,
    u_range: Optional[Tuple[float, float]] = None,
    v_range: Optional[Tuple[float, float]] = None,
    noise_std: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Sample points and compute 3x3 GLOBAL metric tensors.
    u_range/v_range default to the range stored in spec (per-surface override).

    Returns:
        points:   (N, 3) 3D positions (possibly noisy)
        metric:   (N, 3, 3) global metric tensor
        dir1_out: (N, 3) principal direction 1 (larger eigenvalue)
        dir2_out: (N, 3) principal direction 2 (smaller eigenvalue)
        normals:  (N, 3) outward unit surface normals, or None if normals_fn not set
        params:   (N, 2) (u, v) parameter values
    """
    if seed is not None:
        np.random.seed(seed)

    # Use per-surface range unless caller overrides
    u_range = u_range if u_range is not None else spec.u_range
    v_range = v_range if v_range is not None else spec.v_range

    # Sample parameters uniformly
    u = np.random.uniform(u_range[0], u_range[1], n_points)
    v = np.random.uniform(v_range[0], v_range[1], n_points)

    # Compute points
    points = spec.func(u, v)  # (n_points, 3)

    # Compute normals BEFORE adding noise (normals are surface properties, not of noisy cloud)
    normals = spec.normals_fn(u, v) if spec.normals_fn is not None else None  # (N, 3) or None

    # Add noise if requested
    if noise_std > 0:
        points += np.random.normal(0, noise_std, points.shape)

    # Compute metric tensors (N, 3, 3)
    metric = spec.metric_tensor(u, v)

    # Compute principal directions (unit vectors)
    dir1, dir2 = spec.principal_dirs(u, v)

    # ── Convention: dir1 = larger metric eigenvalue direction ──────────────
    lam_d1 = np.einsum('ni,nij,nj->n', dir1, metric, dir1)   # (N,)
    lam_d2 = np.einsum('ni,nij,nj->n', dir2, metric, dir2)   # (N,)
    swap = lam_d1 < lam_d2                                    # (N,)
    dir1_out = np.where(swap[:, None], dir2, dir1)
    dir2_out = np.where(swap[:, None], dir1, dir2)

    params = np.stack([u, v], axis=-1)

    return points, metric, dir1_out, dir2_out, normals, params


def generate_mixed_synthetic_dataset(
    n_total: int,
    proportions: dict = None,
    noise_std: float = 0.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate mixed dataset.
    """
    if proportions is None:
        proportions = {'cylinder': 1./3, 'torus': 1./3, 'saddle': 1./3}
    np.random.seed(seed)
    points_list, metrics_list, dir1_list, dir2_list, labels_list = [], [], [], [], []
    for name, frac in proportions.items():
        n = int(round(n_total * frac))
        if n == 0:
            continue
        spec = SURFACES[name]
        p, m, d1, d2, _normals, _ = sample_surface(spec, n, noise_std=noise_std, seed=np.random.randint(1e6))
        points_list.append(p)
        metrics_list.append(m)
        dir1_list.append(d1)
        dir2_list.append(d2)
        labels_list.append(np.full(n, fill_value=name, dtype=object))
    
    if not points_list:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        
    points = np.concatenate(points_list, axis=0)
    metrics = np.concatenate(metrics_list, axis=0)
    dir1 = np.concatenate(dir1_list, axis=0)
    dir2 = np.concatenate(dir2_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return points, metrics, dir1, dir2, labels
