"""Anisotropic mesh deformation via editable per-quad target metrics."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..geometry.metric_utils import interpolate_metric
from ..optimization.pd_solver import ProjectiveDynamicsSolver


def metric_from_anisotropy(
    scale_u: float = 2.0,
    scale_v: float = 0.5,
    angle_deg: float = 0.0,
) -> np.ndarray:
    """Construct a 2x2 SPD metric from principal scales and in-plane angle."""
    su = max(float(scale_u), 1e-6)
    sv = max(float(scale_v), 1e-6)
    theta = np.deg2rad(float(angle_deg))
    ct, st = np.cos(theta), np.sin(theta)
    R = np.array([[ct, -st], [st, ct]], dtype=np.float64)
    D = np.diag([su * su, sv * sv]).astype(np.float64)
    return R @ D @ R.T


class DeformationSolver(ProjectiveDynamicsSolver):
    """
    PD solver wrapper with mutable target metrics and brush-style local edits.
    """

    def __init__(
        self,
        vertices: np.ndarray,
        quads: np.ndarray,
        base_metric_field: Optional[np.ndarray] = None,
        ref_square: Optional[np.ndarray] = None,
        smoothness_weight: float = 0.01,
        anti_flip_weight: float = 10.0,
        anti_flip_eps: float = 1e-6,
    ):
        if quads.ndim != 2 or quads.shape[1] != 4:
            raise ValueError(
                f"DeformationSolver expects quad faces (M,4); got shape {quads.shape}."
            )
        m_quads = int(quads.shape[0])

        if base_metric_field is None:
            metrics = np.repeat(np.eye(2, dtype=np.float64)[None, :, :], m_quads, axis=0)
        else:
            metrics = np.asarray(base_metric_field, dtype=np.float64)
            if metrics.shape != (m_quads, 2, 2):
                raise ValueError(
                    "base_metric_field must have shape (num_quads, 2, 2), "
                    f"got {metrics.shape}."
                )

        self.metric_field = metrics.copy()

        def target_metric_func(quad_idx: int) -> np.ndarray:
            return self.metric_field[quad_idx]

        super().__init__(
            vertices=vertices,
            quads=quads,
            target_metric_field=target_metric_func,
            ref_square=ref_square,
            smoothness_weight=smoothness_weight,
            anti_flip_weight=anti_flip_weight,
            anti_flip_eps=anti_flip_eps,
        )

    def apply_stroke(
        self,
        center: np.ndarray,
        radius: float,
        new_metric: np.ndarray,
        strength: float = 1.0,
        falloff: str = "gaussian",
        custom_falloff: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> int:
        """
        Blend a target metric into quads near `center`.

        Returns:
            Number of affected quads.
        """
        center = np.asarray(center, dtype=np.float64).reshape(3)
        new_metric = np.asarray(new_metric, dtype=np.float64).reshape(2, 2)
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError(f"radius must be > 0, got {radius}")

        strength = float(np.clip(strength, 0.0, 1.0))
        if strength <= 0.0:
            return 0

        quad_centers = self.V[self.quads].mean(axis=1)  # (M, 3)
        dists = np.linalg.norm(quad_centers - center[None, :], axis=1)
        mask = dists <= radius
        if not np.any(mask):
            return 0

        x = np.clip(dists[mask] / radius, 0.0, 1.0)
        if custom_falloff is not None:
            weights = np.asarray(custom_falloff(x), dtype=np.float64)
        elif falloff == "linear":
            weights = 1.0 - x
        else:
            # gaussian default
            weights = np.exp(-(x * x))

        weights = np.clip(strength * weights, 0.0, 1.0)
        idxs = np.where(mask)[0]
        for qi, w in zip(idxs, weights):
            self.metric_field[qi] = interpolate_metric(self.metric_field[qi], new_metric, float(w))
        return int(len(idxs))

    def deform(
        self,
        surface_points: Optional[np.ndarray] = None,
        surface_normals: Optional[np.ndarray] = None,
        max_iter_per_stroke: int = 5,
        num_strokes: int = 1,
        tol: float = 1e-4,
        callback: Optional[Callable] = None,
    ) -> np.ndarray:
        """
        Run PD relaxation after one or more externally applied strokes.
        """
        if surface_points is None:
            surface_points = self.V

        for _ in range(max(1, int(num_strokes))):
            self.optimize(
                surface_points=surface_points,
                surface_normals=surface_normals,
                max_iter=int(max_iter_per_stroke),
                tol=float(tol),
                callback=callback,
            )
        return self.V
