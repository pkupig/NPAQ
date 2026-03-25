#!/usr/bin/env python
"""
Brush-based anisotropic deformation demo.

This script applies one or more metric-editing strokes to a quad mesh and
runs Projective Dynamics relaxation after each stroke.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.applications.deformation import DeformationSolver, metric_from_anisotropy
from src.utils.mesh_io import read_mesh, write_mesh


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Brush-based anisotropic quad-mesh deformation")
    p.add_argument("--input", required=True, help="Input quad mesh (.obj/.off)")
    p.add_argument("--output", required=True, help="Output deformed quad mesh (.obj)")
    p.add_argument(
        "--center",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Stroke center in world coordinates (default: mesh centroid)",
    )
    p.add_argument(
        "--radius",
        type=float,
        default=0.0,
        help="Stroke radius (default: 0.2 * bbox diagonal)",
    )
    p.add_argument(
        "--strength",
        type=float,
        default=0.9,
        help="Metric blend strength in [0,1]",
    )
    p.add_argument(
        "--falloff",
        choices=["gaussian", "linear"],
        default="gaussian",
        help="Stroke falloff profile",
    )
    p.add_argument("--scale-u", type=float, default=2.0, help="Target metric scale on principal axis u")
    p.add_argument("--scale-v", type=float, default=0.5, help="Target metric scale on principal axis v")
    p.add_argument("--angle-deg", type=float, default=0.0, help="In-plane orientation angle (degrees)")
    p.add_argument("--strokes", type=int, default=1, help="Number of repeated strokes")
    p.add_argument("--iters", type=int, default=20, help="PD iterations after each stroke")
    p.add_argument("--smoothness", type=float, default=0.1, help="PD smoothness weight (mu)")
    p.add_argument("--anti-flip-weight", type=float, default=10.0, help="PD anti-flip correction weight")
    p.add_argument("--anti-flip-eps", type=float, default=1e-6, help="PD anti-flip signed-area threshold")
    p.add_argument("--vis", action="store_true", help="Visualize before/after meshes with polyscope")
    return p.parse_args()


def _resolve_defaults(vertices: np.ndarray, center_arg, radius_arg: float):
    bb_min = vertices.min(axis=0)
    bb_max = vertices.max(axis=0)
    diag = float(np.linalg.norm(bb_max - bb_min))

    if center_arg is None:
        centroid = vertices.mean(axis=0)
        nearest = int(np.argmin(np.linalg.norm(vertices - centroid[None, :], axis=1)))
        center = vertices[nearest]
    else:
        center = np.asarray(center_arg, dtype=np.float64)

    radius = float(radius_arg)
    if radius <= 0.0:
        radius = max(1e-8, 0.2 * diag)

    return center, radius


def _validate_quad_mesh(faces: np.ndarray) -> None:
    if faces.ndim != 2 or faces.shape[1] != 4:
        raise ValueError(
            "demo_deformation.py requires a quad mesh as input; "
            f"got faces with shape {faces.shape}."
        )


def _read_obj_preserve_faces(path: str):
    verts = []
    faces = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                face = []
                for p in parts:
                    idx = p.split("/")[0]
                    face.append(int(idx) - 1)
                faces.append(face)
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _read_off_preserve_faces(path: str):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if lines[0] != "OFF":
        raise ValueError(f"Not a valid OFF file: {path}")
    n_verts, n_faces, _ = map(int, lines[1].split())
    verts = [list(map(float, lines[2 + i].split())) for i in range(n_verts)]
    faces = []
    start = 2 + n_verts
    for i in range(n_faces):
        parts = lines[start + i].split()
        n = int(parts[0])
        faces.append([int(x) for x in parts[1 : 1 + n]])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _read_mesh_preserve_faces(path: str):
    ext = Path(path).suffix.lower()
    if ext == ".obj":
        return _read_obj_preserve_faces(path)
    if ext == ".off":
        return _read_off_preserve_faces(path)
    # Fallback for other formats; may triangulate polygons depending on backend.
    v, f = read_mesh(path)
    return np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64)


def _maybe_visualize(before_v: np.ndarray, faces: np.ndarray, after_v: np.ndarray) -> None:
    try:
        import polyscope as ps
    except ImportError:
        print("[warn] --vis requested but polyscope is not installed; skipping visualization.")
        return

    ps.init()
    ps.set_up_dir("z_up")
    ps.register_surface_mesh("before", before_v, faces, enabled=True)
    ps.register_surface_mesh("after", after_v, faces, enabled=True)
    ps.show()


def main() -> None:
    args = parse_args()

    vertices, faces = _read_mesh_preserve_faces(args.input)
    _validate_quad_mesh(faces)

    center, radius = _resolve_defaults(vertices, args.center, args.radius)

    print(f"[deform] input  : {args.input}")
    print(f"[deform] output : {args.output}")
    print(f"[deform] mesh   : {len(vertices)} verts, {len(faces)} quads")
    print(f"[deform] center : {center.tolist()}")
    print(f"[deform] radius : {radius:.6f}")

    solver = DeformationSolver(
        vertices=vertices,
        quads=faces,
        smoothness_weight=args.smoothness,
        anti_flip_weight=args.anti_flip_weight,
        anti_flip_eps=args.anti_flip_eps,
    )

    target_metric = metric_from_anisotropy(
        scale_u=args.scale_u,
        scale_v=args.scale_v,
        angle_deg=args.angle_deg,
    )

    before = solver.V.copy()
    n_strokes = max(1, int(args.strokes))
    for si in range(n_strokes):
        changed = solver.apply_stroke(
            center=center,
            radius=radius,
            new_metric=target_metric,
            strength=args.strength,
            falloff=args.falloff,
        )
        print(f"[deform] stroke {si+1}/{n_strokes}: updated {changed} quads")
        solver.deform(surface_points=vertices, max_iter_per_stroke=args.iters, num_strokes=1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_mesh(str(out_path), solver.V, faces)
    print(f"[deform] saved  : {out_path}")

    if args.vis:
        _maybe_visualize(before, faces, solver.V)


if __name__ == "__main__":
    main()
