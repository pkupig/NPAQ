#!/usr/bin/env python
"""
run_miq.py  —  Standalone IGL MIQ runner.

Converts any triangle mesh to a field-aligned quad mesh using the C++ MIQ
binary (cpp_miq/build/run_miq) via subprocess.  Works with or without a
trained NPAQ checkpoint:

  • Without --checkpoint : uses an isotropic cross-field (all θ = 0).
    Still produces a structurally valid quad mesh; just not neural-guided.

  • With --checkpoint    : runs DGCNN to get per-vertex metric tensors, then
    computes the Ginzburg–Landau cross-field aligned to the metric.

Usage examples:
  # Isotropic field baseline (no checkpoint needed)
  python scripts/run_miq.py \\
      --input   /path/to/mesh.obj \\
      --output  outputs/mesh_igl_quad.obj \\
      --gradient-size 20

  # Neural-guided field
  python scripts/run_miq.py \\
      --input      /path/to/mesh.obj \\
      --output     outputs/mesh_igl_quad.obj \\
      --checkpoint logs/synthetic_run/20260217_192021/best.pth \\
      --gradient-size 20
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import inspect

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geometry.miq_wrapper import (
    miq_quadrangulate_igl,
    initial_quad_mesh_from_pointcloud,
    poisson_surface_reconstruction,
    _DEFAULT_BINARY,
    compute_frame_field,
)
from src.geometry.crossfield import (
    compute_vertex_frames,
    solve_crossfield_gl,
    detect_singularities_from_crossfield,
)
from src.utils.mesh_io import read_pointcloud, write_mesh
from src.utils.topology_cert import certify_quad_topology, format_topology_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_triangle_mesh(path: str):
    """
    Load a triangle mesh from OBJ/PLY/OFF/STL.
    Returns (V, F) numpy arrays, or raises if not a triangle mesh.
    Ensures CCW winding (outward normals) via signed-volume test.
    """
    import trimesh
    m = trimesh.load(path, force='mesh', process=False)
    if not isinstance(m, trimesh.Trimesh):
        raise ValueError(f"Expected a triangle mesh, got {type(m)} from {path}")
    V = np.array(m.vertices, dtype=np.float64)
    F = np.array(m.faces,    dtype=np.int64)
    if F.shape[1] != 3:
        raise ValueError(f"Expected triangle faces; got shape {F.shape}")
    # Negative signed volume ⟹ CW winding ⟹ all downstream quads will be reversed.
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    signed_vol = float(np.einsum('fi,fi->f', v0, np.cross(v1, v2)).sum())
    if signed_vol < 0:
        F = F[:, [0, 2, 1]]
    return V, F


def _neural_crossfield(V, F, frames, checkpoint_path, k_neighbors=20, device=None):
    """
    Use DGCNN to predict per-vertex metric tensors, then solve GL cross-field.
    Returns u_complex (N,).
    """
    import torch
    from scipy.spatial import KDTree
    from src.models.dgcnn import (
        DGCNN, infer_in_dims_from_checkpoint,
        infer_predict_confidence_from_checkpoint,
    )
    from src.geometry.metric_utils import params_to_tensor
    from src.geometry.laplacian import smooth_metric_field_implicit
    from src.dataset.lcf import compute_local_canonical_frame

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"  [neural] Loading checkpoint: {checkpoint_path}")
    ckpt  = torch.load(checkpoint_path, map_location=device)
    mcfg  = ckpt.get('config', {}).get('model', {})
    k_nn  = mcfg.get('k', k_neighbors)
    state = ckpt.get('model_state_dict', ckpt)
    in_dims = infer_in_dims_from_checkpoint(state)
    predict_confidence = infer_predict_confidence_from_checkpoint(state)

    model = DGCNN(
        k=k_nn,
        emb_dims=mcfg.get('emb_dims', 256),
        dropout=mcfg.get('dropout', 0.5),
        in_dims=in_dims,
        predict_confidence=predict_confidence,
        max_log_half=mcfg.get('max_log_half', 1.5),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    points = V  # use mesh vertices as the "point cloud"
    N = len(points)
    tree = KDTree(points)

    normals = None
    if in_dims == 6:
        # Use mesh geometry to estimate per-vertex normals for 6D checkpoints.
        v0 = V[F[:, 0]]
        v1 = V[F[:, 1]]
        v2 = V[F[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        fn /= (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12)
        normals = np.zeros_like(V, dtype=np.float64)
        np.add.at(normals, F[:, 0], fn)
        np.add.at(normals, F[:, 1], fn)
        np.add.at(normals, F[:, 2], fn)
        normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)

    print(f"  [neural] Computing LCF for {N} vertices …")
    local_feats, basis_list = [], []
    for gi in range(N):
        lc, basis, nbr_normals = compute_local_canonical_frame(
            points, gi, k=k_nn, normals=normals, return_neighbors=False, tree=tree
        )
        if in_dims == 6 and nbr_normals is not None:
            feat = np.concatenate([lc, nbr_normals], axis=-1)  # (k, 6)
        else:
            feat = lc                                           # (k, 3)
        local_feats.append(feat)
        basis_list.append(basis)

    coords_t = torch.from_numpy(np.stack(local_feats)).float().to(device)
    metric_np = np.zeros((N, 2, 2), dtype=np.float32)
    conf_np = np.ones((N,), dtype=np.float32)

    batch = 1024
    for i in range(0, N, batch):
        x = coords_t[i:i+batch].transpose(2, 1)
        with torch.no_grad():
            out = model(x)
        s1 = out[:, 0]; s2 = out[:, 1]; c = out[:, 2]; s_val = out[:, 3]
        metric_np[i:i+batch] = params_to_tensor(s1, s2, c, s_val).cpu().numpy()
        if out.shape[1] in (5, 8):
            conf_np[i:i+batch] = out[:, 4].clamp(0.0, 1.0).detach().cpu().numpy()

    # Light smoothing — λ=0.5 collapses anisotropy; λ=0.05 preserves it while
    # removing high-frequency noise from LCF misalignment between neighbours.
    print("  [neural] Smoothing metric field (λ=0.05) …")
    metric_np = smooth_metric_field_implicit(metric_np, points, lambda_smooth=0.05, k=k_nn)

    print("  [neural] Solving GL cross-field …")
    sig = inspect.signature(solve_crossfield_gl)
    kwargs = dict(
        V=V,
        F=F,
        M_vert=metric_np.astype(np.float64),
        frames=frames,
        mu=10.0,
    )
    if 'guidance_confidence' in sig.parameters:
        kwargs['guidance_confidence'] = conf_np.astype(np.float64)
    u, _ = solve_crossfield_gl(**kwargs)
    # Return both complex field and metric so caller can pass anisotropy to MIQ
    return u, metric_np


def _isotropic_crossfield(N: int) -> np.ndarray:
    """Return u = 1+0j everywhere (isotropic, θ = 0)."""
    return np.ones(N, dtype=complex)


def _mean_edge_length(V: np.ndarray, F: np.ndarray) -> float:
    """Estimate mean triangle-edge length for MIQ auto gradient sizing."""
    edges = np.concatenate(
        [F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]],
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    if len(edges) == 0:
        return 1.0
    lens = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
    return float(np.mean(lens) + 1e-12)


def _build_retry_profiles(
    gradient_size: float,
    stiffness: float,
    max_param_dist: float,
    enable_retry: bool,
) -> list[dict]:
    """Build MIQ retry schedule. First profile is always the user/base profile."""
    base = {
        'gradient_size': float(gradient_size),
        'stiffness': float(stiffness),
        'max_param_dist': float(max_param_dist),
    }
    profiles = [base]
    if not enable_retry:
        return profiles
    profiles.extend([
        {
            'gradient_size': float(gradient_size * 0.8),
            'stiffness': float(stiffness * 0.8),
            'max_param_dist': float(max_param_dist * 0.9),
        },
        {
            'gradient_size': float(gradient_size * 0.65),
            'stiffness': float(stiffness * 0.65),
            'max_param_dist': float(max_param_dist * 0.8),
        },
        {
            'gradient_size': float(gradient_size * 1.2),
            'stiffness': float(stiffness * 1.25),
            'max_param_dist': float(max_param_dist * 0.9),
        },
    ])
    return profiles


def _topology_penalty(report: dict) -> tuple:
    """
    Lower is better. Prioritize manifoldness/integrity over boundary count.
    """
    return (
        int(report.get('invalid_vertex_indices', 0)),
        int(report.get('high_multiplicity_edges', 0)),
        int(report.get('nonmanifold_edges', 0)),
        int(report.get('nonmanifold_vertices', 0)),
        int(report.get('degenerate_faces', 0)),
        int(report.get('duplicate_faces', 0)),
        int(report.get('non_quad_faces', 0)),
        int(report.get('boundary_edges', 0)),
        -int(report.get('num_faces', 0)),
    )


# ---------------------------------------------------------------------------
# Quality report
# ---------------------------------------------------------------------------

def quad_quality_report(V: np.ndarray, Q: np.ndarray, label: str = "") -> dict:
    """
    Compute basic structural quality metrics for a quad mesh.

    Returns a dict with keys:
        n_quads, n_verts, mean_angle_dev_deg, max_angle_dev_deg,
        pct_good_quads (< 30° deviation), mean_aspect_ratio, max_aspect_ratio.
    """
    n_quads = len(Q)
    if n_quads == 0:
        return {}

    angle_devs = []
    aspect_ratios = []

    for q in Q:
        pts = V[q]   # (4, 3)
        # Corner angles
        for i in range(4):
            prev = pts[(i - 1) % 4]
            curr = pts[i]
            nxt  = pts[(i + 1) % 4]
            e1 = prev - curr
            e2 = nxt  - curr
            n1 = np.linalg.norm(e1)
            n2 = np.linalg.norm(e2)
            if n1 < 1e-12 or n2 < 1e-12:
                angle_devs.append(90.0)
                continue
            cos_a = np.clip(np.dot(e1, e2) / (n1 * n2), -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_a))
            angle_devs.append(abs(angle - 90.0))

        # Aspect ratio: max edge / min edge
        edges = [np.linalg.norm(pts[(i+1)%4] - pts[i]) for i in range(4)]
        mx = max(edges); mn = min(edges)
        aspect_ratios.append(mx / (mn + 1e-12))

    angle_arr = np.array(angle_devs)
    ar_arr    = np.array(aspect_ratios)

    metrics = {
        'label'             : label,
        'n_verts'           : len(V),
        'n_quads'           : n_quads,
        'mean_angle_dev_deg': float(angle_arr.mean()),
        'max_angle_dev_deg' : float(angle_arr.max()),
        'pct_good_quads'    : float((angle_arr.reshape(-1, 4).max(axis=1) < 30).mean() * 100),
        'mean_aspect_ratio' : float(ar_arr.mean()),
        'max_aspect_ratio'  : float(ar_arr.max()),
    }
    return metrics


def print_quality(m: dict) -> None:
    lbl = m.get('label', '')
    pad = f"[{lbl}] " if lbl else ""
    print(f"  {pad}Quads              : {m['n_quads']:,}")
    print(f"  {pad}Mean angle dev     : {m['mean_angle_dev_deg']:.2f}°")
    print(f"  {pad}Max angle dev      : {m['max_angle_dev_deg']:.2f}°")
    print(f"  {pad}Good quads (<30°)  : {m['pct_good_quads']:.1f}%")
    print(f"  {pad}Mean aspect ratio  : {m['mean_aspect_ratio']:.3f}")
    print(f"  {pad}Max aspect ratio   : {m['max_aspect_ratio']:.3f}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Standalone IGL MIQ quad mesher (C++ subprocess)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--input',   required=True, help='Triangle mesh (OBJ/PLY/OFF/STL)')
    p.add_argument('--output',  required=True, help='Output quad mesh (.obj)')
    p.add_argument('--checkpoint', default=None,
                   help='NPAQ checkpoint for neural cross-field guidance (optional)')
    p.add_argument('--gradient-size', type=float, default=-1.0,
                   help='MIQ gradient scale: larger → fewer quads; ≤0 = auto (5× avg edge length)')
    p.add_argument('--stiffness', type=float, default=5.0,
                   help='MIQ solver stiffness weight')
    p.add_argument('--max-param-dist', type=float, default=0.7,
                   help='Max UV distance for integer-grid vertex snap')
    p.add_argument('--binary', default=None,
                   help='Path to compiled run_miq binary (auto-detected if omitted)')
    p.add_argument('--k-neighbors', type=int, default=20,
                   help='k-NN count for neural LCF computation')
    p.add_argument('--stats', action='store_true',
                   help='Print quad quality statistics after reconstruction')
    p.add_argument('--allow-invalid-topology', action='store_true',
                   help='Allow writing output even if topology certification fails')
    p.add_argument('--no-retry', action='store_true',
                   help='Disable automatic MIQ parameter retries on topology failure')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    binary = args.binary or os.path.abspath(_DEFAULT_BINARY)
    if not os.path.isfile(binary):
        print(f"[run_miq] ERROR: C++ binary not found at: {binary}")
        print( "         Build it first:  bash cpp_miq/build.sh")
        sys.exit(1)

    print(f"[run_miq] Input  : {args.input}")
    print(f"[run_miq] Output : {args.output}")
    print(f"[run_miq] Binary : {binary}")

    # ── Load mesh ─────────────────────────────────────────────────────────────
    t0 = time.time()
    print("Loading triangle mesh …")
    V, F = _load_triangle_mesh(args.input)
    print(f"  {len(V):,} vertices, {len(F):,} triangles  ({time.time()-t0:.1f}s)")

    # ── Build tangent frames ───────────────────────────────────────────────────
    print("Computing vertex tangent frames …")
    frames = compute_vertex_frames(V, F)

    # ── Cross-field ────────────────────────────────────────────────────────────
    metric_for_miq = None   # will be set only when neural checkpoint is used
    if args.checkpoint:
        print("Computing neural-guided cross-field …")
        u, metric_for_miq = _neural_crossfield(V, F, frames, args.checkpoint,
                                                args.k_neighbors)
    else:
        print("Using isotropic cross-field (no checkpoint supplied) …")
        u = _isotropic_crossfield(len(V))
    sing = detect_singularities_from_crossfield(V, F, u)
    print(
        "  [field] singularities from cross-field: "
        f"total={sing['num_singular_faces']}, +={sing['num_positive']}, -={sing['num_negative']}"
    )

    # ── IGL MIQ via subprocess ─────────────────────────────────────────────────
    print("Running IGL MIQ …")
    t1 = time.time()
    grad_base = float(args.gradient_size)
    if grad_base <= 0.0:
        grad_base = 5.0 * _mean_edge_length(V, F)
        print(f"  [miq] auto gradient-size -> {grad_base:.6f}")

    profiles = _build_retry_profiles(
        gradient_size=grad_base,
        stiffness=float(args.stiffness),
        max_param_dist=float(args.max_param_dist),
        enable_retry=(not args.no_retry),
    )
    best_fallback = None
    attempt_reports = []
    quadV = quadF = None
    topo_ok = False
    topo_report = None

    for ai, prof in enumerate(profiles, start=1):
        try:
            qv, qf = miq_quadrangulate_igl(
                V, F,
                frame_field=frames,
                u_complex=u,
                M_vert=metric_for_miq,
                gradient_size=float(prof['gradient_size']),
                stiffness=float(prof['stiffness']),
                max_param_dist=float(prof['max_param_dist']),
                binary_path=binary,
            )
            ok, rep = certify_quad_topology(
                qv, qf, allow_boundary=True, require_all_quads=True
            )
            msg = (
                f"attempt {ai}/{len(profiles)}: quads={len(qf)}, "
                f"g={float(prof['gradient_size']):.6f}, "
                f"k={float(prof['stiffness']):.6f}, "
                f"max_dist={float(prof['max_param_dist']):.6f}, "
                f"{format_topology_report(rep)}"
            )
            attempt_reports.append(msg)
            print(f"  [topo] {msg} -> {'PASS' if ok else 'FAIL'}")

            cand = (_topology_penalty(rep), len(qf), qv, qf, rep, ok)
            if best_fallback is None or cand[0] < best_fallback[0] or (
                cand[0] == best_fallback[0] and cand[1] > best_fallback[1]
            ):
                best_fallback = cand
            if ok:
                quadV, quadF, topo_report, topo_ok = qv, qf, rep, True
                break
        except Exception as exc:
            msg = (
                f"attempt {ai}/{len(profiles)}: EXCEPTION "
                f"(g={float(prof['gradient_size']):.6f}, "
                f"k={float(prof['stiffness']):.6f}, "
                f"max_dist={float(prof['max_param_dist']):.6f}) -> {exc}"
            )
            attempt_reports.append(msg)
            print(f"  [topo] {msg}")

    if quadV is None:
        if best_fallback is None:
            raise RuntimeError("MIQ failed before producing any candidate mesh.")
        _, _, bqv, bqf, brep, bok = best_fallback
        quadV, quadF, topo_report, topo_ok = bqv, bqf, brep, bok
        print("  [topo] using best fallback candidate from retries.")

    print(f"  {len(quadF):,} quads generated  ({time.time()-t1:.1f}s)")
    print(f"Topology report: {format_topology_report(topo_report)}")
    if (not topo_ok) and (not args.allow_invalid_topology):
        details = "\n".join(attempt_reports)
        raise RuntimeError(
            "Topology certification failed after MIQ retries.\n"
            f"Attempts:\n{details}\n"
            "Pass --allow-invalid-topology to export anyway, or tune "
            "--gradient-size/--stiffness/--max-param-dist."
        )

    # ── Save output ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    write_mesh(args.output, quadV, quadF)
    print(f"Saved → {args.output}")

    # ── Optional quality report ────────────────────────────────────────────────
    if args.stats:
        print("\nQuality report:")
        m = quad_quality_report(quadV, quadF, label='IGL-MIQ')
        print_quality(m)


if __name__ == '__main__':
    main()
