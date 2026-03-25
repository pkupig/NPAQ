#!/usr/bin/env python
"""
compare_miq.py  —  Side-by-side quality comparison: NPAQ vs IGL MIQ baseline.

Runs BOTH quad-meshing pipelines on the same input triangle mesh and prints a
structured quality comparison table.  Both output meshes are saved so they can
be inspected in any mesh viewer (MeshLab, Blender, …).

Pipeline A — NPAQ  (Python, fully differentiable):
    Triangle mesh → GL cross-field → Poisson parametrisation → integer-grid quads

Pipeline B — IGL MIQ  (C++ subprocess, mixed-integer solver):
    Triangle mesh → GL cross-field → run_miq binary → integer-grid quads

Usage:
  python scripts/compare_miq.py \\
      --input   path/to/mesh.obj \\
      --out-dir outputs/compare/ \\
      [--checkpoint logs/.../best.pth] \\
      [--gradient-size-npaq 1.0] \\
      [--gradient-size-igl  20.0] \\
      [--crossfield-mu 10.0] \\
      [--no-igl]       # skip IGL MIQ if binary not built yet

If --checkpoint is omitted both pipelines use an isotropic cross-field (θ = 0).
"""

from __future__ import annotations
import argparse
import os
import sys
import inspect
import time
from typing import Optional

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geometry.crossfield import compute_vertex_frames, solve_crossfield_gl
from src.geometry.miq_wrapper import (
    miq_quadrangulate,
    miq_quadrangulate_igl,
    _DEFAULT_BINARY,
)
from src.utils.mesh_io import write_mesh


# ---------------------------------------------------------------------------
# Shared helpers (adapted from run_miq.py)
# ---------------------------------------------------------------------------

def _load_triangle_mesh(path: str):
    import trimesh
    m = trimesh.load(path, force='mesh', process=False)
    V = np.array(m.vertices, dtype=np.float64)
    F = np.array(m.faces,    dtype=np.int64)
    if F.shape[1] != 3:
        raise ValueError(f"Expected triangles; got {F.shape}")
    # Ensure CCW winding (outward normals) via signed-volume test.
    # Negative signed volume ⟹ CW winding ⟹ all quads will be reversed.
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    signed_vol = float(np.einsum('fi,fi->f', v0, np.cross(v1, v2)).sum())
    if signed_vol < 0:
        F = F[:, [0, 2, 1]]
    return V, F


def _neural_crossfield(V, F, frames, ckpt_path, k=20, device=None):
    import torch
    from scipy.spatial import KDTree
    from src.models.dgcnn import (
        DGCNN, load_legacy_checkpoint, infer_in_dims_from_checkpoint,
        infer_predict_singularity_from_checkpoint, infer_predict_confidence_from_checkpoint
    )
    from src.geometry.metric_utils import params_to_tensor
    from src.geometry.laplacian import smooth_metric_field_implicit
    from src.dataset.lcf import compute_local_canonical_frame
    from src.geometry.feature_lines import estimate_normals

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt  = torch.load(ckpt_path, map_location=device)
    mcfg  = ckpt.get('config', {}).get('model', {})
    state = ckpt.get('model_state_dict', ckpt)
    k_nn  = mcfg.get('k', k)
    in_dims = infer_in_dims_from_checkpoint(state)
    predict_singularity = infer_predict_singularity_from_checkpoint(state)
    predict_confidence = infer_predict_confidence_from_checkpoint(state)
    try:
        model = DGCNN(k=k_nn, emb_dims=mcfg.get('emb_dims', 256),
                      dropout=mcfg.get('dropout', 0.5), in_dims=in_dims,
                      predict_singularity=predict_singularity,
                      predict_confidence=predict_confidence,
                      max_log_half=mcfg.get('max_log_half', 1.5)).to(device)
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
    except (KeyError, RuntimeError):
        model = load_legacy_checkpoint(ckpt_path, device, k=k_nn,
                                       emb_dims=256, dropout=0.5)
    model.eval()
    points = V
    N = len(points)
    tree = KDTree(points)
    normals = None
    if in_dims == 6:
        normals = estimate_normals(points, k=max(k_nn, 20), consistent=True)
    local_feats = []
    for gi in range(N):
        lc, _, nbr_normals = compute_local_canonical_frame(
            points, gi, k=k_nn, normals=normals, return_neighbors=False, tree=tree
        )
        if nbr_normals is not None:
            local_feats.append(np.concatenate([lc, nbr_normals], axis=-1))
        else:
            local_feats.append(lc)
    coords_t = torch.from_numpy(np.stack(local_feats)).float().to(device)
    metric_np = np.zeros((N, 2, 2), dtype=np.float32)
    conf_np = np.ones((N,), dtype=np.float32)
    for i in range(0, N, 1024):
        x = coords_t[i:i+1024].transpose(2, 1)
        with torch.no_grad():
            out = model(x)
        from src.geometry.metric_utils import params_to_tensor
        metric_np[i:i+1024] = params_to_tensor(
            out[:,0], out[:,1], out[:,2], out[:,3]).cpu().numpy()
        if out.shape[1] in (5, 8):
            conf_np[i:i+1024] = out[:, 4].clamp(0.0, 1.0).detach().cpu().numpy()
    metric_np = smooth_metric_field_implicit(metric_np, points,
                                              lambda_smooth=0.5, k=k_nn)
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
    return u, metric_np


def _isotropic_crossfield(N: int) -> np.ndarray:
    return np.ones(N, dtype=complex)


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def quad_metrics(V: np.ndarray, Q: np.ndarray) -> dict:
    """
    Structural quality of a quad mesh.  All metrics are purely geometric
    (no ground-truth required).
    """
    if len(Q) == 0:
        return {'n_quads': 0}

    angle_devs = np.empty(len(Q) * 4)
    aspect_ratios = np.empty(len(Q))

    for qi, q in enumerate(Q):
        pts = V[q]
        for ci in range(4):
            prev = pts[(ci - 1) % 4]
            curr = pts[ci]
            nxt  = pts[(ci + 1) % 4]
            e1 = prev - curr; e2 = nxt - curr
            n1 = np.linalg.norm(e1); n2 = np.linalg.norm(e2)
            if n1 < 1e-12 or n2 < 1e-12:
                angle_devs[qi * 4 + ci] = 90.0
            else:
                cos_a = np.clip(np.dot(e1, e2) / (n1 * n2), -1.0, 1.0)
                angle_devs[qi * 4 + ci] = abs(np.degrees(np.arccos(cos_a)) - 90.0)
        edges = [np.linalg.norm(pts[(i+1)%4] - pts[i]) for i in range(4)]
        aspect_ratios[qi] = max(edges) / (min(edges) + 1e-12)

    per_quad_max_angle = angle_devs.reshape(-1, 4).max(axis=1)
    return {
        'n_quads'           : len(Q),
        'n_verts'           : len(V),
        'mean_angle_dev'    : float(angle_devs.mean()),
        'p50_angle_dev'     : float(np.percentile(angle_devs, 50)),
        'p90_angle_dev'     : float(np.percentile(angle_devs, 90)),
        'max_angle_dev'     : float(angle_devs.max()),
        'pct_good_30'       : float((per_quad_max_angle < 30).mean() * 100),
        'pct_good_45'       : float((per_quad_max_angle < 45).mean() * 100),
        'mean_aspect_ratio' : float(aspect_ratios.mean()),
        'p90_aspect_ratio'  : float(np.percentile(aspect_ratios, 90)),
        'max_aspect_ratio'  : float(aspect_ratios.max()),
    }


def _field_alignment(V: np.ndarray, Q: np.ndarray,
                     frames: np.ndarray, u: np.ndarray) -> float:
    """
    Mean field alignment score ∈ [0, 1].

    For each quad edge (a→b), compute cos²(angle) between the edge direction
    and the nearest principal axis of the cross-field at vertex a.
    Higher is better; 1.0 = perfect alignment.
    """
    from src.geometry.crossfield import crossfield_angles_from_complex
    theta = crossfield_angles_from_complex(u)  # (N,)
    e1_v  = frames[:, :, 0]                    # (N, 3)
    e2_v  = frames[:, :, 1]

    scores = []
    for q in Q:
        for i in range(4):
            a = q[i]; b = q[(i + 1) % 4]
            edge = V[b] - V[a]
            n_e  = np.linalg.norm(edge)
            if n_e < 1e-12:
                continue
            edge /= n_e
            # Cross-field principal directions at vertex a
            th = theta[a]
            d1 = np.cos(th) * e1_v[a] + np.sin(th) * e2_v[a]
            d2 = -np.sin(th) * e1_v[a] + np.cos(th) * e2_v[a]
            # Best alignment (4-RoSy symmetry: 4 directions)
            dots = [abs(np.dot(edge, d1)),  abs(np.dot(edge, d2)),
                    abs(np.dot(edge, -d1)), abs(np.dot(edge, -d2))]
            scores.append(max(dots) ** 2)

    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Comparison table printing
# ---------------------------------------------------------------------------

def _fmt(v, fmt='.3f'):
    if isinstance(v, float):
        return format(v, fmt)
    return str(v)


def print_comparison(m_a: dict, m_b: dict, align_a: float, align_b: float,
                     time_a: float, time_b: float) -> None:
    rows = [
        ('Quads',                  f"{m_a['n_quads']:,}",        f"{m_b['n_quads']:,}"),
        ('Vertices',               f"{m_a['n_verts']:,}",        f"{m_b['n_verts']:,}"),
        ('Time (s)',                _fmt(time_a, '.1f'),           _fmt(time_b, '.1f')),
        ('Mean angle dev (°)',      _fmt(m_a['mean_angle_dev']),   _fmt(m_b['mean_angle_dev'])),
        ('P50 angle dev (°)',       _fmt(m_a['p50_angle_dev']),    _fmt(m_b['p50_angle_dev'])),
        ('P90 angle dev (°)',       _fmt(m_a['p90_angle_dev']),    _fmt(m_b['p90_angle_dev'])),
        ('Max angle dev (°)',       _fmt(m_a['max_angle_dev']),    _fmt(m_b['max_angle_dev'])),
        ('Good quads <30° (%)',     _fmt(m_a['pct_good_30'], '.1f'), _fmt(m_b['pct_good_30'], '.1f')),
        ('Good quads <45° (%)',     _fmt(m_a['pct_good_45'], '.1f'), _fmt(m_b['pct_good_45'], '.1f')),
        ('Mean aspect ratio',      _fmt(m_a['mean_aspect_ratio']), _fmt(m_b['mean_aspect_ratio'])),
        ('P90 aspect ratio',       _fmt(m_a['p90_aspect_ratio']),  _fmt(m_b['p90_aspect_ratio'])),
        ('Max aspect ratio',       _fmt(m_a['max_aspect_ratio']),  _fmt(m_b['max_aspect_ratio'])),
        ('Field alignment',         _fmt(align_a, '.4f'),           _fmt(align_b, '.4f')),
    ]
    col0 = max(len(r[0]) for r in rows) + 2
    col1 = max(max(len(r[1]) for r in rows), 10) + 2
    col2 = max(max(len(r[2]) for r in rows), 10) + 2

    hdr = f"{'Metric':<{col0}}  {'NPAQ':>{col1}}  {'IGL-MIQ':>{col2}}"
    sep = '-' * len(hdr)
    print(sep)
    print(hdr)
    print(sep)
    for name, va, vb in rows:
        print(f"  {name:<{col0-2}}  {va:>{col1}}  {vb:>{col2}}")
    print(sep)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Compare NPAQ vs IGL-MIQ quad mesh quality',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--input',  required=True, help='Triangle mesh (OBJ/PLY/OFF/STL)')
    p.add_argument('--out-dir', default='outputs/compare',
                   help='Directory to save both output meshes')
    p.add_argument('--checkpoint', default=None,
                   help='NPAQ checkpoint (.pth); omit for isotropic cross-field')
    p.add_argument('--gradient-size-npaq', type=float, default=1.0,
                   help='gradient_size for NPAQ Python pipeline')
    p.add_argument('--gradient-size-igl',  type=float, default=-1.0,
                   help='gradient_size for IGL MIQ C++ binary (≤0 = auto from avg edge length)')
    p.add_argument('--crossfield-mu', type=float, default=10.0,
                   help='GL alignment weight μ')
    p.add_argument('--stiffness', type=float, default=5.0,
                   help='IGL MIQ stiffness')
    p.add_argument('--k-neighbors', type=int, default=20)
    p.add_argument('--binary', default=None,
                   help='Path to compiled run_miq binary; auto-detect if omitted')
    p.add_argument('--no-igl', action='store_true',
                   help='Skip IGL-MIQ pipeline (useful if binary not built yet)')
    p.add_argument('--no-npaq', action='store_true',
                   help='Skip NPAQ pipeline (run IGL-MIQ only)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input))[0]

    # ── Load mesh ─────────────────────────────────────────────────────────────
    print(f"Loading: {args.input}")
    V, F = _load_triangle_mesh(args.input)
    print(f"  {len(V):,} vertices, {len(F):,} triangles")

    # ── Shared: tangent frames ────────────────────────────────────────────────
    print("Computing vertex tangent frames …")
    frames = compute_vertex_frames(V, F)

    # ── Shared: cross-field ───────────────────────────────────────────────────
    metric_np: Optional[np.ndarray] = None
    if args.checkpoint:
        print(f"Computing neural cross-field ({args.checkpoint}) …")
        u, metric_np = _neural_crossfield(
            V, F, frames, args.checkpoint, k=args.k_neighbors
        )
    else:
        print("Using isotropic cross-field …")
        u = _isotropic_crossfield(len(V))
        metric_np = None

    results: dict = {}

    # ── Pipeline A: NPAQ ─────────────────────────────────────────────────────
    if not args.no_npaq:
        print("\n── NPAQ (Python GL + Poisson param) ──")
        t0 = time.time()
        try:
            M_vert_iso = (np.stack([np.eye(2)] * len(V))
                          if metric_np is None else metric_np)
            quadV_a, quadF_a = miq_quadrangulate(
                V, F, frame_field=frames,
                metric_field_vert=M_vert_iso,
                gradient_size=args.gradient_size_npaq,
                crossfield_mu=args.crossfield_mu,
            )
            t_a = time.time() - t0
            out_a = os.path.join(args.out_dir, f"{stem}_npaq.obj")
            write_mesh(out_a, quadV_a, quadF_a)
            print(f"  {len(quadF_a):,} quads  ({t_a:.1f}s)  →  {out_a}")
            m_a   = quad_metrics(quadV_a, quadF_a)
            al_a  = _field_alignment(quadV_a, quadF_a, frames, u)
            results['npaq'] = (m_a, al_a, t_a)
        except Exception as exc:
            print(f"  NPAQ failed: {exc}")
            results['npaq'] = None

    # ── Pipeline B: IGL MIQ ──────────────────────────────────────────────────
    if not args.no_igl:
        binary = args.binary or os.path.abspath(_DEFAULT_BINARY)
        if not os.path.isfile(binary):
            print(f"\n[compare] IGL binary not found: {binary}")
            print( "          Build it:  bash cpp_miq/build.sh")
            print( "          Or skip:   --no-igl")
        else:
            print("\n── IGL MIQ (C++ subprocess) ──")
            t0 = time.time()
            try:
                quadV_b, quadF_b = miq_quadrangulate_igl(
                    V, F,
                    frame_field=frames,
                    u_complex=u,
                    gradient_size=args.gradient_size_igl,
                    stiffness=args.stiffness,
                    binary_path=binary,
                )
                t_b = time.time() - t0
                out_b = os.path.join(args.out_dir, f"{stem}_igl_miq.obj")
                write_mesh(out_b, quadV_b, quadF_b)
                print(f"  {len(quadF_b):,} quads  ({t_b:.1f}s)  →  {out_b}")
                m_b  = quad_metrics(quadV_b, quadF_b)
                al_b = _field_alignment(quadV_b, quadF_b, frames, u)
                results['igl'] = (m_b, al_b, t_b)
            except Exception as exc:
                print(f"  IGL MIQ failed: {exc}")
                results['igl'] = None

    # ── Comparison table ──────────────────────────────────────────────────────
    if 'npaq' in results and 'igl' in results and \
            results['npaq'] is not None and results['igl'] is not None:
        m_a, al_a, t_a = results['npaq']
        m_b, al_b, t_b = results['igl']
        print("\n══ Quality comparison ══")
        print_comparison(m_a, m_b, al_a, al_b, t_a, t_b)
    elif 'npaq' in results and results['npaq'] is not None:
        m, al, t = results['npaq']
        print("\n── NPAQ quality ──")
        print(f"  [NPAQ] Quads              : {m['n_quads']:,}")
        print(f"  [NPAQ] Mean angle dev     : {m['mean_angle_dev']:.2f}°")
        print(f"  [NPAQ] P90 angle dev      : {m['p90_angle_dev']:.2f}°")
        print(f"  [NPAQ] Mean aspect ratio  : {m['mean_aspect_ratio']:.3f}")
        print(f"  [NPAQ] Field alignment    : {al:.4f}")
    elif 'igl' in results and results['igl'] is not None:
        m, al, t = results['igl']
        print("\n── IGL-MIQ quality ──")
        print(f"  [IGL-MIQ] Quads              : {m['n_quads']:,}")
        print(f"  [IGL-MIQ] Mean angle dev     : {m['mean_angle_dev']:.2f}°")
        print(f"  [IGL-MIQ] P90 angle dev      : {m['p90_angle_dev']:.2f}°")
        print(f"  [IGL-MIQ] Mean aspect ratio  : {m['mean_aspect_ratio']:.3f}")
        print(f"  [IGL-MIQ] Field alignment    : {al:.4f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
