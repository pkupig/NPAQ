#!/usr/bin/env python
"""
NPAQ reconstruction pipeline: point cloud → field-aligned all-quad mesh.

Implements Algorithm 1 (§5–§6 of PAPER.md):
    1. Neural metric prediction                (DGCNN, §3)
    2. Log-Euclidean metric smoothing          (§4)
    3. Feature line detection                  (§7)
    4. Cross-field + seamless parametrisation  (§5.1–5.3)
       (singularities inferred geometrically from cross-field winding)
    5. Projective dynamics optimisation        (§6, numpy inference path)
    6. Output quad mesh

The pipeline raises RuntimeError rather than silently falling back to
field-independent topology (midpoint subdivision was removed in §1.1).
"""

from __future__ import annotations

import argparse
import os
import sys
import yaml
import torch
import numpy as np
from scipy.spatial import KDTree
from typing import Optional

try:
    import polyscope as ps
except ImportError:
    ps = None

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.dgcnn import (
    DGCNN, load_legacy_checkpoint, infer_in_dims_from_checkpoint,
    infer_predict_singularity_from_checkpoint, infer_predict_confidence_from_checkpoint
)
from src.geometry.metric_utils import params_to_tensor
from src.geometry.laplacian import smooth_metric_field_implicit
from src.geometry.miq_wrapper import initial_quad_mesh_from_pointcloud, initial_quad_mesh_from_mesh
from src.geometry.feature_lines import (
    extract_feature_corners_from_mesh,
    extract_feature_lines,
    extract_feature_lines_from_mesh,
)
from src.geometry.crossfield import compute_vertex_frames
from src.optimization.pd_solver import ProjectiveDynamicsSolver
from src.optimization.constraints import FeaturePointConstraint, FeatureEdgeConstraint
from src.utils.mesh_io import read_pointcloud, read_pointcloud_with_normals, read_mesh, write_mesh
from src.utils.topology_cert import (
    align_triangle_faces_to_reference_normals,
    align_quad_faces_outward_from_centroid,
    cap_even_boundary_loops_with_center_quads,
    cap_even_boundary_loops_with_field_rings,
    cap_rectangular_boundary_loops_with_field,
    certify_quad_topology,
    finalize_quad_orientation_with_reference,
    format_topology_report,
    fill_boundary_quad_holes,
    orient_quad_faces_consistently,
    prune_stacked_parallel_quads,
    prune_to_largest_boundary_light_component,
    stitch_boundary_loop_pairs,
    triangulate_quads_for_reference_preview,
    zipper_self_matched_boundary_loops,
)
from src.utils.vis import visualize_with_matplotlib, init_ps, register_point_cloud, register_mesh


def _field_penalty(diag: dict | None) -> tuple:
    if not isinstance(diag, dict):
        return (0, 0, 0, 0, 0, 0, 0)
    ph = diag.get('poincare_hopf') or {}
    holo = diag.get('holonomy') or {}
    flow = diag.get('flow_conservation') or {}
    miq = diag.get('miq_extraction') or {}
    uvx = miq.get('uv_extraction') or {}
    winding = miq.get('quad_winding') or {}
    rounding = miq.get('miq_rounding') or {}
    ph_bad = 0 if bool(ph.get('satisfied', True)) else 1
    holo_bad = int(holo.get('high_order_count', 0)) + int(holo.get('cotree_violations', 0))
    flow_bad = int(flow.get('num_violations', 0))
    uv_skip = int(uvx.get('skipped_fold_overs', 0))
    uv_degen = int(uvx.get('degenerate_uv_triangles', 0))
    winding_bad = int(winding.get('corrected_quads', 0))
    flips_final = int(rounding.get('flip_count_final', 0))
    return (ph_bad, holo_bad, flow_bad, uv_skip, uv_degen, winding_bad, flips_final)


def _profile_signature(prof: dict) -> tuple:
    return (
        bool(prof.get('use_igl_backend', False)),
        round(float(prof.get('crossfield_mu', 0.0)), 6),
        round(float(prof.get('igl_gradient_size', prof.get('gradient_size', 0.0))), 6),
        round(float(prof.get('igl_stiffness', 0.0)), 6),
        int(prof.get('poisson_depth', 0)),
        int(prof.get('igl_miq_iter', 0)),
    )


def _make_adaptive_profiles(
    prof: dict,
    qdiag: dict | None,
    *,
    closed_input: bool,
    small_sharp_closed: bool = False,
) -> list[dict]:
    if not isinstance(qdiag, dict):
        return []
    flow = qdiag.get('flow_conservation') or {}
    holo = qdiag.get('holonomy') or {}
    ph = qdiag.get('poincare_hopf') or {}
    num_viol = int(flow.get('num_violations', 0))
    high_order = int(holo.get('high_order_count', 0))
    cotree_viol = int(holo.get('cotree_violations', 0))
    ph_deficit = float(ph.get('deficit', 0.0))
    ph_bad = not bool(ph.get('satisfied', True))
    if num_viol <= 0 and high_order <= 0 and cotree_viol <= 0 and not ph_bad:
        return []

    out = []
    if bool(prof.get('use_igl_backend', False)):
        base_mu = float(prof.get('crossfield_mu', 10.0))
        base_g = float(prof.get('igl_gradient_size', 40.0))
        base_s = float(prof.get('igl_stiffness', 5.0))
        base_iter = int(prof.get('igl_miq_iter', 20))

        tightened = dict(prof)
        tightened['crossfield_mu'] = min(60.0, max(base_mu + 10.0, base_mu * 1.35))
        tightened['igl_stiffness'] = min(20.0, max(base_s + 3.0, base_s * 1.3))
        tightened['igl_gradient_size'] = max(40.0 if not closed_input else 20.0, base_g * 0.75)
        tightened['igl_miq_iter'] = min(40, max(base_iter, 24))
        tightened['poisson_depth'] = min(10, int(prof.get('poisson_depth', 6)) + 1)
        out.append(tightened)

        if num_viol > 900 or cotree_viol > 1200:
            conservative = dict(tightened)
            conservative['igl_gradient_size'] = max(32.0 if not closed_input else 16.0, base_g * 0.5)
            conservative['igl_stiffness'] = min(24.0, float(tightened['igl_stiffness']) + 2.0)
            conservative['crossfield_mu'] = min(70.0, float(tightened['crossfield_mu']) + 5.0)
            out.append(conservative)

        if ph_bad and ph_deficit < -0.25:
            relaxed = dict(prof)
            relaxed['crossfield_mu'] = max(4.0, min(base_mu - 8.0, base_mu * 0.6))
            relaxed['anisotropy_eps'] = max(0.005, float(prof.get('anisotropy_eps', 0.03)) * 0.5)
            relaxed['igl_stiffness'] = max(4.0, min(base_s, base_s * 0.9))
            relaxed['igl_gradient_size'] = max(10.0 if small_sharp_closed else 16.0, base_g * 0.85)
            relaxed['igl_miq_iter'] = max(base_iter, 24 if small_sharp_closed else 20)
            out.append(relaxed)

            if small_sharp_closed:
                more_relaxed = dict(relaxed)
                more_relaxed['crossfield_mu'] = max(3.0, min(relaxed['crossfield_mu'] - 3.0, base_mu * 0.45))
                more_relaxed['anisotropy_eps'] = max(0.003, float(relaxed['anisotropy_eps']) * 0.7)
                more_relaxed['igl_gradient_size'] = max(8.0, base_g * 0.72)
                more_relaxed['igl_stiffness'] = max(4.0, float(relaxed['igl_stiffness']) * 0.9)
                out.append(more_relaxed)
                if num_viol > 10 or high_order > 10:
                    repaired_relaxed = dict(more_relaxed)
                    repaired_relaxed['crossfield_mu'] = max(
                        3.5, min(float(more_relaxed['crossfield_mu']) + 0.75, base_mu * 0.55)
                    )
                    repaired_relaxed['igl_gradient_size'] = max(
                        8.0, min(float(more_relaxed['igl_gradient_size']) + 0.4, base_g * 0.8)
                    )
                    repaired_relaxed['igl_stiffness'] = max(
                        4.5, min(float(more_relaxed['igl_stiffness']) + 0.2, base_s)
                    )
                    repaired_relaxed['repair_high_order'] = True
                    repaired_relaxed['repair_flow'] = True
                    repaired_relaxed['guidance_override_reliability_floor'] = max(
                        1.0, float(prof.get('guidance_override_reliability_floor', 1.0))
                    )
                    repaired_relaxed['high_order_repair_max_rounds'] = max(
                        2, int(prof.get('high_order_repair_max_rounds', 1))
                    )
                    repaired_relaxed['flow_repair_max_rounds'] = max(
                        2, int(prof.get('flow_repair_max_rounds', 1))
                    )
                    out.append(repaired_relaxed)
    return out


def _topology_penalty(report: dict, expected_boundary_loops: int = 0, diag: dict | None = None) -> tuple:
    """
    Lower is better. Prefer fewer topology defects, then fewer *excess*
    boundary loops beyond what the input already contains, then fewer open
    boundary edges, then prefer more faces.
    """
    excess_loops = max(0, int(report.get('boundary_loops', 0)) - int(expected_boundary_loops))
    return (
        int(report.get('high_multiplicity_edges', 0)),
        int(report.get('nonmanifold_edges', 0)),
        int(report.get('degenerate_faces', 0)),
        int(report.get('duplicate_faces', 0)),
        int(report.get('non_quad_faces', 0)),
        int(report.get('boundary_chains', 0)),
        int(report.get('boundary_irregular_vertices', 0)),
        excess_loops,
        int(report.get('boundary_edges', 0)),
        *_field_penalty(diag),
        -int(report.get('num_faces', 0)),
    )


def _repair_penalty(report: dict, expected_boundary_loops: int = 0, diag: dict | None = None) -> tuple:
    excess_loops = max(0, int(report.get('boundary_loops', 0)) - int(expected_boundary_loops))
    return (
        int(report.get('high_multiplicity_edges', 0)),
        int(report.get('nonmanifold_edges', 0)),
        int(report.get('degenerate_faces', 0)),
        int(report.get('duplicate_faces', 0)),
        int(report.get('non_quad_faces', 0)),
        int(report.get('boundary_chains', 0)),
        int(report.get('boundary_irregular_vertices', 0)),
        excess_loops,
        int(report.get('boundary_edges', 0)),
        *_field_penalty(diag),
    )


def _repair_improves(before: dict, after: dict, expected_boundary_loops: int = 0, diag: dict | None = None) -> bool:
    return _repair_penalty(after, expected_boundary_loops, diag) < _repair_penalty(before, expected_boundary_loops, diag)


def _quad_quality_metrics(vertices: np.ndarray, faces: np.ndarray) -> dict:
    V = np.asarray(vertices, dtype=np.float64)
    Q = np.asarray(faces, dtype=np.int64)
    if Q.ndim != 2 or Q.shape[0] == 0:
        return {
            'mean_angle_dev': 0.0,
            'p90_angle_dev': 0.0,
            'mean_aspect_ratio': 0.0,
            'p90_aspect_ratio': 0.0,
            'mean_planarity': 0.0,
            'p90_planarity': 0.0,
        }

    angle_devs = np.empty(len(Q) * 4, dtype=np.float64)
    aspect_ratios = np.empty(len(Q), dtype=np.float64)
    planarity = np.empty(len(Q), dtype=np.float64)
    for qi, q in enumerate(Q):
        pts = V[q]
        edges = np.empty(4, dtype=np.float64)
        for i in range(4):
            a = pts[(i - 1) % 4] - pts[i]
            b = pts[(i + 1) % 4] - pts[i]
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            if na < 1e-12 or nb < 1e-12:
                angle_devs[qi * 4 + i] = 90.0
            else:
                cos_a = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
                angle_devs[qi * 4 + i] = abs(np.degrees(np.arccos(cos_a)) - 90.0)
            edges[i] = float(np.linalg.norm(pts[(i + 1) % 4] - pts[i]))
        aspect_ratios[qi] = float(edges.max() / (edges.min() + 1e-12))
        n0 = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        n0n = float(np.linalg.norm(n0))
        d = abs(float(np.dot(pts[3] - pts[0], n0))) / max(n0n, 1e-12)
        scale = max(
            float(np.linalg.norm(pts[2] - pts[0])),
            float(np.linalg.norm(pts[3] - pts[1])),
            1e-12,
        )
        planarity[qi] = d / scale

    return {
        'mean_angle_dev': float(angle_devs.mean()),
        'p90_angle_dev': float(np.percentile(angle_devs, 90)),
        'mean_aspect_ratio': float(aspect_ratios.mean()),
        'p90_aspect_ratio': float(np.percentile(aspect_ratios, 90)),
        'mean_planarity': float(planarity.mean()),
        'p90_planarity': float(np.percentile(planarity, 90)),
    }


def _triangulate_quads(faces: np.ndarray) -> np.ndarray:
    Q = np.asarray(faces, dtype=np.int64)
    if Q.ndim != 2 or Q.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.int64)
    return np.vstack([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]]).astype(np.int64)


def _transfer_metric_to_vertices(points: np.ndarray, metric_field: np.ndarray, V: np.ndarray) -> np.ndarray:
    tree = KDTree(points)
    _, idx = tree.query(V)
    return np.asarray(metric_field, dtype=np.float64)[idx]


def _transfer_scalar_to_vertices(points: np.ndarray, values: np.ndarray, V: np.ndarray) -> np.ndarray:
    tree = KDTree(points)
    _, idx = tree.query(V)
    return np.asarray(values)[idx]


def _metric_field_dirs_on_quads(
    points: np.ndarray,
    metric_field: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    V = np.asarray(vertices, dtype=np.float64)
    M = _transfer_metric_to_vertices(points, metric_field, V)
    tri = _triangulate_quads(faces)
    frames = compute_vertex_frames(V, tri)
    eigvals, eigvecs = np.linalg.eigh(M)
    d2 = eigvecs[:, :, -1]
    d3 = frames[:, :, 0] * d2[:, 0:1] + frames[:, :, 1] * d2[:, 1:2]
    n = np.linalg.norm(d3, axis=1, keepdims=True)
    d3 = d3 / (n + 1e-12)
    return d3


def _quality_gate_accepts(base_qm: dict, cand_qm: dict, topo_cfg: dict) -> bool:
    max_mean_angle_inc = float(topo_cfg.get('repair_quality_max_mean_angle_increase_deg', 1.5))
    max_p90_angle_inc = float(topo_cfg.get('repair_quality_max_p90_angle_increase_deg', 3.0))
    max_mean_aspect_inc = float(topo_cfg.get('repair_quality_max_mean_aspect_increase_ratio', 0.15))
    max_p90_aspect_inc = float(topo_cfg.get('repair_quality_max_p90_aspect_increase_ratio', 0.2))
    max_mean_planarity_inc = float(topo_cfg.get('repair_quality_max_mean_planarity_increase_ratio', 0.2))
    max_p90_planarity_inc = float(topo_cfg.get('repair_quality_max_p90_planarity_increase_ratio', 0.25))
    if cand_qm['mean_angle_dev'] > base_qm['mean_angle_dev'] + max_mean_angle_inc:
        return False
    if cand_qm['p90_angle_dev'] > base_qm['p90_angle_dev'] + max_p90_angle_inc:
        return False
    if cand_qm['mean_aspect_ratio'] > base_qm['mean_aspect_ratio'] * (1.0 + max_mean_aspect_inc):
        return False
    if cand_qm['p90_aspect_ratio'] > base_qm['p90_aspect_ratio'] * (1.0 + max_p90_aspect_inc):
        return False
    if cand_qm['mean_planarity'] > base_qm['mean_planarity'] * (1.0 + max_mean_planarity_inc) + 1e-12:
        return False
    if cand_qm['p90_planarity'] > base_qm['p90_planarity'] * (1.0 + max_p90_planarity_inc) + 1e-12:
        return False
    return True


def _selection_quality_gate_accepts(base_qm: dict, cand_qm: dict, topo_cfg: dict) -> bool:
    if not cand_qm:
        return True
    if float(cand_qm.get('mean_angle_dev', 0.0)) > float(
        topo_cfg.get('selection_quality_max_mean_angle_dev', 12.0)
    ):
        return False
    if float(cand_qm.get('mean_aspect_ratio', 0.0)) > float(
        topo_cfg.get('selection_quality_max_mean_aspect', 1.9)
    ):
        return False
    if float(cand_qm.get('mean_planarity', 0.0)) > float(
        topo_cfg.get('selection_quality_max_mean_planarity', 0.01)
    ):
        return False
    if float(cand_qm.get('p90_angle_dev', 0.0)) > float(
        topo_cfg.get('selection_quality_max_p90_angle_dev', 28.0)
    ):
        return False
    if float(cand_qm.get('p90_aspect_ratio', 0.0)) > float(
        topo_cfg.get('selection_quality_max_p90_aspect', 2.5)
    ):
        return False
    if float(cand_qm.get('p90_planarity', 0.0)) > float(
        topo_cfg.get('selection_quality_max_p90_planarity', 0.10)
    ):
        return False
    return True


def _post_pd_alt_promotion_accepts(
    main_rep: dict,
    main_qm: dict,
    main_faces: int,
    alt_rep: dict,
    alt_qm: dict,
    alt_faces: int,
    topo_cfg: dict,
) -> bool:
    if int(alt_rep.get('boundary_loops', 0)) > int(main_rep.get('boundary_loops', 0)):
        return False
    if int(alt_rep.get('boundary_chains', 0)) > int(main_rep.get('boundary_chains', 0)):
        return False
    if int(alt_rep.get('nonmanifold_edges', 0)) > int(main_rep.get('nonmanifold_edges', 0)):
        return False
    if int(alt_rep.get('duplicate_faces', 0)) > int(main_rep.get('duplicate_faces', 0)):
        return False

    boundary_gain = int(main_rep.get('boundary_edges', 0)) - int(alt_rep.get('boundary_edges', 0))
    if boundary_gain < int(topo_cfg.get('quality_post_pd_alt_promotion_min_boundary_gain', 24)):
        return False

    min_face_ratio = float(topo_cfg.get('quality_post_pd_alt_promotion_min_face_ratio', 0.72))
    if int(alt_faces) < max(1, int(round(int(main_faces) * min_face_ratio))):
        return False

    if float(alt_qm.get('mean_angle_dev', 0.0)) > float(main_qm.get('mean_angle_dev', 0.0)) + float(
        topo_cfg.get('quality_post_pd_alt_max_mean_angle_increase_deg', 1.0)
    ):
        return False
    if float(alt_qm.get('p90_angle_dev', 0.0)) > float(main_qm.get('p90_angle_dev', 0.0)) + float(
        topo_cfg.get('quality_post_pd_alt_max_p90_angle_increase_deg', 2.0)
    ):
        return False
    if float(alt_qm.get('mean_aspect_ratio', 0.0)) > float(main_qm.get('mean_aspect_ratio', 0.0)) * (
        1.0 + float(topo_cfg.get('quality_post_pd_alt_max_mean_aspect_increase_ratio', 0.15))
    ):
        return False
    if float(alt_qm.get('p90_aspect_ratio', 0.0)) > float(main_qm.get('p90_aspect_ratio', 0.0)) * (
        1.0 + float(topo_cfg.get('quality_post_pd_alt_max_p90_aspect_increase_ratio', 0.18))
    ):
        return False
    if float(alt_qm.get('mean_planarity', 0.0)) > float(main_qm.get('mean_planarity', 0.0)) * (
        1.0 + float(topo_cfg.get('quality_post_pd_alt_max_mean_planarity_increase_ratio', 0.20))
    ) + 1e-12:
        return False
    if float(alt_qm.get('p90_planarity', 0.0)) > float(main_qm.get('p90_planarity', 0.0)) * (
        1.0 + float(topo_cfg.get('quality_post_pd_alt_max_p90_planarity_increase_ratio', 0.20))
    ) + 1e-12:
        return False
    return True


def _quad_boundary_vertices(faces: np.ndarray) -> np.ndarray:
    Q = np.asarray(faces, dtype=np.int64)
    if Q.ndim != 2 or Q.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    vmax = int(Q.max()) + 1
    out = np.zeros((vmax,), dtype=bool)
    edge_counts: dict[tuple[int, int], int] = {}
    for q in Q:
        for i in range(4):
            a = int(q[i]); b = int(q[(i + 1) % 4])
            key = (a, b) if a < b else (b, a)
            edge_counts[key] = edge_counts.get(key, 0) + 1
    for (a, b), c in edge_counts.items():
        if c == 1:
            out[a] = True
            out[b] = True
    return out


def _planarize_quads(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    iters: int = 1,
    alpha: float = 0.25,
    boundary_alpha_scale: float = 0.35,
) -> np.ndarray:
    V = np.asarray(vertices, dtype=np.float64).copy()
    Q = np.asarray(faces, dtype=np.int64)
    if Q.ndim != 2 or Q.shape[0] == 0:
        return V

    is_boundary = _quad_boundary_vertices(Q)
    if len(is_boundary) < len(V):
        tmp = np.zeros((len(V),), dtype=bool)
        tmp[:len(is_boundary)] = is_boundary
        is_boundary = tmp

    for _ in range(max(0, int(iters))):
        accum = np.zeros_like(V)
        weight = np.zeros((len(V), 1), dtype=np.float64)
        for q in Q:
            pts = V[q]
            c = pts.mean(axis=0)
            X = pts - c[None, :]
            try:
                _, _, vh = np.linalg.svd(X, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            n = vh[-1]
            n_norm = float(np.linalg.norm(n))
            if n_norm <= 1e-12:
                continue
            n = n / n_norm
            proj = pts - np.dot(pts - c[None, :], n)[:, None] * n[None, :]
            for li, vi in enumerate(q):
                accum[vi] += proj[li]
                weight[vi, 0] += 1.0
        valid = weight[:, 0] > 0.0
        target = V.copy()
        target[valid] = accum[valid] / weight[valid]
        step = np.full((len(V), 1), float(alpha), dtype=np.float64)
        step[is_boundary] *= float(boundary_alpha_scale)
        V = V + step * (target - V)
    return V


def _polish_quad_shape(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    aspect_threshold: float = 3.5,
    alpha: float = 0.12,
    iters: int = 1,
    ring_expand: int = 1,
) -> np.ndarray:
    V = np.asarray(vertices, dtype=np.float64).copy()
    Q = np.asarray(faces, dtype=np.int64)
    if Q.ndim != 2 or Q.shape[0] == 0:
        return V

    is_boundary = _quad_boundary_vertices(Q)
    if len(is_boundary) < len(V):
        tmp = np.zeros((len(V),), dtype=bool)
        tmp[:len(is_boundary)] = is_boundary
        is_boundary = tmp

    # Face adjacency through shared vertices for a conservative local region grow.
    vert_to_faces: list[list[int]] = [[] for _ in range(len(V))]
    for fi, q in enumerate(Q):
        for vi in q:
            vert_to_faces[int(vi)].append(fi)

    bad_faces: set[int] = set()
    for fi, q in enumerate(Q):
        pts = V[q]
        e = np.roll(pts, -1, axis=0) - pts
        lens = np.linalg.norm(e, axis=1) + 1e-12
        aspect = float(lens.max() / lens.min())
        if aspect >= float(aspect_threshold):
            bad_faces.add(fi)
    if not bad_faces:
        return V

    grown_faces = set(bad_faces)
    frontier = set(bad_faces)
    for _ in range(max(0, int(ring_expand))):
        nxt = set()
        for fi in frontier:
            for vi in Q[fi]:
                nxt.update(vert_to_faces[int(vi)])
        nxt.difference_update(grown_faces)
        if not nxt:
            break
        grown_faces.update(nxt)
        frontier = nxt

    target_vertices = np.zeros((len(V),), dtype=bool)
    for fi in grown_faces:
        target_vertices[Q[fi]] = True
    target_vertices &= ~is_boundary
    if not np.any(target_vertices):
        return V

    adjacency: list[set[int]] = [set() for _ in range(len(V))]
    for q in Q:
        for i in range(4):
            a = int(q[i]); b = int(q[(i + 1) % 4])
            adjacency[a].add(b)
            adjacency[b].add(a)

    for _ in range(max(0, int(iters))):
        Vnext = V.copy()
        for vi in np.where(target_vertices)[0]:
            nbrs = sorted(adjacency[vi])
            if not nbrs:
                continue
            avg = V[np.asarray(nbrs, dtype=np.int64)].mean(axis=0)
            Vnext[vi] = (1.0 - float(alpha)) * V[vi] + float(alpha) * avg
        V = Vnext
    return V


def _accept_stack_prune(before_rep: dict, after_rep: dict, prune_stats: dict, topo_cfg: dict) -> bool:
    pair_gain = int(prune_stats.get('initial_pairs', 0)) - int(prune_stats.get('remaining_pairs', 0))
    min_pair_gain = int(topo_cfg.get('stack_prune_min_pair_gain', 20))
    if pair_gain < min_pair_gain:
        return False

    before_boundary = int(before_rep.get('boundary_edges', 0))
    after_boundary = int(after_rep.get('boundary_edges', 0))
    max_boundary_increase = int(topo_cfg.get('stack_prune_max_boundary_increase', 24))
    if after_boundary > before_boundary + max_boundary_increase:
        return False

    before_loops = int(before_rep.get('boundary_loops', 0))
    after_loops = int(after_rep.get('boundary_loops', 0))
    max_loop_increase = int(topo_cfg.get('stack_prune_max_loop_increase', 1))
    if after_loops > before_loops + max_loop_increase:
        return False

    before_chains = int(before_rep.get('boundary_chains', 0))
    after_chains = int(after_rep.get('boundary_chains', 0))
    max_chain_increase = int(topo_cfg.get('stack_prune_max_chain_increase', 0))
    if after_chains > before_chains + max_chain_increase:
        return False

    before_irregular = int(before_rep.get('boundary_irregular_vertices', 0))
    after_irregular = int(after_rep.get('boundary_irregular_vertices', 0))
    max_irregular_increase = int(topo_cfg.get('stack_prune_max_irregular_increase', 0))
    if after_irregular > before_irregular + max_irregular_increase:
        return False

    return True


def _candidate_miq_burden(cand: tuple) -> tuple[float, int, int]:
    qdiag = cand[6] if len(cand) > 6 else {}
    rep = cand[4] if len(cand) > 4 else {}
    faces = max(1, int(rep.get('num_faces', cand[1] if len(cand) > 1 else 1)))
    miq = qdiag.get('miq_extraction') or {}
    winding = miq.get('quad_winding') or {}
    rounding = miq.get('miq_rounding') or {}
    corrected = int(winding.get('corrected_quads', 0))
    final_flips = int(rounding.get('flip_count_final', 0))
    return (corrected / float(faces), corrected, final_flips)


def _small_sharp_field_health(cand: tuple) -> tuple:
    qdiag = cand[6] if len(cand) > 6 else {}
    ph = qdiag.get('poincare_hopf') or {}
    holo = qdiag.get('holonomy') or {}
    flow = qdiag.get('flow_conservation') or {}
    sing = qdiag.get('crossfield_singularities') or {}
    chi = int(ph.get('chi', 0))
    expected_units = 4 * chi
    sum_units = int(sing.get('sum_vertex_units', 0))
    unit_deficit = abs(sum_units - expected_units)
    ph_deficit = abs(float(ph.get('deficit', 0.0)))
    high_order = int(holo.get('high_order_count', 0))
    cotree = int(holo.get('cotree_violations', 0))
    flow_bad = int(flow.get('num_violations', 0))
    ph_bad = 0 if bool(ph.get('satisfied', True)) else 1
    return (
        unit_deficit,
        ph_bad,
        ph_deficit,
        high_order,
        cotree,
        flow_bad,
    )


def _small_sharp_primary_health(cand: tuple) -> tuple:
    unit_deficit, ph_bad, ph_deficit, high_order, cotree, flow_bad = _small_sharp_field_health(cand)
    return (
        unit_deficit,
        ph_bad,
        ph_deficit,
    )


def _dense_quality_sort_key(cand: tuple) -> tuple:
    rep = cand[4]
    meta = cand[8] if len(cand) > 8 else {}
    loops = int(rep.get('boundary_loops', 0))
    boundary = int(rep.get('boundary_edges', 0))
    winding_ratio, corrected, final_flips = _candidate_miq_burden(cand)
    mean_planarity = float(meta.get('mean_planarity', 0.0))
    p90_planarity = float(meta.get('p90_planarity', 0.0))
    mean_aspect = float(meta.get('mean_aspect_ratio', 0.0))
    stack_pairs = int(meta.get('stack_pairs', 0))
    stack_ratio = float(meta.get('stack_pairs_ratio', 0.0))
    return (
        loops,
        boundary,
        stack_ratio,
        stack_pairs,
        p90_planarity,
        mean_planarity,
        final_flips,
        winding_ratio,
        mean_aspect,
        corrected,
        cand[0],
        -int(cand[1]),
    )


def _zipper_profiles(topo_cfg: dict, *, closed_input: bool, dense_faces: int) -> list[dict]:
    profiles = [{
        'min_loop_len': int(topo_cfg.get('zipper_self_boundary_loops_min_loop_len', 64)),
        'min_gap_ratio': float(topo_cfg.get('zipper_self_boundary_loops_min_gap_ratio', 0.1)),
        'max_pair_dist_ratio': float(topo_cfg.get('zipper_self_boundary_loops_max_pair_dist_ratio', 0.06)),
        'iters': int(topo_cfg.get('zipper_self_boundary_loops_iters', 2)),
        'label': 'default',
    }]
    if closed_input and dense_faces >= int(topo_cfg.get('repair_quality_branch_min_faces', 1500)):
        for i, prof in enumerate(topo_cfg.get('zipper_self_closed_profiles', []), start=1):
            profiles.append({
                'min_loop_len': int(prof.get('min_loop_len', profiles[0]['min_loop_len'])),
                'min_gap_ratio': float(prof.get('min_gap_ratio', profiles[0]['min_gap_ratio'])),
                'max_pair_dist_ratio': float(prof.get('max_pair_dist_ratio', profiles[0]['max_pair_dist_ratio'])),
                'iters': int(prof.get('iters', profiles[0]['iters'])),
                'label': str(prof.get('label', f'closed-{i}')),
            })
    return profiles


def _post_orient_zipper_profile(topo_cfg: dict) -> dict:
    return {
        'min_loop_len': int(topo_cfg.get('post_orient_zipper_min_loop_len', 48)),
        'min_gap_ratio': float(topo_cfg.get('post_orient_zipper_min_gap_ratio', 0.08)),
        'max_pair_dist_ratio': float(topo_cfg.get('post_orient_zipper_max_pair_dist_ratio', 0.10)),
        'iters': int(topo_cfg.get('post_orient_zipper_iters', 1)),
    }


def _select_candidate(cands: list[tuple], topo_cfg: dict):
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]

    max_faces = max(int(c[1]) for c in cands)
    dense_face_ratio = float(topo_cfg.get('selection_dense_face_ratio', 0.8))
    topo_override_face_ratio = float(topo_cfg.get('selection_topology_override_face_ratio', 0.55))
    topo_override_min_boundary_gain = int(topo_cfg.get('selection_topology_override_min_boundary_gain', 80))
    topo_override_min_boundary_ratio = float(topo_cfg.get('selection_topology_override_min_boundary_ratio', 0.2))

    dense_min_faces = max(1, int(round(max_faces * dense_face_ratio)))
    topo_override_min_faces = max(1, int(round(max_faces * topo_override_face_ratio)))

    dense_cands = [c for c in cands if int(c[1]) >= dense_min_faces]
    if not dense_cands:
        dense_cands = list(cands)

    best_dense = min(dense_cands, key=lambda c: c[0])
    best_topo = _select_topology_candidate(cands, topo_cfg)

    rep_dense = best_dense[4]
    rep_topo = best_topo[4]
    dense_boundary = int(rep_dense.get('boundary_edges', 0))
    topo_boundary = int(rep_topo.get('boundary_edges', 0))
    boundary_gain = dense_boundary - topo_boundary
    boundary_gain_ratio = (float(boundary_gain) / max(1.0, float(dense_boundary))) if dense_boundary > 0 else 0.0

    if (
        best_topo is not best_dense
        and int(best_topo[1]) >= topo_override_min_faces
        and boundary_gain >= topo_override_min_boundary_gain
        and boundary_gain_ratio >= topo_override_min_boundary_ratio
    ):
        return best_topo
    return best_dense


def _selection_cfg(topo_cfg: dict, *, closed_input: bool) -> dict:
    cfg = dict(topo_cfg)
    if not closed_input:
        return cfg
    override_keys = (
        'selection_dense_face_ratio',
        'selection_quality_max_winding_ratio',
        'selection_quality_max_final_miq_flips',
        'selection_quality_boundary_override_face_ratio',
        'selection_quality_boundary_override_gain',
        'selection_topology_override_face_ratio',
        'selection_topology_override_min_boundary_gain',
        'selection_topology_override_min_boundary_ratio',
        'selection_topology_quality_face_ratio',
        'selection_topology_max_cap_ratio',
        'selection_topology_max_repair_ratio',
    )
    for key in override_keys:
        closed_key = key.replace('selection_', 'selection_closed_', 1)
        if closed_key in topo_cfg:
            cfg[key] = topo_cfg[closed_key]
    return cfg


def _select_dense_candidate(cands: list[tuple], topo_cfg: dict):
    if not cands:
        return None
    max_faces = max(int(c[1]) for c in cands)
    dense_face_ratio = float(topo_cfg.get('selection_dense_face_ratio', 0.8))
    dense_min_faces = max(1, int(round(max_faces * dense_face_ratio)))
    dense_cands = [c for c in cands if int(c[1]) >= dense_min_faces]
    if not dense_cands:
        dense_cands = list(cands)
    max_winding_ratio = float(topo_cfg.get('selection_quality_max_winding_ratio', 0.03))
    max_final_flips = int(topo_cfg.get('selection_quality_max_final_miq_flips', 5))
    gated = []
    for cand in dense_cands:
        winding_ratio, _, final_flips = _candidate_miq_burden(cand)
        if winding_ratio > max_winding_ratio:
            continue
        if final_flips > max_final_flips:
            continue
        gated.append(cand)
    if gated:
        dense_cands = gated
    best_dense = min(dense_cands, key=_dense_quality_sort_key)
    dense_meta = best_dense[8] if len(best_dense) > 8 else {}

    boundary_override_face_ratio = float(
        topo_cfg.get('selection_quality_boundary_override_face_ratio', 0.0)
    )
    boundary_override_gain = int(
        topo_cfg.get('selection_quality_boundary_override_gain', 0)
    )
    if boundary_override_face_ratio > 0.0 and boundary_override_gain > 0:
        dense_rep = best_dense[4]
        dense_loops = int(dense_rep.get('boundary_loops', 0))
        dense_boundary = int(dense_rep.get('boundary_edges', 0))
        override_min_faces = max(1, int(round(max_faces * boundary_override_face_ratio)))
        boundary_override = []
        for cand in cands:
            faces = int(cand[1])
            if faces < override_min_faces:
                continue
            rep = cand[4]
            if int(rep.get('boundary_loops', 0)) > dense_loops:
                continue
            cand_boundary = int(rep.get('boundary_edges', 0))
            if cand_boundary > dense_boundary - boundary_override_gain:
                continue
            winding_ratio, _, final_flips = _candidate_miq_burden(cand)
            if winding_ratio > max_winding_ratio:
                continue
            if final_flips > max_final_flips:
                continue
            boundary_override.append(cand)
        if boundary_override:
            return min(
                boundary_override,
                key=lambda c: (
                    int(c[4].get('boundary_loops', 0)),
                    int(c[4].get('boundary_edges', 0)),
                    float((c[8] if len(c) > 8 else {}).get('stack_pairs_ratio', 0.0)),
                    int((c[8] if len(c) > 8 else {}).get('stack_pairs', 0)),
                    float((c[8] if len(c) > 8 else {}).get('p90_planarity', 0.0)),
                    float((c[8] if len(c) > 8 else {}).get('mean_planarity', 0.0)),
                    -int(c[1]),
                    *_candidate_miq_burden(c),
                    c[0],
                ),
            )
    return best_dense


def _topology_candidate_sort_key(cand: tuple):
    rep = cand[4] if len(cand) > 4 else {}
    diag = cand[6] if len(cand) > 6 else None
    meta = cand[8] if len(cand) > 8 else {}
    repair_total = int(meta.get('repair_total', 0))
    cap_total = int(meta.get('cap_total', 0))
    faces = max(1, int(cand[1]))
    repair_ratio = repair_total / float(faces)
    cap_ratio = cap_total / float(faces)
    field_pen = _field_penalty(diag)
    return (
        int(rep.get('high_multiplicity_edges', 0)),
        int(rep.get('nonmanifold_edges', 0)),
        int(rep.get('degenerate_faces', 0)),
        int(rep.get('duplicate_faces', 0)),
        int(rep.get('non_quad_faces', 0)),
        int(rep.get('boundary_chains', 0)),
        int(rep.get('boundary_irregular_vertices', 0)),
        int(rep.get('boundary_loops', 0)),
        int(rep.get('boundary_edges', 0)),
        field_pen,
        cap_ratio,
        repair_ratio,
        -faces,
    )


def _select_topology_candidate(cands: list[tuple], topo_cfg: dict):
    if not cands:
        return None
    best_dense = _select_dense_candidate(cands, topo_cfg)
    dense_faces = int(best_dense[1]) if best_dense is not None else max(int(c[1]) for c in cands)
    min_face_ratio = float(topo_cfg.get('selection_topology_quality_face_ratio', 0.4))
    max_cap_ratio = float(topo_cfg.get('selection_topology_max_cap_ratio', 0.2))
    max_repair_ratio = float(topo_cfg.get('selection_topology_max_repair_ratio', 0.3))
    min_faces = max(1, int(round(dense_faces * min_face_ratio)))

    filtered = []
    for cand in cands:
        meta = cand[8] if len(cand) > 8 else {}
        faces = max(1, int(cand[1]))
        repair_total = int(meta.get('repair_total', 0))
        cap_total = int(meta.get('cap_total', 0))
        if faces < min_faces:
            continue
        if (cap_total / float(faces)) > max_cap_ratio:
            continue
        if (repair_total / float(faces)) > max_repair_ratio:
            continue
        filtered.append(cand)

    if not filtered:
        filtered = list(cands)
    return min(filtered, key=_topology_candidate_sort_key)


def _resolve_selection_mode(cli_mode: str | None, topo_cfg: dict) -> str:
    mode = cli_mode or str(topo_cfg.get('selection_mode', 'balanced'))
    if mode not in {'quality', 'balanced', 'topology'}:
        return 'balanced'
    return mode


def _mesh_mean_edge_length(V: np.ndarray, F: np.ndarray) -> float:
    tri = np.asarray(F, dtype=np.int64)
    if tri.ndim != 2 or tri.shape[1] != 3 or len(tri) == 0:
        return 0.0
    edges = np.concatenate(
        [tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]],
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    if len(edges) == 0:
        return 0.0
    return float(np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1).mean())


def _subdivide_tri_mesh_for_field(
    V: np.ndarray,
    F: np.ndarray,
    *,
    metric_field: Optional[np.ndarray] = None,
    confidence: Optional[np.ndarray] = None,
    iters: int = 1,
    max_faces: int = 4096,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], int]:
    V_cur = np.asarray(V, dtype=np.float64)
    F_cur = np.asarray(F, dtype=np.int64)
    M_cur = None if metric_field is None else np.asarray(metric_field, dtype=np.float64)
    C_cur = None if confidence is None else np.asarray(confidence, dtype=np.float64).reshape(-1)
    applied = 0

    for _ in range(max(0, int(iters))):
        if F_cur.ndim != 2 or F_cur.shape[1] != 3 or len(F_cur) == 0:
            break
        if len(F_cur) * 4 > int(max_faces):
            break

        edge_mid: dict[tuple[int, int], int] = {}
        verts = [v.copy() for v in V_cur]
        mets = None if M_cur is None else [m.copy() for m in M_cur]
        confs = None if C_cur is None else [float(c) for c in C_cur]

        def midpoint(a: int, b: int) -> int:
            key = (a, b) if a < b else (b, a)
            if key in edge_mid:
                return edge_mid[key]
            idx = len(verts)
            verts.append(0.5 * (V_cur[a] + V_cur[b]))
            if mets is not None:
                mets.append(0.5 * (M_cur[a] + M_cur[b]))
            if confs is not None:
                confs.append(0.5 * (float(C_cur[a]) + float(C_cur[b])))
            edge_mid[key] = idx
            return idx

        faces_new: list[list[int]] = []
        for tri in F_cur:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            faces_new.extend([
                [a, ab, ca],
                [ab, b, bc],
                [ca, bc, c],
                [ab, bc, ca],
            ])

        V_cur = np.asarray(verts, dtype=np.float64)
        F_cur = np.asarray(faces_new, dtype=np.int64)
        if mets is not None:
            M_cur = np.asarray(mets, dtype=np.float64)
        if confs is not None:
            C_cur = np.asarray(confs, dtype=np.float64)
        applied += 1

    return V_cur, F_cur, M_cur, C_cur, applied


def _feature_vertex_mask(points: np.ndarray, feature_lines: list[np.ndarray], radius: float) -> np.ndarray:
    if radius <= 0.0 or points is None or len(points) == 0 or not feature_lines:
        return np.zeros((len(points),), dtype=bool)
    samples = [np.asarray(line, dtype=np.float64) for line in feature_lines if len(line) > 0]
    if not samples:
        return np.zeros((len(points),), dtype=bool)
    feat_pts = np.concatenate(samples, axis=0)
    tree = KDTree(feat_pts)
    dist, _ = tree.query(np.asarray(points, dtype=np.float64), k=1)
    return np.asarray(dist <= float(radius), dtype=bool)


def _polyline_closest_point_and_tangent(point: np.ndarray, line: np.ndarray) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    pts = np.asarray(line, dtype=np.float64)
    if pts.ndim != 2 or len(pts) == 0:
        return np.inf, None, None
    if len(pts) == 1:
        return float(np.linalg.norm(point - pts[0])), pts[0], None
    a = pts[:-1]
    b = pts[1:]
    seg = b - a
    seg_len2 = np.einsum('ij,ij->i', seg, seg)
    valid = seg_len2 > 1e-12
    if not np.any(valid):
        d = np.linalg.norm(point - pts, axis=1)
        idx = int(np.argmin(d))
        return float(d[idx]), pts[idx], None
    ap = point[None, :] - a
    t = np.zeros(len(seg), dtype=np.float64)
    t[valid] = np.clip(np.einsum('ij,ij->i', ap[valid], seg[valid]) / seg_len2[valid], 0.0, 1.0)
    proj = a + t[:, None] * seg
    dist = np.linalg.norm(proj - point[None, :], axis=1)
    idx = int(np.argmin(dist))
    tang = seg[idx]
    t_n = float(np.linalg.norm(tang))
    if t_n <= 1e-12:
        return float(dist[idx]), proj[idx], None
    return float(dist[idx]), proj[idx], tang / t_n


def _densify_feature_lines(feature_lines: list[np.ndarray], spacing: float) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    step = max(float(spacing), 1e-6)
    for line in feature_lines:
        pts = np.asarray(line, dtype=np.float64)
        if pts.ndim != 2 or len(pts) == 0:
            continue
        if len(pts) == 1:
            out.append(pts.copy())
            continue
        dense = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            seg = np.asarray(b - a, dtype=np.float64)
            seg_len = float(np.linalg.norm(seg))
            if seg_len <= 1e-12:
                continue
            n_steps = max(1, int(np.ceil(seg_len / step)))
            for i in range(1, n_steps + 1):
                t = float(i) / float(n_steps)
                dense.append((1.0 - t) * a + t * b)
        out.append(np.asarray(dense, dtype=np.float64))
    return out


def _feature_match_thresholds(config: dict, reference_length: float) -> tuple[float, float]:
    feat_cfg = config.get('features', {})
    v_abs = float(feat_cfg.get('feature_vertex_threshold', 0.05))
    e_abs = float(feat_cfg.get('feature_edge_threshold', v_abs))
    v_rel = float(feat_cfg.get('feature_vertex_threshold_edge_ratio', 0.15))
    e_rel = float(feat_cfg.get('feature_edge_threshold_edge_ratio', 0.35))
    ref = max(float(reference_length), 1e-6)
    return max(v_abs, v_rel * ref), max(e_abs, e_rel * ref)


def _lift_metric_field_to_world(metric_lcf: np.ndarray, basis: np.ndarray) -> np.ndarray:
    M = np.asarray(metric_lcf, dtype=np.float64)
    B = np.asarray(basis, dtype=np.float64)
    R = B[:, :, :2]   # (N, 3, 2)
    return np.einsum('nik,nkl,njl->nij', R, M, R)


def _project_world_metric_to_frames(metric_world: np.ndarray, frames: np.ndarray) -> np.ndarray:
    Mw = np.asarray(metric_world, dtype=np.float64)
    R = np.asarray(frames, dtype=np.float64)[:, :, :2]   # (N, 3, 2)
    M2 = np.einsum('nki,nkl,nlj->nij', R, Mw, R)
    M2 = 0.5 * (M2 + np.transpose(M2, (0, 2, 1)))
    eigvals, eigvecs = np.linalg.eigh(M2)
    eigvals = np.clip(eigvals, 1e-6, None)
    return np.einsum('nik,nk,njk->nij', eigvecs, eigvals, eigvecs)


def _build_feature_constraints(
    quadV: np.ndarray,
    quadF: np.ndarray,
    feature_lines: list[np.ndarray],
    config: dict,
    *,
    reference_length: float,
) -> tuple[list[FeaturePointConstraint], list[FeatureEdgeConstraint]]:
    point_constraints: list[FeaturePointConstraint] = []
    edge_constraints: list[FeatureEdgeConstraint] = []
    if not feature_lines or len(quadV) == 0 or len(quadF) == 0:
        return point_constraints, edge_constraints

    vertex_thresh, edge_thresh = _feature_match_thresholds(config, reference_length)
    valid_lines = [np.asarray(line, dtype=np.float64) for line in feature_lines if len(line) >= 2]
    if not valid_lines:
        return point_constraints, edge_constraints

    feat_cfg = config.get('features', {})
    edge_min_alignment = float(feat_cfg.get('feature_edge_min_alignment', 0.55))
    edge_base_weight = float(feat_cfg.get('feature_edge_weight', 4.0))
    edge_alignment_bonus = float(feat_cfg.get('feature_edge_weight_alignment_bonus', 2.0))
    edge_max_per_quad = int(feat_cfg.get('feature_edge_max_per_quad', 2))

    for vi, vpos in enumerate(np.asarray(quadV, dtype=np.float64)):
        best = None
        for line in valid_lines:
            dist, closest, tang = _polyline_closest_point_and_tangent(vpos, line)
            if tang is None or dist >= vertex_thresh:
                continue
            cand = (float(dist), vi, closest, tang)
            if best is None or cand[0] < best[0]:
                best = cand
        if best is not None:
            _, gi, closest, tang = best
            point_constraints.append(FeaturePointConstraint(int(gi), closest, tang))

    for qi, q in enumerate(np.asarray(quadF, dtype=np.int64)):
        candidates = []
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
            gi = int(q[a]); gj = int(q[b])
            p0 = np.asarray(quadV[gi], dtype=np.float64)
            p1 = np.asarray(quadV[gj], dtype=np.float64)
            mid = 0.5 * (p0 + p1)
            edge = p1 - p0
            e_n = float(np.linalg.norm(edge))
            if e_n <= 1e-12:
                continue
            edge_dir = edge / e_n
            for line in valid_lines:
                dist, closest, tang = _polyline_closest_point_and_tangent(mid, line)
                if tang is None or dist >= edge_thresh:
                    continue
                d0, _, _ = _polyline_closest_point_and_tangent(p0, line)
                d1, _, _ = _polyline_closest_point_and_tangent(p1, line)
                if max(float(d0), float(d1)) >= edge_thresh * 1.2:
                    continue
                align = abs(float(np.dot(edge_dir, tang)))
                if align < edge_min_alignment:
                    continue
                score = (float(dist), -align)
                weight = edge_base_weight * max(0.25, 1.0 - float(dist) / max(edge_thresh, 1e-8))
                weight *= 1.0 + edge_alignment_bonus * max(0.0, align - edge_min_alignment)
                candidates.append((score, gi, gj, closest, tang, weight))
        if candidates:
            candidates.sort(key=lambda item: item[0])
            used_edges = set()
            for _, gi, gj, closest, tang, weight in candidates:
                key = tuple(sorted((int(gi), int(gj))))
                if key in used_edges:
                    continue
                used_edges.add(key)
                if len(used_edges) > edge_max_per_quad:
                    break
                edge_constraints.append(
                    FeatureEdgeConstraint(int(qi), (int(gi), int(gj)), closest, tang, weight=float(weight))
                )

    return point_constraints, edge_constraints


def _align_metric_to_feature_lines(
    metric_field: np.ndarray,
    points: np.ndarray,
    frames: np.ndarray,
    feature_lines: list[np.ndarray],
    *,
    radius: float,
    min_anisotropy_ratio: float = 1.15,
    skip_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, int]:
    if (
        radius <= 0.0
        or metric_field is None
        or points is None
        or frames is None
        or len(points) == 0
        or len(metric_field) != len(points)
        or len(frames) != len(points)
        or not feature_lines
    ):
        return np.asarray(metric_field, dtype=np.float64), 0

    out = np.asarray(metric_field, dtype=np.float64).copy()
    frame_arr = np.asarray(frames, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    if skip_mask is not None:
        skip_mask = np.asarray(skip_mask, dtype=bool).reshape(-1)
        if skip_mask.shape[0] != len(pts):
            skip_mask = None
    changed = 0

    valid_lines = [np.asarray(line, dtype=np.float64) for line in feature_lines if len(line) >= 2]
    if not valid_lines:
        return out, 0

    for vi, p in enumerate(pts):
        if skip_mask is not None and skip_mask[vi]:
            continue
        best = None
        for line in valid_lines:
            dist, _, tang = _polyline_closest_point_and_tangent(p, line)
            if tang is None or dist > radius:
                continue
            cand = (float(dist), tang)
            if best is None or cand[0] < best[0]:
                best = cand
        if best is None:
            continue

        tang = np.asarray(best[1], dtype=np.float64)
        e1 = frame_arr[vi, :, 0]
        e2 = frame_arr[vi, :, 1]
        t2 = np.array([np.dot(tang, e1), np.dot(tang, e2)], dtype=np.float64)
        t2_n = float(np.linalg.norm(t2))
        if t2_n <= 1e-10:
            continue
        t2 /= t2_n
        n2 = np.array([-t2[1], t2[0]], dtype=np.float64)

        evals, _ = np.linalg.eigh(out[vi])
        lam_min = float(max(evals[0], 1e-6))
        lam_max = float(max(evals[1], lam_min * min_anisotropy_ratio))
        R2 = np.stack([t2, n2], axis=1)
        out[vi] = R2 @ np.diag([lam_max, lam_min]) @ R2.T
        changed += 1

    return out, changed


def _distance_mask_to_samples(points: np.ndarray, samples: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0.0 or len(points) == 0 or len(samples) == 0:
        return np.zeros((len(points),), dtype=bool)
    tree = KDTree(np.asarray(samples, dtype=np.float64))
    dist, _ = tree.query(np.asarray(points, dtype=np.float64), k=1)
    return np.asarray(dist <= float(radius), dtype=bool)


def _feature_guidance_override(
    points: np.ndarray,
    frames: np.ndarray,
    feature_lines: list[np.ndarray],
    *,
    radius: float,
    active_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(points)
    override = np.zeros((n,), dtype=np.complex64)
    weight = np.zeros((n,), dtype=np.float64)
    if radius <= 0.0 or n == 0 or len(feature_lines) == 0:
        return override, weight
    if active_mask is not None:
        active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)
        if active_mask.shape[0] != n:
            active_mask = None

    pts = np.asarray(points, dtype=np.float64)
    frame_arr = np.asarray(frames, dtype=np.float64)
    valid_lines = [np.asarray(line, dtype=np.float64) for line in feature_lines if len(line) >= 2]
    if not valid_lines:
        return override, weight

    for vi, p in enumerate(pts):
        if active_mask is not None and not active_mask[vi]:
            continue
        best = None
        for line in valid_lines:
            dist, _, tang = _polyline_closest_point_and_tangent(p, line)
            if tang is None or dist > radius:
                continue
            cand = (float(dist), tang)
            if best is None or cand[0] < best[0]:
                best = cand
        if best is None:
            continue
        tang = np.asarray(best[1], dtype=np.float64)
        e1 = frame_arr[vi, :, 0]
        e2 = frame_arr[vi, :, 1]
        t2 = np.array([np.dot(tang, e1), np.dot(tang, e2)], dtype=np.float64)
        t2_n = float(np.linalg.norm(t2))
        if t2_n <= 1e-10:
            continue
        t2 /= t2_n
        theta = float(np.arctan2(t2[1], t2[0]))
        override[vi] = np.complex64(np.exp(4j * theta))
        weight[vi] = max(0.0, 1.0 - float(best[0]) / max(radius, 1e-12))
    return override, weight


def _build_corner_singularity_prior(
    points: np.ndarray,
    corners: np.ndarray,
    charge_units: int = 1,
    support_radius: float = 0.0,
    max_points_per_corner: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    c = np.asarray(corners, dtype=np.float64)
    N = len(pts)
    mask = np.zeros(N, dtype=bool)
    indices = np.zeros(N, dtype=np.float64)
    if N == 0 or c.size == 0:
        return mask, indices
    radius = max(float(support_radius), 0.0)
    max_pts = max(1, int(max_points_per_corner))
    for corner in np.asarray(c, dtype=np.float64):
        dist = np.linalg.norm(pts - corner[None, :], axis=1)
        order = np.argsort(dist)
        if radius > 0.0:
            selected = [int(i) for i in order if dist[i] <= radius][:max_pts]
        else:
            selected = []
        if not selected:
            selected = [int(order[0])]
        weight = float(charge_units) / float(len(selected))
        for idx in selected:
            mask[idx] = True
            indices[idx] += weight
    return mask, indices


def _is_small_sharp_closed_mesh(
    tri_mesh: tuple[np.ndarray, np.ndarray] | None,
    feature_lines: list[np.ndarray],
    topo_cfg: dict,
    *,
    closed_input: bool,
) -> bool:
    if tri_mesh is None or not closed_input:
        return False
    V, F = tri_mesh
    v_thresh = int(topo_cfg.get('small_sharp_vertex_threshold', 256))
    f_thresh = int(topo_cfg.get('small_sharp_face_threshold', 512))
    line_thresh = int(topo_cfg.get('small_sharp_feature_lines_threshold', 8))
    sample_thresh = int(topo_cfg.get('small_sharp_feature_samples_threshold', 24))
    num_lines = len(feature_lines)
    num_samples = int(sum(len(line) for line in feature_lines))
    return len(V) <= v_thresh and len(F) <= f_thresh and (
        num_lines >= line_thresh or num_samples >= sample_thresh
    )


def _build_profile_sequence(
    base_profile: dict,
    topo_cfg: dict,
    *,
    closed_input: bool,
    selection_mode: str,
    retry_profiles_override: list[dict] | None = None,
) -> list[dict]:
    profiles = [base_profile]
    if retry_profiles_override is not None:
        retry_profiles = retry_profiles_override
    else:
        if closed_input and selection_mode == 'quality':
            retry_key = 'closed_quality_retry_profiles'
        else:
            retry_key = 'closed_retry_profiles' if closed_input else 'retry_profiles'
        retry_profiles = topo_cfg.get(retry_key, []) or []

    for rp in retry_profiles:
        if not isinstance(rp, dict):
            continue
        merged = dict(base_profile)
        merged.update(rp)
        profiles.append(merged)
    return profiles


def _select_small_sharp_quality_candidate(
    cands: list[tuple],
    topo_cfg: dict,
):
    if not cands:
        return None
    min_faces_abs = int(topo_cfg.get('small_sharp_quality_min_faces', 120))
    max_faces = max(int(c[1]) for c in cands)
    min_face_ratio = float(topo_cfg.get('small_sharp_quality_override_face_ratio', 0.25))
    min_faces = max(1, min(min_faces_abs, int(round(max_faces * min_face_ratio))))
    filtered = [c for c in cands if int(c[1]) >= min_faces]
    if not filtered:
        filtered = list(cands)

    def _key(c: tuple):
        rep = c[4]
        winding_ratio, corrected, final_flips = _candidate_miq_burden(c)
        primary_health = _small_sharp_primary_health(c)
        _, _, _, high_order, cotree, flow_bad = _small_sharp_field_health(c)
        return (
            int(rep.get('boundary_loops', 0)),
            int(rep.get('boundary_edges', 0)),
            *primary_health,
            high_order,
            cotree,
            flow_bad,
            final_flips,
            corrected,
            winding_ratio,
            -int(c[1]),
        )

    best_dense = min(filtered, key=_key)
    best_any = min(cands, key=_key)

    alt_min_face_ratio = float(topo_cfg.get('small_sharp_quality_boundary_override_face_ratio', 0.18))
    alt_min_faces_abs = int(topo_cfg.get('small_sharp_quality_boundary_override_min_faces', 100))
    alt_min_faces = max(1, min(alt_min_faces_abs, int(round(max_faces * alt_min_face_ratio))))
    alt_boundary_gain = int(topo_cfg.get('small_sharp_quality_boundary_override_gain', 12))
    best_boundary = min(
        cands,
        key=lambda c: (
            int(c[4].get('boundary_loops', 0)),
            int(c[4].get('boundary_edges', 0)),
            -int(c[1]),
        ),
    )
    if best_any is best_dense:
        return best_dense
    if (
        int(best_boundary[1]) >= alt_min_faces
        and int(best_boundary[4].get('boundary_loops', 0)) <= int(best_dense[4].get('boundary_loops', 0))
        and int(best_boundary[4].get('boundary_edges', 0))
            <= int(best_dense[4].get('boundary_edges', 0)) - alt_boundary_gain
    ):
        return best_boundary
    rep_dense = best_dense[4]
    rep_any = best_any[4]
    if (
        int(best_any[1]) >= alt_min_faces
        and _small_sharp_primary_health(best_any) <= _small_sharp_primary_health(best_dense)
        and int(rep_any.get('boundary_loops', 0)) <= int(rep_dense.get('boundary_loops', 0))
        and int(rep_any.get('boundary_edges', 0)) <= int(rep_dense.get('boundary_edges', 0)) - alt_boundary_gain
    ):
        return best_any
    return best_dense


def _small_sharp_min_quads(topo_cfg: dict) -> int:
    return int(topo_cfg.get('small_sharp_min_quads', topo_cfg.get('min_quads', 1)))


def parse_args():
    p = argparse.ArgumentParser(description='NPAQ: Neural Projective Anisotropic Quadrangulation')
    p.add_argument('--input',      required=True,  help='Input point cloud file')
    p.add_argument('--output',     required=True,  help='Output quad mesh file')
    p.add_argument('--config',     default='configs/reconstruct.yaml')
    p.add_argument('--checkpoint', required=True,  help='Trained model checkpoint (.pth)')
    p.add_argument('--vis',        action='store_true', help='Visualise with Polyscope')
    p.add_argument('--legacy',     action='store_true',
                   help='Load legacy 4-output checkpoint (no singularity head)')
    p.add_argument(
        '--mode',
        choices=['quality', 'balanced', 'topology'],
        default=None,
        help='Output selection mode: dense quality, balanced tradeoff, or topology-first.',
    )
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path, device, legacy=False):
    if legacy:
        print("  [model] Loading legacy 4-output checkpoint …")
        ckpt = torch.load(checkpoint_path, map_location=device)
        cfg  = ckpt.get('config', {}).get('model', {})
        model = load_legacy_checkpoint(
            checkpoint_path, device,
            k=cfg.get('k', 20),
            emb_dims=cfg.get('emb_dims', 256),
            dropout=cfg.get('dropout', 0.5),
        )
    else:
        ckpt  = torch.load(checkpoint_path, map_location=device)
        mcfg  = ckpt.get('config', {}).get('model', {})
        state = ckpt.get('model_state_dict', ckpt)
        # Infer in_dims from actual weight shape — config field may be absent in
        # old checkpoints that were saved before in_dims was added to train.yaml.
        in_dims = infer_in_dims_from_checkpoint(state)
        predict_singularity = infer_predict_singularity_from_checkpoint(state)
        predict_confidence = infer_predict_confidence_from_checkpoint(state)
        model = DGCNN(
            k=mcfg.get('k', 20),
            emb_dims=mcfg.get('emb_dims', 256),
            dropout=mcfg.get('dropout', 0.5),
            in_dims=in_dims,
            predict_singularity=predict_singularity,
            predict_confidence=predict_confidence,
            max_log_half=mcfg.get('max_log_half', 1.5),
        ).to(device)
        # Try strict load; fall back to legacy loader on mismatch
        try:
            model.load_state_dict(ckpt['model_state_dict'], strict=True)
        except RuntimeError:
            print("  [model] Strict load failed — trying legacy key remapping …")
            model = load_legacy_checkpoint(
                checkpoint_path, device,
                k=mcfg.get('k', 20),
                emb_dims=mcfg.get('emb_dims', 256),
                dropout=mcfg.get('dropout', 0.5),
            )
    model.eval()
    return model


def _estimate_normals_open3d(points: np.ndarray, k: int = 30) -> np.ndarray:
    """Estimate per-point normals from point cloud using open3d."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    pcd.orient_normals_consistent_tangent_plane(k=k)
    return np.asarray(pcd.normals).astype(np.float32)


def _filter_nonfinite_pointcloud(points: np.ndarray, normals: np.ndarray | None):
    pts = np.asarray(points, dtype=np.float32)
    mask = np.isfinite(pts).all(axis=1)
    if normals is not None:
        nrms = np.asarray(normals, dtype=np.float32)
        if nrms.shape == pts.shape:
            mask &= np.isfinite(nrms).all(axis=1)
        else:
            nrms = None
    else:
        nrms = None

    bad = int((~mask).sum())
    if bad <= 0:
        return pts, nrms

    pts = pts[mask]
    if nrms is not None:
        nrms = nrms[mask]
    print(f"  [input] dropped {bad} non-finite points before reconstruction.")
    return pts, nrms


def _vertex_normals_from_tri_mesh(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    N = np.zeros_like(V)
    if F.ndim != 2 or F.shape[1] != 3 or len(F) == 0:
        return N.astype(np.float32)
    for tri in F:
        a, b, c = V[tri]
        n = np.cross(b - a, c - a)
        N[tri] += n
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
    return N.astype(np.float32)


def predict_metric_field(model, points, k_neighbors, device, normals=None):
    """
    Run the DGCNN on every point in the cloud and return per-point metrics.

    When the model was trained with in_dims=6 (coords+normals), normals are
    used if provided, otherwise estimated via open3d.

    Args:
        normals: (N, 3) pre-computed normals from the input file, or None.

    Returns:
        metric_field:  (N, 2, 2) in LCF.
        basis:         (N, 3, 3) LCF rotation matrices.
        confidence:    (N,) direction-confidence in [0,1].
    """
    from scipy.spatial import KDTree as SpatialKDTree
    from src.dataset.lcf import compute_local_canonical_frame

    # Detect model input dimensionality from conv1 weight shape
    in_dims = model.encoder.conv1[0].in_channels // 2   # get_graph_feature doubles it

    N = len(points)
    k_neighbors = max(2, min(int(k_neighbors), N))
    tree = SpatialKDTree(points)

    # Use provided normals if available, otherwise estimate
    if in_dims == 6:
        if normals is not None and normals.shape == (N, 3):
            print("  [model] Using normals from input file for 6D input.")
        else:
            print("  [model] Estimating normals for 6D input …")
            normals = _estimate_normals_open3d(points, k=max(k_neighbors, 30))

    all_feats, all_basis = [], []
    for gi in range(N):
        lc, basis, nbr_normals = compute_local_canonical_frame(
            points, gi, k=k_neighbors, normals=normals,
            return_neighbors=False, tree=tree,
        )
        if nbr_normals is not None:
            feat = np.concatenate([lc, nbr_normals], axis=-1)   # (K, 6)
        else:
            feat = lc                                             # (K, 3)
        all_feats.append(feat)
        all_basis.append(basis)

    all_feats = np.stack(all_feats)   # (N, k, in_dims)
    all_basis = np.stack(all_basis)   # (N, 3, 3)

    tensor_feats = torch.from_numpy(all_feats).float().to(device)

    metric_field = np.zeros((N, 2, 2))
    confidence = np.ones((N,), dtype=np.float32)

    batch_pred = 1024
    for i in range(0, N, batch_pred):
        end = min(i + batch_pred, N)
        x   = tensor_feats[i:end].transpose(2, 1)    # (B, in_dims, k)
        with torch.no_grad():
            out = model(x)                             # (B, 7) or (B, 4) legacy
        if out.shape[1] >= 4:
            s1 = out[:, 0]; s2 = out[:, 1]
            c  = out[:, 2]; s  = out[:, 3]
            metric_field[i:end] = params_to_tensor(s1, s2, c, s).cpu().numpy()
        if out.shape[1] in (5, 8):
            confidence[i:end] = out[:, 4].clamp(0.0, 1.0).detach().cpu().numpy()
    return metric_field, all_basis, confidence


def main():
    args   = parse_args()
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── 1. Load input geometry ─────────────────────────────────────────
    print("Loading input geometry …")
    points = None
    input_normals = None   # normals from file (may be None)
    tri_mesh = None        # (V,F) if input is a triangle mesh
    export_ref_points = None
    export_ref_normals = None
    expected_boundary_loops = 0
    try:
        V_in, F_in = read_mesh(args.input)
        V_in = np.asarray(V_in, dtype=np.float32)
        F_in = np.asarray(F_in)
        if F_in.ndim == 2 and F_in.shape[1] == 3 and len(F_in) > 0:
            points = V_in
            tri_mesh = (V_in, F_in)
            export_ref_points = V_in
            export_ref_normals = _vertex_normals_from_tri_mesh(V_in, F_in)
            input_normals = export_ref_normals
            from src.utils.topology_cert import _boundary_loop_lengths
            expected_boundary_loops = len(_boundary_loop_lengths(F_in))
            print(f"  Triangle mesh detected: {len(V_in)} vertices, {len(F_in)} faces.")
        else:
            # No usable faces — treat as point cloud; also try to get normals
            points, input_normals = read_pointcloud_with_normals(args.input)
            points, input_normals = _filter_nonfinite_pointcloud(points, input_normals)
            export_ref_points = points
            export_ref_normals = input_normals
            print(f"  Point cloud loaded: {len(points)} points"
                  f"{' (with normals)' if input_normals is not None else ''}.")
    except Exception:
        points, input_normals = read_pointcloud_with_normals(args.input)
        points, input_normals = _filter_nonfinite_pointcloud(points, input_normals)
        export_ref_points = points
        export_ref_normals = input_normals
        print(f"  Point cloud loaded: {len(points)} points"
              f"{' (with normals)' if input_normals is not None else ''}.")

    # ── 2. Neural prediction ───────────────────────────────────────────
    print("Predicting metric fields …")
    model = load_model(args.checkpoint, device, legacy=args.legacy)
    metric_lcf, basis, confidence = predict_metric_field(
        model, points, config['model']['k'], device, normals=input_normals
    )

    # ── 3. Metric smoothing ────────────────────────────────────────────
    print("Smoothing metric field …")
    smoothed_lcf = smooth_metric_field_implicit(
        metric_lcf, points,
        lambda_smooth=config['smoothing']['lambda'],
        k=config['smoothing']['k'],
    )
    metric_world = _lift_metric_field_to_world(smoothed_lcf, basis)
    metric_for_topology = smoothed_lcf
    if tri_mesh is not None and basis.shape[0] == len(V_in):
        input_frames = compute_vertex_frames(V_in, F_in, normals=export_ref_normals)
        metric_for_topology = _project_world_metric_to_frames(metric_world, input_frames)
        print("  [metric] converted predicted metric from LCF basis to mesh tangent frames.")

    # ── 4. Feature line detection ──────────────────────────────────────
    print("Detecting feature lines …")
    if tri_mesh is not None:
        feature_lines = extract_feature_lines_from_mesh(
            V_in,
            F_in,
            dihedral_threshold_deg=float(config['features'].get('mesh_dihedral_threshold_deg', 25.0)),
            min_polyline_vertices=int(config['features'].get('mesh_min_polyline_vertices', 2)),
            include_boundary=bool(config['features'].get('mesh_include_boundary', True)),
        )
        feature_corners = extract_feature_corners_from_mesh(
            V_in,
            F_in,
            dihedral_threshold_deg=float(config['features'].get('mesh_dihedral_threshold_deg', 25.0)),
            include_boundary=bool(config['features'].get('mesh_include_boundary', True)),
            min_corner_degree=int(config['features'].get('mesh_corner_min_degree', 3)),
        )
    else:
        feature_lines = extract_feature_lines(
            points,
            curvature_threshold=config['features']['curvature_threshold'],
            cluster_eps=config['features']['cluster_eps'],
            min_cluster_size=config['features']['min_cluster_size'],
            smooth_polyline=bool(config['features']['line_smoothing']),
        )
        feature_corners = np.zeros((0, 3), dtype=np.float64)
    print(f"  {len(feature_lines)} feature lines detected.")
    guidance_override = None
    guidance_override_weight = None
    feature_base_length = 0.0
    if len(feature_corners) > 0:
        print(f"  {len(feature_corners)} feature corners detected.")
    if feature_lines:
        if tri_mesh is not None:
            boost_len = _mesh_mean_edge_length(V_in, F_in)
        else:
            bbox = points.max(axis=0) - points.min(axis=0)
            boost_len = float(np.linalg.norm(bbox) / max(np.sqrt(float(len(points))), 1.0))
        feature_base_length = float(boost_len)
        boost_radius = float(config['features'].get('guidance_feature_boost_radius_ratio', 1.5)) * max(boost_len, 1e-6)
        feature_mask = _feature_vertex_mask(points, feature_lines, boost_radius)
        corner_radius = float(config['features'].get('guidance_corner_suppress_radius_ratio', 0.9)) * max(boost_len, 1e-6)
        corner_mask = _distance_mask_to_samples(points, feature_corners, corner_radius)
        edge_only_mask = np.asarray(feature_mask & (~corner_mask), dtype=bool)
        if feature_mask.any():
            min_conf = float(config['features'].get('guidance_feature_min_confidence', 0.85))
            confidence = np.asarray(confidence, dtype=np.float32).copy()
            confidence[edge_only_mask] = np.maximum(confidence[edge_only_mask], min_conf)
            print(
                "  [features] boosted guidance confidence near sharp lines: "
                f"{int(edge_only_mask.sum())} edge-only vertices, radius={boost_radius:.4g}, min_conf={min_conf:.2f}"
            )
        if corner_mask.any():
            corner_conf = float(config['features'].get('guidance_corner_confidence', 0.2))
            confidence = np.asarray(confidence, dtype=np.float32).copy()
            confidence[corner_mask] = np.minimum(confidence[corner_mask], corner_conf)
            print(
                "  [features] suppressed guidance near sharp corners: "
                f"{int(corner_mask.sum())} vertices, radius={corner_radius:.4g}, max_conf={corner_conf:.2f}"
            )
        if tri_mesh is not None:
            metric_align_radius = float(config['features'].get('metric_feature_align_radius_ratio', 1.5)) * max(boost_len, 1e-6)
            if corner_mask.any():
                safe_metric_points = np.asarray(V_in, dtype=np.float64)[~corner_mask[:len(V_in)]]
            else:
                safe_metric_points = np.asarray(V_in, dtype=np.float64)
            metric_for_topology, n_aligned_metric = _align_metric_to_feature_lines(
                metric_for_topology,
                V_in,
                input_frames,
                feature_lines,
                radius=metric_align_radius,
                min_anisotropy_ratio=float(config['features'].get('metric_feature_min_anisotropy_ratio', 1.2)),
                skip_mask=corner_mask[:len(V_in)] if corner_mask.shape[0] >= len(V_in) else None,
            )
            if n_aligned_metric > 0:
                print(
                    "  [metric] aligned principal direction to mesh feature lines: "
                    f"{n_aligned_metric} vertices, radius={metric_align_radius:.4g}"
                )
        if tri_mesh is not None and edge_only_mask.any():
            override_radius = float(config['features'].get('guidance_feature_override_radius_ratio', 1.2)) * max(boost_len, 1e-6)
            guidance_override, guidance_override_weight = _feature_guidance_override(
                V_in,
                input_frames,
                feature_lines,
                radius=override_radius,
                active_mask=edge_only_mask[:len(V_in)] if edge_only_mask.shape[0] >= len(V_in) else None,
            )
            if guidance_override_weight is not None and np.any(guidance_override_weight > 0):
                print(
                    "  [features] built explicit field anchors on sharp edges: "
                    f"{int((guidance_override_weight > 0).sum())} vertices, radius={override_radius:.4g}"
                )

    # ── 5. Cross-field + parametrisation + initial quad mesh ──────────
    print("Generating field-aligned quad topology …")
    miq_cfg = config['miq']
    topo_cfg = config.get('topology', {})
    topo_strict = bool(topo_cfg.get('strict', True))
    topo_allow_boundary = bool(topo_cfg.get('allow_boundary', True))
    selection_mode = _resolve_selection_mode(args.mode, topo_cfg)

    base_profile = {
        'poisson_depth': int(miq_cfg['poisson_depth']),
        'gradient_size': float(miq_cfg['gradient_size']),
        'crossfield_mu': float(miq_cfg.get('crossfield_mu', 10.0)),
        'umbilic_smoothing': bool(miq_cfg.get('umbilic_smoothing', True)),
        'anisotropy_eps': float(miq_cfg.get('anisotropy_eps', 0.03)),
        'mu_min_ratio': float(miq_cfg.get('mu_min_ratio', 0.05)),
        'integer_constraints': bool(miq_cfg.get('integer_constraints', True)),
        'require_closed_topology': bool(miq_cfg.get('require_closed_topology', (not topo_allow_boundary))),
        'anisotropy_schedule': tuple(miq_cfg.get('anisotropy_schedule', [1.0, 0.7, 0.45, 0.25, 0.0])),
        'integer_projection_iters': int(miq_cfg.get('integer_projection_iters', 2)),
        'integer_potential_iters': int(miq_cfg.get('integer_potential_iters', 2)),
        'max_param_dist': float(miq_cfg.get('max_param_dist', 0.7)),
        'use_igl_backend': bool(miq_cfg.get('use_igl_backend', False)),
        'igl_binary_path': miq_cfg.get('igl_binary_path'),
        'igl_gradient_size': float(miq_cfg.get('igl_gradient_size', 20.0)),
        'igl_stiffness': float(miq_cfg.get('igl_stiffness', 5.0)),
        'igl_direct_round': bool(miq_cfg.get('igl_direct_round', True)),
        'igl_miq_iter': int(miq_cfg.get('igl_miq_iter', 5)),
    }
    closed_input = expected_boundary_loops == 0
    profile_overrides = {}
    retry_profiles_override = None
    small_sharp_closed = _is_small_sharp_closed_mesh(
        tri_mesh, feature_lines, topo_cfg, closed_input=closed_input
    )
    topo_min_quads = int(topo_cfg.get('min_quads', 1))
    if small_sharp_closed:
        topo_min_quads = _small_sharp_min_quads(topo_cfg)
    solve_tri_mesh = tri_mesh
    solve_metric_for_topology = metric_for_topology
    solve_confidence = confidence
    solve_guidance_override = guidance_override
    solve_guidance_override_weight = guidance_override_weight
    solve_singularity_mask = None
    solve_singularity_indices = None
    solve_protected_mask = None
    if small_sharp_closed and tri_mesh is not None:
        sub_iters = int(topo_cfg.get('small_sharp_field_subdivide_iters', 1))
        sub_max_faces = int(topo_cfg.get('small_sharp_field_subdivide_max_faces', 4096))
        sv, sf, sm, sc, applied_subdiv = _subdivide_tri_mesh_for_field(
            V_in,
            F_in,
            metric_field=metric_for_topology,
            confidence=confidence,
            iters=sub_iters,
            max_faces=sub_max_faces,
        )
        if applied_subdiv > 0 and sm is not None:
            solve_tri_mesh = (sv.astype(np.float32), sf.astype(np.int64))
            solve_metric_for_topology = sm
            solve_confidence = confidence if sc is None else sc.astype(np.float32)
            solve_guidance_override = None
            solve_guidance_override_weight = None
            print(
                "  [sharp] refined field-solve mesh for sharp CAD: "
                f"iters={applied_subdiv}, V={len(sv)}, F={len(sf)}"
            )
            if feature_lines:
                solve_frames = compute_vertex_frames(sv, sf, normals=_vertex_normals_from_tri_mesh(sv, sf))
                refined_len = max(_mesh_mean_edge_length(sv, sf), 1e-6)
                refined_feature_lines = _densify_feature_lines(
                    feature_lines,
                    spacing=max(
                        refined_len * float(config['features'].get('guidance_feature_refined_sample_spacing_ratio', 0.9)),
                        1e-6,
                    ),
                )
                boost_radius = float(
                    config['features'].get(
                        'guidance_feature_refined_boost_radius_ratio',
                        config['features'].get('guidance_feature_boost_radius_ratio', 1.5) * 0.5,
                    )
                ) * refined_len
                corner_radius = float(
                    config['features'].get(
                        'guidance_corner_refined_suppress_radius_ratio',
                        config['features'].get('guidance_corner_suppress_radius_ratio', 0.9) * 0.4,
                    )
                ) * refined_len
                feature_mask = _feature_vertex_mask(sv, refined_feature_lines, boost_radius)
                corner_mask = _distance_mask_to_samples(sv, feature_corners, corner_radius)
                edge_only_mask = np.asarray(feature_mask & (~corner_mask), dtype=bool)
                solve_protected_mask = np.asarray(feature_mask | corner_mask, dtype=bool)
                if feature_mask.any():
                    min_conf = float(config['features'].get('guidance_feature_min_confidence', 0.85))
                    solve_confidence = np.asarray(solve_confidence, dtype=np.float32).copy()
                    solve_confidence[edge_only_mask] = np.maximum(solve_confidence[edge_only_mask], min_conf)
                    print(
                        "  [sharp] boosted refined edge guidance: "
                        f"{int(edge_only_mask.sum())} edge-only vertices, radius={boost_radius:.4g}"
                    )
                if corner_mask.any():
                    corner_conf = float(config['features'].get('guidance_corner_confidence', 0.2))
                    solve_confidence = np.asarray(solve_confidence, dtype=np.float32).copy()
                    solve_confidence[corner_mask] = np.minimum(solve_confidence[corner_mask], corner_conf)
                    print(
                        "  [sharp] suppressed refined corner guidance: "
                        f"{int(corner_mask.sum())} vertices, radius={corner_radius:.4g}"
                    )
                metric_align_radius = float(
                    config['features'].get(
                        'metric_feature_refined_align_radius_ratio',
                        config['features'].get('metric_feature_align_radius_ratio', 1.5) * 0.6,
                    )
                ) * refined_len
                solve_metric_for_topology, n_align_refined = _align_metric_to_feature_lines(
                    solve_metric_for_topology,
                    sv,
                    solve_frames,
                    refined_feature_lines,
                    radius=metric_align_radius,
                    min_anisotropy_ratio=float(config['features'].get('metric_feature_min_anisotropy_ratio', 1.2)),
                    skip_mask=corner_mask,
                )
                if n_align_refined > 0:
                    print(
                        "  [sharp] aligned refined metric to sharp lines: "
                        f"{n_align_refined} vertices, radius={metric_align_radius:.4g}"
                    )
                if edge_only_mask.any():
                    override_radius = float(
                        config['features'].get(
                            'guidance_feature_refined_override_radius_ratio',
                            config['features'].get('guidance_feature_override_radius_ratio', 1.2) * 0.6,
                        )
                    ) * refined_len
                    solve_guidance_override, solve_guidance_override_weight = _feature_guidance_override(
                        sv,
                        solve_frames,
                        refined_feature_lines,
                        radius=override_radius,
                        active_mask=edge_only_mask,
                    )
                    if np.any(solve_guidance_override_weight > 0):
                        print(
                            "  [sharp] built refined explicit field anchors: "
                            f"{int((solve_guidance_override_weight > 0).sum())} vertices, radius={override_radius:.4g}"
                        )
                corner_prior_count = int(
                    topo_cfg.get('small_sharp_corner_charge_count', min(8, len(feature_corners)))
                )
                if corner_prior_count > 0 and len(feature_corners) > 0:
                    corner_samples = np.asarray(feature_corners[:corner_prior_count], dtype=np.float64)
                    corner_charge_support_radius = float(
                        topo_cfg.get('small_sharp_corner_charge_support_radius_ratio', 0.9)
                    ) * refined_len
                    solve_singularity_mask, solve_singularity_indices = _build_corner_singularity_prior(
                        sv,
                        corner_samples,
                        charge_units=int(topo_cfg.get('small_sharp_corner_charge_units', 1)),
                        support_radius=corner_charge_support_radius,
                        max_points_per_corner=int(topo_cfg.get('small_sharp_corner_charge_support_points', 2)),
                    )
                    if np.any(solve_singularity_mask):
                        print(
                            "  [sharp] built refined corner singularity prior: "
                            f"{int(np.count_nonzero(solve_singularity_mask))} centres"
                        )
    if solve_singularity_mask is None and small_sharp_closed and tri_mesh is not None and len(feature_corners) > 0:
        corner_prior_count = int(
            topo_cfg.get('small_sharp_corner_charge_count', min(8, len(feature_corners)))
        )
        if corner_prior_count > 0:
            corner_samples = np.asarray(feature_corners[:corner_prior_count], dtype=np.float64)
            solve_pts = np.asarray(solve_tri_mesh[0] if solve_tri_mesh is not None else V_in, dtype=np.float64)
            support_len = (
                _mesh_mean_edge_length(solve_tri_mesh[0], solve_tri_mesh[1])
                if solve_tri_mesh is not None else
                _mesh_mean_edge_length(V_in, F_in)
            )
            solve_singularity_mask, solve_singularity_indices = _build_corner_singularity_prior(
                solve_pts,
                corner_samples,
                charge_units=int(topo_cfg.get('small_sharp_corner_charge_units', 1)),
                support_radius=float(
                    topo_cfg.get('small_sharp_corner_charge_support_radius_ratio', 0.9)
                ) * max(float(support_len), 1e-6),
                max_points_per_corner=int(topo_cfg.get('small_sharp_corner_charge_support_points', 2)),
            )
            if np.any(solve_singularity_mask):
                print(
                    "  [sharp] built corner singularity prior: "
                    f"{int(np.count_nonzero(solve_singularity_mask))} centres"
                )
    if closed_input:
        if small_sharp_closed:
            if selection_mode == 'quality':
                profile_overrides = topo_cfg.get(
                    'small_sharp_closed_quality_profile_overrides',
                    topo_cfg.get('small_sharp_closed_profile_overrides', {}),
                )
                retry_profiles_override = topo_cfg.get(
                    'small_sharp_closed_quality_retry_profiles',
                    topo_cfg.get('small_sharp_closed_retry_profiles', []),
                )
            else:
                profile_overrides = topo_cfg.get('small_sharp_closed_profile_overrides', {})
                retry_profiles_override = topo_cfg.get('small_sharp_closed_retry_profiles', [])
            print(
                "  [topo] using small-sharp closed-mesh profiles: "
                f"verts={len(V_in)}, faces={len(F_in)}, feature_lines={len(feature_lines)}"
            )
        elif selection_mode == 'quality':
            profile_overrides = topo_cfg.get('closed_quality_profile_overrides', {})
        else:
            profile_overrides = topo_cfg.get('closed_profile_overrides', {})
    if isinstance(profile_overrides, dict) and profile_overrides:
        base_profile.update(profile_overrides)

    profiles = _build_profile_sequence(
        base_profile, topo_cfg,
        closed_input=closed_input,
        selection_mode=selection_mode,
        retry_profiles_override=retry_profiles_override,
    )
    seen_profiles = {_profile_signature(p) for p in profiles}

    if closed_input:
        cap_max_loop_len = int(
            topo_cfg.get(
                'cap_even_boundary_loops_closed_max_loop_len',
                topo_cfg.get('cap_even_boundary_loops_max_loop_len', 128),
            )
        )
        cap_max_faces = int(topo_cfg.get('cap_even_boundary_loops_closed_max_added_faces', 12))
    else:
        cap_max_loop_len = int(topo_cfg.get('cap_even_boundary_loops_max_loop_len', 128))
        cap_max_faces = int(topo_cfg.get('cap_even_boundary_loops_max_added_faces', 1_000_000))

    quadV = quadF = None
    topo_ok_init = False
    topo_rep_init = None
    attempt_reports = []
    fallback_candidates = []
    valid_candidates = []

    for ai, prof in enumerate(profiles, start=1):
        try:
            repair_meta = {
                'filled': 0,
                'stitched': 0,
                'zipped': 0,
                'capped': 0,
            }
            if tri_mesh is not None:
                v_tri, f_tri = solve_tri_mesh if solve_tri_mesh is not None else tri_mesh
                out = initial_quad_mesh_from_mesh(
                    v_tri, f_tri,
                    metric_field_on_vertices=solve_metric_for_topology,
                    guidance_confidence=solve_confidence,
                    guidance_override=solve_guidance_override,
                    guidance_override_weight=solve_guidance_override_weight,
                    singularity_mask=solve_singularity_mask,
                    singularity_indices=solve_singularity_indices,
                    protected_mask=solve_protected_mask,
                    guidance_override_reliability_floor=float(prof.get('guidance_override_reliability_floor', 0.9)),
                    guidance_override_anchor_eps=float(prof.get('guidance_override_anchor_eps', 0.05)),
                    singularity_cancel_distance_ratio=float(prof.get('singularity_cancel_distance_ratio', 2.5)),
                    singularity_cancel_smoothing_iters=int(prof.get('singularity_cancel_smoothing_iters', 4)),
                    singularity_cancel_max_rounds=int(prof.get('singularity_cancel_max_rounds', 2)),
                    repair_high_order=bool(prof.get('repair_high_order', True)),
                    high_order_repair_max_rounds=int(prof.get('high_order_repair_max_rounds', 1)),
                    high_order_repair_expand_rounds=int(prof.get('high_order_repair_expand_rounds', 1)),
                    high_order_repair_smoothing_iters=int(prof.get('high_order_repair_smoothing_iters', 3)),
                    repair_flow=bool(prof.get('repair_flow', True)),
                    flow_repair_max_rounds=int(prof.get('flow_repair_max_rounds', 1)),
                    flow_repair_expand_rounds=int(prof.get('flow_repair_expand_rounds', 0)),
                    flow_repair_smoothing_iters=int(prof.get('flow_repair_smoothing_iters', 3)),
                    gradient_size=float(prof['gradient_size']),
                    crossfield_mu=float(prof['crossfield_mu']),
                    umbilic_smoothing=bool(prof['umbilic_smoothing']),
                    anisotropy_eps=float(prof['anisotropy_eps']),
                    mu_min_ratio=float(prof['mu_min_ratio']),
                    integer_constraints=bool(prof['integer_constraints']),
                    require_closed_topology=bool(prof.get('require_closed_topology', (not topo_allow_boundary))),
                    anisotropy_schedule=tuple(prof.get('anisotropy_schedule', [1.0, 0.7, 0.45, 0.25, 0.0])),
                    integer_projection_iters=int(prof.get('integer_projection_iters', 2)),
                    integer_potential_iters=int(prof.get('integer_potential_iters', 2)),
                    max_param_dist=float(prof.get('max_param_dist', 0.7)),
                    return_diagnostics=True,
                    use_igl_backend=bool(prof['use_igl_backend']),
                    igl_binary_path=prof.get('igl_binary_path'),
                    igl_gradient_size=float(prof['igl_gradient_size']),
                    igl_stiffness=float(prof['igl_stiffness']),
                    igl_direct_round=bool(prof.get('igl_direct_round', True)),
                    igl_miq_iter=int(prof.get('igl_miq_iter', 5)),
                )
            else:
                out = initial_quad_mesh_from_pointcloud(
                    points,
                    smoothed_lcf,
                    guidance_confidence=confidence,
                    guidance_override=guidance_override,
                    guidance_override_weight=guidance_override_weight,
                    normals=input_normals,
                    poisson_depth=int(prof['poisson_depth']),
                    gradient_size=float(prof['gradient_size']),
                    crossfield_mu=float(prof['crossfield_mu']),
                    umbilic_smoothing=bool(prof['umbilic_smoothing']),
                    anisotropy_eps=float(prof['anisotropy_eps']),
                    mu_min_ratio=float(prof['mu_min_ratio']),
                    integer_constraints=bool(prof['integer_constraints']),
                    require_closed_topology=bool(prof.get('require_closed_topology', (not topo_allow_boundary))),
                    anisotropy_schedule=tuple(prof.get('anisotropy_schedule', [1.0, 0.7, 0.45, 0.25, 0.0])),
                    integer_projection_iters=int(prof.get('integer_projection_iters', 2)),
                    integer_potential_iters=int(prof.get('integer_potential_iters', 2)),
                    max_param_dist=float(prof.get('max_param_dist', 0.7)),
                    return_diagnostics=True,
                    use_igl_backend=bool(prof['use_igl_backend']),
                    igl_binary_path=prof.get('igl_binary_path'),
                    igl_gradient_size=float(prof['igl_gradient_size']),
                    igl_stiffness=float(prof['igl_stiffness']),
                    igl_direct_round=bool(prof.get('igl_direct_round', True)),
                    igl_miq_iter=int(prof.get('igl_miq_iter', 5)),
                )
            if isinstance(out, tuple) and len(out) == 3:
                qv, qf, qdiag = out
            else:
                qv, qf = out
                qdiag = {}
            qv_oriented, qf_oriented, n_oriented = orient_quad_faces_consistently(
                qv, qf,
                ref_points=points,
                ref_normals=input_normals,
            )
            if n_oriented > 0:
                qv, qf = qv_oriented, qf_oriented
                qdiag = dict(qdiag)
                qdiag['quad_orientation_fix'] = {'flipped_quads': int(n_oriented)}
                print(f"  [topo] oriented {n_oriented} quads for consistent winding.")
            ok_topo, rep = certify_quad_topology(
                qv, qf,
                allow_boundary=topo_allow_boundary,
                require_all_quads=True,
            )
            quality_guard = (selection_mode == 'quality')
            base_qm = _quad_quality_metrics(qv, qf) if quality_guard else None
            allow_face_bridge_repairs = not (
                closed_input
                and selection_mode == 'quality'
                and not small_sharp_closed
                and bool(topo_cfg.get('closed_quality_disable_face_bridge_repairs', True))
            )
            if (
                (not allow_face_bridge_repairs)
                and closed_input
                and selection_mode == 'quality'
                and (not small_sharp_closed)
                and bool(topo_cfg.get('closed_quality_enable_face_bridge_repairs_on_boundary', True))
            ):
                min_edges = int(topo_cfg.get('closed_quality_face_bridge_min_boundary_edges', 64))
                min_loops = int(topo_cfg.get('closed_quality_face_bridge_min_boundary_loops', 2))
                if (
                    int(rep.get('boundary_edges', 0)) >= min_edges
                    or int(rep.get('boundary_loops', 0)) >= min_loops
                ):
                    allow_face_bridge_repairs = True
                    print(
                        "  [topo] enabling face-bridge repairs for closed-quality pass "
                        f"(boundary_edges={int(rep.get('boundary_edges', 0))}, "
                        f"boundary_loops={int(rep.get('boundary_loops', 0))})"
                    )
            if allow_face_bridge_repairs and bool(topo_cfg.get('fill_boundary_quads', True)) and len(qf) > 0:
                qv_filled, qf_filled, n_filled = fill_boundary_quad_holes(
                    qv, qf,
                    max_loop_len=int(topo_cfg.get('fill_boundary_quads_max_loop_len', 4)),
                    max_iters=int(topo_cfg.get('fill_boundary_quads_iters', 4)),
                )
                if n_filled > 0:
                    ok_topo_filled, rep_filled = certify_quad_topology(
                        qv_filled, qf_filled,
                        allow_boundary=topo_allow_boundary,
                        require_all_quads=True,
                    )
                    if _repair_improves(rep, rep_filled, expected_boundary_loops, qdiag):
                        cand_qm = None
                        if quality_guard:
                            cand_qm = _quad_quality_metrics(qv_filled, qf_filled)
                            if not _quality_gate_accepts(base_qm, cand_qm, topo_cfg):
                                cand_qm = None
                        if (not quality_guard) or (cand_qm is not None):
                            print(
                                f"  [topo] filled {n_filled} boundary quad holes: "
                                f"{int(rep.get('boundary_edges', 0))} -> {int(rep_filled.get('boundary_edges', 0))}"
                            )
                            repair_meta['filled'] += int(n_filled)
                            qv, qf = qv_filled, qf_filled
                            ok_topo, rep = ok_topo_filled, rep_filled
                            if quality_guard:
                                base_qm = cand_qm
            if allow_face_bridge_repairs and bool(topo_cfg.get('stitch_boundary_pairs', True)) and len(qf) > 0:
                qv_stitched, qf_stitched, n_stitched = stitch_boundary_loop_pairs(
                    qv, qf,
                    max_loop_len=int(topo_cfg.get('stitch_boundary_pairs_max_loop_len', 24)),
                    max_centroid_dist_ratio=float(topo_cfg.get('stitch_boundary_pairs_max_centroid_dist_ratio', 0.2)),
                )
                if n_stitched > 0:
                    ok_topo_stitched, rep_stitched = certify_quad_topology(
                        qv_stitched, qf_stitched,
                        allow_boundary=topo_allow_boundary,
                        require_all_quads=True,
                    )
                    if _repair_improves(rep, rep_stitched, expected_boundary_loops, qdiag):
                        cand_qm = None
                        if quality_guard:
                            cand_qm = _quad_quality_metrics(qv_stitched, qf_stitched)
                            if not _quality_gate_accepts(base_qm, cand_qm, topo_cfg):
                                cand_qm = None
                        if (not quality_guard) or (cand_qm is not None):
                            print(
                                f"  [topo] stitched {n_stitched} loop-bridge quads: "
                                f"{int(rep.get('boundary_edges', 0))} -> {int(rep_stitched.get('boundary_edges', 0))}"
                            )
                            repair_meta['stitched'] += int(n_stitched)
                            qv, qf = qv_stitched, qf_stitched
                            ok_topo, rep = ok_topo_stitched, rep_stitched
                            if quality_guard:
                                base_qm = cand_qm
            if allow_face_bridge_repairs and bool(topo_cfg.get('zipper_self_boundary_loops', True)) and len(qf) > 0:
                base_qm = _quad_quality_metrics(qv, qf)
                best_zip = None
                for zip_prof in _zipper_profiles(topo_cfg, closed_input=closed_input, dense_faces=len(qf)):
                    qv_zipped, qf_zipped, n_zipped = zipper_self_matched_boundary_loops(
                        qv, qf,
                        min_loop_len=int(zip_prof['min_loop_len']),
                        min_cyclic_gap_ratio=float(zip_prof['min_gap_ratio']),
                        max_pair_dist_ratio=float(zip_prof['max_pair_dist_ratio']),
                        max_iters=int(zip_prof['iters']),
                    )
                    if n_zipped <= 0:
                        continue
                    ok_topo_zipped, rep_zipped = certify_quad_topology(
                        qv_zipped, qf_zipped,
                        allow_boundary=topo_allow_boundary,
                        require_all_quads=True,
                    )
                    if not _repair_improves(rep, rep_zipped, expected_boundary_loops, qdiag):
                        continue
                    cand_qm = _quad_quality_metrics(qv_zipped, qf_zipped)
                    if not _quality_gate_accepts(base_qm, cand_qm, topo_cfg):
                        continue
                    cand = (
                        _repair_penalty(rep_zipped, expected_boundary_loops, qdiag),
                        cand_qm['mean_angle_dev'] - base_qm['mean_angle_dev'],
                        cand_qm['mean_aspect_ratio'] - base_qm['mean_aspect_ratio'],
                        -int(n_zipped),
                        zip_prof,
                        qv_zipped,
                        qf_zipped,
                        ok_topo_zipped,
                        rep_zipped,
                        n_zipped,
                    )
                    if best_zip is None or cand < best_zip:
                        best_zip = cand
                if best_zip is not None:
                    _, _, _, _, zip_prof, qv_zipped, qf_zipped, ok_topo_zipped, rep_zipped, n_zipped = best_zip
                    print(
                        f"  [topo] zipped {n_zipped} self-seam quads ({zip_prof['label']}): "
                        f"{int(rep.get('boundary_edges', 0))} -> {int(rep_zipped.get('boundary_edges', 0))}"
                    )
                    repair_meta['zipped'] += int(n_zipped)
                    qv, qf = qv_zipped, qf_zipped
                    ok_topo, rep = ok_topo_zipped, rep_zipped
            if allow_face_bridge_repairs and bool(topo_cfg.get('cap_rectangular_boundary_loops_with_field', True)) and len(qf) > 0:
                field_dirs = _metric_field_dirs_on_quads(points, smoothed_lcf, qv, qf)
                qv_rect, qf_rect, n_rect = cap_rectangular_boundary_loops_with_field(
                    qv, qf,
                    field_dirs,
                    max_loop_len=int(topo_cfg.get('cap_rectangular_boundary_loops_field_max_loop_len', 128)),
                    max_grid_side=int(topo_cfg.get('cap_rectangular_boundary_loops_field_max_grid_side', 32)),
                    planar_ratio_max=float(topo_cfg.get('cap_rectangular_boundary_loops_field_planar_ratio_max', 0.08)),
                    preserve_largest_loops=int(expected_boundary_loops),
                    field_align_min_strength=float(topo_cfg.get('cap_rectangular_boundary_loops_field_align_min_strength', 0.2)),
                )
                if n_rect > 0:
                    ok_topo_rect, rep_rect = certify_quad_topology(
                        qv_rect, qf_rect,
                        allow_boundary=topo_allow_boundary,
                        require_all_quads=True,
                    )
                    if _repair_improves(rep, rep_rect, expected_boundary_loops, qdiag):
                        cand_qm = None
                        if quality_guard:
                            cand_qm = _quad_quality_metrics(qv_rect, qf_rect)
                            if not _quality_gate_accepts(base_qm, cand_qm, topo_cfg):
                                cand_qm = None
                        if (not quality_guard) or (cand_qm is not None):
                            print(
                                f"  [topo] field-capped {n_rect} coons quads: "
                                f"{int(rep.get('boundary_edges', 0))} -> {int(rep_rect.get('boundary_edges', 0))}"
                            )
                            repair_meta['capped'] += int(n_rect)
                            qv, qf = qv_rect, qf_rect
                            ok_topo, rep = ok_topo_rect, rep_rect
                            if quality_guard:
                                base_qm = cand_qm
            if allow_face_bridge_repairs and bool(topo_cfg.get('cap_even_boundary_loops_with_field_rings', True)) and len(qf) > 0:
                field_dirs = _metric_field_dirs_on_quads(points, smoothed_lcf, qv, qf)
                qv_ring, qf_ring, n_ring = cap_even_boundary_loops_with_field_rings(
                    qv, qf,
                    field_dirs,
                    max_loop_len=int(topo_cfg.get('cap_even_boundary_loops_field_max_loop_len', 256)),
                    shrink=float(topo_cfg.get('cap_even_boundary_loops_field_shrink', 0.35)),
                    preserve_largest_loops=int(expected_boundary_loops),
                    field_align_min_strength=float(topo_cfg.get('cap_even_boundary_loops_field_align_min_strength', 0.1)),
                )
                if n_ring > 0:
                    ok_topo_ring, rep_ring = certify_quad_topology(
                        qv_ring, qf_ring,
                        allow_boundary=topo_allow_boundary,
                        require_all_quads=True,
                    )
                    if _repair_improves(rep, rep_ring, expected_boundary_loops, qdiag):
                        cand_qm = None
                        if quality_guard:
                            cand_qm = _quad_quality_metrics(qv_ring, qf_ring)
                            if not _quality_gate_accepts(base_qm, cand_qm, topo_cfg):
                                cand_qm = None
                        if (not quality_guard) or (cand_qm is not None):
                            print(
                                f"  [topo] field-ring capped {n_ring} quads: "
                                f"{int(rep.get('boundary_edges', 0))} -> {int(rep_ring.get('boundary_edges', 0))}"
                            )
                            repair_meta['capped'] += int(n_ring)
                            qv, qf = qv_ring, qf_ring
                            ok_topo, rep = ok_topo_ring, rep_ring
                            if quality_guard:
                                base_qm = cand_qm
            current_boundary_loops = int(rep.get('boundary_loops', 0))
            should_cap_excess_loops = current_boundary_loops > int(expected_boundary_loops)
            if allow_face_bridge_repairs and should_cap_excess_loops and bool(topo_cfg.get('cap_even_boundary_loops', True)) and len(qf) > 0:
                qv_capped, qf_capped, n_capped = cap_even_boundary_loops_with_center_quads(
                    qv, qf,
                    max_loop_len=cap_max_loop_len,
                    max_added_faces_per_loop=cap_max_faces,
                    preserve_largest_loops=int(expected_boundary_loops),
                )
                if n_capped > 0:
                    ok_topo_capped, rep_capped = certify_quad_topology(
                        qv_capped, qf_capped,
                        allow_boundary=topo_allow_boundary,
                        require_all_quads=True,
                    )
                    if _repair_improves(rep, rep_capped, expected_boundary_loops, qdiag):
                        cand_qm = None
                        if quality_guard:
                            cand_qm = _quad_quality_metrics(qv_capped, qf_capped)
                            if not _quality_gate_accepts(base_qm, cand_qm, topo_cfg):
                                cand_qm = None
                        if (not quality_guard) or (cand_qm is not None):
                            print(
                                f"  [topo] capped {n_capped} center-cap quads: "
                                f"{int(rep.get('boundary_edges', 0))} -> {int(rep_capped.get('boundary_edges', 0))}"
                            )
                            repair_meta['capped'] += int(n_capped)
                            qv, qf = qv_capped, qf_capped
                            ok_topo, rep = ok_topo_capped, rep_capped
                            if quality_guard:
                                base_qm = cand_qm
            if allow_face_bridge_repairs and bool(topo_cfg.get('remesh_boundary_with_miq', False)) and len(qf) > 0:
                min_edges = int(topo_cfg.get('remesh_boundary_min_edges', 200))
                min_loops = int(topo_cfg.get('remesh_boundary_min_loops', 1))
                max_faces = int(topo_cfg.get('remesh_boundary_max_faces', 60000))
                min_gain = int(topo_cfg.get('remesh_boundary_min_gain', 32))
                accept_if_better = bool(topo_cfg.get('remesh_boundary_accept_if_better', True))
                bypass_quality = bool(topo_cfg.get('remesh_boundary_bypass_quality_gate', False))
                if len(qf) <= max_faces and (
                    int(rep.get('boundary_edges', 0)) >= min_edges
                    or int(rep.get('boundary_loops', 0)) >= min_loops
                ):
                    tri_remesh = _triangulate_quads(qf)
                    if tri_remesh.size > 0:
                        remesh_prof = dict(prof)
                        if 'remesh_boundary_gradient_size' in topo_cfg:
                            remesh_prof['gradient_size'] = float(topo_cfg.get('remesh_boundary_gradient_size'))
                        if 'remesh_boundary_crossfield_mu' in topo_cfg:
                            remesh_prof['crossfield_mu'] = float(topo_cfg.get('remesh_boundary_crossfield_mu'))
                        if 'remesh_boundary_anisotropy_eps' in topo_cfg:
                            remesh_prof['anisotropy_eps'] = float(topo_cfg.get('remesh_boundary_anisotropy_eps'))
                        if 'remesh_boundary_mu_min_ratio' in topo_cfg:
                            remesh_prof['mu_min_ratio'] = float(topo_cfg.get('remesh_boundary_mu_min_ratio'))
                        if 'remesh_boundary_max_param_dist' in topo_cfg:
                            remesh_prof['max_param_dist'] = float(topo_cfg.get('remesh_boundary_max_param_dist'))
                        if 'remesh_boundary_integer_constraints' in topo_cfg:
                            remesh_prof['integer_constraints'] = bool(topo_cfg.get('remesh_boundary_integer_constraints'))
                        if 'remesh_boundary_use_igl_backend' in topo_cfg:
                            remesh_prof['use_igl_backend'] = bool(topo_cfg.get('remesh_boundary_use_igl_backend'))
                        if 'remesh_boundary_igl_gradient_size' in topo_cfg:
                            remesh_prof['igl_gradient_size'] = float(topo_cfg.get('remesh_boundary_igl_gradient_size'))
                        if 'remesh_boundary_igl_stiffness' in topo_cfg:
                            remesh_prof['igl_stiffness'] = float(topo_cfg.get('remesh_boundary_igl_stiffness'))
                        if 'remesh_boundary_igl_direct_round' in topo_cfg:
                            remesh_prof['igl_direct_round'] = bool(topo_cfg.get('remesh_boundary_igl_direct_round'))
                        if 'remesh_boundary_igl_miq_iter' in topo_cfg:
                            remesh_prof['igl_miq_iter'] = int(topo_cfg.get('remesh_boundary_igl_miq_iter'))
                        if 'remesh_boundary_require_closed_topology' in topo_cfg:
                            remesh_prof['require_closed_topology'] = bool(
                                topo_cfg.get('remesh_boundary_require_closed_topology')
                            )
                        else:
                            remesh_prof['require_closed_topology'] = bool(
                                remesh_prof.get('require_closed_topology', (not topo_allow_boundary))
                            )

                        remesh_frames = compute_vertex_frames(qv, tri_remesh)
                        remesh_metric_world = _transfer_metric_to_vertices(points, metric_world, qv)
                        remesh_metric = _project_world_metric_to_frames(remesh_metric_world, remesh_frames)
                        remesh_conf = None
                        if confidence is not None and len(confidence) == len(points):
                            remesh_conf = _transfer_scalar_to_vertices(points, confidence, qv)

                        try:
                            remesh_out = initial_quad_mesh_from_mesh(
                                qv, tri_remesh,
                                metric_field_on_vertices=remesh_metric,
                                guidance_confidence=remesh_conf,
                                guidance_override=None,
                                guidance_override_weight=None,
                                gradient_size=float(remesh_prof['gradient_size']),
                                crossfield_mu=float(remesh_prof['crossfield_mu']),
                                umbilic_smoothing=bool(remesh_prof['umbilic_smoothing']),
                                anisotropy_eps=float(remesh_prof['anisotropy_eps']),
                                mu_min_ratio=float(remesh_prof['mu_min_ratio']),
                                integer_constraints=bool(remesh_prof['integer_constraints']),
                                require_closed_topology=bool(remesh_prof.get('require_closed_topology', False)),
                                anisotropy_schedule=tuple(
                                    remesh_prof.get('anisotropy_schedule', [1.0, 0.7, 0.45, 0.25, 0.0])
                                ),
                                integer_projection_iters=int(remesh_prof.get('integer_projection_iters', 2)),
                                integer_potential_iters=int(remesh_prof.get('integer_potential_iters', 2)),
                                max_param_dist=float(remesh_prof.get('max_param_dist', 0.7)),
                                return_diagnostics=True,
                                use_igl_backend=bool(remesh_prof['use_igl_backend']),
                                igl_binary_path=remesh_prof.get('igl_binary_path'),
                                igl_gradient_size=float(remesh_prof['igl_gradient_size']),
                                igl_stiffness=float(remesh_prof['igl_stiffness']),
                                igl_direct_round=bool(remesh_prof.get('igl_direct_round', True)),
                                igl_miq_iter=int(remesh_prof.get('igl_miq_iter', 5)),
                            )
                        except Exception as exc:
                            print(f"  [topo] boundary MIQ remesh failed: {exc}")
                            remesh_out = None
                        if remesh_out is None:
                            pass
                        elif isinstance(remesh_out, tuple) and len(remesh_out) == 3:
                            qv_remesh, qf_remesh, remesh_diag = remesh_out
                        else:
                            qv_remesh, qf_remesh = remesh_out
                            remesh_diag = {}
                        if remesh_out is None:
                            qv_remesh = qf_remesh = None

                        if qv_remesh is not None and qf_remesh is not None:
                            qv_remesh, qf_remesh, n_remesh_orient = orient_quad_faces_consistently(
                                qv_remesh, qf_remesh,
                                ref_points=points,
                                ref_normals=input_normals,
                            )
                            if n_remesh_orient > 0:
                                remesh_diag = dict(remesh_diag)
                                remesh_diag['quad_orientation_fix'] = {'flipped_quads': int(n_remesh_orient)}
                            ok_topo_remesh, rep_remesh = certify_quad_topology(
                                qv_remesh, qf_remesh,
                                allow_boundary=topo_allow_boundary,
                                require_all_quads=True,
                            )
                            boundary_gain = int(rep.get('boundary_edges', 0)) - int(rep_remesh.get('boundary_edges', 0))
                            improves = _repair_improves(rep, rep_remesh, expected_boundary_loops, remesh_diag)
                            accept = improves or (accept_if_better and boundary_gain >= min_gain)
                            if accept:
                                cand_qm = None
                                if quality_guard and (not bypass_quality):
                                    cand_qm = _quad_quality_metrics(qv_remesh, qf_remesh)
                                    if not _quality_gate_accepts(base_qm, cand_qm, topo_cfg):
                                        cand_qm = None
                                if (not quality_guard) or bypass_quality or (cand_qm is not None):
                                    print(
                                        "  [topo] remeshed boundary via MIQ: "
                                        f"{int(rep.get('boundary_edges', 0))} -> {int(rep_remesh.get('boundary_edges', 0))}"
                                        f" (gain={boundary_gain})"
                                    )
                                    qv, qf = qv_remesh, qf_remesh
                                    ok_topo, rep = ok_topo_remesh, rep_remesh
                                    qdiag = dict(qdiag) if isinstance(qdiag, dict) else {}
                                    qdiag['miq_boundary_remesh'] = remesh_diag
                                    if quality_guard and (cand_qm is not None):
                                        base_qm = cand_qm
            if bool(topo_cfg.get('prune_boundary_fragments', True)) and len(qf) > 0:
                qv_pruned, qf_pruned = prune_to_largest_boundary_light_component(
                    qv, qf,
                    min_keep_faces=int(topo_cfg.get('prune_boundary_fragments_min_faces', 50)),
                    min_keep_ratio=float(topo_cfg.get('prune_boundary_fragments_min_ratio', 0.25)),
                )
                ok_topo_pruned, rep_pruned = certify_quad_topology(
                    qv_pruned, qf_pruned,
                    allow_boundary=topo_allow_boundary,
                    require_all_quads=True,
                )
                if _repair_improves(rep, rep_pruned, expected_boundary_loops, qdiag):
                    print(
                        f"  [topo] fragment prune improved boundary penalty: "
                        f"{int(rep.get('boundary_edges', 0))} -> {int(rep_pruned.get('boundary_edges', 0))}"
                    )
                    qv, qf = qv_pruned, qf_pruned
                    ok_topo, rep = ok_topo_pruned, rep_pruned
            current_boundary_loops = int(rep.get('boundary_loops', 0))
            should_cap_excess_loops = current_boundary_loops > int(expected_boundary_loops)
            if allow_face_bridge_repairs and should_cap_excess_loops and bool(topo_cfg.get('cap_even_boundary_loops', True)) and len(qf) > 0:
                qv_capped, qf_capped, n_capped = cap_even_boundary_loops_with_center_quads(
                    qv, qf,
                    max_loop_len=cap_max_loop_len,
                    max_added_faces_per_loop=cap_max_faces,
                    preserve_largest_loops=int(expected_boundary_loops),
                )
                if n_capped > 0:
                    ok_topo_capped, rep_capped = certify_quad_topology(
                        qv_capped, qf_capped,
                        allow_boundary=topo_allow_boundary,
                        require_all_quads=True,
                    )
                    if _repair_improves(rep, rep_capped, expected_boundary_loops, qdiag):
                        cand_qm = None
                        if quality_guard:
                            cand_qm = _quad_quality_metrics(qv_capped, qf_capped)
                            if not _quality_gate_accepts(base_qm, cand_qm, topo_cfg):
                                cand_qm = None
                        if (not quality_guard) or (cand_qm is not None):
                            print(
                                f"  [topo] post-prune capped {n_capped} center-cap quads: "
                                f"{int(rep.get('boundary_edges', 0))} -> {int(rep_capped.get('boundary_edges', 0))}"
                            )
                            repair_meta['capped'] += int(n_capped)
                            qv, qf = qv_capped, qf_capped
                            ok_topo, rep = ok_topo_capped, rep_capped
                            if quality_guard:
                                base_qm = cand_qm
            qv_reoriented, qf_reoriented, n_reoriented = orient_quad_faces_consistently(
                qv, qf,
                ref_points=points,
                ref_normals=input_normals,
            )
            if n_reoriented > 0:
                ok_topo_reoriented, rep_reoriented = certify_quad_topology(
                    qv_reoriented, qf_reoriented,
                    allow_boundary=topo_allow_boundary,
                    require_all_quads=True,
                )
                if _repair_improves(rep, rep_reoriented, expected_boundary_loops, qdiag) or rep_reoriented == rep:
                    qv, qf = qv_reoriented, qf_reoriented
                    ok_topo, rep = ok_topo_reoriented, rep_reoriented
                    repair_meta['reoriented'] = int(repair_meta.get('reoriented', 0)) + int(n_reoriented)
                    qdiag = dict(qdiag)
                    orient_fix = dict(qdiag.get('quad_orientation_fix') or {})
                    orient_fix['post_repair_flipped_quads'] = int(
                        orient_fix.get('post_repair_flipped_quads', 0) + int(n_reoriented)
                    )
                    qdiag['quad_orientation_fix'] = orient_fix
                    print(f"  [topo] re-oriented {n_reoriented} quads after repairs.")
            if (
                allow_face_bridge_repairs
                and bool(topo_cfg.get('post_orient_zipper', True))
                and len(qf) > 0
                and int(rep.get('boundary_loops', 0)) == max(1, int(expected_boundary_loops))
                and int(rep.get('boundary_edges', 0)) > 0
            ):
                zip_prof = _post_orient_zipper_profile(topo_cfg)
                qv_zip2, qf_zip2, n_zip2 = zipper_self_matched_boundary_loops(
                    qv, qf,
                    min_loop_len=int(zip_prof['min_loop_len']),
                    min_cyclic_gap_ratio=float(zip_prof['min_gap_ratio']),
                    max_pair_dist_ratio=float(zip_prof['max_pair_dist_ratio']),
                    max_iters=int(zip_prof['iters']),
                )
                if n_zip2 > 0:
                    zip2_qm = None
                    if quality_guard:
                        zip2_qm = _quad_quality_metrics(qv_zip2, qf_zip2)
                        if not _quality_gate_accepts(base_qm, zip2_qm, topo_cfg):
                            n_zip2 = 0
                    if n_zip2 == 0:
                        pass
                    else:
                        qv_zip2, qf_zip2, n_reoriented2 = orient_quad_faces_consistently(
                            qv_zip2, qf_zip2,
                            ref_points=points,
                            ref_normals=input_normals,
                        )
                        ok_topo_zip2, rep_zip2 = certify_quad_topology(
                            qv_zip2, qf_zip2,
                            allow_boundary=topo_allow_boundary,
                            require_all_quads=True,
                        )
                        if _repair_improves(rep, rep_zip2, expected_boundary_loops, qdiag):
                            print(
                                f"  [topo] post-orient zipped {n_zip2} seam quads: "
                                f"{int(rep.get('boundary_edges', 0))} -> {int(rep_zip2.get('boundary_edges', 0))}"
                            )
                            if n_reoriented2 > 0:
                                print(f"  [topo] re-oriented {n_reoriented2} quads after post-orient zipper.")
                            repair_meta['zipped'] += int(n_zip2)
                            qv, qf = qv_zip2, qf_zip2
                            ok_topo, rep = ok_topo_zip2, rep_zip2
                            if quality_guard and zip2_qm is not None:
                                base_qm = zip2_qm
            enough_quads = len(qf) >= topo_min_quads
            ok_case = bool(ok_topo) and bool(enough_quads)
            if ok_case and small_sharp_closed and closed_input:
                max_boundary_edges = int(topo_cfg.get('small_sharp_closed_max_boundary_edges', 24))
                max_boundary_loops = int(topo_cfg.get('small_sharp_closed_max_boundary_loops', 0))
                if (
                    int(rep.get('boundary_edges', 0)) > max_boundary_edges
                    or int(rep.get('boundary_loops', 0)) > max_boundary_loops
                ):
                    ok_case = False
            attempt_reports.append(
                f"attempt {ai}/{len(profiles)}: quads={len(qf)}, "
                f"ok_topo={ok_topo}, min_quads={topo_min_quads}, "
                f"{format_topology_report(rep)}"
            )
            print(
                f"  [topo] attempt {ai}/{len(profiles)} -> "
                f"{'PASS' if ok_case else 'FAIL'} "
                f"(quads={len(qf)}, use_igl={bool(prof['use_igl_backend'])}, "
                f"g={float(prof['gradient_size'])}, mu={float(prof['crossfield_mu'])}, "
                f"boundary_edges={int(rep.get('boundary_edges', 0))})"
            )
            sing = qdiag.get('crossfield_singularities')
            if sing is not None:
                msg = (
                    "  [field] singularities from cross-field: "
                    f"faces={sing['num_singular_faces']}, +={sing['num_positive']}, -={sing['num_negative']}"
                )
                if int(sing.get('num_singular_vertices', 0)) > 0:
                    msg += (
                        f", vertices={int(sing.get('num_singular_vertices', 0))}"
                        f", sum_units={int(sing.get('sum_vertex_units', 0))}"
                    )
                print(msg)
            filt = qdiag.get('crossfield_filter')
            if filt:
                print(
                    "  [field] singularity filter: "
                    f"pairs_cancelled={int(filt.get('pairs_cancelled', 0))}, "
                    f"rounds={int(filt.get('rounds', 0))}"
                )
            flow_fix = qdiag.get('flow_repair')
            if flow_fix and int(flow_fix.get('improved_rounds', 0)) > 0:
                print(
                    "  [field] flow repair: "
                    f"violations={int(flow_fix.get('num_violations_before', 0))} -> "
                    f"{int(flow_fix.get('num_violations_after', 0))}, "
                    f"rounds={int(flow_fix.get('improved_rounds', 0))}"
                )
            if 'anisotropy_alpha' in qdiag:
                print(f"  [field] anisotropy alpha used: {qdiag['anisotropy_alpha']:.2f}")
            miq_diag = qdiag.get('miq_extraction') or {}
            if miq_diag:
                uvx = miq_diag.get('uv_extraction') or {}
                winding = miq_diag.get('quad_winding') or {}
                rounding = miq_diag.get('miq_rounding') or {}
                print(
                    "  [miq] extraction: "
                    f"uv_skip={int(uvx.get('skipped_fold_overs', 0))}, "
                    f"uv_degenerate={int(uvx.get('degenerate_uv_triangles', 0))}, "
                    f"winding_fixed={int(winding.get('corrected_quads', 0))}, "
                    f"final_flips={int(rounding.get('flip_count_final', 0))}"
                )
            adaptive_profiles = _make_adaptive_profiles(
                prof, qdiag,
                closed_input=closed_input,
                small_sharp_closed=small_sharp_closed,
            )
            max_adaptive_profiles = int(topo_cfg.get('max_adaptive_profiles', 12))
            if closed_input and selection_mode == 'quality':
                max_adaptive_profiles = int(
                    topo_cfg.get(
                        'closed_quality_max_adaptive_profiles',
                        topo_cfg.get('quality_max_adaptive_profiles', 0),
                    )
                )
            if small_sharp_closed:
                max_adaptive_profiles = int(topo_cfg.get('small_sharp_max_adaptive_profiles', max_adaptive_profiles))
            for ap in adaptive_profiles:
                if len(profiles) >= max_adaptive_profiles:
                    break
                sig = _profile_signature(ap)
                if sig in seen_profiles:
                    continue
                seen_profiles.add(sig)
                profiles.append(ap)
                print(
                    "  [field] queued adaptive retry: "
                    f"mu={float(ap.get('crossfield_mu', 0.0)):.1f}, "
                    f"igl_g={float(ap.get('igl_gradient_size', 0.0)):.1f}, "
                    f"igl_stiffness={float(ap.get('igl_stiffness', 0.0)):.1f}"
                )
            repair_meta['repair_total'] = int(
                repair_meta['filled']
                + repair_meta['stitched']
                + repair_meta['zipped']
                + repair_meta['capped']
            )
            qm = _quad_quality_metrics(qv, qf)
            repair_meta.update({
                'mean_angle_dev': float(qm['mean_angle_dev']),
                'p90_angle_dev': float(qm['p90_angle_dev']),
                'mean_aspect_ratio': float(qm['mean_aspect_ratio']),
                'p90_aspect_ratio': float(qm['p90_aspect_ratio']),
                'mean_planarity': float(qm.get('mean_planarity', 0.0)),
                'p90_planarity': float(qm.get('p90_planarity', 0.0)),
            })
            if bool(topo_cfg.get('selection_quality_probe_stack_pairs', True)):
                _, _, stack_probe = prune_stacked_parallel_quads(
                    qv, qf,
                    center_dist_ratio=float(topo_cfg.get('stack_prune_center_dist_ratio', 0.01)),
                    normal_dot_min=float(topo_cfg.get('stack_prune_normal_dot_min', 0.95)),
                    max_iters=0,
                    max_candidates_per_face=int(topo_cfg.get('stack_prune_max_candidates_per_face', 8)),
                )
                stack_pairs = int(stack_probe.get('initial_pairs', 0))
                repair_meta['stack_pairs'] = stack_pairs
                repair_meta['stack_pairs_ratio'] = (
                    float(stack_pairs) / max(1.0, float(len(qf)))
                )
            fallback_cand = (
                _topology_penalty(rep, expected_boundary_loops, qdiag),
                len(qf), qv, qf, rep, ok_topo, qdiag, ai, repair_meta,
            )
            fallback_candidates.append(fallback_cand)
            if ok_case:
                valid_cand = (
                    _topology_penalty(rep, expected_boundary_loops, qdiag),
                    len(qf), qv, qf, rep, ok_topo, qdiag, ai, repair_meta,
                )
                valid_candidates.append(valid_cand)
        except Exception as exc:
            attempt_reports.append(f"attempt {ai}/{len(profiles)}: EXCEPTION: {exc}")
            print(f"  [topo] attempt {ai}/{len(profiles)} -> EXCEPTION: {exc}")

    sel_cfg = _selection_cfg(topo_cfg, closed_input=closed_input)
    best_valid_dense = _select_dense_candidate(valid_candidates, sel_cfg)
    best_valid_topology = _select_topology_candidate(valid_candidates, sel_cfg)
    best_valid_balanced = _select_candidate(valid_candidates, sel_cfg)
    best_valid_by_mode = {
        'quality': best_valid_dense,
        'balanced': best_valid_balanced,
        'topology': best_valid_topology,
    }
    best_valid = best_valid_by_mode.get(selection_mode)
    if small_sharp_closed and selection_mode == 'quality':
        best_valid = _select_small_sharp_quality_candidate(valid_candidates, topo_cfg)
    best_fallback = _select_candidate(fallback_candidates, sel_cfg)

    if best_valid is not None:
        _, _, quadV, quadF, topo_rep_init, topo_ok_init, best_qdiag, best_ai, _ = best_valid
        print(
            f"  [topo] selected {selection_mode} attempt "
            f"{best_ai}/{len(profiles)} as main output."
        )
    elif quadV is None:
        if best_fallback is not None and (not topo_strict):
            _, _, quadV, quadF, topo_rep_init, topo_ok_init, best_qdiag, best_ai, _ = best_fallback
            print(
                f"  [topo] using best fallback attempt {best_ai}/{len(profiles)} "
                f"with {len(quadF)} quads (strict=false)."
            )
        else:
            details = "\n".join(attempt_reports)
            raise RuntimeError(
                "Failed to produce topology-valid initial quad mesh.\n"
                f"Attempts:\n{details}\n"
                "Tune topology.retry_profiles / miq parameters in the config."
            )

    print(f"  Initial mesh: {len(quadV)} vertices, {len(quadF)} quads.")

    print(f"  Topology (initial): {format_topology_report(topo_rep_init)}")
    if topo_strict and ((not topo_ok_init) or (len(quadF) < topo_min_quads)):
        raise RuntimeError(
            "Initial quad topology certification failed. "
            "Set topology.strict=false to continue for debugging."
        )

    if args.vis and ps is not None:
        ps.init()
        ps.set_up_dir("z_up")
        register_point_cloud("input_cloud", points, radius=0.002)
        register_mesh("initial_quad", quadV, quadF, color=(0.2, 0.8, 0.2))

    # ── 6. Projective dynamics optimisation ───────────────────────────
    print("Running PD optimisation …")
    pd_cfg = config['pd']

    from scipy.spatial import KDTree as SpatialKDTree
    from src.optimization.constraints import AntiFlipRegularizer, _quad_signed_areas_vectorized

    quad_centers   = np.mean(quadV[quadF], axis=1)
    pt_tree        = SpatialKDTree(points)
    _, near_idx    = pt_tree.query(quad_centers)

    def target_metric_fn(qi):
        return metric_for_topology[near_idx[qi]]

    if tri_mesh is not None:
        feature_ref_length = _mesh_mean_edge_length(V_in, F_in)
    else:
        bbox = points.max(axis=0) - points.min(axis=0)
        feature_ref_length = float(np.linalg.norm(bbox) / max(np.sqrt(float(len(points))), 1.0))
    feature_point_constraints, feature_edge_constraints = _build_feature_constraints(
        quadV, quadF, feature_lines, config, reference_length=feature_ref_length,
    )
    print(
        "  [features] constraints: "
        f"point={len(feature_point_constraints)}, edge={len(feature_edge_constraints)}"
    )

    solver = ProjectiveDynamicsSolver(
        vertices=quadV,
        quads=quadF,
        target_metric_field=target_metric_fn,
        smoothness_weight=pd_cfg['mu'],
        feature_point_constraints=feature_point_constraints or None,
        feature_edge_constraints=feature_edge_constraints or None,
        anti_flip_weight=pd_cfg.get('anti_flip_weight', 10.0),
        anti_flip_eps=pd_cfg.get('anti_flip_eps', 1e-6),
        surface_attach_weight=pd_cfg.get('surface_attach_weight', 0.0),
        flip_backtracking=pd_cfg.get('flip_backtracking', True),
        flip_backtrack_steps=pd_cfg.get('flip_backtrack_steps', 8),
    )

    final_V = solver.optimize(
        surface_points=points,
        surface_normals=input_normals,
        max_iter=pd_cfg['iterations'],
    )

    if export_ref_points is not None and export_ref_normals is not None and len(export_ref_normals) == len(export_ref_points):
        final_V, final_F, n_final_aligned = finalize_quad_orientation_with_reference(
            final_V, quadF,
            ref_points=export_ref_points,
            ref_normals=export_ref_normals,
        )
        quadF = final_F
        if n_final_aligned > 0:
            print(f"  [topo] final reference-normal alignment flipped {n_final_aligned} quads.")
    if closed_input:
        final_V, final_F, n_outward = align_quad_faces_outward_from_centroid(final_V, quadF)
        quadF = final_F
        if n_outward > 0:
            print(f"  [topo] outward-orient flipped {n_outward} quads.")

    if bool(topo_cfg.get('stack_prune_enabled', True)):
        topo_ok_before_prune, topo_rep_before_prune = certify_quad_topology(
            final_V, quadF,
            allow_boundary=topo_allow_boundary,
            require_all_quads=True,
        )
        pruned_V, pruned_F, prune_stats = prune_stacked_parallel_quads(
            final_V, quadF,
            center_dist_ratio=float(topo_cfg.get('stack_prune_center_dist_ratio', 0.01)),
            normal_dot_min=float(topo_cfg.get('stack_prune_normal_dot_min', 0.95)),
            max_iters=int(topo_cfg.get('stack_prune_max_iters', 2)),
            max_candidates_per_face=int(topo_cfg.get('stack_prune_max_candidates_per_face', 8)),
        )
        if int(prune_stats.get('removed_faces', 0)) > 0:
            prune_accept_cfg = topo_cfg
            if small_sharp_closed:
                prune_accept_cfg = dict(topo_cfg)
                if 'small_sharp_stack_prune_min_pair_gain' in topo_cfg:
                    prune_accept_cfg['stack_prune_min_pair_gain'] = topo_cfg['small_sharp_stack_prune_min_pair_gain']
                if 'small_sharp_stack_prune_max_boundary_increase' in topo_cfg:
                    prune_accept_cfg['stack_prune_max_boundary_increase'] = topo_cfg[
                        'small_sharp_stack_prune_max_boundary_increase'
                    ]
                if 'small_sharp_stack_prune_max_loop_increase' in topo_cfg:
                    prune_accept_cfg['stack_prune_max_loop_increase'] = topo_cfg['small_sharp_stack_prune_max_loop_increase']
                if 'small_sharp_stack_prune_max_chain_increase' in topo_cfg:
                    prune_accept_cfg['stack_prune_max_chain_increase'] = topo_cfg[
                        'small_sharp_stack_prune_max_chain_increase'
                    ]
                if 'small_sharp_stack_prune_max_irregular_increase' in topo_cfg:
                    prune_accept_cfg['stack_prune_max_irregular_increase'] = topo_cfg[
                        'small_sharp_stack_prune_max_irregular_increase'
                    ]
            topo_ok_pruned, topo_rep_pruned = certify_quad_topology(
                pruned_V, pruned_F,
                allow_boundary=topo_allow_boundary,
                require_all_quads=True,
            )
            if (
                topo_ok_before_prune == topo_ok_pruned
                and _accept_stack_prune(topo_rep_before_prune, topo_rep_pruned, prune_stats, prune_accept_cfg)
            ):
                print(
                    "  [topo] pruned likely stacked quads: "
                    f"removed={int(prune_stats.get('removed_faces', 0))}, "
                    f"stack_pairs={int(prune_stats.get('initial_pairs', 0))} -> {int(prune_stats.get('remaining_pairs', 0))}, "
                    f"boundary_edges={int(topo_rep_before_prune.get('boundary_edges', 0))} -> {int(topo_rep_pruned.get('boundary_edges', 0))}"
                )
                final_V, quadF = pruned_V, pruned_F
            else:
                print(
                    "  [topo] rejected stacked-quad prune: "
                    f"removed={int(prune_stats.get('removed_faces', 0))}, "
                    f"stack_pairs={int(prune_stats.get('initial_pairs', 0))} -> {int(prune_stats.get('remaining_pairs', 0))}, "
                    f"boundary_edges={int(topo_rep_before_prune.get('boundary_edges', 0))} -> {int(topo_rep_pruned.get('boundary_edges', 0))}"
                )

    topo_ok_current, topo_rep_current = certify_quad_topology(
        final_V, quadF,
        allow_boundary=topo_allow_boundary,
        require_all_quads=True,
    )

    if bool(topo_cfg.get('post_pd_planarize', True)):
        base_qm = _quad_quality_metrics(final_V, quadF)
        planar_V = _planarize_quads(
            final_V, quadF,
            iters=int(topo_cfg.get('post_pd_planarize_iters', 1)),
            alpha=float(topo_cfg.get('post_pd_planarize_alpha', 0.25)),
            boundary_alpha_scale=float(topo_cfg.get('post_pd_planarize_boundary_alpha_scale', 0.35)),
        )
        if export_ref_points is not None and export_ref_normals is not None and len(export_ref_normals) == len(export_ref_points):
            planar_V, planar_F, n_planar_aligned = finalize_quad_orientation_with_reference(
                planar_V, quadF,
                ref_points=export_ref_points,
                ref_normals=export_ref_normals,
            )
        else:
            planar_F = quadF
            n_planar_aligned = 0
        topo_ok_planar, topo_rep_planar = certify_quad_topology(
            planar_V, planar_F,
            allow_boundary=topo_allow_boundary,
            require_all_quads=True,
        )
        planar_qm = _quad_quality_metrics(planar_V, planar_F)
        planarity_gain = (
            planar_qm['mean_planarity'] <= base_qm['mean_planarity'] * float(topo_cfg.get('post_pd_planarize_min_mean_planarity_ratio', 0.95))
            or planar_qm['p90_planarity'] <= base_qm['p90_planarity'] * float(topo_cfg.get('post_pd_planarize_min_p90_planarity_ratio', 0.95))
        )
        if (
            topo_ok_planar == topo_ok_current
            and topo_rep_planar == topo_rep_current
            and planarity_gain
            and _quality_gate_accepts(base_qm, planar_qm, topo_cfg)
        ):
            final_V, quadF = planar_V, planar_F
            if n_planar_aligned > 0:
                print(f"  [topo] planarization re-aligned {n_planar_aligned} quads to reference normals.")
            print(
                "  [topo] accepted post-PD planarization: "
                f"mean_planarity {base_qm['mean_planarity']:.4f} -> {planar_qm['mean_planarity']:.4f}, "
                f"p90_planarity {base_qm['p90_planarity']:.4f} -> {planar_qm['p90_planarity']:.4f}"
            )
            topo_ok_current, topo_rep_current = topo_ok_planar, topo_rep_planar

    if bool(topo_cfg.get('post_pd_shape_polish', True)):
        base_qm = _quad_quality_metrics(final_V, quadF)
        polish_V = _polish_quad_shape(
            final_V, quadF,
            aspect_threshold=float(topo_cfg.get('post_pd_shape_polish_aspect_threshold', 3.5)),
            alpha=float(topo_cfg.get('post_pd_shape_polish_alpha', 0.12)),
            iters=int(topo_cfg.get('post_pd_shape_polish_iters', 1)),
            ring_expand=int(topo_cfg.get('post_pd_shape_polish_ring_expand', 1)),
        )
        if export_ref_points is not None and export_ref_normals is not None and len(export_ref_normals) == len(export_ref_points):
            polish_V, polish_F, n_polish_aligned = finalize_quad_orientation_with_reference(
                polish_V, quadF,
                ref_points=export_ref_points,
                ref_normals=export_ref_normals,
            )
        else:
            polish_F = quadF
            n_polish_aligned = 0
        topo_ok_polish, topo_rep_polish = certify_quad_topology(
            polish_V, polish_F,
            allow_boundary=topo_allow_boundary,
            require_all_quads=True,
        )
        polish_qm = _quad_quality_metrics(polish_V, polish_F)
        shape_gain = (
            polish_qm['mean_aspect_ratio'] <= base_qm['mean_aspect_ratio'] * float(topo_cfg.get('post_pd_shape_polish_max_mean_aspect_ratio', 1.001))
            and polish_qm['p90_aspect_ratio'] <= base_qm['p90_aspect_ratio'] * float(topo_cfg.get('post_pd_shape_polish_max_p90_aspect_ratio', 1.0))
            and polish_qm['mean_angle_dev'] <= base_qm['mean_angle_dev'] - float(topo_cfg.get('post_pd_shape_polish_min_angle_gain_deg', 0.01))
        )
        if (
            topo_ok_polish == topo_ok_current
            and topo_rep_polish == topo_rep_current
            and shape_gain
            and _quality_gate_accepts(base_qm, polish_qm, topo_cfg)
        ):
            final_V, quadF = polish_V, polish_F
            if n_polish_aligned > 0:
                print(f"  [topo] shape polish re-aligned {n_polish_aligned} quads to reference normals.")
            print(
                "  [topo] accepted post-PD shape polish: "
                f"mean_aspect {base_qm['mean_aspect_ratio']:.4f} -> {polish_qm['mean_aspect_ratio']:.4f}, "
                f"p90_aspect {base_qm['p90_aspect_ratio']:.4f} -> {polish_qm['p90_aspect_ratio']:.4f}"
            )
            topo_ok_current, topo_rep_current = topo_ok_polish, topo_rep_polish

    if bool(topo_cfg.get('post_pd_cleanup', True)) and len(quadF) > 0:
        current_qm = _quad_quality_metrics(final_V, quadF)

        def _try_cleanup(label, cand_V, cand_F, desc):
            nonlocal final_V, quadF, topo_ok_current, topo_rep_current, current_qm
            if cand_F.ndim != 2 or len(cand_F) == 0:
                return False
            ok, rep = certify_quad_topology(
                cand_V, cand_F,
                allow_boundary=topo_allow_boundary,
                require_all_quads=True,
            )
            if not _repair_improves(topo_rep_current, rep, expected_boundary_loops, None):
                return False
            cand_qm = _quad_quality_metrics(cand_V, cand_F)
            if not _quality_gate_accepts(current_qm, cand_qm, topo_cfg):
                return False
            before_edges = int(topo_rep_current.get('boundary_edges', 0))
            after_edges = int(rep.get('boundary_edges', 0))
            print(
                f"  [topo] {label} ({desc}): "
                f"boundary_edges {before_edges} -> {after_edges}"
            )
            final_V, quadF = cand_V, cand_F
            topo_ok_current, topo_rep_current = ok, rep
            current_qm = cand_qm
            return True

        qv_filled, qf_filled, n_filled = fill_boundary_quad_holes(
            final_V, quadF,
            max_loop_len=int(topo_cfg.get('fill_boundary_quads_max_loop_len', 4)),
            max_iters=int(topo_cfg.get('fill_boundary_quads_iters', 4)),
        )
        if n_filled > 0:
            _try_cleanup('cleanup fill', qv_filled, qf_filled, f'filled {n_filled} holes')

        if bool(topo_cfg.get('cap_rectangular_boundary_loops_with_field', True)):
            field_dirs = _metric_field_dirs_on_quads(points, smoothed_lcf, final_V, quadF)
            qv_rect, qf_rect, n_rect = cap_rectangular_boundary_loops_with_field(
                final_V, quadF,
                field_dirs,
                max_loop_len=int(topo_cfg.get('cap_rectangular_boundary_loops_field_max_loop_len', 128)),
                max_grid_side=int(topo_cfg.get('cap_rectangular_boundary_loops_field_max_grid_side', 32)),
                planar_ratio_max=float(topo_cfg.get('cap_rectangular_boundary_loops_field_planar_ratio_max', 0.08)),
                preserve_largest_loops=int(expected_boundary_loops),
                field_align_min_strength=float(topo_cfg.get('cap_rectangular_boundary_loops_field_align_min_strength', 0.2)),
            )
            if n_rect > 0:
                _try_cleanup('cleanup rectangular cap', qv_rect, qf_rect, f'capped {n_rect} loops')

        if bool(topo_cfg.get('cap_even_boundary_loops_with_field_rings', True)):
            field_dirs = _metric_field_dirs_on_quads(points, smoothed_lcf, final_V, quadF)
            qv_ring, qf_ring, n_ring = cap_even_boundary_loops_with_field_rings(
                final_V, quadF,
                field_dirs,
                max_loop_len=int(topo_cfg.get('cap_even_boundary_loops_field_max_loop_len', 256)),
                shrink=float(topo_cfg.get('cap_even_boundary_loops_field_shrink', 0.35)),
                preserve_largest_loops=int(expected_boundary_loops),
                field_align_min_strength=float(topo_cfg.get('cap_even_boundary_loops_field_align_min_strength', 0.1)),
            )
            if n_ring > 0:
                _try_cleanup('cleanup ring cap', qv_ring, qf_ring, f'added {n_ring} rings')

        prune_V, prune_F, prune_stats = prune_stacked_parallel_quads(
            final_V, quadF,
            center_dist_ratio=float(topo_cfg.get('stack_prune_center_dist_ratio', 0.01)),
            normal_dot_min=float(topo_cfg.get('stack_prune_normal_dot_min', 0.95)),
            max_iters=int(topo_cfg.get('stack_prune_max_iters', 2)),
            max_candidates_per_face=int(topo_cfg.get('stack_prune_max_candidates_per_face', 8)),
        )
        n_removed = int(prune_stats.get('removed_faces', 0))
        if n_removed > 0:
            _try_cleanup('cleanup prune', prune_V, prune_F, f'removed {n_removed} stacked quads')

    # ── 7. Quality report ──────────────────────────────────────────────
    _af = AntiFlipRegularizer(quadF, eps0=pd_cfg.get('anti_flip_eps', 1e-6))
    n_flip, n_near = _af.count_flipped(final_V)
    print(f"  Post-PD: {n_flip} flipped, {n_near} near-degenerate / {len(quadF)} quads.")

    topo_ok_final, topo_rep_final = certify_quad_topology(
        final_V, quadF,
        allow_boundary=topo_allow_boundary,
        require_all_quads=True,
    )
    main_qm_final = _quad_quality_metrics(final_V, quadF)
    print(f"  Topology (final):   {format_topology_report(topo_rep_final)}")
    if topo_strict and ((not topo_ok_final) or (len(quadF) < topo_min_quads)):
        raise RuntimeError(
            "Final quad topology certification failed. "
            "Set topology.strict=false to export for debugging."
        )

    # ── 8. Write output ────────────────────────────────────────────────
    write_mesh(args.output, final_V, quadF)
    print(f"Quad mesh saved → {args.output}")
    export_tri_preview = bool(topo_cfg.get('export_triangle_preview', True))
    if (
        export_tri_preview
        and export_ref_points is not None
        and export_ref_normals is not None
        and len(export_ref_normals) == len(export_ref_points)
    ):
        tri_V, tri_F = triangulate_quads_for_reference_preview(
            final_V, quadF,
            ref_points=export_ref_points,
            ref_normals=export_ref_normals,
        )
        tri_V, tri_F, tri_flipped = align_triangle_faces_to_reference_normals(
            tri_V, tri_F,
            ref_points=export_ref_points,
            ref_normals=export_ref_normals,
        )
        root, ext = os.path.splitext(args.output)
        tri_output = f"{root}_preview_tri{ext or '.obj'}"
        write_mesh(tri_output, tri_V, tri_F)
        print(f"  [topo] exported triangle preview → {tri_output}")
        if tri_flipped > 0:
            print(f"  [topo] preview reference-normal alignment flipped {tri_flipped} triangles.")

    export_topology_alt = bool(topo_cfg.get('export_topology_alternative', True))
    if export_topology_alt and best_valid_dense is not None and best_valid_topology is not None:
        dense_ai = int(best_valid_dense[7])
        topo_ai = int(best_valid_topology[7])
        if dense_ai != topo_ai:
            alt_quadV = np.asarray(best_valid_topology[2], dtype=np.float64)
            alt_quadF = np.asarray(best_valid_topology[3], dtype=np.int64)
            alt_final_V = alt_quadV
            alt_qm_final = None
            try:
                alt_centers = np.mean(alt_quadV[alt_quadF], axis=1)
                _, alt_near_idx = pt_tree.query(alt_centers)

                def alt_target_metric_fn(qi):
                    return metric_for_topology[alt_near_idx[qi]]

                alt_feature_point_constraints, alt_feature_edge_constraints = _build_feature_constraints(
                    alt_quadV, alt_quadF, feature_lines, config, reference_length=feature_ref_length,
                )

                alt_solver = ProjectiveDynamicsSolver(
                    vertices=alt_quadV,
                    quads=alt_quadF,
                    target_metric_field=alt_target_metric_fn,
                    smoothness_weight=pd_cfg['mu'],
                    feature_point_constraints=alt_feature_point_constraints or None,
                    feature_edge_constraints=alt_feature_edge_constraints or None,
                    anti_flip_weight=pd_cfg.get('anti_flip_weight', 10.0),
                    anti_flip_eps=pd_cfg.get('anti_flip_eps', 1e-6),
                    surface_attach_weight=pd_cfg.get('surface_attach_weight', 0.0),
                    flip_backtracking=pd_cfg.get('flip_backtracking', True),
                    flip_backtrack_steps=pd_cfg.get('flip_backtrack_steps', 8),
                )
                alt_final_V = alt_solver.optimize(
                    surface_points=points,
                    surface_normals=input_normals,
                    max_iter=pd_cfg['iterations'],
                )
            except Exception as exc:
                print(f"  [topo] topology-first alternative PD failed, exporting pre-PD mesh: {exc}")

            if export_ref_points is not None and export_ref_normals is not None and len(export_ref_normals) == len(export_ref_points):
                alt_final_V, alt_quadF, n_alt_aligned = finalize_quad_orientation_with_reference(
                    alt_final_V, alt_quadF,
                    ref_points=export_ref_points,
                    ref_normals=export_ref_normals,
                )
                if n_alt_aligned > 0:
                    print(f"  [topo] topology-first final reference-normal alignment flipped {n_alt_aligned} quads.")
            if closed_input:
                alt_final_V, alt_quadF, n_alt_outward = align_quad_faces_outward_from_centroid(alt_final_V, alt_quadF)
                if n_alt_outward > 0:
                    print(f"  [topo] topology-first outward-orient flipped {n_alt_outward} quads.")

            if bool(topo_cfg.get('stack_prune_enabled', True)):
                alt_ok_before_prune, alt_rep_before_prune = certify_quad_topology(
                    alt_final_V, alt_quadF,
                    allow_boundary=topo_allow_boundary,
                    require_all_quads=True,
                )
                alt_pruned_V, alt_pruned_F, alt_prune_stats = prune_stacked_parallel_quads(
                    alt_final_V, alt_quadF,
                    center_dist_ratio=float(topo_cfg.get('stack_prune_center_dist_ratio', 0.01)),
                    normal_dot_min=float(topo_cfg.get('stack_prune_normal_dot_min', 0.95)),
                    max_iters=int(topo_cfg.get('stack_prune_max_iters', 2)),
                    max_candidates_per_face=int(topo_cfg.get('stack_prune_max_candidates_per_face', 8)),
                )
                if int(alt_prune_stats.get('removed_faces', 0)) > 0:
                    alt_ok_pruned, alt_rep_pruned = certify_quad_topology(
                        alt_pruned_V, alt_pruned_F,
                        allow_boundary=topo_allow_boundary,
                        require_all_quads=True,
                    )
                    if (
                        alt_ok_before_prune == alt_ok_pruned
                        and _accept_stack_prune(alt_rep_before_prune, alt_rep_pruned, alt_prune_stats, topo_cfg)
                    ):
                        print(
                            "  [topo] topology-first pruned likely stacked quads: "
                            f"removed={int(alt_prune_stats.get('removed_faces', 0))}, "
                            f"stack_pairs={int(alt_prune_stats.get('initial_pairs', 0))} -> {int(alt_prune_stats.get('remaining_pairs', 0))}, "
                            f"boundary_edges={int(alt_rep_before_prune.get('boundary_edges', 0))} -> {int(alt_rep_pruned.get('boundary_edges', 0))}"
                        )
                        alt_final_V, alt_quadF = alt_pruned_V, alt_pruned_F
                    else:
                        print(
                            "  [topo] topology-first rejected stacked-quad prune: "
                            f"removed={int(alt_prune_stats.get('removed_faces', 0))}, "
                            f"stack_pairs={int(alt_prune_stats.get('initial_pairs', 0))} -> {int(alt_prune_stats.get('remaining_pairs', 0))}, "
                            f"boundary_edges={int(alt_rep_before_prune.get('boundary_edges', 0))} -> {int(alt_rep_pruned.get('boundary_edges', 0))}"
                        )

            alt_ok_final, alt_rep_final = certify_quad_topology(
                alt_final_V, alt_quadF,
                allow_boundary=topo_allow_boundary,
                require_all_quads=True,
            )
            alt_qm_final = _quad_quality_metrics(alt_final_V, alt_quadF)
            promotion_enabled = bool(topo_cfg.get('quality_post_pd_alt_promotion_enabled', True))
            promotion_cfg = topo_cfg
            if small_sharp_closed:
                promotion_enabled = bool(
                    topo_cfg.get(
                        'small_sharp_quality_post_pd_alt_promotion_enabled',
                        promotion_enabled,
                    )
                )
                promotion_cfg = dict(topo_cfg)
                if 'small_sharp_quality_post_pd_alt_promotion_min_boundary_gain' in topo_cfg:
                    promotion_cfg['quality_post_pd_alt_promotion_min_boundary_gain'] = topo_cfg[
                        'small_sharp_quality_post_pd_alt_promotion_min_boundary_gain'
                    ]
                if 'small_sharp_quality_post_pd_alt_promotion_min_face_ratio' in topo_cfg:
                    promotion_cfg['quality_post_pd_alt_promotion_min_face_ratio'] = topo_cfg[
                        'small_sharp_quality_post_pd_alt_promotion_min_face_ratio'
                    ]
                if 'small_sharp_quality_post_pd_alt_max_mean_angle_increase_deg' in topo_cfg:
                    promotion_cfg['quality_post_pd_alt_max_mean_angle_increase_deg'] = topo_cfg[
                        'small_sharp_quality_post_pd_alt_max_mean_angle_increase_deg'
                    ]
                if 'small_sharp_quality_post_pd_alt_max_p90_angle_increase_deg' in topo_cfg:
                    promotion_cfg['quality_post_pd_alt_max_p90_angle_increase_deg'] = topo_cfg[
                        'small_sharp_quality_post_pd_alt_max_p90_angle_increase_deg'
                    ]
                if 'small_sharp_quality_post_pd_alt_max_mean_aspect_increase_ratio' in topo_cfg:
                    promotion_cfg['quality_post_pd_alt_max_mean_aspect_increase_ratio'] = topo_cfg[
                        'small_sharp_quality_post_pd_alt_max_mean_aspect_increase_ratio'
                    ]
                if 'small_sharp_quality_post_pd_alt_max_p90_aspect_increase_ratio' in topo_cfg:
                    promotion_cfg['quality_post_pd_alt_max_p90_aspect_increase_ratio'] = topo_cfg[
                        'small_sharp_quality_post_pd_alt_max_p90_aspect_increase_ratio'
                    ]
            promote_alt = (
                closed_input
                and selection_mode == 'quality'
                and promotion_enabled
                and alt_ok_final == topo_ok_final
                and _post_pd_alt_promotion_accepts(
                    topo_rep_final,
                    main_qm_final,
                    len(quadF),
                    alt_rep_final,
                    alt_qm_final,
                    len(alt_quadF),
                    promotion_cfg,
                )
            )
            if promote_alt:
                root, ext = os.path.splitext(args.output)
                dense_output = f"{root}_qualitydense{ext or '.obj'}"
                write_mesh(dense_output, final_V, quadF)
                print(f"  [topo] preserved previous dense-quality output → {dense_output}")
                final_V = alt_final_V
                quadF = alt_quadF
                topo_ok_final = alt_ok_final
                topo_rep_final = alt_rep_final
                main_qm_final = alt_qm_final
                write_mesh(args.output, final_V, quadF)
                print(
                    "  [topo] promoted topology-first alternative to main output: "
                    f"boundary_edges={int(topo_rep_final.get('boundary_edges', 0))}"
                )
                if (
                    export_tri_preview
                    and export_ref_points is not None
                    and export_ref_normals is not None
                    and len(export_ref_normals) == len(export_ref_points)
                ):
                    tri_V, tri_F = triangulate_quads_for_reference_preview(
                        final_V, quadF,
                        ref_points=export_ref_points,
                        ref_normals=export_ref_normals,
                    )
                    tri_V, tri_F, tri_flipped = align_triangle_faces_to_reference_normals(
                        tri_V, tri_F,
                        ref_points=export_ref_points,
                        ref_normals=export_ref_normals,
                    )
                    root, ext = os.path.splitext(args.output)
                    tri_output = f"{root}_preview_tri{ext or '.obj'}"
                    write_mesh(tri_output, tri_V, tri_F)
                    print(f"  [topo] refreshed main triangle preview → {tri_output}")
                    if tri_flipped > 0:
                        print(f"  [topo] preview reference-normal alignment flipped {tri_flipped} triangles.")

            root, ext = os.path.splitext(args.output)
            alt_output = f"{root}_topology{ext or '.obj'}"
            write_mesh(alt_output, alt_final_V, alt_quadF)
            print(
                "  [topo] exported topology-first alternative → "
                f"{alt_output} (attempt {topo_ai}/{len(profiles)})"
            )
            if (
                export_tri_preview
                and export_ref_points is not None
                and export_ref_normals is not None
                and len(export_ref_normals) == len(export_ref_points)
            ):
                tri_alt_V, tri_alt_F = triangulate_quads_for_reference_preview(
                    alt_final_V, alt_quadF,
                    ref_points=export_ref_points,
                    ref_normals=export_ref_normals,
                )
                tri_alt_V, tri_alt_F, tri_alt_flipped = align_triangle_faces_to_reference_normals(
                    tri_alt_V, tri_alt_F,
                    ref_points=export_ref_points,
                    ref_normals=export_ref_normals,
                )
                tri_alt_output = f"{root}_topology_preview_tri{ext or '.obj'}"
                write_mesh(tri_alt_output, tri_alt_V, tri_alt_F)
                print(f"  [topo] exported topology-first triangle preview → {tri_alt_output}")
                if tri_alt_flipped > 0:
                    print(
                        f"  [topo] topology-first preview reference-normal alignment "
                        f"flipped {tri_alt_flipped} triangles."
                    )

    if args.vis and ps is not None:
        register_mesh("final_quad", final_V, quadF, color=(0.8, 0.2, 0.2))
        ps.show()


if __name__ == '__main__':
    main()
