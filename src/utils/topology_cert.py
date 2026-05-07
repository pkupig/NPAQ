"""
Topology certification utilities for extracted quad meshes.

The goal is fail-fast behaviour: if the extracted topology violates basic
manifoldness/integrity checks, callers should reject the mesh instead of
silently writing invalid output.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Dict, Tuple

import numpy as np


def orient_quad_faces_consistently(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    ref_points: np.ndarray | None = None,
    ref_normals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Propagate a consistent quad orientation over the dual graph.

    Adjacent quads sharing an edge should traverse that edge in opposite
    directions. This fixes local extraction-stage winding inconsistencies before
    topology checks / PD.
    """
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64).copy()
    if F.ndim != 2 or F.shape[0] == 0:
        return V.copy(), F.copy(), 0

    edge_to_faces: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for fi, face in enumerate(F):
        m = len(face)
        for i in range(m):
            a = int(face[i])
            b = int(face[(i + 1) % m])
            key = (a, b) if a < b else (b, a)
            edge_to_faces[key].append((fi, a, b))

    flip = np.zeros(F.shape[0], dtype=bool)
    seen = np.zeros(F.shape[0], dtype=bool)
    for seed in range(F.shape[0]):
        if seen[seed]:
            continue
        seen[seed] = True
        dq: deque[int] = deque([seed])
        while dq:
            fi = dq.popleft()
            face = F[fi]
            m = len(face)
            for i in range(m):
                a = int(face[i]); b = int(face[(i + 1) % m])
                key = (a, b) if a < b else (b, a)
                for fj, ea, eb in edge_to_faces.get(key, []):
                    if fj == fi:
                        continue
                    same_dir = (a == ea and b == eb)
                    need_flip = same_dir
                    if not seen[fj]:
                        flip[fj] = flip[fi] ^ need_flip
                        seen[fj] = True
                        dq.append(fj)

    flipped = 0
    for fi in np.where(flip)[0]:
        F[fi] = F[fi, [0, 3, 2, 1]]
        flipped += 1

    if ref_points is not None and ref_normals is not None and len(F) > 0:
        centers = V[F].mean(axis=1)
        d1 = V[F[:, 2]] - V[F[:, 0]]
        d2 = V[F[:, 3]] - V[F[:, 1]]
        qn = np.cross(d1, d2)
        qn /= np.linalg.norm(qn, axis=1, keepdims=True) + 1e-12
        from scipy.spatial import KDTree
        tree = KDTree(np.asarray(ref_points, dtype=np.float64))
        _, idx = tree.query(centers)
        mean_dot = float((qn * np.asarray(ref_normals, dtype=np.float64)[idx]).sum(axis=1).mean())
        if mean_dot < 0.0:
            F = F[:, [0, 3, 2, 1]]

    return V.copy(), F, int(flipped)


def finalize_quad_orientation_with_reference(
    vertices: np.ndarray,
    faces: np.ndarray,
    ref_points: np.ndarray,
    ref_normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Final export-time orientation cleanup:

    1. propagate a globally consistent winding,
    2. if the whole component is globally inverted, flip it once,
    3. locally correct the remaining disagreeing quads against nearest
       reference normals.
    """
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64).copy()
    if F.ndim != 2 or F.shape[0] == 0:
        return V.copy(), F.copy(), 0

    Vc, Fc, n_consistent = orient_quad_faces_consistently(
        V, F, ref_points=ref_points, ref_normals=ref_normals
    )

    ref_points = np.asarray(ref_points, dtype=np.float64)
    ref_normals = np.asarray(ref_normals, dtype=np.float64)
    if ref_points.ndim != 2 or ref_normals.shape != ref_points.shape or len(ref_points) == 0:
        return Vc.copy(), Fc.copy(), int(n_consistent)

    centers = Vc[Fc].mean(axis=1)
    d1 = Vc[Fc[:, 2]] - Vc[Fc[:, 0]]
    d2 = Vc[Fc[:, 3]] - Vc[Fc[:, 1]]
    qn = np.cross(d1, d2)
    qn /= np.linalg.norm(qn, axis=1, keepdims=True) + 1e-12

    from scipy.spatial import KDTree
    idx = KDTree(ref_points).query(centers)[1]
    dot = (qn * ref_normals[idx]).sum(axis=1)
    if float(dot.mean()) < 0.0:
        Fc = Fc[:, [0, 3, 2, 1]]
        dot = -dot

    flip_mask = dot < 0.0
    n_local = int(np.count_nonzero(flip_mask))
    if n_local > 0:
        Fc[flip_mask] = Fc[flip_mask][:, [0, 3, 2, 1]]
    return Vc.copy(), Fc.copy(), int(n_consistent + n_local)


def align_quad_faces_outward_from_centroid(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    For roughly closed shapes, ensure the dominant winding points outward
    relative to the mesh centroid.

    This is a coarse global fix for cases where all quads are consistently
    wound but the whole surface is inverted.
    """
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64).copy()
    if F.ndim != 2 or F.shape[0] == 0 or F.shape[1] != 4:
        return V.copy(), F.copy(), 0

    center = V.mean(axis=0)
    d1 = V[F[:, 2]] - V[F[:, 0]]
    d2 = V[F[:, 3]] - V[F[:, 1]]
    qn = np.cross(d1, d2)
    qn /= np.linalg.norm(qn, axis=1, keepdims=True) + 1e-12
    fc = V[F].mean(axis=1)
    orient = np.einsum('ij,ij->i', qn, fc - center)
    n_inward = int(np.count_nonzero(orient < 0.0))
    n_outward = int(np.count_nonzero(orient > 0.0))
    if n_inward > n_outward:
        F = F[:, [0, 3, 2, 1]]
        return V.copy(), F.copy(), int(F.shape[0])
    return V.copy(), F.copy(), 0


def triangulate_quads_for_reference_preview(
    vertices: np.ndarray,
    faces: np.ndarray,
    ref_points: np.ndarray,
    ref_normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Triangulate quads using the diagonal that best agrees with nearby reference normals.

    This is intended for viewer/debug export only. Some viewers silently
    triangulate non-planar quads with an arbitrary diagonal, which can make a
    visually correct quad mesh appear to contain flipped triangles. Here we
    choose the split per quad that maximizes triangle-normal agreement against
    nearest reference normals.
    """
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    if F.ndim != 2 or F.shape[0] == 0 or F.shape[1] != 4:
        return V.copy(), F.copy()

    ref_points = np.asarray(ref_points, dtype=np.float64)
    ref_normals = np.asarray(ref_normals, dtype=np.float64)
    if ref_points.ndim != 2 or ref_normals.shape != ref_points.shape or len(ref_points) == 0:
        tri = np.vstack([F[:, [0, 1, 2]], F[:, [0, 2, 3]]])
        return V.copy(), tri.astype(np.int64)

    from scipy.spatial import KDTree

    tree = KDTree(ref_points)
    tris: list[list[int]] = []
    for face in F:
        a, b, c, d = [int(i) for i in face]
        split0 = np.asarray([[a, b, c], [a, c, d]], dtype=np.int64)
        split1 = np.asarray([[a, b, d], [b, c, d]], dtype=np.int64)

        def _score(tri_faces: np.ndarray) -> tuple[float, float]:
            centers = V[tri_faces].mean(axis=1)
            tn = np.cross(
                V[tri_faces[:, 1]] - V[tri_faces[:, 0]],
                V[tri_faces[:, 2]] - V[tri_faces[:, 0]],
            )
            tn /= np.linalg.norm(tn, axis=1, keepdims=True) + 1e-12
            idx = tree.query(centers)[1]
            dots = (tn * ref_normals[idx]).sum(axis=1)
            return float(dots.min()), float(dots.sum())

        score0 = _score(split0)
        score1 = _score(split1)
        chosen = split1 if score1 > score0 else split0
        tris.extend(chosen.tolist())
    return V.copy(), np.asarray(tris, dtype=np.int64)


def align_triangle_faces_to_reference_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
    ref_points: np.ndarray,
    ref_normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Preview-only helper: independently flip triangles whose normals disagree
    with nearest reference normals.

    Unlike quad export, this is only meant for viewer/debug triangle previews,
    where per-triangle orientation is acceptable and often avoids false-looking
    flips introduced by arbitrary triangulation of mildly non-planar quads.
    """
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64).copy()
    if F.ndim != 2 or F.shape[0] == 0 or F.shape[1] != 3:
        return V.copy(), F.copy(), 0

    ref_points = np.asarray(ref_points, dtype=np.float64)
    ref_normals = np.asarray(ref_normals, dtype=np.float64)
    if ref_points.ndim != 2 or ref_normals.shape != ref_points.shape or len(ref_points) == 0:
        return V.copy(), F.copy(), 0

    from scipy.spatial import KDTree

    centers = V[F].mean(axis=1)
    tn = np.cross(
        V[F[:, 1]] - V[F[:, 0]],
        V[F[:, 2]] - V[F[:, 0]],
    )
    tn /= np.linalg.norm(tn, axis=1, keepdims=True) + 1e-12
    idx = KDTree(ref_points).query(centers)[1]
    dot = (tn * ref_normals[idx]).sum(axis=1)
    flip_mask = dot < 0.0
    if np.any(flip_mask):
        F[flip_mask] = F[flip_mask][:, [0, 2, 1]]
    return V.copy(), F.copy(), int(np.count_nonzero(flip_mask))


def prune_stacked_parallel_quads(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    center_dist_ratio: float = 0.01,
    normal_dot_min: float = 0.95,
    max_iters: int = 2,
    max_candidates_per_face: int = 8,
    min_aspect_ratio: float = 4.0,
    min_area_ratio: float = 1.8,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Conservatively remove likely stacked duplicate-like quads.

    This is a geometric heuristic for post-extraction/post-PD cleanup:
    - nearly coincident quad centers
    - nearly parallel normals
    - low shared-vertex overlap
    For each suspicious pair, keep the geometrically stronger quad
    (larger area, better aspect, larger normal magnitude surrogate).
    """
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64).copy()
    stats = {
        "initial_pairs": 0,
        "initial_faces": int(F.shape[0]) if F.ndim == 2 else 0,
        "removed_faces": 0,
        "remaining_pairs": 0,
    }
    if F.ndim != 2 or F.shape[0] == 0:
        return V.copy(), F.copy(), stats

    from scipy.spatial import cKDTree

    diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    radius = float(center_dist_ratio) * max(diag, 1e-8)
    if radius <= 0.0:
        return V.copy(), F.copy(), stats

    def _face_metrics(face_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pts = V[face_arr]
        centers = pts.mean(axis=1)
        d1 = pts[:, 2] - pts[:, 0]
        d2 = pts[:, 3] - pts[:, 1]
        normals = np.cross(d1, d2)
        normal_norm = np.linalg.norm(normals, axis=1)
        normals_unit = normals / (normal_norm[:, None] + 1e-12)
        e01 = np.linalg.norm(pts[:, 1] - pts[:, 0], axis=1)
        e12 = np.linalg.norm(pts[:, 2] - pts[:, 1], axis=1)
        e23 = np.linalg.norm(pts[:, 3] - pts[:, 2], axis=1)
        e30 = np.linalg.norm(pts[:, 0] - pts[:, 3], axis=1)
        edge_min = np.minimum.reduce([e01, e12, e23, e30]) + 1e-12
        edge_max = np.maximum.reduce([e01, e12, e23, e30])
        aspect = edge_max / edge_min
        tri0 = 0.5 * np.linalg.norm(np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0]), axis=1)
        tri1 = 0.5 * np.linalg.norm(np.cross(pts[:, 3] - pts[:, 0], pts[:, 2] - pts[:, 0]), axis=1)
        area = tri0 + tri1
        return centers, normals_unit, normal_norm, area, aspect

    def _count_pairs(face_arr: np.ndarray) -> int:
        if len(face_arr) <= 1:
            return 0
        centers, normals_unit, _, area, aspect = _face_metrics(face_arr)
        tree = cKDTree(centers)
        total = 0
        for i, neigh in enumerate(tree.query_ball_point(centers, radius)):
            kept = 0
            s_i = set(int(v) for v in face_arr[i].tolist())
            for j in neigh:
                if j <= i:
                    continue
                if kept >= max_candidates_per_face:
                    break
                if abs(float(np.dot(normals_unit[i], normals_unit[j]))) < normal_dot_min:
                    continue
                if len(s_i.intersection(int(v) for v in face_arr[j].tolist())) >= 2:
                    continue
                area_lo = min(float(area[i]), float(area[j])) + 1e-12
                area_hi = max(float(area[i]), float(area[j])) + 1e-12
                aspect_hi = max(float(aspect[i]), float(aspect[j]))
                if aspect_hi < min_aspect_ratio and (area_hi / area_lo) < min_area_ratio:
                    continue
                total += 1
                kept += 1
        return int(total)

    stats["initial_pairs"] = _count_pairs(F)
    if stats["initial_pairs"] == 0:
        return V.copy(), F.copy(), stats

    if int(max_iters) <= 0:
        stats["remaining_pairs"] = int(stats["initial_pairs"])
        return V.copy(), F.copy(), stats

    for _ in range(int(max_iters)):
        centers, normals_unit, normal_norm, area, aspect = _face_metrics(F)
        tree = cKDTree(centers)
        remove: set[int] = set()
        for i, neigh in enumerate(tree.query_ball_point(centers, radius)):
            if i in remove:
                continue
            s_i = set(int(v) for v in F[i].tolist())
            kept = 0
            for j in neigh:
                if j <= i or j in remove:
                    continue
                if kept >= max_candidates_per_face:
                    break
                if abs(float(np.dot(normals_unit[i], normals_unit[j]))) < normal_dot_min:
                    continue
                if len(s_i.intersection(int(v) for v in F[j].tolist())) >= 2:
                    continue
                area_lo = min(float(area[i]), float(area[j])) + 1e-12
                area_hi = max(float(area[i]), float(area[j])) + 1e-12
                aspect_hi = max(float(aspect[i]), float(aspect[j]))
                if aspect_hi < min_aspect_ratio and (area_hi / area_lo) < min_area_ratio:
                    continue
                # Prefer keeping the larger / less distorted quad.
                score_i = (float(area[i]), -float(aspect[i]), float(normal_norm[i]))
                score_j = (float(area[j]), -float(aspect[j]), float(normal_norm[j]))
                if score_i >= score_j:
                    remove.add(j)
                else:
                    remove.add(i)
                    break
                kept += 1
        if not remove:
            break
        keep_mask = np.ones(len(F), dtype=bool)
        keep_mask[list(remove)] = False
        F = F[keep_mask]
        stats["removed_faces"] += int(len(remove))
        if len(F) == 0:
            break

    stats["remaining_pairs"] = _count_pairs(F) if len(F) > 0 else 0
    return V.copy(), F.copy(), stats


def _extract_boundary_loops(faces: np.ndarray) -> list[list[int]]:
    F = np.asarray(faces, dtype=np.int64)
    if F.ndim != 2 or F.shape[0] == 0:
        return []

    edge_counts = Counter()
    for face in F:
        m = len(face)
        for i in range(m):
            a, b = int(face[i]), int(face[(i + 1) % m])
            e = (a, b) if a < b else (b, a)
            edge_counts[e] += 1

    boundary_edges = [e for e, c in edge_counts.items() if c == 1]
    if not boundary_edges:
        return []

    adj = defaultdict(list)
    for a, b in boundary_edges:
        adj[a].append(b)
        adj[b].append(a)

    seen: set[tuple[int, int]] = set()
    loops: list[list[int]] = []
    for a, b in boundary_edges:
        if (a, b) in seen or (b, a) in seen:
            continue
        if len(adj[a]) != 2 or len(adj[b]) != 2:
            continue
        start = a
        prev = a
        cur = b
        loop = [a, b]
        seen.add((a, b))
        seen.add((b, a))
        while cur != start:
            nbrs = [x for x in adj[cur] if x != prev]
            if not nbrs:
                break
            nxt = nbrs[0]
            prev, cur = cur, nxt
            if cur != start and cur in loop:
                break
            if (prev, cur) in seen:
                break
            seen.add((prev, cur))
            seen.add((cur, prev))
            if cur != start:
                loop.append(cur)
        if cur == start and len(loop) >= 3:
            loops.append(loop)
    return loops


def _boundary_loop_lengths(faces: np.ndarray) -> list[int]:
    return [len(loop) for loop in _extract_boundary_loops(faces)]


def _boundary_graph_stats(faces: np.ndarray) -> Dict[str, int]:
    F = np.asarray(faces, dtype=np.int64)
    if F.ndim != 2 or F.shape[0] == 0:
        return {
            "boundary_vertices": 0,
            "boundary_components": 0,
            "boundary_loops": 0,
            "boundary_chains": 0,
            "boundary_irregular_vertices": 0,
        }

    edge_counts = Counter()
    for face in F:
        m = len(face)
        for i in range(m):
            a, b = int(face[i]), int(face[(i + 1) % m])
            e = (a, b) if a < b else (b, a)
            edge_counts[e] += 1

    boundary_edges = [e for e, c in edge_counts.items() if c == 1]
    if not boundary_edges:
        return {
            "boundary_vertices": 0,
            "boundary_components": 0,
            "boundary_loops": 0,
            "boundary_chains": 0,
            "boundary_irregular_vertices": 0,
        }

    adj: dict[int, set[int]] = defaultdict(set)
    for a, b in boundary_edges:
        adj[a].add(b)
        adj[b].add(a)

    seen: set[int] = set()
    components = 0
    loops = 0
    chains = 0
    for seed in adj:
        if seed in seen:
            continue
        components += 1
        stack = [seed]
        comp = []
        seen.add(seed)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adj[cur]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        degs = [len(adj[v]) for v in comp]
        if all(d == 2 for d in degs):
            loops += 1
        else:
            chains += 1

    return {
        "boundary_vertices": int(len(adj)),
        "boundary_components": int(components),
        "boundary_loops": int(loops),
        "boundary_chains": int(chains),
        "boundary_irregular_vertices": int(sum(len(nbs) != 2 for nbs in adj.values())),
    }


def _vertex_fan_stats(faces: np.ndarray, num_vertices: int) -> Dict[str, int]:
    """
    Detect vertices whose incident faces split into multiple disconnected fans.

    Edge incidence catches edge-nonmanifold cases, but a mesh can still be
    non-manifold at a vertex when two otherwise valid face fans touch only at
    that vertex.  For each vertex, connect incident faces that share an edge
    containing the vertex; more than one connected component means a non-
    manifold vertex fan.
    """
    incident_faces: dict[int, list[int]] = defaultdict(list)
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)

    for fi, face in enumerate(faces):
        f = [int(i) for i in face]
        for v in set(f):
            if 0 <= v < num_vertices:
                incident_faces[v].append(fi)
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            if 0 <= a < num_vertices and 0 <= b < num_vertices:
                edge_to_faces[(a, b) if a < b else (b, a)].append(fi)

    nonmanifold_vertices = 0
    max_vertex_fan_components = 0

    for v, faces_at_v in incident_faces.items():
        if len(faces_at_v) <= 1:
            max_vertex_fan_components = max(max_vertex_fan_components, len(faces_at_v))
            continue

        adj: dict[int, set[int]] = {fi: set() for fi in faces_at_v}
        for (a, b), fis in edge_to_faces.items():
            if v not in (a, b):
                continue
            for i, fa in enumerate(fis):
                if fa not in adj:
                    continue
                for fb in fis[i + 1:]:
                    if fb in adj:
                        adj[fa].add(fb)
                        adj[fb].add(fa)

        seen = set()
        components = 0
        for seed in faces_at_v:
            if seed in seen:
                continue
            components += 1
            stack = [seed]
            seen.add(seed)
            while stack:
                cur = stack.pop()
                for nb in adj[cur]:
                    if nb not in seen:
                        seen.add(nb)
                        stack.append(nb)

        max_vertex_fan_components = max(max_vertex_fan_components, components)
        if components > 1:
            nonmanifold_vertices += 1

    return {
        "nonmanifold_vertices": int(nonmanifold_vertices),
        "max_vertex_fan_components": int(max_vertex_fan_components),
    }


def certify_quad_topology(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    allow_boundary: bool = True,
    require_all_quads: bool = True,
) -> Tuple[bool, Dict[str, int]]:
    """
    Certify topological integrity of a polygon/quad mesh.

    Checks:
    - Optional all-quads requirement.
    - Face indices are inside the vertex array.
    - No degenerate faces (repeated vertex indices).
    - No duplicate faces (up to cyclic order / reversal).
    - No high-multiplicity edges (>2 incident faces).
    - No multi-fan non-manifold vertices.
    - Boundary allowance is configurable (edge incidence == 1).

    Returns:
        is_valid, report
    """
    V = np.asarray(vertices)
    F = np.asarray(faces)

    if F.ndim != 2 or F.shape[0] == 0:
        report = {
            "num_vertices": int(V.shape[0]) if V.ndim == 2 else 0,
            "num_faces": 0,
            "non_quad_faces": 0,
            "degenerate_faces": 0,
            "duplicate_faces": 0,
            "boundary_edges": 0,
            "boundary_loops": 0,
            "high_multiplicity_edges": 0,
            "nonmanifold_edges": 0,
            "nonmanifold_vertices": 0,
            "max_vertex_fan_components": 0,
            "invalid_vertex_indices": 0,
        }
        return False, report

    non_quad_faces = int(np.sum([len(f) != 4 for f in F]))
    invalid_vertex_indices = int(np.sum((F < 0) | (F >= V.shape[0]))) if V.ndim == 2 else int(F.size)

    degenerate_faces = 0
    face_keys = Counter()
    edge_counts = Counter()

    for face in F:
        f = [int(i) for i in face]
        if len(set(f)) < len(f):
            degenerate_faces += 1

        # Canonical face key (rotation + reversal invariant)
        m = len(f)
        rots = [tuple(f[i:] + f[:i]) for i in range(m)]
        rf = list(reversed(f))
        rots += [tuple(rf[i:] + rf[:i]) for i in range(m)]
        face_keys[min(rots)] += 1

        for i in range(m):
            a, b = f[i], f[(i + 1) % m]
            if a > b:
                a, b = b, a
            edge_counts[(a, b)] += 1

    duplicate_faces = int(sum(c - 1 for c in face_keys.values() if c > 1))
    boundary_edges = int(sum(c == 1 for c in edge_counts.values()))
    boundary_stats = _boundary_graph_stats(F)
    vertex_stats = _vertex_fan_stats(F, int(V.shape[0]) if V.ndim == 2 else 0)
    high_mult_edges = int(sum(c > 2 for c in edge_counts.values()))
    # Non-manifold edges are edges shared by more than two faces.
    # Boundary edges (incidence==1) are not non-manifold by themselves and are
    # controlled separately via allow_boundary.
    nonmanifold_edges = high_mult_edges

    valid = True
    if invalid_vertex_indices > 0:
        valid = False
    if require_all_quads and non_quad_faces > 0:
        valid = False
    if degenerate_faces > 0:
        valid = False
    if duplicate_faces > 0:
        valid = False
    if high_mult_edges > 0:
        valid = False
    if int(vertex_stats["nonmanifold_vertices"]) > 0:
        valid = False
    if (not allow_boundary) and boundary_edges > 0:
        valid = False

    report = {
        "num_vertices": int(V.shape[0]) if V.ndim == 2 else 0,
        "num_faces": int(F.shape[0]),
        "non_quad_faces": non_quad_faces,
        "degenerate_faces": degenerate_faces,
        "duplicate_faces": duplicate_faces,
        "boundary_edges": boundary_edges,
        "boundary_loops": int(boundary_stats["boundary_loops"]),
        "boundary_chains": int(boundary_stats["boundary_chains"]),
        "boundary_components": int(boundary_stats["boundary_components"]),
        "boundary_vertices": int(boundary_stats["boundary_vertices"]),
        "boundary_irregular_vertices": int(boundary_stats["boundary_irregular_vertices"]),
        "high_multiplicity_edges": high_mult_edges,
        "nonmanifold_edges": nonmanifold_edges,
        "nonmanifold_vertices": int(vertex_stats["nonmanifold_vertices"]),
        "max_vertex_fan_components": int(vertex_stats["max_vertex_fan_components"]),
        "invalid_vertex_indices": invalid_vertex_indices,
    }
    return valid, report


def format_topology_report(report: Dict[str, int]) -> str:
    return (
        "faces={num_faces}, non_quad={non_quad_faces}, degenerate={degenerate_faces}, "
        "duplicate={duplicate_faces}, boundary_edges={boundary_edges}, boundary_loops={boundary_loops}, "
        "boundary_chains={boundary_chains}, irregular_boundary_vertices={boundary_irregular_vertices}, "
        "high_mult_edges={high_multiplicity_edges}, nonmanifold_edges={nonmanifold_edges}, "
        "nonmanifold_vertices={nonmanifold_vertices}, invalid_vertex_indices={invalid_vertex_indices}"
    ).format(**report)


def prune_to_largest_boundary_light_component(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    min_keep_faces: int = 0,
    min_keep_ratio: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Keep the connected quad-face component with the best boundary-to-area ratio.

    This removes detached or weakly connected fragments that often account for a
    disproportionate number of open boundary edges after extraction.
    """
    V = np.asarray(vertices)
    F = np.asarray(faces, dtype=np.int64)
    if F.ndim != 2 or F.shape[0] == 0:
        return V.copy(), F.copy()

    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for fi, face in enumerate(F):
        m = len(face)
        for i in range(m):
            a, b = int(face[i]), int(face[(i + 1) % m])
            e = (a, b) if a < b else (b, a)
            edge_to_faces.setdefault(e, []).append(fi)

    face_adj = [[] for _ in range(len(F))]
    for incident in edge_to_faces.values():
        if len(incident) < 2:
            continue
        for i in incident:
            for j in incident:
                if i != j:
                    face_adj[i].append(j)

    comp_id = -np.ones(len(F), dtype=np.int64)
    comps: list[list[int]] = []
    cid = 0
    for seed in range(len(F)):
        if comp_id[seed] != -1:
            continue
        stack = [seed]
        comp = []
        comp_id[seed] = cid
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in face_adj[cur]:
                if comp_id[nb] == -1:
                    comp_id[nb] = cid
                    stack.append(nb)
        comps.append(comp)
        cid += 1

    if len(comps) <= 1:
        return V.copy(), F.copy()

    total_faces = int(len(F))
    min_keep = max(int(min_keep_faces), int(np.ceil(float(min_keep_ratio) * total_faces)))

    best = None
    for comp in comps:
        comp_set = set(comp)
        boundary_edges = 0
        for incident in edge_to_faces.values():
            inside = sum(fi in comp_set for fi in incident)
            if inside == 1:
                boundary_edges += 1
        face_count = len(comp)
        score = (
            0 if face_count >= min_keep else 1,
            boundary_edges / max(face_count, 1),
            -face_count,
        )
        if best is None or score < best[0]:
            best = (score, comp)

    keep_faces = F[np.asarray(best[1], dtype=np.int64)]
    used = np.unique(keep_faces.reshape(-1))
    remap = {int(v): i for i, v in enumerate(used.tolist())}
    out_faces = np.vectorize(lambda x: remap[int(x)], otypes=[np.int64])(keep_faces)
    out_vertices = V[used]
    return out_vertices, out_faces


def fill_boundary_quad_holes(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    max_loop_len: int = 4,
    max_iters: int = 4,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Fill simple boundary loops of length 4 with a quad face.

    This targets the most common extraction miss: a single absent cell in the
    integer grid, which appears as a 4-edge boundary cycle.
    """
    V = np.asarray(vertices)
    F = np.asarray(faces, dtype=np.int64).copy()
    filled = 0

    def _canon(face: list[int]) -> tuple[int, ...]:
        m = len(face)
        rots = [tuple(face[i:] + face[:i]) for i in range(m)]
        rf = list(reversed(face))
        rots += [tuple(rf[i:] + rf[:i]) for i in range(m)]
        return min(rots)

    for _ in range(max(1, int(max_iters))):
        edge_counts = Counter()
        for face in F:
            m = len(face)
            for i in range(m):
                a, b = int(face[i]), int(face[(i + 1) % m])
                e = (a, b) if a < b else (b, a)
                edge_counts[e] += 1
        boundary_edges = [e for e, c in edge_counts.items() if c == 1]
        if not boundary_edges:
            break

        adj: dict[int, list[int]] = {}
        for a, b in boundary_edges:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)

        existing = {_canon([int(v) for v in face]) for face in F}
        visited_edges: set[tuple[int, int]] = set()
        new_faces: list[list[int]] = []

        for a, b in boundary_edges:
            e0 = (a, b) if a < b else (b, a)
            if e0 in visited_edges:
                continue
            loop = [a, b]
            prev, cur = a, b
            ok = True
            while True:
                nbrs = adj.get(cur, [])
                if len(nbrs) != 2:
                    ok = False
                    break
                nxt = nbrs[0] if nbrs[1] == prev else nbrs[1]
                edge_key = (cur, nxt) if cur < nxt else (nxt, cur)
                visited_edges.add((prev, cur) if prev < cur else (cur, prev))
                if nxt == loop[0]:
                    break
                if nxt in loop or len(loop) >= max_loop_len:
                    ok = False
                    break
                loop.append(nxt)
                prev, cur = cur, nxt

            if not ok:
                continue
            if len(loop) == 4:
                key = _canon(loop)
                if key not in existing:
                    new_faces.append(loop)
                    existing.add(key)

        if not new_faces:
            break

        F = np.vstack([F, np.asarray(new_faces, dtype=np.int64)])
        filled += len(new_faces)

    return V.copy(), F, int(filled)


def stitch_boundary_loop_pairs(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    max_loop_len: int = 24,
    max_centroid_dist_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Bridge nearby boundary-loop pairs of equal length with a quad strip.

    This targets extraction seams that leave two short open rings facing each
    other. The bridge uses only existing boundary vertices, so it does not
    create T-junctions along existing edges.
    """
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64).copy()
    if F.ndim != 2 or F.shape[0] == 0:
        return V.copy(), F.copy(), 0

    loops = _extract_boundary_loops(F)
    if len(loops) < 2:
        return V.copy(), F.copy(), 0

    diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    max_centroid_dist = float(max_centroid_dist_ratio) * max(diag, 1e-8)
    centroids = [V[np.asarray(loop, dtype=np.int64)].mean(axis=0) for loop in loops]

    candidates: list[tuple[float, int, int]] = []
    for i in range(len(loops)):
        li = loops[i]
        if len(li) > max_loop_len:
            continue
        for j in range(i + 1, len(loops)):
            lj = loops[j]
            if len(li) != len(lj) or len(lj) > max_loop_len:
                continue
            dist = float(np.linalg.norm(centroids[i] - centroids[j]))
            if dist <= max_centroid_dist:
                candidates.append((dist, i, j))

    if not candidates:
        return V.copy(), F.copy(), 0

    candidates.sort()
    used: set[int] = set()
    new_faces: list[list[int]] = []
    for _, i, j in candidates:
        if i in used or j in used:
            continue
        loop_a = [int(v) for v in loops[i]]
        loop_b = [int(v) for v in loops[j]]
        n = len(loop_a)

        pts_a = V[np.asarray(loop_a, dtype=np.int64)]
        pts_b = V[np.asarray(loop_b, dtype=np.int64)]
        best = None
        for reverse in (False, True):
            seq_b = list(reversed(loop_b)) if reverse else loop_b
            pts_seq_b = V[np.asarray(seq_b, dtype=np.int64)]
            for shift in range(n):
                rolled = np.roll(pts_seq_b, -shift, axis=0)
                score = float(np.sum(np.linalg.norm(pts_a - rolled, axis=1)))
                if best is None or score < best[0]:
                    best = (score, reverse, shift)

        _, reverse, shift = best
        seq_b = list(reversed(loop_b)) if reverse else loop_b
        seq_b = seq_b[shift:] + seq_b[:shift]

        for k in range(n):
            a0 = loop_a[k]
            a1 = loop_a[(k + 1) % n]
            b1 = seq_b[(k + 1) % n]
            b0 = seq_b[k]
            if len({a0, a1, b1, b0}) == 4:
                new_faces.append([a0, a1, b1, b0])
        used.add(i)
        used.add(j)

    if not new_faces:
        return V.copy(), F.copy(), 0

    F = np.vstack([F, np.asarray(new_faces, dtype=np.int64)])
    return V.copy(), F, int(len(new_faces))


def zipper_self_matched_boundary_loops(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    min_loop_len: int = 64,
    min_cyclic_gap_ratio: float = 0.1,
    max_pair_dist_ratio: float = 0.06,
    max_iters: int = 2,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Zipper long seam-like boundary loops by bridging reciprocally matched
    non-local neighbors along the same loop.

    This is intentionally conservative: it only bridges short, locally
    consistent matched segments to reduce obvious extraction seams.
    """
    from scipy.spatial import cKDTree

    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64).copy()
    if F.ndim != 2 or F.shape[0] == 0:
        return V.copy(), F.copy(), 0

    total_added = 0
    for _ in range(max(1, int(max_iters))):
        loops = _extract_boundary_loops(F)
        if not loops:
            break

        added_this_iter = 0
        existing = {tuple(sorted(int(v) for v in face)) for face in F}
        for loop in loops:
            if len(loop) < int(min_loop_len):
                continue

            pts = V[np.asarray(loop, dtype=np.int64)]
            bbox_diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
            max_pair_dist = float(max_pair_dist_ratio) * max(bbox_diag, 1e-8)
            min_gap = max(6, int(np.ceil(float(min_cyclic_gap_ratio) * len(loop))))

            tree = cKDTree(pts)
            dists, idxs = tree.query(pts, k=min(16, len(loop)))
            best: list[tuple[int, float] | None] = [None] * len(loop)
            for i in range(len(loop)):
                for d, j in zip(np.atleast_1d(dists[i])[1:], np.atleast_1d(idxs[i])[1:]):
                    cyc = min((int(j) - i) % len(loop), (i - int(j)) % len(loop))
                    if cyc >= min_gap and float(d) <= max_pair_dist:
                        best[i] = (int(j), float(d))
                        break

            reciprocal: list[tuple[int, int]] = []
            used: set[int] = set()
            for i, b in enumerate(best):
                if b is None:
                    continue
                j, _ = b
                bj = best[j]
                if bj is None or bj[0] != i or i in used or j in used:
                    continue
                used.add(i)
                used.add(j)
                if i < j:
                    reciprocal.append((i, j))
                else:
                    reciprocal.append((j, i))

            if len(reciprocal) < 2:
                continue
            reciprocal.sort()

            new_faces: list[list[int]] = []
            used_loop_vertices: set[int] = set()
            for (i0, j0), (i1, j1) in zip(reciprocal, reciprocal[1:]):
                if i1 != i0 + 1:
                    continue
                step = (j1 - j0) % len(loop)
                if step == len(loop) - 1:
                    a0 = int(loop[i0])
                    a1 = int(loop[i1])
                    b0 = int(loop[j0])
                    b1 = int(loop[j1])
                elif step == 1:
                    a0 = int(loop[i0])
                    a1 = int(loop[i1])
                    b0 = int(loop[j1])
                    b1 = int(loop[j0])
                else:
                    continue

                face = [a0, a1, b0, b1]
                if len(set(face)) < 4:
                    continue
                key = tuple(sorted(face))
                if key in existing:
                    continue
                if any(v in used_loop_vertices for v in face):
                    continue
                new_faces.append(face)
                existing.add(key)
                used_loop_vertices.update(face)

            if new_faces:
                F = np.vstack([F, np.asarray(new_faces, dtype=np.int64)])
                added_this_iter += len(new_faces)

        if added_this_iter == 0:
            break
        total_added += added_this_iter

    return V.copy(), F, int(total_added)


def cap_even_boundary_loops_with_center_quads(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    max_loop_len: int = 128,
    max_added_faces_per_loop: int = 1_000_000,
    preserve_largest_loops: int = 0,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Fill even-length boundary loops with a center-fan of quads.

    For a loop [v0, v1, ..., v(n-1)] with even n, create a center vertex c and
    quads:
        [v0, v1, v2, c], [v2, v3, v4, c], ..., [v(n-2), v(n-1), v0, c]

    This is intended as a last-resort hole cap for closed-surface reconstructions.
    """
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64).copy()
    if F.ndim != 2 or F.shape[0] == 0:
        return V.copy(), F.copy(), 0

    loops = _extract_boundary_loops(F)
    if not loops:
        return V.copy(), F.copy(), 0

    new_vertices: list[np.ndarray] = []
    new_faces: list[list[int]] = []
    keep_loops: set[tuple[int, ...]] = set()
    if preserve_largest_loops > 0:
        ranked = sorted(loops, key=len, reverse=True)[:int(preserve_largest_loops)]
        keep_loops = {tuple(int(v) for v in loop) for loop in ranked}

    for loop in loops:
        if tuple(int(v) for v in loop) in keep_loops:
            continue
        n = len(loop)
        if n < 6 or n > int(max_loop_len) or (n % 2) != 0:
            continue
        if (n // 2) > int(max_added_faces_per_loop):
            continue
        pts = V[np.asarray(loop, dtype=np.int64)]
        center = pts.mean(axis=0)
        cidx = int(len(V) + len(new_vertices))
        new_vertices.append(center)
        for i in range(0, n, 2):
            face = [
                int(loop[i]),
                int(loop[(i + 1) % n]),
                int(loop[(i + 2) % n]),
                cidx,
            ]
            if len(set(face)) == 4:
                new_faces.append(face)

    if not new_faces:
        return V.copy(), F.copy(), 0

    if new_vertices:
        V = np.vstack([V, np.asarray(new_vertices, dtype=np.float64)])
    F = np.vstack([F, np.asarray(new_faces, dtype=np.int64)])
    return V, F, int(len(new_faces))


def cap_even_boundary_loops_with_shrunk_rings(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    max_loop_len: int = 128,
    shrink: float = 0.35,
    preserve_largest_loops: int = 0,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Fill even boundary loops by recursively building inner quad rings.

    Compared with a single center-fan, this produces a more structured cap:
    an outer loop of length 2m is reduced to an inner loop of length m by
    pairing adjacent boundary vertices, placing a shrunken inner ring, and
    bridging outer-to-inner with m quads. The process repeats until a length-4
    inner loop remains, which is capped by one final quad.

    This is intentionally conservative and targets simple single-loop holes on
    roughly closed CAD-like meshes.
    """
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64).copy()
    if F.ndim != 2 or F.shape[0] == 0:
        return V.copy(), F.copy(), 0

    loops = _extract_boundary_loops(F)
    if not loops:
        return V.copy(), F.copy(), 0

    keep_loops: set[tuple[int, ...]] = set()
    if preserve_largest_loops > 0:
        ranked = sorted(loops, key=len, reverse=True)[:int(preserve_largest_loops)]
        keep_loops = {tuple(int(v) for v in loop) for loop in ranked}

    new_vertices: list[np.ndarray] = []
    new_faces: list[list[int]] = []

    def _vertex_pos(idx: int) -> np.ndarray:
        if idx < len(V):
            return V[idx]
        return np.asarray(new_vertices[idx - len(V)], dtype=np.float64)

    def _add_ring(loop: list[int]) -> None:
        n = len(loop)
        if n < 4 or (n % 2) != 0:
            return
        if n == 4:
            if len(set(loop)) == 4:
                new_faces.append([int(loop[0]), int(loop[1]), int(loop[2]), int(loop[3])])
            return

        pts = np.asarray([_vertex_pos(int(i)) for i in loop], dtype=np.float64)
        center = pts.mean(axis=0)
        m = n // 2
        inner: list[int] = []
        for i in range(m):
            pa = pts[2 * i]
            pb = pts[(2 * i + 1) % n]
            mid = 0.5 * (pa + pb)
            inner_pt = (1.0 - float(shrink)) * mid + float(shrink) * center
            vid = int(len(V) + len(new_vertices))
            new_vertices.append(inner_pt)
            inner.append(vid)

        for i in range(m):
            face = [
                int(loop[2 * i]),
                int(loop[(2 * i + 1) % n]),
                int(inner[(i + 1) % m]),
                int(inner[i]),
            ]
            if len(set(face)) == 4:
                new_faces.append(face)

        if m >= 4 and (m % 2) == 0:
            _add_ring(inner)
        elif m == 4:
            _add_ring(inner)

    for loop in loops:
        if tuple(int(v) for v in loop) in keep_loops:
            continue
        n = len(loop)
        if n < 4 or n > int(max_loop_len) or (n % 2) != 0:
            continue
        _add_ring([int(v) for v in loop])

    if not new_faces:
        return V.copy(), F.copy(), 0

    if new_vertices:
        V = np.vstack([V, np.asarray(new_vertices, dtype=np.float64)])
    F = np.vstack([F, np.asarray(new_faces, dtype=np.int64)])
    return V, F, int(len(new_faces))


def cap_even_boundary_loops_with_field_rings(
    vertices: np.ndarray,
    faces: np.ndarray,
    field_dirs: np.ndarray,
    *,
    max_loop_len: int = 128,
    shrink: float = 0.35,
    preserve_largest_loops: int = 0,
    field_align_min_strength: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Field-aligned variant of cap_even_boundary_loops_with_shrunk_rings.

    Reorders each boundary loop so the first edge is aligned with the average
    projected field direction, then applies the ring-based cap.
    """
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64).copy()
    if F.ndim != 2 or F.shape[0] == 0:
        return V.copy(), F.copy(), 0
    field = np.asarray(field_dirs, dtype=np.float64)
    if field.ndim != 2 or field.shape[0] != V.shape[0]:
        return cap_even_boundary_loops_with_shrunk_rings(
            V, F,
            max_loop_len=max_loop_len,
            shrink=shrink,
            preserve_largest_loops=preserve_largest_loops,
        )

    loops = _extract_boundary_loops(F)
    if not loops:
        return V.copy(), F.copy(), 0

    keep_loops: set[tuple[int, ...]] = set()
    if preserve_largest_loops > 0:
        ranked = sorted(loops, key=len, reverse=True)[:int(preserve_largest_loops)]
        keep_loops = {tuple(int(v) for v in loop) for loop in ranked}

    new_vertices: list[np.ndarray] = []
    new_faces: list[list[int]] = []

    def _vertex_pos(idx: int) -> np.ndarray:
        if idx < len(V):
            return V[idx]
        return np.asarray(new_vertices[idx - len(V)], dtype=np.float64)

    def _add_ring(loop: list[int]) -> None:
        n = len(loop)
        if n < 4 or (n % 2) != 0:
            return
        if n == 4:
            if len(set(loop)) == 4:
                new_faces.append([int(loop[0]), int(loop[1]), int(loop[2]), int(loop[3])])
            return

        pts = np.asarray([_vertex_pos(int(i)) for i in loop], dtype=np.float64)
        center = pts.mean(axis=0)
        m = n // 2
        inner: list[int] = []
        for i in range(m):
            pa = pts[2 * i]
            pb = pts[(2 * i + 1) % n]
            mid = 0.5 * (pa + pb)
            inner_pt = (1.0 - float(shrink)) * mid + float(shrink) * center
            vid = int(len(V) + len(new_vertices))
            new_vertices.append(inner_pt)
            inner.append(vid)

        for i in range(m):
            face = [
                int(loop[2 * i]),
                int(loop[(2 * i + 1) % n]),
                int(inner[(i + 1) % m]),
                int(inner[i]),
            ]
            if len(set(face)) == 4:
                new_faces.append(face)

        if m >= 4 and (m % 2) == 0:
            _add_ring(inner)
        elif m == 4:
            _add_ring(inner)

    for loop in loops:
        if tuple(int(v) for v in loop) in keep_loops:
            continue
        n = len(loop)
        if n < 4 or n > int(max_loop_len) or (n % 2) != 0:
            continue
        loop_ids = [int(v) for v in loop]
        loop_pts = V[np.asarray(loop_ids, dtype=np.int64)]
        ctr = loop_pts.mean(axis=0, keepdims=True)
        X = loop_pts - ctr
        _, _, vh = np.linalg.svd(X, full_matrices=False)
        e1 = vh[0]
        e2 = vh[1]
        d3 = field[np.asarray(loop_ids, dtype=np.int64)]
        d2 = np.stack([d3 @ e1, d3 @ e2], axis=1)
        u_axis = d2.mean(axis=0)
        u_norm = float(np.linalg.norm(u_axis))
        if u_norm >= float(field_align_min_strength):
            u_axis /= u_norm
            best_score = -1e9
            best_ids = loop_ids
            for reverse in (False, True):
                ids = list(reversed(loop_ids)) if reverse else list(loop_ids)
                pts2 = (loop_pts[::-1] if reverse else loop_pts)
                for r in range(n):
                    ids_r = ids[r:] + ids[:r]
                    pts_r = np.vstack([pts2[r:], pts2[:r]])
                    edge = pts_r[1] - pts_r[0]
                    edge2 = np.array([edge @ e1, edge @ e2], dtype=np.float64)
                    en = float(np.linalg.norm(edge2))
                    if en < 1e-12:
                        continue
                    edge2 /= en
                    score = float(np.dot(edge2, u_axis))
                    if score > best_score:
                        best_score = score
                        best_ids = ids_r
            loop_ids = best_ids
        _add_ring(loop_ids)

    if not new_faces:
        return V.copy(), F.copy(), 0

    if new_vertices:
        V = np.vstack([V, np.asarray(new_vertices, dtype=np.float64)])
    F = np.vstack([F, np.asarray(new_faces, dtype=np.int64)])
    return V, F, int(len(new_faces))


def cap_rectangular_boundary_loops_with_coons_patch(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    max_loop_len: int = 128,
    max_grid_side: int = 32,
    planar_ratio_max: float = 0.08,
    preserve_largest_loops: int = 0,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Fill rectangular-like single boundary loops with a structured all-quad Coons patch.

    This targets CAD-style missing face loops where the boundary is close to a
    four-sided cycle with roughly uniform samples per side. For a loop of length
    4m, we build an m x m quad grid whose boundary exactly reuses the existing
    loop vertices, and interpolate interior vertices by a Coons patch.
    """
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64).copy()
    if F.ndim != 2 or F.shape[0] == 0:
        return V.copy(), F.copy(), 0

    loops = _extract_boundary_loops(F)
    if not loops:
        return V.copy(), F.copy(), 0

    keep_loops: set[tuple[int, ...]] = set()
    if preserve_largest_loops > 0:
        ranked = sorted(loops, key=len, reverse=True)[:int(preserve_largest_loops)]
        keep_loops = {tuple(int(v) for v in loop) for loop in ranked}

    new_vertices: list[np.ndarray] = []
    new_faces: list[list[int]] = []

    def _loop_planarity_ratio(loop_pts: np.ndarray) -> float:
        ctr = loop_pts.mean(axis=0, keepdims=True)
        X = loop_pts - ctr
        _, s, _ = np.linalg.svd(X, full_matrices=False)
        if len(s) < 3:
            return 0.0
        return float(s[-1] / (s[0] + 1e-12))

    for loop in loops:
        if tuple(int(v) for v in loop) in keep_loops:
            continue
        n = len(loop)
        if n < 8 or n > int(max_loop_len) or (n % 4) != 0:
            continue
        m = n // 4
        if m < 2 or m > int(max_grid_side):
            continue

        loop_ids = [int(v) for v in loop]
        loop_pts = V[np.asarray(loop_ids, dtype=np.int64)]
        if _loop_planarity_ratio(loop_pts) > float(planar_ratio_max):
            continue

        # Loop ordering:
        #   c0 ---- c1
        #   |        |
        #   c3 ---- c2
        # with 4m samples around the hole.
        c0 = loop_ids[0]
        c1 = loop_ids[m]
        c2 = loop_ids[2 * m]
        c3 = loop_ids[3 * m]
        bottom = loop_ids[0 : m + 1]                    # c0 -> c1
        right = loop_ids[m : 2 * m + 1]                 # c1 -> c2
        top = list(reversed(loop_ids[2 * m : 3 * m + 1]))   # c3 -> c2
        left_raw = loop_ids[3 * m :] + [loop_ids[0]]        # c3 -> c0
        left = list(reversed(left_raw))                      # c0 -> c3

        grid = np.full((m + 1, m + 1), -1, dtype=np.int64)
        for i in range(m + 1):
            grid[i, 0] = int(bottom[i])
            grid[i, m] = int(top[i])
        for j in range(m + 1):
            grid[0, j] = int(left[j])
            grid[m, j] = int(right[j])

        p00 = V[c0]
        p10 = V[c1]
        p11 = V[c2]
        p01 = V[c3]

        for i in range(1, m):
            u = float(i) / float(m)
            bu = V[bottom[i]]
            tu = V[top[i]]
            for j in range(1, m):
                v = float(j) / float(m)
                lv = V[left[j]]
                rv = V[right[j]]
                bilinear = (
                    (1.0 - u) * (1.0 - v) * p00
                    + u * (1.0 - v) * p10
                    + u * v * p11
                    + (1.0 - u) * v * p01
                )
                p = (
                    (1.0 - v) * bu + v * tu
                    + (1.0 - u) * lv + u * rv
                    - bilinear
                )
                vid = int(len(V) + len(new_vertices))
                new_vertices.append(np.asarray(p, dtype=np.float64))
                grid[i, j] = vid

        for i in range(m):
            for j in range(m):
                face = [
                    int(grid[i, j]),
                    int(grid[i + 1, j]),
                    int(grid[i + 1, j + 1]),
                    int(grid[i, j + 1]),
                ]
                if len(set(face)) == 4:
                    new_faces.append(face)

    if not new_faces:
        return V.copy(), F.copy(), 0

    if new_vertices:
        V = np.vstack([V, np.asarray(new_vertices, dtype=np.float64)])
    F = np.vstack([F, np.asarray(new_faces, dtype=np.int64)])
    return V, F, int(len(new_faces))


def cap_rectangular_boundary_loops_with_field(
    vertices: np.ndarray,
    faces: np.ndarray,
    field_dirs: np.ndarray,
    *,
    max_loop_len: int = 128,
    max_grid_side: int = 32,
    planar_ratio_max: float = 0.08,
    preserve_largest_loops: int = 0,
    field_align_min_strength: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Field-aligned rectangular boundary cap using a structured Coons patch.

    This is identical to cap_rectangular_boundary_loops_with_coons_patch, but
    it rotates the loop ordering so the bottom edge is aligned with the average
    field direction projected onto the loop plane.
    """
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64).copy()
    if F.ndim != 2 or F.shape[0] == 0:
        return V.copy(), F.copy(), 0
    field = np.asarray(field_dirs, dtype=np.float64)
    if field.ndim != 2 or field.shape[0] != V.shape[0]:
        return cap_rectangular_boundary_loops_with_coons_patch(
            V, F,
            max_loop_len=max_loop_len,
            max_grid_side=max_grid_side,
            planar_ratio_max=planar_ratio_max,
            preserve_largest_loops=preserve_largest_loops,
        )

    loops = _extract_boundary_loops(F)
    if not loops:
        return V.copy(), F.copy(), 0

    keep_loops: set[tuple[int, ...]] = set()
    if preserve_largest_loops > 0:
        ranked = sorted(loops, key=len, reverse=True)[:int(preserve_largest_loops)]
        keep_loops = {tuple(int(v) for v in loop) for loop in ranked}

    new_vertices: list[np.ndarray] = []
    new_faces: list[list[int]] = []

    def _loop_planarity_ratio(loop_pts: np.ndarray) -> float:
        ctr = loop_pts.mean(axis=0, keepdims=True)
        X = loop_pts - ctr
        _, s, _ = np.linalg.svd(X, full_matrices=False)
        if len(s) < 3:
            return 0.0
        return float(s[-1] / (s[0] + 1e-12))

    for loop in loops:
        if tuple(int(v) for v in loop) in keep_loops:
            continue
        n = len(loop)
        if n < 8 or n > int(max_loop_len) or (n % 4) != 0:
            continue
        m = n // 4
        if m < 2 or m > int(max_grid_side):
            continue

        loop_ids = [int(v) for v in loop]
        loop_pts = V[np.asarray(loop_ids, dtype=np.int64)]
        if _loop_planarity_ratio(loop_pts) > float(planar_ratio_max):
            continue

        ctr = loop_pts.mean(axis=0, keepdims=True)
        X = loop_pts - ctr
        _, _, vh = np.linalg.svd(X, full_matrices=False)
        e1 = vh[0]
        e2 = vh[1]
        loop_2d = np.stack([X @ e1, X @ e2], axis=1)

        d3 = field[np.asarray(loop_ids, dtype=np.int64)]
        d2 = np.stack([d3 @ e1, d3 @ e2], axis=1)
        u_axis = d2.mean(axis=0)
        u_norm = float(np.linalg.norm(u_axis))
        if u_norm < float(field_align_min_strength):
            best_ids = loop_ids
            best_2d = loop_2d
        else:
            u_axis /= u_norm
            best_score = -1e9
            best_ids = loop_ids
            best_2d = loop_2d
            for reverse in (False, True):
                ids = list(reversed(loop_ids)) if reverse else list(loop_ids)
                coords = loop_2d[::-1] if reverse else loop_2d
                for r in range(n):
                    ids_r = ids[r:] + ids[:r]
                    coords_r = np.vstack([coords[r:], coords[:r]])
                    bottom = coords_r[m] - coords_r[0]
                    bn = float(np.linalg.norm(bottom))
                    if bn < 1e-12:
                        continue
                    bottom /= bn
                    score = float(np.dot(bottom, u_axis))
                    if score > best_score:
                        best_score = score
                        best_ids = ids_r
                        best_2d = coords_r

        loop_ids = best_ids
        loop_pts = V[np.asarray(loop_ids, dtype=np.int64)]

        c0 = loop_ids[0]
        c1 = loop_ids[m]
        c2 = loop_ids[2 * m]
        c3 = loop_ids[3 * m]
        bottom = loop_ids[0 : m + 1]                    # c0 -> c1
        right = loop_ids[m : 2 * m + 1]                 # c1 -> c2
        top = list(reversed(loop_ids[2 * m : 3 * m + 1]))   # c3 -> c2
        left_raw = loop_ids[3 * m :] + [loop_ids[0]]        # c3 -> c0
        left = list(reversed(left_raw))                      # c0 -> c3

        grid = np.full((m + 1, m + 1), -1, dtype=np.int64)
        for i in range(m + 1):
            grid[i, 0] = int(bottom[i])
            grid[i, m] = int(top[i])
        for j in range(m + 1):
            grid[0, j] = int(left[j])
            grid[m, j] = int(right[j])

        p00 = V[c0]
        p10 = V[c1]
        p11 = V[c2]
        p01 = V[c3]

        for i in range(1, m):
            u = float(i) / float(m)
            bu = V[bottom[i]]
            tu = V[top[i]]
            for j in range(1, m):
                v = float(j) / float(m)
                lv = V[left[j]]
                rv = V[right[j]]
                bilinear = (
                    (1.0 - u) * (1.0 - v) * p00
                    + u * (1.0 - v) * p10
                    + u * v * p11
                    + (1.0 - u) * v * p01
                )
                p = (
                    (1.0 - v) * bu + v * tu
                    + (1.0 - u) * lv + u * rv
                    - bilinear
                )
                vid = int(len(V) + len(new_vertices))
                new_vertices.append(np.asarray(p, dtype=np.float64))
                grid[i, j] = vid

        for i in range(m):
            for j in range(m):
                face = [
                    int(grid[i, j]),
                    int(grid[i + 1, j]),
                    int(grid[i + 1, j + 1]),
                    int(grid[i, j + 1]),
                ]
                if len(set(face)) == 4:
                    new_faces.append(face)

    if not new_faces:
        return V.copy(), F.copy(), 0

    if new_vertices:
        V = np.vstack([V, np.asarray(new_vertices, dtype=np.float64)])
    F = np.vstack([F, np.asarray(new_faces, dtype=np.int64)])
    return V, F, int(len(new_faces))
