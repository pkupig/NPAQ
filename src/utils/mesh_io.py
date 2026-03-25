"""
Robust mesh and point cloud I/O utilities.
Supports .obj, .off, .ply, .stl, .npz formats.
"""

import numpy as np
import os
from typing import Tuple, Optional, Union, List
import warnings
import struct


def read_mesh(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a mesh from file. Supports .obj, .off, .ply, .stl.
    Returns vertices (N,3) and faces (M,3) (triangular). For quad meshes, faces may be (M,4).
    Uses trimesh if available, otherwise simple fallbacks.
    """
    ext = os.path.splitext(filename)[1].lower()
    try:
        import trimesh
        loaded = trimesh.load(filename, process=False, force='mesh')
        # trimesh may return a Scene (multi-geometry), a Trimesh, or a PointCloud
        if hasattr(loaded, 'geometry') and loaded.geometry:
            # Scene: concatenate all sub-meshes
            geoms = list(loaded.geometry.values())
            meshes = [g for g in geoms if hasattr(g, 'faces') and len(g.faces) > 0]
            if meshes:
                combined = trimesh.util.concatenate(meshes)
                vertices = np.array(combined.vertices)
                faces = np.array(combined.faces)
                if len(vertices) > 0:
                    return vertices, faces
            # Scene with only point-cloud geometry
            loaded = geoms[0]
        vertices = np.array(loaded.vertices)
        faces = np.array(loaded.faces) if hasattr(loaded, 'faces') and len(loaded.faces) > 0 else np.array([])
        if len(vertices) > 0:
            return vertices, faces
    except ImportError:
        warnings.warn("trimesh not available, using simple loaders.")
    except Exception:
        pass
    # Format-specific fallbacks (no trimesh or trimesh failed)
    if ext == '.obj':
        return _load_obj(filename)
    elif ext == '.off':
        return _load_off(filename)
    elif ext == '.ply':
        return _load_ply(filename)
    elif ext == '.stl':
        return _load_stl(filename)
    else:
        raise ValueError(f"Unsupported format: {ext}")


def write_mesh(filename: str, vertices: np.ndarray, faces: np.ndarray):
    """
    Write mesh to file. Format determined by extension.
    Quad meshes (faces with 4 vertices) bypass trimesh, which only handles triangles.
    """
    ext = os.path.splitext(filename)[1].lower()
    faces_arr = np.asarray(faces)
    is_quad = faces_arr.ndim == 2 and faces_arr.shape[1] == 4
    if is_quad:
        # trimesh.Trimesh does not support quads; use format-native writers directly
        if ext in ('.obj', ''):
            _write_obj(filename if ext else filename + '.obj', vertices, faces_arr)
        elif ext == '.off':
            _write_off(filename, vertices, faces_arr)
        else:
            _write_obj(filename, vertices, faces_arr)
        return
    # Triangle mesh: prefer trimesh for its broader format support
    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces_arr)
        mesh.export(filename)
    except ImportError:
        if ext == '.obj':
            _write_obj(filename, vertices, faces_arr)
        elif ext == '.off':
            _write_off(filename, vertices, faces_arr)
        else:
            raise ValueError(f"Without trimesh, only .obj and .off are supported for writing.")


def read_pointcloud(filename: str) -> np.ndarray:
    """
    Read point cloud from file. Supports .xyz, .ply, .obj, .npz.
    For mesh formats (.ply, .obj) the mesh vertices are returned as the point cloud.
    Returns (N,3) array.
    """
    pts, _ = read_pointcloud_with_normals(filename)
    return pts


def read_pointcloud_with_normals(
    filename: str,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Read point cloud and (if available) per-point normals.

    Returns:
        points:  (N, 3) float32
        normals: (N, 3) float32, or None if the file contains no normal data.

    Supports .xyz, .ply, .obj, .off, .npz.
    For PLY files the function prefers open3d (handles binary + normals),
    then trimesh, then a manual ASCII fallback.
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext == '.npz':
        data = np.load(filename)
        pts = data['points'].astype(np.float32)
        nrm = data['normals'].astype(np.float32) if 'normals' in data else None
        return pts, nrm

    if ext == '.xyz':
        raw = np.loadtxt(filename)
        if raw.shape[1] >= 6:
            return raw[:, :3].astype(np.float32), raw[:, 3:6].astype(np.float32)
        return raw[:, :3].astype(np.float32), None

    if ext == '.ply':
        # open3d is the most reliable PLY loader (binary + ASCII, normals, colors)
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(filename)
            pts = np.asarray(pcd.points, dtype=np.float32)
            if len(pts) == 0:
                # Might be a mesh PLY — load as mesh and take vertices
                mesh = o3d.io.read_triangle_mesh(filename)
                pts = np.asarray(mesh.vertices, dtype=np.float32)
                nrm = (np.asarray(mesh.vertex_normals, dtype=np.float32)
                       if mesh.has_vertex_normals() else None)
                if len(pts) > 0:
                    return pts, nrm
            nrm = (np.asarray(pcd.normals, dtype=np.float32)
                   if pcd.has_normals() else None)
            if len(pts) > 0:
                return pts, nrm
        except Exception:
            pass

        # trimesh fallback
        try:
            import trimesh
            loaded = trimesh.load(filename, process=False)
            pts = np.array(getattr(loaded, 'vertices', []), dtype=np.float32)
            nrm = None
            if hasattr(loaded, 'vertex_normals') and loaded.vertex_normals is not None:
                nrm = np.array(loaded.vertex_normals, dtype=np.float32)
                if nrm.shape != pts.shape:
                    nrm = None
            if len(pts) > 0:
                return pts, nrm
        except Exception:
            pass

        # Manual PLY fallback (ASCII + binary, points + optional normals)
        return _load_ply_points_with_normals(filename)

    if ext in ('.obj', '.off', '.stl'):
        verts, _ = read_mesh(filename)
        return verts.astype(np.float32), None

    raise ValueError(f"Unsupported point cloud format: {ext}")


def write_pointcloud(filename: str, points: np.ndarray, normals: Optional[np.ndarray] = None):
    """
    Write point cloud to file. Supports .xyz, .npz.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.npz':
        if normals is not None:
            np.savez(filename, points=points, normals=normals)
        else:
            np.savez(filename, points=points)
    elif ext == '.xyz':
        np.savetxt(filename, points)
    else:
        raise ValueError(f"Unsupported point cloud format for writing: {ext}")


# ---------- Simple loaders (fallback) ----------

def _load_obj(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    vertices = []
    faces = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.strip().split()
                # handle various face formats: f v1 v2 v3  or  f v1/vt1 v2/vt2 v3/vt3  or  f v1//vn1 ...
                face = []
                for p in parts[1:]:
                    idx = p.split('/')[0]
                    face.append(int(idx) - 1)  # 1-indexed
                faces.append(face)
    return np.array(vertices), np.array(faces)


def _write_obj(filename: str, vertices: np.ndarray, faces: np.ndarray):
    with open(filename, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            line = "f"
            for idx in face:
                line += f" {idx+1}"
            f.write(line + "\n")


def _load_off(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(filename, 'r') as f:
        lines = f.readlines()
    if lines[0].strip() != 'OFF':
        raise ValueError("Not a valid OFF file")
    n_verts, n_faces, _ = map(int, lines[1].strip().split())
    vertices = []
    for i in range(n_verts):
        vertices.append(list(map(float, lines[2+i].strip().split())))
    faces = []
    for i in range(n_faces):
        parts = lines[2+n_verts+i].strip().split()
        n = int(parts[0])
        faces.append([int(p) for p in parts[1:1+n]])
    return np.array(vertices), np.array(faces)


def _write_off(filename: str, vertices: np.ndarray, faces: np.ndarray):
    with open(filename, 'w') as f:
        f.write("OFF\n")
        f.write(f"{len(vertices)} {len(faces)} 0\n")
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"{len(face)} " + " ".join(str(idx) for idx in face) + "\n")


def _load_ply(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    PLY loader fallback supporting ASCII and binary little/big endian.
    Returns vertex positions and polygon faces (when present).
    """
    verts, normals, faces = _load_ply_raw(filename)
    return verts, faces


def _load_ply_points_with_normals(filename: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """PLY loader that extracts xyz and optional nx/ny/nz from ASCII or binary files."""
    pts, nrm, _ = _load_ply_raw(filename)
    return pts, nrm


def _parse_ply_header(raw: bytes):
    end_token = b'end_header'
    end = raw.find(end_token)
    if end < 0:
        raise ValueError("PLY header missing end_header.")
    line_end = raw.find(b'\n', end)
    if line_end < 0:
        line_end = len(raw)
    header = raw[:line_end].decode('ascii', errors='replace').splitlines()
    data_start = line_end + 1

    fmt = None
    elements = []
    current = None
    for line in header:
        tok = line.strip().split()
        if not tok:
            continue
        if tok[0] == 'format':
            fmt = tok[1]
        elif tok[0] == 'element':
            current = {'name': tok[1], 'count': int(tok[2]), 'properties': []}
            elements.append(current)
        elif tok[0] == 'property' and current is not None:
            if tok[1] == 'list':
                current['properties'].append({
                    'kind': 'list',
                    'count_type': tok[2],
                    'item_type': tok[3],
                    'name': tok[4],
                })
            else:
                current['properties'].append({
                    'kind': 'scalar',
                    'type': tok[1],
                    'name': tok[2],
                })
    if fmt is None:
        raise ValueError("PLY header missing format.")
    return fmt, elements, data_start


def _ply_type_info(name: str):
    n = name.lower()
    table = {
        'char': ('b', np.int8),
        'int8': ('b', np.int8),
        'uchar': ('B', np.uint8),
        'uint8': ('B', np.uint8),
        'short': ('h', np.int16),
        'int16': ('h', np.int16),
        'ushort': ('H', np.uint16),
        'uint16': ('H', np.uint16),
        'int': ('i', np.int32),
        'int32': ('i', np.int32),
        'uint': ('I', np.uint32),
        'uint32': ('I', np.uint32),
        'float': ('f', np.float32),
        'float32': ('f', np.float32),
        'double': ('d', np.float64),
        'float64': ('d', np.float64),
    }
    if n not in table:
        raise ValueError(f"Unsupported PLY property type: {name}")
    return table[n]


def _load_ply_raw(filename: str) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    with open(filename, 'rb') as f:
        raw = f.read()

    fmt, elements, data_start = _parse_ply_header(raw)
    if fmt == 'ascii':
        return _load_ply_raw_ascii(raw[data_start:], elements)
    if fmt in ('binary_little_endian', 'binary_big_endian'):
        endian = '<' if fmt == 'binary_little_endian' else '>'
        return _load_ply_raw_binary(raw[data_start:], elements, endian)
    raise ValueError(f"Unsupported PLY format: {fmt}")


def _extract_vertex_points_normals(vertex_rows: np.ndarray, prop_names: List[str]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    def _col(name):
        return prop_names.index(name) if name in prop_names else None

    xi, yi, zi = _col('x'), _col('y'), _col('z')
    if xi is None or yi is None or zi is None:
        raise ValueError("PLY file has no x/y/z vertex properties.")
    pts = vertex_rows[:, [xi, yi, zi]].astype(np.float32)

    nxi, nyi, nzi = _col('nx'), _col('ny'), _col('nz')
    nrm = None
    if nxi is not None and nyi is not None and nzi is not None:
        nrm = vertex_rows[:, [nxi, nyi, nzi]].astype(np.float32)
    return pts, nrm


def _load_ply_raw_ascii(data: bytes, elements) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    lines = data.decode('ascii', errors='replace').splitlines()
    idx = 0
    verts = np.zeros((0, 3), dtype=np.float32)
    normals = None
    faces = []
    for elem in elements:
        if elem['name'] == 'vertex':
            rows = []
            prop_names = [p['name'] for p in elem['properties'] if p['kind'] == 'scalar']
            for _ in range(elem['count']):
                while idx < len(lines) and not lines[idx].strip():
                    idx += 1
                parts = lines[idx].strip().split()
                idx += 1
                rows.append([float(v) for v in parts[:len(prop_names)]])
            arr = np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, len(prop_names)), dtype=np.float32)
            verts, normals = _extract_vertex_points_normals(arr, prop_names)
        elif elem['name'] == 'face':
            for _ in range(elem['count']):
                while idx < len(lines) and not lines[idx].strip():
                    idx += 1
                parts = lines[idx].strip().split()
                idx += 1
                if not parts:
                    continue
                count = int(parts[0])
                if count > 0:
                    faces.append([int(v) for v in parts[1:1 + count]])
        else:
            idx += elem['count']
    return verts, normals, np.asarray(faces, dtype=np.int64) if faces else np.array([])


def _load_ply_raw_binary(data: bytes, elements, endian: str) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    offset = 0
    verts = np.zeros((0, 3), dtype=np.float32)
    normals = None
    faces = []

    for elem in elements:
        if elem['name'] == 'vertex':
            rows = []
            prop_names = [p['name'] for p in elem['properties'] if p['kind'] == 'scalar']
            prop_structs = [_ply_type_info(p['type'])[0] for p in elem['properties'] if p['kind'] == 'scalar']
            fmt = endian + ''.join(prop_structs)
            size = struct.calcsize(fmt)
            for _ in range(elem['count']):
                row = struct.unpack_from(fmt, data, offset)
                offset += size
                rows.append(row)
            arr = np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, len(prop_names)), dtype=np.float32)
            verts, normals = _extract_vertex_points_normals(arr, prop_names)
        elif elem['name'] == 'face':
            for _ in range(elem['count']):
                if not elem['properties']:
                    continue
                prop = elem['properties'][0]
                if prop['kind'] != 'list':
                    break
                count_fmt = endian + _ply_type_info(prop['count_type'])[0]
                item_code, _ = _ply_type_info(prop['item_type'])
                count = struct.unpack_from(count_fmt, data, offset)[0]
                offset += struct.calcsize(count_fmt)
                vals_fmt = endian + item_code * int(count)
                vals = struct.unpack_from(vals_fmt, data, offset) if count > 0 else ()
                offset += struct.calcsize(vals_fmt)
                if count > 0:
                    faces.append([int(v) for v in vals])
        else:
            for _ in range(elem['count']):
                for prop in elem['properties']:
                    if prop['kind'] == 'scalar':
                        offset += struct.calcsize(endian + _ply_type_info(prop['type'])[0])
                    else:
                        count_fmt = endian + _ply_type_info(prop['count_type'])[0]
                        item_code, _ = _ply_type_info(prop['item_type'])
                        count = struct.unpack_from(count_fmt, data, offset)[0]
                        offset += struct.calcsize(count_fmt)
                        offset += struct.calcsize(endian + item_code * int(count))

    return verts, normals, np.asarray(faces, dtype=np.int64) if faces else np.array([])


def _load_stl(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    # Binary STL not trivial. Use numpy-stl if available.
    try:
        from stl import mesh
        stl_mesh = mesh.Mesh.from_file(filename)
        vertices = stl_mesh.points.reshape(-1, 3)
        # STL stores triangles, we need face connectivity.
        # In stl format, vertices are repeated for each triangle.
        # We need to deduplicate.
        unique_vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
        faces = inverse.reshape(-1, 3)
        return unique_vertices, faces
    except ImportError:
        raise ImportError("Reading STL requires numpy-stl package.")
