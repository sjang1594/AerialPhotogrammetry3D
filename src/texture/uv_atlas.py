"""
Simple UV atlas: pack each face into a rectangular atlas using a shelf algorithm.
Falls back to xatlas if available.
"""
import numpy as np
import open3d as o3d
from typing import Tuple


ATLAS_SIZE = 2048


def _compute_face_areas(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    v0 = verts[faces[:,0]]; v1 = verts[faces[:,1]]; v2 = verts[faces[:,2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def generate_uv_atlas(mesh: o3d.geometry.TriangleMesh,
                      atlas_size: int = ATLAS_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign UV coordinates per face-vertex.
    Uses xatlas if available, otherwise simple shelf-packing.

    Returns:
      uvs     : (n_faces, 3, 2) float32  [0,1] UV per face-vertex
      face_map: (n_faces,) int  — face index (identity for manual packing)
    """
    try:
        import xatlas
        verts  = np.asarray(mesh.vertices,   dtype=np.float32)
        faces  = np.asarray(mesh.triangles,  dtype=np.uint32)
        atlas  = xatlas.Atlas()
        atlas.add_mesh(verts, faces)
        atlas.generate(xatlas.ChartOptions(), xatlas.PackOptions())
        vmapping, indices, uvs_flat = atlas[0]
        n_faces = len(faces)
        uvs = uvs_flat[indices.reshape(-1)].reshape(n_faces, 3, 2)
        uvs /= np.array([atlas.width, atlas.height], dtype=np.float32)
        print(f"[uv] xatlas: {atlas.width}×{atlas.height}")
        return uvs.astype(np.float32), np.arange(n_faces, dtype=int)
    except ImportError:
        pass

    # Fallback: shelf packing
    areas  = _compute_face_areas(mesh)
    n      = len(areas)
    # Assign each face a fixed-size cell (4×4 px in atlas)
    cell   = 4
    cols   = atlas_size // cell
    uvs    = np.zeros((n, 3, 2), dtype=np.float32)
    inv    = 1.0 / atlas_size

    for fi in range(n):
        col = fi % cols
        row = fi // cols
        u0  = col * cell * inv
        v0  = row * cell * inv
        u1  = u0 + cell * inv
        v1  = v0 + cell * inv
        uvs[fi, 0] = [u0, v0]
        uvs[fi, 1] = [u1, v0]
        uvs[fi, 2] = [u0, v1]

    print(f"[uv] shelf packing: {n} faces → {atlas_size}×{atlas_size}")
    return uvs, np.arange(n, dtype=int)
