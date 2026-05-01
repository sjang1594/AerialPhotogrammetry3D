"""
Mesh post-processing: outlier removal, simplification, bounding box crop.
"""
import open3d as o3d
import numpy as np


def crop_to_scene(mesh: o3d.geometry.TriangleMesh,
                  min_bound: np.ndarray, max_bound: np.ndarray) -> o3d.geometry.TriangleMesh:
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    return mesh.crop(bbox)


def simplify_mesh(mesh: o3d.geometry.TriangleMesh,
                  target_faces: int = 100_000) -> o3d.geometry.TriangleMesh:
    n = len(mesh.triangles)
    if n <= target_faces:
        return mesh
    ratio = target_faces / n
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    mesh.compute_vertex_normals()
    print(f"[simplify] {n} → {len(mesh.triangles)} faces")
    return mesh


def clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    return mesh
