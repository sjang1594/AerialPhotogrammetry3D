"""
Poisson surface reconstruction from dense point cloud.
"""
import open3d as o3d
import numpy as np


def reconstruct_poisson(pcd: o3d.geometry.PointCloud,
                         depth: int = 8,
                         density_threshold: float = 0.01) -> o3d.geometry.TriangleMesh:
    """
    Estimate normals, run Poisson reconstruction, crop by density.
    Returns cleaned TriangleMesh.
    """
    # Normal estimation
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)
    # Flip normals to point up (toward aerial cameras)
    normals = np.asarray(pcd.normals)
    normals[normals[:, 2] < 0] *= -1
    pcd.normals = o3d.utility.Vector3dVector(normals)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, linear_fit=False
    )
    densities = np.asarray(densities)

    # Remove low-density vertices
    threshold = np.quantile(densities, density_threshold)
    vertices_to_remove = densities < threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    mesh.compute_vertex_normals()
    print(f"[poisson] Mesh: {len(mesh.vertices)} verts, {len(mesh.triangles)} faces")
    return mesh
