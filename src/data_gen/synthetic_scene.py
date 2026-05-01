"""
Procedural urban scene generator.
Builds a flat ground (100x100m) + 10-20 rectangular buildings as Open3D TriangleMesh.
"""
import numpy as np
import open3d as o3d


def _box_mesh(cx, cy, w, d, h, color):
    """Axis-aligned box centered at (cx,cy,0), height h, RGBA color."""
    mesh = o3d.geometry.TriangleMesh.create_box(width=w, height=d, depth=h)
    mesh.translate([-w/2 + cx, -d/2 + cy, 0.0])
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def build_scene(n_buildings: int = 15, seed: int = 42) -> o3d.geometry.TriangleMesh:
    """
    Returns a single merged TriangleMesh:
      - Ground plane 100×100 m (gray)
      - n_buildings random rectangular buildings (varied colors)
    """
    rng = np.random.default_rng(seed)
    meshes = []

    # Ground plane as a thin box
    ground = _box_mesh(50, 50, 100, 100, 0.1, [0.5, 0.5, 0.5])
    meshes.append(ground)

    building_colors = [
        [0.8, 0.6, 0.4],
        [0.6, 0.7, 0.8],
        [0.9, 0.8, 0.6],
        [0.7, 0.5, 0.5],
        [0.5, 0.7, 0.6],
    ]

    for i in range(n_buildings):
        cx = rng.uniform(10, 90)
        cy = rng.uniform(10, 90)
        w  = rng.uniform(4, 15)
        d  = rng.uniform(4, 15)
        h  = rng.uniform(5, 30)
        color = building_colors[i % len(building_colors)]
        meshes.append(_box_mesh(cx, cy, w, d, h, color))

    # Merge all
    scene = meshes[0]
    for m in meshes[1:]:
        scene += m
    scene.compute_vertex_normals()
    return scene


def get_scene_bounds():
    """Returns (min_xyz, max_xyz) for the synthetic scene."""
    return np.array([0, 0, 0], dtype=float), np.array([100, 100, 35], dtype=float)
