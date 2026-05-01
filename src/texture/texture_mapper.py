"""
Project selected views into UV atlas → RGB PNG.
"""
import numpy as np
import cv2
import open3d as o3d
from typing import List
from .view_selection import select_best_view, project_point
from .uv_atlas import generate_uv_atlas, ATLAS_SIZE


def build_texture_atlas(mesh: o3d.geometry.TriangleMesh,
                         images: List[np.ndarray],
                         poses: List[dict],
                         atlas_size: int = ATLAS_SIZE) -> np.ndarray:
    """
    Build a texture atlas image (H×W×3 uint8).
    For each face, project vertices into best camera, sample bilinear.
    """
    K      = np.array(poses[0]["K"])
    verts  = np.asarray(mesh.vertices)
    faces  = np.asarray(mesh.triangles)

    best_cam = select_best_view(mesh, poses)
    uvs, _   = generate_uv_atlas(mesh, atlas_size)

    atlas = np.zeros((atlas_size, atlas_size, 3), dtype=np.uint8)

    for fi in range(len(faces)):
        ci = best_cam[fi]
        if ci < 0:
            continue
        img = images[ci]
        R   = np.array(poses[ci]["R"])
        t   = np.array(poses[ci]["t"])
        H_img, W_img = img.shape[:2]

        # Project face vertices
        v_ids = faces[fi]
        pts3d = verts[v_ids]
        pts2d = np.stack([project_point(p, K, R, t) for p in pts3d])

        # Sample face centroid color — bilinear interpolation at float coordinates
        centroid2d = pts2d.mean(axis=0)
        pu, pv = float(centroid2d[0]), float(centroid2d[1])
        if 0.5 <= pu < W_img - 0.5 and 0.5 <= pv < H_img - 0.5:
            patch = cv2.getRectSubPix(img, (1, 1), (pu, pv))
            color = patch[0, 0]
        else:
            color = np.array([128, 128, 128], dtype=np.uint8)

        # Fill atlas cell corresponding to this face's UV bounding box
        uv = uvs[fi]                    # 3×2
        u_min = int(uv[:, 0].min() * atlas_size)
        u_max = int(uv[:, 0].max() * atlas_size) + 1
        v_min = int(uv[:, 1].min() * atlas_size)
        v_max = int(uv[:, 1].max() * atlas_size) + 1
        u_min = np.clip(u_min, 0, atlas_size - 1)
        u_max = np.clip(u_max, 0, atlas_size)
        v_min = np.clip(v_min, 0, atlas_size - 1)
        v_max = np.clip(v_max, 0, atlas_size)
        atlas[v_min:v_max, u_min:u_max] = color

    return atlas


def save_textured_obj(mesh: o3d.geometry.TriangleMesh,
                       uvs: np.ndarray,
                       atlas_path: str,
                       obj_path: str):
    """Write .obj + .mtl referencing the atlas."""
    import os
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    mtl_path = obj_path.replace(".obj", ".mtl")
    base_name = os.path.basename(obj_path).replace(".obj", "")
    atlas_name = os.path.basename(atlas_path)

    with open(mtl_path, "w") as f:
        f.write(f"newmtl material0\nmap_Kd {atlas_name}\n")

    with open(obj_path, "w") as f:
        f.write(f"mtllib {os.path.basename(mtl_path)}\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        # UV coords per face-vertex (flat)
        uv_list = uvs.reshape(-1, 2)
        for uv in uv_list:
            f.write(f"vt {uv[0]:.6f} {1.0 - uv[1]:.6f}\n")
        f.write("usemtl material0\n")
        for fi, face in enumerate(faces):
            ti = fi * 3
            f.write(f"f {face[0]+1}/{ti+1} {face[1]+1}/{ti+2} {face[2]+1}/{ti+3}\n")

    print(f"[texture] Saved {obj_path}")
