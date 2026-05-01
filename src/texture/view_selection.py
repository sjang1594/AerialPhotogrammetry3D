"""
Per-face view selection: choose best camera by normal-view angle + resolution.
"""
import numpy as np
import open3d as o3d
from typing import List, Dict, Tuple


def _face_center_and_normal(mesh: o3d.geometry.TriangleMesh) -> Tuple[np.ndarray, np.ndarray]:
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    v0, v1, v2 = verts[faces[:,0]], verts[faces[:,1]], verts[faces[:,2]]
    centers = (v0 + v1 + v2) / 3.0
    normals_raw = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals_raw, axis=1, keepdims=True) + 1e-9
    normals = normals_raw / norms
    return centers, normals


def select_best_view(mesh: o3d.geometry.TriangleMesh,
                     poses: List[dict]) -> np.ndarray:
    """
    For each face return the index of the best camera.
    Score = cos(angle between face normal and camera direction).
    Returns int array of shape (n_faces,), -1 = no valid view.
    """
    centers, normals = _face_center_and_normal(mesh)
    n_faces = len(centers)
    n_cams  = len(poses)

    best_cam   = -np.ones(n_faces, dtype=int)
    best_score = -np.ones(n_faces)

    for ci, pose in enumerate(poses):
        eye = np.array(pose["eye"])                 # world position
        dirs = eye - centers                        # face→camera
        dists = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9
        dirs_norm = dirs / dists

        score = np.einsum('fd,fd->f', normals, dirs_norm)   # cos angle

        improve = score > best_score
        best_score[improve] = score[improve]
        best_cam[improve]   = ci

    return best_cam


def project_point(pt3d: np.ndarray, K: np.ndarray,
                  R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Project 3D world point to 2D pixel. Returns (u, v)."""
    cam = R @ pt3d + t
    if cam[2] <= 0:
        return np.array([-1.0, -1.0])
    uv = K @ cam
    return uv[:2] / uv[2]
