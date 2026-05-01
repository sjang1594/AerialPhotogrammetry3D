"""
Detect occluded faces in the texture atlas (faces with no valid image coverage).
"""
import numpy as np
import open3d as o3d
from typing import List
from ..texture.view_selection import select_best_view, project_point


def detect_occlusion_mask(atlas: np.ndarray,
                           mesh: o3d.geometry.TriangleMesh,
                           poses: List[dict],
                           atlas_size: int = 2048) -> np.ndarray:
    """
    Returns binary mask (H×W uint8, 255=occluded) for atlas regions with no texture.
    Simple heuristic: dark pixels (sum < 30 across RGB) → occluded.
    """
    gray = atlas.sum(axis=2)
    mask = (gray < 30).astype(np.uint8) * 255

    # Dilate slightly to catch near-occluded seams
    kernel = np.ones((5, 5), np.uint8)
    import cv2
    mask = cv2.dilate(mask, kernel, iterations=1)
    occluded_pct = mask.mean() / 255 * 100
    print(f"[occlusion] Occluded: {occluded_pct:.1f}% of atlas")
    return mask
