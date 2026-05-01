"""
Two-view geometry: Essential matrix decomposition, initial reconstruction.
"""
import cv2
import numpy as np
from typing import Tuple, Optional


def recover_pose_from_match(pts_i: np.ndarray,
                             pts_j: np.ndarray,
                             K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Recover R, t from matched point correspondences via Essential matrix.
    Returns (R 3×3, t 3×1, mask bool array).
    """
    E, mask = cv2.findEssentialMat(pts_i, pts_j, K,
                                   method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask2 = cv2.recoverPose(E, pts_i, pts_j, K, mask=mask)
    combined = (mask.ravel() & mask2.ravel()).astype(bool)
    return R, t.ravel(), combined


def select_initial_pair(match_graph: dict) -> Tuple[int, int]:
    """
    Select the best initial pair for two-view reconstruction.
    Ranks by: n_inliers * (1 - homography_inlier_ratio).
    High inlier count + low H-ratio = many inliers from a non-planar (good baseline) pair.
    A high H-ratio means the scene is nearly planar from this viewpoint — degenerate for SfM init.
    """
    best_pair = None
    best_score = -1.0

    for pair, match in match_graph.items():
        n = len(match.pts_i)
        if n < 30:
            continue
        H, hmask = cv2.findHomography(match.pts_i, match.pts_j, cv2.RANSAC, 4.0)
        h_inliers = int(hmask.sum()) if hmask is not None else n
        h_ratio = h_inliers / n
        score = n * (1.0 - h_ratio)
        if score > best_score:
            best_score = score
            best_pair = pair

    if best_pair is None:
        best_pair = max(match_graph.keys(), key=lambda k: len(match_graph[k].pts_i))
    return best_pair
