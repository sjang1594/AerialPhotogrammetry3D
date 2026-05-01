"""
DLT triangulation + non-linear refinement.
"""
import numpy as np
import cv2
from typing import List, Tuple


def triangulate_dlt(P1: np.ndarray, P2: np.ndarray,
                    pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """
    Triangulate 3D points from two projection matrices using cv2.triangulatePoints.
    pts1, pts2: Nx2 float32.
    Returns Nx3 float64.
    """
    pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)  # 4×N
    pts3d = (pts4d[:3] / pts4d[3]).T                        # N×3
    return pts3d


def filter_cheirality(pts3d: np.ndarray,
                      P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """Boolean mask: keep points with positive depth in both cameras."""
    def depth(P, X):
        r3 = P[2, :3]
        t3 = P[2, 3]
        w = np.sign(np.linalg.det(P[:3, :3]))
        return w * (r3 @ X.T + t3)

    d1 = depth(P1, pts3d)
    d2 = depth(P2, pts3d)
    return (d1 > 0) & (d2 > 0)


def projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """K[R|t] → 3×4."""
    Rt = np.hstack([R, t.reshape(3, 1)])
    return K @ Rt
