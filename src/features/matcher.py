"""
FLANN-based feature matching with Lowe's ratio test + RANSAC F-matrix verification.
"""
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from .detector import ImageFeatures


@dataclass
class ImageMatch:
    img_i: int
    img_j: int
    pts_i: np.ndarray   # Nx2 float32 in image i
    pts_j: np.ndarray   # Nx2 float32 in image j
    F: np.ndarray       # 3×3 fundamental matrix


MatchGraph = Dict[Tuple[int, int], ImageMatch]


def _flann_matcher():
    index_params  = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    return cv2.FlannBasedMatcher(index_params, search_params)


def match_features(features: List[ImageFeatures],
                   ratio: float = 0.75,
                   min_inliers: int = 20) -> MatchGraph:
    """
    Match all pairs (i,j) i<j.
    - Lowe's ratio test
    - RANSAC F-matrix verification
    Returns MatchGraph: {(i,j): ImageMatch}
    """
    flann = _flann_matcher()
    graph: MatchGraph = {}
    n = len(features)

    for i in range(n):
        for j in range(i + 1, n):
            fi, fj = features[i], features[j]
            if len(fi.descriptors) < 8 or len(fj.descriptors) < 8:
                continue

            matches = flann.knnMatch(fi.descriptors, fj.descriptors, k=2)

            # Lowe ratio test
            good = [m for m, n_ in matches if m.distance < ratio * n_.distance]
            if len(good) < min_inliers:
                continue

            pts_i = np.float32([fi.keypoints[m.queryIdx].pt for m in good])
            pts_j = np.float32([fj.keypoints[m.trainIdx].pt for m in good])

            F, mask = cv2.findFundamentalMat(pts_i, pts_j,
                                             cv2.FM_RANSAC, 1.0, 0.999)
            if F is None or mask is None:
                continue

            mask = mask.ravel().astype(bool)
            if mask.sum() < min_inliers:
                continue

            graph[(i, j)] = ImageMatch(
                img_i=i, img_j=j,
                pts_i=pts_i[mask],
                pts_j=pts_j[mask],
                F=F
            )
            print(f"  [matcher] ({i},{j}): {mask.sum()} inliers")

    return graph
