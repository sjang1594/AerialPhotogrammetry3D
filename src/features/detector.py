"""SIFT feature detector wrapper."""
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ImageFeatures:
    image_id: int
    keypoints: list   # list of cv2.KeyPoint
    descriptors: np.ndarray  # Nx128 float32


def detect_features(images: List[np.ndarray], nfeatures: int = 4096) -> List[ImageFeatures]:
    """
    Detect SIFT keypoints + descriptors for each image.
    Returns list of ImageFeatures (one per image).
    """
    sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=0.015,
                           edgeThreshold=15)
    results = []
    for i, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        kps, descs = sift.detectAndCompute(gray, None)
        if descs is None:
            descs = np.zeros((0, 128), dtype=np.float32)
        results.append(ImageFeatures(image_id=i, keypoints=kps, descriptors=descs))
        print(f"  [detector] img {i:04d}: {len(kps)} keypoints")
    return results
