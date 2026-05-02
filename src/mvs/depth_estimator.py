"""
Per-stereo-pair depth estimation using OpenCV StereoSGBM.
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict
import json
import os


def _rectify_pair(img_i: np.ndarray, img_j: np.ndarray,
                  K: np.ndarray,
                  R_i: np.ndarray, t_i: np.ndarray,
                  R_j: np.ndarray, t_j: np.ndarray,
                  img_size: Tuple[int, int]):
    """
    Stereo rectification of an arbitrary image pair.
    Returns rectified images + Q matrix (for reprojectImageTo3D).
    """
    # Relative rotation and translation from i->j
    R_rel = R_j @ R_i.T
    t_rel = t_j - R_rel @ t_i

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K, None, K, None, img_size, R_rel, t_rel,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )

    map1x, map1y = cv2.initUndistortRectifyMap(K, None, R1, P1, img_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K, None, R2, P2, img_size, cv2.CV_32FC1)

    rect_i = cv2.remap(img_i, map1x, map1y, cv2.INTER_LINEAR)
    rect_j = cv2.remap(img_j, map2x, map2y, cv2.INTER_LINEAR)

    return rect_i, rect_j, Q, R1, P1


def compute_depth_map(img_i: np.ndarray, img_j: np.ndarray,
                      K: np.ndarray,
                      R_i: np.ndarray, t_i: np.ndarray,
                      R_j: np.ndarray, t_j: np.ndarray) -> np.ndarray:
    """
    Compute depth map for img_i given img_j as neighbor.
    Returns depth map (H×W float32, invalid=0).
    """
    H, W = img_i.shape[:2]
    img_size = (W, H)

    rect_i, rect_j, Q, R1, P1 = _rectify_pair(
        img_i, img_j, K, R_i, t_i, R_j, t_j, img_size
    )

    gray_i = cv2.cvtColor(rect_i, cv2.COLOR_BGR2GRAY)
    gray_j = cv2.cvtColor(rect_j, cv2.COLOR_BGR2GRAY)

    n_disp = 16 * 8   # must be divisible by 16
    block  = 7
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=n_disp,
        blockSize=block,
        P1=8  * 3 * block**2,
        P2=32 * 3 * block**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disp = sgbm.compute(gray_i, gray_j).astype(np.float32) / 16.0
    disp[disp <= 0] = np.nan

    # Reproject to 3D and extract Z component
    pts3d = cv2.reprojectImageTo3D(np.nan_to_num(disp), Q)
    depth = pts3d[:, :, 2]
    depth[np.isnan(disp)] = 0.0
    depth[depth < 0] = 0.0

    # Map rectified depth back to original image space.
    # For each original pixel (u,v): un-project with K, rotate with R1, project with P1
    # to find the corresponding rectified pixel location, then sample depth there.
    u_grid = np.arange(W, dtype=np.float32)
    v_grid = np.arange(H, dtype=np.float32)
    grid_u, grid_v = np.meshgrid(u_grid, v_grid)

    x_n = (grid_u - K[0, 2]) / K[0, 0]
    y_n = (grid_v - K[1, 2]) / K[1, 1]
    rays = np.stack([x_n, y_n, np.ones_like(x_n)], axis=2)        # H×W×3
    rays_rect = (R1 @ rays.reshape(-1, 3).T).T.reshape(H, W, 3)   # apply rectification rotation

    z = rays_rect[:, :, 2]
    z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    map_back_x = ((rays_rect[:, :, 0] / z) * P1[0, 0] + P1[0, 2]).astype(np.float32)
    map_back_y = ((rays_rect[:, :, 1] / z) * P1[1, 1] + P1[1, 2]).astype(np.float32)

    depth_orig = cv2.remap(depth, map_back_x, map_back_y, cv2.INTER_NEAREST)
    return depth_orig.astype(np.float32)


def estimate_all_depths(images: List[np.ndarray],
                        poses: List[dict],
                        neighbors: int = 2) -> List[np.ndarray]:
    """
    For each image, fuse depth from `neighbors` adjacent images.
    Returns list of depth maps (H×W float32).
    """
    K = np.array(poses[0]["K"])
    n = len(images)
    depth_maps = []

    for i in range(n):
        R_i = np.array(poses[i]["R"])
        t_i = np.array(poses[i]["t"])
        depths = []

        for delta in range(1, neighbors + 1):
            for j in [i - delta, i + delta]:
                if j < 0 or j >= n:
                    continue
                R_j = np.array(poses[j]["R"])
                t_j = np.array(poses[j]["t"])
                d = compute_depth_map(images[i], images[j], K, R_i, t_i, R_j, t_j)
                if (d > 0).sum() > 1000:
                    depths.append(d)

        if depths:
            stack = np.stack(depths, axis=0)
            valid_mask = stack > 0
            count = valid_mask.sum(axis=0).astype(np.float32)
            count[count == 0] = 1
            fused = (stack * valid_mask).sum(axis=0) / count
        else:
            fused = np.zeros_like(images[i][:,:,0], dtype=np.float32)

        depth_maps.append(fused)
        print(f"  [depth] img {i:04d}: valid pixels = {(fused>0).sum()}")

    return depth_maps
