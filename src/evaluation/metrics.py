"""
Evaluation metrics: reprojection error, Chamfer distance, texture PSNR.
"""
import numpy as np
import open3d as o3d
import cv2
from typing import List, Dict


def reprojection_error(recon, K: np.ndarray) -> Dict[str, float]:
    """Compute mean and std reprojection error across all observations."""
    errors = []
    for (cam_id, pt_id), pt2d in recon.observations.items():
        R, t = recon.cameras[cam_id]
        X    = recon.points3d[pt_id].reshape(1, 3)
        rvec, _ = cv2.Rodrigues(R)
        proj, _ = cv2.projectPoints(X, rvec, t, K, None)
        err = np.linalg.norm(proj.ravel() - pt2d)
        errors.append(err)
    errors = np.array(errors)
    return {"mean_px": float(errors.mean()), "std_px": float(errors.std()),
            "median_px": float(np.median(errors))}


def chamfer_distance(pcd_pred: o3d.geometry.PointCloud,
                     pcd_gt: o3d.geometry.PointCloud) -> Dict[str, float]:
    """
    Symmetric Chamfer distance using Open3D KD-tree.
    """
    d1 = np.asarray(pcd_pred.compute_point_cloud_distance(pcd_gt))
    d2 = np.asarray(pcd_gt.compute_point_cloud_distance(pcd_pred))
    chamfer = float(d1.mean() + d2.mean())
    return {"chamfer_mean": chamfer,
            "pred_to_gt_mean": float(d1.mean()),
            "gt_to_pred_mean": float(d2.mean())}


def texture_psnr(atlas: np.ndarray, atlas_ref: np.ndarray) -> float:
    """PSNR between two atlas images."""
    mse = np.mean((atlas.astype(float) - atlas_ref.astype(float))**2)
    if mse == 0:
        return float('inf')
    return float(10 * np.log10(255**2 / mse))
