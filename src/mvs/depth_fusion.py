"""
Multi-view depth fusion → dense point cloud.
Projects each depth map into 3D, filters outliers, merges.
"""
import numpy as np
import open3d as o3d
from typing import List


def depth_to_pointcloud(depth: np.ndarray, color: np.ndarray,
                         K: np.ndarray, R: np.ndarray, t: np.ndarray,
                         max_depth: float = 80.0) -> o3d.geometry.PointCloud:
    """
    Back-project valid depth pixels to world space.
    Returns Open3D PointCloud.
    """
    H, W = depth.shape
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    valid = (depth > 0) & (depth < max_depth)

    z = depth[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy

    pts_cam = np.stack([x, y, z], axis=1)  # N×3 in camera space

    # Camera to world: world = R^T (cam - t) = R^T @ cam - R^T @ t
    Rinv = R.T
    pts_world = (Rinv @ pts_cam.T).T + Rinv @ (-t)  # N×3

    clr = color[valid].astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_world)
    pcd.colors = o3d.utility.Vector3dVector(clr)
    return pcd


def fuse_depth_maps(depth_maps: List[np.ndarray],
                    images: List[np.ndarray],
                    poses: List[dict],
                    voxel_size: float = 0.3) -> o3d.geometry.PointCloud:
    """
    Project all depth maps into 3D, merge, and voxel-downsample.
    Returns dense PointCloud.
    """
    K = np.array(poses[0]["K"])
    combined = o3d.geometry.PointCloud()

    for i, (depth, img, pose) in enumerate(zip(depth_maps, images, poses)):
        R = np.array(pose["R"])
        t = np.array(pose["t"])
        pcd = depth_to_pointcloud(depth, img, K, R, t)
        combined += pcd

    # Voxel downsample + outlier removal
    combined = combined.voxel_down_sample(voxel_size)
    combined, _ = combined.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"[fusion] Dense cloud: {len(combined.points)} points")
    return combined
