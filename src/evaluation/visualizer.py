"""
Visualization helpers: matplotlib plots + Open3D interactive viewer.
"""
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from typing import List, Optional


def plot_reprojection_errors(errors: List[float], out_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(errors, bins=50, color='steelblue', edgecolor='white')
    ax.set_xlabel("Reprojection Error (px)")
    ax.set_ylabel("Count")
    ax.set_title("Reprojection Error Histogram")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"[vis] Saved {out_path}")
    else:
        plt.show()
    plt.close()


def plot_camera_trajectory(poses: List[dict], out_path: Optional[str] = None):
    eyes = np.array([p["eye"] for p in poses])
    fig  = plt.figure(figsize=(8, 6))
    ax   = fig.add_subplot(111, projection='3d')
    ax.scatter(eyes[:,0], eyes[:,1], eyes[:,2], c='red', s=30, label='Cameras')
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("Aerial Camera Trajectory")
    ax.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"[vis] Saved {out_path}")
    else:
        plt.show()
    plt.close()


def visualize_pointcloud(pcd: o3d.geometry.PointCloud, title: str = "Point Cloud"):
    o3d.visualization.draw_geometries([pcd], window_name=title)


def visualize_mesh(mesh: o3d.geometry.TriangleMesh, title: str = "Mesh"):
    o3d.visualization.draw_geometries([mesh], window_name=title)
