"""
Micro-benchmark: time each Python module in isolation with realistic synthetic data.
Run: python benchmarks/bench_python.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

def _random_K(W=1920, H=1080, fov_deg=60):
    f = W / (2 * np.tan(np.radians(fov_deg) / 2))
    return np.array([[f, 0, W/2], [0, f, H/2], [0, 0, 1]], dtype=np.float64)

def _random_camera():
    """Random [R|t] roughly 50 m above origin."""
    angle = np.random.uniform(0, 2*np.pi)
    R, _ = cv2.Rodrigues(np.random.randn(3) * 0.3)
    t = np.array([np.cos(angle)*5, np.sin(angle)*5, -50.0]) + np.random.randn(3)*2
    return R, t

def _project(K, R, t, pts3d):
    P = K @ np.hstack([R, t.reshape(3,1)])
    h = P @ np.hstack([pts3d, np.ones((len(pts3d),1))]).T  # 3×N
    return (h[:2] / h[2]).T  # N×2


def build_ba_problem(n_cams=20, n_pts=2000, obs_per_pt=4):
    """Build a synthetic BA problem."""
    from src.sfm.incremental import SfMReconstruction

    K = _random_K()
    recon = SfMReconstruction(K)

    cameras = {}
    for i in range(n_cams):
        R, t = _random_camera()
        recon.add_camera(i, R, t)
        cameras[i] = (R, t)

    pts3d = np.random.uniform(-20, 20, (n_pts, 3))
    pts3d[:, 2] += 40   # push in front of cameras

    cam_ids = list(range(n_cams))
    for pid in range(n_pts):
        visible = np.random.choice(cam_ids, min(obs_per_pt, n_cams), replace=False)
        for ci in visible:
            R, t = cameras[ci]
            uv = _project(K, R, t, pts3d[pid:pid+1])[0]
            uv += np.random.randn(2) * 0.5   # pixel noise
            recon.points3d[pid] = pts3d[pid]
            recon.observations[(ci, pid)] = uv

    return recon

def build_depth_problem(n_images=8, W=640, H=480):
    """Build small synthetic image + pose list for depth estimation."""
    K = _random_K(W, H)
    images = [np.random.randint(0, 255, (H, W, 3), dtype=np.uint8) for _ in range(n_images)]
    poses = []
    for i in range(n_images):
        R, t = _random_camera()
        poses.append({"K": K.tolist(), "R": R.tolist(), "t": t.tolist()})
    return images, poses

def bench(label, fn, reps=1):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    mean_t = sum(times) / len(times)
    print(f"  {label:<45s} {mean_t*1000:8.1f} ms  (×{reps})")
    return result, mean_t


def run_ba_benchmarks():
    print("\n=== Bundle Adjustment ===")
    from src.sfm.bundle_adjustment import (
        _pack, _build_obs_arrays, _residuals_vec, _jac_sparsity, run_bundle_adjustment
    )

    configs = [
        ("small  ( 5 cams,   500 pts)", 5, 500, 3),
        ("medium (20 cams,  2000 pts)", 20, 2000, 4),
        ("large  (50 cams,  5000 pts)", 50, 5000, 5),
    ]
    for label, nc, np_, obs in configs:
        recon = build_ba_problem(nc, np_, obs)
        params0, cam_ids, pt_ids, cam_idx, pt_idx = _pack(recon)
        obs_cam, obs_pt, obs_xy = _build_obs_arrays(recon.observations, cam_idx, pt_idx)
        n_cams, n_pts = len(cam_ids), len(pt_ids)
        K = recon.K

        print(f"\n  [{label}]  {len(obs_cam)} observations")
        bench("  _residuals_vec (1 call)",
              lambda: _residuals_vec(params0, K, n_cams, n_pts, obs_cam, obs_pt, obs_xy),
              reps=10)
        bench("  _jac_sparsity",
              lambda: _jac_sparsity(n_cams, n_pts, obs_cam, obs_pt),
              reps=5)
        bench("  run_bundle_adjustment (max_iter=15)",
              lambda: run_bundle_adjustment(
                  build_ba_problem(nc, np_, obs), max_iter=15),
              reps=3)

def run_depth_benchmarks():
    print("\n=== MVS Depth (StereoSGBM + back-projection) ===")
    from src.mvs.depth_estimator import compute_depth_map, estimate_all_depths

    K = _random_K(640, 480)
    R_i, t_i = _random_camera()
    R_j, t_j = _random_camera()
    # Make a slightly offset second camera for a valid baseline
    t_j = t_i + np.array([5.0, 0, 0])
    R_j = R_i.copy()

    img_i = np.random.randint(20, 200, (480, 640, 3), dtype=np.uint8)
    img_j = np.random.randint(20, 200, (480, 640, 3), dtype=np.uint8)

    bench("  compute_depth_map  640×480 single pair",
          lambda: compute_depth_map(img_i, img_j, K, R_i, t_i, R_j, t_j),
          reps=3)

    images, poses = build_depth_problem(n_images=8, W=640, H=480)
    bench("  estimate_all_depths  8 imgs, 640×480, neighbors=2",
          lambda: estimate_all_depths(images, poses, neighbors=2),
          reps=1)


def run_rodrigues_benchmarks():
    """Rodrigues is the inner-loop bottleneck in BA residuals."""
    print("\n=== cv2.Rodrigues loop (BA inner loop) ===")
    import time

    for n_cams in [5, 20, 50, 100]:
        vecs = np.random.randn(n_cams, 3) * 0.3

        def loop_rodrigues():
            Rs = np.empty((n_cams, 3, 3))
            for i in range(n_cams):
                Rs[i], _ = cv2.Rodrigues(vecs[i])
            return Rs

        bench(f"  cv2.Rodrigues loop  n_cams={n_cams:3d}",
              loop_rodrigues, reps=100)

if __name__ == "__main__":
    print("Python baseline benchmarks")
    print("=" * 60)
    run_rodrigues_benchmarks()
    run_ba_benchmarks()
    run_depth_benchmarks()
    print("\nDone.")