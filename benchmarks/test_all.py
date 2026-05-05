"""
Integration test: BA wrapper + parallel depth estimation.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import copy

# ── 1. BA wrapper test ────────────────────────────────────────────────────────
print("=" * 60)
print("1. Bundle Adjustment — auto C++ dispatch via run_bundle_adjustment")
print("=" * 60)

from src.sfm.bundle_adjustment import run_bundle_adjustment, _HAS_CPP
from src.sfm.incremental import SfMReconstruction

print(f"   _HAS_CPP = {_HAS_CPP}")

def random_K():
    f = 1920 / (2 * np.tan(np.radians(60) / 2))
    return np.array([[f,0,960],[0,f,540],[0,0,1]], dtype=np.float64)

def random_camera():
    R, _ = cv2.Rodrigues(np.random.randn(3) * 0.2)
    t = np.array([np.random.randn()*5, np.random.randn()*5, -50.0])
    return R, t

def project(K, R, t, pts3d):
    P = K @ np.hstack([R, t.reshape(3,1)])
    h = P @ np.hstack([pts3d, np.ones((len(pts3d),1))]).T
    return (h[:2] / h[2]).T

np.random.seed(42)
K = random_K()
recon = SfMReconstruction(K)
cameras = {}
for i in range(20):
    R, t = random_camera()
    recon.add_camera(i, R, t)
    cameras[i] = (R, t)

pts3d = np.random.uniform(-20, 20, (2000, 3))
pts3d[:, 2] += 40
for pid in range(2000):
    for ci in np.random.choice(20, 4, replace=False):
        uv = project(K, cameras[ci][0], cameras[ci][1], pts3d[pid:pid+1])[0]
        uv += np.random.randn(2) * 0.5
        recon.points3d[pid] = pts3d[pid].copy()
        recon.observations[(ci, pid)] = uv

t0 = time.perf_counter()
err = run_bundle_adjustment(recon, max_iter=30)
elapsed = (time.perf_counter() - t0) * 1000
print(f"   -> {elapsed:.1f} ms,  mean reproj err = {err:.4f} px")
print(f"   [{'PASS' if err < 2.0 else 'FAIL'}] error within acceptable range")

# ── 2. build_remap_grid test ──────────────────────────────────────────────────
print()
print("=" * 60)
print("2. build_remap_grid — C++ vs Python")
print("=" * 60)

from src.mvs._depth_cpp import build_remap_grid

H, W = 480, 640
R1 = np.eye(3)
P1 = np.array([[K[0,0], 0, W/2, 0],
               [0, K[1,1], H/2, 0],
               [0, 0, 1, 0]], dtype=np.float64)

# Python reference
t0 = time.perf_counter()
u_grid = np.arange(W, dtype=np.float32)
v_grid = np.arange(H, dtype=np.float32)
gu, gv = np.meshgrid(u_grid, v_grid)
x_n = (gu - K[0,2]) / K[0,0]
y_n = (gv - K[1,2]) / K[1,1]
rays = np.stack([x_n, y_n, np.ones_like(x_n)], axis=2)
rays_rect = (R1 @ rays.reshape(-1, 3).T).T.reshape(H, W, 3)
z = rays_rect[:,:,2]; z = np.where(np.abs(z)<1e-6, 1e-6, z)
map_x_py = ((rays_rect[:,:,0] / z) * P1[0,0] + P1[0,2]).astype(np.float32)
map_y_py = ((rays_rect[:,:,1] / z) * P1[1,1] + P1[1,2]).astype(np.float32)
t_py = (time.perf_counter() - t0) * 1000

# C++
t0 = time.perf_counter()
for _ in range(10):
    map_x_cpp, map_y_cpp = build_remap_grid(H, W, K, R1, P1[:, :3])
t_cpp = (time.perf_counter() - t0) / 10 * 1000

max_err = max(np.max(np.abs(map_x_cpp - map_x_py)),
              np.max(np.abs(map_y_cpp - map_y_py)))
print(f"   Python:  {t_py:.2f} ms")
print(f"   C++:     {t_cpp:.2f} ms  (speedup {t_py/t_cpp:.1f}×)")
print(f"   Max pixel error: {max_err:.6f}")
print(f"   [{'PASS' if max_err < 0.01 else 'FAIL'}] remap grid matches")

# ── 3. Parallel depth test ────────────────────────────────────────────────────
print()
print("=" * 60)
print("3. estimate_all_depths_parallel — C++ threadpool")
print("=" * 60)

from src.mvs._depth_cpp import estimate_all_depths_parallel
from src.mvs.depth_estimator import compute_depth_map

# Build realistic stereo pair (horizontal offset)
R_base, _ = cv2.Rodrigues(np.zeros(3))
imgs, poses = [], []
n_imgs = 6
baseline = 5.0
for i in range(n_imgs):
    img = (np.random.rand(H, W, 3) * 200 + 20).astype(np.uint8)
    imgs.append(img)
    t_i = np.array([i * baseline, 0.0, 0.0])
    poses.append({"K": K.tolist(), "R": R_base.tolist(), "t": t_i.tolist()})

# Python sequential (original loop, no C++ dispatch)
def _estimate_sequential(images, poses, neighbors):
    K_ = np.array(poses[0]["K"])
    n = len(images)
    result = []
    for i in range(n):
        R_i = np.array(poses[i]["R"]); t_i = np.array(poses[i]["t"])
        depths = []
        for delta in range(1, neighbors + 1):
            for j in [i - delta, i + delta]:
                if j < 0 or j >= n: continue
                R_j = np.array(poses[j]["R"]); t_j = np.array(poses[j]["t"])
                d = compute_depth_map(images[i], images[j], K_, R_i, t_i, R_j, t_j)
                if (d > 0).sum() > 200: depths.append(d)
        if depths:
            stack = np.stack(depths, 0); vm = stack > 0
            cnt = vm.sum(0).astype(np.float32); cnt[cnt==0] = 1
            fused = (stack * vm).sum(0) / cnt
        else:
            fused = np.zeros_like(images[i][:,:,0], dtype=np.float32)
        result.append(fused)
    return result

t0 = time.perf_counter()
depths_seq = _estimate_sequential(imgs, poses, neighbors=1)
t_seq = time.perf_counter() - t0

# C++ parallel
n_cpus = os.cpu_count() or 4
t0 = time.perf_counter()
depths_par = estimate_all_depths_parallel(
    imgs, poses, compute_depth_map, neighbors=1, n_threads=n_cpus)
t_par = time.perf_counter() - t0

print(f"   Python sequential: {t_seq*1000:.0f} ms  ({n_imgs} imgs, neighbors=1)")
print(f"   C++ parallel:      {t_par*1000:.0f} ms  ({n_cpus} threads)")
print(f"   Speedup:           {t_seq/t_par:.1f}×")
print(f"   [{'PASS' if len(depths_par) == n_imgs else 'FAIL'}] correct output count")

print()
print("=" * 60)
print("All tests done.")
