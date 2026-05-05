"""
Accuracy test: C++ BA vs Python BA on identical synthetic inputs.

Both are evaluated with the same metric: mean unweighted reprojection error.
Pass criterion: err_cpp <= err_py * 1.3  (30% slack — different optimizers
  converge to slightly different local minima, both valid).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import copy
import time

from src.sfm.incremental import SfMReconstruction
from src.sfm.bundle_adjustment import _pack, _build_obs_arrays, run_bundle_adjustment
from src.sfm._ba_cpp import run_bundle_adjustment_cpp


def random_K(W=1920, H=1080, fov_deg=60):
    f = W / (2 * np.tan(np.radians(fov_deg) / 2))
    return np.array([[f, 0, W/2], [0, f, H/2], [0, 0, 1]], dtype=np.float64)


def random_camera(seed_offset=0):
    angle = np.random.uniform(0, 2*np.pi)
    R, _ = cv2.Rodrigues(np.random.randn(3) * 0.2)
    t = np.array([np.cos(angle)*5, np.sin(angle)*5, -50.0]) + np.random.randn(3)
    return R, t


def project(K, R, t, pts3d):
    P = K @ np.hstack([R, t.reshape(3,1)])
    h = P @ np.hstack([pts3d, np.ones((len(pts3d),1))]).T
    return (h[:2] / h[2]).T


def build_problem(n_cams, n_pts, obs_per_pt, noise_px=0.5, seed=42):
    np.random.seed(seed)
    K = random_K()
    recon = SfMReconstruction(K)
    cameras = {}
    for i in range(n_cams):
        R, t = random_camera()
        recon.add_camera(i, R, t)
        cameras[i] = (R, t)

    pts3d = np.random.uniform(-20, 20, (n_pts, 3))
    pts3d[:, 2] += 40

    for pid in range(n_pts):
        visible = np.random.choice(list(range(n_cams)),
                                   min(obs_per_pt, n_cams), replace=False)
        for ci in visible:
            R, t = cameras[ci]
            uv = project(K, R, t, pts3d[pid:pid+1])[0]
            uv += np.random.randn(2) * noise_px
            recon.points3d[pid] = pts3d[pid].copy()
            recon.observations[(ci, pid)] = uv

    return recon


def pack_for_cpp(recon):
    params0, cam_ids, pt_ids, cam_idx, pt_idx = _pack(recon)
    obs_cam, obs_pt, obs_xy = _build_obs_arrays(recon.observations, cam_idx, pt_idx)
    n_cams = len(cam_ids)
    n_pts  = len(pt_ids)
    cam_params = params0[:n_cams*6].reshape(n_cams, 6).copy()
    pts3d_arr  = params0[n_cams*6:].reshape(n_pts, 3).copy()
    return (cam_params, pts3d_arr,
            obs_cam.astype(np.int32), obs_pt.astype(np.int32),
            obs_xy.copy(), cam_ids, pt_ids)


def reproj_error_unweighted(recon):
    """Mean unweighted reprojection error in pixels."""
    K = recon.K
    errs = []
    for (ci, pid), uv_obs in recon.observations.items():
        if ci not in recon.cameras or pid not in recon.points3d:
            continue
        R, t = recon.cameras[ci]
        X = recon.points3d[pid].reshape(1, 3)
        rvec, _ = cv2.Rodrigues(R)
        proj, _ = cv2.projectPoints(X, rvec, t, K, None)
        errs.append(np.linalg.norm(proj.ravel() - uv_obs))
    return float(np.mean(errs)) if errs else 0.0


def run_test(label, n_cams, n_pts, obs_per_pt, max_iter=30):
    recon_py  = build_problem(n_cams, n_pts, obs_per_pt, seed=7)
    recon_cpp = copy.deepcopy(recon_py)

    K = recon_py.K
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    # ── Python BA ───────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    run_bundle_adjustment(recon_py, max_iter=max_iter)
    t_py = time.perf_counter() - t0
    err_py = reproj_error_unweighted(recon_py)

    # ── C++ BA ──────────────────────────────────────────────────────────────
    cam_params, pts3d_arr, obs_cam, obs_pt, obs_xy, cam_ids, pt_ids = \
        pack_for_cpp(recon_cpp)

    t0 = time.perf_counter()
    res = run_bundle_adjustment_cpp(
        cam_params.astype(np.float64), pts3d_arr.astype(np.float64),
        obs_cam, obs_pt, obs_xy.astype(np.float64),
        fx, fy, cx, cy, max_iter
    )
    t_cpp = time.perf_counter() - t0

    # Write C++ result back
    new_cams = np.array(res["cam_params"])
    new_pts  = np.array(res["pts3d"])
    for i, ci in enumerate(cam_ids):
        R, _ = cv2.Rodrigues(new_cams[i, :3])
        recon_cpp.cameras[ci] = (R, new_cams[i, 3:])
    for j, pid in enumerate(pt_ids):
        recon_cpp.points3d[pid] = new_pts[j]

    err_cpp = reproj_error_unweighted(recon_cpp)

    speedup = t_py / t_cpp if t_cpp > 0 else float('inf')
    passed  = err_cpp <= err_py * 1.3   # 30% slack

    status = "PASS" if passed else "FAIL"
    print(f"\n[{status}] {label}")
    print(f"  Python:  err={err_py:.4f} px   t={t_py*1000:6.1f} ms")
    print(f"  C++:     err={err_cpp:.4f} px  t={t_cpp*1000:6.1f} ms"
          f"  iters={res['n_iters']}")
    print(f"  speedup={speedup:.1f}×   Δerr={err_cpp-err_py:+.4f} px")

    return passed, speedup, err_py, err_cpp


if __name__ == "__main__":
    print("BA accuracy & speed test  (same metric: unweighted reproj error)")
    print("=" * 60)
    results = []
    results.append(run_test("small  ( 5 cams,   500 pts)",  5,  500, 3))
    results.append(run_test("medium (20 cams,  2000 pts)", 20, 2000, 4))
    results.append(run_test("large  (50 cams,  5000 pts)", 50, 5000, 5))

    n_pass = sum(p for p, *_ in results)
    avg_speedup = np.mean([s for _, s, *_ in results])
    print(f"\n{'='*60}")
    print(f"Result: {n_pass}/{len(results)} passed")
    print(f"Average speedup: {avg_speedup:.1f}×")
    sys.exit(0 if n_pass == len(results) else 1)
