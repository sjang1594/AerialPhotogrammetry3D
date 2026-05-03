"""
Bundle adjustment via scipy.optimize.least_squares (Huber loss).
Optimizes camera poses + 3D point positions jointly.

Vectorized residuals + Jacobian sparsity -- orders of magnitude faster
than the per-observation cv2.projectPoints loop with dense finite-diff.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import cv2
from .incremental import SfMReconstruction


def _pack(recon: SfMReconstruction):
    """Pack cameras and points into parameter vector."""
    cam_ids = sorted(recon.cameras.keys())
    pt_ids  = sorted(recon.points3d.keys())
    cam_idx = {c: i for i, c in enumerate(cam_ids)}
    pt_idx  = {p: i for i, p in enumerate(pt_ids)}

    params = []
    for c in cam_ids:
        R, t = recon.cameras[c]
        rvec, _ = cv2.Rodrigues(R)
        params.extend(rvec.ravel())  # 3
        params.extend(t.ravel())     # 3
    for p in pt_ids:
        params.extend(recon.points3d[p])  # 3

    return (np.array(params, dtype=float),
            cam_ids, pt_ids, cam_idx, pt_idx)


def _unpack(params, n_cams, cam_ids, pt_ids, recon):
    """Update recon in-place from parameter vector."""
    for i, c in enumerate(cam_ids):
        rvec = params[i*6:i*6+3]
        t    = params[i*6+3:i*6+6]
        R, _ = cv2.Rodrigues(rvec)
        recon.cameras[c] = (R, t)
    offset = n_cams * 6
    for j, p in enumerate(pt_ids):
        recon.points3d[p] = params[offset + j*3: offset + j*3+3]


def _build_obs_arrays(observations, cam_idx, pt_idx):
    """Flatten observations dict into parallel numpy arrays."""
    M = len(observations)
    obs_cam = np.empty(M, dtype=np.int64)
    obs_pt  = np.empty(M, dtype=np.int64)
    obs_xy  = np.empty((M, 2), dtype=np.float64)
    for k, ((cam_id, pt_id), pt2d) in enumerate(observations.items()):
        obs_cam[k] = cam_idx[cam_id]
        obs_pt[k]  = pt_idx[pt_id]
        obs_xy[k]  = pt2d
    return obs_cam, obs_pt, obs_xy


def _residuals_vec(params, K, n_cams, n_pts, obs_cam, obs_pt, obs_xy):
    """Vectorized reprojection residuals: one numpy pass over all observations."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    cam_params = params[:n_cams*6].reshape(n_cams, 6)
    pts        = params[n_cams*6:].reshape(n_pts, 3)

    # Rodrigues per camera (only n_cams calls, not per-observation)
    Rs = np.empty((n_cams, 3, 3))
    for i in range(n_cams):
        Rs[i], _ = cv2.Rodrigues(cam_params[i, :3])
    ts = cam_params[:, 3:6]

    R_obs = Rs[obs_cam]                              # (M, 3, 3)
    t_obs = ts[obs_cam]                              # (M, 3)
    X     = pts[obs_pt]                              # (M, 3)

    pts_cam = np.einsum('mij,mj->mi', R_obs, X) + t_obs
    z = pts_cam[:, 2]
    u = fx * (pts_cam[:, 0] / z) + cx
    v = fy * (pts_cam[:, 1] / z) + cy

    return np.stack([u - obs_xy[:, 0], v - obs_xy[:, 1]], axis=1).ravel()


def _jac_sparsity(n_cams, n_pts, obs_cam, obs_pt):
    """Each observation's 2 residual rows depend on 6 cam params + 3 point params."""
    M = len(obs_cam)
    n_params = n_cams * 6 + n_pts * 3
    S = lil_matrix((2 * M, n_params), dtype=np.uint8)

    rows = np.arange(M)
    for k in range(6):
        S[2*rows,     obs_cam*6 + k] = 1
        S[2*rows + 1, obs_cam*6 + k] = 1
    pt_offset = n_cams * 6
    for k in range(3):
        S[2*rows,     pt_offset + obs_pt*3 + k] = 1
        S[2*rows + 1, pt_offset + obs_pt*3 + k] = 1
    return S


def run_bundle_adjustment(recon: SfMReconstruction, max_iter: int = 50) -> float:
    """
    Run BA. Returns mean reprojection error (pixels) after optimization.
    Modifies recon in-place.
    """
    K = recon.K
    params0, cam_ids, pt_ids, cam_idx, pt_idx = _pack(recon)
    n_cams = len(cam_ids)
    n_pts  = len(pt_ids)

    obs_cam, obs_pt, obs_xy = _build_obs_arrays(recon.observations, cam_idx, pt_idx)
    sparsity = _jac_sparsity(n_cams, n_pts, obs_cam, obs_pt)

    def fun(p):
        return _residuals_vec(p, K, n_cams, n_pts, obs_cam, obs_pt, obs_xy)

    result = least_squares(
        fun, params0,
        jac_sparsity=sparsity,
        method='trf',
        loss='huber', f_scale=1.0,
        x_scale='jac',
        max_nfev=max_iter,
        verbose=0,
    )

    _unpack(result.x, n_cams, cam_ids, pt_ids, recon)
    mean_err = np.sqrt(np.mean(result.fun ** 2))
    print(f"[BA] {n_cams} cams, {n_pts} pts, {len(obs_cam)} obs -> "
          f"{result.nfev} fn evals, mean reproj err {mean_err:.3f} px")
    return mean_err
