"""
Bundle adjustment via scipy.optimize.least_squares (Huber loss).
Optimizes camera poses + 3D point positions jointly.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import cv2
from .incremental import SfMReconstruction
from .triangulation import projection_matrix


def _pack(recon: SfMReconstruction):
    """Pack cameras and points into parameter vector."""
    cam_ids  = sorted(recon.cameras.keys())
    pt_ids   = sorted(recon.points3d.keys())
    cam_idx  = {c: i for i, c in enumerate(cam_ids)}
    pt_idx   = {p: i for i, p in enumerate(pt_ids)}

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


def _residuals(params, K, n_cams, cam_ids, pt_ids, cam_idx, pt_idx, observations):
    """Reprojection residuals."""
    res = []
    for (cam_id, pt_id), pt2d in observations.items():
        ci = cam_idx[cam_id]
        pi = pt_idx[pt_id]
        rvec = params[ci*6:ci*6+3]
        t    = params[ci*6+3:ci*6+6]
        X    = params[n_cams*6 + pi*3: n_cams*6 + pi*3+3].reshape(1, 3)
        proj, _ = cv2.projectPoints(X, rvec, t, K, None)
        proj = proj.ravel()
        res.extend([proj[0] - pt2d[0], proj[1] - pt2d[1]])
    return np.array(res)


def run_bundle_adjustment(recon: SfMReconstruction, max_iter: int = 50) -> float:
    """
    Run BA. Returns mean reprojection error (pixels) after optimization.
    Modifies recon in-place.
    """
    K = recon.K
    params0, cam_ids, pt_ids, cam_idx, pt_idx = _pack(recon)
    n_cams = len(cam_ids)
    observations = recon.observations

    def fun(p):
        return _residuals(p, K, n_cams, cam_ids, pt_ids, cam_idx, pt_idx, observations)

    # method='lm' silently ignores loss= (scipy limitation); use 'trf' for Huber
    result = least_squares(fun, params0, method='trf',
                           loss='huber', f_scale=1.0,
                           max_nfev=max_iter * len(params0),
                           verbose=0)

    _unpack(result.x, n_cams, cam_ids, pt_ids, recon)
    res = result.fun
    mean_err = np.sqrt(np.mean(res**2))
    print(f"[BA] mean reprojection error: {mean_err:.3f} px")
    return mean_err
