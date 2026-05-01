"""
Incremental SfM: register new cameras via PnP, triangulate new points.
"""
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from ..features.matcher import MatchGraph
from .triangulation import triangulate_dlt, filter_cheirality, projection_matrix


class SfMReconstruction:
    """Holds growing set of registered cameras + 3D points."""

    def __init__(self, K: np.ndarray):
        self.K = K
        # camera_id -> (R 3x3, t 3,)
        self.cameras: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        # point3d_id -> xyz
        self.points3d: Dict[int, np.ndarray] = {}
        # (cam_id, point3d_id) -> pt2d
        self.observations: Dict[Tuple[int, int], np.ndarray] = {}
        self._next_pt_id = 0

    def add_camera(self, cam_id: int, R: np.ndarray, t: np.ndarray):
        self.cameras[cam_id] = (R, t.ravel())

    def add_points(self, pts3d: np.ndarray,
                   cam_i: int, pts_i: np.ndarray,
                   cam_j: int, pts_j: np.ndarray) -> List[int]:
        ids = []
        for k in range(len(pts3d)):
            pid = self._next_pt_id
            self._next_pt_id += 1
            self.points3d[pid] = pts3d[k]
            self.observations[(cam_i, pid)] = pts_i[k]
            self.observations[(cam_j, pid)] = pts_j[k]
            ids.append(pid)
        return ids

    def get_3d_points_array(self) -> np.ndarray:
        if not self.points3d:
            return np.zeros((0, 3))
        return np.array(list(self.points3d.values()))


def initialize_reconstruction(match_graph: MatchGraph,
                               K: np.ndarray,
                               init_pair: Tuple[int, int]) -> SfMReconstruction:
    """Bootstrap reconstruction from the initial pair."""
    from .two_view import recover_pose_from_match
    from .triangulation import triangulate_dlt, filter_cheirality, projection_matrix

    i, j = init_pair
    match = match_graph[(i, j)]
    R, t, mask = recover_pose_from_match(match.pts_i, match.pts_j, K)

    recon = SfMReconstruction(K)
    R0, t0 = np.eye(3), np.zeros(3)
    recon.add_camera(i, R0, t0)
    recon.add_camera(j, R, t)

    P1 = projection_matrix(K, R0, t0)
    P2 = projection_matrix(K, R, t)

    pts_i = match.pts_i[mask]
    pts_j = match.pts_j[mask]

    pts3d = triangulate_dlt(P1, P2, pts_i, pts_j)
    valid = filter_cheirality(pts3d, P1, P2)
    pts3d = pts3d[valid]
    pts_i = pts_i[valid]
    pts_j = pts_j[valid]

    recon.add_points(pts3d, i, pts_i, j, pts_j)
    print(f"[sfm] Init pair ({i},{j}): {len(pts3d)} 3D points")
    return recon


def register_next_camera(recon: SfMReconstruction,
                          match_graph: MatchGraph,
                          unregistered: List[int]) -> Optional[int]:
    """
    Find unregistered camera with most 2D-3D correspondences and register it via PnP.
    Returns the newly registered camera id, or None.
    """
    K = recon.K
    best_cam, best_pts3d, best_pts2d = None, None, None
    best_count = 0

    for cand in unregistered:
        pts3d_list, pts2d_list = [], []
        for reg_cam in recon.cameras:
            key = (min(reg_cam, cand), max(reg_cam, cand))
            if key not in match_graph:
                continue
            match = match_graph[key]

            # find which observations are already triangulated
            for pid, pt3d in recon.points3d.items():
                if (reg_cam, pid) not in recon.observations:
                    continue
                obs2d_reg = recon.observations[(reg_cam, pid)]
                # find corresponding pt2d in cand
                if key[0] == reg_cam:
                    # reg_cam = i, cand = j
                    dists = np.linalg.norm(match.pts_i - obs2d_reg, axis=1)
                    idx = np.argmin(dists)
                    if dists[idx] < 2.0:
                        pts3d_list.append(pt3d)
                        pts2d_list.append(match.pts_j[idx])
                else:
                    dists = np.linalg.norm(match.pts_j - obs2d_reg, axis=1)
                    idx = np.argmin(dists)
                    if dists[idx] < 2.0:
                        pts3d_list.append(pt3d)
                        pts2d_list.append(match.pts_i[idx])

        if len(pts3d_list) > best_count:
            best_count = len(pts3d_list)
            best_cam   = cand
            best_pts3d = np.array(pts3d_list, dtype=np.float64)
            best_pts2d = np.array(pts2d_list, dtype=np.float32)

    if best_cam is None or best_count < 6:
        return None

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        best_pts3d, best_pts2d, K, None,
        reprojectionError=4.0, confidence=0.999
    )
    if not success or inliers is None:
        return None

    R, _ = cv2.Rodrigues(rvec)
    t    = tvec.ravel()
    recon.add_camera(best_cam, R, t)
    print(f"[sfm] Registered cam {best_cam}: {len(inliers)} inliers")

    # Triangulate new points with all registered cameras
    for reg_cam in list(recon.cameras.keys()):
        if reg_cam == best_cam:
            continue
        key = (min(reg_cam, best_cam), max(reg_cam, best_cam))
        if key not in match_graph:
            continue
        match = match_graph[key]
        R0, t0 = recon.cameras[reg_cam]
        P0 = projection_matrix(K, R0, t0)
        P1 = projection_matrix(K, R, t)

        pts3d = triangulate_dlt(P0, P1, match.pts_i, match.pts_j)
        from .triangulation import filter_cheirality
        valid = filter_cheirality(pts3d, P0, P1)
        if valid.sum() < 4:
            continue
        if key[0] == reg_cam:
            recon.add_points(pts3d[valid], reg_cam, match.pts_i[valid], best_cam, match.pts_j[valid])
        else:
            recon.add_points(pts3d[valid], best_cam, match.pts_i[valid], reg_cam, match.pts_j[valid])

    return best_cam


def run_incremental_sfm(match_graph: MatchGraph, K: np.ndarray) -> SfMReconstruction:
    """Full incremental SfM loop."""
    from .two_view import select_initial_pair
    init_pair = select_initial_pair(match_graph)
    recon = initialize_reconstruction(match_graph, K, init_pair)

    all_cams = set()
    for (i, j) in match_graph:
        all_cams.add(i)
        all_cams.add(j)

    unregistered = list(all_cams - set(recon.cameras.keys()))
    while unregistered:
        cam = register_next_camera(recon, match_graph, unregistered)
        if cam is None:
            break
        unregistered.remove(cam)

    print(f"[sfm] Final: {len(recon.cameras)} cameras, {len(recon.points3d)} points")
    return recon
