"""
AerialPhotogrammetry3D — Full Pipeline Entry Point

Usage:
  python run_pipeline.py --mode synthetic --output outputs/
  python run_pipeline.py --mode synthetic --output outputs/ --skip-sfm  (use GT poses)
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import open3d as o3d


ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data_gen.synthetic_scene import build_scene, get_scene_bounds
from src.data_gen.camera_rig import generate_camera_poses, render_scene
from src.features.detector import detect_features
from src.features.matcher import match_features
from src.sfm.incremental import run_incremental_sfm
from src.sfm.bundle_adjustment import run_bundle_adjustment
from src.mvs.depth_estimator import estimate_all_depths
from src.mvs.depth_fusion import fuse_depth_maps
from src.mesh.reconstruction import reconstruct_poisson
from src.mesh.processing import crop_to_scene, simplify_mesh, clean_mesh
from src.texture.uv_atlas import generate_uv_atlas
from src.texture.texture_mapper import build_texture_atlas, save_textured_obj
from src.occlusion.detector import detect_occlusion_mask
from src.occlusion.inpainter import inpaint_atlas
from src.evaluation.metrics import reprojection_error, chamfer_distance
from src.evaluation.visualizer import plot_camera_trajectory, plot_reprojection_errors


def load_images(img_dir: str):
    paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)
                    if f.endswith('.png')])
    return [cv2.imread(p) for p in paths], paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',   default='synthetic', choices=['synthetic'])
    parser.add_argument('--output', default='outputs/')
    parser.add_argument('--skip-sfm', action='store_true',
                        help='Use GT poses instead of running SfM')
    parser.add_argument('--use-lama', action='store_true',
                        help='Use LaMa for occlusion inpainting')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    t0 = time.time()

    # ── Phase 1: Data Generation ──────────────────────────────────────────
    print("\n[Phase 1] Generating synthetic scene...")
    synth_dir = os.path.join("data", "synthetic")
    os.makedirs(synth_dir, exist_ok=True)

    scene = build_scene(n_buildings=15)
    poses = generate_camera_poses(grid_size=4, altitude=50.0, tilt_deg=10.0)
    img_dir, depth_dir, poses_path = render_scene(scene, poses, synth_dir)

    images, img_paths = load_images(img_dir)
    with open(poses_path) as f:
        gt_poses = json.load(f)
    K = np.array(gt_poses[0]["K"])
    print(f"  {len(images)} images rendered  ({time.time()-t0:.1f}s)")

    plot_camera_trajectory(gt_poses, os.path.join(args.output, "camera_trajectory.png"))

    # ── Phase 2: Feature Detection & Matching ────────────────────────────
    print("\n[Phase 2] Detecting & matching features...")
    features   = detect_features(images)
    match_graph = match_features(features)
    print(f"  {len(match_graph)} image pairs matched  ({time.time()-t0:.1f}s)")

    # ── Phase 3: Structure from Motion ───────────────────────────────────
    if args.skip_sfm:
        print("\n[Phase 3] Skipping SfM — using GT poses")
        # Build a trivial recon object with GT poses + GT depth for downstream
        from src.sfm.incremental import SfMReconstruction
        recon = SfMReconstruction(K)
        for i, p in enumerate(gt_poses):
            recon.cameras[i] = (np.array(p["R"]), np.array(p["t"]))
        sfm_poses = gt_poses
        ba_error  = None
    else:
        print("\n[Phase 3] Running incremental SfM...")
        recon = run_incremental_sfm(match_graph, K)
        print(f"  Running bundle adjustment...")
        ba_error = run_bundle_adjustment(recon, max_iter=30)
        print(f"  BA done  ({time.time()-t0:.1f}s)")

        # Build pose list for downstream modules
        sfm_poses = []
        for i in range(len(gt_poses)):
            if i in recon.cameras:
                R, t = recon.cameras[i]
                entry = dict(gt_poses[i])
                entry["R"] = R.tolist()
                entry["t"] = t.tolist()
            else:
                entry = gt_poses[i]
            sfm_poses.append(entry)

        # Save sparse cloud
        pts = recon.get_3d_points_array()
        if len(pts):
            pcd_sparse = o3d.geometry.PointCloud()
            pcd_sparse.points = o3d.utility.Vector3dVector(pts)
            sparse_path = os.path.join(args.output, "sparse_cloud.ply")
            o3d.io.write_point_cloud(sparse_path, pcd_sparse)
            print(f"  Saved {sparse_path}")

        # Reprojection error plot
        repr_metrics = reprojection_error(recon, K)
        print(f"  Reprojection: {repr_metrics}")
        errors_list = []
        for (cam_id, pt_id), pt2d in recon.observations.items():
            R, t = recon.cameras[cam_id]
            X = recon.points3d[pt_id].reshape(1, 3)
            rvec, _ = cv2.Rodrigues(R)
            proj, _ = cv2.projectPoints(X, rvec, t, K, None)
            errors_list.append(np.linalg.norm(proj.ravel() - pt2d))
        plot_reprojection_errors(errors_list, os.path.join(args.output, "reprojection_errors.png"))

    # ── Phase 4: Dense Depth Estimation ──────────────────────────────────
    print("\n[Phase 4] Estimating dense depth maps...")
    depth_maps = estimate_all_depths(images, sfm_poses, neighbors=2)

    print("\n[Phase 4b] Fusing depth maps → dense cloud...")
    pcd_dense = fuse_depth_maps(depth_maps, images, sfm_poses, voxel_size=0.3)
    dense_path = os.path.join(args.output, "dense_cloud.ply")
    o3d.io.write_point_cloud(dense_path, pcd_dense)
    print(f"  Saved {dense_path}  ({time.time()-t0:.1f}s)")

    # ── Phase 5: Mesh Generation ──────────────────────────────────────────
    print("\n[Phase 5] Poisson surface reconstruction...")
    mesh = reconstruct_poisson(pcd_dense, depth=8)

    min_b, max_b = get_scene_bounds()
    mesh = crop_to_scene(mesh, min_b - 5, max_b + 5)
    mesh = clean_mesh(mesh)
    mesh = simplify_mesh(mesh, target_faces=80_000)

    mesh_path = os.path.join(args.output, "mesh.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"  Saved {mesh_path}  ({time.time()-t0:.1f}s)")

    # ── Phase 6: Texture Atlas ────────────────────────────────────────────
    print("\n[Phase 6] Building texture atlas...")
    atlas = build_texture_atlas(mesh, images, sfm_poses, atlas_size=2048)
    atlas_path = os.path.join(args.output, "texture_atlas.png")
    cv2.imwrite(atlas_path, atlas)
    print(f"  Saved {atlas_path}")

    uvs, _ = generate_uv_atlas(mesh, atlas_size=2048)
    obj_path = os.path.join(args.output, "textured_mesh.obj")
    save_textured_obj(mesh, uvs, atlas_path, obj_path)
    print(f"  Saved {obj_path}  ({time.time()-t0:.1f}s)")

    # ── Phase 7: Occlusion Inpainting ─────────────────────────────────────
    print("\n[Phase 7] Occlusion detection & inpainting...")
    occ_mask  = detect_occlusion_mask(atlas, mesh, sfm_poses)
    inpainted = inpaint_atlas(atlas, occ_mask, use_lama=args.use_lama)
    inp_path  = os.path.join(args.output, "texture_atlas_inpainted.png")
    cv2.imwrite(inp_path, inpainted)
    print(f"  Saved {inp_path}  ({time.time()-t0:.1f}s)")

    # ── Phase 8: Evaluation ────────────────────────────────────────────────
    print("\n[Phase 8] Computing metrics...")
    metrics = {}

    if not args.skip_sfm and ba_error is not None:
        repr_metrics = reprojection_error(recon, K)
        metrics["reprojection"] = repr_metrics
        print(f"  Reprojection error: {repr_metrics}")

    # GT point cloud from depth maps (use GT poses for reference)
    gt_depth_paths = sorted([
        os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith('.npy')
    ])
    if gt_depth_paths:
        from src.mvs.depth_fusion import depth_to_pointcloud
        pcd_gt = o3d.geometry.PointCloud()
        for i, dp in enumerate(gt_depth_paths[:8]):  # sample 8 views for GT cloud
            d = np.load(dp)
            img = images[i]
            R   = np.array(gt_poses[i]["R"])
            t   = np.array(gt_poses[i]["t"])
            pcd_i = depth_to_pointcloud(d, img, K, R, t)
            pcd_gt += pcd_i
        pcd_gt = pcd_gt.voxel_down_sample(0.5)

        cd = chamfer_distance(pcd_dense, pcd_gt)
        metrics["chamfer"] = cd
        print(f"  Chamfer distance: {cd}")

    metrics_path = os.path.join(args.output, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[Done] metrics.json saved. Total time: {time.time()-t0:.1f}s")
    print(f"\nOutputs:")
    for fn in os.listdir(args.output):
        fpath = os.path.join(args.output, fn)
        sz = os.path.getsize(fpath) // 1024
        print(f"  {fn:40s} {sz:>8} KB")


if __name__ == "__main__":
    main()
