"""
One-shot script: insert scale-alignment cells into Phase 3 / 4 / 8 notebooks.

Adds:
  Phase 3 -- Step 5 "Scale Alignment with Ground Truth" (Umeyama similarity).
  Phase 4 -- short markdown linking back to Phase 3 alignment.
  Phase 8 -- evaluation cells: translation error & sparse-cloud Chamfer
             before vs. after similarity alignment.

Run once.  Idempotent: re-running detects the marker tag and skips.
"""
from pathlib import Path
import nbformat as nbf

NB_DIR = Path(__file__).parent
TAG = "scale-alignment-v1"


def already_added(nb):
    return any(TAG in (c.metadata.get("tags") or []) for c in nb.cells)


def md(source, tag=TAG):
    cell = nbf.v4.new_markdown_cell(source)
    cell.metadata["tags"] = [tag]
    return cell


def code(source, tag=TAG):
    cell = nbf.v4.new_code_cell(source)
    cell.metadata["tags"] = [tag]
    return cell


def find_index(nb, needle):
    """Return the index of the first cell whose source contains `needle`."""
    for i, c in enumerate(nb.cells):
        if needle in c.source:
            return i
    raise ValueError(f"anchor not found: {needle!r}")


# ---------------------------------------------------------------- Phase 3
P3_INTRO = r"""---

## Step 5 — Scale Alignment with Ground Truth

### Why this step exists

Bundle adjustment minimizes reprojection error, but reprojection is **invariant
to a 7-DoF similarity transform** of the world: 3 translation + 3 rotation +
1 scale. Multiplying every 3-D point and every camera centre by the same
factor produces the *same* image observations, so SfM cannot recover absolute
scale on its own.

That is why the cloud above is in *SfM units*, not metres. To compare against
ground truth (Phase 8) or to fuse with metric depth (Phase 4), we need a
similarity transform

$$
\mathbf{X}_{\text{metric}} \;=\; s\,\mathbf{R}_{\text{align}}\,\mathbf{X}_{\text{sfm}} \;+\; \mathbf{t}_{\text{align}}.
$$

### How we resolve it here

In a real (no-GT) pipeline the scale is recovered from one of:
GPS / IMU on the rig, a known baseline between two cameras, or a calibration target.
Since we **have** GT poses, we can solve the alignment in closed form using the
[Umeyama (1991)](https://web.stanford.edu/class/cs273/refs/umeyama.pdf) method:
given two corresponding point sets (here, estimated vs. GT camera centres), it
returns the $(s, \mathbf{R}, \mathbf{t})$ that minimises

$$
\sum_i \big\Vert\, s\,\mathbf{R}\,\mathbf{C}^{\text{sfm}}_i + \mathbf{t} - \mathbf{C}^{\text{gt}}_i \,\big\Vert^2.
$$

Camera centres are derived from the BA-optimized poses via $\mathbf{C} = -\mathbf{R}^\top \mathbf{t}$."""


P3_CODE_UMEYAMA = r"""def camera_center(R, t):
    return -R.T @ t

def umeyama(src, dst):
    # Closed-form similarity transform: src -> s R src + t ~= dst
    n = len(src)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    src_c, dst_c = src - mu_s, dst - mu_d
    H = src_c.T @ dst_c / n
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = Vt.T @ S @ U.T
    var_s = (src_c ** 2).sum() / n
    s = (D * np.diag(S)).sum() / var_s
    t = mu_d - s * (R @ mu_s)
    return s, R, t

cam_ids_a = sorted(recon.cameras.keys())
C_sfm = np.array([camera_center(*recon.cameras[c]) for c in cam_ids_a])
C_gt  = np.array([gt_poses[c]['eye']               for c in cam_ids_a])

s, R_align, t_align = umeyama(C_sfm, C_gt)

C_aligned = (s * (R_align @ C_sfm.T)).T + t_align
err_after = np.linalg.norm(C_aligned - C_gt, axis=1)

print(f"Scale factor s     : {s:.4f}      (SfM unit  ->  {s:.3f} m)")
print(f"Rotation R_align   : eigenvalues {np.linalg.eigvals(R_align).round(3)}")
print(f"Translation t_align: {t_align.round(2)} m\n")
print(f"Camera-centre error after similarity alignment:")
print(f"  mean   = {err_after.mean():.3f} m")
print(f"  median = {np.median(err_after):.3f} m")
print(f"  max    = {err_after.max():.3f} m")
print(f"\nNote: the *raw* SfM-frame centres have no metric meaning (different "
      f"origin and scale), so reporting their distance to GT is not informative. "
      f"The alignment converts them into a frame where the comparison is meaningful.")
"""


P3_CODE_VIZ = r"""fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(C_sfm[:, 0], C_sfm[:, 1], c='steelblue', s=60, label='SfM centres (raw)')
for k, cid in enumerate(cam_ids_a):
    axes[0].annotate(str(cid), (C_sfm[k, 0], C_sfm[k, 1]), fontsize=7)
axes[0].set_title('Raw SfM camera centres\n(arbitrary frame, |t_init| = 1)')
axes[0].set_xlabel('x_sfm'); axes[0].set_ylabel('y_sfm')
axes[0].set_aspect('equal'); axes[0].grid(alpha=0.3)

axes[1].scatter(C_gt[:, 0],      C_gt[:, 1],      c='limegreen',  s=80, marker='s', label='GT')
axes[1].scatter(C_aligned[:, 0], C_aligned[:, 1], c='steelblue',  s=40,             label='SfM (aligned)')
for k in range(len(cam_ids_a)):
    axes[1].plot([C_gt[k, 0], C_aligned[k, 0]],
                 [C_gt[k, 1], C_aligned[k, 1]], 'r-', alpha=0.4, lw=0.8)
axes[1].add_patch(plt.Rectangle((0, 0), 100, 100, fill=False, edgecolor='k', lw=1, ls='--'))
axes[1].set_title(f'After similarity transform (s = {s:.3f})\n'
                  f'red lines = residual error (mean {err_after.mean():.2f} m)')
axes[1].set_xlabel('X (m)'); axes[1].set_ylabel('Y (m)')
axes[1].set_aspect('equal'); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)

plt.suptitle('Scale alignment: SfM frame  ->  metric world frame', fontsize=12)
plt.tight_layout(); plt.show()
"""


P3_CODE_APPLY = r"""# Transform sparse cloud and poses into the metric (GT) frame.
def transform_pose(R_sfm, t_sfm, s, R_align, t_align):
    # SfM-frame pose -> metric-frame pose.  Projection is invariant under a
    # global similarity, so the camera *rotates* by R_align and its *centre*
    # undergoes the full similarity (s, R_align, t_align).
    C_sfm = -R_sfm.T @ t_sfm
    C_new = s * (R_align @ C_sfm) + t_align
    R_new = R_sfm @ R_align.T
    t_new = -R_new @ C_new
    return R_new, t_new

pts_aligned = (s * (R_align @ pts.T)).T + t_align

aligned_cameras = {
    cid: transform_pose(R, t, s, R_align, t_align)
    for cid, (R, t) in recon.cameras.items()
}

# Save aligned point cloud
pcd_aligned = o3d.geometry.PointCloud()
pcd_aligned.points = o3d.utility.Vector3dVector(pts_aligned)
aligned_path = os.path.join(out_dir, 'sparse_cloud_aligned.ply')
o3d.io.write_point_cloud(aligned_path, pcd_aligned)
print(f"Aligned sparse cloud saved: {aligned_path}  ({len(pts_aligned)} pts, metric)")

# Save the similarity transform (Phase 8 reads this)
alignment_path = os.path.join(out_dir, 'sfm_to_gt_alignment.json')
with open(alignment_path, 'w') as f:
    json.dump({
        'scale':       float(s),
        'R_align':     R_align.tolist(),
        't_align':     t_align.tolist(),
        'mean_center_error_m':   float(err_after.mean()),
        'median_center_error_m': float(np.median(err_after)),
        'max_center_error_m':    float(err_after.max()),
        'n_cameras_used':        int(len(cam_ids_a)),
        'method': 'umeyama_on_camera_centers',
    }, f, indent=2)
print(f"Similarity transform saved: {alignment_path}")
print(f"\nIn a real (no-GT) pipeline the same transform would come from GPS, IMU, "
      f"a known baseline, or a calibration target.")
"""


# ---------------------------------------------------------------- Phase 4
P4_NOTE = r"""---

### Side note: closing the metric-scale gap

This phase uses **GT poses** for clarity, so depth comes out in metres directly.
In a real pipeline you would feed the *SfM* poses from Phase 3 — but those are
in arbitrary scale. Phase 3 now produces a similarity transform
$(s,\mathbf{R}_{\text{align}},\mathbf{t}_{\text{align}})$ (saved to
`outputs/sfm_to_gt_alignment.json`). Applying it to the SfM camera centres
recovers the metric baseline $B$, after which the disparity-to-depth conversion
$Z = fB/d$ produces depth in metres without any GT.

Phase 8 quantifies how much this alignment matters."""


# ---------------------------------------------------------------- Phase 8
P8_INTRO = r"""---

## Scale-Alignment Sanity Check

Phase 3 produced a similarity transform that takes the SfM frame to the metric
GT frame. This section answers: **how much does that alignment actually
change the metrics?** If the answer is "a lot", then any pipeline that skips
this step is reporting numbers in the wrong units."""


P8_CODE = r"""alignment_path = os.path.join(outputs_dir, 'sfm_to_gt_alignment.json')

if not os.path.exists(alignment_path):
    print("Run phase3_sfm.ipynb (Step 5) first to produce sfm_to_gt_alignment.json.")
else:
    with open(alignment_path) as f:
        align = json.load(f)
    s_align   = align['scale']
    R_align   = np.array(align['R_align'])
    t_align   = np.array(align['t_align'])
    print(f"Loaded similarity transform: s = {s_align:.4f}, "
          f"|t_align| = {np.linalg.norm(t_align):.2f} m")

    # ---- 1.  Translation (camera-centre) error, before vs. after ---------
    cam_ids = sorted(eval_cameras.keys())
    C_sfm   = np.array([(-R.T @ t) for R, t in (eval_cameras[c] for c in cam_ids)])
    C_gt    = np.array([gt_poses[c]['eye'] for c in cam_ids])
    C_aln   = (s_align * (R_align @ C_sfm.T)).T + t_align

    # The "before" number is informational only -- the SfM frame has no metric
    # meaning, so this distance mixes scale and frame error into one figure.
    err_before = np.linalg.norm(C_sfm - C_gt, axis=1)
    err_after  = np.linalg.norm(C_aln - C_gt, axis=1)

    print(f"\nCamera-centre distance to GT:")
    print(f"  Without alignment: {err_before.mean():7.3f}  "
          f"(mixed units; not a real metric error)")
    print(f"  With  alignment  : {err_after.mean():7.3f} m  (true translation error)")
    print(f"  Reduction factor : {err_before.mean() / max(err_after.mean(), 1e-9):.1f} x")

    metrics['translation'] = {
        'mean_m':   float(err_after.mean()),
        'median_m': float(np.median(err_after)),
        'max_m':    float(err_after.max()),
    }

    # ---- 2.  Sparse cloud Chamfer, before vs. after ---------------------
    sparse_path = os.path.join(outputs_dir, 'sparse_cloud.ply')
    if os.path.exists(sparse_path) and pcd_gt is not None:
        pcd_sparse = o3d.io.read_point_cloud(sparse_path)
        pts_sfm    = np.asarray(pcd_sparse.points)
        pts_aligned = (s_align * (R_align @ pts_sfm.T)).T + t_align

        pcd_unaligned = o3d.geometry.PointCloud()
        pcd_unaligned.points = o3d.utility.Vector3dVector(pts_sfm)
        pcd_aligned = o3d.geometry.PointCloud()
        pcd_aligned.points = o3d.utility.Vector3dVector(pts_aligned)

        cd_unaligned = chamfer_distance(pcd_unaligned, pcd_gt)
        cd_aligned   = chamfer_distance(pcd_aligned,   pcd_gt)

        print(f"\nSparse-cloud Chamfer distance vs. GT cloud:")
        print(f"  Without alignment: {cd_unaligned.get('chamfer_mean', 'n/a')}  "
              f"(SfM cloud is in the wrong frame and scale)")
        print(f"  With  alignment  : {cd_aligned.get('chamfer_mean', 'n/a')}  "
              f"(meaningful metric error)")

        metrics['sparse_chamfer_aligned'] = cd_aligned
    else:
        print("\nsparse_cloud.ply not found -- run Phase 3 to enable Chamfer comparison.")

    # ---- 3.  Bar chart -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(cam_ids))
    ax.bar(x - 0.2, err_before, 0.4, color='salmon',     label='Before alignment')
    ax.bar(x + 0.2, err_after,  0.4, color='steelblue',  label='After alignment (metric)')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{c}' for c in cam_ids], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Camera-centre error (log scale)')
    ax.set_title('Per-camera translation error: similarity alignment closes the scale gap')
    ax.legend()
    plt.tight_layout(); plt.show()

    print("\nTakeaway: SfM is invariant to a 7-DoF similarity, so any 'translation "
          "error' reported in the SfM frame mixes pose error with the unknown "
          "global scale. Aligning first is what makes the metric honest.")
"""


# ---------------------------------------------------------------- Splicer
def splice(name, anchor, *cells):
    path = NB_DIR / name
    nb = nbf.read(path, as_version=4)
    if already_added(nb):
        print(f"[skip] {name}: already tagged with {TAG}")
        return
    idx = find_index(nb, anchor)
    nb.cells = nb.cells[: idx + 1] + list(cells) + nb.cells[idx + 1:]
    nbf.write(nb, path)
    print(f"[ok]   {name}: inserted {len(cells)} cells after cell {idx}")


def main():
    splice(
        "phase3_sfm.ipynb",
        "Estimated poses saved",   # in the final save cell
        md(P3_INTRO), code(P3_CODE_UMEYAMA),
        code(P3_CODE_VIZ), code(P3_CODE_APPLY),
    )
    splice(
        "phase4_mvs.ipynb",
        "Why Phase 4 Uses Ground Truth Poses",
        md(P4_NOTE),
    )
    # Anchor inside the dense Chamfer visualization cell.  Inserting after
    # this cell guarantees `pcd_gt`, `eval_cameras`, and `metrics` are all
    # in scope for the alignment block below.
    splice(
        "phase8_evaluation.ipynb",
        "Dense Reconstruction Accuracy",
        md(P8_INTRO),
        code(P8_CODE),
    )


if __name__ == "__main__":
    main()