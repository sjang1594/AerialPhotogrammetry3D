"""
Aerial camera rig: 20 cameras on a grid at 50m altitude with 10° nadir tilt.
Software rasterizer (Z-buffer) replaces Open3D OffscreenRenderer for
Windows compatibility.  Saves images, GT poses, GT depth maps.
"""
import json
import os
import numpy as np
import open3d as o3d
import cv2


IMG_W, IMG_H = 800, 600
FOV_DEG      = 60.0


def _intrinsic_matrix(w=IMG_W, h=IMG_H, fov_deg=FOV_DEG):
    f = (w / 2.0) / np.tan(np.deg2rad(fov_deg / 2.0))
    return np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=float)


def _lookat(eye, target, up=np.array([0, 0, 1])):
    """Return 4×4 world-to-camera extrinsic (camera looks toward +Z in cam space)."""
    z = eye - target
    z = z / np.linalg.norm(z)
    # handle degenerate up
    if abs(np.dot(z, up)) > 0.999:
        up = np.array([0, 1, 0], dtype=float)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=0)   # 3×3
    t = -R @ eye
    E = np.eye(4)
    E[:3, :3] = R
    E[:3, 3]  = t
    return E


def generate_camera_poses(grid_size: int = 4, altitude: float = 50.0, tilt_deg: float = 10.0):
    """
    Returns list of dicts: {eye, R (3×3), t (3,), K (3×3)}
    Grid: grid_size × grid_size cameras spaced evenly over [20,80]×[20,80] m.
    Plus 4 diagonal corner cameras for overlap (total up to grid_size² + 4).
    """
    xs = np.linspace(20, 80, grid_size)
    ys = np.linspace(20, 80, grid_size)
    tilt_rad = np.deg2rad(tilt_deg)
    K = _intrinsic_matrix()

    cameras = []
    for gx in xs:
        for gy in ys:
            eye    = np.array([gx, gy, altitude])
            # slight forward tilt in Y direction
            target = np.array([gx, gy + altitude * np.sin(tilt_rad), 0.0])
            E      = _lookat(eye, target)
            R, t   = E[:3, :3], E[:3, 3]
            cameras.append({"eye": eye.tolist(), "R": R.tolist(), "t": t.tolist(), "K": K.tolist()})

    return cameras


# ---------------------------------------------------------------------------
# Software rasterizer
# ---------------------------------------------------------------------------

def _software_render(mesh: o3d.geometry.TriangleMesh,
                     R: np.ndarray, t: np.ndarray, K: np.ndarray,
                     W: int = IMG_W, H: int = IMG_H):
    """
    Z-buffer rasterizer for a vertex-colored TriangleMesh.

    Camera convention: looks down −Z (OpenGL / _lookat).
    Projection:  u = fx·X/(−Z) + cx,  v = fy·Y/(−Z) + cy  (Z < 0 in front).
    Depth output: positive linear depth along optical axis, 0 for sky pixels.

    Returns (color H×W×3 uint8 RGB, depth H×W float32 metres).
    """
    verts    = np.asarray(mesh.vertices,       dtype=np.float64)
    tris     = np.asarray(mesh.triangles)
    vcolors  = np.asarray(mesh.vertex_colors,  dtype=np.float64)
    vnormals = np.asarray(mesh.vertex_normals, dtype=np.float64)

    if vcolors.shape[0] == 0:
        vcolors = np.full((len(verts), 3), 0.7)

    # Lambertian shading so building faces have visible depth cues
    if vnormals.shape[0] > 0:
        light = np.array([0.3, -0.3, 1.0])
        light /= np.linalg.norm(light)
        diff = np.clip(vnormals @ light, 0.0, 1.0)
        vcolors = vcolors * (0.3 + 0.7 * diff[:, None])

    # World → camera space  (cam_z < 0 for in-front points)
    cam     = (R @ verts.T).T + t    # N×3
    depth_v = -cam[:, 2]             # positive depth per vertex

    # Perspective project
    pxv = K[0, 0] * cam[:, 0] / depth_v + K[0, 2]
    pyv = K[1, 1] * cam[:, 1] / depth_v + K[1, 2]

    sky = np.array([0.53, 0.81, 0.98])
    color_buf = np.tile(sky, (H, W, 1)).copy()
    depth_buf = np.full((H, W), np.inf, dtype=np.float64)

    for tri in tris:
        i0, i1, i2 = tri

        # Skip if any vertex is behind the camera
        if depth_v[i0] <= 0 or depth_v[i1] <= 0 or depth_v[i2] <= 0:
            continue

        p0 = np.array([pxv[i0], pyv[i0]])
        p1 = np.array([pxv[i1], pyv[i1]])
        p2 = np.array([pxv[i2], pyv[i2]])

        # Signed 2D area×2; positive = CCW = front-facing in image space
        area2 = (p1[0]-p0[0])*(p2[1]-p0[1]) - (p1[1]-p0[1])*(p2[0]-p0[0])
        if area2 <= 0:
            continue

        x0 = max(0,   int(np.floor(min(p0[0], p1[0], p2[0]))))
        x1 = min(W-1, int(np.ceil( max(p0[0], p1[0], p2[0]))))
        y0 = max(0,   int(np.floor(min(p0[1], p1[1], p2[1]))))
        y1 = min(H-1, int(np.ceil( max(p0[1], p1[1], p2[1]))))
        if x0 > x1 or y0 > y1:
            continue

        ys, xs = np.mgrid[y0:y1+1, x0:x1+1]
        pxf = xs.astype(np.float64) + 0.5
        pyf = ys.astype(np.float64) + 0.5

        # Edge functions; by convention:
        #   e01 → weight for vertex 2
        #   e12 → weight for vertex 0
        #   e20 → weight for vertex 1
        e01 = (p1[0]-p0[0])*(pyf-p0[1]) - (p1[1]-p0[1])*(pxf-p0[0])
        e12 = (p2[0]-p1[0])*(pyf-p1[1]) - (p2[1]-p1[1])*(pxf-p1[0])
        e20 = (p0[0]-p2[0])*(pyf-p2[1]) - (p0[1]-p2[1])*(pxf-p2[0])

        inside = (e01 >= 0) & (e12 >= 0) & (e20 >= 0)
        if not np.any(inside):
            continue

        inv = 1.0 / area2
        w0 = e12 * inv   # weight for vertex 0
        w1 = e20 * inv   # weight for vertex 1
        w2 = e01 * inv   # weight for vertex 2

        d_px   = w0*depth_v[i0] + w1*depth_v[i1] + w2*depth_v[i2]
        region = depth_buf[y0:y1+1, x0:x1+1]
        update = inside & (d_px < region)
        if not np.any(update):
            continue

        depth_buf[y0:y1+1, x0:x1+1][update] = d_px[update]
        for ch in range(3):
            c_px = w0*vcolors[i0, ch] + w1*vcolors[i1, ch] + w2*vcolors[i2, ch]
            buf_ch = color_buf[y0:y1+1, x0:x1+1, ch]
            buf_ch[update] = c_px[update]

    color_img = (color_buf * 255).clip(0, 255).astype(np.uint8)
    depth_img = depth_buf.copy()
    depth_img[np.isinf(depth_img)] = 0.0
    return color_img, depth_img.astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_scene(scene_mesh: o3d.geometry.TriangleMesh,
                 cameras: list,
                 out_dir: str):
    """
    Renders color + depth for each camera using the software rasterizer.
    Saves:
      out_dir/images/img_XXXX.png
      out_dir/GT_depth/depth_XXXX.npy
      out_dir/GT_poses.json
    """
    img_dir   = os.path.join(out_dir, "images")
    depth_dir = os.path.join(out_dir, "GT_depth")
    os.makedirs(img_dir,   exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    K = np.array(cameras[0]["K"])

    for i, cam in enumerate(cameras):
        R = np.array(cam["R"])
        t = np.array(cam["t"])
        color, depth = _software_render(scene_mesh, R, t, K, IMG_W, IMG_H)

        img_path   = os.path.join(img_dir,   f"img_{i:04d}.png")
        depth_path = os.path.join(depth_dir, f"depth_{i:04d}.npy")
        cv2.imwrite(img_path, cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        np.save(depth_path, depth)

    poses_path = os.path.join(out_dir, "GT_poses.json")
    with open(poses_path, "w") as f:
        json.dump(cameras, f, indent=2)

    print(f"[data_gen] Rendered {len(cameras)} images → {img_dir}")
    return img_dir, depth_dir, poses_path


def render_overview(scene_mesh: o3d.geometry.TriangleMesh,
                    cameras: list = None,
                    eye=(50.0, -40.0, 120.0),
                    target=(50.0, 50.0, 5.0),
                    fov_deg: float = 80.0,
                    w: int = 1200,
                    h: int = 900,
                    marker_size: float = 2.0,
                    marker_color=(1.0, 0.1, 0.1)):
    """
    Render a single wide-angle 3rd-person overview of the whole scene using the
    same software rasterizer.  If `cameras` is given, small colored cubes are
    added at each camera position so the rig is visible in the overview.

    Returns (color H×W×3 uint8 RGB, depth H×W float32 metres).
    """
    combined = o3d.geometry.TriangleMesh()
    combined += scene_mesh

    if cameras is not None:
        s = marker_size
        for cam in cameras:
            e = np.array(cam["eye"], dtype=float)
            marker = o3d.geometry.TriangleMesh.create_box(width=s, height=s, depth=s)
            marker.translate(e - np.array([s/2, s/2, s/2]))
            marker.paint_uniform_color(list(marker_color))
            marker.compute_vertex_normals()
            combined += marker

    combined.compute_vertex_normals()

    K = _intrinsic_matrix(w, h, fov_deg)
    E = _lookat(np.array(eye, dtype=float), np.array(target, dtype=float))
    R, t = E[:3, :3], E[:3, 3]

    return _software_render(combined, R, t, K, w, h)
