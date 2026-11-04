"""Work out how to make a 3DGS PLY upright (+Y up) and where to crop it, for splat-transform.
Up comes from the COLMAP cameras (--transforms) when given, else from a RANSAC ground plane of opaque splats
(sign: most opaque splats above the plane). Prints a rotation matrix, the centre, and a crop box in the
rotated, centred frame (robust percentiles of opaque splats, padded).
Usage: web_orient.py IN.ply [--transforms transforms.json] [--pct 1 99] [--pad 0.1]"""
import argparse, json, numpy as np
from plyfile import PlyData
ap = argparse.ArgumentParser(); ap.add_argument("ply"); ap.add_argument("--transforms")
ap.add_argument("--pct", type=float, nargs=2, default=[1, 99]); ap.add_argument("--pad", type=float, default=0.1)
a = ap.parse_args()
v = PlyData.read(a.ply)["vertex"]
xyz = np.stack([np.asarray(v[k], np.float64) for k in "xyz"], 1)
op = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))
P = xyz[op > 0.5]; rng = np.random.default_rng(0)
cams = None
if a.transforms:
    fr = json.load(open(a.transforms))["frames"]
    M = np.array([f["transform_matrix"] for f in fr])        # camera-to-world, OpenGL axes (y up in camera)
    cams = M[:, :3, 3]; up = M[:, :3, 1].mean(0)
    src = f"mean camera up of {len(fr)} cameras"
else:
    S = P[rng.choice(len(P), min(200000, len(P)), replace=False)]
    tol = 0.01 * np.linalg.norm(np.percentile(S, 95, 0) - np.percentile(S, 5, 0)); best = (0, None)
    for _ in range(3000):
        p = S[rng.choice(len(S), 3, replace=False)]; n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(n) < 1e-9: continue
        n /= np.linalg.norm(n); c = (np.abs((S - p[0]) @ n) < tol).sum()
        if c > best[0]: best = (c, (n, p[0]))
    n, p0 = best[1]; side = (S - p0) @ n
    up = n if (side > tol).sum() > (side < -tol).sum() else -n
    src = f"RANSAC ground plane, {best[0]/len(S):.0%} inliers (tol {tol:.3f})"
up /= np.linalg.norm(up)
# rotation taking `up` to +Y (Rodrigues), keeping the rest as close as possible
y = np.array([0, 1.0, 0]); vx = np.cross(up, y); s = np.linalg.norm(vx); c = up @ y
K = np.array([[0, -vx[2], vx[1]], [vx[2], 0, -vx[0]], [-vx[1], vx[0], 0]])
R = np.eye(3) + K + K @ K * ((1 - c) / (s * s)) if s > 1e-9 else (np.eye(3) if c > 0 else np.diag([1, -1, -1.0]))
centre = np.median(cams if cams is not None else P, 0)
Q = (P - centre) @ R.T
lo, hi = np.percentile(Q, a.pct[0], 0), np.percentile(Q, a.pct[1], 0); pad = a.pad * (hi - lo)
box = np.concatenate([lo - pad, hi + pad])
out = dict(count=int(len(xyz)), opaque=int(len(P)), up_source=src, up=up.round(5).tolist(), R=R.round(6).tolist(),
           centre=centre.round(4).tolist(), box=box.round(3).tolist(),
           inside_box=int((((xyz - centre) @ R.T >= box[:3]) & ((xyz - centre) @ R.T <= box[3:])).all(1).sum()))
if cams is not None:
    C = (cams - centre) @ R.T; out["cams_y_range"] = [round(float(C[:, 1].min()), 3), round(float(C[:, 1].max()), 3)]
print(json.dumps(out))
