"""Downsample a 3DGS PLY to the compact antimatter15 .splat format (32 B/splat: xyz f32, scale f32x3, rgba u8, quat u8 w,x,y,z).
Keeps the N most visible splats (opacity x volume) within --radius of the camera centroid. View-dependent SH is dropped.
Usage: ply_to_splat.py IN.ply OUT.splat [--n 300000] [--radius 12] [--center x y z]"""
import argparse, numpy as np
from plyfile import PlyData
ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out")
ap.add_argument("--n", type=int, default=300000); ap.add_argument("--radius", type=float, default=12.0)
ap.add_argument("--center", type=float, nargs=3, default=[0, 0, 0])
a = ap.parse_args()
v = PlyData.read(a.inp)["vertex"]; f = lambda k: np.asarray(v[k], np.float32)
xyz = np.stack([f("x"), f("y"), f("z")], 1); sc = np.exp(np.stack([f(f"scale_{i}") for i in range(3)], 1))
op = 1 / (1 + np.exp(-f("opacity"))); dc = np.stack([f(f"f_dc_{i}") for i in range(3)], 1)
q = np.stack([f(f"rot_{i}") for i in range(4)], 1); q /= np.linalg.norm(q, axis=1, keepdims=True)
keep = np.linalg.norm(xyz - np.array(a.center, np.float32), axis=1) < a.radius
score = np.where(keep, op * sc.prod(1), -1)
idx = np.argsort(-score)[: min(a.n, int(keep.sum()))]
rgb = np.clip(0.5 + 0.28209479177387814 * dc[idx], 0, 1)
out = np.zeros(len(idx), dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
out["p"] = xyz[idx]; out["s"] = sc[idx]
out["c"] = np.concatenate([rgb, op[idx, None]], 1) * 255
out["r"] = np.clip(q[idx] * 128 + 128, 0, 255)
out.tofile(a.out); print(f"{a.inp}: kept {len(idx):,} of {len(xyz):,} -> {a.out} ({out.nbytes/1e6:.1f} MB)")
