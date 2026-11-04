"""Render a splat-transform-cleaned PLY in PlayCanvas engine space (E = diag(-1,-1,1) * PLY) from a level camera
(+Y up) to confirm the scene is upright. Usage: render_upright_check.py IN.ply OUT.jpg --pos x y z --yaw deg [--fov 70]"""
import argparse, numpy as np, torch
from PIL import Image
from plyfile import PlyData
from gsplat import rasterization
ap = argparse.ArgumentParser(); ap.add_argument("ply"); ap.add_argument("out")
ap.add_argument("--pos", type=float, nargs=3, default=[0, 0, 0]); ap.add_argument("--yaws", type=float, nargs="+", default=[0, 90, 180, 270])
ap.add_argument("--fov", type=float, default=75); ap.add_argument("--w", type=int, default=640); ap.add_argument("--h", type=int, default=400)
a = ap.parse_args()
v = PlyData.read(a.ply)["vertex"]; g = lambda *k: np.stack([np.asarray(v[x], np.float32) for x in k], 1)
F = np.diag([-1, -1, 1]).astype(np.float32)
xyz = g("x", "y", "z") @ F.T
q = g("rot_0", "rot_1", "rot_2", "rot_3"); Rf = torch.tensor(F)
# rotate orientations by F (180 deg about Z): quaternion (w,x,y,z) -> (0,0,0,1) * q = (-z, -y, x, w)... use matrix form instead
from scipy.spatial.transform import Rotation as Rot
qr = Rot.from_quat(q[:, [1, 2, 3, 0]]); qe = (Rot.from_matrix(F) * qr).as_quat()[:, [3, 0, 1, 2]].astype(np.float32)
dc = g("f_dc_0", "f_dc_1", "f_dc_2")[:, None]
t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).cuda()
fx = a.w / 2 / np.tan(np.radians(a.fov) / 2); K = torch.tensor([[fx, 0, a.w / 2], [0, fx, a.h / 2], [0, 0, 1]], dtype=torch.float32).cuda()[None]
tiles = []
for yaw in a.yaws:
    th = np.radians(yaw); fwd = np.array([np.sin(th), 0, -np.cos(th)]); up = np.array([0, 1.0, 0]); right = np.cross(fwd, up)
    Rc = np.stack([right, -up, fwd])            # OpenCV camera: x right, y down, z forward
    w2c = np.eye(4); w2c[:3, :3] = Rc; w2c[:3, 3] = -Rc @ np.array(a.pos)
    with torch.no_grad():
        rgb, _, _ = rasterization(t(xyz), t(qe), torch.exp(t(g("scale_0", "scale_1", "scale_2"))),
                                  torch.sigmoid(t(np.asarray(v["opacity"], np.float32))), t(dc),
                                  torch.tensor(w2c, dtype=torch.float32).cuda()[None], K, a.w, a.h, sh_degree=0)
    tiles.append((rgb[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
Image.fromarray(np.concatenate(tiles, 1)).save(a.out, quality=88); print("saved", a.out)
