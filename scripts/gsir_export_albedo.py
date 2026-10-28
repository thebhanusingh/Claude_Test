"""Export a GS-IR checkpoint as a standard 3DGS PLY whose colour is the learned albedo.
f_dc = (srgb(albedo) - 0.5) / C0, f_rest = 0, only standard 3DGS fields."""
import sys, numpy as np, torch
from plyfile import PlyData, PlyElement
ckpt, out = sys.argv[1], sys.argv[2]
gamma = "--no-gamma" not in sys.argv
g = torch.load(ckpt, map_location="cpu")["gaussians"]
_, xyz, f_dc, f_rest, scale, rot, opac, normal, albedo = g[:9]
alb = torch.sigmoid(albedo.detach()).clamp(0, 1)
if gamma:  # stage 2 ran with --gamma: albedo is linear, convert to sRGB for display
    alb = torch.where(alb <= 0.0031308, 12.92 * alb, 1.055 * alb.clamp_min(1e-8) ** (1 / 2.4) - 0.055)
C0 = 0.28209479177387814
dc = ((alb - 0.5) / C0).numpy()
n = xyz.shape[0]; n_rest = f_rest.shape[1] * 3
cols = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2], "nx": 0 * xyz[:, 0], "ny": 0 * xyz[:, 0], "nz": 0 * xyz[:, 0]}
cols = {k: v.detach().numpy() for k, v in cols.items()}
for i in range(3): cols[f"f_dc_{i}"] = dc[:, i]
for i in range(n_rest): cols[f"f_rest_{i}"] = np.zeros(n, np.float32)
cols["opacity"] = opac[:, 0].detach().numpy()
for i in range(3): cols[f"scale_{i}"] = scale[:, i].detach().numpy()
for i in range(4): cols[f"rot_{i}"] = rot[:, i].detach().numpy()
el = np.empty(n, dtype=[(k, "f4") for k in cols])
for k, v in cols.items(): el[k] = v
PlyData([PlyElement.describe(el, "vertex")]).write(out)
print(f"wrote {out}: {n} Gaussians, gamma={gamma}")
