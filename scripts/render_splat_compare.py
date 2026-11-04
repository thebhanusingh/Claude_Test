"""Render several 3DGS PLYs from COLMAP cameras with gsplat and save photo | ply1 | ply2 | ... strips.
Usage: render_splat_compare.py --colmap MODEL --images DIR --out DIR --frames a,b,c --plys A.ply B.ply ... [--labels ...] [--scale 0.5]
Distortion (OPENCV k1..p2) is ignored, so edges of the frame can be slightly off versus the photo."""
import argparse, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw
from plyfile import PlyData
from gsplat import rasterization
sys.path.insert(0, str(Path(__file__).parent))
from fix_splat_shadows import read_cameras_bin, read_images_bin

def load_ply(p):
    v = PlyData.read(str(p))["vertex"]; names = v.data.dtype.names
    g = lambda *k: np.stack([np.asarray(v[x], np.float32) for x in k], 1)
    rest = sorted([n for n in names if n.startswith("f_rest_")], key=lambda s: int(s.split("_")[-1]))
    dc = g("f_dc_0", "f_dc_1", "f_dc_2")[:, None]
    sh = np.concatenate([dc, g(*rest).reshape(len(dc), 3, -1).transpose(0, 2, 1)], 1) if rest else dc
    t = lambda a: torch.from_numpy(a).cuda()
    q = t(g("rot_0", "rot_1", "rot_2", "rot_3")); q = q / q.norm(dim=1, keepdim=True)
    return dict(means=t(g("x", "y", "z")), quats=q, scales=torch.exp(t(g("scale_0", "scale_1", "scale_2"))),
                opacities=torch.sigmoid(t(np.asarray(v["opacity"], np.float32))), colors=t(sh),
                sh_degree=int(round(np.sqrt(sh.shape[1]))) - 1)

def qvec2rot(q):
    w, x, y, z = q
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y], [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x], [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]])

ap = argparse.ArgumentParser()
ap.add_argument("--colmap", type=Path, required=True); ap.add_argument("--images", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True); ap.add_argument("--frames", required=True)
ap.add_argument("--plys", nargs="+", type=Path, required=True); ap.add_argument("--labels", nargs="+")
ap.add_argument("--scale", type=float, default=0.5)
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
cams = read_cameras_bin(a.colmap / "cameras.bin"); ims = read_images_bin(a.colmap / "images.bin")
byname = {im["name"]: im for im in (ims.values() if isinstance(ims, dict) else ims)}
labels = a.labels or [p.stem for p in a.plys]
splats = [load_ply(p) for p in a.plys]
for stem in a.frames.split(","):
    im = byname[stem + ".png"]; c = cams[im["cam"]]
    W, H = int(c["width"] * a.scale), int(c["height"] * a.scale); fx, fy, cx, cy = (c["params"][:4] * a.scale)
    w2c = np.eye(4); w2c[:3, :3] = qvec2rot(im["q"]); w2c[:3, 3] = im["t"]
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32).cuda()[None]
    vm = torch.tensor(w2c, dtype=torch.float32).cuda()[None]
    tiles = [np.asarray(Image.open(a.images / im["name"]).convert("RGB").resize((W, H), Image.LANCZOS))]
    for s in splats:
        with torch.no_grad():
            rgb, _, _ = rasterization(s["means"], s["quats"], s["scales"], s["opacities"], s["colors"], vm, K, W, H, sh_degree=s["sh_degree"])
        tiles.append((rgb[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    strip = Image.fromarray(np.concatenate(tiles, 1)); d = ImageDraw.Draw(strip)
    for i, t in enumerate(["photo"] + labels): d.text((i * W + 8, 8), t, fill=(255, 255, 0))
    strip.save(a.out / f"compare_{stem}.jpg", quality=90); print("saved", stem)
