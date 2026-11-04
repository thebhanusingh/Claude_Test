#!/usr/bin/env python
"""Orbit renders around the hero object at full quality (all SH bands, gsplat), splats side by side.

The orbit circles the object's crop centre around the ground-plane normal at --elev degrees above the ground and
--dist x the median COLMAP camera distance, starting from the azimuth of a real camera; focal length from the COLMAP
camera. Writes <out>/orbit_NNN.jpg (labelled strips) and, with --video, <out>/orbit.mp4 (OpenCV, H.264 if available).

Usage: render_orbit.py --colmap data/IMG_6705/colmap/sparse/0_refined --object exports/IMG_6705_object/object.json
           --plane-from exports/IMG_6705_shadowfix_o/stats.json --plys A.ply B.ply --labels original "new final"
           --out exports/IMG_6705_orbit [--views 36] [--elev 25] [--dist 0.6] [--size 1280 960] [--video]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from gsplat import rasterization
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from fix_splat_shadows import qvec2rotmat, read_cameras_bin, read_images_bin


def load_ply(p):  # as in render_splat_compare.py (a flat script, so not importable)
    from plyfile import PlyData
    v = PlyData.read(str(p))["vertex"]
    g = lambda *k: np.stack([np.asarray(v[x], np.float32) for x in k], 1)
    rest = sorted([n for n in v.data.dtype.names if n.startswith("f_rest_")], key=lambda s: int(s.split("_")[-1]))
    dc = g("f_dc_0", "f_dc_1", "f_dc_2")[:, None]
    sh = np.concatenate([dc, g(*rest).reshape(len(dc), 3, -1).transpose(0, 2, 1)], 1) if rest else dc
    t = lambda a: torch.from_numpy(a).cuda()
    q = t(g("rot_0", "rot_1", "rot_2", "rot_3"))
    return dict(means=t(g("x", "y", "z")), quats=q / q.norm(dim=1, keepdim=True),
                scales=torch.exp(t(g("scale_0", "scale_1", "scale_2"))),
                opacities=torch.sigmoid(t(np.asarray(v["opacity"], np.float32))), colors=t(sh),
                sh_degree=int(round(np.sqrt(sh.shape[1]))) - 1)


def look_at(pos, target, up):
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(-up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    w2c = np.eye(4)
    w2c[:3, :3] = np.stack([x, y, z])
    w2c[:3, 3] = -w2c[:3, :3] @ pos
    return w2c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--colmap", type=Path, required=True)
    ap.add_argument("--object", type=Path, required=True)
    ap.add_argument("--plane-from", type=Path, required=True)
    ap.add_argument("--plys", nargs="+", type=Path, required=True)
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--views", type=int, default=36)
    ap.add_argument("--elev", type=float, default=25.0)
    ap.add_argument("--dist", type=float, default=0.6, help="x median COLMAP camera distance to the object")
    ap.add_argument("--size", type=int, nargs=2, default=[1280, 960])
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--fps", type=int, default=12)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    obj = json.loads(a.object.read_text())
    gp = json.loads(a.plane_from.read_text())["ground_plane"]
    up = np.array(gp["normal"], float)
    up /= np.linalg.norm(up)
    target = np.array(obj["crop"][:3], float)
    cams = read_cameras_bin(a.colmap / "cameras.bin")
    ims = read_images_bin(a.colmap / "images.bin")
    ims = list(ims.values()) if isinstance(ims, dict) else ims
    C = np.array([-qvec2rotmat(f["q"]).T @ f["t"] for f in ims])
    dist = a.dist * np.median(np.linalg.norm(C - target, axis=1))
    cam = cams[ims[0]["cam"]]
    W, H = a.size
    f = cam["params"][0] * W / cam["width"]
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=torch.float32).cuda()[None]
    e1 = C[0] - target
    e1 -= (e1 @ up) * up
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    labels = a.labels or [p.stem for p in a.plys]
    splats = [load_ply(p) for p in a.plys]
    el = np.radians(a.elev)
    writer = None
    for k in range(a.views):
        az = 2 * np.pi * k / a.views
        pos = target + dist * (np.cos(el) * (np.cos(az) * e1 + np.sin(az) * e2) + np.sin(el) * up)
        vm = torch.tensor(look_at(pos, target, up), dtype=torch.float32).cuda()[None]
        tiles = []
        for s in splats:
            with torch.no_grad():
                rgb, _, _ = rasterization(s["means"], s["quats"], s["scales"], s["opacities"], s["colors"], vm, K, W, H,
                                          sh_degree=s["sh_degree"])
            tiles.append((rgb[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
        strip = Image.fromarray(np.concatenate(tiles, 1))
        d = ImageDraw.Draw(strip)
        for i, t in enumerate(labels):
            d.text((i * W + 10, 10), t, fill=(255, 255, 0))
        strip.save(a.out / f"orbit_{k:03d}.jpg", quality=93)
        if a.video:
            import cv2
            if writer is None:
                for cc in ("avc1", "mp4v"):
                    writer = cv2.VideoWriter(str(a.out / "orbit.mp4"), cv2.VideoWriter_fourcc(*cc), a.fps, strip.size)
                    if writer.isOpened():
                        print(f"video codec {cc}")
                        break
            writer.write(cv2.cvtColor(np.asarray(strip), cv2.COLOR_RGB2BGR))
    if writer is not None:
        writer.release()
    print(f"{a.views} views, {W}x{H} per splat, distance {dist:.2f} units, elevation {a.elev} deg -> {a.out}")


if __name__ == "__main__":
    main()
