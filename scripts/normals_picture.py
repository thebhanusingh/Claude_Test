#!/usr/bin/env python
"""Picture of surface directions on held-out frames: photo | 3DGS blobs' own shortest axis | 2DGS surfel normal.

Each splat is rendered with its normals as colour (x, y, z of the world direction -> red, green, blue). For a fair
look both are turned to face the camera of that frame (a normal and its opposite are the same surface), so a smooth
surface should come out as smooth colour and noise as speckle. Cropped on the object like crop_compare.py.

Usage: normals_picture.py --config outputs/IMG_6705/splatfacto/<run>/config.yml --ply exports/IMG_6705/splat.ply
           --surfels exports/IMG_6705_2dgs_fromsplat/surfels_params.pt --object exports/IMG_6705_object/object.json
           --plane-from exports/IMG_6705_shadowfix_o/stats.json --out exports/IMG_6705_2dgs_compare/normals_check.jpg
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from gsplat import rasterization, rasterization_2dgs
from PIL import Image, ImageDraw

from train_2dgs_ref import flatten_splat, intrinsics, surfel_normals, viewmat


def shortest_axis(quats, log_scales):
    q = torch.nn.functional.normalize(quats, dim=1)
    w, x, y, z = q.unbind(1)
    R = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).view(-1, 3, 3)
    return R[torch.arange(len(R)), :, log_scales.argmin(1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ply", type=Path, required=True, help="3DGS splat (ns-export)")
    ap.add_argument("--surfels", type=Path, required=True, help="surfels_params.pt from train_2dgs_ref.py")
    ap.add_argument("--object", type=Path, required=True)
    ap.add_argument("--plane-from", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--size", type=float, default=1.3, help="crop side, fraction of the object height")
    ap.add_argument("--tile", type=int, default=420)
    a = ap.parse_args()
    dev = "cuda"

    obj = json.loads(a.object.read_text())
    gp = json.loads(a.plane_from.read_text())["ground_plane"]
    up = np.array(gp["normal"], float)
    up /= np.linalg.norm(up)
    centre = np.array(obj["footprint_centre"]) + 0.5 * obj["height"] * up
    config = yaml.load(a.config.read_text(), Loader=yaml.Loader)
    dm = config.pipeline.datamanager.setup(device="cpu", test_mode="val", world_size=1, local_rank=0)
    held = dm.cached_eval
    ec = dm.eval_dataset.cameras

    g = {k: v.to(dev) for k, v in flatten_splat(a.ply).items()}
    g_n = shortest_axis(g["quats"], g["scales"])  # flatten_splat keeps log scales, sorted longest first
    s = {k: v.to(dev) for k, v in torch.load(a.surfels).items()}
    s_n = surfel_normals(s["quats"])

    cands = []
    for j in range(len(ec)):
        K, W, H = intrinsics(ec, j)
        vm = viewmat(ec.camera_to_worlds[j]).numpy()
        pc = vm[:3, :3] @ centre + vm[:3, 3]
        u, v = K[0, 0].item() * pc[0] / pc[2] + K[0, 2].item(), K[1, 1].item() * pc[1] / pc[2] + K[1, 2].item()
        half = 0.5 * a.size * obj["height"] * K[0, 0].item() / pc[2]
        if pc[2] > 0 and half < u < W - half and half < v < H - half:
            cands.append((j, u, v, half))
    pick = [cands[i] for i in np.linspace(0, len(cands) - 1, min(a.frames, len(cands))).astype(int)]

    T = a.tile
    labels = ["photo", "3DGS own axis", "2DGS surfel normal"]
    sheet = Image.new("RGB", (3 * T, len(pick) * (T + 22) + 22), "white")
    d = ImageDraw.Draw(sheet)
    for c, lab in enumerate(labels):
        d.text((c * T + 6, 4), lab, fill="black")
    with torch.no_grad():
        for r, (j, u, v, half) in enumerate(pick):
            K, W, H = intrinsics(ec, j)
            K = K.to(dev)
            vm = viewmat(ec.camera_to_worlds[j].to(dev))
            cam = torch.linalg.inv(vm)[:3, 3]
            tiles = []
            img = held[j]["image"]
            img = img.float() / 255 if img.dtype == torch.uint8 else img
            tiles.append(img[..., :3].numpy())
            for p, n, raster in ((g, g_n, rasterization), (s, s_n, rasterization_2dgs)):
                n = n * torch.where(((cam - p["means"]) * n).sum(1, keepdim=True) < 0, -1.0, 1.0)
                out = raster(means=p["means"], quats=p["quats"], scales=torch.exp(p["scales"]),
                             opacities=torch.sigmoid(p["opacities"]), colors=n * 0.5 + 0.5, viewmats=vm[None],
                             Ks=K[None], width=W, height=H, backgrounds=torch.ones(1, 3, device=dev))
                tiles.append(out[0][0, ..., :3].clamp(0, 1).cpu().numpy())
            y = 22 + r * (T + 22)
            box = (int(u - half), int(v - half), int(u + half), int(v + half))
            for c, t in enumerate(tiles):
                im = Image.fromarray((t * 255).astype(np.uint8)).crop(box).resize((T, T), Image.LANCZOS)
                sheet.paste(im, (c * T, y))
            d.text((6, y + T + 4), f"held-out frame {j}", fill="black")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(a.out, quality=92)
    print(f"wrote {len(pick)} rows -> {a.out}")


if __name__ == "__main__":
    main()
