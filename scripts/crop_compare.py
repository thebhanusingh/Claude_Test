#!/usr/bin/env python
"""Side-by-side crops of one part of the object (default: the top, i.e. bear 2's hat) on held-out frames:
photo | splat A | splat B, from the photo | render images written by ns-eval (A) and train_2dgs_ref.py (B).

The crop centre is the footprint centre lifted --at x the object height along the ground-plane normal, projected
into each held-out camera (from the splatfacto config, undistorted K as in training); the crop side is
--size x the object height, projected.

Usage: crop_compare.py --config outputs/IMG_6705/splatfacto/<run>/config.yml --object exports/IMG_6705_object/object.json
           --plane-from exports/IMG_6705_shadowfix_o/stats.json
           --a exports/IMG_6705_2dgs_compare/renders_3dgs --b exports/IMG_6705_2dgs_fromsplat/eval
           --out exports/IMG_6705_2dgs_compare/hat_crops.jpg
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from train_2dgs_ref import intrinsics, viewmat


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--object", type=Path, required=True)
    ap.add_argument("--plane-from", type=Path, required=True)
    ap.add_argument("--a", type=Path, required=True, help="ns-eval renders (eval_img_0000.png = photo | render)")
    ap.add_argument("--b", type=Path, required=True, help="train_2dgs_ref.py eval dir (eval_000.jpg = photo | render)")
    ap.add_argument("--labels", nargs=3, default=["photo", "3DGS", "2DGS"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--at", type=float, default=0.85, help="crop centre height, fraction of the object height")
    ap.add_argument("--size", type=float, default=0.45, help="crop side, fraction of the object height")
    ap.add_argument("--frames", type=int, default=4, help="held-out frames shown, spread over the orbit")
    ap.add_argument("--tile", type=int, default=360)
    a = ap.parse_args()

    obj = json.loads(a.object.read_text())
    gp = json.loads(a.plane_from.read_text())["ground_plane"]
    up = np.array(gp["normal"], float)
    up /= np.linalg.norm(up)
    centre = np.array(obj["footprint_centre"]) + a.at * obj["height"] * up
    config = yaml.load(a.config.read_text(), Loader=yaml.Loader)
    config.pipeline.datamanager.cache_images = "cpu"
    dm = config.pipeline.datamanager.setup(device="cpu", test_mode="val", world_size=1, local_rank=0)
    _ = dm.cached_eval  # undistorts and updates K
    ec = dm.eval_dataset.cameras

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
    sheet = Image.new("RGB", (3 * T, len(pick) * (T + 22) + 22), "white")
    d = ImageDraw.Draw(sheet)
    for c, lab in enumerate(a.labels):
        d.text((c * T + 6, 4), lab, fill="black")
    for r, (j, u, v, half) in enumerate(pick):
        A = Image.open(a.a / f"eval_img_{j:04d}.png").convert("RGB")
        B = Image.open(a.b / f"eval_{j:03d}.jpg").convert("RGB")
        w = A.width // 2
        box = lambda x0: (int(x0 + u - half), int(v - half), int(x0 + u + half), int(v + half))
        tiles = [A.crop(box(0)), A.crop(box(w)), B.crop(box(B.width // 2))]
        y = 22 + r * (T + 22)
        for c, t in enumerate(tiles):
            sheet.paste(t.resize((T, T), Image.LANCZOS), (c * T, y))
        d.text((6, y + T + 4), f"held-out frame {j}", fill="black")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(a.out, quality=92)
    print(f"{len(cands)} held-out frames see the crop; wrote {len(pick)} rows -> {a.out}")


if __name__ == "__main__":
    main()
