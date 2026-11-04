#!/usr/bin/env python
"""Overlay stage-4 shadow flags on real photos: photo | predicted (red = shadow, cyan = lit) | SDDNet mask outline.

Only front-most object/ground Gaussians are drawn (z-buffer, tight depth tolerance). Read-only apart from the JPGs.
Usage: shadow_overlay.py --run exports/IMG_6556_shadow4_v2 --ply exports/IMG_6556/splat.ply \
          --colmap data/IMG_6556/colmap/sparse/0_refined --images data/IMG_6556/images --masks data/IMG_6556/shadow_masks \
          --frames frame_00263,frame_00317,frame_00540
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cluster_surfaces import read_ply
from fix_splat_shadows import project, qvec2rotmat, read_cameras_bin, read_images_bin

ap = argparse.ArgumentParser()
ap.add_argument("--run", type=Path, required=True)
ap.add_argument("--ply", type=Path, required=True)
ap.add_argument("--colmap", type=Path, required=True)
ap.add_argument("--images", type=Path, required=True)
ap.add_argument("--masks", type=Path, required=True)
ap.add_argument("--frames", required=True)
ap.add_argument("--vis-tol", type=float, default=0.012)
a = ap.parse_args()

_, data = read_ply(a.ply)
xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
if (a.run / "chair_shadow.npz").exists():  # detect_object_shadow.py output: object only, undecided = grazing colour
    f = np.load(a.run / "chair_shadow.npz")
    idx, shadow, graze = f["idx"], f["shadow"], f["undecided"]
else:
    f = np.load(a.run / "shadow_flags.npz")
    idx = np.r_[f["object_idx"], f["ground_idx"]]
    shadow = np.r_[f["attached"] | f["cast_object"], f["cast_ground"]]
    graze = (np.r_[f["grazing"], np.zeros(len(f["ground_idx"]), bool)] if "grazing" in f.files
             else np.zeros(len(idx), bool))
P = xyz[idx]
cams = read_cameras_bin(a.colmap / "cameras.bin")
ims = {im["name"]: im for im in read_images_bin(a.colmap / "images.bin")}
for stem in a.frames.split(","):
    im = ims[stem + ".png"]
    cam = cams[im["cam"]]
    w, h = cam["width"], cam["height"]
    R, t = qvec2rotmat(im["q"]), im["t"]
    pc = P @ R.T + t
    ok = pc[:, 2] > 1e-3
    uv = np.full((len(P), 2), -1.0)
    uv[ok] = project(cam, pc[ok])
    ok &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    s = 0.5
    zw, zh = int(w * s), int(h * s)
    cell = np.clip((uv[:, 1] * s).astype(int) * zw + (uv[:, 0] * s).astype(int), 0, zw * zh - 1)
    zbuf = np.full(zw * zh, np.inf)
    np.minimum.at(zbuf, cell[ok], pc[ok, 2])
    vis = ok & (pc[:, 2] <= zbuf[cell] + a.vis_tol)
    img = cv2.imread(str(a.images / im["name"]))
    mask = cv2.imread(str(a.masks / im["name"]), cv2.IMREAD_GRAYSCALE)
    if mask.shape != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    over = img.copy()
    for i in np.flatnonzero(vis):
        col = (40, 40, 230) if shadow[i] else (230, 220, 40)
        if graze[i]:
            col = (40, 160, 255) if shadow[i] else (200, 255, 120)
        cv2.circle(over, (int(uv[i, 0]), int(uv[i, 1])), 1, col, -1)
    msk = img.copy()
    cnts, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(msk, cnts, -1, (255, 0, 255), 2)
    pts = uv[vis]
    x0, y0 = np.maximum(pts.min(0).astype(int) - 30, 0)
    x1, y1 = np.minimum(pts.max(0).astype(int) + 30, [w, h])
    strip = np.concatenate([img[y0:y1, x0:x1], over[y0:y1, x0:x1], msk[y0:y1, x0:x1]], 1)
    for k, txt in enumerate(["photo", "predicted: red shadow, cyan lit (orange/green = grazing)", "SDDNet mask outline"]):
        cv2.putText(strip, txt, (k * (x1 - x0) + 8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    out = a.run / f"overlay_{stem}.jpg"
    cv2.imwrite(str(out), strip, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("wrote", out)
