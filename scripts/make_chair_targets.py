#!/usr/bin/env python
"""Per-frame training targets for a 2D shadow detector: the SDDNet mask everywhere, except on the hero object, where
the label comes from the 3D chair-shadow labels (detect_object_shadow.py) projected into the frame.

Final 3D label per object Gaussian: the brightness label, except near the terminator (|signed normal . sun| < --graze)
where brightness cannot tell grazing sun from shade, so the geometry label (attached | cast) decides.
Projection: front-most object Gaussians only (z-buffer, absolute depth tolerance). Object pixels = closed/dilated
coverage of the projected points; each object pixel takes the label of the nearest projected point. Pixels further
than --max-dist px from any point, and a --band px ring around the object outline, are IGNORE.
Target PNG values: 0 lit, 255 shadow, 128 ignore. Also writes a few check JPGs.

Usage:
  make_chair_targets.py --labels exports/IMG_6556_chairshadow_photo_s15/chair_shadow.npz --ply exports/IMG_6556/splat.ply \
      --colmap data/IMG_6556/colmap/sparse/0_refined --images data/IMG_6556/images --masks data/IMG_6556/shadow_masks \
      --out exports/IMG_6556_chair_targets [--every 2]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).parent))
from cluster_surfaces import read_ply
from fix_splat_shadows import project, qvec2rotmat, read_cameras_bin, read_images_bin

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--labels", type=Path, required=True)
ap.add_argument("--ply", type=Path, required=True)
ap.add_argument("--colmap", type=Path, required=True)
ap.add_argument("--images", type=Path, required=True)
ap.add_argument("--masks", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--every", type=int, default=2)
ap.add_argument("--graze", type=float, default=0.1)
ap.add_argument("--vis-tol", type=float, default=0.012)
ap.add_argument("--zbuf-radius", type=float, default=2.0)
ap.add_argument("--max-dist", type=float, default=6.0, help="px at half resolution")
ap.add_argument("--band", type=int, default=3, help="ignore ring around the object outline (px, half resolution)")
ap.add_argument("--checks", type=int, default=6)
ap.add_argument("--median", type=int, default=9, help="median filter size on the object labels (px, half res; 0 = off)")
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

lab = np.load(a.labels)
idx, sh_b, cos, geo = lab["idx"], lab["shadow"], lab["cos"], lab["geometry"]
graze = np.abs(cos) < a.graze
final = np.where(graze, geo, sh_b)
print(f"3D labels: {len(idx):,} object Gaussians, shadow {final.mean() * 100:.1f}% "
      f"(brightness {sh_b.mean() * 100:.1f}%, geometry decides {graze.mean() * 100:.1f}% near the terminator)", flush=True)

_, data = read_ply(a.ply)
xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
op = 1 / (1 + np.exp(-data["opacity"].astype(np.float64)))
P = xyz[idx]
occl = xyz[(op > 0.5) & (np.linalg.norm(xyz - P.mean(0), axis=1) < a.zbuf_radius)]
cams = read_cameras_bin(a.colmap / "cameras.bin")
frames = read_images_bin(a.colmap / "images.bin")[:: a.every]
stats = []
check_every = max(1, len(frames) // a.checks)
for k, fr in enumerate(frames):
    mpath = a.masks / (Path(fr["name"]).stem + ".png")
    if not mpath.exists():
        continue
    cam = cams[fr["cam"]]
    W, H = cam["width"], cam["height"]
    w, h = W // 2, H // 2  # work at half resolution
    R, t = qvec2rotmat(fr["q"]), fr["t"]
    zbuf = np.full(w * h, np.inf)
    oc = occl @ R.T + t
    oc = oc[oc[:, 2] > 1e-3]
    uo = project(cam, oc) / 2
    ok = (uo[:, 0] >= 0) & (uo[:, 0] < w) & (uo[:, 1] >= 0) & (uo[:, 1] < h)
    np.minimum.at(zbuf, uo[ok, 1].astype(np.int64) * w + uo[ok, 0].astype(np.int64), oc[ok, 2])
    pc = P @ R.T + t
    front = pc[:, 2] > 1e-3
    up_ = np.full((len(P), 2), -1.0)
    up_[front] = project(cam, pc[front]) / 2
    inb = front & (up_[:, 0] >= 0) & (up_[:, 0] < w) & (up_[:, 1] >= 0) & (up_[:, 1] < h)
    cell = np.clip(up_[:, 1].astype(np.int64) * w + up_[:, 0].astype(np.int64), 0, w * h - 1)
    vis = inb & (pc[:, 2] <= zbuf[cell] + a.vis_tol)
    target_full = (cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8) * 255
    if vis.sum() < 50:
        cv2.imwrite(str(a.out / mpath.name), target_full)
        continue
    px = up_[vis].astype(int)
    cover = np.zeros((h, w), np.uint8)
    lab_s = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    np.add.at(lab_s, (px[:, 1], px[:, 0]), final[vis].astype(np.float32))
    np.add.at(cnt, (px[:, 1], px[:, 0]), 1)
    cover[cnt > 0] = 1
    region = cv2.morphologyEx(cover, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    region = cv2.dilate(region, np.ones((3, 3), np.uint8)).astype(bool)
    dist, near = distance_transform_edt(cnt == 0, return_indices=True)
    plab = np.where(cnt > 0, lab_s / np.maximum(cnt, 1), 0)
    obj_lab = plab[near[0], near[1]] >= 0.5
    if a.median:  # nearest-point labels are speckled; a median filter keeps edges but removes isolated pixels
        obj_lab = cv2.medianBlur(obj_lab.astype(np.uint8) * 255, a.median) > 127
    ring = region & ~cv2.erode(region.astype(np.uint8), np.ones((2 * a.band + 1,) * 2, np.uint8)).astype(bool)
    ignore = region & ((dist > a.max_dist) | ring)
    t_half = cv2.resize(target_full, (w, h), interpolation=cv2.INTER_NEAREST)
    t_half[region] = np.where(obj_lab[region], 255, 0)
    t_half[ignore] = 128
    target = cv2.resize(t_half, (W, H), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(a.out / mpath.name), target)
    (a.out / "regions").mkdir(exist_ok=True)
    cv2.imwrite(str(a.out / "regions" / mpath.name),
                cv2.resize(region.astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_NEAREST))
    reg = region & ~ignore
    m0 = cv2.resize(target_full, (w, h), interpolation=cv2.INTER_NEAREST) > 127
    stats.append(dict(frame=fr["name"], object_px=int(reg.sum()), object_shadow=round(float(obj_lab[reg].mean()), 3),
                      sddnet_shadow_on_object=round(float(m0[reg].mean()), 3)))
    if k % check_every == 0:
        img = cv2.resize(cv2.imread(str(a.images / fr["name"])), (w, h))
        ov = img.copy()
        ov[t_half == 255] = (0.5 * ov[t_half == 255] + [0, 0, 128]).astype(np.uint8)
        ov[t_half == 128] = (0.5 * ov[t_half == 128] + [0, 128, 128]).astype(np.uint8)
        cv2.imwrite(str(a.out / f"check_{Path(fr['name']).stem}.jpg"), np.concatenate([img, ov], 1),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    if (k + 1) % 50 == 0:
        print(f"  {k + 1}/{len(frames)} frames", flush=True)
s_obj = np.mean([s["object_shadow"] for s in stats])
s_sdd = np.mean([s["sddnet_shadow_on_object"] for s in stats])
print(f"targets for {len(stats)} frames: on the object, new labels say {s_obj * 100:.1f}% shadow, SDDNet said "
      f"{s_sdd * 100:.1f}%", flush=True)
(a.out / "targets.json").write_text(json.dumps(dict(frames=stats, params={k2: str(v) for k2, v in vars(a).items()}),
                                               indent=1))
print(f"wrote {a.out}")
