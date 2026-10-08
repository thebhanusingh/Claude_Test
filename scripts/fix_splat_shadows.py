#!/usr/bin/env python
"""Brighten the splats that sit in cast shadows, using per-frame 2D shadow masks.

Single-lighting captures can't be split into albedo and shadow by inverse rendering
(GS-IR left the hard sun shadows in the albedo). This instead treats shadow as
something to detect and correct:

1. For every training frame with a mask (white = shadow), project each Gaussian centre
   into the frame, keep only the ones that are visible (coarse z-buffer of opaque
   Gaussians), and count how often a visible Gaussian lands in shadow.
2. shadow_frac = shadow votes / visible votes. Gaussians seen in at least --min-views
   frames and with shadow_frac >= --lo get brightened; the strength ramps to full
   at --hi.
3. The brightening gain (per RGB channel, linear light) is estimated from the photos:
   median ratio of lit to shadowed pixels in thin bands either side of each mask edge,
   which is roughly the sun/sky ratio for the same surface. --gain overrides it.

Only f_dc / f_rest change; geometry, opacity and every other PLY property are copied.
Works on any standard 3DGS PLY that lives in the COLMAP model's world coordinates
(true for our nerfstudio exports, see lab-notes "Problems found" item 2).

Usage:
  fix_splat_shadows.py --ply exports/IMG_6556/splat.ply \
      --colmap data/IMG_6556/colmap/sparse/0_refined --masks data/IMG_6556/shadow_masks \
      --out exports/IMG_6556_shadowfix [--every 2] [--gain 2.5] [--lo 0.3 --hi 0.7]

Outputs in --out: splat_shadowfix.ply, splat_shadowmask_debug.ply (shadowed splats
painted red), shadow_frac.npy, stats.json.
"""
import argparse
import json
import struct
import time
from pathlib import Path

import cv2
import numpy as np

SH_C0 = 0.28209479177387814

CAMERA_MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5), 4: ("OPENCV", 8)}


# ---------- COLMAP binary readers (subset of models we use) ----------

def read_cameras_bin(path):
    cams = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            cam_id, model_id, w, h = struct.unpack("<iiQQ", f.read(24))
            if model_id not in CAMERA_MODELS:
                raise SystemExit(f"ERROR: unsupported COLMAP camera model id {model_id}")
            name, nparams = CAMERA_MODELS[model_id]
            params = np.array(struct.unpack(f"<{nparams}d", f.read(8 * nparams)))
            cams[cam_id] = dict(model=name, width=w, height=h, params=params)
    return cams


def read_images_bin(path):
    imgs = []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            img_id, qw, qx, qy, qz, tx, ty, tz, cam_id = struct.unpack("<idddddddi", f.read(64))
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            (npts,) = struct.unpack("<Q", f.read(8))
            f.seek(24 * npts, 1)
            imgs.append(dict(id=img_id, q=np.array([qw, qx, qy, qz]), t=np.array([tx, ty, tz]), cam=cam_id, name=name.decode()))
    return sorted(imgs, key=lambda i: i["name"])


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def project(cam, xc):
    """Camera-space points (N,3) with z>0 -> pixel coords (N,2), applying lens distortion."""
    x, y = xc[:, 0] / xc[:, 2], xc[:, 1] / xc[:, 2]
    p, m = cam["params"], cam["model"]
    if m == "SIMPLE_PINHOLE":
        fx = fy = p[0]; cx, cy = p[1], p[2]
    elif m == "PINHOLE":
        fx, fy, cx, cy = p
    elif m in ("SIMPLE_RADIAL", "RADIAL"):
        fx = fy = p[0]; cx, cy = p[1], p[2]
        k1, k2 = p[3], (p[4] if m == "RADIAL" else 0.0)
        r2 = x * x + y * y
        s = 1 + k1 * r2 + k2 * r2 * r2
        x, y = x * s, y * s
    else:  # OPENCV
        fx, fy, cx, cy, k1, k2, p1, p2 = p
        r2 = x * x + y * y
        s = 1 + k1 * r2 + k2 * r2 * r2
        x, y = (x * s + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                y * s + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)
    return np.stack([fx * x + cx, fy * y + cy], 1)


# ---------- PLY ----------

def read_ply(path):
    with open(path, "rb") as f:
        header = []
        while (line := f.readline().decode().strip()) != "end_header":
            header.append(line)
        if "format binary_little_endian 1.0" not in header:
            raise SystemExit("ERROR: expected a binary little-endian PLY")
        n = next(int(l.split()[2]) for l in header if l.startswith("element vertex"))
        types = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4", "uint": "<u4"}
        dtype = [(l.split()[2], types[l.split()[1]]) for l in header if l.startswith("property")]
        data = np.frombuffer(f.read(n * np.dtype(dtype).itemsize), dtype=dtype, count=n).copy()
    return header, data


def write_ply(path, header, data):
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\nend_header\n").encode())
        f.write(data.tobytes())


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.maximum(c, 0), 1 / 2.4) - 0.055)


# ---------- main ----------

def fit_ground_plane(xyz, opacity, frames, tol, iters=2000, seed=0):
    """RANSAC plane through opaque Gaussians whose normal is close to the average camera up vector."""
    rng = np.random.default_rng(seed)
    # COLMAP cameras look down +z with +y pointing down in the image, so world up = -R^T [0,1,0]
    ups = np.array([-qvec2rotmat(f["q"])[1] for f in frames])
    up = ups.mean(0)
    up /= np.linalg.norm(up)
    pts = xyz[opacity > 0.5]
    if len(pts) > 300_000:
        pts = pts[rng.choice(len(pts), 300_000, replace=False)]
    best_n, best_d, best_count = None, None, -1
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n /= norm
        if abs(n @ up) < 0.8:
            continue
        d = -n @ a
        count = int((np.abs(pts @ n + d) < tol).sum())
        if count > best_count:
            best_n, best_d, best_count = n, d, count
    if best_n is None:
        raise SystemExit("ERROR: no ground plane found roughly perpendicular to the camera up direction")
    # refine with least squares on the inliers
    inl = pts[np.abs(pts @ best_n + best_d) < tol]
    centroid = inl.mean(0)
    n = np.linalg.svd(inl - centroid)[2][-1]
    n = n if n @ up > 0 else -n
    return n, -n @ centroid, len(inl) / len(pts), float(n @ up)


def edge_band_ratio(img, mask, band):
    """Median lit/shadow ratio (linear RGB) in thin bands on either side of the mask edge."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    inner = mask & ~cv2.erode(mask.astype(np.uint8), k).astype(bool)
    outer = cv2.dilate(mask.astype(np.uint8), k).astype(bool) & ~mask
    if inner.sum() < 200 or outer.sum() < 200:
        return None
    lin = srgb_to_linear(img.astype(np.float64) / 255.0)
    s, l = np.median(lin[inner], 0), np.median(lin[outer], 0)
    if np.any(s < 1e-3):
        return None
    return l / s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", required=True, type=Path)
    ap.add_argument("--colmap", required=True, type=Path, help="folder with cameras.bin and images.bin")
    ap.add_argument("--masks", required=True, type=Path, help="per-frame PNG masks named like the frames, white = shadow")
    ap.add_argument("--images", type=Path, help="frames folder, for the gain estimate (default: <colmap>/../../images)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--every", type=int, default=1, help="use every Nth frame")
    ap.add_argument("--zbuf-scale", type=float, default=0.25, help="z-buffer resolution relative to the frame")
    ap.add_argument("--z-tol", type=float, default=0.03, help="relative depth tolerance for visibility")
    ap.add_argument("--opaque", type=float, default=0.5, help="opacity above which a Gaussian occludes")
    ap.add_argument("--min-views", type=int, default=5)
    ap.add_argument("--lo", type=float, default=0.3, help="shadow_frac where brightening starts")
    ap.add_argument("--hi", type=float, default=0.7, help="shadow_frac where brightening is full")
    ap.add_argument("--gain", type=float, nargs="+", help="override gain: one value or three (R G B), linear light")
    ap.add_argument("--max-gain", type=float, default=6.0)
    ap.add_argument("--band", type=int, default=6, help="edge band width in pixels for the gain estimate")
    ap.add_argument("--ground-only", action="store_true",
                    help="only brighten Gaussians near the dominant ground plane (skips self-shading on objects, dark foliage, walls)")
    ap.add_argument("--ground-tol", type=float,
                    help="max distance from the ground plane, scene units (default: 2%% of the camera spread)")
    args = ap.parse_args()

    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    images_dir = args.images or args.colmap.parent.parent.parent / "images"
    header, ply = read_ply(args.ply)
    names = ply.dtype.names
    xyz = np.stack([ply["x"], ply["y"], ply["z"]], 1).astype(np.float64)
    opacity = 1 / (1 + np.exp(-ply["opacity"].astype(np.float64)))
    n = len(xyz)
    print(f"{n:,} Gaussians from {args.ply}")

    cams = read_cameras_bin(args.colmap / "cameras.bin")
    frames = read_images_bin(args.colmap / "images.bin")[:: args.every]
    shadow_votes = np.zeros(n, np.float32)
    seen_votes = np.zeros(n, np.int32)
    ratios, used = [], 0

    for i, fr in enumerate(frames):
        mpath = args.masks / (Path(fr["name"]).stem + ".png")
        if not mpath.exists():
            continue
        cam = cams[fr["cam"]]
        w, h = cam["width"], cam["height"]
        mask = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = mask > 127

        xc = xyz @ qvec2rotmat(fr["q"]).T + fr["t"]
        front = xc[:, 2] > 1e-3
        idx = np.nonzero(front)[0]
        uv = project(cam, xc[idx])
        inb = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        idx, uv, z = idx[inb], uv[inb], xc[idx[inb], 2]

        # coarse z-buffer from opaque Gaussians, then keep points near the front surface
        zw, zh = max(1, int(w * args.zbuf_scale)), max(1, int(h * args.zbuf_scale))
        cell = (uv[:, 1] * args.zbuf_scale).astype(np.int64) * zw + (uv[:, 0] * args.zbuf_scale).astype(np.int64)
        cell = np.clip(cell, 0, zw * zh - 1)
        zbuf = np.full(zw * zh, np.inf)
        op = opacity[idx] > args.opaque
        np.minimum.at(zbuf, cell[op], z[op])
        vis = z <= zbuf[cell] * (1 + args.z_tol)
        idx, uv = idx[vis], uv[vis]

        in_shadow = mask[uv[:, 1].astype(np.int64), uv[:, 0].astype(np.int64)]
        shadow_votes[idx] += in_shadow
        seen_votes[idx] += 1
        used += 1

        img_path = images_dir / fr["name"]
        if img_path.exists():
            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            if img.shape[:2] == (h, w):
                r = edge_band_ratio(img, mask, args.band)
                if r is not None:
                    ratios.append(r)
        if i % 25 == 0:
            print(f"  frame {i + 1}/{len(frames)} {fr['name']}: {vis.sum():,} visible, {in_shadow.sum():,} in shadow ({time.time() - t0:.0f}s)")

    if used == 0:
        raise SystemExit(f"ERROR: no masks matched frame names in {args.masks}")

    frac = np.where(seen_votes > 0, shadow_votes / np.maximum(seen_votes, 1), 0.0)
    valid = seen_votes >= args.min_views
    weight = np.clip((frac - args.lo) / max(args.hi - args.lo, 1e-6), 0, 1) * valid
    np.save(args.out / "shadow_frac.npy", frac.astype(np.float32))

    ground = None
    if args.ground_only:
        centres = np.array([-qvec2rotmat(f["q"]).T @ f["t"] for f in frames])
        spread = float(np.median(np.linalg.norm(centres - centres.mean(0), axis=1)))
        tol = args.ground_tol or 0.02 * spread
        n_g, d_g, inlier_share, align = fit_ground_plane(xyz, opacity, frames, tol)
        near = np.abs(xyz @ n_g + d_g) < tol
        weight *= near
        ground = dict(normal=n_g.round(4).tolist(), offset=round(float(d_g), 4), tol=round(tol, 4),
                      camera_spread=round(spread, 4), inlier_share=round(inlier_share, 3),
                      up_alignment=round(align, 3), gaussians_near=int(near.sum()))
        print(f"ground plane: normal {ground['normal']}, tol {tol:.4f}, {near.sum():,} Gaussians near it "
              f"({inlier_share:.0%} of opaque sample), up alignment {align:.3f}")

    if args.gain:
        gain = np.array(args.gain * 3 if len(args.gain) == 1 else args.gain, float)
        gain_src = "override"
    elif ratios:
        gain = np.median(np.array(ratios), 0)
        gain_src = f"median of {len(ratios)} frames' mask-edge ratios"
    else:
        raise SystemExit("ERROR: no gain estimate (no frames found or masks too small); pass --gain")
    gain = np.clip(gain, 1.0, args.max_gain)
    print(f"gain (linear RGB) = {np.round(gain, 3).tolist()} [{gain_src}]")

    # recolour: scale DC colour in linear light, scale view-dependent SH by the same per-channel factor
    dc = np.stack([ply[f"f_dc_{c}"] for c in range(3)], 1).astype(np.float64)
    rgb = np.clip(dc * SH_C0 + 0.5, 0, 1)
    lin = srgb_to_linear(rgb) * (1 + weight[:, None] * (gain - 1))
    new_rgb = np.clip(linear_to_srgb(np.clip(lin, 0, 1)), 0, 1)
    factor = np.where(rgb > 1e-3, new_rgb / np.maximum(rgb, 1e-3), 1.0)
    fixed = ply.copy()
    for c in range(3):
        fixed[f"f_dc_{c}"] = ((new_rgb[:, c] - 0.5) / SH_C0).astype(np.float32)
    rest = sorted((k for k in names if k.startswith("f_rest_")), key=lambda k: int(k.split("_")[-1]))
    per_ch = len(rest) // 3  # f_rest is channel-major: all R coeffs, then G, then B
    for j, k in enumerate(rest):
        fixed[k] = (ply[k] * factor[:, min(j // max(per_ch, 1), 2)]).astype(np.float32)
    write_ply(args.out / "splat_shadowfix.ply", header, fixed)

    debug = ply.copy()
    red = np.array([1.0, 0.0, 0.0])
    dbg_rgb = rgb * (1 - weight[:, None]) + red * weight[:, None]
    for c in range(3):
        debug[f"f_dc_{c}"] = ((dbg_rgb[:, c] - 0.5) / SH_C0).astype(np.float32)
    for k in rest:
        debug[k] = (ply[k] * (1 - weight)).astype(np.float32)
    write_ply(args.out / "splat_shadowmask_debug.ply", header, debug)

    stats = dict(
        ply=str(args.ply), colmap=str(args.colmap), masks=str(args.masks), frames_used=used,
        gaussians=int(n), seen_min_views=int(valid.sum()), brightened_any=int((weight > 0).sum()),
        brightened_full=int((weight >= 1).sum()), gain_linear_rgb=gain.round(4).tolist(), gain_source=gain_src,
        gain_frames=len(ratios), ground_plane=ground, params={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        seconds=round(time.time() - t0, 1))
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "params"}, indent=2))


if __name__ == "__main__":
    main()
