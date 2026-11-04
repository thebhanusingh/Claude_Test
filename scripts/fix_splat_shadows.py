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
4. --fill texture (ground only): instead of brightening, copy colour and SH from sunlit
   ground splats at one in-plane shift (clone stamp), chosen as the shortest shift that
   lands the shadow patch on lit ground. A gain cannot bring back texture the shadow never
   recorded (5-8% of sunlit on concrete); copying keeps the grain. Gain is the fallback.

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
    n = np.linalg.svd(inl - centroid, full_matrices=False)[2][-1]
    n = n if n @ up > 0 else -n
    return n, -n @ centroid, len(inl) / len(pts), float(n @ up)


def edge_band_ratio(img, mask, band, gap):
    """Median lit/shadow ratio (linear RGB) in thin bands either side of the mask edge, skipping
    `gap` pixels next to the edge so soft penumbra doesn't pull the ratio down."""
    def disk(r):
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    m8 = mask.astype(np.uint8)
    inner = cv2.erode(m8, disk(gap)).astype(bool) & ~cv2.erode(m8, disk(gap + band)).astype(bool) if gap else \
        mask & ~cv2.erode(m8, disk(band)).astype(bool)
    outer = cv2.dilate(m8, disk(gap + band)).astype(bool) & ~cv2.dilate(m8, disk(gap)).astype(bool) if gap else \
        cv2.dilate(m8, disk(band)).astype(bool) & ~mask
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
    ap.add_argument("--colour", choices=["lum", "rgb"], default="lum",
                    help="lum: one gain for R, G and B (keeps each splat's hue); rgb: per-channel gains from the photos")
    ap.add_argument("--mode", choices=["target", "scale"], default="target",
                    help="target: a splat is never brightened past the lit ground median luminance, so partly shadowed "
                         "and edge splats get less gain; scale: plain multiply by the gain")
    ap.add_argument("--colour-gate", type=float, default=0.0,
                    help="with --ground-only: only brighten splats whose rg chromaticity is within this distance of the "
                         "median of the shadowed ground (keeps non-ground objects such as chair legs out). 0 = off")
    ap.add_argument("--ceiling-pct", type=float, default=50.0,
                    help="target mode: ceiling = this percentile of lit ground luminance (50 = median)")
    ap.add_argument("--band", type=int, default=6, help="edge band width in pixels for the gain estimate")
    ap.add_argument("--band-gap", type=int, default=8, help="pixels skipped either side of the mask edge (penumbra)")
    ap.add_argument("--ground-only", action="store_true",
                    help="only brighten Gaussians near the dominant ground plane (skips self-shading on objects, dark foliage, walls)")
    ap.add_argument("--ground-tol", type=float,
                    help="max distance from the ground plane, scene units (default: 2%% of the camera spread)")
    ap.add_argument("--ground-spread", type=float, default=0.0,
                    help="with --ground-only: give each ground Gaussian the max shadow weight of ground neighbours within "
                         "this radius in the plane (scene units), so lower grass layers hidden from the z-buffer follow "
                         "the visible top layer. Only dark neighbours receive weight (see --dark-gate). 0 = off")
    ap.add_argument("--dark-gate", action="store_true",
                    help="with --ground-only: also scale the vote weights by how dark each Gaussian is between the lit and "
                         "shadowed ground medians, so sunlit splats near a mask edge are left alone; lets --lo/--hi go lower")
    ap.add_argument("--fill", choices=["gain", "texture"], default="gain",
                    help="gain: brighten each shadowed splat by the gain; texture (needs --ground-only): clone stamp, copy "
                         "colour and SH from the sunlit ground splat at one in-plane shift that moves the shadow patch onto "
                         "nearby lit ground, so the concrete keeps a texture; splats with no lit source fall back to gain")
    ap.add_argument("--tex-radius", type=float, default=0.1,
                    help="texture fill: a shifted position must have a lit ground splat within this radius (scene units)")
    ap.add_argument("--tex-step", type=float, default=0.25, help="texture fill: shift lengths are tried in this step")
    ap.add_argument("--tex-max-shift", type=float, default=6.0, help="texture fill: longest shift tried (scene units)")
    ap.add_argument("--tex-lit", type=float, default=0.1, help="texture fill: sources need shadow_frac below this")
    ap.add_argument("--tex-cover", type=float, default=0.97,
                    help="texture fill: use the shortest shift that covers at least this share of the best coverage")
    ap.add_argument("--tex-feather", type=float, default=0.0,
                    help="texture fill: also fill ground splats within this distance outside the patch (scene units) that "
                         "were seen in shadow at least once and are darker than their copy, fading out with distance, and "
                         "fill edge splats by how much darker they are than the copy rather than by vote weight. 0 = off")
    ap.add_argument("--tex-match", type=int, default=0,
                    help="texture fill: scale each copy by the mean colour of the K nearest lit ground splats around the "
                         "target over the same around its source, so the patch takes the local brightness. 0 = off")
    ap.add_argument("--tex-detail", type=float, default=1.0,
                    help="texture fill with --tex-match: scale each copy's difference from the mean of its source's K "
                         "lit neighbours by this factor, to restore fine contrast lost to the larger shadowed splats")
    args = ap.parse_args()
    if args.fill == "texture" and not args.ground_only:
        raise SystemExit("ERROR: --fill texture needs --ground-only")

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
                r = edge_band_ratio(img, mask, args.band, args.band_gap)
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
    lit_ref = ceiling = None
    if args.ground_only:
        centres = np.array([-qvec2rotmat(f["q"]).T @ f["t"] for f in frames])
        spread = float(np.median(np.linalg.norm(centres - centres.mean(0), axis=1)))
        tol = args.ground_tol or 0.02 * spread
        n_g, d_g, inlier_share, align = fit_ground_plane(xyz, opacity, frames, tol)
        near = np.abs(xyz @ n_g + d_g) < tol
        weight *= near
        # darkness between the lit and shadowed ground medians (linear luminance): 0 = lit, 1 = as dark as shadow
        dc0 = np.stack([ply[f"f_dc_{c}"] for c in range(3)], 1).astype(np.float64)
        lum = srgb_to_linear(np.clip(dc0 * SH_C0 + 0.5, 0, 1)) @ np.array([0.2126, 0.7152, 0.0722])
        lum = np.maximum(lum, 1e-4)
        lit_sel, sh_sel = near & valid & (frac < 0.1), near & valid & (frac >= args.hi)
        dark = np.ones(n)
        sh_ref = None
        if lit_sel.sum() > 100 and sh_sel.sum() > 100:
            lit_ref, sh_ref = np.median(lum[lit_sel]), np.median(lum[sh_sel])
            ceiling = float(np.percentile(lum[lit_sel], args.ceiling_pct))
            print(f"ground luminance: lit median {lit_ref:.4f}, shadow median {sh_ref:.4f} (ratio {lit_ref / sh_ref:.2f})")
        if lit_ref is not None and np.log(lit_ref / sh_ref) > 0.05:
            d = np.clip(np.log(lit_ref / lum) / np.log(lit_ref / sh_ref), 0, 1)
            dark = np.clip(d / 0.5, 0, 1)  # full once darker than the geometric midpoint
        else:
            print("WARNING: lit and shadowed ground not separable by brightness; darkness gate disabled (all pass)")
        if args.dark_gate:
            before = int((weight > 0).sum())
            weight *= dark
            print(f"dark gate: brightened {before:,} -> {int((weight > 0).sum()):,}")
        if args.colour_gate > 0 and sh_sel.sum() > 100:
            lin_c = srgb_to_linear(np.clip(dc0 * SH_C0 + 0.5, 0, 1))
            chrom = lin_c[:, :2] / np.maximum(lin_c.sum(1, keepdims=True), 1e-6)
            ref = np.median(chrom[sh_sel], 0)
            ok = np.linalg.norm(chrom - ref, axis=1) <= args.colour_gate
            before = int((weight > 0).sum())
            weight *= ok
            dark *= ok  # spread receivers must pass too
            print(f"colour gate {args.colour_gate} around shadowed-ground chromaticity {ref.round(3).tolist()}: "
                  f"brightened {before:,} -> {int((weight > 0).sum()):,}")
        if args.ground_spread > 0:
            from scipy.spatial import cKDTree
            gi = np.nonzero(near)[0]
            # 2D coordinates in the plane
            a = np.cross(n_g, [1.0, 0, 0] if abs(n_g[0]) < 0.9 else [0, 1.0, 0])
            a /= np.linalg.norm(a)
            b = np.cross(n_g, a)
            p2 = np.stack([xyz[gi] @ a, xyz[gi] @ b], 1)
            src = weight[gi] > 0
            if src.any():
                tree = cKDTree(p2[src])
                dist, j = tree.query(p2, k=1, distance_upper_bound=args.ground_spread)
                hit = np.isfinite(dist)
                spread_w = np.zeros(len(gi))
                spread_w[hit] = weight[gi][src][j[hit]] * dark[gi][hit]
                before = int((weight > 0).sum())
                weight[gi] = np.maximum(weight[gi], spread_w)
                print(f"ground spread r={args.ground_spread}: brightened {before:,} -> {int((weight > 0).sum()):,}")
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
    gain_raw = gain.copy()
    gain = np.clip(gain, 1.0, args.max_gain)
    print(f"gain (linear RGB) = {np.round(gain, 3).tolist()}, before clipping {np.round(gain_raw, 3).tolist()} [{gain_src}]")

    # recolour: scale DC colour in linear light, scale view-dependent SH by the same per-channel factor
    # only rewrite Gaussians that get brightened; everything else is copied bit-for-bit
    dc = np.stack([ply[f"f_dc_{c}"] for c in range(3)], 1).astype(np.float64)
    rgb = np.clip(dc * SH_C0 + 0.5, 0, 1)
    lin0 = srgb_to_linear(rgb)
    lum_w = np.array([0.2126, 0.7152, 0.0722])
    g_lum = float(gain @ lum_w)
    per_splat = np.full(n, g_lum)
    if args.mode == "target":
        if ceiling is None:
            sel = valid & (frac < 0.1)
            ceiling = float(np.percentile(lin0[sel] @ lum_w, args.ceiling_pct)) if sel.sum() > 100 else None
        if ceiling is not None:
            # cap each splat's gain so its luminance doesn't pass the lit ceiling
            per_splat = np.clip(ceiling / np.maximum(lin0 @ lum_w, 1e-4), 1.0, g_lum)
            print(f"target mode: ceiling = lit luminance p{args.ceiling_pct:g} {ceiling:.4f}; "
                  f"mean per-splat gain on touched splats {per_splat[weight > 0].mean():.2f}")
        else:
            print("WARNING: no lit reference for target mode; falling back to scale")
    if args.colour == "lum":
        ch_gain = per_splat[:, None] * np.ones(3)
    else:
        ch_gain = gain[None, :] * (per_splat / g_lum)[:, None]
    lin = lin0 * (1 + weight[:, None] * (ch_gain - 1))
    lin = lin / np.maximum(lin.max(1, keepdims=True), 1.0)  # clip without shifting hue
    rest = sorted((k for k in names if k.startswith("f_rest_")), key=lambda k: int(k.split("_")[-1]))
    texture = None
    src = np.full(n, -1)
    if args.fill == "texture":
        from scipy.spatial import cKDTree
        a = np.cross(n_g, [1.0, 0, 0] if abs(n_g[0]) < 0.9 else [0, 1.0, 0])
        a /= np.linalg.norm(a)
        b = np.cross(n_g, a)
        p2 = np.stack([xyz @ a, xyz @ b], 1)
        lin_dc = srgb_to_linear(np.clip(dc0 * SH_C0 + 0.5, 0, 1))
        lit = near & valid & (frac < args.tex_lit) & (opacity > args.opaque)
        if args.colour_gate > 0:
            # sources must look like lit ground too (keeps the bear's feet and other objects out)
            chrom = lin_dc[:, :2] / np.maximum(lin_dc.sum(1, keepdims=True), 1e-6)
            lit &= np.linalg.norm(chrom - np.median(chrom[lit], 0), axis=1) <= args.colour_gate
        li = np.nonzero(lit)[0]
        ti = np.nonzero(weight > 0)[0]
        tree = cKDTree(p2[li])
        angles = np.deg2rad(np.arange(0, 360, 15))
        lengths = np.arange(args.tex_step, args.tex_max_shift + 1e-9, args.tex_step)
        shifts = np.array([(m * np.cos(t), m * np.sin(t)) for m in lengths for t in angles])
        cover, hits = np.zeros(len(shifts)), []
        for s_i, s in enumerate(shifts):
            d, j = tree.query(p2[ti] + s, k=1, distance_upper_bound=args.tex_radius)
            ok = np.isfinite(d)
            cover[s_i] = ok.mean()
            hits.append(np.where(ok, li[np.minimum(j, len(li) - 1)], -1))
        best = int(np.argmax(cover))
        chosen = next(i for i in range(len(shifts)) if cover[i] >= args.tex_cover * cover[best])  # shortest first
        src[ti] = hits[chosen]
        order = np.argsort(np.linalg.norm(shifts, axis=1), kind="stable")
        # splats the chosen shift misses: shortest other shift that lands on lit ground
        for s_i in order:
            miss = src[ti] < 0
            if not miss.any():
                break
            src[ti[miss]] = hits[s_i][miss]
        wfill = np.where(src >= 0, weight, 0.0)
        band = np.zeros(0, int)
        if args.tex_feather > 0:
            # penumbra splats just outside the patch, which the vote thresholds left dark
            cand = np.nonzero(near & valid & (weight == 0) & (frac > 0))[0]
            dist, _ = cKDTree(p2[ti]).query(p2[cand], k=1, distance_upper_bound=args.tex_feather)
            band = cand[np.isfinite(dist)]
            ramp = 1 - dist[np.isfinite(dist)] / args.tex_feather
            for s in shifts[order]:
                miss = src[band] < 0
                if not miss.any():
                    break
                d, j = tree.query(p2[band[miss]] + s, k=1, distance_upper_bound=args.tex_radius)
                ok = np.isfinite(d)
                src[band[miss][ok]] = li[j[ok]]
        gi = np.nonzero(src >= 0)[0]
        copy = lin_dc[src[gi]]
        ratio = None
        if args.tex_match > 0:
            k = min(args.tex_match, len(li))
            _, jt = tree.query(p2[gi], k=k)
            _, js = tree.query(p2[src[gi]], k=k)
            mean_s = lin_dc[li[js]].mean(1)
            ratio = np.clip(lin_dc[li[jt]].mean(1) / np.maximum(mean_s, 1e-4), 0.5, 2.0)
            copy = np.maximum(mean_s + args.tex_detail * (copy - mean_s), 0) * ratio
        if args.tex_feather > 0:
            # fill by how much darker a splat is than its copy: full once below 40% of it, none when as bright
            lum_own, lum_copy = lin0[gi] @ lum_w, np.maximum(copy @ lum_w, 1e-4)
            w_dark = np.clip((1 - lum_own / lum_copy) / 0.6, 0, 1)
            in_band = np.isin(gi, band)
            band_ramp = np.zeros(n)
            band_ramp[band] = ramp
            wfill[gi] = np.where(in_band, w_dark * band_ramp[gi], np.maximum(wfill[gi], w_dark))
        wt = wfill[gi, None]
        lin[gi] = (1 - wt) * lin0[gi] + wt * copy
        rat_full = np.ones((n, 3))
        if ratio is not None:
            rat_full[gi] = ratio
        filled = src >= 0
        texture = dict(shift=shifts[chosen].round(3).tolist(), shift_length=round(float(np.linalg.norm(shifts[chosen])), 3),
                       coverage_chosen=round(float(cover[chosen]), 4), coverage_best=round(float(cover[best]), 4),
                       best_shift=shifts[best].round(3).tolist(), lit_sources=int(len(li)), targets=int(len(ti)),
                       filled=int((src[ti] >= 0).sum()), filled_by_chosen_shift=int((hits[chosen] >= 0).sum()),
                       gain_fallback=int((src[ti] < 0).sum()), feather_band=int(len(band)),
                       feather_filled=int((wfill[band] > 0).sum()),
                       match_ratio_median=None if ratio is None else np.median(ratio, 0).round(3).tolist(),
                       match_ratio_p10_p90=None if ratio is None else
                       np.percentile(ratio @ lum_w, [10, 90]).round(3).tolist())
        print(f"texture fill: {len(li):,} lit ground sources, {len(ti):,} targets; shift {texture['shift']} "
              f"(length {texture['shift_length']}) covers {cover[chosen]:.1%} (best {cover[best]:.1%} at "
              f"{texture['best_shift']}); filled {texture['filled']:,}, gain fallback {texture['gain_fallback']:,}; "
              f"feather band {len(band):,} ({texture['feather_filled']:,} changed); "
              f"match ratio median {texture['match_ratio_median']}, p10/p90 {texture['match_ratio_p10_p90']}")
    new_rgb = np.clip(linear_to_srgb(lin), 0, 1)
    touched = weight > 0
    if texture is not None:
        touched = touched | (wfill > 0)
    factor = np.where(touched[:, None] & (rgb > 1e-3), new_rgb / np.maximum(rgb, 1e-3), 1.0)
    fixed = ply.copy()
    for c in range(3):
        fixed[f"f_dc_{c}"] = np.where(touched, (new_rgb[:, c] - 0.5) / SH_C0, ply[f"f_dc_{c}"]).astype(np.float32)
    per_ch = len(rest) // 3  # f_rest is channel-major: all R coeffs, then G, then B
    for j, k in enumerate(rest):
        fixed[k] = (ply[k] * factor[:, min(j // max(per_ch, 1), 2)]).astype(np.float32)
        if texture is not None:  # copied splats take the source's view-dependent colour too
            c = min(j // max(per_ch, 1), 2)
            rest_copy = ply[k][np.maximum(src, 0)] * (1.0 if ratio is None else rat_full[:, c])
            fixed[k] = np.where(src >= 0, (1 - wfill) * ply[k] + wfill * rest_copy, fixed[k]).astype(np.float32)
    write_ply(args.out / "splat_shadowfix.ply", header, fixed)
    if texture is not None:
        np.save(args.out / "texture_source.npy", src.astype(np.int64))

    debug = ply.copy()
    red = np.array([1.0, 0.0, 0.0])
    dbg_rgb = rgb * (1 - weight[:, None]) + red * weight[:, None]
    for c in range(3):
        debug[f"f_dc_{c}"] = np.where(touched, (dbg_rgb[:, c] - 0.5) / SH_C0, ply[f"f_dc_{c}"]).astype(np.float32)
    for k in rest:
        debug[k] = (ply[k] * (1 - weight)).astype(np.float32)
    write_ply(args.out / "splat_shadowmask_debug.ply", header, debug)

    stats = dict(
        ply=str(args.ply), colmap=str(args.colmap), masks=str(args.masks), frames_used=used,
        gaussians=int(n), seen_min_views=int(valid.sum()), brightened_any=int((weight > 0).sum()),
        brightened_full=int((weight >= 1).sum()), changed_gaussians=int(touched.sum()), gain_linear_rgb=gain.round(4).tolist(), gain_raw_rgb=gain_raw.round(4).tolist(), gain_luminance=round(g_lum, 4),
        lit_ref_luminance=None if lit_ref is None else round(float(lit_ref), 4),
        ceiling_luminance=None if ceiling is None else round(float(ceiling), 4), gain_source=gain_src,
        gain_frames=len(ratios), ground_plane=ground, texture_fill=texture, params={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        seconds=round(time.time() - t0, 1))
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "params"}, indent=2))


if __name__ == "__main__":
    main()
