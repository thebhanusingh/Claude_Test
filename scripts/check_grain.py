#!/usr/bin/env python
"""Check stage-3 grain against the photos: does each Gaussian's grain (clusters.npz local_dir) follow the texture
direction the camera actually saw?

For NF frames that see the crop centre from different sides (closest camera per azimuth bin around the crop):
  1. project the clustered Gaussians, keeping only front-most ones (z-buffer on 4-px bins from all opaque Gaussians)
     and only those whose grain is not along the view ray;
  2. project each grain direction into the image;
  3. compare with the photo's texture direction at that pixel: the along-edge eigenvector of a gradient structure
     tensor (Sobel, Gaussian sigma --sigma px), only where the tensor coherence > --img-coh and the surface is not seen
     at a grazing angle (--min-facing). --pixels edges keeps pixels above the frame's median gradient energy, which
     are mostly outlines and slat edges; --pixels interior keeps only pixels more than --edge-gap px from Canny edges,
     so it scores surface texture alone. 3DGS densification stretches Gaussians along image edges, so agreement on
     edges is partly built in; the interior score is the real test of grain.
Angles are unsigned, 0-90 deg. Baselines on the same pixels:
  rot90  - grain rotated 90 deg in the tangent plane of the cluster normal (cluster normal x grain), then projected;
  random - a random direction in the same tangent plane, then projected.
If the grain carries information, grain << random (~45 deg) and rot90 >> random. Results are split into grown,
attached-flat and attached-non-flat members (needs 'grown' and 'attach_off' in clusters.npz) and per cluster.

Usage:
  check_grain.py --run exports/IMG_6556_clusters_chair_r30agree_v3 --ply exports/IMG_6556/splat.ply \
      --colmap data/IMG_6556/colmap/sparse/0_refined --images data/IMG_6556/images --crop 0.713 1.091 0.464 1.065 \
      [--up-from exports/IMG_6556_shadowfix_v8_o/stats.json] [--frames 6] [--pixels edges|interior]
Writes <run>/grain_check_<pixels>/: grain_<frame>.jpg (photo | grain segments: green < 20, orange 20-45, red > 45 deg) and
grain_check.json. Reads everything else; trains nothing.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cluster_surfaces import read_ply
from fix_splat_shadows import qvec2rotmat, read_cameras_bin, read_images_bin


def intrinsics(cam):
    p = cam["params"]
    if cam["model"].startswith("SIMPLE"):
        return p[0], p[0], p[1], p[2]
    return p[0], p[1], p[2], p[3]  # distortion ignored


def angle_deg(a, b):
    return np.degrees(np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", a, b)), 0, 1)))


def summary(a):
    if len(a) == 0:
        return dict(n=0)
    return dict(n=int(len(a)), median_deg=round(float(np.median(a)), 1), under20=round(float((a < 20).mean()), 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="cluster_surfaces.py output folder")
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--colmap", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--crop", type=float, nargs=4, metavar=("X", "Y", "Z", "R"), required=True)
    ap.add_argument("--up-from", type=Path, help="stats.json with ground_plane.normal (else mean camera up)")
    ap.add_argument("--frames", type=int, default=6, help="azimuth bins, one frame each")
    ap.add_argument("--min-local-coh", type=float, default=0.3, help="skip Gaussians with weaker local grain")
    ap.add_argument("--img-coh", type=float, default=0.3, help="min structure-tensor coherence in the photo")
    ap.add_argument("--sigma", type=float, default=3.0, help="structure tensor smoothing (px)")
    ap.add_argument("--pixels", choices=["edges", "interior"], default="edges",
                    help="edges: pixels above median gradient energy (mostly outlines); interior: pixels more than "
                         "--edge-gap px from Canny edges (surface texture only)")
    ap.add_argument("--edge-gap", type=float, default=8.0)
    ap.add_argument("--min-facing", type=float, default=0.4,
                    help="min |cluster normal . view ray|; below it any tangent direction projects nearly the same")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = a.run / f"grain_check_{a.pixels}"
    out.mkdir(exist_ok=True)

    _, data = read_ply(a.ply)
    xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
    op = 1 / (1 + np.exp(-data["opacity"].astype(np.float64)))
    z = np.load(a.run / "clusters.npz")
    lab, ldir, lcoh = z["labels"], z["local_dir"], z["local_coh"]
    if "grown" in z.files:
        grown = z["grown"]
        att_flat = (lab >= 0) & ~grown & ~np.isnan(z["attach_off"])
    else:
        grown, att_flat = np.zeros(len(lab), bool), np.zeros(len(lab), bool)
        print("note: clusters.npz has no 'grown'; group split unavailable")
    group = np.where(grown, 0, np.where(att_flat, 1, 2))
    gnames = ["grown", "attached_flat", "attached_nonflat"]
    stats = {c["cluster"]: c for c in json.loads((a.run / "stats.json").read_text())["clusters"]}
    cnorm = np.zeros((len(stats), 3))
    for c, s in stats.items():
        cnorm[c] = s["normal_axis"]

    crop_c, crop_r = np.array(a.crop[:3]), a.crop[3]
    cand = np.flatnonzero((lab >= 0) & (np.linalg.norm(xyz - crop_c, axis=1) < crop_r) & (lcoh > a.min_local_coh))
    occl = np.flatnonzero(op > 0.5)
    print(f"{(lab >= 0).sum():,} clustered Gaussians; {len(cand):,} in the crop with local coherence > {a.min_local_coh}")

    # baseline directions in the tangent plane of each Gaussian's cluster normal
    d3 = ldir[cand]
    n3 = cnorm[lab[cand]]
    rot = np.cross(n3, d3)
    rot /= np.maximum(np.linalg.norm(rot, axis=1, keepdims=True), 1e-12)
    rnd = rng.normal(size=d3.shape)
    rnd -= np.einsum("ij,ij->i", rnd, n3)[:, None] * n3
    rnd /= np.maximum(np.linalg.norm(rnd, axis=1, keepdims=True), 1e-12)

    cams = read_cameras_bin(a.colmap / "cameras.bin")
    ims = read_images_bin(a.colmap / "images.bin")
    if a.up_from:
        up = np.array(json.loads(a.up_from.read_text())["ground_plane"]["normal"])
    else:
        up = -np.mean([qvec2rotmat(im["q"])[1] for im in ims], 0)
    up /= np.linalg.norm(up)
    ax = np.cross(up, [1, 0, 0]) if abs(up[0]) < 0.9 else np.cross(up, [0, 1, 0])
    ax /= np.linalg.norm(ax)
    bx = np.cross(up, ax)

    info = []
    for im in ims:
        R, t, cam = qvec2rotmat(im["q"]), im["t"], cams[im["cam"]]
        pc = R @ crop_c + t
        if pc[2] <= 0:
            continue
        fx, fy, cx, cy = intrinsics(cam)
        u, v = fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy
        if 0.25 * cam["width"] < u < 0.75 * cam["width"] and 0.2 * cam["height"] < v < 0.8 * cam["height"]:
            dv = -R.T @ t - crop_c
            info.append((np.arctan2(dv @ bx, dv @ ax), np.linalg.norm(dv), im))
    bins = [int((i[0] + np.pi) / (2 * np.pi) * a.frames) % a.frames for i in info]
    frames = [min((info[i] for i in range(len(info)) if bins[i] == b), key=lambda r: r[1])[2]
              for b in range(a.frames) if b in bins]
    print(f"{len(info)} frames see the crop centred; using {len(frames)}: {', '.join(f['name'] for f in frames)}")

    res = {k: [] for k in ("grain", "rot90", "random", "group", "label")}
    rows = []
    for im in frames:
        cam = cams[im["cam"]]
        W, H = cam["width"], cam["height"]
        fx, fy, cx, cy = intrinsics(cam)
        R, t = qvec2rotmat(im["q"]), im["t"]
        C = -R.T @ t

        def proj(P):
            Q = P @ R.T + t
            return np.stack([fx * Q[:, 0] / Q[:, 2] + cx, fy * Q[:, 1] / Q[:, 2] + cy], 1), Q[:, 2]

        B, nbx = 4, W // 4 + 1
        uo, zo = proj(xyz[occl])
        ok = (zo > 0.05) & (uo[:, 0] >= 0) & (uo[:, 0] < W) & (uo[:, 1] >= 0) & (uo[:, 1] < H)
        zbuf = np.full((H // B + 1) * nbx, np.inf)
        np.minimum.at(zbuf, (uo[ok, 1] // B).astype(int) * nbx + (uo[ok, 0] // B).astype(int), zo[ok])

        uv, zc = proj(xyz[cand])
        inb = (zc > 0.05) & (uv[:, 0] >= 2) & (uv[:, 0] < W - 2) & (uv[:, 1] >= 2) & (uv[:, 1] < H - 2)
        bi = np.where(inb, (np.nan_to_num(uv[:, 1]) // B).astype(int) * nbx + (np.nan_to_num(uv[:, 0]) // B).astype(int), 0)
        vis = inb & (zc <= zbuf[bi] * 1.02 + 0.02)
        ray = xyz[cand] - C
        ray /= np.linalg.norm(ray, axis=1, keepdims=True)
        vis &= np.linalg.norm(d3 - np.einsum("ij,ij->i", d3, ray)[:, None] * ray, axis=1) > 0.5
        facing = np.abs(np.einsum("ij,ij->i", n3, ray))  # grazing views squash every tangent direction together
        idx = np.flatnonzero(vis)

        def dir2d(D):
            q, _ = proj(xyz[cand[idx]] + 0.01 * D[idx])
            g = q - uv[idx]
            return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)

        g_grain, g_rot, g_rnd = dir2d(d3), dir2d(rot), dir2d(rnd)

        img = cv2.imread(str(a.images / im["name"]))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        Jxx, Jxy, Jyy = (cv2.GaussianBlur(m, (0, 0), a.sigma) for m in (gx * gx, gx * gy, gy * gy))
        theta = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy) + np.pi / 2  # along edges
        energy = Jxx + Jyy
        coh = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2) / np.maximum(energy, 1e-9)
        px = uv[idx].astype(int)
        keep = (coh[px[:, 1], px[:, 0]] > a.img_coh) & (facing[idx] > a.min_facing)
        if a.pixels == "edges":
            keep &= energy[px[:, 1], px[:, 0]] > np.median(energy)
        else:  # interior: away from strong edges/silhouettes, so only surface texture is scored
            edges = cv2.Canny((cv2.GaussianBlur(gray, (0, 0), 1.5) * 255).astype(np.uint8), 40, 120)
            dist = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
            keep &= (dist[px[:, 1], px[:, 0]] > a.edge_gap) & (energy[px[:, 1], px[:, 0]] > np.percentile(energy, 20))
        th = theta[px[keep, 1], px[keep, 0]]
        tdir = np.stack([np.cos(th), np.sin(th)], 1)
        ag, ar, an = angle_deg(g_grain[keep], tdir), angle_deg(g_rot[keep], tdir), angle_deg(g_rnd[keep], tdir)
        res["grain"].append(ag); res["rot90"].append(ar); res["random"].append(an)
        res["group"].append(group[cand[idx[keep]]]); res["label"].append(lab[cand[idx[keep]]])
        rows.append(dict(frame=im["name"], visible=int(len(idx)), grain=summary(ag), rot90=summary(ar), random=summary(an)))
        print(f"{im['name']}: {len(idx):6,d} visible, {keep.sum():6,d} on textured pixels | median grain {np.median(ag):5.1f}  "
              f"rot90 {np.median(ar):5.1f}  random {np.median(an):5.1f} deg | under 20: {100 * (ag < 20).mean():4.1f} / "
              f"{100 * (ar < 20).mean():4.1f} / {100 * (an < 20).mean():4.1f} %")

        pts, gk = uv[idx[keep]], g_grain[keep]
        if len(pts):
            x0, y0 = np.maximum(pts.min(0).astype(int) - 40, 0)
            x1, y1 = np.minimum(pts.max(0).astype(int) + 40, [W, H])
            over = img.copy()
            for i in rng.choice(len(pts), min(3000, len(pts)), replace=False):
                col = (60, 220, 60) if ag[i] < 20 else (0, 200, 255) if ag[i] < 45 else (60, 60, 255)
                cv2.line(over, tuple((pts[i] - 7 * gk[i]).astype(int)), tuple((pts[i] + 7 * gk[i]).astype(int)),
                         col, 1, cv2.LINE_AA)
            side = np.concatenate([img[y0:y1, x0:x1], over[y0:y1, x0:x1]], 1)
            cv2.putText(side, f"{im['name']}  green <20  orange 20-45  red >45 deg", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imwrite(str(out / f"grain_{Path(im['name']).stem}.jpg"), side, [cv2.IMWRITE_JPEG_QUALITY, 88])

    G, Rt, Rn = (np.concatenate(res[k]) for k in ("grain", "rot90", "random"))
    grp, L = np.concatenate(res["group"]), np.concatenate(res["label"])
    report = dict(run=str(a.run), frames=rows,
                  overall=dict(grain=summary(G), rot90=summary(Rt), random=summary(Rn)), groups={}, clusters=[])
    print(f"\nALL: n {len(G):,} | median grain {np.median(G):.1f}  rot90 {np.median(Rt):.1f}  random {np.median(Rn):.1f} deg"
          f" | under 20: {100 * (G < 20).mean():.1f} / {100 * (Rt < 20).mean():.1f} / {100 * (Rn < 20).mean():.1f} %")
    for k, name in enumerate(gnames):
        m = grp == k
        report["groups"][name] = dict(grain=summary(G[m]), rot90=summary(Rt[m]), random=summary(Rn[m]))
        if m.sum():
            print(f"  {name:17s} n {m.sum():6,d} | median grain {np.median(G[m]):5.1f}  rot90 {np.median(Rt[m]):5.1f}  "
                  f"random {np.median(Rn[m]):5.1f} deg | under 20: {100 * (G[m] < 20).mean():4.1f} %")
    print("per cluster (>= 100 comparisons):")
    for c in np.unique(L):
        m = L == c
        if m.sum() < 100:
            continue
        s = stats.get(int(c), {})
        report["clusters"].append(dict(cluster=int(c), grain=summary(G[m]), rot90=summary(Rt[m]), random=summary(Rn[m]),
                                       grain_coherence=s.get("grain_coherence"), gaussians=s.get("gaussians"),
                                       grown=s.get("grown")))
        print(f"  cluster {c:3d}: n {m.sum():6,d} | median grain {np.median(G[m]):5.1f}  rot90 {np.median(Rt[m]):5.1f}  "
              f"random {np.median(Rn[m]):5.1f} deg | coherence {s.get('grain_coherence')}")
    (out / "grain_check.json").write_text(json.dumps(report, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
