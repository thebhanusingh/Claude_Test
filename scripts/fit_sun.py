#!/usr/bin/env python
"""Stage 2 of the shadow-free splat plan: estimate the sun direction from cast-shadow geometry.

Idea: the 2D shadow masks (SDDNet, one per frame) already tell us which ground Gaussians lie in shadow, by
multi-view voting. For a candidate sun direction s, a ground Gaussian is predicted to be in shadow when an opaque
Gaussian above the ground lies between it and the sun. The fitted sun is the direction whose predicted shadow best
matches the observed one (IoU on the ground receivers). This is the same ray test stage 4 needs, so stages 2 and 4
share it.

1. Observed shadow on receivers: ground Gaussians (|height| <= --ground-tol around the plane in --plane-from) within
   --radius of --centre. Each receiver is projected into every --every-th frame, kept if it passes a coarse z-buffer
   (same test as fix_splat_shadows.py), and votes shadow / lit from that frame's mask. Receivers seen in fewer than
   --min-views frames are ignored. Votes are cached in <out>/votes.npz.
2. Prediction for s (orthographic shadow map): occluders are opaque Gaussians (opacity > --opaque) higher than
   --occluder-min above the plane within the radius. Project occluders and receivers onto the plane perpendicular to s
   on a --cell grid; keep per cell the furthest-toward-the-sun occluder depth, dilated by --dilate cells (splats are
   bigger than their centres). A receiver is shadowed if that depth exceeds its own depth along s by --eps.
3. Search: elevation 5-85 deg x azimuth 0-360 deg at 5 deg, then 1 deg and 0.25 deg around the best. Uncertainty: the
   angular radius of fine-grid directions within --tie of the best IoU. Hold-out: refit on each half of the receivers
   (split through the centre) and report how far the two half-fits land from each other and how each scores on the
   other half.

Usage:
  fit_sun.py --ply exports/IMG_6556/splat.ply --colmap data/IMG_6556/colmap/sparse/0_refined \
      --masks data/IMG_6556/shadow_masks --plane-from exports/IMG_6556_shadowfix_v8_o/stats.json \
      --centre 0.713 1.091 0.464 --radius 3.0 --out exports/IMG_6556_sun
  fit_sun.py --selftest --out /tmp/sun_selftest

Outputs in --out: sun.json (direction in world coords, elevation/azimuth about the ground normal, IoU, uncertainty,
hold-out), votes.npz, score_map.npy (coarse IoU grid), and sun_fit.png (observed vs predicted shadow, top view, if
matplotlib is available). Trains nothing.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter

sys.path.insert(0, str(Path(__file__).parent))


def plane_frame(up):
    up = up / np.linalg.norm(up)
    e1 = np.cross(up, [1.0, 0, 0]) if abs(up[0]) < 0.9 else np.cross(up, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    return up, e1, np.cross(up, e1)


def sun_vec(elev, azim, frame):
    up, e1, e2 = frame
    el, az = np.radians(elev), np.radians(azim)
    return np.cos(el) * (np.cos(az) * e1 + np.sin(az) * e2) + np.sin(el) * up


def predict(occ, rec, s, cell, dilate, eps):
    """Boolean shadow prediction for receivers rec (N,3) given occluders occ (M,3) and sun direction s."""
    a = np.cross(s, [1.0, 0, 0]) if abs(s[0]) < 0.9 else np.cross(s, [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(s, a)
    po, pr = np.c_[occ @ a, occ @ b], np.c_[rec @ a, rec @ b]
    lo = np.minimum(po.min(0), pr.min(0)) - cell * (dilate + 1)
    io, ir = ((po - lo) / cell).astype(int), ((pr - lo) / cell).astype(int)
    shape = np.maximum(io.max(0), ir.max(0)) + dilate + 2
    depth = np.full(shape, -np.inf)
    np.maximum.at(depth, (io[:, 0], io[:, 1]), occ @ s)
    if dilate:
        depth = maximum_filter(depth, size=2 * dilate + 1, mode="constant", cval=-np.inf)
    return depth[ir[:, 0], ir[:, 1]] > rec @ s + eps


def iou(pred, obs):
    u = (pred | obs).sum()
    return float((pred & obs).sum() / u) if u else 0.0


def search(occ, rec, obs, frame, args, centre_el=None, centre_az=None):
    """Coarse-to-fine grid search. Returns best (elev, azim, iou), fine-grid samples and coarse map."""
    def grid(els, azs):
        out = []
        for el in els:
            if not 1 <= el <= 89:
                continue
            for az in azs:
                out.append((el, az % 360, iou(predict(occ, rec, sun_vec(el, az, frame), args.cell, args.dilate,
                                                      args.eps), obs)))
        return out

    coarse_els, coarse_azs = np.arange(5, 86, 5.0), np.arange(0, 360, 5.0)
    coarse = grid(coarse_els, coarse_azs)
    cmap = np.array([c[2] for c in coarse]).reshape(len(coarse_els), len(coarse_azs))
    el, az, _ = max(coarse, key=lambda r: r[2])
    mid = grid(el + np.arange(-5, 5.1, 1.0), az + np.arange(-5, 5.1, 1.0))
    el, az, _ = max(mid, key=lambda r: r[2])
    fine = grid(el + np.arange(-1.5, 1.51, 0.25), az + np.arange(-1.5, 1.51, 0.25))
    best = max(fine, key=lambda r: r[2])
    return best, mid + fine, cmap


def angle_between(u, v):
    return float(np.degrees(np.arccos(np.clip(u @ v, -1, 1))))


def collect_votes(xyz, opacity, rec_idx, args):
    import cv2
    from fix_splat_shadows import project, qvec2rotmat, read_cameras_bin, read_images_bin
    cams = read_cameras_bin(args.colmap / "cameras.bin")
    frames = read_images_bin(args.colmap / "images.bin")[:: args.every]
    rec = xyz[rec_idx]
    shadow = np.zeros(len(rec_idx), np.int32)
    seen = np.zeros(len(rec_idx), np.int32)
    occl = xyz[opacity > args.opaque]
    used = 0
    print(f"collecting shadow votes from up to {len(frames)} frames (progress every 25)", flush=True)
    for fr in frames:
        mpath = args.masks / (Path(fr["name"]).stem + ".png")
        if not mpath.exists():
            continue
        cam = cams[fr["cam"]]
        w, h = cam["width"], cam["height"]
        mask = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        R = qvec2rotmat(fr["q"])
        zw, zh = max(1, int(w * args.zbuf_scale)), max(1, int(h * args.zbuf_scale))
        zbuf = np.full(zw * zh, np.inf)
        oc = occl @ R.T + fr["t"]
        oc = oc[oc[:, 2] > 1e-3]
        uo = project(cam, oc)
        ok = (uo[:, 0] >= 0) & (uo[:, 0] < w) & (uo[:, 1] >= 0) & (uo[:, 1] < h)
        cell = (uo[ok, 1] * args.zbuf_scale).astype(np.int64) * zw + (uo[ok, 0] * args.zbuf_scale).astype(np.int64)
        np.minimum.at(zbuf, np.clip(cell, 0, zw * zh - 1), oc[ok, 2])
        rc = rec @ R.T + fr["t"]
        front = np.flatnonzero(rc[:, 2] > 1e-3)
        ur = project(cam, rc[front])
        inb = (ur[:, 0] >= 0) & (ur[:, 0] < w) & (ur[:, 1] >= 0) & (ur[:, 1] < h)
        front, ur = front[inb], ur[inb]
        cell = np.clip((ur[:, 1] * args.zbuf_scale).astype(np.int64) * zw + (ur[:, 0] * args.zbuf_scale).astype(np.int64),
                       0, zw * zh - 1)
        vis = rc[front, 2] <= zbuf[cell] * (1 + args.z_tol)
        front, ur = front[vis], ur[vis]
        shadow[front] += mask[ur[:, 1].astype(int), ur[:, 0].astype(int)] > 127
        seen[front] += 1
        used += 1
        if used % 25 == 0:
            print(f"  votes: {used} frames done ({fr['name']})", flush=True)
    print(f"votes from {used} masked frames: receivers seen >= {args.min_views}x: {(seen >= args.min_views).sum():,}")
    return shadow, seen


def fit(xyz, opacity, up, offset, tol, centre, args, votes=None):
    t0 = time.time()
    height = xyz @ up + offset
    near = np.linalg.norm(xyz - centre, axis=1) < args.radius
    rec_idx = np.flatnonzero(near & (np.abs(height) <= tol))
    occ_r = args.occluder_radius or 1.5 * args.radius  # shadows inside the region can come from objects outside it
    occ_mask = (np.linalg.norm(xyz - centre, axis=1) < occ_r) & (opacity > args.opaque) & (height > args.occluder_min)
    occ = xyz[occ_mask]
    print(f"{len(rec_idx):,} ground receivers within {args.radius} of the centre, {len(occ):,} occluders "
          f"within {occ_r:g} (opacity > {args.opaque}, > {args.occluder_min} above the plane)")
    if votes is None:
        shadow, seen = collect_votes(xyz, opacity, rec_idx, args)
        np.savez_compressed(args.out / "votes.npz", rec_idx=rec_idx, shadow=shadow, seen=seen)
    else:
        shadow, seen = votes
    keep = seen >= args.min_views
    rec_idx, frac = rec_idx[keep], shadow[keep] / seen[keep]
    rec = xyz[rec_idx]
    obs = frac >= 0.5
    print(f"{len(rec):,} receivers used, {obs.mean() * 100:.1f}% observed in shadow")
    if len(occ) > args.max_occluders:
        occ = occ[np.random.default_rng(0).choice(len(occ), args.max_occluders, replace=False)]

    frame = plane_frame(up)
    (el, az, best), samples, cmap = search(occ, rec, obs, frame, args)
    s = sun_vec(el, az, frame)
    tied = [sun_vec(e, a, frame) for e, a, v in samples if v >= best - args.tie]
    unc = max(angle_between(s, t) for t in tied)
    print(f"best sun: elevation {el:.2f} deg, azimuth {az:.2f} deg, IoU {best:.3f} "
          f"(uncertainty radius {unc:.2f} deg over {len(tied)} tied directions), {time.time() - t0:.0f} s")

    # hold-out: split receivers through the centre along the sun's horizontal direction's normal
    _, e1, e2 = frame
    side = (rec - centre) @ np.cross(up, np.cos(np.radians(az)) * e1 + np.sin(np.radians(az)) * e2) > 0
    halves = []
    for h in (side, ~side):
        if obs[h].sum() < 20 or obs[~h].sum() < 20:
            halves = None
            break
        (hel, haz, hbest), _, _ = search(occ, rec[h], obs[h], frame, args)
        hs = sun_vec(hel, haz, frame)
        other = iou(predict(occ, rec[~h], hs, args.cell, args.dilate, args.eps), obs[~h])
        halves.append(dict(elevation=hel, azimuth=haz, iou_fit=round(hbest, 4), iou_other_half=round(other, 4),
                           deg_from_full_fit=round(angle_between(hs, s), 2)))
    if halves:
        d = angle_between(sun_vec(halves[0]["elevation"], halves[0]["azimuth"], frame),
                          sun_vec(halves[1]["elevation"], halves[1]["azimuth"], frame))
        print(f"hold-out: half fits {d:.2f} deg apart; IoU on the other half {halves[0]['iou_other_half']:.3f} / "
              f"{halves[1]['iou_other_half']:.3f}")
    pred = predict(occ, rec, s, args.cell, args.dilate, args.eps)
    return dict(sun_world=s.round(5).tolist(), elevation_deg=round(el, 2), azimuth_deg=round(az, 2),
                azimuth_ref=dict(e1=frame[1].round(4).tolist(), e2=frame[2].round(4).tolist()),
                up=up.round(5).tolist(), iou=round(best, 4), uncertainty_deg=round(unc, 2),
                receivers=int(len(rec)), observed_shadow_share=round(float(obs.mean()), 4),
                predicted_shadow_share=round(float(pred.mean()), 4), occluders=int(occ_mask.sum()),
                holdout=halves, seconds=round(time.time() - t0, 1)), (rec, obs, pred, cmap)


def plot(rec, obs, pred, up, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping sun_fit.png")
        return
    _, e1, e2 = plane_frame(up)
    p = np.c_[rec @ e1, rec @ e2]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for axx, m, title in zip(axs, [obs, pred, None], ["observed shadow (masks)", "predicted (fitted sun)", "agreement"]):
        if m is None:
            col = np.where(obs & pred, 0, np.where(obs & ~pred, 1, np.where(~obs & pred, 2, 3)))
            cmapc = np.array(["#2b8a3e", "#e8590c", "#1971c2", "#dee2e6"])
            axx.scatter(p[:, 0], p[:, 1], c=cmapc[col], s=0.5)
            axx.set_title(title + ": green both, orange missed, blue extra")
        else:
            axx.scatter(p[:, 0], p[:, 1], c=np.where(m, "#212529", "#dee2e6"), s=0.5)
            axx.set_title(title)
        axx.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out / "sun_fit.png", dpi=110)
    print(f"wrote {out / 'sun_fit.png'}")


def selftest(args):
    """Box and pole standing on a ground plane, a known sun, observed shadow from the predictor at a finer cell with
    10% label noise. The fit must land within 2 deg."""
    rng = np.random.default_rng(0)
    up, offset, tol = np.array([0, 0, 1.0]), 0.0, 0.02
    ground = np.c_[rng.uniform(-3, 3, (60000, 2)), rng.normal(0, 0.003, 60000)]
    box = rng.uniform([-0.4, -0.3, 0.05], [0.4, 0.3, 0.9], (20000, 3))
    pole = np.c_[rng.normal(0.9, 0.03, (3000, 2)), rng.uniform(0.05, 1.5, 3000)]
    xyz = np.r_[ground, box, pole]
    opacity = np.ones(len(xyz))
    frame = plane_frame(up)
    true_el, true_az = 38.0, 127.0
    s = sun_vec(true_el, true_az, frame)
    rec_idx = np.arange(len(ground))
    obs = predict(np.r_[box, pole], ground, s, args.cell / 2, args.dilate, args.eps)
    flip = rng.random(len(obs)) < 0.10
    noisy = obs ^ flip
    seen = np.full(len(rec_idx), 10)
    shadow = np.where(noisy, 8, 2)
    args.radius = 10.0  # every synthetic ground Gaussian is a receiver, matching the votes
    res, extra = fit(xyz, opacity, up, offset, tol, np.zeros(3), args, votes=(shadow, seen))
    err = angle_between(np.array(res["sun_world"]), s)
    print(f"true sun: elevation {true_el}, azimuth {true_az}; fitted {res['elevation_deg']}, {res['azimuth_deg']}; "
          f"error {err:.2f} deg")
    plot(*extra[:3], up, args.out)
    ok = err < 2.0 and res["holdout"] is not None
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", type=Path)
    ap.add_argument("--colmap", type=Path)
    ap.add_argument("--masks", type=Path)
    ap.add_argument("--plane-from", type=Path, help="fix_splat_shadows.py stats.json with ground_plane")
    ap.add_argument("--centre", type=float, nargs=3, help="centre of the region (the hero object)")
    ap.add_argument("--radius", type=float, default=3.0, help="region radius; must cover the object's whole shadow")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ground-tol", type=float, help="receiver band around the plane (default: the plane's own tol)")
    ap.add_argument("--occluder-radius", type=float, help="occluder search radius (default 1.5 x --radius)")
    ap.add_argument("--occluder-min", type=float, default=0.05, help="min height above the plane for occluders")
    ap.add_argument("--opaque", type=float, default=0.5)
    ap.add_argument("--max-occluders", type=int, default=150_000, help="random subsample for speed")
    ap.add_argument("--cell", type=float, default=0.02, help="shadow-map cell size (scene units)")
    ap.add_argument("--dilate", type=int, default=1, help="shadow-map dilation in cells")
    ap.add_argument("--eps", type=float, default=0.02, help="depth margin along the sun direction")
    ap.add_argument("--tie", type=float, default=0.005, help="IoU margin for the uncertainty radius")
    ap.add_argument("--every", type=int, default=2, help="use every Nth frame for the votes")
    ap.add_argument("--min-views", type=int, default=5)
    ap.add_argument("--zbuf-scale", type=float, default=0.25)
    ap.add_argument("--z-tol", type=float, default=0.03)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.selftest:
        raise SystemExit(selftest(args))
    for k in ("ply", "colmap", "masks", "plane_from", "centre"):
        if getattr(args, k) is None:
            ap.error(f"--{k.replace('_', '-')} is required (or use --selftest)")

    from cluster_surfaces import read_ply
    _, data = read_ply(args.ply)
    xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
    opacity = 1 / (1 + np.exp(-data["opacity"].astype(np.float64)))
    gp = json.loads(args.plane_from.read_text())["ground_plane"]
    up, offset = np.array(gp["normal"], float), float(gp["offset"])
    tol = args.ground_tol if args.ground_tol else float(gp["tol"])
    votes = None
    vpath = args.out / "votes.npz"
    centre = np.array(args.centre)
    if vpath.exists():
        v = np.load(vpath)
        height = xyz @ up + offset
        rec_now = np.flatnonzero((np.linalg.norm(xyz - centre, axis=1) < args.radius) & (np.abs(height) <= tol))
        if np.array_equal(v["rec_idx"], rec_now):
            votes = (v["shadow"], v["seen"])
            print(f"reusing votes from {vpath}")
    res, extra = fit(xyz, opacity, up, offset, tol, centre, args, votes=votes)
    np.save(args.out / "score_map.npy", extra[3])
    res["params"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (args.out / "sun.json").write_text(json.dumps(res, indent=1))
    plot(*extra[:3], up, args.out)
    print(f"wrote {args.out / 'sun.json'}")


if __name__ == "__main__":
    main()
