#!/usr/bin/env python
"""Refit the sun direction so the shadow it casts through the 2DGS surfels matches the shadow seen in the photos.

For a candidate sun, a near-orthographic depth map is rendered from the sun through the surfels near the object (as in
sun_shadow_map.py); a ground receiver (3DGS blob with photo-mask votes, fit_sun votes.npz) is predicted in shadow when
none of its --k nearest surfels is seen by the sun. Score = IoU with the receivers the photo masks vote as shadow
(share > 0.5, seen >= 5) within --ring x object height of the footprint. Search: elevation x azimuth grid around the
start sun (--from), then a finer grid around the best. Also draws overlays on a few photos: predicted-only red,
observed-only blue, both white, for the start sun and the refitted one.

Usage: sun_refit.py --surfels exports/IMG_6705_2dgs_fromsplat/surfels_params.pt --ply exports/IMG_6705/splat.ply
           --from exports/IMG_6705_shadow4/shadow_flags.npz --votes exports/IMG_6705_sun/votes.npz
           --object exports/IMG_6705_object/object.json --plane-from exports/IMG_6705_shadowfix_o/stats.json
           --colmap data/IMG_6705/colmap/sparse/0_refined --images data/IMG_6705/images --out exports/IMG_6705_sun_refit
Writes <out>/sun_refit.json (start / best sun, elevation, azimuth, IoU, grid), sun_refit.npz (sun), overlay_*.jpg.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from fix_splat_shadows import qvec2rotmat, read_cameras_bin, read_images_bin, read_ply
from render_orbit import look_at
from train_2dgs_ref import render


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("surfels", "ply", "from", "votes", "object", "plane-from", "colmap", "images", "out"):
        ap.add_argument(f"--{k}", type=Path, required=True)
    ap.add_argument("--res", type=int, default=1536)
    ap.add_argument("--cover", type=float, default=4.0, help="map width, x object height (must hold the ground shadow)")
    ap.add_argument("--far", type=float, default=200.0)
    ap.add_argument("--eps", type=float, default=0.03, help="depth margin (units; 0.03 = 3 mm on bear 2)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--opacity", type=float, default=0.1, help="surfels below this opacity are left out of the map")
    ap.add_argument("--object-casters", action="store_true",
                    help="only object surfels (crop, above the ground) cast shadows; the ground cannot shadow itself")
    ap.add_argument("--ring", type=float, nargs=2, default=[0.3, 1.5])
    ap.add_argument("--radius", type=float, default=2.5, help="surfels within this x object height")
    ap.add_argument("--el-range", type=float, default=20.0)
    ap.add_argument("--az-range", type=float, default=30.0)
    ap.add_argument("--step", type=float, default=2.5)
    ap.add_argument("--frames", type=int, default=3)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    t0 = time.time()

    obj = json.loads(a.object.read_text())
    H = obj["height"]
    gp = json.loads(a.plane_from.read_text())["ground_plane"]
    up = np.array(gp["normal"], float)
    up /= np.linalg.norm(up)
    s0 = np.load(a.__dict__["from"])["sun"].astype(float)
    s0 /= np.linalg.norm(s0)
    e1 = s0 - (s0 @ up) * up
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    el0 = np.degrees(np.arcsin(s0 @ up))
    target = np.array(obj["crop"][:3], float)
    foot = np.array(obj["footprint_centre"], float)

    p = {k: v.to(dev) for k, v in torch.load(a.surfels).items()}
    keep = (torch.sigmoid(p["opacities"]) >= a.opacity) & \
           ((p["means"] - torch.tensor(target, dtype=torch.float32, device=dev)).norm(dim=1) < a.radius * H)
    if a.object_casters:  # only the object can cast: surfels inside the object crop and clearly above the ground
        h = p["means"] @ torch.tensor(up, dtype=torch.float32, device=dev) + float(gp["offset"])
        keep &= (h > 2 * float(gp["tol"])) & \
            ((p["means"] - torch.tensor(target, dtype=torch.float32, device=dev)).norm(dim=1) < float(obj["crop"][3]))
    p = {k: v[keep] for k, v in p.items()}
    print(f"{int(keep.sum()):,} surfels in the sun map", flush=True)
    sm = torch.as_tensor(p["means"], dtype=torch.float32)

    _, g = read_ply(a.ply)
    xyz = np.c_[g["x"], g["y"], g["z"]].astype(float)
    vt = np.load(a.votes)
    rec = vt["rec_idx"]
    rr = np.linalg.norm(xyz[rec] - foot, axis=1) / H
    ok = (vt["seen"] >= 5) & (rr > a.ring[0]) & (rr < a.ring[1])
    rec = rec[ok]
    obs = (vt["shadow"][ok] / np.maximum(vt["seen"][ok], 1)) > 0.5
    rec_t = torch.tensor(xyz[rec], dtype=torch.float32, device=dev)
    obs_t = torch.tensor(obs, device=dev)
    N = a.res
    f = N * a.far / (a.cover * H)
    K = torch.tensor([[f, 0, N / 2], [0, f, N / 2], [0, 0, 1]], dtype=torch.float32, device=dev)

    def sun_dir(el, az):
        el, az = np.radians(el), np.radians(az)
        return np.cos(el) * (np.cos(az) * e1 + np.sin(az) * e2) + np.sin(el) * up

    def predict(s):
        cu = up - (up @ s) * s
        w2c = look_at(target + a.far * s, target, cu / np.linalg.norm(cu))
        vm = torch.tensor(w2c, dtype=torch.float32, device=dev)
        with torch.no_grad():
            out = render(p, vm, K, N, N, 0)
            depth = torch.where(out[1][0, ..., 0] > 0.5, out[5][0, ..., 0], torch.full((N, N), float("inf"), device=dev))
            # the receiver point itself against the map: ground surfels form a layer, so testing surfels marks the
            # buried ones as shadowed everywhere; a caster on the object is centimetres above, far beyond --eps
            pc = rec_t @ vm[:3, :3].T + vm[:3, 3]
            u = (f * pc[:, 0] / pc[:, 2] + N / 2).round().long()
            v = (f * pc[:, 1] / pc[:, 2] + N / 2).round().long()
            inside = (u >= 0) & (u < N) & (v >= 0) & (v < N)
            d = torch.full_like(pc[:, 2], float("inf"))
            d[inside] = depth[v[inside], u[inside]]
            return d < pc[:, 2] - a.eps

    def iou(pred):
        inter = (pred & obs_t).sum().item()
        return inter / max((pred | obs_t).sum().item(), 1), pred.float().mean().item(), inter / max(pred.sum().item(), 1), \
            inter / max(obs_t.sum().item(), 1)

    start = iou(predict(s0))
    print(f"{len(rec):,} receivers, observed shadow {obs.mean():.1%}; start sun elevation {el0:.2f}: IoU {start[0]:.3f} "
          f"(predicted {start[1]:.1%}, precision {start[2]:.3f}, recall {start[3]:.3f})", flush=True)
    grid = []
    best = (start[0], el0, 0.0)
    for rnd, (rng_el, rng_az, step) in enumerate([(a.el_range, a.az_range, a.step), (a.step, a.step, a.step / 5)]):
        ce, ca = best[1], best[2]
        for el in np.arange(ce - rng_el, ce + rng_el + 1e-6, step):
            if not 5 <= el <= 89:
                continue
            for az in np.arange(ca - rng_az, ca + rng_az + 1e-6, step):
                sc = iou(predict(sun_dir(el, az)))
                grid.append([round(float(el), 3), round(float(az), 3), round(sc[0], 4), round(sc[1], 4)])
                if sc[0] > best[0]:
                    best = (sc[0], el, az)
        print(f"round {rnd + 1}: best IoU {best[0]:.3f} at elevation {best[1]:.2f}, azimuth offset {best[2]:+.2f} deg "
              f"({len(grid)} suns, {time.time() - t0:.0f} s)", flush=True)
    s1 = sun_dir(best[1], best[2])
    fin = iou(predict(s1))
    res = {"start": {"sun": s0.tolist(), "elevation": el0, "iou": start[0], "predicted": start[1], "precision": start[2],
                     "recall": start[3]},
           "best": {"sun": s1.tolist(), "elevation": float(best[1]), "azimuth_offset": float(best[2]), "iou": fin[0],
                    "predicted": fin[1], "precision": fin[2], "recall": fin[3]},
           "observed_share": float(obs.mean()), "grid": grid, "args": {k: str(v) for k, v in vars(a).items()}}
    (a.out / "sun_refit.json").write_text(json.dumps(res, indent=1))
    np.savez(a.out / "sun_refit.npz", sun=s1)
    print(f"best sun {s1.round(4)}: elevation {best[1]:.2f} (start {el0:.2f}), azimuth {best[2]:+.2f} deg from start; "
          f"IoU {fin[0]:.3f}, predicted {fin[1]:.1%} vs observed {obs.mean():.1%}, precision {fin[2]:.3f}, recall {fin[3]:.3f}")

    # overlays: project the ring receivers into a few photos
    cams = read_cameras_bin(a.colmap / "cameras.bin")
    ims = read_images_bin(a.colmap / "images.bin")
    ims = sorted(ims.values() if isinstance(ims, dict) else ims, key=lambda f: f["name"])
    pred0, pred1 = predict(s0).cpu().numpy(), predict(s1).cpu().numpy()
    for fi in np.linspace(0, len(ims) - 1, a.frames + 2)[1:-1].astype(int):
        im = ims[fi]
        c = cams[im["cam"]]
        R = qvec2rotmat(im["q"])
        pc = xyz[rec] @ R.T + im["t"]
        fx, fy, cx, cy = c["params"][:4]
        u, v = fx * pc[:, 0] / pc[:, 2] + cx, fy * pc[:, 1] / pc[:, 2] + cy
        tiles = []
        for name, pr in (("start sun", pred0), ("refitted sun", pred1)):
            photo = Image.open(a.images / im["name"]).convert("RGB")
            d = ImageDraw.Draw(photo)
            for uu, vv, z, o, q in zip(u, v, pc[:, 2], obs, pr):
                if z <= 0 or not (o or q):
                    continue
                col = (255, 255, 255) if (o and q) else ((255, 40, 40) if q else (40, 120, 255))
                d.point((uu, vv), fill=col)
            d.text((10, 10), f"{name}: red = predicted only, blue = photo only, white = both", fill=(255, 255, 0))
            tiles.append(photo.resize((photo.width // 2, photo.height // 2)))
        strip = Image.new("RGB", (tiles[0].width * 2, tiles[0].height))
        strip.paste(tiles[0], (0, 0))
        strip.paste(tiles[1], (tiles[0].width, 0))
        strip.save(a.out / f"overlay_{im['name'].rsplit('.', 1)[0]}.jpg", quality=90)
    print(f"overlays written ({time.time() - t0:.0f} s) -> {a.out}")


if __name__ == "__main__":
    main()
