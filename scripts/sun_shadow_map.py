#!/usr/bin/env python
"""Which blobs of the object can see the sun? A fine shadow map rendered from the sun through the 2DGS surfels.

The 2DGS surfels (train_2dgs_ref.py) are thin and sit on the real surface, so a depth map rendered from the sun's
direction has no splat-thickness problem and needs no "a surface cannot shadow itself" rule. The sun camera sits
--far units away along the sun direction with a long focal length (near-orthographic), covering --cover x the object
height around the object at --res pixels (about 0.1-0.2 mm per pixel for bear 2). Depth = 2DGS median depth.

Each object blob (normal from borrow_normals.py) is pushed --lift along its normal toward the sun, projected, and
compared with the map over a 3 x 3 pixel window (fraction of pixels that are closer to the sun by more than --eps):
  sun_visible = 1 - that fraction   (soft at shadow edges);  attached = normal faces away from the sun.
Writes <out>/sun_visibility.npz (index into the ply, visible, attached, n.s), shadow_map.png, stats.json and
<out>/splat_shadow_labels.ply: the full splat with object blobs recoloured (cast shadow magenta, attached shade blue,
lit unchanged) for render_orbit.py.

Usage: sun_shadow_map.py --surfels exports/IMG_6705_2dgs_fromsplat/surfels_params.pt --ply exports/IMG_6705/splat.ply
           --borrow exports/IMG_6705_2dgs_borrow/borrowed.npz --sun-from exports/IMG_6705_shadow4/shadow_flags.npz
           --object exports/IMG_6705_object/object.json --plane-from exports/IMG_6705_shadowfix_o/stats.json
           --out exports/IMG_6705_sunmap
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from fix_splat_shadows import read_ply, write_ply
from render_orbit import look_at
from train_2dgs_ref import render

SH_C0 = 0.28209479177387814


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surfels", type=Path, required=True)
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--borrow", type=Path, required=True)
    ap.add_argument("--sun-from", type=Path, required=True)
    ap.add_argument("--object", type=Path, required=True)
    ap.add_argument("--plane-from", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--res", type=int, default=3072)
    ap.add_argument("--cover", type=float, default=1.6, help="map width, x object height")
    ap.add_argument("--far", type=float, default=200.0)
    ap.add_argument("--eps", type=float, default=0.012, help="depth margin (units; bear 2: 0.012 = 1.3 mm)")
    ap.add_argument("--lift", type=float, default=0.006, help="push toward the sun along the normal (units)")
    ap.add_argument("--opacity", type=float, default=0.1, help="surfels below this opacity are left out of the map")
    ap.add_argument("--k", type=int, default=8, help="nearest surfels per blob")
    ap.add_argument("--sun-npz", type=Path, help="sun from sun_refit.py (overrides the sun in --sun-from)")
    ap.add_argument("--object-casters", action="store_true", help="only object surfels cast shadows")
    ap.add_argument("--test", choices=["surfel", "point"], default="surfel",
                    help="surfel: nearest surfels' answer; point: the blob position itself (use --eps about 0.03)")
    ap.add_argument("--votes", type=Path, help="fit_sun votes.npz: check the predicted ground shadow against the photos")
    ap.add_argument("--radius", type=float, default=2.0, help="surfels within this x object height of the object only")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"

    obj = json.loads(a.object.read_text())
    gp = json.loads(a.plane_from.read_text())["ground_plane"]
    up = np.array(gp["normal"], float)
    up /= np.linalg.norm(up)
    s = np.load(a.sun_npz or a.sun_from)["sun"].astype(float)
    s /= np.linalg.norm(s)
    target = np.array(obj["crop"][:3], float)

    p = {k: v.to(dev) for k, v in torch.load(a.surfels).items()}
    tgt = torch.tensor(target, dtype=torch.float32, device=dev)
    keep = (torch.sigmoid(p["opacities"]) >= a.opacity) & \
           ((p["means"] - tgt).norm(dim=1) < a.radius * obj["height"])  # no far floaters between object and sun
    if a.object_casters:  # only the object casts (the ground and stray surfels were false blockers)
        h = p["means"] @ torch.tensor(up, dtype=torch.float32, device=dev) + float(gp["offset"])
        keep &= (h > 2 * float(gp["tol"])) & ((p["means"] - tgt).norm(dim=1) < float(obj["crop"][3]))
    p = {k: v[keep] for k, v in p.items()}
    pos = target + a.far * s
    cam_up = up - (up @ s) * s
    cam_up /= np.linalg.norm(cam_up)
    w2c = look_at(pos, target, cam_up)
    N = a.res
    f = N * a.far / (a.cover * obj["height"])
    K = torch.tensor([[f, 0, N / 2], [0, f, N / 2], [0, 0, 1]], dtype=torch.float32, device=dev)
    vm = torch.tensor(w2c, dtype=torch.float32, device=dev)
    with torch.no_grad():
        out = render(p, vm, K, N, N, 0)
    alpha = out[1][0, ..., 0]
    depth = torch.where(alpha > 0.5, out[5][0, ..., 0], torch.full_like(alpha, np.inf))  # median depth
    px_mm = a.cover * obj["height"] / N / obj["units_per_m"] * 1000
    print(f"sun map {N} x {N}, {px_mm:.2f} mm per pixel, {keep.sum().item():,} surfels, covered {(alpha > 0.5).float().mean():.1%}")

    _, g = read_ply(a.ply)
    xyz = np.c_[g["x"], g["y"], g["z"]].astype(float)
    b = np.load(a.borrow)
    idx = b["index"][b["on_object"]]
    n = b["normal"][b["on_object"]].astype(float)
    ns = n @ s
    # test the SURFELS (they lie on the surface the map was made from), then give each blob its nearest surfels' answer:
    # blob centres sit up to a few mm off the surface and the borrowed normal signs are not reliable enough
    sm = p["means"].cpu().numpy().astype(float)
    q = xyz[idx] if a.test == "point" else sm  # point: each blob itself against the map (needs a margin > blob offset)
    pc = q @ w2c[:3, :3].T + w2c[:3, 3]
    u = f * pc[:, 0] / pc[:, 2] + N / 2
    v = f * pc[:, 1] / pc[:, 2] + N / 2
    z = torch.tensor(pc[:, 2], device=dev, dtype=torch.float32)
    blocked = torch.zeros(len(q), device=dev)
    cnt = torch.zeros(len(q), device=dev)
    gap = torch.full((len(q),), float("-inf"), device=dev)  # how far behind the nearest-to-sun surface (units)
    ui = torch.tensor(np.round(u), device=dev).long()
    vi = torch.tensor(np.round(v), device=dev).long()
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            uu, vv = ui + du, vi + dv
            ok = (uu >= 0) & (uu < N) & (vv >= 0) & (vv < N)
            d = torch.full_like(z, np.inf)
            d[ok] = depth[vv[ok], uu[ok]]
            blocked += ok * (d < z - a.eps)
            cnt += ok
            gap = torch.maximum(gap, torch.where(torch.isfinite(d), z - d, torch.full_like(z, float("-inf"))))
    svis = (1 - blocked / cnt.clamp_min(1)).cpu().numpy()  # per surfel: 1 = the sun sees it
    from scipy.spatial import cKDTree
    # surfels form a layer a few mm thick; only the outermost ones meet the map, so a blob is as lit as the most
    # lit of its --k nearest surfels (a real cast shadow covers all of them)
    if a.test == "point":
        vis = svis
    else:
        dd, nn = cKDTree(sm).query(xyz[idx], k=a.k)
        vis = svis[nn].max(1)
    if a.votes:  # sun check: predicted ground shadow vs the photo shadow masks (multi-view votes) near the object
        vt = np.load(a.votes)
        rec = vt["rec_idx"]
        rr = np.linalg.norm(xyz[rec] - np.array(obj["footprint_centre"]), axis=1) / obj["height"]
        okv = (vt["seen"] >= 5) & (rr < 1.5) & (rr > 0.3)
        obs = (vt["shadow"] / np.maximum(vt["seen"], 1))[okv] > 0.5
        _, gn = cKDTree(sm).query(xyz[rec[okv]], k=a.k)
        pred = svis[gn].max(1) < 0.5
        iou = (obs & pred).sum() / max((obs | pred).sum(), 1)
        print(f"ground check ({okv.sum():,} receivers within 1.5 x height): observed shadow {obs.mean():.1%}, "
              f"predicted {pred.mean():.1%}, IoU {iou:.3f} (precision {(obs & pred).sum() / max(pred.sum(), 1):.3f}, "
              f"recall {(obs & pred).sum() / max(obs.sum(), 1):.3f})")
    # "attached" (faces away) is only used as a label: a blob the sun cannot see whose normal points away from it
    attached = (vis < 0.5) & (ns <= 0)
    if a.test == "point":  # the map already hides the far side; the borrowed normal signs are not reliable labels
        attached = np.zeros(len(idx), bool)
    cast = (vis < 0.5) & ~attached
    np.savez_compressed(a.out / "sun_visibility.npz", index=idx, visible=vis.astype(np.float32), attached=attached,
                        ns=ns.astype(np.float32), gap=gap.cpu().numpy())
    d = depth.cpu().numpy()
    fin = np.isfinite(d)
    img = np.zeros_like(d)
    img[fin] = 1 - (d[fin] - d[fin].min()) / max(np.ptp(d[fin]), 1e-6)
    Image.fromarray((img * 255).astype(np.uint8)).resize((1024, 1024)).save(a.out / "shadow_map.png")

    lab = g.copy()
    col = np.stack([lab[f"f_dc_{c}"] for c in range(3)], 1)
    col[idx[cast]] = (np.array([0.95, 0.1, 0.85]) - 0.5) / SH_C0
    col[idx[attached]] = (np.array([0.15, 0.35, 0.95]) - 0.5) / SH_C0
    soft = (~attached) & (vis >= 0.5) & (vis < 1)
    col[idx[soft]] = (np.array([0.95, 0.6, 0.9]) - 0.5) / SH_C0
    for c in range(3):
        lab[f"f_dc_{c}"] = col[:, c].astype(np.float32)
    rest = [k for k in lab.dtype.names if k.startswith("f_rest_")]
    changed = np.zeros(len(lab), bool)
    changed[idx[cast | attached | soft]] = True
    for k in rest:
        lab[k][changed] = 0
    header, _ = read_ply(a.ply)
    write_ply(a.out / "splat_shadow_labels.ply", header, lab)

    old = np.load(a.sun_from)
    _, j = __import__("scipy.spatial", fromlist=["cKDTree"]).cKDTree(xyz[old["object_idx"]]).query(xyz[idx])
    old_cast = old["cast_object"][j]
    st = {"px_mm": round(px_mm, 3), "object_blobs": int(len(idx)), "attached_share": float(attached.mean()),
          "cast_share": float(cast.mean()), "soft_edge_share": float(soft.mean()),
          "lit_share": float(((~attached) & (vis >= 1)).mean()),
          "old_map_cast_share": float(old_cast.mean()),
          "agree_with_old_map": float((old_cast == cast)[~attached].mean()),
          "args": {k: str(v) for k, v in vars(a).items()}}
    (a.out / "stats.json").write_text(json.dumps(st, indent=1))
    print(f"object {len(idx):,} blobs: attached (faces away) {st['attached_share']:.1%}, cast shadow {st['cast_share']:.1%}, "
          f"soft edge {st['soft_edge_share']:.1%}, lit {st['lit_share']:.1%}; old map said cast {st['old_map_cast_share']:.1%} "
          f"(agreement on sun-facing blobs {st['agree_with_old_map']:.1%}) -> {a.out}")


if __name__ == "__main__":
    main()
