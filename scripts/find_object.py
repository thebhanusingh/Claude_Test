#!/usr/bin/env python
"""Find the hero object of an orbit video in its splat, and the scene scale from its known real height.

1. Focus: least-squares point closest to all camera optical axes (the object the video orbits).
2. Opaque Gaussians (opacity > --opaque) between --min-height x plane tol and the median camera height above the
   ground plane, within half the median camera-to-focus distance, are binned on a ground-plane grid; the 8-connected
   occupied component nearest the focus is the object.
3. Height H = 99.5th percentile of its heights above the plane; horizontal radius = 99th percentile of the distance
   from its footprint centre. Crop = (footprint centre lifted to H/2, 1.04 x sqrt(radius^2 + (H/2)^2)), as used by
   the stage scripts. Scale: units/m = H / --height-m; length factor vs a reference scene k = (units/m) / ref units/m.

Usage: find_object.py --ply exports/S/splat.ply --colmap data/S/colmap/sparse/0_refined
           --plane-from exports/S_shadowfix_o/stats.json --height-m 0.16 --ref-units-per-m 12.17 --out exports/S_object
Writes <out>/object.json (crop, height, radius, units_per_m, k, ...) and object_map.png.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from fix_splat_shadows import qvec2rotmat, read_images_bin, read_ply


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--colmap", type=Path, required=True)
    ap.add_argument("--plane-from", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--opaque", type=float, default=0.5)
    ap.add_argument("--min-height", type=float, default=2.0, help="in multiples of the plane's tol")
    ap.add_argument("--cells", type=int, default=200, help="grid cells across the search radius")
    ap.add_argument("--min-count", type=int, default=3, help="Gaussians for a cell to count as occupied")
    ap.add_argument("--height-m", type=float, help="real object height in metres")
    ap.add_argument("--ref-units-per-m", type=float, help="reference scene's units/m for the length factor k")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    _, data = read_ply(a.ply)
    xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
    opacity = 1 / (1 + np.exp(-data["opacity"].astype(np.float64)))
    gp = json.loads(a.plane_from.read_text())["ground_plane"]
    up, offset, tol = np.array(gp["normal"], float), float(gp["offset"]), float(gp["tol"])
    up /= np.linalg.norm(up)

    imgs = read_images_bin(a.colmap / "images.bin")
    R = [qvec2rotmat(f["q"]) for f in imgs]
    C = np.array([-r.T @ f["t"] for r, f in zip(R, imgs)])
    D = np.array([r.T @ [0, 0, 1.0] for r in R])
    A = np.zeros((3, 3)); b = np.zeros(3)
    for c, d in zip(C, D):
        P = np.eye(3) - np.outer(d, d)
        A += P; b += P @ c
    focus = np.linalg.solve(A, b)
    cam_h = np.median(C @ up + offset)
    cam_d = np.median(np.linalg.norm(C - focus, axis=1))
    foot0 = focus - (focus @ up + offset) * up

    e1 = np.cross(up, [1.0, 0, 0]) if abs(up[0]) < 0.9 else np.cross(up, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1)
    h = xyz @ up + offset
    uv = np.c_[(xyz - foot0) @ e1, (xyz - foot0) @ e2]
    search = 0.5 * cam_d
    cand = (opacity > a.opaque) & (h > a.min_height * tol) & (h < cam_h) & (np.linalg.norm(uv, axis=1) < search)
    cell = 2 * search / a.cells
    ij = np.clip(((uv[cand] + search) / cell).astype(int), 0, a.cells - 1)
    cnt = np.zeros((a.cells, a.cells), int)
    np.add.at(cnt, (ij[:, 0], ij[:, 1]), 1)
    occ = ndimage.binary_closing(cnt >= a.min_count, iterations=1)
    lab, n = ndimage.label(occ, structure=np.ones((3, 3)))
    if n == 0:
        raise SystemExit("ERROR: nothing above the ground near the camera focus")
    cells = np.argwhere(lab > 0)
    centre_cell = np.array([a.cells / 2, a.cells / 2])
    pick = lab[tuple(cells[np.argmin(np.linalg.norm(cells + 0.5 - centre_cell, axis=1))])]
    in_obj = np.zeros(len(xyz), bool)
    in_obj[np.flatnonzero(cand)[lab[ij[:, 0], ij[:, 1]] == pick]] = True

    H = float(np.percentile(h[in_obj], 99.5))
    cuv = uv[in_obj].mean(0)
    rad = float(np.percentile(np.linalg.norm(uv[in_obj] - cuv, axis=1), 99))
    ground_c = foot0 + cuv[0] * e1 + cuv[1] * e2
    centre = ground_c + up * (H / 2)
    crop_r = 1.04 * float(np.hypot(rad, H / 2))
    res = dict(focus=focus.round(4).tolist(), camera_height=round(float(cam_h), 4), camera_distance=round(float(cam_d), 4),
               gaussians=int(in_obj.sum()), components=int(n), height=round(H, 4), radius=round(rad, 4),
               footprint_centre=ground_c.round(4).tolist(), crop=[*centre.round(4).tolist(), round(crop_r, 4)])
    if a.height_m:
        res["height_m"] = a.height_m
        res["units_per_m"] = round(H / a.height_m, 4)
        if a.ref_units_per_m:
            res["ref_units_per_m"] = a.ref_units_per_m
            res["k"] = round(res["units_per_m"] / a.ref_units_per_m, 4)
    (a.out / "object.json").write_text(json.dumps(res, indent=1))
    np.save(a.out / "object_mask.npy", in_obj)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(np.where(lab == pick, 2, occ.astype(int)).T, origin="lower", cmap="Greys",
              extent=[-search, search, -search, search])
    cu, cv_ = (C - foot0) @ e1, (C - foot0) @ e2
    ax.plot(cu, cv_, ".", ms=2, color="tab:blue", label="cameras")
    ax.plot(0, 0, "r+", ms=12, label="focus"); ax.plot(*cuv, "gx", ms=10, label="object centre")
    ax.add_patch(plt.Circle(cuv, rad, fill=False, color="g"))
    ax.set_xlim(-search, search); ax.set_ylim(-search, search); ax.set_aspect("equal"); ax.legend(loc="upper right")
    ax.set_title(f"object: H {H:.3f} units, radius {rad:.3f}" + (f", {res['units_per_m']:.2f} units/m" if a.height_m else ""))
    fig.tight_layout(); fig.savefig(a.out / "object_map.png", dpi=100)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
