#!/usr/bin/env python
"""Give each normal (3DGS) splat Gaussian the surface direction of the nearest 2DGS surfels, and count how many are confident.

For every Gaussian in the object crop (opacity >= --min-opacity):
- borrowed normal: sign-aligned, opacity-weighted mean of the --k nearest surfels' normals (surfels from
  train_2dgs_ref.py, already turned to face the cameras that see them);
- confident when the nearest surfel is within --max-dist AND at least --share of the k surfels lie within --agree
  degrees of the borrowed normal (the surfel field is one consistent surface there, not a crease or noise).
For comparison the old rule from cluster_surfaces.py (--normal-source agree) on the same Gaussians: flat
(shortest / middle scale <= --flat) and shortest-axis vs PCA-of-16-neighbours normal within --agree degrees.
Numbers are given for the whole crop and for the object alone (height above the ground plane > 2 x plane tol).

Usage: borrow_normals.py --ply exports/IMG_6705/splat.ply --surfels exports/IMG_6705_2dgs/surfels.npz
           --object exports/IMG_6705_object/object.json --plane-from exports/IMG_6705_shadowfix_o/stats.json
           --out exports/IMG_6705_2dgs_borrow
Writes <out>/borrowed.npz (index into the ply, normal, confident), stats.json, and normals_*.ply point clouds
(colour = normal, grey = not confident) for viewing.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from fix_splat_shadows import qvec2rotmat, read_ply


def angle(a, b):
    return np.degrees(np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", a, b)), 0, 1)))


def write_points(path, xyz, n, ok):
    rgb = ((n * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    rgb[~ok] = 110
    v = np.empty(len(xyz), dtype=[(k, "<f4") for k in ("x", "y", "z", "nx", "ny", "nz")]
                 + [(k, "u1") for k in ("red", "green", "blue")])
    for k, col in zip(("x", "y", "z", "nx", "ny", "nz"), np.c_[xyz, n].T):
        v[k] = col
    v["red"], v["green"], v["blue"] = rgb.T
    props = "".join(f"property float {k}\n" for k in ("x", "y", "z", "nx", "ny", "nz"))
    props += "".join(f"property uchar {k}\n" for k in ("red", "green", "blue"))
    with open(path, "wb") as f:
        f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\n{props}end_header\n".encode())
        f.write(v.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--surfels", type=Path, required=True)
    ap.add_argument("--object", type=Path, required=True, help="object.json from find_object.py (crop)")
    ap.add_argument("--plane-from", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-opacity", type=float, default=0.3)
    ap.add_argument("--surfel-opacity", type=float, default=0.3, help="surfels used as reference")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-dist", type=float, default=0.05, help="nearest surfel distance (units)")
    ap.add_argument("--agree", type=float, default=30.0)
    ap.add_argument("--share", type=float, default=0.75)
    ap.add_argument("--flat", type=float, default=0.3, help="old rule: max shortest/middle scale")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    _, g = read_ply(a.ply)
    xyz = np.c_[g["x"], g["y"], g["z"]].astype(float)
    op = 1 / (1 + np.exp(-g["opacity"].astype(float)))
    cx, cy, cz, cr = json.loads(a.object.read_text())["crop"]
    sel = np.flatnonzero((np.linalg.norm(xyz - [cx, cy, cz], axis=1) < cr) & (op >= a.min_opacity))
    gp = json.loads(a.plane_from.read_text())["ground_plane"]
    up = np.array(gp["normal"], float)
    up /= np.linalg.norm(up)
    on_obj = (xyz[sel] @ up + gp["offset"]) > 2 * gp["tol"]
    p = xyz[sel]

    s = np.load(a.surfels)
    keep = (s["opacity"] >= a.surfel_opacity) & (np.linalg.norm(s["means"] - [cx, cy, cz], axis=1) < 1.5 * cr)
    sm, sn, so = s["means"][keep], s["normals"][keep], s["opacity"][keep]
    d, nb = cKDTree(sm).query(p, k=a.k)
    ref = sn[nb[:, 0]]
    nk = sn[nb] * np.sign(np.einsum("ikj,ij->ik", sn[nb], ref))[..., None]  # align signs to the nearest surfel
    bn = (nk * so[nb][..., None]).sum(1)
    bn /= np.linalg.norm(bn, axis=1, keepdims=True)
    within = (np.degrees(np.arccos(np.clip(np.einsum("ikj,ij->ik", nk, bn), -1, 1))) <= a.agree).mean(1)
    conf = (d[:, 0] <= a.max_dist) & (within >= a.share)

    # old rule on the same Gaussians
    q = np.c_[g["rot_0"], g["rot_1"], g["rot_2"], g["rot_3"]][sel].astype(float)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    sc = np.exp(np.c_[g["scale_0"], g["scale_1"], g["scale_2"]][sel].astype(float))
    order = np.argsort(sc, axis=1)
    R = np.stack([qvec2rotmat(x) for x in q])
    axis_n = R[np.arange(len(R)), :, order[:, 0]]
    flat = sc[np.arange(len(sc)), order[:, 0]] / sc[np.arange(len(sc)), order[:, 1]] <= a.flat
    _, nb16 = cKDTree(p).query(p, k=16)
    P = p[nb16] - p[nb16].mean(1, keepdims=True)
    pca = np.linalg.eigh(np.einsum("ikj,ikl->ijl", P, P))[1][:, :, 0]
    old_conf = flat & (angle(axis_n, pca) <= a.agree)

    def summary(m):
        n = int(m.sum())
        return {"gaussians": n, "confident_2dgs": float(conf[m].mean()), "confident_old_rule": float(old_conf[m].mean()),
                "nearest_surfel_median": float(np.median(d[m, 0])), "nearest_surfel_p90": float(np.percentile(d[m, 0], 90)),
                "borrowed_vs_pca_median_deg": float(np.median(angle(bn[m], pca[m]))),
                "borrowed_vs_axis_median_deg": float(np.median(angle(bn[m], axis_n[m]))),
                "axis_vs_pca_median_deg": float(np.median(angle(axis_n[m], pca[m])))}

    stats = {"surfels_used": int(keep.sum()), "crop": summary(np.ones(len(p), bool)), "object": summary(on_obj),
             "ground": summary(~on_obj), "args": {k: str(v) for k, v in vars(a).items()}}
    (a.out / "stats.json").write_text(json.dumps(stats, indent=1))
    np.savez_compressed(a.out / "borrowed.npz", index=sel, normal=bn.astype(np.float32), confident=conf,
                        nearest=d[:, 0].astype(np.float32), on_object=on_obj)
    write_points(a.out / "normals_2dgs.ply", p, bn, conf)
    write_points(a.out / "normals_old_rule.ply", p, pca, old_conf)
    for k in ("object", "ground"):
        r = stats[k]
        print(f"{k}: {r['gaussians']:,} Gaussians; confident 2DGS {r['confident_2dgs']:.1%} vs old rule "
              f"{r['confident_old_rule']:.1%}; nearest surfel median {r['nearest_surfel_median']:.4f} "
              f"(p90 {r['nearest_surfel_p90']:.4f}); borrowed vs PCA {r['borrowed_vs_pca_median_deg']:.1f} deg, "
              f"axis vs PCA {r['axis_vs_pca_median_deg']:.1f} deg")


if __name__ == "__main__":
    main()
