#!/usr/bin/env python
"""Stage 5 prototype: orientation-guided fill of shadowed Gaussians on the hero object.

For each shadowed object Gaussian, borrow colour from LIT Gaussians of the SAME stage-3 cluster (surface), preferring
donors that lie along the surface grain: donors are the --k nearest in an anisotropic distance where offsets across the
grain cost --across times more than offsets along it. Rule from the brief: same surface and lit, not brightest nearby.
  --mode copy  : the shadowed Gaussian's colour (all SH coefficients) becomes the distance-weighted mean of its donors'.
  --mode ratio : texture-preserving variant: the Gaussian keeps its own colour variation and is multiplied (in linear
                 RGB) by donors' mean colour / the mean colour of shadowed neighbours along the grain (same cluster).
Clusters with fewer than --min-donors lit Gaussians (surfaces that are almost entirely in shadow) have no lit
counterpart on the same surface; they fall back to the object's lit Gaussians nearest along the grain (any surface).
This breaks the same-surface rule, so fallback use is reported per cluster.
Only colour changes; geometry, opacity and every other property are copied. The input splat is never modified.

Usage:
  fill_shadows.py --ply exports/IMG_6556/splat.ply --labels exports/IMG_6556_chairshadow_adj_r04/chair_shadow.npz \
      --clusters exports/IMG_6556_clusters_chair_r30agree_v3/clusters.npz --out exports/IMG_6556_fill_v1 [--mode copy]
Writes <out>/splat_fill.ply (full scene), chair_before.ply / chair_after.ply (object only), stats.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from cluster_surfaces import SH_C0, read_ply, write_ply


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True, help="chair_shadow.npz (idx, shadow, cos, geometry)")
    ap.add_argument("--clusters", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=["copy", "ratio"], default="copy")
    ap.add_argument("--graze", type=float, default=0.1, help="|n.s| below this: geometry label decides")
    ap.add_argument("--k", type=int, default=8, help="donors per shadowed Gaussian")
    ap.add_argument("--across", type=float, default=4.0, help="cost of across-grain offset relative to along-grain")
    ap.add_argument("--min-donors", type=int, default=50, help="lit Gaussians a cluster needs to serve as donor pool")
    ap.add_argument("--donor-min", type=float, default=0.0,
                    help="if > 0: donors must be at least this fraction of the surface's median lit brightness")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    header, data = read_ply(a.ply)
    lab = np.load(a.labels)
    idx, sh_b, cos, geo = lab["idx"], lab["shadow"], lab["cos"], lab["geometry"]
    shadow = np.where(np.abs(cos) < a.graze, geo, sh_b)
    z = np.load(a.clusters)
    cl, grain = z["labels"][idx], z["local_dir"][idx]
    gnorm = np.linalg.norm(grain, axis=1)
    cdir = z["cluster_dir"]
    grain = np.where(gnorm[:, None] > 0.5, grain / np.maximum(gnorm[:, None], 1e-9), cdir[cl])  # no local grain: cluster's
    xyz = np.stack([data["x"][idx], data["y"][idx], data["z"][idx]], 1).astype(np.float64)
    names_sh = [f"f_dc_{i}" for i in range(3)] + sorted([n for n in data.dtype.names if n.startswith("f_rest_")],
                                                        key=lambda s: int(s.split("_")[-1]))
    SH = np.stack([data[n][idx] for n in names_sh], 1).astype(np.float64)
    rgb_lin = srgb_to_linear(np.clip(SH[:, :3] * SH_C0 + 0.5, 0, 1))
    print(f"{len(idx):,} object Gaussians, {shadow.sum():,} shadowed ({shadow.mean() * 100:.1f}%), mode {a.mode}", flush=True)

    def donors_for(si, pool, along):
        """k donors from pool for shadowed set si, anisotropic distance (stretch across-grain offsets)."""
        # transform: along-grain component kept, across-grain components scaled by `across`, per query grain.
        # Exact anisotropic kNN is expensive; approximate: take 4k Euclidean candidates, re-rank anisotropically.
        tree = cKDTree(xyz[pool])
        kk = min(len(pool), 4 * a.k)
        d, nb = tree.query(xyz[si], k=kk)
        nb = np.atleast_2d(nb)
        off = xyz[pool][nb] - xyz[si][:, None, :]
        g = along[:, None, :]
        al = np.einsum("nkj,nkj->nk", off, np.broadcast_to(g, off.shape))
        acr2 = np.maximum(np.einsum("nkj,nkj->nk", off, off) - al ** 2, 0)
        dist = np.sqrt(al ** 2 + (a.across ** 2) * acr2)
        order = np.argsort(dist, 1)[:, : a.k]
        sel = np.take_along_axis(nb, order, 1)
        w = 1 / np.maximum(np.take_along_axis(dist, order, 1), 1e-4)
        return pool[sel], w / w.sum(1, keepdims=True)

    new_SH = SH.copy()
    fallback = np.zeros(len(idx), bool)
    lit_all = np.flatnonzero(~shadow)
    per = []
    for c in np.unique(cl):
        m = cl == c
        si = np.flatnonzero(m & shadow)
        if len(si) == 0:
            continue
        pool = np.flatnonzero(m & ~shadow)
        if a.donor_min > 0 and len(pool):  # skip dim 'lit' donors (penumbra, grazing sun, mislabels)
            lum_p = rgb_lin[pool] @ [0.2126, 0.7152, 0.0722]
            pool = pool[lum_p >= a.donor_min * np.median(lum_p)]
        fb = len(pool) < a.min_donors
        if fb:
            pool = lit_all
            fallback[si] = True
        dn, w = donors_for(si, pool, grain[si])
        if a.mode == "copy":
            new_SH[si] = np.einsum("nk,nkj->nj", w, SH[dn])
        else:
            donor_lin = np.einsum("nk,nkj->nj", w, rgb_lin[dn])
            sh_pool = np.flatnonzero(m & shadow)
            sdn, sw = donors_for(si, sh_pool, grain[si])  # shadowed neighbours along the grain (texture baseline)
            shadow_lin = np.einsum("nk,nkj->nj", sw, rgb_lin[sdn])
            gain = donor_lin / np.maximum(shadow_lin, 1e-4)
            out_lin = np.clip(rgb_lin[si] * gain, 0, 1)
            new_SH[si, :3] = (linear_to_srgb(out_lin) - 0.5) / SH_C0
            new_SH[si, 3:] = np.einsum("nk,nkj->nj", w, SH[dn][:, :, 3:])
        per.append(dict(cluster=int(c), shadowed=int(len(si)), donors_available=int((m & ~shadow).sum()),
                        fallback=bool(fb)))
        print(f"  cluster {c:3d}: {len(si):6,d} shadowed, {(m & ~shadow).sum():6,d} lit in the same surface"
              f"{'  -> FALLBACK (too few lit on this surface)' if fb else ''}", flush=True)

    before_lum = (rgb_lin[shadow] @ [0.2126, 0.7152, 0.0722])
    after_lin = srgb_to_linear(np.clip(new_SH[:, :3] * SH_C0 + 0.5, 0, 1))
    after_lum = after_lin[shadow] @ [0.2126, 0.7152, 0.0722]
    lit_lum = rgb_lin[~shadow] @ [0.2126, 0.7152, 0.0722]
    print(f"brightness (linear, median): shadowed before {np.median(before_lum):.4f}, after {np.median(after_lum):.4f}, "
          f"lit {np.median(lit_lum):.4f}; fallback used for {fallback[shadow].mean() * 100:.1f}% of shadowed", flush=True)

    out = data.copy()
    for j, n in enumerate(names_sh):
        out[n][idx] = new_SH[:, j].astype(out[n].dtype)
    write_ply(a.out / "splat_fill.ply", header, out)
    write_ply(a.out / "chair_before.ply", header, data[idx])
    write_ply(a.out / "chair_after.ply", header, out[idx])
    stats = dict(mode=a.mode, object_gaussians=int(len(idx)), shadowed=int(shadow.sum()),
                 fallback_share_of_shadowed=round(float(fallback[shadow].mean()), 4),
                 median_brightness=dict(shadowed_before=float(np.median(before_lum)),
                                        shadowed_after=float(np.median(after_lum)), lit=float(np.median(lit_lum))),
                 per_cluster=per, seconds=round(time.time() - t0, 1),
                 params={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()})
    (a.out / "stats.json").write_text(json.dumps(stats, indent=1))
    np.savez_compressed(a.out / "fill.npz", idx=idx, shadow=shadow, fallback=fallback)
    print(f"wrote {a.out}: splat_fill.ply, chair_before.ply, chair_after.ply, stats.json ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
