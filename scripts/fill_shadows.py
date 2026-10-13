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
    ap.add_argument("--mode", choices=["copy", "ratio", "continuous"], default="copy",
                    help="continuous: no yes/no labels; raise each Gaussian toward the lit level of its same-surface, "
                         "along-grain neighbourhood")
    ap.add_argument("--ref-k", type=int, default=64, help="continuous: neighbourhood size for the lit level")
    ap.add_argument("--ref-pct", type=float, default=80.0, help="continuous: percentile used as the lit level")
    ap.add_argument("--include-unclustered", action="store_true",
                    help="also fill unclustered / semi-transparent Gaussians on the object (needs --crop, --plane-from)")
    ap.add_argument("--crop", type=float, nargs=4, metavar=("X", "Y", "Z", "R"))
    ap.add_argument("--plane-from", type=Path)
    ap.add_argument("--min-opacity", type=float, default=0.05)
    ap.add_argument("--attach-dist", type=float, default=0.04)
    ap.add_argument("--dark-full", type=float, default=0.3, help="continuous: local/lit ratio below which gain is full")
    ap.add_argument("--dark-none", type=float, default=0.6, help="continuous: local/lit ratio above which no gain")
    ap.add_argument("--graze", type=float, default=0.1, help="|n.s| below this: geometry label decides")
    ap.add_argument("--k", type=int, default=8, help="donors per shadowed Gaussian")
    ap.add_argument("--across", type=float, default=4.0, help="cost of across-grain offset relative to along-grain")
    ap.add_argument("--min-donors", type=int, default=50, help="lit Gaussians a cluster needs to serve as donor pool")
    ap.add_argument("--soft", action="store_true",
                    help="anti-mottling: soft shadow amount smoothed in 3D, fill blended by it, gains smoothed along grain")
    ap.add_argument("--smooth-radius", type=float, default=0.04, help="label majority-vote radius (scene units)")
    ap.add_argument("--smooth-iters", type=int, default=2)
    ap.add_argument("--feather-radius", type=float, default=0.02, help="soft-edge radius after denoising")
    ap.add_argument("--gain-smooth", type=int, default=2, help="passes of along-grain smoothing of the fill")
    ap.add_argument("--max-gain", type=float, default=20.0, help="cap on the brightening ratio (ratio mode)")
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
    if a.include_unclustered:
        # The labelled set holds only clustered, opaque object Gaussians; many other Gaussians on the object (not in a
        # stage-3 cluster, or semi-transparent) also carry the baked shadow colour. Add every Gaussian inside --crop
        # above the ground with opacity > --min-opacity, giving it the cluster and grain of its nearest labelled
        # Gaussian (if within --attach-dist); its own shadow label is unknown (False), so use with --mode continuous.
        allxyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
        op = 1 / (1 + np.exp(-data["opacity"].astype(np.float64)))
        gp = json.loads(a.plane_from.read_text())["ground_plane"]
        up = np.array(gp["normal"], float); up /= np.linalg.norm(up)
        hgt = allxyz @ up + float(gp["offset"])
        cc, rr = np.array(a.crop[:3]), a.crop[3]
        cand = np.flatnonzero((np.linalg.norm(allxyz - cc, axis=1) < rr) & (hgt > 0.05) & (op > a.min_opacity))
        cand = np.setdiff1d(cand, idx)
        d, nb = cKDTree(xyz).query(allxyz[cand], k=1)
        ok = d < a.attach_dist
        cand, nb = cand[ok], nb[ok]
        print(f"including {len(cand):,} more Gaussians on the object (unclustered or semi-transparent, within "
              f"{a.attach_dist} of a labelled one); {(~ok).sum():,} further away left unchanged", flush=True)
        idx = np.r_[idx, cand]
        cl, grain = np.r_[cl, cl[nb]], np.r_[grain, grain[nb]]
        shadow = np.r_[shadow, np.zeros(len(cand), bool)]
        cos, geo = np.r_[cos, cos[nb]], np.r_[geo, geo[nb]]
        xyz = allxyz[idx]
    names_sh = [f"f_dc_{i}" for i in range(3)] + sorted([n for n in data.dtype.names if n.startswith("f_rest_")],
                                                        key=lambda s: int(s.split("_")[-1]))
    SH = np.stack([data[n][idx] for n in names_sh], 1).astype(np.float64)
    rgb_lin = srgb_to_linear(np.clip(SH[:, :3] * SH_C0 + 0.5, 0, 1))
    print(f"{len(idx):,} object Gaussians, {shadow.sum():,} shadowed ({shadow.mean() * 100:.1f}%), mode {a.mode}", flush=True)

    def donors_for(si, pool, along, floor=1e-4):
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
        w = 1 / np.maximum(np.take_along_axis(dist, order, 1), floor)
        return pool[sel], w / w.sum(1, keepdims=True)

    new_SH = SH.copy()
    fallback = np.zeros(len(idx), bool)
    lit_all = np.flatnonzero(~shadow)
    per = []
    alpha = shadow.astype(np.float64)
    if a.soft:
        # --- anti-mottling: soft shadow amount, smoothed in 3D within each surface ---
        # 1) denoise: majority vote of the labels within --smooth-radius on the same surface (--smooth-iters passes)
        # 2) feather: soft amount = mean of the denoised labels within --feather-radius (edges fade, interiors stay 0/1)
        lab_d = shadow.copy()
        for c in np.unique(cl):
            m = np.flatnonzero(cl == c)
            if len(m) < 2:
                continue
            tree = cKDTree(xyz[m])
            nbrs = tree.query_ball_point(xyz[m], a.smooth_radius)
            for _ in range(a.smooth_iters):
                lab_d[m] = np.array([lab_d[m][nb].mean() > 0.5 for nb in nbrs])
            fn = tree.query_ball_point(xyz[m], a.feather_radius)
            alpha[m] = np.array([lab_d[m][nb].mean() for nb in fn])
        print(f"labels: majority smoothing (radius {a.smooth_radius}, {a.smooth_iters} passes) changed "
              f"{(lab_d != shadow).mean() * 100:.1f}%; after feathering (radius {a.feather_radius}): "
              f"{(alpha > 0.95).mean() * 100:.1f}% full shadow, {((alpha > 0.05) & (alpha <= 0.95)).mean() * 100:.1f}% "
              f"edge, {(alpha <= 0.05).mean() * 100:.1f}% lit", flush=True)
        shadow = alpha > 0.05            # Gaussians that get (some) fill
        lit_all = np.flatnonzero(~lab_d & (alpha < 0.5))
    for c in np.unique(cl):
        if a.soft or a.mode == "continuous":
            break  # handled by the soft / continuous paths below

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

    if a.mode == "continuous":
        # No yes/no labels. Per Gaussian, along the grain on the same surface:
        #   L_s = local median brightness (k nearest, anisotropic)  -> the local illumination level (texture kept)
        #   R   = --ref-pct percentile of brightness over a wider along-grain neighbourhood (--ref-k) -> the lit level
        #   gain = clip(R / L_s, 1, max_gain), smoothed in log space along the grain; colour *= gain.
        # Surfaces whose lit level is far below the object's (no sunlit part) use the object's lit level (fallback).
        lum_w = np.array([0.2126, 0.7152, 0.0722])
        L = np.maximum(rgb_lin @ lum_w, 1e-4)
        logg = np.zeros(len(idx))
        glob_ref = np.percentile(L, a.ref_pct)
        for c in np.unique(cl):
            si = np.flatnonzero(cl == c)
            if len(si) < a.k:
                continue
            ka = a.k
            a.k = min(16, len(si))
            dn, _ = donors_for(si, si, grain[si], floor=0.02)
            L_s = np.median(L[dn], 1)
            a.k = min(a.ref_k, len(si))
            rn, _ = donors_for(si, si, grain[si], floor=0.02)
            R = np.percentile(L[rn], a.ref_pct, axis=1)
            a.k = ka
            fb = np.percentile(L[si], a.ref_pct) < 0.5 * glob_ref
            if fb:
                R = np.full(len(si), glob_ref)
                fallback[si] = True
            # only clearly darker neighbourhoods are raised: full gain when L_s/R < --dark-full, none above --dark-none
            q = L_s / np.maximum(R, 1e-6)
            wq = np.clip((a.dark_none - q) / (a.dark_none - a.dark_full), 0, 1)
            logg[si] = wq * np.log(np.clip(R / L_s, 1, a.max_gain))
            a.k = min(16, len(si))
            for _ in range(a.gain_smooth):
                gdn, gw = donors_for(si, si, grain[si], floor=0.02)
                logg[si] = np.einsum("nk,nk->n", gw, logg[gdn])
            a.k = ka
            per.append(dict(cluster=int(c), gaussians=int(len(si)), median_gain=round(float(np.exp(np.median(logg[si]))), 3),
                            raised=int((logg[si] > np.log(1.5)).sum()), fallback=bool(fb)))
            print(f"  cluster {c:3d}: {len(si):6,d} Gaussians, median gain {np.exp(np.median(logg[si])):.2f}, "
                  f"{(logg[si] > np.log(1.5)).sum():6,d} raised > 1.5x{'  -> FALLBACK (no sunlit part)' if fb else ''}",
                  flush=True)
        gain = np.exp(logg)[:, None]
        out_lin = np.clip(rgb_lin * gain, 0, 1)
        new_SH[:, :3] = (linear_to_srgb(out_lin) - 0.5) / SH_C0
        new_SH[:, 3:] = SH[:, 3:] * np.minimum(gain, 3)  # view-dependent part scaled with the base colour (capped)
        print(f"continuous: {(np.exp(logg) > 1.5).mean() * 100:.1f}% of the object raised more than 1.5x", flush=True)

    if a.soft and a.mode != "continuous":
        # donors: clearly lit (alpha <= 0.05) Gaussians of the same surface, nearest along the grain.
        # ratio mode: per-Gaussian gain = donors / core-shadow baseline, smoothed in log space along the grain within the
        # surface, applied as gain ** alpha (edges fade); copy mode: donor colour smoothed along the grain, blended by alpha.
        lum_w = np.array([0.2126, 0.7152, 0.0722])
        target = np.zeros((len(idx), 3))  # ratio: log-gain; copy uses SH directly
        fill_SH = SH.copy()
        for c in np.unique(cl):
            m = cl == c
            si = np.flatnonzero(m & shadow)
            if len(si) == 0:
                continue
            pool = np.flatnonzero(m & ~lab_d & (alpha < 0.5))
            if a.donor_min > 0 and len(pool):
                lum_p = rgb_lin[pool] @ lum_w
                pool = pool[lum_p >= a.donor_min * np.median(lum_p)]
            fb = len(pool) < a.min_donors
            if fb:
                pool = lit_all
                if a.donor_min > 0:
                    lum_p = rgb_lin[pool] @ lum_w
                    pool = pool[lum_p >= a.donor_min * np.median(lum_p)]
                fallback[si] = True
            dn, w = donors_for(si, pool, grain[si])
            donor_lin = np.einsum("nk,nkj->nj", w, rgb_lin[dn])
            fill_SH[si] = np.einsum("nk,nkj->nj", w, SH[dn])
            core = np.flatnonzero(m & (alpha > 0.5))
            if len(core) < a.k:
                core = si
            sdn, sw = donors_for(si, core, grain[si])
            base_lin = np.einsum("nk,nkj->nj", sw, rgb_lin[sdn])
            target[si] = np.log(np.maximum(donor_lin, 1e-4)) - np.log(np.maximum(base_lin, 1e-4))
            # smooth along the grain within the surface (fill set only)
            for _ in range(a.gain_smooth):
                gdn, gw = donors_for(si, si, grain[si], floor=0.02)  # includes itself; floor keeps neighbours weighted
                target[si] = np.einsum("nk,nkj->nj", gw, target[gdn])
                fill_SH[si] = np.einsum("nk,nkj->nj", gw, fill_SH[gdn])
            per.append(dict(cluster=int(c), filled=int(len(si)), donors_available=int(len(pool)), fallback=bool(fb)))
            print(f"  cluster {c:3d}: {len(si):6,d} filled (soft), {len(pool):6,d} clean lit donors"
                  f"{'  -> FALLBACK (too few lit on this surface)' if fb else ''}", flush=True)
        al = alpha[:, None]
        if a.mode == "ratio":
            gain = np.exp(np.clip(target, 0, np.log(a.max_gain)))  # never darken; cap the gain
            out_lin = np.clip(rgb_lin * gain ** al, 0, 1)
            new_SH[:, :3] = np.where(shadow[:, None], (linear_to_srgb(out_lin) - 0.5) / SH_C0, SH[:, :3])
            new_SH[:, 3:] = np.where(shadow[:, None], SH[:, 3:] + al * (fill_SH[:, 3:] - SH[:, 3:]), SH[:, 3:])
        else:
            new_SH = np.where(shadow[:, None], SH + al * (fill_SH - SH), SH)
        shadow = alpha > 0.5  # for the brightness report: the core shadow

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
