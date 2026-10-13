#!/usr/bin/env python
"""Detect shadow ON the hero object from its own brightness, explained by a simple sun + sky lighting model.

The chair is one plastic, so brightness differences on it are lighting, not paint. Model per Gaussian:
    L = albedo * (E_sky + E_sun * max(0, n . s) * v)          v = 1 lit, 0 in shadow
with n the signed normal (from shadow_visibility.py) and s the sun. From the data:
    L_sky  = median brightness of clearly sun-averted Gaussians (n . s < --averted)       -> albedo * E_sky
    L_full = median brightness of squarely sun-facing Gaussians (n . s > --facing), divided by their mean n . s
             after removing L_sky                                                         -> albedo * E_sun
Expected lit brightness at a Gaussian: L_lit = L_sky + L_full * max(0, n . s). A Gaussian is SHADOW when its brightness
is closer (in log) to L_sky than to L_lit; Gaussians where L_lit < --min-contrast * L_sky (grazing: lit and shadow
look the same) are UNDECIDED. Brightness = linear luminance of the splat's view-independent colour (f_dc), median-
smoothed over the --knn nearest Gaussians of the same cluster. No photo masks are used; they are only compared.

Usage:
  detect_object_shadow.py --ply exports/IMG_6556/splat.ply --flags exports/IMG_6556_shadow4_v2/shadow_flags.npz \
      --clusters exports/IMG_6556_clusters_chair_r30agree_v3/clusters.npz \
      --object-votes exports/IMG_6556_shadow4_v2/votes_object_tight.npz --out exports/IMG_6556_chairshadow
Writes <out>/chair_shadow.npz (idx, shadow, undecided, brightness, expected), stats.json, chair_shadow.png and
chair_shadow_debug.ply (red shadow, yellow undecided, original colour lit). Trains nothing.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from cluster_surfaces import SH_C0, read_ply, write_ply


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def prf(pred, obs):
    tp, fp, fn = (pred & obs).sum(), (pred & ~obs).sum(), (~pred & obs).sum()
    return dict(n=int(len(obs)), observed=round(float(obs.mean()), 3), predicted=round(float(pred.mean()), 3),
                precision=round(float(tp / max(tp + fp, 1)), 3), recall=round(float(tp / max(tp + fn, 1)), 3),
                iou=round(float(tp / max(tp + fp + fn, 1)), 3), accuracy=round(float((pred == obs).mean()), 3))


def otsu(x, bins=256):
    h, e = np.histogram(x, bins=bins)
    c = 0.5 * (e[1:] + e[:-1])
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return float(c[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def photo_brightness(xyz, opacity, obj, a):
    """Median linear luminance each object Gaussian shows in the photos (front-most only, tight z-buffer)."""
    import cv2
    from fix_splat_shadows import project, qvec2rotmat, read_cameras_bin, read_images_bin
    cams = read_cameras_bin(a.colmap / "cameras.bin")
    frames = read_images_bin(a.colmap / "images.bin")[:: a.every]
    P = xyz[obj]
    occl = xyz[(opacity > 0.5) & (np.linalg.norm(xyz - P.mean(0), axis=1) < a.zbuf_radius)]
    vals = np.full((len(obj), len(frames)), np.nan, np.float32)
    chroma = np.full((len(obj), len(frames), 2), np.nan, np.float16)  # linear r/(r+g+b), g/(r+g+b)
    wl = np.array([0.2126, 0.7152, 0.0722])
    for k, fr in enumerate(frames):
        cam = cams[fr["cam"]]
        w, h = cam["width"], cam["height"]
        R, t = qvec2rotmat(fr["q"]), fr["t"]
        zw, zh = int(w * a.zbuf_scale), int(h * a.zbuf_scale)
        zbuf = np.full(zw * zh, np.inf)
        oc = occl @ R.T + t
        oc = oc[oc[:, 2] > 1e-3]
        uo = project(cam, oc)
        ok = (uo[:, 0] >= 0) & (uo[:, 0] < w) & (uo[:, 1] >= 0) & (uo[:, 1] < h)
        np.minimum.at(zbuf, np.clip((uo[ok, 1] * a.zbuf_scale).astype(np.int64) * zw
                                    + (uo[ok, 0] * a.zbuf_scale).astype(np.int64), 0, zw * zh - 1), oc[ok, 2])
        pc = P @ R.T + t
        front = pc[:, 2] > 1e-3
        up_ = np.full((len(P), 2), -1.0)
        up_[front] = project(cam, pc[front])
        inb = front & (up_[:, 0] >= 0) & (up_[:, 0] < w - 1) & (up_[:, 1] >= 0) & (up_[:, 1] < h - 1)
        cl = np.clip((up_[:, 1] * a.zbuf_scale).astype(np.int64) * zw + (up_[:, 0] * a.zbuf_scale).astype(np.int64),
                     0, zw * zh - 1)
        vis = inb & (pc[:, 2] <= zbuf[cl] + a.vis_tol)
        img = cv2.imread(str(a.images / fr["name"]))[:, :, ::-1].astype(np.float32) / 255
        px = up_[vis].astype(int)
        lin = srgb_to_linear(img[px[:, 1], px[:, 0]])
        vals[vis, k] = lin @ wl
        ssum = np.maximum(lin.sum(1), 1e-6)
        chroma[vis, k, 0], chroma[vis, k, 1] = lin[:, 0] / ssum, lin[:, 1] / ssum
        if (k + 1) % 40 == 0:
            print(f"  photo brightness: {k + 1}/{len(frames)} frames", flush=True)
    n_seen = np.sum(~np.isnan(vals), 1)
    print(f"photo brightness: {len(frames)} frames, object Gaussians seen in median {int(np.median(n_seen))} frames, "
          f"{(n_seen == 0).sum():,} never", flush=True)
    lum = np.nanmedian(np.where(n_seen[:, None] > 0, vals, 0.0), 1)
    ch = np.nanmedian(np.where(n_seen[:, None, None] > 0, chroma.astype(np.float32), 1 / 3), 1)
    return lum, vals, ch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--flags", type=Path, required=True, help="shadow_flags.npz from shadow_visibility.py")
    ap.add_argument("--clusters", type=Path, required=True)
    ap.add_argument("--object-votes", type=Path, help="mask votes for the object (comparison only)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--facing", type=float, default=0.6, help="n.s above this: squarely sun-facing reference")
    ap.add_argument("--averted", type=float, default=-0.3, help="n.s below this: sun-averted (sky-only) reference")
    ap.add_argument("--min-contrast", type=float, default=1.5, help="L_lit / L_sky below this: undecided")
    ap.add_argument("--min-views", type=int, default=5)
    ap.add_argument("--source", choices=["splat", "photos"], default="splat",
                    help="brightness from the splat's f_dc colour, or sampled from the photos (front-most only)")
    ap.add_argument("--classify", choices=["model", "otsu", "adjacent"], default="model",
                    help="model: sun + sky lighting model with normals; otsu: brightness valley only; adjacent: "
                         "compare with nearby parts of the same colour")
    ap.add_argument("--adj-radius", type=float, default=0.25, help="neighbourhood radius (scene units)")
    ap.add_argument("--adj-chroma", type=float, default=0.12,
                    help="max chromaticity difference to the lit neighbours for 'same material' (allows sky-blue shift)")
    ap.add_argument("--adj-pct", type=float, default=90.0, help="percentile of neighbour brightness used as lit ref")
    ap.add_argument("--adj-ratio", type=float, default=0.3, help="shadow when brightness / lit ref is below this")
    ap.add_argument("--smooth", type=int, default=0, help="if > 0: k-nearest same-cluster majority vote on the labels")
    ap.add_argument("--colmap", type=Path)
    ap.add_argument("--images", type=Path)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--zbuf-scale", type=float, default=0.5)
    ap.add_argument("--zbuf-radius", type=float, default=2.0)
    ap.add_argument("--vis-tol", type=float, default=0.012)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    header, data = read_ply(a.ply)
    xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
    f = np.load(a.flags)
    obj, s = f["object_idx"], f["sun"]
    labels = np.load(a.clusters)["labels"]
    lab = labels[obj]
    P = xyz[obj]

    # signed normal for every object Gaussian: trusted ones directly, the rest from their nearest trusted neighbour
    ti, ns = f["trusted_idx"], f["signed_normal"].astype(np.float64)
    seen = f["orient_seen"]
    ok = seen > 0
    _, nb = cKDTree(xyz[ti][ok]).query(P, k=1)
    n = ns[ok][nb]
    pos = {g: j for j, g in enumerate(ti)}
    for k, g in enumerate(obj):
        j = pos.get(g)
        if j is not None and ok[j]:
            n[k] = ns[j]
    cos = n @ s

    # brightness per Gaussian: from the splat colour (f_dc), or from the photos themselves
    wl = np.array([0.2126, 0.7152, 0.0722])
    agree_views = None
    if a.source == "photos":
        opacity = 1 / (1 + np.exp(-data["opacity"].astype(np.float64)))
        lum, per_view, chroma = photo_brightness(xyz, opacity, obj, a)
    else:
        rgb = np.clip(np.stack([data[f"f_dc_{i}"][obj] for i in range(3)], 1) * SH_C0 + 0.5, 0, 1)
        lin = srgb_to_linear(rgb)
        lum = lin @ wl
        chroma = lin[:, :2] / np.maximum(lin.sum(1, keepdims=True), 1e-6)
        per_view = None
    L = lum.copy()
    for c in np.unique(lab):
        m = np.flatnonzero(lab == c)
        if len(m) > a.knn:
            _, nn = cKDTree(P[m]).query(P[m], k=a.knn)
            L[m] = np.median(lum[m][nn], 1)
    L = np.maximum(L, 1e-4)

    # lighting model from the data
    averted, facing = cos < a.averted, cos > a.facing
    L_sky = float(np.median(L[averted]))
    L_sun = float(np.median((L[facing] - L_sky) / cos[facing]))
    expected = L_sky + L_sun * np.maximum(cos, 0)
    if a.classify == "model":
        undecided = expected < a.min_contrast * L_sky
        shadow = (np.log(L) < 0.5 * (np.log(L_sky) + np.log(expected))) & ~undecided
    elif a.classify == "adjacent":
        # compare each Gaussian with ADJACENT parts of the same material: neighbours within --adj-radius whose colour
        # (chromaticity) matches; the lit reference is their bright end (--adj-pct percentile). Shadow when the
        # Gaussian is much darker (ratio < --adj-ratio) than that reference. If the whole neighbourhood is dark
        # (no lit part nearby, e.g. under the seat), the neighbourhood's reference is itself compared with the bright
        # end of the same material over the whole object, so large fully shadowed areas are still caught.
        undecided = np.zeros(len(L), bool)
        tree = cKDTree(P)
        nbrs = tree.query_ball_point(P, a.adj_radius)
        # NOTE shadow under blue sky light is itself bluer than sunlit plastic, so "same colour" must allow that shift:
        # the lit reference is the bright end of ALL neighbours, and its colour (median chromaticity of the bright
        # neighbours) must be within --adj-chroma of the Gaussian's colour, otherwise the darker spot is taken to be a
        # different material (dark paint), not shadow.
        ref = np.empty(len(L))
        ref_ch = np.empty((len(L), 2))
        n_nb = np.zeros(len(L), int)
        for i, nb_i in enumerate(nbrs):
            nb_i = np.asarray(nb_i)
            n_nb[i] = len(nb_i)
            ref[i] = np.percentile(L[nb_i], a.adj_pct)
            bright = nb_i[L[nb_i] >= 0.7 * ref[i]]
            ref_ch[i] = np.median(chroma[bright], 0)
        gref = np.percentile(L, a.adj_pct)
        gch = np.median(chroma[L >= 0.7 * gref], 0)
        same_mat = np.linalg.norm(chroma - ref_ch, axis=1) < a.adj_chroma
        same_mat_g = np.linalg.norm(chroma - gch, axis=1) < a.adj_chroma
        local_ratio = L / np.maximum(ref, 1e-6)
        area_ratio = ref / gref
        shadow = ((local_ratio < a.adj_ratio) & same_mat) | ((area_ratio < a.adj_ratio) & same_mat_g)
        dch = np.linalg.norm(chroma - ref_ch, axis=1)
        print(f"adjacent rule: radius {a.adj_radius}, ratio < {a.adj_ratio}, colour tol {a.adj_chroma}; median "
              f"{int(np.median(n_nb))} neighbours; dark vs neighbours {(local_ratio < a.adj_ratio).mean() * 100:.1f}% "
              f"(colour matches for {(same_mat & (local_ratio < a.adj_ratio)).mean() * 100:.1f}%), whole neighbourhood "
              f"dark {(area_ratio < a.adj_ratio).mean() * 100:.1f}%; colour shift dark-vs-ref median "
              f"{np.median(dch[local_ratio < a.adj_ratio]):.3f}, lit-vs-ref {np.median(dch[local_ratio >= a.adj_ratio]):.3f}",
              flush=True)
    else:  # otsu: one plastic, so brightness alone is bimodal (sun vs sky); split at the valley, no normals needed
        thr = otsu(np.log10(L))
        shadow = np.log10(L) < thr
        undecided = np.zeros(len(L), bool)
        print(f"Otsu threshold on log10 brightness: {thr:.2f} (brightness {10 ** thr:.4f})", flush=True)
        if per_view is not None:
            agree_views = np.nanmean(np.where(np.isnan(per_view), np.nan,
                                              (np.log10(np.maximum(per_view, 1e-5)) < thr) == shadow[:, None]), 1)
            print(f"per-view agreement with the Gaussian's class: median {np.nanmedian(agree_views):.2f}", flush=True)
    if a.smooth:  # spatial majority vote within the same cluster: removes speckle from edge pixels / view bleed
        sm = shadow.copy()
        for c in np.unique(lab):
            m = np.flatnonzero(lab == c)
            if len(m) > a.smooth:
                _, nn = cKDTree(P[m]).query(P[m], k=a.smooth)
                sm[m] = shadow[m][nn].mean(1) > 0.5
        print(f"smoothing (k={a.smooth} same-cluster majority) changed {(sm != shadow).mean() * 100:.1f}% of labels")
        shadow = sm
    print(f"lighting model: sky-only brightness {L_sky:.4f} ({averted.sum():,} averted Gaussians), sun term {L_sun:.4f} "
          f"({facing.sum():,} facing); full sun / sky = {(L_sky + L_sun) / L_sky:.1f}x", flush=True)
    print(f"object: {len(obj):,} Gaussians; shadow {shadow.mean() * 100:.1f}%, undecided (grazing) "
          f"{undecided.mean() * 100:.1f}%, lit {(~shadow & ~undecided).mean() * 100:.1f}%", flush=True)

    geo = f["attached"] | f["cast_object"]
    stats = dict(L_sky=L_sky, L_sun=L_sun, sun_to_sky=round((L_sky + L_sun) / L_sky, 2),
                 shadow_share=round(float(shadow.mean()), 4), undecided_share=round(float(undecided.mean()), 4),
                 params={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()})
    d = ~undecided
    stats["vs_geometry"] = prf(shadow[d], geo[d])
    print(f"brightness vs geometry (decided only): {stats['vs_geometry']}")
    if a.object_votes:
        v = np.load(a.object_votes)
        pv = {g: j for j, g in enumerate(v["rec_idx"])}
        jj = np.array([pv.get(g, -1) for g in obj])
        okv = jj >= 0
        okv[okv] &= v["seen"][jj[okv]] >= a.min_views
        obs = np.zeros(len(obj), bool)
        obs[okv] = v["shadow"][jj[okv]] / v["seen"][jj[okv]] >= 0.5
        stats["vs_masks"] = prf(shadow[okv & d], obs[okv & d])
        stats["geometry_vs_masks"] = prf(geo[okv & d], obs[okv & d])
        print(f"brightness vs masks (decided only): {stats['vs_masks']}")
        print(f"geometry   vs masks (same set):     {stats['geometry_vs_masks']}")
    per = []
    for c in np.unique(lab):
        m = lab == c
        per.append(dict(cluster=int(c), n=int(m.sum()), mean_cos=round(float(cos[m].mean()), 2),
                        shadow=round(float(shadow[m].mean()), 3), undecided=round(float(undecided[m].mean()), 3),
                        geometry=round(float(geo[m].mean()), 3), median_brightness=round(float(np.median(L[m])), 4)))
        print(f"  cluster {c:3d}: n {m.sum():6,d}  mean n.s {cos[m].mean():+.2f}  brightness {np.median(L[m]):.4f}  "
              f"shadow {shadow[m].mean():.2f}  undecided {undecided[m].mean():.2f}  geometry {geo[m].mean():.2f}")
    stats["per_cluster"] = per
    (a.out / "stats.json").write_text(json.dumps(stats, indent=1))
    np.savez_compressed(a.out / "chair_shadow.npz", idx=obj, shadow=shadow, undecided=undecided, brightness=L,
                        expected=expected, cos=cos, geometry=geo)

    sub = data[obj].copy()
    col = np.clip(np.stack([sub[f"f_dc_{i}"] for i in range(3)], 1) * SH_C0 + 0.5, 0, 1)
    col[shadow] = [0.9, 0.1, 0.1]
    col[undecided] = [0.95, 0.85, 0.1]
    for i in range(3):
        sub[f"f_dc_{i}"] = (col[:, i] - 0.5) / SH_C0
    for nme in sub.dtype.names:
        if nme.startswith("f_rest_"):
            sub[nme] = 0
    write_ply(a.out / "chair_shadow_debug.ply", header, sub)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    axs[0].scatter(cos, L, s=0.5, c=np.where(shadow, "#c92a2a", np.where(undecided, "#f59f00", "#1971c2")))
    cc = np.linspace(-1, 1, 200)
    axs[0].plot(cc, L_sky + L_sun * np.maximum(cc, 0), "k-", label="expected if lit")
    axs[0].axhline(L_sky, color="k", ls="--", label="sky only (shadow)")
    axs[0].set_yscale("log"); axs[0].set_xlabel("signed normal . sun"); axs[0].set_ylabel("brightness (linear)")
    axs[0].legend(); axs[0].set_title("red shadow, orange undecided, blue lit")
    axs[1].hist(np.log10(L[~undecided]), bins=80, color="#495057")
    axs[1].set_xlabel("log10 brightness (decided)"); axs[1].set_title("brightness histogram")
    fig.tight_layout(); fig.savefig(a.out / "chair_shadow.png", dpi=100)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
