#!/usr/bin/env python
"""Geometry-driven object fill: lift only the sun's missing share, spot by spot, using 2DGS surface directions.

Lighting model per blob (linear brightness):  L = albedo * (sky + sun * max(0, n . s) * v),  sky = 1, v in [0, 1]
- sun / sky is MEASURED on the concrete next to the object, not fitted on the painted object: ground blobs that the
  photo masks vote as shadow vs lit (fit_sun votes), ratio of median brightness -> sun = (ratio - 1) / (up . s).
- n: surface direction borrowed from the 2DGS surfels (borrow_normals.py); blobs without a confident direction take
  the mean of their confident neighbours'.
- albedo: the user's adjacent brightness-and-colour rule (credit: the user), with the radius in object units: among
  neighbours within --radius x object height of similar colour (chromaticity distance < --chroma), each one's
  brightness divided by ITS expected lit irradiance (sky + sun * max(0, n . s)); the --pct percentile of that is the
  local paint brightness. Grows to 2 x radius when fewer than --min-nb neighbours qualify.
- observed irradiance e = L / albedo; expected unshadowed t = sky + sun * max(0, n . s).
Rules from the user:
- no relighting on the sun side: a blob whose e is within --margin of t is left exactly as it is, and so are the
  sun-averted blobs (t = sky), whose darkness is ordinary shading, not shadow;
- shadow strength varies (soft edges, partial and occluded shadows): each blob gets its own gain t / e, clipped to
  [1, t / --e-min], smoothed in log space over its --smooth-k nearest object neighbours that face the same way.
Cross-check with the sunlit parts (user): after smoothing, a filled blob may not end up brighter, per unit of expected
light, than the median of clearly sunlit blobs (n . s > --sunlit-cos, not shadowed) of the same paint within
2 x radius; gains above that are trimmed, and the filled / sunlit match is reported next to the sunlit blobs' own spread.
Only colours of object blobs change (DC in linear RGB; view-dependent SH scaled with the gain, capped at 3).

Usage: fill_geometry.py --ply exports/IMG_6705_shadowfix_tex_u/splat_shadowfix.ply --orig exports/IMG_6705/splat.ply
           --borrow exports/IMG_6705_2dgs_borrow/borrowed.npz --sun-from exports/IMG_6705_shadow4/shadow_flags.npz
           --object exports/IMG_6705_object/object.json --plane-from exports/IMG_6705_shadowfix_o/stats.json
           --votes exports/IMG_6705_sun/votes.npz --out exports/IMG_6705_fill_geom
Writes <out>/splat_fill.ply (same blobs as --ply), object_before.ply / object_after.ply, gain_debug.ply (grey =
unchanged, yellow -> red = gain), stats.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from fill_shadows import linear_to_srgb, srgb_to_linear
from fix_splat_shadows import read_ply, write_ply

SH_C0 = 0.28209479177387814
LUM = np.array([0.2126, 0.7152, 0.0722])


def colours(data):
    names = [f"f_dc_{i}" for i in range(3)] + sorted([n for n in data.dtype.names if n.startswith("f_rest_")],
                                                     key=lambda s: int(s.split("_")[-1]))
    SH = np.stack([data[n] for n in names], 1).astype(np.float64)
    return names, SH, srgb_to_linear(np.clip(SH[:, :3] * SH_C0 + 0.5, 0, 1))


def chroma(rgb):
    return rgb / np.maximum(rgb.sum(1, keepdims=True), 1e-6)


def points(path, header, sub, rgb_lin):
    d = sub.copy()
    names, SH, _ = colours(d)
    d["f_dc_0"], d["f_dc_1"], d["f_dc_2"] = ((linear_to_srgb(rgb_lin) - 0.5) / SH_C0).T.astype(np.float32)
    h = [l for l in header if not l.startswith("element vertex")]
    h.insert(next(i for i, l in enumerate(header) if l.startswith("element vertex")), f"element vertex {len(d)}")
    write_ply(path, h, d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", type=Path, required=True, help="splat to fill (e.g. after the ground fill)")
    ap.add_argument("--orig", type=Path, required=True, help="original splat (ground brightness for sun / sky)")
    ap.add_argument("--borrow", type=Path, required=True, help="borrowed.npz from borrow_normals.py")
    ap.add_argument("--sun-from", type=Path, required=True, help="shadow_flags.npz (sun) from shadow_visibility.py")
    ap.add_argument("--object", type=Path, required=True)
    ap.add_argument("--plane-from", type=Path, required=True)
    ap.add_argument("--votes", type=Path, required=True, help="ground mask votes (fit_sun votes.npz)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ring", type=float, nargs=2, default=[0.6, 2.0], help="ground ring, x object height from the footprint")
    ap.add_argument("--radius", type=float, default=0.06, help="albedo neighbourhood, x object height")
    ap.add_argument("--chroma", type=float, default=0.12, help="max chromaticity distance for 'same paint'")
    ap.add_argument("--pct", type=float, default=80.0)
    ap.add_argument("--min-nb", type=int, default=8)
    ap.add_argument("--margin", type=float, default=0.15, help="leave alone when e >= (1 - margin) * t")
    ap.add_argument("--e-min", type=float, default=0.7, help="floor on e (deep crevices get some sky)")
    ap.add_argument("--graze", type=float, default=0.05, help="n . s below this counts as sun-averted (not filled)")
    ap.add_argument("--sunlit-cos", type=float, default=0.3, help="cross-check reference: unshadowed blobs with n.s above this")
    ap.add_argument("--smooth-k", type=int, default=8)
    ap.add_argument("--smooth-iters", type=int, default=2)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    header, data = read_ply(a.ply)
    _, orig = read_ply(a.orig)
    if len(orig) != len(data):
        raise SystemExit("ERROR: --ply and --orig must hold the same blobs")
    xyz = np.c_[data["x"], data["y"], data["z"]].astype(float)
    obj = json.loads(a.object.read_text())
    H = obj["height"]
    gp = json.loads(a.plane_from.read_text())["ground_plane"]
    up = np.array(gp["normal"], float)
    up /= np.linalg.norm(up)
    s = np.load(a.sun_from)["sun"].astype(float)
    s /= np.linalg.norm(s)

    # sun / sky from the concrete beside the object (original splat, before the ground fill)
    v = np.load(a.votes)
    _, _, o_lin = colours(orig)
    foot = np.array(obj["footprint_centre"])
    r = np.linalg.norm(xyz[v["rec_idx"]] - foot, axis=1) / H
    ok = (v["seen"] >= 5) & (r > a.ring[0]) & (r < a.ring[1])
    share = v["shadow"] / np.maximum(v["seen"], 1)
    lum = o_lin[v["rec_idx"]] @ LUM
    sh_l, lit_l = np.median(lum[ok & (share > 0.7)]), np.median(lum[ok & (share < 0.1)])
    sun = (lit_l / sh_l - 1) / (up @ s)
    print(f"ground ring: {(ok & (share > 0.7)).sum():,} shadow / {(ok & (share < 0.1)).sum():,} lit blobs; "
          f"brightness {sh_l:.4f} vs {lit_l:.4f} (ratio {lit_l / sh_l:.2f}); up.s {up @ s:.3f} -> sun/sky {sun:.2f}")

    # object blobs and their directions
    b = np.load(a.borrow)
    idx = b["index"][b["on_object"]]
    n = b["normal"][b["on_object"]].astype(float)
    conf = b["confident"][b["on_object"]]
    p = xyz[idx]
    tree = cKDTree(p)
    _, nb = tree.query(p, k=a.smooth_k + 1)
    ci = np.flatnonzero(conf)
    _, cnb = cKDTree(p[ci]).query(p[~conf], k=a.smooth_k)
    m = n[ci][cnb]
    m = (m * np.sign(np.einsum("ikj,ij->ik", m, m[:, 0]))[..., None]).sum(1)
    n[~conf] = m / np.linalg.norm(m, axis=1, keepdims=True)
    ns = n @ s
    t = 1 + sun * np.maximum(ns, 0)

    names, SH, rgb = colours(data)
    rgb_o = rgb[idx]
    L = rgb_o @ LUM
    ch = chroma(rgb_o)
    alb_lit = L / t  # each blob's paint brightness IF it were lit
    albedo = np.zeros(len(p))
    grown = 0
    for i, nbrs in enumerate(tree.query_ball_point(p, a.radius * H)):
        nbrs = np.asarray(nbrs)
        same = nbrs[np.abs(ch[nbrs] - ch[i]).sum(1) < a.chroma]
        if len(same) < a.min_nb:
            nbrs = np.asarray(tree.query_ball_point(p[i], 2 * a.radius * H))
            same = nbrs[np.abs(ch[nbrs] - ch[i]).sum(1) < a.chroma]
            grown += 1
        albedo[i] = np.percentile(alb_lit[same], a.pct) if len(same) else alb_lit[i]
    e = L / np.maximum(albedo, 1e-6)
    shadowed = (ns > a.graze) & (e < (1 - a.margin) * t)  # sun-averted blobs: ordinary shading, left alone
    logg = np.where(shadowed, np.log(t / np.maximum(e, a.e_min)), 0.0)
    logg = np.clip(logg, 0, None)
    for _ in range(a.smooth_iters):  # smooth over same-facing neighbours; lit blobs stay at 0
        same_side = np.einsum("ij,ikj->ik", n, n[nb]) > 0.7
        sm = (logg[nb] * same_side).sum(1) / np.maximum(same_side.sum(1), 1)
        logg = np.where(shadowed, sm, 0.0)
    gain = np.exp(logg)

    # cross-check with the sunlit parts (user): the filled brightness may not exceed what clearly sunlit blobs of the
    # same paint nearby show, per unit of expected light; this also trims speckle from the smoothing step
    sunlit = ~shadowed & (ns > a.sunlit_cos)
    sl = np.flatnonzero(sunlit)
    ref = np.full(len(p), np.nan)
    sl_tree = cKDTree(p[sl])
    for i in np.flatnonzero(shadowed):
        cand = sl[np.asarray(sl_tree.query_ball_point(p[i], 2 * a.radius * H), int)]
        cand = cand[np.abs(ch[cand] - ch[i]).sum(1) < a.chroma]
        if len(cand) >= 3:
            ref[i] = np.median(alb_lit[cand])
    has = shadowed & np.isfinite(ref)
    cap = np.where(has, ref * t / np.maximum(L, 1e-6), np.inf)
    trimmed = has & (gain > cap)
    gain = np.where(trimmed, np.maximum(cap, 1), gain)
    after_vs_sunlit = (L * gain / t)[has] / ref[has]
    lit_vs_sunlit = None
    if sunlit.sum() > 10:  # same measure for the untouched sunlit blobs: their own spread
        _, k = sl_tree.query(p[sl], k=min(9, len(sl)))
        lit_vs_sunlit = alb_lit[sl] / np.median(alb_lit[sl][k[:, 1:]], axis=1)
    print(f"cross-check vs sunlit same-paint neighbours: {has.sum():,} of {shadowed.sum():,} filled blobs have a "
          f"sunlit reference; trimmed {trimmed.sum():,}; filled / sunlit median {np.median(after_vs_sunlit):.2f} "
          f"(p10 {np.percentile(after_vs_sunlit, 10):.2f}, p90 {np.percentile(after_vs_sunlit, 90):.2f})"
          + (f"; sunlit / sunlit neighbours p10-p90 {np.percentile(lit_vs_sunlit, 10):.2f}-"
             f"{np.percentile(lit_vs_sunlit, 90):.2f}" if lit_vs_sunlit is not None else ""))

    new = data.copy()
    out_lin = np.clip(rgb_o * gain[:, None], 0, 1)
    dc = (linear_to_srgb(out_lin) - 0.5) / SH_C0
    for j, nm in enumerate(names):
        col = SH[idx, j] if j >= 3 else dc[:, j]
        new[nm][idx] = (col * np.minimum(gain, 3) if j >= 3 else col).astype(np.float32)
    write_ply(a.out / "splat_fill.ply", header, new)
    points(a.out / "object_before.ply", header, data[idx], rgb_o)
    points(a.out / "object_after.ply", header, new[idx], out_lin)
    dbg = np.full((len(idx), 3), 0.25)
    gn = np.clip((gain - 1) / 3, 0, 1)
    dbg[shadowed] = np.c_[np.ones(shadowed.sum()), 1 - gn[shadowed], np.zeros(shadowed.sum())]
    points(a.out / "gain_debug.ply", header, data[idx], srgb_to_linear(dbg))

    st = {"sun_over_sky": round(float(sun), 3), "ground_ratio": round(float(lit_l / sh_l), 3),
          "object_blobs": int(len(idx)), "confident_share": float(conf.mean()), "grown_neighbourhood": int(grown),
          "sun_facing_share": float((ns > 0).mean()), "shadowed_share": float(shadowed.mean()),
          "shadowed_sun_facing_share": float((shadowed & (ns > 0)).mean()),
          "gain_median_shadowed": float(np.median(gain[shadowed])) if shadowed.any() else 1.0,
          "gain_p90_shadowed": float(np.percentile(gain[shadowed], 90)) if shadowed.any() else 1.0,
          "raised_over_1.5x_share": float((gain > 1.5).mean()),
          "unchanged_share": float((gain == 1).mean()),
          "e_over_t_median_shadowed": float(np.median((e / t)[shadowed])) if shadowed.any() else 1.0,
          "crosscheck_with_ref": int(has.sum()), "crosscheck_trimmed": int(trimmed.sum()),
          "filled_over_sunlit_p10_50_90": [float(np.percentile(after_vs_sunlit, q)) for q in (10, 50, 90)],
          "sunlit_over_sunlit_p10_50_90": [float(np.percentile(lit_vs_sunlit, q)) for q in (10, 50, 90)]
          if lit_vs_sunlit is not None else None,
          "args": {k: str(v) for k, v in vars(a).items()}}
    (a.out / "stats.json").write_text(json.dumps(st, indent=1))
    print(f"object: {len(idx):,} blobs, {st['sun_facing_share']:.1%} face the sun; shadowed {st['shadowed_share']:.1%} "
          f"(on the sun side {st['shadowed_sun_facing_share']:.1%}); unchanged {st['unchanged_share']:.1%}; gain in "
          f"shadow median {st['gain_median_shadowed']:.2f} (p90 {st['gain_p90_shadowed']:.2f}); raised > 1.5x "
          f"{st['raised_over_1.5x_share']:.1%}; neighbourhood grown for {grown:,} -> {a.out / 'splat_fill.ply'}")


if __name__ == "__main__":
    main()
