#!/usr/bin/env python
"""Fill only the object blobs the sun map says are in shadow, from lit neighbours of the same paint.

Labels: sun_visibility.npz from sun_shadow_map.py (visible < 0.5 = shadow). For each shadow blob (the user's adjacent
brightness-and-colour rule, credit: the user):
- lit pool: lit object blobs within --radius x object height (grown x2, x4 when fewer than --min-nb match), whose
  chromaticity is within --chroma of the shadow blob's after removing the blue tint of shade (shade-to-sun colour
  shift measured on the concrete, as in relight_sh.py);
- shadow pool: other shadow blobs of the same paint in the same neighbourhood;
- new colour = own colour x median(lit pool) / median(shadow pool), per channel, in linear RGB. The ratio of two local
  medians sets the strength spot by spot (soft edges, partial shadow) and removes the tint, while the blob keeps its
  own light-and-dark texture; a single blob's noisy colour is never multiplied on its own.
- the ratio is smoothed over the --smooth-k nearest shadow blobs, never darkens, is capped at --max-gain, and the
  result may not exceed the lit pool's median brightness (the user's sunlit cross-check).
Lit blobs are not touched. Writes <out>/splat_fill.ply (--ply with the object's shadow blobs recoloured, matched by
position through --orig) and stats.json.

Usage: fill_labels.py --ply exports/IMG_6705_clone_y_base/splat_clone.ply --orig exports/IMG_6705/splat.ply
           --labels exports/IMG_6705_sunmap3/sun_visibility.npz --object exports/IMG_6705_object/object.json
           --plane-from exports/IMG_6705_shadowfix_o/stats.json --votes exports/IMG_6705_sun/votes.npz --out exports/IMG_6705_fill_labels
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from fill_geometry import LUM, SH_C0, chroma, colours
from fill_shadows import linear_to_srgb
from fix_splat_shadows import read_ply, write_ply


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("ply", "orig", "labels", "object", "plane-from", "votes", "out"):
        ap.add_argument(f"--{k}", type=Path, required=True)
    ap.add_argument("--radius", type=float, default=0.06)
    ap.add_argument("--chroma", type=float, default=0.12)
    ap.add_argument("--min-nb", type=int, default=8)
    ap.add_argument("--smooth-k", type=int, default=12)
    ap.add_argument("--max-gain", type=float, default=8.0)
    ap.add_argument("--ring", type=float, nargs=2, default=[0.6, 2.0])
    ap.add_argument("--near-mm", type=float, default=0.0,
                    help="> 0: blobs this far or more behind the sun-facing surface (but under the label margin) become "
                         "shadow when also dark; only blobs closer than this are lit references")
    ap.add_argument("--eps-mm", type=float, default=10.0, help="the label margin used by sun_shadow_map (for the report)")
    ap.add_argument("--propagate", type=float, default=0.0,
                    help="> 0: unmatched shadow blobs take the correction of filled shadow blobs within this x height")
    ap.add_argument("--local-strength", action="store_true",
                    help="brightness per spot from its local darkness vs the lit reference (removes hard cores)")
    ap.add_argument("--local-k", type=int, default=6)
    ap.add_argument("--brightness-only", action="store_true", help="lift brightness only, keep each spot's own colour")
    ap.add_argument("--tint-colour", action="store_true",
                    help="with --local-tint: colour from the area's shade tint only, brightness from the per-spot gain")
    ap.add_argument("--local-tint", type=float, default=0.0,
                    help="> 0: measure the shade tint around each shadow blob within this x height (instead of the concrete's)")
    ap.add_argument("--dark", type=float, default=0.6, help="'dark' = below this x the clearly lit same-paint median")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    header, data = read_ply(a.ply)
    _, orig = read_ply(a.orig)
    oxyz = np.c_[orig["x"], orig["y"], orig["z"]].astype(float)
    obj = json.loads(a.object.read_text())
    H = obj["height"]
    lab = np.load(a.labels)
    idx = lab["index"]
    shadow = lab["visible"] < 0.5
    p = oxyz[idx]

    # shade-to-sun colour shift from the concrete (lit / shaded chromaticity)
    _, _, o_lin = colours(orig)
    vt = np.load(a.votes)
    r = np.linalg.norm(oxyz[vt["rec_idx"]] - np.array(obj["footprint_centre"]), axis=1) / H
    ok = (vt["seen"] >= 5) & (r > a.ring[0]) & (r < a.ring[1])
    share = vt["shadow"] / np.maximum(vt["seen"], 1)
    sh_c = np.median(o_lin[vt["rec_idx"][ok & (share > 0.7)]], 0)
    lit_c = np.median(o_lin[vt["rec_idx"][ok & (share < 0.1)]], 0)
    wb = (lit_c / lit_c.sum()) / (sh_c / sh_c.sum())

    rgb = o_lin[idx]
    tree = cKDTree(p)
    if a.near_mm > 0:
        # geometry + brightness: blobs only a few mm behind the sun-facing surface (--near-mm .. label margin) are
        # ambiguous (splat thickness or a close caster such as the brim edge); call them shadow when they are also
        # much darker than clearly lit (< --near-mm) blobs of the same paint nearby. Only clearly lit blobs are references.
        gap_mm = lab["gap"] / (obj["units_per_m"] / 1000)
        clear = gap_mm < a.near_mm
        maybe = ~shadow & ~clear
        cf = chroma(rgb * wb)
        cl = np.flatnonzero(clear)
        added = 0
        ct = cKDTree(p[cl])
        for i in np.flatnonzero(maybe):
            nb = cl[np.asarray(ct.query_ball_point(p[i], 2 * a.radius * H), int)]
            nb = nb[np.abs(chroma(rgb[nb]) - cf[i]).sum(1) < a.chroma]
            if len(nb) >= 3 and rgb[i] @ LUM < a.dark * np.median(rgb[nb] @ LUM):
                shadow[i] = True
                added += 1
        print(f"ambiguous {maybe.sum():,} blobs {a.near_mm}-{a.eps_mm} mm behind the surface: {added:,} dark enough -> shadow; "
              f"clearly lit references {clear.sum():,}")
    else:
        clear = ~shadow
    ch = chroma(rgb)
    if a.local_tint > 0:
        # the shade tint is not the concrete's everywhere (bear 2's hat: shade is warmer, not bluer, from bounce off the
        # bear and ground): measure it around each shadow blob as median lit chroma / median shadow chroma
        tint = np.ones((len(p), 3))
        lt = cKDTree(p[clear & ~shadow])
        lit_ix = np.flatnonzero(clear & ~shadow)
        st_ = cKDTree(p[shadow])
        sh_ix = np.flatnonzero(shadow)
        for i in sh_ix:
            ln = lit_ix[np.asarray(lt.query_ball_point(p[i], a.local_tint * H), int)]
            sn = sh_ix[np.asarray(st_.query_ball_point(p[i], a.local_tint * H), int)]
            if len(ln) >= 10 and len(sn) >= 10:
                tint[i] = np.median(ch[ln], 0) / np.maximum(np.median(ch[sn], 0), 1e-4)
            else:
                tint[i] = wb / wb.mean()
        ch_fix = chroma(np.where(shadow[:, None], rgb * tint, rgb))
    else:
        ch_fix = chroma(np.where(shadow[:, None], rgb * wb, rgb))  # shade blobs compared as if their tint were removed
    lit_i = np.flatnonzero(~shadow)
    sh_i = np.flatnonzero(shadow)
    ratio = np.ones((len(p), 3))
    found = np.zeros(len(p), bool)
    ref_l = np.full(len(p), np.inf)
    for i in sh_i:
        for grow in (1, 2, 4):
            nb = np.asarray(tree.query_ball_point(p[i], grow * a.radius * H), int)
            same = nb[np.abs(ch_fix[nb] - ch_fix[i]).sum(1) < a.chroma]
            lp, sp = same[clear[same] & ~shadow[same]], same[shadow[same]]
            if len(lp) >= a.min_nb and len(sp) >= 3:
                break
        if len(lp) >= 3 and len(sp) >= 1:
            lm, sm = np.median(rgb[lp], 0), np.median(rgb[sp], 0)
            ratio[i] = lm / np.maximum(sm, 1e-5)
            ref_l[i] = lm @ LUM
            found[i] = True
    # smooth the log ratio over nearby shadow blobs that found a reference
    fi = np.flatnonzero(found)
    lr = np.log(np.clip(ratio, 1, a.max_gain))
    if len(fi) > a.smooth_k:
        _, nn = cKDTree(p[fi]).query(p[fi], k=a.smooth_k)
        lr[fi] = lr[fi][nn].mean(1)
    filled_from_nb = np.zeros(len(p), bool)
    if a.propagate and len(fi) > a.smooth_k:
        # shadow blobs with no lit paint of their own colour nearby: the correction is a property of the light, not
        # of the paint, so take the (log) ratio of the nearest shadow blobs that did find a reference
        miss = np.flatnonzero(shadow & ~found)
        dmiss, nn = cKDTree(p[fi]).query(p[miss], k=a.smooth_k)
        near = dmiss[:, 0] < a.propagate * H
        lr[miss[near]] = lr[fi][nn[near]].mean(1)
        filled_from_nb[miss[near]] = True
        found |= filled_from_nb
        print(f"propagated the correction to {near.sum():,} of {len(miss):,} unmatched shadow blobs")
    gain = np.exp(lr)
    if a.local_strength:
        # one ratio per neighbourhood lifts a soft halo and a hard core by the same factor, so the core stays dark and
        # its edge stays hard. Brightness instead comes per spot: lit reference / LOCAL darkness (median of the spot's
        # --local-k nearest shadow blobs, so fine texture survives); the pool ratio only sets the colour balance.
        si = np.flatnonzero(shadow & found & np.isfinite(ref_l))
        _, ln = cKDTree(p[si]).query(p[si], k=min(a.local_k, len(si)))
        local = np.median((rgb @ LUM)[si][ln], 1)
        bal = gain[si] / np.maximum(gain[si] @ LUM, 1e-6)[:, None]  # colour balance of the pool ratio, unit luminance
        g_new = np.clip(ref_l[si] / np.maximum(local, 1e-5), 1, a.max_gain)
        gain[si] = bal * g_new[:, None]
    if a.brightness_only:  # keep every spot's own colour; lift brightness only (no new hues can appear)
        fs = np.flatnonzero(shadow & found)
        gain[fs] = np.clip(gain[fs] @ LUM, 1, a.max_gain)[:, None]
    elif a.tint_colour and a.local_tint > 0:
        # colour from the area's smooth shade tint only (the same shift for every spot in an area), applied in
        # proportion to the light added; per-channel neighbourhood ratios on large lifts made purple / orange patches
        fs = np.flatnonzero(shadow & found)
        g_l = np.clip(gain[fs] @ LUM, 1, a.max_gain)
        tn = tint[fs] / np.maximum(tint[fs] @ LUM, 1e-6)[:, None]
        added = 1 - 1 / g_l
        gain[fs] = g_l[:, None] * (1 + added[:, None] * (tn - 1))
        print(f"tint colour: {len(fs):,} spots, median shade tint {np.median(tn, 0).round(3)}")
        # spots that only took a propagated correction keep it
        print(f"local strength: {len(si):,} spots, brightness gain median {np.median(g_new):.2f} (p90 {np.percentile(g_new, 90):.2f})")
    new = np.clip(rgb * gain, 0, 1)
    lim = np.maximum(ref_l, rgb @ LUM)  # the cap trims a lift, it never darkens a blob below its own brightness
    over = (new @ LUM) > lim
    new[over] *= (lim[over] / np.maximum(new[over] @ LUM, 1e-6))[:, None]
    new[~found] = rgb[~found]

    xyz = np.c_[data["x"], data["y"], data["z"]].astype(float)
    dd, j = cKDTree(xyz).query(p)
    hit = (dd < 1e-6) & found
    names, SH, _ = colours(data)
    out = data.copy()
    dc = (linear_to_srgb(new) - 0.5) / SH_C0
    g_l = (new @ LUM) / np.maximum(rgb @ LUM, 1e-6)
    for c, nm in enumerate(names):
        col = dc[:, c] if c < 3 else SH[j, c] * np.minimum(g_l, 3)
        out[nm][j[hit]] = col[hit].astype(np.float32)
    write_ply(a.out / "splat_fill.ply", header, out)
    st = {"object_blobs": int(len(p)), "shadow_blobs": int(shadow.sum()), "filled": int(hit.sum()),
          "no_reference": int((shadow & ~found).sum()), "shade_to_sun_shift": wb.round(3).tolist(),
          "gain_lum_median": float(np.median(g_l[hit])) if hit.any() else 1.0,
          "gain_lum_p90": float(np.percentile(g_l[hit], 90)) if hit.any() else 1.0,
          "capped_by_lit_level": int((over & found).sum()), "args": {k: str(v) for k, v in vars(a).items()}}
    (a.out / "stats.json").write_text(json.dumps(st, indent=1))
    print(f"{st['shadow_blobs']:,} shadow blobs of {len(p):,}: filled {st['filled']:,}, no lit reference "
          f"{st['no_reference']:,}; brightness gain median {st['gain_lum_median']:.2f} (p90 {st['gain_lum_p90']:.2f}); "
          f"capped at the lit level {st['capped_by_lit_level']:,}; shade-to-sun shift {wb.round(3)} -> {a.out}")


if __name__ == "__main__":
    main()
