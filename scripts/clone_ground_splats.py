#!/usr/bin/env python
"""Replace the splats of a ground shadow with copies of nearby sunlit ground splats (geometry clone stamp).

fix_splat_shadows.py --fill texture only recolours the splats that are already there. Under a hard shadow those were
trained on an almost black, featureless patch, so they are fewer and larger than on sunlit concrete, and no colour can
give them the concrete's fine grain. This script instead:

1. Marks the shadow patch on the ground plane as a grid of cells whose ground splats were mostly seen in shadow
   (shadow_frac from fix_splat_shadows.py), keeping only patches that the texture fill touched and leaving out
   cells under the object.
2. Gives each cell a source: the cell at one of a few whole-cell shifts that is sunlit ground and not a crack or
   stain (--crack: cells much darker than the sunlit ground around them). The first shift is the shortest that covers
   nearly as much as the best one; up to --max-shifts more are added greedily for the cells still empty.
3. Deletes the ground splats in those cells and adds copies of every ground splat from the source cells, moved back
   by the shift along the plane: same size, shape, rotation, opacity and colour, so the patch gets the same splat
   density and grain as real sunlit concrete. Each copy's colour is scaled by the mean lit ground colour around its
   new spot over that around its source (--match), as in --tex-match.

Splats outside the kept cells, including the blended rim from the texture fill, come from --base unchanged.

Usage:
  clone_ground_splats.py --ply exports/IMG_6704/splat.ply --base exports/IMG_6704_shadowfix_tex_u/splat_shadowfix.ply \
      --fix-dir exports/IMG_6704_shadowfix_tex_u --out exports/IMG_6704_clone
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from fix_splat_shadows import SH_C0, linear_to_srgb, read_ply, srgb_to_linear, write_ply


def save_map(path, cell_lum, shadow, hole, fill, group, crack, obj, cands, assign, xs, ys, px=4):
    """Top-down map of the patch: sunlit ground in grey, cracks red, object purple, filled cells coloured by shift
    with their source cells outlined in the same colour, cells left to the texture fill in black."""
    import cv2
    pal = np.array([(66, 135, 245), (46, 184, 92), (245, 166, 35), (155, 89, 182), (26, 188, 156), (230, 126, 34)], float)
    src = np.full(hole.shape, -1)
    for c_i in np.unique(assign[assign >= 0]):
        sel = assign == c_i
        src[xs[sel] + cands[c_i][0], ys[sel] + cands[c_i][1]] = c_i
    lab, _ = ndimage.label(hole, np.ones((3, 3)))
    main = lab == np.argmax(np.bincount(lab[lab > 0])) if hole.any() else hole  # the object's own shadow
    main_src = np.zeros_like(main)
    for c_i in np.unique(assign[assign >= 0]):
        sel = (assign == c_i) & main[xs, ys]
        main_src[xs[sel] + cands[c_i][0], ys[sel] + cands[c_i][1]] = True
    r = np.nonzero(ndimage.binary_dilation(main | main_src, iterations=8))
    x0, x1, y0, y1 = r[0].min(), r[0].max() + 1, r[1].min(), r[1].max() + 1
    sl = (slice(x0, x1), slice(y0, y1))
    v = cell_lum[sl]
    top = np.percentile(v[v > 0], 99) if (v > 0).any() else 1
    img = np.repeat((np.clip(v / top, 0, 1) ** (1 / 2.2) * 200 + 30)[..., None], 3, 2)
    img[(shadow & ~hole)[sl]] = (60, 60, 90)
    img[crack[sl]] = (220, 40, 40)
    img[obj[sl]] = (120, 60, 140)
    order = np.unique(assign[assign >= 0])
    for k, c_i in enumerate(order):
        g_sel = (group == c_i)[sl]
        img[g_sel] = 0.35 * img[g_sel] + 0.65 * pal[k % len(pal)]
        s_sel = (src == c_i)[sl]
        edge = s_sel & ~ndimage.binary_erosion(s_sel)
        img[edge] = pal[k % len(pal)]
    img[(hole & ~fill)[sl]] = (0, 0, 0)
    img = np.kron(img, np.ones((px, px, 1)))
    cv2.imwrite(str(path), np.ascontiguousarray(np.transpose(img, (1, 0, 2))[::-1, :, ::-1]).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", required=True, type=Path, help="original splat (same order as --base)")
    ap.add_argument("--base", required=True, type=Path, help="texture-filled splat to start from")
    ap.add_argument("--fix-dir", required=True, type=Path,
                    help="fix_splat_shadows.py --fill texture output (stats.json with plane and shift, shadow_frac.npy)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cell", type=float, default=0.05, help="grid cell on the ground plane (scene units)")
    ap.add_argument("--hole-frac", type=float, default=0.3, help="cell is shadow if its ground splats' mean shadow_frac is above")
    ap.add_argument("--close", type=int, default=2, help="close gaps in the patch by this many cells")
    ap.add_argument("--object-height", type=float, default=0.15,
                    help="opaque splats higher than this above the plane are the object; their cells are left alone")
    ap.add_argument("--object-margin", type=int, default=1, help="grow the object footprint by this many cells")
    ap.add_argument("--source-lit", type=float, default=0.1, help="source cell: ground splats' mean shadow_frac below this")
    ap.add_argument("--opaque", type=float, default=0.5)
    ap.add_argument("--match", type=int, default=200, help="K lit ground neighbours for the brightness match; 0 = off")
    ap.add_argument("--crack", type=float, default=0.0,
                    help="cells darker than this share of the surrounding sunlit ground (Gaussian mean, --crack-sigma "
                         "cells) are cracks or stains and never used as sources; 0 = off")
    ap.add_argument("--crack-sigma", type=float, default=4.0)
    ap.add_argument("--crack-margin", type=int, default=1, help="grow crack cells by this many cells")
    ap.add_argument("--crack-min-cells", type=int, default=8,
                    help="a crack is a connected run of at least this many dark cells (single dark cells are texture)")
    ap.add_argument("--shift-step", type=float, default=0.25, help="shift lengths tried (scene units)")
    ap.add_argument("--max-shift", type=float, default=5.0)
    ap.add_argument("--max-shifts", type=int, default=1,
                    help="up to this many shifts, chosen greedily to fill cells the earlier ones could not")
    ap.add_argument("--cover", type=float, default=0.97,
                    help="first shift: the shortest covering at least this share of the best single shift")
    ap.add_argument("--map", action="store_true", help="write clone_map.png (patch, shifts, cracks) for figures")
    args = ap.parse_args()

    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    header, orig = read_ply(args.ply)
    _, base = read_ply(args.base)
    assert len(orig) == len(base), "--base must be the same splat as --ply with colours changed"
    st = json.loads((args.fix_dir / "stats.json").read_text())
    frac = np.load(args.fix_dir / "shadow_frac.npy")
    n_g = np.array(st["ground_plane"]["normal"]); d_g = st["ground_plane"]["offset"]; tol = st["ground_plane"]["tol"]
    shift = np.array(st["texture_fill"]["shift"])

    xyz = np.stack([orig["x"], orig["y"], orig["z"]], 1).astype(np.float64)
    opacity = 1 / (1 + np.exp(-orig["opacity"].astype(np.float64)))
    h = xyz @ n_g + d_g
    # same in-plane axes as fix_splat_shadows.py, so the shift means the same thing
    a = np.cross(n_g, [1.0, 0, 0] if abs(n_g[0]) < 0.9 else [0, 1.0, 0]); a /= np.linalg.norm(a)
    b = np.cross(n_g, a)
    p2 = np.stack([xyz @ a, xyz @ b], 1)
    near = np.abs(h) < tol
    changed = np.any(np.stack([orig[f"f_dc_{c}"] != base[f"f_dc_{c}"] for c in range(3)], 1), 1)

    # grid over the ground splats
    lo = p2[near].min(0) - 2 * np.abs(shift).max() - 1
    ij = np.floor((p2 - lo) / args.cell).astype(np.int64)
    shape = tuple(ij[near].max(0) + 2 * int(np.ceil(np.abs(shift).max() / args.cell)) + 4)
    inside = (ij >= 0).all(1) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
    g = near & inside & (opacity > args.opaque)
    count = np.zeros(shape); fsum = np.zeros(shape); chg = np.zeros(shape, bool); lit_n = np.zeros(shape)
    np.add.at(count, (ij[g, 0], ij[g, 1]), 1)
    np.add.at(fsum, (ij[g, 0], ij[g, 1]), frac[g])
    np.logical_or.at(chg, (ij[near & inside & changed, 0], ij[near & inside & changed, 1]), True)
    mean_frac = np.where(count > 0, fsum / np.maximum(count, 1), 0)

    shadow = ndimage.binary_closing(mean_frac >= args.hole_frac, np.ones((3, 3)), iterations=args.close)
    shadow = ndimage.binary_fill_holes(shadow)
    lab, nlab = ndimage.label(shadow)
    keep = np.unique(lab[chg & shadow]); keep = keep[keep > 0]
    hole = np.isin(lab, keep)
    obj = (h > args.object_height) & (opacity > args.opaque) & inside
    obj_cells = np.zeros(shape, bool); obj_cells[ij[obj, 0], ij[obj, 1]] = True
    obj_cells = ndimage.binary_dilation(obj_cells, iterations=args.object_margin)
    hole &= ~obj_cells

    # cracks and stains: sunlit cells much darker than the sunlit concrete around them are never used as sources,
    # so the clone does not repeat a crack next to the real one
    dc = np.stack([orig[f"f_dc_{c}"] for c in range(3)], 1).astype(np.float64)
    lin_dc = srgb_to_linear(np.clip(dc * SH_C0 + 0.5, 0, 1))
    lum = lin_dc @ np.array([0.2126, 0.7152, 0.0722])
    gl = g & (frac < args.source_lit)
    lsum = np.zeros(shape); lcnt = np.zeros(shape)
    np.add.at(lsum, (ij[gl, 0], ij[gl, 1]), lum[gl]); np.add.at(lcnt, (ij[gl, 0], ij[gl, 1]), 1)
    cell_lum = np.where(lcnt > 0, lsum / np.maximum(lcnt, 1), 0)
    has = (lcnt > 0).astype(float)
    local = ndimage.gaussian_filter(cell_lum, args.crack_sigma) / np.maximum(ndimage.gaussian_filter(has, args.crack_sigma), 1e-6)
    crack = np.zeros(shape, bool)
    if args.crack > 0:
        dark = (lcnt > 0) & (cell_lum < args.crack * local)
        # cracks are lines: keep connected runs of dark cells, not the single dark speckles that are part of the texture
        lab, nl = ndimage.label(dark, np.ones((3, 3)))
        size = np.bincount(lab.ravel())
        crack = dark & (size[lab] >= args.crack_min_cells)
        crack = ndimage.binary_dilation(crack, iterations=args.crack_margin)

    # sources: every hole cell takes the cell at one of a few whole-cell shifts that is lit ground, not object, not
    # shadow, not crack. Greedy cover: the first shift is the shortest that covers nearly as much as the best one,
    # then each next shift is the one that fills most of what is still empty.
    src_ok = (count > 0) & (mean_frac < args.source_lit) & ~obj_cells & ~shadow & ~crack
    xs, ys = np.nonzero(hole)
    cands = []
    for length in np.arange(args.shift_step, args.max_shift + 1e-9, args.shift_step):
        for t in np.deg2rad(np.arange(0, 360, 15)):
            s = (int(round(length * np.cos(t) / args.cell)), int(round(length * np.sin(t) / args.cell)))
            if s not in cands:
                cands.append(s)
    cands = np.array(cands)
    ok_c = np.zeros((len(cands), len(xs)), bool)
    for c_i, s in enumerate(cands):
        x2, y2 = xs + s[0], ys + s[1]
        okb = (x2 >= 0) & (x2 < shape[0]) & (y2 >= 0) & (y2 < shape[1])
        ok_c[c_i, okb] = src_ok[x2[okb], y2[okb]]
    assign = np.full(len(xs), -1)
    used = []
    for r in range(args.max_shifts):
        free = assign < 0
        gain = ok_c[:, free].sum(1)
        if gain.max() == 0:
            break
        c_i = int(np.argmax(gain >= args.cover * gain.max())) if r == 0 else int(np.argmax(gain))  # cands go shortest first
        assign[free & ok_c[c_i]] = c_i
        used.append(dict(shift_cells=cands[c_i].tolist(), length=round(float(np.linalg.norm(cands[c_i]) * args.cell), 3),
                         cells=int((free & ok_c[c_i]).sum())))
    good = assign >= 0
    fill = np.zeros(shape, bool); fill[xs[good], ys[good]] = True
    group = np.full(shape, -1); group[xs[good], ys[good]] = assign[good]

    cell_of = np.clip(ij[:, 0], 0, shape[0] - 1), np.clip(ij[:, 1], 0, shape[1] - 1)
    in_fill = near & inside & fill[cell_of]
    remove = np.nonzero(in_fill)[0]
    copy_parts, xyz_parts = [], []
    for c_i in np.unique(assign[good]):
        s = cands[c_i]
        src_cells = np.zeros(shape, bool)
        sel = assign == c_i
        src_cells[xs[sel] + s[0], ys[sel] + s[1]] = True
        idx = np.nonzero(near & inside & src_cells[cell_of])[0]
        copy_parts.append(idx)
        xyz_parts.append(xyz[idx] - (s[0] * args.cell * a + s[1] * args.cell * b))  # whole cells: lands in the hole cell
    copy_i = np.concatenate(copy_parts) if copy_parts else np.zeros(0, int)
    new_xyz = np.concatenate(xyz_parts) if xyz_parts else np.zeros((0, 3))

    new = orig[copy_i].copy()
    for k, c in zip("xyz", range(3)):
        new[k] = new_xyz[:, c].astype(np.float32)
    if args.map:
        save_map(args.out / "clone_map.png", cell_lum, shadow, hole, fill, group, crack, obj_cells, cands, assign, xs, ys)
    ratio = None
    if args.match > 0:
        lit = np.nonzero(near & (frac < 0.1) & (opacity > args.opaque) & ~in_fill)[0]
        tree = cKDTree(p2[lit])
        k = min(args.match, len(lit))
        _, jt = tree.query(new_xyz @ np.stack([a, b], 1), k=k)
        _, js = tree.query(p2[copy_i], k=k)
        ratio = np.clip(lin_dc[lit[jt]].mean(1) / np.maximum(lin_dc[lit[js]].mean(1), 1e-4), 0.5, 2.0)
        rgb = np.clip(dc[copy_i] * SH_C0 + 0.5, 0, 1)
        lin = srgb_to_linear(rgb) * ratio
        lin = lin / np.maximum(lin.max(1, keepdims=True), 1.0)
        new_rgb = np.clip(linear_to_srgb(lin), 0, 1)
        factor = np.where(rgb > 1e-3, new_rgb / np.maximum(rgb, 1e-3), 1.0)
        for c in range(3):
            new[f"f_dc_{c}"] = ((new_rgb[:, c] - 0.5) / SH_C0).astype(np.float32)
        rest = sorted((k for k in orig.dtype.names if k.startswith("f_rest_")), key=lambda k: int(k.split("_")[-1]))
        per_ch = len(rest) // 3
        for j, kk in enumerate(rest):
            new[kk] = (orig[kk][copy_i] * factor[:, min(j // max(per_ch, 1), 2)]).astype(np.float32)

    keep_mask = np.ones(len(base), bool); keep_mask[remove] = False
    out = np.concatenate([base[keep_mask], new])
    hdr = [f"element vertex {len(out)}" if l.startswith("element vertex") else l for l in header]
    write_ply(args.out / "splat_clone.ply", hdr, out)
    np.savez_compressed(args.out / "clone_index.npz", removed=remove, copied_from=copy_i)

    stats = dict(ply=str(args.ply), base=str(args.base), texture_shift=shift.tolist(), shifts=used,
                 crack_cells_near_patch=int((crack & ndimage.binary_dilation(hole, iterations=int(args.max_shift / args.cell))).sum()),
                 cell=args.cell, shadow_cells=int(shadow.sum()), patch_cells=int(hole.sum()), filled_cells=int(fill.sum()),
                 unfilled_cells=int(hole.sum() - fill.sum()), removed=int(len(remove)), added=int(len(copy_i)),
                 removed_per_cell=round(len(remove) / max(fill.sum(), 1), 1), added_per_cell=round(len(copy_i) / max(fill.sum(), 1), 1),
                 removed_median_scale=float(np.median(np.exp(np.stack([orig[f"scale_{c}"][remove] for c in range(3)], 1)).max(1))) if len(remove) else None,
                 added_median_scale=float(np.median(np.exp(np.stack([orig[f"scale_{c}"][copy_i] for c in range(3)], 1)).max(1))) if len(copy_i) else None,
                 match_ratio_median=None if ratio is None else np.median(ratio, 0).round(3).tolist(),
                 gaussians_before=int(len(base)), gaussians_after=int(len(out)),
                 params={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                 seconds=round(time.time() - t0, 1))
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "params"}, indent=2))


if __name__ == "__main__":
    main()
