#!/usr/bin/env python
"""Group the Gaussians of a splat into continuous surfaces and find each surface's grain direction.

Stage 3 of the shadow-free splat plan (lab-notes "Shadow-free splat plan"). No training,
no images: it reads an existing 3DGS PLY and uses only each Gaussian's position, shape and
orientation. Colour is deliberately NOT used, so a shadow edge never splits a surface.

1. Per Gaussian: rotation from the quaternion, scales from the log-scales. The shortest
   axis is the surface normal (unsigned); it is trusted when the Gaussian is flat
   (shortest / middle scale < --flat). The longest axis, projected into the tangent plane,
   is the grain cue, weighted by in-plane elongation (1 - middle / longest).
2. Surfaces: connect each flat Gaussian to its k nearest flat neighbours when they are
   close (< --radius x the local splat size) and their normals agree (< --angle degrees).
   Because the test is between neighbours, a curved surface still grows as one piece,
   while a crease (box corner, chair seat meeting a leg) splits it. Connected components
   below --min-size are dropped; non-flat Gaussians join the cluster of their nearest
   flat neighbour if it is within the same distance.
   --region-angle A (2026-10-12, after the first real run chained across the whole chair): grow
   regions instead, admitting a Gaussian only if its normal is within A of the region's running
   mean normal. Creases hold on noisy normals; curved surfaces split into patches.
   --normal-source pca|agree: replace the shortest-axis normal (median 24-45 deg off the local
   surface on IMG_6556) with a plane fit to the k nearest neighbours; agree also drops Gaussians
   whose two normals differ by more than --agree.
3. Orientation: structure tensor of the projected major axes (sum of w a a^T, so the
   sign of a doesn't matter). Per cluster: the top eigenvector is the dominant direction
   and coherence (l1 - l2) / (l1 + l2) says how strong it is (0 = no grain, 1 = all
   parallel). Per Gaussian: the same over its neighbours in the same cluster, for the
   inpainting stage.

Usage:
  cluster_surfaces.py --ply exports/IMG_6556/splat.ply --out exports/IMG_6556_clusters \
      [--crop X Y Z R] [--angle 20] [--k 16]
  cluster_surfaces.py --selftest --out /tmp/cluster_selftest

Outputs in --out: clusters_debug.ply (random colour per cluster, grey = unclustered),
orientation_debug.ply (hue = grain angle within the cluster, saturation = local coherence,
grey = no grain), clusters.npz (per-Gaussian arrays), stats.json (per-cluster table).
"""
import argparse
import colorsys
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

SH_C0 = 0.28209479177387814


# ---------- PLY (same format as fix_splat_shadows.py) ----------

def read_ply(path):
    with open(path, "rb") as f:
        header = []
        while (line := f.readline().decode().strip()) != "end_header":
            header.append(line)
        if "format binary_little_endian 1.0" not in header:
            raise SystemExit("ERROR: expected a binary little-endian PLY")
        n = next(int(l.split()[2]) for l in header if l.startswith("element vertex"))
        types = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4", "uint": "<u4"}
        dtype = [(l.split()[2], types[l.split()[1]]) for l in header if l.startswith("property")]
        data = np.frombuffer(f.read(n * np.dtype(dtype).itemsize), dtype=dtype, count=n).copy()
    return header, data


def write_ply(path, header, data):
    header = [f"element vertex {len(data)}" if l.startswith("element vertex") else l for l in header]
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\nend_header\n").encode())
        f.write(data.tobytes())


def ply_header(dtype):
    names = {"<f4": "float", "u1": "uchar", "<i4": "int"}
    return (["ply", "format binary_little_endian 1.0", "element vertex 0"]
            + [f"property {names[np.dtype(t).str]} {n}" for n, t in dtype])


# ---------- geometry ----------

def quat_to_rotmat(q):
    """3DGS stores rot_0..3 as (w, x, y, z), not necessarily normalised. Returns (N, 3, 3)."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], 1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], 1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1),
    ], 1)


def gaussian_frames(data):
    """Normal (shortest axis), major axis (longest), sorted scales, per Gaussian."""
    R = quat_to_rotmat(np.stack([data[f"rot_{i}"] for i in range(4)], 1).astype(np.float64))
    s = np.exp(np.stack([data[f"scale_{i}"] for i in range(3)], 1).astype(np.float64))
    order = np.argsort(s, axis=1)  # small, mid, large
    idx = np.arange(len(s))
    axes = R.transpose(0, 2, 1)  # axes[i, j] = j-th column of R_i = j-th local axis in world
    normal = axes[idx, order[:, 0]]
    major = axes[idx, order[:, 2]]
    s_sorted = np.take_along_axis(s, order, 1)
    return normal, major, s_sorted


def structure_dir(vecs, w):
    """Dominant unsigned direction and coherence of weighted 3D line directions."""
    T = np.einsum("n,ni,nj->ij", w, vecs, vecs)
    lam, V = np.linalg.eigh(T)
    l1, l2 = lam[2], lam[1]
    coh = float((l1 - l2) / (l1 + l2)) if l1 + l2 > 1e-12 else 0.0
    return V[:, 2], coh


# ---------- main ----------

def grow_regions(normal, src, dst, n, cos_lim):
    """Region growing over the neighbour links: a Gaussian joins a region only if its normal is within
    the limit of the region's running mean normal, so many small steps can't chain across a crease.
    Seeds are taken in order of how many links they have (interior of smooth surfaces first)."""
    order = np.argsort(np.r_[src, dst], kind="stable")
    both = np.r_[dst, src][order]
    start = np.searchsorted(np.r_[src, dst][order], np.arange(n + 1))
    comp = -np.ones(n, int)
    ncomp = 0
    for seed in np.argsort(-np.diff(start), kind="stable"):
        if comp[seed] >= 0:
            continue
        comp[seed] = ncomp
        acc = normal[seed].copy()
        stack = [seed]
        while stack:
            i = stack.pop()
            for j in both[start[i]:start[i + 1]]:
                if comp[j] >= 0:
                    continue
                m = acc / np.linalg.norm(acc)
                c = normal[j] @ m
                if abs(c) >= cos_lim:
                    comp[j] = ncomp
                    acc += normal[j] if c > 0 else -normal[j]
                    stack.append(j)
        ncomp += 1
    return comp, ncomp


def cluster(data, args):
    t0 = time.time()
    xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
    opacity = 1 / (1 + np.exp(-data["opacity"].astype(np.float64)))
    normal, major, s = gaussian_frames(data)
    flatness = s[:, 0] / np.maximum(s[:, 1], 1e-12)
    elong = 1 - s[:, 1] / np.maximum(s[:, 2], 1e-12)
    size = s[:, 2]

    use = opacity >= args.min_opacity
    if args.crop:
        c, r = np.array(args.crop[:3]), args.crop[3]
        use &= np.linalg.norm(xyz - c, axis=1) < r
    flat = use & (flatness < args.flat)
    if args.normal_source != "axis":
        # PCA normal from the positions of the k nearest kept Gaussians (sign-free, like the axis normal)
        ui = np.flatnonzero(use)
        _, nbp = cKDTree(xyz[ui]).query(xyz[flat], k=min(args.k + 1, len(ui)))
        P = xyz[ui][nbp] - xyz[ui][nbp].mean(1, keepdims=True)
        lam, V = np.linalg.eigh(np.einsum("nki,nkj->nij", P, P))
        pca = V[:, :, 0]
        disagree = np.degrees(np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", pca, normal[flat])), 0, 1)))
        print(f"axis vs PCA normal: median {np.median(disagree):.1f} deg, p90 {np.percentile(disagree, 90):.1f} deg")
        if args.normal_source == "agree":
            fl = np.flatnonzero(flat)
            drop = disagree > args.agree
            flat[fl[drop]] = False
            pca, disagree = pca[~drop], disagree[~drop]
            print(f"dropped {drop.sum():,} flat Gaussians whose axis and PCA normals differ by > {args.agree} deg")
        normal = normal.copy()
        normal[flat] = pca
    fi = np.flatnonzero(flat)
    print(f"{len(data):,} Gaussians, {use.sum():,} kept (opacity/crop), {len(fi):,} flat enough for a normal "
          f"(normals: {args.normal_source})")
    if len(fi) < args.min_size:
        raise SystemExit("ERROR: too few flat Gaussians to cluster")

    # 2. neighbour graph over flat Gaussians
    tree = cKDTree(xyz[fi])
    k = min(args.k + 1, len(fi))
    dist, nb = tree.query(xyz[fi], k=k)
    dist, nb = dist[:, 1:], nb[:, 1:]
    src = np.repeat(np.arange(len(fi)), k - 1)
    dst = nb.ravel()
    reach = args.radius * np.maximum(size[fi][src], size[fi][dst])
    nf = normal[fi]
    cosang = np.abs(np.einsum("ij,ij->i", nf[src], nf[dst]))
    ok = (dist.ravel() < reach) & (cosang > np.cos(np.radians(args.angle)))
    if args.region_angle > 0:
        comp, ncomp = grow_regions(nf, src[ok], dst[ok], len(fi), np.cos(np.radians(args.region_angle)))
    else:
        g = coo_matrix((np.ones(ok.sum()), (src[ok], dst[ok])), shape=(len(fi), len(fi)))
        ncomp, comp = connected_components(g, directed=False)
    counts = np.bincount(comp)
    keep = counts >= args.min_size
    remap = -np.ones(ncomp, int)
    order = np.argsort(-counts)
    order = order[keep[order]]
    remap[order] = np.arange(len(order))  # cluster 0 = largest
    labels = -np.ones(len(data), int)
    labels[fi] = remap[comp]
    mode = f"regions within {args.region_angle} deg of their mean normal" if args.region_angle > 0 else "components"
    print(f"{ok.sum():,} of {len(src):,} neighbour links kept, {ncomp:,} {mode}, "
          f"{len(order):,} clusters >= {args.min_size} Gaussians")

    # non-flat (and dropped) Gaussians join the nearest clustered flat neighbour
    clustered = np.flatnonzero(labels >= 0)
    rest = np.flatnonzero(use & (labels < 0))
    if len(rest) and len(clustered):
        d, j = cKDTree(xyz[clustered]).query(xyz[rest])
        near = d < args.radius * np.maximum(size[rest], size[clustered[j]])
        labels[rest[near]] = labels[clustered[j[near]]]
        print(f"{near.sum():,} of {len(rest):,} other Gaussians attached to a nearby cluster")

    # 3. orientation: project major axis into each Gaussian's tangent plane
    a = major - np.einsum("ij,ij->i", major, normal)[:, None] * normal
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    w = opacity * elong * flat  # only flat Gaussians carry a trustworthy in-plane direction

    stats = []
    dom = np.zeros((len(order), 3))
    for c in range(len(order)):
        m = labels == c
        mn = labels[fi] == c
        cn, _ = structure_dir(normal[fi][mn], np.ones(mn.sum()))
        d, coh = structure_dir(a[m], w[m])
        dom[c] = d
        p = xyz[m]
        ext = np.linalg.svd(p - p.mean(0), full_matrices=False)[1] / np.sqrt(len(p))
        stats.append({
            "cluster": c, "gaussians": int(m.sum()), "centre": p.mean(0).round(4).tolist(),
            "normal_axis": cn.round(4).tolist(),
            "normal_spread_deg": float(np.degrees(np.arccos(np.clip(
                np.abs(normal[fi][mn] @ cn).mean(), -1, 1)))),
            "grain_dir": d.round(4).tolist(), "grain_coherence": round(coh, 4),
            "mean_elongation": float(elong[m].mean()), "extent_std": ext.round(4).tolist(),
        })

    # per-Gaussian local orientation: structure tensor over same-cluster neighbours
    cl = np.flatnonzero(labels >= 0)
    local_dir = np.zeros((len(data), 3))
    local_coh = np.zeros(len(data))
    if len(cl):
        kk = min(args.k + 1, len(cl))
        _, nb2 = cKDTree(xyz[cl]).query(xyz[cl], k=kk)
        same = labels[cl][nb2] == labels[cl][:, None]
        av, wv = a[cl][nb2], (w[cl][nb2] * same)
        T = np.einsum("nk,nki,nkj->nij", wv, av, av)
        lam, V = np.linalg.eigh(T)
        s12 = lam[:, 2] + lam[:, 1]
        local_dir[cl] = V[:, :, 2]
        local_coh[cl] = np.where(s12 > 1e-12, (lam[:, 2] - lam[:, 1]) / np.maximum(s12, 1e-12), 0)

    print(f"done in {time.time() - t0:.1f} s")
    return dict(labels=labels, normal=normal, grain=a, grain_weight=w, flatness=flatness,
                elongation=elong, local_dir=local_dir, local_coh=local_coh, cluster_dir=dom), stats


def paint(data, rgb):
    out = data.copy()
    for i in range(3):
        out[f"f_dc_{i}"] = (rgb[:, i] - 0.5) / SH_C0
    for name in out.dtype.names:
        if name.startswith("f_rest_"):
            out[name] = 0
    return out


def write_outputs(header, data, res, stats, out):
    out.mkdir(parents=True, exist_ok=True)
    labels = res["labels"]
    rng = np.random.default_rng(1)
    pal = rng.uniform(0.15, 1.0, (max(labels.max() + 1, 1), 3))
    rgb = np.full((len(data), 3), 0.5)
    rgb[labels >= 0] = pal[labels[labels >= 0]]
    write_ply(out / "clusters_debug.ply", header, paint(data, rgb))

    # orientation: angle of the local grain within the cluster's tangent frame (mod 180 deg)
    rgb = np.full((len(data), 3), 0.5)
    for c, cd in enumerate(res["cluster_dir"]):
        m = np.flatnonzero(labels == c)
        if not len(m):
            continue
        n = np.array(stats[c]["normal_axis"])
        e1 = cd - (cd @ n) * n
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(n, [1, 0, 0] if abs(n[0]) < 0.9 else [0, 1, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        ld = res["local_dir"][m]
        ang = np.mod(np.arctan2(ld @ e2, ld @ e1), np.pi) / np.pi  # 0..1, 0 = cluster grain
        rgb[m] = [colorsys.hsv_to_rgb(h, min(1, 1.5 * cc), 0.95) for h, cc in zip(ang, res["local_coh"][m])]
    write_ply(out / "orientation_debug.ply", header, paint(data, rgb))

    np.savez_compressed(out / "clusters.npz", **res)
    (out / "stats.json").write_text(json.dumps({"clusters": stats}, indent=1))
    print(f"wrote {out}/clusters_debug.ply, orientation_debug.ply, clusters.npz, stats.json")


# ---------- synthetic check ----------

def synth_scene(seed=0):
    """Ground plane with grain along x (left half darkened like a shadow), a box standing on it
    with vertical grain on its sides and grain along y on top, and a sphere with grain along its
    meridians. Each Gaussian's grain gets 8 deg of random in-plane noise."""
    rng = np.random.default_rng(seed)
    pos, nrm, maj, col = [], [], [], []

    def add(p, n, m, c):
        pos.append(p); nrm.append(n); maj.append(m); col.append(c)

    # ground z = 0, 4 x 4, grain along x
    p = np.c_[rng.uniform(-2, 2, (6000, 2)), np.zeros(6000)]
    c = np.where(p[:, :1] < 0, 0.2, 0.6) * np.ones((1, 3))
    add(p, np.tile([0, 0, 1.0], (6000, 1)), np.tile([1.0, 0, 0], (6000, 1)), c)
    # box [0.5, 1.3] x [0.5, 1.3] x [0, 0.8]: 4 sides (vertical grain) + top (grain along y)
    for axis, val, n in [(0, 0.5, [-1, 0, 0]), (0, 1.3, [1, 0, 0]), (1, 0.5, [0, -1, 0]), (1, 1.3, [0, 1, 0])]:
        p = rng.uniform([0.5, 0.5, 0.02], [1.3, 1.3, 0.8], (1200, 3))
        p[:, axis] = val
        add(p, np.tile(n, (1200, 1)).astype(float), np.tile([0, 0, 1.0], (1200, 1)), np.full((1200, 3), 0.7))
    p = np.c_[rng.uniform(0.5, 1.3, (1200, 2)), np.full(1200, 0.8)]
    add(p, np.tile([0, 0, 1.0], (1200, 1)), np.tile([0, 1.0, 0], (1200, 1)), np.full((1200, 3), 0.8))
    # sphere r = 0.4 at (-1, -1, 0.6), grain along meridians (towards the poles)
    v = rng.normal(size=(4000, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    v = v[np.abs(v[:, 2]) < 0.9]  # meridians are undefined at the poles
    mer = np.c_[-v[:, 0] * v[:, 2], -v[:, 1] * v[:, 2], 1 - v[:, 2] ** 2]
    mer /= np.linalg.norm(mer, axis=1, keepdims=True)
    add(np.array([-1, -1, 0.6]) + 0.4 * v, v, mer, np.full((len(v), 3), 0.5))

    pos, nrm, maj, col = map(np.concatenate, (pos, nrm, maj, col))
    pos += rng.normal(0, 0.002, pos.shape)
    # rotation with columns (major, n x major, n); scales: major 0.04, minor 0.015, normal 0.002
    third = np.cross(nrm, maj)
    R = np.stack([maj, third, nrm], 2)
    jitter = np.radians(rng.normal(0, 8, len(R)))  # 8 deg of in-plane noise on the grain
    cj, sj = np.cos(jitter), np.sin(jitter)
    Rz = np.zeros((len(R), 3, 3)); Rz[:, 0, 0] = cj; Rz[:, 0, 1] = -sj; Rz[:, 1, 0] = sj; Rz[:, 1, 1] = cj; Rz[:, 2, 2] = 1
    R = R @ Rz
    # rotation matrix -> quaternion (w, x, y, z); R is proper and well away from 180 deg cases here
    tr = np.trace(R, axis1=1, axis2=2)
    q = np.empty((len(R), 4))
    for i in range(len(R)):
        M = R[i]
        if tr[i] > 0:
            S = np.sqrt(tr[i] + 1) * 2
            q[i] = [0.25 * S, (M[2, 1] - M[1, 2]) / S, (M[0, 2] - M[2, 0]) / S, (M[1, 0] - M[0, 1]) / S]
        elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
            S = np.sqrt(1 + M[0, 0] - M[1, 1] - M[2, 2]) * 2
            q[i] = [(M[2, 1] - M[1, 2]) / S, 0.25 * S, (M[0, 1] + M[1, 0]) / S, (M[0, 2] + M[2, 0]) / S]
        elif M[1, 1] > M[2, 2]:
            S = np.sqrt(1 + M[1, 1] - M[0, 0] - M[2, 2]) * 2
            q[i] = [(M[0, 2] - M[2, 0]) / S, (M[0, 1] + M[1, 0]) / S, 0.25 * S, (M[1, 2] + M[2, 1]) / S]
        else:
            S = np.sqrt(1 + M[2, 2] - M[0, 0] - M[1, 1]) * 2
            q[i] = [(M[1, 0] - M[0, 1]) / S, (M[0, 2] + M[2, 0]) / S, (M[1, 2] + M[2, 1]) / S, 0.25 * S]

    dtype = ([(n, "<f4") for n in ("x", "y", "z", "nx", "ny", "nz")]
             + [(f"f_dc_{i}", "<f4") for i in range(3)] + [(f"f_rest_{i}", "<f4") for i in range(45)]
             + [("opacity", "<f4")] + [(f"scale_{i}", "<f4") for i in range(3)] + [(f"rot_{i}", "<f4") for i in range(4)])
    data = np.zeros(len(pos), dtype=dtype)
    for i, n in enumerate("xyz"):
        data[n] = pos[:, i]
    for i in range(3):
        data[f"f_dc_{i}"] = (col[:, i] - 0.5) / SH_C0
        data[f"rot_{i}"] = q[:, i]
    data["rot_3"] = q[:, 3]
    data["opacity"] = 3.0
    for i, s in enumerate((0.04, 0.015, 0.002)):
        data[f"scale_{i}"] = np.log(s)
    return ply_header(dtype), data


def selftest(args):
    header, data = synth_scene()
    args.out.mkdir(parents=True, exist_ok=True)
    write_ply(args.out / "synthetic_input.ply", header, data)
    res, stats = cluster(data, args)
    write_outputs(header, data, res, stats, args.out)
    labels = res["labels"]
    # expected groups by construction: ground 0..5999, box sides 4 x 1200, box top 1200, sphere rest
    groups = {"ground": slice(0, 6000), "box -x": slice(6000, 7200), "box +x": slice(7200, 8400),
              "box -y": slice(8400, 9600), "box +y": slice(9600, 10800), "box top": slice(10800, 12000),
              "sphere": slice(12000, len(data))}
    ok = True
    print("\ngroup      main cluster  share  grain_dir                coherence")
    used = set()
    for name, sl in groups.items():
        lab = labels[sl]
        main = np.bincount(lab[lab >= 0]).argmax() if (lab >= 0).any() else -1
        share = float((lab == main).mean())
        st = stats[main] if main >= 0 else {}
        print(f"{name:9s}  {main:12d}  {share:5.2f}  {str(st.get('grain_dir')):24s} {st.get('grain_coherence')}")
        # region growing splits a curved surface into patches by design; flat faces must stay whole
        ok &= main not in used and (share > 0.9 or (name == "sphere" and args.region_angle > 0))
        used.add(main)
    # every cluster must come from one group only (no surface leaks into another)
    gid = np.zeros(len(data), int)
    for g, sl in enumerate(groups.values()):
        gid[sl] = g
    purity = min(np.bincount(gid[labels == c]).max() / (labels == c).sum() for c in range(labels.max() + 1))
    print(f"clusters: {labels.max() + 1}, worst purity {purity:.3f}")
    ok &= purity > 0.95  # PCA normals blur creases: a few edge splats can land on the adjacent face
    gd = np.abs(np.array(stats[labels[0]]["grain_dir"]))
    ok &= gd[0] > 0.95 and stats[labels[0]]["grain_coherence"] > 0.8  # ground grain along x
    side = stats[labels[6000]]
    ok &= abs(side["grain_dir"][2]) > 0.95  # box side grain vertical
    # on the curved sphere one 3D direction is only a summary; the local field must follow the meridians
    v = (np.stack([data["x"], data["y"], data["z"]], 1)[12000:] - [-1, -1, 0.6]) / 0.4
    mer = np.c_[-v[:, 0] * v[:, 2], -v[:, 1] * v[:, 2], 1 - v[:, 2] ** 2]
    mer /= np.linalg.norm(mer, axis=1, keepdims=True)
    err = np.degrees(np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", res["local_dir"][12000:], mer)), 0, 1)))
    gerr = np.degrees(np.arccos(np.clip(np.abs(res["local_dir"][:6000, 0]), 0, 1)))
    print(f"local grain error: ground median {np.median(gerr):.1f} deg, sphere median {np.median(err):.1f} deg")
    ok &= np.median(err) < 10 and np.median(gerr) < 5
    # the darkened half of the ground must not split off (colour is not used); a few splats where the
    # box stands on the ground (lit half) may go to a box face with PCA normals
    dark = data["x"][:6000] < 0
    ok &= (labels[:6000][dark] == labels[0]).mean() > 0.99 and (labels[:6000][~dark] == labels[0]).mean() > 0.95
    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--crop", type=float, nargs=4, metavar=("X", "Y", "Z", "R"),
                    help="keep only Gaussians within R of (X, Y, Z), e.g. around the hero object")
    ap.add_argument("--min-opacity", type=float, default=0.3)
    ap.add_argument("--flat", type=float, default=0.3, help="max shortest/middle scale for a trusted normal")
    ap.add_argument("--k", type=int, default=16, help="neighbours per Gaussian")
    ap.add_argument("--radius", type=float, default=3.0, help="link distance, in multiples of the larger splat's longest axis")
    ap.add_argument("--angle", type=float, default=20.0, help="max normal difference between linked neighbours (degrees)")
    ap.add_argument("--min-size", type=int, default=200, help="smallest cluster kept")
    ap.add_argument("--region-angle", type=float, default=0.0,
                    help="if > 0, grow regions against their running mean normal (degrees) instead of plain "
                         "connected components; stops chaining across creases on noisy normals")
    ap.add_argument("--normal-source", choices=["axis", "pca", "agree"], default="axis",
                    help="axis: each Gaussian's shortest axis; pca: plane fit to its k nearest neighbours; "
                         "agree: pca, keeping only Gaussians whose two normals differ by <= --agree")
    ap.add_argument("--agree", type=float, default=30.0, help="max axis/PCA normal difference for --normal-source agree")
    ap.add_argument("--selftest", action="store_true", help="run on a synthetic ground + box + sphere scene")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(selftest(args))
    if not args.ply:
        ap.error("--ply is required (or use --selftest)")
    header, data = read_ply(args.ply)
    res, stats = cluster(data, args)
    write_outputs(header, data, res, stats, args.out)
    for s in stats[:15]:
        print(f"cluster {s['cluster']:3d}: {s['gaussians']:8,d} Gaussians, normal {s['normal_axis']}, "
              f"spread {s['normal_spread_deg']:.1f} deg, grain {s['grain_dir']} coherence {s['grain_coherence']}")


if __name__ == "__main__":
    main()
