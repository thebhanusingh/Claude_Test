#!/usr/bin/env python
"""Stage 4 prototype: flag the Gaussians the sun cannot reach (hero object + the ground around it).

Two kinds of shadow, per Gaussian:
  attached - the surface faces away from the sun (signed normal . sun < --attached-cos). Needs SIGNED normals: each
             grown (trusted-normal) object Gaussian is oriented toward the cameras that actually see it, using a z-buffer
             with a tight ABSOLUTE depth tolerance (--vis-tol) so the two faces of a thin slat are told apart. Object
             Gaussians without a trusted normal take the majority vote of their --knn nearest grown neighbours.
             Ground receivers use the plane normal (never attached).
  cast     - something lies between the Gaussian and the sun: fit_sun.predict (orthographic shadow map along the sun).
             Casters are the object's clustered Gaussians (grass/floaters are not casters). Object receivers are only
             shadowed by OTHER clusters (a surface must not shadow itself through its own thickness).
shadow = attached | cast. Gaussians with |signed normal . sun| < --graze sit near the terminator, where a 1-2 deg sun
error flips the answer; they are flagged ('grazing') and scored separately. --revote recomputes the object's mask
votes with a tight depth test (the fit_sun votes let a thin slat's back face inherit its front face's votes).
Evaluated against the multi-view mask votes cached by fit_sun.py (ground) and
fit_sun_self.py (object): precision / recall / IoU, split by ground / object and by shadow kind.

Usage:
  shadow_visibility.py --ply exports/IMG_6556/splat.ply --colmap data/IMG_6556/colmap/sparse/0_refined \
      --plane-from exports/IMG_6556_shadowfix_v8_o/stats.json --sun-from exports/IMG_6556_sun_edges_fitshift/sun_edges.json \
      --clusters exports/IMG_6556_clusters_chair_r30agree_v3/clusters.npz --crop 0.713 1.091 0.464 1.065 \
      --ground-votes exports/IMG_6556_sun_chair/votes.npz --object-votes exports/IMG_6556_sun_self/votes_object.npz \
      --out exports/IMG_6556_shadow4
Writes <out>/shadow_flags.npz (indices + flags + signed normals), shadow_debug.ply (region only: red cast, blue
attached, magenta both, original colour lit), shadow4.png, stats.json. Trains nothing.
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
from fit_sun import plane_frame, predict, sun_vec
from fix_splat_shadows import project, qvec2rotmat, read_cameras_bin, read_images_bin


def prf(pred, obs):
    tp, fp, fn = (pred & obs).sum(), (pred & ~obs).sum(), (~pred & obs).sum()
    return dict(n=int(len(obs)), observed=round(float(obs.mean()), 4), predicted=round(float(pred.mean()), 4),
                precision=round(float(tp / max(tp + fp, 1)), 3), recall=round(float(tp / max(tp + fn, 1)), 3),
                iou=round(float(tp / max(tp + fp + fn, 1)), 3), accuracy=round(float((pred == obs).mean()), 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--colmap", type=Path, required=True)
    ap.add_argument("--plane-from", type=Path, required=True)
    ap.add_argument("--sun-from", type=Path, required=True, help="sun_edges.json (nelder_mead) or sun.json (sun_world)")
    ap.add_argument("--clusters", type=Path, required=True)
    ap.add_argument("--crop", type=float, nargs=4, metavar=("X", "Y", "Z", "R"), required=True)
    ap.add_argument("--ground-radius", type=float, default=1.8, help="ground receivers within this of the footprint")
    ap.add_argument("--ground-votes", type=Path)
    ap.add_argument("--object-votes", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--opaque", type=float, default=0.5)
    ap.add_argument("--occluder-min", type=float, default=0.05)
    ap.add_argument("--min-views", type=int, default=5)
    ap.add_argument("--cell", type=float, default=0.02)
    ap.add_argument("--dilate", type=int, default=1)
    ap.add_argument("--eps", type=float, default=0.02, help="cast-shadow depth margin, ground")
    ap.add_argument("--eps-object", type=float, default=0.03, help="cast-shadow depth margin, object")
    ap.add_argument("--attached-cos", type=float, default=0.0, help="attached when signed normal . sun < this")
    ap.add_argument("--every", type=int, default=4, help="every Nth frame for the normal orientation")
    ap.add_argument("--zbuf-scale", type=float, default=0.5)
    ap.add_argument("--vis-tol", type=float, default=0.012, help="absolute depth tolerance for visibility (units)")
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--graze", type=float, default=0.15, help="|signed normal . sun| below this = grazing band")
    ap.add_argument("--revote", action="store_true", help="recompute object mask votes with a tight depth test")
    ap.add_argument("--revote-ztol", type=float, default=0.005, help="relative depth tolerance for --revote")
    ap.add_argument("--masks", type=Path, help="shadow masks folder (needed for --revote)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    header, data = read_ply(a.ply)
    xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
    opacity = 1 / (1 + np.exp(-data["opacity"].astype(np.float64)))
    gp = json.loads(a.plane_from.read_text())["ground_plane"]
    up, offset, tol = np.array(gp["normal"], float), float(gp["offset"]), float(gp["tol"])
    up /= np.linalg.norm(up)
    frame = plane_frame(up)
    sj = json.loads(a.sun_from.read_text())
    if "nelder_mead" in sj:
        s = sun_vec(sj["nelder_mead"]["elevation_deg"], sj["nelder_mead"]["azimuth_deg"], frame)
    else:
        s = np.array(sj["sun_world"], float)
    s /= np.linalg.norm(s)
    print(f"sun {np.round(s, 4).tolist()}: elevation {np.degrees(np.arcsin(s @ up)):.2f} deg above the plane", flush=True)

    z = np.load(a.clusters)
    labels, nrm = z["labels"], z["normal"]
    grown_all = z["grown"] if "grown" in z.files else np.zeros(len(labels), bool)
    height = xyz @ up + offset
    c, r = np.array(a.crop[:3]), a.crop[3]
    obj = np.flatnonzero((labels >= 0) & (opacity > a.opaque) & (np.linalg.norm(xyz - c, axis=1) < r)
                         & (height > a.occluder_min))
    foot = c - (c @ up + offset) * up
    gnd = np.flatnonzero((opacity > a.opaque) & (np.abs(height) <= tol)
                         & (np.linalg.norm(xyz - foot, axis=1) < a.ground_radius))
    print(f"region: {len(obj):,} object Gaussians, {len(gnd):,} ground Gaussians", flush=True)

    # ---- signed normals for grown object Gaussians: orient toward the cameras that see them ----
    trusted = grown_all[obj]
    ti = obj[trusted]
    n_t = nrm[ti] / np.maximum(np.linalg.norm(nrm[ti], axis=1, keepdims=True), 1e-12)
    cams = read_cameras_bin(a.colmap / "cameras.bin")
    frames = read_images_bin(a.colmap / "images.bin")[:: a.every]
    region_pts = xyz[np.r_[obj, gnd]]
    vote = np.zeros(len(ti))
    seen = np.zeros(len(ti), int)
    for k, fr in enumerate(frames):
        cam = cams[fr["cam"]]
        w, h = cam["width"], cam["height"]
        R, t = qvec2rotmat(fr["q"]), fr["t"]
        C = -R.T @ t
        zw, zh = int(w * a.zbuf_scale), int(h * a.zbuf_scale)
        zbuf = np.full(zw * zh, np.inf)
        pc = region_pts @ R.T + t
        pc = pc[pc[:, 2] > 1e-3]
        uo = project(cam, pc)
        ok = (uo[:, 0] >= 0) & (uo[:, 0] < w) & (uo[:, 1] >= 0) & (uo[:, 1] < h)
        cell = (uo[ok, 1] * a.zbuf_scale).astype(np.int64) * zw + (uo[ok, 0] * a.zbuf_scale).astype(np.int64)
        np.minimum.at(zbuf, np.clip(cell, 0, zw * zh - 1), pc[ok, 2])
        tc = xyz[ti] @ R.T + t
        front = tc[:, 2] > 1e-3
        ut = np.full((len(ti), 2), -1.0)
        ut[front] = project(cam, tc[front])
        inb = front & (ut[:, 0] >= 0) & (ut[:, 0] < w) & (ut[:, 1] >= 0) & (ut[:, 1] < h)
        cl = np.clip((ut[:, 1] * a.zbuf_scale).astype(np.int64) * zw + (ut[:, 0] * a.zbuf_scale).astype(np.int64),
                     0, zw * zh - 1)
        vis = inb & (tc[:, 2] <= zbuf[cl] + a.vis_tol)
        view = C - xyz[ti]
        d = np.einsum("ij,ij->i", n_t, view / np.linalg.norm(view, axis=1, keepdims=True))
        vote[vis] += np.sign(d[vis]) * np.abs(d[vis])  # weight by how squarely the camera sees the face
        seen[vis] += 1
    sign = np.where(vote >= 0, 1.0, -1.0)
    n_signed = n_t * sign[:, None]
    conf = np.abs(vote) / np.maximum(seen, 1)
    print(f"signed normals: {len(ti):,} grown object Gaussians, seen by median {int(np.median(seen))} of "
          f"{len(frames)} frames, {(seen == 0).sum():,} never seen; median orientation confidence {np.median(conf):.2f} "
          f"({time.time() - t0:.0f} s)", flush=True)

    # ---- attached shadow ----
    att_t = (n_signed @ s) < a.attached_cos
    att_t[seen == 0] = False
    graze_t = np.abs(n_signed @ s) < a.graze
    att = np.zeros(len(obj), bool)
    graze = np.zeros(len(obj), bool)
    att[trusted], graze[trusted] = att_t, graze_t
    if (~trusted).any():
        tree = cKDTree(xyz[ti][seen > 0])
        _, nb = tree.query(xyz[obj[~trusted]], k=a.knn)
        att[~trusted] = att_t[seen > 0][nb].mean(1) > 0.5
        graze[~trusted] = graze_t[seen > 0][nb].mean(1) > 0.5
    print(f"attached: {att.mean() * 100:.1f}% of object Gaussians face away from the sun "
          f"(trusted {att_t.mean() * 100:.1f}%)", flush=True)

    # ---- cast shadow ----
    occ, occ_lab = xyz[obj], labels[obj]
    cast_g = predict(occ, xyz[gnd], s, a.cell, a.dilate, a.eps)
    cast_o = np.zeros(len(obj), bool)
    for L in np.unique(occ_lab):
        m = occ_lab == L
        cast_o[m] = predict(occ[~m], occ[m], s, a.cell, a.dilate, a.eps_object)
    shadow_o = att | cast_o
    print(f"cast: ground {cast_g.mean() * 100:.1f}%, object {cast_o.mean() * 100:.1f}%; object in shadow "
          f"(attached or cast) {shadow_o.mean() * 100:.1f}% ({time.time() - t0:.0f} s)", flush=True)

    # ---- evaluate against the mask votes ----
    stats = dict(sun_world=s.round(5).tolist(), object_gaussians=int(len(obj)), ground_gaussians=int(len(gnd)),
                 attached_share=round(float(att.mean()), 4), cast_object_share=round(float(cast_o.mean()), 4),
                 cast_ground_share=round(float(cast_g.mean()), 4), params={k: str(v) if isinstance(v, Path) else v
                                                                          for k, v in vars(a).items()})
    obs_o = None
    if a.ground_votes:
        v = np.load(a.ground_votes)
        pos = {i: j for j, i in enumerate(v["rec_idx"])}
        jj = np.array([pos.get(i, -1) for i in gnd])
        okv = (jj >= 0)
        okv[okv] &= v["seen"][jj[okv]] >= a.min_views
        obs_g = v["shadow"][jj[okv]] / v["seen"][jj[okv]] >= 0.5
        stats["ground_vs_masks"] = prf(cast_g[okv], obs_g)
        print(f"ground vs masks: {stats['ground_vs_masks']}", flush=True)
    if a.revote:
        # object votes again, with a tight depth test so a slat's back face no longer inherits its front face's votes
        from fit_sun import collect_votes
        vpath = a.out / "votes_object_tight.npz"
        if vpath.exists() and np.array_equal(np.load(vpath)["rec_idx"], obj):
            print(f"reusing {vpath}")
        else:
            va = argparse.Namespace(colmap=a.colmap, masks=a.masks, every=2, opaque=a.opaque, min_views=a.min_views,
                                    zbuf_scale=a.zbuf_scale, z_tol=a.revote_ztol, zbuf_radius=3.0)
            sh, se = collect_votes(xyz, opacity, obj, va)[:2]  # fit_sun now also returns the camera direction
            np.savez_compressed(vpath, rec_idx=obj, shadow=sh, seen=se)
        a.object_votes = vpath
    if a.object_votes:
        v = np.load(a.object_votes)
        pos = {i: j for j, i in enumerate(v["rec_idx"])}
        jj = np.array([pos.get(i, -1) for i in obj])
        okv = jj >= 0
        okv[okv] &= v["seen"][jj[okv]] >= a.min_views
        obs_o = np.zeros(len(obj), bool)
        obs_o[okv] = v["shadow"][jj[okv]] / v["seen"][jj[okv]] >= 0.5
        clear = okv & ~graze  # near the terminator a 1-2 deg sun error flips the answer: scored separately
        stats["object_vs_masks"] = dict(shadow=prf(shadow_o[okv], obs_o[okv]), attached_only=prf(att[okv], obs_o[okv]),
                                        cast_only=prf(cast_o[okv], obs_o[okv]),
                                        shadow_excluding_grazing=prf(shadow_o[clear], obs_o[clear]),
                                        grazing_only=prf(shadow_o[okv & graze], obs_o[okv & graze]))
        print(f"grazing band |n.s| < {a.graze}: {graze.mean() * 100:.1f}% of object Gaussians", flush=True)
        for k2, v2 in stats["object_vs_masks"].items():
            print(f"object vs masks, {k2:13s}: {v2}", flush=True)
        per = []
        for L in np.unique(occ_lab):
            m = (occ_lab == L) & okv
            if m.sum() >= 100:
                per.append(dict(cluster=int(L), n=int(m.sum()), attached=round(float(att[m].mean()), 3),
                                cast=round(float(cast_o[m].mean()), 3), observed=round(float(obs_o[m].mean()), 3),
                                agree=round(float((shadow_o[m] == obs_o[m]).mean()), 3)))
        stats["object_per_cluster"] = per
        print("per cluster (attached / cast / observed shadow share, agreement):")
        for p in per:
            print(f"  cluster {p['cluster']:3d}: n {p['n']:6,d}  {p['attached']:.2f} / {p['cast']:.2f} / "
                  f"{p['observed']:.2f}  agree {p['agree']:.2f}")
    stats["seconds"] = round(time.time() - t0, 1)
    (a.out / "stats.json").write_text(json.dumps(stats, indent=1))
    np.savez_compressed(a.out / "shadow_flags.npz", object_idx=obj, ground_idx=gnd, attached=att, grazing=graze, cast_object=cast_o,
                        cast_ground=cast_g, trusted_idx=ti, signed_normal=n_signed.astype(np.float32),
                        orient_seen=seen, orient_conf=conf.astype(np.float32), sun=s)

    # debug PLY: region only
    idx = np.r_[obj, gnd]
    sub = data[idx].copy()
    rgb = np.clip(np.stack([sub[f"f_dc_{i}"] for i in range(3)], 1) * SH_C0 + 0.5, 0, 1)
    cast_all, att_all = np.r_[cast_o, cast_g], np.r_[att, np.zeros(len(gnd), bool)]
    rgb[cast_all & ~att_all] = [0.9, 0.1, 0.1]
    rgb[att_all & ~cast_all] = [0.1, 0.3, 0.95]
    rgb[att_all & cast_all] = [0.85, 0.1, 0.85]
    for i in range(3):
        sub[f"f_dc_{i}"] = (rgb[:, i] - 0.5) / SH_C0
    for nme in sub.dtype.names:
        if nme.startswith("f_rest_"):
            sub[nme] = 0
    write_ply(a.out / "shadow_debug.ply", header, sub)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _, e1, e2 = frame
    az = np.arctan2(s @ e2, s @ e1)
    hdir = np.cos(az) * e1 + np.sin(az) * e2
    side = np.cross(up, hdir)
    P = xyz[obj]
    x, y, dpt = P @ hdir, P @ up + offset, P @ side
    order = np.argsort(dpt)
    panels = [(shadow_o, "predicted: attached or cast"), (att, "attached (faces away from sun)"),
              (cast_o, "cast (by other chair parts)")]
    if obs_o is not None:
        panels.insert(0, (obs_o, "observed (mask votes)"))
    fig, axs = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 5.5))
    for axx, (m, title) in zip(axs, panels):
        axx.scatter(x[order], y[order], c=np.where(m[order], "#212529", "#ced4da"), s=0.6)
        axx.set_title(title); axx.set_aspect("equal"); axx.set_xlabel("towards the sun (horizontal)")
    axs[0].set_ylabel("height above ground")
    fig.tight_layout(); fig.savefig(a.out / "shadow4.png", dpi=100)
    print(f"wrote {a.out}: shadow_flags.npz, shadow_debug.ply, shadow4.png, stats.json ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
