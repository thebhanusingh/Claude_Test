#!/usr/bin/env python
"""Sun direction from the hero object's cast shadow on the ground AND its self-shadow (extension of fit_sun.py).

Receivers:
  ground - ground Gaussians near the object's footprint (votes reused from a fit_sun.py run: --ground-votes);
  object - the object's own Gaussians (stage-3 clusters, opacity > --opaque, inside --crop). Their shadow/lit votes come
           from the same multi-view mask voting as fit_sun.py (cached in <out>/votes_object.npz).
Casters: the object's clustered Gaussians only (grass and floaters are diffusely lit, not casters).
Prediction for sun s: fit_sun.predict (orthographic shadow map along s, furthest-toward-sun occluder depth per cell);
an object receiver is shadowed when part of the object lies between it and the sun by more than --eps-object. Faces
turned away from the sun are covered through the object's own thickness (the front face occludes the back face).
Score per direction: IoU on ground receivers, IoU on object receivers, and their mean. Each of the three is searched
coarse-to-fine (5, 1, 0.25 deg) and reported with an uncertainty radius (directions within --tie of the best).

Usage:
  fit_sun_self.py --ply exports/IMG_6556/splat.ply --colmap data/IMG_6556/colmap/sparse/0_refined \
      --masks data/IMG_6556/shadow_masks --plane-from exports/IMG_6556_shadowfix_v8_o/stats.json \
      --clusters exports/IMG_6556_clusters_chair_r30agree_v3/clusters.npz --crop 0.713 1.091 0.464 1.065 \
      --ground-votes exports/IMG_6556_sun_chair/votes.npz --score-radius 1.2 --out exports/IMG_6556_sun_self \
      [--true-elevation 55.51]
Writes <out>/sun_self.json, votes_object.npz, sun_self.png. Trains nothing.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fit_sun import angle_between, collect_votes, plane_frame, predict, sun_vec


def iou(p, o):
    u = (p | o).sum()
    return float((p & o).sum() / u) if u else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--colmap", type=Path, required=True)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--plane-from", type=Path, required=True)
    ap.add_argument("--clusters", type=Path, required=True, help="clusters.npz: the object = labels >= 0")
    ap.add_argument("--crop", type=float, nargs=4, metavar=("X", "Y", "Z", "R"), required=True)
    ap.add_argument("--ground-votes", type=Path, required=True, help="votes.npz from a fit_sun.py run")
    ap.add_argument("--score-radius", type=float, default=1.2, help="ground receivers within this of the footprint")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--opaque", type=float, default=0.5)
    ap.add_argument("--occluder-min", type=float, default=0.05)
    ap.add_argument("--cell", type=float, default=0.02)
    ap.add_argument("--dilate", type=int, default=1)
    ap.add_argument("--eps", type=float, default=0.02, help="depth margin for ground receivers")
    ap.add_argument("--eps-object", type=float, default=0.03,
                    help="depth margin for object receivers (larger: a surface must not shadow itself)")
    ap.add_argument("--self-mode", choices=["all", "other-clusters"], default="other-clusters",
                    help="which object Gaussians may shadow an object receiver: all, or only other stage-3 clusters")
    ap.add_argument("--tie", type=float, default=0.005)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--min-views", type=int, default=5)
    ap.add_argument("--zbuf-scale", type=float, default=0.25)
    ap.add_argument("--z-tol", type=float, default=0.03)
    ap.add_argument("--zbuf-radius", type=float, default=0.0)
    ap.add_argument("--true-elevation", type=float, help="reference elevation for the report (e.g. sun calculator)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    from cluster_surfaces import read_ply
    _, data = read_ply(a.ply)
    xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
    opacity = 1 / (1 + np.exp(-data["opacity"].astype(np.float64)))
    gp = json.loads(a.plane_from.read_text())["ground_plane"]
    up, offset = np.array(gp["normal"], float), float(gp["offset"])
    up /= np.linalg.norm(up)
    height = xyz @ up + offset
    labels = np.load(a.clusters)["labels"]
    c, r = np.array(a.crop[:3]), a.crop[3]
    obj_mask = (labels >= 0) & (opacity > a.opaque) & (np.linalg.norm(xyz - c, axis=1) < r) & (height > a.occluder_min)
    obj_idx = np.flatnonzero(obj_mask)
    occ = xyz[obj_idx]
    print(f"object: {len(obj_idx):,} clustered opaque Gaussians (casters and object receivers)", flush=True)

    # ground receivers from the cached fit_sun votes
    gv = np.load(a.ground_votes)
    g_idx, g_sh, g_seen = gv["rec_idx"], gv["shadow"], gv["seen"]
    foot = c - (c @ up + offset) * up
    keep = (g_seen >= a.min_views) & (np.linalg.norm(xyz[g_idx] - foot, axis=1) < a.score_radius)
    g_rec, g_obs = xyz[g_idx[keep]], (g_sh[keep] / g_seen[keep]) >= 0.5
    print(f"ground: {len(g_rec):,} receivers within {a.score_radius} of the footprint, {g_obs.mean() * 100:.1f}% in shadow")

    # object receivers: own votes (cached)
    vpath = a.out / "votes_object.npz"
    if vpath.exists() and np.array_equal(np.load(vpath)["rec_idx"], obj_idx):
        v = np.load(vpath)
        o_sh, o_seen = v["shadow"], v["seen"]
        print(f"reusing object votes from {vpath}")
    else:
        o_sh, o_seen = collect_votes(xyz, opacity, obj_idx, a)
        np.savez_compressed(vpath, rec_idx=obj_idx, shadow=o_sh, seen=o_seen)
    ok = o_seen >= a.min_views
    o_rec, o_obs = occ[ok], (o_sh[ok] / o_seen[ok]) >= 0.5
    occ_lab, o_lab = labels[obj_idx], labels[obj_idx][ok]

    def predict_object(s):
        if a.self_mode == "all":
            return predict(occ, o_rec, s, a.cell, a.dilate, a.eps_object)
        # only OTHER parts cast onto a part: a slat's front face no longer "shadows" its back face through thickness,
        # which at grazing sun angles disagrees with the masks (they call grazing-lit faces lit)
        out = np.zeros(len(o_rec), bool)
        for L in np.unique(o_lab):
            m = o_lab == L
            out[m] = predict(occ[occ_lab != L], o_rec[m], s, a.cell, a.dilate, a.eps_object)
        return out
    print(f"object: {len(o_rec):,} receivers seen >= {a.min_views}x, {o_obs.mean() * 100:.1f}% in shadow "
          f"({time.time() - t0:.0f} s)", flush=True)

    frame = plane_frame(up)
    cache = {}

    def scores(el, az):
        key = (round(el, 3), round(az % 360, 3))
        if key not in cache:
            s = sun_vec(el, az, frame)
            pg = predict(occ, g_rec, s, a.cell, a.dilate, a.eps)
            po = predict_object(s)
            ig, io = iou(pg, g_obs), iou(po, o_obs)
            cache[key] = (ig, io, 0.5 * (ig + io))
        return cache[key]

    def search(k):
        def grid(els, azs):
            return [(e, z % 360, scores(e, z)[k]) for e in els if 1 <= e <= 89 for z in azs]
        coarse = grid(np.arange(5, 86, 5.0), np.arange(0, 360, 5.0))
        e, z, _ = max(coarse, key=lambda t: t[2])
        mid = grid(e + np.arange(-5, 5.1, 1.0), z + np.arange(-5, 5.1, 1.0))
        e, z, _ = max(mid, key=lambda t: t[2])
        fine = grid(e + np.arange(-1.5, 1.51, 0.25), z + np.arange(-1.5, 1.51, 0.25))
        be, bz, bv = max(fine, key=lambda t: t[2])
        s = sun_vec(be, bz, frame)
        tied = [sun_vec(e2, z2, frame) for e2, z2, v2 in mid + fine if v2 >= bv - a.tie]
        return dict(elevation_deg=float(be), azimuth_deg=float(bz), score=round(bv, 4),
                    uncertainty_deg=round(max(angle_between(s, t) for t in tied), 2), tied=len(tied),
                    iou_ground=round(scores(be, bz)[0], 4), iou_object=round(scores(be, bz)[1], 4))

    res = {}
    for k, name in enumerate(["ground", "object", "combined"]):
        res[name] = search(k)
        rr = res[name]
        err = f", error {rr['elevation_deg'] - a.true_elevation:+.2f} vs {a.true_elevation}" if a.true_elevation else ""
        print(f"{name:9s} fit: elevation {rr['elevation_deg']:.2f} deg, azimuth {rr['azimuth_deg']:.2f} deg, "
              f"uncertainty {rr['uncertainty_deg']:.2f} deg ({rr['tied']} tied) | IoU ground {rr['iou_ground']:.3f}, "
              f"object {rr['iou_object']:.3f}{err} ({time.time() - t0:.0f} s)", flush=True)

    # elevation profile at the combined azimuth (how sharply each term pins the elevation)
    az_c = res["combined"]["azimuth_deg"]
    prof = [(e, *scores(e, az_c)) for e in np.arange(40, 70.1, 1.0)]
    res["profile_at_combined_azimuth"] = [dict(elevation=float(e), ground=round(g, 4), object=round(o, 4),
                                               combined=round(m, 4)) for e, g, o, m in prof]
    res.update(object_receivers=int(len(o_rec)), ground_receivers=int(len(g_rec)), casters=int(len(occ)),
               true_elevation=a.true_elevation, seconds=round(time.time() - t0, 1),
               params={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()})
    (a.out / "sun_self.json").write_text(json.dumps(res, indent=1))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping sun_self.png")
        return
    s = sun_vec(res["combined"]["elevation_deg"], az_c, frame)
    po = predict_object(s)
    _, e1, e2 = frame
    # side view of the object, looking along the horizontal perpendicular to the sun's azimuth
    h_dir = np.cos(np.radians(az_c)) * e1 + np.sin(np.radians(az_c)) * e2
    x, y = o_rec @ h_dir, o_rec @ up + offset
    order = np.argsort(o_rec @ np.cross(up, h_dir))
    fig, axs = plt.subplots(1, 4, figsize=(22, 5.5))
    for axx, m, title in [(axs[0], o_obs, "object: observed shadow (masks)"), (axs[1], po, "object: predicted")]:
        axx.scatter(x[order], y[order], c=np.where(m[order], "#212529", "#ced4da"), s=0.6)
        axx.set_title(title); axx.set_aspect("equal"); axx.set_xlabel("towards the sun (horizontal)"); axx.set_ylabel("height")
    col = np.where(o_obs & po, 0, np.where(o_obs & ~po, 1, np.where(~o_obs & po, 2, 3)))
    axs[2].scatter(x[order], y[order], c=np.array(["#2b8a3e", "#e8590c", "#1971c2", "#dee2e6"])[col[order]], s=0.6)
    axs[2].set_title("agreement: green both, orange missed, blue extra"); axs[2].set_aspect("equal")
    E = [p["elevation"] for p in res["profile_at_combined_azimuth"]]
    for key, colr in [("ground", "#1971c2"), ("object", "#e8590c"), ("combined", "#212529")]:
        axs[3].plot(E, [p[key] for p in res["profile_at_combined_azimuth"]], color=colr, label=key)
    if a.true_elevation:
        axs[3].axvline(a.true_elevation, color="#2b8a3e", ls="--", label=f"true {a.true_elevation}")
    axs[3].set_xlabel("elevation [deg]"); axs[3].set_ylabel("IoU"); axs[3].legend(); axs[3].set_title("IoU vs elevation")
    fig.tight_layout(); fig.savefig(a.out / "sun_self.png", dpi=100)
    print(f"wrote {a.out}/sun_self.json, sun_self.png ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
