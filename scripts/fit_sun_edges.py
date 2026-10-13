#!/usr/bin/env python
"""Sun direction from the hero object's ground shadow, scored on the shadow BOUNDARY instead of area IoU.

Same inputs and prediction as fit_sun.py (orthographic shadow map of the object's clustered Gaussians), but:
  - observed and predicted shadow labels on the ground receivers are rasterised in the ground plane (--raster cell),
    cleaned (median 3x3), and their boundaries extracted;
  - score = symmetric mean chamfer distance between predicted and observed boundaries (scene units, lower is better),
    so the direction information carried by the shadow outline is not diluted by the shadow interior;
  - coarse-to-fine grid (5, 1, 0.25 deg) then a continuous Nelder-Mead refinement from the best grid direction;
  - uncertainty = angular radius of grid directions whose chamfer is within --tie-rel of the best (relative);
  - hold-out: refit on each half of the receivers (split through the footprint perpendicular to the sun azimuth).
Area IoU is reported alongside for comparison.

Usage:
  fit_sun_edges.py --ply exports/IMG_6556/splat.ply --plane-from exports/IMG_6556_shadowfix_v8_o/stats.json \
      --clusters exports/IMG_6556_clusters_chair_r30agree_v3/clusters.npz --crop 0.713 1.091 0.464 1.065 \
      --ground-votes exports/IMG_6556_sun_chair/votes.npz --score-radius 1.2 --out exports/IMG_6556_sun_edges \
      [--true-elevation 55.51]
Writes <out>/sun_edges.json and sun_edges.png. Trains nothing.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from fit_sun import angle_between, plane_frame, predict, sun_vec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--plane-from", type=Path, required=True)
    ap.add_argument("--clusters", type=Path, required=True)
    ap.add_argument("--crop", type=float, nargs=4, metavar=("X", "Y", "Z", "R"), required=True)
    ap.add_argument("--ground-votes", type=Path, required=True)
    ap.add_argument("--score-radius", type=float, default=1.2)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--opaque", type=float, default=0.5)
    ap.add_argument("--occluder-min", type=float, default=0.05)
    ap.add_argument("--min-views", type=int, default=5)
    ap.add_argument("--cell", type=float, default=0.02, help="shadow-map cell (as fit_sun.py)")
    ap.add_argument("--dilate", type=int, default=1)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--raster", type=float, default=0.02, help="ground raster cell for the boundary comparison")
    ap.add_argument("--tie-rel", type=float, default=0.03, help="relative chamfer margin for the uncertainty radius")
    ap.add_argument("--receiver-shift", type=float, default=0.0,
                    help="move ground receivers this far DOWN along the plane normal (scene units) before predicting")
    ap.add_argument("--fit-shift", type=float, nargs=3, metavar=("LO", "HI", "STEP"),
                    help="also fit the receiver shift: scan LO..HI by STEP (scene units), keep the lowest chamfer")
    ap.add_argument("--true-elevation", type=float)
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
    frame = plane_frame(up)
    _, e1, e2 = frame
    labels = np.load(a.clusters)["labels"]
    c, r = np.array(a.crop[:3]), a.crop[3]
    height = xyz @ up + offset
    occ = xyz[(labels >= 0) & (opacity > a.opaque) & (np.linalg.norm(xyz - c, axis=1) < r) & (height > a.occluder_min)]

    gv = np.load(a.ground_votes)
    g_idx, g_sh, g_seen = gv["rec_idx"], gv["shadow"], gv["seen"]
    foot = c - (c @ up + offset) * up
    keep = (g_seen >= a.min_views) & (np.linalg.norm(xyz[g_idx] - foot, axis=1) < a.score_radius)
    rec0 = xyz[g_idx[keep]]
    state = dict(shift=a.receiver_shift)  # receivers sit at rec0 - shift * up (shift > 0: shadow surface lower)
    rec = rec0
    obs = (g_sh[keep] / g_seen[keep]) >= 0.5
    print(f"{len(occ):,} casters (object clusters), {len(rec):,} ground receivers within {a.score_radius} of the "
          f"footprint, {obs.mean() * 100:.1f}% observed in shadow", flush=True)

    # ground raster: receivers -> cells; empty cells take the nearest filled cell's value
    uv = np.c_[(rec - foot) @ e1, (rec - foot) @ e2]
    lo = uv.min(0) - a.raster
    ij = ((uv - lo) / a.raster).astype(int)
    shape = tuple(ij.max(0) + 2)
    cnt = np.zeros(shape)
    np.add.at(cnt, (ij[:, 0], ij[:, 1]), 1)
    filled = cnt > 0
    inside = cv2.dilate(filled.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)  # scoring support
    _, near = distance_transform_edt(~filled, return_indices=True)

    def raster(lbl):
        s = np.zeros(shape)
        np.add.at(s, (ij[:, 0], ij[:, 1]), lbl.astype(float))
        m = np.where(filled, s / np.maximum(cnt, 1) >= 0.5, False)
        m = m[near[0], near[1]].astype(np.uint8)
        return cv2.medianBlur(m, 3).astype(bool)

    def edges(m):
        e = (cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0) & inside
        return e

    obs_r = raster(obs)
    obs_e = edges(obs_r)
    d_obs = distance_transform_edt(~obs_e) * a.raster
    print(f"raster {shape[0]} x {shape[1]} cells of {a.raster}; observed boundary {obs_e.sum():,} cells", flush=True)

    cache = {}

    def evaluate(el, az, sub=None):
        key = (round(el, 4), round(az % 360, 4), None if sub is None else id(sub),  # halves stay alive, ids stable
               round(state["shift"], 5))
        if key not in cache:
            s = sun_vec(el, az, frame)
            pred = predict(occ, rec0 - state["shift"] * up, s, a.cell, a.dilate, a.eps)
            pr = raster(pred)
            pe = edges(pr)
            if sub is not None:
                pe, oe, dob = pe & sub, obs_e & sub, np.where(sub, d_obs, 0)
            else:
                oe, dob = obs_e, d_obs
            if pe.sum() == 0 or oe.sum() == 0:
                cham = 9.9
            else:
                d_pred = distance_transform_edt(~pe) * a.raster
                cham = 0.5 * (dob[pe].mean() + d_pred[oe].mean())
            u = (pred | obs).sum()
            cache[key] = (cham, float((pred & obs).sum() / u) if u else 0.0)
        return cache[key]

    def grid_search(sub=None):
        def grid(els, azs):
            return [(e, z % 360, evaluate(e, z, sub)[0]) for e in els if 1 <= e <= 89 for z in azs]
        coarse = grid(np.arange(5, 86, 5.0), np.arange(0, 360, 5.0))
        e, z, _ = min(coarse, key=lambda t: t[2])
        mid = grid(e + np.arange(-5, 5.1, 1.0), z + np.arange(-5, 5.1, 1.0))
        e, z, _ = min(mid, key=lambda t: t[2])
        fine = grid(e + np.arange(-1.5, 1.51, 0.25), z + np.arange(-1.5, 1.51, 0.25))
        return min(fine, key=lambda t: t[2]), mid + fine

    shift_profile = None
    if a.fit_shift:
        # joint fit of the receiver (grass-top) offset: full grid at the starting shift, then a local search per shift
        # around the previous best; the shift with the lowest boundary chamfer wins (IoU reported, not used)
        (e0, z0, _), _ = grid_search()
        shift_profile = []
        for sh in np.arange(a.fit_shift[0], a.fit_shift[1] + 1e-9, a.fit_shift[2]):
            state["shift"] = float(sh)
            mid = [(e, z % 360, evaluate(e, z)[0]) for e in e0 + np.arange(-3, 3.1, 1.0) if 1 <= e <= 89
                   for z in z0 + np.arange(-3, 3.1, 1.0)]
            e1_, z1_, _ = min(mid, key=lambda t: t[2])
            fine = [(e, z % 360, evaluate(e, z)[0]) for e in e1_ + np.arange(-1, 1.01, 0.25) if 1 <= e <= 89
                    for z in z1_ + np.arange(-1, 1.01, 0.25)]
            be, bz, bc = min(fine, key=lambda t: t[2])
            e0, z0 = be, bz
            shift_profile.append(dict(shift=round(float(sh), 4), elevation=float(be), azimuth=float(bz),
                                      chamfer=round(bc, 5), iou=round(evaluate(be, bz)[1], 4)))
            print(f"  shift {sh:+.3f} (receivers {'down' if sh > 0 else 'up'} {abs(sh) / 1.331 * 100:.1f} cm): "
                  f"elevation {be:.2f}, azimuth {bz:.2f}, chamfer {bc:.4f}, IoU {evaluate(be, bz)[1]:.3f}", flush=True)
        best_sh = min(shift_profile, key=lambda p: p["chamfer"])
        state["shift"] = best_sh["shift"]
        print(f"fitted receiver shift {state['shift']:+.3f} units (lowest chamfer); continuing the fit there", flush=True)

    (ge, gz, gc), samples = grid_search()
    gs = sun_vec(ge, gz, frame)
    tied = [sun_vec(e, z, frame) for e, z, v in samples if v <= gc * (1 + a.tie_rel)]
    unc = max(angle_between(gs, t) for t in tied)
    print(f"grid: elevation {ge:.2f}, azimuth {gz:.2f}, chamfer {gc:.4f} units, IoU {evaluate(ge, gz)[1]:.3f}, "
          f"uncertainty {unc:.2f} deg ({len(tied)} tied) ({time.time() - t0:.0f} s)", flush=True)
    opt = minimize(lambda p: evaluate(p[0], p[1])[0], [ge, gz], method="Nelder-Mead",
                   options=dict(initial_simplex=[[ge, gz], [ge + 0.3, gz], [ge, gz + 0.3]], xatol=0.02, fatol=1e-5,
                                maxiter=150))
    oe_, oz_ = float(opt.x[0]), float(opt.x[1]) % 360
    oc_, oi_ = evaluate(oe_, oz_)
    print(f"Nelder-Mead: elevation {oe_:.2f}, azimuth {oz_:.2f}, chamfer {oc_:.4f}, IoU {oi_:.3f} "
          f"({opt.nfev} evaluations)", flush=True)

    # hold-out halves split through the footprint, perpendicular to the sun's horizontal direction
    hdir = np.cos(np.radians(gz)) * e1 + np.sin(np.radians(gz)) * e2
    perp = np.cross(up, hdir)
    gi, gj = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    cu, cv_ = lo[0] + (gi + 0.5) * a.raster, lo[1] + (gj + 0.5) * a.raster
    side = (cu * (perp @ e1) + cv_ * (perp @ e2)) > 0
    halves = []
    for sub in (side, ~side):
        (he, hz, hc), _ = grid_search(sub)
        halves.append(dict(elevation=float(he), azimuth=float(hz), chamfer=round(hc, 4),
                           deg_from_full=round(angle_between(sun_vec(he, hz, frame), gs), 2)))
    hd = angle_between(sun_vec(halves[0]["elevation"], halves[0]["azimuth"], frame),
                       sun_vec(halves[1]["elevation"], halves[1]["azimuth"], frame))
    print(f"hold-out: halves at elevation {halves[0]['elevation']:.2f} / {halves[1]['elevation']:.2f}, "
          f"{hd:.2f} deg apart", flush=True)
    if a.true_elevation:
        print(f"error vs true {a.true_elevation}: grid {ge - a.true_elevation:+.2f}, Nelder-Mead {oe_ - a.true_elevation:+.2f}")

    prof = [(e, *evaluate(e, gz)) for e in np.arange(45, 66.1, 0.5)]
    res = dict(grid=dict(elevation_deg=float(ge), azimuth_deg=float(gz), chamfer=round(gc, 5),
                         iou=round(evaluate(ge, gz)[1], 4), uncertainty_deg=round(unc, 2), tied=len(tied)),
               nelder_mead=dict(elevation_deg=round(oe_, 3), azimuth_deg=round(oz_, 3), chamfer=round(oc_, 5),
                                iou=round(oi_, 4), evaluations=int(opt.nfev)),
               holdout=halves, holdout_apart_deg=round(hd, 2), true_elevation=a.true_elevation,
               receiver_shift=state["shift"], shift_profile=shift_profile,
               profile_at_grid_azimuth=[dict(elevation=float(e), chamfer=round(cc, 5), iou=round(ii, 4)) for e, cc, ii in prof],
               casters=int(len(occ)), receivers=int(len(rec)), seconds=round(time.time() - t0, 1),
               params={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()})
    (a.out / "sun_edges.json").write_text(json.dumps(res, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pred_r = raster(predict(occ, rec, sun_vec(oe_, oz_, frame), a.cell, a.dilate, a.eps))
    pe = edges(pred_r)
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    img = np.full(shape + (3,), 0.92)
    img[obs_r] = 0.6
    img[obs_e] = [0.9, 0.35, 0.05]
    img[pe] = [0.1, 0.45, 0.85]
    img[obs_e & pe] = [0.15, 0.55, 0.25]
    img[~inside] = 1.0
    axs[0].imshow(img.transpose(1, 0, 2), origin="lower")
    axs[0].set_title("boundaries: orange observed, blue predicted (Nelder-Mead sun), green both")
    E = [p["elevation"] for p in res["profile_at_grid_azimuth"]]
    axs[1].plot(E, [p["chamfer"] for p in res["profile_at_grid_azimuth"]], "k-")
    axs[1].set_ylabel("boundary chamfer [units] (lower better)"); axs[1].set_xlabel("elevation [deg]")
    axs[2].plot(E, [p["iou"] for p in res["profile_at_grid_azimuth"]], "b-")
    axs[2].set_ylabel("area IoU (higher better)"); axs[2].set_xlabel("elevation [deg]")
    for axx in axs[1:]:
        axx.axvline(oe_, color="k", ls=":", label=f"fit {oe_:.2f}")
        if a.true_elevation:
            axx.axvline(a.true_elevation, color="g", ls="--", label=f"true {a.true_elevation}")
        axx.legend()
    fig.tight_layout(); fig.savefig(a.out / "sun_edges.png", dpi=100)
    print(f"wrote {a.out}/sun_edges.json, sun_edges.png ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
