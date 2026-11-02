#!/usr/bin/env python
"""Bear relight: split each object blob into paint colour x light, then light it again with spherical-harmonic (SH)
lighting that has no shadows.

Light, per colour channel, in units where the sky gives irradiance 1 on an upward surface:
- sky: uniform radiance over the upper hemisphere (around the ground-plane normal);
- ground bounce: uniform radiance over the lower hemisphere, beta = ratio_c x --ground-albedo times the sky's
  (the sunlit concrete is ratio_c times brighter than the shaded concrete);
- sun: irradiance k_c at normal incidence along the measured sun direction, k_c = (ratio_c - 1) / (up . s),
  ratio_c = lit / shaded concrete brightness per channel in the ring beside the object (photo-mask votes).
All three are projected onto 9 SH coefficients (order 2), and irradiance at a surface direction n uses the
Ramamoorthi-Hanrahan cosine-lobe factors. Order 2 is near exact for the sky and bounce and gives a soft
terminator for the sun.

Decomposition per blob (n = 2DGS borrowed direction; unsure blobs take their confident neighbours' mean):
  observed L = paint x (D(n) + v x S(n)),   D = sky + bounce SH irradiance, S = sun SH irradiance, v in [0, 1]
v (how much sun reaches the blob: per spot, so soft and partial shadows get their own strength) comes from the
paint level of sunlit neighbours of the same colour (the user's adjacent brightness-and-colour rule, radius in
object units), smoothed over same-facing neighbours; blobs with v >= --lit-v keep v = 1.
Relight: new = L x (D_t(n) + S_t(n)) / (D(n) + v S(n)), per channel, with the target light t = the measured light
(--target same) or the same sun turned down to --soft-sun of its strength (--target soft). With --target same a
sunlit blob is unchanged exactly and a sun-averted one too; there is no visibility term, so no cast shadows.
With --cast-from the sunlit pool is limited to sun-facing blobs the stage-4 shadow map says are lit (and those stay
unchanged); without it shadowed sun-facing blobs pull the lit level down and lit blobs get lifted too. With
--lift-averted the sun-averted side is lit as if it faced the sun like a typical sunlit blob (attached shade removed).
Cross-check (user): a relit blob may not end up brighter, per unit of target light, than the median of its sunlit
same-colour neighbours. Only the object's colours change; the output keeps every blob of --ply (matched by position).

Usage: relight_sh.py --ply exports/IMG_6705_clone_y_geom/splat_clone.ply --orig exports/IMG_6705/splat.ply
           --borrow exports/IMG_6705_2dgs_borrow/borrowed.npz --sun-from exports/IMG_6705_shadow4/shadow_flags.npz
           --object exports/IMG_6705_object/object.json --plane-from exports/IMG_6705_shadowfix_o/stats.json
           --votes exports/IMG_6705_sun/votes.npz --out exports/IMG_6705_relight_sh
Writes <out>/splat_relit.ply, sun_visibility.ply (object: yellow = sun reaches, blue = blocked), stats.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from fill_geometry import LUM, SH_C0, chroma, colours, points
from fill_shadows import linear_to_srgb, srgb_to_linear
from fix_splat_shadows import read_ply, write_ply

A_L = np.array([np.pi] + [2 * np.pi / 3] * 3 + [np.pi / 4] * 5)  # cosine-lobe factors per SH band


def sh9(d):
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    return np.stack([0.282095 * np.ones_like(x), 0.488603 * y, 0.488603 * z, 0.488603 * x, 1.092548 * x * y,
                     1.092548 * y * z, 0.315392 * (3 * z * z - 1), 1.092548 * x * z, 0.546274 * (x * x - y * y)], -1)


def sphere(n=20000):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(1 - z * z)
    phi = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[r * np.cos(phi), r * np.sin(phi), z]


def env_sh(up, s, sky, bounce, sun):
    """9 x 3 SH radiance coefficients: sky radiance sky/pi above, bounce/pi below, a sun of irradiance `sun`."""
    d = sphere()
    rad = np.where((d @ up > 0)[:, None], sky / np.pi, bounce / np.pi)  # [n, 3]
    L = sh9(d).T @ rad * (4 * np.pi / len(d))
    return L + np.outer(sh9(s), sun)


def irradiance(L, n):
    return np.maximum(sh9(n) @ (A_L[:, None] * L), 0)


def crf_display(lin, crf_json):
    """Linear radiance -> display colour through PPISP's learned camera: mean exposure, then the per-channel
    toe-shoulder curve (formula from ppisp_math.cuh); vignetting and per-frame colour correction are left out."""
    d = json.loads(Path(crf_json).read_text())
    e = np.mean(d["exposure"]["per_frame"])
    x = np.clip(lin * 2.0 ** e, 0, 1)
    out = np.empty_like(x)
    cam = next(iter(d["crf"]["by_camera"].values()))
    for c, ch in enumerate(("red", "green", "blue")):
        q = cam[ch]["effective"]
        toe, sh, g, cen = q["toe"], q["shoulder"], q["gamma"], q["center"]
        aa = sh * cen / (toe + (sh - toe) * cen)
        y = np.where(x[:, c] <= cen, aa * (x[:, c] / cen) ** toe,
                     1 - (1 - aa) * (np.clip(1 - x[:, c], 0, None) / (1 - cen)) ** sh)
        out[:, c] = np.maximum(y, 0) ** g
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", type=Path, required=True, help="splat to write the relit object into (e.g. the new final)")
    ap.add_argument("--orig", type=Path, required=True, help="original splat (observed colours, ground brightness)")
    ap.add_argument("--borrow", type=Path, required=True)
    ap.add_argument("--sun-from", type=Path, required=True)
    ap.add_argument("--object", type=Path, required=True)
    ap.add_argument("--plane-from", type=Path, required=True)
    ap.add_argument("--votes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ring", type=float, nargs=2, default=[0.6, 2.0])
    ap.add_argument("--ground-albedo", type=float, default=0.3, help="concrete albedo, sets the bounce light")
    ap.add_argument("--radius", type=float, default=0.06, help="paint neighbourhood, x object height")
    ap.add_argument("--chroma", type=float, default=0.15)
    ap.add_argument("--sunlit-cos", type=float, default=0.3)
    ap.add_argument("--lit-v", type=float, default=0.85, help="v at or above this counts as fully sunlit (unchanged)")
    ap.add_argument("--smooth-k", type=int, default=8)
    ap.add_argument("--smooth-iters", type=int, default=2)
    ap.add_argument("--max-gain", type=float, default=6.0)
    ap.add_argument("--target", choices=["same", "soft"], default="same")
    ap.add_argument("--soft-sun", type=float, default=0.3)
    ap.add_argument("--cast-from", type=Path, help="shadow_flags.npz: sunlit pool = sun-facing blobs the map says are lit")
    ap.add_argument("--ref-pct", type=float, default=50.0, help="percentile of the sunlit pool used as the lit paint level")
    ap.add_argument("--lift-averted", action="store_true",
                    help="also light the sun-averted side as if it faced the sun (removes attached shade)")
    ap.add_argument("--averted-cos", type=float, default=0.05)
    ap.add_argument("--linear", action="store_true", help="--orig colours are linear (PPISP splat, same blob order)")
    ap.add_argument("--gain-mode", choices=["channel", "lum"], default="channel",
                    help="lum: one brightness factor per blob + a global shade-to-sun colour shift")
    ap.add_argument("--match-ply", type=Path, help="splat --orig was fine-tuned from (same blob order): positions for matching")
    ap.add_argument("--crf-json", type=Path, help="ppisp_params.json: bake the relit colours through the learned camera")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    header, data = read_ply(a.ply)
    oh, orig = read_ply(a.orig)
    oxyz = np.c_[orig["x"], orig["y"], orig["z"]].astype(float)
    obj = json.loads(a.object.read_text())
    H = obj["height"]
    gp = json.loads(a.plane_from.read_text())["ground_plane"]
    up = np.array(gp["normal"], float)
    up /= np.linalg.norm(up)
    s = np.load(a.sun_from)["sun"].astype(float)
    s /= np.linalg.norm(s)

    # light per channel from the concrete beside the object
    v = np.load(a.votes)
    _, _, o_lin = colours(orig)
    if a.linear:  # PPISP splat: base colours are already linear radiance
        o_lin = np.clip(np.stack([orig[f"f_dc_{c}"] for c in range(3)], 1) * SH_C0 + 0.5, 0, None)
    r = np.linalg.norm(oxyz[v["rec_idx"]] - np.array(obj["footprint_centre"]), axis=1) / H
    ok = (v["seen"] >= 5) & (r > a.ring[0]) & (r < a.ring[1])
    share = v["shadow"] / np.maximum(v["seen"], 1)
    sh_c = np.median(o_lin[v["rec_idx"][ok & (share > 0.7)]], 0)
    lit_c = np.median(o_lin[v["rec_idx"][ok & (share < 0.1)]], 0)
    ratio = lit_c / sh_c
    k = (ratio - 1) / (up @ s)
    beta = ratio * a.ground_albedo
    L_diff = env_sh(up, s, np.ones(3), beta, np.zeros(3))
    L_sun = env_sh(up, s, np.zeros(3), np.zeros(3), k)
    tgt_sun = k * (1.0 if a.target == "same" else a.soft_sun)
    L_tsun = env_sh(up, s, np.zeros(3), np.zeros(3), tgt_sun)
    print(f"concrete shaded {sh_c.round(4)} vs sunlit {lit_c.round(4)}: ratio {ratio.round(2)}; sun/sky k {k.round(2)}; "
          f"bounce {beta.round(2)} x sky (ground albedo {a.ground_albedo}); target {a.target}")

    # object blobs, directions
    b = np.load(a.borrow)
    idx = b["index"][b["on_object"]]
    n = b["normal"][b["on_object"]].astype(float)
    conf = b["confident"][b["on_object"]]
    p = oxyz[idx]
    tree = cKDTree(p)
    ci = np.flatnonzero(conf)
    _, cnb = cKDTree(p[ci]).query(p[~conf], k=a.smooth_k)
    m = n[ci][cnb]
    m = (m * np.sign(np.einsum("ikj,ij->ik", m, m[:, 0]))[..., None]).sum(1)
    n[~conf] = m / np.linalg.norm(m, axis=1, keepdims=True)
    D, S = irradiance(L_diff, n), irradiance(L_sun, n)  # [N, 3]
    Dt, St = D, irradiance(L_tsun, n)
    ns = n @ s

    # sun visibility per blob from sunlit same-colour neighbours; the sunlit pool is the blobs that face the sun AND
    # that the stage-4 shadow map says the sun reaches (shadowed sun-facing blobs must not set the lit level)
    rgb = o_lin[idx]
    Ll = rgb @ LUM
    ch = chroma(rgb)
    Dl, Sl = D @ LUM, S @ LUM
    paint_if_lit = Ll / (Dl + Sl)
    sunlit = ns > a.sunlit_cos
    if a.cast_from:
        f = np.load(a.cast_from)
        _, jf = cKDTree(oxyz[f["object_idx"]]).query(p)
        cast = f["cast_object"][jf]
        sunlit &= ~cast
        print(f"shadow map: {cast.mean():.1%} of the object cast-shadowed; sunlit pool {sunlit.sum():,} blobs")
    sl = np.flatnonzero(sunlit)
    sl_tree = cKDTree(p[sl])
    ref = np.full(len(p), np.nan)
    for i, nb in enumerate(sl_tree.query_ball_point(p, a.radius * H)):
        cand = sl[np.asarray(nb, int)]
        cand = cand[np.abs(ch[cand] - ch[i]).sum(1) < a.chroma]
        if len(cand) < 3:
            cand = sl[np.asarray(sl_tree.query_ball_point(p[i], 2 * a.radius * H), int)]
            cand = cand[np.abs(ch[cand] - ch[i]).sum(1) < a.chroma]
        if len(cand) >= 3:
            ref[i] = np.percentile(paint_if_lit[cand], a.ref_pct)
    has = np.isfinite(ref) & (ref > 1e-6) & (Sl > 1e-3)
    vis = np.ones(len(p))
    vis[has] = np.clip((Ll[has] / ref[has] - Dl[has]) / Sl[has], 0, 1)
    lit0 = (vis >= a.lit_v) | sunlit  # blobs in the sunlit pool are never changed
    _, nb = tree.query(p, k=a.smooth_k + 1)
    same_side = np.einsum("ij,ikj->ik", n, n[nb]) > 0.7
    same_side &= ~lit0[nb]  # smooth only over other not-fully-lit blobs, so lit neighbours do not pull v up
    same_side[:, 0] = True  # nb[:, 0] is the blob itself
    for _ in range(a.smooth_iters):
        vis = np.where(lit0, 1.0, (vis[nb] * same_side).sum(1) / np.maximum(same_side.sum(1), 1))
    vis[lit0] = 1.0

    gain = (Dt + St) / np.maximum(D + vis[:, None] * S, 1e-6)  # per channel
    averted = ns <= a.averted_cos
    if a.lift_averted:
        # sun-averted side: light it as if it faced the sun like a typical sunlit blob (no attached shade either)
        S_ref = irradiance(L_tsun, np.median(n[sl], 0, keepdims=True) / np.linalg.norm(np.median(n[sl], 0)))
        gain[averted] = (Dt[averted] + S_ref) / np.maximum(D[averted], 1e-6)
        ref_av = ref.copy()
        ref_av[~np.isfinite(ref_av)] = np.nanmedian(ref)  # no sunlit neighbour: the object's median paint level
        ref = np.where(averted, ref_av, ref)
        has = has | averted
    if a.target == "same":
        gain[lit0 & ~(a.lift_averted & averted)] = 1.0
    gain = np.clip(gain, 0, a.max_gain)
    if a.gain_mode == "lum":
        # one brightness factor per blob (keeps its own colour, so dark noisy colours are not amplified per channel),
        # plus one global shade-to-sun colour shift from the concrete, applied in proportion to the light added
        g = np.clip((gain * (D + vis[:, None] * S)) @ LUM / np.maximum((D + vis[:, None] * S) @ LUM, 1e-6), 1, a.max_gain)
        wb = (lit_c / (lit_c @ LUM)) / (sh_c / (sh_c @ LUM))
        added = 1 - 1 / g
        gain = g[:, None] * (1 + added[:, None] * (wb - 1))
        print(f"brightness-only gains; shade-to-sun colour shift {wb.round(3)} applied in proportion to added light")
    new_lin = np.clip(rgb * gain, 0, 1)
    # cross-check with sunlit same-colour neighbours (luminance, per unit of target light)
    tgt_l = ((Dt + St) @ LUM).copy()
    if a.lift_averted:
        tgt_l[averted] = ((Dt[averted] + S_ref) @ LUM)
    cap = np.where(has, ref * tgt_l, np.inf)
    over = ((new_lin @ LUM) > cap) & ((new_lin @ LUM) > Ll)  # only trims a lift, never darkens a blob
    new_lin[over] *= (np.maximum(cap[over], Ll[over]) / np.maximum(new_lin[over] @ LUM, 1e-6))[:, None]
    g_l = (new_lin @ LUM) / np.maximum(Ll, 1e-6)

    # write into --ply, matched by position
    xyz = np.c_[data["x"], data["y"], data["z"]].astype(float)
    pm = p
    if a.match_ply:  # --orig was fine-tuned (PPISP): its blobs moved slightly, so match with the source positions
        _, mp = read_ply(a.match_ply)
        pm = np.c_[mp["x"], mp["y"], mp["z"]].astype(float)[idx]
    dd, j = cKDTree(xyz).query(pm)
    hit = dd < 1e-6
    names, SH, _ = colours(data)  # view-dependent part from --ply itself (display colour space)
    new = data.copy()
    if a.crf_json:  # apply the change as a display-space ratio, so untouched blobs keep --ply's colour exactly
        base = np.clip(np.stack([data[f"f_dc_{c}"][j] for c in range(3)], 1) * SH_C0 + 0.5, 0, 1)
        ratio = crf_display(new_lin, a.crf_json) / np.maximum(crf_display(rgb, a.crf_json), 1e-4)
        disp = np.clip(base * ratio, 0, 1)
        # cross-check again in display colours: a lifted blob may not end up brighter than the median of the sunlit
        # pool blobs of the same colour near it (as shown), which also catches over-lift on the sun-averted side
        b_l = base @ LUM
        d_l = disp @ LUM
        cap_d = np.full(len(p), np.inf)
        for i in np.flatnonzero(d_l > b_l + 1e-4):
            cand = sl[np.asarray(sl_tree.query_ball_point(p[i], 2 * a.radius * H), int)]
            cand = cand[np.abs(ch[cand] - ch[i]).sum(1) < a.chroma]
            if len(cand) >= 3:
                cap_d[i] = np.median(b_l[cand])
        trim = d_l > np.maximum(cap_d, b_l)
        disp[trim] *= (np.maximum(cap_d[trim], b_l[trim]) / np.maximum(d_l[trim], 1e-6))[:, None]
        print(f"display cross-check trimmed {trim.sum():,} lifted blobs")
    else:
        disp = linear_to_srgb(new_lin)
    dc = (disp - 0.5) / SH_C0
    for c, nm in enumerate(names):
        col = dc[:, c] if c < 3 else SH[j, c] * np.minimum(g_l, 3)
        new[nm][j[hit]] = col[hit].astype(np.float32)
    write_ply(a.out / "splat_relit.ply", header, new)
    vis_rgb = np.c_[vis, vis, 1 - vis] * np.array([1.0, 0.85, 1.0])
    points(a.out / "sun_visibility.ply", oh, orig[idx], srgb_to_linear(np.clip(vis_rgb, 0, 1)))

    changed = np.abs(g_l - 1) > 0.02
    st = {"concrete_ratio_rgb": ratio.tolist(), "sun_over_sky_rgb": k.tolist(), "bounce_over_sky_rgb": beta.tolist(),
          "target": a.target, "object_blobs": int(len(p)), "matched_in_ply": int(hit.sum()),
          "sun_facing_share": float((ns > 0).mean()), "with_sunlit_reference": float(has.mean()),
          "sun_blocked_share_v_below_0.5": float((vis < 0.5).mean()), "partly_lit_share": float(((vis >= 0.5) & (vis < 1)).mean()),
          "changed_share": float(changed.mean()), "gain_median_changed": float(np.median(g_l[changed])) if changed.any() else 1.0,
          "gain_p90_changed": float(np.percentile(g_l[changed], 90)) if changed.any() else 1.0,
          "capped_by_crosscheck": int(over.sum()), "args": {k_: str(v_) for k_, v_ in vars(a).items()}}
    (a.out / "stats.json").write_text(json.dumps(st, indent=1))
    print(f"object {len(p):,} blobs ({hit.sum():,} matched in --ply); sun blocked (v < 0.5) {st['sun_blocked_share_v_below_0.5']:.1%}, "
          f"partly {st['partly_lit_share']:.1%}; changed {st['changed_share']:.1%}, gain median {st['gain_median_changed']:.2f} "
          f"(p90 {st['gain_p90_changed']:.2f}); capped by the sunlit cross-check {over.sum():,} -> {a.out / 'splat_relit.ply'}")


if __name__ == "__main__":
    main()
