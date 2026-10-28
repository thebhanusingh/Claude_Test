#!/usr/bin/env python
"""Train a 2DGS (flat surfel) splat of an existing splatfacto scene, used only as a surface-direction reference.

The scene's own nerfstudio config supplies the frames, the held-out split (train_split_fraction 0.9), the
undistortion and the coordinate frame, so the surfels line up with exports/<scene>/splat.ply (ns-export writes
model.means unchanged) and the held-out scores compare directly with ns-eval on the splatfacto run. No COLMAP rerun.
Training cameras use the splatfacto run's refined poses (SO3xR3 adjustment from its checkpoint) unless
--raw-poses; held-out cameras use the COLMAP poses, as in ns-eval.

Training follows gsplat's 2DGS example: L1 + 0.2 (1 - SSIM); depth-distortion loss from --dist-start; normal
consistency (rendered surfel normal vs normal of the rendered depth) from --normal-start; DefaultStrategy
densification on the 2DGS gradient until --refine-stop.

At the end:
- held-out PSNR / SSIM / LPIPS with splatfacto's metric functions (pytorch_msssim SSIM, LPIPS alex), photo | render
  jpgs in <out>/eval/, numbers in <out>/metrics.json;
- surfel normals (each surfel's flat axis), turned to face out by a vote of training cameras that see the surfel
  (its depth within --vis-tol of the rendered median depth); <out>/surfels.npz and <out>/surfels_normals.ply.

Usage: train_2dgs_ref.py --config outputs/IMG_6705/splatfacto/2026-10-27_194441/config.yml --out exports/IMG_6705_2dgs
Run from the repo root (the config's data path is relative). --steps 500 is a quick smoke test.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from gsplat import rasterization_2dgs
from gsplat.strategy import DefaultStrategy
from PIL import Image
from pytorch_msssim import SSIM, ssim
from scipy.spatial import cKDTree
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

C0 = 0.28209479177387814  # SH band-0 constant


def viewmat(c2w34):
    """nerfstudio camera-to-world (OpenGL axes) -> gsplat world-to-camera (OpenCV axes)."""
    c2w = torch.eye(4, device=c2w34.device)
    c2w[:3, :4] = c2w34
    c2w[:3, 1:3] *= -1
    return torch.linalg.inv(c2w)


def intrinsics(cams, i):
    K = torch.eye(3)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = cams.fx[i, 0], cams.fy[i, 0], cams.cx[i, 0], cams.cy[i, 0]
    return K, int(cams.width[i, 0]), int(cams.height[i, 0])


def refined_c2w(config_path, cams):
    """Apply the splatfacto run's SO3xR3 pose adjustments to the training cameras (None if not found)."""
    from nerfstudio.cameras.lie_groups import exp_map_SO3xR3
    ckpts = sorted((config_path.parent / "nerfstudio_models").glob("step-*.ckpt"))
    if not ckpts:
        return None
    sd = torch.load(ckpts[-1], map_location="cpu", weights_only=False)["pipeline"]
    key = next((k for k in sd if k.endswith("camera_optimizer.pose_adjustment")), None)
    if key is None or sd[key].shape[0] != len(cams):
        return None
    adj = exp_map_SO3xR3(sd[key].float())  # [N, 3, 4]
    adj = torch.cat([adj, torch.tensor([0, 0, 0, 1.0]).expand(len(adj), 1, 4)], 1)
    c2w = torch.cat([cams.camera_to_worlds, torch.tensor([0, 0, 0, 1.0]).expand(len(adj), 1, 4)], 1)
    print(f"refined poses from {ckpts[-1].name}: mean shift {sd[key][:, :3].norm(dim=1).mean():.4f} units")
    return torch.bmm(c2w, adj)[:, :3]


def render(p, vm, K, W, H, deg, distloss=False):
    return rasterization_2dgs(
        means=p["means"], quats=p["quats"], scales=torch.exp(p["scales"]), opacities=torch.sigmoid(p["opacities"]),
        colors=torch.cat([p["sh0"], p["shN"]], 1), viewmats=vm[None], Ks=K[None], width=W, height=H,
        sh_degree=deg, render_mode="RGB+ED", distloss=distloss, packed=False, absgrad=False)


def surfel_normals(quats):
    q = torch.nn.functional.normalize(quats, dim=1)
    w, x, y, z = q.unbind(1)
    return torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], 1)  # R[:, :, 2]


def flatten_splat(path):
    """Load an ns-export splat.ply as 2DGS parameters: each blob's shortest axis becomes the surfel normal (local z),
    its two longer axes the surfel's x and y."""
    from fix_splat_shadows import read_ply
    from scipy.spatial.transform import Rotation
    _, g = read_ply(path)
    col = lambda *k: np.stack([g[x] for x in k], 1).astype(np.float32)
    n = len(g)
    q = col("rot_0", "rot_1", "rot_2", "rot_3")
    R = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()  # scipy wants xyzw
    s = col("scale_0", "scale_1", "scale_2")
    order = np.argsort(-s, axis=1)  # longest, middle, shortest
    R = np.take_along_axis(R, order[:, None, :], axis=2)
    R[np.linalg.det(R) < 0, :, 2] *= -1  # keep it a rotation
    q = Rotation.from_matrix(R).as_quat()[:, [3, 0, 1, 2]]
    s = np.take_along_axis(s, order, axis=1)
    rest = [k for k in g.dtype.names if k.startswith("f_rest_")]
    shn = col(*sorted(rest, key=lambda k: int(k[7:]))).reshape(n, 3, -1).transpose(0, 2, 1) if rest else \
        np.zeros((n, 15, 3), np.float32)
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    return {"means": t(col("x", "y", "z")), "scales": t(s), "quats": t(q), "opacities": t(g["opacity"]),
            "sh0": t(col("f_dc_0", "f_dc_1", "f_dc_2"))[:, None], "shN": t(shn)}


def write_normal_ply(path, xyz, n):
    rgb = ((n * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    v = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
                                  ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for k, a in zip(("x", "y", "z", "nx", "ny", "nz"), np.concatenate([xyz, n], 1).T):
        v[k] = a
    v["red"], v["green"], v["blue"] = rgb.T
    props = "".join(f"property float {k}\n" for k in ("x", "y", "z", "nx", "ny", "nz"))
    props += "".join(f"property uchar {k}\n" for k in ("red", "green", "blue"))
    with open(path, "wb") as f:
        f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\n{props}end_header\n".encode())
        f.write(v.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True, help="splatfacto run config.yml")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--refine-stop", type=int, default=15000, help="last densification step")
    ap.add_argument("--dist-start", type=int, default=3000)
    ap.add_argument("--dist-lambda", type=float, default=0.01)
    ap.add_argument("--normal-start", type=int, default=7000)
    ap.add_argument("--normal-lambda", type=float, default=0.05)
    ap.add_argument("--raw-poses", action="store_true", help="train on the COLMAP poses, not the refined ones")
    ap.add_argument("--init-ply", type=Path, help="start from this trained splat (blobs flattened to surfels) instead "
                    "of the COLMAP points; e.g. --steps 10000 --dist-start 0 --normal-start 1000 --refine-stop 0")
    ap.add_argument("--vote-cams", type=int, default=60, help="training cameras used to orient the normals")
    ap.add_argument("--vis-tol", type=float, default=0.01, help="visible if |depth - rendered depth| < tol x depth")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "eval").mkdir(exist_ok=True)
    dev = "cuda"
    t0 = time.time()

    config = yaml.load(a.config.read_text(), Loader=yaml.Loader)
    dm = config.pipeline.datamanager.setup(device=dev, test_mode="val", world_size=1, local_rank=0)
    train, held = dm.cached_train, dm.cached_eval  # loading also undistorts and updates the cameras' K
    tc, ec = dm.train_dataset.cameras, dm.eval_dataset.cameras
    print(f"{len(train)} training / {len(held)} held-out frames, {int(tc.width[0, 0])} x {int(tc.height[0, 0])}"
          f" ({time.time() - t0:.0f} s)")
    c2w = None if a.raw_poses else refined_c2w(a.config, tc)
    if c2w is None:
        print("using COLMAP poses for training")
        c2w = tc.camera_to_worlds
    tvm = [viewmat(c.to(dev)) for c in c2w]
    evm = [viewmat(c.to(dev)) for c in ec.camera_to_worlds]
    centers = c2w[:, :3, 3]
    scene_scale = float((centers - centers.mean(0)).norm(dim=1).max()) * 1.1

    if a.init_ply:
        p = torch.nn.ParameterDict(flatten_splat(a.init_ply)).to(dev)
        print(f"init: {len(p['means']):,} blobs of {a.init_ply}, flattened; scene scale {scene_scale:.3f}")
    else:
        md = dm.train_dataparser_outputs.metadata
        xyz = md["points3D_xyz"].float()
        rgb = md["points3D_rgb"].float() / 255
        d, _ = cKDTree(xyz.numpy()).query(xyz.numpy(), k=4)
        n = len(xyz)
        print(f"init: {n:,} COLMAP points, scene scale {scene_scale:.3f}")
        p = torch.nn.ParameterDict({
            "means": xyz,
            "scales": torch.log(torch.from_numpy(d[:, 1:].mean(1)).float().clamp_min(1e-7))[:, None].repeat(1, 3),
            "quats": torch.nn.functional.normalize(torch.rand(n, 4), dim=1),
            "opacities": torch.logit(torch.full((n,), 0.1)),
            "sh0": ((rgb - 0.5) / C0)[:, None],
            "shN": torch.zeros(n, 15, 3),
        }).to(dev)
    sh_start = 3000 if a.init_ply else 0  # a trained splat already has all SH bands
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3,
           "shN": 2.5e-3 / 20}
    opt = {k: torch.optim.Adam([{"params": p[k], "lr": lr, "name": k}], eps=1e-15) for k, lr in lrs.items()}
    sched = torch.optim.lr_scheduler.ExponentialLR(opt["means"], gamma=0.01 ** (1.0 / a.steps))
    strategy = DefaultStrategy(key_for_gradient="gradient_2dgs", refine_stop_iter=a.refine_stop, absgrad=False,
                               verbose=False)
    strategy.check_sanity(p, opt)
    state = strategy.initialize_state(scene_scale=scene_scale)
    Ks = [intrinsics(tc, i) for i in range(len(train))]

    order = []
    for step in range(a.steps):
        if not order:
            order = torch.randperm(len(train)).tolist()
        i = order.pop()
        K, W, H = Ks[i]
        gt = train[i]["image"].to(dev).float()
        gt = gt / 255 if train[i]["image"].dtype == torch.uint8 else gt
        rc, ra, rn, rnd, rdist, _, info = render(p, tvm[i], K.to(dev), W, H, min((step + sh_start) // 1000, 3),
                                                 distloss=step >= a.dist_start)
        info.setdefault("width", W), info.setdefault("height", H), info.setdefault("n_cameras", 1)
        pred = rc[0, ..., :3]
        l1 = (pred - gt).abs().mean()
        ss = ssim(pred.permute(2, 0, 1)[None], gt[..., :3].permute(2, 0, 1)[None], data_range=1.0)
        loss = 0.8 * l1 + 0.2 * (1 - ss)
        if step >= a.dist_start:
            loss = loss + a.dist_lambda * rdist.mean()
        if step >= a.normal_start:
            loss = loss + a.normal_lambda * (1 - (rn * (rnd * ra.detach())).sum(-1)).mean()
        strategy.step_pre_backward(p, opt, state, step, info)
        loss.backward()
        for o in opt.values():
            o.step()
            o.zero_grad(set_to_none=True)
        sched.step()
        strategy.step_post_backward(p, opt, state, step, info, packed=False)
        if step % 1000 == 0 or step == a.steps - 1:
            print(f"step {step:6d}  loss {loss.item():.4f}  l1 {l1.item():.4f}  surfels {len(p['means']):,}  "
                  f"GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB  {time.time() - t0:.0f} s", flush=True)
    t_train = time.time() - t0
    torch.save({k: v.detach().cpu() for k, v in p.items()}, a.out / "surfels_params.pt")

    # held-out scores, same metric functions as splatfacto
    psnr_f, ssim_f = PeakSignalNoiseRatio(data_range=1.0), SSIM(data_range=1.0, size_average=True, channel=3)
    lpips_f = LearnedPerceptualImagePatchSimilarity(normalize=True).to(dev)
    rows = []
    with torch.no_grad():
        for j, data in enumerate(held):
            K, W, H = intrinsics(ec, j)
            gt = data["image"].to(dev).float()
            gt = (gt / 255 if data["image"].dtype == torch.uint8 else gt)[..., :3]
            pred = render(p, evm[j], K.to(dev), W, H, 3)[0][0, ..., :3].clamp(0, 1)
            g, r = gt.permute(2, 0, 1)[None], pred.permute(2, 0, 1)[None]
            rows.append({"frame": j, "psnr": float(psnr_f(g.cpu(), r.cpu())), "ssim": float(ssim_f(g, r)),
                         "lpips": float(lpips_f(g, r))})
            Image.fromarray((torch.cat([gt, pred], 1).cpu().numpy() * 255).astype(np.uint8)).save(
                a.out / "eval" / f"eval_{j:03d}.jpg", quality=92)
    m = {k: float(np.mean([r[k] for r in rows])) for k in ("psnr", "ssim", "lpips")}
    print(f"held-out ({len(rows)} frames): PSNR {m['psnr']:.2f}  SSIM {m['ssim']:.4f}  LPIPS {m['lpips']:.4f}")

    # surfel normals, turned to face the cameras that see them
    with torch.no_grad():
        means, nrm = p["means"].detach(), surfel_normals(p["quats"].detach())
        votes = torch.zeros(len(means), device=dev)
        seen = torch.zeros(len(means), device=dev)
        for i in np.linspace(0, len(train) - 1, min(a.vote_cams, len(train))).astype(int):
            K, W, H = Ks[i]
            med = render(p, tvm[i], K.to(dev), W, H, 0)[5][0, ..., 0]
            pc = means @ tvm[i][:3, :3].T + tvm[i][:3, 3]
            z = pc[:, 2]
            u = (K[0, 0] * pc[:, 0] / z + K[0, 2]).round().long()
            v = (K[1, 1] * pc[:, 1] / z + K[1, 2]).round().long()
            ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            vis = torch.zeros_like(ok)
            vis[ok] = (z[ok] - med[v[ok], u[ok]]).abs() < a.vis_tol * z[ok]
            cam = torch.linalg.inv(tvm[i])[:3, 3]
            votes += vis * torch.sign(((cam - means) * nrm).sum(1))
            seen += vis
        nrm = nrm * torch.where(votes < 0, -1.0, 1.0)[:, None]
    np.savez_compressed(
        a.out / "surfels.npz", means=means.cpu().numpy(), normals=nrm.cpu().numpy(),
        scales=torch.exp(p["scales"]).detach().cpu().numpy(), opacity=torch.sigmoid(p["opacities"]).detach().cpu().numpy(),
        rgb=(p["sh0"][:, 0] * C0 + 0.5).clamp(0, 1).detach().cpu().numpy(), seen=seen.cpu().numpy(),
        vote_margin=(votes.abs() / seen.clamp_min(1)).cpu().numpy())
    write_normal_ply(a.out / "surfels_normals.ply", means.cpu().numpy(), nrm.cpu().numpy())
    seen_np = seen.cpu().numpy()
    stats = {"config": str(a.config), "steps": a.steps, "train_frames": len(train), "held_out_frames": len(rows),
             "poses": "colmap" if a.raw_poses else "refined", "surfels": len(means), "train_s": round(t_train),
             "held_out": m, "per_frame": rows, "surfels_seen_by_vote_cams": float((seen_np > 0).mean()),
             "max_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "total_s": round(time.time() - t0)}
    (a.out / "metrics.json").write_text(json.dumps(stats, indent=1))
    print(f"{len(means):,} surfels, {stats['surfels_seen_by_vote_cams']:.1%} seen by the vote cameras; "
          f"train {t_train / 60:.1f} min, total {stats['total_s'] / 60:.1f} min -> {a.out}")


if __name__ == "__main__":
    main()
