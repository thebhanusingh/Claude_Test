#!/usr/bin/env python
"""Fine-tune an existing splatfacto splat with PPISP so its colours become camera-free linear radiance.

PPISP (NVIDIA, CVPR 2026, github nv-tlabs/ppisp) wraps every rendered image in a learned camera pipeline: per-frame
exposure and colour correction, vignetting, and the camera response curve (CRF). The splat then only has to explain
the scene's light, not the phone's processing. The splat starts from --ply with its base colours converted from sRGB
to linear (view-dependent SH scaled to match), so the CRF, not the splat, takes the tone curve; geometry is kept
(no densification, small position learning rate). Frames, held-out split, undistortion and refined training poses
come from the splatfacto config, as in train_2dgs_ref.py. After --controller-ratio of the steps the scene is frozen
and PPISP trains its controller (exposure / colour for new views, frame_idx = -1).

Writes <out>/splat_linear.ply (same blobs and format as --ply, colours linear), ppisp.pt (PPISP state), ppisp_report/
(PDF + JSON from export_ppisp_report), metrics.json (held-out PSNR / SSIM / LPIPS with the controller), eval/ renders.

Usage: train_ppisp_ft.py --config outputs/IMG_6705/splatfacto/<run>/config.yml --ply exports/IMG_6705/splat.ply
           --out exports/IMG_6705_ppisp [--steps 15000]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from gsplat import rasterization
from PIL import Image
from ppisp import PPISP
from pytorch_msssim import SSIM, ssim
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

sys.path.insert(0, str(Path(__file__).parent))
from fill_shadows import srgb_to_linear
from fix_splat_shadows import read_ply, write_ply
from train_2dgs_ref import C0, intrinsics, refined_c2w, viewmat


def load_splat(path):
    header, g = read_ply(path)
    col = lambda *k: np.stack([g[x] for x in k], 1).astype(np.float32)
    n = len(g)
    rest = sorted([k for k in g.dtype.names if k.startswith("f_rest_")], key=lambda k: int(k[7:]))
    shn = col(*rest).reshape(n, 3, -1).transpose(0, 2, 1)
    dc = col("f_dc_0", "f_dc_1", "f_dc_2")
    srgb = np.clip(dc * C0 + 0.5, 1e-4, 1)
    lin = srgb_to_linear(srgb)
    shn = shn * (lin / srgb)[:, None, :]  # first-order: keep the view-dependent part in proportion
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    params = {"means": t(col("x", "y", "z")), "scales": t(col("scale_0", "scale_1", "scale_2")),
              "quats": t(col("rot_0", "rot_1", "rot_2", "rot_3")), "opacities": t(g["opacity"]),
              "sh0": t((lin - 0.5) / C0)[:, None], "shN": t(shn)}
    return header, g, rest, params


def save_splat(path, header, g, rest, p):
    d = g.copy()
    P = {k: v.detach().cpu().numpy() for k, v in p.items()}
    for i, k in enumerate("xyz"):
        d[k] = P["means"][:, i]
    for i in range(3):
        d[f"scale_{i}"] = P["scales"][:, i]
        d[f"f_dc_{i}"] = P["sh0"][:, 0, i]
    for i in range(4):
        d[f"rot_{i}"] = P["quats"][:, i]
    d["opacity"] = P["opacities"]
    fr = P["shN"].transpose(0, 2, 1).reshape(len(d), -1)
    for i, k in enumerate(rest):
        d[k] = fr[:, i]
    write_ply(path, header, d)


def render(p, vm, K, W, H, deg=3):
    return rasterization(p["means"], torch.nn.functional.normalize(p["quats"], dim=1), torch.exp(p["scales"]),
                         torch.sigmoid(p["opacities"]), torch.cat([p["sh0"], p["shN"]], 1), vm[None], K[None], W, H,
                         sh_degree=deg)[0][0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--controller-ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "eval").mkdir(exist_ok=True)
    dev = "cuda"
    t0 = time.time()

    config = yaml.load(a.config.read_text(), Loader=yaml.Loader)
    dm = config.pipeline.datamanager.setup(device=dev, test_mode="val", world_size=1, local_rank=0)
    train, held = dm.cached_train, dm.cached_eval
    tc, ec = dm.train_dataset.cameras, dm.eval_dataset.cameras
    c2w = refined_c2w(a.config, tc)
    c2w = tc.camera_to_worlds if c2w is None else c2w
    tvm = [viewmat(c.to(dev)) for c in c2w]
    evm = [viewmat(c.to(dev)) for c in ec.camera_to_worlds]
    centers = c2w[:, :3, 3]
    scene_scale = float((centers - centers.mean(0)).norm(dim=1).max()) * 1.1
    Ks = [intrinsics(tc, i) for i in range(len(train))]
    print(f"{len(train)} training / {len(held)} held-out frames ({time.time() - t0:.0f} s)")

    header, g, rest, init = load_splat(a.ply)
    p = torch.nn.ParameterDict(init).to(dev)
    print(f"{len(g):,} blobs from {a.ply}, colours converted to linear")
    lrs = {"means": 1.6e-5 * scene_scale, "scales": 2e-3, "quats": 5e-4, "opacities": 2e-2, "sh0": 2.5e-3,
           "shN": 1.25e-4}
    opt = {k: torch.optim.Adam([p[k]], lr=lr, eps=1e-15) for k, lr in lrs.items()}
    isp = PPISP(num_cameras=1, num_frames=len(train)).to(dev)
    iopt = isp.create_optimizers()
    isched = isp.create_schedulers(iopt, a.steps)
    freeze_at = int(a.controller_ratio * a.steps)

    W0, H0 = Ks[0][1], Ks[0][2]
    yy, xx = torch.meshgrid(torch.arange(H0, device=dev), torch.arange(W0, device=dev), indexing="ij")
    coords = torch.stack([xx, yy], -1).float() + 0.5
    order = []
    for step in range(a.steps):
        if not order:
            order = torch.randperm(len(train)).tolist()
        i = order.pop()
        K, W, H = Ks[i]
        gt = train[i]["image"].to(dev).float()
        gt = (gt / 255 if train[i]["image"].dtype == torch.uint8 else gt)[..., :3]
        raw = render(p, tvm[i], K.to(dev), W, H)
        pc = coords if (W, H) == (W0, H0) else torch.stack(torch.meshgrid(
            torch.arange(W, device=dev), torch.arange(H, device=dev), indexing="xy"), -1).float() + 0.5
        out = isp(raw, pc, resolution=(W, H), camera_idx=0, frame_idx=i)
        l1 = (out - gt).abs().mean()
        loss = 0.8 * l1 + 0.2 * (1 - ssim(out.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], data_range=1.0))
        loss = loss + isp.get_regularization_loss()
        loss.backward()
        if step < freeze_at:
            for o in opt.values():
                o.step()
        for o in opt.values():
            o.zero_grad(set_to_none=True)
        for o in iopt:
            o.step()
            o.zero_grad(set_to_none=True)
        for s in isched:
            s.step()
        if step % 1000 == 0 or step == a.steps - 1:
            print(f"step {step:6d}  loss {loss.item():.4f}  l1 {l1.item():.4f}  "
                  f"{'controller' if step >= freeze_at else 'scene+isp'}  GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB  "
                  f"{time.time() - t0:.0f} s", flush=True)
    save_splat(a.out / "splat_linear.ply", header, g, rest, p)
    torch.save(isp.state_dict(), a.out / "ppisp.pt")
    try:
        from ppisp.report import export_ppisp_report
        export_ppisp_report(isp, frames_per_camera=[len(train)], output_dir=a.out / "ppisp_report", camera_names=["phone"])
    except Exception as e:  # the report is a convenience; the state dict holds everything
        print(f"report not written: {e}")

    psnr_f, ssim_f = PeakSignalNoiseRatio(data_range=1.0), SSIM(data_range=1.0, size_average=True, channel=3)
    lpips_f = LearnedPerceptualImagePatchSimilarity(normalize=True).to(dev)
    rows = []
    isp.eval()  # inference: frame_idx = -1 uses the controller
    with torch.no_grad():
        for j, data in enumerate(held):
            K, W, H = intrinsics(ec, j)
            gt = data["image"].to(dev).float()
            gt = (gt / 255 if data["image"].dtype == torch.uint8 else gt)[..., :3]
            raw = render(p, evm[j], K.to(dev), W, H)
            pred = isp(raw, coords, resolution=(W, H), camera_idx=0, frame_idx=-1).clamp(0, 1)
            gg, rr = gt.permute(2, 0, 1)[None], pred.permute(2, 0, 1)[None]
            rows.append({"frame": j, "psnr": float(psnr_f(gg.cpu(), rr.cpu())), "ssim": float(ssim_f(gg, rr)),
                         "lpips": float(lpips_f(gg, rr))})
            Image.fromarray((torch.cat([gt, pred, raw.clamp(0, 1)], 1).cpu().numpy() * 255).astype(np.uint8)).save(
                a.out / "eval" / f"eval_{j:03d}.jpg", quality=90)
    m = {k: float(np.mean([r[k] for r in rows])) for k in ("psnr", "ssim", "lpips")}
    st = {"steps": a.steps, "held_out_with_controller": m, "per_frame": rows, "train_s": round(time.time() - t0),
          "max_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    (a.out / "metrics.json").write_text(json.dumps(st, indent=1))
    print(f"held-out ({len(rows)}): PSNR {m['psnr']:.2f}  SSIM {m['ssim']:.4f}  LPIPS {m['lpips']:.4f}; "
          f"total {st['train_s'] / 60:.1f} min -> {a.out}")


if __name__ == "__main__":
    main()
