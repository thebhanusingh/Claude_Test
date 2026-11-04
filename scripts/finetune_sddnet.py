#!/usr/bin/env python
"""Fine-tune SDDNet (SBU checkpoint) on one scene's frames with chair-aware targets (make_chair_targets.py).

Targets: 0 lit, 255 shadow, 128 ignore. Loss: BCE on the shadow-mask logit over non-ignore pixels only (SDDNet's
auxiliary reconstruction losses are not used). BatchNorm statistics stay frozen (small batches, one scene).
Split: a contiguous block of frames at the end of the capture is held out (--val-from), with a --gap of frames dropped
before it so neighbouring frames don't leak. The original checkpoint is never modified.

Evaluation on the held-out frames, before and after: on the object (non-ignore object pixels from the targets) IoU and
accuracy of the predicted mask vs the 3D labels; off the object, agreement with the original SDDNet masks (does it
still find the ground shadows the same way?).

Run in the 'sddnet' conda env:
  finetune_sddnet.py --images data/IMG_6556/images --targets exports/IMG_6556_chair_targets \
      --orig-masks data/IMG_6556/shadow_masks --out exports/IMG_6556_sddnet_ft [--epochs 4 --lr 2e-5]
Writes <out>/sddnet_chair_ft.ckpt, train_log.json, val_<frame>.jpg (photo | before | after | target).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SDD = Path.home() / "third_party/SDDNet_ACMMM23"
sys.path.insert(0, str(SDD))
from networks.sddnet import SDDNet


def build():
    m = SDDNet(backbone="efficientnet-b3", proj_planes=16, pred_planes=32, use_pretrained=False, fix_backbone=False,
               has_se=False, dropout_2d=0, normalize=True, mu_init=0.4, reweight_mode="manual")
    m.load_state_dict(torch.load(SDD / "ckpt/sbu.ckpt", map_location="cpu", weights_only=False)["model"])
    return m.cuda()


def load(img_path, tgt_path, size):
    im = Image.open(img_path).convert("RGB")
    x = torch.from_numpy(np.asarray(im.resize((size, size), Image.BILINEAR))).permute(2, 0, 1).float().div(255)
    t = np.asarray(Image.open(tgt_path).convert("L").resize((size, size), Image.NEAREST))
    y = torch.from_numpy((t == 255).astype(np.float32))[None]
    valid = torch.from_numpy((t != 128).astype(np.float32))[None]
    return x, y, valid, t


def logits(model, x):
    out = model(x)
    return out["logit"] if isinstance(out, dict) else out[1]


def freeze_bn(model):
    for mod in model.modules():
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            mod.eval()


@torch.no_grad()
def evaluate(model, items, size, orig_dir, obj_regions):
    model.eval()
    acc = dict(obj_tp=0, obj_fp=0, obj_fn=0, obj_ok=0, obj_n=0, off_ok=0, off_n=0)
    preds = {}
    for name, (img_p, tgt_p) in items:
        x, y, valid, t = load(img_p, tgt_p, size)
        p = (torch.sigmoid(logits(model, x[None].cuda()))[0, 0].cpu().numpy() >= 0.5)
        preds[name] = p
        obj = obj_regions[name] & (t != 128)
        gt = t == 255
        acc["obj_tp"] += int((p & gt & obj).sum()); acc["obj_fp"] += int((p & ~gt & obj).sum())
        acc["obj_fn"] += int((~p & gt & obj).sum()); acc["obj_ok"] += int(((p == gt) & obj).sum())
        acc["obj_n"] += int(obj.sum())
        orig = np.asarray(Image.open(orig_dir / Path(tgt_p).name).convert("L").resize((size, size), Image.NEAREST)) > 127
        off = ~obj_regions[name]
        acc["off_ok"] += int(((p == orig) & off).sum()); acc["off_n"] += int(off.sum())
    r = dict(object_iou=round(acc["obj_tp"] / max(acc["obj_tp"] + acc["obj_fp"] + acc["obj_fn"], 1), 3),
             object_accuracy=round(acc["obj_ok"] / max(acc["obj_n"], 1), 3),
             off_object_agreement_with_original=round(acc["off_ok"] / max(acc["off_n"], 1), 4))
    return r, preds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--orig-masks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--val-from", type=float, default=0.85, help="frames after this fraction of the capture = val")
    ap.add_argument("--gap", type=float, default=0.05, help="fraction of frames dropped before the val block")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    tg = sorted(p for p in a.targets.glob("frame_*.png"))
    items = [(p.stem, (a.images / p.name, p)) for p in tg if (a.images / p.name).exists()]
    n = len(items)
    v0, g0 = int(n * a.val_from), int(n * (a.val_from - a.gap))
    train, val = items[:g0], items[v0:]
    print(f"{n} frames with targets: train {len(train)} ({train[0][0]}..{train[-1][0]}), "
          f"val {len(val)} ({val[0][0]}..{val[-1][0]}), gap {v0 - g0}", flush=True)
    obj_regions = {}  # object pixels per val frame, written by make_chair_targets.py (regions/)
    for name, (img_p, tgt_p) in val:
        rp = a.targets / "regions" / Path(tgt_p).name
        obj_regions[name] = (np.asarray(Image.open(rp).convert("L").resize((a.size, a.size), Image.NEAREST)) > 127
                             if rp.exists() else np.zeros((a.size, a.size), bool))

    model = build()
    before, preds0 = evaluate(model, val, a.size, a.orig_masks, obj_regions)
    print(f"before fine-tuning (val): {before}", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=1e-4)
    log = dict(before=before, epochs=[])
    t0 = time.time()
    for ep in range(a.epochs):
        model.train(); freeze_bn(model)
        perm = torch.randperm(len(train)).tolist()
        tot, nb = 0.0, 0
        for i in range(0, len(perm), a.batch):
            xs, ys, vs = [], [], []
            for j in perm[i:i + a.batch]:
                x, y, v, _ = load(*train[j][1], a.size)
                if torch.rand(1).item() < 0.5:
                    x, y, v = x.flip(-1), y.flip(-1), v.flip(-1)
                xs.append(x); ys.append(y); vs.append(v)
            x, y, v = torch.stack(xs).cuda(), torch.stack(ys).cuda(), torch.stack(vs).cuda()
            lg = logits(model, x)
            if lg.shape[-2:] != y.shape[-2:]:
                lg = F.interpolate(lg, size=y.shape[-2:], mode="bilinear", align_corners=False)
            loss = (F.binary_cross_entropy_with_logits(lg, y, reduction="none") * v).sum() / v.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        res, _ = evaluate(model, val, a.size, a.orig_masks, obj_regions)
        log["epochs"].append(dict(epoch=ep + 1, train_loss=round(tot / nb, 4), val=res, seconds=round(time.time() - t0)))
        print(f"epoch {ep + 1}/{a.epochs}: train loss {tot / nb:.4f} | val {res} ({time.time() - t0:.0f} s)", flush=True)
    torch.save({"model": model.state_dict(), "note": "SDDNet SBU fine-tuned on IMG_6556 chair-aware targets"},
               a.out / "sddnet_chair_ft.ckpt")
    after, preds1 = evaluate(model, val, a.size, a.orig_masks, obj_regions)
    log["after"] = after
    (a.out / "train_log.json").write_text(json.dumps(log, indent=1))
    print(f"after fine-tuning (val): {after}", flush=True)
    # visual check on a few val frames
    for name, (img_p, tgt_p) in val[:: max(1, len(val) // 4)]:
        im = np.asarray(Image.open(img_p).convert("RGB").resize((a.size, a.size)))
        t = np.asarray(Image.open(tgt_p).convert("L").resize((a.size, a.size), Image.NEAREST))

        def ov(m):
            o = im.copy().astype(np.float32)
            o[m] = 0.5 * o[m] + [128, 0, 0]
            return o.astype(np.uint8)
        strip = np.concatenate([im, ov(preds0[name]), ov(preds1[name]), ov(t == 255)], 1)
        Image.fromarray(strip).save(a.out / f"val_{name}.jpg", quality=88)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
