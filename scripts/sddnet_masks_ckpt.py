"""Shadow masks (white = shadow) for a frames folder with SDDNet and a chosen checkpoint (e.g. a fine-tuned one).
Same pre/post-processing as sddnet_masks.py (512x512, sigmoid, resize back, threshold).
Usage: sddnet_masks_ckpt.py IMAGES OUT_MASKS --ckpt PATH [--thr 0.5]   (run in the 'sddnet' env)"""
import argparse, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
SDD = Path.home() / "third_party/SDDNet_ACMMM23"
sys.path.insert(0, str(SDD))
from networks.sddnet import SDDNet

ap = argparse.ArgumentParser()
ap.add_argument("images", type=Path); ap.add_argument("out", type=Path)
ap.add_argument("--ckpt", type=Path, required=True)
ap.add_argument("--thr", type=float, default=0.5); ap.add_argument("--size", type=int, default=512)
a = ap.parse_args()
model = SDDNet(backbone="efficientnet-b3", proj_planes=16, pred_planes=32, use_pretrained=False,
               fix_backbone=False, has_se=False, dropout_2d=0, normalize=True, mu_init=0.4, reweight_mode="manual")
model.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False)["model"])
model.cuda().eval()
frames = sorted(p for p in a.images.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
a.out.mkdir(parents=True, exist_ok=True)
t0 = time.time(); fracs = []
with torch.no_grad():
    for i, p in enumerate(frames):
        im = Image.open(p).convert("RGB"); w, h = im.size
        x = torch.from_numpy(np.array(im.resize((a.size, a.size), Image.BILINEAR))).permute(2, 0, 1).float().div(255)[None].cuda()
        prob = torch.sigmoid(model(x)["logit"])
        prob = F.interpolate(prob, size=(h, w), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        m = prob >= a.thr; fracs.append(m.mean())
        Image.fromarray((m * 255).astype(np.uint8)).save(a.out / (p.stem + ".png"))
        if (i + 1) % 100 == 0: print(f"{i+1}/{len(frames)} {time.time()-t0:.0f}s", flush=True)
print(f"done {len(frames)} frames in {time.time()-t0:.1f}s, mean shadow fraction {np.mean(fracs):.3f} (ckpt {a.ckpt})")
