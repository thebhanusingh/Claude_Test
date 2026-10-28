"""SfM with hloc: SuperPoint features + LightGlue matching, pairs = sequential window + NetVLAD retrieval.
Writes COLMAP binary models to <scene>/colmap/sparse/<n> (largest first as 0).
Usage: hloc_sfm.py SCENE_DIR [--window 20] [--retrieval 20] [--feature superpoint_max] [--camera-model OPENCV]"""
import argparse, shutil, time, json
from pathlib import Path
import pycolmap
from hloc import extract_features, match_features, pairs_from_retrieval, reconstruction

ap = argparse.ArgumentParser()
ap.add_argument("scene", type=Path)
ap.add_argument("--window", type=int, default=20)
ap.add_argument("--retrieval", type=int, default=20)
ap.add_argument("--feature", default="superpoint_max")
ap.add_argument("--camera-model", default="OPENCV")
a = ap.parse_args()
images = a.scene / "images"; out = a.scene / "hloc"; out.mkdir(parents=True, exist_ok=True)
names = sorted(p.name for p in images.iterdir() if p.suffix.lower() in (".png", ".jpg"))
t = {}; t0 = time.time()

# pairs: each frame with the next N frames (video order), plus NetVLAD top-K for loop closure
seq = {(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, min(i + 1 + a.window, len(names)))}
ret_conf = extract_features.confs["netvlad"]
ret_path = extract_features.main(ret_conf, images, out)
ret_pairs = out / "pairs-netvlad.txt"
pairs_from_retrieval.main(ret_path, ret_pairs, num_matched=a.retrieval)
ret = {tuple(sorted(l.split())) for l in ret_pairs.read_text().splitlines() if l.strip()}
allp = sorted({tuple(sorted(p)) for p in seq} | ret)
pairs = out / "pairs-all.txt"; pairs.write_text("\n".join(f"{x} {y}" for x, y in allp))
t["retrieval_s"] = round(time.time() - t0); print(f"pairs: {len(seq)} sequential + {len(ret)} retrieval -> {len(allp)} unique")

t1 = time.time()
feat_conf = extract_features.confs[a.feature]; match_conf = match_features.confs["superpoint+lightglue"]
feats = extract_features.main(feat_conf, images, out)
matches = match_features.main(match_conf, pairs, feat_conf["output"], out)
t["features_matching_s"] = round(time.time() - t1)

t2 = time.time()
sfm = out / "sfm"
model = reconstruction.main(sfm, images, pairs, feats, matches, camera_mode=pycolmap.CameraMode.SINGLE,
                            image_options={"camera_model": a.camera_model}, verbose=False)
t["reconstruction_s"] = round(time.time() - t2)

# collect all sub-models, largest first, into colmap/sparse/<n>
cand = [sfm] + sorted((sfm / "models").glob("*")) if (sfm / "models").exists() else [sfm]
seen, models = set(), []
for d in cand:
    if (d / "images.bin").exists():
        r = pycolmap.Reconstruction(d); key = (r.num_reg_images(), r.num_points3D())
        if d != sfm and key in seen: continue
        seen.add(key); models.append((r.num_reg_images(), d))
models.sort(key=lambda x: -x[0])
dst = a.scene / "colmap" / "sparse"; dst.mkdir(parents=True, exist_ok=True)
for k, (n, d) in enumerate(models):
    o = dst / str(k); o.mkdir(exist_ok=True)
    for f in ("cameras.bin", "images.bin", "points3D.bin"): shutil.copy(d / f, o / f)
cam = pycolmap.Reconstruction(dst / "0").cameras
info = dict(images=len(names), models=[n for n, _ in models], camera_model=[c.model.name for c in cam.values()],
            pairs=len(allp), timings=t, total_s=round(time.time() - t0))
(out / "summary.json").write_text(json.dumps(info, indent=1)); print("SUMMARY", json.dumps(info))
