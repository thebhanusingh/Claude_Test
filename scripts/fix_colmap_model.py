#!/usr/bin/env python
"""Rebuild transforms.json from the largest COLMAP sub-model, in COLMAP world coordinates.

ns-process-data always uses colmap/sparse/0, but COLMAP sometimes splits a scene and
sparse/0 can hold only a handful of frames. This picks the sub-model with the most
registered images, bundle-adjusts it (including principal point), and writes
transforms.json + sparse_pc.ply with keep_original_world_coordinate=True so the
exported splat lines up with the SfM points.

Usage: fix_colmap_model.py <scene_dir>
"""
import json
import shutil
import subprocess
import os
import sys
from pathlib import Path

from nerfstudio.data.utils.colmap_parsing_utils import read_images_binary
from nerfstudio.process_data.colmap_utils import colmap_to_json

scene = Path(sys.argv[1])
# Minimum share of frames that must be posed (env MIN_POSED_FRAC, default 0.8). Lower it only on purpose,
# e.g. when the unposed frames are known to be blank walls or ceiling.
min_frac = float(os.environ.get("MIN_POSED_FRAC", "0.8"))
sparse = scene / "colmap" / "sparse"
num_images = len(list((scene / "images").glob("*.png")))

models = {m: len(read_images_binary(m / "images.bin")) for m in sparse.iterdir() if (m / "images.bin").exists() and m.name.isdigit()}
if not models:
    sys.exit("ERROR: no COLMAP models found")
best = max(models, key=models.get)
for m, n in sorted(models.items()):
    print(f"model {m.name}: {n} images{'  <- using this' if m == best else ''}")
if models[best] < min_frac * num_images:
    print(f"WARNING: best model has only {models[best]}/{num_images} frames")

refined = sparse / f"{best.name}_refined"
refined.mkdir(exist_ok=True)
subprocess.run(["colmap", "bundle_adjuster", "--input_path", str(best), "--output_path", str(refined),
                "--BundleAdjustment.refine_principal_point", "1"], check=True, capture_output=True)

for name in ["transforms.json", "sparse_pc.ply"]:
    if (scene / name).exists():
        shutil.move(scene / name, scene / f"{Path(name).stem}_nsprocess{Path(name).suffix}")
n = colmap_to_json(recon_dir=refined, output_dir=scene, keep_original_world_coordinate=True)
frames = len(json.load(open(scene / "transforms.json"))["frames"])
print(f"transforms.json written with {frames} frames (of {num_images} images)")
if frames < min_frac * num_images:
    sys.exit("ERROR: too few frames have camera poses")
