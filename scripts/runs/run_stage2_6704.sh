#!/bin/bash
# IMG_6704 stage 2: sun votes + boundary fit. Physical-size settings from IMG_6556 scaled by ~5.6 (1.33 -> ~7.5 units/m).
set -o pipefail
source ~/miniforge3/bin/activate gsplat; cd ~/Claude_Test
S=IMG_6704; CL=exports/${S}_clusters_r30agree_ms50/clusters.npz; PL=exports/${S}_shadowfix_o/stats.json; COL=data/$S/colmap/sparse/0_refined
echo "=== [2a] fit_sun votes $(date +%T)"
python scripts/fit_sun.py --ply exports/$S/splat.ply --colmap $COL --masks data/$S/shadow_masks --plane-from $PL \
  --centre 0.178 0.683 1.86 --radius 3.5 --out exports/${S}_sun --occluder-labels $CL --score-radius 3.5 \
  --occluder-min 0.25 --opaque 0.5 --max-occluders 150000 --cell 0.1 --dilate 1 --eps 0.1 --tie 0.005 --every 2 \
  --min-views 5 --zbuf-scale 0.25 --z-tol 0.15 --ground-tol 0.1 || { echo "FIT_SUN FAILED"; exit 1; }
echo "=== [2b] fit_sun_edges --fit-shift $(date +%T)"
python scripts/fit_sun_edges.py --ply exports/$S/splat.ply --plane-from $PL --clusters $CL --crop 0.178 0.683 1.86 1.3 \
  --ground-votes exports/${S}_sun/votes.npz --score-radius 3.5 --out exports/${S}_sun_edges_fitshift --opaque 0.5 \
  --occluder-min 0.25 --min-views 5 --cell 0.1 --dilate 1 --eps 0.1 --raster 0.1 --tie-rel 0.03 \
  --fit-shift -0.45 0.15 0.05 --true-elevation 44.27 || { echo "FIT_SUN_EDGES FAILED"; exit 1; }
echo "=== STAGE2 DONE $(date +%T)"
