#!/bin/bash
# IMG_6704 grain check, stage 4 (visibility + adjacent detection), stage 5 (continuous fill), comparison renders.
# Same settings as IMG_6556; physical-size settings scaled by ~5.6 (IMG_6556 1.33 units/m -> IMG_6704 ~7.5 units/m).
set -o pipefail
source ~/miniforge3/bin/activate gsplat; cd ~/Claude_Test
S=IMG_6704; CL=exports/${S}_clusters_r30agree_ms50; PL=exports/${S}_shadowfix_o/stats.json; COL=data/$S/colmap/sparse/0_refined
CROP="0.178 0.683 1.86 1.3"; SUN=exports/${S}_sun_edges_fitshift/sun_edges.json
# (done 14:47) echo "=== [3b] grain check $(date +%T)"
# (done 14:47) for px in interior edges; do
# (done 14:47)   python scripts/check_grain.py --run $CL --ply exports/$S/splat.ply --colmap $COL --images data/$S/images --crop $CROP --up-from $PL --frames 6 --pixels $px || { echo "GRAIN CHECK FAILED"; exit 1; }
# (done 14:47) done
echo "=== [4a] shadow_visibility $(date +%T)"
python scripts/shadow_visibility.py --ply exports/$S/splat.ply --colmap $COL --plane-from $PL --sun-from $SUN --clusters $CL/clusters.npz \
  --crop $CROP --ground-radius 3.5 --ground-votes exports/${S}_sun/votes.npz --out exports/${S}_shadow4 --opaque 0.5 --occluder-min 0.25 \
  --min-views 5 --cell 0.1 --dilate 1 --eps 0.1 --eps-object 0.34 --attached-cos 0.0 --every 4 --zbuf-scale 0.5 --vis-tol 0.07 \
  --knn 8 --graze 0.15 --revote --revote-ztol 0.03 --masks data/$S/shadow_masks || { echo "VISIBILITY FAILED"; exit 1; }
OV=""; [ -f exports/${S}_shadow4/votes_object_tight.npz ] && OV="--object-votes exports/${S}_shadow4/votes_object_tight.npz"
echo "=== [4b] detect_object_shadow adjacent $(date +%T)"
python scripts/detect_object_shadow.py --ply exports/$S/splat.ply --flags exports/${S}_shadow4/shadow_flags.npz --clusters $CL/clusters.npz $OV \
  --out exports/${S}_objshadow_adj --knn 8 --facing 0.6 --averted -0.3 --min-contrast 1.5 --min-views 5 --source photos --classify adjacent \
  --adj-radius 2.24 --adj-chroma 0.12 --adj-pct 90 --adj-ratio 0.3 --smooth 15 --colmap $COL --images data/$S/images --every 4 \
  --zbuf-scale 0.5 --zbuf-radius 2.0 --vis-tol 0.07 || { echo "DETECT FAILED"; exit 1; }
echo "=== [5] fill_shadows continuous $(date +%T)"
python scripts/fill_shadows.py --ply exports/$S/splat.ply --labels exports/${S}_objshadow_adj/chair_shadow.npz --clusters $CL/clusters.npz \
  --out exports/${S}_fill_continuous_sref_fb40 --mode continuous --ref-k 300 --ref-pct 80 --include-unclustered --crop $CROP --plane-from $PL \
  --min-opacity 0.05 --attach-dist 0.22 --fallback-pct 40 --surface-ref --dark-full 0.3 --dark-none 0.6 --graze 0.1 --k 8 --across 4 \
  --min-donors 50 --smooth-radius 0.22 --smooth-iters 2 --feather-radius 0.11 --gain-smooth 3 --max-gain 20 || { echo "FILL FAILED"; exit 1; }
echo "=== [6] renders $(date +%T)"
python scripts/render_splat_compare.py --colmap $COL --images data/$S/images --out exports/${S}_fill_compare \
  --frames frame_00040,frame_00200,frame_00360,frame_00520 --plys exports/$S/splat.ply exports/${S}_fill_continuous_sref_fb40/splat_fill.ply \
  --labels original "shadow fill (stage 5)" --scale 0.5 || { echo "RENDER FAILED"; exit 1; }
echo "=== STAGE345 DONE $(date +%T)"
