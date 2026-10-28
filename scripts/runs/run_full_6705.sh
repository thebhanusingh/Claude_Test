#!/bin/bash
# IMG_6705 (bear 2) end to end in one command: video -> splat -> shadow-free splat (object fill + ground clone stamp).
# Settings: bear 1 (IMG_6704) final settings. Every length is bear 1's value x K, where K = (bear 2 units/m) / (bear 1
# units/m), from the splat height of each bear and its real height (bear 1 2.13 units / 0.175 m = 12.18 units/m;
# bear 2 0.16 m, user measured). Sun: elevation fixed from the video's GPS + capture time (no compass heading in the
# metadata), azimuth and receiver offset fitted from the shadow outline; our own free fits are logged next to it.
# Run: scripts/runs/run_full_6705.sh 2>&1 | tee logs/IMG_6705_full.log
set -o pipefail
source ~/miniforge3/bin/activate gsplat; cd ~/Claude_Test
S=IMG_6705; V=/mnt/c/Users/roach/Downloads/$S.MOV; HEIGHT_M=0.16; REF_UPM=12.18
COL=data/$S/colmap/sparse/0_refined; E=exports
step() { echo; echo "=== [$1] $2 $(date +%T)"; }
fail() { echo "$1 FAILED $(date +%T)"; exit 1; }
x() { python -c "print(round($1 * $K, 4))"; }   # bear 1 length -> bear 2 length

step 0 "sun from the video's metadata (GPS + capture time)"
python scripts/sun_from_metadata.py $V --out $E/${S}_sun_meta/sun_metadata.json || fail SUN_METADATA
ELEV=$(python -c "import json; print(json.load(open('$E/${S}_sun_meta/sun_metadata.json'))['elevation_deg'])")
SUN_AZ=$(python -c "import json; print(json.load(open('$E/${S}_sun_meta/sun_metadata.json'))['azimuth_deg'])")

step 1 "capture: frames, COLMAP poses, splatfacto-big 30k, export"
MAX_WIDTH=1920 STOP_SPLIT=7000 ./scripts/make_splat_hq.sh $V $S || fail SPLAT
[ -f $E/$S/splat.ply ] && [ -d $COL ] || fail "SPLAT OUTPUT CHECK"

step 1b "SDDNet shadow masks"
source ~/miniforge3/bin/activate sddnet
python scripts/sddnet_masks.py data/$S/images data/$S/shadow_masks || fail MASKS
source ~/miniforge3/bin/activate gsplat

step 2a "ground plane (bear 1 'O' run, plane only used)"
python scripts/fix_splat_shadows.py --ply $E/$S/splat.ply --colmap $COL --masks data/$S/shadow_masks --images data/$S/images \
  --out $E/${S}_shadowfix_o --every 2 --zbuf-scale 0.25 --z-tol 0.03 --opaque 0.5 --min-views 5 --lo 0.15 --hi 0.45 \
  --max-gain 12 --colour lum --mode target --colour-gate 0.13 --ceiling-pct 50 --band 6 --band-gap 8 --ground-only \
  --ground-tol 0.045 --ground-spread 0.06 --dark-gate || fail PLANE
PL=$E/${S}_shadowfix_o/stats.json

step 2b "find the bear and the scale"
python scripts/find_object.py --ply $E/$S/splat.ply --colmap $COL --plane-from $PL --height-m $HEIGHT_M \
  --ref-units-per-m $REF_UPM --out $E/${S}_object || fail FIND_OBJECT
OBJ=$E/${S}_object/object.json
CROP=$(python -c "import json; print(*json.load(open('$OBJ'))['crop'])")
CENTRE=$(echo $CROP | cut -d' ' -f1-3)
K=$(python -c "import json; print(json.load(open('$OBJ'))['k'])")
echo "crop $CROP, K $K"

step 2c "surfaces (stage 3 settings, min size 50)"
CL=$E/${S}_clusters_r30agree_ms50
python scripts/cluster_surfaces.py --ply $E/$S/splat.ply --out $CL --crop $CROP --region-angle 30 --normal-source agree \
  --agree 30 --min-size 50 || fail CLUSTERS

step 2d "grain check against the photos"
for px in interior edges; do
  python scripts/check_grain.py --run $CL --ply $E/$S/splat.ply --colmap $COL --images data/$S/images --crop $CROP \
    --up-from $PL --frames 6 --pixels $px || fail "GRAIN CHECK"
done

step 3a "sun: shadow votes + our own area-IoU fit (logged only)"
python scripts/fit_sun.py --ply $E/$S/splat.ply --colmap $COL --masks data/$S/shadow_masks --plane-from $PL \
  --centre $CENTRE --radius $(x 3.5) --out $E/${S}_sun --occluder-labels $CL/clusters.npz --score-radius $(x 3.5) \
  --occluder-min $(x 0.25) --opaque 0.5 --max-occluders 150000 --cell $(x 0.1) --dilate 1 --eps $(x 0.1) --tie 0.005 \
  --every 2 --min-views 5 --zbuf-scale 0.25 --z-tol $(x 0.15) --ground-tol $(x 0.1) || fail FIT_SUN

SUNARGS="--ply $E/$S/splat.ply --plane-from $PL --clusters $CL/clusters.npz --crop $CROP --ground-votes $E/${S}_sun/votes.npz
  --score-radius $(x 3.5) --opaque 0.5 --occluder-min $(x 0.25) --min-views 5 --cell $(x 0.1) --dilate 1 --eps $(x 0.1)
  --raster $(x 0.1) --tie-rel 0.03 --fit-shift $(x -0.45) $(x 0.15) $(x 0.05) --true-elevation $ELEV"
step 3b "sun: our own boundary fit, elevation free (logged only)"
python scripts/fit_sun_edges.py $SUNARGS --out $E/${S}_sun_edges_free || fail "FIT_SUN_EDGES FREE"
step 3c "sun used downstream: elevation $ELEV from metadata, azimuth + offset fitted"
python scripts/fit_sun_edges.py $SUNARGS --fix-elevation $ELEV --out $E/${S}_sun_edges_meta || fail "FIT_SUN_EDGES META"
SUN=$E/${S}_sun_edges_meta/sun_edges.json
python - <<EOF
import json
f = json.load(open("$E/${S}_sun_edges_free/sun_edges.json"))["nelder_mead"]
m = json.load(open("$SUN"))["nelder_mead"]
north = (m["azimuth_deg"] + $SUN_AZ) % 360
print(f"sun check: metadata elevation $ELEV, compass azimuth $SUN_AZ; own free fit elevation {f['elevation_deg']:.2f} "
      f"(error {f['elevation_deg'] - $ELEV:+.2f}), azimuth {f['azimuth_deg']:.2f}; used: elevation {m['elevation_deg']:.2f}, "
      f"azimuth {m['azimuth_deg']:.2f} (scene frame, counter-clockwise) -> true north at scene azimuth {north:.2f}; "
      f"free vs used azimuth {((f['azimuth_deg'] - m['azimuth_deg'] + 180) % 360) - 180:+.2f} deg")
EOF

step 4a "visibility (sun-blocked + facing-away)"
python scripts/shadow_visibility.py --ply $E/$S/splat.ply --colmap $COL --plane-from $PL --sun-from $SUN --clusters $CL/clusters.npz \
  --crop $CROP --ground-radius $(x 3.5) --ground-votes $E/${S}_sun/votes.npz --out $E/${S}_shadow4 --opaque 0.5 \
  --occluder-min $(x 0.25) --min-views 5 --cell $(x 0.1) --dilate 1 --eps $(x 0.1) --eps-object $(x 0.34) --attached-cos 0.0 \
  --every 4 --zbuf-scale 0.5 --vis-tol $(x 0.07) --knn 8 --graze 0.15 --revote --revote-ztol $(x 0.03) \
  --masks data/$S/shadow_masks || fail VISIBILITY
OV=""; [ -f $E/${S}_shadow4/votes_object_tight.npz ] && OV="--object-votes $E/${S}_shadow4/votes_object_tight.npz"

step 4b "object shadow, adjacent brightness-and-colour rule"
python scripts/detect_object_shadow.py --ply $E/$S/splat.ply --flags $E/${S}_shadow4/shadow_flags.npz --clusters $CL/clusters.npz $OV \
  --out $E/${S}_objshadow_adj --knn 8 --facing 0.6 --averted -0.3 --min-contrast 1.5 --min-views 5 --source photos \
  --classify adjacent --adj-radius $(x 2.24) --adj-chroma 0.12 --adj-pct 90 --adj-ratio 0.3 --smooth 15 --colmap $COL \
  --images data/$S/images --every 4 --zbuf-scale 0.5 --zbuf-radius 2.0 --vis-tol $(x 0.07) || fail DETECT

step 5a "ground fill: texture + outline blend (bear 1 'U')"
GU=$E/${S}_shadowfix_tex_u
python scripts/fix_splat_shadows.py --ply $E/$S/splat.ply --colmap $COL --masks data/$S/shadow_masks --images data/$S/images \
  --out $GU --every 2 --zbuf-scale 0.25 --z-tol $(x 0.03) --opaque 0.5 --min-views 5 --lo 0.1 --hi 0.35 --gain 20 15 9.4 \
  --max-gain 25 --colour rgb --mode target --colour-gate 0.13 --ceiling-pct 50 --band 6 --band-gap 8 --ground-only \
  --ground-tol $(x 0.1) --ground-spread $(x 0.06) --dark-gate --fill texture --tex-radius $(x 0.1) --tex-step $(x 0.25) \
  --tex-max-shift $(x 6.0) --tex-lit 0.1 --tex-cover 0.97 --tex-feather $(x 0.2) --tex-match 200 || fail "GROUND FILL"

step 5b "object fill (stage 5 continuous), on top of the ground fill"
FO=$E/${S}_fill_continuous_sref_fb40
python scripts/fill_shadows.py --ply $GU/splat_shadowfix.ply --labels $E/${S}_objshadow_adj/chair_shadow.npz --clusters $CL/clusters.npz \
  --out $FO --mode continuous --ref-k 300 --ref-pct 80 --include-unclustered --crop $CROP --plane-from $PL \
  --min-opacity 0.05 --attach-dist $(x 0.22) --fallback-pct 40 --surface-ref --dark-full 0.3 --dark-none 0.6 --graze 0.1 --k 8 \
  --across 4 --min-donors 50 --smooth-radius $(x 0.22) --smooth-iters 2 --feather-radius $(x 0.11) --gain-smooth 3 \
  --max-gain 20 || fail "OBJECT FILL"

step 5c "ground clone stamp with crack check (bear 1 'Y')"
CY=$E/${S}_clone_y
python scripts/clone_ground_splats.py --ply $E/$S/splat.ply --base $FO/splat_fill.ply --fix-dir $GU --out $CY \
  --cell $(x 0.05) --hole-frac 0.3 --close 2 --object-height $(x 0.15) --object-margin 1 --source-lit 0.1 --opaque 0.5 \
  --match 200 --crack 0.6 --crack-sigma 4 --crack-margin 1 --crack-min-cells 8 --shift-step $(x 0.25) --max-shift $(x 5.0) \
  --max-shifts 6 --cover 0.97 --map || fail CLONE
mkdir -p $E/${S}_final && cp $CY/splat_clone.ply $E/${S}_final/splat_shadowfree.ply

step 6 "renders: photo | original | shadow-free"
FR=$(ls data/$S/images | sort | awk '{a[NR]=$0} END {n=NR; for (i=1;i<=4;i++) {f=a[int(n*(2*i-1)/8)+1]; sub(/\.[^.]*$/,"",f); printf "%s%s", (i>1?",":""), f}}')
python scripts/render_splat_compare.py --colmap $COL --images data/$S/images --out $E/${S}_final_compare --frames $FR \
  --plys $E/$S/splat.ply $E/${S}_final/splat_shadowfree.ply --labels original "shadow-free (object fill + ground clone)" \
  --scale 0.5 || fail RENDER
echo; echo "=== ALL DONE $(date +%T): $E/${S}_final/splat_shadowfree.ply"
