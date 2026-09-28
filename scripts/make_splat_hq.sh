#!/usr/bin/env bash
# Crisp, high-quality Gaussian Splat from a video, aligned to COLMAP coordinates.
#
# Pipeline: sharpest-frame selection -> COLMAP (ns-process-data images)
#           -> use largest COLMAP sub-model, refine intrinsics (fix_colmap_model.py)
#           -> splatfacto-big at full resolution with pose refinement
#              (falls back to splatfacto if the big model fails, e.g. GPU out of memory)
#           -> export .ply
#
# Usage: ./scripts/make_splat_hq.sh <video_path> [scene_name] [num_frames] [max_iters]
set -uo pipefail
source ~/miniforge3/bin/activate gsplat

VIDEO="${1:?Usage: make_splat_hq.sh <video_path> [scene_name] [num_frames] [max_iters]}"
SCENE="${2:-$(basename "${VIDEO%.*}")}"
NUM_FRAMES="${3:-650}"
MAX_ITERS="${4:-30000}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
DATA="data/$SCENE"

# Steps 1-3 are skipped if a previous run already produced poses for this scene.
if [[ -f "$DATA/transforms_nsprocess.json" && -f "$DATA/transforms.json" ]]; then
  echo "=== [1-3/5] Reusing existing frames + COLMAP poses in $DATA ==="
else
  echo "=== [1/5] Selecting $NUM_FRAMES sharpest frames ($(date)) ==="
  python scripts/select_sharp_frames.py "$VIDEO" "$DATA/raw_frames" "$NUM_FRAMES" || { echo "FRAME SELECTION FAILED"; exit 1; }

  echo "=== [2/5] COLMAP ($(date)) ==="
  ns-process-data images --data "$DATA/raw_frames" --output-dir "$DATA" || { echo "PROCESSING FAILED"; exit 1; }

  echo "=== [3/5] Fixing COLMAP model ($(date)) ==="
  python scripts/fix_colmap_model.py "$DATA" || { echo "POSE FIX FAILED"; exit 1; }
fi

train() {  # $1 = method
  TORCHDYNAMO_DISABLE=1 ns-train "$1" \
    --output-dir outputs --experiment-name "$SCENE" \
    --max-num-iterations "$MAX_ITERS" \
    --pipeline.datamanager.cache-images cpu \
    --pipeline.model.camera-optimizer.mode SO3xR3 \
    --viewer.quit-on-train-completion True \
    nerfstudio-data --data "$DATA" \
    --downscale-factor 1 \
    --orientation-method none --center-method none --auto-scale-poses False
}
echo "=== [4/5] Training splatfacto-big ($(date)) ==="
if ! train splatfacto-big; then
  echo "=== [4/5] splatfacto-big FAILED, retrying with splatfacto ($(date)) ==="
  train splatfacto || { echo "TRAINING FAILED"; exit 1; }
fi

CFG=$(ls -t outputs/$SCENE/*/*/config.yml | head -1)
echo "=== [5/5] Export ($(date)) ==="
ns-export gaussian-splat --load-config "$CFG" --output-dir "exports/$SCENE" || { echo "EXPORT FAILED"; exit 1; }
echo "=== DONE $SCENE ($(date)) config: $CFG ==="
