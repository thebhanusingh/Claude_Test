#!/bin/bash
# 600-frame retrain of IMG_6449.MOV, short run, low GPU memory.
set -uo pipefail
source ~/miniforge3/bin/activate gsplat
cd ~/Claude_Test
SCENE=my_scene_600f

# SKIP_PROCESS=1 reuses existing frames/poses. Note: if COLMAP splits the scene,
# nerfstudio only reads sparse/0; check transforms.json frame count before training.
if [[ "${SKIP_PROCESS:-0}" != 1 ]]; then
  echo "=== [1/3] Frames + COLMAP ($(date)) ==="
  ns-process-data video --data /mnt/c/Users/roach/Downloads/IMG_6449.MOV \
    --output-dir data/$SCENE --num-frames-target 600 || { echo "PROCESSING FAILED"; exit 1; }
fi

echo "=== [2/3] Training ($(date)) ==="
# cache-images cpu: keep the 600 frames in system RAM instead of GPU memory.
# stop-split-at 5000: stop growing the splat early so Gaussian count stays bounded.
TORCHDYNAMO_DISABLE=1 ns-train splatfacto \
  --data data/$SCENE --output-dir outputs --experiment-name $SCENE \
  --max-num-iterations 7000 \
  --pipeline.datamanager.cache-images cpu \
  --pipeline.model.stop-split-at 5000 \
  --viewer.quit-on-train-completion True || { echo "TRAINING FAILED"; exit 1; }

CFG=$(ls -t outputs/$SCENE/splatfacto/*/config.yml | head -1)
echo "=== [3/3] Export ($(date)) ==="
ns-export gaussian-splat --load-config "$CFG" --output-dir exports/$SCENE

echo "=== DONE ($(date)) - reopening viewer on $CFG ==="
exec ns-viewer --load-config "$CFG" --viewer.websocket-port 7007
