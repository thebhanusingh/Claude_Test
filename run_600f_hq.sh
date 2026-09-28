#!/bin/bash
# High-quality retrain of my_scene_600f (poses already computed; transforms.json is in
# original COLMAP world coordinates, so the exported splat lines up with the SfM points).
set -uo pipefail
source ~/miniforge3/bin/activate gsplat
cd ~/Claude_Test
SCENE=my_scene_600f

echo "=== [2/3] Training HQ ($(date)) ==="
# 30k steps at full 1920x1080 (default would auto-downscale to 960x540).
# Frames cached in system RAM to leave GPU memory for Gaussians.
# orientation/center/auto-scale disabled: keep COLMAP's coordinate frame.
TORCHDYNAMO_DISABLE=1 ns-train splatfacto \
  --output-dir outputs --experiment-name ${SCENE}_hq \
  --max-num-iterations 30000 \
  --pipeline.datamanager.cache-images cpu \
  --viewer.quit-on-train-completion True \
  nerfstudio-data --data data/$SCENE \
  --downscale-factor 1 \
  --orientation-method none --center-method none --auto-scale-poses False \
  || { echo "TRAINING FAILED"; exit 1; }

CFG=$(ls -t outputs/${SCENE}_hq/splatfacto/*/config.yml | head -1)
echo "=== [3/3] Export ($(date)) ==="
ns-export gaussian-splat --load-config "$CFG" --output-dir exports/${SCENE}_hq

echo "=== DONE ($(date)) - reopening viewer on $CFG ==="
exec ns-viewer --load-config "$CFG" --viewer.websocket-port 7007
