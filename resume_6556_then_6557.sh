#!/bin/bash
# Resume IMG_6556 splatfacto-big from its step-10000 checkpoint with densification capped at
# step 11000 (GPU memory was ~7.2/8 GB and step time climbing), export it, then run IMG_6557.
cd ~/Claude_Test
source ~/miniforge3/bin/activate gsplat
# PyTorch >= 2.6 defaults torch.load(weights_only=True), which rejects nerfstudio's own
# checkpoints (numpy scalars inside). Safe here: we only load checkpoints we wrote.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
SCENE=IMG_6556
CKPT_DIR=outputs/IMG_6556/splatfacto/2026-09-28_143534/nerfstudio_models

echo "=== [4/5] Resuming $SCENE splatfacto-big from step 10000, stop-split-at 10000 ($(date)) ==="
TORCHDYNAMO_DISABLE=1 ns-train splatfacto-big \
  --output-dir outputs --experiment-name $SCENE \
  --max-num-iterations 30000 \
  --load-dir $CKPT_DIR \
  --pipeline.model.stop-split-at 10000 \
  --pipeline.datamanager.cache-images cpu \
  --pipeline.model.camera-optimizer.mode SO3xR3 \
  --viewer.quit-on-train-completion True \
  nerfstudio-data --data data/$SCENE \
  --downscale-factor 1 \
  --orientation-method none --center-method none --auto-scale-poses False \
  || { echo "=== $SCENE RESUME FAILED - stopping, IMG_6557 not started ==="; exit 1; }

CFG=$(ls -t outputs/$SCENE/*/*/config.yml | head -1)
echo "=== [5/5] Export $SCENE ($(date)) ==="
ns-export gaussian-splat --load-config "$CFG" --output-dir exports/$SCENE && echo "=== DONE $SCENE ($(date)) config: $CFG ==="

./scripts/make_splat_hq.sh /mnt/c/Users/roach/Downloads/IMG_6557.MOV IMG_6557 650 30000 || echo "=== IMG_6557 FAILED ==="
echo "=== ALL DONE ($(date)) ==="
CFG=$(ls -t outputs/IMG_655*/*/*/config.yml 2>/dev/null | head -1)
[ -n "$CFG" ] && exec ns-viewer --load-config "$CFG" --viewer.websocket-port 7007
