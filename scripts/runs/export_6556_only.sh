#!/bin/bash
# Wait for IMG_6556 training (pid 161049) to finish, then export it. Does NOT start IMG_6557.
cd ~/Claude_Test; source ~/miniforge3/bin/activate gsplat
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
while kill -0 161049 2>/dev/null; do sleep 10; done
CFG=$(ls -t outputs/IMG_6556/*/*/config.yml | head -1)
echo "=== [5/5] Export IMG_6556 ($(date)) from $CFG ==="
ns-export gaussian-splat --load-config "$CFG" --output-dir exports/IMG_6556 && echo "=== DONE IMG_6556 ($(date)) ===" || echo "=== EXPORT FAILED ==="
