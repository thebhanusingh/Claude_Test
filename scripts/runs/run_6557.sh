#!/bin/bash
# Wait for the GaussianShader matched render to finish, then run IMG_6557 through the HQ pipeline.
cd ~/Claude_Test
while pgrep -f "^/bin/bash ./run_gs_render_matched_6556.sh" >/dev/null; do sleep 15; done
echo "=== GPU free, starting IMG_6557 ($(date)) ==="
./scripts/make_splat_hq.sh /mnt/c/Users/roach/Downloads/IMG_6557.MOV IMG_6557 650 30000 || { echo "=== IMG_6557 FAILED ==="; exit 1; }
echo "=== ALL DONE ($(date)) ==="
