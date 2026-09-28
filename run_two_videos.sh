#!/bin/bash
# Train IMG_6556 then IMG_6557 one after the other (RAM can't fit both at once).
cd ~/Claude_Test
for v in IMG_6556 IMG_6557; do
  ./scripts/make_splat_hq.sh /mnt/c/Users/roach/Downloads/$v.MOV $v 650 30000 || echo "=== $v FAILED ==="
done
echo "=== ALL DONE ($(date)) ==="
CFG=$(ls -t outputs/IMG_655*/*/*/config.yml 2>/dev/null | head -1)
[ -n "$CFG" ] && exec ~/miniforge3/envs/gsplat/bin/ns-viewer --load-config "$CFG" --viewer.websocket-port 7007
