#!/bin/bash
# GS-IR on IMG_6556: stage 1 (3DGS + normals) -> baking (occlusion) -> stage 2 (PBR decomposition) -> export.
set -o pipefail
source ~/miniforge3/etc/profile.d/conda.sh; conda activate gsir; cd ~/GS-IR
SRC=~/Claude_Test/data_gaussianshader/IMG_6556
OUT=~/Claude_Test/outputs_gsir/IMG_6556
EXP=~/Claude_Test/exports/IMG_6556_gsir
BOUND=6.0   # all 650 cameras within 4.63 of origin; ~92% of SfM points within 6
COMMON="-s $SRC -m $OUT -r 2 --data_device cpu --eval"
mkdir -p $OUT $EXP
BAKE_RES=${BAKE_RES:-256}
# variable-size Gaussian subsets per bake cell fragment the CUDA cache; this keeps VRAM from creeping to 8 GB
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log(){ echo "=== $* ($(date '+%F %T')) ==="; }
run(){ echo "+ $*"; "$@" 2>&1 | tee $OUT/last_stage.log; }
oom(){ grep -qiE "out of memory|CUDA error: out" $OUT/last_stage.log; }

if [ -z "$SKIP_STAGE1" ]; then
log STAGE1 start
run python train.py $COMMON --iterations 30000 --densify_grad_threshold 0.0004 --densify_until_iter 7000
if [ $? -ne 0 ]; then
  if oom; then log STAGE1 OOM, retry with densify_grad_threshold 0.0004 until 7000
    run python train.py $COMMON --iterations 30000 --densify_grad_threshold 0.0004 --densify_until_iter 7000 || { log STAGE1 FAILED; exit 1; }
  else log STAGE1 FAILED; exit 1; fi
fi
log STAGE1 done
else log STAGE1 skipped, using existing $OUT/chkpnt30000.pth; fi

log BAKING start res $BAKE_RES
run python baking.py -m $OUT --checkpoint $OUT/chkpnt30000.pth --bound $BOUND --valid $BOUND --occlu_res $BAKE_RES --occlusion 0.4 --cubemap_res 256
if [ $? -ne 0 ]; then
  if oom; then log BAKING OOM, retry res 192
    run python baking.py -m $OUT --checkpoint $OUT/chkpnt30000.pth --bound $BOUND --valid $BOUND --occlu_res 192 --occlusion 0.4 --cubemap_res 256 || { log BAKING FAILED; exit 1; }
  else log BAKING FAILED; exit 1; fi
fi
log BAKING done

log STAGE2 start
run python train.py $COMMON --start_checkpoint $OUT/chkpnt30000.pth --iterations 40000 --bound $BOUND --indirect --gamma || { log STAGE2 FAILED; exit 1; }
log STAGE2 done

log EXPORT start
run python ~/Claude_Test/scripts/gsir_export_albedo.py $OUT/chkpnt40000.pth $EXP/splat_albedo.ply || { log EXPORT FAILED; exit 1; }
run python render.py $COMMON --checkpoint $OUT/chkpnt40000.pth --skip_train --pbr --indirect --gamma || { log RENDER FAILED; exit 1; }
log EXPORT done
log ALL DONE
