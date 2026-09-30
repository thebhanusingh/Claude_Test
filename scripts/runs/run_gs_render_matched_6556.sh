#!/bin/bash
# After run_gs_render_6556.sh finishes: relight IMG_6556 with HDRs rescaled to the learned
# envmap's mean radiance (envmaps/*_matched.hdr), so the comparison isn't dominated by exposure.
set -o pipefail
while pgrep -f "^/bin/bash ./run_gs_render_6556.sh" >/dev/null; do sleep 15; done
cd ~/Claude_Test/third_party/GaussianShader
source ~/miniforge3/bin/activate gaussian_shader
export CC=x86_64-conda-linux-gnu-gcc CXX=x86_64-conda-linux-gnu-g++ TORCH_CUDA_ARCH_LIST="8.9" CUDA_HOME=$CONDA_PREFIX
T=$CONDA_PREFIX/targets/x86_64-linux
export CPATH=$T/include LIBRARY_PATH=$T/lib:$T/lib/stubs:/usr/lib/wsl/lib:$CONDA_PREFIX/lib
BASE=~/Claude_Test/outputs_relightable/IMG_6556
FFMPEG=~/miniforge3/envs/gsplat/bin/ffmpeg
VIDS=~/Claude_Test/exports/IMG_6556_relightable

for env in studio_small_08 kloppenheim_06; do
  M=${BASE}_relit_${env}_matched
  mkdir -p "$M/brdf_mlp/iteration_30000"
  for f in point_cloud cameras.json input.ply cfg_args; do ln -sfn "$BASE/$f" "$M/$f"; done
  cp ~/Claude_Test/envmaps/${env}_1k_matched.hdr "$M/brdf_mlp/iteration_30000/brdf_mlp.hdr"
  echo "=== Rendering relit_${env}_matched ($(date +%T)) ==="
  python render.py -m "$M" --brdf_dim 0 --sh_degree -1 --brdf_mode envmap --brdf_env 512 --iteration 30000 --skip_test \
    || { echo "=== RENDER FAILED ${env}_matched ==="; exit 1; }
  $FFMPEG -y -loglevel error -framerate 30 -i "$M/train/ours_30000/renders/%05d.png" \
    -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -crf 18 -pix_fmt yuv420p "$VIDS/relit_${env}_matched.mp4" && echo "video: $VIDS/relit_${env}_matched.mp4"
done

echo "=== Side-by-side matched video ($(date +%T)) ==="
$FFMPEG -y -loglevel error -i "$VIDS/learned_lighting.mp4" -i "$VIDS/relit_studio_small_08_matched.mp4" -i "$VIDS/relit_kloppenheim_06_matched.mp4" \
  -filter_complex "hstack=inputs=3" -c:v libx264 -crf 20 -pix_fmt yuv420p "$VIDS/compare_matched_learned_studio_sunset.mp4" && echo "video: $VIDS/compare_matched_learned_studio_sunset.mp4"
echo "=== MATCHED RENDER DONE ($(date +%T)) ==="
