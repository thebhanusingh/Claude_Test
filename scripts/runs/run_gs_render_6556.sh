#!/bin/bash
# Render the GaussianShader IMG_6556 model with its learned lighting and relit under new HDR
# environment maps (swap brdf_mlp/iteration_30000/brdf_mlp.hdr), then make MP4 flythroughs.
set -o pipefail
cd ~/Claude_Test/third_party/GaussianShader
source ~/miniforge3/bin/activate gaussian_shader
export CC=x86_64-conda-linux-gnu-gcc CXX=x86_64-conda-linux-gnu-g++ TORCH_CUDA_ARCH_LIST="8.9" CUDA_HOME=$CONDA_PREFIX
T=$CONDA_PREFIX/targets/x86_64-linux
export CPATH=$T/include LIBRARY_PATH=$T/lib:$T/lib/stubs:/usr/lib/wsl/lib:$CONDA_PREFIX/lib
BASE=~/Claude_Test/outputs_relightable/IMG_6556
FFMPEG=~/miniforge3/envs/gsplat/bin/ffmpeg
VIDS=~/Claude_Test/exports/IMG_6556_relightable
mkdir -p "$VIDS"

render() {  # $1 = model dir, $2 = label
  echo "=== Rendering $2 ($(date +%T)) ==="
  python render.py -m "$1" --brdf_dim 0 --sh_degree -1 --brdf_mode envmap --brdf_env 512 --iteration 30000 --skip_test \
    || { echo "=== RENDER FAILED $2 ==="; exit 1; }
  $FFMPEG -y -loglevel error -framerate 30 -i "$1/train/ours_30000/renders/%05d.png" \
    -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -crf 18 -pix_fmt yuv420p "$VIDS/$2.mp4" && echo "video: $VIDS/$2.mp4"
}

render "$BASE" learned_lighting

for env in studio_small_08 kloppenheim_06; do
  M=${BASE}_relit_$env
  mkdir -p "$M/brdf_mlp/iteration_30000"
  for f in point_cloud cameras.json input.ply cfg_args; do ln -sfn "$BASE/$f" "$M/$f"; done
  cp ~/Claude_Test/envmaps/${env}_1k.hdr "$M/brdf_mlp/iteration_30000/brdf_mlp.hdr"
  render "$M" "relit_$env"
done

echo "=== Side-by-side video ($(date +%T)) ==="
$FFMPEG -y -loglevel error -i "$VIDS/learned_lighting.mp4" -i "$VIDS/relit_studio_small_08.mp4" -i "$VIDS/relit_kloppenheim_06.mp4" \
  -filter_complex "hstack=inputs=3" -c:v libx264 -crf 20 -pix_fmt yuv420p "$VIDS/compare_learned_studio_sunset.mp4" && echo "video: $VIDS/compare_learned_studio_sunset.mp4"
echo "=== RENDER DONE ($(date +%T)) ==="
