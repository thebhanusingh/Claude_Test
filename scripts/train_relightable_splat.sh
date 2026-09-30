#!/usr/bin/env bash
# Train a BRDF-shaded, relightable Gaussian Splat (GaussianShader) from the COLMAP
# data already produced by scripts/make_splat.sh.
#
# Usage:
#   ./scripts/train_relightable_splat.sh <scene_name> [iterations]
#
# Prereqs:
#   ./scripts/make_splat.sh <video> <scene_name> already run
#     (so data/<scene_name>/{images,colmap/sparse/0} exists)
#   ./scripts/setup_relightable_splat.sh gaussianshader already run

set -euo pipefail

SCENE_NAME="${1:?Usage: train_relightable_splat.sh <scene_name> [iterations]}"
EXTRA_ITER_ARGS=()
[[ -n "${2:-}" ]] && EXTRA_ITER_ARGS=(--iterations "$2")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COLMAP_DATA_DIR="$REPO_ROOT/data/$SCENE_NAME"
GS_REPO="$REPO_ROOT/third_party/GaussianShader"
GS_DATA_DIR="$REPO_ROOT/data_gaussianshader/$SCENE_NAME"
OUTPUT_DIR="$REPO_ROOT/outputs_relightable/$SCENE_NAME"

if [[ ! -d "$COLMAP_DATA_DIR/colmap/sparse/0" ]]; then
  echo "ERROR: $COLMAP_DATA_DIR/colmap/sparse/0 not found. Run scripts/make_splat.sh first." >&2
  exit 1
fi
if [[ ! -d "$GS_REPO" ]]; then
  echo "ERROR: $GS_REPO not found. Run scripts/setup_relightable_splat.sh gaussianshader first." >&2
  exit 1
fi

# GaussianShader (built on the original 3DGS data loader) expects:
#   <source_path>/images
#   <source_path>/sparse/0/{cameras.bin,images.bin,points3D.bin}
# nerfstudio's ns-process-data nests the model one level deeper, under
# <data_dir>/colmap/sparse/0 -- rebuild the expected layout with symlinks
# instead of copying (COLMAP data can be large).
mkdir -p "$GS_DATA_DIR/sparse"
ln -sfn "$COLMAP_DATA_DIR/images" "$GS_DATA_DIR/images"
ln -sfn "$COLMAP_DATA_DIR/colmap/sparse/0" "$GS_DATA_DIR/sparse/0"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gaussian_shader

cd "$GS_REPO"
python train.py \
  -s "$GS_DATA_DIR" \
  --eval \
  -m "$OUTPUT_DIR" \
  -w \
  --brdf_dim 0 \
  --sh_degree -1 \
  --lambda_predicted_normal 2e-1 \
  --brdf_env 512 \
  "${EXTRA_ITER_ARGS[@]}"

echo ""
echo "Done. Trained relightable splat in: $OUTPUT_DIR"
echo "Relight it under a new environment map with:"
echo "  python render.py -m \"$OUTPUT_DIR\" --brdf_dim 0 --sh_degree -1 --brdf_mode envmap --brdf_env 512"
