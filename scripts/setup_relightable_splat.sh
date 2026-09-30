#!/usr/bin/env bash
# Sets up BRDF/relightable Gaussian Splatting codebases as third-party dependencies.
#
# Two options are cloned side by side (each needs its OWN conda env — their pinned
# dependencies conflict with each other and with the base gsplat/nerfstudio env):
#
#   gaussianshader          (recommended default) "GaussianShader: 3D Gaussian
#                            Splatting with Shading Functions for Reflective
#                            Surfaces". Trains on plain COLMAP data with the same
#                            -s <path> convention as the original 3DGS repo, so it
#                            plugs straight into the data scripts/make_splat.sh
#                            already produces.
#                            https://github.com/Asparagus15/GaussianShader
#
#   relightable3dgaussian    "Relightable 3D Gaussian: Real-time Point Cloud
#                            Relighting with BRDF Decomposition and Ray Tracing"
#                            (NJU-3DV, ECCV 2024) -- the closest name/paper match
#                            to "BRDF relightable Gaussian Splat", but it expects a
#                            custom "neilfpp-like" dataset (depth/normal/mask maps
#                            + sfm_scene.json), and its own README states a
#                            video/custom-data prep script was not yet released as
#                            of last check. Reference hardware: a single RTX 3090
#                            (24GB). Vendored here for reference/future use, not
#                            wired into scripts/make_splat.sh's output.
#                            https://github.com/NJU-3DV/Relightable3DGaussian
#
# Both are research code with non-commercial / research-only licenses in places
# (e.g. the CUDA rasterizer submodules) -- check each repo's LICENSE before any
# commercial use.
#
# Usage:
#   ./scripts/setup_relightable_splat.sh [gaussianshader|relightable3dgaussian|all]

set -euo pipefail

WHAT="${1:-all}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$REPO_ROOT/third_party"
mkdir -p "$THIRD_PARTY"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. An NVIDIA GPU is required." >&2
  exit 1
fi

CONDA_BIN="conda"
command -v mamba >/dev/null 2>&1 && CONDA_BIN="mamba"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

setup_gaussianshader() {
  echo "=== Setting up GaussianShader ==="
  if [[ ! -d "$THIRD_PARTY/GaussianShader" ]]; then
    git clone https://github.com/Asparagus15/GaussianShader.git "$THIRD_PARTY/GaussianShader"
  fi
  cd "$THIRD_PARTY/GaussianShader"
  git submodule update --init --recursive
  "$CONDA_BIN" env create -f environment.yml || "$CONDA_BIN" env update -f environment.yml
  echo "GaussianShader ready. Activate with: conda activate gaussian_shader"
}

setup_relightable3dgaussian() {
  echo "=== Setting up Relightable3DGaussian (NJU-3DV) ==="
  if [[ ! -d "$THIRD_PARTY/Relightable3DGaussian" ]]; then
    git clone https://github.com/NJU-3DV/Relightable3DGaussian.git "$THIRD_PARTY/Relightable3DGaussian"
  fi
  cd "$THIRD_PARTY/Relightable3DGaussian"
  git submodule update --init --recursive
  "$CONDA_BIN" env create -f environment.yml || "$CONDA_BIN" env update -f environment.yml
  conda activate r3dg
  conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
  pip install torch_scatter==2.1.1 kornia==0.6.12
  pip install ./submodules/simple-knn ./bvh ./r3dg-rasterization
  if [[ ! -d "$THIRD_PARTY/nvdiffrast" ]]; then
    git clone https://github.com/NVlabs/nvdiffrast "$THIRD_PARTY/nvdiffrast"
  fi
  pip install "$THIRD_PARTY/nvdiffrast"
  echo "Relightable3DGaussian ready. Activate with: conda activate r3dg"
  echo "NOTE: expects a neilfpp-like dataset, not plain COLMAP output -- see its README."
}

case "$WHAT" in
  gaussianshader) setup_gaussianshader ;;
  relightable3dgaussian) setup_relightable3dgaussian ;;
  all) setup_gaussianshader; setup_relightable3dgaussian ;;
  *) echo "Usage: $0 [gaussianshader|relightable3dgaussian|all]" >&2; exit 1 ;;
esac
