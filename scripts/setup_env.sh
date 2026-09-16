#!/usr/bin/env bash
# One-time environment setup for the video -> Gaussian Splat pipeline.
#
# Requirements before running this:
#   - Linux, or Windows via WSL2 (nerfstudio/COLMAP do not run natively on Windows)
#   - An NVIDIA GPU with a recent driver (check with `nvidia-smi`)
#   - conda or mamba installed (https://docs.conda.io/en/latest/miniconda.html)
#
# Usage:
#   ./scripts/setup_env.sh

set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. An NVIDIA GPU + driver is required to train a Gaussian Splat." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

CONDA_BIN="conda"
if command -v mamba >/dev/null 2>&1; then
  CONDA_BIN="mamba"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Creating conda environment 'gsplat' from environment.yml (this can take a while)"
"$CONDA_BIN" env create -f "$REPO_ROOT/environment.yml" || "$CONDA_BIN" env update -f "$REPO_ROOT/environment.yml"

echo "==> Installing nerfstudio (provides ns-process-data, ns-train splatfacto, ns-export gaussian-splat)"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gsplat
pip install --upgrade pip
pip install nerfstudio

echo ""
echo "Setup complete. Next time, activate the environment with:"
echo "  conda activate gsplat"
echo "Then run:"
echo "  ./scripts/make_splat.sh /path/to/video.mp4 my_scene"
