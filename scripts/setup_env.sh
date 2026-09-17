#!/usr/bin/env bash
# One-time environment setup for the video -> Gaussian Splat pipeline.
# Written to work across different lab machines/GPUs (laptop Ada cards up
# through workstation Blackwell cards) without editing this script per machine.
#
# Requirements before running this:
#   - Linux, or Windows via WSL2 (nerfstudio/COLMAP do not run natively on Windows)
#   - An NVIDIA GPU with a driver already installed (usually pre-installed by IT
#     on lab machines -- do not try to install/replace the driver yourself)
#   - conda or mamba installed in your OWN user space (no admin rights needed:
#     https://github.com/conda-forge/miniforge)
#
# Usage:
#   ./scripts/setup_env.sh

set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. An NVIDIA GPU + driver is required to train a Gaussian Splat." >&2
  exit 1
fi

echo "==> Detected GPU(s):"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader

DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1 | cut -d. -f1)"
if [[ "$DRIVER_MAJOR" -lt 570 ]]; then
  echo ""
  echo "WARNING: driver version is $DRIVER_MAJOR, below the R570 minimum for Blackwell GPUs" >&2
  echo "         (RTX PRO 4500/5000/6000 Blackwell, RTX 50-series). Older GPUs generally still" >&2
  echo "         work fine on this driver, but Blackwell cards will fail to run CUDA workloads." >&2
  echo "         On a lab machine, ask IT to update the driver -- don't attempt it yourself" >&2
  echo "         without permission, it typically needs admin rights and can affect other users." >&2
  echo ""
fi

CONDA_BIN="conda"
if command -v mamba >/dev/null 2>&1; then
  CONDA_BIN="mamba"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Creating conda environment 'gsplat' from environment.yml (this can take a while)"
"$CONDA_BIN" env create -f "$REPO_ROOT/environment.yml" || "$CONDA_BIN" env update -f "$REPO_ROOT/environment.yml"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gsplat
pip install --upgrade pip

echo "==> Installing nerfstudio (provides ns-process-data, ns-train splatfacto, ns-export gaussian-splat)"
pip install nerfstudio

echo "==> Installing PyTorch with CUDA 12.8 wheels (overrides whatever nerfstudio pulled in above)"
echo "    This is required for Blackwell (sm_120) GPUs; it also runs fine on older Ada/Ampere cards"
echo "    as long as the driver is reasonably current (see warning above if any)."
pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "==> Patching nerfstudio checkpoint loading for PyTorch 2.6+ weights_only default"
# PyTorch 2.6 changed torch.load's default weights_only from False to True, which
# blocks unpickling the numpy scalar type nerfstudio's checkpoints use, breaking
# ns-export/ns-eval with "Unsupported global: GLOBAL numpy.core.multiarray.scalar".
# Safe here since we're only ever loading our own freshly-trained checkpoints.
EVAL_UTILS="$CONDA_PREFIX/lib/python3.10/site-packages/nerfstudio/utils/eval_utils.py"
if [[ -f "$EVAL_UTILS" ]]; then
  sed -i 's/torch\.load(load_path, map_location="cpu")/torch.load(load_path, map_location="cpu", weights_only=False)/' "$EVAL_UTILS"
fi

echo "==> Ensuring any pip-bundled nvcc binaries are executable"
# gsplat (nerfstudio's splatfacto rasterizer) and torch.compile both shell out to
# 'nvcc' at runtime. The nvcc bundled by the 'cuda-toolkit'/'nvidia-cuda-nvcc-cu12'
# pip packages has been observed installed without the executable bit set, causing
# "PermissionError: [Errno 13] Permission denied: 'nvcc'" the first time anything
# tries to run it. Fix proactively rather than waiting to hit it mid-training.
find "$CONDA_PREFIX" -type f -name nvcc -exec chmod +x {} \; 2>/dev/null || true

echo "==> Verifying GPU is visible to PyTorch"
python -c "
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
    print('compute capability:', torch.cuda.get_device_capability(0))
"

echo ""
echo "Setup complete. Next time, activate the environment with:"
echo "  conda activate gsplat"
echo "Then run:"
echo "  ./scripts/make_splat.sh /path/to/video.mp4 my_scene"
echo ""
echo "NOTE: if the ns-viewer or ns-train crashes with a torch/CUDA error despite the check above"
echo "      passing, this is a known rough edge on very new (Blackwell) GPUs with nerfstudio's"
echo "      own pinned dependencies -- see https://github.com/nerfstudio-project/nerfstudio/issues/3732"
