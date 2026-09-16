#!/usr/bin/env bash
# Turn a video into a trained, exported Gaussian Splat.
#
# Pipeline: video -> frame extraction + COLMAP camera poses (ns-process-data)
#           -> Gaussian Splatting training (ns-train splatfacto)
#           -> export to a .ply splat file (ns-export gaussian-splat)
#
# Usage:
#   ./scripts/make_splat.sh <video_path> [scene_name] [max_num_iterations]
#
# Examples:
#   ./scripts/make_splat.sh ~/videos/room.mp4
#   ./scripts/make_splat.sh ~/videos/room.mp4 living_room 30000
#
# Env overrides:
#   NUM_FRAMES   Target number of frames to extract from the video (default 300)

set -euo pipefail

VIDEO_PATH="${1:?Usage: make_splat.sh <video_path> [scene_name] [max_num_iterations]}"
SCENE_NAME="${2:-$(basename "${VIDEO_PATH%.*}")}"
MAX_ITERS="${3:-30000}"
NUM_FRAMES="${NUM_FRAMES:-300}"

if [[ ! -f "$VIDEO_PATH" ]]; then
  echo "ERROR: video not found: $VIDEO_PATH" >&2
  exit 1
fi

for cmd in ns-process-data ns-train ns-export colmap ffmpeg; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: '$cmd' not found on PATH. Did you run scripts/setup_env.sh and 'conda activate gsplat'?" >&2
    exit 1
  fi
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_ROOT/data/$SCENE_NAME"
OUTPUT_DIR="$REPO_ROOT/outputs"
EXPORT_DIR="$REPO_ROOT/exports/$SCENE_NAME"

echo "=== [1/3] Extracting frames + estimating camera poses with COLMAP ==="
mkdir -p "$DATA_DIR"
ns-process-data video \
  --data "$VIDEO_PATH" \
  --output-dir "$DATA_DIR" \
  --num-frames-target "$NUM_FRAMES"

echo "=== [2/3] Training Gaussian Splat (splatfacto) for $MAX_ITERS iterations ==="
# TORCHDYNAMO_DISABLE avoids a torch.compile/inductor crash on some setups where
# a pip-bundled nvcc binary isn't marked executable (PermissionError: 'nvcc'),
# which otherwise masks the real underlying compile error. splatfacto runs fine
# in eager mode, just somewhat slower per-iteration.
TORCHDYNAMO_DISABLE=1 ns-train splatfacto \
  --data "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --experiment-name "$SCENE_NAME" \
  --max-num-iterations "$MAX_ITERS" \
  --viewer.quit-on-train-completion True

CONFIG_PATH="$(find "$OUTPUT_DIR/$SCENE_NAME/splatfacto" -name config.yml -print0 \
  | xargs -0 ls -t | head -n1)"
if [[ -z "$CONFIG_PATH" ]]; then
  echo "ERROR: could not locate trained config.yml under $OUTPUT_DIR/$SCENE_NAME/splatfacto" >&2
  exit 1
fi
echo "Trained config: $CONFIG_PATH"

echo "=== [3/3] Exporting Gaussian Splat to PLY ==="
mkdir -p "$EXPORT_DIR"
ns-export gaussian-splat \
  --load-config "$CONFIG_PATH" \
  --output-dir "$EXPORT_DIR"

echo ""
echo "Done. Splat exported to: $EXPORT_DIR"
echo "View it by dragging the .ply file into a web viewer such as https://playcanvas.com/supersplat/editor"
echo "or re-open the nerfstudio viewer with:"
echo "  ns-viewer --load-config \"$CONFIG_PATH\""
