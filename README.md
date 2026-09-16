# Claude_Test

## Video → Gaussian Splat pipeline

Turns a video into a trained [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) scene,
using [nerfstudio](https://docs.nerf.studio/) (`ns-process-data`, `ns-train splatfacto`, `ns-export`) on top of COLMAP
for structure-from-motion.

**This requires a machine with an NVIDIA GPU** (CUDA). It will not run in this cloud session — clone this repo and run
it locally, e.g. on your university machine (Linux, or Windows via WSL2).

### 1. Set up the environment (once)

Requires [conda](https://docs.conda.io/en/latest/miniconda.html) (or mamba) and an NVIDIA driver already installed.

```bash
git clone <this-repo>
cd Claude_Test
./scripts/setup_env.sh
```

This creates a `gsplat` conda environment with PyTorch (CUDA 11.8), ffmpeg, COLMAP, and nerfstudio.

### 2. Generate a splat from a video

```bash
conda activate gsplat
./scripts/make_splat.sh /path/to/video.mp4 my_scene
```

What it does:
1. **`ns-process-data video`** — extracts frames from the video and runs COLMAP to estimate camera poses
   and a sparse point cloud (`data/my_scene/`).
2. **`ns-train splatfacto`** — trains the Gaussian Splat (default 30,000 iterations; pass a third argument
   to override, e.g. `./scripts/make_splat.sh video.mp4 my_scene 15000`). A live preview is served at
   `http://localhost:7007` while training runs.
3. **`ns-export gaussian-splat`** — exports the trained splat to a `.ply` file in `exports/my_scene/`.

### 3. View the result

- During/after training: `ns-viewer --load-config outputs/my_scene/splatfacto/<timestamp>/config.yml`
- The exported `.ply`: drag it into a web viewer such as [SuperSplat](https://playcanvas.com/supersplat/editor).

### Tips for good source video

- Move slowly around the subject, overlapping each frame with the last (orbit the object/room rather than
  panning past it once).
- Good, even lighting; avoid strong motion blur.
- 20–60 seconds of footage is usually plenty; `NUM_FRAMES` (env var, default 300) controls how many frames
  are sampled from it, e.g. `NUM_FRAMES=500 ./scripts/make_splat.sh video.mp4 my_scene`.
- If COLMAP fails to register most images, the camera motion/overlap is usually the issue — reshoot with
  more overlap between frames.

### Repo layout

```
environment.yml        conda environment spec (torch/cuda, ffmpeg, colmap)
scripts/setup_env.sh    one-time environment setup
scripts/make_splat.sh   video -> trained, exported Gaussian Splat
data/, outputs/, exports/   generated at runtime, gitignored
```
