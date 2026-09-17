# Claude_Test

## Video → Gaussian Splat pipeline

Turns a video into a trained [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) scene,
using [nerfstudio](https://docs.nerf.studio/) (`ns-process-data`, `ns-train splatfacto`, `ns-export`) on top of COLMAP
for structure-from-motion.

**This requires a machine with an NVIDIA GPU** (CUDA). It will not run in this cloud session — clone this repo and run
it locally, e.g. on your university machine (Linux, or Windows via WSL2).

Working across a university lab machine and also want this splat in Blender/Unreal Engine, without admin rights?
See [`docs/lab-setup.md`](docs/lab-setup.md).

Hit an error running any of this (WSL/conda/ffmpeg/COLMAP/nvcc issues, GitHub auth, keeping a long training run
alive)? Check [`docs/troubleshooting.md`](docs/troubleshooting.md) first — it covers every issue hit setting this
up from scratch, with the actual fix for each.

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

## Relightable (BRDF) Gaussian Splat

Beyond a plain splat, there's ongoing research on Gaussian Splats that decompose each Gaussian into
material/BRDF parameters so the result can be **relit** under new lighting after training, instead of
being baked under the original video's lighting. Two vendored options, set up with:

```bash
./scripts/setup_relightable_splat.sh [gaussianshader|relightable3dgaussian|all]
```

| | [GaussianShader](https://github.com/Asparagus15/GaussianShader) (recommended) | [Relightable3DGaussian](https://github.com/NJU-3DV/Relightable3DGaussian) (NJU-3DV, ECCV 2024) |
|---|---|---|
| Paper | "3D Gaussian Splatting with Shading Functions for Reflective Surfaces" | "Relightable 3D Gaussian: Real-time Point Cloud Relighting with BRDF Decomposition and Ray Tracing" — closest name/paper match to "BRDF relightable Gaussian Splat" |
| Input data | Plain COLMAP output (`images/` + `sparse/0/*.bin`) — same convention as the base 3DGS repo | Custom "neilfpp-like" dataset: images + depth + normal + object-mask maps + `sfm_scene.json`. The authors' own README says a video/custom-capture prep script was not yet released as of last check |
| Fits this repo's pipeline | **Yes** — `scripts/train_relightable_splat.sh` wires it directly to `scripts/make_splat.sh`'s COLMAP output | Not directly — vendored for reference only; you'd need to build the neilfpp-style preprocessing yourself |
| Reference GPU | Not stated | Single RTX 3090, **24GB VRAM** |
| Relighting | Render under a new environment map (`render.py ... --brdf_mode envmap`) | Full BRDF decomposition + ray-traced relighting/shadows, more physically complete but heavier |

Both are research code, not production libraries — check each repo's `LICENSE` before any commercial use
(the CUDA rasterizer submodules in this space are commonly non-commercial/research-only licensed).

### Train a relightable splat (GaussianShader)

Run `scripts/make_splat.sh` first (need its COLMAP output, not the trained nerfstudio model), then:

```bash
./scripts/setup_relightable_splat.sh gaussianshader
conda activate gaussian_shader
./scripts/train_relightable_splat.sh my_scene
```

This symlinks `data/my_scene/{images,colmap/sparse/0}` into the layout GaussianShader's data loader
expects, then runs its `train.py`. Output lands in `outputs_relightable/my_scene/`; relight it with the
`render.py --brdf_mode envmap` command it prints at the end.

**VRAM note:** an 8GB laptop GPU (e.g. RTX 4070 Laptop) is untested territory for this method — no VRAM
figure is published. If it OOMs, reduce `NUM_FRAMES`/resolution in the earlier COLMAP step, or try
`--resolution` downscaling flags in GaussianShader's `train.py --help`.
