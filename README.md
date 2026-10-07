# Claude_Test

## Video → Gaussian Splat pipeline

Turns a video into a trained [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) scene,
using [nerfstudio](https://docs.nerf.studio/) (`ns-process-data`, `ns-train splatfacto`, `ns-export`) on top of COLMAP
for structure-from-motion.

**This requires a machine with an NVIDIA GPU** (CUDA). It will not run in this cloud session — clone this repo and run
it locally, e.g. on your university machine (Linux, or Windows via WSL2).

**Start here: [`docs/quickstart.md`](docs/quickstart.md)** — the complete, ordered, copy-paste-ready path from
"I have a video" to "viewing the splat in Blender," including video capture tips, one-time machine setup (WSL2 on
Windows, conda, CUDA toolkit), and the actual run commands with the settings that worked. Written so a fresh run
doesn't need to be re-derived from scratch each time.

Other docs:
- [`docs/lab-setup.md`](docs/lab-setup.md) — working across a university lab machine, and getting the splat into
  Blender/Unreal Engine, without admin rights.
- [`docs/troubleshooting.md`](docs/troubleshooting.md) — every issue hit setting this up from scratch (WSL/conda/
  ffmpeg/COLMAP/nvcc/PyTorch issues, GitHub auth, keeping a long training run alive), with the actual fix for each.

### Quick reference

```bash
git clone <this-repo> && cd Claude_Test
./scripts/setup_env.sh                                       # once per machine
conda activate gsplat
MAX_JOBS=2 ./scripts/make_splat.sh /path/to/video.mp4 my_scene   # generates exports/my_scene/splat.ply
```
See `docs/quickstart.md` for what each step does, WSL2/Windows setup, and why `MAX_JOBS=2`.

Everything is on `main`: `git clone https://github.com/thebhanusingh/Claude_Test && cd Claude_Test`.

**Quality settings.** `make_splat.sh` defaults are tuned for an 8 GB laptop GPU. Override them with env vars:
`METHOD=splatfacto-big` (needs about 16 GB+ VRAM at 1080p), `MATCHING=exhaustive|sequential|vocab_tree`, and
`DOWNSCALE=1`. Without `DOWNSCALE`, nerfstudio **silently halves any image wider than 1600 px**, so 1080p video trains
at 960x540. For the sharpest results use **`scripts/make_splat_hq.sh`**: it picks the sharpest frames, handles split
COLMAP reconstructions, trains splatfacto-big at full resolution with pose refinement, and keeps COLMAP's coordinates.
Use `STOP_SPLIT=15000` or higher on 16 GB+ GPUs.

### Repo layout

```
CLAUDE.md                       quick reference for Claude Code sessions (auto-loaded)
environment.yml                 conda environment spec (ffmpeg, colmap, libusb; torch installed by setup_env.sh)
scripts/setup_env.sh            one-time environment setup (+ nerfstudio patches for PyTorch 2.6+ and COLMAP 3.12+)
scripts/make_splat.sh           video -> trained, exported Gaussian Splat (8 GB defaults, env overrides)
scripts/make_splat_hq.sh        high-quality pipeline (sharpest frames, largest COLMAP model, full res)
scripts/select_sharp_frames.py  sharpest frame per window of a video
scripts/fix_colmap_model.py     rebuild transforms.json from the largest COLMAP sub-model, in COLMAP coordinates
scripts/watch_*.sh              progress/crash watchers for long runs
scripts/runs/                   exact one-off run scripts behind each result in docs/lab-notes.md
docs/quickstart.md              step-by-step runbook
docs/troubleshooting.md         setup problems and fixes
docs/lab-notes.md               full log of runs, numbers and problems (Sept 2026)
docs/session-report.md          narrative summary of the Sept 2026 sessions
data/, outputs/, exports/       generated at runtime, gitignored
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
| Fits this repo's pipeline | **Partly** — `scripts/train_relightable_splat.sh` wires it to `scripts/make_splat.sh`'s COLMAP output, but the images must be undistorted first (see below) | Not directly — vendored for reference only; you'd need to build the neilfpp-style preprocessing yourself |
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

**What actually happened when we ran it (Sept 2026, see `docs/lab-notes.md` problem 13):**
- GaussianShader's own `environment.yml` is a 2022 snapshot (Python 3.7, torch 1.10) and **can't be created**.
  The working recipe builds a modern env (Python 3.10, torch 2.4+cu124, conda `cuda-toolkit` and gcc 12, `libusb`) and
  sets `CPATH`/`LIBRARY_PATH` to the conda CUDA folders. See `scripts/runs/run_gs_6556.sh`. Blackwell GPUs need a
  CUDA 12.8+ torch instead.
- It only accepts **undistorted PINHOLE** cameras, so run `colmap image_undistorter` first. `train_relightable_splat.sh`
  doesn't do this yet.
- On the 8 GB RTX 4070 Laptop it worked at half resolution (`-r 2 --data_device cpu`) with reduced densification
  (`--densify_grad_threshold 0.0004 --densify_until_iter 7000`): about 3.1 GB VRAM, 3 h 12 min for 30k steps. It has
  **no resumable checkpoints**.
- **It relights only reflections.** Its base colour isn't lit by the environment map, so sun, shade and cast shadows
  stay baked in. On an outdoor grass scene a new HDR produced haze and purple streaks. It suits shiny objects, not
  outdoor delighting. See `docs/session-report.md` for alternatives (DiffusionRenderer, intrinsic 3DGS, GaRe).
