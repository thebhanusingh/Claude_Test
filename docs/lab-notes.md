# Lab notes: changes, new tools, and problems found

Running log of everything changed or added on the home laptop (`Murph`, WSL2 Ubuntu 26.04,
RTX 4070 Laptop 8 GB, 15.5 GB RAM visible to WSL, 32 cores), mostly driven remotely via
Claude Remote Control. Newest entries at the bottom of each section. Use this to trace
problems back to when something changed.

## Machine / environment

| Date | Change | Notes |
|---|---|---|
| 2026-09-23 | Checked env | Claude Code 2.1.280, driver 610.47, system `nvcc` 12.4, PyTorch 2.11.0+cu128 in `gsplat` env. nvcc/PyTorch CUDA mismatch (12.4 vs 12.8) is the likely reason `torch.compile` had to be disabled (`TORCHDYNAMO_DISABLE=1`). |
| 2026-09-23 | Installed `cloudflared` 2026.9.1 to `~/.local/bin` | No sudo. Used for public "quick tunnel" links (random 4-word `*.trycloudflare.com` names, new each start). |
| 2026-09-23 | Keep-alive heartbeat `~/.local/bin/keepalive.sh` | Logs `viewer:up/DOWN tunnel:up/DOWN` to `~/keepalive.log` every 5 min. Only logs, never restarts anything. |
| 2026-09-23 | Windows keep-awake (hidden PowerShell loop calling `SetThreadExecutionState`) | Asks Windows not to sleep while it runs; changes no settings. Does NOT stop lid-close sleep. Could not verify with `powercfg /requests` (needs admin). |
| 2026-09-28 | Installed COLMAP 4.1.1 in a separate conda env `colmap4` (+ `openimageio=3.1`, which the first install was missing) | Has `global_mapper` (GLOMAP merged into COLMAP). Not used in any pipeline yet. Harmless warning: `libcusolver.so.12: no version information available`. |
| 2026-09-28 | Installed `ttyd` 1.7.7 to `~/.local/bin` | Read-only web terminal, see "Remote viewing". |
| 2026-09-29 | Installed GitHub CLI `gh` 2.101.0 to `~/.local/bin` | **Not logged in** (device login started, then cancelled at the user's request). To push: `gh auth login --web`, approve the code at github.com/login/device, then `git push`. 6+ local commits are unpushed. The repo is private. |
| 2026-09-28 | Cloned GaussianShader (commit de77861) to `third_party/GaussianShader`, created a **modern** conda env `gaussian_shader` | The repo's `environment.yml` is a 2022 freeze (Python 3.7, torch 1.10+cu111, local-only pip packages) and can't be created as is. Built instead: Python 3.10, torch 2.4.1+cu124, conda `cuda-toolkit=12.4`, conda gcc/g++ 12 (system gcc 15 is too new for CUDA 12.4), plus plyfile/tqdm/opencv/imageio/scipy/matplotlib/scikit-image/tensorboard/open3d, numpy<2. Log: `setup_gaussianshader.log`. |
| 2026-10-07 | Created conda env `gsir` for GS-IR (lzhnb/GS-IR, cloned to `~/GS-IR`) | Same recipe as `gaussian_shader`: Python 3.10, torch 2.4.1+cu124, conda `cuda-toolkit=12.4`, conda gcc 12, plus conda-forge `libusb` and build env vars (see "GS-IR shadow removal"). Existing envs untouched. Log `~/gsir_install.log`; failed first attempt kept as `~/gsir_install_attempt1.log`. |

## Scripts added (repo)

| File | Purpose |
|---|---|
| `scripts/select_sharp_frames.py` | Scores every video frame by Laplacian variance and keeps the sharpest frame per window (default 650). Replaces ffmpeg's evenly spaced extraction. |
| `scripts/fix_colmap_model.py` | Picks the COLMAP sub-model with the most images, bundle-adjusts it (incl. principal point), writes `transforms.json` + `sparse_pc.ply` in original COLMAP world coordinates. Original nerfstudio outputs kept as `*_nsprocess.*`. |
| `scripts/make_splat_hq.sh` | Full HQ pipeline: sharp frames -> `ns-process-data images` -> `fix_colmap_model.py` -> `splatfacto-big` (falls back to `splatfacto` on failure) -> export. Skips steps 1-3 if the scene already has poses. |
| `scripts/runs/*.sh` | One-off runners for specific scenes (see `scripts/runs/README.md`). **Moved out of the repo root on 2026-09-30.** The script names used in the entries below now live in that folder. |
| `scripts/watch_nerfstudio.sh` (was `watch_600f.sh`) | Watcher: status line every 15 s to `<log>_status.log`; stdout summary every 5 min, on stage change, errors, finish, or process disappearing. Args: `PID LOG [DONE_PATTERN]`. |
| `scripts/watch_gaussianshader.sh` (was `watch_gs.sh`) | Same idea for GaussianShader's tqdm progress bar. Args: `PID LOG TOTAL_ITERS`. |
| 2026-09-30 repo corrections | `setup_env.sh` patches nerfstudio for COLMAP >= 3.12 option names. `make_splat.sh` gained `METHOD` / `MATCHING` / `DOWNSCALE` overrides, a split-COLMAP check (refuses to train if fewer than 80% of frames are posed) and nerfstudio-viewer-first wording. `*.log` and local artifacts are gitignored. Docs updated for gcc 15 / conda CUDA headers / WSL admin. |
| `scripts/cluster_surfaces.py` (2026-10-12) | Stage 3 of the shadow-free splat plan: groups Gaussians into continuous surfaces from normals and neighbour links only (no colour), and gives each surface and each Gaussian a grain direction with a coherence score. Writes debug PLYs coloured by cluster and by grain angle. `--selftest` runs a synthetic ground + box + sphere scene. |
| `scripts/check_grain.py` (2026-10-12) | Validates stage 3 grain against the photos: projects each clustered Gaussian's local grain into 6 frames around the object (z-buffer, grazing views skipped) and compares it with the photo's structure-tensor texture direction, against rotated-90 and random baselines; `--pixels edges|interior`. Writes overlays + grain_check.json. Imports `fix_splat_shadows.py`. |

## Training runs

| Date | Scene | Settings | Result |
|---|---|---|---|
| 2026-09-16 | `my_scene` (IMG_6449, 313 frames) | splatfacto 30k | Finished (`2026-09-16_204930`). Five earlier runs that day have no checkpoint. |
| 2026-09-23 | `my_scene` retrain | splatfacto 30k | Stopped at 64 % to make room for the next run. |
| 2026-09-23 | `my_scene_600f` (IMG_6449, 659 frames) | splatfacto 7k, `cache-images cpu`, `stop-split-at 5000` | 1st try trained on only 2 frames (see Problem 1), stopped. Retry OK: 3.5 min, ~1 GB GPU, `exports/my_scene_600f/splat.ply` 92 MB. Useless 2-frame run: `outputs/my_scene_600f/splatfacto/2026-09-23_153700/`. |
| 2026-09-23 | `my_scene_600f_hq` | splatfacto 30k, full res (`--downscale-factor 1`), `cache-images cpu`, COLMAP coords (`--orientation-method none --center-method none --auto-scale-poses False`) | OK, 54 min, peak ~2.2 GB GPU, 10 GB RAM, GPU up to 81 °C. 595k Gaussians, `exports/my_scene_600f_hq/splat.ply` 148 MB. Aligned to COLMAP (median 0.058 units to nearest SfM point). |
| 2026-09-28 | `IMG_6556`, then `IMG_6557` (two separate scenes) | `make_splat_hq.sh`: 650 sharpest frames, splatfacto-big 30k, full res, `cache-images cpu`, pose optimisation `SO3xR3`, COLMAP coords | IMG_6556 COLMAP: all 650 frames posed in one model (13:39-14:29). 1st training try stopped for low RAM (Problem 4); restarted 14:35 without `--eval-mode all`. |

## Problems found and fixes

1. **COLMAP split the scene; nerfstudio used the wrong piece** (2026-09-23, `my_scene_600f`).
   `ns-process-data` always reads `colmap/sparse/0`, which held 2 images; `sparse/1` held all 659.
   It only printed "COLMAP only found poses for 0.30% of the images" and trained anyway.
   Fix: rebuild `transforms.json` from the largest model -> now automatic in `fix_colmap_model.py`.
   **Check:** always compare the frame count in `transforms.json` with the image count before training.

2. **Exported splat didn't line up with SfM points** (2026-09-23). Nerfstudio re-orients, re-centres
   and re-scales the scene by default. Fix: `keep_original_world_coordinate=True` when writing
   transforms + `--orientation-method none --center-method none --auto-scale-poses False`.
   Nerfstudio still records an axis swap in `dataparser_transforms.json`, but measurement showed the
   exported `.ply` already matches COLMAP coordinates. **Side effect:** the viewer's orbit controls feel
   tilted; use "Reset Up Direction" in the viewer Controls panel.

3. **Viewer `AssertionError` traceback in websockets `_drain_helper`**: harmless, happens when a browser
   tab connects or disconnects during training. Training continues. The watcher now filters it out.

4. **RAM nearly exhausted with `--eval-mode all`** (2026-09-28, IMG_6556). It made nerfstudio cache all
   650 full-res frames twice (as eval and as train images), so 11.8 GB in `ns-train`, 0.7 GB free, swap full.
   Stopped before WSL could die. Fix: removed `--eval-mode all` (back to the default 90/10 split).
   **Rule of thumb:** ~650 full-res 1080p frames with `cache-images cpu` is about 10-11 GB RAM, near the
   WSL limit. More frames require raising the WSL memory limit (`.wslconfig`, needs `wsl --shutdown`
   at the laptop, which kills Remote Control).

5. **Laptop slept overnight** (2026-09-27 18:15 to 2026-09-28): heartbeat log has a gap. Keep-awake
   does not prevent lid-close or unplugged sleep. Keep plugged in, lid open or lid action "Do nothing".

6. **Remote Control auth expired** once (`OAuth access token has been revoked`, seen in the tmux
   `claude` session log). It recovered on its own; if Remote Control stops responding, run `/login`
   at the laptop.

7. **COLMAP 4 cannot open databases created by COLMAP 3.10** ("Migrating pose_priors table ...
   SQLite error"). If using `colmap4`, run the whole pipeline (features, matching, mapper) with it.

8. **splatfacto-big nearly fills the 8 GB GPU** (2026-09-28, IMG_6556, 650 full-res frames).
   GPU memory: 3.0 GB at step 1.1k, 5.3 GB at 5.7k, 7.2 GB at 8.2k (densification continues to 15k).
   Step time went from about 40 ms to about 194 ms. The user chose to let it run. Risk on WSL: the NVIDIA
   driver may spill into Windows system RAM (sysmem fallback) instead of failing with a clean OOM, which
   slows training badly and never triggers the script's fallback to `splatfacto`.

   Outcome: at step ~10.6k the step time hit 562 ms (likely spilling), so the run was stopped and
   resumed from the step-10000 checkpoint (see Problems 10 and 11).

10. **Resuming a checkpoint fails with `WeightsUnpickler error: Unsupported global: numpy.core.multiarray.scalar`**
   (PyTorch 2.11 defaults `torch.load(weights_only=True)`). Fix: `export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`
   before `ns-train --load-dir ...` (safe for our own checkpoints). Also: the first resume script kept going
   after the failure and started IMG_6557. Runner scripts must `exit` on a failed step.

11. **Resuming mid-densification crashes: `CUDA error: device-side assert ... index out of bounds`**
   at the first refinement after the resume (step ~10080-10100). The checkpoint does not match the
   densification bookkeeping. Fix: resume with `--pipeline.model.stop-split-at` <= the checkpoint step (used
   10000), so no more splitting or culling happens. Result: GPU 7.2 GB -> 6.1 GB, step time 562 ms -> 144 ms.
   `resume_6556_then_6557.sh` resumes IMG_6556 from `outputs/IMG_6556/splatfacto/2026-09-28_143534/`.
   `make_splat_hq.sh` now defaults to `STOP_SPLIT=11000` for fresh runs (IMG_6557).
   (Nerfstudio names the output folder `splatfacto` even for `splatfacto-big`.)

12. **Resume adds `--max-num-iterations` on top of the checkpoint step** (2026-09-28). Resuming at step 10000
   with `--max-num-iterations 30000` trained to 40000 (the progress display showed 131 %), about 1.5 h longer
   than expected. When resuming, pass `max_iters - checkpoint_step` (for example 20000).
   With a viewer tab connected, GPU memory went from 6.1 to 7.9 GB and steps slowed from about 140 ms to
   400-450 ms. The last ~10k steps took until 18:55. **Keep the viewer closed during training on 8 GB.**
   Result: `exports/IMG_6556/splat.ply`, 2.89M Gaussians, 717 MB, exported 18:58.
   IMG_6557 put on hold at the user's request (runner stopped; `export_6556_only.sh` exported IMG_6556 only).

13. **GaussianShader (BRDF/relightable) setup on IMG_6556** (2026-09-28). Steps that were needed:
   - The CUDA extensions (`diff-gaussian-rasterization`, `simple-knn`, nvdiffrast) build with
     `CC/CXX=x86_64-conda-linux-gnu-gcc/g++`, `TORCH_CUDA_ARCH_LIST=8.9`, `CUDA_HOME=$CONDA_PREFIX`,
     `pip install --no-build-isolation`, and `#include <cstdint>` added to `rasterizer_impl.h`.
   - `open3d` import failed with `libusb-1.0.so.0` missing (same as in gsplat). Fix: `mamba install -n gaussian_shader libusb`.
   - Its JIT plugin `renderutils_plugin` failed with `cuda_runtime.h: No such file`. The conda CUDA headers are in
     `$CONDA_PREFIX/targets/x86_64-linux/include`. Fix: `CPATH=$T/include LIBRARY_PATH=$T/lib:$T/lib/stubs:/usr/lib/wsl/lib:$CONDA_PREFIX/lib`.
   - It only accepts PINHOLE cameras, so run `colmap image_undistorter` on `data/IMG_6556/colmap/sparse/0_refined`
     -> `data_gaussianshader/IMG_6556` (650 frames, 1884x1059, 2.2 GB), then move `sparse/*.bin` into `sparse/0/`.
   - Test (500 iters, `-r 2 --data_device cpu`): about 2.5 it/s, about 2.8 GB GPU at iter 500, PSNR 15.3 at iter 500.
   - `train.py` has **no `--checkpoint_iterations`** (argparse error), so runs can't be resumed. Only `--save_iterations` point clouds.
   - The full run with default densification (grad 0.0002 until 15k) grew GPU memory about 0.5 GB every 2 min:
     1.6 GB @640, 3.4 @2280, 4.6 @2990, heading for 8 GB around step 4.5-5k, before the first save at 7k.
     Stopped at about 3k (21:43-22:05 lost). Restarted 22:06 with `--densify_grad_threshold 0.0004
     --densify_until_iter 7000` and saves every 5k.
   - Result: finished 2026-09-29 01:18 (3 h 12 min, about 2.6 it/s). GPU peaked about 3.1 GB, RAM about 6.2 GB, no errors.
     Train PSNR 24.58 @15k, 25.06 @30k. Snapshots every 5k in `outputs_relightable/IMG_6556/point_cloud/` (about 106 MB each).
     The heartbeat showed no overnight sleep gaps. The watcher's "finished" notification only reached the chat at 07:27
     (delivery delay on the session side, not a laptop problem).
   - Relighting: GaussianShader's `render.py` has no "new envmap" flag. The learned lighting is
     `brdf_mlp/iteration_N/brdf_mlp.hdr` (lat-long HDR, loaded with `load_env`), so to relight, make a model dir that
     symlinks `point_cloud`, `cameras.json`, `input.ply`, `cfg_args` and holds a different `.hdr` there.
     `run_gs_render_6556.sh` renders learned + Poly Haven CC0 `studio_small_08` and `kloppenheim_06` (1k HDRs in
     `envmaps/`) and writes MP4s to `exports/IMG_6556_relightable/`. Don't use `set -u` in scripts that `source`
     conda activate (the gcc activation script fails with `SYS_SYSROOT: unbound variable`).
   - Result: the learned-lighting render matches the input well (frame 300 mean 122.8 vs GT 123.8). Relit with the raw
     studio HDR it came out washed out (mean 187) with purple/blue streaks in shadowed grass (specular on noisy grass
     normals). Cause: raw Poly Haven HDRs are much brighter than the learned light (mean radiance 0.129 learned vs
     0.698 studio = 5.4x, 0.494 sunset = 3.8x). Made `envmaps/*_1k_matched.hdr` scaled to 0.129 and rendered them with
     `run_gs_render_matched_6556.sh`. Baked cast shadows (sun on grass) stay in any relight; GaussianShader has no shadow model.
   - **Key finding: GaussianShader only relights the specular (reflection) term.** Frame 570: `diffuse_color` is 0.441 in
     both the learned and studio renders (unchanged); only `specular_color` changed (0.179 -> 0.439). Its shading is
     diffuse albedo (not lit by the envmap) + specular tint x reflected env light. So the sun, shade and shadows are
     baked into the diffuse part. A new envmap only changes reflections. On grass the learned normals are noisy, so the
     reflected light shows up as "cloudy" haze and purple streaks everywhere (the user reported "lots of cloudy artifacts
     and too bright"). GaussianShader targets shiny objects; it is not a real relighting method for an outdoor diffuse scene.
   - GaussianShader `render.py` holds about 12.5 GB RAM (all 650 frames loaded), leaving about 2.6 GB. Don't run it next to COLMAP
     or nerfstudio training. `run_6557.sh` waits for it to finish.

14. **IMG_6557: splatfacto-big crashed with `RuntimeError: CUDA driver error: device not ready`** (2026-09-29 10:07,
   step 7390, about 196 ms/step, which suggests VRAM was already spilling to system RAM). This happened despite
   `stop-split-at 10000` and `expandable_segments`. The script fell back to `splatfacto`, which completed 30k steps (10:07-10:58,
   about 115 ms/step, GPU about 3.4 GB). Result `exports/IMG_6557/splat.ply`, 1.45M Gaussians, 359 MB. The partial big run is
   in `outputs/IMG_6557/splatfacto/2026-09-29_095338` (ckpt step 6000).
   **Monitoring gap:** no watcher ran from about 09:53 to 10:50 (the expiry notice arrived late), so the crash was only
   reported 45 min later. Conclusion: on 8 GB, splatfacto-big at full res with 650 frames is unreliable. Use
   `stop-split-at` <= 7000 or plain splatfacto.

15. **Before/after screenshots: `ns-render dataset` pairs a *distorted* `gt-rgb` with an *undistorted* render**
   (2026-09-29). The offset was about 0 px in the centre and about ±12 px in opposite corners, which gave a misleading PSNR of about 15 dB.
   Fix: `cv2.undistort(gt, K, D)` with the same K (OPENCV k1,k2,p1,p2 from `transforms.json`), then crop 30 px. This gives
   held-out PSNR medians of **IMG_6556 21.2 dB** (19.7-23.4) and **IMG_6557 22.5 dB** (20.0-24.5). Grass-heavy scenes
   score low on PSNR even when they look right. Also note: with `camera-optimizer SO3xR3`, held-out views use unrefined
   poses. Images are in `exports/before_after/` (median, best and worst per scene, plus relighting panels). Also: run
   `ns-render` with the gsplat env *activated* (otherwise "Ninja is required").

16. **Memory/temperature observations:** COLMAP feature matching pushed the GPU to 86 °C (throttling
   starts about 87 °C). splatfacto-big used about 3-4.7 GB GPU early in training, compared with about
   2.2 GB peak for splatfacto.

## Feedback from the school machine (2026-09-29)

Second machine: RTX PRO 6000 Blackwell (96 GB), fresh WSL Ubuntu 26.04, set up from the **pushed** repo (without this
laptop's 6+ unpushed commits). The Claude session there reported:

| Problem there | Cause | Status on this laptop / in local commits |
|---|---|---|
| COLMAP crashed on `--SiftExtraction.use_gpu` | `environment.yml` leaves `colmap` unpinned; conda gave 3.13, which renamed the option (nerfstudio's `ns-process-data` still passes the old name) | Not hit here (laptop has COLMAP 3.10). **Unfixed**, and affects both `make_splat.sh` and `make_splat_hq.sh`. Fix: pin `colmap<3.12` or patch the option names |
| CUDA 12.4 nvcc vs gcc 15 | Docs assume older Ubuntu; `wsl --install` now gives 26.04 | Hit here too (GaussianShader). Worked around with **conda `gxx_linux-64=12`** in the env (problem 13). Not in README/setup_env.sh |
| `cuda_runtime.h` not found building gsplat | conda CUDA headers live in `$CONDA_PREFIX/targets/x86_64-linux/include` | Hit here too. Fix: `CPATH`/`LIBRARY_PATH` (problem 13, `run_gs_6556.sh`). Only in unpushed commits |
| Branch only reachable via the PR; `main` has only a README | Work lives on `claude/clever-hawking-j0d7up` | Still true. The quickstart should name the branch, or merge |
| Quality settings hardcoded, nerfstudio silently caps width at 1600 px | `make_splat.sh` has fixed method/matching/resolution | `scripts/make_splat_hq.sh` (unpushed) does full res (`--downscale-factor 1`), splatfacto-big, and takes frames/iters/`STOP_SPLIT` as args, but method/matching aren't overridable yet |
| Docs contradict (SuperSplat vs nerfstudio viewer; admin needed for WSL vs no admin) | Written at different times | Unfixed |
| Only ffmpeg pinned | nerfstudio, COLMAP, CUDA libs float | Unfixed. COLMAP was the first to break |

**Coordination:** push this laptop's commits **before** the school session writes its fixes, so the two don't conflict
(`gh auth login --web`, then `git push`). Then the school session can build on `make_splat_hq.sh` and these notes.

## Performance on other GPUs (estimates, 2026-09-29)

Projected from our own runs (650 frames, 1080p, 30k steps) + gsplat's benchmark (3.2M Gaussians, 30k steps: 19 min,
5.6 GB on an A100, smaller images) + bandwidth/VRAM specs. Not measured.

| GPU | VRAM | splatfacto | splatfacto-big uncapped | GaussianShader full res |
|---|---|---|---|---|
| RTX 4070 Laptop (ours) | 8 GB | ~50 min (measured) | won't fit (measured) | half res only, 3 h 12 min (measured) |
| RTX 4080 / 4070 Ti Super | 16 GB | ~25 min | ~50-70 min | ~2-2.5 h |
| RTX 4090 | 24 GB | ~15-20 min | ~35-45 min | ~1-1.5 h |
| RTX 5090 | 32 GB | ~12-15 min | ~25-35 min | ~45-60 min |
| A100 40/80 GB | 40-80 GB | ~15-20 min | ~30-40 min | ~1-1.5 h |
| H100 | 80 GB | ~10-15 min | ~20-30 min | ~40-60 min |

COLMAP (CPU-bound mapper) and the 15.5 GB WSL RAM limit don't improve with a better GPU. 16 GB+ VRAM is the threshold for uncapped splatfacto-big.

## Remote viewing

| Thing | How |
|---|---|
| Remote Control | `claude remote-control` running in tmux session `claude` (survives closing the Ubuntu window). |
| Splat viewer | `localhost:7007` (training's built-in viewer, or `ns-viewer --load-config ... --viewer.websocket-port 7007`). Public via `cloudflared tunnel --url http://localhost:7007` (new random link each start). |
| Terminal dashboard | tmux session `dash` (top: `tail -F` of the current training log; bottom: GPU/RAM every 5 s). Served read-only by `ttyd -p 7681 -i 127.0.0.1 -c murph:<password> tmux attach -r -t dash`, password in `~/.ttyd_pass`, public through a second `cloudflared` tunnel to port 7681. |
| Stop everything public | `pkill -f "cloudflared tunnel"` and `pkill -f ttyd` |

## GS-IR shadow removal (started 2026-10-07)

Goal: remove hard object shadows from splats for use in Blender (KIRI add-on) and UE5, without
lengthening the pipeline much. Blender/UE5 splat plugins don't relight; they only read the standard
PLY fields. So the target output is the shadow-free **albedo written into `f_dc`, with `f_rest` zeroed**.

### Method selection (2026-10-07)

Compared 3DGS inverse-rendering methods that separate shadows/visibility from albedo. Ranked fastest
viable first. Runtime multipliers are estimates from each method's design, not measured here.

| Rank | Option | Shadow handling | Inputs | Cost | Code |
|---|---|---|---|---|---|
| 1 | Diffuse light at capture (overcast, softbox, cross-polarised) | Avoids shadows | Capture change | None | n/a |
| 2 | SideFX Labs "Delight GSplats" (Houdini, Aug 2026 update) | Removes baked lighting from an existing splat | Existing splat | No retrain; speed unknown | Houdini Labs |
| 3 | **GS-IR** (CVPR 2024) | Baked occlusion volumes (probes) | COLMAP / transforms | ~1-1.5x 3DGS (est.) | Public |
| 4 | Relightable 3D Gaussians / R3DG (ECCV 2024) | BVH ray-traced visibility | Same as 3DGS | ~2-3x (est.) | Public |
| 5 | IRGS (CVPR 2025) | 2D-Gaussian ray tracing, indirect light | Same as 3DGS (2DGS surfels) | Slowest | Public |
| - | SSD-GS (ICLR 2026), GS³ | Shadow decomposition | Point-light / OLAT captures | n/a | Public; doesn't fit casual captures |
| - | GHPT (CVPR 2026), GaRe | Path tracing / outdoor collections | - | - | No usable code found |

GS-IR chosen as the first method to try on existing captures. Full settings sheet (defaults read
from source, suggested values): project file `gs-ir/gs-ir-settings.md`.

### Setup on Murph (2026-10-07)

- GPU: RTX 4070 Laptop, 8 GB (driver 610.47). 8 GB is tight for GS-IR; plan `-r 2` or `-r 4`.
- First scene: `data_gaussianshader/IMG_6556` (undistorted, `images/` + `sparse/0`, PINHOLE,
  650 images at 1884x1059). Usable as-is. The other scenes (`data/IMG_6556`, `IMG_6557`,
  `my_scene_600f`, `my_scene`) have OPENCV cameras at 1920x1080 and need undistorting first.
  IMG_6557 previously ran out of memory on this GPU.
- Live view: tmux `dash` + ttyd on `localhost:7681` restarted. Public cloudflared tunnel was **blocked
  by Claude Code auto mode** ("External Ingress Tunnel"), so there's no phone link until it's allowed
  on the laptop.

### Problems found and fixes (GS-IR)

1. **GS-IR CUDA extension failed to compile: `identifier "uint32_t" is undefined`** (2026-10-07,
   install attempt 1). Newer gcc no longer pulls in `<cstdint>` transitively.
   Fix: added `#include <cstdint>` to `gs-ir/src/utils.h`, `gs-ir/src/pbr_utils.cuh` and
   `submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h` (local edits in `~/GS-IR`,
   not upstream). Attempt 2: gs-ir, simple-knn, diff-gaussian-rasterization and nvdiffrast all build;
   `torch.cuda.is_available()` is True.
2. **Import check after the build failed** (2026-10-07, attempt 2). The script's success line didn't
   print and the following renderutils JIT step didn't run. The log couldn't be read (auto mode block), so
   each module was imported directly in `gsir`. Two causes found and fixed:
   - open3d failed to load because `libusb` was missing. Fix: `conda install -c conda-forge libusb` into `gsir`.
   - GS-IR's renderutils JIT build couldn't find the CUDA headers or WSL's `libcuda`. Fix: env vars saved
     into the `gsir` env (applied on activate): `CUDA_HOME`, `CPATH`, `LIBRARY_PATH` (incl. `/usr/lib/wsl/lib`),
     `CC`/`CXX` = conda gcc 12, `TORCH_CUDA_ARCH_LIST=8.9`.
   Result: torch 2.4.1+cu124 sees the GPU; kornia, opencv, open3d 0.20, nvdiffrast (CUDA context starts),
   gs_ir, simple_knn, diff_gaussian_rasterization, plyfile and lpips all import; renderutils compiles and
   loads. **Install complete; no training run yet.**
3. **Auto mode blocks on the laptop session** (2026-10-07): reading the install log (flagged as output
   of externally sourced code) and starting cloudflared tunnels. Neither shows a permission prompt,
   so neither can be approved remotely; both need a permission rule on the laptop.
4. **First GS-IR training launch crashed: `NameError: SummaryWriter`** (2026-10-07). tensorboard wasn't
   in the `gsir` env (GS-IR imports it optionally and then uses it anyway). Fix: installed tensorboard into
   `gsir` and relaunched. Failed log kept as `~/gsir_train_attempt1.log`.

5. **GS-IR stage 1 with default densification slowed to a crawl on 8 GB** (2026-10-07, run attempt 2,
   started 15:33). Grad 0.0002, densify until 15k: ~14 it/s at first, but by 16:05 GPU memory reached
   7.1 of 8 GB and at iteration 4,950/30,000 it was 4.46 s/it (~31 h projected; loss 0.348). No crash, so the
   OOM retry never triggered. Same spill-to-RAM pattern as GaussianShader/splatfacto-big on this laptop.
   Fix: stopped it and restarted stage 1 at 16:06 with `--densify_grad_threshold 0.0004 --densify_until_iter 7000`
   (the GaussianShader settings that fit here). Kept: `~/gsir_train_attempt2_slow.log`,
   `outputs_gsir/IMG_6556_attempt2_slow`. **Rule:** on this laptop, slowdown (not a crash) is the failure
   signal; a watcher should trigger the fallback when s/it rises past a threshold or VRAM passes ~7 GB.
6. **Progress watcher sent nothing during attempt 2** (2026-10-07). Replaced with `~/gsir_watch.sh`.

7. **GS-IR baking at `occlu_res 256` filled the 8 GB GPU and slowed down** (2026-10-07, 16:54-17:07).
   GPU memory climbed from 2.9 GB (16:57) to 7.9 GB (17:06); the bake rate fell from ~41 to ~22-26 cells/s
   (4-5 h projected) at 29,619/420,159 cells (7%). No OOM error, same crawl pattern as item 5.
   Cause: the bake loop selects a different-sized set of Gaussians per cell, and PyTorch's CUDA caching
   allocator keeps growing to fit them (fragmentation, not real need).
   Fix: stopped at 17:07 (log `~/gsir_train_stage1_and_bake256.log`); re-baked at `occlu_res 192` with
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (doesn't change results), reusing `chkpnt30000.pth`.
   Runner gained `SKIP_STAGE1=1` and `BAKE_RES=<n>`. 192 bake started 17:08: 240,894 cells, ~44 cells/s,
   2.7 GB GPU, ~1.5 h projected. **Open question for the thesis:** whether 256 fits with
   `expandable_segments` alone (untested).

8. **Albedo export script failed on first run** (2026-10-07 19:33:41): the albedo tensor was converted
   to numpy without `.detach()` (still tracked by autograd). Fix in the export script, re-run via
   `~/gsir_export_6556.sh`. Our own script, not GS-IR.

### GS-IR runs

| Date | Scene | Settings | Result |
|---|---|---|---|
| 2026-10-07 15:33 (laptop time) | `IMG_6556` (`data_gaussianshader`, undistorted) | Runner `scripts/runs/run_gsir_6556.sh` (tmux `gsir`, log `~/gsir_train.log`, out `outputs_gsir/IMG_6556`). Stage 1: 30k, `-r 2` (942x530), `--data_device cpu`, `--eval` (every 8th frame held out), default densify; OOM fallback `--densify_grad_threshold 0.0004 --densify_until_iter 7000`. Baking: `--bound 6.0 --valid 6.0 --occlu_res 256 --occlusion 0.4 --cubemap_res 256` (all 650 cameras within 4.63 units of origin, ~92% of 445k SfM points within 6; cell ~4.7 cm); OOM fallback `occlu_res 192`. Stage 2: to 40k, `--indirect --gamma`, metallic off, `brdf_tv 1.0`, `env_tv 0.01`. Export: `exports/IMG_6556_gsir/splat_albedo.ply`, f_dc = (albedo - 0.5)/0.28209 (sRGB albedo due to `--gamma`), f_rest = 0. | Attempt 1: crashed at launch (tensorboard, item 4). Attempt 2: stopped at 4,950 (too slow, item 5). |
| 2026-10-07 16:06 | `IMG_6556` | Attempt 3: as above but stage 1 `--densify_grad_threshold 0.0004 --densify_until_iter 7000`. | Stage 1 done 16:06-16:54 (48 min, ~10-11 it/s, no errors). Held-out (82 frames): 7k PSNR 24.29 / SSIM 0.838 / L1 0.0408; 30k PSNR 26.29 / SSIM 0.897 / L1 0.0317. Train: 7k 24.75 / 0.832; 30k 27.31 / 0.905. GPU peak ~3.9 GB near 7k, then steady 3.5 GB; 78-82 °C; WSL RAM steady 7.1 GB. `chkpnt30000.pth` 1.38 GB. (Not directly comparable: nerfstudio held-out median on IMG_6556 was 21.2 dB at full res, different split.) Baking started 16:54 (`occlu_res 256`, bound 6.0): 420,159 occupied cells at ~38 cells/s (6,155 after 2.5 min) = ~3 h projected; GPU 2.9 GB, 71 °C, RAM 2.8 GB. Baking, not training, is the slow step on this laptop at 256. Stopped at 7% (VRAM full, item 7); re-baked at 192: 17:08:35-19:00:25 (1 h 52 min, 240,894 cells, ~36 cells/s avg, rate varied 29-59/s with scene density), GPU steady ~3.2 GB with `expandable_segments`, no errors; `occlusion_volumes.pth` 43.7 MB. Stage 2 started 19:00:25 (30k-40k, ~5 it/s, ~33 min est.; first loss 0.085, a different loss from stage 1's), GPU 3.2 GB, 71 °C, WSL RAM 12 of 15 GB (watched). RAM dropped back to ~7 GB after loading. Stage 2 done 19:33:39 (33 min, 5.3 it/s, final loss ~0.05). At 40k (albedo x learned lighting, `--gamma`): held-out PSNR 23.41 / SSIM 0.775 / L1 0.0481; train 23.85 / 0.773. Lower than stage 1 (26.29) because colour is now constrained to albedo x lighting instead of free per-Gaussian SH (expected for inverse rendering). Export: `exports/IMG_6556_gsir/splat_albedo.ply`, 414 MB, 1,668,597 Gaussians, f_dc = (sRGB(albedo) - 0.5)/0.28209, f_rest = 0, normals 0, no extra fields. Held-out renders (`render.py --pbr`, 82 views): PBR vs GT PSNR 23.34 / SSIM 0.773 / LPIPS 0.199; outputs `outputs_gsir/IMG_6556/test/ours_None/` (`pbr/`, `*_brdf.png` = albedo|roughness|metallic linear, `normal/`); comparison strips (original | sRGB albedo | PBR) `exports/IMG_6556_gsir/compare_{0,20,40,60,80}.jpg`. All done 19:36:53. **Total GS-IR time for this scene on the laptop: 48 min + 1 h 52 min + 33 min = ~3 h 13 min** (excluding failed attempts). |

### GS-IR outcome on IMG_6556 (2026-10-07)

**GS-IR did not remove the hard cast shadows.** In the held-out albedo (e.g. test index 40 / frame 321) the
chair's sun shadow on the grass is still clearly dark. The albedo does even out overall sun/shade brightness
somewhat and lightens the shaded sides of the chairs.

Why (consistent with the method's design): GS-IR's occlusion is baked **ambient** occlusion from probes,
and its light is a smooth environment map with no explicit directional sun visibility. A hard cast shadow
that the model can't explain with AO x env light gets absorbed into the albedo. Outdoor sun scenes are a
poor fit for this model; this is a negative result worth keeping for the thesis.

Exact versions for reproducibility: GS-IR commit `e5a030b`; Python 3.10; conda-forge `cuda-toolkit` 12.4.1,
gcc/g++ 12.4; torch 2.4.1+cu124, torchvision 0.19.1; numpy<2; extensions built with
`pip install --no-build-isolation`. Scripts on the laptop (WSL, not yet committed):
`scripts/runs/run_gsir_6556.sh`, `scripts/gsir_export_albedo.py`, `~/gsir_export_6556.sh`.

Next candidates (not run): (1) stage 2 with a larger `env_tv`/sharper env map so a sun can be represented:
unlikely to be enough; (2) a method with explicit ray-traced visibility or a directional sun + visibility
term (R3DG; outdoor sun/sky models such as LumiGauss); (3) masking/inpainting shadows in the training
images before reconstruction (adds pipeline time).

### Sun + sky methods: survey (2026-10-08)

The user picked a sun + sky model as the next approach after GS-IR. Checked the candidates' inputs and code:

| Method | Inputs | Sun shadow model | Cost reported | Code |
|---|---|---|---|---|
| LumiGauss (WACV 2025) | Photo collections, **varying lighting**; per-image lighting latent | Per-Gaussian SH radiance transfer (0 = shadowed, 1 = lit), env light as SH; shadowed model trained as a second stage | ~1 h 20 min (A100) | github.com/joaxkal/lumigauss |
| ROSGS (2025, arXiv 2509.11275) | Multi-view, **unconstrained/varying lighting**, per-image embeddings | Sun = one spherical Gaussian; BVH ray-traced visibility against a 2DGS mesh; sky via PRT | ~2.9 h (RTX 3090 24 GB) | Not released |
| OSDR-GS (IJCAI 2025) | **Multiple lighting conditions**, clustered into lighting groups | Per-Gaussian SH sun visibility, pushed toward binary | ~30 min (RTX 4090) | Not released |
| GaRe (ICCV 2025) | Unconstrained photo collections | Outdoor relighting | - | - |

**Key finding:** every sun + sky method found relies on the **same scene seen under different sun positions**
to tell shadow from albedo. IMG_6556 is one video under one lighting condition, so shadow vs dark
material is ambiguous for all of them (the same reason GS-IR failed). Only LumiGauss has public code; on a
single-lighting capture it would get no lighting variation to learn from (inferred from the method, not tested).

### 3D shadow-mask fix (started 2026-10-08)

The user parked LumiGauss and picked a detect-and-correct approach that works on existing splats with no
retraining: `scripts/fix_splat_shadows.py`.

How it works: (1) per-frame 2D shadow masks from an off-the-shelf shadow detector; (2) every Gaussian centre is
projected into each masked frame (COLMAP intrinsics incl. OPENCV distortion), visibility is checked against a
coarse z-buffer of opaque Gaussians (default 1/4 res, 3% depth tolerance, opacity > 0.5), and shadow votes /
visible votes gives `shadow_frac`; (3) Gaussians seen in >= 5 frames get brightened with a weight ramping from
`shadow_frac` 0.3 to 0.7; (4) the gain is the median lit/shadow ratio (linear RGB) in thin bands either side of the
mask edges across frames (clipped to 1-6), which approximates sun/sky for the same surface. f_dc is scaled in
linear light; f_rest gets the same per-channel factor. A debug PLY paints the corrected splats red.

Synthetic check (cloud, 2026-10-08): plane of 1,800 Gaussians, left half dark (0.2) and masked in 6 frames:
900 marked, 870 at full weight, lit half unchanged (0.6), shadow half 0.2 -> 0.48 because the measured ratio (9.5)
hit the 6.0 gain cap. **Known limits:** shadowed surfaces keep their flatter look (no texture recovery), and
results depend on mask quality and on the splat being in COLMAP coordinates.

**SDDNet masks on IMG_6556 (2026-10-08, laptop).** SDDNet (commit `dddcdd4`, authors' SBU checkpoint `sbu.ckpt`,
158 MB, EfficientNet-B3 backbone) in its own conda env `sddnet` (Python 3.10, torch 2.4.1+cu124). Wrapper
`scripts/sddnet_masks.py` (laptop): resize to 512x512, sigmoid, resize back, threshold 0.5; ~0.3 s/frame.
5-frame check (frames 1, 321, 481): mean shadow fraction 22.6%. **Good:** chair cast shadows on the grass and the
shaded lawn strip by the shed/fence. **Wrong or out of scope:** self-shaded chair sides and legs (consistent across
views, so multi-view voting won't remove them), dark foliage (dark, not shadowed), fence/house wall in tree shade.
**Decision:** restrict the correction to the ground. Added `--ground-only` to `fix_splat_shadows.py`: RANSAC
plane through opaque Gaussians, constrained to within ~37° of the average camera up vector, then least-squares
refined; only Gaussians within `--ground-tol` (default 2% of the camera spread) are brightened. Synthetic check:
ground plane at y = 1 plus an occluding masked object: plane found (up alignment 1.000), 151 visible masked ground
Gaussians brightened, 0 of 800 object Gaussians touched.

**Laptop run 1 on IMG_6556 (2026-10-08).** All 650 frames masked with SDDNet in ~60 s (mean shadow fraction
22.5%). Fix run on `exports/IMG_6556/splat.ply` with `data/IMG_6556/colmap/sparse/0_refined`, `--every 2
--ground-only`, ~149 s per run.

| Run | Settings | Ground plane | Brightened | Gain (R/G/B) | Shadow/lit ratio on 3 test frames |
|---|---|---|---|---|---|
| before | - | - | - | - | 0.44 / 0.38 / 0.48 |
| run 1 (`exports/IMG_6556_shadowfix/`) | auto gain, lo 0.3, hi 0.7 | normal (-0.014, -0.950, -0.312), up alignment 0.998, tol 0.0712, 1,352,189 Gaussians near plane | 141,825 | 4.00 / 4.31 / 3.09 (auto) | 0.71 / 0.63 / 0.77 |
| run 2 (`exports/IMG_6556_shadowfix_g55/`) | gain 5.5, lo 0.2, hi 0.6 | same | - | 5.5 fixed | 0.77 / 0.68 / 0.84 |

Result: cast shadows on the grass clearly lighter but not gone (a perfect fix would give ratios near 1.0).

Problems found and fixes (shadow fix):
1. **Ground-plane SVD ran out of memory (131 GiB request).** `np.linalg.svd` on ~1.3M inlier points built the full
   U matrix. Fix: `full_matrices=False`.
2. **~400k untouched splats changed.** The whole f_dc array went through linear->sRGB with clipping, which altered
   out-of-range DC values on Gaussians with weight 0. Fix: rewrite f_dc only where weight > 0; untouched rows are
   now bit-identical (synthetic check).
3. **Auto gain too low (~4 vs ~5.4 measured from the images).** The edge bands sampled penumbra pixels right next
   to the mask edge. Fix: `--band-gap` (default 8 px) skips pixels next to the edge before sampling the lit and
   shadow bands.
4. **Lower grass layers stayed dark.** Gaussians under the top grass layer fail the z-buffer visibility test, so
   they never collect shadow votes. Fix: `--ground-spread R`: each ground Gaussian takes the highest weight of any
   brightened ground neighbour within R (plane 2D coordinates, cKDTree). Synthetic check with R = 0.1: brightened
   ground Gaussians 151 -> 301, object Gaussians touched 0 of 800.
Next: rerun on the laptop with `--ground-only --ground-spread` and the new auto gain, then re-measure the same
3 frames.

**Laptop round 2 on IMG_6556 (2026-10-08).** Script at `a490f08`, `--ground-only`, auto gain, same ground plane.
158 s per run. Renders in `exports/IMG_6556_shadowfix_v2_compare/` (photo | before | s03 | s06).

| Run | Spread | Brightened (before -> after spread) | Full weight | Gain (R/G/B) | Shadow/lit ratio (frames 321/481/161) | Nearby sunlit grass |
|---|---|---|---|---|---|---|
| v2 s03 (`exports/IMG_6556_shadowfix_v2_s03/`) | 0.03 | 141,825 -> 190,288 | 80,063 | 6.0 / 6.0 / 6.0 (at the `--max-gain` cap) | 0.77 / 0.67 / 0.84 | +21 / +21 / +16% |
| v2 s06 (`exports/IMG_6556_shadowfix_v2_s06/`) | 0.06 | 141,825 -> 219,073 | 80,186 | 6.0 / 6.0 / 6.0 (cap) | 0.76 / 0.66 / 0.83 | +23 / +23 / +18% |

Result: **no better than round 1 at gain 5.5** (0.77 / 0.68 / 0.84), and the sunlit grass beside the shadows
brightened 2-3x more than in round 1 (+7-10%).

Problems found and fixes (round 2):
5. **The spread brightened sunlit grass.** It copies weight to any ground neighbour within R in the plane, which
   includes lit grass beside the shadow, not only the layers underneath. Shadow and surroundings both got brighter,
   so the ratio barely moved. Fix: a darkness gate. Lit and shadowed ground references are the median linear
   luminance of ground Gaussians with `shadow_frac` < 0.1 and >= `--hi`. Each Gaussian's darkness is its log
   position between them, reaching full at the geometric midpoint. The spread now only passes weight to dark
   Gaussians. `--dark-gate` also applies the gate to the vote weights, so `--lo/--hi` can go lower (more shadowed
   splats at full weight) without touching lit ones. If lit and shadow can't be told apart by brightness, the gate
   is disabled with a warning.
6. **Auto gain hit the 6.0 cap on all channels; the true estimate was hidden.** Fix: the script prints and saves
   (`gain_raw_rgb`) the estimate before clipping; `--max-gain` default raised to 12.
Synthetic check (cloud): ground lit 0.6 / shadow 0.28 sRGB (linear ratio 5) plus a copy layer 0.03 below. With
spread 0.3, lit top and lower-layer Gaussians changed 4 and 3 -> 0 with the gate; shadowed top and lower
layers 672 each, mean 0.28 -> 0.46.

**Laptop round 3 on IMG_6556 (2026-10-08).** Script `eff7310`; `--every 2 --ground-only --dark-gate --lo 0.15
--hi 0.45`, auto gain. Ground luminance: lit median 0.6858, shadow median 0.0583 (ratio 11.76). Dark gate
172,695 -> 152,080 splats. Gain before clipping 11.291 / 11.584 / 6.829 (cap 12, so unclipped).
Renders `exports/IMG_6556_shadowfix_v3_compare/` and close-up `crop_00321_before_C.jpg`.

| Run | Spread | Brightened after spread | Full weight | Shadow/lit ratio (321/481/161) | Nearby sunlit change | Time |
|---|---|---|---|---|---|---|
| v3 C (`exports/IMG_6556_shadowfix_v3_c/`) | 0.06 | 232,261 | 89,800 | 0.87 / 0.80 / 0.95 | +28 / +23 / +22% | 156 s |
| v3 D (`exports/IMG_6556_shadowfix_v3_d/`) | 0.15 | 301,759 | 89,832 | 0.87 / 0.80 / 0.95 | +28 / +23 / +22% | 160 s |

Result: best ratios so far, but **visually wrong**. Former shadows turn neon yellow-green and over-saturated,
brighter at their edges, and the chair-leg bottoms go pale. Spread 0.15 gives the same numbers as 0.06. Round 1 at
gain 5.5 still looks the most natural.

Problems found and fixes (round 3):
7. **Yellow-green cast.** Per-channel gains were unequal (R/G ~11.4, B 6.8): skylight is bluish, so the photo
   ratio is lower in blue. Multiplying per channel and then clipping each channel at 1 also shifts hue. Fix:
   `--colour lum` (new default) uses one luminance gain for R, G and B. Overflow is handled by dividing all three
   channels by the largest, so hue is kept. `--colour rgb` keeps the old behaviour.
8. **Overshoot and bright edges.** The ~11x ratio holds for the darkest splats, but a rendered pixel blends several
   splats. Partly shadowed and edge splats are already brighter, so multiplying them by 11 overshoots. The photo
   ratio before the fix (0.44 sRGB, about 0.16 linear) points to about 6x for the blended pixel. Fix:
   `--mode target` (new default) caps each splat's gain at lit_median / own luminance (never below 1 or above
   the gain), so no splat is pushed past the lit ground median. `--max-gain` default back to 6. `--mode scale`
   keeps plain multiplication.
9. **Pale chair-leg bottoms.** Leg splats within the 7 cm ground band (`tol` 0.0712) were treated as ground.
   Next run uses `--ground-tol 0.03`.
Synthetic check (cloud, true linear ratio 5, auto estimate 6.75): scale/rgb at cap 12 pushed shadowed splats to
max 0.687 sRGB, past the lit 0.600. Target/lum: max 0.600, mean per-splat gain 4.99, lit splats unchanged.

**Laptop round 4 on IMG_6556 (2026-10-08).** Script `b0c0bf8`. Flags: `--every 2 --ground-only --dark-gate
--lo 0.15 --hi 0.45 --ground-spread 0.06 --ground-tol 0.03`, defaults `--colour lum --mode target`.
Ground plane with tol 0.03: normal (-0.006, -0.948, -0.317), up alignment 0.999, 892,073 Gaussians near it.
Ground luminance: lit median 0.6966, shadow median 0.0407 (ratio 17.11). Counts: dark gate 76,923 -> 64,032,
spread -> 97,317 (= changed_gaussians), full weight 37,051. Raw gain 11.291 / 11.584 / 6.829. 159 s per run.
Renders `exports/IMG_6556_shadowfix_v4_compare/` (photo | before | round-1 g5.5 | E | F), close-ups
`crop_00321_before_g55_E.jpg`, `crop_00321_E_F.jpg`.

| Run | Max gain | Luminance gain | Mean gain per touched splat | Shadow/lit ratio (321/481/161) | Nearby sunlit change | Saturation shadow/sunlit (321, 481, 161) |
|---|---|---|---|---|---|---|
| before | - | - | - | 0.44 / 0.38 / 0.48 | - | 0.41/0.32, 0.54/0.46, 0.37/0.29 |
| round 1 g5.5 | - | 5.5 rgb | - | 0.77 / 0.68 / 0.84 | +19 / +15 / +15% (approx.*) | 0.35/0.30, 0.48/0.44, 0.32/0.27 |
| v4 E (`exports/IMG_6556_shadowfix_v4_e/`) | 6 | 6.0 | 3.83 | 0.68 / 0.65 / 0.64 | +3.5 / +7.3 / +2.0% | 0.39/0.32, 0.51/0.46, 0.36/0.30 |
| v4 F (`exports/IMG_6556_shadowfix_v4_f/`) | 12 | 11.18 | 5.89 | 0.79 / 0.76 / 0.72 | +4.6 / +10.5 / +2.6% | 0.39/0.33, 0.51/0.47, 0.36/0.30 |

\* The 5-panel renders shift the round-1 crop relative to its mask. Earlier 4-panel measurement gave +7-10%.

Result: **F is the most natural so far.** Shadow lifted to 0.72-0.79 of the sunlit grass, grass colour and
saturation stay close to the original photo (no neon), and the sunlit grass beside the shadows barely changes. Round 1 lowers
saturation (washed out, whitish speckles, pale halo). Chair-leg bases are no longer pale with tol 0.03.
Not solved: the shadow is still visible as a lighter patch with a faint darker outline. Likely cause: edge splats
with partial weight (ramp `--lo` 0.15 to `--hi` 0.45). Next: narrower/lower ramp; the dark gate and lit ceiling
should keep lit splats safe.

**Laptop round 5 on IMG_6556 (2026-10-08).** Script `b0c0bf8`, flags as v4 F (`--every 2 --ground-only
--dark-gate --ground-spread 0.06 --ground-tol 0.03 --max-gain 12`), only the weight ramp changed. Luminance gain
11.18, ceiling 0.6966. All versions re-measured in one layout (photo | before | F | G | H). Renders
`exports/IMG_6556_shadowfix_v5_compare/`, close-up `crop_00321_before_F_G_H.jpg`.

| Run | Ramp lo-hi | Dark gate | After spread | Full weight | Mean gain | Whole shadow | Edge band ±3 px | Core (6 px in) | Nearby sunlit | Time |
|---|---|---|---|---|---|---|---|---|---|---|
| before | - | - | - | - | - | 0.44 / 0.38 / 0.48 | 0.71 / 0.73 / 0.68 | 0.35 / 0.27 / 0.37 | - | - |
| v4 F | 0.15-0.45 | 76,923 -> 64,032 | 97,317 | 37,051 | 5.89 | 0.79 / 0.76 / 0.72 | 0.86 / 0.94 / 0.80 | 0.77 / 0.64 / 0.70 | +4.6 / +10.5 / +2.6% | 159 s |
| v5 G (`..._v5_g/`) | 0.10-0.25 | 108,807 -> 81,604 | 145,600 | 41,017 | 5.00 | 0.79 / 0.75 / 0.71 | 0.86 / 0.93 / 0.80 | 0.76 / 0.63 / 0.69 | +5.8 / +11.8 / +3.5% | 161 s |
| v5 H (`..._v5_h/`) | 0.05-0.15 | 224,278 -> 142,240 | 261,707 | 45,660 | 4.26 | 0.78 / 0.74 / 0.71 | 0.85 / 0.93 / 0.79 | 0.75 / 0.63 / 0.69 | +7.2 / +13.3 / +4.4% | 162 s |

(Frames 321 / 481 / 161. Saturation identical for F, G, H: 0.39/0.33, 0.51/0.47, 0.36/0.30.)

Result: **a lower ramp doesn't help; F stays the best.** The "dark outline" is not an edge problem: the edge band is
already brighter than the core in every version (F: 0.86 vs 0.77). What reads as an outline is a darker core
with a slightly brighter rim. A lower ramp only adds weakly flagged splats on the lit side (more sunlit change,
no core gain). F copied to `C:\Users\roach\Downloads\IMG_6556_shadowfix_best.ply` (717 MB, byte-identical to
`exports/IMG_6556_shadowfix_v4_f/splat_shadowfix.ply`) for checking in Blender.

**Correction to round 4:** the round-1 g5.5 sunlit change of +15-19% is real (panels are 960 px wide in every
layout, so masks lined up). The earlier +7-10% was round 1's auto gain 4.0, not g5.5. The round-4 table's "approx."
note is wrong.

Hypothesis for the darker core: core splats are capped at the lit median, but unflagged lower layers and splats
outside the 3 cm band still show through. Added `--ceiling-pct` (default 50 = median, unchanged) to set the
target-mode ceiling to a higher percentile of lit ground luminance. Next: ceiling p75, and `--ground-tol 0.045`
with the dark gate.

**Laptop round 6 on IMG_6556 (2026-10-08).** Script `40709ca`. Base = v4 F flags (`--every 2 --ground-only
--dark-gate --lo 0.15 --hi 0.45 --ground-spread 0.06 --max-gain 12`). Renders
`exports/IMG_6556_shadowfix_v6_compare/` (photo | before | F | I | J | K), close-up `crop_00321_before_F_I_J_K.jpg`.

| Run | Ground tol | Ceiling | Brightened (gate -> spread) | Full | Mean gain | Whole shadow | Edge band | Core | Nearby sunlit | Time |
|---|---|---|---|---|---|---|---|---|---|---|
| v4 F | 0.03 | p50 0.6966 | 64,032 -> 97,317 | 37,051 | 5.89 | 0.79 / 0.76 / 0.72 | 0.86 / 0.94 / 0.80 | 0.77 / 0.64 / 0.70 | +4.6 / +10.5 / +2.6% | 159 s |
| v6 I (`..._v6_i/`) | 0.03 | p75 0.8387 | 64,032 -> 97,317 | 37,051 | 6.26 | 0.80 / 0.77 / 0.72 | 0.87 / 0.95 / 0.81 | 0.77 / 0.65 / 0.70 | +5.1 / +11.0 / +2.9% | 157 s |
| v6 J (`..._v6_j/`) | 0.045 | p50 0.6905 | 102,829 -> 153,113 | 59,476 | 5.82 | 0.87 / 0.80 / 0.86 | 0.91 / 0.98 / 0.89 | 0.84 / 0.68 / 0.84 | +10.0 / +14.0 / +6.1% | 161 s |
| v6 K (`..._v6_k/`) | 0.045 | p75 0.8341 | 102,829 -> 153,113 | 59,476 | 6.21 | 0.88 / 0.81 / 0.87 | 0.92 / 0.99 / 0.90 | 0.84 / 0.68 / 0.84 | +10.9 / +14.7 / +6.7% | 162 s |

(Frames 321 / 481 / 161. Tol 0.045: 1,140,669 Gaussians near the plane, up alignment 0.998; dark gate 120,071 -> 102,829.
Saturation unchanged in all versions; no neon.)

Result: **no clear winner over F; Downloads still holds F.** The ceiling percentile barely matters (I ≈ F, K ≈ J): the
core is limited by which splats are included, not by the ceiling. Hypothesis from round 5 confirmed in part: a
wider band (J) lifts the core a lot (frame 161: 0.70 -> 0.84), but the chair-leg bottoms go pale or bleached blue
again and the nearby sunlit grass brightens more.

Fix for J's legs: added `--colour-gate D`. Each splat's linear rg chromaticity must be within D of the
median chromaticity of the shadowed ground splats. This is generic: it takes the ground's own colour, not
"green". Synthetic check: 595 brown (0.30/0.18/0.12) splats inside the shadow region. Without the gate,
269 were changed; with D 0.05, 0 were changed, and grass-shadow splats changed 1,075 -> 1,067.
Next: J + colour gate.

Figures: pushing the comparison JPEGs to this branch from the laptop was blocked by auto mode ("Out-of-Place
Publication"). Waiting on the user's OK.

**Laptop round 7 on IMG_6556 (2026-10-08).** Script `1374f1d`. Base = v6 J flags (`--ground-tol 0.045`, ceiling p50
0.6905). Shadowed-ground reference chromaticity (r, g) = (0.381, 0.419). Renders `exports/IMG_6556_shadowfix_v7_compare/`,
close-ups `crop_00321_before_F_J_L_M.jpg`, `legs_00321_before_F_J_L_M.jpg`.

| Run | Colour gate | Dark gate -> colour gate -> spread | Full | Mean gain | Whole shadow | Edge band | Core | Nearby sunlit | Time |
|---|---|---|---|---|---|---|---|---|---|
| v4 F | - (tol 0.03) | 64,032 -> - -> 97,317 | 37,051 | 5.89 | 0.79 / 0.76 / 0.72 | 0.86 / 0.94 / 0.80 | 0.77 / 0.64 / 0.70 | +4.6 / +10.5 / +2.6% | 159 s |
| v6 J | - | 102,829 -> - -> 153,113 | 59,476 | 5.82 | 0.87 / 0.80 / 0.86 | 0.91 / 0.98 / 0.89 | 0.84 / 0.68 / 0.84 | +10.0 / +14.0 / +6.1% | 161 s |
| v7 L (`..._v7_l/`) | 0.04 | 102,829 -> 34,585 -> 49,246 | 13,505 | 3.70 | 0.64 / 0.58 / 0.65 | 0.78 / 0.82 / 0.76 | 0.61 / 0.50 / 0.60 | +1.9 / +3.5 / +1.7% | 168 s |
| v7 M (`..._v7_m/`) | 0.07 | 102,829 -> 58,004 -> 85,167 | 25,631 | 4.08 | 0.76 / 0.70 / 0.76 | 0.84 / 0.89 / 0.82 | 0.74 / 0.60 / 0.74 | +4.8 / +7.3 / +3.6% | 169 s |

Saturation (shadow/sunlit): L 0.36/0.32, 0.49/0.45, 0.34/0.29; M 0.34/0.32, 0.46/0.44, 0.33/0.29 (M slightly below the photo).

Result: **the colour gate fixes J's pale chair legs (L and M dark like F) but drops too much grass.** In-shadow grass
spans a wide colour range (lower layers are more yellow-green or brown), so a gate centred on the mean shadowed-ground
colour rejects much of it: at 0.07, 58k of 103k splats are kept. Ranking: F > M (≈ F on frame 161) > J (best core,
pale legs) > L. Downloads keeps F.

Also on 2026-10-08: a phone 3D viewer artifact of F vs the original (300k most visible splats per version, DC
colour only) was published for the user: https://claude.ai/artifact/8m58aabfViQxQfer36aVxb (private).
Next: wider gates (0.10, 0.13).

**Laptop round 8 on IMG_6556 (2026-10-08).** Script `1374f1d`, v6 J flags plus wider colour gates. Renders
`exports/IMG_6556_shadowfix_v8_compare/`.

| Run | Colour gate | Dark gate -> colour gate -> spread | Full | Mean gain | Whole shadow | Edge band | Core | Nearby sunlit | Time |
|---|---|---|---|---|---|---|---|---|---|
| v4 F | - (tol 0.03) | 64,032 -> - -> 97,317 | 37,051 | 5.89 | 0.79 / 0.76 / 0.72 | 0.86 / 0.94 / 0.80 | 0.77 / 0.64 / 0.70 | +4.6 / +10.5 / +2.6% | 159 s |
| v8 N (`..._v8_n/`) | 0.10 | 102,829 -> 69,791 -> 106,254 | 32,662 | 4.33 | 0.81 / 0.74 / 0.81 | 0.88 / 0.93 / 0.85 | 0.78 / 0.63 / 0.79 | +6.7 / +9.3 / +4.6% | 169 s |
| v8 O (`..._v8_o/`) | 0.13 | 102,829 -> 75,281 -> 116,823 | 36,088 | 4.53 | 0.82 / 0.76 / 0.82 | 0.89 / 0.94 / 0.86 | 0.79 / 0.64 / 0.79 | +7.8 / +10.3 / +5.2% | 170 s |

Saturation (shadow/sunlit) N and O: 0.34/0.32, 0.46/0.44, 0.32/0.29 (about 0.05 below F in the shadows). Chair-leg
bases dark in N and O (like F).

Result: **O is a modest improvement on F with no leg problem.** Core: frame 161 0.70 -> 0.79, frame 321 0.77 -> 0.79,
frame 481 tie at 0.64. Costs: 1-3 points more brightening of nearby sunlit grass and slightly greyer shadow areas. N is a
weaker O. Choice between F and O handed to the user (visual call).

**Where the method stands after 8 rounds.** The best versions lift the cast shadows from 0.38-0.48 to about
0.76-0.82 of the sunlit grass, without colour artefacts, at about 2.5-3 min per run plus 1 min of SDDNet masking
(650 frames). The remaining gap is in the shadow cores. Likely causes: splats outside the band and lower layers not
reached by votes, and the flattened texture of shadowed surfaces. A per-splat correction can't recover detail the
capture never recorded in sunlight.

**Decision (2026-10-08 12:56 UTC):** after comparing F, O and the original in the 3D viewer, the user chose to **keep F**
(`exports/IMG_6556_shadowfix_v4_f/splat_shadowfix.ply`, copy in `C:\Users\roach\Downloads\IMG_6556_shadowfix_best.ply`)
as the final shadow-fixed splat for IMG_6556. O stays in exports for the thesis comparison. Final F command:
`fix_splat_shadows.py --ply exports/IMG_6556/splat.ply --colmap data/IMG_6556/colmap/sparse/0_refined
--masks data/IMG_6556/shadow_masks --images data/IMG_6556/images --every 2 --ground-only --dark-gate --lo 0.15 --hi 0.45
--ground-spread 0.06 --ground-tol 0.03 --max-gain 12` (script `b0c0bf8`; defaults `--colour lum --mode target`).

**Update (2026-10-08 22:03 UTC):** after more viewing, the user said **O does a better job on the ground shadows**, so O
becomes the preferred version (Downloads swap queued on the laptop, which was offline). User feedback also raised two
open issues: (1) the splat looks incomplete in the phone viewer, which shows only 300k of 2.89M splats; to check whether
the gaps are also in the full PLY; (2) the shadow on the chair stays, which is expected because `--ground-only` excludes
everything off the ground plane; to find out whether it is self-shading or a cast shadow before designing an object pass.

**Storage clean-up (2026-10-08).** C: was 99% full (942 GB used, ~11 GB free); the WSL disk `ext4.vhdx` was 176.5 GB.
Inside WSL, the main users were: outputs_relightable 37.2 GB (Sep), conda pkgs cache 31.1 GB, data 25.7 GB, 17 shadow-fix dirs
24.0 GB (11.9 GB of it debug PLYs), envs gsplat 17.0 / gsir 11.9 / gaussian_shader 11.8 / sddnet 5.5 / colmap4 4.5 GB,
pip cache 6.6 GB, outputs_gsir 3.3 GB. User approved the "safe set" and ran it themselves (`~/cleanup_shadowfix.sh`). It deleted
the 30 PLYs of the 15 superseded shadow-fix runs (stats.json, shadow_frac.npy and images kept; F and O untouched) and ran
`conda clean -a` and `pip cache purge`. They then compacted the vhdx with diskpart. **Result: 76 GB free on C:.**
Lesson: each shadow-fix test run writes ~1.4 GB (fixed + debug PLY); delete superseded runs' PLYs as you go, and remember
WSL disk space only returns to Windows after compaction.

## New scene IMG_6615 (planned 2026-10-08)

Source: `Downloads\IMG_6615.MOV`: 3840x2160, 30 fps, 177.8 s (5,334 frames), H.264 10.3 Mb/s, yuv420p, bt709 (SDR), 224 MB.
Indoor office or storage room (white table and chairs, dark cabinets, shelves, grey carpet, jacket rack, corkboard, large white walls and
ceiling, fluorescent light). Hand-held walk-around with several tilts to the ceiling; some motion blur. Risk: blank white
surfaces may leave ceiling frames unposed. Laptop state: 8 GB VRAM free, 15 GB WSL RAM (14 available) + 4 GB swap, 811 GB
free in WSL, 75 GB free on C:. nerfstudio 1.1.5, gsplat 1.4.0.
Settings sheet with every tunable and three presets (HQ laptop / Safe / Max detail): project file
`new-splat/IMG_6615-settings.md`. Constraints applied: 650 frames at 1080p (RAM), splatfacto-big `stop-split-at` <= 7000 (VRAM,
IMG_6557 crash), exhaustive matching (better result at school), antialiased + scale regularization for blank walls. Waiting
on the user's preset choice.

**IMG_6615 attempt 1 (2026-10-08 19:58-20:07 laptop time, HQ laptop preset).** `MAX_WIDTH=1920 MATCHING=exhaustive STOP_SPLIT=7000
EXTRA_TRAIN_ARGS="--pipeline.model.rasterize-mode antialiased --pipeline.model.use-scale-regularization True"
./scripts/make_splat_hq.sh videos/IMG_6615.MOV IMG_6615 650 30000` (script `c9edb1b`; video copied into WSL first).
- Frame selection: 1.5 min, sharpness median 89 (all) vs 127 (chosen).
- COLMAP exhaustive: 6.5 min (much faster than the ~50 min estimate). **Split into two models: model 1 = 442 frames (68%), model 0
  = 4 frames** (303, 463, 482, 628). nerfstudio's summary reported "poses for 0.62%" because it read the tiny model (known problem 1).
- `fix_colmap_model.py` refused with "too few frames have camera poses" (exit 1). No training ran. The safety check worked as intended.
- Unposed frames: ~19 short gaps of 5-18 frames where the camera faces plain white wall or ceiling, plus one long gap, frames
  550-594 (~2:30-2:42 of the video).
Next: attempt 2 with 1,000 frames (denser overlap through blank stretches), same flags, scene `IMG_6615_1000f`. Sequential matching
with loop detection was not used because it needs a vocab-tree download. RAM watch: 650 full-res frames used 10-11.6 GB before.

**IMG_6615 attempt 2 (2026-10-08 20:07-20:22 laptop time), scene `IMG_6615_1000f`.** Same flags, 1,000 frames (chosen sharpness
median 111.5). Frames picked 20:07-20:09; COLMAP exhaustive 20:09-20:21 (~12 min; matching on GPU at 80 °C, mapping on CPU; RAM
<= 2.4 GB). Two models: **720 frames (72%)** and 2 frames. The pose fix refused again (< 80%). Unposed: ~20 stretches of 6-15 frames
plus frames 846-914 (~2:30-2:42). Sampled frames are almost all ceiling, fluorescent lights and bare white wall (a few show
the corkboard wall and the table from above). Doubling frame density raised coverage only 68% -> 72%: the textureless views stay
unmatched. Furniture (table, chairs, cabinets, rack) is in the posed set.
Decision: train on the 720 posed frames (user can switch to a stronger matcher or a re-shoot). Added `MIN_POSED_FRAC` env to
`fix_colmap_model.py` (default 0.8) so the threshold can be lowered on purpose; this run uses 0.7. Capture lesson: avoid pointing
at bare ceiling or wall; keep furniture or edges in frame.

**IMG_6615 baseline training (720 posed frames), 2026-10-08 20:23-21:06 laptop time.** `MIN_POSED_FRAC=0.7`, otherwise the
HQ laptop flags (splatfacto-big, full res 1920x1080, stop-split-at 7000, antialiased, scale regularization, SO3xR3, cache cpu,
30k). 43 min, ~10-10.5 it/s after densification. Readings: step 6,070 at 20:28 (111 ms/step), GPU 2.6 GB; after 7k GPU flat at
**2.7 GB** (peak), 78-82 °C; RAM 9.0-9.1 GB with all 720 images cached (14 GB guard never hit). No errors. Output
`exports/IMG_6615_1000f/splat.ply`: **988,855 Gaussians, 245 MB**; config `outputs/IMG_6615_1000f/splatfacto/2026-10-08_202317/`.
Observation: with 720 indoor frames, big capped at 7k used only 2.7 of 8 GB, far below IMG_6556 (7.2 GB at 8.2k). The cap could
go higher on this scene (e.g. 10-12k) for more detail. VRAM growth depends on the scene, not only on frame count.

**hloc install (2026-10-08 20:46-~20:53).** New conda env `hloc` (gsplat untouched): torch 2.4.1+cu124 (GPU OK), pycolmap 4.2.1,
hloc commit c13273b. Script `scripts/hloc_sfm.py` (laptop). SfM on the same 1,000 frames in `data/IMG_6615_hloc` started 21:07:
pairs = next 20 frames + NetVLAD top-20 retrieval; SuperPoint (max) + LightGlue; pycolmap mapping, single shared OPENCV camera.

**IMG_6615 hloc SfM result (2026-10-08 21:07-21:48 laptop time, 41 min, no errors).** **981 of 1,000 frames posed in a single model
(98%)**, against COLMAP exhaustive 720 (72%, two models) on the same frames. Timings: NetVLAD retrieval 76 s (incl. one-time 529 MB
weight download), 26,738 pairs (next 20 + NetVLAD top 20); SuperPoint (max) + LightGlue matching 671 s; pycolmap reconstruction
1,695 s (last ~6 min global bundle adjustment on 8 CPU cores). Peaks GPU 1.4 GB, RAM 4.2 GB. Model: 156,467 points, mean reprojection
error 1.43 px; OPENCV 1920x1080, fx 1564.3, fy 1562.2, cx/cy 960/540 (not refined), k1 0.139, k2 -0.233, p1/p2 ~0. Unposed: 522,
539, 729-737, 801-803, 820, 846, 866-868 (short ceiling glances). The end-of-video corner (846-914), a 69-frame hole with COLMAP,
is now almost fully covered. Output `data/IMG_6615_hloc/colmap/sparse/0`, summary `data/IMG_6615_hloc/hloc/summary.json`, log
`logs/IMG_6615_hloc.log`. **Finding:** on low-texture indoor video, learned features + LightGlue with retrieval pairs solved the
pose coverage problem that denser frames could not (68% -> 72% with COLMAP vs 98% with hloc).

**IMG_6615 training on hloc poses, attempt 1 (2026-10-08 22:09-22:11 laptop time): stopped for RAM.** Prep: `fix_colmap_model.py`
read the pycolmap 4.2.1 model with COLMAP 3.10 without problems (137 s incl. bundle adjustment with principal-point refinement);
transforms.json 981 frames, OPENCV fx 1564.3 fy 1562.2 cx 960.3 cy 539.5 k1 0.139 k2 -0.233. Copied to transforms_nsprocess.json
so `make_splat_hq.sh` skipped steps 1-3. Flags: STOP_SPLIT=12000, antialiased, scale regularization. At ~step 700 **RAM reached
14.8 of 15.5 GB** (1 GB available, 0.9 GB swap in use). The watcher killed it at the 14 GB rule. RAM scales roughly linearly with
cached full-res frames: 720 frames -> 9.0 GB, 981 -> ~15 GB. WSL has no `.wslconfig`, so it gets the default half of 31.7 GB.
Options: raise WSL memory (`.wslconfig` `memory=24GB` + `wsl --shutdown`, needs the user at the laptop), 1600-px frames, or
fewer frames. Chose 3 of every 4 frames (~735, keeping frames next to unposed gaps), scene `IMG_6615_hloc735`, same flags.
**Rule:** on this laptop, full-res 1080p caching allows about 750-800 frames before WSL RAM runs out.

**735-frame subset run (`IMG_6615_hloc735`), 2026-10-08 ~22:05-22:12: stopped by choice at step 1,940** (6.5%, 45.6 ms/step, no
errors) when the user chose to give WSL more RAM and train on all 981 frames. Data and log kept.

**WSL memory raised to 24 GB (2026-10-08 ~22:12 laptop time).** Created `C:\Users\roach\.wslconfig` (none existed before):
```
[wsl2]
memory=24GB
```
then `wsl --shutdown` and restarted Ubuntu from the Windows side (the user did not need to be at the laptop). `free -g`: 23 GB
total, 22 GB available, 6 GB swap (was 15.5 GB). Side effects: the restart ended the old Ubuntu Claude process and closed the
Windows Terminal dashboard window; keep-awake heartbeat, tmux `dash` and ttyd (localhost:7681 only) were restarted by hand.
Windows keeps ~7.7 GB of the 31.7 GB.

**IMG_6615 HQ splat on all 981 hloc frames, attempt 2 (2026-10-08 22:13:14-23:03:06 laptop time, 50 min, no errors).**
`STOP_SPLIT=12000 EXTRA_TRAIN_ARGS="--pipeline.model.rasterize-mode antialiased --pipeline.model.use-scale-regularization True"
./scripts/make_splat_hq.sh videos/IMG_6615.MOV IMG_6615_hloc 1000 30000` (steps 1-3 skipped; log `logs/IMG_6615_hloc_train2.log`).
splatfacto-big, full res 1920x1080, cache cpu, SO3xR3, COLMAP coordinates. Watchers: 5-min progress, RAM stop at 22 GB, VRAM alert
at 7.2 GB, slowdown and crash alerts (re-armed at the 30-min limit).

| Time | Step | ms/step | GPU mem | GPU temp | RAM |
|---|---|---|---|---|---|
| 22:18 | 5,370 | 69 | 3.0 GB | 76 °C | 15.2 / 23.5 GB |
| 22:23 | 8,000 | - | 3.5 GB | 78 °C | 15.2 |
| 22:28 | 10,650 | 118 | 3.0-3.5 GB | 78 °C | 15.2 |
| 22:33 | 13,200 | - | 3.5 GB (peak) | 79 °C | 15.2 |
| 22:38 | 15,950 | ~105 (9 it/s) | 3.5 GB | 79 °C | 15.2 |
| 22:43 | 18,810 | - | 3.5 GB | 81 °C | 15.2 |
| 22:48 | 21,650 | - | 3.5 GB | 79 °C | 15.2 |
| 22:53 | 24,510 | - | 3.5 GB | 79 °C | 15.2 |
| 22:58 | 27,350 | - | 3.5 GB | 81 °C | 15.2 |

No swap used. Export 40 s. Output `exports/IMG_6615_hloc/splat.ply`: **1,302,446 Gaussians, 323 MB** (+32% Gaussians vs the 720-frame
COLMAP baseline, 988,855 / 245 MB). Config `outputs/IMG_6615_hloc/splatfacto/2026-10-08_221319/config.yml`.
**Findings:** (1) 981 cached full-res frames need ~15.2 GB RAM, so the 24 GB WSL limit is what made this run possible on the laptop.
(2) Even uncapped to 12k, this indoor scene peaked at 3.5 of 8 GB VRAM, so VRAM was never the limit here; RAM was.
(3) Total pipeline for this scene: frame pick + 2 COLMAP attempts (both under 80% posed) -> 720-frame baseline (43 min) -> hloc SfM
(41 min) -> WSL 24 GB -> this run (50 min).

## Web-ready SOG exports for the portfolio (2026-10-09)

Request: crop floaters, rotate +Y up, convert to PlayCanvas SOG (single file) under 30 MB each, save to Downloads.
Tool: `@playcanvas/splat-transform` 3.10.1, installed locally on the laptop.

**Orientation.** Room (IMG_6615_hloc, COLMAP frame): up vector from the 981 camera poses. Backyard (`Downloads\splat.ply`, school
IMG_6557): this file is in nerfstudio's z-up frame (school run), not COLMAP; up from a RANSAC ground-plane fit (58% of splats on
the plane). Both centred on the origin. **Gotcha:** splat-transform silently flips `.ply` input 180° about Z into PlayCanvas space;
its rotate convention was measured on a 4-point test PLY before computing the angles.

**Crop.** Box = 1st-99th percentile of opaque splats, padded 10%, then splat-transform's floater filter.
- Room: 1,302,446 -> 1,270,340 (box) -> 1,236,049 (floaters).
- Backyard: 4,154,121 -> 4,083,823 -> 4,063,803.

**Size.**
| Output | Gaussians | SH bands | Size | Alternatives measured |
|---|---|---|---|---|
| `Downloads\room-IMG_6615.sog` | 1,236,049 | 3 | 19,874,798 B (18.95 MiB) | SH2 19.0, SH1 18.4, SH0 15.6 MB |
| `Downloads\backyard-IMG_6557.sog` | 2,300,000 (adaptive decimation) | 0 | 28,461,124 B (27.14 MiB) | full SH, 4.06M: 58.5 MB; SH1 at 2.3M: 33.4 MB |

**Checks.** Both files read back as valid SOG with the expected counts; level renders with +Y up show both scenes upright.
**Not done:** superspl.at/editor check. The laptop's built-in browser blocks superspl.at from loading a local file
(`ERR_BLOCKED_BY_CLIENT`) and can't drive the file picker. User to drag the files onto https://superspl.at/editor, or allow
Claude in Chrome.
**Observation:** the backyard has spiky white/blue splats around the treetops against the sky. They are in the original school
splat (training's sky model), not added by the conversion. Trimming them is offered (risk: clipping branch tips).

## Shadow-free splat plan (brief received 2026-10-12)

Sections below follow the brief's five stages (1 capture/SfM, 2 sun, 3 clustering + orientation, 4 visibility,
5 inpainting), numbered n.m in run order. Steps not in the brief are marked [addition].

Labelled test images for stages 2-4 (20 images with captions, numbers and source folders):
[`docs/figures/2026-10-12_chair_shadow_tests/README.md`](figures/2026-10-12_chair_shadow_tests/README.md). Plain-language
explainer: [`docs/shadow-free-explained.md`](shadow-free-explained.md).
Word document for readers outside the field (2026-10-13): [`docs/shadow-free-process-explained.docx`](shadow-free-process-explained.docx)
with a PDF copy, built on the laptop and ordered by the brief's stages 1-5. It replaced an earlier cloud-built version
(commit 72cbb80, removed with its builder script).

The user posted a project brief: a shadow-free splat for a single dominant light (sun) and one hero object, by
treating shadow removal as surface-aware inpainting instead of BRDF decomposition. Five stages: (1) capture + SfM +
3DGS, (2) sun direction from cast-shadow geometry and shading, (3) surface clustering with a grain direction per
cluster from the Gaussians' own elongation, (4) visibility toward the sun to flag shadowed Gaussians, (5) fill shadowed
Gaussians with colour from lit Gaussians of the same cluster, propagated along the grain. Stage 3 is named as the main
risk, to be prototyped first.

### Brief vs what was already done (assessment, 2026-10-12)

| Stage | Status before this brief | What is new |
|---|---|---|
| 1 Capture + SfM + 3DGS | Done: `make_splat_hq.sh`, splats kept in COLMAP coordinates (needed for stages 2 and 4). | Nothing. A hero-object capture (one object, sun, nothing else casting onto it) does not exist yet; IMG_6556 is chairs on grass with a shed and fence. |
| 2 Sun estimation | Not done. GS-IR (smooth env map, no sun) and the sun + sky survey (all need varying lighting) did not give one. | All of it. The 3D shadow-mask fix already stores a per-Gaussian `shadow_frac` (multi-view SDDNet votes), so the sun can be fitted as the direction whose stage-4 visibility best reproduces those votes on the ground. Stages 2 and 4 then share one ray-casting routine. Shading gradients on one lighting condition are ambiguous with dark materials (same reason GS-IR failed), so they are a secondary cue (inferred, not tested). |
| 3 Surface clusters + grain | Not done. The only surface model so far is the single RANSAC ground plane in `fix_splat_shadows.py --ground-only`. | All of it. Prototype added (below). |
| 4 Visibility check | Partly: `fix_splat_shadows.py` has a z-buffer visibility test toward each camera, not toward a light. | Ray casting toward the sun. **Gap in the brief:** a blocker test only finds cast shadows. Surfaces facing away from the sun (attached shadow, e.g. the chair sides SDDNet flagged on 2026-10-08) are not blocked by anything; they need a normal-vs-sun test, which needs signed normals (orient each normal toward the cameras that see it). |
| 5 Orientation-guided inpainting | The 3D shadow-mask fix brightens with a global gain (best version O: shadow/sunlit 0.76-0.82, cores lag, texture stays flat). | Replaces the gain with copying from lit Gaussians of the same cluster. Rounds 7-8 already showed why "same material" matters (the colour gate) and why a gain can't restore detail the shadow never recorded; the brief's rule answers both. |

Risks noted for stage 3 (hypotheses to test, not results): (a) 3DGS densification also stretches Gaussians along image
edges and silhouettes, so elongation may follow capture artefacts as well as true grain; (b) grass is volumetric, not a
surface, so the ground will not form clean flat clusters (expected low flatness and low coherence there); (c) one 3D
"dominant direction" per cluster is only meaningful for flat clusters; on curved ones the per-Gaussian local field is
what stage 5 should use (seen in the synthetic sphere below).

### Decisions (2026-10-12)
- **Tooling:** plain Python with numpy + scipy (cKDTree, sparse connected components), same PLY reader as
  `fix_splat_shadows.py`. No new conda env, no deep-learning dependency, no training. It runs on an exported PLY after
  training, so it adds no time to the training pipeline (seconds on the synthetic scene; minutes estimated for a
  cropped real splat, not measured yet).
- **Clustering rule:** neighbour links between flat Gaussians (shortest/middle scale < 0.3) within 3x splat size whose
  unsigned normals differ by < 20 deg, then connected components (min 200). Colour is not used, so a shadow edge cannot
  split a surface. Local angle tests let curved surfaces grow while creases split.
- **Grain:** longest axis projected into the tangent plane, weighted by opacity x in-plane elongation; structure tensor
  per cluster (dominant direction + coherence) and per Gaussian over same-cluster neighbours.
- **First real test:** IMG_6556 (it has masks, `shadow_frac` and baseline O for later comparison), cropped to one chair
  with `--crop`. A dedicated hero-object capture is the user's call.

### Stage 1: capture and SfM

Done before the brief: IMG_6556 video, COLMAP poses, splatfacto-big splat `exports/IMG_6556/splat.ply` (2.9M
Gaussians, COLMAP coordinates). See the IMG_6556 rows earlier in these notes.

### Stage 3: surface clustering and orientation

#### 3.1 Prototype: `scripts/cluster_surfaces.py` (2026-10-12, cloud)

Synthetic check (`--selftest`): 15,613 Gaussians: 4 x 4 ground with grain along x and its left half darkened like a
shadow, a 0.8 m box with vertical grain on the sides and grain along y on top, a sphere with grain along its meridians;
8 deg random in-plane noise on every grain, 2 mm position noise. Result: **7 clusters, each group 100% in its own
cluster, darkened half not split off.** Dominant grain: ground (-1.0, 0.002, 0) coherence 0.96, box sides vertical
coherence 0.96, box top along y 0.96. Sphere: dominant (0.007, 0, 1.0) coherence 0.66 (meridians all lean vertical, so a
single direction is only a summary); per-Gaussian local grain error vs the true meridian: median 1.7 deg (ground 1.4 deg).
Runtime 0.2 s. `--crop` checked on the same scene (box only).

Problem found and fix:
1. **First selftest "failed" on the sphere.** The test expected low coherence on a sphere, but meridian directions share
   a vertical component, so the cluster tensor gave 0.66. The expectation was wrong, not the code. Fix: the check now
   measures the per-Gaussian local field against the true meridians (median 1.7 deg).

Next (laptop, no training, needs the user's go-ahead): run on `exports/IMG_6556/splat.ply` cropped around one chair,
view `clusters_debug.ply` and `orientation_debug.ply`, and record whether seat, back and legs form separate clusters,
whether the shadow on the chair splits anything, and how flat/coherent the grass is. Planned metric for grain without
labels: compare each cluster's projected grain with the image structure-tensor direction in a few frames.

#### 3.2 Chair runs (laptop, 2026-10-12)

**First real run: IMG_6556 chair (2026-10-12, laptop, no training).** Script from `claude/project-thread-txhi4k`,
conda env `gsplat` (numpy 1.26.4, scipy 1.15.3); laptop selftest PASSED in 1.0 s. Commands were run in a visible tmux
window `work` (log `~/work_session.log` in WSL), as the user asked.
- Chair location: ground plane from `exports/IMG_6556_shadowfix_v8_o/stats.json` (n = [-0.0089, -0.9496, -0.3134],
  d = 1.787). 212,323 opaque Gaussians lie 5-60 cm above it. The densest 50 cm blob near the camera centroid (8,076 pts,
  centre [0.714, 1.173, 0.492]) is the chair. Crop: `--crop 0.713 1.091 0.464 1.065` (about 0.8 m). The metre-to-unit
  conversion, 1 m = 1.331 units, **assumes** a hand-held phone at 1.4 m; the true scale is unknown.
- Outputs: `exports/IMG_6556_clusters_chair/` and `exports/IMG_6556_clusters_chair_a30/`. About 15.5 s per run wall time,
  most of it PLY loading; clustering itself takes 1.7-1.9 s.

| Run | Kept / flat | Links kept | Components -> clusters >= 200 | Cluster 0 | Cluster 1 |
|---|---|---|---|---|---|
| angle 20 | 76,211 / 31,473 (41%) | 185,321 of 503,568 | 5,605 -> 2 | 31,108 Gaussians, normal spread 51.9 deg, coherence 0.53 | 618, thin strip, spread 20.7 deg, coherence 0.85 |
| angle 30 | same | 244,548 | 3,325 -> 2 | 31,770, spread 52.5 deg, coherence 0.54 | 6,787, ground patch at about 0 cm, spread 48.4 deg, coherence 0.02 |

**Result: negative on the first try. The chair is not split into seat, back and legs.** Almost every flat Gaussian ends
up in one cluster spanning about 52 deg of normals. Diagnosis (inferred, being tested): the neighbour-to-neighbour angle
test **chains**. Noisy real normals give many small steps under 20 deg, which link across creases that a clean synthetic
box never has. At the same time the graph is very fragmented: 5,605 components, so most Gaussians fall in clusters
below 200. The ground under the chair hardly forms a cluster at all (grass is volumetric; this matches risk (b)).
Next: a sweep of angle 10 / 15, radius 1.5 and stricter opacity/flatness, plus a check of the shortest-axis normal
against a neighbour-PCA normal. Planned code change if chaining is confirmed: grow each region against its seed or
running-mean normal rather than only the neighbour's normal (PCL-style region growing with a curvature limit).

**Sweep and normal check (2026-10-12, laptop, same crop).** Tighter plain-component settings only fragment the chair:
angle 10 gives 16 clusters, angle 15 + radius 1.5 gives 15, angle 15 + opacity 0.7 + flat 0.2 gives 10. Every cluster
still spans 20-48 deg of normals. **Normal check:** angle between each flat Gaussian's shortest axis and a PCA plane fit
to its 16 nearest neighbours. 0-5 cm above the ground plane: median 44.7 deg, p90 80.3 deg (n = 3,964). More than 5 cm
above: median 23.7 deg, p90 64.1 deg (n = 21,379). Below the plane: median 47.5 deg, p90 80.9 deg (n = 6,130). So on real
splats the shortest axis is a poor normal, and close to random on grass. On the synthetic scene the two agree to a
median of 1.1 deg. **Chaining confirmed as the cause.**

Fix (commit `a40e125`): `--region-angle A` (seeded region growing; a Gaussian joins only if its normal is within A of
the region's running mean normal) and `--normal-source axis|pca|agree` (agree = PCA normal, dropping Gaussians whose
axis and PCA normals differ by more than `--agree`). Selftest updated: it now also checks cluster purity (> 0.95), and
in region mode the sphere may split into patches. It passes in all four modes. With PCA normals, about 4% of
crease-edge splats land on the adjacent face (purity 0.96).

| Run (all `--crop 0.713 1.091 0.464 1.065`) | Flat used | Regions -> clusters | Notes |
|---|---|---|---|
| (d) `--region-angle 30` | 31,473 | 6,489 -> 18 | backrest still one 14.8k cluster |
| (e) `--region-angle 30 --normal-source pca` | 31,473 | 6,864 -> 21 | axis vs PCA in crop: median 29.7 deg, p90 73.3 deg |
| **(f) `--region-angle 30 --normal-source agree --agree 30`** | 15,862 (15,611 dropped) | 2,345 -> 14 | **best: chair split into its parts** |
| (g) `--angle 15 --region-angle 20 --normal-source agree` | 15,862 | 3,463 -> 13 | similar to f, fewer clustered (24.5k) |

**Result (f), checked visually** on `exports/IMG_6556_clusters_chair_r30agree/cluster_views.png` (top view and two
side views, 50k of 57,557 opaque Gaussians). The chair is an Adirondack chair: seat about 0.4 m, armrests about
0.62 m, backrest top about 1.1 m. Those are normal proportions, which supports the assumed scale (1 unit = 0.751 m). The
clusters follow the parts: the fanned backrest slats form one large cluster plus one for the top curve and outer slats;
the seat is one cluster (normal 16.7 deg from up); each armrest is its own cluster; front and back legs are separate.
The ground stays unclustered, and there is no visible bleed between backrest and seat.

Problems found and fixes (stage 3):
2. **Reported normal spread was inflated (40-45 deg on big clusters, above the 30 deg region limit).** The stat
   included flat Gaussians from regions under `--min-size`, which are attached to the nearest cluster afterwards
   with no normal test. In (f) only about 8.2k of the 30,070 cluster members were grown; 7.7k small-region flat and
   14.2k non-flat Gaussians were attached. Fix: normal stats now use grown members only, and stats.json and the
   printout add a `grown` count. **Caveat for the thesis:** most cluster members are assigned by nearest-neighbour
   attachment, not by the normal test. Attachment quality needs its own check before stage 5 relies on it.

**(f) rerun with corrected stats (2026-10-12, script `b883c48`, `exports/IMG_6556_clusters_chair_r30agree_v2/`).**
Clustering is identical to (f): 2,345 regions, 14 clusters, 8,193 grown members (= 76,211 kept - 68,018 attached),
2.6 s script time, 17.8 s wall time. **Normal spread over grown members is 12.1-20.5 deg in every cluster** (previously
reported as 25-44 deg), inside the 30 deg region limit. Grain coherence is 0.44-0.94. The seat (35 cm, 834 grown of
3,208) has coherence 0.44, the lowest. It is plausibly made of slats running across the seat, but that is not checked
against the photos yet. Largest: backrest 9,095 (1,872 grown), spread 17.0 deg, coherence 0.75. Picture copied to the
Windows repo folder as `chair_clusters_r30agree.png`.
Open for stage 3: (1) check the 73% of members that come from attachment (for example, the normal test
against their cluster for attached flat Gaussians, or a visual check per part); (2) validate the grain against the
image structure tensor in a few frames; (3) repeat on a second object.

#### 3.3 Attachment and grain checks against the photos

**Attachment and grain checks (2026-10-12, laptop; scripts `1b0b942` and `check_grain.py`).**
Attachment (`--attach-angle`, rerun of (f) as `..._r30agree_v3` without and `..._r30agree_att30` with a 30 deg limit):
of the 4,457 flat Gaussians that are attached after growing, only 7% face within 30 deg of their cluster normal
(median 59.7 deg off). With the limit, 293 stay attached; clusters lose 10-20% of members; grown cores and spreads are unchanged.

Grain vs photos (`check_grain.py` on `..._r30agree_v3`, 6 frames, one per 60 deg azimuth bin; image texture direction =
along-edge eigenvector of a Sobel structure tensor, sigma 3 px, coherence > 0.3; surfaces seen at |n . ray| < 0.4
skipped, because a first version without that gave unfairly easy baselines):

| Pixels | n | Grain median | Rot90 median | Random median | Grain < 20 deg | Random < 20 deg |
|---|---|---|---|---|---|---|
| edges (above median gradient energy) | 17,324 | 5.5 deg | 81.8 deg | 43.5 deg | 87% | 23% |
| **interior (> 8 px from Canny edges)** | 3,941 | **9.1 deg** | 79.7 deg | 43.6 deg | **74%** | 23% |

Interior by group: grown 10.0 deg, attached flat 5.9 deg, attached non-flat 9.6 deg. Per cluster 3-14 deg for most;
cluster 6 21 deg, cluster 9 17 deg. With `--attach-angle 30` the interior median worsens to 10.6 deg.

**Result: the Gaussians' elongation follows the visible surface texture (the slat wood-grain emboss), not just outlines.**
The interior score is the meaningful one: densification stretches Gaussians along image edges, so edge agreement is
partly built in. Most interior misses lie where the cast-shadow edge crosses the back slats; there the photo's texture
direction follows the shadow line, not the material, which is the behaviour stage 5 needs (inferred from the overlays
in `grain_check_interior/`).

**Decision (2026-10-12): keep `--attach-angle` off by default.** Attached flat Gaussians have wrong normals but the best
grain, so they stay as grain carriers. The normal test decides surface membership only while regions grow.
Stage 3 on the chair: **done** (clusters = chair parts, normal spread 12-21 deg, grain verified at 9 deg median).
Carried forward: signed normals for stage 4 (PCA normals oriented toward the observing cameras); the ground has no
usable surface or grain (grass); repeat on a dedicated hero-object capture.

Process problem (laptop terminal): two tmux `send-keys` commands were glued onto one line (`...log{ python ...`):
the previous command had been recalled from shell history at the prompt, and the next one was typed after it, so bash
gave `syntax error near unexpected token '}'`. The user spotted it in the visible terminal. Fix: clear the prompt line
(C-u) before typing, never send while the pane is busy, and wait for each command's done marker. Typing in that pane while
Claude works gets wiped, so use a second tmux window (Ctrl-b c) for manual commands.

### Stage 2: sun estimation

#### 2.1 First fit: all casters, area IoU (2026-10-12)

`scripts/fit_sun.py` (commits `b888783`, `159982f` progress output, `be856e1` optional `--zbuf-radius`). For each
candidate sun direction s, it builds an orthographic shadow map of the opaque Gaussians more than 5 cm above the ground
plane (2 cm cells, dilated by 1 cell) and predicts which ground receivers are shadowed. It scores IoU against the
receivers' own multi-view SDDNet votes (every 2nd frame, z-buffer visibility, at least 5 views, shadow if 50% or more of
votes). The search runs over a 5 deg grid, then 1 deg, then 0.25 deg. Uncertainty = angular radius of fine-grid directions
within 0.005 IoU of the best. Hold-out = refit on each half of the receivers, split through the centre. Self-test (box +
pole on a plane, true sun elevation 38 / azimuth 127, 10% label noise): fitted 38.25 / 126.75, **error 0.35 deg**; same
on the laptop.

**Ground truth.** The user supplied a sun calculator readout: altitude 55.51 deg, azimuth 172.57 deg (sunrise 07:16:00,
culmination 13:14:48, sunset 19:13:08). The video metadata confirms the time and place: creation_time
2026-09-28T16:58:11Z (12:58:11 EDT), GPS +32.0482 -081.0601. An independent NOAA calculation gives 55.5 / 172.6 deg. The user notes the readout may be off by about 0.1 deg; that is well below the fit's spread (about +/-2 deg), so it does not affect any conclusion here, and errors under about 0.1 deg are not meaningful.

**First run on IMG_6556** (`--centre 0.713 1.091 0.464 --radius 3.0`, out `exports/IMG_6556_sun/`, 10 min 47 s, almost
all of it the vote pass over about 325 frames with all 2.9M opaque Gaussians in the z-buffer):
fitted **elevation 57.0 deg vs true 55.51 deg (error +1.5 deg)**. Uncertainty radius 3.1 deg; the two hold-out half-fits
agree within 0.4 deg. In `sun_fit.png` the chair's predicted shadow footprint overlaps the observed one almost exactly,
including the gap under the seat. Overall IoU is only 0.14, because scattered grass blades and floaters more than 5 cm
above the plane cast small false shadows across the lawn. They lower the score but don't move the fit.
Azimuth can't be checked against 172.57 deg: the splat has no north reference.

Problem found and fix (stage 2):
1. **The terminal looked frozen for 10+ min.** Python block-buffers stdout when piped to `tee`, and the vote pass printed
   nothing until the end. Fix: run with `python -u`; the script now prints progress every 25 frames. Optional
   `--zbuf-radius` limits the view z-buffer to nearby Gaussians for speed.
Next: sensitivity runs (`--cell 0.03`, `--radius 2.0`, `--occluder-min` about 11 cm) and the azimuth relative to the camera path.

**Sensitivity run: `--cell 0.03`** (votes reused, 162 s): elevation 58.00 deg, azimuth 148.25 deg, IoU 0.098, uncertainty
5.45 deg (245 tied directions). Error vs truth +2.5 deg, inside its own uncertainty. The coarser cells widen the tiny false
shadows from grass and floaters, so the score drops and the uncertainty grows; 2 cm stays the default.

#### 2.1a Refinement options from the literature (survey, 2026-10-12)
The user asked how to cut the sun-fit error. Options found, ranked by fit to our setup (none implemented yet):
1. **Soft visibility (opacity transmittance) instead of a binary shadow map.** ShadowGS (arXiv 2601.00939) casts a ray
   from each Gaussian toward the sun and multiplies (1 - alpha) over the Gaussians it hits, as in 3DGRT/R3DG ray tracing.
   For us: thin, low-opacity grass and floaters would stop casting full-strength false shadows. Scoring changes to the
   continuous vote fraction instead of thresholded IoU. ShadowGS takes the sun from satellite metadata; it does not
   estimate it.
2. **Object-shadow association.** Instance Shadow Detection / LISA (Wang et al., CVPR 2020) pairs each shadow with its
   casting object and predicts light direction. Our 3D equivalent comes for free from stage 3: use only the chair's
   clusters as occluders, and score only receivers near the chair's predicted shadow. This targets the clutter that
   currently drags IoU to 0.14.
3. **Shadow-tip correspondences.** Classical light-from-shadows geometry (e.g. Cao & Foroosh, CVPR 2005): each
   object point / shadow point pair, such as a leg tip and its shadow tip, defines one sun ray, combined robustly
   (RANSAC). This gives an independent geometric estimate to cross-check the IoU search.
4. **Combine cues with a prior.** Lalonde, Efros & Narasimhan (ICCV 2009, IJCV 2012) combine shadows, shading of
   vertical surfaces and the sky in a probabilistic sun estimate. For us: a Lambertian shading cue on the lit,
   same-material stage-3 clusters (brightness vs n . s) as a second likelihood, multiplied with the shadow likelihood.
   This would mainly sharpen elevation.
5. **Metadata prior.** Phone videos carry time and GPS (IMG_6556 does), which fixes elevation and true azimuth with
   no added pipeline time. The shadow fit then only has to find the scene's north (a 1D search), and stage 2 becomes a
   check, not the sole estimate.
Caveat for claims of improvement: there is one ground-truth sample so far (55.51 deg). More test cases are needed
before reporting an error rate, for example separate fits on the other chairs in IMG_6556 and future captures with metadata.

#### 2.2 Chair-only casters (laptop, 2026-10-12)
The user pointed out that the grass is lit mostly by diffuse skylight, so its blades are not real sun casters. Commit
`d7b68f1` added `--occluder-labels` (only Gaussians in a stage-3 cluster cast, here
`exports/IMG_6556_clusters_chair_r30agree_v3/clusters.npz`) and `--score-radius` (score only ground receivers within that
distance of the chair's footprint). Self-test unchanged (0.35 deg). Results vs the true elevation 55.51 deg:

| run | elevation | error | IoU | uncertainty | hold-out half fits |
|---|---|---|---|---|---|
| all casters, radius 3.0 | 57.00 | +1.5 | 0.14 | 3.1 | 56.75 / 56.5 |
| all casters, cell 0.03 | 58.00 | +2.5 | 0.10 | 5.5 | 58.0 / 58.0 (4.2 deg apart in azimuth) |
| all casters, radius 2.0 | 57.50 | +2.0 | 0.24 | 2.0 | 57.5 / 58.0 |
| chair only, score radius 1.8 (`exports/IMG_6556_sun_chair`) | 57.75 | +2.2 | 0.67 | 1.7 | 57.25 / 56.75 |
| chair only, score radius 1.2 (`exports/IMG_6556_sun_chair_s12`) | 55.00 | -0.5 | 0.69 | 1.0 | 57.0 / 56.0 |

With grass excluded, the IoU rises from 0.14 to about 0.69. In `sun_fit.png` the predicted chair shadow matches the
observed one: seat, back and arm outlines and the lit gap under the seat. The only misses are a thin rim where the
prediction is slightly fatter and a leg tip about 10 cm too long. Chair-only runs take 1-3 min (the first full run took
10.8 min). Fitted azimuth 144.75 deg in scene coordinates; it can't be checked against the true 172.6 deg compass
azimuth without a north reference.

Reading: the -0.5 deg of the best run is partly luck. Its half fits land at 56-57 deg and the five settings spread over
55-58 deg, so the method is good to about +/-2 deg on this capture, with a slight high bias (most runs are 1.5-2.5 deg
high). The user notes the grass was not a capture target in IMG_6556, so its reconstruction quality varies; the ground receivers sit on it. Possible causes: the ground plane is fitted to grass tops rather than soil (shadows fall a few cm lower), the
masks may be slightly fat, or the plane is not perfectly level. Still one ground-truth sample.

#### 2.3 Chair self-shadow as extra evidence (2026-10-12; parked)
The user asked to also use the chair's own shadow. `--self-shadow` (needs `--occluder-labels`) makes the chair's
clustered Gaussians receivers too:
- Mask votes for them come from the same multi-view pass, cached in `<out>/votes_self.npz`. The pass also sums the unit
  directions to the cameras that see each Gaussian.
- Normal = its cluster's dominant normal (grown members), signed to face those cameras. Stage 4 needs the same signed
  normals.
- Predicted shadow = **attached** (n . s < 0, facing away from the sun) OR **cast by another chair part** (the same
  orthographic shadow map built from the chair only, with `--self-eps 0.05` and no dilation, so a slanted slat does not
  shadow its own neighbours: with 2 cm cells and 1-cell dilation, a 2 cm margin falsely shadows any surface more than
  about 27 deg off facing the sun).
- Score = (1 - `--self-weight`) x ground IoU + `--self-weight` x chair IoU (default 0.5); both IoUs are reported, and
  the hold-out splits both receiver sets.
Self-test extended: 10,000 receivers on the box's top and side faces with random normal signs, signed back by noisy
camera directions (4 sign errors), 10% label noise. Fitted 38.25 / 127.00 vs true 38 / 127, **error 0.27 deg**, chair
IoU 0.78; the ground-only case still gives 0.35 deg.
Caveat: SDDNet is trained on cast shadows, and how it marks self-shaded chair sides decides how much this helps.

Planned self-shadow runs (laptop, tmux `work`, `python -u`, one command at a time, nothing deleted or trained):
1. Fetch, then `python -u scripts/fit_sun.py --selftest --out /tmp/sun_selftest`: two fits (0.35 and 0.27 deg), PASSED.
2. `exports/IMG_6556_sun_self_s12`: copy in `exports/IMG_6556_sun/votes.npz`, then the chair-only score-radius 1.2
   command plus `--self-shadow --zbuf-radius 3.0`. A new vote pass for the chair Gaussians writes `votes_self.npz`.
3. `exports/IMG_6556_sun_self_s18`: same with `--score-radius 1.8` (copy both vote files in first).
4. `exports/IMG_6556_sun_self_only`: step 2 with `--self-weight 1.0`, i.e. the chair's own shadow alone (copy both vote files).
Report per run vs 55.51 deg: elevation, error, score, ground and object IoU, uncertainty, hold-out halves, and the
object observed / predicted / attached shadow shares from `sun.json`.

**Self-shadow results (laptop, 2026-10-12, `scripts/fit_sun_self.py`, an untracked laptop script that reuses
`fit_sun.py`; same idea as `--self-shadow`).** True elevation 55.51 deg:

| run | elevation | error | uncertainty | ground IoU | chair IoU |
|---|---|---|---|---|---|
| ground shadow only (chair casters, score radius 1.2) | 55.00 | -0.51 | 1.0 | 0.69 | - |
| chair self-shadow only | 29.25 (wrong azimuth too) | fails | - | - | 0.39 at its own best |
| combined | 55.50 | -0.01 | 2.5 | 0.69 | 0.30 |
| combined, first version (a part could shadow itself through its thickness) | 55.25 | -0.26 | - | - | - |

The combined hit is mostly luck: self-shadow alone cannot find the sun, and it widens the uncertainty from 1.0 to
2.5 deg. Shadows cast by one part onto another agree well (legs, under the seat and arms). The backrest fails: the masks
call nearly all of it lit, front and back faces alike, while the geometry puts the side away from the sun in shadow.
Likely cause: the slats are 2-3 cm thick, but the vote z-buffer tolerance (`--z-tol 0.03`, relative) is about 7 cm at
that distance, so back-face Gaussians count as visible whenever the front face is and inherit its lit vote. Fix needed
before self-shadow can help: front/back-aware votes (tighter depth test, and only count a view when the Gaussian's
outward normal faces that camera). Decision: the ground shadow stays the sun estimate; self-shadow is parked until
the votes are fixed.

Literature notes from this round: Wang & Samaras (known shape, shading plus cast shadows), Ling et al. ICCV 2009
(L1-regularised least squares with cast shadows), Nillius & Eklundh, and outdoor relightable splat methods GaRe
(arXiv 2507.20512), OSDR-GS (IJCAI 2025) and ROSGS, which keep an explicit sun and ray-traced visibility fitted to
photos. None found estimates the sun from shadows alone as done here. Candidate next steps: (1) score the shadow
boundary (edge distance) instead of area IoU, (2) a shading term on sun-lit chair faces, (3) front/back-aware
self-shadow votes, (4) continuous refinement after the grid search.

#### 2.4 Shadow-boundary scoring (laptop, 2026-10-12)
`scripts/fit_sun_edges.py` (laptop, untracked; reuses `fit_sun.predict`, `plane_frame`, `sun_vec`). Same chair casters
and ground votes as `exports/IMG_6556_sun_chair`. It rasterises the observed and predicted ground shadow, scores the
symmetric chamfer distance between their boundaries, searches coarse-to-fine, polishes with Nelder-Mead, and refits
on two halves. True elevation 55.51 deg (+/-0.1):

| score | score radius | elevation | error | uncertainty | half fits |
|---|---|---|---|---|---|
| area IoU | 1.2 | 55.00 | -0.51 | 1.0 | 57.0 / 56.0 |
| area IoU | 1.8 | 57.75 | +2.24 | 1.7 | 57.25 / 56.75 |
| boundary | 1.2 | 56.30 | +0.79 | 1.2 | 57.25 / 56.0 |
| boundary | 1.8 | 56.98 | +1.47 | 1.4 | 57.5 / 57.25  (3.24 deg apart, mostly azimuth; 24,445 casters, 509,739 receivers, 4 min 23 s) |

Outlines agree to 0.020 units (about 1.5 cm, one raster cell). Changing the score radius now moves the answer by
0.7 deg instead of 2.75 deg, so boundary scoring is **more stable, not more accurate**. The error is almost always
positive (0.8-2 deg high), which points to a systematic input bias rather than noise. Candidates, in test order:
1. ground plane fitted to grass tops, so modelled shadows come out shorter and the sun too high (test: shift the
   plane down 1-3 cm);
2. fuzzy chair top edge (the opposite effect);
3. mask bias (fat or thin SDDNet masks);
4. plane tilt (the plane's up direction vs the camera up: 0.998, about 3.6 deg).
More captures with known times will separate a constant bias from scene noise.

#### 2.5 Grass-height (receiver offset) test

**Receiver-height test (laptop, 2026-10-12, `--receiver-shift`, boundary score, score radius 1.2).** Hypothesis: the
grass Gaussians sit above where the shadow shows in the photos (on the soil between blades). Positive shift = receivers
moved down, in scene units (0.02 = about 1.5 cm):

| shift | elevation (Nelder-Mead) | error | chamfer | IoU | grid uncertainty |
|---|---|---|---|---|---|
| +0.02 (down 1.5 cm) | 57.30 | +1.79 | 0.0198 | 0.680 | 1.76 |
| 0 | 56.30 | +0.79 | 0.0200 | - | 1.2 |
| -0.02 (up 1.5 cm) | 55.03 | -0.48 | 0.0193 | 0.693 | 1.03 |
| -0.04 (up 3 cm) | 54.86 | -0.65 | 0.0189 | 0.685 | 0.4 (but hold-out halves 56.0 / 54.0) |

Lowering the receivers raises the fitted sun (the same outline on lower ground needs a steeper sun), so "grass sits too
high" is ruled out in that form. The elevation moves about 1.1-1.3 deg per 1.5 cm of receiver height, and zero error
sits near a 1.2 cm upward shift. The raised version also fits the outline slightly better (lowest chamfer, highest
IoU), so the data itself weakly prefers it. Effective equivalents of "receivers higher": casters (chair splats)
slightly too tall or fat, or masks slightly wide. The capture also shows how sensitive elevation is to centimetre-level
geometry: about 1 deg per cm here. The 3 cm run shows the curve flattens above 1.5 cm. Edge distance keeps falling with height while IoU peaks at 1.5 cm, so the two scores disagree on the best offset and the differences are under one raster cell. Height and elevation partly trade off. A joint fit of sun plus receiver offset should bound the offset (0 to 3 cm) and report the elevation spread over plausible offsets as part of the uncertainty. Selecting the offset by the fit's own score (no use of the true value) gives 54.9-55.0 deg, about 0.5-0.65 deg from truth.

#### 2.6 Joint fit of sun + grass offset (stage 2 result)

**Joint fit of sun + receiver offset (laptop, 2026-10-12, `fit_sun_edges.py --fit-shift LO HI STEP`, 3 min 49 s).**
For each receiver offset from 6 cm up to 2.3 cm down, in 0.75 cm steps, it solves for the sun, keeps the offset with the
lowest outline (chamfer) distance, and refines the sun there. The true value is never used.

| receiver offset | elevation | chamfer |
|---|---|---|
| up 6.0 cm | 53.75 | 0.0200 |
| up 3.0 cm | 54.75 | 0.0194 |
| **up 2.3 cm (picked)** | 55.00 | 0.0193 |
| up 1.5 cm | 55.00 | 0.0198 |
| up 0.8 cm | 55.75 | 0.0196 |
| as is | 56.75 | 0.0205 |
| down 1.5 cm | 57.25 | 0.0199 |
| down 2.3 cm | 56.25 | 0.0195 |

Final: offset up 2.3 cm, **elevation 55.08 deg vs true 55.51 (error -0.43 deg)**, azimuth 145.1 deg (scene frame);
hold-out halves 55.75 / 55.25. Caveat: the offset is only weakly determined. The score varies by about 6% across the
scan and the profile is bumpy (down 2.3 cm nearly ties the winner). Reported result for IMG_6556: **55 +/- 1 deg**.
More captures will show whether the offset is constant (fix it once) or scene-specific (keep the joint fit).

Stage 2 progression on IMG_6556 (true 55.51 deg): area IoU with all casters 57.0 (+1.5); chair-only casters 55.0
(-0.5, unstable across settings); boundary score 56.3 (+0.8, stable); boundary + fitted receiver offset 55.08 (-0.43,
halves 55.25 / 55.75). Self-shadow is parked (thin-slat vote leak).

### Stage 4: visibility check

#### 4.1 Sun-blocked test, plus facing-away test [addition] (laptop, 2026-10-12/13)
`scripts/shadow_visibility.py` and `scripts/shadow_overlay.py` (laptop, untracked at the time). Output
`exports/IMG_6556_shadow4_v2` (per-Gaussian flags, debug PLY of the region: red cast, blue attached, magenta both;
photo overlays). About 15 s per run with cached votes.
- Sun: the stage 2 joint fit (elevation 55.07 deg). Casters: chair clusters only.
- Cast shadow: shadow map from the chair; chair parts only shadow other parts (not themselves through their thickness).
- Attached shadow: normal faces away from the sun. 6,861 trusted chair Gaussians got normals signed toward the cameras
  that see them (median 131 of 163 frames); the rest copy their nearest trusted neighbours.

Scores against SDDNet mask votes:

| receivers | precision | recall | IoU | accuracy | predicted / observed shadow |
|---|---|---|---|---|---|
| ground | 0.71 | 0.98 | 0.70 | 97% | thin extra rim |
| chair, all | - | - | 0.25 | 57% | 53% / 18% |

Per chair cluster: 84-100% agreement on the seat, arm tops and some legs (clusters 1, 5, 7, 8); 26-47% on the backrest
front (clusters 0, 2, 4), which the masks call lit and the model calls partly shadowed. 19% of chair Gaussians are
within about 9 deg of grazing (a 1-2 deg sun error flips them); excluding them barely changes the score. Re-collecting
chair votes with a much tighter depth test changed little (observed shadow 18.3% vs 19.8%), so the thin-slat leak is
not the main cause.
Photo overlays: front view (frame 263) has the predicted ground shadow on the real one; the backrest front is lit in
the photo except a dark arc near the top, and the model gives a red/cyan mix there. Rear view (frame 540): the back of
the backrest is dark, the model marks it shadowed and SDDNet outlines it too, yet the vote score counts it as a miss
because most backrest votes come from front (lit-side) views. So the chair IoU undercounts; the overlays are the better
check there.
Reading: ground shadow and part-on-part shadows are good enough for stage 5. The backrest front (near-grazing sun,
slats partly shadowing each other) needs a soft sun amount (n.s times sun visibility) or a check against image
brightness rather than a yes/no flag.

#### 4.2 [Addition] Shadow on the chair from photo brightness (laptop, 2026-10-13)
Focus on shadows on the chair itself (the stage 4 weak spot). `scripts/detect_object_shadow.py`
(laptop, untracked at the time), about 11 s per run, no training.
1. Each chair Gaussian is looked up in every photo that sees it (front-most surfaces only; about 128 of 163 frames),
   and the median brightness is taken.
2. The chair is one plastic, so brightness is mostly lighting. The histogram is bimodal: sunlit about 0.3-0.5, shaded
   about 0.02-0.04 (linear), a ratio of about 13x, matching the sun/sky ratio measured earlier on the grass. Split at
   the Otsu threshold.
3. Clean-up: majority vote over the 15 nearest Gaussians in the same cluster (changed 4.6% of labels).
Result: 45% of the chair in shadow. Per cluster: under-seat and inner legs (6, 7, 11, 13) 83-95% shadow; seat front
edge and arm tops (5, 8, 12) 0-3%; backrest (0, 2, 4) mixed, 39-69%, which matches the part-lit backrest in the photos.
Overlays: frame 263 (front) shows the dark arc on the upper backrest marked shadow and the lit band below it marked lit;
frame 317 (three-quarter) agrees too.
Cross-checks: vs the independent geometry prediction (stage 4), precision 0.76, IoU 0.54, accuracy 71%. vs SDDNet,
IoU 0.29: SDDNet calls only 18% of the chair shadowed and misses chair shadows in the overlays (dark navy plastic).
Weak spots: thin slats (front/back faces mix in projection, which adds brightness noise); grazing-lit surfaces can be
as dark as shade (geometry can separate those); one chair, one scene.
Next: combine brightness (decides) with geometry (tie-break at grazing angles and on thin parts). Fine-tuning the 2D
shadow detector on these labels would be training: held until the user approves it.

#### 4.3 / 4.4 [Addition] Combined chair labels and detector fine-tune (laptop, 2026-10-13)
1. **Combined labels:** brightness label kept, except near the shadow edge where the sun is near grazing (16% of chair
   Gaussians); there the geometry decides. Result: 46% of the chair in shadow.
2. **Targets** (`exports/IMG_6556_chair_targets`, 325 frames, 18 s): original SDDNet mask everywhere, chair labels
   projected onto the chair, a 3 px ring around the chair outline and pixels far from any projected point marked
   ignore, median-filtered. On the chair the targets say 50% shadow (SDDNet: 35%).
3. **Fine-tune** (`scripts/finetune_sddnet.py`, laptop, untracked at the time): from the SBU checkpoint, shadow-mask
   output only (EfficientNet-B3 encoder + decoder), 4 epochs, AdamW lr 2e-5, batch 4, 512 px, random horizontal flips,
   binary cross-entropy with ignore pixels skipped, normalisation statistics frozen; targets from
   `scripts/make_chair_targets.py` (nearest visible projected Gaussian per chair pixel); train loss 0.148 -> 0.112; 2 min 36 s on the RTX 4070.
   Train frames 1-519, held-out frames 553-649 (49 frames), 16-frame gap. New checkpoint
   `exports/IMG_6556_sddnet_ft/sddnet_chair_ft.ckpt`; `sbu.ckpt` untouched. This was the first training step in
   stage 4; the user approved it ("lets try").

| held-out | chair IoU | chair accuracy | off-chair agreement with old masks |
|---|---|---|---|
| before | 0.38 | 63% | 99.9% |
| epoch 1 | 0.55 | 74% | 97.6% |
| epoch 2 | 0.60 | 77% | 97.6% |
| epoch 3 | 0.61 | 79% | 97.5% |
| epoch 4 | 0.64 | 80% | 97.4% |

Overlays (frames 601, 649): the whole dark backrest arc and the shaded inner sides are now marked, the sunlit middle
band and arm tops stay lit; ground shadows look unchanged. Limits: one chair, one scene, held-out frames from the same
capture (shows it can learn the pattern, not that it generalises); 2.6% of off-chair pixels changed (not verified as
improvement); both old and new detectors call the red chair in the background shadow (dark/saturated objects); it
inherits the 3D labels' errors on thin slats. Note: re-scoring stage 4 against masks from this detector would be
circular, since it was trained on the same 3D labels. The real test is a new capture.

#### 4.5 Adjacent brightness-and-colour detection (laptop, 2026-10-13)
`detect_object_shadow.py --classify adjacent`. First try (19 cm) in `exports/IMG_6556_chairshadow_adj`; final (30 cm) in `exports/IMG_6556_chairshadow_adj_r04` (overlays frames 263, 317, 422, 540; chair_shadow_debug.ply; chair_shadow.png brightness vs sun angle; stats.json).
Per chair Gaussian: neighbours within about 30 cm; the bright end of their brightness is the "lit" reference; shadow if
below 0.3x that reference and the chromaticity still matches the lit neighbours (same plastic, not dark paint). A fully
dark neighbourhood (under the seat) is compared with the bright end of the same colour over the whole chair.
- Colour signature: dark spots differ in chromaticity from lit neighbours twice as much as lit spots do (0.054 vs
  0.025), shifted bluer because shade is lit only by the sky. The colour check allows for that shift.
- Neighbourhood size: at 19 cm part of the backrest arc was missed (its neighbourhood is mostly dark); 30 cm catches it.
- Result at 30 cm: 44.5% of the chair in shadow; vs the sun/3D prediction IoU 0.52, accuracy 69%; 92% identical to
  the global Otsu split (expected on a one-colour chair). Per part: under-seat and inner legs 84-92%, arm tops and seat
  front 0-7%, backrest mixed.
The method's advantage should show on multi-colour objects (wood grain, paint), where a whole-object split fails.

#### 4.4b [Addition] Retrained detector on all frames

**Retrained detector on all 650 frames (laptop, 2026-10-13).** Masks `exports/IMG_6556_shadow_masks_ft` (64 s);
originals untouched. Masks turned into per-Gaussian chair votes with the tight visibility test, then compared with the
adjacent brightness/colour labels and the sun/3D prediction:

| masks | chair shadow share | vs brightness labels (IoU / acc) | vs sun/3D prediction (IoU / acc) |
|---|---|---|---|
| old, all frames | 18% | 0.30 / 66% | 0.25 / 57% |
| new, all frames | 42% | 0.63 / 80% | 0.56 / 73% |
| old, held-out 553-649 | 15% | 0.22 / 62% | 0.21 / 55% |
| new, held-out 553-649 | 33% | 0.55 / 77% | 0.48 / 70% |

The sun/3D comparison is the less circular check: the geometry only decided the near-grazing 16% of the training
labels. Its held-out agreement more than doubled (0.21 to 0.48), so the detector learned something physically
consistent rather than only copying labels. On held-out views the new detector is more cautious (33% vs about 45%).
Masks, brightness labels and sun geometry now largely agree on the chair. One capture only.

### Code record

**Code record (2026-10-13).** The eight laptop scripts behind stages 2b-4d were committed exactly as run in `c4f25c3`
(`fit_sun_self.py`, `fit_sun_edges.py`, `shadow_visibility.py`, `shadow_overlay.py`, `detect_object_shadow.py`,
`make_chair_targets.py`, `finetune_sddnet.py`, `sddnet_masks_ckpt.py`). Not in git by design: the fine-tuned detector
weights (`exports/IMG_6556_sddnet_ft/sddnet_chair_ft.ckpt`, about 160 MB), large outputs (PLYs, masks), and the full
command/output log `~/work_session.log` on Murph. Each run folder keeps its own `stats.json` / `sun.json` with the
exact parameters.

### Stage 5: orientation-guided inpainting

**First fill test (laptop, 2026-10-13).** `scripts/fill_shadows.py` on the IMG_6556 chair. Inputs: the adjacent
brightness-and-colour labels (`exports/IMG_6556_chairshadow_adj_r04/chair_shadow.npz`; near the terminator,
|signed normal . sun| < 0.1, the sun-geometry label decides) and the stage-3 clusters and per-Gaussian grain
(`exports/IMG_6556_clusters_chair_r30agree_v3/clusters.npz`). Only colour (all SH coefficients) changes; geometry and
opacity are copied; the input splat is untouched.

Rule (from the brief: same surface and lit, not brightest nearby): for each shadowed chair Gaussian, the 8 nearest LIT
Gaussians of the SAME cluster, in an anisotropic distance where offsets across the grain cost 4x offsets along it
(4k Euclidean candidates re-ranked), weighted by 1/distance.
- `--mode copy`: the Gaussian takes the donors' weighted mean SH.
- `--mode ratio`: texture-preserving; the Gaussian keeps its own colour variation and is multiplied (linear RGB) by
  donors' mean / mean of its shadowed neighbours along the grain (same cluster); higher SH bands from the donors.
- `--donor-min 0.7`: donors must be at least 0.7x the surface's median lit brightness. Without it (first try,
  `exports/IMG_6556_fill_copy`) dim edge-of-shadow "lit" donors left the fill at 0.074 median brightness vs 0.262 lit.
- Fallback: clusters with fewer than 50 usable lit Gaussians have no lit counterpart on the same surface and borrow
  from the object's lit Gaussians nearest along the grain (breaks the same-surface rule; reported). Here clusters 11
  and 13 (under / inside the chair), 12.5% of the shadowed Gaussians.

| run | shadowed Gaussians | median brightness of shadowed (linear) before -> after | lit median | fallback |
|---|---|---|---|---|
| copy, no donor filter (`exports/IMG_6556_fill_copy`) | 11,287 of 24,445 (46.2%) | 0.026 -> 0.074 | 0.262 | 2.7% |
| copy, donor-min 0.7 (`exports/IMG_6556_fill_copy_d07`) | 11,287 | 0.026 -> 0.229 | 0.262 | 12.5% |
| ratio, donor-min 0.7 (`exports/IMG_6556_fill_ratio_d07`) | 11,287 | 0.026 -> 0.226 | 0.262 | 12.5% |

Each fill takes about 2 s; outputs per run: `splat_fill.ply` (full scene), `chair_before.ply` / `chair_after.ply`
(chair only, for SuperSplat), `stats.json`, `fill.npz`. Renders from the real cameras with
`scripts/render_splat_compare.py` (gsplat, scale 0.5) in `exports/IMG_6556_fill_compare`; committed copies:
`docs/figures/2026-10-12_chair_shadow_tests/s5_compare_frame_00263.jpg`, `..._00317.jpg`, `..._00540.jpg`,
`..._00649.jpg` (each: real photo | original splat | copy fill | ratio fill).

What the renders show: the dark arc across the upper backrest and the shaded inner legs are mostly lifted, to about
87% of sunlit brightness. Copy looks flatter and slightly too light blue; ratio keeps more of the plastic texture.
Open problems, in order: (1) blotchy / mottled fill, because the shadow labels are speckled and neighbouring
Gaussians are filled differently (needs 3D label smoothing and blending across shadow edges); (2) colour and
brightness not an exact match to the sunlit plastic; (3) thin slats (front and back faces interfere); (4) the chair's
shadow on the grass is not filled (chair only in this test). Next: fixing the mottling (approved 2026-10-13).

**Blotchy-fill fix, first round (laptop, 2026-10-13).** Same script, new options; close-ups (chair region of each
render panel) in `docs/figures/2026-10-12_chair_shadow_tests/s5_fix_*.jpg`.
- `--soft`: shadow labels cleaned in 3D first (majority vote within 0.04 units on the same surface, 2 passes, flipped
  12.9% of labels), then feathered to a 0-1 shadow amount over 0.02 units; donors only from clearly lit Gaussians;
  fill (ratio or copied colour) averaged along the grain (4 passes) and blended by the shadow amount. With
  `--donor-min 0.5` the shadowed median goes 0.028 -> 0.244 vs lit 0.235 (the first test overshot or fell short).
  Seat, arms and legs much smoother; backrest still blotchy; fallback for 14% of filled Gaussians
  (`exports/IMG_6556_fill_copy_soft`, `exports/IMG_6556_fill_ratio_soft2`; close-ups `s5_fix_soft_00263.jpg`,
  `s5_fix_soft_00317.jpg`: photo | original | soft copy | soft ratio).
- Main cause of the blotches found: the labelled set holds only the 24,445 clustered, opaque chair Gaussians.
  Another 18,262 Gaussians on the chair (unclustered or semi-transparent, opacity > 0.05, within 0.04 units of a
  labelled one) were never repainted and kept the baked shadow colour. `--include-unclustered` (with `--crop`,
  `--plane-from`) adds them, each taking the cluster and grain of its nearest labelled Gaussian: 42,707 in total.
- `--mode continuous` (the adjacent idea applied directly, no yes/no labels): per Gaussian, along the grain on the
  same surface, local level L_s = median brightness of the 16 nearest; lit level R = 80th percentile over the
  `--ref-k` nearest (300); gain = R / L_s, full when L_s/R < 0.3, none above 0.6, capped at 20x, smoothed in log space
  along the grain (3 passes); colour multiplied by the gain, so fine texture is kept and lit areas are left alone.
  Surfaces with no sunlit part (lit level < half the object's) use the object's lit level (fallback; clusters 6, 7,
  11, 13; 26% of the shadowed Gaussians).
- Result (`exports/IMG_6556_fill_continuous_all`; close-ups `s5_fix_continuous_00263.jpg`, `..._00317.jpg`:
  photo | original | clustered Gaussians only | all Gaussians on the chair): the least blotchy so far; legs, arms, seat
  and inner sides look like even plastic with texture and natural blue kept. Still open: the dark arc across the top
  of the backrest is lighter but visible; under-seat surfaces (fallback) partly too bright.
- Next: the arc. Two hypotheses to check: part of its darkness sits in the view-dependent SH bands (the fill only
  scales them), or its along-grain neighbourhood is mostly shadow so the lit level R comes out too low.

**Dark-arc fix (laptop, 2026-10-13).** Close-ups in `docs/figures/2026-10-12_chair_shadow_tests/s5_arc_*.jpg`
(`s5_arc_00263.jpg`, `..._00317.jpg`, `..._00649.jpg`: photo | original | before arc fix | after arc fix;
`s5_arc2_00263.jpg`, `..._00317.jpg`: photo | original | arc fix | arc fix with `--fallback-pct 40`).
- Cause: the second hypothesis. The arc's along-grain neighbourhood is mostly shadow, so the local lit level R came out
  far too low: the median gain on the arc was 2.9x where about 16x was needed to reach the sunlit part of the same
  slats. View-dependent colour was ruled out: the darkness is in the base colour (DC term), not the higher SH bands.
- Fix, `--surface-ref`: the surface-wide lit level (same cluster) is used as a floor for the local lit level, so a
  Gaussian deep in a long shadow still sees how bright its surface is in the sun. The arc went from 6% to 53% of the
  brightness of the lit band above it (`exports/IMG_6556_fill_continuous_sref`).
- Side effect: the fallback surfaces (no sunlit part: under the seat, inside the legs) became too light, shadowed
  median 0.28 vs lit 0.20, because the fallback level (80th percentile of the whole object) is brighter than those
  surfaces would be even in sun.
- Fix, `--fallback-pct 40`: fallback surfaces use the object's 40th percentile instead. Shadowed median now 0.168 vs
  lit 0.195 (26% of the shadowed Gaussians on fallback; clusters 6, 7, 11, 13).
- Final settings: `--mode continuous --include-unclustered --surface-ref --ref-k 300 --gain-smooth 3
  --fallback-pct 40`; output `exports/IMG_6556_fill_continuous_sref_fb40`.
- Still open: the arc's outline is faint but still visible at the top of the backrest; filled areas are a little flatter
  and paler blue than the sunlit plastic in places; the under-seat level is a guess (no sunlit reference on those
  surfaces); thin slats (front and back faces interfere); the chair's shadow on the grass is not filled yet.

## Concrete comparison run: IMG_6704 and IMG_6705 (laptop, 2026-10-21)

Goal: run the shadow-free pipeline as it stands (stages 1 to 5, same settings as IMG_6556) on two new outdoor
videos and compare the numbers with IMG_6556. Both are on concrete (the user confirmed: no grass scene, comparison only).

**Step 0, the inputs.** Both from `C:\Users\roach\Downloads`, iPhone 15 Pro Max, 3840x2160 at 30 fps, same driveway.
| Video | Length | Frames | Captured (EDT) | GPS | Object | True sun (NOAA formula, start to end) |
|---|---|---|---|---|---|---|
| IMG_6704 | 118.3 s | 3548 | 2026-10-21 12:01:39 | +32.0481, -81.0601 (±4.7 m) | standing scarecrow bear with pumpkin | elev 44.18 to 44.35, az 156.70 to 157.33 |
| IMG_6705 | 75.1 s | 2253 | 2026-10-21 12:04:17 | +32.0481, -81.0600 (±2 m) | seated bear in a cowboy hat | elev 44.40 to 44.50, az 157.55 to 157.96 |
- Time is from `com.apple.quicktime.creationdate` (capture start). `creation_time` (16:22Z / 16:24Z) is about 20 min
  later and looks like the file write time, so it was not used.
- The photographer's own shadow is in a few frames of both clips: possible stage 4 false positives.

**Repo sync.** `~/Claude_Test` was on `main`, 80 commits behind `claude/project-thread-txhi4k`. Three edited scripts
were byte-identical to the branch (stashed: `pre-merge local edits (identical to txhi4k)`). 13 untracked stage
scripts blocked the fast-forward. 12 were identical; `fit_sun.py` differed. All 13 were moved to
`~/premerge_backup_20261021/scripts/` (user approved), then fast-forwarded to 0730fc7.

**Stage 1, IMG_6704.** Started 12:35 in tmux `splat`: `MAX_WIDTH=1920 STOP_SPLIT=7000 ./scripts/make_splat_hq.sh
/mnt/c/Users/roach/Downloads/IMG_6704.MOV IMG_6704` (650 frames, 30k, 1080p as the user asked), log `logs/IMG_6704_hq.log`.
- Stage 1 result: DONE 14:27. Frames 1.5 min; COLMAP 12:37-13:37 (60 min, 49 of it mapping, CPU only; much slower than
  IMG_6615's 12 min), **650/650 posed, one model, transforms.json 650 frames**; splatfacto-big 30k 13:40-14:26 (46 min),
  GPU peak 2.5 GB / 83 C, RAM 10.7 GB, no errors. `exports/IMG_6704/splat.ply` 194 MB, 782,433 Gaussians exported
  (154,327 low-opacity dropped). Config `outputs/IMG_6704/splatfacto/2026-10-21_134029/`.
- SDDNet masks (SBU checkpoint, `sddnet_masks.py`, env `sddnet`): 650 frames in 57 s, mean shadow fraction 0.085,
  `data/IMG_6704/shadow_masks`.
- Ground plane: `fix_splat_shadows.py` with the IMG_6556 v8_o parameters (command in `logs/IMG_6704_shadowfix.cmd`),
  `exports/IMG_6704_shadowfix_o`, 87 s: n = [0.0442, -0.6949, -0.7178], offset 2.8677, inliers 47.6%, up alignment 0.997.
- Hero object: least-squares focus of the camera view rays, then the opaque column above the plane there. The bear is
  cleanly isolated: horizontal radius <= 0.66, height 0-2.13 units (55,846 opaque Gaussians). Camera height median 2.87
  units, distance to focus 4.0, so 1 unit is roughly 12-15 cm (assumes the phone about 40 cm off the ground; true
  scale unknown). Crop `--crop 0.178 0.683 1.86 1.3` (centre at mid-height, includes the base and a little ground).
- **Stage 3 with the IMG_6556 settings fails on the bear** (`cluster_surfaces.py --region-angle 30 --normal-source agree
  --agree 30`, default `--min-size 200`; `exports/IMG_6704_clusters_r30agree`, 0.8 s). Of 80,190 kept Gaussians only
  14,257 are flat enough for a normal and 29,564 more were dropped by the axis-vs-PCA test (median 42.6 deg apart). 3,833
  regions grow, but only **1 cluster >= 200, and it is the ground patch** (10,605 Gaussians, coherence 0.05). The bear
  has none, so stages 2, 4 and 5, which take the object's clusters as casters and surfaces, cannot run as on IMG_6556.
  Diagnostic only (`/tmp`, not a result):

| Run | Clusters | On the object | Object Gaussians | Object spread median (range) | Grain coherence median | Grown median |
|---|---|---|---|---|---|---|
| IMG_6556 chair, min 200 | 14 | 14 | 30,070 | 14.4 (12.1-20.5) | 0.73 | 480 |
| IMG_6704 bear, min 200 | 1 | 0 | 0 | - | - | - |
| IMG_6704 bear, min 50 | 27 | 26 | 22,396 | 18.2 (15.8-19.9) | 0.23 | 69 |
| IMG_6704 bear, min 20 | 129 | 123 | 43,296 | 17.0 (10.7-20.2) | 0.28 | 30 |
  Reading: the sculpted, curved figurine breaks into many small patches with weak grain, unlike the chair's flat,
  slatted plastic. Waiting for the user's choice before going on.
- Renders (`render_splat_compare.py`, scale 0.4): `exports/IMG_6704_compare/compare_frame_{00040,00200,00360,00520}.jpg`
  (photo | splat | clusters min 200 | clusters min 50 diagnostic). The splat matches the photos closely. Min 200: only
  the ground patch. Min 50: hat, head and torso split into about 26 patches; legs and boots get none. Copy of the min-50
  debug PLY in the same folder.
- Scale: IMG_6556 is about 1.33 units/m, IMG_6704 roughly 7-8 units/m (assumed phone height). Settings that were
  physical sizes on IMG_6556 (detection `--adj-radius 0.4` = 30 cm, crop, `--fit-shift` scan range) are converted with
  this ratio; sun angles do not depend on it.
- Live terminal for the user at school (user asked): ttyd read-only on tmux `splat` (`-i 127.0.0.1`, password
  `~/.ttyd_pass`) plus a cloudflared quick tunnel (log `~/cloudflared_ttyd.log`); not blocked this time. Stop with
  `pkill -f "[c]loudflared tunnel"; pkill -f "[t]tyd -p"`.
- **User chose to go on with the bear** (asked for the full bear results in the process document), so option A:
  stage 3 at `--min-size 50`, everything else as IMG_6556. Official run `exports/IMG_6704_clusters_r30agree_ms50`:
  27 clusters (26 on the bear, 1 ground patch). Coverage of opaque bear Gaussians by height: 0.15-0.6 (legs) 16%,
  0.6-1.2 29%, 1.2-1.7 51%, 1.7-2.2 (hat) 37%.
- Physical-size settings from IMG_6556 scaled by about 5.6 (cell / eps / raster 0.02 -> 0.1, occluder-min 0.05 -> 0.25,
  z-tol 0.03 -> 0.15, ground-tol 0.1, fit-shift [-0.08, 0.03] -> [-0.45, 0.15] step 0.05; region and score radius 3.5 to
  cover the shadow). Runner `scripts/runs/run_stage2_6704.sh`, log `logs/IMG_6704_stage2.log`.
- **Stage 2a** (`fit_sun.py`, area IoU, `exports/IMG_6704_sun`, 50 s): 325 masked frames, 251,871 receivers, 4.9%
  observed shadow. Best sun **elevation 15.0 deg (error -29.3)**, IoU 0.53; hold-out halves 44.5 and 46.5 deg (13.4 deg
  apart). The full fit smears the predicted shadow out to the edge of the region: with the legs mostly missing from the
  casters, a long shadow from the head scores as well as the right one. By hand: observed shadow about 2.4 units long
  behind a 2.13-unit bear gives about 42 deg.
- **Stage 2b** (`fit_sun_edges.py --fit-shift`, boundary score, `exports/IMG_6704_sun_edges_fitshift`, 196 s): 19,110
  casters. Elevation across the offset scan 40.75-49.25 deg; lowest chamfer at the scan edge (+0.15 = receivers down
  about 2 cm in this scene; the script's "cm" labels use IMG_6556's scale), still falling there. Final **elevation 49.25 deg
  vs true 44.27, error +4.98 deg**, uncertainty 0.53 deg, halves 48.0 / 49.5 (10.6 deg apart, mostly azimuth), chamfer
  0.068, IoU 0.56. Chair for comparison: -0.43 deg. Predicted outline wider than observed on one side.
  Candidate causes (untested): incomplete casters (legs 16% clustered); offset scan range too narrow; the scale guess
  behind the converted settings.
- **Bug fix (2026-10-21):** `shadow_visibility.py --revote` crashed (`ValueError: too many values to unpack`): the
  committed `fit_sun.collect_votes` returns (shadow, seen, toward) since the self-shadow work, while IMG_6556 ran against
  the older untracked copy that returned two values. One-line fix: keep the first two (`[:2]`). Uncommitted.
- **Grain check** (`check_grain.py`, 6 frames): interior pixels n = 10,334, grain median **32.9 deg** (rot90 53.1, random
  44.3), 34% within 20 deg; edges n = 22,731, 28.6 deg (rot90 58.6, random 44.8). Chair: 9.1 / 5.5 deg.
- **Stage 4a** (`shadow_visibility.py`, sun from stage 2b, `exports/IMG_6704_shadow4`): 19,110 object Gaussians; attached
  26.3%, cast on object 57.8%, predicted shadow 69.9%; ground vs masks IoU 0.465 (chair 0.696). SDDNet marks only 0.4%
  of the bear as shadow.
- **Stage 4b** (`detect_object_shadow.py --classify adjacent`, `--adj-radius 2.24` = 30 cm scaled, `exports/IMG_6704_objshadow_adj`):
  **29.5% of the bear in shadow** (chair 44.5%); vs geometry IoU 0.35, precision 0.88, recall 0.37 (chair IoU 0.52).
  The sun + sky model fit is not physical on the bear: L_sky 0.034, L_sun 0.017, ratio 1.5 (chair 12.7). The bear is
  painted in many colours, so brightness is mostly paint, not light.
- **Stage 5** (`fill_shadows.py`, final IMG_6556 settings, distances scaled, `exports/IMG_6704_fill_continuous_sref_fb40`, 17 s):
  80,978 object Gaussians (incl. 61,868 unclustered), 6,267 shadowed (7.7%); median brightness shadowed **0.022 -> 0.118,
  lit 0.075 (overshoot about 60%)** (chair 0.026 -> 0.168, lit 0.195); 66.6% of the object raised > 1.5x; fallback for 19.9%
  of shadowed (clusters 0, 7, 13, 18, 20, 24, 26). Renders `exports/IMG_6704_fill_compare/` (photo | original | fill):
  shaded hat, jacket and trousers turn purple / over-saturated; ground shadow untouched (not implemented).
  Runner `scripts/runs/run_stage345_6704.sh`, logs `logs/IMG_6704_stage345.log` (crash) and `logs/IMG_6704_stage45.log`.
- **Process document:** new chapter 8 "Second test: a painted bear figurine on concrete (IMG_6704)" (8.1-8.8, figures 24-31,
  side-by-side table), old chapters 8/9 renumbered 9/10, title page and summary updated. Written as a new file
  `Shadow_Free_Splat_Process_v2.docx` / `.pdf` (29 pages) in the Windows repo folder; the original is untouched. Builder
  script in the session scratchpad (python-docx; TOC refreshed and PDF exported with Word).
- **User gave the bear's height: about 15-20 cm.** At 17.5 cm, 2.13 units -> about 12.2 units/m, so IMG_6556 physical
  settings scale by about 9.1 (not 5.6). User asked to rerun the sun test only (one command at a time this time).
  Exception: `--ground-tol` kept at 0.1 (measured concrete layer); 0.045 x 9.1 = 0.41 would include the bear's feet.
- Sun rerun, scale 9.1 (cell / eps / raster 0.18, occluder-min 0.46, z-tol 0.27, fit-shift -0.73..0.27 step 0.09):
  - 2a area IoU (`exports/IMG_6704_sun_s91`, 50 s): elevation **14.75 deg** (error -29.5), IoU 0.44, halves 30.6 deg apart.
  - 2b boundary + offset (`exports/IMG_6704_sun_edges_fitshift_s91`, 195 s): elevation across the scan 44.75-55.25;
    picked the top of the scan again (+0.26 = receivers down about 2.1 cm); final **55.25 deg, error +10.98**,
    uncertainty 1.4, halves 48.5 / 46.75 (23.8 deg apart), chamfer 0.125, IoU 0.38. Worse than at scale 5.6 (49.25, +4.98).
  - Reading: the cell size should follow the object's size, not metres. On a 17 cm bear a 1.5 cm cell leaves the whole
    shadow outline only 80 raster cells (40 x 40 grid), against 151 at 0.1. Both runs choose the edge of the offset
    scan, so the offset fit is not settled either.
- **Ground shadow, bear 1** (user: "forgot the ground test"). The `fix_splat_shadows.py` run with the IMG_6556 "O"
  parameters from earlier today is the ground fix: 9,330 ground Gaussians changed (8,360 at full weight), gain capped at
  12 in red (linear RGB 12.0 / 9.6 / 6.0, luminance 9.85), lit reference luminance 0.49. Renders
  `exports/IMG_6704_ground_compare/` (frames 40, 200, 360, 520 and 92, 289, 405, 452, 631: frames picked by largest
  shadow mask not touching the border).
  Measurement (scratch `ratio.py`): median linear luminance inside the frame's SDDNet mask (largest component in the lower
  half, eroded 4 px) / ring 8-30 px outside it, grey pixels only (saturation < 0.15), 5 frames: **photo 0.07, original
  0.16, fixed 0.58** (range 0.24-0.84). First version used the whole mask and was wrong: the mask also covers shade
  under a parked car, which the fix (near the bear only) does not touch. 3D check on f_dc (ground within 3.5 of the
  foot, `shadow_frac` > 0.5): 0.026 -> 0.239 vs lit 0.492, i.e. 5% -> 49%. Looks: shadow gone but a flat, slightly
  blue-grey patch with the outline still visible; the concrete shadow (5-7% of sunlit) is too dark for a 12x cap.
- **Document restructured on request** ("don't make comparisons yet, comparisons at the end"): chapter 8 is now bear 1 only
  (no chair columns, side-by-side table removed, new 8.7 Ground shadow with figures 33-35, 8.8 What this test showed).
  Comparisons go in a closing chapter after all tests. v2 docx/pdf rebuilt (29 pages, figures up to 35); link page
  https://claude.ai/artifact/NzkYAg2SsFoSzTYzLGvxS1 updated to match (version 2).
- **Live viewer** (user asked): `ns-viewer` for IMG_6704 started in tmux window `splat:viewer` on localhost:7007. The
  public cloudflared tunnel to it was **blocked by auto mode** this time ("External Ingress Tunnel"; the ttyd tunnel at
  14:3x was allowed). Instead: a private claude.ai page with a small WebGL splat renderer
  (https://claude.ai/artifact/7eTcHiWU3R4bcUeqQSj12d): original / bear fill / ground fix, each the 300,000 most visible
  Gaussians within 5 units of the bear (`ply_to_splat.py --n 300000 --radius 5`, same selection in all three since only
  colour differs), packed losslessly into RGB PNGs (2048 x 1563, 8.9 MB each) because the host does not serve raw
  binaries. Base colour only (no view-dependent SH).
- **Concrete ground-fix tuning (user: "lets try fixing the shadows", "the concrete is bad"; viewer not wanted, ns-viewer
  stopped).** Diagnosis of O on concrete: `--colour lum` keeps the shadow's blue cast; gain from the photo edge band (about
  10 in luminance) is too low for a 5-8% shadow; `--ground-tol 0.045` is thinner than the concrete layer (about ±0.1).
  - **B** `exports/IMG_6704_shadowfix_g25rgb`: as O but `--colour rgb --max-gain 25 --ground-tol 0.1`. Gain measured R 12.9,
    G 9.6, B 6.0 (luminance 10.0); 11,064 brightened (9,911 full), 83 s. Blue cast gone.
  - **C** `exports/IMG_6704_shadowfix_c`: as B plus `--gain 20 15 9.4` (B's channel ratios x1.55, luminance 15.7) and
    `--lo 0.10 --hi 0.35`; target mode keeps the ceiling at the lit ground median (0.484). 11,920 brightened (9,977 full), 84 s.
  - 8-angle renders `exports/IMG_6704_ground_angles` (O, B) and `exports/IMG_6704_ground_angles_c` (O, B, C), frames chosen
    one per 45-deg camera azimuth bin (far cameras only; 45-135 deg has only close-ups, so 452 and 607 fill in; 160 is a
    close-up). Measured on 7 views (scratch `ratio2.py`; shadow core vs ring of sunlit concrete, grey pixels; colour
    error = rg-chromaticity distance x1000): median brightness photo 0.08, original 0.28, O 0.58, B 0.58, C 0.66; colour
    error original 13, O 23, B 16, C 19. Per-view values are noisy where the 2D mask and the rendered shadow do not line up.
  - Visually: B and C remove the blue; the patch stays flat and pale with a thin rim. A per-Gaussian gain cannot restore
    concrete texture that was never recorded in sunlight; a texture-carrying fill (copy from nearby sunlit concrete, as
    stage 5 does on objects) is the likely next step.
- **Texture fill T (2026-10-26, user: "try texture fill and see").** New `fix_splat_shadows.py --fill texture` (clone stamp,
  ground only; uncommitted): one in-plane shift (24 directions x 0.25..6 units) moves the shadow patch onto lit ground;
  each shadowed splat takes DC + SH from the lit ground splat within 0.1 of its shifted spot (sources: shadow_frac < 0.1,
  opaque, lit-ground chromaticity within the colour gate), blended by the shadow weight; misses try the next shortest
  shift, then gain. Same selection as C. `exports/IMG_6704_shadowfix_tex`, 87 s: 183,715 lit sources, 11,920 targets,
  shift [-1.08, 0.63] (1.25 units, about 10 cm) covers 92.5% (best 94.9% at length 5), filled 11,757, gain fallback 163.
  Renders `exports/IMG_6704_ground_angles_tex` (original | C | T, same 8 views) and `crops_photo_orig_C_T.jpg`.
  Visually: concrete grain and speckle now continue through the patch (C was flat pale blue-grey); a faint outline and a
  slightly smoother, lighter patch remain. A first run crashed on a stats indexing bug (fixed).
- **Outline blend U (user: "lets try matching the outline so that it blends").** New `--tex-feather F` (fill ground splats
  within F outside the patch that were seen in shadow at least once and are darker than their copy, ramped by distance;
  edge splats filled by darkness vs the copy, `clip((1 - own/copy) / 0.6)`, instead of vote weight) and `--tex-match K`
  (copy x mean colour of the K nearest lit ground splats around the target / around the source, clipped 0.5-2).
  `exports/IMG_6704_shadowfix_tex_u` = T + `--tex-feather 0.2 --tex-match 200`, 90 s: band 44,844, 22,649 changed; match
  ratio median 1.0, p10/p90 0.86/1.17. Renders `exports/IMG_6704_ground_angles_tex_u` (original | T | U).
  Measured on the renders (linear luminance vs a sunlit ring 6-20 px outside the photo mask; 5 views, frame 104 excluded
  because the mask does not line up): edge/ring T 0.87 -> U 0.96; centre/ring 0.90 -> 0.96; fine-detail contrast inside
  the patch vs ring T 0.73, U 0.70. Outline mostly gone; the patch stays softer than real concrete (faint smudge, worst
  near the feet). Likely cause: the shadowed ground has fewer, larger Gaussians, so copying colour alone cannot add detail.
- **More detail V (user: "i need a lil more detail").** New `--tex-detail G` (with `--tex-match`): copy = mean of the
  source K lit neighbours + G x (copy - that mean), then the match ratio. `exports/IMG_6704_shadowfix_tex_v` = U +
  `--tex-detail 1.4`, 93 s. Renders `exports/IMG_6704_ground_angles_tex_v` (original | U | V). Render measure (scratch
  `measure_renders.py`, 5 views, medians of centre / edge / detail vs ring): original 0.06 / 0.56 / 0.37, C 0.63 / 0.85 /
  0.44, T 0.90 / 0.87 / 0.73, U 0.96 / 0.96 / 0.70, V 0.91 / 0.95 / 0.93. V restores most fine contrast but has a slightly
  darker, blotchy area near the feet.
- **Process document v3** (user: "do add this on the document too"): `Shadow_Free_Splat_Process_v3.docx` / `.pdf` (31
  pages) in the Windows repo folder, built from v2 (v2 untouched) by scratch `build_doc_v3.py`: section 8.7 gains B/C, T,
  U, V, the table above and figures 36-38; 8.8 bullet, chapter summary box, report summary and test dates updated. TOC
  refreshed and PDF exported with Word (Word dropped its COM connection after the export; files complete).
- **Splat clone W (user: "still a lil smudgy. doesnt feel like concrete noise").** New `scripts/clone_ground_splats.py`
  (uncommitted): ground-plane grid of 0.05-unit cells; shadow cells = mean shadow_frac of opaque ground splats >= 0.3,
  closed 2 cells, holes filled, components the texture fill touched, minus the object footprint (opaque splats > 0.15
  above the plane, grown 1 cell); a cell is filled only if the cell at the texture shift (rounded to whole cells, -22 / +12)
  is lit ground. Deletes the ground splats in filled cells and adds copies of all ground splats from the source cells,
  moved back by the shift (size, shape, rotation, opacity, SH kept), colour x mean of 200 nearest lit ground splats
  around the new spot / around the source (median 0.97). Base = U. `exports/IMG_6704_clone_w/splat_clone.ply`, 2 s:
  732 of 928 patch cells filled (196 without a clean source keep U), 11,213 removed, 11,875 added (782,433 -> 783,095).
  Confirms the cause of the smudge: removed splats median max scale 0.0205 vs added 0.0146.
  Renders `exports/IMG_6704_ground_angles_clone_w` (original | U | W); measure (5 views, medians): centre 0.97, edge 0.98,
  detail 1.01 (U 0.70, V 0.93). Looks like sunlit concrete; new clone-stamp artefact: part of the real crack is copied
  into the patch (frame 405).
- **Crack fix (user: "fix the repeated crack").** `clone_ground_splats.py` now picks its own whole-cell shifts (24
  directions x 0.25..5 units; first = shortest covering >= 97% of the best, then greedy up to `--max-shifts`) and never
  copies from crack cells (`--crack R`: sunlit cells darker than R x the Gaussian-weighted (sigma 4 cells) mean of the
  sunlit cells around them, kept only in 8-connected runs >= `--crack-min-cells`, grown by `--crack-margin`); `--map`
  writes a top-down figure. W above was made with the first version (texture shift only, no crack test).
  - X `exports/IMG_6704_clone_x` (`--crack 0.75`, no run filter, 6 shifts): 13,210 "crack" cells, i.e. ordinary dark
    specks too; sources biased to plain bright concrete, detail 0.76-0.82 in 4 of 5 views. Rejected.
  - Y `exports/IMG_6704_clone_y` (`--crack 0.6 --crack-min-cells 8 --max-shifts 6 --map`, 6 s): 1,524 crack cells; shift
    [-24, -6] cells (1.24 units) fills 732 cells, 5 more shifts 25; 757 of 928 filled (171 near the feet keep U); 11,436
    removed, 19,311 added (25.5 per cell vs 15.1 removed; added median scale 0.0120); 790,308 splats. Renders
    `exports/IMG_6704_ground_angles_clone_y` (original | W | Y); medians centre 0.97, edge 0.98, detail 0.87 (W 1.01).
    No copied crack in any view; a mild darker stain from the source area in the middle; lighter rim at the feet (frame 452).
  - Map bug fixed: the texture fill also touched small shadow spots elsewhere on the driveway (cloned too, harmless),
    which made the first map cover the whole driveway; it now crops to the largest patch.
- **Research paper (user: "the paper should be research oriented with each step and the mathematics behind explained.
  and add photos").** `Shadow_Free_Splat_Research_Paper.docx` / `.pdf` (28 pages) in the Windows repo folder; v3 untouched.
  Abstract, introduction with contributions, background and notation, method with 33 numbered native Word equations
  (every stage, written from the scripts: votes, RANSAC plane, PCA/agree normals, region growing, structure-tensor grain,
  sun parameterisation, orthographic shadow map, IoU, chamfer, receiver offset, signed normals, sun+sky model, the
  user's adjacent rule (credited), BCE fine-tuning, anisotropic donors, copy/ratio/continuous fills, gain, colour clone
  stamp, blend/match/detail, geometry clone stamp with crack test and greedy shifts, render metrics), the two experiments
  reported separately (cross-scene comparison deferred), discussion, conclusion, 17 references, parameter appendix.
  44 figures (all 38 from v3, plus pipeline diagram, U|W and W|Y crops, splat-size histogram, clone map, metrics chart)
  and 7 tables. Built by scratch `paper/build_paper.py` (python-docx; equations as UnicodeMath) + `finish_paper.ps1`
  (Word COM: build-up, TOC, PDF). Word gotcha: a bare "max"/"min" in a sub/superscript is read as a function and
  swallowed the next paragraph's start; labels are now quoted as upright text, and a check confirms all 207 text
  paragraphs survive. `clone_ground_splats.py` now also writes clone_index.npz (removed / copied-from indices).

## Bear 2, IMG_6705: one command from video to shadow-free splat (laptop, 2026-10-27)

Goal (user): run the whole pipeline on bear 2 as ONE command, to see whether it works on a new scene. Sun from the
video metadata. Reported on its own (no comparisons with other scenes yet). Ground shadow removed too.

- Terminal: new tmux window `splat:bear2` (session `splat`), Windows Terminal window "Bear 2 (tmux splat:bear2)";
  the read-only ttyd + cloudflared view from 2026-10-21 was still running, so no new tunnel.
- Repo: `~/Claude_Test` `main` = `claude/project-thread-txhi4k` (fetched, nothing to merge); bear 1 uncommitted scripts kept.
- Metadata: GPS +32.0481 -81.0600 (2 m), capture start 2026-10-21 12:04:17 EDT, 75.1 s. **No compass heading** in the
  file (all tags checked), so only the sun's elevation can be used in the scene; its compass azimuth cannot be placed
  without knowing north in the splat. User: the bear's back faces roughly north (exact angle unknown); used only as a check.
- Plan: elevation fixed from metadata, azimuth + receiver offset fitted from the shadow outline; our own free fits
  (area IoU and boundary) logged against the metadata elevation.
- Bear 2 height (user, iPhone Measure app): **16 +- 0.5 cm**. Lengths = bear 1 final values x K, K = bear 2 units/m / 12.18
  (bear 1: 2.13 units / 17.5 cm).
- New / changed code (uncommitted):
  - `scripts/sun_from_metadata.py`: NOAA solar position from the QuickTime GPS + creationdate (with refraction), start /
    mid / end. Check: IMG_6704 44.20 -> 44.36 / 156.70 -> 157.34, IMG_6705 44.42 -> 44.52 / 157.55 -> 157.96 (earlier
    values 44.18-44.35 / 156.70-157.33 and 44.40-44.50 / 157.55-157.96; difference = refraction, about 0.02 deg).
  - `scripts/find_object.py`: hero object without hand work (least-squares focus of the camera axes, opaque column above
    the plane, 8-connected footprint component nearest the focus), crop and scale. Check on bear 1: height 2.1317 units,
    radius 0.654, crop `0.1779 0.6833 1.8598 1.3002`, 55,892 Gaussians (hand values 2.13, 0.66, `0.178 0.683 1.86 1.3`, 55,846).
  - `fit_sun_edges.py --fix-elevation E`: elevation held; grid, offset scan and Nelder-Mead over azimuth only. Code test
    on bear 1 data (E = 44.27): runs, 17 s.
  - Stage 3 flags confirmed: rerunning bear 1 with `--region-angle 30 --normal-source agree --agree 30 --min-size 50`
    gives labels identical to `exports/IMG_6704_clusters_r30agree_ms50`.
  - Runner `scripts/runs/run_full_6705.sh` (log `logs/IMG_6705_full.log`): 0 metadata sun; 1 make_splat_hq (650 frames,
    1920 wide, splatfacto-big 30k, STOP_SPLIT 7000); 1b SDDNet masks; 2a ground plane (bear 1 "O"); 2b find_object -> crop, K;
    2c stage 3; 2d grain check; 3a fit_sun votes + own IoU fit; 3b own boundary fit (free); 3c boundary fit at the metadata
    elevation (used); 4a visibility; 4b adjacent detection; 5a ground texture fill + outline blend ("U"); 5b object
    continuous fill on top of it; 5c ground splat clone stamp with crack check ("Y") -> `exports/IMG_6705_final/splat_shadowfree.ply`;
    6 renders photo | original | shadow-free.
- **Run started 18:45:51 EDT (user pressed Enter)** in `splat:bear2`. Stage 0: elevation 44.42 -> 44.52 (mid 44.47), azimuth 157.55 -> 157.96 (mid 157.75).
- Timing (user: "lets time this run too"; laptop on max power this time, Windows plan "Acer"): step times from the runner timestamps; machine load every 30 s by new `scripts/sample_load.py` -> `logs/IMG_6705_load.csv` (CPU %, cores > 50 %, top process, RAM, GPU util / memory / temperature / power). Started 18:55. Frames 18:45:52-18:47:00 (68 s); COLMAP features done 18:47:41 (GPU).
- Matching 18:47:41-18:57:30 (9 min 49 s, GPU 83-92 %, 87 C, 111 W); mapping from 18:57:30, CPU only, colmap about 7-8 cores. User asked why mapping cannot use more cores (incremental SfM is serial) and why global mapping is not used (never switched on; `colmap4` env has `global_mapper`). **Pending, user: "dont queue it but remind me when we are done"**: after this run, offer a GLOMAP vs incremental timing test on the same bear 2 frames and matches.
- COLMAP: mapping 18:57:30-19:38:07 (40.6 min, CPU, colmap mean about 4.9 cores, peak 8.3), intrinsics refine to 19:42:43 (4.6 min); **650/650 posed**; ns-process-data done 19:42:50 (COLMAP total 55.8 min). Pose fix from 19:42:52.
- Pose fix 19:42:52-19:44:37; training splatfacto-big 30k 19:44:37-20:27:49 (43.2 min; late steps about 95 ms, GPU 93 %, 71 C, 81 W, 2.4 GB); export 20:27:49-20:28:16: `exports/IMG_6705/splat.ply` 178 MB, **719,172 Gaussians** (127,180 low-opacity dropped of 846,352). Stage 1 total 18:45:52-20:28:16 = 1 h 42 min. Masks from 20:28:16.
- Masks 20:28:16-20:29:09 (650 in 49 s, mean shadow fraction 0.090). Ground plane (O) 20:29:09-20:30:32: n = [0.0006, -0.6958, -0.7183], offset 2.8535, up alignment 0.997. **find_object**: one component, 36,106 opaque Gaussians, **height 1.494 units**, radius 0.799, crop `0.1182 1.2286 1.7426 1.1375`; 16 cm -> **9.34 units/m, K = 0.767**; camera height median 2.65 units = 28 cm, distance to focus 4.10 = 44 cm. Stage 3 (0.7 s): 56,109 kept, 24 clusters >= 50, 25,439 attached. Grain check (interior): median 33.0 deg (rot90 52.8, random 43.6), 33.5% within 20 deg.
- **Sun.** 3a fit_sun votes + area IoU (45 s): 325 masked frames, 229,813 receivers, 3.2% in shadow; elevation 52.00, scene azimuth 285.00, IoU 0.466, halves 36.25 deg apart. 3b boundary, free (about 3 min): 15,182 casters; elevation **51.50 (+7.03 vs metadata 44.47)**, azimuth 289.75, chamfer 0.111, IoU 0.448, halves 49.0 / 21.5 (37.0 deg apart); receiver shift +0.115 = top of the scan. 3c boundary at the metadata elevation (used): azimuth **291.07** (free fit 1.32 deg away), chamfer 0.120, IoU 0.464, uncertainty 0.54 deg, halves 13.0 deg apart; receiver shift -0.345 = bottom of the scan (receivers 3.7 cm UP, not physical on concrete), so the offset fit is not settled (it only steers the azimuth fit; downstream stages use the direction only). True north at scene azimuth 88.83 (from 291.07 + compass 157.75).
- Stage 4a visibility (8 s): sun [0.6663, -0.6714, -0.3243]; 15,182 object / 121,911 ground Gaussians; attached 13.9%
  (trusted 17.5%); cast on object 39.7%; object in shadow 44.2%; ground cast 11.1%. Ground vs masks: observed 3.95%,
  predicted 11.5%, precision 0.343, recall 1.0, IoU 0.343: the predicted ground shadow is about 3x too big (the sun
  figure shows the same: predicted outline wider than observed on all sides). Untested causes: shadow-map cell / eps
  (0.077 / 0.077 units after scaling) coarse for a 1.49-unit object; receiver offset at the scan edge.
- Stage 4b adjacent detection (37 s): 30.3% of the bear in shadow; vs geometry IoU 0.466 (precision 0.781, recall 0.536).
  SDDNet marks 0% of the bear. Sun + sky fit: sky 0.055, sun 0.087, ratio 2.6. **Scaling problem:** `--adj-radius`
  2.24 x 0.767 = 1.72 units (18 cm) is larger than the bear (radius 0.80), so every Gaussian has all 15,182 as
  neighbours and the "adjacent" rule compares with the whole bear, not the surroundings.
- Stage 5a ground fill U (88 s): 13,659 targets; texture shift [0, 1.342] (1.34 units, 14 cm) covers 94.7% (best 97.0%
  at length 4.4); filled 13,431, gain fallback 228; feather band 42,904, 19,377 changed; match ratio median 0.98
  (p10/p90 0.83/1.16); lit ground luminance 0.475.
- Stage 5b object fill (11 s), on top of U: 55,461 object Gaussians (incl. unclustered), 4,699 shadowed (8.5%); median
  linear brightness shadowed 0.038 -> 0.115, lit 0.136; **52.8% of the object raised > 1.5x**; fallback for 46.3% of the
  shadowed (clusters 17, 18, 21, 22 and others "no sunlit part").
- Stage 5c clone stamp Y (3 s): cell 0.038; 1,606 shadow cells, patch 938, **811 filled**, 127 unfilled (keep U); 1,242
  crack cells near the patch excluded as sources; shifts [-24, 6] cells (0.95 units) 792 cells, then 17, 1, 1; removed
  7,080, added 14,437 (8.7 vs 17.8 per cell; median scale 0.0198 -> 0.0117); 719,172 -> 726,529 Gaussians.
  Final: `exports/IMG_6705_final/splat_shadowfree.ply`.
- Stage 6 renders `exports/IMG_6705_final_compare/compare_frame_{00082,00244,00407,00569}.jpg` (photo | original | shadow-free).
  Visually: **ground shadow gone** in all views, concrete grain and cracks continue through the patch; a soft grey
  smudge remains right at the base (the 127 unfilled cells next to the footprint). **Bear fill failed:** the whole bear
  turns orange / amber and glassy, over-saturated, hat blotchy; consistent with 52.8% raised and the adjacent radius
  covering the whole bear.
- Shadow-behind-back check (by eye, frames 244 and 569): the shadow falls behind the bear and a little to one side, as
  expected for a south-facing bear with the sun at SSE. Consistent with the user's "back faces N".
- **Run total 18:45:51-20:37:38 = 1 h 51 min 47 s, one command, no hand values, no crash.** Step times: frames 68 s;
  COLMAP features 41 s (copy + extract), matching 9 min 49 s, mapping 40 min 37 s, refine 4 min 36 s, pose fix 1 min 45 s;
  training 43 min 12 s; export 27 s; masks 53 s; plane 83 s; find_object 1 s; stage 3 3 s; grain 3 s; sun 3a 62 s,
  3b 3 min 11 s, 3c 15 s; 4a 8 s; 4b 37 s; 5a 88 s; 5b 11 s; 5c 4 s; renders 3 s.
  Load (sampler, 30 s): matching GPU 87%, 112 W, max 87 C, colmap 2.8 cores; mapping GPU 0%, colmap mean 5.0 cores (max
  8.3), RAM max 7.8 GB; refine 6.7 cores (max 12.3); training GPU 91%, mean 78 W, max 72 C, 2.9 cores, RAM max 11.2 GB;
  stages 1b-6 mean 18 cores (max 32, all cores), GPU 3%.
- Open: object-fill scaling (adjacent radius vs object size); ground over-prediction in 4a; receiver offset at the scan
  edge; smudge at the base. Reminder owed to the user: GLOMAP vs incremental timing test (not queued).
- After the run (user, 2026-10-27 evening): asked why the bear fill failed (explained: adjacent radius > bear, paint read as shade); user restated the goal of sun tracking = light response of surface blobs in 3D; proposed geometry-driven fill (sun-blocked + facing + sun/sky ratio from the concrete). User: "there has to be a better way of splatting so we know what the surface is" -> options 2DGS (gsplat 1.4.0 has the 2DGS rasteriser), DN-Splatter, surfels/PGSR/GOF; quality concern -> two-splat idea (normal splat for looks, 2DGS only as geometry reference) with a held-out quality check. **User will decide tomorrow**; meanwhile will try filming with better angles (filming tips given). Still pending: commit of the new scripts, GLOMAP timing test (reminder), geometry-driven fill.
- 2026-10-28: user chose the **surface test (2DGS)** next; asked how to manage the long conversation. Wrote `docs/status.md` (one-page hand-off: state, failures, next step, pending decisions, house rules) for a fresh session.
