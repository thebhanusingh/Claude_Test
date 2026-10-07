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

### GS-IR runs

| Date | Scene | Settings | Result |
|---|---|---|---|
| 2026-10-07 15:33 (laptop time) | `IMG_6556` (`data_gaussianshader`, undistorted) | Runner `scripts/runs/run_gsir_6556.sh` (tmux `gsir`, log `~/gsir_train.log`, out `outputs_gsir/IMG_6556`). Stage 1: 30k, `-r 2` (942x530), `--data_device cpu`, `--eval` (every 8th frame held out), default densify; OOM fallback `--densify_grad_threshold 0.0004 --densify_until_iter 7000`. Baking: `--bound 6.0 --valid 6.0 --occlu_res 256 --occlusion 0.4 --cubemap_res 256` (all 650 cameras within 4.63 units of origin, ~92% of 445k SfM points within 6; cell ~4.7 cm); OOM fallback `occlu_res 192`. Stage 2: to 40k, `--indirect --gamma`, metallic off, `brdf_tv 1.0`, `env_tv 0.01`. Export: `exports/IMG_6556_gsir/splat_albedo.ply`, f_dc = (albedo - 0.5)/0.28209 (sRGB albedo due to `--gamma`), f_rest = 0. | Attempt 1: crashed at launch (tensorboard, item 4). Attempt 2: stopped at 4,950 (too slow, item 5). |
| 2026-10-07 16:06 | `IMG_6556` | Attempt 3: as above but stage 1 `--densify_grad_threshold 0.0004 --densify_until_iter 7000`. | Stage 1 done 16:06-16:54 (48 min, ~10-11 it/s, no errors). Held-out (82 frames): 7k PSNR 24.29 / SSIM 0.838 / L1 0.0408; 30k PSNR 26.29 / SSIM 0.897 / L1 0.0317. Train: 7k 24.75 / 0.832; 30k 27.31 / 0.905. GPU peak ~3.9 GB near 7k, then steady 3.5 GB; 78-82 °C; WSL RAM steady 7.1 GB. `chkpnt30000.pth` 1.38 GB. (Not directly comparable: nerfstudio held-out median on IMG_6556 was 21.2 dB at full res, different split.) Baking started 16:54 (`occlu_res 256`, bound 6.0): 420,159 occupied cells at ~38 cells/s (6,155 after 2.5 min) = ~3 h projected; GPU 2.9 GB, 71 °C, RAM 2.8 GB. Baking, not training, is the slow step on this laptop at 256. Stopped at 7% (VRAM full, item 7); re-baked at 192 from 17:08 (240,894 cells, ~44 cells/s, ~1.5 h). |
