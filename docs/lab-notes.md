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
