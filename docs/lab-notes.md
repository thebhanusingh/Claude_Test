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
| 2026-09-28 | Cloned GaussianShader (commit de77861) to `third_party/GaussianShader`, created a **modern** conda env `gaussian_shader` | The repo's `environment.yml` is a 2022 freeze (Python 3.7, torch 1.10+cu111, local-only pip packages) and can't be created as is. Built instead: Python 3.10, torch 2.4.1+cu124, conda `cuda-toolkit=12.4`, conda gcc/g++ 12 (system gcc 15 is too new for CUDA 12.4), plus plyfile/tqdm/opencv/imageio/scipy/matplotlib/scikit-image/tensorboard/open3d, numpy<2. Log: `setup_gaussianshader.log`. |

## Scripts added (repo)

| File | Purpose |
|---|---|
| `scripts/select_sharp_frames.py` | Scores every video frame by Laplacian variance and keeps the sharpest frame per window (default 650). Replaces ffmpeg's evenly spaced extraction. |
| `scripts/fix_colmap_model.py` | Picks the COLMAP sub-model with the most images, bundle-adjusts it (incl. principal point), writes `transforms.json` + `sparse_pc.ply` in original COLMAP world coordinates. Original nerfstudio outputs kept as `*_nsprocess.*`. |
| `scripts/make_splat_hq.sh` | Full HQ pipeline: sharp frames -> `ns-process-data images` -> `fix_colmap_model.py` -> `splatfacto-big` (falls back to `splatfacto` on failure) -> export. Skips steps 1-3 if the scene already has poses. |
| `run_600f.sh`, `run_600f_hq.sh`, `run_two_videos.sh` (repo root) | One-off runners for specific scenes. |
| `watch_600f.sh` (repo root) | Watcher: status line every 15 s to `<log>_status.log`; stdout summary every 5 min, on stage change, errors, finish, or process disappearing. Args: `PID LOG [DONE_PATTERN]`. |

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

14. **Memory/temperature observations:** COLMAP feature matching pushed the GPU to 86 °C (throttling
   starts about 87 °C). splatfacto-big used about 3-4.7 GB GPU early in training, compared with about
   2.2 GB peak for splatfacto.

## Remote viewing

| Thing | How |
|---|---|
| Remote Control | `claude remote-control` running in tmux session `claude` (survives closing the Ubuntu window). |
| Splat viewer | `localhost:7007` (training's built-in viewer, or `ns-viewer --load-config ... --viewer.websocket-port 7007`). Public via `cloudflared tunnel --url http://localhost:7007` (new random link each start). |
| Terminal dashboard | tmux session `dash` (top: `tail -F` of the current training log; bottom: GPU/RAM every 5 s). Served read-only by `ttyd -p 7681 -i 127.0.0.1 -c murph:<password> tmux attach -r -t dash`, password in `~/.ttyd_pass`, public through a second `cloudflared` tunnel to port 7681. |
| Stop everything public | `pkill -f "cloudflared tunnel"` and `pkill -f ttyd` |
